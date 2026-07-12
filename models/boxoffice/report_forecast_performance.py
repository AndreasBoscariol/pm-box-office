#!/usr/bin/env python3
"""Report persisted box-office forecast performance metrics."""

from __future__ import annotations

import argparse
import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pm_box_office.db.connection import connect_database

from .constants import FORECAST_TABLE, LIVE_REGIMES, PRE_RELEASE_REGIME


DEFAULT_OUTPUT_DIR = Path("data/reports/boxoffice_performance")
ACTIVE_MODEL_PATH = Path(__file__).with_name("ACTIVE_MODEL")
OPENING_WEEKEND_TARGET = "opening_weekend"
CANONICAL_KEY = ["model_version", "release_run_id", "origin_key", "target"]


def fetch_forecasts(conn: Any, *, model_version: str | None = None) -> pd.DataFrame:
    predicates = [
        "actual_usd IS NOT NULL",
        "point_usd IS NOT NULL",
        "actual_usd > 0",
        "point_usd > 0",
    ]
    params: list[Any] = []
    if model_version:
        predicates.append("model_version = %s")
        params.append(model_version)
    sql = f"""
        SELECT
            forecast_id,
            run_id,
            model_version,
            movie_id,
            release_run_id,
            title,
            opening_weekend_start,
            regime,
            origin_key,
            origin_day,
            forecast_origin_utc,
            as_of_utc,
            target,
            point_usd,
            lo80_usd,
            hi80_usd,
            lo95_usd,
            hi95_usd,
            point_model,
            interval_model,
            component_source,
            feature_quality_bucket,
            source_count,
            amc_coverage,
            amc_snapshot_count,
            amc_lateness_p50_minutes,
            actual_usd,
            is_live,
            is_backtest,
            created_at
        FROM {FORECAST_TABLE}
        WHERE {" AND ".join(predicates)}
        ORDER BY opening_weekend_start, release_run_id, target, forecast_origin_utc
    """
    cursor = conn.execute(sql, tuple(params))
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def active_model_version() -> str:
    return ACTIVE_MODEL_PATH.read_text(encoding="utf-8").strip()


def _float_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column), errors="coerce")


def _safe_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if len(values) else float("nan")


def _safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def _safe_quantile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(q)) if len(values) else float("nan")


def interval_score(actual: pd.Series, lower: pd.Series, upper: pd.Series, alpha: float) -> pd.Series:
    width = upper - lower
    lower_penalty = (2.0 / alpha) * (lower - actual).clip(lower=0)
    upper_penalty = (2.0 / alpha) * (actual - upper).clip(lower=0)
    return width + lower_penalty + upper_penalty


def pre_release_bucket(origin_day: Any) -> str:
    if pd.isna(origin_day):
        return "pre_release_unknown"
    day = int(origin_day)
    if -14 <= day <= -8:
        return "D-14..D-8"
    if -7 <= day <= -3:
        return "D-7..D-3"
    if day == -2:
        return "D-2"
    if day == -1:
        return "D-1"
    return f"D{day:+d}"


def origin_group(row: pd.Series) -> str:
    regime = str(row["regime"])
    if regime == PRE_RELEASE_REGIME:
        return pre_release_bucket(row.get("origin_day"))
    return str(row.get("origin_key") or "unknown")


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    for column in [
        "actual_usd",
        "point_usd",
        "lo80_usd",
        "hi80_usd",
        "lo95_usd",
        "hi95_usd",
        "source_count",
        "amc_coverage",
        "amc_snapshot_count",
        "amc_lateness_p50_minutes",
    ]:
        out[column] = _float_series(out, column)
    out["opening_weekend_start"] = pd.to_datetime(out["opening_weekend_start"], errors="coerce").dt.date
    out["forecast_origin_utc"] = pd.to_datetime(out["forecast_origin_utc"], errors="coerce", utc=True)
    out["pct_error_signed"] = out["actual_usd"] / out["point_usd"] - 1.0
    out["pct_error_abs"] = out["pct_error_signed"].abs()
    out["log_error"] = np.log(out["actual_usd"] / out["point_usd"])
    out["pinball_50_usd"] = 0.5 * (out["actual_usd"] - out["point_usd"]).abs()
    out["inside_80"] = (out["actual_usd"] >= out["lo80_usd"]) & (out["actual_usd"] <= out["hi80_usd"])
    out["below_80"] = out["actual_usd"] < out["lo80_usd"]
    out["above_80"] = out["actual_usd"] > out["hi80_usd"]
    out["inside_95"] = (out["actual_usd"] >= out["lo95_usd"]) & (out["actual_usd"] <= out["hi95_usd"])
    out["below_95"] = out["actual_usd"] < out["lo95_usd"]
    out["above_95"] = out["actual_usd"] > out["hi95_usd"]
    out["width_80"] = out["hi80_usd"] - out["lo80_usd"]
    out["width_95"] = out["hi95_usd"] - out["lo95_usd"]
    out["width_80_over_point"] = out["width_80"] / out["point_usd"]
    out["width_95_over_point"] = out["width_95"] / out["point_usd"]
    out["interval_score_80"] = interval_score(out["actual_usd"], out["lo80_usd"], out["hi80_usd"], 0.20)
    out["interval_score_95"] = interval_score(out["actual_usd"], out["lo95_usd"], out["hi95_usd"], 0.05)
    out["interval_score_80_over_point"] = out["interval_score_80"] / out["point_usd"]
    out["interval_score_95_over_point"] = out["interval_score_95"] / out["point_usd"]
    out["origin_group"] = out.apply(origin_group, axis=1)
    out["evaluation_status"] = np.where(
        out["regime"].isin(LIVE_REGIMES) & (out["amc_snapshot_count"].fillna(0) <= 0),
        "historical_persisted_panel_no_amc_baseline",
        "historical_persisted_panel",
    )
    return out


def canonicalize(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if frame.empty:
        return frame.copy(), 0
    sort_columns = [column for column in ["created_at", "as_of_utc", "forecast_origin_utc"] if column in frame.columns]
    out = frame.copy()
    for column in sort_columns:
        out[column] = pd.to_datetime(out[column], errors="coerce", utc=True)
    if sort_columns:
        out = out.sort_values(sort_columns)
    before = len(out)
    out = out.drop_duplicates(CANONICAL_KEY, keep="last")
    return out, before - len(out)


def sample_integrity(raw: pd.DataFrame, frame: pd.DataFrame, *, rows_removed: int) -> pd.DataFrame:
    primary = frame[frame["target"] == OPENING_WEEKEND_TARGET]
    duplicate_count = int(raw.duplicated(CANONICAL_KEY).sum()) if not raw.empty else 0
    rows = [
        ("input_rows", len(raw)),
        ("distinct_model_versions", raw["model_version"].nunique() if "model_version" in raw else 0),
        ("distinct_targets", raw["target"].nunique() if "target" in raw else 0),
        ("distinct_canonical_forecast_keys", raw[CANONICAL_KEY].drop_duplicates().shape[0] if not raw.empty else 0),
        ("duplicate_key_count", duplicate_count),
        ("rows_removed_by_canonicalization", rows_removed),
        ("opening_weekend_rows", len(primary)),
        ("opening_weekend_unique_movies", primary["movie_id"].nunique() if not primary.empty else 0),
        (
            "opening_weekend_unique_movie_days",
            primary[["release_run_id", "origin_key"]].drop_duplicates().shape[0] if not primary.empty else 0,
        ),
        ("opening_weekend_release_weekends", primary["opening_weekend_start"].nunique() if not primary.empty else 0),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def _unit_balanced(frame: pd.DataFrame, unit_columns: list[str]) -> pd.DataFrame:
    metric_columns = [
        "pct_error_signed",
        "pct_error_abs",
        "log_error",
        "pinball_50_usd",
        "inside_80",
        "below_80",
        "above_80",
        "inside_95",
        "below_95",
        "above_95",
        "width_80",
        "width_80_over_point",
        "interval_score_80_over_point",
        "width_95",
        "width_95_over_point",
        "interval_score_95_over_point",
        "source_count",
        "amc_coverage",
        "amc_snapshot_count",
        "amc_lateness_p50_minutes",
    ]
    available = [column for column in metric_columns if column in frame.columns]
    averaged = frame.groupby(unit_columns, dropna=False)[available].mean(numeric_only=True).reset_index()
    for column in ["movie_id", "release_run_id", "opening_weekend_start", "forecast_origin_utc"]:
        if column not in averaged and column in frame.columns and column in unit_columns:
            averaged[column] = averaged[column]
    return averaged


def _metric_source(frame: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    if aggregation == "movie_balanced":
        return _unit_balanced(frame, ["release_run_id"])
    if aggregation == "release_weekend_balanced":
        return _unit_balanced(frame, ["opening_weekend_start"])
    return frame


def _compute_metric(frame: pd.DataFrame, metric: str, aggregation: str) -> float:
    metrics = _metric_source(frame, aggregation)
    return _compute_metric_from_metrics(metrics, metric)


def _compute_metric_from_metrics(metrics: pd.DataFrame, metric: str) -> float:
    if metric == "mdape":
        return _safe_median(metrics["pct_error_abs"])
    if metric == "mae_log":
        return _safe_mean(metrics["log_error"].abs())
    if metric == "coverage_80":
        return _safe_mean(metrics["inside_80"])
    if metric == "interval_score_80":
        return _safe_mean(metrics["interval_score_80_over_point"])
    if metric == "coverage_95":
        return _safe_mean(metrics["inside_95"])
    if metric == "interval_score_95":
        return _safe_mean(metrics["interval_score_95_over_point"])
    raise ValueError(f"unknown bootstrap metric: {metric}")


def release_weekend_bootstrap_ci(
    frame: pd.DataFrame,
    *,
    aggregation: str,
    samples: int,
    seed: int,
) -> dict[str, float]:
    metrics = ["mdape", "mae_log", "coverage_80", "interval_score_80", "coverage_95", "interval_score_95"]
    if samples <= 0 or frame.empty:
        return {f"{metric}_{suffix}": float("nan") for metric in metrics for suffix in ["ci_low", "ci_high"]}
    if aggregation == "movie_balanced":
        boot_frame = _unit_balanced(frame, ["opening_weekend_start", "release_run_id"])
    elif aggregation == "release_weekend_balanced":
        boot_frame = _unit_balanced(frame, ["opening_weekend_start"])
    else:
        boot_frame = frame
    weekends = pd.Series(boot_frame["opening_weekend_start"].dropna().unique())
    if weekends.empty:
        return {f"{metric}_{suffix}": float("nan") for metric in metrics for suffix in ["ci_low", "ci_high"]}
    by_weekend = {weekend: group for weekend, group in boot_frame.groupby("opening_weekend_start", dropna=False)}
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in metrics}
    for _ in range(samples):
        sampled = rng.choice(weekends.to_numpy(), size=len(weekends), replace=True)
        sample_frame = pd.concat([by_weekend[weekend] for weekend in sampled], ignore_index=True)
        for metric in metrics:
            values[metric].append(_compute_metric_from_metrics(sample_frame, metric))
    out: dict[str, float] = {}
    for metric, metric_values in values.items():
        clean = pd.Series(metric_values, dtype="float64").dropna()
        out[f"{metric}_ci_low"] = float(clean.quantile(0.025)) if len(clean) else float("nan")
        out[f"{metric}_ci_high"] = float(clean.quantile(0.975)) if len(clean) else float("nan")
    return out


def summarize_group(
    frame: pd.DataFrame,
    *,
    regime: str,
    origin_group_name: str,
    target: str,
    aggregation: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    metrics = _metric_source(frame, aggregation)
    status_values = sorted(str(value) for value in frame["evaluation_status"].dropna().unique())
    evaluation_status = status_values[0] if len(status_values) == 1 else "mixed:" + ",".join(status_values)
    row = {
        "regime": regime,
        "origin_group": origin_group_name,
        "target": target,
        "aggregation": aggregation,
        "evaluation_status": evaluation_status,
        "forecast_rows": int(len(frame)),
        "unique_movies": int(frame["movie_id"].nunique()),
        "unique_movie_days": int(frame[["release_run_id", "origin_key"]].drop_duplicates().shape[0]),
        "release_weekends": int(frame["opening_weekend_start"].nunique()),
        "date_min": str(frame["opening_weekend_start"].min()) if len(frame) else "",
        "date_max": str(frame["opening_weekend_start"].max()) if len(frame) else "",
        "median_signed_pct_error": _safe_median(metrics["pct_error_signed"]),
        "mdape": _safe_median(metrics["pct_error_abs"]),
        "mae_log": _safe_mean(metrics["log_error"].abs()),
        "rmse_log": math.sqrt(_safe_mean(metrics["log_error"] ** 2)),
        "median_pinball_50_over_point": _safe_median(metrics["pinball_50_usd"] / frame["point_usd"]) if aggregation == "row_weighted" else float("nan"),
        "coverage_80": _safe_mean(metrics["inside_80"]),
        "lower_miss_80": _safe_mean(metrics["below_80"]),
        "upper_miss_80": _safe_mean(metrics["above_80"]),
        "median_width_80": _safe_median(metrics["width_80"]),
        "median_width_80_over_point": _safe_median(metrics["width_80_over_point"]),
        "mean_interval_score_80_over_point": _safe_mean(metrics["interval_score_80_over_point"]),
        "coverage_95": _safe_mean(metrics["inside_95"]),
        "lower_miss_95": _safe_mean(metrics["below_95"]),
        "upper_miss_95": _safe_mean(metrics["above_95"]),
        "median_width_95": _safe_median(metrics["width_95"]),
        "median_width_95_over_point": _safe_median(metrics["width_95_over_point"]),
        "mean_interval_score_95_over_point": _safe_mean(metrics["interval_score_95_over_point"]),
        "median_source_count": _safe_median(metrics["source_count"]),
        "single_source_rate": _safe_mean(frame["source_count"] == 1),
        "amc_available_rate": _safe_mean(metrics["amc_snapshot_count"] > 0),
        "median_amc_coverage": _safe_median(metrics["amc_coverage"]),
        "median_amc_lateness_minutes": _safe_median(metrics["amc_lateness_p50_minutes"]),
    }
    if aggregation == "movie_balanced" and target == OPENING_WEEKEND_TARGET:
        row.update(
            release_weekend_bootstrap_ci(
                frame,
                aggregation=aggregation,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
        )
    return row


def summarize(frame: pd.DataFrame, *, bootstrap_samples: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (regime, group_name, target), group in frame.groupby(["regime", "origin_group", "target"], dropna=False):
        for aggregation in ["row_weighted", "movie_balanced", "release_weekend_balanced"]:
            seed_payload = f"{regime}|{group_name}|{target}|{aggregation}".encode("utf-8")
            seed = int(hashlib.sha1(seed_payload).hexdigest()[:8], 16)
            rows.append(
                summarize_group(
                    group,
                    regime=str(regime),
                    origin_group_name=str(group_name),
                    target=str(target),
                    aggregation=aggregation,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=seed,
                )
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    regime_order = {PRE_RELEASE_REGIME: 0, **{regime: index + 1 for index, regime in enumerate(LIVE_REGIMES)}}
    target_order = {OPENING_WEEKEND_TARGET: 0, "friday": 1, "saturday": 2, "sunday": 3}
    origin_order = {
        "D-14..D-8": 0,
        "D-7..D-3": 1,
        "D-2": 2,
        "D-1": 3,
        "FRI_10:00": 10,
        "FRI_12:00": 12,
        "FRI_14:00": 14,
        "FRI_16:00": 16,
        "FRI_18:00": 18,
        "FRI_20:00": 20,
        "FRI_EOD": 24,
        "SAT_10:00": 10,
        "SAT_12:00": 12,
        "SAT_14:00": 14,
        "SAT_16:00": 16,
        "SAT_18:00": 18,
        "SAT_20:00": 20,
        "SAT_EOD": 24,
        "SUN_10:00": 10,
        "SUN_12:00": 12,
        "SUN_14:00": 14,
        "SUN_16:00": 16,
        "SUN_18:00": 18,
        "SUN_20:00": 20,
        "SUN_EOD": 24,
    }
    out["_regime_order"] = out["regime"].map(regime_order).fillna(99)
    out["_target_order"] = out["target"].map(target_order).fillna(99)
    out["_origin_order"] = out["origin_group"].map(origin_order).fillna(99)
    aggregation_order = {"row_weighted": 0, "movie_balanced": 1, "release_weekend_balanced": 2}
    out["_aggregation_order"] = out["aggregation"].map(aggregation_order).fillna(99)
    return out.sort_values(["_regime_order", "_target_order", "_origin_order", "_aggregation_order"]).drop(
        columns=["_regime_order", "_target_order", "_origin_order", "_aggregation_order"]
    )


def revision_summary(frame: pd.DataFrame) -> pd.DataFrame:
    regimes = sorted(frame["regime"].dropna().unique()) if not frame.empty else []
    return pd.DataFrame(
        [
            {
                "regime": str(regime),
                "status": "not_evaluable_from_latest-only_table",
                "revision_pairs": 0,
            }
            for regime in regimes
        ]
    )


def _with_candidate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    return enrich(frame)


def _latest_pre_release_ow(frame: pd.DataFrame) -> pd.DataFrame:
    pre = frame[
        (frame["regime"] == PRE_RELEASE_REGIME)
        & (frame["target"] == OPENING_WEEKEND_TARGET)
        & (pd.to_numeric(frame["origin_day"], errors="coerce").isin([-2, -1]))
    ].copy()
    if pre.empty:
        return pre
    pre["_origin_rank"] = pd.to_numeric(pre["origin_day"], errors="coerce")
    pre = pre.sort_values(["release_run_id", "_origin_rank"])
    return pre.drop_duplicates(["release_run_id"], keep="last").drop(columns=["_origin_rank"])


def _interval_log_distances(
    frame: pd.DataFrame,
    *,
    point_col: str,
    lo80_col: str,
    hi80_col: str,
    lo95_col: str,
    hi95_col: str,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    point = pd.to_numeric(frame[point_col], errors="coerce")
    lo80 = pd.to_numeric(frame[lo80_col], errors="coerce")
    hi80 = pd.to_numeric(frame[hi80_col], errors="coerce")
    lo95 = pd.to_numeric(frame[lo95_col], errors="coerce")
    hi95 = pd.to_numeric(frame[hi95_col], errors="coerce")
    return (
        np.log(point / lo80),
        np.log(hi80 / point),
        np.log(point / lo95),
        np.log(hi95 / point),
    )


def friday_no_amc_challenger_rows(frame: pd.DataFrame, *, origin_key: str = "FRI_20:00") -> pd.DataFrame:
    friday = frame[
        (frame["regime"] == "live_friday")
        & (frame["origin_key"] == origin_key)
        & (frame["target"] == OPENING_WEEKEND_TARGET)
        & (frame["amc_snapshot_count"].fillna(0) <= 0)
    ].copy()
    pre = _latest_pre_release_ow(frame)
    if friday.empty or pre.empty:
        return pd.DataFrame()

    joined = friday.merge(
        pre,
        on="release_run_id",
        suffixes=("_fri", "_pre"),
        how="inner",
    )
    if joined.empty:
        return pd.DataFrame()

    common = {
        "model_version": joined["model_version_fri"],
        "movie_id": joined["movie_id_fri"],
        "release_run_id": joined["release_run_id"],
        "title": joined["title_fri"],
        "opening_weekend_start": joined["opening_weekend_start_fri"],
        "regime": "friday_no_amc_challenger",
        "origin_key": origin_key,
        "origin_day": np.nan,
        "forecast_origin_utc": joined["forecast_origin_utc_fri"],
        "as_of_utc": joined["as_of_utc_fri"],
        "target": OPENING_WEEKEND_TARGET,
        "actual_usd": joined["actual_usd_fri"],
        "amc_snapshot_count": 0,
        "amc_coverage": np.nan,
        "source_count": joined.get("source_count_fri", np.nan),
        "amc_lateness_p50_minutes": np.nan,
        "is_live": True,
        "is_backtest": joined.get("is_backtest_fri", False),
        "created_at": joined["created_at_fri"],
    }

    current = pd.DataFrame(
        {
            **common,
            "forecast_id": joined["forecast_id_fri"],
            "run_id": joined["run_id_fri"],
            "point_usd": joined["point_usd_fri"],
            "lo80_usd": joined["lo80_usd_fri"],
            "hi80_usd": joined["hi80_usd_fri"],
            "lo95_usd": joined["lo95_usd_fri"],
            "hi95_usd": joined["hi95_usd_fri"],
            "point_model": "current_friday_no_amc_composition",
            "interval_model": joined["interval_model_fri"],
            "component_source": "current_friday_no_amc_composition",
            "candidate": "A_current_friday_no_amc",
            "pre_release_origin_key": joined["origin_key_pre"],
        }
    )
    carry_forward = pd.DataFrame(
        {
            **common,
            "forecast_id": joined["forecast_id_pre"],
            "run_id": joined["run_id_pre"],
            "point_usd": joined["point_usd_pre"],
            "lo80_usd": joined["lo80_usd_pre"],
            "hi80_usd": joined["hi80_usd_pre"],
            "lo95_usd": joined["lo95_usd_pre"],
            "hi95_usd": joined["hi95_usd_pre"],
            "point_model": "latest_pre_release_carry_forward",
            "interval_model": joined["interval_model_pre"],
            "component_source": "latest_pre_release_carry_forward",
            "candidate": "B_latest_pre_release_carry_forward",
            "pre_release_origin_key": joined["origin_key_pre"],
        }
    )

    fri_lo80, fri_hi80, fri_lo95, fri_hi95 = _interval_log_distances(
        joined,
        point_col="point_usd_fri",
        lo80_col="lo80_usd_fri",
        hi80_col="hi80_usd_fri",
        lo95_col="lo95_usd_fri",
        hi95_col="hi95_usd_fri",
    )
    pre_lo80, pre_hi80, pre_lo95, pre_hi95 = _interval_log_distances(
        joined,
        point_col="point_usd_pre",
        lo80_col="lo80_usd_pre",
        hi80_col="hi80_usd_pre",
        lo95_col="lo95_usd_pre",
        hi95_col="hi95_usd_pre",
    )
    point = pd.to_numeric(joined["point_usd_fri"], errors="coerce")
    protected = pd.DataFrame(
        {
            **common,
            "forecast_id": joined["forecast_id_fri"],
            "run_id": joined["run_id_fri"],
            "point_usd": point,
            "lo80_usd": point / np.exp(np.maximum(fri_lo80, pre_lo80)),
            "hi80_usd": point * np.exp(np.maximum(fri_hi80, pre_hi80)),
            "lo95_usd": point / np.exp(np.maximum(fri_lo95, pre_lo95)),
            "hi95_usd": point * np.exp(np.maximum(fri_hi95, pre_hi95)),
            "point_model": "current_friday_point_pre_release_uncertainty_floor",
            "interval_model": "max_log_distance_current_vs_latest_pre_release",
            "component_source": "current_friday_point_pre_release_uncertainty_floor",
            "candidate": "C_current_point_protected_uncertainty",
            "pre_release_origin_key": joined["origin_key_pre"],
        }
    )
    rows = pd.concat([current, carry_forward, protected], ignore_index=True, sort=False)
    rows["evaluation_status"] = "historical_persisted_panel_no_amc_fallback_diagnostic"
    return _with_candidate_metrics(rows)


def summarize_candidate_group(frame: pd.DataFrame, *, candidate: str, bootstrap_samples: int) -> dict[str, Any]:
    row = summarize_group(
        frame,
        regime="friday_no_amc_challenger",
        origin_group_name=str(frame["origin_key"].iloc[0]) if len(frame) else "",
        target=OPENING_WEEKEND_TARGET,
        aggregation="movie_balanced",
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=int(hashlib.sha1(str(candidate).encode("utf-8")).hexdigest()[:8], 16),
    )
    row["candidate"] = candidate
    return row


def friday_no_amc_challenger_summary(
    frame: pd.DataFrame,
    *,
    bootstrap_samples: int,
    origin_key: str = "FRI_20:00",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = friday_no_amc_challenger_rows(frame, origin_key=origin_key)
    if rows.empty:
        return rows, pd.DataFrame(), pd.DataFrame()
    summary = pd.DataFrame(
        [
            summarize_candidate_group(group, candidate=str(candidate), bootstrap_samples=bootstrap_samples)
            for candidate, group in rows.groupby("candidate", sort=True)
        ]
    )
    diffs = paired_candidate_differences(rows, bootstrap_samples=bootstrap_samples)
    return rows, summary, diffs


def _candidate_metric_table(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = []
    for candidate, group in rows.groupby("candidate", sort=True):
        metrics.append(
            {
                "candidate": candidate,
                "mdape": _safe_median(group["pct_error_abs"]),
                "mae_log": _safe_mean(group["log_error"].abs()),
                "coverage_80": _safe_mean(group["inside_80"]),
                "coverage_95": _safe_mean(group["inside_95"]),
                "interval_score_80": _safe_mean(group["interval_score_80_over_point"]),
                "interval_score_95": _safe_mean(group["interval_score_95_over_point"]),
            }
        )
    return pd.DataFrame(metrics).set_index("candidate")


def paired_candidate_differences(rows: pd.DataFrame, *, bootstrap_samples: int) -> pd.DataFrame:
    current = "A_current_friday_no_amc"
    table = _candidate_metric_table(rows)
    if current not in table.index:
        return pd.DataFrame()
    weekends = pd.Series(rows["opening_weekend_start"].dropna().unique())
    by_weekend = {weekend: group for weekend, group in rows.groupby("opening_weekend_start", dropna=False)}
    rng = np.random.default_rng(4927)
    records = []
    for candidate in table.index:
        if candidate == current:
            continue
        record = {"candidate": candidate, "baseline": current}
        for metric in table.columns:
            observed = float(table.loc[candidate, metric] - table.loc[current, metric])
            record[f"{metric}_diff"] = observed
        values = {metric: [] for metric in table.columns}
        if bootstrap_samples > 0 and not weekends.empty:
            for _ in range(bootstrap_samples):
                sampled = rng.choice(weekends.to_numpy(), size=len(weekends), replace=True)
                sample_rows = pd.concat([by_weekend[weekend] for weekend in sampled], ignore_index=True)
                sample_table = _candidate_metric_table(sample_rows)
                if current not in sample_table.index or candidate not in sample_table.index:
                    continue
                for metric in table.columns:
                    values[metric].append(float(sample_table.loc[candidate, metric] - sample_table.loc[current, metric]))
        for metric, metric_values in values.items():
            clean = pd.Series(metric_values, dtype="float64").dropna()
            record[f"{metric}_diff_ci_low"] = float(clean.quantile(0.025)) if len(clean) else float("nan")
            record[f"{metric}_diff_ci_high"] = float(clean.quantile(0.975)) if len(clean) else float("nan")
        records.append(record)
    return pd.DataFrame(records)


def _format_pct(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) * 100:.1f}%"


def _format_float(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.3f}"


def _format_usd(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"${float(value) / 1_000_000:.1f}M"


def markdown_table(frame: pd.DataFrame, columns: list[str], formatters: dict[str, Any] | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    subset = frame.loc[:, columns].copy()
    for column, formatter in (formatters or {}).items():
        if column in subset:
            subset[column] = subset[column].map(formatter)
    rendered = subset.fillna("").astype(str)
    header = "| " + " | ".join(rendered.columns) + " |"
    separator = "| " + " | ".join("---" for _ in rendered.columns) + " |"
    body = [
        "| " + " | ".join(value.replace("\n", " ") for value in row) + " |"
        for row in rendered.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *body])


def write_report(
    *,
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    integrity: pd.DataFrame,
    revisions: pd.DataFrame,
    friday_challenger_rows: pd.DataFrame,
    friday_challenger_summary: pd.DataFrame,
    friday_challenger_diffs: pd.DataFrame,
    output_dir: Path,
    model_version: str | None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    summary.to_csv(output_dir / "forecast_performance_summary.csv", index=False)
    integrity.to_csv(output_dir / "forecast_sample_integrity.csv", index=False)
    revisions.to_csv(output_dir / "forecast_revision_summary.csv", index=False)
    frame.to_csv(output_dir / "forecast_performance_rows.csv", index=False)
    friday_challenger_rows.to_csv(output_dir / "friday_no_amc_challenger_rows.csv", index=False)
    friday_challenger_summary.to_csv(output_dir / "friday_no_amc_challenger_summary.csv", index=False)
    friday_challenger_diffs.to_csv(output_dir / "friday_no_amc_challenger_differences.csv", index=False)

    ow_summary = summary[
        (summary["target"] == OPENING_WEEKEND_TARGET) & (summary["aggregation"] == "movie_balanced")
    ].copy()
    daily_summary = summary[
        (summary["target"] != OPENING_WEEKEND_TARGET) & (summary["aggregation"] == "movie_balanced")
    ].copy()
    row_weighted_ow = summary[
        (summary["target"] == OPENING_WEEKEND_TARGET) & (summary["aggregation"] == "row_weighted")
    ].copy()
    report_path = output_dir / "forecast_performance_report.md"
    report_path.write_text(
        "\n\n".join(
            [
                "# Box Office Forecast Performance",
                f"Generated: `{generated_at}`",
                f"Model version filter: `{model_version or 'all persisted model versions'}`",
                (
                    "Scope note: this is a diagnostic baseline report over currently persisted forecast rows "
                    "with non-null actuals. It is not a prospective emitted-forecast scorecard. Live rows with "
                    "zero AMC snapshots are labeled as no-AMC persisted baseline rows."
                ),
                "## Sample Integrity",
                markdown_table(integrity, ["metric", "value"]),
                "## Opening Weekend Metrics - Movie Balanced",
                markdown_table(
                    ow_summary,
                    [
                        "regime",
                        "origin_group",
                        "evaluation_status",
                        "forecast_rows",
                        "unique_movies",
                        "release_weekends",
                        "amc_available_rate",
                        "median_signed_pct_error",
                        "mdape",
                        "mdape_ci_low",
                        "mdape_ci_high",
                        "mae_log",
                        "mae_log_ci_low",
                        "mae_log_ci_high",
                        "coverage_80",
                        "coverage_80_ci_low",
                        "coverage_80_ci_high",
                        "lower_miss_80",
                        "upper_miss_80",
                        "median_width_80_over_point",
                        "mean_interval_score_80_over_point",
                        "interval_score_80_ci_low",
                        "interval_score_80_ci_high",
                        "coverage_95",
                        "coverage_95_ci_low",
                        "coverage_95_ci_high",
                        "lower_miss_95",
                        "upper_miss_95",
                        "median_width_95_over_point",
                        "mean_interval_score_95_over_point",
                    ],
                    {
                        "amc_available_rate": _format_pct,
                        "median_signed_pct_error": _format_pct,
                        "mdape": _format_pct,
                        "mdape_ci_low": _format_pct,
                        "mdape_ci_high": _format_pct,
                        "coverage_80": _format_pct,
                        "coverage_80_ci_low": _format_pct,
                        "coverage_80_ci_high": _format_pct,
                        "lower_miss_80": _format_pct,
                        "upper_miss_80": _format_pct,
                        "median_width_80_over_point": _format_pct,
                        "mean_interval_score_80_over_point": _format_float,
                        "interval_score_80_ci_low": _format_float,
                        "interval_score_80_ci_high": _format_float,
                        "coverage_95": _format_pct,
                        "coverage_95_ci_low": _format_pct,
                        "coverage_95_ci_high": _format_pct,
                        "lower_miss_95": _format_pct,
                        "upper_miss_95": _format_pct,
                        "median_width_95_over_point": _format_pct,
                        "mean_interval_score_95_over_point": _format_float,
                        "mae_log": _format_float,
                        "mae_log_ci_low": _format_float,
                        "mae_log_ci_high": _format_float,
                    },
                ),
                "## Opening Weekend Metrics - Row Weighted",
                markdown_table(
                    row_weighted_ow,
                    [
                        "regime",
                        "origin_group",
                        "forecast_rows",
                        "unique_movies",
                        "release_weekends",
                        "median_signed_pct_error",
                        "mdape",
                        "mae_log",
                        "coverage_80",
                        "lower_miss_80",
                        "upper_miss_80",
                        "median_width_80_over_point",
                        "mean_interval_score_80_over_point",
                    ],
                    {
                        "median_signed_pct_error": _format_pct,
                        "mdape": _format_pct,
                        "coverage_80": _format_pct,
                        "lower_miss_80": _format_pct,
                        "upper_miss_80": _format_pct,
                        "median_width_80_over_point": _format_pct,
                        "mean_interval_score_80_over_point": _format_float,
                        "mae_log": _format_float,
                    },
                ),
                "## Daily Target Metrics - Movie Balanced",
                markdown_table(
                    daily_summary,
                    [
                        "regime",
                        "origin_group",
                        "target",
                        "evaluation_status",
                        "forecast_rows",
                        "unique_movies",
                        "median_signed_pct_error",
                        "mdape",
                        "mae_log",
                        "coverage_80",
                        "coverage_95",
                        "amc_available_rate",
                        "median_amc_coverage",
                    ],
                    {
                        "median_signed_pct_error": _format_pct,
                        "mdape": _format_pct,
                        "coverage_80": _format_pct,
                        "coverage_95": _format_pct,
                        "amc_available_rate": _format_pct,
                        "median_amc_coverage": _format_pct,
                        "mae_log": _format_float,
                    },
                ),
                "## Forecast Revisions",
                markdown_table(
                    revisions,
                    [
                        "regime",
                        "status",
                        "revision_pairs",
                    ],
                ),
                "## Friday No-AMC Fallback Diagnostic",
                markdown_table(
                    friday_challenger_summary,
                    [
                        "candidate",
                        "forecast_rows",
                        "unique_movies",
                        "release_weekends",
                        "median_signed_pct_error",
                        "mdape",
                        "mdape_ci_low",
                        "mdape_ci_high",
                        "mae_log",
                        "coverage_80",
                        "coverage_80_ci_low",
                        "coverage_80_ci_high",
                        "lower_miss_80",
                        "upper_miss_80",
                        "median_width_80_over_point",
                        "mean_interval_score_80_over_point",
                        "coverage_95",
                        "lower_miss_95",
                        "upper_miss_95",
                        "median_width_95_over_point",
                        "mean_interval_score_95_over_point",
                    ],
                    {
                        "median_signed_pct_error": _format_pct,
                        "mdape": _format_pct,
                        "mdape_ci_low": _format_pct,
                        "mdape_ci_high": _format_pct,
                        "coverage_80": _format_pct,
                        "coverage_80_ci_low": _format_pct,
                        "coverage_80_ci_high": _format_pct,
                        "lower_miss_80": _format_pct,
                        "upper_miss_80": _format_pct,
                        "median_width_80_over_point": _format_pct,
                        "coverage_95": _format_pct,
                        "lower_miss_95": _format_pct,
                        "upper_miss_95": _format_pct,
                        "median_width_95_over_point": _format_pct,
                        "mae_log": _format_float,
                        "mean_interval_score_80_over_point": _format_float,
                        "mean_interval_score_95_over_point": _format_float,
                    },
                ),
                "## Friday No-AMC Paired Differences vs Current",
                markdown_table(
                    friday_challenger_diffs,
                    [
                        "candidate",
                        "baseline",
                        "mdape_diff",
                        "mdape_diff_ci_low",
                        "mdape_diff_ci_high",
                        "mae_log_diff",
                        "mae_log_diff_ci_low",
                        "mae_log_diff_ci_high",
                        "coverage_80_diff",
                        "coverage_80_diff_ci_low",
                        "coverage_80_diff_ci_high",
                        "interval_score_80_diff",
                        "interval_score_80_diff_ci_low",
                        "interval_score_80_diff_ci_high",
                    ],
                    {
                        "mdape_diff": _format_pct,
                        "mdape_diff_ci_low": _format_pct,
                        "mdape_diff_ci_high": _format_pct,
                        "coverage_80_diff": _format_pct,
                        "coverage_80_diff_ci_low": _format_pct,
                        "coverage_80_diff_ci_high": _format_pct,
                        "mae_log_diff": _format_float,
                        "mae_log_diff_ci_low": _format_float,
                        "mae_log_diff_ci_high": _format_float,
                        "interval_score_80_diff": _format_float,
                        "interval_score_80_diff_ci_low": _format_float,
                        "interval_score_80_diff_ci_high": _format_float,
                    },
                ),
                "## Files",
                "- `forecast_performance_summary.csv`",
                "- `forecast_sample_integrity.csv`",
                "- `forecast_revision_summary.csv`",
                "- `forecast_performance_rows.csv`",
                "- `friday_no_amc_challenger_rows.csv`",
                "- `friday_no_amc_challenger_summary.csv`",
                "- `friday_no_amc_challenger_differences.csv`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return report_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url")
    parser.add_argument(
        "--model-version",
        default=active_model_version(),
        help="Model version filter. Defaults to models/boxoffice/ACTIVE_MODEL.",
    )
    parser.add_argument(
        "--all-model-versions",
        action="store_true",
        help="Pool all persisted model versions. Intended only for audits.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    model_version = None if args.all_model_versions else args.model_version
    conn = connect_database(args.database_url)
    try:
        raw = fetch_forecasts(conn, model_version=model_version)
    finally:
        conn.close()
    canonical, rows_removed = canonicalize(raw)
    frame = enrich(canonical)
    integrity = sample_integrity(raw, frame, rows_removed=rows_removed)
    summary = summarize(frame, bootstrap_samples=args.bootstrap_samples)
    revisions = revision_summary(frame)
    friday_rows, friday_summary, friday_diffs = friday_no_amc_challenger_summary(
        frame,
        bootstrap_samples=args.bootstrap_samples,
    )
    report_path = write_report(
        frame=frame,
        summary=summary,
        integrity=integrity,
        revisions=revisions,
        friday_challenger_rows=friday_rows,
        friday_challenger_summary=friday_summary,
        friday_challenger_diffs=friday_diffs,
        output_dir=args.output_dir,
        model_version=model_version,
    )
    print(f"Wrote {report_path}")
    print(f"Evaluated {len(frame):,} forecast rows with actuals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
