#!/usr/bin/env python3
"""Freeze production box-office forecast artifacts from diagnostic outputs."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pm_box_office.db.connection import connect_database

from .artifacts import DEFAULT_ARTIFACT_ROOT, utc_now_iso, write_json
from .constants import ACTUAL_COLUMNS, BASELINE_COLUMNS
from .live_amc_plugin import build_amc_interval_policy, empty_amc_interval_policy

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTICS_DIR = REPO_ROOT / "data" / "diagnostics"
PREDICTIONS_DIR = REPO_ROOT / "data" / "predictions"


def default_model_version() -> str:
    return datetime.now(timezone.utc).strftime("boxoffice_%Y_%m_%d_001")


def copy_if_exists(source: Path, destination: Path) -> str | None:
    if not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination.name


def build_locked_point_policy(point_policy_csv: Path | None) -> dict[str, object]:
    policy = {
        "default": "dollar_median_consensus_usd",
        "by_origin_day": {"-1": "recency_weighted_log_consensus_lambda_1_usd"},
        "excluded_sources": ["the_numbers_predictions", "boxofficeguru"],
        "promotion_rule": "training scripts decide policies; production scripts apply policies",
    }
    if point_policy_csv and point_policy_csv.exists():
        frame = pd.read_csv(point_policy_csv)
        method_col = "selected_forecast_method" if "selected_forecast_method" in frame.columns else "forecast_method"
        if {"origin_day", method_col}.issubset(frame.columns):
            policy["by_origin_day"] = {
                str(int(row.origin_day)): str(getattr(row, method_col))
                for row in frame.itertuples(index=False)
                if pd.notna(row.origin_day) and pd.notna(getattr(row, method_col))
            }
    return policy


def _finite_positive_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce")
    return values[np.isfinite(values) & values.gt(0)]


def _interval_origin_bucket(origin_day: object) -> str:
    value = pd.to_numeric(pd.Series([origin_day]), errors="coerce").iloc[0]
    if not np.isfinite(value):
        return "unknown"
    day = int(value)
    if day <= -8:
        return "P_-14_to_-8"
    if day <= -3:
        return "P_-7_to_-3"
    if day == -2:
        return "P_-2"
    if day == -1:
        return "P_-1"
    return "P_0"


def _interval_source_count_bucket(value: object) -> str:
    count = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "one_source" if np.isfinite(count) and count <= 1 else "multi_source"


def _interval_point_bucket(value: object) -> str:
    point = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(point) or point <= 0:
        return "unknown"
    if point < 1_000_000:
        return "lt_1m"
    if point < 5_000_000:
        return "1m_5m"
    if point < 15_000_000:
        return "5m_15m"
    if point < 50_000_000:
        return "15m_50m"
    return "50m_plus"


def _interval_release_scope_bucket(row: pd.Series) -> str:
    release_type = str(row.get("release_type", "") or "").lower()
    width_bucket = str(row.get("release_width_bucket", "") or "").lower()
    if "platform" in release_type or width_bucket in {"limited", "platform"}:
        return "limited_or_platform"
    if width_bucket in {"wide", "large_wide"}:
        return "wide_or_large_wide"
    return "unknown"


def _weighted_quantile(values: pd.Series, weights: pd.Series, q: float) -> float:
    frame = pd.DataFrame({"value": values, "weight": weights}).replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame.loc[frame["weight"].gt(0)].sort_values("value")
    if frame.empty:
        return np.nan
    cumulative = frame["weight"].cumsum()
    threshold = q * frame["weight"].sum()
    return float(frame.loc[cumulative.ge(threshold), "value"].iloc[0])


def _add_bucket_movie_weights(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    movie_key = "release_run_id" if "release_run_id" in out.columns else "movie_id"
    if movie_key not in out.columns:
        out["movie_weight"] = 1.0
        return out
    counts = out.groupby(group_cols + [movie_key], dropna=False)["log_residual"].transform("count")
    out["movie_weight"] = 1.0 / counts.replace(0, np.nan)
    return out


def _residual_quantile_payload(
    residuals: pd.Series,
) -> dict[str, object]:
    clean = residuals.replace([np.inf, -np.inf], np.nan).dropna()
    quantiles = clean.quantile([0.025, 0.10, 0.50, 0.90, 0.975])
    return {
        "n": int(clean.size),
        "method": "empirical_log_residual_quantile",
        "sigma_log": float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
        "lo95_log": float(quantiles.loc[0.025]),
        "lo80_log": float(quantiles.loc[0.10]),
        "median_log": float(quantiles.loc[0.50]),
        "hi80_log": float(quantiles.loc[0.90]),
        "hi95_log": float(quantiles.loc[0.975]),
        "residual_samples_log": [float(value) for value in clean.to_numpy()],
    }


def _conditional_cell_key(row: pd.Series, group_cols: list[str]) -> str:
    return "|".join(str(row.get(column, "unknown")) for column in group_cols)


def _build_conditional_quantile_scope(
    work: pd.DataFrame,
    group_cols: list[str],
    *,
    shrink_k: int = 20,
) -> dict[str, object]:
    clean = work.dropna(subset=["log_residual"]).copy()
    movie_key = "release_run_id" if "release_run_id" in clean.columns else "movie_id"
    if movie_key in clean.columns:
        global_counts = clean.groupby(movie_key, dropna=False)["log_residual"].transform("count")
        clean["global_movie_weight"] = 1.0 / global_counts.replace(0, np.nan)
    else:
        clean["global_movie_weight"] = 1.0
    global_center = _weighted_quantile(clean["log_residual"], clean["global_movie_weight"], 0.50)
    global_centered = clean["log_residual"] - global_center
    global_quantiles = {
        "lo80": _weighted_quantile(global_centered, clean["global_movie_weight"], 0.10),
        "hi80": _weighted_quantile(global_centered, clean["global_movie_weight"], 0.90),
        "lo95": _weighted_quantile(global_centered, clean["global_movie_weight"], 0.025),
        "hi95": _weighted_quantile(global_centered, clean["global_movie_weight"], 0.975),
    }

    cells: dict[str, object] = {}
    for key_values, group in clean.groupby(group_cols, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        group = _add_bucket_movie_weights(group, group_cols)
        n = int(group["log_residual"].notna().sum())
        if n == 0:
            continue
        raw_center = _weighted_quantile(group["log_residual"], group["movie_weight"], 0.50)
        weight = n / (n + shrink_k)
        center = weight * raw_center + (1.0 - weight) * global_center
        centered = group["log_residual"] - center
        q = {
            "lo80": _weighted_quantile(centered, group["movie_weight"], 0.10),
            "hi80": _weighted_quantile(centered, group["movie_weight"], 0.90),
            "lo95": _weighted_quantile(centered, group["movie_weight"], 0.025),
            "hi95": _weighted_quantile(centered, group["movie_weight"], 0.975),
        }
        shrunk = {name: weight * value + (1.0 - weight) * global_quantiles[name] for name, value in q.items()}
        key = "|".join(str(value) for value in key_values)
        movie_weights = group.groupby(movie_key, dropna=False)["movie_weight"].sum() if movie_key in group.columns else pd.Series([1.0])
        cells[key] = {
            "n": n,
            "n_movies": int(group[movie_key].nunique()) if movie_key in group.columns else n,
            "top_movie_weight_share": float(movie_weights.max() / movie_weights.sum()) if float(movie_weights.sum()) > 0 else np.nan,
            "center_log": center,
            "lo80_centered_log": shrunk["lo80"],
            "hi80_centered_log": shrunk["hi80"],
            "lo95_centered_log": shrunk["lo95"],
            "hi95_centered_log": shrunk["hi95"],
        }

    return {
        "group_cols": group_cols,
        "global_center_log": global_center,
        "global_centered_quantiles": global_quantiles,
        "cells": cells,
        "shrink_k": shrink_k,
    }


def _build_conditional_quantile_policy(work: pd.DataFrame) -> dict[str, object]:
    scopes = [
        ["origin_bucket", "source_count_bucket", "point_bucket", "release_scope_bucket"],
        ["origin_bucket", "source_count_bucket", "point_bucket"],
        ["origin_bucket", "source_count_bucket"],
        ["origin_bucket"],
    ]
    return {
        "method": "movie_weighted_conditional_centered_empirical_log_residual_quantile_shrunk",
        "required_interval_model": "movie_weighted_conditional_centered_empirical_log_residual_quantile_shrunk",
        "scopes": [_build_conditional_quantile_scope(work, scope) for scope in scopes],
    }


def build_interval_policy(panel_csv: Path | None) -> dict[str, object]:
    policy: dict[str, object] = {
        "default_sigma_log": 0.55,
        "sigma_log_by_origin_day": {},
        "method": "normal_log_error_interval",
        "levels": [80, 95],
    }
    if not panel_csv or not panel_csv.exists():
        return policy
    frame = pd.read_csv(panel_csv)
    selected_columns = {"primary_point_forecast_usd", "selected_lo_95", "selected_hi_95"}
    if selected_columns.issubset(frame.columns):
        policy["method"] = "selected_interval_panel_with_sigma_fallback"
        if "sigma_global" in frame.columns:
            global_sigmas = _finite_positive_series(frame, "sigma_global")
            if not global_sigmas.empty:
                policy["default_sigma_log"] = float(max(0.05, global_sigmas.median()))

        point = pd.to_numeric(frame["primary_point_forecast_usd"], errors="coerce")
        lo95 = pd.to_numeric(frame["selected_lo_95"], errors="coerce")
        hi95 = pd.to_numeric(frame["selected_hi_95"], errors="coerce")
        valid = point.gt(0) & lo95.gt(0) & hi95.gt(point) & np.isfinite(point) & np.isfinite(lo95) & np.isfinite(hi95)
        if "origin_day" in frame.columns and valid.any():
            work = frame.loc[valid, ["origin_day"]].copy()
            work["implied_sigma"] = np.log(hi95.loc[valid] / point.loc[valid]) / 1.95996
            sigmas = work.groupby("origin_day")["implied_sigma"].median().dropna()
            if not sigmas.empty:
                policy["sigma_log_by_origin_day"] = {str(int(k)): float(max(0.05, v)) for k, v in sigmas.items()}
        actual_col = "actual_opening_weekend_gross_usd"
        if actual_col in frame.columns and "origin_day" in frame.columns:
            actual = pd.to_numeric(frame[actual_col], errors="coerce")
            valid_residual = valid & actual.gt(0) & np.isfinite(actual)
            if valid_residual.any():
                keep_cols = [
                    column
                    for column in ["origin_day", "release_run_id", "movie_id", "release_type", "release_width_bucket"]
                    if column in frame.columns
                ]
                work = frame.loc[valid_residual, keep_cols].copy()
                work["source_count_bucket"] = frame.loc[valid_residual].get("source_count", pd.Series(index=work.index)).map(
                    _interval_source_count_bucket
                )
                work["point_bucket"] = point.loc[valid_residual].map(_interval_point_bucket)
                work["origin_bucket"] = work["origin_day"].map(_interval_origin_bucket)
                work["release_scope_bucket"] = work.apply(_interval_release_scope_bucket, axis=1)
                work["log_residual"] = np.log(actual.loc[valid_residual] / point.loc[valid_residual])
                policy["conditional_log_residual_quantiles"] = _build_conditional_quantile_policy(work)
                policy["required_interval_model"] = "movie_weighted_conditional_centered_empirical_log_residual_quantile_shrunk"
                policy["method"] = "movie_weighted_conditional_centered_empirical_log_residual_quantile_shrunk"
                policy["log_residual_quantiles_by_origin_day"] = {
                    str(int(origin_day)): _residual_quantile_payload(group["log_residual"])
                    for origin_day, group in work.groupby("origin_day")
                    if len(group) >= 5
                }
                policy["statistical_interval_note"] = (
                    "Pre-release estimate intervals use uncapped empirical log-residual quantiles. "
                    "Operational caps, if needed, must be applied and stored separately."
                )
        return policy

    forecast_col = "primary_point_forecast_usd" if "primary_point_forecast_usd" in frame.columns else None
    actual_col = "actual_opening_weekend_gross_usd"
    if not forecast_col or actual_col not in frame.columns or "origin_day" not in frame.columns:
        return policy
    actual = pd.to_numeric(frame[actual_col], errors="coerce")
    pred = pd.to_numeric(frame[forecast_col], errors="coerce")
    mask = actual.gt(0) & pred.gt(0)
    work = frame.loc[mask, ["origin_day"]].copy()
    work["log_error"] = np.log(actual.loc[mask] / pred.loc[mask])
    sigmas = work.groupby("origin_day")["log_error"].std(ddof=1).dropna()
    if not sigmas.empty:
        policy["sigma_log_by_origin_day"] = {str(int(k)): float(max(0.05, v)) for k, v in sigmas.items()}
        policy["default_sigma_log"] = float(max(0.05, work["log_error"].std(ddof=1)))
    return policy


def build_daily_interval_policy(daily_baseline_csv: Path | None) -> dict[str, object]:
    policy: dict[str, object] = {
        "method": "empirical_daily_log_residual_quantile",
        "levels": [80, 95],
        "by_regime_day": {},
        "joint_by_regime_days": {},
        "conditional_by_regime_day": {},
    }
    if not daily_baseline_csv or not daily_baseline_csv.exists():
        return policy
    frame = pd.read_csv(daily_baseline_csv)
    for (regime, day), (forecast_col, _source) in BASELINE_COLUMNS.items():
        actual_col = ACTUAL_COLUMNS[day]
        if forecast_col not in frame.columns or actual_col not in frame.columns:
            continue
        forecast = pd.to_numeric(frame[forecast_col], errors="coerce")
        actual = pd.to_numeric(frame[actual_col], errors="coerce")
        mask = forecast.gt(0) & actual.gt(0) & np.isfinite(forecast) & np.isfinite(actual)
        if mask.sum() < 5:
            continue
        residuals = np.log(actual.loc[mask] / forecast.loc[mask])
        by_regime = policy["by_regime_day"].setdefault(regime, {})
        by_regime[day] = _residual_quantile_payload(residuals)

    joint_specs = {
        "live_saturday": {
            "days": ["Saturday", "Sunday"],
            "forecast_cols": {
                "Saturday": "after_fri_sat_usd",
                "Sunday": "after_fri_sun_usd",
            },
        },
    }
    for regime, spec in joint_specs.items():
        days = list(spec["days"])
        forecast_cols = dict(spec["forecast_cols"])
        required_cols = [forecast_cols[day] for day in days] + [ACTUAL_COLUMNS[day] for day in days]
        if not all(column in frame.columns for column in required_cols):
            continue
        mask = pd.Series(True, index=frame.index)
        residual_frame = pd.DataFrame(index=frame.index)
        for day in days:
            forecast = pd.to_numeric(frame[forecast_cols[day]], errors="coerce")
            actual = pd.to_numeric(frame[ACTUAL_COLUMNS[day]], errors="coerce")
            day_mask = forecast.gt(0) & actual.gt(0) & np.isfinite(forecast) & np.isfinite(actual)
            mask &= day_mask
            residual_frame[f"{day}_log_residual"] = np.log(actual / forecast)
        residual_frame = residual_frame.loc[mask]
        if len(residual_frame) < 5:
            continue
        records = [
            {day: float(row[f"{day}_log_residual"]) for day in days}
            for _, row in residual_frame.iterrows()
        ]
        matrix = residual_frame[[f"{day}_log_residual" for day in days]].to_numpy(dtype="float64")
        residual_correlation = 0.0
        if len(residual_frame) > 1 and all(float(np.std(matrix[:, idx])) > 0 for idx in range(matrix.shape[1])):
            residual_correlation = float(np.corrcoef(matrix.T)[0, 1])
        policy["joint_by_regime_days"][regime] = {
            "|".join(days): {
                "method": "joint_empirical_daily_log_residual_bootstrap",
                "n": int(len(residual_frame)),
                "days": days,
                "residual_samples_log": records,
                "residual_correlation": residual_correlation,
            }
        }

    conditional_specs = {
        ("live_sunday", "Sunday"): {
            "conditioning_variable": "signed_saturday_surprise",
            "conditioning_forecast_col": "after_fri_sat_usd",
            "conditioning_actual_col": "actual_sat_usd",
            "target_forecast_col": "after_sat_sun_usd",
            "target_actual_col": "actual_sun_usd",
            "bandwidth": 0.35,
            "shrink_k": 20.0,
        },
    }
    for (regime, day), spec in conditional_specs.items():
        required_cols = [
            spec["conditioning_forecast_col"],
            spec["conditioning_actual_col"],
            spec["target_forecast_col"],
            spec["target_actual_col"],
        ]
        if not all(column in frame.columns for column in required_cols):
            continue
        conditioning_forecast = pd.to_numeric(frame[spec["conditioning_forecast_col"]], errors="coerce")
        conditioning_actual = pd.to_numeric(frame[spec["conditioning_actual_col"]], errors="coerce")
        target_forecast = pd.to_numeric(frame[spec["target_forecast_col"]], errors="coerce")
        target_actual = pd.to_numeric(frame[spec["target_actual_col"]], errors="coerce")
        mask = (
            conditioning_forecast.gt(0)
            & conditioning_actual.gt(0)
            & target_forecast.gt(0)
            & target_actual.gt(0)
            & np.isfinite(conditioning_forecast)
            & np.isfinite(conditioning_actual)
            & np.isfinite(target_forecast)
            & np.isfinite(target_actual)
        )
        if mask.sum() < 5:
            continue
        records = [
            {
                "conditioning_value": float(conditioning_value),
                "log_residual": float(log_residual),
            }
            for conditioning_value, log_residual in zip(
                np.log(conditioning_actual.loc[mask] / conditioning_forecast.loc[mask]),
                np.log(target_actual.loc[mask] / target_forecast.loc[mask]),
            )
        ]
        by_regime = policy["conditional_by_regime_day"].setdefault(regime, {})
        by_regime[day] = {
            "method": "signed_saturday_surprise_weighted_empirical_log_residual_bootstrap",
            "conditioning_variable": spec["conditioning_variable"],
            "n": int(mask.sum()),
            "bandwidth": spec["bandwidth"],
            "shrink_k": spec["shrink_k"],
            "residual_samples": records,
            "fallback": "by_regime_day pooled empirical residuals",
        }
    return policy


def build_amc_interval_policy_from_database(database_url: str | None, *, min_bucket_n: int) -> dict[str, object]:
    if not database_url:
        return empty_amc_interval_policy(sample_key="unknown", min_bucket_n=min_bucket_n)
    conn = connect_database(database_url)
    try:
        return build_amc_interval_policy(conn, min_bucket_n=min_bucket_n)
    finally:
        conn.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-version", default=default_model_version())
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--database-url")
    parser.add_argument("--training-data-cutoff-utc", default=utc_now_iso())
    parser.add_argument("--pre-release-panel", type=Path, default=DIAGNOSTICS_DIR / "rolling_forecast_origin_interval_panel.csv")
    parser.add_argument("--point-policy-csv", type=Path, default=DIAGNOSTICS_DIR / "rolling_forecast_origin_point_model_policy.csv")
    parser.add_argument("--daily-baseline-csv", type=Path, default=PREDICTIONS_DIR / "daily_regime_baseline_forecasts.csv")
    parser.add_argument("--live-plugin-nowcasts-csv", type=Path, default=PREDICTIONS_DIR / "live_daily_plugin_nowcasts.csv")
    parser.add_argument("--thursday-amc-preview-policy", type=Path, default=DIAGNOSTICS_DIR / "thursday_amc_preview_policy.json")
    parser.add_argument("--amc-interval-min-bucket-n", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    artifact_dir = args.artifact_root / args.model_version
    artifact_dir.mkdir(parents=True, exist_ok=True)

    pre_release_panel = copy_if_exists(args.pre_release_panel, artifact_dir / "pre_release_panel.csv")
    daily_baseline = copy_if_exists(args.daily_baseline_csv, artifact_dir / "daily_regime_baseline_forecasts.csv")
    live_plugin = copy_if_exists(args.live_plugin_nowcasts_csv, artifact_dir / "live_daily_plugin_nowcasts.csv")
    thursday_amc_preview_policy_path = copy_if_exists(
        args.thursday_amc_preview_policy,
        artifact_dir / "thursday_amc_preview_policy.json",
    )

    point_policy = build_locked_point_policy(args.point_policy_csv)
    interval_policy = build_interval_policy(args.pre_release_panel)
    daily_interval_policy = build_daily_interval_policy(args.daily_baseline_csv)
    amc_interval_policy = build_amc_interval_policy_from_database(
        args.database_url,
        min_bucket_n=max(2, args.amc_interval_min_bucket_n),
    )
    amc_diagnostics = amc_interval_policy.pop("diagnostics", {})
    write_json(artifact_dir / "pre_release_point_policy.json", point_policy)
    write_json(artifact_dir / "pre_release_interval_policy.json", interval_policy)
    write_json(artifact_dir / "daily_interval_policy.json", daily_interval_policy)
    write_json(artifact_dir / "amc_interval_policy.json", amc_interval_policy)
    for name, rows in amc_diagnostics.items():
        pd.DataFrame(rows).to_csv(artifact_dir / f"{name}.csv", index=False)

    manifest = {
        "model_version": args.model_version,
        "created_at_utc": utc_now_iso(),
        "training_data_cutoff_utc": args.training_data_cutoff_utc,
        "pre_release": {
            "point_policy_path": "pre_release_point_policy.json",
            "interval_policy_path": "pre_release_interval_policy.json",
            "panel_path": pre_release_panel,
            "point_policy": point_policy,
            "interval_policy": interval_policy,
        },
        "daily_baseline": {
            "baseline_path": daily_baseline,
        },
        "amc": {
            "live_plugin_nowcasts_path": live_plugin,
            "interval_policy_path": "amc_interval_policy.json",
            "interval_cell_stability_path": "amc_interval_cell_stability.csv",
            "tail_case_audit_path": "amc_tail_case_audit.csv",
            "origins": ["10:00", "12:00", "14:00", "16:00", "18:00", "20:00", "EOD"],
        },
        "thursday_amc_preview": {
            "policy_path": thursday_amc_preview_policy_path,
        },
        "composition": {
            "n_sim": 50000,
            "interval_levels": [80, 95],
            "daily_interval_policy_path": "daily_interval_policy.json",
            "use_daily_error_covariance": False,
            "fallback_to_independent_components": True,
            "fallback_daily_sigma_log": 0.45,
        },
    }
    write_json(artifact_dir / "manifest.json", manifest)
    print(f"Wrote artifacts to {artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
