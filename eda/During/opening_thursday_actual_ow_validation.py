#!/usr/bin/env python3
"""Validate official opening-Thursday daily gross as an OW updater.

This diagnostic is intentionally independent from the AMC Thursday nowcast.
It answers a narrower production question:

    If The Numbers has the Thursday daily gross immediately preceding opening
    Friday/weekend, does updating the leakage-safe pre-Thursday OW baseline
    improve OW forecasts on historical rolling-origin holdouts?

The target is not a canonical preview tag. It is the daily gross row where
``box_office_date = opening_weekend_start - 1 day`` for eligible wide releases.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eda.During.thursday_preview_lift_diagnostic import (  # noqa: E402
    DEFAULT_MIN_TRAIN_N,
    active_pre_release_panel_path,
    load_pre_release_panel,
    ols_alpha_beta,
)
from models.boxoffice.opening_thursday_actual import (  # noqa: E402
    BUCKET_THRESHOLDS_USD,
    fit_ratio_update_policy,
)
from pm_box_office.db.connection import connect_database  # noqa: E402
from pm_box_office.sources.common.cli import add_database_arg  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "diagnostics" / "opening_thursday_actual_ow_validation"
@dataclass(frozen=True)
class Outputs:
    panel: pd.DataFrame
    year_folds: pd.DataFrame
    metrics_by_origin_baseline: pd.DataFrame
    predictions: pd.DataFrame
    bucket_scores: pd.DataFrame
    promotion_summary: dict[str, Any]
    exclusion_audit: pd.DataFrame
    subgroup_metrics: pd.DataFrame
    frozen_policy: dict[str, Any]


def opening_thursday_daily_sql(start_year: int = 2017, *, wide_theater_threshold: int = 600) -> str:
    return f"""
        SELECT
            o.release_run_id,
            o.movie_id,
            o.title,
            o.opening_date::date AS opening_date,
            o.opening_weekend_start::date AS opening_weekend_start,
            o.release_year,
            o.release_width_bucket,
            o.release_type,
            o.opening_weekend_theaters,
            o.opening_weekend_gross_usd::double precision AS actual_opening_weekend_gross_usd,
            d.daily_box_office_id,
            d.box_office_date::date AS opening_thursday_date,
            d.gross_usd::double precision AS opening_thursday_daily_gross_usd,
            d.source AS opening_thursday_source,
            d.source_url AS opening_thursday_source_url,
            d.fetched_at AS opening_thursday_fetched_at
        FROM analytics.eda_movie_openings o
        JOIN daily_box_office d
          ON d.release_run_id = o.release_run_id
         AND d.box_office_date::date = o.opening_weekend_start::date - INTERVAL '1 day'
        WHERE o.opening_weekend_start::date >= DATE '{int(start_year)}-01-01'
          AND COALESCE(o.opening_weekend_theaters, 0) >= {int(wide_theater_threshold)}
          AND o.opening_weekend_gross_usd IS NOT NULL
          AND o.opening_weekend_gross_usd > 0
          AND d.gross_usd IS NOT NULL
          AND d.gross_usd > 0
          AND LOWER(COALESCE(d.source, '')) LIKE '%number%'
        ORDER BY o.opening_weekend_start::date, o.movie_id, d.daily_box_office_id
    """


def fetch_frame(conn: Any, sql: str) -> pd.DataFrame:
    cursor = conn.execute(sql)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def fetch_opening_thursday_daily_rows(
    database_url: str | None = None,
    *,
    start_year: int = 2017,
    wide_theater_threshold: int = 600,
) -> pd.DataFrame:
    conn = connect_database(database_url)
    try:
        return fetch_frame(conn, opening_thursday_daily_sql(start_year, wide_theater_threshold=wide_theater_threshold))
    finally:
        conn.close()


def _date_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def validate_opening_thursday_daily(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    work = rows.copy()
    for column in ["opening_date", "opening_weekend_start", "opening_thursday_date", "opening_thursday_fetched_at"]:
        if column in work:
            work[column] = _date_series(work[column])
    work["opening_thursday_daily_gross_usd"] = pd.to_numeric(work["opening_thursday_daily_gross_usd"], errors="coerce")
    work["actual_opening_weekend_gross_usd"] = pd.to_numeric(work["actual_opening_weekend_gross_usd"], errors="coerce")

    output: list[dict[str, Any]] = []
    for (release_run_id, movie_id), group in work.groupby(["release_run_id", "movie_id"], sort=False):
        positive = group.loc[group["opening_thursday_daily_gross_usd"].gt(0)].copy()
        distinct = positive[["opening_thursday_date", "opening_thursday_daily_gross_usd"]].drop_duplicates()
        date_count = int(positive["opening_thursday_date"].nunique(dropna=True))
        gross_count = int(positive["opening_thursday_daily_gross_usd"].nunique(dropna=True))
        raw_row_count = int(len(group))
        distinct_row_count = int(len(distinct))
        unclear = date_count != 1 or gross_count != 1 or distinct_row_count != 1
        duplicate_identical = raw_row_count > 1 and not unclear
        first = group.sort_values(["opening_thursday_date", "daily_box_office_id"], na_position="last").iloc[0]
        expected_date = pd.to_datetime(first["opening_weekend_start"], errors="coerce").normalize() - pd.Timedelta(days=1)
        output.append(
            {
                "release_run_id": int(release_run_id),
                "movie_id": int(movie_id),
                "title": first.get("title"),
                "opening_date": first.get("opening_date"),
                "opening_weekend_start": first.get("opening_weekend_start"),
                "opening_thursday_date": positive["opening_thursday_date"].min() if not positive.empty else pd.NaT,
                "release_year": first.get("release_year"),
                "release_width_bucket": first.get("release_width_bucket"),
                "release_type": first.get("release_type"),
                "opening_weekend_theaters": first.get("opening_weekend_theaters"),
                "actual_opening_weekend_gross_usd": first.get("actual_opening_weekend_gross_usd"),
                "opening_thursday_daily_row_count": raw_row_count,
                "opening_thursday_daily_distinct_row_count": distinct_row_count,
                "opening_thursday_daily_date_count": date_count,
                "opening_thursday_daily_gross_value_count": gross_count,
                "opening_thursday_daily_gross_usd": (
                    float(positive["opening_thursday_daily_gross_usd"].iloc[0])
                    if not positive.empty and not unclear
                    else np.nan
                ),
                "opening_thursday_source": first.get("opening_thursday_source"),
                "opening_thursday_source_url": first.get("opening_thursday_source_url"),
                "opening_thursday_fetched_at": first.get("opening_thursday_fetched_at"),
                "target_definition": "opening_thursday_daily_gross",
                "target_date_matches_opening_weekend_minus_one": bool(
                    pd.notna(expected_date)
                    and not positive.empty
                    and positive["opening_thursday_date"].eq(expected_date).all()
                ),
                "target_validation_status": (
                    "unclear_opening_thursday_daily_aggregation"
                    if unclear
                    else "duplicate_opening_thursday_daily_rows"
                    if duplicate_identical
                    else "validated_opening_thursday_daily"
                ),
            }
        )
    return pd.DataFrame(output)


def select_pre_thursday_baselines(targets: pd.DataFrame, pre_release_panel: pd.DataFrame) -> pd.DataFrame:
    if targets.empty:
        return targets.copy()
    required = {
        "release_run_id",
        "movie_id",
        "origin_day",
        "forecast_origin_date",
        "latest_estimate_date",
        "primary_point_forecast_usd",
        "primary_point_method",
    }
    missing = required - set(pre_release_panel.columns)
    if missing:
        raise ValueError(f"Pre-release panel missing required columns: {sorted(missing)}")

    work = targets.copy()
    work["opening_thursday_date"] = _date_series(work["opening_thursday_date"])
    panel = pre_release_panel.copy()
    panel["forecast_origin_date"] = _date_series(panel["forecast_origin_date"])
    panel["latest_estimate_date"] = _date_series(panel["latest_estimate_date"])
    panel["primary_point_forecast_usd"] = pd.to_numeric(panel["primary_point_forecast_usd"], errors="coerce")

    baseline_rows: list[dict[str, Any]] = []
    for row in work.itertuples(index=False):
        movie_panel = panel.loc[
            panel["release_run_id"].eq(row.release_run_id) & panel["movie_id"].eq(row.movie_id)
        ].copy()
        has_panel_row = bool(not movie_panel.empty)
        has_any_positive_baseline = bool(movie_panel["primary_point_forecast_usd"].gt(0).any()) if has_panel_row else False
        candidates = movie_panel.loc[
            movie_panel["primary_point_forecast_usd"].gt(0)
            & movie_panel["forecast_origin_date"].lt(row.opening_thursday_date)
            & movie_panel["latest_estimate_date"].lt(row.opening_thursday_date)
        ].copy()
        if candidates.empty:
            baseline_rows.append(
                {
                    "release_run_id": row.release_run_id,
                    "movie_id": row.movie_id,
                    "baseline_available": False,
                    "baseline_candidate_count": 0,
                    "has_panel_row": has_panel_row,
                    "has_any_positive_baseline": has_any_positive_baseline,
                    "baseline_origin_day": np.nan,
                    "baseline_forecast_origin_date": pd.NaT,
                    "baseline_latest_estimate_date": pd.NaT,
                    "baseline_forecast_usd": np.nan,
                    "baseline_primary_point_method": None,
                    "baseline_source_count": np.nan,
                    "baseline_selected_interval_model": None,
                    "baseline_lo80_usd": np.nan,
                    "baseline_hi80_usd": np.nan,
                    "baseline_lo95_usd": np.nan,
                    "baseline_hi95_usd": np.nan,
                }
            )
            continue
        selected = candidates.sort_values(["forecast_origin_date", "origin_day"]).iloc[-1]
        baseline_rows.append(
            {
                "release_run_id": row.release_run_id,
                "movie_id": row.movie_id,
                "baseline_available": True,
                "baseline_candidate_count": int(len(candidates)),
                "has_panel_row": has_panel_row,
                "has_any_positive_baseline": has_any_positive_baseline,
                "baseline_origin_day": int(selected["origin_day"]),
                "baseline_forecast_origin_date": selected["forecast_origin_date"],
                "baseline_latest_estimate_date": selected["latest_estimate_date"],
                "baseline_forecast_usd": float(selected["primary_point_forecast_usd"]),
                "baseline_primary_point_method": selected.get("primary_point_method"),
                "baseline_source_count": selected.get("source_count", np.nan),
                "baseline_selected_interval_model": selected.get("selected_interval_model"),
                "baseline_lo80_usd": selected.get("selected_lo_80", np.nan),
                "baseline_hi80_usd": selected.get("selected_hi_80", np.nan),
                "baseline_lo95_usd": selected.get("selected_lo_95", np.nan),
                "baseline_hi95_usd": selected.get("selected_hi_95", np.nan),
            }
        )
    return work.merge(pd.DataFrame(baseline_rows), on=["release_run_id", "movie_id"], how="left")


def build_panel(target_rows: pd.DataFrame, pre_release_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    audited = select_pre_thursday_baselines(validate_opening_thursday_daily(target_rows), pre_release_panel)
    if audited.empty:
        return audited.copy(), audited.copy()
    for column in ["actual_opening_weekend_gross_usd", "baseline_forecast_usd", "opening_thursday_daily_gross_usd"]:
        audited[column] = pd.to_numeric(audited[column], errors="coerce")

    reasons: list[str] = []
    included: list[bool] = []
    for row in audited.itertuples(index=False):
        reason = ""
        ok = True
        if row.target_validation_status == "unclear_opening_thursday_daily_aggregation":
            reason = "unclear_opening_thursday_daily_aggregation"
            ok = False
        elif not bool(row.target_date_matches_opening_weekend_minus_one):
            reason = "target_date_not_opening_weekend_minus_one"
            ok = False
        elif not np.isfinite(row.actual_opening_weekend_gross_usd) or row.actual_opening_weekend_gross_usd <= 0:
            reason = "invalid_actual_opening_weekend"
            ok = False
        elif not bool(row.baseline_available):
            reason = "missing_leakage_safe_pre_thursday_baseline"
            if bool(row.has_panel_row) and not bool(row.has_any_positive_baseline):
                reason = "non_positive_baseline"
            ok = False
        elif not np.isfinite(row.baseline_forecast_usd) or row.baseline_forecast_usd <= 0:
            reason = "non_positive_baseline"
            ok = False
        elif not np.isfinite(row.opening_thursday_daily_gross_usd) or row.opening_thursday_daily_gross_usd <= 0:
            reason = "invalid_opening_thursday_daily_gross"
            ok = False
        reasons.append(reason)
        included.append(ok)

    audited["included"] = included
    audited["exclusion_reason"] = reasons
    panel = audited.loc[audited["included"]].copy()
    panel["e_log"] = np.log(panel["actual_opening_weekend_gross_usd"] / panel["baseline_forecast_usd"])
    panel["x_log_thursday_ratio"] = np.log(panel["opening_thursday_daily_gross_usd"] / panel["baseline_forecast_usd"])
    panel["ow_scale_bucket"] = panel["actual_opening_weekend_gross_usd"].map(scale_bucket)
    panel["thursday_scale_bucket"] = panel["opening_thursday_daily_gross_usd"].map(scale_bucket)
    panel = panel.sort_values(["opening_weekend_start", "movie_id"]).reset_index(drop=True)
    audited = audited.sort_values(["opening_weekend_start", "movie_id"]).reset_index(drop=True)
    return panel, audited


def _fit_log_linear(train: pd.DataFrame) -> tuple[float, float, float]:
    y = np.log(pd.to_numeric(train["actual_opening_weekend_gross_usd"], errors="coerce"))
    x_b = np.log(pd.to_numeric(train["baseline_forecast_usd"], errors="coerce"))
    x_t = np.log(pd.to_numeric(train["opening_thursday_daily_gross_usd"], errors="coerce"))
    valid = np.isfinite(y) & np.isfinite(x_b) & np.isfinite(x_t)
    if valid.sum() < 3:
        return np.nan, np.nan, np.nan
    design = np.column_stack([np.ones(int(valid.sum())), x_b[valid], x_t[valid]])
    alpha, beta_b, beta_t = np.linalg.lstsq(design, y[valid], rcond=None)[0]
    return float(alpha), float(beta_b), float(beta_t)


def _interval_from_residuals(point: float, residuals: np.ndarray) -> dict[str, float]:
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size < 5 or not math.isfinite(point) or point <= 0:
        return {"lo80_usd": np.nan, "hi80_usd": np.nan, "lo95_usd": np.nan, "hi95_usd": np.nan}
    return {
        "lo80_usd": float(point * np.exp(np.quantile(residuals, 0.10))),
        "hi80_usd": float(point * np.exp(np.quantile(residuals, 0.90))),
        "lo95_usd": float(point * np.exp(np.quantile(residuals, 0.025))),
        "hi95_usd": float(point * np.exp(np.quantile(residuals, 0.975))),
    }


def build_rolling_predictions(panel: pd.DataFrame, *, min_train_n: int = DEFAULT_MIN_TRAIN_N) -> pd.DataFrame:
    rows = panel.sort_values(["opening_weekend_start", "movie_id"]).reset_index(drop=True).copy()
    if rows.empty:
        return rows
    outputs: list[dict[str, Any]] = []
    for _, test in rows.iterrows():
        prior = rows.loc[rows["opening_weekend_start"].lt(test["opening_weekend_start"])].copy()
        train_n = int(len(prior))
        scored = train_n >= min_train_n
        baseline = float(test["baseline_forecast_usd"])
        forecasts = {
            "base_pre_thursday": baseline,
            "historical_baseline_bias": np.nan,
            "ratio_update": np.nan,
            "log_linear_update": np.nan,
        }
        params = {
            "bias_alpha": np.nan,
            "ratio_alpha": np.nan,
            "ratio_beta": np.nan,
            "log_linear_alpha": np.nan,
            "log_linear_beta_baseline": np.nan,
            "log_linear_beta_thursday": np.nan,
        }
        residual_pools: dict[str, np.ndarray] = {
            "base_pre_thursday": np.array([], dtype="float64"),
            "historical_baseline_bias": np.array([], dtype="float64"),
            "ratio_update": np.array([], dtype="float64"),
            "log_linear_update": np.array([], dtype="float64"),
        }
        if scored:
            bias_alpha = float(prior["e_log"].mean())
            ratio_alpha, ratio_beta = ols_alpha_beta(prior["x_log_thursday_ratio"], prior["e_log"])
            ll_alpha, ll_beta_b, ll_beta_t = _fit_log_linear(prior)
            forecasts["historical_baseline_bias"] = float(baseline * np.exp(bias_alpha))
            forecasts["ratio_update"] = float(baseline * np.exp(ratio_alpha + ratio_beta * test["x_log_thursday_ratio"]))
            forecasts["log_linear_update"] = float(
                np.exp(ll_alpha + ll_beta_b * np.log(baseline) + ll_beta_t * np.log(float(test["opening_thursday_daily_gross_usd"])))
            )
            params.update(
                {
                    "bias_alpha": bias_alpha,
                    "ratio_alpha": ratio_alpha,
                    "ratio_beta": ratio_beta,
                    "log_linear_alpha": ll_alpha,
                    "log_linear_beta_baseline": ll_beta_b,
                    "log_linear_beta_thursday": ll_beta_t,
                }
            )
            actual_log = np.log(prior["actual_opening_weekend_gross_usd"].astype(float))
            base_log = np.log(prior["baseline_forecast_usd"].astype(float))
            ratio_pred_log = base_log + ratio_alpha + ratio_beta * prior["x_log_thursday_ratio"].astype(float)
            ll_pred_log = (
                ll_alpha
                + ll_beta_b * np.log(prior["baseline_forecast_usd"].astype(float))
                + ll_beta_t * np.log(prior["opening_thursday_daily_gross_usd"].astype(float))
            )
            residual_pools = {
                "base_pre_thursday": (actual_log - base_log).to_numpy(dtype="float64"),
                "historical_baseline_bias": (actual_log - (base_log + bias_alpha)).to_numpy(dtype="float64"),
                "ratio_update": (actual_log - ratio_pred_log).to_numpy(dtype="float64"),
                "log_linear_update": (actual_log - ll_pred_log).to_numpy(dtype="float64"),
            }

        base_payload = {**test.to_dict(), "rolling_train_n": train_n, "scored": scored, **params}
        for model, point in forecasts.items():
            intervals = _interval_from_residuals(float(point) if pd.notna(point) else np.nan, residual_pools[model])
            outputs.append(
                {
                    **base_payload,
                    "ow_model": model,
                    "forecast_ow_usd": point,
                    "residual_pool_n": int(np.isfinite(residual_pools[model]).sum()),
                    **intervals,
                }
            )
    return pd.DataFrame(outputs)


def _error_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    frame = predictions.loc[predictions["scored"]].copy()
    actual = pd.to_numeric(frame["actual_opening_weekend_gross_usd"], errors="coerce")
    forecast = pd.to_numeric(frame["forecast_ow_usd"], errors="coerce")
    valid = actual.gt(0) & forecast.gt(0)
    out = frame.loc[valid].copy()
    out["actual_usd"] = actual.loc[valid].to_numpy(dtype="float64")
    out["forecast_usd"] = forecast.loc[valid].to_numpy(dtype="float64")
    out["log_error"] = np.log(out["actual_usd"] / out["forecast_usd"])
    out["abs_log_error"] = out["log_error"].abs()
    out["squared_log_error"] = out["log_error"] ** 2
    out["abs_dollar_error"] = (out["actual_usd"] - out["forecast_usd"]).abs()
    return out


def interval_score(actual: pd.Series, lo: pd.Series, hi: pd.Series, *, alpha: float) -> pd.Series:
    width = hi - lo
    lower_penalty = (2.0 / alpha) * (lo - actual).clip(lower=0)
    upper_penalty = (2.0 / alpha) * (actual - hi).clip(lower=0)
    return width + lower_penalty + upper_penalty


def summarize_errors(errors: pd.DataFrame, *, group_cols: list[str] | None = None) -> pd.DataFrame:
    group_cols = group_cols or []
    if errors.empty:
        return pd.DataFrame()
    grouped = [((), errors)] if not group_cols else errors.groupby(group_cols, dropna=False)
    rows: list[dict[str, Any]] = []
    for key, group in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        payload = dict(zip(group_cols, key_tuple))
        valid80 = group["lo80_usd"].notna() & group["hi80_usd"].notna()
        valid95 = group["lo95_usd"].notna() & group["hi95_usd"].notna()
        wis80 = interval_score(group.loc[valid80, "actual_usd"], group.loc[valid80, "lo80_usd"], group.loc[valid80, "hi80_usd"], alpha=0.20)
        wis95 = interval_score(group.loc[valid95, "actual_usd"], group.loc[valid95, "lo95_usd"], group.loc[valid95, "hi95_usd"], alpha=0.05)
        rows.append(
            {
                **payload,
                "n": int(len(group)),
                "movie_n": int(group["movie_id"].nunique()),
                "release_week_n": int(group["opening_weekend_start"].nunique()),
                "ME_log": float(group["log_error"].mean()),
                "MAE_log": float(group["abs_log_error"].mean()),
                "RMSE_log": float(np.sqrt(group["squared_log_error"].mean())),
                "dollar_MAE": float(group["abs_dollar_error"].mean()),
                "coverage80": float(group.loc[valid80, "actual_usd"].between(group.loc[valid80, "lo80_usd"], group.loc[valid80, "hi80_usd"]).mean()) if valid80.any() else np.nan,
                "coverage95": float(group.loc[valid95, "actual_usd"].between(group.loc[valid95, "lo95_usd"], group.loc[valid95, "hi95_usd"]).mean()) if valid95.any() else np.nan,
                "WIS": float(np.nanmean([wis80.mean() if len(wis80) else np.nan, wis95.mean() if len(wis95) else np.nan])),
                "harmful_update_rate_vs_base": np.nan,
                "median_harmful_update_log_magnitude": np.nan,
            }
        )
    return pd.DataFrame(rows)


def add_harmful_update_rates(metrics: pd.DataFrame, errors: pd.DataFrame, *, group_cols: list[str] | None = None) -> pd.DataFrame:
    group_cols = group_cols or []
    if metrics.empty or errors.empty:
        return metrics
    base = errors.loc[errors["ow_model"].eq("base_pre_thursday"), ["release_run_id", "movie_id", "abs_log_error", *group_cols]].rename(
        columns={"abs_log_error": "base_abs_log_error"}
    )
    paired = errors.merge(base, on=["release_run_id", "movie_id", *group_cols], how="left")
    paired["harmful"] = paired["abs_log_error"].gt(paired["base_abs_log_error"])
    paired["harmful_magnitude"] = (paired["abs_log_error"] - paired["base_abs_log_error"]).where(paired["harmful"])
    grouped = paired.groupby(["ow_model", *group_cols], dropna=False)
    harm = grouped.agg(
        harmful_update_rate_vs_base=("harmful", "mean"),
        median_harmful_update_log_magnitude=("harmful_magnitude", "median"),
    ).reset_index()
    return metrics.drop(columns=["harmful_update_rate_vs_base", "median_harmful_update_log_magnitude"], errors="ignore").merge(
        harm, on=["ow_model", *group_cols], how="left"
    )


def evaluate_predictions(predictions: pd.DataFrame, *, group_cols: list[str] | None = None) -> pd.DataFrame:
    errors = _error_frame(predictions)
    metrics = summarize_errors(errors, group_cols=["ow_model", *(group_cols or [])])
    return add_harmful_update_rates(metrics, errors, group_cols=group_cols)


def _bucket_index(value: float) -> int:
    return int(np.searchsorted(BUCKET_THRESHOLDS_USD, value, side="right"))


def bucket_scores(predictions: pd.DataFrame, *, smoothing_alpha: float = 0.5) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    errors = _error_frame(predictions)
    for row in errors.itertuples(index=False):
        if not np.isfinite(row.forecast_usd) or row.forecast_usd <= 0 or int(row.residual_pool_n) < 5:
            continue
        # Approximate each row's predictive distribution from its interval width
        # when full residual pools are not persisted row-by-row. This keeps the
        # probability score deterministic and fold-local.
        if np.isfinite(row.lo80_usd) and np.isfinite(row.hi80_usd) and row.lo80_usd > 0 and row.hi80_usd > row.lo80_usd:
            sigma = (np.log(row.hi80_usd) - np.log(row.lo80_usd)) / (2 * 1.281551565545)
        else:
            sigma = np.nan
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        bucket_edges = np.r_[0.0, BUCKET_THRESHOLDS_USD, np.inf]
        mu = math.log(row.forecast_usd)
        probs = []
        for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:]):
            cdf_hi = 1.0 if math.isinf(hi) else _normal_cdf((math.log(hi) - mu) / sigma)
            cdf_lo = 0.0 if lo <= 0 else _normal_cdf((math.log(lo) - mu) / sigma)
            probs.append(max(0.0, cdf_hi - cdf_lo))
        probs_arr = np.asarray(probs, dtype="float64") + float(smoothing_alpha)
        probs_arr = probs_arr / probs_arr.sum()
        observed = _bucket_index(float(row.actual_usd))
        observed_cdf = (np.arange(len(probs_arr)) >= observed).astype("float64")
        rows.append(
            {
                "release_run_id": row.release_run_id,
                "movie_id": row.movie_id,
                "title": row.title,
                "opening_weekend_start": row.opening_weekend_start,
                "ow_model": row.ow_model,
                "actual_ow_usd": row.actual_usd,
                "forecast_ow_usd": row.forecast_usd,
                "observed_bucket": observed,
                "observed_bucket_probability": float(probs_arr[observed]),
                "bucket_log_score": float(-np.log(max(probs_arr[observed], 1e-12))),
                "RPS": float(np.sum((np.cumsum(probs_arr)[:-1] - observed_cdf[:-1]) ** 2)),
            }
        )
    return pd.DataFrame(rows)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def scale_bucket(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(v) or v <= 0:
        return "unknown"
    if v < 5_000_000:
        return "lt_5m"
    if v < 15_000_000:
        return "5m_15m"
    if v < 50_000_000:
        return "15m_50m"
    if v < 100_000_000:
        return "50m_100m"
    return "100m_plus"


def build_year_folds(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    scored = predictions.loc[predictions["scored"]].copy()
    if scored.empty:
        return pd.DataFrame()
    movie_level = scored.drop_duplicates(["release_run_id", "movie_id"])
    return (
        movie_level.groupby("release_year", dropna=False)
        .agg(
            test_movie_n=("movie_id", "nunique"),
            test_release_week_n=("opening_weekend_start", "nunique"),
            min_train_n=("rolling_train_n", "min"),
            max_train_n=("rolling_train_n", "max"),
        )
        .reset_index()
        .sort_values("release_year")
    )


def subgroup_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for label, cols in [
        ("release_year", ["release_year"]),
        ("ow_scale_bucket", ["ow_scale_bucket"]),
        ("thursday_scale_bucket", ["thursday_scale_bucket"]),
        ("baseline_origin_day", ["baseline_origin_day"]),
        ("release_width_bucket", ["release_width_bucket"]),
        ("calendar_date", ["opening_weekend_start"]),
    ]:
        metrics = evaluate_predictions(predictions, group_cols=cols)
        if not metrics.empty:
            metrics.insert(0, "grouping", label)
            metric_group_col = cols[0]
            metrics["group"] = metrics[metric_group_col].astype(str)
            frames.append(metrics)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_promotion_summary(panel: pd.DataFrame, predictions: pd.DataFrame, metrics: pd.DataFrame, bucket: pd.DataFrame, *, min_promote_n: int) -> dict[str, Any]:
    model_metrics = metrics.set_index("ow_model") if not metrics.empty and "ow_model" in metrics else pd.DataFrame()
    base = model_metrics.loc["base_pre_thursday"].to_dict() if "base_pre_thursday" in model_metrics.index else {}
    update_candidates = model_metrics.loc[
        [idx for idx in ["ratio_update", "log_linear_update"] if idx in model_metrics.index]
    ].copy() if not model_metrics.empty else pd.DataFrame()
    if not update_candidates.empty:
        selected_update_model = str(update_candidates.sort_values(["MAE_log", "RMSE_log"]).index[0])
        update = model_metrics.loc[selected_update_model].to_dict()
    else:
        selected_update_model = None
        update = {}
    ratio_update = model_metrics.loc["ratio_update"].to_dict() if "ratio_update" in model_metrics.index else {}
    log_linear_update = model_metrics.loc["log_linear_update"].to_dict() if "log_linear_update" in model_metrics.index else {}
    scored = predictions.loc[predictions["scored"]].drop_duplicates(["release_run_id", "movie_id"]) if not predictions.empty else pd.DataFrame()
    sufficient_n = len(scored) >= min_promote_n
    improves_mae = bool(update.get("MAE_log", np.inf) < base.get("MAE_log", -np.inf))
    improves_rmse = bool(update.get("RMSE_log", np.inf) <= base.get("RMSE_log", -np.inf))
    harmful_rate = float(update.get("harmful_update_rate_vs_base", np.nan))
    harmful_ok = bool(np.isfinite(harmful_rate) and harmful_rate <= 0.50)
    recommendation = "enable_actual_opening_thursday_to_ow" if sufficient_n and improves_mae and improves_rmse and harmful_ok else "hold"
    bucket_summary = (
        bucket.groupby("ow_model")
        .agg(bucket_log_score=("bucket_log_score", "mean"), RPS=("RPS", "mean"), n=("release_run_id", "count"))
        .reset_index()
        .to_dict(orient="records")
        if not bucket.empty
        else []
    )
    return {
        "target_definition": "opening_thursday_daily_gross",
        "model_scope": "official_the_numbers_opening_thursday_actual_to_opening_weekend",
        "independent_from_amc_nowcast": True,
        "promotion_recommendation": recommendation,
        "production_ow_update_enabled": recommendation == "enable_actual_opening_thursday_to_ow",
        "selected_update_model": selected_update_model,
        "min_promote_n": int(min_promote_n),
        "eligible_panel_movie_n": int(panel["movie_id"].nunique()) if not panel.empty else 0,
        "eligible_panel_release_run_n": int(panel["release_run_id"].nunique()) if not panel.empty else 0,
        "scored_movie_n": int(scored["movie_id"].nunique()) if not scored.empty else 0,
        "scored_release_week_n": int(scored["opening_weekend_start"].nunique()) if not scored.empty else 0,
        "sufficient_history": sufficient_n,
        "base_pre_thursday": base,
        "selected_update": update,
        "ratio_update": ratio_update,
        "log_linear_update": log_linear_update,
        "bucket_scores": bucket_summary,
        "safeguards_required": [
            "use_only_the_numbers_opening_thursday_daily_gross",
            "baseline_forecast_origin_and_latest_estimate_before_thursday",
            "rolling_origin_frozen_coefficients",
            "idempotent_actual_thursday_daily_updates",
            "fallback_to_baseline_when_thursday_actual_missing",
        ],
    }


def run(
    database_url: str | None,
    *,
    pre_release_panel_path: Path | None,
    start_year: int,
    wide_theater_threshold: int,
    min_train_n: int,
    min_promote_n: int,
    output_dir: Path,
) -> Outputs:
    raw = fetch_opening_thursday_daily_rows(database_url, start_year=start_year, wide_theater_threshold=wide_theater_threshold)
    pre_release = load_pre_release_panel(pre_release_panel_path)
    panel, audit = build_panel(raw, pre_release)
    predictions = build_rolling_predictions(panel, min_train_n=min_train_n)
    metrics = evaluate_predictions(predictions)
    metrics_by_origin = evaluate_predictions(predictions, group_cols=["baseline_origin_day"])
    buckets = bucket_scores(predictions)
    if not buckets.empty:
        bucket_summary = buckets.groupby("ow_model").agg(bucket_log_score=("bucket_log_score", "mean"), RPS=("RPS", "mean"), n=("release_run_id", "count")).reset_index()
        metrics_by_origin = pd.concat(
            [
                metrics_by_origin,
                bucket_summary.assign(baseline_origin_day="ALL"),
            ],
            ignore_index=True,
            sort=False,
        )
    year_folds = build_year_folds(predictions)
    subgroups = subgroup_metrics(predictions)
    summary = build_promotion_summary(panel, predictions, metrics, buckets, min_promote_n=min_promote_n)
    frozen_policy = fit_ratio_update_policy(
        panel,
        validation_metrics=summary,
        min_train_n=min_train_n,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_dir / "opening_thursday_actual_panel.csv", index=False)
    year_folds.to_csv(output_dir / "opening_thursday_actual_year_folds.csv", index=False)
    metrics_by_origin.to_csv(output_dir / "opening_thursday_actual_metrics_by_origin_baseline.csv", index=False)
    predictions.to_csv(output_dir / "opening_thursday_actual_ow_predictions.csv", index=False)
    buckets.to_csv(output_dir / "opening_thursday_actual_bucket_scores.csv", index=False)
    audit.to_csv(output_dir / "opening_thursday_actual_exclusion_audit.csv", index=False)
    subgroups.to_csv(output_dir / "opening_thursday_actual_subgroup_metrics.csv", index=False)
    (output_dir / "opening_thursday_actual_promotion_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "opening_thursday_actual_ratio_update_policy.json").write_text(
        json.dumps(frozen_policy, indent=2, sort_keys=True) + "\n"
    )
    return Outputs(
        panel=panel,
        year_folds=year_folds,
        metrics_by_origin_baseline=metrics_by_origin,
        predictions=predictions,
        bucket_scores=buckets,
        promotion_summary=summary,
        exclusion_audit=audit,
        subgroup_metrics=subgroups,
        frozen_policy=frozen_policy,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_database_arg(parser)
    parser.add_argument("--pre-release-panel", type=Path, default=None, help="Defaults to the active model pre_release_panel.csv")
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--wide-theater-threshold", type=int, default=600)
    parser.add_argument("--min-train-n", type=int, default=DEFAULT_MIN_TRAIN_N)
    parser.add_argument("--min-promote-n", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    panel_path = args.pre_release_panel or active_pre_release_panel_path()
    outputs = run(
        args.database_url,
        pre_release_panel_path=panel_path,
        start_year=args.start_year,
        wide_theater_threshold=args.wide_theater_threshold,
        min_train_n=args.min_train_n,
        min_promote_n=args.min_promote_n,
        output_dir=args.output_dir,
    )
    for name in [
        "opening_thursday_actual_panel.csv",
        "opening_thursday_actual_year_folds.csv",
        "opening_thursday_actual_metrics_by_origin_baseline.csv",
        "opening_thursday_actual_ow_predictions.csv",
        "opening_thursday_actual_bucket_scores.csv",
        "opening_thursday_actual_promotion_summary.json",
        "opening_thursday_actual_ratio_update_policy.json",
    ]:
        print(f"Wrote {args.output_dir / name}")
    print(json.dumps(outputs.promotion_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
