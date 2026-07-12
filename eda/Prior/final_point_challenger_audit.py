#!/usr/bin/env python3
"""Final bounded challenger audit for pre-release point forecasts.

This intentionally avoids a broad model search.  It compares the frozen daily
point policy with a small set of plausible forecast-combination challengers:

* robust trimmed / winsorized log means;
* shrunk source-composition policies;
* shrunk inverse-error stacking;
* shrunk Bates-Granger covariance weights;
* time-decayed performance weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eda.Prior.fallback_adjusted_daily_policy_validation import (
    DIAGNOSTICS_DIR,
    OUTPUT_DIR,
    forecast_with_fallback,
    production_source_panel,
)
from eda.Prior.rolling_forecast_consensus_benchmark import (
    BenchmarkConfig,
    DEFAULT_INTERVAL_SHRINK_K,
    DEFAULT_MAX_SOURCE_AGE_DAYS,
    DEFAULT_MIN_SOURCE_RELIABILITY_N,
    DEFAULT_MIN_TRAIN_N_FOR_MODEL_SELECTION,
    DEFAULT_ORIGIN_DAYS,
    DEFAULT_RECENCY_LAMBDAS,
    DEFAULT_SOURCE_BIAS_SHRINK_K,
    DEFAULT_TEST_START_YEAR,
    DEFAULT_TRAIN_YEARS,
    add_locked_point_forecasts,
    add_range_features_to_consensus_panel,
    build_consensus_panel,
    evaluate_forecast,
)


SOURCE_PATH = DIAGNOSTICS_DIR / "rolling_forecast_origin_source_estimates.csv"
POINT_POLICY_PATH = OUTPUT_DIR / "frozen_simplified_daily_point_policy.json"
CORE_SOURCES = {"boxofficereport", "boxofficepro"}
TODD_SOURCE = "toddmthatcher"
THEORY_SOURCE = "boxofficetheory"
MIN_TRAIN_SOURCE_ROWS = 8
STACK_SHRINK_K = 40.0
COV_SHRINK = 0.60


def frozen_method_by_origin() -> dict[int, str]:
    policy = json.loads(POINT_POLICY_PATH.read_text(encoding="utf-8"))
    return {int(row["origin_day"]): str(row["simplified_method"]) for row in policy.get("daily_policy", [])}


def config() -> BenchmarkConfig:
    return BenchmarkConfig(
        origin_days=DEFAULT_ORIGIN_DAYS,
        excluded_estimate_sources=(),
        excluded_release_years=(2020, 2021),
        recency_lambdas=DEFAULT_RECENCY_LAMBDAS,
        train_years=DEFAULT_TRAIN_YEARS,
        test_start_year=DEFAULT_TEST_START_YEAR,
        min_source_reliability_n=DEFAULT_MIN_SOURCE_RELIABILITY_N,
        source_bias_shrink_k=DEFAULT_SOURCE_BIAS_SHRINK_K,
        max_source_age_days=DEFAULT_MAX_SOURCE_AGE_DAYS,
        min_train_n_for_model_selection=DEFAULT_MIN_TRAIN_N_FOR_MODEL_SELECTION,
        interval_shrink_k=DEFAULT_INTERVAL_SHRINK_K,
    )


def positive_log(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return np.log(numeric[np.isfinite(numeric) & (numeric > 0)])


def robust_log_mean(group: pd.DataFrame, *, mode: str) -> float:
    logs = positive_log(group["source_bias_adjusted_estimate_mid_usd"])
    if len(logs) == 0:
        return np.nan
    if len(logs) >= 5:
        lo, hi = np.quantile(logs, [0.10, 0.90])
        if mode == "trimmed":
            logs = logs[(logs >= lo) & (logs <= hi)]
        else:
            logs = np.clip(logs, lo, hi)
    return float(np.exp(np.mean(logs))) if len(logs) else np.nan


def source_set_mask(frame: pd.DataFrame, policy: str, origin_day: int) -> pd.Series:
    source = frame["estimate_source"].astype(str)
    if policy == "core_only":
        return source.isin(CORE_SOURCES)
    if policy == "core_plus_todd_when_eligible":
        eligible = set(CORE_SOURCES)
        if -9 <= origin_day <= -3:
            eligible.add(TODD_SOURCE)
        return source.isin(eligible)
    if policy == "core_plus_theory_early":
        eligible = set(CORE_SOURCES)
        if origin_day <= -8:
            eligible.add(THEORY_SOURCE)
        return source.isin(eligible)
    return pd.Series(True, index=frame.index)


def weighted_log_mean(group: pd.DataFrame, weights: dict[str, float]) -> float:
    work = group.copy()
    logs = positive_log(work["source_bias_adjusted_estimate_mid_usd"])
    if len(logs) == 0:
        return np.nan
    work = work.loc[pd.to_numeric(work["source_bias_adjusted_estimate_mid_usd"], errors="coerce").gt(0)].copy()
    source_weights = work["estimate_source"].astype(str).map(weights).fillna(0.0).to_numpy(dtype=float)
    logs = np.log(pd.to_numeric(work["source_bias_adjusted_estimate_mid_usd"], errors="coerce").to_numpy(dtype=float))
    mask = np.isfinite(logs) & np.isfinite(source_weights) & (source_weights > 0)
    if not mask.any():
        return float(np.exp(np.mean(logs[np.isfinite(logs)])))
    return float(np.exp(np.average(logs[mask], weights=source_weights[mask])))


def normalize(weights: dict[str, float], sources: list[str]) -> dict[str, float]:
    vals = np.array([max(float(weights.get(source, 0.0)), 0.0) for source in sources], dtype=float)
    if vals.sum() <= 0:
        vals = np.ones(len(sources), dtype=float)
    vals = vals / vals.sum()
    return dict(zip(sources, vals))


def inverse_error_weights(train: pd.DataFrame, sources: list[str], *, decayed: bool = False) -> dict[str, float]:
    rows = []
    max_year = float(pd.to_numeric(train["release_year"], errors="coerce").max())
    for source in sources:
        g = train.loc[train["estimate_source"].astype(str).eq(source)].copy()
        residual = pd.to_numeric(g["source_bias_adjusted_residual_log"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        residual = residual.dropna()
        if len(residual) < MIN_TRAIN_SOURCE_ROWS:
            rows.append((source, 1.0, len(residual)))
            continue
        if decayed:
            years = pd.to_numeric(g.loc[residual.index, "release_year"], errors="coerce").fillna(max_year).to_numpy(dtype=float)
            weight = np.exp(-0.35 * np.maximum(max_year - years, 0))
            rmse = float(np.sqrt(np.average(residual.to_numpy(dtype=float) ** 2, weights=weight)))
        else:
            rmse = float(np.sqrt(np.mean(residual.to_numpy(dtype=float) ** 2)))
        rows.append((source, 1.0 / max(rmse**2, 1e-6), len(residual)))
    raw = normalize({source: value for source, value, _ in rows}, sources)
    equal = 1.0 / len(sources) if sources else 0.0
    shrunk = {}
    for source, _, n in rows:
        alpha = n / (n + STACK_SHRINK_K)
        shrunk[source] = alpha * raw[source] + (1.0 - alpha) * equal
    return normalize(shrunk, sources)


def covariance_weights(train: pd.DataFrame, sources: list[str]) -> dict[str, float]:
    if len(sources) <= 1:
        return {source: 1.0 for source in sources}
    pivot = (
        train.loc[train["estimate_source"].astype(str).isin(sources)]
        .pivot_table(
            index=["release_run_id", "origin_day"],
            columns="estimate_source",
            values="source_bias_adjusted_residual_log",
            aggfunc="first",
        )
        .reindex(columns=sources)
    )
    common = pivot.dropna()
    if len(common) < max(12, len(sources) + 3):
        return inverse_error_weights(train, sources)
    cov = np.cov(common.to_numpy(dtype=float), rowvar=False)
    diag = np.diag(np.diag(cov))
    sigma = COV_SHRINK * diag + (1.0 - COV_SHRINK) * cov
    sigma += np.eye(len(sources)) * 1e-5
    ones = np.ones(len(sources))
    try:
        raw = np.linalg.solve(sigma, ones)
    except np.linalg.LinAlgError:
        return inverse_error_weights(train, sources)
    raw = np.clip(raw, 0.0, None)
    if raw.sum() <= 0:
        return inverse_error_weights(train, sources)
    raw = raw / raw.sum()
    equal = np.ones(len(sources)) / len(sources)
    raw = 0.70 * raw + 0.30 * equal
    return dict(zip(sources, raw / raw.sum()))


def grouped_forecast(source_group: pd.DataFrame, method: str, train: pd.DataFrame, origin_day: int) -> float:
    if source_group.empty:
        return np.nan
    if method == "trimmed_log_mean":
        return robust_log_mean(source_group, mode="trimmed")
    if method == "winsorized_log_mean":
        return robust_log_mean(source_group, mode="winsorized")
    if method.startswith("composition_"):
        policy = method.replace("composition_", "")
        subset = source_group.loc[source_set_mask(source_group, policy, origin_day)]
        if subset.empty:
            return np.nan
        return float(np.exp(np.median(positive_log(subset["source_bias_adjusted_estimate_mid_usd"]))))
    sources = sorted(source_group["estimate_source"].dropna().astype(str).unique())
    if method == "shrunk_convex_inverse_error":
        weights = inverse_error_weights(train, sources)
        return weighted_log_mean(source_group, weights)
    if method == "shrunk_bates_granger_covariance":
        weights = covariance_weights(train, sources)
        return weighted_log_mean(source_group, weights)
    if method == "time_decayed_performance_weights":
        weights = inverse_error_weights(train, sources, decayed=True)
        return weighted_log_mean(source_group, weights)
    raise ValueError(f"Unknown challenger method {method}")


def score_frame(df: pd.DataFrame, forecast_col: str) -> dict[str, float | int]:
    temp = df.dropna(subset=[forecast_col, "actual_opening_weekend_gross_usd"]).copy()
    forecast = pd.to_numeric(temp[forecast_col], errors="coerce")
    actual = pd.to_numeric(temp["actual_opening_weekend_gross_usd"], errors="coerce")
    mask = forecast.gt(0) & actual.gt(0)
    temp = temp.loc[mask].copy()
    forecast = forecast.loc[mask]
    actual = actual.loc[mask]
    if temp.empty:
        return {"n": 0, "MAE_log": np.nan, "RMSE_log": np.nan, "MdAPE": np.nan, "ME_log": np.nan}
    residual = np.log(actual / forecast)
    return {
        "n": int(len(temp)),
        "MAE_log": float(np.mean(np.abs(residual))),
        "RMSE_log": float(np.sqrt(np.mean(residual**2))),
        "MdAPE": float(np.median(np.abs(actual / forecast - 1.0))),
        "ME_log": float(np.mean(residual)),
    }


def run_audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(SOURCE_PATH, low_memory=False)
    for col in ["opening_weekend_start", "forecast_origin_date", "estimate_date"]:
        if col in source.columns:
            source[col] = pd.to_datetime(source[col], errors="coerce")
    source = production_source_panel(source)
    panel = build_consensus_panel(source, config(), include_extended_candidates=True)
    panel = add_locked_point_forecasts(panel)
    panel = add_range_features_to_consensus_panel(panel, source)
    frozen_methods = frozen_method_by_origin()
    challenger_methods = [
        "trimmed_log_mean",
        "winsorized_log_mean",
        "composition_core_only",
        "composition_core_plus_todd_when_eligible",
        "composition_core_plus_theory_early",
        "shrunk_convex_inverse_error",
        "shrunk_bates_granger_covariance",
        "time_decayed_performance_weights",
    ]
    rows = []
    forecast_rows = []
    years = sorted(int(year) for year in panel["release_year"].dropna().unique() if int(year) not in (2020, 2021))
    for year in years:
        source_train_year = source.loc[pd.to_numeric(source["release_year"], errors="coerce").lt(year)].copy()
        source_holdout_year = source.loc[pd.to_numeric(source["release_year"], errors="coerce").eq(year)].copy()
        for origin_day in sorted(int(value) for value in panel["origin_day"].dropna().unique()):
            holdout = panel.loc[panel["origin_day"].eq(origin_day) & panel["release_year"].eq(year)].copy()
            if holdout.empty:
                continue
            train_source_origin = source_train_year.loc[source_train_year["origin_day"].eq(origin_day)].copy()
            holdout_source_origin = source_holdout_year.loc[source_holdout_year["origin_day"].eq(origin_day)].copy()
            frozen_method = frozen_methods.get(origin_day, "dollar_median_consensus_usd")
            frozen = forecast_with_fallback(holdout, frozen_method, origin_day)
            fold = holdout[
                [
                    "release_run_id",
                    "movie_id",
                    "title",
                    "opening_weekend_start",
                    "release_year",
                    "origin_day",
                    "actual_opening_weekend_gross_usd",
                ]
            ].copy()
            fold["holdout_year"] = year
            fold["frozen_daily_policy"] = frozen.values
            for method in challenger_methods:
                values = []
                for row in fold.itertuples(index=False):
                    group = holdout_source_origin.loc[
                        holdout_source_origin["release_run_id"].eq(row.release_run_id)
                    ].copy()
                    values.append(grouped_forecast(group, method, train_source_origin, origin_day))
                fold[method] = values
                fold[f"{method}_missing_before_fallback"] = pd.to_numeric(fold[method], errors="coerce").isna()
                fold[method] = pd.to_numeric(fold[method], errors="coerce").fillna(fold["frozen_daily_policy"])
            forecast_rows.append(fold)
            for method in ["frozen_daily_policy", *challenger_methods]:
                score = score_frame(fold, method)
                missing_before = (
                    int(fold[f"{method}_missing_before_fallback"].sum())
                    if f"{method}_missing_before_fallback" in fold.columns
                    else 0
                )
                rows.append(
                    {
                        "holdout_year": year,
                        "origin_day": origin_day,
                        "forecast_method": method,
                        **score,
                        "missing_before_fallback": missing_before,
                        "missing_after_fallback": int(pd.to_numeric(fold[method], errors="coerce").isna().sum()),
                    }
                )
    folds = pd.DataFrame(rows)
    forecasts = pd.concat(forecast_rows, ignore_index=True) if forecast_rows else pd.DataFrame()
    summary = (
        folds.groupby(["origin_day", "forecast_method"])
        .agg(
            folds=("holdout_year", "nunique"),
            rows=("n", "sum"),
            mean_MAE_log=("MAE_log", "mean"),
            median_MAE_log=("MAE_log", "median"),
            mean_RMSE_log=("RMSE_log", "mean"),
            mean_MdAPE=("MdAPE", "mean"),
            missing_before_fallback=("missing_before_fallback", "sum"),
            missing_after_fallback=("missing_after_fallback", "sum"),
        )
        .reset_index()
    )
    weighted = []
    for keys, group in folds.groupby(["origin_day", "forecast_method"]):
        weights = pd.to_numeric(group["n"], errors="coerce").fillna(0).to_numpy(dtype=float)
        row = {"origin_day": keys[0], "forecast_method": keys[1]}
        for col in ["MAE_log", "RMSE_log", "MdAPE"]:
            values = pd.to_numeric(group[col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(values) & (weights > 0)
            row[f"weighted_{col}"] = float(np.average(values[mask], weights=weights[mask])) if mask.any() else np.nan
        weighted.append(row)
    summary = summary.merge(pd.DataFrame(weighted), on=["origin_day", "forecast_method"], how="left")
    frozen = summary.loc[summary["forecast_method"].eq("frozen_daily_policy"), ["origin_day", "mean_MAE_log", "mean_RMSE_log"]]
    summary = summary.merge(frozen, on="origin_day", suffixes=("", "_frozen"), how="left")
    summary["delta_MAE_vs_frozen"] = summary["mean_MAE_log"] - summary["mean_MAE_log_frozen"]
    summary["delta_RMSE_vs_frozen"] = summary["mean_RMSE_log"] - summary["mean_RMSE_log_frozen"]
    frozen_weighted = summary.loc[
        summary["forecast_method"].eq("frozen_daily_policy"), ["origin_day", "weighted_MAE_log", "weighted_RMSE_log"]
    ]
    summary = summary.merge(frozen_weighted, on="origin_day", suffixes=("", "_frozen"), how="left")
    summary["weighted_delta_MAE_vs_frozen"] = summary["weighted_MAE_log"] - summary["weighted_MAE_log_frozen"]
    summary["weighted_delta_RMSE_vs_frozen"] = summary["weighted_RMSE_log"] - summary["weighted_RMSE_log_frozen"]
    return folds, summary, forecasts


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    folds, summary, forecasts = run_audit()
    folds.to_csv(OUTPUT_DIR / "final_point_challenger_year_folds.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "final_point_challenger_summary_by_origin.csv", index=False)
    forecasts.to_csv(OUTPUT_DIR / "final_point_challenger_oof_forecasts.csv", index=False)
    winners = (
        summary.sort_values(["origin_day", "mean_MAE_log", "mean_RMSE_log", "forecast_method"])
        .groupby("origin_day")
        .head(1)
        .reset_index(drop=True)
    )
    winners.to_csv(OUTPUT_DIR / "final_point_challenger_winners_by_origin.csv", index=False)
    print(f"Wrote final point challenger audit to {OUTPUT_DIR}")
    print(f"Fold rows: {len(folds):,}")
    print(f"Forecast rows: {len(forecasts):,}")
    print("Winners by origin:")
    print(winners[["origin_day", "forecast_method", "mean_MAE_log", "delta_MAE_vs_frozen"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
