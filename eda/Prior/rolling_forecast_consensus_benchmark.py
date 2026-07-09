#!/usr/bin/env python3
"""Rolling-origin benchmark for late opening-weekend forecast consensus.

This script generalizes ``late_forecast_consensus_benchmark.ipynb`` from a
single Thursday forecast origin to a movie x forecast-origin panel.  For each
origin day from -14 through -1, it keeps the latest estimate per source that
was available by that origin, computes simple log-scale consensus forecasts,
and evaluates recency/source-reliability weighting as candidate benchmarks.

Outputs are written under ``data/diagnostics`` by default:

* ``rolling_forecast_origin_source_estimates.csv``: one row per
  release/origin/source latest eligible estimate.
* ``rolling_forecast_origin_consensus_panel.csv``: one row per release/origin
  with all consensus forecasts.
* ``rolling_forecast_origin_consensus_metrics.csv``: accuracy summaries by
  origin, method, and chronological evaluation subset.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from pm_box_office.db.connection import connect_database
from pm_box_office.sources.common.cli import add_database_arg


REPO_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS_DIR = REPO_ROOT / "data" / "diagnostics"

DEFAULT_ORIGIN_DAYS = tuple(range(-14, 0))
DEFAULT_EXCLUDED_ESTIMATE_SOURCES = ("the_numbers_predictions", "boxofficeguru")
DEFAULT_EXCLUDED_RELEASE_YEARS = (2020, 2021)
DEFAULT_RECENCY_LAMBDAS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00)
DEFAULT_TEST_START_YEAR = 2024
DEFAULT_TRAIN_YEARS = tuple(range(2010, 2020)) + (2022, 2023)
DEFAULT_MIN_SOURCE_RELIABILITY_N = 5
DEFAULT_SOURCE_BIAS_SHRINK_K = 10
DEFAULT_MAX_SOURCE_AGE_DAYS = (3, 7, 14)
DEFAULT_MIN_TRAIN_N_FOR_MODEL_SELECTION = 10
DEFAULT_INTERVAL_SHRINK_K = 20
DEFAULT_INTERVAL_LEVELS = (80, 95)
DEFAULT_POINT_BASELINE_METHOD = "dollar_median_consensus_usd"
DEFAULT_SELECTED_POINT_COL = "selected_point_forecast_usd"
DEFAULT_PRIMARY_POINT_COL = "primary_point_forecast_usd"
DEFAULT_MEAN_POINT_COL = "mean_point_forecast_usd"


@dataclass(frozen=True)
class BenchmarkConfig:
    origin_days: tuple[int, ...]
    excluded_estimate_sources: tuple[str, ...]
    excluded_release_years: tuple[int, ...]
    recency_lambdas: tuple[float, ...]
    train_years: tuple[int, ...]
    test_start_year: int
    min_source_reliability_n: int
    source_bias_shrink_k: int
    max_source_age_days: tuple[int, ...]
    min_train_n_for_model_selection: int
    interval_shrink_k: int


def fetch_frame(conn: Any, sql: str) -> pd.DataFrame:
    cursor = conn.execute(sql)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def is_doc_concert_frame(df: pd.DataFrame) -> pd.Series:
    text = (
        df.get("genre", pd.Series("", index=df.index)).fillna("").astype(str)
        + " "
        + df.get("release_type", pd.Series("", index=df.index)).fillna("").astype(str)
        + " "
        + df.get("title", pd.Series("", index=df.index)).fillna("").astype(str)
    ).str.lower()
    return text.str.contains(r"documentary|concert|music documentary|live concert", regex=True, na=False)


def target_matches_opening(df: pd.DataFrame) -> pd.Series:
    metric = df["forecast_metric"].fillna("").astype(str).str.lower()
    return (
        df["target_start_date"].isna()
        | (df["target_start_date"] == df["opening_weekend_start"])
        | metric.str.contains("opening", na=False)
    )


def safe_log(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return np.log(numeric.where(numeric > 0))


def weighted_log_forecast(group: pd.DataFrame, weight_col: str, log_col: str = "log_estimate_mid") -> float:
    weights = pd.to_numeric(group[weight_col], errors="coerce").to_numpy(dtype="float64")
    logs = pd.to_numeric(group[log_col], errors="coerce").to_numpy(dtype="float64")
    mask = np.isfinite(weights) & np.isfinite(logs) & (weights > 0)
    if not mask.any():
        return float(np.exp(np.nanmean(logs)))
    weights = weights[mask]
    logs = logs[mask]
    return float(np.exp(np.sum(weights * logs) / np.sum(weights)))


def add_age_capped_candidates(
    base: pd.DataFrame,
    source_panel: pd.DataFrame,
    caps: tuple[int, ...],
) -> pd.DataFrame:
    group_cols = ["release_run_id", "origin_day"]
    out = base.copy()

    for cap in caps:
        fresh = source_panel.loc[source_panel["source_age_days"] <= cap].copy()
        strict = (
            fresh.groupby(group_cols)
            .agg(
                **{
                    f"age_cap_{cap}d_strict_dollar_median_consensus_usd": (
                        "estimate_mid_usd",
                        "median",
                    ),
                    f"age_cap_{cap}d_strict_log_mean_consensus_usd": (
                        "log_estimate_mid",
                        lambda s: float(np.exp(s.mean())),
                    ),
                    f"age_cap_{cap}d_strict_log_median_consensus_usd": (
                        "log_estimate_mid",
                        lambda s: float(np.exp(s.median())),
                    ),
                    f"age_cap_{cap}d_strict_source_count": (
                        "estimate_source",
                        "nunique",
                    ),
                }
            )
            .reset_index()
        )
        out = out.merge(strict, on=group_cols, how="left")
        out[f"age_cap_{cap}d_fallback_dollar_median_consensus_usd"] = out[
            f"age_cap_{cap}d_strict_dollar_median_consensus_usd"
        ].fillna(out["dollar_median_consensus_usd"])
        out[f"age_cap_{cap}d_fallback_log_mean_consensus_usd"] = out[
            f"age_cap_{cap}d_strict_log_mean_consensus_usd"
        ].fillna(out["log_mean_consensus_usd"])
        out[f"age_cap_{cap}d_fallback_log_median_consensus_usd"] = out[
            f"age_cap_{cap}d_strict_log_median_consensus_usd"
        ].fillna(out["log_median_consensus_usd"])

    return out


def openings_sql() -> str:
    return """
        SELECT
            release_run_id,
            movie_id,
            title,
            opening_date,
            opening_weekend_start,
            release_year,
            release_month,
            season_bucket,
            market,
            release_type,
            release_width_bucket,
            is_wide_release,
            is_large_release,
            genre,
            distributor,
            franchise,
            is_franchise,
            opening_weekend_theaters,
            opening_weekend_gross_usd
        FROM analytics.eda_movie_openings
        WHERE opening_weekend_gross_usd IS NOT NULL
    """


def estimates_sql() -> str:
    return """
        SELECT
            eda_estimate_id,
            estimate_source,
            source_prediction_id,
            release_run_id,
            movie_id,
            title,
            opening_date,
            opening_weekend_start,
            actual_opening_weekend_gross_usd,
            release_width_bucket,
            genre,
            distributor,
            franchise,
            is_franchise,
            estimate_date,
            target_start_date,
            target_end_date,
            forecast_metric,
            estimate_low_usd,
            estimate_high_usd,
            estimate_mid_usd,
            estimate_width_usd,
            actual_inside_range,
            days_before_opening_weekend
        FROM analytics.eda_news_estimates
        WHERE estimate_mid_usd IS NOT NULL
    """


def load_inputs(database_url: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = connect_database(database_url)
    try:
        openings = fetch_frame(conn, openings_sql())
        estimates = fetch_frame(conn, estimates_sql())
    finally:
        conn.close()

    for df in (openings, estimates):
        for col in [
            "opening_date",
            "opening_weekend_start",
            "estimate_date",
            "target_start_date",
            "target_end_date",
        ]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    return openings, estimates


def build_source_origin_panel(
    openings: pd.DataFrame,
    estimates: pd.DataFrame,
    config: BenchmarkConfig,
) -> pd.DataFrame:
    eligible_openings = openings.loc[
        (pd.to_numeric(openings["opening_weekend_gross_usd"], errors="coerce") > 0)
        & ~openings["release_year"].isin(config.excluded_release_years)
    ].copy()
    eligible_openings["is_doc_concert"] = is_doc_concert_frame(eligible_openings)

    opening_cols = [
        "release_run_id",
        "movie_id",
        "title",
        "opening_weekend_start",
        "release_year",
        "release_month",
        "season_bucket",
        "release_type",
        "release_width_bucket",
        "is_wide_release",
        "is_large_release",
        "genre",
        "distributor",
        "franchise",
        "is_franchise",
        "is_doc_concert",
        "opening_weekend_theaters",
        "opening_weekend_gross_usd",
    ]
    estimate_candidates = estimates.merge(
        eligible_openings[opening_cols],
        on="release_run_id",
        how="inner",
        suffixes=("_estimate", ""),
    )
    estimate_candidates = estimate_candidates.loc[
        ~estimate_candidates["estimate_source"].isin(config.excluded_estimate_sources)
        & (pd.to_numeric(estimate_candidates["estimate_mid_usd"], errors="coerce") > 0)
        & estimate_candidates["estimate_date"].notna()
        & target_matches_opening(estimate_candidates)
    ].copy()
    estimate_candidates["log_estimate_mid"] = safe_log(estimate_candidates["estimate_mid_usd"])
    estimate_candidates = estimate_candidates.dropna(subset=["log_estimate_mid"])

    origins = pd.DataFrame({"origin_day": list(config.origin_days)})
    movie_origins = eligible_openings[opening_cols].merge(origins, how="cross")
    movie_origins["forecast_origin_date"] = movie_origins["opening_weekend_start"] + pd.to_timedelta(
        movie_origins["origin_day"], unit="D"
    )

    available = movie_origins.merge(
        estimate_candidates[
            [
                "eda_estimate_id",
                "estimate_source",
                "source_prediction_id",
                "release_run_id",
                "estimate_date",
                "target_start_date",
                "target_end_date",
                "forecast_metric",
                "estimate_low_usd",
                "estimate_high_usd",
                "estimate_mid_usd",
                "estimate_width_usd",
                "actual_inside_range",
                "days_before_opening_weekend",
                "log_estimate_mid",
            ]
        ],
        on="release_run_id",
        how="inner",
    )
    available = available.loc[available["estimate_date"] <= available["forecast_origin_date"]].copy()
    available["estimate_lead_day"] = (
        available["estimate_date"] - available["opening_weekend_start"]
    ).dt.days
    available["source_age_days"] = available["origin_day"] - available["estimate_lead_day"]

    latest = (
        available.sort_values(
            ["release_run_id", "origin_day", "estimate_source", "estimate_date", "eda_estimate_id"],
            ascending=[True, True, True, False, False],
        )
        .drop_duplicates(["release_run_id", "origin_day", "estimate_source"], keep="first")
        .copy()
    )
    latest["log_actual_opening_weekend"] = safe_log(latest["opening_weekend_gross_usd"])
    latest["source_residual_log"] = latest["log_actual_opening_weekend"] - latest["log_estimate_mid"]
    return latest


def add_source_reliability(source_panel: pd.DataFrame, min_n: int) -> pd.DataFrame:
    out = source_panel.sort_values(
        ["origin_day", "estimate_source", "opening_weekend_start", "release_run_id"]
    ).copy()
    squared_error = out["source_residual_log"] ** 2
    grouped = out.assign(source_squared_error=squared_error).groupby(["origin_day", "estimate_source"], sort=False)
    prior_count = grouped.cumcount()
    prior_sse = grouped["source_squared_error"].cumsum() - squared_error
    out["source_prior_n"] = prior_count
    out["source_prior_rmse_log"] = np.sqrt(prior_sse / prior_count.replace(0, np.nan))

    valid = (
        out["source_prior_n"].ge(min_n)
        & out["source_prior_rmse_log"].notna()
        & out["source_prior_rmse_log"].gt(0)
    )
    out["source_reliability_raw_weight"] = np.where(valid, 1.0 / (out["source_prior_rmse_log"] ** 2), np.nan)
    return out


def add_source_bias_calibration(source_panel: pd.DataFrame, min_n: int, shrink_k: int) -> pd.DataFrame:
    out = source_panel.sort_values(
        ["origin_day", "estimate_source", "opening_weekend_start", "release_run_id"]
    ).copy()

    grouped = out.groupby(["origin_day", "estimate_source"], sort=False)
    prior_n = grouped.cumcount()
    prior_sum = grouped["source_residual_log"].cumsum() - out["source_residual_log"]

    out["source_prior_bias_n"] = prior_n
    out["source_prior_bias_log"] = prior_sum / prior_n.replace(0, np.nan)

    shrink = prior_n / (prior_n + shrink_k)
    out["source_bias_shrunk_log"] = np.where(
        prior_n.ge(min_n) & out["source_prior_bias_log"].notna(),
        shrink * out["source_prior_bias_log"],
        0.0,
    )

    out["log_estimate_mid_source_bias_adj"] = out["log_estimate_mid"] + out["source_bias_shrunk_log"]
    out["source_bias_adjusted_estimate_mid_usd"] = np.exp(out["log_estimate_mid_source_bias_adj"])
    out["source_bias_adjusted_residual_log"] = (
        out["log_actual_opening_weekend"] - out["log_estimate_mid_source_bias_adj"]
    )
    return out


def add_source_bias_adjusted_reliability(source_panel: pd.DataFrame, min_n: int) -> pd.DataFrame:
    out = source_panel.sort_values(
        ["origin_day", "estimate_source", "opening_weekend_start", "release_run_id"]
    ).copy()
    squared_error = out["source_bias_adjusted_residual_log"] ** 2
    grouped = out.assign(source_bias_adjusted_squared_error=squared_error).groupby(
        ["origin_day", "estimate_source"], sort=False
    )
    prior_count = grouped.cumcount()
    prior_sse = grouped["source_bias_adjusted_squared_error"].cumsum() - squared_error
    out["source_bias_adjusted_prior_n"] = prior_count
    out["source_bias_adjusted_prior_rmse_log"] = np.sqrt(prior_sse / prior_count.replace(0, np.nan))
    valid = (
        out["source_bias_adjusted_prior_n"].ge(min_n)
        & out["source_bias_adjusted_prior_rmse_log"].notna()
        & out["source_bias_adjusted_prior_rmse_log"].gt(0)
    )
    out["source_bias_adjusted_reliability_raw_weight"] = np.where(
        valid, 1.0 / (out["source_bias_adjusted_prior_rmse_log"] ** 2), np.nan
    )
    return out


def build_consensus_panel(source_panel: pd.DataFrame, config: BenchmarkConfig) -> pd.DataFrame:
    if "source_reliability_raw_weight" not in source_panel.columns:
        source_panel = add_source_reliability(source_panel, config.min_source_reliability_n)
    if "log_estimate_mid_source_bias_adj" not in source_panel.columns:
        source_panel = add_source_bias_calibration(
            source_panel,
            config.min_source_reliability_n,
            config.source_bias_shrink_k,
        )
    if "source_bias_adjusted_reliability_raw_weight" not in source_panel.columns:
        source_panel = add_source_bias_adjusted_reliability(source_panel, config.min_source_reliability_n)

    group_cols = ["release_run_id", "origin_day"]
    base = (
        source_panel.groupby(group_cols)
        .agg(
            movie_id=("movie_id", "first"),
            title=("title", "first"),
            opening_weekend_start=("opening_weekend_start", "first"),
            forecast_origin_date=("forecast_origin_date", "first"),
            release_year=("release_year", "first"),
            release_month=("release_month", "first"),
            season_bucket=("season_bucket", "first"),
            release_type=("release_type", "first"),
            release_width_bucket=("release_width_bucket", "first"),
            is_wide_release=("is_wide_release", "first"),
            is_large_release=("is_large_release", "first"),
            genre=("genre", "first"),
            distributor=("distributor", "first"),
            franchise=("franchise", "first"),
            is_franchise=("is_franchise", "first"),
            is_doc_concert=("is_doc_concert", "first"),
            opening_weekend_theaters=("opening_weekend_theaters", "first"),
            actual_opening_weekend_gross_usd=("opening_weekend_gross_usd", "first"),
            latest_estimate_date=("estimate_date", "max"),
            earliest_estimate_date=("estimate_date", "min"),
            source_count=("estimate_source", "nunique"),
            estimate_count=("eda_estimate_id", "count"),
            dollar_median_consensus_usd=("estimate_mid_usd", "median"),
            mean_estimate_mid_usd=("estimate_mid_usd", "mean"),
            min_estimate_mid_usd=("estimate_mid_usd", "min"),
            max_estimate_mid_usd=("estimate_mid_usd", "max"),
            estimate_mid_stddev_usd=("estimate_mid_usd", "std"),
            log_mean_consensus_usd=("log_estimate_mid", lambda s: float(np.exp(s.mean()))),
            log_median_consensus_usd=("log_estimate_mid", lambda s: float(np.exp(s.median()))),
            source_bias_adjusted_log_mean_consensus_usd=(
                "log_estimate_mid_source_bias_adj",
                lambda s: float(np.exp(s.mean())),
            ),
            source_bias_adjusted_log_median_consensus_usd=(
                "log_estimate_mid_source_bias_adj",
                lambda s: float(np.exp(s.median())),
            ),
            mean_source_age_days=("source_age_days", "mean"),
            min_source_age_days=("source_age_days", "min"),
            max_source_age_days=("source_age_days", "max"),
            source_range_inside_rate=("actual_inside_range", "mean"),
            reliability_weighted_sources=("source_reliability_raw_weight", lambda s: int(s.notna().sum())),
            source_bias_adjusted_reliability_weighted_sources=(
                "source_bias_adjusted_reliability_raw_weight",
                lambda s: int(s.notna().sum()),
            ),
        )
        .reset_index()
    )
    for col in [
        "actual_opening_weekend_gross_usd",
        "dollar_median_consensus_usd",
        "mean_estimate_mid_usd",
        "min_estimate_mid_usd",
        "max_estimate_mid_usd",
        "estimate_mid_stddev_usd",
        "log_mean_consensus_usd",
        "log_median_consensus_usd",
        "source_bias_adjusted_log_mean_consensus_usd",
        "source_bias_adjusted_log_median_consensus_usd",
    ]:
        base[col] = pd.to_numeric(base[col], errors="coerce")
    base["log_dispersion"] = np.log(base["max_estimate_mid_usd"]) - np.log(base["min_estimate_mid_usd"])

    source_lists = (
        source_panel.groupby(group_cols)["estimate_source"]
        .apply(lambda s: ", ".join(sorted(s.dropna().unique())))
        .rename("estimate_sources")
        .reset_index()
    )
    base = base.merge(source_lists, on=group_cols, how="left")

    for recency_lambda in config.recency_lambdas:
        weight_col = f"recency_weight_lambda_{recency_lambda:g}"
        forecast_col = f"recency_weighted_log_consensus_lambda_{recency_lambda:g}_usd"
        raw_recency_reliability_weight_col = f"recency_reliability_weight_lambda_{recency_lambda:g}"
        raw_recency_reliability_forecast_col = (
            f"recency_reliability_weighted_log_consensus_lambda_{recency_lambda:g}_usd"
        )
        adjusted_forecast_col = (
            f"source_bias_adjusted_recency_weighted_log_consensus_lambda_{recency_lambda:g}_usd"
        )
        adjusted_recency_reliability_weight_col = (
            f"source_bias_adjusted_recency_reliability_weight_lambda_{recency_lambda:g}"
        )
        adjusted_recency_reliability_forecast_col = (
            f"source_bias_adjusted_recency_reliability_weighted_lambda_{recency_lambda:g}_usd"
        )
        source_panel[weight_col] = np.exp(-recency_lambda * source_panel["source_age_days"])
        source_panel[raw_recency_reliability_weight_col] = (
            source_panel[weight_col] * source_panel["source_reliability_raw_weight"].fillna(1.0)
        )
        source_panel[adjusted_recency_reliability_weight_col] = (
            source_panel[weight_col] * source_panel["source_bias_adjusted_reliability_raw_weight"].fillna(1.0)
        )
        weighted = (
            source_panel.groupby(group_cols)
            .apply(lambda g, col=weight_col: weighted_log_forecast(g, col))
            .rename(forecast_col)
            .reset_index()
        )
        base = base.merge(weighted, on=group_cols, how="left")
        raw_recency_reliability_weighted = (
            source_panel.groupby(group_cols)
            .apply(lambda g, col=raw_recency_reliability_weight_col: weighted_log_forecast(g, col))
            .rename(raw_recency_reliability_forecast_col)
            .reset_index()
        )
        base = base.merge(raw_recency_reliability_weighted, on=group_cols, how="left")
        adjusted_weighted = (
            source_panel.groupby(group_cols)
            .apply(
                lambda g, col=weight_col: weighted_log_forecast(
                    g,
                    col,
                    log_col="log_estimate_mid_source_bias_adj",
                )
            )
            .rename(adjusted_forecast_col)
            .reset_index()
        )
        base = base.merge(adjusted_weighted, on=group_cols, how="left")
        adjusted_recency_reliability_weighted = (
            source_panel.groupby(group_cols)
            .apply(
                lambda g, col=adjusted_recency_reliability_weight_col: weighted_log_forecast(
                    g,
                    col,
                    log_col="log_estimate_mid_source_bias_adj",
                )
            )
            .rename(adjusted_recency_reliability_forecast_col)
            .reset_index()
        )
        base = base.merge(adjusted_recency_reliability_weighted, on=group_cols, how="left")

    reliability = (
        source_panel.groupby(group_cols)
        .apply(
            lambda g: weighted_log_forecast(g, "source_reliability_raw_weight"),
        )
        .rename("source_reliability_weighted_log_consensus_usd")
        .reset_index()
    )
    base = base.merge(reliability, on=group_cols, how="left")
    adjusted_reliability = (
        source_panel.groupby(group_cols)
        .apply(
            lambda g: weighted_log_forecast(
                g,
                "source_bias_adjusted_reliability_raw_weight",
                log_col="log_estimate_mid_source_bias_adj",
            ),
        )
        .rename("source_bias_adjusted_reliability_weighted_log_consensus_usd")
        .reset_index()
    )
    base = base.merge(adjusted_reliability, on=group_cols, how="left")
    base = add_age_capped_candidates(base, source_panel, config.max_source_age_days)
    for col in forecast_columns(base):
        base[col] = pd.to_numeric(base[col], errors="coerce")

    base["log_actual_opening_weekend"] = safe_log(base["actual_opening_weekend_gross_usd"])
    base["evaluation_subset"] = np.select(
        [base["release_year"].isin(config.train_years), base["release_year"].ge(config.test_start_year)],
        ["train", "test"],
        default="holdout_unused",
    )
    return base


def forecast_columns(panel: pd.DataFrame) -> list[str]:
    suffix = "_usd"
    excluded = {
        "actual_opening_weekend_gross_usd",
        "mean_estimate_mid_usd",
        "min_estimate_mid_usd",
        "max_estimate_mid_usd",
        "estimate_mid_stddev_usd",
    }
    return [
        col
        for col in panel.columns
        if col.endswith(suffix)
        and col not in excluded
        and (
            col.endswith("_consensus_usd")
            or col.startswith("recency_weighted_log_consensus")
            or col.startswith("recency_reliability_weighted")
            or col.startswith("source_reliability_weighted_log_consensus")
            or col.startswith("source_bias_adjusted")
            or col.startswith("age_cap_")
        )
    ]


def evaluate_forecast(df: pd.DataFrame, forecast_col: str) -> dict[str, float | int | str]:
    x = df.dropna(subset=[forecast_col, "actual_opening_weekend_gross_usd"]).copy()
    x[forecast_col] = pd.to_numeric(x[forecast_col], errors="coerce")
    x["actual_opening_weekend_gross_usd"] = pd.to_numeric(
        x["actual_opening_weekend_gross_usd"], errors="coerce"
    )
    x = x.loc[(x[forecast_col] > 0) & (x["actual_opening_weekend_gross_usd"] > 0)]
    if x.empty:
        return {
            "forecast_method": forecast_col,
            "n": 0,
            "ME_log": np.nan,
            "MAE_log": np.nan,
            "RMSE_log": np.nan,
            "MdAPE": np.nan,
            "underprediction_rate": np.nan,
        }
    residual = np.log(x["actual_opening_weekend_gross_usd"]) - np.log(x[forecast_col])
    return {
        "forecast_method": forecast_col,
        "n": int(len(x)),
        "ME_log": float(residual.mean()),
        "MAE_log": float(residual.abs().mean()),
        "RMSE_log": float(np.sqrt(np.mean(residual**2))),
        "MdAPE": float(np.median(np.abs(x["actual_opening_weekend_gross_usd"] / x[forecast_col] - 1))),
        "underprediction_rate": float((residual > 0).mean()),
    }


def build_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    methods = forecast_columns(panel)
    subsets: list[tuple[str, pd.DataFrame]] = [("all", panel)]
    subsets.extend((name, frame) for name, frame in panel.groupby("evaluation_subset", dropna=False))

    for subset_name, subset in subsets:
        for origin_day, origin_df in subset.groupby("origin_day"):
            for method in methods:
                row = evaluate_forecast(origin_df, method)
                row["evaluation_subset"] = str(subset_name)
                row["origin_day"] = int(origin_day)
                rows.append(row)
        for method in methods:
            row = evaluate_forecast(subset, method)
            row["evaluation_subset"] = str(subset_name)
            row["origin_day"] = "pooled"
            rows.append(row)
    return pd.DataFrame(rows)


def build_train_selected_metrics(panel: pd.DataFrame, criterion: str = "MAE_log") -> pd.DataFrame:
    methods = forecast_columns(panel)
    rows: list[dict[str, float | int | str]] = []

    for origin_day in sorted(panel["origin_day"].dropna().unique()):
        origin_panel = panel.loc[panel["origin_day"] == origin_day]
        train = origin_panel.loc[origin_panel["evaluation_subset"] == "train"]
        test = origin_panel.loc[origin_panel["evaluation_subset"] == "test"]

        train_scores = pd.DataFrame(evaluate_forecast(train, method) for method in methods)
        train_scores = train_scores.loc[train_scores["n"] > 0].dropna(subset=[criterion])
        if train_scores.empty:
            continue

        selected = train_scores.sort_values([criterion, "RMSE_log", "forecast_method"]).iloc[0]
        selected_method = str(selected["forecast_method"])
        test_score = evaluate_forecast(test, selected_method)
        test_score["origin_day"] = int(origin_day)
        test_score["evaluation_subset"] = "test_train_selected"
        test_score["selected_by"] = criterion
        test_score["train_n"] = int(selected["n"])
        test_score[f"train_{criterion}"] = float(selected[criterion])
        rows.append(test_score)

    return pd.DataFrame(rows)


def add_train_selected_forecast(
    panel: pd.DataFrame,
    train_selected_metrics: pd.DataFrame,
    selected_col: str = "train_selected_forecast_usd",
) -> pd.DataFrame:
    out = panel.copy()
    out[selected_col] = np.nan
    selected_by_origin = train_selected_metrics.set_index("origin_day")["forecast_method"].to_dict()
    out["train_selected_method"] = out["origin_day"].map(selected_by_origin)
    for origin_day, method in selected_by_origin.items():
        mask = out["origin_day"].eq(origin_day)
        out.loc[mask, selected_col] = pd.to_numeric(out.loc[mask, method], errors="coerce")
    return out


def point_candidate_shortlist(panel: pd.DataFrame) -> list[str]:
    methods = forecast_columns(panel)
    return [
        method
        for method in methods
        if method == "dollar_median_consensus_usd"
        or method == "log_median_consensus_usd"
        or method == "log_mean_consensus_usd"
        or method.startswith("recency_weighted_log_consensus")
        or method.startswith("age_cap_")
        or method == "source_bias_adjusted_log_mean_consensus_usd"
        or method.startswith("source_bias_adjusted_recency_weighted_log_consensus")
    ]


def fallback_point_method_for_origin(origin_day: int, available_methods: set[str]) -> str:
    if origin_day <= -3:
        return DEFAULT_POINT_BASELINE_METHOD

    if origin_day == -2:
        preferred = "age_cap_7d_fallback_dollar_median_consensus_usd"
        return preferred if preferred in available_methods else DEFAULT_POINT_BASELINE_METHOD

    if origin_day == -1:
        preferred_order = [
            "age_cap_7d_fallback_log_mean_consensus_usd",
            "recency_weighted_log_consensus_lambda_1_usd",
            "recency_weighted_log_consensus_lambda_0.75_usd",
            "recency_weighted_log_consensus_lambda_0.5_usd",
            DEFAULT_POINT_BASELINE_METHOD,
        ]
        for method in preferred_order:
            if method in available_methods:
                return method

    return DEFAULT_POINT_BASELINE_METHOD


def build_point_model_policy(
    panel: pd.DataFrame,
    min_train_n: int,
    baseline_method: str = DEFAULT_POINT_BASELINE_METHOD,
    criterion: str = "MAE_log",
    mdape_guardrail: float = 0.05,
) -> pd.DataFrame:
    methods = point_candidate_shortlist(panel)
    available_methods = set(methods)
    rows: list[dict[str, float | int | str]] = []

    for origin_day_value in sorted(panel["origin_day"].dropna().unique()):
        origin_day = int(origin_day_value)
        origin_panel = panel.loc[panel["origin_day"] == origin_day]
        train = origin_panel.loc[origin_panel["evaluation_subset"] == "train"]
        fallback_method = fallback_point_method_for_origin(origin_day, available_methods)

        baseline_train = evaluate_forecast(train, baseline_method)
        train_scores = pd.DataFrame(evaluate_forecast(train, method) for method in methods)
        train_scores = train_scores.loc[train_scores["n"] > 0].dropna(subset=[criterion])

        if not train_scores.empty and baseline_train["n"] and baseline_train["n"] > 0:
            min_n = 0.95 * float(baseline_train["n"])
            train_scores = train_scores.loc[train_scores["n"] >= min_n]

        if len(train_scores) == 0 or train_scores["n"].max() < min_train_n:
            selected_method = fallback_method
            selection_reason = "fallback_insufficient_train_support"
            selected_train = evaluate_forecast(train, selected_method)
        else:
            baseline_mdape = baseline_train.get("MdAPE", np.nan)
            if np.isfinite(baseline_mdape):
                train_scores = train_scores.loc[
                    train_scores["MdAPE"] <= float(baseline_mdape) + mdape_guardrail
                ]
            if train_scores.empty:
                selected_method = fallback_method
                selection_reason = "fallback_guardrail_failed"
                selected_train = evaluate_forecast(train, selected_method)
            else:
                selected = train_scores.sort_values([criterion, "RMSE_log", "forecast_method"]).iloc[0]
                selected_method = str(selected["forecast_method"])
                selection_reason = "train_selected_with_guardrails"
                selected_train = selected.to_dict()

        rows.append(
            {
                "origin_day": origin_day,
                "selected_point_method": selected_method,
                "selection_reason": selection_reason,
                "train_n": int(selected_train.get("n", 0) or 0),
                "train_ME_log": selected_train.get("ME_log", np.nan),
                "train_MAE_log": selected_train.get("MAE_log", np.nan),
                "train_RMSE_log": selected_train.get("RMSE_log", np.nan),
                "train_MdAPE": selected_train.get("MdAPE", np.nan),
                "fallback_method": fallback_method,
            }
        )

    return pd.DataFrame(rows)


def add_selected_point_forecast(
    panel: pd.DataFrame,
    point_policy: pd.DataFrame,
    selected_col: str = DEFAULT_SELECTED_POINT_COL,
) -> pd.DataFrame:
    out = panel.copy()
    out[selected_col] = np.nan
    selected_by_origin = point_policy.set_index("origin_day")["selected_point_method"].to_dict()
    out["selected_point_method"] = out["origin_day"].map(selected_by_origin)

    for origin_day, method in selected_by_origin.items():
        mask = out["origin_day"].eq(origin_day)
        out.loc[mask, selected_col] = pd.to_numeric(out.loc[mask, method], errors="coerce")

    return out


def build_selected_point_metrics(
    panel: pd.DataFrame,
    selected_col: str = DEFAULT_SELECTED_POINT_COL,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    subsets: list[tuple[str, pd.DataFrame]] = [("all", panel)]
    subsets.extend((name, frame) for name, frame in panel.groupby("evaluation_subset", dropna=False))

    for subset_name, subset in subsets:
        for origin_day, origin_df in subset.groupby("origin_day"):
            row = evaluate_forecast(origin_df, selected_col)
            row["evaluation_subset"] = str(subset_name)
            row["origin_day"] = int(origin_day)
            rows.append(row)

        row = evaluate_forecast(subset, selected_col)
        row["evaluation_subset"] = str(subset_name)
        row["origin_day"] = "pooled"
        rows.append(row)

    return pd.DataFrame(rows)


def locked_primary_point_method_for_origin(origin_day: int) -> str:
    if origin_day <= -2:
        return DEFAULT_POINT_BASELINE_METHOD
    if origin_day == -1:
        return "recency_weighted_log_consensus_lambda_1_usd"
    return DEFAULT_POINT_BASELINE_METHOD


def locked_mean_point_method_for_origin(origin_day: int) -> str:
    if origin_day in (-2, -1):
        return "source_bias_adjusted_log_mean_consensus_usd"
    return "log_mean_consensus_usd"


def add_locked_point_forecasts(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["primary_point_method"] = out["origin_day"].apply(
        lambda x: locked_primary_point_method_for_origin(int(x))
    )
    out["mean_point_method"] = out["origin_day"].apply(
        lambda x: locked_mean_point_method_for_origin(int(x))
    )
    out[DEFAULT_PRIMARY_POINT_COL] = np.nan
    out[DEFAULT_MEAN_POINT_COL] = np.nan

    for method in out["primary_point_method"].dropna().unique():
        mask = out["primary_point_method"].eq(method)
        out.loc[mask, DEFAULT_PRIMARY_POINT_COL] = pd.to_numeric(out.loc[mask, method], errors="coerce")

    for method in out["mean_point_method"].dropna().unique():
        mask = out["mean_point_method"].eq(method)
        out.loc[mask, DEFAULT_MEAN_POINT_COL] = pd.to_numeric(out.loc[mask, method], errors="coerce")

    return out


def build_paired_deltas(
    panel: pd.DataFrame,
    baseline_method: str = "dollar_median_consensus_usd",
) -> pd.DataFrame:
    methods = [method for method in forecast_columns(panel) if method != baseline_method]
    rows: list[dict[str, float | int | str]] = []
    subsets: list[tuple[str, pd.DataFrame]] = [("all", panel)]
    subsets.extend((name, frame) for name, frame in panel.groupby("evaluation_subset", dropna=False))

    for subset_name, subset in subsets:
        for origin_day, origin_df in subset.groupby("origin_day"):
            rows.extend(
                paired_delta_rows_for_frame(origin_df, methods, baseline_method, str(subset_name), int(origin_day))
            )
        rows.extend(paired_delta_rows_for_frame(subset, methods, baseline_method, str(subset_name), "pooled"))
    return pd.DataFrame(rows)


def paired_delta_rows_for_frame(
    df: pd.DataFrame,
    methods: list[str],
    baseline_method: str,
    evaluation_subset: str,
    origin_day: int | str,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    actual = pd.to_numeric(df["actual_opening_weekend_gross_usd"], errors="coerce")
    baseline = pd.to_numeric(df[baseline_method], errors="coerce")
    baseline_error = np.log(actual) - np.log(baseline)
    for method in methods:
        candidate = pd.to_numeric(df[method], errors="coerce")
        candidate_error = np.log(actual) - np.log(candidate)
        valid = (
            actual.gt(0)
            & baseline.gt(0)
            & candidate.gt(0)
            & np.isfinite(baseline_error)
            & np.isfinite(candidate_error)
        )
        if not valid.any():
            rows.append(
                {
                    "evaluation_subset": evaluation_subset,
                    "origin_day": origin_day,
                    "candidate_method": method,
                    "baseline_method": baseline_method,
                    "n": 0,
                    "mean_delta_abs_log_error": np.nan,
                    "median_delta_abs_log_error": np.nan,
                    "candidate_win_rate": np.nan,
                    "mean_delta_squared_log_error": np.nan,
                }
            )
            continue
        delta_abs = candidate_error[valid].abs() - baseline_error[valid].abs()
        delta_squared = candidate_error[valid] ** 2 - baseline_error[valid] ** 2
        rows.append(
            {
                "evaluation_subset": evaluation_subset,
                "origin_day": origin_day,
                "candidate_method": method,
                "baseline_method": baseline_method,
                "n": int(valid.sum()),
                "mean_delta_abs_log_error": float(delta_abs.mean()),
                "median_delta_abs_log_error": float(delta_abs.median()),
                "candidate_win_rate": float((delta_abs < 0).mean()),
                "mean_delta_squared_log_error": float(delta_squared.mean()),
            }
        )
    return rows


def add_grouped_log_residual_intervals(
    panel: pd.DataFrame,
    forecast_col: str,
    shrink_k: int = DEFAULT_INTERVAL_SHRINK_K,
) -> pd.DataFrame:
    out = panel.copy()
    out[forecast_col] = pd.to_numeric(out[forecast_col], errors="coerce")
    out["actual_opening_weekend_gross_usd"] = pd.to_numeric(
        out["actual_opening_weekend_gross_usd"], errors="coerce"
    )
    log_forecast = safe_log(out[forecast_col])
    residual_col = f"{forecast_col}_residual_log"
    out[residual_col] = out["log_actual_opening_weekend"] - log_forecast

    train = out.loc[out["evaluation_subset"] == "train"].copy()
    global_sigma = train[residual_col].std(ddof=1)

    out["sigma_global"] = global_sigma

    origin_sigma = train.groupby("origin_day")[residual_col].std(ddof=1)
    out["sigma_origin"] = out["origin_day"].map(origin_sigma).fillna(global_sigma)

    grouped_specs = {
        "origin_source_count_sigma_shrunk": ["origin_day", "source_count"],
        "origin_franchise_sigma_shrunk": ["origin_day", "is_franchise"],
        "origin_source_count_franchise_sigma_shrunk": [
            "origin_day",
            "source_count",
            "is_franchise",
        ],
    }

    for interval_model, group_cols in grouped_specs.items():
        stats = (
            train.groupby(group_cols)[residual_col]
            .agg(cell_n="count", cell_sigma="std")
            .reset_index()
        )
        temp = out[group_cols].merge(stats, on=group_cols, how="left")
        shrink = temp["cell_n"] / (temp["cell_n"] + shrink_k)
        sigma = (
            shrink * temp["cell_sigma"] + (1.0 - shrink) * out["sigma_origin"]
        ).fillna(out["sigma_origin"])
        out[f"sigma_{interval_model}"] = sigma

    interval_sigma_cols = {
        "global_sigma": "sigma_global",
        "origin_sigma": "sigma_origin",
        "origin_source_count_sigma_shrunk": "sigma_origin_source_count_sigma_shrunk",
        "origin_franchise_sigma_shrunk": "sigma_origin_franchise_sigma_shrunk",
        "origin_source_count_franchise_sigma_shrunk": "sigma_origin_source_count_franchise_sigma_shrunk",
    }

    for interval_model, sigma_col in interval_sigma_cols.items():
        for level, z in [(80, 1.28155), (95, 1.95996)]:
            sigma = out[sigma_col]
            out[f"{forecast_col}_{interval_model}_lo_{level}"] = np.exp(log_forecast - z * sigma)
            out[f"{forecast_col}_{interval_model}_hi_{level}"] = np.exp(log_forecast + z * sigma)

    return out


def add_log_residual_intervals(
    panel: pd.DataFrame,
    forecast_col: str,
    shrink_k: int = DEFAULT_INTERVAL_SHRINK_K,
) -> pd.DataFrame:
    return add_grouped_log_residual_intervals(panel, forecast_col, shrink_k=shrink_k)


def add_empirical_quantile_intervals(
    panel: pd.DataFrame,
    forecast_col: str,
    group_cols: list[str],
    interval_model: str,
    shrink_k: int = DEFAULT_INTERVAL_SHRINK_K,
) -> pd.DataFrame:
    out = panel.copy()
    residual_col = f"{forecast_col}_residual_log"

    if residual_col not in out.columns:
        log_forecast = safe_log(out[forecast_col])
        out[residual_col] = out["log_actual_opening_weekend"] - log_forecast

    train = out.loc[out["evaluation_subset"] == "train"].copy()
    log_forecast = safe_log(out[forecast_col])
    levels = {80: (0.10, 0.90), 95: (0.025, 0.975)}

    for level, (lo_q, hi_q) in levels.items():
        global_lo = train[residual_col].quantile(lo_q)
        global_hi = train[residual_col].quantile(hi_q)

        if group_cols:
            q = (
                train.groupby(group_cols)[residual_col]
                .agg(
                    n="count",
                    lo=lambda s, q=lo_q: s.quantile(q),
                    hi=lambda s, q=hi_q: s.quantile(q),
                )
                .reset_index()
            )
            q["lo_shrunk"] = (
                q["n"] / (q["n"] + shrink_k)
            ) * q["lo"] + (
                shrink_k / (q["n"] + shrink_k)
            ) * global_lo
            q["hi_shrunk"] = (
                q["n"] / (q["n"] + shrink_k)
            ) * q["hi"] + (
                shrink_k / (q["n"] + shrink_k)
            ) * global_hi

            temp = out[group_cols].merge(
                q[group_cols + ["lo_shrunk", "hi_shrunk"]],
                on=group_cols,
                how="left",
            )
            lo = temp["lo_shrunk"].fillna(global_lo)
            hi = temp["hi_shrunk"].fillna(global_hi)
        else:
            lo = pd.Series(global_lo, index=out.index)
            hi = pd.Series(global_hi, index=out.index)

        out[f"{forecast_col}_{interval_model}_lo_{level}"] = np.exp(log_forecast + lo)
        out[f"{forecast_col}_{interval_model}_hi_{level}"] = np.exp(log_forecast + hi)

    return out


def add_selected_intervals(
    interval_panel: pd.DataFrame,
    interval_policy: pd.DataFrame,
    forecast_col: str = DEFAULT_PRIMARY_POINT_COL,
) -> pd.DataFrame:
    out = interval_panel.copy()
    selected_by_origin = interval_policy.set_index("origin_day")["selected_interval_model"].to_dict()
    out["selected_interval_model"] = out["origin_day"].map(selected_by_origin)

    for level in DEFAULT_INTERVAL_LEVELS:
        out[f"selected_lo_{level}"] = np.nan
        out[f"selected_hi_{level}"] = np.nan

    for origin_day, interval_model in selected_by_origin.items():
        mask = out["origin_day"].eq(origin_day)
        for level in DEFAULT_INTERVAL_LEVELS:
            out.loc[mask, f"selected_lo_{level}"] = out.loc[
                mask,
                f"{forecast_col}_{interval_model}_lo_{level}",
            ]
            out.loc[mask, f"selected_hi_{level}"] = out.loc[
                mask,
                f"{forecast_col}_{interval_model}_hi_{level}",
            ]

    return out


def evaluate_selected_intervals(
    df: pd.DataFrame,
    forecast_col: str = DEFAULT_PRIMARY_POINT_COL,
) -> dict[str, float | int | str]:
    cols = [
        "actual_opening_weekend_gross_usd",
        forecast_col,
        "selected_lo_80",
        "selected_hi_80",
        "selected_lo_95",
        "selected_hi_95",
    ]
    x = df.dropna(subset=cols).copy()
    for col in cols:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    x = x.loc[
        (x["actual_opening_weekend_gross_usd"] > 0)
        & (x[forecast_col] > 0)
    ]

    if x.empty:
        return {
            "n": 0,
            "coverage_80": np.nan,
            "coverage_95": np.nan,
            "median_width_80_pct": np.nan,
            "median_width_95_pct": np.nan,
        }

    actual = x["actual_opening_weekend_gross_usd"]
    width_80 = (x["selected_hi_80"] - x["selected_lo_80"]) / x[forecast_col]
    width_95 = (x["selected_hi_95"] - x["selected_lo_95"]) / x[forecast_col]

    return {
        "n": int(len(x)),
        "coverage_80": float(((actual >= x["selected_lo_80"]) & (actual <= x["selected_hi_80"])).mean()),
        "coverage_95": float(((actual >= x["selected_lo_95"]) & (actual <= x["selected_hi_95"])).mean()),
        "median_width_80_pct": float(width_80.median()),
        "median_width_95_pct": float(width_95.median()),
    }


def build_selected_interval_metrics(
    panel: pd.DataFrame,
    forecast_col: str = DEFAULT_PRIMARY_POINT_COL,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    subsets: list[tuple[str, pd.DataFrame]] = [("all", panel)]
    subsets.extend((name, frame) for name, frame in panel.groupby("evaluation_subset", dropna=False))

    for subset_name, subset in subsets:
        for origin_day, origin_df in subset.groupby("origin_day"):
            row = evaluate_selected_intervals(origin_df, forecast_col=forecast_col)
            row["evaluation_subset"] = str(subset_name)
            row["origin_day"] = int(origin_day)
            rows.append(row)

        row = evaluate_selected_intervals(subset, forecast_col=forecast_col)
        row["evaluation_subset"] = str(subset_name)
        row["origin_day"] = "pooled"
        rows.append(row)

    return pd.DataFrame(rows)


def _old_add_log_residual_intervals_unused(
    panel: pd.DataFrame,
    forecast_col: str,
    shrink_k: int = 10,
) -> pd.DataFrame:
    out = panel.copy()
    out[forecast_col] = pd.to_numeric(out[forecast_col], errors="coerce")
    out["actual_opening_weekend_gross_usd"] = pd.to_numeric(
        out["actual_opening_weekend_gross_usd"], errors="coerce"
    )
    log_forecast = safe_log(out[forecast_col])
    residual_col = f"{forecast_col}_residual_log"
    out[residual_col] = out["log_actual_opening_weekend"] - log_forecast

    train = out.loc[out["evaluation_subset"] == "train"].copy()
    global_sigma = train[residual_col].std(ddof=1)
    origin_sigma = train.groupby("origin_day")[residual_col].std(ddof=1)
    origin_source_stats = (
        train.groupby(["origin_day", "source_count"])[residual_col]
        .agg(["count", "std"])
        .rename(columns={"count": "cell_n", "std": "cell_sigma"})
        .reset_index()
    )
    out = out.merge(origin_source_stats, on=["origin_day", "source_count"], how="left")
    out["sigma_global"] = global_sigma
    out["sigma_origin"] = out["origin_day"].map(origin_sigma).fillna(global_sigma)
    shrink = out["cell_n"] / (out["cell_n"] + shrink_k)
    out["sigma_origin_source_count_shrunk"] = (
        shrink * out["cell_sigma"] + (1.0 - shrink) * out["sigma_origin"]
    ).fillna(out["sigma_origin"])

    for interval_model, sigma_col in [
        ("global_sigma", "sigma_global"),
        ("origin_sigma", "sigma_origin"),
        ("origin_source_count_sigma_shrunk", "sigma_origin_source_count_shrunk"),
    ]:
        for level, z in [(80, 1.28155), (95, 1.95996)]:
            sigma = out[sigma_col]
            out[f"{forecast_col}_{interval_model}_lo_{level}"] = np.exp(log_forecast - z * sigma)
            out[f"{forecast_col}_{interval_model}_hi_{level}"] = np.exp(log_forecast + z * sigma)

    return out.drop(columns=["cell_n", "cell_sigma"])


def evaluate_interval_frame(df: pd.DataFrame, forecast_col: str, interval_model: str) -> dict[str, float | int | str]:
    cols = [
        "actual_opening_weekend_gross_usd",
        forecast_col,
        f"{forecast_col}_{interval_model}_lo_80",
        f"{forecast_col}_{interval_model}_hi_80",
        f"{forecast_col}_{interval_model}_lo_95",
        f"{forecast_col}_{interval_model}_hi_95",
    ]
    x = df.dropna(subset=cols).copy()
    for col in cols:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.loc[(x["actual_opening_weekend_gross_usd"] > 0) & (x[forecast_col] > 0)]
    if x.empty:
        return {
            "forecast_method": forecast_col,
            "interval_model": interval_model,
            "n": 0,
            "coverage_80": np.nan,
            "coverage_95": np.nan,
            "mean_width_80_pct": np.nan,
            "mean_width_95_pct": np.nan,
            "median_width_80_pct": np.nan,
            "median_width_95_pct": np.nan,
        }
    actual = x["actual_opening_weekend_gross_usd"]
    width_80 = (x[f"{forecast_col}_{interval_model}_hi_80"] - x[f"{forecast_col}_{interval_model}_lo_80"]) / x[
        forecast_col
    ]
    width_95 = (x[f"{forecast_col}_{interval_model}_hi_95"] - x[f"{forecast_col}_{interval_model}_lo_95"]) / x[
        forecast_col
    ]
    return {
        "forecast_method": forecast_col,
        "interval_model": interval_model,
        "n": int(len(x)),
        "coverage_80": float(
            ((actual >= x[f"{forecast_col}_{interval_model}_lo_80"]) & (actual <= x[f"{forecast_col}_{interval_model}_hi_80"])).mean()
        ),
        "coverage_95": float(
            ((actual >= x[f"{forecast_col}_{interval_model}_lo_95"]) & (actual <= x[f"{forecast_col}_{interval_model}_hi_95"])).mean()
        ),
        "mean_width_80_pct": float(width_80.mean()),
        "mean_width_95_pct": float(width_95.mean()),
        "median_width_80_pct": float(width_80.median()),
        "median_width_95_pct": float(width_95.median()),
    }


def build_interval_metrics(panel_with_intervals: pd.DataFrame, forecast_col: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    interval_models = [
        "global_sigma",
        "origin_sigma",
        "origin_source_count_sigma_shrunk",
        "origin_franchise_sigma_shrunk",
        "origin_source_count_franchise_sigma_shrunk",
        "empirical_global_quantile",
        "empirical_origin_quantile_shrunk",
        "empirical_origin_franchise_quantile_shrunk",
    ]
    subsets: list[tuple[str, pd.DataFrame]] = [("all", panel_with_intervals)]
    subsets.extend((name, frame) for name, frame in panel_with_intervals.groupby("evaluation_subset", dropna=False))

    for subset_name, subset in subsets:
        for origin_day, origin_df in subset.groupby("origin_day"):
            for interval_model in interval_models:
                row = evaluate_interval_frame(origin_df, forecast_col, interval_model)
                row["evaluation_subset"] = str(subset_name)
                row["origin_day"] = int(origin_day)
                rows.append(row)
        for interval_model in interval_models:
            row = evaluate_interval_frame(subset, forecast_col, interval_model)
            row["evaluation_subset"] = str(subset_name)
            row["origin_day"] = "pooled"
            rows.append(row)
    return pd.DataFrame(rows)


def build_interval_model_policy(
    interval_metrics: pd.DataFrame,
    min_train_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    origin_days = sorted(
        int(origin_day)
        for origin_day in interval_metrics.loc[interval_metrics["origin_day"].ne("pooled"), "origin_day"]
        .dropna()
        .unique()
    )
    train_metrics = interval_metrics.loc[
        interval_metrics["evaluation_subset"].eq("train")
        & interval_metrics["origin_day"].ne("pooled")
    ].copy()

    for origin_day in origin_days:
        group = train_metrics.loc[train_metrics["origin_day"].astype(int).eq(origin_day)]
        g = group.loc[group["n"] >= min_train_n].copy()

        if g.empty:
            selected_model = "global_sigma"
            reason = "fallback_insufficient_train_support"
            selected: dict[str, float | int | str] = {}
        else:
            g["coverage_error"] = (
                (g["coverage_80"] - 0.80).abs()
                + (g["coverage_95"] - 0.95).abs()
            )
            g["interval_score"] = g["coverage_error"] + 0.05 * g["median_width_80_pct"]
            selected = g.sort_values(["interval_score", "median_width_80_pct"]).iloc[0].to_dict()
            selected_model = str(selected["interval_model"])
            reason = "train_selected_interval_score"

        rows.append(
            {
                "origin_day": int(origin_day),
                "selected_interval_model": selected_model,
                "selection_reason": reason,
                "train_n": int(selected.get("n", 0) or 0),
                "train_coverage_80": selected.get("coverage_80", np.nan),
                "train_coverage_95": selected.get("coverage_95", np.nan),
                "train_median_width_80_pct": selected.get("median_width_80_pct", np.nan),
                "train_median_width_95_pct": selected.get("median_width_95_pct", np.nan),
            }
        )

    return pd.DataFrame(rows)


def locked_interval_model_for_origin(origin_day: int) -> str:
    if origin_day <= -3:
        return "global_sigma"
    if origin_day == -2:
        return "origin_source_count_sigma_shrunk"
    if origin_day == -1:
        return "empirical_global_quantile"
    return "global_sigma"


def build_locked_interval_model_policy(interval_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    origin_days = sorted(
        int(origin_day)
        for origin_day in interval_metrics.loc[interval_metrics["origin_day"].ne("pooled"), "origin_day"]
        .dropna()
        .unique()
    )
    train_metrics = interval_metrics.loc[
        interval_metrics["evaluation_subset"].eq("train")
        & interval_metrics["origin_day"].ne("pooled")
    ].copy()

    for origin_day in origin_days:
        selected_model = locked_interval_model_for_origin(origin_day)
        selected_rows = train_metrics.loc[
            train_metrics["origin_day"].astype(int).eq(origin_day)
            & train_metrics["interval_model"].eq(selected_model)
        ]
        selected = selected_rows.iloc[0].to_dict() if not selected_rows.empty else {}
        rows.append(
            {
                "origin_day": origin_day,
                "selected_interval_model": selected_model,
                "selection_reason": "locked_interval_policy",
                "train_n": int(selected.get("n", 0) or 0),
                "train_coverage_80": selected.get("coverage_80", np.nan),
                "train_coverage_95": selected.get("coverage_95", np.nan),
                "train_median_width_80_pct": selected.get("median_width_80_pct", np.nan),
                "train_median_width_95_pct": selected.get("median_width_95_pct", np.nan),
            }
        )

    return pd.DataFrame(rows)


def write_outputs(
    source_panel: pd.DataFrame,
    consensus_panel: pd.DataFrame,
    metrics: pd.DataFrame,
    train_selected_metrics: pd.DataFrame,
    paired_deltas: pd.DataFrame,
    point_policy: pd.DataFrame,
    selected_point_panel: pd.DataFrame,
    selected_point_metrics: pd.DataFrame,
    primary_point_panel: pd.DataFrame,
    primary_point_metrics: pd.DataFrame,
    mean_point_metrics: pd.DataFrame,
    interval_metrics: pd.DataFrame,
    interval_panel: pd.DataFrame,
    interval_policy: pd.DataFrame,
    selected_interval_metrics: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "source": output_dir / "rolling_forecast_origin_source_estimates.csv",
        "panel": output_dir / "rolling_forecast_origin_consensus_panel.csv",
        "metrics": output_dir / "rolling_forecast_origin_consensus_metrics.csv",
        "train_selected": output_dir / "rolling_forecast_origin_train_selected_metrics.csv",
        "paired_deltas": output_dir / "rolling_forecast_origin_paired_deltas.csv",
        "point_policy": output_dir / "rolling_forecast_origin_point_model_policy.csv",
        "selected_point_panel": output_dir / "rolling_forecast_origin_selected_point_panel.csv",
        "selected_point_metrics": output_dir / "rolling_forecast_origin_selected_point_metrics.csv",
        "primary_point_panel": output_dir / "rolling_forecast_origin_primary_point_panel.csv",
        "primary_point_metrics": output_dir / "rolling_forecast_origin_primary_point_metrics.csv",
        "mean_point_metrics": output_dir / "rolling_forecast_origin_mean_point_metrics.csv",
        "interval_metrics": output_dir / "rolling_forecast_origin_interval_metrics.csv",
        "interval_panel": output_dir / "rolling_forecast_origin_interval_panel.csv",
        "interval_policy": output_dir / "rolling_forecast_origin_interval_model_policy.csv",
        "selected_interval_metrics": output_dir / "rolling_forecast_origin_selected_interval_metrics.csv",
    }
    source_panel.sort_values(["opening_weekend_start", "release_run_id", "origin_day", "estimate_source"]).to_csv(
        paths["source"], index=False
    )
    consensus_panel.sort_values(["opening_weekend_start", "release_run_id", "origin_day"]).to_csv(
        paths["panel"], index=False
    )
    write_sorted_metrics(metrics, paths["metrics"], ["evaluation_subset", "_origin_day_order", "forecast_method"])
    write_sorted_metrics(
        train_selected_metrics,
        paths["train_selected"],
        ["evaluation_subset", "_origin_day_order", "forecast_method"],
    )
    write_sorted_metrics(
        paired_deltas,
        paths["paired_deltas"],
        ["evaluation_subset", "_origin_day_order", "candidate_method"],
    )
    point_policy.sort_values("origin_day").to_csv(paths["point_policy"], index=False)
    selected_point_panel.sort_values(["opening_weekend_start", "release_run_id", "origin_day"]).to_csv(
        paths["selected_point_panel"], index=False
    )
    write_sorted_metrics(
        selected_point_metrics,
        paths["selected_point_metrics"],
        ["evaluation_subset", "_origin_day_order", "forecast_method"],
    )
    primary_point_panel.sort_values(["opening_weekend_start", "release_run_id", "origin_day"]).to_csv(
        paths["primary_point_panel"], index=False
    )
    write_sorted_metrics(
        primary_point_metrics,
        paths["primary_point_metrics"],
        ["evaluation_subset", "_origin_day_order", "forecast_method"],
    )
    write_sorted_metrics(
        mean_point_metrics,
        paths["mean_point_metrics"],
        ["evaluation_subset", "_origin_day_order", "forecast_method"],
    )
    write_sorted_metrics(
        interval_metrics,
        paths["interval_metrics"],
        ["evaluation_subset", "_origin_day_order", "interval_model"],
    )
    interval_panel.sort_values(["opening_weekend_start", "release_run_id", "origin_day"]).to_csv(
        paths["interval_panel"], index=False
    )
    interval_policy.sort_values("origin_day").to_csv(paths["interval_policy"], index=False)
    write_sorted_metrics(
        selected_interval_metrics,
        paths["selected_interval_metrics"],
        ["evaluation_subset", "_origin_day_order"],
    )
    return paths


def write_sorted_metrics(df: pd.DataFrame, path: Path, sort_cols: list[str]) -> None:
    out = df.copy()
    if "origin_day" in out.columns:
        out["_origin_day_order"] = pd.to_numeric(out["origin_day"], errors="coerce").fillna(999)
    existing_sort_cols = [col for col in sort_cols if col in out.columns]
    drop_cols = [col for col in ["_origin_day_order"] if col in out.columns]
    out.sort_values(existing_sort_cols).drop(columns=drop_cols).to_csv(path, index=False)


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def parse_str_list(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a rolling-origin consensus benchmark from analytics EDA forecast tables."
    )
    add_database_arg(parser)
    parser.add_argument("--output-dir", type=Path, default=DIAGNOSTICS_DIR)
    parser.add_argument(
        "--origin-days",
        default=",".join(str(day) for day in DEFAULT_ORIGIN_DAYS),
        help="Comma-separated opening-weekend-relative forecast origin days, e.g. -14,-13,...,-1.",
    )
    parser.add_argument(
        "--recency-lambdas",
        default=",".join(str(value) for value in DEFAULT_RECENCY_LAMBDAS),
        help="Comma-separated lambda values for exp(-lambda * source_age_days).",
    )
    parser.add_argument(
        "--exclude-sources",
        default=",".join(DEFAULT_EXCLUDED_ESTIMATE_SOURCES),
        help="Comma-separated estimate_source values to exclude.",
    )
    parser.add_argument(
        "--exclude-release-years",
        default=",".join(str(year) for year in DEFAULT_EXCLUDED_RELEASE_YEARS),
        help="Comma-separated release years to exclude.",
    )
    parser.add_argument(
        "--train-years",
        default=",".join(str(year) for year in DEFAULT_TRAIN_YEARS),
        help="Comma-separated years labelled train in the metrics table.",
    )
    parser.add_argument("--test-start-year", type=int, default=DEFAULT_TEST_START_YEAR)
    parser.add_argument("--min-source-reliability-n", type=int, default=DEFAULT_MIN_SOURCE_RELIABILITY_N)
    parser.add_argument("--source-bias-shrink-k", type=int, default=DEFAULT_SOURCE_BIAS_SHRINK_K)
    parser.add_argument(
        "--max-source-age-days",
        default=",".join(str(day) for day in DEFAULT_MAX_SOURCE_AGE_DAYS),
        help="Comma-separated source-age caps to test, e.g. 3,7,14.",
    )
    parser.add_argument(
        "--min-train-n-for-model-selection",
        type=int,
        default=DEFAULT_MIN_TRAIN_N_FOR_MODEL_SELECTION,
    )
    parser.add_argument("--interval-shrink-k", type=int, default=DEFAULT_INTERVAL_SHRINK_K)
    return parser


def config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    return BenchmarkConfig(
        origin_days=parse_int_list(args.origin_days),
        excluded_estimate_sources=parse_str_list(args.exclude_sources),
        excluded_release_years=parse_int_list(args.exclude_release_years),
        recency_lambdas=parse_float_list(args.recency_lambdas),
        train_years=parse_int_list(args.train_years),
        test_start_year=args.test_start_year,
        min_source_reliability_n=args.min_source_reliability_n,
        source_bias_shrink_k=args.source_bias_shrink_k,
        max_source_age_days=parse_int_list(args.max_source_age_days),
        min_train_n_for_model_selection=args.min_train_n_for_model_selection,
        interval_shrink_k=args.interval_shrink_k,
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)

    openings, estimates = load_inputs(args.database_url)
    source_panel = build_source_origin_panel(openings, estimates, config)
    source_panel = add_source_reliability(source_panel, config.min_source_reliability_n)
    source_panel = add_source_bias_calibration(
        source_panel,
        config.min_source_reliability_n,
        config.source_bias_shrink_k,
    )
    source_panel = add_source_bias_adjusted_reliability(source_panel, config.min_source_reliability_n)
    consensus_panel = build_consensus_panel(source_panel, config)
    metrics = build_metrics(consensus_panel)
    train_selected_metrics = build_train_selected_metrics(consensus_panel)
    paired_deltas = build_paired_deltas(consensus_panel)

    point_policy = build_point_model_policy(
        consensus_panel,
        min_train_n=config.min_train_n_for_model_selection,
    )
    selected_point_panel = add_selected_point_forecast(consensus_panel, point_policy)
    selected_point_metrics = build_selected_point_metrics(selected_point_panel)
    locked_point_panel = add_locked_point_forecasts(consensus_panel)
    primary_point_metrics = build_selected_point_metrics(
        locked_point_panel,
        selected_col=DEFAULT_PRIMARY_POINT_COL,
    )
    mean_point_metrics = build_selected_point_metrics(
        locked_point_panel,
        selected_col=DEFAULT_MEAN_POINT_COL,
    )

    interval_panel = add_grouped_log_residual_intervals(
        locked_point_panel,
        DEFAULT_PRIMARY_POINT_COL,
        shrink_k=config.interval_shrink_k,
    )
    interval_panel = add_empirical_quantile_intervals(
        interval_panel,
        DEFAULT_PRIMARY_POINT_COL,
        group_cols=[],
        interval_model="empirical_global_quantile",
        shrink_k=config.interval_shrink_k,
    )
    interval_panel = add_empirical_quantile_intervals(
        interval_panel,
        DEFAULT_PRIMARY_POINT_COL,
        group_cols=["origin_day"],
        interval_model="empirical_origin_quantile_shrunk",
        shrink_k=config.interval_shrink_k,
    )
    interval_panel = add_empirical_quantile_intervals(
        interval_panel,
        DEFAULT_PRIMARY_POINT_COL,
        group_cols=["origin_day", "is_franchise"],
        interval_model="empirical_origin_franchise_quantile_shrunk",
        shrink_k=config.interval_shrink_k,
    )
    interval_metrics = build_interval_metrics(interval_panel, DEFAULT_PRIMARY_POINT_COL)
    interval_policy = build_locked_interval_model_policy(interval_metrics)
    interval_panel = add_selected_intervals(
        interval_panel,
        interval_policy,
        forecast_col=DEFAULT_PRIMARY_POINT_COL,
    )
    selected_interval_metrics = build_selected_interval_metrics(
        interval_panel,
        forecast_col=DEFAULT_PRIMARY_POINT_COL,
    )
    paths = write_outputs(
        source_panel,
        consensus_panel,
        metrics,
        train_selected_metrics,
        paired_deltas,
        point_policy,
        selected_point_panel,
        selected_point_metrics,
        locked_point_panel,
        primary_point_metrics,
        mean_point_metrics,
        interval_metrics,
        interval_panel,
        interval_policy,
        selected_interval_metrics,
        args.output_dir,
    )

    print(f"Wrote {paths['source']} ({len(source_panel):,} rows)")
    print(f"Wrote {paths['panel']} ({len(consensus_panel):,} rows)")
    print(f"Wrote {paths['metrics']} ({len(metrics):,} rows)")
    print(f"Wrote {paths['train_selected']} ({len(train_selected_metrics):,} rows)")
    print(f"Wrote {paths['paired_deltas']} ({len(paired_deltas):,} rows)")
    print(f"Wrote {paths['point_policy']} ({len(point_policy):,} rows)")
    print(f"Wrote {paths['selected_point_panel']} ({len(selected_point_panel):,} rows)")
    print(f"Wrote {paths['selected_point_metrics']} ({len(selected_point_metrics):,} rows)")
    print(f"Wrote {paths['primary_point_panel']} ({len(locked_point_panel):,} rows)")
    print(f"Wrote {paths['primary_point_metrics']} ({len(primary_point_metrics):,} rows)")
    print(f"Wrote {paths['mean_point_metrics']} ({len(mean_point_metrics):,} rows)")
    print(f"Wrote {paths['interval_metrics']} ({len(interval_metrics):,} rows)")
    print(f"Wrote {paths['interval_panel']} ({len(interval_panel):,} rows)")
    print(f"Wrote {paths['interval_policy']} ({len(interval_policy):,} rows)")
    print(f"Wrote {paths['selected_interval_metrics']} ({len(selected_interval_metrics):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
