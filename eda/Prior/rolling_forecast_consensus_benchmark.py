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


REPO_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTICS_DIR = REPO_ROOT / "data" / "diagnostics"

DEFAULT_ORIGIN_DAYS = tuple(range(-14, 0))
DEFAULT_EXCLUDED_ESTIMATE_SOURCES = ("the_numbers_predictions", "boxofficeguru")
DEFAULT_EXCLUDED_RELEASE_YEARS = (2020, 2021)
DEFAULT_RECENCY_LAMBDAS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00)
DEFAULT_TEST_START_YEAR = 2024
DEFAULT_TRAIN_YEARS = tuple(range(2010, 2020)) + (2022, 2023)
DEFAULT_MIN_SOURCE_RELIABILITY_N = 5
DEFAULT_SOURCE_BIAS_SHRINK_K = 10
DEFAULT_MAX_SOURCE_AGE_DAYS = (1, 3, 5, 7, 14)
DEFAULT_MIN_TRAIN_N_FOR_MODEL_SELECTION = 10
DEFAULT_INTERVAL_SHRINK_K = 20
DEFAULT_INTERVAL_LEVELS = (80, 95)
DEFAULT_POINT_BASELINE_METHOD = "dollar_median_consensus_usd"
DEFAULT_SELECTED_POINT_COL = "selected_point_forecast_usd"
DEFAULT_PRIMARY_POINT_COL = "primary_point_forecast_usd"
DEFAULT_MEAN_POINT_COL = "mean_point_forecast_usd"
INCLUDED_TARGET_ALIGNMENTS = frozenset({"exact_target_date", "no_target_date_opening_metric"})


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


def classify_estimate_target_alignment(
    df: pd.DataFrame,
    *,
    forecast_target_col: str = "opening_weekend_start",
) -> pd.Series:
    """Classify whether an estimate targets the weekend being scored.

    Opening-weekend metrics with explicit target dates must match the forecast
    target weekend. A mismatched date is not uncertainty; it is a different
    event target and must be excluded from calibration.
    """

    metric = df["forecast_metric"].fillna("").astype(str).str.lower()
    target_start = pd.to_datetime(df["target_start_date"], errors="coerce")
    forecast_target = pd.to_datetime(df[forecast_target_col], errors="coerce")
    has_target = target_start.notna()
    opening_metric = metric.str.contains("opening", na=False)
    release_type = df.get("release_type", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    width_bucket = df.get("release_width_bucket", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    is_platform = release_type.str.contains("platform", na=False) | width_bucket.eq("platform")

    return pd.Series(
        np.select(
            [
                has_target & target_start.eq(forecast_target),
                has_target & target_start.ne(forecast_target),
                target_start.isna() & is_platform & opening_metric,
                target_start.isna() & opening_metric,
                target_start.isna() & metric.eq(""),
            ],
            [
                "exact_target_date",
                "target_date_mismatch",
                "platform_target_ambiguous",
                "no_target_date_opening_metric",
                "no_target_date_unknown_metric",
            ],
            default="non_opening_metric",
        ),
        index=df.index,
    )


def add_target_alignment_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["forecast_target_weekend_start"] = out["opening_weekend_start"]
    out["target_alignment"] = classify_estimate_target_alignment(out)
    target_start = pd.to_datetime(out["target_start_date"], errors="coerce")
    forecast_target = pd.to_datetime(out["forecast_target_weekend_start"], errors="coerce")
    out["alignment_delta_days"] = (target_start - forecast_target).dt.days
    out["included_by_target_alignment"] = out["target_alignment"].isin(INCLUDED_TARGET_ALIGNMENTS)
    return out


def target_alignment_exclusion_reason(df: pd.DataFrame) -> pd.Series:
    excluded_source = df["estimate_source"].isin(df.attrs.get("excluded_estimate_sources", ()))
    estimate_mid = pd.to_numeric(df["estimate_mid_usd"], errors="coerce")
    no_estimate_date = df["estimate_date"].isna()
    after_origin = df["estimate_date"].gt(df["forecast_origin_date"])
    target_day_count = pd.to_numeric(df.get("target_day_count", pd.Series(index=df.index)), errors="coerce")
    return pd.Series(
        np.select(
            [
                excluded_source,
                estimate_mid.le(0) | estimate_mid.isna(),
                no_estimate_date,
                after_origin,
                target_day_count.ne(3),
                ~df["included_by_target_alignment"],
            ],
            [
                "excluded_source",
                "non_positive_estimate",
                "missing_estimate_date",
                "estimate_after_forecast_origin",
                "non_3_day_target",
                df["target_alignment"],
            ],
            default="included",
        ),
        index=df.index,
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


def add_canonical_source_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "estimate_source_raw" not in out.columns:
        out["estimate_source_raw"] = out["estimate_source"]
    raw = out["estimate_source_raw"].fillna("").astype(str)
    out["publication_channel"] = np.select(
        [
            raw.eq("boxofficetheory_substack"),
            raw.eq("boxofficetheory"),
        ],
        [
            "substack",
            "legacy_site",
        ],
        default=raw,
    )
    estimate_date = pd.to_datetime(out.get("estimate_date"), errors="coerce")
    out["source_era"] = np.select(
        [
            raw.eq("boxofficetheory_substack"),
            raw.eq("boxofficetheory") & estimate_date.ge(pd.Timestamp("2024-01-01")),
            raw.eq("boxofficetheory"),
        ],
        [
            "boxofficetheory_substack",
            "boxofficetheory_recent_legacy",
            "boxofficetheory_historical_legacy",
        ],
        default="source_default",
    )
    out["estimate_source"] = raw.replace({"boxofficetheory_substack": "boxofficetheory"})
    return out


def aggregate_source_rule(
    source_panel: pd.DataFrame,
    rule_name: str,
    *,
    recency_lambda: float = 1.0,
) -> pd.DataFrame:
    group_cols = ["release_run_id", "origin_day"]
    rows = []
    for keys, group in source_panel.groupby(group_cols):
        release_run_id, origin_day = keys
        record: dict[str, float | int | str] = {
            "release_run_id": release_run_id,
            "origin_day": origin_day,
            f"{rule_name}_source_count": int(group["estimate_source"].nunique()),
            f"{rule_name}_estimate_count": int(len(group)),
            f"{rule_name}_same_day_source_count": int(group.loc[group["source_age_days"].eq(0), "estimate_source"].nunique()),
            f"{rule_name}_oldest_estimate_age_days": float(pd.to_numeric(group["source_age_days"], errors="coerce").max()),
            f"{rule_name}_newest_estimate_age_days": float(pd.to_numeric(group["source_age_days"], errors="coerce").min()),
            f"{rule_name}_source_composition": ", ".join(sorted(group["estimate_source"].dropna().unique())),
        }
        low = pd.to_numeric(group["estimate_mid_usd"], errors="coerce")
        log_mid = pd.to_numeric(group["log_estimate_mid"], errors="coerce")
        log_adj = pd.to_numeric(group["log_estimate_mid_source_bias_adj"], errors="coerce")
        record[f"{rule_name}_dollar_median_consensus_usd"] = float(low.median())
        record[f"{rule_name}_arithmetic_mean_consensus_usd"] = float(low.mean())
        record[f"{rule_name}_log_mean_consensus_usd"] = float(np.exp(log_mid.mean()))
        record[f"{rule_name}_log_median_consensus_usd"] = float(np.exp(log_mid.median()))
        record[f"{rule_name}_bias_adjusted_median_consensus_usd"] = float(np.exp(log_adj.median()))
        record[f"{rule_name}_bias_adjusted_log_mean_consensus_usd"] = float(np.exp(log_adj.mean()))

        temp = group.copy()
        temp[f"{rule_name}_recency_weight"] = np.exp(-recency_lambda * temp["source_age_days"])
        temp[f"{rule_name}_reliability_weight"] = temp["source_reliability_raw_weight"].fillna(1.0)
        temp[f"{rule_name}_recency_reliability_weight"] = (
            temp[f"{rule_name}_recency_weight"] * temp[f"{rule_name}_reliability_weight"]
        )
        record[f"{rule_name}_recency_weighted_log_mean_consensus_usd"] = weighted_log_forecast(
            temp,
            f"{rule_name}_recency_weight",
        )
        record[f"{rule_name}_reliability_weighted_log_mean_consensus_usd"] = weighted_log_forecast(
            temp,
            f"{rule_name}_reliability_weight",
        )
        record[f"{rule_name}_recency_reliability_weighted_log_mean_consensus_usd"] = weighted_log_forecast(
            temp,
            f"{rule_name}_recency_reliability_weight",
        )
        record[f"{rule_name}_log_sd_dispersion"] = float(log_mid.std(ddof=1))
        record[f"{rule_name}_log_mad_dispersion"] = float((log_mid - log_mid.median()).abs().median())
        record[f"{rule_name}_log_range_dispersion"] = float(log_mid.max() - log_mid.min())
        rows.append(record)
    return pd.DataFrame(rows)


def add_availability_rule_candidates(
    base: pd.DataFrame,
    source_panel: pd.DataFrame,
) -> pd.DataFrame:
    group_cols = ["release_run_id", "origin_day"]
    out = base.copy()
    rules: list[tuple[str, pd.DataFrame]] = [("latest_available", source_panel.copy())]
    rules.append(("same_day", source_panel.loc[source_panel["source_age_days"].eq(0)].copy()))
    for cap in (1, 3, 5, 7):
        rules.append((f"max_age_{cap}d", source_panel.loc[source_panel["source_age_days"].le(cap)].copy()))

    eligible_parts = []
    for _, group in source_panel.groupby(group_cols):
        chosen = group.loc[group["source_age_days"].le(1)]
        if chosen.empty:
            chosen = group.loc[group["source_age_days"].le(3)]
        if chosen.empty:
            chosen = group.loc[group["source_age_days"].le(7)]
        if chosen.empty:
            chosen = group
        eligible_parts.append(chosen)
    if eligible_parts:
        rules.append(("hierarchical_freshness", pd.concat(eligible_parts, ignore_index=True)))

    for rule_name, frame in rules:
        if frame.empty:
            continue
        out = out.merge(aggregate_source_rule(frame, rule_name), on=group_cols, how="left")
    return out


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
            target_day_count,
            forecast_metric,
            estimate_low_usd,
            estimate_high_usd,
            estimate_mid_usd,
            estimate_width_usd,
            source_movie_title,
            raw_forecast_text,
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
    *,
    return_alignment_audit: bool = False,
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
    estimate_candidates = add_target_alignment_columns(estimate_candidates)
    estimate_candidates["log_estimate_mid"] = safe_log(estimate_candidates["estimate_mid_usd"])

    origins = pd.DataFrame({"origin_day": list(config.origin_days)})
    movie_origins = eligible_openings[opening_cols].merge(origins, how="cross")
    movie_origins["forecast_origin_date"] = movie_origins["opening_weekend_start"] + pd.to_timedelta(
        movie_origins["origin_day"], unit="D"
    )

    estimate_panel_cols = [
        "eda_estimate_id",
        "estimate_source",
        "source_prediction_id",
        "release_run_id",
        "estimate_date",
        "target_start_date",
        "target_end_date",
        "target_day_count",
        "forecast_metric",
        "forecast_target_weekend_start",
        "target_alignment",
        "alignment_delta_days",
        "included_by_target_alignment",
        "estimate_low_usd",
        "estimate_high_usd",
        "estimate_mid_usd",
        "estimate_width_usd",
        "source_movie_title",
        "raw_forecast_text",
        "actual_inside_range",
        "days_before_opening_weekend",
        "log_estimate_mid",
    ]
    available = movie_origins.merge(
        estimate_candidates[[col for col in estimate_panel_cols if col in estimate_candidates.columns]],
        on="release_run_id",
        how="inner",
    )
    available.attrs["excluded_estimate_sources"] = config.excluded_estimate_sources
    available = add_canonical_source_columns(available)
    available["included_in_calibration"] = (
        ~available["estimate_source"].isin(config.excluded_estimate_sources)
        & (pd.to_numeric(available["estimate_mid_usd"], errors="coerce") > 0)
        & available["estimate_date"].notna()
        & available["estimate_date"].le(available["forecast_origin_date"])
        & pd.to_numeric(available["target_day_count"], errors="coerce").eq(3)
        & available["included_by_target_alignment"]
        & available["log_estimate_mid"].notna()
    )
    available["exclusion_reason"] = target_alignment_exclusion_reason(available)
    alignment_audit = build_estimate_target_alignment_audit(available)
    available = available.loc[available["included_in_calibration"]].copy()
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
    latest["estimate_point_provenance"] = infer_estimate_point_provenance(latest)
    return (latest, alignment_audit) if return_alignment_audit else latest


def infer_estimate_point_provenance(df: pd.DataFrame) -> pd.Series:
    low = pd.to_numeric(df.get("estimate_low_usd"), errors="coerce")
    high = pd.to_numeric(df.get("estimate_high_usd"), errors="coerce")
    mid = pd.to_numeric(df.get("estimate_mid_usd"), errors="coerce")
    width = pd.to_numeric(df.get("estimate_width_usd"), errors="coerce")
    arithmetic_mid = (low + high) / 2.0
    geometric_mid = np.sqrt(low * high)
    has_range = low.notna() & high.notna() & low.gt(0) & high.gt(0) & high.gt(low)
    point_range = low.notna() & high.notna() & mid.notna() & np.isclose(low, high) & np.isclose(low, mid)
    arithmetic = has_range & mid.notna() & np.isclose(mid, arithmetic_mid)
    geometric = has_range & mid.notna() & np.isclose(mid, geometric_mid)
    zero_width = width.fillna(0).eq(0)
    return pd.Series(
        np.select(
            [
                point_range | (zero_width & mid.notna() & ~has_range),
                arithmetic,
                geometric,
                has_range & mid.isna(),
                mid.notna(),
            ],
            [
                "explicit_point",
                "arithmetic_range_midpoint",
                "geometric_range_midpoint",
                "range_only",
                "unknown_point_provenance",
            ],
            default="missing_point",
        ),
        index=df.index,
    )


def build_estimate_target_alignment_audit(available: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "release_run_id",
        "movie_id",
        "title",
        "release_type",
        "opening_weekend_start",
        "forecast_target_weekend_start",
        "origin_day",
        "estimate_source",
        "estimate_source_raw",
        "publication_channel",
        "source_era",
        "source_prediction_id",
        "forecast_metric",
        "estimate_date",
        "target_start_date",
        "target_end_date",
        "target_day_count",
        "estimate_mid_usd",
        "target_alignment",
        "alignment_delta_days",
        "included_in_calibration",
        "exclusion_reason",
    ]
    existing = [column for column in columns if column in available.columns]
    out = available[existing].copy()
    for column in ["opening_weekend_start", "forecast_target_weekend_start", "estimate_date", "target_start_date", "target_end_date"]:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce").dt.date
    return out.sort_values(["opening_weekend_start", "release_run_id", "origin_day", "estimate_source", "estimate_date"])


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


def build_consensus_panel(
    source_panel: pd.DataFrame,
    config: BenchmarkConfig,
    *,
    include_extended_candidates: bool = True,
) -> pd.DataFrame:
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
    if include_extended_candidates:
        base = add_availability_rule_candidates(base, source_panel)
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
            or col.startswith("latest_available_")
            or col.startswith("same_day_")
            or col.startswith("max_age_")
            or col.startswith("hierarchical_freshness_")
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
        or method.startswith("latest_available_")
        or method.startswith("same_day_")
        or method.startswith("max_age_")
        or method.startswith("hierarchical_freshness_")
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


def add_interval_calibration_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    origin = pd.to_numeric(out["origin_day"], errors="coerce")
    out["origin_bucket"] = pd.cut(
        origin,
        bins=[-15, -8, -3, -2, -1, 0],
        labels=["P_-14_to_-8", "P_-7_to_-3", "P_-2", "P_-1", "P_0"],
        right=True,
    ).astype(str)

    source_count = pd.to_numeric(out.get("source_count"), errors="coerce")
    out["source_count_bucket"] = np.where(source_count.le(1), "one_source", "multi_source")

    point = pd.to_numeric(out[DEFAULT_PRIMARY_POINT_COL], errors="coerce")
    out["point_bucket"] = pd.cut(
        point,
        bins=[0, 1_000_000, 5_000_000, 15_000_000, 50_000_000, np.inf],
        labels=["lt_1m", "1m_5m", "5m_15m", "15m_50m", "50m_plus"],
        right=False,
    ).astype(str)

    release_type = out.get("release_type", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
    width_bucket = out.get("release_width_bucket", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
    out["platform_bucket"] = np.where(
        release_type.str.contains("platform") | width_bucket.eq("platform"),
        "platform",
        "non_platform",
    )
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


def add_centered_empirical_quantile_intervals(
    panel: pd.DataFrame,
    forecast_col: str,
    group_cols: list[str],
    interval_model: str,
    shrink_k: int = DEFAULT_INTERVAL_SHRINK_K,
) -> pd.DataFrame:
    out = panel.copy()
    log_forecast = safe_log(out[forecast_col])
    residual_col = f"{forecast_col}_residual_log"

    if residual_col not in out.columns:
        out[residual_col] = out["log_actual_opening_weekend"] - log_forecast

    train = out.loc[out["evaluation_subset"].eq("train")].copy()
    residuals = pd.to_numeric(train[residual_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if residuals.empty:
        for level in DEFAULT_INTERVAL_LEVELS:
            out[f"{forecast_col}_{interval_model}_lo_{level}"] = np.nan
            out[f"{forecast_col}_{interval_model}_hi_{level}"] = np.nan
        out[f"{interval_model}_center_log"] = np.nan
        return out

    global_center = float(residuals.median())
    train["centered_residual"] = pd.to_numeric(train[residual_col], errors="coerce") - global_center
    global_stats = {
        80: {
            "lo": float(train["centered_residual"].quantile(0.10)),
            "hi": float(train["centered_residual"].quantile(0.90)),
        },
        95: {
            "lo": float(train["centered_residual"].quantile(0.025)),
            "hi": float(train["centered_residual"].quantile(0.975)),
        },
    }

    center_stats = (
        train.groupby(group_cols, dropna=False)[residual_col]
        .agg(n="count", center="median")
        .reset_index()
    )
    out_temp = out[group_cols].merge(center_stats, on=group_cols, how="left")
    w_center = out_temp["n"] / (out_temp["n"] + shrink_k)
    center = (w_center * out_temp["center"] + (1.0 - w_center) * global_center).fillna(global_center)

    train = train.merge(center_stats, on=group_cols, how="left", suffixes=("", "_cell"))
    w_train = train["n"] / (train["n"] + shrink_k)
    train["shrunk_center"] = (w_train * train["center"] + (1.0 - w_train) * global_center).fillna(global_center)
    train["centered_residual"] = train[residual_col] - train["shrunk_center"]

    for level, probs in {80: (0.10, 0.90), 95: (0.025, 0.975)}.items():
        lo_q, hi_q = probs
        q = (
            train.groupby(group_cols, dropna=False)["centered_residual"]
            .agg(
                n_q="count",
                lo=lambda s, q=lo_q: s.quantile(q),
                hi=lambda s, q=hi_q: s.quantile(q),
            )
            .reset_index()
        )
        w_q = q["n_q"] / (q["n_q"] + shrink_k)
        q["lo_shrunk"] = w_q * q["lo"] + (1.0 - w_q) * global_stats[level]["lo"]
        q["hi_shrunk"] = w_q * q["hi"] + (1.0 - w_q) * global_stats[level]["hi"]

        temp = out[group_cols].merge(q[group_cols + ["lo_shrunk", "hi_shrunk"]], on=group_cols, how="left")
        lo = temp["lo_shrunk"].fillna(global_stats[level]["lo"])
        hi = temp["hi_shrunk"].fillna(global_stats[level]["hi"])

        out[f"{forecast_col}_{interval_model}_lo_{level}"] = np.exp(log_forecast + center + lo)
        out[f"{forecast_col}_{interval_model}_hi_{level}"] = np.exp(log_forecast + center + hi)

    out[f"{interval_model}_center_log"] = center
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


def interval_score(actual: pd.Series, lo: pd.Series, hi: pd.Series, alpha: float) -> pd.Series:
    width = hi - lo
    return width + (2.0 / alpha) * (lo - actual).clip(lower=0) + (2.0 / alpha) * (actual - hi).clip(lower=0)


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
    lo80 = x[f"{forecast_col}_{interval_model}_lo_80"]
    hi80 = x[f"{forecast_col}_{interval_model}_hi_80"]
    lo95 = x[f"{forecast_col}_{interval_model}_lo_95"]
    hi95 = x[f"{forecast_col}_{interval_model}_hi_95"]
    width_80 = (hi80 - lo80) / x[forecast_col]
    width_95 = (hi95 - lo95) / x[forecast_col]
    out = {
        "forecast_method": forecast_col,
        "interval_model": interval_model,
        "n": int(len(x)),
        "coverage_80": float(((actual >= lo80) & (actual <= hi80)).mean()),
        "coverage_95": float(((actual >= lo95) & (actual <= hi95)).mean()),
        "mean_width_80_pct": float(width_80.mean()),
        "mean_width_95_pct": float(width_95.mean()),
        "median_width_80_pct": float(width_80.median()),
        "median_width_95_pct": float(width_95.median()),
        "lower_miss_80": float((actual < lo80).mean()),
        "upper_miss_80": float((actual > hi80).mean()),
        "lower_miss_95": float((actual < lo95).mean()),
        "upper_miss_95": float((actual > hi95).mean()),
        "mean_interval_score_80_pct": float((interval_score(actual, lo80, hi80, 0.20) / x[forecast_col]).mean()),
        "mean_interval_score_95_pct": float((interval_score(actual, lo95, hi95, 0.05) / x[forecast_col]).mean()),
    }
    out["miss_95_imbalance"] = abs(out["lower_miss_95"] - out["upper_miss_95"])
    return out


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
        "empirical_origin_range_disagreement_quantile_shrunk",
        "empirical_origin_bucket_source_count_point_bucket_centered_quantile_shrunk",
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
            selected_model = "empirical_origin_bucket_source_count_point_bucket_centered_quantile_shrunk"
            reason = "fallback_insufficient_train_support"
            selected: dict[str, float | int | str] = {}
        else:
            feasible = g.loc[
                g["coverage_80"].ge(0.72)
                & g["coverage_95"].ge(0.88)
                & g["miss_95_imbalance"].le(0.10)
            ].copy()
            selection_pool = feasible if not feasible.empty else g
            selected = selection_pool.sort_values(
                ["mean_interval_score_95_pct", "miss_95_imbalance", "median_width_95_pct"]
            ).iloc[0].to_dict()
            selected_model = str(selected["interval_model"])
            reason = "train_selected_normalized_interval_score"

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
                "train_lower_miss_95": selected.get("lower_miss_95", np.nan),
                "train_upper_miss_95": selected.get("upper_miss_95", np.nan),
                "train_mean_interval_score_95_pct": selected.get("mean_interval_score_95_pct", np.nan),
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


def target_universe_mask(df: pd.DataFrame) -> pd.Series:
    release_type = df.get("release_type", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    width_bucket = df.get("release_width_bucket", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    forecast = pd.to_numeric(
        df.get(DEFAULT_PRIMARY_POINT_COL, df.get("estimate_mid_usd", pd.Series(np.nan, index=df.index))),
        errors="coerce",
    )
    wide_large = (
        df.get("is_wide_release", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        | df.get("is_large_release", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        | width_bucket.isin(["wide", "large_wide"])
    )
    non_comparable = (
        release_type.str.contains("limited|platform|event|concert", regex=True, na=False)
        | width_bucket.str.contains("limited|platform|event|concert", regex=True, na=False)
        | df.get("is_doc_concert", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    )
    return wide_large & ~non_comparable & forecast.ge(10_000_000)


def summarize_source_error_frame(
    df: pd.DataFrame,
    *,
    source_col: str = "estimate_source",
    residual_col: str = "source_residual_log",
    forecast_col: str = "estimate_mid_usd",
    actual_col: str = "opening_weekend_gross_usd",
) -> dict[str, float | int | str]:
    x = df.copy()
    residual = pd.to_numeric(x[residual_col], errors="coerce")
    forecast = pd.to_numeric(x[forecast_col], errors="coerce")
    actual = pd.to_numeric(x[actual_col], errors="coerce")
    valid = residual.notna() & np.isfinite(residual) & forecast.gt(0) & actual.gt(0)
    x = x.loc[valid].copy()
    residual = residual.loc[valid]
    forecast = forecast.loc[valid]
    actual = actual.loc[valid]
    if x.empty:
        return {
            "n_rows": 0,
            "n_movies": 0,
            "first_forecast_date": np.nan,
            "last_forecast_date": np.nan,
        "median_source_age_days": np.nan,
        "max_source_age_days": np.nan,
        "mean_log_bias": np.nan,
        "median_log_bias": np.nan,
            "MAE_log": np.nan,
            "RMSE_log": np.nan,
            "MdAPE": np.nan,
            "underprediction_rate": np.nan,
            "residual_q025": np.nan,
        "residual_q10": np.nan,
        "residual_q50": np.nan,
        "residual_q90": np.nan,
            "residual_q975": np.nan,
        }
    return {
        "n_rows": int(len(x)),
        "n_movies": int(x["release_run_id"].nunique()),
        "first_forecast_date": pd.to_datetime(x["estimate_date"], errors="coerce").min(),
        "last_forecast_date": pd.to_datetime(x["estimate_date"], errors="coerce").max(),
        "median_source_age_days": float(pd.to_numeric(x.get("source_age_days"), errors="coerce").median()),
        "max_source_age_days": float(pd.to_numeric(x.get("source_age_days"), errors="coerce").max()),
        "mean_log_bias": float(residual.mean()),
        "median_log_bias": float(residual.median()),
        "MAE_log": float(residual.abs().mean()),
        "RMSE_log": float(np.sqrt(np.mean(residual**2))),
        "MdAPE": float(np.median(np.abs(actual / forecast - 1.0))),
        "underprediction_rate": float((residual > 0).mean()),
        "residual_q025": float(residual.quantile(0.025)),
        "residual_q10": float(residual.quantile(0.10)),
        "residual_q50": float(residual.quantile(0.50)),
        "residual_q90": float(residual.quantile(0.90)),
        "residual_q975": float(residual.quantile(0.975)),
    }


def build_source_data_coverage_audit(
    source_panel: pd.DataFrame,
    target_alignment_audit: pd.DataFrame,
    openings: pd.DataFrame,
) -> pd.DataFrame:
    target_openings = openings.loc[
        (pd.to_numeric(openings["opening_weekend_gross_usd"], errors="coerce") > 0)
        & (
            openings.get("is_wide_release", pd.Series(False, index=openings.index)).fillna(False).astype(bool)
            | openings.get("is_large_release", pd.Series(False, index=openings.index)).fillna(False).astype(bool)
            | openings.get("release_width_bucket", pd.Series("", index=openings.index)).fillna("").isin(["wide", "large_wide"])
        )
    ]
    target_movies = max(int(target_openings["release_run_id"].nunique()), 1)
    rows: list[dict[str, float | int | str]] = []
    for source, group in source_panel.groupby("estimate_source", dropna=False):
        audit_group = target_alignment_audit.loc[target_alignment_audit["estimate_source"].eq(source)]
        rows.append(
            {
                "estimate_source": source,
                "eligible_origin_rows": int(len(group)),
                "eligible_movies": int(group["release_run_id"].nunique()),
                "target_universe_coverage": float(group["release_run_id"].nunique() / target_movies),
                "raw_audit_rows": int(len(audit_group)),
                "raw_unique_predictions": int(audit_group["source_prediction_id"].nunique()),
                "included_raw_rows": int(audit_group.get("included_in_calibration", pd.Series(dtype=bool)).astype(bool).sum()),
                "excluded_raw_rows": int((~audit_group.get("included_in_calibration", pd.Series(dtype=bool)).astype(bool)).sum()),
                "first_estimate_date": pd.to_datetime(group["estimate_date"], errors="coerce").min(),
                "last_estimate_date": pd.to_datetime(group["estimate_date"], errors="coerce").max(),
                "missing_estimate_date_rows": int(audit_group["estimate_date"].isna().sum()) if "estimate_date" in audit_group else 0,
                "target_alignment_values": "; ".join(
                    f"{key}:{value}" for key, value in audit_group["target_alignment"].value_counts(dropna=False).sort_index().items()
                )
                if "target_alignment" in audit_group
                else "",
                "point_provenance_values": "; ".join(
                    f"{key}:{value}" for key, value in group["estimate_point_provenance"].value_counts(dropna=False).sort_index().items()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_source_individual_performance(source_panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    specs: list[tuple[str, list[str], pd.Series | None]] = [
        ("origin", ["estimate_source", "origin_day"], None),
        ("pooled", ["estimate_source"], None),
        ("target_aligned", ["estimate_source", "origin_day"], target_universe_mask(source_panel)),
        ("large_wide", ["estimate_source", "origin_day"], source_panel.get("is_large_release", pd.Series(False, index=source_panel.index)).fillna(False).astype(bool)),
        ("franchise", ["estimate_source", "origin_day"], source_panel.get("is_franchise", pd.Series(False, index=source_panel.index)).fillna(False).astype(bool)),
        ("non_franchise", ["estimate_source", "origin_day"], ~source_panel.get("is_franchise", pd.Series(False, index=source_panel.index)).fillna(False).astype(bool)),
        ("release_year", ["estimate_source", "release_year"], None),
    ]
    for segment, group_cols, mask in specs:
        frame = source_panel.loc[mask].copy() if mask is not None else source_panel.copy()
        for keys, group in frame.groupby(group_cols, dropna=False):
            key_values = keys if isinstance(keys, tuple) else (keys,)
            row = {"segment": segment, **dict(zip(group_cols, key_values))}
            row.update(summarize_source_error_frame(group))
            rows.append(row)
    return pd.DataFrame(rows)


def clustered_bootstrap_ci(
    df: pd.DataFrame,
    value_col: str,
    cluster_col: str = "release_run_id",
    *,
    iterations: int = 500,
    seed: int = 17,
) -> tuple[float, float]:
    values = df[[cluster_col, value_col]].dropna()
    clusters = values[cluster_col].dropna().unique()
    if len(clusters) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means: list[float] = []
    grouped = {cluster: group[value_col].to_numpy(dtype=float) for cluster, group in values.groupby(cluster_col)}
    for _ in range(iterations):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        sample_values = np.concatenate([grouped[cluster] for cluster in sampled])
        means.append(float(np.mean(sample_values)))
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def build_source_common_support_paired_scores(
    source_panel: pd.DataFrame,
    locked_point_panel: pd.DataFrame,
    baseline_col: str = DEFAULT_PRIMARY_POINT_COL,
) -> pd.DataFrame:
    baseline = locked_point_panel[["release_run_id", "origin_day", baseline_col, "evaluation_subset"]].copy()
    merged = source_panel.merge(baseline, on=["release_run_id", "origin_day"], how="inner")
    actual = pd.to_numeric(merged["opening_weekend_gross_usd"], errors="coerce")
    source_forecast = pd.to_numeric(merged["estimate_mid_usd"], errors="coerce")
    baseline_forecast = pd.to_numeric(merged[baseline_col], errors="coerce")
    merged["source_abs_error"] = (np.log(actual) - np.log(source_forecast)).abs()
    merged["baseline_abs_error"] = (np.log(actual) - np.log(baseline_forecast)).abs()
    merged["delta_abs_log_error"] = merged["source_abs_error"] - merged["baseline_abs_error"]
    merged = merged.loc[
        actual.gt(0)
        & source_forecast.gt(0)
        & baseline_forecast.gt(0)
        & np.isfinite(merged["delta_abs_log_error"])
    ].copy()
    rows: list[dict[str, float | int | str]] = []
    for keys, group in merged.groupby(["estimate_source", "origin_day"], dropna=False):
        source, origin_day = keys
        ci_lo, ci_hi = clustered_bootstrap_ci(group, "delta_abs_log_error")
        rows.append(
            {
                "estimate_source": source,
                "origin_day": int(origin_day),
                "baseline_method": baseline_col,
                "n": int(len(group)),
                "n_movies": int(group["release_run_id"].nunique()),
                "mean_delta_abs_log_error": float(group["delta_abs_log_error"].mean()),
                "median_delta_abs_log_error": float(group["delta_abs_log_error"].median()),
                "source_win_rate": float((group["delta_abs_log_error"] < 0).mean()),
                "cluster_bootstrap_mean_delta_lo95": ci_lo,
                "cluster_bootstrap_mean_delta_hi95": ci_hi,
            }
        )
    return pd.DataFrame(rows)


def build_source_error_correlation(source_panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    frames: list[tuple[str, pd.DataFrame]] = [("pooled", source_panel)]
    frames.extend((str(origin_day), group) for origin_day, group in source_panel.groupby("origin_day"))
    for origin_label, frame in frames:
        pivot = frame.pivot_table(
            index=["release_run_id", "origin_day"],
            columns="estimate_source",
            values="source_residual_log",
            aggfunc="first",
        )
        corr = pivot.corr(min_periods=3)
        sources = list(corr.columns)
        for idx, source_a in enumerate(sources):
            for source_b in sources[idx + 1 :]:
                pair = pivot[[source_a, source_b]].dropna()
                rows.append(
                    {
                        "origin_day": origin_label,
                        "source_a": source_a,
                        "source_b": source_b,
                        "n_common_rows": int(len(pair)),
                        "n_common_movies": int(pair.reset_index()["release_run_id"].nunique()) if len(pair) else 0,
                        "error_correlation": float(corr.loc[source_a, source_b]) if pd.notna(corr.loc[source_a, source_b]) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def build_source_leave_one_in_out_scores(source_panel: pd.DataFrame, config: BenchmarkConfig) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    all_panel = build_consensus_panel(source_panel, config, include_extended_candidates=False)
    sources = sorted(source_panel["estimate_source"].dropna().unique())
    methods = ["dollar_median_consensus_usd", "log_median_consensus_usd", "log_mean_consensus_usd"]
    established_sources = {"boxofficepro", "boxofficereport"}
    for source in sources:
        without_source = source_panel.loc[~source_panel["estimate_source"].eq(source)].copy()
        only_source = source_panel.loc[source_panel["estimate_source"].eq(source)].copy()
        panels = {
            "all_sources": all_panel,
            "leave_one_out": build_consensus_panel(without_source, config, include_extended_candidates=False)
            if not without_source.empty
            else pd.DataFrame(),
            "source_only": build_consensus_panel(only_source, config, include_extended_candidates=False)
            if not only_source.empty
            else pd.DataFrame(),
        }
        for scenario, panel in panels.items():
            if panel.empty:
                continue
            for method in methods:
                score = evaluate_forecast(panel, method)
                score.update(
                    {
                        "estimate_source": source,
                        "scenario": scenario,
                        "evaluation_subset": "all",
                        "origin_day": "pooled",
                    }
                )
                rows.append(score)
            for origin_day, origin_df in panel.groupby("origin_day"):
                for method in methods:
                    score = evaluate_forecast(origin_df, method)
                    score.update(
                        {
                            "estimate_source": source,
                            "scenario": scenario,
                            "evaluation_subset": "all",
                            "origin_day": int(origin_day),
                        }
                    )
                    rows.append(score)
    established_frame = source_panel.loc[source_panel["estimate_source"].isin(established_sources)].copy()
    if not established_frame.empty:
        established_panel = build_consensus_panel(established_frame, config, include_extended_candidates=False)
        comparison_sources = [source for source in sources if source not in established_sources]
        for source in comparison_sources:
            source_frame = source_panel.loc[source_panel["estimate_source"].eq(source)].copy()
            if source_frame.empty:
                continue
            plus_frame = pd.concat([established_frame, source_frame], ignore_index=True)
            panels = {
                "established_only": established_panel,
                "established_plus_source": build_consensus_panel(
                    plus_frame,
                    config,
                    include_extended_candidates=False,
                ),
            }
            for scenario, panel in panels.items():
                for method in methods:
                    score = evaluate_forecast(panel, method)
                    score.update(
                        {
                            "estimate_source": source,
                            "scenario": scenario,
                            "evaluation_subset": "all",
                            "origin_day": "pooled",
                        }
                    )
                    rows.append(score)
                for origin_day, origin_df in panel.groupby("origin_day"):
                    for method in methods:
                        score = evaluate_forecast(origin_df, method)
                        score.update(
                            {
                                "estimate_source": source,
                                "scenario": scenario,
                                "evaluation_subset": "all",
                                "origin_day": int(origin_day),
                            }
                        )
                        rows.append(score)
    return pd.DataFrame(rows)


def add_range_diagnostic_columns(source_panel: pd.DataFrame) -> pd.DataFrame:
    out = source_panel.copy()
    low = pd.to_numeric(out["estimate_low_usd"], errors="coerce")
    high = pd.to_numeric(out["estimate_high_usd"], errors="coerce")
    mid = pd.to_numeric(out["estimate_mid_usd"], errors="coerce")
    actual = pd.to_numeric(out["opening_weekend_gross_usd"], errors="coerce")
    out["source_internal_log_half_width"] = 0.5 * (np.log(high) - np.log(low))
    out["source_range_asymmetry_log"] = np.log(high / mid) - np.log(mid / low)
    out["range_contains_actual"] = actual.ge(low) & actual.le(high)
    out["range_lower_miss"] = actual.lt(low)
    out["range_upper_miss"] = actual.gt(high)
    out["source_abs_residual_log"] = pd.to_numeric(out["source_residual_log"], errors="coerce").abs()
    sorted_out = out.sort_values(["estimate_source", "origin_day", "opening_weekend_start", "release_run_id"])
    prior_median = sorted_out.groupby(
        ["estimate_source", "origin_day"],
        sort=False,
    )["source_internal_log_half_width"].transform(lambda s: s.expanding().median().shift(1))
    out["source_prior_median_internal_log_half_width"] = np.nan
    out.loc[sorted_out.index, "source_prior_median_internal_log_half_width"] = prior_median.to_numpy()
    out["source_relative_internal_log_half_width"] = (
        out["source_internal_log_half_width"] / out["source_prior_median_internal_log_half_width"]
    )
    return out


def build_source_range_calibration(source_panel: pd.DataFrame) -> pd.DataFrame:
    frame = add_range_diagnostic_columns(source_panel)
    frame = frame.loc[
        pd.to_numeric(frame["estimate_low_usd"], errors="coerce").gt(0)
        & pd.to_numeric(frame["estimate_high_usd"], errors="coerce").gt(pd.to_numeric(frame["estimate_low_usd"], errors="coerce"))
    ].copy()
    rows: list[dict[str, float | int | str]] = []
    specs = [
        ("origin", ["estimate_source", "origin_day"]),
        ("pooled", ["estimate_source"]),
        ("franchise", ["estimate_source", "origin_day", "is_franchise"]),
        ("release_year", ["estimate_source", "release_year"]),
    ]
    for segment, group_cols in specs:
        for keys, group in frame.groupby(group_cols, dropna=False):
            key_values = keys if isinstance(keys, tuple) else (keys,)
            rows.append(
                {
                    "segment": segment,
                    **dict(zip(group_cols, key_values)),
                    "n": int(len(group)),
                    "n_movies": int(group["release_run_id"].nunique()),
                    "coverage": float(group["range_contains_actual"].mean()),
                    "lower_miss_rate": float(group["range_lower_miss"].mean()),
                    "upper_miss_rate": float(group["range_upper_miss"].mean()),
                    "median_internal_log_half_width": float(group["source_internal_log_half_width"].median()),
                    "median_relative_width": float(((group["estimate_high_usd"] - group["estimate_low_usd"]) / group["estimate_mid_usd"]).median()),
                    "median_range_asymmetry_log": float(group["source_range_asymmetry_log"].median()),
                }
            )
    return pd.DataFrame(rows)


def build_source_range_resolution(source_panel: pd.DataFrame) -> pd.DataFrame:
    frame = add_range_diagnostic_columns(source_panel)
    rows: list[dict[str, float | int | str]] = []
    for keys, group in frame.groupby(["estimate_source", "origin_day"], dropna=False):
        source, origin_day = keys
        x = group[["source_internal_log_half_width", "source_relative_internal_log_half_width", "source_abs_residual_log"]].replace(
            [np.inf, -np.inf],
            np.nan,
        )
        rows.append(
            {
                "estimate_source": source,
                "origin_day": int(origin_day),
                "n": int(x["source_abs_residual_log"].notna().sum()),
                "corr_internal_width_abs_error": float(x["source_internal_log_half_width"].corr(x["source_abs_residual_log"])),
                "corr_relative_width_abs_error": float(x["source_relative_internal_log_half_width"].corr(x["source_abs_residual_log"])),
            }
        )
    return pd.DataFrame(rows)


def add_range_features_to_consensus_panel(consensus_panel: pd.DataFrame, source_panel: pd.DataFrame) -> pd.DataFrame:
    source_ranges = add_range_diagnostic_columns(source_panel)
    range_features = (
        source_ranges.groupby(["release_run_id", "origin_day"])
        .agg(
            aggregate_internal_log_half_width=("source_internal_log_half_width", "median"),
            aggregate_relative_internal_log_half_width=("source_relative_internal_log_half_width", "median"),
            aggregate_range_asymmetry_log=("source_range_asymmetry_log", "median"),
        )
        .reset_index()
    )
    out = consensus_panel.merge(range_features, on=["release_run_id", "origin_day"], how="left")
    out["cross_source_log_disagreement"] = pd.to_numeric(out.get("log_dispersion"), errors="coerce")
    out["range_width_bucket"] = median_bucket(
        out["aggregate_relative_internal_log_half_width"],
        low_label="range_narrow",
        high_label="range_wide",
        unknown_label="range_unknown",
    )
    out["disagreement_bucket"] = median_bucket(
        out["cross_source_log_disagreement"],
        low_label="disagreement_low",
        high_label="disagreement_high",
        unknown_label="disagreement_unknown",
    )
    return out


def median_bucket(
    values: pd.Series,
    *,
    low_label: str,
    high_label: str,
    unknown_label: str,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    median = numeric.median()
    if not np.isfinite(median):
        return pd.Series(unknown_label, index=values.index)
    return pd.Series(np.where(numeric.notna() & numeric.gt(median), high_label, low_label), index=values.index).where(
        numeric.notna(),
        unknown_label,
    )


def build_range_feature_effect_rolling(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    features = [
        "cross_source_log_disagreement",
        "aggregate_internal_log_half_width",
        "aggregate_relative_internal_log_half_width",
        "source_count",
        "is_franchise",
    ]
    residual = (safe_log(panel["actual_opening_weekend_gross_usd"]) - safe_log(panel[DEFAULT_PRIMARY_POINT_COL])).abs()
    frame = panel.assign(abs_primary_residual_log=residual)
    for origin_day, group in frame.groupby("origin_day"):
        for feature in features:
            x = pd.to_numeric(group[feature], errors="coerce") if feature != "is_franchise" else group[feature].astype(float)
            y = pd.to_numeric(group["abs_primary_residual_log"], errors="coerce")
            valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
            if valid.sum() < 3:
                slope = corr = np.nan
            else:
                slope = float(np.polyfit(x.loc[valid], y.loc[valid], 1)[0])
                corr = float(x.loc[valid].corr(y.loc[valid]))
            rows.append(
                {
                    "origin_day": int(origin_day),
                    "feature": feature,
                    "n": int(valid.sum()),
                    "univariate_slope": slope,
                    "correlation_with_abs_primary_residual": corr,
                }
            )
    return pd.DataFrame(rows)


def build_point_candidate_rolling_scores(consensus_panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    candidates = point_candidate_shortlist(consensus_panel)
    for subset_name, subset in [("all", consensus_panel), *consensus_panel.groupby("evaluation_subset", dropna=False)]:
        for origin_day, origin_df in subset.groupby("origin_day"):
            for method in candidates:
                row = evaluate_forecast(origin_df, method)
                row["candidate_method"] = method
                row["evaluation_subset"] = str(subset_name)
                row["origin_day"] = int(origin_day)
                rows.append(row)
    return pd.DataFrame(rows)


def build_range_interval_candidate_rolling_scores(interval_metrics: pd.DataFrame) -> pd.DataFrame:
    out = interval_metrics.copy()
    out["uses_range_feature"] = out["interval_model"].astype(str).str.contains("range|disagreement|source_count", regex=True)
    return out


def build_full_candidate_stability_summary(point_scores: pd.DataFrame, interval_scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    test_point = point_scores.loc[point_scores["evaluation_subset"].eq("test")].copy()
    for method, group in test_point.groupby("candidate_method"):
        rows.append(
            {
                "candidate_family": "point",
                "candidate": method,
                "n_origin_slices": int(group["origin_day"].nunique()),
                "median_MAE_log": float(group["MAE_log"].median()),
                "worst_MAE_log": float(group["MAE_log"].max()),
                "median_RMSE_log": float(group["RMSE_log"].median()),
            }
        )
    test_interval = interval_scores.loc[interval_scores["evaluation_subset"].eq("test")].copy()
    for method, group in test_interval.groupby("interval_model"):
        rows.append(
            {
                "candidate_family": "interval",
                "candidate": method,
                "n_origin_slices": int(group["origin_day"].nunique()),
                "median_coverage_80": float(group["coverage_80"].median()),
                "median_coverage_95": float(group["coverage_95"].median()),
                "median_interval_score_95_pct": float(group["mean_interval_score_95_pct"].median()),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    source_panel: pd.DataFrame,
    target_alignment_audit: pd.DataFrame,
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
    source_data_coverage_audit: pd.DataFrame,
    source_individual_performance: pd.DataFrame,
    source_common_support_paired_scores: pd.DataFrame,
    source_error_correlation: pd.DataFrame,
    source_leave_one_in_out_scores: pd.DataFrame,
    source_range_calibration: pd.DataFrame,
    source_range_resolution: pd.DataFrame,
    range_feature_effect_rolling: pd.DataFrame,
    point_candidate_rolling_scores: pd.DataFrame,
    range_interval_candidate_rolling_scores: pd.DataFrame,
    full_candidate_stability_summary: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "source": output_dir / "rolling_forecast_origin_source_estimates.csv",
        "target_alignment_audit": output_dir / "estimate_target_alignment_audit.csv",
        "target_alignment_exclusions": output_dir / "estimate_target_alignment_exclusions.csv",
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
        "source_data_coverage_audit": output_dir / "source_data_coverage_audit.csv",
        "source_individual_performance": output_dir / "source_individual_performance.csv",
        "source_common_support_paired_scores": output_dir / "source_common_support_paired_scores.csv",
        "source_error_correlation": output_dir / "source_error_correlation.csv",
        "source_leave_one_in_out_scores": output_dir / "source_leave_one_in_out_scores.csv",
        "source_range_calibration": output_dir / "source_range_calibration.csv",
        "source_range_resolution": output_dir / "source_range_resolution.csv",
        "range_feature_effect_rolling": output_dir / "range_feature_effect_rolling.csv",
        "point_candidate_rolling_scores": output_dir / "point_candidate_rolling_scores.csv",
        "range_interval_candidate_rolling_scores": output_dir / "range_interval_candidate_rolling_scores.csv",
        "full_candidate_stability_summary": output_dir / "full_candidate_stability_summary.csv",
    }
    source_panel.sort_values(["opening_weekend_start", "release_run_id", "origin_day", "estimate_source"]).to_csv(
        paths["source"], index=False
    )
    target_alignment_audit.to_csv(paths["target_alignment_audit"], index=False)
    target_alignment_audit.loc[~target_alignment_audit["included_in_calibration"].astype(bool)].to_csv(
        paths["target_alignment_exclusions"], index=False
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
    source_data_coverage_audit.sort_values("estimate_source").to_csv(paths["source_data_coverage_audit"], index=False)
    write_sorted_metrics(
        source_individual_performance,
        paths["source_individual_performance"],
        ["estimate_source", "segment", "_origin_day_order", "release_year"],
    )
    write_sorted_metrics(
        source_common_support_paired_scores,
        paths["source_common_support_paired_scores"],
        ["estimate_source", "_origin_day_order"],
    )
    write_sorted_metrics(
        source_error_correlation,
        paths["source_error_correlation"],
        ["_origin_day_order", "source_a", "source_b"],
    )
    write_sorted_metrics(
        source_leave_one_in_out_scores,
        paths["source_leave_one_in_out_scores"],
        ["estimate_source", "scenario", "_origin_day_order", "forecast_method"],
    )
    write_sorted_metrics(
        source_range_calibration,
        paths["source_range_calibration"],
        ["estimate_source", "segment", "_origin_day_order", "release_year"],
    )
    write_sorted_metrics(
        source_range_resolution,
        paths["source_range_resolution"],
        ["estimate_source", "_origin_day_order"],
    )
    write_sorted_metrics(
        range_feature_effect_rolling,
        paths["range_feature_effect_rolling"],
        ["_origin_day_order", "feature"],
    )
    write_sorted_metrics(
        point_candidate_rolling_scores,
        paths["point_candidate_rolling_scores"],
        ["evaluation_subset", "_origin_day_order", "candidate_method"],
    )
    write_sorted_metrics(
        range_interval_candidate_rolling_scores,
        paths["range_interval_candidate_rolling_scores"],
        ["evaluation_subset", "_origin_day_order", "interval_model"],
    )
    full_candidate_stability_summary.sort_values(["candidate_family", "candidate"]).to_csv(
        paths["full_candidate_stability_summary"], index=False
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
    source_panel, target_alignment_audit = build_source_origin_panel(
        openings,
        estimates,
        config,
        return_alignment_audit=True,
    )
    source_panel = add_source_reliability(source_panel, config.min_source_reliability_n)
    source_panel = add_source_bias_calibration(
        source_panel,
        config.min_source_reliability_n,
        config.source_bias_shrink_k,
    )
    source_panel = add_source_bias_adjusted_reliability(source_panel, config.min_source_reliability_n)
    consensus_panel = build_consensus_panel(source_panel, config)
    consensus_panel = add_range_features_to_consensus_panel(consensus_panel, source_panel)
    metrics = build_metrics(consensus_panel)
    train_selected_metrics = build_train_selected_metrics(consensus_panel)
    paired_deltas = build_paired_deltas(consensus_panel)

    point_policy = build_point_model_policy(
        consensus_panel,
        min_train_n=config.min_train_n_for_model_selection,
    )
    selected_point_panel = add_selected_point_forecast(consensus_panel, point_policy)
    selected_point_metrics = build_selected_point_metrics(selected_point_panel)
    selected_daily_point_panel = selected_point_panel.copy()
    selected_daily_point_panel[DEFAULT_PRIMARY_POINT_COL] = selected_daily_point_panel[DEFAULT_SELECTED_POINT_COL]
    selected_daily_point_panel["primary_point_method"] = selected_daily_point_panel["selected_point_method"]
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
        selected_daily_point_panel,
        DEFAULT_PRIMARY_POINT_COL,
        shrink_k=config.interval_shrink_k,
    )
    interval_panel = add_interval_calibration_features(interval_panel)
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
    interval_panel = add_empirical_quantile_intervals(
        interval_panel,
        DEFAULT_PRIMARY_POINT_COL,
        group_cols=["origin_day", "range_width_bucket", "disagreement_bucket"],
        interval_model="empirical_origin_range_disagreement_quantile_shrunk",
        shrink_k=config.interval_shrink_k,
    )
    interval_panel = add_centered_empirical_quantile_intervals(
        interval_panel,
        DEFAULT_PRIMARY_POINT_COL,
        group_cols=["origin_bucket", "source_count_bucket", "point_bucket"],
        interval_model="empirical_origin_bucket_source_count_point_bucket_centered_quantile_shrunk",
        shrink_k=config.interval_shrink_k,
    )
    interval_metrics = build_interval_metrics(interval_panel, DEFAULT_PRIMARY_POINT_COL)
    interval_policy = build_interval_model_policy(
        interval_metrics,
        min_train_n=config.min_train_n_for_model_selection,
    )
    interval_panel = add_selected_intervals(
        interval_panel,
        interval_policy,
        forecast_col=DEFAULT_PRIMARY_POINT_COL,
    )
    selected_interval_metrics = build_selected_interval_metrics(
        interval_panel,
        forecast_col=DEFAULT_PRIMARY_POINT_COL,
    )
    source_data_coverage_audit = build_source_data_coverage_audit(source_panel, target_alignment_audit, openings)
    source_individual_performance = build_source_individual_performance(source_panel)
    source_common_support_paired_scores = build_source_common_support_paired_scores(source_panel, locked_point_panel)
    source_error_correlation = build_source_error_correlation(source_panel)
    source_leave_one_in_out_scores = build_source_leave_one_in_out_scores(source_panel, config)
    source_range_calibration = build_source_range_calibration(source_panel)
    source_range_resolution = build_source_range_resolution(source_panel)
    range_feature_effect_rolling = build_range_feature_effect_rolling(selected_daily_point_panel)
    point_candidate_rolling_scores = build_point_candidate_rolling_scores(consensus_panel)
    range_interval_candidate_rolling_scores = build_range_interval_candidate_rolling_scores(interval_metrics)
    full_candidate_stability_summary = build_full_candidate_stability_summary(
        point_candidate_rolling_scores,
        range_interval_candidate_rolling_scores,
    )
    paths = write_outputs(
        source_panel,
        target_alignment_audit,
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
        source_data_coverage_audit,
        source_individual_performance,
        source_common_support_paired_scores,
        source_error_correlation,
        source_leave_one_in_out_scores,
        source_range_calibration,
        source_range_resolution,
        range_feature_effect_rolling,
        point_candidate_rolling_scores,
        range_interval_candidate_rolling_scores,
        full_candidate_stability_summary,
        args.output_dir,
    )

    print(f"Wrote {paths['source']} ({len(source_panel):,} rows)")
    print(f"Wrote {paths['target_alignment_audit']} ({len(target_alignment_audit):,} rows)")
    print(
        f"Wrote {paths['target_alignment_exclusions']} "
        f"({int((~target_alignment_audit['included_in_calibration'].astype(bool)).sum()):,} rows)"
    )
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
    print(f"Wrote {paths['source_data_coverage_audit']} ({len(source_data_coverage_audit):,} rows)")
    print(f"Wrote {paths['source_individual_performance']} ({len(source_individual_performance):,} rows)")
    print(f"Wrote {paths['source_common_support_paired_scores']} ({len(source_common_support_paired_scores):,} rows)")
    print(f"Wrote {paths['source_error_correlation']} ({len(source_error_correlation):,} rows)")
    print(f"Wrote {paths['source_leave_one_in_out_scores']} ({len(source_leave_one_in_out_scores):,} rows)")
    print(f"Wrote {paths['source_range_calibration']} ({len(source_range_calibration):,} rows)")
    print(f"Wrote {paths['source_range_resolution']} ({len(source_range_resolution):,} rows)")
    print(f"Wrote {paths['range_feature_effect_rolling']} ({len(range_feature_effect_rolling):,} rows)")
    print(f"Wrote {paths['point_candidate_rolling_scores']} ({len(point_candidate_rolling_scores):,} rows)")
    print(f"Wrote {paths['range_interval_candidate_rolling_scores']} ({len(range_interval_candidate_rolling_scores):,} rows)")
    print(f"Wrote {paths['full_candidate_stability_summary']} ({len(full_candidate_stability_summary):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
