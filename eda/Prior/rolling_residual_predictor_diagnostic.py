#!/usr/bin/env python3
"""Rolling-origin residual-predictor diagnostic for consensus forecasts.

This script keeps the production-style consensus baseline locked, then asks
whether a small set of origin-safe predictors explains the remaining log error:

    log(actual opening weekend) - log(locked consensus forecast)

The first diagnostic pass deliberately stays small:

* M0: no point correction.
* M1: origin-specific standardized Wiki rigor residual correction.
* M2: pooled Wiki rigor residual correction plus a late-origin interaction.
* M3: franchise-group residual correction.
* M4: source-dispersion residual correction.
* V1: franchise-conditioned residual scale.
* V2: estimate-dispersion-conditioned residual scale.

Outputs are written under ``data/diagnostics`` by default.
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

import rolling_forecast_consensus_benchmark as benchmark


REPO_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS_DIR = REPO_ROOT / "data" / "diagnostics"

DEFAULT_BASELINE_POINT_COL = benchmark.DEFAULT_PRIMARY_POINT_COL
DEFAULT_OUTPUT_PREFIX = "rolling_residual_predictor"
DEFAULT_MIN_TRAIN_N = 25
DEFAULT_RIDGE_ALPHA = 1.0
DEFAULT_INTERVAL_SHRINK_K = benchmark.DEFAULT_INTERVAL_SHRINK_K
DEFAULT_ORIGIN_DAYS = benchmark.DEFAULT_ORIGIN_DAYS
DEFAULT_TRAIN_YEARS = benchmark.DEFAULT_TRAIN_YEARS
DEFAULT_TEST_START_YEAR = benchmark.DEFAULT_TEST_START_YEAR
DEFAULT_EXCLUDED_ESTIMATE_SOURCES = benchmark.DEFAULT_EXCLUDED_ESTIMATE_SOURCES
DEFAULT_EXCLUDED_RELEASE_YEARS = benchmark.DEFAULT_EXCLUDED_RELEASE_YEARS
DEFAULT_RECENCY_LAMBDAS = benchmark.DEFAULT_RECENCY_LAMBDAS
DEFAULT_MAX_SOURCE_AGE_DAYS = benchmark.DEFAULT_MAX_SOURCE_AGE_DAYS
INTERVAL_LEVELS = (80, 95)
Z_BY_LEVEL = {80: 1.28155, 95: 1.95996}
BASELINE_INTERVAL_MODEL = "V0_origin_sigma"
PRODUCTION_INTERVAL_MODEL = "V2_dispersion_scale"
SHADOW_INTERVAL_MODEL = "V1_franchise_scale"
PRODUCTION_INTERVAL_COL_PREFIX = "production_candidate_interval"
SHADOW_INTERVAL_COL_PREFIX = "shadow_candidate_interval"
LOWER_TAIL_GRID = (
    ("V2_symmetric_95", 1.95996, 1.95996),
    ("V2_lower_tail_plus_5pct", 2.06, 1.95996),
    ("V2_lower_tail_plus_10pct", 2.16, 1.95996),
    ("V2_lower_tail_plus_15pct", 2.25, 1.95996),
    ("V2_lower_tail_plus_20pct", 2.35, 1.95996),
)
LEADING_LOWER_TAIL_CANDIDATE = "V2_lower_tail_plus_15pct"
GUARDRAIL_CANDIDATES = (
    "V2_lower_15",
    "G1_single_stale_lower_v0",
    "G2_single_stale_nonfranchise_lower_v0",
    "G3_single_stale_full_v0",
    "G4_single_stale_lower_3sigma",
)
STALE_SINGLE_SOURCE_MIN_AGE_DAYS = 5


@dataclass(frozen=True)
class ResidualDiagnosticConfig:
    benchmark_config: benchmark.BenchmarkConfig
    baseline_point_col: str
    min_train_n: int
    ridge_alpha: float
    interval_shrink_k: int
    output_prefix: str


def fetch_frame(conn: Any, sql: str, params: tuple[Any, ...] | None = None) -> pd.DataFrame:
    cursor = conn.execute(sql, params or ())
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def safe_log(values: pd.Series | np.ndarray) -> pd.Series:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    return np.log(numeric.where(numeric > 0))


def clean_numeric(values: pd.Series | np.ndarray) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan)


def pct_improvement(baseline: float, candidate: float) -> float:
    if not np.isfinite(baseline) or baseline == 0 or not np.isfinite(candidate):
        return np.nan
    return float((baseline - candidate) / baseline)


def fetch_wiki_origin_features(
    database_url: str | None,
    origin_days: tuple[int, ...],
) -> pd.DataFrame:
    """Fetch cumulative Wiki variables available by each forecast origin.

    The feature window starts 30 days before local opening date and ends at the
    forecast origin date, inclusive. This mirrors the exploratory paper-style
    Wiki variables while preserving forecast-origin availability.
    """
    if not origin_days:
        return pd.DataFrame()

    values_sql = ", ".join(["(%s)"] * len(origin_days))
    sql = f"""
        WITH forecast_origins(origin_day) AS (
            VALUES {values_sql}
        ), wiki_matches AS (
            SELECT
                src.movie_id,
                split_part(src.source_movie_id, ':', 1) AS language,
                split_part(src.source_movie_id, ':', 2)::integer AS wiki_page_id
            FROM movie_source_ids src
            WHERE src.source = 'wikipedia'
              AND src.match_status IN ('matched', 'manual_override')
              AND src.source_movie_id ~ '^[a-z-]+:[0-9]+$'
        ), base AS (
            SELECT
                o.release_run_id,
                o.movie_id,
                o.opening_date::date AS opening_date,
                o.opening_weekend_start::date AS opening_weekend_start,
                fo.origin_day,
                wm.language,
                wm.wiki_page_id,
                (o.opening_weekend_start::date + (fo.origin_day * INTERVAL '1 day'))::date AS feature_end_date
            FROM analytics.eda_movie_openings o
            JOIN wiki_matches wm ON wm.movie_id = o.movie_id
            CROSS JOIN forecast_origins fo
            WHERE o.opening_weekend_gross_usd > 0
        ), pageviews AS (
            SELECT
                b.release_run_id,
                b.origin_day,
                SUM(pv.views)::numeric AS wiki_views
            FROM base b
            LEFT JOIN wiki_pageviews_daily pv
              ON pv.language = b.language
             AND pv.wiki_page_id = b.wiki_page_id
             AND pv.view_date::date BETWEEN b.opening_date - INTERVAL '30 days'
                                      AND b.feature_end_date
            GROUP BY b.release_run_id, b.origin_day
        ), human_revisions AS (
            SELECT
                wr.language,
                wr.wiki_page_id,
                wr.rev_id,
                wr.rev_timestamp,
                wr.rev_date::date AS rev_date,
                wr.user_key,
                CASE
                    WHEN LAG(wr.user_key) OVER (
                        PARTITION BY wr.language, wr.wiki_page_id
                        ORDER BY wr.rev_timestamp, wr.rev_id
                    ) IS NULL THEN 1
                    WHEN LAG(wr.user_key) OVER (
                        PARTITION BY wr.language, wr.wiki_page_id
                        ORDER BY wr.rev_timestamp, wr.rev_id
                    ) <> wr.user_key THEN 1
                    ELSE 0
                END AS rigor_increment
            FROM wiki_revisions wr
            JOIN (SELECT DISTINCT language, wiki_page_id FROM base) pages
              ON pages.language = wr.language
             AND pages.wiki_page_id = wr.wiki_page_id
            WHERE wr.is_bot = 0
        ), revisions AS (
            SELECT
                b.release_run_id,
                b.origin_day,
                COUNT(hr.rev_id)::numeric AS wiki_edits,
                COUNT(DISTINCT hr.user_key)::numeric AS wiki_unique_editors,
                COALESCE(SUM(hr.rigor_increment), 0)::numeric AS wiki_rigor
            FROM base b
            LEFT JOIN human_revisions hr
              ON hr.language = b.language
             AND hr.wiki_page_id = b.wiki_page_id
             AND hr.rev_date BETWEEN b.opening_date - INTERVAL '30 days'
                                 AND b.feature_end_date
            GROUP BY b.release_run_id, b.origin_day
        )
        SELECT
            b.release_run_id,
            b.movie_id,
            b.origin_day,
            MAX(b.feature_end_date) AS wiki_feature_end_date,
            COALESCE(MAX(p.wiki_views), 0) AS wiki_views,
            COALESCE(MAX(r.wiki_edits), 0) AS wiki_edits,
            COALESCE(MAX(r.wiki_unique_editors), 0) AS wiki_unique_editors,
            COALESCE(MAX(r.wiki_rigor), 0) AS wiki_rigor
        FROM base b
        LEFT JOIN pageviews p
          ON p.release_run_id = b.release_run_id
         AND p.origin_day = b.origin_day
        LEFT JOIN revisions r
          ON r.release_run_id = b.release_run_id
         AND r.origin_day = b.origin_day
        GROUP BY b.release_run_id, b.movie_id, b.origin_day
    """
    conn = connect_database(database_url)
    try:
        wiki = fetch_frame(conn, sql, tuple(origin_days))
    finally:
        conn.close()

    if wiki.empty:
        return wiki
    for col in ["wiki_views", "wiki_edits", "wiki_unique_editors", "wiki_rigor"]:
        wiki[col] = pd.to_numeric(wiki[col], errors="coerce").fillna(0).clip(lower=0)
        wiki[f"log_{col}"] = np.log1p(wiki[col])
    wiki["wiki_feature_end_date"] = pd.to_datetime(wiki["wiki_feature_end_date"], errors="coerce")
    return wiki


def build_consensus_panel(database_url: str | None, config: benchmark.BenchmarkConfig) -> pd.DataFrame:
    openings, estimates = benchmark.load_inputs(database_url)
    source_panel = benchmark.build_source_origin_panel(openings, estimates, config)
    source_panel = benchmark.add_source_reliability(source_panel, config.min_source_reliability_n)
    source_panel = benchmark.add_source_bias_calibration(
        source_panel,
        config.min_source_reliability_n,
        config.source_bias_shrink_k,
    )
    source_panel = benchmark.add_source_bias_adjusted_reliability(
        source_panel,
        config.min_source_reliability_n,
    )
    consensus = benchmark.build_consensus_panel(source_panel, config)
    consensus = benchmark.add_locked_point_forecasts(consensus)
    return consensus


def add_residual_features(
    consensus: pd.DataFrame,
    wiki: pd.DataFrame,
    baseline_point_col: str,
) -> pd.DataFrame:
    panel = consensus.merge(
        wiki,
        on=["release_run_id", "movie_id", "origin_day"],
        how="left",
    )
    for col in ["wiki_views", "wiki_edits", "wiki_unique_editors", "wiki_rigor"]:
        panel[col] = pd.to_numeric(panel.get(col), errors="coerce").fillna(0).clip(lower=0)
        panel[f"log_{col}"] = np.log1p(panel[col])

    baseline = pd.to_numeric(panel[baseline_point_col], errors="coerce")
    actual = pd.to_numeric(panel["actual_opening_weekend_gross_usd"], errors="coerce")
    panel["log_baseline_forecast"] = safe_log(baseline)
    panel["log_actual_opening_weekend"] = safe_log(actual)
    panel["baseline_residual_log"] = panel["log_actual_opening_weekend"] - panel["log_baseline_forecast"]
    panel["franchise_group"] = np.where(
        panel["is_franchise"].fillna(False).astype(bool),
        "Franchise",
        "NonFranchise",
    )
    panel["franchise_flag"] = panel["franchise_group"].eq("Franchise").astype(float)
    panel["non_franchise_flag"] = 1.0 - panel["franchise_flag"]
    panel["late_origin_flag"] = panel["origin_day"].isin([-2, -1]).astype(float)
    panel["source_count_numeric"] = pd.to_numeric(panel["source_count"], errors="coerce")

    log_estimate_sd = panel.get("estimate_mid_stddev_usd")
    panel["estimate_log_sd"] = np.nan
    if log_estimate_sd is not None:
        panel["estimate_log_sd"] = clean_numeric(log_estimate_sd)
    panel["estimate_log_range"] = (
        np.log(pd.to_numeric(panel["max_estimate_mid_usd"], errors="coerce"))
        - np.log(pd.to_numeric(panel["min_estimate_mid_usd"], errors="coerce"))
    )
    panel["estimate_dispersion"] = clean_numeric(panel["log_dispersion"]).fillna(panel["estimate_log_range"])
    panel["estimate_dispersion"] = panel["estimate_dispersion"].replace([np.inf, -np.inf], np.nan).fillna(0)

    return panel.sort_values(["opening_weekend_start", "release_run_id", "origin_day"]).reset_index(drop=True)


def train_prior_rows(panel: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    train = panel.loc[
        panel["evaluation_subset"].eq("train")
        & panel["opening_weekend_start"].lt(row["opening_weekend_start"])
    ].copy()
    return train


def rolling_bin_for_value(train_values: pd.Series, row_value: float) -> str:
    values = clean_numeric(train_values).dropna()
    row_numeric = pd.to_numeric(pd.Series([row_value]), errors="coerce").iloc[0]
    if len(values) < 25 or values.nunique() < 3 or not np.isfinite(row_numeric):
        return "Unknown"

    q1, q2 = values.quantile([1 / 3, 2 / 3])
    if row_numeric <= q1:
        return "Low"
    if row_numeric <= q2:
        return "Medium"
    return "High"


def add_rolling_feature_states(panel: pd.DataFrame, min_train_n: int) -> pd.DataFrame:
    out = panel.copy()
    out["dispersion_state_rolling"] = "Unknown"
    out["wiki_rigor_state_rolling"] = "Unknown"

    for idx, row in out.iterrows():
        prior_train = train_prior_rows(out, row)
        origin_train = prior_train.loc[prior_train["origin_day"].eq(row["origin_day"])]

        source_count = pd.to_numeric(pd.Series([row.get("source_count")]), errors="coerce").iloc[0]
        if source_count == 1:
            out.loc[idx, "dispersion_state_rolling"] = "SingleSource"
        else:
            multi_source_train = origin_train.loc[
                pd.to_numeric(origin_train["source_count"], errors="coerce").gt(1)
            ]
            disp_bin = rolling_bin_for_value(
                multi_source_train["estimate_dispersion"],
                row["estimate_dispersion"],
            )
            out.loc[idx, "dispersion_state_rolling"] = f"{disp_bin}Dispersion"

        wiki_bin = rolling_bin_for_value(origin_train["log_wiki_rigor"], row["log_wiki_rigor"])
        out.loc[idx, "wiki_rigor_state_rolling"] = f"{wiki_bin}WikiRigor"

    return out


def fit_ridge_predict(
    train: pd.DataFrame,
    row: pd.Series,
    target_col: str,
    feature_cols: list[str],
    min_train_n: int,
    ridge_alpha: float,
) -> tuple[float, int]:
    cols = [target_col] + feature_cols
    clean = train[cols].replace([np.inf, -np.inf], np.nan).dropna()
    row_x = row[feature_cols].replace([np.inf, -np.inf], np.nan)
    if len(clean) < min_train_n or row_x.isna().any():
        return 0.0, int(len(clean))

    x = clean[feature_cols].to_numpy(dtype="float64")
    y = clean[target_col].to_numpy(dtype="float64")
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    design = np.column_stack([np.ones(len(x)), (x - mu) / sd])
    penalty = np.eye(design.shape[1]) * ridge_alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    row_z = (row_x.to_numpy(dtype="float64") - mu) / sd
    return float(np.r_[1.0, row_z] @ coef), int(len(clean))


def fit_group_mean_predict(
    train: pd.DataFrame,
    row: pd.Series,
    group_col: str,
    target_col: str,
    min_train_n: int,
    shrink_k: int,
) -> tuple[float, int]:
    clean = train[[group_col, target_col]].dropna()
    if len(clean) < min_train_n:
        return 0.0, int(len(clean))
    global_mean = float(clean[target_col].mean())
    group_value = row[group_col]
    group = clean.loc[clean[group_col].eq(group_value), target_col]
    if group.empty:
        return global_mean, int(len(clean))
    shrink = len(group) / (len(group) + shrink_k)
    return float(shrink * group.mean() + (1.0 - shrink) * global_mean), int(len(clean))


def add_point_residual_predictions(panel: pd.DataFrame, config: ResidualDiagnosticConfig) -> pd.DataFrame:
    out = panel.copy()
    model_specs = {
        "M0_locked_baseline": [],
        "M1_wiki_log_rigor_origin_specific": ["log_wiki_rigor"],
        "M2_wiki_log_rigor_late_interaction": ["log_wiki_rigor", "log_wiki_rigor_late"],
        "M3_franchise_group": ["franchise_flag", "non_franchise_flag"],
        "M4_estimate_dispersion": ["estimate_dispersion", "source_count_numeric"],
    }
    out["log_wiki_rigor_late"] = out["log_wiki_rigor"] * out["late_origin_flag"]

    for model_name in model_specs:
        out[f"{model_name}_predicted_residual_log"] = 0.0
        out[f"{model_name}_train_n"] = 0
        out[f"{model_name}_forecast_usd"] = pd.to_numeric(out[config.baseline_point_col], errors="coerce")

    for idx, row in out.iterrows():
        origin_day = int(row["origin_day"])
        prior_train = train_prior_rows(out, row)
        origin_train = prior_train.loc[prior_train["origin_day"].eq(origin_day)]

        out.loc[idx, "M0_locked_baseline_train_n"] = int(len(origin_train.dropna(subset=["baseline_residual_log"])))

        pred, train_n = fit_ridge_predict(
            origin_train,
            row,
            "baseline_residual_log",
            model_specs["M1_wiki_log_rigor_origin_specific"],
            config.min_train_n,
            config.ridge_alpha,
        )
        out.loc[idx, "M1_wiki_log_rigor_origin_specific_predicted_residual_log"] = pred
        out.loc[idx, "M1_wiki_log_rigor_origin_specific_train_n"] = train_n

        pred, train_n = fit_ridge_predict(
            prior_train,
            row,
            "baseline_residual_log",
            model_specs["M2_wiki_log_rigor_late_interaction"],
            config.min_train_n,
            config.ridge_alpha,
        )
        out.loc[idx, "M2_wiki_log_rigor_late_interaction_predicted_residual_log"] = pred
        out.loc[idx, "M2_wiki_log_rigor_late_interaction_train_n"] = train_n

        pred, train_n = fit_group_mean_predict(
            origin_train,
            row,
            "franchise_group",
            "baseline_residual_log",
            config.min_train_n,
            config.interval_shrink_k,
        )
        out.loc[idx, "M3_franchise_group_predicted_residual_log"] = pred
        out.loc[idx, "M3_franchise_group_train_n"] = train_n

        pred, train_n = fit_group_mean_predict(
            origin_train,
            row,
            "dispersion_state_rolling",
            "baseline_residual_log",
            config.min_train_n,
            config.interval_shrink_k,
        )
        out.loc[idx, "M4_estimate_dispersion_predicted_residual_log"] = pred
        out.loc[idx, "M4_estimate_dispersion_train_n"] = train_n

    baseline = pd.to_numeric(out[config.baseline_point_col], errors="coerce")
    for model_name in model_specs:
        residual_hat = out[f"{model_name}_predicted_residual_log"]
        out[f"{model_name}_forecast_usd"] = baseline * np.exp(residual_hat)
        out[f"{model_name}_post_correction_residual_log"] = (
            out["log_actual_opening_weekend"] - safe_log(out[f"{model_name}_forecast_usd"])
        )
    return out


def evaluate_point_forecast(df: pd.DataFrame, forecast_col: str) -> dict[str, float | int | str]:
    x = df.copy()
    actual = pd.to_numeric(x["actual_opening_weekend_gross_usd"], errors="coerce")
    forecast = pd.to_numeric(x[forecast_col], errors="coerce")
    valid = actual.gt(0) & forecast.gt(0)
    if not valid.any():
        return {"point_model": forecast_col, "n": 0, "ME_log": np.nan, "MAE_log": np.nan, "RMSE_log": np.nan, "MdAPE": np.nan}
    residual = np.log(actual.loc[valid]) - np.log(forecast.loc[valid])
    return {
        "point_model": forecast_col.removesuffix("_forecast_usd"),
        "n": int(valid.sum()),
        "ME_log": float(residual.mean()),
        "MAE_log": float(residual.abs().mean()),
        "RMSE_log": float(np.sqrt(np.mean(residual**2))),
        "MdAPE": float(np.median(np.abs(actual.loc[valid] / forecast.loc[valid] - 1))),
    }


def build_point_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    forecast_cols = [col for col in panel.columns if col.startswith("M") and col.endswith("_forecast_usd")]
    rows: list[dict[str, float | int | str]] = []
    subsets: list[tuple[str, pd.DataFrame]] = [("all", panel)]
    subsets.extend((name, frame) for name, frame in panel.groupby("evaluation_subset", dropna=False))

    for subset_name, subset in subsets:
        for origin_day, origin_df in subset.groupby("origin_day"):
            for forecast_col in forecast_cols:
                row = evaluate_point_forecast(origin_df, forecast_col)
                row["evaluation_subset"] = str(subset_name)
                row["origin_day"] = int(origin_day)
                rows.append(row)
        for forecast_col in forecast_cols:
            row = evaluate_point_forecast(subset, forecast_col)
            row["evaluation_subset"] = str(subset_name)
            row["origin_day"] = "pooled"
            rows.append(row)
    metrics = pd.DataFrame(rows)
    return add_baseline_lift(metrics)


def add_baseline_lift(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    baseline = out.loc[out["point_model"].eq("M0_locked_baseline")][
        ["evaluation_subset", "origin_day", "MAE_log", "MdAPE", "RMSE_log"]
    ].rename(
        columns={
            "MAE_log": "baseline_MAE_log",
            "MdAPE": "baseline_MdAPE",
            "RMSE_log": "baseline_RMSE_log",
        }
    )
    out = out.merge(baseline, on=["evaluation_subset", "origin_day"], how="left")
    out["MAE_log_lift_vs_M0"] = out.apply(lambda r: pct_improvement(r["baseline_MAE_log"], r["MAE_log"]), axis=1)
    out["MdAPE_lift_vs_M0"] = out.apply(lambda r: pct_improvement(r["baseline_MdAPE"], r["MdAPE"]), axis=1)
    out["RMSE_log_lift_vs_M0"] = out.apply(lambda r: pct_improvement(r["baseline_RMSE_log"], r["RMSE_log"]), axis=1)
    return out


def interval_score(actual: pd.Series, lo: pd.Series, hi: pd.Series, alpha: float) -> pd.Series:
    width = hi - lo
    below = actual < lo
    above = actual > hi
    return width + (2.0 / alpha) * (lo - actual) * below + (2.0 / alpha) * (actual - hi) * above


def rolling_sigma_for_row(
    train: pd.DataFrame,
    row: pd.Series,
    residual_col: str,
    group_col: str | None,
    min_train_n: int,
    shrink_k: int,
) -> tuple[float, int, int]:
    clean = train[[residual_col] + ([group_col] if group_col else [])].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < min_train_n:
        return np.nan, int(len(clean)), 0
    global_sigma = clean[residual_col].std(ddof=1)
    if not group_col:
        return float(global_sigma), int(len(clean)), int(len(clean))
    group = clean.loc[clean[group_col].eq(row[group_col]), residual_col]
    if len(group) < 2 or not np.isfinite(group.std(ddof=1)):
        return float(global_sigma), int(len(clean)), int(len(group))
    shrink = len(group) / (len(group) + shrink_k)
    sigma = shrink * group.std(ddof=1) + (1.0 - shrink) * global_sigma
    return float(sigma), int(len(clean)), int(len(group))


def add_scale_intervals(
    panel: pd.DataFrame,
    point_model: str = "M0_locked_baseline",
    min_train_n: int = DEFAULT_MIN_TRAIN_N,
    shrink_k: int = DEFAULT_INTERVAL_SHRINK_K,
) -> pd.DataFrame:
    out = panel.copy()
    forecast_col = f"{point_model}_forecast_usd"
    residual_col = f"{point_model}_post_correction_residual_log"
    scale_specs = {
        "V0_origin_sigma": None,
        "V1_franchise_scale": "franchise_group",
        "V2_dispersion_scale": "dispersion_state_rolling",
        "V3_wiki_rigor_scale": "wiki_rigor_state_rolling",
    }
    for scale_model in scale_specs:
        out[f"{scale_model}_sigma"] = np.nan
        out[f"{scale_model}_train_n"] = 0
        out[f"{scale_model}_group_n"] = 0

    for idx, row in out.iterrows():
        prior_train = train_prior_rows(out, row)
        origin_train = prior_train.loc[prior_train["origin_day"].eq(row["origin_day"])]
        for scale_model, group_col in scale_specs.items():
            sigma, train_n, group_n = rolling_sigma_for_row(
                origin_train,
                row,
                residual_col,
                group_col,
                min_train_n,
                shrink_k,
            )
            out.loc[idx, f"{scale_model}_sigma"] = sigma
            out.loc[idx, f"{scale_model}_train_n"] = train_n
            out.loc[idx, f"{scale_model}_group_n"] = group_n

    log_forecast = safe_log(out[forecast_col])
    for scale_model in scale_specs:
        for level in INTERVAL_LEVELS:
            z = Z_BY_LEVEL[level]
            sigma = out[f"{scale_model}_sigma"]
            out[f"{point_model}_{scale_model}_lo_{level}"] = np.exp(log_forecast - z * sigma)
            out[f"{point_model}_{scale_model}_hi_{level}"] = np.exp(log_forecast + z * sigma)
    return out


def evaluate_interval(df: pd.DataFrame, point_model: str, scale_model: str) -> dict[str, float | int | str]:
    forecast_col = f"{point_model}_forecast_usd"
    actual = pd.to_numeric(df["actual_opening_weekend_gross_usd"], errors="coerce")
    forecast = pd.to_numeric(df[forecast_col], errors="coerce")
    valid = actual.gt(0) & forecast.gt(0)
    for level in INTERVAL_LEVELS:
        valid &= pd.to_numeric(df[f"{point_model}_{scale_model}_lo_{level}"], errors="coerce").gt(0)
        valid &= pd.to_numeric(df[f"{point_model}_{scale_model}_hi_{level}"], errors="coerce").gt(0)
    if not valid.any():
        return {
            "point_model": point_model,
            "scale_model": scale_model,
            "n": 0,
            "coverage_80": np.nan,
            "coverage_95": np.nan,
            "median_width_80_pct": np.nan,
            "median_width_95_pct": np.nan,
            "mean_interval_score_80_pct": np.nan,
            "mean_interval_score_95_pct": np.nan,
        }

    rows = df.loc[valid].copy()
    actual_v = actual.loc[valid]
    forecast_v = forecast.loc[valid]
    out: dict[str, float | int | str] = {"point_model": point_model, "scale_model": scale_model, "n": int(valid.sum())}
    for level in INTERVAL_LEVELS:
        lo = pd.to_numeric(rows[f"{point_model}_{scale_model}_lo_{level}"], errors="coerce")
        hi = pd.to_numeric(rows[f"{point_model}_{scale_model}_hi_{level}"], errors="coerce")
        width_pct = (hi - lo) / forecast_v
        score = interval_score(actual_v, lo, hi, alpha=1.0 - level / 100.0) / forecast_v
        out[f"coverage_{level}"] = float(((actual_v >= lo) & (actual_v <= hi)).mean())
        out[f"median_width_{level}_pct"] = float(width_pct.median())
        out[f"mean_interval_score_{level}_pct"] = float(score.mean())
    return out


def build_interval_metrics(panel: pd.DataFrame, point_model: str = "M0_locked_baseline") -> pd.DataFrame:
    scale_models = ["V0_origin_sigma", "V1_franchise_scale", "V2_dispersion_scale", "V3_wiki_rigor_scale"]
    rows: list[dict[str, float | int | str]] = []
    subsets: list[tuple[str, pd.DataFrame]] = [("all", panel)]
    subsets.extend((name, frame) for name, frame in panel.groupby("evaluation_subset", dropna=False))

    for subset_name, subset in subsets:
        for origin_day, origin_df in subset.groupby("origin_day"):
            for scale_model in scale_models:
                row = evaluate_interval(origin_df, point_model, scale_model)
                row["evaluation_subset"] = str(subset_name)
                row["origin_day"] = int(origin_day)
                rows.append(row)
        for scale_model in scale_models:
            row = evaluate_interval(subset, point_model, scale_model)
            row["evaluation_subset"] = str(subset_name)
            row["origin_day"] = "pooled"
            rows.append(row)
    metrics = pd.DataFrame(rows)
    return add_interval_lift(metrics)


def add_interval_lift(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    baseline = out.loc[out["scale_model"].eq("V0_origin_sigma")][
        [
            "evaluation_subset",
            "origin_day",
            "coverage_80",
            "coverage_95",
            "median_width_80_pct",
            "median_width_95_pct",
            "mean_interval_score_80_pct",
            "mean_interval_score_95_pct",
        ]
    ].rename(
        columns={
            "coverage_80": "baseline_coverage_80",
            "coverage_95": "baseline_coverage_95",
            "median_width_80_pct": "baseline_median_width_80_pct",
            "median_width_95_pct": "baseline_median_width_95_pct",
            "mean_interval_score_80_pct": "baseline_mean_interval_score_80_pct",
            "mean_interval_score_95_pct": "baseline_mean_interval_score_95_pct",
        }
    )
    out = out.merge(baseline, on=["evaluation_subset", "origin_day"], how="left")
    out["coverage_error"] = (out["coverage_80"] - 0.80).abs() + (out["coverage_95"] - 0.95).abs()
    out["baseline_coverage_error"] = (
        (out["baseline_coverage_80"] - 0.80).abs() + (out["baseline_coverage_95"] - 0.95).abs()
    )
    out["coverage_error_lift_vs_V0"] = out.apply(
        lambda r: pct_improvement(r["baseline_coverage_error"], r["coverage_error"]),
        axis=1,
    )
    out["interval_score_80_lift_vs_V0"] = out.apply(
        lambda r: pct_improvement(r["baseline_mean_interval_score_80_pct"], r["mean_interval_score_80_pct"]),
        axis=1,
    )
    out["interval_score_95_lift_vs_V0"] = out.apply(
        lambda r: pct_improvement(r["baseline_mean_interval_score_95_pct"], r["mean_interval_score_95_pct"]),
        axis=1,
    )
    return out


def candidate_interval_model_for_origin(origin_day: int, *, shadow: bool = False) -> str:
    if int(origin_day) != -1:
        return BASELINE_INTERVAL_MODEL
    return SHADOW_INTERVAL_MODEL if shadow else PRODUCTION_INTERVAL_MODEL


def build_candidate_interval_policy(interval_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    origin_days = sorted(
        int(origin_day)
        for origin_day in interval_metrics.loc[interval_metrics["origin_day"].ne("pooled"), "origin_day"]
        .dropna()
        .unique()
    )
    test_metrics = interval_metrics.loc[
        interval_metrics["evaluation_subset"].eq("test")
        & interval_metrics["origin_day"].ne("pooled")
    ].copy()

    for origin_day in origin_days:
        production_model = candidate_interval_model_for_origin(origin_day)
        shadow_model = candidate_interval_model_for_origin(origin_day, shadow=True)
        production_row = test_metrics.loc[
            test_metrics["origin_day"].astype(int).eq(origin_day)
            & test_metrics["scale_model"].eq(production_model)
        ]
        shadow_row = test_metrics.loc[
            test_metrics["origin_day"].astype(int).eq(origin_day)
            & test_metrics["scale_model"].eq(shadow_model)
        ]
        production = production_row.iloc[0].to_dict() if not production_row.empty else {}
        shadow = shadow_row.iloc[0].to_dict() if not shadow_row.empty else {}
        rows.append(
            {
                "origin_day": origin_day,
                "production_interval_model": production_model,
                "shadow_interval_model": shadow_model,
                "selection_reason": "late_origin_dispersion_candidate" if origin_day == -1 else "existing_policy_unchanged",
                "test_n": int(production.get("n", 0) or 0),
                "production_coverage_80": production.get("coverage_80", np.nan),
                "production_coverage_95": production.get("coverage_95", np.nan),
                "production_median_width_80_pct": production.get("median_width_80_pct", np.nan),
                "production_median_width_95_pct": production.get("median_width_95_pct", np.nan),
                "production_interval_score_80_pct": production.get("mean_interval_score_80_pct", np.nan),
                "production_interval_score_95_pct": production.get("mean_interval_score_95_pct", np.nan),
                "production_interval_score_80_lift_vs_V0": production.get("interval_score_80_lift_vs_V0", np.nan),
                "production_interval_score_95_lift_vs_V0": production.get("interval_score_95_lift_vs_V0", np.nan),
                "shadow_coverage_80": shadow.get("coverage_80", np.nan),
                "shadow_coverage_95": shadow.get("coverage_95", np.nan),
                "shadow_median_width_80_pct": shadow.get("median_width_80_pct", np.nan),
                "shadow_median_width_95_pct": shadow.get("median_width_95_pct", np.nan),
                "shadow_interval_score_80_pct": shadow.get("mean_interval_score_80_pct", np.nan),
                "shadow_interval_score_95_pct": shadow.get("mean_interval_score_95_pct", np.nan),
            }
        )
    return pd.DataFrame(rows)


def add_candidate_interval_columns(panel: pd.DataFrame, point_model: str = "M0_locked_baseline") -> pd.DataFrame:
    out = panel.copy()
    out["production_interval_model"] = out["origin_day"].apply(candidate_interval_model_for_origin)
    out["shadow_interval_model"] = out["origin_day"].apply(
        lambda origin_day: candidate_interval_model_for_origin(origin_day, shadow=True)
    )

    for prefix, model_col in [
        (PRODUCTION_INTERVAL_COL_PREFIX, "production_interval_model"),
        (SHADOW_INTERVAL_COL_PREFIX, "shadow_interval_model"),
    ]:
        for level in INTERVAL_LEVELS:
            out[f"{prefix}_lo_{level}"] = np.nan
            out[f"{prefix}_hi_{level}"] = np.nan
        for interval_model in out[model_col].dropna().unique():
            mask = out[model_col].eq(interval_model)
            for level in INTERVAL_LEVELS:
                out.loc[mask, f"{prefix}_lo_{level}"] = out.loc[
                    mask,
                    f"{point_model}_{interval_model}_lo_{level}",
                ]
                out.loc[mask, f"{prefix}_hi_{level}"] = out.loc[
                    mask,
                    f"{point_model}_{interval_model}_hi_{level}",
                ]
    return out


def evaluate_named_interval(
    df: pd.DataFrame,
    interval_name: str,
    lo_hi_prefix: str,
    forecast_col: str,
) -> dict[str, float | int | str]:
    actual = pd.to_numeric(df["actual_opening_weekend_gross_usd"], errors="coerce")
    forecast = pd.to_numeric(df[forecast_col], errors="coerce")
    valid = actual.gt(0) & forecast.gt(0)
    for level in INTERVAL_LEVELS:
        valid &= pd.to_numeric(df[f"{lo_hi_prefix}_lo_{level}"], errors="coerce").gt(0)
        valid &= pd.to_numeric(df[f"{lo_hi_prefix}_hi_{level}"], errors="coerce").gt(0)

    if not valid.any():
        return {
            "interval_name": interval_name,
            "n": 0,
            "n_below_80": 0,
            "n_above_80": 0,
            "n_below_95": 0,
            "n_above_95": 0,
            "coverage_80": np.nan,
            "coverage_95": np.nan,
            "below_80_rate": np.nan,
            "above_80_rate": np.nan,
            "below_95_rate": np.nan,
            "above_95_rate": np.nan,
            "median_width_80_pct": np.nan,
            "median_width_95_pct": np.nan,
            "mean_interval_score_80_pct": np.nan,
            "mean_interval_score_95_pct": np.nan,
        }

    rows = df.loc[valid].copy()
    actual_v = actual.loc[valid]
    forecast_v = forecast.loc[valid]
    out: dict[str, float | int | str] = {"interval_name": interval_name, "n": int(valid.sum())}
    for level in INTERVAL_LEVELS:
        lo = pd.to_numeric(rows[f"{lo_hi_prefix}_lo_{level}"], errors="coerce")
        hi = pd.to_numeric(rows[f"{lo_hi_prefix}_hi_{level}"], errors="coerce")
        below = actual_v < lo
        above = actual_v > hi
        width_pct = (hi - lo) / forecast_v
        score = interval_score(actual_v, lo, hi, alpha=1.0 - level / 100.0) / forecast_v
        out[f"coverage_{level}"] = float((~below & ~above).mean())
        out[f"n_below_{level}"] = int(below.sum())
        out[f"n_above_{level}"] = int(above.sum())
        out[f"below_{level}_rate"] = float(below.mean())
        out[f"above_{level}_rate"] = float(above.mean())
        out[f"median_width_{level}_pct"] = float(width_pct.median())
        out[f"mean_interval_score_{level}_pct"] = float(score.mean())
    return out


def build_final_interval_audit(
    panel: pd.DataFrame,
    point_model: str = "M0_locked_baseline",
) -> pd.DataFrame:
    forecast_col = f"{point_model}_forecast_usd"
    interval_specs = [
        ("baseline_V0_origin_sigma", f"{point_model}_{BASELINE_INTERVAL_MODEL}"),
        ("production_candidate", PRODUCTION_INTERVAL_COL_PREFIX),
        ("shadow_franchise_candidate", SHADOW_INTERVAL_COL_PREFIX),
    ]
    segment_specs = [
        ("all", None),
        ("franchise_group", "franchise_group"),
        ("dispersion_state_rolling", "dispersion_state_rolling"),
    ]
    rows: list[dict[str, float | int | str]] = []
    subsets: list[tuple[str, pd.DataFrame]] = [("all", panel)]
    subsets.extend((name, frame) for name, frame in panel.groupby("evaluation_subset", dropna=False))

    for subset_name, subset in subsets:
        for origin_day, origin_df in subset.groupby("origin_day"):
            for segment_name, segment_col in segment_specs:
                segment_groups = [("all", origin_df)] if segment_col is None else origin_df.groupby(segment_col, dropna=False)
                for segment_value, segment_df in segment_groups:
                    for interval_name, lo_hi_prefix in interval_specs:
                        row = evaluate_named_interval(segment_df, interval_name, lo_hi_prefix, forecast_col)
                        row["evaluation_subset"] = str(subset_name)
                        row["origin_day"] = int(origin_day)
                        row["segment_name"] = segment_name
                        row["segment_value"] = str(segment_value)
                        rows.append(row)
    return pd.DataFrame(rows)


def add_lower_tail_grid_columns(panel: pd.DataFrame, point_model: str = "M0_locked_baseline") -> pd.DataFrame:
    out = panel.copy()
    forecast_col = f"{point_model}_forecast_usd"
    log_forecast = safe_log(out[forecast_col])
    sigma = out[f"{PRODUCTION_INTERVAL_MODEL}_sigma"]
    active = out["origin_day"].eq(-1)

    for candidate_name, lower_multiplier, upper_multiplier in LOWER_TAIL_GRID:
        lo_col = f"{candidate_name}_lo_95"
        hi_col = f"{candidate_name}_hi_95"
        out[lo_col] = np.nan
        out[hi_col] = np.nan
        out.loc[active, lo_col] = np.exp(log_forecast.loc[active] - lower_multiplier * sigma.loc[active])
        out.loc[active, hi_col] = np.exp(log_forecast.loc[active] + upper_multiplier * sigma.loc[active])
    return out


def evaluate_lower_tail_candidate(
    df: pd.DataFrame,
    candidate_name: str,
    forecast_col: str,
) -> dict[str, float | int | str]:
    actual = pd.to_numeric(df["actual_opening_weekend_gross_usd"], errors="coerce")
    forecast = pd.to_numeric(df[forecast_col], errors="coerce")
    lo = pd.to_numeric(df[f"{candidate_name}_lo_95"], errors="coerce")
    hi = pd.to_numeric(df[f"{candidate_name}_hi_95"], errors="coerce")
    valid = actual.gt(0) & forecast.gt(0) & lo.gt(0) & hi.gt(0)
    if not valid.any():
        return {
            "tail_candidate": candidate_name,
            "n": 0,
            "n_below_95": 0,
            "n_above_95": 0,
            "coverage_95": np.nan,
            "below_95_rate": np.nan,
            "above_95_rate": np.nan,
            "median_width_95_pct": np.nan,
            "mean_interval_score_95_pct": np.nan,
        }

    actual_v = actual.loc[valid]
    forecast_v = forecast.loc[valid]
    lo_v = lo.loc[valid]
    hi_v = hi.loc[valid]
    below = actual_v < lo_v
    above = actual_v > hi_v
    width_pct = (hi_v - lo_v) / forecast_v
    score = interval_score(actual_v, lo_v, hi_v, alpha=0.05) / forecast_v
    return {
        "tail_candidate": candidate_name,
        "n": int(valid.sum()),
        "n_below_95": int(below.sum()),
        "n_above_95": int(above.sum()),
        "coverage_95": float((~below & ~above).mean()),
        "below_95_rate": float(below.mean()),
        "above_95_rate": float(above.mean()),
        "median_width_95_pct": float(width_pct.median()),
        "mean_interval_score_95_pct": float(score.mean()),
    }


def build_lower_tail_calibration_audit(
    panel: pd.DataFrame,
    point_model: str = "M0_locked_baseline",
) -> pd.DataFrame:
    forecast_col = f"{point_model}_forecast_usd"
    segment_specs = [
        ("all", None),
        ("franchise_group", "franchise_group"),
        ("dispersion_state_rolling", "dispersion_state_rolling"),
    ]
    rows: list[dict[str, float | int | str]] = []
    subsets: list[tuple[str, pd.DataFrame]] = [("all", panel)]
    subsets.extend((name, frame) for name, frame in panel.groupby("evaluation_subset", dropna=False))

    for subset_name, subset in subsets:
        origin_df = subset.loc[subset["origin_day"].eq(-1)]
        if origin_df.empty:
            continue
        for segment_name, segment_col in segment_specs:
            segment_groups = [("all", origin_df)] if segment_col is None else origin_df.groupby(segment_col, dropna=False)
            for segment_value, segment_df in segment_groups:
                for candidate_name, lower_multiplier, upper_multiplier in LOWER_TAIL_GRID:
                    row = evaluate_lower_tail_candidate(segment_df, candidate_name, forecast_col)
                    row["evaluation_subset"] = str(subset_name)
                    row["origin_day"] = -1
                    row["segment_name"] = segment_name
                    row["segment_value"] = str(segment_value)
                    row["lower_multiplier"] = lower_multiplier
                    row["upper_multiplier"] = upper_multiplier
                    rows.append(row)

    audit = pd.DataFrame(rows)
    if audit.empty:
        return audit

    baseline = audit.loc[
        audit["tail_candidate"].eq("V2_symmetric_95")
        & audit["segment_name"].eq("all")
    ][
        ["evaluation_subset", "mean_interval_score_95_pct", "median_width_95_pct"]
    ].rename(
        columns={
            "mean_interval_score_95_pct": "v2_symmetric_score_95_pct",
            "median_width_95_pct": "v2_symmetric_width_95_pct",
        }
    )
    audit = audit.merge(baseline, on="evaluation_subset", how="left")
    audit["score_95_lift_vs_v2_symmetric"] = audit.apply(
        lambda r: pct_improvement(r["v2_symmetric_score_95_pct"], r["mean_interval_score_95_pct"]),
        axis=1,
    )
    audit["width_95_lift_vs_v2_symmetric"] = audit.apply(
        lambda r: pct_improvement(r["v2_symmetric_width_95_pct"], r["median_width_95_pct"]),
        axis=1,
    )
    return audit


def build_lower_tail_miss_case_audit(
    panel: pd.DataFrame,
    candidate_name: str = LEADING_LOWER_TAIL_CANDIDATE,
    point_model: str = "M0_locked_baseline",
) -> pd.DataFrame:
    forecast_col = f"{point_model}_forecast_usd"
    out = panel.loc[
        panel["evaluation_subset"].eq("test")
        & panel["origin_day"].eq(-1)
        & panel["dispersion_state_rolling"].eq("LowDispersion")
    ].copy()
    if out.empty:
        return pd.DataFrame()

    actual = pd.to_numeric(out["actual_opening_weekend_gross_usd"], errors="coerce")
    forecast = pd.to_numeric(out[forecast_col], errors="coerce")
    lo = pd.to_numeric(out[f"{candidate_name}_lo_95"], errors="coerce")
    hi = pd.to_numeric(out[f"{candidate_name}_hi_95"], errors="coerce")
    out["tail_candidate"] = candidate_name
    out["interval_95_lo_usd"] = lo
    out["interval_95_hi_usd"] = hi
    out["interval_95_width_pct"] = (hi - lo) / forecast
    out["actual_vs_lower_pct"] = actual / lo - 1.0
    out["actual_vs_upper_pct"] = actual / hi - 1.0
    out["miss_direction_95"] = np.select(
        [actual < lo, actual > hi],
        ["below_lower", "above_upper"],
        default="inside",
    )
    out["is_95_miss"] = out["miss_direction_95"].ne("inside")
    out["abs_baseline_residual_log"] = clean_numeric(out["baseline_residual_log"]).abs()
    keep = [
        "release_run_id",
        "movie_id",
        "title",
        "opening_weekend_start",
        "release_year",
        "tail_candidate",
        "franchise_group",
        "dispersion_state_rolling",
        "source_count",
        "estimate_sources",
        "latest_estimate_date",
        "earliest_estimate_date",
        "mean_source_age_days",
        "max_source_age_days",
        "actual_opening_weekend_gross_usd",
        forecast_col,
        "interval_95_lo_usd",
        "interval_95_hi_usd",
        "interval_95_width_pct",
        "is_95_miss",
        "miss_direction_95",
        "actual_vs_lower_pct",
        "actual_vs_upper_pct",
        "baseline_residual_log",
        "abs_baseline_residual_log",
        "estimate_dispersion",
    ]
    existing_keep = [col for col in keep if col in out.columns]
    return out[existing_keep].sort_values(
        ["is_95_miss", "miss_direction_95", "abs_baseline_residual_log"],
        ascending=[False, True, False],
    )


def build_lower_tail_promotion_diagnostic(
    lower_tail_audit: pd.DataFrame,
    miss_case_audit: pd.DataFrame,
    candidate_name: str = LEADING_LOWER_TAIL_CANDIDATE,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subset = lower_tail_audit.loc[
        lower_tail_audit["evaluation_subset"].eq("test")
        & lower_tail_audit["tail_candidate"].eq(candidate_name)
        & lower_tail_audit["segment_name"].eq("dispersion_state_rolling")
        & lower_tail_audit["segment_value"].eq("LowDispersion")
    ]
    if subset.empty:
        return pd.DataFrame()

    row = subset.iloc[0]
    miss_cases = miss_case_audit.loc[miss_case_audit.get("miss_direction_95", pd.Series(dtype=str)).ne("inside")].copy()
    stale_median = clean_numeric(miss_cases.get("max_source_age_days", pd.Series(dtype="float64"))).median()
    source_count_median = clean_numeric(miss_cases.get("source_count", pd.Series(dtype="float64"))).median()
    nonfranchise_misses = int(miss_cases.get("franchise_group", pd.Series(dtype=str)).eq("NonFranchise").sum())
    rows.append(
        {
            "tail_candidate": candidate_name,
            "segment_name": "dispersion_state_rolling",
            "segment_value": "LowDispersion",
            "n": int(row["n"]),
            "coverage_95": row["coverage_95"],
            "n_below_95": int(row["n_below_95"]),
            "n_above_95": int(row["n_above_95"]),
            "below_95_rate": row["below_95_rate"],
            "above_95_rate": row["above_95_rate"],
            "median_width_95_pct": row["median_width_95_pct"],
            "mean_interval_score_95_pct": row["mean_interval_score_95_pct"],
            "missed_titles": "; ".join(miss_cases["title"].dropna().astype(str).tolist()),
            "miss_source_count_median": source_count_median,
            "miss_max_source_age_median": stale_median,
            "nonfranchise_miss_count": nonfranchise_misses,
            "decision_case": (
                "keep_shadow_only"
                if int(row["n"]) >= 30 and float(row["coverage_95"]) < 0.95
                else "does_not_block_promotion"
            ),
        }
    )
    return pd.DataFrame(rows)


def add_guardrail_candidate_columns(panel: pd.DataFrame, point_model: str = "M0_locked_baseline") -> pd.DataFrame:
    out = panel.copy()
    active = out["origin_day"].eq(-1)
    source_count = pd.to_numeric(out["source_count"], errors="coerce")
    max_age = pd.to_numeric(out["max_source_age_days"], errors="coerce")
    stale_single = source_count.eq(1) & max_age.ge(STALE_SINGLE_SOURCE_MIN_AGE_DAYS)
    stale_single_nonfranchise = stale_single & out["franchise_group"].eq("NonFranchise")

    v2_lo = pd.to_numeric(out[f"{LEADING_LOWER_TAIL_CANDIDATE}_lo_95"], errors="coerce")
    v2_hi = pd.to_numeric(out[f"{LEADING_LOWER_TAIL_CANDIDATE}_hi_95"], errors="coerce")
    v2_symmetric_hi = pd.to_numeric(out["V2_symmetric_95_hi_95"], errors="coerce")
    v0_lo = pd.to_numeric(out[f"{point_model}_{BASELINE_INTERVAL_MODEL}_lo_95"], errors="coerce")
    v0_hi = pd.to_numeric(out[f"{point_model}_{BASELINE_INTERVAL_MODEL}_hi_95"], errors="coerce")
    forecast_col = f"{point_model}_forecast_usd"
    log_forecast = safe_log(out[forecast_col])
    v2_sigma = pd.to_numeric(out[f"{PRODUCTION_INTERVAL_MODEL}_sigma"], errors="coerce")

    out["stale_single_source_guardrail_flag"] = active & stale_single
    out["stale_single_source_nonfranchise_guardrail_flag"] = active & stale_single_nonfranchise

    for candidate in GUARDRAIL_CANDIDATES:
        out[f"{candidate}_lo_95"] = np.nan
        out[f"{candidate}_hi_95"] = np.nan

    out.loc[active, "V2_lower_15_lo_95"] = v2_lo.loc[active]
    out.loc[active, "V2_lower_15_hi_95"] = v2_hi.loc[active]

    for candidate, flag in [
        ("G1_single_stale_lower_v0", stale_single),
        ("G2_single_stale_nonfranchise_lower_v0", stale_single_nonfranchise),
    ]:
        out.loc[active, f"{candidate}_lo_95"] = v2_lo.loc[active]
        out.loc[active, f"{candidate}_hi_95"] = v2_hi.loc[active]
        guard = active & flag
        out.loc[guard, f"{candidate}_lo_95"] = np.minimum(v2_lo.loc[guard], v0_lo.loc[guard])
        out.loc[guard, f"{candidate}_hi_95"] = v2_symmetric_hi.loc[guard]

    out.loc[active, "G3_single_stale_full_v0_lo_95"] = v2_lo.loc[active]
    out.loc[active, "G3_single_stale_full_v0_hi_95"] = v2_hi.loc[active]
    guard = active & stale_single
    out.loc[guard, "G3_single_stale_full_v0_lo_95"] = v0_lo.loc[guard]
    out.loc[guard, "G3_single_stale_full_v0_hi_95"] = v0_hi.loc[guard]

    out.loc[active, "G4_single_stale_lower_3sigma_lo_95"] = v2_lo.loc[active]
    out.loc[active, "G4_single_stale_lower_3sigma_hi_95"] = v2_hi.loc[active]
    out.loc[guard, "G4_single_stale_lower_3sigma_lo_95"] = np.exp(
        log_forecast.loc[guard] - 3.0 * v2_sigma.loc[guard]
    )
    out.loc[guard, "G4_single_stale_lower_3sigma_hi_95"] = v2_symmetric_hi.loc[guard]
    return out


def evaluate_guardrail_candidate(
    df: pd.DataFrame,
    candidate_name: str,
    forecast_col: str,
) -> dict[str, float | int | str]:
    actual = pd.to_numeric(df["actual_opening_weekend_gross_usd"], errors="coerce")
    forecast = pd.to_numeric(df[forecast_col], errors="coerce")
    lo = pd.to_numeric(df[f"{candidate_name}_lo_95"], errors="coerce")
    hi = pd.to_numeric(df[f"{candidate_name}_hi_95"], errors="coerce")
    valid = actual.gt(0) & forecast.gt(0) & lo.gt(0) & hi.gt(0)
    if not valid.any():
        return {
            "guardrail_candidate": candidate_name,
            "n": 0,
            "coverage_95": np.nan,
            "n_below_95": 0,
            "n_above_95": 0,
            "below_95_rate": np.nan,
            "above_95_rate": np.nan,
            "median_width_95_pct": np.nan,
            "mean_interval_score_95_pct": np.nan,
        }

    actual_v = actual.loc[valid]
    forecast_v = forecast.loc[valid]
    lo_v = lo.loc[valid]
    hi_v = hi.loc[valid]
    below = actual_v < lo_v
    above = actual_v > hi_v
    width_pct = (hi_v - lo_v) / forecast_v
    score = interval_score(actual_v, lo_v, hi_v, alpha=0.05) / forecast_v
    return {
        "guardrail_candidate": candidate_name,
        "n": int(valid.sum()),
        "coverage_95": float((~below & ~above).mean()),
        "n_below_95": int(below.sum()),
        "n_above_95": int(above.sum()),
        "below_95_rate": float(below.mean()),
        "above_95_rate": float(above.mean()),
        "median_width_95_pct": float(width_pct.median()),
        "mean_interval_score_95_pct": float(score.mean()),
    }


def build_guardrail_interval_audit(
    panel: pd.DataFrame,
    point_model: str = "M0_locked_baseline",
) -> pd.DataFrame:
    forecast_col = f"{point_model}_forecast_usd"
    segment_specs = [
        ("all", None),
        ("franchise_group", "franchise_group"),
        ("dispersion_state_rolling", "dispersion_state_rolling"),
        ("stale_single_source_guardrail_flag", "stale_single_source_guardrail_flag"),
    ]
    rows: list[dict[str, float | int | str]] = []
    subsets: list[tuple[str, pd.DataFrame]] = [("all", panel)]
    subsets.extend((name, frame) for name, frame in panel.groupby("evaluation_subset", dropna=False))

    for subset_name, subset in subsets:
        origin_df = subset.loc[subset["origin_day"].eq(-1)]
        if origin_df.empty:
            continue
        for segment_name, segment_col in segment_specs:
            segment_groups = [("all", origin_df)] if segment_col is None else origin_df.groupby(segment_col, dropna=False)
            for segment_value, segment_df in segment_groups:
                for candidate_name in GUARDRAIL_CANDIDATES:
                    row = evaluate_guardrail_candidate(segment_df, candidate_name, forecast_col)
                    row["evaluation_subset"] = str(subset_name)
                    row["origin_day"] = -1
                    row["segment_name"] = segment_name
                    row["segment_value"] = str(segment_value)
                    rows.append(row)

    audit = pd.DataFrame(rows)
    if audit.empty:
        return audit

    reference = audit.loc[
        audit["evaluation_subset"].eq("test")
        & audit["segment_name"].eq("all")
        & audit["guardrail_candidate"].eq("V2_lower_15")
    ][["mean_interval_score_95_pct", "median_width_95_pct"]]
    if not reference.empty:
        ref_score = float(reference.iloc[0]["mean_interval_score_95_pct"])
        ref_width = float(reference.iloc[0]["median_width_95_pct"])
        audit["score_95_lift_vs_V2_lower_15"] = audit["mean_interval_score_95_pct"].apply(
            lambda value: pct_improvement(ref_score, value)
        )
        audit["width_95_lift_vs_V2_lower_15"] = audit["median_width_95_pct"].apply(
            lambda value: pct_improvement(ref_width, value)
        )
    return audit


def build_extreme_residual_audit(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.loc[
        panel["evaluation_subset"].eq("test")
        & panel["origin_day"].eq(-1)
    ].copy()
    out["abs_baseline_residual_log"] = clean_numeric(out["baseline_residual_log"]).abs()
    out["forecast_actual_ratio"] = np.exp(out["abs_baseline_residual_log"])
    out = out.loc[out["abs_baseline_residual_log"].ge(1.0)].copy()
    keep = [
        "release_run_id",
        "movie_id",
        "title",
        "opening_weekend_start",
        "release_year",
        "franchise_group",
        "dispersion_state_rolling",
        "source_count",
        "estimate_sources",
        "latest_estimate_date",
        "mean_source_age_days",
        "max_source_age_days",
        "actual_opening_weekend_gross_usd",
        "M0_locked_baseline_forecast_usd",
        "baseline_residual_log",
        "abs_baseline_residual_log",
        "forecast_actual_ratio",
        "release_type",
        "release_width_bucket",
        "genre",
        "distributor",
        "is_doc_concert",
    ]
    existing_keep = [col for col in keep if col in out.columns]
    return out[existing_keep].sort_values("abs_baseline_residual_log", ascending=False)


def build_scale_diagnostic(panel: pd.DataFrame, point_model: str = "M0_locked_baseline") -> pd.DataFrame:
    residual_col = f"{point_model}_post_correction_residual_log"
    rows: list[dict[str, float | int | str]] = []
    grouping_specs = {
        "franchise_group": "franchise_group",
        "dispersion_state_rolling": "dispersion_state_rolling",
        "wiki_rigor_state_rolling": "wiki_rigor_state_rolling",
    }
    subsets: list[tuple[str, pd.DataFrame]] = [("all", panel)]
    subsets.extend((name, frame) for name, frame in panel.groupby("evaluation_subset", dropna=False))
    for subset_name, subset in subsets:
        for origin_day, origin_df in subset.groupby("origin_day"):
            for diagnostic, group_col in grouping_specs.items():
                for group_value, group in origin_df.groupby(group_col, dropna=False):
                    residual = clean_numeric(group[residual_col]).dropna()
                    rows.append(
                        {
                            "evaluation_subset": str(subset_name),
                            "origin_day": int(origin_day),
                            "scale_diagnostic": diagnostic,
                            "group_value": str(group_value),
                            "n": int(len(residual)),
                            "mean_abs_residual_log": float(residual.abs().mean()) if len(residual) else np.nan,
                            "rmse_residual_log": float(np.sqrt(np.mean(residual**2))) if len(residual) else np.nan,
                            "residual_std_log": float(residual.std(ddof=1)) if len(residual) > 1 else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def mean_positive(values: list[float]) -> float:
    clean = pd.Series(values, dtype="float64").dropna()
    return float(clean.mean()) if len(clean) else np.nan


def build_daily_feature_effect_audit(
    point_metrics: pd.DataFrame,
    interval_metrics: pd.DataFrame,
    scale_diagnostic: pd.DataFrame,
) -> pd.DataFrame:
    feature_map = [
        {
            "feature": "wiki_log_rigor",
            "point_model": "M1_wiki_log_rigor_origin_specific",
            "scale_model": "V3_wiki_rigor_scale",
        },
        {
            "feature": "franchise_group",
            "point_model": "M3_franchise_group",
            "scale_model": "V1_franchise_scale",
        },
        {
            "feature": "estimate_dispersion",
            "point_model": "M4_estimate_dispersion",
            "scale_model": "V2_dispersion_scale",
        },
    ]

    rows: list[dict[str, float | int | str | bool]] = []
    test_point = point_metrics.loc[
        point_metrics["evaluation_subset"].eq("test")
        & point_metrics["origin_day"].ne("pooled")
    ].copy()
    test_interval = interval_metrics.loc[
        interval_metrics["evaluation_subset"].eq("test")
        & interval_metrics["origin_day"].ne("pooled")
    ].copy()

    for origin_day in sorted(test_point["origin_day"].dropna().unique(), key=lambda x: int(x)):
        for spec in feature_map:
            p = test_point.loc[
                test_point["origin_day"].eq(origin_day)
                & test_point["point_model"].eq(spec["point_model"])
            ]
            v = test_interval.loc[
                test_interval["origin_day"].eq(origin_day)
                & test_interval["scale_model"].eq(spec["scale_model"])
            ]
            p_row = p.iloc[0].to_dict() if not p.empty else {}
            v_row = v.iloc[0].to_dict() if not v.empty else {}

            point_promote = (
                p_row.get("MAE_log_lift_vs_M0", np.nan) > 0
                and p_row.get("MdAPE_lift_vs_M0", np.nan) > 0
            )
            point_shadow = (
                not point_promote
                and (
                    p_row.get("MAE_log_lift_vs_M0", np.nan) > 0
                    or p_row.get("MdAPE_lift_vs_M0", np.nan) > 0
                    or p_row.get("RMSE_log_lift_vs_M0", np.nan) > 0
                )
            )
            interval_score_lift = mean_positive(
                [
                    v_row.get("interval_score_80_lift_vs_V0", np.nan),
                    v_row.get("interval_score_95_lift_vs_V0", np.nan),
                ]
            )
            interval_promote = (
                v_row.get("coverage_error_lift_vs_V0", np.nan) > 0
                and interval_score_lift > 0
            )
            interval_shadow = (
                not interval_promote
                and (
                    v_row.get("interval_score_80_lift_vs_V0", np.nan) > 0
                    or v_row.get("interval_score_95_lift_vs_V0", np.nan) > 0
                )
            )
            rows.append(
                {
                    "origin_day": int(origin_day),
                    "feature": spec["feature"],
                    "point_model": spec["point_model"],
                    "scale_model": spec["scale_model"],
                    "point_n": p_row.get("n", 0),
                    "MAE_log_lift_vs_M0": p_row.get("MAE_log_lift_vs_M0", np.nan),
                    "MdAPE_lift_vs_M0": p_row.get("MdAPE_lift_vs_M0", np.nan),
                    "RMSE_log_lift_vs_M0": p_row.get("RMSE_log_lift_vs_M0", np.nan),
                    "point_promote": bool(point_promote),
                    "point_decision": "promote" if point_promote else "shadow" if point_shadow else "reject",
                    "interval_n": v_row.get("n", 0),
                    "coverage_80": v_row.get("coverage_80", np.nan),
                    "coverage_95": v_row.get("coverage_95", np.nan),
                    "coverage_error_lift_vs_V0": v_row.get("coverage_error_lift_vs_V0", np.nan),
                    "interval_score_80_lift_vs_V0": v_row.get("interval_score_80_lift_vs_V0", np.nan),
                    "interval_score_95_lift_vs_V0": v_row.get("interval_score_95_lift_vs_V0", np.nan),
                    "interval_score_lift_vs_V0": interval_score_lift,
                    "interval_promote": bool(interval_promote),
                    "interval_decision": (
                        "promote"
                        if interval_promote
                        else "shadow"
                        if interval_shadow
                        else "insufficient_support"
                        if v_row.get("n", 0) == 0
                        else "reject"
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_feature_stability_summary(daily_effects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for feature, group in daily_effects.groupby("feature"):
        g = group.copy()
        best_point = g.sort_values("MAE_log_lift_vs_M0", ascending=False).iloc[0]
        best_interval = g.sort_values("interval_score_95_lift_vs_V0", ascending=False).iloc[0]
        rows.append(
            {
                "feature": feature,
                "n_origins": int(g["origin_day"].nunique()),
                "point_positive_MAE_origins": int((g["MAE_log_lift_vs_M0"] > 0).sum()),
                "point_positive_MdAPE_origins": int((g["MdAPE_lift_vs_M0"] > 0).sum()),
                "point_promoted_origins": int(g["point_promote"].sum()),
                "point_shadow_origins": int(g["point_decision"].eq("shadow").sum()),
                "interval_positive_80_score_origins": int((g["interval_score_80_lift_vs_V0"] > 0).sum()),
                "interval_positive_95_score_origins": int((g["interval_score_95_lift_vs_V0"] > 0).sum()),
                "interval_promoted_origins": int(g["interval_promote"].sum()),
                "interval_shadow_origins": int(g["interval_decision"].eq("shadow").sum()),
                "best_point_origin_by_MAE": int(best_point["origin_day"]),
                "best_point_MAE_lift_vs_M0": best_point["MAE_log_lift_vs_M0"],
                "best_interval_origin_by_95_score": int(best_interval["origin_day"]),
                "best_interval_score_95_lift_vs_V0": best_interval["interval_score_95_lift_vs_V0"],
            }
        )
    return pd.DataFrame(rows)


def build_feature_policy_decision(daily_effects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    for origin_day in range(-14, 0):
        origin = daily_effects.loc[daily_effects["origin_day"].eq(origin_day)]
        shadow_point_models = origin.loc[
            origin["point_promote"].eq(False)
            & (
                origin["MAE_log_lift_vs_M0"].gt(0)
                | origin["RMSE_log_lift_vs_M0"].gt(0)
            ),
            "point_model",
        ].dropna().astype(str).unique()
        shadow_interval_models = origin.loc[
            origin["interval_promote"].eq(False)
            & (
                origin["interval_score_80_lift_vs_V0"].gt(0)
                | origin["interval_score_95_lift_vs_V0"].gt(0)
            ),
            "scale_model",
        ].dropna().astype(str).unique()

        rows.append(
            {
                "origin_day": origin_day,
                "production_point_model": "M0_locked_baseline",
                "production_point_reason": "no feature improves both MAE_log and MdAPE",
                "production_interval_model": "V1_franchise_scale" if origin_day == -1 else "V0_origin_sigma",
                "production_interval_reason": (
                    "only strict daily interval promotion"
                    if origin_day == -1
                    else "no strict interval promotion"
                ),
                "shadow_point_models": "; ".join(shadow_point_models),
                "shadow_interval_models": "; ".join(shadow_interval_models),
            }
        )
    return pd.DataFrame(rows)


def build_decision_table(point_metrics: pd.DataFrame, interval_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    test_point = point_metrics.loc[
        point_metrics["evaluation_subset"].eq("test") & point_metrics["origin_day"].ne("pooled")
    ].copy()
    for origin_day, group in test_point.groupby("origin_day"):
        for model_name, label in [
            ("M1_wiki_log_rigor_origin_specific", "Point lift"),
            ("M2_wiki_log_rigor_late_interaction", "Late wiki lift"),
            ("M3_franchise_group", "Category point lift"),
            ("M4_estimate_dispersion", "Dispersion point lift"),
        ]:
            row = group.loc[group["point_model"].eq(model_name)]
            if row.empty:
                continue
            item = row.iloc[0]
            rows.append(
                {
                    "origin_day": int(origin_day),
                    "test": label,
                    "target": "baseline_residual_log",
                    "candidate_feature": model_name,
                    "promotion_criterion": "lower out-of-sample MAE_log and MdAPE vs M0",
                    "n": item["n"],
                    "MAE_log_lift_vs_M0": item["MAE_log_lift_vs_M0"],
                    "MdAPE_lift_vs_M0": item["MdAPE_lift_vs_M0"],
                    "promote": bool(item["MAE_log_lift_vs_M0"] > 0 and item["MdAPE_lift_vs_M0"] > 0),
                }
            )

    test_interval = interval_metrics.loc[
        interval_metrics["evaluation_subset"].eq("test") & interval_metrics["origin_day"].ne("pooled")
    ].copy()
    for origin_day, group in test_interval.groupby("origin_day"):
        for model_name, label, feature in [
            ("V3_wiki_rigor_scale", "Wiki scale lift", "wiki_rigor_state_rolling"),
            ("V1_franchise_scale", "Scale lift", "franchise_group"),
            ("V2_dispersion_scale", "Dispersion scale lift", "dispersion_state_rolling"),
        ]:
            row = group.loc[group["scale_model"].eq(model_name)]
            if row.empty:
                continue
            item = row.iloc[0]
            score_values = pd.Series(
                [item["interval_score_80_lift_vs_V0"], item["interval_score_95_lift_vs_V0"]],
                dtype="float64",
            ).dropna()
            score_lift = float(score_values.mean()) if len(score_values) else np.nan
            rows.append(
                {
                    "origin_day": int(origin_day),
                    "test": label,
                    "target": "post_correction_residual_scale",
                    "candidate_feature": feature,
                    "promotion_criterion": "better coverage error and lower interval score vs V0",
                    "n": item["n"],
                    "coverage_error_lift_vs_V0": item["coverage_error_lift_vs_V0"],
                    "interval_score_lift_vs_V0": score_lift,
                    "promote": bool(item["coverage_error_lift_vs_V0"] > 0 and score_lift > 0),
                }
            )
    return pd.DataFrame(rows)


def write_outputs(
    panel: pd.DataFrame,
    point_metrics: pd.DataFrame,
    scale_diagnostic: pd.DataFrame,
    interval_metrics: pd.DataFrame,
    candidate_interval_policy: pd.DataFrame,
    final_interval_audit: pd.DataFrame,
    lower_tail_calibration_audit: pd.DataFrame,
    lower_tail_miss_case_audit: pd.DataFrame,
    lower_tail_promotion_diagnostic: pd.DataFrame,
    guardrail_interval_audit: pd.DataFrame,
    extreme_residual_audit: pd.DataFrame,
    daily_feature_effect_audit: pd.DataFrame,
    feature_stability_summary: pd.DataFrame,
    feature_policy_decision: pd.DataFrame,
    decision_table: pd.DataFrame,
    output_dir: Path,
    prefix: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "panel": output_dir / f"{prefix}_panel.csv",
        "point_metrics": output_dir / f"{prefix}_point_metrics.csv",
        "scale_diagnostic": output_dir / f"{prefix}_scale_diagnostic.csv",
        "interval_metrics": output_dir / f"{prefix}_interval_metrics.csv",
        "candidate_interval_policy": output_dir / f"{prefix}_candidate_interval_policy.csv",
        "final_interval_audit": output_dir / f"{prefix}_final_interval_audit.csv",
        "lower_tail_calibration_audit": output_dir / f"{prefix}_lower_tail_calibration_audit.csv",
        "lower_tail_miss_case_audit": output_dir / f"{prefix}_lower_tail_miss_case_audit.csv",
        "lower_tail_promotion_diagnostic": output_dir / f"{prefix}_lower_tail_promotion_diagnostic.csv",
        "guardrail_interval_audit": output_dir / f"{prefix}_guardrail_interval_audit.csv",
        "extreme_residual_audit": output_dir / f"{prefix}_extreme_residual_audit.csv",
        "daily_feature_effect_audit": output_dir / f"{prefix}_daily_feature_effect_audit.csv",
        "feature_stability_summary": output_dir / f"{prefix}_feature_stability_summary.csv",
        "feature_policy_decision": output_dir / f"{prefix}_feature_policy_decision.csv",
        "decision_table": output_dir / f"{prefix}_decision_table.csv",
    }
    panel.sort_values(["opening_weekend_start", "release_run_id", "origin_day"]).to_csv(paths["panel"], index=False)
    write_sorted(point_metrics, paths["point_metrics"], ["evaluation_subset", "_origin_day_order", "point_model"])
    write_sorted(scale_diagnostic, paths["scale_diagnostic"], ["evaluation_subset", "_origin_day_order", "scale_diagnostic", "group_value"])
    write_sorted(interval_metrics, paths["interval_metrics"], ["evaluation_subset", "_origin_day_order", "scale_model"])
    write_sorted(candidate_interval_policy, paths["candidate_interval_policy"], ["_origin_day_order"])
    write_sorted(
        final_interval_audit,
        paths["final_interval_audit"],
        ["evaluation_subset", "_origin_day_order", "segment_name", "segment_value", "interval_name"],
    )
    write_sorted(
        lower_tail_calibration_audit,
        paths["lower_tail_calibration_audit"],
        ["evaluation_subset", "_origin_day_order", "segment_name", "segment_value", "tail_candidate"],
    )
    write_sorted(
        lower_tail_miss_case_audit,
        paths["lower_tail_miss_case_audit"],
        ["is_95_miss", "miss_direction_95", "abs_baseline_residual_log"],
    )
    lower_tail_promotion_diagnostic.to_csv(paths["lower_tail_promotion_diagnostic"], index=False)
    write_sorted(
        guardrail_interval_audit,
        paths["guardrail_interval_audit"],
        ["evaluation_subset", "_origin_day_order", "segment_name", "segment_value", "guardrail_candidate"],
    )
    extreme_residual_audit.to_csv(paths["extreme_residual_audit"], index=False)
    write_sorted(
        daily_feature_effect_audit,
        paths["daily_feature_effect_audit"],
        ["_origin_day_order", "feature"],
    )
    feature_stability_summary.sort_values("feature").to_csv(paths["feature_stability_summary"], index=False)
    feature_policy_decision.sort_values("origin_day").to_csv(paths["feature_policy_decision"], index=False)
    write_sorted(decision_table, paths["decision_table"], ["_origin_day_order", "test", "candidate_feature"])
    return paths


def write_sorted(df: pd.DataFrame, path: Path, sort_cols: list[str]) -> None:
    out = df.copy()
    if "origin_day" in out.columns:
        out["_origin_day_order"] = pd.to_numeric(out["origin_day"], errors="coerce").fillna(999)
    existing = [col for col in sort_cols if col in out.columns]
    drop_cols = [col for col in ["_origin_day_order"] if col in out.columns]
    out.sort_values(existing).drop(columns=drop_cols).to_csv(path, index=False)


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def parse_str_list(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run rolling-origin residual-predictor diagnostics against a locked consensus baseline."
    )
    add_database_arg(parser)
    parser.add_argument("--output-dir", type=Path, default=DIAGNOSTICS_DIR)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--baseline-point-col", default=DEFAULT_BASELINE_POINT_COL)
    parser.add_argument("--origin-days", default=",".join(str(day) for day in DEFAULT_ORIGIN_DAYS))
    parser.add_argument("--train-years", default=",".join(str(year) for year in DEFAULT_TRAIN_YEARS))
    parser.add_argument("--test-start-year", type=int, default=DEFAULT_TEST_START_YEAR)
    parser.add_argument("--exclude-sources", default=",".join(DEFAULT_EXCLUDED_ESTIMATE_SOURCES))
    parser.add_argument("--exclude-release-years", default=",".join(str(year) for year in DEFAULT_EXCLUDED_RELEASE_YEARS))
    parser.add_argument("--recency-lambdas", default=",".join(str(value) for value in DEFAULT_RECENCY_LAMBDAS))
    parser.add_argument("--max-source-age-days", default=",".join(str(day) for day in DEFAULT_MAX_SOURCE_AGE_DAYS))
    parser.add_argument("--min-source-reliability-n", type=int, default=benchmark.DEFAULT_MIN_SOURCE_RELIABILITY_N)
    parser.add_argument("--source-bias-shrink-k", type=int, default=benchmark.DEFAULT_SOURCE_BIAS_SHRINK_K)
    parser.add_argument("--min-train-n", type=int, default=DEFAULT_MIN_TRAIN_N)
    parser.add_argument("--ridge-alpha", type=float, default=DEFAULT_RIDGE_ALPHA)
    parser.add_argument("--interval-shrink-k", type=int, default=DEFAULT_INTERVAL_SHRINK_K)
    return parser


def config_from_args(args: argparse.Namespace) -> ResidualDiagnosticConfig:
    benchmark_config = benchmark.BenchmarkConfig(
        origin_days=parse_int_list(args.origin_days),
        excluded_estimate_sources=parse_str_list(args.exclude_sources),
        excluded_release_years=parse_int_list(args.exclude_release_years),
        recency_lambdas=parse_float_list(args.recency_lambdas),
        train_years=parse_int_list(args.train_years),
        test_start_year=args.test_start_year,
        min_source_reliability_n=args.min_source_reliability_n,
        source_bias_shrink_k=args.source_bias_shrink_k,
        max_source_age_days=parse_int_list(args.max_source_age_days),
        min_train_n_for_model_selection=args.min_train_n,
        interval_shrink_k=args.interval_shrink_k,
    )
    return ResidualDiagnosticConfig(
        benchmark_config=benchmark_config,
        baseline_point_col=args.baseline_point_col,
        min_train_n=args.min_train_n,
        ridge_alpha=args.ridge_alpha,
        interval_shrink_k=args.interval_shrink_k,
        output_prefix=args.output_prefix,
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)

    consensus = build_consensus_panel(args.database_url, config.benchmark_config)
    wiki = fetch_wiki_origin_features(args.database_url, config.benchmark_config.origin_days)
    panel = add_residual_features(consensus, wiki, config.baseline_point_col)
    panel = add_rolling_feature_states(panel, config.min_train_n)
    panel = add_point_residual_predictions(panel, config)
    panel = add_scale_intervals(
        panel,
        point_model="M0_locked_baseline",
        min_train_n=config.min_train_n,
        shrink_k=config.interval_shrink_k,
    )
    panel = add_candidate_interval_columns(panel, point_model="M0_locked_baseline")
    panel = add_lower_tail_grid_columns(panel, point_model="M0_locked_baseline")
    panel = add_guardrail_candidate_columns(panel, point_model="M0_locked_baseline")

    point_metrics = build_point_metrics(panel)
    scale_diagnostic = build_scale_diagnostic(panel, point_model="M0_locked_baseline")
    interval_metrics = build_interval_metrics(panel, point_model="M0_locked_baseline")
    candidate_interval_policy = build_candidate_interval_policy(interval_metrics)
    final_interval_audit = build_final_interval_audit(panel, point_model="M0_locked_baseline")
    lower_tail_calibration_audit = build_lower_tail_calibration_audit(panel, point_model="M0_locked_baseline")
    lower_tail_miss_case_audit = build_lower_tail_miss_case_audit(panel)
    lower_tail_promotion_diagnostic = build_lower_tail_promotion_diagnostic(
        lower_tail_calibration_audit,
        lower_tail_miss_case_audit,
    )
    guardrail_interval_audit = build_guardrail_interval_audit(panel, point_model="M0_locked_baseline")
    extreme_residual_audit = build_extreme_residual_audit(panel)
    daily_feature_effect_audit = build_daily_feature_effect_audit(
        point_metrics,
        interval_metrics,
        scale_diagnostic,
    )
    feature_stability_summary = build_feature_stability_summary(daily_feature_effect_audit)
    feature_policy_decision = build_feature_policy_decision(daily_feature_effect_audit)
    decision_table = build_decision_table(point_metrics, interval_metrics)
    paths = write_outputs(
        panel,
        point_metrics,
        scale_diagnostic,
        interval_metrics,
        candidate_interval_policy,
        final_interval_audit,
        lower_tail_calibration_audit,
        lower_tail_miss_case_audit,
        lower_tail_promotion_diagnostic,
        guardrail_interval_audit,
        extreme_residual_audit,
        daily_feature_effect_audit,
        feature_stability_summary,
        feature_policy_decision,
        decision_table,
        args.output_dir,
        config.output_prefix,
    )

    print("Wrote residual-predictor diagnostics:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
