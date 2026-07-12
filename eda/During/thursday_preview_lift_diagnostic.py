#!/usr/bin/env python3
"""Thursday preview lift diagnostic for opening-weekend consensus forecasts."""

from __future__ import annotations

import argparse
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

from models.boxoffice.artifacts import DEFAULT_ARTIFACT_ROOT, latest_model_version
from pm_box_office.db.connection import connect_database
from pm_box_office.sources.common.cli import add_database_arg


DIAGNOSTICS_DIR = REPO_ROOT / "data" / "diagnostics"
DEFAULT_MIN_TRAIN_N = 20
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260711

MODEL_FORECAST_COLUMNS = {
    "consensus": "forecast_consensus_usd",
    "preview_status_bias": "forecast_preview_status_bias_usd",
    "preview_amount_update": "forecast_preview_amount_update_usd",
}
TRAINING_POLICY_FORECAST_COL = "forecast_preview_amount_update_usd"


@dataclass(frozen=True)
class DiagnosticOutputs:
    analysis_panel: pd.DataFrame
    exclusion_audit: pd.DataFrame
    rolling_predictions: pd.DataFrame
    model_comparison: pd.DataFrame
    year_stability: pd.DataFrame
    paired_bootstrap: pd.DataFrame
    training_policy_predictions: pd.DataFrame
    training_policy_comparison: pd.DataFrame
    training_policy_year_stability: pd.DataFrame
    training_policy_bootstrap: pd.DataFrame
    downstream_live_friday_predictions: pd.DataFrame
    downstream_live_friday_comparison: pd.DataFrame
    downstream_live_friday_bootstrap: pd.DataFrame
    downstream_live_friday_influence_audit: pd.DataFrame
    downstream_live_friday_bucket_metrics: pd.DataFrame
    downstream_live_friday_leave_one_out: pd.DataFrame
    downstream_live_friday_top_influence_removed: pd.DataFrame


def active_pre_release_panel_path(artifact_root: Path = DEFAULT_ARTIFACT_ROOT) -> Path:
    model_version = latest_model_version(artifact_root)
    path = artifact_root / model_version / "pre_release_panel.csv"
    if not path.exists():
        raise FileNotFoundError(f"Active model pre-release panel not found: {path}")
    return path


def load_pre_release_panel(path: Path | None = None) -> pd.DataFrame:
    panel_path = path or active_pre_release_panel_path()
    return pd.read_csv(
        panel_path,
        parse_dates=["opening_weekend_start", "forecast_origin_date", "latest_estimate_date"],
    )


def preview_rows_sql(start_year: int = 2022) -> str:
    return """
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
            d.box_office_date::date AS preview_date,
            d.gross_usd::double precision AS preview_gross_usd,
            d.source AS preview_source,
            d.source_url AS preview_source_url,
            d.fetched_at AS preview_fetched_at
        FROM analytics.eda_movie_openings o
        JOIN daily_box_office d ON d.release_run_id = o.release_run_id
        WHERE o.opening_weekend_start::date >= DATE '{start_year}-01-01'
          AND d.is_preview = 1
          AND d.gross_usd IS NOT NULL
          AND d.gross_usd > 0
          AND d.box_office_date::date < o.opening_date::date
        ORDER BY o.opening_weekend_start::date, o.movie_id, d.box_office_date::date, d.daily_box_office_id
    """.format(start_year=int(start_year))


def fetch_frame(conn: Any, sql: str) -> pd.DataFrame:
    cursor = conn.execute(sql)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def fetch_preview_rows(database_url: str | None = None, *, start_year: int = 2022) -> pd.DataFrame:
    conn = connect_database(database_url)
    try:
        return fetch_frame(conn, preview_rows_sql(start_year=start_year))
    finally:
        conn.close()


def _date_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def validate_preview_gross(preview_rows: pd.DataFrame) -> pd.DataFrame:
    if preview_rows.empty:
        return pd.DataFrame()

    rows = preview_rows.copy()
    rows["preview_date"] = _date_series(rows["preview_date"])
    rows["opening_weekend_start"] = _date_series(rows["opening_weekend_start"])
    rows["opening_date"] = _date_series(rows["opening_date"])
    rows["preview_gross_usd"] = pd.to_numeric(rows["preview_gross_usd"], errors="coerce")
    rows["actual_opening_weekend_gross_usd"] = pd.to_numeric(
        rows["actual_opening_weekend_gross_usd"], errors="coerce"
    )

    group_cols = ["release_run_id", "movie_id"]
    summaries: list[dict[str, Any]] = []
    for (release_run_id, movie_id), group in rows.groupby(group_cols, sort=False):
        positive = group.loc[group["preview_gross_usd"].gt(0)].copy()
        distinct_pairs = positive[["preview_date", "preview_gross_usd"]].drop_duplicates()
        preview_date_count = int(positive["preview_date"].nunique(dropna=True))
        preview_gross_count = int(positive["preview_gross_usd"].nunique(dropna=True))
        raw_row_count = int(len(group))
        distinct_row_count = int(len(distinct_pairs))

        unclear = preview_date_count != 1 or preview_gross_count != 1 or distinct_row_count != 1
        duplicate_identical = raw_row_count > 1 and not unclear

        first = group.sort_values(["preview_date", "daily_box_office_id"], na_position="last").iloc[0]
        summaries.append(
            {
                "release_run_id": int(release_run_id),
                "movie_id": int(movie_id),
                "title": first.get("title"),
                "opening_date": first.get("opening_date"),
                "opening_weekend_start": first.get("opening_weekend_start"),
                "release_year": first.get("release_year"),
                "release_width_bucket": first.get("release_width_bucket"),
                "release_type": first.get("release_type"),
                "opening_weekend_theaters": first.get("opening_weekend_theaters"),
                "actual_opening_weekend_gross_usd": first.get("actual_opening_weekend_gross_usd"),
                "preview_row_count": raw_row_count,
                "preview_distinct_row_count": distinct_row_count,
                "preview_date_count": preview_date_count,
                "preview_gross_value_count": preview_gross_count,
                "first_preview_date": positive["preview_date"].min() if not positive.empty else pd.NaT,
                "last_preview_date": positive["preview_date"].max() if not positive.empty else pd.NaT,
                "preview_gross_usd": (
                    float(positive["preview_gross_usd"].iloc[0])
                    if not positive.empty and not unclear
                    else np.nan
                ),
                "preview_validation_status": (
                    "unclear_preview_aggregation"
                    if unclear
                    else "duplicate_preview_rows"
                    if duplicate_identical
                    else "validated_preview"
                ),
            }
        )
    return pd.DataFrame(summaries)


def select_leakage_safe_baselines(preview_summary: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    if preview_summary.empty:
        return preview_summary.copy()

    required = {
        "release_run_id",
        "movie_id",
        "origin_day",
        "forecast_origin_date",
        "latest_estimate_date",
        "primary_point_forecast_usd",
        "primary_point_method",
    }
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Pre-release panel missing required columns: {sorted(missing)}")

    work = preview_summary.copy()
    work["first_preview_date"] = _date_series(work["first_preview_date"])

    panel_work = panel.copy()
    panel_work["forecast_origin_date"] = _date_series(panel_work["forecast_origin_date"])
    panel_work["latest_estimate_date"] = _date_series(panel_work["latest_estimate_date"])
    panel_work["primary_point_forecast_usd"] = pd.to_numeric(
        panel_work["primary_point_forecast_usd"], errors="coerce"
    )

    baseline_rows: list[dict[str, Any]] = []
    for row in work.itertuples(index=False):
        movie_panel = panel_work.loc[
            (panel_work["release_run_id"].eq(row.release_run_id))
            & (panel_work["movie_id"].eq(row.movie_id))
        ].copy()
        any_positive = bool(movie_panel["primary_point_forecast_usd"].gt(0).any()) if not movie_panel.empty else False
        has_panel_row = bool(not movie_panel.empty)
        candidates = movie_panel.loc[
            movie_panel["primary_point_forecast_usd"].gt(0)
            & movie_panel["forecast_origin_date"].lt(row.first_preview_date)
            & movie_panel["latest_estimate_date"].lt(row.first_preview_date)
        ].copy()
        if candidates.empty:
            baseline_rows.append(
                {
                    "release_run_id": row.release_run_id,
                    "movie_id": row.movie_id,
                    "baseline_available": False,
                    "baseline_candidate_count": 0,
                    "has_panel_row": has_panel_row,
                    "has_any_positive_baseline": any_positive,
                    "baseline_origin_day": np.nan,
                    "baseline_forecast_origin_date": pd.NaT,
                    "baseline_latest_estimate_date": pd.NaT,
                    "baseline_forecast_usd": np.nan,
                    "baseline_primary_point_method": None,
                    "baseline_source_count": np.nan,
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
                "has_any_positive_baseline": any_positive,
                "baseline_origin_day": int(selected["origin_day"]),
                "baseline_forecast_origin_date": selected["forecast_origin_date"],
                "baseline_latest_estimate_date": selected["latest_estimate_date"],
                "baseline_forecast_usd": float(selected["primary_point_forecast_usd"]),
                "baseline_primary_point_method": selected.get("primary_point_method"),
                "baseline_source_count": selected.get("source_count", np.nan),
            }
        )

    baseline = pd.DataFrame(baseline_rows)
    return work.merge(baseline, on=["release_run_id", "movie_id"], how="left")


def build_analysis_panel(preview_rows: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    preview_summary = validate_preview_gross(preview_rows)
    audited = select_leakage_safe_baselines(preview_summary, panel)
    if audited.empty:
        return audited.copy(), audited.copy()

    audited["actual_opening_weekend_gross_usd"] = pd.to_numeric(
        audited["actual_opening_weekend_gross_usd"], errors="coerce"
    )
    audited["baseline_forecast_usd"] = pd.to_numeric(audited["baseline_forecast_usd"], errors="coerce")
    audited["preview_gross_usd"] = pd.to_numeric(audited["preview_gross_usd"], errors="coerce")

    reasons = []
    included = []
    for row in audited.itertuples(index=False):
        reason = ""
        is_included = True
        if row.preview_validation_status == "unclear_preview_aggregation":
            reason = "unclear_preview_aggregation"
            is_included = False
        elif not np.isfinite(row.actual_opening_weekend_gross_usd) or row.actual_opening_weekend_gross_usd <= 0:
            reason = "invalid_actual"
            is_included = False
        elif not bool(row.baseline_available):
            reason = "missing_leakage_safe_baseline"
            if bool(row.has_panel_row) and not bool(row.has_any_positive_baseline):
                reason = "non_positive_baseline"
            is_included = False
        elif not np.isfinite(row.baseline_forecast_usd) or row.baseline_forecast_usd <= 0:
            reason = "non_positive_baseline"
            is_included = False
        elif not np.isfinite(row.preview_gross_usd) or row.preview_gross_usd <= 0:
            reason = "invalid_preview_gross"
            is_included = False
        reasons.append(reason)
        included.append(is_included)

    audited["included"] = included
    audited["exclusion_reason"] = reasons

    analysis = audited.loc[audited["included"]].copy()
    analysis["e_log"] = np.log(
        analysis["actual_opening_weekend_gross_usd"] / analysis["baseline_forecast_usd"]
    )
    analysis["x_log_preview_ratio"] = np.log(analysis["preview_gross_usd"] / analysis["baseline_forecast_usd"])
    analysis = analysis.sort_values(["opening_weekend_start", "movie_id"]).reset_index(drop=True)
    audited = audited.sort_values(["opening_weekend_start", "movie_id"]).reset_index(drop=True)
    return analysis, audited


def ols_alpha_beta(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> tuple[float, float]:
    x_arr = np.asarray(x, dtype="float64")
    y_arr = np.asarray(y, dtype="float64")
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if valid.sum() < 2:
        return np.nan, np.nan
    design = np.column_stack([np.ones(valid.sum()), x_arr[valid]])
    alpha, beta = np.linalg.lstsq(design, y_arr[valid], rcond=None)[0]
    return float(alpha), float(beta)


def build_rolling_predictions(analysis_panel: pd.DataFrame, min_train_n: int = DEFAULT_MIN_TRAIN_N) -> pd.DataFrame:
    rows = analysis_panel.sort_values(["opening_weekend_start", "movie_id"]).reset_index(drop=True).copy()
    if rows.empty:
        return rows

    outputs: list[dict[str, Any]] = []
    for idx, row in rows.iterrows():
        prior = rows.iloc[:idx].copy()
        train_n = int(len(prior))
        scored = train_n >= min_train_n
        consensus = float(row["baseline_forecast_usd"])
        bias_forecast = np.nan
        update_forecast = np.nan
        alpha = np.nan
        beta = np.nan
        historical_bias_log = np.nan

        if scored:
            historical_bias_log = float(prior["e_log"].mean())
            alpha, beta = ols_alpha_beta(prior["x_log_preview_ratio"], prior["e_log"])
            bias_forecast = float(consensus * np.exp(historical_bias_log))
            update_forecast = float(consensus * np.exp(alpha + beta * row["x_log_preview_ratio"]))

        outputs.append(
            {
                **row.to_dict(),
                "rolling_train_n": train_n,
                "scored": scored,
                "historical_preview_bias_log": historical_bias_log,
                "rolling_alpha": alpha,
                "rolling_beta": beta,
                "forecast_consensus_usd": consensus if scored else np.nan,
                "forecast_preview_status_bias_usd": bias_forecast,
                "forecast_preview_amount_update_usd": update_forecast,
            }
        )
    return pd.DataFrame(outputs)


def _model_error_frame(predictions: pd.DataFrame, model: str, forecast_col: str) -> pd.DataFrame:
    frame = predictions.loc[predictions["scored"]].copy()
    actual = pd.to_numeric(frame["actual_opening_weekend_gross_usd"], errors="coerce")
    forecast = pd.to_numeric(frame[forecast_col], errors="coerce")
    valid = actual.gt(0) & forecast.gt(0)
    out = frame.loc[valid].copy()
    out["model"] = model
    out["forecast_usd"] = forecast.loc[valid].to_numpy(dtype="float64")
    out["actual_usd"] = actual.loc[valid].to_numpy(dtype="float64")
    out["log_error"] = np.log(out["actual_usd"] / out["forecast_usd"])
    out["abs_log_error"] = out["log_error"].abs()
    out["squared_log_error"] = out["log_error"] ** 2
    out["abs_dollar_error"] = (out["actual_usd"] - out["forecast_usd"]).abs()
    return out


def evaluate_models(predictions: pd.DataFrame) -> pd.DataFrame:
    error_frames = {
        model: _model_error_frame(predictions, model, col)
        for model, col in MODEL_FORECAST_COLUMNS.items()
    }
    baseline_mae = float(error_frames["consensus"]["abs_log_error"].mean()) if not error_frames["consensus"].empty else np.nan
    bias_mae = (
        float(error_frames["preview_status_bias"]["abs_log_error"].mean())
        if not error_frames["preview_status_bias"].empty
        else np.nan
    )

    rows = []
    for model, errors in error_frames.items():
        if errors.empty:
            rows.append(
                {
                    "model": model,
                    "n": 0,
                    "ME_log": np.nan,
                    "MAE_log": np.nan,
                    "RMSE_log": np.nan,
                    "dollar_MAE": np.nan,
                    "improvement_vs_consensus": np.nan,
                    "improvement_vs_preview_status_bias": np.nan,
                    "proportion_improved_vs_consensus": np.nan,
                    "proportion_improved_vs_preview_status_bias": np.nan,
                }
            )
            continue
        mae = float(errors["abs_log_error"].mean())
        rows.append(
            {
                "model": model,
                "n": int(len(errors)),
                "ME_log": float(errors["log_error"].mean()),
                "MAE_log": mae,
                "RMSE_log": float(np.sqrt(errors["squared_log_error"].mean())),
                "dollar_MAE": float(errors["abs_dollar_error"].mean()),
                "improvement_vs_consensus": (
                    (baseline_mae - mae) / baseline_mae if np.isfinite(baseline_mae) and baseline_mae > 0 else np.nan
                ),
                "improvement_vs_preview_status_bias": (
                    (bias_mae - mae) / bias_mae if np.isfinite(bias_mae) and bias_mae > 0 else np.nan
                ),
                "proportion_improved_vs_consensus": _proportion_improved(
                    error_frames["consensus"], errors, "abs_log_error"
                ),
                "proportion_improved_vs_preview_status_bias": _proportion_improved(
                    error_frames["preview_status_bias"], errors, "abs_log_error"
                ),
            }
        )
    return pd.DataFrame(rows)


def _proportion_improved(baseline: pd.DataFrame, candidate: pd.DataFrame, loss_col: str) -> float:
    if baseline.empty or candidate.empty:
        return np.nan
    keys = ["release_run_id", "movie_id"]
    paired = baseline[keys + [loss_col]].merge(
        candidate[keys + [loss_col]], on=keys, suffixes=("_baseline", "_candidate")
    )
    if paired.empty:
        return np.nan
    return float((paired[f"{loss_col}_candidate"] < paired[f"{loss_col}_baseline"]).mean())


def evaluate_models_by_year(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for year, group in predictions.groupby("release_year", dropna=False):
        metrics = evaluate_models(group)
        metrics.insert(0, "release_year", year)
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def paired_improvement_bootstrap(
    predictions: pd.DataFrame,
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    bias = _model_error_frame(predictions, "preview_status_bias", MODEL_FORECAST_COLUMNS["preview_status_bias"])
    update = _model_error_frame(
        predictions, "preview_amount_update", MODEL_FORECAST_COLUMNS["preview_amount_update"]
    )
    keys = ["release_run_id", "movie_id"]
    paired = bias.merge(update, on=keys, suffixes=("_bias", "_update"))
    if paired.empty:
        return pd.DataFrame()

    specs = {
        "abs_log_error": paired["abs_log_error_update"].to_numpy() - paired["abs_log_error_bias"].to_numpy(),
        "squared_log_error": paired["squared_log_error_update"].to_numpy()
        - paired["squared_log_error_bias"].to_numpy(),
        "abs_dollar_error": paired["abs_dollar_error_update"].to_numpy()
        - paired["abs_dollar_error_bias"].to_numpy(),
    }
    rng = np.random.default_rng(seed)
    output = []
    n = len(paired)
    for metric, diff in specs.items():
        draws = rng.integers(0, n, size=(resamples, n))
        means = diff[draws].mean(axis=1)
        output.append(
            {
                "comparison": "preview_amount_update_minus_preview_status_bias",
                "loss_metric": metric,
                "n": int(n),
                "bootstrap_resamples": int(resamples),
                "seed": int(seed),
                "mean_loss_difference": float(diff.mean()),
                "ci_lower_95": float(np.quantile(means, 0.025)),
                "ci_upper_95": float(np.quantile(means, 0.975)),
                "p_loss_difference_lt_0": float((means < 0).mean()),
            }
        )
    return pd.DataFrame(output)


def build_training_policy_predictions(
    analysis_panel: pd.DataFrame,
    *,
    min_current_train_n: int = DEFAULT_MIN_TRAIN_N,
) -> pd.DataFrame:
    """Compare post-2022-only vs pooled non-COVID expanding training windows.

    Both policies score only common current-era test movies. The current-regime
    policy trains on prior movies from 2023 onward. The pooled policy trains on
    all pre-2020 leakage-safe movies plus prior movies from 2023 onward.
    """

    rows = analysis_panel.sort_values(["opening_weekend_start", "movie_id"]).reset_index(drop=True).copy()
    if rows.empty:
        return pd.DataFrame()

    pre_2020 = rows.loc[rows["release_year"].lt(2020)].copy()
    current = rows.loc[rows["release_year"].gt(2022)].copy()
    outputs: list[dict[str, Any]] = []

    for _, test_row in current.iterrows():
        earlier_current = current.loc[
            (current["opening_weekend_start"].lt(test_row["opening_weekend_start"]))
            | (
                current["opening_weekend_start"].eq(test_row["opening_weekend_start"])
                & current["movie_id"].lt(test_row["movie_id"])
            )
        ].copy()
        if len(earlier_current) < min_current_train_n:
            continue

        policy_training_frames = {
            "post_2022_only": earlier_current,
            "pooled_non_covid": pd.concat([pre_2020, earlier_current], ignore_index=True),
        }
        for training_policy, train in policy_training_frames.items():
            alpha, beta = ols_alpha_beta(train["x_log_preview_ratio"], train["e_log"])
            baseline = float(test_row["baseline_forecast_usd"])
            forecast = float(baseline * np.exp(alpha + beta * test_row["x_log_preview_ratio"]))
            outputs.append(
                {
                    **test_row.to_dict(),
                    "training_policy": training_policy,
                    "scored": True,
                    "current_train_n": int(len(earlier_current)),
                    "pooled_pre_2020_train_n": int(len(pre_2020)) if training_policy == "pooled_non_covid" else 0,
                    "training_n": int(len(train)),
                    "rolling_alpha": alpha,
                    "rolling_beta": beta,
                    "forecast_consensus_usd": baseline,
                    TRAINING_POLICY_FORECAST_COL: forecast,
                }
            )

    return pd.DataFrame(outputs)


def _policy_error_frame(predictions: pd.DataFrame, policy: str) -> pd.DataFrame:
    frame = predictions.loc[predictions["training_policy"].eq(policy)].copy()
    actual = pd.to_numeric(frame["actual_opening_weekend_gross_usd"], errors="coerce")
    forecast = pd.to_numeric(frame[TRAINING_POLICY_FORECAST_COL], errors="coerce")
    valid = actual.gt(0) & forecast.gt(0)
    out = frame.loc[valid].copy()
    out["actual_usd"] = actual.loc[valid].to_numpy(dtype="float64")
    out["forecast_usd"] = forecast.loc[valid].to_numpy(dtype="float64")
    out["log_error"] = np.log(out["actual_usd"] / out["forecast_usd"])
    out["abs_log_error"] = out["log_error"].abs()
    out["squared_log_error"] = out["log_error"] ** 2
    out["abs_dollar_error"] = (out["actual_usd"] - out["forecast_usd"]).abs()
    return out


def evaluate_training_policies(predictions: pd.DataFrame) -> pd.DataFrame:
    policies = ["post_2022_only", "pooled_non_covid"]
    errors = {policy: _policy_error_frame(predictions, policy) for policy in policies}
    baseline = errors["post_2022_only"]
    rows = []
    for policy in policies:
        frame = errors[policy]
        if frame.empty:
            rows.append(
                {
                    "training_policy": policy,
                    "n": 0,
                    "ME_log": np.nan,
                    "MAE_log": np.nan,
                    "RMSE_log": np.nan,
                    "dollar_MAE": np.nan,
                    "improvement_vs_post_2022_only": np.nan,
                    "movie_win_rate_vs_post_2022_only": np.nan,
                }
            )
            continue
        mae = float(frame["abs_log_error"].mean())
        rmse = float(np.sqrt(frame["squared_log_error"].mean()))
        baseline_mae = float(baseline["abs_log_error"].mean()) if not baseline.empty else np.nan
        rows.append(
            {
                "training_policy": policy,
                "n": int(len(frame)),
                "ME_log": float(frame["log_error"].mean()),
                "MAE_log": mae,
                "RMSE_log": rmse,
                "dollar_MAE": float(frame["abs_dollar_error"].mean()),
                "improvement_vs_post_2022_only": (
                    (baseline_mae - mae) / baseline_mae
                    if policy != "post_2022_only" and np.isfinite(baseline_mae) and baseline_mae > 0
                    else 0.0
                    if policy == "post_2022_only"
                    else np.nan
                ),
                "movie_win_rate_vs_post_2022_only": _policy_win_rate(baseline, frame),
            }
        )
    return pd.DataFrame(rows)


def _policy_win_rate(baseline: pd.DataFrame, candidate: pd.DataFrame) -> float:
    if baseline.empty or candidate.empty:
        return np.nan
    paired = baseline[["release_run_id", "movie_id", "abs_log_error"]].merge(
        candidate[["release_run_id", "movie_id", "abs_log_error"]],
        on=["release_run_id", "movie_id"],
        suffixes=("_baseline", "_candidate"),
    )
    if paired.empty:
        return np.nan
    return float((paired["abs_log_error_candidate"] < paired["abs_log_error_baseline"]).mean())


def evaluate_training_policies_by_year(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for year, group in predictions.groupby("release_year", dropna=False):
        metrics = evaluate_training_policies(group)
        metrics.insert(0, "release_year", year)
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def training_policy_paired_bootstrap(
    predictions: pd.DataFrame,
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    current = _policy_error_frame(predictions, "post_2022_only")
    pooled = _policy_error_frame(predictions, "pooled_non_covid")
    paired = current.merge(
        pooled,
        on=["release_run_id", "movie_id"],
        suffixes=("_post_2022_only", "_pooled_non_covid"),
    )
    if paired.empty:
        return pd.DataFrame()

    specs = {
        "abs_log_error": paired["abs_log_error_pooled_non_covid"].to_numpy()
        - paired["abs_log_error_post_2022_only"].to_numpy(),
        "squared_log_error": paired["squared_log_error_pooled_non_covid"].to_numpy()
        - paired["squared_log_error_post_2022_only"].to_numpy(),
        "abs_dollar_error": paired["abs_dollar_error_pooled_non_covid"].to_numpy()
        - paired["abs_dollar_error_post_2022_only"].to_numpy(),
    }
    rng = np.random.default_rng(seed)
    output = []
    n = len(paired)
    for metric, diff in specs.items():
        draws = rng.integers(0, n, size=(resamples, n))
        means = diff[draws].mean(axis=1)
        output.append(
            {
                "comparison": "pooled_non_covid_minus_post_2022_only",
                "loss_metric": metric,
                "n": int(n),
                "bootstrap_resamples": int(resamples),
                "seed": int(seed),
                "mean_loss_difference": float(diff.mean()),
                "ci_lower_95": float(np.quantile(means, 0.025)),
                "ci_upper_95": float(np.quantile(means, 0.975)),
                "p_loss_difference_lt_0": float((means < 0).mean()),
            }
        )
    return pd.DataFrame(output)


def build_downstream_live_friday_predictions(
    training_policy_predictions: pd.DataFrame,
    daily_baseline: pd.DataFrame,
    *,
    training_policy: str = "post_2022_only",
) -> pd.DataFrame:
    """Apply the rolling Thursday update to existing Live-Friday daily components.

    The daily baseline artifact is the downstream pre-weekend component pipeline.
    This recomputes all pre-weekend daily components by preserving existing daily
    shape and replacing the old OW scale with the rolling Thursday update.
    """

    if training_policy_predictions.empty or daily_baseline.empty:
        return pd.DataFrame()
    required = {
        "release_run_id",
        "movie_id",
        "pre_fri_usd",
        "pre_sat_usd",
        "pre_sun_usd",
        "actual_ow_usd",
        "total_forecast_usd",
    }
    missing = required - set(daily_baseline.columns)
    if missing:
        raise ValueError(f"Daily baseline missing required columns: {sorted(missing)}")

    policy = training_policy_predictions.loc[
        training_policy_predictions["training_policy"].eq(training_policy)
    ].copy()
    if policy.empty:
        return pd.DataFrame()

    daily_cols = [
        "release_run_id",
        "movie_id",
        "pre_fri_usd",
        "pre_sat_usd",
        "pre_sun_usd",
        "actual_ow_usd",
        "total_forecast_usd",
        "forecast_origin_date",
        "total_forecast_source",
        "shape_model",
    ]
    daily_cols = [column for column in daily_cols if column in daily_baseline.columns]
    merged = policy.merge(
        daily_baseline[daily_cols],
        on=["release_run_id", "movie_id"],
        how="left",
        suffixes=("", "_daily"),
    )
    for column in [
        "pre_fri_usd",
        "pre_sat_usd",
        "pre_sun_usd",
        "actual_ow_usd",
        "total_forecast_usd",
        TRAINING_POLICY_FORECAST_COL,
    ]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    valid = (
        merged["pre_fri_usd"].gt(0)
        & merged["pre_sat_usd"].gt(0)
        & merged["pre_sun_usd"].gt(0)
        & merged["actual_ow_usd"].gt(0)
        & merged["total_forecast_usd"].gt(0)
        & merged[TRAINING_POLICY_FORECAST_COL].gt(0)
    )
    out = merged.loc[valid].copy()
    out["original_friday_component_usd"] = out["pre_fri_usd"]
    out["original_saturday_component_usd"] = out["pre_sat_usd"]
    out["original_sunday_component_usd"] = out["pre_sun_usd"]
    out["forecast_original_consensus_prior_usd"] = (
        out["pre_fri_usd"] + out["pre_sat_usd"] + out["pre_sun_usd"]
    )
    out["pipeline_old_ow_scale_usd"] = out["total_forecast_usd"]
    out["thursday_preview_updated_ow_usd"] = out[TRAINING_POLICY_FORECAST_COL]
    out["component_rescale_factor"] = out["thursday_preview_updated_ow_usd"] / out["pipeline_old_ow_scale_usd"]
    out["updated_friday_component_usd"] = out["pre_fri_usd"] * out["component_rescale_factor"]
    out["updated_saturday_component_usd"] = out["pre_sat_usd"] * out["component_rescale_factor"]
    out["updated_sunday_component_usd"] = out["pre_sun_usd"] * out["component_rescale_factor"]
    out["forecast_thursday_preview_prior_usd"] = (
        out["updated_friday_component_usd"]
        + out["updated_saturday_component_usd"]
        + out["updated_sunday_component_usd"]
    )
    out["actual_opening_weekend_gross_usd"] = out["actual_ow_usd"]
    out["live_friday_origin"] = "pre_weekend_daily_baseline"
    return out.sort_values(["opening_weekend_start", "movie_id"]).reset_index(drop=True)


def evaluate_downstream_live_friday(predictions: pd.DataFrame) -> pd.DataFrame:
    specs = {
        "original_consensus_prior": "forecast_original_consensus_prior_usd",
        "thursday_preview_updated_prior": "forecast_thursday_preview_prior_usd",
    }
    errors = {
        model: _downstream_error_frame(predictions, model, forecast_col)
        for model, forecast_col in specs.items()
    }
    baseline = errors["original_consensus_prior"]
    baseline_mae = float(baseline["abs_log_error"].mean()) if not baseline.empty else np.nan
    rows = []
    for model, frame in errors.items():
        if frame.empty:
            rows.append(
                {
                    "model": model,
                    "n": 0,
                    "ME_log": np.nan,
                    "MAE_log": np.nan,
                    "RMSE_log": np.nan,
                    "dollar_MAE": np.nan,
                    "improvement_vs_original_consensus_prior": np.nan,
                    "movie_win_rate_vs_original_consensus_prior": np.nan,
                }
            )
            continue
        mae = float(frame["abs_log_error"].mean())
        rows.append(
            {
                "model": model,
                "n": int(len(frame)),
                "ME_log": float(frame["log_error"].mean()),
                "MAE_log": mae,
                "RMSE_log": float(np.sqrt(frame["squared_log_error"].mean())),
                "dollar_MAE": float(frame["abs_dollar_error"].mean()),
                "improvement_vs_original_consensus_prior": (
                    (baseline_mae - mae) / baseline_mae
                    if model != "original_consensus_prior" and np.isfinite(baseline_mae) and baseline_mae > 0
                    else 0.0
                    if model == "original_consensus_prior"
                    else np.nan
                ),
                "movie_win_rate_vs_original_consensus_prior": _policy_win_rate(baseline, frame),
            }
        )
    return pd.DataFrame(rows)


def _downstream_error_frame(predictions: pd.DataFrame, model: str, forecast_col: str) -> pd.DataFrame:
    frame = predictions.copy()
    actual = pd.to_numeric(frame["actual_opening_weekend_gross_usd"], errors="coerce")
    forecast = pd.to_numeric(frame[forecast_col], errors="coerce")
    valid = actual.gt(0) & forecast.gt(0)
    out = frame.loc[valid].copy()
    out["model"] = model
    out["actual_usd"] = actual.loc[valid].to_numpy(dtype="float64")
    out["forecast_usd"] = forecast.loc[valid].to_numpy(dtype="float64")
    out["log_error"] = np.log(out["actual_usd"] / out["forecast_usd"])
    out["abs_log_error"] = out["log_error"].abs()
    out["squared_log_error"] = out["log_error"] ** 2
    out["abs_dollar_error"] = (out["actual_usd"] - out["forecast_usd"]).abs()
    return out


def downstream_live_friday_bootstrap(
    predictions: pd.DataFrame,
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    original = _downstream_error_frame(
        predictions,
        "original_consensus_prior",
        "forecast_original_consensus_prior_usd",
    )
    updated = _downstream_error_frame(
        predictions,
        "thursday_preview_updated_prior",
        "forecast_thursday_preview_prior_usd",
    )
    paired = original.merge(
        updated,
        on=["release_run_id", "movie_id"],
        suffixes=("_original", "_updated"),
    )
    if paired.empty:
        return pd.DataFrame()
    specs = {
        "abs_log_error": paired["abs_log_error_updated"].to_numpy()
        - paired["abs_log_error_original"].to_numpy(),
        "squared_log_error": paired["squared_log_error_updated"].to_numpy()
        - paired["squared_log_error_original"].to_numpy(),
        "abs_dollar_error": paired["abs_dollar_error_updated"].to_numpy()
        - paired["abs_dollar_error_original"].to_numpy(),
    }
    rng = np.random.default_rng(seed)
    output = []
    n = len(paired)
    for metric, diff in specs.items():
        draws = rng.integers(0, n, size=(resamples, n))
        means = diff[draws].mean(axis=1)
        output.append(
            {
                "comparison": "thursday_preview_updated_prior_minus_original_consensus_prior",
                "loss_metric": metric,
                "n": int(n),
                "bootstrap_resamples": int(resamples),
                "seed": int(seed),
                "mean_loss_difference": float(diff.mean()),
                "ci_lower_95": float(np.quantile(means, 0.025)),
                "ci_upper_95": float(np.quantile(means, 0.975)),
                "p_loss_difference_lt_0": float((means < 0).mean()),
            }
        )
    return pd.DataFrame(output)


def baseline_size_bucket(value: Any) -> str:
    point = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(point) or float(point) <= 0:
        return "unknown"
    point = float(point)
    if point < 5_000_000:
        return "lt_5m"
    if point < 15_000_000:
        return "5m_15m"
    if point < 50_000_000:
        return "15m_50m"
    if point < 100_000_000:
        return "50m_100m"
    return "100m_plus"


def build_downstream_influence_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "movie_id": predictions["movie_id"],
            "release_run_id": predictions["release_run_id"],
            "title": predictions["title"],
            "release_year": predictions["release_year"],
            "original_forecast": predictions["forecast_original_consensus_prior_usd"],
            "updated_forecast": predictions["forecast_thursday_preview_prior_usd"],
            "actual_ow": predictions["actual_opening_weekend_gross_usd"],
        }
    )
    out["baseline_size_bucket"] = out["original_forecast"].map(baseline_size_bucket)
    out["original_abs_log_error"] = np.abs(np.log(out["actual_ow"] / out["original_forecast"]))
    out["updated_abs_log_error"] = np.abs(np.log(out["actual_ow"] / out["updated_forecast"]))
    out["delta_abs_log_error"] = out["updated_abs_log_error"] - out["original_abs_log_error"]
    out["original_abs_dollar_error"] = np.abs(out["actual_ow"] - out["original_forecast"])
    out["updated_abs_dollar_error"] = np.abs(out["actual_ow"] - out["updated_forecast"])
    out["delta_abs_dollar_error"] = out["updated_abs_dollar_error"] - out["original_abs_dollar_error"]
    return out.sort_values(["release_year", "movie_id"]).reset_index(drop=True)


def downstream_influence_bucket_metrics(influence: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bucket_order = ["lt_5m", "5m_15m", "15m_50m", "50m_100m", "100m_plus", "unknown"]
    for bucket in bucket_order:
        group = influence.loc[influence["baseline_size_bucket"].eq(bucket)]
        if group.empty:
            continue
        rows.append(_paired_influence_metrics(group, {"baseline_size_bucket": bucket}))
    return pd.DataFrame(rows)


def downstream_leave_one_out(influence: pd.DataFrame) -> pd.DataFrame:
    full = _paired_influence_metrics(influence, {})
    rows = []
    for row in influence.itertuples(index=False):
        remaining = influence.loc[~influence["movie_id"].eq(row.movie_id)].copy()
        metrics = _paired_influence_metrics(remaining, {})
        rows.append(
            {
                "movie_id": row.movie_id,
                "release_run_id": row.release_run_id,
                "title": row.title,
                "release_year": row.release_year,
                "baseline_size_bucket": row.baseline_size_bucket,
                "removed_delta_abs_log_error": row.delta_abs_log_error,
                "removed_delta_abs_dollar_error": row.delta_abs_dollar_error,
                "loo_n": metrics["n"],
                "loo_delta_MAE_log": metrics["delta_MAE_log"],
                "loo_delta_RMSE_log": metrics["delta_RMSE_log"],
                "loo_delta_dollar_MAE": metrics["delta_dollar_MAE"],
                "change_in_delta_MAE_log": metrics["delta_MAE_log"] - full["delta_MAE_log"],
                "change_in_delta_RMSE_log": metrics["delta_RMSE_log"] - full["delta_RMSE_log"],
                "change_in_delta_dollar_MAE": metrics["delta_dollar_MAE"] - full["delta_dollar_MAE"],
            }
        )
    return pd.DataFrame(rows).sort_values(
        "change_in_delta_dollar_MAE",
        key=lambda s: s.abs(),
        ascending=False,
    ).reset_index(drop=True)


def downstream_top_influence_removed_metrics(influence: pd.DataFrame) -> pd.DataFrame:
    if influence.empty:
        return pd.DataFrame()
    loo = downstream_leave_one_out(influence)
    top = loo.iloc[0]
    remaining = influence.loc[~influence["movie_id"].eq(top["movie_id"])].copy()
    rows = [
        _paired_influence_metrics(influence, {"sample": "full_sample"}),
        _paired_influence_metrics(
            remaining,
            {
                "sample": "top_dollar_influence_removed",
                "removed_movie_id": top["movie_id"],
                "removed_title": top["title"],
            },
        ),
    ]
    return pd.DataFrame(rows)


def _paired_influence_metrics(frame: pd.DataFrame, extra: dict[str, Any]) -> dict[str, Any]:
    if frame.empty:
        base = {
            "n": 0,
            "original_MAE_log": np.nan,
            "updated_MAE_log": np.nan,
            "delta_MAE_log": np.nan,
            "original_RMSE_log": np.nan,
            "updated_RMSE_log": np.nan,
            "delta_RMSE_log": np.nan,
            "original_dollar_MAE": np.nan,
            "updated_dollar_MAE": np.nan,
            "delta_dollar_MAE": np.nan,
            "movie_win_rate_log": np.nan,
            "movie_win_rate_dollar": np.nan,
        }
        return {**extra, **base}
    original_sq = frame["original_abs_log_error"] ** 2
    updated_sq = frame["updated_abs_log_error"] ** 2
    original_mae = float(frame["original_abs_log_error"].mean())
    updated_mae = float(frame["updated_abs_log_error"].mean())
    original_rmse = float(np.sqrt(original_sq.mean()))
    updated_rmse = float(np.sqrt(updated_sq.mean()))
    original_dollar = float(frame["original_abs_dollar_error"].mean())
    updated_dollar = float(frame["updated_abs_dollar_error"].mean())
    return {
        **extra,
        "n": int(len(frame)),
        "original_MAE_log": original_mae,
        "updated_MAE_log": updated_mae,
        "delta_MAE_log": updated_mae - original_mae,
        "original_RMSE_log": original_rmse,
        "updated_RMSE_log": updated_rmse,
        "delta_RMSE_log": updated_rmse - original_rmse,
        "original_dollar_MAE": original_dollar,
        "updated_dollar_MAE": updated_dollar,
        "delta_dollar_MAE": updated_dollar - original_dollar,
        "movie_win_rate_log": float(frame["delta_abs_log_error"].lt(0).mean()),
        "movie_win_rate_dollar": float(frame["delta_abs_dollar_error"].lt(0).mean()),
    }


def write_residual_relationship_plot(analysis_panel: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise SystemExit(
            "matplotlib is required to write thursday_preview_residual_relationship.png"
        ) from exc

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        analysis_panel["x_log_preview_ratio"],
        analysis_panel["e_log"],
        alpha=0.75,
        edgecolor="none",
    )
    if len(analysis_panel) >= 2:
        alpha, beta = ols_alpha_beta(analysis_panel["x_log_preview_ratio"], analysis_panel["e_log"])
        xs = np.linspace(
            float(analysis_panel["x_log_preview_ratio"].min()),
            float(analysis_panel["x_log_preview_ratio"].max()),
            100,
        )
        ax.plot(xs, alpha + beta * xs, color="#b22222", linewidth=2)
    ax.axhline(0.0, color="#777777", linewidth=1, linestyle="--")
    ax.set_xlabel("log(Thursday preview gross / pre-preview consensus)")
    ax.set_ylabel("log(actual Friday-Sunday OW / pre-preview consensus)")
    ax.set_title("Thursday Preview Strength vs Consensus Residual")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_diagnostic(
    *,
    preview_rows: pd.DataFrame,
    panel: pd.DataFrame,
    daily_baseline: pd.DataFrame | None = None,
    min_train_n: int = DEFAULT_MIN_TRAIN_N,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> DiagnosticOutputs:
    analysis, audit = build_analysis_panel(preview_rows, panel)
    rolling = build_rolling_predictions(analysis, min_train_n=min_train_n)
    comparison = evaluate_models(rolling)
    year = evaluate_models_by_year(rolling)
    bootstrap = paired_improvement_bootstrap(
        rolling, resamples=bootstrap_resamples, seed=bootstrap_seed
    )
    policy_predictions = build_training_policy_predictions(analysis, min_current_train_n=min_train_n)
    policy_comparison = evaluate_training_policies(policy_predictions)
    policy_year = evaluate_training_policies_by_year(policy_predictions)
    policy_bootstrap = training_policy_paired_bootstrap(
        policy_predictions,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    daily = daily_baseline if daily_baseline is not None else pd.DataFrame()
    downstream_predictions = build_downstream_live_friday_predictions(policy_predictions, daily)
    downstream_comparison = evaluate_downstream_live_friday(downstream_predictions)
    downstream_bootstrap = downstream_live_friday_bootstrap(
        downstream_predictions,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    downstream_influence = build_downstream_influence_audit(downstream_predictions)
    downstream_bucket = downstream_influence_bucket_metrics(downstream_influence)
    downstream_loo = downstream_leave_one_out(downstream_influence)
    downstream_removed = downstream_top_influence_removed_metrics(downstream_influence)
    return DiagnosticOutputs(
        analysis_panel=analysis,
        exclusion_audit=audit,
        rolling_predictions=rolling,
        model_comparison=comparison,
        year_stability=year,
        paired_bootstrap=bootstrap,
        training_policy_predictions=policy_predictions,
        training_policy_comparison=policy_comparison,
        training_policy_year_stability=policy_year,
        training_policy_bootstrap=policy_bootstrap,
        downstream_live_friday_predictions=downstream_predictions,
        downstream_live_friday_comparison=downstream_comparison,
        downstream_live_friday_bootstrap=downstream_bootstrap,
        downstream_live_friday_influence_audit=downstream_influence,
        downstream_live_friday_bucket_metrics=downstream_bucket,
        downstream_live_friday_leave_one_out=downstream_loo,
        downstream_live_friday_top_influence_removed=downstream_removed,
    )


def write_outputs(outputs: DiagnosticOutputs, output_dir: Path, *, write_plot: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs.analysis_panel.to_csv(output_dir / "thursday_preview_analysis_panel.csv", index=False)
    outputs.exclusion_audit.to_csv(output_dir / "thursday_preview_exclusion_audit.csv", index=False)
    outputs.rolling_predictions.to_csv(output_dir / "thursday_preview_rolling_predictions.csv", index=False)
    outputs.model_comparison.to_csv(output_dir / "thursday_preview_model_comparison.csv", index=False)
    outputs.year_stability.to_csv(output_dir / "thursday_preview_year_stability.csv", index=False)
    outputs.paired_bootstrap.to_csv(output_dir / "thursday_preview_paired_improvement_bootstrap.csv", index=False)
    outputs.training_policy_predictions.to_csv(
        output_dir / "thursday_preview_training_policy_predictions.csv",
        index=False,
    )
    outputs.training_policy_comparison.to_csv(
        output_dir / "thursday_preview_training_policy_comparison.csv",
        index=False,
    )
    outputs.training_policy_year_stability.to_csv(
        output_dir / "thursday_preview_training_policy_year_stability.csv",
        index=False,
    )
    outputs.training_policy_bootstrap.to_csv(
        output_dir / "thursday_preview_training_policy_bootstrap.csv",
        index=False,
    )
    outputs.downstream_live_friday_predictions.to_csv(
        output_dir / "thursday_preview_live_friday_downstream_predictions.csv",
        index=False,
    )
    outputs.downstream_live_friday_comparison.to_csv(
        output_dir / "thursday_preview_live_friday_downstream_comparison.csv",
        index=False,
    )
    outputs.downstream_live_friday_bootstrap.to_csv(
        output_dir / "thursday_preview_live_friday_downstream_bootstrap.csv",
        index=False,
    )
    outputs.downstream_live_friday_influence_audit.to_csv(
        output_dir / "thursday_preview_live_friday_downstream_influence_audit.csv",
        index=False,
    )
    outputs.downstream_live_friday_bucket_metrics.to_csv(
        output_dir / "thursday_preview_live_friday_downstream_bucket_metrics.csv",
        index=False,
    )
    outputs.downstream_live_friday_leave_one_out.to_csv(
        output_dir / "thursday_preview_live_friday_downstream_leave_one_out.csv",
        index=False,
    )
    outputs.downstream_live_friday_top_influence_removed.to_csv(
        output_dir / "thursday_preview_live_friday_downstream_top_influence_removed.csv",
        index=False,
    )
    if write_plot:
        write_residual_relationship_plot(
            outputs.analysis_panel,
            output_dir / "thursday_preview_residual_relationship.png",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_database_arg(parser)
    parser.add_argument("--panel", type=Path, default=None, help="Pre-release panel path. Defaults to active model.")
    parser.add_argument("--output-dir", type=Path, default=DIAGNOSTICS_DIR)
    parser.add_argument("--start-year", type=int, default=2009, help="Earliest preview movie year to fetch.")
    parser.add_argument("--min-train-n", type=int, default=DEFAULT_MIN_TRAIN_N)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--skip-plot", action="store_true", help="Write CSV outputs only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = load_pre_release_panel(args.panel)
    artifact_dir = active_pre_release_panel_path().parent if args.panel is None else args.panel.parent
    daily_baseline_path = artifact_dir / "daily_regime_baseline_forecasts.csv"
    daily_baseline = pd.read_csv(daily_baseline_path) if daily_baseline_path.exists() else pd.DataFrame()
    preview_rows = fetch_preview_rows(args.database_url, start_year=args.start_year)
    outputs = run_diagnostic(
        preview_rows=preview_rows,
        panel=panel,
        daily_baseline=daily_baseline,
        min_train_n=args.min_train_n,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_outputs(outputs, args.output_dir, write_plot=not args.skip_plot)
    print(
        "Thursday preview diagnostic complete: "
        f"{len(outputs.analysis_panel)} included movies, "
        f"{int(outputs.rolling_predictions['scored'].sum()) if not outputs.rolling_predictions.empty else 0} scored."
    )


if __name__ == "__main__":
    main()
