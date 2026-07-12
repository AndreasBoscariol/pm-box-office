#!/usr/bin/env python3
"""Nested validation for pre-release box-office predictive distributions.

This stage treats the frozen daily point policy as the center of the forecast
and evaluates full bucket-probability distributions around that center.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eda.Prior.fallback_adjusted_daily_policy_validation import (
    OUTPUT_DIR,
    production_source_panel,
    prepare_interval_features,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTICS_DIR = REPO_ROOT / "data" / "diagnostics"
OOF_PATH = OUTPUT_DIR / "locked_oof_daily_point_residual_panel.csv"
SOURCE_PATH = DIAGNOSTICS_DIR / "rolling_forecast_origin_source_estimates.csv"
POLICY_PATH = OUTPUT_DIR / "frozen_simplified_daily_point_policy.json"

MIN_ORIGIN_RESIDUALS = 30
MIN_SOURCE_RESIDUALS = 20
MIN_GROUP_RESIDUALS = 30
BUCKET_FLOOR = 1e-5
SELECTION_TOLERANCE_LOGLOSS = 0.005
QUANTILE_GRID = np.linspace(0.001, 0.999, 201)


@dataclass(frozen=True)
class GridSpec:
    name: str
    edges: np.ndarray


def write_frame(frame: pd.DataFrame, parquet_path: Path) -> Path:
    """Write parquet when an engine exists; otherwise write a CSV fallback."""
    try:
        frame.to_parquet(parquet_path, index=False)
        return parquet_path
    except ImportError:
        csv_path = parquet_path.with_suffix(".csv")
        frame.to_csv(csv_path, index=False)
        return csv_path


def canonical_grids() -> list[GridSpec]:
    return [
        GridSpec("market_2m_20_50", np.array([0, *range(20_000_000, 52_000_000, 2_000_000), np.inf], dtype=float)),
        GridSpec("canonical_2m_0_100", np.array([0, *range(2_000_000, 102_000_000, 2_000_000), np.inf], dtype=float)),
        GridSpec("canonical_5m_0_150", np.array([0, *range(5_000_000, 155_000_000, 5_000_000), np.inf], dtype=float)),
    ]


def empirical_cdf(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.full_like(x, np.nan, dtype=float)
    sorted_values = np.sort(values)
    return np.searchsorted(sorted_values, x, side="right") / sorted_values.size


def smooth_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    probs = np.where(np.isfinite(probs) & (probs > 0), probs, 0.0)
    probs = probs + BUCKET_FLOOR
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(len(probs), 1.0 / len(probs))
    return probs / total


def residual_thresholds(point: float, edges: np.ndarray) -> np.ndarray:
    thresholds = np.empty(len(edges), dtype=float)
    thresholds[0] = -np.inf
    thresholds[-1] = np.inf
    thresholds[1:-1] = np.nan
    if np.isfinite(point) and point > 0:
        finite_edges = np.isfinite(edges[1:-1]) & (edges[1:-1] > 0)
        middle = np.full(len(edges) - 2, -np.inf, dtype=float)
        middle[finite_edges] = np.log(edges[1:-1][finite_edges] / point)
        thresholds[1:-1] = middle
    else:
        thresholds[1:-1] = -np.inf
    return thresholds


def probs_from_residuals(point: float, residuals: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thresholds = residual_thresholds(point, edges)
    cdf = empirical_cdf(residuals, thresholds)
    cdf[0] = 0.0
    cdf[-1] = 1.0
    cdf = np.maximum.accumulate(np.nan_to_num(cdf, nan=0.0))
    return smooth_probs(np.diff(cdf)), cdf


def fit_size_scale(train: pd.DataFrame) -> tuple[float, float] | None:
    residual = pd.to_numeric(train["signed_log_error"], errors="coerce")
    point = pd.to_numeric(train["oof_point_forecast_usd"], errors="coerce")
    mask = residual.notna() & point.gt(0)
    if int(mask.sum()) < MIN_ORIGIN_RESIDUALS:
        return None
    x = np.log(point.loc[mask].to_numpy(dtype=float))
    y = np.log(np.abs(residual.loc[mask].to_numpy(dtype=float)) + 0.03)
    if np.nanstd(x) < 1e-8:
        return float(np.nanmedian(y)), 0.0
    beta, alpha = np.polyfit(x, y, 1)
    return float(alpha), float(beta)


def point_scale(point: float, fit: tuple[float, float] | None) -> float:
    if fit is None or not np.isfinite(point) or point <= 0:
        return 1.0
    alpha, beta = fit
    return float(np.clip(np.exp(alpha + beta * np.log(point)), 0.05, 2.5))


def probs_d1_size(train: pd.DataFrame, point: float, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fit = fit_size_scale(train)
    if fit is None:
        return probs_from_residuals(point, train["signed_log_error"].to_numpy(dtype=float), edges)
    train_point = pd.to_numeric(train["oof_point_forecast_usd"], errors="coerce").to_numpy(dtype=float)
    residual = pd.to_numeric(train["signed_log_error"], errors="coerce").to_numpy(dtype=float)
    scales = np.array([point_scale(value, fit) for value in train_point])
    standardized = residual / scales
    this_scale = point_scale(point, fit)
    thresholds = residual_thresholds(point, edges) / this_scale
    cdf = empirical_cdf(standardized, thresholds)
    cdf[0] = 0.0
    cdf[-1] = 1.0
    cdf = np.maximum.accumulate(np.nan_to_num(cdf, nan=0.0))
    return smooth_probs(np.diff(cdf)), cdf


def prepare_oof() -> pd.DataFrame:
    oof = pd.read_csv(OOF_PATH, low_memory=False)
    oof["opening_weekend_start"] = pd.to_datetime(oof["opening_weekend_start"], errors="coerce")
    return prepare_interval_features(oof)


def prepare_source_panel() -> pd.DataFrame:
    source = pd.read_csv(SOURCE_PATH, low_memory=False)
    source = production_source_panel(source)
    source["opening_weekend_start"] = pd.to_datetime(source["opening_weekend_start"], errors="coerce")
    numeric_cols = [
        "source_bias_adjusted_estimate_mid_usd",
        "source_bias_adjusted_residual_log",
        "source_bias_adjusted_reliability_raw_weight",
    ]
    for col in numeric_cols:
        source[col] = pd.to_numeric(source[col], errors="coerce")
    return source


def source_train_map(source_train: pd.DataFrame, origin_day: int) -> dict[str, np.ndarray]:
    frame = source_train.loc[source_train["origin_day"].eq(origin_day)].copy()
    out: dict[str, np.ndarray] = {}
    for source, group in frame.groupby("estimate_source"):
        residual = group["source_bias_adjusted_residual_log"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if len(residual) >= MIN_SOURCE_RESIDUALS:
            out[str(source)] = residual
    return out


def row_sources(source_holdout: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    mask = (
        source_holdout["movie_id"].eq(row["movie_id"])
        & source_holdout["opening_weekend_start"].eq(row["opening_weekend_start"])
        & source_holdout["origin_day"].eq(row["origin_day"])
    )
    out = source_holdout.loc[mask].copy()
    out = out.loc[out["source_bias_adjusted_estimate_mid_usd"].gt(0)]
    return out


def probs_d3_linear_pool(
    row: pd.Series,
    available_sources: pd.DataFrame,
    residual_by_source: dict[str, np.ndarray],
    origin_residuals: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    pieces = []
    weights = []
    fallback_count = 0
    for _, source_row in available_sources.iterrows():
        source = str(source_row["estimate_source"])
        residuals = residual_by_source.get(source)
        if residuals is None:
            residuals = origin_residuals
            fallback_count += 1
        probs, cdf = probs_from_residuals(float(source_row["source_bias_adjusted_estimate_mid_usd"]), residuals, edges)
        pieces.append((probs, cdf))
        weight = source_row.get("source_bias_adjusted_reliability_raw_weight", np.nan)
        weights.append(float(weight) if np.isfinite(weight) and weight > 0 else 1.0)
    if not pieces:
        probs, cdf = probs_from_residuals(float(row["oof_point_forecast_usd"]), origin_residuals, edges)
        return probs, cdf, "D0_no_sources"
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    probs = smooth_probs(np.sum([w[i] * pieces[i][0] for i in range(len(pieces))], axis=0))
    cdf = np.concatenate([[0.0], np.cumsum(probs)])
    return probs, cdf, f"source_fallbacks_{fallback_count}_of_{len(pieces)}"


def probs_d2_quantile_average(
    row: pd.Series,
    available_sources: pd.DataFrame,
    residual_by_source: dict[str, np.ndarray],
    origin_residuals: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    quantile_curves = []
    weights = []
    fallback_count = 0
    for _, source_row in available_sources.iterrows():
        source = str(source_row["estimate_source"])
        residuals = residual_by_source.get(source)
        if residuals is None:
            residuals = origin_residuals
            fallback_count += 1
        point = float(source_row["source_bias_adjusted_estimate_mid_usd"])
        quantile_curves.append(point * np.exp(np.quantile(residuals, QUANTILE_GRID)))
        weight = source_row.get("source_bias_adjusted_reliability_raw_weight", np.nan)
        weights.append(float(weight) if np.isfinite(weight) and weight > 0 else 1.0)
    if not quantile_curves:
        probs, cdf = probs_from_residuals(float(row["oof_point_forecast_usd"]), origin_residuals, edges)
        return probs, cdf, "D0_no_sources"
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    qa_quantiles = np.sum([w[i] * quantile_curves[i] for i in range(len(quantile_curves))], axis=0)
    qa_quantiles = np.maximum.accumulate(qa_quantiles)
    cdf = np.interp(edges, qa_quantiles, QUANTILE_GRID, left=0.0, right=1.0)
    cdf[0] = 0.0
    cdf[-1] = 1.0
    cdf = np.maximum.accumulate(cdf)
    probs = smooth_probs(np.diff(cdf))
    cdf = np.concatenate([[0.0], np.cumsum(probs)])
    return probs, cdf, f"source_fallbacks_{fallback_count}_of_{len(quantile_curves)}"


def probs_d4_feature_conditioned(
    train: pd.DataFrame,
    row: pd.Series,
    point: float,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    train = train.copy()
    group = train.loc[
        train["range_bucket"].eq(row.get("range_bucket", "unknown"))
        & train["dispersion_bucket"].eq(row.get("dispersion_bucket", "unknown"))
    ]
    if len(group) >= MIN_GROUP_RESIDUALS and row.get("range_bucket") != "unknown":
        return probs_from_residuals(point, group["signed_log_error"].to_numpy(dtype=float), edges) + ("range_dispersion",)
    if len(group) >= MIN_GROUP_RESIDUALS and row.get("dispersion_bucket") != "unknown":
        return probs_from_residuals(point, group["signed_log_error"].to_numpy(dtype=float), edges) + ("dispersion",)
    probs, cdf = probs_d1_size(train, point, edges)
    return probs, cdf, "D1_fallback"


def realized_bucket(actual: float, edges: np.ndarray) -> int:
    return int(np.clip(np.searchsorted(edges, actual, side="right") - 1, 0, len(edges) - 2))


def score_probs(probs: np.ndarray, cdf_edges: np.ndarray, actual: float, edges: np.ndarray) -> dict[str, float | int]:
    k = realized_bucket(actual, edges)
    p = float(np.clip(probs[k], 1e-12, 1.0))
    thresholds = edges[1:-1]
    pred_cdf = np.cumsum(probs)[:-1]
    obs_cdf = (actual < thresholds).astype(float)
    finite = np.isfinite(thresholds)
    rps = float(np.mean((pred_cdf[finite] - obs_cdf[finite]) ** 2)) if finite.any() else np.nan
    pit = np.nan
    if actual > 0:
        pit = float(np.interp(actual, edges, cdf_edges, left=0.0, right=1.0))
    return {"bucket_index": k, "realized_bucket_probability": p, "log_loss": -np.log(p), "rps": rps, "pit": pit}


def distribution_validation(
    oof: pd.DataFrame,
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grids = canonical_grids()
    candidates = ["D0_origin_empirical", "D1_point_size", "D2_source_quantile_average", "D3_source_linear_pool", "D4_range_dispersion_shadow"]
    complexity = {
        "D0_origin_empirical": 0,
        "D1_point_size": 1,
        "D2_source_quantile_average": 2,
        "D3_source_linear_pool": 2,
        "D4_range_dispersion_shadow": 3,
    }
    score_rows = []
    prob_rows = []
    brier_rows = []
    years = sorted(int(year) for year in oof["holdout_year"].dropna().unique())
    origins = sorted(int(value) for value in oof["origin_day"].dropna().unique())
    for year in years:
        for origin_day in origins:
            train = oof.loc[(oof["holdout_year"].lt(year)) & (oof["origin_day"].eq(origin_day))].copy()
            holdout = oof.loc[(oof["holdout_year"].eq(year)) & (oof["origin_day"].eq(origin_day))].copy()
            if holdout.empty or len(train) < MIN_ORIGIN_RESIDUALS:
                continue
            pooled_train = oof.loc[oof["holdout_year"].lt(year)].copy()
            origin_residuals = train["signed_log_error"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            if len(origin_residuals) < MIN_ORIGIN_RESIDUALS:
                origin_residuals = pooled_train["signed_log_error"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            source_train = source.loc[source["release_year"].lt(year)].copy()
            source_holdout = source.loc[source["release_year"].eq(year)].copy()
            source_residuals = source_train_map(source_train, origin_day)
            for grid in grids:
                fold_scores = {candidate: [] for candidate in candidates}
                for _, row in holdout.iterrows():
                    point = float(row["oof_point_forecast_usd"])
                    actual = float(row["actual_opening_weekend_gross_usd"])
                    available_sources = row_sources(source_holdout, row)
                    candidate_probs: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
                    p0, c0 = probs_from_residuals(point, origin_residuals, grid.edges)
                    candidate_probs["D0_origin_empirical"] = (p0, c0, "origin")
                    p1, c1 = probs_d1_size(train, point, grid.edges)
                    candidate_probs["D1_point_size"] = (p1, c1, "point_size")
                    candidate_probs["D2_source_quantile_average"] = probs_d2_quantile_average(
                        row, available_sources, source_residuals, origin_residuals, grid.edges
                    )
                    candidate_probs["D3_source_linear_pool"] = probs_d3_linear_pool(
                        row, available_sources, source_residuals, origin_residuals, grid.edges
                    )
                    candidate_probs["D4_range_dispersion_shadow"] = probs_d4_feature_conditioned(
                        train, row, point, grid.edges
                    )
                    for candidate, (probs, cdf_edges, fallback) in candidate_probs.items():
                        scored = score_probs(probs, cdf_edges, actual, grid.edges)
                        fold_scores[candidate].append(scored)
                        score_rows.append(
                            {
                                "holdout_year": year,
                                "origin_day": origin_day,
                                "grid": grid.name,
                                "candidate": candidate,
                                "movie_id": row["movie_id"],
                                "title": row["title"],
                                "actual_opening_weekend_gross_usd": actual,
                                "point_forecast_usd": point,
                                "bucket_index": scored["bucket_index"],
                                "realized_bucket_probability": scored["realized_bucket_probability"],
                                "log_loss": scored["log_loss"],
                                "rps": scored["rps"],
                                "pit": scored["pit"],
                                "fallback_level": fallback,
                            }
                        )
                        for bucket_idx, prob in enumerate(probs):
                            prob_rows.append(
                                {
                                    "holdout_year": year,
                                    "origin_day": origin_day,
                                    "grid": grid.name,
                                    "candidate": candidate,
                                    "movie_id": row["movie_id"],
                                    "bucket_index": bucket_idx,
                                    "bucket_low_usd": grid.edges[bucket_idx],
                                    "bucket_high_usd": grid.edges[bucket_idx + 1],
                                    "probability": float(prob),
                                    "realized": int(bucket_idx == scored["bucket_index"]),
                                }
                            )
                        for boundary, cdf_value in zip(grid.edges[1:-1], cdf_edges[1:-1]):
                            if np.isfinite(boundary):
                                brier_rows.append(
                                    {
                                        "holdout_year": year,
                                        "origin_day": origin_day,
                                        "grid": grid.name,
                                        "candidate": candidate,
                                        "boundary_usd": boundary,
                                        "brier": float((cdf_value - float(actual <= boundary)) ** 2),
                                    }
                                )
    per_movie = pd.DataFrame(score_rows)
    probabilities = pd.DataFrame(prob_rows)
    threshold_brier = pd.DataFrame(brier_rows)
    fold_scores = (
        per_movie.groupby(["holdout_year", "origin_day", "grid", "candidate"])
        .agg(
            rows=("movie_id", "count"),
            log_loss=("log_loss", "mean"),
            rps=("rps", "mean"),
            median_realized_bucket_probability=("realized_bucket_probability", "median"),
            pit_mean=("pit", "mean"),
        )
        .reset_index()
    )
    selection_rows = []
    for keys, group in fold_scores.groupby(["holdout_year", "origin_day", "grid"]):
        best = float(group["log_loss"].min())
        eligible = group.loc[group["log_loss"].le(best + SELECTION_TOLERANCE_LOGLOSS)].copy()
        eligible["complexity"] = eligible["candidate"].map(complexity)
        selected = eligible.sort_values(["complexity", "log_loss", "rps", "candidate"]).iloc[0]
        selection_rows.append(
            {
                "holdout_year": keys[0],
                "origin_day": keys[1],
                "grid": keys[2],
                "selected_candidate": selected["candidate"],
                "best_log_loss": best,
                "selected_log_loss": float(selected["log_loss"]),
                "selected_minus_best_log_loss": float(selected["log_loss"] - best),
                "selected_rps": float(selected["rps"]),
            }
        )
    selected = pd.DataFrame(selection_rows)
    return per_movie, probabilities, threshold_brier, fold_scores, selected, pd.DataFrame()


def summarize_outputs(
    per_movie: pd.DataFrame,
    probabilities: pd.DataFrame,
    threshold_brier: pd.DataFrame,
    fold_scores: pd.DataFrame,
    selected: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    scores_by_origin = (
        fold_scores.groupby(["origin_day", "grid", "candidate"])
        .agg(
            folds=("holdout_year", "nunique"),
            rows=("rows", "sum"),
            median_log_loss=("log_loss", "median"),
            median_rps=("rps", "median"),
            mean_log_loss=("log_loss", "mean"),
            mean_rps=("rps", "mean"),
            median_realized_bucket_probability=("median_realized_bucket_probability", "median"),
            pit_mean=("pit_mean", "mean"),
        )
        .reset_index()
    )
    selection_frequency = (
        selected.groupby(["origin_day", "grid", "selected_candidate"])
        .size()
        .rename("selected_folds")
        .reset_index()
        .sort_values(["origin_day", "grid", "selected_folds"], ascending=[True, True, False])
    )
    bucket_scores = (
        per_movie.groupby(["origin_day", "grid", "candidate"])
        .agg(rows=("movie_id", "count"), log_loss=("log_loss", "mean"), rps=("rps", "mean"))
        .reset_index()
    )
    brier = (
        threshold_brier.groupby(["origin_day", "grid", "candidate", "boundary_usd"])
        .agg(rows=("brier", "count"), mean_brier=("brier", "mean"))
        .reset_index()
    )
    calibration = probabilities.copy()
    calibration["probability_band"] = pd.cut(
        calibration["probability"],
        bins=[0, 0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0],
        include_lowest=True,
    ).astype(str)
    calibration = (
        calibration.groupby(["grid", "candidate", "probability_band"])
        .agg(rows=("probability", "count"), mean_probability=("probability", "mean"), realized_rate=("realized", "mean"))
        .reset_index()
    )
    pit = per_movie.groupby(["origin_day", "grid", "candidate"]).agg(
        rows=("pit", "count"),
        pit_mean=("pit", "mean"),
        pit_std=("pit", "std"),
        pit_q10=("pit", lambda s: s.quantile(0.10)),
        pit_q50=("pit", "median"),
        pit_q90=("pit", lambda s: s.quantile(0.90)),
    ).reset_index()
    source_ablation = scores_by_origin.loc[
        scores_by_origin["candidate"].isin(["D0_origin_empirical", "D2_source_quantile_average", "D3_source_linear_pool"])
    ].copy()
    return {
        "nested_distribution_scores_by_origin": scores_by_origin,
        "nested_distribution_selection_frequency": selection_frequency,
        "bucket_logloss_rps_by_origin": bucket_scores,
        "threshold_brier_by_boundary": brier,
        "bucket_calibration_by_probability_band": calibration,
        "pit_diagnostics_by_origin": pit,
        "source_distribution_pool_ablation": source_ablation,
    }


def frozen_distribution_policy(selected: pd.DataFrame, scores_by_origin: pd.DataFrame) -> dict[str, object]:
    policy_rows = []
    primary_grid = "market_2m_20_50"
    selection = selected.loc[selected["grid"].eq(primary_grid)].copy()
    for origin_day, group in selection.groupby("origin_day"):
        counts = group["selected_candidate"].value_counts()
        selected_candidate = str(counts.index[0]) if not counts.empty else "D0_origin_empirical"
        d0 = scores_by_origin.loc[
            scores_by_origin["origin_day"].eq(origin_day)
            & scores_by_origin["grid"].eq(primary_grid)
            & scores_by_origin["candidate"].eq("D0_origin_empirical")
        ]
        chosen = scores_by_origin.loc[
            scores_by_origin["origin_day"].eq(origin_day)
            & scores_by_origin["grid"].eq(primary_grid)
            & scores_by_origin["candidate"].eq(selected_candidate)
        ]
        policy_rows.append(
            {
                "origin_day": int(origin_day),
                "distribution_candidate": selected_candidate,
                "selected_folds": int(counts.iloc[0]) if not counts.empty else 0,
                "d0_median_log_loss": float(d0["median_log_loss"].iloc[0]) if not d0.empty else np.nan,
                "selected_median_log_loss": float(chosen["median_log_loss"].iloc[0]) if not chosen.empty else np.nan,
            }
        )
    return {
        "policy_name": "pre_release_distribution_policy_candidate",
        "status": "diagnostic_not_trading_backtest_ready",
        "point_policy": str(POLICY_PATH),
        "default_distribution": "D0_origin_empirical",
        "promotion_rule": {
            "primary_metric": "nested_bucket_log_loss",
            "co_primary_metric": "ranked_probability_score",
            "selection_tolerance_log_loss": SELECTION_TOLERANCE_LOGLOSS,
            "requires_complete_bucket_probability_vector": True,
        },
        "bucket_grids": [grid.name for grid in canonical_grids()],
        "daily_distribution_policy": policy_rows,
    }


def main() -> int:
    oof = prepare_oof()
    source = prepare_source_panel()
    per_movie, probabilities, threshold_brier, fold_scores, selected, _ = distribution_validation(oof, source)
    summaries = summarize_outputs(per_movie, probabilities, threshold_brier, fold_scores, selected)
    probability_path = write_frame(probabilities, OUTPUT_DIR / "oof_daily_bucket_probabilities.parquet")
    draw_rows = []
    for _, row in oof.iterrows():
        residuals = oof.loc[
            (oof["holdout_year"].lt(row["holdout_year"])) & (oof["origin_day"].eq(row["origin_day"])),
            "signed_log_error",
        ].dropna()
        if len(residuals) < MIN_ORIGIN_RESIDUALS:
            continue
        qs = np.linspace(0.01, 0.99, 99)
        draws = float(row["oof_point_forecast_usd"]) * np.exp(np.quantile(residuals, qs))
        for q, value in zip(qs, draws):
            draw_rows.append(
                {
                    "movie_id": row["movie_id"],
                    "holdout_year": row["holdout_year"],
                    "origin_day": row["origin_day"],
                    "candidate": "D0_origin_empirical",
                    "quantile": q,
                    "opening_weekend_gross_usd": float(value),
                }
            )
    draws_path = write_frame(pd.DataFrame(draw_rows), OUTPUT_DIR / "oof_daily_distribution_draws.parquet")
    for stem, frame in summaries.items():
        frame.to_csv(OUTPUT_DIR / f"{stem}.csv", index=False)
    fold_scores.to_csv(OUTPUT_DIR / "nested_distribution_year_folds.csv", index=False)
    selected.to_csv(OUTPUT_DIR / "nested_distribution_selected_year_folds.csv", index=False)
    policy = frozen_distribution_policy(selected, summaries["nested_distribution_scores_by_origin"])
    (OUTPUT_DIR / "frozen_pre_release_distribution_policy.json").write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote distribution diagnostics to {OUTPUT_DIR}")
    print(f"Per-movie scored rows: {len(per_movie):,}")
    print(f"Bucket probability rows: {len(probabilities):,} -> {probability_path.name}")
    print(f"Distribution draw rows: {len(draw_rows):,} -> {draws_path.name}")
    print(f"Fold score rows: {len(fold_scores):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
