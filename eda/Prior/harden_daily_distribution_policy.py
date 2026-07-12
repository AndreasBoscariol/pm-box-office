#!/usr/bin/env python3
"""Hardening diagnostics for the pre-release distribution policy.

This script keeps the frozen point forecast fixed and audits whether source
distribution candidates improve bucket probabilities through shape, through an
implicit point-center shift, or through fragile tail behavior.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eda.Prior.daily_distribution_validation import (
    BUCKET_FLOOR,
    DIAGNOSTICS_DIR,
    MIN_ORIGIN_RESIDUALS,
    OUTPUT_DIR,
    POLICY_PATH,
    QUANTILE_GRID,
    canonical_grids,
    empirical_cdf,
    fit_size_scale,
    point_scale,
    prepare_oof,
    prepare_source_panel,
    probs_d1_size,
    probs_from_residuals,
    realized_bucket,
    residual_thresholds,
    row_sources,
    score_probs,
    smooth_probs,
)


SOURCE_SHRINK_K = 40
WEIGHT_SHRINK_RHO = 0.75
FLOORS = (1e-6, 1e-5, 1e-4)


def hash_probs(probs: np.ndarray) -> str:
    rounded = np.round(np.asarray(probs, dtype=float), 12)
    return hashlib.sha1(rounded.tobytes()).hexdigest()


def normalize_weights(weights: list[float]) -> np.ndarray:
    arr = np.asarray(weights, dtype=float)
    arr = np.where(np.isfinite(arr) & (arr > 0), arr, 1.0)
    if arr.sum() <= 0:
        return np.full(len(arr), 1.0 / len(arr))
    return arr / arr.sum()


def source_role_weight(source: str, origin_day: int) -> float:
    if source in {"boxofficereport", "boxofficepro"}:
        return 1.0
    if source == "toddmthatcher" and -9 <= origin_day <= -3:
        return 0.6
    if source == "boxofficetheory" and origin_day <= -8:
        return 0.4
    return 0.25


def source_residual_pool(
    source_train: pd.DataFrame,
    source: str,
    origin_day: int,
    origin_residuals: np.ndarray,
    global_residuals: np.ndarray,
) -> tuple[np.ndarray, str, int]:
    exact = source_train.loc[
        source_train["estimate_source"].eq(source) & source_train["origin_day"].eq(origin_day),
        "source_bias_adjusted_residual_log",
    ].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    adjacent = source_train.loc[
        source_train["estimate_source"].eq(source) & source_train["origin_day"].between(origin_day - 1, origin_day + 1),
        "source_bias_adjusted_residual_log",
    ].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    pooled = source_train.loc[
        source_train["estimate_source"].eq(source),
        "source_bias_adjusted_residual_log",
    ].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)

    if len(exact) >= 20:
        source_part = exact
        level = "source_exact_origin"
    elif len(adjacent) >= 30:
        source_part = adjacent
        level = "source_adjacent_origin"
    elif len(pooled) >= 40:
        source_part = pooled
        level = "source_pooled"
    elif len(origin_residuals) >= MIN_ORIGIN_RESIDUALS:
        return origin_residuals, "origin_fallback", 0
    else:
        return global_residuals, "global_fallback", 0

    omega = len(source_part) / (len(source_part) + SOURCE_SHRINK_K)
    keep_source = max(1, int(round(omega * 200)))
    keep_origin = max(1, 200 - keep_source)
    source_quantiles = np.quantile(source_part, np.linspace(0.005, 0.995, keep_source))
    fallback = origin_residuals if len(origin_residuals) >= MIN_ORIGIN_RESIDUALS else global_residuals
    fallback_quantiles = np.quantile(fallback, np.linspace(0.005, 0.995, keep_origin))
    return np.concatenate([source_quantiles, fallback_quantiles]), level, len(source_part)


def weighted_source_rows(
    available_sources: pd.DataFrame,
    source_train: pd.DataFrame,
    origin_day: int,
    origin_residuals: np.ndarray,
    global_residuals: np.ndarray,
    weight_policy: str,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    rows = []
    raw_weights = []
    fallback_counts: dict[str, int] = {}
    for _, source_row in available_sources.iterrows():
        source = str(source_row["estimate_source"])
        residuals, fallback, prior_n = source_residual_pool(
            source_train, source, origin_day, origin_residuals, global_residuals
        )
        fallback_counts[fallback] = fallback_counts.get(fallback, 0) + 1
        if weight_policy == "equal":
            raw_weight = 1.0
        elif weight_policy == "source_role":
            raw_weight = source_role_weight(source, origin_day)
        else:
            perf = source_row.get("source_bias_adjusted_reliability_raw_weight", np.nan)
            role = source_role_weight(source, origin_day)
            raw_weight = (1.0 - WEIGHT_SHRINK_RHO) * (perf if np.isfinite(perf) and perf > 0 else 1.0)
            raw_weight += WEIGHT_SHRINK_RHO * role
        rows.append(
            {
                "source": source,
                "point": float(source_row["source_bias_adjusted_estimate_mid_usd"]),
                "residuals": residuals,
                "fallback": fallback,
                "prior_n": prior_n,
            }
        )
        raw_weights.append(float(raw_weight))
    if not rows:
        return [], {}
    weights = normalize_weights(raw_weights)
    for row, weight in zip(rows, weights):
        row["weight"] = float(weight)
    return rows, fallback_counts


def probs_from_quantile_curve(quantiles: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quantiles = np.maximum.accumulate(np.asarray(quantiles, dtype=float))
    cdf = np.interp(edges, quantiles, QUANTILE_GRID, left=0.0, right=1.0)
    cdf[0] = 0.0
    cdf[-1] = 1.0
    cdf = np.maximum.accumulate(cdf)
    probs = smooth_probs(np.diff(cdf))
    return probs, np.concatenate([[0.0], np.cumsum(probs)])


def d2_quantile_average(
    point: float,
    source_rows: list[dict[str, object]],
    origin_residuals: np.ndarray,
    edges: np.ndarray,
    centered: bool,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    if not source_rows:
        probs, cdf = probs_from_residuals(point, origin_residuals, edges)
        return probs, cdf, point * np.exp(np.quantile(origin_residuals, 0.5)), "D0_no_sources"
    curves = []
    weights = []
    for row in source_rows:
        residuals = np.asarray(row["residuals"], dtype=float)
        curves.append(float(row["point"]) * np.exp(np.quantile(residuals, QUANTILE_GRID)))
        weights.append(float(row["weight"]))
    w = normalize_weights(weights)
    curve = np.sum([w[i] * curves[i] for i in range(len(curves))], axis=0)
    median = float(np.interp(0.5, QUANTILE_GRID, curve))
    if centered and np.isfinite(median) and median > 0 and np.isfinite(point) and point > 0:
        curve = point * curve / median
    probs, cdf = probs_from_quantile_curve(curve, edges)
    return probs, cdf, median, "source_quantile_average"


def linear_pool_cdf(source_rows: list[dict[str, object]], y: np.ndarray) -> np.ndarray:
    if not source_rows:
        return np.full(len(y), np.nan)
    out = np.zeros(len(y), dtype=float)
    for row in source_rows:
        point = float(row["point"])
        residuals = np.asarray(row["residuals"], dtype=float)
        thresholds = np.full(len(y), np.inf, dtype=float)
        finite = np.isfinite(y) & (y > 0) & np.isfinite(point) & (point > 0)
        thresholds[finite] = np.log(y[finite] / point)
        thresholds[y <= 0] = -np.inf
        out += float(row["weight"]) * empirical_cdf(residuals, thresholds)
    return out


def d3_linear_pool(
    point: float,
    source_rows: list[dict[str, object]],
    origin_residuals: np.ndarray,
    edges: np.ndarray,
    centered: bool,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    if not source_rows:
        probs, cdf = probs_from_residuals(point, origin_residuals, edges)
        return probs, cdf, point * np.exp(np.quantile(origin_residuals, 0.5)), "D0_no_sources"
    support = []
    for row in source_rows:
        support.append(float(row["point"]) * np.exp(np.quantile(np.asarray(row["residuals"], dtype=float), QUANTILE_GRID)))
    support_grid = np.unique(np.maximum.accumulate(np.concatenate(support)))
    cdf_support = linear_pool_cdf(source_rows, support_grid)
    median = float(support_grid[np.searchsorted(cdf_support, 0.5, side="left")])
    eval_edges = edges.copy()
    if centered and np.isfinite(median) and median > 0 and np.isfinite(point) and point > 0:
        eval_edges = edges * median / point
    cdf = linear_pool_cdf(source_rows, eval_edges)
    cdf[0] = 0.0
    cdf[-1] = 1.0
    cdf = np.maximum.accumulate(np.nan_to_num(cdf, nan=0.0))
    probs = smooth_probs(np.diff(cdf))
    return probs, np.concatenate([[0.0], np.cumsum(probs)]), median, "source_linear_pool"


def score_with_floor(probs: np.ndarray, actual: float, edges: np.ndarray, floor: float) -> tuple[float, float]:
    adjusted = np.asarray(probs, dtype=float)
    adjusted = np.maximum(adjusted, floor)
    adjusted = adjusted / adjusted.sum()
    k = realized_bucket(actual, edges)
    log_loss = -np.log(float(np.clip(adjusted[k], 1e-12, 1.0)))
    thresholds = edges[1:-1]
    finite = np.isfinite(thresholds)
    pred_cdf = np.cumsum(adjusted)[:-1]
    obs_cdf = (actual < thresholds).astype(float)
    rps = float(np.mean((pred_cdf[finite] - obs_cdf[finite]) ** 2)) if finite.any() else np.nan
    return float(log_loss), rps


def run_hardening(oof: pd.DataFrame, source: pd.DataFrame) -> dict[str, pd.DataFrame]:
    grids = canonical_grids()
    score_rows = []
    center_rows = []
    precision_rows = []
    vector_rows = []
    weight_rows = []
    candidates = [
        ("D0_origin_empirical", None, False),
        ("D1_point_size", None, False),
        ("D2_quantile_equal_uncentered", "equal", False),
        ("D2_quantile_equal_centered", "equal", True),
        ("D2_quantile_role_uncentered", "source_role", False),
        ("D2_quantile_role_centered", "source_role", True),
        ("D2_quantile_perf_shrunk_uncentered", "performance_shrunk_to_role", False),
        ("D2_quantile_perf_shrunk_centered", "performance_shrunk_to_role", True),
        ("D3_linear_equal_uncentered", "equal", False),
        ("D3_linear_equal_centered", "equal", True),
        ("D3_linear_role_uncentered", "source_role", False),
        ("D3_linear_role_centered", "source_role", True),
        ("D3_linear_perf_shrunk_uncentered", "performance_shrunk_to_role", False),
        ("D3_linear_perf_shrunk_centered", "performance_shrunk_to_role", True),
    ]
    years = sorted(int(year) for year in oof["holdout_year"].dropna().unique())
    origins = sorted(int(value) for value in oof["origin_day"].dropna().unique())
    global_residuals_by_year = {
        year: oof.loc[oof["holdout_year"].lt(year), "signed_log_error"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        for year in years
    }
    for year in years:
        source_train = source.loc[source["release_year"].lt(year)].copy()
        source_holdout = source.loc[source["release_year"].eq(year)].copy()
        for origin_day in origins:
            train = oof.loc[(oof["holdout_year"].lt(year)) & (oof["origin_day"].eq(origin_day))].copy()
            holdout = oof.loc[(oof["holdout_year"].eq(year)) & (oof["origin_day"].eq(origin_day))].copy()
            if holdout.empty or len(train) < MIN_ORIGIN_RESIDUALS:
                continue
            origin_residuals = train["signed_log_error"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            global_residuals = global_residuals_by_year[year]
            size_fit = fit_size_scale(train)
            for _, row in holdout.iterrows():
                point = float(row["oof_point_forecast_usd"])
                actual = float(row["actual_opening_weekend_gross_usd"])
                available = row_sources(source_holdout, row)
                source_cache = {}
                for weight_policy in {"equal", "source_role", "performance_shrunk_to_role"}:
                    source_rows, fallback_counts = weighted_source_rows(
                        available, source_train, origin_day, origin_residuals, global_residuals, weight_policy
                    )
                    source_cache[weight_policy] = source_rows
                    if source_rows:
                        weight_rows.append(
                            {
                                "holdout_year": year,
                                "origin_day": origin_day,
                                "movie_id": row["movie_id"],
                                "weight_policy": weight_policy,
                                "sources": ",".join(str(item["source"]) for item in source_rows),
                                "weights": ",".join(f"{float(item['weight']):.6f}" for item in source_rows),
                                "fallback_counts": json.dumps(fallback_counts, sort_keys=True),
                            }
                        )
                for grid in grids:
                    probs_by_candidate = {}
                    for candidate, weight_policy, centered in candidates:
                        if candidate == "D0_origin_empirical":
                            probs, cdf = probs_from_residuals(point, origin_residuals, grid.edges)
                            median = point * np.exp(np.quantile(origin_residuals, 0.5))
                            fallback = "origin"
                        elif candidate == "D1_point_size":
                            probs, cdf = probs_d1_size(train, point, grid.edges)
                            median = point * np.exp(np.quantile(origin_residuals, 0.5) * point_scale(point, size_fit))
                            fallback = "point_size"
                        elif candidate.startswith("D2"):
                            probs, cdf, median, fallback = d2_quantile_average(
                                point, source_cache[weight_policy or "equal"], origin_residuals, grid.edges, centered
                            )
                        else:
                            probs, cdf, median, fallback = d3_linear_pool(
                                point, source_cache[weight_policy or "equal"], origin_residuals, grid.edges, centered
                            )
                        probs_by_candidate[candidate] = probs
                        scored = score_probs(probs, cdf, actual, grid.edges)
                        score_rows.append(
                            {
                                "holdout_year": year,
                                "origin_day": origin_day,
                                "grid": grid.name,
                                "candidate": candidate,
                                "movie_id": row["movie_id"],
                                "title": row["title"],
                                "log_loss": scored["log_loss"],
                                "rps": scored["rps"],
                                "realized_bucket_probability": scored["realized_bucket_probability"],
                                "pit": scored["pit"],
                                "fallback_level": fallback,
                            }
                        )
                        is_centered_source_candidate = bool(centered and (candidate.startswith("D2") or candidate.startswith("D3")))
                        effective_median = point if is_centered_source_candidate else median
                        center_rows.append(
                            {
                                "holdout_year": year,
                                "origin_day": origin_day,
                                "grid": grid.name,
                                "candidate": candidate,
                                "movie_id": row["movie_id"],
                                "point_forecast_usd": point,
                                "raw_distribution_median_usd": median,
                                "effective_distribution_median_usd": effective_median,
                                "raw_center_shift_log": float(np.log(median / point)) if median > 0 and point > 0 else np.nan,
                                "effective_center_shift_log": (
                                    float(np.log(effective_median / point))
                                    if effective_median > 0 and point > 0
                                    else np.nan
                                ),
                            }
                        )
                        for floor in FLOORS:
                            ll, rps = score_with_floor(probs, actual, grid.edges, floor)
                            precision_rows.append(
                                {
                                    "holdout_year": year,
                                    "origin_day": origin_day,
                                    "grid": grid.name,
                                    "candidate": candidate,
                                    "movie_id": row["movie_id"],
                                    "probability_floor": floor,
                                    "log_loss": ll,
                                    "rps": rps,
                                }
                            )
                    if -14 <= origin_day <= -10 and grid.name == "market_2m_20_50":
                        for candidate, probs in probs_by_candidate.items():
                            vector_rows.append(
                                {
                                    "holdout_year": year,
                                    "origin_day": origin_day,
                                    "movie_id": row["movie_id"],
                                    "candidate": candidate,
                                    "probability_hash": hash_probs(probs),
                                    "probability_argmax": int(np.argmax(probs)),
                                    "probability_max": float(np.max(probs)),
                                }
                            )
    scores = pd.DataFrame(score_rows)
    centers = pd.DataFrame(center_rows)
    precision = pd.DataFrame(precision_rows)
    vectors = pd.DataFrame(vector_rows)
    weights = pd.DataFrame(weight_rows)

    mean_vs_median = scores.groupby(["origin_day", "grid", "candidate"]).agg(
        rows=("movie_id", "count"),
        mean_log_loss=("log_loss", "mean"),
        median_log_loss=("log_loss", "median"),
        q90_log_loss=("log_loss", lambda s: s.quantile(0.90)),
        q95_log_loss=("log_loss", lambda s: s.quantile(0.95)),
        max_log_loss=("log_loss", "max"),
        mean_rps=("rps", "mean"),
        median_rps=("rps", "median"),
    ).reset_index()
    catastrophic = scores.assign(
        below_1pct=scores["realized_bucket_probability"].lt(0.01),
        below_0_5pct=scores["realized_bucket_probability"].lt(0.005),
        below_0_1pct=scores["realized_bucket_probability"].lt(0.001),
    ).groupby(["origin_day", "grid", "candidate"]).agg(
        rows=("movie_id", "count"),
        mean_realized_bucket_probability=("realized_bucket_probability", "mean"),
        pct_below_1pct=("below_1pct", "mean"),
        pct_below_0_5pct=("below_0_5pct", "mean"),
        pct_below_0_1pct=("below_0_1pct", "mean"),
        max_log_loss=("log_loss", "max"),
    ).reset_index()
    center_audit = centers.groupby(["origin_day", "grid", "candidate"]).agg(
        rows=("movie_id", "count"),
        mean_raw_center_shift_log=("raw_center_shift_log", "mean"),
        median_raw_center_shift_log=("raw_center_shift_log", "median"),
        mean_abs_raw_center_shift_log=("raw_center_shift_log", lambda s: s.abs().mean()),
        q90_abs_raw_center_shift_log=("raw_center_shift_log", lambda s: s.abs().quantile(0.90)),
        mean_effective_center_shift_log=("effective_center_shift_log", "mean"),
        median_effective_center_shift_log=("effective_center_shift_log", "median"),
        mean_abs_effective_center_shift_log=("effective_center_shift_log", lambda s: s.abs().mean()),
        q90_abs_effective_center_shift_log=("effective_center_shift_log", lambda s: s.abs().quantile(0.90)),
    ).reset_index()
    centered_vs_uncentered = mean_vs_median.loc[
        mean_vs_median["candidate"].str.contains("centered|uncentered", regex=True)
    ].copy()
    floor_sensitivity = precision.groupby(["origin_day", "grid", "candidate", "probability_floor"]).agg(
        rows=("movie_id", "count"),
        mean_log_loss=("log_loss", "mean"),
        mean_rps=("rps", "mean"),
    ).reset_index()
    calibration = scores.copy()
    calibration["probability_band"] = pd.cut(
        calibration["realized_bucket_probability"],
        bins=[0, 0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.40, 0.80, 1.0],
        include_lowest=True,
    ).astype(str)
    calibration = calibration.groupby(["origin_day", "grid", "candidate", "probability_band"]).agg(
        rows=("movie_id", "count"),
        mean_realized_bucket_probability=("realized_bucket_probability", "mean"),
        mean_log_loss=("log_loss", "mean"),
    ).reset_index()
    if not vectors.empty:
        early = vectors.sort_values(["candidate", "movie_id", "holdout_year", "origin_day"])
        shifted = early.copy()
        shifted["next_origin_day"] = shifted["origin_day"] + 1
        adjacent = early.merge(
            shifted,
            left_on=["candidate", "movie_id", "holdout_year", "origin_day"],
            right_on=["candidate", "movie_id", "holdout_year", "next_origin_day"],
            suffixes=("", "_prev"),
        )
        vector_audit = adjacent.groupby(["candidate", "origin_day"]).agg(
            pairs=("movie_id", "count"),
            identical_hash_rate=("probability_hash", lambda s: np.nan),
        ).reset_index()
        vector_rows_out = []
        for keys, group in adjacent.groupby(["candidate", "origin_day"]):
            vector_rows_out.append(
                {
                    "candidate": keys[0],
                    "origin_day": int(keys[1]),
                    "pairs": int(len(group)),
                    "identical_hash_rate": float((group["probability_hash"] == group["probability_hash_prev"]).mean()),
                    "same_argmax_rate": float((group["probability_argmax"] == group["probability_argmax_prev"]).mean()),
                    "mean_abs_probability_max_delta": float((group["probability_max"] - group["probability_max_prev"]).abs().mean()),
                }
            )
        vector_audit = pd.DataFrame(vector_rows_out)
    else:
        vector_audit = pd.DataFrame()

    return {
        "distribution_mean_vs_median_logloss": mean_vs_median,
        "distribution_catastrophic_miss_audit": catastrophic,
        "distribution_center_shift_audit": center_audit,
        "distribution_centered_vs_uncentered": centered_vs_uncentered,
        "distribution_probability_precision_audit": pd.DataFrame(
            [
                {
                    "bucket_probability_method": "direct_cdf_difference",
                    "draw_file_method": "deterministic_quantile_grid_for_inspection",
                    "monte_carlo_bucket_scoring_used": False,
                    "probability_floor_default": BUCKET_FLOOR,
                    "floor_sensitivity_values": ",".join(str(v) for v in FLOORS),
                }
            ]
        ),
        "distribution_source_weight_shrinkage": weights,
        "distribution_source_ablation": mean_vs_median.loc[
            mean_vs_median["candidate"].str.contains("D2|D3", regex=True)
        ].copy(),
        "distribution_probability_floor_sensitivity": floor_sensitivity,
        "distribution_calibration_by_origin": calibration,
        "distribution_actual_market_grid_scores": pd.DataFrame(
            [
                {
                    "status": "not_available",
                    "reason": "historical market bucket definition table was not found in local diagnostics",
                    "fallback_grids_scored": ",".join(grid.name for grid in canonical_grids()),
                }
            ]
        ),
        "distribution_paired_bootstrap_deltas": paired_bootstrap(scores),
        "distribution_early_origin_probability_vector_audit": vector_audit,
        "_scores": scores,
    }


def paired_bootstrap(scores: pd.DataFrame, iterations: int = 1000) -> pd.DataFrame:
    primary = scores.loc[scores["grid"].eq("market_2m_20_50")].copy()
    rows = []
    rng = np.random.default_rng(20260711)
    for origin_day, origin_frame in primary.groupby("origin_day"):
        base = origin_frame.loc[origin_frame["candidate"].eq("D0_origin_empirical")][
            ["holdout_year", "movie_id", "log_loss", "rps"]
        ].rename(columns={"log_loss": "base_log_loss", "rps": "base_rps"})
        for candidate, candidate_frame in origin_frame.groupby("candidate"):
            if candidate == "D0_origin_empirical":
                continue
            paired = candidate_frame.merge(base, on=["holdout_year", "movie_id"], how="inner")
            if paired.empty:
                continue
            ll_delta = (paired["log_loss"] - paired["base_log_loss"]).to_numpy(dtype=float)
            rps_delta = (paired["rps"] - paired["base_rps"]).to_numpy(dtype=float)
            n = len(paired)
            boot_ll = np.empty(iterations, dtype=float)
            boot_rps = np.empty(iterations, dtype=float)
            for idx in range(iterations):
                sample = rng.integers(0, n, size=n)
                boot_ll[idx] = float(np.mean(ll_delta[sample]))
                boot_rps[idx] = float(np.mean(rps_delta[sample]))
            rows.append(
                {
                    "origin_day": int(origin_day),
                    "candidate": candidate,
                    "grid": "market_2m_20_50",
                    "paired_rows": int(n),
                    "delta_mean_log_loss_vs_D0": float(np.mean(ll_delta)),
                    "delta_mean_log_loss_ci025": float(np.quantile(boot_ll, 0.025)),
                    "delta_mean_log_loss_ci975": float(np.quantile(boot_ll, 0.975)),
                    "delta_mean_rps_vs_D0": float(np.mean(rps_delta)),
                    "delta_mean_rps_ci025": float(np.quantile(boot_rps, 0.025)),
                    "delta_mean_rps_ci975": float(np.quantile(boot_rps, 0.975)),
                    "bootstrap_iterations": iterations,
                }
            )
    return pd.DataFrame(rows)


def freeze_policy_v1(mean_vs_median: pd.DataFrame) -> dict[str, object]:
    primary = mean_vs_median.loc[mean_vs_median["grid"].eq("market_2m_20_50")].copy()
    complexity = {
        "D0_origin_empirical": 0,
        "D1_point_size": 1,
        "D2_quantile_equal_centered": 2,
        "D2_quantile_role_centered": 2,
        "D2_quantile_perf_shrunk_centered": 3,
        "D3_linear_equal_centered": 3,
        "D3_linear_role_centered": 3,
        "D3_linear_perf_shrunk_centered": 4,
    }
    allowed = primary.loc[primary["candidate"].isin(complexity)].copy()
    allowed["complexity"] = allowed["candidate"].map(complexity)
    rows = []
    for origin_day, group in allowed.groupby("origin_day"):
        best = group["mean_log_loss"].min()
        eligible = group.loc[group["mean_log_loss"].le(best + 0.005)].copy()
        selected = eligible.sort_values(["complexity", "mean_log_loss", "mean_rps", "candidate"]).iloc[0]
        rows.append(
            {
                "origin_day": int(origin_day),
                "distribution_candidate": selected["candidate"],
                "mean_log_loss": float(selected["mean_log_loss"]),
                "median_log_loss": float(selected["median_log_loss"]),
                "mean_rps": float(selected["mean_rps"]),
                "selected_minus_best_mean_log_loss": float(selected["mean_log_loss"] - best),
            }
        )
    return {
        "policy_name": "pre_release_distribution_policy_v1_candidate",
        "status": "forecast_distribution_validated_market_backtest_pending",
        "point_policy": str(POLICY_PATH),
        "primary_grid": "market_2m_20_50",
        "primary_metric": "mean_log_loss",
        "co_primary_metric": "mean_rps",
        "point_centering": "source_distribution_candidates_centered_to_frozen_point_for_policy_selection",
        "daily_distribution_policy": rows,
    }


def main() -> int:
    oof = prepare_oof()
    source = prepare_source_panel()
    outputs = run_hardening(oof, source)
    scores = outputs.pop("_scores")
    for stem, frame in outputs.items():
        frame.to_csv(OUTPUT_DIR / f"{stem}.csv", index=False)
    scores.to_csv(OUTPUT_DIR / "distribution_hardening_per_movie_scores.csv", index=False)
    policy = freeze_policy_v1(outputs["distribution_mean_vs_median_logloss"])
    (OUTPUT_DIR / "frozen_pre_release_distribution_policy_v1.json").write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote hardening diagnostics to {OUTPUT_DIR}")
    print(f"Per-movie hardening score rows: {len(scores):,}")
    print(f"Candidates scored: {scores['candidate'].nunique():,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
