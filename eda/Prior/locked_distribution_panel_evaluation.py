#!/usr/bin/env python3
"""Locked rolling-origin distribution panel and multi-grid evaluation.

The point forecast is frozen. This script evaluates only distribution shape
around that point, using leakage-safe rolling-origin histories.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eda.Prior.daily_distribution_validation import (
    BUCKET_FLOOR,
    MIN_ORIGIN_RESIDUALS,
    OUTPUT_DIR,
    POLICY_PATH,
    empirical_cdf,
    fit_size_scale,
    point_scale,
    prepare_oof,
    prepare_source_panel,
    realized_bucket,
    row_sources,
    smooth_probs,
)
from eda.Prior.harden_daily_distribution_policy import (
    normalize_weights,
    source_role_weight,
)


QUANTILE_GRID = np.r_[0.001, np.linspace(0.005, 0.995, 199), 0.999]
SOURCE_SHRINK_KAPPAS = (20, 40, 80)
BOOTSTRAP_ITERATIONS = 1000
MODEL_VERSION = "locked_distribution_panel_v1"


@dataclass(frozen=True)
class Distribution:
    quantiles: np.ndarray
    fallback_level: str
    source_count: int = 0

    def cdf(self, values: np.ndarray | float) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        return np.interp(x, self.quantiles, QUANTILE_GRID, left=0.0, right=1.0)


@dataclass(frozen=True)
class GridSpec:
    name: str
    edges: np.ndarray


def write_frame(frame: pd.DataFrame, path: Path) -> Path:
    try:
        frame.to_parquet(path.with_suffix(".parquet"), index=False)
        return path.with_suffix(".parquet")
    except ImportError:
        frame.to_csv(path.with_suffix(".csv"), index=False)
        return path.with_suffix(".csv")


def quantile_frame_row(meta: dict[str, object], candidate: str, dist: Distribution) -> dict[str, object]:
    row = dict(meta)
    row.update(
        {
            "candidate": candidate,
            "fallback_level": dist.fallback_level,
            "distribution_source_count": dist.source_count,
            "model_version": MODEL_VERSION,
        }
    )
    for q, value in zip(QUANTILE_GRID, dist.quantiles):
        row[f"q_{q:.3f}"] = float(value)
    return row


def d0_distribution(point: float, residuals: np.ndarray) -> Distribution:
    return Distribution(point * np.exp(np.quantile(residuals, QUANTILE_GRID)), "origin", 0)


def d1_distribution(train: pd.DataFrame, point: float, size_fit: tuple[float, float] | None) -> Distribution:
    residual = pd.to_numeric(train["signed_log_error"], errors="coerce").to_numpy(dtype=float)
    train_point = pd.to_numeric(train["oof_point_forecast_usd"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(residual) & np.isfinite(train_point) & (train_point > 0)
    if int(mask.sum()) < MIN_ORIGIN_RESIDUALS:
        return d0_distribution(point, residual[np.isfinite(residual)])
    scales = np.array([point_scale(value, size_fit) for value in train_point[mask]])
    standardized = residual[mask] / scales
    this_scale = point_scale(point, size_fit)
    return Distribution(point * np.exp(np.quantile(standardized, QUANTILE_GRID) * this_scale), "point_size", 0)


def source_residuals(
    source_train: pd.DataFrame,
    source: str,
    origin_day: int,
    origin_residuals: np.ndarray,
    global_residuals: np.ndarray,
    kappa: int | None,
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
        base, level = exact, "source_exact_origin"
    elif len(adjacent) >= 30:
        base, level = adjacent, "source_adjacent_origin"
    elif len(pooled) >= 40:
        base, level = pooled, "source_pooled"
    elif len(origin_residuals) >= MIN_ORIGIN_RESIDUALS:
        return origin_residuals, "origin_fallback", 0
    else:
        return global_residuals, "global_fallback", 0

    if kappa is None:
        return base, level, len(base)

    fallback = origin_residuals if len(origin_residuals) >= MIN_ORIGIN_RESIDUALS else global_residuals
    lam = len(base) / (len(base) + kappa)
    n_source = max(1, int(round(lam * 240)))
    n_pool = max(1, 240 - n_source)
    source_q = np.quantile(base, np.linspace(0.002, 0.998, n_source))
    pool_q = np.quantile(fallback, np.linspace(0.002, 0.998, n_pool))
    return np.concatenate([source_q, pool_q]), f"{level}_shrunk_k{kappa}", len(base)


def source_distribution(
    point: float,
    available_sources: pd.DataFrame,
    source_train: pd.DataFrame,
    origin_day: int,
    origin_residuals: np.ndarray,
    global_residuals: np.ndarray,
    weight_policy: str,
    centered: bool,
    kappa: int | None,
) -> Distribution:
    rows = []
    weights = []
    fallbacks: dict[str, int] = {}
    for _, source_row in available_sources.iterrows():
        source = str(source_row["estimate_source"])
        residuals, fallback, prior_n = source_residuals(
            source_train, source, origin_day, origin_residuals, global_residuals, kappa
        )
        fallbacks[fallback] = fallbacks.get(fallback, 0) + 1
        source_point = float(source_row["source_bias_adjusted_estimate_mid_usd"])
        rows.append(source_point * np.exp(np.quantile(residuals, QUANTILE_GRID)))
        if weight_policy == "source_role":
            weights.append(source_role_weight(source, origin_day))
        else:
            weights.append(1.0)
    if not rows:
        return d0_distribution(point, origin_residuals)
    w = normalize_weights(weights)
    curve = np.sum([w[i] * rows[i] for i in range(len(rows))], axis=0)
    median = float(np.interp(0.5, QUANTILE_GRID, curve))
    if centered and np.isfinite(median) and median > 0 and point > 0:
        curve = point * curve / median
    suffix = "unshrunk" if kappa is None else f"k{kappa}"
    return Distribution(np.maximum.accumulate(curve), f"{weight_policy}_{suffix}_{json.dumps(fallbacks, sort_keys=True)}", len(rows))


def mix_distributions(dists: list[tuple[float, Distribution]], fallback: str) -> Distribution:
    weights = normalize_weights([weight for weight, _ in dists])
    support = np.unique(np.concatenate([dist.quantiles for _, dist in dists]))
    cdf = np.zeros(len(support), dtype=float)
    for weight, (_, dist) in zip(weights, dists):
        cdf += weight * dist.cdf(support)
    cdf = np.maximum.accumulate(cdf)
    quantiles = np.interp(QUANTILE_GRID, cdf, support, left=support[0], right=support[-1])
    return Distribution(np.maximum.accumulate(quantiles), fallback, 0)


def tail_protect(dist: Distribution, d0: Distribution, weight: float = 0.03) -> Distribution:
    return mix_distributions([(1.0 - weight, dist), (weight, d0)], f"tail_protected_{dist.fallback_level}")


def production_candidate_for_origin(policy: dict[str, object], origin_day: int) -> str:
    rows = policy.get("daily_distribution_policy", [])
    for row in rows:
        if int(row["origin_day"]) == origin_day:
            return str(row["distribution_candidate"])
    return "D0_origin_empirical"


def build_distributions(
    train: pd.DataFrame,
    source_train: pd.DataFrame,
    source_holdout: pd.DataFrame,
    row: pd.Series,
    origin_residuals: np.ndarray,
    global_residuals: np.ndarray,
    production_name: str,
    size_fit: tuple[float, float] | None,
) -> dict[str, Distribution]:
    point = float(row["oof_point_forecast_usd"])
    available = row_sources(source_holdout, row)
    dists: dict[str, Distribution] = {
        "D0_origin_empirical": d0_distribution(point, origin_residuals),
        "D1_point_size": d1_distribution(train, point, size_fit),
    }
    dists["D2_quantile_equal_centered"] = source_distribution(
        point, available, source_train, int(row["origin_day"]), origin_residuals, global_residuals, "equal", True, None
    )
    dists["D2_quantile_role_centered"] = source_distribution(
        point, available, source_train, int(row["origin_day"]), origin_residuals, global_residuals, "source_role", True, None
    )
    for kappa in SOURCE_SHRINK_KAPPAS:
        dists[f"D2_source_shrunk_k{kappa}_equal_centered"] = source_distribution(
            point, available, source_train, int(row["origin_day"]), origin_residuals, global_residuals, "equal", True, kappa
        )
    prod_base = dists.get(production_name, dists["D0_origin_empirical"])
    dists["production_policy_v1"] = prod_base
    dists["pool_shrunk_to_production"] = mix_distributions(
        [
            (0.80, prod_base),
            (0.07, dists["D0_origin_empirical"]),
            (0.07, dists["D1_point_size"]),
            (0.06, dists["D2_quantile_equal_centered"]),
        ],
        f"prod80_d0d1d2_{production_name}",
    )
    dists["production_tail_protected_3pct_D0"] = tail_protect(prod_base, dists["D0_origin_empirical"], 0.03)
    return dists


def absolute_grid(width: int, shift: int) -> GridSpec:
    upper = 250_000_000
    edges = np.r_[0, np.arange(width + shift, upper + width, width), np.inf].astype(float)
    edges = np.unique(edges)
    return GridSpec(f"absolute_{width // 1_000_000}m_shift_{shift // 1_000_000}m", edges)


def relative_grid(point: float, pct: float, shifted: bool) -> GridSpec:
    width = max(500_000.0, point * pct)
    start = width / 2 if shifted else 0.0
    upper = max(250_000_000.0, point * 4)
    edges = np.r_[0.0, np.arange(start + width, upper + width, width), np.inf]
    edges = np.unique(edges)
    return GridSpec(f"relative_{int(pct * 100)}pct_shift_{int(shifted)}", edges)


def grids_for_point(point: float) -> list[GridSpec]:
    grids = []
    for width in (2_000_000, 5_000_000, 10_000_000):
        grids.append(absolute_grid(width, 0))
        grids.append(absolute_grid(width, width // 2))
    for pct in (0.05, 0.10, 0.20):
        grids.append(relative_grid(point, pct, False))
        grids.append(relative_grid(point, pct, True))
    return grids


def score_distribution(dist: Distribution, actual: float, point: float, grid: GridSpec) -> dict[str, float | int]:
    cdf_edges = dist.cdf(grid.edges)
    cdf_edges[0] = 0.0
    cdf_edges[-1] = 1.0
    probs = smooth_probs(np.diff(np.maximum.accumulate(cdf_edges)))
    k = realized_bucket(actual, grid.edges)
    p = float(np.clip(probs[k], 1e-12, 1.0))
    pred_cdf = np.cumsum(probs)[:-1]
    thresholds = grid.edges[1:-1]
    finite = np.isfinite(thresholds)
    obs_cdf = (actual <= thresholds).astype(float)
    rps = float(np.mean((pred_cdf[finite] - obs_cdf[finite]) ** 2)) if finite.any() else np.nan
    threshold_brier = rps
    pinball = (QUANTILE_GRID - (actual < dist.quantiles).astype(float)) * (actual - dist.quantiles)
    crps = float(2.0 * np.trapezoid(pinball, QUANTILE_GRID))
    pit = float(np.interp(actual, dist.quantiles, QUANTILE_GRID, left=0.0, right=1.0))
    return {
        "bucket_index": k,
        "realized_bucket_probability": p,
        "log_loss": float(-np.log(p)),
        "rps": rps,
        "threshold_brier": threshold_brier,
        "crps_usd": crps,
        "ncrps": float(crps / point) if point > 0 else np.nan,
        "pit": pit,
        "covered_50": float(0.25 <= pit <= 0.75),
        "covered_80": float(0.10 <= pit <= 0.90),
        "covered_95": float(0.025 <= pit <= 0.975),
        "iqr_50_usd": float(np.interp(0.75, QUANTILE_GRID, dist.quantiles) - np.interp(0.25, QUANTILE_GRID, dist.quantiles)),
        "iqr_80_usd": float(np.interp(0.90, QUANTILE_GRID, dist.quantiles) - np.interp(0.10, QUANTILE_GRID, dist.quantiles)),
        "tail_side": "lower" if pit < 0.025 else ("upper" if pit > 0.975 else "inside"),
    }


def paired_bootstrap(scores: pd.DataFrame, base_candidate: str = "production_policy_v1") -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(20260711)
    metrics = ["ncrps", "log_loss", "rps", "threshold_brier"]
    base = scores.loc[scores["candidate"].eq(base_candidate)][
        ["movie_id", "origin_day", "grid", *metrics]
    ].rename(columns={metric: f"base_{metric}" for metric in metrics})
    for candidate, cand in scores.groupby("candidate"):
        if candidate == base_candidate:
            continue
        paired = cand.merge(base, on=["movie_id", "origin_day", "grid"], how="inner")
        if paired.empty:
            continue
        movies = paired["movie_id"].drop_duplicates().to_numpy()
        by_movie = {
            movie: paired.loc[paired["movie_id"].eq(movie)]
            for movie in movies
        }
        for metric in metrics:
            movie_delta = np.array(
                [
                    (by_movie[movie][metric] - by_movie[movie][f"base_{metric}"]).mean()
                    for movie in movies
                ],
                dtype=float,
            )
            boot = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)
            for idx in range(BOOTSTRAP_ITERATIONS):
                sample = rng.integers(0, len(movie_delta), size=len(movie_delta))
                boot[idx] = float(np.mean(movie_delta[sample]))
            rows.append(
                {
                    "candidate": candidate,
                    "base_candidate": base_candidate,
                    "metric": metric,
                    "paired_movies": int(len(movie_delta)),
                    "delta_mean": float(np.mean(movie_delta)),
                    "delta_ci025": float(np.quantile(boot, 0.025)),
                    "delta_ci975": float(np.quantile(boot, 0.975)),
                    "probability_challenger_beats_base": float((boot < 0).mean()),
                    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                }
            )
    return pd.DataFrame(rows)


def year_folds(scores: pd.DataFrame, base_candidate: str = "production_policy_v1") -> pd.DataFrame:
    metrics = ["ncrps", "log_loss", "rps", "threshold_brier"]
    base = scores.loc[scores["candidate"].eq(base_candidate)][
        ["holdout_year", "movie_id", "origin_day", "grid", *metrics]
    ].rename(columns={metric: f"base_{metric}" for metric in metrics})
    rows = []
    for candidate, cand in scores.groupby("candidate"):
        if candidate == base_candidate:
            continue
        paired = cand.merge(base, on=["holdout_year", "movie_id", "origin_day", "grid"], how="inner")
        for year, group in paired.groupby("holdout_year"):
            row = {"candidate": candidate, "base_candidate": base_candidate, "holdout_year": int(year), "paired_rows": int(len(group))}
            for metric in metrics:
                row[f"delta_{metric}"] = float((group[metric] - group[f"base_{metric}"]).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summary = scores.groupby(["origin_day", "candidate"]).agg(
        rows=("movie_id", "count"),
        movies=("movie_id", "nunique"),
        mean_ncrps=("ncrps", "mean"),
        mean_crps_usd=("crps_usd", "mean"),
        mean_log_loss=("log_loss", "mean"),
        mean_rps=("rps", "mean"),
        mean_threshold_brier=("threshold_brier", "mean"),
        median_realized_bucket_probability=("realized_bucket_probability", "median"),
        worst_5pct_log_loss=("log_loss", lambda s: s.quantile(0.95)),
        worst_1pct_log_loss=("log_loss", lambda s: s.quantile(0.99)),
        pit_mean=("pit", "mean"),
        coverage_50=("covered_50", "mean"),
        coverage_80=("covered_80", "mean"),
        coverage_95=("covered_95", "mean"),
        mean_iqr_50_usd=("iqr_50_usd", "mean"),
        mean_iqr_80_usd=("iqr_80_usd", "mean"),
    ).reset_index()
    grid_summary = scores.groupby(["grid", "candidate"]).agg(
        rows=("movie_id", "count"),
        mean_log_loss=("log_loss", "mean"),
        mean_rps=("rps", "mean"),
        mean_threshold_brier=("threshold_brier", "mean"),
    ).reset_index()
    tail = scores.groupby(["origin_day", "candidate", "tail_side"]).size().rename("rows").reset_index()
    return {
        "locked_distribution_summary_by_origin": summary,
        "locked_distribution_summary_by_grid": grid_summary,
        "locked_distribution_tail_miss_balance": tail,
        "locked_distribution_bootstrap_vs_production": paired_bootstrap(scores),
        "locked_distribution_leave_year_out_vs_production": year_folds(scores),
    }


def main() -> int:
    oof = prepare_oof()
    source = prepare_source_panel()
    policy = json.loads((OUTPUT_DIR / "frozen_pre_release_distribution_policy_v1.json").read_text(encoding="utf-8"))
    years = sorted(int(year) for year in oof["holdout_year"].dropna().unique())
    origins = sorted(int(origin) for origin in oof["origin_day"].dropna().unique())
    panel_rows = []
    score_rows = []

    for year in years:
        source_train = source.loc[source["release_year"].lt(year)].copy()
        source_holdout = source.loc[source["release_year"].eq(year)].copy()
        global_residuals = oof.loc[
            oof["holdout_year"].lt(year), "signed_log_error"
        ].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        for origin_day in origins:
            train = oof.loc[(oof["holdout_year"].lt(year)) & (oof["origin_day"].eq(origin_day))].copy()
            holdout = oof.loc[(oof["holdout_year"].eq(year)) & (oof["origin_day"].eq(origin_day))].copy()
            if holdout.empty or len(train) < MIN_ORIGIN_RESIDUALS:
                continue
            origin_residuals = train["signed_log_error"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            if len(origin_residuals) < MIN_ORIGIN_RESIDUALS:
                origin_residuals = global_residuals
            size_fit = fit_size_scale(train)
            production_name = production_candidate_for_origin(policy, origin_day)
            for _, row in holdout.iterrows():
                point = float(row["oof_point_forecast_usd"])
                actual = float(row["actual_opening_weekend_gross_usd"])
                dists = build_distributions(
                    train, source_train, source_holdout, row, origin_residuals, global_residuals, production_name, size_fit
                )
                meta = {
                    "movie_id": row["movie_id"],
                    "title": row["title"],
                    "opening_weekend_start": row["opening_weekend_start"],
                    "holdout_year": int(year),
                    "origin_day": int(origin_day),
                    "training_cutoff_year": int(year - 1),
                    "actual_opening_weekend_gross_usd": actual,
                    "frozen_point_forecast_usd": point,
                    "production_policy_candidate": production_name,
                    "source_count": int(row.get("source_count", 0) or 0),
                    "estimate_sources": row.get("estimate_sources", ""),
                }
                for candidate, dist in dists.items():
                    panel_rows.append(quantile_frame_row(meta, candidate, dist))
                    for grid in grids_for_point(point):
                        scored = score_distribution(dist, actual, point, grid)
                        score_rows.append({**meta, "candidate": candidate, "grid": grid.name, **scored})

    panel = pd.DataFrame(panel_rows)
    scores = pd.DataFrame(score_rows)
    panel_path = write_frame(panel, OUTPUT_DIR / "locked_rolling_origin_distribution_quantile_panel")
    scores_path = write_frame(scores, OUTPUT_DIR / "locked_distribution_multigrid_scores")
    outputs = summarize(scores)
    for stem, frame in outputs.items():
        frame.to_csv(OUTPUT_DIR / f"{stem}.csv", index=False)
    manifest = {
        "model_version": MODEL_VERSION,
        "point_policy": str(POLICY_PATH),
        "distribution_policy": str(OUTPUT_DIR / "frozen_pre_release_distribution_policy_v1.json"),
        "quantile_grid_size": int(len(QUANTILE_GRID)),
        "candidates": sorted(panel["candidate"].unique().tolist()),
        "grid_count": int(scores["grid"].nunique()),
        "panel_rows": int(len(panel)),
        "score_rows": int(len(scores)),
        "panel_path": str(panel_path),
        "scores_path": str(scores_path),
        "actual_market_grids": "not_used_in_this_run_no_structured_historical_grid_table_identified",
    }
    (OUTPUT_DIR / "locked_distribution_panel_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote locked distribution panel: {panel_path}")
    print(f"Wrote multi-grid scores: {scores_path}")
    print(f"Panel rows: {len(panel):,}")
    print(f"Score rows: {len(scores):,}")
    print(f"Candidates: {panel['candidate'].nunique():,}")
    print(f"Grids: {scores['grid'].nunique():,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
