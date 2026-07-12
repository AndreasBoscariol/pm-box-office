#!/usr/bin/env python3
"""Audit catastrophic distribution losses and bounded safety-layer challengers."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import t as student_t

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eda.Prior.daily_distribution_validation import OUTPUT_DIR, prepare_oof, realized_bucket
from eda.Prior.locked_distribution_panel_evaluation import (
    QUANTILE_GRID,
    Distribution,
    grids_for_point,
    mix_distributions,
    score_distribution,
)

PANEL_PATH = OUTPUT_DIR / "locked_rolling_origin_distribution_quantile_panel.parquet"
SCORES_PATH = OUTPUT_DIR / "locked_distribution_multigrid_scores.parquet"
EPSILONS = (0.0025, 0.005, 0.01, 0.02)
T_DF = 4
BOOTSTRAPS = 2000
NONINFERIORITY_REL = 0.005
QCOLS = [f"q_{q:.3f}" for q in QUANTILE_GRID]


def dist_from_row(row: pd.Series) -> Distribution:
    return Distribution(row[QCOLS].to_numpy(dtype=float), str(row.get("fallback_level", "locked")))


def grid_for(name: str, point: float):
    return next(grid for grid in grids_for_point(point) if grid.name == name)


def median_center(dist: Distribution, point: float, label: str) -> Distribution:
    median = float(np.interp(0.5, QUANTILE_GRID, dist.quantiles))
    curve = dist.quantiles * point / median if median > 0 else dist.quantiles
    return Distribution(np.maximum.accumulate(curve), label)


def beta_map(p: np.ndarray, a: float, b: float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    z = a * np.log(p) - b * np.log1p(-p) + (a - b) * np.log(2.0)
    return expit(z)


def calibrated_dist(dist: Distribution, a: float, b: float, label: str) -> Distribution:
    mapped = np.maximum.accumulate(beta_map(QUANTILE_GRID, a, b))
    quantiles = np.interp(QUANTILE_GRID, mapped, dist.quantiles, left=dist.quantiles[0], right=dist.quantiles[-1])
    return Distribution(np.maximum.accumulate(quantiles), label)


def origin_group(origin: int) -> str:
    if origin <= -10:
        return "-14_to_-10"
    if origin <= -5:
        return "-9_to_-5"
    if origin <= -2:
        return "-4_to_-2"
    return "-1"


def fit_beta(pits: np.ndarray, strength: float) -> tuple[float, float]:
    u = np.clip(pits[np.isfinite(pits)], 1e-6, 1 - 1e-6)
    if len(u) < 80:
        return 1.0, 1.0

    def objective(theta: np.ndarray) -> float:
        a, b = np.exp(theta)
        mapped = np.sort(beta_map(u, a, b))
        target = (np.arange(len(mapped)) + 0.5) / len(mapped)
        return float(np.mean((mapped - target) ** 2) + strength * ((a - 1) ** 2 + (b - 1) ** 2))

    result = minimize(objective, np.zeros(2), method="L-BFGS-B", bounds=[(-1.5, 1.5), (-1.5, 1.5)])
    return tuple(np.exp(result.x)) if result.success else (1.0, 1.0)


def catastrophic_audit(panel: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    prod_scores = scores.loc[scores.candidate.eq("production_policy_v1")].copy()
    cutoff = float(prod_scores.log_loss.quantile(0.99))
    bad = prod_scores.loc[prod_scores.log_loss.ge(cutoff)].copy()
    prod = panel.loc[panel.candidate.eq("production_policy_v1")]
    shrunk = panel.loc[panel.candidate.eq("D2_source_shrunk_k40_equal_centered")]
    keys = ["movie_id", "origin_day", "holdout_year"]
    prod_map = {tuple(row[k] for k in keys): row for _, row in prod.iterrows()}
    shrunk_map = {tuple(row[k] for k in keys): row for _, row in shrunk.iterrows()}
    rows = []
    for _, score in bad.iterrows():
        key = tuple(score[k] for k in keys)
        prow, srow = prod_map[key], shrunk_map[key]
        pdist, sdist = dist_from_row(prow), dist_from_row(srow)
        point, actual = float(score.frozen_point_forecast_usd), float(score.actual_opening_weekend_gross_usd)
        grid = grid_for(str(score.grid), point)
        k = realized_bucket(actual, grid.edges)
        lo, hi = float(grid.edges[k]), float(grid.edges[k + 1])
        pp = float(max(pdist.cdf(hi) - pdist.cdf(lo), 0))
        sp = float(max(sdist.cdf(hi) - sdist.cdf(lo), 0))
        outside = actual < pdist.quantiles[0] or actual > pdist.quantiles[-1]
        nearest = float(np.min(np.abs(pdist.quantiles - actual)))
        open_ended = not np.isfinite(hi) or lo <= 0
        # Boundary sensitivity: move finite boundaries by 2% of bucket width.
        if np.isfinite(hi):
            width = max(hi - lo, 1.0)
            shifted = max(pdist.cdf(hi + .02 * width) - pdist.cdf(max(0, lo + .02 * width)), 0)
            artifact = pp > 0 and abs(shifted - pp) / pp > 2
        else:
            artifact = False
        if artifact:
            failure = "grid_interpolation_artifact"
        elif outside:
            failure = "hard_support_failure"
        elif pp < 1e-4:
            failure = "sparse_tail_failure"
        else:
            failure = "general_calibration_failure"
        rows.append({
            **{c: score.get(c) for c in ["movie_id", "title", "origin_day", "holdout_year", "source_count", "estimate_sources", "grid", "log_loss"]},
            "actual_opening_weekend_gross_usd": actual, "frozen_point_forecast_usd": point,
            "point_size_group": pd.qcut(panel.frozen_point_forecast_usd, 4, labels=["small", "medium", "large", "very_large"], duplicates="drop").loc[prow.name],
            "realized_bucket_lower_usd": lo, "realized_bucket_upper_usd": hi,
            "production_realized_bucket_probability_raw": pp, "shrunk_k40_realized_bucket_probability_raw": sp,
            "production_pit": float(pdist.cdf(actual)), "minimum_represented_quantile_usd": float(pdist.quantiles[0]),
            "maximum_represented_quantile_usd": float(pdist.quantiles[-1]), "actual_outside_represented_support": outside,
            "distance_to_nearest_production_quantile_usd": nearest, "realized_bucket_open_ended": open_ended,
            "failure_class": failure,
        })
    return pd.DataFrame(rows)


def movie_equal_summary(scores: pd.DataFrame, base: str = "production_policy_v1") -> pd.DataFrame:
    metrics = ["log_loss", "ncrps", "rps"]
    by_origin = scores.groupby(["candidate", "movie_id", "origin_day"], as_index=False)[metrics].mean()
    by_movie = by_origin.groupby(["candidate", "movie_id"], as_index=False)[metrics].mean()
    summary = by_movie.groupby("candidate")[metrics].agg(["mean", "count"]).reset_index()
    summary.columns = ["candidate"] + [f"{a}_{b}" for a, b in summary.columns.tolist()[1:]]
    base_row = summary.loc[summary.candidate.eq(base)].iloc[0]
    for metric in metrics:
        summary[f"delta_{metric}_vs_production"] = summary[f"{metric}_mean"] - base_row[f"{metric}_mean"]
        summary[f"relative_delta_{metric}_vs_production"] = summary[f"delta_{metric}_vs_production"] / base_row[f"{metric}_mean"]
    return summary


def build_challengers(panel: pd.DataFrame, oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = panel.loc[panel.candidate.eq("production_policy_v1")].copy()
    d0 = panel.loc[panel.candidate.eq("D0_origin_empirical")].set_index(["movie_id", "origin_day", "holdout_year"])
    prior_pits: dict[tuple[int, str], np.ndarray] = {}
    pit_rows = []
    for _, row in base.iterrows():
        pit_rows.append({"year": int(row.holdout_year), "group": origin_group(int(row.origin_day)), "pit": float(dist_from_row(row).cdf(float(row.actual_opening_weekend_gross_usd)))})
    pit_df = pd.DataFrame(pit_rows)
    for year in sorted(base.holdout_year.unique()):
        for group in pit_df.group.unique():
            prior_pits[(int(year), group)] = pit_df.loc[(pit_df.year < year) & pit_df.group.eq(group), "pit"].to_numpy()

    oof = oof.copy()
    score_rows, param_rows = [], []
    for _, row in base.iterrows():
        point, actual, year, origin = float(row.frozen_point_forecast_usd), float(row.actual_opening_weekend_gross_usd), int(row.holdout_year), int(row.origin_day)
        prod = dist_from_row(row)
        key = (row.movie_id, row.origin_day, row.holdout_year)
        empirical = median_center(dist_from_row(d0.loc[key]), point, "centered_D0")
        prior = oof.loc[(oof.holdout_year < year) & (oof.origin_day == origin), "signed_log_error"].dropna().to_numpy(dtype=float)
        scale = float(np.median(np.abs(prior)) / student_t.ppf(.75, T_DF)) if len(prior) >= 30 else .35
        tq = point * np.exp(student_t.ppf(QUANTILE_GRID, T_DF) * max(scale, .03))
        heavy = median_center(Distribution(tq, "student_t4"), point, "centered_student_t4")
        candidates = {"production_policy_v1": prod}
        for ref_name, ref in [("D0", empirical), ("t4", heavy)]:
            for eps in EPSILONS:
                name = f"tail_{ref_name}_eps_{eps:g}"
                candidates[name] = mix_distributions([(1 - eps, prod), (eps, ref)], name)
        group = origin_group(origin)
        for strength in (0.01, 0.05, 0.20):
            a, b = fit_beta(prior_pits[(year, group)], strength)
            name = f"beta_strength_{strength:g}"
            candidates[name] = calibrated_dist(prod, a, b, name)
            param_rows.append({"holdout_year": year, "origin_group": group, "strength": strength, "a": a, "b": b, "history_n": len(prior_pits[(year, group)])})
        meta = {c: row[c] for c in ["movie_id", "title", "holdout_year", "origin_day", "actual_opening_weekend_gross_usd", "frozen_point_forecast_usd"]}
        for name, dist in candidates.items():
            for grid in grids_for_point(point):
                score_rows.append({**meta, "candidate": name, "grid": grid.name, **score_distribution(dist, actual, point, grid)})
    return pd.DataFrame(score_rows), pd.DataFrame(param_rows).drop_duplicates()


def nested_outer_selection(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["log_loss", "ncrps", "rps"]
    mo = scores.groupby(["holdout_year", "candidate", "movie_id", "origin_day"], as_index=False)[metrics].mean()
    movie = mo.groupby(["holdout_year", "candidate", "movie_id"], as_index=False)[metrics].mean()
    selections, outer = [], []
    years = sorted(movie.holdout_year.unique())
    for year in years:
        train = movie.loc[movie.holdout_year < year]
        if train.empty:
            continue
        means = train.groupby("candidate")[metrics].mean()
        base = means.loc["production_policy_v1"]
        eligible = means.loc[(means.ncrps <= base.ncrps * (1 + NONINFERIORITY_REL)) & (means.rps <= base.rps * (1 + NONINFERIORITY_REL))]
        chosen = str(eligible.log_loss.idxmin()) if not eligible.empty else "production_policy_v1"
        selections.append({"holdout_year": int(year), "selected_candidate": chosen, "training_years": int(train.holdout_year.nunique()), "train_log_loss": float(means.loc[chosen, "log_loss"]), "train_ncrps": float(means.loc[chosen, "ncrps"]), "train_rps": float(means.loc[chosen, "rps"])})
        selected = scores.loc[(scores.holdout_year.eq(year)) & (scores.candidate.eq(chosen))].copy()
        selected["candidate"] = "nested_selected_policy"
        selected["selected_candidate"] = chosen
        outer.append(selected)
    outer_scores = pd.concat(outer, ignore_index=True) if outer else pd.DataFrame()
    return pd.DataFrame(selections), outer_scores


def bootstrap(scores: pd.DataFrame, candidate: str) -> pd.DataFrame:
    metrics = ["log_loss", "ncrps", "rps"]
    subset = scores.loc[scores.candidate.isin([candidate, "production_policy_v1"])]
    mo = subset.groupby(["candidate", "movie_id", "origin_day"], as_index=False)[metrics].mean()
    movie = mo.groupby(["candidate", "movie_id"], as_index=False)[metrics].mean()
    paired = movie.loc[movie.candidate.eq(candidate)].merge(movie.loc[movie.candidate.eq("production_policy_v1")], on="movie_id", suffixes=("_candidate", "_base"))
    rng, rows = np.random.default_rng(20260711), []
    for metric in metrics:
        delta = paired[f"{metric}_candidate"].to_numpy() - paired[f"{metric}_base"].to_numpy()
        draws = np.array([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(BOOTSTRAPS)])
        rows.append({"candidate": candidate, "metric": metric, "movies": len(delta), "delta_mean": delta.mean(), "ci025": np.quantile(draws, .025), "ci975": np.quantile(draws, .975), "probability_better": np.mean(draws < 0)})
    return pd.DataFrame(rows)


def main() -> int:
    panel, locked_scores, oof = pd.read_parquet(PANEL_PATH), pd.read_parquet(SCORES_PATH), prepare_oof()
    audit = catastrophic_audit(panel, locked_scores)
    audit.to_csv(OUTPUT_DIR / "locked_distribution_catastrophic_loss_audit.csv", index=False)
    audit.groupby("failure_class").size().rename("observations").reset_index().to_csv(OUTPUT_DIR / "locked_distribution_catastrophic_loss_classes.csv", index=False)
    movie_equal_summary(locked_scores).to_csv(OUTPUT_DIR / "locked_distribution_movie_equal_estimand.csv", index=False)
    scores, params = build_challengers(panel, oof)
    scores.to_parquet(OUTPUT_DIR / "locked_distribution_tail_calibration_scores.parquet", index=False)
    params.to_csv(OUTPUT_DIR / "locked_distribution_beta_parameters.csv", index=False)
    summary = movie_equal_summary(scores)
    summary.to_csv(OUTPUT_DIR / "locked_distribution_tail_calibration_summary.csv", index=False)
    selections, outer = nested_outer_selection(scores)
    selections.to_csv(OUTPUT_DIR / "locked_distribution_nested_outer_selections.csv", index=False)
    if not outer.empty:
        combined = pd.concat([scores.loc[scores.candidate.eq("production_policy_v1") & scores.holdout_year.isin(outer.holdout_year.unique())], outer], ignore_index=True)
        movie_equal_summary(combined).to_csv(OUTPUT_DIR / "locked_distribution_nested_outer_summary.csv", index=False)
        bootstrap(combined, "nested_selected_policy").to_csv(OUTPUT_DIR / "locked_distribution_nested_outer_bootstrap.csv", index=False)
    manifest = {"catastrophic_rows": len(audit), "failure_classes": audit.failure_class.value_counts().to_dict(), "candidate_score_rows": len(scores), "noninferiority_relative_tolerance": NONINFERIORITY_REL, "tail_epsilons": EPSILONS, "student_t_df": T_DF}
    (OUTPUT_DIR / "locked_distribution_tail_calibration_manifest.json").write_text(json.dumps(manifest, indent=2, default=float) + "\n")
    print(json.dumps(manifest, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
