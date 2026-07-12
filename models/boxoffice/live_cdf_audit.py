#!/usr/bin/env python3
"""Locked, leakage-safe scorecard for persisted live opening-weekend CDFs.

The runner intentionally scores only rows supplied by an immutable emission or
historical replay panel.  It never refits component models, so candidate versus
benchmark comparisons remain paired on the same movie/state/origin.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED = {"movie_id", "information_state", "clock_origin", "actual_ow", "quantile_levels", "quantile_values_usd"}


def _payload(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def build_panel(emissions: pd.DataFrame) -> pd.DataFrame:
    """Normalize persisted OW emissions into one audit row per forecast."""

    records: list[dict[str, Any]] = []
    for row in emissions.to_dict("records"):
        payload = _payload(row.get("distribution_payload", row.get("payload")))
        levels = payload.get("quantile_levels", row.get("quantile_levels"))
        values = payload.get("quantile_values_usd", row.get("quantile_values_usd"))
        try:
            levels = [float(x) for x in levels]
            values = [float(x) for x in values]
        except (TypeError, ValueError):
            continue
        if len(levels) < 101 or len(levels) != len(values) or any(a >= b for a, b in zip(levels, levels[1:])):
            continue
        actual = row.get("actual_ow", row.get("actual_usd"))
        try:
            actual = float(actual)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(actual) or actual <= 0:
            continue
        state = row.get("information_state") or payload.get("information_state") or row.get("regime")
        origin = row.get("clock_origin") or payload.get("forecast_origin") or row.get("origin_key")
        records.append({
            "movie_id": row.get("movie_id"),
            "release_year": pd.to_datetime(row.get("opening_weekend_start"), errors="coerce").year,
            "information_state": str(state or "unknown"),
            "clock_origin": str(origin or "unknown"),
            "actual_ow": actual,
            "emitted_median": float(payload.get("point_forecast_usd", row.get("point_usd", np.interp(.5, levels, values)))),
            "quantile_levels": levels,
            "quantile_values_usd": values,
            "fixed_grid_probabilities": payload.get("market_bucket_probabilities", {}),
            "component_sources": payload.get("component_sources", row.get("component_sources")),
            "known_actuals": payload.get("known_actual_days", row.get("known_actual_days")),
            "amc_eligible": bool(payload.get("AMC_component")),
            "fallback_reason": row.get("fallback_reason"),
            "policy_version": payload.get("policy_version", row.get("policy_version")),
            "candidate": str(row.get("candidate", "production")),
            "forecast_id": row.get("forecast_id"),
        })
    return pd.DataFrame(records)


def _sample(row: pd.Series) -> np.ndarray:
    levels = np.asarray(row["quantile_levels"], dtype=float)
    values = np.asarray(row["quantile_values_usd"], dtype=float)
    return np.interp(np.arange(1, len(levels) + 1) / (len(levels) + 1), levels, values)


def _crps(sample: np.ndarray, actual: float) -> float:
    x = np.sort(sample)
    n = len(x)
    pairwise = 2 * np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / n**2
    return float(np.mean(np.abs(x - actual)) - .5 * pairwise)


def _entropy(probabilities: np.ndarray) -> float:
    p = probabilities[probabilities > 0]
    return float(-np.sum(p * np.log(p))) if len(p) else 0.0


def _bucket_probabilities(sample: np.ndarray, actual: float, grids: object) -> tuple[np.ndarray, int]:
    if isinstance(grids, dict) and grids:
        for key, raw in grids.items():
            if not isinstance(key, str) or not key.startswith("boundaries:") or not isinstance(raw, list) or len(raw) != 5:
                continue
            try:
                edges = np.asarray([float(value) for value in key.split(":", 1)[1].split(",")], dtype=float)
                probs = np.asarray(raw, dtype=float)
            except ValueError:
                continue
            if len(edges) == 4 and np.isfinite(probs).all() and probs.sum() > 0:
                return probs / probs.sum(), int(np.searchsorted(edges, actual, side="right"))
    # Fixed, declared diagnostic grid; production grid scores use stored rows.
    edges = np.quantile(sample, [.20, .40, .60, .80])
    index = int(np.searchsorted(edges, actual, side="right"))
    counts = np.bincount(np.searchsorted(edges, sample, side="right"), minlength=5)
    return counts / counts.sum(), index


def score_panel(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in panel.iterrows():
        sample, actual = _sample(row), float(row["actual_ow"])
        q = lambda p: float(np.interp(p, row["quantile_levels"], row["quantile_values_usd"]))
        pit = float(np.mean(sample <= actual))
        probabilities, bucket = _bucket_probabilities(sample, actual, row["fixed_grid_probabilities"])
        realized_probability = max(float(probabilities[bucket]), 1e-12)
        observed_cdf = (np.arange(4) >= bucket).astype(float)
        rows.append({
            **row.drop(labels=["quantile_levels", "quantile_values_usd"]).to_dict(),
            "mae_log": abs(math.log(actual / q(.5))),
            "rmse_log_sq": math.log(actual / q(.5)) ** 2,
            "signed_log_error": math.log(actual / q(.5)),
            "crps": _crps(sample, actual),
            "ncrps": _crps(sample, actual) / actual,
            "pit": pit,
            "coverage_50": q(.25) <= actual <= q(.75),
            "coverage_80": q(.10) <= actual <= q(.90),
            "coverage_95": q(.025) <= actual <= q(.975),
            "lower_miss_95": actual < q(.025),
            "upper_miss_95": actual > q(.975),
            "width_80": q(.90) - q(.10),
            "width_95": q(.975) - q(.025),
            "width_80_over_median": (q(.90) - q(.10)) / q(.5),
            "bucket_log_loss": -math.log(realized_probability),
            "rps": float(np.sum((np.cumsum(probabilities)[:-1] - observed_cdf) ** 2)),
            "threshold_brier": float(np.mean((np.cumsum(probabilities)[:-1] - observed_cdf) ** 2)),
            "entropy": _entropy(probabilities),
            "realized_bucket_probability": realized_probability,
        })
    return pd.DataFrame(rows)


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    rows = []
    metrics = ["mae_log", "rmse_log_sq", "signed_log_error", "crps", "ncrps", "pit", "coverage_50", "coverage_80", "coverage_95", "lower_miss_95", "upper_miss_95", "width_80", "width_95", "width_80_over_median", "bucket_log_loss", "rps", "threshold_brier", "entropy", "realized_bucket_probability"]
    for keys, group in scores.groupby(["candidate", "information_state", "clock_origin"], dropna=False):
        item = dict(zip(["candidate", "information_state", "clock_origin"], keys, strict=True))
        item.update({"movies": group.movie_id.nunique(), "forecasts": len(group), "amc_eligible": int(group.amc_eligible.sum()), "fallbacks": int(group.fallback_reason.notna().sum())})
        for metric in metrics:
            item[metric] = float(group[metric].mean())
        item["rmse_log"] = math.sqrt(item.pop("rmse_log_sq"))
        rows.append(item)
    return pd.DataFrame(rows)


def paired_bootstrap(scores: pd.DataFrame, *, iterations: int = 2_000, seed: int = 17) -> pd.DataFrame:
    """Movie-clustered candidate-minus-benchmark score deltas."""

    rows = []
    keys = ["movie_id", "information_state", "clock_origin"]
    for state, group in scores.groupby("information_state"):
        base = group[group.candidate.eq("benchmark")]
        candidate = group[group.candidate.eq("candidate")]
        paired = candidate.merge(base[keys + ["ncrps", "bucket_log_loss", "rps"]], on=keys, suffixes=("_candidate", "_benchmark"))
        if paired.empty:
            continue
        rng = np.random.default_rng(seed)
        movies = paired.movie_id.unique()
        for metric in ["ncrps", "bucket_log_loss", "rps"]:
            delta = paired[f"{metric}_candidate"] - paired[f"{metric}_benchmark"]
            by_movie = {movie: delta[paired.movie_id.eq(movie)].to_numpy() for movie in movies}
            boot = [np.concatenate([by_movie[movie] for movie in rng.choice(movies, len(movies), replace=True)]).mean() for _ in range(iterations)]
            rows.append({"information_state": state, "metric": metric, "pairs": len(paired), "movies": len(movies), "mean_delta": float(delta.mean()), "bootstrap_lo95": float(np.quantile(boot, .025)), "bootstrap_hi95": float(np.quantile(boot, .975)), "probability_candidate_improves": float(np.mean(np.asarray(boot) < 0))})
    return pd.DataFrame(rows)


def dependence_diagnostic(daily: pd.DataFrame) -> pd.DataFrame:
    pairs = [("Friday", "Saturday", "pre_fri_usd", "after_fri_sat_usd"), ("Friday", "Sunday", "pre_fri_usd", "after_fri_sun_usd"), ("Saturday", "Sunday", "after_fri_sat_usd", "after_fri_sun_usd")]
    rows = []
    for left, right, left_pred, right_pred in pairs:
        left_actual, right_actual = f"actual_{left[:3].lower()}_usd", f"actual_{right[:3].lower()}_usd"
        if not {left_pred, right_pred, left_actual, right_actual}.issubset(daily):
            continue
        valid = daily[[left_pred, right_pred, left_actual, right_actual]].apply(pd.to_numeric, errors="coerce").gt(0).all(axis=1)
        a = np.log(pd.to_numeric(daily.loc[valid, left_actual]) / pd.to_numeric(daily.loc[valid, left_pred]))
        b = np.log(pd.to_numeric(daily.loc[valid, right_actual]) / pd.to_numeric(daily.loc[valid, right_pred]))
        rows.append({"pair": f"{left}-{right}", "n": int(valid.sum()), "log_residual_correlation": float(a.corr(b)) if len(a) > 1 else np.nan})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emissions-csv", type=Path, required=True)
    parser.add_argument("--daily-baseline-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/diagnostics/live_cdf_audit"))
    args = parser.parse_args(argv)
    panel = build_panel(pd.read_csv(args.emissions_csv))
    scores = score_panel(panel)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.output_dir / "01_live_cdf_panel.csv", index=False)
    scores.to_csv(args.output_dir / "02_live_cdf_scores.csv", index=False)
    summarize(scores).to_csv(args.output_dir / "03_live_cdf_summary.csv", index=False)
    paired_bootstrap(scores).to_csv(args.output_dir / "04_amc_paired_bootstrap.csv", index=False)
    if args.daily_baseline_csv:
        dependence_diagnostic(pd.read_csv(args.daily_baseline_csv)).to_csv(args.output_dir / "05_daily_dependence.csv", index=False)
    print(f"Wrote locked live-CDF audit to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
