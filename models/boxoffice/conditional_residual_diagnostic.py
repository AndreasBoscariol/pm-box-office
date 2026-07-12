#!/usr/bin/env python3
"""Rolling diagnostics for surprise-conditioned live weekend residual bootstraps."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

THRESHOLDS_USD = np.array([10e6, 15e6, 20e6, 25e6, 30e6, 35e6, 40e6, 50e6, 60e6, 75e6, 100e6, 125e6, 150e6])
BUCKET_COUNT = len(THRESHOLDS_USD) + 1
LAMBDA_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _interval_score(y: float, lower: float, upper: float, alpha: float) -> float:
    return float((upper - lower) + (2 / alpha) * (lower - y) * (y < lower) + (2 / alpha) * (y - upper) * (y > upper))


def _crps_sample(draws: np.ndarray, y: float) -> float:
    sample = np.sort(np.asarray(draws, dtype="float64"))
    n = len(sample)
    if n == 0:
        return math.nan
    coeff = np.arange(1, n + 1)
    mean_pairwise = 2 * np.sum((2 * coeff - n - 1) * sample) / (n * n)
    return float(np.mean(np.abs(sample - y)) - 0.5 * mean_pairwise)


def _bucket_index(actual: float) -> int:
    return int(np.searchsorted(THRESHOLDS_USD, actual))


def _bucket_edges(index: int) -> tuple[float, float]:
    lower = float("-inf") if index == 0 else float(THRESHOLDS_USD[index - 1])
    upper = float("inf") if index == len(THRESHOLDS_USD) else float(THRESHOLDS_USD[index])
    return lower, upper


def _bucket_counts(draws: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(THRESHOLDS_USD, draws)
    return np.bincount(indices, minlength=BUCKET_COUNT).astype("float64")


def _smoothed_bucket_probabilities(draws: np.ndarray, smoothing_alpha: float) -> np.ndarray:
    counts = _bucket_counts(draws)
    return (counts + smoothing_alpha) / (len(draws) + BUCKET_COUNT * smoothing_alpha)


def _probability_scores(draws: np.ndarray, actual: float, *, smoothing_alpha: float) -> dict[str, float]:
    probabilities = np.asarray([np.mean(draws <= threshold) for threshold in THRESHOLDS_USD], dtype="float64")
    observed = (actual <= THRESHOLDS_USD).astype("float64")
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    bucket = _bucket_index(actual)
    bucket_probs = _smoothed_bucket_probabilities(draws, smoothing_alpha)
    observed_cdf = (np.arange(len(THRESHOLDS_USD)) >= bucket).astype("float64")
    realized_probability = float(bucket_probs[bucket])
    rank = int(1 + np.sum(bucket_probs > realized_probability))
    return {
        "threshold_brier": float(np.mean((probabilities - observed) ** 2)),
        "threshold_log_loss": float(np.mean(-(observed * np.log(clipped) + (1 - observed) * np.log1p(-clipped)))),
        "bucket_log_loss": float(-np.log(realized_probability)),
        "canonical_rps": float(np.sum((np.cumsum(bucket_probs)[:-1] - observed_cdf) ** 2)),
        "realized_bucket": bucket,
        "realized_bucket_probability": realized_probability,
        "realized_bucket_rank": rank,
        "near_zero_bucket_count": int(np.sum(bucket_probs < 0.001)),
    }


def _weights(x0: float, x: np.ndarray, bandwidth: float, shrink_k: float) -> tuple[np.ndarray, float, float]:
    local = np.exp(-((x0 - x) ** 2) / (2 * bandwidth**2))
    if not np.isfinite(local).all() or local.sum() <= 0:
        local = np.ones_like(x, dtype="float64")
    local = local / local.sum()
    n_eff = 1.0 / float(np.sum(local**2))
    alpha = n_eff / (n_eff + shrink_k)
    pooled = np.full_like(local, 1.0 / len(local), dtype="float64")
    final = alpha * local + (1 - alpha) * pooled
    final = final / final.sum()
    return final, n_eff, alpha


def _sample_crps_draws(draws: np.ndarray, size: int = 5000) -> np.ndarray:
    if len(draws) <= size:
        return draws
    return draws[:size]


def _score_draws(
    candidate: str,
    row: pd.Series,
    draws: np.ndarray,
    *,
    n_prior: int,
    n_eff: float | None,
    alpha: float | None,
    smoothing_alpha: float,
    selected_lambda: float | None = None,
) -> dict[str, object]:
    quantiles = np.quantile(draws, [0.025, 0.10, 0.25, 0.75, 0.90, 0.975])
    actual = float(row["actual_ow_usd"])
    out: dict[str, object] = {
        "candidate": candidate,
        "movie_id": int(row["movie_id"]),
        "release_run_id": int(row["release_run_id"]) if pd.notna(row.get("release_run_id")) else int(row["movie_id"]),
        "title": row.get("title", ""),
        "release_date": row["release_date"],
        "weekend_key": row.get("friday_date", row["release_date"]),
        "actual_ow_usd": actual,
        "point_median_usd": float(np.quantile(draws, 0.50)),
        "prior_movies": n_prior,
        "n_eff": n_eff,
        "alpha": alpha,
        "lo95": float(quantiles[0]),
        "lo80": float(quantiles[1]),
        "lo50": float(quantiles[2]),
        "hi50": float(quantiles[3]),
        "hi80": float(quantiles[4]),
        "hi95": float(quantiles[5]),
        "crps": _crps_sample(_sample_crps_draws(draws), actual),
        "wis80": _interval_score(actual, float(quantiles[1]), float(quantiles[4]), 0.20),
        "wis95": _interval_score(actual, float(quantiles[0]), float(quantiles[5]), 0.05),
        "selected_lambda": selected_lambda,
    }
    lower, upper = _bucket_edges(_bucket_index(actual))
    out["realized_bucket_lower"] = lower
    out["realized_bucket_upper"] = upper
    out.update(_probability_scores(draws, actual, smoothing_alpha=smoothing_alpha))
    return out


def _summarize_scores(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, group in scored.groupby("candidate", sort=False):
        actual = group["actual_ow_usd"]
        record: dict[str, object] = {"candidate": candidate, "n": int(len(group))}
        for level, lo, hi, alpha in [(50, "lo50", "hi50", 0.50), (80, "lo80", "hi80", 0.20), (95, "lo95", "hi95", 0.05)]:
            record[f"coverage_{level}"] = float(((actual >= group[lo]) & (actual <= group[hi])).mean())
            record[f"lower_miss_{level}"] = float((actual < group[lo]).mean())
            record[f"upper_miss_{level}"] = float((actual > group[hi]).mean())
            record[f"width_{level}_m"] = float(((group[hi] - group[lo]) / 1e6).mean())
            record[f"wis_{level}_m"] = float(np.mean([_interval_score(y, lower, upper, alpha) for y, lower, upper in zip(actual, group[lo], group[hi])]) / 1e6)
        for column in ["crps", "threshold_brier", "threshold_log_loss", "bucket_log_loss", "canonical_rps"]:
            record[column] = float(group[column].mean())
        if group["n_eff"].notna().any():
            record["mean_n_eff"] = float(group["n_eff"].mean())
            record["mean_alpha"] = float(group["alpha"].mean())
        rows.append(record)
    return pd.DataFrame(rows)


def _paired_delta_report(
    scores: pd.DataFrame,
    *,
    benchmark: str,
    candidates: list[str],
    seed: int,
    iterations: int,
) -> pd.DataFrame:
    metrics = ["crps", "wis80", "wis95", "threshold_brier", "threshold_log_loss", "bucket_log_loss", "canonical_rps"]
    keys = ["release_run_id", "weekend_key", "release_date"]
    base = scores.loc[scores["candidate"].eq(benchmark), keys + metrics].rename(columns={metric: f"{metric}_benchmark" for metric in metrics})
    rows = []
    rng = np.random.default_rng(seed)
    for candidate in candidates:
        cand = scores.loc[scores["candidate"].eq(candidate), keys + metrics]
        paired = cand.merge(base, on=keys, how="inner")
        if paired.empty:
            continue
        weekend_keys = paired["weekend_key"].astype(str).to_numpy()
        unique_weekends = np.unique(weekend_keys)
        for metric in metrics:
            delta = paired[metric].to_numpy("float64") - paired[f"{metric}_benchmark"].to_numpy("float64")
            groups = {weekend: delta[weekend_keys == weekend] for weekend in unique_weekends}
            estimates = np.empty(iterations, dtype="float64")
            for idx in range(iterations):
                sampled = rng.choice(unique_weekends, size=len(unique_weekends), replace=True)
                estimates[idx] = float(np.concatenate([groups[weekend] for weekend in sampled]).mean())
            rows.append(
                {
                    "candidate": candidate,
                    "benchmark": benchmark,
                    "metric": metric,
                    "n_rows": int(len(paired)),
                    "n_weekends": int(len(unique_weekends)),
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(np.median(delta)),
                    "bootstrap_lo95": float(np.quantile(estimates, 0.025)),
                    "bootstrap_hi95": float(np.quantile(estimates, 0.975)),
                    "prob_delta_below_zero": float(np.mean(estimates < 0)),
                }
            )
    return pd.DataFrame(rows)


def _leave_one_year_delta_report(scores: pd.DataFrame, *, benchmark: str, candidates: list[str]) -> pd.DataFrame:
    metrics = ["crps", "wis80", "wis95", "threshold_brier", "threshold_log_loss", "bucket_log_loss", "canonical_rps"]
    work = scores.copy()
    work["year"] = pd.to_datetime(work["release_date"]).dt.year
    keys = ["release_run_id", "release_date"]
    base = work.loc[work["candidate"].eq(benchmark), keys + ["year"] + metrics].rename(columns={metric: f"{metric}_benchmark" for metric in metrics})
    rows = []
    for year in sorted(work["year"].dropna().unique()):
        for candidate in candidates:
            cand = work.loc[work["candidate"].eq(candidate), keys + metrics]
            paired = cand.merge(base, on=keys, how="inner")
            paired = paired.loc[paired["year"].ne(year)]
            if paired.empty:
                continue
            for metric in metrics:
                delta = paired[metric] - paired[f"{metric}_benchmark"]
                rows.append(
                    {
                        "left_out_year": int(year),
                        "candidate": candidate,
                        "benchmark": benchmark,
                        "metric": metric,
                        "n_rows": int(len(paired)),
                        "mean_delta": float(delta.mean()),
                        "delta_below_zero": bool(delta.mean() < 0),
                    }
                )
    return pd.DataFrame(rows)


def _after_friday_bucket_audit(scores: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        "AF_independent_pooled",
        "AF_joint_pooled",
        "AF_joint_cond_friday_surprise",
        "AF_mixture_lambda_train_bucket",
    ]
    cols = [
        "release_run_id", "movie_id", "title", "release_date", "actual_ow_usd",
        "realized_bucket", "realized_bucket_lower", "realized_bucket_upper",
        "candidate", "realized_bucket_probability", "bucket_log_loss",
        "near_zero_bucket_count", "realized_bucket_rank", "selected_lambda",
    ]
    long = scores.loc[scores["candidate"].isin(candidates), cols].copy()
    base = long.loc[long["candidate"].eq("AF_independent_pooled"), ["release_run_id", "bucket_log_loss", "realized_bucket_probability"]].rename(
        columns={"bucket_log_loss": "independent_bucket_log_loss", "realized_bucket_probability": "independent_realized_bucket_probability"}
    )
    long = long.merge(base, on="release_run_id", how="left")
    long["bucket_log_loss_delta_vs_independent"] = long["bucket_log_loss"] - long["independent_bucket_log_loss"]
    long["realized_probability_delta_vs_independent"] = long["realized_bucket_probability"] - long["independent_realized_bucket_probability"]
    return long.sort_values(["bucket_log_loss_delta_vs_independent", "bucket_log_loss"], ascending=[False, False])


def _common_weekend_identifiability(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.loc[
        frame["actual_fri_usd"].gt(0) & frame["pre_fri_usd"].gt(0)
        & frame["actual_sat_usd"].gt(0) & frame["actual_sun_usd"].gt(0)
        & frame["after_fri_sat_usd"].gt(0) & frame["after_fri_sun_usd"].gt(0)
    ].copy()
    valid["weekend_key"] = valid.get("friday_date", valid["release_date"])
    valid["friday_surprise"] = np.log(valid["actual_fri_usd"] / valid["pre_fri_usd"])
    valid["sat_log_residual"] = np.log(valid["actual_sat_usd"] / valid["after_fri_sat_usd"])
    valid["sun_log_residual"] = np.log(valid["actual_sun_usd"] / valid["after_fri_sun_usd"])
    peer_rows = []
    for _, row in valid.iterrows():
        peers = valid.loc[valid["weekend_key"].eq(row["weekend_key"]) & valid["release_run_id"].ne(row["release_run_id"])]
        peer_count = int(len(peers))
        median_shock = float(peers["friday_surprise"].median()) if peer_count else math.nan
        aggregate_shock = (
            float(np.log(peers["actual_fri_usd"].sum() / peers["pre_fri_usd"].sum()))
            if peer_count and peers["pre_fri_usd"].sum() > 0
            else math.nan
        )
        peer_rows.append(
            {
                "release_run_id": row["release_run_id"],
                "weekend_key": row["weekend_key"],
                "peer_count": peer_count,
                "median_peer_friday_shock": median_shock,
                "aggregate_peer_friday_shock": aggregate_shock,
                "idiosyncratic_friday_surprise": float(row["friday_surprise"] - median_shock) if math.isfinite(median_shock) else math.nan,
                "sat_log_residual": float(row["sat_log_residual"]),
                "sun_log_residual": float(row["sun_log_residual"]),
            }
        )
    peers = pd.DataFrame(peer_rows)
    weekend_sizes = valid.groupby("weekend_key")["release_run_id"].nunique()
    summary = {
        "unique_weekends": int(weekend_sizes.size),
        "rows": int(len(valid)),
        "movies_per_weekend_mean": float(weekend_sizes.mean()),
        "movies_per_weekend_median": float(weekend_sizes.median()),
        "pct_rows_with_1_peer": float((peers["peer_count"] >= 1).mean()),
        "pct_rows_with_2_peers": float((peers["peer_count"] >= 2).mean()),
        "pct_rows_with_3_peers": float((peers["peer_count"] >= 3).mean()),
        "median_peer_shock_std": float(peers["median_peer_friday_shock"].std(ddof=1)),
        "aggregate_peer_shock_std": float(peers["aggregate_peer_friday_shock"].std(ddof=1)),
        "median_peer_shock_corr_sat": float(peers[["median_peer_friday_shock", "sat_log_residual"]].corr().iloc[0, 1]),
        "median_peer_shock_corr_sun": float(peers[["median_peer_friday_shock", "sun_log_residual"]].corr().iloc[0, 1]),
    }
    weekend_means = valid.groupby("weekend_key")[["sat_log_residual", "sun_log_residual"]].mean()
    for residual_col in ["sat_log_residual", "sun_log_residual"]:
        between = float(weekend_means[residual_col].var(ddof=1))
        total = float(valid[residual_col].var(ddof=1))
        summary[f"{residual_col}_weekend_icc_proxy"] = between / total if total > 0 else math.nan
    return pd.DataFrame([summary])


def _quintile_table(frame: pd.DataFrame, surprise_col: str, residual_cols: list[str]) -> pd.DataFrame:
    work = frame.dropna(subset=[surprise_col] + residual_cols).copy()
    work["quintile"] = pd.qcut(work[surprise_col], 5, labels=False, duplicates="drop") + 1
    rows = []
    for quintile, group in work.groupby("quintile"):
        record: dict[str, object] = {
            "surprise": surprise_col,
            "quintile": int(quintile),
            "n": int(len(group)),
            "surprise_min": float(group[surprise_col].min()),
            "surprise_max": float(group[surprise_col].max()),
        }
        for residual_col in residual_cols:
            residual = group[residual_col]
            prefix = residual_col.replace("_log_residual", "")
            record[f"{prefix}_median"] = float(residual.median())
            record[f"{prefix}_mean_abs"] = float(residual.abs().mean())
            for q in [0.10, 0.20, 0.80, 0.90]:
                record[f"{prefix}_q{int(q*100):02d}"] = float(residual.quantile(q))
        if len(residual_cols) == 2 and len(group) > 1:
            record["sat_sun_corr"] = float(group[residual_cols].corr().iloc[0, 1])
        rows.append(record)
    return pd.DataFrame(rows)


def _load_baseline(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["release_date"])
    if "friday_date" in frame.columns:
        frame["friday_date"] = pd.to_datetime(frame["friday_date"], errors="coerce")
    columns = [
        "actual_fri_usd", "actual_sat_usd", "actual_sun_usd", "actual_ow_usd",
        "pre_fri_usd", "after_fri_sat_usd", "after_fri_sun_usd", "after_sat_sun_usd",
        "movie_id", "release_run_id",
    ]
    return _numeric(frame, columns).sort_values(["release_date", "movie_id"]).reset_index(drop=True)


def _af_draws_for_row(
    row: pd.Series,
    train: pd.DataFrame,
    rng: np.random.Generator,
    n_draws: int,
    *,
    lambda_value: float,
    bandwidth: float,
    shrink_k: float,
) -> tuple[np.ndarray, float | None, float | None]:
    sat_res = train["sat_log_residual"].to_numpy("float64")
    sun_res = train["sun_log_residual"].to_numpy("float64")
    n_prior = len(train)
    weights, n_eff, alpha = _weights(float(row["friday_surprise"]), train["friday_surprise"].to_numpy("float64"), bandwidth, shrink_k)
    use_joint = rng.random(n_draws) < lambda_value
    sat_draw = np.empty(n_draws, dtype="float64")
    sun_draw = np.empty(n_draws, dtype="float64")
    if use_joint.any():
        pair_idx = rng.choice(np.arange(n_prior), int(use_joint.sum()), replace=True, p=weights)
        sat_draw[use_joint] = sat_res[pair_idx]
        sun_draw[use_joint] = sun_res[pair_idx]
    if (~use_joint).any():
        count = int((~use_joint).sum())
        sat_draw[~use_joint] = rng.choice(sat_res, count, replace=True)
        sun_draw[~use_joint] = rng.choice(sun_res, count, replace=True)
    draws = row["actual_fri_usd"] + row["after_fri_sat_usd"] * np.exp(sat_draw) + row["after_fri_sun_usd"] * np.exp(sun_draw)
    return draws, n_eff, alpha


def _select_lambda_for_af_row(
    train: pd.DataFrame,
    *,
    bandwidth: float,
    shrink_k: float,
    min_history: int,
    n_draws: int,
    smoothing_alpha: float,
    seed: int,
) -> float:
    validation_rows = []
    rng = np.random.default_rng(seed)
    for _, validation_row in train.iterrows():
        inner = train.loc[train["release_date"] < validation_row["release_date"]]
        if len(inner) < min_history:
            continue
        for lambda_value in LAMBDA_GRID:
            draws, _, _ = _af_draws_for_row(
                validation_row,
                inner,
                rng,
                n_draws,
                lambda_value=lambda_value,
                bandwidth=bandwidth,
                shrink_k=shrink_k,
            )
            validation_rows.append(
                {
                    "lambda": lambda_value,
                    "bucket_log_loss": _probability_scores(
                        draws,
                        float(validation_row["actual_ow_usd"]),
                        smoothing_alpha=smoothing_alpha,
                    )["bucket_log_loss"],
                }
            )
    if not validation_rows:
        return 0.0
    scores = pd.DataFrame(validation_rows).groupby("lambda")["bucket_log_loss"].mean()
    return float(scores.idxmin())


def evaluate_after_friday(
    frame: pd.DataFrame,
    *,
    bandwidth: float,
    shrink_k: float,
    min_history: int,
    n_draws: int,
    seed: int,
    smoothing_alpha: float,
    lambda_selection_draws: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = frame.loc[
        frame["actual_fri_usd"].gt(0) & frame["actual_sat_usd"].gt(0) & frame["actual_sun_usd"].gt(0)
        & frame["pre_fri_usd"].gt(0) & frame["after_fri_sat_usd"].gt(0) & frame["after_fri_sun_usd"].gt(0)
    ].copy()
    valid["friday_surprise"] = np.log(valid["actual_fri_usd"] / valid["pre_fri_usd"])
    valid["abs_friday_surprise"] = valid["friday_surprise"].abs()
    valid["sat_log_residual"] = np.log(valid["actual_sat_usd"] / valid["after_fri_sat_usd"])
    valid["sun_log_residual"] = np.log(valid["actual_sun_usd"] / valid["after_fri_sun_usd"])
    rng = np.random.default_rng(seed)
    rows = []
    for _, row in valid.iterrows():
        train = valid.loc[valid["release_date"] < row["release_date"]]
        if len(train) < min_history:
            continue
        sat_res = train["sat_log_residual"].to_numpy("float64")
        sun_res = train["sun_log_residual"].to_numpy("float64")
        n_prior = len(train)
        sat_ind = rng.choice(sat_res, n_draws, replace=True)
        sun_ind = rng.choice(sun_res, n_draws, replace=True)
        draws = row["actual_fri_usd"] + row["after_fri_sat_usd"] * np.exp(sat_ind) + row["after_fri_sun_usd"] * np.exp(sun_ind)
        rows.append(_score_draws("AF_independent_pooled", row, draws, n_prior=n_prior, n_eff=None, alpha=None, smoothing_alpha=smoothing_alpha))
        pair_idx = rng.integers(0, n_prior, n_draws)
        draws = row["actual_fri_usd"] + row["after_fri_sat_usd"] * np.exp(sat_res[pair_idx]) + row["after_fri_sun_usd"] * np.exp(sun_res[pair_idx])
        rows.append(_score_draws("AF_joint_pooled", row, draws, n_prior=n_prior, n_eff=None, alpha=None, smoothing_alpha=smoothing_alpha))
        for candidate, surprise_col in [
            ("AF_joint_cond_friday_surprise", "friday_surprise"),
            ("AF_joint_cond_abs_friday_surprise", "abs_friday_surprise"),
        ]:
            weights, n_eff, alpha = _weights(float(row[surprise_col]), train[surprise_col].to_numpy("float64"), bandwidth, shrink_k)
            pair_idx = rng.choice(np.arange(n_prior), n_draws, replace=True, p=weights)
            draws = row["actual_fri_usd"] + row["after_fri_sat_usd"] * np.exp(sat_res[pair_idx]) + row["after_fri_sun_usd"] * np.exp(sun_res[pair_idx])
            rows.append(_score_draws(candidate, row, draws, n_prior=n_prior, n_eff=n_eff, alpha=alpha, smoothing_alpha=smoothing_alpha))
        selected_lambda = _select_lambda_for_af_row(
            train,
            bandwidth=bandwidth,
            shrink_k=shrink_k,
            min_history=min_history,
            n_draws=lambda_selection_draws,
            smoothing_alpha=smoothing_alpha,
            seed=seed + int(row["movie_id"]),
        )
        draws, n_eff, alpha = _af_draws_for_row(
            row,
            train,
            rng,
            n_draws,
            lambda_value=selected_lambda,
            bandwidth=bandwidth,
            shrink_k=shrink_k,
        )
        rows.append(
            _score_draws(
                "AF_mixture_lambda_train_bucket",
                row,
                draws,
                n_prior=n_prior,
                n_eff=n_eff,
                alpha=alpha,
                smoothing_alpha=smoothing_alpha,
                selected_lambda=selected_lambda,
            )
        )
    diagnostic = _quintile_table(valid, "friday_surprise", ["sat_log_residual", "sun_log_residual"])
    diagnostic = pd.concat([diagnostic, _quintile_table(valid, "abs_friday_surprise", ["sat_log_residual", "sun_log_residual"])], ignore_index=True)
    return pd.DataFrame(rows), diagnostic


def evaluate_after_saturday(
    frame: pd.DataFrame,
    *,
    bandwidth: float,
    shrink_k: float,
    min_history: int,
    n_draws: int,
    seed: int,
    smoothing_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = frame.loc[
        frame["actual_fri_usd"].gt(0) & frame["actual_sat_usd"].gt(0) & frame["actual_sun_usd"].gt(0)
        & frame["after_fri_sat_usd"].gt(0) & frame["after_sat_sun_usd"].gt(0)
    ].copy()
    valid["saturday_surprise"] = np.log(valid["actual_sat_usd"] / valid["after_fri_sat_usd"])
    valid["abs_saturday_surprise"] = valid["saturday_surprise"].abs()
    valid["sun_log_residual"] = np.log(valid["actual_sun_usd"] / valid["after_sat_sun_usd"])
    rng = np.random.default_rng(seed)
    rows = []
    for _, row in valid.iterrows():
        train = valid.loc[valid["release_date"] < row["release_date"]]
        if len(train) < min_history:
            continue
        sun_res = train["sun_log_residual"].to_numpy("float64")
        n_prior = len(train)
        for candidate, surprise_col in [
            ("AS_sunday_pooled", None),
            ("AS_sunday_cond_saturday_surprise", "saturday_surprise"),
            ("AS_sunday_cond_abs_saturday_surprise", "abs_saturday_surprise"),
        ]:
            if surprise_col is None:
                idx = rng.integers(0, n_prior, n_draws)
                n_eff = alpha = None
            else:
                weights, n_eff, alpha = _weights(float(row[surprise_col]), train[surprise_col].to_numpy("float64"), bandwidth, shrink_k)
                idx = rng.choice(np.arange(n_prior), n_draws, replace=True, p=weights)
            draws = row["actual_fri_usd"] + row["actual_sat_usd"] + row["after_sat_sun_usd"] * np.exp(sun_res[idx])
            rows.append(_score_draws(candidate, row, draws, n_prior=n_prior, n_eff=n_eff, alpha=alpha, smoothing_alpha=smoothing_alpha))
    diagnostic = _quintile_table(valid, "saturday_surprise", ["sun_log_residual"])
    diagnostic = pd.concat([diagnostic, _quintile_table(valid, "abs_saturday_surprise", ["sun_log_residual"])], ignore_index=True)
    return pd.DataFrame(rows), diagnostic


def run_diagnostic(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    frame = _load_baseline(args.daily_baseline_csv)
    af_scores, af_diag = evaluate_after_friday(
        frame,
        bandwidth=args.bandwidth,
        shrink_k=args.shrink_k,
        min_history=args.min_history,
        n_draws=args.n_draws,
        seed=args.seed,
        smoothing_alpha=args.bucket_smoothing_alpha,
        lambda_selection_draws=args.lambda_selection_draws,
    )
    as_scores, as_diag = evaluate_after_saturday(
        frame,
        bandwidth=args.bandwidth,
        shrink_k=args.shrink_k,
        min_history=args.min_history,
        n_draws=args.n_draws,
        seed=args.seed + 1,
        smoothing_alpha=args.bucket_smoothing_alpha,
    )
    af_candidates = [
        "AF_joint_pooled",
        "AF_joint_cond_friday_surprise",
        "AF_joint_cond_abs_friday_surprise",
        "AF_mixture_lambda_train_bucket",
    ]
    as_candidates = [
        "AS_sunday_cond_saturday_surprise",
        "AS_sunday_cond_abs_saturday_surprise",
    ]
    return {
        "after_friday_scores": af_scores,
        "after_friday_summary": _summarize_scores(af_scores),
        "after_friday_quintiles": af_diag,
        "after_friday_delta_bootstrap_vs_independent": _paired_delta_report(
            af_scores,
            benchmark="AF_independent_pooled",
            candidates=af_candidates,
            seed=args.seed + 10,
            iterations=args.bootstrap_iterations,
        ),
        "after_friday_leave_one_year_deltas_vs_independent": _leave_one_year_delta_report(
            af_scores,
            benchmark="AF_independent_pooled",
            candidates=af_candidates,
        ),
        "after_friday_bucket_audit": _after_friday_bucket_audit(af_scores),
        "after_saturday_scores": as_scores,
        "after_saturday_summary": _summarize_scores(as_scores),
        "after_saturday_quintiles": as_diag,
        "after_saturday_delta_bootstrap_vs_pooled": _paired_delta_report(
            as_scores,
            benchmark="AS_sunday_pooled",
            candidates=as_candidates,
            seed=args.seed + 20,
            iterations=args.bootstrap_iterations,
        ),
        "after_saturday_leave_one_year_deltas_vs_pooled": _leave_one_year_delta_report(
            as_scores,
            benchmark="AS_sunday_pooled",
            candidates=as_candidates,
        ),
        "common_weekend_shock_identifiability": _common_weekend_identifiability(frame),
    }


def _write_outputs(outputs: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-baseline-csv", type=Path, default=Path("data/predictions/daily_regime_baseline_forecasts.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/diagnostics/conditional_residual_bootstrap"))
    parser.add_argument("--min-history", type=int, default=20)
    parser.add_argument("--n-draws", type=int, default=20_000)
    parser.add_argument("--bandwidth", type=float, default=0.35)
    parser.add_argument("--shrink-k", type=float, default=20.0)
    parser.add_argument("--bucket-smoothing-alpha", type=float, default=0.5)
    parser.add_argument("--lambda-selection-draws", type=int, default=5_000)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    outputs = run_diagnostic(args)
    _write_outputs(outputs, args.output_dir)
    print("After Friday summary")
    print(outputs["after_friday_summary"].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nAfter Saturday summary")
    print(outputs["after_saturday_summary"].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nAfter Friday paired delta bootstrap vs independent")
    print(outputs["after_friday_delta_bootstrap_vs_independent"].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nAfter Saturday paired delta bootstrap vs pooled")
    print(outputs["after_saturday_delta_bootstrap_vs_pooled"].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nCommon weekend shock identifiability")
    print(outputs["common_weekend_shock_identifiability"].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nWrote diagnostics to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
