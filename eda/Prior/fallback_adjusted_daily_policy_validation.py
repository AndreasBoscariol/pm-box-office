#!/usr/bin/env python3
"""Fallback-adjusted nested validation for daily pre-release point policies.

This diagnostic consumes the refreshed rolling benchmark CSVs and answers the
remaining point-model freeze questions:

* evaluate daily policy selection with production fallbacks included;
* apply the current production source eligibility stance;
* export an out-of-fold residual panel for interval/distribution work;
* run a first daily V0/V1/V2/V3 interval comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from eda.Prior.rolling_forecast_consensus_benchmark import (
    BenchmarkConfig,
    DEFAULT_INTERVAL_SHRINK_K,
    DEFAULT_MAX_SOURCE_AGE_DAYS,
    DEFAULT_MIN_SOURCE_RELIABILITY_N,
    DEFAULT_MIN_TRAIN_N_FOR_MODEL_SELECTION,
    DEFAULT_ORIGIN_DAYS,
    DEFAULT_PRIMARY_POINT_COL,
    DEFAULT_RECENCY_LAMBDAS,
    DEFAULT_SOURCE_BIAS_SHRINK_K,
    DEFAULT_TEST_START_YEAR,
    DEFAULT_TRAIN_YEARS,
    add_locked_point_forecasts,
    add_range_features_to_consensus_panel,
    build_consensus_panel,
    evaluate_forecast,
    interval_score,
    point_candidate_shortlist,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTICS_DIR = REPO_ROOT / "data" / "diagnostics"
OUTPUT_DIR = DIAGNOSTICS_DIR / "fallback_adjusted_daily_policy"
TODD_ELIGIBLE_ORIGINS = set(range(-9, -2))


@dataclass(frozen=True)
class ForecastResult:
    values: pd.Series
    fallback_level: pd.Series


def production_source_panel(source_panel: pd.DataFrame) -> pd.DataFrame:
    out = source_panel.copy()
    out = out.loc[~out["estimate_source"].isin(["edwarddouglas_substack", "joblo"])].copy()
    out = out.loc[
        ~out["estimate_source"].eq("toddmthatcher") | out["origin_day"].astype(int).isin(TODD_ELIGIBLE_ORIGINS)
    ].copy()
    out = out.loc[
        ~out["estimate_source"].eq("boxofficetheory") | out["origin_day"].astype(int).le(-8)
    ].copy()
    return out


def fallback_chain(method: str, origin_day: int) -> list[tuple[str, str]]:
    if origin_day == -1:
        chain = [
            (method, "selected"),
            ("same_day_bias_adjusted_median_consensus_usd", "same_day_bias_adjusted_median"),
            ("same_day_bias_adjusted_log_mean_consensus_usd", "same_day_bias_adjusted_log_mean"),
            ("max_age_3d_bias_adjusted_median_consensus_usd", "max_age_3d_bias_adjusted_median"),
            ("max_age_3d_bias_adjusted_log_mean_consensus_usd", "max_age_3d_bias_adjusted_log_mean"),
            ("max_age_7d_bias_adjusted_median_consensus_usd", "max_age_7d_bias_adjusted_median"),
            ("max_age_7d_bias_adjusted_log_mean_consensus_usd", "max_age_7d_bias_adjusted_log_mean"),
        ]
    elif origin_day == -9:
        chain = [
            (method, "selected"),
            ("max_age_7d_bias_adjusted_median_consensus_usd", "max_age_7d_bias_adjusted_median"),
            ("max_age_7d_bias_adjusted_log_mean_consensus_usd", "max_age_7d_bias_adjusted_log_mean"),
        ]
    else:
        chain = [
            (method, "selected"),
            ("hierarchical_freshness_bias_adjusted_median_consensus_usd", "hierarchical_bias_adjusted_median"),
            ("hierarchical_freshness_bias_adjusted_log_mean_consensus_usd", "hierarchical_bias_adjusted_log_mean"),
            ("max_age_7d_bias_adjusted_median_consensus_usd", "max_age_7d_bias_adjusted_median"),
            ("max_age_7d_bias_adjusted_log_mean_consensus_usd", "max_age_7d_bias_adjusted_log_mean"),
        ]
    chain.extend(
        [
            ("latest_available_bias_adjusted_median_consensus_usd", "latest_available_bias_adjusted_median"),
            ("latest_available_bias_adjusted_log_mean_consensus_usd", "latest_available_bias_adjusted_log_mean"),
            ("dollar_median_consensus_usd", "dollar_median"),
            (DEFAULT_PRIMARY_POINT_COL, "locked_production_fallback"),
        ]
    )
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for column, level in chain:
        if column not in seen:
            seen.add(column)
            deduped.append((column, level))
    return deduped


def forecast_with_fallback(df: pd.DataFrame, method: str, origin_day: int) -> ForecastResult:
    values = pd.Series(np.nan, index=df.index, dtype="float64")
    levels = pd.Series("missing", index=df.index, dtype="object")
    for column, level in fallback_chain(method, origin_day):
        if column not in df.columns:
            continue
        candidate = pd.to_numeric(df[column], errors="coerce")
        mask = values.isna() & candidate.gt(0)
        values.loc[mask] = candidate.loc[mask]
        levels.loc[mask] = level
    return ForecastResult(values=values, fallback_level=levels)


def evaluate_method_with_fallback(df: pd.DataFrame, method: str, origin_day: int) -> dict[str, float | int | str]:
    temp = df.copy()
    result = forecast_with_fallback(temp, method, origin_day)
    temp["_fallback_forecast_usd"] = result.values
    score = evaluate_forecast(temp, "_fallback_forecast_usd")
    score["forecast_method"] = method
    score["fallback_share"] = float(result.fallback_level.ne("selected").mean()) if len(result.fallback_level) else np.nan
    score["missing_after_fallback"] = int(result.values.isna().sum())
    return score


def candidate_complexity(method: str) -> int:
    score = 0
    if "bias_adjusted" in method or "source_bias_adjusted" in method:
        score += 2
    if "recency" in method:
        score += 2
    if "reliability" in method:
        score += 2
    if "hierarchical" in method:
        score += 1
    if "max_age" in method or "same_day" in method:
        score += 1
    if "lambda_" in method:
        score += 1
    if "median" in method:
        score -= 1
    return score


def build_nested_validation(panel: pd.DataFrame, methods: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    oof_rows: list[pd.DataFrame] = []
    holdout_years = sorted(
        int(year) for year in panel["release_year"].dropna().unique() if int(year) not in (2020, 2021)
    )
    for year in holdout_years:
        for origin_value in sorted(panel["origin_day"].dropna().unique()):
            origin_day = int(origin_value)
            train = panel.loc[(panel["origin_day"].eq(origin_day)) & (panel["release_year"].lt(year))].copy()
            holdout = panel.loc[(panel["origin_day"].eq(origin_day)) & (panel["release_year"].eq(year))].copy()
            if holdout.empty:
                continue

            train_scores = []
            for method in methods:
                score = evaluate_method_with_fallback(train, method, origin_day)
                if score["n"] and score["n"] >= 10 and np.isfinite(score["MAE_log"]):
                    train_scores.append(score)

            if train_scores:
                train_score_frame = pd.DataFrame(train_scores)
                selected = train_score_frame.sort_values(["MAE_log", "RMSE_log", "forecast_method"]).iloc[0]
                selected_method = str(selected["forecast_method"])
                selection_reason = "selected_from_prior_years_with_fallback"
                train_n = int(selected["n"])
                train_mae = float(selected["MAE_log"])
            else:
                selected_method = "dollar_median_consensus_usd"
                selection_reason = "fallback_insufficient_prior_support"
                train_n = 0
                train_mae = np.nan

            holdout_result = forecast_with_fallback(holdout, selected_method, origin_day)
            holdout["_oof_point_forecast_usd"] = holdout_result.values
            holdout["_fallback_level"] = holdout_result.fallback_level
            selected_score = evaluate_forecast(holdout, "_oof_point_forecast_usd")
            baseline_score = evaluate_forecast(holdout, DEFAULT_PRIMARY_POINT_COL)
            rows.append(
                {
                    "holdout_year": year,
                    "origin_day": origin_day,
                    "selected_method": selected_method,
                    "selection_reason": selection_reason,
                    "train_n": train_n,
                    "train_MAE_log": train_mae,
                    "holdout_n": int(selected_score["n"]),
                    "holdout_MAE_log": selected_score["MAE_log"],
                    "holdout_RMSE_log": selected_score["RMSE_log"],
                    "holdout_MdAPE": selected_score["MdAPE"],
                    "holdout_ME_log": selected_score["ME_log"],
                    "holdout_underprediction_rate": selected_score["underprediction_rate"],
                    "baseline_MAE_log": baseline_score["MAE_log"],
                    "baseline_RMSE_log": baseline_score["RMSE_log"],
                    "baseline_MdAPE": baseline_score["MdAPE"],
                    "delta_MAE_vs_locked": selected_score["MAE_log"] - baseline_score["MAE_log"],
                    "missing_after_fallback": int(holdout_result.values.isna().sum()),
                }
            )
            actual = pd.to_numeric(holdout["actual_opening_weekend_gross_usd"], errors="coerce")
            point = pd.to_numeric(holdout["_oof_point_forecast_usd"], errors="coerce")
            residual = np.log(actual / point)
            oof = holdout.copy()
            oof["holdout_year"] = year
            oof["selected_policy"] = selected_method
            oof["fallback_level"] = holdout_result.fallback_level
            oof["oof_point_forecast_usd"] = point
            oof["signed_log_error"] = residual
            oof["absolute_log_error"] = residual.abs()
            oof_rows.append(oof)
    return pd.DataFrame(rows), pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame()


def summarize_nested(nested: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for origin_day, group in nested.groupby("origin_day"):
        weights = pd.to_numeric(group["holdout_n"], errors="coerce").fillna(0).to_numpy(dtype=float)
        if weights.sum() == 0:
            continue
        rows.append(
            {
                "origin_day": int(origin_day),
                "folds": int(len(group)),
                "rows": int(weights.sum()),
                "outer_MAE_log_weighted": float(np.average(group["holdout_MAE_log"], weights=weights)),
                "locked_MAE_log_weighted": float(np.average(group["baseline_MAE_log"], weights=weights)),
                "delta_MAE_vs_locked_weighted": float(np.average(group["delta_MAE_vs_locked"], weights=weights)),
                "median_fold_delta_MAE_vs_locked": float(group["delta_MAE_vs_locked"].median()),
                "missing_after_fallback": int(group["missing_after_fallback"].sum()),
            }
        )
    return pd.DataFrame(rows)


def simplify_policy(panel: pd.DataFrame, methods: list[str], tolerance: float = 0.005) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for origin_value in sorted(panel["origin_day"].dropna().unique()):
        origin_day = int(origin_value)
        origin = panel.loc[panel["origin_day"].eq(origin_day)].copy()
        scores = []
        for method in methods:
            score = evaluate_method_with_fallback(origin, method, origin_day)
            if score["n"] and np.isfinite(score["MAE_log"]):
                score["complexity"] = candidate_complexity(method)
                scores.append(score)
        if not scores:
            continue
        score_frame = pd.DataFrame(scores)
        best_mae = float(score_frame["MAE_log"].min())
        eligible = score_frame.loc[score_frame["MAE_log"].le(best_mae + tolerance)].copy()
        selected = eligible.sort_values(["complexity", "MAE_log", "RMSE_log", "forecast_method"]).iloc[0]
        rows.append(
            {
                "origin_day": origin_day,
                "simplified_method": selected["forecast_method"],
                "best_MAE_log": best_mae,
                "simplified_MAE_log": float(selected["MAE_log"]),
                "simplified_minus_best_MAE_log": float(selected["MAE_log"] - best_mae),
                "complexity": int(selected["complexity"]),
                "n": int(selected["n"]),
                "fallback_share": selected.get("fallback_share", np.nan),
            }
        )
    return pd.DataFrame(rows)


def add_source_metadata(oof: pd.DataFrame, source_panel: pd.DataFrame) -> pd.DataFrame:
    source_panel = source_panel.copy()
    source_panel["is_todd"] = source_panel["estimate_source"].eq("toddmthatcher")
    source_panel["is_theory"] = source_panel["estimate_source"].eq("boxofficetheory")
    grouped = (
        source_panel.groupby(["release_run_id", "origin_day"])
        .agg(
            eligible_source_count=("estimate_source", "nunique"),
            eligible_sources=("estimate_source", lambda s: ", ".join(sorted(s.dropna().unique()))),
            min_source_age_days=("source_age_days", "min"),
            max_source_age_days=("source_age_days", "max"),
            todd_included=("is_todd", "max"),
            boxofficetheory_eras=("source_era", lambda s: ", ".join(sorted(set(s.dropna().astype(str))))),
        )
        .reset_index()
    )
    out = oof.merge(grouped, on=["release_run_id", "origin_day"], how="left", suffixes=("", "_source_meta"))
    return out


def residual_diagnostics(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for origin_day, group in oof.groupby("origin_day"):
        residual = pd.to_numeric(group["signed_log_error"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if residual.empty:
            continue
        rows.append(
            {
                "origin_day": int(origin_day),
                "n": int(len(residual)),
                "mean_signed_log_error": float(residual.mean()),
                "median_signed_log_error": float(residual.median()),
                "MAE_log": float(residual.abs().mean()),
                "RMSE_log": float(np.sqrt(np.mean(residual**2))),
                "underprediction_rate": float((residual > 0).mean()),
                "q025": float(residual.quantile(0.025)),
                "q10": float(residual.quantile(0.10)),
                "q20": float(residual.quantile(0.20)),
                "q80": float(residual.quantile(0.80)),
                "q90": float(residual.quantile(0.90)),
                "q975": float(residual.quantile(0.975)),
                "lower_tail_abs_q025": float(abs(residual.quantile(0.025))),
                "upper_tail_q975": float(residual.quantile(0.975)),
            }
        )
    return pd.DataFrame(rows)


def bucket(values: pd.Series, labels: tuple[str, str]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    median = numeric.median()
    if not np.isfinite(median):
        return pd.Series("unknown", index=values.index)
    return pd.Series(np.where(numeric.gt(median), labels[1], labels[0]), index=values.index).where(
        numeric.notna(), "unknown"
    )


def interval_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    frame = oof.copy()
    frame["dispersion_bucket"] = bucket(frame["cross_source_log_disagreement"], ("dispersion_low", "dispersion_high"))
    frame["range_bucket"] = bucket(
        frame["aggregate_relative_internal_log_half_width"], ("range_narrow", "range_wide")
    )
    candidate_groups = {
        "V0_origin_only": ["origin_day"],
        "V1_dispersion": ["origin_day", "dispersion_bucket"],
        "V2_range_width": ["origin_day", "range_bucket"],
        "V3_dispersion_plus_range": ["origin_day", "dispersion_bucket", "range_bucket"],
    }
    for candidate, group_cols in candidate_groups.items():
        for origin_day, origin_frame in frame.groupby("origin_day"):
            train = frame.loc[frame["origin_day"].eq(origin_day)].copy()
            if train.empty:
                continue
            scored = origin_frame.copy()
            for level, lo_q, hi_q, alpha in [(80, 0.10, 0.90, 0.20), (95, 0.025, 0.975, 0.05)]:
                global_lo = train["signed_log_error"].quantile(lo_q)
                global_hi = train["signed_log_error"].quantile(hi_q)
                q = (
                    train.groupby(group_cols, dropna=False)["signed_log_error"]
                    .agg(
                        n="count",
                        lo=lambda s, q=lo_q: s.quantile(q),
                        hi=lambda s, q=hi_q: s.quantile(q),
                    )
                    .reset_index()
                )
                temp = scored[group_cols].merge(q, on=group_cols, how="left")
                lo = temp["lo"].fillna(global_lo).to_numpy(dtype=float)
                hi = temp["hi"].fillna(global_hi).to_numpy(dtype=float)
                point = pd.to_numeric(scored["oof_point_forecast_usd"], errors="coerce").to_numpy(dtype=float)
                actual = pd.to_numeric(scored["actual_opening_weekend_gross_usd"], errors="coerce")
                lo_usd = point * np.exp(lo)
                hi_usd = point * np.exp(hi)
                width = (hi_usd - lo_usd) / point
                rows.append(
                    {
                        "origin_day": int(origin_day),
                        "candidate": candidate,
                        "level": level,
                        "n": int(len(scored)),
                        "coverage": float(((actual >= lo_usd) & (actual <= hi_usd)).mean()),
                        "lower_miss_rate": float((actual < lo_usd).mean()),
                        "upper_miss_rate": float((actual > hi_usd).mean()),
                        "median_width_pct": float(np.nanmedian(width)),
                        "mean_interval_score_pct": float(
                            np.nanmean(
                                (
                                    (hi_usd - lo_usd)
                                    + (2.0 / alpha) * np.clip(lo_usd - actual.to_numpy(dtype=float), 0, None)
                                    + (2.0 / alpha) * np.clip(actual.to_numpy(dtype=float) - hi_usd, 0, None)
                                )
                                / point
                            )
                        ),
                    }
                )
    return pd.DataFrame(rows)


def interval_bounds_from_train(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    candidate: str,
    level: int,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    alpha = 1.0 - level / 100.0
    lo_q = alpha / 2.0
    hi_q = 1.0 - alpha / 2.0
    origin_lo = float(train["signed_log_error"].quantile(lo_q))
    origin_hi = float(train["signed_log_error"].quantile(hi_q))
    lo = pd.Series(origin_lo, index=holdout.index, dtype="float64")
    hi = pd.Series(origin_hi, index=holdout.index, dtype="float64")
    fallback = pd.Series("V0_origin", index=holdout.index, dtype="object")

    def apply_grouped(group_cols: list[str], label: str, mask: pd.Series | None = None) -> None:
        nonlocal lo, hi, fallback
        stats = (
            train.groupby(group_cols, dropna=False)["signed_log_error"]
            .agg(
                n="count",
                lo=lambda s: s.quantile(lo_q),
                hi=lambda s: s.quantile(hi_q),
            )
            .reset_index()
        )
        joined = holdout[group_cols].merge(stats, on=group_cols, how="left")
        use = joined["n"].fillna(0).ge(10)
        if mask is not None:
            use &= mask.reset_index(drop=True)
        use.index = holdout.index
        lo.loc[use] = joined.loc[use.to_numpy(), "lo"].to_numpy(dtype=float)
        hi.loc[use] = joined.loc[use.to_numpy(), "hi"].to_numpy(dtype=float)
        fallback.loc[use] = label

    has_dispersion = holdout["dispersion_bucket"].ne("unknown")
    has_range = holdout["range_bucket"].ne("unknown")
    if candidate == "V1_dispersion":
        apply_grouped(["origin_day", "dispersion_bucket"], "V1_dispersion", has_dispersion)
    elif candidate == "V2_range_width":
        apply_grouped(["origin_day", "range_bucket"], "V2_range_width", has_range)
    elif candidate == "V3_dispersion_plus_range":
        apply_grouped(["origin_day", "dispersion_bucket"], "V1_dispersion", has_dispersion & ~has_range)
        apply_grouped(["origin_day", "range_bucket"], "V2_range_width", has_range & ~has_dispersion)
        apply_grouped(
            ["origin_day", "dispersion_bucket", "range_bucket"],
            "V3_dispersion_plus_range",
            has_dispersion & has_range,
        )
    point = pd.to_numeric(holdout["oof_point_forecast_usd"], errors="coerce").to_numpy(dtype=float)
    return point * np.exp(lo.to_numpy(dtype=float)), point * np.exp(hi.to_numpy(dtype=float)), fallback


def score_interval(
    holdout: pd.DataFrame,
    lo_usd: np.ndarray,
    hi_usd: np.ndarray,
    level: int,
) -> dict[str, float | int]:
    actual = pd.to_numeric(holdout["actual_opening_weekend_gross_usd"], errors="coerce").to_numpy(dtype=float)
    point = pd.to_numeric(holdout["oof_point_forecast_usd"], errors="coerce").to_numpy(dtype=float)
    alpha = 1.0 - level / 100.0
    score = (
        (hi_usd - lo_usd)
        + (2.0 / alpha) * np.clip(lo_usd - actual, 0, None)
        + (2.0 / alpha) * np.clip(actual - hi_usd, 0, None)
    ) / point
    return {
        "n": int(len(holdout)),
        "coverage": float(np.nanmean((actual >= lo_usd) & (actual <= hi_usd))),
        "lower_miss_rate": float(np.nanmean(actual < lo_usd)),
        "upper_miss_rate": float(np.nanmean(actual > hi_usd)),
        "mean_width_pct": float(np.nanmean((hi_usd - lo_usd) / point)),
        "median_width_pct": float(np.nanmedian((hi_usd - lo_usd) / point)),
        "mean_interval_score_pct": float(np.nanmean(score)),
    }


def prepare_interval_features(oof: pd.DataFrame) -> pd.DataFrame:
    out = oof.copy()
    out["dispersion_bucket"] = bucket(out["cross_source_log_disagreement"], ("dispersion_low", "dispersion_high"))
    range_value = pd.to_numeric(out["aggregate_relative_internal_log_half_width"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    out["range_bucket"] = "unknown"
    available = range_value.notna()
    if available.any():
        labels = ["range_narrow", "range_medium", "range_wide"]
        out.loc[available, "range_bucket"] = pd.qcut(
            range_value.loc[available],
            q=3,
            labels=labels,
            duplicates="drop",
        ).astype(str)
    return out


def nested_interval_validation(oof: pd.DataFrame, tolerance: float = 0.02) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = prepare_interval_features(oof)
    candidates = ["V0_origin_only", "V1_dispersion", "V2_range_width", "V3_dispersion_plus_range"]
    candidate_complexity_map = {
        "V0_origin_only": 0,
        "V1_dispersion": 1,
        "V2_range_width": 1,
        "V3_dispersion_plus_range": 2,
    }
    rows: list[dict[str, float | int | str]] = []
    selected_rows: list[dict[str, float | int | str]] = []
    years = sorted(int(year) for year in frame["holdout_year"].dropna().unique())
    for year in years:
        for origin_day in sorted(int(value) for value in frame["origin_day"].dropna().unique()):
            train = frame.loc[(frame["origin_day"].eq(origin_day)) & (frame["holdout_year"].lt(year))].copy()
            holdout = frame.loc[(frame["origin_day"].eq(origin_day)) & (frame["holdout_year"].eq(year))].copy()
            if len(train) < 20 or holdout.empty:
                continue
            candidate_scores = []
            for candidate in candidates:
                level_scores = {}
                fallback_counts = None
                for level in (80, 95):
                    lo, hi, fallback = interval_bounds_from_train(train, holdout, candidate, level)
                    score = score_interval(holdout, lo, hi, level)
                    level_scores[level] = score
                    if level == 95:
                        fallback_counts = fallback.value_counts(normalize=True).to_dict()
                    rows.append(
                        {
                            "holdout_year": year,
                            "origin_day": origin_day,
                            "candidate": candidate,
                            "level": level,
                            **score,
                            "fallback_mix": json.dumps(fallback.value_counts().to_dict(), sort_keys=True),
                        }
                    )
                candidate_scores.append(
                    {
                        "candidate": candidate,
                        "score95": level_scores[95]["mean_interval_score_pct"],
                        "score80": level_scores[80]["mean_interval_score_pct"],
                        "coverage95": level_scores[95]["coverage"],
                        "coverage80": level_scores[80]["coverage"],
                        "complexity": candidate_complexity_map[candidate],
                        "fallback_counts": fallback_counts or {},
                    }
                )
            score_frame = pd.DataFrame(candidate_scores)
            best_score = float(score_frame["score95"].min())
            eligible = score_frame.loc[score_frame["score95"].le(best_score + tolerance)].copy()
            selected = eligible.sort_values(["complexity", "score95", "candidate"]).iloc[0]
            selected_rows.append(
                {
                    "holdout_year": year,
                    "origin_day": origin_day,
                    "selected_candidate": selected["candidate"],
                    "best_score95": best_score,
                    "selected_score95": selected["score95"],
                    "selected_minus_best_score95": selected["score95"] - best_score,
                    "selected_coverage95": selected["coverage95"],
                    "selected_coverage80": selected["coverage80"],
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(selected_rows), frame


def summarize_nested_intervals(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        scores.groupby(["origin_day", "candidate", "level"])
        .agg(
            folds=("holdout_year", "nunique"),
            rows=("n", "sum"),
            coverage=("coverage", "median"),
            lower_miss_rate=("lower_miss_rate", "median"),
            upper_miss_rate=("upper_miss_rate", "median"),
            mean_width_pct=("mean_width_pct", "median"),
            median_width_pct=("median_width_pct", "median"),
            mean_interval_score_pct=("mean_interval_score_pct", "median"),
        )
        .reset_index()
    )
    selection = (
        scores.loc[scores["level"].eq(95)]
        .pivot_table(
            index=["holdout_year", "origin_day"],
            columns="candidate",
            values="mean_interval_score_pct",
            aggfunc="first",
        )
        .reset_index()
    )
    ablation_rows = []
    for _, row in selection.iterrows():
        base = {
            "holdout_year": row["holdout_year"],
            "origin_day": row["origin_day"],
            "V2_minus_V0_score95": row.get("V2_range_width", np.nan) - row.get("V0_origin_only", np.nan),
            "V3_minus_V2_score95": row.get("V3_dispersion_plus_range", np.nan) - row.get("V2_range_width", np.nan),
            "V1_minus_V0_score95": row.get("V1_dispersion", np.nan) - row.get("V0_origin_only", np.nan),
        }
        ablation_rows.append(base)
    ablation = pd.DataFrame(ablation_rows)
    ablation_summary = (
        ablation.groupby("origin_day")
        .agg(
            folds=("holdout_year", "nunique"),
            median_V2_minus_V0_score95=("V2_minus_V0_score95", "median"),
            median_V3_minus_V2_score95=("V3_minus_V2_score95", "median"),
            median_V1_minus_V0_score95=("V1_minus_V0_score95", "median"),
        )
        .reset_index()
    )
    tail_balance = (
        scores.groupby(["origin_day", "candidate", "level"])
        .agg(
            median_lower_miss=("lower_miss_rate", "median"),
            median_upper_miss=("upper_miss_rate", "median"),
        )
        .reset_index()
    )
    tail_balance["tail_miss_imbalance"] = (
        tail_balance["median_lower_miss"] - tail_balance["median_upper_miss"]
    ).abs()
    return summary, ablation, ablation_summary, tail_balance


def feature_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for origin_day, group in frame.groupby("origin_day"):
        range_available = pd.to_numeric(
            group["aggregate_relative_internal_log_half_width"], errors="coerce"
        ).notna()
        dispersion_available = pd.to_numeric(group["cross_source_log_disagreement"], errors="coerce").notna()
        rows.append(
            {
                "origin_day": int(origin_day),
                "rows": int(len(group)),
                "range_available_rate": float(range_available.mean()),
                "dispersion_available_rate": float(dispersion_available.mean()),
                "multi_source_rate": float(pd.to_numeric(group["eligible_source_count"], errors="coerce").ge(2).mean()),
                "fallback_levels": json.dumps(group["fallback_level"].value_counts().to_dict(), sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def range_resolution_bins(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(["origin_day", "range_bucket"], dropna=False):
        origin_day, range_bucket = keys
        residual = pd.to_numeric(group["signed_log_error"], errors="coerce")
        rows.append(
            {
                "origin_day": int(origin_day),
                "range_bucket": str(range_bucket),
                "rows": int(len(group)),
                "MAE_log": float(residual.abs().mean()),
                "RMSE_log": float(np.sqrt(np.mean(residual**2))),
                "q025": float(residual.quantile(0.025)),
                "q10": float(residual.quantile(0.10)),
                "q90": float(residual.quantile(0.90)),
                "q975": float(residual.quantile(0.975)),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(DIAGNOSTICS_DIR / "rolling_forecast_origin_source_estimates.csv", low_memory=False)
    for col in ["opening_weekend_start", "forecast_origin_date", "estimate_date"]:
        if col in source.columns:
            source[col] = pd.to_datetime(source[col], errors="coerce")

    config = BenchmarkConfig(
        origin_days=DEFAULT_ORIGIN_DAYS,
        excluded_estimate_sources=(),
        excluded_release_years=(2020, 2021),
        recency_lambdas=DEFAULT_RECENCY_LAMBDAS,
        train_years=DEFAULT_TRAIN_YEARS,
        test_start_year=DEFAULT_TEST_START_YEAR,
        min_source_reliability_n=DEFAULT_MIN_SOURCE_RELIABILITY_N,
        source_bias_shrink_k=DEFAULT_SOURCE_BIAS_SHRINK_K,
        max_source_age_days=DEFAULT_MAX_SOURCE_AGE_DAYS,
        min_train_n_for_model_selection=DEFAULT_MIN_TRAIN_N_FOR_MODEL_SELECTION,
        interval_shrink_k=DEFAULT_INTERVAL_SHRINK_K,
    )
    production_source = production_source_panel(source)
    panel = build_consensus_panel(production_source, config, include_extended_candidates=True)
    panel = add_locked_point_forecasts(panel)
    panel = add_range_features_to_consensus_panel(panel, production_source)
    methods = [method for method in point_candidate_shortlist(panel) if method in panel.columns]

    nested, oof = build_nested_validation(panel, methods)
    summary = summarize_nested(nested)
    selection_frequency = (
        nested.groupby(["origin_day", "selected_method"])
        .size()
        .rename("selected_folds")
        .reset_index()
        .sort_values(["origin_day", "selected_folds", "selected_method"], ascending=[True, False, True])
    )
    simplified = simplify_policy(panel, methods)
    oof = add_source_metadata(oof, production_source)
    oof_export_cols = [
        "movie_id",
        "title",
        "opening_weekend_start",
        "holdout_year",
        "origin_day",
        "actual_opening_weekend_gross_usd",
        "oof_point_forecast_usd",
        "selected_policy",
        "fallback_level",
        "signed_log_error",
        "absolute_log_error",
        "eligible_source_count",
        "eligible_sources",
        "min_source_age_days",
        "max_source_age_days",
        "todd_included",
        "boxofficetheory_eras",
        "source_count",
        "estimate_sources",
        "cross_source_log_disagreement",
        "aggregate_internal_log_half_width",
        "aggregate_relative_internal_log_half_width",
        "aggregate_range_asymmetry_log",
    ]
    existing_oof_cols = [col for col in oof_export_cols if col in oof.columns]
    residual = residual_diagnostics(oof)
    intervals = interval_metrics(oof)
    nested_interval_scores, nested_interval_selected, interval_feature_frame = nested_interval_validation(oof)
    (
        nested_interval_summary,
        nested_interval_ablation,
        nested_interval_ablation_summary,
        nested_interval_tail_balance,
    ) = summarize_nested_intervals(nested_interval_scores)
    nested_interval_selection_frequency = (
        nested_interval_selected.groupby(["origin_day", "selected_candidate"])
        .size()
        .rename("selected_folds")
        .reset_index()
        .sort_values(["origin_day", "selected_folds", "selected_candidate"], ascending=[True, False, True])
    )
    interval_feature_coverage = feature_coverage(interval_feature_frame)
    interval_range_resolution = range_resolution_bins(interval_feature_frame)

    frozen_policy = {
        "policy_name": "fallback_adjusted_daily_point_policy_candidate",
        "status": "frozen_for_interval_development",
        "simplification_tolerance_mae_log": 0.005,
        "eligible_sources": {
            "boxofficereport": "core_when_available",
            "boxofficepro": "core_when_available",
            "toddmthatcher": "eligible_origins_-9_to_-3_only",
            "boxofficetheory": "canonical_one_vote_early_or_fallback_only",
            "joblo": "excluded_shadow_diagnostic",
            "edwarddouglas_substack": "excluded_pending_marginal_value",
        },
        "todd_eligible_origins": sorted(TODD_ELIGIBLE_ORIGINS),
        "boxofficetheory_latest_unrestricted_origin_cutoff": -8,
        "freshness_fallback_hierarchy": [
            "selected_candidate",
            "same_day_when_late_origin",
            "max_age_3d_when_late_origin",
            "max_age_7d",
            "hierarchical_freshness",
            "latest_available",
            "locked_production_fallback",
        ],
        "bias_fallback_hierarchy": [
            "source_x_exact_origin_prior_bias",
            "source_pooled_prior_bias",
            "exact_origin_pooled_prior_bias",
            "zero_correction",
        ],
        "training_minimums": {
            "candidate_selection_min_prior_rows": 10,
            "interval_cell_min_prior_rows": 10,
        },
        "daily_policy": simplified.sort_values("origin_day").to_dict(orient="records"),
    }

    nested.to_csv(OUTPUT_DIR / "nested_daily_policy_year_folds.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "nested_daily_policy_summary_by_origin.csv", index=False)
    selection_frequency.to_csv(OUTPUT_DIR / "nested_daily_policy_selection_frequency.csv", index=False)
    simplified.to_csv(OUTPUT_DIR / "simplified_daily_policy_candidates.csv", index=False)
    oof[existing_oof_cols].to_csv(OUTPUT_DIR / "locked_oof_daily_point_residual_panel.csv", index=False)
    residual.to_csv(OUTPUT_DIR / "selected_point_residual_diagnostics_by_origin.csv", index=False)
    intervals.to_csv(OUTPUT_DIR / "daily_interval_v0_v3_comparison.csv", index=False)
    (OUTPUT_DIR / "frozen_simplified_daily_point_policy.json").write_text(
        json.dumps(frozen_policy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    nested_interval_scores.to_csv(OUTPUT_DIR / "nested_daily_interval_year_folds.csv", index=False)
    nested_interval_summary.to_csv(OUTPUT_DIR / "nested_daily_interval_summary_by_origin.csv", index=False)
    nested_interval_selection_frequency.to_csv(
        OUTPUT_DIR / "nested_daily_interval_selection_frequency.csv",
        index=False,
    )
    nested_interval_ablation.to_csv(OUTPUT_DIR / "nested_daily_interval_feature_ablation.csv", index=False)
    nested_interval_ablation_summary.to_csv(
        OUTPUT_DIR / "nested_daily_interval_feature_ablation_summary.csv",
        index=False,
    )
    nested_interval_tail_balance.to_csv(OUTPUT_DIR / "nested_daily_interval_tail_balance.csv", index=False)
    interval_feature_coverage.to_csv(OUTPUT_DIR / "nested_daily_interval_feature_coverage.csv", index=False)
    interval_range_resolution.to_csv(OUTPUT_DIR / "nested_daily_range_resolution_bins.csv", index=False)

    print(f"Wrote {OUTPUT_DIR}")
    print(f"Nested rows: {len(nested):,}")
    print(f"OOF residual rows: {len(oof):,}")
    print(f"Interval score rows: {len(intervals):,}")
    print(f"Nested interval score rows: {len(nested_interval_scores):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
