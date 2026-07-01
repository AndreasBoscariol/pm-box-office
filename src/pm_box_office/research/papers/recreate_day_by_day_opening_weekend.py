#!/usr/bin/env python3
"""Day-by-day opening-weekend forecasts anchored on Boxoffice Pro estimates.

This is the sequential version of the opening-weekend research workflow.  The
first pass builds leak-free pre-release snapshots from t=-14 through t=-1,
predicting the residual around the latest available Boxoffice Pro midpoint and
calibrating prediction intervals from training residuals only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from pm_box_office.db.connection import connect_database, database_url_from_env
from pm_box_office.research.papers import recreate_competition_opening_weekend as opening
from pm_box_office.research.papers.recreate_wikipedia_boxoffice import (
    format_number,
    mean,
    r2_score,
    write_csv,
)


DEFAULT_OUT_DIR = Path("results/papers/day_by_day_opening_weekend")
DEFAULT_SNAPSHOT_DAYS = tuple(range(-14, 0))
DEFAULT_TRAIN_START_YEAR = 2022
DEFAULT_TRAIN_END_YEAR = 2024
DEFAULT_TEST_START_YEAR = 2025
DEFAULT_TEST_END_YEAR = 2026
DEFAULT_MIN_OPENING_DAY_GROSS = 5_000_000
MIN_BUCKET_INTERVAL_RESIDUALS = 5
INTERVAL_METHODS = ("empirical_quantile", "conformal_abs", "loo_conformal_abs")


FIXED_BUCKET_TERMS = [
    "bop_estimate_bucket_15_30m",
    "bop_estimate_bucket_30_60m",
    "bop_estimate_bucket_60_100m",
    "bop_estimate_bucket_100m_plus",
]

SNAPSHOT_MODEL_TERMS = {
    "raw_bop_snapshot": [],
    "calibrated_bop_snapshot": ["log1p_bop_forecast_midpoint"],
    "bop_residual_bucket_snapshot": FIXED_BUCKET_TERMS,
    "bop_residual_wiki_snapshot": FIXED_BUCKET_TERMS + ["log1p_V"],
    "bop_residual_competition_snapshot": FIXED_BUCKET_TERMS + ["log1p_competitor_total_gross_lag7"],
    "bop_residual_wiki_competition_snapshot": FIXED_BUCKET_TERMS
    + [
        "log1p_V",
        "log1p_competitor_total_gross_lag7",
        "bop_forecast_range_width_pct",
    ],
}

FALLBACK_MODEL_NAME = "fallback_wiki_competition_snapshot"
FALLBACK_TERMS = opening.FALLBACK_MODEL_TERMS


def estimate_bucket_label(row: dict[str, object]) -> str:
    if float(row.get("bop_forecast_available", 0.0) or 0.0) <= 0.0:
        return "missing_bop"
    if float(row.get("bop_estimate_bucket_under_15m", 0.0) or 0.0) > 0.0:
        return "under_15m"
    if float(row.get("bop_estimate_bucket_15_30m", 0.0) or 0.0) > 0.0:
        return "15_30m"
    if float(row.get("bop_estimate_bucket_30_60m", 0.0) or 0.0) > 0.0:
        return "30_60m"
    if float(row.get("bop_estimate_bucket_60_100m", 0.0) or 0.0) > 0.0:
        return "60_100m"
    if float(row.get("bop_estimate_bucket_100m_plus", 0.0) or 0.0) > 0.0:
        return "100m_plus"
    return "unknown"


def build_day_by_day_feature_panel(
    movies: list[opening.OpeningWeekendMovie],
    daily_grosses: list[opening.DailyGross],
    wiki_by_movie: dict[int, dict[int, dict[str, float]]],
    bop_forecasts: list[opening.BoxofficeProForecast],
    *,
    snapshot_days: list[int],
    train_start_year: int,
    train_end_year: int,
) -> list[dict[str, object]]:
    gross_by_movie_date = {(row.movie_id, row.box_office_date): row for row in daily_grosses}
    forecasts_by_movie: dict[int, list[opening.BoxofficeProForecast]] = defaultdict(list)
    for forecast in bop_forecasts:
        forecasts_by_movie[forecast.movie_id].append(forecast)

    rows: list[dict[str, object]] = []
    for movie in movies:
        for snapshot_day in snapshot_days:
            as_of_date = movie.opening_date + dt.timedelta(days=snapshot_day)
            wiki = opening.wiki_values_as_of(
                wiki_by_movie,
                movie_id=movie.movie_id,
                timing_day=snapshot_day,
            )
            focal_forecast = opening.latest_forecast(
                forecasts_by_movie.get(movie.movie_id, []),
                as_of_date=as_of_date,
                movie_id=movie.movie_id,
                forecast_metric="domestic_opening_weekend",
                target_start_date=movie.opening_date,
            )
            row: dict[str, object] = {
                "movie_id": movie.movie_id,
                "title": movie.title,
                "release_year": movie.release_year,
                "release_run_id": movie.release_run_id,
                "opening_date": movie.opening_date.isoformat(),
                "forecast_stage": "pre_release",
                "snapshot_day": snapshot_day,
                "timing_day": snapshot_day,
                "wiki_timing_day": snapshot_day,
                "competition_timing_day": snapshot_day,
                "bop_timing_day": snapshot_day,
                "as_of_date": as_of_date.isoformat(),
                "opening_theaters": movie.opening_theaters,
                "opening_day_gross_usd": movie.opening_day_gross_usd,
                "opening_weekend_revenue_usd": movie.opening_weekend_revenue_usd,
                "target_log_opening_weekend": math.log(max(1.0, float(movie.opening_weekend_revenue_usd))),
                "log1p_opening_theaters": opening.log1p(movie.opening_theaters),
                "wiki_available": 1.0 if any(wiki.values()) else 0.0,
                "V": wiki["V"],
                "U": wiki["U"],
                "R": wiki["R"],
                "E": wiki["E"],
                "log1p_V": opening.log1p(wiki["V"]),
                "log1p_U": opening.log1p(wiki["U"]),
                "log1p_R": opening.log1p(wiki["R"]),
                "log1p_E": opening.log1p(wiki["E"]),
                **opening.actual_competition_features(
                    gross_by_movie_date,
                    focal_movie_id=movie.movie_id,
                    as_of_date=as_of_date,
                ),
                **opening.bop_forecast_features(focal_forecast),
            }
            rows.append(row)

    assigned = opening.assign_bop_q4_proxy(
        rows,
        train_start_year=train_start_year,
        train_end_year=train_end_year,
    )
    out_rows = []
    for row in assigned:
        out = dict(row)
        out["bop_estimate_bucket"] = estimate_bucket_label(out)
        midpoint = float(out.get("bop_forecast_midpoint", 0.0) or 0.0)
        out["target_log_bop_residual"] = (
            float(out["target_log_opening_weekend"]) - math.log(midpoint)
            if midpoint > 0.0 and float(out.get("bop_forecast_available", 0.0) or 0.0) > 0.0
            else ""
        )
        out_rows.append(out)
    return out_rows


def train_holdout_rows(
    rows: list[dict[str, object]],
    *,
    snapshot_day: int,
    train_start_year: int,
    train_end_year: int,
    test_start_year: int,
    test_end_year: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    scoped = [row for row in rows if int(row["snapshot_day"]) == snapshot_day]
    train_rows = [row for row in scoped if train_start_year <= int(row["release_year"]) <= train_end_year]
    holdout_rows = [row for row in scoped if test_start_year <= int(row["release_year"]) <= test_end_year]
    return train_rows, holdout_rows


def bop_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if float(row.get("bop_forecast_available", 0.0) or 0.0) > 0.0]


def rows_with_target(rows: list[dict[str, object]], target_key: str) -> list[dict[str, object]]:
    return [row for row in rows if row.get(target_key, "") != ""]


def prediction_log_for_model(
    *,
    model_name: str,
    terms: list[str],
    train_rows: list[dict[str, object]],
    predict_rows: list[dict[str, object]],
) -> tuple[list[float], opening.FittedModel | None]:
    if model_name == "raw_bop_snapshot":
        return [math.log(max(1.0, float(row["bop_forecast_midpoint"]))) for row in predict_rows], None
    if model_name == "calibrated_bop_snapshot":
        fitted = opening.fit_model_for_target(train_rows, terms, "target_log_opening_weekend")
        return opening.predict_log(fitted, predict_rows), fitted
    fitted = opening.fit_model_for_target(train_rows, terms, "target_log_bop_residual")
    residuals = opening.predict_log(fitted, predict_rows)
    return [
        math.log(max(1.0, float(row["bop_forecast_midpoint"]))) + residual
        for row, residual in zip(predict_rows, residuals)
    ], fitted


def prediction_residuals_for_interval(
    *,
    model_name: str,
    pred_log: list[float],
    rows: list[dict[str, object]],
) -> list[float]:
    return [
        float(row["target_log_opening_weekend"]) - value
        for row, value in zip(rows, pred_log)
        if row.get("target_log_opening_weekend") != ""
    ]


def loo_prediction_residuals_for_model(
    *,
    model_name: str,
    terms: list[str],
    rows: list[dict[str, object]],
) -> list[float]:
    if model_name == "raw_bop_snapshot":
        pred_log = [math.log(max(1.0, float(row["bop_forecast_midpoint"]))) for row in rows]
        return prediction_residuals_for_interval(model_name=model_name, pred_log=pred_log, rows=rows)
    if len(rows) < len(terms) + 3:
        pred_log, _ = prediction_log_for_model(
            model_name=model_name,
            terms=terms,
            train_rows=rows,
            predict_rows=rows,
        )
        return prediction_residuals_for_interval(model_name=model_name, pred_log=pred_log, rows=rows)
    residuals: list[float] = []
    for idx, row in enumerate(rows):
        train_subset = rows[:idx] + rows[idx + 1 :]
        pred_log, _ = prediction_log_for_model(
            model_name=model_name,
            terms=terms,
            train_rows=train_subset,
            predict_rows=[row],
        )
        residuals.append(float(row["target_log_opening_weekend"]) - pred_log[0])
    return residuals


def loo_prediction_residuals_for_opening_model(
    *,
    rows: list[dict[str, object]],
    terms: list[str],
) -> list[float]:
    if len(rows) < len(terms) + 3:
        fitted = opening.fit_model_for_target(rows, terms, "target_log_opening_weekend")
        pred_log = opening.predict_log(fitted, rows)
        return prediction_residuals_for_interval(
            model_name=FALLBACK_MODEL_NAME,
            pred_log=pred_log,
            rows=rows,
        )
    residuals: list[float] = []
    for idx, row in enumerate(rows):
        train_subset = rows[:idx] + rows[idx + 1 :]
        fitted = opening.fit_model_for_target(train_subset, terms, "target_log_opening_weekend")
        pred_log = opening.predict_log(fitted, [row])
        residuals.append(float(row["target_log_opening_weekend"]) - pred_log[0])
    return residuals


def residual_quantiles(values: list[float]) -> dict[str, float]:
    return {
        "q05": opening.percentile(values, 0.05) or 0.0,
        "q10": opening.percentile(values, 0.10) or 0.0,
        "q25": opening.percentile(values, 0.25) or 0.0,
        "q50": opening.percentile(values, 0.50) or 0.0,
        "q75": opening.percentile(values, 0.75) or 0.0,
        "q90": opening.percentile(values, 0.90) or 0.0,
        "q95": opening.percentile(values, 0.95) or 0.0,
    }


def interval_residuals_by_bucket(
    *,
    rows: list[dict[str, object]],
    residuals: list[float],
) -> tuple[dict[str, list[float]], list[float]]:
    by_bucket: dict[str, list[float]] = defaultdict(list)
    for row, residual in zip(rows, residuals):
        by_bucket[estimate_bucket_label(row)].append(residual)
    return dict(by_bucket), list(residuals)


def interval_for_row(
    row: dict[str, object],
    *,
    pred_log: float,
    bucket_residuals: dict[str, list[float]],
    global_residuals: list[float],
    interval_method: str = "empirical_quantile",
    min_bucket_residuals: int = MIN_BUCKET_INTERVAL_RESIDUALS,
) -> dict[str, object]:
    bucket = estimate_bucket_label(row)
    values = bucket_residuals.get(bucket, [])
    interval_source = "bucket" if len(values) >= min_bucket_residuals else "global"
    if interval_source == "global":
        values = global_residuals
    if interval_method in {"conformal_abs", "loo_conformal_abs"}:
        abs_values = [abs(value) for value in values]
        q50 = opening.percentile(abs_values, 0.50) or 0.0
        q80 = opening.percentile(abs_values, 0.80) or 0.0
        q90 = opening.percentile(abs_values, 0.90) or 0.0
        lower_50 = math.exp(pred_log - q50)
        upper_50 = math.exp(pred_log + q50)
        lower_80 = math.exp(pred_log - q80)
        upper_80 = math.exp(pred_log + q80)
        lower_90 = math.exp(pred_log - q90)
        upper_90 = math.exp(pred_log + q90)
        p50 = math.exp(pred_log)
    else:
        interval_method = "empirical_quantile"
        qs = residual_quantiles(values)
        lower_50 = math.exp(pred_log + qs["q25"])
        upper_50 = math.exp(pred_log + qs["q75"])
        lower_80 = math.exp(pred_log + qs["q10"])
        upper_80 = math.exp(pred_log + qs["q90"])
        lower_90 = math.exp(pred_log + qs["q05"])
        upper_90 = math.exp(pred_log + qs["q95"])
        p50 = math.exp(pred_log + qs["q50"])
    return {
        "prediction_interval_method": interval_method,
        "prediction_interval_source": interval_source,
        "prediction_interval_train_n": len(values),
        "predicted_p50_opening_weekend_revenue_usd": max(1.0, p50),
        "predicted_lower_50_opening_weekend_revenue_usd": max(1.0, lower_50),
        "predicted_upper_50_opening_weekend_revenue_usd": max(1.0, upper_50),
        "predicted_lower_80_opening_weekend_revenue_usd": max(1.0, lower_80),
        "predicted_upper_80_opening_weekend_revenue_usd": max(1.0, upper_80),
        "predicted_lower_90_opening_weekend_revenue_usd": max(1.0, lower_90),
        "predicted_upper_90_opening_weekend_revenue_usd": max(1.0, upper_90),
    }


def metric_row_from_predictions(
    *,
    base: dict[str, object],
    prediction_rows: list[dict[str, object]],
) -> dict[str, object]:
    actual_log = [float(row["actual_log_opening_weekend"]) for row in prediction_rows]
    pred_log = [float(row["predicted_log_opening_weekend"]) for row in prediction_rows]
    actual_gross = [float(row["actual_opening_weekend_revenue_usd"]) for row in prediction_rows]
    pred_gross = [float(row["predicted_opening_weekend_revenue_usd"]) for row in prediction_rows]
    log_errors = [actual - pred for actual, pred in zip(actual_log, pred_log)]
    apes = [abs(pred - actual) / actual for actual, pred in zip(actual_gross, pred_gross) if actual > 0.0]
    return {
        **base,
        "r2_log_revenue": format_number(r2_score(actual_log, pred_log)),
        "r2_gross": format_number(r2_score(actual_gross, pred_gross)),
        "mape_gross": format_number(mean(apes) if apes else None),
        "rmse_log_revenue": format_number(opening.rmse(log_errors)),
        "mae_log_revenue": format_number(opening.mae(log_errors)),
        "mean_actual_gross": format_number(mean(actual_gross)),
        "mean_predicted_gross": format_number(mean(pred_gross)),
        "mean_interval_80_width_pct": format_number(
            mean(
                (
                    float(row["predicted_upper_80_opening_weekend_revenue_usd"])
                    - float(row["predicted_lower_80_opening_weekend_revenue_usd"])
                )
                / float(row["predicted_opening_weekend_revenue_usd"])
                for row in prediction_rows
                if float(row["predicted_opening_weekend_revenue_usd"]) > 0.0
            )
        ),
        "coverage_50": format_number(
            mean(
                1.0
                if float(row["predicted_lower_50_opening_weekend_revenue_usd"])
                <= float(row["actual_opening_weekend_revenue_usd"])
                <= float(row["predicted_upper_50_opening_weekend_revenue_usd"])
                else 0.0
                for row in prediction_rows
            )
        ),
        "coverage_80": format_number(
            mean(
                1.0
                if float(row["predicted_lower_80_opening_weekend_revenue_usd"])
                <= float(row["actual_opening_weekend_revenue_usd"])
                <= float(row["predicted_upper_80_opening_weekend_revenue_usd"])
                else 0.0
                for row in prediction_rows
            )
        ),
        "coverage_90": format_number(
            mean(
                1.0
                if float(row["predicted_lower_90_opening_weekend_revenue_usd"])
                <= float(row["actual_opening_weekend_revenue_usd"])
                <= float(row["predicted_upper_90_opening_weekend_revenue_usd"])
                else 0.0
                for row in prediction_rows
            )
        ),
        "status": "ok",
    }


def prediction_row(
    row: dict[str, object],
    *,
    model_name: str,
    population: str,
    prediction_source: str,
    pred_log: float,
    interval: dict[str, object],
) -> dict[str, object]:
    actual_gross = float(row["opening_weekend_revenue_usd"])
    pred_gross = max(1.0, math.exp(pred_log))
    return {
        "model": model_name,
        "population": population,
        "prediction_source": prediction_source,
        "forecast_stage": row["forecast_stage"],
        "snapshot_day": row["snapshot_day"],
        "as_of_date": row["as_of_date"],
        "movie_id": row["movie_id"],
        "title": row["title"],
        "release_year": row["release_year"],
        "opening_date": row["opening_date"],
        "bop_forecast_available": row.get("bop_forecast_available", 0.0),
        "bop_forecast_midpoint": row.get("bop_forecast_midpoint", 0.0),
        "bop_estimate_bucket": estimate_bucket_label(row),
        "wiki_available": row.get("wiki_available", 0.0),
        "actual_log_opening_weekend": row["target_log_opening_weekend"],
        "predicted_log_opening_weekend": pred_log,
        "actual_opening_weekend_revenue_usd": row["opening_weekend_revenue_usd"],
        "predicted_opening_weekend_revenue_usd": pred_gross,
        "absolute_percentage_error": abs(pred_gross - actual_gross) / actual_gross if actual_gross > 0.0 else "",
        **interval,
    }


def prediction_rows_for_interval_methods(
    rows: list[dict[str, object]],
    pred_logs: list[float],
    *,
    model_name: str,
    population: str,
    prediction_source: str,
    bucket_residuals: dict[str, list[float]],
    global_residuals: list[float],
    residuals_by_method: dict[str, tuple[dict[str, list[float]], list[float]]] | None = None,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for interval_method in INTERVAL_METHODS:
        method_bucket_residuals, method_global_residuals = (
            residuals_by_method.get(interval_method, (bucket_residuals, global_residuals))
            if residuals_by_method is not None
            else (bucket_residuals, global_residuals)
        )
        out.extend(
            prediction_row(
                row,
                model_name=model_name,
                population=population,
                prediction_source=prediction_source,
                pred_log=pred_log,
                interval=interval_for_row(
                    row,
                    pred_log=pred_log,
                    bucket_residuals=method_bucket_residuals,
                    global_residuals=method_global_residuals,
                    interval_method=interval_method,
                ),
            )
            for row, pred_log in zip(rows, pred_logs)
        )
    return out


def evaluate_day_by_day_snapshots(
    panel_rows: list[dict[str, object]],
    *,
    snapshot_days: list[int],
    train_start_year: int,
    train_end_year: int,
    test_start_year: int,
    test_end_year: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []

    for snapshot_day in snapshot_days:
        train_all, holdout_all = train_holdout_rows(
            panel_rows,
            snapshot_day=snapshot_day,
            train_start_year=train_start_year,
            train_end_year=train_end_year,
            test_start_year=test_start_year,
            test_end_year=test_end_year,
        )
        fallback_model: opening.FittedModel | None = None
        fallback_train = [row for row in train_all if float(row.get("wiki_available", 0.0) or 0.0) > 0.0]
        if len(fallback_train) >= len(FALLBACK_TERMS) + 2:
            fallback_model = opening.fit_model_for_target(fallback_train, FALLBACK_TERMS, "target_log_opening_weekend")
            fallback_train_pred = opening.predict_log(fallback_model, fallback_train)
            fallback_residuals = prediction_residuals_for_interval(
                model_name=FALLBACK_MODEL_NAME,
                pred_log=fallback_train_pred,
                rows=fallback_train,
            )
            fallback_loo_residuals = loo_prediction_residuals_for_opening_model(
                rows=fallback_train,
                terms=FALLBACK_TERMS,
            )
        else:
            fallback_residuals = []
            fallback_loo_residuals = []

        for model_name, terms in SNAPSHOT_MODEL_TERMS.items():
            target_key = "target_log_opening_weekend" if model_name == "calibrated_bop_snapshot" else "target_log_bop_residual"
            train_bop = bop_rows(train_all)
            holdout_bop = bop_rows(holdout_all)
            if model_name != "raw_bop_snapshot":
                train_bop = rows_with_target(train_bop, target_key)

            metric_base = {
                "model": model_name,
                "forecast_stage": "pre_release",
                "snapshot_day": snapshot_day,
                "train_start_year": train_start_year,
                "train_end_year": train_end_year,
                "test_start_year": test_start_year,
                "test_end_year": test_end_year,
            }
            if len(holdout_bop) < 2 or (model_name != "raw_bop_snapshot" and len(train_bop) < len(terms) + 2):
                for interval_method in INTERVAL_METHODS:
                    for population in ("bop_covered", "full_with_fallback"):
                        metric_rows.append(
                            {
                                **metric_base,
                                "population": population,
                                "interval_method": interval_method,
                                "train_n": len(train_bop) if model_name != "raw_bop_snapshot" else 0,
                                "holdout_n": 0,
                                "bop_prediction_n": 0,
                                "fallback_prediction_n": 0,
                                "r2_log_revenue": "",
                                "r2_gross": "",
                                "mape_gross": "",
                                "rmse_log_revenue": "",
                                "mae_log_revenue": "",
                                "mean_actual_gross": "",
                                "mean_predicted_gross": "",
                                "mean_interval_80_width_pct": "",
                                "coverage_50": "",
                                "coverage_80": "",
                                "coverage_90": "",
                                "status": "insufficient_sample",
                            }
                        )
                continue

            train_pred_log, fitted = prediction_log_for_model(
                model_name=model_name,
                terms=terms,
                train_rows=train_bop,
                predict_rows=train_bop,
            )
            holdout_pred_log, _ = prediction_log_for_model(
                model_name=model_name,
                terms=terms,
                train_rows=train_bop,
                predict_rows=holdout_bop,
            )
            train_residuals = prediction_residuals_for_interval(
                model_name=model_name,
                pred_log=train_pred_log,
                rows=train_bop,
            )
            bucket_residuals, global_residuals = interval_residuals_by_bucket(
                rows=train_bop,
                residuals=train_residuals,
            )
            loo_residuals = loo_prediction_residuals_for_model(
                model_name=model_name,
                terms=terms,
                rows=train_bop,
            )
            loo_bucket_residuals, loo_global_residuals = interval_residuals_by_bucket(
                rows=train_bop,
                residuals=loo_residuals,
            )
            residuals_by_method = {
                "empirical_quantile": (bucket_residuals, global_residuals),
                "conformal_abs": (bucket_residuals, global_residuals),
                "loo_conformal_abs": (loo_bucket_residuals, loo_global_residuals),
            }
            if fitted is not None:
                target_label = "log_opening_weekend" if model_name == "calibrated_bop_snapshot" else "log_bop_residual"
                for term, coef, center, scale in zip(
                    ["intercept"] + terms,
                    fitted.beta,
                    [0.0] + fitted.centers,
                    [1.0] + fitted.scales,
                ):
                    coefficient_rows.append(
                        {
                            "model": model_name,
                            "forecast_stage": "pre_release",
                            "snapshot_day": snapshot_day,
                            "target": target_label,
                            "term": term,
                            "standardized_coef": coef,
                            "center": center,
                            "scale": scale,
                            "train_n": len(train_bop),
                        }
                    )

            bop_predictions = prediction_rows_for_interval_methods(
                holdout_bop,
                holdout_pred_log,
                model_name=model_name,
                population="bop_covered",
                prediction_source="bop",
                bucket_residuals=bucket_residuals,
                global_residuals=global_residuals,
                residuals_by_method=residuals_by_method,
            )
            for interval_method in INTERVAL_METHODS:
                method_bop_predictions = [
                    row for row in bop_predictions if row["prediction_interval_method"] == interval_method
                ]
                if len(method_bop_predictions) >= 2:
                    metric_rows.append(
                        metric_row_from_predictions(
                            base={
                                **metric_base,
                                "population": "bop_covered",
                                "interval_method": interval_method,
                                "train_n": len(train_bop) if model_name != "raw_bop_snapshot" else 0,
                                "holdout_n": len(method_bop_predictions),
                                "bop_prediction_n": len(method_bop_predictions),
                                "fallback_prediction_n": 0,
                            },
                            prediction_rows=method_bop_predictions,
                        )
                    )
            prediction_rows.extend(bop_predictions)

            full_predictions = [dict(row, population="full_with_fallback") for row in bop_predictions]
            missing_bop_holdout = [
                row for row in holdout_all if float(row.get("bop_forecast_available", 0.0) or 0.0) <= 0.0
            ]
            if fallback_model is not None and missing_bop_holdout:
                fallback_pred_log = opening.predict_log(fallback_model, missing_bop_holdout)
                fallback_bucket_residuals, fallback_global_residuals = interval_residuals_by_bucket(
                    rows=fallback_train,
                    residuals=fallback_residuals,
                )
                fallback_loo_bucket_residuals, fallback_loo_global_residuals = interval_residuals_by_bucket(
                    rows=fallback_train,
                    residuals=fallback_loo_residuals,
                )
                fallback_residuals_by_method = {
                    "empirical_quantile": (fallback_bucket_residuals, fallback_global_residuals),
                    "conformal_abs": (fallback_bucket_residuals, fallback_global_residuals),
                    "loo_conformal_abs": (fallback_loo_bucket_residuals, fallback_loo_global_residuals),
                }
                full_predictions.extend(
                    prediction_rows_for_interval_methods(
                        missing_bop_holdout,
                        fallback_pred_log,
                        model_name=model_name,
                        population="full_with_fallback",
                        prediction_source="fallback",
                        bucket_residuals=fallback_bucket_residuals,
                        global_residuals=fallback_global_residuals,
                        residuals_by_method=fallback_residuals_by_method,
                    )
                )
            for interval_method in INTERVAL_METHODS:
                method_full_predictions = [
                    row for row in full_predictions if row["prediction_interval_method"] == interval_method
                ]
                if len(method_full_predictions) >= 2:
                    metric_rows.append(
                        metric_row_from_predictions(
                            base={
                                **metric_base,
                                "population": "full_with_fallback",
                                "interval_method": interval_method,
                                "train_n": len(train_bop) if model_name != "raw_bop_snapshot" else 0,
                                "holdout_n": len(method_full_predictions),
                                "bop_prediction_n": sum(
                                    1 for row in method_full_predictions if row["prediction_source"] == "bop"
                                ),
                                "fallback_prediction_n": sum(
                                    1 for row in method_full_predictions if row["prediction_source"] == "fallback"
                                ),
                            },
                            prediction_rows=method_full_predictions,
                        )
                    )
            prediction_rows.extend(full_predictions)

    interval_rows = interval_coverage_rows(prediction_rows)
    return prediction_rows, metric_rows, coefficient_rows, interval_rows


def interval_coverage_rows(prediction_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, int, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in prediction_rows:
        grouped[
            (
                str(row["model"]),
                str(row["population"]),
                int(row["snapshot_day"]),
                str(row["forecast_stage"]),
                str(row["prediction_interval_method"]),
            )
        ].append(row)
    for (model_name, population, snapshot_day, stage, interval_method), group in sorted(grouped.items()):
        for level, lower_key, upper_key in (
            (
                "50",
                "predicted_lower_50_opening_weekend_revenue_usd",
                "predicted_upper_50_opening_weekend_revenue_usd",
            ),
            (
                "80",
                "predicted_lower_80_opening_weekend_revenue_usd",
                "predicted_upper_80_opening_weekend_revenue_usd",
            ),
            (
                "90",
                "predicted_lower_90_opening_weekend_revenue_usd",
                "predicted_upper_90_opening_weekend_revenue_usd",
            ),
        ):
            hits = [
                1.0
                if float(row[lower_key])
                <= float(row["actual_opening_weekend_revenue_usd"])
                <= float(row[upper_key])
                else 0.0
                for row in group
            ]
            widths = [
                (float(row[upper_key]) - float(row[lower_key]))
                / float(row["predicted_opening_weekend_revenue_usd"])
                for row in group
                if float(row["predicted_opening_weekend_revenue_usd"]) > 0.0
            ]
            rows.append(
                {
                    "model": model_name,
                    "population": population,
                    "forecast_stage": stage,
                    "snapshot_day": snapshot_day,
                    "interval_method": interval_method,
                    "interval_level": level,
                    "holdout_n": len(group),
                    "coverage": format_number(mean(hits)),
                    "mean_width_pct": format_number(mean(widths) if widths else None),
                }
            )
    return rows


def prediction_revision_rows(prediction_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in prediction_rows:
        grouped[
            (
                str(row["model"]),
                str(row["population"]),
                str(row["prediction_interval_method"]),
                int(row["movie_id"]),
            )
        ].append(row)
    for (_model_name, _population, _interval_method, _movie_id), group in grouped.items():
        previous: dict[str, object] | None = None
        for row in sorted(group, key=lambda item: int(item["snapshot_day"])):
            pred = float(row["predicted_opening_weekend_revenue_usd"])
            prev_pred = float(previous["predicted_opening_weekend_revenue_usd"]) if previous is not None else None
            out.append(
                {
                    "model": row["model"],
                    "population": row["population"],
                    "interval_method": row["prediction_interval_method"],
                    "movie_id": row["movie_id"],
                    "title": row["title"],
                    "release_year": row["release_year"],
                    "opening_date": row["opening_date"],
                    "snapshot_day": row["snapshot_day"],
                    "as_of_date": row["as_of_date"],
                    "predicted_opening_weekend_revenue_usd": pred,
                    "previous_predicted_opening_weekend_revenue_usd": prev_pred if prev_pred is not None else "",
                    "prediction_change_usd": pred - prev_pred if prev_pred is not None else "",
                    "prediction_change_pct": (pred - prev_pred) / prev_pred if prev_pred else "",
                    "actual_opening_weekend_revenue_usd": row["actual_opening_weekend_revenue_usd"],
                }
            )
            previous = row
    return out


def coverage_rows(panel_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in panel_rows:
        grouped[int(row["snapshot_day"])].append(row)
    for snapshot_day, group in sorted(grouped.items()):
        rows.append(
            {
                "forecast_stage": "pre_release",
                "snapshot_day": snapshot_day,
                "rows": len(group),
                "movies": len({row["movie_id"] for row in group}),
                "bop_available_rows": sum(1 for row in group if float(row.get("bop_forecast_available", 0.0) or 0.0) > 0.0),
                "wiki_available_rows": sum(1 for row in group if float(row.get("wiki_available", 0.0) or 0.0) > 0.0),
                "competitor_lag7_rows": sum(1 for row in group if float(row.get("competitor_count_lag7", 0.0) or 0.0) > 0.0),
            }
        )
    return rows


def svg_escape(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def write_placeholder_svg(path: Path, title: str, message: str) -> None:
    opening.write_placeholder_svg(path, title, message)


def write_metric_by_horizon_svg(path: Path, metric_rows: list[dict[str, object]]) -> None:
    clean = [
        row
        for row in metric_rows
        if row["status"] == "ok" and row["population"] == "bop_covered" and row["mape_gross"] != ""
        and row.get("interval_method") == "loo_conformal_abs"
    ]
    if not clean:
        write_placeholder_svg(path, "MAPE by forecast horizon", "No metrics were available.")
        return
    width, height = 900, 520
    left, right, top, bottom = 74, 250, 52, 76
    days = sorted({int(row["snapshot_day"]) for row in clean})
    models = sorted({str(row["model"]) for row in clean})
    colors = ["#2f6f73", "#c15b3f", "#574b90", "#4c78a8", "#8a6f2a"]
    ymax = max(float(row["mape_gross"]) for row in clean) or 1.0
    xmin, xmax = min(days), max(days)
    if xmin == xmax:
        xmax += 1

    def sx(day: float) -> float:
        return left + (day - xmin) / (xmax - xmin) * (width - left - right)

    def sy(value: float) -> float:
        return top + (ymax - value) / ymax * (height - top - bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">Holdout MAPE by forecast day</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333"/>',
    ]
    for idx, model_name in enumerate(models):
        color = colors[idx % len(colors)]
        model_rows = sorted([row for row in clean if row["model"] == model_name], key=lambda row: int(row["snapshot_day"]))
        points = " ".join(f'{sx(float(row["snapshot_day"])):.1f},{sy(float(row["mape_gross"])):.1f}' for row in model_rows)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.3"/>')
        for row in model_rows:
            parts.append(f'<circle cx="{sx(float(row["snapshot_day"])):.1f}" cy="{sy(float(row["mape_gross"])):.1f}" r="3" fill="{color}"/>')
        y = top + 18 + idx * 22
        parts.append(f'<line x1="{width - right + 24}" y1="{y}" x2="{width - right + 48}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{width - right + 56}" y="{y + 4}" font-family="Arial" font-size="11">{svg_escape(model_name)}</text>')
    for day in days:
        parts.append(f'<text x="{sx(day):.1f}" y="{height - 48}" text-anchor="middle" font-family="Arial" font-size="11">t={day}</text>')
    parts.append(f'<text x="{(left + width - right) / 2:.1f}" y="{height - 18}" font-family="Arial" font-size="12" text-anchor="middle">Snapshot day</text>')
    parts.append(f'<text x="18" y="{(top + height - bottom) / 2:.1f}" font-family="Arial" font-size="12" transform="rotate(-90 18 {(top + height - bottom) / 2:.1f})" text-anchor="middle">MAPE</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_r2_by_horizon_svg(path: Path, metric_rows: list[dict[str, object]]) -> None:
    clean = [
        row
        for row in metric_rows
        if row["status"] == "ok"
        and row["population"] == "bop_covered"
        and row["r2_gross"] != ""
        and row.get("interval_method") == "loo_conformal_abs"
    ]
    if not clean:
        write_placeholder_svg(path, "Gross R2 by forecast horizon", "No metrics were available.")
        return
    width, height = 900, 520
    left, right, top, bottom = 74, 250, 52, 76
    days = sorted({int(row["snapshot_day"]) for row in clean})
    models = sorted({str(row["model"]) for row in clean})
    colors = ["#2f6f73", "#c15b3f", "#574b90", "#4c78a8", "#8a6f2a", "#6b5b95"]
    ymin = min(0.0, min(float(row["r2_gross"]) for row in clean))
    ymax = max(float(row["r2_gross"]) for row in clean)
    if ymin == ymax:
        ymax = ymin + 1.0
    xmin, xmax = min(days), max(days)
    if xmin == xmax:
        xmax += 1

    def sx(day: float) -> float:
        return left + (day - xmin) / (xmax - xmin) * (width - left - right)

    def sy(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * (height - top - bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">Holdout gross R2 by forecast day</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333"/>',
    ]
    if ymin <= 0 <= ymax:
        parts.append(f'<line x1="{left}" y1="{sy(0.0):.1f}" x2="{width - right}" y2="{sy(0.0):.1f}" stroke="#aaa" stroke-dasharray="4 4"/>')
    for idx, model_name in enumerate(models):
        color = colors[idx % len(colors)]
        model_rows = sorted([row for row in clean if row["model"] == model_name], key=lambda row: int(row["snapshot_day"]))
        points = " ".join(f'{sx(float(row["snapshot_day"])):.1f},{sy(float(row["r2_gross"])):.1f}' for row in model_rows)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.3"/>')
        y = top + 18 + idx * 22
        parts.append(f'<line x1="{width - right + 24}" y1="{y}" x2="{width - right + 48}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{width - right + 56}" y="{y + 4}" font-family="Arial" font-size="11">{svg_escape(model_name)}</text>')
    for day in days:
        parts.append(f'<text x="{sx(day):.1f}" y="{height - 48}" text-anchor="middle" font-family="Arial" font-size="11">t={day}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_interval_coverage_svg(path: Path, interval_rows: list[dict[str, object]]) -> None:
    clean = [
        row
        for row in interval_rows
        if row["population"] == "bop_covered"
        and row.get("interval_method") == "loo_conformal_abs"
        and row["interval_level"] == "80"
        and row["coverage"] != ""
    ]
    if not clean:
        write_placeholder_svg(path, "80% interval coverage", "No interval coverage rows were available.")
        return
    width, height = 860, 460
    left, right, top, bottom = 70, 240, 52, 76
    days = sorted({int(row["snapshot_day"]) for row in clean})
    models = sorted({str(row["model"]) for row in clean})
    colors = ["#2f6f73", "#c15b3f", "#574b90", "#4c78a8", "#8a6f2a"]
    xmin, xmax = min(days), max(days)
    if xmin == xmax:
        xmax += 1

    def sx(day: float) -> float:
        return left + (day - xmin) / (xmax - xmin) * (width - left - right)

    def sy(value: float) -> float:
        return top + (1.0 - value) * (height - top - bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">80% interval coverage by forecast day</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{sy(0.8):.1f}" x2="{width - right}" y2="{sy(0.8):.1f}" stroke="#777" stroke-dasharray="4 4"/>',
    ]
    for idx, model_name in enumerate(models):
        color = colors[idx % len(colors)]
        model_rows = sorted([row for row in clean if row["model"] == model_name], key=lambda row: int(row["snapshot_day"]))
        points = " ".join(f'{sx(float(row["snapshot_day"])):.1f},{sy(float(row["coverage"])):.1f}' for row in model_rows)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.3"/>')
        y = top + 18 + idx * 22
        parts.append(f'<line x1="{width - right + 24}" y1="{y}" x2="{width - right + 48}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{width - right + 56}" y="{y + 4}" font-family="Arial" font-size="11">{svg_escape(model_name)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_interval_width_svg(path: Path, interval_rows: list[dict[str, object]]) -> None:
    clean = [
        row
        for row in interval_rows
        if row["population"] == "bop_covered"
        and row.get("interval_method") == "conformal_abs"
        and row["interval_level"] == "80"
        and row["mean_width_pct"] != ""
    ]
    if not clean:
        write_placeholder_svg(path, "80% interval width", "No interval width rows were available.")
        return
    width, height = 860, 460
    left, right, top, bottom = 70, 240, 52, 76
    days = sorted({int(row["snapshot_day"]) for row in clean})
    models = sorted({str(row["model"]) for row in clean})
    colors = ["#2f6f73", "#c15b3f", "#574b90", "#4c78a8", "#8a6f2a"]
    ymax = max(float(row["mean_width_pct"]) for row in clean) or 1.0
    xmin, xmax = min(days), max(days)
    if xmin == xmax:
        xmax += 1

    def sx(day: float) -> float:
        return left + (day - xmin) / (xmax - xmin) * (width - left - right)

    def sy(value: float) -> float:
        return top + (ymax - value) / ymax * (height - top - bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">80% interval width by forecast day</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333"/>',
    ]
    for idx, model_name in enumerate(models):
        color = colors[idx % len(colors)]
        model_rows = sorted([row for row in clean if row["model"] == model_name], key=lambda row: int(row["snapshot_day"]))
        points = " ".join(f'{sx(float(row["snapshot_day"])):.1f},{sy(float(row["mean_width_pct"])):.1f}' for row in model_rows)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.3"/>')
        y = top + 18 + idx * 22
        parts.append(f'<line x1="{width - right + 24}" y1="{y}" x2="{width - right + 48}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{width - right + 56}" y="{y + 4}" font-family="Arial" font-size="11">{svg_escape(model_name)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_fan_chart_svg(path: Path, prediction_rows: list[dict[str, object]]) -> None:
    candidates = [
        row
        for row in prediction_rows
        if row["model"] == "bop_residual_wiki_competition_snapshot"
        and row["population"] == "bop_covered"
        and row.get("prediction_interval_method") == "loo_conformal_abs"
    ]
    if not candidates:
        write_placeholder_svg(path, "Forecast fan chart", "No combined snapshot predictions were available.")
        return
    movie_id = sorted(
        {int(row["movie_id"]) for row in candidates},
        key=lambda mid: max(
            float(row["actual_opening_weekend_revenue_usd"]) for row in candidates if int(row["movie_id"]) == mid
        ),
        reverse=True,
    )[0]
    rows = sorted([row for row in candidates if int(row["movie_id"]) == movie_id], key=lambda row: int(row["snapshot_day"]))
    width, height = 820, 460
    left, right, top, bottom = 74, 36, 52, 76
    days = [int(row["snapshot_day"]) for row in rows]
    values = []
    for row in rows:
        values.extend(
            [
                float(row["predicted_lower_90_opening_weekend_revenue_usd"]),
                float(row["predicted_upper_90_opening_weekend_revenue_usd"]),
                float(row["predicted_opening_weekend_revenue_usd"]),
                float(row["actual_opening_weekend_revenue_usd"]),
            ]
        )
    ymin, ymax = min(values), max(values)
    if ymin == ymax:
        ymax += 1.0
    xmin, xmax = min(days), max(days)
    if xmin == xmax:
        xmax += 1

    def sx(day: float) -> float:
        return left + (day - xmin) / (xmax - xmin) * (width - left - right)

    def sy(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * (height - top - bottom)

    upper = " ".join(f'{sx(float(row["snapshot_day"])):.1f},{sy(float(row["predicted_upper_90_opening_weekend_revenue_usd"])):.1f}' for row in rows)
    lower = " ".join(f'{sx(float(row["snapshot_day"])):.1f},{sy(float(row["predicted_lower_90_opening_weekend_revenue_usd"])):.1f}' for row in reversed(rows))
    pred = " ".join(f'{sx(float(row["snapshot_day"])):.1f},{sy(float(row["predicted_opening_weekend_revenue_usd"])):.1f}' for row in rows)
    actual = float(rows[0]["actual_opening_weekend_revenue_usd"])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">Forecast fan chart: {svg_escape(rows[0]["title"])}</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333"/>',
        f'<polygon points="{upper} {lower}" fill="#8db3b1" fill-opacity="0.35"/>',
        f'<polyline points="{pred}" fill="none" stroke="#2f6f73" stroke-width="2.6"/>',
        f'<line x1="{left}" y1="{sy(actual):.1f}" x2="{width - right}" y2="{sy(actual):.1f}" stroke="#333" stroke-dasharray="5 5"/>',
    ]
    for row in rows:
        parts.append(f'<circle cx="{sx(float(row["snapshot_day"])):.1f}" cy="{sy(float(row["predicted_opening_weekend_revenue_usd"])):.1f}" r="4" fill="#2f6f73"/>')
        parts.append(f'<text x="{sx(float(row["snapshot_day"])):.1f}" y="{height - 48}" text-anchor="middle" font-family="Arial" font-size="11">t={row["snapshot_day"]}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_revision_waterfall_svg(path: Path, revision_rows: list[dict[str, object]]) -> None:
    candidates = [
        row
        for row in revision_rows
        if row["model"] == "bop_residual_wiki_competition_snapshot"
        and row["population"] == "bop_covered"
        and row.get("interval_method") == "loo_conformal_abs"
        and row["prediction_change_usd"] != ""
    ]
    if not candidates:
        write_placeholder_svg(path, "Forecast revision waterfall", "No revision rows were available.")
        return
    movie_id = sorted({int(row["movie_id"]) for row in candidates})[0]
    rows = sorted([row for row in candidates if int(row["movie_id"]) == movie_id], key=lambda row: int(row["snapshot_day"]))
    width, height = 820, 420
    left, right, top, bottom = 74, 36, 52, 76
    values = [float(row["prediction_change_usd"]) for row in rows]
    ymax = max(abs(value) for value in values) or 1.0
    bar_w = (width - left - right) / max(1, len(rows)) * 0.62

    def sx(idx: int) -> float:
        return left + (idx + 0.5) / len(rows) * (width - left - right)

    def sy(value: float) -> float:
        return top + (ymax - value) / (2.0 * ymax) * (height - top - bottom)

    zero_y = sy(0.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">Forecast revisions: {svg_escape(rows[0]["title"])}</text>',
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{width - right}" y2="{zero_y:.1f}" stroke="#333"/>',
    ]
    for idx, row in enumerate(rows):
        value = float(row["prediction_change_usd"])
        y = sy(max(0.0, value))
        h = abs(sy(value) - zero_y)
        color = "#2f6f73" if value >= 0 else "#c15b3f"
        parts.append(f'<rect x="{sx(idx) - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{sx(idx):.1f}" y="{height - 42}" text-anchor="middle" font-family="Arial" font-size="11">t={row["snapshot_day"]}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_residual_bucket_svg(path: Path, prediction_rows: list[dict[str, object]]) -> None:
    rows = [
        row
        for row in prediction_rows
        if row["model"] == "bop_residual_wiki_competition_snapshot"
        and row["population"] == "bop_covered"
        and row.get("prediction_interval_method") == "loo_conformal_abs"
    ]
    if not rows:
        write_placeholder_svg(path, "Residual by estimate bucket", "No combined snapshot predictions were available.")
        return
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        residual = math.log(float(row["actual_opening_weekend_revenue_usd"])) - math.log(
            float(row["predicted_opening_weekend_revenue_usd"])
        )
        grouped[str(row["bop_estimate_bucket"])].append(residual)
    labels = sorted(grouped)
    values = [mean(grouped[label]) for label in labels]
    width, height = 760, 420
    left, right, top, bottom = 84, 36, 52, 90
    ymax = max(abs(value) for value in values) or 1.0
    bar_w = (width - left - right) / max(1, len(labels)) * 0.62

    def sx(idx: int) -> float:
        return left + (idx + 0.5) / len(labels) * (width - left - right)

    def sy(value: float) -> float:
        return top + (ymax - value) / (2.0 * ymax) * (height - top - bottom)

    zero_y = sy(0.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">Mean log residual by BOP estimate bucket</text>',
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{width - right}" y2="{zero_y:.1f}" stroke="#333"/>',
    ]
    for idx, (label, value) in enumerate(zip(labels, values)):
        y = sy(max(0.0, value))
        h = abs(sy(value) - zero_y)
        parts.append(f'<rect x="{sx(idx) - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#574b90"/>')
        parts.append(f'<text x="{sx(idx):.1f}" y="{height - 48}" text-anchor="middle" font-family="Arial" font-size="11">{svg_escape(label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def best_metric_rows_by_horizon(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best: dict[tuple[str, str, int], dict[str, object]] = {}
    for row in metric_rows:
        if row["status"] != "ok" or row["mape_gross"] == "":
            continue
        key = (str(row["population"]), str(row["interval_method"]), int(row["snapshot_day"]))
        current = best.get(key)
        if current is None or (
            float(row["mape_gross"]),
            -float(row["r2_gross"]) if row["r2_gross"] != "" else float("inf"),
        ) < (
            float(current["mape_gross"]),
            -float(current["r2_gross"]) if current["r2_gross"] != "" else float("inf"),
        ):
            best[key] = row
    return [
        {
            "population": row["population"],
            "interval_method": row["interval_method"],
            "snapshot_day": row["snapshot_day"],
            "best_model": row["model"],
            "holdout_n": row["holdout_n"],
            "bop_prediction_n": row["bop_prediction_n"],
            "fallback_prediction_n": row["fallback_prediction_n"],
            "r2_log_revenue": row["r2_log_revenue"],
            "r2_gross": row["r2_gross"],
            "mape_gross": row["mape_gross"],
            "coverage_80": row["coverage_80"],
            "mean_interval_80_width_pct": row["mean_interval_80_width_pct"],
        }
        for row in sorted(best.values(), key=lambda item: (str(item["population"]), str(item["interval_method"]), int(item["snapshot_day"])))
    ]


FEATURE_PANEL_FIELDNAMES = [
    "movie_id",
    "title",
    "release_year",
    "release_run_id",
    "opening_date",
    "forecast_stage",
    "snapshot_day",
    "timing_day",
    "wiki_timing_day",
    "competition_timing_day",
    "bop_timing_day",
    "as_of_date",
    "opening_theaters",
    "log1p_opening_theaters",
    "opening_day_gross_usd",
    "opening_weekend_revenue_usd",
    "target_log_opening_weekend",
    "target_log_bop_residual",
    "bop_forecast_available",
    "bop_prediction_id",
    "bop_article_url",
    "bop_forecast_published_date",
    "bop_forecast_midpoint",
    "log1p_bop_forecast_midpoint",
    "bop_forecast_range_width_pct",
    "bop_source_rank",
    "bop_showtime_market_share_pct",
    "bop_estimate_bucket",
    "bop_estimate_bucket_under_15m",
    "bop_estimate_bucket_15_30m",
    "bop_estimate_bucket_30_60m",
    "bop_estimate_bucket_60_100m",
    "bop_estimate_bucket_100m_plus",
    "bop_q1_threshold",
    "bop_q2_threshold",
    "bop_q3_threshold",
    "bop_q4_threshold",
    "bop_q1_proxy",
    "bop_q2_proxy",
    "bop_q3_proxy",
    "bop_q4_proxy",
    "bop_q1_proxy_x_log1p_bop_forecast_midpoint",
    "bop_q4_proxy_x_log1p_bop_forecast_midpoint",
    "bop_q4_proxy_x_log1p_V",
    "bop_q4_proxy_x_log1p_U",
    "bop_q4_proxy_x_log1p_R",
    "bop_q4_proxy_x_log1p_E",
    "bop_q4_proxy_x_log1p_opening_theaters",
    "bop_q4_proxy_x_log1p_competitor_total_gross_lag1",
    "bop_q4_proxy_x_log1p_competitor_total_gross_lag7",
    "wiki_available",
    "V",
    "U",
    "R",
    "E",
    "log1p_V",
    "log1p_U",
    "log1p_R",
    "log1p_E",
    "competitor_total_gross_lag1",
    "competitor_top1_gross_lag1",
    "competitor_count_lag1",
    "competitor_hhi_lag1",
    "competitor_total_gross_lag3",
    "competitor_top1_gross_lag3",
    "competitor_count_lag3",
    "competitor_hhi_lag3",
    "competitor_total_gross_lag7",
    "competitor_top1_gross_lag7",
    "competitor_count_lag7",
    "competitor_hhi_lag7",
    "competitor_total_gross_previous_weekend",
    "competitor_top1_gross_previous_weekend",
    "competitor_count_previous_weekend",
    "competitor_hhi_previous_weekend",
    "log1p_competitor_total_gross_lag1",
    "log1p_competitor_top1_gross_lag1",
    "log1p_competitor_total_gross_lag3",
    "log1p_competitor_top1_gross_lag3",
    "log1p_competitor_total_gross_lag7",
    "log1p_competitor_top1_gross_lag7",
    "log1p_competitor_total_gross_previous_weekend",
    "log1p_competitor_top1_gross_previous_weekend",
]

PREDICTION_FIELDNAMES = [
    "model",
    "population",
    "prediction_source",
    "forecast_stage",
    "snapshot_day",
    "as_of_date",
    "movie_id",
    "title",
    "release_year",
    "opening_date",
    "bop_forecast_available",
    "bop_forecast_midpoint",
    "bop_estimate_bucket",
    "wiki_available",
    "actual_log_opening_weekend",
    "predicted_log_opening_weekend",
    "actual_opening_weekend_revenue_usd",
    "predicted_opening_weekend_revenue_usd",
    "predicted_p50_opening_weekend_revenue_usd",
    "predicted_lower_50_opening_weekend_revenue_usd",
    "predicted_upper_50_opening_weekend_revenue_usd",
    "predicted_lower_80_opening_weekend_revenue_usd",
    "predicted_upper_80_opening_weekend_revenue_usd",
    "predicted_lower_90_opening_weekend_revenue_usd",
    "predicted_upper_90_opening_weekend_revenue_usd",
    "prediction_interval_method",
    "prediction_interval_source",
    "prediction_interval_train_n",
    "absolute_percentage_error",
]

METRIC_FIELDNAMES = [
    "model",
    "population",
    "forecast_stage",
    "snapshot_day",
    "interval_method",
    "train_start_year",
    "train_end_year",
    "test_start_year",
    "test_end_year",
    "train_n",
    "holdout_n",
    "bop_prediction_n",
    "fallback_prediction_n",
    "r2_log_revenue",
    "r2_gross",
    "mape_gross",
    "rmse_log_revenue",
    "mae_log_revenue",
    "mean_actual_gross",
    "mean_predicted_gross",
    "mean_interval_80_width_pct",
    "coverage_50",
    "coverage_80",
    "coverage_90",
    "status",
]

BEST_METRIC_FIELDNAMES = [
    "population",
    "interval_method",
    "snapshot_day",
    "best_model",
    "holdout_n",
    "bop_prediction_n",
    "fallback_prediction_n",
    "r2_log_revenue",
    "r2_gross",
    "mape_gross",
    "coverage_80",
    "mean_interval_80_width_pct",
]

COEFFICIENT_FIELDNAMES = [
    "model",
    "forecast_stage",
    "snapshot_day",
    "target",
    "term",
    "standardized_coef",
    "center",
    "scale",
    "train_n",
]

INTERVAL_COVERAGE_FIELDNAMES = [
    "model",
    "population",
    "forecast_stage",
    "snapshot_day",
    "interval_method",
    "interval_level",
    "holdout_n",
    "coverage",
    "mean_width_pct",
]

REVISION_FIELDNAMES = [
    "model",
    "population",
    "interval_method",
    "movie_id",
    "title",
    "release_year",
    "opening_date",
    "snapshot_day",
    "as_of_date",
    "predicted_opening_weekend_revenue_usd",
    "previous_predicted_opening_weekend_revenue_usd",
    "prediction_change_usd",
    "prediction_change_pct",
    "actual_opening_weekend_revenue_usd",
]

COVERAGE_FIELDNAMES = [
    "forecast_stage",
    "snapshot_day",
    "rows",
    "movies",
    "bop_available_rows",
    "wiki_available_rows",
    "competitor_lag7_rows",
]


def write_outputs(
    out_dir: Path,
    *,
    panel_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    coefficient_rows: list[dict[str, object]],
    interval_rows: list[dict[str, object]],
    revision_rows: list[dict[str, object]],
    coverage: list[dict[str, object]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "day_by_day_feature_panel.csv", panel_rows, FEATURE_PANEL_FIELDNAMES)
    write_csv(out_dir / "day_by_day_forecast_snapshots.csv", prediction_rows, PREDICTION_FIELDNAMES)
    write_csv(out_dir / "day_by_day_metrics_by_horizon.csv", metric_rows, METRIC_FIELDNAMES)
    write_csv(
        out_dir / "day_by_day_best_metrics_by_horizon.csv",
        best_metric_rows_by_horizon(metric_rows),
        BEST_METRIC_FIELDNAMES,
    )
    write_csv(out_dir / "day_by_day_interval_coverage.csv", interval_rows, INTERVAL_COVERAGE_FIELDNAMES)
    write_csv(out_dir / "day_by_day_coefficients.csv", coefficient_rows, COEFFICIENT_FIELDNAMES)
    write_csv(out_dir / "day_by_day_prediction_revisions.csv", revision_rows, REVISION_FIELDNAMES)
    write_csv(out_dir / "day_by_day_feature_coverage.csv", coverage, COVERAGE_FIELDNAMES)
    write_metric_by_horizon_svg(out_dir / "figure_mape_r2_by_horizon.svg", metric_rows)
    write_r2_by_horizon_svg(out_dir / "figure_gross_r2_by_horizon.svg", metric_rows)
    write_interval_coverage_svg(out_dir / "figure_interval_coverage_by_horizon.svg", interval_rows)
    write_interval_width_svg(out_dir / "figure_interval_width_by_horizon.svg", interval_rows)
    write_fan_chart_svg(out_dir / "figure_forecast_fan_chart.svg", prediction_rows)
    write_revision_waterfall_svg(out_dir / "figure_forecast_revision_waterfall.svg", revision_rows)
    write_residual_bucket_svg(out_dir / "figure_residual_vs_bop_estimate_bucket.svg", prediction_rows)


def parse_day_list(value: str) -> list[int]:
    return opening.parse_day_list(value)


def run(args: argparse.Namespace) -> int:
    snapshot_days = parse_day_list(args.snapshot_days)
    if not snapshot_days:
        raise SystemExit("--snapshot-days must include at least one day")
    if args.train_start_year > args.train_end_year:
        raise SystemExit("--train-start-year must be <= --train-end-year")
    if args.test_start_year > args.test_end_year:
        raise SystemExit("--test-start-year must be <= --test-end-year")
    if args.min_opening_day_gross < 0:
        raise SystemExit("--min-opening-day-gross must be non-negative")

    min_year = min(args.train_start_year, args.test_start_year)
    max_year = max(args.train_end_year, args.test_end_year)
    conn = connect_database(args.database_url)
    try:
        movies = opening.load_opening_weekend_movies(
            conn,
            min_year=min_year,
            max_year=max_year,
            min_opening_day_gross=args.min_opening_day_gross,
        )
        if not movies:
            raise SystemExit("No opening-weekend movies matched the requested cohort.")
        min_opening_date = min(movie.opening_date for movie in movies)
        max_opening_date = max(movie.opening_date for movie in movies)
        daily_grosses = opening.load_daily_grosses(
            conn,
            start_date=min_opening_date + dt.timedelta(days=min(snapshot_days) - 7),
            end_date=max_opening_date + dt.timedelta(days=max(snapshot_days)),
        )
        wiki_by_movie = opening.load_wiki_feature_map(
            conn,
            movies=movies,
            timing_days=snapshot_days,
        )
        bop_forecasts = opening.load_boxofficepro_forecasts(
            conn,
            min_target_date=min_opening_date,
            max_target_date=max_opening_date,
        )
    finally:
        conn.close()

    panel_rows = build_day_by_day_feature_panel(
        movies,
        daily_grosses,
        wiki_by_movie,
        bop_forecasts,
        snapshot_days=snapshot_days,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
    )
    prediction_rows, metric_rows, coefficient_rows, interval_rows = evaluate_day_by_day_snapshots(
        panel_rows,
        snapshot_days=snapshot_days,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        test_start_year=args.test_start_year,
        test_end_year=args.test_end_year,
    )
    revision_rows = prediction_revision_rows(prediction_rows)
    coverage = coverage_rows(panel_rows)
    out_dir = args.out or DEFAULT_OUT_DIR
    write_outputs(
        out_dir,
        panel_rows=panel_rows,
        prediction_rows=prediction_rows,
        metric_rows=metric_rows,
        coefficient_rows=coefficient_rows,
        interval_rows=interval_rows,
        revision_rows=revision_rows,
        coverage=coverage,
    )
    print(f"Built {len(panel_rows)} day-by-day forecast feature rows for {len(movies)} movies.")
    print(f"Snapshot days: {','.join(str(day) for day in snapshot_days)}.")
    print(f"Train years: {args.train_start_year}-{args.train_end_year}; test years: {args.test_start_year}-{args.test_end_year}.")
    print(f"Cohort filter: opening day >= ${args.min_opening_day_gross:,} (retrospective, not production-safe).")
    print(f"Wrote day-by-day opening-weekend artifacts to {out_dir}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day-by-day opening-weekend forecasts anchored on BOP estimates.")
    parser.add_argument("--database-url", default=database_url_from_env())
    parser.add_argument("--out", type=Path)
    parser.add_argument("--snapshot-days", default=",".join(str(day) for day in DEFAULT_SNAPSHOT_DAYS))
    parser.add_argument("--train-start-year", type=int, default=DEFAULT_TRAIN_START_YEAR)
    parser.add_argument("--train-end-year", type=int, default=DEFAULT_TRAIN_END_YEAR)
    parser.add_argument("--test-start-year", type=int, default=DEFAULT_TEST_START_YEAR)
    parser.add_argument("--test-end-year", type=int, default=DEFAULT_TEST_END_YEAR)
    parser.add_argument("--min-opening-day-gross", type=int, default=DEFAULT_MIN_OPENING_DAY_GROSS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
