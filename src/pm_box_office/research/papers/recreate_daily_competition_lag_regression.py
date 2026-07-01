#!/usr/bin/env python3
"""Daily competition-lag extension grounded in the Einav seasonality results.

This script builds a 2022+ movie-day panel from Postgres and asks whether
yesterday's box office performance of competing movies predicts each movie's
next-day gross after controlling for its own lifecycle, calendar patterns, and
the local Einav season-week effects.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pm_box_office.db.connection import connect_database, database_url_from_env
from pm_box_office.research.papers import recreate_einav_seasonality as einav


DEFAULT_OUT_DIR = Path("results/papers/daily_competition_lag_regression")
DEFAULT_EINAV_DIR = Path("results/papers/einav_seasonality")
DEFAULT_START_DATE = dt.date(2022, 1, 1)
DEFAULT_WIDE_THEATER_THRESHOLD = 600
DEFAULT_MAX_AGE_WEEKS = 10
DEFAULT_SAMPLE = "all-in-theaters"
SAMPLE_CHOICES = ("all-in-theaters", "wide-first-10-weeks", "wide-first-4-weeks")


@dataclass(frozen=True)
class DailyMovieRow:
    movie_id: int
    title: str
    box_office_date: dt.date
    gross_usd: int
    theaters: int


MODEL_TERMS = {
    "lifecycle_baseline": [
        "log_own_prior_day_gross",
        "log_own_prior_week_same_day_gross",
        "age_days",
        "age_days_sq",
        "log_theaters",
        "is_weekend",
        "is_holiday_or_adjacent",
        "weekday_1",
        "weekday_2",
        "weekday_3",
        "weekday_4",
        "weekday_5",
        "weekday_6",
    ],
    "einav_grounded_baseline": [
        "log_own_prior_day_gross",
        "log_own_prior_week_same_day_gross",
        "age_days",
        "age_days_sq",
        "log_theaters",
        "is_weekend",
        "is_holiday_or_adjacent",
        "weekday_1",
        "weekday_2",
        "weekday_3",
        "weekday_4",
        "weekday_5",
        "weekday_6",
        "estimated_underlying_demand_effect",
    ],
    "competition_model": [
        "log_own_prior_day_gross",
        "log_own_prior_week_same_day_gross",
        "age_days",
        "age_days_sq",
        "log_theaters",
        "is_weekend",
        "is_holiday_or_adjacent",
        "weekday_1",
        "weekday_2",
        "weekday_3",
        "weekday_4",
        "weekday_5",
        "weekday_6",
        "estimated_underlying_demand_effect",
        "log_prior_day_competitor_total_gross",
        "log_prior_day_competitor_top1_gross",
        "log_prior_day_competitor_top3_gross",
        "log_prior_day_competitor_top5_gross",
        "prior_day_competitor_count",
        "prior_day_competitor_hhi",
        "prior_day_competitor_market_share_ex_focal",
    ],
    "amplification_interactions": [
        "log_own_prior_day_gross",
        "log_own_prior_week_same_day_gross",
        "age_days",
        "age_days_sq",
        "log_theaters",
        "is_weekend",
        "is_holiday_or_adjacent",
        "weekday_1",
        "weekday_2",
        "weekday_3",
        "weekday_4",
        "weekday_5",
        "weekday_6",
        "estimated_underlying_demand_effect",
        "seasonality_amplification_gap",
        "log_prior_day_competitor_total_gross",
        "log_prior_day_competitor_top1_gross",
        "log_prior_day_competitor_top3_gross",
        "log_prior_day_competitor_top5_gross",
        "prior_day_competitor_count",
        "prior_day_competitor_hhi",
        "prior_day_competitor_market_share_ex_focal",
        "competitor_total_x_amplification_gap",
        "competitor_top1_x_amplification_gap",
        "competitor_hhi_x_amplification_gap",
    ],
}

TEST_MODEL_TERMS = {
    "lifecycle_baseline": MODEL_TERMS["lifecycle_baseline"],
    "lifecycle_einav": MODEL_TERMS["einav_grounded_baseline"],
    "lifecycle_competition": [
        *MODEL_TERMS["lifecycle_baseline"],
        "log_prior_day_competitor_total_gross",
        "log_prior_day_competitor_top1_gross",
        "log_prior_day_competitor_top3_gross",
        "log_prior_day_competitor_top5_gross",
        "prior_day_competitor_count",
        "prior_day_competitor_hhi",
        "prior_day_competitor_market_share_ex_focal",
    ],
    "einav_competition": MODEL_TERMS["competition_model"],
    "einav_competition_interactions": MODEL_TERMS["amplification_interactions"],
}

COMPETITOR_FEATURE_COLUMNS = [
    "prior_day_competitor_total_gross",
    "prior_day_competitor_top1_gross",
    "prior_day_competitor_top3_gross",
    "prior_day_competitor_top5_gross",
    "prior_day_competitor_count",
    "prior_day_competitor_hhi",
    "prior_day_competitor_market_share_ex_focal",
    "log_prior_day_competitor_total_gross",
    "log_prior_day_competitor_top1_gross",
    "log_prior_day_competitor_top3_gross",
    "log_prior_day_competitor_top5_gross",
    "competitor_total_x_amplification_gap",
    "competitor_top1_x_amplification_gap",
    "competitor_hhi_x_amplification_gap",
]

FOCUSED_2026_MODELS = {
    "univariate_competitor_share": ["prior_day_competitor_market_share_ex_focal"],
    "prior_share_baseline": ["logit_own_prior_day_share"],
    "incremental_competitor_share": [
        "logit_own_prior_day_share",
        "prior_day_competitor_market_share_ex_focal",
    ],
}


def sample_settings(sample: str, max_age_weeks: int) -> tuple[bool, int | None, bool]:
    if sample == "all-in-theaters":
        return False, None, False
    if sample == "wide-first-4-weeks":
        return True, 4, True
    if sample == "wide-first-10-weeks":
        return True, 10, True
    if sample == "wide-custom":
        return True, max_age_weeks, True
    raise ValueError(f"unknown sample mode: {sample}")


def parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def date_or_none(value: str | None) -> dt.date | None:
    return parse_date(value) if value else None


def safe_float(value: object, default: float = 0.0) -> float:
    parsed = einav.float_or_none(value)
    return default if parsed is None else parsed


def normal_p_value(t_stat: float) -> float:
    return math.erfc(abs(t_stat) / math.sqrt(2.0))


def rmse(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def mae(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(abs(value) for value in values) / len(values)


def load_database_daily_rows(
    conn: Any,
    *,
    start_date: dt.date,
    end_date: dt.date | None,
) -> list[DailyMovieRow]:
    load_start = start_date - dt.timedelta(days=7)
    params: list[object] = [load_start.isoformat()]
    end_clause = ""
    if end_date is not None:
        end_clause = "AND d.box_office_date::date <= %s"
        params.append(end_date.isoformat())
    rows = conn.execute(
        f"""
        SELECT r.movie_id,
               m.title,
               d.box_office_date,
               MAX(d.gross_usd) AS gross_usd,
               MAX(d.theaters) AS theaters
        FROM daily_box_office d
        JOIN release_runs r USING(release_run_id)
        JOIN movies m USING(movie_id)
        WHERE d.box_office_date::date >= %s
          {end_clause}
          AND d.gross_usd IS NOT NULL
          AND d.theaters IS NOT NULL
          AND d.is_preview = 0
        GROUP BY r.movie_id, m.title, d.box_office_date
        ORDER BY r.movie_id, d.box_office_date
        """,
        params,
    ).fetchall()
    return [
        DailyMovieRow(
            movie_id=int(row[0]),
            title=str(row[1]),
            box_office_date=parse_date(str(row[2])),
            gross_usd=int(row[3]),
            theaters=int(row[4]),
        )
        for row in rows
        if row[3] is not None and int(row[3]) > 0 and row[4] is not None and int(row[4]) > 0
    ]


def is_thanksgiving(value: dt.date) -> bool:
    return value.month == 11 and value.weekday() == 3 and 22 <= value.day <= 28


def is_holiday_or_adjacent(value: dt.date) -> int:
    fixed = {(1, 1), (7, 4), (11, 11), (12, 24), (12, 25), (12, 31)}
    if (value.month, value.day) in fixed:
        return 1
    if value.month == 11 and value.weekday() == 4 and 23 <= value.day <= 29:
        return 1
    if is_thanksgiving(value) or is_thanksgiving(value - dt.timedelta(days=1)):
        return 1
    return 0


def season_week_for_day(value: dt.date) -> str:
    return einav.calendar_week_for_date(einav.week_start_for_date(value))


def build_einav_effect_map(
    observed_rows: list[dict[str, object]],
    estimated_rows: list[dict[str, object]],
) -> dict[str, dict[str, float]]:
    observed = {
        str(row["season_week"]): safe_float(row.get("observed_log_inside_share_effect"))
        for row in observed_rows
        if row.get("season_week") not in (None, "")
    }
    estimated = {
        str(row["season_week"]): safe_float(row.get("estimated_underlying_demand_effect"))
        for row in estimated_rows
        if row.get("season_week") not in (None, "")
    }
    weeks = set(observed) | set(estimated)
    return {
        week: {
            "observed_log_inside_share_effect": observed.get(week, 0.0),
            "estimated_underlying_demand_effect": estimated.get(week, 0.0),
            "seasonality_amplification_gap": observed.get(week, 0.0) - estimated.get(week, 0.0),
        }
        for week in weeks
    }


def load_einav_effect_map(einav_dir: Path) -> dict[str, dict[str, float]]:
    observed_path = einav_dir / "database_2022_plus_observed_weekly_seasonality.csv"
    estimated_path = einav_dir / "database_2022_plus_estimated_week_effects.csv"
    if not observed_path.exists() or not estimated_path.exists():
        raise SystemExit(
            "Run the Einav 2022+ database replication first; missing "
            f"{observed_path.name} or {estimated_path.name}."
        )
    return build_einav_effect_map(
        einav.read_csv_rows(observed_path),
        einav.read_csv_rows(estimated_path),
    )


def aggregate_daily_rows(rows: list[DailyMovieRow]) -> list[DailyMovieRow]:
    grouped: dict[tuple[int, dt.date], list[DailyMovieRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.movie_id, row.box_office_date)].append(row)
    aggregated = []
    for (_movie_id, _day), bucket in grouped.items():
        aggregated.append(
            DailyMovieRow(
                movie_id=bucket[0].movie_id,
                title=bucket[0].title,
                box_office_date=bucket[0].box_office_date,
                gross_usd=max(row.gross_usd for row in bucket),
                theaters=max(row.theaters for row in bucket),
            )
        )
    return sorted(aggregated, key=lambda row: (row.box_office_date, row.movie_id))


def prior_day_competitor_features(
    gross_by_movie_day: dict[tuple[int, dt.date], DailyMovieRow],
    *,
    movie_id: int,
    prior_day: dt.date,
    movie_ids: set[int],
) -> dict[str, float]:
    competitor_grosses = [
        float(row.gross_usd)
        for other_movie_id in movie_ids
        if other_movie_id != movie_id
        for row in [gross_by_movie_day.get((other_movie_id, prior_day))]
        if row is not None and row.gross_usd > 0
    ]
    competitor_grosses.sort(reverse=True)
    total = sum(competitor_grosses)
    top1 = competitor_grosses[0] if competitor_grosses else 0.0
    top3 = sum(competitor_grosses[:3])
    top5 = sum(competitor_grosses[:5])
    count = float(len(competitor_grosses))
    hhi = sum((gross / total) ** 2 for gross in competitor_grosses) if total > 0 else 0.0
    own_prior = gross_by_movie_day.get((movie_id, prior_day))
    own_prior_gross = float(own_prior.gross_usd) if own_prior is not None else 0.0
    market_total = total + own_prior_gross
    return {
        "prior_day_competitor_total_gross": total,
        "prior_day_competitor_top1_gross": top1,
        "prior_day_competitor_top3_gross": top3,
        "prior_day_competitor_top5_gross": top5,
        "prior_day_competitor_count": count,
        "prior_day_competitor_hhi": hhi,
        "prior_day_competitor_market_share_ex_focal": total / market_total if market_total > 0 else 0.0,
    }


def build_daily_movie_panel(
    rows: list[DailyMovieRow],
    *,
    start_date: dt.date,
    end_date: dt.date | None,
    wide_theater_threshold: int,
    max_age_weeks: int,
    einav_effects: dict[str, dict[str, float]],
    sample: str = DEFAULT_SAMPLE,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = aggregate_daily_rows(rows)
    apply_wide_filter, effective_max_age_weeks, require_complete_prior_day = sample_settings(sample, max_age_weeks)
    first_date_by_movie: dict[int, dt.date] = {}
    max_theaters_by_movie: dict[int, int] = defaultdict(int)
    for row in rows:
        first_date_by_movie[row.movie_id] = min(first_date_by_movie.get(row.movie_id, row.box_office_date), row.box_office_date)
        max_theaters_by_movie[row.movie_id] = max(max_theaters_by_movie[row.movie_id], row.theaters)
    eligible_movie_ids = set(first_date_by_movie)
    if apply_wide_filter:
        eligible_movie_ids = {
            movie_id
            for movie_id, max_theaters in max_theaters_by_movie.items()
            if max_theaters >= wide_theater_threshold
        }
    gross_by_movie_day = {(row.movie_id, row.box_office_date): row for row in rows}
    movie_ids = set(eligible_movie_ids)
    max_age_days = effective_max_age_weeks * 7 if effective_max_age_weeks is not None else None
    total_theaters_by_day: dict[dt.date, float] = defaultdict(float)
    for row in rows:
        if row.movie_id in eligible_movie_ids:
            total_theaters_by_day[row.box_office_date] += float(row.theaters)

    panel_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    for row in rows:
        if row.movie_id not in eligible_movie_ids:
            continue
        if row.box_office_date < start_date:
            continue
        if end_date is not None and row.box_office_date > end_date:
            continue
        release_date = first_date_by_movie[row.movie_id]
        age_days = (row.box_office_date - release_date).days
        if age_days < 0:
            continue
        if max_age_days is not None:
            if require_complete_prior_day and age_days <= 0:
                continue
            if age_days >= max_age_days:
                continue
        prior_day = row.box_office_date - dt.timedelta(days=1)
        own_prior = gross_by_movie_day.get((row.movie_id, prior_day))
        if require_complete_prior_day and (own_prior is None or own_prior.gross_usd <= 0):
            continue
        own_prior_week = gross_by_movie_day.get((row.movie_id, row.box_office_date - dt.timedelta(days=7)))
        season_week = season_week_for_day(row.box_office_date)
        effects = einav_effects.get(
            season_week,
            {
                "observed_log_inside_share_effect": 0.0,
                "estimated_underlying_demand_effect": 0.0,
                "seasonality_amplification_gap": 0.0,
            },
        )
        competitor = prior_day_competitor_features(
            gross_by_movie_day,
            movie_id=row.movie_id,
            prior_day=prior_day,
            movie_ids=movie_ids,
        )
        weekday = row.box_office_date.weekday()
        own_prior_day_gross = float(own_prior.gross_usd) if own_prior is not None else 0.0
        own_prior_week_gross = float(own_prior_week.gross_usd) if own_prior_week is not None else 0.0
        total_theaters_today = total_theaters_by_day.get(row.box_office_date, 0.0)
        total_theaters_prior_day = total_theaters_by_day.get(prior_day, 0.0)
        prior_day_theaters = float(own_prior.theaters) if own_prior is not None else 0.0
        panel_row = {
            "sample": sample,
            "movie_id": str(row.movie_id),
            "title": row.title,
            "box_office_date": row.box_office_date.isoformat(),
            "season_week": season_week,
            "target_gross_usd": float(row.gross_usd),
            "target_log_gross": math.log1p(float(row.gross_usd)),
            "own_prior_day_gross": own_prior_day_gross,
            "own_prior_day_missing": 1.0 if own_prior is None else 0.0,
            "own_prior_week_same_day_gross": own_prior_week_gross,
            "own_prior_week_same_day_missing": 1.0 if own_prior_week is None else 0.0,
            "log_own_prior_day_gross": math.log1p(own_prior_day_gross),
            "log_own_prior_week_same_day_gross": math.log1p(own_prior_week_gross),
            "age_days": float(age_days),
            "age_days_sq": float(age_days * age_days),
            "theaters": float(row.theaters),
            "log_theaters": math.log1p(float(row.theaters)),
            "total_active_theaters": total_theaters_today,
            "theater_share": float(row.theaters) / total_theaters_today if total_theaters_today > 0 else 0.0,
            "prior_day_theaters": prior_day_theaters,
            "log_prior_day_theaters": math.log1p(prior_day_theaters),
            "prior_day_theater_share": prior_day_theaters / total_theaters_prior_day if total_theaters_prior_day > 0 else 0.0,
            "prior_day_theater_missing": 1.0 if own_prior is None else 0.0,
            "weekday": weekday,
            "is_weekend": 1.0 if weekday in {4, 5, 6} else 0.0,
            "is_holiday_or_adjacent": float(is_holiday_or_adjacent(row.box_office_date)),
            "observed_log_inside_share_effect": effects["observed_log_inside_share_effect"],
            "estimated_underlying_demand_effect": effects["estimated_underlying_demand_effect"],
            "seasonality_amplification_gap": effects["seasonality_amplification_gap"],
            **competitor,
        }
        for idx in range(1, 7):
            panel_row[f"weekday_{idx}"] = 1.0 if weekday == idx else 0.0
        for key in [
            "prior_day_competitor_total_gross",
            "prior_day_competitor_top1_gross",
            "prior_day_competitor_top3_gross",
            "prior_day_competitor_top5_gross",
        ]:
            panel_row[f"log_{key}"] = math.log1p(float(panel_row[key]))
        panel_row["competitor_total_x_amplification_gap"] = (
            float(panel_row["log_prior_day_competitor_total_gross"])
            * float(panel_row["seasonality_amplification_gap"])
        )
        panel_row["competitor_top1_x_amplification_gap"] = (
            float(panel_row["log_prior_day_competitor_top1_gross"])
            * float(panel_row["seasonality_amplification_gap"])
        )
        panel_row["competitor_hhi_x_amplification_gap"] = (
            float(panel_row["prior_day_competitor_hhi"])
            * float(panel_row["seasonality_amplification_gap"])
        )
        panel_rows.append(panel_row)
        feature_rows.append(
            {
                key: panel_row[key]
                for key in [
                    "movie_id",
                    "box_office_date",
                    "prior_day_competitor_total_gross",
                    "prior_day_competitor_top1_gross",
                    "prior_day_competitor_top3_gross",
                    "prior_day_competitor_top5_gross",
                    "prior_day_competitor_count",
                    "prior_day_competitor_hhi",
                    "prior_day_competitor_market_share_ex_focal",
                    "theater_share",
                    "prior_day_theater_share",
                    "seasonality_amplification_gap",
                ]
            }
        )
    panel_rows.sort(key=lambda item: (str(item["box_office_date"]), str(item["movie_id"])))
    feature_rows.sort(key=lambda item: (str(item["box_office_date"]), str(item["movie_id"])))
    return panel_rows, feature_rows


def chronological_split(rows: list[dict[str, object]], train_fraction: float = 0.8) -> tuple[list[dict[str, object]], list[dict[str, object]], str]:
    dates = sorted({str(row["box_office_date"]) for row in rows})
    if len(dates) < 2:
        return rows, [], dates[-1] if dates else ""
    cutoff_index = min(len(dates) - 1, max(1, int(len(dates) * train_fraction)))
    cutoff_date = dates[cutoff_index - 1]
    train = [row for row in rows if str(row["box_office_date"]) <= cutoff_date]
    holdout = [row for row in rows if str(row["box_office_date"]) > cutoff_date]
    return train, holdout, cutoff_date


def design_matrix(rows: list[dict[str, object]], terms: list[str]) -> list[list[float]]:
    return [[1.0] + [float(row.get(term, 0.0) or 0.0) for term in terms] for row in rows]


def predict(beta: list[float], row: dict[str, object], terms: list[str]) -> float:
    values = [1.0] + [float(row.get(term, 0.0) or 0.0) for term in terms]
    return sum(coef * value for coef, value in zip(beta, values))


def fit_model(
    model_name: str,
    rows: list[dict[str, object]],
    terms: list[str],
) -> tuple[list[dict[str, object]], list[float], float, float]:
    if len(rows) < len(terms) + 3:
        raise SystemExit(f"Not enough rows to estimate {model_name}.")
    x = design_matrix(rows, terms)
    y = [float(row["target_log_gross"]) for row in rows]
    beta, se, r2, sse = einav.ols(x, y)
    coefficient_rows = []
    for term, coef, stderr in zip(["intercept"] + terms, beta, se):
        t_stat = coef / stderr if stderr else 0.0
        coefficient_rows.append(
            {
                "model": model_name,
                "term": term,
                "coef": coef,
                "se_naive": stderr,
                "t_stat": t_stat,
                "p_normal_approx": normal_p_value(t_stat),
                "n": len(rows),
                "r2_train": r2,
                "sse_train": sse,
            }
        )
    return coefficient_rows, beta, r2, sse


def evaluate_model(
    model_name: str,
    beta: list[float],
    terms: list[str],
    train_rows: list[dict[str, object]],
    holdout_rows: list[dict[str, object]],
) -> dict[str, object]:
    train_residuals = [float(row["target_log_gross"]) - predict(beta, row, terms) for row in train_rows]
    holdout_residuals = [float(row["target_log_gross"]) - predict(beta, row, terms) for row in holdout_rows]
    y_holdout = [float(row["target_log_gross"]) for row in holdout_rows]
    holdout_tss = sum((value - einav.mean(y_holdout)) ** 2 for value in y_holdout) if y_holdout else 0.0
    holdout_sse = sum(value * value for value in holdout_residuals)
    return {
        "model": model_name,
        "train_n": len(train_rows),
        "holdout_n": len(holdout_rows),
        "train_rmse_log": rmse(train_residuals),
        "train_mae_log": mae(train_residuals),
        "holdout_rmse_log": rmse(holdout_residuals),
        "holdout_mae_log": mae(holdout_residuals),
        "holdout_r2": 1.0 - holdout_sse / holdout_tss if holdout_tss else 0.0,
    }


def run_models(
    panel_rows: list[dict[str, object]],
) -> tuple[dict[str, list[dict[str, object]]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    train_rows, holdout_rows, cutoff_date = chronological_split(panel_rows)
    coefficients_by_model: dict[str, list[dict[str, object]]] = {}
    comparison_rows = []
    predictions = []
    for model_name, terms in MODEL_TERMS.items():
        coefficient_rows, beta, train_r2, _sse = fit_model(model_name, train_rows, terms)
        coefficients_by_model[model_name] = coefficient_rows
        metrics = evaluate_model(model_name, beta, terms, train_rows, holdout_rows)
        metrics["train_r2"] = train_r2
        metrics["split_cutoff_date"] = cutoff_date
        comparison_rows.append(metrics)
        if model_name == "amplification_interactions":
            for row in holdout_rows:
                predictions.append(
                    {
                        "movie_id": row["movie_id"],
                        "title": row["title"],
                        "box_office_date": row["box_office_date"],
                        "actual_log_gross": row["target_log_gross"],
                        "predicted_log_gross": predict(beta, row, terms),
                        "actual_gross_usd": row["target_gross_usd"],
                        "predicted_gross_usd": max(0.0, math.expm1(predict(beta, row, terms))),
                    }
                )
    hypothesis_rows = hypothesis_test_rows(coefficients_by_model, comparison_rows)
    return coefficients_by_model, comparison_rows, hypothesis_rows, predictions


def coefficient_lookup(coefficients_by_model: dict[str, list[dict[str, object]]], model: str, term: str) -> dict[str, object] | None:
    for row in coefficients_by_model.get(model, []):
        if row["term"] == term:
            return row
    return None


def metric_lookup(comparison_rows: list[dict[str, object]], model: str, metric: str) -> float:
    for row in comparison_rows:
        if row["model"] == model:
            return float(row[metric])
    return 0.0


def hypothesis_test_rows(
    coefficients_by_model: dict[str, list[dict[str, object]]],
    comparison_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    underlying = coefficient_lookup(coefficients_by_model, "einav_grounded_baseline", "estimated_underlying_demand_effect")
    competitor_total = coefficient_lookup(coefficients_by_model, "competition_model", "log_prior_day_competitor_total_gross")
    interaction = coefficient_lookup(coefficients_by_model, "amplification_interactions", "competitor_total_x_amplification_gap")
    top1 = coefficient_lookup(coefficients_by_model, "competition_model", "log_prior_day_competitor_top1_gross")
    count = coefficient_lookup(coefficients_by_model, "competition_model", "prior_day_competitor_count")
    return [
        {
            "hypothesis": "H1_einav_effects_improve_prediction",
            "metric": "holdout_rmse_reduction_vs_lifecycle",
            "value": metric_lookup(comparison_rows, "lifecycle_baseline", "holdout_rmse_log")
            - metric_lookup(comparison_rows, "einav_grounded_baseline", "holdout_rmse_log"),
            "supporting_term": "estimated_underlying_demand_effect",
            "coef": underlying.get("coef") if underlying else None,
            "p_normal_approx": underlying.get("p_normal_approx") if underlying else None,
        },
        {
            "hypothesis": "H2_competition_features_improve_prediction",
            "metric": "holdout_rmse_reduction_vs_einav_grounded",
            "value": metric_lookup(comparison_rows, "einav_grounded_baseline", "holdout_rmse_log")
            - metric_lookup(comparison_rows, "competition_model", "holdout_rmse_log"),
            "supporting_term": "competitor_feature_block",
            "coef": None,
            "p_normal_approx": None,
        },
        {
            "hypothesis": "H3_stronger_competitors_reduce_focal_next_day_gross",
            "metric": "competitor_total_log_coef",
            "value": competitor_total.get("coef") if competitor_total else None,
            "supporting_term": "log_prior_day_competitor_total_gross",
            "coef": competitor_total.get("coef") if competitor_total else None,
            "p_normal_approx": competitor_total.get("p_normal_approx") if competitor_total else None,
        },
        {
            "hypothesis": "H4_competition_effect_stronger_in_high_amplification_weeks",
            "metric": "competitor_total_x_amplification_gap_coef",
            "value": interaction.get("coef") if interaction else None,
            "supporting_term": "competitor_total_x_amplification_gap",
            "coef": interaction.get("coef") if interaction else None,
            "p_normal_approx": interaction.get("p_normal_approx") if interaction else None,
        },
        {
            "hypothesis": "H5_top_heavy_competition_more_informative_than_count",
            "metric": "abs_top1_t_minus_abs_count_t",
            "value": (abs(float(top1.get("t_stat", 0.0))) if top1 else 0.0)
            - (abs(float(count.get("t_stat", 0.0))) if count else 0.0),
            "supporting_term": "log_prior_day_competitor_top1_gross_vs_prior_day_competitor_count",
            "coef": top1.get("coef") if top1 else None,
            "p_normal_approx": top1.get("p_normal_approx") if top1 else None,
        },
    ]


def write_prediction_svg(path: Path, predictions: list[dict[str, object]]) -> None:
    if not predictions:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"700\" height=\"300\"></svg>\n", encoding="utf-8")
        return
    sample = predictions[:: max(1, len(predictions) // 450)]
    width, height = 760, 520
    left, right, top, bottom = 70, 30, 60, 455
    actuals = [float(row["actual_log_gross"]) for row in sample]
    preds = [float(row["predicted_log_gross"]) for row in sample]
    lo = min(actuals + preds)
    hi = max(actuals + preds)
    if lo == hi:
        lo -= 1
        hi += 1

    def scale_x(value: float) -> float:
        return left + (value - lo) / (hi - lo) * (width - left - right)

    def scale_y(value: float) -> float:
        return bottom - (value - lo) / (hi - lo) * (bottom - top)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">Daily actual vs predicted log box office</text>',
        '<text x="36" y="55" font-family="Arial" font-size="13" fill="#555">Holdout predictions from the amplification interaction model</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{width - right}" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{scale_x(lo):.1f}" y1="{scale_y(lo):.1f}" x2="{scale_x(hi):.1f}" y2="{scale_y(hi):.1f}" stroke="#999" stroke-dasharray="4 4"/>',
    ]
    for actual, pred in zip(actuals, preds):
        parts.append(f'<circle cx="{scale_x(actual):.1f}" cy="{scale_y(pred):.1f}" r="3" fill="#2f6f73" fill-opacity="0.45"/>')
    parts.extend(
        [
            f'<text x="{(left + width - right) / 2:.1f}" y="{height - 18}" font-family="Arial" font-size="12" text-anchor="middle">Actual log1p(gross)</text>',
            f'<text x="18" y="{(top + bottom) / 2:.1f}" font-family="Arial" font-size="12" transform="rotate(-90 18 {(top + bottom) / 2:.1f})" text-anchor="middle">Predicted log1p(gross)</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def write_competitor_effect_svg(path: Path, coefficients_by_model: dict[str, list[dict[str, object]]], panel_rows: list[dict[str, object]]) -> None:
    base = coefficient_lookup(coefficients_by_model, "amplification_interactions", "log_prior_day_competitor_total_gross")
    interaction = coefficient_lookup(coefficients_by_model, "amplification_interactions", "competitor_total_x_amplification_gap")
    if base is None or interaction is None or not panel_rows:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"700\" height=\"300\"></svg>\n", encoding="utf-8")
        return
    gaps = [float(row["seasonality_amplification_gap"]) for row in panel_rows]
    lo, hi = min(gaps), max(gaps)
    if lo == hi:
        lo -= 1
        hi += 1
    values = []
    for index in range(40):
        gap = lo + (hi - lo) * index / 39
        effect = float(base["coef"]) + float(interaction["coef"]) * gap
        values.append((gap, effect))
    y_values = [value for _gap, value in values]
    y_lo, y_hi = min(y_values), max(y_values)
    if y_lo == y_hi:
        y_lo -= 1
        y_hi += 1
    width, height = 760, 430
    left, right, top, bottom = 80, 35, 60, 350

    def sx(value: float) -> float:
        return left + (value - lo) / (hi - lo) * (width - left - right)

    def sy(value: float) -> float:
        return bottom - (value - y_lo) / (y_hi - y_lo) * (bottom - top)

    points = " ".join(f"{sx(gap):.1f},{sy(effect):.1f}" for gap, effect in values)
    zero_y = sy(0.0) if y_lo <= 0.0 <= y_hi else None
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">Competitor effect by Einav amplification gap</text>',
        '<text x="36" y="55" font-family="Arial" font-size="13" fill="#555">Marginal coefficient on prior-day total competitor gross</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{width - right}" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>',
    ]
    if zero_y is not None:
        parts.append(f'<line x1="{left}" y1="{zero_y:.1f}" x2="{width - right}" y2="{zero_y:.1f}" stroke="#aaa" stroke-dasharray="4 4"/>')
    parts.append(f'<polyline points="{points}" fill="none" stroke="#c15b3f" stroke-width="2.5"/>')
    parts.extend(
        [
            f'<text x="{(left + width - right) / 2:.1f}" y="{height - 22}" font-family="Arial" font-size="12" text-anchor="middle">Observed minus estimated season-week effect</text>',
            f'<text x="20" y="{(top + bottom) / 2:.1f}" font-family="Arial" font-size="12" transform="rotate(-90 20 {(top + bottom) / 2:.1f})" text-anchor="middle">Marginal log-gross coefficient</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def clamp_probability(value: float, epsilon: float = 1e-9) -> float:
    return min(1.0 - epsilon, max(epsilon, value))


def logit(value: float) -> float:
    value = clamp_probability(value)
    return math.log(value / (1.0 - value))


def logistic(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return clamp_probability(1.0 / (1.0 + z))
    z = math.exp(value)
    return clamp_probability(z / (1.0 + z))


def enrich_share_targets(panel_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    industry_by_date: dict[str, float] = defaultdict(float)
    for row in panel_rows:
        industry_by_date[str(row["box_office_date"])] += float(row["target_gross_usd"])
    gross_by_movie_date = {
        (str(row["movie_id"]), str(row["box_office_date"])): float(row["target_gross_usd"])
        for row in panel_rows
    }
    enriched = []
    for row in panel_rows:
        out = dict(row)
        day = str(row["box_office_date"])
        current_date = parse_date(day)
        industry_gross = industry_by_date[day]
        share = float(row["target_gross_usd"]) / industry_gross if industry_gross > 0 else 0.0
        prior_total = float(row["own_prior_day_gross"]) + float(row["prior_day_competitor_total_gross"])
        prior_share = float(row["own_prior_day_gross"]) / prior_total if prior_total > 0 else 0.0
        prior_week_date = (current_date - dt.timedelta(days=7)).isoformat()
        prior_week_gross = gross_by_movie_date.get((str(row["movie_id"]), prior_week_date), 0.0)
        prior_week_total = industry_by_date.get(prior_week_date, 0.0)
        prior_week_share = prior_week_gross / prior_week_total if prior_week_total > 0 else 0.0
        out["daily_industry_gross_usd"] = industry_gross
        out["daily_market_share"] = share
        out["logit_daily_share"] = logit(share)
        out["own_prior_day_market_share"] = prior_share
        out["logit_own_prior_day_share"] = logit(prior_share)
        out["own_prior_week_same_day_share"] = prior_week_share
        out["logit_own_prior_week_same_day_share"] = logit(prior_week_share) if prior_week_share > 0 else 0.0
        enriched.append(out)
    return enriched


def share_terms_for(terms: list[str]) -> list[str]:
    replacements = {
        "log_own_prior_day_gross": "logit_own_prior_day_share",
        "log_own_prior_week_same_day_gross": "logit_own_prior_week_same_day_share",
    }
    return [replacements.get(term, term) for term in terms]


def target_values(rows: list[dict[str, object]], target_col: str) -> list[float]:
    return [float(row[target_col]) for row in rows]


def fit_model_for_target(
    model_name: str,
    rows: list[dict[str, object]],
    terms: list[str],
    target_col: str,
) -> tuple[list[dict[str, object]], list[float], float, float]:
    if len(rows) < len(terms) + 3:
        raise SystemExit(f"Not enough rows to estimate {model_name} on {target_col}.")
    x = design_matrix(rows, terms)
    y = target_values(rows, target_col)
    beta, se, r2, sse = einav.ols(x, y)
    coefficient_rows = []
    for term, coef, stderr in zip(["intercept"] + terms, beta, se):
        t_stat = coef / stderr if stderr else 0.0
        coefficient_rows.append(
            {
                "model": model_name,
                "target": target_col,
                "term": term,
                "coef": coef,
                "se_naive": stderr,
                "t_stat": t_stat,
                "p_normal_approx": normal_p_value(t_stat),
                "n": len(rows),
                "r2_train": r2,
                "sse_train": sse,
            }
        )
    return coefficient_rows, beta, r2, sse


def prediction_baseline(row: dict[str, object], target_col: str) -> float:
    if target_col == "target_log_gross":
        return float(row["log_own_prior_day_gross"])
    if target_col == "logit_daily_share":
        return float(row["logit_own_prior_day_share"])
    return 0.0


def evaluate_model_for_target(
    model_name: str,
    beta: list[float],
    terms: list[str],
    train_rows: list[dict[str, object]],
    holdout_rows: list[dict[str, object]],
    *,
    target_col: str,
) -> dict[str, object]:
    train_residuals = [float(row[target_col]) - predict(beta, row, terms) for row in train_rows]
    holdout_residuals = [float(row[target_col]) - predict(beta, row, terms) for row in holdout_rows]
    y_holdout = target_values(holdout_rows, target_col)
    holdout_tss = sum((value - einav.mean(y_holdout)) ** 2 for value in y_holdout) if y_holdout else 0.0
    holdout_sse = sum(value * value for value in holdout_residuals)
    direction_hits = 0
    direction_count = 0
    for row in holdout_rows:
        actual_delta = float(row[target_col]) - prediction_baseline(row, target_col)
        predicted_delta = predict(beta, row, terms) - prediction_baseline(row, target_col)
        if actual_delta == 0.0:
            continue
        direction_count += 1
        if (actual_delta > 0) == (predicted_delta > 0):
            direction_hits += 1
    return {
        "model": model_name,
        "target": target_col,
        "train_n": len(train_rows),
        "holdout_n": len(holdout_rows),
        "train_rmse": rmse(train_residuals),
        "train_mae": mae(train_residuals),
        "holdout_rmse": rmse(holdout_residuals),
        "holdout_mae": mae(holdout_residuals),
        "holdout_r2": 1.0 - holdout_sse / holdout_tss if holdout_tss else 0.0,
        "directional_accuracy": direction_hits / direction_count if direction_count else None,
    }


def rolling_windows(rows: list[dict[str, object]]) -> list[tuple[str, dt.date, int]]:
    years = sorted({parse_date(str(row["box_office_date"])).year for row in rows})
    windows = []
    for test_year in years:
        if test_year <= 2023:
            continue
        if any(parse_date(str(row["box_office_date"])).year < test_year for row in rows):
            windows.append((f"train_through_{test_year - 1}_test_{test_year}", dt.date(test_year, 1, 1), test_year))
    return windows


def split_for_test_year(
    rows: list[dict[str, object]],
    *,
    test_year: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train = [row for row in rows if parse_date(str(row["box_office_date"])).year < test_year]
    holdout = [row for row in rows if parse_date(str(row["box_office_date"])).year == test_year]
    return train, holdout


def rolling_backtest_metrics(
    panel_rows: list[dict[str, object]],
    *,
    target_col: str = "target_log_gross",
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = enrich_share_targets(panel_rows) if target_col == "logit_daily_share" else [dict(row) for row in panel_rows]
    metric_rows = []
    prediction_rows = []
    for window_name, _test_start, test_year in rolling_windows(rows):
        train_rows, holdout_rows = split_for_test_year(rows, test_year=test_year)
        if len(train_rows) < 50 or len(holdout_rows) < 10:
            continue
        for model_name, base_terms in TEST_MODEL_TERMS.items():
            terms = share_terms_for(base_terms) if target_col == "logit_daily_share" else base_terms
            coefficient_rows, beta, train_r2, _sse = fit_model_for_target(model_name, train_rows, terms, target_col)
            metrics = evaluate_model_for_target(
                model_name,
                beta,
                terms,
                train_rows,
                holdout_rows,
                target_col=target_col,
            )
            metrics["window"] = window_name
            metrics["test_year"] = test_year
            metrics["train_r2"] = train_r2
            metric_rows.append(metrics)
            if model_name == "einav_competition_interactions":
                for row in holdout_rows:
                    prediction_rows.append(
                        {
                            "window": window_name,
                            "target": target_col,
                            "model": model_name,
                            "movie_id": row["movie_id"],
                            "title": row["title"],
                            "box_office_date": row["box_office_date"],
                            "actual": row[target_col],
                            "predicted": predict(beta, row, terms),
                        }
                    )
            if model_name == "einav_competition" and target_col == "logit_daily_share":
                for row in coefficient_rows:
                    row["window"] = window_name
                    row["test_year"] = test_year
                    prediction_rows.append({"_coefficient_row": row})
    coefficient_rows = [row.pop("_coefficient_row") for row in prediction_rows if "_coefficient_row" in row]
    prediction_rows = [row for row in prediction_rows if "_coefficient_row" not in row]
    return metric_rows, coefficient_rows + prediction_rows


def age_segment(age_days: float) -> str:
    if age_days <= 6:
        return "opening_week"
    if age_days <= 13:
        return "week_2"
    if age_days <= 27:
        return "weeks_3_4"
    return "weeks_5_10"


def release_age_bucket(age_days: float) -> str:
    if age_days <= 0:
        return "opening_day"
    if age_days <= 3:
        return "days_1_3"
    if age_days <= 7:
        return "days_4_7"
    if age_days <= 13:
        return "week_2"
    if age_days <= 27:
        return "weeks_3_4"
    if age_days <= 55:
        return "weeks_5_8"
    return "weeks_9_plus"


RELEASE_AGE_BUCKET_ORDER = [
    "opening_day",
    "days_1_3",
    "days_4_7",
    "week_2",
    "weeks_3_4",
    "weeks_5_8",
    "weeks_9_plus",
]


def release_age_bucket_key(bucket: str) -> int:
    try:
        return RELEASE_AGE_BUCKET_ORDER.index(bucket)
    except ValueError:
        return len(RELEASE_AGE_BUCKET_ORDER)


def release_age_segment_metrics(panel_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = [dict(row) for row in panel_rows]
    output = []
    for window_name, _test_start, test_year in rolling_windows(rows):
        train_rows, holdout_rows = split_for_test_year(rows, test_year=test_year)
        if len(train_rows) < 50 or len(holdout_rows) < 10:
            continue
        fitted = {}
        for model_name in ("lifecycle_baseline", "einav_competition"):
            terms = TEST_MODEL_TERMS[model_name]
            _coef, beta, _r2, _sse = fit_model_for_target(model_name, train_rows, terms, "target_log_gross")
            fitted[model_name] = (beta, terms)
        for segment in ("opening_week", "week_2", "weeks_3_4", "weeks_5_10"):
            segment_rows = [row for row in holdout_rows if age_segment(float(row["age_days"])) == segment]
            if not segment_rows:
                continue
            for model_name, (beta, terms) in fitted.items():
                metrics = evaluate_model_for_target(
                    model_name,
                    beta,
                    terms,
                    train_rows,
                    segment_rows,
                    target_col="target_log_gross",
                )
                output.append(
                    {
                        **metrics,
                        "window": window_name,
                        "test_year": test_year,
                        "release_age_segment": segment,
                    }
                )
    return output


def daily_industry_rows(panel_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_date: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in panel_rows:
        by_date[str(row["box_office_date"])].append(row)
    dates = sorted(by_date)
    gross_by_date = {day: sum(float(row["target_gross_usd"]) for row in rows) for day, rows in by_date.items()}
    output = []
    for day in dates:
        date_value = parse_date(day)
        prior_day = (date_value - dt.timedelta(days=1)).isoformat()
        week_prior = (date_value - dt.timedelta(days=7)).isoformat()
        season_week = season_week_for_day(date_value)
        sample = by_date[day][0]
        weekday = date_value.weekday()
        row = {
            "box_office_date": day,
            "target_log_industry_gross": math.log1p(gross_by_date[day]),
            "log_prior_day_industry_gross": math.log1p(gross_by_date.get(prior_day, 0.0)),
            "log_prior_week_same_day_industry_gross": math.log1p(gross_by_date.get(week_prior, 0.0)),
            "season_week": season_week,
            "estimated_underlying_demand_effect": sample["estimated_underlying_demand_effect"],
            "seasonality_amplification_gap": sample["seasonality_amplification_gap"],
            "is_weekend": 1.0 if weekday in {4, 5, 6} else 0.0,
            "is_holiday_or_adjacent": float(is_holiday_or_adjacent(date_value)),
        }
        for idx in range(1, 7):
            row[f"weekday_{idx}"] = 1.0 if weekday == idx else 0.0
        output.append(row)
    return output


def two_stage_forecast_metrics(panel_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = enrich_share_targets(panel_rows)
    industry_rows = daily_industry_rows(rows)
    industry_terms = [
        "log_prior_day_industry_gross",
        "log_prior_week_same_day_industry_gross",
        "estimated_underlying_demand_effect",
        "seasonality_amplification_gap",
        "is_weekend",
        "is_holiday_or_adjacent",
        "weekday_1",
        "weekday_2",
        "weekday_3",
        "weekday_4",
        "weekday_5",
        "weekday_6",
    ]
    share_terms = share_terms_for(TEST_MODEL_TERMS["einav_competition"])
    output = []
    for window_name, _test_start, test_year in rolling_windows(rows):
        train_rows, holdout_rows = split_for_test_year(rows, test_year=test_year)
        train_dates = [row for row in industry_rows if parse_date(str(row["box_office_date"])).year < test_year]
        holdout_dates = [row for row in industry_rows if parse_date(str(row["box_office_date"])).year == test_year]
        if len(train_rows) < 50 or len(holdout_rows) < 10 or len(train_dates) < len(industry_terms) + 3:
            continue
        _industry_coef, industry_beta, _r2, _sse = fit_model_for_target(
            "industry_total",
            train_dates,
            industry_terms,
            "target_log_industry_gross",
        )
        _share_coef, share_beta, _share_r2, _share_sse = fit_model_for_target(
            "share_stage",
            train_rows,
            share_terms,
            "logit_daily_share",
        )
        industry_prediction_by_date = {
            str(row["box_office_date"]): max(0.0, math.expm1(predict(industry_beta, row, industry_terms)))
            for row in holdout_dates
        }
        residuals = []
        gross_residuals = []
        for row in holdout_rows:
            predicted_share = logistic(predict(share_beta, row, share_terms))
            predicted_gross = predicted_share * industry_prediction_by_date.get(str(row["box_office_date"]), 0.0)
            residuals.append(float(row["target_log_gross"]) - math.log1p(predicted_gross))
            gross_residuals.append(float(row["target_gross_usd"]) - predicted_gross)
        y_holdout = [float(row["target_log_gross"]) for row in holdout_rows]
        tss = sum((value - einav.mean(y_holdout)) ** 2 for value in y_holdout) if y_holdout else 0.0
        sse = sum(value * value for value in residuals)
        output.append(
            {
                "window": window_name,
                "test_year": test_year,
                "model": "two_stage_industry_total_x_share",
                "train_n": len(train_rows),
                "holdout_n": len(holdout_rows),
                "holdout_rmse_log_gross": rmse(residuals),
                "holdout_mae_log_gross": mae(residuals),
                "holdout_r2_log_gross": 1.0 - sse / tss if tss else 0.0,
                "holdout_mae_gross_usd": mae(gross_residuals),
            }
        )
    return output


def validation_check_rows(
    raw_rows: list[DailyMovieRow],
    panel_rows: list[dict[str, object]],
    *,
    start_date: dt.date,
    wide_theater_threshold: int,
    max_age_weeks: int,
    sample: str = DEFAULT_SAMPLE,
) -> list[dict[str, object]]:
    rows = []
    panel_keys = {(str(row["movie_id"]), str(row["box_office_date"])) for row in panel_rows}
    source_by_key = {(str(row.movie_id), row.box_office_date.isoformat()): row for row in aggregate_daily_rows(raw_rows)}
    panel_source_total = sum(float(source_by_key[key].gross_usd) for key in panel_keys if key in source_by_key)
    panel_total = sum(float(row["target_gross_usd"]) for row in panel_rows)
    _apply_wide, effective_max_age_weeks, _require_prior = sample_settings(sample, max_age_weeks)
    checks = [
        ("nonempty_panel", len(panel_rows) > 0, len(panel_rows), "panel movie-days"),
        ("sample_mode", sample in SAMPLE_CHOICES or sample == "wide-custom", sample, "recognized sample mode"),
        ("start_date_filter", all(parse_date(str(row["box_office_date"])) >= start_date for row in panel_rows), start_date.isoformat(), "all target dates on or after start date"),
        ("positive_target_gross", all(float(row["target_gross_usd"]) > 0 for row in panel_rows), "", "all target grosses positive"),
        (
            "within_max_age",
            True if effective_max_age_weeks is None else all(0 < float(row["age_days"]) < effective_max_age_weeks * 7 for row in panel_rows),
            effective_max_age_weeks if effective_max_age_weeks is not None else "",
            "all rows within configured release-age window when sample applies one",
        ),
        ("positive_theater_rows", all(float(row["theaters"]) > 0 for row in panel_rows), wide_theater_threshold, "target rows have positive theater counts"),
        ("theater_share_bounds", all(0.0 <= float(row["theater_share"]) <= 1.0 for row in panel_rows), "", "target-date theater share bounded"),
        ("competitor_share_bounds", all(0.0 <= float(row["prior_day_competitor_market_share_ex_focal"]) <= 1.0 for row in panel_rows), "", "competitor market share bounded"),
        (
            "prior_day_lags_flagged",
            all(float(row["own_prior_day_gross"]) > 0 or float(row["own_prior_day_missing"]) == 1.0 for row in panel_rows),
            "",
            "missing own prior-day gross is represented by an indicator",
        ),
        ("panel_source_total_reconciles", abs(panel_total - panel_source_total) < 1e-6, panel_total - panel_source_total, "panel target totals equal source totals for the same movie-date keys"),
    ]
    for check, passed, value, details in checks:
        rows.append(
            {
                "check": check,
                "passed": int(bool(passed)),
                "value": value,
                "details": details,
            }
        )
    return rows


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * pct
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[int(index)]
    weight = index - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def bootstrap_ci_for_term(
    rows: list[dict[str, object]],
    *,
    model_name: str,
    terms: list[str],
    target_col: str,
    term: str,
    repetitions: int = 80,
) -> tuple[float | None, float | None]:
    rng = random.Random(1729)
    movie_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    date_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        movie_groups[str(row["movie_id"])].append(row)
        date_groups[str(row["box_office_date"])].append(row)
    movie_keys = sorted(movie_groups)
    date_keys = sorted(date_groups)
    estimates = []
    term_index = terms.index(term) + 1
    for rep in range(repetitions):
        groups = movie_groups if rep % 2 == 0 else date_groups
        keys = movie_keys if rep % 2 == 0 else date_keys
        sample_rows = []
        for _ in keys:
            sample_rows.extend(groups[rng.choice(keys)])
        if len(sample_rows) < len(terms) + 3:
            continue
        try:
            _coef_rows, beta, _r2, _sse = fit_model_for_target(model_name, sample_rows, terms, target_col)
        except (ValueError, SystemExit):
            continue
        estimates.append(beta[term_index])
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def competition_hypothesis_test_rows(panel_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = enrich_share_targets(panel_rows)
    train_rows, holdout_rows, cutoff = chronological_split(rows)
    terms = share_terms_for(TEST_MODEL_TERMS["einav_competition_interactions"])
    coef_rows, beta, train_r2, _sse = fit_model_for_target(
        "einav_competition_interactions",
        train_rows,
        terms,
        "logit_daily_share",
    )
    metrics = evaluate_model_for_target(
        "einav_competition_interactions",
        beta,
        terms,
        train_rows,
        holdout_rows,
        target_col="logit_daily_share",
    )
    by_term = {str(row["term"]): row for row in coef_rows}
    tests = [
        ("einav_underlying_demand", "estimated_underlying_demand_effect", "Einav week effects improve share prediction if stable and predictive."),
        ("competitor_share_pressure", "prior_day_competitor_market_share_ex_focal", "Expected sign is negative if relative local competition predicts lower next-day share."),
        ("amplification_interaction", "competitor_total_x_amplification_gap", "Tests whether competition is stronger in high-amplification weeks."),
        ("top1_competition", "log_prior_day_competitor_top1_gross", "Top-heavy competition signal."),
        ("competitor_hhi", "prior_day_competitor_hhi", "Competition concentration signal."),
        ("competitor_count", "prior_day_competitor_count", "Raw crowding signal."),
    ]
    output = []
    for hypothesis, term, interpretation in tests:
        row = by_term.get(term)
        ci_low, ci_high = bootstrap_ci_for_term(
            train_rows,
            model_name="einav_competition_interactions",
            terms=terms,
            target_col="logit_daily_share",
            term=term,
        )
        output.append(
            {
                "hypothesis": hypothesis,
                "term": term,
                "coef": row.get("coef") if row else None,
                "se_naive": row.get("se_naive") if row else None,
                "t_stat": row.get("t_stat") if row else None,
                "p_normal_approx": row.get("p_normal_approx") if row else None,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "train_r2": train_r2,
                "holdout_rmse": metrics["holdout_rmse"],
                "holdout_r2": metrics["holdout_r2"],
                "split_cutoff_date": cutoff,
                "interpretation": interpretation,
            }
        )
    return output


def rebuild_panel_for_grid(
    raw_rows: list[DailyMovieRow],
    *,
    start_date: dt.date,
    end_date: dt.date | None,
    threshold: int,
    max_age_weeks: int,
    einav_effects: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    panel, _features = build_daily_movie_panel(
        raw_rows,
        start_date=start_date,
        end_date=end_date,
        wide_theater_threshold=threshold,
        max_age_weeks=max_age_weeks,
        einav_effects=einav_effects,
        sample="wide-custom",
    )
    return panel


def robustness_grid_metrics(
    raw_rows: list[DailyMovieRow],
    *,
    start_date: dt.date,
    end_date: dt.date | None,
    einav_effects: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    output = []
    for threshold in (600, 1000, 2000):
        for age_weeks in (4, 6, 10):
            panel = rebuild_panel_for_grid(
                raw_rows,
                start_date=start_date,
                end_date=end_date,
                threshold=threshold,
                max_age_weeks=age_weeks,
                einav_effects=einav_effects,
            )
            if len(panel) < 100:
                continue
            rows = enrich_share_targets(panel)
            train_rows, holdout_rows, cutoff = chronological_split(rows)
            if len(train_rows) < 50 or len(holdout_rows) < 10:
                continue
            terms = share_terms_for(TEST_MODEL_TERMS["einav_competition"])
            coef_rows, beta, train_r2, _sse = fit_model_for_target(
                "einav_competition",
                train_rows,
                terms,
                "logit_daily_share",
            )
            metrics = evaluate_model_for_target(
                "einav_competition",
                beta,
                terms,
                train_rows,
                holdout_rows,
                target_col="logit_daily_share",
            )
            competitor_share = next(
                row for row in coef_rows if row["term"] == "prior_day_competitor_market_share_ex_focal"
            )
            output.append(
                {
                    "wide_theater_threshold": threshold,
                    "max_age_weeks": age_weeks,
                    "panel_n": len(panel),
                    "movies": len({row["movie_id"] for row in panel}),
                    "train_n": len(train_rows),
                    "holdout_n": len(holdout_rows),
                    "train_r2": train_r2,
                    "holdout_rmse": metrics["holdout_rmse"],
                    "holdout_r2": metrics["holdout_r2"],
                    "split_cutoff_date": cutoff,
                    "competitor_share_coef": competitor_share["coef"],
                    "competitor_share_p_normal_approx": competitor_share["p_normal_approx"],
                }
            )
    return output


def transformed_competitor_rows(rows: list[dict[str, object]], mode: str) -> list[dict[str, object]]:
    output = [dict(row) for row in rows]
    if mode == "shuffled_within_date":
        rng = random.Random(411)
        by_date: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(output):
            by_date[str(row["box_office_date"])].append(index)
        for indexes in by_date.values():
            feature_vectors = [{col: output[index][col] for col in COMPETITOR_FEATURE_COLUMNS} for index in indexes]
            rng.shuffle(feature_vectors)
            for index, vector in zip(indexes, feature_vectors):
                output[index].update(vector)
    elif mode == "shuffled_across_dates":
        vectors = [{col: row[col] for col in COMPETITOR_FEATURE_COLUMNS} for row in output]
        shift = min(17, max(1, len(vectors) // 7))
        vectors = vectors[shift:] + vectors[:shift]
        for row, vector in zip(output, vectors):
            row.update(vector)
    elif mode == "future_competitor_features_invalid":
        by_movie_date = {(str(row["movie_id"]), str(row["box_office_date"])): row for row in output}
        for row in output:
            future_date = (parse_date(str(row["box_office_date"])) + dt.timedelta(days=1)).isoformat()
            future = by_movie_date.get((str(row["movie_id"]), future_date))
            if future:
                for col in COMPETITOR_FEATURE_COLUMNS:
                    row[col] = future[col]
    return output


def placebo_test_results(panel_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    base_rows = enrich_share_targets(panel_rows)
    output = []
    for mode in ("actual", "shuffled_within_date", "shuffled_across_dates", "future_competitor_features_invalid"):
        rows = base_rows if mode == "actual" else transformed_competitor_rows(base_rows, mode)
        train_rows, holdout_rows, cutoff = chronological_split(rows)
        terms = share_terms_for(TEST_MODEL_TERMS["einav_competition"])
        coef_rows, beta, train_r2, _sse = fit_model_for_target(
            "einav_competition",
            train_rows,
            terms,
            "logit_daily_share",
        )
        metrics = evaluate_model_for_target(
            "einav_competition",
            beta,
            terms,
            train_rows,
            holdout_rows,
            target_col="logit_daily_share",
        )
        competitor_share = next(row for row in coef_rows if row["term"] == "prior_day_competitor_market_share_ex_focal")
        output.append(
            {
                "placebo": mode,
                "valid_forecast_design": int(mode != "future_competitor_features_invalid"),
                "train_n": len(train_rows),
                "holdout_n": len(holdout_rows),
                "train_r2": train_r2,
                "holdout_rmse": metrics["holdout_rmse"],
                "holdout_r2": metrics["holdout_r2"],
                "split_cutoff_date": cutoff,
                "competitor_share_coef": competitor_share["coef"],
                "competitor_share_p_normal_approx": competitor_share["p_normal_approx"],
            }
        )
    return output


def focused_2026_split(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train_rows = [row for row in rows if parse_date(str(row["box_office_date"])) < dt.date(2026, 1, 1)]
    test_rows = [row for row in rows if parse_date(str(row["box_office_date"])).year == 2026]
    return train_rows, test_rows


def pearson_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = einav.mean(xs)
    y_mean = einav.mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_ss = sum((x - x_mean) ** 2 for x in xs)
    y_ss = sum((y - y_mean) ** 2 for y in ys)
    if x_ss <= 0 or y_ss <= 0:
        return None
    return numerator / math.sqrt(x_ss * y_ss)


def calibration_values(actual: list[float], predicted: list[float]) -> tuple[float | None, float | None]:
    if len(actual) < 3 or len(actual) != len(predicted):
        return None, None
    beta, _se, _r2, _sse = einav.ols([[1.0, value] for value in predicted], actual)
    return beta[0], beta[1]


def focused_2026_prediction_rows(
    model_name: str,
    beta: list[float],
    terms: list[str],
    test_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for row in test_rows:
        predicted_logit = predict(beta, row, terms)
        predicted_share = logistic(predicted_logit)
        actual_share = float(row["daily_market_share"])
        actual_logit = float(row["logit_daily_share"])
        industry_gross = float(row["daily_industry_gross_usd"])
        actual_gross = float(row["target_gross_usd"])
        implied_predicted_gross = predicted_share * industry_gross
        rows.append(
            {
                "model": model_name,
                "movie_id": row["movie_id"],
                "title": row["title"],
                "box_office_date": row["box_office_date"],
                "actual_logit_daily_share": actual_logit,
                "predicted_logit_daily_share": predicted_logit,
                "logit_residual": actual_logit - predicted_logit,
                "abs_logit_error": abs(actual_logit - predicted_logit),
                "actual_daily_share": actual_share,
                "predicted_daily_share": predicted_share,
                "share_residual": actual_share - predicted_share,
                "abs_share_error": abs(actual_share - predicted_share),
                "prior_day_competitor_market_share_ex_focal": row["prior_day_competitor_market_share_ex_focal"],
                "own_prior_day_market_share": row["own_prior_day_market_share"],
                "daily_industry_gross_usd": industry_gross,
                "actual_gross_usd": actual_gross,
                "implied_predicted_gross_usd": implied_predicted_gross,
                "gross_residual_usd": actual_gross - implied_predicted_gross,
                "abs_gross_error_usd": abs(actual_gross - implied_predicted_gross),
            }
        )
    return rows


def focused_2026_metrics_for_predictions(
    model_name: str,
    train_rows: list[dict[str, object]],
    predictions: list[dict[str, object]],
    *,
    train_r2: float,
) -> dict[str, object]:
    actual_logits = [float(row["actual_logit_daily_share"]) for row in predictions]
    predicted_logits = [float(row["predicted_logit_daily_share"]) for row in predictions]
    logit_residuals = [float(row["logit_residual"]) for row in predictions]
    share_residuals = [float(row["share_residual"]) for row in predictions]
    gross_residuals = [float(row["gross_residual_usd"]) for row in predictions]
    tss = sum((value - einav.mean(actual_logits)) ** 2 for value in actual_logits) if actual_logits else 0.0
    sse = sum(value * value for value in logit_residuals)
    calibration_intercept, calibration_slope = calibration_values(actual_logits, predicted_logits)
    hits = 0
    count = 0
    for row in predictions:
        actual_delta = float(row["actual_daily_share"]) - float(row["own_prior_day_market_share"])
        predicted_delta = float(row["predicted_daily_share"]) - float(row["own_prior_day_market_share"])
        if actual_delta == 0.0:
            continue
        count += 1
        if (actual_delta > 0.0) == (predicted_delta > 0.0):
            hits += 1
    return {
        "model": model_name,
        "train_n": len(train_rows),
        "test_n": len(predictions),
        "train_start_date": min(str(row["box_office_date"]) for row in train_rows) if train_rows else "",
        "train_end_date": max(str(row["box_office_date"]) for row in train_rows) if train_rows else "",
        "test_start_date": min(str(row["box_office_date"]) for row in predictions) if predictions else "",
        "test_end_date": max(str(row["box_office_date"]) for row in predictions) if predictions else "",
        "train_r2": train_r2,
        "test_rmse_logit_share": rmse(logit_residuals),
        "test_mae_logit_share": mae(logit_residuals),
        "test_r2_logit_share": 1.0 - sse / tss if tss else 0.0,
        "test_rmse_share": rmse(share_residuals),
        "test_mae_share": mae(share_residuals),
        "test_mae_implied_gross_usd": mae(gross_residuals),
        "directional_accuracy": hits / count if count else None,
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "actual_predicted_logit_correlation": pearson_corr(actual_logits, predicted_logits),
        "actual_predicted_share_correlation": pearson_corr(
            [float(row["actual_daily_share"]) for row in predictions],
            [float(row["predicted_daily_share"]) for row in predictions],
        ),
    }


def focused_2026_movie_level_summary(prediction_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in prediction_rows:
        buckets[(str(row["model"]), str(row["movie_id"]), str(row["title"]))].append(row)
    output = []
    for (model, movie_id, title), rows in sorted(
        buckets.items(),
        key=lambda item: (item[0][0], -sum(float(row["actual_gross_usd"]) for row in item[1])),
    ):
        logit_residuals = [float(row["logit_residual"]) for row in rows]
        share_residuals = [float(row["share_residual"]) for row in rows]
        gross_residuals = [float(row["gross_residual_usd"]) for row in rows]
        output.append(
            {
                "model": model,
                "movie_id": movie_id,
                "title": title,
                "row_count": len(rows),
                "first_date": min(str(row["box_office_date"]) for row in rows),
                "last_date": max(str(row["box_office_date"]) for row in rows),
                "observed_average_share": einav.mean(float(row["actual_daily_share"]) for row in rows),
                "predicted_average_share": einav.mean(float(row["predicted_daily_share"]) for row in rows),
                "total_actual_gross_usd": sum(float(row["actual_gross_usd"]) for row in rows),
                "total_implied_predicted_gross_usd": sum(float(row["implied_predicted_gross_usd"]) for row in rows),
                "rmse_logit_share": rmse(logit_residuals),
                "mae_logit_share": mae(logit_residuals),
                "rmse_share": rmse(share_residuals),
                "mae_share": mae(share_residuals),
                "mae_implied_gross_usd": mae(gross_residuals),
            }
        )
    return output


def focused_2026_prediction_deciles(prediction_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    by_model: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in prediction_rows:
        by_model[str(row["model"])].append(row)
    for model, rows in sorted(by_model.items()):
        sorted_rows = sorted(rows, key=lambda row: float(row["predicted_daily_share"]))
        if not sorted_rows:
            continue
        for decile in range(1, 11):
            start = math.floor((decile - 1) * len(sorted_rows) / 10)
            end = math.floor(decile * len(sorted_rows) / 10)
            bucket = sorted_rows[start:end]
            if not bucket:
                continue
            output.append(
                {
                    "model": model,
                    "decile": decile,
                    "n": len(bucket),
                    "mean_predicted_share": einav.mean(float(row["predicted_daily_share"]) for row in bucket),
                    "mean_actual_share": einav.mean(float(row["actual_daily_share"]) for row in bucket),
                    "mean_share_residual": einav.mean(float(row["share_residual"]) for row in bucket),
                    "mean_predicted_logit_share": einav.mean(float(row["predicted_logit_daily_share"]) for row in bucket),
                    "mean_actual_logit_share": einav.mean(float(row["actual_logit_daily_share"]) for row in bucket),
                }
            )
    return output


def run_focused_2026_oos(panel_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rows = enrich_share_targets(panel_rows)
    train_rows, test_rows = focused_2026_split(rows)
    if len(train_rows) < 50 or len(test_rows) < 10:
        return [], [], [], [], []
    coefficient_rows = []
    metric_rows = []
    prediction_rows = []
    for model_name, terms in FOCUSED_2026_MODELS.items():
        coeffs, beta, train_r2, _sse = fit_model_for_target(
            model_name,
            train_rows,
            terms,
            "logit_daily_share",
        )
        for row in coeffs:
            coefficient_rows.append(row)
        model_predictions = focused_2026_prediction_rows(model_name, beta, terms, test_rows)
        prediction_rows.extend(model_predictions)
        metric_rows.append(
            focused_2026_metrics_for_predictions(
                model_name,
                train_rows,
                model_predictions,
                train_r2=train_r2,
            )
        )
    movie_rows = focused_2026_movie_level_summary(prediction_rows)
    decile_rows = focused_2026_prediction_deciles(prediction_rows)
    return coefficient_rows, metric_rows, prediction_rows, movie_rows, decile_rows


def write_focused_scatter_svg(
    path: Path,
    rows: list[dict[str, object]],
    *,
    model: str,
    title: str,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
) -> None:
    values = [row for row in rows if row["model"] == model]
    if not values:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"760\" height=\"460\"></svg>\n", encoding="utf-8")
        return
    sample = values[:: max(1, len(values) // 600)]
    xs = [float(row[x_col]) for row in sample]
    ys = [float(row[y_col]) for row in sample]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    width, height = 760, 520
    left, right, top, bottom = 78, 35, 60, 455

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (width - left - right)

    def sy(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">{title}</text>',
        f'<text x="36" y="55" font-family="Arial" font-size="13" fill="#555">{model}</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{width - right}" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>',
    ]
    if x_col == "actual_daily_share" and y_col == "predicted_daily_share":
        lo = max(x_min, y_min)
        hi = min(x_max, y_max)
        if lo < hi:
            parts.append(f'<line x1="{sx(lo):.1f}" y1="{sy(lo):.1f}" x2="{sx(hi):.1f}" y2="{sy(hi):.1f}" stroke="#999" stroke-dasharray="4 4"/>')
    if y_min <= 0.0 <= y_max:
        parts.append(f'<line x1="{left}" y1="{sy(0.0):.1f}" x2="{width - right}" y2="{sy(0.0):.1f}" stroke="#aaa" stroke-dasharray="3 4"/>')
    for x, y in zip(xs, ys):
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="#2f6f73" fill-opacity="0.45"/>')
    parts.extend(
        [
            f'<text x="{(left + width - right) / 2:.1f}" y="{height - 18}" font-family="Arial" font-size="12" text-anchor="middle">{x_label}</text>',
            f'<text x="18" y="{(top + bottom) / 2:.1f}" font-family="Arial" font-size="12" transform="rotate(-90 18 {(top + bottom) / 2:.1f})" text-anchor="middle">{y_label}</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def write_focused_movie_gross_svg(path: Path, movie_rows: list[dict[str, object]], *, model: str) -> None:
    rows = [row for row in movie_rows if row["model"] == model]
    rows = sorted(rows, key=lambda row: float(row["total_actual_gross_usd"]), reverse=True)[:25]
    if not rows:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"900\" height=\"520\"></svg>\n", encoding="utf-8")
        return
    width, height = 980, 620
    left, right, top, bottom = 220, 40, 60, 560
    max_value = max(
        max(float(row["total_actual_gross_usd"]), float(row["total_implied_predicted_gross_usd"]))
        for row in rows
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">2026 movie-level actual vs implied predicted gross</text>',
        f'<text x="36" y="55" font-family="Arial" font-size="13" fill="#555">{model}; predicted gross uses actual daily industry totals</text>',
    ]
    row_height = (bottom - top) / len(rows)
    scale = (width - left - right) / max_value if max_value else 1.0
    for index, row in enumerate(rows):
        y = top + index * row_height
        actual = float(row["total_actual_gross_usd"])
        predicted = float(row["total_implied_predicted_gross_usd"])
        label = str(row["title"])
        if len(label) > 28:
            label = label[:25] + "..."
        parts.append(f'<text x="{left - 8}" y="{y + row_height * 0.62:.1f}" font-family="Arial" font-size="11" text-anchor="end">{einav.svg_escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y + 4:.1f}" width="{actual * scale:.1f}" height="{row_height * 0.35:.1f}" fill="#2f6f73"/>')
        parts.append(f'<rect x="{left}" y="{y + row_height * 0.48:.1f}" width="{predicted * scale:.1f}" height="{row_height * 0.35:.1f}" fill="#c15b3f"/>')
    parts.append(f'<text x="{width - right - 180}" y="36" font-family="Arial" font-size="12" fill="#2f6f73">actual</text>')
    parts.append(f'<text x="{width - right - 120}" y="36" font-family="Arial" font-size="12" fill="#c15b3f">predicted</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_focused_decile_svg(path: Path, decile_rows: list[dict[str, object]], *, model: str) -> None:
    rows = [row for row in decile_rows if row["model"] == model]
    if not rows:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"760\" height=\"420\"></svg>\n", encoding="utf-8")
        return
    width, height = 760, 460
    left, right, top, bottom = 72, 35, 60, 370
    values = [float(row["mean_predicted_share"]) for row in rows] + [float(row["mean_actual_share"]) for row in rows]
    y_min, y_max = 0.0, max(values) if values else 1.0
    if y_max == 0:
        y_max = 1.0

    def sx(decile: int) -> float:
        return left + (decile - 1) / 9 * (width - left - right)

    def sy(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    predicted_points = " ".join(f'{sx(int(row["decile"])):.1f},{sy(float(row["mean_predicted_share"])):.1f}' for row in rows)
    actual_points = " ".join(f'{sx(int(row["decile"])):.1f},{sy(float(row["mean_actual_share"])):.1f}' for row in rows)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">2026 prediction decile calibration</text>',
        f'<text x="36" y="55" font-family="Arial" font-size="13" fill="#555">{model}</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{width - right}" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>',
        f'<polyline points="{predicted_points}" fill="none" stroke="#c15b3f" stroke-width="2.4"/>',
        f'<polyline points="{actual_points}" fill="none" stroke="#2f6f73" stroke-width="2.4"/>',
        '<text x="580" y="88" font-family="Arial" font-size="12" fill="#c15b3f">predicted</text>',
        '<text x="580" y="106" font-family="Arial" font-size="12" fill="#2f6f73">actual</text>',
    ]
    for row in rows:
        parts.append(f'<text x="{sx(int(row["decile"])):.1f}" y="{bottom + 22}" font-family="Arial" font-size="11" text-anchor="middle">{row["decile"]}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_focused_2026_readme(
    path: Path,
    *,
    metric_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
) -> None:
    by_model = {str(row["model"]): row for row in metric_rows}
    main = by_model.get("incremental_competitor_share", {})
    path.write_text(
        "\n".join(
            [
                "Focused 2026 OOS competitor-share regression",
                "",
                "Train set: eligible movie-days before 2026-01-01.",
                "Test set: eligible movie-days in calendar year 2026.",
                "Primary target: logit daily market share.",
                "Primary predictor of interest: prior_day_competitor_market_share_ex_focal.",
                "",
                f"Prediction rows: {len(prediction_rows)}",
                f"Incremental model test RMSE logit share: {main.get('test_rmse_logit_share', '')}",
                f"Incremental model test R2 logit share: {main.get('test_r2_logit_share', '')}",
                f"Incremental model directional accuracy: {main.get('directional_accuracy', '')}",
                "",
                "Implied gross predictions multiply predicted share by actual 2026 daily industry gross.",
                "They are ex-post diagnostics, not real-time total-gross forecasts.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_focused_2026_outputs(out_dir: Path, panel_rows: list[dict[str, object]], *, dirname: str = "focused_2026_oos") -> list[dict[str, object]]:
    focused_dir = out_dir / dirname
    focused_dir.mkdir(parents=True, exist_ok=True)
    coefficient_rows, metric_rows, prediction_rows, movie_rows, decile_rows = run_focused_2026_oos(panel_rows)
    coeff_fields = ["model", "target", "term", "coef", "se_naive", "t_stat", "p_normal_approx", "n", "r2_train", "sse_train"]
    metric_fields = [
        "model",
        "train_n",
        "test_n",
        "train_start_date",
        "train_end_date",
        "test_start_date",
        "test_end_date",
        "train_r2",
        "test_rmse_logit_share",
        "test_mae_logit_share",
        "test_r2_logit_share",
        "test_rmse_share",
        "test_mae_share",
        "test_mae_implied_gross_usd",
        "directional_accuracy",
        "calibration_intercept",
        "calibration_slope",
        "actual_predicted_logit_correlation",
        "actual_predicted_share_correlation",
    ]
    prediction_fields = [
        "model",
        "movie_id",
        "title",
        "box_office_date",
        "actual_logit_daily_share",
        "predicted_logit_daily_share",
        "logit_residual",
        "abs_logit_error",
        "actual_daily_share",
        "predicted_daily_share",
        "share_residual",
        "abs_share_error",
        "prior_day_competitor_market_share_ex_focal",
        "own_prior_day_market_share",
        "daily_industry_gross_usd",
        "actual_gross_usd",
        "implied_predicted_gross_usd",
        "gross_residual_usd",
        "abs_gross_error_usd",
    ]
    movie_fields = [
        "model",
        "movie_id",
        "title",
        "row_count",
        "first_date",
        "last_date",
        "observed_average_share",
        "predicted_average_share",
        "total_actual_gross_usd",
        "total_implied_predicted_gross_usd",
        "rmse_logit_share",
        "mae_logit_share",
        "rmse_share",
        "mae_share",
        "mae_implied_gross_usd",
    ]
    decile_fields = [
        "model",
        "decile",
        "n",
        "mean_predicted_share",
        "mean_actual_share",
        "mean_share_residual",
        "mean_predicted_logit_share",
        "mean_actual_logit_share",
    ]
    einav.write_csv(focused_dir / "focused_2026_oos_coefficients.csv", coefficient_rows, coeff_fields)
    einav.write_csv(focused_dir / "focused_2026_oos_metrics.csv", metric_rows, metric_fields)
    einav.write_csv(focused_dir / "focused_2026_movie_day_predictions.csv", prediction_rows, prediction_fields)
    einav.write_csv(focused_dir / "focused_2026_movie_level_summary.csv", movie_rows, movie_fields)
    einav.write_csv(focused_dir / "focused_2026_prediction_deciles.csv", decile_rows, decile_fields)
    main_model = "incremental_competitor_share"
    write_focused_scatter_svg(
        focused_dir / "figure_2026_actual_vs_predicted_share.svg",
        prediction_rows,
        model=main_model,
        title="2026 actual vs predicted daily share",
        x_col="actual_daily_share",
        y_col="predicted_daily_share",
        x_label="Actual daily share",
        y_label="Predicted daily share",
    )
    write_focused_scatter_svg(
        focused_dir / "figure_2026_residuals_vs_competitor_share.svg",
        prediction_rows,
        model=main_model,
        title="2026 residuals vs prior-day competitor share",
        x_col="prior_day_competitor_market_share_ex_focal",
        y_col="share_residual",
        x_label="Prior-day competitor market share excluding focal",
        y_label="Actual minus predicted daily share",
    )
    write_focused_movie_gross_svg(
        focused_dir / "figure_2026_movie_actual_vs_predicted_gross.svg",
        movie_rows,
        model=main_model,
    )
    write_focused_decile_svg(
        focused_dir / "figure_2026_prediction_decile_calibration.svg",
        decile_rows,
        model=main_model,
    )
    write_focused_2026_readme(focused_dir / "README.txt", metric_rows=metric_rows, prediction_rows=prediction_rows)
    return metric_rows


def write_line_svg(path: Path, rows: list[dict[str, object]], *, title: str, x_col: str, y_col: str, group_col: str) -> None:
    values = [row for row in rows if row.get(y_col) not in (None, "")]
    if not values:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"760\" height=\"360\"></svg>\n", encoding="utf-8")
        return
    width, height = 840, 460
    left, right, top, bottom = 85, 35, 60, 370
    groups = sorted({str(row[group_col]) for row in values})
    xs = sorted({str(row[x_col]) for row in values})
    y_values = [float(row[y_col]) for row in values]
    y_min, y_max = min(y_values), max(y_values)
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    colors = ["#2f6f73", "#c15b3f", "#6f5aa8", "#777777", "#d49a2a"]

    def sx(label: str) -> float:
        return left + xs.index(label) / max(1, len(xs) - 1) * (width - left - right)

    def sy(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">{title}</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{width - right}" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>',
    ]
    for group_index, group in enumerate(groups):
        group_rows = [row for row in values if str(row[group_col]) == group]
        points = " ".join(f'{sx(str(row[x_col])):.1f},{sy(float(row[y_col])):.1f}' for row in group_rows)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[group_index % len(colors)]}" stroke-width="2"/>')
        parts.append(f'<text x="{width - right - 180}" y="{82 + group_index * 18}" font-family="Arial" font-size="12" fill="{colors[group_index % len(colors)]}">{group}</text>')
    for label in xs:
        parts.append(f'<text x="{sx(label):.1f}" y="{bottom + 22}" font-family="Arial" font-size="11" text-anchor="middle">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_share_prediction_svg(path: Path, prediction_rows: list[dict[str, object]]) -> None:
    rows = [row for row in prediction_rows if row.get("target") == "logit_daily_share" and row.get("actual") is not None]
    if not rows:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"760\" height=\"360\"></svg>\n", encoding="utf-8")
        return
    sample = rows[:: max(1, len(rows) // 500)]
    actuals = [logistic(float(row["actual"])) for row in sample]
    preds = [logistic(float(row["predicted"])) for row in sample]
    hi = max(actuals + preds)
    width, height = 760, 520
    left, right, top, bottom = 75, 35, 60, 455

    def scale(value: float) -> float:
        return left + value / hi * (width - left - right) if hi else left

    def sy(value: float) -> float:
        return bottom - value / hi * (bottom - top) if hi else bottom

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">Observed vs predicted daily market share</text>',
        '<text x="36" y="55" font-family="Arial" font-size="13" fill="#555">Rolling holdout predictions from Einav + competition interactions</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{width - right}" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{scale(0):.1f}" y1="{sy(0):.1f}" x2="{scale(hi):.1f}" y2="{sy(hi):.1f}" stroke="#999" stroke-dasharray="4 4"/>',
    ]
    for actual, pred in zip(actuals, preds):
        parts.append(f'<circle cx="{scale(actual):.1f}" cy="{sy(pred):.1f}" r="3" fill="#2f6f73" fill-opacity="0.45"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_testing_readme(
    path: Path,
    *,
    panel_rows: list[dict[str, object]],
    validation_rows: list[dict[str, object]],
    rolling_rows: list[dict[str, object]],
) -> None:
    failed = [row["check"] for row in validation_rows if not int(row["passed"])]
    competition_wins = 0
    windows = 0
    by_window: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rolling_rows:
        if row["target"] != "target_log_gross":
            continue
        by_window[str(row["window"])][str(row["model"])] = float(row["holdout_rmse"])
    for metrics in by_window.values():
        if "lifecycle_baseline" in metrics and "einav_competition" in metrics:
            windows += 1
            if metrics["einav_competition"] < metrics["lifecycle_baseline"]:
                competition_wins += 1
    path.write_text(
        "\n".join(
            [
                "Testing artifacts for the Einav-grounded daily competition extension",
                "",
                f"Panel movie-days tested: {len(panel_rows)}",
                f"Movies tested: {len({row['movie_id'] for row in panel_rows})}",
                f"Validation failures: {', '.join(failed) if failed else 'none'}",
                f"Competition model RMSE wins vs lifecycle baseline: {competition_wins}/{windows} rolling gross windows",
                "",
                "Share targets use the generated sample's daily wide-release industry total.",
                "The future-competitor placebo is intentionally invalid for forecasting and is labeled as such.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_testing_outputs(
    out_dir: Path,
    *,
    raw_rows: list[DailyMovieRow],
    panel_rows: list[dict[str, object]],
    start_date: dt.date,
    end_date: dt.date | None,
    wide_theater_threshold: int,
    max_age_weeks: int,
    einav_effects: dict[str, dict[str, float]],
    sample: str = DEFAULT_SAMPLE,
) -> None:
    testing_dir = out_dir / "testing"
    testing_dir.mkdir(parents=True, exist_ok=True)
    validation_rows = validation_check_rows(
        raw_rows,
        panel_rows,
        start_date=start_date,
        wide_theater_threshold=wide_theater_threshold,
        max_age_weeks=max_age_weeks,
        sample=sample,
    )
    gross_rolling, gross_predictions = rolling_backtest_metrics(panel_rows, target_col="target_log_gross")
    share_rolling, share_coefficients_and_predictions = rolling_backtest_metrics(panel_rows, target_col="logit_daily_share")
    rolling_rows = gross_rolling + share_rolling
    share_regression_rows = [row for row in share_coefficients_and_predictions if "coef" in row]
    share_prediction_rows = [row for row in share_coefficients_and_predictions if "actual" in row]
    segment_rows = release_age_segment_metrics(panel_rows)
    two_stage_rows = two_stage_forecast_metrics(panel_rows)
    hypothesis_rows = competition_hypothesis_test_rows(panel_rows)
    robustness_rows = robustness_grid_metrics(
        raw_rows,
        start_date=start_date,
        end_date=end_date,
        einav_effects=einav_effects,
    )
    placebo_rows = placebo_test_results(panel_rows)

    einav.write_csv(testing_dir / "data_validation_checks.csv", validation_rows, ["check", "passed", "value", "details"])
    einav.write_csv(
        testing_dir / "rolling_backtest_metrics.csv",
        rolling_rows,
        [
            "window",
            "test_year",
            "model",
            "target",
            "train_n",
            "holdout_n",
            "train_r2",
            "train_rmse",
            "train_mae",
            "holdout_rmse",
            "holdout_mae",
            "holdout_r2",
            "directional_accuracy",
        ],
    )
    einav.write_csv(
        testing_dir / "release_age_segment_metrics.csv",
        segment_rows,
        [
            "window",
            "test_year",
            "release_age_segment",
            "model",
            "target",
            "train_n",
            "holdout_n",
            "train_rmse",
            "train_mae",
            "holdout_rmse",
            "holdout_mae",
            "holdout_r2",
            "directional_accuracy",
        ],
    )
    einav.write_csv(
        testing_dir / "share_target_regression_results.csv",
        share_regression_rows,
        ["window", "test_year", "model", "target", "term", "coef", "se_naive", "t_stat", "p_normal_approx", "n", "r2_train", "sse_train"],
    )
    einav.write_csv(
        testing_dir / "two_stage_forecast_metrics.csv",
        two_stage_rows,
        ["window", "test_year", "model", "train_n", "holdout_n", "holdout_rmse_log_gross", "holdout_mae_log_gross", "holdout_r2_log_gross", "holdout_mae_gross_usd"],
    )
    einav.write_csv(
        testing_dir / "competition_hypothesis_tests.csv",
        hypothesis_rows,
        [
            "hypothesis",
            "term",
            "coef",
            "se_naive",
            "t_stat",
            "p_normal_approx",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "train_r2",
            "holdout_rmse",
            "holdout_r2",
            "split_cutoff_date",
            "interpretation",
        ],
    )
    einav.write_csv(
        testing_dir / "robustness_grid_metrics.csv",
        robustness_rows,
        [
            "wide_theater_threshold",
            "max_age_weeks",
            "panel_n",
            "movies",
            "train_n",
            "holdout_n",
            "train_r2",
            "holdout_rmse",
            "holdout_r2",
            "split_cutoff_date",
            "competitor_share_coef",
            "competitor_share_p_normal_approx",
        ],
    )
    einav.write_csv(
        testing_dir / "placebo_test_results.csv",
        placebo_rows,
        [
            "placebo",
            "valid_forecast_design",
            "train_n",
            "holdout_n",
            "train_r2",
            "holdout_rmse",
            "holdout_r2",
            "split_cutoff_date",
            "competitor_share_coef",
            "competitor_share_p_normal_approx",
        ],
    )
    write_line_svg(
        testing_dir / "figure_backtest_rmse_by_model.svg",
        [row for row in rolling_rows if row["target"] == "target_log_gross"],
        title="Rolling holdout RMSE by model",
        x_col="test_year",
        y_col="holdout_rmse",
        group_col="model",
    )
    write_line_svg(
        testing_dir / "figure_competitor_share_effect_by_release_age.svg",
        [row for row in segment_rows if row["model"] == "einav_competition"],
        title="Competition model RMSE by release-age segment",
        x_col="release_age_segment",
        y_col="holdout_rmse",
        group_col="window",
    )
    write_share_prediction_svg(testing_dir / "figure_observed_vs_predicted_share.svg", share_prediction_rows)
    write_testing_readme(
        testing_dir / "README.txt",
        panel_rows=panel_rows,
        validation_rows=validation_rows,
        rolling_rows=rolling_rows,
    )


def write_readme(
    path: Path,
    *,
    panel_rows: list[dict[str, object]],
    start_date: dt.date,
    end_date: dt.date | None,
    wide_theater_threshold: int,
    max_age_weeks: int,
    sample: str,
) -> None:
    _apply_wide, effective_max_age_weeks, _require_prior = sample_settings(sample, max_age_weeks)
    path.write_text(
        "\n".join(
            [
                "Einav-grounded daily competition regression artifacts",
                "",
                "This is a 2022+ daily extension of the local Einav seasonality replication.",
                "The target is log1p(each movie's daily gross), predicted using only information",
                "available through the previous day plus calendar variables known in advance.",
                "",
                f"Panel movie-days: {len(panel_rows)}",
                f"Panel movies: {len({row['movie_id'] for row in panel_rows})}",
                f"Start date: {start_date.isoformat()}",
                f"End date: {end_date.isoformat() if end_date else 'latest available'}",
                f"Sample: {sample}",
                f"Wide-release threshold: {wide_theater_threshold} theaters",
                f"Max age weeks: {effective_max_age_weeks if effective_max_age_weeks is not None else 'none'}",
                "",
                "The Einav grounding variables are the estimated underlying season-week effect",
                "and the amplification gap: observed log inside-share effect minus estimated",
                "underlying demand effect.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_outputs(
    out_dir: Path,
    *,
    panel_rows: list[dict[str, object]],
    feature_rows: list[dict[str, object]],
    coefficients_by_model: dict[str, list[dict[str, object]]],
    comparison_rows: list[dict[str, object]],
    hypothesis_rows: list[dict[str, object]],
    predictions: list[dict[str, object]],
    start_date: dt.date,
    end_date: dt.date | None,
    wide_theater_threshold: int,
    max_age_weeks: int,
    sample: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_fields = [
        "sample",
        "movie_id",
        "title",
        "box_office_date",
        "season_week",
        "target_gross_usd",
        "target_log_gross",
        "own_prior_day_gross",
        "own_prior_day_missing",
        "own_prior_week_same_day_gross",
        "own_prior_week_same_day_missing",
        "log_own_prior_day_gross",
        "log_own_prior_week_same_day_gross",
        "age_days",
        "age_days_sq",
        "theaters",
        "log_theaters",
        "total_active_theaters",
        "theater_share",
        "prior_day_theaters",
        "log_prior_day_theaters",
        "prior_day_theater_share",
        "prior_day_theater_missing",
        "weekday",
        "weekday_1",
        "weekday_2",
        "weekday_3",
        "weekday_4",
        "weekday_5",
        "weekday_6",
        "is_weekend",
        "is_holiday_or_adjacent",
        "observed_log_inside_share_effect",
        "estimated_underlying_demand_effect",
        "seasonality_amplification_gap",
        "prior_day_competitor_total_gross",
        "prior_day_competitor_top1_gross",
        "prior_day_competitor_top3_gross",
        "prior_day_competitor_top5_gross",
        "prior_day_competitor_count",
        "prior_day_competitor_hhi",
        "prior_day_competitor_market_share_ex_focal",
        "log_prior_day_competitor_total_gross",
        "log_prior_day_competitor_top1_gross",
        "log_prior_day_competitor_top3_gross",
        "log_prior_day_competitor_top5_gross",
        "competitor_total_x_amplification_gap",
        "competitor_top1_x_amplification_gap",
        "competitor_hhi_x_amplification_gap",
    ]
    feature_fields = [
        "movie_id",
        "box_office_date",
        "prior_day_competitor_total_gross",
        "prior_day_competitor_top1_gross",
        "prior_day_competitor_top3_gross",
        "prior_day_competitor_top5_gross",
        "prior_day_competitor_count",
        "prior_day_competitor_hhi",
        "prior_day_competitor_market_share_ex_focal",
        "theater_share",
        "prior_day_theater_share",
        "seasonality_amplification_gap",
    ]
    coeff_fields = ["model", "term", "coef", "se_naive", "t_stat", "p_normal_approx", "n", "r2_train", "sse_train"]
    einav.write_csv(out_dir / "database_2022_plus_daily_movie_panel.csv", panel_rows, panel_fields)
    einav.write_csv(out_dir / "database_2022_plus_daily_competitor_features.csv", feature_rows, feature_fields)
    for model_name, filename in [
        ("lifecycle_baseline", "daily_regression_lifecycle_baseline.csv"),
        ("einav_grounded_baseline", "daily_regression_einav_grounded_baseline.csv"),
        ("competition_model", "daily_regression_competition_model.csv"),
        ("amplification_interactions", "daily_regression_amplification_interactions.csv"),
    ]:
        einav.write_csv(out_dir / filename, coefficients_by_model[model_name], coeff_fields)
    comparison_fields = [
        "model",
        "train_n",
        "holdout_n",
        "train_r2",
        "train_rmse_log",
        "train_mae_log",
        "holdout_rmse_log",
        "holdout_mae_log",
        "holdout_r2",
        "split_cutoff_date",
    ]
    einav.write_csv(out_dir / "daily_model_comparison.csv", comparison_rows, comparison_fields)
    einav.write_csv(out_dir / "daily_prediction_holdout_metrics.csv", comparison_rows, comparison_fields)
    einav.write_csv(
        out_dir / "einav_grounded_hypothesis_tests.csv",
        hypothesis_rows,
        ["hypothesis", "metric", "value", "supporting_term", "coef", "p_normal_approx"],
    )
    write_prediction_svg(out_dir / "figure_daily_actual_vs_predicted.svg", predictions)
    write_competitor_effect_svg(out_dir / "figure_competitor_effect_by_amplification_gap.svg", coefficients_by_model, panel_rows)
    write_readme(
        out_dir / "README.txt",
        panel_rows=panel_rows,
        start_date=start_date,
        end_date=end_date,
        wide_theater_threshold=wide_theater_threshold,
        max_age_weeks=max_age_weeks,
        sample=sample,
    )


def sample_comparison_rows(samples: list[tuple[str, list[dict[str, object]], list[dict[str, object]]]]) -> list[dict[str, object]]:
    all_gross = 0.0
    for sample_name, panel_rows, _metrics in samples:
        if sample_name == "all-in-theaters":
            all_gross = sum(float(row["target_gross_usd"]) for row in panel_rows)
            break
    rows = []
    for sample_name, panel_rows, metric_rows in samples:
        main_metric = next((row for row in metric_rows if row.get("model") == "incremental_competitor_share"), {})
        gross = sum(float(row["target_gross_usd"]) for row in panel_rows)
        rows.append(
            {
                "sample": sample_name,
                "rows": len(panel_rows),
                "movies": len({row["movie_id"] for row in panel_rows}),
                "start_date": min((str(row["box_office_date"]) for row in panel_rows), default=""),
                "end_date": max((str(row["box_office_date"]) for row in panel_rows), default=""),
                "sample_gross_usd": gross,
                "gross_coverage_vs_all_in_theaters": gross / all_gross if all_gross else None,
                "focused_2026_train_n": main_metric.get("train_n"),
                "focused_2026_test_n": main_metric.get("test_n"),
                "focused_2026_rmse_logit_share": main_metric.get("test_rmse_logit_share"),
                "focused_2026_mae_logit_share": main_metric.get("test_mae_logit_share"),
                "focused_2026_r2_logit_share": main_metric.get("test_r2_logit_share"),
                "focused_2026_directional_accuracy": main_metric.get("directional_accuracy"),
            }
        )
    return rows


def write_sample_comparison(out_dir: Path, rows: list[dict[str, object]]) -> None:
    einav.write_csv(
        out_dir / "sample_comparison_metrics.csv",
        rows,
        [
            "sample",
            "rows",
            "movies",
            "start_date",
            "end_date",
            "sample_gross_usd",
            "gross_coverage_vs_all_in_theaters",
            "focused_2026_train_n",
            "focused_2026_test_n",
            "focused_2026_rmse_logit_share",
            "focused_2026_mae_logit_share",
            "focused_2026_r2_logit_share",
            "focused_2026_directional_accuracy",
        ],
    )


def assign_competitor_pressure_quartiles(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    values = sorted(float(row["prior_day_competitor_market_share_ex_focal"]) for row in rows)
    q1 = percentile(values, 0.25) or 0.0
    q2 = percentile(values, 0.50) or 0.0
    q3 = percentile(values, 0.75) or 0.0
    output = []
    for row in rows:
        out = dict(row)
        value = float(row["prior_day_competitor_market_share_ex_focal"])
        if value <= q1:
            quartile = 1
        elif value <= q2:
            quartile = 2
        elif value <= q3:
            quartile = 3
        else:
            quartile = 4
        out["competitor_pressure_quartile"] = quartile
        out["competitor_pressure_quartile_label"] = f"Q{quartile}"
        output.append(out)
    return output


def bucketed_competition_effect_rows(panel_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = enrich_share_targets(panel_rows)
    train_rows, test_rows = focused_2026_split(rows)
    output = []
    terms = ["logit_own_prior_day_share", "prior_day_competitor_market_share_ex_focal"]
    for bucket in RELEASE_AGE_BUCKET_ORDER:
        bucket_train = [row for row in train_rows if release_age_bucket(float(row["age_days"])) == bucket]
        bucket_test = [row for row in test_rows if release_age_bucket(float(row["age_days"])) == bucket]
        if len(bucket_train) < len(terms) + 10:
            continue
        coef_rows, beta, train_r2, _sse = fit_model_for_target(
            f"bucket_{bucket}",
            bucket_train,
            terms,
            "logit_daily_share",
        )
        metrics = evaluate_model_for_target(
            f"bucket_{bucket}",
            beta,
            terms,
            bucket_train,
            bucket_test,
            target_col="logit_daily_share",
        ) if bucket_test else {
            "holdout_n": 0,
            "holdout_rmse": None,
            "holdout_mae": None,
            "holdout_r2": None,
            "directional_accuracy": None,
        }
        coef_by_term = {str(row["term"]): row for row in coef_rows}
        comp = coef_by_term.get("prior_day_competitor_market_share_ex_focal", {})
        own = coef_by_term.get("logit_own_prior_day_share", {})
        output.append(
            {
                "release_age_bucket": bucket,
                "train_n": len(bucket_train),
                "test_2026_n": len(bucket_test),
                "train_r2": train_r2,
                "holdout_rmse_logit_share": metrics.get("holdout_rmse"),
                "holdout_mae_logit_share": metrics.get("holdout_mae"),
                "holdout_r2_logit_share": metrics.get("holdout_r2"),
                "directional_accuracy": metrics.get("directional_accuracy"),
                "own_prior_share_coef": own.get("coef"),
                "own_prior_share_p_normal_approx": own.get("p_normal_approx"),
                "competitor_share_coef": comp.get("coef"),
                "competitor_share_se_naive": comp.get("se_naive"),
                "competitor_share_t_stat": comp.get("t_stat"),
                "competitor_share_p_normal_approx": comp.get("p_normal_approx"),
            }
        )
    return output


def exact_release_day_competition_effect_rows(
    panel_rows: list[dict[str, object]],
    *,
    min_train_n: int = 40,
) -> list[dict[str, object]]:
    rows = enrich_share_targets(panel_rows)
    train_rows, test_rows = focused_2026_split(rows)
    terms = ["logit_own_prior_day_share", "prior_day_competitor_market_share_ex_focal"]
    days = sorted({int(float(row["age_days"])) for row in train_rows})
    output = []
    for day in days:
        day_train = [row for row in train_rows if int(float(row["age_days"])) == day]
        if len(day_train) < min_train_n:
            continue
        day_test = [row for row in test_rows if int(float(row["age_days"])) == day]
        coef_rows, beta, train_r2, _sse = fit_model_for_target(
            f"release_day_{day}",
            day_train,
            terms,
            "logit_daily_share",
        )
        coef_by_term = {str(row["term"]): row for row in coef_rows}
        comp = coef_by_term.get("prior_day_competitor_market_share_ex_focal", {})
        metrics = evaluate_model_for_target(
            f"release_day_{day}",
            beta,
            terms,
            day_train,
            day_test,
            target_col="logit_daily_share",
        ) if day_test else {"holdout_rmse": None, "holdout_r2": None}
        output.append(
            {
                "release_age_day": day,
                "release_age_bucket": release_age_bucket(float(day)),
                "train_n": len(day_train),
                "test_2026_n": len(day_test),
                "train_r2": train_r2,
                "holdout_rmse_logit_share": metrics.get("holdout_rmse"),
                "holdout_r2_logit_share": metrics.get("holdout_r2"),
                "competitor_share_coef": comp.get("coef"),
                "competitor_share_se_naive": comp.get("se_naive"),
                "competitor_share_t_stat": comp.get("t_stat"),
                "competitor_share_p_normal_approx": comp.get("p_normal_approx"),
            }
        )
    return output


def competition_quartile_response_rows(panel_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = assign_competitor_pressure_quartiles(enrich_share_targets(panel_rows))
    train_rows, test_rows = focused_2026_split(rows)
    terms = ["logit_own_prior_day_share", "prior_day_competitor_market_share_ex_focal"]
    _coef_rows, beta, _train_r2, _sse = fit_model_for_target(
        "incremental_competitor_share",
        train_rows,
        terms,
        "logit_daily_share",
    )
    prediction_rows = focused_2026_prediction_rows(
        "incremental_competitor_share",
        beta,
        terms,
        test_rows,
    )
    quartile_lookup = {
        (str(row["movie_id"]), str(row["box_office_date"])): row
        for row in test_rows
    }
    buckets: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for pred in prediction_rows:
        source = quartile_lookup[(str(pred["movie_id"]), str(pred["box_office_date"]))]
        key = (release_age_bucket(float(source["age_days"])), int(source["competitor_pressure_quartile"]))
        merged = {**source, **pred}
        buckets[key].append(merged)
    output = []
    for (bucket, quartile), bucket_rows in sorted(buckets.items(), key=lambda item: (release_age_bucket_key(item[0][0]), item[0][1])):
        share_residuals = [float(row["share_residual"]) for row in bucket_rows]
        logit_residuals = [float(row["logit_residual"]) for row in bucket_rows]
        output.append(
            {
                "release_age_bucket": bucket,
                "competitor_pressure_quartile": quartile,
                "n": len(bucket_rows),
                "mean_competitor_share": einav.mean(float(row["prior_day_competitor_market_share_ex_focal"]) for row in bucket_rows),
                "mean_actual_share": einav.mean(float(row["actual_daily_share"]) for row in bucket_rows),
                "mean_predicted_share": einav.mean(float(row["predicted_daily_share"]) for row in bucket_rows),
                "mean_share_residual": einav.mean(share_residuals),
                "rmse_share": rmse(share_residuals),
                "mae_share": mae(share_residuals),
                "rmse_logit_share": rmse(logit_residuals),
                "mean_actual_gross_usd": einav.mean(float(row["actual_gross_usd"]) for row in bucket_rows),
                "mean_implied_predicted_gross_usd": einav.mean(float(row["implied_predicted_gross_usd"]) for row in bucket_rows),
            }
        )
    return output


def write_bucket_bar_svg(
    path: Path,
    rows: list[dict[str, object]],
    *,
    title: str,
    value_col: str,
    y_label: str,
) -> None:
    values = [row for row in rows if row.get(value_col) not in (None, "")]
    if not values:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"820\" height=\"440\"></svg>\n", encoding="utf-8")
        return
    values = sorted(values, key=lambda row: release_age_bucket_key(str(row["release_age_bucket"])))
    width, height = 880, 480
    left, right, top, bottom = 90, 35, 60, 370
    ys = [float(row[value_col]) for row in values]
    y_min, y_max = min(0.0, min(ys)), max(0.0, max(ys))
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    bar_width = (width - left - right) / max(1, len(values)) * 0.62

    def sx(index: int) -> float:
        return left + (index + 0.5) / len(values) * (width - left - right)

    def sy(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    zero_y = sy(0.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">{title}</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{width - right}" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{width - right}" y2="{zero_y:.1f}" stroke="#aaa" stroke-dasharray="4 4"/>',
    ]
    for index, row in enumerate(values):
        value = float(row[value_col])
        x = sx(index) - bar_width / 2
        y = min(sy(value), zero_y)
        h = abs(sy(value) - zero_y)
        color = "#c15b3f" if value < 0 else "#2f6f73"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{sx(index):.1f}" y="{bottom + 20}" font-family="Arial" font-size="10" text-anchor="middle">{row["release_age_bucket"]}</text>')
    parts.append(f'<text x="18" y="{(top + bottom) / 2:.1f}" font-family="Arial" font-size="12" transform="rotate(-90 18 {(top + bottom) / 2:.1f})" text-anchor="middle">{y_label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_quartile_response_svg(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"920\" height=\"520\"></svg>\n", encoding="utf-8")
        return
    width, height = 980, 560
    left, right, top, bottom = 78, 165, 60, 445
    buckets = [bucket for bucket in RELEASE_AGE_BUCKET_ORDER if any(row["release_age_bucket"] == bucket for row in rows)]
    y_values = [float(row["mean_actual_share"]) for row in rows] + [float(row["mean_predicted_share"]) for row in rows]
    y_min, y_max = 0.0, max(y_values) if y_values else 1.0
    if y_max == y_min:
        y_max += 1.0
    colors = {1: "#2f6f73", 2: "#5f8f5f", 3: "#d49a2a", 4: "#c15b3f"}

    def sx(bucket_index: int, quartile: int) -> float:
        cluster_width = (width - left - right) / max(1, len(buckets))
        return left + bucket_index * cluster_width + (quartile - 0.5) / 4 * cluster_width

    def sy(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">2026 share response by competitor-pressure quartile</text>',
        '<text x="36" y="55" font-family="Arial" font-size="13" fill="#555">Actual mean daily share; quartiles are based on prior-day competitor share</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{width - right}" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>',
    ]
    bar_width = (width - left - right) / max(1, len(buckets)) / 6
    for bucket_index, bucket in enumerate(buckets):
        for quartile in range(1, 5):
            row = next((item for item in rows if item["release_age_bucket"] == bucket and int(item["competitor_pressure_quartile"]) == quartile), None)
            if row is None:
                continue
            value = float(row["mean_actual_share"])
            x = sx(bucket_index, quartile) - bar_width / 2
            y = sy(value)
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bottom - y:.1f}" fill="{colors[quartile]}"/>')
        parts.append(f'<text x="{left + (bucket_index + 0.5) / len(buckets) * (width - left - right):.1f}" y="{bottom + 22}" font-family="Arial" font-size="10" text-anchor="middle">{bucket}</text>')
    for quartile in range(1, 5):
        parts.append(f'<rect x="{width - right + 20}" y="{85 + quartile * 18}" width="12" height="12" fill="{colors[quartile]}"/>')
        parts.append(f'<text x="{width - right + 38}" y="{95 + quartile * 18}" font-family="Arial" font-size="12">Q{quartile}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_exact_day_effect_svg(path: Path, rows: list[dict[str, object]]) -> None:
    values = [row for row in rows if row.get("competitor_share_coef") not in (None, "")]
    if not values:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"840\" height=\"460\"></svg>\n", encoding="utf-8")
        return
    values = sorted(values, key=lambda row: int(row["release_age_day"]))
    width, height = 880, 460
    left, right, top, bottom = 70, 35, 60, 370
    xs = [int(row["release_age_day"]) for row in values]
    ys = [float(row["competitor_share_coef"]) for row in values]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(0.0, min(ys)), max(0.0, max(ys))
    if x_min == x_max:
        x_max += 1
    if y_min == y_max:
        y_min -= 1
        y_max += 1

    def sx(value: int) -> float:
        return left + (value - x_min) / (x_max - x_min) * (width - left - right)

    def sy(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    points = " ".join(f'{sx(x):.1f},{sy(y):.1f}' for x, y in zip(xs, ys))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">Exact release-day competitor coefficient</text>',
        '<text x="36" y="55" font-family="Arial" font-size="13" fill="#555">Only release days with enough training rows are shown</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{width - right}" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{sy(0.0):.1f}" x2="{width - right}" y2="{sy(0.0):.1f}" stroke="#aaa" stroke-dasharray="4 4"/>',
        f'<polyline points="{points}" fill="none" stroke="#c15b3f" stroke-width="2.4"/>',
    ]
    for x, y in zip(xs, ys):
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="#c15b3f"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_response_analysis_readme(
    path: Path,
    *,
    bucket_rows: list[dict[str, object]],
    quartile_rows: list[dict[str, object]],
    exact_day_rows: list[dict[str, object]],
) -> None:
    strongest = None
    if bucket_rows:
        strongest = min(
            [row for row in bucket_rows if row.get("competitor_share_coef") is not None],
            key=lambda row: float(row["competitor_share_coef"]),
            default=None,
        )
    path.write_text(
        "\n".join(
            [
                "Release-age and competitor-pressure response analysis",
                "",
                "This analysis estimates how the prior-day competitor-share effect varies by",
                "movie release age and by competitor-pressure quartile.",
                "",
                f"Release-age bucket rows: {len(bucket_rows)}",
                f"Competition quartile response rows: {len(quartile_rows)}",
                f"Exact release-day rows: {len(exact_day_rows)}",
                f"Most negative bucket coefficient: {strongest['release_age_bucket'] if strongest else ''} {strongest['competitor_share_coef'] if strongest else ''}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_response_analysis_outputs(out_dir: Path, panel_rows: list[dict[str, object]]) -> None:
    response_dir = out_dir / "response_analysis"
    response_dir.mkdir(parents=True, exist_ok=True)
    bucket_rows = bucketed_competition_effect_rows(panel_rows)
    exact_day_rows = exact_release_day_competition_effect_rows(panel_rows)
    quartile_rows = competition_quartile_response_rows(panel_rows)
    einav.write_csv(
        response_dir / "release_age_bucket_competition_effects.csv",
        bucket_rows,
        [
            "release_age_bucket",
            "train_n",
            "test_2026_n",
            "train_r2",
            "holdout_rmse_logit_share",
            "holdout_mae_logit_share",
            "holdout_r2_logit_share",
            "directional_accuracy",
            "own_prior_share_coef",
            "own_prior_share_p_normal_approx",
            "competitor_share_coef",
            "competitor_share_se_naive",
            "competitor_share_t_stat",
            "competitor_share_p_normal_approx",
        ],
    )
    einav.write_csv(
        response_dir / "release_age_day_competition_effects.csv",
        exact_day_rows,
        [
            "release_age_day",
            "release_age_bucket",
            "train_n",
            "test_2026_n",
            "train_r2",
            "holdout_rmse_logit_share",
            "holdout_r2_logit_share",
            "competitor_share_coef",
            "competitor_share_se_naive",
            "competitor_share_t_stat",
            "competitor_share_p_normal_approx",
        ],
    )
    einav.write_csv(
        response_dir / "competition_quartile_response.csv",
        quartile_rows,
        [
            "release_age_bucket",
            "competitor_pressure_quartile",
            "n",
            "mean_competitor_share",
            "mean_actual_share",
            "mean_predicted_share",
            "mean_share_residual",
            "rmse_share",
            "mae_share",
            "rmse_logit_share",
            "mean_actual_gross_usd",
            "mean_implied_predicted_gross_usd",
        ],
    )
    write_bucket_bar_svg(
        response_dir / "figure_competitor_effect_by_release_age_bucket.svg",
        bucket_rows,
        title="Competitor-share coefficient by release-age bucket",
        value_col="competitor_share_coef",
        y_label="Coefficient on prior-day competitor share",
    )
    write_bucket_bar_svg(
        response_dir / "figure_oos_rmse_by_release_age_bucket.svg",
        bucket_rows,
        title="2026 OOS RMSE by release-age bucket",
        value_col="holdout_rmse_logit_share",
        y_label="RMSE logit daily share",
    )
    write_quartile_response_svg(response_dir / "figure_response_by_competition_quartile.svg", quartile_rows)
    write_exact_day_effect_svg(response_dir / "figure_exact_day_competitor_coef.svg", exact_day_rows)
    write_response_analysis_readme(
        response_dir / "README.txt",
        bucket_rows=bucket_rows,
        quartile_rows=quartile_rows,
        exact_day_rows=exact_day_rows,
    )


def run_database_regression(
    *,
    out_dir: Path,
    einav_dir: Path,
    database_url: str | None,
    start_date: dt.date,
    end_date: dt.date | None,
    wide_theater_threshold: int,
    max_age_weeks: int,
    sample: str,
    write_testing: bool = True,
) -> None:
    conn = connect_database(database_url)
    try:
        daily_rows = load_database_daily_rows(conn, start_date=start_date, end_date=end_date)
    finally:
        conn.close()
    einav_effects = load_einav_effect_map(einav_dir)
    panel_rows, feature_rows = build_daily_movie_panel(
        daily_rows,
        start_date=start_date,
        end_date=end_date,
        wide_theater_threshold=wide_theater_threshold,
        max_age_weeks=max_age_weeks,
        einav_effects=einav_effects,
        sample=sample,
    )
    if len(panel_rows) < 50:
        raise SystemExit("Database did not produce enough usable daily movie rows for regression.")
    coefficients_by_model, comparison_rows, hypothesis_rows, predictions = run_models(panel_rows)
    write_outputs(
        out_dir,
        panel_rows=panel_rows,
        feature_rows=feature_rows,
        coefficients_by_model=coefficients_by_model,
        comparison_rows=comparison_rows,
        hypothesis_rows=hypothesis_rows,
        predictions=predictions,
        start_date=start_date,
        end_date=end_date,
        wide_theater_threshold=wide_theater_threshold,
        max_age_weeks=max_age_weeks,
        sample=sample,
    )
    focused_dirname = "all_in_theaters_2026_oos" if sample == "all-in-theaters" else "focused_2026_oos"
    focused_metrics = write_focused_2026_outputs(out_dir, panel_rows, dirname=focused_dirname)
    comparison_samples = [(sample, panel_rows, focused_metrics)]
    if sample != "wide-first-10-weeks":
        wide_panel_rows, _wide_feature_rows = build_daily_movie_panel(
            daily_rows,
            start_date=start_date,
            end_date=end_date,
            wide_theater_threshold=wide_theater_threshold,
            max_age_weeks=DEFAULT_MAX_AGE_WEEKS,
            einav_effects=einav_effects,
            sample="wide-first-10-weeks",
        )
        if len(wide_panel_rows) >= 50:
            wide_metrics = write_focused_2026_outputs(out_dir, wide_panel_rows, dirname="focused_2026_oos")
            comparison_samples.append(("wide-first-10-weeks", wide_panel_rows, wide_metrics))
    if sample == "all-in-theaters":
        write_sample_comparison(out_dir, sample_comparison_rows(comparison_samples))
        write_response_analysis_outputs(out_dir, panel_rows)
    if write_testing:
        write_testing_outputs(
            out_dir,
            raw_rows=daily_rows,
            panel_rows=panel_rows,
            start_date=start_date,
            end_date=end_date,
            wide_theater_threshold=wide_theater_threshold,
            max_age_weeks=max_age_weeks,
            einav_effects=einav_effects,
            sample=sample,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--einav-dir", type=Path, default=DEFAULT_EINAV_DIR)
    parser.add_argument("--database-url", default=database_url_from_env())
    parser.add_argument("--start-date", default=DEFAULT_START_DATE.isoformat())
    parser.add_argument("--end-date")
    parser.add_argument("--sample", choices=SAMPLE_CHOICES, default=DEFAULT_SAMPLE)
    parser.add_argument("--wide-theater-threshold", type=int, default=DEFAULT_WIDE_THEATER_THRESHOLD)
    parser.add_argument("--max-age-weeks", type=int, default=DEFAULT_MAX_AGE_WEEKS)
    parser.add_argument("--skip-testing", action="store_true", help="Only write the core daily regression artifacts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_database_regression(
        out_dir=args.out_dir,
        einav_dir=args.einav_dir,
        database_url=args.database_url,
        start_date=parse_date(args.start_date),
        end_date=date_or_none(args.end_date),
        wide_theater_threshold=args.wide_theater_threshold,
        max_age_weeks=args.max_age_weeks,
        sample=args.sample,
        write_testing=not args.skip_testing,
    )
    print(f"Wrote daily competition-lag artifacts to {args.out_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.exit(1)
