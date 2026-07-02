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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pm_box_office.db.connection import connect_database, database_url_from_env
from pm_box_office.research.papers import recreate_competition_opening_weekend as opening
from pm_box_office.research.papers.common import (
    format_number,
    mean,
    r2_score,
    write_csv,
)


DEFAULT_OUT_DIR = Path("results/papers/day_by_day_opening_weekend")
THREE_DAY_TARGET = "3_day"
FIVE_DAY_TARGET = "5_day"
DEFAULT_TARGET_TYPES = (THREE_DAY_TARGET, FIVE_DAY_TARGET)
DEFAULT_SNAPSHOT_DAYS_BY_TARGET = {
    THREE_DAY_TARGET: tuple(range(-14, 4)),
    FIVE_DAY_TARGET: tuple(range(-14, 6)),
}
DEFAULT_SNAPSHOT_DAYS = tuple(range(-14, 6))
DEFAULT_TRAIN_START_YEAR = 2022
DEFAULT_TRAIN_END_YEAR = 2025
DEFAULT_TEST_START_YEAR = 2023
DEFAULT_TEST_END_YEAR = 2026
DEFAULT_MIN_OPENING_DAY_GROSS = 5_000_000
DEFAULT_TEST_MIN_OPENING_DAY_GROSS = 5_000_000
DEFAULT_COMPETITIVE_SELECTION_SPLITS = ((2022, 2023, 2024), (2022, 2024, 2025), (2022, 2025, 2026))
DEFAULT_TRAIN_OPENING_DAY_GROSS_CUTOFFS = (
    0,
    250_000,
    500_000,
    1_000_000,
    2_000_000,
    3_000_000,
    5_000_000,
    10_000_000,
)
MIN_BUCKET_INTERVAL_RESIDUALS = 5
INTERVAL_METHODS = ("empirical_quantile", "conformal_abs", "loo_conformal_abs")
OPENING_WEEKEND_OFFSETS = (0, 1, 2)
FIVE_DAY_OPENING_OFFSETS = (0, 1, 2, 3, 4)
COMPETITIVE_CHECKPOINT_DAYS = (-1, 1, 2)
COMPETITIVE_CHECKPOINT_TRAIN_START_YEAR = 2022
COMPETITIVE_CHECKPOINT_TRAIN_END_YEAR = 2025
COMPETITIVE_CHECKPOINT_TEST_START_YEAR = 2026
COMPETITIVE_CHECKPOINT_TEST_END_YEAR = 2026
DEFAULT_MIN_BOP_FORECAST_MIDPOINT = 5_000_000
WIKI_BOP_TIMING_DAYS = tuple(range(-14, 4))
WIKI_BOP_TIMING_TRAIN_START_YEAR = 2022
WIKI_BOP_TIMING_TRAIN_END_YEAR = 2024
WIKI_BOP_TIMING_TEST_START_YEAR = 2025
WIKI_BOP_TIMING_TEST_END_YEAR = 2026
DEFAULT_CONFIG_SEARCH_OUT_DIR = Path("results/papers/day_by_day_model_config_search")
CONFIG_SEARCH_SNAPSHOT_DAYS = tuple(range(-14, 3))
CONFIG_SEARCH_RANK_DAYS = (-2, -1, 0, 1, 2)
CONFIG_SEARCH_TRAIN_START_POLICIES = (2020, 2021, 2022)
CONFIG_SEARCH_VALIDATION_YEARS = (2023, 2024, 2025)
CONFIG_SEARCH_FINAL_TEST_YEAR = 2026
CONFIG_SEARCH_TRAIN_BOP_CUTOFFS = (0, 2_500_000, 5_000_000, 10_000_000, 15_000_000)
CONFIG_SEARCH_TEST_BOP_CUTOFFS = (5_000_000, 10_000_000, 15_000_000, 25_000_000)
CONFIG_SEARCH_PRIMARY_TEST_BOP_CUTOFF = 5_000_000


FIXED_BUCKET_TERMS = [
    "bop_estimate_bucket_15_30m",
    "bop_estimate_bucket_30_60m",
    "bop_estimate_bucket_60_100m",
    "bop_estimate_bucket_100m_plus",
]
WIKI_ACTIVITY_TERMS = ["log1p_V", "log1p_U", "log1p_R", "log1p_E"]
BOP_HISTORY_TERMS = [
    "log1p_bop_forecast_count_as_of",
    "bop_latest_lead_days",
    "bop_revision_from_first_pct",
    "bop_revision_from_previous_pct",
    "log1p_V",
]

SNAPSHOT_MODEL_TERMS = {
    "raw_bop_snapshot": [],
    "calibrated_bop_snapshot": ["log1p_bop_forecast_midpoint"],
    "bop_residual_bucket_snapshot": FIXED_BUCKET_TERMS,
    "bop_residual_wiki_snapshot": FIXED_BUCKET_TERMS + ["log1p_V"],
    "bop_residual_wiki_history_snapshot": FIXED_BUCKET_TERMS + BOP_HISTORY_TERMS,
    "bop_residual_competition_snapshot": FIXED_BUCKET_TERMS + ["log1p_competitor_total_gross_lag7"],
    "bop_residual_wiki_competition_snapshot": FIXED_BUCKET_TERMS
    + [
        "log1p_V",
        "log1p_competitor_total_gross_lag7",
        "bop_forecast_range_width_pct",
    ],
}

WIKI_BOP_TIMING_MODEL_TERMS = {
    "raw_bop_available": [],
    "bop_residual_wiki_views": ["log1p_V"],
    "bop_residual_wiki_full_activity": WIKI_ACTIVITY_TERMS,
    "bop_residual_wiki_full_activity_plus_theaters": WIKI_ACTIVITY_TERMS + ["log1p_opening_theaters"],
}

WIKI_REMAINING_TIMING_DAYS = (1, 2)
WIKI_REMAINING_MODEL_TERMS = {
    "remaining_baseline_ratio": [],
    "remaining_residual_wiki_views": ["log1p_V"],
    "remaining_residual_wiki_full_activity": WIKI_ACTIVITY_TERMS,
}

WIKI_COMP_COMBO_DAYS = tuple(range(-14, 3))
WIKI_COMP_COMBO_MODEL_TERMS = {
    "raw_bop_available": [],
    "bop_residual_wiki_views": ["log1p_V"],
    "bop_residual_wiki_full_activity": WIKI_ACTIVITY_TERMS,
    "bop_residual_comp_share_attraction": ["share_attraction_competitor_pressure_lag7"],
    "bop_residual_comp_logit_demand": ["logit_demand_focal_vs_competition_lag7"],
    "bop_residual_wiki_views_plus_share_attraction": ["log1p_V", "share_attraction_competitor_pressure_lag7"],
    "bop_residual_wiki_views_plus_logit_demand": ["log1p_V", "logit_demand_focal_vs_competition_lag7"],
    "bop_residual_wiki_full_plus_share_attraction": WIKI_ACTIVITY_TERMS
    + ["share_attraction_competitor_pressure_lag7"],
    "bop_residual_wiki_full_plus_logit_demand": WIKI_ACTIVITY_TERMS
    + ["logit_demand_focal_vs_competition_lag7"],
    "bop_residual_wiki_views_plus_both_comp": [
        "log1p_V",
        "share_attraction_competitor_pressure_lag7",
        "logit_demand_focal_vs_competition_lag7",
    ],
    "bop_residual_wiki_full_plus_both_comp": WIKI_ACTIVITY_TERMS
    + [
        "share_attraction_competitor_pressure_lag7",
        "logit_demand_focal_vs_competition_lag7",
    ],
}
WIKI_COMP_COMBO_SINGLE_SIGNAL_MODELS = {
    "bop_residual_wiki_views",
    "bop_residual_wiki_full_activity",
    "bop_residual_comp_share_attraction",
    "bop_residual_comp_logit_demand",
}
WIKI_COMP_COMBO_COMBINATION_MODELS = {
    "bop_residual_wiki_views_plus_share_attraction",
    "bop_residual_wiki_views_plus_logit_demand",
    "bop_residual_wiki_full_plus_share_attraction",
    "bop_residual_wiki_full_plus_logit_demand",
    "bop_residual_wiki_views_plus_both_comp",
    "bop_residual_wiki_full_plus_both_comp",
}
WIKI_COMP_COMBO_SENSITIVITY_MODELS = {
    "bop_residual_wiki_views_plus_both_comp",
    "bop_residual_wiki_full_plus_both_comp",
}
WIKI_COMP_IMPROVED_MODEL_NAMES = [
    "stacked_views_share",
    "stacked_full_share",
    "stacked_views_logit",
    "stacked_full_logit",
    "gated_timing_rule",
    "wiki_full_plus_residualized_share",
    "wiki_full_plus_residualized_logit",
    "wiki_views_plus_residualized_share",
    "wiki_views_plus_residualized_logit",
    "ridge_views_plus_both_comp",
    "ridge_full_plus_both_comp",
]
WIKI_COMP_IMPROVED_BASE_MODELS = {
    "bop_residual_wiki_views": ["log1p_V"],
    "bop_residual_wiki_full_activity": WIKI_ACTIVITY_TERMS,
    "bop_residual_comp_share_attraction": ["share_attraction_competitor_pressure_lag7"],
    "bop_residual_comp_logit_demand": ["logit_demand_focal_vs_competition_lag7"],
}
WIKI_COMP_STACKED_MODELS = {
    "stacked_views_share": ("bop_residual_wiki_views", "bop_residual_comp_share_attraction"),
    "stacked_full_share": ("bop_residual_wiki_full_activity", "bop_residual_comp_share_attraction"),
    "stacked_views_logit": ("bop_residual_wiki_views", "bop_residual_comp_logit_demand"),
    "stacked_full_logit": ("bop_residual_wiki_full_activity", "bop_residual_comp_logit_demand"),
}
WIKI_COMP_RESIDUALIZED_MODELS = {
    "wiki_full_plus_residualized_share": (
        WIKI_ACTIVITY_TERMS,
        "share_attraction_competitor_pressure_lag7",
        "residualized_share_given_wiki_full",
    ),
    "wiki_full_plus_residualized_logit": (
        WIKI_ACTIVITY_TERMS,
        "logit_demand_focal_vs_competition_lag7",
        "residualized_logit_given_wiki_full",
    ),
    "wiki_views_plus_residualized_share": (
        ["log1p_V"],
        "share_attraction_competitor_pressure_lag7",
        "residualized_share_given_wiki_views",
    ),
    "wiki_views_plus_residualized_logit": (
        ["log1p_V"],
        "logit_demand_focal_vs_competition_lag7",
        "residualized_logit_given_wiki_views",
    ),
}
WIKI_COMP_RIDGE_MODELS = {
    "ridge_views_plus_both_comp": [
        "log1p_V",
        "share_attraction_competitor_pressure_lag7",
        "logit_demand_focal_vs_competition_lag7",
    ],
    "ridge_full_plus_both_comp": WIKI_ACTIVITY_TERMS
    + [
        "share_attraction_competitor_pressure_lag7",
        "logit_demand_focal_vs_competition_lag7",
    ],
}
WIKI_COMP_RIDGE_LAMBDA = 1.0

CHECKPOINT_BY_SNAPSHOT_DAY = {
    -1: "pre_release_tminus1",
    1: "friday_actual",
    2: "saturday_actual",
}

PAPER_COMPETITION_FEATURES = [
    "share_attraction_competitor_pressure_lag7",
    "logit_demand_focal_vs_competition_lag7",
]

COMPETITIVE_CHECKPOINT_MODEL_TERMS = {
    "tminus1_competition": FIXED_BUCKET_TERMS + ["competitive_residual_lag7"],
    "tminus1_share_attraction": FIXED_BUCKET_TERMS + ["share_attraction_competitor_pressure_lag7"],
    "tminus1_logit_demand": FIXED_BUCKET_TERMS + ["logit_demand_focal_vs_competition_lag7"],
    "friday_actual_only": [
        "log1p_preview_gross_known",
        "log1p_known_friday_gross",
        "known_gross_so_far_pct_of_bop_midpoint",
    ],
    "friday_actual_competition": [
        "log1p_preview_gross_known",
        "log1p_known_friday_gross",
        "known_gross_so_far_pct_of_bop_midpoint",
        "competitive_residual_lag7",
    ],
    "friday_actual_share_attraction": [
        "log1p_preview_gross_known",
        "log1p_known_friday_gross",
        "known_gross_so_far_pct_of_bop_midpoint",
        "share_attraction_competitor_pressure_lag7",
    ],
    "friday_actual_logit_demand": [
        "log1p_preview_gross_known",
        "log1p_known_friday_gross",
        "known_gross_so_far_pct_of_bop_midpoint",
        "logit_demand_focal_vs_competition_lag7",
    ],
    "saturday_actual_only": [
        "log1p_preview_gross_known",
        "log1p_known_friday_gross",
        "log1p_known_saturday_gross",
        "known_gross_so_far_pct_of_bop_midpoint",
    ],
    "saturday_actual_competition": [
        "log1p_preview_gross_known",
        "log1p_known_friday_gross",
        "log1p_known_saturday_gross",
        "known_gross_so_far_pct_of_bop_midpoint",
        "competitive_residual_lag7",
    ],
    "saturday_actual_share_attraction": [
        "log1p_preview_gross_known",
        "log1p_known_friday_gross",
        "log1p_known_saturday_gross",
        "known_gross_so_far_pct_of_bop_midpoint",
        "share_attraction_competitor_pressure_lag7",
    ],
    "saturday_actual_logit_demand": [
        "log1p_preview_gross_known",
        "log1p_known_friday_gross",
        "log1p_known_saturday_gross",
        "known_gross_so_far_pct_of_bop_midpoint",
        "logit_demand_focal_vs_competition_lag7",
    ],
}

COMPETITIVE_CHECKPOINT_MODELS_BY_DAY = {
    -1: ["raw_bop_reconciled", "tminus1_competition", "tminus1_share_attraction", "tminus1_logit_demand"],
    1: [
        "raw_bop_reconciled",
        "friday_actual_only",
        "friday_actual_competition",
        "friday_actual_share_attraction",
        "friday_actual_logit_demand",
    ],
    2: [
        "raw_bop_reconciled",
        "saturday_actual_only",
        "saturday_actual_competition",
        "saturday_actual_share_attraction",
        "saturday_actual_logit_demand",
    ],
}

FALLBACK_MODEL_NAME = "fallback_wiki_competition_snapshot"
FALLBACK_TERMS = opening.FALLBACK_MODEL_TERMS
WIKI_FALLBACK_MODEL_NAME = "fallback_wiki_snapshot"
WIKI_FALLBACK_TERMS = opening.WIKI_TERMS


@dataclass(frozen=True)
class TargetSpec:
    target_type: str
    day_count: int
    offsets: tuple[int, ...]
    snapshot_days: tuple[int, ...]


TARGET_SPECS = {
    THREE_DAY_TARGET: TargetSpec(
        target_type=THREE_DAY_TARGET,
        day_count=3,
        offsets=OPENING_WEEKEND_OFFSETS,
        snapshot_days=DEFAULT_SNAPSHOT_DAYS_BY_TARGET[THREE_DAY_TARGET],
    ),
    FIVE_DAY_TARGET: TargetSpec(
        target_type=FIVE_DAY_TARGET,
        day_count=5,
        offsets=FIVE_DAY_OPENING_OFFSETS,
        snapshot_days=DEFAULT_SNAPSHOT_DAYS_BY_TARGET[FIVE_DAY_TARGET],
    ),
}


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


def parse_target_types(value: str) -> list[str]:
    target_types = [part.strip() for part in value.split(",") if part.strip()]
    if not target_types:
        raise ValueError("at least one target type is required")
    unknown = [target_type for target_type in target_types if target_type not in TARGET_SPECS]
    if unknown:
        raise ValueError(f"unknown target type(s): {', '.join(unknown)}")
    deduped: list[str] = []
    for target_type in target_types:
        if target_type not in deduped:
            deduped.append(target_type)
    return deduped


def target_specs_for_types(target_types: Iterable[str] | None) -> list[TargetSpec]:
    values = list(target_types) if target_types is not None else list(DEFAULT_TARGET_TYPES)
    specs: list[TargetSpec] = []
    for target_type in values:
        if target_type not in TARGET_SPECS:
            raise ValueError(f"unknown target type {target_type!r}")
        specs.append(TARGET_SPECS[target_type])
    return specs


def target_day_count_for_type(target_type: str) -> int:
    return TARGET_SPECS.get(target_type, TARGET_SPECS[THREE_DAY_TARGET]).day_count


def default_snapshot_days_for_target_types(target_types: Iterable[str]) -> list[int]:
    days: set[int] = set()
    for spec in target_specs_for_types(target_types):
        days.update(spec.snapshot_days)
    return sorted(days)


def bop_lead_bucket_for_days(lead_days: int | None) -> str:
    if lead_days is None:
        return "missing_lead"
    if lead_days < 0:
        return "after_open"
    if lead_days <= 2:
        return "0_2d"
    if lead_days <= 7:
        return "3_7d"
    if lead_days <= 14:
        return "8_14d"
    return "15d_plus"


def bop_lead_days(forecast: opening.BoxofficeProForecast | None) -> int | None:
    if forecast is None:
        return None
    return (forecast.target_start_date - forecast.published_date).days


def forecast_target_day_count(forecast: opening.BoxofficeProForecast) -> int:
    return (forecast.target_end_date - forecast.target_start_date).days + 1


def is_primary_bop_forecast(forecast: opening.BoxofficeProForecast) -> bool:
    lead_days = bop_lead_days(forecast)
    return lead_days is not None and lead_days >= 0


def eligible_opening_bop_forecasts(
    forecasts: Iterable[opening.BoxofficeProForecast],
    *,
    as_of_date: dt.date,
    movie_id: int,
    target_start_date: dt.date,
    target_day_count: int | None = None,
) -> list[opening.BoxofficeProForecast]:
    return sorted(
        [
            forecast
            for forecast in forecasts
            if forecast.movie_id == movie_id
            and forecast.forecast_metric == "domestic_opening_weekend"
            and forecast.target_start_date == target_start_date
            and forecast.published_date <= as_of_date
            and is_primary_bop_forecast(forecast)
            and (target_day_count is None or forecast_target_day_count(forecast) == target_day_count)
        ],
        key=lambda forecast: (
            forecast.published_date,
            -(forecast.source_rank or 999999),
            forecast.article_id,
            forecast.prediction_id,
        ),
    )


def latest_primary_opening_bop_forecast(
    forecasts: Iterable[opening.BoxofficeProForecast],
    *,
    as_of_date: dt.date,
    movie_id: int,
    target_start_date: dt.date,
    target_day_count: int | None = None,
) -> opening.BoxofficeProForecast | None:
    eligible = eligible_opening_bop_forecasts(
        forecasts,
        as_of_date=as_of_date,
        movie_id=movie_id,
        target_start_date=target_start_date,
        target_day_count=target_day_count,
    )
    return eligible[-1] if eligible else None


def bop_forecast_history_features(
    forecasts: Iterable[opening.BoxofficeProForecast],
    *,
    as_of_date: dt.date,
    movie_id: int,
    target_start_date: dt.date,
    target_day_count: int | None = None,
) -> dict[str, object]:
    eligible = eligible_opening_bop_forecasts(
        forecasts,
        as_of_date=as_of_date,
        movie_id=movie_id,
        target_start_date=target_start_date,
        target_day_count=target_day_count,
    )
    if not eligible:
        return {
            "bop_forecast_count_as_of": 0.0,
            "log1p_bop_forecast_count_as_of": 0.0,
            "bop_first_forecast_midpoint": 0.0,
            "bop_previous_forecast_midpoint": 0.0,
            "bop_latest_lead_days": 0.0,
            "bop_latest_lead_bucket": "missing_lead",
            "bop_revision_from_first_pct": 0.0,
            "bop_revision_from_previous_pct": 0.0,
        }
    latest = eligible[-1]
    first = eligible[0]
    previous = eligible[-2] if len(eligible) >= 2 else None
    latest_midpoint = latest.midpoint_usd
    first_midpoint = first.midpoint_usd
    previous_midpoint = previous.midpoint_usd if previous is not None else 0.0
    latest_lead_days = bop_lead_days(latest)
    return {
        "bop_forecast_count_as_of": float(len(eligible)),
        "log1p_bop_forecast_count_as_of": opening.log1p(len(eligible)),
        "bop_first_forecast_midpoint": first_midpoint,
        "bop_previous_forecast_midpoint": previous_midpoint,
        "bop_latest_lead_days": float(latest_lead_days or 0),
        "bop_latest_lead_bucket": bop_lead_bucket_for_days(latest_lead_days),
        "bop_revision_from_first_pct": (
            (latest_midpoint - first_midpoint) / first_midpoint if first_midpoint > 0.0 else 0.0
        ),
        "bop_revision_from_previous_pct": (
            (latest_midpoint - previous_midpoint) / previous_midpoint
            if previous is not None and previous_midpoint > 0.0
            else 0.0
        ),
    }


def load_preview_grosses(conn: Any, *, start_date: dt.date, end_date: dt.date) -> list[opening.DailyGross]:
    rows = conn.execute(
        """
        SELECT
            rr.movie_id,
            dbo.box_office_date::date,
            SUM(dbo.gross_usd) AS gross_usd,
            MAX(dbo.theaters) AS theaters
        FROM daily_box_office dbo
        JOIN release_runs rr ON rr.release_run_id = dbo.release_run_id
        WHERE dbo.is_preview = 1
          AND dbo.gross_usd IS NOT NULL
          AND dbo.gross_usd > 0
          AND dbo.box_office_date::date BETWEEN %s AND %s
        GROUP BY rr.movie_id, dbo.box_office_date::date
        ORDER BY dbo.box_office_date::date, rr.movie_id
        """,
        (start_date.isoformat(), end_date.isoformat()),
    ).fetchall()
    return [
        opening.DailyGross(
            movie_id=int(row[0]),
            box_office_date=opening.parse_db_date(row[1]),
            gross_usd=int(row[2]),
            theaters=int(row[3] or 0),
        )
        for row in rows
    ]


def preview_features_for_movie(preview_grosses: list[opening.DailyGross], *, movie: opening.OpeningWeekendMovie) -> dict[str, object]:
    eligible = [
        row
        for row in preview_grosses
        if row.movie_id == movie.movie_id and row.box_office_date < movie.opening_date
    ]
    preview_gross = sum(float(row.gross_usd) for row in eligible)
    return {
        "preview_gross_known_usd": preview_gross,
        "preview_day_count": float(len({row.box_office_date for row in eligible})),
        "log1p_preview_gross_known": opening.log1p(preview_gross),
    }


def target_daily_grosses(
    gross_by_movie_date: dict[tuple[int, dt.date], opening.DailyGross],
    *,
    movie: opening.OpeningWeekendMovie,
    spec: TargetSpec,
) -> dict[int, float] | None:
    daily = {
        offset: float(row.gross_usd)
        for offset in spec.offsets
        if (row := gross_by_movie_date.get((movie.movie_id, movie.opening_date + dt.timedelta(days=offset)))) is not None
    }
    missing_offsets = [offset for offset in spec.offsets if offset not in daily]
    if not missing_offsets:
        return daily
    if spec.target_type == THREE_DAY_TARGET and movie.opening_weekend_revenue_usd > 0:
        fallback = {
            0: float(movie.opening_day_gross_usd),
            1: 0.0,
            2: max(0.0, float(movie.opening_weekend_revenue_usd - movie.opening_day_gross_usd)),
        }
        return fallback
    return None


def known_actual_features_for_target(
    *,
    snapshot_day: int,
    spec: TargetSpec,
    daily: dict[int, float],
    midpoint: float,
) -> dict[str, object]:
    known_offsets = tuple(offset for offset in spec.offsets if offset < snapshot_day)
    known_gross = sum(daily.get(offset, 0.0) for offset in known_offsets)
    target_gross = sum(daily.get(offset, 0.0) for offset in spec.offsets)
    remaining = max(0.0, target_gross - known_gross)
    return {
        "known_actual_offsets": ",".join(str(offset) for offset in known_offsets),
        "known_actual_day_count": len(known_offsets),
        "known_gross_so_far": known_gross,
        "known_gross_so_far_usd": known_gross,
        "remaining_gross": remaining,
        "remaining_gross_usd": remaining,
        "known_gross_so_far_pct_of_bop_midpoint": known_gross / midpoint if midpoint > 0.0 else 0.0,
        "target_is_complete_as_of_snapshot": known_offsets == spec.offsets,
    }


def forecast_stage_for_snapshot_and_target(snapshot_day: int, spec: TargetSpec) -> str:
    if snapshot_day < 0:
        return "pre_release"
    if snapshot_day < spec.day_count:
        return "in_window"
    if snapshot_day == spec.day_count:
        return "final_actual"
    return "reconciliation"


def checkpoint_known_actual_features(row: dict[str, object]) -> dict[str, object]:
    snapshot_day = int(row["snapshot_day"])
    midpoint = float(row.get("bop_forecast_midpoint", 0.0) or 0.0)
    preview_gross = float(row.get("preview_gross_known_usd", 0.0) or 0.0)
    friday_gross = float(row.get("opening_weekend_day0_gross_usd", 0.0) or 0.0) if snapshot_day >= 1 else 0.0
    saturday_gross = float(row.get("opening_weekend_day1_gross_usd", 0.0) or 0.0) if snapshot_day >= 2 else 0.0
    known_gross = preview_gross + friday_gross + saturday_gross
    return {
        "known_friday_gross_usd": friday_gross,
        "known_saturday_gross_usd": saturday_gross,
        "known_weekend_gross_so_far_usd": known_gross,
        "known_gross_so_far_pct_of_bop_midpoint": known_gross / midpoint if midpoint > 0.0 else 0.0,
        "log1p_known_friday_gross": opening.log1p(friday_gross),
        "log1p_known_saturday_gross": opening.log1p(saturday_gross),
        "log1p_known_weekend_gross_so_far": opening.log1p(known_gross),
    }


def competition_information_day(opening_date: dt.date, snapshot_day: int) -> dt.date:
    return opening_date + dt.timedelta(days=snapshot_day - 1)


def paper_competition_proxy_features(row: dict[str, object]) -> dict[str, object]:
    focal = max(0.0, float(row.get("bop_forecast_midpoint", 0.0) or 0.0))
    competitor = max(0.0, float(row.get("competitor_total_gross_lag7", 0.0) or 0.0))
    top1 = max(0.0, float(row.get("competitor_top1_gross_lag7", 0.0) or 0.0))
    background = max(1.0, competitor - top1)
    denominator = focal + competitor + background
    return {
        "share_attraction_focal_share_lag7": focal / denominator if denominator > 0.0 else 0.0,
        "share_attraction_competitor_pressure_lag7": competitor / denominator if denominator > 0.0 else 0.0,
        "share_attraction_top1_pressure_lag7": top1 / denominator if denominator > 0.0 else 0.0,
        "logit_demand_focal_vs_competition_lag7": math.log((focal + 1.0) / (competitor + background + 1.0)),
        "logit_demand_focal_vs_top1_lag7": math.log((focal + 1.0) / (top1 + background + 1.0)),
        "log1p_logit_demand_choice_set_lag7": opening.log1p(competitor + background),
    }


def build_day_by_day_feature_panel(
    movies: list[opening.OpeningWeekendMovie],
    daily_grosses: list[opening.DailyGross],
    wiki_by_movie: dict[int, dict[int, dict[str, float]]],
    bop_forecasts: list[opening.BoxofficeProForecast],
    *,
    snapshot_days: list[int],
    train_start_year: int,
    train_end_year: int,
    preview_grosses: list[opening.DailyGross] | None = None,
    target_types: Iterable[str] | None = None,
) -> list[dict[str, object]]:
    gross_by_movie_date = {(row.movie_id, row.box_office_date): row for row in daily_grosses}
    preview_grosses = preview_grosses or []
    specs = target_specs_for_types(target_types)
    forecasts_by_movie: dict[int, list[opening.BoxofficeProForecast]] = defaultdict(list)
    for forecast in bop_forecasts:
        forecasts_by_movie[forecast.movie_id].append(forecast)

    rows: list[dict[str, object]] = []
    for movie in movies:
        for spec in specs:
            daily = target_daily_grosses(gross_by_movie_date, movie=movie, spec=spec)
            if daily is None:
                continue
            target_gross = sum(daily[offset] for offset in spec.offsets)
            if target_gross <= 0.0:
                continue
            target_start_date = movie.opening_date
            target_end_date = movie.opening_date + dt.timedelta(days=spec.day_count - 1)
            for snapshot_day in snapshot_days:
                if snapshot_day not in spec.snapshot_days:
                    continue
                as_of_date = movie.opening_date + dt.timedelta(days=snapshot_day)
                feature_available_date = competition_information_day(movie.opening_date, snapshot_day)
                wiki = opening.wiki_values_as_of(
                    wiki_by_movie,
                    movie_id=movie.movie_id,
                    timing_day=snapshot_day,
                )
                movie_forecasts = forecasts_by_movie.get(movie.movie_id, [])
                focal_forecast = latest_primary_opening_bop_forecast(
                    movie_forecasts,
                    as_of_date=as_of_date,
                    movie_id=movie.movie_id,
                    target_start_date=target_start_date,
                    target_day_count=spec.day_count,
                )
                bop_history = bop_forecast_history_features(
                    movie_forecasts,
                    as_of_date=as_of_date,
                    movie_id=movie.movie_id,
                    target_start_date=target_start_date,
                    target_day_count=spec.day_count,
                )
                preview = preview_features_for_movie(preview_grosses, movie=movie)
                bop_features = opening.bop_forecast_features(focal_forecast)
                midpoint = float(bop_features.get("bop_forecast_midpoint", 0.0) or 0.0)
                known_features = known_actual_features_for_target(
                    snapshot_day=snapshot_day,
                    spec=spec,
                    daily=daily,
                    midpoint=midpoint,
                )
                row: dict[str, object] = {
                    "movie_id": movie.movie_id,
                    "title": movie.title,
                    "release_year": movie.release_year,
                    "release_run_id": movie.release_run_id,
                    "opening_date": movie.opening_date.isoformat(),
                    "target_type": spec.target_type,
                    "target_day_count": spec.day_count,
                    "target_start_date": target_start_date.isoformat(),
                    "target_end_date": target_end_date.isoformat(),
                    "target_gross": target_gross,
                    "target_gross_usd": target_gross,
                    "forecast_stage": forecast_stage_for_snapshot_and_target(snapshot_day, spec),
                    "snapshot_day": snapshot_day,
                    "timing_day": snapshot_day,
                    "wiki_timing_day": snapshot_day,
                    "competition_timing_day": (feature_available_date - movie.opening_date).days,
                    "bop_timing_day": snapshot_day,
                    "as_of_date": as_of_date.isoformat(),
                    "feature_available_date": feature_available_date.isoformat(),
                    "opening_theaters": movie.opening_theaters,
                    "opening_day_gross_usd": movie.opening_day_gross_usd,
                    "opening_weekend_revenue_usd": target_gross,
                    "opening_weekend_day0_gross_usd": daily.get(0, 0.0),
                    "opening_weekend_day1_gross_usd": daily.get(1, 0.0),
                    "opening_weekend_day2_gross_usd": daily.get(2, 0.0),
                    "opening_weekend_day3_gross_usd": daily.get(3, 0.0),
                    "opening_weekend_day4_gross_usd": daily.get(4, 0.0),
                    "target_log_opening_weekend": math.log(max(1.0, target_gross)),
                    "target_log_gross": math.log(max(1.0, target_gross)),
                    "log1p_opening_theaters": opening.log1p(movie.opening_theaters),
                    **preview,
                    **known_features,
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
                        as_of_date=feature_available_date,
                    ),
                    **bop_features,
                    **bop_history,
                }
                row.update(checkpoint_known_actual_features(row))
                row.update(paper_competition_proxy_features(row))
                row["known_weekend_gross_so_far_usd"] = (
                    float(row.get("preview_gross_known_usd", 0.0) or 0.0)
                    + float(row["known_gross_so_far"])
                )
                row["known_gross_so_far_pct_of_bop_midpoint"] = (
                    float(row["known_weekend_gross_so_far_usd"]) / midpoint if midpoint > 0.0 else 0.0
                )
                row["log1p_known_weekend_gross_so_far"] = opening.log1p(row["known_weekend_gross_so_far_usd"])
                row["target_log_remaining_gross"] = (
                    math.log(max(1.0, float(row["remaining_gross"])))
                    if float(row["remaining_gross"] or 0.0) > 0.0
                    else ""
                )
                for month in range(2, 13):
                    row[f"release_month_{month}"] = 1.0 if movie.opening_date.month == month else 0.0
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
        out.update(checkpoint_known_actual_features(out))
        out.update(paper_competition_proxy_features(out))
        out["known_weekend_gross_so_far_usd"] = (
            float(out.get("preview_gross_known_usd", 0.0) or 0.0)
            + float(out["known_gross_so_far"])
        )
        out["known_gross_so_far_pct_of_bop_midpoint"] = (
            float(out["known_weekend_gross_so_far_usd"]) / midpoint if midpoint > 0.0 else 0.0
        )
        out["log1p_known_weekend_gross_so_far"] = opening.log1p(out["known_weekend_gross_so_far_usd"])
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
    train_min_opening_day_gross: int = 0,
    test_min_opening_day_gross: int = 0,
    target_type: str | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    scoped = [
        row
        for row in rows
        if int(row["snapshot_day"]) == snapshot_day
        and (target_type is None or str(row.get("target_type", THREE_DAY_TARGET)) == target_type)
    ]
    train_rows = [
        row
        for row in scoped
        if train_start_year <= int(row["release_year"]) <= train_end_year
        and float(row.get("opening_day_gross_usd", 0.0) or 0.0) >= train_min_opening_day_gross
    ]
    holdout_rows = [
        row
        for row in scoped
        if test_start_year <= int(row["release_year"]) <= test_end_year
        and float(row.get("opening_day_gross_usd", 0.0) or 0.0) >= test_min_opening_day_gross
    ]
    return train_rows, holdout_rows


def bop_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if float(row.get("bop_forecast_available", 0.0) or 0.0) > 0.0]


def rows_with_target(rows: list[dict[str, object]], target_key: str) -> list[dict[str, object]]:
    return [row for row in rows if row.get(target_key, "") != ""]


def target_types_from_rows(rows: list[dict[str, object]], target_types: Iterable[str] | None = None) -> list[str]:
    if target_types is not None:
        return parse_target_types(",".join(target_types))
    discovered = sorted({str(row.get("target_type", THREE_DAY_TARGET)) for row in rows})
    return discovered or [THREE_DAY_TARGET]


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
    gross_errors = [actual - pred for actual, pred in zip(actual_gross, pred_gross)]
    apes = [abs(pred - actual) / actual for actual, pred in zip(actual_gross, pred_gross) if actual > 0.0]
    smapes = [
        abs(pred - actual) / ((abs(actual) + abs(pred)) / 2.0)
        for actual, pred in zip(actual_gross, pred_gross)
        if abs(actual) + abs(pred) > 0.0
    ]
    mape = mean(apes) if apes else None
    raw_baseline_pairs = [
        (float(row["actual_opening_weekend_revenue_usd"]), float(row["bop_forecast_midpoint"]))
        for row in prediction_rows
        if float(row.get("bop_forecast_midpoint", 0.0) or 0.0) > 0.0
    ]
    raw_baseline_mae = (
        mean([abs(actual - baseline) for actual, baseline in raw_baseline_pairs])
        if raw_baseline_pairs
        else None
    )
    actuals_multiplier_pairs = [
        (float(row["actual_opening_weekend_revenue_usd"]), float(row["actuals_multiplier_prediction_usd"]))
        for row in prediction_rows
        if row.get("actuals_multiplier_prediction_usd", "") != ""
    ]
    actuals_multiplier_mae = (
        mean([abs(actual - baseline) for actual, baseline in actuals_multiplier_pairs])
        if actuals_multiplier_pairs
        else None
    )
    model_mae = mean([abs(error) for error in gross_errors])

    def lift_value(baseline: float | None) -> float | None:
        return None if baseline is None else baseline - model_mae

    def lift_pct(baseline: float | None) -> float | None:
        if baseline is None or baseline <= 0.0:
            return None
        return (baseline - model_mae) / baseline

    directional_values = []
    for row in prediction_rows:
        estimate = float(row.get("bop_forecast_midpoint", 0.0) or 0.0)
        if estimate <= 0.0:
            continue
        actual_delta = float(row["actual_opening_weekend_revenue_usd"]) - estimate
        predicted_delta = float(row["predicted_opening_weekend_revenue_usd"]) - estimate
        if actual_delta == 0.0 or predicted_delta == 0.0:
            continue
        directional_values.append(1.0 if (actual_delta > 0.0) == (predicted_delta > 0.0) else 0.0)
    return {
        **base,
        "r2_log_revenue": format_number(r2_score(actual_log, pred_log)),
        "r2_gross": format_number(r2_score(actual_gross, pred_gross)),
        "mape_gross": format_number(mape),
        "smape_gross": format_number(mean(smapes) if smapes else None),
        "accuracy_pct": format_number(1.0 - mape if mape is not None else None),
        "mse_log_revenue": format_number(mean([error**2 for error in log_errors])),
        "mse_gross": format_number(mean([error**2 for error in gross_errors])),
        "mae_usd": format_number(model_mae),
        "rmse_usd": format_number(math.sqrt(mean([error**2 for error in gross_errors]))),
        "rmse_log_revenue": format_number(opening.rmse(log_errors)),
        "mae_log_revenue": format_number(opening.mae(log_errors)),
        "mean_absolute_log_error": format_number(opening.mae(log_errors)),
        "directional_accuracy_vs_estimate": format_number(mean(directional_values) if directional_values else None),
        "raw_estimate_mae_usd": format_number(raw_baseline_mae),
        "raw_estimate_mae_lift_usd": format_number(lift_value(raw_baseline_mae)),
        "raw_estimate_mae_lift_pct": format_number(lift_pct(raw_baseline_mae)),
        "actuals_multiplier_mae_usd": format_number(actuals_multiplier_mae),
        "actuals_multiplier_mae_lift_usd": format_number(lift_value(actuals_multiplier_mae)),
        "actuals_multiplier_mae_lift_pct": format_number(lift_pct(actuals_multiplier_mae)),
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
        "target_type": row.get("target_type", THREE_DAY_TARGET),
        "target_day_count": row.get("target_day_count", 3),
        "target_start_date": row.get("target_start_date", row.get("opening_date", "")),
        "target_end_date": row.get("target_end_date", ""),
        "target_gross_usd": row.get("target_gross_usd", row.get("opening_weekend_revenue_usd", "")),
        "forecast_stage": row["forecast_stage"],
        "snapshot_day": row["snapshot_day"],
        "as_of_date": row["as_of_date"],
        "feature_available_date": row.get("feature_available_date", ""),
        "movie_id": row["movie_id"],
        "title": row["title"],
        "release_year": row["release_year"],
        "opening_date": row["opening_date"],
        "bop_forecast_available": row.get("bop_forecast_available", 0.0),
        "bop_forecast_midpoint": row.get("bop_forecast_midpoint", 0.0),
        "bop_estimate_bucket": estimate_bucket_label(row),
        "bop_forecast_count_as_of": row.get("bop_forecast_count_as_of", 0.0),
        "bop_first_forecast_midpoint": row.get("bop_first_forecast_midpoint", 0.0),
        "bop_previous_forecast_midpoint": row.get("bop_previous_forecast_midpoint", 0.0),
        "bop_latest_lead_days": row.get("bop_latest_lead_days", 0.0),
        "bop_latest_lead_bucket": row.get("bop_latest_lead_bucket", "missing_lead"),
        "bop_revision_from_first_pct": row.get("bop_revision_from_first_pct", 0.0),
        "bop_revision_from_previous_pct": row.get("bop_revision_from_previous_pct", 0.0),
        "wiki_available": row.get("wiki_available", 0.0),
        "known_actual_offsets": row.get("known_actual_offsets", ""),
        "known_gross_so_far": row.get("known_gross_so_far", 0.0),
        "remaining_gross": row.get("remaining_gross", ""),
        "actual_log_opening_weekend": row["target_log_opening_weekend"],
        "predicted_log_opening_weekend": pred_log,
        "actual_opening_weekend_revenue_usd": row["opening_weekend_revenue_usd"],
        "predicted_opening_weekend_revenue_usd": pred_gross,
        "actuals_multiplier_prediction_usd": row.get("actuals_multiplier_prediction_usd", ""),
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


def cutoff_sweep_model_terms(snapshot_day: int) -> dict[str, list[str]]:
    model_terms = {
        name: terms
        for name, terms in SNAPSHOT_MODEL_TERMS.items()
        if name not in {"bop_residual_competition_snapshot", "bop_residual_wiki_competition_snapshot"}
    }
    if snapshot_day > -2:
        model_terms["bop_residual_competition_snapshot"] = SNAPSHOT_MODEL_TERMS[
            "bop_residual_competition_snapshot"
        ]
        model_terms["bop_residual_wiki_competition_snapshot"] = SNAPSHOT_MODEL_TERMS[
            "bop_residual_wiki_competition_snapshot"
        ]
    return model_terms


def evaluate_day_by_day_snapshots(
    panel_rows: list[dict[str, object]],
    *,
    snapshot_days: list[int],
    train_start_year: int,
    train_end_year: int,
    test_start_year: int,
    test_end_year: int,
    train_min_opening_day_gross: int = 0,
    test_min_opening_day_gross: int = 0,
    model_terms_by_snapshot: dict[int, dict[str, list[str]]] | None = None,
    fallback_model_name: str = FALLBACK_MODEL_NAME,
    fallback_terms: list[str] = FALLBACK_TERMS,
    include_standalone_fallback: bool = False,
    target_types: Iterable[str] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []

    for target_type in target_types_from_rows(panel_rows, target_types):
      for snapshot_day in snapshot_days:
        train_all, holdout_all = train_holdout_rows(
            panel_rows,
            snapshot_day=snapshot_day,
            train_start_year=train_start_year,
            train_end_year=train_end_year,
            test_start_year=test_start_year,
            test_end_year=test_end_year,
            train_min_opening_day_gross=train_min_opening_day_gross,
            test_min_opening_day_gross=test_min_opening_day_gross,
            target_type=target_type,
        )
        if not train_all and not holdout_all:
            continue
        fallback_model: opening.FittedModel | None = None
        fallback_train = [row for row in train_all if float(row.get("wiki_available", 0.0) or 0.0) > 0.0]
        if len(fallback_train) >= len(fallback_terms) + 2:
            fallback_model = opening.fit_model_for_target(fallback_train, fallback_terms, "target_log_opening_weekend")
            fallback_train_pred = opening.predict_log(fallback_model, fallback_train)
            fallback_residuals = prediction_residuals_for_interval(
                model_name=fallback_model_name,
                pred_log=fallback_train_pred,
                rows=fallback_train,
            )
            fallback_loo_residuals = loo_prediction_residuals_for_opening_model(
                rows=fallback_train,
                terms=fallback_terms,
            )
        else:
            fallback_residuals = []
            fallback_loo_residuals = []

        if include_standalone_fallback:
            fallback_metric_base = {
                "model": fallback_model_name,
                "target_type": target_type,
                "target_day_count": target_day_count_for_type(target_type),
                "forecast_stage": "all",
                "snapshot_day": snapshot_day,
                "train_start_year": train_start_year,
                "train_end_year": train_end_year,
                "test_start_year": test_start_year,
                "test_end_year": test_end_year,
            }
            missing_bop_holdout = [
                row for row in holdout_all if float(row.get("bop_forecast_available", 0.0) or 0.0) <= 0.0
            ]
            if fallback_model is None or len(missing_bop_holdout) < 2:
                for interval_method in INTERVAL_METHODS:
                    metric_rows.append(
                        {
                            **fallback_metric_base,
                            "population": "fallback_only",
                            "interval_method": interval_method,
                            "train_n": len(fallback_train),
                            "holdout_n": 0,
                            "bop_prediction_n": 0,
                            "fallback_prediction_n": 0,
                            "r2_log_revenue": "",
                            "r2_gross": "",
                            "mape_gross": "",
                            "accuracy_pct": "",
                            "mse_log_revenue": "",
                            "mse_gross": "",
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
            else:
                fallback_pred_log = opening.predict_log(fallback_model, missing_bop_holdout)
                fallback_bucket_residuals, fallback_global_residuals = interval_residuals_by_bucket(
                    rows=fallback_train,
                    residuals=fallback_residuals,
                )
                fallback_loo_bucket_residuals, fallback_loo_global_residuals = interval_residuals_by_bucket(
                    rows=fallback_train,
                    residuals=fallback_loo_residuals,
                )
                fallback_predictions = prediction_rows_for_interval_methods(
                    missing_bop_holdout,
                    fallback_pred_log,
                    model_name=fallback_model_name,
                    population="fallback_only",
                    prediction_source="fallback",
                    bucket_residuals=fallback_bucket_residuals,
                    global_residuals=fallback_global_residuals,
                    residuals_by_method={
                        "empirical_quantile": (fallback_bucket_residuals, fallback_global_residuals),
                        "conformal_abs": (fallback_bucket_residuals, fallback_global_residuals),
                        "loo_conformal_abs": (fallback_loo_bucket_residuals, fallback_loo_global_residuals),
                    },
                )
                for interval_method in INTERVAL_METHODS:
                    method_predictions = [
                        row for row in fallback_predictions if row["prediction_interval_method"] == interval_method
                    ]
                    if len(method_predictions) >= 2:
                        metric_rows.append(
                            metric_row_from_predictions(
                                base={
                                    **fallback_metric_base,
                                    "population": "fallback_only",
                                    "interval_method": interval_method,
                                    "train_n": len(fallback_train),
                                    "holdout_n": len(method_predictions),
                                    "bop_prediction_n": 0,
                                    "fallback_prediction_n": len(method_predictions),
                                },
                                prediction_rows=method_predictions,
                            )
                        )
                prediction_rows.extend(fallback_predictions)

        model_terms = (
            model_terms_by_snapshot.get(snapshot_day, SNAPSHOT_MODEL_TERMS)
            if model_terms_by_snapshot is not None
            else SNAPSHOT_MODEL_TERMS
        )
        for model_name, terms in model_terms.items():
            target_key = "target_log_opening_weekend" if model_name == "calibrated_bop_snapshot" else "target_log_bop_residual"
            train_bop = bop_rows(train_all)
            holdout_bop = bop_rows(holdout_all)
            if model_name != "raw_bop_snapshot":
                train_bop = rows_with_target(train_bop, target_key)

            metric_base = {
                "model": model_name,
                "target_type": target_type,
                "target_day_count": target_day_count_for_type(target_type),
                "forecast_stage": "all",
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
                                "accuracy_pct": "",
                                "mse_log_revenue": "",
                                "mse_gross": "",
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


def evaluate_expanding_window_snapshots(
    panel_rows: list[dict[str, object]],
    *,
    snapshot_days: list[int],
    train_start_year: int,
    test_start_year: int,
    test_end_year: int,
    target_types: Iterable[str] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []
    for test_year in range(test_start_year, test_end_year + 1):
        train_end_year = test_year - 1
        if train_end_year < train_start_year:
            continue
        preds, metrics, coefs, intervals = evaluate_day_by_day_snapshots(
            panel_rows,
            snapshot_days=snapshot_days,
            train_start_year=train_start_year,
            train_end_year=train_end_year,
            test_start_year=test_year,
            test_end_year=test_year,
            target_types=target_types,
        )
        prediction_rows.extend(preds)
        metric_rows.extend(metrics)
        coefficient_rows.extend(coefs)
        interval_rows.extend(intervals)
    return prediction_rows, metric_rows, coefficient_rows, interval_rows


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


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys):
        raise ValueError("correlation inputs must have equal length")
    if len(xs) < 2:
        return None
    mean_x = mean(xs)
    mean_y = mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denominator_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    denominator = denominator_x * denominator_y
    if denominator == 0.0:
        return None
    return numerator / denominator


def bop_estimate_accuracy_rows(
    movies: list[opening.OpeningWeekendMovie],
    bop_forecasts: list[opening.BoxofficeProForecast],
) -> list[dict[str, object]]:
    movies_by_id = {movie.movie_id: movie for movie in movies}
    rows: list[dict[str, object]] = []
    for forecast in bop_forecasts:
        movie = movies_by_id.get(forecast.movie_id)
        if (
            movie is None
            or forecast.forecast_metric != "domestic_opening_weekend"
            or forecast.target_start_date != movie.opening_date
            or movie.opening_weekend_revenue_usd <= 0
        ):
            continue
        actual = float(movie.opening_weekend_revenue_usd)
        midpoint = forecast.midpoint_usd
        signed_error = midpoint - actual
        absolute_error = abs(signed_error)
        lead_days = bop_lead_days(forecast)
        rows.append(
            {
                "prediction_id": forecast.prediction_id,
                "article_id": forecast.article_id,
                "article_url": forecast.article_url,
                "movie_id": movie.movie_id,
                "title": movie.title,
                "release_year": movie.release_year,
                "opening_date": movie.opening_date.isoformat(),
                "published_date": forecast.published_date.isoformat(),
                "lead_days": lead_days if lead_days is not None else "",
                "lead_bucket": bop_lead_bucket_for_days(lead_days),
                "primary_eligible": is_primary_bop_forecast(forecast),
                "source_context": forecast.source_context,
                "range_low_usd": forecast.range_low_usd,
                "range_high_usd": forecast.range_high_usd,
                "midpoint_usd": midpoint,
                "actual_opening_weekend_revenue_usd": actual,
                "signed_error_usd": signed_error,
                "absolute_error_usd": absolute_error,
                "squared_error_usd": signed_error**2,
                "absolute_percentage_error": absolute_error / actual if actual > 0.0 else "",
                "interval_hit": forecast.range_low_usd <= actual <= forecast.range_high_usd,
            }
        )
    return sorted(rows, key=lambda row: (str(row["opening_date"]), str(row["published_date"]), int(row["prediction_id"])))


def bop_estimate_accuracy_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["lead_bucket"])].append(row)

    out: list[dict[str, object]] = []
    bucket_order = {
        "0_2d": 0,
        "3_7d": 1,
        "8_14d": 2,
        "15d_plus": 3,
        "after_open": 4,
        "missing_lead": 5,
    }
    for lead_bucket, group in sorted(grouped.items(), key=lambda item: bucket_order.get(item[0], 999)):
        actuals = [float(row["actual_opening_weekend_revenue_usd"]) for row in group]
        predictions = [float(row["midpoint_usd"]) for row in group]
        signed_errors = [float(row["signed_error_usd"]) for row in group]
        absolute_errors = [float(row["absolute_error_usd"]) for row in group]
        squared_errors = [float(row["squared_error_usd"]) for row in group]
        apes = [float(row["absolute_percentage_error"]) for row in group if row["absolute_percentage_error"] != ""]
        mape = mean(apes) if apes else None
        out.append(
            {
                "lead_bucket": lead_bucket,
                "primary_eligible": all(bool(row["primary_eligible"]) for row in group),
                "estimate_count": len(group),
                "movie_count": len({row["movie_id"] for row in group}),
                "mape": format_number(mape),
                "accuracy_pct": format_number(1.0 - mape if mape is not None else None),
                "mae_usd": format_number(mean(absolute_errors)),
                "mse_usd": format_number(mean(squared_errors)),
                "rmse_usd": format_number(math.sqrt(mean(squared_errors))),
                "r2_gross": format_number(r2_score(actuals, predictions) if len(group) >= 2 else None),
                "r2_log_revenue": format_number(
                    r2_score(
                        [math.log(max(1.0, actual)) for actual in actuals],
                        [math.log(max(1.0, prediction)) for prediction in predictions],
                    )
                    if len(group) >= 2
                    else None
                ),
                "pearson_correlation": format_number(pearson_correlation(predictions, actuals)),
                "mean_signed_error_usd": format_number(mean(signed_errors)),
                "interval_hit_rate": format_number(
                    mean([1.0 if bool(row["interval_hit"]) else 0.0 for row in group])
                ),
            }
        )
    return out


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
    best: dict[tuple[str, str, str, int], dict[str, object]] = {}
    min_lift_usd = 1.0
    for row in metric_rows:
        if row["status"] != "ok" or row["mape_gross"] == "":
            continue
        actuals_lift = row.get("actuals_multiplier_mae_lift_usd", "")
        raw_lift = row.get("raw_estimate_mae_lift_usd", "")
        baseline_lift = actuals_lift if actuals_lift != "" else raw_lift
        if baseline_lift != "" and float(baseline_lift) <= min_lift_usd:
            continue
        key = (
            str(row.get("target_type", THREE_DAY_TARGET)),
            str(row["population"]),
            str(row["interval_method"]),
            int(row["snapshot_day"]),
        )
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
            "target_type": row.get("target_type", THREE_DAY_TARGET),
            "target_day_count": row.get("target_day_count", target_day_count_for_type(str(row.get("target_type", THREE_DAY_TARGET)))),
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
            "accuracy_pct": row["accuracy_pct"],
            "coverage_80": row["coverage_80"],
            "mean_interval_80_width_pct": row["mean_interval_80_width_pct"],
        }
        for row in sorted(
            best.values(),
            key=lambda item: (
                str(item.get("target_type", THREE_DAY_TARGET)),
                str(item["population"]),
                str(item["interval_method"]),
                int(item["snapshot_day"]),
            ),
        )
    ]


def parse_cutoff_list(value: str) -> list[int]:
    cutoffs = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not cutoffs:
        raise ValueError("at least one cutoff is required")
    if any(cutoff < 0 for cutoff in cutoffs):
        raise ValueError("cutoffs must be non-negative")
    return cutoffs


def evaluate_cutoff_sweep(
    panel_rows: list[dict[str, object]],
    *,
    snapshot_days: list[int],
    train_start_year: int,
    train_end_year: int,
    test_start_year: int,
    test_end_year: int,
    train_opening_day_gross_cutoffs: list[int],
    test_min_opening_day_gross: int,
    target_types: Iterable[str] | None = None,
) -> list[dict[str, object]]:
    model_terms_by_snapshot = {day: cutoff_sweep_model_terms(day) for day in snapshot_days}
    metric_rows: list[dict[str, object]] = []
    for train_cutoff in train_opening_day_gross_cutoffs:
        _, cutoff_metrics, _, _ = evaluate_day_by_day_snapshots(
            panel_rows,
            snapshot_days=snapshot_days,
            train_start_year=train_start_year,
            train_end_year=train_end_year,
            test_start_year=test_start_year,
            test_end_year=test_end_year,
            train_min_opening_day_gross=train_cutoff,
            test_min_opening_day_gross=test_min_opening_day_gross,
            model_terms_by_snapshot=model_terms_by_snapshot,
            fallback_model_name=WIKI_FALLBACK_MODEL_NAME,
            fallback_terms=WIKI_FALLBACK_TERMS,
            include_standalone_fallback=True,
            target_types=target_types,
        )
        metric_rows.extend(
            {
                "train_cutoff": train_cutoff,
                "test_cutoff": test_min_opening_day_gross,
                **row,
            }
            for row in cutoff_metrics
        )
    return metric_rows


def best_cutoff_sweep_rows(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best: dict[tuple[str, str, str, str, int], dict[str, object]] = {}
    for row in metric_rows:
        if row["status"] != "ok" or row["r2_gross"] == "":
            continue
        key = (
            str(row.get("target_type", THREE_DAY_TARGET)),
            str(row["model"]),
            str(row["population"]),
            str(row["interval_method"]),
            int(row["snapshot_day"]),
        )
        current = best.get(key)
        row_score = (
            float(row["r2_gross"]),
            float(row["r2_log_revenue"]) if row["r2_log_revenue"] != "" else float("-inf"),
            -float(row["mape_gross"]) if row["mape_gross"] != "" else float("-inf"),
            int(row["holdout_n"]),
        )
        if current is None:
            current_score = (float("-inf"), float("-inf"), float("-inf"), -1)
        else:
            current_score = (
                float(current["r2_gross"]),
                float(current["r2_log_revenue"]) if current["r2_log_revenue"] != "" else float("-inf"),
                -float(current["mape_gross"]) if current["mape_gross"] != "" else float("-inf"),
                int(current["holdout_n"]),
            )
        if current is None or row_score > current_score:
            best[key] = row
    return [
        {
            "target_type": row.get("target_type", THREE_DAY_TARGET),
            "target_day_count": row.get("target_day_count", target_day_count_for_type(str(row.get("target_type", THREE_DAY_TARGET)))),
            "model": row["model"],
            "population": row["population"],
            "interval_method": row["interval_method"],
            "snapshot_day": row["snapshot_day"],
            "best_train_cutoff": row["train_cutoff"],
            "test_cutoff": row["test_cutoff"],
            "train_n": row["train_n"],
            "holdout_n": row["holdout_n"],
            "bop_prediction_n": row["bop_prediction_n"],
            "fallback_prediction_n": row["fallback_prediction_n"],
            "r2_log_revenue": row["r2_log_revenue"],
            "r2_gross": row["r2_gross"],
            "mape_gross": row["mape_gross"],
            "accuracy_pct": row["accuracy_pct"],
            "mse_log_revenue": row["mse_log_revenue"],
            "mse_gross": row["mse_gross"],
            "rmse_log_revenue": row["rmse_log_revenue"],
            "mae_log_revenue": row["mae_log_revenue"],
            "mean_actual_gross": row["mean_actual_gross"],
            "mean_predicted_gross": row["mean_predicted_gross"],
        }
        for row in sorted(
            best.values(),
            key=lambda item: (
                str(item["model"]),
                str(item["population"]),
                str(item["interval_method"]),
                int(item["snapshot_day"]),
            ),
        )
    ]


def default_competitive_selection_splits(
    *,
    min_year: int = DEFAULT_TRAIN_START_YEAR,
    max_year: int = DEFAULT_TEST_END_YEAR,
) -> list[tuple[int, int, int]]:
    return [
        split
        for split in DEFAULT_COMPETITIVE_SELECTION_SPLITS
        if min_year <= split[0] <= split[1] < split[2] <= max_year
    ]


def displayed_prediction_usd_for_snapshot(
    row: dict[str, object],
    *,
    snapshot_day: int,
    base_prediction_usd: float,
    remainder_models: dict[tuple[int, ...], float],
) -> tuple[float, tuple[int, ...]]:
    locked_offsets = tuple(offset for offset in OPENING_WEEKEND_OFFSETS if offset < snapshot_day)
    if not locked_offsets:
        return max(1.0, base_prediction_usd), locked_offsets
    known = sum(float(row.get(f"opening_weekend_day{offset}_gross_usd", 0.0) or 0.0) for offset in locked_offsets)
    if locked_offsets == OPENING_WEEKEND_OFFSETS:
        return max(1.0, known), locked_offsets
    if known > 0.0 and locked_offsets in remainder_models:
        remaining = max(0.0, known * remainder_models[locked_offsets])
    else:
        remaining = max(0.0, base_prediction_usd - known)
    return max(1.0, known + remaining), locked_offsets


def train_remainder_ratio_models(rows: list[dict[str, object]]) -> dict[tuple[int, ...], float]:
    by_movie: dict[int, dict[str, object]] = {}
    for row in rows:
        by_movie.setdefault(int(row["movie_id"]), row)
    ratios: dict[tuple[int, ...], list[float]] = defaultdict(list)
    for row in by_movie.values():
        daily = {
            offset: float(row.get(f"opening_weekend_day{offset}_gross_usd", 0.0) or 0.0)
            for offset in OPENING_WEEKEND_OFFSETS
        }
        total = sum(daily.values())
        if total <= 0.0 or any(daily[offset] <= 0.0 for offset in OPENING_WEEKEND_OFFSETS):
            continue
        for locked_offsets in ((0,), (0, 1)):
            known = sum(daily[offset] for offset in locked_offsets)
            if known > 0.0:
                ratios[locked_offsets].append(max(0.0, total - known) / known)
    return {
        locked_offsets: sum(values) / len(values)
        for locked_offsets, values in ratios.items()
        if values
    }


def wiki_bop_timing_scoped_rows(
    panel_rows: list[dict[str, object]],
    *,
    snapshot_day: int,
    min_bop_forecast_midpoint: int = DEFAULT_MIN_BOP_FORECAST_MIDPOINT,
) -> list[dict[str, object]]:
    return [
        row
        for row in panel_rows
        if str(row.get("target_type", THREE_DAY_TARGET)) == THREE_DAY_TARGET
        and int(row["snapshot_day"]) == snapshot_day
        and float(row.get("bop_forecast_available", 0.0) or 0.0) > 0.0
        and float(row.get("bop_forecast_midpoint", 0.0) or 0.0) >= min_bop_forecast_midpoint
        and row.get("target_log_bop_residual", "") != ""
    ]


def wiki_bop_locked_offsets(snapshot_day: int) -> tuple[int, ...]:
    return tuple(offset for offset in OPENING_WEEKEND_OFFSETS if offset < snapshot_day)


def wiki_bop_actuals_policy_name(snapshot_day: int) -> str:
    locked_offsets = wiki_bop_locked_offsets(snapshot_day)
    if not locked_offsets:
        return "model_only"
    if locked_offsets == OPENING_WEEKEND_OFFSETS:
        return "locked_full_actual"
    return "locked_actuals_remainder_ratio"


def wiki_bop_display_prediction_usd(
    row: dict[str, object],
    *,
    snapshot_day: int,
    base_prediction_usd: float,
    remainder_models: dict[tuple[int, ...], float],
) -> tuple[float, tuple[int, ...]]:
    locked_offsets = wiki_bop_locked_offsets(snapshot_day)
    if not locked_offsets:
        return max(1.0, base_prediction_usd), locked_offsets
    known = sum(float(row.get(f"opening_weekend_day{offset}_gross_usd", 0.0) or 0.0) for offset in locked_offsets)
    if locked_offsets == OPENING_WEEKEND_OFFSETS:
        return max(1.0, known), locked_offsets
    if known > 0.0 and locked_offsets in remainder_models:
        remaining = max(0.0, known * remainder_models[locked_offsets])
    else:
        remaining = max(0.0, base_prediction_usd - known)
    return max(1.0, known + remaining), locked_offsets


def wiki_bop_timing_prediction_rows(
    *,
    snapshot_day: int,
    model_name: str,
    rows: list[dict[str, object]],
    predicted_residuals: list[float],
    remainder_models: dict[tuple[int, ...], float],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    actuals_policy = wiki_bop_actuals_policy_name(snapshot_day)
    for row, predicted_residual in zip(rows, predicted_residuals):
        midpoint = float(row["bop_forecast_midpoint"])
        base_log = math.log(max(1.0, midpoint)) + predicted_residual
        base_prediction = max(1.0, math.exp(base_log))
        displayed_prediction, locked_offsets = wiki_bop_display_prediction_usd(
            row,
            snapshot_day=snapshot_day,
            base_prediction_usd=base_prediction,
            remainder_models=remainder_models,
        )
        actual_gross = float(row["opening_weekend_revenue_usd"])
        out.append(
            {
                "snapshot_day": snapshot_day,
                "model": model_name,
                "population": "bop_covered",
                "actuals_policy": actuals_policy,
                "movie_id": row["movie_id"],
                "title": row["title"],
                "release_year": row["release_year"],
                "opening_date": row["opening_date"],
                "as_of_date": row["as_of_date"],
                "bop_forecast_midpoint": midpoint,
                "wiki_available": row.get("wiki_available", 0.0),
                "V": row.get("V", 0.0),
                "U": row.get("U", 0.0),
                "R": row.get("R", 0.0),
                "E": row.get("E", 0.0),
                "log1p_V": row.get("log1p_V", 0.0),
                "log1p_U": row.get("log1p_U", 0.0),
                "log1p_R": row.get("log1p_R", 0.0),
                "log1p_E": row.get("log1p_E", 0.0),
                "log1p_opening_theaters": row.get("log1p_opening_theaters", 0.0),
                "known_actual_offsets": ",".join(str(offset) for offset in locked_offsets),
                "known_gross_so_far": row.get("known_gross_so_far", 0.0),
                "base_predicted_opening_weekend_revenue_usd": base_prediction,
                "actual_log_bop_residual": row["target_log_bop_residual"],
                "predicted_log_bop_residual": predicted_residual,
                "actual_log_opening_weekend": row["target_log_opening_weekend"],
                "predicted_log_opening_weekend": math.log(displayed_prediction),
                "actual_opening_weekend_revenue_usd": actual_gross,
                "predicted_opening_weekend_revenue_usd": displayed_prediction,
                "absolute_percentage_error": abs(displayed_prediction - actual_gross) / actual_gross
                if actual_gross > 0.0
                else "",
            }
        )
    return out


def wiki_bop_timing_metric_row(
    *,
    snapshot_day: int,
    model_name: str,
    actuals_policy: str,
    train_start_year: int,
    train_end_year: int,
    test_start_year: int,
    test_end_year: int,
    train_n: int,
    prediction_rows: list[dict[str, object]],
) -> dict[str, object]:
    base = {
        "snapshot_day": snapshot_day,
        "model": model_name,
        "population": "bop_covered",
        "actuals_policy": actuals_policy,
        "train_start_year": train_start_year,
        "train_end_year": train_end_year,
        "test_start_year": test_start_year,
        "test_end_year": test_end_year,
        "train_n": train_n,
        "holdout_n": len(prediction_rows),
    }
    if len(prediction_rows) < 2:
        return {
            **base,
            "r2_log_revenue": "",
            "r2_gross": "",
            "mape_gross": "",
            "accuracy_pct": "",
            "rmse_log_revenue": "",
            "mae_log_revenue": "",
            "mean_actual_gross": "",
            "mean_predicted_gross": "",
            "status": "insufficient_sample",
        }
    actual_log = [float(row["actual_log_opening_weekend"]) for row in prediction_rows]
    pred_log = [float(row["predicted_log_opening_weekend"]) for row in prediction_rows]
    actual_gross = [float(row["actual_opening_weekend_revenue_usd"]) for row in prediction_rows]
    pred_gross = [float(row["predicted_opening_weekend_revenue_usd"]) for row in prediction_rows]
    log_errors = [actual - pred for actual, pred in zip(actual_log, pred_log)]
    apes = [abs(pred - actual) / actual for actual, pred in zip(actual_gross, pred_gross) if actual > 0.0]
    mape = mean(apes) if apes else None
    return {
        **base,
        "r2_log_revenue": format_number(r2_score(actual_log, pred_log)),
        "r2_gross": format_number(r2_score(actual_gross, pred_gross)),
        "mape_gross": format_number(mape),
        "accuracy_pct": format_number(1.0 - mape if mape is not None else None),
        "rmse_log_revenue": format_number(opening.rmse(log_errors)),
        "mae_log_revenue": format_number(opening.mae(log_errors)),
        "mean_actual_gross": format_number(mean(actual_gross)),
        "mean_predicted_gross": format_number(mean(pred_gross)),
        "status": "ok",
    }


def evaluate_wiki_bop_timing(
    panel_rows: list[dict[str, object]],
    *,
    snapshot_days: Iterable[int] = WIKI_BOP_TIMING_DAYS,
    train_start_year: int = WIKI_BOP_TIMING_TRAIN_START_YEAR,
    train_end_year: int = WIKI_BOP_TIMING_TRAIN_END_YEAR,
    test_start_year: int = WIKI_BOP_TIMING_TEST_START_YEAR,
    test_end_year: int = WIKI_BOP_TIMING_TEST_END_YEAR,
    min_bop_forecast_midpoint: int = DEFAULT_MIN_BOP_FORECAST_MIDPOINT,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    for snapshot_day in snapshot_days:
        scoped_rows = wiki_bop_timing_scoped_rows(
            panel_rows,
            snapshot_day=snapshot_day,
            min_bop_forecast_midpoint=min_bop_forecast_midpoint,
        )
        train_rows, holdout_rows = split_competitive_checkpoint_rows(
            scoped_rows,
            train_start_year=train_start_year,
            train_end_year=train_end_year,
            test_start_year=test_start_year,
            test_end_year=test_end_year,
        )
        remainder_models = train_remainder_ratio_models(train_rows)
        actuals_policy = wiki_bop_actuals_policy_name(snapshot_day)
        for model_name, terms in WIKI_BOP_TIMING_MODEL_TERMS.items():
            train_n = len(train_rows)
            if not holdout_rows or (model_name != "raw_bop_available" and len(train_rows) < len(terms) + 2):
                metric_rows.append(
                    wiki_bop_timing_metric_row(
                        snapshot_day=snapshot_day,
                        model_name=model_name,
                        actuals_policy=actuals_policy,
                        train_start_year=train_start_year,
                        train_end_year=train_end_year,
                        test_start_year=test_start_year,
                        test_end_year=test_end_year,
                        train_n=train_n,
                        prediction_rows=[],
                    )
                )
                continue
            if model_name == "raw_bop_available":
                predicted_residuals = [0.0 for _row in holdout_rows]
            else:
                fitted = opening.fit_model_for_target(train_rows, terms, "target_log_bop_residual")
                predicted_residuals = opening.predict_log(fitted, holdout_rows)
                for term, coef, center, scale in zip(
                    ["intercept"] + terms,
                    fitted.beta,
                    [0.0] + fitted.centers,
                    [1.0] + fitted.scales,
                ):
                    coefficient_rows.append(
                        {
                            "snapshot_day": snapshot_day,
                            "model": model_name,
                            "target": "target_log_bop_residual",
                            "term": term,
                            "standardized_coef": coef,
                            "center": center,
                            "scale": scale,
                            "train_n": train_n,
                        }
                    )
            rows = wiki_bop_timing_prediction_rows(
                snapshot_day=snapshot_day,
                model_name=model_name,
                rows=holdout_rows,
                predicted_residuals=predicted_residuals,
                remainder_models=remainder_models,
            )
            prediction_rows.extend(rows)
            metric_rows.append(
                wiki_bop_timing_metric_row(
                    snapshot_day=snapshot_day,
                    model_name=model_name,
                    actuals_policy=actuals_policy,
                    train_start_year=train_start_year,
                    train_end_year=train_end_year,
                    test_start_year=test_start_year,
                    test_end_year=test_end_year,
                    train_n=train_n,
                    prediction_rows=rows,
                )
            )
    return prediction_rows, metric_rows, coefficient_rows, wiki_bop_timing_headline_rows(metric_rows)


def wiki_bop_timing_headline_rows(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    primary_models = ("bop_residual_wiki_views", "bop_residual_wiki_full_activity")
    by_key = {
        (int(row["snapshot_day"]), str(row["model"])): row
        for row in metric_rows
        if row["status"] == "ok"
    }
    snapshot_days = sorted({int(row["snapshot_day"]) for row in metric_rows})
    for snapshot_day in snapshot_days:
        baseline = by_key.get((snapshot_day, "raw_bop_available"))
        for model_name in primary_models:
            wiki = by_key.get((snapshot_day, model_name))
            if baseline is None or wiki is None:
                rows.append(
                    {
                        "snapshot_day": snapshot_day,
                        "baseline_model": "raw_bop_available",
                        "wiki_model": model_name,
                        "actuals_policy": wiki_bop_actuals_policy_name(snapshot_day),
                        "holdout_n": "",
                        "baseline_mape_gross": "",
                        "wiki_mape_gross": "",
                        "mape_delta_wiki_minus_baseline": "",
                        "baseline_rmse_log_revenue": "",
                        "wiki_rmse_log_revenue": "",
                        "rmse_log_delta_wiki_minus_baseline": "",
                        "baseline_r2_gross": "",
                        "wiki_r2_gross": "",
                        "r2_gross_delta_wiki_minus_baseline": "",
                        "wiki_better_mape": "",
                        "status": "insufficient_sample",
                    }
                )
                continue
            baseline_mape = float(baseline["mape_gross"])
            wiki_mape = float(wiki["mape_gross"])
            baseline_rmse = float(baseline["rmse_log_revenue"])
            wiki_rmse = float(wiki["rmse_log_revenue"])
            baseline_r2 = float(baseline["r2_gross"])
            wiki_r2 = float(wiki["r2_gross"])
            rows.append(
                {
                    "snapshot_day": snapshot_day,
                    "baseline_model": "raw_bop_available",
                    "wiki_model": model_name,
                    "actuals_policy": wiki["actuals_policy"],
                    "holdout_n": wiki["holdout_n"],
                    "baseline_mape_gross": baseline["mape_gross"],
                    "wiki_mape_gross": wiki["mape_gross"],
                    "mape_delta_wiki_minus_baseline": format_number(wiki_mape - baseline_mape),
                    "baseline_rmse_log_revenue": baseline["rmse_log_revenue"],
                    "wiki_rmse_log_revenue": wiki["rmse_log_revenue"],
                    "rmse_log_delta_wiki_minus_baseline": format_number(wiki_rmse - baseline_rmse),
                    "baseline_r2_gross": baseline["r2_gross"],
                    "wiki_r2_gross": wiki["r2_gross"],
                    "r2_gross_delta_wiki_minus_baseline": format_number(wiki_r2 - baseline_r2),
                    "wiki_better_mape": wiki_mape < baseline_mape,
                    "status": "ok",
                }
            )
    return rows


def remaining_weekend_rows_with_baseline(
    rows: list[dict[str, object]],
    *,
    snapshot_day: int,
    remainder_models: dict[tuple[int, ...], float],
) -> list[dict[str, object]]:
    locked_offsets = wiki_bop_locked_offsets(snapshot_day)
    if not locked_offsets or locked_offsets == OPENING_WEEKEND_OFFSETS:
        return []
    ratio = remainder_models.get(locked_offsets)
    if ratio is None:
        return []
    out: list[dict[str, object]] = []
    for row in rows:
        known = sum(float(row.get(f"opening_weekend_day{offset}_gross_usd", 0.0) or 0.0) for offset in locked_offsets)
        remaining = float(row.get("remaining_gross", 0.0) or 0.0)
        if known <= 0.0 or remaining <= 0.0:
            continue
        baseline_remaining = max(1.0, known * ratio)
        next_row = dict(row)
        next_row["remaining_known_gross_usd"] = known
        next_row["remaining_actual_gross_usd"] = remaining
        next_row["remaining_baseline_gross_usd"] = baseline_remaining
        next_row["target_log_remaining_baseline_residual"] = math.log(max(1.0, remaining)) - math.log(baseline_remaining)
        out.append(next_row)
    return out


def wiki_remaining_prediction_rows(
    *,
    snapshot_day: int,
    model_name: str,
    rows: list[dict[str, object]],
    predicted_residuals: list[float],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row, predicted_residual in zip(rows, predicted_residuals):
        known = float(row["remaining_known_gross_usd"])
        actual_remaining = float(row["remaining_actual_gross_usd"])
        baseline_remaining = float(row["remaining_baseline_gross_usd"])
        predicted_remaining = max(1.0, baseline_remaining * math.exp(predicted_residual))
        predicted_total = max(1.0, known + predicted_remaining)
        actual_total = float(row["opening_weekend_revenue_usd"])
        out.append(
            {
                "snapshot_day": snapshot_day,
                "model": model_name,
                "population": "bop_covered",
                "actuals_policy": wiki_bop_actuals_policy_name(snapshot_day),
                "movie_id": row["movie_id"],
                "title": row["title"],
                "release_year": row["release_year"],
                "opening_date": row["opening_date"],
                "as_of_date": row["as_of_date"],
                "bop_forecast_midpoint": row["bop_forecast_midpoint"],
                "wiki_available": row.get("wiki_available", 0.0),
                "log1p_V": row.get("log1p_V", 0.0),
                "log1p_U": row.get("log1p_U", 0.0),
                "log1p_R": row.get("log1p_R", 0.0),
                "log1p_E": row.get("log1p_E", 0.0),
                "known_actual_offsets": ",".join(str(offset) for offset in wiki_bop_locked_offsets(snapshot_day)),
                "known_gross_so_far": known,
                "actual_remaining_gross_usd": actual_remaining,
                "baseline_remaining_gross_usd": baseline_remaining,
                "predicted_remaining_gross_usd": predicted_remaining,
                "actual_log_remaining_baseline_residual": row["target_log_remaining_baseline_residual"],
                "predicted_log_remaining_baseline_residual": predicted_residual,
                "actual_log_opening_weekend": row["target_log_opening_weekend"],
                "predicted_log_opening_weekend": math.log(predicted_total),
                "actual_opening_weekend_revenue_usd": actual_total,
                "predicted_opening_weekend_revenue_usd": predicted_total,
                "absolute_percentage_error": abs(predicted_total - actual_total) / actual_total if actual_total > 0.0 else "",
                "remaining_absolute_percentage_error": abs(predicted_remaining - actual_remaining) / actual_remaining
                if actual_remaining > 0.0
                else "",
            }
        )
    return out


def wiki_remaining_metric_row(
    *,
    snapshot_day: int,
    model_name: str,
    train_start_year: int,
    train_end_year: int,
    test_start_year: int,
    test_end_year: int,
    train_n: int,
    prediction_rows: list[dict[str, object]],
) -> dict[str, object]:
    base = {
        "snapshot_day": snapshot_day,
        "model": model_name,
        "population": "bop_covered",
        "actuals_policy": wiki_bop_actuals_policy_name(snapshot_day),
        "train_start_year": train_start_year,
        "train_end_year": train_end_year,
        "test_start_year": test_start_year,
        "test_end_year": test_end_year,
        "train_n": train_n,
        "holdout_n": len(prediction_rows),
    }
    if len(prediction_rows) < 2:
        return {
            **base,
            "r2_log_revenue": "",
            "r2_gross": "",
            "mape_gross": "",
            "mape_remaining_gross": "",
            "accuracy_pct": "",
            "rmse_log_revenue": "",
            "mae_log_revenue": "",
            "mean_actual_gross": "",
            "mean_predicted_gross": "",
            "mean_actual_remaining_gross": "",
            "mean_predicted_remaining_gross": "",
            "status": "insufficient_sample",
        }
    actual_log = [float(row["actual_log_opening_weekend"]) for row in prediction_rows]
    pred_log = [float(row["predicted_log_opening_weekend"]) for row in prediction_rows]
    actual_gross = [float(row["actual_opening_weekend_revenue_usd"]) for row in prediction_rows]
    pred_gross = [float(row["predicted_opening_weekend_revenue_usd"]) for row in prediction_rows]
    actual_remaining = [float(row["actual_remaining_gross_usd"]) for row in prediction_rows]
    pred_remaining = [float(row["predicted_remaining_gross_usd"]) for row in prediction_rows]
    log_errors = [actual - pred for actual, pred in zip(actual_log, pred_log)]
    apes = [abs(pred - actual) / actual for actual, pred in zip(actual_gross, pred_gross) if actual > 0.0]
    remaining_apes = [
        abs(pred - actual) / actual
        for actual, pred in zip(actual_remaining, pred_remaining)
        if actual > 0.0
    ]
    mape = mean(apes) if apes else None
    return {
        **base,
        "r2_log_revenue": format_number(r2_score(actual_log, pred_log)),
        "r2_gross": format_number(r2_score(actual_gross, pred_gross)),
        "mape_gross": format_number(mape),
        "mape_remaining_gross": format_number(mean(remaining_apes) if remaining_apes else None),
        "accuracy_pct": format_number(1.0 - mape if mape is not None else None),
        "rmse_log_revenue": format_number(opening.rmse(log_errors)),
        "mae_log_revenue": format_number(opening.mae(log_errors)),
        "mean_actual_gross": format_number(mean(actual_gross)),
        "mean_predicted_gross": format_number(mean(pred_gross)),
        "mean_actual_remaining_gross": format_number(mean(actual_remaining)),
        "mean_predicted_remaining_gross": format_number(mean(pred_remaining)),
        "status": "ok",
    }


def evaluate_wiki_remaining_timing(
    panel_rows: list[dict[str, object]],
    *,
    snapshot_days: Iterable[int] = WIKI_REMAINING_TIMING_DAYS,
    train_start_year: int = WIKI_BOP_TIMING_TRAIN_START_YEAR,
    train_end_year: int = WIKI_BOP_TIMING_TRAIN_END_YEAR,
    test_start_year: int = WIKI_BOP_TIMING_TEST_START_YEAR,
    test_end_year: int = WIKI_BOP_TIMING_TEST_END_YEAR,
    min_bop_forecast_midpoint: int = DEFAULT_MIN_BOP_FORECAST_MIDPOINT,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    for snapshot_day in snapshot_days:
        scoped_rows = wiki_bop_timing_scoped_rows(
            panel_rows,
            snapshot_day=snapshot_day,
            min_bop_forecast_midpoint=min_bop_forecast_midpoint,
        )
        train_rows, holdout_rows = split_competitive_checkpoint_rows(
            scoped_rows,
            train_start_year=train_start_year,
            train_end_year=train_end_year,
            test_start_year=test_start_year,
            test_end_year=test_end_year,
        )
        remainder_models = train_remainder_ratio_models(train_rows)
        train_aug = remaining_weekend_rows_with_baseline(
            train_rows,
            snapshot_day=snapshot_day,
            remainder_models=remainder_models,
        )
        holdout_aug = remaining_weekend_rows_with_baseline(
            holdout_rows,
            snapshot_day=snapshot_day,
            remainder_models=remainder_models,
        )
        for model_name, terms in WIKI_REMAINING_MODEL_TERMS.items():
            if not holdout_aug or (model_name != "remaining_baseline_ratio" and len(train_aug) < len(terms) + 2):
                metric_rows.append(
                    wiki_remaining_metric_row(
                        snapshot_day=snapshot_day,
                        model_name=model_name,
                        train_start_year=train_start_year,
                        train_end_year=train_end_year,
                        test_start_year=test_start_year,
                        test_end_year=test_end_year,
                        train_n=len(train_aug),
                        prediction_rows=[],
                    )
                )
                continue
            if model_name == "remaining_baseline_ratio":
                predicted_residuals = [0.0 for _row in holdout_aug]
            else:
                fitted = opening.fit_model_for_target(train_aug, terms, "target_log_remaining_baseline_residual")
                predicted_residuals = opening.predict_log(fitted, holdout_aug)
                for term, coef, center, scale in zip(
                    ["intercept"] + terms,
                    fitted.beta,
                    [0.0] + fitted.centers,
                    [1.0] + fitted.scales,
                ):
                    coefficient_rows.append(
                        {
                            "snapshot_day": snapshot_day,
                            "model": model_name,
                            "target": "target_log_remaining_baseline_residual",
                            "term": term,
                            "standardized_coef": coef,
                            "center": center,
                            "scale": scale,
                            "train_n": len(train_aug),
                        }
                    )
            rows = wiki_remaining_prediction_rows(
                snapshot_day=snapshot_day,
                model_name=model_name,
                rows=holdout_aug,
                predicted_residuals=predicted_residuals,
            )
            prediction_rows.extend(rows)
            metric_rows.append(
                wiki_remaining_metric_row(
                    snapshot_day=snapshot_day,
                    model_name=model_name,
                    train_start_year=train_start_year,
                    train_end_year=train_end_year,
                    test_start_year=test_start_year,
                    test_end_year=test_end_year,
                    train_n=len(train_aug),
                    prediction_rows=rows,
                )
            )
    return prediction_rows, metric_rows, coefficient_rows, wiki_remaining_headline_rows(metric_rows)


def wiki_remaining_headline_rows(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    by_key = {
        (int(row["snapshot_day"]), str(row["model"])): row
        for row in metric_rows
        if row["status"] == "ok"
    }
    for snapshot_day in sorted({int(row["snapshot_day"]) for row in metric_rows}):
        baseline = by_key.get((snapshot_day, "remaining_baseline_ratio"))
        for model_name in ("remaining_residual_wiki_views", "remaining_residual_wiki_full_activity"):
            wiki = by_key.get((snapshot_day, model_name))
            if baseline is None or wiki is None:
                rows.append(
                    {
                        "snapshot_day": snapshot_day,
                        "baseline_model": "remaining_baseline_ratio",
                        "wiki_model": model_name,
                        "actuals_policy": wiki_bop_actuals_policy_name(snapshot_day),
                        "holdout_n": "",
                        "baseline_mape_gross": "",
                        "wiki_mape_gross": "",
                        "mape_delta_wiki_minus_baseline": "",
                        "baseline_mape_remaining_gross": "",
                        "wiki_mape_remaining_gross": "",
                        "mape_remaining_delta_wiki_minus_baseline": "",
                        "baseline_rmse_log_revenue": "",
                        "wiki_rmse_log_revenue": "",
                        "rmse_log_delta_wiki_minus_baseline": "",
                        "baseline_r2_gross": "",
                        "wiki_r2_gross": "",
                        "r2_gross_delta_wiki_minus_baseline": "",
                        "wiki_better_mape": "",
                        "status": "insufficient_sample",
                    }
                )
                continue
            baseline_mape = float(baseline["mape_gross"])
            wiki_mape = float(wiki["mape_gross"])
            baseline_remaining_mape = float(baseline["mape_remaining_gross"])
            wiki_remaining_mape = float(wiki["mape_remaining_gross"])
            baseline_rmse = float(baseline["rmse_log_revenue"])
            wiki_rmse = float(wiki["rmse_log_revenue"])
            baseline_r2 = float(baseline["r2_gross"])
            wiki_r2 = float(wiki["r2_gross"])
            rows.append(
                {
                    "snapshot_day": snapshot_day,
                    "baseline_model": "remaining_baseline_ratio",
                    "wiki_model": model_name,
                    "actuals_policy": wiki["actuals_policy"],
                    "holdout_n": wiki["holdout_n"],
                    "baseline_mape_gross": baseline["mape_gross"],
                    "wiki_mape_gross": wiki["mape_gross"],
                    "mape_delta_wiki_minus_baseline": format_number(wiki_mape - baseline_mape),
                    "baseline_mape_remaining_gross": baseline["mape_remaining_gross"],
                    "wiki_mape_remaining_gross": wiki["mape_remaining_gross"],
                    "mape_remaining_delta_wiki_minus_baseline": format_number(wiki_remaining_mape - baseline_remaining_mape),
                    "baseline_rmse_log_revenue": baseline["rmse_log_revenue"],
                    "wiki_rmse_log_revenue": wiki["rmse_log_revenue"],
                    "rmse_log_delta_wiki_minus_baseline": format_number(wiki_rmse - baseline_rmse),
                    "baseline_r2_gross": baseline["r2_gross"],
                    "wiki_r2_gross": wiki["r2_gross"],
                    "r2_gross_delta_wiki_minus_baseline": format_number(wiki_r2 - baseline_r2),
                    "wiki_better_mape": wiki_mape < baseline_mape,
                    "status": "ok",
                }
            )
    return rows


def wiki_comp_combo_model_family(model_name: str) -> str:
    if model_name == "raw_bop_available":
        return "baseline"
    if model_name in WIKI_COMP_IMPROVED_MODEL_NAMES:
        return "improved_wiki_competition"
    if model_name in WIKI_COMP_COMBO_SENSITIVITY_MODELS:
        return "sensitivity_wiki_competition"
    if model_name in WIKI_COMP_COMBO_COMBINATION_MODELS:
        return "wiki_competition"
    if model_name.startswith("bop_residual_wiki"):
        return "wiki"
    if model_name.startswith("bop_residual_comp"):
        return "competition"
    return "other"


def wiki_comp_combo_target_policy(snapshot_day: int) -> str:
    return "total_weekend_bop_residual" if snapshot_day <= 0 else "remaining_weekend_ratio_residual"


def wiki_comp_combo_prediction_rows(
    *,
    snapshot_day: int,
    model_name: str,
    rows: list[dict[str, object]],
    predicted_residuals: list[float],
    target_policy: str,
    target_key: str,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    actuals_policy = wiki_bop_actuals_policy_name(snapshot_day)
    model_family = wiki_comp_combo_model_family(model_name)
    for row, predicted_residual in zip(rows, predicted_residuals):
        midpoint = float(row["bop_forecast_midpoint"])
        actual_gross = float(row["opening_weekend_revenue_usd"])
        actual_remaining = row.get("remaining_actual_gross_usd", "")
        baseline_remaining = row.get("remaining_baseline_gross_usd", "")
        predicted_remaining: float | str = ""
        remaining_ape: float | str = ""
        if target_policy == "remaining_weekend_ratio_residual":
            known = float(row["remaining_known_gross_usd"])
            actual_remaining_float = float(row["remaining_actual_gross_usd"])
            baseline_remaining_float = float(row["remaining_baseline_gross_usd"])
            predicted_remaining_float = max(1.0, baseline_remaining_float * math.exp(predicted_residual))
            predicted_gross = max(1.0, known + predicted_remaining_float)
            baseline_display = max(1.0, known + baseline_remaining_float)
            predicted_remaining = predicted_remaining_float
            remaining_ape = (
                abs(predicted_remaining_float - actual_remaining_float) / actual_remaining_float
                if actual_remaining_float > 0.0
                else ""
            )
        else:
            known = float(row.get("known_gross_so_far", 0.0) or 0.0)
            predicted_gross = max(1.0, midpoint * math.exp(predicted_residual))
            baseline_display = max(1.0, midpoint)

        out.append(
            {
                "snapshot_day": snapshot_day,
                "model": model_name,
                "model_family": model_family,
                "target_policy": target_policy,
                "population": "bop_covered",
                "actuals_policy": actuals_policy,
                "movie_id": row["movie_id"],
                "title": row["title"],
                "release_year": row["release_year"],
                "opening_date": row["opening_date"],
                "as_of_date": row["as_of_date"],
                "bop_forecast_midpoint": midpoint,
                "wiki_available": row.get("wiki_available", 0.0),
                "log1p_V": row.get("log1p_V", 0.0),
                "log1p_U": row.get("log1p_U", 0.0),
                "log1p_R": row.get("log1p_R", 0.0),
                "log1p_E": row.get("log1p_E", 0.0),
                "share_attraction_competitor_pressure_lag7": row.get(
                    "share_attraction_competitor_pressure_lag7", 0.0
                ),
                "logit_demand_focal_vs_competition_lag7": row.get(
                    "logit_demand_focal_vs_competition_lag7", 0.0
                ),
                "known_actual_offsets": ",".join(str(offset) for offset in wiki_bop_locked_offsets(snapshot_day)),
                "known_gross_so_far": known,
                "actual_remaining_gross_usd": actual_remaining,
                "baseline_remaining_gross_usd": baseline_remaining,
                "predicted_remaining_gross_usd": predicted_remaining,
                "baseline_display_prediction_usd": baseline_display,
                "actual_log_target_residual": row[target_key],
                "predicted_log_target_residual": predicted_residual,
                "actual_log_opening_weekend": row["target_log_opening_weekend"],
                "predicted_log_opening_weekend": math.log(predicted_gross),
                "actual_opening_weekend_revenue_usd": actual_gross,
                "predicted_opening_weekend_revenue_usd": predicted_gross,
                "absolute_percentage_error": abs(predicted_gross - actual_gross) / actual_gross
                if actual_gross > 0.0
                else "",
                "remaining_absolute_percentage_error": remaining_ape,
            }
        )
    return out


def wiki_comp_combo_metric_row(
    *,
    snapshot_day: int,
    model_name: str,
    target_policy: str,
    train_start_year: int,
    train_end_year: int,
    test_start_year: int,
    test_end_year: int,
    train_n: int,
    prediction_rows: list[dict[str, object]],
) -> dict[str, object]:
    base = {
        "snapshot_day": snapshot_day,
        "model": model_name,
        "model_family": wiki_comp_combo_model_family(model_name),
        "target_policy": target_policy,
        "population": "bop_covered",
        "actuals_policy": wiki_bop_actuals_policy_name(snapshot_day),
        "train_start_year": train_start_year,
        "train_end_year": train_end_year,
        "test_start_year": test_start_year,
        "test_end_year": test_end_year,
        "train_n": train_n,
        "holdout_n": len(prediction_rows),
    }
    empty_metric_values = {
        "r2_log_revenue": "",
        "r2_gross": "",
        "mape_gross": "",
        "mape_remaining_gross": "",
        "accuracy_pct": "",
        "rmse_log_revenue": "",
        "mae_log_revenue": "",
        "mean_actual_gross": "",
        "mean_predicted_gross": "",
    }
    empty_delta_values = {
        "baseline_model": "",
        "baseline_mape_gross": "",
        "mape_delta_vs_baseline": "",
        "baseline_rmse_log_revenue": "",
        "rmse_log_delta_vs_baseline": "",
        "baseline_r2_gross": "",
        "r2_gross_delta_vs_baseline": "",
        "best_single_model": "",
        "best_single_mape_gross": "",
        "mape_delta_vs_best_single": "",
    }
    if len(prediction_rows) < 2:
        return {**base, **empty_metric_values, **empty_delta_values, "status": "insufficient_sample"}
    actual_log = [float(row["actual_log_opening_weekend"]) for row in prediction_rows]
    pred_log = [float(row["predicted_log_opening_weekend"]) for row in prediction_rows]
    actual_gross = [float(row["actual_opening_weekend_revenue_usd"]) for row in prediction_rows]
    pred_gross = [float(row["predicted_opening_weekend_revenue_usd"]) for row in prediction_rows]
    log_errors = [actual - pred for actual, pred in zip(actual_log, pred_log)]
    apes = [abs(pred - actual) / actual for actual, pred in zip(actual_gross, pred_gross) if actual > 0.0]
    remaining_apes = [
        float(row["remaining_absolute_percentage_error"])
        for row in prediction_rows
        if row.get("remaining_absolute_percentage_error", "") != ""
    ]
    mape = mean(apes) if apes else None
    return {
        **base,
        "r2_log_revenue": format_number(r2_score(actual_log, pred_log)),
        "r2_gross": format_number(r2_score(actual_gross, pred_gross)),
        "mape_gross": format_number(mape),
        "mape_remaining_gross": format_number(mean(remaining_apes) if remaining_apes else None),
        "accuracy_pct": format_number(1.0 - mape if mape is not None else None),
        "rmse_log_revenue": format_number(opening.rmse(log_errors)),
        "mae_log_revenue": format_number(opening.mae(log_errors)),
        "mean_actual_gross": format_number(mean(actual_gross)),
        "mean_predicted_gross": format_number(mean(pred_gross)),
        **empty_delta_values,
        "status": "ok",
    }


def wiki_comp_combo_metrics_with_deltas(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_day: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in metric_rows:
        by_day[int(row["snapshot_day"])].append(row)
    out: list[dict[str, object]] = []
    for snapshot_day, rows in by_day.items():
        ok_rows = [row for row in rows if row["status"] == "ok"]
        baseline = next((row for row in ok_rows if row["model"] == "raw_bop_available"), None)
        single_rows = [
            row for row in ok_rows if str(row["model"]) in WIKI_COMP_COMBO_SINGLE_SIGNAL_MODELS
        ]
        best_single = min(single_rows, key=lambda row: float(row["mape_gross"])) if single_rows else None
        for row in rows:
            next_row = dict(row)
            if row["status"] == "ok" and baseline is not None:
                next_row["baseline_model"] = baseline["model"]
                next_row["baseline_mape_gross"] = baseline["mape_gross"]
                next_row["mape_delta_vs_baseline"] = format_number(
                    float(row["mape_gross"]) - float(baseline["mape_gross"])
                )
                next_row["baseline_rmse_log_revenue"] = baseline["rmse_log_revenue"]
                next_row["rmse_log_delta_vs_baseline"] = format_number(
                    float(row["rmse_log_revenue"]) - float(baseline["rmse_log_revenue"])
                )
                next_row["baseline_r2_gross"] = baseline["r2_gross"]
                next_row["r2_gross_delta_vs_baseline"] = format_number(
                    float(row["r2_gross"]) - float(baseline["r2_gross"])
                )
            if row["status"] == "ok" and best_single is not None:
                next_row["best_single_model"] = best_single["model"]
                next_row["best_single_mape_gross"] = best_single["mape_gross"]
                next_row["mape_delta_vs_best_single"] = format_number(
                    float(row["mape_gross"]) - float(best_single["mape_gross"])
                )
            out.append(next_row)
    return out


def evaluate_wiki_comp_combo_timing(
    panel_rows: list[dict[str, object]],
    *,
    snapshot_days: Iterable[int] = WIKI_COMP_COMBO_DAYS,
    train_start_year: int = WIKI_BOP_TIMING_TRAIN_START_YEAR,
    train_end_year: int = WIKI_BOP_TIMING_TRAIN_END_YEAR,
    test_start_year: int = WIKI_BOP_TIMING_TEST_START_YEAR,
    test_end_year: int = WIKI_BOP_TIMING_TEST_END_YEAR,
    min_bop_forecast_midpoint: int = DEFAULT_MIN_BOP_FORECAST_MIDPOINT,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    for snapshot_day in snapshot_days:
        scoped_rows = wiki_bop_timing_scoped_rows(
            panel_rows,
            snapshot_day=snapshot_day,
            min_bop_forecast_midpoint=min_bop_forecast_midpoint,
        )
        train_rows, holdout_rows = split_competitive_checkpoint_rows(
            scoped_rows,
            train_start_year=train_start_year,
            train_end_year=train_end_year,
            test_start_year=test_start_year,
            test_end_year=test_end_year,
        )
        target_policy = wiki_comp_combo_target_policy(snapshot_day)
        if snapshot_day <= 0:
            train_aug = train_rows
            holdout_aug = holdout_rows
            target_key = "target_log_bop_residual"
        else:
            remainder_models = train_remainder_ratio_models(train_rows)
            train_aug = remaining_weekend_rows_with_baseline(
                train_rows,
                snapshot_day=snapshot_day,
                remainder_models=remainder_models,
            )
            holdout_aug = remaining_weekend_rows_with_baseline(
                holdout_rows,
                snapshot_day=snapshot_day,
                remainder_models=remainder_models,
            )
            target_key = "target_log_remaining_baseline_residual"

        for model_name, terms in WIKI_COMP_COMBO_MODEL_TERMS.items():
            train_n = len(train_aug)
            if not holdout_aug or (model_name != "raw_bop_available" and len(train_aug) < len(terms) + 2):
                metric_rows.append(
                    wiki_comp_combo_metric_row(
                        snapshot_day=snapshot_day,
                        model_name=model_name,
                        target_policy=target_policy,
                        train_start_year=train_start_year,
                        train_end_year=train_end_year,
                        test_start_year=test_start_year,
                        test_end_year=test_end_year,
                        train_n=train_n,
                        prediction_rows=[],
                    )
                )
                continue
            if model_name == "raw_bop_available":
                predicted_residuals = [0.0 for _row in holdout_aug]
            else:
                fitted = opening.fit_model_for_target(train_aug, terms, target_key)
                predicted_residuals = opening.predict_log(fitted, holdout_aug)
                for term, coef, center, scale in zip(
                    ["intercept"] + terms,
                    fitted.beta,
                    [0.0] + fitted.centers,
                    [1.0] + fitted.scales,
                ):
                    coefficient_rows.append(
                        {
                            "snapshot_day": snapshot_day,
                            "model": model_name,
                            "model_family": wiki_comp_combo_model_family(model_name),
                            "target_policy": target_policy,
                            "target": target_key,
                            "term": term,
                            "standardized_coef": coef,
                            "center": center,
                            "scale": scale,
                            "train_n": train_n,
                        }
                    )
            rows = wiki_comp_combo_prediction_rows(
                snapshot_day=snapshot_day,
                model_name=model_name,
                rows=holdout_aug,
                predicted_residuals=predicted_residuals,
                target_policy=target_policy,
                target_key=target_key,
            )
            prediction_rows.extend(rows)
            metric_rows.append(
                wiki_comp_combo_metric_row(
                    snapshot_day=snapshot_day,
                    model_name=model_name,
                    target_policy=target_policy,
                    train_start_year=train_start_year,
                    train_end_year=train_end_year,
                    test_start_year=test_start_year,
                    test_end_year=test_end_year,
                    train_n=train_n,
                    prediction_rows=rows,
                )
            )
    metric_rows = wiki_comp_combo_metrics_with_deltas(metric_rows)
    return prediction_rows, metric_rows, coefficient_rows, wiki_comp_combo_headline_rows(metric_rows)


def wiki_comp_combo_headline_rows(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    by_day: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in metric_rows:
        by_day[int(row["snapshot_day"])].append(row)
    for snapshot_day in sorted(by_day):
        ok_rows = [row for row in by_day[snapshot_day] if row["status"] == "ok"]
        if not ok_rows:
            rows.append(
                {
                    "snapshot_day": snapshot_day,
                    "target_policy": wiki_comp_combo_target_policy(snapshot_day),
                    "actuals_policy": wiki_bop_actuals_policy_name(snapshot_day),
                    "holdout_n": "",
                    "baseline_model": "raw_bop_available",
                    "baseline_mape_gross": "",
                    "views_mape_gross": "",
                    "full_mape_gross": "",
                    "views_beats_raw_bop": "",
                    "full_beats_views": "",
                    "best_comp_model": "",
                    "best_comp_mape_gross": "",
                    "competition_beats_raw_bop": "",
                    "best_single_model": "",
                    "best_single_mape_gross": "",
                    "best_combo_model": "",
                    "best_combo_mape_gross": "",
                    "combo_beats_best_single": "",
                    "combo_mape_delta_vs_best_single": "",
                    "best_model": "",
                    "best_model_family": "",
                    "best_mape_gross": "",
                    "best_mape_delta_vs_baseline": "",
                    "status": "insufficient_sample",
                }
            )
            continue
        by_model = {str(row["model"]): row for row in ok_rows}
        baseline = by_model.get("raw_bop_available")
        views = by_model.get("bop_residual_wiki_views")
        full = by_model.get("bop_residual_wiki_full_activity")
        comp_rows = [
            row
            for row in ok_rows
            if row["model"] in {"bop_residual_comp_share_attraction", "bop_residual_comp_logit_demand"}
        ]
        single_rows = [
            row for row in ok_rows if str(row["model"]) in WIKI_COMP_COMBO_SINGLE_SIGNAL_MODELS
        ]
        combo_rows = [
            row for row in ok_rows if str(row["model"]) in WIKI_COMP_COMBO_COMBINATION_MODELS
        ]
        best = min(ok_rows, key=lambda row: float(row["mape_gross"]))
        best_comp = min(comp_rows, key=lambda row: float(row["mape_gross"])) if comp_rows else None
        best_single = min(single_rows, key=lambda row: float(row["mape_gross"])) if single_rows else None
        best_combo = min(combo_rows, key=lambda row: float(row["mape_gross"])) if combo_rows else None
        baseline_mape = float(baseline["mape_gross"]) if baseline is not None else None
        best_combo_mape = float(best_combo["mape_gross"]) if best_combo is not None else None
        best_single_mape = float(best_single["mape_gross"]) if best_single is not None else None
        rows.append(
            {
                "snapshot_day": snapshot_day,
                "target_policy": best["target_policy"],
                "actuals_policy": best["actuals_policy"],
                "holdout_n": best["holdout_n"],
                "baseline_model": "raw_bop_available",
                "baseline_mape_gross": baseline["mape_gross"] if baseline is not None else "",
                "views_mape_gross": views["mape_gross"] if views is not None else "",
                "full_mape_gross": full["mape_gross"] if full is not None else "",
                "views_beats_raw_bop": (
                    float(views["mape_gross"]) < baseline_mape
                    if views is not None and baseline_mape is not None
                    else ""
                ),
                "full_beats_views": (
                    float(full["mape_gross"]) < float(views["mape_gross"])
                    if full is not None and views is not None
                    else ""
                ),
                "best_comp_model": best_comp["model"] if best_comp is not None else "",
                "best_comp_mape_gross": best_comp["mape_gross"] if best_comp is not None else "",
                "competition_beats_raw_bop": (
                    float(best_comp["mape_gross"]) < baseline_mape
                    if best_comp is not None and baseline_mape is not None
                    else ""
                ),
                "best_single_model": best_single["model"] if best_single is not None else "",
                "best_single_mape_gross": best_single["mape_gross"] if best_single is not None else "",
                "best_combo_model": best_combo["model"] if best_combo is not None else "",
                "best_combo_mape_gross": best_combo["mape_gross"] if best_combo is not None else "",
                "combo_beats_best_single": (
                    best_combo_mape < best_single_mape
                    if best_combo_mape is not None and best_single_mape is not None
                    else ""
                ),
                "combo_mape_delta_vs_best_single": (
                    format_number(best_combo_mape - best_single_mape)
                    if best_combo_mape is not None and best_single_mape is not None
                    else ""
                ),
                "best_model": best["model"],
                "best_model_family": best["model_family"],
                "best_mape_gross": best["mape_gross"],
                "best_mape_delta_vs_baseline": (
                    format_number(float(best["mape_gross"]) - baseline_mape)
                    if baseline_mape is not None
                    else ""
                ),
                "status": "ok",
            }
        )
    return rows


def fit_ridge_model_for_target(
    rows: list[dict[str, object]],
    terms: list[str],
    target_key: str,
    *,
    alpha: float = WIKI_COMP_RIDGE_LAMBDA,
) -> opening.FittedModel:
    if len(rows) < len(terms) + 2:
        raise ValueError("not enough training rows")
    x = opening.design_matrix(rows, terms)
    y = [float(row[target_key]) for row in rows]
    centers, scales = opening.standardize_fit(x)
    z = opening.standardize_apply(x, centers, scales)
    width = len(z[0])
    gram = [[sum(row[i] * row[j] for row in z) for j in range(width)] for i in range(width)]
    for idx in range(1, width):
        gram[idx][idx] += alpha
    target = [sum(row[i] * value for row, value in zip(z, y)) for i in range(width)]
    try:
        beta = opening.solve_linear_system(gram, target)
    except ValueError:
        for idx in range(width):
            gram[idx][idx] += 1e-9
        beta = opening.solve_linear_system(gram, target)
    return opening.FittedModel(terms=terms, centers=centers, scales=scales, beta=beta)


def fit_predict_residuals(
    *,
    train_rows: list[dict[str, object]],
    holdout_rows: list[dict[str, object]],
    terms: list[str],
    target_key: str,
    ridge_alpha: float | None = None,
) -> tuple[list[float], opening.FittedModel]:
    fitted = (
        opening.fit_model_for_target(train_rows, terms, target_key)
        if ridge_alpha is None
        else fit_ridge_model_for_target(train_rows, terms, target_key, alpha=ridge_alpha)
    )
    return opening.predict_log(fitted, holdout_rows), fitted


def loo_residual_predictions(
    *,
    rows: list[dict[str, object]],
    terms: list[str],
    target_key: str,
) -> list[float]:
    if len(rows) < len(terms) + 3:
        fitted = opening.fit_model_for_target(rows, terms, target_key)
        return opening.predict_log(fitted, rows)
    predictions: list[float] = []
    for idx, row in enumerate(rows):
        train_subset = rows[:idx] + rows[idx + 1 :]
        fitted = opening.fit_model_for_target(train_subset, terms, target_key)
        predictions.extend(opening.predict_log(fitted, [row]))
    return predictions


def stacked_residual_predictions(
    *,
    train_rows: list[dict[str, object]],
    holdout_rows: list[dict[str, object]],
    base_model_names: tuple[str, str],
    target_key: str,
) -> tuple[list[float], opening.FittedModel, list[str]]:
    train_stack_rows = [dict(row) for row in train_rows]
    holdout_stack_rows = [dict(row) for row in holdout_rows]
    stack_terms: list[str] = []
    for base_name in base_model_names:
        terms = WIKI_COMP_IMPROVED_BASE_MODELS[base_name]
        stack_term = f"predicted_{base_name}"
        stack_terms.append(stack_term)
        train_preds = loo_residual_predictions(rows=train_rows, terms=terms, target_key=target_key)
        holdout_preds, _base_fitted = fit_predict_residuals(
            train_rows=train_rows,
            holdout_rows=holdout_rows,
            terms=terms,
            target_key=target_key,
        )
        for row, pred in zip(train_stack_rows, train_preds):
            row[stack_term] = pred
        for row, pred in zip(holdout_stack_rows, holdout_preds):
            row[stack_term] = pred
    meta = opening.fit_model_for_target(train_stack_rows, stack_terms, target_key)
    return opening.predict_log(meta, holdout_stack_rows), meta, list(base_model_names)


def add_residualized_competition_term(
    train_rows: list[dict[str, object]],
    holdout_rows: list[dict[str, object]],
    *,
    controls: list[str],
    competition_term: str,
    residual_term: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], opening.FittedModel | None]:
    train_aug = [dict(row) for row in train_rows]
    holdout_aug = [dict(row) for row in holdout_rows]
    if len(train_aug) >= len(controls) + 2:
        fitted = opening.fit_model_for_target(train_aug, controls, competition_term)
        train_expected = opening.predict_log(fitted, train_aug)
        holdout_expected = opening.predict_log(fitted, holdout_aug)
    else:
        fitted = None
        fallback = mean(float(row.get(competition_term, 0.0) or 0.0) for row in train_aug) if train_aug else 0.0
        train_expected = [fallback for _row in train_aug]
        holdout_expected = [fallback for _row in holdout_aug]
    for row, expected in zip(train_aug, train_expected):
        row[residual_term] = float(row.get(competition_term, 0.0) or 0.0) - expected
    for row, expected in zip(holdout_aug, holdout_expected):
        row[residual_term] = float(row.get(competition_term, 0.0) or 0.0) - expected
    return train_aug, holdout_aug, fitted


def gated_timing_model_name(snapshot_day: int) -> str:
    if snapshot_day <= -1:
        return "bop_residual_wiki_full_activity"
    if snapshot_day == 0:
        return "bop_residual_comp_share_attraction"
    return "bop_residual_wiki_views"


def evaluate_wiki_comp_improved_timing(
    panel_rows: list[dict[str, object]],
    *,
    snapshot_days: Iterable[int] = WIKI_COMP_COMBO_DAYS,
    train_start_year: int = WIKI_BOP_TIMING_TRAIN_START_YEAR,
    train_end_year: int = WIKI_BOP_TIMING_TRAIN_END_YEAR,
    test_start_year: int = WIKI_BOP_TIMING_TEST_START_YEAR,
    test_end_year: int = WIKI_BOP_TIMING_TEST_END_YEAR,
    min_bop_forecast_midpoint: int = DEFAULT_MIN_BOP_FORECAST_MIDPOINT,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    for snapshot_day in snapshot_days:
        scoped_rows = wiki_bop_timing_scoped_rows(
            panel_rows,
            snapshot_day=snapshot_day,
            min_bop_forecast_midpoint=min_bop_forecast_midpoint,
        )
        train_rows, holdout_rows = split_competitive_checkpoint_rows(
            scoped_rows,
            train_start_year=train_start_year,
            train_end_year=train_end_year,
            test_start_year=test_start_year,
            test_end_year=test_end_year,
        )
        target_policy = wiki_comp_combo_target_policy(snapshot_day)
        if snapshot_day <= 0:
            train_aug = train_rows
            holdout_aug = holdout_rows
            target_key = "target_log_bop_residual"
        else:
            remainder_models = train_remainder_ratio_models(train_rows)
            train_aug = remaining_weekend_rows_with_baseline(
                train_rows,
                snapshot_day=snapshot_day,
                remainder_models=remainder_models,
            )
            holdout_aug = remaining_weekend_rows_with_baseline(
                holdout_rows,
                snapshot_day=snapshot_day,
                remainder_models=remainder_models,
            )
            target_key = "target_log_remaining_baseline_residual"

        for model_name in WIKI_COMP_IMPROVED_MODEL_NAMES:
            train_n = len(train_aug)
            predicted_residuals: list[float] | None = None
            fitted_model: opening.FittedModel | None = None
            auxiliary_terms: list[str] = []
            if holdout_aug:
                if model_name in WIKI_COMP_STACKED_MODELS:
                    base_names = WIKI_COMP_STACKED_MODELS[model_name]
                    required_width = max(
                        len(WIKI_COMP_IMPROVED_BASE_MODELS[base_name])
                        for base_name in base_names
                    )
                    if len(train_aug) >= required_width + 3:
                        predicted_residuals, fitted_model, auxiliary_terms = stacked_residual_predictions(
                            train_rows=train_aug,
                            holdout_rows=holdout_aug,
                            base_model_names=base_names,
                            target_key=target_key,
                        )
                elif model_name == "gated_timing_rule":
                    gated_model = gated_timing_model_name(snapshot_day)
                    terms = WIKI_COMP_IMPROVED_BASE_MODELS[gated_model]
                    if len(train_aug) >= len(terms) + 2:
                        predicted_residuals, fitted_model = fit_predict_residuals(
                            train_rows=train_aug,
                            holdout_rows=holdout_aug,
                            terms=terms,
                            target_key=target_key,
                        )
                        auxiliary_terms = [gated_model]
                elif model_name in WIKI_COMP_RESIDUALIZED_MODELS:
                    controls, competition_term, residual_term = WIKI_COMP_RESIDUALIZED_MODELS[model_name]
                    terms = list(controls) + [residual_term]
                    if len(train_aug) >= len(terms) + 2:
                        resid_train, resid_holdout, residualizer = add_residualized_competition_term(
                            train_aug,
                            holdout_aug,
                            controls=list(controls),
                            competition_term=competition_term,
                            residual_term=residual_term,
                        )
                        predicted_residuals, fitted_model = fit_predict_residuals(
                            train_rows=resid_train,
                            holdout_rows=resid_holdout,
                            terms=terms,
                            target_key=target_key,
                        )
                        if residualizer is not None:
                            auxiliary_terms = [f"residualized_{competition_term}"]
                elif model_name in WIKI_COMP_RIDGE_MODELS:
                    terms = WIKI_COMP_RIDGE_MODELS[model_name]
                    if len(train_aug) >= len(terms) + 2:
                        predicted_residuals, fitted_model = fit_predict_residuals(
                            train_rows=train_aug,
                            holdout_rows=holdout_aug,
                            terms=terms,
                            target_key=target_key,
                            ridge_alpha=WIKI_COMP_RIDGE_LAMBDA,
                        )

            if predicted_residuals is None:
                metric_rows.append(
                    wiki_comp_combo_metric_row(
                        snapshot_day=snapshot_day,
                        model_name=model_name,
                        target_policy=target_policy,
                        train_start_year=train_start_year,
                        train_end_year=train_end_year,
                        test_start_year=test_start_year,
                        test_end_year=test_end_year,
                        train_n=train_n,
                        prediction_rows=[],
                    )
                )
                continue

            if fitted_model is not None:
                for term, coef, center, scale in zip(
                    ["intercept"] + fitted_model.terms,
                    fitted_model.beta,
                    [0.0] + fitted_model.centers,
                    [1.0] + fitted_model.scales,
                ):
                    coefficient_rows.append(
                        {
                            "snapshot_day": snapshot_day,
                            "model": model_name,
                            "model_family": "improved_wiki_competition",
                            "target_policy": target_policy,
                            "target": target_key,
                            "term": term,
                            "standardized_coef": coef,
                            "center": center,
                            "scale": scale,
                            "train_n": train_n,
                        }
                    )
            for term in auxiliary_terms:
                coefficient_rows.append(
                    {
                        "snapshot_day": snapshot_day,
                        "model": model_name,
                        "model_family": "improved_wiki_competition",
                        "target_policy": target_policy,
                        "target": target_key,
                        "term": f"auxiliary:{term}",
                        "standardized_coef": "",
                        "center": "",
                        "scale": "",
                        "train_n": train_n,
                    }
                )

            rows = wiki_comp_combo_prediction_rows(
                snapshot_day=snapshot_day,
                model_name=model_name,
                rows=holdout_aug,
                predicted_residuals=predicted_residuals,
                target_policy=target_policy,
                target_key=target_key,
            )
            prediction_rows.extend(rows)
            metric_rows.append(
                wiki_comp_combo_metric_row(
                    snapshot_day=snapshot_day,
                    model_name=model_name,
                    target_policy=target_policy,
                    train_start_year=train_start_year,
                    train_end_year=train_end_year,
                    test_start_year=test_start_year,
                    test_end_year=test_end_year,
                    train_n=train_n,
                    prediction_rows=rows,
                )
            )
    metric_rows = wiki_comp_combo_metrics_with_deltas(metric_rows)
    return prediction_rows, metric_rows, coefficient_rows, wiki_comp_improved_headline_rows(metric_rows)


def wiki_comp_improved_headline_rows(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    by_day: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in metric_rows:
        by_day[int(row["snapshot_day"])].append(row)
    for snapshot_day in sorted(by_day):
        ok_rows = [row for row in by_day[snapshot_day] if row["status"] == "ok"]
        if not ok_rows:
            rows.append(
                {
                    "snapshot_day": snapshot_day,
                    "target_policy": wiki_comp_combo_target_policy(snapshot_day),
                    "actuals_policy": wiki_bop_actuals_policy_name(snapshot_day),
                    "holdout_n": "",
                    "best_improved_model": "",
                    "best_improved_mape_gross": "",
                    "status": "insufficient_sample",
                }
            )
            continue
        best = min(ok_rows, key=lambda row: float(row["mape_gross"]))
        rows.append(
            {
                "snapshot_day": snapshot_day,
                "target_policy": best["target_policy"],
                "actuals_policy": best["actuals_policy"],
                "holdout_n": best["holdout_n"],
                "best_improved_model": best["model"],
                "best_improved_mape_gross": best["mape_gross"],
                "status": "ok",
            }
        )
    return rows


def config_search_split_rows(
    panel_rows: list[dict[str, object]],
    *,
    snapshot_day: int,
    train_start_year: int,
    train_end_year: int,
    test_start_year: int,
    test_end_year: int,
    train_bop_cutoff: int,
    test_bop_cutoff: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    base_rows = [
        row
        for row in panel_rows
        if str(row.get("target_type", THREE_DAY_TARGET)) == THREE_DAY_TARGET
        and int(row["snapshot_day"]) == snapshot_day
        and float(row.get("bop_forecast_available", 0.0) or 0.0) > 0.0
        and row.get("target_log_bop_residual", "") != ""
    ]
    train_rows = [
        row
        for row in base_rows
        if train_start_year <= int(row["release_year"]) <= train_end_year
        and float(row.get("bop_forecast_midpoint", 0.0) or 0.0) >= train_bop_cutoff
    ]
    holdout_rows = [
        row
        for row in base_rows
        if test_start_year <= int(row["release_year"]) <= test_end_year
        and float(row.get("bop_forecast_midpoint", 0.0) or 0.0) >= test_bop_cutoff
    ]
    return train_rows, holdout_rows


def config_search_prefix(
    *,
    phase: str,
    train_policy: str,
    validation_fold: str,
    train_start_year: int,
    train_end_year: int,
    test_start_year: int,
    test_end_year: int,
    train_bop_cutoff: int,
    test_bop_cutoff: int,
    config_id: str,
) -> dict[str, object]:
    return {
        "search_phase": phase,
        "train_policy": train_policy,
        "validation_fold": validation_fold,
        "config_id": config_id,
        "config_train_start_year": train_start_year,
        "config_train_end_year": train_end_year,
        "config_test_start_year": test_start_year,
        "config_test_end_year": test_end_year,
        "train_bop_cutoff": train_bop_cutoff,
        "test_bop_cutoff": test_bop_cutoff,
    }


def config_search_config_id(train_policy: str, train_bop_cutoff: int, test_bop_cutoff: int) -> str:
    return f"{train_policy}_train_bop_{train_bop_cutoff}_test_bop_{test_bop_cutoff}"


def enrich_config_rows(rows: list[dict[str, object]], prefix: dict[str, object]) -> list[dict[str, object]]:
    return [{**prefix, **row} for row in rows]


def evaluate_config_search_split(
    panel_rows: list[dict[str, object]],
    *,
    phase: str,
    train_policy: str,
    validation_fold: str,
    snapshot_day: int,
    train_start_year: int,
    train_end_year: int,
    test_start_year: int,
    test_end_year: int,
    train_bop_cutoff: int,
    test_bop_cutoff: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    config_id = config_search_config_id(train_policy, train_bop_cutoff, test_bop_cutoff)
    prefix = config_search_prefix(
        phase=phase,
        train_policy=train_policy,
        validation_fold=validation_fold,
        train_start_year=train_start_year,
        train_end_year=train_end_year,
        test_start_year=test_start_year,
        test_end_year=test_end_year,
        train_bop_cutoff=train_bop_cutoff,
        test_bop_cutoff=test_bop_cutoff,
        config_id=config_id,
    )
    train_rows, holdout_rows = config_search_split_rows(
        panel_rows,
        snapshot_day=snapshot_day,
        train_start_year=train_start_year,
        train_end_year=train_end_year,
        test_start_year=test_start_year,
        test_end_year=test_end_year,
        train_bop_cutoff=train_bop_cutoff,
        test_bop_cutoff=test_bop_cutoff,
    )
    split_panel = train_rows + holdout_rows
    combo_predictions, combo_metrics, _combo_coefficients, _combo_headline = evaluate_wiki_comp_combo_timing(
        split_panel,
        snapshot_days=[snapshot_day],
        train_start_year=train_start_year,
        train_end_year=train_end_year,
        test_start_year=test_start_year,
        test_end_year=test_end_year,
        min_bop_forecast_midpoint=0,
    )
    improved_predictions, improved_metrics, _improved_coefficients, _improved_headline = evaluate_wiki_comp_improved_timing(
        split_panel,
        snapshot_days=[snapshot_day],
        train_start_year=train_start_year,
        train_end_year=train_end_year,
        test_start_year=test_start_year,
        test_end_year=test_end_year,
        min_bop_forecast_midpoint=0,
    )
    return (
        enrich_config_rows(combo_metrics + improved_metrics, prefix),
        enrich_config_rows(combo_predictions + improved_predictions, prefix),
    )


def config_search_validation_specs() -> list[tuple[str, int, int, int]]:
    specs: list[tuple[str, int, int, int]] = []
    for train_start in CONFIG_SEARCH_TRAIN_START_POLICIES:
        train_policy = f"{train_start}_start"
        for validation_year in CONFIG_SEARCH_VALIDATION_YEARS:
            train_end = validation_year - 1
            if train_start <= train_end:
                specs.append((train_policy, train_start, train_end, validation_year))
    return specs


def evaluate_config_search(
    panel_rows: list[dict[str, object]],
    *,
    snapshot_days: Iterable[int] = CONFIG_SEARCH_SNAPSHOT_DAYS,
    train_bop_cutoffs: Iterable[int] = CONFIG_SEARCH_TRAIN_BOP_CUTOFFS,
    test_bop_cutoffs: Iterable[int] = CONFIG_SEARCH_TEST_BOP_CUTOFFS,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for train_policy, train_start, train_end, validation_year in config_search_validation_specs():
        validation_fold = f"{train_start}-{train_end}_to_{validation_year}"
        for train_bop_cutoff in train_bop_cutoffs:
            for test_bop_cutoff in test_bop_cutoffs:
                for snapshot_day in snapshot_days:
                    metrics, predictions = evaluate_config_search_split(
                        panel_rows,
                        phase="validation",
                        train_policy=train_policy,
                        validation_fold=validation_fold,
                        snapshot_day=snapshot_day,
                        train_start_year=train_start,
                        train_end_year=train_end,
                        test_start_year=validation_year,
                        test_end_year=validation_year,
                        train_bop_cutoff=int(train_bop_cutoff),
                        test_bop_cutoff=int(test_bop_cutoff),
                    )
                    metric_rows.extend(metrics)
                    prediction_rows.extend(predictions)
    return metric_rows, prediction_rows


def numeric_metric(row: dict[str, object], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def weighted_mean_from_rows(rows: list[dict[str, object]], key: str) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        value = numeric_metric(row, key)
        weight = float(row.get("holdout_n", 0.0) or 0.0)
        if value is None or weight <= 0.0:
            continue
        numerator += value * weight
        denominator += weight
    return numerator / denominator if denominator > 0.0 else None


def aggregate_config_search_rows(
    metric_rows: list[dict[str, object]],
    *,
    rank_days: Iterable[int] | None = None,
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    rank_day_set = set(rank_days) if rank_days is not None else None
    for row in metric_rows:
        if row.get("search_phase") != "validation" or row.get("status") != "ok":
            continue
        snapshot_day = int(row["snapshot_day"])
        if rank_day_set is not None and snapshot_day not in rank_day_set:
            continue
        key = (
            row["train_policy"],
            row["config_id"],
            row["train_bop_cutoff"],
            row["test_bop_cutoff"],
            row["model"],
            row["model_family"],
            row["target_policy"] if rank_day_set is None else "rank_days",
            snapshot_day if rank_day_set is None else "rank_days",
        )
        groups[key].append(row)

    out: list[dict[str, object]] = []
    for key, rows in groups.items():
        (
            train_policy,
            config_id,
            train_bop_cutoff,
            test_bop_cutoff,
            model,
            model_family,
            target_policy,
            snapshot_key,
        ) = key
        mape_values = [float(row["mape_gross"]) for row in rows if row.get("mape_gross", "") != ""]
        train_ns = [int(row.get("train_n", 0) or 0) for row in rows]
        holdout_total = sum(int(row.get("holdout_n", 0) or 0) for row in rows)
        folds = sorted({str(row["validation_fold"]) for row in rows})
        train_starts = sorted({int(row["config_train_start_year"]) for row in rows})
        train_ends = sorted({int(row["config_train_end_year"]) for row in rows})
        out.append(
            {
                "selection_scope": "rank_days" if rank_day_set is not None else "snapshot",
                "train_policy": train_policy,
                "config_id": config_id,
                "train_bop_cutoff": train_bop_cutoff,
                "test_bop_cutoff": test_bop_cutoff,
                "model": model,
                "model_family": model_family,
                "target_policy": target_policy,
                "snapshot_day": snapshot_key,
                "train_start_min": min(train_starts) if train_starts else "",
                "train_end_max": max(train_ends) if train_ends else "",
                "usable_folds": len(folds),
                "validation_folds": ",".join(folds),
                "total_holdout_n": holdout_total,
                "min_train_n": min(train_ns) if train_ns else "",
                "weighted_mape_gross": format_number(weighted_mean_from_rows(rows, "mape_gross")),
                "median_mape_gross": format_number(median(mape_values)),
                "worst_fold_mape_gross": format_number(max(mape_values) if mape_values else None),
                "mape_stdev": format_number(stdev(mape_values)),
                "weighted_rmse_log_revenue": format_number(weighted_mean_from_rows(rows, "rmse_log_revenue")),
                "weighted_r2_gross": format_number(weighted_mean_from_rows(rows, "r2_gross")),
                "low_sample_flag": len(folds) < 2 or holdout_total < 10 or (min(train_ns) if train_ns else 0) < 10,
                "status": "ok",
            }
        )
    return out


def best_config_search_rows(
    snapshot_aggregates: list[dict[str, object]],
    rank_aggregates: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    best_by_snapshot: list[dict[str, object]] = []
    by_snapshot_model: dict[tuple[object, object], list[dict[str, object]]] = defaultdict(list)
    by_snapshot: dict[object, list[dict[str, object]]] = defaultdict(list)
    for row in snapshot_aggregates:
        if row.get("weighted_mape_gross", "") == "":
            continue
        by_snapshot_model[(row["snapshot_day"], row["model"])].append(row)
        by_snapshot[row["snapshot_day"]].append(row)
    for (snapshot_day, model), rows in sorted(by_snapshot_model.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
        best = min(rows, key=lambda row: float(row["weighted_mape_gross"]))
        best_by_snapshot.append({**best, "selection_scope": "best_config_for_model_snapshot"})
    for snapshot_day, rows in sorted(by_snapshot.items(), key=lambda item: str(item[0])):
        best = min(rows, key=lambda row: float(row["weighted_mape_gross"]))
        best_by_snapshot.append({**best, "selection_scope": "best_overall_for_snapshot"})

    best_overall: list[dict[str, object]] = []
    rank_rows = [row for row in rank_aggregates if row.get("weighted_mape_gross", "") != ""]
    by_model: dict[object, list[dict[str, object]]] = defaultdict(list)
    for row in rank_rows:
        by_model[row["model"]].append(row)
    for model, rows in sorted(by_model.items(), key=lambda item: str(item[0])):
        best = min(rows, key=lambda row: float(row["weighted_mape_gross"]))
        best_overall.append({**best, "selection_scope": "best_config_for_model_rank_days"})
    if rank_rows:
        best = min(rank_rows, key=lambda row: float(row["weighted_mape_gross"]))
        best_overall.append({**best, "selection_scope": "best_overall_rank_days"})
    primary_rows = [row for row in rank_rows if int(row["test_bop_cutoff"]) == CONFIG_SEARCH_PRIMARY_TEST_BOP_CUTOFF]
    if primary_rows:
        best = min(primary_rows, key=lambda row: float(row["weighted_mape_gross"]))
        best_overall.append({**best, "selection_scope": "best_primary_test_cutoff_rank_days"})
    return best_by_snapshot, best_overall


def selected_final_confirmation_specs(
    best_by_snapshot: list[dict[str, object]],
    best_overall: list[dict[str, object]],
) -> list[dict[str, object]]:
    selected: dict[tuple[object, ...], dict[str, object]] = {}
    for row in best_overall:
        selected[
            (
                row["train_policy"],
                row["train_bop_cutoff"],
                row["test_bop_cutoff"],
                row["model"],
                "rank_days",
            )
        ] = row
    for row in best_by_snapshot:
        if row["selection_scope"] != "best_overall_for_snapshot":
            continue
        selected[
            (
                row["train_policy"],
                row["train_bop_cutoff"],
                row["test_bop_cutoff"],
                row["model"],
                row["snapshot_day"],
            )
        ] = row
    return list(selected.values())


def evaluate_config_search_final_confirmation(
    panel_rows: list[dict[str, object]],
    selected_specs: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in selected_specs:
        train_policy = str(spec["train_policy"])
        train_start = int(str(train_policy).split("_", maxsplit=1)[0])
        train_bop_cutoff = int(spec["train_bop_cutoff"])
        test_bop_cutoff = int(spec["test_bop_cutoff"])
        model = str(spec["model"])
        if spec["snapshot_day"] == "rank_days":
            snapshot_days = CONFIG_SEARCH_RANK_DAYS
        else:
            snapshot_days = (int(spec["snapshot_day"]),)
        for snapshot_day in snapshot_days:
            metrics, _predictions = evaluate_config_search_split(
                panel_rows,
                phase="final_2026_confirmation",
                train_policy=train_policy,
                validation_fold="train_to_2025_test_2026",
                snapshot_day=snapshot_day,
                train_start_year=train_start,
                train_end_year=2025,
                test_start_year=CONFIG_SEARCH_FINAL_TEST_YEAR,
                test_end_year=CONFIG_SEARCH_FINAL_TEST_YEAR,
                train_bop_cutoff=train_bop_cutoff,
                test_bop_cutoff=test_bop_cutoff,
            )
            for metric in metrics:
                if metric["model"] == model:
                    rows.append(
                        {
                            "selected_from": spec["selection_scope"],
                            "selected_validation_weighted_mape_gross": spec.get("weighted_mape_gross", ""),
                            **metric,
                        }
                    )
    return rows


def write_config_search_outputs(
    out_dir: Path,
    *,
    metric_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    best_by_snapshot_rows: list[dict[str, object]],
    best_overall_rows: list[dict[str, object]],
    final_confirmation_rows: list[dict[str, object]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "config_search_metrics.csv", metric_rows, CONFIG_SEARCH_METRIC_FIELDNAMES)
    write_csv(out_dir / "config_search_predictions.csv", prediction_rows, CONFIG_SEARCH_PREDICTION_FIELDNAMES)
    write_csv(out_dir / "config_search_best_by_snapshot.csv", best_by_snapshot_rows, CONFIG_SEARCH_BEST_FIELDNAMES)
    write_csv(out_dir / "config_search_best_overall.csv", best_overall_rows, CONFIG_SEARCH_BEST_FIELDNAMES)
    write_csv(
        out_dir / "config_search_final_2026_confirmation.csv",
        final_confirmation_rows,
        CONFIG_SEARCH_FINAL_FIELDNAMES,
    )


def run_config_search(panel_rows: list[dict[str, object]], *, out_dir: Path) -> dict[str, int]:
    metric_rows, prediction_rows = evaluate_config_search(panel_rows)
    snapshot_aggregates = aggregate_config_search_rows(metric_rows)
    rank_aggregates = aggregate_config_search_rows(metric_rows, rank_days=CONFIG_SEARCH_RANK_DAYS)
    best_by_snapshot, best_overall = best_config_search_rows(snapshot_aggregates, rank_aggregates)
    final_confirmation = evaluate_config_search_final_confirmation(
        panel_rows,
        selected_final_confirmation_specs(best_by_snapshot, best_overall),
    )
    write_config_search_outputs(
        out_dir,
        metric_rows=metric_rows,
        prediction_rows=prediction_rows,
        best_by_snapshot_rows=best_by_snapshot,
        best_overall_rows=best_overall,
        final_confirmation_rows=final_confirmation,
    )
    return {
        "metrics": len(metric_rows),
        "predictions": len(prediction_rows),
        "best_by_snapshot": len(best_by_snapshot),
        "best_overall": len(best_overall),
        "final_confirmation": len(final_confirmation),
    }


def checkpoint_name_for_snapshot_day(snapshot_day: int) -> str:
    return CHECKPOINT_BY_SNAPSHOT_DAY.get(snapshot_day, f"snapshot_day_{snapshot_day}")


def competitive_checkpoint_scoped_rows(
    panel_rows: list[dict[str, object]],
    *,
    snapshot_day: int,
    min_bop_forecast_midpoint: int = 0,
) -> list[dict[str, object]]:
    return [
        row
        for row in panel_rows
        if int(row["snapshot_day"]) == snapshot_day
        and float(row.get("bop_forecast_available", 0.0) or 0.0) > 0.0
        and float(row.get("bop_forecast_midpoint", 0.0) or 0.0) >= min_bop_forecast_midpoint
        and row.get("target_log_bop_residual", "") != ""
    ]


def split_competitive_checkpoint_rows(
    rows: list[dict[str, object]],
    *,
    train_start_year: int,
    train_end_year: int,
    test_start_year: int,
    test_end_year: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train_rows = [
        row for row in rows if train_start_year <= int(row["release_year"]) <= train_end_year
    ]
    holdout_rows = [
        row for row in rows if test_start_year <= int(row["release_year"]) <= test_end_year
    ]
    return train_rows, holdout_rows


def fit_expected_competition_model(
    train_rows: list[dict[str, object]],
    *,
    target_key: str,
) -> tuple[opening.FittedModel | None, float]:
    values = [float(row.get(target_key, 0.0) or 0.0) for row in train_rows]
    fallback_mean = mean(values) if values else 0.0
    terms = ["log1p_bop_forecast_midpoint"]
    if len(train_rows) < len(terms) + 2:
        return None, fallback_mean
    try:
        return opening.fit_model_for_target(train_rows, terms, target_key), fallback_mean
    except ValueError:
        return None, fallback_mean


def predict_expected_competition(
    model: opening.FittedModel | None,
    rows: list[dict[str, object]],
    *,
    fallback_mean: float,
) -> list[float]:
    if model is None:
        return [fallback_mean for _row in rows]
    return opening.predict_log(model, rows)


def add_competitive_residual_features(
    train_rows: list[dict[str, object]],
    holdout_rows: list[dict[str, object]],
    *,
    checkpoint: str,
    snapshot_day: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    train_aug = [dict(row) for row in train_rows]
    holdout_aug = [dict(row) for row in holdout_rows]
    coefficient_rows: list[dict[str, object]] = []
    signal_targets = {
        "lag1": "log1p_competitor_total_gross_lag1",
        "lag7": "log1p_competitor_total_gross_lag7",
        "previous_weekend": "log1p_competitor_total_gross_previous_weekend",
    }
    for label, target_key in signal_targets.items():
        model, fallback_mean = fit_expected_competition_model(train_aug, target_key=target_key)
        train_expected = predict_expected_competition(model, train_aug, fallback_mean=fallback_mean)
        holdout_expected = predict_expected_competition(model, holdout_aug, fallback_mean=fallback_mean)
        for row, expected in zip(train_aug, train_expected):
            row[f"competitive_residual_{label}"] = float(row.get(target_key, 0.0) or 0.0) - expected
        for row, expected in zip(holdout_aug, holdout_expected):
            row[f"competitive_residual_{label}"] = float(row.get(target_key, 0.0) or 0.0) - expected
        if model is None:
            coefficient_rows.append(
                {
                    "checkpoint": checkpoint,
                    "snapshot_day": snapshot_day,
                    "model": f"expected_competition_{label}",
                    "target": target_key,
                    "term": "fallback_mean",
                    "standardized_coef": fallback_mean,
                    "center": "",
                    "scale": "",
                    "train_n": len(train_aug),
                }
            )
        else:
            for term, coef, center, scale in zip(
                ["intercept"] + model.terms,
                model.beta,
                [0.0] + model.centers,
                [1.0] + model.scales,
            ):
                coefficient_rows.append(
                    {
                        "checkpoint": checkpoint,
                        "snapshot_day": snapshot_day,
                        "model": f"expected_competition_{label}",
                        "target": target_key,
                        "term": term,
                        "standardized_coef": coef,
                        "center": center,
                        "scale": scale,
                        "train_n": len(train_aug),
                    }
                )
    return train_aug, holdout_aug, coefficient_rows


def reconciled_prediction_gross(row: dict[str, object], *, predicted_log: float) -> float:
    base = max(1.0, math.exp(predicted_log))
    known = float(row.get("known_weekend_gross_so_far_usd", 0.0) or 0.0)
    return max(base, known, 1.0)


def competitive_checkpoint_prediction_rows(
    *,
    checkpoint: str,
    snapshot_day: int,
    model_name: str,
    rows: list[dict[str, object]],
    predicted_residuals: list[float],
) -> list[dict[str, object]]:
    out = []
    for row, predicted_residual in zip(rows, predicted_residuals):
        midpoint = float(row["bop_forecast_midpoint"])
        predicted_log = math.log(max(1.0, midpoint)) + predicted_residual
        predicted_gross = reconciled_prediction_gross(row, predicted_log=predicted_log)
        actual_gross = float(row["opening_weekend_revenue_usd"])
        out.append(
            {
                "checkpoint": checkpoint,
                "snapshot_day": snapshot_day,
                "model": model_name,
                "movie_id": row["movie_id"],
                "title": row["title"],
                "release_year": row["release_year"],
                "opening_date": row["opening_date"],
                "as_of_date": row["as_of_date"],
                "bop_forecast_midpoint": midpoint,
                "preview_gross_known_usd": row.get("preview_gross_known_usd", 0.0),
                "preview_day_count": row.get("preview_day_count", 0.0),
                "known_friday_gross_usd": row.get("known_friday_gross_usd", 0.0),
                "known_saturday_gross_usd": row.get("known_saturday_gross_usd", 0.0),
                "known_weekend_gross_so_far_usd": row.get("known_weekend_gross_so_far_usd", 0.0),
                "known_gross_so_far_pct_of_bop_midpoint": row.get("known_gross_so_far_pct_of_bop_midpoint", 0.0),
                "competitive_residual_lag1": row.get("competitive_residual_lag1", 0.0),
                "competitive_residual_lag7": row.get("competitive_residual_lag7", 0.0),
                "competitive_residual_previous_weekend": row.get("competitive_residual_previous_weekend", 0.0),
                "share_attraction_focal_share_lag7": row.get("share_attraction_focal_share_lag7", 0.0),
                "share_attraction_competitor_pressure_lag7": row.get("share_attraction_competitor_pressure_lag7", 0.0),
                "share_attraction_top1_pressure_lag7": row.get("share_attraction_top1_pressure_lag7", 0.0),
                "logit_demand_focal_vs_competition_lag7": row.get("logit_demand_focal_vs_competition_lag7", 0.0),
                "logit_demand_focal_vs_top1_lag7": row.get("logit_demand_focal_vs_top1_lag7", 0.0),
                "log1p_logit_demand_choice_set_lag7": row.get("log1p_logit_demand_choice_set_lag7", 0.0),
                "actual_log_bop_residual": row["target_log_bop_residual"],
                "predicted_log_bop_residual": predicted_residual,
                "actual_log_opening_weekend": row["target_log_opening_weekend"],
                "predicted_log_opening_weekend": math.log(predicted_gross),
                "actual_opening_weekend_revenue_usd": actual_gross,
                "predicted_opening_weekend_revenue_usd": predicted_gross,
                "absolute_percentage_error": abs(predicted_gross - actual_gross) / actual_gross if actual_gross > 0.0 else "",
            }
        )
    return out


def competitive_checkpoint_metric_row(
    *,
    checkpoint: str,
    snapshot_day: int,
    model_name: str,
    train_start_year: int,
    train_end_year: int,
    test_start_year: int,
    test_end_year: int,
    train_n: int,
    prediction_rows: list[dict[str, object]],
) -> dict[str, object]:
    base = {
        "checkpoint": checkpoint,
        "snapshot_day": snapshot_day,
        "model": model_name,
        "train_start_year": train_start_year,
        "train_end_year": train_end_year,
        "test_start_year": test_start_year,
        "test_end_year": test_end_year,
        "train_n": train_n,
        "holdout_n": len(prediction_rows),
    }
    if len(prediction_rows) < 2:
        return {
            **base,
            "r2_log_revenue": "",
            "r2_gross": "",
            "mape_gross": "",
            "accuracy_pct": "",
            "rmse_log_revenue": "",
            "mae_log_revenue": "",
            "mean_actual_gross": "",
            "mean_predicted_gross": "",
            "status": "insufficient_sample",
        }
    actual_log = [float(row["actual_log_opening_weekend"]) for row in prediction_rows]
    pred_log = [float(row["predicted_log_opening_weekend"]) for row in prediction_rows]
    actual_gross = [float(row["actual_opening_weekend_revenue_usd"]) for row in prediction_rows]
    pred_gross = [float(row["predicted_opening_weekend_revenue_usd"]) for row in prediction_rows]
    log_errors = [actual - pred for actual, pred in zip(actual_log, pred_log)]
    apes = [abs(pred - actual) / actual for actual, pred in zip(actual_gross, pred_gross) if actual > 0.0]
    mape = mean(apes) if apes else None
    return {
        **base,
        "r2_log_revenue": format_number(r2_score(actual_log, pred_log)),
        "r2_gross": format_number(r2_score(actual_gross, pred_gross)),
        "mape_gross": format_number(mape),
        "accuracy_pct": format_number(1.0 - mape if mape is not None else None),
        "rmse_log_revenue": format_number(opening.rmse(log_errors)),
        "mae_log_revenue": format_number(opening.mae(log_errors)),
        "mean_actual_gross": format_number(mean(actual_gross)),
        "mean_predicted_gross": format_number(mean(pred_gross)),
        "status": "ok",
    }


def evaluate_competitive_checkpoints(
    panel_rows: list[dict[str, object]],
    *,
    train_start_year: int = COMPETITIVE_CHECKPOINT_TRAIN_START_YEAR,
    train_end_year: int = COMPETITIVE_CHECKPOINT_TRAIN_END_YEAR,
    test_start_year: int = COMPETITIVE_CHECKPOINT_TEST_START_YEAR,
    test_end_year: int = COMPETITIVE_CHECKPOINT_TEST_END_YEAR,
    min_bop_forecast_midpoint: int = DEFAULT_MIN_BOP_FORECAST_MIDPOINT,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    for snapshot_day in COMPETITIVE_CHECKPOINT_DAYS:
        checkpoint = checkpoint_name_for_snapshot_day(snapshot_day)
        scoped_rows = competitive_checkpoint_scoped_rows(
            panel_rows,
            snapshot_day=snapshot_day,
            min_bop_forecast_midpoint=min_bop_forecast_midpoint,
        )
        train_rows, holdout_rows = split_competitive_checkpoint_rows(
            scoped_rows,
            train_start_year=train_start_year,
            train_end_year=train_end_year,
            test_start_year=test_start_year,
            test_end_year=test_end_year,
        )
        train_aug, holdout_aug, expected_coefficients = add_competitive_residual_features(
            train_rows,
            holdout_rows,
            checkpoint=checkpoint,
            snapshot_day=snapshot_day,
        )
        coefficient_rows.extend(expected_coefficients)
        for model_name in COMPETITIVE_CHECKPOINT_MODELS_BY_DAY[snapshot_day]:
            terms = COMPETITIVE_CHECKPOINT_MODEL_TERMS.get(model_name, [])
            train_n = 0 if model_name == "raw_bop_reconciled" else len(train_aug)
            if not holdout_aug or (model_name != "raw_bop_reconciled" and len(train_aug) < len(terms) + 2):
                metric_rows.append(
                    competitive_checkpoint_metric_row(
                        checkpoint=checkpoint,
                        snapshot_day=snapshot_day,
                        model_name=model_name,
                        train_start_year=train_start_year,
                        train_end_year=train_end_year,
                        test_start_year=test_start_year,
                        test_end_year=test_end_year,
                        train_n=train_n,
                        prediction_rows=[],
                    )
                )
                continue
            if model_name == "raw_bop_reconciled":
                predicted_residuals = [0.0 for _row in holdout_aug]
            else:
                fitted = opening.fit_model_for_target(train_aug, terms, "target_log_bop_residual")
                predicted_residuals = opening.predict_log(fitted, holdout_aug)
                for term, coef, center, scale in zip(
                    ["intercept"] + terms,
                    fitted.beta,
                    [0.0] + fitted.centers,
                    [1.0] + fitted.scales,
                ):
                    coefficient_rows.append(
                        {
                            "checkpoint": checkpoint,
                            "snapshot_day": snapshot_day,
                            "model": model_name,
                            "target": "target_log_bop_residual",
                            "term": term,
                            "standardized_coef": coef,
                            "center": center,
                            "scale": scale,
                            "train_n": len(train_aug),
                        }
                    )
            rows = competitive_checkpoint_prediction_rows(
                checkpoint=checkpoint,
                snapshot_day=snapshot_day,
                model_name=model_name,
                rows=holdout_aug,
                predicted_residuals=predicted_residuals,
            )
            prediction_rows.extend(rows)
            metric_rows.append(
                competitive_checkpoint_metric_row(
                    checkpoint=checkpoint,
                    snapshot_day=snapshot_day,
                    model_name=model_name,
                    train_start_year=train_start_year,
                    train_end_year=train_end_year,
                    test_start_year=test_start_year,
                    test_end_year=test_end_year,
                    train_n=train_n,
                    prediction_rows=rows,
                )
            )
    headline_rows = competitive_checkpoint_headline_rows(metric_rows)
    return prediction_rows, metric_rows, coefficient_rows, headline_rows


def competitive_checkpoint_headline_rows(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    comparisons = [
        ("pre_release_tminus1", "raw_bop_reconciled", "tminus1_competition"),
        ("pre_release_tminus1", "raw_bop_reconciled", "tminus1_share_attraction"),
        ("pre_release_tminus1", "raw_bop_reconciled", "tminus1_logit_demand"),
        ("friday_actual", "friday_actual_only", "friday_actual_competition"),
        ("friday_actual", "friday_actual_only", "friday_actual_share_attraction"),
        ("friday_actual", "friday_actual_only", "friday_actual_logit_demand"),
        ("saturday_actual", "saturday_actual_only", "saturday_actual_competition"),
        ("saturday_actual", "saturday_actual_only", "saturday_actual_share_attraction"),
        ("saturday_actual", "saturday_actual_only", "saturday_actual_logit_demand"),
    ]
    by_key = {
        (str(row["checkpoint"]), str(row["model"])): row
        for row in metric_rows
        if row["status"] == "ok"
    }
    for checkpoint, baseline_model, competition_model in comparisons:
        baseline = by_key.get((checkpoint, baseline_model))
        competition = by_key.get((checkpoint, competition_model))
        if baseline is None or competition is None:
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "snapshot_day": "",
                    "baseline_model": baseline_model,
                    "competition_model": competition_model,
                    "holdout_n": "",
                    "baseline_mape_gross": "",
                    "competition_mape_gross": "",
                    "mape_delta_competition_minus_baseline": "",
                    "baseline_rmse_log_revenue": "",
                    "competition_rmse_log_revenue": "",
                    "rmse_log_delta_competition_minus_baseline": "",
                    "baseline_r2_gross": "",
                    "competition_r2_gross": "",
                    "r2_gross_delta_competition_minus_baseline": "",
                    "competition_better_mape": "",
                    "status": "insufficient_sample",
                }
            )
            continue
        baseline_mape = float(baseline["mape_gross"])
        competition_mape = float(competition["mape_gross"])
        baseline_rmse = float(baseline["rmse_log_revenue"])
        competition_rmse = float(competition["rmse_log_revenue"])
        baseline_r2 = float(baseline["r2_gross"])
        competition_r2 = float(competition["r2_gross"])
        rows.append(
            {
                "checkpoint": checkpoint,
                "snapshot_day": competition["snapshot_day"],
                "baseline_model": baseline_model,
                "competition_model": competition_model,
                "holdout_n": competition["holdout_n"],
                "baseline_mape_gross": baseline["mape_gross"],
                "competition_mape_gross": competition["mape_gross"],
                "mape_delta_competition_minus_baseline": format_number(competition_mape - baseline_mape),
                "baseline_rmse_log_revenue": baseline["rmse_log_revenue"],
                "competition_rmse_log_revenue": competition["rmse_log_revenue"],
                "rmse_log_delta_competition_minus_baseline": format_number(competition_rmse - baseline_rmse),
                "baseline_r2_gross": baseline["r2_gross"],
                "competition_r2_gross": competition["r2_gross"],
                "r2_gross_delta_competition_minus_baseline": format_number(competition_r2 - baseline_r2),
                "competition_better_mape": competition_mape < baseline_mape,
                "status": "ok",
            }
        )
    return rows


def prediction_rows_for_selection(
    rows: list[dict[str, object]],
    pred_logs: list[float],
    *,
    model_name: str,
    snapshot_day: int,
    switch_day: int,
    train_cutoff: int,
    test_cutoff: int,
    split_label: str,
    train_start_year: int,
    train_end_year: int,
    test_year: int,
    remainder_models: dict[tuple[int, ...], float],
) -> list[dict[str, object]]:
    out = []
    for row, pred_log in zip(rows, pred_logs):
        base_prediction = max(1.0, math.exp(pred_log))
        displayed_prediction, locked_offsets = displayed_prediction_usd_for_snapshot(
            row,
            snapshot_day=snapshot_day,
            base_prediction_usd=base_prediction,
            remainder_models=remainder_models,
        )
        actual = float(row["opening_weekend_revenue_usd"])
        out.append(
            {
                "model": model_name,
                "population": "bop_covered",
                "prediction_source": "bop",
                "forecast_stage": row["forecast_stage"],
                "snapshot_day": snapshot_day,
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
                "predicted_log_opening_weekend": math.log(displayed_prediction),
                "actual_opening_weekend_revenue_usd": row["opening_weekend_revenue_usd"],
                "predicted_opening_weekend_revenue_usd": displayed_prediction,
                "predicted_p50_opening_weekend_revenue_usd": displayed_prediction,
                "predicted_lower_50_opening_weekend_revenue_usd": displayed_prediction,
                "predicted_upper_50_opening_weekend_revenue_usd": displayed_prediction,
                "predicted_lower_80_opening_weekend_revenue_usd": displayed_prediction,
                "predicted_upper_80_opening_weekend_revenue_usd": displayed_prediction,
                "predicted_lower_90_opening_weekend_revenue_usd": displayed_prediction,
                "predicted_upper_90_opening_weekend_revenue_usd": displayed_prediction,
                "prediction_interval_method": "displayed_forecast",
                "prediction_interval_source": "reconciled" if locked_offsets else "model",
                "prediction_interval_train_n": "",
                "absolute_percentage_error": abs(displayed_prediction - actual) / actual if actual > 0.0 else "",
                "split": split_label,
                "train_start_year": train_start_year,
                "train_end_year": train_end_year,
                "test_year": test_year,
                "train_cutoff": train_cutoff,
                "test_cutoff": test_cutoff,
                "switch_day": switch_day,
                "reconciled": bool(locked_offsets),
            }
        )
    return out


def selection_metric_row(
    *,
    base: dict[str, object],
    prediction_rows: list[dict[str, object]],
) -> dict[str, object]:
    row = metric_row_from_predictions(base=base, prediction_rows=prediction_rows)
    row["reconciled"] = any(bool(prediction.get("reconciled")) for prediction in prediction_rows)
    return row


def evaluate_competitive_selection(
    panel_rows: list[dict[str, object]],
    *,
    snapshot_days: list[int],
    train_opening_day_gross_cutoffs: list[int],
    test_min_opening_day_gross: int,
    splits: list[tuple[int, int, int]] | None = None,
    switch_days: list[int] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    split_specs = splits or default_competitive_selection_splits()
    candidate_switch_days = switch_days or list(snapshot_days)
    metric_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    wiki_terms = SNAPSHOT_MODEL_TERMS["bop_residual_wiki_snapshot"]
    combined_terms = SNAPSHOT_MODEL_TERMS["bop_residual_wiki_competition_snapshot"]

    for train_start_year, train_end_year, test_year in split_specs:
        split_label = f"{train_start_year}-{train_end_year}_to_{test_year}"
        for train_cutoff in train_opening_day_gross_cutoffs:
            train_rows_for_remainder = [
                row
                for row in panel_rows
                if train_start_year <= int(row["release_year"]) <= train_end_year
                and float(row.get("opening_day_gross_usd", 0.0) or 0.0) >= train_cutoff
            ]
            remainder_models = train_remainder_ratio_models(train_rows_for_remainder)
            models_by_day: dict[int, dict[str, tuple[list[dict[str, object]], list[float]]]] = {}
            train_n_by_day: dict[int, int] = {}
            for snapshot_day in snapshot_days:
                train_all, holdout_all = train_holdout_rows(
                    panel_rows,
                    snapshot_day=snapshot_day,
                    train_start_year=train_start_year,
                    train_end_year=train_end_year,
                    test_start_year=test_year,
                    test_end_year=test_year,
                    train_min_opening_day_gross=train_cutoff,
                    test_min_opening_day_gross=test_min_opening_day_gross,
                )
                train_bop = rows_with_target(bop_rows(train_all), "target_log_bop_residual")
                train_n_by_day[snapshot_day] = len(train_bop)
                holdout_bop = [
                    row
                    for row in bop_rows(holdout_all)
                    if float(row.get("wiki_available", 0.0) or 0.0) > 0.0
                ]
                if len(holdout_bop) < 2:
                    continue
                fitted_models: dict[str, opening.FittedModel] = {}
                if len(train_bop) >= len(wiki_terms) + 2:
                    fitted_models["bop_residual_wiki_snapshot"] = opening.fit_model_for_target(
                        train_bop,
                        wiki_terms,
                        "target_log_bop_residual",
                    )
                if len(train_bop) >= len(combined_terms) + 2:
                    fitted_models["bop_residual_wiki_competition_snapshot"] = opening.fit_model_for_target(
                        train_bop,
                        combined_terms,
                        "target_log_bop_residual",
                    )
                if "bop_residual_wiki_snapshot" not in fitted_models:
                    continue

                models_by_day[snapshot_day] = {}
                for model_name, fitted in fitted_models.items():
                    terms = wiki_terms if model_name == "bop_residual_wiki_snapshot" else combined_terms
                    eligible = [
                        row
                        for row in holdout_bop
                        if model_name == "bop_residual_wiki_snapshot"
                        or float(row.get("competitor_count_lag7", 0.0) or 0.0) > 0.0
                    ]
                    if len(eligible) < 2:
                        continue
                    residual_logs = opening.predict_log(fitted, eligible)
                    pred_logs = [
                        math.log(max(1.0, float(row["bop_forecast_midpoint"]))) + residual
                        for row, residual in zip(eligible, residual_logs)
                    ]
                    models_by_day[snapshot_day][model_name] = (eligible, pred_logs)

                wiki_model = models_by_day[snapshot_day].get("bop_residual_wiki_snapshot")
                combined_model = models_by_day[snapshot_day].get("bop_residual_wiki_competition_snapshot")
                if wiki_model is not None and combined_model is not None:
                    combined_ids = {int(row["movie_id"]) for row in combined_model[0]}
                    wiki_rows = [row for row in wiki_model[0] if int(row["movie_id"]) in combined_ids]
                    wiki_log_by_movie = {
                        int(row["movie_id"]): pred_log
                        for row, pred_log in zip(wiki_model[0], wiki_model[1])
                    }
                    wiki_preds = prediction_rows_for_selection(
                        wiki_rows,
                        [wiki_log_by_movie[int(row["movie_id"])] for row in wiki_rows],
                        model_name="bop_residual_wiki_snapshot",
                        snapshot_day=snapshot_day,
                        switch_day=999,
                        train_cutoff=train_cutoff,
                        test_cutoff=test_min_opening_day_gross,
                        split_label=split_label,
                        train_start_year=train_start_year,
                        train_end_year=train_end_year,
                        test_year=test_year,
                        remainder_models=remainder_models,
                    )
                    combined_preds = prediction_rows_for_selection(
                        combined_model[0],
                        combined_model[1],
                        model_name="bop_residual_wiki_competition_snapshot",
                        snapshot_day=snapshot_day,
                        switch_day=-999,
                        train_cutoff=train_cutoff,
                        test_cutoff=test_min_opening_day_gross,
                        split_label=split_label,
                        train_start_year=train_start_year,
                        train_end_year=train_end_year,
                        test_year=test_year,
                        remainder_models=remainder_models,
                    )
                    wiki_mape = float(metric_row_from_predictions(base={}, prediction_rows=wiki_preds)["mape_gross"])
                    combined_mape = float(metric_row_from_predictions(base={}, prediction_rows=combined_preds)["mape_gross"])
                    delta_rows.append(
                        {
                            "split": split_label,
                            "train_start_year": train_start_year,
                            "train_end_year": train_end_year,
                            "test_year": test_year,
                            "train_cutoff": train_cutoff,
                            "test_cutoff": test_min_opening_day_gross,
                            "snapshot_day": snapshot_day,
                            "holdout_n": len(combined_preds),
                            "wiki_mape_gross": format_number(wiki_mape),
                            "wiki_competition_mape_gross": format_number(combined_mape),
                            "mape_delta_competition_minus_wiki": format_number(combined_mape - wiki_mape),
                            "competition_better": combined_mape < wiki_mape,
                            "reconciled": any(bool(row["reconciled"]) for row in combined_preds),
                        }
                    )

            for switch_day in candidate_switch_days:
                for snapshot_day in snapshot_days:
                    day_models = models_by_day.get(snapshot_day, {})
                    wiki_model = day_models.get("bop_residual_wiki_snapshot")
                    if wiki_model is None:
                        continue
                    combined_model = day_models.get("bop_residual_wiki_competition_snapshot")
                    use_combined = combined_model is not None and snapshot_day >= switch_day
                    if use_combined:
                        combined_log_by_movie = {
                            int(row["movie_id"]): pred_log
                            for row, pred_log in zip(combined_model[0], combined_model[1])
                        }
                        selected_rows = []
                        selected_logs = []
                        combined_n = 0
                        wiki_n = 0
                        for row, wiki_log in zip(wiki_model[0], wiki_model[1]):
                            combined_log = combined_log_by_movie.get(int(row["movie_id"]))
                            if combined_log is not None and float(row.get("competitor_count_lag7", 0.0) or 0.0) > 0.0:
                                selected_rows.append(row)
                                selected_logs.append(combined_log)
                                combined_n += 1
                            else:
                                selected_rows.append(row)
                                selected_logs.append(wiki_log)
                                wiki_n += 1
                        model_used = "switch_wiki_competition"
                    else:
                        selected_rows, selected_logs = wiki_model
                        combined_n = 0
                        wiki_n = len(selected_rows)
                        model_used = "bop_residual_wiki_snapshot"
                    if len(selected_rows) < 2:
                        continue
                    predictions = prediction_rows_for_selection(
                        selected_rows,
                        selected_logs,
                        model_name=model_used,
                        snapshot_day=snapshot_day,
                        switch_day=switch_day,
                        train_cutoff=train_cutoff,
                        test_cutoff=test_min_opening_day_gross,
                        split_label=split_label,
                        train_start_year=train_start_year,
                        train_end_year=train_end_year,
                        test_year=test_year,
                        remainder_models=remainder_models,
                    )
                    metric_rows.append(
                        selection_metric_row(
                            base={
                                "split": split_label,
                                "train_start_year": train_start_year,
                                "train_end_year": train_end_year,
                                "test_year": test_year,
                                "train_cutoff": train_cutoff,
                                "test_cutoff": test_min_opening_day_gross,
                                "switch_day": switch_day,
                                "model_used": model_used,
                                "wiki_prediction_n": wiki_n,
                                "wiki_competition_prediction_n": combined_n,
                                "population": "bop_covered",
                                "snapshot_day": snapshot_day,
                                "interval_method": "displayed_forecast",
                                "train_n": train_n_by_day.get(snapshot_day, 0),
                                "holdout_n": len(predictions),
                                "bop_prediction_n": len(predictions),
                                "fallback_prediction_n": 0,
                            },
                            prediction_rows=predictions,
                        )
                    )
    return metric_rows, delta_rows


def weighted_mean_metric(rows: list[dict[str, object]], metric: str) -> float | None:
    values = [
        (float(row[metric]), int(row["holdout_n"]))
        for row in rows
        if row.get(metric, "") != "" and int(row.get("holdout_n", 0) or 0) > 0
    ]
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        return None
    return sum(value * weight for value, weight in values) / total_weight


def best_competitive_selection_rows(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in metric_rows:
        if row.get("status") == "ok" and row.get("mape_gross", "") != "":
            grouped[(int(row["train_cutoff"]), int(row["switch_day"]))].append(row)
    summaries: list[dict[str, object]] = []
    for (train_cutoff, switch_day), rows in grouped.items():
        avg_mape = weighted_mean_metric(rows, "mape_gross")
        avg_rmse = weighted_mean_metric(rows, "rmse_log_revenue")
        avg_r2 = weighted_mean_metric(rows, "r2_gross")
        if avg_mape is None:
            continue
        summaries.append(
            {
                "train_cutoff": train_cutoff,
                "test_cutoff": rows[0]["test_cutoff"],
                "switch_day": switch_day,
                "valid_metric_rows": len(rows),
                "valid_splits": len({row["split"] for row in rows}),
                "valid_snapshot_days": len({int(row["snapshot_day"]) for row in rows}),
                "holdout_n": sum(int(row["holdout_n"]) for row in rows),
                "avg_mape_gross": format_number(avg_mape),
                "avg_rmse_log_revenue": format_number(avg_rmse),
                "avg_r2_gross": format_number(avg_r2),
            }
        )
    if not summaries:
        return []
    best = min(
        summaries,
        key=lambda row: (
            float(row["avg_mape_gross"]),
            float(row["avg_rmse_log_revenue"]) if row["avg_rmse_log_revenue"] != "" else float("inf"),
            -float(row["avg_r2_gross"]) if row["avg_r2_gross"] != "" else float("inf"),
            -int(row["holdout_n"]),
            -int(row["switch_day"]),
        ),
    )
    return [
        {
            **row,
            "rank": index + 1,
            "selected": row is best,
        }
        for index, row in enumerate(
            sorted(
                summaries,
                key=lambda row: (
                    float(row["avg_mape_gross"]),
                    float(row["avg_rmse_log_revenue"]) if row["avg_rmse_log_revenue"] != "" else float("inf"),
                    -float(row["avg_r2_gross"]) if row["avg_r2_gross"] != "" else float("inf"),
                    -int(row["holdout_n"]),
                    -int(row["switch_day"]),
                ),
            )
        )
    ]


FEATURE_PANEL_FIELDNAMES = [
    "movie_id",
    "title",
    "release_year",
    "release_run_id",
    "opening_date",
    "target_type",
    "target_day_count",
    "target_start_date",
    "target_end_date",
    "target_gross",
    "target_gross_usd",
    "target_is_complete_as_of_snapshot",
    "forecast_stage",
    "snapshot_day",
    "timing_day",
    "wiki_timing_day",
    "competition_timing_day",
    "bop_timing_day",
    "as_of_date",
    "feature_available_date",
    "opening_theaters",
    "log1p_opening_theaters",
    "opening_day_gross_usd",
    "opening_weekend_revenue_usd",
    "opening_weekend_day0_gross_usd",
    "opening_weekend_day1_gross_usd",
    "opening_weekend_day2_gross_usd",
    "opening_weekend_day3_gross_usd",
    "opening_weekend_day4_gross_usd",
    "preview_gross_known_usd",
    "preview_day_count",
    "log1p_preview_gross_known",
    "known_friday_gross_usd",
    "known_saturday_gross_usd",
    "known_weekend_gross_so_far_usd",
    "known_gross_so_far_pct_of_bop_midpoint",
    "log1p_known_friday_gross",
    "log1p_known_saturday_gross",
    "log1p_known_weekend_gross_so_far",
    "target_log_opening_weekend",
    "target_log_gross",
    "target_log_bop_residual",
    "target_log_remaining_gross",
    "known_actual_offsets",
    "known_actual_day_count",
    "known_gross_so_far",
    "known_gross_so_far_usd",
    "remaining_gross",
    "remaining_gross_usd",
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
    "bop_forecast_count_as_of",
    "log1p_bop_forecast_count_as_of",
    "bop_first_forecast_midpoint",
    "bop_previous_forecast_midpoint",
    "bop_latest_lead_days",
    "bop_latest_lead_bucket",
    "bop_revision_from_first_pct",
    "bop_revision_from_previous_pct",
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
    "share_attraction_focal_share_lag7",
    "share_attraction_competitor_pressure_lag7",
    "share_attraction_top1_pressure_lag7",
    "logit_demand_focal_vs_competition_lag7",
    "logit_demand_focal_vs_top1_lag7",
    "log1p_logit_demand_choice_set_lag7",
    "release_month_2",
    "release_month_3",
    "release_month_4",
    "release_month_5",
    "release_month_6",
    "release_month_7",
    "release_month_8",
    "release_month_9",
    "release_month_10",
    "release_month_11",
    "release_month_12",
]

PREDICTION_FIELDNAMES = [
    "model",
    "population",
    "prediction_source",
    "target_type",
    "target_day_count",
    "target_start_date",
    "target_end_date",
    "target_gross_usd",
    "forecast_stage",
    "snapshot_day",
    "as_of_date",
    "feature_available_date",
    "movie_id",
    "title",
    "release_year",
    "opening_date",
    "bop_forecast_available",
    "bop_forecast_midpoint",
    "bop_estimate_bucket",
    "bop_forecast_count_as_of",
    "bop_first_forecast_midpoint",
    "bop_previous_forecast_midpoint",
    "bop_latest_lead_days",
    "bop_latest_lead_bucket",
    "bop_revision_from_first_pct",
    "bop_revision_from_previous_pct",
    "wiki_available",
    "known_actual_offsets",
    "known_gross_so_far",
    "remaining_gross",
    "actual_log_opening_weekend",
    "predicted_log_opening_weekend",
    "actual_opening_weekend_revenue_usd",
    "predicted_opening_weekend_revenue_usd",
    "actuals_multiplier_prediction_usd",
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
    "target_type",
    "target_day_count",
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
    "smape_gross",
    "accuracy_pct",
    "mse_log_revenue",
    "mse_gross",
    "mae_usd",
    "rmse_usd",
    "rmse_log_revenue",
    "mae_log_revenue",
    "mean_absolute_log_error",
    "directional_accuracy_vs_estimate",
    "raw_estimate_mae_usd",
    "raw_estimate_mae_lift_usd",
    "raw_estimate_mae_lift_pct",
    "actuals_multiplier_mae_usd",
    "actuals_multiplier_mae_lift_usd",
    "actuals_multiplier_mae_lift_pct",
    "mean_actual_gross",
    "mean_predicted_gross",
    "mean_interval_80_width_pct",
    "coverage_50",
    "coverage_80",
    "coverage_90",
    "status",
]

BEST_METRIC_FIELDNAMES = [
    "target_type",
    "target_day_count",
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
    "accuracy_pct",
    "coverage_80",
    "mean_interval_80_width_pct",
]

CUTOFF_SWEEP_METRIC_FIELDNAMES = [
    "train_cutoff",
    "test_cutoff",
    *METRIC_FIELDNAMES,
]

CUTOFF_SWEEP_BEST_FIELDNAMES = [
    "target_type",
    "target_day_count",
    "model",
    "population",
    "interval_method",
    "snapshot_day",
    "best_train_cutoff",
    "test_cutoff",
    "train_n",
    "holdout_n",
    "bop_prediction_n",
    "fallback_prediction_n",
    "r2_log_revenue",
    "r2_gross",
    "mape_gross",
    "smape_gross",
    "accuracy_pct",
    "mse_log_revenue",
    "mse_gross",
    "mae_usd",
    "rmse_usd",
    "rmse_log_revenue",
    "mae_log_revenue",
    "mean_absolute_log_error",
    "directional_accuracy_vs_estimate",
    "raw_estimate_mae_usd",
    "raw_estimate_mae_lift_usd",
    "raw_estimate_mae_lift_pct",
    "actuals_multiplier_mae_usd",
    "actuals_multiplier_mae_lift_usd",
    "actuals_multiplier_mae_lift_pct",
    "mean_actual_gross",
    "mean_predicted_gross",
]

COMPETITIVE_SELECTION_METRIC_FIELDNAMES = [
    "split",
    "train_start_year",
    "train_end_year",
    "test_year",
    "train_cutoff",
    "test_cutoff",
    "switch_day",
    "model_used",
    "wiki_prediction_n",
    "wiki_competition_prediction_n",
    "population",
    "snapshot_day",
    "interval_method",
    "train_n",
    "holdout_n",
    "bop_prediction_n",
    "fallback_prediction_n",
    "r2_log_revenue",
    "r2_gross",
    "mape_gross",
    "smape_gross",
    "accuracy_pct",
    "mse_log_revenue",
    "mse_gross",
    "mae_usd",
    "rmse_usd",
    "rmse_log_revenue",
    "mae_log_revenue",
    "mean_absolute_log_error",
    "directional_accuracy_vs_estimate",
    "raw_estimate_mae_usd",
    "raw_estimate_mae_lift_usd",
    "raw_estimate_mae_lift_pct",
    "actuals_multiplier_mae_usd",
    "actuals_multiplier_mae_lift_usd",
    "actuals_multiplier_mae_lift_pct",
    "mean_actual_gross",
    "mean_predicted_gross",
    "mean_interval_80_width_pct",
    "coverage_50",
    "coverage_80",
    "coverage_90",
    "status",
    "reconciled",
]

COMPETITIVE_SELECTION_BEST_FIELDNAMES = [
    "rank",
    "selected",
    "train_cutoff",
    "test_cutoff",
    "switch_day",
    "valid_metric_rows",
    "valid_splits",
    "valid_snapshot_days",
    "holdout_n",
    "avg_mape_gross",
    "avg_rmse_log_revenue",
    "avg_r2_gross",
]

COMPETITIVE_MAPE_DELTA_FIELDNAMES = [
    "split",
    "train_start_year",
    "train_end_year",
    "test_year",
    "train_cutoff",
    "test_cutoff",
    "snapshot_day",
    "holdout_n",
    "wiki_mape_gross",
    "wiki_competition_mape_gross",
    "mape_delta_competition_minus_wiki",
    "competition_better",
    "reconciled",
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

BOP_ESTIMATE_ACCURACY_ROW_FIELDNAMES = [
    "prediction_id",
    "article_id",
    "article_url",
    "movie_id",
    "title",
    "release_year",
    "opening_date",
    "published_date",
    "lead_days",
    "lead_bucket",
    "primary_eligible",
    "source_context",
    "range_low_usd",
    "range_high_usd",
    "midpoint_usd",
    "actual_opening_weekend_revenue_usd",
    "signed_error_usd",
    "absolute_error_usd",
    "squared_error_usd",
    "absolute_percentage_error",
    "interval_hit",
]

BOP_ESTIMATE_ACCURACY_SUMMARY_FIELDNAMES = [
    "lead_bucket",
    "primary_eligible",
    "estimate_count",
    "movie_count",
    "mape",
    "accuracy_pct",
    "mae_usd",
    "mse_usd",
    "rmse_usd",
    "r2_gross",
    "r2_log_revenue",
    "pearson_correlation",
    "mean_signed_error_usd",
    "interval_hit_rate",
]

COMPETITIVE_CHECKPOINT_PREDICTION_FIELDNAMES = [
    "checkpoint",
    "snapshot_day",
    "model",
    "movie_id",
    "title",
    "release_year",
    "opening_date",
    "as_of_date",
    "bop_forecast_midpoint",
    "preview_gross_known_usd",
    "preview_day_count",
    "known_friday_gross_usd",
    "known_saturday_gross_usd",
    "known_weekend_gross_so_far_usd",
    "known_gross_so_far_pct_of_bop_midpoint",
    "competitive_residual_lag1",
    "competitive_residual_lag7",
    "competitive_residual_previous_weekend",
    "share_attraction_focal_share_lag7",
    "share_attraction_competitor_pressure_lag7",
    "share_attraction_top1_pressure_lag7",
    "logit_demand_focal_vs_competition_lag7",
    "logit_demand_focal_vs_top1_lag7",
    "log1p_logit_demand_choice_set_lag7",
    "actual_log_bop_residual",
    "predicted_log_bop_residual",
    "actual_log_opening_weekend",
    "predicted_log_opening_weekend",
    "actual_opening_weekend_revenue_usd",
    "predicted_opening_weekend_revenue_usd",
    "absolute_percentage_error",
]

COMPETITIVE_CHECKPOINT_METRIC_FIELDNAMES = [
    "checkpoint",
    "snapshot_day",
    "model",
    "train_start_year",
    "train_end_year",
    "test_start_year",
    "test_end_year",
    "train_n",
    "holdout_n",
    "r2_log_revenue",
    "r2_gross",
    "mape_gross",
    "accuracy_pct",
    "rmse_log_revenue",
    "mae_log_revenue",
    "mean_actual_gross",
    "mean_predicted_gross",
    "status",
]

COMPETITIVE_CHECKPOINT_COEFFICIENT_FIELDNAMES = [
    "checkpoint",
    "snapshot_day",
    "model",
    "target",
    "term",
    "standardized_coef",
    "center",
    "scale",
    "train_n",
]

COMPETITIVE_CHECKPOINT_HEADLINE_FIELDNAMES = [
    "checkpoint",
    "snapshot_day",
    "baseline_model",
    "competition_model",
    "holdout_n",
    "baseline_mape_gross",
    "competition_mape_gross",
    "mape_delta_competition_minus_baseline",
    "baseline_rmse_log_revenue",
    "competition_rmse_log_revenue",
    "rmse_log_delta_competition_minus_baseline",
    "baseline_r2_gross",
    "competition_r2_gross",
    "r2_gross_delta_competition_minus_baseline",
    "competition_better_mape",
    "status",
]

WIKI_BOP_TIMING_PREDICTION_FIELDNAMES = [
    "snapshot_day",
    "model",
    "population",
    "actuals_policy",
    "movie_id",
    "title",
    "release_year",
    "opening_date",
    "as_of_date",
    "bop_forecast_midpoint",
    "wiki_available",
    "V",
    "U",
    "R",
    "E",
    "log1p_V",
    "log1p_U",
    "log1p_R",
    "log1p_E",
    "log1p_opening_theaters",
    "known_actual_offsets",
    "known_gross_so_far",
    "base_predicted_opening_weekend_revenue_usd",
    "actual_log_bop_residual",
    "predicted_log_bop_residual",
    "actual_log_opening_weekend",
    "predicted_log_opening_weekend",
    "actual_opening_weekend_revenue_usd",
    "predicted_opening_weekend_revenue_usd",
    "absolute_percentage_error",
]

WIKI_BOP_TIMING_METRIC_FIELDNAMES = [
    "snapshot_day",
    "model",
    "population",
    "actuals_policy",
    "train_start_year",
    "train_end_year",
    "test_start_year",
    "test_end_year",
    "train_n",
    "holdout_n",
    "r2_log_revenue",
    "r2_gross",
    "mape_gross",
    "accuracy_pct",
    "rmse_log_revenue",
    "mae_log_revenue",
    "mean_actual_gross",
    "mean_predicted_gross",
    "status",
]

WIKI_BOP_TIMING_COEFFICIENT_FIELDNAMES = [
    "snapshot_day",
    "model",
    "target",
    "term",
    "standardized_coef",
    "center",
    "scale",
    "train_n",
]

WIKI_BOP_TIMING_HEADLINE_FIELDNAMES = [
    "snapshot_day",
    "baseline_model",
    "wiki_model",
    "actuals_policy",
    "holdout_n",
    "baseline_mape_gross",
    "wiki_mape_gross",
    "mape_delta_wiki_minus_baseline",
    "baseline_rmse_log_revenue",
    "wiki_rmse_log_revenue",
    "rmse_log_delta_wiki_minus_baseline",
    "baseline_r2_gross",
    "wiki_r2_gross",
    "r2_gross_delta_wiki_minus_baseline",
    "wiki_better_mape",
    "status",
]

WIKI_REMAINING_PREDICTION_FIELDNAMES = [
    "snapshot_day",
    "model",
    "population",
    "actuals_policy",
    "movie_id",
    "title",
    "release_year",
    "opening_date",
    "as_of_date",
    "bop_forecast_midpoint",
    "wiki_available",
    "log1p_V",
    "log1p_U",
    "log1p_R",
    "log1p_E",
    "known_actual_offsets",
    "known_gross_so_far",
    "actual_remaining_gross_usd",
    "baseline_remaining_gross_usd",
    "predicted_remaining_gross_usd",
    "actual_log_remaining_baseline_residual",
    "predicted_log_remaining_baseline_residual",
    "actual_log_opening_weekend",
    "predicted_log_opening_weekend",
    "actual_opening_weekend_revenue_usd",
    "predicted_opening_weekend_revenue_usd",
    "absolute_percentage_error",
    "remaining_absolute_percentage_error",
]

WIKI_REMAINING_METRIC_FIELDNAMES = [
    "snapshot_day",
    "model",
    "population",
    "actuals_policy",
    "train_start_year",
    "train_end_year",
    "test_start_year",
    "test_end_year",
    "train_n",
    "holdout_n",
    "r2_log_revenue",
    "r2_gross",
    "mape_gross",
    "mape_remaining_gross",
    "accuracy_pct",
    "rmse_log_revenue",
    "mae_log_revenue",
    "mean_actual_gross",
    "mean_predicted_gross",
    "mean_actual_remaining_gross",
    "mean_predicted_remaining_gross",
    "status",
]

WIKI_REMAINING_COEFFICIENT_FIELDNAMES = [
    "snapshot_day",
    "model",
    "target",
    "term",
    "standardized_coef",
    "center",
    "scale",
    "train_n",
]

WIKI_REMAINING_HEADLINE_FIELDNAMES = [
    "snapshot_day",
    "baseline_model",
    "wiki_model",
    "actuals_policy",
    "holdout_n",
    "baseline_mape_gross",
    "wiki_mape_gross",
    "mape_delta_wiki_minus_baseline",
    "baseline_mape_remaining_gross",
    "wiki_mape_remaining_gross",
    "mape_remaining_delta_wiki_minus_baseline",
    "baseline_rmse_log_revenue",
    "wiki_rmse_log_revenue",
    "rmse_log_delta_wiki_minus_baseline",
    "baseline_r2_gross",
    "wiki_r2_gross",
    "r2_gross_delta_wiki_minus_baseline",
    "wiki_better_mape",
    "status",
]

WIKI_COMP_COMBO_PREDICTION_FIELDNAMES = [
    "snapshot_day",
    "model",
    "model_family",
    "target_policy",
    "population",
    "actuals_policy",
    "movie_id",
    "title",
    "release_year",
    "opening_date",
    "as_of_date",
    "bop_forecast_midpoint",
    "wiki_available",
    "log1p_V",
    "log1p_U",
    "log1p_R",
    "log1p_E",
    "share_attraction_competitor_pressure_lag7",
    "logit_demand_focal_vs_competition_lag7",
    "known_actual_offsets",
    "known_gross_so_far",
    "actual_remaining_gross_usd",
    "baseline_remaining_gross_usd",
    "predicted_remaining_gross_usd",
    "baseline_display_prediction_usd",
    "actual_log_target_residual",
    "predicted_log_target_residual",
    "actual_log_opening_weekend",
    "predicted_log_opening_weekend",
    "actual_opening_weekend_revenue_usd",
    "predicted_opening_weekend_revenue_usd",
    "absolute_percentage_error",
    "remaining_absolute_percentage_error",
]

WIKI_COMP_COMBO_METRIC_FIELDNAMES = [
    "snapshot_day",
    "model",
    "model_family",
    "target_policy",
    "population",
    "actuals_policy",
    "train_start_year",
    "train_end_year",
    "test_start_year",
    "test_end_year",
    "train_n",
    "holdout_n",
    "r2_log_revenue",
    "r2_gross",
    "mape_gross",
    "mape_remaining_gross",
    "accuracy_pct",
    "rmse_log_revenue",
    "mae_log_revenue",
    "mean_actual_gross",
    "mean_predicted_gross",
    "baseline_model",
    "baseline_mape_gross",
    "mape_delta_vs_baseline",
    "baseline_rmse_log_revenue",
    "rmse_log_delta_vs_baseline",
    "baseline_r2_gross",
    "r2_gross_delta_vs_baseline",
    "best_single_model",
    "best_single_mape_gross",
    "mape_delta_vs_best_single",
    "status",
]

WIKI_COMP_COMBO_COEFFICIENT_FIELDNAMES = [
    "snapshot_day",
    "model",
    "model_family",
    "target_policy",
    "target",
    "term",
    "standardized_coef",
    "center",
    "scale",
    "train_n",
]

WIKI_COMP_COMBO_HEADLINE_FIELDNAMES = [
    "snapshot_day",
    "target_policy",
    "actuals_policy",
    "holdout_n",
    "baseline_model",
    "baseline_mape_gross",
    "views_mape_gross",
    "full_mape_gross",
    "views_beats_raw_bop",
    "full_beats_views",
    "best_comp_model",
    "best_comp_mape_gross",
    "competition_beats_raw_bop",
    "best_single_model",
    "best_single_mape_gross",
    "best_combo_model",
    "best_combo_mape_gross",
    "combo_beats_best_single",
    "combo_mape_delta_vs_best_single",
    "best_model",
    "best_model_family",
    "best_mape_gross",
    "best_mape_delta_vs_baseline",
    "status",
]

WIKI_COMP_IMPROVED_HEADLINE_FIELDNAMES = [
    "snapshot_day",
    "target_policy",
    "actuals_policy",
    "holdout_n",
    "best_improved_model",
    "best_improved_mape_gross",
    "status",
]

CONFIG_SEARCH_PREFIX_FIELDNAMES = [
    "search_phase",
    "train_policy",
    "validation_fold",
    "config_id",
    "config_train_start_year",
    "config_train_end_year",
    "config_test_start_year",
    "config_test_end_year",
    "train_bop_cutoff",
    "test_bop_cutoff",
]

CONFIG_SEARCH_METRIC_FIELDNAMES = CONFIG_SEARCH_PREFIX_FIELDNAMES + WIKI_COMP_COMBO_METRIC_FIELDNAMES
CONFIG_SEARCH_PREDICTION_FIELDNAMES = CONFIG_SEARCH_PREFIX_FIELDNAMES + WIKI_COMP_COMBO_PREDICTION_FIELDNAMES
CONFIG_SEARCH_BEST_FIELDNAMES = [
    "selection_scope",
    "train_policy",
    "config_id",
    "train_bop_cutoff",
    "test_bop_cutoff",
    "model",
    "model_family",
    "target_policy",
    "snapshot_day",
    "train_start_min",
    "train_end_max",
    "usable_folds",
    "validation_folds",
    "total_holdout_n",
    "min_train_n",
    "weighted_mape_gross",
    "median_mape_gross",
    "worst_fold_mape_gross",
    "mape_stdev",
    "weighted_rmse_log_revenue",
    "weighted_r2_gross",
    "low_sample_flag",
    "status",
]
CONFIG_SEARCH_FINAL_FIELDNAMES = [
    "selected_from",
    "selected_validation_weighted_mape_gross",
    *CONFIG_SEARCH_METRIC_FIELDNAMES,
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
    bop_estimate_rows: list[dict[str, object]] | None = None,
    bop_estimate_summary_rows: list[dict[str, object]] | None = None,
    cutoff_sweep_metric_rows: list[dict[str, object]] | None = None,
    competitive_selection_metric_rows: list[dict[str, object]] | None = None,
    competitive_mape_delta_rows: list[dict[str, object]] | None = None,
    competitive_checkpoint_prediction_rows: list[dict[str, object]] | None = None,
    competitive_checkpoint_metric_rows: list[dict[str, object]] | None = None,
    competitive_checkpoint_coefficient_rows: list[dict[str, object]] | None = None,
    competitive_checkpoint_headline_rows: list[dict[str, object]] | None = None,
    wiki_bop_timing_prediction_rows: list[dict[str, object]] | None = None,
    wiki_bop_timing_metric_rows: list[dict[str, object]] | None = None,
    wiki_bop_timing_coefficient_rows: list[dict[str, object]] | None = None,
    wiki_bop_timing_headline_rows: list[dict[str, object]] | None = None,
    wiki_remaining_prediction_rows: list[dict[str, object]] | None = None,
    wiki_remaining_metric_rows: list[dict[str, object]] | None = None,
    wiki_remaining_coefficient_rows: list[dict[str, object]] | None = None,
    wiki_remaining_headline_rows: list[dict[str, object]] | None = None,
    wiki_comp_combo_prediction_rows: list[dict[str, object]] | None = None,
    wiki_comp_combo_metric_rows: list[dict[str, object]] | None = None,
    wiki_comp_combo_coefficient_rows: list[dict[str, object]] | None = None,
    wiki_comp_combo_headline_rows: list[dict[str, object]] | None = None,
    wiki_comp_improved_prediction_rows: list[dict[str, object]] | None = None,
    wiki_comp_improved_metric_rows: list[dict[str, object]] | None = None,
    wiki_comp_improved_coefficient_rows: list[dict[str, object]] | None = None,
    wiki_comp_improved_headline_rows: list[dict[str, object]] | None = None,
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
    write_csv(
        out_dir / "day_by_day_bop_estimate_accuracy_rows.csv",
        bop_estimate_rows or [],
        BOP_ESTIMATE_ACCURACY_ROW_FIELDNAMES,
    )
    write_csv(
        out_dir / "day_by_day_bop_estimate_accuracy_by_lead_bucket.csv",
        bop_estimate_summary_rows or [],
        BOP_ESTIMATE_ACCURACY_SUMMARY_FIELDNAMES,
    )
    if cutoff_sweep_metric_rows is not None:
        write_csv(
            out_dir / "day_by_day_cutoff_sweep_metrics.csv",
            cutoff_sweep_metric_rows,
            CUTOFF_SWEEP_METRIC_FIELDNAMES,
        )
        write_csv(
            out_dir / "day_by_day_cutoff_sweep_best.csv",
            best_cutoff_sweep_rows(cutoff_sweep_metric_rows),
            CUTOFF_SWEEP_BEST_FIELDNAMES,
        )
    if competitive_selection_metric_rows is not None:
        write_csv(
            out_dir / "day_by_day_competitive_selection_metrics.csv",
            competitive_selection_metric_rows,
            COMPETITIVE_SELECTION_METRIC_FIELDNAMES,
        )
        write_csv(
            out_dir / "day_by_day_competitive_selection_best.csv",
            best_competitive_selection_rows(competitive_selection_metric_rows),
            COMPETITIVE_SELECTION_BEST_FIELDNAMES,
        )
    if competitive_mape_delta_rows is not None:
        write_csv(
            out_dir / "day_by_day_competitive_mape_delta_by_snapshot.csv",
            competitive_mape_delta_rows,
            COMPETITIVE_MAPE_DELTA_FIELDNAMES,
        )
    write_csv(
        out_dir / "competitive_checkpoint_predictions.csv",
        competitive_checkpoint_prediction_rows or [],
        COMPETITIVE_CHECKPOINT_PREDICTION_FIELDNAMES,
    )
    write_csv(
        out_dir / "competitive_checkpoint_metrics.csv",
        competitive_checkpoint_metric_rows or [],
        COMPETITIVE_CHECKPOINT_METRIC_FIELDNAMES,
    )
    write_csv(
        out_dir / "competitive_checkpoint_coefficients.csv",
        competitive_checkpoint_coefficient_rows or [],
        COMPETITIVE_CHECKPOINT_COEFFICIENT_FIELDNAMES,
    )
    write_csv(
        out_dir / "competitive_checkpoint_headline.csv",
        competitive_checkpoint_headline_rows or [],
        COMPETITIVE_CHECKPOINT_HEADLINE_FIELDNAMES,
    )
    write_csv(
        out_dir / "wiki_bop_timing_predictions.csv",
        wiki_bop_timing_prediction_rows or [],
        WIKI_BOP_TIMING_PREDICTION_FIELDNAMES,
    )
    write_csv(
        out_dir / "wiki_bop_timing_metrics.csv",
        wiki_bop_timing_metric_rows or [],
        WIKI_BOP_TIMING_METRIC_FIELDNAMES,
    )
    write_csv(
        out_dir / "wiki_bop_timing_coefficients.csv",
        wiki_bop_timing_coefficient_rows or [],
        WIKI_BOP_TIMING_COEFFICIENT_FIELDNAMES,
    )
    write_csv(
        out_dir / "wiki_bop_timing_headline.csv",
        wiki_bop_timing_headline_rows or [],
        WIKI_BOP_TIMING_HEADLINE_FIELDNAMES,
    )
    write_csv(
        out_dir / "wiki_remaining_predictions.csv",
        wiki_remaining_prediction_rows or [],
        WIKI_REMAINING_PREDICTION_FIELDNAMES,
    )
    write_csv(
        out_dir / "wiki_remaining_metrics.csv",
        wiki_remaining_metric_rows or [],
        WIKI_REMAINING_METRIC_FIELDNAMES,
    )
    write_csv(
        out_dir / "wiki_remaining_coefficients.csv",
        wiki_remaining_coefficient_rows or [],
        WIKI_REMAINING_COEFFICIENT_FIELDNAMES,
    )
    write_csv(
        out_dir / "wiki_remaining_headline.csv",
        wiki_remaining_headline_rows or [],
        WIKI_REMAINING_HEADLINE_FIELDNAMES,
    )
    write_csv(
        out_dir / "wiki_comp_combo_predictions.csv",
        wiki_comp_combo_prediction_rows or [],
        WIKI_COMP_COMBO_PREDICTION_FIELDNAMES,
    )
    write_csv(
        out_dir / "wiki_comp_combo_metrics.csv",
        wiki_comp_combo_metric_rows or [],
        WIKI_COMP_COMBO_METRIC_FIELDNAMES,
    )
    write_csv(
        out_dir / "wiki_comp_combo_coefficients.csv",
        wiki_comp_combo_coefficient_rows or [],
        WIKI_COMP_COMBO_COEFFICIENT_FIELDNAMES,
    )
    write_csv(
        out_dir / "wiki_comp_combo_headline.csv",
        wiki_comp_combo_headline_rows or [],
        WIKI_COMP_COMBO_HEADLINE_FIELDNAMES,
    )
    write_csv(
        out_dir / "wiki_comp_improved_predictions.csv",
        wiki_comp_improved_prediction_rows or [],
        WIKI_COMP_COMBO_PREDICTION_FIELDNAMES,
    )
    write_csv(
        out_dir / "wiki_comp_improved_metrics.csv",
        wiki_comp_improved_metric_rows or [],
        WIKI_COMP_COMBO_METRIC_FIELDNAMES,
    )
    write_csv(
        out_dir / "wiki_comp_improved_coefficients.csv",
        wiki_comp_improved_coefficient_rows or [],
        WIKI_COMP_COMBO_COEFFICIENT_FIELDNAMES,
    )
    write_csv(
        out_dir / "wiki_comp_improved_headline.csv",
        wiki_comp_improved_headline_rows or [],
        WIKI_COMP_IMPROVED_HEADLINE_FIELDNAMES,
    )
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
    target_types = parse_target_types(args.target_types)
    snapshot_days = parse_day_list(args.snapshot_days)
    try:
        train_cutoffs = parse_cutoff_list(args.train_opening_day_gross_cutoffs)
    except ValueError as exc:
        raise SystemExit(f"--train-opening-day-gross-cutoffs: {exc}") from exc
    if not snapshot_days:
        raise SystemExit("--snapshot-days must include at least one day")
    if args.train_start_year > args.train_end_year:
        raise SystemExit("--train-start-year must be <= --train-end-year")
    if args.test_start_year > args.test_end_year:
        raise SystemExit("--test-start-year must be <= --test-end-year")
    if args.min_opening_day_gross < 0:
        raise SystemExit("--min-opening-day-gross must be non-negative")
    if args.test_min_opening_day_gross < 0:
        raise SystemExit("--test-min-opening-day-gross must be non-negative")

    min_year = min(args.train_start_year, args.test_start_year)
    max_year = max(args.train_end_year, args.test_end_year)
    if args.run_config_search:
        min_year = min(min_year, min(CONFIG_SEARCH_TRAIN_START_POLICIES))
        max_year = max(max_year, CONFIG_SEARCH_FINAL_TEST_YEAR)
    load_min_opening_day_gross = (
        min(min(train_cutoffs), args.test_min_opening_day_gross)
        if args.run_cutoff_sweep
        else args.min_opening_day_gross
    )
    if args.run_config_search:
        load_min_opening_day_gross = min(load_min_opening_day_gross, 0)
    conn = connect_database(args.database_url)
    try:
        movies = opening.load_opening_weekend_movies(
            conn,
            min_year=min_year,
            max_year=max_year,
            min_opening_day_gross=load_min_opening_day_gross,
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
        preview_grosses = load_preview_grosses(
            conn,
            start_date=min_opening_date - dt.timedelta(days=14),
            end_date=max_opening_date,
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
        preview_grosses=preview_grosses,
        target_types=target_types,
    )
    if args.run_config_search and args.config_search_only:
        counts = run_config_search(panel_rows, out_dir=args.config_search_out)
        print(f"Built {len(panel_rows)} day-by-day forecast feature rows for {len(movies)} movies.")
        print(
            "Config search: "
            f"{counts['metrics']} metric rows, {counts['predictions']} prediction rows, "
            f"{counts['best_by_snapshot']} best-by-snapshot rows, "
            f"{counts['best_overall']} best-overall rows, "
            f"{counts['final_confirmation']} final confirmation rows."
        )
        print(f"Wrote config-search artifacts to {args.config_search_out}.")
        return 0
    if args.single_holdout:
        prediction_rows, metric_rows, coefficient_rows, interval_rows = evaluate_day_by_day_snapshots(
            panel_rows,
            snapshot_days=snapshot_days,
            train_start_year=args.train_start_year,
            train_end_year=args.train_end_year,
            test_start_year=args.test_start_year,
            test_end_year=args.test_end_year,
            target_types=target_types,
        )
    else:
        prediction_rows, metric_rows, coefficient_rows, interval_rows = evaluate_expanding_window_snapshots(
            panel_rows,
            snapshot_days=snapshot_days,
            train_start_year=args.train_start_year,
            test_start_year=args.test_start_year,
            test_end_year=args.test_end_year,
            target_types=target_types,
        )
    revision_rows = prediction_revision_rows(prediction_rows)
    coverage = coverage_rows(panel_rows)
    estimate_accuracy_rows = bop_estimate_accuracy_rows(movies, bop_forecasts)
    estimate_accuracy_summary_rows = bop_estimate_accuracy_summary_rows(estimate_accuracy_rows)
    cutoff_sweep_metric_rows = (
        evaluate_cutoff_sweep(
            panel_rows,
            snapshot_days=snapshot_days,
            train_start_year=args.train_start_year,
            train_end_year=args.train_end_year,
            test_start_year=args.test_start_year,
            test_end_year=args.test_end_year,
            train_opening_day_gross_cutoffs=train_cutoffs,
            test_min_opening_day_gross=args.test_min_opening_day_gross,
            target_types=target_types,
        )
        if args.run_cutoff_sweep
        else None
    )
    competitive_selection_metric_rows, competitive_mape_delta_rows = (
        evaluate_competitive_selection(
            panel_rows,
            snapshot_days=snapshot_days,
            train_opening_day_gross_cutoffs=train_cutoffs,
            test_min_opening_day_gross=args.test_min_opening_day_gross,
            switch_days=snapshot_days,
        )
        if args.run_competitive_selection
        else (None, None)
    )
    competitive_checkpoint_predictions, competitive_checkpoint_metrics, competitive_checkpoint_coefficients, competitive_checkpoint_headline = (
        evaluate_competitive_checkpoints(
            panel_rows,
            train_start_year=args.train_start_year,
            train_end_year=args.train_end_year,
            test_start_year=args.test_start_year,
            test_end_year=args.test_end_year,
            min_bop_forecast_midpoint=args.min_bop_forecast_midpoint,
        )
    )
    wiki_bop_timing_predictions, wiki_bop_timing_metrics, wiki_bop_timing_coefficients, wiki_bop_timing_headline = (
        evaluate_wiki_bop_timing(
            panel_rows,
            snapshot_days=[day for day in snapshot_days if day in WIKI_BOP_TIMING_DAYS],
            train_start_year=args.train_start_year,
            train_end_year=args.train_end_year,
            test_start_year=args.test_start_year,
            test_end_year=args.test_end_year,
            min_bop_forecast_midpoint=args.min_bop_forecast_midpoint,
        )
    )
    wiki_remaining_predictions, wiki_remaining_metrics, wiki_remaining_coefficients, wiki_remaining_headline = (
        evaluate_wiki_remaining_timing(
            panel_rows,
            snapshot_days=[day for day in snapshot_days if day in WIKI_REMAINING_TIMING_DAYS],
            train_start_year=args.train_start_year,
            train_end_year=args.train_end_year,
            test_start_year=args.test_start_year,
            test_end_year=args.test_end_year,
            min_bop_forecast_midpoint=args.min_bop_forecast_midpoint,
        )
    )
    wiki_comp_combo_predictions, wiki_comp_combo_metrics, wiki_comp_combo_coefficients, wiki_comp_combo_headline = (
        evaluate_wiki_comp_combo_timing(
            panel_rows,
            snapshot_days=[day for day in snapshot_days if day in WIKI_COMP_COMBO_DAYS],
            train_start_year=args.train_start_year,
            train_end_year=args.train_end_year,
            test_start_year=args.test_start_year,
            test_end_year=args.test_end_year,
            min_bop_forecast_midpoint=args.min_bop_forecast_midpoint,
        )
    )
    wiki_comp_improved_predictions, wiki_comp_improved_metrics, wiki_comp_improved_coefficients, wiki_comp_improved_headline = (
        evaluate_wiki_comp_improved_timing(
            panel_rows,
            snapshot_days=[day for day in snapshot_days if day in WIKI_COMP_COMBO_DAYS],
            train_start_year=args.train_start_year,
            train_end_year=args.train_end_year,
            test_start_year=args.test_start_year,
            test_end_year=args.test_end_year,
            min_bop_forecast_midpoint=args.min_bop_forecast_midpoint,
        )
    )
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
        bop_estimate_rows=estimate_accuracy_rows,
        bop_estimate_summary_rows=estimate_accuracy_summary_rows,
        cutoff_sweep_metric_rows=cutoff_sweep_metric_rows,
        competitive_selection_metric_rows=competitive_selection_metric_rows,
        competitive_mape_delta_rows=competitive_mape_delta_rows,
        competitive_checkpoint_prediction_rows=competitive_checkpoint_predictions,
        competitive_checkpoint_metric_rows=competitive_checkpoint_metrics,
        competitive_checkpoint_coefficient_rows=competitive_checkpoint_coefficients,
        competitive_checkpoint_headline_rows=competitive_checkpoint_headline,
        wiki_bop_timing_prediction_rows=wiki_bop_timing_predictions,
        wiki_bop_timing_metric_rows=wiki_bop_timing_metrics,
        wiki_bop_timing_coefficient_rows=wiki_bop_timing_coefficients,
        wiki_bop_timing_headline_rows=wiki_bop_timing_headline,
        wiki_remaining_prediction_rows=wiki_remaining_predictions,
        wiki_remaining_metric_rows=wiki_remaining_metrics,
        wiki_remaining_coefficient_rows=wiki_remaining_coefficients,
        wiki_remaining_headline_rows=wiki_remaining_headline,
        wiki_comp_combo_prediction_rows=wiki_comp_combo_predictions,
        wiki_comp_combo_metric_rows=wiki_comp_combo_metrics,
        wiki_comp_combo_coefficient_rows=wiki_comp_combo_coefficients,
        wiki_comp_combo_headline_rows=wiki_comp_combo_headline,
        wiki_comp_improved_prediction_rows=wiki_comp_improved_predictions,
        wiki_comp_improved_metric_rows=wiki_comp_improved_metrics,
        wiki_comp_improved_coefficient_rows=wiki_comp_improved_coefficients,
        wiki_comp_improved_headline_rows=wiki_comp_improved_headline,
    )
    if args.run_config_search:
        counts = run_config_search(panel_rows, out_dir=args.config_search_out)
        print(
            "Config search: "
            f"{counts['metrics']} metric rows, {counts['predictions']} prediction rows, "
            f"{counts['best_by_snapshot']} best-by-snapshot rows, "
            f"{counts['best_overall']} best-overall rows, "
            f"{counts['final_confirmation']} final confirmation rows."
        )
    print(f"Built {len(panel_rows)} day-by-day forecast feature rows for {len(movies)} movies.")
    print(f"Snapshot days: {','.join(str(day) for day in snapshot_days)}.")
    print(f"Train years: {args.train_start_year}-{args.train_end_year}; test years: {args.test_start_year}-{args.test_end_year}.")
    print(f"Cohort filter: opening day >= ${args.min_opening_day_gross:,} (retrospective, not production-safe).")
    if args.run_cutoff_sweep:
        print(
            "Cutoff sweep: train opening-day cutoffs "
            f"{','.join(str(cutoff) for cutoff in train_cutoffs)}; "
            f"test opening day >= ${args.test_min_opening_day_gross:,}."
        )
    if args.run_competitive_selection:
        print(
            "Competitive selection: switch days "
            f"{','.join(str(day) for day in snapshot_days)}; "
            f"train opening-day cutoffs {','.join(str(cutoff) for cutoff in train_cutoffs)}."
        )
    print(f"Target types: {','.join(target_types)}.")
    print(
        "Competitive checkpoints: "
        f"train years {args.train_start_year}-{args.train_end_year}; "
        f"test years {args.test_start_year}-{args.test_end_year}; "
        "snapshot days -1,1,2; "
        f"BOP midpoint >= ${args.min_bop_forecast_midpoint:,}."
    )
    print(
        "Wiki+BOP timing: "
        f"3-day target; train years {args.train_start_year}-{args.train_end_year}; "
        f"test years {args.test_start_year}-{args.test_end_year}; "
        f"snapshot days {','.join(str(day) for day in snapshot_days if day in WIKI_BOP_TIMING_DAYS)}; "
        f"BOP midpoint >= ${args.min_bop_forecast_midpoint:,}."
    )
    print(
        "Wiki remaining timing: "
        f"snapshot days {','.join(str(day) for day in snapshot_days if day in WIKI_REMAINING_TIMING_DAYS)}; "
        "target is remaining weekend residual around train-only remainder ratios."
    )
    print(
        "Wiki+competition combo timing: "
        f"snapshot days {','.join(str(day) for day in snapshot_days if day in WIKI_COMP_COMBO_DAYS)}; "
        "total-weekend BOP residual through t=0, remaining-weekend residual at t=1/t=2."
    )
    print(
        "Improved Wiki+competition combo timing: stacked, gated, residualized-competition, "
        f"and ridge variants; ridge alpha={WIKI_COMP_RIDGE_LAMBDA:g}."
    )
    print(f"Wrote day-by-day opening-weekend artifacts to {out_dir}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day-by-day opening-weekend forecasts anchored on BOP estimates.")
    parser.add_argument("--database-url", default=database_url_from_env())
    parser.add_argument("--out", type=Path)
    parser.add_argument("--snapshot-days", default=",".join(str(day) for day in DEFAULT_SNAPSHOT_DAYS))
    parser.add_argument("--target-types", default=",".join(DEFAULT_TARGET_TYPES))
    parser.add_argument("--train-start-year", type=int, default=DEFAULT_TRAIN_START_YEAR)
    parser.add_argument("--train-end-year", type=int, default=DEFAULT_TRAIN_END_YEAR)
    parser.add_argument("--test-start-year", type=int, default=DEFAULT_TEST_START_YEAR)
    parser.add_argument("--test-end-year", type=int, default=DEFAULT_TEST_END_YEAR)
    parser.add_argument("--min-opening-day-gross", type=int, default=DEFAULT_MIN_OPENING_DAY_GROSS)
    parser.add_argument("--test-min-opening-day-gross", type=int, default=DEFAULT_TEST_MIN_OPENING_DAY_GROSS)
    parser.add_argument("--min-bop-forecast-midpoint", type=int, default=DEFAULT_MIN_BOP_FORECAST_MIDPOINT)
    parser.add_argument(
        "--train-opening-day-gross-cutoffs",
        default=",".join(str(cutoff) for cutoff in DEFAULT_TRAIN_OPENING_DAY_GROSS_CUTOFFS),
    )
    parser.add_argument("--run-cutoff-sweep", action="store_true")
    parser.add_argument("--run-competitive-selection", action="store_true")
    parser.add_argument("--run-config-search", action="store_true")
    parser.add_argument("--config-search-only", action="store_true")
    parser.add_argument("--config-search-out", type=Path, default=DEFAULT_CONFIG_SEARCH_OUT_DIR)
    parser.add_argument("--single-holdout", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
