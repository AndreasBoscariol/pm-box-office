#!/usr/bin/env python3
"""Opening-weekend regression with actual prior-day competition dynamics.

This analysis mirrors the fresh Wikipedia opening-weekend validation, but asks
one narrower question: whether actual grosses from other movies in the days
before opening improve the Wikipedia baseline.  It intentionally avoids
Boxoffice Pro forecasts and other forward-looking estimates for the clean
actual-competition artifact family.  A separate Boxoffice Pro artifact family
tests forecast-based size proxies and calibration when those estimates are
available in the database.
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
from pm_box_office.research.papers.recreate_wikipedia_boxoffice import (
    format_number,
    mean,
    parse_db_date,
    r2_score,
    solve_linear_system,
    write_csv,
)


DEFAULT_OUT_DIR = Path("results/papers/competition_opening_weekend")
DEFAULT_TIMING_DAYS = (-30, -14, -7, -1)
DEFAULT_TRAIN_START_YEAR = 2022
DEFAULT_TRAIN_END_YEAR = 2024
DEFAULT_TEST_START_YEAR = 2025
DEFAULT_TEST_END_YEAR = 2026
DEFAULT_MIN_OPENING_DAY_GROSS = 5_000_000
DEFAULT_BOP_TIMING_DAYS = (-7, -3, -2, -1)


WIKI_TERMS = ["log1p_V", "log1p_U", "log1p_R", "log1p_E", "log1p_opening_theaters"]

MODEL_TERMS = {
    "wiki_log_all": WIKI_TERMS,
    "wiki_plus_competitor_total_lag1": WIKI_TERMS + ["log1p_competitor_total_gross_lag1"],
    "wiki_plus_competitor_top1_lag1": WIKI_TERMS + ["log1p_competitor_top1_gross_lag1"],
    "wiki_plus_compact_competition": WIKI_TERMS
    + [
        "log1p_competitor_total_gross_lag1",
        "log1p_competitor_top1_gross_lag1",
        "competitor_count_lag1",
        "competitor_hhi_lag1",
    ],
    "wiki_plus_rolling_competition": WIKI_TERMS
    + [
        "log1p_competitor_total_gross_lag1",
        "log1p_competitor_total_gross_lag3",
        "log1p_competitor_total_gross_lag7",
    ],
    "wiki_plus_competitor_previous_weekend": WIKI_TERMS
    + ["log1p_competitor_total_gross_previous_weekend"],
}

BOP_Q4_INTERACTION_TERMS = [
    "bop_q4_proxy_x_log1p_V",
    "bop_q4_proxy_x_log1p_U",
    "bop_q4_proxy_x_log1p_R",
    "bop_q4_proxy_x_log1p_E",
    "bop_q4_proxy_x_log1p_opening_theaters",
]

BOP_Q4_MODEL_TERMS = {
    "wiki_log_all": WIKI_TERMS,
    "wiki_plus_bop_q4_interactions": WIKI_TERMS + ["bop_q4_proxy"] + BOP_Q4_INTERACTION_TERMS,
    "wiki_plus_competition": WIKI_TERMS + ["log1p_competitor_total_gross_lag1"],
    "wiki_plus_competition_bop_q4_interactions": WIKI_TERMS
    + ["log1p_competitor_total_gross_lag1", "bop_q4_proxy"]
    + BOP_Q4_INTERACTION_TERMS
    + ["bop_q4_proxy_x_log1p_competitor_total_gross_lag1"],
}

BOP_CALIBRATION_MODEL_TERMS = {
    "calibrated_midpoint": ["log1p_bop_forecast_midpoint"],
    "midpoint_plus_range_width": ["log1p_bop_forecast_midpoint", "bop_forecast_range_width_pct"],
    "midpoint_plus_rank_showtime": [
        "log1p_bop_forecast_midpoint",
        "bop_source_rank",
        "bop_showtime_market_share_pct",
    ],
    "midpoint_plus_wiki": ["log1p_bop_forecast_midpoint"] + WIKI_TERMS,
}

MULTISOURCE_MODEL_TERMS = {
    "calibrated_bop": ["log1p_bop_forecast_midpoint"],
    "bucket_calibrated_bop": [
        "log1p_bop_forecast_midpoint",
        "bop_q1_proxy",
        "bop_q4_proxy",
    ],
    "bop_plus_wiki": [
        "log1p_bop_forecast_midpoint",
        "log1p_opening_theaters",
        "log1p_V",
        "log1p_U",
    ],
    "bop_plus_competition": [
        "log1p_bop_forecast_midpoint",
        "log1p_opening_theaters",
        "log1p_competitor_total_gross_lag7",
    ],
    "bop_plus_wiki_competition": [
        "log1p_bop_forecast_midpoint",
        "log1p_opening_theaters",
        "log1p_V",
        "log1p_U",
        "log1p_competitor_total_gross_lag7",
        "bop_forecast_range_width_pct",
    ],
    "bop_plus_wiki_competition_buckets": [
        "log1p_bop_forecast_midpoint",
        "log1p_opening_theaters",
        "log1p_V",
        "log1p_U",
        "log1p_competitor_total_gross_lag7",
        "bop_forecast_range_width_pct",
        "bop_q1_proxy",
        "bop_q4_proxy",
        "bop_q1_proxy_x_log1p_bop_forecast_midpoint",
        "bop_q4_proxy_x_log1p_bop_forecast_midpoint",
        "bop_q4_proxy_x_log1p_V",
        "bop_q4_proxy_x_log1p_competitor_total_gross_lag7",
    ],
}

FALLBACK_MODEL_NAME = "fallback_wiki_rolling_competition"
FALLBACK_MODEL_TERMS = WIKI_TERMS + [
    "log1p_competitor_total_gross_lag1",
    "log1p_competitor_total_gross_lag3",
    "log1p_competitor_total_gross_lag7",
]

RESIDUAL_LIFT_MODEL_TERMS = {
    "residual_intercept": [],
    "residual_buckets": ["bop_q1_proxy", "bop_q4_proxy"],
    "residual_quartile_buckets": ["bop_q1_proxy", "bop_q3_proxy", "bop_q4_proxy"],
    "residual_fixed_estimate_buckets": [
        "bop_estimate_bucket_15_30m",
        "bop_estimate_bucket_30_60m",
        "bop_estimate_bucket_60_100m",
        "bop_estimate_bucket_100m_plus",
    ],
    "residual_wiki": ["bop_q1_proxy", "bop_q4_proxy", "log1p_V"],
    "residual_competition": ["bop_q1_proxy", "bop_q4_proxy", "log1p_competitor_total_gross_lag7"],
    "residual_quartile_competition": [
        "bop_q1_proxy",
        "bop_q3_proxy",
        "bop_q4_proxy",
        "log1p_competitor_total_gross_lag7",
    ],
    "residual_compact_combined": [
        "bop_q1_proxy",
        "bop_q4_proxy",
        "log1p_V",
        "log1p_competitor_total_gross_lag7",
    ],
}


@dataclass(frozen=True)
class OpeningWeekendMovie:
    movie_id: int
    title: str
    release_year: int
    release_run_id: int
    opening_date: dt.date
    opening_theaters: int
    opening_day_gross_usd: int
    opening_weekend_revenue_usd: int


@dataclass(frozen=True)
class DailyGross:
    movie_id: int
    box_office_date: dt.date
    gross_usd: int
    theaters: int


@dataclass(frozen=True)
class BoxofficeProForecast:
    prediction_id: int
    movie_id: int
    article_id: int
    article_url: str
    source_movie_title: str
    forecast_metric: str
    source_context: str
    source_rank: int | None
    target_start_date: dt.date
    target_end_date: dt.date
    range_low_usd: float
    range_high_usd: float
    showtime_market_share_pct: float | None
    published_date: dt.date

    @property
    def midpoint_usd(self) -> float:
        return (self.range_low_usd + self.range_high_usd) / 2.0


@dataclass(frozen=True)
class FittedModel:
    terms: list[str]
    centers: list[float]
    scales: list[float]
    beta: list[float]


def parse_day_list(value: str) -> list[int]:
    return sorted({int(part.strip()) for part in value.split(",") if part.strip()})


def log1p(value: object) -> float:
    return math.log1p(max(0.0, float(value or 0.0)))


def load_opening_weekend_movies(
    conn: Any,
    *,
    min_year: int,
    max_year: int,
    min_opening_day_gross: int,
) -> list[OpeningWeekendMovie]:
    rows = conn.execute(
        """
        WITH opening AS (
            SELECT
                rr.release_run_id,
                rr.movie_id,
                MIN(dbo.box_office_date::date) AS opening_date
            FROM release_runs rr
            JOIN daily_box_office dbo ON dbo.release_run_id = rr.release_run_id
            WHERE dbo.is_preview = 0
              AND dbo.gross_usd IS NOT NULL
            GROUP BY rr.release_run_id, rr.movie_id
        ),
        opening_features AS (
            SELECT
                opening.release_run_id,
                opening.movie_id,
                opening.opening_date,
                opening_day.theaters AS opening_theaters,
                opening_day.gross_usd AS opening_day_gross_usd,
                SUM(weekend.gross_usd) AS opening_weekend_revenue_usd
            FROM opening
            JOIN daily_box_office opening_day
              ON opening_day.release_run_id = opening.release_run_id
             AND opening_day.box_office_date::date = opening.opening_date
            JOIN daily_box_office weekend
              ON weekend.release_run_id = opening.release_run_id
             AND weekend.is_preview = 0
             AND weekend.box_office_date::date >= opening.opening_date
             AND weekend.box_office_date::date < opening.opening_date + INTERVAL '3 days'
            GROUP BY
                opening.release_run_id,
                opening.movie_id,
                opening.opening_date,
                opening_day.theaters,
                opening_day.gross_usd
        )
        SELECT
            m.movie_id,
            m.title,
            COALESCE(m.release_year, EXTRACT(YEAR FROM ofe.opening_date)::integer) AS release_year,
            ofe.release_run_id,
            ofe.opening_date,
            ofe.opening_theaters,
            ofe.opening_day_gross_usd,
            ofe.opening_weekend_revenue_usd
        FROM opening_features ofe
        JOIN movies m ON m.movie_id = ofe.movie_id
        WHERE COALESCE(m.release_year, EXTRACT(YEAR FROM ofe.opening_date)::integer) BETWEEN %s AND %s
          AND ofe.opening_day_gross_usd >= %s
          AND ofe.opening_weekend_revenue_usd > 0
          AND ofe.opening_theaters IS NOT NULL
        ORDER BY ofe.opening_date, m.title
        """,
        (min_year, max_year, min_opening_day_gross),
    ).fetchall()
    return [
        OpeningWeekendMovie(
            movie_id=int(row[0]),
            title=str(row[1]),
            release_year=int(row[2]),
            release_run_id=int(row[3]),
            opening_date=parse_db_date(row[4]),
            opening_theaters=int(row[5]),
            opening_day_gross_usd=int(row[6]),
            opening_weekend_revenue_usd=int(row[7]),
        )
        for row in rows
    ]


def load_daily_grosses(conn: Any, *, start_date: dt.date, end_date: dt.date) -> list[DailyGross]:
    rows = conn.execute(
        """
        SELECT
            rr.movie_id,
            dbo.box_office_date::date,
            SUM(dbo.gross_usd) AS gross_usd,
            MAX(dbo.theaters) AS theaters
        FROM daily_box_office dbo
        JOIN release_runs rr ON rr.release_run_id = dbo.release_run_id
        WHERE dbo.is_preview = 0
          AND dbo.is_estimate = 0
          AND dbo.gross_usd IS NOT NULL
          AND dbo.gross_usd > 0
          AND dbo.box_office_date::date BETWEEN %s AND %s
        GROUP BY rr.movie_id, dbo.box_office_date::date
        ORDER BY dbo.box_office_date::date, rr.movie_id
        """,
        (start_date.isoformat(), end_date.isoformat()),
    ).fetchall()
    return [
        DailyGross(
            movie_id=int(row[0]),
            box_office_date=parse_db_date(row[1]),
            gross_usd=int(row[2]),
            theaters=int(row[3] or 0),
        )
        for row in rows
    ]


def load_boxofficepro_forecasts(
    conn: Any,
    *,
    min_target_date: dt.date,
    max_target_date: dt.date,
) -> list[BoxofficeProForecast]:
    rows = conn.execute(
        """
        SELECT
            p.prediction_id,
            p.matched_movie_id,
            p.article_id,
            a.article_url,
            p.source_movie_title,
            p.forecast_metric,
            p.source_context,
            p.source_rank,
            p.target_start_date::date,
            p.target_end_date::date,
            p.range_low_usd,
            p.range_high_usd,
            p.showtime_market_share_pct,
            a.discovered_date::date AS published_date
        FROM boxofficepro_weekend_predictions p
        JOIN boxofficepro_articles a ON a.article_id = p.article_id
        WHERE p.matched_movie_id IS NOT NULL
          AND p.target_start_date IS NOT NULL
          AND p.target_end_date IS NOT NULL
          AND a.discovered_date IS NOT NULL
          AND p.range_low_usd > 0
          AND p.range_high_usd > 0
          AND p.target_start_date::date BETWEEN %s AND %s
          AND p.forecast_metric IN ('domestic_opening_weekend', 'domestic_weekend')
        ORDER BY p.target_start_date::date, a.discovered_date::date, p.article_id, p.source_rank NULLS LAST
        """,
        (min_target_date.isoformat(), max_target_date.isoformat()),
    ).fetchall()
    return [
        BoxofficeProForecast(
            prediction_id=int(row[0]),
            movie_id=int(row[1]),
            article_id=int(row[2]),
            article_url=str(row[3]),
            source_movie_title=str(row[4]),
            forecast_metric=str(row[5]),
            source_context=str(row[6]),
            source_rank=int(row[7]) if row[7] is not None else None,
            target_start_date=parse_db_date(row[8]),
            target_end_date=parse_db_date(row[9]),
            range_low_usd=float(row[10]),
            range_high_usd=float(row[11]),
            showtime_market_share_pct=float(row[12]) if row[12] is not None else None,
            published_date=parse_db_date(row[13]),
        )
        for row in rows
    ]


def load_wiki_feature_map(
    conn: Any,
    *,
    movies: list[OpeningWeekendMovie],
    timing_days: list[int],
    language: str = "en",
) -> dict[int, dict[int, dict[str, float]]]:
    if not movies:
        return {}
    movies_by_id = {movie.movie_id: movie for movie in movies}
    try:
        match_rows = conn.execute(
            """
            SELECT movie_id, language, wiki_page_id
            FROM movie_wiki_pages
            WHERE movie_id = ANY(%s)
              AND language = %s
              AND match_status IN ('matched', 'manual_override')
              AND wiki_page_id IS NOT NULL
            ORDER BY movie_id, language, wiki_page_id
            """,
            (list(movies_by_id), language),
        ).fetchall()
    except Exception:
        return {}

    selected_pages: dict[int, tuple[str, int]] = {}
    for row in match_rows:
        selected_pages.setdefault(int(row[0]), (str(row[1]), int(row[2])))

    by_movie: dict[int, dict[int, dict[str, float]]] = {}
    sorted_days = sorted(timing_days)
    for movie_id, (page_language, wiki_page_id) in selected_pages.items():
        movie = movies_by_id[movie_id]
        max_as_of = movie.opening_date + dt.timedelta(days=max(sorted_days))
        pageview_rows = conn.execute(
            """
            SELECT view_date::date, SUM(views)
            FROM wiki_pageviews_daily
            WHERE language = %s
              AND wiki_page_id = %s
              AND agent = 'user'
              AND view_date::date <= %s
            GROUP BY view_date::date
            ORDER BY view_date::date
            """,
            (page_language, wiki_page_id, max_as_of.isoformat()),
        ).fetchall()
        views_by_date = {parse_db_date(row[0]): float(row[1] or 0.0) for row in pageview_rows}
        revision_rows = conn.execute(
            """
            SELECT rev_date::date, rev_timestamp, rev_id, user_key
            FROM wiki_revisions
            WHERE language = %s
              AND wiki_page_id = %s
              AND is_bot = 0
              AND rev_date::date <= %s
            ORDER BY rev_timestamp, rev_id
            """,
            (page_language, wiki_page_id, max_as_of.isoformat()),
        ).fetchall()
        revisions = [
            (parse_db_date(row[0]), str(row[3]))
            for row in revision_rows
        ]

        movie_values: dict[int, dict[str, float]] = {}
        for timing_day in sorted_days:
            as_of = movie.opening_date + dt.timedelta(days=timing_day)
            views = sum(value for day, value in views_by_date.items() if day <= as_of)
            edits = 0.0
            rigor = 0.0
            users: set[str] = set()
            previous_user: str | None = None
            for rev_date, user_key in revisions:
                if rev_date > as_of:
                    break
                edits += 1.0
                users.add(user_key)
                if previous_user is None or previous_user != user_key:
                    rigor += 1.0
                previous_user = user_key
            movie_values[timing_day] = {
                "V": views,
                "U": float(len(users)),
                "R": rigor,
                "E": edits,
            }
        by_movie[movie_id] = movie_values
    return dict(by_movie)


def wiki_values_as_of(
    wiki_by_movie: dict[int, dict[int, dict[str, float]]],
    *,
    movie_id: int,
    timing_day: int,
) -> dict[str, float]:
    values = wiki_by_movie.get(movie_id, {})
    eligible_days = [day for day in values if day <= timing_day]
    if not eligible_days:
        return {"V": 0.0, "U": 0.0, "R": 0.0, "E": 0.0}
    return values[max(eligible_days)]


def aggregate_grosses(values: Iterable[float], *, focal_proxy_gross: float = 0.0) -> dict[str, float]:
    grosses = sorted([float(value) for value in values if value and value > 0.0], reverse=True)
    total = sum(grosses)
    top1 = grosses[0] if grosses else 0.0
    top3 = sum(grosses[:3])
    count = float(len(grosses))
    hhi = sum((gross / total) ** 2 for gross in grosses) if total > 0.0 else 0.0
    share = total / (total + focal_proxy_gross) if total + focal_proxy_gross > 0.0 else 0.0
    return {
        "total": total,
        "top1": top1,
        "top3": top3,
        "count": count,
        "hhi": hhi,
        "share_ex_focal": share,
    }


def actual_competitor_lag_features(
    gross_by_movie_date: dict[tuple[int, dt.date], DailyGross],
    *,
    focal_movie_id: int,
    as_of_date: dt.date,
    lag_days: int,
) -> dict[str, float]:
    start_date = as_of_date - dt.timedelta(days=lag_days - 1)
    gross_by_competitor: dict[int, float] = defaultdict(float)
    for (movie_id, day), row in gross_by_movie_date.items():
        if movie_id == focal_movie_id:
            continue
        if start_date <= day <= as_of_date and row.gross_usd > 0:
            gross_by_competitor[movie_id] += float(row.gross_usd)
    stats = aggregate_grosses(gross_by_competitor.values())
    return {
        f"competitor_total_gross_lag{lag_days}": stats["total"],
        f"competitor_top1_gross_lag{lag_days}": stats["top1"],
        f"competitor_count_lag{lag_days}": stats["count"],
        f"competitor_hhi_lag{lag_days}": stats["hhi"],
    }


def previous_weekend_window(as_of_date: dt.date) -> tuple[dt.date, dt.date]:
    days_since_sunday = (as_of_date.weekday() - 6) % 7
    end_date = as_of_date - dt.timedelta(days=days_since_sunday)
    return end_date - dt.timedelta(days=2), end_date


def actual_competitor_window_features(
    gross_by_movie_date: dict[tuple[int, dt.date], DailyGross],
    *,
    focal_movie_id: int,
    start_date: dt.date,
    end_date: dt.date,
    label: str,
) -> dict[str, float]:
    gross_by_competitor: dict[int, float] = defaultdict(float)
    for (movie_id, day), row in gross_by_movie_date.items():
        if movie_id == focal_movie_id:
            continue
        if start_date <= day <= end_date and row.gross_usd > 0:
            gross_by_competitor[movie_id] += float(row.gross_usd)
    stats = aggregate_grosses(gross_by_competitor.values())
    return {
        f"competitor_total_gross_{label}": stats["total"],
        f"competitor_top1_gross_{label}": stats["top1"],
        f"competitor_count_{label}": stats["count"],
        f"competitor_hhi_{label}": stats["hhi"],
        f"log1p_competitor_total_gross_{label}": log1p(stats["total"]),
        f"log1p_competitor_top1_gross_{label}": log1p(stats["top1"]),
    }


def actual_competition_features(
    gross_by_movie_date: dict[tuple[int, dt.date], DailyGross],
    *,
    focal_movie_id: int,
    as_of_date: dt.date,
) -> dict[str, float]:
    features: dict[str, float] = {}
    for lag_days in (1, 3, 7):
        lag_features = actual_competitor_lag_features(
            gross_by_movie_date,
            focal_movie_id=focal_movie_id,
            as_of_date=as_of_date,
            lag_days=lag_days,
        )
        features.update(lag_features)
        features[f"log1p_competitor_total_gross_lag{lag_days}"] = log1p(
            lag_features[f"competitor_total_gross_lag{lag_days}"]
        )
        features[f"log1p_competitor_top1_gross_lag{lag_days}"] = log1p(
            lag_features[f"competitor_top1_gross_lag{lag_days}"]
        )
    weekend_start, weekend_end = previous_weekend_window(as_of_date)
    features.update(
        actual_competitor_window_features(
            gross_by_movie_date,
            focal_movie_id=focal_movie_id,
            start_date=weekend_start,
            end_date=weekend_end,
            label="previous_weekend",
        )
    )
    return {
        **features,
    }


def latest_forecast(
    forecasts: Iterable[BoxofficeProForecast],
    *,
    as_of_date: dt.date,
    movie_id: int | None = None,
    forecast_metric: str | None = None,
    target_start_date: dt.date | None = None,
) -> BoxofficeProForecast | None:
    eligible = [
        forecast
        for forecast in forecasts
        if forecast.published_date <= as_of_date
        and (movie_id is None or forecast.movie_id == movie_id)
        and (forecast_metric is None or forecast.forecast_metric == forecast_metric)
        and (target_start_date is None or forecast.target_start_date == target_start_date)
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda forecast: (
            forecast.published_date,
            -(forecast.source_rank or 999999),
            forecast.article_id,
            forecast.prediction_id,
        ),
    )


def bop_forecast_features(forecast: BoxofficeProForecast | None) -> dict[str, object]:
    if forecast is None:
        return {
            "bop_forecast_available": 0.0,
            "bop_prediction_id": "",
            "bop_article_url": "",
            "bop_forecast_published_date": "",
            "bop_forecast_midpoint": 0.0,
            "log1p_bop_forecast_midpoint": 0.0,
            "bop_forecast_range_width_pct": 0.0,
            "bop_source_rank": 0.0,
            "bop_showtime_market_share_pct": 0.0,
        }
    midpoint = forecast.midpoint_usd
    width_pct = (forecast.range_high_usd - forecast.range_low_usd) / midpoint if midpoint > 0.0 else 0.0
    return {
        "bop_forecast_available": 1.0,
        "bop_prediction_id": forecast.prediction_id,
        "bop_article_url": forecast.article_url,
        "bop_forecast_published_date": forecast.published_date.isoformat(),
        "bop_forecast_midpoint": midpoint,
        "log1p_bop_forecast_midpoint": log1p(midpoint),
        "bop_forecast_range_width_pct": width_pct,
        "bop_source_rank": float(forecast.source_rank or 0.0),
        "bop_showtime_market_share_pct": float(forecast.showtime_market_share_pct or 0.0),
    }


def bop_same_weekend_competitor_features(
    forecasts: list[BoxofficeProForecast],
    *,
    focal_movie_id: int,
    target_start_date: dt.date,
    as_of_date: dt.date,
) -> dict[str, float]:
    latest_by_movie: dict[int, BoxofficeProForecast] = {}
    for forecast in forecasts:
        if forecast.movie_id == focal_movie_id:
            continue
        if forecast.target_start_date != target_start_date or forecast.published_date > as_of_date:
            continue
        current = latest_by_movie.get(forecast.movie_id)
        if current is None or (
            forecast.published_date,
            -(forecast.source_rank or 999999),
            forecast.article_id,
            forecast.prediction_id,
        ) > (
            current.published_date,
            -(current.source_rank or 999999),
            current.article_id,
            current.prediction_id,
        ):
            latest_by_movie[forecast.movie_id] = forecast
    stats = aggregate_grosses(forecast.midpoint_usd for forecast in latest_by_movie.values())
    return {
        "bop_same_weekend_competitor_total": stats["total"],
        "bop_same_weekend_competitor_top1": stats["top1"],
        "bop_same_weekend_competitor_count": stats["count"],
        "bop_same_weekend_competitor_hhi": stats["hhi"],
        "log1p_bop_same_weekend_competitor_total": log1p(stats["total"]),
        "log1p_bop_same_weekend_competitor_top1": log1p(stats["top1"]),
    }


def add_bop_interactions(row: dict[str, object]) -> None:
    q1 = float(row.get("bop_q1_proxy", 0.0) or 0.0)
    q4 = float(row.get("bop_q4_proxy", 0.0) or 0.0)
    for term in WIKI_TERMS:
        row[f"bop_q4_proxy_x_{term}"] = q4 * float(row.get(term, 0.0) or 0.0)
    row["bop_q1_proxy_x_log1p_bop_forecast_midpoint"] = q1 * float(
        row.get("log1p_bop_forecast_midpoint", 0.0) or 0.0
    )
    row["bop_q4_proxy_x_log1p_bop_forecast_midpoint"] = q4 * float(
        row.get("log1p_bop_forecast_midpoint", 0.0) or 0.0
    )
    row["bop_q4_proxy_x_log1p_competitor_total_gross_lag1"] = q4 * float(
        row.get("log1p_competitor_total_gross_lag1", 0.0) or 0.0
    )
    row["bop_q4_proxy_x_log1p_competitor_total_gross_lag7"] = q4 * float(
        row.get("log1p_competitor_total_gross_lag7", 0.0) or 0.0
    )


def add_bop_fixed_estimate_buckets(row: dict[str, object]) -> None:
    midpoint = float(row.get("bop_forecast_midpoint", 0.0) or 0.0)
    has_bop = float(row.get("bop_forecast_available", 0.0) or 0.0) > 0.0
    row["bop_estimate_bucket_under_15m"] = 1.0 if has_bop and midpoint < 15_000_000 else 0.0
    row["bop_estimate_bucket_15_30m"] = 1.0 if has_bop and 15_000_000 <= midpoint < 30_000_000 else 0.0
    row["bop_estimate_bucket_30_60m"] = 1.0 if has_bop and 30_000_000 <= midpoint < 60_000_000 else 0.0
    row["bop_estimate_bucket_60_100m"] = 1.0 if has_bop and 60_000_000 <= midpoint < 100_000_000 else 0.0
    row["bop_estimate_bucket_100m_plus"] = 1.0 if has_bop and midpoint >= 100_000_000 else 0.0


def percentile(values: Iterable[float], p: float) -> float | None:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    rank = p * (len(clean) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return clean[lo]
    weight = rank - lo
    return clean[lo] * (1.0 - weight) + clean[hi] * weight


def assign_bop_q4_proxy(
    rows: list[dict[str, object]],
    *,
    train_start_year: int,
    train_end_year: int,
) -> list[dict[str, object]]:
    q1_thresholds: dict[int, float] = {}
    q2_thresholds: dict[int, float] = {}
    q3_thresholds: dict[int, float] = {}
    q4_thresholds: dict[int, float] = {}
    by_bop_day: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_bop_day[int(row["bop_timing_day"])].append(row)
    for bop_timing_day, group in by_bop_day.items():
        train_midpoints = [
            float(row["bop_forecast_midpoint"])
            for row in group
            if train_start_year <= int(row["release_year"]) <= train_end_year
            and float(row.get("bop_forecast_available", 0.0)) > 0.0
        ]
        q1_threshold = percentile(train_midpoints, 0.25)
        q2_threshold = percentile(train_midpoints, 0.50)
        q3_threshold = percentile(train_midpoints, 0.75)
        q4_threshold = percentile(train_midpoints, 0.75)
        if q1_threshold is not None:
            q1_thresholds[bop_timing_day] = q1_threshold
        if q2_threshold is not None:
            q2_thresholds[bop_timing_day] = q2_threshold
        if q3_threshold is not None:
            q3_thresholds[bop_timing_day] = q3_threshold
        if q4_threshold is not None:
            q4_thresholds[bop_timing_day] = q4_threshold
    out_rows = []
    for row in rows:
        out = dict(row)
        q1_threshold = q1_thresholds.get(int(row["bop_timing_day"]))
        q2_threshold = q2_thresholds.get(int(row["bop_timing_day"]))
        q3_threshold = q3_thresholds.get(int(row["bop_timing_day"]))
        q4_threshold = q4_thresholds.get(int(row["bop_timing_day"]))
        midpoint = float(out.get("bop_forecast_midpoint", 0.0) or 0.0)
        has_bop = float(out.get("bop_forecast_available", 0.0)) > 0.0
        out["bop_q1_threshold"] = q1_threshold if q1_threshold is not None else ""
        out["bop_q2_threshold"] = q2_threshold if q2_threshold is not None else ""
        out["bop_q3_threshold"] = q3_threshold if q3_threshold is not None else ""
        out["bop_q4_threshold"] = q4_threshold if q4_threshold is not None else ""
        out["bop_q1_proxy"] = (
            1.0
            if q1_threshold is not None
            and has_bop
            and midpoint <= q1_threshold
            else 0.0
        )
        out["bop_q4_proxy"] = (
            1.0
            if q4_threshold is not None
            and has_bop
            and midpoint >= q4_threshold
            else 0.0
        )
        out["bop_q2_proxy"] = (
            1.0
            if q1_threshold is not None
            and q2_threshold is not None
            and has_bop
            and q1_threshold < midpoint <= q2_threshold
            else 0.0
        )
        out["bop_q3_proxy"] = (
            1.0
            if q2_threshold is not None
            and q3_threshold is not None
            and has_bop
            and q2_threshold < midpoint < q3_threshold
            else 0.0
        )
        add_bop_fixed_estimate_buckets(out)
        add_bop_interactions(out)
        out_rows.append(out)
    return out_rows


def build_feature_panel(
    movies: list[OpeningWeekendMovie],
    daily_grosses: list[DailyGross],
    wiki_by_movie: dict[int, dict[int, dict[str, float]]],
    *,
    timing_days: list[int],
    competition_timing_days: list[int] | None = None,
) -> list[dict[str, object]]:
    gross_by_movie_date = {(row.movie_id, row.box_office_date): row for row in daily_grosses}
    competition_timing_days = competition_timing_days or timing_days
    rows: list[dict[str, object]] = []
    for movie in movies:
        for timing_day in timing_days:
            wiki_as_of_date = movie.opening_date + dt.timedelta(days=timing_day)
            for competition_timing_day in competition_timing_days:
                competition_as_of_date = movie.opening_date + dt.timedelta(days=competition_timing_day)
                weekend_start, weekend_end = previous_weekend_window(competition_as_of_date)
                competition = actual_competition_features(
                    gross_by_movie_date,
                    focal_movie_id=movie.movie_id,
                    as_of_date=competition_as_of_date,
                )
                wiki = wiki_values_as_of(wiki_by_movie, movie_id=movie.movie_id, timing_day=timing_day)
                row: dict[str, object] = {
                    "movie_id": movie.movie_id,
                    "title": movie.title,
                    "release_year": movie.release_year,
                    "release_run_id": movie.release_run_id,
                    "opening_date": movie.opening_date.isoformat(),
                    "timing_day": timing_day,
                    "wiki_timing_day": timing_day,
                    "competition_timing_day": competition_timing_day,
                    "as_of_date": wiki_as_of_date.isoformat(),
                    "wiki_as_of_date": wiki_as_of_date.isoformat(),
                    "competition_as_of_date": competition_as_of_date.isoformat(),
                    "competitor_previous_weekend_start": weekend_start.isoformat(),
                    "competitor_previous_weekend_end": weekend_end.isoformat(),
                    "opening_theaters": movie.opening_theaters,
                    "opening_day_gross_usd": movie.opening_day_gross_usd,
                    "opening_weekend_revenue_usd": movie.opening_weekend_revenue_usd,
                    "target_log_opening_weekend": math.log(max(1.0, float(movie.opening_weekend_revenue_usd))),
                    "log1p_opening_theaters": log1p(movie.opening_theaters),
                    "wiki_available": 1.0 if any(wiki.values()) else 0.0,
                    "V": wiki["V"],
                    "U": wiki["U"],
                    "R": wiki["R"],
                    "E": wiki["E"],
                    "log1p_V": log1p(wiki["V"]),
                    "log1p_U": log1p(wiki["U"]),
                    "log1p_R": log1p(wiki["R"]),
                    "log1p_E": log1p(wiki["E"]),
                    **competition,
                }
                for month in range(2, 13):
                    row[f"release_month_{month}"] = 1.0 if movie.opening_date.month == month else 0.0
                rows.append(row)
    return rows


def build_bop_feature_panel(
    movies: list[OpeningWeekendMovie],
    daily_grosses: list[DailyGross],
    wiki_by_movie: dict[int, dict[int, dict[str, float]]],
    bop_forecasts: list[BoxofficeProForecast],
    *,
    timing_days: list[int],
    competition_timing_days: list[int],
    bop_timing_days: list[int],
    train_start_year: int,
    train_end_year: int,
) -> list[dict[str, object]]:
    gross_by_movie_date = {(row.movie_id, row.box_office_date): row for row in daily_grosses}
    forecasts_by_movie: dict[int, list[BoxofficeProForecast]] = defaultdict(list)
    for forecast in bop_forecasts:
        forecasts_by_movie[forecast.movie_id].append(forecast)
    rows: list[dict[str, object]] = []
    for movie in movies:
        for timing_day in timing_days:
            wiki_as_of_date = movie.opening_date + dt.timedelta(days=timing_day)
            wiki = wiki_values_as_of(wiki_by_movie, movie_id=movie.movie_id, timing_day=timing_day)
            for competition_timing_day in competition_timing_days:
                competition_as_of_date = movie.opening_date + dt.timedelta(days=competition_timing_day)
                competition = actual_competition_features(
                    gross_by_movie_date,
                    focal_movie_id=movie.movie_id,
                    as_of_date=competition_as_of_date,
                )
                for bop_timing_day in bop_timing_days:
                    bop_as_of_date = movie.opening_date + dt.timedelta(days=bop_timing_day)
                    focal_forecast = latest_forecast(
                        forecasts_by_movie.get(movie.movie_id, []),
                        as_of_date=bop_as_of_date,
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
                        "timing_day": timing_day,
                        "wiki_timing_day": timing_day,
                        "competition_timing_day": competition_timing_day,
                        "bop_timing_day": bop_timing_day,
                        "as_of_date": wiki_as_of_date.isoformat(),
                        "wiki_as_of_date": wiki_as_of_date.isoformat(),
                        "competition_as_of_date": competition_as_of_date.isoformat(),
                        "bop_as_of_date": bop_as_of_date.isoformat(),
                        "opening_theaters": movie.opening_theaters,
                        "opening_day_gross_usd": movie.opening_day_gross_usd,
                        "opening_weekend_revenue_usd": movie.opening_weekend_revenue_usd,
                        "target_log_opening_weekend": math.log(max(1.0, float(movie.opening_weekend_revenue_usd))),
                        "log1p_opening_theaters": log1p(movie.opening_theaters),
                        "wiki_available": 1.0 if any(wiki.values()) else 0.0,
                        "V": wiki["V"],
                        "U": wiki["U"],
                        "R": wiki["R"],
                        "E": wiki["E"],
                        "log1p_V": log1p(wiki["V"]),
                        "log1p_U": log1p(wiki["U"]),
                        "log1p_R": log1p(wiki["R"]),
                        "log1p_E": log1p(wiki["E"]),
                        **competition,
                        **bop_forecast_features(focal_forecast),
                        **bop_same_weekend_competitor_features(
                            bop_forecasts,
                            focal_movie_id=movie.movie_id,
                            target_start_date=movie.opening_date,
                            as_of_date=bop_as_of_date,
                        ),
                    }
                    for month in range(2, 13):
                        row[f"release_month_{month}"] = 1.0 if movie.opening_date.month == month else 0.0
                    rows.append(row)
    return assign_bop_q4_proxy(
        rows,
        train_start_year=train_start_year,
        train_end_year=train_end_year,
    )


def design_matrix(rows: list[dict[str, object]], terms: list[str]) -> list[list[float]]:
    return [[float(row.get(term, 0.0) or 0.0) for term in terms] for row in rows]


def standardize_fit(x_train: list[list[float]]) -> tuple[list[float], list[float]]:
    width = len(x_train[0]) if x_train else 0
    centers = [mean(row[j] for row in x_train) for j in range(width)]
    scales = []
    for j, center in enumerate(centers):
        variance = mean((row[j] - center) ** 2 for row in x_train)
        scales.append(math.sqrt(variance) or 1.0)
    return centers, scales


def standardize_apply(rows: list[list[float]], centers: list[float], scales: list[float]) -> list[list[float]]:
    return [[1.0] + [(row[j] - centers[j]) / scales[j] for j in range(len(centers))] for row in rows]


def fit_model_for_target(rows: list[dict[str, object]], terms: list[str], target_key: str) -> FittedModel:
    if len(rows) < len(terms) + 2:
        raise ValueError("not enough training rows")
    x = design_matrix(rows, terms)
    y = [float(row[target_key]) for row in rows]
    centers, scales = standardize_fit(x)
    z = standardize_apply(x, centers, scales)
    width = len(z[0])
    gram = [[sum(row[i] * row[j] for row in z) for j in range(width)] for i in range(width)]
    target = [sum(row[i] * value for row, value in zip(z, y)) for i in range(width)]
    try:
        beta = solve_linear_system(gram, target)
    except ValueError:
        for i in range(width):
            gram[i][i] += 1e-9
        beta = solve_linear_system(gram, target)
    return FittedModel(terms=terms, centers=centers, scales=scales, beta=beta)


def fit_model(rows: list[dict[str, object]], terms: list[str]) -> FittedModel:
    return fit_model_for_target(rows, terms, "target_log_opening_weekend")


def predict_log(model: FittedModel, rows: list[dict[str, object]]) -> list[float]:
    x = design_matrix(rows, model.terms)
    z = standardize_apply(x, model.centers, model.scales)
    return [sum(coef * value for coef, value in zip(model.beta, row)) for row in z]


def rmse(errors: Iterable[float]) -> float | None:
    values = list(errors)
    if not values:
        return None
    return math.sqrt(sum(value * value for value in values) / len(values))


def mae(errors: Iterable[float]) -> float | None:
    values = list(errors)
    if not values:
        return None
    return sum(abs(value) for value in values) / len(values)


def evaluate_time_split(
    panel_rows: list[dict[str, object]],
    *,
    timing_days: list[int],
    competition_timing_days: list[int] | None = None,
    train_start_year: int,
    train_end_year: int,
    test_start_year: int,
    test_end_year: int,
    require_wiki_available: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    competition_timing_days = competition_timing_days or timing_days
    for timing_day in timing_days:
        for competition_timing_day in competition_timing_days:
            day_rows = [
                row
                for row in panel_rows
                if int(row.get("wiki_timing_day", row["timing_day"])) == timing_day
                and int(row.get("competition_timing_day", row["timing_day"])) == competition_timing_day
            ]
            if require_wiki_available:
                day_rows = [row for row in day_rows if float(row.get("wiki_available", 1.0)) > 0.0]
            train_rows = [
                row for row in day_rows if train_start_year <= int(row["release_year"]) <= train_end_year
            ]
            holdout_rows = [
                row for row in day_rows if test_start_year <= int(row["release_year"]) <= test_end_year
            ]
            for model_name, terms in MODEL_TERMS.items():
                metric_base = {
                    "model": model_name,
                    "timing_day": timing_day,
                    "wiki_timing_day": timing_day,
                    "competition_timing_day": competition_timing_day,
                    "train_start_year": train_start_year,
                    "train_end_year": train_end_year,
                    "test_start_year": test_start_year,
                    "test_end_year": test_end_year,
                    "train_n": len(train_rows),
                    "holdout_n": len(holdout_rows),
                }
                if len(train_rows) < len(terms) + 2 or len(holdout_rows) < 2:
                    metric_rows.append(
                        {
                            **metric_base,
                        "r2_log_revenue": "",
                        "r2_gross": "",
                        "mape_gross": "",
                        "rmse_log_revenue": "",
                        "mae_log_revenue": "",
                        "mean_actual_gross": "",
                        "mean_predicted_gross": "",
                        "status": "insufficient_sample",
                        }
                    )
                    continue
                fitted = fit_model(train_rows, terms)
                holdout_pred = predict_log(fitted, holdout_rows)
                actual_log = [float(row["target_log_opening_weekend"]) for row in holdout_rows]
                actual_gross = [float(row["opening_weekend_revenue_usd"]) for row in holdout_rows]
                predicted_gross = [max(1.0, math.exp(value)) for value in holdout_pred]
                log_errors = [actual - pred for actual, pred in zip(actual_log, holdout_pred)]
                apes = [
                    abs(pred - actual) / actual
                    for actual, pred in zip(actual_gross, predicted_gross)
                    if actual > 0.0
                ]
                metric_rows.append(
                    {
                    **metric_base,
                    "r2_log_revenue": format_number(r2_score(actual_log, holdout_pred)),
                    "r2_gross": format_number(r2_score(actual_gross, predicted_gross)),
                    "mape_gross": format_number(mean(apes) if apes else None),
                    "rmse_log_revenue": format_number(rmse(log_errors)),
                    "mae_log_revenue": format_number(mae(log_errors)),
                    "mean_actual_gross": format_number(mean(actual_gross)),
                    "mean_predicted_gross": format_number(mean(predicted_gross)),
                    "status": "ok",
                    }
                )
                coefficient_items = zip(
                    ["intercept"] + terms,
                    fitted.beta,
                    [0.0] + fitted.centers,
                    [1.0] + fitted.scales,
                )
                for term, coef, center, scale in coefficient_items:
                    coefficient_rows.append(
                        {
                            "model": model_name,
                            "timing_day": timing_day,
                            "wiki_timing_day": timing_day,
                            "competition_timing_day": competition_timing_day,
                            "term": term,
                            "standardized_coef": coef,
                            "center": center,
                            "scale": scale,
                            "train_n": len(train_rows),
                        }
                    )
                for row, pred_log, pred_gross in zip(holdout_rows, holdout_pred, predicted_gross):
                    prediction_rows.append(
                        {
                            "model": model_name,
                            "timing_day": timing_day,
                            "wiki_timing_day": timing_day,
                            "competition_timing_day": competition_timing_day,
                        "movie_id": row["movie_id"],
                        "title": row["title"],
                        "release_year": row["release_year"],
                        "opening_date": row["opening_date"],
                        "actual_log_opening_weekend": row["target_log_opening_weekend"],
                        "predicted_log_opening_weekend": pred_log,
                        "actual_opening_weekend_revenue_usd": row["opening_weekend_revenue_usd"],
                        "predicted_opening_weekend_revenue_usd": pred_gross,
                        "absolute_percentage_error": (
                            abs(pred_gross - float(row["opening_weekend_revenue_usd"]))
                            / float(row["opening_weekend_revenue_usd"])
                        ),
                        "competitor_total_gross_lag1": row.get("competitor_total_gross_lag1", 0.0),
                        "competitor_top1_gross_lag1": row.get("competitor_top1_gross_lag1", 0.0),
                        "competitor_total_gross_previous_weekend": row.get(
                            "competitor_total_gross_previous_weekend", 0.0
                        ),
                        "competitor_count_lag1": row.get("competitor_count_lag1", 0.0),
                        }
                    )
    return prediction_rows, metric_rows, coefficient_rows


def evaluate_bop_q4_split(
    panel_rows: list[dict[str, object]],
    *,
    timing_days: list[int],
    competition_timing_days: list[int],
    bop_timing_days: list[int],
    train_start_year: int,
    train_end_year: int,
    test_start_year: int,
    test_end_year: int,
    require_wiki_available: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    for timing_day in timing_days:
        for competition_timing_day in competition_timing_days:
            for bop_timing_day in bop_timing_days:
                day_rows = [
                    row
                    for row in panel_rows
                    if int(row["wiki_timing_day"]) == timing_day
                    and int(row["competition_timing_day"]) == competition_timing_day
                    and int(row["bop_timing_day"]) == bop_timing_day
                ]
                if require_wiki_available:
                    day_rows = [row for row in day_rows if float(row.get("wiki_available", 1.0)) > 0.0]
                train_rows = [
                    row for row in day_rows if train_start_year <= int(row["release_year"]) <= train_end_year
                ]
                holdout_rows = [
                    row for row in day_rows if test_start_year <= int(row["release_year"]) <= test_end_year
                ]
                for model_name, terms in BOP_Q4_MODEL_TERMS.items():
                    metric_base = {
                        "model": model_name,
                        "timing_day": timing_day,
                        "wiki_timing_day": timing_day,
                        "competition_timing_day": competition_timing_day,
                        "bop_timing_day": bop_timing_day,
                        "train_start_year": train_start_year,
                        "train_end_year": train_end_year,
                        "test_start_year": test_start_year,
                        "test_end_year": test_end_year,
                        "train_n": len(train_rows),
                        "holdout_n": len(holdout_rows),
                    }
                    if len(train_rows) < len(terms) + 2 or len(holdout_rows) < 2:
                        metric_rows.append(
                            {
                                **metric_base,
                                "r2_log_revenue": "",
                                "r2_gross": "",
                                "mape_gross": "",
                                "rmse_log_revenue": "",
                                "mae_log_revenue": "",
                                "mean_actual_gross": "",
                                "mean_predicted_gross": "",
                                "status": "insufficient_sample",
                            }
                        )
                        continue
                    fitted = fit_model(train_rows, terms)
                    holdout_pred = predict_log(fitted, holdout_rows)
                    actual_log = [float(row["target_log_opening_weekend"]) for row in holdout_rows]
                    actual_gross = [float(row["opening_weekend_revenue_usd"]) for row in holdout_rows]
                    predicted_gross = [max(1.0, math.exp(value)) for value in holdout_pred]
                    log_errors = [actual - pred for actual, pred in zip(actual_log, holdout_pred)]
                    apes = [
                        abs(pred - actual) / actual
                        for actual, pred in zip(actual_gross, predicted_gross)
                        if actual > 0.0
                    ]
                    metric_rows.append(
                        {
                            **metric_base,
                            "r2_log_revenue": format_number(r2_score(actual_log, holdout_pred)),
                            "r2_gross": format_number(r2_score(actual_gross, predicted_gross)),
                            "mape_gross": format_number(mean(apes) if apes else None),
                            "rmse_log_revenue": format_number(rmse(log_errors)),
                            "mae_log_revenue": format_number(mae(log_errors)),
                            "mean_actual_gross": format_number(mean(actual_gross)),
                            "mean_predicted_gross": format_number(mean(predicted_gross)),
                            "status": "ok",
                        }
                    )
                    for term, coef, center, scale in zip(
                        ["intercept"] + terms,
                        fitted.beta,
                        [0.0] + fitted.centers,
                        [1.0] + fitted.scales,
                    ):
                        coefficient_rows.append(
                            {
                                "model": model_name,
                                "timing_day": timing_day,
                                "wiki_timing_day": timing_day,
                                "competition_timing_day": competition_timing_day,
                                "bop_timing_day": bop_timing_day,
                                "term": term,
                                "standardized_coef": coef,
                                "center": center,
                                "scale": scale,
                                "train_n": len(train_rows),
                            }
                        )
                    for row, pred_log, pred_gross in zip(holdout_rows, holdout_pred, predicted_gross):
                        prediction_rows.append(
                            {
                                "model": model_name,
                                "timing_day": timing_day,
                                "wiki_timing_day": timing_day,
                                "competition_timing_day": competition_timing_day,
                                "bop_timing_day": bop_timing_day,
                                "movie_id": row["movie_id"],
                                "title": row["title"],
                                "release_year": row["release_year"],
                                "opening_date": row["opening_date"],
                                "actual_log_opening_weekend": row["target_log_opening_weekend"],
                                "predicted_log_opening_weekend": pred_log,
                                "actual_opening_weekend_revenue_usd": row["opening_weekend_revenue_usd"],
                                "predicted_opening_weekend_revenue_usd": pred_gross,
                                "absolute_percentage_error": (
                                    abs(pred_gross - float(row["opening_weekend_revenue_usd"]))
                                    / float(row["opening_weekend_revenue_usd"])
                                ),
                                "bop_forecast_midpoint": row.get("bop_forecast_midpoint", 0.0),
                                "bop_q4_proxy": row.get("bop_q4_proxy", 0.0),
                                "competitor_total_gross_lag1": row.get("competitor_total_gross_lag1", 0.0),
                            }
                        )
    return prediction_rows, metric_rows, coefficient_rows


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    xbar, ybar = mean(xs), mean(ys)
    numerator = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    xden = math.sqrt(sum((x - xbar) ** 2 for x in xs))
    yden = math.sqrt(sum((y - ybar) ** 2 for y in ys))
    if xden == 0.0 or yden == 0.0:
        return None
    return numerator / (xden * yden)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    out = [0.0] * len(values)
    idx = 0
    while idx < len(order):
        end = idx + 1
        while end < len(order) and values[order[end]] == values[order[idx]]:
            end += 1
        rank = (idx + end - 1) / 2.0 + 1.0
        for pos in range(idx, end):
            out[order[pos]] = rank
        idx = end
    return out


def spearman_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson_correlation(ranks(xs), ranks(ys))


def evaluate_bop_calibration(
    panel_rows: list[dict[str, object]],
    *,
    timing_days: list[int],
    bop_timing_days: list[int],
    train_start_year: int,
    train_end_year: int,
    test_start_year: int,
    test_end_year: int,
    require_wiki_available_for_wiki_model: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    for timing_day in timing_days:
        for bop_timing_day in bop_timing_days:
            day_rows_by_movie: dict[int, dict[str, object]] = {}
            for row in panel_rows:
                if (
                    int(row["wiki_timing_day"]) == timing_day
                    and int(row["bop_timing_day"]) == bop_timing_day
                    and float(row.get("bop_forecast_available", 0.0)) > 0.0
                ):
                    day_rows_by_movie.setdefault(int(row["movie_id"]), row)
            day_rows = list(day_rows_by_movie.values())
            train_rows_all = [
                row for row in day_rows if train_start_year <= int(row["release_year"]) <= train_end_year
            ]
            holdout_rows_all = [
                row for row in day_rows if test_start_year <= int(row["release_year"]) <= test_end_year
            ]
            if len(holdout_rows_all) >= 2:
                log_midpoints = [float(row["log1p_bop_forecast_midpoint"]) for row in holdout_rows_all]
                actual_log = [float(row["target_log_opening_weekend"]) for row in holdout_rows_all]
                correlation_rows.append(
                    {
                        "timing_day": timing_day,
                        "wiki_timing_day": timing_day,
                        "bop_timing_day": bop_timing_day,
                        "holdout_n": len(holdout_rows_all),
                        "pearson_log_midpoint_actual": format_number(pearson_correlation(log_midpoints, actual_log)),
                        "spearman_midpoint_actual": format_number(
                            spearman_correlation(
                                [float(row["bop_forecast_midpoint"]) for row in holdout_rows_all],
                                [float(row["opening_weekend_revenue_usd"]) for row in holdout_rows_all],
                            )
                        ),
                    }
                )
            models = {"raw_midpoint": []} | BOP_CALIBRATION_MODEL_TERMS
            for model_name, terms in models.items():
                train_rows = train_rows_all
                holdout_rows = holdout_rows_all
                if model_name == "midpoint_plus_wiki" and require_wiki_available_for_wiki_model:
                    train_rows = [row for row in train_rows if float(row.get("wiki_available", 0.0)) > 0.0]
                    holdout_rows = [row for row in holdout_rows if float(row.get("wiki_available", 0.0)) > 0.0]
                metric_base = {
                    "model": model_name,
                    "timing_day": timing_day,
                    "wiki_timing_day": timing_day,
                    "bop_timing_day": bop_timing_day,
                    "train_start_year": train_start_year,
                    "train_end_year": train_end_year,
                    "test_start_year": test_start_year,
                    "test_end_year": test_end_year,
                    "train_n": len(train_rows),
                    "holdout_n": len(holdout_rows),
                }
                if model_name == "raw_midpoint":
                    if len(holdout_rows) < 2:
                        metric_rows.append({**metric_base, "r2_log_revenue": "", "r2_gross": "", "mape_gross": "", "rmse_log_revenue": "", "mae_log_revenue": "", "mean_actual_gross": "", "mean_predicted_gross": "", "calibration_intercept": "", "calibration_slope_log_midpoint": "", "status": "insufficient_sample"})
                        continue
                    pred_log = [math.log(max(1.0, float(row["bop_forecast_midpoint"]))) for row in holdout_rows]
                    intercept, slope = 0.0, 1.0
                else:
                    if len(train_rows) < len(terms) + 2 or len(holdout_rows) < 2:
                        metric_rows.append({**metric_base, "r2_log_revenue": "", "r2_gross": "", "mape_gross": "", "rmse_log_revenue": "", "mae_log_revenue": "", "mean_actual_gross": "", "mean_predicted_gross": "", "calibration_intercept": "", "calibration_slope_log_midpoint": "", "status": "insufficient_sample"})
                        continue
                    fitted = fit_model(train_rows, terms)
                    pred_log = predict_log(fitted, holdout_rows)
                    natural_slopes = [coef / scale for coef, scale in zip(fitted.beta[1:], fitted.scales)]
                    intercept = fitted.beta[0] - sum(
                        slope_value * center for slope_value, center in zip(natural_slopes, fitted.centers)
                    )
                    slope = natural_slopes[0] if terms and terms[0] == "log1p_bop_forecast_midpoint" else ""
                actual_log = [float(row["target_log_opening_weekend"]) for row in holdout_rows]
                actual_gross = [float(row["opening_weekend_revenue_usd"]) for row in holdout_rows]
                predicted_gross = [max(1.0, math.exp(value)) for value in pred_log]
                log_errors = [actual - pred for actual, pred in zip(actual_log, pred_log)]
                apes = [
                    abs(pred - actual) / actual
                    for actual, pred in zip(actual_gross, predicted_gross)
                    if actual > 0.0
                ]
                metric_rows.append(
                    {
                        **metric_base,
                        "r2_log_revenue": format_number(r2_score(actual_log, pred_log)),
                        "r2_gross": format_number(r2_score(actual_gross, predicted_gross)),
                        "mape_gross": format_number(mean(apes) if apes else None),
                        "rmse_log_revenue": format_number(rmse(log_errors)),
                        "mae_log_revenue": format_number(mae(log_errors)),
                        "mean_actual_gross": format_number(mean(actual_gross)),
                        "mean_predicted_gross": format_number(mean(predicted_gross)),
                        "calibration_intercept": format_number(intercept) if intercept != "" else "",
                        "calibration_slope_log_midpoint": format_number(slope) if slope != "" else "",
                        "status": "ok",
                    }
                )
                for row, predicted_log, predicted_gross_value in zip(holdout_rows, pred_log, predicted_gross):
                    prediction_rows.append(
                        {
                            "model": model_name,
                            "timing_day": timing_day,
                            "wiki_timing_day": timing_day,
                            "bop_timing_day": bop_timing_day,
                            "movie_id": row["movie_id"],
                            "title": row["title"],
                            "release_year": row["release_year"],
                            "opening_date": row["opening_date"],
                            "bop_forecast_midpoint": row["bop_forecast_midpoint"],
                            "actual_log_opening_weekend": row["target_log_opening_weekend"],
                            "predicted_log_opening_weekend": predicted_log,
                            "actual_opening_weekend_revenue_usd": row["opening_weekend_revenue_usd"],
                            "predicted_opening_weekend_revenue_usd": predicted_gross_value,
                            "absolute_percentage_error": (
                                abs(predicted_gross_value - float(row["opening_weekend_revenue_usd"]))
                                / float(row["opening_weekend_revenue_usd"])
                            ),
                        }
                    )
    return prediction_rows, metric_rows, correlation_rows


def smearing_factor(model: FittedModel, rows: list[dict[str, object]]) -> float:
    if not rows:
        return 1.0
    train_pred = predict_log(model, rows)
    factors = [
        math.exp(float(row["target_log_opening_weekend"]) - pred)
        for row, pred in zip(rows, train_pred)
    ]
    return mean(factors) if factors else 1.0


def metric_row_from_predictions(
    *,
    base: dict[str, object],
    actual_log: list[float],
    actual_gross: list[float],
    predicted_log: list[float],
    predicted_gross: list[float],
) -> dict[str, object]:
    log_errors = [actual - pred for actual, pred in zip(actual_log, predicted_log)]
    apes = [
        abs(pred - actual) / actual
        for actual, pred in zip(actual_gross, predicted_gross)
        if actual > 0.0
    ]
    return {
        **base,
        "r2_log_revenue": format_number(r2_score(actual_log, predicted_log)),
        "r2_gross": format_number(r2_score(actual_gross, predicted_gross)),
        "mape_gross": format_number(mean(apes) if apes else None),
        "rmse_log_revenue": format_number(rmse(log_errors)),
        "mae_log_revenue": format_number(mae(log_errors)),
        "mean_actual_gross": format_number(mean(actual_gross)),
        "mean_predicted_gross": format_number(mean(predicted_gross)),
        "status": "ok",
    }


def unique_rows_for_timing(
    panel_rows: list[dict[str, object]],
    *,
    wiki_timing_day: int,
    competition_timing_day: int,
    bop_timing_day: int | None = None,
) -> list[dict[str, object]]:
    by_movie: dict[int, dict[str, object]] = {}
    for row in panel_rows:
        if int(row["wiki_timing_day"]) != wiki_timing_day:
            continue
        if int(row["competition_timing_day"]) != competition_timing_day:
            continue
        if bop_timing_day is not None and int(row.get("bop_timing_day", bop_timing_day)) != bop_timing_day:
            continue
        by_movie.setdefault(int(row["movie_id"]), row)
    return list(by_movie.values())


def population_rows(rows: list[dict[str, object]], population: str) -> list[dict[str, object]]:
    if population == "bop_covered":
        return [row for row in rows if float(row.get("bop_forecast_available", 0.0)) > 0.0]
    if population == "bop_wiki_covered":
        return [
            row
            for row in rows
            if float(row.get("bop_forecast_available", 0.0)) > 0.0
            and float(row.get("wiki_available", 0.0)) > 0.0
        ]
    if population == "full_with_fallback":
        return rows
    raise ValueError(f"unknown population {population!r}")


def evaluate_multisource_models(
    bop_panel_rows: list[dict[str, object]],
    actual_panel_rows: list[dict[str, object]],
    *,
    timing_days: list[int],
    competition_timing_days: list[int],
    bop_timing_days: list[int],
    train_start_year: int,
    train_end_year: int,
    test_start_year: int,
    test_end_year: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    populations = ("bop_covered", "bop_wiki_covered", "full_with_fallback")
    for timing_day in timing_days:
        for competition_timing_day in competition_timing_days:
            fallback_rows = unique_rows_for_timing(
                actual_panel_rows,
                wiki_timing_day=timing_day,
                competition_timing_day=competition_timing_day,
            )
            fallback_train_rows = [
                row for row in fallback_rows if train_start_year <= int(row["release_year"]) <= train_end_year
            ]
            fallback_model: FittedModel | None = None
            if len(fallback_train_rows) >= len(FALLBACK_MODEL_TERMS) + 2:
                fallback_model = fit_model(fallback_train_rows, FALLBACK_MODEL_TERMS)
            fallback_holdout_by_movie = {
                int(row["movie_id"]): row
                for row in fallback_rows
                if test_start_year <= int(row["release_year"]) <= test_end_year
            }
            for bop_timing_day in bop_timing_days:
                timing_rows = unique_rows_for_timing(
                    bop_panel_rows,
                    wiki_timing_day=timing_day,
                    competition_timing_day=competition_timing_day,
                    bop_timing_day=bop_timing_day,
                )
                for population in populations:
                    scoped_rows = population_rows(timing_rows, population)
                    train_rows_all = [
                        row for row in scoped_rows if train_start_year <= int(row["release_year"]) <= train_end_year
                    ]
                    holdout_rows_all = [
                        row for row in scoped_rows if test_start_year <= int(row["release_year"]) <= test_end_year
                    ]
                    for model_name, terms in ({"raw_bop_midpoint": []} | MULTISOURCE_MODEL_TERMS).items():
                        metric_base = {
                            "model": model_name,
                            "population": population,
                            "prediction_transform": "midpoint" if model_name == "raw_bop_midpoint" else "exp",
                            "timing_day": timing_day,
                            "wiki_timing_day": timing_day,
                            "competition_timing_day": competition_timing_day,
                            "bop_timing_day": bop_timing_day,
                            "train_start_year": train_start_year,
                            "train_end_year": train_end_year,
                            "test_start_year": test_start_year,
                            "test_end_year": test_end_year,
                        }
                        train_rows = train_rows_all
                        holdout_rows = holdout_rows_all
                        if population == "full_with_fallback":
                            train_rows = [
                                row for row in train_rows_all if float(row.get("bop_forecast_available", 0.0)) > 0.0
                            ]
                            holdout_rows = [
                                row for row in holdout_rows_all if float(row.get("bop_forecast_available", 0.0)) > 0.0
                            ]
                        if model_name == "raw_bop_midpoint":
                            combined_rows = list(holdout_rows)
                            pred_log = [
                                math.log(max(1.0, float(row["bop_forecast_midpoint"])))
                                for row in holdout_rows
                            ]
                            pred_gross = [max(1.0, float(row["bop_forecast_midpoint"])) for row in holdout_rows]
                            combined_sources = ["bop"] * len(holdout_rows)
                            if population == "full_with_fallback" and fallback_model is not None:
                                bop_movie_ids = {int(row["movie_id"]) for row in holdout_rows}
                                missing_rows = [
                                    fallback_holdout_by_movie[movie_id]
                                    for movie_id in sorted(fallback_holdout_by_movie)
                                    if movie_id not in bop_movie_ids
                                ]
                                fallback_pred = predict_log(fallback_model, missing_rows)
                                for row, log_value in zip(missing_rows, fallback_pred):
                                    combined_rows.append(row)
                                    pred_log.append(log_value)
                                    pred_gross.append(max(1.0, math.exp(log_value)))
                                    combined_sources.append("fallback")
                            if len(combined_rows) < 2:
                                metric_rows.append(
                                    {
                                        **metric_base,
                                        "train_n": 0,
                                        "holdout_n": len(combined_rows),
                                        "bop_prediction_n": sum(1 for source in combined_sources if source == "bop"),
                                        "fallback_prediction_n": sum(
                                            1 for source in combined_sources if source == "fallback"
                                        ),
                                        "r2_log_revenue": "",
                                        "r2_gross": "",
                                        "mape_gross": "",
                                        "rmse_log_revenue": "",
                                        "mae_log_revenue": "",
                                        "mean_actual_gross": "",
                                        "mean_predicted_gross": "",
                                        "smearing_factor": "",
                                        "status": "insufficient_sample",
                                    }
                                )
                                continue
                            actual_log = [float(row["target_log_opening_weekend"]) for row in combined_rows]
                            actual_gross = [float(row["opening_weekend_revenue_usd"]) for row in combined_rows]
                            metric_rows.append(
                                metric_row_from_predictions(
                                    base={
                                        **metric_base,
                                        "train_n": 0,
                                        "holdout_n": len(combined_rows),
                                        "bop_prediction_n": sum(1 for source in combined_sources if source == "bop"),
                                        "fallback_prediction_n": sum(
                                            1 for source in combined_sources if source == "fallback"
                                        ),
                                        "smearing_factor": "",
                                    },
                                    actual_log=actual_log,
                                    actual_gross=actual_gross,
                                    predicted_log=pred_log,
                                    predicted_gross=pred_gross,
                                )
                            )
                            for row, source, log_value, gross_value in zip(
                                combined_rows,
                                combined_sources,
                                pred_log,
                                pred_gross,
                            ):
                                prediction_rows.append(
                                    multisource_prediction_row(
                                        row,
                                        model_name=model_name,
                                        population=population,
                                        prediction_transform="midpoint",
                                        prediction_source=source,
                                        predicted_log=log_value,
                                        predicted_gross=gross_value,
                                    )
                                )
                            continue
                        if len(train_rows) < len(terms) + 2 or len(holdout_rows_all) < 2:
                            for transform in ("exp", "smeared"):
                                metric_rows.append(
                                    {
                                        **metric_base,
                                        "prediction_transform": transform,
                                        "train_n": len(train_rows),
                                        "holdout_n": len(holdout_rows_all),
                                        "bop_prediction_n": len(holdout_rows),
                                        "fallback_prediction_n": 0,
                                        "r2_log_revenue": "",
                                        "r2_gross": "",
                                        "mape_gross": "",
                                        "rmse_log_revenue": "",
                                        "mae_log_revenue": "",
                                        "mean_actual_gross": "",
                                        "mean_predicted_gross": "",
                                        "smearing_factor": "",
                                        "status": "insufficient_sample",
                                    }
                                )
                            continue
                        fitted = fit_model(train_rows, terms)
                        smear = smearing_factor(fitted, train_rows)
                        for term, coef, center, scale in zip(
                            ["intercept"] + terms,
                            fitted.beta,
                            [0.0] + fitted.centers,
                            [1.0] + fitted.scales,
                        ):
                            coefficient_rows.append(
                                {
                                    "model": model_name,
                                    "population": population,
                                    "timing_day": timing_day,
                                    "wiki_timing_day": timing_day,
                                    "competition_timing_day": competition_timing_day,
                                    "bop_timing_day": bop_timing_day,
                                    "term": term,
                                    "standardized_coef": coef,
                                    "center": center,
                                    "scale": scale,
                                    "train_n": len(train_rows),
                                }
                            )
                        bop_pred_log = predict_log(fitted, holdout_rows)
                        for transform in ("exp", "smeared"):
                            transform_smear = smear if transform == "smeared" else 1.0
                            combined_rows: list[dict[str, object]] = []
                            combined_log: list[float] = []
                            combined_gross: list[float] = []
                            combined_sources: list[str] = []
                            for row, log_value in zip(holdout_rows, bop_pred_log):
                                pred_log_value = log_value + math.log(transform_smear)
                                combined_rows.append(row)
                                combined_log.append(pred_log_value)
                                combined_gross.append(max(1.0, math.exp(pred_log_value)))
                                combined_sources.append("bop")
                            if population == "full_with_fallback" and fallback_model is not None:
                                missing_rows = [
                                    fallback_holdout_by_movie[movie_id]
                                    for movie_id in sorted(fallback_holdout_by_movie)
                                    if movie_id
                                    not in {int(row["movie_id"]) for row in holdout_rows}
                                ]
                                fallback_pred = predict_log(fallback_model, missing_rows)
                                for row, log_value in zip(missing_rows, fallback_pred):
                                    combined_rows.append(row)
                                    combined_log.append(log_value)
                                    combined_gross.append(max(1.0, math.exp(log_value)))
                                    combined_sources.append("fallback")
                            if len(combined_rows) < 2:
                                continue
                            actual_log = [float(row["target_log_opening_weekend"]) for row in combined_rows]
                            actual_gross = [float(row["opening_weekend_revenue_usd"]) for row in combined_rows]
                            metric_rows.append(
                                metric_row_from_predictions(
                                    base={
                                        **metric_base,
                                        "prediction_transform": transform,
                                        "train_n": len(train_rows),
                                        "holdout_n": len(combined_rows),
                                        "bop_prediction_n": sum(1 for source in combined_sources if source == "bop"),
                                        "fallback_prediction_n": sum(
                                            1 for source in combined_sources if source == "fallback"
                                        ),
                                        "smearing_factor": format_number(transform_smear),
                                    },
                                    actual_log=actual_log,
                                    actual_gross=actual_gross,
                                    predicted_log=combined_log,
                                    predicted_gross=combined_gross,
                                )
                            )
                            for row, source, log_value, gross_value in zip(
                                combined_rows,
                                combined_sources,
                                combined_log,
                                combined_gross,
                            ):
                                prediction_rows.append(
                                    multisource_prediction_row(
                                        row,
                                        model_name=model_name,
                                        population=population,
                                        prediction_transform=transform,
                                        prediction_source=source,
                                        predicted_log=log_value,
                                        predicted_gross=gross_value,
                                    )
                                )
    return prediction_rows, metric_rows, coefficient_rows


def multisource_prediction_row(
    row: dict[str, object],
    *,
    model_name: str,
    population: str,
    prediction_transform: str,
    prediction_source: str,
    predicted_log: float,
    predicted_gross: float,
) -> dict[str, object]:
    actual_gross = float(row["opening_weekend_revenue_usd"])
    return {
        "model": model_name,
        "population": population,
        "prediction_transform": prediction_transform,
        "prediction_source": prediction_source,
        "timing_day": row.get("timing_day", row.get("wiki_timing_day", "")),
        "wiki_timing_day": row.get("wiki_timing_day", ""),
        "competition_timing_day": row.get("competition_timing_day", ""),
        "bop_timing_day": row.get("bop_timing_day", ""),
        "movie_id": row["movie_id"],
        "title": row["title"],
        "release_year": row["release_year"],
        "opening_date": row["opening_date"],
        "bop_forecast_available": row.get("bop_forecast_available", 0.0),
        "bop_forecast_midpoint": row.get("bop_forecast_midpoint", 0.0),
        "bop_q1_proxy": row.get("bop_q1_proxy", 0.0),
        "bop_q4_proxy": row.get("bop_q4_proxy", 0.0),
        "actual_log_opening_weekend": row["target_log_opening_weekend"],
        "predicted_log_opening_weekend": predicted_log,
        "actual_opening_weekend_revenue_usd": row["opening_weekend_revenue_usd"],
        "predicted_opening_weekend_revenue_usd": predicted_gross,
        "absolute_percentage_error": abs(predicted_gross - actual_gross) / actual_gross if actual_gross > 0.0 else "",
    }


def rows_with_bop_residual_target(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out_rows = []
    for row in rows:
        midpoint = float(row.get("bop_forecast_midpoint", 0.0) or 0.0)
        if midpoint <= 0.0 or float(row.get("bop_forecast_available", 0.0)) <= 0.0:
            continue
        out = dict(row)
        out["target_log_bop_residual"] = float(row["target_log_opening_weekend"]) - math.log(midpoint)
        out_rows.append(out)
    return out_rows


def evaluate_residual_lift_models(
    bop_panel_rows: list[dict[str, object]],
    *,
    timing_days: list[int],
    competition_timing_days: list[int],
    bop_timing_days: list[int],
    train_start_year: int,
    train_end_year: int,
    test_start_year: int,
    test_end_year: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    populations = ("bop_covered", "bop_wiki_covered")
    for timing_day in timing_days:
        for competition_timing_day in competition_timing_days:
            for bop_timing_day in bop_timing_days:
                timing_rows = unique_rows_for_timing(
                    bop_panel_rows,
                    wiki_timing_day=timing_day,
                    competition_timing_day=competition_timing_day,
                    bop_timing_day=bop_timing_day,
                )
                for population in populations:
                    scoped_rows = rows_with_bop_residual_target(population_rows(timing_rows, population))
                    train_rows = [
                        row for row in scoped_rows if train_start_year <= int(row["release_year"]) <= train_end_year
                    ]
                    holdout_rows = [
                        row for row in scoped_rows if test_start_year <= int(row["release_year"]) <= test_end_year
                    ]
                    for model_name, terms in ({"raw_bop_midpoint": []} | RESIDUAL_LIFT_MODEL_TERMS).items():
                        metric_base = {
                            "model": model_name,
                            "population": population,
                            "timing_day": timing_day,
                            "wiki_timing_day": timing_day,
                            "competition_timing_day": competition_timing_day,
                            "bop_timing_day": bop_timing_day,
                            "train_start_year": train_start_year,
                            "train_end_year": train_end_year,
                            "test_start_year": test_start_year,
                            "test_end_year": test_end_year,
                            "train_n": 0 if model_name == "raw_bop_midpoint" else len(train_rows),
                            "holdout_n": len(holdout_rows),
                        }
                        if len(holdout_rows) < 2 or (
                            model_name != "raw_bop_midpoint" and len(train_rows) < len(terms) + 2
                        ):
                            metric_rows.append(
                                {
                                    **metric_base,
                                    "r2_log_revenue": "",
                                    "r2_gross": "",
                                    "mape_gross": "",
                                    "rmse_log_revenue": "",
                                    "mae_log_revenue": "",
                                    "mean_actual_gross": "",
                                    "mean_predicted_gross": "",
                                    "mean_predicted_residual": "",
                                    "status": "insufficient_sample",
                                }
                            )
                            continue
                        if model_name == "raw_bop_midpoint":
                            predicted_residual = [0.0 for _ in holdout_rows]
                            fitted = None
                        else:
                            fitted = fit_model_for_target(train_rows, terms, "target_log_bop_residual")
                            predicted_residual = predict_log(fitted, holdout_rows)
                            for term, coef, center, scale in zip(
                                ["intercept"] + terms,
                                fitted.beta,
                                [0.0] + fitted.centers,
                                [1.0] + fitted.scales,
                            ):
                                coefficient_rows.append(
                                    {
                                        "model": model_name,
                                        "population": population,
                                        "timing_day": timing_day,
                                        "wiki_timing_day": timing_day,
                                        "competition_timing_day": competition_timing_day,
                                        "bop_timing_day": bop_timing_day,
                                        "term": term,
                                        "standardized_coef": coef,
                                        "center": center,
                                        "scale": scale,
                                        "train_n": len(train_rows),
                                    }
                                )
                        predicted_log = [
                            math.log(max(1.0, float(row["bop_forecast_midpoint"]))) + residual
                            for row, residual in zip(holdout_rows, predicted_residual)
                        ]
                        predicted_gross = [max(1.0, math.exp(value)) for value in predicted_log]
                        actual_log = [float(row["target_log_opening_weekend"]) for row in holdout_rows]
                        actual_gross = [float(row["opening_weekend_revenue_usd"]) for row in holdout_rows]
                        metric_rows.append(
                            {
                                **metric_row_from_predictions(
                                    base=metric_base,
                                    actual_log=actual_log,
                                    actual_gross=actual_gross,
                                    predicted_log=predicted_log,
                                    predicted_gross=predicted_gross,
                                ),
                                "mean_predicted_residual": format_number(mean(predicted_residual)),
                            }
                        )
                        for row, residual, log_value, gross_value in zip(
                            holdout_rows,
                            predicted_residual,
                            predicted_log,
                            predicted_gross,
                        ):
                            actual_gross = float(row["opening_weekend_revenue_usd"])
                            prediction_rows.append(
                                {
                                    "model": model_name,
                                    "population": population,
                                    "timing_day": timing_day,
                                    "wiki_timing_day": timing_day,
                                    "competition_timing_day": competition_timing_day,
                                    "bop_timing_day": bop_timing_day,
                                    "movie_id": row["movie_id"],
                                    "title": row["title"],
                                    "release_year": row["release_year"],
                                    "opening_date": row["opening_date"],
                                    "bop_forecast_midpoint": row["bop_forecast_midpoint"],
                                    "bop_q1_proxy": row.get("bop_q1_proxy", 0.0),
                                    "bop_q4_proxy": row.get("bop_q4_proxy", 0.0),
                                    "actual_log_bop_residual": row["target_log_bop_residual"],
                                    "predicted_log_bop_residual": residual,
                                    "actual_log_opening_weekend": row["target_log_opening_weekend"],
                                    "predicted_log_opening_weekend": log_value,
                                    "actual_opening_weekend_revenue_usd": row["opening_weekend_revenue_usd"],
                                    "predicted_opening_weekend_revenue_usd": gross_value,
                                    "absolute_percentage_error": (
                                        abs(gross_value - actual_gross) / actual_gross if actual_gross > 0.0 else ""
                                    ),
                                }
                            )
    return prediction_rows, metric_rows, coefficient_rows


def best_metric_rows(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best_by_model: dict[str, dict[str, object]] = {}
    for row in metric_rows:
        if row["status"] != "ok" or row["r2_log_revenue"] == "":
            continue
        current = best_by_model.get(str(row["model"]))
        if current is None or float(row["r2_log_revenue"]) > float(current["r2_log_revenue"]):
            best_by_model[str(row["model"])] = row
    return [
        {
            "model": row["model"],
            "best_timing_day": row["timing_day"],
            "best_wiki_timing_day": row["wiki_timing_day"],
            "best_competition_timing_day": row["competition_timing_day"],
            "r2_log_revenue": row["r2_log_revenue"],
            "mape_gross": row["mape_gross"],
            "r2_gross": row["r2_gross"],
            "train_n": row["train_n"],
            "holdout_n": row["holdout_n"],
        }
        for row in best_by_model.values()
    ]


def best_bop_metric_rows(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best_by_model: dict[str, dict[str, object]] = {}
    for row in metric_rows:
        if row["status"] != "ok" or row["r2_log_revenue"] == "":
            continue
        current = best_by_model.get(str(row["model"]))
        if current is None or float(row["r2_log_revenue"]) > float(current["r2_log_revenue"]):
            best_by_model[str(row["model"])] = row
    return [
        {
            "model": row["model"],
            "best_timing_day": row["timing_day"],
            "best_wiki_timing_day": row["wiki_timing_day"],
            "best_competition_timing_day": row.get("competition_timing_day", ""),
            "best_bop_timing_day": row["bop_timing_day"],
            "r2_log_revenue": row["r2_log_revenue"],
            "mape_gross": row["mape_gross"],
            "r2_gross": row["r2_gross"],
            "train_n": row["train_n"],
            "holdout_n": row["holdout_n"],
        }
        for row in best_by_model.values()
    ]


def best_multisource_metric_rows(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best_by_model_population: dict[tuple[str, str], dict[str, object]] = {}
    for row in metric_rows:
        if row["status"] != "ok" or row["mape_gross"] == "":
            continue
        key = (str(row["model"]), str(row["population"]))
        current = best_by_model_population.get(key)
        if current is None:
            best_by_model_population[key] = row
            continue
        row_key = (
            float(row["mape_gross"]),
            -float(row["r2_gross"]) if row["r2_gross"] != "" else float("inf"),
            -float(row["r2_log_revenue"]) if row["r2_log_revenue"] != "" else float("inf"),
        )
        current_key = (
            float(current["mape_gross"]),
            -float(current["r2_gross"]) if current["r2_gross"] != "" else float("inf"),
            -float(current["r2_log_revenue"]) if current["r2_log_revenue"] != "" else float("inf"),
        )
        if row_key < current_key:
            best_by_model_population[key] = row
    return [
        {
            "model": row["model"],
            "population": row["population"],
            "prediction_transform": row["prediction_transform"],
            "best_wiki_timing_day": row["wiki_timing_day"],
            "best_competition_timing_day": row["competition_timing_day"],
            "best_bop_timing_day": row["bop_timing_day"],
            "r2_log_revenue": row["r2_log_revenue"],
            "mape_gross": row["mape_gross"],
            "r2_gross": row["r2_gross"],
            "train_n": row["train_n"],
            "holdout_n": row["holdout_n"],
            "bop_prediction_n": row["bop_prediction_n"],
            "fallback_prediction_n": row["fallback_prediction_n"],
            "smearing_factor": row["smearing_factor"],
        }
        for row in best_by_model_population.values()
    ]


def best_residual_lift_metric_rows(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best_by_model_population: dict[tuple[str, str], dict[str, object]] = {}
    for row in metric_rows:
        if row["status"] != "ok" or row["mape_gross"] == "":
            continue
        key = (str(row["model"]), str(row["population"]))
        current = best_by_model_population.get(key)
        if current is None or (
            float(row["mape_gross"]),
            -float(row["r2_gross"]) if row["r2_gross"] != "" else float("inf"),
        ) < (
            float(current["mape_gross"]),
            -float(current["r2_gross"]) if current["r2_gross"] != "" else float("inf"),
        ):
            best_by_model_population[key] = row
    return [
        {
            "model": row["model"],
            "population": row["population"],
            "best_wiki_timing_day": row["wiki_timing_day"],
            "best_competition_timing_day": row["competition_timing_day"],
            "best_bop_timing_day": row["bop_timing_day"],
            "r2_log_revenue": row["r2_log_revenue"],
            "mape_gross": row["mape_gross"],
            "r2_gross": row["r2_gross"],
            "train_n": row["train_n"],
            "holdout_n": row["holdout_n"],
            "mean_predicted_residual": row["mean_predicted_residual"],
        }
        for row in best_by_model_population.values()
    ]


def coverage_rows(
    panel_rows: list[dict[str, object]],
    *,
    train_start_year: int = DEFAULT_TRAIN_START_YEAR,
    train_end_year: int = DEFAULT_TRAIN_END_YEAR,
    test_start_year: int = DEFAULT_TEST_START_YEAR,
    test_end_year: int = DEFAULT_TEST_END_YEAR,
) -> list[dict[str, object]]:
    rows = []
    by_day: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in panel_rows:
        by_day[
            (
                int(row.get("wiki_timing_day", row["timing_day"])),
                int(row.get("competition_timing_day", row["timing_day"])),
            )
        ].append(row)
    for (timing_day, competition_timing_day), group in sorted(by_day.items()):
        rows.append(
            {
                "timing_day": timing_day,
                "wiki_timing_day": timing_day,
                "competition_timing_day": competition_timing_day,
                "rows": len(group),
                "movies": len({row["movie_id"] for row in group}),
                "wiki_available_rows": sum(1 for row in group if float(row["wiki_available"]) > 0.0),
                "wiki_available_train_rows": sum(
                    1
                    for row in group
                    if train_start_year <= int(row["release_year"]) <= train_end_year
                    and float(row["wiki_available"]) > 0.0
                ),
                "wiki_available_holdout_rows": sum(
                    1
                    for row in group
                    if test_start_year <= int(row["release_year"]) <= test_end_year
                    and float(row["wiki_available"]) > 0.0
                ),
                "competitor_lag1_rows": sum(1 for row in group if float(row["competitor_count_lag1"]) > 0.0),
                "competitor_lag3_rows": sum(1 for row in group if float(row["competitor_count_lag3"]) > 0.0),
                "competitor_lag7_rows": sum(1 for row in group if float(row["competitor_count_lag7"]) > 0.0),
            }
        )
    return rows


def svg_escape(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def write_placeholder_svg(path: Path, title: str, message: str) -> None:
    path.write_text(
        "\n".join(
            [
                '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="320" viewBox="0 0 760 320">',
                '<rect width="100%" height="100%" fill="#fff"/>',
                f'<text x="36" y="44" font-family="Arial" font-size="20" font-weight="700">{svg_escape(title)}</text>',
                f'<text x="36" y="86" font-family="Arial" font-size="13">{svg_escape(message)}</text>',
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )


def write_metric_svg(path: Path, rows: list[dict[str, object]]) -> None:
    clean = [row for row in rows if row["status"] == "ok" and row["r2_log_revenue"] != ""]
    if not clean:
        write_placeholder_svg(path, "Competition opening-weekend metrics", "No model metrics were available.")
        return
    best = best_metric_rows(clean)
    width, height = 960, 540
    left, right, top, bottom = 260, 36, 58, 78
    values = [float(row["r2_log_revenue"]) for row in best]
    lo, hi = min(0.0, min(values)), max(values)
    if lo == hi:
        hi = lo + 1.0
    bar_h = min(34, (height - top - bottom) / max(1, len(best)) - 8)

    def sx(value: float) -> float:
        return left + (value - lo) / (hi - lo) * (width - left - right)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">Best holdout R2 by model</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333"/>',
    ]
    for idx, row in enumerate(best):
        y = top + idx * (bar_h + 8)
        x0, x1 = sx(0.0), sx(float(row["r2_log_revenue"]))
        parts.append(f'<text x="{left - 12}" y="{y + bar_h * 0.68:.1f}" text-anchor="end" font-family="Arial" font-size="12">{svg_escape(row["model"])}</text>')
        parts.append(f'<rect x="{min(x0, x1):.1f}" y="{y:.1f}" width="{abs(x1 - x0):.1f}" height="{bar_h:.1f}" fill="#386f6b"/>')
        parts.append(f'<text x="{max(x0, x1) + 6:.1f}" y="{y + bar_h * 0.68:.1f}" font-family="Arial" font-size="12">{float(row["r2_log_revenue"]):.2f} @ t={row["best_timing_day"]}</text>')
    parts.append(f'<text x="{(left + width - right) / 2:.1f}" y="{height - 26}" font-family="Arial" font-size="12" text-anchor="middle">Holdout R2 on log opening-weekend revenue</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_prediction_svg(path: Path, rows: list[dict[str, object]], metric_rows: list[dict[str, object]]) -> None:
    best = best_metric_rows(metric_rows)
    if not best:
        write_placeholder_svg(path, "Actual vs predicted opening weekend", "No holdout predictions were available.")
        return
    chosen = max(best, key=lambda row: float(row["r2_log_revenue"]))
    points = [
        row for row in rows if row["model"] == chosen["model"] and int(row["timing_day"]) == int(chosen["best_timing_day"])
    ]
    if not points:
        write_placeholder_svg(path, "Actual vs predicted opening weekend", "No predictions matched the best model.")
        return
    width, height = 760, 620
    left, right, top, bottom = 82, 38, 58, 82
    actual = [float(row["actual_opening_weekend_revenue_usd"]) for row in points]
    predicted = [float(row["predicted_opening_weekend_revenue_usd"]) for row in points]
    lo = math.floor(math.log10(max(1.0, min(actual + predicted))))
    hi = math.ceil(math.log10(max(actual + predicted)))
    if lo == hi:
        hi += 1

    def sx(value: float) -> float:
        return left + (math.log10(max(1.0, value)) - lo) / (hi - lo) * (width - left - right)

    def sy(value: float) -> float:
        return top + (hi - math.log10(max(1.0, value))) / (hi - lo) * (height - top - bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">Actual vs predicted: {svg_escape(chosen["model"])}</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333"/>',
        f'<line x1="{sx(10 ** lo):.1f}" y1="{sy(10 ** lo):.1f}" x2="{sx(10 ** hi):.1f}" y2="{sy(10 ** hi):.1f}" stroke="#888" stroke-dasharray="5 5"/>',
    ]
    for exp in range(lo, hi + 1):
        label = f"1e{exp}"
        x, y = sx(10**exp), sy(10**exp)
        parts.append(f'<text x="{x:.1f}" y="{height - 46}" text-anchor="middle" font-family="Arial" font-size="11">{label}</text>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{label}</text>')
    for row in points:
        parts.append(
            f'<circle cx="{sx(float(row["actual_opening_weekend_revenue_usd"])):.1f}" '
            f'cy="{sy(float(row["predicted_opening_weekend_revenue_usd"])):.1f}" r="4" fill="#574b90" fill-opacity="0.65"/>'
        )
    parts.append(f'<text x="{(left + width - right) / 2:.1f}" y="{height - 14}" font-family="Arial" font-size="12" text-anchor="middle">Actual opening weekend gross</text>')
    parts.append(f'<text x="18" y="{(top + height - bottom) / 2:.1f}" font-family="Arial" font-size="12" transform="rotate(-90 18 {(top + height - bottom) / 2:.1f})" text-anchor="middle">Predicted opening weekend gross</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_competition_residual_svg(path: Path, rows: list[dict[str, object]]) -> None:
    points = []
    for row in rows:
        if row["model"] != "wiki_plus_compact_competition":
            continue
        residual = math.log(float(row["actual_opening_weekend_revenue_usd"])) - math.log(
            float(row["predicted_opening_weekend_revenue_usd"])
        )
        pressure = log1p(float(row["competitor_total_gross_lag1"]))
        points.append((pressure, residual))
    if not points:
        write_placeholder_svg(path, "Competition pressure vs residual", "No combined-model residuals were available.")
        return
    width, height = 760, 460
    left, right, top, bottom = 76, 34, 52, 72
    xs, ys = [x for x, _ in points], [y for _, y in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax:
        xmax += 1.0
    if ymin == ymax:
        ymax += 1.0

    def sx(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * (width - left - right)

    def sy(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * (height - top - bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">Competition pressure vs residual</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333"/>',
    ]
    if ymin <= 0 <= ymax:
        parts.append(f'<line x1="{left}" y1="{sy(0):.1f}" x2="{width - right}" y2="{sy(0):.1f}" stroke="#aaa" stroke-dasharray="4 4"/>')
    for x, y in points:
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5" fill="#c15b3f" fill-opacity="0.6"/>')
    parts.append(f'<text x="{(left + width - right) / 2:.1f}" y="{height - 18}" font-family="Arial" font-size="12" text-anchor="middle">log1p(actual competitor gross, lag1)</text>')
    parts.append(f'<text x="18" y="{(top + height - bottom) / 2:.1f}" font-family="Arial" font-size="12" transform="rotate(-90 18 {(top + height - bottom) / 2:.1f})" text-anchor="middle">Actual - predicted log gross</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_coverage_svg(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        write_placeholder_svg(path, "Feature coverage by timing", "No coverage rows were available.")
        return
    width, height = 820, 440
    left, right, top, bottom = 74, 190, 52, 76
    series = [
        ("wiki_available_rows", "#2f6f73"),
        ("competitor_lag1_rows", "#c15b3f"),
        ("competitor_lag3_rows", "#4c78a8"),
        ("competitor_lag7_rows", "#574b90"),
    ]
    ymax = max(float(row[key]) for row in rows for key, _ in series) or 1.0
    days = [int(row["timing_day"]) for row in rows]
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
        '<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">Feature coverage by timing day</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333"/>',
    ]
    for idx, (key, color) in enumerate(series):
        points = " ".join(f'{sx(float(row["timing_day"])):.1f},{sy(float(row[key])):.1f}' for row in rows)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        y = top + 16 + idx * 24
        parts.append(f'<line x1="{width - right + 26}" y1="{y}" x2="{width - right + 50}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{width - right + 58}" y="{y + 4}" font-family="Arial" font-size="12">{svg_escape(key)}</text>')
    for row in rows:
        x = sx(float(row["timing_day"]))
        parts.append(f'<text x="{x:.1f}" y="{height - 44}" text-anchor="middle" font-family="Arial" font-size="11">{row["timing_day"]}</text>')
    parts.append(f'<text x="{(left + width - right) / 2:.1f}" y="{height - 16}" font-family="Arial" font-size="12" text-anchor="middle">Timing day before opening</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_bop_midpoint_svg(path: Path, rows: list[dict[str, object]]) -> None:
    points = [
        row
        for row in rows
        if float(row.get("bop_forecast_available", 0.0)) > 0.0
        and int(row.get("bop_timing_day", 999)) == -1
    ]
    if not points:
        write_placeholder_svg(path, "BOP midpoint vs actual", "No Boxoffice Pro midpoint rows were available.")
        return
    width, height = 760, 620
    left, right, top, bottom = 82, 38, 58, 82
    actual = [float(row["opening_weekend_revenue_usd"]) for row in points]
    midpoint = [float(row["bop_forecast_midpoint"]) for row in points]
    lo = math.floor(math.log10(max(1.0, min(actual + midpoint))))
    hi = math.ceil(math.log10(max(actual + midpoint)))
    if lo == hi:
        hi += 1

    def sx(value: float) -> float:
        return left + (math.log10(max(1.0, value)) - lo) / (hi - lo) * (width - left - right)

    def sy(value: float) -> float:
        return top + (hi - math.log10(max(1.0, value))) / (hi - lo) * (height - top - bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">Boxoffice Pro midpoint vs actual opening weekend</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333"/>',
        f'<line x1="{sx(10 ** lo):.1f}" y1="{sy(10 ** lo):.1f}" x2="{sx(10 ** hi):.1f}" y2="{sy(10 ** hi):.1f}" stroke="#888" stroke-dasharray="5 5"/>',
    ]
    for row in points:
        parts.append(
            f'<circle cx="{sx(float(row["opening_weekend_revenue_usd"])):.1f}" '
            f'cy="{sy(float(row["bop_forecast_midpoint"])):.1f}" r="4" fill="#2f6f73" fill-opacity="0.62"/>'
        )
    parts.append(f'<text x="{(left + width - right) / 2:.1f}" y="{height - 14}" font-family="Arial" font-size="12" text-anchor="middle">Actual opening weekend gross</text>')
    parts.append(f'<text x="18" y="{(top + height - bottom) / 2:.1f}" font-family="Arial" font-size="12" transform="rotate(-90 18 {(top + height - bottom) / 2:.1f})" text-anchor="middle">BOP forecast midpoint</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_bop_raw_error_svg(path: Path, rows: list[dict[str, object]]) -> None:
    raw_rows = [row for row in rows if row["model"] == "raw_midpoint"]
    if not raw_rows:
        write_placeholder_svg(path, "Raw midpoint error by timing", "No raw midpoint predictions were available.")
        return
    by_day: dict[int, list[float]] = defaultdict(list)
    for row in raw_rows:
        by_day[int(row["bop_timing_day"])].append(float(row["absolute_percentage_error"]))
    values = [(day, mean(errors)) for day, errors in sorted(by_day.items())]
    width, height = 760, 420
    left, right, top, bottom = 74, 34, 52, 72
    ymax = max(value for _, value in values) or 1.0
    bar_w = min(70, (width - left - right) / max(1, len(values)) - 18)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">Raw BOP midpoint MAPE by timing</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333"/>',
    ]
    for idx, (day, value) in enumerate(values):
        x = left + (idx + 0.5) * (width - left - right) / len(values)
        bar_h = value / ymax * (height - top - bottom)
        parts.append(f'<rect x="{x - bar_w / 2:.1f}" y="{height - bottom - bar_h:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="#c15b3f"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 44}" text-anchor="middle" font-family="Arial" font-size="11">t={day}</text>')
        parts.append(f'<text x="{x:.1f}" y="{height - bottom - bar_h - 6:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{value:.2f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_calibrated_vs_raw_svg(path: Path, metric_rows: list[dict[str, object]]) -> None:
    clean = [
        row
        for row in metric_rows
        if row["model"] in {"raw_midpoint", "calibrated_midpoint"}
        and row["status"] == "ok"
        and row["r2_log_revenue"] != ""
    ]
    if not clean:
        write_placeholder_svg(path, "Calibrated vs raw BOP", "No calibration metrics were available.")
        return
    clean = [{**row, "competition_timing_day": row.get("competition_timing_day", "")} for row in clean]
    write_metric_svg(path, clean)


def write_multisource_metric_svg(path: Path, rows: list[dict[str, object]]) -> None:
    best = [
        row
        for row in best_multisource_metric_rows(rows)
        if row["population"] in {"bop_covered", "full_with_fallback"}
    ]
    if not best:
        write_placeholder_svg(path, "Multisource opening-weekend metrics", "No multisource metrics were available.")
        return
    width, height = 1120, 620
    left, right, top, bottom = 360, 36, 58, 78
    values = [float(row["mape_gross"]) for row in best]
    hi = max(values) or 1.0
    bar_h = min(28, (height - top - bottom) / max(1, len(best)) - 7)

    def sx(value: float) -> float:
        return left + value / hi * (width - left - right)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">Best multisource MAPE by model/population</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333"/>',
    ]
    for idx, row in enumerate(best):
        y = top + idx * (bar_h + 7)
        label = f'{row["model"]} / {row["population"]} / {row["prediction_transform"]}'
        parts.append(f'<text x="{left - 12}" y="{y + bar_h * 0.68:.1f}" text-anchor="end" font-family="Arial" font-size="11">{svg_escape(label)}</text>')
        parts.append(f'<rect x="{left:.1f}" y="{y:.1f}" width="{sx(float(row["mape_gross"])) - left:.1f}" height="{bar_h:.1f}" fill="#386f6b"/>')
        parts.append(f'<text x="{sx(float(row["mape_gross"])) + 6:.1f}" y="{y + bar_h * 0.68:.1f}" font-family="Arial" font-size="11">{float(row["mape_gross"]):.2f}</text>')
    parts.append(f'<text x="{(left + width - right) / 2:.1f}" y="{height - 26}" font-family="Arial" font-size="12" text-anchor="middle">Holdout MAPE on opening-weekend gross</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_residual_lift_metric_svg(path: Path, rows: list[dict[str, object]]) -> None:
    best = best_residual_lift_metric_rows(rows)
    if not best:
        write_placeholder_svg(path, "BOP residual-lift metrics", "No residual-lift metrics were available.")
        return
    width, height = 980, 520
    left, right, top, bottom = 300, 36, 58, 78
    values = [float(row["mape_gross"]) for row in best]
    hi = max(values) or 1.0
    bar_h = min(30, (height - top - bottom) / max(1, len(best)) - 8)

    def sx(value: float) -> float:
        return left + value / hi * (width - left - right)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="36" y="34" font-family="Arial" font-size="20" font-weight="700">BOP residual-lift MAPE by model</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333"/>',
    ]
    for idx, row in enumerate(best):
        y = top + idx * (bar_h + 8)
        label = f'{row["model"]} / {row["population"]}'
        parts.append(f'<text x="{left - 12}" y="{y + bar_h * 0.68:.1f}" text-anchor="end" font-family="Arial" font-size="11">{svg_escape(label)}</text>')
        parts.append(f'<rect x="{left:.1f}" y="{y:.1f}" width="{sx(float(row["mape_gross"])) - left:.1f}" height="{bar_h:.1f}" fill="#574b90"/>')
        parts.append(f'<text x="{sx(float(row["mape_gross"])) + 6:.1f}" y="{y + bar_h * 0.68:.1f}" font-family="Arial" font-size="11">{float(row["mape_gross"]):.2f}</text>')
    parts.append(f'<text x="{(left + width - right) / 2:.1f}" y="{height - 26}" font-family="Arial" font-size="12" text-anchor="middle">Holdout MAPE on opening-weekend gross</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


FEATURE_PANEL_FIELDNAMES = [
    "movie_id",
    "title",
    "release_year",
    "release_run_id",
    "opening_date",
    "timing_day",
    "wiki_timing_day",
    "competition_timing_day",
    "as_of_date",
    "wiki_as_of_date",
    "competition_as_of_date",
    "competitor_previous_weekend_start",
    "competitor_previous_weekend_end",
    "opening_theaters",
    "opening_day_gross_usd",
    "opening_weekend_revenue_usd",
    "target_log_opening_weekend",
    "log1p_opening_theaters",
    "V",
    "U",
    "R",
    "E",
    "log1p_V",
    "log1p_U",
    "log1p_R",
    "log1p_E",
    "wiki_available",
    "competitor_total_gross_lag1",
    "competitor_top1_gross_lag1",
    "competitor_count_lag1",
    "competitor_hhi_lag1",
    "log1p_competitor_total_gross_lag1",
    "log1p_competitor_top1_gross_lag1",
    "competitor_total_gross_lag3",
    "competitor_top1_gross_lag3",
    "competitor_count_lag3",
    "competitor_hhi_lag3",
    "log1p_competitor_total_gross_lag3",
    "log1p_competitor_top1_gross_lag3",
    "competitor_total_gross_lag7",
    "competitor_top1_gross_lag7",
    "competitor_count_lag7",
    "competitor_hhi_lag7",
    "log1p_competitor_total_gross_lag7",
    "log1p_competitor_top1_gross_lag7",
    "competitor_total_gross_previous_weekend",
    "competitor_top1_gross_previous_weekend",
    "competitor_count_previous_weekend",
    "competitor_hhi_previous_weekend",
    "log1p_competitor_total_gross_previous_weekend",
    "log1p_competitor_top1_gross_previous_weekend",
    *[f"release_month_{month}" for month in range(2, 13)],
]

METRIC_FIELDNAMES = [
    "model",
    "timing_day",
    "wiki_timing_day",
    "competition_timing_day",
    "train_start_year",
    "train_end_year",
    "test_start_year",
    "test_end_year",
    "train_n",
    "holdout_n",
    "r2_log_revenue",
    "r2_gross",
    "mape_gross",
    "rmse_log_revenue",
    "mae_log_revenue",
    "mean_actual_gross",
    "mean_predicted_gross",
    "status",
]

PREDICTION_FIELDNAMES = [
    "model",
    "timing_day",
    "wiki_timing_day",
    "competition_timing_day",
    "movie_id",
    "title",
    "release_year",
    "opening_date",
    "actual_log_opening_weekend",
    "predicted_log_opening_weekend",
    "actual_opening_weekend_revenue_usd",
    "predicted_opening_weekend_revenue_usd",
    "absolute_percentage_error",
    "competitor_total_gross_lag1",
    "competitor_top1_gross_lag1",
    "competitor_total_gross_previous_weekend",
    "competitor_count_lag1",
]

COEFFICIENT_FIELDNAMES = [
    "model",
    "timing_day",
    "wiki_timing_day",
    "competition_timing_day",
    "term",
    "standardized_coef",
    "center",
    "scale",
    "train_n",
]

COVERAGE_FIELDNAMES = [
    "timing_day",
    "wiki_timing_day",
    "competition_timing_day",
    "rows",
    "movies",
    "wiki_available_rows",
    "wiki_available_train_rows",
    "wiki_available_holdout_rows",
    "competitor_lag1_rows",
    "competitor_lag3_rows",
    "competitor_lag7_rows",
]

HEADLINE_FIELDNAMES = [
    "model",
    "best_timing_day",
    "best_wiki_timing_day",
    "best_competition_timing_day",
    "r2_log_revenue",
    "mape_gross",
    "r2_gross",
    "train_n",
    "holdout_n",
]

BOP_FEATURE_PANEL_FIELDNAMES = FEATURE_PANEL_FIELDNAMES + [
    "bop_timing_day",
    "bop_as_of_date",
    "bop_forecast_available",
    "bop_prediction_id",
    "bop_article_url",
    "bop_forecast_published_date",
    "bop_forecast_midpoint",
    "log1p_bop_forecast_midpoint",
    "bop_forecast_range_width_pct",
    "bop_source_rank",
    "bop_showtime_market_share_pct",
    "bop_same_weekend_competitor_total",
    "bop_same_weekend_competitor_top1",
    "bop_same_weekend_competitor_count",
    "bop_same_weekend_competitor_hhi",
    "log1p_bop_same_weekend_competitor_total",
    "log1p_bop_same_weekend_competitor_top1",
    "bop_q1_threshold",
    "bop_q2_threshold",
    "bop_q3_threshold",
    "bop_q4_threshold",
    "bop_q1_proxy",
    "bop_q2_proxy",
    "bop_q3_proxy",
    "bop_q4_proxy",
    "bop_estimate_bucket_under_15m",
    "bop_estimate_bucket_15_30m",
    "bop_estimate_bucket_30_60m",
    "bop_estimate_bucket_60_100m",
    "bop_estimate_bucket_100m_plus",
    *BOP_Q4_INTERACTION_TERMS,
    "bop_q1_proxy_x_log1p_bop_forecast_midpoint",
    "bop_q4_proxy_x_log1p_bop_forecast_midpoint",
    "bop_q4_proxy_x_log1p_competitor_total_gross_lag1",
    "bop_q4_proxy_x_log1p_competitor_total_gross_lag7",
]

BOP_Q4_METRIC_FIELDNAMES = [
    "model",
    "timing_day",
    "wiki_timing_day",
    "competition_timing_day",
    "bop_timing_day",
    "train_start_year",
    "train_end_year",
    "test_start_year",
    "test_end_year",
    "train_n",
    "holdout_n",
    "r2_log_revenue",
    "r2_gross",
    "mape_gross",
    "rmse_log_revenue",
    "mae_log_revenue",
    "mean_actual_gross",
    "mean_predicted_gross",
    "status",
]

BOP_Q4_PREDICTION_FIELDNAMES = [
    "model",
    "timing_day",
    "wiki_timing_day",
    "competition_timing_day",
    "bop_timing_day",
    "movie_id",
    "title",
    "release_year",
    "opening_date",
    "actual_log_opening_weekend",
    "predicted_log_opening_weekend",
    "actual_opening_weekend_revenue_usd",
    "predicted_opening_weekend_revenue_usd",
    "absolute_percentage_error",
    "bop_forecast_midpoint",
    "bop_q4_proxy",
    "competitor_total_gross_lag1",
]

BOP_Q4_COEFFICIENT_FIELDNAMES = [
    "model",
    "timing_day",
    "wiki_timing_day",
    "competition_timing_day",
    "bop_timing_day",
    "term",
    "standardized_coef",
    "center",
    "scale",
    "train_n",
]

BOP_HEADLINE_FIELDNAMES = [
    "model",
    "best_timing_day",
    "best_wiki_timing_day",
    "best_competition_timing_day",
    "best_bop_timing_day",
    "r2_log_revenue",
    "mape_gross",
    "r2_gross",
    "train_n",
    "holdout_n",
]

BOP_CALIBRATION_METRIC_FIELDNAMES = [
    "model",
    "timing_day",
    "wiki_timing_day",
    "bop_timing_day",
    "train_start_year",
    "train_end_year",
    "test_start_year",
    "test_end_year",
    "train_n",
    "holdout_n",
    "r2_log_revenue",
    "r2_gross",
    "mape_gross",
    "rmse_log_revenue",
    "mae_log_revenue",
    "mean_actual_gross",
    "mean_predicted_gross",
    "calibration_intercept",
    "calibration_slope_log_midpoint",
    "status",
]

BOP_CALIBRATION_PREDICTION_FIELDNAMES = [
    "model",
    "timing_day",
    "wiki_timing_day",
    "bop_timing_day",
    "movie_id",
    "title",
    "release_year",
    "opening_date",
    "bop_forecast_midpoint",
    "actual_log_opening_weekend",
    "predicted_log_opening_weekend",
    "actual_opening_weekend_revenue_usd",
    "predicted_opening_weekend_revenue_usd",
    "absolute_percentage_error",
]

BOP_CORRELATION_FIELDNAMES = [
    "timing_day",
    "wiki_timing_day",
    "bop_timing_day",
    "holdout_n",
    "pearson_log_midpoint_actual",
    "spearman_midpoint_actual",
]

MULTISOURCE_METRIC_FIELDNAMES = [
    "model",
    "population",
    "prediction_transform",
    "timing_day",
    "wiki_timing_day",
    "competition_timing_day",
    "bop_timing_day",
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
    "smearing_factor",
    "status",
]

MULTISOURCE_PREDICTION_FIELDNAMES = [
    "model",
    "population",
    "prediction_transform",
    "prediction_source",
    "timing_day",
    "wiki_timing_day",
    "competition_timing_day",
    "bop_timing_day",
    "movie_id",
    "title",
    "release_year",
    "opening_date",
    "bop_forecast_available",
    "bop_forecast_midpoint",
    "bop_q1_proxy",
    "bop_q4_proxy",
    "actual_log_opening_weekend",
    "predicted_log_opening_weekend",
    "actual_opening_weekend_revenue_usd",
    "predicted_opening_weekend_revenue_usd",
    "absolute_percentage_error",
]

MULTISOURCE_COEFFICIENT_FIELDNAMES = [
    "model",
    "population",
    "timing_day",
    "wiki_timing_day",
    "competition_timing_day",
    "bop_timing_day",
    "term",
    "standardized_coef",
    "center",
    "scale",
    "train_n",
]

MULTISOURCE_HEADLINE_FIELDNAMES = [
    "model",
    "population",
    "prediction_transform",
    "best_wiki_timing_day",
    "best_competition_timing_day",
    "best_bop_timing_day",
    "r2_log_revenue",
    "mape_gross",
    "r2_gross",
    "train_n",
    "holdout_n",
    "bop_prediction_n",
    "fallback_prediction_n",
    "smearing_factor",
]

RESIDUAL_LIFT_METRIC_FIELDNAMES = [
    "model",
    "population",
    "timing_day",
    "wiki_timing_day",
    "competition_timing_day",
    "bop_timing_day",
    "train_start_year",
    "train_end_year",
    "test_start_year",
    "test_end_year",
    "train_n",
    "holdout_n",
    "r2_log_revenue",
    "r2_gross",
    "mape_gross",
    "rmse_log_revenue",
    "mae_log_revenue",
    "mean_actual_gross",
    "mean_predicted_gross",
    "mean_predicted_residual",
    "status",
]

RESIDUAL_LIFT_PREDICTION_FIELDNAMES = [
    "model",
    "population",
    "timing_day",
    "wiki_timing_day",
    "competition_timing_day",
    "bop_timing_day",
    "movie_id",
    "title",
    "release_year",
    "opening_date",
    "bop_forecast_midpoint",
    "bop_q1_proxy",
    "bop_q4_proxy",
    "actual_log_bop_residual",
    "predicted_log_bop_residual",
    "actual_log_opening_weekend",
    "predicted_log_opening_weekend",
    "actual_opening_weekend_revenue_usd",
    "predicted_opening_weekend_revenue_usd",
    "absolute_percentage_error",
]

RESIDUAL_LIFT_COEFFICIENT_FIELDNAMES = [
    "model",
    "population",
    "timing_day",
    "wiki_timing_day",
    "competition_timing_day",
    "bop_timing_day",
    "term",
    "standardized_coef",
    "center",
    "scale",
    "train_n",
]

RESIDUAL_LIFT_HEADLINE_FIELDNAMES = [
    "model",
    "population",
    "best_wiki_timing_day",
    "best_competition_timing_day",
    "best_bop_timing_day",
    "r2_log_revenue",
    "mape_gross",
    "r2_gross",
    "train_n",
    "holdout_n",
    "mean_predicted_residual",
]


def write_outputs(
    out_dir: Path,
    *,
    panel_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    coefficient_rows: list[dict[str, object]],
    coverage: list[dict[str, object]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "actual_competition_opening_weekend_feature_panel.csv", panel_rows, FEATURE_PANEL_FIELDNAMES)
    write_csv(out_dir / "actual_competition_opening_weekend_metrics.csv", metric_rows, METRIC_FIELDNAMES)
    write_csv(out_dir / "actual_competition_opening_weekend_predictions.csv", prediction_rows, PREDICTION_FIELDNAMES)
    write_csv(out_dir / "actual_competition_opening_weekend_coefficients.csv", coefficient_rows, COEFFICIENT_FIELDNAMES)
    write_csv(out_dir / "actual_competition_opening_weekend_coverage.csv", coverage, COVERAGE_FIELDNAMES)
    write_csv(out_dir / "actual_competition_opening_weekend_headline_metrics.csv", best_metric_rows(metric_rows), HEADLINE_FIELDNAMES)
    write_metric_svg(out_dir / "figure_actual_competition_opening_weekend_metrics.svg", metric_rows)
    write_prediction_svg(out_dir / "figure_actual_competition_opening_weekend_actual_vs_predicted.svg", prediction_rows, metric_rows)
    write_competition_residual_svg(out_dir / "figure_actual_competition_gross_vs_residual.svg", prediction_rows)
    write_coverage_svg(out_dir / "figure_actual_competition_opening_weekend_coverage.svg", coverage)


def write_bop_outputs(
    out_dir: Path,
    *,
    panel_rows: list[dict[str, object]],
    q4_prediction_rows: list[dict[str, object]],
    q4_metric_rows: list[dict[str, object]],
    q4_coefficient_rows: list[dict[str, object]],
    calibration_prediction_rows: list[dict[str, object]],
    calibration_metric_rows: list[dict[str, object]],
    correlation_rows: list[dict[str, object]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "bop_estimate_feature_panel.csv", panel_rows, BOP_FEATURE_PANEL_FIELDNAMES)
    write_csv(out_dir / "bop_q4_interaction_metrics.csv", q4_metric_rows, BOP_Q4_METRIC_FIELDNAMES)
    write_csv(out_dir / "bop_q4_interaction_predictions.csv", q4_prediction_rows, BOP_Q4_PREDICTION_FIELDNAMES)
    write_csv(out_dir / "bop_q4_interaction_coefficients.csv", q4_coefficient_rows, BOP_Q4_COEFFICIENT_FIELDNAMES)
    write_csv(out_dir / "bop_q4_interaction_headline_metrics.csv", best_bop_metric_rows(q4_metric_rows), BOP_HEADLINE_FIELDNAMES)
    write_csv(out_dir / "bop_estimate_calibration_metrics.csv", calibration_metric_rows, BOP_CALIBRATION_METRIC_FIELDNAMES)
    write_csv(
        out_dir / "bop_estimate_calibration_predictions.csv",
        calibration_prediction_rows,
        BOP_CALIBRATION_PREDICTION_FIELDNAMES,
    )
    write_csv(out_dir / "bop_estimate_correlation_summary.csv", correlation_rows, BOP_CORRELATION_FIELDNAMES)
    write_metric_svg(out_dir / "figure_bop_q4_interaction_model_comparison.svg", q4_metric_rows)
    write_bop_midpoint_svg(out_dir / "figure_bop_midpoint_vs_actual_opening_weekend.svg", panel_rows)
    write_bop_raw_error_svg(out_dir / "figure_bop_raw_midpoint_error_by_timing.svg", calibration_prediction_rows)
    write_calibrated_vs_raw_svg(out_dir / "figure_bop_calibrated_vs_raw_prediction_error.svg", calibration_metric_rows)


def write_multisource_outputs(
    out_dir: Path,
    *,
    prediction_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    coefficient_rows: list[dict[str, object]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        out_dir / "multisource_opening_weekend_metrics.csv",
        metric_rows,
        MULTISOURCE_METRIC_FIELDNAMES,
    )
    write_csv(
        out_dir / "multisource_opening_weekend_predictions.csv",
        prediction_rows,
        MULTISOURCE_PREDICTION_FIELDNAMES,
    )
    write_csv(
        out_dir / "multisource_opening_weekend_coefficients.csv",
        coefficient_rows,
        MULTISOURCE_COEFFICIENT_FIELDNAMES,
    )
    write_csv(
        out_dir / "multisource_opening_weekend_headline_metrics.csv",
        best_multisource_metric_rows(metric_rows),
        MULTISOURCE_HEADLINE_FIELDNAMES,
    )
    write_multisource_metric_svg(out_dir / "figure_multisource_opening_weekend_metrics.svg", metric_rows)


def write_residual_lift_outputs(
    out_dir: Path,
    *,
    prediction_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    coefficient_rows: list[dict[str, object]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        out_dir / "bop_residual_lift_metrics.csv",
        metric_rows,
        RESIDUAL_LIFT_METRIC_FIELDNAMES,
    )
    write_csv(
        out_dir / "bop_residual_lift_predictions.csv",
        prediction_rows,
        RESIDUAL_LIFT_PREDICTION_FIELDNAMES,
    )
    write_csv(
        out_dir / "bop_residual_lift_coefficients.csv",
        coefficient_rows,
        RESIDUAL_LIFT_COEFFICIENT_FIELDNAMES,
    )
    write_csv(
        out_dir / "bop_residual_lift_headline_metrics.csv",
        best_residual_lift_metric_rows(metric_rows),
        RESIDUAL_LIFT_HEADLINE_FIELDNAMES,
    )
    write_residual_lift_metric_svg(out_dir / "figure_bop_residual_lift_metrics.svg", metric_rows)


def run(args: argparse.Namespace) -> int:
    timing_days = parse_day_list(args.timing_days)
    competition_timing_days = parse_day_list(args.competition_timing_days) if args.competition_timing_days else timing_days
    bop_timing_days = parse_day_list(args.bop_timing_days)
    if args.train_start_year > args.train_end_year:
        raise SystemExit("--train-start-year must be <= --train-end-year")
    if args.test_start_year > args.test_end_year:
        raise SystemExit("--test-start-year must be <= --test-end-year")
    if not timing_days:
        raise SystemExit("--timing-days must include at least one day")
    if not competition_timing_days:
        raise SystemExit("--competition-timing-days must include at least one day when provided")
    if not bop_timing_days:
        raise SystemExit("--bop-timing-days must include at least one day")
    if args.min_opening_day_gross < 0:
        raise SystemExit("--min-opening-day-gross must be non-negative")

    min_year = min(args.train_start_year, args.test_start_year)
    max_year = max(args.train_end_year, args.test_end_year)
    conn = connect_database(args.database_url)
    try:
        movies = load_opening_weekend_movies(
            conn,
            min_year=min_year,
            max_year=max_year,
            min_opening_day_gross=args.min_opening_day_gross,
        )
        if not movies:
            raise SystemExit("No opening-weekend movies matched the requested cohort.")
        min_opening_date = min(movie.opening_date for movie in movies)
        max_opening_date = max(movie.opening_date for movie in movies)
        daily_grosses = load_daily_grosses(
            conn,
            start_date=min_opening_date + dt.timedelta(days=min(competition_timing_days) - 7),
            end_date=max_opening_date + dt.timedelta(days=max(competition_timing_days)),
        )
        wiki_by_movie = load_wiki_feature_map(
            conn,
            movies=movies,
            timing_days=timing_days,
        )
        bop_forecasts = load_boxofficepro_forecasts(
            conn,
            min_target_date=min_opening_date,
            max_target_date=max_opening_date,
        )
    finally:
        conn.close()

    panel_rows = build_feature_panel(
        movies,
        daily_grosses,
        wiki_by_movie,
        timing_days=timing_days,
        competition_timing_days=competition_timing_days,
    )
    prediction_rows, metric_rows, coefficient_rows = evaluate_time_split(
        panel_rows,
        timing_days=timing_days,
        competition_timing_days=competition_timing_days,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        test_start_year=args.test_start_year,
        test_end_year=args.test_end_year,
        require_wiki_available=not args.allow_missing_wiki,
    )
    coverage = coverage_rows(
        panel_rows,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        test_start_year=args.test_start_year,
        test_end_year=args.test_end_year,
    )
    out_dir = args.out or DEFAULT_OUT_DIR
    write_outputs(
        out_dir,
        panel_rows=panel_rows,
        prediction_rows=prediction_rows,
        metric_rows=metric_rows,
        coefficient_rows=coefficient_rows,
        coverage=coverage,
    )
    bop_panel_rows = build_bop_feature_panel(
        movies,
        daily_grosses,
        wiki_by_movie,
        bop_forecasts,
        timing_days=timing_days,
        competition_timing_days=competition_timing_days,
        bop_timing_days=bop_timing_days,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
    )
    q4_prediction_rows, q4_metric_rows, q4_coefficient_rows = evaluate_bop_q4_split(
        bop_panel_rows,
        timing_days=timing_days,
        competition_timing_days=competition_timing_days,
        bop_timing_days=bop_timing_days,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        test_start_year=args.test_start_year,
        test_end_year=args.test_end_year,
        require_wiki_available=not args.allow_missing_wiki,
    )
    calibration_prediction_rows, calibration_metric_rows, correlation_rows = evaluate_bop_calibration(
        bop_panel_rows,
        timing_days=timing_days,
        bop_timing_days=bop_timing_days,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        test_start_year=args.test_start_year,
        test_end_year=args.test_end_year,
        require_wiki_available_for_wiki_model=not args.allow_missing_wiki,
    )
    write_bop_outputs(
        out_dir,
        panel_rows=bop_panel_rows,
        q4_prediction_rows=q4_prediction_rows,
        q4_metric_rows=q4_metric_rows,
        q4_coefficient_rows=q4_coefficient_rows,
        calibration_prediction_rows=calibration_prediction_rows,
        calibration_metric_rows=calibration_metric_rows,
        correlation_rows=correlation_rows,
    )
    multisource_prediction_rows, multisource_metric_rows, multisource_coefficient_rows = evaluate_multisource_models(
        bop_panel_rows,
        panel_rows,
        timing_days=timing_days,
        competition_timing_days=competition_timing_days,
        bop_timing_days=bop_timing_days,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        test_start_year=args.test_start_year,
        test_end_year=args.test_end_year,
    )
    write_multisource_outputs(
        out_dir,
        prediction_rows=multisource_prediction_rows,
        metric_rows=multisource_metric_rows,
        coefficient_rows=multisource_coefficient_rows,
    )
    residual_prediction_rows, residual_metric_rows, residual_coefficient_rows = evaluate_residual_lift_models(
        bop_panel_rows,
        timing_days=timing_days,
        competition_timing_days=competition_timing_days,
        bop_timing_days=bop_timing_days,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        test_start_year=args.test_start_year,
        test_end_year=args.test_end_year,
    )
    write_residual_lift_outputs(
        out_dir,
        prediction_rows=residual_prediction_rows,
        metric_rows=residual_metric_rows,
        coefficient_rows=residual_coefficient_rows,
    )
    print(f"Built {len(panel_rows)} actual-competition opening-weekend feature rows for {len(movies)} movies.")
    print(f"Built {len(bop_panel_rows)} Boxoffice Pro estimate feature rows from {len(bop_forecasts)} forecast rows.")
    print(f"Train years: {args.train_start_year}-{args.train_end_year}; test years: {args.test_start_year}-{args.test_end_year}.")
    print(f"Wiki timing days: {','.join(str(day) for day in timing_days)}.")
    print(f"Competition timing days: {','.join(str(day) for day in competition_timing_days)}.")
    print(f"Boxoffice Pro timing days: {','.join(str(day) for day in bop_timing_days)}.")
    if not args.allow_missing_wiki:
        print("Evaluated only rows with Wikipedia activity available at each timing day.")
    print(
        "Wrote actual-competition, Boxoffice Pro, multisource, and residual-lift "
        f"opening-weekend artifacts to {out_dir}."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Opening-weekend regression with pre-release competition dynamics.")
    parser.add_argument("--database-url", default=database_url_from_env())
    parser.add_argument("--out", type=Path)
    parser.add_argument("--timing-days", default=",".join(str(day) for day in DEFAULT_TIMING_DAYS))
    parser.add_argument(
        "--competition-timing-days",
        help="Comma-separated competitor as-of days before opening. Defaults to --timing-days.",
    )
    parser.add_argument("--bop-timing-days", default=",".join(str(day) for day in DEFAULT_BOP_TIMING_DAYS))
    parser.add_argument("--train-start-year", type=int, default=DEFAULT_TRAIN_START_YEAR)
    parser.add_argument("--train-end-year", type=int, default=DEFAULT_TRAIN_END_YEAR)
    parser.add_argument("--test-start-year", type=int, default=DEFAULT_TEST_START_YEAR)
    parser.add_argument("--test-end-year", type=int, default=DEFAULT_TEST_END_YEAR)
    parser.add_argument("--min-opening-day-gross", type=int, default=DEFAULT_MIN_OPENING_DAY_GROSS)
    parser.add_argument(
        "--allow-missing-wiki",
        action="store_true",
        help="Score rows with no Wikipedia activity by filling Wikipedia features with zero.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
