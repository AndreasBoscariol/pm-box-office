from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from dataclasses import dataclass, field
from html import escape
from typing import Any

from pm_box_office.research.papers import recreate_competition_opening_weekend as opening
from pm_box_office.research.papers import recreate_day_by_day_opening_weekend as day_by_day
from pm_box_office.models.opening_weekend import (
    DEFAULT_REGISTRY,
    OpeningWindowPredictionEngine,
    feature_snapshot_from_panel_row,
)
from pm_box_office.models.opening_weekend.targets import target_day_count_from_type, target_type_for_day_count


DEFAULT_TRAIN_START_YEAR = 2022
DEFAULT_TRAIN_END_YEAR = 2025
DEFAULT_MIN_OPENING_DAY_GROSS = 5_000_000
DEFAULT_WIKI_MIN_OPENING_DAY_GROSS = 2_000_000
DEFAULT_COMPETITION_RESIDUAL_SWITCH_DAY = -1
OPENING_WEEKEND_OFFSETS = (0, 1, 2)
FORECAST_SNAPSHOT_DAYS = tuple(range(-14, 6))
SNAPSHOT_DAYS = list(FORECAST_SNAPSHOT_DAYS)
SNAPSHOT_DAYS_BY_TARGET_TYPE = {
    "3_day": list(range(-14, 4)),
    "4_day": list(range(-14, 5)),
    "5_day": list(range(-14, 6)),
}
PRE_RELEASE_WIKI_RESIDUAL_LIVE_DAYS = {-2, -1}
PRE_RELEASE_POOL_WINDOWS = (
    tuple(range(-14, -7)),
    tuple(range(-7, -2)),
    (-2, -1),
)
PRIMARY_INTERVAL_METHOD = "loo_conformal_abs"
RAW_MODEL = "raw_bop_snapshot"
RESIDUAL_MODEL = "bop_residual_wiki_competition_snapshot"
WIKI_RESIDUAL_MODEL = "bop_residual_wiki_snapshot"
HISTORY_RESIDUAL_MODEL = "bop_residual_wiki_history_snapshot"
COMPETITION_RESIDUAL_MODEL = "bop_residual_competition_snapshot"
BUCKET_RESIDUAL_MODEL = "bop_residual_bucket_snapshot"
CALIBRATED_MODEL = "calibrated_bop_snapshot"
AMC_RESIDUAL_MODEL = "bop_residual_amc_snapshot"
FALLBACK_MODEL = "fallback_wiki_snapshot"
PRE_RELEASE_WIKI_RESIDUAL_FALLBACK_TERMS = ["log1p_V"]
FALLBACK_TERMS = opening.WIKI_TERMS
AMC_RESIDUAL_TERMS = [
    "log1p_bop_forecast_midpoint",
    "bop_forecast_range_width_pct",
    "amc_log1p_occupied_proxy",
    "amc_snapshot_coverage",
    "amc_premium_format_share",
]
AMC_FEATURE_DEFAULTS = {
    "amc_available": 0.0,
    "amc_cutoff": "",
    "amc_occupied_proxy": 0.0,
    "amc_capacity": 0.0,
    "amc_occupancy": 0.0,
    "amc_showtime_count": 0.0,
    "amc_theatre_count": 0.0,
    "amc_snapshot_coverage": 0.0,
    "amc_late_snapshot_count": 0.0,
    "amc_failed_snapshot_count": 0.0,
    "amc_premium_format_share": 0.0,
    "amc_log1p_occupied_proxy": 0.0,
}

DEPLOYED_ENGINE = OpeningWindowPredictionEngine(DEFAULT_REGISTRY)
MIN_INTERVAL_WIDTH_PCT_BY_STATE = {
    "pre_release": 0.35,
    "same_day_proxy_available": 0.30,
    "official_actuals_available": 0.20,
    "final": 0.0,
}
TARGET_INTERVAL_WIDTH_MULTIPLIER = {
    "3_day": 1.0,
    "4_day": 1.10,
    "5_day": 1.20,
}


@dataclass(frozen=True)
class ForecastCandidate:
    movie_id: int
    title: str
    release_year: int | None
    opening_date: dt.date
    target_start_date: dt.date
    target_end_date: dt.date
    target_day_count: int
    target_type: str
    latest_forecast_date: dt.date
    latest_bop_midpoint: float
    source_movie_title: str
    bop_target_start_date: dt.date | None = None
    bop_target_end_date: dt.date | None = None
    bop_target_day_count: int | None = None
    bop_target_type: str | None = None
    manual_target_override: bool = False


@dataclass(frozen=True)
class ForecastDayModel:
    snapshot_day: int
    model_name: str
    terms: list[str]
    train_rows: list[dict[str, object]]
    fitted: opening.FittedModel | None
    bucket_residuals: dict[str, list[float]]
    global_residuals: list[float]


@dataclass(frozen=True)
class WeekendRemainderModel:
    locked_offsets: tuple[int, ...]
    train_rows: int
    mean_remaining_to_known: float
    mean_known_share: float


@dataclass(frozen=True)
class ForecastModelBundle:
    train_start_year: int
    train_end_year: int
    min_opening_day_gross: int
    models: dict[tuple[int, str], ForecastDayModel]
    wiki_min_opening_day_gross: int = DEFAULT_WIKI_MIN_OPENING_DAY_GROSS
    remainder_models: dict[tuple[int, ...], WeekendRemainderModel] = field(default_factory=dict)


_MODEL_CACHE: dict[tuple[int, int, int, int], ForecastModelBundle] = {}


def clear_model_cache() -> None:
    _MODEL_CACHE.clear()


def parse_db_date(value: object) -> dt.date:
    return opening.parse_db_date(value)


def money(value: float | int | None) -> str:
    if value is None:
        return "-"
    amount = float(value)
    if amount >= 1_000_000:
        return f"${amount / 1_000_000:.1f}M"
    if amount >= 1_000:
        return f"${amount / 1_000:.0f}K"
    return f"${amount:.0f}"


def prediction_certainty(
    prediction_usd: float | int | None,
    lower_usd: float | int | None,
    upper_usd: float | int | None,
    *,
    is_final_actual: bool = False,
) -> dict[str, object]:
    if is_final_actual:
        return {
            "prediction_certainty": 1.0,
            "prediction_certainty_pct": 100,
            "prediction_certainty_label": "100%",
            "prediction_uncertainty_width_pct": 0.0,
        }
    if prediction_usd is None or lower_usd is None or upper_usd is None:
        return {
            "prediction_certainty": None,
            "prediction_certainty_pct": None,
            "prediction_certainty_label": "-",
            "prediction_uncertainty_width_pct": None,
        }
    prediction = float(prediction_usd)
    if prediction <= 0.0:
        return {
            "prediction_certainty": None,
            "prediction_certainty_pct": None,
            "prediction_certainty_label": "-",
            "prediction_uncertainty_width_pct": None,
        }
    width_pct = max(0.0, float(upper_usd) - float(lower_usd)) / prediction
    certainty = max(0.0, min(1.0, 1.0 / (1.0 + width_pct)))
    pct = int(round(certainty * 100))
    return {
        "prediction_certainty": certainty,
        "prediction_certainty_pct": pct,
        "prediction_certainty_label": f"{pct}%",
        "prediction_uncertainty_width_pct": width_pct,
    }


def prediction_confidence(
    prediction_usd: float | int | None,
    lower_usd: float | int | None,
    upper_usd: float | int | None,
    *,
    interval_train_n: int,
    interval_source: str,
    interval_method: str,
    is_final_actual: bool = False,
) -> dict[str, object]:
    if not is_final_actual and interval_train_n <= 0:
        certainty = prediction_certainty(None, None, None)
    else:
        certainty = prediction_certainty(
            prediction_usd,
            lower_usd,
            upper_usd,
            is_final_actual=is_final_actual,
        )
    source = "actual" if is_final_actual else (interval_source if interval_train_n > 0 else "unavailable")
    return {
        **certainty,
        "prediction_confidence_label": certainty["prediction_certainty_label"],
        "prediction_confidence_source": source,
        "prediction_confidence_method": interval_method,
        "prediction_confidence_train_n": interval_train_n,
    }


def search_forecast_movies(conn: Any, *, query: str = "", limit: int = 12) -> list[dict[str, object]]:
    pattern = f"%{query.strip()}%"
    rows = conn.execute(
        """
        WITH latest AS (
            SELECT DISTINCT ON (p.matched_movie_id)
                p.matched_movie_id,
                m.title,
                m.release_year,
                p.source_movie_title,
                p.target_start_date::date AS opening_date,
                p.target_start_date::date AS target_start_date,
                p.target_end_date::date AS target_end_date,
                (p.target_end_date::date - p.target_start_date::date + 1)::integer AS target_day_count,
                a.discovered_date::date AS latest_forecast_date,
                ((p.range_low_usd + p.range_high_usd) / 2.0)::double precision AS latest_bop_midpoint
            FROM boxofficepro_weekend_predictions p
            JOIN boxofficepro_articles a ON a.article_id = p.article_id
            JOIN movies m ON m.movie_id = p.matched_movie_id
            WHERE p.matched_movie_id IS NOT NULL
              AND p.forecast_metric = 'domestic_opening_weekend'
              AND p.target_start_date IS NOT NULL
              AND p.target_end_date IS NOT NULL
              AND p.range_low_usd > 0
              AND p.range_high_usd > 0
              AND (
                    %s = ''
                 OR m.title ILIKE %s
                 OR p.source_movie_title ILIKE %s
              )
            ORDER BY
                p.matched_movie_id,
                (p.target_end_date::date - p.target_start_date::date + 1)::integer DESC,
                a.discovered_date DESC NULLS LAST,
                p.article_id DESC,
                p.prediction_id DESC
        )
        SELECT matched_movie_id, title, release_year, source_movie_title,
               opening_date, target_start_date, target_end_date, target_day_count,
               latest_forecast_date, latest_bop_midpoint
        FROM latest
        ORDER BY opening_date DESC, title
        LIMIT %s
        """,
        (query.strip(), pattern, pattern, limit),
    ).fetchall()
    return [
        {
            "movie_id": int(row[0]),
            "title": str(row[1]),
            "release_year": int(row[2]) if row[2] is not None else None,
            "source_movie_title": str(row[3]),
            "opening_date": parse_db_date(row[4]).isoformat(),
            "target_start_date": parse_db_date(row[5]).isoformat(),
            "target_end_date": parse_db_date(row[6]).isoformat(),
            "target_days": int(row[7]),
            "target_type": target_type_for_day_count(int(row[7])),
            "latest_forecast_date": parse_db_date(row[8]).isoformat(),
            "latest_bop_midpoint": float(row[9]),
            "latest_bop_midpoint_label": money(float(row[9])),
        }
        for row in rows
    ]


def snapshot_days_for_target_type(target_type: str) -> list[int]:
    return SNAPSHOT_DAYS_BY_TARGET_TYPE.get(target_type, list(range(-14, target_day_count_from_type(target_type) + 1)))


def available_target_types_for_movie(conn: Any, movie_id: int) -> list[str]:
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT (p.target_end_date::date - p.target_start_date::date + 1)::integer AS target_day_count
            FROM boxofficepro_weekend_predictions p
            WHERE p.matched_movie_id = %s
              AND p.forecast_metric = 'domestic_opening_weekend'
              AND p.target_start_date IS NOT NULL
              AND p.target_end_date IS NOT NULL
              AND p.range_low_usd > 0
              AND p.range_high_usd > 0
            ORDER BY target_day_count
            """,
            (movie_id,),
        ).fetchall()
    except Exception:
        return []
    try:
        return [target_type_for_day_count(int(row[0])) for row in rows]
    except TypeError:
        return []


def latest_candidate_for_movie(
    conn: Any,
    movie_id: int,
    *,
    target_type: str | None = None,
    target_start_date: dt.date | None = None,
) -> ForecastCandidate | None:
    requested_target_day_count = target_day_count_from_type(target_type) if target_type else None

    def fetch_candidate(target_day_count_filter: int | None) -> tuple[object, ...] | None:
        return conn.execute(
            """
            SELECT
                p.matched_movie_id,
                m.title,
                m.release_year,
                p.source_movie_title,
                p.target_start_date::date AS opening_date,
                p.target_start_date::date AS target_start_date,
                p.target_end_date::date AS target_end_date,
                (p.target_end_date::date - p.target_start_date::date + 1)::integer AS target_day_count,
                a.discovered_date::date AS latest_forecast_date,
                ((p.range_low_usd + p.range_high_usd) / 2.0)::double precision AS latest_bop_midpoint
            FROM boxofficepro_weekend_predictions p
            JOIN boxofficepro_articles a ON a.article_id = p.article_id
            JOIN movies m ON m.movie_id = p.matched_movie_id
            WHERE p.matched_movie_id = %s
              AND p.forecast_metric = 'domestic_opening_weekend'
              AND p.target_start_date IS NOT NULL
              AND p.target_end_date IS NOT NULL
              AND p.range_low_usd > 0
              AND p.range_high_usd > 0
              AND (%s::integer IS NULL OR (p.target_end_date::date - p.target_start_date::date + 1)::integer = %s::integer)
            ORDER BY
                (p.target_end_date::date - p.target_start_date::date + 1)::integer DESC,
                a.discovered_date DESC NULLS LAST,
                p.article_id DESC,
                p.prediction_id DESC
            LIMIT 1
            """,
            (movie_id, target_day_count_filter, target_day_count_filter),
        ).fetchone()

    rows = fetch_candidate(requested_target_day_count)
    if rows is None and requested_target_day_count is not None:
        rows = fetch_candidate(None)
    if rows is None:
        return None

    bop_target_start_date = parse_db_date(rows[5])
    bop_target_end_date = parse_db_date(rows[6])
    bop_target_day_count = int(rows[7])
    target_day_count = requested_target_day_count or bop_target_day_count
    selected_target_type = target_type or target_type_for_day_count(target_day_count)
    selected_target_start_date = target_start_date or bop_target_start_date
    selected_target_end_date = selected_target_start_date + dt.timedelta(days=target_day_count - 1)
    manual_target_override = (
        selected_target_start_date != bop_target_start_date
        or selected_target_end_date != bop_target_end_date
        or target_day_count != bop_target_day_count
    )
    return ForecastCandidate(
        movie_id=int(rows[0]),
        title=str(rows[1]),
        release_year=int(rows[2]) if rows[2] is not None else None,
        source_movie_title=str(rows[3]),
        opening_date=selected_target_start_date,
        target_start_date=selected_target_start_date,
        target_end_date=selected_target_end_date,
        target_day_count=target_day_count,
        target_type=selected_target_type,
        latest_forecast_date=parse_db_date(rows[8]),
        latest_bop_midpoint=float(rows[9]),
        bop_target_start_date=bop_target_start_date,
        bop_target_end_date=bop_target_end_date,
        bop_target_day_count=bop_target_day_count,
        bop_target_type=target_type_for_day_count(bop_target_day_count),
        manual_target_override=manual_target_override,
    )


def load_movie_actual(
    conn: Any,
    *,
    movie_id: int,
    opening_date: dt.date,
    target_day_count: int = 3,
) -> dict[str, object]:
    row = conn.execute(
        """
        SELECT
            rr.release_run_id,
            opening_day.theaters,
            opening_day.gross_usd,
            SUM(weekend.gross_usd) AS opening_weekend_revenue_usd
        FROM release_runs rr
        JOIN daily_box_office opening_day
          ON opening_day.release_run_id = rr.release_run_id
         AND opening_day.box_office_date::date = %s::date
         AND opening_day.is_preview = 0
        JOIN daily_box_office weekend
          ON weekend.release_run_id = rr.release_run_id
         AND weekend.is_preview = 0
         AND weekend.box_office_date::date >= %s::date
         AND weekend.box_office_date::date < %s::date + (%s::integer * INTERVAL '1 day')
        WHERE rr.movie_id = %s
        GROUP BY rr.release_run_id, opening_day.theaters, opening_day.gross_usd
        ORDER BY rr.release_run_id
        LIMIT 1
        """,
        (
            opening_date.isoformat(),
            opening_date.isoformat(),
            opening_date.isoformat(),
            target_day_count,
            movie_id,
        ),
    ).fetchone()
    if row is None:
        return {
            "release_run_id": 0,
            "opening_theaters": 0,
            "opening_day_gross_usd": 0,
            "opening_weekend_revenue_usd": 0,
            "has_actual": False,
        }
    return {
        "release_run_id": int(row[0]),
        "opening_theaters": int(row[1] or 0),
        "opening_day_gross_usd": int(row[2] or 0),
        "opening_weekend_revenue_usd": int(row[3] or 0),
        "has_actual": True,
    }


def load_opening_weekend_daily_actuals(
    conn: Any,
    *,
    movie_id: int,
    opening_date: dt.date,
    target_day_count: int = 3,
) -> list[dict[str, object]]:
    rows = conn.execute(
        """
        SELECT
            (dbo.box_office_date::date - %s::date)::integer AS weekend_offset,
            dbo.box_office_date::date,
            dbo.gross_usd,
            dbo.is_estimate,
            dbo.source_url,
            dbo.fetched_at
        FROM release_runs rr
        JOIN daily_box_office dbo ON dbo.release_run_id = rr.release_run_id
        WHERE rr.movie_id = %s
          AND dbo.source = 'the_numbers'
          AND dbo.is_preview = 0
          AND dbo.gross_usd IS NOT NULL
          AND dbo.gross_usd > 0
          AND dbo.box_office_date::date >= %s::date
          AND dbo.box_office_date::date < %s::date + (%s::integer * INTERVAL '1 day')
        ORDER BY dbo.box_office_date::date, dbo.is_estimate ASC
        """,
        (opening_date.isoformat(), movie_id, opening_date.isoformat(), opening_date.isoformat(), target_day_count),
    ).fetchall()
    by_offset: dict[int, dict[str, object]] = {}
    for row in rows:
        offset = int(row[0])
        if offset not in tuple(range(target_day_count)):
            continue
        gross = float(row[2] or 0.0)
        is_estimate = int(row[3] or 0) != 0
        current = by_offset.get(offset)
        if current is not None and bool(current["is_actual"]):
            continue
        by_offset[offset] = {
            "offset": offset,
            "box_office_date": parse_db_date(row[1]).isoformat(),
            "gross_usd": gross,
            "gross_label": money(gross),
            "is_actual": not is_estimate,
            "is_estimate": is_estimate,
            "status": "estimate" if is_estimate else "actual",
            "source_url": str(row[4] or ""),
            "fetched_at": str(row[5] or ""),
        }
    actuals: list[dict[str, object]] = []
    for offset in tuple(range(target_day_count)):
        day = opening_date + dt.timedelta(days=offset)
        actuals.append(
            by_offset.get(
                offset,
                {
                    "offset": offset,
                    "box_office_date": day.isoformat(),
                    "gross_usd": None,
                    "gross_label": "-",
                    "is_actual": False,
                    "is_estimate": False,
                    "status": "missing",
                    "source_url": "",
                    "fetched_at": "",
                },
            )
        )
    return actuals


def weekend_actual_totals(
    weekend_actuals: list[dict[str, object]],
    *,
    snapshot_day: int | None = None,
    target_day_count: int = 3,
) -> dict[str, object]:
    target_offsets = tuple(range(target_day_count))
    locked: list[dict[str, object]] = []
    estimates: list[dict[str, object]] = []
    for row in weekend_actuals:
        offset = int(row["offset"])
        is_visible_at_snapshot = snapshot_day is None or offset < snapshot_day or snapshot_day >= target_day_count
        if not is_visible_at_snapshot:
            continue
        if bool(row.get("is_actual")):
            locked.append(row)
        elif bool(row.get("is_estimate")):
            estimates.append(row)
    known = sum(float(row["gross_usd"] or 0.0) for row in locked)
    estimate = sum(float(row["gross_usd"] or 0.0) for row in estimates)
    locked_offsets = tuple(sorted(int(row["offset"]) for row in locked))
    return {
        "locked_offsets": locked_offsets,
        "actual_weekend_gross_known_usd": known,
        "actual_weekend_gross_known_label": money(known) if known > 0.0 else "-",
        "estimated_weekend_gross_usd": estimate,
        "estimated_weekend_gross_label": money(estimate) if estimate > 0.0 else "-",
        "actual_day_count": len(locked),
        "estimate_day_count": len(estimates),
        "is_complete": locked_offsets == target_offsets,
    }


def train_weekend_remainder_models(
    movies: list[opening.OpeningWeekendMovie],
    daily_grosses: list[opening.DailyGross],
) -> dict[tuple[int, ...], WeekendRemainderModel]:
    gross_by_movie_date = {(row.movie_id, row.box_office_date): float(row.gross_usd) for row in daily_grosses}
    ratios: dict[tuple[int, ...], list[tuple[float, float]]] = defaultdict(list)
    for movie in movies:
        daily = {
            offset: gross_by_movie_date.get((movie.movie_id, movie.opening_date + dt.timedelta(days=offset)), 0.0)
            for offset in OPENING_WEEKEND_OFFSETS
        }
        total = sum(daily.values())
        if total <= 0.0 or any(daily[offset] <= 0.0 for offset in OPENING_WEEKEND_OFFSETS):
            continue
        for locked_offsets in ((0,), (0, 1)):
            known = sum(daily[offset] for offset in locked_offsets)
            remaining = max(0.0, total - known)
            if known > 0.0:
                ratios[locked_offsets].append((remaining / known, known / total))
    return {
        locked_offsets: WeekendRemainderModel(
            locked_offsets=locked_offsets,
            train_rows=len(values),
            mean_remaining_to_known=sum(value[0] for value in values) / len(values),
            mean_known_share=sum(value[1] for value in values) / len(values),
        )
        for locked_offsets, values in ratios.items()
        if values
    }


def load_amc_feature_map(
    conn: Any,
    *,
    movie_id: int,
    opening_date: dt.date,
) -> dict[int, dict[str, object]]:
    try:
        view_row = conn.execute("SELECT to_regclass('analytics.amc_movie_day_blocks_v1')").fetchone()
        if view_row is None or view_row[0] is None:
            return {}
        rows = conn.execute(
            """
            SELECT
                (b.exhibition_date::date - %s::date)::integer AS weekend_offset,
                b.s1_occupied_proxy,
                b.c1_capacity,
                b.o1_occupancy,
                b.s2_occupied_proxy,
                b.c2_capacity,
                b.o2_occupancy,
                b.s3_occupied_proxy,
                b.c3_capacity,
                b.o3_occupancy,
                b.s4_occupied_proxy,
                b.c4_capacity,
                b.o4_occupancy,
                b.full_day_showtime_count,
                b.movie_theatre_count,
                b.snapshot_coverage,
                b.late_snapshot_count,
                b.failed_snapshot_count,
                b.premium_format_share
            FROM analytics.amc_movie_day_blocks_v1 b
            JOIN movie_source_ids src
              ON src.source = 'amc'
             AND src.source_movie_id = b.amc_movie_id
            WHERE src.movie_id = %s
              AND b.exhibition_date::date >= %s::date
              AND b.exhibition_date::date < %s::date + INTERVAL '3 days'
            ORDER BY b.exhibition_date::date
            """,
            (opening_date.isoformat(), movie_id, opening_date.isoformat(), opening_date.isoformat()),
        ).fetchall()
    except Exception:
        return {}
    features: dict[int, dict[str, object]] = {}
    try:
        for row in rows:
            offset = int(row[0])
            occupied = sum(float(row[index] or 0.0) for index in (1, 4, 7, 10))
            capacity = sum(float(row[index] or 0.0) for index in (2, 5, 8, 11))
            occupancies = [float(row[index] or 0.0) for index in (3, 6, 9, 12) if row[index] is not None]
            features[offset] = {
                "amc_available": 1.0 if capacity > 0.0 else 0.0,
                "amc_cutoff": "full_day_blocks" if capacity > 0.0 else "",
                "amc_occupied_proxy": occupied,
                "amc_capacity": capacity,
                "amc_occupancy": sum(occupancies) / len(occupancies) if occupancies else 0.0,
                "amc_showtime_count": float(row[13] or 0.0),
                "amc_theatre_count": float(row[14] or 0.0),
                "amc_snapshot_coverage": float(row[15] or 0.0),
                "amc_late_snapshot_count": float(row[16] or 0.0),
                "amc_failed_snapshot_count": float(row[17] or 0.0),
                "amc_premium_format_share": float(row[18] or 0.0),
                "amc_log1p_occupied_proxy": opening.log1p(occupied),
            }
    except Exception:
        return {}
    return features


def amc_features_for_snapshot(amc_by_offset: dict[int, dict[str, object]], snapshot_day: int) -> dict[str, object]:
    if snapshot_day < 0:
        return dict(AMC_FEATURE_DEFAULTS)
    eligible_offsets = [offset for offset in OPENING_WEEKEND_OFFSETS if offset <= min(snapshot_day, 2)]
    available = [
        amc_by_offset[offset]
        for offset in eligible_offsets
        if float(amc_by_offset.get(offset, {}).get("amc_available", 0.0) or 0.0) > 0.0
    ]
    if not available:
        return dict(AMC_FEATURE_DEFAULTS)
    occupied = sum(float(row["amc_occupied_proxy"] or 0.0) for row in available)
    capacity = sum(float(row["amc_capacity"] or 0.0) for row in available)
    return {
        "amc_available": 1.0,
        "amc_cutoff": "weekend_to_date",
        "amc_occupied_proxy": occupied,
        "amc_capacity": capacity,
        "amc_occupancy": occupied / capacity if capacity > 0.0 else 0.0,
        "amc_showtime_count": sum(float(row["amc_showtime_count"] or 0.0) for row in available),
        "amc_theatre_count": max(float(row["amc_theatre_count"] or 0.0) for row in available),
        "amc_snapshot_coverage": sum(float(row["amc_snapshot_coverage"] or 0.0) for row in available) / len(available),
        "amc_late_snapshot_count": sum(float(row["amc_late_snapshot_count"] or 0.0) for row in available),
        "amc_failed_snapshot_count": sum(float(row["amc_failed_snapshot_count"] or 0.0) for row in available),
        "amc_premium_format_share": sum(float(row["amc_premium_format_share"] or 0.0) for row in available)
        / len(available),
        "amc_log1p_occupied_proxy": opening.log1p(occupied),
    }


def opening_day_gross_at_least(row: dict[str, object], min_opening_day_gross: int) -> bool:
    return float(row.get("opening_day_gross_usd", 0.0) or 0.0) >= min_opening_day_gross


def rows_at_opening_day_floor(
    rows: list[dict[str, object]],
    min_opening_day_gross: int,
) -> list[dict[str, object]]:
    return [row for row in rows if opening_day_gross_at_least(row, min_opening_day_gross)]


def pre_release_pool_days(snapshot_day: int) -> tuple[int, ...]:
    for days in PRE_RELEASE_POOL_WINDOWS:
        if snapshot_day in days:
            return days
    return (snapshot_day,)


def pre_release_pooled_bop_rows(
    rows: list[dict[str, object]],
    snapshot_day: int,
) -> list[dict[str, object]]:
    pool_days = set(pre_release_pool_days(snapshot_day))
    return [
        row
        for row in rows
        if int(row["snapshot_day"]) in pool_days
        and float(row.get("bop_forecast_available", 0.0) or 0.0) > 0.0
    ]


def train_floor_for_model(
    model_name: str,
    *,
    min_opening_day_gross: int,
    wiki_min_opening_day_gross: int,
) -> int:
    if model_name == FALLBACK_MODEL:
        return wiki_min_opening_day_gross
    return min_opening_day_gross


def get_model_bundle(
    conn: Any,
    *,
    train_start_year: int = DEFAULT_TRAIN_START_YEAR,
    train_end_year: int = DEFAULT_TRAIN_END_YEAR,
    min_opening_day_gross: int = DEFAULT_MIN_OPENING_DAY_GROSS,
    wiki_min_opening_day_gross: int = DEFAULT_WIKI_MIN_OPENING_DAY_GROSS,
) -> ForecastModelBundle:
    key = (train_start_year, train_end_year, min_opening_day_gross, wiki_min_opening_day_gross)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    load_min_opening_day_gross = min(min_opening_day_gross, wiki_min_opening_day_gross)
    movies = opening.load_opening_weekend_movies(
        conn,
        min_year=train_start_year,
        max_year=train_end_year,
        min_opening_day_gross=load_min_opening_day_gross,
    )
    if not movies:
        bundle = ForecastModelBundle(
            train_start_year=train_start_year,
            train_end_year=train_end_year,
            min_opening_day_gross=min_opening_day_gross,
            models={},
            wiki_min_opening_day_gross=wiki_min_opening_day_gross,
        )
        _MODEL_CACHE[key] = bundle
        return bundle

    standard_movies = [
        movie for movie in movies if movie.opening_day_gross_usd >= min_opening_day_gross
    ]
    min_opening = min(movie.opening_date for movie in movies)
    max_opening = max(movie.opening_date for movie in movies)
    daily_grosses = opening.load_daily_grosses(
        conn,
        start_date=min_opening + dt.timedelta(days=min(SNAPSHOT_DAYS) - 7),
        end_date=max_opening + dt.timedelta(days=max(SNAPSHOT_DAYS)),
    )
    weekend_daily_grosses = opening.load_daily_grosses(
        conn,
        start_date=min_opening,
        end_date=max_opening + dt.timedelta(days=max(OPENING_WEEKEND_OFFSETS)),
    )
    remainder_models = train_weekend_remainder_models(standard_movies, weekend_daily_grosses)
    wiki_by_movie = opening.load_wiki_feature_map(conn, movies=movies, timing_days=SNAPSHOT_DAYS)
    forecasts = opening.load_boxofficepro_forecasts(
        conn,
        min_target_date=min_opening,
        max_target_date=max_opening,
    )
    panel_rows = day_by_day.build_day_by_day_feature_panel(
        movies,
        daily_grosses,
        wiki_by_movie,
        forecasts,
        snapshot_days=SNAPSHOT_DAYS,
        train_start_year=train_start_year,
        train_end_year=train_end_year,
    )
    amc_by_movie = {
        movie.movie_id: load_amc_feature_map(conn, movie_id=movie.movie_id, opening_date=movie.opening_date)
        for movie in movies
    }
    for row in panel_rows:
        movie_id = int(row["movie_id"])
        snapshot_day = int(row["snapshot_day"])
        row.update(amc_features_for_snapshot(amc_by_movie.get(movie_id, {}), snapshot_day))

    models: dict[tuple[int, str], ForecastDayModel] = {}
    for snapshot_day in SNAPSHOT_DAYS:
        all_day_rows = [
            row
            for row in panel_rows
            if int(row["snapshot_day"]) == snapshot_day
        ]
        fallback_train_rows = [
            row
            for row in day_by_day.rows_with_target(all_day_rows, "target_log_opening_weekend")
            if float(row.get("wiki_available", 0.0) or 0.0) > 0.0
            and opening_day_gross_at_least(row, wiki_min_opening_day_gross)
        ]
        if len(fallback_train_rows) >= len(FALLBACK_TERMS) + 2:
            fitted = opening.fit_model_for_target(
                fallback_train_rows,
                FALLBACK_TERMS,
                "target_log_opening_weekend",
            )
            loo_residuals = day_by_day.loo_prediction_residuals_for_opening_model(
                rows=fallback_train_rows,
                terms=FALLBACK_TERMS,
            )
            bucket_residuals, global_residuals = day_by_day.interval_residuals_by_bucket(
                rows=fallback_train_rows,
                residuals=loo_residuals,
            )
            models[(snapshot_day, FALLBACK_MODEL)] = ForecastDayModel(
                snapshot_day=snapshot_day,
                model_name=FALLBACK_MODEL,
                terms=FALLBACK_TERMS,
                train_rows=fallback_train_rows,
                fitted=fitted,
                bucket_residuals=bucket_residuals,
                global_residuals=global_residuals,
            )

        day_rows = [
            row
            for row in all_day_rows
            if float(row.get("bop_forecast_available", 0.0) or 0.0) > 0.0
        ]
        if not day_rows:
            continue
        amc_train_rows = [
            row
            for row in day_by_day.rows_with_target(day_rows, "target_log_bop_residual")
            if float(row.get("amc_available", 0.0) or 0.0) > 0.0
            and opening_day_gross_at_least(row, min_opening_day_gross)
        ]
        if len(amc_train_rows) >= len(AMC_RESIDUAL_TERMS) + 2:
            fitted = opening.fit_model_for_target(amc_train_rows, AMC_RESIDUAL_TERMS, "target_log_bop_residual")
            loo_residuals = day_by_day.loo_prediction_residuals_for_model(
                model_name=RESIDUAL_MODEL,
                terms=AMC_RESIDUAL_TERMS,
                rows=amc_train_rows,
            )
            bucket_residuals, global_residuals = day_by_day.interval_residuals_by_bucket(
                rows=amc_train_rows,
                residuals=loo_residuals,
            )
            models[(snapshot_day, AMC_RESIDUAL_MODEL)] = ForecastDayModel(
                snapshot_day=snapshot_day,
                model_name=AMC_RESIDUAL_MODEL,
                terms=AMC_RESIDUAL_TERMS,
                train_rows=amc_train_rows,
                fitted=fitted,
                bucket_residuals=bucket_residuals,
                global_residuals=global_residuals,
            )
        for model_name, terms in day_by_day.SNAPSHOT_MODEL_TERMS.items():
            target_key = (
                "target_log_opening_weekend"
                if model_name == "calibrated_bop_snapshot"
                else "target_log_bop_residual"
            )
            model_floor = train_floor_for_model(
                model_name,
                min_opening_day_gross=min_opening_day_gross,
                wiki_min_opening_day_gross=wiki_min_opening_day_gross,
            )
            training_source_rows = (
                pre_release_pooled_bop_rows(panel_rows, snapshot_day)
                if snapshot_day < 0 and model_name in {RAW_MODEL, WIKI_RESIDUAL_MODEL}
                else day_rows
            )
            model_day_rows = rows_at_opening_day_floor(training_source_rows, model_floor)
            train_rows = (
                day_by_day.rows_with_target(model_day_rows, target_key)
                if model_name != RAW_MODEL
                else model_day_rows
            )
            if (
                snapshot_day < 0
                and model_name == WIKI_RESIDUAL_MODEL
                and len(train_rows) < len(terms) + 2
                and len(train_rows) >= len(PRE_RELEASE_WIKI_RESIDUAL_FALLBACK_TERMS) + 2
            ):
                terms = PRE_RELEASE_WIKI_RESIDUAL_FALLBACK_TERMS
            if model_name != RAW_MODEL and len(train_rows) < len(terms) + 2:
                continue
            fitted: opening.FittedModel | None = None
            if model_name != RAW_MODEL:
                fitted = opening.fit_model_for_target(train_rows, terms, target_key)
            loo_residuals = day_by_day.loo_prediction_residuals_for_model(
                model_name=model_name,
                terms=terms,
                rows=train_rows,
            )
            bucket_residuals, global_residuals = day_by_day.interval_residuals_by_bucket(
                rows=train_rows,
                residuals=loo_residuals,
            )
            models[(snapshot_day, model_name)] = ForecastDayModel(
                snapshot_day=snapshot_day,
                model_name=model_name,
                terms=terms,
                train_rows=train_rows,
                fitted=fitted,
                bucket_residuals=bucket_residuals,
                global_residuals=global_residuals,
            )

    bundle = ForecastModelBundle(
        train_start_year=train_start_year,
        train_end_year=train_end_year,
        min_opening_day_gross=min_opening_day_gross,
        models=models,
        wiki_min_opening_day_gross=wiki_min_opening_day_gross,
        remainder_models=remainder_models,
    )
    _MODEL_CACHE[key] = bundle
    return bundle


def selected_model_for_row(bundle: ForecastModelBundle, row: dict[str, object]) -> ForecastDayModel | None:
    snapshot_day = int(row["snapshot_day"])
    if float(row.get("bop_forecast_available", 0.0) or 0.0) <= 0.0:
        return bundle.models.get((snapshot_day, FALLBACK_MODEL))
    if snapshot_day < 0:
        return pre_release_bop_model_for_row(bundle, row)
    model_names = bop_model_priority_for_row(row)
    exact_model = first_model_for_snapshot(bundle, snapshot_day, model_names)
    if exact_model is not None:
        return exact_model
    return nearest_model_for_snapshot(bundle, snapshot_day, model_names)


def pre_release_bop_model_for_row(bundle: ForecastModelBundle, row: dict[str, object]) -> ForecastDayModel | None:
    snapshot_day = int(row["snapshot_day"])
    if snapshot_day in PRE_RELEASE_WIKI_RESIDUAL_LIVE_DAYS and float(row.get("wiki_available", 0.0) or 0.0) > 0.0:
        model = bundle.models.get((snapshot_day, WIKI_RESIDUAL_MODEL))
        if model is not None:
            return model
    return bundle.models.get((snapshot_day, RAW_MODEL)) or raw_bop_fallback_model(snapshot_day)


def raw_bop_fallback_model(snapshot_day: int) -> ForecastDayModel:
    return ForecastDayModel(
        snapshot_day=snapshot_day,
        model_name=RAW_MODEL,
        terms=[],
        train_rows=[],
        fitted=None,
        bucket_residuals={},
        global_residuals=[],
    )


def bop_model_priority_for_row(row: dict[str, object]) -> list[str]:
    snapshot_day = int(row["snapshot_day"])
    if snapshot_day < 0:
        return [WIKI_RESIDUAL_MODEL, RAW_MODEL] if snapshot_day in PRE_RELEASE_WIKI_RESIDUAL_LIVE_DAYS else [RAW_MODEL]
    has_wiki = float(row.get("wiki_available", 0.0) or 0.0) > 0.0
    has_competition = float(row.get("competitor_count_lag7", 0.0) or 0.0) > 0.0
    has_amc = float(row.get("amc_available", 0.0) or 0.0) > 0.0
    models: list[str] = []
    if has_wiki and float(row.get("bop_forecast_count_as_of", 0.0) or 0.0) > 0.0:
        models.append(HISTORY_RESIDUAL_MODEL)
    if has_amc:
        models.append(AMC_RESIDUAL_MODEL)
    if has_wiki and has_competition and snapshot_day >= DEFAULT_COMPETITION_RESIDUAL_SWITCH_DAY:
        models.append(RESIDUAL_MODEL)
    if has_wiki:
        models.append(WIKI_RESIDUAL_MODEL)
    if has_competition and snapshot_day >= DEFAULT_COMPETITION_RESIDUAL_SWITCH_DAY:
        models.append(COMPETITION_RESIDUAL_MODEL)
    models.extend([BUCKET_RESIDUAL_MODEL, CALIBRATED_MODEL, RAW_MODEL])
    return models


def first_model_for_snapshot(
    bundle: ForecastModelBundle,
    snapshot_day: int,
    model_names: list[str],
) -> ForecastDayModel | None:
    for model_name in model_names:
        model = bundle.models.get((snapshot_day, model_name))
        if model is not None:
            return model
    return None


def nearest_model_for_snapshot(
    bundle: ForecastModelBundle,
    snapshot_day: int,
    model_names: list[str],
) -> ForecastDayModel | None:
    candidates = [
        (abs(day - snapshot_day), model_names.index(model_name), day, model)
        for (day, model_name), model in bundle.models.items()
        if model_name in model_names and model_name != FALLBACK_MODEL
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1], item[2]))[3]


def predict_log_for_day(model: ForecastDayModel, row: dict[str, object]) -> float:
    if model.model_name == RAW_MODEL:
        return math.log(max(1.0, float(row["bop_forecast_midpoint"])))
    if model.model_name == FALLBACK_MODEL:
        if model.fitted is None:
            return math.log(1.0)
        return opening.predict_log(model.fitted, [row])[0]
    if model.fitted is None:
        return math.log(max(1.0, float(row["bop_forecast_midpoint"])))
    if model.model_name == "calibrated_bop_snapshot":
        return opening.predict_log(model.fitted, [row])[0]
    residual = opening.predict_log(model.fitted, [row])[0]
    return math.log(max(1.0, float(row["bop_forecast_midpoint"]))) + residual


def forecast_stage_for_snapshot(snapshot_day: int, *, target_day_count: int = 3) -> str:
    if snapshot_day < 0:
        return "pre_release"
    if snapshot_day < target_day_count:
        return "in_window"
    return "reconciliation"


def remaining_weekend_prediction(
    *,
    bundle: ForecastModelBundle,
    locked_offsets: tuple[int, ...],
    known_actual_usd: float,
    base_total_prediction_usd: float,
    target_day_count: int = 3,
) -> float:
    if not locked_offsets:
        return max(0.0, base_total_prediction_usd)
    if locked_offsets == tuple(range(target_day_count)):
        return 0.0
    model = bundle.remainder_models.get(locked_offsets)
    if model is not None and known_actual_usd > 0.0:
        return max(0.0, known_actual_usd * model.mean_remaining_to_known)
    return max(0.0, base_total_prediction_usd - known_actual_usd)


def reconcile_snapshot_prediction(
    *,
    bundle: ForecastModelBundle,
    snapshot_day: int,
    base_prediction_usd: float,
    base_lower_usd: float | None,
    base_upper_usd: float | None,
    weekend_actuals: list[dict[str, object]],
    target_day_count: int = 3,
) -> dict[str, object]:
    totals = weekend_actual_totals(weekend_actuals, snapshot_day=snapshot_day, target_day_count=target_day_count)
    known = float(totals["actual_weekend_gross_known_usd"])
    locked_offsets = totals["locked_offsets"]  # type: ignore[assignment]
    if snapshot_day >= target_day_count and not bool(totals["is_complete"]) and base_prediction_usd <= 0.0:
        return {
            **totals,
            "remaining_weekend_prediction_usd": None,
            "remaining_weekend_prediction_label": "-",
            "reconciled_opening_weekend_prediction_usd": None,
            "reconciled_opening_weekend_prediction_label": "-",
            "reconciled_lower_80_opening_weekend_revenue_usd": None,
            "reconciled_upper_80_opening_weekend_revenue_usd": None,
            "reconciled_lower_80_label": "-",
            "reconciled_upper_80_label": "-",
            "reconciliation_status": "incomplete",
        }
    remaining = remaining_weekend_prediction(
        bundle=bundle,
        locked_offsets=locked_offsets,  # type: ignore[arg-type]
        known_actual_usd=known,
        base_total_prediction_usd=base_prediction_usd,
        target_day_count=target_day_count,
    )
    reconciled = known + remaining if locked_offsets else base_prediction_usd
    if bool(totals["is_complete"]):
        reconciled = known
    lower = None if base_lower_usd is None else max(known, float(base_lower_usd))
    upper = None if base_upper_usd is None else max(reconciled, known, float(base_upper_usd))
    if bool(totals["is_complete"]):
        lower = known
        upper = known
    reconciliation_status = "final" if bool(totals["is_complete"]) else ("partial" if locked_offsets else "modeled")
    if snapshot_day >= target_day_count and not bool(totals["is_complete"]):
        reconciliation_status = "incomplete_modeled"
    return {
        **totals,
        "remaining_weekend_prediction_usd": remaining,
        "remaining_weekend_prediction_label": money(remaining),
        "reconciled_opening_weekend_prediction_usd": reconciled,
        "reconciled_opening_weekend_prediction_label": money(reconciled),
        "reconciled_lower_80_opening_weekend_revenue_usd": lower,
        "reconciled_upper_80_opening_weekend_revenue_usd": upper,
        "reconciled_lower_80_label": money(lower) if lower is not None else "-",
        "reconciled_upper_80_label": money(upper) if upper is not None else "-",
        "reconciliation_status": reconciliation_status,
    }


def movie_for_candidate(conn: Any, candidate: ForecastCandidate) -> opening.OpeningWeekendMovie:
    actual = load_movie_actual(
        conn,
        movie_id=candidate.movie_id,
        opening_date=candidate.target_start_date,
        target_day_count=candidate.target_day_count,
    )
    return opening.OpeningWeekendMovie(
        movie_id=candidate.movie_id,
        title=candidate.title,
        release_year=candidate.release_year or candidate.opening_date.year,
        release_run_id=int(actual["release_run_id"]),
        opening_date=candidate.target_start_date,
        opening_theaters=int(actual["opening_theaters"]),
        opening_day_gross_usd=int(actual["opening_day_gross_usd"]),
        opening_weekend_revenue_usd=int(actual["opening_weekend_revenue_usd"]),
    )


def build_movie_feature_rows(
    conn: Any,
    candidate: ForecastCandidate,
    *,
    today: dt.date,
    train_start_year: int = DEFAULT_TRAIN_START_YEAR,
    train_end_year: int = DEFAULT_TRAIN_END_YEAR,
) -> tuple[list[dict[str, object]], bool]:
    movie = movie_for_candidate(conn, candidate)
    has_actual = movie.opening_weekend_revenue_usd > 0
    target_snapshot_days = snapshot_days_for_target_type(candidate.target_type)
    min_as_of = candidate.target_start_date + dt.timedelta(days=min(target_snapshot_days) - 7)
    max_as_of = min(today, candidate.target_start_date + dt.timedelta(days=max(target_snapshot_days)))
    if max_as_of < min_as_of:
        daily_grosses: list[opening.DailyGross] = []
    else:
        daily_grosses = opening.load_daily_grosses(conn, start_date=min_as_of, end_date=max_as_of)
    wiki_by_movie = opening.load_wiki_feature_map(conn, movies=[movie], timing_days=target_snapshot_days)
    bop_target_start_date = candidate.bop_target_start_date or candidate.target_start_date
    bop_target_day_count = candidate.bop_target_day_count or candidate.target_day_count
    forecasts = opening.load_boxofficepro_forecasts(
        conn,
        min_target_date=min(candidate.target_start_date, bop_target_start_date),
        max_target_date=max(candidate.target_start_date, bop_target_start_date),
    )
    rows = day_by_day.build_day_by_day_feature_panel(
        [movie],
        daily_grosses,
        wiki_by_movie,
        forecasts,
        snapshot_days=target_snapshot_days,
        train_start_year=train_start_year,
        train_end_year=train_end_year,
        target_types=[candidate.target_type],
    )
    if not rows:
        rows = build_live_unresolved_feature_rows(
            movie=movie,
            daily_grosses=daily_grosses,
            wiki_by_movie=wiki_by_movie,
            forecasts=forecasts,
            snapshot_days=target_snapshot_days,
            target_day_count=candidate.target_day_count,
            target_type=candidate.target_type,
            bop_target_start_date=bop_target_start_date,
            bop_target_day_count=bop_target_day_count,
            manual_target_override=candidate.manual_target_override,
        )
    for row in rows:
        row["manual_target_override"] = candidate.manual_target_override
        row["bop_source_target_type"] = candidate.bop_target_type or candidate.target_type
        row["bop_source_target_start_date"] = bop_target_start_date.isoformat()
        row["bop_source_target_end_date"] = (
            candidate.bop_target_end_date or bop_target_start_date + dt.timedelta(days=bop_target_day_count - 1)
        ).isoformat()
        row["bop_source_target_day_count"] = bop_target_day_count
    amc_by_offset = load_amc_feature_map(conn, movie_id=candidate.movie_id, opening_date=candidate.target_start_date)
    for row in rows:
        row.update(amc_features_for_snapshot(amc_by_offset, int(row["snapshot_day"])))
    visible_rows = [
        row
        for row in rows
        if parse_db_date(row["as_of_date"]) <= today
        and int(row["snapshot_day"]) <= max(target_snapshot_days)
    ]
    return visible_rows, has_actual


def build_live_unresolved_feature_rows(
    *,
    movie: opening.OpeningWeekendMovie,
    daily_grosses: list[opening.DailyGross],
    wiki_by_movie: dict[int, dict[int, dict[str, float]]],
    forecasts: list[opening.BoxofficeProForecast],
    snapshot_days: list[int],
    target_day_count: int,
    target_type: str,
    bop_target_start_date: dt.date | None = None,
    bop_target_day_count: int | None = None,
    manual_target_override: bool = False,
) -> list[dict[str, object]]:
    gross_by_movie_date = {(row.movie_id, row.box_office_date): row for row in daily_grosses}
    target_start_date = movie.opening_date
    target_end_date = target_start_date + dt.timedelta(days=target_day_count - 1)
    source_bop_target_start_date = bop_target_start_date or target_start_date
    source_bop_target_day_count = bop_target_day_count or target_day_count
    source_bop_target_end_date = source_bop_target_start_date + dt.timedelta(days=source_bop_target_day_count - 1)
    rows: list[dict[str, object]] = []
    for snapshot_day in snapshot_days:
        as_of_date = target_start_date + dt.timedelta(days=snapshot_day)
        feature_available_date = day_by_day.competition_information_day(target_start_date, snapshot_day)
        wiki = opening.wiki_values_as_of(
            wiki_by_movie,
            movie_id=movie.movie_id,
            timing_day=snapshot_day,
        )
        focal_forecast = day_by_day.latest_primary_opening_bop_forecast(
            forecasts,
            as_of_date=as_of_date,
            movie_id=movie.movie_id,
            target_start_date=source_bop_target_start_date,
            target_day_count=source_bop_target_day_count,
        )
        bop_history = day_by_day.bop_forecast_history_features(
            forecasts,
            as_of_date=as_of_date,
            movie_id=movie.movie_id,
            target_start_date=source_bop_target_start_date,
            target_day_count=source_bop_target_day_count,
        )
        bop_features = opening.bop_forecast_features(focal_forecast)
        midpoint = float(bop_features.get("bop_forecast_midpoint", 0.0) or 0.0)
        competition = opening.actual_competition_features(
            gross_by_movie_date,
            focal_movie_id=movie.movie_id,
            as_of_date=feature_available_date,
        )
        row: dict[str, object] = {
            "movie_id": movie.movie_id,
            "title": movie.title,
            "release_year": movie.release_year,
            "release_run_id": movie.release_run_id,
            "opening_date": target_start_date.isoformat(),
            "target_type": target_type,
            "target_day_count": target_day_count,
            "target_start_date": target_start_date.isoformat(),
            "target_end_date": target_end_date.isoformat(),
            "manual_target_override": manual_target_override,
            "bop_source_target_type": target_type_for_day_count(source_bop_target_day_count),
            "bop_source_target_day_count": source_bop_target_day_count,
            "bop_source_target_start_date": source_bop_target_start_date.isoformat(),
            "bop_source_target_end_date": source_bop_target_end_date.isoformat(),
            "target_gross": 0.0,
            "target_gross_usd": 0.0,
            "forecast_stage": forecast_stage_for_snapshot(snapshot_day, target_day_count=target_day_count),
            "snapshot_day": snapshot_day,
            "timing_day": snapshot_day,
            "wiki_timing_day": snapshot_day,
            "competition_timing_day": (feature_available_date - target_start_date).days,
            "bop_timing_day": snapshot_day,
            "as_of_date": as_of_date.isoformat(),
            "feature_available_date": feature_available_date.isoformat(),
            "opening_theaters": movie.opening_theaters,
            "opening_day_gross_usd": movie.opening_day_gross_usd,
            "opening_weekend_revenue_usd": 0.0,
            "opening_weekend_day0_gross_usd": 0.0,
            "opening_weekend_day1_gross_usd": 0.0,
            "opening_weekend_day2_gross_usd": 0.0,
            "opening_weekend_day3_gross_usd": 0.0,
            "opening_weekend_day4_gross_usd": 0.0,
            "target_log_opening_weekend": "",
            "target_log_gross": "",
            "target_log_bop_residual": "",
            "log1p_opening_theaters": opening.log1p(movie.opening_theaters),
            "known_actual_offsets": "",
            "known_actual_day_count": 0,
            "known_gross_so_far": 0.0,
            "known_gross_so_far_usd": 0.0,
            "remaining_gross": 0.0,
            "remaining_gross_usd": 0.0,
            "known_gross_so_far_pct_of_bop_midpoint": 0.0,
            "target_is_complete_as_of_snapshot": False,
            "wiki_available": 1.0 if any(wiki.values()) else 0.0,
            "V": wiki["V"],
            "U": wiki["U"],
            "R": wiki["R"],
            "E": wiki["E"],
            "log1p_V": opening.log1p(wiki["V"]),
            "log1p_U": opening.log1p(wiki["U"]),
            "log1p_R": opening.log1p(wiki["R"]),
            "log1p_E": opening.log1p(wiki["E"]),
            "preview_gross_known_usd": 0.0,
            "preview_day_count": 0.0,
            "log1p_preview_gross_known": 0.0,
            **bop_features,
            **bop_history,
            **competition,
        }
        opening.add_bop_fixed_estimate_buckets(row)
        row["bop_estimate_bucket"] = day_by_day.estimate_bucket_label(row)
        row.update(day_by_day.paper_competition_proxy_features(row))
        rows.append(row)
    return rows


def actual_gross_by_offset(weekend_actuals: list[dict[str, object]]) -> dict[int, float]:
    return {
        int(row["offset"]): float(row["gross_usd"] or 0.0)
        for row in weekend_actuals
        if bool(row.get("is_actual")) and row.get("gross_usd") is not None
    }


def deployment_prediction_payload(
    row: dict[str, object],
    *,
    weekend_actuals: list[dict[str, object]],
    model_prediction_usd: float | None,
    model_lower_usd: float | None,
    model_upper_usd: float | None,
    remaining_gross_forecast: float | None,
) -> dict[str, object]:
    snapshot = feature_snapshot_from_panel_row(
        row,
        actual_gross_by_offset=actual_gross_by_offset(weekend_actuals),
    )
    bop_midpoint = float(row.get("bop_forecast_midpoint", 0.0) or 0.0)
    predicted_log_residual = 0.0
    if model_prediction_usd is not None and model_prediction_usd > 0.0 and bop_midpoint > 0.0:
        predicted_log_residual = math.log(model_prediction_usd) - math.log(bop_midpoint)
    interval_log_width = 0.25
    if model_prediction_usd is not None and model_upper_usd is not None and model_prediction_usd > 0.0:
        interval_log_width = max(0.01, abs(math.log(max(model_upper_usd, 1.0)) - math.log(model_prediction_usd)))
    prediction = DEPLOYED_ENGINE.predict(
        snapshot,
        predicted_log_residual=predicted_log_residual,
        expected_remaining_gross=remaining_gross_forecast,
        predicted_remaining_log_residual=0.0,
        interval_log_width=interval_log_width,
    )
    payload = prediction.to_payload()
    if bool(row.get("manual_target_override")):
        source_type = str(row.get("bop_source_target_type") or "")
        source_start = str(row.get("bop_source_target_start_date") or "")
        source_end = str(row.get("bop_source_target_end_date") or "")
        selected_type = str(row.get("target_type") or "")
        reason = "Manual target-window override"
        if source_type:
            reason = f"{reason}; BOP anchor source is {source_type}"
        if source_start and source_end:
            reason = f"{reason} ({source_start} to {source_end})"
        if selected_type:
            reason = f"{reason}, selected target is {selected_type}"
        payload["deployment_status"] = "shadow_manual_override"
        payload["shadow_reason"] = reason
    payload["manual_target_override"] = bool(row.get("manual_target_override"))
    payload["bop_source_target_type"] = row.get("bop_source_target_type", "")
    payload["bop_source_target_days"] = int(row.get("bop_source_target_day_count", 0) or 0)
    payload["bop_source_target_start_date"] = row.get("bop_source_target_start_date", "")
    payload["bop_source_target_end_date"] = row.get("bop_source_target_end_date", "")
    payload["model_point_forecast_usd"] = prediction.point_forecast_usd
    payload["model_lower_80_usd"] = prediction.lower_80_usd
    payload["model_upper_80_usd"] = prediction.upper_80_usd
    return payload


def minimum_interval_width_pct(snapshot: dict[str, object]) -> float:
    state = str(snapshot.get("forecast_state") or "")
    if not state:
        state = "official_actuals_available" if int(snapshot.get("actual_day_count", 0) or 0) > 0 else "pre_release"
    target_type = str(snapshot.get("target_type") or "3_day")
    base = MIN_INTERVAL_WIDTH_PCT_BY_STATE.get(state, MIN_INTERVAL_WIDTH_PCT_BY_STATE["pre_release"])
    multiplier = TARGET_INTERVAL_WIDTH_MULTIPLIER.get(target_type, 1.0)
    return base * multiplier


def stabilize_prediction_intervals(snapshots: list[dict[str, object]]) -> None:
    previous_width_pct: float | None = None
    for snapshot in sorted(snapshots, key=lambda row: int(row["snapshot_day"])):
        prediction = snapshot.get("predicted_opening_weekend_revenue_usd")
        if prediction is None:
            continue
        point = float(prediction)
        if point <= 0.0:
            continue
        if snapshot.get("reconciliation_status") == "final":
            lower = upper = point
            width_pct = 0.0
        else:
            raw_lower = snapshot.get("predicted_lower_80_opening_weekend_revenue_usd")
            raw_upper = snapshot.get("predicted_upper_80_opening_weekend_revenue_usd")
            raw_width_pct = 0.0
            if raw_lower is not None and raw_upper is not None:
                raw_width_pct = max(0.0, float(raw_upper) - float(raw_lower)) / point
            width_pct = max(raw_width_pct, minimum_interval_width_pct(snapshot))
            if previous_width_pct is not None:
                width_pct = min(width_pct, previous_width_pct)
            half_width = width_pct / 2.0
            known = float(snapshot.get("actual_weekend_gross_known_usd", 0.0) or 0.0)
            lower = max(known, point * max(0.01, 1.0 - half_width))
            upper = max(point, point * (1.0 + half_width), lower)
        previous_width_pct = width_pct
        snapshot["predicted_lower_80_opening_weekend_revenue_usd"] = lower
        snapshot["predicted_upper_80_opening_weekend_revenue_usd"] = upper
        snapshot["predicted_lower_80_label"] = money(lower)
        snapshot["predicted_upper_80_label"] = money(upper)
        snapshot["prediction_uncertainty_width_pct"] = width_pct
        snapshot.update(
            prediction_confidence(
                point,
                lower,
                upper,
                interval_train_n=int(snapshot.get("prediction_interval_train_n", 0) or 0),
                interval_source=str(snapshot.get("prediction_interval_source", "")),
                interval_method=str(snapshot.get("prediction_interval_method", PRIMARY_INTERVAL_METHOD)),
                is_final_actual=snapshot.get("reconciliation_status") == "final",
            )
        )


def forecast_movie(
    conn: Any,
    *,
    movie_id: int,
    target_type: str | None = None,
    target_start_date: dt.date | None = None,
    today: dt.date | None = None,
    train_start_year: int = DEFAULT_TRAIN_START_YEAR,
    train_end_year: int = DEFAULT_TRAIN_END_YEAR,
    min_opening_day_gross: int = DEFAULT_MIN_OPENING_DAY_GROSS,
) -> dict[str, object] | None:
    today = today or dt.date.today()
    train_end_year = max(train_end_year, today.year)
    candidate = latest_candidate_for_movie(conn, movie_id, target_type=target_type, target_start_date=target_start_date)
    if candidate is None:
        return None
    available_target_types = available_target_types_for_movie(conn, movie_id)
    if candidate.target_type not in available_target_types:
        available_target_types.append(candidate.target_type)
    for manual_target_type in ("3_day", "4_day", "5_day"):
        if manual_target_type not in available_target_types:
            available_target_types.append(manual_target_type)
    available_target_types.sort(key=target_day_count_from_type)

    bundle = get_model_bundle(
        conn,
        train_start_year=train_start_year,
        train_end_year=train_end_year,
        min_opening_day_gross=min_opening_day_gross,
    )
    rows, _has_partial_actual = build_movie_feature_rows(
        conn,
        candidate,
        today=today,
        train_start_year=train_start_year,
        train_end_year=train_end_year,
    )
    weekend_actuals = load_opening_weekend_daily_actuals(
        conn,
        movie_id=candidate.movie_id,
        opening_date=candidate.target_start_date,
        target_day_count=candidate.target_day_count,
    )
    full_actual_totals = weekend_actual_totals(
        weekend_actuals,
        snapshot_day=candidate.target_day_count,
        target_day_count=candidate.target_day_count,
    )
    snapshots: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: int(item["snapshot_day"])):
        snapshot_day = int(row["snapshot_day"])
        model = selected_model_for_row(bundle, row)
        if model is None:
            snapshot = empty_snapshot(row)
            reconciliation = reconcile_snapshot_prediction(
                bundle=bundle,
                snapshot_day=snapshot_day,
                base_prediction_usd=0.0,
                base_lower_usd=None,
                base_upper_usd=None,
                weekend_actuals=weekend_actuals,
                target_day_count=candidate.target_day_count,
            )
            snapshot.update(reconciliation)
            snapshot["forecast_stage"] = forecast_stage_for_snapshot(
                snapshot_day,
                target_day_count=candidate.target_day_count,
            )
            if reconciliation["reconciliation_status"] == "final":
                snapshot["status"] = "ok"
                snapshot["model"] = "actuals"
                snapshot["prediction_source"] = "actuals"
                snapshot["predicted_opening_weekend_revenue_usd"] = reconciliation[
                    "reconciled_opening_weekend_prediction_usd"
                ]
                snapshot["predicted_opening_weekend_revenue_label"] = reconciliation[
                    "reconciled_opening_weekend_prediction_label"
                ]
                snapshot["predicted_lower_80_opening_weekend_revenue_usd"] = reconciliation[
                    "reconciled_lower_80_opening_weekend_revenue_usd"
                ]
                snapshot["predicted_upper_80_opening_weekend_revenue_usd"] = reconciliation[
                    "reconciled_upper_80_opening_weekend_revenue_usd"
                ]
                snapshot["predicted_lower_80_label"] = reconciliation["reconciled_lower_80_label"]
                snapshot["predicted_upper_80_label"] = reconciliation["reconciled_upper_80_label"]
            elif snapshot_day >= candidate.target_day_count:
                snapshot["status"] = "reconciliation_incomplete"
            snapshot.update(
                prediction_confidence(
                    snapshot["predicted_opening_weekend_revenue_usd"],
                    snapshot["predicted_lower_80_opening_weekend_revenue_usd"],
                    snapshot["predicted_upper_80_opening_weekend_revenue_usd"],
                    interval_train_n=0,
                    interval_source="",
                    interval_method=PRIMARY_INTERVAL_METHOD,
                    is_final_actual=snapshot.get("reconciliation_status") == "final",
                )
            )
            snapshot.update(
                deployment_prediction_payload(
                    row,
                    weekend_actuals=weekend_actuals,
                    model_prediction_usd=snapshot["predicted_opening_weekend_revenue_usd"],
                    model_lower_usd=snapshot["predicted_lower_80_opening_weekend_revenue_usd"],
                    model_upper_usd=snapshot["predicted_upper_80_opening_weekend_revenue_usd"],
                    remaining_gross_forecast=snapshot.get("remaining_weekend_prediction_usd"),
                )
            )
            snapshots.append(snapshot)
            continue
        pred_log = predict_log_for_day(model, row)
        interval = day_by_day.interval_for_row(
            row,
            pred_log=pred_log,
            bucket_residuals=model.bucket_residuals,
            global_residuals=model.global_residuals,
            interval_method=PRIMARY_INTERVAL_METHOD,
        )
        model_pred_gross = max(1.0, math.exp(pred_log))
        model_lower = float(interval["predicted_lower_80_opening_weekend_revenue_usd"])
        model_upper = float(interval["predicted_upper_80_opening_weekend_revenue_usd"])
        reconciliation = reconcile_snapshot_prediction(
            bundle=bundle,
            snapshot_day=snapshot_day,
            base_prediction_usd=model_pred_gross,
            base_lower_usd=model_lower,
            base_upper_usd=model_upper,
            weekend_actuals=weekend_actuals,
            target_day_count=candidate.target_day_count,
        )
        reconciled_gross = reconciliation["reconciled_opening_weekend_prediction_usd"]
        status = "ok" if reconciled_gross is not None else "reconciliation_incomplete"
        if reconciliation["reconciliation_status"] == "final":
            model_name = "actuals"
            prediction_source = "actuals"
        else:
            model_name = model.model_name
            prediction_source = prediction_source_for_model(model.model_name)
        interval_train_n = int(interval["prediction_interval_train_n"])
        certainty = prediction_confidence(
            reconciled_gross,
            reconciliation["reconciled_lower_80_opening_weekend_revenue_usd"],
            reconciliation["reconciled_upper_80_opening_weekend_revenue_usd"],
            interval_train_n=interval_train_n,
            interval_source=str(interval["prediction_interval_source"]),
            interval_method=str(interval["prediction_interval_method"]),
            is_final_actual=reconciliation["reconciliation_status"] == "final",
        )
        has_bop = float(row.get("bop_forecast_available", 0.0) or 0.0) > 0.0
        snapshot = {
                "snapshot_day": snapshot_day,
                "as_of_date": str(row["as_of_date"]),
                "status": status,
                "forecast_stage": forecast_stage_for_snapshot(
                    snapshot_day,
                    target_day_count=candidate.target_day_count,
                ),
                "model": model_name,
                "model_snapshot_day": model.snapshot_day,
                "model_is_exact_snapshot": model.snapshot_day == snapshot_day,
                "model_train_n": len(model.train_rows),
                "model_term_count": len(model.terms),
                "model_terms": model.terms,
                "prediction_source": prediction_source,
                "bop_forecast_available": has_bop,
                "bop_prediction_id": row.get("bop_prediction_id", ""),
                "bop_article_url": row.get("bop_article_url", ""),
                "bop_forecast_published_date": row.get("bop_forecast_published_date", ""),
                "bop_forecast_midpoint": float(row.get("bop_forecast_midpoint", 0.0) or 0.0),
                "bop_forecast_midpoint_label": (
                    money(float(row.get("bop_forecast_midpoint", 0.0) or 0.0)) if has_bop else "-"
                ),
                "bop_forecast_date": row.get("bop_forecast_published_date", ""),
                "bop_estimate_bucket": day_by_day.estimate_bucket_label(row),
                "bop_forecast_count_as_of": row.get("bop_forecast_count_as_of", 0.0),
                "bop_first_forecast_midpoint": row.get("bop_first_forecast_midpoint", 0.0),
                "bop_first_forecast_midpoint_label": money(
                    float(row.get("bop_first_forecast_midpoint", 0.0) or 0.0)
                ) if has_bop else "-",
                "bop_previous_forecast_midpoint": row.get("bop_previous_forecast_midpoint", 0.0),
                "bop_previous_forecast_midpoint_label": money(
                    float(row.get("bop_previous_forecast_midpoint", 0.0) or 0.0)
                ) if float(row.get("bop_previous_forecast_midpoint", 0.0) or 0.0) > 0.0 else "-",
                "bop_latest_lead_days": row.get("bop_latest_lead_days", 0.0),
                "bop_latest_lead_bucket": row.get("bop_latest_lead_bucket", "missing_lead"),
                "bop_revision_from_first_pct": row.get("bop_revision_from_first_pct", 0.0),
                "bop_revision_from_previous_pct": row.get("bop_revision_from_previous_pct", 0.0),
                "wiki_available": bool(float(row.get("wiki_available", 0.0) or 0.0) > 0.0),
                "competition_available": bool(float(row.get("competitor_count_lag7", 0.0) or 0.0) > 0.0),
                "amc_available": bool(float(row.get("amc_available", 0.0) or 0.0) > 0.0),
                "amc_cutoff": row.get("amc_cutoff", ""),
                "model_opening_weekend_prediction_usd": model_pred_gross,
                "model_opening_weekend_prediction_label": money(model_pred_gross),
                "predicted_opening_weekend_revenue_usd": reconciled_gross,
                "predicted_opening_weekend_revenue_label": (
                    reconciliation["reconciled_opening_weekend_prediction_label"]
                    if reconciled_gross is not None
                    else "-"
                ),
                "predicted_lower_80_opening_weekend_revenue_usd": reconciliation[
                    "reconciled_lower_80_opening_weekend_revenue_usd"
                ],
                "predicted_upper_80_opening_weekend_revenue_usd": reconciliation[
                    "reconciled_upper_80_opening_weekend_revenue_usd"
                ],
                "predicted_lower_80_label": reconciliation["reconciled_lower_80_label"],
                "predicted_upper_80_label": reconciliation["reconciled_upper_80_label"],
                "prediction_interval_method": interval["prediction_interval_method"],
                "prediction_interval_source": interval["prediction_interval_source"],
                "prediction_interval_train_n": interval_train_n,
                **certainty,
                **reconciliation,
            }
        snapshot.update(
            deployment_prediction_payload(
                row,
                weekend_actuals=weekend_actuals,
                model_prediction_usd=reconciled_gross,
                model_lower_usd=reconciliation["reconciled_lower_80_opening_weekend_revenue_usd"],
                model_upper_usd=reconciliation["reconciled_upper_80_opening_weekend_revenue_usd"],
                remaining_gross_forecast=reconciliation["remaining_weekend_prediction_usd"],
            )
        )
        snapshots.append(snapshot)

    actual = (
        float(full_actual_totals["actual_weekend_gross_known_usd"])
        if bool(full_actual_totals["is_complete"])
        else None
    )
    stabilize_prediction_intervals(snapshots)
    latest = latest_ok_snapshot(snapshots)
    response = {
        "movie": {
            "movie_id": candidate.movie_id,
            "title": candidate.title,
            "release_year": candidate.release_year,
            "opening_date": candidate.target_start_date.isoformat(),
            "target_type": candidate.target_type,
            "target_days": candidate.target_day_count,
            "target_start_date": candidate.target_start_date.isoformat(),
            "target_end_date": candidate.target_end_date.isoformat(),
            "days_until_opening": (candidate.target_start_date - today).days,
            "release_timing_label": release_timing_label(candidate.target_start_date, today),
            "source_movie_title": candidate.source_movie_title,
            "latest_bop_midpoint": candidate.latest_bop_midpoint,
            "latest_bop_midpoint_label": money(candidate.latest_bop_midpoint),
            "latest_forecast_date": candidate.latest_forecast_date.isoformat(),
            "manual_target_override": candidate.manual_target_override,
            "bop_source_target_type": candidate.bop_target_type or candidate.target_type,
            "bop_source_target_days": candidate.bop_target_day_count or candidate.target_day_count,
            "bop_source_target_start_date": (
                candidate.bop_target_start_date or candidate.target_start_date
            ).isoformat(),
            "bop_source_target_end_date": (
                candidate.bop_target_end_date or candidate.target_end_date
            ).isoformat(),
            "has_actual": bool(full_actual_totals["is_complete"]),
            "actual_opening_weekend_revenue_usd": actual,
            "actual_opening_weekend_label": money(actual) if actual is not None else "",
        },
        "latest_snapshot": latest,
        "target_type": candidate.target_type,
        "target_days": candidate.target_day_count,
        "target_start_date": candidate.target_start_date.isoformat(),
        "target_end_date": candidate.target_end_date.isoformat(),
        "manual_target_override": candidate.manual_target_override,
        "bop_source_target_type": candidate.bop_target_type or candidate.target_type,
        "bop_source_target_days": candidate.bop_target_day_count or candidate.target_day_count,
        "bop_source_target_start_date": (candidate.bop_target_start_date or candidate.target_start_date).isoformat(),
        "bop_source_target_end_date": (candidate.bop_target_end_date or candidate.target_end_date).isoformat(),
        "available_target_types": available_target_types,
        "bop_segment": latest.get("bop_segment") if latest is not None else "",
        "forecast_state": latest.get("forecast_state") if latest is not None else "",
        "deployment_status": latest.get("deployment_status") if latest is not None else "",
        "model_version": latest.get("model_version") if latest is not None else "",
        "model_family": latest.get("model_family") if latest is not None else "",
        "registry_reason": latest.get("registry_reason") if latest is not None else "",
        "shadow_reason": latest.get("shadow_reason") if latest is not None else "",
        "known_actual_offsets": latest.get("known_actual_offsets") if latest is not None else [],
        "known_actual_gross_so_far": latest.get("known_actual_gross_so_far") if latest is not None else 0.0,
        "weekend_actuals": weekend_actuals,
        "actual_weekend_gross_known_usd": full_actual_totals["actual_weekend_gross_known_usd"],
        "actual_weekend_gross_known_label": full_actual_totals["actual_weekend_gross_known_label"],
        "remaining_weekend_prediction_usd": (
            latest.get("remaining_weekend_prediction_usd") if latest is not None else None
        ),
        "remaining_weekend_prediction_label": (
            latest.get("remaining_weekend_prediction_label") if latest is not None else "-"
        ),
        "remaining_gross_forecast": latest.get("remaining_gross_forecast") if latest is not None else None,
        "reconciled_opening_weekend_prediction_usd": (
            latest.get("reconciled_opening_weekend_prediction_usd") if latest is not None else None
        ),
        "reconciled_opening_weekend_prediction_label": (
            latest.get("reconciled_opening_weekend_prediction_label") if latest is not None else "-"
        ),
        "forecast_stage": latest.get("forecast_stage") if latest is not None else "",
        "snapshots": snapshots,
        "chart_svg": forecast_chart_svg(candidate.title, snapshots, actual_opening_weekend=actual),
        "model_training": {
            "train_start_year": bundle.train_start_year,
            "train_end_year": bundle.train_end_year,
            "min_opening_day_gross": bundle.min_opening_day_gross,
            "wiki_min_opening_day_gross": bundle.wiki_min_opening_day_gross,
        },
    }
    return response


def prediction_source_for_model(model_name: str) -> str:
    if model_name == "actuals":
        return "actuals"
    if model_name == RAW_MODEL:
        return "bop"
    if model_name == FALLBACK_MODEL:
        return "fallback_model"
    if model_name == AMC_RESIDUAL_MODEL:
        return "amc_residual_model"
    return "residual_model"


def release_timing_label(opening_date: dt.date, today: dt.date) -> str:
    days_until = (opening_date - today).days
    if days_until > 0:
        unit = "day" if days_until == 1 else "days"
        return f"{days_until} {unit} until opening"
    days_since = -days_until
    if days_since == 0:
        return "released today"
    unit = "day" if days_since == 1 else "days"
    return f"released {days_since} {unit} ago"


def empty_snapshot(row: dict[str, object]) -> dict[str, object]:
    has_bop = float(row.get("bop_forecast_available", 0.0) or 0.0) > 0.0
    return {
        "snapshot_day": int(row["snapshot_day"]),
        "as_of_date": str(row["as_of_date"]),
        "status": "no_estimate" if not has_bop else "insufficient_model",
        "model": "",
        "model_snapshot_day": None,
        "model_is_exact_snapshot": False,
        "model_train_n": 0,
        "model_term_count": 0,
        "model_terms": [],
        "prediction_source": "",
        "bop_forecast_available": has_bop,
        "bop_prediction_id": row.get("bop_prediction_id", ""),
        "bop_article_url": row.get("bop_article_url", ""),
        "bop_forecast_published_date": row.get("bop_forecast_published_date", ""),
        "bop_forecast_midpoint": float(row.get("bop_forecast_midpoint", 0.0) or 0.0),
        "bop_forecast_midpoint_label": money(float(row.get("bop_forecast_midpoint", 0.0) or 0.0)) if has_bop else "-",
        "bop_forecast_date": row.get("bop_forecast_published_date", ""),
        "bop_estimate_bucket": day_by_day.estimate_bucket_label(row),
        "bop_forecast_count_as_of": row.get("bop_forecast_count_as_of", 0.0),
        "bop_first_forecast_midpoint": row.get("bop_first_forecast_midpoint", 0.0),
        "bop_first_forecast_midpoint_label": money(
            float(row.get("bop_first_forecast_midpoint", 0.0) or 0.0)
        ) if has_bop else "-",
        "bop_previous_forecast_midpoint": row.get("bop_previous_forecast_midpoint", 0.0),
        "bop_previous_forecast_midpoint_label": money(
            float(row.get("bop_previous_forecast_midpoint", 0.0) or 0.0)
        ) if float(row.get("bop_previous_forecast_midpoint", 0.0) or 0.0) > 0.0 else "-",
        "bop_latest_lead_days": row.get("bop_latest_lead_days", 0.0),
        "bop_latest_lead_bucket": row.get("bop_latest_lead_bucket", "missing_lead"),
        "bop_revision_from_first_pct": row.get("bop_revision_from_first_pct", 0.0),
        "bop_revision_from_previous_pct": row.get("bop_revision_from_previous_pct", 0.0),
        "wiki_available": bool(float(row.get("wiki_available", 0.0) or 0.0) > 0.0),
        "competition_available": bool(float(row.get("competitor_count_lag7", 0.0) or 0.0) > 0.0),
        "amc_available": bool(float(row.get("amc_available", 0.0) or 0.0) > 0.0),
        "amc_cutoff": row.get("amc_cutoff", ""),
        "forecast_stage": forecast_stage_for_snapshot(
            int(row["snapshot_day"]),
            target_day_count=int(row.get("target_day_count", 3) or 3),
        ),
        "model_opening_weekend_prediction_usd": None,
        "model_opening_weekend_prediction_label": "-",
        "predicted_opening_weekend_revenue_usd": None,
        "predicted_opening_weekend_revenue_label": "-",
        "predicted_lower_80_opening_weekend_revenue_usd": None,
        "predicted_upper_80_opening_weekend_revenue_usd": None,
        "predicted_lower_80_label": "-",
        "predicted_upper_80_label": "-",
        "prediction_interval_method": PRIMARY_INTERVAL_METHOD,
        "prediction_interval_source": "",
        "prediction_interval_train_n": 0,
        "prediction_certainty": None,
        "prediction_certainty_pct": None,
        "prediction_certainty_label": "-",
        "prediction_uncertainty_width_pct": None,
        "prediction_confidence_label": "-",
        "prediction_confidence_source": "unavailable",
        "prediction_confidence_method": PRIMARY_INTERVAL_METHOD,
        "prediction_confidence_train_n": 0,
    }


def actual_opening_weekend_value(rows: list[dict[str, object]], has_actual: bool) -> float | None:
    if not has_actual or not rows:
        return None
    value = float(rows[0].get("opening_weekend_revenue_usd", 0.0) or 0.0)
    return value if value > 0.0 else None


def latest_ok_snapshot(snapshots: list[dict[str, object]]) -> dict[str, object] | None:
    ok = [row for row in snapshots if row["status"] == "ok"]
    return ok[-1] if ok else None


def forecast_chart_svg(
    title: str,
    snapshots: list[dict[str, object]],
    *,
    actual_opening_weekend: float | None,
) -> str:
    points = [row for row in snapshots if row["status"] == "ok"]
    if not points:
        return (
            '<svg class="forecast-chart" viewBox="0 0 820 360" role="img" aria-label="No forecast data">'
            '<rect width="100%" height="100%" fill="#fff"/>'
            '<text x="32" y="44" font-family="Arial" font-size="18" font-weight="700">No forecast path yet</text>'
            '<text x="32" y="78" font-family="Arial" font-size="13" fill="#5d6b75">No eligible estimate is available for the visible opening-window target.</text>'
            "</svg>"
        )
    width, height = 820, 360
    left, right, top, bottom = 78, 28, 42, 66
    values: list[float] = []
    for row in points:
        values.extend(
            [
                float(row["predicted_opening_weekend_revenue_usd"]),
                float(row["predicted_lower_80_opening_weekend_revenue_usd"]),
                float(row["predicted_upper_80_opening_weekend_revenue_usd"]),
            ]
        )
    if actual_opening_weekend is not None:
        values.append(float(actual_opening_weekend))
    ymin, ymax = min(values), max(values)
    padding = (ymax - ymin) * 0.08 or max(1.0, ymax * 0.08)
    ymin = max(0.0, ymin - padding)
    ymax += padding
    xmin = min(-14, min(int(row["snapshot_day"]) for row in points))
    xmax = max(3, max(int(row["snapshot_day"]) for row in points))

    def sx(day: float) -> float:
        return left + (day - xmin) / (xmax - xmin) * (width - left - right)

    def sy(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * (height - top - bottom)

    upper = " ".join(
        f'{sx(float(row["snapshot_day"])):.1f},{sy(float(row["predicted_upper_80_opening_weekend_revenue_usd"])):.1f}'
        for row in points
    )
    lower = " ".join(
        f'{sx(float(row["snapshot_day"])):.1f},{sy(float(row["predicted_lower_80_opening_weekend_revenue_usd"])):.1f}'
        for row in reversed(points)
    )
    line = " ".join(
        f'{sx(float(row["snapshot_day"])):.1f},{sy(float(row["predicted_opening_weekend_revenue_usd"])):.1f}'
        for row in points
    )
    parts = [
        f'<svg class="forecast-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Forecast path for {escape(title)}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="32" y="28" font-family="Arial" font-size="16" font-weight="700">{escape(title)}</text>',
    ]
    for day in range(xmin, xmax + 1):
        x = sx(day)
        stroke = "#d6dee3" if day == 0 else "#edf1f3"
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom}" stroke="{stroke}" stroke-width="1"/>')
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = ymin + (ymax - ymin) * frac
        y = sy(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#edf1f3" stroke-width="1"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#5d6b75">{escape(money(value))}</text>')
    parts.extend(
        [
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#334"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#334"/>',
            f'<polygon points="{upper} {lower}" fill="#8db3b1" fill-opacity="0.35"/>',
            f'<polyline points="{line}" fill="none" stroke="#2f6f73" stroke-width="2.6"/>',
        ]
    )
    if actual_opening_weekend is not None:
        y = sy(float(actual_opening_weekend))
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#8c2727" stroke-dasharray="5 5"/>')
    for row in points:
        x = sx(float(row["snapshot_day"]))
        y = sy(float(row["predicted_opening_weekend_revenue_usd"]))
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#2f6f73"/>')
    for day in range(xmin, xmax + 1):
        label = str(day)
        parts.append(f'<text x="{sx(day):.1f}" y="{height - 30}" text-anchor="middle" font-family="Arial" font-size="10" fill="#5d6b75">{label}</text>')
    parts.append(f'<text x="{(left + width - right) / 2:.1f}" y="{height - 10}" text-anchor="middle" font-family="Arial" font-size="11" fill="#5d6b75">days from opening</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def feature_availability_summary(snapshots: list[dict[str, object]]) -> dict[str, int]:
    return {
        "bop_days": sum(1 for row in snapshots if row["bop_forecast_available"]),
        "wiki_days": sum(1 for row in snapshots if row["wiki_available"]),
        "competition_days": sum(1 for row in snapshots if row["competition_available"]),
        "amc_days": sum(1 for row in snapshots if row.get("amc_available")),
        "actual_days": max((int(row.get("actual_day_count", 0) or 0) for row in snapshots), default=0),
        "forecast_days": sum(1 for row in snapshots if row["status"] == "ok"),
    }
