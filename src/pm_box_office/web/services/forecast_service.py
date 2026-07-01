from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from dataclasses import dataclass
from html import escape
from typing import Any

from pm_box_office.research.papers import recreate_competition_opening_weekend as opening
from pm_box_office.research.papers import recreate_day_by_day_opening_weekend as day_by_day


DEFAULT_TRAIN_START_YEAR = 2022
DEFAULT_TRAIN_END_YEAR = 2025
DEFAULT_MIN_OPENING_DAY_GROSS = 5_000_000
SNAPSHOT_DAYS = list(day_by_day.DEFAULT_SNAPSHOT_DAYS)
PRIMARY_INTERVAL_METHOD = "loo_conformal_abs"
RAW_MODEL = "raw_bop_snapshot"
RESIDUAL_MODEL = "bop_residual_wiki_competition_snapshot"
FALLBACK_MODEL = day_by_day.FALLBACK_MODEL_NAME


@dataclass(frozen=True)
class ForecastCandidate:
    movie_id: int
    title: str
    release_year: int | None
    opening_date: dt.date
    latest_forecast_date: dt.date
    latest_bop_midpoint: float
    source_movie_title: str


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
class ForecastModelBundle:
    train_start_year: int
    train_end_year: int
    min_opening_day_gross: int
    models: dict[tuple[int, str], ForecastDayModel]


_MODEL_CACHE: dict[tuple[int, int, int], ForecastModelBundle] = {}


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
                a.discovered_date::date AS latest_forecast_date,
                ((p.range_low_usd + p.range_high_usd) / 2.0)::double precision AS latest_bop_midpoint
            FROM boxofficepro_weekend_predictions p
            JOIN boxofficepro_articles a ON a.article_id = p.article_id
            JOIN movies m ON m.movie_id = p.matched_movie_id
            WHERE p.matched_movie_id IS NOT NULL
              AND p.forecast_metric = 'domestic_opening_weekend'
              AND p.target_start_date IS NOT NULL
              AND p.range_low_usd > 0
              AND p.range_high_usd > 0
              AND (
                    %s = ''
                 OR m.title ILIKE %s
                 OR p.source_movie_title ILIKE %s
              )
            ORDER BY
                p.matched_movie_id,
                a.discovered_date DESC NULLS LAST,
                p.article_id DESC,
                p.prediction_id DESC
        )
        SELECT matched_movie_id, title, release_year, source_movie_title,
               opening_date, latest_forecast_date, latest_bop_midpoint
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
            "latest_forecast_date": parse_db_date(row[5]).isoformat(),
            "latest_bop_midpoint": float(row[6]),
            "latest_bop_midpoint_label": money(float(row[6])),
        }
        for row in rows
    ]


def latest_candidate_for_movie(conn: Any, movie_id: int) -> ForecastCandidate | None:
    rows = conn.execute(
        """
        SELECT
            p.matched_movie_id,
            m.title,
            m.release_year,
            p.source_movie_title,
            p.target_start_date::date AS opening_date,
            a.discovered_date::date AS latest_forecast_date,
            ((p.range_low_usd + p.range_high_usd) / 2.0)::double precision AS latest_bop_midpoint
        FROM boxofficepro_weekend_predictions p
        JOIN boxofficepro_articles a ON a.article_id = p.article_id
        JOIN movies m ON m.movie_id = p.matched_movie_id
        WHERE p.matched_movie_id = %s
          AND p.forecast_metric = 'domestic_opening_weekend'
          AND p.target_start_date IS NOT NULL
          AND p.range_low_usd > 0
          AND p.range_high_usd > 0
        ORDER BY a.discovered_date DESC NULLS LAST, p.article_id DESC, p.prediction_id DESC
        LIMIT 1
        """,
        (movie_id,),
    ).fetchone()
    if rows is None:
        return None
    return ForecastCandidate(
        movie_id=int(rows[0]),
        title=str(rows[1]),
        release_year=int(rows[2]) if rows[2] is not None else None,
        source_movie_title=str(rows[3]),
        opening_date=parse_db_date(rows[4]),
        latest_forecast_date=parse_db_date(rows[5]),
        latest_bop_midpoint=float(rows[6]),
    )


def load_movie_actual(conn: Any, *, movie_id: int, opening_date: dt.date) -> dict[str, object]:
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
         AND weekend.box_office_date::date < %s::date + INTERVAL '3 days'
        WHERE rr.movie_id = %s
        GROUP BY rr.release_run_id, opening_day.theaters, opening_day.gross_usd
        ORDER BY rr.release_run_id
        LIMIT 1
        """,
        (opening_date.isoformat(), opening_date.isoformat(), opening_date.isoformat(), movie_id),
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


def get_model_bundle(
    conn: Any,
    *,
    train_start_year: int = DEFAULT_TRAIN_START_YEAR,
    train_end_year: int = DEFAULT_TRAIN_END_YEAR,
    min_opening_day_gross: int = DEFAULT_MIN_OPENING_DAY_GROSS,
) -> ForecastModelBundle:
    key = (train_start_year, train_end_year, min_opening_day_gross)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    movies = opening.load_opening_weekend_movies(
        conn,
        min_year=train_start_year,
        max_year=train_end_year,
        min_opening_day_gross=min_opening_day_gross,
    )
    if not movies:
        bundle = ForecastModelBundle(train_start_year, train_end_year, min_opening_day_gross, {})
        _MODEL_CACHE[key] = bundle
        return bundle

    min_opening = min(movie.opening_date for movie in movies)
    max_opening = max(movie.opening_date for movie in movies)
    daily_grosses = opening.load_daily_grosses(
        conn,
        start_date=min_opening + dt.timedelta(days=min(SNAPSHOT_DAYS) - 7),
        end_date=max_opening + dt.timedelta(days=-1),
    )
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
        ]
        if len(fallback_train_rows) >= len(day_by_day.FALLBACK_TERMS) + 2:
            fitted = opening.fit_model_for_target(
                fallback_train_rows,
                day_by_day.FALLBACK_TERMS,
                "target_log_opening_weekend",
            )
            loo_residuals = day_by_day.loo_prediction_residuals_for_opening_model(
                rows=fallback_train_rows,
                terms=day_by_day.FALLBACK_TERMS,
            )
            bucket_residuals, global_residuals = day_by_day.interval_residuals_by_bucket(
                rows=fallback_train_rows,
                residuals=loo_residuals,
            )
            models[(snapshot_day, FALLBACK_MODEL)] = ForecastDayModel(
                snapshot_day=snapshot_day,
                model_name=FALLBACK_MODEL,
                terms=day_by_day.FALLBACK_TERMS,
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
        for model_name, terms in day_by_day.SNAPSHOT_MODEL_TERMS.items():
            target_key = (
                "target_log_opening_weekend"
                if model_name == "calibrated_bop_snapshot"
                else "target_log_bop_residual"
            )
            train_rows = day_by_day.rows_with_target(day_rows, target_key) if model_name != RAW_MODEL else day_rows
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

    bundle = ForecastModelBundle(train_start_year, train_end_year, min_opening_day_gross, models)
    _MODEL_CACHE[key] = bundle
    return bundle


def selected_model_for_row(bundle: ForecastModelBundle, row: dict[str, object]) -> ForecastDayModel | None:
    snapshot_day = int(row["snapshot_day"])
    if float(row.get("bop_forecast_available", 0.0) or 0.0) <= 0.0:
        return bundle.models.get((snapshot_day, FALLBACK_MODEL))
    if snapshot_day == -1:
        residual = bundle.models.get((snapshot_day, RESIDUAL_MODEL))
        if (
            residual is not None
            and float(row.get("wiki_available", 0.0) or 0.0) > 0.0
            and float(row.get("competitor_count_lag7", 0.0) or 0.0) > 0.0
        ):
            return residual
    return bundle.models.get((snapshot_day, RAW_MODEL))


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


def movie_for_candidate(conn: Any, candidate: ForecastCandidate) -> opening.OpeningWeekendMovie:
    actual = load_movie_actual(conn, movie_id=candidate.movie_id, opening_date=candidate.opening_date)
    return opening.OpeningWeekendMovie(
        movie_id=candidate.movie_id,
        title=candidate.title,
        release_year=candidate.release_year or candidate.opening_date.year,
        release_run_id=int(actual["release_run_id"]),
        opening_date=candidate.opening_date,
        opening_theaters=int(actual["opening_theaters"]),
        opening_day_gross_usd=int(actual["opening_day_gross_usd"]),
        opening_weekend_revenue_usd=int(actual["opening_weekend_revenue_usd"]),
    )


def build_movie_feature_rows(conn: Any, candidate: ForecastCandidate, *, today: dt.date) -> tuple[list[dict[str, object]], bool]:
    movie = movie_for_candidate(conn, candidate)
    has_actual = movie.opening_weekend_revenue_usd > 0
    min_as_of = candidate.opening_date + dt.timedelta(days=min(SNAPSHOT_DAYS) - 7)
    max_as_of = min(today, candidate.opening_date + dt.timedelta(days=-1))
    if max_as_of < min_as_of:
        daily_grosses: list[opening.DailyGross] = []
    else:
        daily_grosses = opening.load_daily_grosses(conn, start_date=min_as_of, end_date=max_as_of)
    wiki_by_movie = opening.load_wiki_feature_map(conn, movies=[movie], timing_days=SNAPSHOT_DAYS)
    forecasts = opening.load_boxofficepro_forecasts(
        conn,
        min_target_date=candidate.opening_date,
        max_target_date=candidate.opening_date,
    )
    rows = day_by_day.build_day_by_day_feature_panel(
        [movie],
        daily_grosses,
        wiki_by_movie,
        forecasts,
        snapshot_days=SNAPSHOT_DAYS,
        train_start_year=DEFAULT_TRAIN_START_YEAR,
        train_end_year=DEFAULT_TRAIN_END_YEAR,
    )
    visible_rows = [
        row
        for row in rows
        if parse_db_date(row["as_of_date"]) <= today
        and int(row["snapshot_day"]) < 0
    ]
    return visible_rows, has_actual


def forecast_movie(
    conn: Any,
    *,
    movie_id: int,
    today: dt.date | None = None,
    train_start_year: int = DEFAULT_TRAIN_START_YEAR,
    train_end_year: int = DEFAULT_TRAIN_END_YEAR,
    min_opening_day_gross: int = DEFAULT_MIN_OPENING_DAY_GROSS,
) -> dict[str, object] | None:
    today = today or dt.date.today()
    candidate = latest_candidate_for_movie(conn, movie_id)
    if candidate is None:
        return None

    bundle = get_model_bundle(
        conn,
        train_start_year=train_start_year,
        train_end_year=train_end_year,
        min_opening_day_gross=min_opening_day_gross,
    )
    rows, has_actual = build_movie_feature_rows(conn, candidate, today=today)
    snapshots: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: int(item["snapshot_day"])):
        model = selected_model_for_row(bundle, row)
        if model is None:
            snapshots.append(empty_snapshot(row))
            continue
        pred_log = predict_log_for_day(model, row)
        interval = day_by_day.interval_for_row(
            row,
            pred_log=pred_log,
            bucket_residuals=model.bucket_residuals,
            global_residuals=model.global_residuals,
            interval_method=PRIMARY_INTERVAL_METHOD,
        )
        pred_gross = max(1.0, math.exp(pred_log))
        has_bop = float(row.get("bop_forecast_available", 0.0) or 0.0) > 0.0
        snapshots.append(
            {
                "snapshot_day": int(row["snapshot_day"]),
                "as_of_date": str(row["as_of_date"]),
                "status": "ok",
                "model": model.model_name,
                "prediction_source": prediction_source_for_model(model.model_name),
                "bop_forecast_available": has_bop,
                "bop_forecast_midpoint": float(row.get("bop_forecast_midpoint", 0.0) or 0.0),
                "bop_forecast_midpoint_label": money(float(row.get("bop_forecast_midpoint", 0.0) or 0.0)) if has_bop else "-",
                "bop_forecast_date": row.get("bop_forecast_published_date", ""),
                "bop_estimate_bucket": day_by_day.estimate_bucket_label(row),
                "wiki_available": bool(float(row.get("wiki_available", 0.0) or 0.0) > 0.0),
                "competition_available": bool(float(row.get("competitor_count_lag7", 0.0) or 0.0) > 0.0),
                "predicted_opening_weekend_revenue_usd": pred_gross,
                "predicted_opening_weekend_revenue_label": money(pred_gross),
                "predicted_lower_80_opening_weekend_revenue_usd": float(
                    interval["predicted_lower_80_opening_weekend_revenue_usd"]
                ),
                "predicted_upper_80_opening_weekend_revenue_usd": float(
                    interval["predicted_upper_80_opening_weekend_revenue_usd"]
                ),
                "predicted_lower_80_label": money(float(interval["predicted_lower_80_opening_weekend_revenue_usd"])),
                "predicted_upper_80_label": money(float(interval["predicted_upper_80_opening_weekend_revenue_usd"])),
                "prediction_interval_method": interval["prediction_interval_method"],
                "prediction_interval_source": interval["prediction_interval_source"],
                "prediction_interval_train_n": interval["prediction_interval_train_n"],
            }
        )

    actual = actual_opening_weekend_value(rows, has_actual)
    latest = latest_ok_snapshot(snapshots)
    response = {
        "movie": {
            "movie_id": candidate.movie_id,
            "title": candidate.title,
            "release_year": candidate.release_year,
            "opening_date": candidate.opening_date.isoformat(),
            "days_until_opening": (candidate.opening_date - today).days,
            "release_timing_label": release_timing_label(candidate.opening_date, today),
            "source_movie_title": candidate.source_movie_title,
            "latest_bop_midpoint": candidate.latest_bop_midpoint,
            "latest_bop_midpoint_label": money(candidate.latest_bop_midpoint),
            "latest_forecast_date": candidate.latest_forecast_date.isoformat(),
            "has_actual": has_actual,
            "actual_opening_weekend_revenue_usd": actual,
            "actual_opening_weekend_label": money(actual) if actual is not None else "",
        },
        "latest_snapshot": latest,
        "snapshots": snapshots,
        "chart_svg": forecast_chart_svg(candidate.title, snapshots, actual_opening_weekend=actual),
        "model_training": {
            "train_start_year": bundle.train_start_year,
            "train_end_year": bundle.train_end_year,
            "min_opening_day_gross": bundle.min_opening_day_gross,
        },
    }
    return response


def prediction_source_for_model(model_name: str) -> str:
    if model_name == RAW_MODEL:
        return "bop"
    if model_name == FALLBACK_MODEL:
        return "fallback_model"
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
        "prediction_source": "",
        "bop_forecast_available": has_bop,
        "bop_forecast_midpoint": float(row.get("bop_forecast_midpoint", 0.0) or 0.0),
        "bop_forecast_midpoint_label": money(float(row.get("bop_forecast_midpoint", 0.0) or 0.0)) if has_bop else "-",
        "bop_forecast_date": row.get("bop_forecast_published_date", ""),
        "bop_estimate_bucket": day_by_day.estimate_bucket_label(row),
        "wiki_available": bool(float(row.get("wiki_available", 0.0) or 0.0) > 0.0),
        "competition_available": bool(float(row.get("competitor_count_lag7", 0.0) or 0.0) > 0.0),
        "predicted_opening_weekend_revenue_usd": None,
        "predicted_opening_weekend_revenue_label": "-",
        "predicted_lower_80_opening_weekend_revenue_usd": None,
        "predicted_upper_80_opening_weekend_revenue_usd": None,
        "predicted_lower_80_label": "-",
        "predicted_upper_80_label": "-",
        "prediction_interval_method": PRIMARY_INTERVAL_METHOD,
        "prediction_interval_source": "",
        "prediction_interval_train_n": 0,
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
            '<text x="32" y="78" font-family="Arial" font-size="13" fill="#5d6b75">No eligible BOP estimate is available for visible pre-release days.</text>'
            "</svg>"
        )
    width, height = 820, 360
    left, right, top, bottom = 74, 28, 42, 58
    xs = [int(row["snapshot_day"]) for row in points]
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
    xmin, xmax = min(SNAPSHOT_DAYS), max(SNAPSHOT_DAYS)

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
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#334"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#334"/>',
        f'<polygon points="{upper} {lower}" fill="#8db3b1" fill-opacity="0.35"/>',
        f'<polyline points="{line}" fill="none" stroke="#2f6f73" stroke-width="2.6"/>',
    ]
    if actual_opening_weekend is not None:
        y = sy(float(actual_opening_weekend))
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#8c2727" stroke-dasharray="5 5"/>')
    for row in points:
        x = sx(float(row["snapshot_day"]))
        y = sy(float(row["predicted_opening_weekend_revenue_usd"]))
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#2f6f73"/>')
    for day in SNAPSHOT_DAYS:
        if day in {-14, -10, -7, -3, -1}:
            parts.append(f'<text x="{sx(day):.1f}" y="{height - 28}" text-anchor="middle" font-family="Arial" font-size="11" fill="#5d6b75">t={day}</text>')
    for frac in (0.0, 0.5, 1.0):
        value = ymin + (ymax - ymin) * frac
        parts.append(f'<text x="{left - 8}" y="{sy(value) + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#5d6b75">{escape(money(value))}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def feature_availability_summary(snapshots: list[dict[str, object]]) -> dict[str, int]:
    return {
        "bop_days": sum(1 for row in snapshots if row["bop_forecast_available"]),
        "wiki_days": sum(1 for row in snapshots if row["wiki_available"]),
        "competition_days": sum(1 for row in snapshots if row["competition_available"]),
        "forecast_days": sum(1 for row in snapshots if row["status"] == "ok"),
    }
