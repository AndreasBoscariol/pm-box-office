"""Future release candidates for production forecast generation.

These helpers bridge the gap between upstream estimate/AMC inventory tables and
the canonical forecast tables. They deliberately do not create source identity
records or release runs; they build stable, read-time candidate openings that can
be persisted as forecast rows.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from .constants import PRE_RELEASE_ORIGIN_DAYS
from .schema import MovieOpening


@dataclass(frozen=True)
class FutureCandidateArtifacts:
    movies: list[MovieOpening]
    pre_release_panel: pd.DataFrame
    daily_baseline: pd.DataFrame


def normalize_title_key(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\([^)]*(?:wide|limited|imax|dolby|\d{4})[^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def virtual_release_run_id(movie_id: int, opening_weekend_start: date) -> int:
    raw = f"future-opening:{movie_id}:{opening_weekend_start.isoformat()}".encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()
    return -(1_000_000_000_000 + (int(digest[:12], 16) % 8_000_000_000_000))


def fetch_frame(conn: Any, sql: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    cursor = conn.execute(sql, params)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def relation_exists(conn: Any, name: str, schema: str = "public") -> bool:
    row = conn.execute(
        """
        SELECT to_regclass(%s) IS NOT NULL
        """,
        (f"{schema}.{name}" if schema != "public" else name,),
    ).fetchone()
    return bool(row and row[0])


def _safe_date(value: Any) -> date | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        return None
    return parsed.date()


def _opening_weekend_start(value: Any) -> date | None:
    day = _safe_date(value)
    if day is None:
        return None
    return day + timedelta(days=(4 - day.weekday()) % 7)


def _season_bucket(day: date) -> str:
    if day.month in {5, 6, 7, 8}:
        return "summer"
    if day.month in {11, 12}:
        return "holiday"
    if day.month in {1, 2, 3, 4}:
        return "winter_spring"
    return "fall"


def _parse_numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _estimate_queries(conn: Any) -> list[str]:
    queries: list[str] = []
    # Some publishers use the generic "domestic_weekend" label even for a
    # film's first wide weekend.  The label is useful, but the release history
    # is the authority: a three-day prediction is an opening forecast when the
    # referenced movie has not opened before that target weekend.
    has_openings = relation_exists(conn, "eda_movie_openings", schema="analytics")

    def opening_context(alias: str, target_date: str) -> str:
        metric = f"{alias}.forecast_metric ILIKE '%%opening%%'"
        if not has_openings:
            return metric
        return f"""(
            {metric}
            OR NOT EXISTS (
                SELECT 1
                FROM analytics.eda_movie_openings opening
                WHERE opening.movie_id = {alias}.movie_id
                  AND opening.opening_weekend_start < {target_date}::date
            )
            AND EXISTS (
                SELECT 1
                FROM movies movie
                WHERE movie.movie_id = {alias}.movie_id
                  AND movie.release_date BETWEEN {target_date}::date
                      AND {target_date}::date + 2
            )
        )"""

    if relation_exists(conn, "boxofficepro_weekend_predictions"):
        article_date = "p.target_start_date"
        if relation_exists(conn, "boxofficepro_articles"):
            article_date = "COALESCE(a.discovered_date, p.target_start_date)"
        queries.append(
            f"""
            SELECT
                'boxofficepro'::text AS estimate_source,
                p.prediction_id::bigint AS source_prediction_id,
                p.movie_id::bigint AS source_movie_id,
                p.source_movie_title,
                p.target_start_date::date AS release_date,
                {article_date}::date AS estimate_date,
                p.forecast_metric,
                (p.target_end_date - p.target_start_date + 1)::integer AS target_day_count,
                p.range_low_usd::numeric AS estimate_low_usd,
                p.range_high_usd::numeric AS estimate_high_usd,
                ((p.range_low_usd::numeric + p.range_high_usd::numeric) / 2.0) AS estimate_mid_usd
            FROM boxofficepro_weekend_predictions p
            {"LEFT JOIN boxofficepro_articles a ON a.article_id = p.article_id" if relation_exists(conn, "boxofficepro_articles") else ""}
            WHERE p.range_low_usd IS NOT NULL
              AND p.range_high_usd IS NOT NULL
              AND p.target_start_date IS NOT NULL
              AND p.target_end_date IS NOT NULL
              AND (p.target_end_date - p.target_start_date) = 2
              AND {opening_context('p', 'p.target_start_date')}
            """
        )
    if relation_exists(conn, "boxofficereport_weekend_predictions"):
        article_date = "p.target_start_date"
        if relation_exists(conn, "boxofficereport_articles"):
            article_date = "COALESCE(p.prediction_made_at::date, a.prediction_made_date, p.target_start_date)"
        queries.append(
            f"""
            SELECT
                'boxofficereport'::text AS estimate_source,
                p.prediction_id::bigint AS source_prediction_id,
                p.movie_id::bigint AS source_movie_id,
                p.source_movie_title,
                p.target_start_date::date AS release_date,
                {article_date}::date AS estimate_date,
                p.forecast_metric,
                (p.target_end_date - p.target_start_date + 1)::integer AS target_day_count,
                p.weekend_gross_prediction_usd::numeric AS estimate_low_usd,
                p.weekend_gross_prediction_usd::numeric AS estimate_high_usd,
                p.weekend_gross_prediction_usd::numeric AS estimate_mid_usd
            FROM boxofficereport_weekend_predictions p
            {"LEFT JOIN boxofficereport_articles a ON a.article_id = p.article_id" if relation_exists(conn, "boxofficereport_articles") else ""}
            WHERE p.weekend_gross_prediction_usd IS NOT NULL
              AND p.target_start_date IS NOT NULL
              AND p.target_end_date IS NOT NULL
              AND (p.target_end_date - p.target_start_date) = 2
              AND {opening_context('p', 'p.target_start_date')}
            """
        )
    if relation_exists(conn, "boxofficetheory_predictions"):
        queries.append(
            f"""
            SELECT
                'boxofficetheory'::text AS estimate_source,
                p.prediction_id::bigint AS source_prediction_id,
                p.movie_id::bigint AS source_movie_id,
                p.source_movie_title,
                p.release_date::date AS release_date,
                p.prediction_made_date::date AS estimate_date,
                p.forecast_metric,
                p.opening_weekend_day_count::integer AS target_day_count,
                COALESCE(p.opening_weekend_low_usd, p.opening_weekend_pinpoint_usd)::numeric AS estimate_low_usd,
                COALESCE(p.opening_weekend_high_usd, p.opening_weekend_pinpoint_usd)::numeric AS estimate_high_usd,
                COALESCE(
                    p.opening_weekend_pinpoint_usd::numeric,
                    (p.opening_weekend_low_usd::numeric + p.opening_weekend_high_usd::numeric) / 2.0
                ) AS estimate_mid_usd
            FROM boxofficetheory_predictions p
            WHERE p.release_date IS NOT NULL
              AND {opening_context('p', 'p.release_date')}
              AND p.opening_weekend_day_count = 3
              AND COALESCE(p.opening_weekend_low_usd, p.opening_weekend_pinpoint_usd) IS NOT NULL
            """
        )
    if relation_exists(conn, "boxofficetheory_substack_predictions"):
        queries.append(
            f"""
            SELECT
                'boxofficetheory_substack'::text AS estimate_source,
                p.prediction_id::bigint AS source_prediction_id,
                p.movie_id::bigint AS source_movie_id,
                p.source_movie_title,
                p.release_date::date AS release_date,
                p.prediction_made_date::date AS estimate_date,
                p.forecast_metric,
                p.opening_weekend_day_count::integer AS target_day_count,
                COALESCE(p.opening_weekend_low_usd, p.opening_weekend_pinpoint_usd)::numeric AS estimate_low_usd,
                COALESCE(p.opening_weekend_high_usd, p.opening_weekend_pinpoint_usd)::numeric AS estimate_high_usd,
                COALESCE(
                    p.opening_weekend_pinpoint_usd::numeric,
                    (p.opening_weekend_low_usd::numeric + p.opening_weekend_high_usd::numeric) / 2.0
                ) AS estimate_mid_usd
            FROM boxofficetheory_substack_predictions p
            WHERE p.release_date IS NOT NULL
              AND {opening_context('p', 'p.release_date')}
              AND p.opening_weekend_day_count = 3
              AND COALESCE(p.opening_weekend_low_usd, p.opening_weekend_pinpoint_usd) IS NOT NULL
            """
        )
    return queries


def fetch_future_estimates(
    conn: Any,
    *,
    release_start: date,
    release_end: date,
    as_of_date: date,
) -> pd.DataFrame:
    queries = _estimate_queries(conn)
    if not queries:
        return pd.DataFrame()
    sql = " UNION ALL ".join(queries)
    frame = fetch_frame(
        conn,
        f"""
        WITH estimates AS ({sql})
        SELECT *
        FROM estimates
        WHERE release_date BETWEEN %s AND %s
          AND (estimate_date IS NULL OR estimate_date <= %s)
          AND estimate_mid_usd > 0
        """,
        (release_start, release_end, as_of_date),
    )
    if frame.empty:
        return frame
    frame["opening_weekend_start"] = frame["release_date"].map(_opening_weekend_start)
    frame["title_key"] = frame["source_movie_title"].map(normalize_title_key)
    _parse_numeric(frame, ["source_movie_id", "estimate_low_usd", "estimate_high_usd", "estimate_mid_usd"])
    return frame.dropna(subset=["title_key", "opening_weekend_start", "estimate_mid_usd"])


def fetch_future_amc_inventory(
    conn: Any,
    *,
    release_start: date,
    release_end: date,
) -> pd.DataFrame:
    if not relation_exists(conn, "amc_showtimes") or not relation_exists(conn, "movies"):
        return pd.DataFrame()
    frame = fetch_frame(
        conn,
        """
        SELECT
            COALESCE(s.movie_id, a.movie_id)::bigint AS movie_id,
            MAX(s.amc_movie_name) AS title,
            COALESCE(m.release_date, MIN(s.exhibition_date))::date AS release_date,
            MIN(s.exhibition_date)::date AS first_show_date,
            MAX(s.exhibition_date)::date AS last_show_date,
            COUNT(*)::integer AS showtime_count,
            COUNT(DISTINCT s.amc_theatre_id)::integer AS theatre_count
        FROM amc_showtimes s
        LEFT JOIN amc_movies a ON a.amc_movie_id = s.amc_movie_id
        LEFT JOIN movies m ON m.movie_id = COALESCE(s.movie_id, a.movie_id)
        WHERE COALESCE(s.movie_id, a.movie_id) IS NOT NULL
        GROUP BY COALESCE(s.movie_id, a.movie_id), m.release_date
        HAVING COALESCE(m.release_date, MIN(s.exhibition_date)) BETWEEN %s AND %s
        """,
        (release_start, release_end),
    )
    if frame.empty:
        return frame
    frame["opening_weekend_start"] = frame["release_date"].map(_opening_weekend_start)
    frame["title_key"] = frame["title"].map(normalize_title_key)
    _parse_numeric(frame, ["movie_id", "showtime_count", "theatre_count"])
    return frame.dropna(subset=["movie_id", "title_key", "opening_weekend_start"])


def fetch_existing_openings(conn: Any, *, release_start: date, release_end: date) -> dict[tuple[str, date], tuple[int, int, str]]:
    if not relation_exists(conn, "eda_movie_openings", schema="analytics"):
        return set()
    frame = fetch_frame(
        conn,
        """
        SELECT movie_id, release_run_id, title, opening_weekend_start
        FROM analytics.eda_movie_openings
        WHERE opening_weekend_start BETWEEN %s AND %s
        """,
        (release_start, release_end),
    )
    if frame.empty:
        return {}
    return {
        (normalize_title_key(row.title), pd.Timestamp(row.opening_weekend_start).date()): (
            int(row.movie_id), int(row.release_run_id), str(row.title)
        )
        for row in frame.itertuples(index=False)
    }


def _candidate_rows(estimates: pd.DataFrame, amc: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    estimate_columns = {
        "title_key", "opening_weekend_start", "estimate_date", "estimate_source",
        "source_movie_id", "source_movie_title",
    }
    # Query helpers intentionally return a bare empty frame when their source
    # tables are absent or no rows match the refresh window. Treat that as no
    # candidates rather than assuming the grouping columns exist: a metadata
    # refresh must still be able to update existing canonical forecasts.
    if estimate_columns.issubset(estimates.columns):
        for key, group in estimates.groupby(["title_key", "opening_weekend_start"], dropna=False):
            title_key, opening_weekend_start = key
            first = group.sort_values(["estimate_date", "estimate_source"], na_position="first").iloc[-1]
            movie_id = pd.to_numeric(pd.Series([first.source_movie_id]), errors="coerce").iloc[0]
            if not np.isfinite(movie_id):
                continue
            rows.append(
                {
                    "title_key": title_key,
                    "opening_weekend_start": opening_weekend_start,
                    "movie_id": int(movie_id),
                    "title": str(first.source_movie_title),
                    "rank": 20,
                }
            )
    amc_columns = {"title_key", "opening_weekend_start", "movie_id", "title"}
    if amc_columns.issubset(amc.columns):
        for row in amc.itertuples(index=False):
            movie_id = pd.to_numeric(pd.Series([row.movie_id]), errors="coerce").iloc[0]
            if not np.isfinite(movie_id):
                continue
            rows.append(
                {
                    "title_key": row.title_key,
                    "opening_weekend_start": row.opening_weekend_start,
                    "movie_id": int(movie_id),
                    "title": str(row.title),
                    "rank": 0,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["title_key", "opening_weekend_start", "movie_id", "title"])
    frame = pd.DataFrame(rows)
    return (
        frame.sort_values(["title_key", "opening_weekend_start", "rank", "movie_id"])
        .drop_duplicates(["title_key", "opening_weekend_start"], keep="first")
        .drop(columns=["rank"])
    )


def _latest_estimates_as_of(group: pd.DataFrame, as_of_day: date) -> pd.DataFrame:
    work = group.copy()
    work["estimate_date_sort"] = pd.to_datetime(work["estimate_date"], errors="coerce")
    mask = work["estimate_date_sort"].isna() | (work["estimate_date_sort"].dt.date <= as_of_day)
    work = work.loc[mask].copy()
    if work.empty:
        return work
    return work.sort_values(["estimate_source", "estimate_date_sort"]).drop_duplicates("estimate_source", keep="last")


def _panel_row(candidate: pd.Series, estimates: pd.DataFrame, origin_day: int) -> dict[str, Any] | None:
    opening = pd.Timestamp(candidate.opening_weekend_start).date()
    origin_date = opening + timedelta(days=origin_day)
    available = _latest_estimates_as_of(estimates, origin_date)
    if available.empty:
        return None
    mids = pd.to_numeric(available["estimate_mid_usd"], errors="coerce").dropna()
    lows = pd.to_numeric(available["estimate_low_usd"], errors="coerce").dropna()
    highs = pd.to_numeric(available["estimate_high_usd"], errors="coerce").dropna()
    if mids.empty or mids.median() <= 0:
        return None
    point = float(mids.median())
    log_point = float(math.exp(np.log(mids).mean())) if len(mids) else point
    sources = sorted(available["estimate_source"].dropna().astype(str).unique().tolist())
    latest = available.sort_values(["estimate_date", "estimate_source"], na_position="first").iloc[-1]
    row: dict[str, Any] = {
        "release_run_id": int(candidate.release_run_id),
        "origin_day": int(origin_day),
        "movie_id": int(candidate.movie_id),
        "title": str(candidate.title),
        "opening_weekend_start": opening,
        "forecast_origin_date": origin_date,
        "release_year": opening.year,
        "release_month": opening.month,
        "season_bucket": _season_bucket(opening),
        "release_type": "future_candidate",
        "release_width_bucket": "unknown",
        "is_wide_release": pd.NA,
        "is_large_release": pd.NA,
        "actual_opening_weekend_gross_usd": np.nan,
        "latest_estimate_date": latest.estimate_date,
        "earliest_estimate_date": available["estimate_date"].min(),
        "source_count": int(len(sources)),
        "estimate_count": int(len(available)),
        "dollar_median_consensus_usd": point,
        "mean_estimate_mid_usd": float(mids.mean()),
        "min_estimate_mid_usd": float(mids.min()),
        "max_estimate_mid_usd": float(mids.max()),
        "estimate_mid_stddev_usd": float(mids.std(ddof=1)) if len(mids) > 1 else np.nan,
        "log_mean_consensus_usd": log_point,
        "log_median_consensus_usd": point,
        "source_bias_adjusted_log_mean_consensus_usd": log_point,
        "source_bias_adjusted_log_median_consensus_usd": point,
        "log_dispersion": float(np.log(mids).std(ddof=1)) if len(mids) > 1 else 0.0,
        "estimate_sources": ",".join(sources),
        "primary_point_forecast_usd": point,
        "mean_point_forecast_usd": float(mids.mean()),
        "primary_point_method": "dollar_median_consensus_usd",
        "mean_point_method": "mean_estimate_mid_usd",
    }
    if not lows.empty:
        row["latest_estimate_low_usd"] = float(lows.median())
    if not highs.empty:
        row["latest_estimate_high_usd"] = float(highs.median())
    row["latest_estimate_mid_usd"] = point
    row["latest_estimate_width_usd"] = (
        row.get("latest_estimate_high_usd", np.nan) - row.get("latest_estimate_low_usd", np.nan)
    )
    for column in [
        "recency_weighted_log_consensus_lambda_1_usd",
        "recency_weighted_log_consensus_lambda_0_usd",
        "recency_reliability_weighted_log_consensus_lambda_1_usd",
        "source_reliability_weighted_log_consensus_usd",
    ]:
        row[column] = point
    return row


def build_pre_release_panel(candidates: pd.DataFrame, estimates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in candidates.itertuples(index=False):
        group = estimates.loc[
            estimates["title_key"].eq(candidate.title_key)
            & estimates["opening_weekend_start"].eq(candidate.opening_weekend_start)
        ].copy()
        if group.empty:
            continue
        for origin_day in PRE_RELEASE_ORIGIN_DAYS:
            row = _panel_row(pd.Series(candidate._asdict()), group, origin_day)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def build_daily_baseline(candidates: pd.DataFrame, estimates: pd.DataFrame, *, as_of_date: date) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in candidates.itertuples(index=False):
        group = estimates.loc[
            estimates["title_key"].eq(candidate.title_key)
            & estimates["opening_weekend_start"].eq(candidate.opening_weekend_start)
        ].copy()
        available = _latest_estimates_as_of(group, as_of_date)
        if available.empty:
            continue
        mids = pd.to_numeric(available["estimate_mid_usd"], errors="coerce").dropna()
        if mids.empty or mids.median() <= 0:
            continue
        ow_point = float(mids.median())
        opening = pd.Timestamp(candidate.opening_weekend_start).date()
        rows.append(
            {
                "movie_id": int(candidate.movie_id),
                "release_date": opening,
                "friday_date": opening,
                "saturday_date": opening + timedelta(days=1),
                "sunday_date": opening + timedelta(days=2),
                "pre_fri_usd": ow_point * 0.42,
                "pre_sat_usd": ow_point * 0.34,
                "pre_sun_usd": ow_point * 0.24,
                "after_fri_sat_usd": ow_point * 0.34,
                "after_fri_sun_usd": ow_point * 0.24,
                "after_sat_sun_usd": ow_point * 0.24,
                "actual_fri_usd": np.nan,
                "actual_sat_usd": np.nan,
                "actual_sun_usd": np.nan,
                "actual_ow_usd": np.nan,
                "release_run_id": int(candidate.release_run_id),
                "title": str(candidate.title),
                "pre_weekend_origin_day": -1,
                "forecast_origin_date": opening + timedelta(days=-1),
                "total_forecast_usd": ow_point,
                "total_forecast_source": "future_candidate_estimate_consensus",
                "shape_model": "default_opening_weekend_shape",
                "after_friday_baseline_model": "default_opening_weekend_shape",
                "after_saturday_baseline_model": "default_opening_weekend_shape",
            }
        )
    return pd.DataFrame(rows)


def build_future_candidate_artifacts(
    conn: Any,
    *,
    as_of_utc: datetime,
    release_start: date | None = None,
    release_end: date | None = None,
    movie_id: int | None = None,
    release_run_id: int | None = None,
) -> FutureCandidateArtifacts:
    as_of_date = as_of_utc.astimezone(timezone.utc).date()
    start = release_start or (as_of_date - timedelta(days=14))
    end = release_end or (as_of_date + timedelta(days=370))
    estimates = fetch_future_estimates(conn, release_start=start, release_end=end, as_of_date=as_of_date)
    amc = fetch_future_amc_inventory(conn, release_start=start, release_end=end)
    candidates = _candidate_rows(estimates, amc)
    if candidates.empty:
        return FutureCandidateArtifacts([], pd.DataFrame(), pd.DataFrame())

    existing = fetch_existing_openings(conn, release_start=start, release_end=end)
    for index, candidate in candidates.iterrows():
        key = (candidate["title_key"], candidate["opening_weekend_start"])
        actual = existing.get(key)
        if actual is None:
            candidates.loc[index, "release_run_id"] = virtual_release_run_id(
                int(candidate["movie_id"]), pd.Timestamp(candidate["opening_weekend_start"]).date()
            )
            continue
        actual_movie_id, actual_release_run_id, actual_title = actual
        candidates.loc[index, "movie_id"] = actual_movie_id
        candidates.loc[index, "release_run_id"] = actual_release_run_id
        candidates.loc[index, "title"] = actual_title
    candidates["release_run_id"] = candidates["release_run_id"].astype(int)
    if movie_id is not None:
        candidates = candidates.loc[candidates["movie_id"].eq(movie_id)].copy()
    if release_run_id is not None:
        candidates = candidates.loc[candidates["release_run_id"].eq(release_run_id)].copy()
    if candidates.empty:
        return FutureCandidateArtifacts([], pd.DataFrame(), pd.DataFrame())

    panel = build_pre_release_panel(candidates, estimates)
    daily = build_daily_baseline(candidates, estimates, as_of_date=as_of_date)
    movies = [
        MovieOpening(
            movie_id=int(row.movie_id),
            release_run_id=int(row.release_run_id),
            title=str(row.title),
            opening_weekend_start=pd.Timestamp(row.opening_weekend_start).date(),
        )
        for row in candidates.itertuples(index=False)
    ]
    return FutureCandidateArtifacts(movies, panel, daily)
