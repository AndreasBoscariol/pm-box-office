"""As-of-safe overlay of reported daily actuals onto frozen baselines."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any, Iterable

import pandas as pd

from .artifacts import ModelArtifacts
from .schema import MovieOpening


def fetch_reported_actuals(
    conn: Any,
    *,
    movies: Iterable[MovieOpening],
    as_of_utc: datetime,
) -> pd.DataFrame:
    """Return final reported opening-weekend daily grosses available by ``as_of_utc``.

    ``daily_box_office`` is the live system of record.  The fetched-at predicate
    prevents a historical run from seeing a row that had not yet been ingested.
    Rows marked as estimates are deliberately excluded from the zero-variance
    actual handoff.
    """

    openings = {int(movie.release_run_id): movie for movie in movies if movie.release_run_id > 0}
    if not openings:
        return pd.DataFrame()
    vintage_exists = bool(conn.execute("SELECT to_regclass('daily_box_office_vintages') IS NOT NULL").fetchone()[0])
    source_table = "daily_box_office_vintages" if vintage_exists else "daily_box_office"
    # The immutable vintage table uses its own primary-key name. Selecting the
    # source-table key keeps the actual provenance link valid for both live and
    # versioned reporting schemas.
    record_id_column = "daily_box_office_vintage_id" if vintage_exists else "daily_box_office_id"
    cursor = conn.execute(
        f"""
        SELECT {record_id_column} AS actual_record_id, release_run_id,
               box_office_date::date, gross_usd, source AS actual_source,
               fetched_at AS actual_ingested_at, fetched_at AS actual_published_at
        FROM {source_table}
        WHERE release_run_id = ANY(%s)
          AND box_office_date::date BETWEEN %s AND %s
          AND gross_usd > 0
          AND COALESCE(is_estimate, 0) = 0
          AND fetched_at::timestamptz <= %s
        ORDER BY release_run_id, box_office_date::date, fetched_at::timestamptz
        """,
        (
            list(openings),
            min(movie.opening_weekend_start for movie in openings.values()),
            max(movie.opening_weekend_start + timedelta(days=2) for movie in openings.values()),
            as_of_utc,
        ),
    )
    rows = cursor.fetchall()
    columns = [description[0] for description in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def overlay_reported_actuals(
    artifacts: ModelArtifacts,
    *,
    movies: Iterable[MovieOpening],
    actuals: pd.DataFrame,
) -> ModelArtifacts:
    """Overlay available actuals without mutating the frozen artifact frame."""

    if artifacts.daily_baseline.empty or actuals.empty:
        return artifacts
    baseline = artifacts.daily_baseline.copy()
    baseline["release_run_id"] = pd.to_numeric(baseline["release_run_id"], errors="coerce")
    work = actuals.copy()
    work["release_run_id"] = pd.to_numeric(work["release_run_id"], errors="coerce")
    work["box_office_date"] = pd.to_datetime(work["box_office_date"], errors="coerce").dt.date
    work["gross_usd"] = pd.to_numeric(work["gross_usd"], errors="coerce")
    timestamp_column = "actual_ingested_at" if "actual_ingested_at" in work.columns else "fetched_at"
    if timestamp_column in work.columns:
        work = work.sort_values(timestamp_column).drop_duplicates(
            ["release_run_id", "box_office_date"], keep="last"
        )

    actual_columns = ["actual_fri_usd", "actual_sat_usd", "actual_sun_usd"]
    for column in actual_columns:
        marker = f"{column}_available_as_of"
        if marker not in baseline:
            baseline[marker] = False
        baseline[marker] = baseline[marker].fillna(False).astype(bool)

    movie_by_run = {int(movie.release_run_id): movie for movie in movies}
    for row in work.itertuples(index=False):
        run_id = int(row.release_run_id)
        movie = movie_by_run.get(run_id)
        if movie is None or not pd.notna(row.gross_usd) or float(row.gross_usd) <= 0:
            continue
        offset = (row.box_office_date - movie.opening_weekend_start).days
        column = {0: "actual_fri_usd", 1: "actual_sat_usd", 2: "actual_sun_usd"}.get(offset)
        if column is None:
            continue
        mask = baseline["release_run_id"].eq(run_id)
        baseline.loc[mask, column] = float(row.gross_usd)
        baseline.loc[mask, f"{column}_available_as_of"] = True
        prefix = column
        metadata = {
            f"{prefix}_source": getattr(row, "actual_source", "daily_box_office"),
            f"{prefix}_record_id": getattr(row, "actual_record_id", None),
            f"{prefix}_published_at": getattr(row, "actual_published_at", None),
            f"{prefix}_ingested_at": getattr(row, "actual_ingested_at", getattr(row, "fetched_at", None)),
            f"{prefix}_revision": getattr(row, "actual_revision", None),
        }
        for name, value in metadata.items():
            baseline.loc[mask, name] = value

    for column in actual_columns:
        if column not in baseline:
            baseline[column] = float("nan")
        marker = f"{column}_available_as_of"
    complete = baseline[actual_columns].apply(pd.to_numeric, errors="coerce")
    baseline["actual_ow_usd"] = complete.sum(axis=1, min_count=3)
    return replace(artifacts, daily_baseline=baseline)


def fetch_reported_preview_actuals_as_of(
    conn: Any,
    *,
    movies: Iterable[MovieOpening],
    as_of_utc: datetime,
) -> pd.DataFrame:
    run_ids = [int(movie.release_run_id) for movie in movies if movie.release_run_id > 0]
    if not run_ids:
        return pd.DataFrame()
    exists = bool(conn.execute("SELECT to_regclass('reported_preview_actuals') IS NOT NULL").fetchone()[0])
    if not exists:
        return pd.DataFrame()
    cursor = conn.execute(
        """
        SELECT r.release_run_id, r.preview_gross_usd, r.preview_target_type,
               r.preview_published_at, r.received_at, r.preview_source
        FROM reported_preview_actuals r
        LEFT JOIN reported_preview_actuals newer ON newer.supersedes_id = r.reported_preview_actual_id
        WHERE r.release_run_id = ANY(%s)
          AND r.preview_published_at <= %s AND r.received_at <= %s
          AND newer.reported_preview_actual_id IS NULL
        ORDER BY r.release_run_id, r.preview_published_at, r.received_at, r.revision
        """,
        (run_ids, as_of_utc, as_of_utc),
    )
    return pd.DataFrame(cursor.fetchall(), columns=[item[0] for item in cursor.description])


def overlay_reported_preview_actuals(artifacts: ModelArtifacts, previews: pd.DataFrame) -> ModelArtifacts:
    if artifacts.daily_baseline.empty or previews.empty:
        return artifacts
    baseline = artifacts.daily_baseline.copy()
    baseline["release_run_id"] = pd.to_numeric(baseline["release_run_id"], errors="coerce")
    work = previews.sort_values(["preview_published_at", "received_at"]).drop_duplicates("release_run_id", keep="last")
    for row in work.itertuples(index=False):
        mask = baseline["release_run_id"].eq(int(row.release_run_id))
        baseline.loc[mask, "thursday_preview_gross_usd"] = float(row.preview_gross_usd)
        baseline.loc[mask, "thursday_preview_cutoff_date"] = pd.Timestamp(row.preview_published_at).date()
        baseline.loc[mask, "thursday_preview_target_type"] = str(row.preview_target_type)
        baseline.loc[mask, "thursday_preview_published_at"] = row.preview_published_at
        baseline.loc[mask, "thursday_preview_source"] = str(row.preview_source)
    return replace(artifacts, daily_baseline=baseline)


def merge_reported_actuals(
    conn: Any,
    *,
    artifacts: ModelArtifacts,
    movies: Iterable[MovieOpening],
    as_of_utc: datetime,
) -> ModelArtifacts:
    movies = list(movies)
    actuals = fetch_reported_actuals(conn, movies=movies, as_of_utc=as_of_utc)
    updated = overlay_reported_actuals(artifacts, movies=movies, actuals=actuals)
    previews = fetch_reported_preview_actuals_as_of(conn, movies=movies, as_of_utc=as_of_utc)
    return overlay_reported_preview_actuals(updated, previews)
