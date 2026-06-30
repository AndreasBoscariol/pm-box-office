"""Campaign movie selection service."""

from __future__ import annotations

import datetime as dt
from typing import Any

from scripts.ingest.amc import db


def list_movies_for_date(conn: Any, *, exhibition_date: dt.date) -> list[db.MovieInventoryRow]:
    return db.list_movies_for_date(conn, exhibition_date)


def set_movie_selected(
    conn: Any,
    *,
    exhibition_date: dt.date,
    amc_movie_id: str,
    selected: bool,
) -> None:
    campaign_id = db.ensure_campaign(conn, exhibition_date)
    db.set_campaign_movie_selected(
        conn,
        campaign_id=campaign_id,
        amc_movie_id=amc_movie_id,
        selected=selected,
    )
    if not selected:
        db.cancel_future_seat_tasks_for_movie(
            conn,
            campaign_id=campaign_id,
            amc_movie_id=amc_movie_id,
        )


def create_seat_collection_run(
    conn: Any,
    *,
    exhibition_date: dt.date,
    target_offset_minutes: int = 5,
) -> tuple[str, int]:
    campaign_id = db.ensure_campaign(conn, exhibition_date)
    run_id = db.create_run(conn, campaign_id=campaign_id, run_type="seat_collection", status="queued")
    selected_rows = [
        row
        for row in db.list_movies_for_date(conn, exhibition_date)
        if row.selected
    ]
    task_count = 0
    for movie in selected_rows:
        showtimes = db.select_showtimes_for_target(
            conn,
            target_date=exhibition_date.isoformat(),
            target_amc_movie_id=movie.amc_movie_id,
            target_amc_movie_name=None,
        )
        task_count += db.create_seat_scan_tasks(
            conn,
            run_id=run_id,
            showtimes=showtimes,
            target_offset_minutes=target_offset_minutes,
        )
    return str(run_id), task_count
