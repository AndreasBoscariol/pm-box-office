"""Campaign movie selection service."""

from __future__ import annotations

import datetime as dt
from typing import Any

from pm_box_office.sources.amc import db


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


def set_movies_selected(
    conn: Any,
    *,
    exhibition_date: dt.date,
    amc_movie_ids: list[str],
    selected: bool,
) -> int:
    campaign_id = db.ensure_campaign(conn, exhibition_date)
    count = db.set_campaign_movies_selected(
        conn,
        campaign_id=campaign_id,
        amc_movie_ids=amc_movie_ids,
        selected=selected,
    )
    if not selected:
        for amc_movie_id in amc_movie_ids:
            db.cancel_future_seat_tasks_for_movie(
                conn,
                campaign_id=campaign_id,
                amc_movie_id=amc_movie_id,
            )
    return count


def create_seat_collection_run(
    conn: Any,
    *,
    exhibition_date: dt.date,
    target_offsets_minutes: tuple[int, ...] = (5,),
) -> tuple[str, int]:
    campaign_id = db.ensure_campaign(conn, exhibition_date)
    active_run = db.find_active_run(conn, campaign_id=campaign_id, run_type="seat_collection")
    if active_run is not None:
        run_id, task_count = active_run
        return str(run_id), task_count
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
            target_offsets_minutes=target_offsets_minutes,
        )
    if task_count == 0:
        db.mark_run_status(conn, run_id, status="completed")
    return str(run_id), task_count
