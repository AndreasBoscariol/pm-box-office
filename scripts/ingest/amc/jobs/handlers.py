"""Task handlers for AMC collection workers."""

from __future__ import annotations

import datetime as dt
from typing import Any

from scripts.ingest.amc import db
from scripts.ingest.amc.client import HtmlFetcher
from scripts.ingest.amc.services import seat_service, showtime_service


def execute(conn: Any, fetcher: HtmlFetcher, task: db.CollectionTask) -> None:
    if task.task_type == "collect_theatre_showtimes":
        _collect_theatre_showtimes(conn, fetcher, task)
        return
    if task.task_type == "collect_seat_snapshot":
        _collect_seat_snapshot(conn, fetcher, task)
        return
    raise ValueError(f"Unknown AMC task type: {task.task_type}")


def _collect_theatre_showtimes(conn: Any, fetcher: HtmlFetcher, task: db.CollectionTask) -> None:
    if task.amc_theatre_id is None:
        raise ValueError(f"Inventory task {task.task_id} is missing amc_theatre_id")
    theatre = db.select_theatre_by_id(conn, task.amc_theatre_id)
    if theatre is None:
        raise ValueError(f"No AMC theatre found for task {task.task_id}: {task.amc_theatre_id}")
    showtime_service.collect_theatre_showtimes(
        conn,
        fetcher,
        theatre=theatre,
        exhibition_date=run_exhibition_date(conn, task),
    )


def _collect_seat_snapshot(conn: Any, fetcher: HtmlFetcher, task: db.CollectionTask) -> None:
    if task.showtime_id is None:
        raise ValueError(f"Seat task {task.task_id} is missing showtime_id")
    showtime = db.select_showtime_by_id(conn, task.showtime_id)
    if showtime is None:
        raise ValueError(f"No AMC showtime found for task {task.task_id}: {task.showtime_id}")
    seat_service.collect_snapshot(
        conn,
        fetcher,
        showtime=showtime,
        target_offset_minutes=task.priority or 5,
    )


def run_exhibition_date(conn: Any, task: db.CollectionTask) -> dt.date:
    row = conn.execute(
        """
        SELECT c.exhibition_date
        FROM collection_runs r
        JOIN collection_campaigns c ON c.campaign_id = r.campaign_id
        WHERE r.run_id = %s
        """,
        (task.run_id,),
    ).fetchone()
    if not row:
        raise ValueError(f"Run {task.run_id} has no campaign exhibition date")
    if isinstance(row[0], dt.date):
        return row[0]
    return dt.date.fromisoformat(str(row[0]))
