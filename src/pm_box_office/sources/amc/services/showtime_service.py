"""Showtime inventory collection service."""

from __future__ import annotations

import datetime as dt
from typing import Any

from pm_box_office.sources.amc import db
from pm_box_office.sources.amc.client import HtmlFetcher
from pm_box_office.sources.amc.parsers import (
    extract_rendered_showtimes,
    extract_showtimes,
    maybe_parse_apollo_data,
    showtimes_url,
)


def collect_theatre_showtimes(
    conn: Any,
    fetcher: HtmlFetcher,
    *,
    theatre: db.StoredTheatre,
    exhibition_date: dt.date,
) -> int:
    url = showtimes_url(exhibition_date, theatre.slug)
    result = fetcher.get_result(url)
    apollo = maybe_parse_apollo_data(result.body, source_url=url)
    rows = (
        extract_showtimes(apollo, theatre_slug=theatre.slug, date=exhibition_date)
        if apollo is not None
        else extract_rendered_showtimes(result.body, theatre_slug=theatre.slug, date=exhibition_date)
    )
    return db.upsert_showtimes(
        conn,
        theatre=theatre,
        showtimes=rows,
        raw_cache_path=str(result.cache_path) if result.cache_path is not None else None,
        fetched_at=result.fetched_at,
    )


def create_inventory_run(conn: Any, *, exhibition_date: dt.date) -> tuple[str, int]:
    campaign_id = db.ensure_campaign(conn, exhibition_date)
    active_run = db.find_active_run(conn, campaign_id=campaign_id, run_type="showtime_inventory")
    if active_run is not None:
        run_id, task_count = active_run
        return str(run_id), task_count
    run_id = db.create_run(conn, campaign_id=campaign_id, run_type="showtime_inventory", status="queued")
    theatres = db.select_active_theatres_basic(conn)
    task_count = db.create_inventory_tasks(conn, run_id=run_id, theatres=theatres)
    if task_count == 0:
        db.mark_run_status(conn, run_id, status="completed")
    return str(run_id), task_count
