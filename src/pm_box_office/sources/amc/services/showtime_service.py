"""Showtime inventory collection service."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from pm_box_office.sources.amc import db
from pm_box_office.sources.amc.client import FetchResult, HtmlFetcher
from pm_box_office.sources.amc.parsers import (
    ShowtimeRecord,
    extract_rendered_showtimes,
    extract_showtimes,
    maybe_parse_apollo_data,
    showtimes_url,
)


DEFAULT_INVENTORY_DAYS = 7


@dataclass(frozen=True)
class ShowtimePage:
    rows: list[ShowtimeRecord]
    result: FetchResult


def collect_theatre_showtimes(
    conn: Any,
    fetcher: HtmlFetcher,
    *,
    theatre: db.StoredTheatre,
    exhibition_date: dt.date,
    inventory_days: int = DEFAULT_INVENTORY_DAYS,
) -> int:
    if inventory_days < 1:
        raise ValueError("inventory_days must be at least 1")

    target_dates = {exhibition_date + dt.timedelta(days=offset) for offset in range(inventory_days)}
    pending_dates = set(target_dates)
    seen_showtime_ids: set[str] = set()
    upserted_count = 0

    while pending_dates:
        page_date = min(pending_dates)
        page = fetch_theatre_showtime_page(fetcher, theatre_slug=theatre.slug, exhibition_date=page_date)
        page_rows = [
            row
            for row in page.rows
            if dt.date.fromisoformat(row.date) in target_dates and row.showtime_id not in seen_showtime_ids
        ]
        for row in page_rows:
            seen_showtime_ids.add(row.showtime_id)

        observed_dates = {dt.date.fromisoformat(row.date) for row in page_rows}
        pending_dates.difference_update(observed_dates)
        pending_dates.discard(page_date)

        if not page_rows:
            continue
        upserted_count += db.upsert_showtimes(
            conn,
            theatre=theatre,
            showtimes=page_rows,
            raw_cache_path=str(page.result.cache_path) if page.result.cache_path is not None else None,
            fetched_at=page.result.fetched_at,
        )

    return upserted_count


def fetch_theatre_showtime_page(
    fetcher: HtmlFetcher,
    *,
    theatre_slug: str,
    exhibition_date: dt.date,
) -> ShowtimePage:
    url = showtimes_url(exhibition_date, theatre_slug)
    result = fetcher.get_result(url)
    apollo = maybe_parse_apollo_data(result.body, source_url=url)
    rows = (
        extract_showtimes(apollo, theatre_slug=theatre_slug, date=exhibition_date)
        if apollo is not None
        else extract_rendered_showtimes(result.body, theatre_slug=theatre_slug, date=exhibition_date)
    )
    return ShowtimePage(rows=rows, result=result)


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
