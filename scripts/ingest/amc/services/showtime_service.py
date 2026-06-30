"""Showtime inventory collection service."""

from __future__ import annotations

import datetime as dt
from typing import Any

from scripts.ingest.amc import db
from scripts.ingest.amc.client import HtmlFetcher
from scripts.ingest.amc.parsers import (
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
    result = (
        fetcher.get_result(url)
        if hasattr(fetcher, "get_result")
        else _legacy_fetch_result(fetcher, url)
    )
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
    run_id = db.create_run(conn, campaign_id=campaign_id, run_type="showtime_inventory", status="queued")
    theatres = db.select_active_theatres_basic(conn)
    task_count = db.create_inventory_tasks(conn, run_id=run_id, theatres=theatres)
    return str(run_id), task_count


def _legacy_fetch_result(fetcher: Any, url: str) -> Any:
    body, cache_path, fetched = fetcher.get(url)

    class Result:
        pass

    result = Result()
    result.body = body
    result.cache_path = cache_path
    result.fetched_at = db.utc_now()
    result.from_cache = not fetched
    return result
