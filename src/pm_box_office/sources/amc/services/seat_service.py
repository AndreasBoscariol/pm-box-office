"""Seat snapshot collection and scheduling service."""

from __future__ import annotations

import datetime as dt
import urllib.parse
from pathlib import Path
from typing import Any

from pm_box_office.sources.amc import db
from pm_box_office.sources.amc.client import HtmlFetcher
from pm_box_office.sources.amc.parsers import fetch_seat_fill


def collect_snapshot(
    conn: Any,
    fetcher: HtmlFetcher,
    *,
    showtime: db.StoredShowtime,
    target_offset_minutes: int = 5,
    observed_at: dt.datetime | None = None,
) -> None:
    archive_fetcher = SeatArchiveFetcher(
        fetcher,
        exhibition_date=dt.date.fromisoformat(showtime.local_show_date),
        showtime_id=showtime.showtime_id,
    )
    fill = fetch_seat_fill(
        archive_fetcher,
        theatre_slug=showtime.theatre_slug,
        date=dt.date.fromisoformat(showtime.local_show_date),
        showtime_id=showtime.showtime_id,
        prefer_rsc=True,
    )
    db.upsert_seat_snapshot(
        conn,
        showtime=showtime,
        seat_fill=fill,
        snapshot_utc_at=observed_at or db.utc_now(),
        minutes_before_showtime=target_offset_minutes,
        raw_cache_path=fill.raw_cache_path,
        fetched_at=observed_at or db.utc_now(),
    )


class SeatArchiveFetcher:
    def __init__(self, fetcher: HtmlFetcher, *, exhibition_date: dt.date, showtime_id: str) -> None:
        self.fetcher = fetcher
        self.exhibition_date = exhibition_date
        self.showtime_id = showtime_id

    def get_live_result(self, url: str) -> Any:
        if hasattr(self.fetcher, "get_live_result"):
            return self.fetcher.get_live_result(url, archive_path=self._archive_path(url))
        return self.get_result(url)

    def get_result(self, url: str) -> Any:
        if hasattr(self.fetcher, "get_result"):
            return self.fetcher.get_result(url, refresh=True, archive_path=self._archive_path(url))
        body, cache_path, fetched = self.fetcher.get(url)

        class Result:
            pass

        result = Result()
        result.body = body
        result.cache_path = cache_path
        result.fetched_at = db.utc_now()
        result.from_cache = not fetched
        result.status_code = 200
        return result

    def get(self, url: str) -> tuple[str, Path | None, bool]:
        result = self.get_live_result(url)
        return result.body, result.cache_path, not result.from_cache

    def _archive_path(self, url: str) -> Path:
        suffix = ".rsc.txt" if "_rsc=" in urllib.parse.urlparse(url).query else ".html"
        timestamp = db.utc_now().strftime("%Y%m%dT%H%M%SZ")
        return (
            self.fetcher.cache_dir
            / "seats"
            / self.exhibition_date.isoformat()
            / self.showtime_id
            / f"{timestamp}{suffix}"
        )
