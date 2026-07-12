"""Seat snapshot collection and scheduling service."""

from __future__ import annotations

import datetime as dt
import logging
import os
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.boxoffice import refresh_queue
from models.boxoffice.input_snapshots import record_ingest_event
from pm_box_office.sources.amc import db
from pm_box_office.sources.amc.client import HtmlFetcher
from pm_box_office.sources.amc.parsers import (
    current_showtime_seats_rsc_url,
    current_showtime_seats_url,
    fetch_seat_fill,
)


LOGGER = logging.getLogger(__name__)
FORECAST_REFRESH_ENABLED_ENV = "AMC_FORECAST_REFRESH_ENABLED"
FORECAST_REFRESH_DEBOUNCE_SECONDS_ENV = "AMC_FORECAST_REFRESH_DEBOUNCE_SECONDS"
FORECAST_REFRESH_MODEL_VERSION_ENV = "FORECAST_REFRESH_MODEL_VERSION"


@dataclass(frozen=True)
class SeatCacheDuplicateCleanupResult:
    archived_files_scanned: int
    duplicate_paths_found: int
    duplicate_files_deleted: int
    bytes_deleted: int


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
    observed = observed_at or db.utc_now()
    db.upsert_seat_snapshot(
        conn,
        showtime=showtime,
        seat_fill=fill,
        snapshot_utc_at=observed,
        minutes_before_showtime=target_offset_minutes,
        raw_cache_path=fill.raw_cache_path,
        fetched_at=observed,
    )
    try:
        record_ingest_event(
            conn,
            source_key="amc_worker",
            source_run_id=None,
            available_at_utc=observed,
            payload={
                "showtime_id": showtime.showtime_id,
                "amc_movie_id": showtime.amc_movie_id,
                "observed_at": observed,
                "event": "seat_snapshot_collected",
            },
        )
    except Exception:
        # Seat capture is the source of truth; provenance is best-effort when
        # a deployment is still catching up on the pipeline schema.
        LOGGER.exception("could not record AMC ingest event for showtime_id=%s", showtime.showtime_id)
    enqueue_forecast_refresh(conn, showtime=showtime, observed_at=observed)


def enqueue_forecast_refresh(conn: Any, *, showtime: db.StoredShowtime, observed_at: dt.datetime) -> int | None:
    if not forecast_refresh_enabled():
        return None
    try:
        return refresh_queue.enqueue_from_amc_showtime(
            conn,
            showtime_id=showtime.showtime_id,
            model_version=os.environ.get(FORECAST_REFRESH_MODEL_VERSION_ENV, refresh_queue.DEFAULT_MODEL_VERSION),
            debounce_seconds=forecast_refresh_debounce_seconds(),
            source_updated_at=observed_at,
        )
    except Exception:
        LOGGER.exception("could not enqueue forecast refresh for showtime_id=%s", showtime.showtime_id)
        return None


def forecast_refresh_enabled() -> bool:
    raw = os.environ.get(FORECAST_REFRESH_ENABLED_ENV, "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def forecast_refresh_debounce_seconds() -> int:
    raw = os.environ.get(FORECAST_REFRESH_DEBOUNCE_SECONDS_ENV)
    if raw is None:
        return refresh_queue.DEFAULT_DEBOUNCE_SECONDS
    try:
        return max(0, int(raw))
    except ValueError:
        return refresh_queue.DEFAULT_DEBOUNCE_SECONDS


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


def cleanup_top_level_seat_cache_duplicates(
    cache_dir: Path,
    *,
    dry_run: bool = False,
) -> SeatCacheDuplicateCleanupResult:
    """Delete top-level SHA cache files that duplicate archived seat snapshots."""

    fetcher = HtmlFetcher(cache_dir, delay_seconds=0)
    archived_files_scanned = 0
    duplicate_paths: set[Path] = set()
    seats_dir = cache_dir / "seats"
    if not seats_dir.exists():
        return SeatCacheDuplicateCleanupResult(0, 0, 0, 0)

    for root, _dirs, files in os.walk(seats_dir):
        showtime_id = Path(root).name
        if not showtime_id:
            continue
        for filename in files:
            url: str | None = None
            if filename.endswith(".rsc.txt"):
                url = current_showtime_seats_rsc_url(showtime_id)
            elif filename.endswith(".html"):
                url = current_showtime_seats_url(showtime_id)
            if url is None:
                continue
            archived_files_scanned += 1
            duplicate_paths.add(fetcher.cache_path(url))

    duplicate_paths_found = 0
    duplicate_files_deleted = 0
    bytes_deleted = 0
    for path in sorted(duplicate_paths):
        if not path.is_file():
            continue
        duplicate_paths_found += 1
        size = path.stat().st_size
        bytes_deleted += size
        if not dry_run:
            path.unlink()
            duplicate_files_deleted += 1

    return SeatCacheDuplicateCleanupResult(
        archived_files_scanned=archived_files_scanned,
        duplicate_paths_found=duplicate_paths_found,
        duplicate_files_deleted=duplicate_files_deleted,
        bytes_deleted=bytes_deleted,
    )
