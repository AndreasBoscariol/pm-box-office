"""HTTP client helpers for AMC ingestion."""

from __future__ import annotations

import datetime as dt
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from pm_box_office.sources.amc import diagnostics
from pm_box_office.sources.common.fetch import (
    DEFAULT_TRANSIENT_STATUSES,
    cache_path_for_url,
    retry_delay_seconds,
)


DEFAULT_CACHE_DIR = Path("data/raw/amc")
DEFAULT_USER_AGENT = "pm-box-office-amc/0.1"
TRANSIENT_STATUSES = DEFAULT_TRANSIENT_STATUSES


@dataclass(frozen=True)
class FetchResult:
    body: str
    source_url: str
    fetched_at: dt.datetime
    cache_path: Path | None
    from_cache: bool
    status_code: int


class HtmlFetcher:
    """Cache-first AMC fetcher with polite request spacing."""

    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        *,
        refresh: bool = False,
        offline: bool = False,
        delay_seconds: float = 1.0,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout_seconds: float = 45.0,
        retries: int = 3,
        diagnostics_conn: object | None = None,
        rate_limit_conn: object | None = None,
        global_requests_per_minute: int | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.refresh = refresh
        self.offline = offline
        self.delay_seconds = delay_seconds
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.diagnostics_conn = diagnostics_conn
        self.rate_limit_conn = rate_limit_conn
        self.global_requests_per_minute = (
            int(os.environ.get("AMC_GLOBAL_REQUESTS_PER_MINUTE", "20"))
            if global_requests_per_minute is None
            else global_requests_per_minute
        )
        self.global_rate_limit_enabled = os.environ.get("AMC_GLOBAL_RATE_LIMIT_ENABLED", "true").lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self._last_request_at = 0.0
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(self, url: str) -> Path:
        suffix = ".xml" if url.endswith(".xml") else ".html"
        return cache_path_for_url(self.cache_dir, url, suffix=suffix)

    def get_result(
        self,
        url: str,
        *,
        refresh: bool | None = None,
        archive_path: Path | None = None,
    ) -> FetchResult:
        cache_path = self.cache_path(url)
        should_refresh = self.refresh if refresh is None else refresh
        kind = diagnostics.url_kind(url)
        now = dt.datetime.now(dt.timezone.utc)
        if cache_path.exists() and not should_refresh:
            return FetchResult(
                body=cache_path.read_text(encoding="utf-8"),
                source_url=url,
                fetched_at=now,
                cache_path=cache_path,
                from_cache=True,
                status_code=200,
            )
        if self.offline:
            raise FileNotFoundError(f"Cache miss in offline mode: {url}")

        last_error: Exception | None = None
        for attempt in range(self.retries):
            self._wait()
            self._wait_for_global_request_slot(url, kind)
            request_started_at = time.monotonic()
            self._record_request_event("http_request_attempt", url=url, url_kind=kind, attempt=attempt + 1)
            headers = {
                "Accept": "text/html,application/xhtml+xml,application/xml,text/xml",
                "Accept-Encoding": "identity",
                "User-Agent": self.user_agent,
            }
            if "_rsc=" in url:
                headers.update({"Accept": "text/x-component", "RSC": "1"})
            request = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    body = response.read().decode("utf-8", errors="replace")
                    status_code = int(response.status)
                duration_ms = int((time.monotonic() - request_started_at) * 1000)
                if not body.strip():
                    diagnostics.log_backoff_event(
                        "empty_body",
                        url=url,
                        url_kind=kind,
                        status_code=status_code,
                        attempt=attempt + 1,
                        body_length=len(body),
                        cache_path=cache_path,
                        archive_path=archive_path,
                    )
                    exc = RuntimeError(f"GET {url} returned an empty response body")
                    diagnostics.log_backoff_event(
                        "http_failed",
                        url=url,
                        url_kind=kind,
                        status_code=status_code,
                        attempt=attempt + 1,
                        body_length=len(body),
                        cache_path=cache_path,
                        archive_path=archive_path,
                        error_type=type(exc).__name__,
                        error_message=diagnostics.short_error(exc),
                    )
                    raise exc
                write_path = archive_path or cache_path
                write_path.parent.mkdir(parents=True, exist_ok=True)
                write_path.write_text(body, encoding="utf-8")
                self._last_request_at = time.monotonic()
                self._record_request_event(
                    "http_request_success",
                    url=url,
                    url_kind=kind,
                    attempt=attempt + 1,
                    status_code=status_code,
                    duration_ms=duration_ms,
                    metadata={"body_length": len(body)},
                )
                return FetchResult(
                    body=body,
                    source_url=url,
                    fetched_at=dt.datetime.now(dt.timezone.utc),
                    cache_path=archive_path or cache_path,
                    from_cache=False,
                    status_code=status_code,
                )
            except urllib.error.HTTPError as exc:
                duration_ms = int((time.monotonic() - request_started_at) * 1000)
                last_error = exc
                retry_after = exc.headers.get("Retry-After")
                if exc.code not in TRANSIENT_STATUSES or attempt == self.retries - 1:
                    diagnostics.log_backoff_event(
                        "http_failed",
                        url=url,
                        url_kind=kind,
                        status_code=exc.code,
                        attempt=attempt + 1,
                        retry_after=retry_after,
                        cache_path=cache_path,
                        archive_path=archive_path,
                        error_type=type(exc).__name__,
                        error_message=diagnostics.short_error(exc),
                    )
                    self._record_request_event(
                        "http_request_failed",
                        url=url,
                        url_kind=kind,
                        attempt=attempt + 1,
                        status_code=exc.code,
                        duration_ms=duration_ms,
                        metadata={"error_type": type(exc).__name__, "error_message": diagnostics.short_error(exc)},
                    )
                    raise
                delay = retry_delay_seconds(exc, fallback_seconds=self.delay_seconds * (attempt + 1))
                diagnostics.log_backoff_event(
                    "http_retry",
                    url=url,
                    url_kind=kind,
                    status_code=exc.code,
                    attempt=attempt + 1,
                    retry_delay_seconds=delay,
                    retry_after=retry_after,
                    cache_path=cache_path,
                    archive_path=archive_path,
                    error_type=type(exc).__name__,
                    error_message=diagnostics.short_error(exc),
                )
                self._record_request_event(
                    "http_retry_scheduled",
                    url=url,
                    url_kind=kind,
                    attempt=attempt + 1,
                    status_code=exc.code,
                    duration_ms=duration_ms,
                    metadata={"retry_delay_seconds": delay, "error_type": type(exc).__name__},
                )
                time.sleep(delay)
            except (TimeoutError, urllib.error.URLError) as exc:
                duration_ms = int((time.monotonic() - request_started_at) * 1000)
                last_error = exc
                if attempt == self.retries - 1:
                    break
                delay = self.delay_seconds * (attempt + 1)
                diagnostics.log_backoff_event(
                    "http_retry",
                    url=url,
                    url_kind=kind,
                    attempt=attempt + 1,
                    retry_delay_seconds=delay,
                    cache_path=cache_path,
                    archive_path=archive_path,
                    error_type=type(exc).__name__,
                    error_message=diagnostics.short_error(exc),
                )
                self._record_request_event(
                    "http_retry_scheduled",
                    url=url,
                    url_kind=kind,
                    attempt=attempt + 1,
                    duration_ms=duration_ms,
                    metadata={"retry_delay_seconds": delay, "error_type": type(exc).__name__},
                )
                time.sleep(delay)
        exc = RuntimeError(f"GET {url} failed after retry: {last_error}")
        logged_error = last_error or exc
        diagnostics.log_backoff_event(
            "http_failed",
            url=url,
            url_kind=kind,
            attempt=self.retries,
            cache_path=cache_path,
            archive_path=archive_path,
            error_type=type(logged_error).__name__,
            error_message=diagnostics.short_error(logged_error),
        )
        self._record_request_event(
            "http_request_failed",
            url=url,
            url_kind=kind,
            attempt=self.retries,
            metadata={"error_type": type(logged_error).__name__, "error_message": diagnostics.short_error(logged_error)},
        )
        raise exc

    def get_live_result(self, url: str, *, archive_path: Path | None = None) -> FetchResult:
        """Fetch from AMC even when a cache file exists.

        Seat maps use this path so a later observation cannot silently reuse an
        older seat-state page.
        """

        return self.get_result(url, refresh=True, archive_path=archive_path)

    def get(self, url: str) -> tuple[str, Path, bool]:
        result = self.get_result(url)
        if result.cache_path is None:
            raise RuntimeError(f"GET {url} did not produce a cache path")
        return result.body, result.cache_path, not result.from_cache

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        delay = max(0.0, self.delay_seconds - elapsed)
        if delay:
            time.sleep(delay)

    def _wait_for_global_request_slot(self, url: str, url_kind: str) -> None:
        if not self.global_rate_limit_enabled or self.rate_limit_conn is None:
            return
        from pm_box_office.sources.amc import db

        while True:
            wait_until = db.acquire_live_request_slot(
                self.rate_limit_conn,
                max_requests_per_minute=self.global_requests_per_minute,
                worker_id=str(diagnostics.current_context().get("worker_id") or ""),
                url_kind=url_kind,
                source_url=url,
            )
            if hasattr(self.rate_limit_conn, "commit"):
                self.rate_limit_conn.commit()
            if wait_until is None:
                return
            delay = max(0.0, (wait_until - db.utc_now()).total_seconds())
            time.sleep(delay)

    def _record_request_event(
        self,
        event_type: str,
        *,
        url: str,
        url_kind: str,
        attempt: int,
        status_code: int | None = None,
        duration_ms: int | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if self.diagnostics_conn is None:
            return
        from pm_box_office.sources.amc import db

        context = diagnostics.current_context()
        try:
            db.record_collection_diagnostic_event(
                self.diagnostics_conn,
                event_type=event_type,
                run_id=context.get("run_id"),
                task_id=int(context["task_id"]) if context.get("task_id") is not None else None,
                worker_id=str(context.get("worker_id") or ""),
                showtime_id=str(context.get("showtime_id")) if context.get("showtime_id") is not None else None,
                amc_movie_id=str(context.get("amc_movie_id")) if context.get("amc_movie_id") is not None else None,
                amc_theatre_id=int(context["amc_theatre_id"]) if context.get("amc_theatre_id") is not None else None,
                url_kind=url_kind,
                status_code=status_code,
                attempt_count=attempt,
                scheduled_for=db.row_datetime(context["scheduled_for"]) if context.get("scheduled_for") else None,
                duration_ms=duration_ms,
                metadata={"url": url, **(metadata or {})},
            )
            if hasattr(self.diagnostics_conn, "commit"):
                self.diagnostics_conn.commit()
        except Exception:
            if hasattr(self.diagnostics_conn, "rollback"):
                self.diagnostics_conn.rollback()
