"""HTTP client helpers for AMC ingestion."""

from __future__ import annotations

import hashlib
import datetime as dt
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from pm_box_office.sources.amc import diagnostics


DEFAULT_CACHE_DIR = Path("data/raw/amc")
DEFAULT_USER_AGENT = "pm-box-office-amc/0.1"
TRANSIENT_STATUSES = {429, 500, 502, 503, 504}


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
    ) -> None:
        self.cache_dir = cache_dir
        self.refresh = refresh
        self.offline = offline
        self.delay_seconds = delay_seconds
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self._last_request_at = 0.0
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(self, url: str) -> Path:
        suffix = ".xml" if url.endswith(".xml") else ".html"
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}{suffix}"

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
                cache_path.write_text(body, encoding="utf-8")
                if archive_path is not None:
                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                    archive_path.write_text(body, encoding="utf-8")
                self._last_request_at = time.monotonic()
                return FetchResult(
                    body=body,
                    source_url=url,
                    fetched_at=dt.datetime.now(dt.timezone.utc),
                    cache_path=archive_path or cache_path,
                    from_cache=False,
                    status_code=status_code,
                )
            except urllib.error.HTTPError as exc:
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
                    raise
                delay = float(retry_after) if retry_after else self.delay_seconds * (attempt + 1)
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
                time.sleep(delay)
            except (TimeoutError, urllib.error.URLError) as exc:
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
