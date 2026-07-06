"""Cache-first HTTP fetch helpers shared by source ingests."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import urllib.error
import urllib.request


DEFAULT_TRANSIENT_STATUSES = {429, 500, 502, 503, 504}


def cache_path_for_url(cache_dir: Path, url: str, suffix: str = ".html") -> Path:
    normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}{normalized_suffix}"


def retry_delay_seconds(exc: urllib.error.HTTPError, *, fallback_seconds: float) -> float:
    retry_after = exc.headers.get("Retry-After")
    return float(retry_after) if retry_after else fallback_seconds


@dataclass(frozen=True)
class FetchResult:
    body: bytes
    source_url: str
    fetched_at: dt.datetime
    cache_path: Path
    from_cache: bool
    status_code: int


class CacheFirstFetcher:
    """Small urllib-based fetcher with hashed disk cache and polite retries."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        refresh: bool = False,
        offline: bool = False,
        delay_seconds: float = 1.0,
        user_agent: str,
        timeout_seconds: float = 60.0,
        retries: int = 3,
        transient_statuses: set[int] | None = None,
        default_accept: str = "*/*",
        offline_error_prefix: str = "Cache miss in offline mode",
    ) -> None:
        self.cache_dir = cache_dir
        self.refresh = refresh
        self.offline = offline
        self.delay_seconds = delay_seconds
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.transient_statuses = transient_statuses or DEFAULT_TRANSIENT_STATUSES
        self.default_accept = default_accept
        self.offline_error_prefix = offline_error_prefix
        self._last_request_at = 0.0
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(self, url: str, suffix: str = ".html") -> Path:
        return cache_path_for_url(self.cache_dir, url, suffix=suffix)

    def get_bytes(
        self,
        url: str,
        *,
        suffix: str = ".html",
        accept: str | None = None,
        refresh: bool | None = None,
        not_found_body: bytes | None = None,
    ) -> tuple[bytes, Path, bool]:
        result = self.get_result(
            url,
            suffix=suffix,
            accept=accept,
            refresh=refresh,
            not_found_body=not_found_body,
        )
        return result.body, result.cache_path, not result.from_cache

    def get_text(
        self,
        url: str,
        *,
        suffix: str = ".html",
        accept: str | None = None,
        refresh: bool | None = None,
        not_found_text: str | None = None,
    ) -> tuple[str, Path, bool]:
        result = self.get_result(
            url,
            suffix=suffix,
            accept=accept,
            refresh=refresh,
            not_found_body=not_found_text.encode("utf-8") if not_found_text is not None else None,
        )
        return result.body.decode("utf-8", errors="replace"), result.cache_path, not result.from_cache

    def get_json(
        self,
        url: str,
        *,
        suffix: str = ".json",
        accept: str = "application/json",
        refresh: bool | None = None,
        not_found_json: Any | None = None,
    ) -> tuple[Any, Path, bool]:
        text, cache_path, fetched = self.get_text(
            url,
            suffix=suffix,
            accept=accept,
            refresh=refresh,
            not_found_text=json.dumps(not_found_json) if not_found_json is not None else None,
        )
        return json.loads(text), cache_path, fetched

    def get_result(
        self,
        url: str,
        *,
        suffix: str = ".html",
        accept: str | None = None,
        refresh: bool | None = None,
        not_found_body: bytes | None = None,
    ) -> FetchResult:
        cache_path = self.cache_path(url, suffix)
        should_refresh = self.refresh if refresh is None else refresh
        now = dt.datetime.now(dt.UTC)
        if cache_path.exists() and not should_refresh:
            return FetchResult(
                body=cache_path.read_bytes(),
                source_url=url,
                fetched_at=now,
                cache_path=cache_path,
                from_cache=True,
                status_code=200,
            )
        if self.offline:
            raise FileNotFoundError(f"{self.offline_error_prefix}: {url}")

        last_error: BaseException | None = None
        attempts = max(1, self.retries)
        for attempt in range(attempts):
            self._wait()
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": accept or self.default_accept,
                    "User-Agent": self.user_agent,
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    body = response.read()
                    status_code = int(response.status)
                cache_path.write_bytes(body)
                self._last_request_at = time.monotonic()
                return FetchResult(
                    body=body,
                    source_url=url,
                    fetched_at=dt.datetime.now(dt.UTC),
                    cache_path=cache_path,
                    from_cache=False,
                    status_code=status_code,
                )
            except urllib.error.HTTPError as exc:
                last_error = exc
                self._last_request_at = time.monotonic()
                if exc.code == 404 and not_found_body is not None:
                    cache_path.write_bytes(not_found_body)
                    return FetchResult(
                        body=not_found_body,
                        source_url=url,
                        fetched_at=dt.datetime.now(dt.UTC),
                        cache_path=cache_path,
                        from_cache=False,
                        status_code=404,
                    )
                if exc.code not in self.transient_statuses or attempt == attempts - 1:
                    raise
                time.sleep(self._retry_delay(exc, attempt))
            except (TimeoutError, urllib.error.URLError) as exc:
                last_error = exc
                if attempt == attempts - 1:
                    break
                time.sleep(self.delay_seconds * (attempt + 1))
        raise RuntimeError(f"GET {url} failed after retry: {last_error}")

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        delay = max(0.0, self.delay_seconds - elapsed)
        if delay:
            time.sleep(delay)

    def _retry_delay(self, exc: urllib.error.HTTPError, attempt: int) -> float:
        return retry_delay_seconds(exc, fallback_seconds=self.delay_seconds * (attempt + 1))
