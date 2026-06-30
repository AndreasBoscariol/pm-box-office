"""HTTP client helpers for AMC ingestion."""

from __future__ import annotations

import hashlib
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_CACHE_DIR = Path("data/raw/amc")
DEFAULT_USER_AGENT = "pm-box-office-amc/0.1"
TRANSIENT_STATUSES = {429, 500, 502, 503, 504}


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

    def get(self, url: str) -> tuple[str, Path, bool]:
        cache_path = self.cache_path(url)
        if cache_path.exists() and not self.refresh:
            return cache_path.read_text(encoding="utf-8"), cache_path, False
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
                if not body.strip():
                    raise RuntimeError(f"GET {url} returned an empty response body")
                cache_path.write_text(body, encoding="utf-8")
                self._last_request_at = time.monotonic()
                return body, cache_path, True
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in TRANSIENT_STATUSES or attempt == self.retries - 1:
                    raise
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else self.delay_seconds * (attempt + 1)
                time.sleep(delay)
            except (TimeoutError, urllib.error.URLError) as exc:
                last_error = exc
                if attempt == self.retries - 1:
                    break
                time.sleep(self.delay_seconds * (attempt + 1))
        raise RuntimeError(f"GET {url} failed after retry: {last_error}")

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        delay = max(0.0, self.delay_seconds - elapsed)
        if delay:
            time.sleep(delay)

