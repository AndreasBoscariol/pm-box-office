#!/usr/bin/env python3
"""Ingest Boxoffice Pro Weekend Preview forecasts into PostgreSQL.

This source intentionally parses only high-confidence Weekend Preview
"Boxoffice Podium" blocks. Older generic Boxoffice Pro prediction tables are
dropped during initialization because they admitted too much prose garbage.
"""

from __future__ import annotations

import argparse
import email.utils
import datetime as dt
import hashlib
from html.parser import HTMLParser
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from pm_box_office.db.connection import connect_database, database_url_from_env


BASE_URL = "https://www.boxofficepro.com"
FORECAST_ARCHIVE_URL = f"{BASE_URL}/category/forecasts-tracking/"
FORECAST_RSS_URL = f"{FORECAST_ARCHIVE_URL}feed/"
DEFAULT_START_DATE = dt.date(2026, 6, 1)
DEFAULT_END_DATE = dt.date(2026, 6, 30)
DEFAULT_CACHE_DIR = Path("data/raw/boxofficepro")
DEFAULT_USER_AGENT = "pm-box-office-boxofficepro-bot/1.0 (+personal research; set --user-agent contact)"
DEFAULT_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)
TRANSIENT_STATUSES = {429, 500, 502, 503, 504}
MIN_DELAY_SECONDS = 20.0
DOMESTIC_MARKET = "US_CA"
DOMESTIC_CURRENCY = "USD"
PARSER_VERSION = "weekend_podium_v1"
LEGACY_TABLE_PARSER_VERSION = "legacy_table_v1"
LEGACY_HEADING_PARSER_VERSION = "legacy_heading_v1"


@dataclass(frozen=True)
class ArchiveArticle:
    article_url: str
    title: str
    author: str | None
    published_date: str | None
    article_type: str
    source_url: str
    excerpt: str | None = None


@dataclass(frozen=True)
class WeekendPrediction:
    article_url: str
    source_movie_title: str
    normalized_movie_title: str
    source_movie_id: str
    distributor: str
    release_status: str
    source_rank: int | None
    market: str
    currency: str
    forecast_metric: str
    range_low_usd: int
    range_high_usd: int
    showtime_market_share_pct: float | None
    target_start_date: str | None
    target_end_date: str | None
    raw_forecast_text: str
    source_context: str
    parser_version: str
    row_ordinal: int
    source_row_key: str
    matched_movie_id: int | None = None
    match_status: str = "unmatched"
    match_method: str | None = None
    match_score: float | None = None
    match_notes: str | None = None


ForecastPrediction = WeekendPrediction


@dataclass(frozen=True)
class MovieCandidate:
    movie_id: int
    movie_url: str | None
    title: str
    release_year: int | None
    release_date: str | None
    normalized_title: str


@dataclass(frozen=True)
class MovieMatch:
    movie_id: int | None
    status: str
    method: str | None
    score: float | None
    notes: str | None


@dataclass(frozen=True)
class HeadingBlock:
    level: int
    text: str


@dataclass(frozen=True)
class HtmlTable:
    headers: list[str]
    rows: list[list[str]]


@dataclass(frozen=True)
class RejectedBlock:
    raw_text: str
    reason: str


class FetchBlocked(RuntimeError):
    """Raised when a Boxoffice Pro page cannot be fetched and no cache is available."""


class HtmlFetcher:
    def __init__(
        self,
        cache_dir: Path,
        *,
        refresh: bool,
        offline: bool,
        delay_seconds: float,
        user_agent: str,
        fetch_mode: str = "auto",
        browser_user_agent: str = DEFAULT_BROWSER_USER_AGENT,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.cache_dir = cache_dir
        self.refresh = refresh
        self.offline = offline
        self.delay_seconds = delay_seconds
        self.user_agent = user_agent
        self.fetch_mode = fetch_mode
        self.browser_user_agent = browser_user_agent
        self.timeout_seconds = timeout_seconds
        self._last_request_at = 0.0
        self._playwright: Any | None = None
        self._browser: Any | None = None
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(self, url: str, suffix: str | None = None) -> Path:
        if suffix is None:
            suffix = ".xml" if is_rss_url(url) else ".html"
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}{suffix}"

    def get(self, url: str) -> tuple[str, Path, bool]:
        cache_path = self.cache_path(url)
        if cache_path.exists() and (not self.refresh or self.offline):
            return cache_path.read_text(encoding="utf-8"), cache_path, False
        if self.offline:
            raise FetchBlocked(
                f"Cache miss in offline mode: {url}\n"
                f"Expected cached HTML at: {cache_path}\n"
                "Open the URL in a browser, save the page source as that file, then rerun with --offline."
            )

        last_error: Exception | None = None
        if self.fetch_mode in {"http", "auto"}:
            try:
                body = self._get_http(url)
                cache_path.write_text(body, encoding="utf-8")
                return body, cache_path, True
            except FetchBlocked as exc:
                last_error = exc
                if self.fetch_mode == "http":
                    raise

        if self.fetch_mode in {"browser", "auto"}:
            try:
                body = self._get_browser(url)
                cache_path.write_text(body, encoding="utf-8")
                return body, cache_path, True
            except FetchBlocked as exc:
                if last_error is not None:
                    raise FetchBlocked(f"{last_error}\nBrowser fallback failed: {exc}") from exc
                raise

        raise RuntimeError(f"GET {url} failed: unsupported fetch mode {self.fetch_mode!r}")

    def _get_http(self, url: str) -> str:
        last_error: Exception | None = None
        accept = "application/rss+xml,application/xml,text/xml,text/html,application/xhtml+xml"
        for attempt in range(2):
            self._wait()
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": accept,
                    "User-Agent": self.user_agent,
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    body = response.read().decode("utf-8", errors="replace")
                self._last_request_at = time.monotonic()
                return body
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code == 403:
                    raise FetchBlocked(
                        "Boxoffice Pro returned HTTP 403. The site may block automated HTTP clients.\n"
                        f"URL: {url}"
                    ) from exc
                if exc.code not in TRANSIENT_STATUSES or attempt == 1:
                    raise
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else self.delay_seconds
                time.sleep(delay)
            except (TimeoutError, urllib.error.URLError) as exc:
                last_error = exc
                if attempt == 1:
                    break
                time.sleep(self.delay_seconds)
        raise RuntimeError(f"GET {url} failed after retry: {last_error}")

    def _get_browser(self, url: str) -> str:
        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise FetchBlocked(
                "Playwright is required for --fetch-mode browser/auto when HTTP is blocked. "
                "Install project dependencies, then run `playwright install chromium` if needed."
            ) from exc
        try:
            self._wait()
            if self._playwright is None or self._browser is None:
                self._playwright = sync_playwright().start()
                self._browser = self._playwright.chromium.launch(headless=True)
            page = self._browser.new_page(user_agent=self.browser_user_agent)
            try:
                response = page.goto(url, wait_until="domcontentloaded", timeout=int(self.timeout_seconds * 1000))
                status = response.status if response is not None else None
                if status == 403:
                    raise FetchBlocked(f"Browser fetch was blocked with HTTP 403: {url}")
                if status is not None and status >= 400:
                    raise FetchBlocked(f"Browser fetch returned HTTP {status}: {url}")
                if is_rss_url(url):
                    body = page.locator("body").inner_text(timeout=int(self.timeout_seconds * 1000)).strip()
                    if body.startswith("<?xml") or body.startswith("<rss"):
                        return body
                return page.content()
            finally:
                page.close()
                self._last_request_at = time.monotonic()
        except FetchBlocked:
            raise
        except PlaywrightError as exc:
            raise FetchBlocked(f"Browser fetch failed for {url}: {exc}") from exc

    def close(self) -> None:
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        delay = max(0.0, self.delay_seconds - elapsed)
        if delay:
            time.sleep(delay)


class SingleItemArchiveParser(HTMLParser):
    """Extractor for Boxoffice Pro archive cards: article.single-item."""

    def __init__(self, source_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.source_url = source_url
        self.articles: list[ArchiveArticle] = []
        self._in_card = False
        self._card_depth = 0
        self._article_url: str | None = None
        self._title: str | None = None
        self._excerpt: str | None = None
        self._capture: str | None = None
        self._parts: list[str] = []
        self._time: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        classes = set((attrs_dict.get("class") or "").split())
        if tag == "article" and "single-item" in classes:
            self._in_card = True
            self._card_depth = 1
            self._article_url = None
            self._title = None
            self._excerpt = None
            self._time = None
            self._capture = None
            self._parts = []
            return
        if not self._in_card:
            return
        if tag == "article":
            self._card_depth += 1
        if tag == "a" and "single-item__link" in classes:
            href = attrs_dict.get("href")
            if href:
                self._article_url = absolute_url(href)
        elif tag in {"h1", "h2", "h3"} and "single-item__heading" in classes:
            self._capture = "title"
            self._parts = []
        elif tag == "div" and "single-item__excerpt" in classes:
            self._capture = "excerpt"
            self._parts = []
        elif tag == "time":
            timestamp = attrs_dict.get("datetime")
            if timestamp:
                self._time = timestamp

    def handle_endtag(self, tag: str) -> None:
        if not self._in_card:
            return
        if self._capture == "title" and tag in {"h1", "h2", "h3"}:
            self._title = clean_text(" ".join(self._parts))
            self._capture = None
            self._parts = []
        elif self._capture == "excerpt" and tag == "div":
            self._excerpt = clean_text(" ".join(self._parts))
            self._capture = None
            self._parts = []
        elif tag == "article":
            self._card_depth -= 1
            if self._card_depth == 0:
                self._finish_card()

    def handle_data(self, data: str) -> None:
        if self._in_card and self._capture:
            self._parts.append(data)

    def _finish_card(self) -> None:
        if self._article_url and self._title:
            published_date = parse_dateish(self._time) if self._time else forecast_date_from_text(self._excerpt or self._title)
            self.articles.append(
                ArchiveArticle(
                    article_url=self._article_url,
                    title=self._title,
                    author=None,
                    published_date=published_date,
                    article_type=classify_article_type(self._title),
                    source_url=self.source_url,
                    excerpt=self._excerpt,
                )
            )
        self._in_card = False
        self._card_depth = 0


class FallbackArchiveParser(HTMLParser):
    """Minimal fallback for small fixtures that do not use article.single-item."""

    def __init__(self, source_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.source_url = source_url
        self.articles: list[ArchiveArticle] = []
        self._capture_heading = False
        self._heading_parts: list[str] = []
        self._link_href: str | None = None
        self._time: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "a" and attrs_dict.get("href"):
            self._link_href = absolute_url(str(attrs_dict["href"]))
        elif tag in {"h1", "h2", "h3"}:
            self._capture_heading = True
            self._heading_parts = []
        elif tag == "time" and attrs_dict.get("datetime"):
            self._time = str(attrs_dict["datetime"])

    def handle_endtag(self, tag: str) -> None:
        if self._capture_heading and tag in {"h1", "h2", "h3"}:
            title = clean_text(" ".join(self._heading_parts))
            if title and self._link_href and "/category/" not in self._link_href:
                self.articles.append(
                    ArchiveArticle(
                        article_url=self._link_href,
                        title=title,
                        author=None,
                        published_date=parse_dateish(self._time) if self._time else forecast_date_from_text(title),
                        article_type=classify_article_type(title),
                        source_url=self.source_url,
                    )
                )
            self._capture_heading = False
            self._heading_parts = []
            self._link_href = None
            self._time = None

    def handle_data(self, data: str) -> None:
        if self._capture_heading:
            self._heading_parts.append(data)


class HtmlTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


class WeekendPreviewParser(HTMLParser):
    """Collect article headings and tables while preserving <br> boundaries."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: str | None = None
        self.author: str | None = None
        self.published_date: str | None = None
        self.blocks: list[HeadingBlock] = []
        self.tables: list[HtmlTable] = []
        self._capture_heading: tuple[str, int] | None = None
        self._heading_parts: list[str] = []
        self._in_table = False
        self._capture_cell = False
        self._table_depth = 0
        self._cell_parts: list[str] = []
        self._current_row: list[str] = []
        self._current_table: list[list[str]] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag in {"script", "style", "nav", "footer"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "time" and attrs_dict.get("datetime") and not self.published_date:
            self.published_date = parse_dateish(str(attrs_dict["datetime"]))
        elif tag == "meta":
            name = attrs_dict.get("property") or attrs_dict.get("name")
            content = attrs_dict.get("content")
            if name in {"og:title", "twitter:title"} and content and not self.title:
                self.title = clean_text(content)
            elif name == "article:published_time" and content and not self.published_date:
                self.published_date = parse_dateish(content)
            elif name in {"author", "article:author"} and content and not self.author:
                self.author = clean_text(content)
        elif tag in {"h1", "h2", "h3", "h4"}:
            self._capture_heading = (tag, int(tag[1]))
            self._heading_parts = []
        elif tag == "br" and self._capture_heading:
            self._heading_parts.append("\n")
        elif tag == "table":
            self._in_table = True
            self._table_depth += 1
            if self._table_depth == 1:
                self._current_table = []
        elif self._in_table and tag == "tr":
            self._current_row = []
        elif self._in_table and tag in {"td", "th"}:
            self._capture_cell = True
            self._cell_parts = []
        elif tag == "br" and self._capture_cell:
            self._cell_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav", "footer"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if self._capture_heading and self._capture_heading[0] == tag:
            text = clean_multiline_text("".join(self._heading_parts))
            if text:
                if tag == "h1" and not self.title:
                    self.title = text
                self.blocks.append(HeadingBlock(level=self._capture_heading[1], text=text))
            self._capture_heading = None
            self._heading_parts = []
        elif self._in_table and tag in {"td", "th"} and self._capture_cell:
            self._current_row.append(clean_multiline_text("".join(self._cell_parts)))
            self._capture_cell = False
            self._cell_parts = []
        elif self._in_table and tag == "tr":
            if any(cell for cell in self._current_row):
                self._current_table.append(self._current_row)
            self._current_row = []
        elif tag == "table" and self._in_table:
            self._table_depth -= 1
            if self._table_depth == 0:
                table = table_from_rows(self._current_table)
                if table is not None:
                    self.tables.append(table)
                self._in_table = False
                self._current_table = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._capture_heading:
            self._heading_parts.append(data)
        if self._capture_cell:
            self._cell_parts.append(data)


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()


def clean_multiline_text(value: str) -> str:
    text = value.replace("\xa0", " ").replace("\r", "\n")
    lines = [clean_text(line) for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def absolute_url(href: str) -> str:
    return urllib.parse.urljoin(BASE_URL, href)


def canonical_article_url(href: str) -> str:
    url = absolute_url(href)
    parsed = urllib.parse.urlsplit(url)
    query = [
        (key, value)
        for key, value in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
    ]
    return urllib.parse.urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urllib.parse.urlencode(query),
            "",
        )
    )


def is_rss_url(url: str) -> bool:
    path = urllib.parse.urlparse(url).path.rstrip("/")
    return path.endswith("/feed") or path.endswith("/rss")


def archive_url(page: int) -> str:
    if page <= 1:
        return FORECAST_ARCHIVE_URL
    return f"{FORECAST_ARCHIVE_URL}page/{page}/"


def parse_date_arg(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO date: {value}") from exc


def parse_dateish(value: str) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    if re.fullmatch(r"\d{10}", text):
        return dt.datetime.fromtimestamp(int(text), tz=dt.UTC).date().isoformat()
    try:
        return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        pass
    text = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", text, flags=re.IGNORECASE)
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return dt.datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


MONTH_PATTERN = (
    r"(January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)


def extract_date_range(text: str, fallback_date: str | None = None) -> tuple[str | None, str | None]:
    year = int(fallback_date[:4]) if fallback_date else None
    explicit_year = re.search(r"\b(20\d{2})\b", text)
    if explicit_year:
        year = int(explicit_year.group(1))
    if year is None:
        return None, None
    cleaned = clean_text(text).replace("\u2013", "-").replace("\u2014", "-")
    range_match = re.search(
        rf"{MONTH_PATTERN}\s+(\d{{1,2}})\s*(?:-|to)\s*(?:(?:{MONTH_PATTERN})\s+)?(\d{{1,2}})",
        cleaned,
        flags=re.IGNORECASE,
    )
    if range_match:
        month1 = range_match.group(1)
        day1 = int(range_match.group(2))
        month2 = range_match.group(3) or month1
        day2 = int(range_match.group(4))
        start = parse_month_day(month1, day1, year)
        end = parse_month_day(month2, day2, year)
        if start and end:
            if end < start:
                end = dt.date(year + 1, end.month, end.day)
            return start.isoformat(), end.isoformat()
    return None, None


def parse_month_day(month: str, day: int, year: int) -> dt.date | None:
    for fmt in ("%B", "%b"):
        try:
            month_number = dt.datetime.strptime(month[:3] if fmt == "%b" else month, fmt).month
            return dt.date(year, month_number, day)
        except ValueError:
            continue
    return None


def forecast_date_from_text(value: str) -> str | None:
    start_date, _end_date = extract_date_range(value)
    if start_date:
        return start_date
    match = re.search(rf"{MONTH_PATTERN}\s+\d{{1,2}},?\s+20\d{{2}}", value, flags=re.IGNORECASE)
    return parse_dateish(match.group(0)) if match else None


def classify_article_type(title: str) -> str:
    normalized = title.lower()
    if (
        "weekend preview" in normalized
        or "weekend forecast" in normalized
        or "weekend box office forecast" in normalized
    ):
        return "weekend_preview"
    if "long range" in normalized:
        return "long_range_forecast"
    if "forecast" in normalized or "tracking" in normalized:
        return "forecast_tracking"
    return "other"


def is_domestic_article(article: ArchiveArticle) -> bool:
    text = article.title.lower()
    if re.search(r"\bu\.?k\.?\b", text) or "ireland" in text:
        return False
    return True


def is_weekend_preview_article(article: ArchiveArticle) -> bool:
    return is_domestic_article(article) and article.article_type == "weekend_preview"


def normalize_movie_title(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"\s*\(\d{4}\)\s*$", "", text)
    text = re.sub(
        r"\b(disney|marvel|pixar|dreamworks|illumination|warner bros|dc|universal|paramount|sony|a24)'?s\s+",
        "",
        text,
    )
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def boxofficepro_source_movie_id(
    *,
    market: str,
    normalized_movie_title: str,
    target_start_date: str | None,
) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", normalized_movie_title).strip("-") or "unknown-title"
    release_key = target_start_date or "unknown-date"
    return f"boxofficepro:{market}:{slug}:{release_key}"


RSS_NAMESPACES = {
    "dc": "http://purl.org/dc/elements/1.1/",
    "content": "http://purl.org/rss/1.0/modules/content/",
}


def strip_html(value: str) -> str:
    parser = HtmlTextParser()
    parser.feed(value)
    return clean_text(" ".join(parser.parts))


def parse_rss(xml_text: str, *, source_url: str) -> list[ArchiveArticle]:
    try:
        root = ET.fromstring(xml_text.strip())
    except ET.ParseError as exc:
        raise ValueError(f"invalid Boxoffice Pro RSS XML from {source_url}: {exc}") from exc
    articles: list[ArchiveArticle] = []
    for item in root.findall("./channel/item"):
        title = clean_text(item.findtext("title") or "")
        link = clean_text(item.findtext("link") or "")
        if not title or not link:
            continue
        pub_date = parse_rss_date(item.findtext("pubDate") or "")
        author = item.findtext("dc:creator", namespaces=RSS_NAMESPACES)
        description = item.findtext("description") or ""
        excerpt = strip_html(description) if description else None
        articles.append(
            ArchiveArticle(
                article_url=canonical_article_url(link),
                title=title,
                author=clean_text(author) if author else None,
                published_date=pub_date or forecast_date_from_text(excerpt or title),
                article_type=classify_article_type(title),
                source_url=source_url,
                excerpt=excerpt,
            )
        )
    deduped: dict[str, ArchiveArticle] = {}
    for article in articles:
        if article.article_type == "other":
            continue
        deduped.setdefault(article.article_url, article)
    return list(deduped.values())


def parse_rss_date(value: str) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        return email.utils.parsedate_to_datetime(text).date().isoformat()
    except (TypeError, ValueError):
        return parse_dateish(text)


def parse_archive(html: str, *, source_url: str) -> list[ArchiveArticle]:
    parser = SingleItemArchiveParser(source_url)
    parser.feed(html)
    articles = parser.articles
    if not articles:
        fallback = FallbackArchiveParser(source_url)
        fallback.feed(html)
        articles = fallback.articles
    deduped: dict[str, ArchiveArticle] = {}
    for article in articles:
        if classify_article_type(article.title) == "other":
            continue
        if article.published_date is None:
            article = replace(article, published_date=forecast_date_from_text(article.excerpt or article.title))
        deduped.setdefault(article.article_url, article)
    return list(deduped.values())


def dedupe_articles_by_content(articles: list[ArchiveArticle]) -> list[ArchiveArticle]:
    deduped: dict[tuple[str | None, str], ArchiveArticle] = {}
    for article in articles:
        key = (article.published_date, normalize_article_title_key(article.title))
        existing = deduped.get(key)
        if existing is None or prefer_archive_duplicate(article, existing):
            deduped[key] = article
    return list(deduped.values())


def normalize_article_title_key(title: str) -> str:
    text = unicodedata.normalize("NFKD", title)
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def prefer_archive_duplicate(candidate: ArchiveArticle, existing: ArchiveArticle) -> bool:
    candidate_path = urllib.parse.urlparse(candidate.article_url).path.rstrip("/")
    existing_path = urllib.parse.urlparse(existing.article_url).path.rstrip("/")
    candidate_penalty = 1 if re.search(r"-2$", candidate_path) else 0
    existing_penalty = 1 if re.search(r"-2$", existing_path) else 0
    return (candidate_penalty, len(candidate.article_url)) < (existing_penalty, len(existing.article_url))


def parse_article(
    html: str,
    *,
    article_url: str,
    fallback: ArchiveArticle | None = None,
) -> tuple[ArchiveArticle, list[WeekendPrediction], list[RejectedBlock]]:
    parser = WeekendPreviewParser()
    parser.feed(html)
    title = parser.title or (fallback.title if fallback else "")
    published_date = parser.published_date or (fallback.published_date if fallback else None)
    article = ArchiveArticle(
        article_url=article_url,
        title=title,
        author=parser.author or (fallback.author if fallback else None),
        published_date=published_date,
        article_type=classify_article_type(title),
        source_url=fallback.source_url if fallback else article_url,
        excerpt=fallback.excerpt if fallback else None,
    )
    predictions, rejected = parse_weekend_podium_blocks(
        parser.blocks,
        article_url=article_url,
        article_title=title,
        published_date=published_date,
    )
    if not predictions:
        predictions, legacy_rejected = parse_legacy_forecast_tables(
            parser.tables,
            article_url=article_url,
            article_title=title,
            published_date=published_date,
        )
        rejected.extend(legacy_rejected)
    if not predictions:
        predictions, heading_rejected = parse_standalone_forecast_heading_blocks(
            parser.blocks,
            article_url=article_url,
            article_title=title,
            published_date=published_date,
        )
        rejected.extend(heading_rejected)
    return article, predictions, rejected


def table_from_rows(rows: list[list[str]]) -> HtmlTable | None:
    normalized_rows = [[clean_text(cell) for cell in row] for row in rows if any(clean_text(cell) for cell in row)]
    if len(normalized_rows) < 2:
        return None
    headers = normalized_rows[0]
    body_rows = [row for row in normalized_rows[1:] if len(row) >= 2]
    if not body_rows:
        return None
    return HtmlTable(headers=headers, rows=body_rows)


def parse_legacy_forecast_tables(
    tables: list[HtmlTable],
    *,
    article_url: str,
    article_title: str,
    published_date: str | None,
) -> tuple[list[WeekendPrediction], list[RejectedBlock]]:
    predictions: list[WeekendPrediction] = []
    rejected: list[RejectedBlock] = []
    target_start, target_end = infer_legacy_target_dates(tables, article_title=article_title, published_date=published_date)
    for table in tables:
        parsed_table = parse_legacy_forecast_table(
            table,
            article_url=article_url,
            target_start_date=target_start,
            target_end_date=target_end,
            first_row_ordinal=len(predictions) + 1,
        )
        if parsed_table:
            predictions.extend(parsed_table)
        elif looks_like_forecast_table(table):
            rejected.append(
                RejectedBlock(
                    raw_text=legacy_table_text(table),
                    reason="legacy_forecast_table_missing_required_fields",
                )
            )
    return predictions, rejected


def parse_legacy_forecast_table(
    table: HtmlTable,
    *,
    article_url: str,
    target_start_date: str | None,
    target_end_date: str | None,
    first_row_ordinal: int,
) -> list[WeekendPrediction]:
    indexes = legacy_table_indexes(table.headers)
    if indexes is None:
        return []
    predictions: list[WeekendPrediction] = []
    for row in table.rows:
        required_indexes = [index for index in indexes.values() if index is not None]
        if len(row) <= max(required_indexes):
            continue
        title = clean_legacy_movie_title(row[indexes["title"]])
        distributor = clean_text(row[indexes["distributor"]])
        forecast_cell = clean_text(row[indexes["forecast"]])
        if not title or not distributor:
            continue
        money_range = parse_money_range(forecast_cell)
        if money_range is None:
            exact_value = parse_money_value(forecast_cell)
            if exact_value is None:
                continue
            money_range = (exact_value, exact_value)
        release_date = None
        if indexes["release_date"] is not None:
            release_date = parse_table_release_date(
                row[indexes["release_date"]],
                target_start_date=target_start_date,
                published_date=None,
            )
        status_cell = clean_text(row[indexes["status"]]) if indexes["status"] is not None else None
        row_ordinal = first_row_ordinal + len(predictions)
        release_status = infer_release_status(release_date, target_start_date, status_cell=status_cell)
        forecast_metric = "domestic_opening_weekend" if release_status == "NEW" else "domestic_weekend"
        normalized = normalize_movie_title(title)
        raw_text = legacy_row_text(table.headers, row)
        key_material = "|".join(
            [
                article_url,
                str(row_ordinal),
                normalized,
                forecast_metric,
                str(money_range[0]),
                str(money_range[1]),
                str(target_start_date),
                str(target_end_date),
                LEGACY_TABLE_PARSER_VERSION,
            ]
        )
        predictions.append(
            WeekendPrediction(
                article_url=article_url,
                source_movie_title=title,
                normalized_movie_title=normalized,
                source_movie_id=boxofficepro_source_movie_id(
                    market=DOMESTIC_MARKET,
                    normalized_movie_title=normalized,
                    target_start_date=target_start_date,
                ),
                distributor=distributor,
                release_status=release_status,
                source_rank=row_ordinal,
                market=DOMESTIC_MARKET,
                currency=DOMESTIC_CURRENCY,
                forecast_metric=forecast_metric,
                range_low_usd=money_range[0],
                range_high_usd=money_range[1],
                showtime_market_share_pct=None,
                target_start_date=target_start_date,
                target_end_date=target_end_date,
                raw_forecast_text=raw_text,
                source_context="legacy_forecast_table",
                parser_version=LEGACY_TABLE_PARSER_VERSION,
                row_ordinal=row_ordinal,
                source_row_key=hashlib.sha256(key_material.encode("utf-8")).hexdigest(),
            )
        )
    return predictions


def parse_standalone_forecast_heading_blocks(
    blocks: list[HeadingBlock],
    *,
    article_url: str,
    article_title: str,
    published_date: str | None,
) -> tuple[list[WeekendPrediction], list[RejectedBlock]]:
    predictions: list[WeekendPrediction] = []
    rejected: list[RejectedBlock] = []
    index = 0
    while index < len(blocks) - 3:
        title_block = blocks[index]
        distributor_block = blocks[index + 1]
        date_block = blocks[index + 2]
        range_block = blocks[index + 3]
        if title_block.level != 2 or distributor_block.level > 3 or date_block.level > 3 or range_block.level > 3:
            index += 1
            continue
        title = clean_legacy_movie_title(title_block.text)
        distributor = clean_text(distributor_block.text)
        release_date = forecast_date_from_text(date_block.text) or parse_dateish(re.sub(r"\s*\(.+\)\s*$", "", date_block.text))
        range_text = clean_text(range_block.text)
        money_range = parse_money_range(range_text)
        if not title or not distributor or release_date is None or money_range is None:
            index += 1
            continue
        target_start, target_end = target_dates_from_release_and_range(release_date, range_text, article_title)
        row_ordinal = len(predictions) + 1
        normalized = normalize_movie_title(title)
        forecast_metric = "domestic_opening_weekend" if re.search(r"opening", range_text, re.IGNORECASE) else "domestic_weekend"
        key_material = "|".join(
            [
                article_url,
                str(row_ordinal),
                normalized,
                forecast_metric,
                str(money_range[0]),
                str(money_range[1]),
                str(target_start),
                str(target_end),
                LEGACY_HEADING_PARSER_VERSION,
            ]
        )
        predictions.append(
            WeekendPrediction(
                article_url=article_url,
                source_movie_title=title,
                normalized_movie_title=normalized,
                source_movie_id=boxofficepro_source_movie_id(
                    market=DOMESTIC_MARKET,
                    normalized_movie_title=normalized,
                    target_start_date=target_start,
                ),
                distributor=distributor,
                release_status="NEW" if forecast_metric == "domestic_opening_weekend" else "UNKNOWN",
                source_rank=row_ordinal,
                market=DOMESTIC_MARKET,
                currency=DOMESTIC_CURRENCY,
                forecast_metric=forecast_metric,
                range_low_usd=money_range[0],
                range_high_usd=money_range[1],
                showtime_market_share_pct=None,
                target_start_date=target_start,
                target_end_date=target_end,
                raw_forecast_text="\n".join([title_block.text, distributor_block.text, date_block.text, range_block.text]),
                source_context="legacy_forecast_heading",
                parser_version=LEGACY_HEADING_PARSER_VERSION,
                row_ordinal=row_ordinal,
                source_row_key=hashlib.sha256(key_material.encode("utf-8")).hexdigest(),
            )
        )
        index += 4
    return predictions, rejected


def target_dates_from_release_and_range(
    release_date: str,
    range_text: str,
    article_title: str,
) -> tuple[str, str]:
    start = dt.date.fromisoformat(release_date)
    if re.search(r"\b(4|5)-day\b", f"{article_title} {range_text}", re.IGNORECASE):
        end = start + dt.timedelta(days=3)
    else:
        end = start + dt.timedelta(days=2)
    return start.isoformat(), end.isoformat()


def legacy_table_indexes(headers: list[str]) -> dict[str, int] | None:
    normalized = [normalize_header(header) for header in headers]
    title_index = find_header_index(normalized, {"title", "film", "movie"})
    release_index = find_header_containing(normalized, ["release", "date"])
    distributor_index = find_header_index(normalized, {"distributor", "studio", "studios"})
    forecast_index = find_weekend_forecast_index(normalized)
    status_index = find_status_header_index(normalized)
    if None in {title_index, distributor_index, forecast_index}:
        return None
    assert title_index is not None
    assert distributor_index is not None
    assert forecast_index is not None
    return {
        "title": title_index,
        "release_date": release_index,
        "distributor": distributor_index,
        "forecast": forecast_index,
        "status": status_index,
    }


def normalize_header(value: str) -> str:
    text = clean_text(value).lower().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def find_header_index(headers: list[str], choices: set[str]) -> int | None:
    for index, header in enumerate(headers):
        if header in choices:
            return index
    return None


def find_header_containing(headers: list[str], required_words: list[str]) -> int | None:
    for index, header in enumerate(headers):
        if all(word in header.split() for word in required_words):
            return index
    return None


def find_weekend_forecast_index(headers: list[str]) -> int | None:
    for index, header in enumerate(headers):
        if header == "weekend":
            return index
    for index, header in enumerate(headers):
        if "weekend" in header and "forecast" in header and "total" not in header:
            return index
    for index, header in enumerate(headers):
        if "predicted opening range" in header or "opening weekend range" in header:
            return index
    for index, header in enumerate(headers):
        if (
            ("forecast" in header or "weekend" in header)
            and ("day" in header or "weekend" in header)
            and "total" not in header
            and "location" not in header
            and "change" not in header
        ):
            return index
    return None


def find_status_header_index(headers: list[str]) -> int | None:
    for index, header in enumerate(headers):
        if "change" in header and ("last" in header or "wknd" in header or "weekend" in header):
            return index
    return None


def looks_like_forecast_table(table: HtmlTable) -> bool:
    headers = [normalize_header(header) for header in table.headers]
    return any("weekend" in header or "opening range" in header for header in headers)


def legacy_table_text(table: HtmlTable) -> str:
    return "\n".join([legacy_row_text(table.headers, row) for row in table.rows[:3]])


def legacy_row_text(headers: list[str], row: list[str]) -> str:
    parts: list[str] = []
    for index, cell in enumerate(row):
        header = headers[index] if index < len(headers) else f"column_{index + 1}"
        parts.append(f"{header}: {cell}")
    return " | ".join(parts)


def clean_legacy_movie_title(value: str) -> str:
    text = strip_wildcard_title_prefix(value)
    return clean_text(re.sub(r"\s+\((?:expansion|limited|wide)\)\s*$", "", text, flags=re.IGNORECASE))


def strip_wildcard_title_prefix(value: str) -> str:
    return clean_text(re.sub(r"^\s*wild\s*card\s*:\s*", "", value, flags=re.IGNORECASE))


def infer_legacy_target_dates(
    tables: list[HtmlTable],
    *,
    article_title: str,
    published_date: str | None,
) -> tuple[str | None, str | None]:
    target_start, target_end = extract_date_range(article_title, published_date)
    if target_start and target_end:
        return target_start, target_end
    if published_date:
        published = dt.date.fromisoformat(published_date)
        days_until_friday = (4 - published.weekday()) % 7
        if days_until_friday == 0:
            days_until_friday = 7
        start = published + dt.timedelta(days=days_until_friday)
        end = start + dt.timedelta(days=2)
        monday_end = monday_end_date_from_tables(tables, start.year)
        if monday_end and monday_end >= start:
            end = monday_end
        elif any("four day" in legacy_table_text(table).lower() or "4 day" in legacy_table_text(table).lower() for table in tables):
            end = start + dt.timedelta(days=3)
        return start.isoformat(), end.isoformat()
    return None, None


def monday_end_date_from_tables(tables: list[HtmlTable], year: int) -> dt.date | None:
    text = " ".join(" ".join(table.headers) for table in tables)
    match = re.search(r"through\s+monday,?\s+([A-Za-z]+)\s+(\d{1,2})", text, flags=re.IGNORECASE)
    if not match:
        return None
    return parse_month_day(match.group(1), int(match.group(2)), year)


def parse_table_release_date(
    value: str,
    *,
    target_start_date: str | None,
    published_date: str | None,
) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    parsed = parse_dateish(text)
    if parsed:
        return parsed
    year = int((target_start_date or published_date or "")[:4]) if (target_start_date or published_date) else None
    if year is None:
        return None
    match = re.search(r"^(\d{1,2})/(\d{1,2})$", text)
    if match:
        month = int(match.group(1))
        day = int(match.group(2))
        try:
            return dt.date(year, month, day).isoformat()
        except ValueError:
            return None
    return None


def infer_release_status(
    release_date: str | None,
    target_start_date: str | None,
    *,
    status_cell: str | None = None,
) -> str:
    if status_cell:
        normalized_status = clean_text(status_cell)
        if re.fullmatch(r"new", normalized_status, flags=re.IGNORECASE):
            return "NEW"
        if re.search(r"\d", normalized_status) or normalized_status in {"-", "\u2014", "\u2013"}:
            return f"Change From Last Weekend: {normalized_status}"
    if not release_date or not target_start_date:
        return "UNKNOWN"
    release = dt.date.fromisoformat(release_date)
    target_start = dt.date.fromisoformat(target_start_date)
    if target_start - dt.timedelta(days=4) <= release <= target_start + dt.timedelta(days=3):
        return "NEW"
    if release < target_start:
        week = max(1, ((target_start - release).days // 7) + 1)
        return f"Week {week}"
    return f"Release Date: {release.isoformat()}"


def parse_weekend_podium_blocks(
    blocks: list[HeadingBlock],
    *,
    article_url: str,
    article_title: str,
    published_date: str | None,
) -> tuple[list[WeekendPrediction], list[RejectedBlock]]:
    context = "\n".join(block.text for block in blocks)
    target_start, target_end = extract_date_range(f"{article_title}\n{context}", published_date)
    predictions: list[WeekendPrediction] = []
    rejected: list[RejectedBlock] = []
    in_podium = False
    fallback_rank: int | None = None
    for block in blocks:
        lowered = block.text.lower()
        if (
            "the boxoffice podium" in lowered
            or "boxoffice pro podium" in lowered
            or "boxoffice barometer" in lowered
        ):
            in_podium = True
            fallback_rank = None
            continue
        if block.level <= 2 and in_podium and "boxoffice podium" not in lowered:
            if is_podium_continuation_heading(block.text):
                fallback_rank = podium_continuation_rank(block.text)
                continue
            if not is_weekend_block_candidate(block.text):
                in_podium = False
                fallback_rank = None
        if not in_podium:
            continue
        parsed = parse_weekend_movie_block(
            block.text,
            article_url=article_url,
            target_start_date=target_start,
            target_end_date=target_end,
            row_ordinal=len(predictions) + 1,
            fallback_rank=fallback_rank,
        )
        if parsed is None:
            if mentions_weekend_range(block.text):
                rejected.append(RejectedBlock(raw_text=block.text, reason="weekend_block_missing_required_fields"))
            continue
        predictions.append(parsed)
    return predictions, rejected


def is_podium_continuation_heading(text: str) -> bool:
    lowered = text.lower()
    return "battle for" in lowered or re.search(r"\bfor\s+(second|third|2nd|3rd|#2|#3)\b", lowered) is not None


def podium_continuation_rank(text: str) -> int | None:
    lowered = text.lower()
    if re.search(r"\b(second|2nd|#2)\b", lowered):
        return 2
    if re.search(r"\b(third|3rd|#3)\b", lowered):
        return 3
    return None


def mentions_weekend_range(text: str) -> bool:
    return (
        re.search(
            r"(?:weekend\s+)?\d+-day\s+range|(?:\d+-day\s+)?opening\s+weekend\s+range|weekend\s+range",
            text,
            re.IGNORECASE,
        )
        is not None
    )


def is_weekend_block_candidate(text: str) -> bool:
    return mentions_weekend_range(text) and "$" in text


def parse_weekend_movie_block(
    text: str,
    *,
    article_url: str,
    target_start_date: str | None,
    target_end_date: str | None,
    row_ordinal: int,
    fallback_rank: int | None = None,
) -> WeekendPrediction | None:
    lines = [line for line in clean_multiline_text(text).split("\n") if line]
    if len(lines) < 3:
        return None
    rank, title = parse_rank_title(lines[0])
    if not title:
        return None
    distributor, release_status = parse_distributor_status(lines[1])
    if not distributor or not release_status:
        return None
    range_line = next(
        (
            line
            for line in lines
            if re.search(
                r"((?:\d+-day\s+)?opening\s+weekend|weekend(?:\s+\d+-day)?|(?:\d+-day))\s+range",
                line,
                re.I,
            )
        ),
        None,
    )
    if not range_line:
        return None
    money_range = parse_money_range(range_line)
    if money_range is None:
        return None
    showtime_share = parse_showtime_share("\n".join(lines))
    if rank is None:
        rank = fallback_rank
    forecast_metric = infer_forecast_metric(range_line, release_status)
    normalized = normalize_movie_title(title)
    key_material = "|".join(
        [
            article_url,
            str(row_ordinal),
            normalized,
            forecast_metric,
            str(money_range[0]),
            str(money_range[1]),
            str(target_start_date),
            str(target_end_date),
            PARSER_VERSION,
        ]
    )
    return WeekendPrediction(
        article_url=article_url,
        source_movie_title=title,
        normalized_movie_title=normalized,
        source_movie_id=boxofficepro_source_movie_id(
            market=DOMESTIC_MARKET,
            normalized_movie_title=normalized,
            target_start_date=target_start_date,
        ),
        distributor=distributor,
        release_status=release_status,
        source_rank=rank,
        market=DOMESTIC_MARKET,
        currency=DOMESTIC_CURRENCY,
        forecast_metric=forecast_metric,
        range_low_usd=money_range[0],
        range_high_usd=money_range[1],
        showtime_market_share_pct=showtime_share,
        target_start_date=target_start_date,
        target_end_date=target_end_date,
        raw_forecast_text="\n".join(lines),
        source_context="weekend_podium",
        parser_version=PARSER_VERSION,
        row_ordinal=row_ordinal,
        source_row_key=hashlib.sha256(key_material.encode("utf-8")).hexdigest(),
    )


def parse_rank_title(value: str) -> tuple[int | None, str]:
    text = clean_text(value)
    match = re.match(r"^(?:(\d+)\s*[\.)]\s*)?(.+)$", text)
    if not match:
        return None, strip_wildcard_title_prefix(text)
    rank = int(match.group(1)) if match.group(1) else None
    title = strip_wildcard_title_prefix(re.sub(r"^[:\-\s]+", "", match.group(2)).strip())
    return rank, title


def parse_distributor_status(value: str) -> tuple[str | None, str | None]:
    if "|" not in value:
        return None, None
    distributor, status = value.split("|", 1)
    distributor = clean_text(distributor)
    status = clean_text(status)
    return (distributor or None), (status or None)


def infer_forecast_metric(range_line: str, release_status: str) -> str:
    if re.search(r"opening\s+weekend\s+range", range_line, re.I) and not is_holdover_status(release_status):
        return "domestic_opening_weekend"
    return "domestic_weekend"


def is_holdover_status(release_status: str) -> bool:
    return re.search(r"\bweek\s+\d+\b", release_status, re.I) is not None


def parse_money_range(value: str) -> tuple[int, int] | None:
    text = clean_text(value).replace("\u2013", "-").replace("\u2014", "-")
    if "$" not in text:
        return None
    match = re.search(r"\$\s*(\d+(?:\.\d+)?)\s*([kmbKMB])?\s*(?:-|to)\s*\$?\s*(\d+(?:\.\d+)?)\s*([kmbKMB])?", text)
    if not match:
        return None
    low_unit = match.group(2)
    high_unit = match.group(4)
    unit = high_unit or low_unit or "M"
    low = money_to_usd(match.group(1), low_unit or unit)
    high = money_to_usd(match.group(3), high_unit or unit)
    if low is None or high is None:
        return None
    return min(low, high), max(low, high)


def parse_money_value(value: str) -> int | None:
    text = clean_text(value)
    if "$" not in text:
        return None
    match = re.search(r"\$\s*(\d[\d,]*(?:\.\d+)?)\s*([kmbKMB])?", text)
    if not match:
        return None
    number = match.group(1).replace(",", "")
    unit = match.group(2)
    if unit:
        return money_to_usd(number, unit)
    try:
        return int(round(float(number)))
    except ValueError:
        return None


def money_to_usd(number: str, unit: str) -> int | None:
    try:
        amount = float(number)
    except ValueError:
        return None
    multiplier = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}.get(unit.lower() or "m")
    if multiplier is None:
        return None
    return int(round(amount * multiplier))


def parse_showtime_share(value: str) -> float | None:
    match = re.search(r"showtime\s+market\s*share\s*:?\s*(\d+(?:\.\d+)?)\s*%", value, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def initialize_database(conn: Any) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS boxofficepro_predictions CASCADE;
        DROP TABLE IF EXISTS boxofficepro_forecast_articles CASCADE;
        DROP TABLE IF EXISTS boxofficepro_import_issues CASCADE;

        CREATE TABLE IF NOT EXISTS movies (
            movie_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            title TEXT NOT NULL,
            release_date DATE,
            movie_url TEXT,
            release_year INTEGER,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );

        ALTER TABLE movies
            ADD COLUMN IF NOT EXISTS title TEXT,
            ADD COLUMN IF NOT EXISTS movie_url TEXT,
            ADD COLUMN IF NOT EXISTS release_year INTEGER,
            ADD COLUMN IF NOT EXISTS release_date DATE,
            ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP;

        ALTER TABLE movies
            ALTER COLUMN movie_url DROP NOT NULL;

        CREATE UNIQUE INDEX IF NOT EXISTS uq_movies_movie_url_not_null
            ON movies(movie_url) WHERE movie_url IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_movies_title_release_year
            ON movies(title, release_year);
        CREATE INDEX IF NOT EXISTS idx_movies_release_date
            ON movies(release_date);

        CREATE TABLE IF NOT EXISTS movie_source_ids (
            movie_id BIGINT REFERENCES movies(movie_id),
            source TEXT NOT NULL,
            source_movie_id TEXT NOT NULL,
            source_title TEXT,
            match_status TEXT NOT NULL DEFAULT 'unmatched',
            match_method TEXT,
            match_score DOUBLE PRECISION,
            matched_at TIMESTAMPTZ,
            PRIMARY KEY (source, source_movie_id)
        );

        ALTER TABLE movie_source_ids
            ADD COLUMN IF NOT EXISTS source_title TEXT,
            ADD COLUMN IF NOT EXISTS match_status TEXT DEFAULT 'unmatched',
            ADD COLUMN IF NOT EXISTS match_method TEXT,
            ADD COLUMN IF NOT EXISTS match_score DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS matched_at TIMESTAMPTZ;

        CREATE TABLE IF NOT EXISTS boxofficepro_articles (
            article_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            article_url TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            author TEXT,
            discovered_date DATE,
            article_type TEXT NOT NULL,
            source_url TEXT NOT NULL,
            status TEXT NOT NULL,
            fetched_at TEXT,
            raw_cache_path TEXT,
            sha256 TEXT,
            parser_version TEXT,
            created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP::text),
            updated_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP::text)
        );

        CREATE TABLE IF NOT EXISTS boxofficepro_movie_match_overrides (
            override_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            source_movie_title TEXT NOT NULL,
            normalized_movie_title TEXT NOT NULL,
            article_url TEXT,
            movie_url TEXT NOT NULL,
            notes TEXT,
            active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP::text)
        );

        CREATE UNIQUE INDEX IF NOT EXISTS uq_boxofficepro_overrides_scope
            ON boxofficepro_movie_match_overrides(normalized_movie_title, (COALESCE(article_url, '')));

        CREATE TABLE IF NOT EXISTS boxofficepro_weekend_predictions (
            prediction_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            article_id BIGINT NOT NULL REFERENCES boxofficepro_articles(article_id),
            source_row_key TEXT NOT NULL,
            source_movie_id TEXT NOT NULL,
            source_movie_title TEXT NOT NULL,
            normalized_movie_title TEXT NOT NULL,
            distributor TEXT NOT NULL,
            release_status TEXT NOT NULL,
            source_rank INTEGER,
            market TEXT NOT NULL,
            currency TEXT NOT NULL,
            forecast_metric TEXT NOT NULL,
            range_low_usd BIGINT NOT NULL,
            range_high_usd BIGINT NOT NULL,
            showtime_market_share_pct DOUBLE PRECISION,
            target_start_date DATE,
            target_end_date DATE,
            raw_forecast_text TEXT NOT NULL,
            source_context TEXT NOT NULL,
            parser_version TEXT NOT NULL,
            row_ordinal INTEGER NOT NULL,
            matched_movie_id BIGINT REFERENCES movies(movie_id),
            match_status TEXT NOT NULL,
            match_method TEXT,
            match_score DOUBLE PRECISION,
            match_notes TEXT,
            fetched_at TEXT NOT NULL,
            raw_cache_path TEXT NOT NULL,
            UNIQUE(article_id, source_row_key)
        );

        ALTER TABLE boxofficepro_weekend_predictions
            ADD COLUMN IF NOT EXISTS source_movie_id TEXT;

        CREATE TABLE IF NOT EXISTS boxofficepro_ingest_issues (
            issue_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            issue_source TEXT NOT NULL,
            issue_type TEXT NOT NULL,
            article_url TEXT,
            source_movie_title TEXT,
            details TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP::text),
            UNIQUE(issue_source, issue_type, article_url, source_movie_title, details)
        );

        CREATE INDEX IF NOT EXISTS idx_boxofficepro_articles_status
            ON boxofficepro_articles(status);
        CREATE INDEX IF NOT EXISTS idx_boxofficepro_articles_discovered_date
            ON boxofficepro_articles(discovered_date);
        CREATE INDEX IF NOT EXISTS idx_boxofficepro_weekend_predictions_movie
            ON boxofficepro_weekend_predictions(matched_movie_id);
        CREATE INDEX IF NOT EXISTS idx_boxofficepro_weekend_predictions_title
            ON boxofficepro_weekend_predictions(normalized_movie_title);
        """
    )


def upsert_article(
    conn: Any,
    article: ArchiveArticle,
    *,
    status: str,
    fetched_at: str | None = None,
    raw_cache_path: Path | None = None,
    html: str | None = None,
) -> int:
    sha256 = hashlib.sha256(html.encode("utf-8")).hexdigest() if html is not None else None
    conn.execute(
        """
        INSERT INTO boxofficepro_articles (
            article_url, title, author, discovered_date, article_type, source_url,
            status, fetched_at, raw_cache_path, sha256, parser_version, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT(article_url) DO UPDATE SET
            title = excluded.title,
            author = excluded.author,
            discovered_date = excluded.discovered_date,
            article_type = excluded.article_type,
            source_url = excluded.source_url,
            status = excluded.status,
            fetched_at = COALESCE(excluded.fetched_at, boxofficepro_articles.fetched_at),
            raw_cache_path = COALESCE(excluded.raw_cache_path, boxofficepro_articles.raw_cache_path),
            sha256 = COALESCE(excluded.sha256, boxofficepro_articles.sha256),
            parser_version = excluded.parser_version,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            article.article_url,
            article.title,
            article.author,
            article.published_date,
            article.article_type,
            article.source_url,
            status,
            fetched_at,
            str(raw_cache_path) if raw_cache_path is not None else None,
            sha256,
            PARSER_VERSION,
        ),
    )
    return int(
        conn.execute(
            "SELECT article_id FROM boxofficepro_articles WHERE article_url = %s",
            (article.article_url,),
        ).fetchone()[0]
    )


def insert_predictions(
    conn: Any,
    article_id: int,
    predictions: list[WeekendPrediction],
    *,
    fetched_at: str,
    raw_cache_path: Path,
) -> None:
    if not predictions:
        return
    conn.executemany(
        """
        INSERT INTO boxofficepro_weekend_predictions (
            article_id, source_row_key, source_movie_id, source_movie_title, normalized_movie_title,
            distributor, release_status, source_rank, market, currency, forecast_metric,
            range_low_usd, range_high_usd, showtime_market_share_pct,
            target_start_date, target_end_date, raw_forecast_text, source_context,
            parser_version, row_ordinal, matched_movie_id, match_status, match_method,
            match_score, match_notes, fetched_at, raw_cache_path
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(article_id, source_row_key) DO UPDATE SET
            source_movie_id = excluded.source_movie_id,
            source_movie_title = excluded.source_movie_title,
            normalized_movie_title = excluded.normalized_movie_title,
            distributor = excluded.distributor,
            release_status = excluded.release_status,
            source_rank = excluded.source_rank,
            market = excluded.market,
            currency = excluded.currency,
            forecast_metric = excluded.forecast_metric,
            range_low_usd = excluded.range_low_usd,
            range_high_usd = excluded.range_high_usd,
            showtime_market_share_pct = excluded.showtime_market_share_pct,
            target_start_date = excluded.target_start_date,
            target_end_date = excluded.target_end_date,
            raw_forecast_text = excluded.raw_forecast_text,
            source_context = excluded.source_context,
            parser_version = excluded.parser_version,
            row_ordinal = excluded.row_ordinal,
            matched_movie_id = excluded.matched_movie_id,
            match_status = excluded.match_status,
            match_method = excluded.match_method,
            match_score = excluded.match_score,
            match_notes = excluded.match_notes,
            fetched_at = excluded.fetched_at,
            raw_cache_path = excluded.raw_cache_path
        """,
        [
            (
                article_id,
                prediction.source_row_key,
                prediction.source_movie_id,
                prediction.source_movie_title,
                prediction.normalized_movie_title,
                prediction.distributor,
                prediction.release_status,
                prediction.source_rank,
                prediction.market,
                prediction.currency,
                prediction.forecast_metric,
                prediction.range_low_usd,
                prediction.range_high_usd,
                prediction.showtime_market_share_pct,
                prediction.target_start_date,
                prediction.target_end_date,
                prediction.raw_forecast_text,
                prediction.source_context,
                prediction.parser_version,
                prediction.row_ordinal,
                prediction.matched_movie_id,
                prediction.match_status,
                prediction.match_method,
                prediction.match_score,
                prediction.match_notes,
                fetched_at,
                str(raw_cache_path),
            )
            for prediction in predictions
        ],
    )


def insert_issue(
    conn: Any,
    *,
    issue_source: str,
    issue_type: str,
    article_url: str | None,
    source_movie_title: str | None,
    details: str,
) -> None:
    conn.execute(
        """
        INSERT INTO boxofficepro_ingest_issues (
            issue_source, issue_type, article_url, source_movie_title, details
        ) VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """,
        (issue_source, issue_type, article_url, source_movie_title, details),
    )


def clear_article_issues(conn: Any, *, issue_source: str, article_url: str) -> None:
    conn.execute(
        """
        DELETE FROM boxofficepro_ingest_issues
        WHERE issue_source = %s
          AND article_url = %s
        """,
        (issue_source, article_url),
    )


def load_movie_candidates(conn: Any) -> list[MovieCandidate]:
    columns = movie_table_columns(conn)
    movie_url_expr = "movie_url" if "movie_url" in columns else "NULL AS movie_url"
    release_year_expr = "release_year" if "release_year" in columns else "NULL AS release_year"
    release_date_expr = "release_date" if "release_date" in columns else "NULL AS release_date"
    rows = conn.execute(
        f"""
        SELECT movie_id, {movie_url_expr}, title, {release_year_expr}, {release_date_expr}
        FROM movies
        WHERE title IS NOT NULL
        ORDER BY movie_id
        """
    ).fetchall()
    return [
        MovieCandidate(
            movie_id=int(row[0]),
            movie_url=str(row[1]) if row[1] is not None else None,
            title=str(row[2]),
            release_year=int(row[3]) if row[3] is not None else None,
            release_date=str(row[4]) if row[4] is not None else None,
            normalized_title=normalize_movie_title(str(row[2])),
        )
        for row in rows
    ]


def movie_table_columns(conn: Any) -> set[str]:
    rows = conn.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'movies'
        """
    ).fetchall()
    return {str(row[0]) for row in rows}


def relation_exists(conn: Any, relation_name: str) -> bool:
    row = conn.execute("SELECT to_regclass(%s)", (relation_name,)).fetchone()
    return bool(row and row[0])


def match_predictions(conn: Any, predictions: list[WeekendPrediction]) -> list[WeekendPrediction]:
    candidates = load_movie_candidates(conn)
    matched: list[WeekendPrediction] = []
    for prediction in predictions:
        match = match_prediction(conn, prediction, candidates)
        if match.movie_id is not None:
            upsert_boxofficepro_movie_source_id(
                conn,
                movie_id=match.movie_id,
                prediction=prediction,
                match_status=match.status,
                match_method=match.method,
                match_score=match.score,
            )
            repoint_boxofficepro_predictions(
                conn,
                prediction=prediction,
                movie_id=match.movie_id,
                match_status=match.status,
                match_method=match.method,
                match_score=match.score,
                match_notes=match.notes,
            )
        matched.append(
            replace(
                prediction,
                matched_movie_id=match.movie_id,
                match_status=match.status,
                match_method=match.method,
                match_score=match.score,
                match_notes=match.notes,
            )
        )
    return matched


def match_prediction(
    conn: Any,
    prediction: WeekendPrediction,
    candidates: list[MovieCandidate],
) -> MovieMatch:
    override = find_manual_override(conn, prediction)
    if override is not None:
        return override
    matches = [candidate for candidate in candidates if candidate.normalized_title == prediction.normalized_movie_title]
    existing = choose_existing_movie_match(prediction, matches)
    if existing is not None:
        return existing
    source_id_match = find_boxofficepro_source_id_match(conn, prediction)
    if source_id_match is not None:
        return source_id_match
    if not matches:
        if can_provision_movie(prediction):
            return provision_boxofficepro_movie(conn, prediction)
        return MovieMatch(None, "unmatched", "normalized_exact", 0.0, "No movie title matched")
    return MovieMatch(None, "ambiguous", "normalized_exact", 0.5, "Multiple movies share the title")


def choose_existing_movie_match(
    prediction: WeekendPrediction,
    matches: list[MovieCandidate],
) -> MovieMatch | None:
    if not matches:
        return None
    target_date = prediction.target_start_date if prediction.forecast_metric == "domestic_opening_weekend" else None
    if target_date is not None:
        exact_date_matches = [candidate for candidate in matches if candidate.release_date == target_date]
        if exact_date_matches:
            candidate = preferred_movie_candidate(exact_date_matches)
            status = "matched" if candidate.movie_url is not None else "provisional"
            return MovieMatch(
                candidate.movie_id,
                status,
                "normalized_exact_release_date",
                1.0,
                f"Chose movie with release_date {target_date}",
            )
    if len(matches) == 1:
        candidate = matches[0]
        status = "matched" if candidate.movie_url is not None else "provisional"
        return MovieMatch(candidate.movie_id, status, "normalized_exact", 1.0, None)
    target_year = infer_prediction_year(prediction)
    if target_year is not None:
        distances = [(abs((candidate.release_year or target_year) - target_year), candidate) for candidate in matches]
        best_distance = min(distance for distance, _candidate in distances)
        best_matches = [candidate for distance, candidate in distances if distance == best_distance]
        canonical_best_matches = [candidate for candidate in best_matches if candidate.movie_url is not None]
        if len(canonical_best_matches) == 1:
            candidate = canonical_best_matches[0]
            return MovieMatch(
                candidate.movie_id,
                "matched",
                "normalized_exact_release_year",
                1.0 - min(float(best_distance) / 10.0, 0.9),
                f"Chose closest release_year to {target_year}",
            )
        if len(best_matches) == 1:
            candidate = best_matches[0]
            status = "matched" if candidate.movie_url is not None else "provisional"
            return MovieMatch(
                candidate.movie_id,
                status,
                "normalized_exact_release_year",
                1.0 - min(float(best_distance) / 10.0, 0.9),
                f"Chose closest release_year to {target_year}",
            )
    return None


def preferred_movie_candidate(candidates: list[MovieCandidate]) -> MovieCandidate:
    return sorted(candidates, key=lambda candidate: (candidate.movie_url is None, candidate.movie_id))[0]


def find_boxofficepro_source_id_match(conn: Any, prediction: WeekendPrediction) -> MovieMatch | None:
    if not relation_exists(conn, "movie_source_ids"):
        return None
    row = conn.execute(
        """
        SELECT m.movie_id, m.movie_url, src.match_status, src.match_score
        FROM movie_source_ids src
        JOIN movies m ON m.movie_id = src.movie_id
        WHERE src.source = 'boxofficepro'
          AND src.source_movie_id = %s
        LIMIT 1
        """,
        (prediction.source_movie_id,),
    ).fetchone()
    if row is None:
        return None
    status = "matched" if row[1] is not None else "provisional"
    stored_status = str(row[2]) if row[2] is not None else status
    if stored_status in {"matched", "provisional"}:
        status = stored_status
    return MovieMatch(
        int(row[0]),
        status,
        "boxofficepro_source_id",
        float(row[3]) if row[3] is not None else 1.0,
        f"Matched existing Boxoffice Pro source id {prediction.source_movie_id}",
    )


def can_provision_movie(prediction: WeekendPrediction) -> bool:
    return prediction.forecast_metric == "domestic_opening_weekend" and prediction.target_start_date is not None


def provision_boxofficepro_movie(conn: Any, prediction: WeekendPrediction) -> MovieMatch:
    existing = find_boxofficepro_source_id_match(conn, prediction)
    if existing is not None:
        return existing
    release_year = infer_prediction_year(prediction)
    row = conn.execute(
        """
        INSERT INTO movies (title, release_year, release_date, updated_at)
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        RETURNING movie_id
        """,
        (prediction.source_movie_title, release_year, prediction.target_start_date),
    ).fetchone()
    movie_id = int(row[0])
    upsert_boxofficepro_movie_source_id(
        conn,
        movie_id=movie_id,
        prediction=prediction,
        match_status="provisional",
        match_method="provisional_boxofficepro_identity",
        match_score=1.0,
    )
    return MovieMatch(
        movie_id,
        "provisional",
        "provisional_boxofficepro_identity",
        1.0,
        f"Created provisional movie from Boxoffice Pro source id {prediction.source_movie_id}",
    )


def upsert_boxofficepro_movie_source_id(
    conn: Any,
    *,
    movie_id: int,
    prediction: WeekendPrediction,
    match_status: str,
    match_method: str | None,
    match_score: float | None,
) -> None:
    if not relation_exists(conn, "movie_source_ids"):
        return
    conn.execute(
        """
        INSERT INTO movie_source_ids (
            movie_id, source, source_movie_id, source_title,
            match_status, match_method, match_score, matched_at
        )
        VALUES (%s, 'boxofficepro', %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT(source, source_movie_id) DO UPDATE SET
            movie_id = excluded.movie_id,
            source_title = excluded.source_title,
            match_status = excluded.match_status,
            match_method = excluded.match_method,
            match_score = excluded.match_score,
            matched_at = excluded.matched_at
        """,
        (
            movie_id,
            prediction.source_movie_id,
            prediction.source_movie_title,
            match_status,
            match_method,
            match_score,
        ),
    )


def repoint_boxofficepro_predictions(
    conn: Any,
    *,
    prediction: WeekendPrediction,
    movie_id: int,
    match_status: str,
    match_method: str | None,
    match_score: float | None,
    match_notes: str | None,
) -> None:
    if not relation_exists(conn, "boxofficepro_weekend_predictions"):
        return
    columns = boxofficepro_prediction_columns(conn)
    if "source_movie_id" not in columns:
        return
    conn.execute(
        """
        UPDATE boxofficepro_weekend_predictions
        SET matched_movie_id = %s,
            match_status = %s,
            match_method = %s,
            match_score = %s,
            match_notes = %s
        WHERE source_movie_id = %s
        """,
        (movie_id, match_status, match_method, match_score, match_notes, prediction.source_movie_id),
    )


def boxofficepro_prediction_columns(conn: Any) -> set[str]:
    rows = conn.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'boxofficepro_weekend_predictions'
        """
    ).fetchall()
    return {str(row[0]) for row in rows}


def find_manual_override(conn: Any, prediction: WeekendPrediction) -> MovieMatch | None:
    if "movie_url" not in movie_table_columns(conn):
        return None
    row = conn.execute(
        """
        SELECT m.movie_id, o.movie_url
        FROM boxofficepro_movie_match_overrides o
        JOIN movies m ON m.movie_url = o.movie_url
        WHERE o.active = TRUE
          AND o.normalized_movie_title = %s
          AND (o.article_url = %s OR o.article_url IS NULL)
        ORDER BY CASE WHEN o.article_url = %s THEN 0 ELSE 1 END
        LIMIT 1
        """,
        (prediction.normalized_movie_title, prediction.article_url, prediction.article_url),
    ).fetchone()
    if row is None:
        return None
    return MovieMatch(int(row[0]), "matched", "manual_override", 1.0, f"Override movie_url={row[1]}")


def infer_prediction_year(prediction: WeekendPrediction) -> int | None:
    for value in (prediction.target_start_date, prediction.target_end_date):
        if value and re.match(r"20\d{2}", value):
            return int(value[:4])
    return None


def article_already_parsed(conn: Any, article_url: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM boxofficepro_articles
        WHERE article_url = %s
          AND status = 'parsed'
        LIMIT 1
        """,
        (article_url,),
    ).fetchone()
    return row is not None


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    archive_urls = [archive_url(page) for page in range(1, args.max_pages + 1)]
    if args.print_cache_paths:
        fetcher = HtmlFetcher(
            args.cache_dir,
            refresh=False,
            offline=True,
            delay_seconds=args.delay_seconds,
            user_agent=args.user_agent,
            fetch_mode=args.fetch_mode,
            browser_user_agent=args.browser_user_agent,
        )
        urls = discovery_urls_for_dry_run(args, archive_urls) + args.cache_url
        for url in urls:
            print(f"{url}\t{fetcher.cache_path(url)}")
        return 0
    if args.dry_run:
        urls = discovery_urls_for_dry_run(args, archive_urls)
        for url in urls:
            print(url)
        print(f"Discovery URLs: {len(urls)}", file=sys.stderr)
        return 0

    fetcher = HtmlFetcher(
        args.cache_dir,
        refresh=args.refresh,
        offline=args.offline,
        delay_seconds=args.delay_seconds,
        user_agent=args.user_agent,
        fetch_mode=args.fetch_mode,
        browser_user_agent=args.browser_user_agent,
    )
    conn = connect_database(args.database_url)
    try:
        initialize_database(conn)
        conn.commit()
        articles = discover_articles(fetcher, args)
        imported_articles = 0
        imported_predictions = 0
        skipped_articles = 0
        for index, archive_article in enumerate(articles, start=1):
            if not is_weekend_preview_article(archive_article):
                continue
            if not args.refresh and article_already_parsed(conn, archive_article.article_url):
                skipped_articles += 1
                print(f"Skipping parsed article {index}/{len(articles)} {archive_article.title}", file=sys.stderr)
                continue
            upsert_article(conn, archive_article, status="discovered")
            conn.commit()
            print(f"Reading article {index}/{len(articles)} {archive_article.title}", file=sys.stderr)
            try:
                html, cache_path, _fetched = fetcher.get(archive_article.article_url)
            except FetchBlocked as exc:
                fetched_at = dt.datetime.now(dt.UTC).isoformat()
                upsert_article(
                    conn,
                    archive_article,
                    status="article_page_unavailable",
                    fetched_at=fetched_at,
                    raw_cache_path=fetcher.cache_path(archive_article.article_url),
                    html="",
                )
                insert_issue(
                    conn,
                    issue_source=args.issue_source,
                    issue_type="article_page_unavailable",
                    article_url=archive_article.article_url,
                    source_movie_title=None,
                    details=str(exc),
                )
                conn.commit()
                imported_articles += 1
                continue
            fetched_at = dt.datetime.now(dt.UTC).isoformat()
            article, predictions, rejected = parse_article(
                html,
                article_url=archive_article.article_url,
                fallback=archive_article,
            )
            article_id = upsert_article(
                conn,
                article,
                status="parsed",
                fetched_at=fetched_at,
                raw_cache_path=cache_path,
                html=html,
            )
            predictions = match_predictions(conn, predictions)
            clear_article_issues(conn, issue_source=args.issue_source, article_url=article.article_url)
            for rejected_block in rejected:
                insert_issue(
                    conn,
                    issue_source=args.issue_source,
                    issue_type="rejected_weekend_block",
                    article_url=article.article_url,
                    source_movie_title=None,
                    details=f"{rejected_block.reason}: {rejected_block.raw_text}",
                )
            if not predictions:
                insert_issue(
                    conn,
                    issue_source=args.issue_source,
                    issue_type="no_weekend_predictions_parsed",
                    article_url=article.article_url,
                    source_movie_title=None,
                    details=f"No high-confidence Weekend Preview blocks parsed from {article.title}",
                )
            insert_predictions(conn, article_id, predictions, fetched_at=fetched_at, raw_cache_path=cache_path)
            conn.commit()
            imported_articles += 1
            imported_predictions += len(predictions)
        print(
            f"Imported {imported_articles} articles and {imported_predictions} weekend predictions; "
            f"skipped {skipped_articles} parsed articles.",
            file=sys.stderr,
        )
    finally:
        fetcher.close()
        conn.close()
    return 0


def discovery_urls_for_dry_run(args: argparse.Namespace, archive_urls: list[str]) -> list[str]:
    if args.discovery == "rss":
        return [FORECAST_RSS_URL]
    if args.discovery == "archive":
        return archive_urls
    return [FORECAST_RSS_URL, *archive_urls]


def discover_articles(fetcher: HtmlFetcher, args: argparse.Namespace) -> list[ArchiveArticle]:
    discovery = getattr(args, "discovery", "archive")
    if discovery == "rss":
        return discover_rss_articles(fetcher, args)
    if discovery == "archive":
        return discover_archive_articles(fetcher, args)

    all_rss_articles = fetch_rss_articles(fetcher)
    rss_articles = filter_discovered_articles(all_rss_articles, args)
    oldest_rss_date = oldest_published_date(all_rss_articles)
    if oldest_rss_date is None or args.start_date < oldest_rss_date:
        archive_articles = discover_archive_articles(fetcher, args)
        return merge_discovered_articles([rss_articles, archive_articles], args)
    return limit_articles(rss_articles, args)


def discover_rss_articles(fetcher: HtmlFetcher, args: argparse.Namespace) -> list[ArchiveArticle]:
    return limit_articles(filter_discovered_articles(fetch_rss_articles(fetcher), args), args)


def fetch_rss_articles(fetcher: HtmlFetcher) -> list[ArchiveArticle]:
    print(f"Reading RSS feed {FORECAST_RSS_URL}", file=sys.stderr)
    xml_text, _cache_path, _fetched = fetcher.get(FORECAST_RSS_URL)
    return parse_rss(xml_text, source_url=FORECAST_RSS_URL)


def discover_archive_articles(fetcher: HtmlFetcher, args: argparse.Namespace) -> list[ArchiveArticle]:
    articles_by_url: dict[str, ArchiveArticle] = {}
    for page in range(1, args.max_pages + 1):
        url = archive_url(page)
        print(f"Reading archive page {page} {url}", file=sys.stderr)
        try:
            html, _cache_path, _fetched = fetcher.get(url)
        except FetchBlocked as exc:
            if articles_by_url:
                print(f"Stopping archive discovery at page {page}: {exc}", file=sys.stderr)
                break
            raise
        page_articles = parse_archive(html, source_url=url)
        if not page_articles:
            break
        page_has_in_window = False
        page_all_older = True
        for article in page_articles:
            if article.published_date is None:
                page_all_older = False
                if is_weekend_preview_article(article):
                    articles_by_url.setdefault(article.article_url, article)
                continue
            published = dt.date.fromisoformat(article.published_date)
            if published >= args.start_date:
                page_all_older = False
            if not is_weekend_preview_article(article):
                continue
            if args.start_date <= published <= args.end_date:
                page_has_in_window = True
                articles_by_url.setdefault(article.article_url, article)
        if page_all_older and not page_has_in_window:
            break
    articles = sorted(
        dedupe_articles_by_content(list(articles_by_url.values())),
        key=lambda article: (article.published_date or "", article.article_url),
    )
    return limit_articles(articles, args)


def filter_discovered_articles(articles: list[ArchiveArticle], args: argparse.Namespace) -> list[ArchiveArticle]:
    filtered: list[ArchiveArticle] = []
    for article in articles:
        if not is_weekend_preview_article(article):
            continue
        if article.published_date is None:
            filtered.append(article)
            continue
        published = dt.date.fromisoformat(article.published_date)
        if args.start_date <= published <= args.end_date:
            filtered.append(article)
    return sorted(filtered, key=lambda article: (article.published_date or "", article.article_url))


def oldest_published_date(articles: list[ArchiveArticle]) -> dt.date | None:
    dates = [
        dt.date.fromisoformat(article.published_date)
        for article in articles
        if article.published_date is not None
    ]
    return min(dates) if dates else None


def merge_discovered_articles(article_groups: list[list[ArchiveArticle]], args: argparse.Namespace) -> list[ArchiveArticle]:
    articles_by_url: dict[str, ArchiveArticle] = {}
    for group in article_groups:
        for article in group:
            articles_by_url.setdefault(article.article_url, article)
    articles = sorted(
        dedupe_articles_by_content(list(articles_by_url.values())),
        key=lambda article: (article.published_date or "", article.article_url),
    )
    return limit_articles(articles, args)


def limit_articles(articles: list[ArchiveArticle], args: argparse.Namespace) -> list[ArchiveArticle]:
    if args.max_articles is not None:
        return articles[: args.max_articles]
    return articles


def validate_args(args: argparse.Namespace) -> None:
    if args.end_date < args.start_date:
        raise SystemExit("--end-date must be on or after --start-date")
    if args.discovery not in {"auto", "rss", "archive"}:
        raise SystemExit("--discovery must be one of: auto, rss, archive")
    if args.fetch_mode not in {"auto", "http", "browser"}:
        raise SystemExit("--fetch-mode must be one of: auto, http, browser")
    if args.delay_seconds < MIN_DELAY_SECONDS and not args.offline and not args.dry_run:
        raise SystemExit(f"--delay-seconds must be at least {MIN_DELAY_SECONDS:g}")
    if args.fetch_mode in {"auto", "http"} and "bot" not in args.user_agent.lower() and not args.offline and not args.dry_run:
        raise SystemExit("--user-agent must identify the scraper as a bot")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import Boxoffice Pro Weekend Preview forecasts.")
    parser.add_argument("--start-date", type=parse_date_arg, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=parse_date_arg, default=DEFAULT_END_DATE)
    parser.add_argument(
        "--database-url",
        default=database_url_from_env(),
        help="PostgreSQL connection URL. Defaults to DATABASE_URL or POSTGRES_DSN.",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Raw HTML cache directory.")
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=MIN_DELAY_SECONDS,
        help="Delay between uncached HTTP requests. Must be at least 20.",
    )
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="HTTP User-Agent. Must identify as a bot.")
    parser.add_argument(
        "--browser-user-agent",
        default=DEFAULT_BROWSER_USER_AGENT,
        help="User-Agent used only for Playwright browser fallback requests.",
    )
    parser.add_argument(
        "--discovery",
        choices=("auto", "rss", "archive"),
        default="auto",
        help="Article discovery source. auto uses RSS unless the requested start date predates the RSS window.",
    )
    parser.add_argument(
        "--fetch-mode",
        choices=("auto", "http", "browser"),
        default="auto",
        help="Fetch method. auto tries HTTP first, then Playwright browser fallback on HTTP 403.",
    )
    parser.add_argument("--refresh", action="store_true", help="Reparse even when article status is parsed.")
    parser.add_argument("--offline", action="store_true", help="Require all pages to exist in cache.")
    parser.add_argument("--dry-run", action="store_true", help="Print discovery URLs and exit.")
    parser.add_argument(
        "--print-cache-paths",
        action="store_true",
        help="Print expected cache file paths for discovery URLs and any --cache-url values, then exit.",
    )
    parser.add_argument(
        "--cache-url",
        action="append",
        default=[],
        help="Extra article URL to include when printing cache paths. May be repeated.",
    )
    parser.add_argument("--max-pages", type=int, default=25, help="Maximum archive pages to inspect.")
    parser.add_argument("--max-articles", type=int, help="Optional cap for smoke tests after archive discovery.")
    parser.add_argument("--issue-source", default="boxofficepro_weekend_import", help="Label used for import issues.")
    return parser


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
