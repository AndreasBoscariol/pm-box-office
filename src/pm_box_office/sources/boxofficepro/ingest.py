#!/usr/bin/env python3
"""Ingest Boxoffice Pro Weekend Preview forecasts into PostgreSQL.

This source intentionally parses only high-confidence Weekend Preview
"Boxoffice Podium" blocks. Older generic Boxoffice Pro prediction tables are
dropped during initialization because they admitted too much prose garbage.
"""

from __future__ import annotations

import argparse
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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from pm_box_office.db.connection import connect_database, database_url_from_env


BASE_URL = "https://www.boxofficepro.com"
FORECAST_ARCHIVE_URL = f"{BASE_URL}/category/forecasts-tracking/"
DEFAULT_START_DATE = dt.date(2026, 6, 1)
DEFAULT_END_DATE = dt.date(2026, 6, 30)
DEFAULT_CACHE_DIR = Path("data/raw/boxofficepro")
DEFAULT_USER_AGENT = "pm-box-office-boxofficepro-bot/1.0 (+personal research; set --user-agent contact)"
TRANSIENT_STATUSES = {429, 500, 502, 503, 504}
MIN_DELAY_SECONDS = 20.0
DOMESTIC_MARKET = "US_CA"
DOMESTIC_CURRENCY = "USD"
PARSER_VERSION = "weekend_podium_v1"


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
        timeout_seconds: float = 60.0,
    ) -> None:
        self.cache_dir = cache_dir
        self.refresh = refresh
        self.offline = offline
        self.delay_seconds = delay_seconds
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self._last_request_at = 0.0
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.html"

    def get(self, url: str) -> tuple[str, Path, bool]:
        cache_path = self.cache_path(url)
        if cache_path.exists() and not self.refresh:
            return cache_path.read_text(encoding="utf-8"), cache_path, False
        if self.offline:
            raise FetchBlocked(
                f"Cache miss in offline mode: {url}\n"
                f"Expected cached HTML at: {cache_path}\n"
                "Open the URL in a browser, save the page source as that file, then rerun with --offline."
            )

        last_error: Exception | None = None
        for attempt in range(2):
            self._wait()
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": "text/html,application/xhtml+xml",
                    "User-Agent": self.user_agent,
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    body = response.read().decode("utf-8", errors="replace")
                cache_path.write_text(body, encoding="utf-8")
                self._last_request_at = time.monotonic()
                return body, cache_path, True
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code == 403:
                    raise FetchBlocked(
                        "Boxoffice Pro returned HTTP 403. The site may block automated HTTP clients.\n"
                        f"URL: {url}\n"
                        f"Expected cached HTML path: {cache_path}\n"
                        "Open the URL in a browser, save the page source as that file, then rerun with --offline."
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


class WeekendPreviewParser(HTMLParser):
    """Collect article headings while preserving <br> boundaries."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: str | None = None
        self.author: str | None = None
        self.published_date: str | None = None
        self.blocks: list[HeadingBlock] = []
        self._capture_heading: tuple[str, int] | None = None
        self._heading_parts: list[str] = []
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

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._capture_heading:
            self._heading_parts.append(data)


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()


def clean_multiline_text(value: str) -> str:
    text = value.replace("\xa0", " ").replace("\r", "\n")
    lines = [clean_text(line) for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def absolute_url(href: str) -> str:
    return urllib.parse.urljoin(BASE_URL, href)


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
    if "weekend preview" in normalized or "weekend forecast" in normalized:
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
    return article, predictions, rejected


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
    for block in blocks:
        lowered = block.text.lower()
        if "the boxoffice podium" in lowered:
            in_podium = True
            continue
        if block.level <= 2 and in_podium and "boxoffice podium" not in lowered:
            in_podium = False
        if not in_podium:
            continue
        parsed = parse_weekend_movie_block(
            block.text,
            article_url=article_url,
            target_start_date=target_start,
            target_end_date=target_end,
            row_ordinal=len(predictions) + 1,
        )
        if parsed is None:
            if mentions_weekend_range(block.text):
                rejected.append(RejectedBlock(raw_text=block.text, reason="weekend_block_missing_required_fields"))
            continue
        predictions.append(parsed)
    return predictions, rejected


def mentions_weekend_range(text: str) -> bool:
    return "weekend range" in text.lower()


def is_weekend_block_candidate(text: str) -> bool:
    return mentions_weekend_range(text) and "$" in text


def parse_weekend_movie_block(
    text: str,
    *,
    article_url: str,
    target_start_date: str | None,
    target_end_date: str | None,
    row_ordinal: int,
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
    range_line = next((line for line in lines if re.search(r"(opening\s+weekend|weekend)\s+range", line, re.I)), None)
    if not range_line:
        return None
    money_range = parse_money_range(range_line)
    if money_range is None:
        return None
    showtime_share = parse_showtime_share("\n".join(lines))
    forecast_metric = (
        "domestic_opening_weekend"
        if re.search(r"opening\s+weekend\s+range", range_line, re.I)
        else "domestic_weekend"
    )
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
        return None, text
    rank = int(match.group(1)) if match.group(1) else None
    title = re.sub(r"^[:\-\s]+", "", match.group(2)).strip()
    return rank, title


def parse_distributor_status(value: str) -> tuple[str | None, str | None]:
    if "|" not in value:
        return None, None
    distributor, status = value.split("|", 1)
    distributor = clean_text(distributor)
    status = clean_text(status)
    return (distributor or None), (status or None)


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
            article_id, source_row_key, source_movie_title, normalized_movie_title,
            distributor, release_status, source_rank, market, currency, forecast_metric,
            range_low_usd, range_high_usd, showtime_market_share_pct,
            target_start_date, target_end_date, raw_forecast_text, source_context,
            parser_version, row_ordinal, matched_movie_id, match_status, match_method,
            match_score, match_notes, fetched_at, raw_cache_path
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(article_id, source_row_key) DO UPDATE SET
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


def load_movie_candidates(conn: Any) -> list[MovieCandidate]:
    columns = movie_table_columns(conn)
    movie_url_expr = "movie_url" if "movie_url" in columns else "NULL AS movie_url"
    release_year_expr = "release_year" if "release_year" in columns else "NULL AS release_year"
    rows = conn.execute(
        f"""
        SELECT movie_id, {movie_url_expr}, title, {release_year_expr}
        FROM movies
        ORDER BY movie_id
        """
    ).fetchall()
    return [
        MovieCandidate(
            movie_id=int(row[0]),
            movie_url=str(row[1]) if row[1] is not None else None,
            title=str(row[2]),
            release_year=int(row[3]) if row[3] is not None else None,
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


def match_predictions(conn: Any, predictions: list[WeekendPrediction]) -> list[WeekendPrediction]:
    candidates = load_movie_candidates(conn)
    matched: list[WeekendPrediction] = []
    for prediction in predictions:
        match = match_prediction(conn, prediction, candidates)
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
    if not matches:
        return MovieMatch(None, "unmatched", "normalized_exact", 0.0, "No The Numbers title matched")
    if len(matches) == 1:
        return MovieMatch(matches[0].movie_id, "matched", "normalized_exact", 1.0, None)
    target_year = infer_prediction_year(prediction)
    if target_year is not None:
        distances = [
            (abs((candidate.release_year or target_year) - target_year), candidate.movie_id, candidate)
            for candidate in matches
        ]
        distances.sort(key=lambda item: (item[0], item[1]))
        if len(distances) == 1 or distances[0][0] < distances[1][0]:
            candidate = distances[0][2]
            return MovieMatch(
                candidate.movie_id,
                "matched",
                "normalized_exact_release_year",
                1.0 - min(float(distances[0][0]) / 10.0, 0.9),
                f"Chose closest release_year to {target_year}",
            )
    return MovieMatch(None, "ambiguous", "normalized_exact", 0.5, "Multiple The Numbers movies share the title")


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
        )
        urls = archive_urls + args.cache_url
        for url in urls:
            print(f"{url}\t{fetcher.cache_path(url)}")
        return 0
    if args.dry_run:
        for url in archive_urls:
            print(url)
        print(f"Archive URLs: {len(archive_urls)}", file=sys.stderr)
        return 0

    fetcher = HtmlFetcher(
        args.cache_dir,
        refresh=args.refresh,
        offline=args.offline,
        delay_seconds=args.delay_seconds,
        user_agent=args.user_agent,
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
        conn.close()
    return 0


def discover_articles(fetcher: HtmlFetcher, args: argparse.Namespace) -> list[ArchiveArticle]:
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
    articles = sorted(articles_by_url.values(), key=lambda article: (article.published_date or "", article.article_url))
    if args.max_articles is not None:
        articles = articles[: args.max_articles]
    return articles


def validate_args(args: argparse.Namespace) -> None:
    if args.end_date < args.start_date:
        raise SystemExit("--end-date must be on or after --start-date")
    if args.delay_seconds < MIN_DELAY_SECONDS and not args.offline and not args.dry_run:
        raise SystemExit(f"--delay-seconds must be at least {MIN_DELAY_SECONDS:g}")
    if "bot" not in args.user_agent.lower() and not args.offline and not args.dry_run:
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
    parser.add_argument("--refresh", action="store_true", help="Reparse even when article status is parsed.")
    parser.add_argument("--offline", action="store_true", help="Require all pages to exist in cache.")
    parser.add_argument("--dry-run", action="store_true", help="Print archive URLs and exit.")
    parser.add_argument(
        "--print-cache-paths",
        action="store_true",
        help="Print expected cache file paths for archive URLs and any --cache-url values, then exit.",
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
