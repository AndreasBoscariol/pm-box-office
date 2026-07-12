#!/usr/bin/env python3
"""Ingest Todd M. Thatcher box office prediction posts into PostgreSQL."""

from __future__ import annotations

import argparse
import datetime as dt
import email.utils
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
from pm_box_office.domain import movies as movie_identity
from pm_box_office.sources.common.cli import parse_date_arg
from pm_box_office.sources.common.parsing import clean_text
from pm_box_office.sources.common.schema import acquire_schema_init_lock


BASE_URL = "https://toddmthatcher.com"
CATEGORY_URL = f"{BASE_URL}/category/box-office-predictionsresults/"
CATEGORY_RSS_URL = f"{CATEGORY_URL}feed/"
DEFAULT_CACHE_DIR = Path("data/raw/toddmthatcher")
DEFAULT_START_DATE = dt.date(2026, 6, 1)
DEFAULT_END_DATE = dt.date(2026, 6, 30)
FULL_REFRESH_START_DATE = dt.date(2012, 1, 1)
FULL_REFRESH_END_DATE = dt.date(9999, 12, 31)
DEFAULT_MAX_PAGES = 25
FULL_REFRESH_MAX_PAGES = 10_000
DEFAULT_USER_AGENT = "pm-box-office-toddmthatcher-bot/1.0 (+personal research; set --user-agent contact)"
MIN_DELAY_SECONDS = 5.0
DOMESTIC_MARKET = "US_CA"
DOMESTIC_CURRENCY = "USD"
SOURCE_KEY = "toddmthatcher"
PARSER_VERSION = "toddmthatcher_predictions_v1"
WEEKLY_SOURCE_CONTEXT = "weekly_ranked_predictions"
SINGLE_MOVIE_SOURCE_CONTEXT = "single_movie_prediction"


@dataclass(frozen=True)
class ArchiveArticle:
    article_url: str
    title: str
    author: str | None
    published_date: str | None
    source_url: str
    excerpt: str | None = None


@dataclass(frozen=True)
class ParsedArticle:
    article_url: str
    title: str
    author: str | None
    published_date: str | None
    source_url: str


@dataclass(frozen=True)
class ThatcherPrediction:
    article_url: str
    source_row_key: str
    source_movie_id: str
    source_movie_title: str
    normalized_movie_title: str
    source_rank: int | None
    market: str
    currency: str
    forecast_metric: str
    weekend_gross_prediction_usd: int
    target_start_date: str | None
    target_end_date: str | None
    prediction_made_date: str | None
    raw_forecast_text: str
    source_context: str
    parser_version: str
    row_ordinal: int
    movie_id: int | None = None
    match_status: str = "unmatched"
    match_method: str | None = None
    match_score: float | None = None
    match_notes: str | None = None


@dataclass(frozen=True)
class MovieCandidate:
    movie_id: int
    movie_url: str | None
    title: str
    release_year: int | None
    release_date: str | None
    normalized_title: str
    match_keys: frozenset[str]


@dataclass(frozen=True)
class MovieMatch:
    movie_id: int | None
    status: str
    method: str | None
    score: float | None
    notes: str | None


class FetchBlocked(RuntimeError):
    """Raised when a page cannot be fetched and no cache is available."""


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
        return cache_path_for_url(self.cache_dir, url)

    def get(self, url: str) -> tuple[str, Path, bool]:
        cache_path = self.cache_path(url)
        if cache_path.exists() and (not self.refresh or self.offline):
            return cache_path.read_text(encoding="utf-8"), cache_path, False
        if self.offline:
            raise FetchBlocked(f"Cache miss in offline mode: {url}\nExpected cached HTML at: {cache_path}")
        self._wait()
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/rss+xml,application/xml,text/xml,text/html,application/xhtml+xml",
                "User-Agent": self.user_agent,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            raise FetchBlocked(f"HTTP {exc.code} while fetching {url}") from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            raise FetchBlocked(f"GET {url} failed: {exc}") from exc
        self._last_request_at = time.monotonic()
        cache_path.write_text(body, encoding="utf-8")
        return body, cache_path, True

    def close(self) -> None:
        return None

    def _wait(self) -> None:
        delay = max(0.0, self.delay_seconds - (time.monotonic() - self._last_request_at))
        if delay:
            time.sleep(delay)


class ArchiveParser(HTMLParser):
    def __init__(self, source_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.source_url = source_url
        self.articles: list[ArchiveArticle] = []
        self._in_article = False
        self._article_depth = 0
        self._capture_title = False
        self._title_parts: list[str] = []
        self._title_url: str | None = None
        self._time: str | None = None
        self._capture_excerpt = False
        self._excerpt_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        classes = set((attrs_dict.get("class") or "").split())
        if tag == "article":
            self._in_article = True
            self._article_depth += 1
            if self._article_depth == 1:
                self._title_parts = []
                self._title_url = None
                self._time = None
                self._excerpt_parts = []
            return
        if not self._in_article:
            return
        if tag in {"h1", "h2"} and "entry-title" in classes:
            self._capture_title = True
            self._title_parts = []
        elif self._capture_title and tag == "a" and attrs_dict.get("href"):
            self._title_url = canonical_url(str(attrs_dict["href"]))
        elif tag == "time" and attrs_dict.get("datetime"):
            self._time = str(attrs_dict["datetime"])
        elif tag in {"div", "section"} and ("entry-summary" in classes or "entry-content" in classes):
            self._capture_excerpt = True
            self._excerpt_parts = []

    def handle_endtag(self, tag: str) -> None:
        if self._capture_title and tag in {"h1", "h2"}:
            self._capture_title = False
        elif self._capture_excerpt and tag in {"div", "section"}:
            self._capture_excerpt = False
        if tag == "article" and self._in_article:
            self._article_depth -= 1
            if self._article_depth == 0:
                self._finish_article()

    def handle_data(self, data: str) -> None:
        if self._capture_title:
            self._title_parts.append(data)
        if self._capture_excerpt:
            self._excerpt_parts.append(data)

    def _finish_article(self) -> None:
        title = clean_text(" ".join(self._title_parts))
        if self._title_url and is_supported_title(title):
            self.articles.append(
                ArchiveArticle(
                    article_url=self._title_url,
                    title=title,
                    author=None,
                    published_date=parse_dateish(self._time) if self._time else None,
                    source_url=self.source_url,
                    excerpt=clean_text(" ".join(self._excerpt_parts)) or None,
                )
            )
        self._in_article = False
        self._article_depth = 0


class ArticleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: str | None = None
        self.author: str | None = None
        self.published_date: str | None = None
        self.lines: list[str] = []
        self._skip_depth = 0
        self._entry_depth = 0
        self._seen_entry = False
        self._capture_title = False
        self._title_parts: list[str] = []
        self._capture_line: str | None = None
        self._line_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag in {"script", "style", "nav", "footer"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        classes = set((attrs_dict.get("class") or "").split())
        if tag in {"div", "article", "section"} and "entry-content" in classes:
            self._seen_entry = True
            self._entry_depth = 1
        elif self._entry_depth and tag in {"div", "article", "section"}:
            self._entry_depth += 1
        if tag == "meta":
            name = attrs_dict.get("property") or attrs_dict.get("name")
            content = attrs_dict.get("content")
            if name in {"og:title", "twitter:title"} and content and not self.title:
                self.title = clean_text(content)
            elif name == "article:published_time" and content and not self.published_date:
                self.published_date = parse_dateish(content)
            elif name in {"author", "article:author"} and content and not self.author:
                self.author = clean_text(content)
        elif tag == "time" and attrs_dict.get("datetime") and not self.published_date:
            self.published_date = parse_dateish(str(attrs_dict["datetime"]))
        elif tag == "h1" and ("entry-title" in classes or self.title is None):
            self._capture_title = True
            self._title_parts = []
        elif tag in {"p", "li", "h2", "h3"} and self._inside_content():
            self._capture_line = tag
            self._line_parts = []
        elif tag == "br" and self._capture_line:
            self._line_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav", "footer"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if self._capture_title and tag == "h1":
            title = clean_text(" ".join(self._title_parts))
            if title and not self.title:
                self.title = title
            self._capture_title = False
        elif self._capture_line == tag:
            text = clean_multiline_text("".join(self._line_parts))
            if text:
                self.lines.extend(text.split("\n"))
            self._capture_line = None
            self._line_parts = []
        if self._entry_depth and tag in {"div", "article", "section"}:
            self._entry_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._capture_title:
            self._title_parts.append(data)
        if self._capture_line:
            self._line_parts.append(data)

    def _inside_content(self) -> bool:
        return self._entry_depth > 0 or not self._seen_entry


def cache_path_for_url(cache_dir: Path, url: str) -> Path:
    suffix = ".xml" if is_rss_url(url) else ".html"
    return cache_dir / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}{suffix}"


def is_rss_url(url: str) -> bool:
    return urllib.parse.urlparse(url).path.rstrip("/").endswith("/feed")


def canonical_url(href: str) -> str:
    url = urllib.parse.urljoin(BASE_URL, href)
    parsed = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def archive_url(page: int) -> str:
    return CATEGORY_URL if page <= 1 else f"{CATEGORY_URL}page/{page}/"


def clean_multiline_text(value: str) -> str:
    text = value.replace("\xa0", " ").replace("\r", "\n")
    lines = [clean_text(line) for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def is_supported_title(title: str) -> bool:
    normalized = clean_text(title).lower()
    return normalized.endswith("box office prediction") or re.search(r"\bbox office predictions\b", normalized) is not None


def parse_dateish(value: str) -> str | None:
    text = clean_text(re.sub(r"(\d+)(st|nd|rd|th)", r"\1", value, flags=re.IGNORECASE))
    if not text:
        return None
    try:
        return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        pass
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


def parse_month_day(month: str, day: int, year: int) -> dt.date | None:
    for fmt in ("%B", "%b"):
        try:
            month_number = dt.datetime.strptime(month[:3] if fmt == "%b" else month, fmt).month
            return dt.date(year, month_number, day)
        except ValueError:
            continue
    return None


def extract_date_range(text: str, fallback_date: str | None = None) -> tuple[str | None, str | None]:
    year = int(fallback_date[:4]) if fallback_date else None
    explicit_year = re.search(r"\b(20\d{2})\b", text)
    if explicit_year:
        year = int(explicit_year.group(1))
    if year is None:
        return None, None
    cleaned = clean_text(text).replace("\u2013", "-").replace("\u2014", "-")
    match = re.search(
        rf"{MONTH_PATTERN}\s+(\d{{1,2}})\s*(?:-|to)\s*(?:(?:{MONTH_PATTERN})\s+)?(\d{{1,2}})",
        cleaned,
        flags=re.IGNORECASE,
    )
    if match:
        start = parse_month_day(match.group(1), int(match.group(2)), year)
        end = parse_month_day(match.group(3) or match.group(1), int(match.group(4)), year)
        if start and end:
            if end < start:
                end = dt.date(year + 1, end.month, end.day)
            return start.isoformat(), end.isoformat()
    holiday = holiday_weekend_dates(cleaned, year)
    if holiday is not None:
        return holiday[0].isoformat(), holiday[1].isoformat()
    return None, None


def holiday_weekend_dates(text: str, year: int) -> tuple[dt.date, dt.date] | None:
    lowered = text.lower()
    if "thanksgiving" in lowered:
        holiday = nth_weekday_of_month(year, 11, weekday=3, ordinal=4)
        start = holiday + dt.timedelta(days=1)
        return start, start + dt.timedelta(days=2)
    if "christmas" in lowered:
        holiday = dt.date(year, 12, 25)
        start = holiday if holiday.weekday() == 4 else next_weekday(holiday, 4)
        return start, start + dt.timedelta(days=2)
    if "fourth of july" in lowered or "4th of july" in lowered:
        holiday = dt.date(year, 7, 4)
        start = holiday if holiday.weekday() == 4 else next_weekday(holiday, 4)
        return start, start + dt.timedelta(days=2)
    if "memorial day" in lowered:
        holiday = last_weekday_of_month(year, 5, weekday=0)
        start = holiday - dt.timedelta(days=3)
        return start, start + dt.timedelta(days=3)
    if "labor day" in lowered:
        holiday = nth_weekday_of_month(year, 9, weekday=0, ordinal=1)
        start = holiday - dt.timedelta(days=3)
        return start, start + dt.timedelta(days=3)
    return None


def nth_weekday_of_month(year: int, month: int, *, weekday: int, ordinal: int) -> dt.date:
    day = dt.date(year, month, 1)
    while day.weekday() != weekday:
        day += dt.timedelta(days=1)
    return day + dt.timedelta(days=7 * (ordinal - 1))


def last_weekday_of_month(year: int, month: int, *, weekday: int) -> dt.date:
    day = dt.date(year + (month // 12), (month % 12) + 1, 1) - dt.timedelta(days=1)
    while day.weekday() != weekday:
        day -= dt.timedelta(days=1)
    return day


def next_weekday(value: dt.date, weekday: int) -> dt.date:
    return value + dt.timedelta(days=(weekday - value.weekday()) % 7)


def previous_weekday(value: dt.date, weekday: int) -> dt.date:
    return value - dt.timedelta(days=(value.weekday() - weekday) % 7)


def first_release_date(lines: list[str], fallback_date: str | None) -> str | None:
    if fallback_date is None:
        return None
    try:
        fallback_day = dt.date.fromisoformat(fallback_date)
    except ValueError:
        return None
    year = fallback_day.year
    text = "\n".join(lines[:24])
    release_verbs = (
        r"opens|out|arrives|premieres|debuts|bows|launches|"
        r"hits(?:\s+theaters)?|lands(?:\s+in\s+theaters)?|expands"
    )
    match = re.search(rf"(?:{release_verbs})[^.\n]*?\b{MONTH_PATTERN}\s+(\d{{1,2}})(?:st|nd|rd|th)?", text, re.I)
    if not match:
        match = re.search(rf"\b{MONTH_PATTERN}\s+(\d{{1,2}})(?:st|nd|rd|th)?[^.\n]*?(?:{release_verbs})", text, re.I)
    if match:
        parsed = parse_month_day(match.group(1), int(match.group(2)), year)
        if parsed and parsed < fallback_day - dt.timedelta(days=180):
            parsed = parsed.replace(year=year + 1)
        return parsed.isoformat() if parsed else None
    if re.search(rf"\b(?:{release_verbs})[^.\n]*?\bthis\s+wednesday\b", text, re.I):
        return next_weekday(fallback_day, 2).isoformat()
    if re.search(rf"\b(?:{release_verbs})[^.\n]*?\bthis\s+thursday\b", text, re.I):
        return next_weekday(fallback_day, 3).isoformat()
    if re.search(rf"\b(?:{release_verbs})[^.\n]*?\bthis\s+friday\b", text, re.I):
        return next_weekday(fallback_day, 4).isoformat()
    if re.search(rf"\b(?:{release_verbs})[^.\n]*?\bthis\s+weekend\b", text, re.I):
        return next_weekday(fallback_day, 4).isoformat()
    if re.search(r"\b(?:opens|out|arrives|premieres|debuts)[^.\n]*?\bnext\s+wednesday\b", text, re.I):
        return next_weekday(fallback_day + dt.timedelta(days=1), 2).isoformat()
    if re.search(r"\b(?:opens|out|arrives|premieres|debuts)[^.\n]*?\bnext\s+weekend\b", text, re.I):
        return next_weekday(fallback_day + dt.timedelta(days=7), 4).isoformat()
    if re.search(r"\b(?:opens|out|arrives|premieres|debuts)[^.\n]*?\bnext\s+friday\b", text, re.I):
        return next_weekday(fallback_day + dt.timedelta(days=1), 4).isoformat()
    if re.search(r"\bfirst\s+week\s+of\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b", text, re.I):
        month_name = re.search(r"\bfirst\s+week\s+of\s+([A-Za-z]+)\b", text, re.I)
        if month_name:
            first_day = parse_month_day(month_name.group(1), 1, year)
            if first_day:
                return next_weekday(first_day, 4).isoformat()
    holiday = holiday_weekend_dates(text, year)
    if holiday is not None:
        return holiday[0].isoformat()
    return None


def normalize_movie_title(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_match_keys(value: str) -> set[str]:
    normalized = normalize_movie_title(value)
    keys = {normalized} if normalized else set()
    without_year = re.sub(r"\b(?:19|20)\d{2}\b$", "", normalized).strip()
    if without_year:
        keys.add(without_year)
    without_distribution = re.sub(
        r"\b(?:wide|limited|re release|rerelease|reissue|anniversary|imax|3d|2d|edition|estimate)\b",
        " ",
        normalized,
    )
    without_distribution = re.sub(r"\b\d+(?:st|nd|rd|th)?\b(?=\s+anniversary\b)", " ", without_distribution)
    without_distribution = re.sub(r"\s+", " ", without_distribution).strip()
    if without_distribution:
        keys.add(without_distribution)
    compact = normalized.replace(" and ", " ")
    compact = re.sub(r"\s+", " ", compact).strip()
    if compact:
        keys.add(compact)
    shorthand = normalized
    shorthand = re.sub(r"\bbreaking dawn 2\b", "breaking dawn part 2", shorthand)
    shorthand = re.sub(r"\bfive night at freddy s\b", "five nights at freddy s", shorthand)
    shorthand = re.sub(r"\bhow to train you dragon\b", "how to train your dragon", shorthand)
    shorthand = re.sub(r"\ba good to die hard\b", "a good day to die hard", shorthand)
    if shorthand:
        keys.add(shorthand)
    return {key for key in keys if key}


def is_rerelease_title(value: str) -> bool:
    normalized = normalize_movie_title(value)
    return bool(
        re.search(
            r"\b(?:re release|rerelease|reissue|anniversary|3d|imax|special edition)\b",
            normalized,
        )
    )


def source_movie_id(
    *,
    normalized_movie_title: str,
    target_start_date: str | None,
    forecast_metric: str | None = None,
) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", normalized_movie_title).strip("-") or "unknown-title"
    if forecast_metric == "domestic_weekend":
        return f"{SOURCE_KEY}:{DOMESTIC_MARKET}:{slug}"
    return f"{SOURCE_KEY}:{DOMESTIC_MARKET}:{slug}:{target_start_date or 'unknown-date'}"


def money_to_usd(number: str, unit: str | None) -> int | None:
    try:
        amount = float(number.replace(",", ""))
    except ValueError:
        return None
    multiplier = {
        "k": 1_000,
        "m": 1_000_000,
        "million": 1_000_000,
        "b": 1_000_000_000,
        "billion": 1_000_000_000,
    }
    if unit:
        scale = multiplier.get(unit.lower())
        return int(round(amount * scale)) if scale is not None else None
    return int(round(amount * 1_000_000)) if amount < 1_000 else int(round(amount))


def parse_money_value(value: str) -> int | None:
    match = re.search(r"\$\s*(\d[\d,]*(?:\.\d+)?)\s*(million|billion|[kmb])?", value, re.I)
    if not match:
        return None
    return money_to_usd(match.group(1), match.group(2))


def opening_weekend_dates(release_date: str | None) -> tuple[str | None, str | None]:
    if release_date is None:
        return None, None
    start = dt.date.fromisoformat(release_date)
    return start.isoformat(), (start + dt.timedelta(days=2)).isoformat()


def parse_article(
    html: str,
    *,
    article_url: str,
    fallback: ArchiveArticle | None = None,
) -> tuple[ParsedArticle, list[ThatcherPrediction]]:
    parser = ArticleParser()
    parser.feed(html)
    title = parser.title or (fallback.title if fallback else "")
    published_date = parser.published_date or (fallback.published_date if fallback else None)
    article = ParsedArticle(
        article_url=article_url,
        title=title,
        author=parser.author or (fallback.author if fallback else None),
        published_date=published_date,
        source_url=fallback.source_url if fallback else article_url,
    )
    predictions = parse_weekly_predictions(
        parser.lines,
        article_url=article_url,
        article_title=title,
        published_date=published_date,
        first_row_ordinal=1,
    )
    predictions.extend(
        parse_single_movie_predictions(
            parser.lines,
            article_url=article_url,
            article_title=title,
            published_date=published_date,
            first_row_ordinal=len(predictions) + 1,
        )
    )
    return article, dedupe_predictions(predictions)


def parse_weekly_predictions(
    lines: list[str],
    *,
    article_url: str,
    article_title: str,
    published_date: str | None,
    first_row_ordinal: int,
) -> list[ThatcherPrediction]:
    if re.search(r"\bbox office predictions\b", article_title, re.I) is None:
        return []
    target_start, target_end = extract_date_range(article_title, published_date)
    predictions: list[ThatcherPrediction] = []
    pending_rank: int | None = None
    pending_title: str | None = None
    for line in lines:
        if re.search(r"box office results", line, re.I):
            break
        rank_match = re.match(r"^(\d+)\.\s+(.+)$", line)
        if rank_match:
            pending_rank = int(rank_match.group(1))
            pending_title = clean_text(rank_match.group(2))
            continue
        if pending_title and re.search(r"predicted gross\s*:", line, re.I):
            amount = parse_money_value(line)
            if amount is not None:
                row_ordinal = first_row_ordinal + len(predictions)
                predictions.append(
                    build_prediction(
                        article_url=article_url,
                        title=pending_title,
                        rank=pending_rank,
                        amount=amount,
                        target_start_date=target_start,
                        target_end_date=target_end,
                        prediction_made_date=published_date,
                        raw_forecast_text=f"{pending_rank}. {pending_title}\n{line}" if pending_rank else f"{pending_title}\n{line}",
                        source_context=WEEKLY_SOURCE_CONTEXT,
                        row_ordinal=row_ordinal,
                        forecast_metric="domestic_weekend",
                    )
                )
            pending_rank = None
            pending_title = None
    return predictions


def parse_single_movie_predictions(
    lines: list[str],
    *,
    article_url: str,
    article_title: str,
    published_date: str | None,
    first_row_ordinal: int,
) -> list[ThatcherPrediction]:
    predictions: list[ThatcherPrediction] = []
    default_title = re.sub(r"\s+Box Office\s+Prediction\s*$", "", article_title, flags=re.I).strip()
    target_start, target_end = opening_weekend_dates(first_release_date(lines, published_date))
    for line in lines:
        if re.search(r"opening weekend prediction\s*:", line, re.I) is None:
            continue
        amount = parse_money_value(line)
        if amount is None:
            continue
        title_match = re.match(r"^(.+?)\s+opening weekend prediction\s*:", line, re.I)
        title = clean_text(title_match.group(1)) if title_match else default_title
        row_ordinal = first_row_ordinal + len(predictions)
        predictions.append(
            build_prediction(
                article_url=article_url,
                title=title or default_title,
                rank=None,
                amount=amount,
                target_start_date=target_start,
                target_end_date=target_end,
                prediction_made_date=published_date,
                raw_forecast_text=line,
                source_context=SINGLE_MOVIE_SOURCE_CONTEXT,
                row_ordinal=row_ordinal,
                forecast_metric="domestic_opening_weekend",
            )
        )
    return predictions


def build_prediction(
    *,
    article_url: str,
    title: str,
    rank: int | None,
    amount: int,
    target_start_date: str | None,
    target_end_date: str | None,
    prediction_made_date: str | None,
    raw_forecast_text: str,
    source_context: str,
    row_ordinal: int,
    forecast_metric: str,
) -> ThatcherPrediction:
    normalized = normalize_movie_title(title)
    key_material = "|".join(
        [
            article_url,
            str(row_ordinal),
            normalized,
            forecast_metric,
            str(amount),
            str(target_start_date),
            str(target_end_date),
            PARSER_VERSION,
        ]
    )
    return ThatcherPrediction(
        article_url=article_url,
        source_row_key=hashlib.sha256(key_material.encode("utf-8")).hexdigest(),
        source_movie_id=source_movie_id(
            normalized_movie_title=normalized,
            target_start_date=target_start_date,
            forecast_metric=forecast_metric,
        ),
        source_movie_title=title,
        normalized_movie_title=normalized,
        source_rank=rank,
        market=DOMESTIC_MARKET,
        currency=DOMESTIC_CURRENCY,
        forecast_metric=forecast_metric,
        weekend_gross_prediction_usd=amount,
        target_start_date=target_start_date,
        target_end_date=target_end_date,
        prediction_made_date=prediction_made_date,
        raw_forecast_text=raw_forecast_text,
        source_context=source_context,
        parser_version=PARSER_VERSION,
        row_ordinal=row_ordinal,
    )


def dedupe_predictions(predictions: list[ThatcherPrediction]) -> list[ThatcherPrediction]:
    deduped: list[ThatcherPrediction] = []
    seen: set[tuple[object, ...]] = set()
    for prediction in predictions:
        key = (
            prediction.normalized_movie_title,
            prediction.forecast_metric,
            prediction.weekend_gross_prediction_usd,
            prediction.target_start_date,
            prediction.target_end_date,
        )
        if key in seen:
            continue
        deduped.append(prediction)
        seen.add(key)
    return deduped


RSS_NAMESPACES = {"dc": "http://purl.org/dc/elements/1.1/"}


def strip_html(value: str) -> str:
    parser = ArticleParser()
    parser.feed(f"<div class='entry-content'><p>{value}</p></div>")
    return clean_text(" ".join(parser.lines))


def parse_rss(xml_text: str, *, source_url: str) -> list[ArchiveArticle]:
    root = ET.fromstring(xml_text.strip())
    articles: list[ArchiveArticle] = []
    for item in root.findall("./channel/item"):
        title = clean_text(item.findtext("title") or "")
        link = clean_text(item.findtext("link") or "")
        if not title or not link or not is_supported_title(title):
            continue
        author = item.findtext("dc:creator", namespaces=RSS_NAMESPACES)
        articles.append(
            ArchiveArticle(
                article_url=canonical_url(link),
                title=title,
                author=clean_text(author) if author else None,
                published_date=parse_rss_date(item.findtext("pubDate") or ""),
                source_url=source_url,
                excerpt=strip_html(item.findtext("description") or "") or None,
            )
        )
    return dedupe_articles(articles)


def parse_rss_date(value: str) -> str | None:
    try:
        return email.utils.parsedate_to_datetime(clean_text(value)).date().isoformat()
    except (TypeError, ValueError):
        return parse_dateish(value)


def parse_archive(html: str, *, source_url: str) -> list[ArchiveArticle]:
    parser = ArchiveParser(source_url)
    parser.feed(html)
    return dedupe_articles(parser.articles)


def dedupe_articles(articles: list[ArchiveArticle]) -> list[ArchiveArticle]:
    by_url: dict[str, ArchiveArticle] = {}
    for article in articles:
        by_url.setdefault(article.article_url, article)
    return list(by_url.values())


def initialize_database(conn: Any) -> None:
    acquire_schema_init_lock(conn)
    movie_identity.ensure_movie_identity_schema(conn)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS toddmthatcher_articles (
            article_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            article_url TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            author TEXT,
            discovered_date DATE,
            source_url TEXT NOT NULL,
            status TEXT NOT NULL,
            fetched_at TEXT,
            raw_cache_path TEXT,
            sha256 TEXT,
            parser_version TEXT,
            created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP::text),
            updated_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP::text)
        );

        CREATE TABLE IF NOT EXISTS toddmthatcher_weekend_predictions (
            prediction_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            article_id BIGINT NOT NULL REFERENCES toddmthatcher_articles(article_id),
            source_row_key TEXT NOT NULL,
            source_movie_id TEXT NOT NULL,
            source_movie_title TEXT NOT NULL,
            normalized_movie_title TEXT NOT NULL,
            source_rank INTEGER,
            market TEXT NOT NULL,
            currency TEXT NOT NULL,
            forecast_metric TEXT NOT NULL,
            weekend_gross_prediction_usd BIGINT NOT NULL,
            target_start_date DATE,
            target_end_date DATE,
            prediction_made_date DATE,
            raw_forecast_text TEXT NOT NULL,
            source_context TEXT NOT NULL,
            parser_version TEXT NOT NULL,
            row_ordinal INTEGER NOT NULL,
            movie_id BIGINT REFERENCES movies(movie_id),
            match_status TEXT NOT NULL,
            match_method TEXT,
            match_score DOUBLE PRECISION,
            match_notes TEXT,
            fetched_at TEXT NOT NULL,
            raw_cache_path TEXT NOT NULL,
            UNIQUE(article_id, source_row_key)
        );

        CREATE TABLE IF NOT EXISTS toddmthatcher_ingest_issues (
            issue_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            issue_source TEXT NOT NULL,
            issue_type TEXT NOT NULL,
            article_url TEXT,
            source_movie_title TEXT,
            details TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP::text),
            UNIQUE(issue_source, issue_type, article_url, source_movie_title, details)
        );

        CREATE INDEX IF NOT EXISTS idx_toddmthatcher_articles_status
            ON toddmthatcher_articles(status);
        CREATE INDEX IF NOT EXISTS idx_toddmthatcher_articles_discovered_date
            ON toddmthatcher_articles(discovered_date);
        CREATE INDEX IF NOT EXISTS idx_toddmthatcher_predictions_movie_id
            ON toddmthatcher_weekend_predictions(movie_id);
        CREATE INDEX IF NOT EXISTS idx_toddmthatcher_predictions_title
            ON toddmthatcher_weekend_predictions(normalized_movie_title);
        """
    )


def upsert_article(
    conn: Any,
    article: ArchiveArticle | ParsedArticle,
    *,
    status: str,
    fetched_at: str | None = None,
    raw_cache_path: Path | None = None,
    html: str | None = None,
) -> int:
    sha256 = hashlib.sha256(html.encode("utf-8")).hexdigest() if html is not None else None
    conn.execute(
        """
        INSERT INTO toddmthatcher_articles (
            article_url, title, author, discovered_date, source_url, status,
            fetched_at, raw_cache_path, sha256, parser_version, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT(article_url) DO UPDATE SET
            title = excluded.title,
            author = excluded.author,
            discovered_date = excluded.discovered_date,
            source_url = excluded.source_url,
            status = excluded.status,
            fetched_at = COALESCE(excluded.fetched_at, toddmthatcher_articles.fetched_at),
            raw_cache_path = COALESCE(excluded.raw_cache_path, toddmthatcher_articles.raw_cache_path),
            sha256 = COALESCE(excluded.sha256, toddmthatcher_articles.sha256),
            parser_version = excluded.parser_version,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            article.article_url,
            article.title,
            article.author,
            article.published_date,
            article.source_url,
            status,
            fetched_at,
            str(raw_cache_path) if raw_cache_path is not None else None,
            sha256,
            PARSER_VERSION,
        ),
    )
    row = conn.execute("SELECT article_id FROM toddmthatcher_articles WHERE article_url = %s", (article.article_url,)).fetchone()
    return int(row[0])


def insert_predictions(
    conn: Any,
    article_id: int,
    predictions: list[ThatcherPrediction],
    *,
    fetched_at: str,
    raw_cache_path: Path,
) -> None:
    if not predictions:
        return
    conn.executemany(
        """
        INSERT INTO toddmthatcher_weekend_predictions (
            article_id, source_row_key, source_movie_id, source_movie_title, normalized_movie_title,
            source_rank, market, currency, forecast_metric, weekend_gross_prediction_usd,
            target_start_date, target_end_date, prediction_made_date, raw_forecast_text,
            source_context, parser_version, row_ordinal, movie_id, match_status, match_method,
            match_score, match_notes, fetched_at, raw_cache_path
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(article_id, source_row_key) DO UPDATE SET
            source_movie_id = excluded.source_movie_id,
            source_movie_title = excluded.source_movie_title,
            normalized_movie_title = excluded.normalized_movie_title,
            source_rank = excluded.source_rank,
            market = excluded.market,
            currency = excluded.currency,
            forecast_metric = excluded.forecast_metric,
            weekend_gross_prediction_usd = excluded.weekend_gross_prediction_usd,
            target_start_date = excluded.target_start_date,
            target_end_date = excluded.target_end_date,
            prediction_made_date = excluded.prediction_made_date,
            raw_forecast_text = excluded.raw_forecast_text,
            source_context = excluded.source_context,
            parser_version = excluded.parser_version,
            row_ordinal = excluded.row_ordinal,
            movie_id = excluded.movie_id,
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
                prediction.source_rank,
                prediction.market,
                prediction.currency,
                prediction.forecast_metric,
                prediction.weekend_gross_prediction_usd,
                prediction.target_start_date,
                prediction.target_end_date,
                prediction.prediction_made_date,
                prediction.raw_forecast_text,
                prediction.source_context,
                prediction.parser_version,
                prediction.row_ordinal,
                prediction.movie_id,
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
        INSERT INTO toddmthatcher_ingest_issues (
            issue_source, issue_type, article_url, source_movie_title, details
        ) VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """,
        (issue_source, issue_type, article_url, source_movie_title, details),
    )


def clear_article_issues(conn: Any, *, issue_source: str, article_url: str) -> None:
    conn.execute(
        "DELETE FROM toddmthatcher_ingest_issues WHERE issue_source = %s AND article_url = %s",
        (issue_source, article_url),
    )


def relation_exists(conn: Any, relation_name: str) -> bool:
    row = conn.execute("SELECT to_regclass(%s)", (relation_name,)).fetchone()
    return bool(row and row[0])


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
            match_keys=frozenset(title_match_keys(str(row[2]))),
        )
        for row in rows
    ]


def match_predictions(conn: Any, predictions: list[ThatcherPrediction]) -> list[ThatcherPrediction]:
    candidates = load_movie_candidates(conn)
    matched: list[ThatcherPrediction] = []
    for prediction in predictions:
        match = match_prediction(conn, prediction, candidates)
        if match.movie_id is not None:
            movie_identity.upsert_movie_source_id(
                conn,
                movie_id=match.movie_id,
                source=SOURCE_KEY,
                source_movie_id=prediction.source_movie_id,
                source_title=prediction.source_movie_title,
                match_status=match.status,
                match_method=match.method,
                match_score=match.score,
            )
        matched.append(
            replace(
                prediction,
                movie_id=match.movie_id,
                match_status=match.status,
                match_method=match.method,
                match_score=match.score,
                match_notes=match.notes,
            )
        )
    return matched


def match_prediction(conn: Any, prediction: ThatcherPrediction, candidates: list[MovieCandidate]) -> MovieMatch:
    if is_rerelease_title(prediction.source_movie_title):
        return MovieMatch(None, "ignored_rerelease", "rerelease_title_filter", 0.0, "Ignored re-release or anniversary title")
    if prediction.forecast_metric == "domestic_weekend" and prediction.target_start_date:
        weekly_match = find_weekly_forecast_match(conn, prediction)
        if weekly_match is not None:
            return weekly_match
    matches = title_candidates(prediction, candidates)
    if prediction.forecast_metric == "domestic_weekend" and prediction.target_start_date:
        weekly_release_match = find_weekly_release_window_match(prediction, matches)
        if weekly_release_match is not None:
            return weekly_release_match
    if prediction.forecast_metric == "domestic_opening_weekend" and not prediction.target_start_date:
        single_release_match = find_single_future_release_match(prediction, matches)
        if single_release_match is not None:
            return single_release_match
    source_id_match = find_source_id_match(conn, prediction)
    if source_id_match is not None:
        return source_id_match
    if prediction.target_start_date:
        exact = [candidate for candidate in matches if candidate.release_date == prediction.target_start_date]
        if exact:
            candidate = choose_canonical_candidate(exact)
            if candidate is None:
                return MovieMatch(None, "ambiguous", "normalized_exact_release_date", 0.5, "Multiple canonical candidates share the release date")
            return MovieMatch(candidate.movie_id, "matched" if candidate.movie_url else "provisional", "normalized_exact_release_date", 1.0, None)
    candidate = choose_canonical_candidate(matches)
    if candidate is not None:
        return MovieMatch(candidate.movie_id, "matched" if candidate.movie_url else "provisional", "normalized_exact", 1.0, None)
    if not matches and prediction.forecast_metric == "domestic_opening_weekend" and prediction.target_start_date:
        return provision_movie(conn, prediction)
    if not matches:
        return MovieMatch(None, "unmatched", "normalized_exact", 0.0, "No movie title matched")
    return MovieMatch(None, "ambiguous", "normalized_exact", 0.5, "Multiple movies share the title")


def title_candidates(prediction: ThatcherPrediction, candidates: list[MovieCandidate]) -> list[MovieCandidate]:
    keys = title_match_keys(prediction.source_movie_title)
    keys.add(prediction.normalized_movie_title)
    matches = [
        candidate
        for candidate in candidates
        if candidate.normalized_title in keys or candidate.match_keys.intersection(keys)
    ]
    by_id: dict[int, MovieCandidate] = {}
    for candidate in matches:
        by_id.setdefault(candidate.movie_id, candidate)
    return list(by_id.values())


def candidate_release_date(candidate: MovieCandidate) -> dt.date | None:
    if candidate.release_date is None:
        return None
    try:
        return dt.date.fromisoformat(candidate.release_date)
    except ValueError:
        return None


def choose_canonical_candidate(candidates: list[MovieCandidate]) -> MovieCandidate | None:
    if not candidates:
        return None
    dated = [candidate for candidate in candidates if candidate.release_date is not None]
    pool = dated or candidates
    release_dates = {candidate.release_date for candidate in pool}
    if len(release_dates) > 1:
        return None
    canonical = [candidate for candidate in pool if candidate.movie_url is not None]
    if len(canonical) == 1:
        return canonical[0]
    if len(pool) == 1:
        return pool[0]
    return None


def find_weekly_release_window_match(prediction: ThatcherPrediction, matches: list[MovieCandidate]) -> MovieMatch | None:
    target_start = dt.date.fromisoformat(prediction.target_start_date)
    target_end = dt.date.fromisoformat(prediction.target_end_date or prediction.target_start_date)
    scored: list[tuple[int, dt.date, MovieCandidate]] = []
    for candidate in matches:
        release_date = candidate_release_date(candidate)
        if release_date is None:
            continue
        if not target_start - dt.timedelta(days=120) <= release_date <= target_end:
            continue
        if release_date <= target_start:
            score = 3
        elif release_date <= target_end:
            score = 2
        else:
            score = 1
        scored.append((score, release_date, candidate))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1], item[2].movie_url is not None, -item[2].movie_id), reverse=True)
    best_score, best_release_date, best_candidate = scored[0]
    ties = [
        candidate
        for score, release_date, candidate in scored
        if score == best_score and release_date == best_release_date
    ]
    best_candidate = choose_canonical_candidate(ties)
    if best_candidate is None:
        return None
    status = "matched" if best_candidate.movie_url else "provisional"
    return MovieMatch(
        best_candidate.movie_id,
        status,
        "weekly_forecast_title_release_window",
        0.9 if best_candidate.movie_url else 0.8,
        f"Matched weekly forecast to release date {best_release_date.isoformat()} before target weekend",
    )


def find_single_future_release_match(prediction: ThatcherPrediction, matches: list[MovieCandidate]) -> MovieMatch | None:
    if prediction.prediction_made_date is None:
        return None
    try:
        prediction_date = dt.date.fromisoformat(prediction.prediction_made_date)
    except ValueError:
        return None
    scored: list[tuple[int, dt.date, MovieCandidate]] = []
    for candidate in matches:
        release_date = candidate_release_date(candidate)
        if release_date is None:
            continue
        if not prediction_date - dt.timedelta(days=14) <= release_date <= prediction_date + dt.timedelta(days=90):
            continue
        score = 2 if release_date >= prediction_date else 1
        scored.append((score, release_date, candidate))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1], item[2].movie_url is None, item[2].movie_id))
    best_score, best_release_date, best_candidate = scored[0]
    ties = [
        candidate
        for score, release_date, candidate in scored
        if score == best_score and release_date == best_release_date
    ]
    best_candidate = choose_canonical_candidate(ties)
    if best_candidate is None:
        return None
    status = "matched" if best_candidate.movie_url else "provisional"
    return MovieMatch(
        best_candidate.movie_id,
        status,
        "single_movie_title_future_release",
        0.9 if best_candidate.movie_url else 0.8,
        f"Matched single-movie prediction to release date {best_release_date.isoformat()} near prediction date",
    )


def find_weekly_forecast_match(conn: Any, prediction: ThatcherPrediction) -> MovieMatch | None:
    if not relation_exists(conn, "release_runs") or not relation_exists(conn, "daily_box_office"):
        return None
    rows = conn.execute(
        """
        WITH candidates AS (
            SELECT
                m.movie_id,
                m.movie_url,
                MIN(dbo.box_office_date::date) AS first_daily_date,
                MAX(dbo.box_office_date::date) AS last_daily_date,
                COUNT(*) AS daily_rows
            FROM movies m
            JOIN release_runs rr ON rr.movie_id = m.movie_id
            JOIN daily_box_office dbo ON dbo.release_run_id = rr.release_run_id
            WHERE regexp_replace(
                    regexp_replace(
                        replace(
                            lower(regexp_replace(COALESCE(m.title, ''), '\\s*\\(\\d{4}\\)\\s*$', '', 'g')),
                            '&',
                            ' and '
                        ),
                        '[^a-z0-9]+',
                        ' ',
                        'g'
                    ),
                    '\\s+',
                    ' ',
                    'g'
                  ) = %s
              AND dbo.box_office_date::date <= %s::date + 6
            GROUP BY m.movie_id, m.movie_url
        ),
        scored AS (
            SELECT *,
                   CASE
                     WHEN %s::date BETWEEN first_daily_date AND last_daily_date + 21 THEN 3
                     WHEN %s::date >= first_daily_date AND %s::date <= first_daily_date + 120 THEN 2
                     WHEN %s::date >= first_daily_date - 7 AND %s::date <= first_daily_date + 14 THEN 1
                     ELSE 0
                   END AS activity_score
            FROM candidates
        )
        SELECT movie_id, movie_url, first_daily_date, last_daily_date, daily_rows, activity_score
        FROM scored
        WHERE activity_score > 0
        ORDER BY activity_score DESC, (movie_url IS NOT NULL) DESC, daily_rows DESC, first_daily_date DESC, movie_id
        LIMIT 2
        """,
        (
            prediction.normalized_movie_title,
            prediction.target_start_date,
            prediction.target_start_date,
            prediction.target_start_date,
            prediction.target_start_date,
            prediction.target_start_date,
            prediction.target_start_date,
        ),
    ).fetchall()
    if not rows:
        return None
    best = rows[0]
    if len(rows) > 1 and rows[1][5] == best[5] and rows[1][0] != best[0]:
        return None
    status = "matched" if best[1] is not None else "provisional"
    return MovieMatch(
        int(best[0]),
        status,
        "weekly_forecast_title_activity_window",
        0.95 if best[1] is not None else 0.85,
        f"Matched weekly forecast to active release window {best[2]} through {best[3]}",
    )


def find_source_id_match(conn: Any, prediction: ThatcherPrediction) -> MovieMatch | None:
    if not relation_exists(conn, "movie_source_ids"):
        return None
    row = conn.execute(
        """
        SELECT m.movie_id, m.movie_url, src.match_status, src.match_score
        FROM movie_source_ids src
        JOIN movies m ON m.movie_id = src.movie_id
        WHERE src.source = %s
          AND src.source_movie_id = %s
        LIMIT 1
        """,
        (SOURCE_KEY, prediction.source_movie_id),
    ).fetchone()
    if row is None:
        return None
    status = str(row[2]) if row[2] in {"matched", "provisional"} else ("matched" if row[1] else "provisional")
    return MovieMatch(int(row[0]), status, "toddmthatcher_source_id", float(row[3]) if row[3] is not None else 1.0, None)


def provision_movie(conn: Any, prediction: ThatcherPrediction) -> MovieMatch:
    row = conn.execute(
        """
        INSERT INTO movies (title, release_year, release_date, updated_at)
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        RETURNING movie_id
        """,
        (
            prediction.source_movie_title,
            int(prediction.target_start_date[:4]) if prediction.target_start_date else None,
            prediction.target_start_date,
        ),
    ).fetchone()
    return MovieMatch(int(row[0]), "provisional", "provisional_toddmthatcher_identity", 1.0, "Created provisional movie")


def article_already_parsed(conn: Any, article_url: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM toddmthatcher_articles WHERE article_url = %s AND status = 'parsed' LIMIT 1",
        (article_url,),
    ).fetchone()
    return row is not None


def import_article(conn: Any, archive_article: ArchiveArticle, fetcher: HtmlFetcher, *, issue_source: str) -> tuple[int, int]:
    fetched_at = dt.datetime.now(dt.UTC).isoformat()
    try:
        html, cache_path, _fetched = fetcher.get(archive_article.article_url)
        article, predictions = parse_article(html, article_url=archive_article.article_url, fallback=archive_article)
    except FetchBlocked as exc:
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
            issue_source=issue_source,
            issue_type="article_page_unavailable",
            article_url=archive_article.article_url,
            source_movie_title=None,
            details=str(exc),
        )
        conn.commit()
        return 1, 0
    article_id = upsert_article(conn, article, status="parsed", fetched_at=fetched_at, raw_cache_path=cache_path, html=html)
    predictions = match_predictions(conn, predictions)
    conn.execute("DELETE FROM toddmthatcher_weekend_predictions WHERE article_id = %s", (article_id,))
    clear_article_issues(conn, issue_source=issue_source, article_url=article.article_url)
    if not predictions:
        insert_issue(
            conn,
            issue_source=issue_source,
            issue_type="no_predictions_parsed",
            article_url=article.article_url,
            source_movie_title=None,
            details=f"No predictions parsed from {article.title}",
        )
    insert_predictions(conn, article_id, predictions, fetched_at=fetched_at, raw_cache_path=cache_path)
    conn.commit()
    return 1, len(predictions)


def discover_articles(fetcher: HtmlFetcher, args: argparse.Namespace) -> list[ArchiveArticle]:
    if args.discovery == "rss":
        return limit_articles(filter_articles(fetch_rss_articles(fetcher), args), args)
    if args.discovery == "archive":
        return discover_archive_articles(fetcher, args)
    rss_articles = filter_articles(fetch_rss_articles(fetcher), args)
    archive_articles = discover_archive_articles(fetcher, args) if args.start_date < oldest_published_date(rss_articles, DEFAULT_START_DATE) else []
    return limit_articles(dedupe_articles([*rss_articles, *archive_articles]), args)


def fetch_rss_articles(fetcher: HtmlFetcher) -> list[ArchiveArticle]:
    print(f"Reading RSS feed {CATEGORY_RSS_URL}", file=sys.stderr)
    xml_text, _cache_path, _fetched = fetcher.get(CATEGORY_RSS_URL)
    return parse_rss(xml_text, source_url=CATEGORY_RSS_URL)


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
        page_all_older = True
        for article in page_articles:
            if article.published_date is None:
                page_all_older = False
                articles_by_url.setdefault(article.article_url, article)
                continue
            published = dt.date.fromisoformat(article.published_date)
            if published >= args.start_date:
                page_all_older = False
            if args.start_date <= published <= args.end_date:
                articles_by_url.setdefault(article.article_url, article)
        if page_all_older:
            break
    return limit_articles(sorted(articles_by_url.values(), key=lambda article: (article.published_date or "", article.article_url)), args)


def filter_articles(articles: list[ArchiveArticle], args: argparse.Namespace) -> list[ArchiveArticle]:
    filtered = []
    for article in articles:
        if article.published_date is None:
            filtered.append(article)
            continue
        published = dt.date.fromisoformat(article.published_date)
        if args.start_date <= published <= args.end_date:
            filtered.append(article)
    return sorted(filtered, key=lambda article: (article.published_date or "", article.article_url))


def oldest_published_date(articles: list[ArchiveArticle], default: dt.date) -> dt.date:
    dates = [dt.date.fromisoformat(article.published_date) for article in articles if article.published_date]
    return min(dates) if dates else default


def limit_articles(articles: list[ArchiveArticle], args: argparse.Namespace) -> list[ArchiveArticle]:
    return articles[: args.max_articles] if args.max_articles is not None else articles


def configure_full_refresh_args(args: argparse.Namespace) -> None:
    if not getattr(args, "full_refresh", False):
        return
    args.refresh = True
    args.discovery = "archive"
    args.start_date = FULL_REFRESH_START_DATE
    args.end_date = FULL_REFRESH_END_DATE
    if args.max_pages == DEFAULT_MAX_PAGES:
        args.max_pages = FULL_REFRESH_MAX_PAGES


def validate_args(args: argparse.Namespace) -> None:
    if args.end_date < args.start_date:
        raise SystemExit("--end-date must be on or after --start-date")
    if args.delay_seconds < MIN_DELAY_SECONDS and not args.offline and not args.dry_run:
        raise SystemExit(f"--delay-seconds must be at least {MIN_DELAY_SECONDS:g}")
    if "bot" not in args.user_agent.lower() and not args.offline and not args.dry_run:
        raise SystemExit("--user-agent must identify the scraper as a bot")


def discovery_urls_for_dry_run(args: argparse.Namespace) -> list[str]:
    archive_urls = [archive_url(page) for page in range(1, args.max_pages + 1)]
    if args.discovery == "rss":
        return [CATEGORY_RSS_URL]
    if args.discovery == "archive":
        return archive_urls
    return [CATEGORY_RSS_URL, *archive_urls]


def run(args: argparse.Namespace) -> int:
    configure_full_refresh_args(args)
    validate_args(args)
    if args.print_cache_paths:
        fetcher = HtmlFetcher(args.cache_dir, refresh=False, offline=True, delay_seconds=args.delay_seconds, user_agent=args.user_agent)
        for url in [*discovery_urls_for_dry_run(args), *args.cache_url]:
            print(f"{url}\t{fetcher.cache_path(url)}")
        return 0
    if args.dry_run:
        urls = discovery_urls_for_dry_run(args)
        for url in urls:
            print(url)
        print(f"Discovery URLs: {len(urls)}", file=sys.stderr)
        return 0
    fetcher = HtmlFetcher(args.cache_dir, refresh=args.refresh, offline=args.offline, delay_seconds=args.delay_seconds, user_agent=args.user_agent)
    conn = connect_database(args.database_url)
    try:
        initialize_database(conn)
        conn.commit()
        articles = discover_articles(fetcher, args)
        imported_articles = 0
        imported_predictions = 0
        skipped_articles = 0
        for index, archive_article in enumerate(articles, start=1):
            if not args.refresh and article_already_parsed(conn, archive_article.article_url):
                skipped_articles += 1
                print(f"Skipping parsed article {index}/{len(articles)} {archive_article.title}", file=sys.stderr)
                continue
            upsert_article(conn, archive_article, status="discovered")
            conn.commit()
            print(f"Reading article {index}/{len(articles)} {archive_article.title}", file=sys.stderr)
            article_count, prediction_count = import_article(conn, archive_article, fetcher, issue_source=args.issue_source)
            imported_articles += article_count
            imported_predictions += prediction_count
        print(
            f"Imported {imported_articles} articles and {imported_predictions} predictions; skipped {skipped_articles} parsed articles.",
            file=sys.stderr,
        )
    finally:
        fetcher.close()
        conn.close()
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import Todd M. Thatcher box office predictions.")
    parser.add_argument("--start-date", type=parse_date_arg, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=parse_date_arg, default=DEFAULT_END_DATE)
    parser.add_argument("--database-url", default=database_url_from_env(), help="PostgreSQL connection URL. Defaults to DATABASE_URL or POSTGRES_DSN.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Raw HTML cache directory.")
    parser.add_argument("--delay-seconds", type=float, default=MIN_DELAY_SECONDS, help="Delay between uncached HTTP requests.")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="HTTP User-Agent. Must identify as a bot.")
    parser.add_argument("--discovery", choices=("auto", "rss", "archive"), default="auto")
    parser.add_argument("--refresh", action="store_true", help="Reparse even when article status is parsed.")
    parser.add_argument("--full-refresh", action="store_true", help="Reparse every supported post from the full category archive.")
    parser.add_argument("--offline", action="store_true", help="Require all pages to exist in cache.")
    parser.add_argument("--dry-run", action="store_true", help="Print discovery URLs and exit.")
    parser.add_argument("--print-cache-paths", action="store_true", help="Print expected cache file paths and exit.")
    parser.add_argument("--cache-url", action="append", default=[], help="Extra article URL to include when printing cache paths.")
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES, help="Maximum archive pages to inspect.")
    parser.add_argument("--max-articles", type=int, help="Optional cap for smoke tests after discovery.")
    parser.add_argument("--issue-source", default="toddmthatcher_import", help="Label used for import issues.")
    return parser


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
