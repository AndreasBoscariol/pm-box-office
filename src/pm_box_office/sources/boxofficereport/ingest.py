#!/usr/bin/env python3
"""Ingest BoxOfficeReport.com weekend box office predictions into PostgreSQL."""

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
from pm_box_office.domain import movies as movie_identity
from pm_box_office.sources.common.cli import parse_date_arg
from pm_box_office.sources.common.parsing import clean_text
from pm_box_office.sources.common.schema import acquire_schema_init_lock


BASE_URL = "http://www.boxofficereport.com"
ARCHIVE_URL = f"{BASE_URL}/predictions/predictions.html"
DEFAULT_CACHE_DIR = Path("data/raw/boxofficereport")
DEFAULT_START_DATE = dt.date(2026, 6, 1)
DEFAULT_END_DATE = dt.date(2026, 6, 30)
FULL_REFRESH_START_DATE = dt.date(1900, 1, 1)
FULL_REFRESH_END_DATE = dt.date(9999, 12, 31)
DEFAULT_USER_AGENT = "pm-box-office-boxofficereport-bot/1.0 (+personal research; set --user-agent contact)"
MIN_DELAY_SECONDS = 5.0
DOMESTIC_MARKET = "US_CA"
DOMESTIC_CURRENCY = "USD"
PARSER_VERSION = "boxofficereport_weekend_table_v1"


@dataclass(frozen=True)
class ArchivePredictionPage:
    article_url: str
    title: str
    target_start_date: str | None
    target_end_date: str | None
    archive_top_film: str | None
    archive_top_prediction_usd: int | None
    archive_top_actual_usd: int | None
    source_url: str


@dataclass(frozen=True)
class Article:
    article_url: str
    title: str
    author: str | None
    prediction_made_at: str | None
    prediction_made_date: str | None
    target_start_date: str | None
    target_end_date: str | None
    source_url: str


@dataclass(frozen=True)
class WeekendPrediction:
    article_url: str
    source_row_key: str
    source_movie_id: str
    source_movie_title: str
    normalized_movie_title: str
    distributor: str
    source_rank: int
    market: str
    currency: str
    forecast_metric: str
    weekend_gross_prediction_usd: int
    total_gross_prediction_usd: int | None
    percent_change: float | None
    change_label: str | None
    week_number: int | None
    target_start_date: str | None
    target_end_date: str | None
    prediction_made_at: str | None
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
class TableCell:
    text: str
    links: tuple[str, ...] = ()


@dataclass(frozen=True)
class HtmlTable:
    rows: list[list[TableCell]]


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
class ParseResult:
    archive_page: ArchivePredictionPage
    article: Article | None
    predictions: list[WeekendPrediction]
    fetched_at: str
    raw_cache_path: Path
    html: str
    error: str | None = None


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
            return cache_path.read_text(encoding="utf-8-sig"), cache_path, False
        if self.offline:
            raise FetchBlocked(f"Cache miss in offline mode: {url}\nExpected cached HTML at: {cache_path}")
        self._wait()
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8-sig", errors="replace")
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
        elapsed = time.monotonic() - self._last_request_at
        delay = max(0.0, self.delay_seconds - elapsed)
        if delay:
            time.sleep(delay)


class BoxOfficeReportHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[HtmlTable] = []
        self.title: str | None = None
        self.headings: list[str] = []
        self.document_parts: list[str] = []
        self._table_depth = 0
        self._current_rows: list[list[TableCell]] = []
        self._current_row: list[TableCell] | None = None
        self._cell_parts: list[str] | None = None
        self._cell_links: list[str] = []
        self._capture_title = False
        self._title_parts: list[str] = []
        self._capture_heading: str | None = None
        self._heading_parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag in {"script", "style"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._capture_title = True
            self._title_parts = []
        elif tag in {"h1", "h2", "h3", "h4", "h5"}:
            self._capture_heading = tag
            self._heading_parts = []
        elif self._capture_heading is not None and tag == "br":
            self._heading_parts.append("\n")
        elif tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._current_rows = []
        elif self._table_depth and tag == "tr":
            self._current_row = []
        elif self._table_depth and tag in {"td", "th"}:
            self._cell_parts = []
            self._cell_links = []
        elif self._cell_parts is not None and tag == "br":
            self._cell_parts.append("\n")
        elif self._cell_parts is not None and tag == "a" and attrs_dict.get("href"):
            self._cell_links.append(absolute_url(str(attrs_dict["href"])))

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "title" and self._capture_title:
            self.title = clean_text(" ".join(self._title_parts))
            self._capture_title = False
        elif tag == self._capture_heading:
            text = clean_multiline_text("".join(self._heading_parts))
            if text:
                self.headings.append(text)
            self._capture_heading = None
            self._heading_parts = []
        elif self._table_depth and tag in {"td", "th"} and self._cell_parts is not None:
            if self._current_row is not None:
                self._current_row.append(
                    TableCell(
                        text=clean_multiline_text("".join(self._cell_parts)),
                        links=tuple(self._cell_links),
                    )
                )
            self._cell_parts = None
            self._cell_links = []
        elif self._table_depth and tag == "tr":
            if self._current_row is not None and any(cell.text or cell.links for cell in self._current_row):
                self._current_rows.append(self._current_row)
            self._current_row = None
        elif tag == "table" and self._table_depth:
            self._table_depth -= 1
            if self._table_depth == 0 and self._current_rows:
                self.tables.append(HtmlTable(rows=self._current_rows))
                self._current_rows = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self.document_parts.append(data)
        if self._capture_title:
            self._title_parts.append(data)
        if self._capture_heading is not None:
            self._heading_parts.append(data)
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    @property
    def document_text(self) -> str:
        return clean_text(" ".join(self.document_parts))


def cache_path_for_url(cache_dir: Path, url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.html"


def absolute_url(href: str) -> str:
    return urllib.parse.urljoin(BASE_URL, href)


def canonical_url(href: str) -> str:
    parsed = urllib.parse.urlsplit(absolute_url(href))
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def clean_multiline_text(value: str) -> str:
    text = value.replace("\xa0", " ").replace("\r", "\n")
    lines = [clean_text(line) for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def parse_archive(html: str, *, source_url: str = ARCHIVE_URL) -> list[ArchivePredictionPage]:
    parser = BoxOfficeReportHtmlParser()
    parser.feed(html)
    pages: list[ArchivePredictionPage] = []
    for table in parser.tables:
        rows = table.rows
        if not rows:
            continue
        headers = [normalize_header(cell.text) for cell in rows[0]]
        if headers[:4] != ["weekend", "1 film", "prediction", "actual"]:
            continue
        for row in rows[1:]:
            if len(row) < 4 or not row[0].links:
                continue
            target_start, target_end = parse_date_range(row[0].text)
            if target_start is None and target_end is None:
                continue
            pages.append(
                ArchivePredictionPage(
                    article_url=canonical_url(row[0].links[0]),
                    title=f"Weekend Box Office Predictions for {row[0].text}",
                    target_start_date=target_start,
                    target_end_date=target_end,
                    archive_top_film=clean_text(row[1].text) or None,
                    archive_top_prediction_usd=parse_money_value(row[2].text),
                    archive_top_actual_usd=parse_money_value(row[3].text),
                    source_url=source_url,
                )
            )
    deduped: dict[str, ArchivePredictionPage] = {}
    for page in pages:
        deduped.setdefault(page.article_url, page)
    return list(deduped.values())


def parse_article(
    html: str,
    *,
    article_url: str,
    fallback: ArchivePredictionPage | None = None,
) -> tuple[Article, list[WeekendPrediction]]:
    parser = BoxOfficeReportHtmlParser()
    parser.feed(html)
    title = parse_article_title(parser, fallback)
    target_start, target_end = parse_article_target_dates(parser, fallback)
    prediction_made_at, prediction_made_date = parse_published_timestamp(parser.document_text)
    article = Article(
        article_url=canonical_url(article_url),
        title=title,
        author="Daniel Garris" if "Daniel Garris" in parser.document_text else None,
        prediction_made_at=prediction_made_at,
        prediction_made_date=prediction_made_date,
        target_start_date=target_start,
        target_end_date=target_end,
        source_url=fallback.source_url if fallback else article_url,
    )
    predictions = parse_prediction_tables(
        parser.tables,
        article=article,
    )
    return article, predictions


def parse_article_title(parser: BoxOfficeReportHtmlParser, fallback: ArchivePredictionPage | None) -> str:
    for heading in parser.headings:
        if "Weekend Box Office Predictions" in heading:
            return clean_multiline_text(heading.replace("\n", " "))
    if parser.title:
        return parser.title
    return fallback.title if fallback else "Weekend Box Office Predictions"


def parse_article_target_dates(
    parser: BoxOfficeReportHtmlParser,
    fallback: ArchivePredictionPage | None,
) -> tuple[str | None, str | None]:
    for value in [*parser.headings, parser.title or ""]:
        target_start, target_end = parse_date_range(value)
        if target_start or target_end:
            return target_start, target_end
    if fallback:
        return fallback.target_start_date, fallback.target_end_date
    return None, None


def parse_prediction_tables(tables: list[HtmlTable], *, article: Article) -> list[WeekendPrediction]:
    predictions: list[WeekendPrediction] = []
    for table in tables:
        if not table.rows:
            continue
        headers = [normalize_header(cell.text) for cell in table.rows[0]]
        indexes = prediction_table_indexes(headers)
        if indexes is None:
            continue
        for row in table.rows[1:]:
            parsed = parse_prediction_row(
                row,
                indexes=indexes,
                article=article,
                row_ordinal=len(predictions) + 1,
            )
            if parsed is not None:
                predictions.append(parsed)
    return predictions


def prediction_table_indexes(headers: list[str]) -> dict[str, int] | None:
    rank_index = find_header(headers, "rank")
    film_index = find_header(headers, "film distributor")
    weekend_index = find_header(headers, "weekend gross")
    total_index = find_header(headers, "total gross")
    change_index = find_header(headers, "change")
    week_index = find_week_number_header(headers)
    if rank_index is None and film_index == 1:
        rank_index = 0
    if None in {rank_index, film_index, weekend_index}:
        return None
    assert rank_index is not None
    assert film_index is not None
    assert weekend_index is not None
    return {
        "rank": rank_index,
        "film": film_index,
        "weekend": weekend_index,
        "total": total_index if total_index is not None else -1,
        "change": change_index if change_index is not None else -1,
        "week": week_index if week_index is not None else -1,
    }


def parse_prediction_row(
    row: list[TableCell],
    *,
    indexes: dict[str, int],
    article: Article,
    row_ordinal: int,
) -> WeekendPrediction | None:
    required_indexes = [indexes["rank"], indexes["film"], indexes["weekend"]]
    if len(row) <= max(required_indexes):
        return None
    rank = parse_int(row[indexes["rank"]].text)
    title, distributor = parse_film_distributor(row[indexes["film"]].text)
    weekend_prediction = parse_money_value(row[indexes["weekend"]].text)
    if rank is None or not title or weekend_prediction is None:
        return None
    total_prediction = cell_money(row, indexes["total"])
    change_label = cell_text(row, indexes["change"])
    percent_change = parse_percent_change(change_label)
    week_number = parse_int(cell_text(row, indexes["week"]))
    normalized = normalize_movie_title(title)
    forecast_metric = "domestic_opening_weekend" if change_label and change_label.upper() == "NEW" else "domestic_weekend"
    key_material = "|".join(
        [
            article.article_url,
            str(row_ordinal),
            normalized,
            str(rank),
            str(weekend_prediction),
            str(total_prediction),
            str(article.target_start_date),
            str(article.target_end_date),
            PARSER_VERSION,
        ]
    )
    raw_text = " | ".join(cell.text for cell in row)
    return WeekendPrediction(
        article_url=article.article_url,
        source_row_key=hashlib.sha256(key_material.encode("utf-8")).hexdigest(),
        source_movie_id=boxofficereport_source_movie_id(
            normalized_movie_title=normalized,
            target_start_date=article.target_start_date,
        ),
        source_movie_title=title,
        normalized_movie_title=normalized,
        distributor=distributor or "Unknown",
        source_rank=rank,
        market=DOMESTIC_MARKET,
        currency=DOMESTIC_CURRENCY,
        forecast_metric=forecast_metric,
        weekend_gross_prediction_usd=weekend_prediction,
        total_gross_prediction_usd=total_prediction,
        percent_change=percent_change,
        change_label=change_label or None,
        week_number=week_number,
        target_start_date=article.target_start_date,
        target_end_date=article.target_end_date,
        prediction_made_at=article.prediction_made_at,
        raw_forecast_text=raw_text,
        source_context="weekend_prediction_table",
        parser_version=PARSER_VERSION,
        row_ordinal=row_ordinal,
    )


def find_header(headers: list[str], needle: str) -> int | None:
    compact_needle = needle.replace(" ", "")
    for index, header in enumerate(headers):
        compact_header = header.replace(" ", "")
        if header == needle or needle in header or compact_header == compact_needle or compact_needle in compact_header:
            return index
    return None


def find_week_number_header(headers: list[str]) -> int | None:
    for index, header in enumerate(headers):
        if header in {"week", "week 1"} or header.startswith("week "):
            return index
    return None


def cell_text(row: list[TableCell], index: int) -> str:
    if index < 0 or len(row) <= index:
        return ""
    return clean_text(row[index].text)


def cell_money(row: list[TableCell], index: int) -> int | None:
    return parse_money_value(cell_text(row, index))


def parse_film_distributor(value: str) -> tuple[str | None, str | None]:
    lines = [clean_text(line) for line in value.split("\n") if clean_text(line)]
    if not lines:
        return None, None
    if len(lines) >= 2 and re.fullmatch(r"\(.+\)", lines[-1]):
        return clean_text(" ".join(lines[:-1])), clean_text(lines[-1].strip("()"))
    match = re.match(r"^(.+?)\s*\(([^()]*)\)\s*$", clean_text(value))
    if match:
        return clean_text(match.group(1)), clean_text(match.group(2))
    return clean_text(value), None


def parse_published_timestamp(document_text: str) -> tuple[str | None, str | None]:
    match = re.search(
        r"Published\s+on\s+([A-Za-z]+\s+\d{1,2},\s+20\d{2})\s+at\s+"
        r"(\d{1,2}:\d{2}\s*[AP]M)\s+Pacific",
        document_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None, None
    date_part = parse_single_date(match.group(1))
    if date_part is None:
        return None, None
    try:
        naive = dt.datetime.strptime(
            f"{match.group(1)} {match.group(2).replace(' ', '')}",
            "%B %d, %Y %I:%M%p",
        )
    except ValueError:
        return None, date_part
    # BoxOfficeReport labels publication time as Pacific. Store a fixed ISO offset
    # rather than a naive timestamp so the source's stated timing is preserved.
    offset_hours = -7 if is_pacific_daylight_time(naive.date()) else -8
    aware = naive.replace(tzinfo=dt.timezone(dt.timedelta(hours=offset_hours)))
    return aware.isoformat(), date_part


def is_pacific_daylight_time(value: dt.date) -> bool:
    return second_sunday(value.year, 3) <= value < first_sunday(value.year, 11)


def second_sunday(year: int, month: int) -> dt.date:
    first = dt.date(year, month, 1)
    days_until_sunday = (6 - first.weekday()) % 7
    return first + dt.timedelta(days=days_until_sunday + 7)


def first_sunday(year: int, month: int) -> dt.date:
    first = dt.date(year, month, 1)
    days_until_sunday = (6 - first.weekday()) % 7
    return first + dt.timedelta(days=days_until_sunday)


MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


DATE_SEGMENT_RE = re.compile(
    r"\b("
    + "|".join(sorted(MONTHS, key=len, reverse=True))
    + r")\.?\s+(\d{1,2})(?:,\s*(20\d{2}))?\b",
    flags=re.IGNORECASE,
)


def parse_date_range(value: str) -> tuple[str | None, str | None]:
    text = clean_text(value).replace("\u2013", "-").replace("\u2014", "-")
    matches = list(DATE_SEGMENT_RE.finditer(text))
    if len(matches) < 2:
        return None, None
    first, second = matches[0], matches[1]
    first_year = int(first.group(3)) if first.group(3) else None
    second_year = int(second.group(3)) if second.group(3) else first_year
    if second_year is None:
        return None, None
    if first_year is None:
        first_year = second_year
    start = make_date(first.group(1), int(first.group(2)), first_year)
    end = make_date(second.group(1), int(second.group(2)), second_year)
    if start is None or end is None:
        return None, None
    if end < start:
        end = make_date(second.group(1), int(second.group(2)), second_year + 1)
    return start.isoformat(), end.isoformat() if end else None


def parse_single_date(value: str) -> str | None:
    match = DATE_SEGMENT_RE.search(clean_text(value))
    if not match or not match.group(3):
        return None
    parsed = make_date(match.group(1), int(match.group(2)), int(match.group(3)))
    return parsed.isoformat() if parsed else None


def make_date(month: str, day: int, year: int) -> dt.date | None:
    month_number = MONTHS.get(month.lower().rstrip("."))
    if month_number is None:
        return None
    try:
        return dt.date(year, month_number, day)
    except ValueError:
        return None


def parse_money_value(value: str) -> int | None:
    text = clean_text(value)
    if not text or text.upper() in {"N/A", "NA"}:
        return None
    match = re.search(r"\$\s*(\d[\d,]*(?:\.\d+)?)\s*([kmbKMB])?", text)
    if not match:
        return None
    number = match.group(1).replace(",", "")
    unit = (match.group(2) or "").lower()
    multiplier = {"": 1, "k": 1_000, "m": 1_000_000, "b": 1_000_000_000}.get(unit)
    if multiplier is None:
        return None
    try:
        return int(round(float(number) * multiplier))
    except ValueError:
        return None


def parse_percent_change(value: str) -> float | None:
    text = clean_text(value)
    match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%", text)
    return float(match.group(1)) if match else None


def parse_int(value: str) -> int | None:
    match = re.search(r"\d+", clean_text(value))
    return int(match.group(0)) if match else None


def normalize_header(value: str) -> str:
    text = re.sub(r"#\s*1", "1", clean_text(value).lower()).replace("%", "")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def normalize_movie_title(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"\s*\(\d{4}\)\s*$", "", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def boxofficereport_source_movie_id(
    *,
    normalized_movie_title: str,
    target_start_date: str | None,
) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", normalized_movie_title).strip("-") or "unknown-title"
    return f"boxofficereport:{DOMESTIC_MARKET}:{slug}:{target_start_date or 'unknown-date'}"


def initialize_database(conn: Any) -> None:
    acquire_schema_init_lock(conn)
    movie_identity.ensure_movie_identity_schema(conn)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS boxofficereport_articles (
            article_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            article_url TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            author TEXT,
            prediction_made_at TIMESTAMPTZ,
            prediction_made_date DATE,
            target_start_date DATE,
            target_end_date DATE,
            source_url TEXT NOT NULL,
            status TEXT NOT NULL,
            fetched_at TIMESTAMPTZ,
            raw_cache_path TEXT,
            sha256 TEXT,
            parser_version TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS boxofficereport_weekend_predictions (
            prediction_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            article_id BIGINT NOT NULL REFERENCES boxofficereport_articles(article_id),
            source_row_key TEXT NOT NULL,
            source_movie_id TEXT NOT NULL,
            source_movie_title TEXT NOT NULL,
            normalized_movie_title TEXT NOT NULL,
            distributor TEXT NOT NULL,
            source_rank INTEGER NOT NULL,
            market TEXT NOT NULL,
            currency TEXT NOT NULL,
            forecast_metric TEXT NOT NULL,
            weekend_gross_prediction_usd BIGINT NOT NULL,
            total_gross_prediction_usd BIGINT,
            percent_change DOUBLE PRECISION,
            change_label TEXT,
            week_number INTEGER,
            target_start_date DATE,
            target_end_date DATE,
            prediction_made_at TIMESTAMPTZ,
            raw_forecast_text TEXT NOT NULL,
            source_context TEXT NOT NULL,
            parser_version TEXT NOT NULL,
            row_ordinal INTEGER NOT NULL,
            movie_id BIGINT REFERENCES movies(movie_id),
            match_status TEXT NOT NULL,
            match_method TEXT,
            match_score DOUBLE PRECISION,
            match_notes TEXT,
            fetched_at TIMESTAMPTZ NOT NULL,
            raw_cache_path TEXT NOT NULL,
            UNIQUE(article_id, source_row_key)
        );

        ALTER TABLE boxofficereport_weekend_predictions
            ADD COLUMN IF NOT EXISTS movie_id BIGINT REFERENCES movies(movie_id);

        CREATE TABLE IF NOT EXISTS boxofficereport_ingest_issues (
            issue_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            issue_source TEXT NOT NULL,
            issue_type TEXT NOT NULL,
            article_url TEXT,
            source_movie_title TEXT,
            details TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(issue_source, issue_type, article_url, source_movie_title, details)
        );

        CREATE INDEX IF NOT EXISTS idx_boxofficereport_articles_prediction_made
            ON boxofficereport_articles(prediction_made_at);
        CREATE INDEX IF NOT EXISTS idx_boxofficereport_predictions_movie_id
            ON boxofficereport_weekend_predictions(movie_id);
        CREATE INDEX IF NOT EXISTS idx_boxofficereport_predictions_title
            ON boxofficereport_weekend_predictions(normalized_movie_title);
        """
    )
    movie_identity.ensure_movie_identity_schema(conn)


def upsert_article(
    conn: Any,
    article: Article | ArchivePredictionPage,
    *,
    status: str,
    fetched_at: str | None = None,
    raw_cache_path: Path | None = None,
    html: str | None = None,
) -> int:
    if isinstance(article, ArchivePredictionPage):
        article = Article(
            article_url=article.article_url,
            title=article.title,
            author=None,
            prediction_made_at=None,
            prediction_made_date=None,
            target_start_date=article.target_start_date,
            target_end_date=article.target_end_date,
            source_url=article.source_url,
        )
    sha256 = hashlib.sha256(html.encode("utf-8")).hexdigest() if html is not None else None
    conn.execute(
        """
        INSERT INTO boxofficereport_articles (
            article_url, title, author, prediction_made_at, prediction_made_date,
            target_start_date, target_end_date, source_url, status, fetched_at,
            raw_cache_path, sha256, parser_version, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT(article_url) DO UPDATE SET
            title = excluded.title,
            author = COALESCE(excluded.author, boxofficereport_articles.author),
            prediction_made_at = COALESCE(excluded.prediction_made_at, boxofficereport_articles.prediction_made_at),
            prediction_made_date = COALESCE(excluded.prediction_made_date, boxofficereport_articles.prediction_made_date),
            target_start_date = COALESCE(excluded.target_start_date, boxofficereport_articles.target_start_date),
            target_end_date = COALESCE(excluded.target_end_date, boxofficereport_articles.target_end_date),
            source_url = excluded.source_url,
            status = excluded.status,
            fetched_at = COALESCE(excluded.fetched_at, boxofficereport_articles.fetched_at),
            raw_cache_path = COALESCE(excluded.raw_cache_path, boxofficereport_articles.raw_cache_path),
            sha256 = COALESCE(excluded.sha256, boxofficereport_articles.sha256),
            parser_version = excluded.parser_version,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            article.article_url,
            article.title,
            article.author,
            article.prediction_made_at,
            article.prediction_made_date,
            article.target_start_date,
            article.target_end_date,
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
            "SELECT article_id FROM boxofficereport_articles WHERE article_url = %s",
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
        INSERT INTO boxofficereport_weekend_predictions (
            article_id, source_row_key, source_movie_id, source_movie_title,
            normalized_movie_title, distributor, source_rank, market, currency,
            forecast_metric, weekend_gross_prediction_usd, total_gross_prediction_usd,
            percent_change, change_label, week_number, target_start_date, target_end_date,
            prediction_made_at, raw_forecast_text, source_context, parser_version,
            row_ordinal, movie_id, match_status, match_method, match_score,
            match_notes, fetched_at, raw_cache_path
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(article_id, source_row_key) DO UPDATE SET
            source_movie_id = excluded.source_movie_id,
            source_movie_title = excluded.source_movie_title,
            normalized_movie_title = excluded.normalized_movie_title,
            distributor = excluded.distributor,
            source_rank = excluded.source_rank,
            market = excluded.market,
            currency = excluded.currency,
            forecast_metric = excluded.forecast_metric,
            weekend_gross_prediction_usd = excluded.weekend_gross_prediction_usd,
            total_gross_prediction_usd = excluded.total_gross_prediction_usd,
            percent_change = excluded.percent_change,
            change_label = excluded.change_label,
            week_number = excluded.week_number,
            target_start_date = excluded.target_start_date,
            target_end_date = excluded.target_end_date,
            prediction_made_at = excluded.prediction_made_at,
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
                prediction.distributor,
                prediction.source_rank,
                prediction.market,
                prediction.currency,
                prediction.forecast_metric,
                prediction.weekend_gross_prediction_usd,
                prediction.total_gross_prediction_usd,
                prediction.percent_change,
                prediction.change_label,
                prediction.week_number,
                prediction.target_start_date,
                prediction.target_end_date,
                prediction.prediction_made_at,
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
        INSERT INTO boxofficereport_ingest_issues (
            issue_source, issue_type, article_url, source_movie_title, details
        ) VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """,
        (issue_source, issue_type, article_url, source_movie_title, details),
    )


def clear_article_issues(conn: Any, *, issue_source: str, article_url: str) -> None:
    conn.execute(
        """
        DELETE FROM boxofficereport_ingest_issues
        WHERE issue_source = %s
          AND article_url = %s
        """,
        (issue_source, article_url),
    )


def article_already_parsed(conn: Any, article_url: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM boxofficereport_articles
        WHERE article_url = %s
          AND status = 'parsed'
        LIMIT 1
        """,
        (article_url,),
    ).fetchone()
    return row is not None


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
            upsert_movie_source_id(
                conn,
                movie_id=match.movie_id,
                prediction=prediction,
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


def match_prediction(
    conn: Any,
    prediction: WeekendPrediction,
    candidates: list[MovieCandidate],
) -> MovieMatch:
    source_id_match = find_source_id_match(conn, prediction)
    if source_id_match is not None:
        return source_id_match
    matches = [candidate for candidate in candidates if candidate.normalized_title == prediction.normalized_movie_title]
    if not matches:
        if can_provision_movie(prediction):
            return provision_movie(conn, prediction)
        return MovieMatch(None, "unmatched", "normalized_exact", 0.0, "No movie title matched")
    target_date = prediction.target_start_date if prediction.forecast_metric == "domestic_opening_weekend" else None
    if target_date is not None:
        exact = [candidate for candidate in matches if candidate.release_date == target_date]
        if exact:
            candidate = preferred_movie_candidate(exact)
            status = "matched" if candidate.movie_url is not None else "provisional"
            return MovieMatch(candidate.movie_id, status, "normalized_exact_release_date", 1.0, None)
    if len(matches) == 1:
        candidate = matches[0]
        status = "matched" if candidate.movie_url is not None else "provisional"
        return MovieMatch(candidate.movie_id, status, "normalized_exact", 1.0, None)
    return MovieMatch(None, "ambiguous", "normalized_exact", 0.5, "Multiple movies share the title")


def find_source_id_match(conn: Any, prediction: WeekendPrediction) -> MovieMatch | None:
    if not relation_exists(conn, "movie_source_ids"):
        return None
    row = conn.execute(
        """
        SELECT m.movie_id, m.movie_url, src.match_status, src.match_score
        FROM movie_source_ids src
        JOIN movies m ON m.movie_id = src.movie_id
        WHERE src.source = 'boxofficereport'
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
        "boxofficereport_source_id",
        float(row[3]) if row[3] is not None else 1.0,
        f"Matched existing Box Office Report source id {prediction.source_movie_id}",
    )


def can_provision_movie(prediction: WeekendPrediction) -> bool:
    return prediction.forecast_metric == "domestic_opening_weekend" and prediction.target_start_date is not None


def provision_movie(conn: Any, prediction: WeekendPrediction) -> MovieMatch:
    row = conn.execute(
        """
        INSERT INTO movies (title, release_year, release_date, updated_at)
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        RETURNING movie_id
        """,
        (prediction.source_movie_title, infer_prediction_year(prediction), prediction.target_start_date),
    ).fetchone()
    movie_id = int(row[0])
    upsert_movie_source_id(
        conn,
        movie_id=movie_id,
        prediction=prediction,
        match_status="provisional",
        match_method="provisional_boxofficereport_identity",
        match_score=1.0,
    )
    return MovieMatch(movie_id, "provisional", "provisional_boxofficereport_identity", 1.0, None)


def upsert_movie_source_id(
    conn: Any,
    *,
    movie_id: int,
    prediction: WeekendPrediction,
    match_status: str,
    match_method: str | None,
    match_score: float | None,
) -> None:
    movie_identity.upsert_movie_source_id(
        conn,
        movie_id=movie_id,
        source=movie_identity.SOURCE_BOXOFFICEREPORT,
        source_movie_id=prediction.source_movie_id,
        source_title=prediction.source_movie_title,
        match_status=match_status,
        match_method=match_method,
        match_score=match_score,
    )


def preferred_movie_candidate(candidates: list[MovieCandidate]) -> MovieCandidate:
    return sorted(candidates, key=lambda candidate: (candidate.movie_url is None, candidate.movie_id))[0]


def infer_prediction_year(prediction: WeekendPrediction) -> int | None:
    for value in (prediction.target_start_date, prediction.target_end_date, prediction.prediction_made_at):
        if value and re.match(r"20\d{2}", value):
            return int(value[:4])
    return None


def discover_pages(fetcher: HtmlFetcher, args: argparse.Namespace) -> list[ArchivePredictionPage]:
    print(f"Reading Box Office Report prediction archive {ARCHIVE_URL}", file=sys.stderr)
    html, _cache_path, _fetched = fetcher.get(ARCHIVE_URL)
    pages = parse_archive(html, source_url=ARCHIVE_URL)
    filtered: list[ArchivePredictionPage] = []
    for page in pages:
        if page.target_start_date is None:
            filtered.append(page)
            continue
        target_start = dt.date.fromisoformat(page.target_start_date)
        if args.start_date <= target_start <= args.end_date:
            filtered.append(page)
    if args.max_articles is not None:
        filtered = filtered[: args.max_articles]
    return filtered


def import_parse_result(conn: Any, result: ParseResult, *, issue_source: str) -> tuple[int, int]:
    if result.error is not None:
        upsert_article(
            conn,
            result.archive_page,
            status="article_page_unavailable",
            fetched_at=result.fetched_at,
            raw_cache_path=result.raw_cache_path,
            html="",
        )
        insert_issue(
            conn,
            issue_source=issue_source,
            issue_type="article_page_unavailable",
            article_url=result.archive_page.article_url,
            source_movie_title=None,
            details=result.error,
        )
        conn.commit()
        return 1, 0
    if result.article is None:
        raise RuntimeError("successful parse result is missing article")
    article_id = upsert_article(
        conn,
        result.article,
        status="parsed",
        fetched_at=result.fetched_at,
        raw_cache_path=result.raw_cache_path,
        html=result.html,
    )
    predictions = match_predictions(conn, result.predictions)
    clear_article_issues(conn, issue_source=issue_source, article_url=result.article.article_url)
    if not predictions:
        insert_issue(
            conn,
            issue_source=issue_source,
            issue_type="no_weekend_predictions_parsed",
            article_url=result.article.article_url,
            source_movie_title=None,
            details=f"No prediction table parsed from {result.article.title}",
        )
    insert_predictions(conn, article_id, predictions, fetched_at=result.fetched_at, raw_cache_path=result.raw_cache_path)
    conn.commit()
    return 1, len(predictions)


def configure_full_refresh_args(args: argparse.Namespace) -> None:
    if not getattr(args, "full_refresh", False):
        return
    args.refresh = True
    args.start_date = FULL_REFRESH_START_DATE
    args.end_date = FULL_REFRESH_END_DATE


def validate_args(args: argparse.Namespace) -> None:
    if args.end_date < args.start_date:
        raise SystemExit("--end-date must be on or after --start-date")
    if args.delay_seconds < MIN_DELAY_SECONDS and not args.offline and not args.dry_run:
        raise SystemExit(f"--delay-seconds must be at least {MIN_DELAY_SECONDS:g}")
    if "bot" not in args.user_agent.lower() and not args.offline and not args.dry_run:
        raise SystemExit("--user-agent must identify the scraper as a bot")


def run(args: argparse.Namespace) -> int:
    configure_full_refresh_args(args)
    validate_args(args)
    if args.print_cache_paths:
        fetcher = HtmlFetcher(
            args.cache_dir,
            refresh=False,
            offline=True,
            delay_seconds=args.delay_seconds,
            user_agent=args.user_agent,
        )
        for url in [ARCHIVE_URL, *args.cache_url]:
            print(f"{url}\t{fetcher.cache_path(url)}")
        return 0
    if args.dry_run:
        print(ARCHIVE_URL)
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
        pages = discover_pages(fetcher, args)
        imported_articles = 0
        imported_predictions = 0
        skipped_articles = 0
        for index, page in enumerate(pages, start=1):
            if not args.refresh and article_already_parsed(conn, page.article_url):
                skipped_articles += 1
                print(f"Skipping parsed page {index}/{len(pages)} {page.title}", file=sys.stderr)
                continue
            upsert_article(conn, page, status="discovered")
            conn.commit()
            print(f"Reading prediction page {index}/{len(pages)} {page.title}", file=sys.stderr)
            fetched_at = dt.datetime.now(dt.UTC).isoformat()
            try:
                html, cache_path, _fetched = fetcher.get(page.article_url)
                article, predictions = parse_article(html, article_url=page.article_url, fallback=page)
                result = ParseResult(page, article, predictions, fetched_at, cache_path, html)
            except FetchBlocked as exc:
                result = ParseResult(
                    page,
                    None,
                    [],
                    fetched_at,
                    fetcher.cache_path(page.article_url),
                    "",
                    error=str(exc),
                )
            article_count, prediction_count = import_parse_result(conn, result, issue_source=args.issue_source)
            imported_articles += article_count
            imported_predictions += prediction_count
        print(
            f"Imported {imported_articles} Box Office Report pages and {imported_predictions} predictions; "
            f"skipped {skipped_articles} parsed pages.",
            file=sys.stderr,
        )
    finally:
        fetcher.close()
        conn.close()
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import Box Office Report weekend predictions.")
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
        help="Delay between uncached HTTP requests. Must be at least 5.",
    )
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="HTTP User-Agent. Must identify as a bot.")
    parser.add_argument("--refresh", action="store_true", help="Reparse even when page status is parsed.")
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Reparse every prediction page in the Box Office Report archive.",
    )
    parser.add_argument("--offline", action="store_true", help="Require all pages to exist in cache.")
    parser.add_argument("--dry-run", action="store_true", help="Print discovery URL and exit.")
    parser.add_argument(
        "--print-cache-paths",
        action="store_true",
        help="Print expected cache file paths for discovery and any --cache-url values, then exit.",
    )
    parser.add_argument(
        "--cache-url",
        action="append",
        default=[],
        help="Extra URL to include when printing cache paths. May be repeated.",
    )
    parser.add_argument("--max-articles", type=int, help="Optional cap for smoke tests after archive discovery.")
    parser.add_argument("--issue-source", default="boxofficereport_weekend_import")
    return parser


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
