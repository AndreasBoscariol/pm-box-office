#!/usr/bin/env python3
"""Ingest BoxOfficeGuru.com historical box office predictions."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from html.parser import HTMLParser
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pm_box_office.domain import movies as movie_identity
from pm_box_office.db.connection import connect_database, database_url_from_env
from pm_box_office.sources.common.cli import parse_date_arg
from pm_box_office.sources.common.parsing import clean_text


BASE_URL = "http://www.boxofficeguru.com"
ARCHIVE_URL = f"{BASE_URL}/archives2.htm"
WAYBACK_CDX_URL = "https://web.archive.org/cdx"
DEFAULT_CACHE_DIR = Path("data/raw/boxofficeguru")
DEFAULT_START_DATE = dt.date(2026, 6, 1)
DEFAULT_END_DATE = dt.date(2026, 6, 30)
FULL_REFRESH_START_DATE = dt.date(1997, 1, 1)
FULL_REFRESH_END_DATE = dt.date(9999, 12, 31)
DEFAULT_USER_AGENT = "pm-box-office-boxofficeguru-bot/1.0 (+personal research; set --user-agent contact)"
MIN_DELAY_SECONDS = 5.0
DOMESTIC_MARKET = "US_CA"
DOMESTIC_CURRENCY = "USD"
PARSER_VERSION = "boxofficeguru_predictions_v1"
EASTERN = dt.timezone(dt.timedelta(hours=-5))


@dataclass(frozen=True)
class ArchiveWeekendPage:
    article_url: str
    title: str
    target_start_date: str | None
    target_end_date: str | None
    source_url: str


@dataclass(frozen=True)
class ArticleSnapshot:
    article_url: str
    snapshot_url: str | None
    snapshot_timestamp: str | None
    title: str
    status: str
    prediction_made_at: str | None
    prediction_made_date: str | None
    prediction_made_inference: str | None
    target_start_date: str | None
    target_end_date: str | None
    source_url: str


@dataclass(frozen=True)
class GuruPrediction:
    article_url: str
    snapshot_timestamp: str | None
    source_row_key: str
    source_movie_id: str
    source_movie_title: str
    normalized_movie_title: str
    distributor: str | None
    source_rank: int | None
    market: str
    currency: str
    forecast_metric: str
    weekend_gross_prediction_usd: int | None
    total_gross_prediction_usd: int | None
    percent_change_prediction: float | None
    theaters: int | None
    week_number: int | None
    target_start_date: str | None
    target_end_date: str | None
    prediction_made_at: str | None
    prediction_made_inference: str | None
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
class ParseResult:
    archive_page: ArchiveWeekendPage
    article: ArticleSnapshot | None
    predictions: list[GuruPrediction]
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
                body = response.read().decode("windows-1252", errors="replace")
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


class GuruHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[HtmlTable] = []
        self.title: str | None = None
        self.headings: list[str] = []
        self.document_parts: list[str] = []
        self._skip_depth = 0
        self._capture_title = False
        self._title_parts: list[str] = []
        self._capture_heading: str | None = None
        self._heading_parts: list[str] = []
        self._table_depth = 0
        self._current_rows: list[list[TableCell]] = []
        self._current_row: list[TableCell] | None = None
        self._cell_parts: list[str] | None = None
        self._cell_links: list[str] = []

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
                self._current_row.append(TableCell(clean_multiline_text("".join(self._cell_parts)), tuple(self._cell_links)))
            self._cell_parts = None
            self._cell_links = []
        elif self._table_depth and tag == "tr":
            if self._current_row is not None and any(cell.text or cell.links for cell in self._current_row):
                self._current_rows.append(self._current_row)
            self._current_row = None
        elif tag == "table" and self._table_depth:
            self._table_depth -= 1
            if self._table_depth == 0 and self._current_rows:
                self.tables.append(HtmlTable(self._current_rows))
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
    return urllib.parse.urljoin(BASE_URL + "/", href)


def canonical_url(href: str) -> str:
    parsed = urllib.parse.urlsplit(absolute_url(href))
    return urllib.parse.urlunsplit(("http", parsed.netloc.lower(), parsed.path, "", ""))


def wayback_snapshot_url(timestamp: str, article_url: str) -> str:
    return f"https://web.archive.org/web/{timestamp}id_/{article_url}"


def clean_multiline_text(value: str) -> str:
    text = value.replace("\xa0", " ").replace("\r", "\n")
    lines = [clean_text(line) for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def parse_archive(html: str, *, source_url: str = ARCHIVE_URL) -> list[ArchiveWeekendPage]:
    pages: list[ArchiveWeekendPage] = []
    year_blocks = split_archive_year_blocks(html)
    for year, block in year_blocks:
        for href, label in re.findall(r"<A\s+HREF=[\"']([^\"']+)[\"'][^>]*>(.*?)</A>", block, flags=re.IGNORECASE | re.DOTALL):
            page_url = canonical_url(href)
            if not re.search(r"/\d{6}\.htm$", page_url):
                continue
            label_text = clean_text(re.sub(r"<[^>]+>", " ", label).replace(",", "."))
            target_start, target_end = parse_archive_date_range(label_text, year)
            if not target_start:
                continue
            pages.append(
                ArchiveWeekendPage(
                    article_url=page_url,
                    title=f"Weekend Box Office ({label_text}, {year})",
                    target_start_date=target_start,
                    target_end_date=target_end,
                    source_url=source_url,
                )
            )
    deduped: dict[str, ArchiveWeekendPage] = {}
    for page in pages:
        deduped.setdefault(page.article_url, page)
    return list(deduped.values())


def split_archive_year_blocks(html: str) -> list[tuple[int, str]]:
    matches = list(re.finditer(r">\s*((?:19|20)\d{2})\s*<", html))
    blocks: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        year = int(match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(html)
        blocks.append((year, html[start:end]))
    return blocks


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

MONTH_PATTERN = "|".join(sorted(MONTHS, key=len, reverse=True))


def parse_archive_date_range(value: str, section_year: int) -> tuple[str | None, str | None]:
    text = clean_text(value).replace("\u2013", "-").replace("\u2014", "-").replace(",", ".")
    match = re.search(
        rf"\b(?P<m1>{MONTH_PATTERN})\.?\s*(?P<d1>\d{{1,2}})\s*-\s*"
        rf"(?:(?P<m2>{MONTH_PATTERN})\.?\s*)?(?P<d2>\d{{1,2}})\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None, None
    m1 = MONTHS[match.group("m1").lower().rstrip(".")]
    m2 = MONTHS[(match.group("m2") or match.group("m1")).lower().rstrip(".")]
    start = safe_date(section_year, m1, int(match.group("d1")))
    end_year = section_year + 1 if m2 < m1 else section_year
    end = safe_date(end_year, m2, int(match.group("d2")))
    return start.isoformat() if start else None, end.isoformat() if end else None


def parse_article_date_range(value: str) -> tuple[str | None, str | None]:
    match = re.search(
        rf"\b(?P<m1>{MONTH_PATTERN})\.?\s*(?P<d1>\d{{1,2}})\s*-\s*"
        rf"(?:(?P<m2>{MONTH_PATTERN})\.?\s*)?(?P<d2>\d{{1,2}}),?\s*(?P<year>(?:19|20)\d{{2}})\b",
        clean_text(value).replace("\u2013", "-").replace("\u2014", "-"),
        flags=re.IGNORECASE,
    )
    if not match:
        return None, None
    year = int(match.group("year"))
    m1 = MONTHS[match.group("m1").lower().rstrip(".")]
    m2 = MONTHS[(match.group("m2") or match.group("m1")).lower().rstrip(".")]
    start = safe_date(year, m1, int(match.group("d1")))
    end_year = year + 1 if m2 < m1 else year
    end = safe_date(end_year, m2, int(match.group("d2")))
    return start.isoformat() if start else None, end.isoformat() if end else None


def safe_date(year: int, month: int, day: int) -> dt.date | None:
    try:
        return dt.date(year, month, day)
    except ValueError:
        return None


def parse_article(
    html: str,
    *,
    article_url: str,
    fallback: ArchiveWeekendPage | None = None,
    snapshot_timestamp: str | None = None,
    snapshot_url: str | None = None,
) -> tuple[ArticleSnapshot, list[GuruPrediction]]:
    parser = GuruHtmlParser()
    parser.feed(html)
    title = parse_article_title(parser, fallback)
    target_start, target_end = parse_article_target_dates(parser, fallback)
    prediction_made_at, prediction_made_date, inference = parse_prediction_made(parser.document_text, target_start, snapshot_timestamp)
    article = ArticleSnapshot(
        article_url=canonical_url(article_url),
        snapshot_url=snapshot_url,
        snapshot_timestamp=snapshot_timestamp,
        title=title,
        status=classify_snapshot(parser.document_text, prediction_made_at, target_start, target_end),
        prediction_made_at=prediction_made_at,
        prediction_made_date=prediction_made_date,
        prediction_made_inference=inference,
        target_start_date=target_start,
        target_end_date=target_end,
        source_url=fallback.source_url if fallback else article_url,
    )
    predictions = parse_recap_forecast_text(parser.document_text, article)
    if article.status in {"forecast_snapshot", "estimate_snapshot"}:
        predictions.extend(parse_prediction_tables(parser.tables, article=article, start_ordinal=len(predictions) + 1))
    return article, predictions


def parse_article_title(parser: GuruHtmlParser, fallback: ArchiveWeekendPage | None) -> str:
    for heading in parser.headings:
        if "Weekend" in heading and "Box Office" in heading:
            return clean_multiline_text(heading.replace("\n", " "))
    if parser.title:
        return parser.title
    return fallback.title if fallback else "Box Office Guru Weekend Box Office"


def parse_article_target_dates(parser: GuruHtmlParser, fallback: ArchiveWeekendPage | None) -> tuple[str | None, str | None]:
    for value in [*parser.headings, parser.title or "", parser.document_text[:500]]:
        start, end = parse_article_date_range(value)
        if start:
            return start, end
    if fallback:
        return fallback.target_start_date, fallback.target_end_date
    return None, None


def parse_prediction_made(
    document_text: str,
    target_start_date: str | None,
    snapshot_timestamp: str | None,
) -> tuple[str | None, str | None, str | None]:
    updated = parse_last_updated(document_text)
    if updated is not None:
        if target_start_date is None or updated.date() <= dt.date.fromisoformat(target_start_date):
            return updated.isoformat(), updated.date().isoformat(), "last_updated_pre_weekend"
    if snapshot_timestamp:
        captured = parse_wayback_timestamp(snapshot_timestamp)
        if captured is not None and (target_start_date is None or captured.date() <= dt.date.fromisoformat(target_start_date)):
            return captured.isoformat(), captured.date().isoformat(), "wayback_capture_pre_weekend"
    if target_start_date and re.search(r"\bmy\s+(?:forecasts?|projections?)\b", document_text, flags=re.IGNORECASE):
        inferred = infer_guru_preview_timestamp(dt.date.fromisoformat(target_start_date))
        return inferred.isoformat(), inferred.date().isoformat(), "inferred_thursday_preview_from_recap"
    if updated is not None:
        return updated.isoformat(), updated.date().isoformat(), "last_updated_after_weekend"
    return None, None, None


def parse_last_updated(document_text: str) -> dt.datetime | None:
    match = re.search(
        rf"Last\s+Updated\s*:?\s*({MONTH_PATTERN})\.?\s+(\d{{1,2}}),?\s*((?:19|20)\d{{2}})"
        r"(?:\s+at\s+(\d{1,2}:\d{2})\s*([AP]M)\s*(?:E[DS]T|ET)?)?",
        document_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    month = MONTHS[match.group(1).lower().rstrip(".")]
    base = safe_date(int(match.group(3)), month, int(match.group(2)))
    if base is None:
        return None
    if match.group(4):
        try:
            time_part = dt.datetime.strptime(f"{match.group(4)}{match.group(5).upper()}", "%I:%M%p").time()
        except ValueError:
            time_part = dt.time(12, 0)
    else:
        time_part = dt.time(12, 0)
    return dt.datetime.combine(base, time_part, tzinfo=EASTERN)


def parse_wayback_timestamp(value: str) -> dt.datetime | None:
    try:
        parsed = dt.datetime.strptime(value[:14], "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return parsed.replace(tzinfo=dt.UTC)


def infer_guru_preview_timestamp(target_start: dt.date) -> dt.datetime:
    return dt.datetime.combine(target_start - dt.timedelta(days=1), dt.time(12, 0), tzinfo=EASTERN)


def classify_snapshot(
    document_text: str,
    prediction_made_at: str | None,
    target_start_date: str | None,
    target_end_date: str | None,
) -> str:
    text = document_text.lower()
    if re.search(r"\bmy\s+(?:forecasts?|projections?)\b", text):
        return "recap_with_forecast_mentions"
    made_date = dt.datetime.fromisoformat(prediction_made_at).date() if prediction_made_at else None
    start = dt.date.fromisoformat(target_start_date) if target_start_date else None
    end = dt.date.fromisoformat(target_end_date) if target_end_date else None
    if made_date and start and made_date < start:
        return "forecast_snapshot"
    if made_date and start and end and start <= made_date <= end:
        return "estimate_snapshot"
    if "final studio figures" in text or "final" in text:
        return "final_actual_snapshot"
    return "unclassified_snapshot"


def parse_recap_forecast_text(document_text: str, article: ArticleSnapshot) -> list[GuruPrediction]:
    predictions: list[GuruPrediction] = []
    text = clean_text(document_text)
    respectively = re.finditer(
        r"(?P<title1>[A-Z][A-Za-z0-9 '&:.,!-]{1,80}?)\s+and\s+"
        r"(?P<title2>[A-Z][A-Za-z0-9 '&:.,!-]{1,80}?)\s+"
        r"(?:went|opened|came|performed|grossed)[^.]{0,140}?"
        r"(?:forecasts?|projections?)\s+of\s+(?P<amount1>\$\s*\d[\d,.]*\s*(?:[KMB]|million)?)\s+and\s+"
        r"(?P<amount2>\$\s*\d[\d,.]*\s*(?:[KMB]|million)?)\s+respectively",
        text,
        flags=re.IGNORECASE,
    )
    for match in respectively:
        for title_key, amount_key in (("title1", "amount1"), ("title2", "amount2")):
            prediction = prediction_from_prose(
                article,
                title=match.group(title_key),
                amount_text=match.group(amount_key),
                raw_text=match.group(0),
                row_ordinal=len(predictions) + 1,
            )
            if prediction:
                predictions.append(prediction)
    single_patterns = [
        r"(?P<title>[A-Z][A-Za-z0-9 '&:,!-]{1,80}?)\s+opened\s+close\s+to\s+my\s+(?P<amount>\$\s*\d[\d,.]*\s*(?:[KMB]|million)?)\s+projection",
        r"my\s+(?P<amount>\$\s*\d[\d,.]*\s*(?:[KMB]|million)?)\s+(?:forecast|projection)\s+for\s+(?P<title>[A-Z][A-Za-z0-9 '&:,!-]{1,80})",
    ]
    seen_keys = {prediction.source_row_key for prediction in predictions}
    for pattern in single_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            prediction = prediction_from_prose(
                article,
                title=match.group("title"),
                amount_text=match.group("amount"),
                raw_text=match.group(0),
                row_ordinal=len(predictions) + 1,
            )
            if prediction and prediction.source_row_key not in seen_keys:
                predictions.append(prediction)
                seen_keys.add(prediction.source_row_key)
    return predictions


def prediction_from_prose(
    article: ArticleSnapshot,
    *,
    title: str,
    amount_text: str,
    raw_text: str,
    row_ordinal: int,
) -> GuruPrediction | None:
    movie_title = clean_prediction_title(title)
    amount = parse_money_value(amount_text)
    if not movie_title or amount is None:
        return None
    return build_prediction(
        article,
        source_movie_title=movie_title,
        source_rank=None,
        distributor=None,
        weekend_gross_prediction_usd=amount,
        total_gross_prediction_usd=None,
        percent_change_prediction=None,
        theaters=None,
        week_number=None,
        raw_forecast_text=clean_text(raw_text),
        source_context="retrospective_forecast_mention",
        row_ordinal=row_ordinal,
    )


def clean_prediction_title(value: str) -> str:
    text = clean_text(value)
    text = re.sub(r"^.*\bas\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:as|while|and|but|with)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(?:went|opened|came|performed|grossed)\s*$", "", text, flags=re.IGNORECASE)
    return text.strip(" ,.;")


def parse_prediction_tables(tables: list[HtmlTable], *, article: ArticleSnapshot, start_ordinal: int) -> list[GuruPrediction]:
    predictions: list[GuruPrediction] = []
    for table in tables:
        if not table.rows:
            continue
        headers = [normalize_header(cell.text) for cell in table.rows[0]]
        indexes = guru_table_indexes(headers)
        if indexes is None:
            continue
        for row in table.rows[1:]:
            parsed = parse_table_prediction_row(
                row,
                indexes=indexes,
                article=article,
                row_ordinal=start_ordinal + len(predictions),
            )
            if parsed:
                predictions.append(parsed)
    return predictions


def guru_table_indexes(headers: list[str]) -> dict[str, int] | None:
    rank = find_header(headers, "#")
    title = find_header(headers, "title")
    weekend = first_weekend_gross_header(headers)
    if rank is None or title is None or weekend is None:
        return None
    return {
        "rank": rank,
        "title": title,
        "weekend": weekend,
        "change": find_header(headers, "chg"),
        "theaters": find_header(headers, "theaters"),
        "week": find_header(headers, "weeks"),
        "total": find_header(headers, "cumulative"),
        "distributor": find_header(headers, "distributor") or find_header(headers, "dist"),
    }


def parse_table_prediction_row(
    row: list[TableCell],
    *,
    indexes: dict[str, int],
    article: ArticleSnapshot,
    row_ordinal: int,
) -> GuruPrediction | None:
    if len(row) <= max(indexes["rank"], indexes["title"], indexes["weekend"]):
        return None
    rank = parse_int(row[indexes["rank"]].text)
    title = clean_prediction_title(row[indexes["title"]].text)
    amount = parse_money_value(row[indexes["weekend"]].text)
    if not title or amount is None:
        return None
    if title.lower().startswith(("top ", "below ")):
        return None
    return build_prediction(
        article,
        source_movie_title=title,
        source_rank=rank,
        distributor=cell_text(row, indexes.get("distributor")) or None,
        weekend_gross_prediction_usd=amount,
        total_gross_prediction_usd=cell_money(row, indexes.get("total")),
        percent_change_prediction=parse_percent_change(cell_text(row, indexes.get("change"))),
        theaters=parse_int(cell_text(row, indexes.get("theaters"))),
        week_number=parse_int(cell_text(row, indexes.get("week"))),
        raw_forecast_text=" | ".join(cell.text for cell in row),
        source_context=article.status,
        row_ordinal=row_ordinal,
    )


def build_prediction(
    article: ArticleSnapshot,
    *,
    source_movie_title: str,
    source_rank: int | None,
    distributor: str | None,
    weekend_gross_prediction_usd: int | None,
    total_gross_prediction_usd: int | None,
    percent_change_prediction: float | None,
    theaters: int | None,
    week_number: int | None,
    raw_forecast_text: str,
    source_context: str,
    row_ordinal: int,
) -> GuruPrediction:
    normalized = normalize_movie_title(source_movie_title)
    key_material = "|".join(
        [
            article.article_url,
            article.snapshot_timestamp or "",
            str(row_ordinal),
            normalized,
            str(source_rank),
            str(weekend_gross_prediction_usd),
            str(total_gross_prediction_usd),
            source_context,
            PARSER_VERSION,
        ]
    )
    return GuruPrediction(
        article_url=article.article_url,
        snapshot_timestamp=article.snapshot_timestamp,
        source_row_key=hashlib.sha256(key_material.encode("utf-8")).hexdigest(),
        source_movie_id=boxofficeguru_source_movie_id(normalized, article.target_start_date),
        source_movie_title=source_movie_title,
        normalized_movie_title=normalized,
        distributor=distributor,
        source_rank=source_rank,
        market=DOMESTIC_MARKET,
        currency=DOMESTIC_CURRENCY,
        forecast_metric="domestic_weekend",
        weekend_gross_prediction_usd=weekend_gross_prediction_usd,
        total_gross_prediction_usd=total_gross_prediction_usd,
        percent_change_prediction=percent_change_prediction,
        theaters=theaters,
        week_number=week_number,
        target_start_date=article.target_start_date,
        target_end_date=article.target_end_date,
        prediction_made_at=article.prediction_made_at,
        prediction_made_inference=article.prediction_made_inference,
        raw_forecast_text=raw_forecast_text,
        source_context=source_context,
        parser_version=PARSER_VERSION,
        row_ordinal=row_ordinal,
    )


def find_header(headers: list[str], needle: str) -> int | None:
    normalized = normalize_header(needle)
    for index, header in enumerate(headers):
        if header == normalized or normalized in header:
            return index
    return None


def first_weekend_gross_header(headers: list[str]) -> int | None:
    for index, header in enumerate(headers):
        if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", header):
            return index
    return None


def cell_text(row: list[TableCell], index: int | None) -> str:
    if index is None or index < 0 or len(row) <= index:
        return ""
    return clean_text(row[index].text)


def cell_money(row: list[TableCell], index: int | None) -> int | None:
    return parse_money_value(cell_text(row, index))


def parse_money_value(value: str) -> int | None:
    text = clean_text(value).lower()
    if not text or text in {"n/a", "na", "-"}:
        return None
    match = re.search(r"\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(k|m|b|million)?", text)
    if not match:
        return None
    number = match.group(1).replace(",", "")
    unit = match.group(2) or ""
    multiplier = {"": 1, "k": 1_000, "m": 1_000_000, "million": 1_000_000, "b": 1_000_000_000}[unit]
    try:
        return int(round(float(number) * multiplier))
    except ValueError:
        return None


def parse_percent_change(value: str) -> float | None:
    match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%", clean_text(value))
    return float(match.group(1)) if match else None


def parse_int(value: str) -> int | None:
    match = re.search(r"\d+", clean_text(value).replace(",", ""))
    return int(match.group(0)) if match else None


def normalize_header(value: str) -> str:
    text = clean_text(value).lower().replace("%", "")
    return re.sub(r"[^a-z0-9#]+", " ", text).strip()


def normalize_movie_title(value: str) -> str:
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def boxofficeguru_source_movie_id(normalized_movie_title: str, target_start_date: str | None) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", normalized_movie_title).strip("-") or "unknown-title"
    return f"boxofficeguru:{DOMESTIC_MARKET}:{slug}:{target_start_date or 'unknown-date'}"


def discover_pages(fetcher: HtmlFetcher, args: argparse.Namespace) -> list[ArchiveWeekendPage]:
    print(f"Reading Box Office Guru archive {ARCHIVE_URL}", file=sys.stderr)
    html, _cache_path, _fetched = fetcher.get(ARCHIVE_URL)
    pages = parse_archive(html)
    filtered = [
        page
        for page in pages
        if page.target_start_date is None or args.start_date <= dt.date.fromisoformat(page.target_start_date) <= args.end_date
    ]
    if args.max_articles is not None:
        filtered = filtered[: args.max_articles]
    return filtered


def discover_snapshot_urls(fetcher: HtmlFetcher, page: ArchiveWeekendPage, args: argparse.Namespace) -> list[tuple[str, str | None]]:
    if not args.use_wayback:
        return [(page.article_url, None)]
    try:
        snapshots = wayback_snapshots(fetcher, page.article_url)
    except FetchBlocked as exc:
        if args.include_live_fallback:
            print(f"Wayback snapshot discovery failed for {page.article_url}; using live fallback: {exc}", file=sys.stderr)
            return [(page.article_url, None)]
        raise
    if page.target_start_date:
        target_start = dt.date.fromisoformat(page.target_start_date)
        lower = target_start - dt.timedelta(days=args.wayback_days_before)
        upper = target_start + dt.timedelta(days=args.wayback_days_after)
        snapshots = [
            timestamp
            for timestamp in snapshots
            if (captured := parse_wayback_timestamp(timestamp)) is not None and lower <= captured.date() <= upper
        ]
    if args.max_snapshots_per_article is not None:
        snapshots = snapshots[: args.max_snapshots_per_article]
    if not snapshots and args.include_live_fallback:
        return [(page.article_url, None)]
    return [(wayback_snapshot_url(timestamp, page.article_url), timestamp) for timestamp in snapshots]


def wayback_snapshots(fetcher: HtmlFetcher, article_url: str) -> list[str]:
    parsed = urllib.parse.urlsplit(article_url)
    query = urllib.parse.urlencode(
        {
            "url": f"{parsed.netloc}{parsed.path}",
            "output": "json",
            "fl": "timestamp,statuscode,mimetype,digest",
            "filter": "statuscode:200",
            "collapse": "digest",
        }
    )
    cdx_url = f"{WAYBACK_CDX_URL}?{query}"
    body, _cache_path, _fetched = fetcher.get(cdx_url)
    try:
        rows = json.loads(body)
    except json.JSONDecodeError:
        return []
    if not isinstance(rows, list) or len(rows) <= 1:
        return []
    timestamps: list[str] = []
    for row in rows[1:]:
        if isinstance(row, list) and row:
            timestamps.append(str(row[0]))
    return timestamps


def initialize_database(conn: Any) -> None:
    movie_identity.ensure_movie_identity_schema(conn)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS boxofficeguru_articles (
            article_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            article_url TEXT NOT NULL,
            snapshot_timestamp TEXT,
            snapshot_url TEXT,
            title TEXT NOT NULL,
            status TEXT NOT NULL,
            prediction_made_at TIMESTAMPTZ,
            prediction_made_date DATE,
            prediction_made_inference TEXT,
            target_start_date DATE,
            target_end_date DATE,
            source_url TEXT,
            fetched_at TIMESTAMPTZ,
            raw_cache_path TEXT,
            sha256 TEXT,
            parser_version TEXT NOT NULL,
            UNIQUE(article_url, snapshot_timestamp)
        );

        CREATE TABLE IF NOT EXISTS boxofficeguru_predictions (
            prediction_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            article_id BIGINT NOT NULL REFERENCES boxofficeguru_articles(article_id),
            source_row_key TEXT NOT NULL UNIQUE,
            source_movie_id TEXT NOT NULL,
            source_movie_title TEXT NOT NULL,
            normalized_movie_title TEXT NOT NULL,
            distributor TEXT,
            source_rank INTEGER,
            market TEXT NOT NULL,
            currency TEXT NOT NULL,
            forecast_metric TEXT NOT NULL,
            weekend_gross_prediction_usd BIGINT,
            total_gross_prediction_usd BIGINT,
            percent_change_prediction DOUBLE PRECISION,
            theaters INTEGER,
            week_number INTEGER,
            target_start_date DATE,
            target_end_date DATE,
            prediction_made_at TIMESTAMPTZ,
            prediction_made_inference TEXT,
            raw_forecast_text TEXT NOT NULL,
            source_context TEXT NOT NULL,
            parser_version TEXT NOT NULL,
            row_ordinal INTEGER NOT NULL,
            movie_id BIGINT REFERENCES movies(movie_id),
            match_status TEXT NOT NULL DEFAULT 'unmatched',
            match_method TEXT,
            match_score DOUBLE PRECISION,
            match_notes TEXT,
            fetched_at TIMESTAMPTZ,
            raw_cache_path TEXT
        );

        ALTER TABLE boxofficeguru_predictions
            ADD COLUMN IF NOT EXISTS movie_id BIGINT REFERENCES movies(movie_id);

        CREATE TABLE IF NOT EXISTS boxofficeguru_ingest_issues (
            issue_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            issue_source TEXT NOT NULL,
            issue_type TEXT NOT NULL,
            article_url TEXT,
            snapshot_timestamp TEXT,
            source_movie_title TEXT,
            details TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(issue_source, issue_type, article_url, snapshot_timestamp, source_movie_title, details)
        );

        CREATE INDEX IF NOT EXISTS idx_boxofficeguru_predictions_title
            ON boxofficeguru_predictions(normalized_movie_title);
        CREATE INDEX IF NOT EXISTS idx_boxofficeguru_predictions_movie_id
            ON boxofficeguru_predictions(movie_id);
        CREATE INDEX IF NOT EXISTS idx_boxofficeguru_predictions_target_start
            ON boxofficeguru_predictions(target_start_date);
        CREATE INDEX IF NOT EXISTS idx_boxofficeguru_articles_prediction_made
            ON boxofficeguru_articles(prediction_made_at);
        """
    )


def upsert_article(conn: Any, article: ArticleSnapshot | ArchiveWeekendPage, *, status: str | None = None, fetched_at: str | None = None, raw_cache_path: Path | None = None, html: str = "") -> int:
    if isinstance(article, ArchiveWeekendPage):
        article_url = article.article_url
        snapshot_timestamp = None
        snapshot_url = None
        title = article.title
        prediction_made_at = None
        prediction_made_date = None
        prediction_made_inference = None
        target_start_date = article.target_start_date
        target_end_date = article.target_end_date
        source_url = article.source_url
        article_status = status or "discovered"
    else:
        article_url = article.article_url
        snapshot_timestamp = article.snapshot_timestamp
        snapshot_url = article.snapshot_url
        title = article.title
        prediction_made_at = article.prediction_made_at
        prediction_made_date = article.prediction_made_date
        prediction_made_inference = article.prediction_made_inference
        target_start_date = article.target_start_date
        target_end_date = article.target_end_date
        source_url = article.source_url
        article_status = status or article.status
    row = conn.execute(
        """
        INSERT INTO boxofficeguru_articles (
            article_url, snapshot_timestamp, snapshot_url, title, status,
            prediction_made_at, prediction_made_date, prediction_made_inference,
            target_start_date, target_end_date, source_url, fetched_at, raw_cache_path,
            sha256, parser_version
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (article_url, snapshot_timestamp) DO UPDATE SET
            snapshot_url = COALESCE(excluded.snapshot_url, boxofficeguru_articles.snapshot_url),
            title = excluded.title,
            status = excluded.status,
            prediction_made_at = COALESCE(excluded.prediction_made_at, boxofficeguru_articles.prediction_made_at),
            prediction_made_date = COALESCE(excluded.prediction_made_date, boxofficeguru_articles.prediction_made_date),
            prediction_made_inference = COALESCE(excluded.prediction_made_inference, boxofficeguru_articles.prediction_made_inference),
            target_start_date = COALESCE(excluded.target_start_date, boxofficeguru_articles.target_start_date),
            target_end_date = COALESCE(excluded.target_end_date, boxofficeguru_articles.target_end_date),
            source_url = COALESCE(excluded.source_url, boxofficeguru_articles.source_url),
            fetched_at = COALESCE(excluded.fetched_at, boxofficeguru_articles.fetched_at),
            raw_cache_path = COALESCE(excluded.raw_cache_path, boxofficeguru_articles.raw_cache_path),
            sha256 = COALESCE(excluded.sha256, boxofficeguru_articles.sha256),
            parser_version = excluded.parser_version
        RETURNING article_id
        """,
        (
            article_url,
            snapshot_timestamp,
            snapshot_url,
            title,
            article_status,
            prediction_made_at,
            prediction_made_date,
            prediction_made_inference,
            target_start_date,
            target_end_date,
            source_url,
            fetched_at,
            str(raw_cache_path) if raw_cache_path else None,
            hashlib.sha256(html.encode("utf-8")).hexdigest() if html else None,
            PARSER_VERSION,
        ),
    ).fetchone()
    return int(row[0])


def insert_predictions(conn: Any, article_id: int, predictions: list[GuruPrediction], *, fetched_at: str, raw_cache_path: Path) -> None:
    if not predictions:
        return
    conn.executemany(
        """
        INSERT INTO boxofficeguru_predictions (
            article_id, source_row_key, source_movie_id, source_movie_title,
            normalized_movie_title, distributor, source_rank, market, currency,
            forecast_metric, weekend_gross_prediction_usd, total_gross_prediction_usd,
            percent_change_prediction, theaters, week_number, target_start_date,
            target_end_date, prediction_made_at, prediction_made_inference,
            raw_forecast_text, source_context, parser_version, row_ordinal,
            movie_id, match_status, match_method, match_score,
            match_notes, fetched_at, raw_cache_path
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (source_row_key) DO UPDATE SET
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
            percent_change_prediction = excluded.percent_change_prediction,
            theaters = excluded.theaters,
            week_number = excluded.week_number,
            target_start_date = excluded.target_start_date,
            target_end_date = excluded.target_end_date,
            prediction_made_at = excluded.prediction_made_at,
            prediction_made_inference = excluded.prediction_made_inference,
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
                prediction.percent_change_prediction,
                prediction.theaters,
                prediction.week_number,
                prediction.target_start_date,
                prediction.target_end_date,
                prediction.prediction_made_at,
                prediction.prediction_made_inference,
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
    for prediction in predictions:
        if prediction.movie_id is None:
            continue
        movie_identity.upsert_movie_source_id(
            conn,
            movie_id=prediction.movie_id,
            source=movie_identity.SOURCE_BOXOFFICEGURU,
            source_movie_id=prediction.source_movie_id,
            source_title=prediction.source_movie_title,
            match_status=prediction.match_status,
            match_method=prediction.match_method,
            match_score=prediction.match_score,
        )


def insert_issue(conn: Any, *, issue_source: str, issue_type: str, article_url: str | None, snapshot_timestamp: str | None, source_movie_title: str | None, details: str) -> None:
    conn.execute(
        """
        INSERT INTO boxofficeguru_ingest_issues (
            issue_source, issue_type, article_url, snapshot_timestamp, source_movie_title, details
        ) VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """,
        (issue_source, issue_type, article_url, snapshot_timestamp, source_movie_title, details),
    )


def article_snapshot_already_parsed(conn: Any, article_url: str, snapshot_timestamp: str | None) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM boxofficeguru_articles
        WHERE article_url = %s
          AND snapshot_timestamp IS NOT DISTINCT FROM %s
          AND status IN ('forecast_snapshot', 'estimate_snapshot', 'recap_with_forecast_mentions', 'final_actual_snapshot', 'unclassified_snapshot')
        LIMIT 1
        """,
        (article_url, snapshot_timestamp),
    ).fetchone()
    return row is not None


def import_parse_result(conn: Any, result: ParseResult, *, issue_source: str) -> tuple[int, int]:
    if result.error is not None:
        upsert_article(conn, result.archive_page, status="article_page_unavailable", fetched_at=result.fetched_at, raw_cache_path=result.raw_cache_path)
        insert_issue(
            conn,
            issue_source=issue_source,
            issue_type="article_page_unavailable",
            article_url=result.archive_page.article_url,
            snapshot_timestamp=None,
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
        fetched_at=result.fetched_at,
        raw_cache_path=result.raw_cache_path,
        html=result.html,
    )
    if not result.predictions:
        insert_issue(
            conn,
            issue_source=issue_source,
            issue_type="no_predictions_parsed",
            article_url=result.article.article_url,
            snapshot_timestamp=result.article.snapshot_timestamp,
            source_movie_title=None,
            details=f"No Box Office Guru prediction rows parsed from {result.article.title}; status={result.article.status}",
        )
    insert_predictions(conn, article_id, result.predictions, fetched_at=result.fetched_at, raw_cache_path=result.raw_cache_path)
    conn.commit()
    return 1, len(result.predictions)


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
        fetcher = HtmlFetcher(args.cache_dir, refresh=False, offline=True, delay_seconds=args.delay_seconds, user_agent=args.user_agent)
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
        skipped_snapshots = 0
        for page_index, page in enumerate(pages, start=1):
            upsert_article(conn, page, status="discovered")
            conn.commit()
            snapshot_urls = discover_snapshot_urls(fetcher, page, args)
            for snapshot_index, (snapshot_url_value, snapshot_timestamp) in enumerate(snapshot_urls, start=1):
                if not args.refresh and article_snapshot_already_parsed(conn, page.article_url, snapshot_timestamp):
                    skipped_snapshots += 1
                    continue
                print(
                    f"Reading Guru page {page_index}/{len(pages)} snapshot {snapshot_index}/{len(snapshot_urls)} {page.title}",
                    file=sys.stderr,
                )
                fetched_at = dt.datetime.now(dt.UTC).isoformat()
                try:
                    html, cache_path, _fetched = fetcher.get(snapshot_url_value)
                    article, predictions = parse_article(
                        html,
                        article_url=page.article_url,
                        fallback=page,
                        snapshot_timestamp=snapshot_timestamp,
                        snapshot_url=snapshot_url_value if snapshot_timestamp else None,
                    )
                    result = ParseResult(page, article, predictions, fetched_at, cache_path, html)
                except FetchBlocked as exc:
                    result = ParseResult(page, None, [], fetched_at, fetcher.cache_path(snapshot_url_value), "", error=str(exc))
                article_count, prediction_count = import_parse_result(conn, result, issue_source=args.issue_source)
                imported_articles += article_count
                imported_predictions += prediction_count
        print(
            f"Imported {imported_articles} Box Office Guru snapshots and {imported_predictions} predictions; "
            f"skipped {skipped_snapshots} parsed snapshots.",
            file=sys.stderr,
        )
    finally:
        fetcher.close()
        conn.close()
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import Box Office Guru historical weekend predictions.")
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
    parser.add_argument("--refresh", action="store_true", help="Reparse even when a snapshot was already parsed.")
    parser.add_argument("--full-refresh", action="store_true", help="Scan the complete Box Office Guru archive.")
    parser.add_argument("--offline", action="store_true", help="Require all pages to exist in cache.")
    parser.add_argument("--dry-run", action="store_true", help="Print discovery URL and exit.")
    parser.add_argument("--print-cache-paths", action="store_true", help="Print expected cache file paths and exit.")
    parser.add_argument("--cache-url", action="append", default=[], help="Extra URL to include when printing cache paths.")
    parser.add_argument("--max-articles", type=int, help="Optional cap for smoke tests after archive discovery.")
    parser.add_argument(
        "--use-wayback",
        action="store_true",
        help="Use Internet Archive CDX snapshots for each Guru weekend URL to find preserved prediction versions.",
    )
    parser.add_argument("--wayback-days-before", type=int, default=7)
    parser.add_argument("--wayback-days-after", type=int, default=3)
    parser.add_argument("--max-snapshots-per-article", type=int, default=5)
    parser.add_argument(
        "--include-live-fallback",
        action="store_true",
        help="When --use-wayback finds no matching snapshots, also parse the live archive page.",
    )
    parser.add_argument("--issue-source", default="boxofficeguru_prediction_import")
    return parser


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
