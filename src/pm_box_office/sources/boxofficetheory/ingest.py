#!/usr/bin/env python3
"""Ingest BoxOfficeTheory.com movie box office predictions into PostgreSQL."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from html.parser import HTMLParser
import html
import json
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

from pm_box_office.domain import movies as movie_identity
from pm_box_office.db.connection import connect_database, database_url_from_env
from pm_box_office.sources.common.cli import parse_date_arg
from pm_box_office.sources.common.parsing import clean_text


BASE_URL = "https://boxofficetheory.com"
POSTS_API_URL = f"{BASE_URL}/wp-json/wp/v2/posts"
TRACKING_FORECASTS_CATEGORY_ID = 6
DEFAULT_CACHE_DIR = Path("data/raw/boxofficetheory")
DEFAULT_START_DATE = dt.date(2026, 6, 1)
DEFAULT_END_DATE = dt.date(2026, 6, 30)
FULL_REFRESH_START_DATE = dt.date(1900, 1, 1)
FULL_REFRESH_END_DATE = dt.date(9999, 12, 31)
DEFAULT_USER_AGENT = "pm-box-office-boxofficetheory-bot/1.0 (+personal research; set --user-agent contact)"
MIN_DELAY_SECONDS = 5.0
DOMESTIC_MARKET = "US_CA"
DOMESTIC_CURRENCY = "USD"
PARSER_VERSION = "boxofficetheory_public_predictions_v1"
WP_API_FIELDS = "id,date_gmt,link,title,content,excerpt,categories"


@dataclass(frozen=True)
class PostRecord:
    source_post_id: int
    post_url: str
    title: str
    author: str | None
    published_at: str
    published_date: str
    excerpt: str | None
    content_html: str
    source_url: str


@dataclass(frozen=True)
class TableCell:
    text: str


@dataclass(frozen=True)
class HtmlTable:
    rows: list[list[TableCell]]


@dataclass(frozen=True)
class ParsedMoney:
    low_usd: int | None
    high_usd: int | None
    is_plus: bool = False


@dataclass(frozen=True)
class TheoryPrediction:
    post_url: str
    source_row_key: str
    source_movie_id: str
    source_movie_title: str
    normalized_movie_title: str
    distributor: str | None
    release_date: str | None
    market: str
    currency: str
    prediction_scope: str
    forecast_metric: str
    opening_weekend_low_usd: int | None
    opening_weekend_high_usd: int | None
    opening_weekend_pinpoint_usd: int | None
    opening_weekend_day_count: int | None
    alternate_opening_weekend_usd: int | None
    alternate_opening_weekend_day_count: int | None
    domestic_total_low_usd: int | None
    domestic_total_high_usd: int | None
    domestic_total_pinpoint_usd: int | None
    weekend_forecast_usd: int | None
    weekend_day_count: int | None
    projected_domestic_total_usd: int | None
    percent_change: float | None
    change_label: str | None
    location_count: int | None
    domestic_multiplier_pinpoint: float | None
    prediction_made_at: str
    prediction_made_date: str
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


@dataclass(frozen=True)
class MovieMatch:
    movie_id: int | None
    status: str
    method: str | None
    score: float | None
    notes: str | None


@dataclass(frozen=True)
class PageParseResult:
    post: PostRecord
    predictions: list[TheoryPrediction]
    fetched_at: str
    raw_cache_path: Path
    raw_json: str
    error: str | None = None


class FetchBlocked(RuntimeError):
    """Raised when API data cannot be fetched and no cache is available."""


class CachedJsonFetcher:
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
        return self.cache_dir / f"{digest}.json"

    def get(self, url: str) -> tuple[str, Path, bool, dict[str, str]]:
        cache_path = self.cache_path(url)
        headers_path = cache_path.with_suffix(".headers.json")
        if cache_path.exists() and (not self.refresh or self.offline):
            headers = {}
            if headers_path.exists():
                headers = json.loads(headers_path.read_text(encoding="utf-8"))
            return cache_path.read_text(encoding="utf-8"), cache_path, False, headers
        if self.offline:
            raise FetchBlocked(f"Cache miss in offline mode: {url}\nExpected cached JSON at: {cache_path}")
        self._wait()
        request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8", errors="replace")
                headers = {key.lower(): value for key, value in response.headers.items()}
        except urllib.error.HTTPError as exc:
            raise FetchBlocked(f"HTTP {exc.code} while fetching {url}") from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            raise FetchBlocked(f"GET {url} failed: {exc}") from exc
        self._last_request_at = time.monotonic()
        cache_path.write_text(body, encoding="utf-8")
        headers_path.write_text(json.dumps(headers, indent=2, sort_keys=True), encoding="utf-8")
        return body, cache_path, True, headers

    def close(self) -> None:
        return None

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        delay = max(0.0, self.delay_seconds - elapsed)
        if delay:
            time.sleep(delay)


class PredictionHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[HtmlTable] = []
        self.blocks: list[str] = []
        self._table_depth = 0
        self._current_rows: list[list[TableCell]] = []
        self._current_row: list[TableCell] | None = None
        self._cell_parts: list[str] | None = None
        self._block_tag: str | None = None
        self._block_parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._current_rows = []
        elif self._table_depth and tag == "tr":
            self._current_row = []
        elif self._table_depth and tag in {"td", "th"}:
            self._cell_parts = []
        elif self._cell_parts is not None and tag == "br":
            self._cell_parts.append("\n")
        elif not self._table_depth and tag in {"p", "li"}:
            self._block_tag = tag
            self._block_parts = []
        elif self._block_tag is not None and tag == "br":
            self._block_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if self._table_depth and tag in {"td", "th"} and self._cell_parts is not None:
            if self._current_row is not None:
                self._current_row.append(TableCell(text=clean_multiline_text("".join(self._cell_parts))))
            self._cell_parts = None
        elif self._table_depth and tag == "tr":
            if self._current_row is not None and any(cell.text for cell in self._current_row):
                self._current_rows.append(self._current_row)
            self._current_row = None
        elif tag == "table" and self._table_depth:
            self._table_depth -= 1
            if self._table_depth == 0 and self._current_rows:
                self.tables.append(HtmlTable(rows=self._current_rows))
                self._current_rows = []
        elif tag == self._block_tag:
            text = clean_multiline_text("".join(self._block_parts))
            if text:
                self.blocks.append(text)
            self._block_tag = None
            self._block_parts = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._cell_parts is not None:
            self._cell_parts.append(data)
        if self._block_tag is not None:
            self._block_parts.append(data)


def clean_multiline_text(value: str) -> str:
    text = html.unescape(value).replace("\xa0", " ").replace("\r", "\n")
    lines = [clean_text(line) for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def api_posts_url(*, page: int, per_page: int, category_id: int) -> str:
    query = urllib.parse.urlencode(
        {
            "categories": str(category_id),
            "per_page": str(per_page),
            "page": str(page),
            "_fields": WP_API_FIELDS,
        }
    )
    return f"{POSTS_API_URL}?{query}"


def parse_post_records(raw_json: str, *, source_url: str) -> list[PostRecord]:
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid Box Office Theory API JSON from {source_url}: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"expected list from Box Office Theory API: {source_url}")
    posts: list[PostRecord] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        post_id = item.get("id")
        link = item.get("link")
        date_gmt = item.get("date_gmt")
        title_obj = item.get("title") or {}
        content_obj = item.get("content") or {}
        excerpt_obj = item.get("excerpt") or {}
        if not isinstance(post_id, int) or not link or not date_gmt:
            continue
        published_at = parse_wp_gmt_datetime(str(date_gmt))
        published_date = published_at[:10]
        posts.append(
            PostRecord(
                source_post_id=post_id,
                post_url=canonical_url(str(link)),
                title=strip_html_text(str(title_obj.get("rendered") or "")),
                author="Shawn Robbins",
                published_at=published_at,
                published_date=published_date,
                excerpt=strip_html_text(str(excerpt_obj.get("rendered") or "")) or None,
                content_html=str(content_obj.get("rendered") or ""),
                source_url=source_url,
            )
        )
    return posts


def parse_wp_gmt_datetime(value: str) -> str:
    try:
        parsed = dt.datetime.fromisoformat(value).replace(tzinfo=dt.UTC)
    except ValueError:
        parsed = dt.datetime.now(dt.UTC)
    return parsed.isoformat()


class TextOnlyParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def strip_html_text(value: str) -> str:
    parser = TextOnlyParser()
    parser.feed(value)
    return clean_text(" ".join(parser.parts))


def canonical_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def parse_predictions(post: PostRecord) -> list[TheoryPrediction]:
    parser = PredictionHtmlParser()
    parser.feed(post.content_html)
    predictions: list[TheoryPrediction] = []
    for table in parser.tables:
        table_predictions = parse_prediction_table(
            table,
            post=post,
            first_row_ordinal=len(predictions) + 1,
        )
        predictions.extend(table_predictions)
    prose_predictions = parse_prose_predictions(
        parser.blocks,
        post=post,
        first_row_ordinal=len(predictions) + 1,
    )
    predictions.extend(dedupe_predictions(prose_predictions, existing=predictions))
    return predictions


def parse_prediction_table(
    table: HtmlTable,
    *,
    post: PostRecord,
    first_row_ordinal: int,
) -> list[TheoryPrediction]:
    if len(table.rows) < 2:
        return []
    headers = [normalize_header(cell.text) for cell in table.rows[0]]
    if is_tracking_table(headers):
        return parse_tracking_table(table, headers=headers, post=post, first_row_ordinal=first_row_ordinal)
    if is_weekend_table(headers):
        return parse_weekend_table(table, headers=headers, post=post, first_row_ordinal=first_row_ordinal)
    return []


def is_tracking_table(headers: list[str]) -> bool:
    return (
        find_header(headers, {"release date"}) is not None
        and find_header(headers, {"title"}) is not None
        and find_header_containing(headers, ["opening"]) is not None
    )


def is_weekend_table(headers: list[str]) -> bool:
    return (
        find_header(headers, {"film", "title"}) is not None
        and find_header(headers, {"distributor", "studio", "studio distributor"}) is not None
        and find_header_containing(headers, ["weekend", "forecast"]) is not None
    )


def parse_tracking_table(
    table: HtmlTable,
    *,
    headers: list[str],
    post: PostRecord,
    first_row_ordinal: int,
) -> list[TheoryPrediction]:
    indexes = {
        "release_date": find_header(headers, {"release date"}),
        "title": find_header(headers, {"title"}),
        "distributor": find_header(headers, {"distributor", "studio", "studio distributor", "distribution studio"}),
        "opening_low": find_header_containing(headers, ["low", "opening"]),
        "opening_high": find_header_containing(headers, ["high", "opening"]),
        "opening_pinpoint": find_header_containing(headers, ["pinpoint", "opening"])
        or find_header_containing(headers, ["target", "opening"]),
        "alternate_opening": find_alternate_opening_index(headers),
        "total_low": find_header_containing(headers, ["total", "low"]),
        "total_high": find_header_containing(headers, ["total", "high"]),
        "total_pinpoint": find_header_containing(headers, ["total", "pinpoint"])
        or find_header_containing(headers, ["total", "target"]),
        "multiplier": find_header_containing(headers, ["multiplier"]),
    }
    required = [indexes["release_date"], indexes["title"]]
    if any(index is None for index in required):
        return []
    predictions: list[TheoryPrediction] = []
    for row in table.rows[1:]:
        title = cell_text(row, indexes["title"])
        if not title or normalize_header(title) in {"comparisons", "available for paid substack subscribers"}:
            continue
        release_date = parse_release_date(cell_text(row, indexes["release_date"]), post.published_date)
        distributor = cell_text(row, indexes["distributor"]) or None
        opening_low = first_money(cell_text(row, indexes["opening_low"]))
        opening_high = first_money(cell_text(row, indexes["opening_high"]))
        opening_pinpoint, alternate_value, alternate_days_from_pinpoint = parse_pinpoint_with_alternate(
            cell_text(row, indexes["opening_pinpoint"])
        )
        alternate_opening = first_money(cell_text(row, indexes["alternate_opening"])) or alternate_value
        alternate_day_count = day_count_from_text(headers[indexes["alternate_opening"]]) if indexes["alternate_opening"] is not None else None
        alternate_day_count = alternate_day_count or alternate_days_from_pinpoint
        total_low = first_money(cell_text(row, indexes["total_low"]))
        total_high = first_money(cell_text(row, indexes["total_high"]))
        total_pinpoint = first_money(cell_text(row, indexes["total_pinpoint"]))
        if not any([opening_low, opening_high, opening_pinpoint, alternate_opening, total_low, total_high, total_pinpoint]):
            continue
        row_ordinal = first_row_ordinal + len(predictions)
        predictions.append(
            build_prediction(
                post=post,
                title=title,
                distributor=distributor,
                release_date=release_date,
                prediction_scope="pre_release_tracking",
                forecast_metric="domestic_opening_and_total",
                opening_weekend_low_usd=opening_low,
                opening_weekend_high_usd=opening_high,
                opening_weekend_pinpoint_usd=opening_pinpoint,
                opening_weekend_day_count=day_count_from_text(" ".join(headers)) or 3,
                alternate_opening_weekend_usd=alternate_opening,
                alternate_opening_weekend_day_count=alternate_day_count,
                domestic_total_low_usd=total_low,
                domestic_total_high_usd=total_high,
                domestic_total_pinpoint_usd=total_pinpoint,
                domestic_multiplier_pinpoint=parse_float(cell_text(row, indexes["multiplier"])),
                raw_forecast_text=raw_row_text(table.rows[0], row),
                source_context="tracking_forecast_table",
                row_ordinal=row_ordinal,
            )
        )
    return predictions


def parse_weekend_table(
    table: HtmlTable,
    *,
    headers: list[str],
    post: PostRecord,
    first_row_ordinal: int,
) -> list[TheoryPrediction]:
    indexes = {
        "title": find_header(headers, {"film", "title"}),
        "distributor": find_header(headers, {"distributor", "studio", "studio distributor"}),
        "weekend": find_header_containing(headers, ["weekend", "forecast"]),
        "four_day": find_header_containing(headers, ["4", "day", "forecast"]),
        "total": find_header_containing(headers, ["projected", "domestic", "total"]),
        "change": find_header_containing(headers, ["change"]),
        "locations": find_header_containing(headers, ["location", "count"]),
    }
    if indexes["title"] is None or indexes["weekend"] is None:
        return []
    predictions: list[TheoryPrediction] = []
    for row in table.rows[1:]:
        title = cell_text(row, indexes["title"])
        if not title or normalize_header(title) in {"comparisons"}:
            continue
        weekend_forecast = first_money(cell_text(row, indexes["weekend"]))
        four_day_forecast = first_money(cell_text(row, indexes["four_day"]))
        projected_total = first_money(cell_text(row, indexes["total"]))
        if not any([weekend_forecast, four_day_forecast, projected_total]):
            continue
        row_ordinal = first_row_ordinal + len(predictions)
        change_label = cell_text(row, indexes["change"]) or None
        predictions.append(
            build_prediction(
                post=post,
                title=title,
                distributor=cell_text(row, indexes["distributor"]) or None,
                release_date=None,
                prediction_scope="weekend_forecast",
                forecast_metric="domestic_weekend",
                weekend_forecast_usd=weekend_forecast,
                weekend_day_count=day_count_from_text(headers[indexes["weekend"]]) or 3,
                alternate_opening_weekend_usd=four_day_forecast,
                alternate_opening_weekend_day_count=day_count_from_text(headers[indexes["four_day"]])
                if indexes["four_day"] is not None
                else None,
                projected_domestic_total_usd=projected_total,
                percent_change=parse_percent_change(change_label or ""),
                change_label=change_label,
                location_count=parse_location_count(cell_text(row, indexes["locations"])),
                raw_forecast_text=raw_row_text(table.rows[0], row),
                source_context="weekend_forecast_table",
                row_ordinal=row_ordinal,
            )
        )
    return predictions


def parse_prose_predictions(
    blocks: list[str],
    *,
    post: PostRecord,
    first_row_ordinal: int,
) -> list[TheoryPrediction]:
    predictions: list[TheoryPrediction] = []
    for block in blocks:
        if "BOT Domestic Opening Weekend Forecast Range" not in block:
            continue
        lines = [line for line in block.split("\n") if clean_text(line)]
        range_line_index = next(
            (index for index, line in enumerate(lines) if "BOT Domestic Opening Weekend Forecast Range" in line),
            None,
        )
        if range_line_index is None:
            continue
        identity = "\n".join(lines[:range_line_index])
        title, distributor, release_date = parse_prose_identity(identity, post.published_date)
        if not title:
            continue
        range_text = " ".join(lines[range_line_index:])
        parsed_range = parse_money_range(range_text)
        if parsed_range.low_usd is None and parsed_range.high_usd is None:
            continue
        row_ordinal = first_row_ordinal + len(predictions)
        predictions.append(
            build_prediction(
                post=post,
                title=title,
                distributor=distributor,
                release_date=release_date,
                prediction_scope="pre_release_tracking",
                forecast_metric="domestic_opening_weekend",
                opening_weekend_low_usd=parsed_range.low_usd,
                opening_weekend_high_usd=parsed_range.high_usd,
                opening_weekend_day_count=day_count_from_text(range_text) or 3,
                raw_forecast_text=block,
                source_context="prose_opening_range",
                row_ordinal=row_ordinal,
            )
        )
    return predictions


def parse_prose_identity(value: str, published_date: str) -> tuple[str | None, str | None, str | None]:
    text = clean_multiline_text(value).replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None, None, None
    match = re.match(r"(.+?)\s*\(([^()]*)\)\s*$", text)
    distributor = None
    release_date = None
    title = text
    if match:
        title = clean_text(match.group(1))
        parenthetical = clean_text(match.group(2))
        date_match = re.search(r"\b(?:\d{1,2}/\d{1,2}/\d{2,4}|[A-Za-z]+\s+\d{1,2})\b", parenthetical)
        if date_match:
            release_date = parse_release_date(date_match.group(0), published_date)
            distributor = clean_text(parenthetical.replace(date_match.group(0), "").strip(" /-")) or None
        else:
            distributor = parenthetical or None
    return clean_movie_title(title), distributor, release_date


def dedupe_predictions(
    candidates: list[TheoryPrediction],
    *,
    existing: list[TheoryPrediction],
) -> list[TheoryPrediction]:
    seen = {prediction_identity_key(prediction) for prediction in existing}
    deduped: list[TheoryPrediction] = []
    for candidate in candidates:
        key = prediction_identity_key(candidate)
        if key in seen:
            continue
        deduped.append(candidate)
        seen.add(key)
    return deduped


def prediction_identity_key(prediction: TheoryPrediction) -> tuple[object, ...]:
    return (
        prediction.normalized_movie_title,
        prediction.release_date,
        prediction.prediction_scope,
        prediction.opening_weekend_low_usd,
        prediction.opening_weekend_high_usd,
        prediction.opening_weekend_pinpoint_usd,
        prediction.weekend_forecast_usd,
        prediction.projected_domestic_total_usd,
    )


def build_prediction(
    *,
    post: PostRecord,
    title: str,
    distributor: str | None,
    release_date: str | None,
    prediction_scope: str,
    forecast_metric: str,
    raw_forecast_text: str,
    source_context: str,
    row_ordinal: int,
    opening_weekend_low_usd: int | None = None,
    opening_weekend_high_usd: int | None = None,
    opening_weekend_pinpoint_usd: int | None = None,
    opening_weekend_day_count: int | None = None,
    alternate_opening_weekend_usd: int | None = None,
    alternate_opening_weekend_day_count: int | None = None,
    domestic_total_low_usd: int | None = None,
    domestic_total_high_usd: int | None = None,
    domestic_total_pinpoint_usd: int | None = None,
    weekend_forecast_usd: int | None = None,
    weekend_day_count: int | None = None,
    projected_domestic_total_usd: int | None = None,
    percent_change: float | None = None,
    change_label: str | None = None,
    location_count: int | None = None,
    domestic_multiplier_pinpoint: float | None = None,
) -> TheoryPrediction:
    clean_title = clean_movie_title(title)
    normalized = normalize_movie_title(clean_title)
    key_material = "|".join(
        [
            post.post_url,
            str(row_ordinal),
            normalized,
            release_date or "",
            prediction_scope,
            forecast_metric,
            str(opening_weekend_low_usd),
            str(opening_weekend_high_usd),
            str(opening_weekend_pinpoint_usd),
            str(weekend_forecast_usd),
            str(projected_domestic_total_usd),
            PARSER_VERSION,
        ]
    )
    return TheoryPrediction(
        post_url=post.post_url,
        source_row_key=hashlib.sha256(key_material.encode("utf-8")).hexdigest(),
        source_movie_id=boxofficetheory_source_movie_id(
            normalized_movie_title=normalized,
            release_date=release_date,
            published_date=post.published_date,
        ),
        source_movie_title=clean_title,
        normalized_movie_title=normalized,
        distributor=clean_text(distributor or "") or None,
        release_date=release_date,
        market=DOMESTIC_MARKET,
        currency=DOMESTIC_CURRENCY,
        prediction_scope=prediction_scope,
        forecast_metric=forecast_metric,
        opening_weekend_low_usd=opening_weekend_low_usd,
        opening_weekend_high_usd=opening_weekend_high_usd,
        opening_weekend_pinpoint_usd=opening_weekend_pinpoint_usd,
        opening_weekend_day_count=opening_weekend_day_count,
        alternate_opening_weekend_usd=alternate_opening_weekend_usd,
        alternate_opening_weekend_day_count=alternate_opening_weekend_day_count,
        domestic_total_low_usd=domestic_total_low_usd,
        domestic_total_high_usd=domestic_total_high_usd,
        domestic_total_pinpoint_usd=domestic_total_pinpoint_usd,
        weekend_forecast_usd=weekend_forecast_usd,
        weekend_day_count=weekend_day_count,
        projected_domestic_total_usd=projected_domestic_total_usd,
        percent_change=percent_change,
        change_label=clean_text(change_label or "") or None,
        location_count=location_count,
        domestic_multiplier_pinpoint=domestic_multiplier_pinpoint,
        prediction_made_at=post.published_at,
        prediction_made_date=post.published_date,
        raw_forecast_text=raw_forecast_text,
        source_context=source_context,
        parser_version=PARSER_VERSION,
        row_ordinal=row_ordinal,
    )


def find_header(headers: list[str], choices: set[str]) -> int | None:
    compact_choices = {choice.replace(" ", "") for choice in choices}
    for index, header in enumerate(headers):
        if header in choices or header.replace(" ", "") in compact_choices:
            return index
    return None


def find_header_containing(headers: list[str], required_words: list[str]) -> int | None:
    for index, header in enumerate(headers):
        words = set(header.split())
        compact = header.replace(" ", "")
        if all(word in words or word in compact for word in required_words):
            return index
    return None


def find_alternate_opening_index(headers: list[str]) -> int | None:
    for index, header in enumerate(headers):
        if "opening" in header and ("4 day" in header or "5 day" in header or "wtfss" in header or "fssm" in header):
            return index
    return None


def cell_text(row: list[TableCell], index: int | None) -> str:
    if index is None or index < 0 or len(row) <= index:
        return ""
    return clean_text(row[index].text)


def raw_row_text(headers: list[TableCell], row: list[TableCell]) -> str:
    parts: list[str] = []
    for index, cell in enumerate(row):
        header = headers[index].text if index < len(headers) else f"column_{index + 1}"
        parts.append(f"{header}: {cell.text}")
    return " | ".join(parts)


def normalize_header(value: str) -> str:
    text = clean_text(value).lower().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def clean_movie_title(value: str) -> str:
    text = clean_multiline_text(value).replace("\n", " ")
    return clean_text(re.sub(r"\s+\((?:wide|limited|re-issue|reissue)\)\s*$", "", text, flags=re.IGNORECASE))


def normalize_movie_title(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"\s*\(\d{4}\)\s*$", "", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def boxofficetheory_source_movie_id(
    *,
    normalized_movie_title: str,
    release_date: str | None,
    published_date: str,
) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", normalized_movie_title).strip("-") or "unknown-title"
    date_key = release_date or published_date
    return f"boxofficetheory:{DOMESTIC_MARKET}:{slug}:{date_key}"


def parse_release_date(value: str, published_date: str) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    match = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", text)
    if match:
        year = int(match.group(3))
        if year < 100:
            year += 2000
        return date_or_none(year, int(match.group(1)), int(match.group(2)))
    match = re.search(r"\b(\d{1,2})/(\d{1,2})\b", text)
    if match:
        return date_or_none(int(published_date[:4]), int(match.group(1)), int(match.group(2)))
    match = re.search(r"\b([A-Za-z]+)\s+(\d{1,2})(?:,\s*(20\d{2}))?\b", text)
    if match:
        year = int(match.group(3)) if match.group(3) else int(published_date[:4])
        month = month_number(match.group(1))
        if month is not None:
            return date_or_none(year, month, int(match.group(2)))
    return None


def date_or_none(year: int, month: int, day: int) -> str | None:
    try:
        return dt.date(year, month, day).isoformat()
    except ValueError:
        return None


def month_number(value: str) -> int | None:
    for fmt in ("%B", "%b"):
        try:
            return dt.datetime.strptime(value[:3] if fmt == "%b" else value, fmt).month
        except ValueError:
            continue
    return None


def first_money(value: str) -> int | None:
    parsed = parse_money_range(value)
    return parsed.low_usd


def parse_money_range(value: str) -> ParsedMoney:
    text = clean_text(value).replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    text = text.replace("—", "-")
    if "$" not in text:
        return ParsedMoney(None, None)
    match = re.search(
        r"\$\s*(\d[\d,]*(?:\.\d+)?)\s*([kmbKMB]|million|billion)?\s*(?:-|to)\s*\$?\s*(\d[\d,]*(?:\.\d+)?)\s*([kmbKMB]|million|billion)?\+?",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        low_unit = match.group(2)
        high_unit = match.group(4)
        unit = high_unit or low_unit or infer_money_unit(text)
        low = money_to_usd(match.group(1), low_unit or unit)
        high = money_to_usd(match.group(3), high_unit or unit)
        return ParsedMoney(low, high, text.rstrip().endswith("+"))
    match = re.search(r"\$\s*(\d[\d,]*(?:\.\d+)?)\s*([kmbKMB]|million|billion)?\+?", text, flags=re.IGNORECASE)
    if not match:
        return ParsedMoney(None, None)
    value_usd = money_to_usd(match.group(1), match.group(2) or infer_money_unit(text))
    return ParsedMoney(value_usd, None, text.rstrip().endswith("+"))


def infer_money_unit(text: str) -> str:
    lowered = text.lower()
    if "billion" in lowered:
        return "b"
    if "million" in lowered:
        return "m"
    return ""


def money_to_usd(number: str, unit: str | None) -> int | None:
    try:
        amount = float(number.replace(",", ""))
    except ValueError:
        return None
    normalized = (unit or "").lower()
    multiplier = {
        "": 1,
        "k": 1_000,
        "m": 1_000_000,
        "million": 1_000_000,
        "b": 1_000_000_000,
        "billion": 1_000_000_000,
    }.get(normalized)
    if multiplier is None:
        return None
    return int(round(amount * multiplier))


def parse_pinpoint_with_alternate(value: str) -> tuple[int | None, int | None, int | None]:
    primary = first_money(value)
    alternate = None
    alternate_days = None
    parenthetical = re.search(r"\(([^)]*\$[^)]*)\)", value)
    if parenthetical:
        alternate = first_money(parenthetical.group(1))
        alternate_days = day_count_from_text(parenthetical.group(1))
    return primary, alternate, alternate_days


def day_count_from_text(value: str) -> int | None:
    match = re.search(r"\b([345])\s*[- ]?\s*day\b", value, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    if re.search(r"\bFSS\b", value, flags=re.IGNORECASE):
        return 3
    if re.search(r"\bFSSM\b", value, flags=re.IGNORECASE):
        return 4
    if re.search(r"\bWTFSS\b", value, flags=re.IGNORECASE):
        return 5
    return None


def parse_percent_change(value: str) -> float | None:
    match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%", clean_text(value))
    return float(match.group(1)) if match else None


def parse_location_count(value: str) -> int | None:
    text = clean_text(value).replace("~", "").replace(",", "")
    match = re.search(r"\d+", text)
    return int(match.group(0)) if match else None


def parse_float(value: str) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", clean_text(value))
    return float(match.group(0)) if match else None


def initialize_database(conn: Any) -> None:
    movie_identity.ensure_movie_identity_schema(conn)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS boxofficetheory_posts (
            post_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            source_post_id BIGINT NOT NULL UNIQUE,
            post_url TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            author TEXT,
            published_at TIMESTAMPTZ NOT NULL,
            published_date DATE NOT NULL,
            excerpt TEXT,
            source_url TEXT NOT NULL,
            status TEXT NOT NULL,
            fetched_at TIMESTAMPTZ,
            raw_cache_path TEXT,
            sha256 TEXT,
            parser_version TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS boxofficetheory_predictions (
            prediction_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            post_id BIGINT NOT NULL REFERENCES boxofficetheory_posts(post_id),
            source_row_key TEXT NOT NULL,
            source_movie_id TEXT NOT NULL,
            source_movie_title TEXT NOT NULL,
            normalized_movie_title TEXT NOT NULL,
            distributor TEXT,
            release_date DATE,
            market TEXT NOT NULL,
            currency TEXT NOT NULL,
            prediction_scope TEXT NOT NULL,
            forecast_metric TEXT NOT NULL,
            opening_weekend_low_usd BIGINT,
            opening_weekend_high_usd BIGINT,
            opening_weekend_pinpoint_usd BIGINT,
            opening_weekend_day_count INTEGER,
            alternate_opening_weekend_usd BIGINT,
            alternate_opening_weekend_day_count INTEGER,
            domestic_total_low_usd BIGINT,
            domestic_total_high_usd BIGINT,
            domestic_total_pinpoint_usd BIGINT,
            weekend_forecast_usd BIGINT,
            weekend_day_count INTEGER,
            projected_domestic_total_usd BIGINT,
            percent_change DOUBLE PRECISION,
            change_label TEXT,
            location_count INTEGER,
            domestic_multiplier_pinpoint DOUBLE PRECISION,
            prediction_made_at TIMESTAMPTZ NOT NULL,
            prediction_made_date DATE NOT NULL,
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
            UNIQUE(post_id, source_row_key)
        );

        ALTER TABLE boxofficetheory_predictions
            ADD COLUMN IF NOT EXISTS movie_id BIGINT REFERENCES movies(movie_id);

        CREATE TABLE IF NOT EXISTS boxofficetheory_ingest_issues (
            issue_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            issue_source TEXT NOT NULL,
            issue_type TEXT NOT NULL,
            post_url TEXT,
            source_movie_title TEXT,
            details TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(issue_source, issue_type, post_url, source_movie_title, details)
        );

        CREATE INDEX IF NOT EXISTS idx_boxofficetheory_predictions_movie_id
            ON boxofficetheory_predictions(movie_id);
        CREATE INDEX IF NOT EXISTS idx_boxofficetheory_predictions_title
            ON boxofficetheory_predictions(normalized_movie_title);
        CREATE INDEX IF NOT EXISTS idx_boxofficetheory_predictions_release_date
            ON boxofficetheory_predictions(release_date);
        """
    )


def upsert_post(
    conn: Any,
    post: PostRecord,
    *,
    status: str,
    fetched_at: str | None,
    raw_cache_path: Path | None,
    raw_json: str | None,
) -> int:
    sha256 = hashlib.sha256(raw_json.encode("utf-8")).hexdigest() if raw_json is not None else None
    conn.execute(
        """
        INSERT INTO boxofficetheory_posts (
            source_post_id, post_url, title, author, published_at, published_date,
            excerpt, source_url, status, fetched_at, raw_cache_path, sha256,
            parser_version, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT(source_post_id) DO UPDATE SET
            post_url = excluded.post_url,
            title = excluded.title,
            author = excluded.author,
            published_at = excluded.published_at,
            published_date = excluded.published_date,
            excerpt = excluded.excerpt,
            source_url = excluded.source_url,
            status = excluded.status,
            fetched_at = COALESCE(excluded.fetched_at, boxofficetheory_posts.fetched_at),
            raw_cache_path = COALESCE(excluded.raw_cache_path, boxofficetheory_posts.raw_cache_path),
            sha256 = COALESCE(excluded.sha256, boxofficetheory_posts.sha256),
            parser_version = excluded.parser_version,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            post.source_post_id,
            post.post_url,
            post.title,
            post.author,
            post.published_at,
            post.published_date,
            post.excerpt,
            post.source_url,
            status,
            fetched_at,
            str(raw_cache_path) if raw_cache_path is not None else None,
            sha256,
            PARSER_VERSION,
        ),
    )
    return int(
        conn.execute(
            "SELECT post_id FROM boxofficetheory_posts WHERE source_post_id = %s",
            (post.source_post_id,),
        ).fetchone()[0]
    )


def insert_predictions(
    conn: Any,
    post_id: int,
    predictions: list[TheoryPrediction],
    *,
    fetched_at: str,
    raw_cache_path: Path,
) -> None:
    if not predictions:
        return
    conn.executemany(
        """
        INSERT INTO boxofficetheory_predictions (
            post_id, source_row_key, source_movie_id, source_movie_title,
            normalized_movie_title, distributor, release_date, market, currency,
            prediction_scope, forecast_metric, opening_weekend_low_usd,
            opening_weekend_high_usd, opening_weekend_pinpoint_usd,
            opening_weekend_day_count, alternate_opening_weekend_usd,
            alternate_opening_weekend_day_count, domestic_total_low_usd,
            domestic_total_high_usd, domestic_total_pinpoint_usd,
            weekend_forecast_usd, weekend_day_count, projected_domestic_total_usd,
            percent_change, change_label, location_count, domestic_multiplier_pinpoint,
            prediction_made_at, prediction_made_date, raw_forecast_text, source_context,
            parser_version, row_ordinal, movie_id, match_status, match_method,
            match_score, match_notes, fetched_at, raw_cache_path
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(post_id, source_row_key) DO UPDATE SET
            source_movie_id = excluded.source_movie_id,
            source_movie_title = excluded.source_movie_title,
            normalized_movie_title = excluded.normalized_movie_title,
            distributor = excluded.distributor,
            release_date = excluded.release_date,
            market = excluded.market,
            currency = excluded.currency,
            prediction_scope = excluded.prediction_scope,
            forecast_metric = excluded.forecast_metric,
            opening_weekend_low_usd = excluded.opening_weekend_low_usd,
            opening_weekend_high_usd = excluded.opening_weekend_high_usd,
            opening_weekend_pinpoint_usd = excluded.opening_weekend_pinpoint_usd,
            opening_weekend_day_count = excluded.opening_weekend_day_count,
            alternate_opening_weekend_usd = excluded.alternate_opening_weekend_usd,
            alternate_opening_weekend_day_count = excluded.alternate_opening_weekend_day_count,
            domestic_total_low_usd = excluded.domestic_total_low_usd,
            domestic_total_high_usd = excluded.domestic_total_high_usd,
            domestic_total_pinpoint_usd = excluded.domestic_total_pinpoint_usd,
            weekend_forecast_usd = excluded.weekend_forecast_usd,
            weekend_day_count = excluded.weekend_day_count,
            projected_domestic_total_usd = excluded.projected_domestic_total_usd,
            percent_change = excluded.percent_change,
            change_label = excluded.change_label,
            location_count = excluded.location_count,
            domestic_multiplier_pinpoint = excluded.domestic_multiplier_pinpoint,
            prediction_made_at = excluded.prediction_made_at,
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
                post_id,
                prediction.source_row_key,
                prediction.source_movie_id,
                prediction.source_movie_title,
                prediction.normalized_movie_title,
                prediction.distributor,
                prediction.release_date,
                prediction.market,
                prediction.currency,
                prediction.prediction_scope,
                prediction.forecast_metric,
                prediction.opening_weekend_low_usd,
                prediction.opening_weekend_high_usd,
                prediction.opening_weekend_pinpoint_usd,
                prediction.opening_weekend_day_count,
                prediction.alternate_opening_weekend_usd,
                prediction.alternate_opening_weekend_day_count,
                prediction.domestic_total_low_usd,
                prediction.domestic_total_high_usd,
                prediction.domestic_total_pinpoint_usd,
                prediction.weekend_forecast_usd,
                prediction.weekend_day_count,
                prediction.projected_domestic_total_usd,
                prediction.percent_change,
                prediction.change_label,
                prediction.location_count,
                prediction.domestic_multiplier_pinpoint,
                prediction.prediction_made_at,
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
    post_url: str | None,
    source_movie_title: str | None,
    details: str,
) -> None:
    conn.execute(
        """
        INSERT INTO boxofficetheory_ingest_issues (
            issue_source, issue_type, post_url, source_movie_title, details
        ) VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """,
        (issue_source, issue_type, post_url, source_movie_title, details),
    )


def clear_post_issues(conn: Any, *, issue_source: str, post_url: str) -> None:
    conn.execute(
        """
        DELETE FROM boxofficetheory_ingest_issues
        WHERE issue_source = %s
          AND post_url = %s
        """,
        (issue_source, post_url),
    )


def post_already_parsed(conn: Any, source_post_id: int) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM boxofficetheory_posts
        WHERE source_post_id = %s
          AND status = 'parsed'
        LIMIT 1
        """,
        (source_post_id,),
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


def match_predictions(conn: Any, predictions: list[TheoryPrediction]) -> list[TheoryPrediction]:
    candidates = load_movie_candidates(conn)
    matched: list[TheoryPrediction] = []
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
    prediction: TheoryPrediction,
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
    if prediction.release_date is not None:
        exact = [candidate for candidate in matches if candidate.release_date == prediction.release_date]
        if exact:
            candidate = preferred_movie_candidate(exact)
            status = "matched" if candidate.movie_url is not None else "provisional"
            return MovieMatch(candidate.movie_id, status, "normalized_exact_release_date", 1.0, None)
    if len(matches) == 1:
        candidate = matches[0]
        status = "matched" if candidate.movie_url is not None else "provisional"
        return MovieMatch(candidate.movie_id, status, "normalized_exact", 1.0, None)
    return MovieMatch(None, "ambiguous", "normalized_exact", 0.5, "Multiple movies share the title")


def find_source_id_match(conn: Any, prediction: TheoryPrediction) -> MovieMatch | None:
    if not relation_exists(conn, "movie_source_ids"):
        return None
    row = conn.execute(
        """
        SELECT m.movie_id, m.movie_url, src.match_status, src.match_score
        FROM movie_source_ids src
        JOIN movies m ON m.movie_id = src.movie_id
        WHERE src.source = 'boxofficetheory'
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
        "boxofficetheory_source_id",
        float(row[3]) if row[3] is not None else 1.0,
        f"Matched existing Box Office Theory source id {prediction.source_movie_id}",
    )


def can_provision_movie(prediction: TheoryPrediction) -> bool:
    return prediction.release_date is not None and prediction.prediction_scope == "pre_release_tracking"


def provision_movie(conn: Any, prediction: TheoryPrediction) -> MovieMatch:
    row = conn.execute(
        """
        INSERT INTO movies (title, release_year, release_date, updated_at)
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        RETURNING movie_id
        """,
        (
            prediction.source_movie_title,
            int(prediction.release_date[:4]) if prediction.release_date else None,
            prediction.release_date,
        ),
    ).fetchone()
    movie_id = int(row[0])
    upsert_movie_source_id(
        conn,
        movie_id=movie_id,
        prediction=prediction,
        match_status="provisional",
        match_method="provisional_boxofficetheory_identity",
        match_score=1.0,
    )
    return MovieMatch(movie_id, "provisional", "provisional_boxofficetheory_identity", 1.0, None)


def upsert_movie_source_id(
    conn: Any,
    *,
    movie_id: int,
    prediction: TheoryPrediction,
    match_status: str,
    match_method: str | None,
    match_score: float | None,
) -> None:
    movie_identity.upsert_movie_source_id(
        conn,
        movie_id=movie_id,
        source=movie_identity.SOURCE_BOXOFFICETHEORY,
        source_movie_id=prediction.source_movie_id,
        source_title=prediction.source_movie_title,
        match_status=match_status,
        match_method=match_method,
        match_score=match_score,
    )


def preferred_movie_candidate(candidates: list[MovieCandidate]) -> MovieCandidate:
    return sorted(candidates, key=lambda candidate: (candidate.movie_url is None, candidate.movie_id))[0]


def discover_posts(fetcher: CachedJsonFetcher, args: argparse.Namespace) -> list[tuple[PostRecord, Path, str]]:
    discovered: list[tuple[PostRecord, Path, str]] = []
    total_pages = None
    page = 1
    while total_pages is None or page <= total_pages:
        url = api_posts_url(page=page, per_page=args.per_page, category_id=args.category_id)
        print(f"Reading Box Office Theory API page {page} {url}", file=sys.stderr)
        raw_json, cache_path, _fetched, headers = fetcher.get(url)
        total_pages = int(headers.get("x-wp-totalpages") or total_pages or page)
        posts = parse_post_records(raw_json, source_url=url)
        if not posts:
            break
        for post in posts:
            published = dt.date.fromisoformat(post.published_date)
            if args.start_date <= published <= args.end_date:
                discovered.append((post, cache_path, raw_json))
        page += 1
        if args.max_pages is not None and page > args.max_pages:
            break
    discovered.sort(key=lambda item: (item[0].published_at, item[0].source_post_id))
    if args.max_articles is not None:
        discovered = discovered[: args.max_articles]
    return discovered


def import_parse_result(conn: Any, result: PageParseResult, *, issue_source: str) -> tuple[int, int]:
    if result.error is not None:
        upsert_post(
            conn,
            result.post,
            status="post_unavailable",
            fetched_at=result.fetched_at,
            raw_cache_path=result.raw_cache_path,
            raw_json=result.raw_json,
        )
        insert_issue(
            conn,
            issue_source=issue_source,
            issue_type="post_unavailable",
            post_url=result.post.post_url,
            source_movie_title=None,
            details=result.error,
        )
        conn.commit()
        return 1, 0
    post_id = upsert_post(
        conn,
        result.post,
        status="parsed",
        fetched_at=result.fetched_at,
        raw_cache_path=result.raw_cache_path,
        raw_json=result.raw_json,
    )
    predictions = match_predictions(conn, result.predictions)
    clear_post_issues(conn, issue_source=issue_source, post_url=result.post.post_url)
    if not predictions and looks_like_forecast_post(result.post):
        insert_issue(
            conn,
            issue_source=issue_source,
            issue_type="no_predictions_parsed",
            post_url=result.post.post_url,
            source_movie_title=None,
            details=f"No public prediction table or BOT range parsed from {result.post.title}",
        )
    insert_predictions(conn, post_id, predictions, fetched_at=result.fetched_at, raw_cache_path=result.raw_cache_path)
    conn.commit()
    return 1, len(predictions)


def looks_like_forecast_post(post: PostRecord) -> bool:
    title = post.title.lower()
    return "forecast" in title or "tracking" in title


def configure_full_refresh_args(args: argparse.Namespace) -> None:
    if not getattr(args, "full_refresh", False):
        return
    args.refresh = True
    args.start_date = FULL_REFRESH_START_DATE
    args.end_date = FULL_REFRESH_END_DATE
    args.max_pages = None


def validate_args(args: argparse.Namespace) -> None:
    if args.end_date < args.start_date:
        raise SystemExit("--end-date must be on or after --start-date")
    if args.delay_seconds < MIN_DELAY_SECONDS and not args.offline and not args.dry_run:
        raise SystemExit(f"--delay-seconds must be at least {MIN_DELAY_SECONDS:g}")
    if "bot" not in args.user_agent.lower() and not args.offline and not args.dry_run:
        raise SystemExit("--user-agent must identify the scraper as a bot")
    if args.per_page < 1 or args.per_page > 100:
        raise SystemExit("--per-page must be between 1 and 100")


def run(args: argparse.Namespace) -> int:
    configure_full_refresh_args(args)
    validate_args(args)
    if args.print_cache_paths:
        fetcher = CachedJsonFetcher(
            args.cache_dir,
            refresh=False,
            offline=True,
            delay_seconds=args.delay_seconds,
            user_agent=args.user_agent,
        )
        for page in range(1, (args.max_pages or 3) + 1):
            url = api_posts_url(page=page, per_page=args.per_page, category_id=args.category_id)
            print(f"{url}\t{fetcher.cache_path(url)}")
        return 0
    if args.dry_run:
        for page in range(1, (args.max_pages or 3) + 1):
            print(api_posts_url(page=page, per_page=args.per_page, category_id=args.category_id))
        return 0

    fetcher = CachedJsonFetcher(
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
        discovered = discover_posts(fetcher, args)
        imported_posts = 0
        imported_predictions = 0
        skipped_posts = 0
        for index, (post, cache_path, raw_json) in enumerate(discovered, start=1):
            if not args.refresh and post_already_parsed(conn, post.source_post_id):
                skipped_posts += 1
                print(f"Skipping parsed post {index}/{len(discovered)} {post.title}", file=sys.stderr)
                continue
            print(f"Parsing post {index}/{len(discovered)} {post.title}", file=sys.stderr)
            fetched_at = dt.datetime.now(dt.UTC).isoformat()
            try:
                predictions = parse_predictions(post)
                result = PageParseResult(post, predictions, fetched_at, cache_path, raw_json)
            except Exception as exc:  # pragma: no cover - defensive per-post failure handling.
                result = PageParseResult(post, [], fetched_at, cache_path, raw_json, error=str(exc))
            post_count, prediction_count = import_parse_result(conn, result, issue_source=args.issue_source)
            imported_posts += post_count
            imported_predictions += prediction_count
        print(
            f"Imported {imported_posts} Box Office Theory posts and {imported_predictions} predictions; "
            f"skipped {skipped_posts} parsed posts.",
            file=sys.stderr,
        )
    finally:
        fetcher.close()
        conn.close()
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import Box Office Theory public movie box office predictions.")
    parser.add_argument("--start-date", type=parse_date_arg, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=parse_date_arg, default=DEFAULT_END_DATE)
    parser.add_argument(
        "--database-url",
        default=database_url_from_env(),
        help="PostgreSQL connection URL. Defaults to DATABASE_URL or POSTGRES_DSN.",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Raw JSON cache directory.")
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=MIN_DELAY_SECONDS,
        help="Delay between uncached API requests. Must be at least 5.",
    )
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="HTTP User-Agent. Must identify as a bot.")
    parser.add_argument("--refresh", action="store_true", help="Reparse even when post status is parsed.")
    parser.add_argument("--full-refresh", action="store_true", help="Reparse every Tracking & Forecasts post.")
    parser.add_argument("--offline", action="store_true", help="Require all API pages to exist in cache.")
    parser.add_argument("--dry-run", action="store_true", help="Print discovery URLs and exit.")
    parser.add_argument(
        "--print-cache-paths",
        action="store_true",
        help="Print expected cache paths for API discovery URLs, then exit.",
    )
    parser.add_argument("--category-id", type=int, default=TRACKING_FORECASTS_CATEGORY_ID)
    parser.add_argument("--per-page", type=int, default=100)
    parser.add_argument("--max-pages", type=int, help="Optional API page cap for smoke tests.")
    parser.add_argument("--max-articles", type=int, help="Optional post cap after discovery.")
    parser.add_argument("--issue-source", default="boxofficetheory_prediction_import")
    return parser


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
