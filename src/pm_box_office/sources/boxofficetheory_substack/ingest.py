#!/usr/bin/env python3
"""Ingest Box Office Theory Substack prediction posts into PostgreSQL."""

from __future__ import annotations

import argparse
import datetime as dt
from email.utils import parsedate_to_datetime
import hashlib
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from pm_box_office.db.connection import connect_database, database_url_from_env
from pm_box_office.domain import movies as movie_identity
from pm_box_office.sources.boxofficetheory import ingest as theory
from pm_box_office.sources.common.cli import parse_date_arg
from pm_box_office.sources.common.fetch import CacheFirstFetcher
from pm_box_office.sources.common.ocr import OcrResult, OcrToken, TesseractOcr
from pm_box_office.sources.common.schema import acquire_schema_init_lock


BASE_URL = "https://boxofficetheory.substack.com"
FEED_URL = f"{BASE_URL}/feed"
ARCHIVE_API_URL = f"{BASE_URL}/api/v1/archive"
DEFAULT_CACHE_DIR = Path("data/raw/boxofficetheory_substack")
DEFAULT_START_DATE = dt.date(2026, 6, 1)
DEFAULT_END_DATE = dt.date(2026, 6, 30)
FULL_REFRESH_START_DATE = dt.date(1900, 1, 1)
FULL_REFRESH_END_DATE = dt.date(9999, 12, 31)
DEFAULT_USER_AGENT = "pm-box-office-boxofficetheory-substack-bot/1.0 (+personal research; set --user-agent contact)"
MIN_DELAY_SECONDS = 5.0
PARSER_VERSION = "boxofficetheory_substack_predictions_v2"
DEFAULT_PER_PAGE = 20
DEFAULT_MIN_OCR_CONFIDENCE = 50.0
CHART_IMAGE_EXTENSIONS = {".gif", ".jpg", ".jpeg", ".png", ".webp"}
CHART_IMAGE_TERMS = ("tracking", "forecast", "forecasts", "prediction", "predictions", "projection", "projections")
RSS_NAMESPACES = {
    "content": "http://purl.org/rss/1.0/modules/content/",
    "dc": "http://purl.org/dc/elements/1.1/",
}


@dataclass(frozen=True)
class SubstackChartImage:
    post_url: str
    post_title: str
    published_date: str
    image_url: str
    width: int | None
    height: int | None
    alt_text: str
    title: str
    context: str
    ordinal: int


@dataclass(frozen=True)
class OcrChartRow:
    image_url: str
    row_ordinal: int
    title: str
    distributor: str | None
    release_date: str | None
    opening_weekend_low_usd: int | None
    opening_weekend_high_usd: int | None
    opening_weekend_pinpoint_usd: int | None
    domestic_total_low_usd: int | None
    domestic_total_high_usd: int | None
    domestic_total_pinpoint_usd: int | None
    percent_change: float | None
    domestic_multiplier_pinpoint: float | None
    raw_text: str


class FetchBlocked(RuntimeError):
    """Raised when feed data cannot be fetched and no cache is available."""


class CachedFeedFetcher:
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
        suffix = ".json" if "/api/" in urllib.parse.urlsplit(url).path else ".xml"
        return self.cache_dir / f"{digest}{suffix}"

    def get(self, url: str) -> tuple[str, Path, bool]:
        cache_path = self.cache_path(url)
        if cache_path.exists() and (not self.refresh or self.offline):
            return cache_path.read_text(encoding="utf-8-sig"), cache_path, False
        if self.offline:
            raise FetchBlocked(f"Cache miss in offline mode: {url}\nExpected cached RSS at: {cache_path}")
        self._wait()
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json,application/rss+xml,application/xml,text/xml,text/html",
                "User-Agent": self.user_agent,
            },
        )
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


class CachedImageFetcher(CacheFirstFetcher):
    def __init__(
        self,
        cache_dir: Path,
        *,
        refresh: bool,
        offline: bool,
        delay_seconds: float,
        user_agent: str,
    ) -> None:
        super().__init__(
            cache_dir,
            refresh=refresh,
            offline=offline,
            delay_seconds=delay_seconds,
            user_agent=user_agent,
            retries=2,
            default_accept="image/avif,image/webp,image/png,image/jpeg,image/*,*/*",
            offline_error_prefix="Cache miss in offline mode for Substack chart image",
        )

    def get(self, url: str) -> tuple[bytes, Path, bool]:
        suffix = Path(urllib.parse.urlsplit(url).path).suffix or ".img"
        return self.get_bytes(url, suffix=suffix)


def parse_rss_posts(xml_text: str, *, source_url: str) -> list[theory.PostRecord]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"invalid Box Office Theory Substack RSS from {source_url}: {exc}") from exc
    posts: list[theory.PostRecord] = []
    for item in root.findall("./channel/item"):
        link = text_or_none(item.find("link"))
        published = text_or_none(item.find("pubDate"))
        if not link or not published:
            continue
        published_at = parse_rss_datetime(published)
        post_url = theory.canonical_url(link)
        title = text_or_none(item.find("title")) or ""
        description = text_or_none(item.find("description")) or ""
        content_html = text_or_none(item.find("content:encoded", RSS_NAMESPACES)) or description
        posts.append(
            theory.PostRecord(
                source_post_id=substack_source_post_id(item, post_url),
                post_url=post_url,
                title=theory.clean_text(title),
                author=text_or_none(item.find("dc:creator", RSS_NAMESPACES)) or "Shawn Robbins",
                published_at=published_at.isoformat(),
                published_date=published_at.date().isoformat(),
                excerpt=theory.strip_html_text(description) or None,
                content_html=content_html,
                source_url=source_url,
            )
        )
    return posts


def parse_archive_items(raw_json: str, *, source_url: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid Box Office Theory Substack archive JSON from {source_url}: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"expected list from Box Office Theory Substack archive: {source_url}")
    return [item for item in payload if isinstance(item, dict)]


def parse_post_record_json(raw_json: str, *, source_url: str) -> theory.PostRecord:
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid Box Office Theory Substack post JSON from {source_url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected object from Box Office Theory Substack post: {source_url}")
    slug = str(payload.get("slug") or "").strip()
    post_id = payload.get("id")
    post_url = theory.canonical_url(str(payload.get("canonical_url") or f"{BASE_URL}/p/{slug}"))
    published_at = parse_iso_datetime(str(payload.get("post_date") or ""))
    description = str(payload.get("description") or payload.get("subtitle") or "")
    body_html = str(payload.get("body_html") or "")
    return theory.PostRecord(
        source_post_id=slug or str(post_id or substack_url_slug(post_url)),
        post_url=post_url,
        title=theory.clean_text(str(payload.get("title") or "")),
        author=published_author(payload) or "Shawn Robbins",
        published_at=published_at.isoformat(),
        published_date=published_at.date().isoformat(),
        excerpt=theory.strip_html_text(description) or None,
        content_html=body_html or description,
        source_url=source_url,
    )


def published_author(payload: dict[str, Any]) -> str | None:
    bylines = payload.get("publishedBylines")
    if not isinstance(bylines, list) or not bylines:
        return None
    first = bylines[0]
    if not isinstance(first, dict):
        return None
    name = first.get("name")
    return theory.clean_text(str(name)) if name else None


def text_or_none(element: ET.Element[str] | None) -> str | None:
    if element is None or element.text is None:
        return None
    return element.text.strip()


def parse_rss_datetime(value: str) -> dt.datetime:
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return dt.datetime.now(dt.UTC)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def parse_iso_datetime(value: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return dt.datetime.now(dt.UTC)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def substack_source_post_id(item: ET.Element[str], post_url: str) -> str:
    guid = text_or_none(item.find("guid"))
    value = guid or post_url
    parsed = urllib.parse.urlsplit(value)
    if parsed.path:
        slug = parsed.path.rstrip("/").rsplit("/", 1)[-1]
        if slug:
            return slug
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def substack_url_slug(post_url: str) -> str:
    parsed = urllib.parse.urlsplit(post_url)
    return parsed.path.rstrip("/").rsplit("/", 1)[-1] if parsed.path else ""


def archive_url(*, offset: int, limit: int) -> str:
    query = urllib.parse.urlencode({"sort": "new", "offset": str(offset), "limit": str(limit)})
    return f"{ARCHIVE_API_URL}?{query}"


def post_api_url(slug: str) -> str:
    return f"{BASE_URL}/api/v1/posts/{urllib.parse.quote(slug)}"


def parse_predictions(post: theory.PostRecord) -> list[theory.TheoryPrediction]:
    predictions = theory.parse_predictions(post)
    return [retag_prediction(prediction) for prediction in predictions]


def parse_image_chart_predictions(
    post: theory.PostRecord,
    *,
    image_fetcher: CachedImageFetcher,
    ocr_engine: TesseractOcr,
    first_row_ordinal: int,
    image_limit: int | None = None,
    min_ocr_confidence: float | None = DEFAULT_MIN_OCR_CONFIDENCE,
) -> list[theory.TheoryPrediction]:
    images = discover_chart_images(post)
    if image_limit is not None and image_limit > 0:
        images = images[:image_limit]
    predictions: list[theory.TheoryPrediction] = []
    for image in images:
        try:
            _body, image_cache_path, _fetched = image_fetcher.get(image.image_url)
            ocr = ocr_engine.read(image_cache_path)
            if min_ocr_confidence is not None and ocr.mean_confidence is not None and ocr.mean_confidence < min_ocr_confidence:
                print(
                    f"Skipping low-confidence Substack chart OCR {ocr.mean_confidence:.1f} {image.image_url}",
                    file=sys.stderr,
                )
                continue
            rows = parse_tracking_chart_ocr_rows(
                post,
                image=image,
                ocr=ocr,
                first_row_ordinal=first_row_ordinal + len(predictions),
            )
        except Exception as exc:  # pragma: no cover - OCR failures are non-fatal during long backfills.
            print(f"Skipping Substack chart image OCR for {image.image_url}: {exc}", file=sys.stderr)
            continue
        for row in rows:
            prediction = theory.build_prediction(
                post=post,
                title=row.title,
                distributor=row.distributor,
                release_date=row.release_date,
                prediction_scope="pre_release_tracking",
                forecast_metric="domestic_opening_and_total",
                opening_weekend_low_usd=row.opening_weekend_low_usd,
                opening_weekend_high_usd=row.opening_weekend_high_usd,
                opening_weekend_pinpoint_usd=row.opening_weekend_pinpoint_usd,
                opening_weekend_day_count=3,
                domestic_total_low_usd=row.domestic_total_low_usd,
                domestic_total_high_usd=row.domestic_total_high_usd,
                domestic_total_pinpoint_usd=row.domestic_total_pinpoint_usd,
                projected_domestic_total_usd=row.domestic_total_pinpoint_usd,
                percent_change=row.percent_change,
                domestic_multiplier_pinpoint=row.domestic_multiplier_pinpoint,
                raw_forecast_text=row.raw_text,
                source_context=f"ocr_tracking_forecast_image:{row.image_url}",
                row_ordinal=row.row_ordinal,
            )
            predictions.append(retag_prediction(prediction))
    return predictions


def discover_chart_images(post: theory.PostRecord) -> list[SubstackChartImage]:
    seen: set[str] = set()
    images: list[SubstackChartImage] = []
    for ordinal, attrs in enumerate(extract_substack_image_attrs(post.content_html), start=1):
        image = chart_image_from_attrs(post, attrs, ordinal=ordinal)
        if image is None or image.image_url in seen:
            continue
        seen.add(image.image_url)
        images.append(image)
    return images


def extract_substack_image_attrs(html_text: str) -> list[dict[str, Any]]:
    attrs_list: list[dict[str, Any]] = []
    for match in re.finditer(r'data-attrs="([^"]+)"', html_text):
        raw = html.unescape(match.group(1))
        try:
            attrs = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(attrs, dict) and attrs.get("src"):
            attrs_list.append(attrs)
    return attrs_list


def chart_image_from_attrs(post: theory.PostRecord, attrs: dict[str, Any], *, ordinal: int) -> SubstackChartImage | None:
    image_url = str(attrs.get("src") or "").strip()
    if not image_url:
        return None
    parsed = urllib.parse.urlsplit(image_url)
    if Path(parsed.path).suffix.lower() not in CHART_IMAGE_EXTENSIONS:
        return None
    width = int_or_none(attrs.get("width"))
    height = int_or_none(attrs.get("height"))
    if attrs.get("topImage") is True:
        return None
    if width is not None and height is not None and (width < 1000 or height < 250):
        return None
    alt_text = theory.clean_text(str(attrs.get("alt") or ""))
    title = theory.clean_text(str(attrs.get("title") or ""))
    haystack = " ".join([post.title, post.excerpt or "", alt_text, title, Path(parsed.path).name]).lower()
    if not any(term in haystack for term in CHART_IMAGE_TERMS) and not (width and width >= 1200 and height and height >= 350):
        return None
    return SubstackChartImage(
        post_url=post.post_url,
        post_title=post.title,
        published_date=post.published_date,
        image_url=image_url,
        width=width,
        height=height,
        alt_text=alt_text,
        title=title,
        context=post.title,
        ordinal=ordinal,
    )


def parse_tracking_chart_ocr_rows(
    post: theory.PostRecord,
    *,
    image: SubstackChartImage,
    ocr: OcrResult,
    first_row_ordinal: int,
) -> list[OcrChartRow]:
    tokens = positioned_tokens(ocr)
    if not tokens:
        return []
    table_width = max((token.left or 0) + (token.width or 0) for token in tokens)
    anchors = chart_row_anchors(tokens, table_width=table_width)
    rows: list[OcrChartRow] = []
    for anchor in anchors:
        parsed = parse_tracking_chart_row(
            post,
            image=image,
            tokens=tokens,
            table_width=table_width,
            row_top=anchor,
            row_ordinal=first_row_ordinal + len(rows),
        )
        if parsed is not None:
            rows.append(parsed)
    return rows


def positioned_tokens(ocr: OcrResult) -> list[OcrToken]:
    return [
        token
        for token in ocr.tokens
        if token.text and token.left is not None and token.top is not None and token.width is not None and token.height is not None
    ]


def chart_row_anchors(tokens: list[OcrToken], *, table_width: int) -> list[int]:
    money_tokens = [
        token
        for token in tokens
        if "$" in token.text and token.top is not None and token.left is not None and token.left / max(table_width, 1) > 0.30
    ]
    grouped: list[list[OcrToken]] = []
    for token in sorted(money_tokens, key=lambda item: item.top or 0):
        if not grouped or abs((token.top or 0) - median_top(grouped[-1])) > 10:
            grouped.append([token])
        else:
            grouped[-1].append(token)
    anchors: list[int] = []
    for group in grouped:
        if len(group) < 3:
            continue
        anchor = median_top(group)
        if not anchors or abs(anchor - anchors[-1]) > 20:
            anchors.append(anchor)
    return anchors


def parse_tracking_chart_row(
    post: theory.PostRecord,
    *,
    image: SubstackChartImage,
    tokens: list[OcrToken],
    table_width: int,
    row_top: int,
    row_ordinal: int,
) -> OcrChartRow | None:
    cells = {
        "release_date": ocr_column_text(tokens, table_width, row_top, 0.000, 0.075),
        "title": ocr_column_text(tokens, table_width, row_top, 0.075, 0.235, min_confidence=45.0),
        "distributor": ocr_column_text(
            tokens,
            table_width,
            row_top,
            0.235,
            0.340,
            vertical_before=44,
            vertical_after=8,
            min_confidence=35.0,
        ),
        "opening_low": ocr_column_text(tokens, table_width, row_top, 0.340, 0.415),
        "opening_high": ocr_column_text(tokens, table_width, row_top, 0.415, 0.490),
        "opening_pinpoint": ocr_column_text(tokens, table_width, row_top, 0.490, 0.570),
        "opening_change": ocr_column_text(tokens, table_width, row_top, 0.570, 0.625),
        "total_low": ocr_column_text(tokens, table_width, row_top, 0.625, 0.700),
        "total_high": ocr_column_text(tokens, table_width, row_top, 0.700, 0.775),
        "total_pinpoint": ocr_column_text(tokens, table_width, row_top, 0.775, 0.855),
        "total_change": ocr_column_text(tokens, table_width, row_top, 0.855, 0.910),
        "multiplier": ocr_column_text(tokens, table_width, row_top, 0.910, 1.000),
    }
    title = clean_ocr_title(cells["title"])
    if not title or not any(cells[key] for key in ("opening_low", "opening_high", "opening_pinpoint", "total_pinpoint")):
        return None
    release_date = theory.parse_release_date(cells["release_date"], post.published_date)
    distributor = clean_ocr_distributor(cells["distributor"])
    opening_pinpoint = first_ocr_money(cells["opening_pinpoint"])
    total_low = first_ocr_money(cells["total_low"])
    total_high = first_ocr_money(cells["total_high"])
    total_pinpoint = first_ocr_money(cells["total_pinpoint"])
    multiplier = last_ocr_float(cells["multiplier"])
    raw_text = " | ".join(f"{name}: {value}" for name, value in cells.items() if value)
    return OcrChartRow(
        image_url=image.image_url,
        row_ordinal=row_ordinal,
        title=title,
        distributor=distributor,
        release_date=release_date,
        opening_weekend_low_usd=first_ocr_money(cells["opening_low"]),
        opening_weekend_high_usd=first_ocr_money(cells["opening_high"]),
        opening_weekend_pinpoint_usd=opening_pinpoint,
        domestic_total_low_usd=total_low,
        domestic_total_high_usd=total_high,
        domestic_total_pinpoint_usd=reconcile_total_pinpoint(
            opening_pinpoint=opening_pinpoint,
            total_low=total_low,
            total_high=total_high,
            total_pinpoint=total_pinpoint,
            multiplier=multiplier,
        ),
        percent_change=theory.parse_percent_change(cells["opening_change"]) or theory.parse_percent_change(cells["total_change"]),
        domestic_multiplier_pinpoint=multiplier,
        raw_text=f"Image: {image.image_url} | {raw_text}",
    )


def ocr_column_text(
    tokens: list[OcrToken],
    table_width: int,
    row_top: int,
    left_ratio: float,
    right_ratio: float,
    *,
    vertical_tolerance: int = 16,
    vertical_before: int | None = None,
    vertical_after: int | None = None,
    min_confidence: float | None = None,
) -> str:
    before = vertical_before if vertical_before is not None else vertical_tolerance
    after = vertical_after if vertical_after is not None else vertical_tolerance
    selected: list[OcrToken] = []
    for token in tokens:
        assert token.left is not None
        assert token.top is not None
        assert token.width is not None
        center_x = token.left + token.width / 2
        ratio = center_x / max(table_width, 1)
        if ratio < left_ratio or ratio >= right_ratio:
            continue
        offset = token.top - row_top
        if offset < -before or offset > after:
            continue
        if min_confidence is not None and token.confidence is not None and token.confidence < min_confidence:
            continue
        selected.append(token)
    selected.sort(key=lambda token: token.left or 0)
    return theory.clean_text(" ".join(token.text for token in selected))


def first_ocr_money(value: str) -> int | None:
    text = value.replace("S", "$").replace("§", "$")
    return theory.first_money(text)


def last_ocr_float(value: str) -> float | None:
    text = theory.clean_text(value)
    matches = re.findall(r"(?<![\d-])(\d+(?:\.\d+)?)(?!\s*%)", text)
    return float(matches[-1]) if matches else None


def reconcile_total_pinpoint(
    *,
    opening_pinpoint: int | None,
    total_low: int | None,
    total_high: int | None,
    total_pinpoint: int | None,
    multiplier: float | None,
) -> int | None:
    if opening_pinpoint is None or multiplier is None or total_low is None or total_high is None:
        return total_pinpoint
    derived = round_money_to_nearest(opening_pinpoint * multiplier, 100_000)
    if total_low <= derived <= total_high and (total_pinpoint is None or not total_low <= total_pinpoint <= total_high):
        return derived
    return total_pinpoint


def round_money_to_nearest(value: float, increment: int) -> int:
    return int(round(value / increment) * increment)


def clean_ocr_title(value: str) -> str:
    text = theory.clean_text(value)
    text = re.sub(r"\bBrandNewDay\b", "Brand New Day", text)
    year_prefix = re.match(r"^\((20\d{2})\)\s+(.+)$", text)
    if year_prefix:
        text = f"{year_prefix.group(2)} ({year_prefix.group(1)})"
    text = re.sub(r"\s+([:;,.])", r"\1", text)
    text = re.sub(r"^[^A-Za-z0-9$]+", "", text)
    return theory.clean_movie_title(text)


def clean_ocr_distributor(value: str) -> str | None:
    text = theory.clean_text(value)
    compact = re.sub(r"[^a-z0-9]+", "", text.lower())
    if "warner" in compact and "bros" in compact:
        return "Warner Bros. Pictures"
    if "universal" in compact:
        return "Universal Pictures"
    if "sony" in compact:
        return "Sony Pictures"
    if "horror" in compact and "section" in compact:
        return "The Horror Section"
    if "disney" in compact:
        return "Disney"
    cleaned = re.sub(r"\b(Pictures|Studios|Section)\s+\1\b", r"\1", text, flags=re.IGNORECASE)
    return theory.clean_text(cleaned) or None


def median_top(tokens: list[OcrToken]) -> int:
    values = sorted(token.top or 0 for token in tokens)
    return values[len(values) // 2]


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def retag_prediction(prediction: theory.TheoryPrediction) -> theory.TheoryPrediction:
    key_material = "|".join(
        [
            prediction.post_url,
            str(prediction.row_ordinal),
            prediction.normalized_movie_title,
            prediction.release_date or "",
            prediction.prediction_scope,
            prediction.forecast_metric,
            str(prediction.opening_weekend_low_usd),
            str(prediction.opening_weekend_high_usd),
            str(prediction.opening_weekend_pinpoint_usd),
            str(prediction.weekend_forecast_usd),
            str(prediction.projected_domestic_total_usd),
            PARSER_VERSION,
        ]
    )
    return replace(
        prediction,
        source_row_key=hashlib.sha256(key_material.encode("utf-8")).hexdigest(),
        parser_version=PARSER_VERSION,
    )


def initialize_database(conn: Any) -> None:
    acquire_schema_init_lock(conn)
    movie_identity.ensure_movie_identity_schema(conn)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS boxofficetheory_substack_posts (
            post_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            source_post_id TEXT NOT NULL UNIQUE,
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

        CREATE TABLE IF NOT EXISTS boxofficetheory_substack_predictions (
            prediction_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            post_id BIGINT NOT NULL REFERENCES boxofficetheory_substack_posts(post_id),
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

        ALTER TABLE boxofficetheory_substack_predictions
            ADD COLUMN IF NOT EXISTS movie_id BIGINT REFERENCES movies(movie_id);

        CREATE TABLE IF NOT EXISTS boxofficetheory_substack_ingest_issues (
            issue_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            issue_source TEXT NOT NULL,
            issue_type TEXT NOT NULL,
            post_url TEXT,
            source_movie_title TEXT,
            details TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(issue_source, issue_type, post_url, source_movie_title, details)
        );

        CREATE INDEX IF NOT EXISTS idx_boxofficetheory_substack_predictions_movie_id
            ON boxofficetheory_substack_predictions(movie_id);
        CREATE INDEX IF NOT EXISTS idx_boxofficetheory_substack_predictions_title
            ON boxofficetheory_substack_predictions(normalized_movie_title);
        CREATE INDEX IF NOT EXISTS idx_boxofficetheory_substack_predictions_release_date
            ON boxofficetheory_substack_predictions(release_date);
        """
    )


def upsert_post(
    conn: Any,
    post: theory.PostRecord,
    *,
    status: str,
    fetched_at: str | None,
    raw_xml: str | None,
    raw_cache_path: Path | None,
) -> int:
    sha256 = hashlib.sha256(raw_xml.encode("utf-8")).hexdigest() if raw_xml is not None else None
    conn.execute(
        """
        INSERT INTO boxofficetheory_substack_posts (
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
            fetched_at = COALESCE(excluded.fetched_at, boxofficetheory_substack_posts.fetched_at),
            raw_cache_path = COALESCE(excluded.raw_cache_path, boxofficetheory_substack_posts.raw_cache_path),
            sha256 = COALESCE(excluded.sha256, boxofficetheory_substack_posts.sha256),
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
            "SELECT post_id FROM boxofficetheory_substack_posts WHERE source_post_id = %s",
            (post.source_post_id,),
        ).fetchone()[0]
    )


def insert_predictions(
    conn: Any,
    post_id: int,
    predictions: list[theory.TheoryPrediction],
    *,
    fetched_at: str,
    raw_cache_path: Path,
) -> None:
    if not predictions:
        return
    conn.executemany(
        """
        INSERT INTO boxofficetheory_substack_predictions (
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
        INSERT INTO boxofficetheory_substack_ingest_issues (
            issue_source, issue_type, post_url, source_movie_title, details
        ) VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """,
        (issue_source, issue_type, post_url, source_movie_title, details),
    )


def clear_post_issues(conn: Any, *, issue_source: str, post_url: str) -> None:
    conn.execute(
        """
        DELETE FROM boxofficetheory_substack_ingest_issues
        WHERE issue_source = %s
          AND post_url = %s
        """,
        (issue_source, post_url),
    )


def post_already_parsed(conn: Any, source_post_id: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM boxofficetheory_substack_posts
        WHERE source_post_id = %s
          AND status = 'parsed'
        LIMIT 1
        """,
        (source_post_id,),
    ).fetchone()
    return row is not None


def discover_posts(fetcher: CachedFeedFetcher, args: argparse.Namespace) -> list[tuple[theory.PostRecord, Path, str]]:
    if args.discovery == "rss":
        return discover_rss_posts(fetcher, args)
    return discover_archive_posts(fetcher, args)


def discover_rss_posts(fetcher: CachedFeedFetcher, args: argparse.Namespace) -> list[tuple[theory.PostRecord, Path, str]]:
    print(f"Reading Box Office Theory Substack RSS {args.feed_url}", file=sys.stderr)
    raw_xml, cache_path, _fetched = fetcher.get(args.feed_url)
    discovered: list[tuple[theory.PostRecord, Path, str]] = []
    for post in parse_rss_posts(raw_xml, source_url=args.feed_url):
        published = dt.date.fromisoformat(post.published_date)
        if args.start_date <= published <= args.end_date:
            discovered.append((post, cache_path, raw_xml))
    discovered.sort(key=lambda item: (item[0].published_at, item[0].source_post_id))
    if args.max_articles is not None:
        discovered = discovered[: args.max_articles]
    return discovered


def discover_archive_posts(fetcher: CachedFeedFetcher, args: argparse.Namespace) -> list[tuple[theory.PostRecord, Path, str]]:
    discovered: list[tuple[theory.PostRecord, Path, str]] = []
    offset = 0
    page = 1
    done = False
    while not done:
        url = archive_url(offset=offset, limit=args.per_page)
        print(f"Reading Box Office Theory Substack archive page {page} {url}", file=sys.stderr)
        raw_json, _archive_cache_path, _fetched = fetcher.get(url)
        items = parse_archive_items(raw_json, source_url=url)
        if not items:
            break
        for item in items:
            post_date = parse_iso_datetime(str(item.get("post_date") or ""))
            published = post_date.date()
            if published < args.start_date:
                done = True
                continue
            if published > args.end_date:
                continue
            slug = str(item.get("slug") or "").strip()
            if not slug:
                continue
            post_url = post_api_url(slug)
            print(f"Reading Box Office Theory Substack post {slug}", file=sys.stderr)
            post_json, post_cache_path, _post_fetched = fetcher.get(post_url)
            post = parse_post_record_json(post_json, source_url=post_url)
            discovered.append((post, post_cache_path, post_json))
            if args.max_articles is not None and len(discovered) >= args.max_articles:
                done = True
                break
        page += 1
        offset += args.per_page
        if args.max_pages is not None and page > args.max_pages:
            break
    discovered.sort(key=lambda item: (item[0].published_at, item[0].source_post_id))
    return discovered


def import_parse_result(
    conn: Any,
    result: theory.PageParseResult,
    *,
    issue_source: str,
) -> tuple[int, int]:
    if result.error is not None:
        upsert_post(
            conn,
            result.post,
            status="post_unavailable",
            fetched_at=result.fetched_at,
            raw_cache_path=result.raw_cache_path,
            raw_xml=result.raw_json,
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
        raw_xml=result.raw_json,
    )
    predictions = theory.match_predictions(conn, result.predictions)
    clear_post_issues(conn, issue_source=issue_source, post_url=result.post.post_url)
    if not predictions and theory.looks_like_forecast_post(result.post):
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


def configure_full_refresh_args(args: argparse.Namespace) -> None:
    if not getattr(args, "full_refresh", False):
        return
    args.refresh = True
    args.start_date = FULL_REFRESH_START_DATE
    args.end_date = FULL_REFRESH_END_DATE
    args.discovery = "archive"
    args.max_pages = None
    args.ocr_image_charts = True


def validate_args(args: argparse.Namespace) -> None:
    if args.end_date < args.start_date:
        raise SystemExit("--end-date must be on or after --start-date")
    if args.delay_seconds < MIN_DELAY_SECONDS and not args.offline and not args.dry_run:
        raise SystemExit(f"--delay-seconds must be at least {MIN_DELAY_SECONDS:g}")
    if "bot" not in args.user_agent.lower() and not args.offline and not args.dry_run:
        raise SystemExit("--user-agent must identify the scraper as a bot")
    if args.per_page < 1 or args.per_page > 100:
        raise SystemExit("--per-page must be between 1 and 100")
    if args.discovery not in {"archive", "rss"}:
        raise SystemExit("--discovery must be one of: archive, rss")
    if args.ocr_image_limit is not None and args.ocr_image_limit < 1:
        raise SystemExit("--ocr-image-limit must be at least 1")
    if args.min_ocr_confidence is not None and args.min_ocr_confidence < 0:
        raise SystemExit("--min-ocr-confidence must be non-negative")


def run(args: argparse.Namespace) -> int:
    configure_full_refresh_args(args)
    validate_args(args)
    if args.print_cache_paths:
        fetcher = CachedFeedFetcher(
            args.cache_dir,
            refresh=False,
            offline=True,
            delay_seconds=args.delay_seconds,
            user_agent=args.user_agent,
        )
        if args.discovery == "rss":
            print(f"{args.feed_url}\t{fetcher.cache_path(args.feed_url)}")
        else:
            url = archive_url(offset=0, limit=args.per_page)
            print(f"{url}\t{fetcher.cache_path(url)}")
        return 0
    if args.dry_run:
        if args.discovery == "rss":
            print(args.feed_url)
        else:
            for page in range(1, (args.max_pages or 3) + 1):
                print(archive_url(offset=(page - 1) * args.per_page, limit=args.per_page))
        return 0

    fetcher = CachedFeedFetcher(
        args.cache_dir,
        refresh=args.refresh,
        offline=args.offline,
        delay_seconds=args.delay_seconds,
        user_agent=args.user_agent,
    )
    image_fetcher = CachedImageFetcher(
        args.cache_dir / "images",
        refresh=args.refresh,
        offline=args.offline,
        delay_seconds=args.delay_seconds,
        user_agent=args.user_agent,
    )
    ocr_engine = TesseractOcr(command=args.tesseract_command, psm=args.tesseract_psm, language=args.tesseract_language)
    conn = connect_database(args.database_url)
    try:
        initialize_database(conn)
        conn.commit()
        discovered = discover_posts(fetcher, args)
        imported_posts = 0
        imported_predictions = 0
        skipped_posts = 0
        for index, (post, cache_path, raw_payload) in enumerate(discovered, start=1):
            if not args.refresh and post_already_parsed(conn, str(post.source_post_id)):
                skipped_posts += 1
                print(f"Skipping parsed Substack post {index}/{len(discovered)} {post.title}", file=sys.stderr)
                continue
            print(f"Parsing Substack post {index}/{len(discovered)} {post.title}", file=sys.stderr)
            fetched_at = dt.datetime.now(dt.UTC).isoformat()
            try:
                predictions = parse_predictions(post)
                if args.ocr_image_charts:
                    image_predictions = parse_image_chart_predictions(
                        post,
                        image_fetcher=image_fetcher,
                        ocr_engine=ocr_engine,
                        first_row_ordinal=len(predictions) + 1,
                        image_limit=args.ocr_image_limit,
                        min_ocr_confidence=args.min_ocr_confidence,
                    )
                    predictions.extend(theory.dedupe_predictions(image_predictions, existing=predictions))
                result = theory.PageParseResult(post, predictions, fetched_at, cache_path, raw_payload)
            except Exception as exc:  # pragma: no cover - defensive per-post failure handling.
                result = theory.PageParseResult(post, [], fetched_at, cache_path, raw_payload, error=str(exc))
            post_count, prediction_count = import_parse_result(conn, result, issue_source=args.issue_source)
            imported_posts += post_count
            imported_predictions += prediction_count
        print(
            f"Imported {imported_posts} Box Office Theory Substack posts and {imported_predictions} predictions; "
            f"skipped {skipped_posts} parsed posts.",
            file=sys.stderr,
        )
    finally:
        fetcher.close()
        conn.close()
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import Box Office Theory Substack movie box office predictions.")
    parser.add_argument("--start-date", type=parse_date_arg, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=parse_date_arg, default=DEFAULT_END_DATE)
    parser.add_argument(
        "--database-url",
        default=database_url_from_env(),
        help="PostgreSQL connection URL. Defaults to DATABASE_URL or POSTGRES_DSN.",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Raw Substack response cache directory.")
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=MIN_DELAY_SECONDS,
        help="Delay between uncached Substack requests. Must be at least 5.",
    )
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="HTTP User-Agent. Must identify as a bot.")
    parser.add_argument("--feed-url", default=FEED_URL, help="Substack RSS feed URL.")
    parser.add_argument(
        "--discovery",
        default="archive",
        choices=("archive", "rss"),
        help="Use archive pagination plus post JSON, or only the current RSS feed.",
    )
    parser.add_argument("--refresh", action="store_true", help="Reparse even when post status is parsed.")
    parser.add_argument("--full-refresh", action="store_true", help="Reparse every accessible post in the Substack archive.")
    parser.add_argument("--offline", action="store_true", help="Require all Substack responses to exist in cache.")
    parser.add_argument("--dry-run", action="store_true", help="Print discovery URLs and exit.")
    parser.add_argument(
        "--print-cache-paths",
        action="store_true",
        help="Print expected first discovery cache path, then exit.",
    )
    parser.add_argument("--per-page", type=int, default=DEFAULT_PER_PAGE)
    parser.add_argument("--max-pages", type=int, help="Optional archive page cap for smoke tests.")
    parser.add_argument("--max-articles", type=int, help="Optional post cap after discovery.")
    parser.add_argument(
        "--ocr-image-charts",
        action="store_true",
        help="OCR likely Substack forecast chart images and import rows as predictions.",
    )
    parser.add_argument("--ocr-image-limit", type=int, help="Optional per-post chart image cap for OCR smoke tests.")
    parser.add_argument(
        "--min-ocr-confidence",
        type=float,
        default=DEFAULT_MIN_OCR_CONFIDENCE,
        help="Skip OCR chart images below this mean confidence. Use 0 to keep every OCR result.",
    )
    parser.add_argument("--tesseract-command", default="tesseract", help="Tesseract executable used for image OCR.")
    parser.add_argument("--tesseract-psm", type=int, default=6, help="Tesseract page segmentation mode for chart OCR.")
    parser.add_argument("--tesseract-language", default="eng", help="Tesseract language for chart OCR.")
    parser.add_argument("--issue-source", default="boxofficetheory_substack_prediction_import")
    return parser


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
