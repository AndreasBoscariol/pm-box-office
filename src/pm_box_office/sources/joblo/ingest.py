#!/usr/bin/env python3
"""Ingest JoBlo weekend box office estimates into PostgreSQL."""

from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import hashlib
from html.parser import HTMLParser
import re
import sys
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from pm_box_office.db.connection import connect_database, database_url_from_env
from pm_box_office.domain import movies as movie_identity
from pm_box_office.sources.common.cli import parse_date_arg
from pm_box_office.sources.common.fetch import CacheFirstFetcher
from pm_box_office.sources.common.parsing import clean_text
from pm_box_office.sources.common.schema import acquire_schema_init_lock


BASE_URL = "https://www.joblo.com"
ARCHIVE_URL = f"{BASE_URL}/weekend-box-office/"
RSS_URL = f"{ARCHIVE_URL}feed/"
DEFAULT_START_DATE = dt.date(2026, 6, 1)
DEFAULT_END_DATE = dt.date(2026, 6, 30)
FULL_REFRESH_START_DATE = dt.date(1900, 1, 1)
FULL_REFRESH_END_DATE = dt.date(9999, 12, 31)
DEFAULT_MAX_PAGES = 25
FULL_REFRESH_MAX_PAGES = 10_000
DEFAULT_CACHE_DIR = Path("data/raw/joblo")
DEFAULT_USER_AGENT = "pm-box-office-joblo-bot/1.0 (+personal research; set --user-agent contact)"
MIN_DELAY_SECONDS = 10.0
DOMESTIC_MARKET = "US_CA"
DOMESTIC_CURRENCY = "USD"
PARSER_VERSION = "joblo_predictions_list_v1"
MAX_MOVIE_TITLE_LENGTH = 100


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
    movie_id: int | None = None
    match_status: str = "unmatched"
    match_method: str | None = None
    match_score: float | None = None
    match_notes: str | None = None


@dataclass(frozen=True)
class ArticleParseResult:
    archive_article: ArchiveArticle
    article: ArchiveArticle | None
    predictions: list[WeekendPrediction]
    fetched_at: str
    raw_cache_path: Path
    html: str
    error: str | None = None


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
class TextBlock:
    tag: str
    text: str
    emphasized_titles: tuple[str, ...] = ()


class HtmlTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "a" and attrs_dict.get("href"):
            self._href = canonical_article_url(str(attrs_dict["href"]))
            self._parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href:
            self.links.append((self._href, clean_text(" ".join(self._parts))))
            self._href = None
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._href:
            self._parts.append(data)


class ArchiveParser(HTMLParser):
    def __init__(self, source_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.source_url = source_url
        self.articles: list[ArchiveArticle] = []
        self._current_href: str | None = None
        self._capture_heading = False
        self._heading_parts: list[str] = []
        self._category: str | None = None
        self._capture_category = False
        self._excerpt: str | None = None
        self._capture_excerpt = False
        self._excerpt_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        classes = set((attrs_dict.get("class") or "").split())
        if tag == "a" and attrs_dict.get("href"):
            self._current_href = canonical_article_url(str(attrs_dict["href"]))
        if tag in {"h1", "h2", "h3"}:
            self._capture_heading = True
            self._heading_parts = []
        if tag in {"h4", "span", "div"} and ("category" in classes or "post-category" in classes):
            self._capture_category = True
        if tag in {"p", "div"} and ("excerpt" in classes or "summary" in classes):
            self._capture_excerpt = True
            self._excerpt_parts = []

    def handle_endtag(self, tag: str) -> None:
        if self._capture_heading and tag in {"h1", "h2", "h3"}:
            title = clean_text(" ".join(self._heading_parts))
            if title and self._current_href and is_joblo_article_url(self._current_href):
                article_type = classify_article_type(title, self._category)
                if article_type != "other":
                    self.articles.append(
                        ArchiveArticle(
                            article_url=self._current_href,
                            title=title,
                            author=None,
                            published_date=None,
                            article_type=article_type,
                            source_url=self.source_url,
                            excerpt=self._excerpt,
                        )
                    )
            self._capture_heading = False
            self._heading_parts = []
        if self._capture_category and tag in {"h4", "span", "div"}:
            self._capture_category = False
        if self._capture_excerpt and tag in {"p", "div"}:
            self._excerpt = clean_text(" ".join(self._excerpt_parts)) or None
            self._capture_excerpt = False
            self._excerpt_parts = []

    def handle_data(self, data: str) -> None:
        if self._capture_heading:
            self._heading_parts.append(data)
        if self._capture_category:
            value = clean_text(data)
            if value:
                self._category = value
        if self._capture_excerpt:
            self._excerpt_parts.append(data)


class ArticleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: str | None = None
        self.author: str | None = None
        self.published_date: str | None = None
        self.blocks: list[TextBlock] = []
        self._capture_tag: str | None = None
        self._parts: list[str] = []
        self._emphasis_depth = 0
        self._emphasis_parts: list[str] = []
        self._emphasized_titles: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag in {"script", "style", "nav", "footer"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        name = attrs_dict.get("property") or attrs_dict.get("name")
        content = attrs_dict.get("content")
        if tag == "meta" and name and content:
            if name in {"og:title", "twitter:title"} and not self.title:
                self.title = clean_text(content)
            elif name == "article:published_time" and not self.published_date:
                self.published_date = parse_dateish(content)
            elif name in {"author", "article:author"} and not self.author:
                self.author = clean_text(content)
        elif tag == "time" and attrs_dict.get("datetime") and not self.published_date:
            self.published_date = parse_dateish(str(attrs_dict["datetime"]))
        elif tag in {"h1", "h2", "h3", "p", "li"}:
            self._capture_tag = tag
            self._parts = []
            self._emphasis_depth = 0
            self._emphasis_parts = []
            self._emphasized_titles = []
        elif tag in {"em", "i"} and self._capture_tag:
            self._emphasis_depth += 1
            if self._emphasis_depth == 1:
                self._emphasis_parts = []
        elif tag == "br" and self._capture_tag:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav", "footer"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if self._capture_tag == tag:
            text = clean_multiline_text("".join(self._parts))
            if text:
                if tag == "h1" and not self.title:
                    self.title = text
                self.blocks.append(TextBlock(tag=tag, text=text, emphasized_titles=tuple(self._emphasized_titles)))
            self._capture_tag = None
            self._parts = []
            self._emphasis_depth = 0
            self._emphasis_parts = []
            self._emphasized_titles = []
        elif tag in {"em", "i"} and self._capture_tag and self._emphasis_depth:
            title = clean_text(" ".join(self._emphasis_parts))
            if title:
                self._emphasized_titles.append(title)
            self._emphasis_depth -= 1
            if self._emphasis_depth == 0:
                self._emphasis_parts = []

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and self._capture_tag:
            self._parts.append(data)
            if self._emphasis_depth:
                self._emphasis_parts.append(data)


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
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(query), ""))


def archive_url(page: int) -> str:
    if page <= 1:
        return ARCHIVE_URL
    return f"{ARCHIVE_URL}page/{page}/"


def rss_url(page: int) -> str:
    if page <= 1:
        return RSS_URL
    return f"{RSS_URL}?paged={page}"


def is_joblo_article_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return parsed.netloc.endswith("joblo.com") and parsed.path.strip("/") not in {"", "weekend-box-office"}


def clean_multiline_text(value: str) -> str:
    text = value.replace("\xa0", " ").replace("\r", "\n")
    lines = [clean_text(line) for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def strip_html(value: str) -> str:
    parser = HtmlTextParser()
    parser.feed(value)
    return clean_text(" ".join(parser.parts))


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


def parse_rss_date(value: str) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        return email.utils.parsedate_to_datetime(text).date().isoformat()
    except (TypeError, ValueError):
        return parse_dateish(text)


def classify_article_type(title: str, category: str | None = None) -> str:
    normalized = f"{title} {category or ''}".lower()
    if "box office predictions" in normalized:
        return "box_office_predictions"
    return "other"


def classify_prediction_link(url: str, text: str) -> str:
    normalized_url = urllib.parse.urlparse(url).path.strip("/").lower()
    normalized_text = normalize_header(text)
    if "box-office-predictions" in normalized_url or "weekend-box-office-predictions" in normalized_url:
        return "box_office_predictions"
    if "predicted" in normalized_url or "prediction" in normalized_url:
        return "box_office_predictions"
    if "predicted" in normalized_text or "prediction" in normalized_text:
        return "box_office_predictions"
    return "other"


def is_supported_article(article: ArchiveArticle) -> bool:
    return article.article_type == "box_office_predictions"


RSS_NAMESPACES = {
    "dc": "http://purl.org/dc/elements/1.1/",
    "content": "http://purl.org/rss/1.0/modules/content/",
}


def parse_rss(xml_text: str, *, source_url: str) -> list[ArchiveArticle]:
    try:
        root = ET.fromstring(xml_text.strip())
    except ET.ParseError as exc:
        raise ValueError(f"invalid JoBlo RSS XML from {source_url}: {exc}") from exc
    articles: list[ArchiveArticle] = []
    for item in root.findall("./channel/item"):
        title = clean_text(item.findtext("title") or "")
        link = clean_text(item.findtext("link") or "")
        if not title or not link:
            continue
        article_type = classify_article_type(title)
        description = item.findtext("description") or ""
        author = item.findtext("dc:creator", namespaces=RSS_NAMESPACES)
        published_date = parse_rss_date(item.findtext("pubDate") or "")
        if article_type != "other":
            articles.append(
                ArchiveArticle(
                    article_url=canonical_article_url(link),
                    title=title,
                    author=clean_text(author) if author else None,
                    published_date=published_date,
                    article_type=article_type,
                    source_url=source_url,
                    excerpt=strip_html(description) if description else None,
                )
            )
        content = item.findtext("content:encoded", namespaces=RSS_NAMESPACES) or ""
        for href, text in extract_links(content):
            linked_type = classify_prediction_link(href, text)
            if linked_type == "other":
                continue
            articles.append(
                ArchiveArticle(
                    article_url=href,
                    title=text if text else urllib.parse.urlparse(href).path.strip("/").replace("-", " ").title(),
                    author=clean_text(author) if author else None,
                    published_date=published_date,
                    article_type=linked_type,
                    source_url=source_url,
                    excerpt=f"Discovered from RSS backlink in {title}",
                )
            )
    return dedupe_articles(articles)


def extract_links(html: str) -> list[tuple[str, str]]:
    parser = LinkParser()
    parser.feed(html)
    return [
        (href, text)
        for href, text in parser.links
        if is_joblo_article_url(href)
    ]


def parse_archive(html: str, *, source_url: str) -> list[ArchiveArticle]:
    parser = ArchiveParser(source_url)
    parser.feed(html)
    return dedupe_articles(parser.articles)


def dedupe_articles(articles: list[ArchiveArticle]) -> list[ArchiveArticle]:
    deduped: dict[str, ArchiveArticle] = {}
    for article in articles:
        deduped.setdefault(article.article_url, article)
    return list(deduped.values())


def parse_article(
    html: str,
    *,
    article_url: str,
    fallback: ArchiveArticle | None = None,
) -> tuple[ArchiveArticle, list[WeekendPrediction]]:
    parser = ArticleParser()
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
    return article, parse_prediction_blocks(
        parser.blocks,
        article_url=article_url,
        article_title=title,
        published_date=published_date,
    )


def parse_prediction_blocks(
    blocks: list[TextBlock],
    *,
    article_url: str,
    article_title: str,
    published_date: str | None,
) -> list[WeekendPrediction]:
    in_predictions = False
    candidates: list[str] = []
    for block in blocks:
        normalized = normalize_header(block.text)
        if "prediction" in normalized and block.tag in {"h2", "h3", "p"}:
            in_predictions = True
            continue
        if in_predictions:
            if block.tag in {"h2", "h3"} and "prediction" not in normalized:
                break
            if block.tag == "li" or parse_ranked_prediction_line(block.text) is not None:
                candidates.append(block.text)
    if not candidates:
        candidates = [block.text for block in blocks if block.tag == "li" and parse_ranked_prediction_line(block.text)]
    target_start, target_end = infer_target_dates(article_title=article_title, published_date=published_date)
    predictions: list[WeekendPrediction] = []
    for line in candidates:
        parsed = parse_ranked_prediction_line(line)
        if parsed is None:
            continue
        rank, title, money_range = parsed
        row_ordinal = len(predictions) + 1
        normalized_title = normalize_movie_title(title)
        key_material = "|".join(
            [
                article_url,
                str(row_ordinal),
                normalized_title,
                "domestic_weekend",
                str(money_range[0]),
                str(money_range[1]),
                str(target_start),
                str(target_end),
                PARSER_VERSION,
            ]
        )
        predictions.append(
            WeekendPrediction(
                article_url=article_url,
                source_movie_title=title,
                normalized_movie_title=normalized_title,
                source_movie_id=joblo_source_movie_id(
                    market=DOMESTIC_MARKET,
                    normalized_movie_title=normalized_title,
                    target_start_date=target_start,
                ),
                distributor="Unknown",
                release_status="UNKNOWN",
                source_rank=rank,
                market=DOMESTIC_MARKET,
                currency=DOMESTIC_CURRENCY,
                forecast_metric="domestic_weekend",
                range_low_usd=money_range[0],
                range_high_usd=money_range[1],
                showtime_market_share_pct=None,
                target_start_date=target_start,
                target_end_date=target_end,
                raw_forecast_text=line,
                source_context="predictions_list",
                parser_version=PARSER_VERSION,
                row_ordinal=row_ordinal,
                source_row_key=hashlib.sha256(key_material.encode("utf-8")).hexdigest(),
            )
        )
    if predictions:
        return predictions
    return parse_prose_prediction_blocks(
        blocks,
        article_url=article_url,
        article_title=article_title,
        target_start=target_start,
        target_end=target_end,
    )


def parse_prose_prediction_blocks(
    blocks: list[TextBlock],
    *,
    article_url: str,
    article_title: str,
    target_start: str | None,
    target_end: str | None,
) -> list[WeekendPrediction]:
    predictions: list[WeekendPrediction] = []
    main_title = main_movie_title_from_article_title(article_title)
    seen: set[tuple[str, int, int]] = set()
    article_started = False
    for block in blocks:
        if block.tag == "h1" and normalize_header(block.text) == normalize_header(article_title):
            article_started = True
            continue
        if not article_started:
            continue
        if block.tag != "p" or "$" not in block.text:
            continue
        candidates = list(block.emphasized_titles)
        if main_title:
            candidates.insert(0, main_title)
        for title, segment in prediction_segments(block.text, candidates):
            if not prose_segment_has_prediction_language(segment):
                continue
            money_range = parse_money_range(segment)
            if money_range is None:
                continue
            clean_title = clean_movie_title(title)
            if not clean_title or not is_prose_movie_title(clean_title):
                continue
            key = (normalize_movie_title(clean_title), money_range[0], money_range[1])
            if key in seen:
                continue
            seen.add(key)
            row_ordinal = len(predictions) + 1
            normalized_title = normalize_movie_title(clean_title)
            key_material = "|".join(
                [
                    article_url,
                    str(row_ordinal),
                    normalized_title,
                    "domestic_weekend",
                    str(money_range[0]),
                    str(money_range[1]),
                    str(target_start),
                    str(target_end),
                    PARSER_VERSION,
                ]
            )
            predictions.append(
                WeekendPrediction(
                    article_url=article_url,
                    source_movie_title=clean_title,
                    normalized_movie_title=normalized_title,
                    source_movie_id=joblo_source_movie_id(
                        market=DOMESTIC_MARKET,
                        normalized_movie_title=normalized_title,
                        target_start_date=target_start,
                    ),
                    distributor="Unknown",
                    release_status="UNKNOWN",
                    source_rank=None,
                    market=DOMESTIC_MARKET,
                    currency=DOMESTIC_CURRENCY,
                    forecast_metric="domestic_weekend",
                    range_low_usd=money_range[0],
                    range_high_usd=money_range[1],
                    showtime_market_share_pct=None,
                    target_start_date=target_start,
                    target_end_date=target_end,
                    raw_forecast_text=segment,
                    source_context="prediction_prose",
                    parser_version=PARSER_VERSION,
                    row_ordinal=row_ordinal,
                    source_row_key=hashlib.sha256(key_material.encode("utf-8")).hexdigest(),
                )
            )
    return predictions


def prediction_segments(text: str, candidate_titles: list[str]) -> list[tuple[str, str]]:
    normalized_candidates = []
    lowered = text.lower()
    for title in candidate_titles:
        clean_title = clean_movie_title(title)
        if not clean_title:
            continue
        index = lowered.find(clean_title.lower())
        if index >= 0 and not text_index_inside_parentheses(text, index):
            normalized_candidates.append((index, clean_title))
    if not normalized_candidates:
        return []
    normalized_candidates = sorted(set(normalized_candidates))
    filtered_candidates: list[tuple[int, str]] = []
    for start, title in normalized_candidates:
        title_end = start + len(title)
        if any(
            other_start <= start
            and title_end <= other_start + len(other_title)
            and len(other_title) > len(title)
            for other_start, other_title in normalized_candidates
        ):
            continue
        filtered_candidates.append((start, title))
    segments: list[tuple[str, str]] = []
    for index, (start, title) in enumerate(filtered_candidates):
        end = filtered_candidates[index + 1][0] if index + 1 < len(filtered_candidates) else len(text)
        segments.append((title, clean_text(text[start:end])))
    return segments


def text_index_inside_parentheses(text: str, index: int) -> bool:
    last_open = text.rfind("(", 0, index)
    last_close = text.rfind(")", 0, index)
    return last_open > last_close


def prose_segment_has_prediction_language(segment: str) -> bool:
    normalized = normalize_header(segment)
    if re.search(r"^[^$]{0,120}\b(?:rotten tomatoes|rt score|fresh score|audience score|review|reviews)\b", segment, re.I):
        return False
    if re.match(r"^.{1,80}(?:'s|’s)\b", segment):
        return False
    return bool(
        re.search(
            r"\b(expected|expect|estimate|predicted|prediction|should|could|might|will|likely|poised|projected|tracking)\b",
            normalized,
        )
        and re.search(r"\b(open|opening|make|gross|earn|take|bring|haul|range)\b", normalized)
    )


def main_movie_title_from_article_title(title: str) -> str | None:
    match = re.search(r"predictions?\s*:\s*(.+)", title, flags=re.IGNORECASE)
    if not match:
        return None
    value = clean_text(match.group(1))
    split = re.split(
        r"\s+(?:will|should|could|might|is|are|to|set|has|have|takes?|wins?|opens?|shoots?)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    return clean_movie_title(split[0]) if split else None


def is_prose_movie_title(title: str) -> bool:
    normalized = normalize_header(title)
    if not normalized or len(title) > MAX_MOVIE_TITLE_LENGTH:
        return False
    if re.search(r"(?:'s|’s)$", title):
        return False
    blocked = {
        "movies",
        "box office",
        "weekend box office",
        "fourth of july weekend",
        "weekend",
        "kids",
        "families",
        "audiences",
        "moviegoers",
    }
    if normalized in blocked:
        return False
    return not re.search(r"\b(?:weekend|audiences|moviegoers|kids|families|sequel|opening|gross|box office|competition)\b", normalized)


def parse_ranked_prediction_line(value: str) -> tuple[int | None, str, tuple[int, int]] | None:
    text = clean_text(value)
    match = re.match(r"^(?:(\d+)[.)]\s*)?(.+?)\s*[:\-]\s*(\$.*)$", text)
    if not match:
        return None
    rank = int(match.group(1)) if match.group(1) else None
    title = clean_movie_title(match.group(2))
    money_range = parse_money_range(match.group(3))
    if not title or len(title) > MAX_MOVIE_TITLE_LENGTH or money_range is None:
        return None
    return rank, title, money_range


def parse_money_range(value: str) -> tuple[int, int] | None:
    text = clean_text(value).replace("\u2013", "-").replace("\u2014", "-")
    between_match = re.search(
        r"(?:between|anywhere between)\s+\$?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:and|to)\s*\$?\s*([0-9]+(?:\.[0-9]+)?)\s*(billion|million|m|b)?",
        text,
        flags=re.IGNORECASE,
    )
    if between_match:
        unit = between_match.group(3) or infer_money_unit(text)
        low = money_number_to_usd(float(between_match.group(1)), unit)
        high = money_number_to_usd(float(between_match.group(2)), unit)
        return (min(low, high), max(low, high))
    range_match = re.search(
        r"\$?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:-|to)\s*\$?\s*([0-9]+(?:\.[0-9]+)?)\s*(billion|million|m|b)?",
        text,
        flags=re.IGNORECASE,
    )
    if range_match:
        unit = range_match.group(3) or infer_money_unit(text)
        low = money_number_to_usd(float(range_match.group(1)), unit)
        high = money_number_to_usd(float(range_match.group(2)), unit)
        return (min(low, high), max(low, high))
    exact_match = re.search(r"\$\s*([0-9]+(?:\.[0-9]+)?)\s*(billion|million|m|b)?", text, flags=re.IGNORECASE)
    if not exact_match:
        return None
    value_usd = money_number_to_usd(float(exact_match.group(1)), exact_match.group(2) or infer_money_unit(text))
    return value_usd, value_usd


def infer_money_unit(text: str) -> str:
    if re.search(r"\bbillion\b|\bb\b", text, flags=re.IGNORECASE):
        return "billion"
    return "million"


def money_number_to_usd(value: float, unit: str | None) -> int:
    normalized = (unit or "million").lower()
    multiplier = 1_000_000_000 if normalized in {"billion", "b"} else 1_000_000
    return int(round(value * multiplier))


def infer_target_dates(*, article_title: str, published_date: str | None) -> tuple[str | None, str | None]:
    explicit = extract_date_range(article_title)
    if explicit != (None, None):
        return explicit
    if published_date is None:
        return None, None
    published = dt.date.fromisoformat(published_date)
    days_until_friday = (4 - published.weekday()) % 7
    start = published + dt.timedelta(days=days_until_friday)
    end = start + dt.timedelta(days=2)
    return start.isoformat(), end.isoformat()


MONTH_PATTERN = (
    r"(January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)


def extract_date_range(text: str) -> tuple[str | None, str | None]:
    year_match = re.search(r"\b(20\d{2})\b", text)
    if not year_match:
        return None, None
    year = int(year_match.group(1))
    range_match = re.search(
        rf"{MONTH_PATTERN}\s+(\d{{1,2}})\s*(?:-|to)\s*(?:(?:{MONTH_PATTERN})\s+)?(\d{{1,2}})",
        text.replace("\u2013", "-").replace("\u2014", "-"),
        flags=re.IGNORECASE,
    )
    if not range_match:
        return None, None
    month1 = range_match.group(1)
    day1 = int(range_match.group(2))
    month2 = range_match.group(3) or month1
    day2 = int(range_match.group(4))
    start = parse_month_day(month1, day1, year)
    end = parse_month_day(month2, day2, year)
    if not start or not end:
        return None, None
    if end < start:
        end = dt.date(year + 1, end.month, end.day)
    return start.isoformat(), end.isoformat()


def parse_month_day(month: str, day: int, year: int) -> dt.date | None:
    for fmt in ("%B", "%b"):
        try:
            month_number = dt.datetime.strptime(month[:3] if fmt == "%b" else month, fmt).month
            return dt.date(year, month_number, day)
        except ValueError:
            continue
    return None


def clean_movie_title(value: str) -> str:
    title = clean_text(value)
    title = re.sub(r"^\s*#?\d+[.)]\s+", "", title)
    return title.strip(" :.-")


def normalize_header(value: str) -> str:
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def normalize_movie_title(value: str) -> str:
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
    text = text.replace("&", " and ")
    text = re.sub(r"\s*\(\d{4}\)\s*$", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def joblo_source_movie_id(*, market: str, normalized_movie_title: str, target_start_date: str | None) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", normalized_movie_title).strip("-") or "unknown-title"
    release_key = target_start_date or "unknown-date"
    return f"joblo:{market}:{slug}:{release_key}"


def initialize_database(conn: Any) -> None:
    acquire_schema_init_lock(conn)
    movie_identity.ensure_movie_identity_schema(conn)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS joblo_articles (
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

        CREATE TABLE IF NOT EXISTS joblo_weekend_predictions (
            prediction_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            article_id BIGINT NOT NULL REFERENCES joblo_articles(article_id),
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
            movie_id BIGINT REFERENCES movies(movie_id),
            match_status TEXT NOT NULL,
            match_method TEXT,
            match_score DOUBLE PRECISION,
            match_notes TEXT,
            fetched_at TEXT NOT NULL,
            raw_cache_path TEXT NOT NULL,
            UNIQUE(article_id, source_row_key)
        );

        CREATE TABLE IF NOT EXISTS joblo_ingest_issues (
            issue_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            issue_source TEXT NOT NULL,
            issue_type TEXT NOT NULL,
            article_url TEXT,
            source_movie_title TEXT,
            details TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP::text),
            UNIQUE(issue_source, issue_type, article_url, source_movie_title, details)
        );

        CREATE INDEX IF NOT EXISTS idx_joblo_articles_status
            ON joblo_articles(status);
        CREATE INDEX IF NOT EXISTS idx_joblo_articles_discovered_date
            ON joblo_articles(discovered_date);
        CREATE INDEX IF NOT EXISTS idx_joblo_weekend_predictions_movie_id
            ON joblo_weekend_predictions(movie_id);
        CREATE INDEX IF NOT EXISTS idx_joblo_weekend_predictions_title
            ON joblo_weekend_predictions(normalized_movie_title);
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
        INSERT INTO joblo_articles (
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
            fetched_at = COALESCE(excluded.fetched_at, joblo_articles.fetched_at),
            raw_cache_path = COALESCE(excluded.raw_cache_path, joblo_articles.raw_cache_path),
            sha256 = COALESCE(excluded.sha256, joblo_articles.sha256),
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
    return int(conn.execute("SELECT article_id FROM joblo_articles WHERE article_url = %s", (article.article_url,)).fetchone()[0])


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
        INSERT INTO joblo_weekend_predictions (
            article_id, source_row_key, source_movie_id, source_movie_title, normalized_movie_title,
            distributor, release_status, source_rank, market, currency, forecast_metric,
            range_low_usd, range_high_usd, showtime_market_share_pct,
            target_start_date, target_end_date, raw_forecast_text, source_context,
            parser_version, row_ordinal, movie_id, match_status, match_method,
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
        INSERT INTO joblo_ingest_issues (
            issue_source, issue_type, article_url, source_movie_title, details
        ) VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """,
        (issue_source, issue_type, article_url, source_movie_title, details),
    )


def clear_article_issues(conn: Any, *, issue_source: str, article_url: str) -> None:
    conn.execute("DELETE FROM joblo_ingest_issues WHERE issue_source = %s AND article_url = %s", (issue_source, article_url))


def article_already_parsed(conn: Any, article_url: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM joblo_articles WHERE article_url = %s AND status = 'parsed' LIMIT 1",
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
            upsert_joblo_movie_source_id(
                conn,
                movie_id=match.movie_id,
                prediction=prediction,
                match_status=match.status,
                match_method=match.method,
                match_score=match.score,
            )
            repoint_joblo_predictions(
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
    source_id_match = find_joblo_source_id_match(conn, prediction)
    if source_id_match is not None:
        return source_id_match
    matches = [candidate for candidate in candidates if candidate.normalized_title == prediction.normalized_movie_title]
    existing = choose_existing_movie_match(prediction, matches)
    if existing is not None:
        return existing
    if not matches:
        return MovieMatch(None, "unmatched", "normalized_exact", 0.0, "No movie title matched")
    return MovieMatch(None, "ambiguous", "normalized_exact", 0.5, "Multiple movies share the title")


def choose_existing_movie_match(
    prediction: WeekendPrediction,
    matches: list[MovieCandidate],
) -> MovieMatch | None:
    if not matches:
        return None
    if len(matches) == 1:
        candidate = matches[0]
        status = "matched" if candidate.movie_url is not None else "provisional"
        return MovieMatch(candidate.movie_id, status, "normalized_exact", 1.0, None)
    target_year = infer_prediction_year(prediction)
    if target_year is None:
        return None
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


def find_joblo_source_id_match(conn: Any, prediction: WeekendPrediction) -> MovieMatch | None:
    if not relation_exists(conn, "movie_source_ids"):
        return None
    row = conn.execute(
        """
        SELECT m.movie_id, m.movie_url, src.match_status, src.match_score
        FROM movie_source_ids src
        JOIN movies m ON m.movie_id = src.movie_id
        WHERE src.source = 'joblo'
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
        "joblo_source_id",
        float(row[3]) if row[3] is not None else 1.0,
        f"Matched existing JoBlo source id {prediction.source_movie_id}",
    )


def upsert_joblo_movie_source_id(
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
        source=movie_identity.SOURCE_JOBLO,
        source_movie_id=prediction.source_movie_id,
        source_title=prediction.source_movie_title,
        match_status=match_status,
        match_method=match_method,
        match_score=match_score,
    )


def repoint_joblo_predictions(
    conn: Any,
    *,
    prediction: WeekendPrediction,
    movie_id: int,
    match_status: str,
    match_method: str | None,
    match_score: float | None,
    match_notes: str | None,
) -> None:
    if not relation_exists(conn, "joblo_weekend_predictions"):
        return
    conn.execute(
        """
        UPDATE joblo_weekend_predictions
        SET movie_id = %s,
            match_status = %s,
            match_method = %s,
            match_score = %s,
            match_notes = %s
        WHERE source_movie_id = %s
        """,
        (movie_id, match_status, match_method, match_score, match_notes, prediction.source_movie_id),
    )


def infer_prediction_year(prediction: WeekendPrediction) -> int | None:
    for value in (prediction.target_start_date, prediction.target_end_date):
        if value and re.match(r"20\d{2}", value):
            return int(value[:4])
    return None


def import_article_parse_result(conn: Any, result: ArticleParseResult, *, issue_source: str) -> tuple[int, int]:
    if result.error is not None:
        upsert_article(
            conn,
            result.archive_article,
            status="article_page_unavailable",
            fetched_at=result.fetched_at,
            raw_cache_path=result.raw_cache_path,
            html="",
        )
        insert_issue(
            conn,
            issue_source=issue_source,
            issue_type="article_page_unavailable",
            article_url=result.archive_article.article_url,
            source_movie_title=None,
            details=result.error,
        )
        conn.commit()
        return 1, 0
    if result.article is None:
        raise RuntimeError("successful article parse result is missing article")
    article_id = upsert_article(
        conn,
        result.article,
        status="parsed",
        fetched_at=result.fetched_at,
        raw_cache_path=result.raw_cache_path,
        html=result.html,
    )
    predictions = match_predictions(conn, result.predictions)
    conn.execute("DELETE FROM joblo_weekend_predictions WHERE article_id = %s", (article_id,))
    clear_article_issues(conn, issue_source=issue_source, article_url=result.article.article_url)
    if not predictions:
        insert_issue(
            conn,
            issue_source=issue_source,
            issue_type="no_weekend_predictions_parsed",
            article_url=result.article.article_url,
            source_movie_title=None,
            details=f"No JoBlo prediction list parsed from {result.article.title}",
        )
    insert_predictions(conn, article_id, predictions, fetched_at=result.fetched_at, raw_cache_path=result.raw_cache_path)
    conn.commit()
    return 1, len(predictions)


def configure_full_refresh_args(args: argparse.Namespace) -> None:
    if not getattr(args, "full_refresh", False):
        return
    args.refresh = True
    args.discovery = "rss"
    args.start_date = FULL_REFRESH_START_DATE
    args.end_date = FULL_REFRESH_END_DATE
    if getattr(args, "max_pages", None) == DEFAULT_MAX_PAGES:
        args.max_pages = FULL_REFRESH_MAX_PAGES


def discover_articles(fetcher: CacheFirstFetcher, args: argparse.Namespace) -> list[ArchiveArticle]:
    if args.discovery == "rss":
        return discover_rss_articles(fetcher, args)
    if args.discovery == "archive":
        return discover_archive_articles(fetcher, args)
    return discover_rss_articles(fetcher, args)


def fetch_rss_articles(fetcher: CacheFirstFetcher) -> list[ArchiveArticle]:
    return fetch_rss_page_articles(fetcher, page=1)


def fetch_rss_page_articles(fetcher: CacheFirstFetcher, *, page: int) -> list[ArchiveArticle]:
    url = rss_url(page)
    print(f"Reading RSS feed page {page} {url}", file=sys.stderr)
    xml_text, _cache_path, _fetched = fetcher.get_text(url, suffix=".xml", accept="application/rss+xml,application/xml,text/xml")
    return parse_rss(xml_text, source_url=url)


def discover_rss_articles(fetcher: CacheFirstFetcher, args: argparse.Namespace) -> list[ArchiveArticle]:
    articles_by_url: dict[str, ArchiveArticle] = {}
    for page in range(1, args.max_pages + 1):
        page_articles = fetch_rss_page_articles(fetcher, page=page)
        supported = [article for article in page_articles if is_supported_article(article)]
        if not supported:
            break
        page_all_older = True
        page_added = False
        for article in supported:
            if article.published_date is None:
                page_all_older = False
                articles_by_url.setdefault(article.article_url, article)
                page_added = True
                continue
            published = dt.date.fromisoformat(article.published_date)
            if published >= args.start_date:
                page_all_older = False
            if args.start_date <= published <= args.end_date:
                if article.article_url not in articles_by_url:
                    page_added = True
                articles_by_url.setdefault(article.article_url, article)
        if not page_added:
            break
        if page_all_older and not page_added:
            break
    articles = sorted(articles_by_url.values(), key=lambda article: (article.published_date or "", article.article_url))
    return limit_articles(articles, args)


def discover_archive_articles(fetcher: CacheFirstFetcher, args: argparse.Namespace) -> list[ArchiveArticle]:
    articles_by_url: dict[str, ArchiveArticle] = {}
    for page in range(1, args.max_pages + 1):
        url = archive_url(page)
        print(f"Reading archive page {page} {url}", file=sys.stderr)
        html, _cache_path, _fetched = fetcher.get_text(url)
        page_articles = parse_archive(html, source_url=url)
        if not page_articles:
            break
        for article in page_articles:
            if not is_supported_article(article):
                continue
            if article.published_date is None:
                articles_by_url.setdefault(article.article_url, article)
                continue
            published = dt.date.fromisoformat(article.published_date)
            if args.start_date <= published <= args.end_date:
                articles_by_url.setdefault(article.article_url, article)
    return limit_articles(list(articles_by_url.values()), args)


def filter_discovered_articles(articles: list[ArchiveArticle], args: argparse.Namespace) -> list[ArchiveArticle]:
    filtered = []
    for article in articles:
        if not is_supported_article(article):
            continue
        if article.published_date is None:
            filtered.append(article)
            continue
        published = dt.date.fromisoformat(article.published_date)
        if args.start_date <= published <= args.end_date:
            filtered.append(article)
    return sorted(filtered, key=lambda article: (article.published_date or "", article.article_url))


def oldest_published_date(articles: list[ArchiveArticle]) -> dt.date | None:
    dates = [dt.date.fromisoformat(article.published_date) for article in articles if article.published_date]
    return min(dates) if dates else None


def limit_articles(articles: list[ArchiveArticle], args: argparse.Namespace) -> list[ArchiveArticle]:
    if args.max_articles is not None:
        return articles[: args.max_articles]
    return articles


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
    if args.dry_run:
        urls = [rss_url(page) for page in range(1, args.max_pages + 1)] if args.discovery == "rss" else [archive_url(page) for page in range(1, args.max_pages + 1)]
        if args.discovery == "auto":
            urls = [rss_url(page) for page in range(1, args.max_pages + 1)]
        for url in urls:
            print(url)
        return 0
    fetcher = CacheFirstFetcher(
        args.cache_dir,
        refresh=args.refresh,
        offline=args.offline,
        delay_seconds=args.delay_seconds,
        user_agent=args.user_agent,
        default_accept="text/html,application/xhtml+xml,application/xml",
        offline_error_prefix="JoBlo cache miss in offline mode",
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
            if not args.refresh and article_already_parsed(conn, archive_article.article_url):
                skipped_articles += 1
                print(f"Skipping parsed article {index}/{len(articles)} {archive_article.title}", file=sys.stderr)
                continue
            upsert_article(conn, archive_article, status="discovered")
            conn.commit()
            print(f"Reading article {index}/{len(articles)} {archive_article.title}", file=sys.stderr)
            try:
                html, cache_path, _fetched = fetcher.get_text(archive_article.article_url)
                article, predictions = parse_article(html, article_url=archive_article.article_url, fallback=archive_article)
                result = ArticleParseResult(
                    archive_article=archive_article,
                    article=article,
                    predictions=predictions,
                    fetched_at=dt.datetime.now(dt.UTC).isoformat(),
                    raw_cache_path=cache_path,
                    html=html,
                )
            except Exception as exc:
                result = ArticleParseResult(
                    archive_article=archive_article,
                    article=None,
                    predictions=[],
                    fetched_at=dt.datetime.now(dt.UTC).isoformat(),
                    raw_cache_path=fetcher.cache_path(archive_article.article_url),
                    html="",
                    error=str(exc),
                )
            article_count, prediction_count = import_article_parse_result(conn, result, issue_source=args.issue_source)
            imported_articles += article_count
            imported_predictions += prediction_count
        print(
            f"Imported {imported_articles} JoBlo articles and {imported_predictions} weekend predictions; "
            f"skipped {skipped_articles} parsed articles.",
            file=sys.stderr,
        )
    finally:
        conn.close()
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import JoBlo weekend box office predictions.")
    parser.add_argument("--start-date", type=parse_date_arg, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=parse_date_arg, default=DEFAULT_END_DATE)
    parser.add_argument("--database-url", default=database_url_from_env())
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--delay-seconds", type=float, default=MIN_DELAY_SECONDS)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--discovery", choices=("auto", "rss", "archive"), default="auto")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--full-refresh", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    parser.add_argument("--max-articles", type=int)
    parser.add_argument("--issue-source", default="joblo_weekend_import")
    return parser


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
