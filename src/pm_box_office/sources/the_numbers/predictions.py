#!/usr/bin/env python3
"""Collect The Numbers prediction table images and parse OCR text.

The Numbers sometimes publishes forecast/projection tables as PNGs in news
articles. This module keeps the pipeline explicit: discover candidate images,
cache the source image, OCR with an external adapter, then parse the OCR text
into auditable rows.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from dataclasses import dataclass, replace
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any
import urllib.parse

from pm_box_office.db.connection import connect_database
from pm_box_office.domain import movies as movie_identity
from pm_box_office.sources.common.cli import add_database_arg
from pm_box_office.sources.common.fetch import CacheFirstFetcher
from pm_box_office.sources.common.parsing import clean_text, parse_money
from pm_box_office.sources.common.schema import acquire_schema_init_lock


BASE_URL = "https://www.the-numbers.com"
DEFAULT_NEWS_URL = f"{BASE_URL}/"
DEFAULT_CACHE_DIR = Path("data/raw/the_numbers_predictions")
DEFAULT_USER_AGENT = "pm-box-office-the-numbers-prediction-bot/1.0 (+personal research; set --user-agent contact)"
TRANSIENT_STATUSES = {429, 500, 502, 503, 504}
MIN_DELAY_SECONDS = 1.0
DEFAULT_MIN_OCR_CONFIDENCE = 70.0
IMAGE_EXTENSIONS = {".gif", ".jpg", ".jpeg", ".png", ".webp"}
PREDICTION_TERMS = ("prediction", "predictions", "projection", "projections", "forecast", "forecasts")
COMMON_DISTRIBUTORS = (
    "20th Century Studios",
    "Amazon MGM Studios",
    "Angel Studios",
    "Bleecker Street",
    "Crunchyroll",
    "DreamWorks Animation",
    "Focus Features",
    "IFC Films",
    "Lionsgate",
    "Metro-Goldwyn-Mayer",
    "Neon",
    "Paramount Pictures",
    "Roadside Attractions",
    "Searchlight Pictures",
    "Sony Pictures",
    "Sony Pictures Classics",
    "Trafalgar Releasing",
    "United Artists",
    "Universal",
    "Walt Disney",
    "Warner Bros.",
    "Warner Bros",
    "A24",
)
DOMESTIC_MARKET = "US_CA"
DOMESTIC_CURRENCY = "USD"
PARSER_VERSION = "the_numbers_prediction_image_ocr_v1"
SOURCE_THE_NUMBERS_PREDICTIONS = "the_numbers_predictions"


@dataclass(frozen=True)
class PredictionImage:
    article_id: str | None
    article_title: str
    article_date: str | None
    page_url: str
    image_url: str
    linked_url: str | None
    alt_text: str
    width: int | None
    height: int | None
    context: str


@dataclass(frozen=True)
class OcrToken:
    text: str
    confidence: float | None
    left: int | None
    top: int | None
    width: int | None
    height: int | None


@dataclass(frozen=True)
class OcrResult:
    text: str
    mean_confidence: float | None
    tokens: list[OcrToken]


@dataclass(frozen=True)
class PredictionRow:
    article_title: str
    article_date: str | None
    image_url: str
    linked_url: str | None
    table_kind: str
    row_ordinal: int
    row_label: str
    source_movie_title: str | None
    distributor: str | None
    metric: str
    value_usd: int | None
    actual_usd: int | None
    predicted_usd: int | None
    weekend_usd: int | None
    release_date: str | None
    multiplier: float | None
    pct_change: float | None
    pct_vs_prediction: float | None
    confidence: float | None
    raw_text: str
    raw_ocr_text: str
    source_row_key: str = ""
    source_movie_id: str | None = None
    normalized_movie_title: str | None = None
    movie_id: int | None = None
    match_status: str = "unmatched"
    match_method: str | None = None
    match_score: float | None = None
    match_notes: str | None = None


@dataclass(frozen=True)
class PredictionImageResult:
    image: PredictionImage
    image_cache_path: Path | None
    ocr: OcrResult | None
    rows: list[PredictionRow]
    fetched_at: str


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


class NewsPredictionImageParser(HTMLParser):
    """Find likely prediction/projection table images in The Numbers news HTML."""

    def __init__(self, page_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.page_url = page_url
        self.images: list[PredictionImage] = []
        self._in_article = False
        self._article_id: str | None = None
        self._article_title = ""
        self._article_date: str | None = None
        self._current_anchor: str | None = None
        self._capture: str | None = None
        self._capture_parts: list[str] = []
        self._context_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "article":
            self._in_article = True
            self._article_id = attrs_dict.get("id")
            self._article_title = ""
            self._article_date = None
            self._context_parts = []
        elif tag == "a":
            self._current_anchor = attrs_dict.get("href")
        elif self._in_article and tag == "h1":
            self._capture = "title"
            self._capture_parts = []
        elif self._in_article and tag == "p":
            class_name = attrs_dict.get("class") or ""
            self._capture = "date" if class_name == "news-date" else "context"
            self._capture_parts = []
        elif self._in_article and tag == "img":
            image = self._image_from_attrs(attrs_dict)
            if image and is_prediction_image_candidate(image):
                self.images.append(image)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._current_anchor = None
        elif self._in_article and tag == "article":
            self._in_article = False
            self._article_id = None
            self._article_title = ""
            self._article_date = None
            self._context_parts = []
        elif self._capture and ((self._capture == "title" and tag == "h1") or tag == "p"):
            text = clean_text(" ".join(self._capture_parts))
            if self._capture == "title":
                self._article_title = text
            elif self._capture == "date":
                self._article_date = parse_article_date(text)
            elif text:
                self._context_parts.append(text)
                self._context_parts = self._context_parts[-3:]
            self._capture = None
            self._capture_parts = []

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._capture_parts.append(data)

    def _image_from_attrs(self, attrs: dict[str, str | None]) -> PredictionImage | None:
        src = attrs.get("src")
        if not src:
            return None
        image_url = urllib.parse.urljoin(BASE_URL, src)
        linked_url = urllib.parse.urljoin(BASE_URL, self._current_anchor) if self._current_anchor else None
        return PredictionImage(
            article_id=self._article_id,
            article_title=self._article_title,
            article_date=self._article_date,
            page_url=self.page_url,
            image_url=image_url,
            linked_url=linked_url,
            alt_text=attrs.get("alt") or "",
            width=parse_optional_int(attrs.get("width")),
            height=parse_optional_int(attrs.get("height")),
            context=" ".join(self._context_parts[-2:]),
        )


class HtmlFetcher(CacheFirstFetcher):
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
            transient_statuses=TRANSIENT_STATUSES,
            default_accept="text/html,application/xhtml+xml",
        )

    def get(self, url: str) -> tuple[str, Path, bool]:
        return self.get_text(url, suffix=".html")


class ImageFetcher(CacheFirstFetcher):
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
            transient_statuses=TRANSIENT_STATUSES,
            default_accept="image/avif,image/webp,image/png,image/jpeg,image/*,*/*",
        )

    def get(self, url: str) -> tuple[bytes, Path, bool]:
        suffix = Path(urllib.parse.urlparse(url).path).suffix or ".img"
        return self.get_bytes(url, suffix=suffix)


class TesseractOcr:
    def __init__(self, command: str = "tesseract", psm: int = 6, language: str = "eng") -> None:
        self.command = command
        self.psm = psm
        self.language = language

    def read(self, image_path: Path) -> OcrResult:
        executable = shutil.which(self.command)
        if executable is None:
            raise RuntimeError(
                f"{self.command!r} is not installed. Install Tesseract or use --ocr-text-dir "
                "with precomputed OCR text fixtures."
            )
        process = subprocess.run(
            [
                executable,
                str(image_path),
                "stdout",
                "--psm",
                str(self.psm),
                "-l",
                self.language,
                "-c",
                "preserve_interword_spaces=1",
                "tsv",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return ocr_result_from_tesseract_tsv(process.stdout)


def parse_optional_int(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_article_date(value: str) -> str | None:
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return dt.datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def is_prediction_image_candidate(image: PredictionImage) -> bool:
    path = urllib.parse.urlparse(image.image_url).path
    suffix = Path(path).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS or "/images/news/" not in path:
        return False
    haystack = " ".join(
        [
            image.article_title,
            image.alt_text,
            image.context,
            Path(path).name,
            image.linked_url or "",
        ]
    ).lower()
    if not any(term in haystack for term in PREDICTION_TERMS):
        return False
    reject_terms = ("cover", "table of contents", "logo")
    return not any(term in haystack for term in reject_terms)


def discover_prediction_images(html: str, *, page_url: str) -> list[PredictionImage]:
    parser = NewsPredictionImageParser(page_url)
    parser.feed(html)
    seen: set[str] = set()
    images: list[PredictionImage] = []
    for image in parser.images:
        if image.image_url in seen:
            continue
        seen.add(image.image_url)
        images.append(image)
    return images


def ocr_result_from_text(text: str, *, confidence: float | None = None) -> OcrResult:
    return OcrResult(text=normalize_ocr_text(text), mean_confidence=confidence, tokens=[])


def ocr_result_from_tesseract_tsv(tsv_text: str) -> OcrResult:
    reader = csv.DictReader(tsv_text.splitlines(), delimiter="\t")
    line_words: dict[tuple[int, int, int], list[tuple[int, str]]] = {}
    tokens: list[OcrToken] = []
    confidences: list[float] = []
    for row in reader:
        text = clean_text(row.get("text", ""))
        if not text:
            continue
        confidence = parse_float(row.get("conf"))
        if confidence is not None and confidence >= 0:
            confidences.append(confidence)
        left = parse_optional_int(row.get("left"))
        top = parse_optional_int(row.get("top"))
        width = parse_optional_int(row.get("width"))
        height = parse_optional_int(row.get("height"))
        tokens.append(OcrToken(text=text, confidence=confidence, left=left, top=top, width=width, height=height))
        key = (
            parse_optional_int(row.get("block_num")) or 0,
            parse_optional_int(row.get("par_num")) or 0,
            parse_optional_int(row.get("line_num")) or 0,
        )
        line_words.setdefault(key, []).append((left or 0, text))
    lines = [
        " ".join(word for _, word in sorted(words))
        for _, words in sorted(line_words.items())
        if words
    ]
    mean_confidence = sum(confidences) / len(confidences) if confidences else None
    return OcrResult(
        text=normalize_ocr_text("\n".join(lines)),
        mean_confidence=mean_confidence,
        tokens=tokens,
    )


def normalize_ocr_text(text: str) -> str:
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u00a0", " ")
    return "\n".join(clean_text(line) for line in text.splitlines() if clean_text(line))


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_pct(value: str) -> float | None:
    match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%", value)
    return float(match.group(1)) if match else None


def parse_date(value: str) -> str | None:
    text = clean_text(value)
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return dt.datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def parse_prediction_rows(image: PredictionImage, ocr: OcrResult) -> list[PredictionRow]:
    lines = normalize_ocr_text(ocr.text).splitlines()
    rows: list[PredictionRow] = []
    for line in lines:
        comparison_rows = parse_comparison_line(image, ocr, line, len(rows) + 1)
        if comparison_rows:
            rows.extend(comparison_rows)
            continue
        rows.extend(parse_projection_line(image, ocr, line, len(rows) + 1))
    return [finalize_prediction_row(row) for row in rows]


def parse_projection_line(
    image: PredictionImage,
    ocr: OcrResult,
    line: str,
    row_ordinal: int,
) -> list[PredictionRow]:
    if should_skip_ocr_line(line):
        return []
    money_matches = list(re.finditer(r"\$[\d,]+(?:\.\d+)?", line))
    if len(money_matches) < 2:
        return []
    prefix = clean_text(line[: money_matches[0].start()])
    title, distributor = split_title_distributor(prefix)
    if not title:
        return []
    pct_matches = list(re.finditer(r"([+-]?\d+(?:\.\d+)?)\s*%", line[money_matches[1].end() :]))
    pct_change = float(pct_matches[0].group(1)) if len(pct_matches) > 1 else None
    pct_vs_prediction = float(pct_matches[-1].group(1)) if pct_matches else None
    return [
        base_prediction_row(
            image,
            ocr,
            table_kind="weekend_projection",
            row_ordinal=row_ordinal,
            row_label=title,
            source_movie_title=title,
            distributor=distributor,
            metric="weekend_gross",
            actual_usd=parse_money(money_matches[0].group(0)),
            predicted_usd=parse_money(money_matches[1].group(0)),
            pct_change=pct_change,
            pct_vs_prediction=pct_vs_prediction,
            raw_text=line,
        )
    ]


def parse_comparison_line(
    image: PredictionImage,
    ocr: OcrResult,
    line: str,
    row_ordinal: int,
) -> list[PredictionRow]:
    predicted_matches = list(
        re.finditer(
            r"\b(?P<label>Predicted\s+Fri[- ]Sun|Predicted\s+Opening|Final\s+opening\s+prediction|Predicted\s+Total)\b\s*"
            r"(?P<value>\$[\d,]+)",
            line,
            re.IGNORECASE,
        )
    )
    if predicted_matches:
        rows: list[PredictionRow] = []
        for index, predicted_match in enumerate(predicted_matches):
            row_label = clean_text(predicted_match.group("label")).title().replace("Fri-Sun", "Fri-Sun")
            metric = comparison_prediction_metric(row_label)
            rows.append(
                base_prediction_row(
                    image,
                    ocr,
                    table_kind="comparison_prediction",
                    row_ordinal=row_ordinal + index,
                    row_label=row_label,
                    source_movie_title=None,
                    distributor=None,
                    metric=metric,
                    value_usd=parse_money(predicted_match.group("value")),
                    predicted_usd=parse_money(predicted_match.group("value")),
                    raw_text=line,
                )
            )
        return rows
    predicted_match = re.search(
        r"\b(?P<label>Previews\s+Prediction|Fundamentals\s+Prediction)\b\s*"
        r"(?P<value>\$[\d,]+)",
        line,
        re.IGNORECASE,
    )
    if predicted_match:
        row_label = clean_text(predicted_match.group("label")).title().replace("Fri-Sun", "Fri-Sun")
        return [
            base_prediction_row(
                image,
                ocr,
                table_kind="comparison_prediction",
                row_ordinal=row_ordinal,
                row_label=row_label,
                source_movie_title=None,
                distributor=None,
                metric="opening_prediction_component",
                value_usd=parse_money(predicted_match.group("value")),
                predicted_usd=parse_money(predicted_match.group("value")),
                raw_text=line,
            )
        ]
    if should_skip_ocr_line(line):
        return []
    date_match = re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", line)
    if date_match is None:
        return []
    money_matches = list(re.finditer(r"\$[\d,]+(?:\.\d+)?", line))
    if len(money_matches) < 2:
        return []
    title = clean_text(line[: date_match.start()])
    if not title:
        return []
    multiplier = None
    tail = clean_text(line[money_matches[-1].end() :])
    multiplier_match = re.search(r"\b(\d+(?:\.\d+)?)\b", tail)
    if multiplier_match:
        multiplier = parse_multiplier(multiplier_match.group(1))
    weekend_usd = parse_money(money_matches[2].group(0)) if len(money_matches) >= 3 else None
    return [
        base_prediction_row(
            image,
            ocr,
            table_kind="opening_comparison",
            row_ordinal=row_ordinal,
            row_label=title,
            source_movie_title=title,
            distributor=None,
            metric="historical_comp",
            actual_usd=parse_money(money_matches[0].group(0)),
            predicted_usd=None,
            weekend_usd=weekend_usd,
            release_date=parse_date(date_match.group(0)),
            multiplier=multiplier,
            raw_text=line,
        )
    ]


def should_skip_ocr_line(line: str) -> bool:
    lowered = line.lower()
    return (
        not lowered
        or lowered.startswith("movie ")
        or lowered.startswith("release date ")
        or lowered.startswith("top 10 ")
        or lowered.startswith("medians")
        or "reported weekend box office" in lowered
        or "actual predicted" in lowered
    )


def parse_multiplier(value: str) -> float | None:
    multiplier = parse_float(value)
    if multiplier is None:
        return None
    if "." not in value and 100 <= multiplier < 1000:
        return multiplier / 100
    return multiplier


def comparison_prediction_metric(row_label: str) -> str:
    if re.search(r"\bFri[- ]Sun\b", row_label, flags=re.IGNORECASE):
        return "predicted_weekend"
    if re.search(r"\btotal\b", row_label, flags=re.IGNORECASE):
        return "predicted_total"
    if re.search(r"\bfinal\b", row_label, flags=re.IGNORECASE):
        return "final_opening_prediction"
    return "predicted_opening"


def split_title_distributor(prefix: str) -> tuple[str | None, str | None]:
    text = clean_text(prefix)
    if not text:
        return None, None
    for distributor in sorted(COMMON_DISTRIBUTORS, key=len, reverse=True):
        pattern = rf"\s+{re.escape(distributor)}$"
        if re.search(pattern, text, flags=re.IGNORECASE):
            title = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
            return (title or None), distributor
    parts = re.split(r"\s{2,}", text)
    if len(parts) >= 2:
        return clean_text(" ".join(parts[:-1])), clean_text(parts[-1])
    return text, None


def base_prediction_row(
    image: PredictionImage,
    ocr: OcrResult,
    *,
    table_kind: str,
    row_ordinal: int,
    row_label: str,
    source_movie_title: str | None,
    distributor: str | None,
    metric: str,
    value_usd: int | None = None,
    actual_usd: int | None = None,
    predicted_usd: int | None = None,
    weekend_usd: int | None = None,
    release_date: str | None = None,
    multiplier: float | None = None,
    pct_change: float | None = None,
    pct_vs_prediction: float | None = None,
    raw_text: str,
) -> PredictionRow:
    return PredictionRow(
        article_title=image.article_title,
        article_date=image.article_date,
        image_url=image.image_url,
        linked_url=image.linked_url,
        table_kind=table_kind,
        row_ordinal=row_ordinal,
        row_label=row_label,
        source_movie_title=source_movie_title,
        distributor=distributor,
        metric=metric,
        value_usd=value_usd,
        actual_usd=actual_usd,
        predicted_usd=predicted_usd,
        weekend_usd=weekend_usd,
        release_date=release_date,
        multiplier=multiplier,
        pct_change=pct_change,
        pct_vs_prediction=pct_vs_prediction,
        confidence=ocr.mean_confidence,
        raw_text=raw_text,
        raw_ocr_text=ocr.text,
    )


def finalize_prediction_row(row: PredictionRow) -> PredictionRow:
    source_row_key = f"{row.table_kind}:{row.row_ordinal}"
    normalized_title = normalize_movie_title(row.source_movie_title) if row.source_movie_title else None
    source_movie_id = prediction_source_movie_id(row, normalized_title=normalized_title) if normalized_title else None
    return replace(
        row,
        source_row_key=source_row_key,
        source_movie_id=source_movie_id,
        normalized_movie_title=normalized_title,
        match_status="unmatched" if normalized_title else "not_applicable",
        match_method=None if normalized_title else "no_source_movie_title",
        match_score=None if normalized_title else 0.0,
        match_notes=None if normalized_title else "OCR row is a summary or aggregate prediction row",
    )


def normalize_movie_title(value: str) -> str:
    return movie_identity.normalize_title(value)


def prediction_source_movie_id(row: PredictionRow, *, normalized_title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", normalized_title).strip("-") or "unknown-title"
    year = infer_prediction_row_year(row)
    date_key = row.release_date or row.article_date or "unknown-date"
    return f"{SOURCE_THE_NUMBERS_PREDICTIONS}:{slug}:{year or 'unknown-year'}:{date_key}"


def infer_prediction_row_year(row: PredictionRow) -> int | None:
    for value in (row.release_date, row.article_date):
        if value and re.match(r"\d{4}", value):
            return int(value[:4])
    return None


def load_ocr_fixture(image_path: Path, image_url: str, fixture_dir: Path) -> str | None:
    candidates = [
        fixture_dir / f"{image_path.stem}.txt",
        fixture_dir / f"{Path(urllib.parse.urlparse(image_url).path).stem}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return None


def collect_prediction_image_results(args: argparse.Namespace) -> list[PredictionImageResult]:
    html_fetcher = HtmlFetcher(
        args.cache_dir / "html",
        refresh=args.refresh,
        offline=args.offline,
        delay_seconds=args.delay_seconds,
        user_agent=args.user_agent,
    )
    image_fetcher = ImageFetcher(
        args.cache_dir / "images",
        refresh=args.refresh,
        offline=args.offline,
        delay_seconds=args.delay_seconds,
        user_agent=args.user_agent,
    )
    html, _, _ = html_fetcher.get(args.news_url)
    images = discover_prediction_images(html, page_url=args.news_url)
    results: list[PredictionImageResult] = []
    ocr_engine = TesseractOcr(args.tesseract_command, args.tesseract_psm, args.tesseract_language)
    fetched_at = dt.datetime.now(dt.timezone.utc).isoformat()
    for image in images[: args.limit or None]:
        _, image_path, _ = image_fetcher.get(image.image_url)
        fixture_text = load_ocr_fixture(image_path, image.image_url, args.ocr_text_dir) if args.ocr_text_dir else None
        if fixture_text is not None:
            ocr = ocr_result_from_text(fixture_text)
        elif args.ocr == "none":
            ocr = None
        else:
            ocr = ocr_engine.read(image_path)
        if (
            ocr is not None
            and ocr.mean_confidence is not None
            and ocr.mean_confidence < args.min_ocr_confidence
        ):
            parsed_rows = []
        else:
            parsed_rows = parse_prediction_rows(image, ocr) if ocr is not None else []
        if parsed_rows or args.include_empty:
            results.append(
                PredictionImageResult(
                    image=image,
                    image_cache_path=image_path,
                    ocr=ocr,
                    rows=parsed_rows,
                    fetched_at=fetched_at,
                )
            )
    return results


def collect_prediction_rows(args: argparse.Namespace) -> list[PredictionRow]:
    rows: list[PredictionRow] = []
    for result in collect_prediction_image_results(args):
        rows.extend(result.rows)
    return rows


def collect_prediction_images(args: argparse.Namespace) -> list[PredictionImage]:
    html_fetcher = HtmlFetcher(
        args.cache_dir / "html",
        refresh=args.refresh,
        offline=args.offline,
        delay_seconds=args.delay_seconds,
        user_agent=args.user_agent,
    )
    html, _, _ = html_fetcher.get(args.news_url)
    images = discover_prediction_images(html, page_url=args.news_url)
    return images[: args.limit or None]


def image_to_dict(image: PredictionImage) -> dict[str, Any]:
    return image.__dict__.copy()


def write_images(images: list[PredictionImage], args: argparse.Namespace) -> None:
    output = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8", newline="")
    with output:
        if args.format == "json":
            json.dump([image_to_dict(image) for image in images], output, indent=2, sort_keys=True)
            output.write("\n")
            return
        fieldnames = list(image_to_dict(images[0]).keys()) if images else [
            "article_id",
            "article_title",
            "article_date",
            "page_url",
            "image_url",
            "linked_url",
            "alt_text",
            "width",
            "height",
            "context",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for image in images:
            writer.writerow(image_to_dict(image))


def row_to_dict(row: PredictionRow, *, include_raw_ocr: bool) -> dict[str, Any]:
    values = row.__dict__.copy()
    if not include_raw_ocr:
        values.pop("raw_ocr_text", None)
    return values


def write_rows(rows: list[PredictionRow], args: argparse.Namespace) -> None:
    output = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8", newline="")
    with output:
        if args.format == "json":
            json.dump(
                [row_to_dict(row, include_raw_ocr=args.include_raw_ocr) for row in rows],
                output,
                indent=2,
                sort_keys=True,
            )
            output.write("\n")
            return
        fieldnames = list(row_to_dict(rows[0], include_raw_ocr=args.include_raw_ocr).keys()) if rows else [
            "article_title",
            "article_date",
            "image_url",
            "linked_url",
            "table_kind",
            "row_ordinal",
            "row_label",
            "source_movie_title",
            "distributor",
            "metric",
            "value_usd",
            "actual_usd",
            "predicted_usd",
            "weekend_usd",
            "release_date",
            "multiplier",
            "pct_change",
            "pct_vs_prediction",
            "confidence",
            "raw_text",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row_to_dict(row, include_raw_ocr=args.include_raw_ocr))


def initialize_database(conn: Any) -> None:
    acquire_schema_init_lock(conn)
    movie_identity.ensure_movie_identity_schema(conn)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS the_numbers_prediction_articles (
            article_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            article_key TEXT NOT NULL UNIQUE,
            source_article_id TEXT,
            page_url TEXT NOT NULL,
            article_title TEXT NOT NULL,
            article_date DATE,
            fetched_at TIMESTAMPTZ,
            parser_version TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS the_numbers_prediction_images (
            image_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            article_id BIGINT NOT NULL REFERENCES the_numbers_prediction_articles(article_id),
            image_url TEXT NOT NULL UNIQUE,
            linked_url TEXT,
            alt_text TEXT,
            width INTEGER,
            height INTEGER,
            context TEXT,
            raw_image_cache_path TEXT,
            raw_image_sha256 TEXT,
            raw_ocr_text TEXT,
            ocr_confidence DOUBLE PRECISION,
            fetched_at TIMESTAMPTZ NOT NULL,
            parser_version TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS the_numbers_prediction_rows (
            prediction_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            image_id BIGINT NOT NULL REFERENCES the_numbers_prediction_images(image_id),
            source_row_key TEXT NOT NULL,
            source_movie_id TEXT,
            source_movie_title TEXT,
            normalized_movie_title TEXT,
            distributor TEXT,
            table_kind TEXT NOT NULL,
            row_ordinal INTEGER NOT NULL,
            row_label TEXT NOT NULL,
            market TEXT NOT NULL,
            currency TEXT NOT NULL,
            metric TEXT NOT NULL,
            value_usd BIGINT,
            actual_usd BIGINT,
            predicted_usd BIGINT,
            weekend_usd BIGINT,
            release_date DATE,
            multiplier DOUBLE PRECISION,
            pct_change DOUBLE PRECISION,
            pct_vs_prediction DOUBLE PRECISION,
            confidence DOUBLE PRECISION,
            raw_text TEXT NOT NULL,
            raw_ocr_text TEXT NOT NULL,
            movie_id BIGINT REFERENCES movies(movie_id),
            match_status TEXT NOT NULL,
            match_method TEXT,
            match_score DOUBLE PRECISION,
            match_notes TEXT,
            fetched_at TIMESTAMPTZ NOT NULL,
            parser_version TEXT NOT NULL,
            UNIQUE(image_id, source_row_key)
        );

        CREATE INDEX IF NOT EXISTS idx_the_numbers_prediction_rows_movie_id
            ON the_numbers_prediction_rows(movie_id);
        CREATE INDEX IF NOT EXISTS idx_the_numbers_prediction_rows_title
            ON the_numbers_prediction_rows(normalized_movie_title);
        CREATE INDEX IF NOT EXISTS idx_the_numbers_prediction_rows_fetched
            ON the_numbers_prediction_rows(fetched_at);
        """
    )


def import_prediction_results(conn: Any, results: list[PredictionImageResult]) -> tuple[int, int, int]:
    initialize_database(conn)
    article_count = 0
    image_count = 0
    row_count = 0
    for result in results:
        article_id = upsert_prediction_article(conn, result.image, fetched_at=result.fetched_at)
        image_id = upsert_prediction_image(conn, article_id=article_id, result=result)
        matched_rows = match_prediction_rows(conn, result.rows)
        clear_prediction_rows(conn, image_id=image_id)
        upsert_prediction_rows(conn, image_id=image_id, rows=matched_rows, fetched_at=result.fetched_at)
        article_count += 1
        image_count += 1
        row_count += len(matched_rows)
    return article_count, image_count, row_count


def upsert_prediction_article(conn: Any, image: PredictionImage, *, fetched_at: str) -> int:
    article_key = prediction_article_key(image)
    conn.execute(
        """
        INSERT INTO the_numbers_prediction_articles (
            article_key, source_article_id, page_url, article_title, article_date,
            fetched_at, parser_version, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT(article_key) DO UPDATE SET
            source_article_id = excluded.source_article_id,
            page_url = excluded.page_url,
            article_title = excluded.article_title,
            article_date = excluded.article_date,
            fetched_at = excluded.fetched_at,
            parser_version = excluded.parser_version,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            article_key,
            image.article_id,
            image.page_url,
            image.article_title,
            image.article_date,
            fetched_at,
            PARSER_VERSION,
        ),
    )
    return int(
        conn.execute(
            "SELECT article_id FROM the_numbers_prediction_articles WHERE article_key = %s",
            (article_key,),
        ).fetchone()[0]
    )


def upsert_prediction_image(conn: Any, *, article_id: int, result: PredictionImageResult) -> int:
    image = result.image
    conn.execute(
        """
        INSERT INTO the_numbers_prediction_images (
            article_id, image_url, linked_url, alt_text, width, height, context,
            raw_image_cache_path, raw_image_sha256, raw_ocr_text, ocr_confidence,
            fetched_at, parser_version, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT(image_url) DO UPDATE SET
            article_id = excluded.article_id,
            linked_url = excluded.linked_url,
            alt_text = excluded.alt_text,
            width = excluded.width,
            height = excluded.height,
            context = excluded.context,
            raw_image_cache_path = excluded.raw_image_cache_path,
            raw_image_sha256 = excluded.raw_image_sha256,
            raw_ocr_text = excluded.raw_ocr_text,
            ocr_confidence = excluded.ocr_confidence,
            fetched_at = excluded.fetched_at,
            parser_version = excluded.parser_version,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            article_id,
            image.image_url,
            image.linked_url,
            image.alt_text,
            image.width,
            image.height,
            image.context,
            str(result.image_cache_path) if result.image_cache_path is not None else None,
            file_sha256(result.image_cache_path),
            result.ocr.text if result.ocr is not None else None,
            result.ocr.mean_confidence if result.ocr is not None else None,
            result.fetched_at,
            PARSER_VERSION,
        ),
    )
    return int(
        conn.execute(
            "SELECT image_id FROM the_numbers_prediction_images WHERE image_url = %s",
            (image.image_url,),
        ).fetchone()[0]
    )


def upsert_prediction_rows(conn: Any, *, image_id: int, rows: list[PredictionRow], fetched_at: str) -> None:
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO the_numbers_prediction_rows (
            image_id, source_row_key, source_movie_id, source_movie_title,
            normalized_movie_title, distributor, table_kind, row_ordinal, row_label,
            market, currency, metric, value_usd, actual_usd, predicted_usd,
            weekend_usd, release_date, multiplier, pct_change, pct_vs_prediction,
            confidence, raw_text, raw_ocr_text, movie_id, match_status, match_method,
            match_score, match_notes, fetched_at, parser_version
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(image_id, source_row_key) DO UPDATE SET
            source_movie_id = excluded.source_movie_id,
            source_movie_title = excluded.source_movie_title,
            normalized_movie_title = excluded.normalized_movie_title,
            distributor = excluded.distributor,
            table_kind = excluded.table_kind,
            row_ordinal = excluded.row_ordinal,
            row_label = excluded.row_label,
            market = excluded.market,
            currency = excluded.currency,
            metric = excluded.metric,
            value_usd = excluded.value_usd,
            actual_usd = excluded.actual_usd,
            predicted_usd = excluded.predicted_usd,
            weekend_usd = excluded.weekend_usd,
            release_date = excluded.release_date,
            multiplier = excluded.multiplier,
            pct_change = excluded.pct_change,
            pct_vs_prediction = excluded.pct_vs_prediction,
            confidence = excluded.confidence,
            raw_text = excluded.raw_text,
            raw_ocr_text = excluded.raw_ocr_text,
            movie_id = excluded.movie_id,
            match_status = excluded.match_status,
            match_method = excluded.match_method,
            match_score = excluded.match_score,
            match_notes = excluded.match_notes,
            fetched_at = excluded.fetched_at,
            parser_version = excluded.parser_version
        """,
        [
            (
                image_id,
                row.source_row_key,
                row.source_movie_id,
                row.source_movie_title,
                row.normalized_movie_title,
                row.distributor,
                row.table_kind,
                row.row_ordinal,
                row.row_label,
                DOMESTIC_MARKET,
                DOMESTIC_CURRENCY,
                row.metric,
                row.value_usd,
                row.actual_usd,
                row.predicted_usd,
                row.weekend_usd,
                row.release_date,
                row.multiplier,
                row.pct_change,
                row.pct_vs_prediction,
                row.confidence,
                row.raw_text,
                row.raw_ocr_text,
                row.movie_id,
                row.match_status,
                row.match_method,
                row.match_score,
                row.match_notes,
                fetched_at,
                PARSER_VERSION,
            )
            for row in rows
        ],
    )


def clear_prediction_rows(conn: Any, *, image_id: int) -> None:
    conn.execute("DELETE FROM the_numbers_prediction_rows WHERE image_id = %s", (image_id,))


def match_prediction_rows(conn: Any, rows: list[PredictionRow]) -> list[PredictionRow]:
    candidates = load_movie_candidates(conn)
    matched: list[PredictionRow] = []
    for row in rows:
        match = match_prediction_row(conn, row, candidates)
        if match.movie_id is not None and row.source_movie_id and row.source_movie_title:
            upsert_prediction_movie_source_id(
                conn,
                movie_id=match.movie_id,
                row=row,
                match_status=match.status,
                match_method=match.method,
                match_score=match.score,
            )
        matched.append(
            replace(
                row,
                movie_id=match.movie_id,
                match_status=match.status,
                match_method=match.method,
                match_score=match.score,
                match_notes=match.notes,
            )
        )
    return matched


def match_prediction_row(conn: Any, row: PredictionRow, candidates: list[MovieCandidate]) -> MovieMatch:
    if not row.normalized_movie_title or not row.source_movie_id:
        return MovieMatch(None, "not_applicable", "no_source_movie_title", 0.0, "No source movie title")
    source_id_match = find_prediction_source_id_match(conn, row)
    if source_id_match is not None:
        return source_id_match
    matches = [candidate for candidate in candidates if candidate.normalized_title == row.normalized_movie_title]
    if not matches:
        return MovieMatch(None, "unmatched", "normalized_exact", 0.0, "No movie title matched")
    if row.release_date is not None:
        exact_date_matches = [candidate for candidate in matches if candidate.release_date == row.release_date]
        if exact_date_matches:
            candidate = preferred_movie_candidate(exact_date_matches)
            status = "matched" if candidate.movie_url is not None else "provisional"
            return MovieMatch(candidate.movie_id, status, "normalized_exact_release_date", 1.0, None)
    if len(matches) == 1:
        candidate = matches[0]
        status = "matched" if candidate.movie_url is not None else "provisional"
        return MovieMatch(candidate.movie_id, status, "normalized_exact", 1.0, None)
    row_year = infer_prediction_row_year(row)
    if row_year is not None:
        year_matches = [candidate for candidate in matches if candidate.release_year == row_year]
        if len(year_matches) == 1:
            candidate = year_matches[0]
            status = "matched" if candidate.movie_url is not None else "provisional"
            return MovieMatch(candidate.movie_id, status, "normalized_exact_year", 0.9, None)
    return MovieMatch(None, "ambiguous", "normalized_exact", 0.5, "Multiple movies share the title")


def find_prediction_source_id_match(conn: Any, row: PredictionRow) -> MovieMatch | None:
    if not relation_exists(conn, "movie_source_ids"):
        return None
    db_row = conn.execute(
        """
        SELECT m.movie_id, m.movie_url, src.match_status, src.match_score
        FROM movie_source_ids src
        JOIN movies m ON m.movie_id = src.movie_id
        WHERE src.source = %s
          AND src.source_movie_id = %s
        LIMIT 1
        """,
        (SOURCE_THE_NUMBERS_PREDICTIONS, row.source_movie_id),
    ).fetchone()
    if db_row is None:
        return None
    status = str(db_row[2]) if db_row[2] in {"matched", "provisional"} else "matched"
    return MovieMatch(
        int(db_row[0]),
        status,
        "the_numbers_predictions_source_id",
        float(db_row[3]) if db_row[3] is not None else 1.0,
        f"Matched existing The Numbers prediction source id {row.source_movie_id}",
    )


def upsert_prediction_movie_source_id(
    conn: Any,
    *,
    movie_id: int,
    row: PredictionRow,
    match_status: str,
    match_method: str | None,
    match_score: float | None,
) -> None:
    movie_identity.upsert_movie_source_id(
        conn,
        movie_id=movie_id,
        source=SOURCE_THE_NUMBERS_PREDICTIONS,
        source_movie_id=row.source_movie_id or "",
        source_title=row.source_movie_title,
        match_status=match_status,
        match_method=match_method,
        match_score=match_score,
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


def preferred_movie_candidate(candidates: list[MovieCandidate]) -> MovieCandidate:
    return sorted(candidates, key=lambda candidate: (candidate.movie_url is None, candidate.movie_id))[0]


def prediction_article_key(image: PredictionImage) -> str:
    if image.article_id:
        return f"{image.page_url}#{image.article_id}"
    digest = hashlib.sha256(
        "|".join([image.page_url, image.article_title, image.article_date or ""]).encode("utf-8")
    ).hexdigest()[:16]
    return f"{image.page_url}#article-{digest}"


def file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_prediction_results_from_args(args: argparse.Namespace) -> tuple[int, int, int]:
    results = collect_prediction_image_results(args)
    if args.dry_run:
        row_count = sum(len(result.rows) for result in results)
        return len(results), len(results), row_count
    conn = connect_database(args.database_url)
    try:
        counts = import_prediction_results(conn, results)
        conn.commit()
        return counts
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect OCR-parsed The Numbers prediction image tables.")
    add_database_arg(parser)
    parser.add_argument("--news-url", default=DEFAULT_NEWS_URL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--delay-seconds", type=float, default=MIN_DELAY_SECONDS)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--limit", type=int, default=0, help="Maximum candidate images to process; 0 means all.")
    parser.add_argument("--list-images", action="store_true", help="Only list discovered candidate table images; do not OCR.")
    parser.add_argument("--ocr", choices=("none", "tesseract"), default="tesseract")
    parser.add_argument("--ocr-text-dir", type=Path, help="Directory of precomputed OCR .txt files keyed by image hash or filename stem.")
    parser.add_argument("--tesseract-command", default="tesseract")
    parser.add_argument("--tesseract-psm", type=int, default=6)
    parser.add_argument("--tesseract-language", default="eng")
    parser.add_argument("--min-ocr-confidence", type=float, default=DEFAULT_MIN_OCR_CONFIDENCE)
    parser.add_argument("--include-empty", action="store_true", help="Process images even when OCR yields no rows.")
    parser.add_argument("--include-raw-ocr", action="store_true")
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    parser.add_argument("--output", help="Write parsed rows to CSV/JSON instead of importing to PostgreSQL. Use - for stdout.")
    parser.add_argument("--dry-run", action="store_true", help="Parse but do not write database changes.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.list_images:
        if args.output is None:
            args.output = "-"
        write_images(collect_prediction_images(args), args)
        return 0
    if args.output is None:
        article_count, image_count, row_count = import_prediction_results_from_args(args)
        action = "Would import" if args.dry_run else "Imported"
        print(
            f"{action} {article_count} articles, {image_count} images, and {row_count} prediction rows.",
            file=sys.stderr,
        )
        return 0
    rows = collect_prediction_rows(args)
    write_rows(rows, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
