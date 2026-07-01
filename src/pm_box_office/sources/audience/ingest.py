#!/usr/bin/env python3
"""Ingest IMDb and Letterboxd audience-count snapshots for box-office movies.

The pipeline is cache-first and resumable. Candidate movies come from The
Numbers data already in the database plus the The Numbers release schedule.
IMDb ratings/vote counts are read from the official non-commercial TSV files;
Letterboxd counts are parsed from public aggregate film pages.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import gzip
import hashlib
from html.parser import HTMLParser
import io
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
from typing import Any, Iterable

from pm_box_office.db.connection import connect_database, database_url_from_env


THE_NUMBERS_BASE_URL = "https://www.the-numbers.com"
THE_NUMBERS_RELEASE_SCHEDULE_URL = f"{THE_NUMBERS_BASE_URL}/movies/release-schedule"
LETTERBOXD_BASE_URL = "https://letterboxd.com"
IMDB_DATASET_BASE_URL = "https://datasets.imdbws.com"
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
DEFAULT_CACHE_DIR = Path("data/raw/audience")
USER_AGENT_PRODUCT = "pm-box-office-audience-ingest/0.1"
USER_AGENT_COMMENT = "(+personal research; set --user-agent contact)"
DEFAULT_USER_AGENT = "random"
FALLBACK_USER_AGENT = f"{USER_AGENT_PRODUCT} {USER_AGENT_COMMENT}"
MIN_DELAY_SECONDS = 20.0
TRANSIENT_STATUSES = {429, 500, 502, 503, 504}
PARSER_VERSION = "audience_v1"
WIKIDATA_BATCH_SIZE = 25


@dataclass(frozen=True)
class ParsedTable:
    rows: list[list[str]]
    hrefs: list[list[str]]


@dataclass(frozen=True)
class TheNumbersReleaseRow:
    movie_url: str
    title: str
    release_date: str | None
    release_pattern: str | None
    distributor: str | None
    domestic_box_office_to_date_usd: int | None
    source_url: str


@dataclass(frozen=True)
class CandidateMovie:
    movie_id: int
    movie_url: str | None
    title: str
    release_year: int | None
    release_date: str | None
    normalized_title: str


@dataclass(frozen=True)
class ImdbTitle:
    tconst: str
    primary_title: str
    original_title: str | None
    start_year: int | None
    title_type: str
    is_adult: int
    genres: str | None


@dataclass(frozen=True)
class ImdbAka:
    title_id: str
    title: str
    region: str | None
    language: str | None
    types: str | None
    is_original_title: int


@dataclass(frozen=True)
class ImdbRating:
    tconst: str
    average_rating: float | None
    num_votes: int | None


@dataclass(frozen=True)
class ImdbMatch:
    movie_id: int
    tconst: str | None
    match_status: str
    match_method: str
    match_score: float | None
    notes: str | None = None


@dataclass(frozen=True)
class LetterboxdSearchResult:
    film_url: str
    letterboxd_slug: str
    title: str | None = None
    year: int | None = None


@dataclass(frozen=True)
class LetterboxdFilmPage:
    letterboxd_slug: str
    film_url: str
    source_title: str | None
    source_year: int | None
    imdb_tconst: str | None
    tmdb_id: str | None
    average_rating: float | None
    watched_count: int | None
    rating_count: int | None
    review_count: int | None
    log_count: int | None
    fan_count: int | None
    parse_status: str


@dataclass(frozen=True)
class LetterboxdMatch:
    movie_id: int
    letterboxd_slug: str | None
    match_status: str
    match_method: str
    match_score: float | None
    notes: str | None = None


@dataclass(frozen=True)
class WikidataMovieMatch:
    movie_id: int
    qid: str
    label: str
    release_year: int | None
    imdb_tconst: str | None
    tmdb_id: str | None
    letterboxd_slug: str | None
    score: float
    status: str
    notes: str | None = None


class HtmlTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[ParsedTable] = []
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._table_rows: list[list[str]] = []
        self._table_hrefs: list[list[str]] = []
        self._row_cells: list[str] = []
        self._row_hrefs: list[str] = []
        self._cell_parts: list[str] = []
        self._cell_hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "table":
            self._in_table = True
            self._table_rows = []
            self._table_hrefs = []
        elif self._in_table and tag == "tr":
            self._in_row = True
            self._row_cells = []
            self._row_hrefs = []
        elif self._in_row and tag in {"td", "th"}:
            self._in_cell = True
            self._cell_parts = []
            self._cell_hrefs = []
        elif self._in_cell and tag == "a":
            href = attrs_dict.get("href")
            if href:
                self._cell_hrefs.append(href)

    def handle_endtag(self, tag: str) -> None:
        if self._in_cell and tag in {"td", "th"}:
            self._row_cells.append(clean_text(" ".join(self._cell_parts)))
            self._row_hrefs.append(" ".join(self._cell_hrefs))
            self._in_cell = False
            self._cell_parts = []
            self._cell_hrefs = []
        elif self._in_row and tag == "tr":
            if any(self._row_cells):
                self._table_rows.append(self._row_cells)
                self._table_hrefs.append(self._row_hrefs)
            self._in_row = False
        elif self._in_table and tag == "table":
            self.tables.append(ParsedTable(rows=self._table_rows, hrefs=self._table_hrefs))
            self._in_table = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_parts.append(data)


class LetterboxdSearchParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[LetterboxdSearchResult] = []
        self._capture_href: str | None = None
        self._capture_parts: list[str] = []
        self._seen_slugs: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href") or ""
        slug = letterboxd_slug_from_url(href)
        if slug and slug not in self._seen_slugs:
            self._capture_href = href
            self._capture_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self._capture_href:
            return
        slug = letterboxd_slug_from_url(self._capture_href)
        if slug and slug not in self._seen_slugs:
            text = clean_text(" ".join(self._capture_parts)) or None
            self.results.append(
                LetterboxdSearchResult(
                    film_url=absolute_letterboxd_url(self._capture_href),
                    letterboxd_slug=slug,
                    title=text,
                    year=parse_year(text or ""),
                )
            )
            self._seen_slugs.add(slug)
        self._capture_href = None
        self._capture_parts = []

    def handle_data(self, data: str) -> None:
        if self._capture_href:
            self._capture_parts.append(data)


class CachedFetcher:
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

    def cache_path(self, url: str, suffix: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}{suffix}"

    def get_text(self, url: str) -> tuple[str, Path, bool]:
        body, cache_path, fetched = self.get_bytes(url, suffix=".html")
        return body.decode("utf-8", errors="replace"), cache_path, fetched

    def get_bytes(self, url: str, *, suffix: str) -> tuple[bytes, Path, bool]:
        cache_path = self.cache_path(url, suffix)
        if cache_path.exists() and not self.refresh:
            return cache_path.read_bytes(), cache_path, False
        if self.offline:
            raise FileNotFoundError(f"Cache miss in offline mode: {url}")

        last_error: Exception | None = None
        for attempt in range(3):
            self._wait()
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": "*/*",
                    "User-Agent": self.user_agent,
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    body = response.read()
                cache_path.write_bytes(body)
                self._last_request_at = time.monotonic()
                return body, cache_path, True
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code == 404:
                    cache_path.write_bytes(b"")
                    return b"", cache_path, True
                if exc.code not in TRANSIENT_STATUSES or attempt == 2:
                    raise
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else self.delay_seconds * (attempt + 1)
                time.sleep(delay)
            except (TimeoutError, urllib.error.URLError) as exc:
                last_error = exc
                if attempt == 2:
                    break
                time.sleep(self.delay_seconds * (attempt + 1))
        raise RuntimeError(f"GET {url} failed after retry: {last_error}")

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        delay = max(0.0, self.delay_seconds - elapsed)
        if delay:
            time.sleep(delay)


def initialize_database(conn: Any) -> None:
    conn.executescript(
        """
        CREATE SCHEMA IF NOT EXISTS analytics;

        DROP VIEW IF EXISTS analytics.box_office_audience_panel_v1;
        DROP VIEW IF EXISTS analytics.movie_audience_daily_features_v1;

        CREATE TABLE IF NOT EXISTS movies (
            movie_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            title TEXT NOT NULL,
            release_date DATE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        ALTER TABLE movies
            ADD COLUMN IF NOT EXISTS movie_url TEXT,
            ADD COLUMN IF NOT EXISTS release_year INTEGER,
            ADD COLUMN IF NOT EXISTS opusdata_id TEXT,
            ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP;

        CREATE UNIQUE INDEX IF NOT EXISTS uq_movies_movie_url_not_null
            ON movies(movie_url) WHERE movie_url IS NOT NULL;
        CREATE UNIQUE INDEX IF NOT EXISTS uq_movies_opusdata_id_not_null
            ON movies(opusdata_id) WHERE opusdata_id IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_movies_title_release_year
            ON movies(title, release_year);

        CREATE TABLE IF NOT EXISTS release_runs (
            release_run_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            movie_id BIGINT NOT NULL REFERENCES movies(movie_id),
            market TEXT NOT NULL DEFAULT 'US_CA',
            release_type TEXT,
            source TEXT NOT NULL,
            source_release_key TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP::text),
            UNIQUE(movie_id, market, source, source_release_key)
        );

        CREATE TABLE IF NOT EXISTS daily_box_office (
            daily_box_office_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            release_run_id BIGINT NOT NULL REFERENCES release_runs(release_run_id),
            box_office_date TEXT NOT NULL,
            market TEXT NOT NULL DEFAULT 'US_CA',
            day_number INTEGER,
            rank TEXT,
            gross_usd INTEGER,
            percent_yesterday DOUBLE PRECISION,
            percent_last_week DOUBLE PRECISION,
            theaters INTEGER,
            per_theater_usd INTEGER,
            cumulative_gross_usd INTEGER,
            is_preview INTEGER NOT NULL DEFAULT 0,
            is_estimate INTEGER NOT NULL DEFAULT 0,
            source TEXT NOT NULL,
            source_url TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            raw_cache_path TEXT NOT NULL,
            UNIQUE(release_run_id, box_office_date, source)
        );

        CREATE TABLE IF NOT EXISTS the_numbers_release_schedule (
            release_schedule_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            movie_url TEXT NOT NULL,
            title TEXT NOT NULL,
            release_date DATE,
            release_pattern TEXT,
            distributor TEXT,
            domestic_box_office_to_date_usd BIGINT,
            source_url TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            raw_cache_path TEXT NOT NULL,
            UNIQUE(movie_url, release_date, release_pattern)
        );

        CREATE TABLE IF NOT EXISTS imdb_titles (
            tconst TEXT PRIMARY KEY,
            primary_title TEXT NOT NULL,
            original_title TEXT,
            start_year INTEGER,
            title_type TEXT NOT NULL,
            is_adult INTEGER NOT NULL DEFAULT 0,
            genres TEXT,
            last_seen_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS movie_imdb_titles (
            movie_id BIGINT NOT NULL REFERENCES movies(movie_id),
            tconst TEXT REFERENCES imdb_titles(tconst),
            match_status TEXT NOT NULL CHECK (
                match_status IN ('matched', 'not_found', 'ambiguous', 'manual_override')
            ),
            match_method TEXT NOT NULL,
            match_score DOUBLE PRECISION,
            matched_at TEXT NOT NULL,
            notes TEXT,
            UNIQUE(movie_id),
            UNIQUE(tconst)
        );

        CREATE TABLE IF NOT EXISTS letterboxd_films (
            letterboxd_slug TEXT PRIMARY KEY,
            film_url TEXT NOT NULL,
            source_title TEXT,
            source_year INTEGER,
            imdb_tconst TEXT,
            tmdb_id TEXT,
            last_seen_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS movie_letterboxd_films (
            movie_id BIGINT NOT NULL REFERENCES movies(movie_id),
            letterboxd_slug TEXT REFERENCES letterboxd_films(letterboxd_slug),
            match_status TEXT NOT NULL CHECK (
                match_status IN ('matched', 'not_found', 'ambiguous', 'manual_override')
            ),
            match_method TEXT NOT NULL,
            match_score DOUBLE PRECISION,
            matched_at TEXT NOT NULL,
            notes TEXT,
            UNIQUE(movie_id),
            UNIQUE(letterboxd_slug)
        );

        CREATE TABLE IF NOT EXISTS imdb_title_snapshots (
            tconst TEXT NOT NULL REFERENCES imdb_titles(tconst),
            snapshot_date DATE NOT NULL,
            observed_at TIMESTAMPTZ NOT NULL,
            average_rating DOUBLE PRECISION,
            num_votes INTEGER,
            fetched_at TEXT NOT NULL,
            raw_cache_path TEXT NOT NULL,
            UNIQUE(tconst, snapshot_date)
        );

        ALTER TABLE imdb_title_snapshots
            DROP CONSTRAINT IF EXISTS imdb_title_snapshots_tconst_snapshot_date_source_kind_key;

        ALTER TABLE imdb_title_snapshots
            DROP COLUMN IF EXISTS user_review_count,
            DROP COLUMN IF EXISTS source_kind;

        CREATE UNIQUE INDEX IF NOT EXISTS uq_imdb_title_snapshots_tconst_date
            ON imdb_title_snapshots(tconst, snapshot_date);

        CREATE TABLE IF NOT EXISTS letterboxd_film_snapshots (
            letterboxd_slug TEXT NOT NULL REFERENCES letterboxd_films(letterboxd_slug),
            snapshot_date DATE NOT NULL,
            observed_at TIMESTAMPTZ NOT NULL,
            fan_count INTEGER,
            average_rating DOUBLE PRECISION,
            parse_status TEXT NOT NULL,
            source_url TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            raw_cache_path TEXT NOT NULL,
            parser_version TEXT NOT NULL,
            UNIQUE(letterboxd_slug, snapshot_date)
        );

        ALTER TABLE letterboxd_film_snapshots
            DROP COLUMN IF EXISTS watched_count,
            DROP COLUMN IF EXISTS rating_count,
            DROP COLUMN IF EXISTS review_count,
            DROP COLUMN IF EXISTS log_count;

        CREATE TABLE IF NOT EXISTS audience_ingest_state (
            movie_id BIGINT NOT NULL REFERENCES movies(movie_id),
            source TEXT NOT NULL,
            stage TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            UNIQUE(movie_id, source)
        );

        CREATE INDEX IF NOT EXISTS idx_tn_release_schedule_date
            ON the_numbers_release_schedule(release_date);
        CREATE INDEX IF NOT EXISTS idx_imdb_titles_title_year
            ON imdb_titles(primary_title, start_year);
        CREATE INDEX IF NOT EXISTS idx_imdb_snapshots_date
            ON imdb_title_snapshots(snapshot_date);
        CREATE INDEX IF NOT EXISTS idx_letterboxd_snapshots_date
            ON letterboxd_film_snapshots(snapshot_date);
        CREATE INDEX IF NOT EXISTS idx_audience_ingest_state_status
            ON audience_ingest_state(status, source);

        DROP VIEW IF EXISTS analytics.box_office_audience_panel_v1;
        DROP VIEW IF EXISTS analytics.movie_audience_daily_features_v1;

        CREATE VIEW analytics.movie_audience_daily_features_v1 AS
        WITH dates AS (
            SELECT mit.movie_id, ims.snapshot_date
            FROM movie_imdb_titles mit
            JOIN imdb_title_snapshots ims ON ims.tconst = mit.tconst
            WHERE mit.match_status IN ('matched', 'manual_override')
            UNION
            SELECT mlf.movie_id, lfs.snapshot_date
            FROM movie_letterboxd_films mlf
            JOIN letterboxd_film_snapshots lfs ON lfs.letterboxd_slug = mlf.letterboxd_slug
            WHERE mlf.match_status IN ('matched', 'manual_override')
        ),
        base AS (
            SELECT
                d.movie_id,
                d.snapshot_date,
                ims.average_rating AS imdb_average_rating,
                ims.num_votes AS imdb_num_votes,
                lfs.fan_count AS letterboxd_fan_count,
                lfs.average_rating AS letterboxd_average_rating
            FROM dates d
            LEFT JOIN movie_imdb_titles mit
              ON mit.movie_id = d.movie_id
             AND mit.match_status IN ('matched', 'manual_override')
            LEFT JOIN imdb_title_snapshots ims
              ON ims.tconst = mit.tconst
             AND ims.snapshot_date = d.snapshot_date
            LEFT JOIN movie_letterboxd_films mlf
              ON mlf.movie_id = d.movie_id
             AND mlf.match_status IN ('matched', 'manual_override')
            LEFT JOIN letterboxd_film_snapshots lfs
              ON lfs.letterboxd_slug = mlf.letterboxd_slug
             AND lfs.snapshot_date = d.snapshot_date
        )
        SELECT
            b.*,
            b.imdb_num_votes - LAG(b.imdb_num_votes, 1) OVER (
                PARTITION BY b.movie_id ORDER BY b.snapshot_date
            ) AS imdb_num_votes_delta_1d,
            b.imdb_num_votes - LAG(b.imdb_num_votes, 7) OVER (
                PARTITION BY b.movie_id ORDER BY b.snapshot_date
            ) AS imdb_num_votes_delta_7d
        FROM base b;

        CREATE VIEW analytics.box_office_audience_panel_v1 AS
        WITH opening_dates AS (
            SELECT
                rr.movie_id,
                MIN(dbo.box_office_date::date) AS opening_date
            FROM release_runs rr
            JOIN daily_box_office dbo ON dbo.release_run_id = rr.release_run_id
            WHERE dbo.is_preview = 0
              AND dbo.gross_usd IS NOT NULL
            GROUP BY rr.movie_id
        )
        SELECT
            rr.movie_id,
            rr.release_run_id,
            dbo.box_office_date::date AS box_office_date,
            od.opening_date,
            dbo.box_office_date::date - od.opening_date AS movie_time_day,
            dbo.gross_usd,
            dbo.theaters,
            dbo.cumulative_gross_usd,
            aud.snapshot_date AS audience_snapshot_date,
            aud.imdb_average_rating,
            aud.imdb_num_votes,
            aud.letterboxd_fan_count,
            aud.letterboxd_average_rating
        FROM daily_box_office dbo
        JOIN release_runs rr ON rr.release_run_id = dbo.release_run_id
        LEFT JOIN opening_dates od ON od.movie_id = rr.movie_id
        LEFT JOIN LATERAL (
            SELECT *
            FROM analytics.movie_audience_daily_features_v1 f
            WHERE f.movie_id = rr.movie_id
              AND f.snapshot_date <= dbo.box_office_date::date
            ORDER BY f.snapshot_date DESC
            LIMIT 1
        ) aud ON TRUE;
        """
    )


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def generated_user_agent() -> str:
    try:
        from fake_useragent import UserAgent  # type: ignore[import-not-found]
    except ImportError:
        return FALLBACK_USER_AGENT
    try:
        browser_user_agent = UserAgent().random
    except Exception:
        return FALLBACK_USER_AGENT
    if not browser_user_agent:
        return FALLBACK_USER_AGENT
    return f"{browser_user_agent} {USER_AGENT_PRODUCT} {USER_AGENT_COMMENT}"


def resolve_user_agent(value: str) -> str:
    return generated_user_agent() if value.strip().lower() == "random" else value


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()


def normalize_title(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s*\(\d{4}\)\s*$", "", text)
    text = text.lower()
    text = re.sub(r"\b(disney|marvel|warner bros|universal|paramount|sony)'?s\b", " ", text)
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def strip_title_year(value: str) -> str:
    return re.sub(r"\s*\(\d{4}\)\s*$", "", value).strip()


def reorder_trailing_article(value: str) -> str | None:
    match = re.match(r"(.+?)\s+(The|A|An)$", value.strip(), re.IGNORECASE)
    if match:
        return f"{match.group(2)} {match.group(1)}"
    match = re.match(r"(.+?),\s*(The|A|An)$", value.strip(), re.IGNORECASE)
    if match:
        return f"{match.group(2)} {match.group(1)}"
    return None


def movie_url_title(value: str | None) -> str | None:
    if not value:
        return None
    tail = urllib.parse.unquote(value.rstrip("/").split("/")[-1])
    tail = re.sub(r"\(\d{4}(?:-[^)]+)?\)$", "", tail)
    tail = clean_text(tail.replace("-", " "))
    return tail or None


def title_variants(title: str, movie_url: str | None = None) -> list[str]:
    variants: list[str] = []
    for value in (title, strip_title_year(title), movie_url_title(movie_url)):
        if not value:
            continue
        variants.append(value)
        reordered = reorder_trailing_article(value)
        if reordered:
            variants.append(reordered)
    deduped: list[str] = []
    seen: set[str] = set()
    for value in variants:
        cleaned = clean_text(value)
        key = normalize_title(cleaned)
        if cleaned and key not in seen:
            deduped.append(cleaned)
            seen.add(key)
    return deduped


def strip_release_qualifier(value: str) -> str:
    return re.sub(
        r"\s*\((?:wide|limited|imax|3d|special engagement|re-release|event|fathom event)[^)]*\)\s*$",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()


def parse_year(value: str) -> int | None:
    match = re.search(r"\b(19\d{2}|20\d{2})\b", value)
    return int(match.group(1)) if match else None


def parse_int(value: str) -> int | None:
    text = clean_text(value).lower()
    if not text or text in {"-", "n/a", "\\n"}:
        return None
    multiplier = 1
    if text.endswith("k"):
        multiplier = 1_000
        text = text[:-1]
    elif text.endswith("m"):
        multiplier = 1_000_000
        text = text[:-1]
    elif text.endswith("b"):
        multiplier = 1_000_000_000
        text = text[:-1]
    number = re.sub(r"[^0-9.]", "", text)
    if not number:
        return None
    return int(float(number) * multiplier)


def parse_money(value: str) -> int | None:
    return parse_int(value)


def parse_float(value: str) -> float | None:
    text = clean_text(value)
    if not text or text in {"-", "n/a", "\\N"}:
        return None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    return float(match.group(1)) if match else None


def parse_release_date(value: str, *, default_year: int | None = None) -> str | None:
    text = clean_text(value)
    if not text or text.upper() == "TBD":
        return None
    text = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", text)
    formats = ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y")
    for fmt in formats:
        try:
            return dt.datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    if default_year is not None:
        dated_text = f"{text}, {default_year}"
        for fmt in ("%b %d, %Y", "%B %d, %Y"):
            try:
                return dt.datetime.strptime(dated_text, fmt).date().isoformat()
            except ValueError:
                pass
    return None


def absolute_the_numbers_url(href: str) -> str:
    return urllib.parse.urljoin(THE_NUMBERS_BASE_URL, href)


def absolute_letterboxd_url(href: str) -> str:
    return urllib.parse.urljoin(LETTERBOXD_BASE_URL, href)


def letterboxd_slug_from_url(value: str) -> str | None:
    match = re.search(r"/film/([^/]+)/?", value)
    return match.group(1) if match else None


def parse_the_numbers_release_schedule(
    html: str,
    *,
    source_url: str = THE_NUMBERS_RELEASE_SCHEDULE_URL,
    default_year: int | None = None,
) -> list[TheNumbersReleaseRow]:
    parser = HtmlTableParser()
    parser.feed(html)
    rows: list[TheNumbersReleaseRow] = []
    for table in parser.tables:
        if not table.rows:
            continue
        headers = [normalize_header(cell) for cell in table.rows[0]]
        if "movie" not in headers and "title" not in headers:
            continue
        if not any(header in headers for header in ("release_date", "date")):
            continue
        for cells, hrefs in zip(table.rows[1:], table.hrefs[1:]):
            row = dict(zip(headers, cells))
            movie_idx = first_existing_index(headers, ["movie", "title"])
            movie_href = first_movie_href(hrefs[movie_idx:movie_idx + 1] if movie_idx is not None else hrefs)
            title = row.get("movie") or row.get("title") or ""
            if not movie_href or not title:
                continue
            date_value = row.get("release_date") or row.get("date") or ""
            rows.append(
                TheNumbersReleaseRow(
                    movie_url=absolute_the_numbers_url(movie_href),
                    title=title,
                    release_date=parse_release_date(date_value, default_year=default_year),
                    release_pattern=first_present(row, ["release_pattern", "release_type", "type", "pattern"]),
                    distributor=first_present(row, ["distributor", "distribution"]),
                    domestic_box_office_to_date_usd=parse_money(
                        first_present(row, ["domestic_box_office", "domestic_total", "box_office_to_date"]) or ""
                    ),
                    source_url=source_url,
                )
            )
    return rows


def normalize_header(value: str) -> str:
    text = normalize_title(value)
    replacements = {
        "release date": "release_date",
        "domestic box office": "domestic_box_office",
        "box office to date": "box_office_to_date",
        "release pattern": "release_pattern",
        "release type": "release_type",
    }
    return replacements.get(text, text.replace(" ", "_"))


def first_existing_index(values: list[str], candidates: list[str]) -> int | None:
    for candidate in candidates:
        if candidate in values:
            return values.index(candidate)
    return None


def first_present(row: dict[str, str], keys: list[str]) -> str | None:
    for key in keys:
        value = clean_text(row.get(key, ""))
        if value:
            return value
    return None


def first_movie_href(hrefs: Iterable[str]) -> str | None:
    for href_blob in hrefs:
        for href in href_blob.split():
            if href.startswith("/movie/") or href.startswith(f"{THE_NUMBERS_BASE_URL}/movie/"):
                return href
    return None


def upsert_release_schedule_rows(
    conn: Any,
    rows: list[TheNumbersReleaseRow],
    *,
    fetched_at: str,
    raw_cache_path: Path,
) -> None:
    conn.executemany(
        """
        INSERT INTO the_numbers_release_schedule (
            movie_url, title, release_date, release_pattern, distributor,
            domestic_box_office_to_date_usd, source_url, fetched_at, raw_cache_path
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(movie_url, release_date, release_pattern) DO UPDATE SET
            title = excluded.title,
            distributor = excluded.distributor,
            domestic_box_office_to_date_usd = excluded.domestic_box_office_to_date_usd,
            source_url = excluded.source_url,
            fetched_at = excluded.fetched_at,
            raw_cache_path = excluded.raw_cache_path
        """,
        [
            (
                row.movie_url,
                row.title,
                row.release_date,
                row.release_pattern,
                row.distributor,
                row.domestic_box_office_to_date_usd,
                row.source_url,
                fetched_at,
                str(raw_cache_path),
            )
            for row in rows
        ],
    )


def upsert_movies_from_release_schedule(conn: Any, rows: list[TheNumbersReleaseRow]) -> None:
    for row in rows:
        release_year = int(row.release_date[:4]) if row.release_date else parse_year(row.title)
        conn.execute(
            """
            INSERT INTO movies (movie_url, title, release_year, release_date, updated_at)
            VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT(movie_url) WHERE movie_url IS NOT NULL DO UPDATE SET
                title = excluded.title,
                release_year = COALESCE(excluded.release_year, movies.release_year),
                release_date = COALESCE(excluded.release_date, movies.release_date),
                updated_at = CURRENT_TIMESTAMP
            """,
            (row.movie_url, row.title, release_year, row.release_date),
        )


def parse_release_year_from_movie_url(movie_url: str | None) -> int | None:
    if not movie_url:
        return None
    match = re.search(r"\((\d{4})(?:-[^)]+)?\)", urllib.parse.unquote(movie_url))
    return int(match.group(1)) if match else None


def upsert_movies_from_current_chart_pages(
    conn: Any,
    *,
    snapshot_date: dt.date,
    active_days: int,
) -> int:
    if not relation_exists(conn, "daily_chart_pages"):
        return 0
    rows = conn.execute(
        """
        SELECT DISTINCT ON (movie_url)
            movie_url,
            title,
            chart_date,
            days_in_release
        FROM daily_chart_pages
        WHERE chart_date::date BETWEEN %s AND %s
          AND COALESCE(days_in_release, 0) <= 365
        ORDER BY movie_url, chart_date::date DESC
        """,
        (snapshot_date - dt.timedelta(days=active_days), snapshot_date),
    ).fetchall()
    for movie_url, title, _chart_date, _days_in_release in rows:
        conn.execute(
            """
            INSERT INTO movies (movie_url, title, release_year, updated_at)
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT(movie_url) WHERE movie_url IS NOT NULL DO UPDATE SET
                title = COALESCE(movies.title, excluded.title),
                release_year = COALESCE(movies.release_year, excluded.release_year),
                updated_at = CURRENT_TIMESTAMP
            """,
            (movie_url, title, parse_release_year_from_movie_url(movie_url)),
        )
    return len(rows)


def relation_exists(conn: Any, relation_name: str) -> bool:
    row = conn.execute("SELECT to_regclass(%s)", (relation_name,)).fetchone()
    return bool(row and row[0])


def select_candidate_movies(
    conn: Any,
    *,
    snapshot_date: dt.date,
    lookahead_days: int,
    active_days: int,
    movie_limit: int | None,
) -> list[CandidateMovie]:
    selects: list[str] = []
    params: list[Any] = []
    if relation_exists(conn, "the_numbers_release_schedule"):
        selects.append(
            """
            SELECT
                m.movie_id,
                m.movie_url,
                m.title,
                m.release_year,
                COALESCE(m.release_date, tn.release_date)::text AS release_date
            FROM the_numbers_release_schedule tn
            JOIN movies m ON m.movie_url = tn.movie_url
            WHERE tn.release_date BETWEEN %s AND %s
              AND tn.title NOT ILIKE '%%untitled%%'
              AND tn.title NOT ILIKE '%%re-release%%'
              AND COALESCE(tn.release_pattern, '') NOT ILIKE '%%re-release%%'
            """
        )
        params.extend([snapshot_date, snapshot_date + dt.timedelta(days=lookahead_days)])
    if relation_exists(conn, "daily_chart_pages"):
        selects.append(
            """
            SELECT DISTINCT
                m.movie_id,
                m.movie_url,
                m.title,
                m.release_year,
                m.release_date::text AS release_date
            FROM daily_chart_pages dcp
            JOIN movies m ON m.movie_url = dcp.movie_url
            WHERE dcp.chart_date::date BETWEEN %s AND %s
              AND COALESCE(dcp.days_in_release, 0) <= 365
              AND dcp.title NOT ILIKE '%%re-release%%'
              AND m.title NOT ILIKE '%%re-release%%'
            """
        )
        params.extend([snapshot_date - dt.timedelta(days=active_days), snapshot_date])
    if not selects:
        return []
    sql = " UNION ".join(selects) + " ORDER BY release_date NULLS LAST, title"
    if movie_limit is not None:
        sql += " LIMIT %s"
        params.append(movie_limit)
    rows = conn.execute(sql, params).fetchall()
    return [
        CandidateMovie(
            movie_id=int(row[0]),
            movie_url=row[1],
            title=row[2],
            release_year=row[3],
            release_date=row[4],
            normalized_title=normalize_title(row[2]),
        )
        for row in rows
    ]


def iter_tsv_rows(data: bytes) -> Iterable[dict[str, str]]:
    if data.startswith(b"\x1f\x8b"):
        with gzip.open(io.BytesIO(data), "rt", encoding="utf-8", errors="replace", newline="") as handle:
            yield from csv.DictReader(handle, delimiter="\t")
        return
    text = data.decode("utf-8", errors="replace")
    yield from csv.DictReader(io.StringIO(text), delimiter="\t")


def parse_tsv_rows(data: bytes) -> list[dict[str, str]]:
    return list(iter_tsv_rows(data))


def iter_tsv_fields(data: bytes) -> Iterable[list[str]]:
    if data.startswith(b"\x1f\x8b"):
        with gzip.open(io.BytesIO(data), "rt", encoding="utf-8", errors="replace", newline="") as handle:
            next(handle, None)
            for line in handle:
                yield line.rstrip("\n").split("\t")
        return
    handle = io.StringIO(data.decode("utf-8", errors="replace"))
    next(handle, None)
    for line in handle:
        yield line.rstrip("\n").split("\t")


def none_if_missing(value: str | None) -> str | None:
    if value is None or value == "\\N" or value == "":
        return None
    return value


def int_or_none(value: str | None) -> int | None:
    value = none_if_missing(value)
    return int(value) if value is not None and re.fullmatch(r"-?\d+", value) else None


def imdb_title_from_row(row: dict[str, str]) -> ImdbTitle:
    return ImdbTitle(
        tconst=row["tconst"],
        title_type=row.get("titleType") or "",
        primary_title=row.get("primaryTitle") or "",
        original_title=none_if_missing(row.get("originalTitle")),
        is_adult=int_or_none(row.get("isAdult")) or 0,
        start_year=int_or_none(row.get("startYear")),
        genres=none_if_missing(row.get("genres")),
    )


def imdb_aka_from_row(row: dict[str, str]) -> ImdbAka:
    return ImdbAka(
        title_id=row["titleId"],
        title=row.get("title") or "",
        region=none_if_missing(row.get("region")),
        language=none_if_missing(row.get("language")),
        types=none_if_missing(row.get("types")),
        is_original_title=int_or_none(row.get("isOriginalTitle")) or 0,
    )


def imdb_rating_from_row(row: dict[str, str]) -> ImdbRating:
    return ImdbRating(
        tconst=row["tconst"],
        average_rating=parse_float(row.get("averageRating") or ""),
        num_votes=int_or_none(row.get("numVotes")),
    )


def parse_imdb_titles(data: bytes) -> list[ImdbTitle]:
    return [imdb_title_from_row(row) for row in iter_tsv_rows(data)]


def parse_imdb_titles_filtered(
    data: bytes,
    *,
    wanted_titles: set[str],
    wanted_tconsts: set[str],
) -> list[ImdbTitle]:
    rows = []
    for fields in iter_tsv_fields(data):
        if len(fields) < 9:
            continue
        tconst = fields[0]
        primary_title = fields[2]
        original_title = none_if_missing(fields[3])
        if (
            tconst in wanted_tconsts
            or normalize_title(primary_title) in wanted_titles
            or (original_title and normalize_title(original_title) in wanted_titles)
        ):
            rows.append(
                ImdbTitle(
                    tconst=tconst,
                    title_type=fields[1],
                    primary_title=primary_title,
                    original_title=original_title,
                    is_adult=int_or_none(fields[4]) or 0,
                    start_year=int_or_none(fields[5]),
                    genres=none_if_missing(fields[8]),
                )
            )
    return rows


def parse_imdb_akas(data: bytes) -> list[ImdbAka]:
    return [imdb_aka_from_row(row) for row in iter_tsv_rows(data)]


def parse_imdb_akas_filtered(
    data: bytes,
    *,
    wanted_titles: set[str],
    wanted_tconsts: set[str],
    search_titles: bool = False,
) -> list[ImdbAka]:
    rows = []
    if not wanted_tconsts and not search_titles:
        return rows
    for fields in iter_tsv_fields(data):
        if len(fields) < 8:
            continue
        title_id = fields[0]
        title = fields[2]
        if title_id in wanted_tconsts or (search_titles and normalize_title(title) in wanted_titles):
            rows.append(
                ImdbAka(
                    title_id=title_id,
                    title=title,
                    region=none_if_missing(fields[3]),
                    language=none_if_missing(fields[4]),
                    types=none_if_missing(fields[5]),
                    is_original_title=int_or_none(fields[7]) or 0,
                )
            )
    return rows


def parse_imdb_ratings(data: bytes) -> list[ImdbRating]:
    return [imdb_rating_from_row(row) for row in iter_tsv_rows(data)]


def parse_imdb_ratings_filtered(data: bytes, *, wanted_tconsts: set[str]) -> dict[str, ImdbRating]:
    ratings: dict[str, ImdbRating] = {}
    if not wanted_tconsts:
        return ratings
    for fields in iter_tsv_fields(data):
        if len(fields) < 3:
            continue
        tconst = fields[0]
        if tconst in wanted_tconsts:
            ratings[tconst] = ImdbRating(
                tconst=tconst,
                average_rating=parse_float(fields[1]),
                num_votes=int_or_none(fields[2]),
            )
    return ratings


def commit_if_possible(conn: Any) -> None:
    commit = getattr(conn, "commit", None)
    if commit is not None:
        commit()


def rollback_if_possible(conn: Any) -> None:
    rollback = getattr(conn, "rollback", None)
    if rollback is not None:
        rollback()


def upsert_imdb_titles(conn: Any, titles: Iterable[ImdbTitle], *, last_seen_at: str) -> None:
    conn.executemany(
        """
        INSERT INTO imdb_titles (
            tconst, primary_title, original_title, start_year, title_type,
            is_adult, genres, last_seen_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(tconst) DO UPDATE SET
            primary_title = excluded.primary_title,
            original_title = excluded.original_title,
            start_year = excluded.start_year,
            title_type = excluded.title_type,
            is_adult = excluded.is_adult,
            genres = excluded.genres,
            last_seen_at = excluded.last_seen_at
        """,
        [
            (
                title.tconst,
                title.primary_title,
                title.original_title,
                title.start_year,
                title.title_type,
                title.is_adult,
                title.genres,
                last_seen_at,
            )
            for title in titles
        ],
    )


def match_imdb_title(
    movie: CandidateMovie,
    titles: list[ImdbTitle],
    akas: list[ImdbAka],
) -> ImdbMatch:
    title_names_by_tconst: dict[str, set[str]] = {}
    title_by_tconst = {title.tconst: title for title in titles}
    for title in titles:
        if title.title_type != "movie" or title.is_adult:
            continue
        title_names_by_tconst.setdefault(title.tconst, set()).add(normalize_title(title.primary_title))
        if title.original_title:
            title_names_by_tconst.setdefault(title.tconst, set()).add(normalize_title(title.original_title))
    for aka in akas:
        if aka.title_id in title_by_tconst and (aka.region in {None, "US", "CA", "XWW"}):
            title_names_by_tconst.setdefault(aka.title_id, set()).add(normalize_title(aka.title))

    candidates: list[tuple[ImdbTitle, float, str]] = []
    movie_title_names = {normalize_title(value) for value in title_variants(movie.title, movie.movie_url)}
    for tconst, names in title_names_by_tconst.items():
        title = title_by_tconst[tconst]
        if not movie_title_names.intersection(names):
            continue
        if movie.release_year is not None and title.start_year is not None:
            if abs(title.start_year - movie.release_year) > 1:
                continue
            year_score = 1.0 if title.start_year == movie.release_year else 0.75
        else:
            year_score = 0.5
        method = "title_year_exact" if year_score == 1.0 else "title_year_near_or_missing"
        candidates.append((title, year_score, method))

    if len(candidates) == 1:
        title, score, method = candidates[0]
        return ImdbMatch(movie.movie_id, title.tconst, "matched", method, score)
    if len(candidates) > 1:
        notes = ", ".join(sorted(title.tconst for title, _score, _method in candidates))
        return ImdbMatch(movie.movie_id, None, "ambiguous", "title_year", None, notes)
    return ImdbMatch(movie.movie_id, None, "not_found", "title_year", None, "No IMDb title matched title/year")


def upsert_imdb_match(conn: Any, match: ImdbMatch) -> None:
    conn.execute(
        """
        INSERT INTO movie_imdb_titles (
            movie_id, tconst, match_status, match_method, match_score, matched_at, notes
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(movie_id) DO UPDATE SET
            tconst = excluded.tconst,
            match_status = excluded.match_status,
            match_method = excluded.match_method,
            match_score = excluded.match_score,
            matched_at = excluded.matched_at,
            notes = excluded.notes
        """,
        (
            match.movie_id,
            match.tconst,
            match.match_status,
            match.match_method,
            match.match_score,
            utc_now(),
            match.notes,
        ),
    )


def insert_imdb_snapshot(
    conn: Any,
    *,
    rating: ImdbRating,
    snapshot_date: dt.date,
    fetched_at: str,
    raw_cache_path: Path,
) -> None:
    conn.execute(
        """
        INSERT INTO imdb_title_snapshots (
            tconst, snapshot_date, observed_at, average_rating, num_votes,
            fetched_at, raw_cache_path
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(tconst, snapshot_date) DO UPDATE SET
            observed_at = excluded.observed_at,
            average_rating = excluded.average_rating,
            num_votes = excluded.num_votes,
            fetched_at = excluded.fetched_at,
            raw_cache_path = excluded.raw_cache_path
        """,
        (
            rating.tconst,
            snapshot_date,
            dt.datetime.now(dt.UTC),
            rating.average_rating,
            rating.num_votes,
            fetched_at,
            str(raw_cache_path),
        ),
    )


def sparql_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"@en'


def parse_wikidata_year(value: str | None) -> int | None:
    if not value:
        return None
    match = re.match(r"(\d{4})", value)
    return int(match.group(1)) if match else None


def wikidata_sparql_query(candidates: list[CandidateMovie]) -> str:
    labels: list[str] = []
    seen: set[str] = set()
    for movie in candidates:
        for variant in title_variants(movie.title, movie.movie_url):
            key = normalize_title(variant)
            if key and key not in seen:
                labels.append(variant)
                seen.add(key)
    values = " ".join(sparql_string(label) for label in labels)
    return f"""
        SELECT ?item ?itemLabel ?imdb ?tmdb ?letterboxd ?pubdate WHERE {{
          VALUES ?wantedLabel {{ {values} }}
          ?item rdfs:label ?wantedLabel.
          ?item wdt:P31/wdt:P279* wd:Q11424.
          OPTIONAL {{ ?item wdt:P345 ?imdb. }}
          OPTIONAL {{ ?item wdt:P4947 ?tmdb. }}
          OPTIONAL {{ ?item wdt:P6127 ?letterboxd. }}
          OPTIONAL {{ ?item wdt:P577 ?pubdate. }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT 500
    """


def wikidata_sparql_url(candidates: list[CandidateMovie]) -> str:
    query = wikidata_sparql_query(candidates)
    return f"{WIKIDATA_SPARQL_URL}?{urllib.parse.urlencode({'query': query, 'format': 'json'})}"


def parse_wikidata_bindings(data: dict[str, Any]) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for binding in data.get("results", {}).get("bindings", []):
        item = binding.get("item", {}).get("value", "")
        qid = item.rstrip("/").split("/")[-1]
        if not qid:
            continue
        record = records.setdefault(
            qid,
            {
                "qid": qid,
                "label": binding.get("itemLabel", {}).get("value", ""),
                "years": set(),
                "imdb": None,
                "tmdb": None,
                "letterboxd": None,
            },
        )
        year = parse_wikidata_year(binding.get("pubdate", {}).get("value"))
        if year is not None:
            record["years"].add(year)
        record["imdb"] = record["imdb"] or binding.get("imdb", {}).get("value")
        record["tmdb"] = record["tmdb"] or binding.get("tmdb", {}).get("value")
        record["letterboxd"] = record["letterboxd"] or binding.get("letterboxd", {}).get("value")
    return list(records.values())


def merge_wikidata_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for record in records:
        qid = str(record.get("qid") or "")
        if not qid:
            continue
        current = merged.setdefault(
            qid,
            {
                "qid": qid,
                "label": record.get("label") or "",
                "years": set(),
                "imdb": None,
                "tmdb": None,
                "letterboxd": None,
            },
        )
        current["years"].update(record.get("years") or [])
        current["imdb"] = current["imdb"] or record.get("imdb")
        current["tmdb"] = current["tmdb"] or record.get("tmdb")
        current["letterboxd"] = current["letterboxd"] or record.get("letterboxd")
        if not current["label"] and record.get("label"):
            current["label"] = record.get("label")
    return list(merged.values())


def wikidata_record_score(movie: CandidateMovie, record: dict[str, Any]) -> tuple[float, int | None]:
    label = str(record.get("label") or "")
    title_score = max(
        (
            title_similarity(variant, label)
            for variant in title_variants(movie.title, movie.movie_url)
        ),
        default=0.0,
    )
    years = sorted(record.get("years") or [])
    best_year = None
    year_score = 0.0
    if movie.release_year is not None and years:
        best_year = min(years, key=lambda year: abs(year - movie.release_year))
        diff = abs(best_year - movie.release_year)
        if diff == 0:
            year_score = 30.0
        elif diff == 1:
            year_score = 20.0
        elif diff <= 2:
            year_score = 10.0
        else:
            year_score = -30.0
    elif movie.release_year is None or not years:
        year_score = 5.0
    id_score = 10.0 if any(record.get(key) for key in ("imdb", "tmdb", "letterboxd")) else 0.0
    return title_score + year_score + id_score, best_year


def title_similarity(left: str, right: str) -> float:
    import difflib

    return difflib.SequenceMatcher(None, normalize_title(left), normalize_title(right)).ratio() * 100.0


def match_wikidata_movies(
    candidates: list[CandidateMovie],
    records: list[dict[str, Any]],
) -> list[WikidataMovieMatch]:
    matches: list[WikidataMovieMatch] = []
    for movie in candidates:
        scored: list[tuple[float, int | None, dict[str, Any]]] = []
        for record in records:
            score, matched_year = wikidata_record_score(movie, record)
            if score >= 100:
                scored.append((score, matched_year, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored:
            continue
        best_score, best_year, best_record = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else None
        status = "matched"
        notes = None
        if best_score < 130:
            status = "ambiguous"
            notes = "Best Wikidata score below high-confidence threshold"
        elif second_score is not None and best_score - second_score < 8:
            status = "ambiguous"
            notes = f"Competing Wikidata candidate within score margin: {second_score:.1f}"
        matches.append(
            WikidataMovieMatch(
                movie_id=movie.movie_id,
                qid=str(best_record["qid"]),
                label=str(best_record.get("label") or ""),
                release_year=best_year,
                imdb_tconst=best_record.get("imdb"),
                tmdb_id=best_record.get("tmdb"),
                letterboxd_slug=best_record.get("letterboxd"),
                score=best_score,
                status=status,
                notes=notes,
            )
        )
    return matches


def existing_match_status(conn: Any, *, table: str, movie_id: int) -> str | None:
    if table not in {"movie_imdb_titles", "movie_letterboxd_films"}:
        raise ValueError(f"Unsupported match table: {table}")
    row = conn.execute(
        f"SELECT match_status FROM {table} WHERE movie_id = %s",
        (movie_id,),
    ).fetchone()
    return row[0] if row else None


def upsert_wikidata_matches(
    conn: Any,
    matches: list[WikidataMovieMatch],
) -> int:
    accepted = 0
    now = utc_now()
    for match in matches:
        if match.status != "matched":
            continue
        seeded = False
        if match.imdb_tconst and existing_match_status(
            conn,
            table="movie_imdb_titles",
            movie_id=match.movie_id,
        ) != "manual_override":
            upsert_imdb_titles(
                conn,
                [
                    ImdbTitle(
                        tconst=match.imdb_tconst,
                        primary_title=match.label,
                        original_title=match.label,
                        start_year=match.release_year,
                        title_type="movie",
                        is_adult=0,
                        genres=None,
                    )
                ],
                last_seen_at=now,
            )
            upsert_imdb_match(
                conn,
                ImdbMatch(
                    movie_id=match.movie_id,
                    tconst=match.imdb_tconst,
                    match_status="matched",
                    match_method="wikidata_sparql",
                    match_score=match.score,
                    notes=f"qid={match.qid}; tmdb={match.tmdb_id}; letterboxd={match.letterboxd_slug}",
                ),
            )
            seeded = True
        if match.letterboxd_slug and existing_match_status(
            conn,
            table="movie_letterboxd_films",
            movie_id=match.movie_id,
        ) != "manual_override":
            page = LetterboxdFilmPage(
                letterboxd_slug=match.letterboxd_slug,
                film_url=f"{LETTERBOXD_BASE_URL}/film/{match.letterboxd_slug}/",
                source_title=match.label,
                source_year=match.release_year,
                imdb_tconst=match.imdb_tconst,
                tmdb_id=match.tmdb_id,
                average_rating=None,
                watched_count=None,
                rating_count=None,
                review_count=None,
                log_count=None,
                fan_count=None,
                parse_status="wikidata_seed",
            )
            upsert_letterboxd_film(conn, page, last_seen_at=now)
            upsert_letterboxd_match(
                conn,
                LetterboxdMatch(
                    movie_id=match.movie_id,
                    letterboxd_slug=match.letterboxd_slug,
                    match_status="matched",
                    match_method="wikidata_sparql",
                    match_score=match.score,
                    notes=f"qid={match.qid}; imdb={match.imdb_tconst}; tmdb={match.tmdb_id}",
                ),
            )
            seeded = True
        if seeded:
            accepted += 1
    return accepted


def ingest_wikidata_matches(
    args: argparse.Namespace,
    conn: Any,
    candidates: list[CandidateMovie],
    fetcher: CachedFetcher,
) -> int:
    if not candidates:
        return 0
    log(args, f"Starting Wikidata batch match for {len(candidates)} candidate movies")
    records: list[dict[str, Any]] = []
    for start in range(0, len(candidates), WIKIDATA_BATCH_SIZE):
        batch = candidates[start:start + WIKIDATA_BATCH_SIZE]
        try:
            html, _cache_path, _fetched = fetcher.get_text(wikidata_sparql_url(batch))
            data = json.loads(html)
        except Exception as exc:
            log(args, f"Wikidata batch match failed for batch starting at {start + 1}; continuing: {exc}")
            continue
        records.extend(parse_wikidata_bindings(data))
    if not records:
        log(args, "Wikidata returned no usable records")
        return 0
    records = merge_wikidata_records(records)
    matches = match_wikidata_movies(candidates, records)
    accepted = upsert_wikidata_matches(conn, matches)
    ambiguous = sum(1 for match in matches if match.status == "ambiguous")
    log(args, f"Wikidata matched {accepted} high-confidence movies; {ambiguous} ambiguous candidates kept out")
    return accepted


def letterboxd_search_url(title: str, year: int | None) -> str:
    query = f"{title} {year}" if year else title
    return f"{LETTERBOXD_BASE_URL}/search/films/{urllib.parse.quote(query)}/"


def letterboxd_slugify_title(value: str) -> str | None:
    normalized = normalize_title(strip_release_qualifier(strip_title_year(value)))
    return normalized.replace(" ", "-") if normalized else None


def letterboxd_candidate_slugs(movie: CandidateMovie) -> list[str]:
    slugs: list[str] = []
    seen: set[str] = set()
    for variant in title_variants(movie.title, movie.movie_url):
        slug = letterboxd_slugify_title(variant)
        if not slug:
            continue
        for candidate_slug in (f"{slug}-{movie.release_year}" if movie.release_year else None, slug):
            if candidate_slug and candidate_slug not in seen:
                slugs.append(candidate_slug)
                seen.add(candidate_slug)
    return slugs


def parse_letterboxd_search(html: str) -> list[LetterboxdSearchResult]:
    parser = LetterboxdSearchParser()
    parser.feed(html)
    return parser.results


def parse_json_ld_blocks(html: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for match in re.finditer(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html,
        re.IGNORECASE | re.DOTALL,
    ):
        raw = match.group(1).strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            blocks.append(data)
        elif isinstance(data, list):
            blocks.extend(item for item in data if isinstance(item, dict))
    return blocks


def parse_letterboxd_film_page(html: str, *, film_url: str) -> LetterboxdFilmPage:
    slug = letterboxd_slug_from_url(film_url) or ""
    title = parse_meta_content(html, "og:title") or parse_title_tag(html)
    year = parse_year(title or "")
    average_rating = parse_float(parse_meta_content(html, "twitter:data2") or "")
    imdb_tconst = parse_imdb_tconst(html)
    tmdb_id = parse_tmdb_id(html)
    watched = count_near_label(html, ["watched", "watched by"])
    ratings = count_near_label(html, ["ratings", "rating"])
    reviews = count_near_label(html, ["reviews", "review"])
    logs = count_near_label(html, ["diary", "logs", "logged"])
    fans = count_near_label(html, ["fans", "fan"])

    for block in parse_json_ld_blocks(html):
        if not title and isinstance(block.get("name"), str):
            title = block["name"]
        if average_rating is None:
            aggregate = block.get("aggregateRating")
            if isinstance(aggregate, dict):
                average_rating = parse_float(str(aggregate.get("ratingValue", "")))
                ratings = ratings or parse_int(str(aggregate.get("ratingCount", "")))
                reviews = reviews or parse_int(str(aggregate.get("reviewCount", "")))

    parse_status = "parsed" if any(
        value is not None for value in (watched, ratings, reviews, logs, fans, average_rating)
    ) else "missing_or_changed_markup"
    return LetterboxdFilmPage(
        letterboxd_slug=slug,
        film_url=film_url,
        source_title=clean_letterboxd_title(title),
        source_year=year,
        imdb_tconst=imdb_tconst,
        tmdb_id=tmdb_id,
        average_rating=average_rating,
        watched_count=watched,
        rating_count=ratings,
        review_count=reviews,
        log_count=logs,
        fan_count=fans,
        parse_status=parse_status,
    )


def parse_meta_content(html: str, property_name: str) -> str | None:
    pattern = (
        r'<meta[^>]+(?:property|name)=["\']'
        + re.escape(property_name)
        + r'["\'][^>]+content=["\']([^"\']+)["\']'
    )
    match = re.search(pattern, html, re.IGNORECASE)
    if match:
        return clean_text(html_unescape(match.group(1)))
    pattern = (
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\']'
        + re.escape(property_name)
        + r'["\']'
    )
    match = re.search(pattern, html, re.IGNORECASE)
    return clean_text(html_unescape(match.group(1))) if match else None


def parse_title_tag(html: str) -> str | None:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    return clean_text(html_unescape(re.sub(r"<[^>]+>", " ", match.group(1)))) if match else None


def html_unescape(value: str) -> str:
    import html as html_module

    return html_module.unescape(value)


def clean_letterboxd_title(value: str | None) -> str | None:
    if not value:
        return None
    text = re.sub(r"\s*\(\d{4}\)\s*$", "", value)
    text = re.sub(r"\s*-\s*Letterboxd.*$", "", text, flags=re.IGNORECASE)
    return clean_text(text)


def parse_imdb_tconst(html: str) -> str | None:
    match = re.search(r"https?://(?:www\.)?imdb\.com/title/(tt\d+)", html, re.IGNORECASE)
    return match.group(1) if match else None


def parse_tmdb_id(html: str) -> str | None:
    match = re.search(r"https?://(?:www\.)?themoviedb\.org/movie/(\d+)", html, re.IGNORECASE)
    return match.group(1) if match else None


def count_near_label(html: str, labels: list[str]) -> int | None:
    text = clean_text(re.sub(r"<[^>]+>", " ", html_unescape(html))).lower()
    for label in labels:
        label_pattern = re.escape(label.lower())
        patterns = [
            rf"([0-9][0-9,\.]*\s*[kmb]?)\s+{label_pattern}\b",
            rf"{label_pattern}\b\s+([0-9][0-9,\.]*\s*[kmb]?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return parse_int(match.group(1))
    return None


def match_letterboxd_page(
    *,
    movie: CandidateMovie,
    page: LetterboxdFilmPage,
    imdb_tconst: str | None,
) -> LetterboxdMatch:
    if imdb_tconst and page.imdb_tconst and imdb_tconst == page.imdb_tconst:
        return LetterboxdMatch(movie.movie_id, page.letterboxd_slug, "matched", "imdb_tconst", 1.0)
    if page.source_title and normalize_title(page.source_title) == movie.normalized_title:
        if movie.release_year is None or page.source_year is None or abs(page.source_year - movie.release_year) <= 1:
            score = 1.0 if page.source_year == movie.release_year else 0.75
            return LetterboxdMatch(movie.movie_id, page.letterboxd_slug, "matched", "title_year", score)
    return LetterboxdMatch(
        movie.movie_id,
        page.letterboxd_slug,
        "not_found",
        "candidate_rejected",
        None,
        f"Rejected {page.source_title or page.film_url}",
    )


def upsert_letterboxd_film(conn: Any, page: LetterboxdFilmPage, *, last_seen_at: str) -> None:
    conn.execute(
        """
        INSERT INTO letterboxd_films (
            letterboxd_slug, film_url, source_title, source_year, imdb_tconst, tmdb_id, last_seen_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(letterboxd_slug) DO UPDATE SET
            film_url = excluded.film_url,
            source_title = excluded.source_title,
            source_year = excluded.source_year,
            imdb_tconst = COALESCE(excluded.imdb_tconst, letterboxd_films.imdb_tconst),
            tmdb_id = COALESCE(excluded.tmdb_id, letterboxd_films.tmdb_id),
            last_seen_at = excluded.last_seen_at
        """,
        (
            page.letterboxd_slug,
            page.film_url,
            page.source_title,
            page.source_year,
            page.imdb_tconst,
            page.tmdb_id,
            last_seen_at,
        ),
    )


def upsert_letterboxd_match(conn: Any, match: LetterboxdMatch) -> None:
    conn.execute(
        """
        INSERT INTO movie_letterboxd_films (
            movie_id, letterboxd_slug, match_status, match_method, match_score, matched_at, notes
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(movie_id) DO UPDATE SET
            letterboxd_slug = excluded.letterboxd_slug,
            match_status = excluded.match_status,
            match_method = excluded.match_method,
            match_score = excluded.match_score,
            matched_at = excluded.matched_at,
            notes = excluded.notes
        """,
        (
            match.movie_id,
            match.letterboxd_slug,
            match.match_status,
            match.match_method,
            match.match_score,
            utc_now(),
            match.notes,
        ),
    )


def insert_letterboxd_snapshot(
    conn: Any,
    *,
    page: LetterboxdFilmPage,
    snapshot_date: dt.date,
    fetched_at: str,
    raw_cache_path: Path,
) -> None:
    conn.execute(
        """
        INSERT INTO letterboxd_film_snapshots (
            letterboxd_slug, snapshot_date, observed_at, fan_count, average_rating, parse_status,
            source_url, fetched_at, raw_cache_path, parser_version
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(letterboxd_slug, snapshot_date) DO UPDATE SET
            observed_at = excluded.observed_at,
            fan_count = excluded.fan_count,
            average_rating = excluded.average_rating,
            parse_status = excluded.parse_status,
            source_url = excluded.source_url,
            fetched_at = excluded.fetched_at,
            raw_cache_path = excluded.raw_cache_path,
            parser_version = excluded.parser_version
        """,
        (
            page.letterboxd_slug,
            snapshot_date,
            dt.datetime.now(dt.UTC),
            page.fan_count,
            page.average_rating,
            page.parse_status,
            page.film_url,
            fetched_at,
            str(raw_cache_path),
            PARSER_VERSION,
        ),
    )


def upsert_state(
    conn: Any,
    *,
    movie_id: int,
    source: str,
    stage: str,
    status: str,
    last_error: str | None = None,
) -> None:
    now = utc_now()
    conn.execute(
        """
        INSERT INTO audience_ingest_state (
            movie_id, source, stage, status, started_at, updated_at,
            completed_at, attempt_count, last_error
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, 1, %s)
        ON CONFLICT(movie_id, source) DO UPDATE SET
            stage = excluded.stage,
            status = excluded.status,
            updated_at = excluded.updated_at,
            completed_at = excluded.completed_at,
            attempt_count = audience_ingest_state.attempt_count + CASE
                WHEN excluded.status = 'running' AND audience_ingest_state.status != 'running'
                THEN 1 ELSE 0 END,
            last_error = excluded.last_error
        """,
        (
            movie_id,
            source,
            stage,
            status,
            now,
            now,
            now if status == "completed" else None,
            last_error,
        ),
    )


def reset_failed_states(conn: Any) -> None:
    conn.execute("DELETE FROM audience_ingest_state WHERE status = 'failed'")


def dataset_url(name: str) -> str:
    return f"{IMDB_DATASET_BASE_URL}/{name}.tsv.gz"


def log(args: argparse.Namespace, message: str) -> None:
    if not getattr(args, "quiet", False):
        print(message, file=sys.stderr, flush=True)


def load_imdb_datasets(
    fetcher: CachedFetcher,
    *,
    wanted_titles: set[str] | None = None,
    seeded_tconsts: set[str] | None = None,
    args: argparse.Namespace | None = None,
) -> tuple[list[ImdbTitle], list[ImdbAka], dict[str, ImdbRating], Path]:
    wanted_titles = wanted_titles or set()
    seeded_tconsts = seeded_tconsts or set()
    if args is not None:
        log(args, "Reading IMDb title.basics.tsv.gz")
    basics_body, basics_path, _ = fetcher.get_bytes(dataset_url("title.basics"), suffix=".tsv.gz")
    if args is not None:
        log(args, f"Parsing IMDb title.basics.tsv.gz from {basics_path} for candidate titles")
    title_pool = parse_imdb_titles_filtered(
        basics_body,
        wanted_titles=wanted_titles,
        wanted_tconsts=seeded_tconsts,
    )
    title_by_tconst = {title.tconst: title for title in title_pool}
    tconst_pool = set(title_by_tconst) | seeded_tconsts

    if args is not None:
        log(args, "Reading IMDb title.akas.tsv.gz")
    akas_body, _akas_path, _ = fetcher.get_bytes(dataset_url("title.akas"), suffix=".tsv.gz")
    if args is not None:
        log(args, "Parsing IMDb title.akas.tsv.gz for candidate title aliases")
    aka_pool = parse_imdb_akas_filtered(
        akas_body,
        wanted_titles=wanted_titles,
        wanted_tconsts=tconst_pool,
        search_titles=bool(getattr(args, "imdb_search_aka_titles", False)),
    )
    tconst_pool.update(aka.title_id for aka in aka_pool)
    missing_tconsts = tconst_pool - set(title_by_tconst)
    if missing_tconsts:
        for title in parse_imdb_titles_filtered(
            basics_body,
            wanted_titles=set(),
            wanted_tconsts=missing_tconsts,
        ):
            title_by_tconst[title.tconst] = title

    if args is not None:
        log(args, "Reading IMDb title.ratings.tsv.gz")
    ratings_body, ratings_path, _ = fetcher.get_bytes(dataset_url("title.ratings"), suffix=".tsv.gz")
    if args is not None:
        log(args, f"Parsing IMDb title.ratings.tsv.gz for {len(tconst_pool)} matched title IDs")
    return list(title_by_tconst.values()), aka_pool, parse_imdb_ratings_filtered(
        ratings_body,
        wanted_tconsts=tconst_pool,
    ), ratings_path


def load_imdb_ratings_for_tconsts(
    fetcher: CachedFetcher,
    *,
    wanted_tconsts: set[str],
    args: argparse.Namespace | None = None,
) -> tuple[dict[str, ImdbRating], Path]:
    if args is not None:
        log(args, f"Reading IMDb title.ratings.tsv.gz for {len(wanted_tconsts)} existing IMDb IDs")
    ratings_body, ratings_path, _ = fetcher.get_bytes(dataset_url("title.ratings"), suffix=".tsv.gz")
    if args is not None:
        log(args, "Parsing IMDb title.ratings.tsv.gz for existing IMDb IDs")
    return parse_imdb_ratings_filtered(ratings_body, wanted_tconsts=wanted_tconsts), ratings_path


def ingest_release_schedule(args: argparse.Namespace, conn: Any, fetcher: CachedFetcher) -> int:
    log(args, f"Reading The Numbers release schedule: {THE_NUMBERS_RELEASE_SCHEDULE_URL}")
    html, cache_path, _ = fetcher.get_text(THE_NUMBERS_RELEASE_SCHEDULE_URL)
    rows = parse_the_numbers_release_schedule(
        html,
        source_url=THE_NUMBERS_RELEASE_SCHEDULE_URL,
        default_year=args.snapshot_date.year,
    )
    fetched_at = utc_now()
    upsert_release_schedule_rows(conn, rows, fetched_at=fetched_at, raw_cache_path=cache_path)
    upsert_movies_from_release_schedule(conn, rows)
    log(args, f"Stored {len(rows)} The Numbers release schedule rows")
    return len(rows)


def ingest_imdb(args: argparse.Namespace, conn: Any, candidates: list[CandidateMovie], fetcher: CachedFetcher) -> int:
    if not candidates:
        log(args, "Skipping IMDb ingest: no candidate movies")
        return 0
    log(args, f"Starting IMDb ingest for {len(candidates)} candidate movies")
    existing_tconst_by_movie = {
        movie.movie_id: tconst
        for movie in candidates
        if (tconst := load_movie_imdb_tconst(conn, movie.movie_id))
    }
    inserted = 0
    processed_movie_ids: set[int] = set()
    if existing_tconst_by_movie:
        seeded_ratings, seeded_ratings_path = load_imdb_ratings_for_tconsts(
            fetcher,
            wanted_tconsts=set(existing_tconst_by_movie.values()),
            args=args,
        )
        log(
            args,
            f"Loaded {len(seeded_ratings)} IMDb ratings rows for {len(existing_tconst_by_movie)} existing IMDb matches",
        )
        for index, movie in enumerate(candidates, start=1):
            existing_tconst = existing_tconst_by_movie.get(movie.movie_id)
            if not existing_tconst:
                continue
            try:
                log(args, f"IMDb {index}/{len(candidates)}: using existing match {movie.title} -> {existing_tconst}")
                upsert_state(conn, movie_id=movie.movie_id, source="imdb", stage="snapshot", status="running")
                rating = seeded_ratings.get(existing_tconst)
                if rating:
                    insert_imdb_snapshot(
                        conn,
                        rating=rating,
                        snapshot_date=args.snapshot_date,
                        fetched_at=utc_now(),
                        raw_cache_path=seeded_ratings_path,
                    )
                    inserted += 1
                    log(args, f"IMDb {index}/{len(candidates)}: stored rating snapshot for {existing_tconst}")
                else:
                    log(args, f"IMDb {index}/{len(candidates)}: no title.ratings row for {existing_tconst}")
                upsert_state(conn, movie_id=movie.movie_id, source="imdb", stage="completed", status="completed")
                processed_movie_ids.add(movie.movie_id)
                commit_if_possible(conn)
            except Exception as exc:
                rollback_if_possible(conn)
                upsert_state(
                    conn,
                    movie_id=movie.movie_id,
                    source="imdb",
                    stage="error",
                    status="failed",
                    last_error=str(exc),
                )
                commit_if_possible(conn)
                log(args, f"IMDb {index}/{len(candidates)}: failed for existing match {movie.title}; continuing: {exc}")

    candidates_to_match = [movie for movie in candidates if movie.movie_id not in processed_movie_ids]
    if not candidates_to_match:
        log(args, f"Stored {inserted} IMDb snapshots")
        return inserted
    if getattr(args, "skip_imdb_title_matching", False):
        log(
            args,
            f"Skipping IMDb title matching for {len(candidates_to_match)} movies without existing IMDb IDs",
        )
        log(args, f"Stored {inserted} IMDb snapshots")
        return inserted
    log(args, f"IMDb title matching needed for {len(candidates_to_match)} movies without existing ratings snapshots")
    wanted_titles = {
        normalize_title(variant)
        for movie in candidates_to_match
        for variant in title_variants(movie.title, movie.movie_url)
    }
    title_pool, aka_pool, rating_by_tconst, ratings_path = load_imdb_datasets(
        fetcher,
        wanted_titles=wanted_titles,
        seeded_tconsts=set(),
        args=args,
    )
    log(args, f"Loaded {len(title_pool)} IMDb title candidates and {len(aka_pool)} AKA rows after filtering")
    upsert_imdb_titles(conn, title_pool, last_seen_at=utc_now())
    commit_if_possible(conn)
    for index, movie in enumerate(candidates_to_match, start=1):
        try:
            log(args, f"IMDb {index}/{len(candidates_to_match)}: matching {movie.title}")
            upsert_state(conn, movie_id=movie.movie_id, source="imdb", stage="matching", status="running")
            match = match_imdb_title(movie, title_pool, aka_pool)
            existing_tconst = load_movie_imdb_tconst(conn, movie.movie_id)
            if not match.tconst and existing_tconst:
                match = ImdbMatch(
                    movie_id=movie.movie_id,
                    tconst=existing_tconst,
                    match_status="matched",
                    match_method="existing_id",
                    match_score=1.0,
                    notes="Preserved existing IMDb match from manual override or ID pre-match",
                )
                log(args, f"IMDb {index}/{len(candidates_to_match)}: keeping existing match {existing_tconst}")
            elif existing_match_status(conn, table="movie_imdb_titles", movie_id=movie.movie_id) == "manual_override":
                match = ImdbMatch(
                    movie_id=movie.movie_id,
                    tconst=existing_tconst,
                    match_status="manual_override",
                    match_method="existing_manual_override",
                    match_score=1.0,
                    notes="Preserved manual IMDb override",
                )
                log(args, f"IMDb {index}/{len(candidates_to_match)}: preserving manual override {existing_tconst}")
            else:
                upsert_imdb_match(conn, match)
            if match.tconst and match.tconst in rating_by_tconst:
                insert_imdb_snapshot(
                    conn,
                    rating=rating_by_tconst[match.tconst],
                    snapshot_date=args.snapshot_date,
                    fetched_at=utc_now(),
                    raw_cache_path=ratings_path,
                )
                inserted += 1
                log(args, f"IMDb {index}/{len(candidates_to_match)}: matched {movie.title} -> {match.tconst}")
            else:
                log(args, f"IMDb {index}/{len(candidates_to_match)}: {match.match_status} for {movie.title}")
            upsert_state(conn, movie_id=movie.movie_id, source="imdb", stage="completed", status="completed")
            commit_if_possible(conn)
        except Exception as exc:
            rollback_if_possible(conn)
            upsert_state(
                conn,
                movie_id=movie.movie_id,
                source="imdb",
                stage="error",
                status="failed",
                last_error=str(exc),
            )
            commit_if_possible(conn)
            log(args, f"IMDb {index}/{len(candidates_to_match)}: failed for {movie.title}; continuing: {exc}")
    log(args, f"Stored {inserted} IMDb snapshots")
    return inserted


def load_movie_imdb_tconst(conn: Any, movie_id: int) -> str | None:
    row = conn.execute(
        """
        SELECT tconst
        FROM movie_imdb_titles
        WHERE movie_id = %s
          AND match_status IN ('matched', 'manual_override')
        """,
        (movie_id,),
    ).fetchone()
    return row[0] if row else None


def load_movie_letterboxd_slug(conn: Any, movie_id: int) -> str | None:
    row = conn.execute(
        """
        SELECT letterboxd_slug
        FROM movie_letterboxd_films
        WHERE movie_id = %s
          AND match_status IN ('matched', 'manual_override')
        """,
        (movie_id,),
    ).fetchone()
    return row[0] if row else None


def ingest_seeded_letterboxd_slug(
    args: argparse.Namespace,
    conn: Any,
    *,
    movie: CandidateMovie,
    slug: str,
    fetcher: CachedFetcher,
) -> bool:
    film_url = f"{LETTERBOXD_BASE_URL}/film/{slug}/"
    log(args, f"Letterboxd: fetching seeded slug for {movie.title}: {film_url}")
    film_html, film_cache, _ = fetcher.get_text(film_url)
    page = parse_letterboxd_film_page(film_html, film_url=film_url)
    if not page.letterboxd_slug:
        return False
    imdb_tconst = load_movie_imdb_tconst(conn, movie.movie_id)
    upsert_letterboxd_film(conn, page, last_seen_at=utc_now())
    match = match_letterboxd_page(movie=movie, page=page, imdb_tconst=imdb_tconst)
    if match.match_status != "matched":
        match = LetterboxdMatch(
            movie_id=movie.movie_id,
            letterboxd_slug=page.letterboxd_slug,
            match_status="matched",
            match_method="existing_slug",
            match_score=1.0,
            notes="Fetched pre-matched Letterboxd slug",
        )
    upsert_letterboxd_match(conn, match)
    insert_letterboxd_snapshot(
        conn,
        page=page,
        snapshot_date=args.snapshot_date,
        fetched_at=utc_now(),
        raw_cache_path=film_cache,
    )
    return True


def ingest_predicted_letterboxd_slug(
    args: argparse.Namespace,
    conn: Any,
    *,
    movie: CandidateMovie,
    slug: str,
    fetcher: CachedFetcher,
) -> bool:
    film_url = f"{LETTERBOXD_BASE_URL}/film/{slug}/"
    log(args, f"Letterboxd: trying predicted slug for {movie.title}: {film_url}")
    film_html, film_cache, _ = fetcher.get_text(film_url)
    page = parse_letterboxd_film_page(film_html, film_url=film_url)
    if not page.letterboxd_slug:
        return False
    imdb_tconst = load_movie_imdb_tconst(conn, movie.movie_id)
    match = match_letterboxd_page(movie=movie, page=page, imdb_tconst=imdb_tconst)
    if match.match_status != "matched":
        return False
    upsert_letterboxd_film(conn, page, last_seen_at=utc_now())
    upsert_letterboxd_match(
        conn,
        LetterboxdMatch(
            movie_id=match.movie_id,
            letterboxd_slug=match.letterboxd_slug,
            match_status=match.match_status,
            match_method=f"predicted_slug_{match.match_method}",
            match_score=match.match_score,
            notes=match.notes,
        ),
    )
    insert_letterboxd_snapshot(
        conn,
        page=page,
        snapshot_date=args.snapshot_date,
        fetched_at=utc_now(),
        raw_cache_path=film_cache,
    )
    return True


def ingest_letterboxd(args: argparse.Namespace, conn: Any, candidates: list[CandidateMovie], fetcher: CachedFetcher) -> int:
    if not candidates:
        log(args, "Skipping Letterboxd ingest: no candidate movies")
        return 0
    log(args, f"Starting Letterboxd ingest for {len(candidates)} candidate movies")
    inserted = 0
    for index, movie in enumerate(candidates, start=1):
        log(args, f"Letterboxd {index}/{len(candidates)}: searching {movie.title}")
        upsert_state(conn, movie_id=movie.movie_id, source="letterboxd", stage="search", status="running")
        try:
            seeded_slug = load_movie_letterboxd_slug(conn, movie.movie_id)
            if seeded_slug:
                try:
                    if ingest_seeded_letterboxd_slug(
                        args,
                        conn,
                        movie=movie,
                        slug=seeded_slug,
                        fetcher=fetcher,
                    ):
                        inserted += 1
                        log(args, f"Letterboxd {index}/{len(candidates)}: used seeded slug {seeded_slug}")
                        upsert_state(
                            conn,
                            movie_id=movie.movie_id,
                            source="letterboxd",
                            stage="completed",
                            status="completed",
                        )
                        commit_if_possible(conn)
                        continue
                except Exception as exc:
                    log(
                        args,
                        f"Letterboxd {index}/{len(candidates)}: seeded slug {seeded_slug} failed; falling back to search: {exc}",
                    )

            predicted_slugs = [
                slug for slug in letterboxd_candidate_slugs(movie)
                if slug != seeded_slug
            ]
            if predicted_slugs:
                log(
                    args,
                    f"Letterboxd {index}/{len(candidates)}: trying {len(predicted_slugs)} predictable URLs before search",
                )
            predicted_matched = False
            for slug in predicted_slugs:
                try:
                    if ingest_predicted_letterboxd_slug(
                        args,
                        conn,
                        movie=movie,
                        slug=slug,
                        fetcher=fetcher,
                    ):
                        inserted += 1
                        predicted_matched = True
                        log(args, f"Letterboxd {index}/{len(candidates)}: matched predictable slug {slug}")
                        upsert_state(
                            conn,
                            movie_id=movie.movie_id,
                            source="letterboxd",
                            stage="completed",
                            status="completed",
                        )
                        commit_if_possible(conn)
                        break
                except Exception as exc:
                    rollback_if_possible(conn)
                    log(args, f"Letterboxd {index}/{len(candidates)}: predicted slug {slug} failed: {exc}")
            if predicted_matched:
                continue

            search_url = letterboxd_search_url(movie.title, movie.release_year)
            search_html, _search_cache, _ = fetcher.get_text(search_url)
            results = parse_letterboxd_search(search_html)
            log(args, f"Letterboxd {index}/{len(candidates)}: found {len(results)} candidate links")
            if not results:
                upsert_letterboxd_match(
                    conn,
                    LetterboxdMatch(movie.movie_id, None, "not_found", "letterboxd_search", None, "No film results"),
                )
                upsert_state(conn, movie_id=movie.movie_id, source="letterboxd", stage="completed", status="completed")
                commit_if_possible(conn)
                continue

            imdb_tconst = load_movie_imdb_tconst(conn, movie.movie_id)
            accepted_match: LetterboxdMatch | None = None
            accepted_page: LetterboxdFilmPage | None = None
            accepted_cache: Path | None = None
            rejected: list[str] = []
            for result_index, result in enumerate(results[:5], start=1):
                log(args, f"Letterboxd {index}/{len(candidates)}: checking candidate {result_index} {result.film_url}")
                film_html, film_cache, _ = fetcher.get_text(result.film_url)
                page = parse_letterboxd_film_page(film_html, film_url=result.film_url)
                if not page.letterboxd_slug:
                    continue
                upsert_letterboxd_film(conn, page, last_seen_at=utc_now())
                match = match_letterboxd_page(movie=movie, page=page, imdb_tconst=imdb_tconst)
                if match.match_status == "matched":
                    accepted_match = match
                    accepted_page = page
                    accepted_cache = film_cache
                    break
                rejected.append(match.notes or result.film_url)

            if accepted_match and accepted_page and accepted_cache:
                upsert_letterboxd_match(conn, accepted_match)
                insert_letterboxd_snapshot(
                    conn,
                    page=accepted_page,
                    snapshot_date=args.snapshot_date,
                    fetched_at=utc_now(),
                    raw_cache_path=accepted_cache,
                )
                inserted += 1
                log(
                    args,
                    f"Letterboxd {index}/{len(candidates)}: matched {movie.title} -> {accepted_page.letterboxd_slug}",
                )
            else:
                upsert_letterboxd_match(
                    conn,
                    LetterboxdMatch(
                        movie.movie_id,
                        None,
                        "not_found",
                        "letterboxd_search",
                        None,
                        "; ".join(rejected[:3]) or "No accepted candidates",
                    ),
                )
                log(args, f"Letterboxd {index}/{len(candidates)}: no accepted match for {movie.title}")
            upsert_state(conn, movie_id=movie.movie_id, source="letterboxd", stage="completed", status="completed")
            commit_if_possible(conn)
        except Exception as exc:
            rollback_if_possible(conn)
            upsert_state(
                conn,
                movie_id=movie.movie_id,
                source="letterboxd",
                stage="error",
                status="failed",
                last_error=str(exc),
            )
            commit_if_possible(conn)
            log(args, f"Letterboxd {index}/{len(candidates)}: failed for {movie.title}; continuing: {exc}")
    log(args, f"Stored {inserted} Letterboxd snapshots")
    return inserted


def validate_args(args: argparse.Namespace) -> None:
    if args.delay_seconds < MIN_DELAY_SECONDS and not args.offline and not args.dry_run:
        raise SystemExit(f"--delay-seconds must be at least {MIN_DELAY_SECONDS:g}")
    if USER_AGENT_PRODUCT not in args.user_agent and not args.offline and not args.dry_run:
        raise SystemExit("--user-agent must include the pm-box-office audience ingest identifier")


def parse_date_arg(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO date: {value}") from exc


def make_fetcher(args: argparse.Namespace) -> CachedFetcher:
    return CachedFetcher(
        args.cache_dir,
        refresh=args.refresh,
        offline=args.offline,
        delay_seconds=args.delay_seconds,
        user_agent=args.user_agent,
    )


def run_audience_source_stage(
    args: argparse.Namespace,
    *,
    source: str,
    candidates: list[CandidateMovie],
) -> int:
    conn = connect_database(database_url=args.database_url)
    fetcher = make_fetcher(args)
    try:
        if source == "imdb":
            row_count = ingest_imdb(args, conn, candidates, fetcher)
            conn.commit()
            log(args, "Committed IMDb stage")
            return row_count
        if source == "letterboxd":
            row_count = ingest_letterboxd(args, conn, candidates, fetcher)
            conn.commit()
            log(args, "Committed Letterboxd stage")
            return row_count
        raise ValueError(f"Unknown audience source: {source}")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def run_audience_source_stages(
    args: argparse.Namespace,
    *,
    candidates: list[CandidateMovie],
) -> tuple[int, int]:
    stages: list[str] = []
    if not args.skip_imdb:
        stages.append("imdb")
    if not args.skip_letterboxd:
        stages.append("letterboxd")
    if not stages:
        return 0, 0
    if len(stages) == 1:
        source = stages[0]
        row_count = run_audience_source_stage(args, source=source, candidates=candidates)
        return (row_count, 0) if source == "imdb" else (0, row_count)

    log(args, "Running IMDb and Letterboxd concurrently with separate DB connections and rate-limit timers")
    results: dict[str, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="audience") as executor:
        futures = {
            executor.submit(run_audience_source_stage, args, source=source, candidates=candidates): source
            for source in stages
        }
        for future in concurrent.futures.as_completed(futures):
            source = futures[future]
            results[source] = future.result()
            log(args, f"Finished {source} stage")
    return results.get("imdb", 0), results.get("letterboxd", 0)


def run(args: argparse.Namespace) -> int:
    args.user_agent = resolve_user_agent(args.user_agent)
    validate_args(args)
    log(args, f"Starting audience ingest for snapshot date {args.snapshot_date.isoformat()}")
    log(args, f"Using cache directory {args.cache_dir}")
    log(args, f"Using User-Agent: {args.user_agent}")
    conn = connect_database(database_url=args.database_url)
    try:
        log(args, "Initializing audience database schema")
        initialize_database(conn)
        if args.reset_failed:
            log(args, "Resetting failed audience ingest states")
            reset_failed_states(conn)
            conn.commit()
        fetcher = make_fetcher(args)
        if args.dry_run:
            log(args, "Dry run: selecting candidate movies without fetching")
            candidates = select_candidate_movies(
                conn,
                snapshot_date=args.snapshot_date,
                lookahead_days=args.lookahead_days,
                active_days=args.active_days,
                movie_limit=args.max_movies,
            )
            for movie in candidates:
                print(f"{movie.movie_id}\t{movie.title}\t{movie.release_date or ''}")
            return 0

        release_rows = 0 if args.skip_release_schedule else ingest_release_schedule(args, conn, fetcher)
        conn.commit()
        log(args, "Committed The Numbers release schedule stage")
        chart_movie_rows = upsert_movies_from_current_chart_pages(
            conn,
            snapshot_date=args.snapshot_date,
            active_days=args.active_days,
        )
        conn.commit()
        log(args, f"Upserted {chart_movie_rows} current chart-page movies into movies")
        log(
            args,
            f"Selecting candidates: upcoming <= {args.lookahead_days} days, current weekly window <= {args.active_days} days",
        )
        candidates = select_candidate_movies(
            conn,
            snapshot_date=args.snapshot_date,
            lookahead_days=args.lookahead_days,
            active_days=args.active_days,
            movie_limit=args.max_movies,
        )
        log(args, f"Selected {len(candidates)} candidate movies")
        for index, movie in enumerate(candidates, start=1):
            log(args, f"Candidate {index}/{len(candidates)}: {movie.title} ({movie.release_date or 'no release date'})")
        wikidata_rows = 0
        if not args.skip_wikidata:
            wikidata_rows = ingest_wikidata_matches(args, conn, candidates, fetcher)
            conn.commit()
            log(args, f"Committed Wikidata match stage ({wikidata_rows} movies seeded)")
        imdb_rows, letterboxd_rows = run_audience_source_stages(args, candidates=candidates)
        print(
            f"Imported {release_rows} The Numbers release rows, selected {len(candidates)} movies, "
            f"seeded {wikidata_rows} movies from Wikidata, stored {imdb_rows} IMDb snapshots "
            f"and {letterboxd_rows} Letterboxd snapshots.",
            file=sys.stderr,
            flush=True,
        )
    finally:
        conn.close()
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import IMDb and Letterboxd audience-count snapshots.")
    parser.add_argument(
        "--lookahead-days",
        type=int,
        default=14,
        help="Include upcoming The Numbers releases through snapshot date plus this many days.",
    )
    parser.add_argument(
        "--active-days",
        type=int,
        default=7,
        help="Include current theatrical movies with The Numbers box-office activity in this recent-day window.",
    )
    parser.add_argument("--snapshot-date", type=parse_date_arg, default=dt.date.today())
    parser.add_argument("--database-url", default=database_url_from_env())
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--delay-seconds", type=float, default=MIN_DELAY_SECONDS)
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help=(
            "HTTP User-Agent. Use 'random' to generate a browser User-Agent with fake-useragent "
            "and append the pm-box-office identifier."
        ),
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-movies", type=int)
    parser.add_argument("--quiet", action="store_true", help="Only print dry-run rows and the final summary.")
    parser.add_argument("--reset-failed", action="store_true")
    parser.add_argument("--skip-release-schedule", action="store_true")
    parser.add_argument("--skip-wikidata", action="store_true")
    parser.add_argument("--skip-imdb", action="store_true")
    parser.add_argument(
        "--skip-imdb-title-matching",
        action="store_true",
        help="Only snapshot existing/manual/Wikidata IMDb matches; skip slower title.basics/title.akas matching.",
    )
    parser.add_argument(
        "--imdb-search-aka-titles",
        action="store_true",
        help="During IMDb fallback matching, scan AKA titles by normalized title. Slower but can find alias-only matches.",
    )
    parser.add_argument("--skip-letterboxd", action="store_true")
    return parser


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
