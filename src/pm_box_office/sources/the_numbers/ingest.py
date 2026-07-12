#!/usr/bin/env python3
"""Scrape a minimal The Numbers daily box-office sample into PostgreSQL.

The default run targets June 25-July 1, 2026. It discovers movies from daily
domestic chart pages, then imports each discovered movie page's full daily
domestic run. The fetcher is deliberately cache-first, single-threaded, and
slow because The Numbers restricts automated scraping in its terms; cached
movie-page parsing can run concurrently because it only reads local files.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
from html.parser import HTMLParser
import re
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pm_box_office.domain import movies as movie_identity
from pm_box_office.db.connection import connect_database, insert_ignore_sql
from pm_box_office.sources.common.cli import add_cache_args, add_database_arg, parse_date_arg
from pm_box_office.sources.common.fetch import CacheFirstFetcher
from pm_box_office.sources.common.parsing import clean_text, parse_int, parse_money
from pm_box_office.sources.common.schema import acquire_schema_init_lock


BASE_URL = "https://www.the-numbers.com"
DEFAULT_START_DATE = dt.date(2026, 7, 9)
DEFAULT_END_DATE = dt.date(2026, 7, 11)
TRANSIENT_STATUSES = {429, 500, 502, 503, 504}
MIN_DELAY_SECONDS = 1
DEFAULT_PARSE_WORKERS = 16


@dataclass(frozen=True)
class ParsedTable:
    heading: str
    rows: list[list[str]]
    hrefs: list[list[str]]


@dataclass(frozen=True)
class DailyChartRow:
    chart_date: str
    movie_url: str
    title: str
    rank: str | None
    prev_rank: str | None
    gross_usd: int | None
    daily_change_pct: float | None
    weekly_change_pct: float | None
    theaters: int | None
    per_theater_usd: int | None
    cumulative_gross_usd: int | None
    days_in_release: int | None
    source_url: str


@dataclass(frozen=True)
class MovieDailyRow:
    movie_url: str
    title: str
    release_year: int | None
    opusdata_id: str | None
    box_office_date: str
    rank: str | None
    gross_usd: int | None
    percent_yesterday: float | None
    percent_last_week: float | None
    theaters: int | None
    per_theater_usd: int | None
    cumulative_gross_usd: int | None
    days_in_release: int | None
    is_preview: int
    source_url: str


@dataclass(frozen=True)
class MovieMetadata:
    movie_url: str
    title: str
    release_year: int | None
    opusdata_id: str | None
    mpa_rating: str | None
    mpa_rating_details: str | None
    genre: str | None
    production_budget_usd: int | None
    franchise: str | None
    source_url: str


@dataclass(frozen=True)
class MovieParseResult:
    movie_url: str
    title: str
    rows: list[MovieDailyRow]
    metadata: MovieMetadata
    fetched_at: str
    raw_cache_path: Path
    html: str


class TableParser(HTMLParser):
    """Small HTML table extractor tailored to The Numbers pages."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[ParsedTable] = []
        self.headings: list[tuple[str, str]] = []
        self._current_heading = ""
        self._capture_heading: str | None = None
        self._heading_parts: list[str] = []
        self._in_table = False
        self._table_rows: list[list[str]] = []
        self._table_hrefs: list[list[str]] = []
        self._in_row = False
        self._row_cells: list[str] = []
        self._row_hrefs: list[str] = []
        self._in_cell = False
        self._cell_parts: list[str] = []
        self._cell_hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag in {"h1", "h2", "h3"}:
            self._capture_heading = tag
            self._heading_parts = []
        elif tag == "table":
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
        if self._capture_heading == tag:
            text = clean_text(" ".join(self._heading_parts))
            if text:
                self._current_heading = text
                self.headings.append((tag, text))
            self._capture_heading = None
            self._heading_parts = []
        elif self._in_cell and tag in {"td", "th"}:
            self._row_cells.append(clean_text(" ".join(self._cell_parts)))
            self._row_hrefs.append(" ".join(self._cell_hrefs))
            self._in_cell = False
            self._cell_parts = []
            self._cell_hrefs = []
        elif self._in_row and tag == "tr":
            if any(cell for cell in self._row_cells):
                self._table_rows.append(self._row_cells)
                self._table_hrefs.append(self._row_hrefs)
            self._in_row = False
            self._row_cells = []
            self._row_hrefs = []
        elif self._in_table and tag == "table":
            self.tables.append(
                ParsedTable(
                    heading=self._current_heading,
                    rows=self._table_rows,
                    hrefs=self._table_hrefs,
                )
            )
            self._in_table = False
            self._table_rows = []
            self._table_hrefs = []

    def handle_data(self, data: str) -> None:
        if self._capture_heading:
            self._heading_parts.append(data)
        if self._in_cell:
            self._cell_parts.append(data)


class HtmlFetcher(CacheFirstFetcher):
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
        super().__init__(
            cache_dir,
            refresh=refresh,
            offline=offline,
            delay_seconds=delay_seconds,
            user_agent=user_agent,
            timeout_seconds=timeout_seconds,
            retries=2,
            transient_statuses=TRANSIENT_STATUSES,
            default_accept="text/html,application/xhtml+xml",
        )

    def get(self, url: str, *, refresh: bool | None = None) -> tuple[str, Path, bool]:
        return self.get_text(url, suffix=".html", refresh=refresh)


def absolute_url(href: str) -> str:
    return urllib.parse.urljoin(BASE_URL, href)


def parse_percent(value: str) -> float | None:
    text = clean_text(value)
    if not text or text in {"-", "n/a"}:
        return None
    match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%", text)
    if not match:
        return None
    return float(match.group(1))


def parse_numbered_date(value: str) -> str | None:
    text = clean_text(value)
    text = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", text)
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return dt.datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def daily_chart_url(day: dt.date) -> str:
    return f"{BASE_URL}/box-office-chart/daily/{day:%Y/%m/%d}"


def date_range(start_date: dt.date, end_date: dt.date) -> list[dt.date]:
    if end_date < start_date:
        raise ValueError("end date must be on or after start date")
    days = (end_date - start_date).days
    return [start_date + dt.timedelta(days=offset) for offset in range(days + 1)]


def find_table(html: str, predicate: Any) -> ParsedTable | None:
    parser = TableParser()
    parser.feed(html)
    for table in parser.tables:
        if table.rows and predicate(table):
            return table
    return None


def parse_daily_chart(html: str, *, chart_date: dt.date, source_url: str) -> list[DailyChartRow]:
    table = find_table(
        html,
        lambda table: is_daily_chart_header(" ".join(table.rows[0])),
    )
    if table is None:
        raise ValueError(f"Could not find daily chart table in {source_url}")

    rows: list[DailyChartRow] = []
    for cells, hrefs in zip(table.rows[1:], table.hrefs[1:]):
        movie_href = first_movie_href(hrefs)
        if not movie_href or len(cells) < 9:
            continue
        rows.append(
            DailyChartRow(
                chart_date=chart_date.isoformat(),
                movie_url=absolute_url(movie_href),
                title=cells[2],
                rank=normalize_rank(cells[0]),
                prev_rank=normalize_rank(cells[1]),
                gross_usd=parse_money(cells[3]),
                daily_change_pct=parse_percent(cells[4]),
                weekly_change_pct=parse_percent(cells[5]),
                theaters=parse_int(cells[6]),
                per_theater_usd=parse_money(cells[7]),
                cumulative_gross_usd=parse_money(cells[8]),
                days_in_release=parse_int(cells[9]) if len(cells) > 9 else None,
                source_url=source_url,
            )
        )
    return rows


def is_daily_chart_header(header_text: str) -> bool:
    return (
        ("Title" in header_text or "Movie" in header_text)
        and "Gross" in header_text
        and "Days in Release" in header_text
    )


def parse_movie_page(html: str, *, movie_url: str, source_url: str) -> list[MovieDailyRow]:
    parser = TableParser()
    parser.feed(html)
    title = page_title(parser)
    release_year = parse_release_year(title)
    opusdata_id = parse_opusdata_id(html)
    table = next(
        (
            table
            for table in parser.tables
            if table.heading == "Daily Box Office Performance"
            and table.rows
            and "Date" in table.rows[0]
            and "Gross" in table.rows[0]
        ),
        None,
    )
    if table is None:
        raise ValueError(f"Could not find Daily Box Office Performance table in {source_url}")

    rows: list[MovieDailyRow] = []
    for cells in table.rows[1:]:
        if len(cells) < 8:
            continue
        box_office_date = parse_numbered_date(cells[0])
        if not box_office_date:
            continue
        rank = normalize_rank(cells[1])
        is_preview = 1 if rank == "P" else 0
        rows.append(
            MovieDailyRow(
                movie_url=movie_url,
                title=title,
                release_year=release_year,
                opusdata_id=opusdata_id,
                box_office_date=box_office_date,
                rank=rank,
                gross_usd=parse_money(cells[2]),
                percent_yesterday=parse_percent(cells[3]) if len(cells) > 3 else None,
                percent_last_week=parse_percent(cells[4]) if len(cells) > 4 else None,
                theaters=parse_int(cells[5]) if len(cells) > 5 else None,
                per_theater_usd=parse_money(cells[6]) if len(cells) > 6 else None,
                cumulative_gross_usd=parse_money(cells[7]) if len(cells) > 7 else None,
                days_in_release=parse_int(cells[8]) if len(cells) > 8 else None,
                is_preview=is_preview,
                source_url=source_url,
            )
        )
    return rows


def parse_movie_metadata(html: str, *, movie_url: str, source_url: str) -> MovieMetadata:
    parser = TableParser()
    parser.feed(html)
    title = page_title(parser)
    details = movie_detail_values(parser)
    mpa_rating_details = details.get("MPA Rating")
    return MovieMetadata(
        movie_url=movie_url,
        title=title,
        release_year=parse_release_year(title),
        opusdata_id=parse_opusdata_id(html),
        mpa_rating=parse_mpa_rating(mpa_rating_details),
        mpa_rating_details=mpa_rating_details,
        genre=details.get("Genre"),
        production_budget_usd=parse_money(details.get("Production Budget", "")),
        franchise=details.get("Franchise"),
        source_url=source_url,
    )


def movie_detail_values(parser: TableParser) -> dict[str, str]:
    values: dict[str, str] = {}
    for table in parser.tables:
        for cells in table.rows:
            if len(cells) < 2:
                continue
            label = clean_text(cells[0]).rstrip(":")
            value = clean_text(cells[1])
            if label and value and label not in values:
                values[label] = value
    return values


def parse_mpa_rating(value: str | None) -> str | None:
    if not value:
        return None
    match = re.match(r"(PG-13|NC-17|Not Rated|Unrated|G|PG|R)\b", clean_text(value), re.IGNORECASE)
    if not match:
        return None
    rating = match.group(1).upper()
    return rating.replace("NOT RATED", "Not Rated").replace("UNRATED", "Unrated")


def first_movie_href(hrefs: list[str]) -> str | None:
    for href_blob in hrefs:
        for href in href_blob.split():
            if href.startswith("/movie/") or href.startswith(f"{BASE_URL}/movie/"):
                return href
    return None


def normalize_rank(value: str) -> str | None:
    text = clean_text(value)
    return text or None


def page_title(parser: TableParser) -> str:
    for tag, text in parser.headings:
        if tag == "h1" and text:
            return text
    return ""


def parse_release_year(title: str) -> int | None:
    match = re.search(r"\((\d{4})\)\s*$", title)
    return int(match.group(1)) if match else None


def parse_opusdata_id(html: str) -> str | None:
    match = re.search(r"OpusData ID:\s*</?[^>]*>*\s*([0-9]+)", html, re.IGNORECASE)
    if match:
        return match.group(1)
    text = re.sub(r"<[^>]+>", " ", html)
    match = re.search(r"OpusData ID:\s*([0-9]+)", clean_text(text), re.IGNORECASE)
    return match.group(1) if match else None


def initialize_database(conn: Any) -> None:
    acquire_schema_init_lock(conn)
    movie_identity.ensure_movie_identity_schema(conn)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS raw_source_pages (
            source_url TEXT PRIMARY KEY,
            source_page_type TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            raw_cache_path TEXT NOT NULL,
            sha256 TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS daily_chart_pages (
            daily_chart_page_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            movie_id BIGINT REFERENCES movies(movie_id),
            chart_date TEXT NOT NULL,
            movie_url TEXT NOT NULL,
            title TEXT NOT NULL,
            rank TEXT,
            prev_rank TEXT,
            gross_usd INTEGER,
            daily_change_pct DOUBLE PRECISION,
            weekly_change_pct DOUBLE PRECISION,
            theaters INTEGER,
            per_theater_usd INTEGER,
            cumulative_gross_usd INTEGER,
            days_in_release INTEGER,
            source_url TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            raw_cache_path TEXT NOT NULL,
            UNIQUE(chart_date, movie_url, source_url)
        );

        CREATE TABLE IF NOT EXISTS movies (
            movie_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            movie_url TEXT,
            title TEXT NOT NULL,
            release_year INTEGER,
            release_date DATE,
            opusdata_id TEXT UNIQUE,
            created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP::text),
            updated_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP::text)
        );

        ALTER TABLE movies
            ADD COLUMN IF NOT EXISTS release_date DATE;

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

        CREATE TABLE IF NOT EXISTS daily_box_office_vintages (
            daily_box_office_vintage_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
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
            UNIQUE(release_run_id, box_office_date, source, fetched_at, raw_cache_path)
        );

        CREATE TABLE IF NOT EXISTS the_numbers_movie_metadata (
            movie_id BIGINT NOT NULL REFERENCES movies(movie_id),
            movie_url TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            release_year INTEGER,
            opusdata_id TEXT,
            mpa_rating TEXT,
            mpa_rating_details TEXT,
            genre TEXT,
            production_budget_usd BIGINT,
            franchise TEXT,
            source_url TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            raw_cache_path TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP::text),
            PRIMARY KEY(movie_id)
        );

        ALTER TABLE the_numbers_movie_metadata
            ADD COLUMN IF NOT EXISTS production_budget_usd BIGINT;

        ALTER TABLE the_numbers_movie_metadata
            ADD COLUMN IF NOT EXISTS franchise TEXT;

        ALTER TABLE daily_chart_pages
            ADD COLUMN IF NOT EXISTS movie_id BIGINT REFERENCES movies(movie_id);

        CREATE TABLE IF NOT EXISTS box_office_import_issues (
            issue_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            issue_source TEXT NOT NULL,
            issue_type TEXT NOT NULL,
            movie_url TEXT NOT NULL,
            box_office_date TEXT NOT NULL,
            chart_value TEXT,
            movie_page_value TEXT,
            details TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP::text),
            UNIQUE(issue_source, issue_type, movie_url, box_office_date, details)
        );

        CREATE INDEX IF NOT EXISTS idx_daily_chart_pages_date
            ON daily_chart_pages(chart_date);
        CREATE INDEX IF NOT EXISTS idx_daily_chart_pages_movie_id
            ON daily_chart_pages(movie_id);
        CREATE INDEX IF NOT EXISTS idx_daily_box_office_date
            ON daily_box_office(box_office_date);
        CREATE INDEX IF NOT EXISTS idx_daily_box_office_run_date
            ON daily_box_office(release_run_id, box_office_date);
        CREATE INDEX IF NOT EXISTS idx_daily_box_office_vintages_as_of
            ON daily_box_office_vintages(release_run_id, box_office_date, fetched_at);
        CREATE INDEX IF NOT EXISTS idx_movies_title_year
            ON movies(title, release_year);
        CREATE INDEX IF NOT EXISTS idx_tn_movie_metadata_rating
            ON the_numbers_movie_metadata(mpa_rating);
        CREATE INDEX IF NOT EXISTS idx_tn_movie_metadata_genre
            ON the_numbers_movie_metadata(genre);
        CREATE INDEX IF NOT EXISTS idx_tn_movie_metadata_franchise
            ON the_numbers_movie_metadata(franchise);
        """
    )
    movie_identity.ensure_movie_identity_schema(conn)


def record_raw_page(
    conn: Any,
    *,
    source_url: str,
    source_page_type: str,
    fetched_at: str,
    cache_path: Path,
    html: str,
) -> None:
    conn.execute(
        """
        INSERT INTO raw_source_pages
            (source_url, source_page_type, fetched_at, raw_cache_path, sha256)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT(source_url) DO UPDATE SET
            source_page_type = excluded.source_page_type,
            fetched_at = excluded.fetched_at,
            raw_cache_path = excluded.raw_cache_path,
            sha256 = excluded.sha256
        """,
        (
            source_url,
            source_page_type,
            fetched_at,
            str(cache_path),
            hashlib.sha256(html.encode("utf-8")).hexdigest(),
        ),
    )


def parse_release_year_from_movie_url(movie_url: str | None) -> int | None:
    if not movie_url:
        return None
    match = re.search(r"\((\d{4})(?:-[^)]+)?\)", urllib.parse.unquote(movie_url))
    return int(match.group(1)) if match else None


def insert_daily_chart_rows(
    conn: Any,
    rows: list[DailyChartRow],
    *,
    fetched_at: str,
    raw_cache_path: Path,
) -> None:
    conn.executemany(
        """
        INSERT INTO daily_chart_pages (
            movie_id, chart_date, movie_url, title, rank, prev_rank, gross_usd,
            daily_change_pct, weekly_change_pct, theaters, per_theater_usd,
            cumulative_gross_usd, days_in_release, source_url, fetched_at,
            raw_cache_path
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(chart_date, movie_url, source_url) DO UPDATE SET
            movie_id = excluded.movie_id,
            title = excluded.title,
            rank = excluded.rank,
            prev_rank = excluded.prev_rank,
            gross_usd = excluded.gross_usd,
            daily_change_pct = excluded.daily_change_pct,
            weekly_change_pct = excluded.weekly_change_pct,
            theaters = excluded.theaters,
            per_theater_usd = excluded.per_theater_usd,
            cumulative_gross_usd = excluded.cumulative_gross_usd,
            days_in_release = excluded.days_in_release,
            fetched_at = excluded.fetched_at,
            raw_cache_path = excluded.raw_cache_path
        """,
        [
            (
                movie_identity.upsert_movie_by_source(
                    conn,
                    source=movie_identity.SOURCE_THE_NUMBERS,
                    source_movie_id=row.movie_url,
                    title=row.title,
                    movie_url=row.movie_url,
                    release_year=parse_release_year_from_movie_url(row.movie_url),
                ),
                row.chart_date,
                row.movie_url,
                row.title,
                row.rank,
                row.prev_rank,
                row.gross_usd,
                row.daily_change_pct,
                row.weekly_change_pct,
                row.theaters,
                row.per_theater_usd,
                row.cumulative_gross_usd,
                row.days_in_release,
                row.source_url,
                fetched_at,
                str(raw_cache_path),
            )
            for row in rows
        ],
    )


def source_page_recorded(
    conn: Any,
    *,
    source_url: str,
    source_page_type: str,
) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM raw_source_pages
        WHERE source_url = %s
          AND source_page_type = %s
        """,
        (source_url, source_page_type),
    ).fetchone()
    return row is not None


def load_daily_chart_rows(conn: Any, *, source_url: str) -> list[DailyChartRow]:
    rows = conn.execute(
        """
        SELECT
            chart_date, movie_url, title, rank, prev_rank, gross_usd,
            daily_change_pct, weekly_change_pct, theaters, per_theater_usd,
            cumulative_gross_usd, days_in_release, source_url
        FROM daily_chart_pages
        WHERE source_url = %s
        ORDER BY daily_chart_page_id
        """,
        (source_url,),
    ).fetchall()
    return [
        DailyChartRow(
            chart_date=row[0],
            movie_url=row[1],
            title=row[2],
            rank=row[3],
            prev_rank=row[4],
            gross_usd=row[5],
            daily_change_pct=row[6],
            weekly_change_pct=row[7],
            theaters=row[8],
            per_theater_usd=row[9],
            cumulative_gross_usd=row[10],
            days_in_release=row[11],
            source_url=row[12],
        )
        for row in rows
    ]


def load_movie_urls_for_metadata_backfill(
    conn: Any,
    *,
    include_complete: bool = False,
) -> list[tuple[str, str]]:
    where_sql = ""
    if not include_complete:
        where_sql = "AND (tnmm.movie_id IS NULL OR tnmm.mpa_rating IS NULL)"
    rows = conn.execute(
        f"""
        SELECT m.movie_url, m.title
        FROM movies m
        LEFT JOIN the_numbers_movie_metadata tnmm ON tnmm.movie_id = m.movie_id
        WHERE m.movie_url IS NOT NULL
          {where_sql}
        ORDER BY m.movie_url
        """
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def load_movie_urls_missing_metadata(conn: Any) -> list[tuple[str, str]]:
    return load_movie_urls_for_metadata_backfill(conn)


def upsert_movie(conn: Any, row: MovieDailyRow) -> int:
    return movie_identity.upsert_movie_by_source(
        conn,
        source=movie_identity.SOURCE_THE_NUMBERS,
        source_movie_id=row.movie_url,
        title=row.title,
        movie_url=row.movie_url,
        release_year=row.release_year,
        opusdata_id=row.opusdata_id,
    )


def upsert_movie_source_id(
    conn: Any,
    *,
    movie_id: int,
    source: str,
    source_movie_id: str,
    source_title: str,
) -> None:
    if not relation_exists(conn, "movie_source_ids"):
        return
    movie_identity.upsert_movie_source_id(
        conn,
        movie_id=movie_id,
        source=source,
        source_movie_id=source_movie_id,
        source_title=source_title,
    )


def upsert_movie_from_metadata(conn: Any, metadata: MovieMetadata) -> int:
    return movie_identity.upsert_movie_by_source(
        conn,
        source=movie_identity.SOURCE_THE_NUMBERS,
        source_movie_id=metadata.movie_url,
        title=metadata.title,
        movie_url=metadata.movie_url,
        release_year=metadata.release_year,
        opusdata_id=metadata.opusdata_id,
    )


def upsert_movie_metadata(
    conn: Any,
    metadata: MovieMetadata,
    *,
    fetched_at: str,
    raw_cache_path: Path,
) -> None:
    movie_id = upsert_movie_from_metadata(conn, metadata)
    conn.execute(
        """
        INSERT INTO the_numbers_movie_metadata (
            movie_id, movie_url, title, release_year, opusdata_id, mpa_rating,
            mpa_rating_details, genre, production_budget_usd, franchise, source_url,
            fetched_at, raw_cache_path, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT(movie_id) DO UPDATE SET
            movie_url = excluded.movie_url,
            title = excluded.title,
            release_year = excluded.release_year,
            opusdata_id = COALESCE(excluded.opusdata_id, the_numbers_movie_metadata.opusdata_id),
            mpa_rating = excluded.mpa_rating,
            mpa_rating_details = excluded.mpa_rating_details,
            genre = excluded.genre,
            production_budget_usd = excluded.production_budget_usd,
            franchise = excluded.franchise,
            source_url = excluded.source_url,
            fetched_at = excluded.fetched_at,
            raw_cache_path = excluded.raw_cache_path,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            movie_id,
            metadata.movie_url,
            metadata.title,
            metadata.release_year,
            metadata.opusdata_id,
            metadata.mpa_rating,
            metadata.mpa_rating_details,
            metadata.genre,
            metadata.production_budget_usd,
            metadata.franchise,
            metadata.source_url,
            fetched_at,
            str(raw_cache_path),
        ),
    )


def relation_exists(conn: Any, relation_name: str) -> bool:
    row = conn.execute("SELECT to_regclass(%s)", (relation_name,)).fetchone()
    return bool(row and row[0])


def upsert_release_run(conn: Any, *, movie_id: int, movie_url: str) -> int:
    conn.execute(
        """
        INSERT INTO release_runs (movie_id, market, release_type, source, source_release_key)
        VALUES (%s, 'US_CA', 'movie_page_full_run', 'the_numbers', %s)
        ON CONFLICT(movie_id, market, source, source_release_key) DO NOTHING
        """,
        (movie_id, movie_url),
    )
    release_run_id = conn.execute(
        """
        SELECT release_run_id
        FROM release_runs
        WHERE movie_id = %s
          AND market = 'US_CA'
          AND source = 'the_numbers'
          AND source_release_key = %s
        """,
        (movie_id, movie_url),
    ).fetchone()[0]
    return int(release_run_id)


def insert_movie_daily_rows(
    conn: Any,
    rows: list[MovieDailyRow],
    *,
    fetched_at: str,
    raw_cache_path: Path,
) -> None:
    if not rows:
        return
    movie_id = upsert_movie(conn, rows[0])
    release_run_id = upsert_release_run(conn, movie_id=movie_id, movie_url=rows[0].movie_url)
    values = [
        (
            release_run_id, row.box_office_date, row.days_in_release, row.rank,
            row.gross_usd, row.percent_yesterday, row.percent_last_week,
            row.theaters, row.per_theater_usd, row.cumulative_gross_usd,
            row.is_preview, row.source_url, fetched_at, str(raw_cache_path),
        )
        for row in rows
    ]
    conn.executemany(
        """
        INSERT INTO daily_box_office_vintages (
            release_run_id, box_office_date, market, day_number, rank, gross_usd,
            percent_yesterday, percent_last_week, theaters, per_theater_usd,
            cumulative_gross_usd, is_preview, is_estimate, source, source_url,
            fetched_at, raw_cache_path
        ) VALUES (%s, %s, 'US_CA', %s, %s, %s, %s, %s, %s, %s, %s, %s, 0, 'the_numbers', %s, %s, %s)
        ON CONFLICT(release_run_id, box_office_date, source, fetched_at, raw_cache_path) DO NOTHING
        """,
        values,
    )
    conn.executemany(
        """
        INSERT INTO daily_box_office (
            release_run_id, box_office_date, market, day_number, rank, gross_usd,
            percent_yesterday, percent_last_week, theaters, per_theater_usd,
            cumulative_gross_usd, is_preview, is_estimate, source, source_url,
            fetched_at, raw_cache_path
        ) VALUES (%s, %s, 'US_CA', %s, %s, %s, %s, %s, %s, %s, %s, %s, 0, 'the_numbers', %s, %s, %s)
        ON CONFLICT(release_run_id, box_office_date, source) DO UPDATE SET
            day_number = excluded.day_number,
            rank = excluded.rank,
            gross_usd = excluded.gross_usd,
            percent_yesterday = excluded.percent_yesterday,
            percent_last_week = excluded.percent_last_week,
            theaters = excluded.theaters,
            per_theater_usd = excluded.per_theater_usd,
            cumulative_gross_usd = excluded.cumulative_gross_usd,
            is_preview = excluded.is_preview,
            is_estimate = excluded.is_estimate,
            source_url = excluded.source_url,
            fetched_at = excluded.fetched_at,
            raw_cache_path = excluded.raw_cache_path
        """,
        values,
    )


def movie_page_imported(
    conn: Any,
    *,
    movie_url: str,
    chart_dates: list[str] | None = None,
) -> bool:
    if chart_dates is not None:
        if not chart_dates:
            return True
        row = conn.execute(
            """
            SELECT COUNT(DISTINCT dbo.box_office_date)
            FROM movies m
            JOIN release_runs rr ON rr.movie_id = m.movie_id
            JOIN daily_box_office dbo ON dbo.release_run_id = rr.release_run_id
            WHERE m.movie_url = %s
              AND rr.source = 'the_numbers'
              AND rr.source_release_key = %s
              AND dbo.source = 'the_numbers'
              AND dbo.box_office_date = ANY(%s)
            """,
            (movie_url, movie_url, chart_dates),
        ).fetchone()
        return bool(row and int(row[0]) == len(set(chart_dates)))

    row = conn.execute(
        """
        SELECT 1
        FROM movies m
        JOIN release_runs rr ON rr.movie_id = m.movie_id
        JOIN daily_box_office dbo ON dbo.release_run_id = rr.release_run_id
        WHERE m.movie_url = %s
          AND dbo.source = 'the_numbers'
        LIMIT 1
        """,
        (movie_url,),
    ).fetchone()
    return row is not None


def movie_metadata_imported(conn: Any, *, movie_url: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM the_numbers_movie_metadata
        WHERE movie_url = %s
        LIMIT 1
        """,
        (movie_url,),
    ).fetchone()
    return row is not None


def reconcile(
    conn: Any,
    *,
    issue_source: str,
    chart_dates: list[str] | None = None,
    movie_urls: list[str] | None = None,
) -> int:
    where_clauses: list[str] = []
    params: list[Any] = []
    delete_clauses: list[str] = ["issue_source = %s"]
    delete_params: list[Any] = [issue_source]

    if chart_dates is not None:
        if not chart_dates:
            return 0
        where_clauses.append("chart_date = ANY(%s)")
        params.append(chart_dates)
        delete_clauses.append("box_office_date = ANY(%s)")
        delete_params.append(chart_dates)
    if movie_urls is not None:
        if not movie_urls:
            return 0
        where_clauses.append("movie_url = ANY(%s)")
        params.append(movie_urls)
        delete_clauses.append("movie_url = ANY(%s)")
        delete_params.append(movie_urls)

    conn.execute(
        f"DELETE FROM box_office_import_issues WHERE {' AND '.join(delete_clauses)}",
        delete_params,
    )
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    chart_rows = conn.execute(
        f"""
        SELECT chart_date, movie_url, title, gross_usd, theaters, cumulative_gross_usd
        FROM daily_chart_pages
        {where_sql}
        """,
        params,
    ).fetchall()
    issue_count = 0
    for chart_date, movie_url, title, gross, theaters, cumulative in chart_rows:
        movie_row = conn.execute(
            """
            SELECT dbo.gross_usd, dbo.theaters, dbo.cumulative_gross_usd
            FROM movies m
            JOIN release_runs rr ON rr.movie_id = m.movie_id
            JOIN daily_box_office dbo ON dbo.release_run_id = rr.release_run_id
            WHERE m.movie_url = %s
              AND dbo.box_office_date = %s
              AND dbo.source = 'the_numbers'
            """,
            (movie_url, chart_date),
        ).fetchone()
        if movie_row is None:
            issue_count += insert_issue(
                conn,
                issue_source=issue_source,
                issue_type="missing_movie_page_row",
                movie_url=movie_url,
                box_office_date=chart_date,
                chart_value=str(gross),
                movie_page_value=None,
                details=f"{title}: chart row has no matching movie-page daily row",
            )
            continue
        movie_gross, movie_theaters, movie_cumulative = movie_row
        issue_count += compare_money_value(
            conn,
            issue_source,
            "gross_mismatch",
            movie_url,
            chart_date,
            gross,
            movie_gross,
            f"{title}: chart gross does not match movie page",
        )
        if theaters is not None and movie_theaters is not None:
            issue_count += compare_value(
                conn,
                issue_source,
                "theaters_mismatch",
                movie_url,
                chart_date,
                theaters,
                movie_theaters,
                f"{title}: chart theaters do not match movie page",
            )
        if cumulative is not None and movie_cumulative is not None:
            issue_count += compare_money_value(
                conn,
                issue_source,
                "cumulative_mismatch",
                movie_url,
                chart_date,
                cumulative,
                movie_cumulative,
                f"{title}: chart cumulative gross does not match movie page",
            )
    return issue_count


def compare_money_value(
    conn: Any,
    issue_source: str,
    issue_type: str,
    movie_url: str,
    box_office_date: str,
    chart_value: int | None,
    movie_page_value: int | None,
    details: str,
) -> int:
    if money_values_match(chart_value, movie_page_value):
        return 0
    return insert_issue(
        conn,
        issue_source=issue_source,
        issue_type=issue_type,
        movie_url=movie_url,
        box_office_date=box_office_date,
        chart_value=str(chart_value),
        movie_page_value=str(movie_page_value),
        details=details,
    )


def money_values_match(chart_value: int | None, movie_page_value: int | None) -> bool:
    if chart_value == movie_page_value:
        return True
    if chart_value is None or movie_page_value is None:
        return False
    return abs(chart_value - movie_page_value) <= rounded_chart_tolerance(chart_value)


def rounded_chart_tolerance(value: int) -> int:
    text = str(abs(value))
    trailing_zero_count = len(text) - len(text.rstrip("0"))
    if trailing_zero_count <= 0:
        return 5
    rounded_unit = 10 ** min(trailing_zero_count, 4)
    return max(5, rounded_unit // 2)


def compare_value(
    conn: Any,
    issue_source: str,
    issue_type: str,
    movie_url: str,
    box_office_date: str,
    chart_value: int | None,
    movie_page_value: int | None,
    details: str,
) -> int:
    if chart_value == movie_page_value:
        return 0
    return insert_issue(
        conn,
        issue_source=issue_source,
        issue_type=issue_type,
        movie_url=movie_url,
        box_office_date=box_office_date,
        chart_value=str(chart_value),
        movie_page_value=str(movie_page_value),
        details=details,
    )


def insert_issue(
    conn: Any,
    *,
    issue_source: str,
    issue_type: str,
    movie_url: str,
    box_office_date: str,
    chart_value: str | None,
    movie_page_value: str | None,
    details: str,
) -> int:
    cursor = conn.execute(
        insert_ignore_sql(
            "box_office_import_issues",
            [
                "issue_source",
                "issue_type",
                "movie_url",
                "box_office_date",
                "chart_value",
                "movie_page_value",
                "details",
            ],
        ),
        (
            issue_source,
            issue_type,
            movie_url,
            box_office_date,
            chart_value,
            movie_page_value,
            details,
        ),
    )
    return 1 if getattr(cursor, "rowcount", 0) > 0 else 0


def parse_cached_movie_page(
    movie_url: str,
    title: str,
    *,
    cache_path: Path,
) -> MovieParseResult:
    html = cache_path.read_text(encoding="utf-8")
    return parse_movie_page_result(
        movie_url,
        title,
        html=html,
        cache_path=cache_path,
    )


def parse_movie_page_result(
    movie_url: str,
    title: str,
    *,
    html: str,
    cache_path: Path,
) -> MovieParseResult:
    fetched_at = dt.datetime.now(dt.UTC).isoformat()
    rows = parse_movie_page(html, movie_url=movie_url, source_url=movie_url)
    metadata = parse_movie_metadata(html, movie_url=movie_url, source_url=movie_url)
    return MovieParseResult(
        movie_url=movie_url,
        title=title,
        rows=rows,
        metadata=metadata,
        fetched_at=fetched_at,
        raw_cache_path=cache_path,
        html=html,
    )


def parse_cached_movie_pages_concurrently(
    movie_items: list[tuple[str, str, Path]],
    *,
    workers: int,
) -> list[MovieParseResult]:
    results_by_index: dict[int, MovieParseResult] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                parse_cached_movie_page,
                movie_url,
                title,
                cache_path=cache_path,
            ): index
            for index, (movie_url, title, cache_path) in enumerate(movie_items)
        }
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            results_by_index[index] = future.result()
    return [results_by_index[index] for index in range(len(movie_items))]


def import_movie_parse_result(conn: Any, result: MovieParseResult) -> tuple[int, int]:
    record_raw_page(
        conn,
        source_url=result.movie_url,
        source_page_type="movie_page",
        fetched_at=result.fetched_at,
        cache_path=result.raw_cache_path,
        html=result.html,
    )
    insert_movie_daily_rows(
        conn,
        result.rows,
        fetched_at=result.fetched_at,
        raw_cache_path=result.raw_cache_path,
    )
    upsert_movie_metadata(
        conn,
        result.metadata,
        fetched_at=result.fetched_at,
        raw_cache_path=result.raw_cache_path,
    )
    conn.commit()
    return len(result.rows), 1


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    days = date_range(args.start_date, args.end_date)
    chart_urls = [daily_chart_url(day) for day in days]
    refresh_chart_dates = {day for day in days if should_refresh_recent_day(day, args)}
    if args.dry_run:
        for url in chart_urls:
            print(url)
        print(f"Chart URLs: {len(chart_urls)}", file=sys.stderr)
        return 0

    fetcher = HtmlFetcher(
        args.cache_dir,
        refresh=args.refresh,
        offline=args.offline,
        delay_seconds=args.delay_seconds,
        user_agent=args.user_agent,
    )
    conn = connect_database(database_url=args.database_url)
    try:
        initialize_database(conn)
        conn.commit()
        if args.metadata_backfill:
            return run_metadata_backfill(args, conn, fetcher)
        discovered_movie_urls: dict[str, str] = {}
        discovered_movie_dates: dict[str, set[dt.date]] = {}
        chart_row_count = 0
        skipped_chart_count = 0
        for day, url in zip(days, chart_urls):
            refresh_chart = args.refresh or day in refresh_chart_dates
            if not refresh_chart and source_page_recorded(
                conn,
                source_url=url,
                source_page_type="daily_chart",
            ):
                print(f"Skipping recorded chart {day.isoformat()} {url}", file=sys.stderr)
                rows = load_daily_chart_rows(conn, source_url=url)
                skipped_chart_count += 1
            else:
                conn.commit()
                print(f"Reading chart {day.isoformat()} {url}", file=sys.stderr)
                html, cache_path, _fetched = fetcher.get(url, refresh=refresh_chart)
                fetched_at = dt.datetime.now(dt.UTC).isoformat()
                rows = parse_daily_chart(html, chart_date=day, source_url=url)
                record_raw_page(
                    conn,
                    source_url=url,
                    source_page_type="daily_chart",
                    fetched_at=fetched_at,
                    cache_path=cache_path,
                    html=html,
                )
                insert_daily_chart_rows(conn, rows, fetched_at=fetched_at, raw_cache_path=cache_path)
                conn.commit()
                chart_row_count += len(rows)
            for row in rows:
                discovered_movie_urls.setdefault(row.movie_url, row.title)
                discovered_movie_dates.setdefault(row.movie_url, set()).add(day)

        movie_urls = sorted(discovered_movie_urls)
        if args.max_movies is not None:
            movie_urls = movie_urls[: args.max_movies]
        movie_row_count = 0
        metadata_row_count = 0
        skipped_movie_count = 0
        cached_movie_items: list[tuple[str, str, Path]] = []
        uncached_movie_items: list[tuple[str, bool]] = []
        for index, movie_url in enumerate(movie_urls, start=1):
            refresh_movie = args.refresh or bool(discovered_movie_dates.get(movie_url, set()) & refresh_chart_dates)
            required_chart_dates = sorted(
                day.isoformat() for day in discovered_movie_dates.get(movie_url, set())
            )
            if (
                not refresh_movie
                and movie_page_imported(conn, movie_url=movie_url, chart_dates=required_chart_dates)
                and movie_metadata_imported(conn, movie_url=movie_url)
            ):
                print(
                    f"Skipping imported movie {index}/{len(movie_urls)} "
                    f"{discovered_movie_urls[movie_url]}",
                    file=sys.stderr,
                )
                skipped_movie_count += 1
                continue
            cache_path = fetcher.cache_path(movie_url)
            if cache_path.exists() and not refresh_movie:
                cached_movie_items.append((movie_url, discovered_movie_urls[movie_url], cache_path))
            else:
                uncached_movie_items.append((movie_url, refresh_movie))

        if cached_movie_items and args.parse_workers > 1:
            print(
                f"Reading and parsing {len(cached_movie_items)} cached movie pages "
                f"with {args.parse_workers} workers.",
                file=sys.stderr,
            )
            results = parse_cached_movie_pages_concurrently(
                cached_movie_items,
                workers=args.parse_workers,
            )
            for index, result in enumerate(results, start=1):
                print(
                    f"Importing parsed movie {index}/{len(results)} {result.title}",
                    file=sys.stderr,
                )
                row_count, metadata_count = import_movie_parse_result(conn, result)
                movie_row_count += row_count
                metadata_row_count += metadata_count
        else:
            for index, (movie_url, title, cache_path) in enumerate(cached_movie_items, start=1):
                print(f"Reading cached movie {index}/{len(cached_movie_items)} {title}", file=sys.stderr)
                result = parse_cached_movie_page(movie_url, title, cache_path=cache_path)
                row_count, metadata_count = import_movie_parse_result(conn, result)
                movie_row_count += row_count
                metadata_row_count += metadata_count

        for index, (movie_url, refresh_movie) in enumerate(uncached_movie_items, start=1):
            print(
                f"Reading uncached movie {index}/{len(uncached_movie_items)} "
                f"{discovered_movie_urls[movie_url]}",
                file=sys.stderr,
            )
            conn.commit()
            html, cache_path, _fetched = fetcher.get(movie_url, refresh=refresh_movie)
            result = parse_movie_page_result(
                movie_url,
                discovered_movie_urls[movie_url],
                html=html,
                cache_path=cache_path,
            )
            row_count, metadata_count = import_movie_parse_result(conn, result)
            movie_row_count += row_count
            metadata_row_count += metadata_count

        issue_count = reconcile(
            conn,
            issue_source=args.issue_source,
            chart_dates=[day.isoformat() for day in days],
            movie_urls=movie_urls,
        )
        conn.commit()
        print(
            f"Imported {chart_row_count} chart rows from fetched charts, {len(movie_urls)} discovered movies, "
            f"{movie_row_count} movie daily rows, {metadata_row_count} movie metadata rows, "
            f"skipped {skipped_chart_count} chart pages and {skipped_movie_count} movie pages, "
            f"{issue_count} reconciliation issues.",
            file=sys.stderr,
        )
    finally:
        conn.close()
    return 0


def run_metadata_backfill(args: argparse.Namespace, conn: Any, fetcher: HtmlFetcher) -> int:
    movie_items = load_movie_urls_for_metadata_backfill(
        conn,
        include_complete=args.metadata_backfill_all,
    )
    if args.max_movies is not None:
        movie_items = movie_items[: args.max_movies]

    metadata_row_count = 0
    skipped_movie_count = 0
    for index, (movie_url, title) in enumerate(movie_items, start=1):
        print(f"Reading metadata {index}/{len(movie_items)} {title}", file=sys.stderr)
        conn.commit()
        try:
            html, cache_path, _fetched = fetcher.get(movie_url)
        except FileNotFoundError as exc:
            print(f"Skipping metadata cache miss {movie_url}: {exc}", file=sys.stderr)
            skipped_movie_count += 1
            continue
        fetched_at = dt.datetime.now(dt.UTC).isoformat()
        metadata = parse_movie_metadata(html, movie_url=movie_url, source_url=movie_url)
        record_raw_page(
            conn,
            source_url=movie_url,
            source_page_type="movie_page",
            fetched_at=fetched_at,
            cache_path=cache_path,
            html=html,
        )
        upsert_movie_metadata(conn, metadata, fetched_at=fetched_at, raw_cache_path=cache_path)
        conn.commit()
        metadata_row_count += 1

    print(
        f"Backfilled {metadata_row_count} The Numbers movie metadata rows, "
        f"skipped {skipped_movie_count} movie pages.",
        file=sys.stderr,
    )
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.delay_seconds < MIN_DELAY_SECONDS and not args.offline and not args.dry_run:
        raise SystemExit(f"--delay-seconds must be at least {MIN_DELAY_SECONDS:g}")
    if "bot" not in args.user_agent.lower() and not args.offline and not args.dry_run:
        raise SystemExit("--user-agent must identify the scraper as a bot")
    if args.parse_workers < 1:
        raise SystemExit("--parse-workers must be at least 1")
    if args.refresh_recent_days < 0:
        raise SystemExit("--refresh-recent-days must be non-negative")


def should_refresh_recent_day(day: dt.date, args: argparse.Namespace) -> bool:
    if args.refresh_recent_days <= 0:
        return False
    start_date = args.end_date - dt.timedelta(days=args.refresh_recent_days - 1)
    return start_date <= day <= args.end_date


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import a minimal last-month The Numbers US+Canada box-office sample."
    )
    parser.add_argument("--start-date", type=parse_date_arg, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=parse_date_arg, default=DEFAULT_END_DATE)
    add_database_arg(parser)
    add_cache_args(
        parser,
        default_cache_dir=Path("data/raw/the_numbers"),
        cache_help="Raw HTML cache directory.",
        include_dry_run=False,
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=MIN_DELAY_SECONDS,
        help=f"Delay between uncached HTTP requests. Must be at least {MIN_DELAY_SECONDS:g}.",
    )
    parser.add_argument(
        "--user-agent",
        default="pm-box-office-the-numbers-bot/1.0 (+personal research; set --user-agent contact)",
        help="HTTP User-Agent. Must identify as a bot when fetching.",
    )
    parser.add_argument(
        "--refresh-recent-days",
        type=int,
        default=0,
        help="Within the requested range, refetch this many trailing days even when already recorded.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print chart URLs and exit.")
    parser.add_argument(
        "--metadata-backfill",
        action="store_true",
        help="Fetch existing movies missing The Numbers metadata or MPA rating, without chart discovery.",
    )
    parser.add_argument(
        "--metadata-backfill-all",
        action="store_true",
        help="With --metadata-backfill, reparse every existing movie URL instead of only missing metadata.",
    )
    parser.add_argument(
        "--max-movies",
        type=int,
        help="Optional cap for smoke tests after chart discovery.",
    )
    parser.add_argument(
        "--parse-workers",
        type=int,
        default=DEFAULT_PARSE_WORKERS,
        help=(
            "Concurrent workers for parsing cached movie pages. "
            "Uncached HTTP fetches remain single-threaded."
        ),
    )
    parser.add_argument(
        "--issue-source",
        default="the_numbers_import",
        help="Label used for reconciliation issues.",
    )
    return parser


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
