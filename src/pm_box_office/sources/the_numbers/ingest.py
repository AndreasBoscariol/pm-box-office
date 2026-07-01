#!/usr/bin/env python3
"""Scrape a minimal The Numbers daily box-office sample into PostgreSQL.

The default run targets May 1-31, 2026. It discovers movies from daily
domestic chart pages, then imports each discovered movie page's full daily
domestic run. The fetcher is deliberately cache-first, single-threaded, and
slow because The Numbers restricts automated scraping in its terms.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from html.parser import HTMLParser
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pm_box_office.db.connection import connect_database, database_url_from_env, insert_ignore_sql


BASE_URL = "https://www.the-numbers.com"
DEFAULT_START_DATE = dt.date(2026, 6, 1)
DEFAULT_END_DATE = dt.date(2026, 6, 30)
TRANSIENT_STATUSES = {429, 500, 502, 503, 504}
MIN_DELAY_SECONDS = 20.0


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
            raise FileNotFoundError(f"Cache miss in offline mode: {url}")

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
                    raise
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


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()


def absolute_url(href: str) -> str:
    return urllib.parse.urljoin(BASE_URL, href)


def parse_money(value: str) -> int | None:
    text = clean_text(value)
    if not text or text in {"-", "n/a"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    digits = re.sub(r"[^0-9]", "", text)
    if not digits:
        return None
    amount = int(digits)
    return -amount if negative else amount


def parse_int(value: str) -> int | None:
    text = clean_text(value)
    if not text or text in {"-", "n/a"}:
        return None
    digits = re.sub(r"[^0-9-]", "", text)
    if not digits or digits == "-":
        return None
    return int(digits)


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
            movie_url TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            release_year INTEGER,
            opusdata_id TEXT UNIQUE,
            created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP::text),
            updated_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP::text)
        );

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
        CREATE INDEX IF NOT EXISTS idx_daily_box_office_date
            ON daily_box_office(box_office_date);
        CREATE INDEX IF NOT EXISTS idx_daily_box_office_run_date
            ON daily_box_office(release_run_id, box_office_date);
        CREATE INDEX IF NOT EXISTS idx_movies_title_year
            ON movies(title, release_year);
        """
    )


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
            chart_date, movie_url, title, rank, prev_rank, gross_usd,
            daily_change_pct, weekly_change_pct, theaters, per_theater_usd,
            cumulative_gross_usd, days_in_release, source_url, fetched_at,
            raw_cache_path
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(chart_date, movie_url, source_url) DO UPDATE SET
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


def upsert_movie(conn: Any, row: MovieDailyRow) -> int:
    conn.execute(
        """
        INSERT INTO movies (movie_url, title, release_year, opusdata_id, updated_at)
        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT(movie_url) DO UPDATE SET
            title = excluded.title,
            release_year = excluded.release_year,
            opusdata_id = COALESCE(excluded.opusdata_id, movies.opusdata_id),
            updated_at = CURRENT_TIMESTAMP
        """,
        (row.movie_url, row.title, row.release_year, row.opusdata_id),
    )
    movie_id = conn.execute(
        "SELECT movie_id FROM movies WHERE movie_url = %s", (row.movie_url,)
    ).fetchone()[0]
    upsert_movie_source_id(
        conn,
        movie_id=int(movie_id),
        source="the_numbers",
        source_movie_id=row.movie_url,
        source_title=row.title,
    )
    return int(movie_id)


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
    conn.execute(
        """
        INSERT INTO movie_source_ids (
            movie_id, source, source_movie_id, source_title,
            match_status, match_method, match_score, matched_at
        )
        VALUES (%s, %s, %s, %s, 'matched', 'source_primary_key', 1.0, CURRENT_TIMESTAMP)
        ON CONFLICT(source, source_movie_id) DO UPDATE SET
            movie_id = excluded.movie_id,
            source_title = excluded.source_title,
            match_status = excluded.match_status,
            match_method = excluded.match_method,
            match_score = excluded.match_score,
            matched_at = excluded.matched_at
        """,
        (movie_id, source, source_movie_id, source_title),
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
        [
            (
                release_run_id,
                row.box_office_date,
                row.days_in_release,
                row.rank,
                row.gross_usd,
                row.percent_yesterday,
                row.percent_last_week,
                row.theaters,
                row.per_theater_usd,
                row.cumulative_gross_usd,
                row.is_preview,
                row.source_url,
                fetched_at,
                str(raw_cache_path),
            )
            for row in rows
        ],
    )


def movie_page_imported(conn: Any, *, movie_url: str) -> bool:
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


def reconcile(conn: Any, *, issue_source: str) -> int:
    conn.execute(
        "DELETE FROM box_office_import_issues WHERE issue_source = %s",
        (issue_source,),
    )
    chart_rows = conn.execute(
        """
        SELECT chart_date, movie_url, title, gross_usd, theaters, cumulative_gross_usd
        FROM daily_chart_pages
        """
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
        issue_count += compare_value(
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
            issue_count += compare_value(
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


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    days = date_range(args.start_date, args.end_date)
    chart_urls = [daily_chart_url(day) for day in days]
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
        discovered_movie_urls: dict[str, str] = {}
        chart_row_count = 0
        skipped_chart_count = 0
        for day, url in zip(days, chart_urls):
            if not args.refresh and source_page_recorded(
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
                html, cache_path, _fetched = fetcher.get(url)
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

        movie_urls = sorted(discovered_movie_urls)
        if args.max_movies is not None:
            movie_urls = movie_urls[: args.max_movies]
        movie_row_count = 0
        skipped_movie_count = 0
        for index, movie_url in enumerate(movie_urls, start=1):
            if not args.refresh and movie_page_imported(conn, movie_url=movie_url):
                print(
                    f"Skipping imported movie {index}/{len(movie_urls)} "
                    f"{discovered_movie_urls[movie_url]}",
                    file=sys.stderr,
                )
                skipped_movie_count += 1
                continue
            print(
                f"Reading movie {index}/{len(movie_urls)} {discovered_movie_urls[movie_url]}",
                file=sys.stderr,
            )
            conn.commit()
            html, cache_path, _fetched = fetcher.get(movie_url)
            fetched_at = dt.datetime.now(dt.UTC).isoformat()
            rows = parse_movie_page(html, movie_url=movie_url, source_url=movie_url)
            record_raw_page(
                conn,
                source_url=movie_url,
                source_page_type="movie_page",
                fetched_at=fetched_at,
                cache_path=cache_path,
                html=html,
            )
            insert_movie_daily_rows(conn, rows, fetched_at=fetched_at, raw_cache_path=cache_path)
            conn.commit()
            movie_row_count += len(rows)

        issue_count = reconcile(conn, issue_source=args.issue_source)
        conn.commit()
        print(
            f"Imported {chart_row_count} new chart rows, {len(movie_urls)} discovered movies, "
            f"{movie_row_count} new movie daily rows, skipped {skipped_chart_count} chart pages "
            f"and {skipped_movie_count} movie pages, {issue_count} reconciliation issues.",
            file=sys.stderr,
        )
    finally:
        conn.close()
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.delay_seconds < MIN_DELAY_SECONDS and not args.offline and not args.dry_run:
        raise SystemExit(f"--delay-seconds must be at least {MIN_DELAY_SECONDS:g}")
    if "bot" not in args.user_agent.lower() and not args.offline and not args.dry_run:
        raise SystemExit("--user-agent must identify the scraper as a bot")


def parse_date_arg(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO date: {value}") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import a minimal last-month The Numbers US+Canada box-office sample."
    )
    parser.add_argument("--start-date", type=parse_date_arg, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=parse_date_arg, default=DEFAULT_END_DATE)
    parser.add_argument(
        "--database-url",
        default=database_url_from_env(),
        help="PostgreSQL connection URL. Defaults to DATABASE_URL or POSTGRES_DSN.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/the_numbers"),
        help="Raw HTML cache directory.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=MIN_DELAY_SECONDS,
        help="Delay between uncached HTTP requests. Must be at least 20.",
    )
    parser.add_argument(
        "--user-agent",
        default="pm-box-office-the-numbers-bot/1.0 (+personal research; set --user-agent contact)",
        help="HTTP User-Agent. Must identify as a bot when fetching.",
    )
    parser.add_argument("--refresh", action="store_true", help="Refetch even when cache exists.")
    parser.add_argument("--offline", action="store_true", help="Require all pages to exist in cache.")
    parser.add_argument("--dry-run", action="store_true", help="Print chart URLs and exit.")
    parser.add_argument(
        "--max-movies",
        type=int,
        help="Optional cap for smoke tests after chart discovery.",
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
