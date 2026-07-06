#!/usr/bin/env python3
"""Ingest Wikimedia activity for The Numbers box-office movies into PostgreSQL.

The tables created here are intentionally raw/auditable. Analysis code should
use the `wiki_movie_time_features` view, which computes the cumulative
Wikipedia predictors used by Mestyan, Yasseri, and Kertesz (2013):

    V = pageviews
    U = distinct human editors
    R = collaborative rigor, edit-train user switches
    E = human edits

The script is cache-first and resumable. It processes movies already present in
the existing The Numbers database; it does not discover new The Numbers pages.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pm_box_office.domain import movies as movie_identity
from pm_box_office.db.connection import connect_database, insert_ignore_sql
from pm_box_office.sources.common.cli import add_cache_args, add_database_arg
from pm_box_office.sources.common.fetch import CacheFirstFetcher


DEFAULT_CACHE_DIR = Path("data/raw/wikimedia")
DEFAULT_USER_AGENT = "pm-box-office-wikipedia-ingest/0.1"
TRANSIENT_STATUSES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class CandidateMovie:
    movie_id: int
    movie_url: str | None
    title: str
    release_year: int | None
    release_run_id: int
    opening_date: str
    opening_theaters: int | None
    opening_day_gross_usd: int | None
    opening_weekend_revenue_usd: int | None


@dataclass(frozen=True)
class WikiMatch:
    status: str
    method: str
    query: str | None
    rank: int | None
    score: float | None
    page_id: int | None
    page_title: str | None
    notes: str | None = None


@dataclass(frozen=True)
class PageviewRow:
    language: str
    wiki_page_id: int
    view_date: str
    views: int
    access: str
    agent: str
    source_url: str


@dataclass(frozen=True)
class RevisionRow:
    language: str
    wiki_page_id: int
    rev_id: int
    rev_timestamp: str
    rev_date: str
    user_name: str | None
    user_id: int | None
    user_key: str
    is_bot: int
    is_minor: int
    parent_id: int | None
    source_url: str


class JsonFetcher(CacheFirstFetcher):
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
            retries=3,
            transient_statuses=TRANSIENT_STATUSES,
            default_accept="application/json",
        )

    def get_json(self, url: str) -> tuple[dict[str, Any], Path, bool]:
        data, cache_path, fetched = super().get_json(url, suffix=".json", not_found_json={"items": []})
        return data, cache_path, fetched


def initialize_wikipedia_database(conn: Any) -> None:
    movie_identity.ensure_movie_identity_schema(conn)
    conn.executescript(
        """
            CREATE TABLE IF NOT EXISTS wiki_pages (
                language TEXT NOT NULL,
                wiki_page_id INTEGER NOT NULL,
                page_title TEXT NOT NULL,
                canonical_url TEXT NOT NULL,
                first_seen_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP::text),
                last_seen_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP::text),
                PRIMARY KEY(language, wiki_page_id)
            );

            CREATE TABLE IF NOT EXISTS wiki_pageviews_daily (
                language TEXT NOT NULL,
                wiki_page_id INTEGER NOT NULL,
                view_date TEXT NOT NULL,
                views INTEGER NOT NULL,
                access TEXT NOT NULL,
                agent TEXT NOT NULL,
                source_url TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                raw_cache_path TEXT NOT NULL,
                UNIQUE(language, wiki_page_id, view_date, access, agent)
            );

            CREATE TABLE IF NOT EXISTS wiki_revisions (
                language TEXT NOT NULL,
                wiki_page_id INTEGER NOT NULL,
                rev_id INTEGER NOT NULL,
                rev_timestamp TEXT NOT NULL,
                rev_date TEXT NOT NULL,
                user_name TEXT,
                user_id INTEGER,
                user_key TEXT NOT NULL,
                is_bot INTEGER NOT NULL DEFAULT 0,
                is_minor INTEGER NOT NULL DEFAULT 0,
                parent_id INTEGER,
                source_url TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                raw_cache_path TEXT NOT NULL,
                PRIMARY KEY(language, wiki_page_id, rev_id)
            );

            CREATE TABLE IF NOT EXISTS wiki_ingest_state (
                movie_id BIGINT NOT NULL REFERENCES movies(movie_id),
                language TEXT NOT NULL,
                day_start INTEGER NOT NULL,
                day_end INTEGER NOT NULL,
                stage TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                pageviews_rows INTEGER NOT NULL DEFAULT 0,
                revision_rows INTEGER NOT NULL DEFAULT 0,
                UNIQUE(movie_id, language, day_start, day_end)
            );

            CREATE TABLE IF NOT EXISTS wiki_import_issues (
                issue_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                issue_source TEXT NOT NULL,
                issue_type TEXT NOT NULL,
                movie_id BIGINT,
                movie_url TEXT,
                wiki_page_id INTEGER,
                details TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP::text),
                UNIQUE(issue_source, issue_type, movie_id, details)
            );

            CREATE INDEX IF NOT EXISTS idx_wiki_pageviews_page_date
                ON wiki_pageviews_daily(language, wiki_page_id, view_date);
            CREATE INDEX IF NOT EXISTS idx_wiki_revisions_page_timestamp
                ON wiki_revisions(language, wiki_page_id, rev_timestamp);
            CREATE INDEX IF NOT EXISTS idx_wiki_ingest_state_status_stage
                ON wiki_ingest_state(status, stage);

            DROP VIEW IF EXISTS wiki_movie_time_features;
            DROP VIEW IF EXISTS box_office_opening_features;

            CREATE VIEW box_office_opening_features AS
            WITH opening AS (
                SELECT
                    rr.release_run_id,
                    rr.movie_id,
                    MIN(dbo.box_office_date) AS opening_date
                FROM release_runs rr
                JOIN daily_box_office dbo ON dbo.release_run_id = rr.release_run_id
                WHERE dbo.is_preview = 0
                  AND dbo.gross_usd IS NOT NULL
                GROUP BY rr.release_run_id, rr.movie_id
            )
            SELECT
                opening.release_run_id,
                opening.movie_id,
                opening.opening_date,
                opening_day.theaters AS opening_theaters,
                opening_day.gross_usd AS opening_day_gross_usd,
                SUM(weekend.gross_usd) AS opening_weekend_revenue_usd
            FROM opening
            JOIN daily_box_office opening_day
              ON opening_day.release_run_id = opening.release_run_id
             AND opening_day.box_office_date = opening.opening_date
            JOIN daily_box_office weekend
              ON weekend.release_run_id = opening.release_run_id
             AND weekend.is_preview = 0
             AND weekend.box_office_date::date >= opening.opening_date::date
             AND weekend.box_office_date::date < opening.opening_date::date + INTERVAL '3 days'
            GROUP BY
                opening.release_run_id,
                opening.movie_id,
                opening.opening_date,
                opening_day.theaters,
                opening_day.gross_usd;

            CREATE VIEW wiki_movie_time_features AS
            WITH matched AS (
                SELECT
                    src.movie_id,
                    split_part(src.source_movie_id, ':', 1) AS language,
                    split_part(src.source_movie_id, ':', 2)::integer AS wiki_page_id,
                    bof.release_run_id,
                    bof.opening_date,
                    bof.opening_theaters,
                    bof.opening_day_gross_usd,
                    bof.opening_weekend_revenue_usd
                FROM movie_source_ids src
                JOIN box_office_opening_features bof ON bof.movie_id = src.movie_id
                WHERE src.source = 'wikipedia'
                  AND src.match_status IN ('matched', 'manual_override')
                  AND src.source_movie_id ~ '^[a-z-]+:[0-9]+$'
            ),
            days AS (
                SELECT
                    movie_id,
                    language,
                    wiki_page_id,
                    (view_date::date - opening_date::date) AS movie_time_day
                FROM matched
                JOIN wiki_pageviews_daily USING(language, wiki_page_id)
                UNION
                SELECT
                    movie_id,
                    language,
                    wiki_page_id,
                    (rev_date::date - opening_date::date) AS movie_time_day
                FROM matched
                JOIN wiki_revisions USING(language, wiki_page_id)
                WHERE is_bot = 0
            ),
            human_revisions AS (
                SELECT
                    language,
                    wiki_page_id,
                    rev_id,
                    rev_timestamp,
                    rev_date,
                    user_key,
                    CASE
                        WHEN LAG(user_key) OVER (
                            PARTITION BY language, wiki_page_id
                            ORDER BY rev_timestamp, rev_id
                        ) IS NULL THEN 1
                        WHEN LAG(user_key) OVER (
                            PARTITION BY language, wiki_page_id
                            ORDER BY rev_timestamp, rev_id
                        ) <> user_key THEN 1
                        ELSE 0
                    END AS rigor_increment
                FROM wiki_revisions
                WHERE is_bot = 0
            )
            SELECT
                d.movie_id,
                m.release_run_id,
                d.language,
                d.wiki_page_id,
                d.movie_time_day,
                COALESCE((
                    SELECT SUM(pv.views)
                    FROM wiki_pageviews_daily pv
                    WHERE pv.language = d.language
                      AND pv.wiki_page_id = d.wiki_page_id
                      AND (pv.view_date::date - m.opening_date::date) <= d.movie_time_day
                ), 0) AS V,
                COALESCE((
                    SELECT COUNT(*)
                    FROM human_revisions hr
                    WHERE hr.language = d.language
                      AND hr.wiki_page_id = d.wiki_page_id
                      AND (hr.rev_date::date - m.opening_date::date) <= d.movie_time_day
                ), 0) AS E,
                COALESCE((
                    SELECT COUNT(DISTINCT hr.user_key)
                    FROM human_revisions hr
                    WHERE hr.language = d.language
                      AND hr.wiki_page_id = d.wiki_page_id
                      AND (hr.rev_date::date - m.opening_date::date) <= d.movie_time_day
                ), 0) AS U,
                COALESCE((
                    SELECT SUM(hr.rigor_increment)
                    FROM human_revisions hr
                    WHERE hr.language = d.language
                      AND hr.wiki_page_id = d.wiki_page_id
                      AND (hr.rev_date::date - m.opening_date::date) <= d.movie_time_day
                ), 0) AS R,
                m.opening_theaters,
                m.opening_day_gross_usd,
                m.opening_weekend_revenue_usd
            FROM days d
            JOIN matched m
              ON m.movie_id = d.movie_id
             AND m.language = d.language
             AND m.wiki_page_id = d.wiki_page_id;
        """
    )
    movie_identity.ensure_movie_identity_schema(conn)


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def clean_movie_title(title: str) -> str:
    title = re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()
    return title or title


def canonical_wiki_url(language: str, page_title: str) -> str:
    quoted = urllib.parse.quote(page_title.replace(" ", "_"))
    return f"https://{language}.wikipedia.org/wiki/{quoted}"


def mediawiki_api_url(language: str, params: dict[str, Any]) -> str:
    query = urllib.parse.urlencode(params)
    return f"https://{language}.wikipedia.org/w/api.php?{query}"


def pageviews_url(language: str, page_title: str, start: dt.date, end: dt.date) -> str:
    quoted = urllib.parse.quote(page_title.replace(" ", "_"), safe="")
    return (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"{language}.wikipedia/all-access/user/{quoted}/daily/"
        f"{start:%Y%m%d}/{end:%Y%m%d}"
    )


def latest_wikimedia_pageview_date() -> dt.date:
    """Return the newest daily pageview date expected to be complete."""
    return dt.datetime.now(dt.UTC).date() - dt.timedelta(days=1)


def cap_activity_window(
    *,
    start_date: dt.date,
    end_date: dt.date,
    latest_available_date: dt.date,
) -> tuple[dt.date, dt.date, bool] | None:
    if start_date > latest_available_date:
        return None
    capped_end = min(end_date, latest_available_date)
    return start_date, capped_end, capped_end == end_date


def select_candidate_movies(
    conn: Any,
    *,
    release_year: int | None,
    min_opening_theaters: int | None,
    movie_limit: int | None,
) -> list[CandidateMovie]:
    selects = [
        """
        SELECT
            m.movie_id,
            m.movie_url,
            m.title,
            m.release_year,
            bof.release_run_id,
            bof.opening_date,
            bof.opening_theaters,
            bof.opening_day_gross_usd,
            bof.opening_weekend_revenue_usd,
            0 AS source_priority
        FROM movies m
        JOIN box_office_opening_features bof ON bof.movie_id = m.movie_id
        WHERE 1 = 1
        """
    ]
    params: list[Any] = []
    if relation_exists(conn, "boxofficepro_weekend_predictions"):
        selects.append(
            """
            SELECT DISTINCT
                m.movie_id,
                m.movie_url,
                COALESCE(m.title, p.source_movie_title) AS title,
                COALESCE(m.release_year, EXTRACT(YEAR FROM p.target_start_date)::integer) AS release_year,
                0 AS release_run_id,
                p.target_start_date::text AS opening_date,
                NULL::integer AS opening_theaters,
                NULL::integer AS opening_day_gross_usd,
                NULL::integer AS opening_weekend_revenue_usd,
                1 AS source_priority
            FROM boxofficepro_weekend_predictions p
            JOIN movies m ON m.movie_id = p.movie_id
            WHERE p.forecast_metric = 'domestic_opening_weekend'
              AND p.target_start_date IS NOT NULL
              AND p.movie_id IS NOT NULL
              AND p.source_movie_title NOT ILIKE '%%untitled%%'
              AND p.source_movie_title NOT ILIKE '%%re-release%%'
              AND COALESCE(m.title, p.source_movie_title) NOT ILIKE '%%re-release%%'
            """
        )
    if relation_exists(conn, "the_numbers_release_schedule"):
        selects.append(
            """
            SELECT DISTINCT
                m.movie_id,
                m.movie_url,
                COALESCE(m.title, tn.title) AS title,
                COALESCE(m.release_year, EXTRACT(YEAR FROM tn.release_date)::integer) AS release_year,
                0 AS release_run_id,
                tn.release_date::text AS opening_date,
                NULL::integer AS opening_theaters,
                NULL::integer AS opening_day_gross_usd,
                NULL::integer AS opening_weekend_revenue_usd,
                2 AS source_priority
            FROM the_numbers_release_schedule tn
            LEFT JOIN movie_source_ids msi
              ON msi.source = 'the_numbers'
             AND msi.source_movie_id = tn.movie_url
            JOIN movies m ON m.movie_id = msi.movie_id OR m.movie_url = tn.movie_url
            WHERE tn.release_date IS NOT NULL
              AND tn.title NOT ILIKE '%%untitled%%'
              AND tn.title NOT ILIKE '%%re-release%%'
              AND COALESCE(tn.release_pattern, '') NOT ILIKE '%%re-release%%'
            """
        )
    sql = f"""
        SELECT
            movie_id,
            movie_url,
            title,
            release_year,
            release_run_id,
            opening_date,
            opening_theaters,
            opening_day_gross_usd,
            opening_weekend_revenue_usd
        FROM (
            SELECT DISTINCT ON (movie_id)
                movie_id,
                movie_url,
                title,
                release_year,
                release_run_id,
                opening_date,
                opening_theaters,
                opening_day_gross_usd,
                opening_weekend_revenue_usd,
                source_priority
            FROM (
                {" UNION ALL ".join(selects)}
            ) candidate_sources
            WHERE 1 = 1
    """
    if release_year is not None:
        sql += " AND release_year = %s"
        params.append(release_year)
    if min_opening_theaters is not None:
        sql += " AND COALESCE(opening_theaters, 0) >= %s"
        params.append(min_opening_theaters)
    sql += """
            ORDER BY movie_id, source_priority, opening_date, title
        ) deduped
        ORDER BY opening_date, title
    """
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
            release_run_id=int(row[4]),
            opening_date=row[5],
            opening_theaters=row[6],
            opening_day_gross_usd=row[7],
            opening_weekend_revenue_usd=row[8],
        )
        for row in rows
    ]


def relation_exists(conn: Any, relation_name: str) -> bool:
    row = conn.execute("SELECT to_regclass(%s)", (relation_name,)).fetchone()
    return bool(row and row[0])


def state_is_completed(
    conn: Any,
    *,
    movie_id: int,
    language: str,
    day_start: int,
    day_end: int,
) -> bool:
    row = conn.execute(
        """
        SELECT status
        FROM wiki_ingest_state
        WHERE movie_id = %s
          AND language = %s
          AND day_start = %s
          AND day_end = %s
        """,
        (movie_id, language, day_start, day_end),
    ).fetchone()
    return row is not None and row[0] == "completed"


def reset_failed_states(conn: Any) -> None:
    conn.execute("DELETE FROM wiki_ingest_state WHERE status = 'failed'")


def mark_running_states_interrupted(conn: Any) -> int:
    cursor = conn.execute(
        """
        UPDATE wiki_ingest_state
        SET status = 'failed',
            stage = 'interrupted',
            updated_at = %s,
            last_error = 'Interrupted before completion; safe to resume'
        WHERE status = 'running'
        """,
        (utc_now(),),
    )
    return int(getattr(cursor, "rowcount", 0))


def upsert_state(
    conn: Any,
    *,
    movie_id: int,
    language: str,
    day_start: int,
    day_end: int,
    stage: str,
    status: str,
    last_error: str | None = None,
    pageviews_rows: int | None = None,
    revision_rows: int | None = None,
) -> None:
    now = utc_now()
    completed_at = now if status == "completed" else None
    conn.execute(
        """
        INSERT INTO wiki_ingest_state (
            movie_id, language, day_start, day_end, stage, status, started_at,
            updated_at, completed_at, attempt_count, last_error,
            pageviews_rows, revision_rows
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 1, %s, COALESCE(%s, 0), COALESCE(%s, 0))
        ON CONFLICT(movie_id, language, day_start, day_end) DO UPDATE SET
            stage = excluded.stage,
            status = excluded.status,
            updated_at = excluded.updated_at,
            completed_at = excluded.completed_at,
            attempt_count = wiki_ingest_state.attempt_count + CASE
                WHEN excluded.status = 'running' AND wiki_ingest_state.status != 'running'
                THEN 1 ELSE 0 END,
            last_error = excluded.last_error,
            pageviews_rows = COALESCE(excluded.pageviews_rows, wiki_ingest_state.pageviews_rows),
            revision_rows = COALESCE(excluded.revision_rows, wiki_ingest_state.revision_rows)
        """,
        (
            movie_id,
            language,
            day_start,
            day_end,
            stage,
            status,
            now,
            now,
            completed_at,
            last_error,
            pageviews_rows,
            revision_rows,
        ),
    )


def insert_issue(
    conn: Any,
    *,
    issue_source: str,
    issue_type: str,
    movie: CandidateMovie,
    wiki_page_id: int | None,
    details: str,
) -> None:
    conn.execute(
        insert_ignore_sql(
            "wiki_import_issues",
            ["issue_source", "issue_type", "movie_id", "movie_url", "wiki_page_id", "details"],
        ),
        (issue_source, issue_type, movie.movie_id, movie.movie_url, wiki_page_id, details),
    )


def match_wikipedia_page(
    fetcher: JsonFetcher,
    *,
    movie: CandidateMovie,
    language: str,
) -> WikiMatch:
    base_title = clean_movie_title(movie.title)
    year = movie.release_year
    queries = []
    if year is not None:
        queries.extend([f"{base_title} {year} film", f"{base_title} ({year} film)"])
    queries.append(base_title)

    seen: set[str] = set()
    for query in queries:
        if query in seen:
            continue
        seen.add(query)
        url = mediawiki_api_url(
            language,
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 5,
                "format": "json",
            },
        )
        data, _cache_path, _fetched = fetcher.get_json(url)
        results = data.get("query", {}).get("search", [])
        for rank, result in enumerate(results, start=1):
            if result.get("ns") != 0:
                continue
            page_title = str(result.get("title", ""))
            snippet = strip_markup(str(result.get("snippet", ""))).lower()
            combined = f"{page_title} {snippet}".lower()
            if year is None or str(year) in combined or "film" in combined or rank == 1:
                return WikiMatch(
                    status="matched",
                    method="mediawiki_search",
                    query=query,
                    rank=rank,
                    score=float(result.get("size", 0) or 0),
                    page_id=int(result["pageid"]),
                    page_title=page_title,
                )
    return WikiMatch(
        status="not_found",
        method="mediawiki_search",
        query=queries[-1] if queries else None,
        rank=None,
        score=None,
        page_id=None,
        page_title=None,
        notes="No namespace-0 MediaWiki search result accepted",
    )


def strip_markup(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value)).strip()


def upsert_wiki_match(
    conn: Any,
    *,
    movie: CandidateMovie,
    language: str,
    match: WikiMatch,
) -> None:
    now = utc_now()
    if match.page_id is not None and match.page_title:
        conn.execute(
            """
            INSERT INTO wiki_pages (
                language, wiki_page_id, page_title, canonical_url, first_seen_at, last_seen_at
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT(language, wiki_page_id) DO UPDATE SET
                page_title = excluded.page_title,
                canonical_url = excluded.canonical_url,
                last_seen_at = excluded.last_seen_at
            """,
            (
                language,
                match.page_id,
                match.page_title,
                canonical_wiki_url(language, match.page_title),
                now,
                now,
            ),
        )
    if match.page_id is not None and match.status in {"matched", "manual_override"}:
        conn.execute(
            """
            DELETE FROM movie_source_ids
            WHERE source = %s
              AND movie_id = %s
              AND source_movie_id LIKE %s
              AND source_movie_id <> %s
              AND match_status != 'manual_override'
            """,
            (
                movie_identity.SOURCE_WIKIPEDIA,
                movie.movie_id,
                f"{language}:%",
                f"{language}:{match.page_id}",
            ),
        )
        movie_identity.upsert_movie_source_id(
            conn,
            movie_id=movie.movie_id,
            source=movie_identity.SOURCE_WIKIPEDIA,
            source_movie_id=f"{language}:{match.page_id}",
            source_title=match.page_title,
            match_status=match.status,
            match_method=match.method,
            match_score=match.score,
        )


def fetch_pageviews(
    fetcher: JsonFetcher,
    *,
    language: str,
    wiki_page_id: int,
    page_title: str,
    start_date: dt.date,
    end_date: dt.date,
) -> tuple[list[PageviewRow], Path]:
    url = pageviews_url(language, page_title, start_date, end_date)
    data, cache_path, _fetched = fetcher.get_json(url)
    rows: list[PageviewRow] = []
    for item in data.get("items", []):
        timestamp = str(item["timestamp"])
        view_date = f"{timestamp[0:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
        rows.append(
            PageviewRow(
                language=language,
                wiki_page_id=wiki_page_id,
                view_date=view_date,
                views=int(item.get("views", 0)),
                access=str(item.get("access", "all-access")),
                agent=str(item.get("agent", "user")),
                source_url=url,
            )
        )
    return rows, cache_path


def fetch_revisions(
    fetcher: JsonFetcher,
    *,
    language: str,
    wiki_page_id: int,
    page_title: str,
    start_date: dt.date,
    end_date: dt.date,
) -> tuple[list[RevisionRow], Path | None]:
    rows: list[RevisionRow] = []
    cont: dict[str, Any] = {}
    last_cache_path: Path | None = None
    while True:
        params: dict[str, Any] = {
            "action": "query",
            "prop": "revisions",
            "titles": page_title,
            "rvprop": "ids|timestamp|user|userid|flags",
            "rvlimit": "max",
            "rvdir": "newer",
            "rvstart": f"{start_date.isoformat()}T00:00:00Z",
            "rvend": f"{end_date.isoformat()}T23:59:59Z",
            "format": "json",
        }
        params.update(cont)
        url = mediawiki_api_url(language, params)
        data, cache_path, _fetched = fetcher.get_json(url)
        last_cache_path = cache_path
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            for rev in page.get("revisions", []):
                timestamp = str(rev["timestamp"])
                user_name = rev.get("user")
                user_id = rev.get("userid")
                user_key = stable_user_key(user_name, user_id)
                rows.append(
                    RevisionRow(
                        language=language,
                        wiki_page_id=wiki_page_id,
                        rev_id=int(rev["revid"]),
                        rev_timestamp=timestamp,
                        rev_date=timestamp[:10],
                        user_name=str(user_name) if user_name is not None else None,
                        user_id=int(user_id) if user_id is not None else None,
                        user_key=user_key,
                        is_bot=1 if is_probable_bot(str(user_name or "")) else 0,
                        is_minor=1 if "minor" in rev else 0,
                        parent_id=int(rev["parentid"]) if rev.get("parentid") is not None else None,
                        source_url=url,
                    )
                )
        if "continue" not in data:
            break
        cont = dict(data["continue"])
    return rows, last_cache_path


def stable_user_key(user_name: Any, user_id: Any) -> str:
    if user_id is not None:
        return f"id:{user_id}"
    if user_name is None:
        return "anonymous:unknown"
    return f"name:{str(user_name).strip().lower()}"


def is_probable_bot(user_name: str) -> bool:
    lowered = user_name.lower()
    return "bot" in lowered or lowered.endswith("script")


def insert_pageviews(
    conn: Any,
    rows: list[PageviewRow],
    *,
    fetched_at: str,
    raw_cache_path: Path,
) -> None:
    conn.executemany(
        """
        INSERT INTO wiki_pageviews_daily (
            language, wiki_page_id, view_date, views, access, agent,
            source_url, fetched_at, raw_cache_path
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(language, wiki_page_id, view_date, access, agent) DO UPDATE SET
            views = excluded.views,
            source_url = excluded.source_url,
            fetched_at = excluded.fetched_at,
            raw_cache_path = excluded.raw_cache_path
        """,
        [
            (
                row.language,
                row.wiki_page_id,
                row.view_date,
                row.views,
                row.access,
                row.agent,
                row.source_url,
                fetched_at,
                str(raw_cache_path),
            )
            for row in rows
        ],
    )


def insert_revisions(
    conn: Any,
    rows: list[RevisionRow],
    *,
    fetched_at: str,
    raw_cache_path: Path,
) -> None:
    conn.executemany(
        """
        INSERT INTO wiki_revisions (
            language, wiki_page_id, rev_id, rev_timestamp, rev_date, user_name,
            user_id, user_key, is_bot, is_minor, parent_id, source_url,
            fetched_at, raw_cache_path
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(language, wiki_page_id, rev_id) DO UPDATE SET
            rev_timestamp = excluded.rev_timestamp,
            rev_date = excluded.rev_date,
            user_name = excluded.user_name,
            user_id = excluded.user_id,
            user_key = excluded.user_key,
            is_bot = excluded.is_bot,
            is_minor = excluded.is_minor,
            parent_id = excluded.parent_id,
            source_url = excluded.source_url,
            fetched_at = excluded.fetched_at,
            raw_cache_path = excluded.raw_cache_path
        """,
        [
            (
                row.language,
                row.wiki_page_id,
                row.rev_id,
                row.rev_timestamp,
                row.rev_date,
                row.user_name,
                row.user_id,
                row.user_key,
                row.is_bot,
                row.is_minor,
                row.parent_id,
                row.source_url,
                fetched_at,
                str(raw_cache_path),
            )
            for row in rows
        ],
    )


def ingest_movie(
    conn: Any,
    fetcher: JsonFetcher,
    *,
    movie: CandidateMovie,
    language: str,
    day_start: int,
    day_end: int,
    issue_source: str,
    latest_available_date: dt.date | None = None,
) -> tuple[int, int]:
    upsert_state(
        conn,
        movie_id=movie.movie_id,
        language=language,
        day_start=day_start,
        day_end=day_end,
        stage="matching",
        status="running",
    )
    conn.commit()
    match = match_wikipedia_page(fetcher, movie=movie, language=language)
    upsert_wiki_match(conn, movie=movie, language=language, match=match)
    conn.commit()
    if match.status != "matched" or match.page_id is None or match.page_title is None:
        insert_issue(
            conn,
            issue_source=issue_source,
            issue_type="wiki_match_failed",
            movie=movie,
            wiki_page_id=None,
            details=match.notes or f"No Wikipedia page matched for {movie.title}",
        )
        upsert_state(
            conn,
            movie_id=movie.movie_id,
            language=language,
            day_start=day_start,
            day_end=day_end,
            stage="matching",
            status="failed",
            last_error=match.notes or "No Wikipedia page matched",
        )
        return 0, 0

    opening_date = dt.date.fromisoformat(movie.opening_date)
    requested_start_date = opening_date + dt.timedelta(days=day_start)
    requested_end_date = opening_date + dt.timedelta(days=day_end)
    latest_available = latest_available_date or latest_wikimedia_pageview_date()
    capped_window = cap_activity_window(
        start_date=requested_start_date,
        end_date=requested_end_date,
        latest_available_date=latest_available,
    )
    if capped_window is None:
        details = (
            "Requested Wikipedia activity window starts after latest available "
            f"pageview date {latest_available.isoformat()}"
        )
        insert_issue(
            conn,
            issue_source=issue_source,
            issue_type="wiki_activity_window_unavailable",
            movie=movie,
            wiki_page_id=match.page_id,
            details=details,
        )
        upsert_state(
            conn,
            movie_id=movie.movie_id,
            language=language,
            day_start=day_start,
            day_end=day_end,
            stage="waiting_for_activity_window",
            status="deferred",
            last_error=details,
        )
        return 0, 0
    start_date, end_date, fetched_complete_window = capped_window

    upsert_state(
        conn,
        movie_id=movie.movie_id,
        language=language,
        day_start=day_start,
        day_end=day_end,
        stage="pageviews",
        status="running",
    )
    conn.commit()
    pageviews, pageview_cache_path = fetch_pageviews(
        fetcher,
        language=language,
        wiki_page_id=match.page_id,
        page_title=match.page_title,
        start_date=start_date,
        end_date=end_date,
    )
    fetched_at = utc_now()
    insert_pageviews(conn, pageviews, fetched_at=fetched_at, raw_cache_path=pageview_cache_path)
    conn.commit()

    upsert_state(
        conn,
        movie_id=movie.movie_id,
        language=language,
        day_start=day_start,
        day_end=day_end,
        stage="revisions",
        status="running",
        pageviews_rows=len(pageviews),
    )
    conn.commit()
    revisions, revision_cache_path = fetch_revisions(
        fetcher,
        language=language,
        wiki_page_id=match.page_id,
        page_title=match.page_title,
        start_date=start_date,
        end_date=end_date,
    )
    if revision_cache_path is None:
        revision_cache_path = pageview_cache_path
    fetched_at = utc_now()
    insert_revisions(conn, revisions, fetched_at=fetched_at, raw_cache_path=revision_cache_path)
    final_stage = "completed" if fetched_complete_window else "activity_partial"
    final_status = "completed" if fetched_complete_window else "partial"
    final_error = None
    if not fetched_complete_window:
        final_error = (
            "Fetched through latest available Wikimedia pageview date "
            f"{latest_available.isoformat()}; requested through {requested_end_date.isoformat()}"
        )
    upsert_state(
        conn,
        movie_id=movie.movie_id,
        language=language,
        day_start=day_start,
        day_end=day_end,
        stage=final_stage,
        status=final_status,
        last_error=final_error,
        pageviews_rows=len(pageviews),
        revision_rows=len(revisions),
    )
    return len(pageviews), len(revisions)


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    conn = connect_database(args.database_url)
    try:
        initialize_wikipedia_database(conn)
        interrupted = mark_running_states_interrupted(conn)
        if interrupted:
            print(f"Marked {interrupted} interrupted running ingest states as failed.", file=sys.stderr)
        if args.reset_failed:
            reset_failed_states(conn)
        conn.commit()
        candidates = select_candidate_movies(
            conn,
            release_year=args.release_year,
            min_opening_theaters=args.min_opening_theaters,
            movie_limit=args.movie_limit,
        )
        if args.dry_run:
            for movie in candidates:
                print(
                    f"{movie.movie_id}\t{movie.opening_date}\t"
                    f"{movie.opening_theaters}\t{movie.title}\t{movie.movie_url}"
                )
            print(f"Candidate movies: {len(candidates)}", file=sys.stderr)
            return 0
        conn.commit()

        fetcher = JsonFetcher(
            args.cache_dir,
            refresh=args.refresh,
            offline=args.offline,
            delay_seconds=args.delay_seconds,
            user_agent=args.user_agent,
        )
        processed = 0
        skipped = 0
        pageview_rows = 0
        revision_rows = 0
        for index, movie in enumerate(candidates, start=1):
            if (
                not args.refresh
                and state_is_completed(
                    conn,
                    movie_id=movie.movie_id,
                    language=args.language,
                    day_start=args.day_start,
                    day_end=args.day_end,
                )
            ):
                print(f"Skipping completed {index}/{len(candidates)} {movie.title}", file=sys.stderr)
                skipped += 1
                continue
            print(f"Ingesting {index}/{len(candidates)} {movie.title}", file=sys.stderr)
            try:
                pv_count, rev_count = ingest_movie(
                    conn,
                    fetcher,
                    movie=movie,
                    language=args.language,
                    day_start=args.day_start,
                    day_end=args.day_end,
                    issue_source=args.issue_source,
                )
                pageview_rows += pv_count
                revision_rows += rev_count
                processed += 1
                conn.commit()
            except Exception as exc:  # noqa: BLE001 - persist state and continue.
                upsert_state(
                    conn,
                    movie_id=movie.movie_id,
                    language=args.language,
                    day_start=args.day_start,
                    day_end=args.day_end,
                    stage="error",
                    status="failed",
                    last_error=str(exc),
                )
                insert_issue(
                    conn,
                    issue_source=args.issue_source,
                    issue_type="ingest_failed",
                    movie=movie,
                    wiki_page_id=None,
                    details=str(exc),
                )
                conn.commit()
                if args.fail_fast:
                    raise
            if args.batch_size and processed % args.batch_size == 0:
                conn.commit()
        conn.commit()
        print(
            f"Processed={processed} skipped={skipped} "
            f"pageviews={pageview_rows} revisions={revision_rows}",
            file=sys.stderr,
        )
        return 0
    finally:
        conn.close()


def validate_args(args: argparse.Namespace) -> None:
    if args.day_end < args.day_start:
        raise SystemExit("--day-end must be greater than or equal to --day-start")
    if args.delay_seconds < 0:
        raise SystemExit("--delay-seconds must be non-negative")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.movie_limit is not None and args.movie_limit < 1:
        raise SystemExit("--movie-limit must be positive")
    if args.min_opening_theaters is not None and args.min_opening_theaters < 0:
        raise SystemExit("--min-opening-theaters must be non-negative")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_database_arg(parser)
    add_cache_args(parser, default_cache_dir=DEFAULT_CACHE_DIR, include_dry_run=False)
    parser.add_argument("--language", default="en")
    parser.add_argument("--day-start", type=int, default=-500)
    parser.add_argument("--day-end", type=int, default=100)
    parser.add_argument("--delay-seconds", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--movie-limit", type=int)
    parser.add_argument("--release-year", type=int)
    parser.add_argument("--min-opening-theaters", type=int)
    parser.add_argument("--reset-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--issue-source", default="wikimedia_activity")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
