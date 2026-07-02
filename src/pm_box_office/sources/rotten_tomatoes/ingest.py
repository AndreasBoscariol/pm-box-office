#!/usr/bin/env python3
"""Import Rotten Tomatoes critic reviews for movies already present in the DB.

The ingest is cache-first, single-threaded, and conservative. It avoids the RT
search page, derives candidate vanity URLs from local movie titles, and only
accepts high-confidence title/year matches.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
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

from pm_box_office.db.connection import connect_database, database_url_from_env, insert_ignore_sql


BASE_URL = "https://www.rottentomatoes.com"
DEFAULT_CACHE_DIR = Path("data/raw/rotten_tomatoes")
DEFAULT_USER_AGENT = "pm-box-office-rotten-tomatoes-ingest/0.1 (+research; cache-first)"
PARSER_VERSION = "rt_critic_reviews_v1"
TRANSIENT_STATUSES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class CandidateMovie:
    movie_id: int
    title: str
    release_year: int | None
    release_date: dt.date | None
    movie_url: str | None


@dataclass(frozen=True)
class RottenTomatoesMedia:
    ems_id: str
    vanity_slug: str
    canonical_url: str
    title: str
    release_year: int | None
    media_type: str
    tomatometer_score: int | None
    tomatometer_sentiment: str | None
    certified_fresh: bool | None
    critic_review_count: int | None
    top_critic_review_count: int | None
    source_url: str


@dataclass(frozen=True)
class MovieMatch:
    movie_id: int
    ems_id: str | None
    match_status: str
    match_method: str
    match_score: float | None
    notes: str | None


@dataclass(frozen=True)
class CriticReview:
    review_key: str
    ems_id: str
    movie_id: int
    critic_name: str | None
    critic_id: str | None
    critic_url: str | None
    publication_name: str | None
    publication_url: str | None
    is_top_critic: bool
    is_tomatometer_approved: bool
    review_type: str
    sentiment: str | None
    fresh_rotten: str | None
    original_score: str | None
    review_quote: str | None
    publication_review_url: str | None
    review_date: dt.date | None
    raw_json: dict[str, Any]


class TextFetcher:
    def __init__(
        self,
        cache_dir: Path,
        *,
        refresh: bool = False,
        offline: bool = False,
        delay_seconds: float = 5.0,
        timeout_seconds: float = 30.0,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self.cache_dir = cache_dir
        self.refresh = refresh
        self.offline = offline
        self.delay_seconds = delay_seconds
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent
        self._last_request_at = 0.0
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(self, url: str, *, suffix: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.{suffix}"

    def get_text(self, url: str, *, suffix: str = "html") -> tuple[str, Path, bool]:
        cache_path = self.cache_path(url, suffix=suffix)
        if cache_path.exists() and not self.refresh:
            return cache_path.read_text(encoding="utf-8"), cache_path, False
        if self.offline:
            raise FileNotFoundError(f"Missing cached Rotten Tomatoes response for {url}: {cache_path}")
        body, fetched = self._fetch(url)
        cache_path.write_text(body, encoding="utf-8")
        return body, cache_path, fetched

    def get_json(self, url: str) -> tuple[dict[str, Any], Path, bool]:
        text, cache_path, fetched = self.get_text(url, suffix="json")
        return json.loads(text), cache_path, fetched

    def _fetch(self, url: str) -> tuple[str, bool]:
        last_error: BaseException | None = None
        for attempt in range(3):
            self._wait()
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json,text/html,application/xhtml+xml",
                    "User-Agent": self.user_agent,
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    body = response.read().decode("utf-8", errors="replace")
                self._last_request_at = time.monotonic()
                return body, True
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code == 404:
                    return "", True
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

        CREATE TABLE IF NOT EXISTS movies (
            movie_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            title TEXT NOT NULL,
            release_date DATE,
            movie_url TEXT,
            release_year INTEGER,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );

        ALTER TABLE movies
            ADD COLUMN IF NOT EXISTS movie_url TEXT,
            ADD COLUMN IF NOT EXISTS release_year INTEGER,
            ADD COLUMN IF NOT EXISTS release_date DATE,
            ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP;

        CREATE TABLE IF NOT EXISTS movie_source_ids (
            movie_id BIGINT REFERENCES movies(movie_id),
            source TEXT NOT NULL,
            source_movie_id TEXT NOT NULL,
            source_title TEXT,
            match_status TEXT NOT NULL DEFAULT 'unmatched',
            match_method TEXT,
            match_score DOUBLE PRECISION,
            matched_at TIMESTAMPTZ,
            PRIMARY KEY (source, source_movie_id)
        );

        ALTER TABLE movie_source_ids
            ADD COLUMN IF NOT EXISTS source_title TEXT,
            ADD COLUMN IF NOT EXISTS match_status TEXT DEFAULT 'unmatched',
            ADD COLUMN IF NOT EXISTS match_method TEXT,
            ADD COLUMN IF NOT EXISTS match_score DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS matched_at TIMESTAMPTZ;

        CREATE TABLE IF NOT EXISTS rotten_tomatoes_media (
            ems_id TEXT PRIMARY KEY,
            vanity_slug TEXT UNIQUE,
            canonical_url TEXT,
            title TEXT,
            release_year INTEGER,
            media_type TEXT,
            tomatometer_score INTEGER,
            tomatometer_sentiment TEXT,
            certified_fresh BOOLEAN,
            critic_review_count INTEGER,
            top_critic_review_count INTEGER,
            source_url TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            raw_cache_path TEXT NOT NULL,
            parser_version TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS movie_rotten_tomatoes_media (
            movie_id BIGINT NOT NULL REFERENCES movies(movie_id),
            ems_id TEXT REFERENCES rotten_tomatoes_media(ems_id),
            match_status TEXT NOT NULL,
            match_method TEXT NOT NULL,
            match_score DOUBLE PRECISION,
            matched_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            notes TEXT,
            UNIQUE(movie_id),
            UNIQUE(ems_id)
        );

        CREATE TABLE IF NOT EXISTS rotten_tomatoes_movie_match_overrides (
            override_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            movie_id BIGINT REFERENCES movies(movie_id),
            normalized_title TEXT,
            release_year INTEGER,
            ems_id TEXT,
            vanity_slug TEXT,
            notes TEXT,
            active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP::text)
        );

        CREATE UNIQUE INDEX IF NOT EXISTS uq_rt_overrides_movie_active
            ON rotten_tomatoes_movie_match_overrides(movie_id) WHERE active AND movie_id IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_rt_overrides_title_year
            ON rotten_tomatoes_movie_match_overrides(normalized_title, release_year) WHERE active;

        CREATE TABLE IF NOT EXISTS rotten_tomatoes_reviews (
            review_key TEXT PRIMARY KEY,
            ems_id TEXT NOT NULL REFERENCES rotten_tomatoes_media(ems_id),
            movie_id BIGINT NOT NULL REFERENCES movies(movie_id),
            critic_name TEXT,
            critic_id TEXT,
            critic_url TEXT,
            publication_name TEXT,
            publication_url TEXT,
            is_top_critic BOOLEAN NOT NULL DEFAULT FALSE,
            is_tomatometer_approved BOOLEAN NOT NULL DEFAULT FALSE,
            review_type TEXT NOT NULL,
            sentiment TEXT,
            fresh_rotten TEXT,
            original_score TEXT,
            review_quote TEXT,
            publication_review_url TEXT,
            review_date DATE,
            raw_json JSONB NOT NULL,
            fetched_at TEXT NOT NULL,
            raw_cache_path TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS rotten_tomatoes_ingest_state (
            movie_id BIGINT NOT NULL REFERENCES movies(movie_id),
            source TEXT NOT NULL DEFAULT 'rotten_tomatoes',
            stage TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            UNIQUE(movie_id, source)
        );

        CREATE TABLE IF NOT EXISTS rotten_tomatoes_ingest_issues (
            issue_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            issue_source TEXT NOT NULL,
            issue_type TEXT NOT NULL,
            movie_id BIGINT,
            ems_id TEXT,
            source_url TEXT,
            details TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP::text),
            UNIQUE(issue_source, issue_type, movie_id, ems_id, source_url, details)
        );

        CREATE INDEX IF NOT EXISTS idx_rt_reviews_movie_date
            ON rotten_tomatoes_reviews(movie_id, review_date);
        CREATE INDEX IF NOT EXISTS idx_rt_reviews_critic
            ON rotten_tomatoes_reviews(critic_id, critic_name);
        CREATE INDEX IF NOT EXISTS idx_rt_ingest_state_status
            ON rotten_tomatoes_ingest_state(status, source);

        DROP VIEW IF EXISTS analytics.rotten_tomatoes_movie_review_features_v1;
        DROP VIEW IF EXISTS analytics.rotten_tomatoes_critic_history_v1;

        CREATE VIEW analytics.rotten_tomatoes_movie_review_features_v1 AS
        SELECT
            m.movie_id,
            COUNT(r.review_key)::integer AS rt_critic_review_count,
            COUNT(*) FILTER (WHERE r.is_top_critic)::integer AS rt_top_critic_review_count,
            COUNT(*) FILTER (WHERE r.fresh_rotten = 'fresh')::integer AS rt_fresh_review_count,
            COUNT(*) FILTER (WHERE r.fresh_rotten = 'rotten')::integer AS rt_rotten_review_count,
            COUNT(*) FILTER (WHERE r.is_top_critic AND r.fresh_rotten = 'fresh')::integer AS rt_top_fresh_review_count,
            COUNT(*) FILTER (WHERE r.is_top_critic AND r.fresh_rotten = 'rotten')::integer AS rt_top_rotten_review_count,
            CASE
                WHEN COUNT(r.review_key) > 0
                THEN COUNT(*) FILTER (WHERE r.fresh_rotten = 'fresh')::double precision / COUNT(r.review_key)
                ELSE NULL
            END AS rt_fresh_share,
            MIN(r.review_date) AS rt_first_review_date,
            MAX(r.review_date) AS rt_last_review_date,
            COUNT(*) FILTER (WHERE r.review_date IS NOT NULL AND m.release_date IS NOT NULL AND r.review_date < m.release_date)::integer
                AS rt_prerelease_review_count,
            COUNT(*) FILTER (WHERE r.is_top_critic AND r.review_date IS NOT NULL AND m.release_date IS NOT NULL AND r.review_date < m.release_date)::integer
                AS rt_top_prerelease_review_count
        FROM movies m
        LEFT JOIN rotten_tomatoes_reviews r ON r.movie_id = m.movie_id
        GROUP BY m.movie_id, m.release_date;

        CREATE VIEW analytics.rotten_tomatoes_critic_history_v1 AS
        SELECT
            COALESCE(r.critic_id, r.critic_name) AS critic_key,
            MAX(r.critic_name) AS critic_name,
            MAX(r.critic_url) AS critic_url,
            COUNT(DISTINCT r.movie_id)::integer AS reviewed_movie_count,
            COUNT(r.review_key)::integer AS review_count,
            COUNT(*) FILTER (WHERE r.fresh_rotten = 'fresh')::integer AS fresh_review_count,
            COUNT(*) FILTER (WHERE r.fresh_rotten = 'rotten')::integer AS rotten_review_count,
            COUNT(*) FILTER (WHERE r.is_top_critic)::integer AS top_critic_review_count,
            BOOL_OR(r.is_top_critic) AS ever_top_critic,
            MIN(r.review_date) AS first_review_date,
            MAX(r.review_date) AS last_review_date
        FROM rotten_tomatoes_reviews r
        GROUP BY COALESCE(r.critic_id, r.critic_name);
        """
    )


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", html.unescape(value).replace("\xa0", " ")).strip()


def normalize_title(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s*\(\d{4}\)\s*$", "", text)
    text = text.lower()
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def parse_optional_date(value: object) -> dt.date | None:
    if value in (None, ""):
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return dt.datetime.fromisoformat(text).date()
    except ValueError:
        try:
            return dt.date.fromisoformat(text[:10])
        except ValueError:
            return None


def parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "certified-fresh"}:
        return True
    if text in {"0", "false", "no", "rotten", "fresh"}:
        return False
    return None


def parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    match = re.search(r"-?\d+", str(value).replace(",", ""))
    return int(match.group(0)) if match else None


def slugify_rt_title(title: str, release_year: int | None = None) -> str:
    normalized = normalize_title(title)
    slug = normalized.replace(" ", "_")
    if release_year:
        return f"{slug}_{release_year}"
    return slug


def vanity_slug_candidates(movie: CandidateMovie) -> list[str]:
    titles = [movie.title]
    if movie.movie_url:
        tail = urllib.parse.unquote(movie.movie_url.rstrip("/").split("/")[-1])
        tail = re.sub(r"\(\d{4}(?:-[^)]+)?\)$", "", tail).replace("-", " ")
        titles.append(tail)
    candidates: list[str] = []
    for title in titles:
        for year in (movie.release_year, None):
            slug = slugify_rt_title(title, year)
            if slug and slug not in candidates:
                candidates.append(slug)
    return candidates


def movie_page_url(slug: str) -> str:
    return f"{BASE_URL}/m/{urllib.parse.quote(slug)}"


def review_api_url(ems_id: str, *, top_only: bool = False, after: str | None = None) -> str:
    query: dict[str, str] = {"type": "critic"}
    if top_only:
        query["topOnly"] = "true"
    if after:
        query["after"] = after
    return f"{BASE_URL}/napi/rtcf/v1/movies/{urllib.parse.quote(ems_id)}/reviews?{urllib.parse.urlencode(query)}"


def extract_json_assignment(html_text: str, assignment: str) -> dict[str, Any] | None:
    marker = f"{assignment} = "
    start = html_text.find(marker)
    if start < 0:
        return None
    index = start + len(marker)
    while index < len(html_text) and html_text[index].isspace():
        index += 1
    if index >= len(html_text) or html_text[index] != "{":
        return None
    end = balanced_json_end(html_text, index)
    if end is None:
        return None
    raw = html_text[index : end + 1].replace(":undefined", ":null")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def balanced_json_end(text: str, start: int) -> int | None:
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def parse_media_page(html_text: str, *, source_url: str) -> RottenTomatoesMedia | None:
    review_context = extract_json_assignment(html_text, "root.RottenTomatoes.context.review") or {}
    mps = extract_json_assignment(html_text, "window.mpscall") or {}
    ems_id = first_text(review_context.get("emsId"), mps.get("field[rtid]"), regex_group(html_text, r'"titleId":"([^"]+)"'))
    title = first_text(review_context.get("title"), mps.get("title"), mps.get("cag[movieshow]"))
    if not ems_id or not title:
        return None
    vanity_slug = source_url.rstrip("/").split("/m/")[-1].split("?")[0]
    media_type = first_text(review_context.get("mediaType"), "movie") or "movie"
    release_year = parse_int(first_text(mps.get("cag[release]"), regex_group(html_text, r'"releaseDate":"?(\d{4})')))
    score = parse_int(mps.get("cag[score]"))
    sentiment = first_text(mps.get("cag[fresh_rotten]"))
    certified = parse_bool(mps.get("cag[certified_fresh]"))
    return RottenTomatoesMedia(
        ems_id=ems_id,
        vanity_slug=vanity_slug,
        canonical_url=source_url.split("?")[0],
        title=clean_text(title),
        release_year=release_year,
        media_type=media_type,
        tomatometer_score=score,
        tomatometer_sentiment=sentiment,
        certified_fresh=certified,
        critic_review_count=None,
        top_critic_review_count=None,
        source_url=source_url,
    )


def regex_group(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1) if match else None


def first_text(*values: Any) -> str | None:
    for value in values:
        if value not in (None, ""):
            text = str(value)
            if text:
                return text
    return None


def score_media_match(movie: CandidateMovie, media: RottenTomatoesMedia) -> float:
    movie_title = normalize_title(movie.title)
    media_title = normalize_title(media.title)
    if not movie_title or not media_title:
        return 0.0
    if movie_title == media_title:
        title_score = 1.0
    elif movie_title in media_title or media_title in movie_title:
        title_score = 0.9
    else:
        return 0.0
    if movie.release_year and media.release_year:
        year_delta = abs(movie.release_year - media.release_year)
        if year_delta == 0:
            year_score = 1.0
        elif year_delta == 1:
            year_score = 0.85
        else:
            return 0.0
    else:
        year_score = 0.8
    return round((title_score * 0.75) + (year_score * 0.25), 4)


def parse_reviews_payload(payload: dict[str, Any], *, ems_id: str, movie_id: int) -> tuple[list[CriticReview], str | None]:
    reviews: list[CriticReview] = []
    for item in payload.get("reviews", []) or []:
        if not isinstance(item, dict):
            continue
        critic = item.get("critic") if isinstance(item.get("critic"), dict) else {}
        publication = item.get("publication") if isinstance(item.get("publication"), dict) else {}
        sentiment = first_text(item.get("scoreSentiment"), item.get("sentiment"))
        reviews.append(
            CriticReview(
                review_key=review_key(ems_id, item),
                ems_id=ems_id,
                movie_id=movie_id,
                critic_name=clean_text(first_text(critic.get("displayName"), item.get("criticName"))) or None,
                critic_id=first_text(critic.get("encryptedCriticId"), critic.get("vanity")),
                critic_url=first_text(critic.get("rottenTomatoesUrl"), critic.get("url")),
                publication_name=clean_text(first_text(publication.get("name"), item.get("publicationName"))) or None,
                publication_url=first_text(publication.get("editorialUrl"), publication.get("url")),
                is_top_critic=bool(critic.get("isTopCritic") or item.get("isTopReview")),
                is_tomatometer_approved=bool(critic.get("tomatometerApproved") or publication.get("tomatometerApproved")),
                review_type="critic",
                sentiment=sentiment,
                fresh_rotten=fresh_rotten_from_sentiment(sentiment),
                original_score=first_text(item.get("originalScore")),
                review_quote=clean_text(first_text(item.get("reviewQuote"), item.get("quote"))) or None,
                publication_review_url=first_text(item.get("publicationReviewUrl")),
                review_date=parse_optional_date(item.get("createDate") or item.get("reviewDate")),
                raw_json=item,
            )
        )
    page_info = payload.get("pageInfo") if isinstance(payload.get("pageInfo"), dict) else {}
    next_cursor = page_info.get("endCursor") if page_info.get("hasNextPage") else None
    return reviews, first_text(next_cursor)


def fresh_rotten_from_sentiment(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"positive", "fresh"}:
        return "fresh"
    if normalized in {"negative", "rotten"}:
        return "rotten"
    return None


def review_key(ems_id: str, item: dict[str, Any]) -> str:
    explicit = first_text(item.get("reviewId"), item.get("ratingId"), item.get("id"))
    if explicit:
        return f"{ems_id}:{explicit}"
    parts = [
        ems_id,
        first_text(item.get("createDate")) or "",
        clean_text(first_text(item.get("reviewQuote"))),
        json.dumps(item.get("critic", {}), sort_keys=True),
        json.dumps(item.get("publication", {}), sort_keys=True),
    ]
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return f"{ems_id}:sha256:{digest}"


def select_candidate_movies(
    conn: Any,
    *,
    release_year: int | None = None,
    movie_limit: int | None = None,
) -> list[CandidateMovie]:
    where = "WHERE (%s::integer IS NULL OR m.release_year = %s::integer OR EXTRACT(YEAR FROM m.release_date)::integer = %s::integer)"
    sql = f"""
        SELECT m.movie_id, m.title, m.release_year, m.release_date, m.movie_url
        FROM movies m
        {where}
        ORDER BY COALESCE(m.release_date, make_date(COALESCE(m.release_year, 9999), 1, 1)), m.title
    """
    params: list[Any] = [release_year, release_year, release_year]
    if movie_limit is not None:
        sql += " LIMIT %s"
        params.append(movie_limit)
    rows = conn.execute(sql, params).fetchall()
    return [
        CandidateMovie(
            movie_id=int(row[0]),
            title=str(row[1]),
            release_year=int(row[2]) if row[2] is not None else None,
            release_date=parse_optional_date(row[3]),
            movie_url=str(row[4]) if row[4] is not None else None,
        )
        for row in rows
    ]


def existing_match(conn: Any, movie: CandidateMovie) -> str | None:
    row = conn.execute(
        """
        SELECT source_movie_id
        FROM movie_source_ids
        WHERE source = 'rottentomatoes'
          AND movie_id = %s
          AND match_status IN ('matched', 'manual_override')
        LIMIT 1
        """,
        (movie.movie_id,),
    ).fetchone()
    return str(row[0]) if row else None


def override_for_movie(conn: Any, movie: CandidateMovie) -> tuple[str | None, str | None]:
    normalized = normalize_title(movie.title)
    row = conn.execute(
        """
        SELECT ems_id, vanity_slug
        FROM rotten_tomatoes_movie_match_overrides
        WHERE active
          AND (
            movie_id = %s
            OR (
                normalized_title = %s
                AND (%s::integer IS NULL OR release_year IS NULL OR release_year = %s::integer)
            )
          )
        ORDER BY movie_id NULLS LAST
        LIMIT 1
        """,
        (movie.movie_id, normalized, movie.release_year, movie.release_year),
    ).fetchone()
    if not row:
        return None, None
    return (str(row[0]) if row[0] else None, str(row[1]) if row[1] else None)


def match_movie(conn: Any, fetcher: TextFetcher, movie: CandidateMovie) -> tuple[MovieMatch, RottenTomatoesMedia | None, Path | None]:
    override_ems, override_slug = override_for_movie(conn, movie)
    existing_ems = existing_match(conn, movie)
    if override_ems and not override_slug:
        return MovieMatch(movie.movie_id, override_ems, "matched", "manual_override", 1.0, "Matched by EMS override"), None, None
    slugs = [override_slug] if override_slug else vanity_slug_candidates(movie)
    best: tuple[float, RottenTomatoesMedia, Path] | None = None
    ambiguous = False
    for slug in [s for s in slugs if s]:
        url = movie_page_url(slug)
        html_text, cache_path, _fetched = fetcher.get_text(url, suffix="html")
        if not html_text:
            continue
        media = parse_media_page(html_text, source_url=url)
        if media is None:
            continue
        score = score_media_match(movie, media)
        if override_slug:
            score = max(score, 1.0)
        if score >= 0.95:
            if best and best[1].ems_id != media.ems_id and score == best[0]:
                ambiguous = True
            elif best is None or score > best[0]:
                best = (score, media, cache_path)
    if best and not ambiguous:
        score, media, cache_path = best
        return MovieMatch(movie.movie_id, media.ems_id, "matched", "rt_vanity_probe", score, None), media, cache_path
    if existing_ems:
        return MovieMatch(movie.movie_id, existing_ems, "matched", "existing_source_id", 1.0, "Matched existing source id"), None, None
    status = "ambiguous" if ambiguous else "not_found"
    notes = "Multiple high-confidence RT vanity matches" if ambiguous else "No high-confidence RT vanity match"
    return MovieMatch(movie.movie_id, None, status, "rt_vanity_probe", 0.0, notes), None, None


def upsert_media(conn: Any, media: RottenTomatoesMedia, *, fetched_at: str, raw_cache_path: Path) -> None:
    conn.execute(
        """
        INSERT INTO rotten_tomatoes_media (
            ems_id, vanity_slug, canonical_url, title, release_year, media_type,
            tomatometer_score, tomatometer_sentiment, certified_fresh,
            critic_review_count, top_critic_review_count,
            source_url, fetched_at, raw_cache_path, parser_version
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(ems_id) DO UPDATE SET
            vanity_slug = COALESCE(excluded.vanity_slug, rotten_tomatoes_media.vanity_slug),
            canonical_url = COALESCE(excluded.canonical_url, rotten_tomatoes_media.canonical_url),
            title = COALESCE(excluded.title, rotten_tomatoes_media.title),
            release_year = COALESCE(excluded.release_year, rotten_tomatoes_media.release_year),
            media_type = COALESCE(excluded.media_type, rotten_tomatoes_media.media_type),
            tomatometer_score = COALESCE(excluded.tomatometer_score, rotten_tomatoes_media.tomatometer_score),
            tomatometer_sentiment = COALESCE(excluded.tomatometer_sentiment, rotten_tomatoes_media.tomatometer_sentiment),
            certified_fresh = COALESCE(excluded.certified_fresh, rotten_tomatoes_media.certified_fresh),
            critic_review_count = COALESCE(excluded.critic_review_count, rotten_tomatoes_media.critic_review_count),
            top_critic_review_count = COALESCE(excluded.top_critic_review_count, rotten_tomatoes_media.top_critic_review_count),
            source_url = excluded.source_url,
            fetched_at = excluded.fetched_at,
            raw_cache_path = excluded.raw_cache_path,
            parser_version = excluded.parser_version,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            media.ems_id,
            media.vanity_slug,
            media.canonical_url,
            media.title,
            media.release_year,
            media.media_type,
            media.tomatometer_score,
            media.tomatometer_sentiment,
            media.certified_fresh,
            media.critic_review_count,
            media.top_critic_review_count,
            media.source_url,
            fetched_at,
            str(raw_cache_path),
            PARSER_VERSION,
        ),
    )


def upsert_movie_match(conn: Any, match: MovieMatch) -> None:
    conn.execute(
        """
        INSERT INTO movie_rotten_tomatoes_media (
            movie_id, ems_id, match_status, match_method, match_score, matched_at, notes
        ) VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP, %s)
        ON CONFLICT(movie_id) DO UPDATE SET
            ems_id = excluded.ems_id,
            match_status = excluded.match_status,
            match_method = excluded.match_method,
            match_score = excluded.match_score,
            matched_at = excluded.matched_at,
            notes = excluded.notes
        """,
        (match.movie_id, match.ems_id, match.match_status, match.match_method, match.match_score, match.notes),
    )
    if match.ems_id and match.match_status in {"matched", "manual_override"}:
        conn.execute(
            """
            INSERT INTO movie_source_ids (
                movie_id, source, source_movie_id, source_title,
                match_status, match_method, match_score, matched_at
            )
            SELECT movie_id, 'rottentomatoes', ems_id, rtm.title,
                   match_status, match_method, match_score, CURRENT_TIMESTAMP
            FROM movie_rotten_tomatoes_media mrtm
            LEFT JOIN rotten_tomatoes_media rtm ON rtm.ems_id = mrtm.ems_id
            WHERE mrtm.movie_id = %s
            ON CONFLICT(source, source_movie_id) DO UPDATE SET
                movie_id = excluded.movie_id,
                source_title = excluded.source_title,
                match_status = excluded.match_status,
                match_method = excluded.match_method,
                match_score = excluded.match_score,
                matched_at = excluded.matched_at
            """,
            (match.movie_id,),
        )


def update_media_review_counts(conn: Any, ems_id: str) -> None:
    conn.execute(
        """
        UPDATE rotten_tomatoes_media media
        SET critic_review_count = counts.review_count,
            top_critic_review_count = counts.top_count,
            updated_at = CURRENT_TIMESTAMP
        FROM (
            SELECT ems_id,
                   COUNT(*)::integer AS review_count,
                   COUNT(*) FILTER (WHERE is_top_critic)::integer AS top_count
            FROM rotten_tomatoes_reviews
            WHERE ems_id = %s
            GROUP BY ems_id
        ) counts
        WHERE media.ems_id = counts.ems_id
        """,
        (ems_id,),
    )


def insert_reviews(conn: Any, reviews: list[CriticReview], *, fetched_at: str, raw_cache_path: Path) -> None:
    if not reviews:
        return
    conn.executemany(
        """
        INSERT INTO rotten_tomatoes_reviews (
            review_key, ems_id, movie_id, critic_name, critic_id, critic_url,
            publication_name, publication_url, is_top_critic, is_tomatometer_approved,
            review_type, sentiment, fresh_rotten, original_score, review_quote,
            publication_review_url, review_date, raw_json, fetched_at, raw_cache_path
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s
        )
        ON CONFLICT(review_key) DO UPDATE SET
            critic_name = excluded.critic_name,
            critic_id = excluded.critic_id,
            critic_url = excluded.critic_url,
            publication_name = excluded.publication_name,
            publication_url = excluded.publication_url,
            is_top_critic = excluded.is_top_critic,
            is_tomatometer_approved = excluded.is_tomatometer_approved,
            review_type = excluded.review_type,
            sentiment = excluded.sentiment,
            fresh_rotten = excluded.fresh_rotten,
            original_score = excluded.original_score,
            review_quote = excluded.review_quote,
            publication_review_url = excluded.publication_review_url,
            review_date = excluded.review_date,
            raw_json = excluded.raw_json,
            fetched_at = excluded.fetched_at,
            raw_cache_path = excluded.raw_cache_path,
            updated_at = CURRENT_TIMESTAMP
        """,
        [
            (
                review.review_key,
                review.ems_id,
                review.movie_id,
                review.critic_name,
                review.critic_id,
                review.critic_url,
                review.publication_name,
                review.publication_url,
                review.is_top_critic,
                review.is_tomatometer_approved,
                review.review_type,
                review.sentiment,
                review.fresh_rotten,
                review.original_score,
                review.review_quote,
                review.publication_review_url,
                review.review_date,
                json.dumps(review.raw_json, sort_keys=True),
                fetched_at,
                str(raw_cache_path),
            )
            for review in reviews
        ],
    )


def upsert_state(
    conn: Any,
    *,
    movie_id: int,
    stage: str,
    status: str,
    last_error: str | None = None,
) -> None:
    now = utc_now()
    conn.execute(
        """
        INSERT INTO rotten_tomatoes_ingest_state (
            movie_id, source, stage, status, started_at, updated_at, completed_at, attempt_count, last_error
        ) VALUES (
            %s, 'rotten_tomatoes', %s, %s, %s, %s,
            CASE WHEN %s IN ('completed', 'skipped') THEN %s ELSE NULL END,
            1, %s
        )
        ON CONFLICT(movie_id, source) DO UPDATE SET
            stage = excluded.stage,
            status = excluded.status,
            updated_at = excluded.updated_at,
            completed_at = excluded.completed_at,
            attempt_count = rotten_tomatoes_ingest_state.attempt_count + CASE
                WHEN excluded.status = 'running' AND rotten_tomatoes_ingest_state.status != 'running'
                THEN 1 ELSE 0 END,
            last_error = excluded.last_error
        """,
        (movie_id, stage, status, now, now, status, now, last_error),
    )


def state_is_completed(conn: Any, movie_id: int) -> bool:
    row = conn.execute(
        """
        SELECT status FROM rotten_tomatoes_ingest_state
        WHERE movie_id = %s AND source = 'rotten_tomatoes'
        """,
        (movie_id,),
    ).fetchone()
    return bool(row and row[0] == "completed")


def reset_failed_states(conn: Any) -> None:
    conn.execute("DELETE FROM rotten_tomatoes_ingest_state WHERE source = 'rotten_tomatoes' AND status = 'failed'")


def mark_running_states_interrupted(conn: Any) -> int:
    rows = conn.execute(
        """
        UPDATE rotten_tomatoes_ingest_state
        SET status = 'failed',
            stage = 'interrupted',
            updated_at = %s,
            last_error = 'Interrupted previous run'
        WHERE source = 'rotten_tomatoes'
          AND status = 'running'
        RETURNING movie_id
        """,
        (utc_now(),),
    ).fetchall()
    return len(rows)


def insert_issue(
    conn: Any,
    *,
    issue_source: str,
    issue_type: str,
    movie_id: int | None,
    ems_id: str | None,
    source_url: str | None,
    details: str,
) -> None:
    conn.execute(
        insert_ignore_sql(
            "rotten_tomatoes_ingest_issues",
            ["issue_source", "issue_type", "movie_id", "ems_id", "source_url", "details"],
        ),
        (issue_source, issue_type, movie_id, ems_id, source_url, details),
    )


def fetch_all_reviews(
    fetcher: TextFetcher,
    *,
    ems_id: str,
    movie_id: int,
    top_only: bool = False,
    max_pages: int = 100,
) -> tuple[list[CriticReview], Path | None]:
    reviews: list[CriticReview] = []
    cursor: str | None = None
    last_cache_path: Path | None = None
    for _page in range(max_pages):
        payload, cache_path, _fetched = fetcher.get_json(review_api_url(ems_id, top_only=top_only, after=cursor))
        last_cache_path = cache_path
        page_reviews, cursor = parse_reviews_payload(payload, ems_id=ems_id, movie_id=movie_id)
        reviews.extend(page_reviews)
        if not cursor:
            break
    return reviews, last_cache_path


def ingest_movie(
    conn: Any,
    fetcher: TextFetcher,
    *,
    movie: CandidateMovie,
    issue_source: str,
) -> int:
    upsert_state(conn, movie_id=movie.movie_id, stage="matching", status="running")
    match, media, media_cache_path = match_movie(conn, fetcher, movie)
    if match.ems_id is None:
        upsert_movie_match(conn, match)
        insert_issue(
            conn,
            issue_source=issue_source,
            issue_type="rt_match_failed",
            movie_id=movie.movie_id,
            ems_id=None,
            source_url=None,
            details=match.notes or "No RT match",
        )
        upsert_state(conn, movie_id=movie.movie_id, stage="matching", status="failed", last_error=match.notes)
        return 0
    if media is not None and media_cache_path is not None:
        upsert_media(conn, media, fetched_at=utc_now(), raw_cache_path=media_cache_path)
    elif media is None:
        media = minimal_media_for_existing_match(match.ems_id)
        upsert_media(conn, media, fetched_at=utc_now(), raw_cache_path=Path(""))
    upsert_movie_match(conn, match)
    upsert_state(conn, movie_id=movie.movie_id, stage="reviews", status="running")
    reviews, review_cache_path = fetch_all_reviews(fetcher, ems_id=match.ems_id, movie_id=movie.movie_id)
    if review_cache_path is None:
        review_cache_path = media_cache_path or Path("")
    insert_reviews(conn, reviews, fetched_at=utc_now(), raw_cache_path=review_cache_path)
    top_reviews, top_review_cache_path = fetch_all_reviews(
        fetcher,
        ems_id=match.ems_id,
        movie_id=movie.movie_id,
        top_only=True,
    )
    if top_reviews:
        insert_reviews(
            conn,
            top_reviews,
            fetched_at=utc_now(),
            raw_cache_path=top_review_cache_path or review_cache_path,
        )
    update_media_review_counts(conn, match.ems_id)
    if not reviews and not top_reviews:
        insert_issue(
            conn,
            issue_source=issue_source,
            issue_type="empty_reviews",
            movie_id=movie.movie_id,
            ems_id=match.ems_id,
            source_url=media.source_url if media else None,
            details="Matched RT media but critic review API returned no reviews",
        )
    upsert_state(conn, movie_id=movie.movie_id, stage="done", status="completed")
    return len(reviews) + len(top_reviews)


def minimal_media_for_existing_match(ems_id: str) -> RottenTomatoesMedia:
    return RottenTomatoesMedia(
        ems_id=ems_id,
        vanity_slug=ems_id,
        canonical_url="",
        title="",
        release_year=None,
        media_type="movie",
        tomatometer_score=None,
        tomatometer_sentiment=None,
        certified_fresh=None,
        critic_review_count=None,
        top_critic_review_count=None,
        source_url="",
    )


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    conn = connect_database(args.database_url)
    try:
        initialize_database(conn)
        interrupted = mark_running_states_interrupted(conn)
        if interrupted:
            print(f"Marked {interrupted} interrupted Rotten Tomatoes ingest states as failed.", file=sys.stderr)
        if args.reset_failed:
            reset_failed_states(conn)
        conn.commit()
        candidates = select_candidate_movies(conn, release_year=args.release_year, movie_limit=args.movie_limit)
        if args.dry_run:
            for movie in candidates:
                print(f"{movie.movie_id}\t{movie.release_year or ''}\t{movie.release_date or ''}\t{movie.title}")
            print(f"Candidate movies: {len(candidates)}", file=sys.stderr)
            return 0
        fetcher = TextFetcher(
            args.cache_dir,
            refresh=args.refresh,
            offline=args.offline,
            delay_seconds=args.delay_seconds,
            user_agent=args.user_agent,
        )
        processed = 0
        skipped = 0
        review_rows = 0
        for index, movie in enumerate(candidates, start=1):
            if not args.refresh and state_is_completed(conn, movie.movie_id):
                print(f"Skipping completed {index}/{len(candidates)} {movie.title}", file=sys.stderr)
                skipped += 1
                continue
            print(f"Ingesting {index}/{len(candidates)} {movie.title}", file=sys.stderr)
            try:
                review_rows += ingest_movie(conn, fetcher, movie=movie, issue_source=args.issue_source)
                processed += 1
                conn.commit()
            except Exception as exc:  # noqa: BLE001 - persist state and continue.
                conn.rollback()
                upsert_state(conn, movie_id=movie.movie_id, stage="error", status="failed", last_error=str(exc))
                insert_issue(
                    conn,
                    issue_source=args.issue_source,
                    issue_type="ingest_failed",
                    movie_id=movie.movie_id,
                    ems_id=None,
                    source_url=None,
                    details=str(exc),
                )
                conn.commit()
                if args.fail_fast:
                    raise
        print(f"Processed={processed} skipped={skipped} reviews={review_rows}", file=sys.stderr)
        return 0
    finally:
        conn.close()


def validate_args(args: argparse.Namespace) -> None:
    if args.delay_seconds < 5.0 and not args.offline:
        raise SystemExit("--delay-seconds must be at least 5.0 unless --offline is set")
    if args.movie_limit is not None and args.movie_limit < 1:
        raise SystemExit("--movie-limit must be positive")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=database_url_from_env(),
        help="PostgreSQL connection URL. Defaults to DATABASE_URL or POSTGRES_DSN.",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--delay-seconds", type=float, default=5.0)
    parser.add_argument("--movie-limit", type=int)
    parser.add_argument("--release-year", type=int)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--reset-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--issue-source", default="rotten_tomatoes_critic_reviews")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
