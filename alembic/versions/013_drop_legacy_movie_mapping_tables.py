"""drop legacy movie mapping tables

Revision ID: 013_drop_legacy_movie_mapping_tables
Revises: 012_canonical_movie_source_backfills
Create Date: 2026-07-06
"""

from __future__ import annotations

from alembic import op


revision = "013_drop_legacy_movie_mapping_tables"
down_revision = "012_canonical_movie_source_backfills"
branch_labels = None
depends_on = None


LEGACY_TABLES = (
    "movie_imdb_titles",
    "movie_letterboxd_films",
    "movie_wiki_pages",
    "movie_rotten_tomatoes_media",
)


def upgrade() -> None:
    bind = op.get_bind()
    _assert_legacy_source_coverage(bind)
    for table in LEGACY_TABLES:
        op.execute(f"DROP TABLE IF EXISTS {table}")


def downgrade() -> None:
    op.execute(
        """
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
        )
        """
    )
    op.execute(
        """
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
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS movie_wiki_pages (
            movie_id BIGINT NOT NULL REFERENCES movies(movie_id),
            language TEXT NOT NULL,
            wiki_page_id INTEGER,
            match_status TEXT NOT NULL CHECK (
                match_status IN ('matched', 'not_found', 'ambiguous', 'manual_override')
            ),
            match_method TEXT NOT NULL,
            match_query TEXT,
            match_rank INTEGER,
            match_score DOUBLE PRECISION,
            matched_at TEXT NOT NULL,
            notes TEXT,
            UNIQUE(movie_id, language)
        )
        """
    )
    op.execute(
        """
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
        )
        """
    )


def _assert_legacy_source_coverage(bind: object) -> None:
    checks = (
        (
            "movie_imdb_titles",
            "imdb",
            "legacy.tconst",
            "legacy.tconst IS NOT NULL",
        ),
        (
            "movie_letterboxd_films",
            "letterboxd",
            "legacy.letterboxd_slug",
            "legacy.letterboxd_slug IS NOT NULL",
        ),
        (
            "movie_wiki_pages",
            "wikipedia",
            "legacy.language || ':' || legacy.wiki_page_id::text",
            "legacy.wiki_page_id IS NOT NULL",
        ),
        (
            "movie_rotten_tomatoes_media",
            "rottentomatoes",
            "legacy.ems_id",
            "legacy.ems_id IS NOT NULL",
        ),
    )
    for table, source, source_expr, key_predicate in checks:
        if not _relation_exists(bind, table):
            continue
        missing = bind.exec_driver_sql(  # type: ignore[attr-defined]
            f"""
            SELECT COUNT(*)::bigint
            FROM {table} legacy
            LEFT JOIN movie_source_ids src
              ON src.source = %s
             AND src.source_movie_id = {source_expr}
            WHERE {key_predicate}
              AND legacy.match_status IN ('matched', 'manual_override')
              AND src.movie_id IS NULL
            """,
            (source,),
        ).scalar()
        if int(missing or 0) > 0:
            raise RuntimeError(
                f"Cannot drop {table}: {missing} matched legacy rows are missing from movie_source_ids"
            )


def _relation_exists(bind: object, relation_name: str) -> bool:
    row = bind.exec_driver_sql("SELECT to_regclass(%s)", (relation_name,)).fetchone()  # type: ignore[attr-defined]
    return bool(row and row[0])
