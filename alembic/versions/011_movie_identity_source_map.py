"""movie identity source map

Revision ID: 011_movie_identity_source_map
Revises: 010_forecast_shadow_status_columns
Create Date: 2026-07-06
"""

from __future__ import annotations

from alembic import op


revision = "011_movie_identity_source_map"
down_revision = "010_forecast_shadow_status_columns"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE movies
            ADD COLUMN IF NOT EXISTS movie_url TEXT,
            ADD COLUMN IF NOT EXISTS release_year INTEGER,
            ADD COLUMN IF NOT EXISTS release_date DATE,
            ADD COLUMN IF NOT EXISTS opusdata_id TEXT,
            ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        """
    )
    op.execute("ALTER TABLE movies ALTER COLUMN movie_url DROP NOT NULL")
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_movies_movie_url_not_null
            ON movies(movie_url) WHERE movie_url IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_movies_opusdata_id_not_null
            ON movies(opusdata_id) WHERE opusdata_id IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_movies_title_release_year
            ON movies(title, release_year)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_movies_release_date
            ON movies(release_date)
        """
    )
    op.execute(
        """
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
        )
        """
    )
    op.execute(
        """
        ALTER TABLE movie_source_ids
            ADD COLUMN IF NOT EXISTS source_title TEXT,
            ADD COLUMN IF NOT EXISTS match_status TEXT DEFAULT 'unmatched',
            ADD COLUMN IF NOT EXISTS match_method TEXT,
            ADD COLUMN IF NOT EXISTS match_score DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS matched_at TIMESTAMPTZ
        """
    )
    op.execute(
        """
        INSERT INTO movie_source_ids (
            movie_id, source, source_movie_id, source_title,
            match_status, match_method, match_score, matched_at
        )
        SELECT
            movie_id,
            'the_numbers',
            movie_url,
            title,
            'matched',
            'backfilled_movie_url',
            1.0,
            CURRENT_TIMESTAMP
        FROM movies
        WHERE movie_url IS NOT NULL
        ON CONFLICT(source, source_movie_id) DO UPDATE SET
            movie_id = excluded.movie_id,
            source_title = excluded.source_title,
            match_status = excluded.match_status,
            match_method = excluded.match_method,
            match_score = excluded.match_score,
            matched_at = excluded.matched_at
        """
    )
    op.execute(
        """
        INSERT INTO movie_source_ids (
            movie_id, source, source_movie_id, source_title,
            match_status, match_method, match_score, matched_at
        )
        SELECT
            movie_id,
            'the_numbers_opusdata',
            opusdata_id,
            title,
            'matched',
            'backfilled_opusdata_id',
            1.0,
            CURRENT_TIMESTAMP
        FROM movies
        WHERE opusdata_id IS NOT NULL
        ON CONFLICT(source, source_movie_id) DO UPDATE SET
            movie_id = excluded.movie_id,
            source_title = excluded.source_title,
            match_status = excluded.match_status,
            match_method = excluded.match_method,
            match_score = excluded.match_score,
            matched_at = excluded.matched_at
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_movies_release_date")
    op.execute("DROP INDEX IF EXISTS idx_movies_title_release_year")
    op.execute("DROP INDEX IF EXISTS uq_movies_opusdata_id_not_null")
    op.execute("DROP INDEX IF EXISTS uq_movies_movie_url_not_null")
