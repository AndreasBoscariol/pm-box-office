"""the numbers movie metadata

Revision ID: 006_the_numbers_movie_metadata
Revises: 005_ingest_orchestration
Create Date: 2026-07-01
"""

from __future__ import annotations

from alembic import op


revision = "006_the_numbers_movie_metadata"
down_revision = "005_ingest_orchestration"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS the_numbers_movie_metadata (
            movie_id BIGINT NOT NULL REFERENCES movies(movie_id),
            movie_url TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            release_year INTEGER,
            opusdata_id TEXT,
            mpa_rating TEXT,
            mpa_rating_details TEXT,
            genre TEXT,
            source_url TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            raw_cache_path TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP::text),
            PRIMARY KEY(movie_id)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_tn_movie_metadata_rating
            ON the_numbers_movie_metadata(mpa_rating)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_tn_movie_metadata_genre
            ON the_numbers_movie_metadata(genre)
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS the_numbers_movie_metadata CASCADE")
