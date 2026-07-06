"""the numbers budget and franchise metadata

Revision ID: 008_the_numbers_budget_franchise_metadata
Revises: 007_forecast_deployment_tables
Create Date: 2026-07-03
"""

from __future__ import annotations

from alembic import op


revision = "008_the_numbers_budget_franchise_metadata"
down_revision = "007_forecast_deployment_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE the_numbers_movie_metadata
            ADD COLUMN IF NOT EXISTS production_budget_usd BIGINT
        """
    )
    op.execute(
        """
        ALTER TABLE the_numbers_movie_metadata
            ADD COLUMN IF NOT EXISTS franchise TEXT
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_tn_movie_metadata_franchise
            ON the_numbers_movie_metadata(franchise)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_tn_movie_metadata_franchise")
    op.execute("ALTER TABLE the_numbers_movie_metadata DROP COLUMN IF EXISTS franchise")
    op.execute("ALTER TABLE the_numbers_movie_metadata DROP COLUMN IF EXISTS production_budget_usd")
