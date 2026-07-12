"""Persist complete production forecast distributions with forecast emissions.

Revision ID: 022_forecast_distribution_payloads
Revises: 021_thursday_preview_distribution_fields
"""

from __future__ import annotations

from alembic import op


revision = "022_forecast_distribution_payloads"
down_revision = "021_thursday_preview_distribution_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE analytics.movie_opening_weekend_forecasts
            ADD COLUMN IF NOT EXISTS distribution_payload JSONB;
        ALTER TABLE analytics.movie_forecast_emissions
            ADD COLUMN IF NOT EXISTS distribution_payload JSONB;
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE analytics.movie_forecast_emissions DROP COLUMN IF EXISTS distribution_payload;")
    op.execute("ALTER TABLE analytics.movie_opening_weekend_forecasts DROP COLUMN IF EXISTS distribution_payload;")
