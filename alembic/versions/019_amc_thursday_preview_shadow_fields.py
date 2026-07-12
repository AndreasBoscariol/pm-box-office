"""AMC Thursday preview shadow fields

Revision ID: 019_amc_thursday_preview_shadow_fields
Revises: 018_amc_shadow_target_alignment
Create Date: 2026-07-11
"""

from __future__ import annotations

from alembic import op


revision = "019_amc_thursday_preview_shadow_fields"
down_revision = "018_amc_shadow_target_alignment"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE analytics.amc_opening_shadow_forecasts
            ADD COLUMN IF NOT EXISTS thursday_amc_observed_preview_seats NUMERIC,
            ADD COLUMN IF NOT EXISTS thursday_amc_shadow_ow_prior_usd NUMERIC,
            ADD COLUMN IF NOT EXISTS thursday_amc_shadow_ow_prior_source TEXT;
        """
    )


def downgrade() -> None:
    for column in [
        "thursday_amc_shadow_ow_prior_source",
        "thursday_amc_shadow_ow_prior_usd",
        "thursday_amc_observed_preview_seats",
    ]:
        op.execute(f"ALTER TABLE analytics.amc_opening_shadow_forecasts DROP COLUMN IF EXISTS {column}")
