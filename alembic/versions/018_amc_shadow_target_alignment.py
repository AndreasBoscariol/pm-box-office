"""AMC shadow availability and Friday target alignment

Revision ID: 018_amc_shadow_target_alignment
Revises: 017_amc_opening_shadow_forecasts
Create Date: 2026-07-10
"""

from __future__ import annotations

from alembic import op


revision = "018_amc_shadow_target_alignment"
down_revision = "017_amc_opening_shadow_forecasts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE analytics.amc_opening_shadow_forecasts
            ADD COLUMN IF NOT EXISTS amc_candidate_available BOOLEAN,
            ADD COLUMN IF NOT EXISTS amc_candidate_unavailable_reason TEXT,
            ADD COLUMN IF NOT EXISTS thursday_amc_seats_collected BOOLEAN,
            ADD COLUMN IF NOT EXISTS predicted_thursday_previews_usd NUMERIC,
            ADD COLUMN IF NOT EXISTS friday_only_amc_observed_seats NUMERIC,
            ADD COLUMN IF NOT EXISTS evaluation_target_includes_previews BOOLEAN,
            ADD COLUMN IF NOT EXISTS reported_friday_gross_usd NUMERIC,
            ADD COLUMN IF NOT EXISTS thursday_preview_actual_usd NUMERIC,
            ADD COLUMN IF NOT EXISTS friday_calendar_actual_usd NUMERIC;
        """
    )


def downgrade() -> None:
    for column in [
        "friday_calendar_actual_usd", "thursday_preview_actual_usd", "reported_friday_gross_usd",
        "evaluation_target_includes_previews", "friday_only_amc_observed_seats",
        "predicted_thursday_previews_usd", "thursday_amc_seats_collected",
        "amc_candidate_unavailable_reason", "amc_candidate_available",
    ]:
        op.execute(f"ALTER TABLE analytics.amc_opening_shadow_forecasts DROP COLUMN IF EXISTS {column}")
