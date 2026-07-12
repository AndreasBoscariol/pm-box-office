"""Persist Thursday preview and OW shadow distributions.

Revision ID: 021_thursday_preview_distribution_fields
Revises: 020_reported_preview_actuals
"""

from __future__ import annotations

from alembic import op


revision = "021_thursday_preview_distribution_fields"
down_revision = "020_reported_preview_actuals"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE analytics.amc_opening_shadow_forecasts
            ADD COLUMN IF NOT EXISTS predicted_thursday_previews_lo80_usd NUMERIC,
            ADD COLUMN IF NOT EXISTS predicted_thursday_previews_hi80_usd NUMERIC,
            ADD COLUMN IF NOT EXISTS predicted_thursday_previews_lo95_usd NUMERIC,
            ADD COLUMN IF NOT EXISTS predicted_thursday_previews_hi95_usd NUMERIC,
            ADD COLUMN IF NOT EXISTS preview_residual_pool_scope TEXT,
            ADD COLUMN IF NOT EXISTS thursday_amc_shadow_ow_lo80_usd NUMERIC,
            ADD COLUMN IF NOT EXISTS thursday_amc_shadow_ow_hi80_usd NUMERIC,
            ADD COLUMN IF NOT EXISTS thursday_amc_shadow_ow_lo95_usd NUMERIC,
            ADD COLUMN IF NOT EXISTS thursday_amc_shadow_ow_hi95_usd NUMERIC,
            ADD COLUMN IF NOT EXISTS thursday_selected_candidate TEXT,
            ADD COLUMN IF NOT EXISTS thursday_target_classification TEXT,
            ADD COLUMN IF NOT EXISTS thursday_training_cutoff TIMESTAMPTZ;
        """
    )


def downgrade() -> None:
    for column in [
        "thursday_training_cutoff", "thursday_target_classification", "thursday_selected_candidate",
        "thursday_amc_shadow_ow_hi95_usd", "thursday_amc_shadow_ow_lo95_usd",
        "thursday_amc_shadow_ow_hi80_usd", "thursday_amc_shadow_ow_lo80_usd",
        "preview_residual_pool_scope", "predicted_thursday_previews_hi95_usd",
        "predicted_thursday_previews_lo95_usd", "predicted_thursday_previews_hi80_usd",
        "predicted_thursday_previews_lo80_usd",
    ]:
        op.execute(f"ALTER TABLE analytics.amc_opening_shadow_forecasts DROP COLUMN IF EXISTS {column}")
