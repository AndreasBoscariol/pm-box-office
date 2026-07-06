"""forecast shadow status columns

Revision ID: 010_forecast_shadow_status_columns
Revises: 009_forecast_lab_tables
Create Date: 2026-07-05
"""

from __future__ import annotations

from alembic import op


revision = "010_forecast_shadow_status_columns"
down_revision = "009_forecast_lab_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS forecast_predictions (
            forecast_prediction_id BIGSERIAL PRIMARY KEY,
            feature_snapshot_id BIGINT,
            model_version_id BIGINT,
            model_version TEXT NOT NULL,
            model_family TEXT NOT NULL,
            deployment_status TEXT NOT NULL,
            point_forecast_usd NUMERIC(14, 2),
            lower_80_usd NUMERIC(14, 2),
            upper_80_usd NUMERIC(14, 2),
            remaining_gross_forecast_usd NUMERIC(14, 2),
            prediction_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    op.execute("ALTER TABLE forecast_predictions ALTER COLUMN feature_snapshot_id DROP NOT NULL")
    op.execute("ALTER TABLE forecast_predictions ADD COLUMN IF NOT EXISTS candidate_id TEXT")
    op.execute("ALTER TABLE forecast_predictions ADD COLUMN IF NOT EXISTS forecast_status TEXT NOT NULL DEFAULT 'production'")
    op.execute("ALTER TABLE forecast_predictions ADD COLUMN IF NOT EXISTS movie_id BIGINT")
    op.execute("ALTER TABLE forecast_predictions ADD COLUMN IF NOT EXISTS target_type TEXT")
    op.execute("ALTER TABLE forecast_predictions ADD COLUMN IF NOT EXISTS target_start_date DATE")
    op.execute("ALTER TABLE forecast_predictions ADD COLUMN IF NOT EXISTS target_end_date DATE")
    op.execute("ALTER TABLE forecast_predictions ADD COLUMN IF NOT EXISTS origin_day INTEGER")
    op.execute("ALTER TABLE forecast_predictions ADD COLUMN IF NOT EXISTS origin_timestamp_utc TIMESTAMPTZ")
    op.execute("ALTER TABLE forecast_predictions ADD COLUMN IF NOT EXISTS forecast_log_gross DOUBLE PRECISION")
    op.execute("ALTER TABLE forecast_predictions ADD COLUMN IF NOT EXISTS interval_80_low_usd NUMERIC(14, 2)")
    op.execute("ALTER TABLE forecast_predictions ADD COLUMN IF NOT EXISTS interval_80_high_usd NUMERIC(14, 2)")
    op.execute("ALTER TABLE forecast_predictions ADD COLUMN IF NOT EXISTS interval_95_low_usd NUMERIC(14, 2)")
    op.execute("ALTER TABLE forecast_predictions ADD COLUMN IF NOT EXISTS interval_95_high_usd NUMERIC(14, 2)")
    op.execute("ALTER TABLE forecast_predictions ADD COLUMN IF NOT EXISTS active_forecast_layer TEXT")
    op.execute("ALTER TABLE forecast_predictions ADD COLUMN IF NOT EXISTS interval_calibrator_id TEXT")
    op.execute("ALTER TABLE forecast_predictions ADD COLUMN IF NOT EXISTS strict_known_at_policy BOOLEAN NOT NULL DEFAULT FALSE")
    op.execute("ALTER TABLE forecast_predictions ADD COLUMN IF NOT EXISTS diagnostics_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb")
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_forecast_predictions_candidate_shadow_key
            ON forecast_predictions (
                candidate_id,
                forecast_status,
                movie_id,
                target_type,
                target_start_date,
                origin_day,
                origin_timestamp_utc
            )
            WHERE candidate_id IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_forecast_predictions_candidate_status
            ON forecast_predictions (candidate_id, forecast_status)
            WHERE candidate_id IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_forecast_predictions_candidate_status")
    op.execute("DROP INDEX IF EXISTS ux_forecast_predictions_candidate_shadow_key")
    op.execute("ALTER TABLE forecast_predictions DROP COLUMN IF EXISTS diagnostics_jsonb")
    op.execute("ALTER TABLE forecast_predictions DROP COLUMN IF EXISTS strict_known_at_policy")
    op.execute("ALTER TABLE forecast_predictions DROP COLUMN IF EXISTS interval_calibrator_id")
    op.execute("ALTER TABLE forecast_predictions DROP COLUMN IF EXISTS active_forecast_layer")
    op.execute("ALTER TABLE forecast_predictions DROP COLUMN IF EXISTS interval_95_high_usd")
    op.execute("ALTER TABLE forecast_predictions DROP COLUMN IF EXISTS interval_95_low_usd")
    op.execute("ALTER TABLE forecast_predictions DROP COLUMN IF EXISTS interval_80_high_usd")
    op.execute("ALTER TABLE forecast_predictions DROP COLUMN IF EXISTS interval_80_low_usd")
    op.execute("ALTER TABLE forecast_predictions DROP COLUMN IF EXISTS forecast_log_gross")
    op.execute("ALTER TABLE forecast_predictions DROP COLUMN IF EXISTS origin_timestamp_utc")
    op.execute("ALTER TABLE forecast_predictions DROP COLUMN IF EXISTS origin_day")
    op.execute("ALTER TABLE forecast_predictions DROP COLUMN IF EXISTS target_end_date")
    op.execute("ALTER TABLE forecast_predictions DROP COLUMN IF EXISTS target_start_date")
    op.execute("ALTER TABLE forecast_predictions DROP COLUMN IF EXISTS target_type")
    op.execute("ALTER TABLE forecast_predictions DROP COLUMN IF EXISTS movie_id")
    op.execute("ALTER TABLE forecast_predictions DROP COLUMN IF EXISTS forecast_status")
    op.execute("ALTER TABLE forecast_predictions DROP COLUMN IF EXISTS candidate_id")
    op.execute("ALTER TABLE forecast_predictions ALTER COLUMN feature_snapshot_id SET NOT NULL")
