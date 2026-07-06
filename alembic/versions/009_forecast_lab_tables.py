"""forecast lab tables

Revision ID: 009_forecast_lab_tables
Revises: 008_the_numbers_budget_franchise_metadata
Create Date: 2026-07-03
"""

from __future__ import annotations

from alembic import op


revision = "009_forecast_lab_tables"
down_revision = "008_the_numbers_budget_franchise_metadata"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS forecast_lab_runs (
            lab_run_id BIGSERIAL PRIMARY KEY,
            run_name TEXT NOT NULL,
            config_jsonb JSONB NOT NULL,
            config_hash TEXT NOT NULL,
            code_version TEXT,
            data_cutoff_utc TIMESTAMPTZ,
            target_types TEXT[] NOT NULL,
            origin_days INTEGER[] NOT NULL,
            status TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            completed_at TIMESTAMPTZ,
            notes TEXT
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS forecast_lab_panel_rows (
            lab_run_id BIGINT NOT NULL REFERENCES forecast_lab_runs(lab_run_id) ON DELETE CASCADE,
            movie_id BIGINT NOT NULL,
            title TEXT,
            release_date DATE,
            target_type TEXT NOT NULL,
            target_start_date DATE NOT NULL,
            target_end_date DATE NOT NULL,
            origin_day INTEGER NOT NULL,
            origin_timestamp_utc TIMESTAMPTZ NOT NULL,
            label_gross_usd NUMERIC,
            label_log_gross DOUBLE PRECISION,
            feature_values_jsonb JSONB NOT NULL,
            feature_availability_jsonb JSONB NOT NULL,
            source_known_at_jsonb JSONB NOT NULL,
            source_payload_hash_jsonb JSONB NOT NULL,
            row_quality_flags_jsonb JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (lab_run_id, movie_id, target_type, target_start_date, origin_day)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_forecast_lab_panel_rows_movie
            ON forecast_lab_panel_rows(movie_id)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_forecast_lab_panel_rows_origin
            ON forecast_lab_panel_rows(origin_timestamp_utc)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_forecast_lab_panel_rows_run_origin
            ON forecast_lab_panel_rows(lab_run_id, origin_day)
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS forecast_lab_predictions (
            lab_run_id BIGINT NOT NULL REFERENCES forecast_lab_runs(lab_run_id) ON DELETE CASCADE,
            model_id TEXT NOT NULL,
            fold_id TEXT NOT NULL DEFAULT '',
            movie_id BIGINT NOT NULL,
            target_type TEXT NOT NULL,
            target_start_date DATE NOT NULL,
            origin_day INTEGER NOT NULL,
            origin_timestamp_utc TIMESTAMPTZ,
            prediction_kind TEXT NOT NULL,
            forecast_log_gross DOUBLE PRECISION,
            forecast_gross_usd NUMERIC,
            actual_gross_usd NUMERIC,
            actual_log_gross DOUBLE PRECISION,
            interval_80_low_usd NUMERIC,
            interval_80_high_usd NUMERIC,
            interval_95_low_usd NUMERIC,
            interval_95_high_usd NUMERIC,
            module_predictions_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
            diagnostics_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (lab_run_id, model_id, fold_id, movie_id, target_type, target_start_date, origin_day)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS forecast_lab_metrics (
            lab_run_id BIGINT NOT NULL REFERENCES forecast_lab_runs(lab_run_id) ON DELETE CASCADE,
            model_id TEXT NOT NULL,
            metric_scope TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            origin_day INTEGER NOT NULL,
            segment_name TEXT NOT NULL DEFAULT '',
            segment_value TEXT NOT NULL DEFAULT '',
            metric_value DOUBLE PRECISION,
            n_rows INTEGER NOT NULL,
            n_movies INTEGER NOT NULL,
            baseline_model_id TEXT,
            lift_value DOUBLE PRECISION,
            diagnostics_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (lab_run_id, model_id, metric_scope, metric_name, origin_day, segment_name, segment_value)
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS forecast_lab_metrics")
    op.execute("DROP TABLE IF EXISTS forecast_lab_predictions")
    op.execute("DROP INDEX IF EXISTS idx_forecast_lab_panel_rows_run_origin")
    op.execute("DROP INDEX IF EXISTS idx_forecast_lab_panel_rows_origin")
    op.execute("DROP INDEX IF EXISTS idx_forecast_lab_panel_rows_movie")
    op.execute("DROP TABLE IF EXISTS forecast_lab_panel_rows")
    op.execute("DROP TABLE IF EXISTS forecast_lab_runs")
