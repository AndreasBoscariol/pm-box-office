"""Append-only source events and model input provenance snapshots.

Revision ID: 023_forecast_input_provenance
Revises: 022_forecast_distribution_payloads
"""

from __future__ import annotations

from alembic import op


revision = "023_forecast_input_provenance"
down_revision = "022_forecast_distribution_payloads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE SCHEMA IF NOT EXISTS analytics;
        CREATE TABLE IF NOT EXISTS analytics.forecast_ingest_events (
            ingest_event_id TEXT PRIMARY KEY,
            source_key TEXT NOT NULL,
            event_type TEXT NOT NULL,
            source_run_id UUID,
            available_at_utc TIMESTAMPTZ NOT NULL,
            payload JSONB NOT NULL DEFAULT '{}',
            payload_hash TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (source_key, event_type, source_run_id, payload_hash)
        );
        CREATE INDEX IF NOT EXISTS idx_forecast_ingest_events_source_time
            ON analytics.forecast_ingest_events (source_key, available_at_utc DESC);
        CREATE TABLE IF NOT EXISTS analytics.forecast_input_snapshots (
            input_snapshot_id TEXT PRIMARY KEY,
            release_run_id BIGINT NOT NULL,
            movie_id BIGINT,
            model_version TEXT NOT NULL,
            as_of_utc TIMESTAMPTZ NOT NULL,
            availability JSONB NOT NULL,
            quality_bucket TEXT NOT NULL,
            snapshot_hash TEXT NOT NULL UNIQUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_forecast_input_snapshots_release_time
            ON analytics.forecast_input_snapshots (release_run_id, model_version, as_of_utc DESC);
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS analytics.forecast_input_snapshots")
    op.execute("DROP TABLE IF EXISTS analytics.forecast_ingest_events")
