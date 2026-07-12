"""prospective shadow cohort and immutable forecast ledgers

Revision ID: 017_prospective_shadow
Revises: 016_prediction_market_backtest
"""
from __future__ import annotations
from alembic import op

revision="017_prospective_shadow";down_revision="016_prediction_market_backtest";branch_labels=None;depends_on=None

def upgrade()->None:
    op.execute("""
    CREATE TABLE prediction_market_backtest.prospective_cohort (
      cohort_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, artifact_checksum TEXT NOT NULL,
      movie_id BIGINT NOT NULL REFERENCES movies(movie_id), title TEXT NOT NULL, opening_weekend_start DATE NOT NULL,
      eligibility_decided_at TIMESTAMPTZ NOT NULL, target_universe_status TEXT NOT NULL, eligibility_status TEXT NOT NULL,
      exclusion_reason TEXT, first_supported_origin INTEGER, latest_valid_origin INTEGER, actual_availability_status TEXT NOT NULL DEFAULT 'awaiting',
      primary_panel_status TEXT NOT NULL DEFAULT 'pending', all_origin_panel_status TEXT NOT NULL DEFAULT 'pending',
      polymarket_available BOOLEAN, data_quality_warnings JSONB NOT NULL DEFAULT '[]', UNIQUE(artifact_checksum,movie_id));
    CREATE TABLE prediction_market_backtest.prospective_forecasts (
      prospective_forecast_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, record_key TEXT NOT NULL,
      record_version TIMESTAMPTZ NOT NULL, artifact_checksum TEXT NOT NULL, movie_id BIGINT NOT NULL REFERENCES movies(movie_id),
      origin INTEGER NOT NULL, information_cutoff_utc TIMESTAMPTZ NOT NULL, forecast_created_utc TIMESTAMPTZ NOT NULL,
      forecast_available_utc TIMESTAMPTZ NOT NULL, point_forecast NUMERIC(24,2), policies JSONB NOT NULL,
      base_parameters JSONB NOT NULL, calibration_parameters JSONB NOT NULL, threshold_probabilities JSONB NOT NULL,
      random_seed NUMERIC(20,0) NOT NULL, draw_checksum TEXT NOT NULL, input_data_checksum TEXT NOT NULL,
      code_commit TEXT NOT NULL, fallback_status TEXT NOT NULL, generation_status TEXT NOT NULL,
      UNIQUE(record_key,record_version));
    CREATE TABLE prediction_market_backtest.prospective_actuals (
      actual_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, movie_id BIGINT NOT NULL REFERENCES movies(movie_id),
      gross_usd NUMERIC(24,2) NOT NULL, source TEXT NOT NULL, source_published_at TIMESTAMPTZ NOT NULL,
      received_at TIMESTAMPTZ NOT NULL, duration_days INTEGER NOT NULL, currency TEXT NOT NULL, geography TEXT NOT NULL,
      revision_of BIGINT REFERENCES prediction_market_backtest.prospective_actuals(actual_id), final_approved BOOLEAN NOT NULL DEFAULT FALSE,
      UNIQUE(movie_id,source,source_published_at,received_at));
    CREATE TABLE prediction_market_backtest.prospective_decisions (
      decision_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, artifact_checksum TEXT NOT NULL, look_size INTEGER NOT NULL CHECK(look_size IN (30,50)),
      decided_at TIMESTAMPTZ NOT NULL, confidence_level NUMERIC(5,4) NOT NULL, decision TEXT NOT NULL,
      metrics JSONB NOT NULL, UNIQUE(artifact_checksum,look_size));
    """)
def downgrade()->None:
    for table in ("prospective_decisions","prospective_actuals","prospective_forecasts","prospective_cohort"):op.execute(f"DROP TABLE IF EXISTS prediction_market_backtest.{table}")
