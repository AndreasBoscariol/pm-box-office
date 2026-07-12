"""read-only prediction market paper-trading schema

Revision ID: 016_prediction_market_backtest
Revises: 015_forecast_refresh_queue
Create Date: 2026-07-10
"""

from __future__ import annotations

from alembic import op

revision = "016_prediction_market_backtest"
down_revision = "015_forecast_refresh_queue"
branch_labels = None
depends_on = None


TABLES = (
    "portfolio_ledger", "paper_trades", "paper_signals", "forecast_book_pairings",
    "bucket_probabilities", "forecast_distributions", "order_book_snapshots", "order_book_events",
    "movie_matches", "event_bucket_validations", "contract_semantics", "polymarket_markets", "polymarket_events",
)


def upgrade() -> None:
    op.execute("""
    CREATE SCHEMA IF NOT EXISTS prediction_market_backtest;
    CREATE TABLE prediction_market_backtest.polymarket_events (
      event_id TEXT PRIMARY KEY, event_slug TEXT, title TEXT NOT NULL, description TEXT, category TEXT,
      tags JSONB NOT NULL DEFAULT '[]', active BOOLEAN, closed BOOLEAN, start_time TIMESTAMPTZ, end_time TIMESTAMPTZ,
      resolution_source TEXT, resolution_rules_raw TEXT, resolution_rules_hash TEXT,
      created_at_exchange TIMESTAMPTZ, updated_at_exchange TIMESTAMPTZ, synced_at_utc TIMESTAMPTZ NOT NULL,
      raw_gamma_json JSONB NOT NULL);
    CREATE TABLE prediction_market_backtest.polymarket_markets (
      market_id TEXT PRIMARY KEY, event_id TEXT NOT NULL REFERENCES prediction_market_backtest.polymarket_events(event_id),
      condition_id TEXT, question TEXT NOT NULL, slug TEXT, active BOOLEAN, closed BOOLEAN, accepting_orders BOOLEAN,
      neg_risk BOOLEAN, tick_size NUMERIC(20,10), minimum_order_size NUMERIC(30,10), fee_rate NUMERIC(20,10),
      fee_schedule_raw JSONB, yes_token_id TEXT, no_token_id TEXT, outcome_labels JSONB,
      resolution_status TEXT, winning_outcome TEXT, raw_gamma_json JSONB NOT NULL, raw_clob_json JSONB,
      CHECK (yes_token_id IS NULL OR yes_token_id <> ''), CHECK (no_token_id IS NULL OR no_token_id <> ''));
    CREATE TABLE prediction_market_backtest.contract_semantics (
      market_id TEXT PRIMARY KEY REFERENCES prediction_market_backtest.polymarket_markets(market_id), target_metric TEXT,
      geographic_scope TEXT, currency TEXT, weekend_duration INTEGER, opening_weekend_start_date DATE,
      opening_weekend_end_date DATE, bucket_lower NUMERIC(24,2), bucket_upper NUMERIC(24,2),
      include_lower BOOLEAN NOT NULL DEFAULT TRUE, include_upper BOOLEAN NOT NULL DEFAULT FALSE,
      lower_unbounded BOOLEAN NOT NULL, upper_unbounded BOOLEAN NOT NULL, gross_unit TEXT NOT NULL DEFAULT 'dollars',
      resolution_source TEXT, parser_version TEXT NOT NULL, parse_confidence NUMERIC(6,5), parse_status TEXT NOT NULL,
      reviewer_override JSONB, review_notes TEXT);
    CREATE TABLE prediction_market_backtest.event_bucket_validations (
      event_id TEXT PRIMARY KEY REFERENCES prediction_market_backtest.polymarket_events(event_id), validation_status TEXT NOT NULL,
      validation_version TEXT NOT NULL, validation_errors JSONB NOT NULL DEFAULT '[]', validated_bucket_count INTEGER NOT NULL,
      validated_at TIMESTAMPTZ NOT NULL, reviewer_override JSONB);
    CREATE TABLE prediction_market_backtest.movie_matches (
      event_id TEXT PRIMARY KEY REFERENCES prediction_market_backtest.polymarket_events(event_id), movie_id BIGINT REFERENCES movies(movie_id),
      normalized_market_title TEXT, normalized_internal_title TEXT, release_date_distance INTEGER, title_similarity NUMERIC(7,6),
      semantic_compatibility JSONB, match_score NUMERIC(7,6), match_method TEXT, match_status TEXT NOT NULL,
      reviewer TEXT, reviewer_timestamp TIMESTAMPTZ, manual_override BOOLEAN NOT NULL DEFAULT FALSE, rejection_reason TEXT);
    CREATE TABLE prediction_market_backtest.order_book_events (
      book_event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, token_id TEXT NOT NULL, exchange_timestamp TIMESTAMPTZ,
      local_receipt_timestamp TIMESTAMPTZ NOT NULL, message_type TEXT NOT NULL, sequence_number BIGINT, book_hash TEXT,
      raw_json JSONB NOT NULL, UNIQUE(token_id, sequence_number, message_type));
    CREATE TABLE prediction_market_backtest.order_book_snapshots (
      snapshot_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, token_id TEXT NOT NULL, exchange_timestamp TIMESTAMPTZ,
      local_receipt_timestamp TIMESTAMPTZ NOT NULL, bids JSONB NOT NULL, asks JSONB NOT NULL, best_bid NUMERIC(20,10),
      best_ask NUMERIC(20,10), spread NUMERIC(20,10), cumulative_depth JSONB, tick_size NUMERIC(20,10),
      minimum_order_size NUMERIC(30,10), book_hash TEXT NOT NULL, reconstruction_status TEXT NOT NULL,
      UNIQUE(token_id, book_hash, local_receipt_timestamp));
    CREATE TABLE prediction_market_backtest.forecast_distributions (
      forecast_distribution_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, movie_id BIGINT NOT NULL REFERENCES movies(movie_id),
      forecast_regime TEXT NOT NULL, forecast_origin TEXT NOT NULL, information_cutoff_timestamp TIMESTAMPTZ NOT NULL,
      computation_started_at TIMESTAMPTZ, forecast_creation_timestamp TIMESTAMPTZ NOT NULL,
      forecast_available_timestamp TIMESTAMPTZ NOT NULL, model_artifact_version TEXT NOT NULL,
      distribution_artifact_version TEXT NOT NULL, simulator_version TEXT NOT NULL, random_seed BIGINT NOT NULL,
      number_of_draws INTEGER NOT NULL CHECK(number_of_draws >= 50000), residual_coordinate TEXT, residual_source TEXT,
      weighting_policy JSONB, fallback_policy JSONB, model_metadata JSONB, distribution_diagnostics JSONB,
      simulation_output_location TEXT, compressed_draws BYTEA, approved BOOLEAN NOT NULL DEFAULT FALSE,
      CHECK(simulation_output_location IS NOT NULL OR compressed_draws IS NOT NULL));
    CREATE TABLE prediction_market_backtest.bucket_probabilities (
      bucket_probability_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
      forecast_distribution_id BIGINT NOT NULL REFERENCES prediction_market_backtest.forecast_distributions(forecast_distribution_id),
      event_id TEXT NOT NULL REFERENCES prediction_market_backtest.polymarket_events(event_id),
      market_id TEXT NOT NULL REFERENCES prediction_market_backtest.polymarket_markets(market_id), model_probability NUMERIC(18,17) NOT NULL,
      simulation_count_within_bucket INTEGER NOT NULL, bucket_lower NUMERIC(24,2), bucket_upper NUMERIC(24,2),
      include_lower BOOLEAN NOT NULL, include_upper BOOLEAN NOT NULL, probability_generation_version TEXT NOT NULL,
      probability_vector_sum NUMERIC(18,17) NOT NULL, validation_status TEXT NOT NULL,
      CHECK(model_probability >= 0 AND model_probability <= 1), UNIQUE(forecast_distribution_id, market_id));
    CREATE TABLE prediction_market_backtest.forecast_book_pairings (
      pairing_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, forecast_distribution_id BIGINT NOT NULL REFERENCES prediction_market_backtest.forecast_distributions(forecast_distribution_id),
      token_id TEXT NOT NULL, book_snapshot_id BIGINT NOT NULL REFERENCES prediction_market_backtest.order_book_snapshots(snapshot_id),
      information_cutoff TIMESTAMPTZ NOT NULL, forecast_available_time TIMESTAMPTZ NOT NULL, book_exchange_time TIMESTAMPTZ,
      book_receipt_time TIMESTAMPTZ NOT NULL, assumed_execution_latency INTERVAL NOT NULL, forecast_age INTERVAL,
      book_age INTERVAL, pairing_rule_version TEXT NOT NULL, status TEXT NOT NULL, rejection_reason TEXT);
    CREATE TABLE prediction_market_backtest.paper_signals (
      signal_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, event_id TEXT NOT NULL, market_id TEXT NOT NULL, token_id TEXT NOT NULL,
      probability_source TEXT NOT NULL, model_probability NUMERIC(18,17), market_probability NUMERIC(18,17), blended_probability NUMERIC(18,17),
      trade_side TEXT NOT NULL, requested_risk_budget NUMERIC(20,4), requested_shares NUMERIC(30,10), edge_threshold NUMERIC(18,17),
      estimated_expected_value NUMERIC(20,6), estimated_return_on_risk NUMERIC(20,10), signal_policy_version TEXT NOT NULL,
      portfolio_decision TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE prediction_market_backtest.paper_trades (
      trade_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, signal_id BIGINT NOT NULL REFERENCES prediction_market_backtest.paper_signals(signal_id),
      filled_shares NUMERIC(30,10) NOT NULL, unfilled_shares NUMERIC(30,10) NOT NULL, vwap NUMERIC(20,10), maximum_fill_price NUMERIC(20,10),
      gross_premium NUMERIC(20,6) NOT NULL, fees NUMERIC(20,6) NOT NULL, fill_details JSONB NOT NULL, no_fill_reason TEXT,
      settlement_result TEXT, realized_pnl NUMERIC(20,6));
    CREATE TABLE prediction_market_backtest.portfolio_ledger (
      ledger_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, occurred_at TIMESTAMPTZ NOT NULL, strategy_version TEXT NOT NULL,
      signal_id BIGINT REFERENCES prediction_market_backtest.paper_signals(signal_id), trade_id BIGINT REFERENCES prediction_market_backtest.paper_trades(trade_id),
      cash_delta NUMERIC(20,6) NOT NULL, cash_balance NUMERIC(20,6) NOT NULL, inventory_delta JSONB NOT NULL,
      premium_at_risk NUMERIC(20,6) NOT NULL, realized_pnl NUMERIC(20,6) NOT NULL, metadata JSONB NOT NULL DEFAULT '{}');
    CREATE INDEX ON prediction_market_backtest.polymarket_markets(event_id);
    CREATE INDEX ON prediction_market_backtest.order_book_events(token_id, local_receipt_timestamp);
    CREATE INDEX ON prediction_market_backtest.order_book_snapshots(token_id, local_receipt_timestamp);
    CREATE INDEX ON prediction_market_backtest.forecast_distributions(movie_id, forecast_available_timestamp);
    """)


def downgrade() -> None:
    for table in TABLES:
        op.execute(f"DROP TABLE IF EXISTS prediction_market_backtest.{table} CASCADE")
    op.execute("DROP SCHEMA IF EXISTS prediction_market_backtest")

