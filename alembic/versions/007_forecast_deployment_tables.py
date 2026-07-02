"""forecast deployment tables

Revision ID: 007_forecast_deployment_tables
Revises: 006_the_numbers_movie_metadata
Create Date: 2026-07-02
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "007_forecast_deployment_tables"
down_revision = "006_the_numbers_movie_metadata"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "forecast_model_versions",
        sa.Column("model_version_id", sa.BigInteger(), sa.Identity(always=False), primary_key=True),
        sa.Column("model_version", sa.Text(), nullable=False, unique=True),
        sa.Column("registry_version", sa.Text(), nullable=False),
        sa.Column("model_family", sa.Text(), nullable=False),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("target_type", sa.Text(), nullable=False),
        sa.Column("bop_segment", sa.Text(), nullable=False),
        sa.Column("forecast_state", sa.Text(), nullable=False),
        sa.Column("deployment_status", sa.Text(), nullable=False),
        sa.Column("train_start_year", sa.Integer()),
        sa.Column("train_end_year", sa.Integer()),
        sa.Column("train_policy", sa.Text(), nullable=False, server_default=""),
        sa.Column("train_bop_floor_usd", sa.Numeric(14, 2)),
        sa.Column("live_bop_floor_usd", sa.Numeric(14, 2)),
        sa.Column("feature_set", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="[]"),
        sa.Column("registry_reason", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_table(
        "forecast_feature_snapshots",
        sa.Column("feature_snapshot_id", sa.BigInteger(), sa.Identity(always=False), primary_key=True),
        sa.Column("movie_id", sa.BigInteger(), sa.ForeignKey("movies.movie_id", ondelete="CASCADE"), nullable=False),
        sa.Column("forecast_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("forecast_state", sa.Text(), nullable=False),
        sa.Column("target_type", sa.Text(), nullable=False),
        sa.Column("target_start_date", sa.Date(), nullable=False),
        sa.Column("target_end_date", sa.Date(), nullable=False),
        sa.Column("target_days", sa.Integer(), nullable=False),
        sa.Column("bop_midpoint_usd", sa.Numeric(14, 2)),
        sa.Column("bop_prediction_id", sa.BigInteger()),
        sa.Column("bop_published_date", sa.Date()),
        sa.Column("bop_segment", sa.Text(), nullable=False),
        sa.Column("known_actual_offsets", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="[]"),
        sa.Column("known_actual_gross_so_far", sa.Numeric(14, 2), nullable=False, server_default="0"),
        sa.Column("source_availability", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("feature_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index(
        "ix_forecast_feature_snapshots_movie_target_time",
        "forecast_feature_snapshots",
        ["movie_id", "target_type", "forecast_timestamp"],
    )
    op.create_table(
        "forecast_predictions",
        sa.Column("forecast_prediction_id", sa.BigInteger(), sa.Identity(always=False), primary_key=True),
        sa.Column(
            "feature_snapshot_id",
            sa.BigInteger(),
            sa.ForeignKey("forecast_feature_snapshots.feature_snapshot_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "model_version_id",
            sa.BigInteger(),
            sa.ForeignKey("forecast_model_versions.model_version_id", ondelete="RESTRICT"),
        ),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("model_family", sa.Text(), nullable=False),
        sa.Column("deployment_status", sa.Text(), nullable=False),
        sa.Column("point_forecast_usd", sa.Numeric(14, 2)),
        sa.Column("lower_80_usd", sa.Numeric(14, 2)),
        sa.Column("upper_80_usd", sa.Numeric(14, 2)),
        sa.Column("remaining_gross_forecast_usd", sa.Numeric(14, 2)),
        sa.Column("prediction_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index(
        "ix_forecast_predictions_snapshot_created",
        "forecast_predictions",
        ["feature_snapshot_id", "created_at"],
    )
    op.create_table(
        "forecast_backtest_results",
        sa.Column("forecast_backtest_result_id", sa.BigInteger(), sa.Identity(always=False), primary_key=True),
        sa.Column("registry_version", sa.Text(), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("target_type", sa.Text(), nullable=False),
        sa.Column("bop_segment", sa.Text(), nullable=False),
        sa.Column("forecast_state", sa.Text(), nullable=False),
        sa.Column("test_year", sa.Integer(), nullable=False),
        sa.Column("holdout_n", sa.Integer(), nullable=False),
        sa.Column("mae_usd", sa.Numeric(14, 2)),
        sa.Column("rmse_usd", sa.Numeric(14, 2)),
        sa.Column("mape", sa.Numeric(10, 6)),
        sa.Column("smape", sa.Numeric(10, 6)),
        sa.Column("mean_absolute_log_error", sa.Numeric(10, 6)),
        sa.Column("baseline_model", sa.Text(), nullable=False, server_default=""),
        sa.Column("baseline_lift_mae_usd", sa.Numeric(14, 2)),
        sa.Column("low_sample_flag", sa.Boolean(), nullable=False, server_default=sa.text("FALSE")),
        sa.Column("metrics_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index(
        "ix_forecast_backtest_results_registry_model",
        "forecast_backtest_results",
        ["registry_version", "model_version", "test_year"],
    )


def downgrade() -> None:
    op.drop_index("ix_forecast_backtest_results_registry_model", table_name="forecast_backtest_results")
    op.drop_table("forecast_backtest_results")
    op.drop_index("ix_forecast_predictions_snapshot_created", table_name="forecast_predictions")
    op.drop_table("forecast_predictions")
    op.drop_index("ix_forecast_feature_snapshots_movie_target_time", table_name="forecast_feature_snapshots")
    op.drop_table("forecast_feature_snapshots")
    op.drop_table("forecast_model_versions")
