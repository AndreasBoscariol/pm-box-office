"""canonical movie source backfills

Revision ID: 012_canonical_movie_source_backfills
Revises: 011_movie_identity_source_map
Create Date: 2026-07-06
"""

from __future__ import annotations

from typing import Any

from alembic import op

from pm_box_office.domain import movies


revision = "012_canonical_movie_source_backfills"
down_revision = "011_movie_identity_source_map"
branch_labels = None
depends_on = None


class AlembicConnectionAdapter:
    def __init__(self, connection: object) -> None:
        self.connection = connection

    def execute(self, sql: str, params: Any | None = None) -> object:
        if params is None:
            return self.connection.exec_driver_sql(sql)  # type: ignore[attr-defined]
        return self.connection.exec_driver_sql(sql, params)  # type: ignore[attr-defined]

    def executescript(self, sql: str) -> None:
        for statement in split_sql_script(sql):
            self.execute(statement)


def upgrade() -> None:
    movies.ensure_movie_identity_schema(AlembicConnectionAdapter(op.get_bind()), backfill_sources=True)


def downgrade() -> None:
    for index in (
        "idx_collection_tasks_movie_id",
        "idx_amc_showtimes_movie_id_date",
        "idx_amc_movies_movie_id",
        "idx_boxofficereport_predictions_movie_id",
        "idx_boxofficepro_weekend_predictions_movie_id",
        "idx_daily_chart_pages_movie_id",
        "idx_tn_release_schedule_movie_id",
        "idx_movie_source_ids_source_title",
        "idx_movie_source_ids_source_status",
        "idx_movie_source_ids_movie",
    ):
        op.execute(f"DROP INDEX IF EXISTS {index}")
    for table in (
        "amc_collection_diagnostic_events",
        "collection_tasks",
        "campaign_movies",
        "amc_showtimes",
        "amc_movies",
        "boxofficereport_weekend_predictions",
        "boxofficepro_weekend_predictions",
        "daily_chart_pages",
        "the_numbers_release_schedule",
    ):
        op.execute(f"ALTER TABLE IF EXISTS {table} DROP COLUMN IF EXISTS movie_id")
    op.execute(
        """
        DELETE FROM movie_source_ids
        WHERE source IN ('amc', 'boxofficereport', 'imdb', 'letterboxd', 'wikipedia')
        """
    )


def split_sql_script(sql: str) -> list[str]:
    return [statement.strip() for statement in sql.split(";") if statement.strip()]
