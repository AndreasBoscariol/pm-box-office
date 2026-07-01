"""amc core tables

Revision ID: 001_amc_core_tables
Revises:
Create Date: 2026-06-30
"""

from __future__ import annotations

from alembic import op

from pm_box_office.sources.amc import db


revision = "001_amc_core_tables"
down_revision = None
branch_labels = None
depends_on = None


class AlembicConnectionAdapter:
    def __init__(self, connection: object) -> None:
        self.connection = connection

    def execute(self, sql: str, params: object | None = None) -> object:
        if params is not None:
            raise NotImplementedError("Alembic initializer does not use bound parameters.")
        return self.connection.exec_driver_sql(sql)  # type: ignore[attr-defined]

    def executescript(self, sql: str) -> None:
        for statement in split_sql_script(sql):
            self.execute(statement)


def upgrade() -> None:
    db.initialize_amc_database(AlembicConnectionAdapter(op.get_bind()))


def downgrade() -> None:
    op.execute("DROP SCHEMA IF EXISTS analytics CASCADE")
    for table in (
        "movie_day_estimates",
        "amc_seat_snapshots",
        "collection_tasks",
        "collection_runs",
        "campaign_movies",
        "collection_campaigns",
        "amc_showtimes",
        "movie_source_ids",
        "movies",
        "amc_theatre_sample_members",
        "amc_theatre_sample_sets",
        "amc_movies",
        "amc_theatres",
    ):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")


def split_sql_script(sql: str) -> list[str]:
    return [statement.strip() for statement in sql.split(";") if statement.strip()]
