"""audience snapshots

Revision ID: 004_audience_snapshots
Revises: 003_analysis_views
Create Date: 2026-06-30
"""

from __future__ import annotations

from alembic import op

from pm_box_office.sources.audience import ingest


revision = "004_audience_snapshots"
down_revision = "003_analysis_views"
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
    ingest.initialize_database(AlembicConnectionAdapter(op.get_bind()))


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS analytics.box_office_audience_panel_v1")
    op.execute("DROP VIEW IF EXISTS analytics.movie_audience_daily_features_v1")
    for table in (
        "audience_ingest_state",
        "letterboxd_film_snapshots",
        "imdb_title_snapshots",
        "movie_letterboxd_films",
        "letterboxd_films",
        "movie_imdb_titles",
        "imdb_titles",
        "the_numbers_release_schedule",
    ):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")


def split_sql_script(sql: str) -> list[str]:
    return [statement.strip() for statement in sql.split(";") if statement.strip()]
