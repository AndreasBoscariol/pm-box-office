"""ingest orchestration

Revision ID: 005_ingest_orchestration
Revises: 004_audience_snapshots
Create Date: 2026-06-30
"""

from __future__ import annotations

from alembic import op

from pm_box_office.orchestration import repository


revision = "005_ingest_orchestration"
down_revision = "004_audience_snapshots"
branch_labels = None
depends_on = None


class AlembicConnectionAdapter:
    def __init__(self, connection: object) -> None:
        self.connection = connection

    def execute(self, sql: str, params: object | None = None) -> object:
        return self.connection.exec_driver_sql(sql, params)  # type: ignore[attr-defined]

    def executescript(self, sql: str) -> None:
        for statement in split_sql_script(sql):
            self.execute(statement)


def upgrade() -> None:
    conn = AlembicConnectionAdapter(op.get_bind())
    repository.initialize_orchestration_database(conn)
    repository.seed_sources(conn)


def downgrade() -> None:
    for table in (
        "source_freshness",
        "ingest_run_logs",
        "ingest_runs",
        "ingest_sources",
    ):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")


def split_sql_script(sql: str) -> list[str]:
    return [statement.strip() for statement in sql.split(";") if statement.strip()]
