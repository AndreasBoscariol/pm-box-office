from __future__ import annotations

import unittest
from unittest.mock import patch

from pm_box_office.web import db_init


class FakeCursor:
    def __init__(self, rows: list[tuple[str]]) -> None:
        self.rows = rows

    def fetchall(self) -> list[tuple[str]]:
        return self.rows


class FakeConnection:
    def __init__(self, *, tables: set[str], columns: dict[str, set[str]]) -> None:
        self.tables = tables
        self.columns = columns

    def execute(self, _sql: str, params: tuple[str, ...] | None = None) -> FakeCursor:
        if params:
            table_name = params[0]
            return FakeCursor([(column,) for column in self.columns.get(table_name, set())])
        return FakeCursor([(table,) for table in self.tables])


class DbInitTests(unittest.TestCase):
    def setUp(self) -> None:
        db_init._initialized = False

    def tearDown(self) -> None:
        db_init._initialized = False

    def test_runtime_schema_ready_accepts_existing_required_tables_and_columns(self) -> None:
        conn = FakeConnection(
            tables=set(db_init._REQUIRED_TABLES),
            columns={table: set(columns) for table, columns in db_init._REQUIRED_COLUMNS.items()},
        )

        self.assertTrue(db_init.runtime_schema_ready(conn))

    def test_ensure_initialized_skips_ddl_when_runtime_schema_is_ready(self) -> None:
        conn = FakeConnection(
            tables=set(db_init._REQUIRED_TABLES),
            columns={table: set(columns) for table, columns in db_init._REQUIRED_COLUMNS.items()},
        )
        with (
            patch.object(db_init.db, "initialize_amc_database") as initialize_amc_database,
            patch.object(db_init.repository, "initialize_orchestration_database") as initialize_orchestration_database,
            patch.object(db_init.repository, "seed_sources") as seed_sources,
        ):
            db_init.ensure_initialized(conn)

        initialize_amc_database.assert_not_called()
        initialize_orchestration_database.assert_not_called()
        seed_sources.assert_called_once_with(conn)
        self.assertTrue(db_init._initialized)

    def test_ensure_initialized_runs_full_init_when_schema_is_missing(self) -> None:
        conn = FakeConnection(tables=set(), columns={})
        with (
            patch.object(db_init.db, "initialize_amc_database") as initialize_amc_database,
            patch.object(db_init.repository, "initialize_orchestration_database") as initialize_orchestration_database,
            patch.object(db_init.repository, "seed_sources") as seed_sources,
        ):
            db_init.ensure_initialized(conn)

        initialize_amc_database.assert_called_once_with(conn)
        initialize_orchestration_database.assert_called_once_with(conn)
        seed_sources.assert_called_once_with(conn)
        self.assertTrue(db_init._initialized)


if __name__ == "__main__":
    unittest.main()
