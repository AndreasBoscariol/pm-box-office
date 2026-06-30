from __future__ import annotations

import os
import unittest
import uuid
from typing import Any

from pm_box_office.db.connection import connect_database, database_url_from_env


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def make_isolated_postgres_schema() -> tuple[Any, str]:
    database_url = os.environ.get("TEST_DATABASE_URL") or database_url_from_env()
    if not database_url:
        raise unittest.SkipTest("Set TEST_DATABASE_URL or DATABASE_URL to run PostgreSQL integration tests.")
    try:
        conn = connect_database(database_url)
    except SystemExit as exc:
        raise unittest.SkipTest(str(exc)) from exc
    except Exception as exc:
        raise unittest.SkipTest(f"Could not connect to PostgreSQL test database: {exc}") from exc

    schema = f"test_{uuid.uuid4().hex}"
    conn.execute(f"CREATE SCHEMA {quote_ident(schema)}")
    conn.execute(f"SET search_path TO {quote_ident(schema)}")
    conn.commit()
    return conn, schema


def drop_isolated_postgres_schema(conn: Any, schema: str) -> None:
    try:
        conn.execute(f"DROP SCHEMA IF EXISTS {quote_ident(schema)} CASCADE")
        conn.commit()
    finally:
        conn.close()
