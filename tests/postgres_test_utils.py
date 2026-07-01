from __future__ import annotations

import os
import unittest
import urllib.parse
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


def url_with_search_path(database_url: str, schema: str) -> str:
    parsed = urllib.parse.urlsplit(database_url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query.append(("options", f"-csearch_path={schema}"))
    return urllib.parse.urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urllib.parse.urlencode(query),
            parsed.fragment,
        )
    )


def drop_isolated_postgres_schema(conn: Any, schema: str) -> None:
    try:
        conn.execute(f"DROP SCHEMA IF EXISTS {quote_ident(schema)} CASCADE")
        conn.commit()
    finally:
        conn.close()
