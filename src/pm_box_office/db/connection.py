"""PostgreSQL helpers shared by ingest and analysis scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from pm_box_office.config import REPO_ROOT


DEFAULT_DATABASE_URL_ENV = "DATABASE_URL"


class PostgresConnection:
    """Small wrapper around psycopg connections with script execution helpers."""

    def __init__(self, raw: Any) -> None:
        self.raw = raw

    def execute(self, sql: str, params: Iterable[Any] | None = None) -> Any:
        if params is None:
            return self.raw.execute(sql)
        return self.raw.execute(sql, tuple(params))

    def executemany(self, sql: str, rows: Iterable[Iterable[Any]]) -> Any:
        cursor = self.raw.cursor()
        try:
            return cursor.executemany(sql, [tuple(row) for row in rows])
        finally:
            cursor.close()

    def executescript(self, sql: str) -> None:
        for statement in split_sql_script(sql):
            self.execute(statement)

    def commit(self) -> None:
        self.raw.commit()

    def rollback(self) -> None:
        self.raw.rollback()

    def close(self) -> None:
        self.raw.close()


def database_url_from_env() -> str | None:
    load_dotenv()
    return os.environ.get(DEFAULT_DATABASE_URL_ENV) or os.environ.get("POSTGRES_DSN")


def connect_database(database_url: str | None = None) -> PostgresConnection:
    url = database_url or database_url_from_env()
    if not url:
        raise SystemExit(
            f"Set {DEFAULT_DATABASE_URL_ENV} or POSTGRES_DSN to a PostgreSQL connection URL, "
            "or copy `.env.example` to `.env`. See docs/postgres_setup.md."
        )
    validate_postgres_url(url)
    try:
        import psycopg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on local env.
        raise SystemExit(
            "PostgreSQL support requires psycopg. Install it with `python3 -m pip install -r requirements.txt`."
        ) from exc
    return PostgresConnection(psycopg.connect(url))


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def validate_postgres_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"postgresql", "postgres"}:
        raise SystemExit(
            "Database URL must be a PostgreSQL URL, e.g. "
            "`postgresql://user:password@localhost:5432/pm_box_office`."
        )


def insert_ignore_sql(table: str, columns: list[str]) -> str:
    placeholders = ", ".join("%s" for _ in columns)
    column_list = ", ".join(columns)
    if table:
        return f"INSERT INTO {table} ({column_list}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"
    raise ValueError("table is required")


def split_sql_script(sql: str) -> list[str]:
    return [statement.strip() for statement in sql.split(";") if statement.strip()]


def table_names(conn: Any) -> list[str]:
    rows = conn.execute(
        """
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = current_schema()
        ORDER BY tablename
        """
    ).fetchall()
    return [str(row[0]) for row in rows]
