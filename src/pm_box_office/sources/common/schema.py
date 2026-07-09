"""Shared schema-initialization helpers for source ingests."""

from __future__ import annotations

from typing import Any


SCHEMA_INIT_LOCK_KEY = "pm_box_office_source_schema_init"


def acquire_schema_init_lock(conn: Any) -> None:
    """Serialize DDL-heavy ingest initialization across concurrent Run All jobs."""
    conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (SCHEMA_INIT_LOCK_KEY,))
