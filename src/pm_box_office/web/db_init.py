"""One-time database initialization for the web process."""

from __future__ import annotations

import threading
from typing import Any

from pm_box_office.sources.amc import db
from pm_box_office.orchestration import repository


_initialized = False
_lock = threading.Lock()
_REQUIRED_TABLES = {
    "amc_theatres",
    "amc_theatre_sample_sets",
    "amc_theatre_sample_members",
    "amc_movies",
    "amc_showtimes",
    "collection_campaigns",
    "campaign_movies",
    "collection_runs",
    "collection_tasks",
    "amc_seat_snapshots",
    "ingest_sources",
    "ingest_runs",
    "ingest_run_logs",
    "source_freshness",
}
_REQUIRED_COLUMNS = {
    "amc_showtimes": {"showtime_id", "amc_movie_id", "exhibition_date", "starts_at_utc"},
    "amc_theatre_sample_sets": {"sample_set_id", "sample_key", "sample_size", "status"},
    "amc_theatre_sample_members": {
        "sample_set_id",
        "amc_theatre_id",
        "inclusion_probability",
        "analysis_weight",
    },
    "collection_tasks": {
        "task_id",
        "run_id",
        "task_type",
        "showtime_id",
        "scheduled_for",
        "status",
        "priority",
        "worker_id",
        "last_error_type",
        "last_error_message",
    },
    "amc_seat_snapshots": {"showtime_id", "target_offset_minutes", "scheduled_for", "lateness_seconds"},
    "ingest_sources": {"source_key", "display_name", "command", "default_args"},
}


def ensure_initialized(conn: Any) -> None:
    global _initialized
    if _initialized:
        return
    with _lock:
        if _initialized:
            return
        if runtime_schema_ready(conn):
            repository.seed_sources(conn)
            _initialized = True
            return
        db.initialize_amc_database(conn)
        repository.initialize_orchestration_database(conn)
        repository.seed_sources(conn)
        _initialized = True


def runtime_schema_ready(conn: Any) -> bool:
    table_rows = conn.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = current_schema()
        """
    ).fetchall()
    existing_tables = {str(row[0]) for row in table_rows}
    if not _REQUIRED_TABLES.issubset(existing_tables):
        return False
    for table_name, required_columns in _REQUIRED_COLUMNS.items():
        column_rows = conn.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = %s
            """,
            (table_name,),
        ).fetchall()
        existing_columns = {str(row[0]) for row in column_rows}
        if not required_columns.issubset(existing_columns):
            return False
    return True
