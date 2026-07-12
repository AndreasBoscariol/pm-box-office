"""Immutable provenance snapshots for the live forecasting pipeline.

Collectors own raw source tables.  This module is the narrow boundary between
those tables and model execution: each completed source run is recorded as an
ingest event, and every forecast refresh captures the source state it used.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import uuid
from dataclasses import dataclass
from typing import Any


INGEST_EVENT_TABLE = "analytics.forecast_ingest_events"
SNAPSHOT_TABLE = "analytics.forecast_input_snapshots"


@dataclass(frozen=True)
class InputSnapshot:
    snapshot_id: str
    release_run_id: int
    movie_id: int | None
    as_of_utc: dt.datetime
    availability: dict[str, dict[str, object]]
    quality_bucket: str


def relation_exists(conn: Any, relation: str) -> bool:
    row = conn.execute("SELECT to_regclass(%s) IS NOT NULL", (relation,)).fetchone()
    return bool(row and row[0])


def ensure_input_snapshot_tables(conn: Any) -> None:
    conn.executescript(
        f"""
        CREATE SCHEMA IF NOT EXISTS analytics;

        CREATE TABLE IF NOT EXISTS {INGEST_EVENT_TABLE} (
            ingest_event_id TEXT PRIMARY KEY,
            source_key TEXT NOT NULL,
            event_type TEXT NOT NULL,
            source_run_id UUID,
            available_at_utc TIMESTAMPTZ NOT NULL,
            payload JSONB NOT NULL DEFAULT '{{}}',
            payload_hash TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (source_key, event_type, source_run_id, payload_hash)
        );
        CREATE INDEX IF NOT EXISTS idx_forecast_ingest_events_source_time
            ON {INGEST_EVENT_TABLE} (source_key, available_at_utc DESC);

        CREATE TABLE IF NOT EXISTS {SNAPSHOT_TABLE} (
            input_snapshot_id TEXT PRIMARY KEY,
            release_run_id BIGINT NOT NULL,
            movie_id BIGINT,
            model_version TEXT NOT NULL,
            as_of_utc TIMESTAMPTZ NOT NULL,
            availability JSONB NOT NULL,
            quality_bucket TEXT NOT NULL,
            snapshot_hash TEXT NOT NULL UNIQUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_forecast_input_snapshots_release_time
            ON {SNAPSHOT_TABLE} (release_run_id, model_version, as_of_utc DESC);
        """
    )


def _canonical_payload(value: dict[str, object]) -> tuple[str, str]:
    raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return raw, hashlib.sha256(raw.encode("utf-8")).hexdigest()


def record_ingest_event(
    conn: Any,
    *,
    source_key: str,
    source_run_id: str | None,
    payload: dict[str, object],
    available_at_utc: dt.datetime | None = None,
) -> str:
    """Append a deduplicated, auditable event after a collector succeeds."""
    ensure_input_snapshot_tables(conn)
    available = available_at_utc or dt.datetime.now(dt.timezone.utc)
    _, payload_hash = _canonical_payload(payload)
    event_id = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"pm-box-office-ingest|{source_key}|{source_run_id or ''}|{payload_hash}",
    ).hex
    conn.execute(
        f"""
        INSERT INTO {INGEST_EVENT_TABLE} (
            ingest_event_id, source_key, event_type, source_run_id,
            available_at_utc, payload, payload_hash
        ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
        ON CONFLICT (source_key, event_type, source_run_id, payload_hash) DO NOTHING
        """,
        (event_id, source_key, "source_sync_completed", source_run_id, available, json.dumps(payload, default=str), payload_hash),
    )
    return event_id


def _source_state(conn: Any, *, relation: str, predicate: str, params: tuple[object, ...], timestamp: str | None) -> dict[str, object]:
    if not relation_exists(conn, relation):
        return {"available": False, "records": 0, "latest_at": None}
    latest_sql = f"MAX({timestamp})" if timestamp else "NULL"
    row = conn.execute(
        f"SELECT COUNT(*)::integer, {latest_sql} FROM {relation} WHERE {predicate}", params
    ).fetchone()
    return {"available": bool(row and int(row[0] or 0) > 0), "records": int(row[0] or 0) if row else 0, "latest_at": row[1] if row else None}


def capture_input_snapshot(
    conn: Any,
    *,
    release_run_id: int,
    movie_id: int | None,
    model_version: str,
    as_of_utc: dt.datetime,
) -> InputSnapshot:
    """Capture the source availability visible to one model refresh.

    The snapshot is deliberately metadata-sized: raw observations remain in
    their source tables, while this row records the exact source state and
    timestamp that model/UI provenance need.
    """
    ensure_input_snapshot_tables(conn)
    estimates: dict[str, dict[str, object]] = {}
    for source, table in {
        "boxofficepro": "boxofficepro_weekend_predictions",
        "boxofficereport": "boxofficereport_weekend_predictions",
        "boxofficetheory": "boxofficetheory_predictions",
        "boxofficetheory_substack": "boxofficetheory_substack_predictions",
    }.items():
        estimates[source] = _source_state(
            conn, relation=table, predicate="movie_id = %s", params=(movie_id,), timestamp=None
        ) if movie_id is not None else {"available": False, "records": 0, "latest_at": None}
    availability = {
        "estimates": {
            "available": any(bool(item["available"]) for item in estimates.values()),
            "records": sum(int(item["records"]) for item in estimates.values()),
            "sources": estimates,
        },
        "daily_actuals": _source_state(
            conn, relation="daily_box_office", predicate="release_run_id = %s", params=(release_run_id,), timestamp="fetched_at"
        ),
        "amc_seats": _source_state(
            conn,
            relation="amc_seat_snapshots",
            predicate="showtime_id IN (SELECT showtime_id FROM amc_showtimes WHERE movie_id = %s)",
            params=(movie_id,), timestamp="fetched_at",
        ) if movie_id is not None else {"available": False, "records": 0, "latest_at": None},
        "polymarket": _source_state(
            conn,
            relation="prediction_market_backtest.movie_matches",
            predicate="movie_id = %s AND COALESCE(match_status, '') NOT ILIKE 'rejected%%'",
            params=(movie_id,), timestamp=None,
        ) if movie_id is not None else {"available": False, "records": 0, "latest_at": None},
    }
    source_count = int(bool(availability["estimates"]["available"])) + sum(
        int(bool(availability[name]["available"])) for name in ("daily_actuals", "amc_seats", "polymarket")
    )
    quality = "four_source" if source_count == 4 else "multi_source" if source_count >= 2 else "single_source" if source_count else "no_source"
    payload = {
        "release_run_id": release_run_id,
        "movie_id": movie_id,
        "model_version": model_version,
        "as_of_utc": as_of_utc,
        "availability": availability,
        "quality_bucket": quality,
    }
    _, snapshot_hash = _canonical_payload(payload)
    snapshot_id = uuid.uuid5(uuid.NAMESPACE_URL, f"pm-box-office-input|{snapshot_hash}").hex
    conn.execute(
        f"""
        INSERT INTO {SNAPSHOT_TABLE} (
            input_snapshot_id, release_run_id, movie_id, model_version,
            as_of_utc, availability, quality_bucket, snapshot_hash
        ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s)
        ON CONFLICT (snapshot_hash) DO NOTHING
        """,
        (snapshot_id, release_run_id, movie_id, model_version, as_of_utc, json.dumps(availability, default=str), quality, snapshot_hash),
    )
    return InputSnapshot(snapshot_id, release_run_id, movie_id, as_of_utc, availability, quality)


def latest_input_snapshot(conn: Any, *, release_run_id: int, model_version: str | None) -> dict[str, object] | None:
    if not relation_exists(conn, SNAPSHOT_TABLE):
        return None
    predicates = ["release_run_id = %s"]
    params: list[object] = [release_run_id]
    if model_version:
        predicates.append("model_version = %s")
        params.append(model_version)
    cursor = conn.execute(
        f"""
        SELECT input_snapshot_id, as_of_utc, availability, quality_bucket, model_version
        FROM {SNAPSHOT_TABLE}
        WHERE {' AND '.join(predicates)}
        ORDER BY as_of_utc DESC, created_at DESC
        LIMIT 1
        """,
        tuple(params),
    )
    row = cursor.fetchone()
    if not row:
        return None
    availability = row[2]
    if isinstance(availability, str):
        availability = json.loads(availability)
    return {"input_snapshot_id": row[0], "as_of_utc": row[1], "availability": availability or {}, "quality_bucket": row[3], "model_version": row[4]}
