"""Database DDL and persistence helpers for production forecast tables."""

from __future__ import annotations

import uuid
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable

from .constants import COMPONENT_TABLE, EMISSION_COMPONENT_TABLE, EMISSION_CONFLICT_TABLE, EMISSION_TABLE, FORECAST_TABLE, RUN_TABLE

FORECAST_COLUMNS = [
    "forecast_id",
    "run_id",
    "model_version",
    "movie_id",
    "release_run_id",
    "title",
    "opening_weekend_start",
    "regime",
    "origin_key",
    "origin_day",
    "forecast_origin_local",
    "forecast_origin_utc",
    "as_of_utc",
    "target",
    "point_usd",
    "lo80_usd",
    "hi80_usd",
    "lo95_usd",
    "hi95_usd",
    "point_model",
    "interval_model",
    "distribution_payload",
    "component_source",
    "feature_quality_bucket",
    "source_count",
    "estimate_sources",
    "amc_coverage",
    "amc_snapshot_count",
    "amc_lateness_p50_minutes",
    "actual_usd",
    "log_error",
    "abs_pct_error",
    "is_live",
    "is_backtest",
    "created_at",
]

COMPONENT_COLUMNS = [
    "forecast_id",
    "component_day",
    "component_type",
    "component_point_usd",
    "component_sigma_log",
    "component_lo80_usd",
    "component_hi80_usd",
    "component_lo95_usd",
    "component_hi95_usd",
    "component_model",
    "component_source",
    "component_notes",
]

EMISSION_FORECAST_COLUMNS = [column for column in FORECAST_COLUMNS if column != "forecast_id"]

EMISSION_COLUMNS = [
    "emitted_forecast_id",
    "forecast_id",
    "forecast_generated_at",
    "forecast_role",
    "payload_hash",
    "production_policy_label",
    *EMISSION_FORECAST_COLUMNS,
]

EMISSION_COMPONENT_COLUMNS = [
    "emitted_forecast_id",
    *COMPONENT_COLUMNS,
]


def create_forecast_tables_sql() -> str:
    return f"""
    CREATE SCHEMA IF NOT EXISTS analytics;

    CREATE TABLE IF NOT EXISTS {RUN_TABLE} (
        run_id text PRIMARY KEY,
        model_version text NOT NULL,
        manifest_path text,
        mode text,
        as_of_utc timestamptz,
        created_at timestamptz NOT NULL DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS {FORECAST_TABLE} (
        forecast_id text PRIMARY KEY,
        run_id text NOT NULL,
        model_version text NOT NULL,
        movie_id bigint NOT NULL,
        release_run_id bigint NOT NULL,
        title text NOT NULL,
        opening_weekend_start date NOT NULL,
        regime text NOT NULL,
        origin_key text NOT NULL,
        origin_day integer,
        forecast_origin_local timestamptz NOT NULL,
        forecast_origin_utc timestamptz NOT NULL,
        as_of_utc timestamptz NOT NULL,
        target text NOT NULL,
        point_usd numeric,
        lo80_usd numeric,
        hi80_usd numeric,
        lo95_usd numeric,
        hi95_usd numeric,
        point_model text,
        interval_model text,
        distribution_payload jsonb,
        component_source text,
        feature_quality_bucket text,
        source_count integer,
        estimate_sources text,
        amc_coverage numeric,
        amc_snapshot_count integer,
        amc_lateness_p50_minutes numeric,
        actual_usd numeric,
        log_error numeric,
        abs_pct_error numeric,
        is_live boolean NOT NULL DEFAULT false,
        is_backtest boolean NOT NULL DEFAULT false,
        created_at timestamptz NOT NULL DEFAULT now(),
        UNIQUE (model_version, release_run_id, origin_key, target)
    );

    CREATE TABLE IF NOT EXISTS {COMPONENT_TABLE} (
        forecast_id text NOT NULL REFERENCES {FORECAST_TABLE}(forecast_id) ON DELETE CASCADE,
        component_day text NOT NULL,
        component_type text NOT NULL,
        component_point_usd numeric,
        component_sigma_log numeric,
        component_lo80_usd numeric,
        component_hi80_usd numeric,
        component_lo95_usd numeric,
        component_hi95_usd numeric,
        component_model text,
        component_source text,
        component_notes text,
        PRIMARY KEY (forecast_id, component_day)
    );

    CREATE TABLE IF NOT EXISTS {EMISSION_TABLE} (
        emitted_forecast_id text PRIMARY KEY,
        forecast_id text NOT NULL,
        forecast_generated_at timestamptz NOT NULL,
        forecast_role text NOT NULL DEFAULT 'production',
        payload_hash text NOT NULL,
        production_policy_label text,
        run_id text NOT NULL,
        model_version text NOT NULL,
        movie_id bigint NOT NULL,
        release_run_id bigint NOT NULL,
        title text NOT NULL,
        opening_weekend_start date NOT NULL,
        regime text NOT NULL,
        origin_key text NOT NULL,
        origin_day integer,
        forecast_origin_local timestamptz NOT NULL,
        forecast_origin_utc timestamptz NOT NULL,
        as_of_utc timestamptz NOT NULL,
        target text NOT NULL,
        point_usd numeric,
        lo80_usd numeric,
        hi80_usd numeric,
        lo95_usd numeric,
        hi95_usd numeric,
        point_model text,
        interval_model text,
        distribution_payload jsonb,
        component_source text,
        feature_quality_bucket text,
        source_count integer,
        estimate_sources text,
        amc_coverage numeric,
        amc_snapshot_count integer,
        amc_lateness_p50_minutes numeric,
        actual_usd numeric,
        first_final_actual_usd numeric,
        latest_final_actual_usd numeric,
        log_error numeric,
        abs_pct_error numeric,
        is_live boolean NOT NULL DEFAULT false,
        is_backtest boolean NOT NULL DEFAULT false,
        created_at timestamptz NOT NULL DEFAULT now(),
        UNIQUE (model_version, release_run_id, origin_key, target, as_of_utc, forecast_role)
    );

    CREATE TABLE IF NOT EXISTS {EMISSION_COMPONENT_TABLE} (
        emitted_forecast_id text NOT NULL REFERENCES {EMISSION_TABLE}(emitted_forecast_id) ON DELETE CASCADE,
        forecast_id text NOT NULL,
        component_day text NOT NULL,
        component_type text NOT NULL,
        component_point_usd numeric,
        component_sigma_log numeric,
        component_lo80_usd numeric,
        component_hi80_usd numeric,
        component_lo95_usd numeric,
        component_hi95_usd numeric,
        component_model text,
        component_source text,
        component_notes text,
        PRIMARY KEY (emitted_forecast_id, forecast_id, component_day)
    );

    CREATE TABLE IF NOT EXISTS {EMISSION_CONFLICT_TABLE} (
        conflict_id text PRIMARY KEY,
        detected_at timestamptz NOT NULL DEFAULT now(),
        emitted_forecast_id text NOT NULL,
        forecast_id text NOT NULL,
        model_version text NOT NULL,
        release_run_id bigint NOT NULL,
        origin_key text NOT NULL,
        target text NOT NULL,
        as_of_utc timestamptz NOT NULL,
        forecast_role text NOT NULL,
        existing_payload_hash text NOT NULL,
        incoming_payload_hash text NOT NULL,
        reason text NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_movie_opening_weekend_forecasts_movie
        ON {FORECAST_TABLE}(movie_id, model_version, origin_key, target);
    CREATE INDEX IF NOT EXISTS idx_movie_opening_weekend_forecasts_release
        ON {FORECAST_TABLE}(release_run_id, opening_weekend_start);
    CREATE INDEX IF NOT EXISTS idx_movie_forecast_emissions_key
        ON {EMISSION_TABLE}(model_version, release_run_id, origin_key, target, forecast_generated_at);
    CREATE INDEX IF NOT EXISTS idx_movie_forecast_emissions_origin
        ON {EMISSION_TABLE}(release_run_id, forecast_origin_utc, target);
    CREATE INDEX IF NOT EXISTS idx_movie_forecast_emissions_awaiting_actuals
        ON {EMISSION_TABLE}(model_version, target, opening_weekend_start)
        WHERE actual_usd IS NULL;
    """


def ensure_forecast_tables(conn: Any) -> None:
    conn.executescript(create_forecast_tables_sql())


def _insert_sql(table: str, columns: list[str], conflict: str) -> str:
    placeholders = ", ".join("%s" for _ in columns)
    column_list = ", ".join(columns)
    updates = ", ".join(f"{column}=EXCLUDED.{column}" for column in columns if column != "forecast_id")
    return f"INSERT INTO {table} ({column_list}) VALUES ({placeholders}) {conflict} {updates}"


def _plain_insert_sql(table: str, columns: list[str]) -> str:
    placeholders = ", ".join("%s" for _ in columns)
    column_list = ", ".join(columns)
    return f"INSERT INTO {table} ({column_list}) VALUES ({placeholders})"


def _insert_emission_sql() -> str:
    placeholders = ", ".join("%s" for _ in EMISSION_COLUMNS)
    column_list = ", ".join(EMISSION_COLUMNS)
    updates = ", ".join(f"{column}=EXCLUDED.{column}" for column in EMISSION_COLUMNS if column != "emitted_forecast_id")
    return f"""
        INSERT INTO {EMISSION_TABLE} ({column_list}) VALUES ({placeholders})
        ON CONFLICT (model_version, release_run_id, origin_key, target, as_of_utc, forecast_role)
        DO UPDATE SET {updates}
    """


def _emitted_forecast_id(row: dict[str, Any], forecast_role: str) -> str:
    key = "|".join(
        str(row.get(column) or "")
        for column in ["model_version", "release_run_id", "origin_key", "target", "as_of_utc"]
    )
    return uuid.uuid5(uuid.NAMESPACE_URL, f"pm-box-office|{forecast_role}|{key}").hex


def _forecast_role(row: dict[str, Any]) -> str:
    return str(row.get("forecast_role") or "production")


def _json_default(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _forecast_column_value(row: dict[str, Any], column: str) -> Any:
    value = row.get(column)
    if column == "distribution_payload" and isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=_json_default, separators=(",", ":"))
    return value


def _payload_hash(row: dict[str, Any], component_rows: list[dict[str, Any]], policy_label: str) -> str:
    payload = {
        "forecast": {
            column: row.get(column)
            for column in FORECAST_COLUMNS
            if column not in {"run_id", "actual_usd", "log_error", "abs_pct_error", "created_at"}
        },
        "forecast_role": _forecast_role(row),
        "production_policy_label": policy_label,
        "components": [
            {column: component.get(column) for column in COMPONENT_COLUMNS}
            for component in sorted(
                component_rows,
                key=lambda item: (str(item.get("forecast_id") or ""), str(item.get("component_day") or "")),
            )
        ],
        "simulation_seed": row.get("simulation_seed"),
        "amc_snapshot_id": row.get("amc_snapshot_id"),
        "actual_vintage_id": row.get("actual_vintage_id"),
        "artifact_version": row.get("artifact_version"),
        "residual_pool_id": row.get("residual_pool_id"),
    }
    raw = json.dumps(payload, sort_keys=True, default=_json_default, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _production_policy_label(row: dict[str, Any]) -> str:
    source = str(row.get("component_source") or "")
    if source == "latest_pre_release_carry_forward":
        return "friday_no_amc_pre_release_carry_forward"
    if source.lower().startswith("amc") or source == "AMC_plugin":
        return "amc_live_composition"
    if row.get("is_live"):
        return "live_baseline_composition"
    if row.get("is_backtest"):
        return "historical_backtest"
    return "production_forecast"


def _record_payload_conflict(conn: Any, row: dict[str, Any], emitted_forecast_id: str, existing_hash: str, incoming_hash: str) -> None:
    conflict_id = uuid.uuid5(uuid.NAMESPACE_URL, f"pm-box-office-conflict|{emitted_forecast_id}|{existing_hash}|{incoming_hash}").hex
    conn.execute(
        f"""
        INSERT INTO {EMISSION_CONFLICT_TABLE} (
            conflict_id, emitted_forecast_id, forecast_id, model_version, release_run_id,
            origin_key, target, as_of_utc, forecast_role, existing_payload_hash,
            incoming_payload_hash, reason
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (conflict_id) DO NOTHING
        """,
        (
            conflict_id,
            emitted_forecast_id,
            row.get("forecast_id"),
            row.get("model_version"),
            row.get("release_run_id"),
            row.get("origin_key"),
            row.get("target"),
            row.get("as_of_utc"),
            _forecast_role(row),
            existing_hash,
            incoming_hash,
            "logical_key_payload_hash_mismatch",
        ),
    )


def _check_payload_conflicts(
    conn: Any,
    forecast_rows: list[dict[str, Any]],
    emission_ids: dict[str, str],
    payload_hashes: dict[str, str],
) -> set[str]:
    conflicts: set[str] = set()
    for row in forecast_rows:
        forecast_id = str(row.get("forecast_id") or "")
        emitted_forecast_id = emission_ids.get(forecast_id)
        incoming_hash = payload_hashes.get(forecast_id)
        if not emitted_forecast_id or not incoming_hash:
            continue
        existing = conn.execute(
            f"""
            SELECT payload_hash
            FROM {EMISSION_TABLE}
            WHERE model_version = %s
              AND release_run_id = %s
              AND origin_key = %s
              AND target = %s
              AND as_of_utc = %s
              AND forecast_role = %s
            """,
            (
                row.get("model_version"),
                row.get("release_run_id"),
                row.get("origin_key"),
                row.get("target"),
                row.get("as_of_utc"),
                _forecast_role(row),
            ),
        ).fetchone()
        if existing and str(existing[0]) != incoming_hash:
            _record_payload_conflict(conn, row, emitted_forecast_id, str(existing[0]), incoming_hash)
            conflicts.add(forecast_id)
    if conflicts:
        first = next(row for row in forecast_rows if str(row.get("forecast_id") or "") in conflicts)
        raise RuntimeError(
            "forecast emission idempotency conflict for "
            f"{first.get('model_version')} release_run_id={first.get('release_run_id')} "
            f"origin={first.get('origin_key')} target={first.get('target')} role={_forecast_role(first)}"
        )
    return conflicts


def write_run(conn: Any, *, run_id: str, model_version: str, manifest_path: str, mode: str, as_of_utc: Any) -> None:
    conn.execute(
        f"""
        INSERT INTO {RUN_TABLE} (run_id, model_version, manifest_path, mode, as_of_utc)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (run_id) DO UPDATE SET
            model_version = EXCLUDED.model_version,
            manifest_path = EXCLUDED.manifest_path,
            mode = EXCLUDED.mode,
            as_of_utc = EXCLUDED.as_of_utc
        """,
        (run_id, model_version, manifest_path, mode, as_of_utc),
    )


def write_forecast_rows(
    conn: Any,
    forecast_rows: Iterable[dict[str, Any]],
    component_rows: Iterable[dict[str, Any]],
    *,
    skip_conflicting_emissions: bool = False,
) -> dict[str, Any]:
    forecast_row_list = list(forecast_rows)
    component_row_list = list(component_rows)
    audit: dict[str, Any] = {
        "input_forecast_rows": len(forecast_row_list),
        "input_components": len(component_row_list),
        "accepted_latest_rows": 0,
        "accepted_latest_components": 0,
        "accepted_emission_rows": 0,
        "accepted_emission_components": 0,
        "excluded_emission_rows": 0,
        "excluded_components": 0,
        "exclusion_reasons": {},
    }

    # A temporary AMC-feature gap must never erase a previously persisted live
    # AMC forecast for the same origin.  Actuals and a fresh AMC nowcast remain
    # authoritative; only a baseline-only replacement is rejected.
    protected_ids: set[str] = set()
    forecast_ids = [str(row["forecast_id"]) for row in forecast_row_list if row.get("forecast_id")]
    if forecast_ids:
        placeholders = ", ".join("%s" for _ in forecast_ids)
        existing = conn.execute(
            f"""
            SELECT forecast_id, component_source
            FROM {FORECAST_TABLE}
            WHERE forecast_id IN ({placeholders})
            """,
            tuple(forecast_ids),
        ).fetchall()
        existing_sources = {str(row[0]): str(row[1] or "") for row in existing}
        for row in forecast_row_list:
            forecast_id = str(row.get("forecast_id") or "")
            previous = existing_sources.get(forecast_id, "").lower()
            incoming = str(row.get("component_source") or "").lower()
            if previous.startswith("amc") and incoming == "daily_baseline":
                protected_ids.add(forecast_id)

    if protected_ids:
        audit["exclusion_reasons"] = dict(Counter({"protected_existing_amc_from_baseline_replacement": len(protected_ids)}))
        forecast_row_list = [row for row in forecast_row_list if str(row.get("forecast_id") or "") not in protected_ids]
        component_row_list = [row for row in component_row_list if str(row.get("forecast_id") or "") not in protected_ids]

    forecast_values = [tuple(_forecast_column_value(row, column) for column in FORECAST_COLUMNS) for row in forecast_row_list]
    component_values = [tuple(row.get(column) for column in COMPONENT_COLUMNS) for row in component_row_list]
    emission_ids = {
        str(row["forecast_id"]): _emitted_forecast_id(row, _forecast_role(row))
        for row in forecast_row_list
        if row.get("forecast_id")
    }
    components_by_forecast: dict[str, list[dict[str, Any]]] = {}
    for component in component_row_list:
        components_by_forecast.setdefault(str(component.get("forecast_id") or ""), []).append(component)
    policy_labels = {str(row.get("forecast_id") or ""): _production_policy_label(row) for row in forecast_row_list}
    payload_hashes = {
        str(row["forecast_id"]): _payload_hash(row, components_by_forecast.get(str(row["forecast_id"]), []), policy_labels[str(row["forecast_id"])])
        for row in forecast_row_list
        if row.get("forecast_id")
    }
    try:
        _check_payload_conflicts(conn, forecast_row_list, emission_ids, payload_hashes)
    except RuntimeError:
        if not skip_conflicting_emissions:
            raise
        # Re-check rows one at a time so an immutable, already-recorded origin
        # cannot block newer live origins in the same refresh batch.
        accepted_rows: list[dict[str, Any]] = []
        conflicting_ids: set[str] = set()
        for row in forecast_row_list:
            forecast_id = str(row.get("forecast_id") or "")
            try:
                _check_payload_conflicts(conn, [row], emission_ids, payload_hashes)
            except RuntimeError:
                conflicting_ids.add(forecast_id)
            else:
                accepted_rows.append(row)
        if not accepted_rows:
            _write_latest_forecast_rows(conn, forecast_row_list, component_row_list)
            audit["exclusion_reasons"] = dict(Counter({"immutable_emission_conflict": len(conflicting_ids)}))
            audit["accepted_latest_rows"] = len(forecast_row_list)
            audit["accepted_latest_components"] = len(component_row_list)
            audit["excluded_emission_rows"] = len(conflicting_ids)
            audit["excluded_components"] = len(component_row_list)
            audit["ui_events"] = _publish_ui_updates(conn, forecast_row_list)
            return audit
        accepted_components = [row for row in component_row_list if str(row.get("forecast_id") or "") not in conflicting_ids]
        result = write_forecast_rows(
            conn,
            accepted_rows,
            accepted_components,
            skip_conflicting_emissions=False,
        )
        conflicting_rows = [row for row in forecast_row_list if str(row.get("forecast_id") or "") in conflicting_ids]
        conflicting_components = [row for row in component_row_list if str(row.get("forecast_id") or "") in conflicting_ids]
        if conflicting_rows:
            # An emission is immutable evidence, but the current forecast is
            # deliberately mutable.  A changed AMC snapshot at an already
            # emitted origin must therefore update the UI/read model while the
            # original emission remains intact for auditability.
            _write_latest_forecast_rows(conn, conflicting_rows, conflicting_components)
            result["accepted_latest_rows"] += len(conflicting_rows)
            result["accepted_latest_components"] += len(conflicting_components)
            result["ui_events"] = int(result.get("ui_events", 0)) + _publish_ui_updates(conn, conflicting_rows)
        result["input_forecast_rows"] = audit["input_forecast_rows"]
        result["input_components"] = audit["input_components"]
        result["excluded_emission_rows"] += len(conflicting_ids)
        result["excluded_components"] += len(component_row_list) - len(accepted_components)
        result["exclusion_reasons"] = dict(Counter({"immutable_emission_conflict": len(conflicting_ids)}))
        return result
    generated_at = datetime.now(timezone.utc)
    emission_values = [
        (
            emission_ids[str(row["forecast_id"])],
            row.get("forecast_id"),
            row.get("created_at") or generated_at,
            _forecast_role(row),
            payload_hashes[str(row["forecast_id"])],
            policy_labels[str(row["forecast_id"])],
            *[_forecast_column_value(row, column) for column in EMISSION_FORECAST_COLUMNS],
        )
        for row in forecast_row_list
        if row.get("forecast_id") and str(row["forecast_id"]) in emission_ids
    ]
    emission_component_values = [
        (
            emission_ids[str(row["forecast_id"])],
            *[row.get(column) for column in COMPONENT_COLUMNS],
        )
        for row in component_row_list
        if row.get("forecast_id") and str(row["forecast_id"]) in emission_ids
    ]
    if forecast_values:
        audit["accepted_latest_rows"] = len(forecast_values)
        conn.executemany(
            _insert_sql(
                FORECAST_TABLE,
                FORECAST_COLUMNS,
                "ON CONFLICT (forecast_id) DO UPDATE SET",
            ),
            forecast_values,
        )
    if component_values:
        audit["accepted_latest_components"] = len(component_values)
        placeholders = ", ".join("%s" for _ in COMPONENT_COLUMNS)
        column_list = ", ".join(COMPONENT_COLUMNS)
        updates = ", ".join(
            f"{column}=EXCLUDED.{column}" for column in COMPONENT_COLUMNS if column not in {"forecast_id", "component_day"}
        )
        conn.executemany(
            f"""
            INSERT INTO {COMPONENT_TABLE} ({column_list}) VALUES ({placeholders})
            ON CONFLICT (forecast_id, component_day) DO UPDATE SET {updates}
            """,
            component_values,
        )
    if emission_values:
        audit["accepted_emission_rows"] = len(emission_values)
        conn.executemany(_insert_emission_sql(), emission_values)
    if emission_component_values:
        audit["accepted_emission_components"] = len(emission_component_values)
        conn.executemany(
            f"""
            INSERT INTO {EMISSION_COMPONENT_TABLE} ({", ".join(EMISSION_COMPONENT_COLUMNS)})
            VALUES ({", ".join("%s" for _ in EMISSION_COMPONENT_COLUMNS)})
            ON CONFLICT (emitted_forecast_id, forecast_id, component_day) DO UPDATE SET
                component_type = EXCLUDED.component_type,
                component_point_usd = EXCLUDED.component_point_usd,
                component_sigma_log = EXCLUDED.component_sigma_log,
                component_lo80_usd = EXCLUDED.component_lo80_usd,
                component_hi80_usd = EXCLUDED.component_hi80_usd,
                component_lo95_usd = EXCLUDED.component_lo95_usd,
                component_hi95_usd = EXCLUDED.component_hi95_usd,
                component_model = EXCLUDED.component_model,
                component_source = EXCLUDED.component_source,
                component_notes = EXCLUDED.component_notes
            """,
            emission_component_values,
        )
    # Maintain the UI read model and notification outbox in this same database
    # transaction. Consumers can safely react only after the worker commits.
    if forecast_row_list and hasattr(conn, "executescript"):
        from .ui_projection import publish_forecast_updates

        audit["ui_events"] = publish_forecast_updates(conn, forecast_row_list)
    else:
        audit["ui_events"] = 0
    audit["excluded_emission_rows"] = audit["input_forecast_rows"] - audit["accepted_emission_rows"]
    audit["excluded_components"] = audit["input_components"] - audit["accepted_emission_components"]
    return audit


def _write_latest_forecast_rows(
    conn: Any,
    forecast_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
) -> None:
    """Upsert the mutable current read model without creating new emissions."""
    values = [tuple(_forecast_column_value(row, column) for column in FORECAST_COLUMNS) for row in forecast_rows]
    if values:
        conn.executemany(
            _insert_sql(FORECAST_TABLE, FORECAST_COLUMNS, "ON CONFLICT (forecast_id) DO UPDATE SET"),
            values,
        )
    component_values = [tuple(row.get(column) for column in COMPONENT_COLUMNS) for row in component_rows]
    if component_values:
        placeholders = ", ".join("%s" for _ in COMPONENT_COLUMNS)
        updates = ", ".join(
            f"{column}=EXCLUDED.{column}" for column in COMPONENT_COLUMNS if column not in {"forecast_id", "component_day"}
        )
        conn.executemany(
            f"INSERT INTO {COMPONENT_TABLE} ({', '.join(COMPONENT_COLUMNS)}) VALUES ({placeholders}) "
            f"ON CONFLICT (forecast_id, component_day) DO UPDATE SET {updates}",
            component_values,
        )


def _publish_ui_updates(conn: Any, forecast_rows: list[dict[str, Any]]) -> int:
    if not forecast_rows or not hasattr(conn, "executescript"):
        return 0
    from .ui_projection import publish_forecast_updates

    return publish_forecast_updates(conn, forecast_rows)


def write_control_emissions(
    conn: Any,
    forecast_rows: Iterable[dict[str, Any]],
    component_rows: Iterable[dict[str, Any]],
    *,
    forecast_role: str = "amc_no_amc_control",
) -> int:
    """Persist an immutable paired-control emission without replacing latest rows."""

    rows = [dict(row, forecast_role=forecast_role) for row in forecast_rows]
    components = list(component_rows)
    if not rows:
        return 0
    by_forecast: dict[str, list[dict[str, Any]]] = {}
    for component in components:
        by_forecast.setdefault(str(component.get("forecast_id") or ""), []).append(component)
    ids = {str(row["forecast_id"]): _emitted_forecast_id(row, forecast_role) for row in rows}
    labels = {str(row["forecast_id"]): "amc_forward_validation_no_amc_control" for row in rows}
    hashes = {str(row["forecast_id"]): _payload_hash(row, by_forecast.get(str(row["forecast_id"]), []), labels[str(row["forecast_id"])]) for row in rows}
    _check_payload_conflicts(conn, rows, ids, hashes)
    generated_at = datetime.now(timezone.utc)
    values = [
        (ids[str(row["forecast_id"])], row.get("forecast_id"), row.get("created_at") or generated_at, forecast_role,
         hashes[str(row["forecast_id"])], labels[str(row["forecast_id"])],
         *[_forecast_column_value(row, column) for column in EMISSION_FORECAST_COLUMNS])
        for row in rows
    ]
    conn.executemany(_insert_emission_sql(), values)
    component_values = [
        (ids[str(component["forecast_id"])], *[component.get(column) for column in COMPONENT_COLUMNS])
        for component in components if str(component.get("forecast_id") or "") in ids
    ]
    if component_values:
        conn.executemany(
            f"INSERT INTO {EMISSION_COMPONENT_TABLE} ({', '.join(EMISSION_COMPONENT_COLUMNS)}) VALUES ({', '.join('%s' for _ in EMISSION_COMPONENT_COLUMNS)}) "
            "ON CONFLICT (emitted_forecast_id, forecast_id, component_day) DO NOTHING",
            component_values,
        )
    return len(values)


def attach_actuals_to_forecast_emissions(conn: Any) -> int:
    """Attach later known outcomes to immutable emissions without changing forecast inputs."""

    cursor = conn.execute(
        f"""
        UPDATE {EMISSION_TABLE} emission
        SET
            actual_usd = latest.actual_usd,
            first_final_actual_usd = COALESCE(emission.first_final_actual_usd, latest.actual_usd),
            latest_final_actual_usd = latest.actual_usd,
            log_error = CASE
                WHEN latest.actual_usd > 0 AND emission.point_usd > 0
                THEN LN(latest.actual_usd / emission.point_usd)
                ELSE NULL
            END,
            abs_pct_error = CASE
                WHEN latest.actual_usd > 0 AND emission.point_usd > 0
                THEN ABS(latest.actual_usd - emission.point_usd) / latest.actual_usd
                ELSE NULL
            END
        FROM {FORECAST_TABLE} latest
        WHERE latest.forecast_id = emission.forecast_id
          AND latest.actual_usd IS NOT NULL
          AND emission.actual_usd IS NULL
        """
    )
    return int(cursor.rowcount or 0)


def remove_obsolete_future_forecasts(
    conn: Any,
    *,
    model_version: str,
    opening_weekend_start: Any,
    opening_weekend_end: Any,
    active_release_run_ids: Iterable[int],
) -> int:
    """Remove virtual future-candidate forecasts absent from a full refresh.

    Candidate release runs are negative IDs. A full live generation is
    authoritative for its forward-looking window, so rows no longer backed by
    an estimate or AMC candidate must not remain discoverable in the UI.
    """
    active_ids = sorted({int(release_run_id) for release_run_id in active_release_run_ids})
    predicates = [
        "model_version = %s",
        "release_run_id < 0",
        "opening_weekend_start BETWEEN %s AND %s",
    ]
    params: list[Any] = [model_version, opening_weekend_start, opening_weekend_end]
    if active_ids:
        placeholders = ", ".join("%s" for _ in active_ids)
        predicates.append(f"release_run_id NOT IN ({placeholders})")
        params.extend(active_ids)
    cursor = conn.execute(
        f"DELETE FROM {FORECAST_TABLE} WHERE {' AND '.join(predicates)}",
        tuple(params),
    )
    return int(cursor.rowcount or 0)


def rows_from_results(results: Iterable[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    forecast_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for result in results:
        forecast_rows.append(result.to_forecast_row())
        component_rows.extend(result.to_component_rows())
    return forecast_rows, component_rows
