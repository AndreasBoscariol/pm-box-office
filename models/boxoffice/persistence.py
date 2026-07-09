"""Database DDL and persistence helpers for production forecast tables."""

from __future__ import annotations

from typing import Any, Iterable

from .constants import COMPONENT_TABLE, FORECAST_TABLE, RUN_TABLE

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

    CREATE INDEX IF NOT EXISTS idx_movie_opening_weekend_forecasts_movie
        ON {FORECAST_TABLE}(movie_id, model_version, origin_key, target);
    CREATE INDEX IF NOT EXISTS idx_movie_opening_weekend_forecasts_release
        ON {FORECAST_TABLE}(release_run_id, opening_weekend_start);
    """


def ensure_forecast_tables(conn: Any) -> None:
    conn.executescript(create_forecast_tables_sql())


def _insert_sql(table: str, columns: list[str], conflict: str) -> str:
    placeholders = ", ".join("%s" for _ in columns)
    column_list = ", ".join(columns)
    updates = ", ".join(f"{column}=EXCLUDED.{column}" for column in columns if column != "forecast_id")
    return f"INSERT INTO {table} ({column_list}) VALUES ({placeholders}) {conflict} {updates}"


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


def write_forecast_rows(conn: Any, forecast_rows: Iterable[dict[str, Any]], component_rows: Iterable[dict[str, Any]]) -> None:
    forecast_values = [tuple(row.get(column) for column in FORECAST_COLUMNS) for row in forecast_rows]
    component_values = [tuple(row.get(column) for column in COMPONENT_COLUMNS) for row in component_rows]
    if forecast_values:
        conn.executemany(
            _insert_sql(
                FORECAST_TABLE,
                FORECAST_COLUMNS,
                "ON CONFLICT (forecast_id) DO UPDATE SET",
            ),
            forecast_values,
        )
    if component_values:
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


def rows_from_results(results: Iterable[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    forecast_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for result in results:
        forecast_rows.append(result.to_forecast_row())
        component_rows.extend(result.to_component_rows())
    return forecast_rows, component_rows

