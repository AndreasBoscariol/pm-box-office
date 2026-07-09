from __future__ import annotations

import datetime as dt
import decimal
from dataclasses import dataclass
from typing import Any


FORECAST_TABLE = "analytics.movie_opening_weekend_forecasts"
COMPONENT_TABLE = "analytics.movie_forecast_components"

TARGETS = ("opening_weekend", "friday", "saturday", "sunday", "remaining_weekend")
PRE_RELEASE_ORIGINS = [f"P_{day}" for day in range(-14, 0)]
LIVE_ORIGIN_TIMES = ("10:00", "12:00", "14:00", "16:00", "18:00", "20:00", "EOD")
LIVE_ORIGINS = [
    *[f"FRI_{origin}" for origin in LIVE_ORIGIN_TIMES],
    *[f"SAT_{origin}" for origin in LIVE_ORIGIN_TIMES],
    *[f"SUN_{origin}" for origin in LIVE_ORIGIN_TIMES],
]
EXPECTED_ORIGIN_KEYS = PRE_RELEASE_ORIGINS + LIVE_ORIGINS

REGIME_LABELS = {
    "pre_release": "Pre-release",
    "live_friday": "Friday live",
    "live_saturday": "Saturday live",
    "live_sunday": "Sunday live",
}

DAY_TARGETS = {
    "Friday": "friday",
    "Saturday": "saturday",
    "Sunday": "sunday",
}


@dataclass(frozen=True)
class ForecastFilters:
    q: str = ""
    release_start: str = ""
    release_end: str = ""
    model_version: str = ""
    target: str = "opening_weekend"


def forecast_tables_exist(conn: Any) -> bool:
    row = conn.execute(
        """
        SELECT COUNT(*)::integer
        FROM information_schema.tables
        WHERE table_schema = 'analytics'
          AND table_name IN ('movie_opening_weekend_forecasts', 'movie_forecast_components')
        """
    ).fetchone()
    return int(row[0] or 0) == 2 if row else False


def latest_model_version(conn: Any) -> str | None:
    if not forecast_tables_exist(conn):
        return None
    row = conn.execute(
        f"""
        SELECT model_version
        FROM {FORECAST_TABLE}
        GROUP BY model_version
        ORDER BY MAX(created_at) DESC NULLS LAST, model_version DESC
        LIMIT 1
        """
    ).fetchone()
    return str(row[0]) if row and row[0] else None


def model_versions(conn: Any) -> list[str]:
    if not forecast_tables_exist(conn):
        return []
    rows = conn.execute(
        f"""
        SELECT model_version
        FROM {FORECAST_TABLE}
        GROUP BY model_version
        ORDER BY MAX(created_at) DESC NULLS LAST, model_version DESC
        """
    ).fetchall()
    return [str(row[0]) for row in rows]


def normalize_filters(raw: dict[str, str | None], *, default_model_version: str | None = None) -> ForecastFilters:
    target = (raw.get("target") or "opening_weekend").strip()
    if target not in TARGETS:
        target = "opening_weekend"
    return ForecastFilters(
        q=(raw.get("q") or "").strip(),
        release_start=(raw.get("release_start") or "").strip(),
        release_end=(raw.get("release_end") or "").strip(),
        model_version=(raw.get("model_version") or default_model_version or "").strip(),
        target=target,
    )


def dashboard(conn: Any, filters: ForecastFilters) -> dict[str, Any]:
    if not forecast_tables_exist(conn):
        return {
            "summary": empty_summary(),
            "rows": [],
            "model_versions": [],
            "filters": filters,
            "tables_ready": False,
        }

    versions = model_versions(conn)
    selected_version = filters.model_version or (versions[0] if versions else "")
    selected_filters = ForecastFilters(
        q=filters.q,
        release_start=filters.release_start,
        release_end=filters.release_end,
        model_version=selected_version,
        target=filters.target,
    )
    return {
        "summary": dashboard_summary(conn, selected_filters),
        "rows": dashboard_rows(conn, selected_filters),
        "model_versions": versions,
        "filters": selected_filters,
        "tables_ready": True,
    }


def empty_summary() -> dict[str, Any]:
    return {
        "movie_count": 0,
        "latest_model_version": None,
        "latest_run_time": None,
        "row_count": 0,
        "backtest_rows": 0,
        "live_rows": 0,
    }


def dashboard_summary(conn: Any, filters: ForecastFilters) -> dict[str, Any]:
    predicates, params = dashboard_predicates(filters, alias="f", include_target=False)
    row = conn.execute(
        f"""
        SELECT
            COUNT(DISTINCT f.release_run_id)::integer AS movie_count,
            MAX(f.created_at) AS latest_run_time,
            COUNT(*)::integer AS row_count,
            COUNT(*) FILTER (WHERE f.is_backtest)::integer AS backtest_rows,
            COUNT(*) FILTER (WHERE f.is_live)::integer AS live_rows
        FROM {FORECAST_TABLE} f
        WHERE {' AND '.join(predicates)}
        """,
        params,
    ).fetchone()
    return {
        "movie_count": int(row[0] or 0) if row else 0,
        "latest_model_version": filters.model_version or None,
        "latest_run_time": row[1] if row else None,
        "row_count": int(row[2] or 0) if row else 0,
        "backtest_rows": int(row[3] or 0) if row else 0,
        "live_rows": int(row[4] or 0) if row else 0,
    }


def dashboard_rows(conn: Any, filters: ForecastFilters) -> list[dict[str, Any]]:
    predicates, params = dashboard_predicates(filters, alias="f", include_target=True)
    cursor = conn.execute(
        f"""
        WITH ranked AS (
            SELECT
                f.*,
                ROW_NUMBER() OVER (
                    PARTITION BY f.release_run_id
                    ORDER BY {origin_order_sql('f')} DESC, f.forecast_origin_utc DESC
                ) AS rank
            FROM {FORECAST_TABLE} f
            WHERE {' AND '.join(predicates)}
        )
        SELECT
            forecast_id,
            run_id,
            model_version,
            movie_id,
            release_run_id,
            title,
            opening_weekend_start,
            regime,
            origin_key,
            forecast_origin_utc,
            as_of_utc,
            target,
            point_usd,
            lo80_usd,
            hi80_usd,
            lo95_usd,
            hi95_usd,
            component_source,
            feature_quality_bucket,
            actual_usd,
            log_error,
            abs_pct_error,
            is_live,
            is_backtest
        FROM ranked
        WHERE rank = 1
        ORDER BY opening_weekend_start DESC, title
        LIMIT 200
        """,
        params,
    )
    return rows_as_dicts(cursor)


def dashboard_predicates(filters: ForecastFilters, *, alias: str, include_target: bool) -> tuple[list[str], tuple[Any, ...]]:
    predicates = [f"{alias}.model_version = %s"]
    params: list[Any] = [filters.model_version]
    if include_target:
        predicates.append(f"{alias}.target = %s")
        params.append(filters.target)
    if filters.q:
        predicates.append(f"{alias}.title ILIKE %s")
        params.append(f"%{filters.q}%")
    if filters.release_start:
        predicates.append(f"{alias}.opening_weekend_start >= %s")
        params.append(filters.release_start)
    if filters.release_end:
        predicates.append(f"{alias}.opening_weekend_start <= %s")
        params.append(filters.release_end)
    return predicates, tuple(params)


def movie_timeline(
    conn: Any,
    *,
    release_run_id: int,
    model_version: str | None = None,
) -> dict[str, Any]:
    if not forecast_tables_exist(conn):
        return {
            "movie": None,
            "latest": None,
            "timeline": pending_timeline([]),
            "chart": forecast_chart(pending_timeline([]), None),
            "model_versions": [],
            "selected_model_version": model_version or "",
            "tables_ready": False,
        }

    versions = model_versions_for_release(conn, release_run_id)
    selected_version = model_version or (versions[0] if versions else latest_model_version(conn))
    rows = timeline_rows(conn, release_run_id=release_run_id, model_version=selected_version)
    components = component_rows(conn, [row["forecast_id"] for row in rows])
    attach_components(rows, components)
    latest = latest_opening_weekend_row(rows)
    timeline = pending_timeline(rows)
    return {
        "movie": movie_from_rows(rows),
        "latest": latest,
        "timeline": timeline,
        "chart": forecast_chart(timeline, latest),
        "model_versions": versions,
        "selected_model_version": selected_version or "",
        "tables_ready": True,
    }


def model_versions_for_release(conn: Any, release_run_id: int) -> list[str]:
    rows = conn.execute(
        f"""
        SELECT model_version
        FROM {FORECAST_TABLE}
        WHERE release_run_id = %s
        GROUP BY model_version
        ORDER BY MAX(created_at) DESC NULLS LAST, model_version DESC
        """,
        (release_run_id,),
    ).fetchall()
    return [str(row[0]) for row in rows]


def timeline_rows(conn: Any, *, release_run_id: int, model_version: str | None) -> list[dict[str, Any]]:
    predicates = ["f.release_run_id = %s"]
    params: list[Any] = [release_run_id]
    if model_version:
        predicates.append("f.model_version = %s")
        params.append(model_version)
    cursor = conn.execute(
        f"""
        SELECT
            f.*
        FROM {FORECAST_TABLE} f
        WHERE {' AND '.join(predicates)}
        ORDER BY {origin_order_sql('f')}, {target_order_sql('f')}
        """,
        tuple(params),
    )
    return rows_as_dicts(cursor)


def component_rows(conn: Any, forecast_ids: list[str]) -> list[dict[str, Any]]:
    if not forecast_ids:
        return []
    cursor = conn.execute(
        f"""
        SELECT *
        FROM {COMPONENT_TABLE}
        WHERE forecast_id = ANY(%s)
        ORDER BY
            forecast_id,
            CASE component_day
                WHEN 'Friday' THEN 1
                WHEN 'Saturday' THEN 2
                WHEN 'Sunday' THEN 3
                ELSE 9
            END
        """,
        (forecast_ids,),
    )
    return rows_as_dicts(cursor)


def attach_components(rows: list[dict[str, Any]], components: list[dict[str, Any]]) -> None:
    by_forecast: dict[str, list[dict[str, Any]]] = {}
    for component in components:
        by_forecast.setdefault(str(component["forecast_id"]), []).append(component)
    for row in rows:
        row["components"] = by_forecast.get(str(row["forecast_id"]), [])


def latest_opening_weekend_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("target") == "opening_weekend"]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: origin_sort_value(str(row.get("origin_key") or "")))[-1]


def movie_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    row = rows[0]
    return {
        "movie_id": row.get("movie_id"),
        "release_run_id": row.get("release_run_id"),
        "title": row.get("title"),
        "opening_weekend_start": row.get("opening_weekend_start"),
    }


def pending_timeline(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_origin: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("target") == "opening_weekend":
            by_origin[str(row.get("origin_key"))] = row

    timeline = []
    for key in EXPECTED_ORIGIN_KEYS:
        row = by_origin.get(key)
        if row is None:
            timeline.append(
                {
                    "origin_key": key,
                    "regime": regime_for_origin(key),
                    "regime_label": REGIME_LABELS[regime_for_origin(key)],
                    "status": "pending",
                    "day_components": {},
                    "components": [],
                }
            )
            continue
        prepared = dict(row)
        prepared["status"] = "ready"
        prepared["regime_label"] = REGIME_LABELS.get(str(row.get("regime")), str(row.get("regime") or ""))
        prepared["day_components"] = day_components(row.get("components") or [])
        timeline.append(prepared)
    return timeline


def day_components(components: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(component.get("component_day")): component for component in components}


def forecast_chart(timeline: list[dict[str, Any]], latest: dict[str, Any] | None) -> dict[str, Any]:
    width = 1180
    height = 420
    left = 68
    right = 26
    top = 24
    bottom = 54
    plot_width = width - left - right
    plot_height = height - top - bottom

    ready = []
    values: list[float] = []
    for index, row in enumerate(timeline):
        if row.get("status") != "ready":
            continue
        point = coerce_float(row.get("point_usd"))
        lo80 = coerce_float(row.get("lo80_usd"))
        hi80 = coerce_float(row.get("hi80_usd"))
        lo95 = coerce_float(row.get("lo95_usd"))
        hi95 = coerce_float(row.get("hi95_usd"))
        if point is None:
            continue
        ready.append(
            {
                "index": index,
                "origin_key": row.get("origin_key"),
                "regime": row.get("regime"),
                "point": point,
                "lo80": lo80,
                "hi80": hi80,
                "lo95": lo95,
                "hi95": hi95,
            }
        )
        values.extend(value for value in [point, lo80, hi80, lo95, hi95] if value is not None)

    actual = coerce_float(latest.get("actual_usd")) if latest else None
    if actual is not None:
        values.append(actual)
    if not ready or not values:
        return {
            "width": width,
            "height": height,
            "has_data": False,
            "points": [],
            "point_path": "",
            "band80_path": "",
            "band95_path": "",
            "y_ticks": [],
            "x_labels": x_axis_labels(width=width, left=left, right=right),
            "regime_markers": regime_markers(width=width, left=left, right=right),
            "actual_y": None,
            "actual_label": "-",
        }

    low = min(values)
    high = max(values)
    if high <= low:
        high = low + max(low * 0.1, 1.0)
    padding = (high - low) * 0.08
    y_min = max(0.0, low - padding)
    y_max = high + padding

    def x_for(index: int) -> float:
        if len(EXPECTED_ORIGIN_KEYS) == 1:
            return left
        return left + plot_width * index / (len(EXPECTED_ORIGIN_KEYS) - 1)

    def y_for(value: float) -> float:
        return top + plot_height * (y_max - value) / (y_max - y_min)

    chart_points = []
    for item in ready:
        chart_points.append(
            {
                **item,
                "x": round(x_for(int(item["index"])), 2),
                "y": round(y_for(float(item["point"])), 2),
                "lo80_y": round(y_for(item["lo80"]), 2) if item["lo80"] is not None else None,
                "hi80_y": round(y_for(item["hi80"]), 2) if item["hi80"] is not None else None,
                "lo95_y": round(y_for(item["lo95"]), 2) if item["lo95"] is not None else None,
                "hi95_y": round(y_for(item["hi95"]), 2) if item["hi95"] is not None else None,
                "point_label": compact_usd(item["point"]),
            }
        )

    tick_values = nice_ticks(y_min, y_max, count=5)
    return {
        "width": width,
        "height": height,
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "plot_width": plot_width,
        "plot_height": plot_height,
        "has_data": True,
        "points": chart_points,
        "point_path": line_path(chart_points, "y"),
        "band80_path": band_path(chart_points, "hi80_y", "lo80_y"),
        "band95_path": band_path(chart_points, "hi95_y", "lo95_y"),
        "y_ticks": [
            {
                "value": value,
                "label": compact_usd(value),
                "y": round(y_for(value), 2),
            }
            for value in tick_values
        ],
        "x_labels": x_axis_labels(width=width, left=left, right=right),
        "regime_markers": regime_markers(width=width, left=left, right=right),
        "actual_y": round(y_for(actual), 2) if actual is not None else None,
        "actual_label": compact_usd(actual),
    }


def line_path(points: list[dict[str, Any]], y_key: str) -> str:
    parts = []
    for point in points:
        y = point.get(y_key)
        if y is None:
            continue
        command = "M" if not parts else "L"
        parts.append(f"{command}{point['x']},{y}")
    return " ".join(parts)


def band_path(points: list[dict[str, Any]], upper_key: str, lower_key: str) -> str:
    upper = [point for point in points if point.get(upper_key) is not None and point.get(lower_key) is not None]
    if len(upper) < 2:
        return ""
    upper_path = [f"{point['x']},{point[upper_key]}" for point in upper]
    lower_path = [f"{point['x']},{point[lower_key]}" for point in reversed(upper)]
    return f"M{' L'.join(upper_path)} L{' L'.join(lower_path)} Z"


def x_axis_labels(*, width: int, left: int, right: int) -> list[dict[str, Any]]:
    plot_width = width - left - right
    label_keys = ["P_-14", "P_-7", "P_-1", "FRI_EOD", "SAT_EOD", "SUN_EOD"]
    return [
        {
            "origin_key": key,
            "label": key.replace("_", " "),
            "x": round(left + plot_width * origin_sort_value(key) / (len(EXPECTED_ORIGIN_KEYS) - 1), 2),
        }
        for key in label_keys
    ]


def regime_markers(*, width: int, left: int, right: int) -> list[dict[str, Any]]:
    plot_width = width - left - right
    markers = [
        ("P_-14", "Pre-release"),
        ("FRI_10:00", "Fri"),
        ("SAT_10:00", "Sat"),
        ("SUN_10:00", "Sun"),
    ]
    return [
        {
            "origin_key": key,
            "label": label,
            "x": round(left + plot_width * origin_sort_value(key) / (len(EXPECTED_ORIGIN_KEYS) - 1), 2),
        }
        for key, label in markers
    ]


def nice_ticks(y_min: float, y_max: float, *, count: int) -> list[float]:
    if count <= 1 or y_max <= y_min:
        return [y_min, y_max]
    step = (y_max - y_min) / (count - 1)
    return [y_min + step * index for index in range(count)]


def regime_for_origin(origin_key: str) -> str:
    if origin_key.startswith("P_"):
        return "pre_release"
    if origin_key.startswith("FRI_"):
        return "live_friday"
    if origin_key.startswith("SAT_"):
        return "live_saturday"
    if origin_key.startswith("SUN_"):
        return "live_sunday"
    return "unknown"


def origin_sort_value(origin_key: str) -> int:
    if origin_key.startswith("P_"):
        try:
            return int(origin_key.split("_", 1)[1]) + 14
        except ValueError:
            return 0
    live_index = {origin: index for index, origin in enumerate(LIVE_ORIGINS, start=len(PRE_RELEASE_ORIGINS))}
    return live_index.get(origin_key, 999)


def origin_order_sql(alias: str) -> str:
    cases = " ".join(f"WHEN '{origin}' THEN {index}" for index, origin in enumerate(EXPECTED_ORIGIN_KEYS))
    return f"CASE {alias}.origin_key {cases} ELSE 999 END"


def target_order_sql(alias: str) -> str:
    return f"""
        CASE {alias}.target
            WHEN 'opening_weekend' THEN 1
            WHEN 'friday' THEN 2
            WHEN 'saturday' THEN 3
            WHEN 'sunday' THEN 4
            ELSE 9
        END
    """


def rows_as_dicts(cursor: Any) -> list[dict[str, Any]]:
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row, strict=False)) for row in rows]


def compact_usd(value: object) -> str:
    number = coerce_float(value)
    if number is None:
        return "-"
    sign = "-" if number < 0 else ""
    number = abs(number)
    if number >= 1_000_000_000:
        return f"{sign}${number / 1_000_000_000:.1f}B"
    if number >= 1_000_000:
        return f"{sign}${number / 1_000_000:.1f}M"
    if number >= 1_000:
        return f"{sign}${number / 1_000:.0f}K"
    return f"{sign}${number:.0f}"


def interval_label(lo: object, hi: object) -> str:
    if coerce_float(lo) is None or coerce_float(hi) is None:
        return "-"
    return f"{compact_usd(lo)} - {compact_usd(hi)}"


def signed_number(value: object, *, digits: int = 2) -> str:
    number = coerce_float(value)
    if number is None:
        return "-"
    return f"{number:+.{digits}f}"


def percent(value: object) -> str:
    number = coerce_float(value)
    if number is None:
        return "-"
    return f"{number * 100:.1f}%"


def coerce_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, decimal.Decimal):
        value = float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def date_value(value: object) -> str:
    if isinstance(value, dt.date):
        return value.isoformat()
    return str(value or "")
