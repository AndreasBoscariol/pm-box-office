from __future__ import annotations

import datetime as dt
import decimal
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from models.boxoffice.forecast_distribution import ForecastDistribution
from models.boxoffice.market_buckets import (
    TARGET_LISTING_ORIGIN,
    MarketGrid,
    generate_rounding_variants,
)


FORECAST_TABLE = "analytics.movie_opening_weekend_forecasts"
COMPONENT_TABLE = "analytics.movie_forecast_components"
AMC_SHADOW_TABLE = "analytics.amc_opening_shadow_forecasts"

TARGETS = ("opening_weekend", "friday", "saturday", "sunday", "remaining_weekend")
PRE_RELEASE_ORIGINS = [f"P_{day}" for day in range(-14, 0)]
LIVE_ORIGIN_TIMES = ("10:00", "12:00", "14:00", "16:00", "18:00", "20:00", "EOD")
THURSDAY_ORIGINS = [f"THU_{origin}" for origin in LIVE_ORIGIN_TIMES]
LIVE_ORIGINS = [
    *[f"FRI_{origin}" for origin in LIVE_ORIGIN_TIMES],
    *[f"SAT_{origin}" for origin in LIVE_ORIGIN_TIMES],
    *[f"SUN_{origin}" for origin in LIVE_ORIGIN_TIMES],
]
EXPECTED_ORIGIN_KEYS = PRE_RELEASE_ORIGINS + THURSDAY_ORIGINS + LIVE_ORIGINS
AMC_COMPONENT_SOURCES = ["AMC_plugin"]
BASELINE_COMPONENT_SOURCE = "daily_baseline"

REGIME_LABELS = {
    "pre_release": "Pre-release",
    "thursday_preview": "Thursday preview",
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
    include_coverage_fallback: bool = False


@dataclass(frozen=True)
class PolymarketMarketGrid:
    release_run_id: int
    target_listing_origin: str
    effective_listing_origin: str
    anchor_forecast_usd: int
    anchor_forecast_emission_hash: str
    point_policy_version: str
    width_rule_version: str
    rounding_rule_version: str
    selected_width_usd: int | None
    ideal_start_usd: int
    rounded_start_usd: int
    boundaries_usd: tuple[int, int, int, int]
    rounding_variant: str
    width_selection_rule: str
    grid_id: str
    market_source: str
    event_id: str
    event_slug: str | None
    event_title: str
    market_ids: tuple[str, ...]
    market_questions: tuple[str, ...]
    bucket_count: int = 5

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


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
    # The dashboard must follow the explicitly promoted production artifact,
    # rather than whichever model happened to write most recently.
    try:
        from models.boxoffice.artifacts import latest_model_version as active_model_version

        active = active_model_version()
        row = conn.execute(
            f"SELECT 1 FROM {FORECAST_TABLE} WHERE model_version = %s LIMIT 1",
            (active,),
        ).fetchone()
        if row:
            return active
    except (FileNotFoundError, ValueError):
        pass
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
    versions = [str(row[0]) for row in rows]
    # Keep the explicitly promoted artifact at the top of every model selector.
    # Database write time is not a deployment signal: a historical backfill can
    # otherwise make the UI silently default to a non-production model.
    active = latest_model_version(conn)
    if active in versions:
        return [active, *[version for version in versions if version != active]]
    return versions


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
        include_coverage_fallback=(raw.get("include_coverage_fallback") or "").strip().lower() in {"1", "true", "on", "yes"},
    )


def dashboard(conn: Any, filters: ForecastFilters) -> dict[str, Any]:
    if not forecast_tables_exist(conn):
        return {
            "summary": empty_summary(),
            "rows": [],
            "model_versions": [],
            "filters": filters,
            "distribution_policy": distribution_policy_context(None),
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
        include_coverage_fallback=filters.include_coverage_fallback,
    )
    return {
        "summary": dashboard_summary(conn, selected_filters),
        "rows": dashboard_rows(conn, selected_filters),
        "model_versions": versions,
        "filters": selected_filters,
        "distribution_policy": distribution_policy_context(selected_version),
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
    # The compact current-release projection is the primary UI read model. It
    # is updated atomically with forecast persistence, so it cannot combine a
    # fresh source input with a stale model row.
    if relation_exists(conn, "analytics.current_release_forecasts"):
        predicates, params = dashboard_predicates(filters, alias="f", include_target=True)
        extra_predicates = predicates[2:]
        extra_sql = f" AND {' AND '.join(extra_predicates)}" if extra_predicates else ""
        cursor = conn.execute(
            f"""
            SELECT
                f.forecast_id, f.run_id, f.model_version, f.movie_id,
                f.release_run_id, f.title, f.opening_weekend_start, f.regime,
                f.origin_key, f.forecast_origin_utc, f.as_of_utc, f.target,
                f.point_usd, f.lo80_usd, f.hi80_usd, f.lo95_usd, f.hi95_usd,
                f.component_source, f.feature_quality_bucket, f.actual_usd,
                f.log_error, f.abs_pct_error, f.is_live, f.is_backtest,
                FALSE AS coverage_fallback,
                'available' AS prediction_status
            FROM analytics.current_release_forecasts projection
            JOIN {FORECAST_TABLE} f ON f.forecast_id = projection.forecast_id
            WHERE projection.model_version = %s
              AND projection.target = %s
              {extra_sql}
            ORDER BY f.opening_weekend_start DESC, f.title
            LIMIT 200
            """,
            (filters.model_version, filters.target, *params[2:]),
        )
        return rows_as_dicts(cursor)
    predicates, params = dashboard_predicates(filters, alias="f", include_target=True)
    coverage_model = most_complete_model_version(conn)
    model_priority = "0"
    if filters.include_coverage_fallback and coverage_model and coverage_model != filters.model_version:
        predicates[0] = "f.model_version = ANY(%s)"
        params = ([filters.model_version, coverage_model], *params[1:])
        model_priority = "CASE WHEN f.model_version = %s THEN 1 ELSE 0 END"
        # The active model wins when it has a row; the coverage model fills
        # historical gaps so the lookup never shrinks to a recent subset.
        params = (filters.model_version, *params)
    cursor = conn.execute(
        f"""
        WITH ranked AS (
            SELECT
                f.*,
                ROW_NUMBER() OVER (
                    PARTITION BY f.release_run_id
                    ORDER BY {model_priority} DESC, {origin_order_sql('f')} DESC, f.forecast_origin_utc DESC
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
            is_backtest,
            (model_version <> %s) AS coverage_fallback,
            'available' AS prediction_status
        FROM ranked
        WHERE rank = 1
        ORDER BY opening_weekend_start DESC, title
        LIMIT 200
        """,
        (*params, filters.model_version),
    )
    rows = rows_as_dicts(cursor)
    promote_dashboard_amc_live_rows(conn, rows)
    for row in rows:
        row.setdefault("prediction_status", "available")
    return rows


def most_complete_model_version(conn: Any) -> str | None:
    if not forecast_tables_exist(conn):
        return None
    row = conn.execute(
        f"""
        SELECT model_version
        FROM {FORECAST_TABLE}
        GROUP BY model_version
        ORDER BY COUNT(DISTINCT release_run_id) DESC, MAX(created_at) DESC, model_version DESC
        LIMIT 1
        """
    ).fetchone()
    return str(row[0]) if row and row[0] else None


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
    grid_variant: str | None = None,
) -> dict[str, Any]:
    if not forecast_tables_exist(conn):
        return {
            "movie": None,
            "latest": None,
            "component_breakdown": [],
            "pre_release_breakdown": [],
            "timeline": pending_timeline([]),
            "chart": forecast_chart(pending_timeline([]), None),
            **empty_market_context(),
            "thursday_preview": empty_thursday_preview_context(),
            "model_versions": [],
            "selected_model_version": model_version or "",
            "distribution_policy": distribution_policy_context(model_version),
            "input_snapshot": None,
            "tables_ready": False,
        }

    versions = model_versions_for_release(conn, release_run_id)
    active_version = latest_model_version(conn)
    # A release-specific historical fallback is useful when the promoted model
    # has not emitted this movie yet, but the normal detail view must always
    # start on the promoted model when it is available for the release.
    selected_version = model_version or (
        active_version if active_version in versions else (versions[0] if versions else active_version)
    )
    rows = timeline_rows(conn, release_run_id=release_run_id, model_version=selected_version)
    components = component_rows(conn, [row["forecast_id"] for row in rows])
    attach_components(rows, components)
    latest = latest_opening_weekend_row(rows)
    preview_context = thursday_preview_context(conn, release_run_id=release_run_id)
    timeline = pending_timeline(rows)
    inject_thursday_preview_timeline_rows(timeline, rows, preview_context)
    market = market_context(
        conn,
        release_run_id=release_run_id,
        model_version=selected_version,
        timeline=timeline,
        latest=latest,
        requested_variant=grid_variant,
    )
    try:
        from models.boxoffice.input_snapshots import latest_input_snapshot

        input_snapshot = latest_input_snapshot(
            conn,
            release_run_id=release_run_id,
            model_version=selected_version,
        )
    except Exception:
        input_snapshot = None
    return {
        "movie": movie_from_rows(rows),
        "latest": latest,
        "component_breakdown": forecast_component_breakdown(latest),
        "pre_release_breakdown": pre_release_breakdown(timeline),
        "timeline": timeline,
        "chart": forecast_chart(timeline, latest),
        **market,
        "thursday_preview": preview_context,
        "model_versions": versions,
        "selected_model_version": selected_version or "",
        "distribution_policy": distribution_policy_context(selected_version),
        "input_snapshot": input_snapshot,
        "tables_ready": True,
    }


def empty_market_context() -> dict[str, Any]:
    return {
        "market_chart": {
            "has_grid": False,
            "reason": "Synthetic market unavailable",
            "width": 1180,
            "height": 420,
            "x_labels": x_axis_labels(width=1180, left=68, right=26),
        },
        "bucket_table": {"rows": [], "listing_origin": None, "previous_origin": None, "current_origin": None},
        "bucket_definition": None,
        "distribution_status": "full_distribution_unavailable",
        "selected_grid_variant": "canonical",
        "grid_variants_available": False,
    }


def parse_distribution_payload(value: object) -> ForecastDistribution | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    return ForecastDistribution.from_payload(value)


def immutable_market_anchor(
    conn: Any,
    *,
    release_run_id: int,
    model_version: str | None,
) -> dict[str, Any] | None:
    if not relation_exists(conn, "analytics.movie_forecast_emissions"):
        return None
    predicates = ["release_run_id = %s", "target = 'opening_weekend'", "origin_day IS NOT NULL"]
    params: list[Any] = [release_run_id]
    if model_version:
        predicates.append("model_version = %s")
        params.append(model_version)
    common = " AND ".join(predicates)
    fields = "emitted_forecast_id, payload_hash, origin_key, origin_day, point_usd, point_model, movie_id"
    early = conn.execute(
        f"""
        SELECT {fields}
        FROM analytics.movie_forecast_emissions
        WHERE {common} AND origin_day <= -10
        ORDER BY origin_day DESC, forecast_generated_at DESC
        LIMIT 1
        """,
        tuple(params),
    )
    rows = rows_as_dicts(early)
    if rows:
        return rows[0]
    later = conn.execute(
        f"""
        SELECT {fields}
        FROM analytics.movie_forecast_emissions
        WHERE {common} AND origin_day > -10 AND origin_day < 0
        ORDER BY origin_day ASC, forecast_generated_at DESC
        LIMIT 1
        """,
        tuple(params),
    )
    rows = rows_as_dicts(later)
    return rows[0] if rows else None


def polymarket_market_grid(
    conn: Any,
    *,
    release_run_id: int,
    anchor: dict[str, Any],
) -> PolymarketMarketGrid | None:
    required = (
        "prediction_market_backtest.polymarket_events",
        "prediction_market_backtest.polymarket_markets",
        "prediction_market_backtest.contract_semantics",
        "prediction_market_backtest.event_bucket_validations",
        "prediction_market_backtest.movie_matches",
    )
    if not all(relation_exists(conn, relation) for relation in required):
        return None
    movie_id = coerce_int(anchor.get("movie_id"))
    if movie_id is None:
        return None
    cursor = conn.execute(
        """
        SELECT
            e.event_id,
            e.event_slug,
            e.title AS event_title,
            m.market_id,
            m.question,
            cs.bucket_lower,
            cs.bucket_upper,
            cs.include_lower,
            cs.include_upper,
            cs.lower_unbounded,
            cs.upper_unbounded,
            v.validated_bucket_count
        FROM prediction_market_backtest.movie_matches mm
        JOIN prediction_market_backtest.polymarket_events e
          ON e.event_id = mm.event_id
        JOIN prediction_market_backtest.event_bucket_validations v
          ON v.event_id = e.event_id
        JOIN prediction_market_backtest.polymarket_markets m
          ON m.event_id = e.event_id
        JOIN prediction_market_backtest.contract_semantics cs
          ON cs.market_id = m.market_id
        WHERE mm.movie_id = %s
          AND COALESCE(mm.match_status, '') NOT ILIKE 'rejected%%'
          AND COALESCE(cs.target_metric, 'opening_weekend_gross') ILIKE '%%opening%%weekend%%'
          AND COALESCE(cs.currency, 'USD') = 'USD'
          AND COALESCE(cs.weekend_duration, 3) = 3
          AND COALESCE(cs.parse_status, 'parsed') NOT ILIKE 'rejected%%'
          AND LOWER(COALESCE(v.validation_status, 'valid')) IN ('valid', 'approved', 'validated')
          AND COALESCE(jsonb_array_length(v.validation_errors), 0) = 0
          AND COALESCE(v.validated_bucket_count, 0) = 5
        ORDER BY
            COALESCE(e.active, FALSE) DESC,
            COALESCE(e.closed, FALSE) ASC,
            e.updated_at_exchange DESC NULLS LAST,
            e.event_id,
            cs.bucket_lower NULLS FIRST,
            cs.bucket_upper NULLS LAST
        """,
        (movie_id,),
    )
    rows = rows_as_dicts(cursor)
    for event_rows in _group_rows_by_event(rows):
        grid = _polymarket_grid_from_rows(
            release_run_id=release_run_id,
            anchor=anchor,
            rows=event_rows,
        )
        if grid is not None:
            return grid
    return None


def _group_rows_by_event(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("event_id") or ""), []).append(row)
    return list(grouped.values())


def _polymarket_grid_from_rows(
    *,
    release_run_id: int,
    anchor: dict[str, Any],
    rows: list[dict[str, Any]],
) -> PolymarketMarketGrid | None:
    if len(rows) != 5:
        return None
    ordered = sorted(rows, key=_contract_bucket_sort_key)
    if not _is_complete_five_bucket_market(ordered):
        return None
    boundaries = tuple(int(round(float(ordered[index]["bucket_upper"]))) for index in range(4))
    if len(boundaries) != 4 or any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        return None
    widths = [right - left for left, right in zip(boundaries, boundaries[1:])]
    selected_width = widths[0] if widths and all(width == widths[0] for width in widths) else None
    anchor_point = int(round(float(anchor["point_usd"])))
    effective_origin = str(anchor.get("origin_key") or "")
    identity = {
        "release_run_id": int(release_run_id),
        "market_source": "polymarket",
        "event_id": str(ordered[0].get("event_id") or ""),
        "boundaries_usd": boundaries,
        "effective_listing_origin": effective_origin,
    }
    import hashlib

    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:20]
    return PolymarketMarketGrid(
        release_run_id=int(release_run_id),
        target_listing_origin=TARGET_LISTING_ORIGIN,
        effective_listing_origin=effective_origin,
        anchor_forecast_usd=anchor_point,
        anchor_forecast_emission_hash=str(anchor.get("payload_hash") or anchor.get("emitted_forecast_id") or ""),
        point_policy_version=str(anchor.get("point_model") or "unknown"),
        width_rule_version="polymarket_contract_semantics",
        rounding_rule_version="polymarket_contract_semantics",
        selected_width_usd=selected_width,
        ideal_start_usd=boundaries[0],
        rounded_start_usd=boundaries[0],
        boundaries_usd=boundaries,  # type: ignore[arg-type]
        rounding_variant="polymarket",
        width_selection_rule="polymarket_market_data",
        grid_id=f"polymarket-grid-{digest}",
        market_source="polymarket",
        event_id=str(ordered[0].get("event_id") or ""),
        event_slug=str(ordered[0].get("event_slug") or "") or None,
        event_title=str(ordered[0].get("event_title") or ""),
        market_ids=tuple(str(row.get("market_id") or "") for row in ordered),
        market_questions=tuple(str(row.get("question") or "") for row in ordered),
    )


def _contract_bucket_sort_key(row: dict[str, Any]) -> tuple[int, float]:
    lower = coerce_float(row.get("bucket_lower"))
    upper = coerce_float(row.get("bucket_upper"))
    if bool(row.get("lower_unbounded")) or lower is None:
        return (0, -math.inf)
    if bool(row.get("upper_unbounded")) or upper is None:
        return (2, lower)
    return (1, lower)


def _is_complete_five_bucket_market(rows: list[dict[str, Any]]) -> bool:
    if len(rows) != 5:
        return False
    if not bool(rows[0].get("lower_unbounded")) or coerce_float(rows[0].get("bucket_upper")) is None:
        return False
    if not bool(rows[-1].get("upper_unbounded")) or coerce_float(rows[-1].get("bucket_lower")) is None:
        return False
    for left, right in zip(rows, rows[1:]):
        left_upper = coerce_float(left.get("bucket_upper"))
        right_lower = coerce_float(right.get("bucket_lower"))
        if left_upper is None or right_lower is None or not math.isclose(left_upper, right_lower, abs_tol=0.5):
            return False
    return True


def market_context(
    conn: Any,
    *,
    release_run_id: int,
    model_version: str | None,
    timeline: list[dict[str, Any]],
    latest: dict[str, Any] | None,
    requested_variant: str | None,
) -> dict[str, Any]:
    context = empty_market_context()
    anchor = immutable_market_anchor(conn, release_run_id=release_run_id, model_version=model_version)
    anchor_point = coerce_float(anchor.get("point_usd")) if anchor else None
    effective_origin = str(anchor.get("origin_key") or "") if anchor else ""
    if anchor_point is None or not effective_origin:
        context["market_chart"] = {**context["market_chart"], "reason": "Immutable anchor forecast unavailable"}
        return context
    polymarket_grid = polymarket_market_grid(conn, release_run_id=release_run_id, anchor=anchor)
    if polymarket_grid is not None:
        probability_rows = market_probability_rows(timeline, polymarket_grid, effective_origin)
        context.update(
            {
                "market_chart": market_chart(timeline, latest, polymarket_grid, probability_rows),
                "bucket_table": bucket_probability_table(timeline, polymarket_grid, probability_rows, effective_origin),
                "bucket_definition": polymarket_grid.as_dict(),
                "distribution_status": distribution_status(probability_rows),
                "selected_grid_variant": "polymarket",
                "grid_variants_available": False,
            }
        )
        return context
    try:
        variants = generate_rounding_variants(
            release_run_id=release_run_id,
            effective_listing_origin=effective_origin,
            anchor_forecast_usd=anchor_point,
            anchor_forecast_emission_hash=str(anchor.get("payload_hash") or anchor.get("emitted_forecast_id") or ""),
            point_policy_version=str(anchor.get("point_model") or "unknown"),
        )
    except ValueError:
        context["market_chart"] = {**context["market_chart"], "reason": "Anchor forecast is below the $1M synthetic-market domain"}
        context["distribution_status"] = "out_of_domain"
        return context
    requested = str(requested_variant or "canonical")
    by_variant = {grid.rounding_variant: grid for grid in variants}
    # The lower outcome is the canonical grid; only a genuine tie exposes higher_strike.
    selected = by_variant.get("higher_strike") if requested == "higher_strike" else variants[0]
    selected = selected or variants[0]
    selected_query_variant = (
        requested
        if requested in {"canonical", "lower_strike", "higher_strike"}
        and (requested == "canonical" or len(variants) > 1)
        else "canonical"
    )
    probability_rows = market_probability_rows(timeline, selected, effective_origin)
    context.update(
        {
            "market_chart": market_chart(timeline, latest, selected, probability_rows),
            "bucket_table": bucket_probability_table(timeline, selected, probability_rows, effective_origin),
            "bucket_definition": selected.as_dict(),
            "distribution_status": distribution_status(probability_rows),
            "selected_grid_variant": selected_query_variant,
            "grid_variants_available": len(variants) > 1,
        }
    )
    return context


def market_probability_rows(
    timeline: list[dict[str, Any]],
    grid: MarketGrid,
    effective_origin: str,
) -> dict[str, dict[str, Any]]:
    first_index = origin_sort_value(effective_origin)
    results: dict[str, dict[str, Any]] = {}
    for row in timeline:
        key = str(row.get("origin_key") or "")
        if row.get("status") != "ready" or origin_sort_value(key) < first_index:
            continue
        distribution = parse_distribution_payload(row.get("distribution_payload"))
        if distribution is None:
            row["central50_lo_usd"] = None
            row["central50_hi_usd"] = None
            results[key] = {"status": "full_distribution_unavailable", "probabilities": None, "distribution": None}
            continue
        try:
            probabilities = distribution.bucket_probabilities(grid.boundaries_usd, grid_id=grid.grid_id)
        except ValueError:
            row["central50_lo_usd"] = None
            row["central50_hi_usd"] = None
            results[key] = {"status": "full_distribution_invalid", "probabilities": None, "distribution": None}
            continue
        row["central50_lo_usd"] = distribution.quantile(0.25)
        row["central50_hi_usd"] = distribution.quantile(0.75)
        row["distribution_policy"] = distribution.distribution_policy
        payload = row.get("distribution_payload")
        row["information_state"] = payload.get("information_state") if isinstance(payload, dict) else None
        results[key] = {"status": "available", "probabilities": probabilities, "distribution": distribution}
    return results


def distribution_status(probability_rows: dict[str, dict[str, Any]]) -> str:
    if any(row.get("status") == "available" for row in probability_rows.values()):
        return "available"
    if probability_rows:
        return "full_distribution_unavailable"
    return "not_listed"


def distribution_policy_context(model_version: str | None) -> dict[str, Any]:
    if not model_version:
        return {
            "policy_name": None,
            "status": None,
            "label": "-",
            "detail": "No promoted distribution policy",
            "tail_label": "-",
        }
    try:
        from models.boxoffice.artifacts import load_model_artifacts

        policy = load_model_artifacts(model_version).pre_release_distribution_policy
    except (FileNotFoundError, ValueError):
        policy = {}
    policy_name = str(policy.get("policy_name") or "") or None
    status = str(policy.get("status") or "") or None
    tail_enabled = bool(policy.get("tail_safety_enabled"))
    weight = policy.get("tail_contamination_weight")
    family = str(policy.get("tail_reference_family") or "")
    df = policy.get("tail_reference_df")
    if tail_enabled and weight is not None:
        tail_label = f"{float(weight) * 100:.0f}% {family or 'tail'}"
        if df:
            tail_label = f"{tail_label} df{df}"
    else:
        tail_label = "disabled"
    return {
        "policy_name": policy_name,
        "status": status,
        "label": policy_name or "-",
        "detail": "Production CDF policy" if status == "production" else "Distribution policy not promoted",
        "tail_label": tail_label,
        "base_distribution_policy": policy.get("base_distribution_policy"),
        "source_shrinkage_status": policy.get("source_shrinkage_status"),
        "market_calibration_status": policy.get("market_calibration_status"),
    }


def empty_thursday_preview_context() -> dict[str, Any]:
    return {
        "reported": None,
        "amc_shadow_rows": [],
        "latest_amc_shadow": None,
    }


def relation_exists(conn: Any, relation_name: str) -> bool:
    row = conn.execute("SELECT to_regclass(%s) IS NOT NULL", (relation_name,)).fetchone()
    return bool(row and row[0])


def table_columns(conn: Any, *, schema: str, table: str) -> set[str]:
    cursor = conn.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
        """,
        (schema, table),
    )
    return {str(row[0]) for row in cursor.fetchall()}


def optional_column_sql(column: str, available_columns: set[str], *, alias: str = "shadow") -> str:
    if column in available_columns:
        return f"{alias}.{column}"
    return f"NULL AS {column}"


def thursday_preview_context(conn: Any, *, release_run_id: int) -> dict[str, Any]:
    return {
        "reported": reported_thursday_preview(conn, release_run_id=release_run_id),
        "amc_shadow_rows": amc_shadow_preview_rows(conn, release_run_id=release_run_id),
        "latest_amc_shadow": latest_amc_shadow_preview(conn, release_run_id=release_run_id),
    }


def reported_thursday_preview(conn: Any, *, release_run_id: int) -> dict[str, Any] | None:
    if not relation_exists(conn, "daily_box_office"):
        return None
    cursor = conn.execute(
        """
        SELECT
            d.box_office_date::date AS preview_date,
            d.gross_usd,
            d.source,
            d.source_url,
            d.fetched_at
        FROM daily_box_office d
        WHERE d.release_run_id = %s
          AND COALESCE(d.is_preview, 0) = 1
          AND d.gross_usd > 0
        ORDER BY d.box_office_date::date DESC, d.fetched_at DESC NULLS LAST
        LIMIT 1
        """,
        (release_run_id,),
    )
    rows = rows_as_dicts(cursor)
    return rows[0] if rows else None


def amc_shadow_preview_rows(conn: Any, *, release_run_id: int) -> list[dict[str, Any]]:
    if not relation_exists(conn, AMC_SHADOW_TABLE):
        return []
    columns = table_columns(conn, schema="analytics", table="amc_opening_shadow_forecasts")
    cursor = conn.execute(
        f"""
        SELECT
            shadow.release_run_id,
            shadow.movie_id,
            shadow.title,
            shadow.origin_key,
            shadow.forecast_origin,
            shadow.forecast_origin_utc,
            shadow.as_of_utc,
            shadow.model_version,
            shadow.run_id,
            shadow.sample_key,
            {optional_column_sql("thursday_amc_seats_collected", columns)},
            {optional_column_sql("predicted_thursday_previews_usd", columns)},
            {optional_column_sql("thursday_amc_observed_preview_seats", columns)},
            {optional_column_sql("thursday_amc_shadow_ow_prior_usd", columns)},
            {optional_column_sql("thursday_amc_shadow_ow_prior_source", columns)},
            {optional_column_sql("thursday_preview_actual_usd", columns)},
            {optional_column_sql("friday_only_amc_observed_seats", columns)},
            shadow.coverage,
            shadow.snapshot_count,
            shadow.staleness_p50_minutes,
            shadow.feature_quality_bucket,
            {optional_column_sql("amc_candidate_available", columns)},
            {optional_column_sql("amc_candidate_unavailable_reason", columns)},
            shadow.opening_weekend_usd,
            shadow.production_daily_usd
        FROM {AMC_SHADOW_TABLE} shadow
        WHERE shadow.release_run_id = %s
        ORDER BY {origin_order_sql('shadow')}, shadow.as_of_utc DESC
        """,
        (release_run_id,),
    )
    return rows_as_dicts(cursor)


def latest_amc_shadow_preview(conn: Any, *, release_run_id: int) -> dict[str, Any] | None:
    rows = amc_shadow_preview_rows(conn, release_run_id=release_run_id)
    available = [
        row
        for row in rows
        if row.get("thursday_amc_seats_collected") and coerce_float(row.get("predicted_thursday_previews_usd")) is not None
    ]
    thursday_available = [row for row in available if str(row.get("origin_key") or "").startswith("THU_")]
    return thursday_available[-1] if thursday_available else available[-1] if available else (rows[-1] if rows else None)


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
    rows = rows_as_dicts(cursor)
    promote_timeline_amc_live_rows(conn, rows)
    suppress_stale_baseline_live_rows(rows)
    return rows


def is_baseline_live_row(row: dict[str, Any]) -> bool:
    return bool(row.get("is_live")) and str(row.get("component_source") or "") == BASELINE_COMPONENT_SOURCE


def is_amc_live_row(row: dict[str, Any]) -> bool:
    return bool(row.get("is_live")) and str(row.get("component_source") or "") in AMC_COMPONENT_SOURCES


def promote_dashboard_amc_live_rows(conn: Any, rows: list[dict[str, Any]]) -> None:
    """Replace baseline dashboard picks with the latest persisted AMC-live row.

    The active production artifact remains the dashboard default, but a live
    refresh can temporarily persist baseline-only rows for a movie that has
    real AMC snapshots. In that case the UI should surface the best stored AMC
    forecast instead of the newer-but-less-informative baseline row.
    """

    release_ids = sorted(
        {
            int(row["release_run_id"])
            for row in rows
            if row.get("release_run_id") is not None and is_baseline_live_row(row)
        }
    )
    if not release_ids:
        return
    cursor = conn.execute(
        f"""
        WITH ranked AS (
            SELECT
                f.*,
                ROW_NUMBER() OVER (
                    PARTITION BY f.release_run_id, f.target
                    ORDER BY {origin_order_sql('f')} DESC, f.forecast_origin_utc DESC, f.created_at DESC
                ) AS rank
            FROM {FORECAST_TABLE} f
            WHERE f.release_run_id = ANY(%s)
              AND f.target = ANY(%s)
              AND f.is_live
              AND f.component_source = ANY(%s)
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
        """,
        (release_ids, list(TARGETS), AMC_COMPONENT_SOURCES),
    )
    replacements = {
        (int(row["release_run_id"]), str(row["target"])): row
        for row in rows_as_dicts(cursor)
        if row.get("release_run_id") is not None
    }
    for index, row in enumerate(rows):
        key = (int(row["release_run_id"]), str(row["target"]))
        if is_baseline_live_row(row) and key in replacements:
            rows[index] = replacements[key]


def promote_timeline_amc_live_rows(conn: Any, rows: list[dict[str, Any]]) -> None:
    release_ids = sorted({int(row["release_run_id"]) for row in rows if row.get("release_run_id") is not None})
    if not release_ids:
        return
    cursor = conn.execute(
        f"""
        WITH ranked AS (
            SELECT
                f.*,
                ROW_NUMBER() OVER (
                    PARTITION BY f.release_run_id, f.origin_key, f.target
                    ORDER BY f.forecast_origin_utc DESC, f.created_at DESC, f.model_version DESC
                ) AS rank
            FROM {FORECAST_TABLE} f
            WHERE f.release_run_id = ANY(%s)
              AND f.is_live
              AND f.component_source = ANY(%s)
        )
        SELECT *
        FROM ranked
        WHERE rank = 1
        """,
        (release_ids, AMC_COMPONENT_SOURCES),
    )
    replacements = {
        (int(row["release_run_id"]), str(row["origin_key"]), str(row["target"])): row
        for row in rows_as_dicts(cursor)
        if row.get("release_run_id") is not None
    }
    for index, row in enumerate(rows):
        key = (int(row["release_run_id"]), str(row["origin_key"]), str(row["target"]))
        if is_baseline_live_row(row) and key in replacements:
            rows[index] = replacements[key]


def suppress_stale_baseline_live_rows(rows: list[dict[str, Any]]) -> None:
    latest_amc_origin_by_release: dict[int, int] = {}
    for row in rows:
        if row.get("target") != "opening_weekend" or not is_amc_live_row(row):
            continue
        release_run_id = int(row["release_run_id"])
        latest_amc_origin_by_release[release_run_id] = max(
            latest_amc_origin_by_release.get(release_run_id, -1),
            origin_sort_value(str(row.get("origin_key") or "")),
        )
    if not latest_amc_origin_by_release:
        return
    rows[:] = [
        row
        for row in rows
        if not (
            is_baseline_live_row(row)
            and int(row["release_run_id"]) in latest_amc_origin_by_release
            and origin_sort_value(str(row.get("origin_key") or ""))
            > latest_amc_origin_by_release[int(row["release_run_id"])]
        )
    ]


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
        payload = row.get("distribution_payload")
        if isinstance(payload, dict):
            row["information_state"] = payload.get("information_state")


def forecast_component_breakdown(latest: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Shape stored component rows into a decision-facing latest forecast view."""

    if not latest:
        return []
    payload = latest.get("distribution_payload")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = None
    payload = payload if isinstance(payload, dict) else {}
    actual_provenance = payload.get("actual_provenance") if isinstance(payload.get("actual_provenance"), dict) else {}
    total = coerce_float(latest.get("point_usd"))
    origin = str(latest.get("origin_key") or "latest refresh")
    snapshots = coerce_int(latest.get("amc_snapshot_count"))
    coverage = coerce_float(latest.get("amc_coverage"))
    breakdown: list[dict[str, Any]] = []

    for component in latest.get("components") or []:
        day = str(component.get("component_day") or "Weekend component")
        component_type = str(component.get("component_type") or "unknown")
        point = coerce_float(component.get("component_point_usd"))
        source = str(component.get("component_source") or component.get("component_model") or "Model estimate")
        notes = str(component.get("component_notes") or "").strip()
        detail = source
        window = "Weekend day"
        uncertainty_label = "80% likely range"

        if component_type == "actual":
            provenance = actual_provenance.get(day) if isinstance(actual_provenance.get(day), dict) else {}
            actual_source = str(provenance.get("actual_source") or source or "reported box office")
            source = "Reported box office"
            detail = f"Reported by {actual_source}"
            uncertainty_label = "Observed result"
        elif component_type == "AMC_nowcast":
            source = "AMC seats and showtimes"
            detail_parts = ["Intraday AMC signal"]
            if snapshots is not None:
                detail_parts.append(f"{snapshots} snapshots")
            if coverage is not None:
                detail_parts.append(f"{percent(coverage)} coverage")
            if notes:
                detail_parts.append(notes.replace("_", " "))
            detail = " · ".join(detail_parts)
            window = f"Updated {origin.replace('_', ' ')}"
        elif component_type == "baseline":
            source = "Historical daily pattern"
            detail = str(component.get("component_model") or "Daily baseline")
            if notes:
                detail = f"{detail} · {notes}"
        elif notes:
            detail = f"{detail} · {notes}"

        share = point / total if point is not None and total is not None and total > 0 else None
        breakdown.append(
            {
                "day": day,
                "window": window,
                "source": source,
                "source_detail": detail,
                "point_usd": component.get("component_point_usd"),
                "share": share,
                "lo80_usd": component.get("component_lo80_usd"),
                "hi80_usd": component.get("component_hi80_usd"),
                "uncertainty_label": uncertainty_label,
                "is_observed": component_type == "actual",
            }
        )
    return breakdown


def pre_release_breakdown(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return each stored pre-release consensus window in display order."""

    rows: list[dict[str, Any]] = []
    for row in timeline:
        origin_key = str(row.get("origin_key") or "")
        if row.get("status") != "ready" or not origin_key.startswith("P_"):
            continue
        sources = display_estimate_sources(row.get("estimate_sources"))
        source_count = coerce_int(row.get("source_count"))
        rows.append(
            {
                "window": pre_release_window_label(origin_key),
                "origin_key": origin_key,
                "point_usd": row.get("point_usd"),
                "source_count": source_count if source_count is not None else len(sources) or None,
                "sources": sources,
                "lo80_usd": row.get("lo80_usd"),
                "hi80_usd": row.get("hi80_usd"),
            }
        )
    return rows


def pre_release_window_label(origin_key: str) -> str:
    try:
        days = abs(int(origin_key.split("_", 1)[1]))
    except (IndexError, ValueError):
        return origin_key.replace("_", " ")
    return f"{days} day out" if days == 1 else f"{days} days out"


def display_estimate_sources(value: object) -> list[str]:
    if not value:
        return []
    labels = {
        "boxofficepro": "Boxoffice Pro",
        "boxofficeguru": "Box Office Guru",
        "boxofficetheory": "Box Office Theory",
        "boxofficetheory_substack": "Box Office Theory",
        "edwarddouglas_substack": "Edward Douglas",
        "joblo": "JoBlo",
        "toddmthatcher": "Todd M. Thatcher",
        "the_numbers": "The Numbers",
    }
    raw_sources = [item.strip() for item in str(value).split(",") if item.strip()]
    return [labels.get(item.lower(), item.replace("_", " ").title()) for item in raw_sources]


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


def inject_thursday_preview_timeline_rows(
    timeline: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    preview_context: dict[str, Any],
) -> None:
    """Insert UI-only Thursday preview prior points between P_-1 and Live Friday."""

    for shadow in preview_context.get("amc_shadow_rows") or []:
        origin_key = str(shadow.get("origin_key") or "")
        if not origin_key.startswith("THU_"):
            continue
        point = coerce_float(shadow.get("thursday_amc_shadow_ow_prior_usd") or shadow.get("opening_weekend_usd"))
        if point is None:
            continue
        virtual_shadow = {
            "forecast_id": f"thursday-amc-shadow-{shadow.get('release_run_id')}-{origin_key}",
            "run_id": shadow.get("run_id"),
            "model_version": shadow.get("model_version"),
            "movie_id": shadow.get("movie_id"),
            "release_run_id": shadow.get("release_run_id"),
            "title": shadow.get("title"),
            "opening_weekend_start": None,
            "regime": "thursday_preview",
            "origin_key": origin_key,
            "forecast_origin_utc": shadow.get("forecast_origin_utc"),
            "as_of_utc": shadow.get("as_of_utc"),
            "target": "opening_weekend",
            "point_usd": point,
            "lo80_usd": None,
            "hi80_usd": None,
            "lo95_usd": None,
            "hi95_usd": None,
            "point_model": "thursday_amc_preview_shadow_prior",
            "interval_model": "shadow_no_interval",
            "component_source": shadow.get("thursday_amc_shadow_ow_prior_source") or "thursday_amc_preview_nowcast",
            "feature_quality_bucket": shadow.get("feature_quality_bucket"),
            "actual_usd": None,
            "log_error": None,
            "abs_pct_error": None,
            "components": [
                {
                    "component_day": "Thursday Preview",
                    "component_type": "AMC_nowcast",
                    "component_point_usd": shadow.get("predicted_thursday_previews_usd"),
                    "component_sigma_log": None,
                    "component_lo80_usd": None,
                    "component_hi80_usd": None,
                    "component_lo95_usd": None,
                    "component_hi95_usd": None,
                    "component_model": "thursday_amc_preview_nowcast",
                    "component_source": shadow.get("thursday_amc_shadow_ow_prior_source") or "thursday_amc_preview_nowcast",
                    "component_notes": (
                        f"OW shadow {compact_usd(point)}"
                        if point is not None
                        else "OW shadow unavailable"
                    ),
                }
            ],
            "status": "ready",
            "regime_label": REGIME_LABELS["thursday_preview"],
        }
        virtual_shadow["day_components"] = day_components(virtual_shadow["components"])
        replace_timeline_origin(timeline, origin_key, virtual_shadow)

    if not preview_context.get("reported"):
        return
    candidates = [
        row
        for row in rows
        if row.get("target") == "opening_weekend"
        and str(row.get("component_source") or "") == "thursday_preview_actual"
    ]
    if not candidates:
        return
    source = sorted(candidates, key=lambda row: origin_sort_value(str(row.get("origin_key") or "")))[0]
    virtual = dict(source)
    virtual.update(
        {
            "origin_key": "THU_EOD",
            "regime": "thursday_preview",
            "regime_label": REGIME_LABELS["thursday_preview"],
            "status": "ready",
            "point_model": "thursday_preview_actual_prior",
            "component_source": "thursday_preview_actual",
            "day_components": day_components(source.get("components") or []),
            "virtual_origin_source": source.get("origin_key"),
        }
    )
    replace_timeline_origin(timeline, "THU_EOD", virtual)


def replace_timeline_origin(timeline: list[dict[str, Any]], origin_key: str, row: dict[str, Any]) -> None:
    for index, item in enumerate(timeline):
        if item.get("origin_key") == origin_key:
            timeline[index] = row
            return


def day_components(components: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(component.get("component_day")): component for component in components}


def bucket_label(
    index: int,
    boundaries: tuple[int, int, int, int],
    *,
    market_questions: tuple[str, ...] = (),
) -> str:
    """Use the contract wording when a validated Polymarket bucket is available."""
    if len(market_questions) == 5 and market_questions[index].strip():
        return market_questions[index]
    if index == 0:
        return f"Below {compact_usd(boundaries[0])}"
    if index == 4:
        return f"{compact_usd(boundaries[-1])} or more"
    return f"{compact_usd(boundaries[index - 1])}-{compact_usd(boundaries[index])}"


def probability_color(probability: float) -> str:
    """Use a legible 0%-50%+ teal heat scale across every movie chart."""

    # A square-root scale keeps plausible-but-small buckets visible while still
    # giving the leading bucket the strongest color.
    intensity = math.sqrt(max(0.0, min(1.0, probability / 0.5)))
    start = (244, 249, 247)
    end = (8, 99, 81)
    rgb = tuple(round(left + (right - left) * intensity) for left, right in zip(start, end))
    return "#%02x%02x%02x" % rgb


def market_chart(
    timeline: list[dict[str, Any]],
    latest: dict[str, Any] | None,
    grid: MarketGrid,
    probability_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    # The bucket labels are the useful vertical scale for this chart. Reserve
    # enough room for them and leave the numeric gridlines unlabeled so the two
    # scales do not collide in the left margin.
    width, height, left, right, top, bottom = 1180, 460, 148, 28, 30, 58
    plot_width, plot_height = width - left - right, height - top - bottom
    listing_index = origin_sort_value(grid.effective_listing_origin)
    values: list[float] = [float(boundary) for boundary in grid.boundaries_usd]
    points: list[dict[str, Any]] = []
    distributions: list[ForecastDistribution] = []
    for index, row in enumerate(timeline):
        if row.get("status") != "ready":
            continue
        point = coerce_float(row.get("point_usd"))
        if point is None:
            continue
        values.append(point)
        probability = probability_rows.get(str(row.get("origin_key") or ""))
        if probability and isinstance(probability.get("distribution"), ForecastDistribution):
            distributions.append(probability["distribution"])
        points.append(
            {
                "index": index,
                "origin_key": row.get("origin_key"),
                "regime": row.get("regime"),
                "point": point,
                "distribution_policy": (
                    probability["distribution"].distribution_policy
                    if probability and probability.get("distribution")
                    else "-"
                ),
            }
        )
    for distribution in distributions:
        values.extend([float(distribution.quantile(0.01)), float(distribution.quantile(0.99))])
    actual = coerce_float(latest.get("actual_usd")) if latest else None
    if actual is not None:
        values.append(actual)
    low, high = min(values), max(values)
    if high <= low:
        high = low + max(low * 0.1, 1.0)
    padding = (high - low) * 0.08
    y_min, y_max = max(0.0, low - padding), high + padding

    column_width = plot_width / len(EXPECTED_ORIGIN_KEYS)

    def x_for(index: int) -> float:
        """Return the center of a timeline bucket, not its boundary."""
        return left + column_width * (index + 0.5)

    def x_edge(index: int) -> float:
        return left + column_width * index

    boundaries = grid.boundaries_usd
    bucket_height = plot_height / 5

    def y_for(value: float) -> float:
        """Map dollars into five equal-height market bucket regions."""
        if value < boundaries[0]:
            span = max(float(boundaries[0]) - y_min, 1.0)
            fraction = (float(boundaries[0]) - value) / span
            return top + bucket_height * (4 + max(0.0, min(1.0, fraction)))
        if value < boundaries[1]:
            fraction = (float(boundaries[1]) - value) / max(float(boundaries[1] - boundaries[0]), 1.0)
            return top + bucket_height * (3 + fraction)
        if value < boundaries[2]:
            fraction = (float(boundaries[2]) - value) / max(float(boundaries[2] - boundaries[1]), 1.0)
            return top + bucket_height * (2 + fraction)
        if value < boundaries[3]:
            fraction = (float(boundaries[3]) - value) / max(float(boundaries[3] - boundaries[2]), 1.0)
            return top + bucket_height * (1 + fraction)
        span = max(y_max - float(boundaries[3]), 1.0)
        fraction = (y_max - value) / span
        return top + bucket_height * max(0.0, min(1.0, fraction))

    for point in points:
        point["x"] = round(x_for(int(point["index"])), 2)
        point["y"] = round(y_for(float(point["point"])), 2)
        point["tooltip"] = (
            f"{point['origin_key']}: {compact_usd(point['point'])}\n"
            f"point forecast: {compact_usd(point['point'])}\n"
            f"distribution policy: {point['distribution_policy']}\n"
            f"grid ID: {grid.grid_id}"
        )
    ranges = [
        (4, top, top + bucket_height),
        (3, top + bucket_height, top + bucket_height * 2),
        (2, top + bucket_height * 2, top + bucket_height * 3),
        (1, top + bucket_height * 3, top + bucket_height * 4),
        (0, top + bucket_height * 4, top + plot_height),
    ]
    regions = [
        {
            "index": index,
            # Contract wording belongs in the table.  The chart needs concise,
            # scannable labels that fit inside its fixed left margin.
            "label": bucket_label(index, boundaries),
            "tail": "upper" if index == 4 else "lower" if index == 0 else None,
            "y": round(y_start, 2),
            "height": round(y_end - y_start, 2),
        }
        for index, y_start, y_end in ranges
    ]
    cells: list[dict[str, Any]] = []
    latest_probability_origin = next(
        (
            str(row.get("origin_key") or "")
            for row in reversed(timeline)
            if probability_rows.get(str(row.get("origin_key") or ""), {}).get("probabilities") is not None
        ),
        None,
    )
    for index, row in enumerate(timeline):
        key = str(row.get("origin_key") or "")
        if index < listing_index:
            continue
        probability_data = probability_rows.get(key)
        probabilities = probability_data.get("probabilities") if probability_data else None
        status = str(probability_data.get("status") if probability_data else "full_distribution_unavailable")
        for region in regions:
            probability = probabilities[region["index"]] if probabilities is not None else None
            cells.append(
                {
                    "origin_key": key,
                    "bucket_index": region["index"],
                    "x": round(x_edge(index) + 0.5, 2),
                    "width": round(column_width - 1, 2),
                    "y": region["y"],
                    "height": region["height"],
                    "probability": probability,
                    "color": probability_color(float(probability)) if probability is not None else "url(#missing-probability)",
                    # The 42-column chart has no room for numeric labels.
                    # Current values are rendered in a readable table below.
                    "show_label": False,
                    "label": percent(probability),
                    "label_tone": "light" if probability is not None and probability >= 0.35 else "dark",
                    "tooltip": (
                        f"origin: {key}\n"
                        f"bucket: {region['label']}\n"
                        f"model probability: {percent(probability)}\n"
                        f"point forecast: {compact_usd(row.get('point_usd'))}\n"
                        f"distribution policy: {probability_data['distribution'].distribution_policy if probability_data and probability_data.get('distribution') else '-'}\n"
                        f"grid ID: {grid.grid_id}\n"
                        f"probability status: {status}"
                    ),
                }
            )
    actual_bucket_index = None
    if actual is not None:
        actual_bucket_index = next((index for index, boundary in enumerate(boundaries) if actual < boundary), 4)
    latest_probabilities = []
    if latest_probability_origin:
        values = probability_rows[latest_probability_origin]["probabilities"]
        latest_probabilities = [
            {"label": region["label"], "probability": values[region["index"]]}
            for region in reversed(regions)
        ]
    return {
        "has_grid": True,
        "width": width,
        "height": height,
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "points": points,
        "point_path": line_path(points, "y"),
        # Bucket boundaries, rather than arbitrary dollar ticks, define the
        # vertical grid on a market-probability chart.
        "y_ticks": [],
        "x_labels": x_axis_labels(width=width, left=left, right=right),
        "regime_markers": regime_markers(width=width, left=left, right=right),
        "bucket_regions": regions,
        "boundary_lines": [{"value": value, "y": round(y_for(value), 2)} for value in boundaries],
        "cells": cells,
        "has_probabilities": latest_probability_origin is not None,
        "latest_probability_origin": latest_probability_origin,
        "latest_probabilities": latest_probabilities,
        "listing_x": round(x_edge(listing_index), 2),
        "listing_label": grid.effective_listing_origin,
        "pre_listing_width": max(0, round(x_edge(listing_index) - left, 2)),
        "actual_y": round(y_for(actual), 2) if actual is not None else None,
        "actual_label": compact_usd(actual),
        "actual_bucket_index": actual_bucket_index,
        "realized_y": next((region["y"] for region in regions if region["index"] == actual_bucket_index), None),
        "realized_height": next((region["height"] for region in regions if region["index"] == actual_bucket_index), None),
        "realized_x": round(x_edge(listing_index), 2),
        "realized_width": round(width - right - x_edge(listing_index), 2),
        "anchor_x": round(x_for(listing_index), 2),
        "anchor_y": round(y_for(grid.anchor_forecast_usd), 2),
        "anchor_label": compact_usd(grid.anchor_forecast_usd),
    }


def bucket_probability_table(
    timeline: list[dict[str, Any]],
    grid: MarketGrid,
    probability_rows: dict[str, dict[str, Any]],
    effective_origin: str,
) -> dict[str, Any]:
    ready = [row for row in timeline if row.get("status") == "ready" and origin_sort_value(str(row.get("origin_key") or "")) >= origin_sort_value(effective_origin)]
    current = ready[-1] if ready else None
    previous = ready[-2] if len(ready) > 1 else None
    listing = next((row for row in ready if row.get("origin_key") == effective_origin), None)

    def values(row: dict[str, Any] | None) -> list[float] | None:
        if not row:
            return None
        data = probability_rows.get(str(row.get("origin_key") or ""))
        return data.get("probabilities") if data else None

    listing_values, previous_values, current_values = values(listing), values(previous), values(current)
    ordering = sorted(
        range(5),
        key=lambda index: (current_values is None, -(current_values[index] if current_values is not None else 0), index),
    )
    ranks = {index: rank + 1 for rank, index in enumerate(ordering)} if current_values is not None else {}
    market_questions = tuple(getattr(grid, "market_questions", ()))
    rows = []
    for index in range(5):
        current_probability = current_values[index] if current_values is not None else None
        previous_probability = previous_values[index] if previous_values is not None else None
        rows.append(
            {
                "label": bucket_label(index, grid.boundaries_usd, market_questions=market_questions),
                "listing_probability": listing_values[index] if listing_values is not None else None,
                "previous_probability": previous_probability,
                "current_probability": current_probability,
                "change": current_probability - previous_probability if current_probability is not None and previous_probability is not None else None,
                "rank": ranks.get(index),
            }
        )
    return {
        "rows": rows,
        "listing_origin": listing.get("origin_key") if listing else effective_origin,
        "previous_origin": previous.get("origin_key") if previous else None,
        "current_origin": current.get("origin_key") if current else None,
        "listing_total": sum(listing_values) if listing_values is not None else None,
        "previous_total": sum(previous_values) if previous_values is not None else None,
        "current_total": sum(current_values) if current_values is not None else None,
    }


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

    def multiple(value: float | None, point: float) -> float | None:
        return value / point if value is not None and point > 0 else None

    def multiple_label(value: float | None) -> str:
        return "-" if value is None else f"{value:.2f}x"

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
                "target": row.get("target"),
                "model_version": row.get("model_version"),
                "interval_model": row.get("interval_model"),
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
                "tooltip": (
                    f"{item.get('origin_key')}: {compact_usd(item['point'])}\n"
                    f"target: {item.get('target') or '-'}\n"
                    f"model: {item.get('model_version') or '-'}\n"
                    f"interval: {item.get('interval_model') or '-'}\n"
                    f"80% lo/hi/width: "
                    f"{multiple_label(multiple(item['lo80'], item['point']))} / "
                    f"{multiple_label(multiple(item['hi80'], item['point']))} / "
                    f"{multiple_label((item['hi80'] - item['lo80']) / item['point'] if item['lo80'] is not None and item['hi80'] is not None and item['point'] > 0 else None)}\n"
                    f"95% lo/hi: "
                    f"{multiple_label(multiple(item['lo95'], item['point']))} / "
                    f"{multiple_label(multiple(item['hi95'], item['point']))}"
                ),
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
    column_width = plot_width / len(EXPECTED_ORIGIN_KEYS)
    # These stages are deliberately spaced across the 42-point timeline. The
    # previous P_-1 and THU_10:00 labels occupied adjacent columns and
    # overlapped on screen.
    label_keys = [
        ("P_-14", "14 days out"),
        ("P_-1", "1 day out"),
        ("THU_EOD", "Thursday"),
        ("FRI_EOD", "Friday"),
        ("SAT_EOD", "Saturday"),
        ("SUN_EOD", "Sunday"),
    ]
    return [
        {
            "origin_key": key,
            "label": label,
            "x": round(left + column_width * (origin_sort_value(key) + 0.5), 2),
        }
        for key, label in label_keys
    ]


def regime_markers(*, width: int, left: int, right: int) -> list[dict[str, Any]]:
    plot_width = width - left - right
    column_width = plot_width / len(EXPECTED_ORIGIN_KEYS)
    markers = [
        ("P_-14", ""),
        ("THU_10:00", ""),
        ("FRI_10:00", ""),
        ("SAT_10:00", ""),
        ("SUN_10:00", ""),
    ]
    return [
        {
            "origin_key": key,
            "label": label,
            # Regime markers separate columns, so they belong on the leading
            # edge of the first bucket in each regime.
            "x": round(left + column_width * origin_sort_value(key), 2),
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
    if origin_key.startswith("THU_"):
        return "thursday_preview"
    if origin_key.startswith("FRI_"):
        return "live_friday"
    if origin_key.startswith("SAT_"):
        return "live_saturday"
    if origin_key.startswith("SUN_"):
        return "live_sunday"
    return "unknown"


def origin_sort_value(origin_key: str) -> int:
    expected_index = {origin: index for index, origin in enumerate(EXPECTED_ORIGIN_KEYS)}
    if origin_key in expected_index:
        return expected_index[origin_key]
    if origin_key.startswith("P_"):
        try:
            return int(origin_key.split("_", 1)[1]) + 14
        except ValueError:
            return 0
    return 999


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


def coerce_int(value: object) -> int | None:
    number = coerce_float(value)
    return int(number) if number is not None and math.isfinite(number) else None


def date_value(value: object) -> str:
    if isinstance(value, dt.date):
        return value.isoformat()
    return str(value or "")
