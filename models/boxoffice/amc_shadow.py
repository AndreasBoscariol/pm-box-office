"""Immutable shadow records for genuinely prospective AMC opening forecasts."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .artifacts import ModelArtifacts
from .live_composition import select_daily_baseline_row
from .schema import ForecastResult, MovieOpening

SHADOW_TABLE = "analytics.amc_opening_shadow_forecasts"
SHADOW_ORIGINS = {"FRI_12:00", "FRI_16:00", "FRI_20:00", "FRI_EOD"}
PREVIEW_TREATMENT = "thursday_amc_excluded; reported_friday_may_include_previews"
THURSDAY_PREVIEW_TREATMENT = "thursday_amc_preview_shadow; production_uses_reported_preview_actual_only"

SHADOW_COLUMNS = [
    "release_run_id", "movie_id", "title", "opening_weekend_start", "origin_key",
    "forecast_origin", "forecast_origin_utc", "as_of_utc", "model_version", "run_id",
    "sample_key", "multiplicative_daily_usd", "additive_daily_usd", "hybrid_daily_usd",
    "production_daily_usd", "baseline_daily_usd", "provisional_blend_daily_usd",
    "daily_lo80_usd", "daily_hi80_usd", "daily_lo95_usd", "daily_hi95_usd",
    "opening_weekend_usd", "opening_weekend_lo80_usd", "opening_weekend_hi80_usd",
    "opening_weekend_lo95_usd", "opening_weekend_hi95_usd", "coverage", "snapshot_count",
    "staleness_p50_minutes", "feature_quality_bucket", "thursday_preview_treatment",
    "amc_candidate_available", "amc_candidate_unavailable_reason", "thursday_amc_seats_collected",
    "predicted_thursday_previews_usd", "thursday_amc_observed_preview_seats",
    "predicted_thursday_previews_lo80_usd", "predicted_thursday_previews_hi80_usd",
    "predicted_thursday_previews_lo95_usd", "predicted_thursday_previews_hi95_usd",
    "preview_residual_pool_scope",
    "thursday_amc_shadow_ow_prior_usd", "thursday_amc_shadow_ow_prior_source",
    "thursday_amc_shadow_ow_lo80_usd", "thursday_amc_shadow_ow_hi80_usd",
    "thursday_amc_shadow_ow_lo95_usd", "thursday_amc_shadow_ow_hi95_usd",
    "thursday_selected_candidate", "thursday_target_classification", "thursday_training_cutoff",
    "friday_only_amc_observed_seats",
    "evaluation_target_includes_previews", "reported_friday_gross_usd",
    "thursday_preview_actual_usd", "friday_calendar_actual_usd",
    "actual_friday_usd", "actual_opening_weekend_usd",
]


def build_shadow_rows(
    *,
    movie: MovieOpening,
    results: Iterable[ForecastResult],
    live_plugin: pd.DataFrame,
    artifacts: ModelArtifacts,
    thursday_preview_nowcasts: pd.DataFrame | None = None,
) -> list[dict[str, object]]:
    by_origin_target = {(item.origin.origin_key, item.target): item for item in results}
    try:
        baseline = select_daily_baseline_row(artifacts.daily_baseline, movie)
    except (KeyError, ValueError):
        return []
    rows = []
    for origin_key in sorted(SHADOW_ORIGINS):
        daily = by_origin_target.get((origin_key, "friday"))
        weekend = by_origin_target.get((origin_key, "opening_weekend"))
        if daily is None or weekend is None:
            continue
        plugin_movie = live_plugin.get("movie_id", pd.Series(index=live_plugin.index, dtype=object))
        plugin_regime = live_plugin.get("regime", pd.Series(index=live_plugin.index, dtype=object))
        plugin_origin = live_plugin.get("forecast_origin", pd.Series(index=live_plugin.index, dtype=object))
        plugin_rows = live_plugin.loc[
            pd.to_numeric(plugin_movie, errors="coerce").eq(movie.movie_id)
            & plugin_regime.astype(str).eq("live_friday")
            & plugin_origin.astype(str).eq(str(daily.origin.forecast_origin))
        ]
        plugin = plugin_rows.iloc[-1] if not plugin_rows.empty else pd.Series(dtype=object)
        thursday_nowcast = select_thursday_preview_nowcast(
            thursday_preview_nowcasts,
            movie_id=movie.movie_id,
            forecast_origin=daily.origin.forecast_origin,
        )
        baseline_daily = finite(baseline.get("pre_fri_usd"))
        production = finite(daily.point_usd)
        amc_point = finite(plugin.get("pred_daily_gross_multiplicative_usd"))
        beta = 0.30 if str(daily.origin.forecast_origin) in {"10:00", "12:00", "14:00"} else 0.44
        provisional_blend = (
            baseline_daily * math.exp(beta * math.log(amc_point / baseline_daily))
            if positive(amc_point) and positive(baseline_daily)
            else math.nan
        )
        candidate_available = positive(amc_point)
        rows.append({
            "release_run_id": movie.release_run_id,
            "movie_id": movie.movie_id,
            "title": movie.title,
            "opening_weekend_start": movie.opening_weekend_start,
            "origin_key": origin_key,
            "forecast_origin": daily.origin.forecast_origin,
            "forecast_origin_utc": daily.origin.forecast_origin_utc,
            "as_of_utc": daily.origin.as_of_utc,
            "model_version": artifacts.model_version,
            "run_id": daily.run_id,
            "sample_key": plugin.get("sample_key"),
            "multiplicative_daily_usd": plugin.get("pred_daily_gross_multiplicative_usd"),
            "additive_daily_usd": plugin.get("pred_daily_gross_additive_usd"),
            "hybrid_daily_usd": plugin.get("pred_daily_gross_hybrid_usd"),
            "production_daily_usd": daily.point_usd,
            "baseline_daily_usd": baseline_daily,
            "provisional_blend_daily_usd": provisional_blend,
            "daily_lo80_usd": daily.lo80_usd,
            "daily_hi80_usd": daily.hi80_usd,
            "daily_lo95_usd": daily.lo95_usd,
            "daily_hi95_usd": daily.hi95_usd,
            "opening_weekend_usd": weekend.point_usd,
            "opening_weekend_lo80_usd": weekend.lo80_usd,
            "opening_weekend_hi80_usd": weekend.hi80_usd,
            "opening_weekend_lo95_usd": weekend.lo95_usd,
            "opening_weekend_hi95_usd": weekend.hi95_usd,
            "coverage": plugin.get("amc_coverage"),
            "snapshot_count": plugin.get("amc_snapshot_count"),
            "staleness_p50_minutes": plugin.get("amc_staleness_p50_minutes"),
            "feature_quality_bucket": plugin.get("feature_quality_bucket"),
            "thursday_preview_treatment": PREVIEW_TREATMENT,
            "amc_candidate_available": candidate_available,
            "amc_candidate_unavailable_reason": None if candidate_available else "no_eligible_showtimes_or_insufficient_training_snapshots",
            "thursday_amc_seats_collected": bool(thursday_nowcast.get("thursday_amc_seats_collected", False)),
            "predicted_thursday_previews_usd": thursday_nowcast.get("predicted_thursday_previews_usd"),
            "thursday_amc_observed_preview_seats": thursday_nowcast.get("amc_observed_preview_seats"),
            "predicted_thursday_previews_lo80_usd": thursday_nowcast.get("predicted_thursday_previews_lo80_usd"),
            "predicted_thursday_previews_hi80_usd": thursday_nowcast.get("predicted_thursday_previews_hi80_usd"),
            "predicted_thursday_previews_lo95_usd": thursday_nowcast.get("predicted_thursday_previews_lo95_usd"),
            "predicted_thursday_previews_hi95_usd": thursday_nowcast.get("predicted_thursday_previews_hi95_usd"),
            "preview_residual_pool_scope": thursday_nowcast.get("preview_residual_pool_scope"),
            "thursday_amc_shadow_ow_prior_usd": thursday_nowcast.get("shadow_ow_prior_usd"),
            "thursday_amc_shadow_ow_prior_source": thursday_nowcast.get("shadow_ow_prior_source"),
            "thursday_amc_shadow_ow_lo80_usd": thursday_nowcast.get("shadow_ow_lo80_usd"),
            "thursday_amc_shadow_ow_hi80_usd": thursday_nowcast.get("shadow_ow_hi80_usd"),
            "thursday_amc_shadow_ow_lo95_usd": thursday_nowcast.get("shadow_ow_lo95_usd"),
            "thursday_amc_shadow_ow_hi95_usd": thursday_nowcast.get("shadow_ow_hi95_usd"),
            "thursday_selected_candidate": thursday_nowcast.get("selected_candidate"),
            "thursday_target_classification": thursday_nowcast.get("target_classification"),
            "thursday_training_cutoff": thursday_nowcast.get("training_cutoff"),
            "friday_only_amc_observed_seats": plugin.get("amc_observed_seats"),
            "evaluation_target_includes_previews": None,
            "reported_friday_gross_usd": daily.actual_usd,
            "thursday_preview_actual_usd": thursday_nowcast.get("thursday_preview_actual_usd"),
            "friday_calendar_actual_usd": None,
            "actual_friday_usd": daily.actual_usd,
            "actual_opening_weekend_usd": weekend.actual_usd,
        })
    return rows


def build_thursday_preview_shadow_rows(
    *,
    movie: MovieOpening,
    thursday_preview_nowcasts: pd.DataFrame,
    artifacts: ModelArtifacts,
    run_id: str,
) -> list[dict[str, object]]:
    if thursday_preview_nowcasts.empty:
        return []
    try:
        baseline = select_daily_baseline_row(artifacts.daily_baseline, movie)
    except (KeyError, ValueError):
        baseline = pd.Series(dtype=object)
    rows = []
    movie_rows = thursday_preview_nowcasts.loc[
        pd.to_numeric(thursday_preview_nowcasts.get("movie_id"), errors="coerce").eq(movie.movie_id)
    ].copy()
    for nowcast in movie_rows.itertuples(index=False):
        origin = str(getattr(nowcast, "forecast_origin", "") or "")
        if not origin:
            continue
        shadow_ow = finite(getattr(nowcast, "shadow_ow_prior_usd", math.nan))
        predicted_preview = finite(getattr(nowcast, "predicted_thursday_previews_usd", math.nan))
        if not positive(shadow_ow) and not positive(predicted_preview):
            continue
        rows.append(
            {
                "release_run_id": movie.release_run_id,
                "movie_id": movie.movie_id,
                "title": movie.title,
                "opening_weekend_start": movie.opening_weekend_start,
                "origin_key": f"THU_{origin}",
                "forecast_origin": origin,
                "forecast_origin_utc": getattr(nowcast, "forecast_origin_utc", None),
                "as_of_utc": getattr(nowcast, "as_of_utc", None),
                "model_version": artifacts.model_version,
                "run_id": run_id,
                "sample_key": getattr(nowcast, "sample_key", None),
                "multiplicative_daily_usd": getattr(nowcast, "predicted_thursday_previews_multiplicative_usd", None),
                "additive_daily_usd": getattr(nowcast, "predicted_thursday_previews_additive_usd", None),
                "hybrid_daily_usd": getattr(nowcast, "predicted_thursday_previews_hybrid_usd", None),
                "production_daily_usd": None,
                "baseline_daily_usd": None,
                "provisional_blend_daily_usd": None,
                "daily_lo80_usd": None,
                "daily_hi80_usd": None,
                "daily_lo95_usd": None,
                "daily_hi95_usd": None,
                "opening_weekend_usd": shadow_ow if positive(shadow_ow) else None,
                "opening_weekend_lo80_usd": None,
                "opening_weekend_hi80_usd": None,
                "opening_weekend_lo95_usd": None,
                "opening_weekend_hi95_usd": None,
                "coverage": getattr(nowcast, "amc_coverage", None),
                "snapshot_count": getattr(nowcast, "amc_snapshot_count", None),
                "staleness_p50_minutes": getattr(nowcast, "amc_staleness_p50_minutes", None),
                "feature_quality_bucket": getattr(nowcast, "feature_quality_bucket", None),
                "thursday_preview_treatment": THURSDAY_PREVIEW_TREATMENT,
                "amc_candidate_available": positive(predicted_preview),
                "amc_candidate_unavailable_reason": None if positive(predicted_preview) else "no_thursday_preview_nowcast",
                "thursday_amc_seats_collected": bool(getattr(nowcast, "thursday_amc_seats_collected", False)),
                "predicted_thursday_previews_usd": predicted_preview if positive(predicted_preview) else None,
                "thursday_amc_observed_preview_seats": getattr(nowcast, "amc_observed_preview_seats", None),
                "predicted_thursday_previews_lo80_usd": getattr(nowcast, "predicted_thursday_previews_lo80_usd", None),
                "predicted_thursday_previews_hi80_usd": getattr(nowcast, "predicted_thursday_previews_hi80_usd", None),
                "predicted_thursday_previews_lo95_usd": getattr(nowcast, "predicted_thursday_previews_lo95_usd", None),
                "predicted_thursday_previews_hi95_usd": getattr(nowcast, "predicted_thursday_previews_hi95_usd", None),
                "preview_residual_pool_scope": getattr(nowcast, "preview_residual_pool_scope", None),
                "thursday_amc_shadow_ow_prior_usd": shadow_ow if positive(shadow_ow) else None,
                "thursday_amc_shadow_ow_prior_source": getattr(nowcast, "shadow_ow_prior_source", None),
                "thursday_amc_shadow_ow_lo80_usd": getattr(nowcast, "shadow_ow_lo80_usd", None),
                "thursday_amc_shadow_ow_hi80_usd": getattr(nowcast, "shadow_ow_hi80_usd", None),
                "thursday_amc_shadow_ow_lo95_usd": getattr(nowcast, "shadow_ow_lo95_usd", None),
                "thursday_amc_shadow_ow_hi95_usd": getattr(nowcast, "shadow_ow_hi95_usd", None),
                "thursday_selected_candidate": getattr(nowcast, "selected_candidate", None),
                "thursday_target_classification": getattr(nowcast, "target_classification", None),
                "thursday_training_cutoff": getattr(nowcast, "training_cutoff", None),
                "friday_only_amc_observed_seats": None,
                "evaluation_target_includes_previews": None,
                "reported_friday_gross_usd": None,
                "thursday_preview_actual_usd": getattr(nowcast, "thursday_preview_actual_usd", None),
                "friday_calendar_actual_usd": None,
                "actual_friday_usd": finite(baseline.get("actual_fri_usd")) if not baseline.empty else None,
                "actual_opening_weekend_usd": finite(baseline.get("actual_ow_usd")) if not baseline.empty else None,
            }
        )
    return rows


def write_shadow_rows(conn: Any, rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    insert_columns = existing_shadow_columns(conn)
    values = [tuple(row.get(column) for column in insert_columns) for row in rows]
    if not values:
        return 0
    placeholders = ", ".join("%s" for _ in insert_columns)
    columns = ", ".join(insert_columns)
    conflict_columns = {"release_run_id", "origin_key", "model_version"}
    update_columns = [column for column in insert_columns if column not in conflict_columns]
    updates = ", ".join(f"{column} = EXCLUDED.{column}" for column in update_columns)
    cursor = conn.executemany(
        f"""
        INSERT INTO {SHADOW_TABLE} ({columns}) VALUES ({placeholders})
        ON CONFLICT (release_run_id, origin_key, model_version) DO UPDATE SET {updates}
        WHERE LEFT(EXCLUDED.origin_key, 4) = 'THU_'
        """,
        values,
    )
    return int(getattr(cursor, "rowcount", 0) or 0)


def existing_shadow_columns(conn: Any) -> list[str]:
    try:
        cursor = conn.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            """,
            ("analytics", "amc_opening_shadow_forecasts"),
        )
        existing = {str(row[0]) for row in cursor.fetchall()}
    except Exception:
        return SHADOW_COLUMNS
    columns = [column for column in SHADOW_COLUMNS if column in existing]
    return columns or SHADOW_COLUMNS


def finite(value: object) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if np.isfinite(out) else math.nan


def positive(value: object) -> bool:
    return math.isfinite(finite(value)) and finite(value) > 0


def select_thursday_preview_nowcast(
    nowcasts: pd.DataFrame | None,
    *,
    movie_id: int,
    forecast_origin: object,
) -> pd.Series:
    if nowcasts is None or nowcasts.empty:
        return pd.Series(dtype=object)
    frame = nowcasts.copy()
    rows = frame.loc[
        pd.to_numeric(frame.get("movie_id"), errors="coerce").eq(movie_id)
        & frame.get("forecast_origin", pd.Series(index=frame.index, dtype=object)).astype(str).eq(str(forecast_origin))
    ]
    return rows.iloc[-1] if not rows.empty else pd.Series(dtype=object)
