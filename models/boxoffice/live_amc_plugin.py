"""Build live AMC plug-in daily nowcasts from collected seat snapshots."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .constants import (
    LIVE_FRIDAY_REGIME,
    LIVE_ORIGINS,
    LIVE_REGIMES,
    LIVE_SATURDAY_REGIME,
    LIVE_SUNDAY_REGIME,
    PLUGIN_TARGET_DAY,
    REGIME_DAY_OFFSET,
)
from .amc_features import (
    aggregate_as_of_features,
    aggregate_final_eod_features,
    build_origin_grid,
    fetch_actuals,
    fetch_schedule,
    fetch_snapshots,
    latest_snapshots_as_of,
    latest_snapshots_final,
)
from .origins import build_forecast_origins
from .schema import MovieOpening


MIN_TRAIN_ROWS = 3
DEFAULT_SIGMA_LOG_DAILY = 0.45
MIN_SIGMA_LOG_DAILY = 0.25
DEFAULT_INTERVAL_MIN_BUCKET_N = 5
DEFAULT_STANDALONE_80_MIN_UNIQUE = 40
DEFAULT_STANDALONE_95_MIN_UNIQUE = 100
PARTIAL_POOL_PRIOR_UNIQUE = 40

DAY_TO_REGIME = {
    "Friday": LIVE_FRIDAY_REGIME,
    "Saturday": LIVE_SATURDAY_REGIME,
    "Sunday": LIVE_SUNDAY_REGIME,
}

AMC_INTERVAL_SCOPES = (
    ("sample_key", "regime", "forecast_origin", "target_day", "feature_quality_bucket"),
    ("sample_key", "regime", "forecast_origin", "target_day"),
    ("sample_key", "regime", "target_day"),
    ("sample_key",),
    ("regime", "forecast_origin", "target_day", "feature_quality_bucket"),
    ("regime", "forecast_origin", "target_day"),
    ("regime", "target_day"),
    (),
)


@dataclass(frozen=True)
class LiveAmcPluginConfig:
    origin_timezone: str = "America/New_York"
    min_train_rows: int = MIN_TRAIN_ROWS
    default_sigma_log_daily: float = DEFAULT_SIGMA_LOG_DAILY
    min_sigma_log_daily: float = MIN_SIGMA_LOG_DAILY


def active_sample_key(conn: Any) -> str | None:
    row = conn.execute(
        """
        SELECT sample_key
        FROM amc_theatre_sample_sets
        WHERE status = 'active'
        ORDER BY (sample_key = 'top_hybrid_30') DESC, sample_key
        LIMIT 1
        """
    ).fetchone()
    return str(row[0]) if row and row[0] else None


def build_live_amc_plugin_nowcasts(
    conn: object,
    *,
    movies: Iterable[MovieOpening],
    as_of_utc: datetime,
    config: LiveAmcPluginConfig = LiveAmcPluginConfig(),
) -> pd.DataFrame:
    """Return rows compatible with ``ModelArtifacts.live_plugin_nowcasts``.

    The model is intentionally simple and online-safe:
    observed seats as of the forecast origin -> historical pace-adjusted final
    AMC seats -> historical AMC-final-seat to national-gross bridge.
    """

    movie_list = list(movies)
    if not movie_list:
        return empty_plugin_frame()

    schedule = fetch_schedule(conn)
    snapshots = fetch_snapshots(conn)
    actuals = fetch_actuals(conn)
    if schedule.empty or snapshots.empty:
        return empty_plugin_frame()

    for frame in [schedule, snapshots, actuals]:
        if "exhibition_date" in frame.columns:
            frame["exhibition_date"] = pd.to_datetime(frame["exhibition_date"], errors="coerce").dt.date

    sample_key = active_sample_key(conn) or "unknown"
    target_origins = sorted(
        {
            origin.forecast_origin
            for movie in movie_list
            for origin in build_forecast_origins(
                movie,
                mode="live",
                as_of_utc=as_of_utc,
                origin_timezone=config.origin_timezone,
                include_pre_release=False,
            )
            if origin.regime in LIVE_REGIMES and origin.forecast_origin
        }
    )
    if not target_origins:
        return empty_plugin_frame()

    grid = build_origin_grid(schedule, target_origins, config.origin_timezone)
    asof = latest_snapshots_as_of(snapshots, grid)
    features = aggregate_as_of_features(asof)
    final_features = aggregate_final_eod_features(latest_snapshots_final(snapshots))
    panel = grid.merge(schedule, on=["movie_id", "amc_movie_id", "exhibition_date"], how="left")
    panel = panel.merge(features, on=["movie_id", "amc_movie_id", "exhibition_date", "forecast_origin", "forecast_origin_utc"], how="left")
    panel = panel.merge(final_features, on=["movie_id", "amc_movie_id", "exhibition_date"], how="left")
    panel = panel.merge(actuals, on=["movie_id", "exhibition_date"], how="left")
    panel = finalize_live_panel(panel)
    if panel.empty:
        return empty_plugin_frame()

    rows: list[dict[str, object]] = []
    for movie in movie_list:
        for origin in build_forecast_origins(
            movie,
            mode="live",
            as_of_utc=as_of_utc,
            origin_timezone=config.origin_timezone,
            include_pre_release=False,
        ):
            if origin.regime not in LIVE_REGIMES or not origin.forecast_origin:
                continue
            target_day = PLUGIN_TARGET_DAY[origin.regime]
            exhibition_date = movie.opening_weekend_start + pd.Timedelta(days=origin.origin_day or 0).to_pytimedelta()
            target = select_target_panel_row(
                panel,
                movie_id=movie.movie_id,
                exhibition_date=exhibition_date,
                forecast_origin=origin.forecast_origin,
            )
            if target is None:
                continue
            plugin = nowcast_from_panel_row(panel, target, config=config)
            if plugin is None:
                continue
            rows.append(
                {
                    "movie_id": movie.movie_id,
                    "regime": origin.regime,
                    "forecast_origin": origin.forecast_origin,
                    "target_day": target_day,
                    "sample_key": sample_key,
                    "pred_daily_gross_usd": plugin["pred_daily_gross_usd"],
                    "pred_daily_gross_multiplicative_usd": plugin["pred_daily_gross_multiplicative_usd"],
                    "pred_daily_gross_additive_usd": plugin["pred_daily_gross_additive_usd"],
                    "pred_daily_gross_hybrid_usd": plugin["pred_daily_gross_hybrid_usd"],
                    "pred_final_eod_seats_multiplicative": plugin["pred_final_eod_seats_multiplicative"],
                    "pred_final_eod_seats_additive": plugin["pred_final_eod_seats_additive"],
                    "pred_final_eod_seats_hybrid": plugin["pred_final_eod_seats_hybrid"],
                    "sigma_log_daily": plugin["sigma_log_daily"],
                    "source": "AMC_live_db",
                    "model": plugin["model"],
                    "feature_quality_bucket": plugin["feature_quality_bucket"],
                    "amc_coverage": plugin["amc_coverage"],
                    "amc_snapshot_count": plugin["amc_snapshot_count"],
                    "amc_lateness_p50_minutes": plugin["amc_lateness_p50_minutes"],
                    "amc_staleness_p50_minutes": plugin["amc_staleness_p50_minutes"],
                    "amc_observed_seats": plugin["amc_observed_seats"],
                }
            )
    if not rows:
        return empty_plugin_frame()
    return pd.DataFrame(rows).drop_duplicates(["movie_id", "regime", "forecast_origin", "target_day"], keep="last")


def build_amc_interval_policy(
    conn: Any,
    *,
    config: LiveAmcPluginConfig = LiveAmcPluginConfig(),
    min_bucket_n: int = DEFAULT_INTERVAL_MIN_BUCKET_N,
    sample_key: str | None = None,
) -> dict[str, object]:
    """Replay historical AMC nowcasts and freeze empirical log-error buckets."""

    resolved_sample_key = sample_key or active_sample_key(conn) or "unknown"
    schedule = fetch_schedule(conn)
    snapshots = fetch_snapshots(conn)
    actuals = fetch_actuals(conn)
    if schedule.empty or snapshots.empty or actuals.empty:
        return empty_amc_interval_policy(sample_key=resolved_sample_key, min_bucket_n=min_bucket_n)

    for frame in [schedule, snapshots, actuals]:
        if "exhibition_date" in frame.columns:
            frame["exhibition_date"] = pd.to_datetime(frame["exhibition_date"], errors="coerce").dt.date

    grid = build_origin_grid(schedule, LIVE_ORIGINS, config.origin_timezone)
    asof = latest_snapshots_as_of(snapshots, grid)
    features = aggregate_as_of_features(asof)
    final_features = aggregate_final_eod_features(latest_snapshots_final(snapshots))
    panel = grid.merge(schedule, on=["movie_id", "amc_movie_id", "exhibition_date"], how="left")
    panel = panel.merge(features, on=["movie_id", "amc_movie_id", "exhibition_date", "forecast_origin", "forecast_origin_utc"], how="left")
    panel = panel.merge(final_features, on=["movie_id", "amc_movie_id", "exhibition_date"], how="left")
    panel = panel.merge(actuals, on=["movie_id", "exhibition_date"], how="left")
    panel = finalize_live_panel(panel)
    if panel.empty:
        return empty_amc_interval_policy(sample_key=resolved_sample_key, min_bucket_n=min_bucket_n)

    rows: list[dict[str, object]] = []
    for _, target in panel.iterrows():
        day = str(target.get("day_of_week") or "")
        regime = DAY_TO_REGIME.get(day)
        if not regime:
            continue
        actual = positive_float(target.get("actual_gross_usd"))
        if not math.isfinite(actual):
            continue
        plugin = nowcast_from_panel_row(panel, target, config=config)
        if plugin is None:
            continue
        pred = positive_float(plugin.get("pred_daily_gross_usd"))
        if not math.isfinite(pred):
            continue
        rows.append(
            {
                "sample_key": resolved_sample_key,
                "regime": regime,
                "forecast_origin": str(target.get("forecast_origin")),
                "target_day": day,
                "feature_quality_bucket": str(plugin.get("feature_quality_bucket") or "unknown"),
                "movie_id": target.get("movie_id"),
                "amc_movie_id": target.get("amc_movie_id"),
                "exhibition_date": target.get("exhibition_date"),
                "pred_daily_gross_usd": pred,
                "actual_gross_usd": actual,
                "log_residual": math.log(actual / pred),
            }
        )
    residuals = pd.DataFrame(rows)
    return build_amc_interval_policy_from_residuals(
        residuals,
        sample_key=resolved_sample_key,
        min_bucket_n=min_bucket_n,
    )


def build_amc_interval_policy_from_residuals(
    residuals: pd.DataFrame,
    *,
    sample_key: str,
    min_bucket_n: int = DEFAULT_INTERVAL_MIN_BUCKET_N,
) -> dict[str, object]:
    policy = empty_amc_interval_policy(sample_key=sample_key, min_bucket_n=min_bucket_n)
    if residuals.empty or "log_residual" not in residuals.columns:
        return policy

    work = residuals.copy()
    for column in ["sample_key", "regime", "forecast_origin", "target_day", "feature_quality_bucket"]:
        if column not in work.columns:
            work[column] = "*" if column != "sample_key" else sample_key
        work[column] = work[column].fillna("*").astype(str)
    work["sample_key"] = work["sample_key"].replace("", sample_key)
    work["log_residual"] = pd.to_numeric(work["log_residual"], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_residual"])
    if work.empty:
        return policy

    cells: dict[str, object] = {}
    global_payload = stabilize_underpowered_tail(residual_quantile_payload(work["log_residual"], frame=work))
    for scope in AMC_INTERVAL_SCOPES:
        if scope:
            grouped = work.groupby(list(scope), dropna=False)
            for key_values, group in grouped:
                if not isinstance(key_values, tuple):
                    key_values = (key_values,)
                if group_unique_movie_days(group) < min_bucket_n:
                    continue
                key = amc_interval_cell_key(dict(zip(scope, key_values, strict=True)))
                raw = residual_quantile_payload(group["log_residual"], frame=group)
                cells[key] = partially_pool_residual_payload(raw, global_payload)
        elif group_unique_movie_days(work) >= 2:
            cells[amc_interval_cell_key({})] = global_payload

    policy["cells"] = cells
    policy["n_residuals"] = int(len(work))
    policy["n_residual_rows"] = int(len(work))
    policy["n_unique_movie_days"] = group_unique_movie_days(work)
    policy["n_unique_release_weekends"] = group_unique_release_weekends(work)
    policy["tail_policy"] = {
        "method": "partial_pooling_by_unique_movie_day",
        "standalone_80_min_unique_movie_days": DEFAULT_STANDALONE_80_MIN_UNIQUE,
        "standalone_95_min_unique_movie_days": DEFAULT_STANDALONE_95_MIN_UNIQUE,
        "prior_unique_movie_days": PARTIAL_POOL_PRIOR_UNIQUE,
        "pooled_origin_weighting": "equal_total_weight_per_movie_day",
    }
    stability, tail_cases = build_amc_interval_diagnostics(work)
    policy["diagnostics"] = {
        "amc_interval_cell_stability": stability,
        "amc_tail_case_audit": tail_cases,
    }
    return policy


def empty_amc_interval_policy(*, sample_key: str, min_bucket_n: int) -> dict[str, object]:
    return {
        "method": "empirical_amc_live_log_residual_quantile",
        "sample_key": sample_key,
        "min_bucket_n": int(min_bucket_n),
        "fallback_order": [
            "|".join(scope) if scope else "global"
            for scope in AMC_INTERVAL_SCOPES
        ],
        "cells": {},
        "n_residuals": 0,
        "n_residual_rows": 0,
        "n_unique_movie_days": 0,
        "n_unique_release_weekends": 0,
    }


def amc_interval_cell_key(values: dict[str, object]) -> str:
    return "|".join(
        f"{column}={values.get(column, '*') or '*'}"
        for column in ["sample_key", "regime", "forecast_origin", "target_day", "feature_quality_bucket"]
    )


def residual_quantile_payload(residuals: pd.Series, *, frame: pd.DataFrame | None = None) -> dict[str, object]:
    clean = pd.to_numeric(residuals, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    clean_frame = frame.loc[clean.index].copy() if frame is not None else pd.DataFrame(index=clean.index)
    weights = equal_movie_day_weights(clean_frame)
    quantiles = weighted_quantiles(clean.to_numpy(dtype="float64"), weights, [0.025, 0.10, 0.50, 0.90, 0.975])
    median = float(quantiles[2])
    centered = clean.to_numpy(dtype="float64") - median
    sigma = weighted_robust_sigma(centered, weights)
    return {
        "n": int(clean.size),
        "n_residual_rows": int(clean.size),
        "n_unique_movie_days": group_unique_movie_days(clean_frame),
        "n_unique_release_weekends": group_unique_release_weekends(clean_frame),
        "method": "empirical_amc_live_log_residual_quantile",
        "sigma_log": sigma,
        "lo95_log": float(quantiles[0]),
        "lo80_log": float(quantiles[1]),
        "median_log": median,
        "hi80_log": float(quantiles[3]),
        "hi95_log": float(quantiles[4]),
        "residual_samples_log": [float(value) for value in clean.to_numpy()],
        "residual_sample_weights": [float(value) for value in weights],
    }


def partially_pool_residual_payload(cell: dict[str, object], parent: dict[str, object]) -> dict[str, object]:
    """Shrink cell bias/scale and borrow stable standardized tails from the AMC pool."""

    n_unique = int(cell.get("n_unique_movie_days") or 0)
    if n_unique >= DEFAULT_STANDALONE_95_MIN_UNIQUE:
        return cell
    weight = n_unique / (n_unique + PARTIAL_POOL_PRIOR_UNIQUE)
    cell_center = float(cell.get("median_log") or 0.0)
    parent_center = float(parent.get("median_log") or 0.0)
    cell_scale = max(float(cell.get("sigma_log") or 0.0), 1e-6)
    parent_scale = max(float(parent.get("sigma_log") or 0.0), 1e-6)
    center = weight * cell_center + (1.0 - weight) * parent_center
    scale = weight * cell_scale + (1.0 - weight) * parent_scale
    parent_samples = np.asarray(parent.get("residual_samples_log") or [], dtype="float64")
    parent_weights = np.asarray(parent.get("residual_sample_weights") or [], dtype="float64")
    standardized = (parent_samples - parent_center) / parent_scale
    pooled_samples = center + scale * standardized
    qs = weighted_quantiles(pooled_samples, parent_weights, [0.025, 0.10, 0.50, 0.90, 0.975])
    return {
        **cell,
        "method": "partially_pooled_amc_live_log_residual_quantile",
        "pooling_weight": float(weight),
        "pooling_parent": "global_amc_movie_day_weighted",
        "sigma_log": float(scale),
        "lo95_log": float(qs[0]),
        "lo80_log": float(qs[1]),
        "median_log": float(qs[2]),
        "hi80_log": float(qs[3]),
        "hi95_log": float(qs[4]),
        "residual_samples_log": [float(value) for value in pooled_samples],
        "residual_sample_weights": [float(value) for value in parent_weights],
    }


def stabilize_underpowered_tail(payload: dict[str, object]) -> dict[str, object]:
    """Prevent unsupported empirical order statistics from controlling production tails."""

    n_unique = int(payload.get("n_unique_movie_days") or 0)
    if n_unique >= DEFAULT_STANDALONE_95_MIN_UNIQUE:
        return payload
    center = float(payload.get("median_log") or 0.0)
    sigma = max(float(payload.get("sigma_log") or 0.0), MIN_SIGMA_LOG_DAILY)
    lo95 = center - 1.95996 * sigma
    hi95 = center + 1.95996 * sigma
    if n_unique < DEFAULT_STANDALONE_80_MIN_UNIQUE:
        lo80 = center - 1.28155 * sigma
        hi80 = center + 1.28155 * sigma
    else:
        lo80 = float(payload["lo80_log"])
        hi80 = float(payload["hi80_log"])
    samples = np.asarray(payload.get("residual_samples_log") or [], dtype="float64")
    clipped = np.clip(samples, lo95, hi95)
    return {
        **payload,
        "method": "robust_tail_stabilized_amc_live_log_residual_quantile",
        "tail_stabilized": True,
        "tail_stabilization_reason": "fewer_than_100_unique_movie_days",
        "lo95_log": float(min(lo95, lo80)),
        "lo80_log": float(lo80),
        "hi80_log": float(hi80),
        "hi95_log": float(max(hi95, hi80)),
        "residual_samples_log": [float(value) for value in clipped],
    }


def movie_day_keys(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="object")
    movie = frame.get("movie_id", frame.get("amc_movie_id", pd.Series(frame.index, index=frame.index))).astype(str)
    day = frame.get("exhibition_date", pd.Series("unknown", index=frame.index)).astype(str)
    return movie + "|" + day


def group_unique_movie_days(frame: pd.DataFrame) -> int:
    keys = movie_day_keys(frame)
    return int(keys.nunique()) if not keys.empty else int(len(frame))


def group_unique_release_weekends(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    if "release_run_id" in frame.columns:
        return int(frame["release_run_id"].dropna().astype(str).nunique())
    if "movie_id" in frame.columns:
        return int(frame["movie_id"].dropna().astype(str).nunique())
    return group_unique_movie_days(frame)


def equal_movie_day_weights(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.ones(len(frame), dtype="float64")
    keys = movie_day_keys(frame)
    counts = keys.map(keys.value_counts()).to_numpy(dtype="float64")
    weights = 1.0 / counts
    return weights / weights.sum()


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, probabilities: list[float]) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    weights = np.asarray(weights, dtype="float64")
    if values.size == 0:
        return np.full(len(probabilities), np.nan)
    if weights.size != values.size or not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(values.size, dtype="float64")
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = (np.cumsum(sorted_weights) - 0.5 * sorted_weights) / sorted_weights.sum()
    return np.interp(probabilities, cumulative, sorted_values, left=sorted_values[0], right=sorted_values[-1])


def weighted_robust_sigma(centered: np.ndarray, weights: np.ndarray) -> float:
    mad = float(weighted_quantiles(np.abs(centered), weights, [0.5])[0])
    if math.isfinite(mad) and mad > 0:
        return 1.4826 * mad
    variance = float(np.average(centered**2, weights=weights)) if centered.size else 0.0
    return math.sqrt(max(variance, 0.0))


def build_amc_interval_diagnostics(work: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    stability: list[dict[str, object]] = []
    detailed_scope = AMC_INTERVAL_SCOPES[0]
    for key_values, group in work.groupby(list(detailed_scope), dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        cell_values = dict(zip(detailed_scope, key_values, strict=True))
        full = residual_quantile_payload(group["log_residual"], frame=group)
        movie_keys = movie_day_keys(group)
        full_hi80 = math.exp(float(full["hi80_log"]))
        full_hi95 = math.exp(float(full["hi95_log"]))
        for omitted_key in sorted(movie_keys.unique()):
            retained = group.loc[~movie_keys.eq(omitted_key)]
            if retained.empty:
                continue
            loo = residual_quantile_payload(retained["log_residual"], frame=retained)
            omitted = group.loc[movie_keys.eq(omitted_key)].iloc[0]
            stability.append(
                {
                    **cell_values,
                    "n_residual_rows": int(full["n_residual_rows"]),
                    "n_unique_movie_days": int(full["n_unique_movie_days"]),
                    "n_unique_release_weekends": int(full["n_unique_release_weekends"]),
                    "full_hi80_multiplier": full_hi80,
                    "full_hi95_multiplier": full_hi95,
                    "omitted_movie_day": omitted_key,
                    "omitted_movie_id": omitted.get("movie_id"),
                    "omitted_exhibition_date": omitted.get("exhibition_date"),
                    "loo_hi80_multiplier": math.exp(float(loo["hi80_log"])),
                    "loo_hi95_multiplier": math.exp(float(loo["hi95_log"])),
                }
            )

    identity_columns = [
        "sample_key", "movie_id", "amc_movie_id", "exhibition_date", "regime",
        "forecast_origin", "target_day", "feature_quality_bucket",
        "pred_daily_gross_usd", "actual_gross_usd", "log_residual",
    ]
    available = [column for column in identity_columns if column in work.columns]
    ranked = work.sort_values("log_residual")
    tails = pd.concat([ranked.head(20), ranked.tail(20)]).drop_duplicates()
    tail_cases = tails.loc[:, available].copy()
    if not tail_cases.empty:
        tail_cases["actual_to_forecast_multiplier"] = np.exp(tail_cases["log_residual"])
        tail_cases["tail_side"] = np.where(tail_cases["log_residual"].lt(0), "negative", "positive")
    return stability, tail_cases.to_dict(orient="records")


def finalize_live_panel(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    numeric_defaults = {
        "s_obs": 0.0,
        "c_obs": 0.0,
        "n_snapshots": 0,
        "n_theatres_observed": 0,
        "s_final_eod": 0.0,
        "c_final_eod": 0.0,
        "n_final_snapshots": 0,
        "n_final_theatres": 0,
        "rsc_success_rate": 0.0,
        "rescue_share": 0.0,
        "share_snapshots_late_gt_30m": 0.0,
        "share_capacity_observed_late_gt_30m": 0.0,
    }
    for column, default in numeric_defaults.items():
        if column in panel.columns:
            panel[column] = pd.to_numeric(panel[column], errors="coerce").fillna(default)
    if "c_scheduled_known" not in panel.columns:
        panel["c_scheduled_known"] = np.nan
    panel["coverage"] = panel["c_obs"] / pd.to_numeric(panel["c_scheduled_known"], errors="coerce").replace(0, np.nan)
    panel["day_of_week"] = pd.to_datetime(panel["exhibition_date"], errors="coerce").dt.day_name()
    delay = pd.to_numeric(panel.get("delay_p90_minutes"), errors="coerce")
    rsc = pd.to_numeric(panel.get("rsc_success_rate"), errors="coerce").fillna(0.0)
    coverage = pd.to_numeric(panel["coverage"], errors="coerce")
    panel["collection_quality_bucket"] = np.select(
        [
            coverage.ge(0.80) & delay.le(15) & rsc.ge(0.80),
            coverage.ge(0.50) & delay.le(45),
        ],
        ["high", "medium"],
        default="low",
    )
    return panel.replace([np.inf, -np.inf], np.nan).sort_values(
        ["exhibition_date", "movie_id", "forecast_origin"]
    ).reset_index(drop=True)


def select_target_panel_row(
    panel: pd.DataFrame,
    *,
    movie_id: int,
    exhibition_date: object,
    forecast_origin: str,
) -> pd.Series | None:
    frame = panel.loc[
        panel["movie_id"].eq(movie_id)
        & panel["exhibition_date"].eq(exhibition_date)
        & panel["forecast_origin"].astype(str).eq(forecast_origin)
    ].copy()
    if frame.empty:
        return None
    for column in ["s_obs", "n_snapshots", "c_obs", "c_scheduled_known", "n_scheduled_showtimes"]:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame = frame.sort_values(
        ["s_obs", "n_snapshots", "c_obs", "c_scheduled_known", "n_scheduled_showtimes"],
        ascending=[False, False, False, False, False],
    )
    return frame.iloc[0]


def nowcast_from_panel_row(
    panel: pd.DataFrame,
    target: pd.Series,
    *,
    config: LiveAmcPluginConfig,
) -> dict[str, object] | None:
    s_obs = positive_float(target.get("s_obs"))
    if not math.isfinite(s_obs):
        return None
    forecast_origin = str(target.get("forecast_origin"))
    day_of_week = str(target.get("day_of_week"))
    current_date = pd.Timestamp(target.get("exhibition_date")).date()
    train = panel.loc[pd.to_datetime(panel["exhibition_date"], errors="coerce").dt.date < current_date].copy()
    if train.empty:
        return None

    pace_train = train.loc[
        train["forecast_origin"].astype(str).eq(forecast_origin)
        & pd.to_numeric(train["s_obs"], errors="coerce").gt(0)
        & pd.to_numeric(train["s_final_eod"], errors="coerce").gt(0)
    ].copy()
    if len(pace_train) < config.min_train_rows:
        return None
    pace_train["raw_pace"] = pd.to_numeric(pace_train["s_obs"], errors="coerce") / pd.to_numeric(
        pace_train["s_final_eod"], errors="coerce"
    )
    pace_pool = pace_train.loc[pace_train["day_of_week"].astype(str).eq(day_of_week), "raw_pace"].dropna()
    pace_source = "day_origin_pace"
    if len(pace_pool) < config.min_train_rows:
        pace_pool = pace_train["raw_pace"].dropna()
        pace_source = "origin_pooled_pace"
    pace = float(pace_pool.median()) if len(pace_pool) else math.nan
    if not math.isfinite(pace) or pace <= 0:
        return None
    pace = float(np.clip(pace, 0.01, 1.0))
    pred_final_eod_seats = s_obs / pace

    bridge_train = train.loc[
        pd.to_numeric(train["actual_gross_usd"], errors="coerce").gt(0)
        & pd.to_numeric(train["s_final_eod"], errors="coerce").gt(0)
    ].copy()
    if len(bridge_train) < config.min_train_rows:
        return None
    bridge_train["bridge_residual"] = np.log(pd.to_numeric(bridge_train["actual_gross_usd"], errors="coerce")) - np.log1p(
        pd.to_numeric(bridge_train["s_final_eod"], errors="coerce")
    )
    residual_pool = bridge_train.loc[bridge_train["day_of_week"].astype(str).eq(day_of_week), "bridge_residual"].dropna()
    bridge_source = "pooled_same_day_bridge"
    if len(residual_pool) < config.min_train_rows:
        residual_pool = bridge_train["bridge_residual"].dropna()
        bridge_source = "pooled_all_days_bridge"
    if len(residual_pool) < config.min_train_rows:
        return None
    bridge_residual = float(residual_pool.median())
    pred_final_eod_seats_additive = additive_final_seat_prediction(pace_train, target)
    if not math.isfinite(pred_final_eod_seats_additive):
        pred_final_eod_seats_additive = pred_final_eod_seats
    pred_final_eod_seats_hybrid = 0.5 * (pred_final_eod_seats + pred_final_eod_seats_additive)
    pred_daily_gross = float(math.exp(bridge_residual + math.log1p(pred_final_eod_seats)))
    pred_daily_gross_additive = float(math.exp(bridge_residual + math.log1p(pred_final_eod_seats_additive)))
    pred_daily_gross_hybrid = float(math.exp(bridge_residual + math.log1p(pred_final_eod_seats_hybrid)))
    if not math.isfinite(pred_daily_gross) or pred_daily_gross <= 0:
        return None

    sigma = robust_sigma(residual_pool, default=config.default_sigma_log_daily)
    sigma = max(config.min_sigma_log_daily, sigma)
    coverage = finite_float(target.get("coverage"))
    snapshots = int(finite_float(target.get("n_snapshots")) or 0)
    return {
        "pred_daily_gross_usd": pred_daily_gross,
        "pred_daily_gross_multiplicative_usd": pred_daily_gross,
        "pred_daily_gross_additive_usd": pred_daily_gross_additive,
        "pred_daily_gross_hybrid_usd": pred_daily_gross_hybrid,
        "pred_final_eod_seats_multiplicative": pred_final_eod_seats,
        "pred_final_eod_seats_additive": pred_final_eod_seats_additive,
        "pred_final_eod_seats_hybrid": pred_final_eod_seats_hybrid,
        "sigma_log_daily": sigma,
        "model": f"AMC_live_pace_bridge_v1:{pace_source}+{bridge_source}",
        "feature_quality_bucket": str(target.get("collection_quality_bucket") or "low"),
        "amc_coverage": coverage if math.isfinite(coverage) else None,
        "amc_snapshot_count": snapshots,
        "amc_lateness_p50_minutes": finite_float(target.get("delay_p50_minutes")),
        "amc_staleness_p50_minutes": finite_float(target.get("staleness_p50_minutes")),
        "amc_observed_seats": s_obs,
    }


def additive_final_seat_prediction(train: pd.DataFrame, target: pd.Series) -> float:
    required = {"s_obs", "s_final_eod", "c_obs", "c_scheduled_known"}
    if not required.issubset(train.columns) or not required.issubset(target.index):
        return math.nan
    work = train.copy()
    remaining_capacity = (
        pd.to_numeric(work["c_scheduled_known"], errors="coerce")
        - pd.to_numeric(work["c_obs"], errors="coerce")
    ).clip(lower=0)
    remaining_sales = (
        pd.to_numeric(work["s_final_eod"], errors="coerce")
        - pd.to_numeric(work["s_obs"], errors="coerce")
    ).clip(lower=0)
    valid = remaining_capacity.gt(0) & remaining_sales.notna()
    if valid.sum() < MIN_TRAIN_ROWS or remaining_capacity.loc[valid].sum() <= 0:
        return math.nan
    occupancy = float(remaining_sales.loc[valid].sum() / remaining_capacity.loc[valid].sum())
    target_observed = positive_float(target.get("s_obs"))
    target_remaining_capacity = max(
        finite_float(target.get("c_scheduled_known")) - finite_float(target.get("c_obs")),
        0.0,
    )
    prediction = target_observed + target_remaining_capacity * float(np.clip(occupancy, 0.0, 1.0))
    return prediction if math.isfinite(prediction) and prediction > 0 else math.nan


def robust_sigma(values: pd.Series, *, default: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return default
    median = float(clean.median())
    mad = float((clean - median).abs().median())
    sigma = 1.4826 * mad if math.isfinite(mad) and mad > 0 else float(clean.std(ddof=1))
    return sigma if math.isfinite(sigma) and sigma > 0 else default


def positive_float(value: object) -> float:
    out = finite_float(value)
    return out if math.isfinite(out) and out > 0 else math.nan


def finite_float(value: object) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if np.isfinite(out) else math.nan


def empty_plugin_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "movie_id",
            "regime",
            "forecast_origin",
            "target_day",
            "sample_key",
            "pred_daily_gross_usd",
            "pred_daily_gross_multiplicative_usd",
            "pred_daily_gross_additive_usd",
            "pred_daily_gross_hybrid_usd",
            "pred_final_eod_seats_multiplicative",
            "pred_final_eod_seats_additive",
            "pred_final_eod_seats_hybrid",
            "sigma_log_daily",
            "source",
            "model",
            "feature_quality_bucket",
            "amc_coverage",
            "amc_snapshot_count",
            "amc_lateness_p50_minutes",
            "amc_staleness_p50_minutes",
            "amc_observed_seats",
        ]
    )
