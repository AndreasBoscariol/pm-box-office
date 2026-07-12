"""Shadow-only Thursday AMC preview prior nowcasts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .amc_features import (
    aggregate_as_of_features,
    aggregate_final_eod_features,
    build_origin_grid,
    fetch_actuals,
    fetch_preview_actuals,
    fetch_schedule,
    fetch_snapshots,
    latest_snapshots_as_of,
    latest_snapshots_final,
    origin_timestamp_utc,
)
from .live_amc_plugin import (
    LiveAmcPluginConfig,
    active_sample_key,
    additive_final_seat_prediction,
    finalize_live_panel,
    nowcast_from_panel_row,
)
from .live_composition import select_daily_baseline_row
from .schema import MovieOpening
from .thursday_preview import apply_preview_update_to_predicted_preview, positive_float
from .thursday_amc_candidates import prepare_candidate_frame


THURSDAY_PREVIEW_ORIGINS = ("10:00", "12:00", "14:00", "16:00", "18:00", "20:00", "EOD")
MIN_PREVIEW_AMC_TRAIN_ROWS = 3
DEFAULT_PREVIEW_SELECTED_POINT_MODEL = "hybrid"
DEFAULT_PREVIEW_DRAWS = 2_000


@dataclass(frozen=True)
class ThursdayAmcPreviewConfig:
    origin_timezone: str = "America/New_York"
    origins: tuple[str, ...] = THURSDAY_PREVIEW_ORIGINS
    live_amc_config: LiveAmcPluginConfig = LiveAmcPluginConfig()
    min_preview_train_rows: int = MIN_PREVIEW_AMC_TRAIN_ROWS


def predict_preview_distribution(
    *,
    point_preview_gross_usd: float,
    forecast_origin: str,
    policy: dict[str, Any] | None,
    draws: int = DEFAULT_PREVIEW_DRAWS,
    seed: int = 0,
) -> dict[str, Any]:
    """Draw a preview-gross distribution from rolling end-to-end residuals."""
    point = positive_float(point_preview_gross_usd)
    if not math.isfinite(point) or not policy or draws < 1:
        return {"enabled": False, "fallback_reason": "missing_point_or_residual_policy"}
    pools = policy.get("residual_pools") or {}
    origin = str(forecast_origin)
    bucket = origin_bucket(origin)
    choices = [
        (f"origin:{origin}", pools.get(f"origin:{origin}")),
        (f"origin_bucket:{bucket}", pools.get(f"origin_bucket:{bucket}")),
        ("pooled_thursday", pools.get("pooled_thursday")),
        ("pooled_transfer", pools.get("pooled_transfer")),
    ]
    min_n = int(policy.get("min_residual_pool_n") or 5)
    scope = None
    residuals: list[float] = []
    for candidate_scope, values in choices:
        clean = [float(value) for value in (values or []) if np.isfinite(value)]
        if len(clean) >= min_n:
            scope, residuals = candidate_scope, clean
            break
    if not residuals:
        return {"enabled": False, "fallback_reason": "insufficient_empirical_residuals"}
    rng = np.random.default_rng(seed)
    sampled = rng.choice(np.asarray(residuals), size=draws, replace=True)
    values = point * np.exp(sampled)
    return {
        "enabled": True,
        "point_usd": point,
        "draws_usd": values,
        "residual_pool_scope": scope,
        "residual_pool_n": len(residuals),
        "lo80_usd": float(np.quantile(values, 0.10)),
        "hi80_usd": float(np.quantile(values, 0.90)),
        "lo95_usd": float(np.quantile(values, 0.025)),
        "hi95_usd": float(np.quantile(values, 0.975)),
    }


def update_ow_distribution(
    *,
    baseline_ow_usd: float,
    preview_distribution: dict[str, Any],
    preview_update_policy: dict[str, Any] | None,
) -> dict[str, Any]:
    """Pass every preview draw through the separately locked OW updater."""
    draws = np.asarray(preview_distribution.get("draws_usd", []), dtype="float64")
    if draws.size == 0:
        return {"enabled": False, "fallback_reason": "missing_preview_distribution"}
    updated = np.asarray([
        apply_preview_update_to_predicted_preview(
            baseline_ow_usd=baseline_ow_usd,
            predicted_preview_gross_usd=float(draw),
            policy=preview_update_policy,
        ).preview_updated_ow_usd
        for draw in draws
    ], dtype="float64")
    updated = updated[np.isfinite(updated) & (updated > 0)]
    if updated.size == 0:
        return {"enabled": False, "fallback_reason": "preview_updater_rejected_draws"}
    return {
        "enabled": True,
        "draws_usd": updated,
        "point_usd": float(np.median(updated)),
        "lo80_usd": float(np.quantile(updated, 0.10)),
        "hi80_usd": float(np.quantile(updated, 0.90)),
        "lo95_usd": float(np.quantile(updated, 0.025)),
        "hi95_usd": float(np.quantile(updated, 0.975)),
    }


def build_thursday_amc_preview_shadow(
    *,
    baseline_ow_usd: float,
    predicted_preview_gross_usd: float | None,
    policy: dict[str, Any] | None,
    thursday_amc_seats_collected: bool = False,
    thursday_preview_actual_usd: float | None = None,
    preview_nowcast_trained: bool = True,
) -> dict[str, Any]:
    """Return a shadow OW prior from an AMC preview nowcast without production side effects."""

    predicted = positive_float(predicted_preview_gross_usd)
    if not thursday_amc_seats_collected or not math.isfinite(predicted):
        return {
            "shadow_ow_prior_source": "baseline_consensus",
            "shadow_ow_prior_usd": baseline_ow_usd,
            "predicted_thursday_previews_usd": None,
            "thursday_amc_seats_collected": bool(thursday_amc_seats_collected),
            "thursday_preview_actual_usd": thursday_preview_actual_usd,
            "shadow_update_applied": False,
            "fallback_reason": "missing_thursday_amc_preview_nowcast",
        }
    if not preview_nowcast_trained:
        return {
            "shadow_ow_prior_source": "baseline_consensus",
            "shadow_ow_prior_usd": baseline_ow_usd,
            "predicted_thursday_previews_usd": predicted,
            "thursday_amc_seats_collected": bool(thursday_amc_seats_collected),
            "thursday_preview_actual_usd": thursday_preview_actual_usd,
            "shadow_update_applied": False,
            "fallback_reason": "insufficient_preview_amc_training_support",
        }

    update = apply_preview_update_to_predicted_preview(
        baseline_ow_usd=baseline_ow_usd,
        predicted_preview_gross_usd=predicted,
        policy=policy,
    )
    return {
        "shadow_ow_prior_source": update.ow_prior_source,
        "shadow_ow_prior_usd": update.preview_updated_ow_usd,
        "predicted_thursday_previews_usd": predicted,
        "thursday_amc_seats_collected": bool(thursday_amc_seats_collected),
        "thursday_preview_actual_usd": thursday_preview_actual_usd,
        "shadow_update_applied": update.preview_update_applied,
        "fallback_reason": update.fallback_reason,
        "policy_version": update.audit.get("policy_version"),
        "training_cutoff": update.audit.get("training_cutoff"),
    }


def build_thursday_amc_preview_nowcasts(
    conn: Any,
    *,
    movies: Iterable[MovieOpening],
    as_of_utc: datetime,
    artifacts: Any,
    config: ThursdayAmcPreviewConfig = ThursdayAmcPreviewConfig(),
) -> pd.DataFrame:
    """Build shadow Thursday preview nowcasts from live AMC seat snapshots.

    The output is intentionally separate from ``live_plugin_nowcasts`` so the
    preview estimate cannot accidentally replace a production daily component.
    """

    movie_list = list(movies)
    if not movie_list:
        return empty_thursday_amc_preview_frame()

    schedule = fetch_schedule(conn)
    snapshots = fetch_snapshots(conn)
    actuals = fetch_preview_actuals(conn)
    if schedule.empty or snapshots.empty:
        return empty_thursday_amc_preview_frame()

    sample_key = active_sample_key(conn) or "unknown"
    return build_thursday_amc_preview_nowcasts_from_frames(
        schedule=schedule,
        snapshots=snapshots,
        actuals=actuals,
        movies=movie_list,
        as_of_utc=as_of_utc,
        artifacts=artifacts,
        sample_key=sample_key,
        config=config,
    )


def build_thursday_amc_preview_nowcasts_from_frames(
    *,
    schedule: pd.DataFrame,
    snapshots: pd.DataFrame,
    actuals: pd.DataFrame,
    movies: Iterable[MovieOpening],
    as_of_utc: datetime,
    artifacts: Any,
    sample_key: str = "unknown",
    config: ThursdayAmcPreviewConfig = ThursdayAmcPreviewConfig(),
) -> pd.DataFrame:
    movie_list = list(movies)
    if not movie_list or schedule.empty or snapshots.empty:
        return empty_thursday_amc_preview_frame()

    schedule = schedule.copy()
    snapshots = snapshots.copy()
    actuals = actuals.copy()
    for frame in [schedule, snapshots, actuals]:
        if "exhibition_date" in frame.columns:
            frame["exhibition_date"] = pd.to_datetime(frame["exhibition_date"], errors="coerce").dt.date
    if "is_primary_training_target" in actuals.columns:
        actuals = actuals.loc[actuals["is_primary_training_target"].fillna(False).astype(bool)].copy()

    preview_dates = {movie.movie_id: movie.opening_weekend_start - timedelta(days=1) for movie in movie_list}
    target_schedule = schedule.loc[
        schedule["movie_id"].isin(preview_dates)
        & schedule["exhibition_date"].eq(schedule["movie_id"].map(preview_dates))
    ].copy()
    if target_schedule.empty:
        return empty_thursday_amc_preview_frame()

    candidate_origins = [
        origin
        for origin in config.origins
        if any(
            origin_timestamp_utc(preview_date, origin, config.origin_timezone) <= pd.Timestamp(as_of_utc)
            for preview_date in preview_dates.values()
        )
    ]
    if not candidate_origins:
        return empty_thursday_amc_preview_frame()

    grid = build_origin_grid(schedule, candidate_origins, config.origin_timezone)
    grid = grid.loc[pd.to_datetime(grid["forecast_origin_utc"], utc=True).le(pd.Timestamp(as_of_utc))].copy()
    if grid.empty:
        return empty_thursday_amc_preview_frame()

    asof = latest_snapshots_as_of(snapshots, grid)
    features = aggregate_as_of_features(asof)
    final_features = aggregate_final_eod_features(latest_snapshots_final(snapshots))
    panel = grid.merge(schedule, on=["movie_id", "amc_movie_id", "exhibition_date"], how="left")
    panel = panel.merge(
        features,
        on=["movie_id", "amc_movie_id", "exhibition_date", "forecast_origin", "forecast_origin_utc"],
        how="left",
    )
    panel = panel.merge(final_features, on=["movie_id", "amc_movie_id", "exhibition_date"], how="left")
    panel = panel.merge(actuals, on=["movie_id", "exhibition_date"], how="left", suffixes=("", "_actual"))
    panel = finalize_live_panel(panel)
    if panel.empty:
        return empty_thursday_amc_preview_frame()

    rows: list[dict[str, object]] = []
    for movie in movie_list:
        movie_panel = panel.loc[
            panel["movie_id"].eq(movie.movie_id)
            & panel["exhibition_date"].eq(preview_dates[movie.movie_id])
        ].sort_values("forecast_origin_utc")
        if movie_panel.empty:
            continue
        baseline_ow = baseline_ow_from_artifacts(artifacts, movie)
        for _, target in movie_panel.iterrows():
            policy = getattr(artifacts, "thursday_amc_preview_policy", {}) or {}
            policy_min_train_rows = int(policy.get("min_train_rows") or config.min_preview_train_rows)
            plugin = preview_nowcast_from_panel_row(
                panel,
                target,
                policy=policy,
                min_train_rows=policy_min_train_rows,
            )
            preview_model_is_trained = plugin is not None
            if plugin is None:
                continue
            predicted = positive_float(plugin.get("pred_daily_gross_usd"))
            if not math.isfinite(predicted):
                continue
            actual_preview = positive_float(target.get("actual_gross_usd"))
            preview_train_n = int(plugin.get("preview_amc_training_rows") or preview_amc_training_support(panel, target))
            preview_nowcast_trained = preview_model_is_trained and preview_train_n >= policy_min_train_rows
            shadow = build_thursday_amc_preview_shadow(
                baseline_ow_usd=baseline_ow,
                predicted_preview_gross_usd=predicted,
                policy=getattr(artifacts, "thursday_preview_policy", {}),
                thursday_amc_seats_collected=True,
                thursday_preview_actual_usd=actual_preview if math.isfinite(actual_preview) else None,
                preview_nowcast_trained=preview_nowcast_trained,
            )
            preview_distribution = predict_preview_distribution(
                point_preview_gross_usd=predicted,
                forecast_origin=str(target.get("forecast_origin")),
                policy=policy,
                seed=int(movie.movie_id),
            )
            ow_distribution = update_ow_distribution(
                baseline_ow_usd=baseline_ow,
                preview_distribution=preview_distribution,
                preview_update_policy=getattr(artifacts, "thursday_preview_policy", {}),
            ) if preview_distribution.get("enabled") else {"enabled": False}
            rows.append(
                {
                    "release_run_id": movie.release_run_id,
                    "movie_id": movie.movie_id,
                    "title": movie.title,
                    "opening_weekend_start": movie.opening_weekend_start,
                    "preview_exhibition_date": preview_dates[movie.movie_id],
                    "forecast_origin": str(target.get("forecast_origin")),
                    "forecast_origin_utc": target.get("forecast_origin_utc"),
                    "as_of_utc": pd.Timestamp(as_of_utc),
                    "sample_key": sample_key,
                    "predicted_thursday_previews_usd": predicted,
                    "predicted_thursday_previews_lo80_usd": preview_distribution.get("lo80_usd"),
                    "predicted_thursday_previews_hi80_usd": preview_distribution.get("hi80_usd"),
                    "predicted_thursday_previews_lo95_usd": preview_distribution.get("lo95_usd"),
                    "predicted_thursday_previews_hi95_usd": preview_distribution.get("hi95_usd"),
                    "preview_residual_pool_scope": preview_distribution.get("residual_pool_scope"),
                    "predicted_thursday_previews_multiplicative_usd": plugin.get("pred_daily_gross_multiplicative_usd"),
                    "predicted_thursday_previews_additive_usd": plugin.get("pred_daily_gross_additive_usd"),
                    "predicted_thursday_previews_hybrid_usd": plugin.get("pred_daily_gross_hybrid_usd"),
                    "pred_final_eod_preview_seats_multiplicative": plugin.get("pred_final_eod_seats_multiplicative"),
                    "pred_final_eod_preview_seats_additive": plugin.get("pred_final_eod_seats_additive"),
                    "pred_final_eod_preview_seats_hybrid": plugin.get("pred_final_eod_seats_hybrid"),
                    "baseline_ow_usd": baseline_ow if math.isfinite(baseline_ow) else None,
                    "shadow_ow_prior_source": shadow["shadow_ow_prior_source"],
                    "shadow_ow_prior_usd": shadow["shadow_ow_prior_usd"],
                    "shadow_ow_lo80_usd": ow_distribution.get("lo80_usd"),
                    "shadow_ow_hi80_usd": ow_distribution.get("hi80_usd"),
                    "shadow_ow_lo95_usd": ow_distribution.get("lo95_usd"),
                    "shadow_ow_hi95_usd": ow_distribution.get("hi95_usd"),
                    "shadow_update_applied": shadow["shadow_update_applied"],
                    "shadow_fallback_reason": shadow["fallback_reason"],
                    "preview_amc_training_rows": preview_train_n,
                    "preview_amc_nowcast_trained": preview_nowcast_trained,
                    "thursday_preview_actual_usd": actual_preview if math.isfinite(actual_preview) else None,
                    "thursday_amc_seats_collected": True,
                    "source": "AMC_live_db",
                    "model": f"AMC_thursday_preview_{plugin['model']}",
                    "feature_quality_bucket": plugin["feature_quality_bucket"],
                    "amc_coverage": plugin["amc_coverage"],
                    "amc_snapshot_count": plugin["amc_snapshot_count"],
                    "amc_lateness_p50_minutes": plugin["amc_lateness_p50_minutes"],
                    "amc_staleness_p50_minutes": plugin["amc_staleness_p50_minutes"],
                    "amc_observed_preview_seats": plugin["amc_observed_seats"],
                    "policy_version": shadow.get("policy_version"),
                    "training_cutoff": shadow.get("training_cutoff"),
                    "selected_candidate": policy.get("selected_candidate"),
                    "target_classification": "thursday_only",
                }
            )
    if not rows:
        return empty_thursday_amc_preview_frame()
    return pd.DataFrame(rows).drop_duplicates(["movie_id", "forecast_origin"], keep="last")


def preview_nowcast_from_panel_row(
    panel: pd.DataFrame,
    target: pd.Series,
    *,
    policy: dict[str, Any] | None,
    min_train_rows: int = MIN_PREVIEW_AMC_TRAIN_ROWS,
) -> dict[str, object] | None:
    if not policy or not bool(policy.get("enabled", False)):
        return None
    if policy.get("selected_candidate"):
        return candidate_nowcast_from_panel_row(target, policy=policy)
    s_obs = positive_float(target.get("s_obs"))
    if not math.isfinite(s_obs):
        return None
    forecast_origin = str(target.get("forecast_origin"))
    current_date = pd.Timestamp(target.get("exhibition_date")).date()
    train = preview_training_frame(panel, current_date=current_date, forecast_origin=forecast_origin)
    all_train = preview_training_frame(panel, current_date=current_date)
    pool_name, pool = select_preview_training_pool(
        train,
        forecast_origin=forecast_origin,
        min_train_rows=min_train_rows,
        all_preview_train=all_train,
    )
    if pool.empty:
        return None

    pool = pool.copy()
    pool["raw_pace"] = pd.to_numeric(pool["s_obs"], errors="coerce") / pd.to_numeric(pool["s_final_eod"], errors="coerce")
    pace = float(pool["raw_pace"].dropna().median()) if len(pool) else math.nan
    if not math.isfinite(pace) or pace <= 0:
        return None
    pace = float(np.clip(pace, 0.01, 1.0))
    pred_final_eod_seats_multiplicative = s_obs / pace
    pred_final_eod_seats_additive = additive_final_seat_prediction(pool, target)
    if not math.isfinite(pred_final_eod_seats_additive):
        pred_final_eod_seats_additive = pred_final_eod_seats_multiplicative
    pred_final_eod_seats_hybrid = 0.5 * (pred_final_eod_seats_multiplicative + pred_final_eod_seats_additive)

    pool["bridge_residual"] = np.log(pd.to_numeric(pool["actual_gross_usd"], errors="coerce")) - np.log1p(
        pd.to_numeric(pool["s_final_eod"], errors="coerce")
    )
    bridge = float(pool["bridge_residual"].dropna().median()) if len(pool) else math.nan
    if not math.isfinite(bridge):
        return None

    predictions = {
        "multiplicative": float(math.exp(bridge + math.log1p(pred_final_eod_seats_multiplicative))),
        "additive": float(math.exp(bridge + math.log1p(pred_final_eod_seats_additive))),
        "hybrid": float(math.exp(bridge + math.log1p(pred_final_eod_seats_hybrid))),
    }
    selected = str(policy.get("selected_point_model") or DEFAULT_PREVIEW_SELECTED_POINT_MODEL)
    if selected not in predictions:
        selected = DEFAULT_PREVIEW_SELECTED_POINT_MODEL
    if not math.isfinite(predictions[selected]) or predictions[selected] <= 0:
        return None

    return {
        "pred_daily_gross_usd": predictions[selected],
        "pred_daily_gross_multiplicative_usd": predictions["multiplicative"],
        "pred_daily_gross_additive_usd": predictions["additive"],
        "pred_daily_gross_hybrid_usd": predictions["hybrid"],
        "pred_final_eod_seats_multiplicative": pred_final_eod_seats_multiplicative,
        "pred_final_eod_seats_additive": pred_final_eod_seats_additive,
        "pred_final_eod_seats_hybrid": pred_final_eod_seats_hybrid,
        "sigma_log_daily": float(policy.get("sigma_log", 0.55) or 0.55),
        "model": f"AMC_thursday_preview_pace_bridge_v1:{pool_name}:{selected}",
        "feature_quality_bucket": str(target.get("collection_quality_bucket") or "low"),
        "amc_coverage": finite_float(target.get("coverage")),
        "amc_snapshot_count": int(finite_float(target.get("n_snapshots")) or 0),
        "amc_lateness_p50_minutes": finite_float(target.get("delay_p50_minutes")),
        "amc_staleness_p50_minutes": finite_float(target.get("staleness_p50_minutes")),
        "amc_observed_seats": s_obs,
        "preview_amc_training_rows": int(len(pool)),
        "preview_amc_training_pool": pool_name,
    }


def candidate_nowcast_from_panel_row(target: pd.Series, *, policy: dict[str, Any]) -> dict[str, object] | None:
    """Apply the frozen winning B2 candidate without fitting on live data."""
    candidate = str(policy.get("selected_candidate") or "")
    stage1 = policy.get("stage1")
    stage2 = policy.get("stage2")
    row = prepare_candidate_frame(pd.DataFrame([target])).iloc[0]

    def linear(model: dict[str, Any] | None) -> float:
        if not isinstance(model, dict):
            return math.nan
        value = finite_float(model.get("intercept"))
        if not math.isfinite(value):
            return math.nan
        for feature, coefficient in (model.get("coefficients") or {}).items():
            feature_value = finite_float(row.get(feature))
            coefficient_value = finite_float(coefficient)
            if not math.isfinite(feature_value) or not math.isfinite(coefficient_value):
                return math.nan
            value += coefficient_value * feature_value
        return value

    if candidate in {"TH2_predicted_eod_bridge", "TH3_pooled_day_bridge", "TH_TRANSFER_V0"}:
        pred_log_eod = linear(stage1)
        if not math.isfinite(pred_log_eod) or not isinstance(stage2, dict):
            return None
        log_prediction = finite_float(stage2.get("alpha")) + finite_float(stage2.get("beta")) * pred_log_eod
        pred_final = math.expm1(pred_log_eod)
    elif candidate in {"TH0_asof_direct", "TH4_prior_assisted"}:
        log_prediction = linear(stage1)
        pred_final = math.nan
    else:
        # TH1 is diagnostic-only because live EOD seats are unavailable intraday.
        return None
    log_prediction += finite_float(policy.get("bias_adjustment")) if math.isfinite(finite_float(policy.get("bias_adjustment"))) else 0.0
    if not math.isfinite(log_prediction):
        return None
    predicted = math.exp(log_prediction)
    return {
        "pred_daily_gross_usd": predicted,
        "pred_daily_gross_multiplicative_usd": predicted,
        "pred_daily_gross_additive_usd": predicted,
        "pred_daily_gross_hybrid_usd": predicted,
        "pred_final_eod_seats_multiplicative": pred_final,
        "pred_final_eod_seats_additive": pred_final,
        "pred_final_eod_seats_hybrid": pred_final,
        "sigma_log_daily": math.nan,
        "model": f"AMC_thursday_preview:{candidate}",
        "feature_quality_bucket": str(target.get("collection_quality_bucket") or "low"),
        "amc_coverage": finite_float(target.get("coverage")),
        "amc_snapshot_count": int(finite_float(target.get("n_snapshots")) or 0),
        "amc_lateness_p50_minutes": finite_float(target.get("delay_p50_minutes")),
        "amc_staleness_p50_minutes": finite_float(target.get("staleness_p50_minutes")),
        "amc_observed_seats": finite_float(target.get("s_obs")),
        "preview_amc_training_rows": int(policy.get("training_rows_scored") or policy.get("min_train_rows") or 0),
        "preview_amc_training_pool": "frozen_candidate_policy",
    }


def preview_training_frame(panel: pd.DataFrame, *, current_date: object, forecast_origin: str | None = None) -> pd.DataFrame:
    if "is_preview" not in panel.columns:
        return pd.DataFrame()
    work = panel.loc[pd.to_datetime(panel["exhibition_date"], errors="coerce").dt.date < pd.Timestamp(current_date).date()].copy()
    if forecast_origin is not None:
        work = work.loc[work["forecast_origin"].astype(str).eq(str(forecast_origin))].copy()
    work = work.loc[
        pd.to_numeric(work["is_preview"], errors="coerce").fillna(0).astype(int).eq(1)
        & pd.to_numeric(work["s_obs"], errors="coerce").gt(0)
        & pd.to_numeric(work["s_final_eod"], errors="coerce").gt(0)
        & pd.to_numeric(work["actual_gross_usd"], errors="coerce").gt(0)
    ].copy()
    return work


def origin_bucket(origin: object) -> str:
    value = str(origin)
    if value in {"10:00", "12:00", "14:00"}:
        return "early"
    if value in {"16:00", "18:00", "20:00", "EOD"}:
        return "late"
    return "all"


def select_preview_training_pool(
    train_same_origin: pd.DataFrame,
    *,
    forecast_origin: str,
    min_train_rows: int,
    all_preview_train: pd.DataFrame | None = None,
) -> tuple[str, pd.DataFrame]:
    all_train = train_same_origin if all_preview_train is None else all_preview_train
    if len(train_same_origin) >= min_train_rows:
        return "origin", train_same_origin
    if all_train is not None and not all_train.empty:
        bucket = origin_bucket(forecast_origin)
        bucket_train = all_train.loc[all_train["forecast_origin"].map(origin_bucket).eq(bucket)].copy()
        if len(bucket_train) >= min_train_rows:
            return f"origin_bucket_{bucket}", bucket_train
        if len(all_train) >= min_train_rows:
            return "all_preview", all_train.copy()
    return "insufficient_preview_history", pd.DataFrame()


def preview_amc_training_support(panel: pd.DataFrame, target: pd.Series) -> int:
    current_date = pd.Timestamp(target.get("exhibition_date")).date()
    forecast_origin = str(target.get("forecast_origin"))
    if "is_preview" not in panel.columns:
        return 0
    train = panel.loc[pd.to_datetime(panel["exhibition_date"], errors="coerce").dt.date < current_date].copy()
    if train.empty:
        return 0
    support = train.loc[
        train["forecast_origin"].astype(str).eq(forecast_origin)
        & pd.to_numeric(train["is_preview"], errors="coerce").fillna(0).astype(int).eq(1)
        & pd.to_numeric(train["s_obs"], errors="coerce").gt(0)
        & pd.to_numeric(train["s_final_eod"], errors="coerce").gt(0)
        & pd.to_numeric(train["actual_gross_usd"], errors="coerce").gt(0)
    ]
    return int(len(support))


def finite_float(value: object) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if np.isfinite(out) else math.nan


def baseline_ow_from_artifacts(artifacts: Any, movie: MovieOpening) -> float:
    try:
        baseline = select_daily_baseline_row(artifacts.daily_baseline, movie)
    except (AttributeError, KeyError, ValueError):
        return math.nan
    values = [positive_float(baseline.get(column)) for column in ("pre_fri_usd", "pre_sat_usd", "pre_sun_usd")]
    return float(sum(values)) if all(math.isfinite(value) for value in values) else math.nan


def empty_thursday_amc_preview_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "release_run_id",
            "movie_id",
            "title",
            "opening_weekend_start",
            "preview_exhibition_date",
            "forecast_origin",
            "forecast_origin_utc",
            "as_of_utc",
            "sample_key",
            "predicted_thursday_previews_usd",
            "predicted_thursday_previews_lo80_usd", "predicted_thursday_previews_hi80_usd",
            "predicted_thursday_previews_lo95_usd", "predicted_thursday_previews_hi95_usd",
            "preview_residual_pool_scope",
            "predicted_thursday_previews_multiplicative_usd",
            "predicted_thursday_previews_additive_usd",
            "predicted_thursday_previews_hybrid_usd",
            "pred_final_eod_preview_seats_multiplicative",
            "pred_final_eod_preview_seats_additive",
            "pred_final_eod_preview_seats_hybrid",
            "baseline_ow_usd",
            "shadow_ow_prior_source",
            "shadow_ow_prior_usd",
            "shadow_ow_lo80_usd", "shadow_ow_hi80_usd", "shadow_ow_lo95_usd", "shadow_ow_hi95_usd",
            "shadow_update_applied",
            "shadow_fallback_reason",
            "preview_amc_training_rows",
            "preview_amc_nowcast_trained",
            "thursday_preview_actual_usd",
            "thursday_amc_seats_collected",
            "source",
            "model",
            "feature_quality_bucket",
            "amc_coverage",
            "amc_snapshot_count",
            "amc_lateness_p50_minutes",
            "amc_staleness_p50_minutes",
            "amc_observed_preview_seats",
            "policy_version",
            "training_cutoff",
            "selected_candidate", "target_classification",
        ]
    )
