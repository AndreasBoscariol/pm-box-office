"""Live weekend composition from daily baselines and optional AMC nowcasts."""

from __future__ import annotations

import math
from datetime import datetime, time, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import ModelArtifacts
from .constants import (
    ACTUAL_COLUMNS,
    BASELINE_COLUMNS,
    DAYS,
    KNOWN_ACTUAL_DAYS,
    LIVE_FRIDAY_REGIME,
    LIVE_SUNDAY_REGIME,
    PLUGIN_TARGET_DAY,
    PRE_RELEASE_REGIME,
    TARGET_TO_DAY,
)
from .intervals import log_normal_interval
from .live_amc_plugin import amc_interval_cell_key
from .live_weekend_distribution import build_payload, deterministic_seed, resolved_policy
from .opening_thursday_actual import update_ow_from_opening_thursday_actual
from .pre_release import forecast_pre_release_opening_weekend
from .schema import DailyComponent, ForecastOrigin, ForecastResult, MovieOpening
from .thursday_preview import DEFAULT_SCALE_TOLERANCE_PCT, update_ow_prior_from_reported_preview

PENDING_ACTUAL_BASELINE_COLUMNS = {
    "Friday": "pre_fri_usd",
    "Saturday": "pre_sat_usd",
    "Sunday": "pre_sun_usd",
}


def _positive_float(value: Any) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if np.isfinite(out) and float(out) > 0 else math.nan


def select_daily_baseline_row(baseline: pd.DataFrame, movie: MovieOpening) -> pd.Series:
    if baseline.empty:
        raise ValueError("daily baseline artifact is empty")
    frame = baseline.copy()
    for column in ["movie_id", "release_run_id"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "release_run_id" in frame.columns:
        rows = frame.loc[frame["release_run_id"].eq(movie.release_run_id)]
    else:
        rows = frame.loc[frame["movie_id"].eq(movie.movie_id)]
    if rows.empty:
        raise KeyError(f"No daily baseline row for release_run_id={movie.release_run_id}")
    return rows.iloc[-1]


def _pre_weekend_baseline_sum(baseline: pd.Series) -> float:
    values = [_positive_float(baseline.get(column)) for column in ("pre_fri_usd", "pre_sat_usd", "pre_sun_usd")]
    return float(sum(values)) if all(np.isfinite(value) for value in values) else math.nan


def _baseline_opening_weekend_scale(baseline: pd.Series, tolerance_pct: float) -> tuple[float, str | None]:
    pre_sum = _pre_weekend_baseline_sum(baseline)
    if not np.isfinite(pre_sum) or pre_sum <= 0:
        return math.nan, "missing_pre_weekend_component_scale"
    total = _positive_float(baseline.get("total_forecast_usd"))
    if np.isfinite(total):
        rel_diff = abs(total - pre_sum) / total
        if rel_diff > tolerance_pct:
            return math.nan, "pre_weekend_components_do_not_match_total_forecast"
    return pre_sum, None


def _scale_columns(row: pd.Series, columns: tuple[str, ...], scale: float) -> None:
    for column in columns:
        value = _positive_float(row.get(column))
        if np.isfinite(value):
            row[column] = value * scale


def _rescale_pending_live_components(
    row: pd.Series,
    *,
    old_ow: float,
    updated_ow: float,
    default_scale: float,
) -> None:
    _scale_columns(row, ("pre_fri_usd", "pre_sat_usd", "pre_sun_usd"), default_scale)

    actual_fri = _positive_float(row.get("actual_fri_usd"))
    after_fri_sum = sum(
        value for value in (_positive_float(row.get("after_fri_sat_usd")), _positive_float(row.get("after_fri_sun_usd"))) if np.isfinite(value)
    )
    if np.isfinite(actual_fri) and after_fri_sum > 0:
        target_remaining = updated_ow - actual_fri
        if target_remaining > 0:
            _scale_columns(row, ("after_fri_sat_usd", "after_fri_sun_usd"), target_remaining / after_fri_sum)
    else:
        _scale_columns(row, ("after_fri_sat_usd", "after_fri_sun_usd"), default_scale)

    actual_sat = _positive_float(row.get("actual_sat_usd"))
    after_sat_sun = _positive_float(row.get("after_sat_sun_usd"))
    if np.isfinite(actual_fri) and np.isfinite(actual_sat) and np.isfinite(after_sat_sun) and after_sat_sun > 0:
        target_sunday = updated_ow - actual_fri - actual_sat
        if target_sunday > 0:
            row["after_sat_sun_usd"] = after_sat_sun * (target_sunday / after_sat_sun)
    elif np.isfinite(after_sat_sun):
        row["after_sat_sun_usd"] = after_sat_sun * default_scale


def _select_adjusted_daily_baseline_row(
    *,
    movie: MovieOpening,
    origin: ForecastOrigin,
    artifacts: ModelArtifacts,
) -> tuple[pd.Series, dict[str, Any]]:
    baseline = select_daily_baseline_row(artifacts.daily_baseline, movie).copy()
    policy = artifacts.thursday_preview_policy
    tolerance_pct = float(policy.get("old_scale_tolerance_pct", DEFAULT_SCALE_TOLERANCE_PCT)) if policy else DEFAULT_SCALE_TOLERANCE_PCT
    old_ow, scale_reason = _baseline_opening_weekend_scale(baseline, tolerance_pct)
    audit: dict[str, Any] = {
        "ow_prior_source": "baseline_consensus",
        "ow_prior_baseline_usd": old_ow if np.isfinite(old_ow) else None,
        "ow_prior_usd": old_ow if np.isfinite(old_ow) else None,
        "thursday_preview_update_applied": False,
        "opening_thursday_actual_update_applied": False,
    }
    if scale_reason:
        audit["thursday_preview_fallback_reason"] = scale_reason
        return baseline, audit

    opening_policy = getattr(artifacts, "opening_thursday_actual_policy", {})
    if opening_policy:
        actual_update = update_ow_from_opening_thursday_actual(
            baseline_ow_usd=old_ow,
            opening_thursday_daily_gross_usd=baseline.get("opening_thursday_daily_gross_usd"),
            frozen_policy=opening_policy,
            row=baseline,
            execution_time=origin.as_of_utc,
        )
        audit.update(
            {
                "ow_prior_source": "opening_thursday_actual_ratio_update_prod" if actual_update.production_update_applied else "baseline_consensus",
                "ow_prior_usd": actual_update.updated_ow_usd,
                "opening_thursday_daily_gross_usd": actual_update.opening_thursday_daily_gross_usd,
                "opening_thursday_actual_update_applied": actual_update.production_update_applied,
                "opening_thursday_actual_policy_name": actual_update.policy_name,
                "opening_thursday_actual_policy_version": actual_update.policy_version,
                "opening_thursday_actual_training_cutoff": actual_update.training_cutoff,
                "opening_thursday_actual_update_multiplier": actual_update.update_multiplier,
                "opening_thursday_actual_log_update": actual_update.audit.get("log_update"),
                "opening_thursday_actual_dollar_update": actual_update.audit.get("dollar_update"),
                "opening_thursday_actual_exclusion_reason": actual_update.exclusion_reason,
                "opening_thursday_actual_baseline_distribution": actual_update.baseline_distribution,
                "opening_thursday_actual_updated_distribution": actual_update.updated_distribution,
                "opening_thursday_actual_artifact_hash": actual_update.audit.get("artifact_hash"),
            }
        )
        if actual_update.production_update_applied:
            _rescale_pending_live_components(
                baseline,
                old_ow=old_ow,
                updated_ow=actual_update.updated_ow_usd,
                default_scale=actual_update.update_multiplier,
            )
            baseline["total_forecast_usd"] = actual_update.updated_ow_usd
            return baseline, audit
        audit["opening_thursday_actual_fallback_reason"] = actual_update.exclusion_reason

    update = update_ow_prior_from_reported_preview(
        baseline_ow_usd=old_ow,
        row=baseline,
        policy=policy,
    )
    audit.update(
        {
            "ow_prior_source": update.ow_prior_source,
            "ow_prior_usd": update.preview_updated_ow_usd,
            "preview_gross_usd": update.preview_gross_usd,
            "preview_scale_factor": update.preview_scale_factor,
            "thursday_preview_update_applied": update.preview_update_applied,
            "thursday_preview_policy_version": update.audit.get("policy_version"),
            "thursday_preview_training_cutoff": update.audit.get("training_cutoff"),
            "interval_calibration_status": policy.get("interval_calibration_status") if policy else None,
        }
    )
    if not update.preview_update_applied:
        audit["thursday_preview_fallback_reason"] = update.fallback_reason
        return baseline, audit

    _rescale_pending_live_components(
        baseline,
        old_ow=old_ow,
        updated_ow=update.preview_updated_ow_usd,
        default_scale=update.preview_scale_factor,
    )
    baseline["total_forecast_usd"] = update.preview_updated_ow_usd
    return baseline, audit


def select_plugin_row(plugin: pd.DataFrame, movie: MovieOpening, origin: ForecastOrigin, day: str) -> pd.Series | None:
    if plugin.empty:
        return None
    frame = plugin.copy()
    frame["movie_id"] = pd.to_numeric(frame["movie_id"], errors="coerce")
    mask = (
        frame["movie_id"].eq(movie.movie_id)
        & frame["regime"].astype(str).eq(origin.regime)
        & frame["forecast_origin"].astype(str).eq(str(origin.forecast_origin))
        & frame["target_day"].astype(str).str.lower().eq(day.lower())
    )
    rows = frame.loc[mask]
    return None if rows.empty else rows.iloc[-1]


def _fallback_sigma(artifacts: ModelArtifacts, regime: str, day: str) -> float:
    composition = artifacts.manifest.get("composition", {})
    daily_sigmas = composition.get("daily_sigma_log", {})
    value = daily_sigmas.get(regime, {}).get(day) if isinstance(daily_sigmas.get(regime), dict) else None
    return float(value if value is not None else composition.get("fallback_daily_sigma_log", 0.45))


def _daily_policy_payload(artifacts: ModelArtifacts, regime: str, day: str) -> dict[str, Any] | None:
    by_regime = artifacts.daily_interval_policy.get("by_regime_day", {})
    if not isinstance(by_regime, dict):
        return None
    by_day = by_regime.get(regime)
    if not isinstance(by_day, dict):
        return None
    payload = by_day.get(day)
    return payload if isinstance(payload, dict) else None


def _amc_policy_payload(
    artifacts: ModelArtifacts,
    origin: ForecastOrigin,
    component: DailyComponent,
) -> dict[str, Any] | None:
    policy = artifacts.amc_interval_policy
    cells = policy.get("cells") if isinstance(policy, dict) else None
    if not isinstance(cells, dict) or not cells:
        return None
    sample_key = str(policy.get("sample_key") or "*")
    quality = str(component.component_notes or "*")
    candidates = [
        {
            "sample_key": sample_key,
            "regime": origin.regime,
            "forecast_origin": origin.forecast_origin,
            "target_day": component.component_day,
            "feature_quality_bucket": quality,
        },
        {
            "sample_key": sample_key,
            "regime": origin.regime,
            "forecast_origin": origin.forecast_origin,
            "target_day": component.component_day,
        },
        {
            "sample_key": sample_key,
            "regime": origin.regime,
            "target_day": component.component_day,
        },
        {"sample_key": sample_key},
        {
            "regime": origin.regime,
            "forecast_origin": origin.forecast_origin,
            "target_day": component.component_day,
            "feature_quality_bucket": quality,
        },
        {
            "regime": origin.regime,
            "forecast_origin": origin.forecast_origin,
            "target_day": component.component_day,
        },
        {
            "regime": origin.regime,
            "target_day": component.component_day,
        },
        {},
    ]
    for candidate in candidates:
        payload = cells.get(amc_interval_cell_key(candidate))
        if isinstance(payload, dict):
            return payload
    return None


def _component_policy_payload(
    artifacts: ModelArtifacts,
    origin: ForecastOrigin,
    component: DailyComponent,
) -> dict[str, Any] | None:
    if component.component_type == "AMC_nowcast":
        amc = _amc_policy_payload(artifacts, origin, component)
        generic = _daily_policy_payload(artifacts, origin.regime, component.component_day)
        if amc and component.component_day == "Friday" and generic:
            return _policy_with_generic_interval_floor(amc, generic)
        return amc or generic
    return _daily_policy_payload(artifacts, origin.regime, component.component_day)


def _conditional_daily_policy_payload(
    artifacts: ModelArtifacts,
    origin: ForecastOrigin,
    day: str,
) -> dict[str, Any] | None:
    by_regime = artifacts.daily_interval_policy.get("conditional_by_regime_day", {})
    if not isinstance(by_regime, dict):
        return None
    by_day = by_regime.get(origin.regime)
    if not isinstance(by_day, dict):
        return None
    payload = by_day.get(day)
    return payload if isinstance(payload, dict) else None


def _conditional_sunday_residual_weights(payload: dict[str, Any], conditioning_value: float) -> tuple[np.ndarray, np.ndarray] | None:
    samples = payload.get("residual_samples")
    if not isinstance(samples, list) or not samples:
        return None
    frame = pd.DataFrame(samples)
    if not {"conditioning_value", "log_residual"}.issubset(frame.columns):
        return None
    x = pd.to_numeric(frame["conditioning_value"], errors="coerce").to_numpy(dtype="float64")
    residuals = pd.to_numeric(frame["log_residual"], errors="coerce").to_numpy(dtype="float64")
    finite = np.isfinite(x) & np.isfinite(residuals)
    x = x[finite]
    residuals = residuals[finite]
    if residuals.size == 0 or not np.isfinite(conditioning_value):
        return None
    bandwidth = float(payload.get("bandwidth", 0.35))
    shrink_k = float(payload.get("shrink_k", 20.0))
    local = np.exp(-((conditioning_value - x) ** 2) / (2 * bandwidth**2))
    if not np.isfinite(local).all() or local.sum() <= 0:
        local = np.ones_like(x, dtype="float64")
    local = local / local.sum()
    n_eff = 1.0 / float(np.sum(local**2))
    alpha = n_eff / (n_eff + shrink_k)
    pooled = np.full_like(local, 1.0 / len(local), dtype="float64")
    weights = alpha * local + (1.0 - alpha) * pooled
    weights = weights / weights.sum()
    return residuals, weights


def _policy_with_generic_interval_floor(amc: dict[str, Any], generic: dict[str, Any]) -> dict[str, Any]:
    """Keep provisional opening-Friday AMC uncertainty at least as wide as the generic policy."""

    out = dict(amc)
    for key in ["lo80_log", "lo95_log"]:
        if key in amc and key in generic:
            out[key] = min(float(amc[key]), float(generic[key]))
    for key in ["hi80_log", "hi95_log"]:
        if key in amc and key in generic:
            out[key] = max(float(amc[key]), float(generic[key]))
    out["sigma_log"] = max(float(amc.get("sigma_log", 0.0)), float(generic.get("sigma_log", 0.0)))
    amc_width = float(amc.get("hi80_log", 0.0)) - float(amc.get("lo80_log", 0.0))
    generic_width = float(generic.get("hi80_log", 0.0)) - float(generic.get("lo80_log", 0.0))
    if generic_width > amc_width and generic.get("residual_samples_log"):
        out["residual_samples_log"] = generic["residual_samples_log"]
        out.pop("residual_sample_weights", None)
    out["opening_friday_generic_floor_applied"] = True
    return out


def _component_interval_model(
    artifacts: ModelArtifacts,
    origin: ForecastOrigin,
    component: DailyComponent,
) -> str:
    if component.component_type == "actual":
        return "daily_log_sigma"
    if component.component_type == "AMC_nowcast" and _amc_policy_payload(artifacts, origin, component):
        return "empirical_amc_quantile_with_generic_floor" if component.component_day == "Friday" else "empirical_amc_quantile"
    if artifacts.daily_interval_policy:
        return "empirical_daily_quantile"
    return "daily_log_sigma"


def _empirical_interval(point: float, payload: dict[str, Any] | None) -> dict[str, float] | None:
    if payload is None:
        return None
    required = ("lo80_log", "hi80_log", "lo95_log", "hi95_log")
    if not all(key in payload for key in required):
        return None
    values = {key: float(payload[key]) for key in required}
    if not all(np.isfinite(value) for value in values.values()):
        return None
    interval = {
        "lo80_usd": point * math.exp(values["lo80_log"]),
        "hi80_usd": point * math.exp(values["hi80_log"]),
        "lo95_usd": point * math.exp(values["lo95_log"]),
        "hi95_usd": point * math.exp(values["hi95_log"]),
    }
    if not (interval["lo95_usd"] <= interval["lo80_usd"] <= point <= interval["hi80_usd"] <= interval["hi95_usd"]):
        return None
    return interval


def _component_from_payload(
    *,
    day: str,
    component_type: str,
    point: float,
    model: str,
    source: str,
    payload: dict[str, Any] | None,
    fallback_sigma: float,
    notes: str | None = None,
) -> DailyComponent:
    sigma = float(payload.get("sigma_log", fallback_sigma)) if payload else fallback_sigma
    interval = _empirical_interval(point, payload)
    return DailyComponent(
        component_day=day,
        component_type=component_type,
        component_point_usd=point,
        component_sigma_log=sigma,
        component_model=model,
        component_source=source,
        component_notes=notes,
        component_lo80_usd=interval["lo80_usd"] if interval else None,
        component_hi80_usd=interval["hi80_usd"] if interval else None,
        component_lo95_usd=interval["lo95_usd"] if interval else None,
        component_hi95_usd=interval["hi95_usd"] if interval else None,
    )


def _simulate_component_sum_from_policies(
    components: list[DailyComponent],
    *,
    artifacts: ModelArtifacts,
    origin: ForecastOrigin,
    baseline: pd.Series,
    n_sim: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    draws = np.empty((n_sim, len(components)), dtype="float64")
    used_signed_saturday_conditioning = False
    for idx, component in enumerate(components):
        point = component.component_point_usd
        if component.component_type == "actual" or component.component_sigma_log == 0:
            draws[:, idx] = point
            continue
        conditional_payload = _conditional_daily_policy_payload(artifacts, origin, component.component_day)
        if (
            conditional_payload
            and origin.regime == LIVE_SUNDAY_REGIME
            and component.component_day == "Sunday"
            and component.component_type == "baseline"
        ):
            sat_actual = _positive_float(baseline.get("actual_sat_usd"))
            sat_af_forecast = _positive_float(baseline.get("after_fri_sat_usd"))
            conditioning_value = math.log(sat_actual / sat_af_forecast) if np.isfinite(sat_actual) and np.isfinite(sat_af_forecast) else math.nan
            weighted = _conditional_sunday_residual_weights(conditional_payload, conditioning_value)
            if weighted is not None:
                residuals, weights = weighted
                draws[:, idx] = point * np.exp(rng.choice(residuals, size=n_sim, replace=True, p=weights))
                used_signed_saturday_conditioning = True
                continue
        payload = _component_policy_payload(artifacts, origin, component)
        samples = payload.get("residual_samples_log") if payload else None
        if isinstance(samples, list) and samples:
            residuals = np.asarray(samples, dtype="float64")
            weights = np.asarray(payload.get("residual_sample_weights") or [], dtype="float64")
            finite = np.isfinite(residuals)
            residuals = residuals[finite]
            if weights.size == finite.size:
                weights = weights[finite]
                weights = weights / weights.sum() if weights.sum() > 0 else np.array([])
            else:
                weights = np.array([])
            if residuals.size:
                draws[:, idx] = point * np.exp(
                    rng.choice(residuals, size=n_sim, replace=True, p=weights if weights.size else None)
                )
                continue
        eps = rng.normal(0.0, component.component_sigma_log, size=n_sim)
        draws[:, idx] = point * np.exp(eps)
    totals = draws.sum(axis=1)
    return {
        "draws": totals,
        "point_usd": float(np.quantile(totals, 0.50)),
        "lo80_usd": float(np.quantile(totals, 0.10)),
        "hi80_usd": float(np.quantile(totals, 0.90)),
        "lo95_usd": float(np.quantile(totals, 0.025)),
        "hi95_usd": float(np.quantile(totals, 0.975)),
        "used_signed_saturday_conditioning": used_signed_saturday_conditioning,
    }


def _actual_is_eligible(baseline: pd.Series, origin: ForecastOrigin, day: str) -> bool:
    """Only the as-of overlay, or a historical stage, can unlock an actual."""

    actual = _positive_float(baseline.get(ACTUAL_COLUMNS[day]))
    marker = bool(baseline.get(f"{ACTUAL_COLUMNS[day]}_available_as_of", False))
    return bool(np.isfinite(actual) and (marker or day in KNOWN_ACTUAL_DAYS[origin.regime]))


def _information_state(baseline: pd.Series, origin: ForecastOrigin) -> str:
    friday = _actual_is_eligible(baseline, origin, "Friday")
    saturday = _actual_is_eligible(baseline, origin, "Saturday")
    sunday = _actual_is_eligible(baseline, origin, "Sunday")
    if friday and saturday and sunday:
        return "weekend_complete"
    if friday and saturday:
        return "after_saturday"
    if friday:
        return "after_friday"
    return "pre_weekend"


def _baseline_for_state(state: str, day: str) -> tuple[str, str]:
    """Choose the most advanced baseline justified by available actuals."""

    if state == "pre_weekend":
        return {
            "Friday": ("pre_fri_usd", "pre-weekend baseline"),
            "Saturday": ("pre_sat_usd", "pre-weekend baseline"),
            "Sunday": ("pre_sun_usd", "pre-weekend baseline"),
        }[day]
    if state == "after_friday":
        return {
            "Saturday": ("after_fri_sat_usd", "after-Friday baseline"),
            "Sunday": ("after_fri_sun_usd", "after-Friday baseline"),
        }[day]
    if state == "after_saturday":
        return ("after_sat_sun_usd", "after-Saturday baseline")
    raise ValueError(f"No baseline is valid for {day} in state {state}")


def _pre_release_origin_for_live(origin: ForecastOrigin, origin_day: int) -> ForecastOrigin:
    tzinfo = origin.forecast_origin_local.tzinfo
    live_day = origin.forecast_origin_local.date()
    if origin.origin_day is not None:
        opening_day = live_day - timedelta(days=int(origin.origin_day))
    else:
        opening_day = live_day
    local_day = opening_day + timedelta(days=origin_day)
    local_dt = datetime.combine(local_day, time(0, 0)).replace(tzinfo=tzinfo)
    utc_dt = local_dt.astimezone(origin.forecast_origin_utc.tzinfo or timezone.utc)
    return ForecastOrigin(
        regime=PRE_RELEASE_REGIME,
        origin_key=f"P_{origin_day}",
        origin_day=origin_day,
        forecast_origin=None,
        forecast_origin_local=local_dt,
        forecast_origin_utc=utc_dt,
        as_of_utc=utc_dt,
    )


def _latest_pre_release_carry_forward_result(
    *,
    movie: MovieOpening,
    origin: ForecastOrigin,
    artifacts: ModelArtifacts,
    run_id: str,
    is_live: bool,
    is_backtest: bool,
) -> ForecastResult | None:
    for origin_day in (-1, -2):
        try:
            pre_origin = _pre_release_origin_for_live(origin, origin_day)
            if pre_origin.as_of_utc > origin.as_of_utc:
                continue
            pre_results = forecast_pre_release_opening_weekend(
                movie=movie,
                origin=pre_origin,
                artifacts=artifacts,
                run_id=run_id,
                is_backtest=is_backtest,
            )
        except (KeyError, ValueError):
            continue
        pre_weekend = next((result for result in pre_results if result.target == "opening_weekend"), None)
        if pre_weekend is None:
            continue
        return ForecastResult(
            movie=movie,
            origin=origin,
            target="opening_weekend",
            point_usd=pre_weekend.point_usd,
            lo80_usd=pre_weekend.lo80_usd,
            hi80_usd=pre_weekend.hi80_usd,
            lo95_usd=pre_weekend.lo95_usd,
            hi95_usd=pre_weekend.hi95_usd,
            point_model=f"latest_pre_release_carry_forward:{pre_weekend.point_model}",
            interval_model=f"latest_pre_release_carry_forward:{pre_weekend.interval_model}",
            component_source="latest_pre_release_carry_forward",
            model_version=artifacts.model_version,
            run_id=run_id,
            components=pre_weekend.components,
            source_count=pre_weekend.source_count,
            estimate_sources=pre_weekend.estimate_sources,
            actual_usd=pre_weekend.actual_usd,
            is_live=is_live,
            is_backtest=is_backtest,
            audit={"fallback_reason": "live_friday_no_amc", "carried_origin_key": pre_origin.origin_key},
        )
    return None


def _requires_friday_no_amc_carry_forward(origin: ForecastOrigin, components: list[DailyComponent]) -> bool:
    if origin.regime != LIVE_FRIDAY_REGIME:
        return False
    has_amc_component = any(component.component_type == "AMC_nowcast" for component in components)
    has_friday_actual = any(
        component.component_day == "Friday" and component.component_type == "actual"
        for component in components
    )
    return not has_amc_component and not has_friday_actual


def build_live_components(
    *,
    movie: MovieOpening,
    origin: ForecastOrigin,
    artifacts: ModelArtifacts,
) -> list[DailyComponent]:
    baseline, _ = _select_adjusted_daily_baseline_row(movie=movie, origin=origin, artifacts=artifacts)
    return _build_live_components_from_baseline(
        baseline=baseline,
        movie=movie,
        origin=origin,
        artifacts=artifacts,
    )


def _build_live_components_from_baseline(
    *,
    baseline: pd.Series,
    movie: MovieOpening,
    origin: ForecastOrigin,
    artifacts: ModelArtifacts,
) -> list[DailyComponent]:
    components = []
    state = _information_state(baseline, origin)
    for day in DAYS:
        actual_point = _positive_float(baseline.get(ACTUAL_COLUMNS[day]))
        if _actual_is_eligible(baseline, origin, day):
            components.append(
                DailyComponent(
                    component_day=day,
                    component_type="actual",
                    component_point_usd=actual_point,
                    component_sigma_log=0.0,
                    component_model="actual",
                    component_source="actual",
                )
            )
            continue
        plugin = (
            select_plugin_row(artifacts.live_plugin_nowcasts, movie, origin, day)
            if PLUGIN_TARGET_DAY[origin.regime] == day
            else None
        )
        if plugin is not None:
            point = _positive_float(plugin.get("pred_daily_gross_usd"))
            sigma = _positive_float(plugin.get("sigma_log_daily"))
            fallback_sigma = _fallback_sigma(artifacts, origin.regime, day)
            draft_component = DailyComponent(
                component_day=day,
                component_type="AMC_nowcast",
                component_point_usd=point,
                component_sigma_log=sigma if np.isfinite(sigma) else fallback_sigma,
                component_model=str(plugin.get("model", "AMC_plugin")),
                component_source=str(plugin.get("source", "AMC_plugin")),
                component_notes=str(plugin.get("feature_quality_bucket", "")) or None,
            )
            components.append(
                _component_from_payload(
                    day=day,
                    component_type="AMC_nowcast",
                    point=point,
                    model=str(plugin.get("model", "AMC_plugin")),
                    source=str(plugin.get("source", "AMC_plugin")),
                    payload=_component_policy_payload(artifacts, origin, draft_component),
                    fallback_sigma=sigma if np.isfinite(sigma) else fallback_sigma,
                    notes=str(plugin.get("feature_quality_bucket", "")) or None,
                )
            )
            continue

        # If the prerequisite actual is missing, stay in the last safe state
        # instead of leaking an after-Friday/after-Saturday baseline.
        column, source = _baseline_for_state(state, day)
        point = _positive_float(baseline.get(column))
        if not np.isfinite(point):
            # An artifact without the conditional baseline cannot safely
            # synthesize it.  Fall back to the still-valid pre-weekend daily
            # component rather than emit an invalid CDF.
            column, source = _baseline_for_state("pre_weekend", day)
            point = _positive_float(baseline.get(column))
            source = f"{source} (conditional baseline unavailable)"
        components.append(
            _component_from_payload(
                day=day,
                component_type="baseline",
                point=point,
                model=source,
                source="daily_baseline",
                payload=_daily_policy_payload(artifacts, origin.regime, day),
                fallback_sigma=_fallback_sigma(artifacts, origin.regime, day),
                notes=None,
            )
        )
    return components


def compose_live_weekend_forecast(
    *,
    movie: MovieOpening,
    origin: ForecastOrigin,
    artifacts: ModelArtifacts,
    run_id: str,
    is_live: bool,
    is_backtest: bool,
    seed: int = 17,
    market_grids: dict[str, tuple[int, int, int, int]] | None = None,
) -> list[ForecastResult]:
    baseline, preview_audit = _select_adjusted_daily_baseline_row(movie=movie, origin=origin, artifacts=artifacts)
    components = _build_live_components_from_baseline(
        baseline=baseline,
        movie=movie,
        origin=origin,
        artifacts=artifacts,
    )
    has_amc_component = any(c.component_type == "AMC_nowcast" for c in components)
    information_state = _information_state(baseline, origin)
    configured_policy = getattr(artifacts, "live_weekend_distribution_policy", {}) or {}
    # Preserve legacy behavior for old artifact versions.  Every promoted v1
    # live-CDF artifact supplies the policy and therefore uses component-level
    # fallback instead of carrying a pre-release interval.
    requires_carry_forward = (
        not configured_policy
        and _requires_friday_no_amc_carry_forward(origin, components)
        and not bool(preview_audit.get("thursday_preview_update_applied"))
        and not bool(preview_audit.get("opening_thursday_actual_update_applied"))
    )
    distribution_policy = resolved_policy(configured_policy)
    n_sim = int(
        distribution_policy["simulation_draw_count"]
        if configured_policy
        else artifacts.manifest.get("composition", {}).get("n_sim", distribution_policy["simulation_draw_count"])
    )
    if not configured_policy:
        distribution_policy["simulation_draw_count"] = n_sim
    simulation_seed = deterministic_seed(
        movie_id=movie.movie_id,
        release_run_id=movie.release_run_id,
        information_state=information_state,
        forecast_origin=origin.forecast_origin,
        policy_version=str(distribution_policy["policy_version"]),
    )
    if artifacts.daily_interval_policy or artifacts.amc_interval_policy:
        sim = _simulate_component_sum_from_policies(
            components,
            artifacts=artifacts,
            origin=origin,
            baseline=baseline,
            n_sim=n_sim,
            seed=simulation_seed,
        )
        interval_model = (
            "amc_empirical_component_residual_simulation"
            if any(c.component_type == "AMC_nowcast" and _amc_policy_payload(artifacts, origin, c) for c in components)
            else "signed_saturday_conditioned_sunday_residual_simulation"
            if sim.get("used_signed_saturday_conditioning")
            else "empirical_component_residual_simulation"
        )
    else:
        rng = np.random.default_rng(simulation_seed)
        draws = np.zeros(n_sim, dtype="float64")
        for component in components:
            if component.component_type == "actual" or component.component_sigma_log == 0:
                draws += component.component_point_usd
            else:
                draws += component.component_point_usd * np.exp(rng.normal(0.0, component.component_sigma_log, n_sim))
        sim = {"draws": draws, "used_signed_saturday_conditioning": False}
        interval_model = "component_log_error_simulation"
    draws = np.asarray(sim["draws"], dtype="float64")
    # A Thursday preview update defines the updated pre-weekend OW prior.  In
    # its no-actual/no-AMC state, retain its locked median exactly while leaving
    # the daily residual shape intact.
    preview_target = _positive_float(preview_audit.get("ow_prior_usd"))
    if (
        information_state == "pre_weekend"
        and not has_amc_component
        and bool(preview_audit.get("thursday_preview_update_applied") or preview_audit.get("opening_thursday_actual_update_applied"))
        and np.isfinite(preview_target)
    ):
        current_median = float(np.quantile(draws, 0.5))
        if current_median > 0:
            adjustment = preview_target / current_median
            draws = draws * adjustment
            preview_audit["preview_distribution_median_alignment"] = adjustment
    sim.update(
        {
            "point_usd": float(round(np.quantile(draws, 0.50))),
            "lo80_usd": float(round(np.quantile(draws, 0.10))),
            "hi80_usd": float(round(np.quantile(draws, 0.90))),
            "lo95_usd": float(round(np.quantile(draws, 0.025))),
            "hi95_usd": float(round(np.quantile(draws, 0.975))),
        }
    )
    actual_ow = _positive_float(baseline.get("actual_ow_usd"))
    component_source = (
        "AMC_plugin"
        if has_amc_component
        else "opening_thursday_actual_ratio_update_prod"
        if preview_audit.get("opening_thursday_actual_update_applied")
        else str(preview_audit.get("ow_prior_source"))
        if preview_audit.get("thursday_preview_update_applied")
        else "daily_baseline"
    )
    carry_forward = (
        _latest_pre_release_carry_forward_result(
            movie=movie,
            origin=origin,
            run_id=run_id,
            artifacts=artifacts,
            is_live=is_live,
            is_backtest=is_backtest,
        )
        if requires_carry_forward
        else None
    )
    if requires_carry_forward and carry_forward is None:
        raise ValueError(
            f"Friday no-AMC opening-weekend fallback requires eligible D-1 or D-2 pre-release forecast "
            f"for release_run_id={movie.release_run_id}"
        )
    actual_provenance = {
        component.component_day: {
            "actual_source": baseline.get(f"{ACTUAL_COLUMNS[component.component_day]}_source", "daily_box_office"),
            "actual_record_id": baseline.get(f"{ACTUAL_COLUMNS[component.component_day]}_record_id"),
            "actual_value": component.component_point_usd,
            "actual_published_at": baseline.get(f"{ACTUAL_COLUMNS[component.component_day]}_published_at"),
            "actual_ingested_at": baseline.get(f"{ACTUAL_COLUMNS[component.component_day]}_ingested_at"),
            "actual_revision": baseline.get(f"{ACTUAL_COLUMNS[component.component_day]}_revision"),
        }
        for component in components
        if component.component_type == "actual"
    }
    component_sources = {component.component_day: component.component_source for component in components}
    amc = next((component for component in components if component.component_type == "AMC_nowcast"), None)
    payload = build_payload(
        draws=draws,
        policy=distribution_policy,
        information_state=information_state,
        forecast_origin=origin.forecast_origin,
        seed=simulation_seed,
        component_sources=component_sources,
        known_actual_days=list(actual_provenance),
        actual_provenance=actual_provenance,
        amc_component=(
            {"day": amc.component_day, "model": amc.component_model, "source": amc.component_source, "point_usd": amc.component_point_usd}
            if amc else None
        ),
        baseline_components=[component.component_day for component in components if component.component_type == "baseline"],
        preview_update_metadata={key: preview_audit.get(key) for key in [
            "ow_prior_source", "ow_prior_baseline_usd", "ow_prior_usd", "thursday_preview_update_applied",
            "opening_thursday_actual_update_applied", "preview_distribution_median_alignment",
        ]},
        market_grids=market_grids,
    )
    opening_weekend_result = carry_forward or ForecastResult(
        movie=movie,
        origin=origin,
        target="opening_weekend",
        point_usd=sim["point_usd"],
        lo80_usd=sim["lo80_usd"],
        hi80_usd=sim["hi80_usd"],
        lo95_usd=sim["lo95_usd"],
        hi95_usd=sim["hi95_usd"],
        point_model="live_weekend_simulated_median",
        interval_model=interval_model,
        component_source=component_source,
        model_version=artifacts.model_version,
        run_id=run_id,
        distribution_payload=payload,
        components=components,
        actual_usd=actual_ow if np.isfinite(actual_ow) else None,
        is_live=is_live,
        is_backtest=is_backtest,
        audit={**preview_audit, "information_state": information_state, "deterministic_seed": simulation_seed},
    )
    results = [opening_weekend_result]

    for component in components:
        component_with_interval = component.with_intervals()
        actual = _positive_float(baseline.get(ACTUAL_COLUMNS[component.component_day]))
        results.append(
            ForecastResult(
                movie=movie,
                origin=origin,
                target=component.component_day.lower(),
                point_usd=component_with_interval.component_point_usd,
                lo80_usd=component_with_interval.component_lo80_usd,
                hi80_usd=component_with_interval.component_hi80_usd,
                lo95_usd=component_with_interval.component_lo95_usd,
                hi95_usd=component_with_interval.component_hi95_usd,
                point_model=component.component_model,
                interval_model=_component_interval_model(artifacts, origin, component),
                component_source=component.component_source,
                model_version=artifacts.model_version,
                run_id=run_id,
                components=[component],
                actual_usd=actual if np.isfinite(actual) else None,
                is_live=is_live,
                is_backtest=is_backtest,
            )
        )
    return results
