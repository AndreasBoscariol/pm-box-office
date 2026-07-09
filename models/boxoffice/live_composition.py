"""Live weekend composition from daily baselines and optional AMC nowcasts."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import ModelArtifacts
from .constants import ACTUAL_COLUMNS, BASELINE_COLUMNS, DAYS, KNOWN_ACTUAL_DAYS, PLUGIN_TARGET_DAY, TARGET_TO_DAY
from .intervals import log_normal_interval, simulate_component_sum
from .schema import DailyComponent, ForecastOrigin, ForecastResult, MovieOpening


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
    n_sim: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    draws = np.empty((n_sim, len(components)), dtype="float64")
    for idx, component in enumerate(components):
        point = component.component_point_usd
        if component.component_type == "actual" or component.component_sigma_log == 0:
            draws[:, idx] = point
            continue
        payload = _daily_policy_payload(artifacts, origin.regime, component.component_day)
        samples = payload.get("residual_samples_log") if payload else None
        if isinstance(samples, list) and samples:
            residuals = np.asarray(samples, dtype="float64")
            residuals = residuals[np.isfinite(residuals)]
            if residuals.size:
                draws[:, idx] = point * np.exp(rng.choice(residuals, size=n_sim, replace=True))
                continue
        eps = rng.normal(0.0, component.component_sigma_log, size=n_sim)
        draws[:, idx] = point * np.exp(eps)
    totals = draws.sum(axis=1)
    return {
        "point_usd": float(np.quantile(totals, 0.50)),
        "lo80_usd": float(np.quantile(totals, 0.10)),
        "hi80_usd": float(np.quantile(totals, 0.90)),
        "lo95_usd": float(np.quantile(totals, 0.025)),
        "hi95_usd": float(np.quantile(totals, 0.975)),
    }


def build_live_components(
    *,
    movie: MovieOpening,
    origin: ForecastOrigin,
    artifacts: ModelArtifacts,
) -> list[DailyComponent]:
    baseline = select_daily_baseline_row(artifacts.daily_baseline, movie)
    components = []
    for day in DAYS:
        if day in KNOWN_ACTUAL_DAYS[origin.regime]:
            point = _positive_float(baseline.get(ACTUAL_COLUMNS[day]))
            components.append(
                DailyComponent(
                    component_day=day,
                    component_type="actual",
                    component_point_usd=point,
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
            components.append(
                _component_from_payload(
                    day=day,
                    component_type="AMC_nowcast",
                    point=point,
                    model=str(plugin.get("model", "AMC_plugin")),
                    source=str(plugin.get("source", "AMC_plugin")),
                    payload=_daily_policy_payload(artifacts, origin.regime, day),
                    fallback_sigma=sigma if np.isfinite(sigma) else fallback_sigma,
                    notes=str(plugin.get("feature_quality_bucket", "")) or None,
                )
            )
            continue

        column, source = BASELINE_COLUMNS[(origin.regime, day)]
        point = _positive_float(baseline.get(column))
        components.append(
            _component_from_payload(
                day=day,
                component_type="baseline",
                point=point,
                model=source,
                source="daily_baseline",
                payload=_daily_policy_payload(artifacts, origin.regime, day),
                fallback_sigma=_fallback_sigma(artifacts, origin.regime, day),
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
) -> list[ForecastResult]:
    components = build_live_components(movie=movie, origin=origin, artifacts=artifacts)
    n_sim = int(artifacts.manifest.get("composition", {}).get("n_sim", 50_000))
    if artifacts.daily_interval_policy:
        sim = _simulate_component_sum_from_policies(
            components,
            artifacts=artifacts,
            origin=origin,
            n_sim=n_sim,
            seed=seed,
        )
        interval_model = "empirical_component_residual_simulation"
    else:
        sim = simulate_component_sum(
            [component.component_point_usd for component in components],
            [component.component_sigma_log for component in components],
            n_sim=n_sim,
            seed=seed,
        )
        interval_model = "component_log_error_simulation"
    baseline = select_daily_baseline_row(artifacts.daily_baseline, movie)
    actual_ow = _positive_float(baseline.get("actual_ow_usd"))
    component_source = "AMC_plugin" if any(c.component_type == "AMC_nowcast" for c in components) else "daily_baseline"
    results = [
        ForecastResult(
            movie=movie,
            origin=origin,
            target="opening_weekend",
            point_usd=sim["point_usd"],
            lo80_usd=sim["lo80_usd"],
            hi80_usd=sim["hi80_usd"],
            lo95_usd=sim["lo95_usd"],
            hi95_usd=sim["hi95_usd"],
            point_model="live_weekend_component_composer",
            interval_model=interval_model,
            component_source=component_source,
            model_version=artifacts.model_version,
            run_id=run_id,
            components=components,
            actual_usd=actual_ow if np.isfinite(actual_ow) else None,
            is_live=is_live,
            is_backtest=is_backtest,
        )
    ]

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
                interval_model="empirical_daily_quantile" if artifacts.daily_interval_policy and component.component_type != "actual" else "daily_log_sigma",
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
