"""Versioned payload helpers for production live opening-weekend CDFs."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import numpy as np


DEFAULT_POLICY: dict[str, Any] = {
    "policy_name": "production_live_weekend_cdf_v1",
    "policy_version": "v1",
    "distribution_schema_version": "live_ow_cdf_v1",
    "simulation_draw_count": 50_000,
    "quantile_grid": {"count": 201, "method": "interior_even"},
    "component_dependence_policy": "independent_component_residual_draws",
    "actual_source_policy": "daily_box_office_non_estimate_as_of",
    "AMC_eligibility_policy": "eligible_current_day_only",
    "AMC_uncertainty_floor_policy": "existing_amc_and_daily_interval_policy",
    "preview_update_policy": "frozen_thursday_preview_updater",
    "missing_actual_fallback_policy": "latest_leakage_safe_state",
    "missing_AMC_fallback_policy": "state_appropriate_baseline",
    "live_tail_policy": "disabled",
    "training_cutoff": None,
}


def resolved_policy(policy: dict[str, Any] | None) -> dict[str, Any]:
    """Fill an artifact policy with the locked v1 production defaults."""

    return {**DEFAULT_POLICY, **(policy or {})}


def quantile_levels(policy: dict[str, Any]) -> np.ndarray:
    count = int((policy.get("quantile_grid") or {}).get("count", 201))
    if count < 101:
        raise ValueError("live weekend distribution requires at least 101 quantiles")
    levels = np.arange(1, count + 1, dtype="float64") / float(count + 1)
    # Keep the standard reporting quantiles in the persisted curve so emitted
    # points and intervals can be read back exactly from the same payload.
    for target in (0.025, 0.10, 0.50, 0.90, 0.975):
        index = int(np.argmin(np.abs(levels - target)))
        if (index == 0 or levels[index - 1] < target) and (index == len(levels) - 1 or target < levels[index + 1]):
            levels[index] = target
    return levels


def deterministic_seed(*, movie_id: int, release_run_id: int, information_state: str, forecast_origin: str | None, policy_version: str) -> int:
    raw = "|".join([str(movie_id), str(release_run_id), information_state, str(forecast_origin or ""), policy_version])
    return int.from_bytes(hashlib.sha256(raw.encode("utf-8")).digest()[:8], "big") % (2**32)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def build_payload(
    *,
    draws: Iterable[float],
    policy: dict[str, Any],
    information_state: str,
    forecast_origin: str | None,
    seed: int,
    component_sources: dict[str, str],
    known_actual_days: list[str],
    actual_provenance: dict[str, dict[str, Any]],
    amc_component: dict[str, Any] | None,
    baseline_components: list[str],
    preview_update_metadata: dict[str, Any],
    market_grids: dict[str, tuple[int, int, int, int]] | None = None,
) -> dict[str, Any]:
    values = np.asarray(list(draws), dtype="float64")
    if values.size != int(policy["simulation_draw_count"]) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("live distribution draws do not satisfy the locked policy")
    levels = quantile_levels(policy)
    quantiles = np.maximum.accumulate(np.rint(np.quantile(values, levels)).astype("int64"))
    bucket_probabilities: dict[str, list[float]] = {}
    for grid_id, boundaries in (market_grids or {}).items():
        if len(boundaries) != 4 or any(left >= right for left, right in zip(boundaries, boundaries[1:])):
            continue
        b0, b1, b2, b3 = boundaries
        probabilities = [
            float(np.mean(values < b0)),
            float(np.mean((values >= b0) & (values < b1))),
            float(np.mean((values >= b1) & (values < b2))),
            float(np.mean((values >= b2) & (values < b3))),
            float(np.mean(values >= b3)),
        ]
        bucket_probabilities[str(grid_id)] = probabilities
        bucket_probabilities["boundaries:" + ",".join(str(int(value)) for value in boundaries)] = probabilities
    payload: dict[str, Any] = {
        "policy_name": str(policy["policy_name"]),
        "distribution_policy": str(policy["policy_name"]),
        "policy_version": str(policy["policy_version"]),
        "distribution_policy_version": str(policy["policy_version"]),
        "distribution_schema_version": str(policy["distribution_schema_version"]),
        "information_state": information_state,
        "forecast_origin": forecast_origin,
        "point_forecast_usd": float(np.quantile(values, 0.5)),
        "quantile_levels": levels.tolist(),
        "quantile_values_usd": quantiles.tolist(),
        "market_bucket_probabilities": bucket_probabilities,
        "simulation_draw_count": int(values.size),
        "component_sources": component_sources,
        "known_actual_days": known_actual_days,
        "actual_provenance": actual_provenance,
        "AMC_component": amc_component,
        "baseline_components": baseline_components,
        "preview_update_metadata": preview_update_metadata,
        "training_cutoff": policy.get("training_cutoff"),
        "deterministic_seed": int(seed),
        "tail_policy": {"name": str(policy["live_tail_policy"])},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_jsonable)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    payload["payload_hash"] = digest
    # ForecastDistribution historically calls this an emission hash; retain it
    # while the payload schema uses the clearer payload_hash name.
    payload["distribution_emission_hash"] = digest
    return payload
