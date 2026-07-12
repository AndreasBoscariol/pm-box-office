"""Full, deterministic CDFs for production pre-release forecasts."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
from scipy.stats import norm

from .distribution_tail_safety import TailSafeDistribution
from .live_weekend_distribution import quantile_levels


def _hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _base_sigma(*, point_usd: float, lo80_usd: float, hi80_usd: float) -> float:
    """Infer the log-normal base scale from the persisted central interval."""
    z80 = float(norm.ppf(0.90))
    if point_usd <= 0 or lo80_usd <= 0 or hi80_usd <= lo80_usd:
        raise ValueError("a positive, ordered 80% interval is required for a pre-release CDF")
    candidates = [
        math.log(hi80_usd / point_usd) / z80,
        math.log(point_usd / lo80_usd) / z80,
        math.log(hi80_usd / lo80_usd) / (2.0 * z80),
    ]
    usable = [value for value in candidates if math.isfinite(value) and value > 0]
    if not usable:
        raise ValueError("unable to infer pre-release CDF scale")
    return max(float(np.median(usable)), 0.03)


def build_pre_release_distribution_payload(
    *,
    point_usd: float,
    lo80_usd: float,
    hi80_usd: float,
    distribution_policy: dict[str, Any] | None,
    forecast_origin: str | None,
    origin_key: str,
    model_version: str,
    information_state: str = "pre_release",
) -> dict[str, Any]:
    """Build a UI-consumable CDF from the promoted pre-release interval policy.

    The base curve is the log-normal interval curve already used for the
    forecast.  When the promoted policy enables it, its fixed Student-t tail
    contamination layer is applied without changing the forecast's centre.
    """
    policy = distribution_policy or {}
    grid_policy = {"quantile_grid": {"count": 201}}
    levels = quantile_levels(grid_policy)
    sigma = _base_sigma(point_usd=point_usd, lo80_usd=lo80_usd, hi80_usd=hi80_usd)
    base = point_usd * np.exp(norm.ppf(levels) * sigma)
    contamination = float(policy.get("tail_contamination_weight", 0.0)) if policy.get("tail_safety_enabled") else 0.0
    df = int(policy.get("tail_reference_df", 4))
    tail_scale = max(float(policy.get("tail_log_scale", sigma)), 0.03)
    curve = TailSafeDistribution(
        base_quantiles=base,
        quantile_levels=levels,
        point_forecast_usd=point_usd,
        tail_log_scale=tail_scale,
        contamination_weight=contamination,
        reference_df=df,
    ).final_quantiles
    values = np.maximum.accumulate(np.maximum(np.rint(curve), 0).astype("int64"))
    policy_name = str(policy.get("policy_name") or "pre_release_interval_cdf_v1")
    policy_version = str(policy.get("policy_version") or policy.get("base_distribution_policy") or "v1")
    payload: dict[str, Any] = {
        "policy_name": policy_name,
        "distribution_policy": policy_name,
        "policy_version": policy_version,
        "distribution_policy_version": policy_version,
        "distribution_schema_version": "pre_release_ow_cdf_v1",
        "information_state": information_state,
        "forecast_origin": forecast_origin,
        "origin_key": origin_key,
        "model_version": model_version,
        "point_forecast_usd": float(point_usd),
        "quantile_levels": levels.tolist(),
        "quantile_values_usd": values.tolist(),
        "market_bucket_probabilities": {},
        "tail_policy": {
            "name": "student_t_contamination" if contamination else "disabled",
            "contamination_weight": contamination,
            "reference_df": df if contamination else None,
            "log_scale": tail_scale if contamination else None,
        },
        "interval_calibration": {"base_family": "log_normal", "sigma_log": sigma},
    }
    digest = _hash(payload)
    payload["payload_hash"] = digest
    payload["distribution_emission_hash"] = digest
    return payload
