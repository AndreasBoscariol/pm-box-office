"""Prediction interval helpers."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

Z80 = 1.28155
Z95 = 1.95996


def log_normal_interval(point_usd: float | None, sigma_log: float | None) -> dict[str, float]:
    """Return 80% and 95% log-error intervals around a positive point forecast."""

    if point_usd is None or sigma_log is None:
        return {"lo80_usd": math.nan, "hi80_usd": math.nan, "lo95_usd": math.nan, "hi95_usd": math.nan}
    point = float(point_usd)
    sigma = float(sigma_log)
    if not math.isfinite(point) or point <= 0 or not math.isfinite(sigma) or sigma < 0:
        return {"lo80_usd": math.nan, "hi80_usd": math.nan, "lo95_usd": math.nan, "hi95_usd": math.nan}
    return {
        "lo80_usd": point * math.exp(-Z80 * sigma),
        "hi80_usd": point * math.exp(Z80 * sigma),
        "lo95_usd": point * math.exp(-Z95 * sigma),
        "hi95_usd": point * math.exp(Z95 * sigma),
    }


def simulate_component_sum(
    points: Iterable[float],
    sigmas: Iterable[float],
    *,
    covariance: np.ndarray | None = None,
    n_sim: int = 50_000,
    seed: int = 17,
) -> dict[str, float]:
    """Simulate a weekend total from daily log-error components."""

    point_array = np.asarray(list(points), dtype="float64")
    sigma_array = np.asarray(list(sigmas), dtype="float64")
    if point_array.size == 0 or point_array.size != sigma_array.size:
        raise ValueError("points and sigmas must have the same non-zero length")
    if np.any(~np.isfinite(point_array)) or np.any(point_array < 0):
        return {"point_usd": math.nan, "lo80_usd": math.nan, "hi80_usd": math.nan, "lo95_usd": math.nan, "hi95_usd": math.nan}
    if np.any(~np.isfinite(sigma_array)) or np.any(sigma_array < 0):
        return {"point_usd": math.nan, "lo80_usd": math.nan, "hi80_usd": math.nan, "lo95_usd": math.nan, "hi95_usd": math.nan}

    rng = np.random.default_rng(seed)
    stochastic = sigma_array > 0
    draws = np.tile(point_array, (n_sim, 1))
    if stochastic.any():
        if covariance is not None:
            cov = np.asarray(covariance, dtype="float64")
            if cov.shape != (point_array.size, point_array.size):
                raise ValueError("covariance shape must match the number of components")
            eps = rng.multivariate_normal(np.zeros(point_array.size), cov, size=n_sim)
        else:
            eps = rng.normal(0.0, sigma_array, size=(n_sim, point_array.size))
        draws[:, stochastic] = point_array[stochastic] * np.exp(eps[:, stochastic])

    totals = draws.sum(axis=1)
    return {
        "point_usd": float(np.quantile(totals, 0.50)),
        "lo80_usd": float(np.quantile(totals, 0.10)),
        "hi80_usd": float(np.quantile(totals, 0.90)),
        "lo95_usd": float(np.quantile(totals, 0.025)),
        "hi95_usd": float(np.quantile(totals, 0.975)),
    }

