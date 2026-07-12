"""Production tail-safety wrapper for pre-release box-office distributions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t


TAIL_CONTAMINATION_WEIGHT = 0.02
TAIL_REFERENCE_DF = 4


@dataclass(frozen=True)
class TailSafeDistribution:
    """A coherent CDF mixture preserving the shared point median."""

    base_quantiles: np.ndarray
    quantile_levels: np.ndarray
    point_forecast_usd: float
    tail_log_scale: float
    contamination_weight: float = TAIL_CONTAMINATION_WEIGHT
    reference_df: int = TAIL_REFERENCE_DF

    def __post_init__(self) -> None:
        q = np.asarray(self.base_quantiles, dtype=float)
        levels = np.asarray(self.quantile_levels, dtype=float)
        if len(q) != len(levels) or len(q) < 3:
            raise ValueError("base quantiles and levels must have equal nontrivial length")
        if np.any(~np.isfinite(q)) or np.any(q < 0) or np.any(np.diff(q) < 0):
            raise ValueError("base dollar quantiles must be finite, nonnegative, and nondecreasing")
        if np.any(np.diff(levels) <= 0) or levels[0] <= 0 or levels[-1] >= 1:
            raise ValueError("quantile levels must be strictly increasing inside (0, 1)")
        if self.point_forecast_usd <= 0 or self.tail_log_scale <= 0:
            raise ValueError("point and tail scale must be positive")
        if not 0 <= self.contamination_weight <= 1:
            raise ValueError("contamination weight must be in [0, 1]")

    @property
    def reference_quantiles(self) -> np.ndarray:
        return self.point_forecast_usd * np.exp(
            student_t.ppf(self.quantile_levels, self.reference_df) * self.tail_log_scale
        )

    def base_cdf(self, values: np.ndarray | float) -> np.ndarray:
        return np.interp(values, self.base_quantiles, self.quantile_levels, left=0.0, right=1.0)

    def reference_cdf(self, values: np.ndarray | float) -> np.ndarray:
        return np.interp(values, self.reference_quantiles, self.quantile_levels, left=0.0, right=1.0)

    @property
    def final_quantiles(self) -> np.ndarray:
        if self.contamination_weight == 0:
            return self.base_quantiles.copy()
        support = np.unique(np.concatenate([self.base_quantiles, self.reference_quantiles]))
        weight = self.contamination_weight
        mapped = (1.0 - weight) * self.base_cdf(support) + weight * self.reference_cdf(support)
        mapped = np.maximum.accumulate(mapped)
        return np.interp(self.quantile_levels, mapped, support, left=support[0], right=support[-1])

    def cdf(self, values: np.ndarray | float) -> np.ndarray:
        if self.contamination_weight == 0:
            return self.base_cdf(values)
        return np.interp(values, self.final_quantiles, self.quantile_levels, left=0.0, right=1.0)

    def ppf(self, probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=float)
        return np.interp(probabilities, self.quantile_levels, self.final_quantiles)

    def bucket_probabilities(self, edges: np.ndarray) -> np.ndarray:
        edges = np.asarray(edges, dtype=float)
        cdf = np.asarray(self.cdf(edges), dtype=float)
        cdf[0], cdf[-1] = 0.0, 1.0
        probs = np.diff(np.maximum.accumulate(cdf))
        probs[np.abs(probs) < 1e-15] = 0.0
        return probs / probs.sum()


def audited_tail_log_scale(prior_log_residuals: np.ndarray, *, df: int = TAIL_REFERENCE_DF) -> float:
    """Exact scale rule frozen by the successful locked audit."""
    residuals = np.asarray(prior_log_residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    raw = float(np.median(np.abs(residuals)) / student_t.ppf(0.75, df)) if len(residuals) >= 30 else 0.35
    return max(raw, 0.03)
