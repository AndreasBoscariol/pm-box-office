"""Persisted full forecast-distribution payloads used by the forecast UI."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Sequence


FULL_DISTRIBUTION_MIN_QUANTILES = 101


@dataclass(frozen=True)
class ForecastDistribution:
    quantile_levels: tuple[float, ...]
    quantile_values_usd: tuple[int, ...]
    distribution_policy: str
    distribution_policy_version: str
    distribution_emission_hash: str
    tail_policy: dict[str, Any]
    direct_bucket_probabilities: dict[str, list[float]]

    @classmethod
    def from_payload(cls, payload: object) -> "ForecastDistribution | None":
        if not isinstance(payload, dict):
            return None
        raw_levels = payload.get("quantile_levels")
        raw_values = payload.get("quantile_values_usd")
        if not isinstance(raw_levels, Sequence) or not isinstance(raw_values, Sequence):
            return None
        if len(raw_levels) < FULL_DISTRIBUTION_MIN_QUANTILES or len(raw_levels) != len(raw_values):
            return None
        try:
            levels = tuple(float(value) for value in raw_levels)
            values = tuple(int(round(float(value))) for value in raw_values)
        except (TypeError, ValueError):
            return None
        if (
            levels[0] <= 0
            or levels[-1] >= 1
            or any(left >= right for left, right in zip(levels, levels[1:]))
            or any(value < 0 for value in values)
            or any(left > right for left, right in zip(values, values[1:]))
        ):
            return None
        policy = str(payload.get("distribution_policy") or "")
        version = str(payload.get("distribution_policy_version") or "")
        emission_hash = str(payload.get("distribution_emission_hash") or payload.get("payload_hash") or "")
        if not policy or not version or not emission_hash:
            return None
        tail_policy = payload.get("tail_policy")
        direct = payload.get("market_bucket_probabilities")
        direct = direct if isinstance(direct, dict) else {}
        return cls(levels, values, policy, version, emission_hash, tail_policy if isinstance(tail_policy, dict) else {}, direct)

    def cdf(self, value_usd: int | float) -> float:
        value = float(value_usd)
        if value < self.quantile_values_usd[0]:
            return 0.0
        if value >= self.quantile_values_usd[-1]:
            return 1.0
        index = bisect_left(self.quantile_values_usd, value)
        if index == 0:
            return self.quantile_levels[0]
        left_value, right_value = self.quantile_values_usd[index - 1], self.quantile_values_usd[index]
        left_level, right_level = self.quantile_levels[index - 1], self.quantile_levels[index]
        if right_value == left_value:
            return right_level
        proportion = (value - left_value) / (right_value - left_value)
        return max(0.0, min(1.0, left_level + proportion * (right_level - left_level)))

    def quantile(self, probability: float) -> int:
        probability = float(probability)
        if probability <= 0:
            return self.quantile_values_usd[0]
        if probability >= 1:
            return self.quantile_values_usd[-1]
        index = bisect_left(self.quantile_levels, probability)
        if index == 0:
            return self.quantile_values_usd[0]
        left_level, right_level = self.quantile_levels[index - 1], self.quantile_levels[index]
        left_value, right_value = self.quantile_values_usd[index - 1], self.quantile_values_usd[index]
        proportion = (probability - left_level) / (right_level - left_level)
        return int(round(left_value + proportion * (right_value - left_value)))

    def bucket_probabilities(self, boundaries_usd: Sequence[int], *, grid_id: str | None = None) -> list[float]:
        if len(boundaries_usd) != 4 or any(left >= right for left, right in zip(boundaries_usd, boundaries_usd[1:])):
            raise ValueError("a five-bucket market requires four increasing boundaries")
        if grid_id and grid_id in self.direct_bucket_probabilities:
            probabilities = self.direct_bucket_probabilities[grid_id]
            if len(probabilities) == 5 and all(value >= 0 for value in probabilities) and sum(probabilities) > 0:
                return [float(value) / float(sum(probabilities)) for value in probabilities]
        boundary_key = "boundaries:" + ",".join(str(int(value)) for value in boundaries_usd)
        if boundary_key in self.direct_bucket_probabilities:
            probabilities = self.direct_bucket_probabilities[boundary_key]
            if len(probabilities) == 5 and all(value >= 0 for value in probabilities) and sum(probabilities) > 0:
                return [float(value) / float(sum(probabilities)) for value in probabilities]
        # A completed weekend is an atom.  Quantile interpolation otherwise
        # represents a continuous approximation, but an atom must honour the
        # market's [lower, upper) convention exactly.
        if len(set(self.quantile_values_usd)) == 1:
            value = self.quantile_values_usd[0]
            index = sum(value >= boundary for boundary in boundaries_usd)
            return [1.0 if bucket == index else 0.0 for bucket in range(5)]
        cdf_values = [self.cdf(boundary) for boundary in boundaries_usd]
        probabilities = [cdf_values[0], *[right - left for left, right in zip(cdf_values, cdf_values[1:])], 1.0 - cdf_values[-1]]
        probabilities = [max(0.0, value) for value in probabilities]
        total = sum(probabilities)
        if total <= 0:
            raise ValueError("forecast distribution has no probability mass")
        return [value / total for value in probabilities]
