"""Coherent forecast and market probability construction."""

from __future__ import annotations

from decimal import Decimal
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .semantics import Bucket, validate_bucket_set


@dataclass(frozen=True, slots=True)
class BucketProbability:
    market_id: str
    probability: float
    count: int
    standard_error: float
    material_monte_carlo_error: bool


def bucket_probabilities(draws: Iterable[Decimal | int | float], buckets: Sequence[Bucket]) -> list[float]:
    validation = validate_bucket_set(buckets)
    if not validation.valid:
        raise ValueError("invalid bucket set: " + "; ".join(validation.errors))
    values = [Decimal(str(value)) for value in draws]
    if not values:
        raise ValueError("forecast draws are required")
    counts = [sum(bucket.contains(value) for value in values) for bucket in buckets]
    if sum(counts) != len(values):
        raise ValueError("draws are not classified exactly once")
    result = [count / len(values) for count in counts]
    if abs(sum(result) - 1.0) > 1e-12:
        raise ValueError("probability vector is incoherent")
    return result


def bucket_probability_details(draws: Iterable[Decimal | int | float], buckets: Sequence[Bucket], *,
                               intended_edge: float = .02, material_fraction: float = .1) -> list[BucketProbability]:
    values = list(draws)
    probabilities = bucket_probabilities(values, buckets)
    size = len(values)
    details = []
    for bucket, probability in zip(buckets, probabilities):
        count = round(probability * size)
        standard_error = float(np.sqrt(probability * (1 - probability) / size))
        details.append(BucketProbability(bucket.market_id, probability, count, standard_error,
            standard_error > intended_edge * material_fraction))
    return details


def project_simplex(values: Sequence[float]) -> list[float]:
    """Euclidean projection onto {x >= 0, sum(x) = 1}."""
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not len(vector) or not np.all(np.isfinite(vector)):
        raise ValueError("finite one-dimensional values are required")
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - 1
    indices = np.arange(1, len(vector) + 1)
    eligible = ordered - cumulative / indices > 0
    rho = indices[eligible][-1]
    theta = cumulative[rho - 1] / rho
    projected = np.maximum(vector - theta, 0)
    projected /= projected.sum()
    return projected.tolist()


def fixed_weight_blend(model: Sequence[float], market: Sequence[float], weight: float) -> list[float]:
    if len(model) != len(market) or not 0 <= weight <= 1:
        raise ValueError("vectors must align and weight must be in [0, 1]")
    return project_simplex([weight * p + (1 - weight) * q for p, q in zip(model, market)])
