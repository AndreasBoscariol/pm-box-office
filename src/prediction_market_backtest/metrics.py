"""Probability scoring and movie-clustered bootstrap inference."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def brier_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    return float(np.mean((np.asarray(probabilities) - np.asarray(outcomes)) ** 2))


def log_loss(probabilities: Sequence[float], outcomes: Sequence[int], floor: float = 1e-6) -> float:
    p = np.clip(np.asarray(probabilities, dtype=float), floor, 1 - floor)
    y = np.asarray(outcomes, dtype=float)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def ranked_probability_score(probabilities: Sequence[float], winning_index: int) -> float:
    p = np.asarray(probabilities, dtype=float)
    if winning_index < 0 or winning_index >= len(p):
        raise ValueError("winning index outside vector")
    observed = np.zeros(len(p)); observed[winning_index] = 1
    return float(np.sum((np.cumsum(p)[:-1] - np.cumsum(observed)[:-1]) ** 2))


def clustered_bootstrap(movie_ids: Sequence[object], values: Sequence[float], *, iterations: int = 5000,
                        seed: int = 0, statistic: Callable[[np.ndarray], float] = np.mean) -> dict[str, float]:
    ids, data = np.asarray(movie_ids), np.asarray(values, dtype=float)
    unique = np.unique(ids)
    if not len(unique) or len(ids) != len(data):
        raise ValueError("aligned non-empty movie ids and values required")
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations)
    groups = {movie: data[ids == movie] for movie in unique}
    for index in range(iterations):
        sample = rng.choice(unique, size=len(unique), replace=True)
        estimates[index] = statistic(np.concatenate([groups[movie] for movie in sample]))
    return {"median": float(np.median(estimates)), "lower_95": float(np.quantile(estimates, .025)),
            "upper_95": float(np.quantile(estimates, .975)), "probability_positive": float(np.mean(estimates > 0))}

