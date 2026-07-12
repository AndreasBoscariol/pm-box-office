"""Chronological model/market/blend probability evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .probabilities import fixed_weight_blend


@dataclass(frozen=True, slots=True)
class ProbabilityCase:
    movie_id: int
    release_ordinal: int
    model: tuple[float, ...]
    market: tuple[float, ...]
    winning_index: int


def multiclass_brier(probabilities: Sequence[float], winning_index: int) -> float:
    observed = np.zeros(len(probabilities)); observed[winning_index] = 1.0
    return float(np.sum((np.asarray(probabilities) - observed) ** 2))


def multiclass_log_loss(probabilities: Sequence[float], winning_index: int, floor: float = 1e-6) -> float:
    return float(-np.log(np.clip(float(probabilities[winning_index]), floor, 1.0)))


def select_blend_weight(training: Iterable[ProbabilityCase], weights: Sequence[float] = tuple(np.linspace(0, 1, 11))) -> tuple[float, list[dict[str, float]]]:
    cases = list(training)
    if not cases: raise ValueError("chronological training cases are required")
    rows = []
    for weight in weights:
        scores = [multiclass_brier(fixed_weight_blend(case.model, case.market, weight), case.winning_index) for case in cases]
        rows.append({"weight": float(weight), "mean_brier": float(np.mean(scores))})
    best = min(rows, key=lambda row: (row["mean_brier"], row["weight"]))
    return best["weight"], rows


def expanding_blend_evaluation(cases: Iterable[ProbabilityCase], minimum_training_movies: int = 10) -> list[dict[str, float | int]]:
    ordered = sorted(cases, key=lambda case: (case.release_ordinal, case.movie_id))
    output = []
    for case in ordered:
        training = [row for row in ordered if row.release_ordinal < case.release_ordinal]
        if len({row.movie_id for row in training}) < minimum_training_movies: continue
        weight, _ = select_blend_weight(training)
        blend = fixed_weight_blend(case.model, case.market, weight)
        output.append({"movie_id": case.movie_id, "release_ordinal": case.release_ordinal, "weight": weight,
            "model_brier": multiclass_brier(case.model, case.winning_index),
            "market_brier": multiclass_brier(case.market, case.winning_index),
            "blend_brier": multiclass_brier(blend, case.winning_index),
            "model_log_loss": multiclass_log_loss(case.model, case.winning_index),
            "market_log_loss": multiclass_log_loss(case.market, case.winning_index),
            "blend_log_loss": multiclass_log_loss(blend, case.winning_index)})
    return output

