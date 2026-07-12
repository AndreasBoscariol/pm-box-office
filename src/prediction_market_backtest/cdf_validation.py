"""Market-blind validation of predeclared 007 full-CDF extensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .policy007 import WeightedPool, shrunk_quantile_function, weighted_quantile


ANCHOR_PROBABILITIES = np.array([.025, .10, .90, .975])
EVALUATION_GRID = np.unique(np.r_[.001, np.linspace(.01, .99, 99), .999])


@dataclass(frozen=True, slots=True)
class CDFCandidate:
    name: str
    probabilities: np.ndarray
    residual_quantiles: np.ndarray
    preserves_anchors: bool

    def __post_init__(self) -> None:
        p, q = np.asarray(self.probabilities, float), np.asarray(self.residual_quantiles, float)
        if p.ndim != 1 or p.shape != q.shape or len(p) < 2 or np.any(np.diff(p) <= 0):
            raise ValueError("candidate requires aligned increasing probabilities")
        if np.any(np.diff(q) < -1e-12) or not np.all(np.isfinite(q)):
            raise ValueError("candidate quantile function must be finite and monotone")

    def quantile(self, probabilities: float | Sequence[float], point: float = 1.0) -> np.ndarray:
        return point * np.exp(np.interp(np.asarray(probabilities, float), self.probabilities, self.residual_quantiles))

    def cdf(self, outcome: float, point: float = 1.0) -> float:
        if outcome <= 0: return 0.0
        residual = np.log(outcome / point)
        return float(np.interp(residual, self.residual_quantiles, self.probabilities, left=0, right=1))


def build_candidates(pool: WeightedPool, grid: np.ndarray = EVALUATION_GRID) -> list[CDFCandidate]:
    grid = np.unique(np.r_[grid, ANCHOR_PROBABILITIES])
    locked = shrunk_quantile_function(pool, ANCHOR_PROBABILITIES)
    current = shrunk_quantile_function(pool, grid)
    raw = weighted_quantile(pool.residuals, pool.weights, grid)
    # Candidate B keeps the empirical shape and applies a monotone piecewise
    # residual displacement whose values at the four anchors are locked.
    raw_anchor = weighted_quantile(pool.residuals, pool.weights, ANCHOR_PROBABILITIES)
    delta = np.interp(grid, np.r_[0, ANCHOR_PROBABILITIES, 1], np.r_[locked[0]-raw_anchor[0], locked-raw_anchor, locked[-1]-raw_anchor[-1]])
    warped = np.maximum.accumulate(raw + delta)
    anchor_indices = np.searchsorted(grid, ANCHOR_PROBABILITIES)
    warped[anchor_indices] = locked
    # Re-enforce monotonicity separately between locked anchors.
    warped = _monotone_with_fixed_anchors(warped, anchor_indices)
    return [CDFCandidate("007_cdf_extension_001", grid, current, True),
            CDFCandidate("007_anchor_preserving_empirical_warp_001", grid, warped, True),
            CDFCandidate("007_raw_weighted_empirical_benchmark", grid, raw, bool(np.allclose(raw_anchor, locked, atol=1e-12)))]


def quantile_loss(outcome: float, quantile: float, probability: float) -> float:
    error = outcome - quantile
    return float(probability * error if error >= 0 else (1 - probability) * -error)


def crps_from_quantiles(candidate: CDFCandidate, outcome: float, point: float) -> float:
    quantiles = candidate.quantile(candidate.probabilities, point)
    losses = np.array([quantile_loss(outcome, q, p) for p, q in zip(candidate.probabilities, quantiles)])
    return float(2 * np.trapezoid(losses, candidate.probabilities))


def weighted_interval_score(candidate: CDFCandidate, outcome: float, point: float) -> float:
    scores=[]
    for alpha in (.2,.05):
        lower,upper=candidate.quantile([alpha/2,1-alpha/2],point)
        score=upper-lower + (2/alpha)*(lower-outcome)*(outcome<lower) + (2/alpha)*(outcome-upper)*(outcome>upper)
        scores.append(float(score))
    return float(np.mean(scores))


def evaluate_candidate(candidate: CDFCandidate, outcome: float, point: float) -> dict[str, float]:
    pit=candidate.cdf(outcome,point)
    return {"crps":crps_from_quantiles(candidate,outcome,point),"pit":pit,
            "weighted_interval_score":weighted_interval_score(candidate,outcome,point),
            "lower_tail":float(pit<.1),"upper_tail":float(pit>.9)}


def select_candidate(rows: Sequence[dict[str, float | int | str]]) -> tuple[str | None, list[dict[str, float | str | bool]]]:
    names=sorted({str(row["candidate"]) for row in rows}); summary=[]
    for name in names:
        selected=[row for row in rows if row["candidate"]==name]
        by_movie={int(row["movie_id"]):[] for row in selected}
        for row in selected:by_movie[int(row["movie_id"])].append(float(row["crps"]))
        movie_crps=float(np.mean([np.mean(values) for values in by_movie.values()]))
        pits=np.asarray([float(row["pit"]) for row in selected])
        anchor_ok=all(bool(row["preserves_anchors"]) for row in selected)
        tail_defect=abs(float(np.mean(pits<.1))-.1)>.05 or abs(float(np.mean(pits>.9))-.1)>.05
        summary.append({"candidate":name,"movie_averaged_crps":movie_crps,"pit_mean":float(np.mean(pits)),
            "pit_variance":float(np.var(pits)),"anchor_identity":anchor_ok,"material_tail_defect":tail_defect,
            "eligible":anchor_ok and not tail_defect})
    eligible=[row for row in summary if row["eligible"]]
    return (str(min(eligible,key=lambda row:float(row["movie_averaged_crps"]))["candidate"]) if eligible else None),summary


def _monotone_with_fixed_anchors(values: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    out=values.copy(); boundaries=np.r_[0,anchors,len(out)-1]
    for start,end in zip(boundaries[:-1],boundaries[1:]):
        segment=np.maximum.accumulate(out[start:end+1]); ceiling=out[end]
        out[start:end+1]=np.minimum(segment,ceiling)
    return out
