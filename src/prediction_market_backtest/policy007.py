"""Chronological full-CDF extension of the production 007 interval policy.

007 itself freezes four shrunk weighted quantiles, not a complete CDF. This
module extends the identical weighted-quantile/shrink operation to every
probability in (0, 1), which reproduces the four policy quantiles while making
the extra assumption explicit and versioned.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


POLICY_NAME = "007"
POLICY_VERSION = "boxoffice_local_007_weighted_interval_calibration"
DISTRIBUTION_VERSION = "boxoffice_local_007_distribution_001"


@dataclass(frozen=True, slots=True)
class WeightedPool:
    residuals: np.ndarray
    weights: np.ndarray
    member_ids: tuple[int, ...]
    fallback_level: int
    group_columns: tuple[str, ...]
    group_key: str
    shrink_weight: float
    global_residuals: np.ndarray
    global_weights: np.ndarray

    def __post_init__(self) -> None:
        residuals, weights = np.asarray(self.residuals, float), np.asarray(self.weights, float)
        if residuals.ndim != 1 or residuals.shape != weights.shape or not len(residuals):
            raise ValueError("aligned non-empty residuals and weights required")
        if not np.all(np.isfinite(residuals)) or not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("residuals and weights must be finite with positive total weight")

    @property
    def normalized_weights(self) -> np.ndarray:
        return self.weights / self.weights.sum()

    @property
    def effective_sample_size(self) -> float:
        weights = self.normalized_weights
        return float(1 / np.sum(weights ** 2))

    @property
    def membership_checksum(self) -> str:
        return _checksum(np.asarray(self.member_ids, dtype="<i8"))

    @property
    def weight_checksum(self) -> str:
        return _checksum(self.normalized_weights.astype("<f8"))


def weighted_quantile(values: Sequence[float], weights: Sequence[float], probabilities: float | Sequence[float]) -> np.ndarray:
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[valid], weights[valid]
    if not len(values): raise ValueError("positive weighted observations required")
    order = np.argsort(values, kind="stable"); values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights); thresholds = np.asarray(probabilities, float) * weights.sum()
    indices = np.searchsorted(cumulative, thresholds, side="left")
    return values[np.clip(indices, 0, len(values) - 1)]


def movie_weights(frame: pd.DataFrame, group_columns: Sequence[str]) -> np.ndarray:
    key = "release_run_id" if "release_run_id" in frame else "movie_id"
    counts = frame.groupby([*group_columns, key], dropna=False)["log_residual"].transform("count")
    return (1 / counts).to_numpy(dtype=float)


def build_chronological_pool(panel: pd.DataFrame, target_row: pd.Series, *, shrink_k: int = 20) -> WeightedPool:
    work = _prepare(panel)
    cutoff = pd.Timestamp(target_row["forecast_origin_date"])
    work = work.loc[work["opening_weekend_start"] < cutoff].copy()
    if work.empty: raise RuntimeError("no chronologically eligible 007 residuals")
    scopes = (("origin_bucket", "source_count_bucket", "point_bucket", "release_scope_bucket"),
              ("origin_bucket", "source_count_bucket", "point_bucket"),
              ("origin_bucket", "source_count_bucket"), ("origin_bucket",))
    target = _features(target_row)
    selected = None; selected_scope = None; level = 0
    for level, scope in enumerate(scopes, 1):
        mask = np.ones(len(work), dtype=bool)
        for column in scope: mask &= work[column].eq(target[column]).to_numpy()
        candidate = work.loc[mask].copy()
        if not candidate.empty:
            selected, selected_scope = candidate, scope; break
    if selected is None or selected_scope is None: raise RuntimeError("007 fallback hierarchy produced no pool")
    selected["movie_weight"] = movie_weights(selected, selected_scope)
    global_scope: tuple[str, ...] = ()
    global_counts = work.groupby("release_run_id", dropna=False)["log_residual"].transform("count")
    global_weights = (1 / global_counts).to_numpy(dtype=float)
    n = len(selected); shrink_weight = n / (n + shrink_k)
    return WeightedPool(selected.log_residual.to_numpy(float), selected.movie_weight.to_numpy(float),
        tuple(selected.movie_id.astype(int)), level, tuple(selected_scope), "|".join(target[c] for c in selected_scope),
        shrink_weight, work.log_residual.to_numpy(float), global_weights)


def shrunk_quantile_function(pool: WeightedPool, probabilities: Sequence[float]) -> np.ndarray:
    p = np.asarray(probabilities, float)
    cell_center = weighted_quantile(pool.residuals, pool.weights, .5).item()
    global_center = weighted_quantile(pool.global_residuals, pool.global_weights, .5).item()
    center = pool.shrink_weight * cell_center + (1 - pool.shrink_weight) * global_center
    cell_centered = weighted_quantile(pool.residuals - center, pool.weights, p)
    global_centered = weighted_quantile(pool.global_residuals - global_center, pool.global_weights, p)
    return center + pool.shrink_weight * cell_centered + (1 - pool.shrink_weight) * global_centered


def generate_draws(point_forecast: float, pool: WeightedPool, *, seed: int, n_draws: int) -> np.ndarray:
    if n_draws < 50_000: raise ValueError("at least 50,000 draws required")
    uniforms = np.random.default_rng(seed).random(n_draws)
    draws = point_forecast * np.exp(shrunk_quantile_function(pool, uniforms))
    if not np.all(np.isfinite(draws)) or np.any(draws < 0): raise ValueError("invalid 007 draws")
    return draws


def validate_policy_panel(policy_names: Iterable[str], *, comparison_diagnostic: bool = False) -> None:
    names = {str(name) for name in policy_names}
    if not names: raise ValueError("forecast policy is required")
    if names != {POLICY_NAME} and not comparison_diagnostic:
        raise ValueError(f"production probability panels require only policy {POLICY_NAME}; got {sorted(names)}")


def _prepare(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    point = pd.to_numeric(frame["primary_point_forecast_usd"], errors="coerce")
    actual = pd.to_numeric(frame["actual_opening_weekend_gross_usd"], errors="coerce")
    frame["opening_weekend_start"] = pd.to_datetime(frame["opening_weekend_start"], errors="coerce")
    frame["log_residual"] = np.log(actual / point)
    frame = frame.loc[point.gt(0) & actual.gt(0) & np.isfinite(frame.log_residual)].copy()
    for column, function in (("origin_bucket", _origin_bucket), ("source_count_bucket", _source_bucket),
                             ("point_bucket", _point_bucket)):
        source = frame["origin_day"] if column == "origin_bucket" else (frame["source_count"] if column == "source_count_bucket" else point.loc[frame.index])
        frame[column] = source.map(function)
    frame["release_scope_bucket"] = frame.apply(_release_bucket, axis=1)
    return frame


def _features(row: pd.Series) -> dict[str, str]:
    return {"origin_bucket": _origin_bucket(row["origin_day"]), "source_count_bucket": _source_bucket(row.get("source_count")),
            "point_bucket": _point_bucket(row["primary_point_forecast_usd"]), "release_scope_bucket": _release_bucket(row)}


def _origin_bucket(value: Any) -> str:
    day = int(value)
    return "P_-14_to_-8" if day <= -8 else ("P_-7_to_-3" if day <= -3 else f"P_{day}")


def _source_bucket(value: Any) -> str: return "one_source" if pd.notna(value) and float(value) <= 1 else "multi_source"
def _point_bucket(value: Any) -> str:
    value = float(value)
    return "lt_1m" if value < 1e6 else ("1m_5m" if value < 5e6 else ("5m_15m" if value < 15e6 else ("15m_50m" if value < 50e6 else "50m_plus")))
def _release_bucket(row: pd.Series) -> str:
    return "limited_or_platform" if "platform" in str(row.get("release_type", "")).lower() or str(row.get("release_width_bucket", "")).lower() in {"limited", "platform"} else "wide_or_large_wide"
def _checksum(values: np.ndarray) -> str: return hashlib.sha256(values.tobytes()).hexdigest()
