"""Auditable adapters from production forecasts to full empirical distributions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, time, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ForecastDistribution:
    movie_id: int
    regime: str
    origin: str
    information_cutoff_utc: datetime
    forecast_created_utc: datetime
    forecast_available_utc: datetime
    point_forecast: Decimal
    draws: np.ndarray
    model_artifact_version: str
    distribution_artifact_version: str
    simulator_version: str
    seed: int
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        draws = np.asarray(self.draws, dtype=np.float64)
        if draws.ndim != 1 or not len(draws) or not np.all(np.isfinite(draws)) or np.any(draws < 0):
            raise ValueError("draws must be a non-empty finite nonnegative vector")
        if self.forecast_available_utc < self.information_cutoff_utc:
            raise ValueError("forecast cannot be available before its information cutoff")
        draws.setflags(write=False)
        object.__setattr__(self, "draws", draws)

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.draws.astype("<f8", copy=False).tobytes()).hexdigest()


class ForecastDistributionAdapter(Protocol):
    def generate_distribution(self, *, movie_id: int, regime: str, origin: str,
                              information_cutoff_utc: datetime, seed: int, n_draws: int) -> ForecastDistribution: ...


class PreReleaseProductionAdapter:
    """Use production panel/policies and chronological, movie-identified OOF errors.

    The adapter deliberately calls production inference for the point forecast. It
    derives only the empirical error distribution, because artifact 005's stored
    residual samples omit member identity and eligibility timestamps.
    """

    def __init__(self, artifact_dir: str | Path, *, minimum_residuals: int = 20,
                 distribution_version: str = "boxoffice_pre_release_distribution_001") -> None:
        from models.boxoffice.artifacts import load_model_artifacts

        path = Path(artifact_dir).resolve()
        self.artifacts = load_model_artifacts(path.name, artifact_root=path.parent)
        self.minimum_residuals = minimum_residuals
        self.distribution_version = distribution_version

    def generate_distribution(self, *, movie_id: int, regime: str, origin: str,
                              information_cutoff_utc: datetime, seed: int, n_draws: int) -> ForecastDistribution:
        from models.boxoffice.constants import PRE_RELEASE_REGIME
        from models.boxoffice.pre_release import forecast_pre_release_opening_weekend, point_method_for_origin
        from models.boxoffice.schema import ForecastOrigin, MovieOpening

        if regime != PRE_RELEASE_REGIME and regime != "pre_release":
            raise ValueError(f"unsupported regime: {regime}")
        origin_day = _origin_day(origin)
        if origin_day not in range(-14, 0):
            raise ValueError(f"unsupported pre-release origin: {origin}")
        if information_cutoff_utc.tzinfo is None:
            raise ValueError("information cutoff must be timezone-aware")
        if n_draws < 50_000:
            raise ValueError("at least 50,000 draws are required")
        panel = self.artifacts.pre_release_panel.copy()
        panel["origin_day"] = pd.to_numeric(panel["origin_day"], errors="coerce")
        panel["movie_id"] = pd.to_numeric(panel["movie_id"], errors="coerce")
        matches = panel.loc[panel["movie_id"].eq(movie_id) & panel["origin_day"].eq(origin_day)]
        if len(matches) != 1:
            raise LookupError(f"expected one production panel row for movie_id={movie_id}, origin={origin_day}; got {len(matches)}")
        row = matches.iloc[0]
        opening = pd.Timestamp(row["opening_weekend_start"]).date()
        forecast_date = pd.Timestamp(row["forecast_origin_date"]).date()
        expected_cutoff = datetime.combine(forecast_date, time.max, timezone.utc)
        if information_cutoff_utc.date() != forecast_date:
            raise ValueError(f"cutoff date {information_cutoff_utc.date()} does not match production origin {forecast_date}")
        movie = MovieOpening(movie_id=movie_id, release_run_id=int(row["release_run_id"]), title=str(row["title"]),
                             opening_weekend_start=opening)
        production_origin = ForecastOrigin(regime=PRE_RELEASE_REGIME, origin_key=f"P{origin_day}", origin_day=origin_day,
            forecast_origin=str(origin_day), forecast_origin_local=information_cutoff_utc,
            forecast_origin_utc=information_cutoff_utc, as_of_utc=information_cutoff_utc)
        forecast = next(result for result in forecast_pre_release_opening_weekend(movie=movie, origin=production_origin,
            artifacts=self.artifacts, run_id="prediction-market-distribution", is_backtest=True)
            if result.target == "opening_weekend")
        method = point_method_for_origin(self.artifacts.pre_release_point_policy, origin_day)
        eligible = panel.loc[panel["origin_day"].eq(origin_day)].copy()
        eligible["opening_weekend_start"] = pd.to_datetime(eligible["opening_weekend_start"], errors="coerce", utc=True)
        eligible = eligible.loc[eligible["opening_weekend_start"] < pd.Timestamp(information_cutoff_utc)]
        point = pd.to_numeric(eligible.get(method), errors="coerce")
        actual = pd.to_numeric(eligible.get("actual_opening_weekend_gross_usd"), errors="coerce")
        valid = point.gt(0) & actual.gt(0) & np.isfinite(point) & np.isfinite(actual)
        eligible, point, actual = eligible.loc[valid], point.loc[valid], actual.loc[valid]
        residuals = np.log(actual.to_numpy(dtype=float) / point.to_numpy(dtype=float))
        if len(residuals) < self.minimum_residuals:
            raise RuntimeError(f"undefined production fallback: only {len(residuals)} chronological residuals")
        rng = np.random.default_rng(seed)
        sampled = rng.choice(residuals, size=n_draws, replace=True)
        raw_draws = float(forecast.point_usd) * np.exp(sampled)
        draws = np.maximum(raw_draws, 0.0)
        member_ids = sorted({int(value) for value in eligible["movie_id"].dropna()})
        metadata = {"target_metric": "domestic_us_canada_three_day_opening_weekend_gross_usd",
            "point_method": method, "residual_coordinate": "log(actual/point)", "residual_count": len(residuals),
            "residual_movie_ids": member_ids, "residual_pool_checksum": _array_checksum(np.sort(residuals)),
            "truncated_draw_count": int(np.sum(raw_draws < 0)), "expected_cutoff_end_utc": expected_cutoff.isoformat(),
            "source_panel": str(self.artifacts.artifact_dir / "pre_release_panel.csv")}
        # The historical panel records a daily information origin but no runtime
        # duration. Use the recorded cutoff as availability and disclose that
        # zero-latency convention; callers may add a configured publication lag.
        return ForecastDistribution(movie_id, "pre_release", str(origin_day), information_cutoff_utc,
            information_cutoff_utc, information_cutoff_utc,
            Decimal(str(forecast.point_usd)), draws, self.artifacts.model_version, self.distribution_version,
            "empirical_log_residual_bootstrap_v1", seed, {**metadata, "availability_assumption": "cutoff_zero_lag"})


class Policy007ProductionAdapter(PreReleaseProductionAdapter):
    """Canonical production adapter using the versioned 007 full-CDF extension."""

    def __init__(self, artifact_dir: str | Path, *, policy_name: str = "007",
                 expected_artifact_checksum: str | None = None) -> None:
        from .policy007 import DISTRIBUTION_VERSION, POLICY_NAME

        if not policy_name:
            raise ValueError("forecast policy is required")
        if policy_name != POLICY_NAME:
            raise ValueError(f"production probability generation requires policy {POLICY_NAME}")
        super().__init__(artifact_dir, distribution_version=DISTRIBUTION_VERSION)
        self.policy_name = policy_name
        self.training_panel_checksum = _file_checksum(self.artifacts.artifact_dir / "pre_release_panel.csv")
        self.policy_checksum = _file_checksum(self.artifacts.artifact_dir / "pre_release_interval_policy.json")
        self.artifact_checksum = hashlib.sha256((self.training_panel_checksum + self.policy_checksum).encode()).hexdigest()
        if expected_artifact_checksum and expected_artifact_checksum != self.artifact_checksum:
            raise ValueError("007 artifact checksum differs from configured checksum")

    def generate_distribution(self, *, movie_id: int, regime: str, origin: str,
                              information_cutoff_utc: datetime, seed: int, n_draws: int) -> ForecastDistribution:
        from .policy007 import POLICY_VERSION, build_chronological_pool, generate_draws

        # Production inference and its fail-closed origin/cutoff checks remain in
        # the parent. Its generic draws are discarded and never persisted.
        base = super().generate_distribution(movie_id=movie_id, regime=regime, origin=origin,
            information_cutoff_utc=information_cutoff_utc, seed=seed, n_draws=n_draws)
        panel = self.artifacts.pre_release_panel
        origin_day = _origin_day(origin)
        matches = panel.loc[pd.to_numeric(panel["movie_id"], errors="coerce").eq(movie_id)
            & pd.to_numeric(panel["origin_day"], errors="coerce").eq(origin_day)]
        if len(matches) != 1: raise LookupError("007 target row identity is ambiguous")
        pool = build_chronological_pool(panel, matches.iloc[0])
        draws = generate_draws(float(base.point_forecast), pool, seed=seed, n_draws=n_draws)
        metadata = {**base.metadata, "forecast_policy_name": self.policy_name,
            "forecast_policy_version": POLICY_VERSION, "point_policy_version": self.artifacts.model_version,
            "training_panel_checksum": self.training_panel_checksum, "policy_checksum": self.policy_checksum,
            "artifact_checksum": self.artifact_checksum, "residual_count": len(pool.residuals),
            "effective_sample_size": pool.effective_sample_size, "fallback_level": pool.fallback_level,
            "fallback_group_columns": pool.group_columns, "fallback_group_key": pool.group_key,
            "residual_membership_checksum": pool.membership_checksum, "weight_checksum": pool.weight_checksum,
            "direct_cdf_available": False,
            "cdf_extension_assumption": "007 weighted empirical quantile and shrink formula extended to all probabilities"}
        return ForecastDistribution(base.movie_id, base.regime, base.origin, base.information_cutoff_utc,
            base.forecast_created_utc, base.forecast_available_utc, base.point_forecast, draws,
            base.model_artifact_version, self.distribution_version, "007_shrunk_weighted_quantile_cdf_v1",
            seed, metadata)


def stable_seed(movie_id: int, origin: str, artifact_version: str, configured_seed: int) -> int:
    raw = json.dumps([movie_id, str(origin), artifact_version, configured_seed], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def save_distribution(distribution: ForecastDistribution, directory: str | Path) -> tuple[Path, Path]:
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    stem = f"movie_{distribution.movie_id}_origin_{distribution.origin}_{distribution.checksum[:12]}"
    draws_path, metadata_path = directory / f"{stem}.npz", directory / f"{stem}.json"
    np.savez_compressed(draws_path, draws=distribution.draws)
    payload = {field: getattr(distribution, field) for field in (
        "movie_id", "regime", "origin", "information_cutoff_utc", "forecast_created_utc", "forecast_available_utc",
        "point_forecast", "model_artifact_version", "distribution_artifact_version", "simulator_version", "seed", "metadata")}
    payload.update({"n_draws": len(distribution.draws), "checksum": distribution.checksum, "draws_path": str(draws_path)})
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return draws_path, metadata_path


def _origin_day(origin: str) -> int:
    value = str(origin).upper().removeprefix("P").strip()
    try: return int(value)
    except ValueError as exc: raise ValueError(f"invalid origin: {origin}") from exc


def _array_checksum(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def _file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
