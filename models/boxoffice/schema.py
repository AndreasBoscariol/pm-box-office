"""Canonical production objects for movie opening-weekend forecasts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from .constants import DAY_TO_TARGET, PRE_RELEASE_REGIME
from .intervals import log_normal_interval


@dataclass(frozen=True)
class MovieOpening:
    movie_id: int
    release_run_id: int
    title: str
    opening_weekend_start: date


@dataclass(frozen=True)
class ForecastOrigin:
    regime: str
    origin_key: str
    origin_day: int | None
    forecast_origin: str | None
    forecast_origin_local: datetime
    forecast_origin_utc: datetime
    as_of_utc: datetime


@dataclass(frozen=True)
class DailyComponent:
    component_day: str
    component_type: str
    component_point_usd: float
    component_sigma_log: float
    component_model: str
    component_source: str
    component_notes: str | None = None
    component_lo80_usd: float | None = None
    component_hi80_usd: float | None = None
    component_lo95_usd: float | None = None
    component_hi95_usd: float | None = None

    def with_intervals(self) -> "DailyComponent":
        if all(
            value is not None and math.isfinite(value)
            for value in [
                self.component_lo80_usd,
                self.component_hi80_usd,
                self.component_lo95_usd,
                self.component_hi95_usd,
            ]
        ):
            return self
        if self.component_type == "actual" or self.component_sigma_log == 0:
            return DailyComponent(
                **{
                    **self.__dict__,
                    "component_lo80_usd": self.component_point_usd,
                    "component_hi80_usd": self.component_point_usd,
                    "component_lo95_usd": self.component_point_usd,
                    "component_hi95_usd": self.component_point_usd,
                }
            )
        interval = log_normal_interval(self.component_point_usd, self.component_sigma_log)
        return DailyComponent(
            **{
                **self.__dict__,
                "component_lo80_usd": interval["lo80_usd"],
                "component_hi80_usd": interval["hi80_usd"],
                "component_lo95_usd": interval["lo95_usd"],
                "component_hi95_usd": interval["hi95_usd"],
            }
        )


@dataclass(frozen=True)
class ForecastResult:
    movie: MovieOpening
    origin: ForecastOrigin
    target: str
    point_usd: float
    lo80_usd: float
    hi80_usd: float
    lo95_usd: float
    hi95_usd: float
    point_model: str
    interval_model: str
    component_source: str
    model_version: str
    run_id: str
    components: list[DailyComponent] = field(default_factory=list)
    feature_quality_bucket: str | None = None
    source_count: int | None = None
    estimate_sources: str | None = None
    amc_coverage: float | None = None
    amc_snapshot_count: int | None = None
    amc_lateness_p50_minutes: float | None = None
    actual_usd: float | None = None
    is_live: bool = False
    is_backtest: bool = False
    audit: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None

    @property
    def forecast_id(self) -> str:
        payload = {
            "model_version": self.model_version,
            "movie_id": self.movie.movie_id,
            "release_run_id": self.movie.release_run_id,
            "origin_key": self.origin.origin_key,
            "target": self.target,
        }
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def to_forecast_row(self) -> dict[str, Any]:
        actual = self.actual_usd
        log_error = (
            math.log(actual / self.point_usd)
            if actual is not None and actual > 0 and self.point_usd > 0 and math.isfinite(self.point_usd)
            else None
        )
        abs_pct_error = (
            abs(actual - self.point_usd) / actual
            if actual is not None and actual > 0 and self.point_usd > 0 and math.isfinite(self.point_usd)
            else None
        )
        created_at = self.created_at or datetime.now(timezone.utc)
        return {
            "forecast_id": self.forecast_id,
            "run_id": self.run_id,
            "model_version": self.model_version,
            "movie_id": self.movie.movie_id,
            "release_run_id": self.movie.release_run_id,
            "title": self.movie.title,
            "opening_weekend_start": self.movie.opening_weekend_start,
            "regime": self.origin.regime,
            "origin_key": self.origin.origin_key,
            "origin_day": self.origin.origin_day,
            "forecast_origin_local": self.origin.forecast_origin_local,
            "forecast_origin_utc": self.origin.forecast_origin_utc,
            "as_of_utc": self.origin.as_of_utc,
            "target": self.target,
            "point_usd": self.point_usd,
            "lo80_usd": self.lo80_usd,
            "hi80_usd": self.hi80_usd,
            "lo95_usd": self.lo95_usd,
            "hi95_usd": self.hi95_usd,
            "point_model": self.point_model,
            "interval_model": self.interval_model,
            "component_source": self.component_source,
            "feature_quality_bucket": self.feature_quality_bucket,
            "source_count": self.source_count,
            "estimate_sources": self.estimate_sources,
            "amc_coverage": self.amc_coverage,
            "amc_snapshot_count": self.amc_snapshot_count,
            "amc_lateness_p50_minutes": self.amc_lateness_p50_minutes,
            "actual_usd": actual,
            "log_error": log_error,
            "abs_pct_error": abs_pct_error,
            "is_live": self.is_live,
            "is_backtest": self.is_backtest,
            "created_at": created_at,
        }

    def to_component_rows(self) -> list[dict[str, Any]]:
        rows = []
        for component in self.components:
            item = component.with_intervals()
            rows.append(
                {
                    "forecast_id": self.forecast_id,
                    "component_day": item.component_day,
                    "component_type": item.component_type,
                    "component_point_usd": item.component_point_usd,
                    "component_sigma_log": item.component_sigma_log,
                    "component_lo80_usd": item.component_lo80_usd,
                    "component_hi80_usd": item.component_hi80_usd,
                    "component_lo95_usd": item.component_lo95_usd,
                    "component_hi95_usd": item.component_hi95_usd,
                    "component_model": item.component_model,
                    "component_source": item.component_source,
                    "component_notes": item.component_notes,
                }
            )
        return rows


def daily_target_for_component(day: str) -> str:
    return DAY_TO_TARGET[day]


def is_pre_release_origin(origin: ForecastOrigin) -> bool:
    return origin.regime == PRE_RELEASE_REGIME
