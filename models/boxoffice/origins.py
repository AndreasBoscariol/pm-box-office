"""Forecast-origin construction for historical and live timelines."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from .constants import (
    LIVE_ORIGINS,
    PRE_RELEASE_ORIGIN_DAYS,
    PRE_RELEASE_REGIME,
    REGIME_DAY_OFFSET,
    REGIME_PREFIX,
)
from .schema import ForecastOrigin, MovieOpening


def _origin_time(origin: str) -> time:
    if origin == "EOD":
        return time(23, 59)
    hour, minute = origin.split(":", 1)
    return time(int(hour), int(minute))


def _localize(day: date, clock: time, tz: ZoneInfo) -> datetime:
    return datetime.combine(day, clock).replace(tzinfo=tz)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def build_forecast_origins(
    movie: MovieOpening,
    *,
    mode: str,
    as_of_utc: datetime | None = None,
    origin_timezone: str = "America/New_York",
    include_pre_release: bool = True,
    include_live: bool = True,
) -> list[ForecastOrigin]:
    """Build leakage-safe forecast origins for a movie timeline."""

    if mode not in {"historical", "live"}:
        raise ValueError("mode must be either 'historical' or 'live'")
    now_utc = _as_utc(as_of_utc or datetime.now(timezone.utc))
    tz = ZoneInfo(origin_timezone)
    origins: list[ForecastOrigin] = []

    if include_pre_release:
        for origin_day in PRE_RELEASE_ORIGIN_DAYS:
            local_day = movie.opening_weekend_start + timedelta(days=origin_day)
            local_dt = _localize(local_day, time(0, 0), tz)
            utc_dt = _as_utc(local_dt)
            if mode == "live" and utc_dt > now_utc:
                continue
            origins.append(
                ForecastOrigin(
                    regime=PRE_RELEASE_REGIME,
                    origin_key=f"P_{origin_day}",
                    origin_day=origin_day,
                    forecast_origin=None,
                    forecast_origin_local=local_dt,
                    forecast_origin_utc=utc_dt,
                    as_of_utc=utc_dt,
                )
            )

    if include_live:
        for regime, day_offset in REGIME_DAY_OFFSET.items():
            local_day = movie.opening_weekend_start + timedelta(days=day_offset)
            for origin in LIVE_ORIGINS:
                local_dt = _localize(local_day, _origin_time(origin), tz)
                utc_dt = _as_utc(local_dt)
                if mode == "live" and utc_dt > now_utc:
                    continue
                origins.append(
                    ForecastOrigin(
                        regime=regime,
                        origin_key=f"{REGIME_PREFIX[regime]}_{origin}",
                        origin_day=day_offset,
                        forecast_origin=origin,
                        forecast_origin_local=local_dt,
                        forecast_origin_utc=utc_dt,
                        as_of_utc=utc_dt if mode == "historical" else min(utc_dt, now_utc),
                    )
                )

    return origins

