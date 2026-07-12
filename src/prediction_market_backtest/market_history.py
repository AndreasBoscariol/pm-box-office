"""Timestamp-valid historical price synchronization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Mapping, Sequence

from .probabilities import project_simplex


@dataclass(frozen=True, slots=True)
class PriceObservation:
    market_id: str
    observed_at: datetime
    yes_price: float
    no_price: float | None = None


@dataclass(frozen=True, slots=True)
class MarketVector:
    market_ids: tuple[str, ...]
    raw: tuple[float, ...]
    coherent: tuple[float, ...]
    observations: tuple[PriceObservation, ...]
    raw_sum: float


def synchronize_prices(market_ids: Sequence[str], observations: Iterable[PriceObservation], at: datetime,
                       maximum_age: timedelta) -> MarketVector:
    if at.tzinfo is None:
        raise ValueError("synchronization time must be timezone-aware")
    selected: dict[str, PriceObservation] = {}
    wanted = set(market_ids)
    for observation in observations:
        if observation.market_id not in wanted or observation.observed_at > at:
            continue
        if at - observation.observed_at > maximum_age:
            continue
        previous = selected.get(observation.market_id)
        if previous is None or observation.observed_at > previous.observed_at:
            selected[observation.market_id] = observation
    missing = [market_id for market_id in market_ids if market_id not in selected]
    if missing:
        raise ValueError("incomplete event vector: " + ", ".join(missing))
    ordered = tuple(selected[market_id] for market_id in market_ids)
    raw = tuple(float(item.yes_price) for item in ordered)
    if any(not 0 <= value <= 1 for value in raw):
        raise ValueError("prices must lie in [0, 1]")
    return MarketVector(tuple(market_ids), raw, tuple(project_simplex(raw)), ordered, sum(raw))

