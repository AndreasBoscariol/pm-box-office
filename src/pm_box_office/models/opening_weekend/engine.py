"""Deterministic deployed prediction routing for opening-window forecasts."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Any

from pm_box_office.models.opening_weekend.registry import DEFAULT_REGISTRY, ModelRegistry, RegistryEntry
from pm_box_office.models.opening_weekend.targets import OpeningTarget, bop_segment_for_midpoint, resolve_opening_target


@dataclass(frozen=True)
class SourceAvailability:
    bop_available: bool = False
    bop_as_of_date: dt.date | None = None
    wiki_available: bool = False
    wiki_as_of_date: dt.date | None = None
    competition_available: bool = False
    competition_as_of_date: dt.date | None = None
    actuals_available: bool = False
    actuals_as_of_date: dt.date | None = None
    amc_available: bool = False
    amc_as_of_date: dt.date | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "bop_available": self.bop_available,
            "bop_as_of_date": self.bop_as_of_date.isoformat() if self.bop_as_of_date else "",
            "wiki_available": self.wiki_available,
            "wiki_as_of_date": self.wiki_as_of_date.isoformat() if self.wiki_as_of_date else "",
            "competition_available": self.competition_available,
            "competition_as_of_date": self.competition_as_of_date.isoformat() if self.competition_as_of_date else "",
            "actuals_available": self.actuals_available,
            "actuals_as_of_date": self.actuals_as_of_date.isoformat() if self.actuals_as_of_date else "",
            "amc_available": self.amc_available,
            "amc_as_of_date": self.amc_as_of_date.isoformat() if self.amc_as_of_date else "",
        }


@dataclass(frozen=True)
class FeatureSnapshot:
    movie_id: int
    snapshot_day: int
    as_of_date: dt.date
    target: OpeningTarget
    bop_midpoint_usd: float | None
    known_actual_offsets: tuple[int, ...]
    known_actual_gross_so_far: float
    source_availability: SourceAvailability

    @property
    def days_elapsed_in_target_window(self) -> int:
        return len(self.known_actual_offsets)

    @property
    def days_remaining_in_target_window(self) -> int:
        return max(0, self.target.target_days - self.days_elapsed_in_target_window)

    @property
    def bop_segment(self) -> str:
        return bop_segment_for_midpoint(self.bop_midpoint_usd)


@dataclass(frozen=True)
class OpeningWindowPrediction:
    target: OpeningTarget
    bop_segment: str
    forecast_state: str
    deployment_status: str
    registry_entry: RegistryEntry
    known_actual_offsets: tuple[int, ...]
    known_actual_gross_so_far: float
    remaining_gross_forecast: float | None
    point_forecast_usd: float | None
    lower_80_usd: float | None
    upper_80_usd: float | None
    source_availability: SourceAvailability
    shadow_reason: str

    def to_payload(self) -> dict[str, object]:
        return {
            "target_type": self.target.target_type,
            "target_days": self.target.target_days,
            "target_start_date": self.target.target_start_date.isoformat(),
            "target_end_date": self.target.target_end_date.isoformat(),
            "bop_segment": self.bop_segment,
            "forecast_state": self.forecast_state,
            "deployment_status": self.deployment_status,
            "known_actual_offsets": list(self.known_actual_offsets),
            "known_actual_gross_so_far": self.known_actual_gross_so_far,
            "remaining_gross_forecast": self.remaining_gross_forecast,
            "model_version": self.registry_entry.model_version,
            "model_family": self.registry_entry.model_family,
            "registry_reason": self.registry_entry.reason,
            "shadow_reason": self.shadow_reason,
            "source_availability": self.source_availability.to_payload(),
        }


class OpeningWindowPredictionEngine:
    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or DEFAULT_REGISTRY

    def select_entry(self, snapshot: FeatureSnapshot) -> tuple[str, RegistryEntry]:
        state = forecast_state_for_snapshot(snapshot)
        entry = self.registry.select(
            target_type=snapshot.target.target_type,
            bop_segment=snapshot.bop_segment,
            forecast_state=state,
        )
        return state, entry

    def predict(
        self,
        snapshot: FeatureSnapshot,
        *,
        predicted_log_residual: float = 0.0,
        expected_remaining_gross: float | None = None,
        predicted_remaining_log_residual: float = 0.0,
        interval_log_width: float = 0.25,
    ) -> OpeningWindowPrediction:
        state, entry = self.select_entry(snapshot)
        point: float | None
        remaining: float | None
        if state == "final":
            remaining = 0.0
            point = snapshot.known_actual_gross_so_far
        elif state == "official_actuals_available":
            base_remaining = max(0.0, expected_remaining_gross or 0.0)
            remaining = base_remaining * math.exp(predicted_remaining_log_residual)
            point = snapshot.known_actual_gross_so_far + remaining
        elif snapshot.bop_midpoint_usd is not None and snapshot.bop_midpoint_usd > 0.0:
            remaining = None
            point = snapshot.bop_midpoint_usd * math.exp(predicted_log_residual)
        else:
            remaining = None
            point = None

        lower = None
        upper = None
        if point is not None:
            lower = max(snapshot.known_actual_gross_so_far, point * math.exp(-interval_log_width))
            upper = max(lower, point * math.exp(interval_log_width))
        deployment_status = entry.deployment_status
        shadow_reason = entry.reason if deployment_status == "shadow" else ""
        return OpeningWindowPrediction(
            target=snapshot.target,
            bop_segment=snapshot.bop_segment,
            forecast_state=state,
            deployment_status=deployment_status,
            registry_entry=entry,
            known_actual_offsets=snapshot.known_actual_offsets,
            known_actual_gross_so_far=snapshot.known_actual_gross_so_far,
            remaining_gross_forecast=remaining,
            point_forecast_usd=point,
            lower_80_usd=lower,
            upper_80_usd=upper,
            source_availability=snapshot.source_availability,
            shadow_reason=shadow_reason,
        )


def forecast_state_for_snapshot(snapshot: FeatureSnapshot) -> str:
    if len(snapshot.known_actual_offsets) >= snapshot.target.target_days:
        return "final"
    if snapshot.known_actual_offsets:
        return "official_actuals_available"
    if snapshot.source_availability.amc_available and snapshot.snapshot_day >= 0:
        return "same_day_proxy_available"
    return "pre_release"


def known_actual_offsets_for_as_of(
    *,
    target_start_date: dt.date,
    target_days: int,
    as_of_date: dt.date,
) -> tuple[int, ...]:
    latest_available_date = as_of_date - dt.timedelta(days=1)
    offsets = [
        offset
        for offset in range(target_days)
        if target_start_date + dt.timedelta(days=offset) <= latest_available_date
    ]
    return tuple(offsets)


def feature_snapshot_from_panel_row(
    row: dict[str, Any],
    *,
    actual_gross_by_offset: dict[int, float] | None = None,
) -> FeatureSnapshot:
    actual_gross_by_offset = actual_gross_by_offset or {}
    as_of = parse_date(row.get("as_of_date"))
    row_target_type = str(row.get("target_type") or "")
    target = resolve_opening_target(
        parse_date(row.get("opening_date") or row.get("target_start_date")),
        target_start_date=parse_optional_date(row.get("target_start_date")),
        target_end_date=parse_optional_date(row.get("target_end_date")),
        target_days=int(row["target_day_count"]) if row.get("target_day_count") not in (None, "") else None,
        target_type=row_target_type or None,
    )
    offsets = tuple(
        offset
        for offset in known_actual_offsets_for_as_of(
            target_start_date=target.target_start_date,
            target_days=target.target_days,
            as_of_date=as_of,
        )
        if float(actual_gross_by_offset.get(offset, 0.0) or 0.0) > 0.0
    )
    known = sum(float(actual_gross_by_offset.get(offset, 0.0) or 0.0) for offset in offsets)
    source_availability = SourceAvailability(
        bop_available=float(row.get("bop_forecast_available", 0.0) or 0.0) > 0.0,
        bop_as_of_date=parse_optional_date(row.get("bop_forecast_published_date")),
        wiki_available=float(row.get("wiki_available", 0.0) or 0.0) > 0.0,
        wiki_as_of_date=as_of,
        competition_available=float(row.get("competitor_count_lag7", 0.0) or 0.0) > 0.0,
        competition_as_of_date=as_of,
        actuals_available=bool(offsets),
        actuals_as_of_date=as_of - dt.timedelta(days=1) if offsets else None,
        amc_available=float(row.get("amc_available", 0.0) or 0.0) > 0.0,
        amc_as_of_date=as_of if float(row.get("amc_available", 0.0) or 0.0) > 0.0 else None,
    )
    midpoint = float(row.get("bop_forecast_midpoint", 0.0) or 0.0)
    return FeatureSnapshot(
        movie_id=int(row["movie_id"]),
        snapshot_day=int(row["snapshot_day"]),
        as_of_date=as_of,
        target=target,
        bop_midpoint_usd=midpoint if midpoint > 0.0 else None,
        known_actual_offsets=offsets,
        known_actual_gross_so_far=known,
        source_availability=source_availability,
    )


def parse_date(value: object) -> dt.date:
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))


def parse_optional_date(value: object) -> dt.date | None:
    if value in (None, ""):
        return None
    return parse_date(value)
