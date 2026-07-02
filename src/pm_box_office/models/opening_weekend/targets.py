"""Target-window resolution and deployment segmentation."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


THREE_DAY_TARGET = "3_day"
FOUR_DAY_TARGET = "4_day"
FIVE_DAY_TARGET = "5_day"

LARGE_BOP_SEGMENT = "large_25m_plus"
MID_BOP_SEGMENT = "mid_5m_25m"
SMALL_BOP_SEGMENT = "small_shadow"
MISSING_BOP_SEGMENT = "missing_bop_shadow"


@dataclass(frozen=True)
class OpeningTarget:
    target_start_date: dt.date
    target_end_date: dt.date
    target_days: int
    target_type: str

    @property
    def offsets(self) -> tuple[int, ...]:
        return tuple(range(self.target_days))


def target_type_for_day_count(day_count: int) -> str:
    if day_count == 3:
        return THREE_DAY_TARGET
    if day_count == 4:
        return FOUR_DAY_TARGET
    if day_count == 5:
        return FIVE_DAY_TARGET
    return f"{day_count}_day"


def resolve_opening_target(
    opening_date: dt.date,
    *,
    target_start_date: dt.date | None = None,
    target_end_date: dt.date | None = None,
    target_days: int | None = None,
    target_type: str | None = None,
    holiday_flag: bool = False,
) -> OpeningTarget:
    """Resolve the exact opening-window target before any forecast is made."""

    start = target_start_date or opening_date
    if target_end_date is not None:
        days = (target_end_date - start).days + 1
    elif target_days is not None:
        days = target_days
        target_end_date = start + dt.timedelta(days=days - 1)
    elif target_type is not None:
        days = target_day_count_from_type(target_type)
        target_end_date = start + dt.timedelta(days=days - 1)
    elif opening_date.weekday() == 2:
        days = 5
        target_end_date = start + dt.timedelta(days=4)
    elif holiday_flag:
        days = 4
        target_end_date = start + dt.timedelta(days=3)
    else:
        days = 3
        target_end_date = start + dt.timedelta(days=2)

    if days <= 0:
        raise ValueError("target_end_date must be on or after target_start_date")
    resolved_type = target_type or target_type_for_day_count(days)
    return OpeningTarget(
        target_start_date=start,
        target_end_date=target_end_date,
        target_days=days,
        target_type=resolved_type,
    )


def target_day_count_from_type(target_type: str) -> int:
    if target_type == THREE_DAY_TARGET:
        return 3
    if target_type == FOUR_DAY_TARGET:
        return 4
    if target_type == FIVE_DAY_TARGET:
        return 5
    if target_type.endswith("_day"):
        try:
            return int(target_type.removesuffix("_day"))
        except ValueError:
            pass
    raise ValueError(f"unknown target type {target_type!r}")


def bop_segment_for_midpoint(midpoint_usd: float | int | None) -> str:
    if midpoint_usd is None:
        return MISSING_BOP_SEGMENT
    midpoint = float(midpoint_usd)
    if midpoint <= 0.0:
        return MISSING_BOP_SEGMENT
    if midpoint >= 25_000_000:
        return LARGE_BOP_SEGMENT
    if midpoint >= 5_000_000:
        return MID_BOP_SEGMENT
    return SMALL_BOP_SEGMENT
