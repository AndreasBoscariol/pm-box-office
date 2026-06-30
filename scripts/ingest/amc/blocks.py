"""Theatrical business-day block helpers for AMC showtimes."""

from __future__ import annotations

import datetime as dt


PREMIUM_ATTRIBUTE_KEYWORDS = (
    "dolby",
    "imax",
    "prime",
    "laser",
    "real 3d",
    "3d",
    "screenx",
    "d-box",
)


def business_minute_for_showtime(
    local_start_at: dt.datetime,
    *,
    exhibition_date: dt.date,
) -> int:
    """Return sortable minute within the AMC theatrical business day.

    AMC may list a 12:30 AM calendar-time show on the prior exhibition date.
    In that case the value becomes 1470 rather than 30, preserving its place at
    the end of the prior business day.
    """

    day_delta = (local_start_at.date() - exhibition_date).days
    return day_delta * 1440 + local_start_at.hour * 60 + local_start_at.minute


def showtime_block_for_business_minute(business_minute: int) -> int:
    if business_minute < 900:
        return 1
    if business_minute < 1080:
        return 2
    if business_minute < 1260:
        return 3
    return 4


def is_premium_attribute_set(attributes: list[str]) -> bool:
    text = " ".join(attributes).lower()
    return any(keyword in text for keyword in PREMIUM_ATTRIBUTE_KEYWORDS)
