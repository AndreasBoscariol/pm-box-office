"""Timezone-aware AMC seat snapshot scheduling."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from pm_box_office.sources.amc.db import StoredShowtime, StoredTheatre
from pm_box_office.sources.amc.timezones import UTC, now_in_timezone


DEFAULT_OFFSETS_MINUTES = (360, 120, 30, 5)


@dataclass(frozen=True)
class ScheduledSnapshot:
    showtime_id: str
    minutes_before_showtime: int
    due_utc_at: dt.datetime
    due_local_at: dt.datetime


def scheduled_snapshots(
    showtime: StoredShowtime,
    *,
    offsets_minutes: tuple[int, ...] = DEFAULT_OFFSETS_MINUTES,
) -> list[ScheduledSnapshot]:
    local_start = showtime.local_start_at
    if local_start.tzinfo is None:
        local_start = local_start.replace(tzinfo=ZoneInfo(showtime.timezone))
    utc_start = local_start.astimezone(UTC)
    snapshots: list[ScheduledSnapshot] = []
    for minutes in offsets_minutes:
        due_utc = utc_start - dt.timedelta(minutes=minutes)
        snapshots.append(
            ScheduledSnapshot(
                showtime_id=showtime.showtime_id,
                minutes_before_showtime=minutes,
                due_utc_at=due_utc,
                due_local_at=due_utc.astimezone(ZoneInfo(showtime.timezone)),
            )
        )
    return snapshots


def due_snapshots(
    showtimes: list[StoredShowtime],
    *,
    now_utc: dt.datetime,
    offsets_minutes: tuple[int, ...] = DEFAULT_OFFSETS_MINUTES,
    grace_minutes: int = 20,
) -> list[tuple[StoredShowtime, ScheduledSnapshot]]:
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=UTC)
    due: list[tuple[StoredShowtime, ScheduledSnapshot]] = []
    grace = dt.timedelta(minutes=grace_minutes)
    for showtime in showtimes:
        for snapshot in scheduled_snapshots(showtime, offsets_minutes=offsets_minutes):
            if snapshot.due_utc_at <= now_utc <= snapshot.due_utc_at + grace:
                due.append((showtime, snapshot))
    return sorted(due, key=lambda item: (item[1].due_utc_at, item[0].showtime_id))


def theatres_in_local_morning(
    theatres: list[StoredTheatre],
    *,
    now_utc: dt.datetime,
    start_hour: int,
    end_hour: int,
) -> list[StoredTheatre]:
    return [
        theatre
        for theatre in theatres
        if start_hour <= now_in_timezone(now_utc, theatre.timezone).hour < end_hour
    ]
