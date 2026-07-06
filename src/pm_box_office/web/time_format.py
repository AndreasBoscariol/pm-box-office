from __future__ import annotations

import datetime as dt
import os
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def display_timezone() -> dt.tzinfo:
    timezone_name = os.environ.get("PM_BOX_OFFICE_DISPLAY_TIMEZONE") or os.environ.get("TZ")
    if timezone_name:
        try:
            return ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            pass
    return dt.datetime.now().astimezone().tzinfo or dt.UTC


def coerce_datetime(value: object) -> dt.datetime | None:
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def time_ago(value: object) -> str:
    timestamp = coerce_datetime(value)
    if timestamp is None:
        return "Never"

    now = dt.datetime.now(timestamp.tzinfo or dt.UTC)
    if timestamp.tzinfo is None:
        now = now.replace(tzinfo=None)

    seconds = max(0, int((now - timestamp).total_seconds()))
    if seconds < 60:
        return "Just now"

    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"

    hours = minutes // 60
    if hours < 48:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"

    days = hours // 24
    return f"{days} day{'s' if days != 1 else ''} ago"


def duration_until(seconds: object) -> str:
    try:
        remaining = max(0, int(seconds))
    except (TypeError, ValueError):
        return "unknown"
    if remaining < 60:
        return "less than 1 minute"
    minutes = remaining // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours = minutes // 60
    if hours < 48:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    days = hours // 24
    extra_hours = hours % 24
    if extra_hours:
        return f"{days} day{'s' if days != 1 else ''}, {extra_hours} hour{'s' if extra_hours != 1 else ''}"
    return f"{days} day{'s' if days != 1 else ''}"


def local_clock_time(value: object) -> str:
    timestamp = coerce_datetime(value)
    if timestamp is None:
        return "-"
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=dt.UTC)
    local = timestamp.astimezone(display_timezone())
    return local.strftime("%I:%M %p").lstrip("0").lower()


def local_short_datetime(value: object) -> str:
    timestamp = coerce_datetime(value)
    if timestamp is None:
        return "-"
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=dt.UTC)
    local = timestamp.astimezone(display_timezone())
    today = dt.datetime.now(display_timezone()).date()
    clock = local.strftime("%I:%M %p").lstrip("0").lower()
    if local.date() == today:
        return clock
    return f"{local.strftime('%b')} {local.day} {clock}"
