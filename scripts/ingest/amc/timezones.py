"""Timezone inference and conversion helpers for AMC theatres."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo


UTC = ZoneInfo("UTC")

STATE_TIMEZONE = {
    "AL": "America/Chicago",
    "AK": "America/Anchorage",
    "AR": "America/Chicago",
    "AZ": "America/Phoenix",
    "CA": "America/Los_Angeles",
    "CO": "America/Denver",
    "CT": "America/New_York",
    "DC": "America/New_York",
    "DE": "America/New_York",
    "GA": "America/New_York",
    "HI": "Pacific/Honolulu",
    "IA": "America/Chicago",
    "IL": "America/Chicago",
    "IN": "America/Indiana/Indianapolis",
    "KS": "America/Chicago",
    "LA": "America/Chicago",
    "MA": "America/New_York",
    "MD": "America/New_York",
    "ME": "America/New_York",
    "MI": "America/Detroit",
    "MN": "America/Chicago",
    "MO": "America/Chicago",
    "MS": "America/Chicago",
    "MT": "America/Denver",
    "NC": "America/New_York",
    "ND": "America/Chicago",
    "NE": "America/Chicago",
    "NH": "America/New_York",
    "NJ": "America/New_York",
    "NM": "America/Denver",
    "NV": "America/Los_Angeles",
    "NY": "America/New_York",
    "OH": "America/New_York",
    "OK": "America/Chicago",
    "OR": "America/Los_Angeles",
    "PA": "America/New_York",
    "RI": "America/New_York",
    "SC": "America/New_York",
    "SD": "America/Chicago",
    "TN": "America/Chicago",
    "TX": "America/Chicago",
    "UT": "America/Denver",
    "VA": "America/New_York",
    "VT": "America/New_York",
    "WA": "America/Los_Angeles",
    "WI": "America/Chicago",
    "WV": "America/New_York",
    "WY": "America/Denver",
}


def infer_us_timezone(latitude: float | None, longitude: float | None, state: str | None) -> str:
    """Infer a practical theatre timezone from sitemap coordinates and state.

    AMC's sitemap gives coordinates but not an IANA timezone. This avoids a
    heavy geospatial dependency while handling the common US split-state cases
    well enough for scheduler safety.
    """

    state_code = (state or "").strip().upper()
    if state_code == "HI":
        return "Pacific/Honolulu"
    if state_code == "AK":
        if longitude is not None and longitude < -169:
            return "America/Adak"
        return "America/Anchorage"
    if state_code == "AZ":
        return "America/Phoenix"
    if state_code in {"ID", "OR"} and longitude is not None and longitude > -116.5:
        return "America/Denver"
    if state_code == "FL" and longitude is not None and longitude < -85.1:
        return "America/Chicago"
    if state_code == "IN" and longitude is not None and longitude < -86.8:
        return "America/Chicago"
    if state_code == "KS" and longitude is not None and longitude < -101.0:
        return "America/Denver"
    if state_code == "KY" and longitude is not None and longitude < -85.2:
        return "America/Chicago"
    if state_code == "MI" and longitude is not None and longitude < -87.0:
        return "America/Chicago"
    if state_code == "ND" and longitude is not None and longitude < -101.0:
        return "America/Denver"
    if state_code == "NE" and longitude is not None and longitude < -101.0:
        return "America/Denver"
    if state_code == "SD" and longitude is not None and longitude < -100.3:
        return "America/Denver"
    if state_code == "TN" and longitude is not None and longitude > -84.9:
        return "America/New_York"
    if state_code == "TX" and longitude is not None and longitude < -104.6:
        return "America/Denver"
    if state_code in STATE_TIMEZONE:
        return STATE_TIMEZONE[state_code]
    return infer_timezone_from_longitude(longitude)


def infer_timezone_from_longitude(longitude: float | None) -> str:
    if longitude is None:
        return "America/New_York"
    if longitude <= -125:
        return "America/Anchorage"
    if longitude <= -115:
        return "America/Los_Angeles"
    if longitude <= -100:
        return "America/Denver"
    if longitude <= -85:
        return "America/Chicago"
    return "America/New_York"


def parse_showtime_to_local_and_utc(value: str, theatre_timezone: str) -> tuple[dt.datetime, dt.datetime]:
    raw = value.replace("Z", "+00:00")
    parsed = dt.datetime.fromisoformat(raw)
    theatre_zone = ZoneInfo(theatre_timezone)
    if parsed.tzinfo is None:
        local_dt = parsed.replace(tzinfo=theatre_zone)
    else:
        local_dt = parsed.astimezone(theatre_zone)
    return local_dt, local_dt.astimezone(UTC)


def now_in_timezone(now_utc: dt.datetime, timezone_name: str) -> dt.datetime:
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=UTC)
    return now_utc.astimezone(ZoneInfo(timezone_name))
