"""Shared constants for the production box-office forecast layer."""

from __future__ import annotations

PRE_RELEASE_REGIME = "pre_release"
LIVE_FRIDAY_REGIME = "live_friday"
LIVE_SATURDAY_REGIME = "live_saturday"
LIVE_SUNDAY_REGIME = "live_sunday"

REGIME_PREFIX = {
    PRE_RELEASE_REGIME: "P",
    LIVE_FRIDAY_REGIME: "FRI",
    LIVE_SATURDAY_REGIME: "SAT",
    LIVE_SUNDAY_REGIME: "SUN",
}

LIVE_REGIMES = (LIVE_FRIDAY_REGIME, LIVE_SATURDAY_REGIME, LIVE_SUNDAY_REGIME)
REGIMES = (PRE_RELEASE_REGIME,) + LIVE_REGIMES

LIVE_ORIGINS = ("10:00", "12:00", "14:00", "16:00", "18:00", "20:00", "EOD")
PRE_RELEASE_ORIGIN_DAYS = tuple(range(-14, 0))

DAYS = ("Friday", "Saturday", "Sunday")
DAY_TO_TARGET = {
    "Friday": "friday",
    "Saturday": "saturday",
    "Sunday": "sunday",
}
TARGET_TO_DAY = {value: key for key, value in DAY_TO_TARGET.items()}

REGIME_DAY_OFFSET = {
    LIVE_FRIDAY_REGIME: 0,
    LIVE_SATURDAY_REGIME: 1,
    LIVE_SUNDAY_REGIME: 2,
}

PLUGIN_TARGET_DAY = {
    LIVE_FRIDAY_REGIME: "Friday",
    LIVE_SATURDAY_REGIME: "Saturday",
    LIVE_SUNDAY_REGIME: "Sunday",
}

KNOWN_ACTUAL_DAYS = {
    LIVE_FRIDAY_REGIME: frozenset(),
    LIVE_SATURDAY_REGIME: frozenset({"Friday"}),
    LIVE_SUNDAY_REGIME: frozenset({"Friday", "Saturday"}),
}

BASELINE_COLUMNS = {
    (LIVE_FRIDAY_REGIME, "Friday"): ("pre_fri_usd", "pre-weekend baseline"),
    (LIVE_FRIDAY_REGIME, "Saturday"): ("pre_sat_usd", "pre-weekend baseline"),
    (LIVE_FRIDAY_REGIME, "Sunday"): ("pre_sun_usd", "pre-weekend baseline"),
    (LIVE_SATURDAY_REGIME, "Saturday"): ("after_fri_sat_usd", "after-Friday baseline"),
    (LIVE_SATURDAY_REGIME, "Sunday"): ("after_fri_sun_usd", "after-Friday baseline"),
    (LIVE_SUNDAY_REGIME, "Sunday"): ("after_sat_sun_usd", "after-Saturday baseline"),
}

ACTUAL_COLUMNS = {
    "Friday": "actual_fri_usd",
    "Saturday": "actual_sat_usd",
    "Sunday": "actual_sun_usd",
}

FORECAST_TABLE = "analytics.movie_opening_weekend_forecasts"
COMPONENT_TABLE = "analytics.movie_forecast_components"
RUN_TABLE = "analytics.forecast_runs"

