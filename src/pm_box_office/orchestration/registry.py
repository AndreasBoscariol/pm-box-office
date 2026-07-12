"""Source registry for local ingest orchestration."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo


BOX_OFFICE_PREDICTION_SOURCE_KEYS = (
    "boxofficepro",
    "boxofficereport",
    "boxofficetheory",
    "boxofficetheory_substack",
    "edwarddouglas_substack",
    "boxofficeguru",
    "toddmthatcher",
    "joblo",
)

AUTORUN_TIMEZONE = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class SourcePollingWindow:
    """A local-time window in which a source is checked at a fixed cadence."""

    weekdays: tuple[int, ...]  # ISO weekdays: Monday=1 through Sunday=7.
    start: dt.time
    end: dt.time
    interval_minutes: int = 15


# These windows come from the publication timestamps already captured in the
# database.  They deliberately include a small contingency window around the
# normal publication time; a source is polled shortly after publication rather
# than waiting for one all-sources batch later that night.
SOURCE_POLLING_WINDOWS: dict[str, tuple[SourcePollingWindow, ...]] = {
    # Market metadata is live state rather than a once-per-day publication.
    # Keep the validated contract grid current so a completed sync can enqueue
    # a forecast refresh and the UI can immediately prefer real contract
    # buckets over a synthetic grid.
    "polymarket_metadata": (
        SourcePollingWindow((1, 2, 3, 4, 5, 6, 7), dt.time(0), dt.time(23, 59), 15),
    ),
    # 117/124 recent Weekend Preview articles were published Wednesday, mainly
    # from noon through 5pm ET. Tuesday/Thursday cover holiday-week exceptions.
    "boxofficepro": (
        SourcePollingWindow((2,), dt.time(12), dt.time(19)),
        SourcePollingWindow((3,), dt.time(9), dt.time(19)),
        SourcePollingWindow((4,), dt.time(12), dt.time(18)),
    ),
    # Recent forecast pages land late Wednesday, throughout Thursday, or just
    # after midnight Friday.
    "boxofficereport": (
        SourcePollingWindow((3,), dt.time(19), dt.time(23, 45)),
        SourcePollingWindow((4,), dt.time(10), dt.time(23, 45)),
        SourcePollingWindow((5,), dt.time(0), dt.time(1, 30)),
    ),
    # The public site and Substack mirror have no single dependable hour; most
    # forecast posts arrive Tuesday-Friday during these editorial windows.
    "boxofficetheory": (
        SourcePollingWindow((2,), dt.time(11), dt.time(20), 30),
        SourcePollingWindow((3,), dt.time(10), dt.time(20), 30),
        SourcePollingWindow((4,), dt.time(10), dt.time(20), 30),
        SourcePollingWindow((5,), dt.time(0), dt.time(20), 30),
    ),
    "boxofficetheory_substack": (
        SourcePollingWindow((2,), dt.time(11), dt.time(20), 30),
        SourcePollingWindow((3,), dt.time(10), dt.time(20), 30),
        SourcePollingWindow((4,), dt.time(10), dt.time(20), 30),
        SourcePollingWindow((5,), dt.time(0), dt.time(20), 30),
    ),
    # Recent Weekend Warrior posts with box-office tables have landed mostly
    # Tuesday/Wednesday ET, with some spillover into Thursday.
    "edwarddouglas_substack": (
        SourcePollingWindow((2,), dt.time(8), dt.time(22), 30),
        SourcePollingWindow((3,), dt.time(8), dt.time(22), 30),
        SourcePollingWindow((4,), dt.time(8), dt.time(18), 30),
    ),
    # BoxOfficeGuru exposes weekly archive pages but not a reliable publish
    # timestamp. Check the likely pre-weekend and post-weekend appearance
    # windows, then stop for the day once the page has landed.
    "boxofficeguru": (
        SourcePollingWindow((4,), dt.time(10), dt.time(23), 60),
        SourcePollingWindow((5,), dt.time(8), dt.time(23), 60),
        SourcePollingWindow((7,), dt.time(10), dt.time(23), 60),
        SourcePollingWindow((1,), dt.time(8), dt.time(18), 60),
    ),
    # Todd M. Thatcher's weekly roundups have recently posted Tuesday evening
    # ET, while single-title forecasts can appear from Friday through Thursday.
    "toddmthatcher": (
        SourcePollingWindow((5, 6, 7, 1), dt.time(8), dt.time(22), 60),
        SourcePollingWindow((2,), dt.time(8), dt.time(23), 30),
        SourcePollingWindow((3,), dt.time(8), dt.time(23), 30),
        SourcePollingWindow((4,), dt.time(8), dt.time(20), 60),
    ),
    # JoBlo's weekend box-office feed is regular for Sunday results; prediction
    # articles are less regular, so keep a Thursday/Friday pre-weekend check.
    "joblo": (
        SourcePollingWindow((4,), dt.time(10), dt.time(23), 30),
        SourcePollingWindow((5,), dt.time(8), dt.time(18), 30),
        SourcePollingWindow((7,), dt.time(10), dt.time(18), 60),
    ),
    # Daily charts are posted once reporting data arrives, not at a dependable
    # hour. Refresh the recent chart dates throughout the reporting day.
    "the_numbers": (
        SourcePollingWindow((1, 2, 3, 4, 5, 6, 7), dt.time(10), dt.time(23, 30), 30),
    ),
    # Weekend prediction articles are normally published on Friday afternoon.
    "the_numbers_predictions": (
        SourcePollingWindow((5,), dt.time(12), dt.time(23), 30),
    ),
}


# Publication-oriented sources stop polling once their daily article/chart has
# arrived. Polymarket metadata represents a continuously changing market and
# must continue to sync for the rest of the day.
CONTINUOUS_POLL_SOURCE_KEYS = frozenset({"polymarket_metadata"})


def source_poll_due(source_key: str, now: dt.datetime) -> bool:
    """Return whether ``source_key`` is due at this source's next poll slot."""
    local_now = now.astimezone(AUTORUN_TIMEZONE)
    for window in SOURCE_POLLING_WINDOWS.get(source_key, ()):
        if (
            local_now.isoweekday() in window.weekdays
            and window.start <= local_now.time().replace(second=0, microsecond=0) <= window.end
            and local_now.minute % window.interval_minutes == 0
        ):
            return True
    return False


@dataclass(frozen=True)
class SourceDefinition:
    source_key: str
    display_name: str
    command: str
    default_args: tuple[str, ...] = ()
    max_concurrency: int = 1
    enabled: bool = True
    requires_movies: bool = False
    include_in_run_all: bool = True


SOURCE_DEFINITIONS: tuple[SourceDefinition, ...] = (
    SourceDefinition(
        source_key="the_numbers",
        display_name="The Numbers",
        command="pm_box_office.sources.the_numbers.ingest",
    ),
    SourceDefinition(
        source_key="the_numbers_predictions",
        display_name="The Numbers Predictions",
        command="pm_box_office.sources.the_numbers.predictions",
        include_in_run_all=False,
    ),
    SourceDefinition(
        source_key="boxofficepro",
        display_name="Boxoffice Pro Predictions",
        command="pm_box_office.sources.boxofficepro.ingest",
    ),
    SourceDefinition(
        source_key="boxofficereport",
        display_name="Box Office Report Predictions",
        command="pm_box_office.sources.boxofficereport.ingest",
    ),
    SourceDefinition(
        source_key="boxofficetheory",
        display_name="Box Office Theory Predictions",
        command="pm_box_office.sources.boxofficetheory.ingest",
    ),
    SourceDefinition(
        source_key="boxofficetheory_substack",
        display_name="Box Office Theory Substack Predictions",
        command="pm_box_office.sources.boxofficetheory_substack.ingest",
    ),
    SourceDefinition(
        source_key="edwarddouglas_substack",
        display_name="Edward Douglas Substack Predictions",
        command="pm_box_office.sources.edwarddouglas_substack.ingest",
    ),
    SourceDefinition(
        source_key="boxofficeguru",
        display_name="Box Office Guru Predictions",
        command="pm_box_office.sources.boxofficeguru.ingest",
    ),
    SourceDefinition(
        source_key="toddmthatcher",
        display_name="Todd M. Thatcher Predictions",
        command="pm_box_office.sources.toddmthatcher.ingest",
    ),
    SourceDefinition(
        source_key="joblo",
        display_name="JoBlo Predictions",
        command="pm_box_office.sources.joblo.ingest",
    ),
    SourceDefinition(
        source_key="wikipedia",
        display_name="Wikipedia Activity",
        command="pm_box_office.sources.wikipedia.ingest",
        requires_movies=True,
    ),
    SourceDefinition(
        source_key="audience",
        display_name="Audience Snapshots",
        command="pm_box_office.sources.audience.ingest",
        requires_movies=True,
    ),
    SourceDefinition(
        source_key="rotten_tomatoes",
        display_name="Rotten Tomatoes Critics",
        command="pm_box_office.sources.rotten_tomatoes.ingest",
        requires_movies=True,
    ),
    SourceDefinition(
        source_key="polymarket_metadata",
        display_name="Polymarket Market Metadata",
        command="pm_box_office.sources.polymarket.ingest",
        requires_movies=True,
        include_in_run_all=False,
    ),
    SourceDefinition(
        source_key="amc_worker",
        display_name="AMC Worker Batch",
        command="pm_box_office.sources.amc.jobs.worker",
        default_args=("--once", "--limit", "1", "--worker-id", "orchestrated-amc", "--verbose"),
        include_in_run_all=False,
    ),
)


SOURCE_BY_KEY = {source.source_key: source for source in SOURCE_DEFINITIONS}
RUN_ALL_SOURCE_KEYS = tuple(
    source.source_key
    for source in SOURCE_DEFINITIONS
    if source.enabled and source.include_in_run_all
)

# Manual "Run all" remains comprehensive.  Automatic prediction polling is
# source-specific, so the nightly job only handles non-prediction maintenance.
DAILY_BACKGROUND_SOURCE_KEYS = tuple(source_key for source_key in RUN_ALL_SOURCE_KEYS if source_key not in SOURCE_POLLING_WINDOWS)
