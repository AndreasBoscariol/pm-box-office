"""Source registry for local ingest orchestration."""

from __future__ import annotations

from dataclasses import dataclass


BOX_OFFICE_PREDICTION_SOURCE_KEYS = (
    "boxofficepro",
    "boxofficereport",
    "boxofficetheory",
    "boxofficeguru",
)


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
        source_key="boxofficeguru",
        display_name="Box Office Guru Predictions",
        command="pm_box_office.sources.boxofficeguru.ingest",
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
