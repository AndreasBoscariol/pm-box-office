"""Source registry for local ingest orchestration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceDefinition:
    source_key: str
    display_name: str
    command: str
    default_args: tuple[str, ...] = ()
    max_concurrency: int = 1
    enabled: bool = True
    requires_movies: bool = False


SOURCE_DEFINITIONS: tuple[SourceDefinition, ...] = (
    SourceDefinition(
        source_key="the_numbers",
        display_name="The Numbers",
        command="pm_box_office.sources.the_numbers.ingest",
    ),
    SourceDefinition(
        source_key="boxofficepro",
        display_name="Boxoffice Pro",
        command="pm_box_office.sources.boxofficepro.ingest",
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
        source_key="amc_worker",
        display_name="AMC Worker Batch",
        command="pm_box_office.sources.amc.jobs.worker",
        default_args=("--once", "--limit", "1", "--worker-id", "orchestrated-amc", "--verbose"),
    ),
)


SOURCE_BY_KEY = {source.source_key: source for source in SOURCE_DEFINITIONS}

