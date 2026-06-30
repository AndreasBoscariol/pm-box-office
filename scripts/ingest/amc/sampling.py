"""Theatre size bucketing and stratified sample selection."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from scripts.ingest.amc.db import StoredTheatre


@dataclass(frozen=True)
class TheatreSample:
    theatre: StoredTheatre
    size_bucket: str
    stratum: str


def size_bucket(theatre: StoredTheatre) -> str:
    size_signal = (
        theatre.inferred_screen_count
        or theatre.median_total_seats
        or theatre.observed_showtime_count
        or 0
    )
    if size_signal >= 18:
        return "large"
    if size_signal >= 10:
        return "medium"
    return "small"


def stratified_sample(
    theatres: Iterable[StoredTheatre],
    *,
    sample_size: int,
    seed: str,
) -> list[TheatreSample]:
    candidates = [TheatreSample(theatre, size_bucket(theatre), "") for theatre in theatres]
    if sample_size <= 0 or sample_size >= len(candidates):
        return [
            TheatreSample(item.theatre, item.size_bucket, stratum_for(item.theatre, item.size_bucket))
            for item in sorted(candidates, key=lambda item: item.theatre.amc_theatre_id)
        ]

    strata: dict[str, list[TheatreSample]] = defaultdict(list)
    for item in candidates:
        stratum = stratum_for(item.theatre, item.size_bucket)
        strata[stratum].append(TheatreSample(item.theatre, item.size_bucket, stratum))

    selected: list[TheatreSample] = []
    sorted_strata = sorted(strata.items())
    for index, (_stratum, items) in enumerate(sorted_strata):
        remaining_slots = sample_size - len(selected)
        remaining_strata = len(sorted_strata) - index
        if remaining_slots <= 0:
            break
        quota = max(1, remaining_slots // remaining_strata)
        ranked = sorted(items, key=lambda item: deterministic_rank(item.theatre.amc_theatre_id, seed))
        selected.extend(ranked[:quota])

    if len(selected) < sample_size:
        selected_ids = {item.theatre.amc_theatre_id for item in selected}
        leftovers = [
            item
            for item in candidates
            if item.theatre.amc_theatre_id not in selected_ids
        ]
        ranked_leftovers = sorted(leftovers, key=lambda item: deterministic_rank(item.theatre.amc_theatre_id, seed))
        selected.extend(
            TheatreSample(item.theatre, item.size_bucket, stratum_for(item.theatre, item.size_bucket))
            for item in ranked_leftovers[: sample_size - len(selected)]
        )

    return sorted(selected[:sample_size], key=lambda item: item.theatre.amc_theatre_id)


def stratum_for(theatre: StoredTheatre, bucket: str) -> str:
    return f"{theatre.timezone}|{bucket}"


def deterministic_rank(theatre_id: int, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{theatre_id}".encode("utf-8")).hexdigest()
