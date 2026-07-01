"""Theatre size bucketing and deterministic AMC theatre sample selection."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from pm_box_office.sources.amc.db import StoredTheatre


@dataclass(frozen=True)
class TheatreSample:
    theatre: StoredTheatre
    size_bucket: str
    stratum: str


@dataclass(frozen=True)
class FixedTheatreSample:
    theatre: StoredTheatre
    size_bucket: str
    stratum: str
    is_certainty: bool
    inclusion_probability: float
    analysis_weight: float
    selection_rank: int


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


def fixed_theatre_sample(
    theatres: Iterable[StoredTheatre],
    *,
    sample_size: int = 175,
    certainty_count: int = 30,
    seed: str = "amc-fixed-balanced-175-v1",
) -> list[FixedTheatreSample]:
    """Select a stable fixed sample for recurring seat-map collection.

    The selection is deterministic for a given theatre frame. Large/high-volume
    theatres are certainty units; the rest are allocated proportionally across
    timezone/size strata and balanced by state inside each stratum.
    """

    frame = sorted(theatres, key=lambda theatre: theatre.amc_theatre_id)
    if sample_size <= 0 or not frame:
        return []
    if sample_size >= len(frame):
        return [
            FixedTheatreSample(
                theatre=theatre,
                size_bucket=size_bucket(theatre),
                stratum=stratum_for(theatre, size_bucket(theatre)),
                is_certainty=True,
                inclusion_probability=1.0,
                analysis_weight=1.0,
                selection_rank=index,
            )
            for index, theatre in enumerate(frame, start=1)
        ]

    certainty_target = min(certainty_count, sample_size, len(frame))
    certainty_theatres = sorted(
        frame,
        key=lambda theatre: (
            -theatre_volume(theatre),
            -(theatre.inferred_screen_count or 0),
            theatre.amc_theatre_id,
        ),
    )[:certainty_target]
    certainty_ids = {theatre.amc_theatre_id for theatre in certainty_theatres}
    remainder = [theatre for theatre in frame if theatre.amc_theatre_id not in certainty_ids]
    remainder_slots = sample_size - len(certainty_theatres)

    quotas = stratum_quotas(remainder, sample_size=remainder_slots)
    selected: list[FixedTheatreSample] = []
    for rank, theatre in enumerate(
        sorted(certainty_theatres, key=lambda item: item.amc_theatre_id),
        start=1,
    ):
        bucket = size_bucket(theatre)
        selected.append(
            FixedTheatreSample(
                theatre=theatre,
                size_bucket=bucket,
                stratum=stratum_for(theatre, bucket),
                is_certainty=True,
                inclusion_probability=1.0,
                analysis_weight=1.0,
                selection_rank=rank,
            )
        )

    next_rank = len(selected) + 1
    for stratum, stratum_theatres in grouped_by_stratum(remainder).items():
        quota = quotas.get(stratum, 0)
        if quota <= 0:
            continue
        ranked = state_balanced_rank(stratum_theatres, seed=seed)
        probability = min(1.0, quota / len(stratum_theatres))
        weight = 1.0 / probability if probability else 0.0
        for theatre in ranked[:quota]:
            bucket = size_bucket(theatre)
            selected.append(
                FixedTheatreSample(
                    theatre=theatre,
                    size_bucket=bucket,
                    stratum=stratum,
                    is_certainty=False,
                    inclusion_probability=probability,
                    analysis_weight=weight,
                    selection_rank=next_rank,
                )
            )
            next_rank += 1

    if len(selected) < sample_size:
        selected_ids = {item.theatre.amc_theatre_id for item in selected}
        leftovers = [theatre for theatre in remainder if theatre.amc_theatre_id not in selected_ids]
        for theatre in sorted(leftovers, key=lambda item: deterministic_rank(item.amc_theatre_id, seed)):
            if len(selected) >= sample_size:
                break
            bucket = size_bucket(theatre)
            stratum = stratum_for(theatre, bucket)
            selected.append(
                FixedTheatreSample(
                    theatre=theatre,
                    size_bucket=bucket,
                    stratum=stratum,
                    is_certainty=False,
                    inclusion_probability=1.0,
                    analysis_weight=1.0,
                    selection_rank=next_rank,
                )
            )
            next_rank += 1

    return sorted(selected[:sample_size], key=lambda item: item.theatre.amc_theatre_id)


def theatre_volume(theatre: StoredTheatre) -> int:
    return max(
        int(theatre.observed_showtime_count or 0),
        int(theatre.inferred_screen_count or 0),
        int(theatre.median_total_seats or 0),
        1,
    )


def grouped_by_stratum(theatres: Iterable[StoredTheatre]) -> dict[str, list[StoredTheatre]]:
    strata: dict[str, list[StoredTheatre]] = defaultdict(list)
    for theatre in theatres:
        bucket = size_bucket(theatre)
        strata[stratum_for(theatre, bucket)].append(theatre)
    return dict(sorted(strata.items()))


def stratum_quotas(theatres: Iterable[StoredTheatre], *, sample_size: int) -> dict[str, int]:
    strata = grouped_by_stratum(theatres)
    if sample_size <= 0 or not strata:
        return {}
    if sample_size >= sum(len(items) for items in strata.values()):
        return {stratum: len(items) for stratum, items in strata.items()}

    volumes = {
        stratum: sum(theatre_volume(theatre) for theatre in items)
        for stratum, items in strata.items()
    }
    total_volume = sum(volumes.values()) or len(strata)
    if sample_size < len(strata):
        selected_strata = {
            stratum
            for stratum, _volume in sorted(
                volumes.items(),
                key=lambda item: (-item[1], item[0]),
            )[:sample_size]
        }
        return {stratum: (1 if stratum in selected_strata else 0) for stratum in strata}

    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for stratum, items in strata.items():
        raw_quota = sample_size * volumes[stratum] / total_volume
        quota = min(len(items), max(1, int(raw_quota)))
        quotas[stratum] = quota
        remainders.append((raw_quota - int(raw_quota), stratum))

    while sum(quotas.values()) > sample_size:
        removable = [
            (remainders_by_stratum(remainders)[stratum], stratum)
            for stratum, quota in quotas.items()
            if quota > 1
        ]
        if not removable:
            break
        _remainder, stratum = min(removable)
        quotas[stratum] -= 1

    while sum(quotas.values()) < sample_size:
        added = False
        for _remainder, stratum in sorted(remainders, reverse=True):
            if quotas[stratum] < len(strata[stratum]):
                quotas[stratum] += 1
                added = True
                break
        if not added:
            break

    return quotas


def remainders_by_stratum(remainders: list[tuple[float, str]]) -> dict[str, float]:
    return {stratum: remainder for remainder, stratum in remainders}


def state_balanced_rank(theatres: Iterable[StoredTheatre], *, seed: str) -> list[StoredTheatre]:
    theatre_rows = list(theatres)
    by_state: dict[str, list[StoredTheatre]] = defaultdict(list)
    for theatre in theatre_rows:
        by_state[theatre.state or ""].append(theatre)
    for state, items in by_state.items():
        by_state[state] = sorted(
            items,
            key=lambda theatre: deterministic_rank(theatre.amc_theatre_id, f"{seed}:{state}"),
        )

    ranked: list[StoredTheatre] = []
    states = sorted(
        by_state,
        key=lambda state: deterministic_rank(len(by_state[state]), f"{seed}:{state}"),
    )
    while states:
        next_states: list[str] = []
        for state in states:
            items = by_state[state]
            if items:
                ranked.append(items.pop(0))
            if items:
                next_states.append(state)
        states = next_states
    return ranked
