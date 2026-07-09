"""Persistent AMC theatre sample service."""

from __future__ import annotations

import datetime as dt
from typing import Any

from pm_box_office.sources.amc import db
from pm_box_office.sources.amc.sampling import FixedTheatreSample, fixed_theatre_sample, size_bucket, stratum_for


TOP_HYBRID_SAMPLE_KEY = "top_hybrid_30"
TOP_HYBRID_THEATRES = (
    ("AMC Streets Of St Charles 8", "MO"),
    ("AMC Fresh Meadows 7", "NY"),
    ("AMC Thoroughbred 20", "TN"),
    ("AMC Stones River 9", "TN"),
    ("AMC DINE-IN Berkshire 8", "PA"),
    ("AMC Town Square 18", "NV"),
    ("AMC Palm Promenade 24", "CA"),
    ("AMC Deer Valley 17", "AZ"),
    ("AMC Roosevelt Collection 16", "IL"),
    ("AMC Tysons Corner 16", "VA"),
    ("AMC Madison Yards 8", "GA"),
    ("AMC Springfield 11", "MO"),
    ("AMC Riverview 14", "FL"),
    ("AMC CLASSIC South Bend 16", "IN"),
    ("AMC Bayou 15", "FL"),
    ("AMC Orange 30", "CA"),
    ("AMC Altamonte Mall 18", "FL"),
    ("AMC Columbus 10", "OH"),
    ("AMC Glendora 12 @ 210/57", "CA"),
    ("AMC Woodlands Square 20", "FL"),
    ("AMC Shirlington 7", "VA"),
    ("AMC Foothills 12", "TN"),
    ("AMC Rainbow Promenade 10", "NV"),
    ("AMC Ontario Mills 30", "CA"),
    ("AMC The Grove 14", "CA"),
    ("AMC Plainville 20", "CT"),
    ("AMC DINE-IN Essex Green 9", "NJ"),
    ("AMC Factoria 8", "WA"),
    ("AMC Livonia 20", "MI"),
    ("AMC Pembroke Lakes 9", "FL"),
)

DEFAULT_SAMPLE_KEY = TOP_HYBRID_SAMPLE_KEY
DEFAULT_SAMPLE_SIZE = len(TOP_HYBRID_THEATRES)
DEFAULT_CERTAINTY_COUNT = len(TOP_HYBRID_THEATRES)
DEFAULT_SAMPLE_SEED = "amc-top-hybrid-30-v1"


def ensure_default_theatre_sample(
    conn: Any,
    *,
    sample_key: str = DEFAULT_SAMPLE_KEY,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    certainty_count: int = DEFAULT_CERTAINTY_COUNT,
    seed: str = DEFAULT_SAMPLE_SEED,
) -> db.TheatreSampleSet:
    existing = db.select_theatre_sample_set(conn, sample_key)
    if existing is not None:
        return existing

    frame = db.select_theatre_sample_frame(conn)
    if not frame:
        raise ValueError("Cannot create AMC theatre sample before active theatres are loaded.")
    members = theatre_sample_members_for_key(
        frame,
        sample_key=sample_key,
        sample_size=sample_size,
        certainty_count=certainty_count,
        seed=seed,
    )
    frame_showtime_count = sum(int(theatre.observed_showtime_count or 0) for theatre in frame)
    sample_set_id = db.create_theatre_sample_set(
        conn,
        sample_key=sample_key,
        sample_size=len(members),
        certainty_count=sum(1 for member in members if member.is_certainty),
        seed=seed,
        frame_theatre_count=len(frame),
        frame_showtime_count=frame_showtime_count,
        notes=theatre_sample_notes(sample_key),
    )
    db.replace_theatre_sample_members(conn, sample_set_id=sample_set_id, members=members)
    created = db.select_theatre_sample_set(conn, sample_key)
    if created is None:
        raise RuntimeError(f"Could not reload AMC theatre sample set {sample_key!r}")
    return created


def theatre_sample_members_for_key(
    frame: list[db.StoredTheatre],
    *,
    sample_key: str,
    sample_size: int,
    certainty_count: int,
    seed: str,
) -> list[FixedTheatreSample]:
    if sample_key == TOP_HYBRID_SAMPLE_KEY:
        return top_hybrid_theatre_sample(frame)
    return fixed_theatre_sample(
        frame,
        sample_size=sample_size,
        certainty_count=certainty_count,
        seed=seed,
    )


def top_hybrid_theatre_sample(frame: list[db.StoredTheatre]) -> list[FixedTheatreSample]:
    theatres_by_key = {
        (normalized_theatre_name(theatre.name), theatre.state.upper()): theatre
        for theatre in frame
    }
    members: list[FixedTheatreSample] = []
    missing: list[str] = []
    for rank, (name, state) in enumerate(TOP_HYBRID_THEATRES, start=1):
        theatre = theatres_by_key.get((normalized_theatre_name(name), state))
        if theatre is None:
            missing.append(f"{name}, {state}")
            continue
        bucket = size_bucket(theatre)
        members.append(
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
    if missing:
        missing_list = "; ".join(missing)
        raise ValueError(
            f"Cannot create AMC top-hybrid theatre sample; missing active theatres: {missing_list}"
        )
    return members


def normalized_theatre_name(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def theatre_sample_notes(sample_key: str) -> str:
    if sample_key == TOP_HYBRID_SAMPLE_KEY:
        return "Fixed AMC top-hybrid 30 theatre universe for recurring seat-map collection."
    return "Fixed AMC theatre sample for recurring seat-map collection; US AMC frame only."


def sample_members(conn: Any, sample_set: db.TheatreSampleSet) -> list[db.TheatreSampleMember]:
    return db.select_theatre_sample_members(conn, sample_set.sample_set_id)


def sample_coverage(conn: Any, *, sample_set: db.TheatreSampleSet, exhibition_date: dt.date) -> dict[str, object]:
    return db.theatre_sample_coverage(
        conn,
        sample_set_id=sample_set.sample_set_id,
        exhibition_date=exhibition_date,
    )
