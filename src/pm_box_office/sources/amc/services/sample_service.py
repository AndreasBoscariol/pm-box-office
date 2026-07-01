"""Persistent AMC theatre sample service."""

from __future__ import annotations

import datetime as dt
from typing import Any

from pm_box_office.sources.amc import db
from pm_box_office.sources.amc.sampling import fixed_theatre_sample


DEFAULT_SAMPLE_KEY = "balanced_175"
DEFAULT_SAMPLE_SIZE = 175
DEFAULT_CERTAINTY_COUNT = 30
DEFAULT_SAMPLE_SEED = "amc-fixed-balanced-175-v1"


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
    members = fixed_theatre_sample(
        frame,
        sample_size=sample_size,
        certainty_count=certainty_count,
        seed=seed,
    )
    frame_showtime_count = sum(int(theatre.observed_showtime_count or 0) for theatre in frame)
    sample_set_id = db.create_theatre_sample_set(
        conn,
        sample_key=sample_key,
        sample_size=sample_size,
        certainty_count=certainty_count,
        seed=seed,
        frame_theatre_count=len(frame),
        frame_showtime_count=frame_showtime_count,
        notes="Fixed AMC theatre sample for recurring seat-map collection; US AMC frame only.",
    )
    db.replace_theatre_sample_members(conn, sample_set_id=sample_set_id, members=members)
    created = db.select_theatre_sample_set(conn, sample_key)
    if created is None:
        raise RuntimeError(f"Could not reload AMC theatre sample set {sample_key!r}")
    return created


def sample_members(conn: Any, sample_set: db.TheatreSampleSet) -> list[db.TheatreSampleMember]:
    return db.select_theatre_sample_members(conn, sample_set.sample_set_id)


def sample_coverage(conn: Any, *, sample_set: db.TheatreSampleSet, exhibition_date: dt.date) -> dict[str, object]:
    return db.theatre_sample_coverage(
        conn,
        sample_set_id=sample_set.sample_set_id,
        exhibition_date=exhibition_date,
    )
