"""Conservative event-to-movie matching."""

from __future__ import annotations

import datetime as dt
import difflib
from dataclasses import dataclass

from pm_box_office.domain.movies import normalize_title


@dataclass(frozen=True, slots=True)
class MovieCandidate:
    movie_id: int
    title: str
    release_date: dt.date | None


@dataclass(frozen=True, slots=True)
class Match:
    movie_id: int | None
    status: str
    score: float
    release_date_distance: int | None


def match_movie(title: str, opening_date: dt.date | None, candidates: list[MovieCandidate]) -> Match:
    normalized = normalize_title(title)
    ranked: list[tuple[float, int | None, MovieCandidate]] = []
    for candidate in candidates:
        similarity = difflib.SequenceMatcher(None, normalized, normalize_title(candidate.title)).ratio()
        distance = abs((candidate.release_date - opening_date).days) if candidate.release_date and opening_date else None
        date_score = 1.0 if distance is not None and distance <= 3 else (.5 if distance is None else 0.0)
        exact = normalized == normalize_title(candidate.title)
        ranked.append(((.8 if exact else .7) * similarity + (.2 if exact else .3) * date_score, distance, candidate))
    ranked.sort(key=lambda row: row[0], reverse=True)
    if not ranked or ranked[0][0] < .72:
        return Match(None, "rejected", ranked[0][0] if ranked else 0.0, ranked[0][1] if ranked else None)
    best = ranked[0]
    if len(ranked) > 1 and best[0] - ranked[1][0] < .03:
        return Match(best[2].movie_id, "ambiguous", best[0], best[1])
    exact = normalized == normalize_title(best[2].title)
    status = "auto_approved" if exact and best[1] is not None and best[1] <= 3 else "reviewable"
    return Match(best[2].movie_id, status, best[0], best[1])


def automated_update_allowed(existing_status: str | None, manual_override: bool) -> bool:
    return not manual_override and existing_status not in {"manually_approved", "manually_rejected"}

