"""PostgreSQL-backed task queue helpers."""

from __future__ import annotations

import urllib.error
from typing import Any

from pm_box_office.sources.amc import db
from pm_box_office.sources.amc import diagnostics
from pm_box_office.sources.amc.parsers import SeatMapUnavailable


SEAT_COLLECTION_THROTTLE_KEY = "seat_collection"
SEAT_BACKOFF_MIN_MAX_ATTEMPTS = 6
SEAT_BACKOFF_RETRY_DELAYS_SECONDS = (75, 90, 120, 180)
SEAT_BACKOFF_ERROR_MARKERS = (
    "Could not find showtime object in AMC RSC payload",
    "Could not find showtime object with seatingLayout in AMC RSC payload",
)
HTTP_BACKOFF_STATUSES = {429, 500, 502, 503, 504}


def claim_due_tasks(conn: Any, *, worker_id: str, limit: int = 20) -> list[db.CollectionTask]:
    return db.claim_due_tasks(conn, worker_id=worker_id, limit=limit)


def mark_succeeded(conn: Any, task_id: int) -> None:
    db.mark_task_succeeded(conn, task_id)


def schedule_retry_or_fail(conn: Any, task: db.CollectionTask, exc: Exception) -> None:
    retry_delay_seconds = retry_delay_seconds_for_failure(task, exc)
    minimum_max_attempts = minimum_max_attempts_for_failure(task, exc)
    if is_seat_backoff_failure(task, exc):
        observed_at = db.utc_now()
        previous_blocked_until = db.active_throttle_until(
            conn,
            SEAT_COLLECTION_THROTTLE_KEY,
            now=observed_at,
        )
        blocked_until = db.extend_throttle(
            conn,
            SEAT_COLLECTION_THROTTLE_KEY,
            delay_seconds=retry_delay_seconds,
            reason=f"{type(exc).__name__}: {diagnostics.short_error(exc)}",
            now=observed_at,
        )
        diagnostics.log_backoff_event(
            "shared_seat_collection_backoff",
            throttle_key=SEAT_COLLECTION_THROTTLE_KEY,
            throttle_scope="all_workers",
            workers_share_server_identity=True,
            backoff_observed_at=observed_at,
            previous_blocked_until=previous_blocked_until,
            blocked_until=blocked_until,
            retry_delay_seconds=retry_delay_seconds,
            wait_remaining_seconds=seconds_until(blocked_until, observed_at),
            previous_wait_remaining_seconds=seconds_until(previous_blocked_until, observed_at),
            throttle_extended_by_seconds=throttle_extension_seconds(
                previous_blocked_until,
                blocked_until,
                observed_at,
            ),
            error_type=type(exc).__name__,
            error_message=diagnostics.short_error(exc),
        )
    db.mark_task_failed(
        conn,
        task.task_id,
        exc=exc,
        retry_delay_seconds=retry_delay_seconds,
        minimum_max_attempts=minimum_max_attempts,
    )


def retry_delay_seconds_for_failure(task: db.CollectionTask, exc: Exception) -> int:
    if not is_seat_backoff_failure(task, exc):
        return 60
    delay_index = max(0, min(task.attempt_count - 1, len(SEAT_BACKOFF_RETRY_DELAYS_SECONDS) - 1))
    return SEAT_BACKOFF_RETRY_DELAYS_SECONDS[delay_index]


def minimum_max_attempts_for_failure(task: db.CollectionTask, exc: Exception) -> int | None:
    if is_seat_backoff_failure(task, exc):
        return SEAT_BACKOFF_MIN_MAX_ATTEMPTS
    return None


def seconds_until(timestamp: object, now: object) -> int | None:
    if timestamp is None:
        return None
    return max(0, int((db.ensure_utc(timestamp) - db.ensure_utc(now)).total_seconds()))


def throttle_extension_seconds(previous_blocked_until: object, blocked_until: object, now: object) -> int:
    baseline = db.ensure_utc(previous_blocked_until) if previous_blocked_until is not None else db.ensure_utc(now)
    return max(0, int((db.ensure_utc(blocked_until) - baseline).total_seconds()))


def is_seat_backoff_failure(task: db.CollectionTask, exc: Exception) -> bool:
    if task.task_type != "collect_seat_snapshot":
        return False
    if isinstance(exc, SeatMapUnavailable):
        return False
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in HTTP_BACKOFF_STATUSES
    message = str(exc)
    return any(marker in message for marker in SEAT_BACKOFF_ERROR_MARKERS)
