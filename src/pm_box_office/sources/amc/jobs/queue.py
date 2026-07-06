"""PostgreSQL-backed task queue helpers."""

from __future__ import annotations

import urllib.error
import datetime as dt
from typing import Any

from pm_box_office.sources.amc import db
from pm_box_office.sources.amc import diagnostics
from pm_box_office.sources.amc.parsers import SeatMapUnavailable


SEAT_COLLECTION_THROTTLE_KEY = "seat_collection"
SEAT_BACKOFF_MIN_MAX_ATTEMPTS = 6
SEAT_BACKOFF_RETRY_DELAYS_SECONDS = (75, 90, 120, 180)
EXPLORATORY_SEAT_CONTENT_RETRY_DELAY_SECONDS = 75
EXPLORATORY_SEAT_CONTENT_MAX_ATTEMPTS = 2
LATE_RESCUE_OFFSETS_MINUTES = (10, 30)
LATE_RESCUE_MAX_ATTEMPTS = 4
SEAT_BACKOFF_ERROR_MARKERS = (
    "Could not find showtime object in AMC RSC payload",
    "Could not find showtime object with seatingLayout in AMC RSC payload",
)
HTTP_BACKOFF_STATUSES = {429, 500, 502, 503, 504}
CONTENT_FAILURE_BACKOFF_CLASS = "exploratory_content_failure"
HTTP_FAILURE_BACKOFF_CLASS = "shared_http_backoff"


def claim_due_tasks(conn: Any, *, worker_id: str, limit: int = 20) -> list[db.CollectionTask]:
    return db.claim_due_tasks(conn, worker_id=worker_id, limit=limit)


def mark_succeeded(conn: Any, task_id: int) -> None:
    db.mark_task_succeeded(conn, task_id)


def schedule_retry_or_fail(conn: Any, task: db.CollectionTask, exc: Exception, *, diagnostics_conn: Any | None = None) -> None:
    event_conn = diagnostics_conn or conn
    if is_exploratory_seat_content_failure(task, exc):
        schedule_exploratory_content_retry_or_fail(conn, event_conn, task, exc)
        return

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
            backoff_class=HTTP_FAILURE_BACKOFF_CLASS,
            operational_backoff=True,
            shared_throttle_extended=True,
            retry_scheduled=True,
            terminal=False,
            error_type=type(exc).__name__,
            error_message=diagnostics.short_error(exc),
        )
        safe_record_task_event(
            event_conn,
            task,
            event_type="shared_seat_collection_backoff",
            observed_at=observed_at,
            retry_decision="retry_shared_http_backoff",
            retry_delay_seconds=retry_delay_seconds,
            recovered=False,
            metadata={
                "error_type": type(exc).__name__,
                "error_message": diagnostics.short_error(exc),
                "blocked_until": blocked_until.isoformat(),
            },
        )
    db.mark_task_failed(
        conn,
        task.task_id,
        exc=exc,
        retry_delay_seconds=retry_delay_seconds,
        minimum_max_attempts=minimum_max_attempts,
    )


def schedule_exploratory_content_retry_or_fail(
    conn: Any,
    diagnostics_conn: Any,
    task: db.CollectionTask,
    exc: Exception,
) -> None:
    observed_at = db.utc_now()
    root_cause_code = classify_seat_failure(task, exc)
    retry_delay_seconds = EXPLORATORY_SEAT_CONTENT_RETRY_DELAY_SECONDS
    if task.showtime_id is None:
        safe_record_task_event(
            diagnostics_conn,
            task,
            event_type="seat_failure_classified",
            observed_at=observed_at,
            root_cause_code=root_cause_code,
            retry_decision="none_internal_task_error",
            recovered=False,
            metadata={"error_type": type(exc).__name__, "error_message": diagnostics.short_error(exc)},
        )
        safe_record_task_event(
            diagnostics_conn,
            task,
            event_type="seat_retry_exhausted_hard_failure",
            observed_at=observed_at,
            root_cause_code=root_cause_code,
            retry_decision="hard_failure_internal_task_error",
            recovered=False,
        )
        db.mark_task_failed(
            conn,
            task.task_id,
            exc=exc,
            retry_delay_seconds=0,
            max_attempts=task.attempt_count,
        )
        return

    retry_allowed = exploratory_retry_allowed(task, retry_delay_seconds=retry_delay_seconds, now=observed_at)
    rescue_target = None if retry_allowed else next_late_rescue_target(conn, task, now=observed_at)
    retry_decision = (
        "retry_after_delay"
        if retry_allowed
        else "retry_late_rescue"
        if rescue_target is not None
        else exploratory_no_retry_reason(task, retry_delay_seconds=retry_delay_seconds, now=observed_at)
    )
    safe_record_task_event(
        diagnostics_conn,
        task,
        event_type="seat_failure_classified",
        observed_at=observed_at,
        root_cause_code=root_cause_code,
        retry_decision=retry_decision,
        retry_delay_seconds=retry_delay_seconds if retry_allowed else None,
        recovered=False,
        metadata={
            "error_type": type(exc).__name__,
            "error_message": diagnostics.short_error(exc),
        },
    )
    if retry_allowed:
        safe_record_task_event(
            diagnostics_conn,
            task,
            event_type="seat_retry_scheduled",
            observed_at=observed_at,
            root_cause_code=root_cause_code,
            retry_decision="retry_after_delay",
            retry_delay_seconds=retry_delay_seconds,
            recovered=False,
        )
        db.mark_task_failed(
            conn,
            task.task_id,
            exc=exc,
            retry_delay_seconds=retry_delay_seconds,
            max_attempts=EXPLORATORY_SEAT_CONTENT_MAX_ATTEMPTS,
        )
        return

    if task.attempt_count >= EXPLORATORY_SEAT_CONTENT_MAX_ATTEMPTS:
        safe_record_task_event(
            diagnostics_conn,
            task,
            event_type="seat_retry_failed_again",
            observed_at=observed_at,
            root_cause_code=root_cause_code,
            retry_decision="hard_failure_after_retry",
            recovered=False,
            metadata={"error_type": type(exc).__name__, "error_message": diagnostics.short_error(exc)},
        )
    if rescue_target is not None:
        rescue_offset_minutes, rescue_at, starts_at = rescue_target
        seconds_after_showtime = int((observed_at - starts_at).total_seconds())
        safe_record_task_event(
            diagnostics_conn,
            task,
            event_type="seat_late_rescue_scheduled",
            observed_at=observed_at,
            root_cause_code=root_cause_code,
            retry_decision="retry_late_rescue",
            retry_delay_seconds=max(0, int((rescue_at - observed_at).total_seconds())),
            recovered=False,
            metadata={
                "original_root_cause_code": root_cause_code,
                "rescue_target_offset_minutes_after_showtime": rescue_offset_minutes,
                "prior_attempt_count": task.attempt_count,
                "seconds_after_showtime_at_failure": seconds_after_showtime,
            },
        )
        db.reschedule_task_retry(
            conn,
            task.task_id,
            exc=exc,
            scheduled_for=rescue_at,
            target_offset_minutes=-rescue_offset_minutes,
            minimum_max_attempts=max(task.attempt_count + 1, LATE_RESCUE_MAX_ATTEMPTS),
        )
        return

    safe_record_task_event(
        diagnostics_conn,
        task,
        event_type="seat_retry_exhausted_hard_failure",
        observed_at=observed_at,
        root_cause_code=root_cause_code,
        retry_decision=retry_decision,
        recovered=False,
    )
    db.mark_task_failed(
        conn,
        task.task_id,
        exc=exc,
        retry_delay_seconds=0,
        max_attempts=max(task.attempt_count, EXPLORATORY_SEAT_CONTENT_MAX_ATTEMPTS),
    )


def retry_delay_seconds_for_failure(task: db.CollectionTask, exc: Exception) -> int:
    if is_exploratory_seat_content_failure(task, exc):
        return EXPLORATORY_SEAT_CONTENT_RETRY_DELAY_SECONDS
    if not is_seat_backoff_failure(task, exc):
        return 60
    delay_index = max(0, min(task.attempt_count - 1, len(SEAT_BACKOFF_RETRY_DELAYS_SECONDS) - 1))
    return SEAT_BACKOFF_RETRY_DELAYS_SECONDS[delay_index]


def minimum_max_attempts_for_failure(task: db.CollectionTask, exc: Exception) -> int | None:
    if is_exploratory_seat_content_failure(task, exc):
        return EXPLORATORY_SEAT_CONTENT_MAX_ATTEMPTS
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
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in HTTP_BACKOFF_STATUSES
    return False


def is_terminal_seat_content_failure(task: db.CollectionTask, exc: Exception) -> bool:
    return False


def is_exploratory_seat_content_failure(task: db.CollectionTask, exc: Exception) -> bool:
    if task.task_type != "collect_seat_snapshot":
        return False
    if isinstance(exc, SeatMapUnavailable):
        return True
    message = str(exc)
    return (
        "AMC RSC payload" in message
        or "apollo-data" in message
        or "empty response body" in message
        or "rendered HTML" in message
        or any(marker in message for marker in SEAT_BACKOFF_ERROR_MARKERS)
    )


def classify_seat_failure(task: db.CollectionTask, exc: Exception | str) -> str:
    message = str(exc)
    if task.showtime_id is None:
        return "internal_task_error"
    if isinstance(exc, SeatMapUnavailable) or "no reserved seat map" in message:
        return "no_seat_map_non_reserved"
    if "Could not find showtime object in AMC RSC payload" in message:
        return "missing_showtime_object"
    if "Could not find showtime object with seatingLayout in AMC RSC payload" in message:
        return "missing_seating_layout"
    if "Could not find showtime.seatingLayout.seats in AMC RSC payload" in message:
        return "missing_seats"
    if "empty" in message.lower() or "decode" in message.lower() or "malformed" in message.lower():
        return "empty_or_malformed_payload"
    if "zero seats" in message.lower() or "rendered" in message.lower():
        return "rendered_html_zero_seats"
    return "unknown_parser_shape"


def classify_previous_task_failure(task: db.CollectionTask) -> str | None:
    if not task.last_error_message:
        return None
    return classify_seat_failure(task, task.last_error_message)


def exploratory_retry_allowed(
    task: db.CollectionTask,
    *,
    retry_delay_seconds: int,
    now: dt.datetime | None = None,
) -> bool:
    if task.task_type != "collect_seat_snapshot" or task.showtime_id is None:
        return False
    if task.attempt_count >= EXPLORATORY_SEAT_CONTENT_MAX_ATTEMPTS:
        return False
    return True


def exploratory_no_retry_reason(
    task: db.CollectionTask,
    *,
    retry_delay_seconds: int,
    now: dt.datetime | None = None,
) -> str:
    if task.showtime_id is None:
        return "none_internal_task_error"
    if task.attempt_count >= EXPLORATORY_SEAT_CONTENT_MAX_ATTEMPTS:
        return "none_exploratory_attempts_exhausted"
    return "none_not_retryable"


def next_late_rescue_target(
    conn: Any,
    task: db.CollectionTask,
    *,
    now: dt.datetime | None = None,
) -> tuple[int, dt.datetime, dt.datetime] | None:
    if task.showtime_id is None:
        return None
    showtime = db.select_showtime_by_id(conn, task.showtime_id)
    if showtime is None:
        return None
    observed_at = db.ensure_utc(now or db.utc_now())
    starts_at = db.ensure_utc(showtime.utc_start_at)
    for rescue_offset_minutes in LATE_RESCUE_OFFSETS_MINUTES:
        rescue_at = starts_at + dt.timedelta(minutes=rescue_offset_minutes)
        if observed_at < rescue_at:
            return rescue_offset_minutes, rescue_at, starts_at
    return None


def safe_record_task_event(conn: Any, task: db.CollectionTask, **kwargs: Any) -> None:
    try:
        db.record_task_diagnostic_event(conn, task, **kwargs)
    except Exception:
        if hasattr(conn, "rollback"):
            conn.rollback()
        return
