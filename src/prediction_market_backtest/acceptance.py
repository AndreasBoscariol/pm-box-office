"""Strict, timestamp-derived adjudication for the formal CLOB run."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

TARGET_DURATION_HOURS = 24.1


def manifest_duration_hours(manifest: Mapping[str, Any]) -> float:
    """Derive duration only from persisted wall-clock timestamps."""
    start = datetime.fromisoformat(str(manifest["start_utc"]))
    end = datetime.fromisoformat(str(manifest["end_utc"]))
    duration = (end - start).total_seconds() / 3600
    if duration < 0:
        raise ValueError("end precedes start")
    return duration


def acceptance_failure_codes(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    failures: list[str] = []
    try:
        duration = manifest_duration_hours(manifest)
    except (KeyError, TypeError, ValueError):
        duration = 0
        failures.append("failed_manifest_integrity")
    if duration < TARGET_DURATION_HOURS:
        failures.append("failed_insufficient_duration")
    checks = (
        ("reconnect_successes", "failed_reconnect"),
        ("sequence_gaps", "failed_sequence_gap_detection"),
        ("state_invalidations", "failed_state_invalidation"),
        ("rest_recovery_successes", "failed_rest_recovery"),
    )
    for field, failure in checks:
        if int(manifest.get(field, 0)) < 1:
            failures.append(failure)
    if int(manifest.get("unrecovered_failures", manifest.get("unrecovered_books", 0))) > 0:
        failures.append("failed_unrecovered_book")
    if int(manifest.get("reconciliation_failures", 0)) > 0:
        failures.append("failed_reconciliation")
    if int(manifest.get("database_write_errors", 0)) > 0:
        failures.append("failed_database_write")
    if int(manifest.get("corrupted_books_used_downstream", 0)) > 0:
        failures.append("failed_state_invalidation")
    if not bool(manifest.get("immutable_snapshot_persistence", int(manifest.get("snapshots", 0)) > 0)):
        failures.append("failed_database_write")
    if not bool(manifest.get("health_metrics_queryable", True)):
        failures.append("failed_manifest_integrity")
    return tuple(dict.fromkeys(failures))


def validate_capture_acceptance(manifest: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
    failures = acceptance_failure_codes(manifest)
    return not failures, failures
