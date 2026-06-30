"""PostgreSQL-backed task queue helpers."""

from __future__ import annotations

from typing import Any

from pm_box_office.sources.amc import db


def claim_due_tasks(conn: Any, *, worker_id: str, limit: int = 20) -> list[db.CollectionTask]:
    return db.claim_due_tasks(conn, worker_id=worker_id, limit=limit)


def mark_succeeded(conn: Any, task_id: int) -> None:
    db.mark_task_succeeded(conn, task_id)


def schedule_retry_or_fail(conn: Any, task_id: int, exc: Exception) -> None:
    db.mark_task_failed(conn, task_id, exc=exc)
