"""Progress queries for collection runs."""

from __future__ import annotations

import uuid
from typing import Any

from pm_box_office.sources.amc import db


def run_progress(conn: Any, run_id: str | uuid.UUID) -> dict[str, Any]:
    parsed_run_id = db.as_uuid(run_id)
    progress = db.run_progress(conn, parsed_run_id)
    progress["events"] = db.run_recent_task_events(conn, parsed_run_id)
    return progress
