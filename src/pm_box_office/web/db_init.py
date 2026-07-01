"""One-time database initialization for the web process."""

from __future__ import annotations

import threading
from typing import Any

from pm_box_office.sources.amc import db
from pm_box_office.orchestration import repository


_initialized = False
_lock = threading.Lock()


def ensure_initialized(conn: Any) -> None:
    global _initialized
    if _initialized:
        return
    with _lock:
        if _initialized:
            return
        db.initialize_amc_database(conn)
        repository.initialize_orchestration_database(conn)
        repository.seed_sources(conn)
        _initialized = True
