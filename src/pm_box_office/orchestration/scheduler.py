"""Lightweight daily scheduler for local ingest orchestration."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from pm_box_office.db.connection import connect_database
from pm_box_office.orchestration import repository, runner


LOGGER = logging.getLogger(__name__)
POLL_SECONDS = 60


def tick_once(*, database_url: str | None = None) -> dict[str, list[str]] | None:
    conn = connect_database(database_url)
    try:
        repository.initialize_orchestration_database(conn)
        if not repository.autorun_due(conn):
            conn.commit()
            return None
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    result = runner.start_run_all(trigger="auto_daily", database_url=database_url)
    if not result["started"]:
        conn = connect_database(database_url)
        try:
            repository.record_autorun_trigger(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    return result


async def run_forever(*, database_url: str | None = None, poll_seconds: int = POLL_SECONDS) -> None:
    while True:
        try:
            await asyncio.to_thread(tick_once, database_url=database_url)
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("Daily ingest autorun tick failed")
        await asyncio.sleep(poll_seconds)


async def shutdown_task(task: Any) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
