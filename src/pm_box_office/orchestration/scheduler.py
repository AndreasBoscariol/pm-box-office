"""Lightweight daily scheduler for local ingest orchestration."""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import logging
from typing import Any

from pm_box_office.db.connection import connect_database
from pm_box_office.orchestration import repository, runner
from pm_box_office.orchestration.registry import (
    AUTORUN_TIMEZONE,
    CONTINUOUS_POLL_SOURCE_KEYS,
    DAILY_BACKGROUND_SOURCE_KEYS,
    SOURCE_POLLING_WINDOWS,
    source_poll_due,
)


LOGGER = logging.getLogger(__name__)
POLL_SECONDS = 60


SOURCE_POLL_TRIGGER = "auto_source_poll"


def due_poll_sources(*, now: dt.datetime) -> tuple[str, ...]:
    return tuple(source_key for source_key in SOURCE_POLLING_WINDOWS if source_poll_due(source_key, now))


def tick_once(*, database_url: str | None = None, now: dt.datetime | None = None) -> dict[str, list[str]] | None:
    current_time = now or dt.datetime.now(dt.UTC)
    poll_sources = due_poll_sources(now=current_time)
    started: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []

    if poll_sources:
        conn = connect_database(database_url)
        try:
            repository.initialize_orchestration_database(conn)
            # Keep a completed source poll from being started again if the app
            # restarts while the scheduler is still inside the same minute.
            slot_start = current_time.replace(second=0, microsecond=0)
            local_date = current_time.astimezone(AUTORUN_TIMEZONE).date()
            runnable = tuple(
                source_key
                for source_key in poll_sources
                if (
                    source_key in CONTINUOUS_POLL_SOURCE_KEYS
                    or not repository.source_window_publication_found(
                        conn,
                        source_key=source_key,
                        local_date=local_date,
                    )
                )
                and not repository.source_has_recent_trigger(
                    conn, source_key=source_key, trigger=SOURCE_POLL_TRIGGER, since=slot_start
                )
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        if runnable:
            result = runner.start_run_all(
                trigger=SOURCE_POLL_TRIGGER,
                database_url=database_url,
                source_keys=runnable,
            )
            started.extend(result["started"])
            skipped.extend(result["skipped"])
            errors.extend(result["errors"])

    conn = connect_database(database_url)
    try:
        repository.initialize_orchestration_database(conn)
        if not repository.autorun_due(conn):
            conn.commit()
            return {"started": started, "skipped": skipped, "errors": errors} if (started or skipped or errors) else None
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    result = runner.start_run_all(
        trigger="auto_daily",
        database_url=database_url,
        source_keys=DAILY_BACKGROUND_SOURCE_KEYS,
    )
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
    started.extend(result["started"])
    skipped.extend(result["skipped"])
    errors.extend(result["errors"])
    return {"started": started, "skipped": skipped, "errors": errors}


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
