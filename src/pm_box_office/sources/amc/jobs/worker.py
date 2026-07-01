#!/usr/bin/env python3
"""Run the AMC collection worker."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import socket
import time
from pathlib import Path

from pm_box_office.db.connection import connect_database
from pm_box_office.sources.amc import db
from pm_box_office.sources.amc.client import DEFAULT_CACHE_DIR, DEFAULT_USER_AGENT, HtmlFetcher
from pm_box_office.sources.amc.diagnostics import diagnostics_context, log_backoff_event, short_error
from pm_box_office.sources.amc.jobs import handlers, queue


LOGGER = logging.getLogger("amc.worker")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", help="PostgreSQL URL. Defaults to DATABASE_URL/POSTGRES_DSN/.env.")
    parser.add_argument("--worker-id", default=f"{socket.gethostname()}-{time.time_ns()}")
    parser.add_argument("--once", action="store_true", help="Claim and process one batch, then exit.")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--idle-seconds", type=float, default=1.0)
    parser.add_argument("--stale-running-minutes", type=int, default=5)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--heartbeat-path", type=Path, default=Path("data/run/amc_worker.heartbeat"))
    parser.add_argument("--delay-seconds", type=float, default=3.0)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--verbose", action="store_true")
    return parser


def run_worker(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    conn = connect_database(args.database_url)
    fetcher = HtmlFetcher(
        args.cache_dir,
        refresh=False,
        offline=False,
        delay_seconds=args.delay_seconds,
        user_agent=args.user_agent,
    )
    try:
        db.initialize_amc_database(conn)
        conn.commit()
        LOGGER.info("worker started worker_id=%s limit=%s", args.worker_id, args.limit)
        while True:
            write_heartbeat(args.heartbeat_path)
            reset_count = db.reset_stale_running_tasks(
                conn,
                stale_after=dt.timedelta(minutes=args.stale_running_minutes),
            )
            if reset_count:
                LOGGER.warning("reset %s stale running tasks", reset_count)
                conn.commit()
            tasks = queue.claim_due_tasks(conn, worker_id=args.worker_id, limit=args.limit)
            conn.commit()
            if not tasks:
                if args.once:
                    LOGGER.info("no due tasks; exiting because --once was set")
                    return 0
                time.sleep(args.idle_seconds)
                continue
            LOGGER.info("claimed %s due tasks", len(tasks))
            for task in tasks:
                LOGGER.info(
                    "task start task_id=%s type=%s theatre=%s showtime=%s attempt=%s",
                    task.task_id,
                    task.task_type,
                    task.amc_theatre_id,
                    task.showtime_id,
                    task.attempt_count,
                )
                with diagnostics_context(**task_diagnostics_fields(args.worker_id, task)):
                    try:
                        handlers.execute(conn, fetcher, task)
                        queue.mark_succeeded(conn, task.task_id)
                        conn.commit()
                        LOGGER.info("task succeeded task_id=%s", task.task_id)
                    except Exception as exc:
                        LOGGER.exception("task failed task_id=%s", task.task_id)
                        log_backoff_event(
                            "seat_task_failed",
                            error_type=type(exc).__name__,
                            error_message=short_error(exc),
                        )
                        conn.rollback()
                        try:
                            queue.schedule_retry_or_fail(conn, task.task_id, exc)
                            conn.commit()
                        except Exception:
                            LOGGER.exception("could not mark task failed task_id=%s", task.task_id)
                            conn.rollback()
                        continue
            if args.once:
                LOGGER.info("processed one batch; exiting because --once was set")
                return 0
    finally:
        conn.close()


def main() -> int:
    return run_worker(build_parser().parse_args())


def write_heartbeat(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(time.time()), encoding="utf-8")


def task_diagnostics_fields(worker_id: str, task: db.CollectionTask) -> dict[str, object]:
    now = db.utc_now()
    scheduled_for = db.ensure_utc(task.scheduled_for)
    return {
        "worker_id": worker_id,
        "task_id": task.task_id,
        "run_id": str(task.run_id),
        "task_type": task.task_type,
        "showtime_id": task.showtime_id,
        "amc_movie_id": task.amc_movie_id,
        "amc_theatre_id": task.amc_theatre_id,
        "scheduled_for": scheduled_for,
        "target_offset_minutes": task.priority or 5,
        "attempt_count": task.attempt_count,
        "seconds_late_at_start": max(0, int((now - scheduled_for).total_seconds())),
    }


if __name__ == "__main__":
    raise SystemExit(main())
