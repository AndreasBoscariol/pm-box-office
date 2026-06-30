#!/usr/bin/env python3
"""Run the AMC collection worker."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import socket
import time
from pathlib import Path

from scripts.db import connect_database
from scripts.ingest.amc import db
from scripts.ingest.amc.client import DEFAULT_CACHE_DIR, DEFAULT_USER_AGENT, HtmlFetcher
from scripts.ingest.amc.jobs import handlers, queue


LOGGER = logging.getLogger("amc.worker")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", default=f"{socket.gethostname()}-{time.time_ns()}")
    parser.add_argument("--once", action="store_true", help="Claim and process one batch, then exit.")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--idle-seconds", type=float, default=1.0)
    parser.add_argument("--stale-running-minutes", type=int, default=5)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--heartbeat-path", type=Path, default=Path("data/run/amc_worker.heartbeat"))
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--verbose", action="store_true")
    return parser


def run_worker(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    conn = connect_database()
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
                try:
                    handlers.execute(conn, fetcher, task)
                    queue.mark_succeeded(conn, task.task_id)
                    conn.commit()
                    LOGGER.info("task succeeded task_id=%s", task.task_id)
                except Exception as exc:
                    LOGGER.exception("task failed task_id=%s", task.task_id)
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


if __name__ == "__main__":
    raise SystemExit(main())
