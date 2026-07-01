"""Supervisor process for a single local ingest run."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import traceback

from pm_box_office.config import REPO_ROOT
from pm_box_office.db.connection import connect_database, database_url_from_env
from pm_box_office.orchestration import repository


def supervise_run(run_id: str, *, database_url: str | None = None) -> int:
    conn = connect_database(database_url)
    stop_heartbeat = threading.Event()
    heartbeat_thread: threading.Thread | None = None
    process: subprocess.Popen[str] | None = None
    try:
        repository.initialize_orchestration_database(conn)
        command, args = repository.load_run_command(conn, run_id)
        child_env = os.environ.copy()
        resolved_url = database_url or database_url_from_env()
        if resolved_url:
            child_env["DATABASE_URL"] = resolved_url
        child_env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            [sys.executable, "-m", command, *args],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=child_env,
            start_new_session=True,
        )
        repository.mark_run_running(conn, run_id=run_id, pid=process.pid)
        repository.append_log(conn, run_id=run_id, stream="system", line=f"Started {command} pid={process.pid}")
        conn.commit()
        heartbeat_thread = start_heartbeat(run_id, stop_heartbeat, database_url=database_url)

        assert process.stdout is not None
        for line in process.stdout:
            repository.append_log(conn, run_id=run_id, stream="stdout", line=line)
            conn.commit()
        exit_code = process.wait()
        status = "succeeded" if exit_code == 0 else "failed"
        repository.complete_run(
            conn,
            run_id=run_id,
            status=status,
            exit_code=exit_code,
            error_summary=None if exit_code == 0 else f"Process exited with code {exit_code}",
        )
        if exit_code == 0:
            repository.refresh_all_source_freshness(conn)
        conn.commit()
        return int(exit_code)
    except Exception as exc:  # noqa: BLE001 - persist supervisor failures.
        try:
            repository.append_log(conn, run_id=run_id, stream="system", line=traceback.format_exc())
            repository.complete_run(conn, run_id=run_id, status="failed", exit_code=None, error_summary=str(exc))
            conn.commit()
        except Exception:
            conn.rollback()
        return 1
    finally:
        stop_heartbeat.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=2)
        conn.close()


def start_heartbeat(run_id: str, stop_event: threading.Event, *, database_url: str | None) -> threading.Thread:
    thread = threading.Thread(
        target=heartbeat_loop,
        args=(run_id, stop_event, database_url),
        name=f"ingest-heartbeat-{run_id}",
        daemon=True,
    )
    thread.start()
    return thread


def heartbeat_loop(run_id: str, stop_event: threading.Event, database_url: str | None) -> None:
    conn = connect_database(database_url)
    try:
        while not stop_event.wait(5):
            repository.heartbeat_run(conn, run_id)
            conn.commit()
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--database-url")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return supervise_run(args.run_id, database_url=args.database_url)


if __name__ == "__main__":
    raise SystemExit(main())
