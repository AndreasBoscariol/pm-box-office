"""Launch and cancel local ingest runs."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import uuid

from pm_box_office.config import REPO_ROOT
from pm_box_office.db.connection import connect_database, database_url_from_env
from pm_box_office.orchestration import repository


def start_source_run(
    source_key: str,
    *,
    trigger: str = "manual",
    database_url: str | None = None,
    extra_args: list[str] | None = None,
) -> uuid.UUID:
    conn = connect_database(database_url)
    try:
        repository.initialize_orchestration_database(conn)
        repository.seed_sources(conn)
        run_id = repository.create_run(conn, source_key=source_key, trigger=trigger, extra_args=extra_args)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    env = os.environ.copy()
    resolved_url = database_url or database_url_from_env()
    if resolved_url:
        env["DATABASE_URL"] = resolved_url
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "pm_box_office.orchestration.supervisor", "--run-id", str(run_id)],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
    except Exception as exc:
        conn = connect_database(database_url)
        try:
            repository.complete_run(
                conn,
                run_id=run_id,
                status="failed",
                exit_code=None,
                error_summary=f"Could not start supervisor: {exc}",
            )
            conn.commit()
        finally:
            conn.close()
        raise

    conn = connect_database(database_url)
    try:
        repository.mark_run_spawned(conn, run_id=run_id, pid=process.pid)
        conn.commit()
    finally:
        conn.close()
    return run_id


def cancel_run(run_id: str | uuid.UUID, *, database_url: str | None = None) -> None:
    conn = connect_database(database_url)
    try:
        repository.initialize_orchestration_database(conn)
        pid = repository.request_cancel(conn, run_id)
        if pid is not None:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError:
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass
        repository.append_log(conn, run_id=run_id, stream="system", line="Cancellation requested.")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
