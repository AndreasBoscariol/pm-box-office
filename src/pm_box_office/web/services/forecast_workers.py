"""Manage the local live forecast refresh worker from the web UI."""

from __future__ import annotations

import errno
import os
import signal
import subprocess
import sys
from pathlib import Path

from pm_box_office.config import REPO_ROOT


WORKER_PID_PATH = REPO_ROOT / "data" / "run" / "forecast_refresh_worker.pid"
WORKER_LOG_PATH = REPO_ROOT / "data" / "logs" / "forecast_refresh_worker.log"
DEFAULT_BATCH_LIMIT = 1
DEFAULT_IDLE_SECONDS = 2.0


def ensure_worker_started() -> int | None:
    pid = worker_pid()
    if pid is not None and pid_is_running(pid):
        return pid

    WORKER_PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    WORKER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    batch_limit = configured_int("FORECAST_REFRESH_BATCH_LIMIT", DEFAULT_BATCH_LIMIT, minimum=1)
    idle_seconds = configured_float("FORECAST_REFRESH_IDLE_SECONDS", DEFAULT_IDLE_SECONDS, minimum=0.1)
    with WORKER_LOG_PATH.open("ab") as log_file:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "models.boxoffice.refresh_worker",
                "--worker-id",
                "web-forecast-refresh",
                "--limit",
                str(batch_limit),
                "--idle-seconds",
                str(idle_seconds),
                "--verbose",
            ],
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    WORKER_PID_PATH.write_text(str(process.pid), encoding="utf-8")
    return process.pid


def restart_worker() -> int | None:
    stop_worker()
    return ensure_worker_started()


def stop_worker() -> bool:
    pid = worker_pid()
    stopped = False
    if pid is not None and pid_is_running(pid):
        try:
            os.kill(pid, signal.SIGTERM)
            stopped = True
        except OSError:
            pass
    try:
        WORKER_PID_PATH.unlink()
    except FileNotFoundError:
        pass
    return stopped


def worker_pid() -> int | None:
    if not WORKER_PID_PATH.exists():
        return None
    try:
        return int(WORKER_PID_PATH.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


def status() -> dict[str, object]:
    pid = worker_pid()
    running = pid is not None and pid_is_running(pid)
    return {
        "pid": pid,
        "running": running,
        "batch_limit": configured_int("FORECAST_REFRESH_BATCH_LIMIT", DEFAULT_BATCH_LIMIT, minimum=1),
        "idle_seconds": configured_float("FORECAST_REFRESH_IDLE_SECONDS", DEFAULT_IDLE_SECONDS, minimum=0.1),
    }


def configured_int(name: str, default: int, *, minimum: int) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except ValueError:
        return max(minimum, default)


def configured_float(name: str, default: float, *, minimum: float) -> float:
    try:
        return max(minimum, float(os.environ.get(name, str(default))))
    except ValueError:
        return max(minimum, default)
