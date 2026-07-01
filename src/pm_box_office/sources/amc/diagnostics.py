"""Structured diagnostics for AMC collection throttling and malformed payloads."""

from __future__ import annotations

import contextlib
import contextvars
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Iterator

from pm_box_office.config import REPO_ROOT


BACKOFF_LOG_PATH = REPO_ROOT / "data" / "logs" / "amc_backoff_events.jsonl"
_CONTEXT: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("amc_diagnostics_context", default={})


@contextlib.contextmanager
def diagnostics_context(**fields: Any) -> Iterator[None]:
    context = current_context()
    context.update({key: _coerce_value(value) for key, value in fields.items() if value is not None})
    token = _CONTEXT.set(context)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def current_context() -> dict[str, Any]:
    return dict(_CONTEXT.get())


def log_backoff_event(event_type: str, **fields: Any) -> None:
    path = Path(fields.pop("log_path", BACKOFF_LOG_PATH))
    payload: dict[str, Any] = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "event_type": event_type,
        "pid": os.getpid(),
    }
    payload.update(current_context())
    payload.update({key: _coerce_value(value) for key, value in fields.items() if value is not None})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def short_error(exc: BaseException | str, *, limit: int = 500) -> str:
    message = str(exc)
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


def url_kind(url: str) -> str:
    if "_rsc=" in url:
        return "rsc"
    if "showtimes" in url:
        return "seat_html" if "/showtimes/" in url else "inventory"
    if "sitemap" in url:
        return "sitemap"
    return "other"


def _coerce_value(value: Any) -> Any:
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _coerce_value(item) for key, item in value.items()}
    return str(value)
