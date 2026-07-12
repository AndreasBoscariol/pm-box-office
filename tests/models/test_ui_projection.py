from __future__ import annotations

from datetime import datetime, timezone

from models.boxoffice.ui_projection import publish_forecast_updates


class Cursor:
    def __init__(self, row: tuple[object, ...] | None = None) -> None:
        self.row = row

    def fetchone(self) -> tuple[object, ...] | None:
        return self.row


class Connection:
    def __init__(self) -> None:
        self.scripts: list[str] = []
        self.calls: list[tuple[str, tuple[object, ...] | None]] = []

    def executescript(self, sql: str) -> None:
        self.scripts.append(sql)

    def execute(self, sql: str, params: tuple[object, ...] | None = None) -> Cursor:
        self.calls.append((sql, params))
        return Cursor((1,)) if "RETURNING outbox_id" in sql else Cursor()


def test_publish_forecast_updates_writes_current_projection_and_outbox_event() -> None:
    conn = Connection()
    rows = [
        {
            "forecast_id": "forecast-1",
            "release_run_id": 10,
            "model_version": "production-model",
            "target": "opening_weekend",
            "run_id": "run-1",
            "origin_key": "FRI_20:00",
            "forecast_origin_utc": datetime(2026, 7, 10, 20, tzinfo=timezone.utc),
            "as_of_utc": datetime(2026, 7, 10, 20, tzinfo=timezone.utc),
            "point_usd": 100,
        }
    ]

    assert publish_forecast_updates(conn, rows) == 1
    assert len(conn.scripts) == 1
    assert any("current_release_forecasts" in sql for sql, _ in conn.calls)
    assert any("forecast_ui_outbox" in sql for sql, _ in conn.calls)
