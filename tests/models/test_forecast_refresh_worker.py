from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from models.boxoffice.refresh_queue import ForecastRefresh
from models.boxoffice.refresh_worker import ForecastRefreshResult, process_refresh


class FakeConnection:
    def __init__(self) -> None:
        self.commits = 0
        self.rollbacks = 0

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


def test_process_refresh_marks_success_and_commits() -> None:
    conn = FakeConnection()
    refresh = ForecastRefresh(
        refresh_id=7,
        release_run_id=123,
        movie_id=456,
        model_version="latest",
        attempt_count=1,
    )

    def runner(*_args: object) -> ForecastRefreshResult:
        return ForecastRefreshResult(
            run_id="forecast_refresh_test",
            resolved_model_version="boxoffice_test",
            row_count=4,
            component_row_count=3,
        )

    with patch("models.boxoffice.refresh_worker.mark_refresh_succeeded") as mark_succeeded:
        result = process_refresh(conn, refresh, artifact_root=Path("models/boxoffice"), runner=runner)

    assert result.row_count == 4
    mark_succeeded.assert_called_once_with(
        conn,
        refresh_id=7,
        run_id="forecast_refresh_test",
        resolved_model_version="boxoffice_test",
        row_count=4,
    )
    assert conn.commits == 1
    assert conn.rollbacks == 0


def test_process_refresh_marks_failure_after_rollback() -> None:
    conn = FakeConnection()
    refresh = ForecastRefresh(
        refresh_id=7,
        release_run_id=123,
        movie_id=456,
        model_version="latest",
        attempt_count=1,
    )

    def runner(*_args: object) -> ForecastRefreshResult:
        raise RuntimeError("missing inputs")

    with (
        patch("models.boxoffice.refresh_worker.mark_refresh_succeeded") as mark_succeeded,
        patch("models.boxoffice.refresh_worker.mark_refresh_failed") as mark_failed,
    ):
        with pytest.raises(RuntimeError):
            process_refresh(
                conn,
                refresh,
                artifact_root=Path("models/boxoffice"),
                max_attempts=5,
                retry_delay_seconds=30,
                runner=runner,
            )

    mark_succeeded.assert_not_called()
    mark_failed.assert_called_once()
    assert mark_failed.call_args.kwargs["refresh_id"] == 7
    assert mark_failed.call_args.kwargs["max_attempts"] == 5
    assert mark_failed.call_args.kwargs["retry_delay_seconds"] == 30
    assert conn.rollbacks == 1
    assert conn.commits == 1
