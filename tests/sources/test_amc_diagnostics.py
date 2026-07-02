from __future__ import annotations

import datetime as dt
import json
import tempfile
import unittest
import urllib.error
import uuid
from pathlib import Path
from unittest.mock import patch

from pm_box_office.sources.amc import diagnostics
from pm_box_office.sources.amc import parsers as amc
from pm_box_office.sources.amc.client import HtmlFetcher
from pm_box_office.sources.amc.db import CollectionTask
from pm_box_office.sources.amc.jobs import queue
from pm_box_office.sources.amc.jobs import worker
from tests.sources.test_collect_amc_showtimes import RENDERED_SEATS_HTML, RSC_NO_SEAT_MAP_PAYLOAD


class FakeFetcher:
    def __init__(self, pages: dict[str, str]) -> None:
        self.pages = pages

    def get(self, url: str) -> tuple[str, Path | None, bool]:
        return self.pages[url], None, True


class FakeResponse:
    status = 200

    def __init__(self, body: str) -> None:
        self.body = body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body.encode("utf-8")


class AmcDiagnosticsTests(unittest.TestCase):
    def test_log_backoff_event_appends_json_with_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "events.jsonl"
            with diagnostics.diagnostics_context(worker_id="local-0", task_id=123):
                diagnostics.log_backoff_event(
                    "seat_task_failed",
                    log_path=path,
                    error_type="ValueError",
                    error_message="bad payload",
                )

            row = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual("seat_task_failed", row["event_type"])
        self.assertEqual("local-0", row["worker_id"])
        self.assertEqual(123, row["task_id"])
        self.assertEqual("ValueError", row["error_type"])
        self.assertIn("timestamp_utc", row)
        self.assertIn("pid", row)

    def test_html_fetcher_logs_http_retry_then_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "events.jsonl"
            error = urllib.error.HTTPError(
                "https://www.amctheatres.com/showtimes/100?_rsc=1",
                429,
                "Too Many Requests",
                {"Retry-After": "0"},
                None,
            )
            fetcher = HtmlFetcher(Path(tmp_dir) / "cache", delay_seconds=0, retries=2)
            with (
                patch.object(diagnostics, "BACKOFF_LOG_PATH", log_path),
                patch("pm_box_office.sources.amc.client.time.sleep"),
                patch("pm_box_office.sources.amc.client.urllib.request.urlopen", side_effect=[error, FakeResponse("ok")]),
            ):
                result = fetcher.get_result("https://www.amctheatres.com/showtimes/100?_rsc=1", refresh=True)

            rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual("ok", result.body)
        self.assertEqual(1, len(rows))
        self.assertEqual("http_retry", rows[0]["event_type"])
        self.assertEqual(429, rows[0]["status_code"])
        self.assertEqual("rsc", rows[0]["url_kind"])

    def test_html_fetcher_logs_final_http_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "events.jsonl"
            error = urllib.error.HTTPError(
                "https://www.amctheatres.com/showtimes/100?_rsc=1",
                403,
                "Forbidden",
                {},
                None,
            )
            fetcher = HtmlFetcher(Path(tmp_dir) / "cache", delay_seconds=0, retries=1)
            with (
                patch.object(diagnostics, "BACKOFF_LOG_PATH", log_path),
                patch("pm_box_office.sources.amc.client.urllib.request.urlopen", side_effect=error),
                self.assertRaises(urllib.error.HTTPError),
            ):
                fetcher.get_result("https://www.amctheatres.com/showtimes/100?_rsc=1", refresh=True)

            rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(1, len(rows))
        self.assertEqual("http_failed", rows[0]["event_type"])
        self.assertEqual(403, rows[0]["status_code"])
        self.assertEqual("HTTPError", rows[0]["error_type"])

    def test_fetch_seat_fill_logs_rsc_failure_and_fallback_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "events.jsonl"
            fetcher = FakeFetcher(
                {
                    amc.current_showtime_seats_rsc_url("100"): "no seats here",
                    amc.current_showtime_seats_url("100"): RENDERED_SEATS_HTML,
                }
            )
            with patch.object(diagnostics, "BACKOFF_LOG_PATH", log_path):
                fill = amc.fetch_seat_fill(
                    fetcher,  # type: ignore[arg-type]
                    theatre_slug="amc-sample-10",
                    date=dt.date(2026, 7, 1),
                    showtime_id="100",
                    prefer_rsc=True,
                )

            rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(2, fill.total_seats)
        self.assertEqual(["rsc_missing_seats", "rsc_fallback_succeeded"], [row["event_type"] for row in rows])
        self.assertEqual("100", rows[0]["showtime_id"])
        self.assertEqual("rendered_html", rows[1]["fallback_method"])

    def test_non_reserved_rsc_payload_logs_no_seat_map_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "events.jsonl"
            fetcher = FakeFetcher(
                {
                    amc.current_showtime_seats_rsc_url("100"): RSC_NO_SEAT_MAP_PAYLOAD,
                }
            )
            with patch.object(diagnostics, "BACKOFF_LOG_PATH", log_path):
                with self.assertRaises(amc.SeatMapUnavailable):
                    amc.fetch_seat_fill(
                        fetcher,  # type: ignore[arg-type]
                        theatre_slug="amc-sample-10",
                        date=dt.date(2026, 7, 1),
                        showtime_id="100",
                        prefer_rsc=True,
                    )

            rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(["seat_map_unavailable_non_reserved"], [row["event_type"] for row in rows])
        self.assertEqual("100", rows[0]["showtime_id"])
        self.assertEqual(["Closed Caption", "Descriptive Video"], rows[0]["attribute_names"])
        self.assertEqual("SeatMapUnavailable", rows[0]["error_type"])

    def test_worker_task_diagnostics_fields_include_task_metadata(self) -> None:
        run_id = uuid.uuid4()
        scheduled_for = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=90)
        task = CollectionTask(
            task_id=42,
            run_id=run_id,
            task_type="collect_seat_snapshot",
            amc_theatre_id=None,
            showtime_id="100",
            amc_movie_id="movie-1",
            scheduled_for=scheduled_for,
            status="running",
            priority=5,
            attempt_count=2,
            max_attempts=3,
        )

        fields = worker.task_diagnostics_fields("local-0", task)

        self.assertEqual("local-0", fields["worker_id"])
        self.assertEqual(42, fields["task_id"])
        self.assertEqual(str(run_id), fields["run_id"])
        self.assertEqual("100", fields["showtime_id"])
        self.assertEqual("movie-1", fields["amc_movie_id"])
        self.assertEqual(5, fields["target_offset_minutes"])
        self.assertGreaterEqual(fields["seconds_late_at_start"], 90)

    def test_seat_backoff_policy_extends_retry_window_and_attempts(self) -> None:
        run_id = uuid.uuid4()
        task = CollectionTask(
            task_id=42,
            run_id=run_id,
            task_type="collect_seat_snapshot",
            amc_theatre_id=None,
            showtime_id="100",
            amc_movie_id="movie-1",
            scheduled_for=dt.datetime.now(dt.timezone.utc),
            status="running",
            priority=5,
            attempt_count=3,
            max_attempts=3,
        )
        exc = ValueError("Could not find showtime object in AMC RSC payload")

        self.assertTrue(queue.is_seat_backoff_failure(task, exc))
        self.assertEqual(120, queue.retry_delay_seconds_for_failure(task, exc))
        self.assertEqual(6, queue.minimum_max_attempts_for_failure(task, exc))

    def test_non_seat_backoff_policy_keeps_default_retry(self) -> None:
        run_id = uuid.uuid4()
        task = CollectionTask(
            task_id=42,
            run_id=run_id,
            task_type="collect_theatre_showtimes",
            amc_theatre_id=123,
            showtime_id=None,
            amc_movie_id=None,
            scheduled_for=dt.datetime.now(dt.timezone.utc),
            status="running",
            priority=0,
            attempt_count=1,
            max_attempts=3,
        )
        exc = ValueError("Could not find showtime.seatingLayout.seats in AMC RSC payload")

        self.assertFalse(queue.is_seat_backoff_failure(task, exc))
        self.assertEqual(60, queue.retry_delay_seconds_for_failure(task, exc))
        self.assertIsNone(queue.minimum_max_attempts_for_failure(task, exc))

    def test_unavailable_seat_map_is_not_backoff(self) -> None:
        run_id = uuid.uuid4()
        task = CollectionTask(
            task_id=42,
            run_id=run_id,
            task_type="collect_seat_snapshot",
            amc_theatre_id=None,
            showtime_id="100",
            amc_movie_id="movie-1",
            scheduled_for=dt.datetime.now(dt.timezone.utc),
            status="running",
            priority=5,
            attempt_count=1,
            max_attempts=3,
        )
        exc = amc.SeatMapUnavailable("AMC RSC payload has no reserved seat map for this showtime")

        self.assertFalse(queue.is_seat_backoff_failure(task, exc))
        self.assertEqual(60, queue.retry_delay_seconds_for_failure(task, exc))
        self.assertIsNone(queue.minimum_max_attempts_for_failure(task, exc))

    def test_seat_backoff_failure_extends_shared_throttle(self) -> None:
        run_id = uuid.uuid4()
        task = CollectionTask(
            task_id=42,
            run_id=run_id,
            task_type="collect_seat_snapshot",
            amc_theatre_id=None,
            showtime_id="100",
            amc_movie_id="movie-1",
            scheduled_for=dt.datetime.now(dt.timezone.utc),
            status="running",
            priority=5,
            attempt_count=1,
            max_attempts=3,
        )
        exc = ValueError("Could not find showtime object in AMC RSC payload")
        observed_at = dt.datetime(2026, 7, 1, 23, 0, tzinfo=dt.timezone.utc)
        blocked_until = observed_at + dt.timedelta(seconds=75)

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch.object(diagnostics, "BACKOFF_LOG_PATH", Path(tmp_dir) / "events.jsonl") as log_path,
            patch.object(queue.db, "utc_now", return_value=observed_at),
            patch.object(queue.db, "active_throttle_until", return_value=None) as active_throttle_until,
            patch.object(queue.db, "extend_throttle", return_value=blocked_until) as extend_throttle,
            patch.object(queue.db, "mark_task_failed") as mark_task_failed,
        ):
            queue.schedule_retry_or_fail(object(), task, exc)
            rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

        active_throttle_until.assert_called_once()
        extend_throttle.assert_called_once()
        self.assertEqual(queue.SEAT_COLLECTION_THROTTLE_KEY, extend_throttle.call_args.args[1])
        self.assertEqual(75, extend_throttle.call_args.kwargs["delay_seconds"])
        self.assertEqual(observed_at, extend_throttle.call_args.kwargs["now"])
        mark_task_failed.assert_called_once()
        self.assertEqual("shared_seat_collection_backoff", rows[0]["event_type"])
        self.assertEqual("all_workers", rows[0]["throttle_scope"])
        self.assertTrue(rows[0]["workers_share_server_identity"])
        self.assertEqual(75, rows[0]["retry_delay_seconds"])
        self.assertEqual(75, rows[0]["throttle_extended_by_seconds"])
        self.assertNotIn("previous_blocked_until", rows[0])

    def test_worker_logs_shared_throttle_wait_and_release(self) -> None:
        blocked_until = dt.datetime(2026, 7, 1, 23, 1, 15, tzinfo=dt.timezone.utc)
        now = dt.datetime(2026, 7, 1, 23, 0, 0, tzinfo=dt.timezone.utc)
        later = dt.datetime(2026, 7, 1, 23, 1, 20, tzinfo=dt.timezone.utc)

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch.object(diagnostics, "BACKOFF_LOG_PATH", Path(tmp_dir) / "events.jsonl") as log_path,
            patch.object(worker.db, "utc_now", side_effect=[now, now, later]),
            patch.object(worker.db, "active_throttle_until", side_effect=[blocked_until, blocked_until, None]),
        ):
            first = worker.log_shared_seat_throttle_wait(object(), worker_id="local-0", previous_blocked_until=None)
            second = worker.log_shared_seat_throttle_wait(object(), worker_id="local-0", previous_blocked_until=first)
            released = worker.log_shared_seat_throttle_wait(object(), worker_id="local-0", previous_blocked_until=second)
            rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(blocked_until, first)
        self.assertEqual(blocked_until, second)
        self.assertIsNone(released)
        self.assertEqual(["shared_seat_collection_wait", "shared_seat_collection_backoff_released"], [row["event_type"] for row in rows])
        self.assertEqual(75, rows[0]["wait_remaining_seconds"])
        self.assertEqual(5, rows[1]["release_lag_seconds"])


if __name__ == "__main__":
    unittest.main()
