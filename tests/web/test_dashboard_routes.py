from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pm_box_office.sources.amc import diagnostics
from pm_box_office.web.routes import dashboard


class FakeRequest:
    def __init__(self, body: bytes) -> None:
        self._body = body
        self.headers: dict[str, str] = {}

    async def body(self) -> bytes:
        return self._body


class DashboardRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_toggle_movie_parses_checkbox_body_without_multipart_dependency(self) -> None:
        conn = Mock()
        with (
            patch.object(dashboard, "connect_database", return_value=conn),
            patch.object(dashboard, "ensure_initialized") as ensure_initialized,
            patch.object(dashboard.movie_service, "set_movie_selected") as set_movie_selected,
        ):
            response = await dashboard.toggle_movie(
                "2026-06-30",
                "disclosure-day-77140",
                FakeRequest(b"selected=on"),
            )

        self.assertEqual(303, response.status_code)
        self.assertEqual("/campaigns/2026-06-30", response.headers["location"])
        ensure_initialized.assert_called_once_with(conn)
        set_movie_selected.assert_called_once()
        self.assertTrue(set_movie_selected.call_args.kwargs["selected"])
        conn.commit.assert_called_once()
        conn.close.assert_called_once()

    async def test_toggle_movie_treats_missing_checkbox_as_unselected(self) -> None:
        conn = Mock()
        with (
            patch.object(dashboard, "connect_database", return_value=conn),
            patch.object(dashboard, "ensure_initialized"),
            patch.object(dashboard.movie_service, "set_movie_selected") as set_movie_selected,
        ):
            await dashboard.toggle_movie(
                "2026-06-30",
                "disclosure-day-77140",
                FakeRequest(b""),
            )

        self.assertFalse(set_movie_selected.call_args.kwargs["selected"])

    async def test_bulk_movies_ignores_removed_select_top_action(self) -> None:
        conn = Mock()
        movies = [
            Mock(amc_movie_id="movie-1"),
            Mock(amc_movie_id="movie-2"),
            Mock(amc_movie_id="movie-3"),
        ]
        with (
            patch.object(dashboard, "connect_database", return_value=conn),
            patch.object(dashboard, "ensure_initialized"),
            patch.object(dashboard.movie_service, "list_movies_for_date", return_value=movies),
            patch.object(dashboard.movie_service, "set_movies_selected") as set_movies_selected,
        ):
            response = await dashboard.bulk_movies(
                "2026-06-30",
                FakeRequest(b"action=select_top&limit=2"),
            )

        self.assertEqual(303, response.status_code)
        set_movies_selected.assert_not_called()
        conn.commit.assert_called_once()
        conn.close.assert_called_once()

    async def test_bulk_movies_selects_the_numbers_active_movies(self) -> None:
        conn = Mock()
        with (
            patch.object(dashboard, "connect_database", return_value=conn),
            patch.object(dashboard, "ensure_initialized"),
            patch.object(dashboard.movie_service, "list_movies_for_date", return_value=[]),
            patch.object(dashboard.movie_service, "select_the_numbers_active_movies") as select_active,
        ):
            response = await dashboard.bulk_movies(
                "2026-07-02",
                FakeRequest(b"action=select_the_numbers_active"),
            )

        self.assertEqual(303, response.status_code)
        select_active.assert_called_once()
        self.assertEqual(7, select_active.call_args.kwargs["lookback_days"])
        conn.commit.assert_called_once()
        conn.close.assert_called_once()

    def test_local_worker_status_reads_configured_worker_settings(self) -> None:
        with patch.dict(
            dashboard.os.environ,
            {
                "AMC_LOCAL_WORKER_COUNT": "3",
                "AMC_LOCAL_WORKER_MAX": "3",
                "AMC_WORKER_BATCH_LIMIT": "4",
                "AMC_WORKER_DELAY_SECONDS": "0.25",
            },
            clear=False,
        ), patch.object(
            dashboard,
            "local_worker_slot_status",
            side_effect=[
                {"index": 0, "fresh": True, "pid": 101},
                {"index": 1, "fresh": False, "pid": None},
                {"index": 2, "fresh": True, "pid": 103},
            ],
        ):
            status = dashboard.local_worker_status()

        self.assertEqual(3, status["desired_count"])
        self.assertEqual(3, status["max_count"])
        self.assertEqual(2, status["running_count"])
        self.assertEqual([101, None, 103], [slot["pid"] for slot in status["slots"]])
        self.assertEqual(4, status["batch_limit"])
        self.assertEqual(0.25, status["delay_seconds"])

    def test_local_worker_status_exposes_conservative_defaults(self) -> None:
        with patch.dict(dashboard.os.environ, {}, clear=True), patch.object(
            dashboard,
            "local_worker_slot_status",
            return_value={"index": 0, "fresh": False, "pid": None},
        ):
            status = dashboard.local_worker_status()

        self.assertEqual(1, status["desired_count"])
        self.assertEqual(2, status["max_count"])
        self.assertEqual(1, status["batch_limit"])
        self.assertEqual(3.0, status["delay_seconds"])
        self.assertEqual(220, status["peak_per_worker"])
        self.assertEqual(1, status["backoff_cap"])

    def test_local_worker_slot_status_exposes_running_pid(self) -> None:
        with (
            patch.object(dashboard, "local_worker_pid", return_value=4321),
            patch.object(dashboard, "worker_heartbeat_age_seconds", return_value=7.4),
            patch.object(dashboard, "pid_is_running", return_value=True),
        ):
            status = dashboard.local_worker_slot_status(2)

        self.assertEqual(2, status["index"])
        self.assertEqual("local-2", status["worker_id"])
        self.assertEqual(4321, status["pid"])
        self.assertEqual("running", status["status"])
        self.assertTrue(status["fresh"])
        self.assertTrue(status["process_running"])
        self.assertEqual(7.4, status["heartbeat_age_seconds"])

    def test_autoscaled_worker_target_uses_baseline_without_backlog(self) -> None:
        with patch.dict(
            dashboard.os.environ,
            {"AMC_LOCAL_WORKER_COUNT": "1", "AMC_LOCAL_WORKER_MAX": "2"},
            clear=True,
        ):
            target = dashboard.autoscaled_worker_target({"due_now": 0, "late": 0})

        self.assertEqual(1, target)

    def test_autoscaled_worker_target_defaults_ramp_conservatively(self) -> None:
        with patch.dict(dashboard.os.environ, {}, clear=True):
            modest_backlog = dashboard.autoscaled_worker_target({"due_now": 9, "late": 4})
            larger_backlog = dashboard.autoscaled_worker_target({"due_now": 11, "late": 6})
            sampled_peak = dashboard.autoscaled_worker_target(
                {"due_now": 0, "late": 0, "peak_tasks_per_minute": 297}
            )
            capped_backlog = dashboard.autoscaled_worker_target({"due_now": 1000, "late": 1000})

        self.assertEqual(1, modest_backlog)
        self.assertEqual(1, larger_backlog)
        self.assertEqual(2, sampled_peak)
        self.assertEqual(2, capped_backlog)

    def test_autoscaled_worker_target_scales_for_due_backlog(self) -> None:
        with patch.dict(
            dashboard.os.environ,
            {
                "AMC_LOCAL_WORKER_COUNT": "1",
                "AMC_LOCAL_WORKER_MAX": "4",
                "AMC_AUTOSCALE_DUE_PER_WORKER": "2",
            },
            clear=True,
        ):
            target = dashboard.autoscaled_worker_target({"due_now": 5, "late": 0})

        self.assertEqual(3, target)

    def test_autoscaled_worker_target_scales_late_backlog_more_aggressively(self) -> None:
        with patch.dict(
            dashboard.os.environ,
            {
                "AMC_LOCAL_WORKER_COUNT": "1",
                "AMC_LOCAL_WORKER_MAX": "4",
                "AMC_AUTOSCALE_LATE_PER_WORKER": "1",
            },
            clear=True,
        ):
            target = dashboard.autoscaled_worker_target({"due_now": 2, "late": 4})

        self.assertEqual(4, target)

    def test_autoscaled_worker_target_scales_for_peak_overlap(self) -> None:
        with patch.dict(
            dashboard.os.environ,
            {
                "AMC_LOCAL_WORKER_COUNT": "1",
                "AMC_LOCAL_WORKER_MAX": "2",
                "AMC_AUTOSCALE_PEAK_PER_WORKER": "220",
            },
            clear=True,
        ):
            target = dashboard.autoscaled_worker_target(
                {"due_now": 0, "late": 0, "peak_tasks_per_minute": 297}
            )

        self.assertEqual(2, target)

    def test_autoscaled_worker_target_caps_for_backoff_pressure(self) -> None:
        with patch.dict(
            dashboard.os.environ,
            {
                "AMC_LOCAL_WORKER_COUNT": "1",
                "AMC_LOCAL_WORKER_MAX": "2",
                "AMC_AUTOSCALE_BACKOFF_CAP": "1",
            },
            clear=True,
        ):
            target = dashboard.autoscaled_worker_target(
                {"due_now": 500, "late": 500, "peak_tasks_per_minute": 500},
                backoff_summary={"backoff_pressure_events": 4},
            )

        self.assertEqual(1, target)

    def test_autoscaled_worker_target_is_capped(self) -> None:
        with patch.dict(
            dashboard.os.environ,
            {
                "AMC_LOCAL_WORKER_COUNT": "1",
                "AMC_LOCAL_WORKER_MAX": "2",
                "AMC_AUTOSCALE_DUE_PER_WORKER": "2",
            },
            clear=True,
        ):
            target = dashboard.autoscaled_worker_target({"due_now": 100, "late": 0})

        self.assertEqual(2, target)

    def test_autoscaled_worker_target_can_be_disabled(self) -> None:
        with patch.dict(
            dashboard.os.environ,
            {
                "AMC_LOCAL_WORKER_COUNT": "1",
                "AMC_LOCAL_WORKER_MAX": "6",
                "AMC_AUTOSCALE_ENABLED": "false",
            },
            clear=True,
        ):
            target = dashboard.autoscaled_worker_target({"due_now": 100, "late": 100})

        self.assertEqual(1, target)

    def test_campaign_route_starts_autoscaled_worker_target(self) -> None:
        conn = Mock()
        conn.execute.side_effect = [
            Mock(fetchone=Mock(return_value=(12, "2026-07-01T12:00:00+00:00"))),
            Mock(fetchall=Mock(return_value=[])),
        ]
        movies = [
            Mock(selected=True, sampled_showtime_count=11),
            Mock(selected=False, sampled_showtime_count=7),
        ]
        queue_health = {"due_now": 5, "late": 0, "queued": 5, "tasks_per_minute": 0.0, "eta_minutes": None}
        with (
            patch.dict(
                dashboard.os.environ,
                {
                    "AMC_LOCAL_WORKER_COUNT": "1",
                    "AMC_LOCAL_WORKER_MAX": "6",
                    "AMC_AUTOSCALE_DUE_PER_WORKER": "2",
                },
                clear=True,
            ),
            patch.object(dashboard, "connect_database", return_value=conn),
            patch.object(dashboard, "ensure_initialized"),
            patch.object(dashboard.db, "ensure_campaign", return_value="campaign-1"),
            patch.object(dashboard.movie_service, "list_movies_for_date", return_value=movies),
            patch.object(dashboard.db, "campaign_queue_health", return_value=queue_health),
            patch.object(dashboard.sample_service, "ensure_default_theatre_sample", return_value=Mock()),
            patch.object(
                dashboard.sample_service,
                "sample_coverage",
                return_value={"full_showtimes": 100, "sampled_showtimes": 40},
            ),
            patch.object(dashboard.db, "theatre_sample_showtime_overlap", return_value={}),
            patch.object(dashboard, "recent_backoff_summary", return_value={"backoff_pressure_events": 0}),
            patch.object(dashboard, "ensure_local_workers_started") as ensure_workers,
            patch.object(dashboard, "local_worker_status", return_value={"running_count": 3}) as worker_status,
            patch.object(dashboard.templates, "TemplateResponse", return_value=Mock()) as template_response,
        ):
            dashboard.campaign(Mock(), "2026-07-01")

        ensure_workers.assert_called_once_with(target_count=3)
        worker_status.assert_called_once_with(target_count=3)
        template_response.assert_called_once()
        context = template_response.call_args.kwargs["context"]
        self.assertEqual(2, context["inventory_movie_count"])
        self.assertEqual(1, context["selected_movie_count"])
        self.assertEqual(11, context["selected_sampled_showtimes"])
        self.assertEqual(100, context["inventory_showtimes"])
        self.assertEqual(40, context["sampled_showtimes"])
        conn.commit.assert_called_once()
        conn.close.assert_called_once()

    async def test_start_seat_collection_starts_autoscaled_worker_target(self) -> None:
        conn = Mock()
        queue_health = {"due_now": 2, "late": 4, "queued": 4, "tasks_per_minute": 0.0, "eta_minutes": None}
        with (
            patch.dict(
                dashboard.os.environ,
                {
                    "AMC_LOCAL_WORKER_COUNT": "1",
                    "AMC_LOCAL_WORKER_MAX": "6",
                    "AMC_AUTOSCALE_LATE_PER_WORKER": "1",
                },
                clear=True,
            ),
            patch.object(dashboard, "connect_database", return_value=conn),
            patch.object(dashboard, "ensure_initialized"),
            patch.object(dashboard.movie_service, "create_seat_collection_run") as create_run,
            patch.object(dashboard.db, "ensure_campaign", return_value="campaign-1"),
            patch.object(dashboard.db, "campaign_queue_health", return_value=queue_health),
            patch.object(dashboard, "recent_backoff_summary", return_value={"backoff_pressure_events": 0}),
            patch.object(dashboard, "ensure_local_workers_started") as ensure_workers,
        ):
            response = await dashboard.start_seat_collection(
                "2026-07-01",
                FakeRequest(b"sample_key=balanced_175"),
            )

        self.assertEqual(303, response.status_code)
        self.assertEqual("/campaigns/2026-07-01", response.headers["location"])
        create_run.assert_called_once()
        self.assertNotIn("use_theatre_sample", create_run.call_args.kwargs)
        self.assertEqual("balanced_175", create_run.call_args.kwargs["sample_key"])
        ensure_workers.assert_called_once_with(target_count=4)
        conn.commit.assert_called_once()
        conn.close.assert_called_once()

    async def test_start_seat_collection_ignores_legacy_full_coverage_scope(self) -> None:
        conn = Mock()
        queue_health = {"due_now": 0, "late": 0, "queued": 0, "tasks_per_minute": 0.0, "eta_minutes": None}
        with (
            patch.object(dashboard, "connect_database", return_value=conn),
            patch.object(dashboard, "ensure_initialized"),
            patch.object(dashboard.movie_service, "create_seat_collection_run") as create_run,
            patch.object(dashboard.db, "ensure_campaign", return_value="campaign-1"),
            patch.object(dashboard.db, "campaign_queue_health", return_value=queue_health),
            patch.object(dashboard, "recent_backoff_summary", return_value={"backoff_pressure_events": 0}),
            patch.object(dashboard, "ensure_local_workers_started"),
        ):
            await dashboard.start_seat_collection(
                "2026-07-01",
                FakeRequest(b"collection_scope=full&sample_key=balanced_175"),
            )

        self.assertNotIn("use_theatre_sample", create_run.call_args.kwargs)
        self.assertEqual("balanced_175", create_run.call_args.kwargs["sample_key"])

    def test_recent_backoff_summary_treats_missing_seats_as_backoff_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            now = dashboard.dt.datetime.now(dashboard.dt.timezone.utc)
            path = Path(tmp_dir) / "amc_backoff_events.jsonl"
            rows = [
                {"timestamp_utc": now.isoformat(), "event_type": "rsc_missing_seats"},
                {"timestamp_utc": now.isoformat(), "event_type": "http_retry", "status_code": 429},
                {"timestamp_utc": now.isoformat(), "event_type": "seat_task_failed"},
                {"timestamp_utc": now.isoformat(), "event_type": "rsc_fallback_succeeded"},
            ]
            path.write_text("\n".join(dashboard.json.dumps(row) for row in rows), encoding="utf-8")

            summary = dashboard.recent_backoff_summary(path=path)

        self.assertEqual(4, summary["events"])
        self.assertEqual(1, summary["rsc_missing_seats"])
        self.assertEqual(1, summary["seat_task_failed"])
        self.assertEqual(1, summary["throttling_events"])
        self.assertEqual(2, summary["seat_backoff_events"])
        self.assertEqual(1, summary["http_backoff_events"])
        self.assertEqual(3, summary["backoff_pressure_events"])
        self.assertEqual(1, summary["successful_fallbacks"])
        self.assertEqual(0.75, summary["backoff_pressure_rate"])

    def test_dashboard_template_renders_decluttered_operator_view(self) -> None:
        template = dashboard.templates.env.get_template("dashboard.html")
        selected_movie = Mock(
            selected=True,
            amc_movie_id="movie-1",
            amc_movie_name="Sample One",
            sampled_theatre_count=12,
            theatre_count=30,
            sampled_showtime_count=55,
            showtime_count=144,
        )
        unselected_movie = Mock(
            selected=False,
            amc_movie_id="movie-2",
            amc_movie_name="Sample Two",
            sampled_theatre_count=8,
            theatre_count=20,
            sampled_showtime_count=34,
            showtime_count=90,
        )
        html = template.render(
            request=Mock(),
            date_value="2026-07-01",
            movies=[selected_movie, unselected_movie],
            selected_movies=[selected_movie],
            selected_movie_count=1,
            selected_sampled_showtimes=55,
            inventory_movie_count=2,
            inventory_showtimes=31158,
            sampled_showtimes=12053,
            active_theatres=524,
            last_theatre_sync=None,
            recent_runs=[],
            worker_running=False,
            worker_status={
                "running_count": 1,
                "desired_count": 1,
                "max_count": 2,
                "batch_limit": 1,
                "delay_seconds": 3.0,
                "peak_per_worker": 220,
                "slots": [],
            },
            queue_health={
                "queued": 1,
                "due_now": 0,
                "late": 0,
                "tasks_per_minute": 0.0,
                "eta_minutes": None,
                "peak_tasks_per_minute": 723,
                "peak_scheduled_for": None,
            },
            sample_set=Mock(sample_key="balanced_175"),
            sample_coverage={
                "active_sample_theatres": 175,
                "active_theatres": 524,
                "sampled_showtimes": 12053,
                "full_showtimes": 31158,
                "sampled_screen_share": 0.38,
                "missing_snapshots": 9828,
                "sample_screens": 2677,
                "active_screens": 7045,
                "sample_states": 41,
                "sample_timezones": 7,
                "weighted_showtimes": 31245,
            },
            sample_overlap={"peak_tasks_per_minute": 311},
            backoff_summary={
                "backoff_pressure_events": 409,
                "events": 412,
                "rsc_missing_seats": 204,
                "seat_task_failed": 205,
                "successful_fallbacks": 3,
                "http_backoff_events": 0,
                "backoff_pressure_rate": 0.99,
            },
        )

        self.assertNotIn("Backoff pressure", html)
        self.assertNotIn("missing-seat backoffs", html)
        self.assertIn("Queue peak 723/min exceeds sampled peak 311/min", html)
        self.assertIn("Clear stale queue state before restarting sampled collection.", html)
        self.assertIn("Select active", html)
        self.assertNotIn("Select The Numbers active", html)
        self.assertIn("Start sampled seat collection", html)
        self.assertNotIn("Cancel active queue", html)
        self.assertNotIn("delay 3.00s", html)
        self.assertNotIn("<button type=\"submit\">Select all</button>", html)
        self.assertNotIn("Select top", html)
        self.assertNotIn("select_top", html)
        self.assertNotIn("name=\"lookback_days\"", html)
        self.assertIn("<details class=\"operator-details\">", html)
        self.assertIn("<span>Movies to track</span>", html)
        self.assertNotIn("1 selected · 2 inventory movies", html)
        self.assertNotIn("<span>All inventory movies</span>", html)
        self.assertIn("<details class=\"operator-details diagnostics-detail\">", html)
        self.assertIn("Refresh theatre list", html)
        self.assertIn("Forecasts", html)
        self.assertIn("Pull data", html)
        self.assertIn("shaded-link", html)
        self.assertNotIn("Ingest sources", html)
        self.assertNotIn("<h1>Maintenance</h1>", html)
        self.assertNotIn("<h1>Theatre sample details</h1>", html)
        self.assertIn("<h1>Backend workers</h1>", html)
        self.assertIn("<h1>Recent runs</h1>", html)
        self.assertIn("Showtimes", html)
        self.assertIn("12053 / 31158", html)
        self.assertIn("screen share:", html)
        self.assertNotIn("<span class=\"label\">Screen share</span>", html)
        self.assertNotIn("Snapshots missing", html)
        self.assertNotIn("States", html)
        self.assertNotIn("Timezones", html)
        self.assertIn("<h1>Worker Log</h1>", html)
        self.assertNotIn("<span>Log output</span>", html)
        self.assertNotIn("<span>Clear worker log</span>", html)
        self.assertIn("Clear worker log", html)
        self.assertNotIn("Backoff diagnostics", html)
        self.assertNotIn("Backoff log", html)
        self.assertNotIn("Clear backoff log", html)
        self.assertNotIn("HTTP throttle diagnostics", html)

    def test_dashboard_template_hides_zero_backoff_headline(self) -> None:
        template = dashboard.templates.env.get_template("dashboard.html")
        html = template.render(
            request=Mock(),
            date_value="2026-07-01",
            movies=[],
            selected_movies=[],
            selected_movie_count=0,
            selected_sampled_showtimes=0,
            inventory_movie_count=0,
            inventory_showtimes=0,
            sampled_showtimes=0,
            active_theatres=0,
            last_theatre_sync=None,
            recent_runs=[],
            worker_running=False,
            worker_status={
                "running_count": 1,
                "desired_count": 1,
                "max_count": 2,
                "batch_limit": 1,
                "delay_seconds": 3.0,
                "peak_per_worker": 220,
                "slots": [],
            },
            queue_health={
                "queued": 0,
                "due_now": 0,
                "late": 0,
                "tasks_per_minute": 0.0,
                "eta_minutes": None,
                "peak_tasks_per_minute": 0,
                "peak_scheduled_for": None,
            },
            sample_set=None,
            sample_coverage={},
            sample_overlap={},
            backoff_summary={
                "backoff_pressure_events": 0,
                "events": 0,
                "rsc_missing_seats": 0,
                "seat_task_failed": 0,
                "successful_fallbacks": 0,
                "http_backoff_events": 0,
                "backoff_pressure_rate": 0.0,
            },
        )

        self.assertNotIn("1h: 0 missing-seat backoffs", html)

    def test_progress_template_includes_dismiss_action(self) -> None:
        template = dashboard.templates.env.get_template("progress.html")
        html = template.render(
            request=Mock(),
            progress={
                "run_id": "11111111-1111-1111-1111-111111111111",
                "run_type": "seat_collection",
                "status": "running",
                "percent": 20,
                "total": 10,
                "queued": 7,
                "running": 1,
                "succeeded": 2,
                "failed": 0,
                "due_now": 0,
                "late": 0,
                "running_worker_id": None,
                "oldest_running_task_id": None,
                "oldest_running_at": None,
                "next_scheduled_for": None,
                "last_error_type": None,
                "last_error_message": None,
                "events": [],
            },
        )

        self.assertIn("/runs/11111111-1111-1111-1111-111111111111/dismiss", html)
        self.assertIn("Dismiss", html)

    def test_dashboard_dismissed_runs_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "dismissed.json"
            dashboard.remember_dashboard_dismissed_run(
                "11111111-1111-1111-1111-111111111111",
                path=path,
            )

            dismissed = dashboard.dashboard_dismissed_run_ids(path=path)

        self.assertEqual({"11111111-1111-1111-1111-111111111111"}, dismissed)

    def test_clear_log_file_empties_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "log.txt"
            path.write_text("line\n", encoding="utf-8")

            dashboard.clear_log_file(path)

            self.assertEqual("", path.read_text(encoding="utf-8"))

    def test_render_backoff_log_tail_escapes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "amc_backoff_events.jsonl"
            path.write_text('{"event_type":"http_failed","error_message":"<bad>"}\n', encoding="utf-8")
            with patch.object(diagnostics, "BACKOFF_LOG_PATH", path), patch.object(dashboard, "BACKOFF_LOG_PATH", path):
                html = dashboard.render_backoff_log_tail()

        self.assertIn("http_failed", html)
        self.assertIn("&lt;bad&gt;", html)


if __name__ == "__main__":
    unittest.main()
