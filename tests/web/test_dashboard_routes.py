from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from pm_box_office.web.routes import dashboard


class FakeRequest:
    def __init__(self, body: bytes) -> None:
        self._body = body

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

    async def test_bulk_movies_selects_top_movies(self) -> None:
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
        set_movies_selected.assert_called_once()
        self.assertEqual(["movie-1", "movie-2"], set_movies_selected.call_args.kwargs["amc_movie_ids"])
        self.assertTrue(set_movies_selected.call_args.kwargs["selected"])
        conn.commit.assert_called_once()
        conn.close.assert_called_once()

    def test_local_worker_status_reads_configured_worker_settings(self) -> None:
        with patch.dict(
            dashboard.os.environ,
            {
                "AMC_LOCAL_WORKER_COUNT": "3",
                "AMC_WORKER_BATCH_LIMIT": "4",
                "AMC_WORKER_DELAY_SECONDS": "0.25",
            },
            clear=False,
        ), patch.object(dashboard, "worker_heartbeat_is_fresh", side_effect=[True, False, True]):
            status = dashboard.local_worker_status()

        self.assertEqual(3, status["desired_count"])
        self.assertEqual(2, status["running_count"])
        self.assertEqual(4, status["batch_limit"])
        self.assertEqual(0.25, status["delay_seconds"])


if __name__ == "__main__":
    unittest.main()
