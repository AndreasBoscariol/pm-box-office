from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from web.routes import dashboard


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


if __name__ == "__main__":
    unittest.main()
