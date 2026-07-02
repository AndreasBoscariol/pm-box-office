from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from pm_box_office.web.routes import forecasts


class FakeRequest:
    headers: dict[str, str] = {}
    query_params: dict[str, str] = {}
    scope = {"type": "http", "path": "/forecasts"}


class ForecastRouteTests(unittest.TestCase):
    def test_forecasts_page_renders_search_and_chart_shell(self) -> None:
        response = forecasts.forecasts_page(FakeRequest())

        self.assertEqual("forecasts.html", response.template.name)
        self.assertIn("selected_movie_id", response.context)
        template_source = response.template.environment.loader.get_source(  # type: ignore[union-attr]
            response.template.environment,
            "forecasts.html",
        )[0]
        self.assertIn('id="latest-model"', template_source)
        self.assertIn("latest.model", template_source)
        self.assertIn('id="target-selector"', template_source)
        self.assertIn("Opening Window", template_source)

    def test_movie_search_returns_service_candidates(self) -> None:
        conn = Mock()
        with (
            patch.object(forecasts, "connect_database", return_value=conn),
            patch.object(
                forecasts.forecast_service,
                "search_forecast_movies",
                return_value=[{"movie_id": 10, "title": "Example Movie"}],
            ) as search_movies,
        ):
            response = forecasts.search_movies(q="example", limit=12)

        self.assertEqual(200, response.status_code)
        self.assertEqual([{"movie_id": 10, "title": "Example Movie"}], response.body and __import__("json").loads(response.body)["movies"])
        search_movies.assert_called_once_with(conn, query="example", limit=12)
        conn.rollback.assert_called_once()
        conn.close.assert_called_once()

    def test_movie_forecast_endpoint_returns_selected_movie_payload(self) -> None:
        conn = Mock()
        payload = {
            "movie": {"movie_id": 10, "title": "Example Movie"},
            "snapshots": [
                {
                    "status": "ok",
                    "bop_forecast_available": True,
                    "wiki_available": True,
                    "competition_available": True,
                }
            ],
            "latest_snapshot": {"status": "ok"},
            "chart_svg": "<svg></svg>",
            "target_type": "3_day",
            "target_days": 3,
            "target_start_date": "2026-07-03",
            "target_end_date": "2026-07-05",
            "deployment_status": "production",
        }
        with (
            patch.object(forecasts, "connect_database", return_value=conn),
            patch.object(forecasts.forecast_service, "forecast_movie", return_value=payload),
        ):
            response = forecasts.movie_forecast(10)

        self.assertEqual(200, response.status_code)
        body = __import__("json").loads(response.body)
        self.assertEqual("Example Movie", body["movie"]["title"])
        self.assertEqual("3_day", body["target_type"])
        self.assertEqual("production", body["deployment_status"])
        self.assertEqual(
            {
                "bop_days": 1,
                "wiki_days": 1,
                "competition_days": 1,
                "amc_days": 0,
                "actual_days": 0,
                "forecast_days": 1,
            },
            body["feature_availability"],
        )
        conn.rollback.assert_called_once()
        conn.close.assert_called_once()

    def test_movie_forecast_endpoint_preserves_live_selected_model(self) -> None:
        conn = Mock()
        payload = {
            "movie": {"movie_id": 10, "title": "Example Movie"},
            "snapshots": [
                {
                    "status": "ok",
                    "model": "bop_residual_wiki_snapshot",
                    "bop_forecast_available": True,
                    "wiki_available": True,
                    "competition_available": True,
                }
            ],
            "latest_snapshot": {"status": "ok", "model": "bop_residual_wiki_snapshot"},
            "chart_svg": "<svg></svg>",
        }
        with (
            patch.object(forecasts, "connect_database", return_value=conn),
            patch.object(forecasts.forecast_service, "forecast_movie", return_value=payload),
        ):
            response = forecasts.movie_forecast(10)

        body = __import__("json").loads(response.body)
        self.assertEqual("bop_residual_wiki_snapshot", body["latest_snapshot"]["model"])
        self.assertEqual("bop_residual_wiki_snapshot", body["snapshots"][0]["model"])


if __name__ == "__main__":
    unittest.main()
