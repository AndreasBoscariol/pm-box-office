from __future__ import annotations

import datetime as dt
import unittest
from decimal import Decimal
from unittest.mock import Mock, patch

from pm_box_office.web.routes import forecasts as forecast_routes
from pm_box_office.web.services import forecasts
from pm_box_office.web.templating import templates


class FakeCursor:
    def __init__(self, columns: list[str], rows: list[tuple[object, ...]]) -> None:
        self.description = [(column,) for column in columns]
        self._rows = rows

    def fetchone(self) -> tuple[object, ...] | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[tuple[object, ...]]:
        return self._rows


class ForecastServiceTests(unittest.TestCase):
    def test_normalize_filters_defaults_to_opening_weekend_and_latest_model(self) -> None:
        result = forecasts.normalize_filters(
            {"q": "  Superman  ", "target": "bad-value", "model_version": ""},
            default_model_version="boxoffice_2026_07_09_001",
        )

        self.assertEqual("Superman", result.q)
        self.assertEqual("opening_weekend", result.target)
        self.assertEqual("boxoffice_2026_07_09_001", result.model_version)

    def test_pending_timeline_fills_missing_expected_origins(self) -> None:
        timeline = forecasts.pending_timeline(
            [
                {
                    "forecast_id": "f1",
                    "origin_key": "P_-14",
                    "regime": "pre_release",
                    "target": "opening_weekend",
                    "components": [],
                },
                {
                    "forecast_id": "f2",
                    "origin_key": "FRI_10:00",
                    "regime": "live_friday",
                    "target": "opening_weekend",
                    "components": [],
                },
            ]
        )

        self.assertEqual(35, len(timeline))
        self.assertEqual("ready", timeline[0]["status"])
        self.assertEqual("pending", timeline[1]["status"])
        self.assertEqual("ready", [row for row in timeline if row["origin_key"] == "FRI_10:00"][0]["status"])
        self.assertEqual("live_sunday", timeline[-1]["regime"])

    def test_latest_opening_weekend_row_uses_timeline_order(self) -> None:
        latest = forecasts.latest_opening_weekend_row(
            [
                {"origin_key": "P_-1", "target": "opening_weekend", "point_usd": 1},
                {"origin_key": "SAT_12:00", "target": "opening_weekend", "point_usd": 2},
                {"origin_key": "FRI_EOD", "target": "opening_weekend", "point_usd": 3},
            ]
        )

        self.assertEqual(2, latest["point_usd"])

    def test_dashboard_selects_latest_model_when_filter_is_empty(self) -> None:
        conn = Mock()
        with (
            patch.object(forecasts, "forecast_tables_exist", return_value=True),
            patch.object(forecasts, "model_versions", return_value=["boxoffice_latest", "boxoffice_old"]),
            patch.object(forecasts, "dashboard_summary", return_value=forecasts.empty_summary()) as summary,
            patch.object(forecasts, "dashboard_rows", return_value=[]) as rows,
        ):
            result = forecasts.dashboard(conn, forecasts.ForecastFilters())

        self.assertEqual("boxoffice_latest", result["filters"].model_version)
        summary.assert_called_once()
        rows.assert_called_once()

    def test_forecast_chart_builds_point_line_and_interval_bands(self) -> None:
        timeline = forecasts.pending_timeline(
            [
                {
                    "forecast_id": "f1",
                    "origin_key": "P_-14",
                    "regime": "pre_release",
                    "target": "opening_weekend",
                    "point_usd": 100_000_000,
                    "lo80_usd": 90_000_000,
                    "hi80_usd": 110_000_000,
                    "lo95_usd": 80_000_000,
                    "hi95_usd": 120_000_000,
                    "components": [],
                },
                {
                    "forecast_id": "f2",
                    "origin_key": "P_-13",
                    "regime": "pre_release",
                    "target": "opening_weekend",
                    "point_usd": 105_000_000,
                    "lo80_usd": 95_000_000,
                    "hi80_usd": 115_000_000,
                    "lo95_usd": 85_000_000,
                    "hi95_usd": 125_000_000,
                    "components": [],
                },
            ]
        )

        chart = forecasts.forecast_chart(timeline, {"actual_usd": 103_000_000})

        self.assertTrue(chart["has_data"])
        self.assertIn("M", chart["point_path"])
        self.assertIn("Z", chart["band80_path"])
        self.assertIn("Z", chart["band95_path"])
        self.assertIsNotNone(chart["actual_y"])

    def test_formatters_handle_null_actuals_and_compact_values(self) -> None:
        self.assertEqual("-", forecasts.compact_usd(None))
        self.assertEqual("$12.3M", forecasts.compact_usd(Decimal("12345000")))
        self.assertEqual("-", forecasts.percent(None))
        self.assertEqual("12.3%", forecasts.percent(Decimal("0.1234")))


class ForecastTemplateTests(unittest.TestCase):
    def test_dashboard_template_renders_empty_forecasts(self) -> None:
        html = templates.env.get_template("forecasts.html").render(
            request=None,
            tables_ready=True,
            summary=forecasts.empty_summary(),
            rows=[],
            model_versions=[],
            filters=forecasts.ForecastFilters(),
        )

        self.assertIn("No stored forecasts match these filters.", html)
        self.assertIn("Movie Lookup", html)

    def test_dashboard_template_renders_latest_model_summary(self) -> None:
        html = templates.env.get_template("forecasts.html").render(
            request=None,
            tables_ready=True,
            summary={
                "movie_count": 1,
                "latest_model_version": "boxoffice_test",
                "latest_run_time": dt.datetime(2026, 7, 9, tzinfo=dt.UTC),
                "row_count": 4,
                "backtest_rows": 4,
                "live_rows": 0,
            },
            rows=[
                {
                    "release_run_id": 10,
                    "model_version": "boxoffice_test",
                    "title": "Example Movie",
                    "opening_weekend_start": dt.date(2026, 7, 10),
                    "origin_key": "P_-1",
                    "as_of_utc": dt.datetime(2026, 7, 9, tzinfo=dt.UTC),
                    "point_usd": 100_000_000,
                    "lo80_usd": 80_000_000,
                    "hi80_usd": 120_000_000,
                    "lo95_usd": 70_000_000,
                    "hi95_usd": 140_000_000,
                    "actual_usd": None,
                    "log_error": None,
                    "abs_pct_error": None,
                    "component_source": "consensus",
                    "feature_quality_bucket": None,
                }
            ],
            model_versions=["boxoffice_test"],
            filters=forecasts.ForecastFilters(model_version="boxoffice_test"),
        )

        self.assertIn("boxoffice_test", html)
        self.assertIn("Example Movie", html)
        self.assertIn("$100.0M", html)
        self.assertIn("<span>-</span>", html)

    def test_detail_template_includes_component_rows(self) -> None:
        timeline = forecasts.pending_timeline(
            [
                {
                    "forecast_id": "f1",
                    "origin_key": "P_-14",
                    "regime": "pre_release",
                    "target": "opening_weekend",
                    "as_of_utc": dt.datetime(2026, 6, 26, tzinfo=dt.UTC),
                    "point_usd": 100_000_000,
                    "lo80_usd": 80_000_000,
                    "hi80_usd": 120_000_000,
                    "lo95_usd": 70_000_000,
                    "hi95_usd": 140_000_000,
                    "amc_coverage": None,
                    "amc_snapshot_count": None,
                    "feature_quality_bucket": None,
                    "actual_usd": None,
                    "log_error": None,
                    "abs_pct_error": None,
                    "components": [
                        {
                            "component_day": "Friday",
                            "component_type": "baseline",
                            "component_point_usd": 40_000_000,
                            "component_sigma_log": Decimal("0.2"),
                            "component_lo80_usd": 32_000_000,
                            "component_hi80_usd": 48_000_000,
                            "component_lo95_usd": 28_000_000,
                            "component_hi95_usd": 52_000_000,
                            "component_model": "daily_shape_model",
                            "component_source": "daily_baseline",
                            "component_notes": None,
                        }
                    ],
                }
            ]
        )
        html = templates.env.get_template("forecast_detail.html").render(
            request=None,
            release_run_id=10,
            tables_ready=True,
            movie={
                "title": "Example Movie",
                "release_run_id": 10,
                "opening_weekend_start": dt.date(2026, 7, 10),
            },
            latest={
                "point_usd": 100_000_000,
                "origin_key": "P_-14",
                "as_of_utc": dt.datetime(2026, 6, 26, tzinfo=dt.UTC),
                "lo80_usd": 80_000_000,
                "hi80_usd": 120_000_000,
                "lo95_usd": 70_000_000,
                "hi95_usd": 140_000_000,
                "interval_model": "normal_log_error_interval",
                "component_source": "consensus",
                "actual_usd": None,
                "log_error": None,
                "abs_pct_error": None,
                "regime": "pre_release",
                "run_id": "run_1",
            },
            timeline=timeline,
            chart=forecasts.forecast_chart(timeline, {"actual_usd": None}),
            model_versions=["boxoffice_test"],
            selected_model_version="boxoffice_test",
        )

        self.assertIn("<svg", html)
        self.assertIn("Point estimate", html)
        self.assertIn("P_-14 components", html)
        self.assertIn("daily_shape_model", html)
        self.assertIn("Origin Components", html)


class ForecastRouteTests(unittest.TestCase):
    def test_forecast_dashboard_route_renders_service_context(self) -> None:
        request = Mock(query_params={"target": "opening_weekend"})
        conn = Mock()
        payload = {
            "summary": forecasts.empty_summary(),
            "rows": [],
            "model_versions": [],
            "filters": forecasts.ForecastFilters(model_version="boxoffice_test"),
            "tables_ready": True,
        }
        with (
            patch.object(forecast_routes, "connect_database", return_value=conn),
            patch.object(forecast_routes.forecast_service, "latest_model_version", return_value="boxoffice_test"),
            patch.object(forecast_routes.forecast_service, "dashboard", return_value=payload) as dashboard,
            patch.object(forecast_routes.templates, "TemplateResponse", return_value="response") as template_response,
        ):
            response = forecast_routes.forecast_dashboard(request)

        self.assertEqual("response", response)
        dashboard.assert_called_once()
        template_response.assert_called_once()
        self.assertEqual("forecasts.html", template_response.call_args.kwargs["name"])
        self.assertTrue(template_response.call_args.kwargs["context"]["tables_ready"])
        conn.rollback.assert_called_once()
        conn.close.assert_called_once()

    def test_forecast_detail_route_renders_timeline_context(self) -> None:
        request = Mock(query_params={"model_version": "boxoffice_test"})
        conn = Mock()
        payload = {
            "movie": None,
            "latest": None,
            "timeline": [],
            "chart": forecasts.forecast_chart([], None),
            "model_versions": ["boxoffice_test"],
            "selected_model_version": "boxoffice_test",
            "tables_ready": True,
        }
        with (
            patch.object(forecast_routes, "connect_database", return_value=conn),
            patch.object(forecast_routes.forecast_service, "movie_timeline", return_value=payload) as movie_timeline,
            patch.object(forecast_routes.templates, "TemplateResponse", return_value="response") as template_response,
        ):
            response = forecast_routes.forecast_detail(request, 123)

        self.assertEqual("response", response)
        movie_timeline.assert_called_once_with(conn, release_run_id=123, model_version="boxoffice_test")
        self.assertEqual("forecast_detail.html", template_response.call_args.kwargs["name"])
        self.assertEqual(123, template_response.call_args.kwargs["context"]["release_run_id"])
        conn.rollback.assert_called_once()
        conn.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
