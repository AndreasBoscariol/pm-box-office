from __future__ import annotations

import datetime as dt
import unittest
from decimal import Decimal
from pathlib import Path
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

        self.assertEqual(42, len(timeline))
        self.assertEqual("ready", timeline[0]["status"])
        self.assertEqual("pending", timeline[1]["status"])
        self.assertEqual("ready", [row for row in timeline if row["origin_key"] == "FRI_10:00"][0]["status"])
        self.assertEqual("pending", [row for row in timeline if row["origin_key"] == "THU_10:00"][0]["status"])
        self.assertEqual("live_sunday", timeline[-1]["regime"])

    def test_thursday_preview_virtual_timeline_rows_use_amc_intraday_and_reported_actual(self) -> None:
        rows = [
            {
                "forecast_id": "fri-preview",
                "origin_key": "FRI_10:00",
                "regime": "live_friday",
                "target": "opening_weekend",
                "point_usd": 108_000_000,
                "component_source": "thursday_preview_actual",
                "components": [{"component_day": "Friday", "component_point_usd": 36_000_000}],
            }
        ]
        timeline = forecasts.pending_timeline(rows)

        forecasts.inject_thursday_preview_timeline_rows(
            timeline,
            rows,
            {
                "reported": {"gross_usd": Decimal("8000000")},
                "amc_shadow_rows": [
                    {
                        "release_run_id": 10,
                        "movie_id": 1,
                        "title": "Example",
                        "origin_key": "THU_16:00",
                        "point_usd": None,
                        "thursday_amc_shadow_ow_prior_usd": Decimal("104000000"),
                        "thursday_amc_shadow_ow_prior_source": "thursday_amc_preview_nowcast",
                    }
                ],
                "latest_amc_shadow": None,
            },
        )

        intraday = [item for item in timeline if item["origin_key"] == "THU_16:00"][0]
        self.assertEqual("ready", intraday["status"])
        self.assertEqual("thursday_preview", intraday["regime"])
        self.assertEqual("thursday_amc_preview_nowcast", intraday["component_source"])
        self.assertEqual(104_000_000, intraday["point_usd"])
        eod = [item for item in timeline if item["origin_key"] == "THU_EOD"][0]
        self.assertEqual("thursday_preview_actual", eod["component_source"])
        self.assertEqual(108_000_000, eod["point_usd"])

    def test_thursday_origins_sort_between_pre_release_and_friday(self) -> None:
        self.assertLess(forecasts.origin_sort_value("P_-1"), forecasts.origin_sort_value("THU_10:00"))
        self.assertLess(forecasts.origin_sort_value("THU_EOD"), forecasts.origin_sort_value("FRI_10:00"))
        self.assertLess(forecasts.origin_sort_value("FRI_EOD"), forecasts.origin_sort_value("SAT_10:00"))

        chart = forecasts.forecast_chart(forecasts.pending_timeline([]), None)
        for label in chart["x_labels"]:
            self.assertGreaterEqual(label["x"], 0)
            self.assertLessEqual(label["x"], chart["width"])
        for marker in chart["regime_markers"]:
            self.assertGreaterEqual(marker["x"], 0)
            self.assertLessEqual(marker["x"], chart["width"])

    def test_latest_opening_weekend_row_uses_timeline_order(self) -> None:
        latest = forecasts.latest_opening_weekend_row(
            [
                {"origin_key": "P_-1", "target": "opening_weekend", "point_usd": 1},
                {"origin_key": "SAT_12:00", "target": "opening_weekend", "point_usd": 2},
                {"origin_key": "FRI_EOD", "target": "opening_weekend", "point_usd": 3},
            ]
        )

        self.assertEqual(2, latest["point_usd"])

    def test_suppress_stale_baseline_live_rows_after_latest_amc_origin(self) -> None:
        rows = [
            {
                "release_run_id": 10,
                "origin_key": "FRI_18:00",
                "target": "opening_weekend",
                "component_source": "AMC_plugin",
                "is_live": True,
            },
            {
                "release_run_id": 10,
                "origin_key": "FRI_20:00",
                "target": "opening_weekend",
                "component_source": "daily_baseline",
                "is_live": True,
            },
            {
                "release_run_id": 10,
                "origin_key": "FRI_20:00",
                "target": "friday",
                "component_source": "daily_baseline",
                "is_live": True,
            },
            {
                "release_run_id": 10,
                "origin_key": "P_-1",
                "target": "opening_weekend",
                "component_source": "consensus",
                "is_live": False,
            },
        ]

        forecasts.suppress_stale_baseline_live_rows(rows)

        self.assertEqual(
            [("FRI_18:00", "opening_weekend"), ("P_-1", "opening_weekend")],
            [(row["origin_key"], row["target"]) for row in rows],
        )

    def test_polymarket_market_grid_uses_validated_contract_boundaries(self) -> None:
        conn = Mock()

        def execute(sql: str, params: tuple[object, ...] | None = None) -> FakeCursor:
            if "to_regclass" in sql:
                return FakeCursor(["exists"], [(True,)])
            if "FROM prediction_market_backtest.movie_matches" in sql:
                return FakeCursor(
                    [
                        "event_id",
                        "event_slug",
                        "event_title",
                        "market_id",
                        "question",
                        "bucket_lower",
                        "bucket_upper",
                        "include_lower",
                        "include_upper",
                        "lower_unbounded",
                        "upper_unbounded",
                        "validated_bucket_count",
                    ],
                    [
                        ("e1", "movie-ow", "Movie Opening Weekend", "m1", "Under $10M", None, Decimal("10000000"), False, False, True, False, 5),
                        ("e1", "movie-ow", "Movie Opening Weekend", "m2", "$10M-$20M", Decimal("10000000"), Decimal("20000000"), True, False, False, False, 5),
                        ("e1", "movie-ow", "Movie Opening Weekend", "m3", "$20M-$30M", Decimal("20000000"), Decimal("30000000"), True, False, False, False, 5),
                        ("e1", "movie-ow", "Movie Opening Weekend", "m4", "$30M-$40M", Decimal("30000000"), Decimal("40000000"), True, False, False, False, 5),
                        ("e1", "movie-ow", "Movie Opening Weekend", "m5", "$40M or more", Decimal("40000000"), None, True, False, False, True, 5),
                    ],
                )
            raise AssertionError(sql)

        conn.execute.side_effect = execute

        grid = forecasts.polymarket_market_grid(
            conn,
            release_run_id=10,
            anchor={
                "movie_id": 1,
                "origin_key": "P_-10",
                "point_usd": Decimal("24000000"),
                "payload_hash": "hash-1",
                "point_model": "raw_consensus",
            },
        )

        self.assertIsNotNone(grid)
        assert grid is not None
        self.assertEqual("polymarket", grid.market_source)
        self.assertEqual((10_000_000, 20_000_000, 30_000_000, 40_000_000), grid.boundaries_usd)
        self.assertEqual(("m1", "m2", "m3", "m4", "m5"), grid.market_ids)
        self.assertEqual(
            "Under $10M",
            forecasts.bucket_label(0, grid.boundaries_usd, market_questions=grid.market_questions),
        )

    def test_polymarket_market_grid_rejects_incomplete_bucket_set(self) -> None:
        conn = Mock()

        def execute(sql: str, params: tuple[object, ...] | None = None) -> FakeCursor:
            if "to_regclass" in sql:
                return FakeCursor(["exists"], [(True,)])
            if "FROM prediction_market_backtest.movie_matches" in sql:
                return FakeCursor(
                    [
                        "event_id",
                        "event_slug",
                        "event_title",
                        "market_id",
                        "question",
                        "bucket_lower",
                        "bucket_upper",
                        "include_lower",
                        "include_upper",
                        "lower_unbounded",
                        "upper_unbounded",
                        "validated_bucket_count",
                    ],
                    [
                        ("e1", "movie-ow", "Movie Opening Weekend", "m1", "Under $10M", None, Decimal("10000000"), False, False, True, False, 4),
                        ("e1", "movie-ow", "Movie Opening Weekend", "m2", "$10M or more", Decimal("10000000"), None, True, False, False, True, 4),
                    ],
                )
            raise AssertionError(sql)

        conn.execute.side_effect = execute

        grid = forecasts.polymarket_market_grid(
            conn,
            release_run_id=10,
            anchor={
                "movie_id": 1,
                "origin_key": "P_-10",
                "point_usd": Decimal("24000000"),
                "payload_hash": "hash-1",
                "point_model": "raw_consensus",
            },
        )

        self.assertIsNone(grid)

    def test_promote_timeline_amc_live_rows_replaces_same_origin_baseline(self) -> None:
        columns = [
            "forecast_id",
            "release_run_id",
            "origin_key",
            "target",
            "component_source",
            "is_live",
            "point_usd",
        ]
        conn = Mock()
        conn.execute.return_value = FakeCursor(
            columns,
            [("amc-1", 10, "FRI_18:00", "opening_weekend", "AMC_plugin", True, 120)],
        )
        rows = [
            {
                "forecast_id": "base-1",
                "release_run_id": 10,
                "origin_key": "FRI_18:00",
                "target": "opening_weekend",
                "component_source": "daily_baseline",
                "is_live": True,
                "point_usd": 100,
            }
        ]

        forecasts.promote_timeline_amc_live_rows(conn, rows)

        self.assertEqual("amc-1", rows[0]["forecast_id"])
        self.assertEqual(120, rows[0]["point_usd"])

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

    def test_distribution_policy_context_reads_promoted_policy(self) -> None:
        context = forecasts.distribution_policy_context("boxoffice_local_007_retrained_from_db")

        self.assertEqual("production_distribution_v2_t4_tail_safe", context["policy_name"])
        self.assertEqual("production", context["status"])
        self.assertEqual("2% student_t df4", context["tail_label"])
        self.assertEqual("production_distribution_v1_raw", context["base_distribution_policy"])

    def test_dashboard_includes_distribution_policy_context(self) -> None:
        conn = Mock()
        with (
            patch.object(forecasts, "forecast_tables_exist", return_value=True),
            patch.object(forecasts, "model_versions", return_value=["boxoffice_latest"]),
            patch.object(forecasts, "dashboard_summary", return_value=forecasts.empty_summary()),
            patch.object(forecasts, "dashboard_rows", return_value=[]),
            patch.object(forecasts, "distribution_policy_context", return_value={"label": "v2"}) as distribution_context,
        ):
            result = forecasts.dashboard(conn, forecasts.ForecastFilters())

        self.assertEqual({"label": "v2"}, result["distribution_policy"])
        distribution_context.assert_called_once_with("boxoffice_latest")

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
        self.assertEqual(
            ["14 days out", "1 day out", "Thursday", "Friday", "Saturday", "Sunday"],
            [label["label"] for label in chart["x_labels"]],
        )

    def test_market_chart_uses_short_bucket_labels_and_contrasting_probability_text(self) -> None:
        grid = forecasts.generate_rounding_variants(
            release_run_id=10,
            effective_listing_origin="P_-10",
            anchor_forecast_usd=100_000_000,
            anchor_forecast_emission_hash="anchor",
            point_policy_version="test",
        )[0]
        timeline = forecasts.pending_timeline(
            [
                {
                    "forecast_id": "f1",
                    "origin_key": "P_-10",
                    "regime": "pre_release",
                    "target": "opening_weekend",
                    "point_usd": 100_000_000,
                    "components": [],
                }
            ]
        )
        chart = forecasts.market_chart(
            timeline,
            None,
            grid,
            {
                "P_-10": {
                    "status": "ready",
                    "probabilities": [0.10, 0.20, 0.30, 0.40, 0.00],
                    "distribution": None,
                }
            },
        )

        self.assertEqual(f"Below {forecasts.compact_usd(grid.boundaries_usd[0])}", chart["bucket_regions"][-1]["label"])
        self.assertEqual("dark", next(cell for cell in chart["cells"] if cell["probability"] == 0.10)["label_tone"])
        self.assertEqual("light", next(cell for cell in chart["cells"] if cell["probability"] == 0.40)["label_tone"])
        self.assertTrue(chart["has_probabilities"])
        self.assertEqual("P_-10", chart["latest_probability_origin"])
        self.assertEqual(0, sum(1 for cell in chart["cells"] if cell["show_label"]))
        self.assertEqual(5, len(chart["latest_probabilities"]))
        self.assertEqual("Below $85.0M", chart["latest_probabilities"][0]["label"])
        plot_right = chart["width"] - chart["right"]
        first_cell = min(chart["cells"], key=lambda cell: cell["x"])
        last_cell = max(chart["cells"], key=lambda cell: cell["x"])
        self.assertGreaterEqual(first_cell["x"], chart["listing_x"])
        self.assertLessEqual(last_cell["x"] + last_cell["width"], plot_right)
        self.assertNotEqual(forecasts.probability_color(0.10), forecasts.probability_color(0.40))
        bucket_heights = [region["height"] for region in chart["bucket_regions"]]
        self.assertLess(max(bucket_heights) - min(bucket_heights), 0.01)

    def test_component_breakdown_explains_amc_signal_and_uncertainty(self) -> None:
        rows = forecasts.forecast_component_breakdown(
            {
                "point_usd": 100_000_000,
                "origin_key": "FRI_16:00",
                "amc_snapshot_count": 12,
                "amc_coverage": Decimal("0.75"),
                "components": [
                    {
                        "component_day": "Friday",
                        "component_type": "AMC_nowcast",
                        "component_point_usd": 40_000_000,
                        "component_lo80_usd": 32_000_000,
                        "component_hi80_usd": 48_000_000,
                        "component_source": "AMC_plugin",
                        "component_model": "AMC_plugin",
                        "component_notes": "medium",
                    },
                    {
                        "component_day": "Saturday",
                        "component_type": "baseline",
                        "component_point_usd": 35_000_000,
                        "component_lo80_usd": 28_000_000,
                        "component_hi80_usd": 42_000_000,
                        "component_source": "daily_baseline",
                        "component_model": "after-Friday baseline",
                    },
                ],
            }
        )

        self.assertEqual("AMC seats and showtimes", rows[0]["source"])
        self.assertEqual("Updated FRI 16:00", rows[0]["window"])
        self.assertIn("12 snapshots", rows[0]["source_detail"])
        self.assertIn("75.0% coverage", rows[0]["source_detail"])
        self.assertEqual(0.4, rows[0]["share"])
        self.assertEqual("80% likely range", rows[0]["uncertainty_label"])
        self.assertEqual("Historical daily pattern", rows[1]["source"])

    def test_pre_release_breakdown_lists_consensus_sources_and_windows(self) -> None:
        timeline = forecasts.pending_timeline(
            [
                {
                    "forecast_id": "pre-1",
                    "origin_key": "P_-10",
                    "regime": "pre_release",
                    "target": "opening_weekend",
                    "point_usd": 100_000_000,
                    "lo80_usd": 80_000_000,
                    "hi80_usd": 120_000_000,
                    "source_count": 2,
                    "estimate_sources": "boxofficepro, the_numbers",
                    "components": [],
                }
            ]
        )

        rows = forecasts.pre_release_breakdown(timeline)

        self.assertEqual("10 days out", rows[0]["window"])
        self.assertEqual(2, rows[0]["source_count"])
        self.assertEqual(["Boxoffice Pro", "The Numbers"], rows[0]["sources"])

    def test_formatters_handle_null_actuals_and_compact_values(self) -> None:
        self.assertEqual("-", forecasts.compact_usd(None))
        self.assertEqual("$12.3M", forecasts.compact_usd(Decimal("12345000")))
        self.assertEqual("-", forecasts.percent(None))
        self.assertEqual("12.3%", forecasts.percent(Decimal("0.1234")))

    def test_thursday_preview_context_fetches_reported_and_amc_shadow_data(self) -> None:
        shadow_columns = [
            "release_run_id", "movie_id", "title", "origin_key", "forecast_origin", "forecast_origin_utc",
            "as_of_utc", "model_version", "run_id", "sample_key", "thursday_amc_seats_collected",
            "predicted_thursday_previews_usd", "thursday_amc_observed_preview_seats",
            "thursday_amc_shadow_ow_prior_usd", "thursday_amc_shadow_ow_prior_source",
            "thursday_preview_actual_usd", "friday_only_amc_observed_seats", "coverage", "snapshot_count",
            "staleness_p50_minutes", "feature_quality_bucket", "amc_candidate_available",
            "amc_candidate_unavailable_reason", "opening_weekend_usd", "production_daily_usd",
        ]
        shadow_row = (
            10, 1, "Example", "FRI_16:00", "16:00", dt.datetime(2026, 7, 10, 20, tzinfo=dt.UTC),
            dt.datetime(2026, 7, 10, 20, tzinfo=dt.UTC), "boxoffice_test", "run_1", "top_hybrid_30",
            True, Decimal("8000000"), Decimal("1234"), Decimal("108000000"),
            "thursday_amc_preview_nowcast", Decimal("8500000"), Decimal("9000"), Decimal("0.75"),
            12, Decimal("4"), "medium", True, None, Decimal("100000000"), Decimal("32000000"),
        )
        conn = Mock()
        conn.execute.side_effect = [
            FakeCursor(["exists"], [(True,)]),
            FakeCursor(
                ["preview_date", "gross_usd", "source", "source_url", "fetched_at"],
                [(dt.date(2026, 7, 9), Decimal("8500000"), "the_numbers", None, dt.datetime(2026, 7, 10, tzinfo=dt.UTC))],
            ),
            FakeCursor(["exists"], [(True,)]),
            FakeCursor(["column_name"], [(column,) for column in shadow_columns]),
            FakeCursor(shadow_columns, [shadow_row]),
            FakeCursor(["exists"], [(True,)]),
            FakeCursor(["column_name"], [(column,) for column in shadow_columns]),
            FakeCursor(shadow_columns, [shadow_row]),
        ]

        context = forecasts.thursday_preview_context(conn, release_run_id=10)

        self.assertEqual(Decimal("8500000"), context["reported"]["gross_usd"])
        self.assertEqual(Decimal("8000000"), context["latest_amc_shadow"]["predicted_thursday_previews_usd"])
        self.assertEqual(Decimal("108000000"), context["latest_amc_shadow"]["thursday_amc_shadow_ow_prior_usd"])

    def test_amc_shadow_preview_rows_tolerates_unapplied_thursday_migration(self) -> None:
        legacy_columns = [
            "release_run_id", "movie_id", "title", "origin_key", "forecast_origin", "forecast_origin_utc",
            "as_of_utc", "model_version", "run_id", "sample_key", "thursday_amc_seats_collected",
            "predicted_thursday_previews_usd", "thursday_amc_observed_preview_seats",
            "thursday_amc_shadow_ow_prior_usd", "thursday_amc_shadow_ow_prior_source",
            "thursday_preview_actual_usd", "friday_only_amc_observed_seats", "coverage", "snapshot_count",
            "staleness_p50_minutes", "feature_quality_bucket", "amc_candidate_available",
            "amc_candidate_unavailable_reason", "opening_weekend_usd", "production_daily_usd",
        ]
        conn = Mock()
        conn.execute.side_effect = [
            FakeCursor(["exists"], [(True,)]),
            FakeCursor(
                ["column_name"],
                [
                    ("release_run_id",), ("movie_id",), ("title",), ("origin_key",), ("forecast_origin",),
                    ("forecast_origin_utc",), ("as_of_utc",), ("model_version",), ("run_id",), ("sample_key",),
                    ("thursday_amc_seats_collected",), ("predicted_thursday_previews_usd",),
                    ("thursday_preview_actual_usd",), ("friday_only_amc_observed_seats",), ("coverage",),
                    ("snapshot_count",), ("staleness_p50_minutes",), ("feature_quality_bucket",),
                    ("amc_candidate_available",), ("amc_candidate_unavailable_reason",), ("opening_weekend_usd",),
                    ("production_daily_usd",),
                ],
            ),
            FakeCursor(
                legacy_columns,
                [
                    (
                        10, 1, "Example", "FRI_16:00", "16:00", dt.datetime(2026, 7, 10, 20, tzinfo=dt.UTC),
                        dt.datetime(2026, 7, 10, 20, tzinfo=dt.UTC), "boxoffice_test", "run_1", "top_hybrid_30",
                        True, Decimal("8000000"), None, None, None, Decimal("8500000"), Decimal("9000"),
                        Decimal("0.75"), 12, Decimal("4"), "medium", True, None, Decimal("100000000"),
                        Decimal("32000000"),
                    )
                ],
            ),
        ]

        rows = forecasts.amc_shadow_preview_rows(conn, release_run_id=10)

        self.assertEqual(1, len(rows))
        self.assertIsNone(rows[0]["thursday_amc_observed_preview_seats"])
        self.assertIsNone(rows[0]["thursday_amc_shadow_ow_prior_usd"])


class ForecastTemplateTests(unittest.TestCase):
    def test_forecast_chart_styles_do_not_override_bucket_cell_colors(self) -> None:
        stylesheet = (Path(__file__).parents[2] / "src/pm_box_office/web/static/app.css").read_text()

        self.assertNotIn(".forecast-chart rect {", stylesheet)
        self.assertIn(".bucket-probability-label-dark", stylesheet)
        self.assertIn(".bucket-probability-label-light", stylesheet)

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

    def test_detail_template_keeps_the_forecast_view_focused(self) -> None:
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
        forecasts.inject_thursday_preview_timeline_rows(
            timeline,
            [],
            {
                "reported": None,
                "latest_amc_shadow": None,
                "amc_shadow_rows": [
                    {
                        "release_run_id": 10,
                        "movie_id": 1,
                        "title": "Example Movie",
                        "origin_key": "THU_16:00",
                        "thursday_amc_shadow_ow_prior_usd": Decimal("108000000"),
                        "thursday_amc_shadow_ow_prior_source": "thursday_amc_preview_nowcast",
                        "predicted_thursday_previews_usd": Decimal("8000000"),
                        "model_version": "boxoffice_test",
                        "run_id": "run_1",
                    }
                ],
            },
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
                "components": timeline[0]["components"],
            },
            component_breakdown=[
                {
                    "day": "Friday",
                    "window": "Weekend day",
                    "source": "Historical daily pattern",
                    "source_detail": "daily_shape_model",
                    "point_usd": 40_000_000,
                    "share": 0.4,
                    "lo80_usd": 32_000_000,
                    "hi80_usd": 48_000_000,
                    "uncertainty_label": "80% likely range",
                    "is_observed": False,
                }
            ],
            pre_release_breakdown=[
                {
                    "window": "14 days out",
                    "origin_key": "P_-14",
                    "point_usd": 100_000_000,
                    "source_count": 2,
                    "sources": ["Boxoffice Pro", "The Numbers"],
                    "lo80_usd": 80_000_000,
                    "hi80_usd": 120_000_000,
                }
            ],
            timeline=timeline,
            chart=forecasts.forecast_chart(timeline, {"actual_usd": None}),
            model_versions=["boxoffice_test"],
            selected_model_version="boxoffice_test",
            thursday_preview={
                "reported": {
                    "gross_usd": Decimal("8500000"),
                    "preview_date": dt.date(2026, 7, 9),
                    "source": "the_numbers",
                },
                "latest_amc_shadow": {
                    "origin_key": "FRI_16:00",
                    "predicted_thursday_previews_usd": Decimal("8000000"),
                    "thursday_amc_observed_preview_seats": Decimal("1234"),
                    "snapshot_count": 12,
                    "coverage": Decimal("0.75"),
                    "feature_quality_bucket": "medium",
                },
                "amc_shadow_rows": [
                    {
                        "origin_key": "FRI_16:00",
                        "predicted_thursday_previews_usd": Decimal("8000000"),
                        "thursday_preview_actual_usd": Decimal("8500000"),
                        "thursday_amc_shadow_ow_prior_usd": Decimal("108000000"),
                        "coverage": Decimal("0.75"),
                        "snapshot_count": 12,
                        "feature_quality_bucket": "medium",
                        "thursday_amc_seats_collected": True,
                    }
                ],
            },
        )

        self.assertIn("Market buckets unavailable", html)
        self.assertIn('hx-get="/forecasts/10/live?model_version=boxoffice_test&grid=canonical"', html)
        self.assertIn("Forecast timeline", html)
        self.assertIn("Latest forecast breakdown", html)
        self.assertIn('<details class="forecast-data-table" open>', html)
        self.assertIn("<thead><tr><th scope=\"col\">Day / window</th>", html)
        self.assertIn("Source data", html)
        self.assertIn("Resulting estimate", html)
        self.assertIn("80% range", html)
        self.assertIn("Historical daily pattern", html)
        self.assertIn("Pre-release inputs", html)
        self.assertIn("Contributing forecasts", html)
        self.assertIn("Boxoffice Pro · The Numbers", html)
        self.assertNotIn("What drives the latest opening-weekend estimate, day by day.", html)
        self.assertNotIn("Consensus estimates available before the opening weekend.", html)
        self.assertNotIn("Lower clean strike", html)
        self.assertNotIn("Prediction Interval Diagnostics", html)
        self.assertNotIn("Origin Components", html)
        self.assertNotIn("Grid ID", html)
        self.assertNotIn("Anchor forecast", html)
        self.assertNotIn("bucket-probability-table", html)
        self.assertIn("new EventSource", html)

    def test_detail_live_partial_includes_polling_panel(self) -> None:
        html = templates.env.get_template("_forecast_detail_live.html").render(
            request=None,
            release_run_id=10,
            tables_ready=True,
            movie=None,
            latest=None,
            timeline=[],
            chart=forecasts.forecast_chart([], None),
            model_versions=["boxoffice_test"],
            selected_model_version="boxoffice_test",
            thursday_preview=forecasts.empty_thursday_preview_context(),
        )

        self.assertIn('id="forecast-live-panel"', html)
        self.assertIn('hx-trigger="every 10s"', html)
        self.assertIn('hx-get="/forecasts/10/live?model_version=boxoffice_test&grid=canonical"', html)
        self.assertIn("Market buckets unavailable", html)


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
        movie_timeline.assert_called_once_with(conn, release_run_id=123, model_version="boxoffice_test", grid_variant=None)
        self.assertEqual("forecast_detail.html", template_response.call_args.kwargs["name"])
        self.assertEqual(123, template_response.call_args.kwargs["context"]["release_run_id"])
        conn.rollback.assert_called_once()
        conn.close.assert_called_once()

    def test_forecast_detail_live_route_renders_partial_context(self) -> None:
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
            response = forecast_routes.forecast_detail_live(request, 123)

        self.assertEqual("response", response)
        movie_timeline.assert_called_once_with(conn, release_run_id=123, model_version="boxoffice_test", grid_variant=None)
        self.assertEqual("_forecast_detail_live.html", template_response.call_args.kwargs["name"])
        self.assertEqual(123, template_response.call_args.kwargs["context"]["release_run_id"])
        conn.rollback.assert_called_once()
        conn.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
