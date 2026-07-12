from __future__ import annotations

import datetime as dt
import unittest

from pm_box_office.orchestration import runner
from pm_box_office.orchestration.registry import DAILY_BACKGROUND_SOURCE_KEYS, source_poll_due


class OrchestrationRunnerTests(unittest.TestCase):
    def test_auto_daily_the_numbers_lags_today_and_refreshes_last_two_actual_days(self) -> None:
        args = runner.autorun_extra_args("the_numbers", trigger="auto_daily", today=dt.date(2026, 7, 3))

        self.assertEqual(
            [
                "--start-date",
                "2026-06-26",
                "--end-date",
                "2026-07-02",
                "--refresh-recent-days",
                "2",
            ],
            args,
        )

    def test_manual_the_numbers_run_does_not_receive_autorun_dates(self) -> None:
        self.assertEqual([], runner.autorun_extra_args("the_numbers", trigger="manual_run_all"))

    def test_run_all_estimate_sources_receive_rolling_week_dates(self) -> None:
        for source_key in (
            "boxofficepro",
            "boxofficereport",
            "boxofficetheory",
            "boxofficetheory_substack",
            "edwarddouglas_substack",
            "boxofficeguru",
            "toddmthatcher",
            "joblo",
        ):
            with self.subTest(source_key=source_key):
                self.assertEqual(
                    [
                        "--start-date",
                        "2026-06-27",
                        "--end-date",
                        "2026-07-03",
                        "--refresh",
                    ],
                    runner.autorun_extra_args(
                        source_key,
                        trigger="manual_run_all",
                        today=dt.date(2026, 7, 3),
                    ),
                )

    def test_manual_substack_ui_run_receives_rolling_week_dates(self) -> None:
        self.assertEqual(
            [
                "--start-date",
                "2026-06-27",
                "--end-date",
                "2026-07-03",
                "--refresh",
            ],
            runner.autorun_extra_args(
                "boxofficetheory_substack",
                trigger="manual",
                today=dt.date(2026, 7, 3),
            ),
        )

    def test_manual_edward_douglas_substack_ui_run_receives_rolling_week_dates(self) -> None:
        self.assertEqual(
            [
                "--start-date",
                "2026-06-27",
                "--end-date",
                "2026-07-03",
                "--refresh",
            ],
            runner.autorun_extra_args(
                "edwarddouglas_substack",
                trigger="manual",
                today=dt.date(2026, 7, 3),
            ),
        )

    def test_non_estimate_manual_run_all_sources_do_not_receive_rolling_week_dates(self) -> None:
        self.assertEqual([], runner.autorun_extra_args("wikipedia", trigger="manual_run_all", today=dt.date(2026, 7, 3)))

    def test_boxofficepro_polling_window_tracks_wednesday_publications_in_new_york(self) -> None:
        self.assertTrue(source_poll_due("boxofficepro", dt.datetime(2026, 7, 8, 19, 30, tzinfo=dt.UTC)))
        self.assertFalse(source_poll_due("boxofficepro", dt.datetime(2026, 7, 8, 23, 30, tzinfo=dt.UTC)))

    def test_boxofficetheory_polling_windows_cover_variable_friday_posts(self) -> None:
        self.assertTrue(source_poll_due("boxofficetheory", dt.datetime(2026, 7, 3, 18, 0, tzinfo=dt.UTC)))
        self.assertFalse(source_poll_due("boxofficetheory", dt.datetime(2026, 7, 5, 18, 0, tzinfo=dt.UTC)))

    def test_edward_douglas_polling_windows_cover_weekend_warrior_posts(self) -> None:
        self.assertTrue(source_poll_due("edwarddouglas_substack", dt.datetime(2026, 7, 8, 17, 0, tzinfo=dt.UTC)))
        self.assertFalse(source_poll_due("edwarddouglas_substack", dt.datetime(2026, 7, 5, 17, 0, tzinfo=dt.UTC)))

    def test_boxofficeguru_polling_windows_cover_archive_page_appearance(self) -> None:
        self.assertTrue(source_poll_due("boxofficeguru", dt.datetime(2026, 7, 10, 16, 0, tzinfo=dt.UTC)))
        self.assertTrue(source_poll_due("boxofficeguru", dt.datetime(2026, 7, 12, 16, 0, tzinfo=dt.UTC)))
        self.assertFalse(source_poll_due("boxofficeguru", dt.datetime(2026, 7, 11, 16, 0, tzinfo=dt.UTC)))

    def test_toddmthatcher_polling_windows_cover_roundup_and_single_movie_posts(self) -> None:
        self.assertTrue(source_poll_due("toddmthatcher", dt.datetime(2026, 7, 8, 2, 30, tzinfo=dt.UTC)))
        self.assertTrue(source_poll_due("toddmthatcher", dt.datetime(2026, 7, 10, 14, 0, tzinfo=dt.UTC)))
        self.assertFalse(source_poll_due("toddmthatcher", dt.datetime(2026, 7, 10, 3, 30, tzinfo=dt.UTC)))

    def test_joblo_polling_windows_cover_predictions_and_sunday_results(self) -> None:
        self.assertTrue(source_poll_due("joblo", dt.datetime(2026, 7, 9, 16, 0, tzinfo=dt.UTC)))
        self.assertTrue(source_poll_due("joblo", dt.datetime(2026, 7, 12, 16, 0, tzinfo=dt.UTC)))
        self.assertFalse(source_poll_due("joblo", dt.datetime(2026, 7, 11, 16, 0, tzinfo=dt.UTC)))

    def test_the_numbers_prediction_polling_covers_friday_prediction_articles(self) -> None:
        self.assertTrue(source_poll_due("the_numbers_predictions", dt.datetime(2026, 7, 3, 17, 0, tzinfo=dt.UTC)))
        self.assertFalse(source_poll_due("the_numbers_predictions", dt.datetime(2026, 7, 4, 17, 0, tzinfo=dt.UTC)))

    def test_the_numbers_daily_charts_poll_throughout_reporting_day(self) -> None:
        self.assertTrue(source_poll_due("the_numbers", dt.datetime(2026, 7, 6, 18, 0, tzinfo=dt.UTC)))
        self.assertFalse(source_poll_due("the_numbers", dt.datetime(2026, 7, 6, 5, 0, tzinfo=dt.UTC)))

    def test_polymarket_metadata_polls_continuously_at_fifteen_minute_intervals(self) -> None:
        self.assertTrue(source_poll_due("polymarket_metadata", dt.datetime(2026, 7, 6, 18, 0, tzinfo=dt.UTC)))
        self.assertTrue(source_poll_due("polymarket_metadata", dt.datetime(2026, 7, 6, 18, 15, tzinfo=dt.UTC)))
        self.assertFalse(source_poll_due("polymarket_metadata", dt.datetime(2026, 7, 6, 18, 7, tzinfo=dt.UTC)))

    def test_the_numbers_prediction_poll_refreshes_homepage(self) -> None:
        self.assertEqual(["--refresh"], runner.autorun_extra_args("the_numbers_predictions", trigger="auto_source_poll"))

    def test_the_numbers_source_poll_refreshes_only_the_two_recent_chart_dates(self) -> None:
        self.assertEqual(
            [
                "--start-date",
                "2026-07-01",
                "--end-date",
                "2026-07-02",
                "--refresh-recent-days",
                "2",
            ],
            runner.autorun_extra_args("the_numbers", trigger="auto_source_poll", today=dt.date(2026, 7, 3)),
        )

    def test_polled_sources_are_excluded_from_the_nightly_background_run(self) -> None:
        self.assertNotIn("the_numbers", DAILY_BACKGROUND_SOURCE_KEYS)
        self.assertNotIn("the_numbers_predictions", DAILY_BACKGROUND_SOURCE_KEYS)
        self.assertNotIn("boxofficeguru", DAILY_BACKGROUND_SOURCE_KEYS)
        self.assertNotIn("toddmthatcher", DAILY_BACKGROUND_SOURCE_KEYS)
        self.assertNotIn("joblo", DAILY_BACKGROUND_SOURCE_KEYS)


if __name__ == "__main__":
    unittest.main()
