from __future__ import annotations

import datetime as dt
import unittest

from pm_box_office.orchestration import runner


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
            "boxofficeguru",
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

    def test_non_estimate_manual_run_all_sources_do_not_receive_rolling_week_dates(self) -> None:
        self.assertEqual([], runner.autorun_extra_args("wikipedia", trigger="manual_run_all", today=dt.date(2026, 7, 3)))


if __name__ == "__main__":
    unittest.main()
