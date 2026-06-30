#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import unittest

from pm_box_office.research.papers import recreate_competitive_dynamics as recreate


def daily_row(
    release_run_id: int,
    day: dt.date,
    gross: int,
    *,
    movie_id: int = 1,
    movie_url: str = "movie://fixture",
    title: str = "Fixture Movie",
    theaters: int = 1000,
    cumulative: int | None = None,
) -> recreate.DailyRow:
    return recreate.DailyRow(
        release_run_id=release_run_id,
        movie_id=movie_id,
        movie_url=movie_url,
        title=title,
        release_year=2026,
        box_office_date=day,
        gross_usd=gross,
        theaters=theaters,
        cumulative_gross_usd=cumulative if cumulative is not None else gross,
        rank="1",
    )


def estimate_fixture(index: int, opening: dt.date, gross: int, half_life: float) -> recreate.LifecycleEstimate:
    rows = [
        daily_row(
            index,
            opening + dt.timedelta(days=week * 7),
            max(1, int(gross / (2 ** week))),
            movie_id=index,
            movie_url=f"movie://{index}",
            title=f"Movie {index}",
            theaters=1200,
        )
        for week in range(4)
    ]
    episode = recreate.split_release_episodes(rows, gap_threshold_days=14)[0]
    return recreate.LifecycleEstimate(
        episode=episode,
        positive_weeks=4,
        opening_7_day_gross_usd=gross,
        decay_beta=1.0 / half_life,
        half_life_weeks=half_life,
        lifecycle_r2=1.0,
        opening_share=0.2 + index * 0.02,
        opening_attraction=0.25,
        chart_coverage_days=7,
        delay_weeks=(opening - dt.date(2026, 5, 1)).days / 7.0,
        is_wide=True,
    )


class CompetitiveDynamicsTests(unittest.TestCase):
    def test_episode_splitting_on_large_date_gap(self) -> None:
        rows = [
            daily_row(10, dt.date(2026, 5, 1), 100),
            daily_row(10, dt.date(2026, 5, 2), 90),
            daily_row(10, dt.date(2026, 6, 1), 80),
        ]
        episodes = recreate.split_release_episodes(rows, gap_threshold_days=14)

        self.assertEqual(2, len(episodes))
        self.assertEqual("10:1", episodes[0].episode_id)
        self.assertEqual("10:2", episodes[1].episode_id)

    def test_weekly_aggregation_and_half_life_estimation(self) -> None:
        start = dt.date(2026, 5, 1)
        rows = [
            daily_row(10, start + dt.timedelta(days=week * 7), gross)
            for week, gross in enumerate([1000, 500, 250, 125])
        ]
        episode = recreate.split_release_episodes(rows, gap_threshold_days=14)[0]
        weekly = recreate.aggregate_weekly(episode)
        positive, opening, beta, half_life, r2 = recreate.estimate_lifecycle_from_weekly(
            weekly,
            min_positive_weeks=4,
        )

        self.assertEqual(4, positive)
        self.assertEqual(1000, opening)
        self.assertIsNotNone(beta)
        self.assertAlmostEqual(1.0, half_life or 0.0, places=6)
        self.assertAlmostEqual(1.0, r2 or 0.0, places=6)

    def test_opening_share_uses_chart_denominator_coverage(self) -> None:
        start = dt.date(2026, 5, 1)
        rows = [daily_row(10, start + dt.timedelta(days=offset), 100) for offset in range(3)]
        episode = recreate.split_release_episodes(rows, gap_threshold_days=14)[0]
        chart_totals = {start + dt.timedelta(days=offset): 400 for offset in range(3)}
        chart_movie_gross = {
            ("movie://fixture", start + dt.timedelta(days=offset)): 100
            for offset in range(3)
        }
        share, attraction, coverage = recreate.opening_share_for_episode(
            episode,
            chart_totals,
            chart_movie_gross,
        )

        self.assertEqual(3, coverage)
        self.assertAlmostEqual(0.25, share or 0.0)
        self.assertAlmostEqual(1.0 / 3.0, attraction or 0.0)

    def test_timing_regression_output_shape(self) -> None:
        start = dt.date(2026, 5, 1)
        estimates = [
            estimate_fixture(index, start + dt.timedelta(days=index), gross, half_life)
            for index, (gross, half_life) in enumerate(
                [
                    (8000, 1.0),
                    (6000, 1.7),
                    (4000, 1.7),
                    (3000, 1.0),
                    (2000, 1.7),
                ],
                start=1,
            )
        ]
        rows = recreate.regression_rows(estimates, start, dt.date(2026, 5, 31))

        gross_rows = [row for row in rows if row["model"] == "delay_on_opening_gross_millions_and_half_life"]
        self.assertEqual(3, len(gross_rows))
        self.assertEqual(
            {"intercept", "opening_7_day_gross_millions", "half_life_weeks"},
            {row["term"] for row in gross_rows},
        )


if __name__ == "__main__":
    unittest.main()
