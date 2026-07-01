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
    season = recreate.season_for_opening_date(opening)
    season_label, season_start = season if season else (None, None)
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
        season_label=season_label,
        season_start=season_start,
        season_delay_weeks=(opening - season_start).days / 7.0 if season_start else None,
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

    def test_2022_plus_episode_filter_uses_opening_date(self) -> None:
        rows = [
            daily_row(10, dt.date(2021, 12, 31), 100),
            daily_row(20, dt.date(2022, 1, 1), 200, movie_id=2, movie_url="movie://two"),
            daily_row(30, dt.date(2026, 7, 1), 300, movie_id=3, movie_url="movie://three"),
        ]
        episodes = recreate.split_release_episodes(rows, gap_threshold_days=14)
        filtered = recreate.filter_release_episodes(
            episodes,
            analysis_start=dt.date(2022, 1, 1),
            analysis_end=dt.date(2026, 6, 30),
        )

        self.assertEqual(["20:1"], [episode.episode_id for episode in filtered])

    def test_season_for_opening_date_assigns_summer_and_holiday_windows(self) -> None:
        self.assertEqual(
            ("summer_2026", dt.date(2026, 5, 25)),
            recreate.season_for_opening_date(dt.date(2026, 6, 1)),
        )
        self.assertEqual(
            ("holiday_2026", dt.date(2026, 11, 1)),
            recreate.season_for_opening_date(dt.date(2026, 12, 20)),
        )
        self.assertEqual(
            ("holiday_2026", dt.date(2026, 11, 1)),
            recreate.season_for_opening_date(dt.date(2027, 1, 3)),
        )
        self.assertIsNone(recreate.season_for_opening_date(dt.date(2026, 10, 1)))

    def test_weekly_aggregation_and_half_life_estimation(self) -> None:
        start = dt.date(2026, 5, 25)
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

    def test_opening_weekend_gross_uses_first_three_calendar_days(self) -> None:
        start = dt.date(2026, 5, 1)
        rows = [
            daily_row(10, start, 100),
            daily_row(10, start + dt.timedelta(days=1), 90),
            daily_row(10, start + dt.timedelta(days=2), 80),
            daily_row(10, start + dt.timedelta(days=3), 70),
        ]
        episode = recreate.split_release_episodes(rows, gap_threshold_days=14)[0]

        self.assertEqual(270, recreate.opening_weekend_gross_usd(episode))

    def test_timing_regression_output_shape(self) -> None:
        start = dt.date(2026, 5, 25)
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
        rows = recreate.regression_rows(estimates, start, dt.date(2026, 6, 15))

        gross_rows = [row for row in rows if row["model"] == "delay_on_opening_gross_millions_and_half_life"]
        self.assertEqual(3, len(gross_rows))
        self.assertEqual(
            {"intercept", "opening_7_day_gross_millions", "half_life_weeks"},
            {row["term"] for row in gross_rows},
        )
        weekend_rows = [
            row
            for row in rows
            if row["model"] == "delay_on_opening_weekend_gross_millions_and_half_life"
        ]
        self.assertEqual(3, len(weekend_rows))
        self.assertEqual(
            {"intercept", "opening_weekend_gross_millions", "half_life_weeks"},
            {row["term"] for row in weekend_rows},
        )

    def test_duopoly_solver_matches_paper_qualitative_examples(self) -> None:
        weak_movie_delays = recreate.solve_duopoly_equilibria(
            recreate.TheoryMovie(0.1, 3.4),
            recreate.TheoryMovie(0.5, 3.4),
        )
        self.assertEqual("movie1_delays", weak_movie_delays[0].classification)
        self.assertGreater(weak_movie_delays[0].movie1_opening_week, 0.0)

        long_legs = recreate.solve_duopoly_equilibria(
            recreate.TheoryMovie(0.5, 4.0),
            recreate.TheoryMovie(0.5, 4.0),
        )
        self.assertEqual("simultaneous_beginning", long_legs[0].classification)
        self.assertEqual(0.0, long_legs[0].movie1_opening_week)
        self.assertEqual(0.0, long_legs[0].movie2_opening_week)

        short_legs = recreate.solve_duopoly_equilibria(
            recreate.TheoryMovie(0.13, 1.0),
            recreate.TheoryMovie(1.0, 1.0),
        )
        self.assertEqual(2, len(short_legs))
        self.assertEqual({"dual_equilibria"}, {row.classification for row in short_legs})

    def test_paper_parameter_table_contains_table1_reported_rows(self) -> None:
        rows = recreate.paper_parameter_equilibrium_rows()
        table1_rows = [
            row
            for row in rows
            if str(row["scenario"]).startswith("table1_group")
        ]

        self.assertEqual(12, len(table1_rows))
        self.assertIn(5.4, {row["paper_reported_opening_week"] for row in table1_rows})


if __name__ == "__main__":
    unittest.main()
