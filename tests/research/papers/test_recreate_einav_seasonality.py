from __future__ import annotations

import datetime as dt
import unittest

from pm_box_office.research.papers import recreate_einav_seasonality as recreate


def daily_row(
    movie_id: int,
    day: dt.date,
    gross: int,
    *,
    title: str | None = None,
    theaters: int = 800,
) -> recreate.DatabaseDailyRow:
    return recreate.DatabaseDailyRow(
        movie_id=movie_id,
        title=title or f"Movie {movie_id}",
        box_office_date=day,
        gross_usd=gross,
        theaters=theaters,
    )


class EinavSeasonalityTests(unittest.TestCase):
    def test_weekly_panel_filters_to_2022_plus_wide_movies_and_max_age(self) -> None:
        rows = [
            daily_row(1, dt.date(2022, 1, 3), 100, theaters=700),
            daily_row(1, dt.date(2022, 1, 4), 50, theaters=650),
            daily_row(1, dt.date(2022, 1, 10), 25, theaters=620),
            daily_row(1, dt.date(2022, 1, 17), 10, theaters=620),
            daily_row(2, dt.date(2022, 1, 3), 100, theaters=100),
            daily_row(3, dt.date(2021, 12, 27), 100, theaters=900),
        ]
        weekly = recreate.aggregate_daily_rows_to_movie_weeks(
            rows,
            start_date=dt.date(2022, 1, 1),
            end_date=None,
            wide_theater_threshold=600,
            max_age_weeks=2,
        )

        self.assertEqual(["1", "1"], [row["movie_id"] for row in weekly])
        self.assertEqual([0.0, 1.0], [row["age"] for row in weekly])
        self.assertEqual(150.0, weekly[0]["weekly_gross_usd"])
        self.assertEqual("01", weekly[0]["calendar_week"])

    def test_market_share_normalization_keeps_inside_share_bounded(self) -> None:
        weekly = [
            {
                "movie_id": "1",
                "title": "One",
                "week_start": "2022-01-03",
                "period_id": "2022-01-03",
                "calendar_week": "01",
                "age": 0.0,
                "weekly_gross_usd": 80.0,
                "max_theaters": 800,
                "observed_days": 7,
                "is_wide": 1,
            },
            {
                "movie_id": "2",
                "title": "Two",
                "week_start": "2022-01-03",
                "period_id": "2022-01-03",
                "calendar_week": "01",
                "age": 0.0,
                "weekly_gross_usd": 20.0,
                "max_theaters": 800,
                "observed_days": 7,
                "is_wide": 1,
            },
        ]
        panel = recreate.normalize_movie_week_market_shares(weekly, market_size_peak_share=0.08)

        self.assertEqual(2, len(panel))
        self.assertAlmostEqual(0.08, panel[0]["inside_share"])
        self.assertAlmostEqual(0.08, panel[1]["inside_share"])
        self.assertAlmostEqual(0.08, sum(row["market_share"] for row in panel))
        self.assertTrue(all(0.0 < row["market_share"] < 1.0 for row in panel))
        self.assertTrue(all(0.0 < row["inside_share"] < 1.0 for row in panel))
        self.assertIn("dependent_log_share", panel[0])
        self.assertIn("log_within_industry_share", panel[0])

    def test_estimate_panel_output_shape(self) -> None:
        raw_rows = []
        for year in range(2022, 2026):
            for movie_index in range(4):
                movie_id = f"{year}-{movie_index}"
                for age in range(4):
                    week = age + 1
                    period_id = f"{year}-w{week:02d}"
                    base_share = 0.0015 + 0.0005 * movie_index + 0.0002 * (year - 2022)
                    raw_rows.append(
                        {
                            "movie_id": movie_id,
                            "calendar_week": f"{week:02d}",
                            "period_id": period_id,
                            "age": str(age),
                            "market_share": str(base_share / (1.4**age)),
                        }
                    )
            if year % 2 == 0:
                for age in range(2):
                    raw_rows.append(
                        {
                            "movie_id": f"{year}-extra",
                            "calendar_week": f"{age + 1:02d}",
                            "period_id": f"{year}-w{age + 1:02d}",
                            "age": str(age),
                            "market_share": str(0.0008 / (1.3**age)),
                        }
                    )
        panel = recreate.prepare_panel(raw_rows)
        estimates, week_rows = recreate.estimate_panel(panel)

        self.assertEqual(
            {"decay_beta", "nested_logit_sigma", "movies_in_release_instrument"},
            {row["parameter"] for row in estimates},
        )
        self.assertEqual({"01", "02", "03", "04"}, {row["season_week"] for row in week_rows})

    def test_paper_vs_database_summary_includes_reported_and_local_values(self) -> None:
        panel = [
            {"movie_id": "1"},
            {"movie_id": "2"},
            {"movie_id": "2"},
        ]
        estimates = [
            {"model": "two_way_fe_2sls", "parameter": "decay_beta", "estimate": -0.3},
            {"model": "two_way_fe_2sls", "parameter": "nested_logit_sigma", "estimate": 0.4},
        ]
        decomposition = [
            {"metric": "estimated_week_effect_sd_with_movie_fe", "database_2022_plus": 0.2},
            {"metric": "observed_log_inside_share_effect_sd_without_movie_fe", "database_2022_plus": 0.3},
            {"metric": "quality_endogeneity_amplification_ratio", "database_2022_plus": 1.5},
        ]
        rows = recreate.paper_vs_database_summary_rows(panel, estimates, decomposition)

        by_result = {row["result"]: row for row in rows}
        self.assertEqual(16103, by_result["sample_movie_weeks"]["paper_reported"])
        self.assertEqual(3, by_result["sample_movie_weeks"]["database_2022_plus"])
        self.assertEqual(-0.220, by_result["decay_beta_with_movie_fe"]["paper_reported"])
        self.assertEqual(-0.3, by_result["decay_beta_with_movie_fe"]["database_2022_plus"])


if __name__ == "__main__":
    unittest.main()
