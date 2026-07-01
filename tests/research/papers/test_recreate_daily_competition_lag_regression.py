from __future__ import annotations

import datetime as dt
import unittest

from pm_box_office.research.papers import recreate_daily_competition_lag_regression as recreate


def daily_row(
    movie_id: int,
    day: dt.date,
    gross: int,
    *,
    title: str | None = None,
    theaters: int = 800,
) -> recreate.DailyMovieRow:
    return recreate.DailyMovieRow(
        movie_id=movie_id,
        title=title or f"Movie {movie_id}",
        box_office_date=day,
        gross_usd=gross,
        theaters=theaters,
    )


class DailyCompetitionLagRegressionTests(unittest.TestCase):
    def test_competitor_features_exclude_focal_movie(self) -> None:
        day = dt.date(2022, 1, 4)
        rows = {
            (1, day): daily_row(1, day, 100),
            (2, day): daily_row(2, day, 50),
            (3, day): daily_row(3, day, 25),
        }

        features = recreate.prior_day_competitor_features(
            rows,
            movie_id=1,
            prior_day=day,
            movie_ids={1, 2, 3},
        )

        self.assertEqual(75.0, features["prior_day_competitor_total_gross"])
        self.assertEqual(50.0, features["prior_day_competitor_top1_gross"])
        self.assertEqual(2.0, features["prior_day_competitor_count"])
        self.assertAlmostEqual(75.0 / 175.0, features["prior_day_competitor_market_share_ex_focal"])

    def test_einav_effect_map_computes_amplification_gap(self) -> None:
        effects = recreate.build_einav_effect_map(
            [{"season_week": "01", "observed_log_inside_share_effect": "0.30"}],
            [{"season_week": "01", "estimated_underlying_demand_effect": "0.10"}],
        )

        self.assertAlmostEqual(0.30, effects["01"]["observed_log_inside_share_effect"])
        self.assertAlmostEqual(0.10, effects["01"]["estimated_underlying_demand_effect"])
        self.assertAlmostEqual(0.20, effects["01"]["seasonality_amplification_gap"])

    def test_daily_panel_construction_uses_previous_day_information(self) -> None:
        rows = [
            daily_row(1, dt.date(2022, 1, 3), 100),
            daily_row(1, dt.date(2022, 1, 4), 80),
            daily_row(1, dt.date(2022, 1, 9), 65),
            daily_row(1, dt.date(2022, 1, 10), 60),
            daily_row(2, dt.date(2022, 1, 3), 70),
            daily_row(2, dt.date(2022, 1, 4), 55),
            daily_row(2, dt.date(2022, 1, 9), 40),
            daily_row(2, dt.date(2022, 1, 10), 35),
            daily_row(3, dt.date(2022, 1, 3), 1000, theaters=100),
            daily_row(3, dt.date(2022, 1, 4), 900, theaters=100),
        ]
        panel, features = recreate.build_daily_movie_panel(
            rows,
            start_date=dt.date(2022, 1, 1),
            end_date=None,
            wide_theater_threshold=600,
            max_age_weeks=2,
            sample="wide-first-10-weeks",
            einav_effects={
                "01": {
                    "observed_log_inside_share_effect": 0.2,
                    "estimated_underlying_demand_effect": 0.1,
                    "seasonality_amplification_gap": 0.1,
                },
                "02": {
                    "observed_log_inside_share_effect": 0.3,
                    "estimated_underlying_demand_effect": 0.1,
                    "seasonality_amplification_gap": 0.2,
                },
            },
        )

        self.assertEqual(4, len(panel))
        first_movie_second_day = next(row for row in panel if row["movie_id"] == "1" and row["box_office_date"] == "2022-01-04")
        self.assertEqual(80.0, first_movie_second_day["target_gross_usd"])
        self.assertEqual(100.0, first_movie_second_day["own_prior_day_gross"])
        self.assertEqual(70.0, first_movie_second_day["prior_day_competitor_total_gross"])
        self.assertEqual(0.1, first_movie_second_day["seasonality_amplification_gap"])
        self.assertEqual(len(panel), len(features))

    def test_all_in_theaters_includes_narrow_releases_below_600_theaters(self) -> None:
        rows = [
            daily_row(1, dt.date(2022, 1, 3), 100, theaters=80),
            daily_row(1, dt.date(2022, 1, 4), 90, theaters=75),
            daily_row(2, dt.date(2022, 1, 3), 300, theaters=900),
            daily_row(2, dt.date(2022, 1, 4), 250, theaters=900),
        ]

        panel, _features = recreate.build_daily_movie_panel(
            rows,
            start_date=dt.date(2022, 1, 1),
            end_date=None,
            wide_theater_threshold=600,
            max_age_weeks=10,
            sample="all-in-theaters",
            einav_effects={},
        )

        self.assertIn("1", {row["movie_id"] for row in panel})
        self.assertTrue(all(row["sample"] == "all-in-theaters" for row in panel))

    def test_wide_first_10_weeks_preserves_filtered_behavior(self) -> None:
        rows = [
            daily_row(1, dt.date(2022, 1, 3), 100, theaters=80),
            daily_row(1, dt.date(2022, 1, 4), 90, theaters=75),
            daily_row(2, dt.date(2022, 1, 3), 300, theaters=900),
            daily_row(2, dt.date(2022, 1, 4), 250, theaters=900),
        ]

        panel, _features = recreate.build_daily_movie_panel(
            rows,
            start_date=dt.date(2022, 1, 1),
            end_date=None,
            wide_theater_threshold=600,
            max_age_weeks=10,
            sample="wide-first-10-weeks",
            einav_effects={},
        )

        self.assertEqual({"2"}, {row["movie_id"] for row in panel})
        self.assertEqual([1.0], [row["age_days"] for row in panel])

    def test_all_in_theaters_includes_post_10_week_movie_days(self) -> None:
        start = dt.date(2022, 1, 3)
        rows = [daily_row(1, start + dt.timedelta(days=offset), 100, theaters=50) for offset in (0, 1, 77)]

        panel, _features = recreate.build_daily_movie_panel(
            rows,
            start_date=start,
            end_date=None,
            wide_theater_threshold=600,
            max_age_weeks=10,
            sample="all-in-theaters",
            einav_effects={},
        )

        self.assertIn(77.0, [row["age_days"] for row in panel])

    def test_all_in_theaters_competitors_use_all_active_movies_excluding_focal(self) -> None:
        day = dt.date(2022, 1, 3)
        rows = [
            daily_row(1, day, 100, theaters=100),
            daily_row(1, day + dt.timedelta(days=1), 80, theaters=100),
            daily_row(2, day, 50, theaters=50),
            daily_row(2, day + dt.timedelta(days=1), 40, theaters=50),
            daily_row(3, day, 25, theaters=30),
            daily_row(3, day + dt.timedelta(days=1), 20, theaters=30),
        ]

        panel, _features = recreate.build_daily_movie_panel(
            rows,
            start_date=day,
            end_date=None,
            wide_theater_threshold=600,
            max_age_weeks=10,
            sample="all-in-theaters",
            einav_effects={},
        )
        focal = next(row for row in panel if row["movie_id"] == "1" and row["box_office_date"] == "2022-01-04")

        self.assertEqual(75.0, focal["prior_day_competitor_total_gross"])
        self.assertEqual(2.0, focal["prior_day_competitor_count"])

    def test_theater_share_features_are_bounded(self) -> None:
        rows = [
            daily_row(1, dt.date(2022, 1, 3), 100, theaters=100),
            daily_row(2, dt.date(2022, 1, 3), 50, theaters=50),
        ]

        panel, _features = recreate.build_daily_movie_panel(
            rows,
            start_date=dt.date(2022, 1, 1),
            end_date=None,
            wide_theater_threshold=600,
            max_age_weeks=10,
            sample="all-in-theaters",
            einav_effects={},
        )

        self.assertTrue(all(0.0 <= row["theater_share"] <= 1.0 for row in panel))
        self.assertAlmostEqual(1.0, sum(row["theater_share"] for row in panel))

    def test_chronological_split_prevents_lookahead(self) -> None:
        rows = [
            {"box_office_date": "2022-01-01"},
            {"box_office_date": "2022-01-02"},
            {"box_office_date": "2022-01-03"},
            {"box_office_date": "2022-01-04"},
            {"box_office_date": "2022-01-05"},
        ]

        train, holdout, cutoff = recreate.chronological_split(rows, train_fraction=0.6)

        self.assertEqual("2022-01-03", cutoff)
        self.assertEqual(["2022-01-01", "2022-01-02", "2022-01-03"], [row["box_office_date"] for row in train])
        self.assertEqual(["2022-01-04", "2022-01-05"], [row["box_office_date"] for row in holdout])

    def test_regression_output_shape_for_model_families(self) -> None:
        rows = []
        start = dt.date(2022, 1, 3)
        effects = {
            f"{week:02d}": {
                "observed_log_inside_share_effect": 0.02 * week,
                "estimated_underlying_demand_effect": 0.01 * week,
                "seasonality_amplification_gap": 0.01 * week,
            }
            for week in range(1, 12)
        }
        for movie_id in range(1, 7):
            for offset in range(56):
                day = start + dt.timedelta(days=offset)
                gross = int((1000 + movie_id * 150) * (0.94**offset) * (1.2 if day.weekday() in {4, 5, 6} else 1.0))
                rows.append(daily_row(movie_id, day, max(10, gross), theaters=800 + movie_id))
        panel, _features = recreate.build_daily_movie_panel(
            rows,
            start_date=start,
            end_date=None,
            wide_theater_threshold=600,
            max_age_weeks=8,
            sample="wide-custom",
            einav_effects=effects,
        )
        coefficients_by_model, comparison, hypotheses, predictions = recreate.run_models(panel)

        self.assertEqual(set(recreate.MODEL_TERMS), set(coefficients_by_model))
        self.assertTrue(all(row["holdout_n"] > 0 for row in comparison))
        self.assertEqual(5, len(hypotheses))
        self.assertTrue(predictions)
        self.assertIn("log_prior_day_competitor_total_gross", {row["term"] for row in coefficients_by_model["competition_model"]})

    def test_share_target_enrichment_uses_daily_industry_total(self) -> None:
        panel = [
            {
                "movie_id": "1",
                "box_office_date": "2022-01-04",
                "target_gross_usd": 80.0,
                "own_prior_day_gross": 100.0,
                "prior_day_competitor_total_gross": 50.0,
            },
            {
                "movie_id": "2",
                "box_office_date": "2022-01-04",
                "target_gross_usd": 20.0,
                "own_prior_day_gross": 50.0,
                "prior_day_competitor_total_gross": 100.0,
            },
        ]

        enriched = recreate.enrich_share_targets(panel)

        self.assertAlmostEqual(0.8, enriched[0]["daily_market_share"])
        self.assertAlmostEqual(100.0 / 150.0, enriched[0]["own_prior_day_market_share"])
        self.assertIn("logit_daily_share", enriched[0])
        self.assertIn("logit_own_prior_day_share", enriched[0])

    def test_rolling_backtest_metrics_include_all_test_model_families(self) -> None:
        rows = []
        start = dt.date(2022, 1, 3)
        effects = {
            f"{week:02d}": {
                "observed_log_inside_share_effect": 0.02 * week,
                "estimated_underlying_demand_effect": 0.01 * week,
                "seasonality_amplification_gap": 0.01 * week,
            }
            for week in range(1, 54)
        }
        for year in range(2022, 2025):
            year_start = dt.date(year, 1, 3)
            for movie_id in range(1, 5):
                for offset in range(28):
                    day = year_start + dt.timedelta(days=offset)
                    gross = int((1000 + movie_id * 100 + (year - 2022) * 50) * (0.95**offset))
                    rows.append(daily_row(movie_id + year * 100, day, max(10, gross), theaters=900))
        panel, _features = recreate.build_daily_movie_panel(
            rows,
            start_date=start,
            end_date=None,
            wide_theater_threshold=600,
            max_age_weeks=4,
            sample="wide-custom",
            einav_effects=effects,
        )

        metrics, _predictions = recreate.rolling_backtest_metrics(panel)

        test_2024 = [row for row in metrics if row["test_year"] == 2024]
        self.assertEqual(set(recreate.TEST_MODEL_TERMS), {row["model"] for row in test_2024})
        self.assertTrue(all(row["holdout_n"] > 0 for row in test_2024))

    def test_validation_rows_pass_for_synthetic_panel(self) -> None:
        rows = [
            daily_row(1, dt.date(2022, 1, 3), 100),
            daily_row(1, dt.date(2022, 1, 4), 80),
            daily_row(2, dt.date(2022, 1, 3), 70),
            daily_row(2, dt.date(2022, 1, 4), 55),
        ]
        panel, _features = recreate.build_daily_movie_panel(
            rows,
            start_date=dt.date(2022, 1, 1),
            end_date=None,
            wide_theater_threshold=600,
            max_age_weeks=2,
            sample="wide-first-10-weeks",
            einav_effects={},
        )

        checks = recreate.validation_check_rows(
            rows,
            panel,
            start_date=dt.date(2022, 1, 1),
            wide_theater_threshold=600,
            max_age_weeks=2,
            sample="wide-first-10-weeks",
        )

        self.assertTrue(all(row["passed"] for row in checks))

    def test_placebo_transform_changes_competitor_features_without_changing_targets(self) -> None:
        rows = []
        start = dt.date(2022, 1, 3)
        for movie_id in range(1, 4):
            for offset in range(5):
                rows.append(daily_row(movie_id, start + dt.timedelta(days=offset), 100 + movie_id * 10 + offset, theaters=900))
        panel, _features = recreate.build_daily_movie_panel(
            rows,
            start_date=start,
            end_date=None,
            wide_theater_threshold=600,
            max_age_weeks=2,
            sample="wide-first-10-weeks",
            einav_effects={},
        )
        enriched = recreate.enrich_share_targets(panel)

        shuffled = recreate.transformed_competitor_rows(enriched, "shuffled_across_dates")

        self.assertEqual([row["target_gross_usd"] for row in enriched], [row["target_gross_usd"] for row in shuffled])
        self.assertNotEqual(
            [row["prior_day_competitor_total_gross"] for row in enriched],
            [row["prior_day_competitor_total_gross"] for row in shuffled],
        )

    def test_focused_2026_split_uses_pre_2026_train_and_2026_test(self) -> None:
        rows = [
            {"box_office_date": "2025-12-31"},
            {"box_office_date": "2026-01-01"},
            {"box_office_date": "2026-06-30"},
            {"box_office_date": "2027-01-01"},
        ]

        train, test = recreate.focused_2026_split(rows)

        self.assertEqual(["2025-12-31"], [row["box_office_date"] for row in train])
        self.assertEqual(["2026-01-01", "2026-06-30"], [row["box_office_date"] for row in test])

    def test_focused_2026_split_works_for_both_sample_modes(self) -> None:
        rows = [
            {"box_office_date": "2025-12-31", "sample": "all-in-theaters"},
            {"box_office_date": "2026-01-01", "sample": "all-in-theaters"},
            {"box_office_date": "2025-12-31", "sample": "wide-first-10-weeks"},
            {"box_office_date": "2026-01-01", "sample": "wide-first-10-weeks"},
        ]

        for sample in ("all-in-theaters", "wide-first-10-weeks"):
            train, test = recreate.focused_2026_split([row for row in rows if row["sample"] == sample])
            self.assertEqual(1, len(train))
            self.assertEqual(1, len(test))

    def test_focused_univariate_model_uses_only_competitor_share(self) -> None:
        self.assertEqual(
            ["prior_day_competitor_market_share_ex_focal"],
            recreate.FOCUSED_2026_MODELS["univariate_competitor_share"],
        )

    def test_focused_predicted_logits_convert_to_bounded_shares(self) -> None:
        self.assertGreater(recreate.logistic(-1000.0), 0.0)
        self.assertLess(recreate.logistic(1000.0), 1.0)
        self.assertAlmostEqual(0.5, recreate.logistic(0.0))

    def test_focused_prediction_rows_include_actual_predicted_and_implied_gross(self) -> None:
        test_rows = [
            {
                "movie_id": "1",
                "title": "One",
                "box_office_date": "2026-01-02",
                "logit_daily_share": recreate.logit(0.25),
                "daily_market_share": 0.25,
                "daily_industry_gross_usd": 1000.0,
                "target_gross_usd": 250.0,
                "prior_day_competitor_market_share_ex_focal": 0.6,
                "own_prior_day_market_share": 0.3,
            }
        ]

        rows = recreate.focused_2026_prediction_rows(
            "incremental_competitor_share",
            [0.0, 1.0, -0.5],
            ["logit_daily_share", "prior_day_competitor_market_share_ex_focal"],
            test_rows,
        )

        self.assertEqual(1, len(rows))
        self.assertIn("actual_daily_share", rows[0])
        self.assertIn("predicted_daily_share", rows[0])
        self.assertIn("share_residual", rows[0])
        self.assertIn("implied_predicted_gross_usd", rows[0])
        self.assertGreater(rows[0]["predicted_daily_share"], 0.0)
        self.assertLess(rows[0]["predicted_daily_share"], 1.0)

    def test_focused_movie_level_summary_aggregates_predictions(self) -> None:
        predictions = [
            {
                "model": "incremental_competitor_share",
                "movie_id": "1",
                "title": "One",
                "box_office_date": "2026-01-02",
                "actual_daily_share": 0.2,
                "predicted_daily_share": 0.25,
                "actual_gross_usd": 200.0,
                "implied_predicted_gross_usd": 250.0,
                "logit_residual": -0.1,
                "share_residual": -0.05,
                "gross_residual_usd": -50.0,
            },
            {
                "model": "incremental_competitor_share",
                "movie_id": "1",
                "title": "One",
                "box_office_date": "2026-01-03",
                "actual_daily_share": 0.4,
                "predicted_daily_share": 0.35,
                "actual_gross_usd": 400.0,
                "implied_predicted_gross_usd": 350.0,
                "logit_residual": 0.2,
                "share_residual": 0.05,
                "gross_residual_usd": 50.0,
            },
        ]

        summary = recreate.focused_2026_movie_level_summary(predictions)

        self.assertEqual(1, len(summary))
        self.assertEqual(2, summary[0]["row_count"])
        self.assertEqual(600.0, summary[0]["total_actual_gross_usd"])
        self.assertEqual(600.0, summary[0]["total_implied_predicted_gross_usd"])
        self.assertAlmostEqual(0.3, summary[0]["observed_average_share"])

    def test_release_age_bucket_mapping(self) -> None:
        self.assertEqual("opening_day", recreate.release_age_bucket(0))
        self.assertEqual("days_1_3", recreate.release_age_bucket(3))
        self.assertEqual("days_4_7", recreate.release_age_bucket(7))
        self.assertEqual("week_2", recreate.release_age_bucket(13))
        self.assertEqual("weeks_3_4", recreate.release_age_bucket(27))
        self.assertEqual("weeks_5_8", recreate.release_age_bucket(55))
        self.assertEqual("weeks_9_plus", recreate.release_age_bucket(56))

    def test_competitor_pressure_quartiles_are_assigned(self) -> None:
        rows = [
            {"prior_day_competitor_market_share_ex_focal": value}
            for value in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
        ]

        assigned = recreate.assign_competitor_pressure_quartiles(rows)

        self.assertEqual({1, 2, 3, 4}, {row["competitor_pressure_quartile"] for row in assigned})
        self.assertEqual("Q1", assigned[0]["competitor_pressure_quartile_label"])
        self.assertEqual("Q4", assigned[-1]["competitor_pressure_quartile_label"])

    def test_bucketed_competition_effect_rows_include_expected_fields(self) -> None:
        rows = []
        for index in range(12):
            day = dt.date(2025, 1, 2) + dt.timedelta(days=index)
            rows.extend(
                [
                    {
                        "movie_id": "1",
                        "title": "One",
                        "box_office_date": day.isoformat(),
                        "target_gross_usd": 150.0 + index * 3,
                        "own_prior_day_gross": 140.0 + index,
                        "prior_day_competitor_total_gross": 100.0 + index * 2,
                        "prior_day_competitor_market_share_ex_focal": 0.35 + index * 0.01,
                        "age_days": float(1 + index % 3),
                    },
                    {
                        "movie_id": "2",
                        "title": "Two",
                        "box_office_date": day.isoformat(),
                        "target_gross_usd": 95.0 + index * 2,
                        "own_prior_day_gross": 90.0 + index * 1.5,
                        "prior_day_competitor_total_gross": 150.0 + index * 3,
                        "prior_day_competitor_market_share_ex_focal": 0.55 - index * 0.01,
                        "age_days": float(1 + index % 3),
                    },
                ]
            )
        for index in range(3):
            day = dt.date(2026, 1, 2) + dt.timedelta(days=index)
            rows.extend(
                [
                    {
                        "movie_id": "1",
                        "title": "One",
                        "box_office_date": day.isoformat(),
                        "target_gross_usd": 180.0 + index,
                        "own_prior_day_gross": 170.0 + index,
                        "prior_day_competitor_total_gross": 120.0 + index,
                        "prior_day_competitor_market_share_ex_focal": 0.40 + index * 0.01,
                        "age_days": float(1 + index),
                    },
                    {
                        "movie_id": "2",
                        "title": "Two",
                        "box_office_date": day.isoformat(),
                        "target_gross_usd": 110.0 + index,
                        "own_prior_day_gross": 100.0 + index,
                        "prior_day_competitor_total_gross": 180.0 + index,
                        "prior_day_competitor_market_share_ex_focal": 0.52 - index * 0.01,
                        "age_days": float(1 + index),
                    },
                ]
            )

        bucket_rows = recreate.bucketed_competition_effect_rows(rows)
        quartile_rows = recreate.competition_quartile_response_rows(rows)
        exact_rows = recreate.exact_release_day_competition_effect_rows(rows, min_train_n=4)

        self.assertEqual(["days_1_3"], [row["release_age_bucket"] for row in bucket_rows])
        self.assertIn("competitor_share_coef", bucket_rows[0])
        self.assertTrue(quartile_rows)
        self.assertIn("mean_actual_share", quartile_rows[0])
        self.assertTrue(exact_rows)
        self.assertIn("release_age_day", exact_rows[0])


if __name__ == "__main__":
    unittest.main()
