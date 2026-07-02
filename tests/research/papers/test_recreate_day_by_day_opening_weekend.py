from __future__ import annotations

import datetime as dt
import math
import tempfile
import unittest
from pathlib import Path

from pm_box_office.research.papers import recreate_competition_opening_weekend as opening
from pm_box_office.research.papers import recreate_day_by_day_opening_weekend as recreate


def movie(
    movie_id: int,
    opening_date: dt.date,
    *,
    release_year: int | None = None,
    opening_day_gross: int = 5_000_000,
    weekend_gross: int = 15_000_000,
    theaters: int = 3000,
) -> opening.OpeningWeekendMovie:
    return opening.OpeningWeekendMovie(
        movie_id=movie_id,
        title=f"Movie {movie_id}",
        release_year=release_year or opening_date.year,
        release_run_id=movie_id * 10,
        opening_date=opening_date,
        opening_theaters=theaters,
        opening_day_gross_usd=opening_day_gross,
        opening_weekend_revenue_usd=weekend_gross,
    )


def daily(movie_id: int, day: dt.date, gross: int) -> opening.DailyGross:
    return opening.DailyGross(movie_id=movie_id, box_office_date=day, gross_usd=gross, theaters=1000)


def bop_forecast(
    movie_id: int,
    *,
    prediction_id: int = 1,
    article_id: int = 1,
    published_date: dt.date = dt.date(2026, 4, 24),
    target_start_date: dt.date = dt.date(2026, 5, 1),
    low: int = 10_000_000,
    high: int = 20_000_000,
) -> opening.BoxofficeProForecast:
    return opening.BoxofficeProForecast(
        prediction_id=prediction_id,
        movie_id=movie_id,
        article_id=article_id,
        article_url=f"https://www.boxofficepro.com/{article_id}/",
        source_movie_title=f"Movie {movie_id}",
        forecast_metric="domestic_opening_weekend",
        source_context="test",
        source_rank=1,
        target_start_date=target_start_date,
        target_end_date=target_start_date + dt.timedelta(days=2),
        range_low_usd=float(low),
        range_high_usd=float(high),
        showtime_market_share_pct=25.0,
        published_date=published_date,
    )


class DayByDayOpeningWeekendTests(unittest.TestCase):
    def test_default_snapshot_range_targets_opening_weekend_through_day_two(self) -> None:
        self.assertEqual(list(range(-14, 4)), list(recreate.DEFAULT_SNAPSHOT_DAYS_BY_TARGET["3_day"]))
        self.assertEqual(list(range(-14, 6)), list(recreate.DEFAULT_SNAPSHOT_DAYS_BY_TARGET["5_day"]))
        self.assertEqual(-14, min(recreate.DEFAULT_SNAPSHOT_DAYS))
        self.assertEqual(5, max(recreate.DEFAULT_SNAPSHOT_DAYS))
        self.assertEqual(list(range(-14, 6)), list(recreate.DEFAULT_SNAPSHOT_DAYS))
        self.assertEqual((-1, 1, 2), recreate.COMPETITIVE_CHECKPOINT_DAYS)

    def test_parse_cutoff_list_sorts_deduplicates_and_rejects_negative_values(self) -> None:
        self.assertEqual([0, 1_000_000, 5_000_000], recreate.parse_cutoff_list("5000000,0,1000000,0"))
        with self.assertRaises(ValueError):
            recreate.parse_cutoff_list("")
        with self.assertRaises(ValueError):
            recreate.parse_cutoff_list("0,-1")

    def test_train_holdout_rows_use_separate_train_and_test_opening_day_cutoffs(self) -> None:
        rows = [
            {"movie_id": 1, "snapshot_day": -1, "release_year": 2022, "opening_day_gross_usd": 500_000},
            {"movie_id": 2, "snapshot_day": -1, "release_year": 2022, "opening_day_gross_usd": 2_000_000},
            {"movie_id": 3, "snapshot_day": -1, "release_year": 2025, "opening_day_gross_usd": 2_000_000},
            {"movie_id": 4, "snapshot_day": -1, "release_year": 2025, "opening_day_gross_usd": 6_000_000},
            {"movie_id": 5, "snapshot_day": -2, "release_year": 2025, "opening_day_gross_usd": 10_000_000},
        ]

        train_rows, holdout_rows = recreate.train_holdout_rows(
            rows,
            snapshot_day=-1,
            train_start_year=2022,
            train_end_year=2024,
            test_start_year=2025,
            test_end_year=2026,
            train_min_opening_day_gross=1_000_000,
            test_min_opening_day_gross=5_000_000,
        )

        self.assertEqual([2], [row["movie_id"] for row in train_rows])
        self.assertEqual([4], [row["movie_id"] for row in holdout_rows])

    def test_snapshot_residual_models_use_views_only_for_wikipedia_signal(self) -> None:
        for model_name in (
            "bop_residual_wiki_snapshot",
            "bop_residual_wiki_competition_snapshot",
        ):
            terms = recreate.SNAPSHOT_MODEL_TERMS[model_name]
            self.assertIn("log1p_V", terms)
            self.assertNotIn("log1p_U", terms)
            self.assertNotIn("log1p_R", terms)
            self.assertNotIn("log1p_E", terms)

    def test_feature_panel_uses_latest_forecast_available_as_of_snapshot(self) -> None:
        focal = movie(1, dt.date(2026, 5, 1), release_year=2026)
        forecasts = [
            bop_forecast(
                1,
                prediction_id=1,
                article_id=1,
                published_date=dt.date(2026, 4, 20),
                target_start_date=focal.opening_date,
                low=10_000_000,
                high=20_000_000,
            ),
            bop_forecast(
                1,
                prediction_id=2,
                article_id=2,
                published_date=dt.date(2026, 4, 25),
                target_start_date=focal.opening_date,
                low=30_000_000,
                high=50_000_000,
            ),
        ]

        panel = recreate.build_day_by_day_feature_panel(
            [focal],
            [],
            {1: {-7: {"V": 100.0, "U": 2.0, "R": 1.0, "E": 1.0}}},
            forecasts,
            snapshot_days=[-7],
            train_start_year=2022,
            train_end_year=2024,
        )

        self.assertEqual(dt.date(2026, 4, 24).isoformat(), panel[0]["as_of_date"])
        self.assertEqual(1, panel[0]["bop_prediction_id"])
        self.assertEqual(15_000_000.0, panel[0]["bop_forecast_midpoint"])
        self.assertEqual(1.0, panel[0]["bop_forecast_count_as_of"])
        self.assertEqual(15_000_000.0, panel[0]["bop_first_forecast_midpoint"])
        self.assertEqual(0.0, panel[0]["bop_previous_forecast_midpoint"])
        self.assertEqual(11.0, panel[0]["bop_latest_lead_days"])
        self.assertEqual("8_14d", panel[0]["bop_latest_lead_bucket"])
        self.assertEqual(0.0, panel[0]["bop_revision_from_first_pct"])

    def test_feature_panel_excludes_after_open_estimates_and_tracks_revisions(self) -> None:
        focal = movie(1, dt.date(2026, 5, 1), release_year=2026)
        forecasts = [
            bop_forecast(
                1,
                prediction_id=1,
                article_id=1,
                published_date=dt.date(2026, 4, 17),
                target_start_date=focal.opening_date,
                low=10_000_000,
                high=20_000_000,
            ),
            bop_forecast(
                1,
                prediction_id=2,
                article_id=2,
                published_date=dt.date(2026, 4, 30),
                target_start_date=focal.opening_date,
                low=30_000_000,
                high=50_000_000,
            ),
            bop_forecast(
                1,
                prediction_id=3,
                article_id=3,
                published_date=dt.date(2026, 5, 2),
                target_start_date=focal.opening_date,
                low=80_000_000,
                high=90_000_000,
            ),
        ]

        panel = recreate.build_day_by_day_feature_panel(
            [focal],
            [],
            {},
            forecasts,
            snapshot_days=[1],
            train_start_year=2022,
            train_end_year=2024,
        )

        self.assertEqual(2, panel[0]["bop_prediction_id"])
        self.assertEqual(2.0, panel[0]["bop_forecast_count_as_of"])
        self.assertEqual(15_000_000.0, panel[0]["bop_first_forecast_midpoint"])
        self.assertEqual(15_000_000.0, panel[0]["bop_previous_forecast_midpoint"])
        self.assertEqual(1.0, panel[0]["bop_latest_lead_days"])
        self.assertEqual("0_2d", panel[0]["bop_latest_lead_bucket"])
        self.assertAlmostEqual((40_000_000 - 15_000_000) / 15_000_000, panel[0]["bop_revision_from_first_pct"])
        self.assertAlmostEqual((40_000_000 - 15_000_000) / 15_000_000, panel[0]["bop_revision_from_previous_pct"])
        self.assertEqual("after_open", recreate.bop_lead_bucket_for_days(-1))

    def test_feature_panel_excludes_focal_movie_from_competition(self) -> None:
        focal = movie(1, dt.date(2026, 5, 1), release_year=2026)
        as_of = dt.date(2026, 4, 24)
        panel = recreate.build_day_by_day_feature_panel(
            [focal],
            [
                daily(1, as_of, 999_000_000),
                daily(2, as_of, 10_000_000),
                daily(3, as_of - dt.timedelta(days=1), 5_000_000),
            ],
            {},
            [bop_forecast(1, target_start_date=focal.opening_date, published_date=dt.date(2026, 4, 20))],
            snapshot_days=[-7],
            train_start_year=2022,
            train_end_year=2024,
        )

        self.assertEqual(5_000_000.0, panel[0]["competitor_total_gross_lag1"])
        self.assertEqual(5_000_000.0, panel[0]["competitor_total_gross_lag7"])

    def test_feature_panel_uses_one_day_lag_for_known_actuals(self) -> None:
        focal = movie(1, dt.date(2026, 5, 1), release_year=2026, weekend_gross=30_000_000)
        grosses = [
            daily(1, focal.opening_date + dt.timedelta(days=0), 10_000_000),
            daily(1, focal.opening_date + dt.timedelta(days=1), 12_000_000),
            daily(1, focal.opening_date + dt.timedelta(days=2), 8_000_000),
            daily(1, focal.opening_date + dt.timedelta(days=3), 4_000_000),
            daily(1, focal.opening_date + dt.timedelta(days=4), 3_000_000),
        ]

        panel = recreate.build_day_by_day_feature_panel(
            [focal],
            grosses,
            {},
            [],
            snapshot_days=[0, 1, 2, 3, 5],
            train_start_year=2022,
            train_end_year=2024,
            target_types=["3_day", "5_day"],
        )

        by_target_day = {(row["target_type"], row["snapshot_day"]): row for row in panel}
        self.assertEqual("", by_target_day[("3_day", 0)]["known_actual_offsets"])
        self.assertEqual("0", by_target_day[("3_day", 1)]["known_actual_offsets"])
        self.assertEqual("0,1", by_target_day[("3_day", 2)]["known_actual_offsets"])
        self.assertEqual("0,1,2", by_target_day[("3_day", 3)]["known_actual_offsets"])
        self.assertEqual(30_000_000.0, by_target_day[("3_day", 3)]["known_gross_so_far"])
        self.assertEqual("0,1,2,3,4", by_target_day[("5_day", 5)]["known_actual_offsets"])
        self.assertEqual(37_000_000.0, by_target_day[("5_day", 5)]["known_gross_so_far"])

    def test_intervals_use_training_residuals_and_fall_back_to_global_when_bucket_is_sparse(self) -> None:
        train_rows = [
            {"bop_estimate_bucket_under_15m": 1.0, "bop_forecast_available": 1.0},
            {"bop_estimate_bucket_under_15m": 1.0, "bop_forecast_available": 1.0},
            {"bop_estimate_bucket_100m_plus": 1.0, "bop_forecast_available": 1.0},
            {"bop_estimate_bucket_100m_plus": 1.0, "bop_forecast_available": 1.0},
            {"bop_estimate_bucket_100m_plus": 1.0, "bop_forecast_available": 1.0},
            {"bop_estimate_bucket_100m_plus": 1.0, "bop_forecast_available": 1.0},
            {"bop_estimate_bucket_100m_plus": 1.0, "bop_forecast_available": 1.0},
        ]
        residuals = [-0.2, 0.0, 0.1, 0.2, 0.3, 0.4, 9.9]
        by_bucket, global_residuals = recreate.interval_residuals_by_bucket(
            rows=train_rows,
            residuals=residuals,
        )

        sparse = recreate.interval_for_row(
            {"bop_estimate_bucket_under_15m": 1.0, "bop_forecast_available": 1.0},
            pred_log=16.0,
            bucket_residuals=by_bucket,
            global_residuals=global_residuals,
            min_bucket_residuals=5,
        )
        dense = recreate.interval_for_row(
            {"bop_estimate_bucket_100m_plus": 1.0, "bop_forecast_available": 1.0},
            pred_log=16.0,
            bucket_residuals=by_bucket,
            global_residuals=global_residuals,
            min_bucket_residuals=5,
        )

        self.assertEqual("global", sparse["prediction_interval_source"])
        self.assertEqual("empirical_quantile", sparse["prediction_interval_method"])
        self.assertEqual(7, sparse["prediction_interval_train_n"])
        self.assertEqual("bucket", dense["prediction_interval_source"])
        self.assertEqual(5, dense["prediction_interval_train_n"])
        conformal = recreate.interval_for_row(
            {"bop_estimate_bucket_100m_plus": 1.0, "bop_forecast_available": 1.0},
            pred_log=16.0,
            bucket_residuals=by_bucket,
            global_residuals=global_residuals,
            interval_method="conformal_abs",
            min_bucket_residuals=5,
        )
        self.assertEqual("conformal_abs", conformal["prediction_interval_method"])

    def test_displayed_selection_prediction_reconciles_in_weekend_actuals(self) -> None:
        row = {
            "opening_weekend_day0_gross_usd": 10_000_000,
            "opening_weekend_day1_gross_usd": 12_000_000,
            "opening_weekend_day2_gross_usd": 8_000_000,
        }

        friday, friday_locked = recreate.displayed_prediction_usd_for_snapshot(
            row,
            snapshot_day=1,
            base_prediction_usd=30_000_000,
            remainder_models={(0,): 2.0, (0, 1): 0.5},
        )
        saturday, saturday_locked = recreate.displayed_prediction_usd_for_snapshot(
            row,
            snapshot_day=2,
            base_prediction_usd=30_000_000,
            remainder_models={(0,): 2.0, (0, 1): 0.5},
        )

        self.assertEqual(30_000_000, friday)
        self.assertEqual((0,), friday_locked)
        self.assertEqual(33_000_000, saturday)
        self.assertEqual((0, 1), saturday_locked)

    def test_checkpoint_known_actuals_include_previews_and_lock_only_available_days(self) -> None:
        focal = movie(1, dt.date(2026, 5, 1), release_year=2026, weekend_gross=30_000_000)
        forecasts = [
            bop_forecast(
                1,
                published_date=dt.date(2026, 4, 30),
                target_start_date=focal.opening_date,
                low=20_000_000,
                high=20_000_000,
            )
        ]
        panel = recreate.build_day_by_day_feature_panel(
            [focal],
            [
                daily(1, focal.opening_date, 10_000_000),
                daily(1, focal.opening_date + dt.timedelta(days=1), 12_000_000),
                daily(1, focal.opening_date + dt.timedelta(days=2), 8_000_000),
                daily(2, focal.opening_date, 1_000_000),
                daily(2, focal.opening_date + dt.timedelta(days=1), 2_000_000),
                daily(2, focal.opening_date + dt.timedelta(days=2), 999_000_000),
            ],
            {},
            forecasts,
            snapshot_days=[1, 2],
            train_start_year=2022,
            train_end_year=2025,
            preview_grosses=[
                daily(1, focal.opening_date - dt.timedelta(days=2), 1_000_000),
                daily(1, focal.opening_date - dt.timedelta(days=1), 2_000_000),
                daily(1, focal.opening_date, 999_000_000),
            ],
        )

        friday = next(row for row in panel if row["snapshot_day"] == 1)
        saturday = next(row for row in panel if row["snapshot_day"] == 2)

        self.assertEqual(3_000_000.0, friday["preview_gross_known_usd"])
        self.assertEqual(2.0, friday["preview_day_count"])
        self.assertEqual(10_000_000.0, friday["known_friday_gross_usd"])
        self.assertEqual(0.0, friday["known_saturday_gross_usd"])
        self.assertEqual(13_000_000.0, friday["known_weekend_gross_so_far_usd"])
        self.assertEqual(1_000_000.0, friday["competitor_total_gross_lag1"])
        self.assertIn("share_attraction_competitor_pressure_lag7", friday)
        self.assertIn("logit_demand_focal_vs_competition_lag7", friday)
        self.assertEqual(12_000_000.0, saturday["known_saturday_gross_usd"])
        self.assertEqual(25_000_000.0, saturday["known_weekend_gross_so_far_usd"])
        self.assertEqual(2_000_000.0, saturday["competitor_total_gross_lag1"])

    def test_wiki_features_are_cumulative_as_of_snapshot_day(self) -> None:
        focal = movie(1, dt.date(2026, 5, 1), release_year=2026, weekend_gross=30_000_000)
        forecasts = [
            bop_forecast(
                1,
                published_date=dt.date(2026, 4, 1),
                target_start_date=focal.opening_date,
                low=20_000_000,
                high=20_000_000,
            )
        ]
        panel = recreate.build_day_by_day_feature_panel(
            [focal],
            [
                daily(1, focal.opening_date, 10_000_000),
                daily(1, focal.opening_date + dt.timedelta(days=1), 12_000_000),
                daily(1, focal.opening_date + dt.timedelta(days=2), 8_000_000),
            ],
            {
                1: {
                    -2: {"V": 5.0, "U": 1.0, "R": 1.0, "E": 1.0},
                    0: {"V": 10.0, "U": 2.0, "R": 2.0, "E": 2.0},
                    2: {"V": 99.0, "U": 9.0, "R": 9.0, "E": 9.0},
                }
            },
            forecasts,
            snapshot_days=[-1, 1],
            train_start_year=2022,
            train_end_year=2024,
            target_types=[recreate.THREE_DAY_TARGET],
        )

        tminus1 = next(row for row in panel if row["snapshot_day"] == -1)
        friday_known = next(row for row in panel if row["snapshot_day"] == 1)

        self.assertEqual(5.0, tminus1["V"])
        self.assertEqual(10.0, friday_known["V"])

    def test_competitive_checkpoint_split_is_train_2022_2025_test_2026(self) -> None:
        rows = [
            {"release_year": 2021},
            {"release_year": 2022},
            {"release_year": 2025},
            {"release_year": 2026},
            {"release_year": 2027},
        ]

        train, test = recreate.split_competitive_checkpoint_rows(
            rows,
            train_start_year=2022,
            train_end_year=2025,
            test_start_year=2026,
            test_end_year=2026,
        )

        self.assertEqual([2022, 2025], [row["release_year"] for row in train])
        self.assertEqual([2026], [row["release_year"] for row in test])

    def test_competitive_checkpoint_scope_filters_by_bop_midpoint(self) -> None:
        rows = [
            {
                "snapshot_day": -1,
                "bop_forecast_available": 1.0,
                "bop_forecast_midpoint": 4_999_999.0,
                "target_log_bop_residual": 0.1,
            },
            {
                "snapshot_day": -1,
                "bop_forecast_available": 1.0,
                "bop_forecast_midpoint": 5_000_000.0,
                "target_log_bop_residual": 0.1,
            },
            {
                "snapshot_day": 1,
                "bop_forecast_available": 1.0,
                "bop_forecast_midpoint": 9_000_000.0,
                "target_log_bop_residual": 0.1,
            },
        ]

        scoped = recreate.competitive_checkpoint_scoped_rows(
            rows,
            snapshot_day=-1,
            min_bop_forecast_midpoint=5_000_000,
        )

        self.assertEqual([5_000_000.0], [row["bop_forecast_midpoint"] for row in scoped])

    def test_wiki_bop_timing_scope_filters_by_bop_midpoint(self) -> None:
        rows = [
            {
                "target_type": "3_day",
                "snapshot_day": -1,
                "bop_forecast_available": 1.0,
                "bop_forecast_midpoint": 4_999_999.0,
                "target_log_bop_residual": 0.1,
            },
            {
                "target_type": "3_day",
                "snapshot_day": -1,
                "bop_forecast_available": 1.0,
                "bop_forecast_midpoint": 5_000_000.0,
                "target_log_bop_residual": 0.1,
            },
            {
                "target_type": "5_day",
                "snapshot_day": -1,
                "bop_forecast_available": 1.0,
                "bop_forecast_midpoint": 9_000_000.0,
                "target_log_bop_residual": 0.1,
            },
        ]

        scoped = recreate.wiki_bop_timing_scoped_rows(
            rows,
            snapshot_day=-1,
            min_bop_forecast_midpoint=5_000_000,
        )

        self.assertEqual([5_000_000.0], [row["bop_forecast_midpoint"] for row in scoped])

    def test_wiki_bop_timing_locks_actuals_by_snapshot_day(self) -> None:
        row = {
            "movie_id": 1,
            "title": "One",
            "release_year": 2026,
            "opening_date": "2026-05-01",
            "as_of_date": "2026-05-04",
            "bop_forecast_midpoint": 25_000_000.0,
            "opening_weekend_day0_gross_usd": 10_000_000.0,
            "opening_weekend_day1_gross_usd": 12_000_000.0,
            "opening_weekend_day2_gross_usd": 8_000_000.0,
            "opening_weekend_revenue_usd": 30_000_000.0,
            "target_log_bop_residual": math.log(30_000_000.0) - math.log(25_000_000.0),
            "target_log_opening_weekend": math.log(30_000_000.0),
        }
        remainder = {(0,): 2.0, (0, 1): 0.5}

        friday = recreate.wiki_bop_timing_prediction_rows(
            snapshot_day=1,
            model_name="raw_bop_available",
            rows=[row],
            predicted_residuals=[0.0],
            remainder_models=remainder,
        )[0]
        saturday = recreate.wiki_bop_timing_prediction_rows(
            snapshot_day=2,
            model_name="raw_bop_available",
            rows=[row],
            predicted_residuals=[0.0],
            remainder_models=remainder,
        )[0]
        monday = recreate.wiki_bop_timing_prediction_rows(
            snapshot_day=3,
            model_name="raw_bop_available",
            rows=[row],
            predicted_residuals=[0.0],
            remainder_models=remainder,
        )[0]

        self.assertEqual("0", friday["known_actual_offsets"])
        self.assertEqual(30_000_000.0, friday["predicted_opening_weekend_revenue_usd"])
        self.assertEqual("0,1", saturday["known_actual_offsets"])
        self.assertEqual(33_000_000.0, saturday["predicted_opening_weekend_revenue_usd"])
        self.assertEqual("0,1,2", monday["known_actual_offsets"])
        self.assertEqual(30_000_000.0, monday["predicted_opening_weekend_revenue_usd"])

    def test_wiki_remaining_rows_predict_remaining_residual_directly(self) -> None:
        row = {
            "movie_id": 1,
            "title": "One",
            "release_year": 2026,
            "opening_date": "2026-05-01",
            "as_of_date": "2026-05-02",
            "bop_forecast_midpoint": 25_000_000.0,
            "opening_weekend_day0_gross_usd": 10_000_000.0,
            "opening_weekend_day1_gross_usd": 12_000_000.0,
            "opening_weekend_day2_gross_usd": 8_000_000.0,
            "remaining_gross": 20_000_000.0,
            "opening_weekend_revenue_usd": 30_000_000.0,
            "target_log_opening_weekend": math.log(30_000_000.0),
        }

        rows = recreate.remaining_weekend_rows_with_baseline(
            [row],
            snapshot_day=1,
            remainder_models={(0,): 1.5},
        )
        predictions = recreate.wiki_remaining_prediction_rows(
            snapshot_day=1,
            model_name="remaining_baseline_ratio",
            rows=rows,
            predicted_residuals=[0.0],
        )

        self.assertEqual(15_000_000.0, rows[0]["remaining_baseline_gross_usd"])
        self.assertAlmostEqual(math.log(20_000_000.0) - math.log(15_000_000.0), rows[0]["target_log_remaining_baseline_residual"])
        self.assertEqual(15_000_000.0, predictions[0]["predicted_remaining_gross_usd"])
        self.assertEqual(25_000_000.0, predictions[0]["predicted_opening_weekend_revenue_usd"])

    def test_raw_checkpoint_prediction_reconciles_up_to_known_gross(self) -> None:
        rows = recreate.competitive_checkpoint_prediction_rows(
            checkpoint="saturday_actual",
            snapshot_day=2,
            model_name="raw_bop_reconciled",
            rows=[
                {
                    "movie_id": 1,
                    "title": "One",
                    "release_year": 2026,
                    "opening_date": "2026-05-01",
                    "as_of_date": "2026-05-03",
                    "bop_forecast_midpoint": 10_000_000.0,
                    "known_weekend_gross_so_far_usd": 12_000_000.0,
                    "opening_weekend_revenue_usd": 15_000_000.0,
                    "target_log_bop_residual": 0.0,
                    "target_log_opening_weekend": 16.0,
                }
            ],
            predicted_residuals=[0.0],
        )

        self.assertEqual(12_000_000.0, rows[0]["predicted_opening_weekend_revenue_usd"])

    def test_competitive_checkpoint_models_are_separate_by_information_day(self) -> None:
        rows = []
        movie_id = 1
        for year in range(2022, 2027):
            for index in range(3):
                midpoint = 10_000_000 + (year - 2022) * 500_000 + index * 1_000_000
                actual = midpoint * (1.05 + index * 0.01)
                for snapshot_day in (-1, 1, 2):
                    row = {
                        "movie_id": movie_id,
                        "title": f"Movie {movie_id}",
                        "release_year": year,
                        "opening_date": f"{year}-05-01",
                        "snapshot_day": snapshot_day,
                        "as_of_date": f"{year}-05-0{max(1, snapshot_day + 1)}",
                        "bop_forecast_available": 1.0,
                        "bop_forecast_midpoint": float(midpoint),
                        "log1p_bop_forecast_midpoint": opening.log1p(midpoint),
                        "opening_weekend_revenue_usd": actual,
                        "target_log_opening_weekend": math.log(actual),
                        "target_log_bop_residual": math.log(actual) - math.log(midpoint),
                        "preview_gross_known_usd": 500_000.0 + index,
                        "preview_day_count": 2.0,
                        "log1p_preview_gross_known": opening.log1p(500_000 + index),
                        "known_friday_gross_usd": 0.0 if snapshot_day < 1 else 4_000_000.0 + index,
                        "known_saturday_gross_usd": 0.0 if snapshot_day < 2 else 5_000_000.0 + index,
                        "known_weekend_gross_so_far_usd": (
                            500_000.0
                            + (4_000_000.0 if snapshot_day >= 1 else 0.0)
                            + (5_000_000.0 if snapshot_day >= 2 else 0.0)
                        ),
                        "known_gross_so_far_pct_of_bop_midpoint": 0.4 + index * 0.01,
                        "log1p_known_friday_gross": opening.log1p(0 if snapshot_day < 1 else 4_000_000 + index),
                        "log1p_known_saturday_gross": opening.log1p(0 if snapshot_day < 2 else 5_000_000 + index),
                        "log1p_competitor_total_gross_lag1": opening.log1p(2_000_000 + index * 100_000),
                        "log1p_competitor_total_gross_lag7": opening.log1p(8_000_000 + index * 200_000 + year),
                        "log1p_competitor_total_gross_previous_weekend": opening.log1p(7_000_000 + index * 150_000),
                        "share_attraction_focal_share_lag7": 0.4 + index * 0.01,
                        "share_attraction_competitor_pressure_lag7": 0.2 + index * 0.01,
                        "share_attraction_top1_pressure_lag7": 0.1 + index * 0.01,
                        "logit_demand_focal_vs_competition_lag7": 0.5 - index * 0.02,
                        "logit_demand_focal_vs_top1_lag7": 0.6 - index * 0.02,
                        "log1p_logit_demand_choice_set_lag7": 16.0 + index * 0.1,
                    }
                    row.update({term: 0.0 for term in recreate.FIXED_BUCKET_TERMS})
                    row["bop_estimate_bucket_30_60m"] = 1.0
                    rows.append(row)
                movie_id += 1

        predictions, metrics, coefficients, headline = recreate.evaluate_competitive_checkpoints(rows)

        self.assertEqual({2026}, {int(row["release_year"]) for row in predictions})
        self.assertIn("tminus1_competition", {row["model"] for row in metrics})
        self.assertIn("tminus1_share_attraction", {row["model"] for row in metrics})
        self.assertIn("tminus1_logit_demand", {row["model"] for row in metrics})
        self.assertIn("friday_actual_competition", {row["model"] for row in metrics})
        self.assertIn("friday_actual_share_attraction", {row["model"] for row in metrics})
        self.assertIn("friday_actual_logit_demand", {row["model"] for row in metrics})
        self.assertIn("saturday_actual_competition", {row["model"] for row in metrics})
        self.assertIn("saturday_actual_share_attraction", {row["model"] for row in metrics})
        self.assertIn("saturday_actual_logit_demand", {row["model"] for row in metrics})
        self.assertEqual(
            {("pre_release_tminus1", -1), ("friday_actual", 1), ("saturday_actual", 2)},
            {(row["checkpoint"], row["snapshot_day"]) for row in headline},
        )
        self.assertIn("tminus1_share_attraction", {row["competition_model"] for row in headline})
        self.assertIn("friday_actual_logit_demand", {row["competition_model"] for row in headline})
        self.assertIn("saturday_actual_share_attraction", {row["competition_model"] for row in headline})
        fitted_models = {row["model"] for row in coefficients if row["target"] == "target_log_bop_residual"}
        self.assertIn("tminus1_competition", fitted_models)
        self.assertIn("tminus1_share_attraction", fitted_models)
        self.assertIn("tminus1_logit_demand", fitted_models)
        self.assertIn("friday_actual_only", fitted_models)
        self.assertIn("friday_actual_share_attraction", fitted_models)
        self.assertIn("friday_actual_logit_demand", fitted_models)
        self.assertIn("saturday_actual_only", fitted_models)
        self.assertIn("saturday_actual_share_attraction", fitted_models)
        self.assertIn("saturday_actual_logit_demand", fitted_models)
        self.assertIn("expected_competition_lag7", {row["model"] for row in coefficients})

    def test_wiki_bop_timing_models_use_requested_split_and_separate_coefficients(self) -> None:
        rows = []
        movie_id = 1
        for year in range(2022, 2027):
            for index in range(3):
                midpoint = 8_000_000 + (year - 2022) * 500_000 + index * 750_000
                actual = midpoint * (1.02 + index * 0.03)
                friday = actual * 0.35
                saturday = actual * 0.4
                sunday = actual - friday - saturday
                for snapshot_day in (-1, 1, 2, 3):
                    row = {
                        "movie_id": movie_id,
                        "title": f"Movie {movie_id}",
                        "release_year": year,
                        "opening_date": f"{year}-05-01",
                        "target_type": "3_day",
                        "snapshot_day": snapshot_day,
                        "as_of_date": f"{year}-05-{max(1, snapshot_day + 1):02d}",
                        "bop_forecast_available": 1.0,
                        "bop_forecast_midpoint": float(midpoint),
                        "opening_weekend_revenue_usd": actual,
                        "opening_weekend_day0_gross_usd": friday,
                        "opening_weekend_day1_gross_usd": saturday,
                        "opening_weekend_day2_gross_usd": sunday,
                        "target_log_opening_weekend": math.log(actual),
                        "target_log_bop_residual": math.log(actual) - math.log(midpoint),
                        "wiki_available": 1.0,
                        "V": 100.0 + index * 10 + snapshot_day,
                        "U": 10.0 + index + max(snapshot_day, 0),
                        "R": 5.0 + index + max(snapshot_day, 0),
                        "E": 15.0 + index + max(snapshot_day, 0),
                        "log1p_V": opening.log1p(100 + index * 10 + snapshot_day),
                        "log1p_U": opening.log1p(10 + index + max(snapshot_day, 0)),
                        "log1p_R": opening.log1p(5 + index + max(snapshot_day, 0)),
                        "log1p_E": opening.log1p(15 + index + max(snapshot_day, 0)),
                        "log1p_opening_theaters": opening.log1p(3_000 + index * 100),
                    }
                    rows.append(row)
                movie_id += 1
        for snapshot_day in (-1, 1, 2, 3):
            rows.append(
                {
                    "movie_id": 10_000 + snapshot_day,
                    "title": "Too Small",
                    "release_year": 2025,
                    "opening_date": "2025-05-01",
                    "target_type": "3_day",
                    "snapshot_day": snapshot_day,
                    "as_of_date": "2025-05-01",
                    "bop_forecast_available": 1.0,
                    "bop_forecast_midpoint": 4_000_000.0,
                    "opening_weekend_revenue_usd": 4_100_000.0,
                    "opening_weekend_day0_gross_usd": 1_400_000.0,
                    "opening_weekend_day1_gross_usd": 1_600_000.0,
                    "opening_weekend_day2_gross_usd": 1_100_000.0,
                    "target_log_opening_weekend": math.log(4_100_000.0),
                    "target_log_bop_residual": math.log(4_100_000.0) - math.log(4_000_000.0),
                    "wiki_available": 1.0,
                    "log1p_V": opening.log1p(100),
                    "log1p_U": opening.log1p(10),
                    "log1p_R": opening.log1p(5),
                    "log1p_E": opening.log1p(15),
                    "log1p_opening_theaters": opening.log1p(1_000),
                }
            )

        predictions, metrics, coefficients, headline = recreate.evaluate_wiki_bop_timing(
            rows,
            snapshot_days=[-1, 1, 2, 3],
            train_start_year=2022,
            train_end_year=2024,
            test_start_year=2025,
            test_end_year=2026,
            min_bop_forecast_midpoint=5_000_000,
        )

        self.assertEqual({2025, 2026}, {int(row["release_year"]) for row in predictions})
        self.assertTrue(all(float(row["bop_forecast_midpoint"]) >= 5_000_000 for row in predictions))
        self.assertEqual(
            {"raw_bop_available", "bop_residual_wiki_views", "bop_residual_wiki_full_activity", "bop_residual_wiki_full_activity_plus_theaters"},
            {row["model"] for row in metrics},
        )
        self.assertEqual({-1, 1, 2, 3}, {int(row["snapshot_day"]) for row in coefficients})
        primary_terms = {
            row["term"]
            for row in coefficients
            if row["model"] in {"bop_residual_wiki_views", "bop_residual_wiki_full_activity"}
        }
        self.assertNotIn("log1p_opening_theaters", primary_terms)
        self.assertIn("log1p_opening_theaters", {row["term"] for row in coefficients if row["model"] == "bop_residual_wiki_full_activity_plus_theaters"})
        self.assertTrue(
            all(
                row["predicted_opening_weekend_revenue_usd"] == row["actual_opening_weekend_revenue_usd"]
                for row in predictions
                if int(row["snapshot_day"]) == 3
            )
        )
        self.assertIn("bop_residual_wiki_views", {row["wiki_model"] for row in headline})
        self.assertIn("bop_residual_wiki_full_activity", {row["wiki_model"] for row in headline})

    def test_wiki_remaining_timing_compares_views_and_full_activity(self) -> None:
        rows = []
        movie_id = 1
        for year in range(2022, 2027):
            for index in range(4):
                total = 10_000_000.0 + index * 1_000_000 + (year - 2022) * 250_000
                friday = total * 0.35
                saturday = total * 0.4
                sunday = total - friday - saturday
                for snapshot_day in (1, 2):
                    row = {
                        "movie_id": movie_id,
                        "title": f"Movie {movie_id}",
                        "release_year": year,
                        "opening_date": f"{year}-05-01",
                        "target_type": "3_day",
                        "snapshot_day": snapshot_day,
                        "as_of_date": f"{year}-05-0{snapshot_day + 1}",
                        "bop_forecast_available": 1.0,
                        "bop_forecast_midpoint": 9_000_000.0 + index * 1_000_000,
                        "opening_weekend_revenue_usd": total,
                        "opening_weekend_day0_gross_usd": friday,
                        "opening_weekend_day1_gross_usd": saturday,
                        "opening_weekend_day2_gross_usd": sunday,
                        "known_gross_so_far": friday if snapshot_day == 1 else friday + saturday,
                        "remaining_gross": total - (friday if snapshot_day == 1 else friday + saturday),
                        "target_log_opening_weekend": math.log(total),
                        "target_log_bop_residual": math.log(total) - math.log(9_000_000.0 + index * 1_000_000),
                        "wiki_available": 1.0,
                        "log1p_V": opening.log1p(100 + index),
                        "log1p_U": opening.log1p(10 + index),
                        "log1p_R": opening.log1p(5 + index),
                        "log1p_E": opening.log1p(15 + index),
                    }
                    rows.append(row)
                movie_id += 1

        predictions, metrics, coefficients, headline = recreate.evaluate_wiki_remaining_timing(
            rows,
            snapshot_days=[1, 2],
            train_start_year=2022,
            train_end_year=2024,
            test_start_year=2025,
            test_end_year=2026,
            min_bop_forecast_midpoint=5_000_000,
        )

        self.assertEqual({2025, 2026}, {int(row["release_year"]) for row in predictions})
        self.assertIn("remaining_baseline_ratio", {row["model"] for row in metrics})
        self.assertIn("remaining_residual_wiki_views", {row["model"] for row in metrics})
        self.assertIn("remaining_residual_wiki_full_activity", {row["model"] for row in metrics})
        self.assertEqual({1, 2}, {int(row["snapshot_day"]) for row in coefficients})
        self.assertIn("target_log_remaining_baseline_residual", {row["target"] for row in coefficients})
        self.assertIn("remaining_residual_wiki_views", {row["wiki_model"] for row in headline})
        self.assertIn("remaining_residual_wiki_full_activity", {row["wiki_model"] for row in headline})

    def test_best_competitive_selection_rows_rank_by_mape_with_tie_breakers(self) -> None:
        rows = [
            {
                "status": "ok",
                "train_cutoff": 5_000_000,
                "test_cutoff": 5_000_000,
                "switch_day": -1,
                "split": "a",
                "snapshot_day": -1,
                "holdout_n": 10,
                "mape_gross": "0.1",
                "rmse_log_revenue": "0.2",
                "r2_gross": "0.7",
            },
            {
                "status": "ok",
                "train_cutoff": 5_000_000,
                "test_cutoff": 5_000_000,
                "switch_day": 0,
                "split": "a",
                "snapshot_day": -1,
                "holdout_n": 10,
                "mape_gross": "0.1",
                "rmse_log_revenue": "0.2",
                "r2_gross": "0.7",
            },
            {
                "status": "ok",
                "train_cutoff": 1_000_000,
                "test_cutoff": 5_000_000,
                "switch_day": -1,
                "split": "a",
                "snapshot_day": -1,
                "holdout_n": 10,
                "mape_gross": "0.2",
                "rmse_log_revenue": "0.1",
                "r2_gross": "0.9",
            },
        ]

        best = recreate.best_competitive_selection_rows(rows)

        self.assertTrue(best[0]["selected"])
        self.assertEqual(0, best[0]["switch_day"])

    def test_day_by_day_scoring_uses_only_2025_2026_holdout_and_reports_models(self) -> None:
        rows = []
        years = [2022] * 8 + [2023] * 4 + [2024] * 4 + [2025, 2025, 2026]
        for idx, year in enumerate(years, start=1):
            midpoint = 12_000_000 + idx * 1_000_000
            actual = 13_000_000 + idx * 900_000
            row = {
                "movie_id": idx,
                "title": f"Movie {idx}",
                "release_year": year,
                "release_run_id": idx * 10,
                "opening_date": f"{year}-05-01",
                "forecast_stage": "pre_release",
                "snapshot_day": -1,
                "as_of_date": f"{year}-04-30",
                "opening_weekend_revenue_usd": actual,
                "target_log_opening_weekend": opening.log1p(actual),
                "bop_forecast_available": 1.0,
                "bop_forecast_midpoint": float(midpoint),
                "log1p_bop_forecast_midpoint": opening.log1p(midpoint),
                "bop_forecast_range_width_pct": 0.2,
                "bop_forecast_count_as_of": 2.0,
                "log1p_bop_forecast_count_as_of": opening.log1p(2),
                "bop_first_forecast_midpoint": float(midpoint - 500_000),
                "bop_previous_forecast_midpoint": float(midpoint - 250_000),
                "bop_latest_lead_days": 1.0,
                "bop_latest_lead_bucket": "0_2d",
                "bop_revision_from_first_pct": 500_000 / (midpoint - 500_000),
                "bop_revision_from_previous_pct": 250_000 / (midpoint - 250_000),
                "wiki_available": 1.0,
                "log1p_V": opening.log1p(100 + idx),
                "log1p_U": opening.log1p(10 + idx),
                "log1p_R": opening.log1p(5 + idx),
                "log1p_E": opening.log1p(15 + idx),
                "log1p_competitor_total_gross_lag7": opening.log1p(20_000_000 + idx),
            }
            row.update({term: 1.0 for term in recreate.FIXED_BUCKET_TERMS})
            row["bop_estimate_bucket_under_15m"] = 0.0
            row["bop_estimate_bucket"] = "15_30m"
            row["target_log_bop_residual"] = float(row["target_log_opening_weekend"]) - opening.log1p(midpoint)
            rows.append(row)

        predictions, metrics, coefficients, intervals = recreate.evaluate_day_by_day_snapshots(
            rows,
            snapshot_days=[-1],
            train_start_year=2022,
            train_end_year=2024,
            test_start_year=2025,
            test_end_year=2026,
        )

        self.assertEqual({2025, 2026}, {int(row["release_year"]) for row in predictions})
        self.assertIn("raw_bop_snapshot", {row["model"] for row in metrics})
        self.assertIn("bop_residual_wiki_history_snapshot", {row["model"] for row in metrics})
        self.assertIn("bop_residual_wiki_competition_snapshot", {row["model"] for row in metrics})
        self.assertTrue(all("mse_gross" in row for row in metrics))
        self.assertTrue(all("accuracy_pct" in row for row in metrics))
        self.assertTrue(coefficients)
        self.assertEqual({"50", "80", "90"}, {row["interval_level"] for row in intervals})
        self.assertEqual(
            {"empirical_quantile", "conformal_abs", "loo_conformal_abs"},
            {row["interval_method"] for row in intervals},
        )

    def test_metric_rows_report_lift_against_raw_estimate_and_actuals_multiplier(self) -> None:
        rows = [
            {
                "actual_log_opening_weekend": opening.log1p(100.0),
                "predicted_log_opening_weekend": opening.log1p(95.0),
                "actual_opening_weekend_revenue_usd": 100.0,
                "predicted_opening_weekend_revenue_usd": 95.0,
                "bop_forecast_midpoint": 80.0,
                "actuals_multiplier_prediction_usd": 90.0,
                "predicted_lower_50_opening_weekend_revenue_usd": 90.0,
                "predicted_upper_50_opening_weekend_revenue_usd": 100.0,
                "predicted_lower_80_opening_weekend_revenue_usd": 90.0,
                "predicted_upper_80_opening_weekend_revenue_usd": 100.0,
                "predicted_lower_90_opening_weekend_revenue_usd": 90.0,
                "predicted_upper_90_opening_weekend_revenue_usd": 100.0,
            }
        ]

        metric = recreate.metric_row_from_predictions(base={}, prediction_rows=rows)

        self.assertEqual("5", metric["mae_usd"])
        self.assertEqual("20", metric["raw_estimate_mae_usd"])
        self.assertEqual("15", metric["raw_estimate_mae_lift_usd"])
        self.assertEqual("10", metric["actuals_multiplier_mae_usd"])
        self.assertEqual("5", metric["actuals_multiplier_mae_lift_usd"])

    def test_writer_creates_day_by_day_artifacts(self) -> None:
        prediction_rows = [
            {
                "model": "bop_residual_wiki_competition_snapshot",
                "population": "bop_covered",
                "prediction_source": "bop",
                "forecast_stage": "pre_release",
                "snapshot_day": -1,
                "as_of_date": "2026-04-30",
                "movie_id": 1,
                "title": "Movie 1",
                "release_year": 2026,
                "opening_date": "2026-05-01",
                "bop_forecast_available": 1.0,
                "bop_forecast_midpoint": 10_000_000,
                "bop_estimate_bucket": "under_15m",
                "bop_forecast_count_as_of": 1.0,
                "bop_first_forecast_midpoint": 10_000_000,
                "bop_previous_forecast_midpoint": 0.0,
                "bop_latest_lead_days": 1.0,
                "bop_latest_lead_bucket": "0_2d",
                "bop_revision_from_first_pct": 0.0,
                "bop_revision_from_previous_pct": 0.0,
                "wiki_available": 1.0,
                "actual_log_opening_weekend": 16.0,
                "predicted_log_opening_weekend": 16.0,
                "actual_opening_weekend_revenue_usd": 10_000_000,
                "predicted_opening_weekend_revenue_usd": 10_000_000,
                "predicted_p50_opening_weekend_revenue_usd": 10_000_000,
                "predicted_lower_50_opening_weekend_revenue_usd": 9_000_000,
                "predicted_upper_50_opening_weekend_revenue_usd": 11_000_000,
                "predicted_lower_80_opening_weekend_revenue_usd": 8_000_000,
                "predicted_upper_80_opening_weekend_revenue_usd": 12_000_000,
                "predicted_lower_90_opening_weekend_revenue_usd": 7_000_000,
                "predicted_upper_90_opening_weekend_revenue_usd": 13_000_000,
                "prediction_interval_method": "conformal_abs",
                "prediction_interval_source": "global",
                "prediction_interval_train_n": 10,
                "absolute_percentage_error": 0.0,
            }
        ]
        metric_rows = [
            {
                "model": "bop_residual_wiki_competition_snapshot",
                "population": "bop_covered",
                "forecast_stage": "pre_release",
                "snapshot_day": -1,
                "interval_method": "conformal_abs",
                "train_start_year": 2022,
                "train_end_year": 2024,
                "test_start_year": 2025,
                "test_end_year": 2026,
                "train_n": 10,
                "holdout_n": 1,
                "bop_prediction_n": 1,
                "fallback_prediction_n": 0,
                "r2_log_revenue": 0.5,
                "r2_gross": 0.5,
                "mape_gross": 0.1,
                "accuracy_pct": 0.9,
                "mse_log_revenue": 0.01,
                "mse_gross": 0.0,
                "rmse_log_revenue": 0.1,
                "mae_log_revenue": 0.1,
                "mean_actual_gross": 10_000_000,
                "mean_predicted_gross": 10_000_000,
                "mean_interval_80_width_pct": 0.4,
                "coverage_50": 1.0,
                "coverage_80": 1.0,
                "coverage_90": 1.0,
                "status": "ok",
            }
        ]
        interval_rows = [
            {
                "model": "bop_residual_wiki_competition_snapshot",
                "population": "bop_covered",
                "forecast_stage": "pre_release",
                "snapshot_day": -1,
                "interval_method": "conformal_abs",
                "interval_level": "80",
                "holdout_n": 1,
                "coverage": 1.0,
                "mean_width_pct": 0.4,
            }
        ]
        competitive_selection_rows = [
            {
                "split": "2022-2024_to_2025",
                "train_start_year": 2022,
                "train_end_year": 2024,
                "test_year": 2025,
                "train_cutoff": 1_000_000,
                "test_cutoff": 5_000_000,
                "switch_day": -1,
                "model_used": "switch_wiki_competition",
                "wiki_prediction_n": 0,
                "wiki_competition_prediction_n": 2,
                "population": "bop_covered",
                "snapshot_day": -1,
                "interval_method": "displayed_forecast",
                "train_n": 10,
                "holdout_n": 2,
                "bop_prediction_n": 2,
                "fallback_prediction_n": 0,
                "r2_log_revenue": 0.5,
                "r2_gross": 0.5,
                "mape_gross": 0.1,
                "accuracy_pct": 0.9,
                "mse_log_revenue": 0.01,
                "mse_gross": 0.0,
                "rmse_log_revenue": 0.1,
                "mae_log_revenue": 0.1,
                "mean_actual_gross": 10_000_000,
                "mean_predicted_gross": 10_000_000,
                "mean_interval_80_width_pct": 0.0,
                "coverage_50": 1.0,
                "coverage_80": 1.0,
                "coverage_90": 1.0,
                "status": "ok",
                "reconciled": False,
            }
        ]
        competitive_delta_rows = [
            {
                "split": "2022-2024_to_2025",
                "train_start_year": 2022,
                "train_end_year": 2024,
                "test_year": 2025,
                "train_cutoff": 1_000_000,
                "test_cutoff": 5_000_000,
                "snapshot_day": -1,
                "holdout_n": 2,
                "wiki_mape_gross": 0.2,
                "wiki_competition_mape_gross": 0.1,
                "mape_delta_competition_minus_wiki": -0.1,
                "competition_better": True,
                "reconciled": False,
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            recreate.write_outputs(
                out_dir,
                panel_rows=[],
                prediction_rows=prediction_rows,
                metric_rows=metric_rows,
                coefficient_rows=[],
                interval_rows=interval_rows,
                revision_rows=recreate.prediction_revision_rows(prediction_rows),
                coverage=[],
                cutoff_sweep_metric_rows=[
                    {
                        "train_cutoff": 1_000_000,
                        "test_cutoff": 5_000_000,
                        **metric_rows[0],
                    }
                ],
                competitive_selection_metric_rows=competitive_selection_rows,
                competitive_mape_delta_rows=competitive_delta_rows,
            )

            for filename in [
                "day_by_day_feature_panel.csv",
                "day_by_day_forecast_snapshots.csv",
                "day_by_day_metrics_by_horizon.csv",
                "day_by_day_best_metrics_by_horizon.csv",
                "day_by_day_interval_coverage.csv",
                "day_by_day_coefficients.csv",
                "day_by_day_prediction_revisions.csv",
                "day_by_day_cutoff_sweep_metrics.csv",
                "day_by_day_cutoff_sweep_best.csv",
                "day_by_day_competitive_selection_metrics.csv",
                "day_by_day_competitive_selection_best.csv",
                "day_by_day_competitive_mape_delta_by_snapshot.csv",
                "competitive_checkpoint_predictions.csv",
                "competitive_checkpoint_metrics.csv",
                "competitive_checkpoint_coefficients.csv",
                "competitive_checkpoint_headline.csv",
                "wiki_bop_timing_predictions.csv",
                "wiki_bop_timing_metrics.csv",
                "wiki_bop_timing_coefficients.csv",
                "wiki_bop_timing_headline.csv",
                "wiki_remaining_predictions.csv",
                "wiki_remaining_metrics.csv",
                "wiki_remaining_coefficients.csv",
                "wiki_remaining_headline.csv",
                "figure_forecast_fan_chart.svg",
                "figure_gross_r2_by_horizon.svg",
                "day_by_day_bop_estimate_accuracy_rows.csv",
                "day_by_day_bop_estimate_accuracy_by_lead_bucket.csv",
            ]:
                self.assertTrue((out_dir / filename).exists(), filename)


if __name__ == "__main__":
    unittest.main()
