from __future__ import annotations

import datetime as dt
import math
import tempfile
import unittest
from pathlib import Path

from pm_box_office.research.papers import recreate_competition_opening_weekend as recreate


def movie(
    movie_id: int,
    opening_date: dt.date,
    *,
    release_year: int | None = None,
    opening_day_gross: int = 5_000_000,
    weekend_gross: int = 15_000_000,
    theaters: int = 3000,
) -> recreate.OpeningWeekendMovie:
    return recreate.OpeningWeekendMovie(
        movie_id=movie_id,
        title=f"Movie {movie_id}",
        release_year=release_year or opening_date.year,
        release_run_id=movie_id * 10,
        opening_date=opening_date,
        opening_theaters=theaters,
        opening_day_gross_usd=opening_day_gross,
        opening_weekend_revenue_usd=weekend_gross,
    )


def daily(movie_id: int, day: dt.date, gross: int) -> recreate.DailyGross:
    return recreate.DailyGross(movie_id=movie_id, box_office_date=day, gross_usd=gross, theaters=1000)


def bop_forecast(
    movie_id: int,
    *,
    prediction_id: int = 1,
    article_id: int = 1,
    published_date: dt.date = dt.date(2026, 4, 30),
    target_start_date: dt.date = dt.date(2026, 5, 1),
    low: int = 10_000_000,
    high: int = 20_000_000,
    metric: str = "domestic_opening_weekend",
    rank: int | None = 1,
) -> recreate.BoxofficeProForecast:
    return recreate.BoxofficeProForecast(
        prediction_id=prediction_id,
        movie_id=movie_id,
        article_id=article_id,
        article_url=f"https://www.boxofficepro.com/{article_id}/",
        source_movie_title=f"Movie {movie_id}",
        forecast_metric=metric,
        source_context="test",
        source_rank=rank,
        target_start_date=target_start_date,
        target_end_date=target_start_date + dt.timedelta(days=2),
        range_low_usd=float(low),
        range_high_usd=float(high),
        showtime_market_share_pct=25.0,
        published_date=published_date,
    )


class CompetitionOpeningWeekendTests(unittest.TestCase):
    def test_daily_gross_loader_excludes_estimates(self) -> None:
        class EmptyResult:
            def fetchall(self) -> list[tuple[object, ...]]:
                return []

        class FakeConn:
            sql = ""

            def execute(self, sql: str, params: tuple[object, ...]) -> EmptyResult:
                self.sql = sql
                return EmptyResult()

        conn = FakeConn()

        recreate.load_daily_grosses(
            conn,
            start_date=dt.date(2026, 1, 1),
            end_date=dt.date(2026, 1, 2),
        )

        self.assertIn("dbo.is_estimate = 0", conn.sql)

    def test_competitor_lag_features_exclude_focal_movie(self) -> None:
        as_of = dt.date(2026, 4, 24)
        rows = {
            (1, as_of): daily(1, as_of, 100),
            (2, as_of): daily(2, as_of, 50),
            (3, as_of): daily(3, as_of, 25),
        }

        features = recreate.actual_competitor_lag_features(
            rows,
            focal_movie_id=1,
            as_of_date=as_of,
            lag_days=1,
        )

        self.assertEqual(75.0, features["competitor_total_gross_lag1"])
        self.assertEqual(50.0, features["competitor_top1_gross_lag1"])
        self.assertEqual(2.0, features["competitor_count_lag1"])

    def test_lag_windows_use_only_dates_on_or_before_as_of_date(self) -> None:
        as_of = dt.date(2026, 4, 24)
        rows = {
            (2, as_of - dt.timedelta(days=2)): daily(2, as_of - dt.timedelta(days=2), 10),
            (2, as_of - dt.timedelta(days=1)): daily(2, as_of - dt.timedelta(days=1), 20),
            (2, as_of): daily(2, as_of, 30),
            (2, as_of + dt.timedelta(days=1)): daily(2, as_of + dt.timedelta(days=1), 999),
            (3, as_of - dt.timedelta(days=6)): daily(3, as_of - dt.timedelta(days=6), 40),
        }

        features = recreate.actual_competition_features(
            rows,
            focal_movie_id=1,
            as_of_date=as_of,
        )

        self.assertEqual(30.0, features["competitor_total_gross_lag1"])
        self.assertEqual(60.0, features["competitor_total_gross_lag3"])
        self.assertEqual(100.0, features["competitor_total_gross_lag7"])

    def test_feature_panel_builds_wiki_baseline_and_actual_competition(self) -> None:
        focal = movie(1, dt.date(2026, 5, 1), release_year=2026)
        as_of = dt.date(2026, 4, 24)
        daily_grosses = [
            daily(1, as_of, 999),
            daily(2, as_of, 10_000_000),
            daily(3, as_of - dt.timedelta(days=1), 5_000_000),
        ]
        wiki = {1: {-7: {"V": 100.0, "U": 10.0, "R": 3.0, "E": 4.0}}}

        panel = recreate.build_feature_panel(
            [focal],
            daily_grosses,
            wiki,
            timing_days=[-7],
        )

        self.assertEqual(100.0, panel[0]["V"])
        self.assertEqual(10_000_000.0, panel[0]["competitor_total_gross_lag1"])
        self.assertEqual(15_000_000.0, panel[0]["competitor_total_gross_lag3"])
        self.assertNotIn("bop_forecast_midpoint_usd", panel[0])

    def test_latest_bop_forecast_excludes_rows_after_as_of_and_uses_latest_eligible(self) -> None:
        target = dt.date(2026, 5, 1)
        forecasts = [
            bop_forecast(1, prediction_id=1, article_id=1, published_date=dt.date(2026, 4, 20), target_start_date=target, low=10, high=20),
            bop_forecast(1, prediction_id=2, article_id=2, published_date=dt.date(2026, 4, 29), target_start_date=target, low=30, high=50),
            bop_forecast(1, prediction_id=3, article_id=3, published_date=dt.date(2026, 5, 1), target_start_date=target, low=90, high=100),
        ]

        selected = recreate.latest_forecast(
            forecasts,
            as_of_date=dt.date(2026, 4, 30),
            movie_id=1,
            forecast_metric="domestic_opening_weekend",
            target_start_date=target,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(2, selected.prediction_id)
        self.assertEqual(40.0, selected.midpoint_usd)

    def test_same_weekend_bop_competitors_exclude_focal_movie(self) -> None:
        target = dt.date(2026, 5, 1)
        forecasts = [
            bop_forecast(1, prediction_id=1, target_start_date=target, low=100, high=100),
            bop_forecast(2, prediction_id=2, target_start_date=target, low=30, high=50),
            bop_forecast(3, prediction_id=3, target_start_date=target, low=10, high=20, metric="domestic_weekend"),
        ]

        features = recreate.bop_same_weekend_competitor_features(
            forecasts,
            focal_movie_id=1,
            target_start_date=target,
            as_of_date=dt.date(2026, 4, 30),
        )

        self.assertEqual(55.0, features["bop_same_weekend_competitor_total"])
        self.assertEqual(40.0, features["bop_same_weekend_competitor_top1"])
        self.assertEqual(2.0, features["bop_same_weekend_competitor_count"])

    def test_bop_q4_threshold_uses_training_rows_only(self) -> None:
        rows = []
        for idx, (year, midpoint) in enumerate(
            [(2022, 10.0), (2022, 20.0), (2023, 30.0), (2024, 40.0), (2025, 1_000.0)],
            start=1,
        ):
            row = {
                "movie_id": idx,
                "release_year": year,
                "bop_timing_day": -1,
                "bop_forecast_available": 1.0,
                "bop_forecast_midpoint": midpoint,
            }
            row.update({term: 1.0 for term in recreate.WIKI_TERMS})
            row["log1p_competitor_total_gross_lag1"] = 1.0
            rows.append(row)

        assigned = recreate.assign_bop_q4_proxy(rows, train_start_year=2022, train_end_year=2024)
        holdout = next(row for row in assigned if row["release_year"] == 2025)

        self.assertEqual(17.5, holdout["bop_q1_threshold"])
        self.assertEqual(25.0, holdout["bop_q2_threshold"])
        self.assertEqual(32.5, holdout["bop_q3_threshold"])
        self.assertEqual(32.5, holdout["bop_q4_threshold"])
        self.assertEqual(1.0, holdout["bop_q4_proxy"])
        self.assertEqual(0.0, holdout["bop_q1_proxy"])
        self.assertEqual(1.0, holdout["bop_estimate_bucket_under_15m"])

    def test_bop_fixed_estimate_buckets_use_midpoint_not_actuals(self) -> None:
        row = {
            "bop_forecast_available": 1.0,
            "bop_forecast_midpoint": 75_000_000.0,
        }

        recreate.add_bop_fixed_estimate_buckets(row)

        self.assertEqual(0.0, row["bop_estimate_bucket_30_60m"])
        self.assertEqual(1.0, row["bop_estimate_bucket_60_100m"])
        self.assertEqual(0.0, row["bop_estimate_bucket_100m_plus"])

    def test_time_split_scores_only_2025_2026_wiki_available_holdout_and_keeps_wiki_baseline(self) -> None:
        rows = []
        for idx, year in enumerate([2022, 2022, 2022, 2023, 2023, 2024, 2024, 2024, 2025, 2026, 2026], start=1):
            base = {
                "movie_id": idx,
                "title": f"Movie {idx}",
                "release_year": year,
                "opening_date": f"{year}-05-01",
                "timing_day": -7,
                "wiki_available": 0.0 if idx == 11 else 1.0,
                "opening_weekend_revenue_usd": float(10_000_000 + idx * 1_000_000),
                "target_log_opening_weekend": 16.0 + idx / 100.0,
            }
            base.update({term: float(idx) for term in recreate.WIKI_TERMS})
            base.update(
                {
                    "log1p_competitor_total_gross_lag1": float(idx),
                    "log1p_competitor_top1_gross_lag1": float(idx),
                    "competitor_count_lag1": 1.0,
                    "competitor_hhi_lag1": 1.0,
                    "log1p_competitor_total_gross_lag3": float(idx),
                    "log1p_competitor_total_gross_lag7": float(idx),
                    "competitor_total_gross_lag1": float(idx),
                    "competitor_top1_gross_lag1": float(idx),
                }
            )
            rows.append(base)

        predictions, metrics, _coefficients = recreate.evaluate_time_split(
            rows,
            timing_days=[-7],
            train_start_year=2022,
            train_end_year=2024,
            test_start_year=2025,
            test_end_year=2026,
        )

        scored_years = {int(row["release_year"]) for row in predictions if row["model"] == "wiki_log_all"}
        scored_movie_ids = {int(row["movie_id"]) for row in predictions if row["model"] == "wiki_log_all"}
        self.assertEqual({2025, 2026}, scored_years)
        self.assertNotIn(11, scored_movie_ids)
        self.assertIn("wiki_log_all", {row["model"] for row in metrics})
        self.assertIn("wiki_plus_competitor_total_lag1", {row["model"] for row in metrics})
        wiki_metric = next(row for row in metrics if row["model"] == "wiki_log_all")
        self.assertEqual(2, wiki_metric["holdout_n"])

    def test_bop_calibration_scores_only_2025_2026_holdout_with_available_forecasts(self) -> None:
        rows = []
        for idx, year in enumerate([2022, 2022, 2023, 2023, 2024, 2024, 2025, 2026, 2026], start=1):
            base = {
                "movie_id": idx,
                "title": f"Movie {idx}",
                "release_year": year,
                "opening_date": f"{year}-05-01",
                "timing_day": -1,
                "wiki_timing_day": -1,
                "bop_timing_day": -1,
                "wiki_available": 1.0,
                "bop_forecast_available": 0.0 if idx == 9 else 1.0,
                "bop_forecast_midpoint": float(10_000_000 + idx * 1_000_000),
                "log1p_bop_forecast_midpoint": recreate.log1p(10_000_000 + idx * 1_000_000),
                "opening_weekend_revenue_usd": float(12_000_000 + idx * 1_000_000),
                "target_log_opening_weekend": 16.0 + idx / 100.0,
                "bop_forecast_range_width_pct": 0.2,
                "bop_source_rank": 1.0,
                "bop_showtime_market_share_pct": 25.0,
            }
            base.update({term: float(idx) for term in recreate.WIKI_TERMS})
            rows.append(base)

        predictions, metrics, correlations = recreate.evaluate_bop_calibration(
            rows,
            timing_days=[-1],
            bop_timing_days=[-1],
            train_start_year=2022,
            train_end_year=2024,
            test_start_year=2025,
            test_end_year=2026,
        )

        raw_predictions = [row for row in predictions if row["model"] == "raw_midpoint"]
        self.assertEqual({2025, 2026}, {int(row["release_year"]) for row in raw_predictions})
        self.assertNotIn(9, {int(row["movie_id"]) for row in raw_predictions})
        self.assertTrue(any(row["model"] == "calibrated_midpoint" for row in metrics))
        self.assertEqual(2, correlations[0]["holdout_n"])

    def test_multisource_models_report_baselines_and_use_fallback_for_missing_bop(self) -> None:
        rows = []
        years = [2022] * 8 + [2023] * 5 + [2024] * 5 + [2025, 2025, 2026, 2026]
        missing_bop_ids = {21, 22}
        for idx, year in enumerate(years, start=1):
            bop_available = 0.0 if idx in missing_bop_ids else 1.0
            row = {
                "movie_id": idx,
                "title": f"Movie {idx}",
                "release_year": year,
                "release_run_id": idx * 10,
                "opening_date": f"{year}-05-01",
                "timing_day": -30,
                "wiki_timing_day": -30,
                "competition_timing_day": -1,
                "bop_timing_day": -1,
                "as_of_date": f"{year}-04-01",
                "wiki_as_of_date": f"{year}-04-01",
                "competition_as_of_date": f"{year}-04-30",
                "bop_as_of_date": f"{year}-04-30",
                "opening_theaters": 2500 + idx,
                "opening_day_gross_usd": 5_000_000 + idx,
                "opening_weekend_revenue_usd": 15_000_000 + idx * 900_000,
                "target_log_opening_weekend": recreate.log1p(15_000_000 + idx * 900_000),
                "wiki_available": 1.0,
                "V": 100.0 + idx,
                "U": 20.0 + idx,
                "R": 5.0 + idx,
                "E": 8.0 + idx,
                "log1p_V": recreate.log1p(100.0 + idx),
                "log1p_U": recreate.log1p(20.0 + idx),
                "log1p_R": recreate.log1p(5.0 + idx),
                "log1p_E": recreate.log1p(8.0 + idx),
                "log1p_opening_theaters": recreate.log1p(2500 + idx),
                "competitor_total_gross_lag1": 4_000_000.0 + idx,
                "competitor_total_gross_lag3": 9_000_000.0 + idx,
                "competitor_total_gross_lag7": 20_000_000.0 + idx,
                "log1p_competitor_total_gross_lag1": recreate.log1p(4_000_000.0 + idx),
                "log1p_competitor_total_gross_lag3": recreate.log1p(9_000_000.0 + idx),
                "log1p_competitor_total_gross_lag7": recreate.log1p(20_000_000.0 + idx),
                "bop_forecast_available": bop_available,
                "bop_forecast_midpoint": (14_000_000.0 + idx * 1_000_000) if bop_available else 0.0,
                "log1p_bop_forecast_midpoint": recreate.log1p(14_000_000.0 + idx * 1_000_000)
                if bop_available
                else 0.0,
                "bop_forecast_range_width_pct": 0.2,
            }
            rows.append(row)
        rows = recreate.assign_bop_q4_proxy(rows, train_start_year=2022, train_end_year=2024)

        predictions, metrics, coefficients = recreate.evaluate_multisource_models(
            rows,
            rows,
            timing_days=[-30],
            competition_timing_days=[-1],
            bop_timing_days=[-1],
            train_start_year=2022,
            train_end_year=2024,
            test_start_year=2025,
            test_end_year=2026,
        )

        model_names = {row["model"] for row in metrics}
        self.assertIn("raw_bop_midpoint", model_names)
        self.assertIn("calibrated_bop", model_names)
        self.assertIn("bop_plus_wiki_competition_buckets", model_names)
        full_raw = next(
            row
            for row in metrics
            if row["model"] == "raw_bop_midpoint"
            and row["population"] == "full_with_fallback"
            and row["status"] == "ok"
        )
        self.assertEqual(4, full_raw["holdout_n"])
        self.assertEqual(2, full_raw["bop_prediction_n"])
        self.assertEqual(2, full_raw["fallback_prediction_n"])
        self.assertEqual({2025, 2026}, {int(row["release_year"]) for row in predictions})
        self.assertFalse(any("opening_day_gross" in term for terms in recreate.MULTISOURCE_MODEL_TERMS.values() for term in terms))
        self.assertTrue(coefficients)

    def test_residual_lift_models_predict_residual_over_bop_midpoint(self) -> None:
        rows = []
        years = [2022] * 8 + [2023] * 5 + [2024] * 5 + [2025, 2025, 2026]
        for idx, year in enumerate(years, start=1):
            actual = 15_000_000 + idx * 800_000
            midpoint = 14_000_000 + idx * 850_000
            row = {
                "movie_id": idx,
                "title": f"Movie {idx}",
                "release_year": year,
                "opening_date": f"{year}-05-01",
                "timing_day": -30,
                "wiki_timing_day": -30,
                "competition_timing_day": -1,
                "bop_timing_day": -1,
                "opening_weekend_revenue_usd": actual,
                "target_log_opening_weekend": recreate.log1p(actual),
                "wiki_available": 1.0,
                "log1p_V": recreate.log1p(100 + idx),
                "log1p_competitor_total_gross_lag7": recreate.log1p(20_000_000 + idx),
                "bop_forecast_available": 1.0,
                "bop_forecast_midpoint": midpoint,
                "log1p_bop_forecast_midpoint": recreate.log1p(midpoint),
            }
            rows.append(row)
        rows = recreate.assign_bop_q4_proxy(rows, train_start_year=2022, train_end_year=2024)

        residual_rows = recreate.rows_with_bop_residual_target(rows)
        self.assertAlmostEqual(
            float(rows[0]["target_log_opening_weekend"]) - math.log(float(rows[0]["bop_forecast_midpoint"])),
            residual_rows[0]["target_log_bop_residual"],
        )

        predictions, metrics, coefficients = recreate.evaluate_residual_lift_models(
            rows,
            timing_days=[-30],
            competition_timing_days=[-1],
            bop_timing_days=[-1],
            train_start_year=2022,
            train_end_year=2024,
            test_start_year=2025,
            test_end_year=2026,
        )

        model_names = {row["model"] for row in metrics}
        self.assertIn("raw_bop_midpoint", model_names)
        self.assertIn("residual_buckets", model_names)
        self.assertIn("residual_quartile_buckets", model_names)
        self.assertIn("residual_fixed_estimate_buckets", model_names)
        self.assertIn("residual_compact_combined", model_names)
        self.assertEqual({2025, 2026}, {int(row["release_year"]) for row in predictions})
        self.assertFalse(any("opening_day_gross" in term for terms in recreate.RESIDUAL_LIFT_MODEL_TERMS.values() for term in terms))
        self.assertTrue(coefficients)

    def test_writer_creates_multisource_artifacts(self) -> None:
        metric_rows = [
            {
                "model": "raw_bop_midpoint",
                "population": "bop_covered",
                "prediction_transform": "midpoint",
                "timing_day": -30,
                "wiki_timing_day": -30,
                "competition_timing_day": -1,
                "bop_timing_day": -1,
                "train_start_year": 2022,
                "train_end_year": 2024,
                "test_start_year": 2025,
                "test_end_year": 2026,
                "train_n": 0,
                "holdout_n": 2,
                "bop_prediction_n": 2,
                "fallback_prediction_n": 0,
                "r2_log_revenue": 0.8,
                "r2_gross": 0.7,
                "mape_gross": 0.2,
                "rmse_log_revenue": 0.1,
                "mae_log_revenue": 0.1,
                "mean_actual_gross": 10_000_000,
                "mean_predicted_gross": 10_500_000,
                "smearing_factor": "",
                "status": "ok",
            }
        ]
        prediction_rows = [
            {
                "model": "raw_bop_midpoint",
                "population": "bop_covered",
                "prediction_transform": "midpoint",
                "prediction_source": "bop",
                "timing_day": -30,
                "wiki_timing_day": -30,
                "competition_timing_day": -1,
                "bop_timing_day": -1,
                "movie_id": 1,
                "title": "Movie 1",
                "release_year": 2025,
                "opening_date": "2025-05-01",
                "bop_forecast_available": 1.0,
                "bop_forecast_midpoint": 10_000_000,
                "bop_q1_proxy": 0.0,
                "bop_q4_proxy": 0.0,
                "actual_log_opening_weekend": 16.0,
                "predicted_log_opening_weekend": 16.0,
                "actual_opening_weekend_revenue_usd": 10_000_000,
                "predicted_opening_weekend_revenue_usd": 10_000_000,
                "absolute_percentage_error": 0.0,
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            recreate.write_multisource_outputs(
                out_dir,
                prediction_rows=prediction_rows,
                metric_rows=metric_rows,
                coefficient_rows=[],
            )

            for filename in [
                "multisource_opening_weekend_metrics.csv",
                "multisource_opening_weekend_predictions.csv",
                "multisource_opening_weekend_coefficients.csv",
                "multisource_opening_weekend_headline_metrics.csv",
                "figure_multisource_opening_weekend_metrics.svg",
            ]:
                self.assertTrue((out_dir / filename).exists(), filename)

    def test_writer_creates_residual_lift_artifacts(self) -> None:
        metric_rows = [
            {
                "model": "raw_bop_midpoint",
                "population": "bop_covered",
                "timing_day": -30,
                "wiki_timing_day": -30,
                "competition_timing_day": -1,
                "bop_timing_day": -1,
                "train_start_year": 2022,
                "train_end_year": 2024,
                "test_start_year": 2025,
                "test_end_year": 2026,
                "train_n": 0,
                "holdout_n": 2,
                "r2_log_revenue": 0.8,
                "r2_gross": 0.7,
                "mape_gross": 0.2,
                "rmse_log_revenue": 0.1,
                "mae_log_revenue": 0.1,
                "mean_actual_gross": 10_000_000,
                "mean_predicted_gross": 10_500_000,
                "mean_predicted_residual": 0.0,
                "status": "ok",
            }
        ]
        prediction_rows = [
            {
                "model": "raw_bop_midpoint",
                "population": "bop_covered",
                "timing_day": -30,
                "wiki_timing_day": -30,
                "competition_timing_day": -1,
                "bop_timing_day": -1,
                "movie_id": 1,
                "title": "Movie 1",
                "release_year": 2025,
                "opening_date": "2025-05-01",
                "bop_forecast_midpoint": 10_000_000,
                "bop_q1_proxy": 0.0,
                "bop_q4_proxy": 0.0,
                "actual_log_bop_residual": 0.0,
                "predicted_log_bop_residual": 0.0,
                "actual_log_opening_weekend": 16.0,
                "predicted_log_opening_weekend": 16.0,
                "actual_opening_weekend_revenue_usd": 10_000_000,
                "predicted_opening_weekend_revenue_usd": 10_000_000,
                "absolute_percentage_error": 0.0,
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            recreate.write_residual_lift_outputs(
                out_dir,
                prediction_rows=prediction_rows,
                metric_rows=metric_rows,
                coefficient_rows=[],
            )

            for filename in [
                "bop_residual_lift_metrics.csv",
                "bop_residual_lift_predictions.csv",
                "bop_residual_lift_coefficients.csv",
                "bop_residual_lift_headline_metrics.csv",
                "figure_bop_residual_lift_metrics.svg",
            ]:
                self.assertTrue((out_dir / filename).exists(), filename)

    def test_writer_creates_actual_competition_artifacts(self) -> None:
        panel_rows = []
        for idx, year in enumerate([2022, 2022, 2023, 2023, 2024, 2024, 2024, 2024, 2025, 2026], start=1):
            base = {
                "movie_id": idx,
                "title": f"Movie {idx}",
                "release_year": year,
                "release_run_id": idx * 10,
                "opening_date": f"{year}-05-01",
                "timing_day": -7,
                "as_of_date": f"{year}-04-24",
                "opening_theaters": 3000 + idx,
                "opening_day_gross_usd": 5_000_000 + idx,
                "opening_weekend_revenue_usd": 15_000_000 + idx * 1_000_000,
                "target_log_opening_weekend": 16.0 + idx / 100.0,
                "V": idx,
                "U": idx,
                "R": idx,
                "E": idx,
                "log1p_V": 1.0 + idx / 10,
                "log1p_U": 1.0,
                "log1p_R": 1.0,
                "log1p_E": 1.0,
                "log1p_opening_theaters": 8.0,
                "wiki_available": 1.0,
                "competitor_total_gross_lag1": 1_000_000.0,
                "competitor_top1_gross_lag1": 1_000_000.0,
                "competitor_count_lag1": 1.0,
                "competitor_hhi_lag1": 1.0,
                "log1p_competitor_total_gross_lag1": 14.0,
                "log1p_competitor_top1_gross_lag1": 14.0,
                "competitor_total_gross_lag3": 3_000_000.0,
                "competitor_top1_gross_lag3": 3_000_000.0,
                "competitor_count_lag3": 1.0,
                "competitor_hhi_lag3": 1.0,
                "log1p_competitor_total_gross_lag3": 15.0,
                "log1p_competitor_top1_gross_lag3": 15.0,
                "competitor_total_gross_lag7": 7_000_000.0,
                "competitor_top1_gross_lag7": 7_000_000.0,
                "competitor_count_lag7": 1.0,
                "competitor_hhi_lag7": 1.0,
                "log1p_competitor_total_gross_lag7": 16.0,
                "log1p_competitor_top1_gross_lag7": 16.0,
            }
            for month in range(2, 13):
                base[f"release_month_{month}"] = 1.0 if month == 5 else 0.0
            panel_rows.append(base)

        predictions, metrics, coefficients = recreate.evaluate_time_split(
            panel_rows,
            timing_days=[-7],
            train_start_year=2022,
            train_end_year=2024,
            test_start_year=2025,
            test_end_year=2026,
        )
        coverage = recreate.coverage_rows(panel_rows)

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            recreate.write_outputs(
                out_dir,
                panel_rows=panel_rows,
                prediction_rows=predictions,
                metric_rows=metrics,
                coefficient_rows=coefficients,
                coverage=coverage,
            )

            for filename in [
                "actual_competition_opening_weekend_feature_panel.csv",
                "actual_competition_opening_weekend_metrics.csv",
                "actual_competition_opening_weekend_predictions.csv",
                "actual_competition_opening_weekend_coefficients.csv",
                "actual_competition_opening_weekend_coverage.csv",
                "actual_competition_opening_weekend_headline_metrics.csv",
                "figure_actual_competition_opening_weekend_metrics.svg",
                "figure_actual_competition_opening_weekend_actual_vs_predicted.svg",
                "figure_actual_competition_gross_vs_residual.svg",
                "figure_actual_competition_opening_weekend_coverage.svg",
            ]:
                self.assertTrue((out_dir / filename).exists(), filename)

    def test_writer_creates_bop_estimate_artifacts(self) -> None:
        panel_rows = []
        for idx, year in enumerate([2022, 2022, 2023, 2023, 2024, 2024, 2024, 2024, 2025, 2026], start=1):
            base = {
                "movie_id": idx,
                "title": f"Movie {idx}",
                "release_year": year,
                "release_run_id": idx * 10,
                "opening_date": f"{year}-05-01",
                "timing_day": -1,
                "wiki_timing_day": -1,
                "competition_timing_day": -1,
                "bop_timing_day": -1,
                "as_of_date": f"{year}-04-30",
                "wiki_as_of_date": f"{year}-04-30",
                "competition_as_of_date": f"{year}-04-30",
                "bop_as_of_date": f"{year}-04-30",
                "opening_theaters": 3000 + idx,
                "opening_day_gross_usd": 5_000_000 + idx,
                "opening_weekend_revenue_usd": 15_000_000 + idx * 1_000_000,
                "target_log_opening_weekend": 16.0 + idx / 100.0,
                "V": idx,
                "U": idx,
                "R": idx,
                "E": idx,
                "log1p_V": 1.0 + idx / 10,
                "log1p_U": 1.0,
                "log1p_R": 1.0,
                "log1p_E": 1.0,
                "log1p_opening_theaters": 8.0,
                "wiki_available": 1.0,
                "competitor_total_gross_lag1": 1_000_000.0,
                "log1p_competitor_total_gross_lag1": 14.0,
                "bop_forecast_available": 1.0,
                "bop_prediction_id": idx,
                "bop_article_url": f"https://www.boxofficepro.com/{idx}/",
                "bop_forecast_published_date": f"{year}-04-30",
                "bop_forecast_midpoint": 12_000_000.0 + idx * 1_000_000,
                "log1p_bop_forecast_midpoint": recreate.log1p(12_000_000.0 + idx * 1_000_000),
                "bop_forecast_range_width_pct": 0.2,
                "bop_source_rank": 1.0,
                "bop_showtime_market_share_pct": 25.0,
                "bop_same_weekend_competitor_total": 5_000_000.0,
                "bop_same_weekend_competitor_top1": 5_000_000.0,
                "bop_same_weekend_competitor_count": 1.0,
                "bop_same_weekend_competitor_hhi": 1.0,
                "log1p_bop_same_weekend_competitor_total": 15.0,
                "log1p_bop_same_weekend_competitor_top1": 15.0,
            }
            for month in range(2, 13):
                base[f"release_month_{month}"] = 1.0 if month == 5 else 0.0
            panel_rows.append(base)
        panel_rows = recreate.assign_bop_q4_proxy(panel_rows, train_start_year=2022, train_end_year=2024)

        q4_predictions, q4_metrics, q4_coefficients = recreate.evaluate_bop_q4_split(
            panel_rows,
            timing_days=[-1],
            competition_timing_days=[-1],
            bop_timing_days=[-1],
            train_start_year=2022,
            train_end_year=2024,
            test_start_year=2025,
            test_end_year=2026,
        )
        calibration_predictions, calibration_metrics, correlations = recreate.evaluate_bop_calibration(
            panel_rows,
            timing_days=[-1],
            bop_timing_days=[-1],
            train_start_year=2022,
            train_end_year=2024,
            test_start_year=2025,
            test_end_year=2026,
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            recreate.write_bop_outputs(
                out_dir,
                panel_rows=panel_rows,
                q4_prediction_rows=q4_predictions,
                q4_metric_rows=q4_metrics,
                q4_coefficient_rows=q4_coefficients,
                calibration_prediction_rows=calibration_predictions,
                calibration_metric_rows=calibration_metrics,
                correlation_rows=correlations,
            )

            for filename in [
                "bop_estimate_feature_panel.csv",
                "bop_q4_interaction_metrics.csv",
                "bop_q4_interaction_predictions.csv",
                "bop_q4_interaction_coefficients.csv",
                "bop_estimate_calibration_metrics.csv",
                "bop_estimate_calibration_predictions.csv",
                "bop_estimate_correlation_summary.csv",
                "figure_bop_midpoint_vs_actual_opening_weekend.svg",
                "figure_bop_raw_midpoint_error_by_timing.svg",
                "figure_bop_calibrated_vs_raw_prediction_error.svg",
                "figure_bop_q4_interaction_model_comparison.svg",
            ]:
                self.assertTrue((out_dir / filename).exists(), filename)


if __name__ == "__main__":
    unittest.main()
