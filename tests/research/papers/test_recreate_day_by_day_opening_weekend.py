from __future__ import annotations

import datetime as dt
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

        self.assertEqual(10_000_000.0, panel[0]["competitor_total_gross_lag1"])
        self.assertEqual(15_000_000.0, panel[0]["competitor_total_gross_lag7"])

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
                "wiki_available": 1.0,
                "log1p_V": opening.log1p(100 + idx),
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
        self.assertIn("bop_residual_wiki_competition_snapshot", {row["model"] for row in metrics})
        self.assertTrue(coefficients)
        self.assertEqual({"50", "80", "90"}, {row["interval_level"] for row in intervals})
        self.assertEqual(
            {"empirical_quantile", "conformal_abs", "loo_conformal_abs"},
            {row["interval_method"] for row in intervals},
        )

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
            )

            for filename in [
                "day_by_day_feature_panel.csv",
                "day_by_day_forecast_snapshots.csv",
                "day_by_day_metrics_by_horizon.csv",
                "day_by_day_best_metrics_by_horizon.csv",
                "day_by_day_interval_coverage.csv",
                "day_by_day_coefficients.csv",
                "day_by_day_prediction_revisions.csv",
                "figure_forecast_fan_chart.svg",
                "figure_gross_r2_by_horizon.svg",
            ]:
                self.assertTrue((out_dir / filename).exists(), filename)


if __name__ == "__main__":
    unittest.main()
