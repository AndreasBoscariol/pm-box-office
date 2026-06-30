from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pm_box_office.models import train as trainer
from tests.postgres_test_utils import drop_isolated_postgres_schema, make_isolated_postgres_schema


def analytics_deps_available() -> bool:
    try:
        trainer.require_training_dependencies()
    except SystemExit:
        return False
    return True


class AmcBoxOfficeTrainerUnitTests(unittest.TestCase):
    def test_cutoff_feature_columns_expand_by_block(self) -> None:
        self.assertEqual(
            [
                "log1p_s1",
                "o1",
                "days_since_release",
                "opening_day_flag",
                "full_day_showtime_count",
                "movie_theatre_count",
                "premium_format_share",
                "day_of_week",
            ],
            trainer.cutoff_feature_columns("3pm"),
        )
        self.assertEqual(
            [
                "log1p_s1",
                "log1p_s2",
                "log1p_s3",
                "log1p_s4",
                "o1",
                "o2",
                "o3",
                "o4",
                "days_since_release",
                "opening_day_flag",
                "full_day_showtime_count",
                "movie_theatre_count",
                "premium_format_share",
                "day_of_week",
            ],
            trainer.cutoff_feature_columns("midnight"),
        )

    def test_target_log_ratio(self) -> None:
        self.assertAlmostEqual(0.0, trainer.target_log_ratio(10_000_000, 10_000_000))
        self.assertAlmostEqual(0.0953101798, trainer.target_log_ratio(11_000_000, 10_000_000))
        with self.assertRaises(ValueError):
            trainer.target_log_ratio(0, 10_000_000)
        with self.assertRaises(ValueError):
            trainer.target_log_ratio(10_000_000, 0)


@unittest.skipUnless(analytics_deps_available(), "Install requirements.txt analytics dependencies to run.")
class AmcBoxOfficeTrainerSmokeTests(unittest.TestCase):
    def test_training_smoke_writes_artifacts_and_plots(self) -> None:
        modules = trainer.require_training_dependencies()
        pd = modules["pandas"]
        rows = []
        for index in range(8):
            estimate = 10_000_000 + index * 250_000
            actual = estimate * (1.02 + index * 0.005)
            rows.append(
                {
                    "movie_id": index + 1,
                    "title": f"Sample {index}",
                    "amc_movie_id": f"amc-{index}",
                    "exhibition_date": f"2026-07-{index + 1:02d}",
                    "initial_estimate_usd": estimate,
                    "official_daily_gross_usd": actual,
                    "s1": 1000 + index * 50,
                    "c1": 2500 + index * 80,
                    "o1": 0.40,
                    "s2": 1500 + index * 60,
                    "c2": 3000 + index * 90,
                    "o2": 0.50,
                    "s3": 2000 + index * 70,
                    "c3": 3500 + index * 100,
                    "o3": 0.57,
                    "s4": 1600 + index * 55,
                    "c4": 3100 + index * 85,
                    "o4": 0.52,
                    "full_day_showtime_count": 100 + index,
                    "movie_theatre_count": 40 + index,
                    "premium_format_share": 0.18,
                    "days_since_release": index,
                    "day_of_week": (index % 7) + 1,
                    "opening_day_flag": index == 0,
                    "snapshot_coverage": 1.0,
                    "late_snapshot_count": 0,
                    "failed_snapshot_count": 0,
                }
            )
        frame = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "model-run"
            trainer.train_models_from_frame(
                frame,
                out_dir=out_dir,
                folds=4,
                min_rows=4,
                modules=modules,
            )

            expected = {
                "model_3pm.joblib",
                "model_6pm.joblib",
                "model_9pm.joblib",
                "model_midnight.joblib",
                "training_rows.csv",
                "predictions.csv",
                "metrics.json",
                "feature_columns.json",
                "actual_vs_predicted_gross_by_cutoff.png",
                "actual_vs_predicted_gross_by_cutoff.svg",
                "residual_distribution_by_cutoff.png",
                "residual_distribution_by_cutoff.svg",
                "absolute_percentage_error_by_cutoff.png",
                "absolute_percentage_error_by_cutoff.svg",
                "cross_validated_metric_comparison.png",
                "cross_validated_metric_comparison.svg",
                "coefficient_magnitude_3pm.png",
                "coefficient_magnitude_3pm.svg",
                "coefficient_magnitude_6pm.png",
                "coefficient_magnitude_6pm.svg",
                "coefficient_magnitude_9pm.png",
                "coefficient_magnitude_9pm.svg",
                "coefficient_magnitude_midnight.png",
                "coefficient_magnitude_midnight.svg",
            }
            self.assertTrue(expected.issubset({path.name for path in out_dir.iterdir()}))
            metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(["3pm", "6pm", "9pm", "midnight"], [row["cutoff"] for row in metrics["cutoffs"]])


@unittest.skipUnless(analytics_deps_available(), "Install requirements.txt analytics dependencies to run.")
class AmcBoxOfficeTrainerPostgresTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn, self.schema = make_isolated_postgres_schema()
        self.conn.execute("CREATE SCHEMA analytics")
        self.conn.execute(
            """
            CREATE TABLE movies (
                movie_id BIGINT PRIMARY KEY,
                title TEXT NOT NULL,
                release_date DATE
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE movie_source_ids (
                movie_id BIGINT REFERENCES movies(movie_id),
                source TEXT NOT NULL,
                source_movie_id TEXT NOT NULL,
                source_title TEXT,
                match_status TEXT NOT NULL DEFAULT 'unmatched',
                matched_at TIMESTAMPTZ,
                PRIMARY KEY (source, source_movie_id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE movie_day_estimates (
                estimate_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                movie_id BIGINT NOT NULL REFERENCES movies(movie_id),
                exhibition_date DATE NOT NULL,
                source TEXT NOT NULL,
                estimate_usd NUMERIC(14,2) NOT NULL,
                published_at TIMESTAMPTZ,
                recorded_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                is_baseline BOOLEAN NOT NULL DEFAULT FALSE,
                notes TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE release_runs (
                release_run_id BIGINT PRIMARY KEY,
                movie_id BIGINT NOT NULL REFERENCES movies(movie_id),
                market TEXT NOT NULL,
                release_type TEXT,
                source TEXT NOT NULL,
                source_release_key TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE daily_box_office (
                daily_box_office_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                release_run_id BIGINT NOT NULL REFERENCES release_runs(release_run_id),
                box_office_date DATE NOT NULL,
                gross_usd INTEGER,
                is_preview INTEGER NOT NULL DEFAULT 0,
                source TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE OR REPLACE VIEW analytics.amc_movie_day_blocks_v1 AS
            SELECT *
            FROM (
                VALUES
                    ('amc-1'::text, '2026-07-03'::date, 1000.0, 2500.0, 0.40, 1500.0, 3000.0, 0.50,
                     2000.0, 3500.0, 0.57, 1600.0, 3100.0, 0.52, 100.0, 40.0, 0.18, 1.0, 0.0, 0.0),
                    ('amc-2'::text, '2026-07-04'::date, 1100.0, 2600.0, 0.42, 1550.0, 3050.0, 0.51,
                     2100.0, 3600.0, 0.58, 1650.0, 3150.0, 0.52, 102.0, 41.0, 0.19, 1.0, 0.0, 0.0)
            ) AS rows(
                amc_movie_id, exhibition_date, s1_occupied_proxy, c1_capacity, o1_occupancy,
                s2_occupied_proxy, c2_capacity, o2_occupancy, s3_occupied_proxy, c3_capacity,
                o3_occupancy, s4_occupied_proxy, c4_capacity, o4_occupancy, full_day_showtime_count,
                movie_theatre_count, premium_format_share, snapshot_coverage, late_snapshot_count,
                failed_snapshot_count
            )
            """
        )
        self.conn.execute("INSERT INTO movies (movie_id, title, release_date) VALUES (1, 'Sample One', '2026-07-03')")
        self.conn.execute("INSERT INTO movies (movie_id, title, release_date) VALUES (2, 'Sample Two', '2026-07-04')")
        self.conn.execute(
            """
            INSERT INTO movie_source_ids (movie_id, source, source_movie_id, source_title, match_status)
            VALUES
                (1, 'amc', 'amc-1', 'Sample One', 'manual'),
                (2, 'amc', 'amc-2', 'Sample Two', 'manual')
            """
        )
        self.conn.execute(
            """
            INSERT INTO movie_day_estimates (movie_id, exhibition_date, source, estimate_usd, is_baseline)
            VALUES
                (1, '2026-07-03', 'baseline', 10000000, TRUE),
                (2, '2026-07-04', 'baseline', 12000000, TRUE)
            """
        )
        self.conn.execute(
            """
            INSERT INTO release_runs (release_run_id, movie_id, market, release_type, source, source_release_key)
            VALUES
                (1, 1, 'US_CA', 'movie_page_full_run', 'the_numbers', 'tn-1'),
                (2, 2, 'US_CA', 'movie_page_full_run', 'the_numbers', 'tn-2')
            """
        )
        self.conn.execute(
            """
            INSERT INTO daily_box_office (release_run_id, box_office_date, gross_usd, is_preview, source)
            VALUES
                (1, '2026-07-03', 11000000, 0, 'the_numbers'),
                (2, '2026-07-04', 11400000, 0, 'the_numbers')
            """
        )
        self.conn.commit()

    def tearDown(self) -> None:
        drop_isolated_postgres_schema(self.conn, self.schema)

    def test_load_training_frame_uses_same_postgres_tables(self) -> None:
        modules = trainer.require_training_dependencies()
        frame = trainer.load_training_frame(self.conn, modules["pandas"])

        self.assertEqual(2, len(frame))
        self.assertEqual(["amc-1", "amc-2"], frame["amc_movie_id"].tolist())
        self.assertEqual([11_000_000.0, 11_400_000.0], frame["official_daily_gross_usd"].tolist())
        self.assertEqual([10_000_000.0, 12_000_000.0], frame["initial_estimate_usd"].tolist())


if __name__ == "__main__":
    unittest.main()
