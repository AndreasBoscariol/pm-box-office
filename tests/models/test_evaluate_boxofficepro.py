from __future__ import annotations

import io
import math
import unittest
from contextlib import redirect_stdout

from pm_box_office.models import evaluate_boxofficepro
from tests.postgres_test_utils import drop_isolated_postgres_schema, make_isolated_postgres_schema


class BoxofficeProEvaluationMetricTests(unittest.TestCase):
    def test_summary_metrics_include_midpoint_errors_correlation_and_interval_coverage(self) -> None:
        rows = [
            evaluate_boxofficepro.ForecastActualRow(
                prediction_id=1,
                article_id=1,
                article_url="https://www.boxofficepro.com/a/",
                article_title="Weekend Preview A",
                movie_id=1,
                movie_title="Movie A",
                source_movie_title="Movie A",
                forecast_metric="domestic_weekend",
                source_context="weekend_podium",
                target_start_date=evaluate_boxofficepro.dt.date(2026, 7, 3),
                target_end_date=evaluate_boxofficepro.dt.date(2026, 7, 5),
                range_low_usd=90.0,
                range_high_usd=110.0,
                midpoint_usd=100.0,
                actual_usd=100.0,
                signed_error_usd=0.0,
                absolute_error_usd=0.0,
                percentage_error=0.0,
                absolute_percentage_error=0.0,
                interval_hit=True,
            ),
            evaluate_boxofficepro.ForecastActualRow(
                prediction_id=2,
                article_id=2,
                article_url="https://www.boxofficepro.com/b/",
                article_title="Weekend Preview B",
                movie_id=2,
                movie_title="Movie B",
                source_movie_title="Movie B",
                forecast_metric="domestic_weekend",
                source_context="legacy_forecast_table",
                target_start_date=evaluate_boxofficepro.dt.date(2026, 7, 3),
                target_end_date=evaluate_boxofficepro.dt.date(2026, 7, 5),
                range_low_usd=160.0,
                range_high_usd=200.0,
                midpoint_usd=180.0,
                actual_usd=200.0,
                signed_error_usd=-20.0,
                absolute_error_usd=20.0,
                percentage_error=-0.1,
                absolute_percentage_error=0.1,
                interval_hit=True,
            ),
            evaluate_boxofficepro.ForecastActualRow(
                prediction_id=3,
                article_id=2,
                article_url="https://www.boxofficepro.com/b/",
                article_title="Weekend Preview B",
                movie_id=3,
                movie_title="Movie C",
                source_movie_title="Movie C",
                forecast_metric="domestic_opening_weekend",
                source_context="legacy_forecast_table",
                target_start_date=evaluate_boxofficepro.dt.date(2026, 7, 3),
                target_end_date=evaluate_boxofficepro.dt.date(2026, 7, 5),
                range_low_usd=280.0,
                range_high_usd=320.0,
                midpoint_usd=300.0,
                actual_usd=250.0,
                signed_error_usd=50.0,
                absolute_error_usd=50.0,
                percentage_error=0.2,
                absolute_percentage_error=0.2,
                interval_hit=False,
            ),
        ]

        summary = evaluate_boxofficepro.summarize_rows("overall", rows)

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(3, summary.row_count)
        self.assertEqual(3, summary.movie_count)
        self.assertEqual(2, summary.article_count)
        self.assertAlmostEqual(550 / 3, summary.mean_actual_usd)
        self.assertAlmostEqual(580 / 3, summary.mean_midpoint_usd)
        self.assertAlmostEqual(0.953820966476532, summary.pearson_correlation or 0.0)
        self.assertAlmostEqual(70 / 3, summary.mae_usd)
        self.assertAlmostEqual(math.sqrt((0 + 400 + 2500) / 3), summary.rmse_usd)
        self.assertAlmostEqual(0.1, summary.mape or 0.0)
        self.assertAlmostEqual(10.0, summary.mean_signed_error_usd)
        self.assertAlmostEqual(2 / 3, summary.interval_coverage)


class BoxofficeProEvaluationPostgresTests(unittest.TestCase):
    def test_fetch_rows_sums_complete_target_windows_and_excludes_incomplete_actuals(self) -> None:
        conn, schema = make_isolated_postgres_schema()
        try:
            seed_schema(conn)
            rows = evaluate_boxofficepro.fetch_forecast_actual_rows(conn)

            self.assertEqual(["Complete Movie", "Second Movie"], [row.source_movie_title for row in rows])
            complete = rows[0]
            self.assertEqual(60_000_000.0, complete.actual_usd)
            self.assertEqual(55_000_000.0, complete.midpoint_usd)
            self.assertEqual(-5_000_000.0, complete.signed_error_usd)
            self.assertEqual(5_000_000.0, complete.absolute_error_usd)
            self.assertAlmostEqual(-5_000_000 / 60_000_000, complete.percentage_error or 0.0)
            self.assertAlmostEqual(5_000_000 / 60_000_000, complete.absolute_percentage_error or 0.0)
            self.assertTrue(complete.interval_hit)

            filtered = evaluate_boxofficepro.fetch_forecast_actual_rows(
                conn,
                forecast_metric="domestic_opening_weekend",
                source_context="weekend_podium",
                min_match_score=0.9,
            )
            self.assertEqual(["Complete Movie"], [row.source_movie_title for row in filtered])
        finally:
            drop_isolated_postgres_schema(conn, schema)

    def test_main_prints_empty_sample_message(self) -> None:
        conn, schema = make_isolated_postgres_schema()
        try:
            seed_schema(conn)
            database_url = postgres_url_for_schema(schema)
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = evaluate_boxofficepro.main(
                    ["--database-url", database_url, "--forecast-metric", "not_a_metric"]
                )

            self.assertEqual(0, exit_code)
            self.assertIn("No eligible Boxoffice Pro forecast rows found", output.getvalue())
        finally:
            drop_isolated_postgres_schema(conn, schema)

    def test_cli_help_includes_database_url(self) -> None:
        parser = evaluate_boxofficepro.build_parser()

        self.assertIn("--database-url", parser.format_help())


def seed_schema(conn) -> None:  # type: ignore[no-untyped-def]
    conn.executescript(
        """
        CREATE TABLE movies (
            movie_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            movie_url TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            release_year INTEGER
        );
        CREATE TABLE release_runs (
            release_run_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            movie_id BIGINT NOT NULL REFERENCES movies(movie_id),
            market TEXT NOT NULL DEFAULT 'US_CA',
            release_type TEXT,
            source TEXT NOT NULL,
            source_release_key TEXT NOT NULL,
            UNIQUE(movie_id, market, source, source_release_key)
        );
        CREATE TABLE daily_box_office (
            daily_box_office_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            release_run_id BIGINT NOT NULL REFERENCES release_runs(release_run_id),
            box_office_date TEXT NOT NULL,
            market TEXT NOT NULL DEFAULT 'US_CA',
            gross_usd INTEGER,
            is_preview INTEGER NOT NULL DEFAULT 0,
            source TEXT NOT NULL
        );
        CREATE TABLE boxofficepro_articles (
            article_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            article_url TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL
        );
        CREATE TABLE boxofficepro_weekend_predictions (
            prediction_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            article_id BIGINT NOT NULL REFERENCES boxofficepro_articles(article_id),
            source_movie_title TEXT NOT NULL,
            forecast_metric TEXT NOT NULL,
            source_context TEXT NOT NULL,
            source_rank INTEGER,
            market TEXT NOT NULL,
            target_start_date DATE,
            target_end_date DATE,
            range_low_usd BIGINT NOT NULL,
            range_high_usd BIGINT NOT NULL,
            matched_movie_id BIGINT REFERENCES movies(movie_id),
            match_score DOUBLE PRECISION
        );

        INSERT INTO movies (movie_url, title, release_year)
        VALUES
            ('https://www.the-numbers.com/movie/Complete', 'Complete Movie (2026)', 2026),
            ('https://www.the-numbers.com/movie/Second', 'Second Movie (2026)', 2026),
            ('https://www.the-numbers.com/movie/Incomplete', 'Incomplete Movie (2026)', 2026);

        INSERT INTO release_runs (movie_id, market, release_type, source, source_release_key)
        SELECT movie_id, 'US_CA', 'movie_page_full_run', 'the_numbers', movie_url
        FROM movies;

        INSERT INTO daily_box_office (release_run_id, box_office_date, market, gross_usd, is_preview, source)
        SELECT rr.release_run_id, d.box_office_date, 'US_CA', d.gross_usd, 0, 'the_numbers'
        FROM release_runs rr
        JOIN movies m ON m.movie_id = rr.movie_id
        JOIN (
            VALUES
                ('Complete Movie (2026)', '2026-07-03', 10000000),
                ('Complete Movie (2026)', '2026-07-04', 20000000),
                ('Complete Movie (2026)', '2026-07-05', 30000000),
                ('Second Movie (2026)', '2026-07-03', 20000000),
                ('Second Movie (2026)', '2026-07-04', 30000000),
                ('Second Movie (2026)', '2026-07-05', 40000000),
                ('Incomplete Movie (2026)', '2026-07-03', 5000000),
                ('Incomplete Movie (2026)', '2026-07-04', 6000000)
        ) AS d(title, box_office_date, gross_usd) ON d.title = m.title;

        INSERT INTO daily_box_office (release_run_id, box_office_date, market, gross_usd, is_preview, source)
        SELECT rr.release_run_id, '2026-07-02', 'US_CA', 999999, 1, 'the_numbers'
        FROM release_runs rr
        JOIN movies m ON m.movie_id = rr.movie_id
        WHERE m.title = 'Complete Movie (2026)';

        INSERT INTO boxofficepro_articles (article_url, title)
        VALUES
            ('https://www.boxofficepro.com/complete/', 'Weekend Preview Complete'),
            ('https://www.boxofficepro.com/second/', 'Weekend Preview Second');

        INSERT INTO boxofficepro_weekend_predictions (
            article_id, source_movie_title, forecast_metric, source_context, source_rank,
            market, target_start_date, target_end_date, range_low_usd, range_high_usd,
            matched_movie_id, match_score
        )
        SELECT a.article_id, 'Complete Movie', 'domestic_opening_weekend', 'weekend_podium', 1,
               'US_CA', '2026-07-03', '2026-07-05', 50000000, 60000000, m.movie_id, 0.98
        FROM boxofficepro_articles a, movies m
        WHERE a.article_url = 'https://www.boxofficepro.com/complete/'
          AND m.title = 'Complete Movie (2026)';

        INSERT INTO boxofficepro_weekend_predictions (
            article_id, source_movie_title, forecast_metric, source_context, source_rank,
            market, target_start_date, target_end_date, range_low_usd, range_high_usd,
            matched_movie_id, match_score
        )
        SELECT a.article_id, 'Second Movie', 'domestic_weekend', 'legacy_forecast_table', 2,
               'US_CA', '2026-07-03', '2026-07-05', 70000000, 80000000, m.movie_id, 0.88
        FROM boxofficepro_articles a, movies m
        WHERE a.article_url = 'https://www.boxofficepro.com/second/'
          AND m.title = 'Second Movie (2026)';

        INSERT INTO boxofficepro_weekend_predictions (
            article_id, source_movie_title, forecast_metric, source_context, source_rank,
            market, target_start_date, target_end_date, range_low_usd, range_high_usd,
            matched_movie_id, match_score
        )
        SELECT a.article_id, 'Incomplete Movie', 'domestic_opening_weekend', 'weekend_podium', 3,
               'US_CA', '2026-07-03', '2026-07-05', 10000000, 12000000, m.movie_id, 0.99
        FROM boxofficepro_articles a, movies m
        WHERE a.article_url = 'https://www.boxofficepro.com/complete/'
          AND m.title = 'Incomplete Movie (2026)';
        """
    )
    conn.commit()


def postgres_url_for_schema(schema: str) -> str:
    import os

    from pm_box_office.db.connection import database_url_from_env
    from tests.postgres_test_utils import url_with_search_path

    database_url = os.environ.get("TEST_DATABASE_URL") or database_url_from_env()
    if database_url is None:
        raise unittest.SkipTest("Set TEST_DATABASE_URL or DATABASE_URL to run PostgreSQL integration tests.")
    return url_with_search_path(database_url, schema)


if __name__ == "__main__":
    unittest.main()
