#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path


from pm_box_office.sources.the_numbers import ingest as scraper
from tests.postgres_test_utils import drop_isolated_postgres_schema, make_isolated_postgres_schema


DAILY_CHART_HTML = """
<html>
  <body>
    <h1>Daily Box Office for May 1, 2026</h1>
    <table>
      <tr>
        <th>Rank</th><th>Prev</th><th>Title</th><th>Gross</th>
        <th>Daily Change</th><th>Weekly Change</th><th>Theaters</th>
        <th>Theater Average</th><th>Total Gross</th><th>Days in Release</th>
      </tr>
      <tr>
        <td>1</td><td>2</td>
        <td><a href="/movie/Sample-Movie-(2026)#tab=box-office">Sample Movie</a></td>
        <td>$1,234,567</td><td>+12.3%</td><td>-5.0%</td>
        <td>3,000</td><td>$412</td><td>$10,000,000</td><td>8</td>
      </tr>
    </table>
  </body>
</html>
"""


MOVIE_PAGE_HTML = """
<html>
  <body>
    <h1>Sample Movie (2026)</h1>
    <p>OpusData ID: 123456</p>
    <table>
      <tr><td><b>MPA&nbsp;Rating:</b></td>
      <td>PG-13 for intense action and brief language.<br>(Rating bulletin 2885)</td></tr>
      <tr><td><b>Genre:</b></td><td>Action</td></tr>
      <tr><td><b>Production Budget:</b></td><td>$125,000,000</td></tr>
      <tr><td><b>Franchise:</b></td><td><a href="/movies/franchise/Sample">Sample Saga</a></td></tr>
    </table>
    <h2>Daily Box Office Performance</h2>
    <table>
      <tr>
        <th>Date</th><th>Rank</th><th>Gross</th><th>%YD</th><th>%LW</th>
        <th>Theaters</th><th>Per Theater</th><th>Total Gross</th><th>Days</th>
      </tr>
      <tr>
        <td>Apr 30, 2026</td><td>P</td><td>$100,000</td><td></td><td></td>
        <td>2,500</td><td>$40</td><td>$100,000</td><td></td>
      </tr>
      <tr>
        <td>May 1, 2026</td><td>1</td><td>$1,234,567</td><td>+12.3%</td><td>-5.0%</td>
        <td>3,000</td><td>$412</td><td>$10,000,000</td><td>8</td>
      </tr>
    </table>
  </body>
</html>
"""


class ScrapeTheNumbersTests(unittest.TestCase):
    def test_may_2026_dry_run_urls(self) -> None:
        days = scraper.date_range(dt.date(2026, 5, 1), dt.date(2026, 5, 31))
        urls = [scraper.daily_chart_url(day) for day in days]

        self.assertEqual(31, len(urls))
        self.assertEqual(
            "https://www.the-numbers.com/box-office-chart/daily/2026/05/01",
            urls[0],
        )
        self.assertEqual(
            "https://www.the-numbers.com/box-office-chart/daily/2026/05/31",
            urls[-1],
        )

    def test_parse_daily_chart(self) -> None:
        rows = scraper.parse_daily_chart(
            DAILY_CHART_HTML,
            chart_date=dt.date(2026, 5, 1),
            source_url="https://www.the-numbers.com/box-office-chart/daily/2026/05/01",
        )

        self.assertEqual(1, len(rows))
        row = rows[0]
        self.assertEqual("2026-05-01", row.chart_date)
        self.assertEqual("Sample Movie", row.title)
        self.assertEqual("https://www.the-numbers.com/movie/Sample-Movie-(2026)#tab=box-office", row.movie_url)
        self.assertEqual(1234567, row.gross_usd)
        self.assertEqual(3000, row.theaters)
        self.assertEqual(10000000, row.cumulative_gross_usd)

    def test_parse_movie_page(self) -> None:
        rows = scraper.parse_movie_page(
            MOVIE_PAGE_HTML,
            movie_url="https://www.the-numbers.com/movie/Sample-Movie-(2026)#tab=box-office",
            source_url="https://www.the-numbers.com/movie/Sample-Movie-(2026)#tab=box-office",
        )

        self.assertEqual(2, len(rows))
        self.assertEqual("Sample Movie (2026)", rows[0].title)
        self.assertEqual(2026, rows[0].release_year)
        self.assertEqual("123456", rows[0].opusdata_id)
        self.assertEqual(1, rows[0].is_preview)
        self.assertEqual("2026-05-01", rows[1].box_office_date)
        self.assertEqual(1234567, rows[1].gross_usd)

    def test_parse_movie_metadata(self) -> None:
        metadata = scraper.parse_movie_metadata(
            MOVIE_PAGE_HTML,
            movie_url="https://www.the-numbers.com/movie/Sample-Movie-(2026)#tab=box-office",
            source_url="https://www.the-numbers.com/movie/Sample-Movie-(2026)#tab=box-office",
        )

        self.assertEqual("Sample Movie (2026)", metadata.title)
        self.assertEqual(2026, metadata.release_year)
        self.assertEqual("123456", metadata.opusdata_id)
        self.assertEqual("PG-13", metadata.mpa_rating)
        self.assertEqual(
            "PG-13 for intense action and brief language. (Rating bulletin 2885)",
            metadata.mpa_rating_details,
        )
        self.assertEqual("Action", metadata.genre)
        self.assertEqual(125000000, metadata.production_budget_usd)
        self.assertEqual("Sample Saga", metadata.franchise)

    def test_movies_schema_keeps_release_date_for_schedule_sources(self) -> None:
        conn, schema = make_isolated_postgres_schema()
        scraper.initialize_database(conn)
        try:
            column = conn.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'movies'
                  AND column_name = 'release_date'
                """
            ).fetchone()

            self.assertEqual(("release_date",), tuple(column))
        finally:
            drop_isolated_postgres_schema(conn, schema)

    def test_the_numbers_movie_url_is_source_id_not_required_movie_key(self) -> None:
        conn, schema = make_isolated_postgres_schema()
        scraper.initialize_database(conn)
        try:
            nullable = conn.execute(
                """
                SELECT is_nullable
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'movies'
                  AND column_name = 'movie_url'
                """
            ).fetchone()[0]
            self.assertEqual("YES", nullable)

            row = scraper.MovieDailyRow(
                movie_url="https://www.the-numbers.com/movie/Sample-Movie-(2026)",
                title="Sample Movie (2026)",
                release_year=2026,
                opusdata_id="123456",
                box_office_date="2026-05-01",
                rank="1",
                gross_usd=100,
                percent_yesterday=None,
                percent_last_week=None,
                theaters=1000,
                per_theater_usd=1,
                cumulative_gross_usd=100,
                days_in_release=1,
                is_preview=0,
                source_url="https://www.the-numbers.com/movie/Sample-Movie-(2026)",
            )
            movie_id = scraper.upsert_movie(conn, row)

            source_rows = conn.execute(
                """
                SELECT source, source_movie_id, movie_id
                FROM movie_source_ids
                WHERE movie_id = %s
                ORDER BY source
                """,
                (movie_id,),
            ).fetchall()

            self.assertEqual(
                [
                    ("the_numbers", "https://www.the-numbers.com/movie/Sample-Movie-(2026)", movie_id),
                    ("the_numbers_opusdata", "123456", movie_id),
                ],
                [tuple(row) for row in source_rows],
            )
        finally:
            drop_isolated_postgres_schema(conn, schema)

    def test_parse_workers_default_to_sixteen(self) -> None:
        args = scraper.build_arg_parser().parse_args(["--dry-run"])

        self.assertEqual(16, args.parse_workers)

    def test_metadata_backfill_all_flag_is_parsed(self) -> None:
        args = scraper.build_arg_parser().parse_args(
            ["--metadata-backfill", "--metadata-backfill-all", "--offline"]
        )

        self.assertTrue(args.metadata_backfill)
        self.assertTrue(args.metadata_backfill_all)

    def test_metadata_backfill_all_selects_complete_rows(self) -> None:
        conn, schema = make_isolated_postgres_schema()
        scraper.initialize_database(conn)
        try:
            conn.execute(
                """
                INSERT INTO movies (movie_url, title, release_year)
                VALUES
                    ('https://www.the-numbers.com/movie/Complete-(2026)', 'Complete (2026)', 2026),
                    ('https://www.the-numbers.com/movie/Missing-(2026)', 'Missing (2026)', 2026)
                """
            )
            complete_movie_id = conn.execute(
                """
                SELECT movie_id
                FROM movies
                WHERE movie_url = 'https://www.the-numbers.com/movie/Complete-(2026)'
                """
            ).fetchone()[0]
            conn.execute(
                """
                INSERT INTO the_numbers_movie_metadata (
                    movie_id, movie_url, title, release_year, mpa_rating,
                    source_url, fetched_at, raw_cache_path
                ) VALUES (
                    %s,
                    'https://www.the-numbers.com/movie/Complete-(2026)',
                    'Complete (2026)',
                    2026,
                    'PG',
                    'https://www.the-numbers.com/movie/Complete-(2026)',
                    '2026-07-03T00:00:00+00:00',
                    'cache.html'
                )
                """,
                (complete_movie_id,),
            )

            missing_only = scraper.load_movie_urls_for_metadata_backfill(conn)
            all_movies = scraper.load_movie_urls_for_metadata_backfill(conn, include_complete=True)

            self.assertEqual(
                [("https://www.the-numbers.com/movie/Missing-(2026)", "Missing (2026)")],
                missing_only,
            )
            self.assertEqual(
                [
                    ("https://www.the-numbers.com/movie/Complete-(2026)", "Complete (2026)"),
                    ("https://www.the-numbers.com/movie/Missing-(2026)", "Missing (2026)"),
                ],
                all_movies,
            )
        finally:
            drop_isolated_postgres_schema(conn, schema)

    def test_parse_workers_must_be_positive(self) -> None:
        args = scraper.build_arg_parser().parse_args(["--parse-workers", "0", "--dry-run"])

        with self.assertRaisesRegex(SystemExit, "--parse-workers must be at least 1"):
            scraper.validate_args(args)

    def test_refresh_recent_days_only_marks_trailing_dates(self) -> None:
        args = scraper.build_arg_parser().parse_args(
            [
                "--start-date",
                "2026-06-27",
                "--end-date",
                "2026-07-03",
                "--refresh-recent-days",
                "2",
                "--dry-run",
            ]
        )

        self.assertFalse(scraper.should_refresh_recent_day(dt.date(2026, 6, 29), args))
        self.assertTrue(scraper.should_refresh_recent_day(dt.date(2026, 7, 2), args))
        self.assertTrue(scraper.should_refresh_recent_day(dt.date(2026, 7, 3), args))

    def test_cached_movie_pages_parse_concurrently_in_input_order(self) -> None:
        movie_urls = [
            "https://www.the-numbers.com/movie/Sample-Movie-(2026)#tab=box-office",
            "https://www.the-numbers.com/movie/Another-Sample-(2026)#tab=box-office",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            fetcher = scraper.HtmlFetcher(
                cache_dir,
                refresh=False,
                offline=True,
                delay_seconds=scraper.MIN_DELAY_SECONDS,
                user_agent="test-bot",
            )
            movie_items = []
            for index, movie_url in enumerate(movie_urls):
                cache_path = fetcher.cache_path(movie_url)
                cache_path.write_text(
                    MOVIE_PAGE_HTML.replace("Sample Movie (2026)", f"Sample Movie {index} (2026)"),
                    encoding="utf-8",
                )
                movie_items.append((movie_url, f"Sample Movie {index}", cache_path))

            results = scraper.parse_cached_movie_pages_concurrently(movie_items, workers=2)

        self.assertEqual(movie_urls, [result.movie_url for result in results])
        self.assertEqual("Sample Movie 0 (2026)", results[0].metadata.title)
        self.assertEqual("Sample Movie 1 (2026)", results[1].metadata.title)

    def test_postgres_import_is_idempotent_and_reconciles(self) -> None:
        chart_rows = scraper.parse_daily_chart(
            DAILY_CHART_HTML,
            chart_date=dt.date(2026, 5, 1),
            source_url="https://www.the-numbers.com/box-office-chart/daily/2026/05/01",
        )
        movie_rows = scraper.parse_movie_page(
            MOVIE_PAGE_HTML,
            movie_url=chart_rows[0].movie_url,
            source_url=chart_rows[0].movie_url,
        )
        metadata = scraper.parse_movie_metadata(
            MOVIE_PAGE_HTML,
            movie_url=chart_rows[0].movie_url,
            source_url=chart_rows[0].movie_url,
        )

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "fixture.html"
            cache_path.write_text("fixture", encoding="utf-8")
            conn, schema = make_isolated_postgres_schema()
            scraper.initialize_database(conn)
            conn.execute(
                """
                CREATE TABLE movie_source_ids (
                    movie_id BIGINT REFERENCES movies(movie_id),
                    source TEXT NOT NULL,
                    source_movie_id TEXT NOT NULL,
                    source_title TEXT,
                    match_status TEXT NOT NULL DEFAULT 'unmatched',
                    match_method TEXT,
                    match_score DOUBLE PRECISION,
                    matched_at TIMESTAMPTZ,
                    PRIMARY KEY (source, source_movie_id)
                )
                """
            )
            try:
                for _ in range(2):
                    scraper.insert_daily_chart_rows(
                        conn,
                        chart_rows,
                        fetched_at="2026-06-28T00:00:00+00:00",
                        raw_cache_path=cache_path,
                    )
                    scraper.insert_movie_daily_rows(
                        conn,
                        movie_rows,
                        fetched_at="2026-06-28T00:00:00+00:00",
                        raw_cache_path=cache_path,
                    )
                    scraper.upsert_movie_metadata(
                        conn,
                        metadata,
                        fetched_at="2026-06-28T00:00:00+00:00",
                        raw_cache_path=cache_path,
                    )
                    conn.commit()

                chart_count = conn.execute("SELECT COUNT(*) FROM daily_chart_pages").fetchone()[0]
                daily_count = conn.execute("SELECT COUNT(*) FROM daily_box_office").fetchone()[0]
                metadata_row = conn.execute(
                    """
                    SELECT mpa_rating, mpa_rating_details, genre, production_budget_usd, franchise
                    FROM the_numbers_movie_metadata
                    WHERE movie_url = %s
                    """,
                    (chart_rows[0].movie_url,),
                ).fetchone()
                source_id_row = conn.execute(
                    """
                    SELECT source_movie_id, source_title, match_status, match_method, match_score
                    FROM movie_source_ids
                    WHERE source = 'the_numbers'
                    """
                ).fetchone()
                issue_count = scraper.reconcile(conn, issue_source="test")

                self.assertEqual(1, chart_count)
                self.assertEqual(2, daily_count)
                self.assertEqual("PG-13", metadata_row[0])
                self.assertEqual(
                    "PG-13 for intense action and brief language. (Rating bulletin 2885)",
                    metadata_row[1],
                )
                self.assertEqual("Action", metadata_row[2])
                self.assertEqual(125000000, metadata_row[3])
                self.assertEqual("Sample Saga", metadata_row[4])
                self.assertEqual(chart_rows[0].movie_url, source_id_row[0])
                self.assertEqual("Sample Movie (2026)", source_id_row[1])
                self.assertEqual("matched", source_id_row[2])
                self.assertEqual("source_primary_key", source_id_row[3])
                self.assertEqual(1.0, source_id_row[4])
                self.assertEqual(0, issue_count)
            finally:
                drop_isolated_postgres_schema(conn, schema)

    def test_reconcile_can_be_scoped_to_current_run(self) -> None:
        chart_rows = scraper.parse_daily_chart(
            DAILY_CHART_HTML,
            chart_date=dt.date(2026, 5, 1),
            source_url="https://www.the-numbers.com/box-office-chart/daily/2026/05/01",
        )
        stale_chart_row = scraper.DailyChartRow(
            chart_date="2026-04-01",
            movie_url="https://www.the-numbers.com/movie/Missing-Movie-(2026)#tab=box-office",
            title="Missing Movie",
            rank="1",
            prev_rank=None,
            gross_usd=100,
            daily_change_pct=None,
            weekly_change_pct=None,
            theaters=10,
            per_theater_usd=10,
            cumulative_gross_usd=100,
            days_in_release=1,
            source_url="https://www.the-numbers.com/box-office-chart/daily/2026/04/01",
        )
        movie_rows = scraper.parse_movie_page(
            MOVIE_PAGE_HTML,
            movie_url=chart_rows[0].movie_url,
            source_url=chart_rows[0].movie_url,
        )

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "fixture.html"
            cache_path.write_text("fixture", encoding="utf-8")
            conn, schema = make_isolated_postgres_schema()
            scraper.initialize_database(conn)
            try:
                scraper.insert_daily_chart_rows(
                    conn,
                    [*chart_rows, stale_chart_row],
                    fetched_at="2026-06-28T00:00:00+00:00",
                    raw_cache_path=cache_path,
                )
                scraper.insert_movie_daily_rows(
                    conn,
                    movie_rows,
                    fetched_at="2026-06-28T00:00:00+00:00",
                    raw_cache_path=cache_path,
                )
                conn.commit()

                issue_count = scraper.reconcile(
                    conn,
                    issue_source="test",
                    chart_dates=["2026-05-01"],
                    movie_urls=[chart_rows[0].movie_url],
                )
                stored_issue_count = conn.execute(
                    "SELECT COUNT(*) FROM box_office_import_issues"
                ).fetchone()[0]

                self.assertEqual(0, issue_count)
                self.assertEqual(0, stored_issue_count)
            finally:
                drop_isolated_postgres_schema(conn, schema)

    def test_resume_helpers_load_recorded_chart_and_movie_state(self) -> None:
        chart_source_url = "https://www.the-numbers.com/box-office-chart/daily/2026/05/01"
        chart_rows = scraper.parse_daily_chart(
            DAILY_CHART_HTML,
            chart_date=dt.date(2026, 5, 1),
            source_url=chart_source_url,
        )
        movie_rows = scraper.parse_movie_page(
            MOVIE_PAGE_HTML,
            movie_url=chart_rows[0].movie_url,
            source_url=chart_rows[0].movie_url,
        )

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "fixture.html"
            cache_path.write_text("fixture", encoding="utf-8")
            conn, schema = make_isolated_postgres_schema()
            scraper.initialize_database(conn)
            try:
                scraper.record_raw_page(
                    conn,
                    source_url=chart_source_url,
                    source_page_type="daily_chart",
                    fetched_at="2026-06-28T00:00:00+00:00",
                    cache_path=cache_path,
                    html=DAILY_CHART_HTML,
                )
                scraper.insert_daily_chart_rows(
                    conn,
                    chart_rows,
                    fetched_at="2026-06-28T00:00:00+00:00",
                    raw_cache_path=cache_path,
                )

                self.assertTrue(
                    scraper.source_page_recorded(
                        conn,
                        source_url=chart_source_url,
                        source_page_type="daily_chart",
                    )
                )
                loaded_rows = scraper.load_daily_chart_rows(conn, source_url=chart_source_url)
                self.assertEqual(chart_rows, loaded_rows)
                self.assertFalse(scraper.movie_page_imported(conn, movie_url=chart_rows[0].movie_url))

                scraper.insert_movie_daily_rows(
                    conn,
                    movie_rows,
                    fetched_at="2026-06-28T00:00:00+00:00",
                    raw_cache_path=cache_path,
                )
                self.assertTrue(scraper.movie_page_imported(conn, movie_url=chart_rows[0].movie_url))
                self.assertFalse(scraper.movie_metadata_imported(conn, movie_url=chart_rows[0].movie_url))

                scraper.upsert_movie_metadata(
                    conn,
                    scraper.parse_movie_metadata(
                        MOVIE_PAGE_HTML,
                        movie_url=chart_rows[0].movie_url,
                        source_url=chart_rows[0].movie_url,
                    ),
                    fetched_at="2026-06-28T00:00:00+00:00",
                    raw_cache_path=cache_path,
                )
                self.assertTrue(scraper.movie_metadata_imported(conn, movie_url=chart_rows[0].movie_url))
            finally:
                drop_isolated_postgres_schema(conn, schema)


if __name__ == "__main__":
    unittest.main()
