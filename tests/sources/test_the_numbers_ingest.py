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

    def test_parse_workers_default_to_sixteen(self) -> None:
        args = scraper.build_arg_parser().parse_args(["--dry-run"])

        self.assertEqual(16, args.parse_workers)

    def test_parse_workers_must_be_positive(self) -> None:
        args = scraper.build_arg_parser().parse_args(["--parse-workers", "0", "--dry-run"])

        with self.assertRaisesRegex(SystemExit, "--parse-workers must be at least 1"):
            scraper.validate_args(args)

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
                    SELECT mpa_rating, mpa_rating_details, genre
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
                self.assertEqual(chart_rows[0].movie_url, source_id_row[0])
                self.assertEqual("Sample Movie (2026)", source_id_row[1])
                self.assertEqual("matched", source_id_row[2])
                self.assertEqual("source_primary_key", source_id_row[3])
                self.assertEqual(1.0, source_id_row[4])
                self.assertEqual(0, issue_count)
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
