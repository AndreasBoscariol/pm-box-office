from __future__ import annotations

import datetime as dt
import json
import unittest

from pm_box_office.sources.amc import collect
from pm_box_office.sources.amc import db
from pm_box_office.sources.amc.client import FetchResult
from pm_box_office.sources.amc.parsers import SeatFill, ShowtimeRecord
from pm_box_office.sources.amc.parsers import showtimes_url
from pm_box_office.sources.amc.sampling import fixed_theatre_sample, stratified_sample
from pm_box_office.sources.amc.scheduler import DEFAULT_OFFSETS_MINUTES, scheduled_snapshots
from pm_box_office.sources.amc.services import movie_service, sample_service, showtime_service
from pm_box_office.sources.amc.sitemap import AmcTheatre
from pm_box_office.sources.amc.sitemap import parse_theatre_sitemap
from pm_box_office.sources.amc.timezones import infer_us_timezone, parse_showtime_to_local_and_utc
from tests.postgres_test_utils import drop_isolated_postgres_schema, make_isolated_postgres_schema


SITEMAP_XML = """
<urlset xmlns:image="http://www.google.com/schemas/sitemap-image/1.1"
        xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://www.amctheatres.com/movie-theatres/new-york-city/amc-empire-25</loc>
    <PageMap xmlns="http://www.google.com/schemas/sitemap-pagemap/1.0">
      <DataObject type="content">
        <Attribute name="title">AMC Empire 25</Attribute>
      </DataObject>
      <DataObject type="theatre">
        <Attribute name="theatreId">123</Attribute>
        <Attribute name="addressLine1">234 W 42nd St</Attribute>
        <Attribute name="city">NEW YORK</Attribute>
        <Attribute name="state">NY</Attribute>
        <Attribute name="postalCode">10036</Attribute>
        <Attribute name="latitude">40.756</Attribute>
        <Attribute name="longitude">-73.988</Attribute>
      </DataObject>
    </PageMap>
  </url>
</urlset>
"""


def apollo_html(payload: dict[str, object]) -> str:
    return (
        "<html><body>"
        f"<script id=\"apollo-data\" type=\"application/json\">{json.dumps(payload)}</script>"
        "</body></html>"
    )


def showtime_payload(*, showtime_id: str, movie_id: str, movie_name: str, when: str) -> dict[str, object]:
    return {
        f"Movie:{movie_id}": {
            "__typename": "Movie",
            "id": movie_id,
            "name": movie_name,
        },
        f"Showtime:{showtime_id}": {
            "__typename": "Showtime",
            "showtimeId": showtime_id,
            "when": when,
            "movie": {"__ref": f"Movie:{movie_id}"},
        },
    }


class FakeShowtimeFetcher:
    def __init__(self, pages: dict[str, str]) -> None:
        self.pages = pages
        self.urls: list[str] = []

    def get_result(self, url: str) -> FetchResult:
        self.urls.append(url)
        return FetchResult(
            body=self.pages[url],
            source_url=url,
            fetched_at=dt.datetime(2026, 7, 1, 12, 0, tzinfo=dt.timezone.utc),
            cache_path=None,
            from_cache=False,
            status_code=200,
        )


def synthetic_theatre(index: int, *, state: str, timezone: str, screens: int) -> AmcTheatre:
    return AmcTheatre(
        amc_theatre_id=1000 + index,
        slug=f"amc-synthetic-{index}",
        theatre_url=f"https://www.amctheatres.com/movie-theatres/test/amc-synthetic-{index}",
        name=f"AMC Synthetic {screens}",
        address_line1="1 Sample Way",
        city="Sample",
        state=state,
        postal_code=f"{index:05d}",
        latitude=None,
        longitude=None,
        timezone=timezone,
        inferred_screen_count=screens,
    )


class AmcPipelineUnitTests(unittest.TestCase):
    def test_parse_theatre_sitemap_extracts_slug_location_timezone_and_screen_count(self) -> None:
        theatres = parse_theatre_sitemap(SITEMAP_XML)

        self.assertEqual(1, len(theatres))
        theatre = theatres[0]
        self.assertEqual(123, theatre.amc_theatre_id)
        self.assertEqual("amc-empire-25", theatre.slug)
        self.assertEqual("10036", theatre.postal_code)
        self.assertEqual(40.756, theatre.latitude)
        self.assertEqual(-73.988, theatre.longitude)
        self.assertEqual("America/New_York", theatre.timezone)
        self.assertEqual(25, theatre.inferred_screen_count)

    def test_timezone_inference_covers_us_theatre_zones(self) -> None:
        cases = [
            (40.756, -73.988, "NY", "America/New_York"),
            (41.881, -87.629, "IL", "America/Chicago"),
            (39.739, -104.990, "CO", "America/Denver"),
            (34.052, -118.244, "CA", "America/Los_Angeles"),
            (61.218, -149.900, "AK", "America/Anchorage"),
            (21.306, -157.858, "HI", "Pacific/Honolulu"),
        ]
        for latitude, longitude, state, expected in cases:
            with self.subTest(state=state):
                self.assertEqual(expected, infer_us_timezone(latitude, longitude, state))

    def test_showtime_conversion_handles_dst_boundaries(self) -> None:
        spring_local, spring_utc = parse_showtime_to_local_and_utc(
            "2026-03-08T19:00:00",
            "America/New_York",
        )
        fall_local, fall_utc = parse_showtime_to_local_and_utc(
            "2026-11-01T19:00:00",
            "America/New_York",
        )

        self.assertEqual("2026-03-08T19:00:00-04:00", spring_local.isoformat())
        self.assertEqual("2026-03-08T23:00:00+00:00", spring_utc.isoformat())
        self.assertEqual("2026-11-01T19:00:00-05:00", fall_local.isoformat())
        self.assertEqual("2026-11-02T00:00:00+00:00", fall_utc.isoformat())

    def test_snapshot_scheduling_uses_utc_due_times(self) -> None:
        local_start, utc_start = parse_showtime_to_local_and_utc(
            "2026-07-01T19:00:00",
            "America/Los_Angeles",
        )
        showtime = db.StoredShowtime(
            showtime_id="100",
            amc_theatre_id=1,
            theatre_slug="amc-sample-10",
            local_show_date="2026-07-01",
            local_start_at=local_start,
            utc_start_at=utc_start,
            timezone="America/Los_Angeles",
            amc_movie_id="movie-1",
            amc_movie_name="Sample One",
        )

        snapshots = scheduled_snapshots(showtime, offsets_minutes=DEFAULT_OFFSETS_MINUTES)

        self.assertEqual([360, 120, 30, 5], [snapshot.minutes_before_showtime for snapshot in snapshots])
        self.assertEqual("2026-07-01T20:00:00+00:00", snapshots[0].due_utc_at.isoformat())
        self.assertEqual("2026-07-01T13:00:00-07:00", snapshots[0].due_local_at.isoformat())

    def test_stratified_sample_is_deterministic(self) -> None:
        theatres = [
            db.StoredTheatre(
                amc_theatre_id=index,
                slug=f"amc-{index}",
                name=f"AMC {index}",
                state="CA" if index % 2 else "NY",
                postal_code="",
                latitude=None,
                longitude=None,
                timezone="America/Los_Angeles" if index % 2 else "America/New_York",
                inferred_screen_count=8 + index,
                observed_showtime_count=None,
                median_total_seats=None,
            )
            for index in range(1, 9)
        ]

        first = stratified_sample(theatres, sample_size=4, seed="2026-07-01")
        second = stratified_sample(theatres, sample_size=4, seed="2026-07-01")

        self.assertEqual(
            [item.theatre.amc_theatre_id for item in first],
            [item.theatre.amc_theatre_id for item in second],
        )
        self.assertEqual(4, len(first))

    def test_fixed_theatre_sample_uses_certainty_units_and_weights(self) -> None:
        theatres = [
            db.StoredTheatre(
                amc_theatre_id=index,
                slug=f"amc-{index}",
                name=f"AMC {index}",
                state="CA" if index % 2 else "NY",
                postal_code="",
                latitude=None,
                longitude=None,
                timezone="America/Los_Angeles" if index % 2 else "America/New_York",
                inferred_screen_count=8 + index,
                observed_showtime_count=100 - index,
                median_total_seats=None,
            )
            for index in range(1, 11)
        ]

        first = fixed_theatre_sample(theatres, sample_size=5, certainty_count=2, seed="sample")
        second = fixed_theatre_sample(theatres, sample_size=5, certainty_count=2, seed="sample")

        self.assertEqual(
            [item.theatre.amc_theatre_id for item in first],
            [item.theatre.amc_theatre_id for item in second],
        )
        self.assertEqual(5, len(first))
        certainty_ids = {item.theatre.amc_theatre_id for item in first if item.is_certainty}
        self.assertEqual({1, 2}, certainty_ids)
        self.assertTrue(all(item.inclusion_probability > 0 for item in first))
        self.assertTrue(all(item.analysis_weight >= 1 for item in first))

    def test_movie_options_group_showtimes_by_movie(self) -> None:
        rows = [
            ShowtimeRecord(
                theatre_slug="amc-empire-25",
                date="2026-07-01",
                showtime_id="100",
                when="2026-07-01T13:00:00-04:00",
                movie_name="Big Movie",
                movie_id="movie-big",
                showtime_url="https://www.amctheatres.com/showtimes/100",
                attribute_names="IMAX|Reserved Seating",
            ),
            ShowtimeRecord(
                theatre_slug="amc-empire-25",
                date="2026-07-01",
                showtime_id="101",
                when="2026-07-01T19:00:00-04:00",
                movie_name="Big Movie",
                movie_id="movie-big",
                showtime_url="https://www.amctheatres.com/showtimes/101",
                attribute_names="Dolby Cinema at AMC|Reserved Seating",
            ),
            ShowtimeRecord(
                theatre_slug="amc-empire-25",
                date="2026-07-01",
                showtime_id="102",
                when="2026-07-01T21:00:00-04:00",
                movie_name="Small Movie",
                movie_id="movie-small",
                showtime_url="https://www.amctheatres.com/showtimes/102",
                attribute_names="",
            ),
        ]

        options = collect.movie_options_from_showtimes(rows)
        rendered = collect.format_movie_options(options)
        selected = collect.choose_movie_option(options, selection=1)

        self.assertEqual(["Big Movie", "Small Movie"], [option.amc_movie_name for option in options])
        self.assertEqual(2, options[0].showtime_count)
        self.assertIn("1. Big Movie", rendered)
        self.assertIn("AMC movie id: movie-big", rendered)
        self.assertEqual("movie-big", selected.amc_movie_id)


class AmcPipelinePostgresTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn, self.schema = make_isolated_postgres_schema()
        db.initialize_amc_database(self.conn)
        self.conn.commit()

    def tearDown(self) -> None:
        drop_isolated_postgres_schema(self.conn, self.schema)

    def test_amc_upserts_are_idempotent(self) -> None:
        theatre = parse_theatre_sitemap(SITEMAP_XML)[0]
        self.assertEqual(1, db.upsert_theatres(self.conn, [theatre]))
        self.assertEqual(1, db.upsert_theatres(self.conn, [theatre]))

        stored_theatre = db.select_active_theatres(self.conn)[0]
        showtime = ShowtimeRecord(
            theatre_slug=stored_theatre.slug,
            date="2026-07-01",
            showtime_id="100",
            when="2026-07-01T19:00:00-04:00",
            movie_name="Sample One",
            movie_id="movie-1",
            showtime_url="https://www.amctheatres.com/showtimes/100",
            attribute_names="IMAX",
        )
        self.assertEqual(1, db.upsert_showtimes(self.conn, theatre=stored_theatre, showtimes=[showtime]))
        self.assertEqual(1, db.upsert_showtimes(self.conn, theatre=stored_theatre, showtimes=[showtime]))

        stored_showtime = db.select_showtimes_for_target(
            self.conn,
            target_date="2026-07-01",
            target_amc_movie_id="movie-1",
            target_amc_movie_name=None,
        )[0]
        fill = SeatFill(
            theatre_slug=stored_theatre.slug,
            date="2026-07-01",
            showtime_id="100",
            showtime_url="https://www.amctheatres.com/showtimes/100",
            total_seats=100,
            available_seats=80,
            filled_or_unavailable_seats=20,
            fill_rate=0.2,
        )
        db.upsert_seat_snapshot(
            self.conn,
            showtime=stored_showtime,
            seat_fill=fill,
            snapshot_utc_at=dt.datetime(2026, 7, 1, 22, 30, tzinfo=dt.timezone.utc),
            minutes_before_showtime=30,
        )
        db.upsert_seat_snapshot(
            self.conn,
            showtime=stored_showtime,
            seat_fill=fill,
            snapshot_utc_at=dt.datetime(2026, 7, 1, 22, 30, tzinfo=dt.timezone.utc),
            minutes_before_showtime=30,
        )
        self.conn.commit()

        self.assertEqual(
            1,
            self.conn.execute("SELECT COUNT(*) FROM amc_theatres").fetchone()[0],
        )
        self.assertEqual(
            1,
            self.conn.execute("SELECT COUNT(*) FROM amc_showtimes").fetchone()[0],
        )
        self.assertEqual(
            1,
            self.conn.execute("SELECT COUNT(*) FROM amc_seat_snapshots").fetchone()[0],
        )

    def test_campaign_movie_selection_is_idempotent(self) -> None:
        db.upsert_amc_movie(
            self.conn,
            amc_movie_id="movie-1",
            amc_movie_name="Sample One",
        )
        campaign_id = db.ensure_campaign(self.conn, dt.date(2026, 7, 1))
        db.set_campaign_movie_selected(
            self.conn,
            campaign_id=campaign_id,
            amc_movie_id="movie-1",
            selected=True,
        )
        db.set_campaign_movie_selected(
            self.conn,
            campaign_id=campaign_id,
            amc_movie_id="movie-1",
            selected=True,
        )
        self.conn.commit()

        row = self.conn.execute(
            """
            SELECT COUNT(*), BOOL_OR(selected)
            FROM campaign_movies
            WHERE campaign_id = %s AND amc_movie_id = %s
            """,
            (campaign_id, "movie-1"),
        ).fetchone()
        self.assertEqual(1, row[0])
        self.assertTrue(row[1])

    def test_select_the_numbers_active_movies_uses_audience_active_chart_logic(self) -> None:
        theatre = parse_theatre_sitemap(SITEMAP_XML)[0]
        db.upsert_theatres(self.conn, [theatre])
        stored_theatre = db.select_active_theatres(self.conn)[0]
        db.upsert_showtimes(
            self.conn,
            theatre=stored_theatre,
            showtimes=[
                ShowtimeRecord(
                    theatre_slug=stored_theatre.slug,
                    date="2026-07-02",
                    showtime_id="100",
                    when="2026-07-02T19:00:00-04:00",
                    movie_name="Sample One",
                    movie_id="movie-1",
                    showtime_url="https://www.amctheatres.com/showtimes/100",
                    attribute_names="",
                ),
                ShowtimeRecord(
                    theatre_slug=stored_theatre.slug,
                    date="2026-07-02",
                    showtime_id="200",
                    when="2026-07-02T20:00:00-04:00",
                    movie_name="Old Movie",
                    movie_id="movie-2",
                    showtime_url="https://www.amctheatres.com/showtimes/200",
                    attribute_names="",
                ),
            ],
        )
        self.conn.execute("ALTER TABLE movies ADD COLUMN IF NOT EXISTS movie_url TEXT")
        self.conn.execute(
            """
            CREATE TABLE daily_chart_pages (
                chart_date TEXT NOT NULL,
                movie_url TEXT NOT NULL,
                title TEXT NOT NULL,
                gross_usd INTEGER,
                theaters INTEGER,
                days_in_release INTEGER
            )
            """
        )
        self.conn.execute(
            """
            INSERT INTO movies (movie_id, movie_url, title)
            VALUES
                (1, 'https://www.the-numbers.com/movie/Sample-One-(2026)', 'Sample One'),
                (2, 'https://www.the-numbers.com/movie/Old-Movie-(2025)', 'Old Movie'),
                (3, 'https://www.the-numbers.com/movie/Citizen-Kane-(1941)', 'Citizen Kane (Special Engagement, re-release)')
            """
        )
        self.conn.execute(
            """
            INSERT INTO daily_chart_pages (chart_date, movie_url, title, gross_usd, theaters, days_in_release)
            VALUES
                ('2026-06-30', 'https://www.the-numbers.com/movie/Sample-One-(2026)', 'Sample One', 100000, 2000, 10),
                ('2026-06-30', 'https://www.the-numbers.com/movie/Old-Movie-(2025)', 'Old Movie', 50000, 1000, 400),
                ('2026-06-30', 'https://www.the-numbers.com/movie/Citizen-Kane-(1941)', 'Citizen Kane (Special Engagement, re-release)', 10000, 200, 2)
            """
        )

        matches = movie_service.select_the_numbers_active_movies(
            self.conn,
            exhibition_date=dt.date(2026, 7, 2),
            lookback_days=7,
        )
        self.conn.commit()

        self.assertEqual(["movie-1"], [match.amc_movie_id for match in matches])
        selected_rows = self.conn.execute(
            """
            SELECT cm.amc_movie_id
            FROM campaign_movies cm
            JOIN collection_campaigns c ON c.campaign_id = cm.campaign_id
            WHERE c.exhibition_date = %s AND cm.selected
            """,
            (dt.date(2026, 7, 2),),
        ).fetchall()
        self.assertEqual(["movie-1"], [str(row[0]) for row in selected_rows])

    def test_create_seat_scan_tasks_supports_multiple_offsets(self) -> None:
        theatre = parse_theatre_sitemap(SITEMAP_XML)[0]
        db.upsert_theatres(self.conn, [theatre])
        stored_theatre = db.select_active_theatres_basic(self.conn)[0]
        db.upsert_showtimes(
            self.conn,
            theatre=stored_theatre,
            showtimes=[
                ShowtimeRecord(
                    theatre_slug=stored_theatre.slug,
                    date="2026-07-01",
                    showtime_id="100",
                    when="2026-07-01T19:00:00-04:00",
                    movie_name="Sample One",
                    movie_id="movie-1",
                    showtime_url="https://www.amctheatres.com/showtimes/100",
                    attribute_names="IMAX",
                )
            ],
        )
        campaign_id = db.ensure_campaign(self.conn, dt.date(2026, 7, 1))
        run_id = db.create_run(self.conn, campaign_id=campaign_id, run_type="seat_collection")
        showtime = db.select_showtimes_for_target(
            self.conn,
            target_date="2026-07-01",
            target_amc_movie_id="movie-1",
            target_amc_movie_name=None,
        )[0]

        task_count = db.create_seat_scan_tasks(
            self.conn,
            run_id=run_id,
            showtimes=[showtime],
            target_offsets_minutes=(120, 30, 5),
        )
        self.conn.commit()

        rows = self.conn.execute(
            """
            SELECT priority
            FROM collection_tasks
            WHERE run_id = %s
            ORDER BY priority DESC
            """,
            (run_id,),
        ).fetchall()
        self.assertEqual(3, task_count)
        self.assertEqual([120, 30, 5], [row[0] for row in rows])

    def test_collection_services_reuse_active_runs(self) -> None:
        theatre = parse_theatre_sitemap(SITEMAP_XML)[0]
        db.upsert_theatres(self.conn, [theatre])
        stored_theatre = db.select_active_theatres_basic(self.conn)[0]
        db.upsert_showtimes(
            self.conn,
            theatre=stored_theatre,
            showtimes=[
                ShowtimeRecord(
                    theatre_slug=stored_theatre.slug,
                    date="2026-07-01",
                    showtime_id="100",
                    when="2026-07-01T19:00:00-04:00",
                    movie_name="Sample One",
                    movie_id="movie-1",
                    showtime_url="https://www.amctheatres.com/showtimes/100",
                    attribute_names="IMAX",
                )
            ],
        )
        db.upsert_amc_movie(self.conn, amc_movie_id="movie-1", amc_movie_name="Sample One")
        campaign_id = db.ensure_campaign(self.conn, dt.date(2026, 7, 1))
        db.set_campaign_movie_selected(
            self.conn,
            campaign_id=campaign_id,
            amc_movie_id="movie-1",
            selected=True,
        )
        self.conn.commit()

        inventory_run_1, inventory_tasks_1 = showtime_service.create_inventory_run(
            self.conn,
            exhibition_date=dt.date(2026, 7, 1),
        )
        inventory_run_2, inventory_tasks_2 = showtime_service.create_inventory_run(
            self.conn,
            exhibition_date=dt.date(2026, 7, 1),
        )
        seat_run_1, seat_tasks_1 = movie_service.create_seat_collection_run(
            self.conn,
            exhibition_date=dt.date(2026, 7, 1),
        )
        seat_run_2, seat_tasks_2 = movie_service.create_seat_collection_run(
            self.conn,
            exhibition_date=dt.date(2026, 7, 1),
        )

        self.assertEqual(inventory_run_1, inventory_run_2)
        self.assertEqual(inventory_tasks_1, inventory_tasks_2)
        self.assertEqual(seat_run_1, seat_run_2)
        self.assertEqual(seat_tasks_1, seat_tasks_2)

    def test_persistent_theatre_sample_is_reused_and_weighted(self) -> None:
        theatres = [
            synthetic_theatre(1, state="CA", timezone="America/Los_Angeles", screens=30),
            synthetic_theatre(2, state="CA", timezone="America/Los_Angeles", screens=18),
            synthetic_theatre(3, state="NY", timezone="America/New_York", screens=24),
            synthetic_theatre(4, state="NY", timezone="America/New_York", screens=12),
            synthetic_theatre(5, state="TX", timezone="America/Chicago", screens=20),
            synthetic_theatre(6, state="TX", timezone="America/Chicago", screens=10),
        ]
        db.upsert_theatres(self.conn, theatres)
        for theatre in db.select_active_theatres_basic(self.conn):
            db.upsert_showtimes(
                self.conn,
                theatre=theatre,
                showtimes=[
                    ShowtimeRecord(
                        theatre_slug=theatre.slug,
                        date="2026-07-01",
                        showtime_id=f"{theatre.amc_theatre_id}",
                        when="2026-07-01T19:00:00",
                        movie_name="Sample One",
                        movie_id="movie-1",
                        showtime_url=f"https://www.amctheatres.com/showtimes/{theatre.amc_theatre_id}",
                        attribute_names="",
                    )
                ],
            )

        first = sample_service.ensure_default_theatre_sample(
            self.conn,
            sample_key="tiny",
            sample_size=3,
            certainty_count=1,
            seed="sample",
        )
        first_members = db.select_theatre_sample_members(self.conn, first.sample_set_id)
        second = sample_service.ensure_default_theatre_sample(
            self.conn,
            sample_key="tiny",
            sample_size=3,
            certainty_count=1,
            seed="different-seed-is-ignored-after-create",
        )
        second_members = db.select_theatre_sample_members(self.conn, second.sample_set_id)
        coverage = sample_service.sample_coverage(
            self.conn,
            sample_set=first,
            exhibition_date=dt.date(2026, 7, 1),
        )
        overlap = db.theatre_sample_showtime_overlap(
            self.conn,
            sample_set_id=first.sample_set_id,
            exhibition_date=dt.date(2026, 7, 1),
        )
        movies = movie_service.list_movies_for_date(
            self.conn,
            exhibition_date=dt.date(2026, 7, 1),
            sample_set_id=first.sample_set_id,
        )

        self.assertEqual(first.sample_set_id, second.sample_set_id)
        self.assertEqual(3, len(first_members))
        self.assertEqual(
            [member.amc_theatre_id for member in first_members],
            [member.amc_theatre_id for member in second_members],
        )
        self.assertEqual(1, sum(1 for member in first_members if member.is_certainty))
        self.assertTrue(all(member.analysis_weight >= 1 for member in first_members))
        self.assertEqual(6, coverage["full_showtimes"])
        self.assertEqual(3, coverage["sampled_showtimes"])
        self.assertEqual(3, coverage["missing_snapshots"])
        self.assertEqual(3, overlap["sampled_showtimes"])
        self.assertGreaterEqual(overlap["peak_tasks_per_minute"], 1)
        self.assertLessEqual(overlap["peak_tasks_per_minute"], 3)
        self.assertEqual(1, len(movies))
        self.assertEqual(6, movies[0].showtime_count)
        self.assertEqual(3, movies[0].sampled_showtime_count)
        self.assertEqual(3, movies[0].sampled_theatre_count)

    def test_seat_collection_run_uses_saved_theatre_sample(self) -> None:
        theatres = [
            synthetic_theatre(index, state="CA", timezone="America/Los_Angeles", screens=10 + index)
            for index in range(1, 7)
        ]
        db.upsert_theatres(self.conn, theatres)
        for theatre in db.select_active_theatres_basic(self.conn):
            db.upsert_showtimes(
                self.conn,
                theatre=theatre,
                showtimes=[
                    ShowtimeRecord(
                        theatre_slug=theatre.slug,
                        date="2026-07-01",
                        showtime_id=f"{theatre.amc_theatre_id}",
                        when="2026-07-01T19:00:00",
                        movie_name="Sample One",
                        movie_id="movie-1",
                        showtime_url=f"https://www.amctheatres.com/showtimes/{theatre.amc_theatre_id}",
                        attribute_names="",
                    )
                ],
            )
        campaign_id = db.ensure_campaign(self.conn, dt.date(2026, 7, 1))
        db.set_campaign_movie_selected(
            self.conn,
            campaign_id=campaign_id,
            amc_movie_id="movie-1",
            selected=True,
        )
        sample_set = sample_service.ensure_default_theatre_sample(
            self.conn,
            sample_key="tiny-seat-run",
            sample_size=3,
            certainty_count=1,
            seed="sample",
        )
        sampled_theatre_ids = {
            member.amc_theatre_id
            for member in db.select_theatre_sample_members(self.conn, sample_set.sample_set_id)
        }
        self.conn.commit()

        run_id, task_count = movie_service.create_seat_collection_run(
            self.conn,
            exhibition_date=dt.date(2026, 7, 1),
            sample_key="tiny-seat-run",
        )

        task_rows = self.conn.execute(
            """
            SELECT s.amc_theatre_id
            FROM collection_tasks t
            JOIN amc_showtimes s ON s.showtime_id = t.showtime_id
            WHERE t.run_id = %s
            ORDER BY s.amc_theatre_id
            """,
            (run_id,),
        ).fetchall()
        self.assertEqual(3, task_count)
        self.assertEqual(sampled_theatre_ids, {int(row[0]) for row in task_rows})

    def test_reset_collection_state_preserves_sample_and_inventory(self) -> None:
        theatres = [
            synthetic_theatre(index, state="CA", timezone="America/Los_Angeles", screens=10 + index)
            for index in range(1, 7)
        ]
        db.upsert_theatres(self.conn, theatres)
        for theatre in db.select_active_theatres_basic(self.conn):
            db.upsert_showtimes(
                self.conn,
                theatre=theatre,
                showtimes=[
                    ShowtimeRecord(
                        theatre_slug=theatre.slug,
                        date="2026-07-01",
                        showtime_id=f"{theatre.amc_theatre_id}",
                        when="2026-07-01T19:00:00",
                        movie_name="Sample One",
                        movie_id="movie-1",
                        showtime_url=f"https://www.amctheatres.com/showtimes/{theatre.amc_theatre_id}",
                        attribute_names="",
                    )
                ],
            )
        campaign_id = db.ensure_campaign(self.conn, dt.date(2026, 7, 1))
        db.set_campaign_movie_selected(
            self.conn,
            campaign_id=campaign_id,
            amc_movie_id="movie-1",
            selected=True,
        )
        sample_set = sample_service.ensure_default_theatre_sample(
            self.conn,
            sample_key="reset-sample",
            sample_size=3,
            certainty_count=1,
            seed="sample",
        )
        run_id, task_count = movie_service.create_seat_collection_run(
            self.conn,
            exhibition_date=dt.date(2026, 7, 1),
            sample_key="reset-sample",
        )
        showtime = db.select_showtime_by_id(self.conn, str(theatres[0].amc_theatre_id))
        self.assertIsNotNone(showtime)
        db.upsert_seat_snapshot(
            self.conn,
            showtime=showtime,
            seat_fill=SeatFill(
                theatre_slug=showtime.theatre_slug,
                date="2026-07-01",
                showtime_id=showtime.showtime_id,
                showtime_url=f"https://www.amctheatres.com/showtimes/{showtime.showtime_id}",
                total_seats=100,
                available_seats=80,
                filled_or_unavailable_seats=20,
                fill_rate=0.2,
            ),
            snapshot_utc_at=dt.datetime(2026, 7, 1, 23, 30, tzinfo=dt.timezone.utc),
            minutes_before_showtime=30,
        )
        self.conn.commit()

        counts = db.reset_collection_state(
            self.conn,
            exhibition_date=dt.date(2026, 7, 1),
        )

        self.assertEqual(task_count, counts["collection_tasks"])
        self.assertEqual(1, counts["collection_runs"])
        self.assertEqual(1, counts["campaign_movies"])
        self.assertEqual(1, counts["collection_campaigns"])
        self.assertEqual(1, counts["amc_seat_snapshots"])
        self.assertEqual(
            0,
            self.conn.execute("SELECT COUNT(*) FROM collection_tasks WHERE run_id = %s", (run_id,)).fetchone()[0],
        )
        self.assertEqual(6, self.conn.execute("SELECT COUNT(*) FROM amc_showtimes").fetchone()[0])
        self.assertEqual(6, self.conn.execute("SELECT COUNT(*) FROM amc_theatres").fetchone()[0])
        self.assertEqual(
            3,
            self.conn.execute(
                "SELECT COUNT(*) FROM amc_theatre_sample_members WHERE sample_set_id = %s",
                (sample_set.sample_set_id,),
            ).fetchone()[0],
        )
        self.assertEqual(0, self.conn.execute("SELECT COUNT(*) FROM amc_seat_snapshots").fetchone()[0])

    def test_reset_collection_state_can_preserve_seat_snapshots(self) -> None:
        theatre = synthetic_theatre(1, state="CA", timezone="America/Los_Angeles", screens=12)
        db.upsert_theatres(self.conn, [theatre])
        stored_theatre = db.select_active_theatres_basic(self.conn)[0]
        db.upsert_showtimes(
            self.conn,
            theatre=stored_theatre,
            showtimes=[
                ShowtimeRecord(
                    theatre_slug=stored_theatre.slug,
                    date="2026-07-01",
                    showtime_id="100",
                    when="2026-07-01T19:00:00",
                    movie_name="Sample One",
                    movie_id="movie-1",
                    showtime_url="https://www.amctheatres.com/showtimes/100",
                    attribute_names="",
                )
            ],
        )
        showtime = db.select_showtime_by_id(self.conn, "100")
        self.assertIsNotNone(showtime)
        db.upsert_seat_snapshot(
            self.conn,
            showtime=showtime,
            seat_fill=SeatFill(
                theatre_slug=showtime.theatre_slug,
                date="2026-07-01",
                showtime_id=showtime.showtime_id,
                showtime_url="https://www.amctheatres.com/showtimes/100",
                total_seats=100,
                available_seats=80,
                filled_or_unavailable_seats=20,
                fill_rate=0.2,
            ),
            snapshot_utc_at=dt.datetime(2026, 7, 1, 23, 30, tzinfo=dt.timezone.utc),
            minutes_before_showtime=30,
        )
        self.conn.commit()

        counts = db.reset_collection_state(
            self.conn,
            exhibition_date=dt.date(2026, 7, 1),
            clear_seat_snapshots=False,
        )

        self.assertEqual(0, counts["amc_seat_snapshots"])
        self.assertEqual(1, self.conn.execute("SELECT COUNT(*) FROM amc_seat_snapshots").fetchone()[0])

    def test_movie_day_blocks_view_uses_sample_weights_for_sampled_snapshots(self) -> None:
        theatres = [
            synthetic_theatre(index, state="CA", timezone="America/Los_Angeles", screens=10 + index)
            for index in range(1, 5)
        ]
        db.upsert_theatres(self.conn, theatres)
        for theatre in db.select_active_theatres_basic(self.conn):
            db.upsert_showtimes(
                self.conn,
                theatre=theatre,
                showtimes=[
                    ShowtimeRecord(
                        theatre_slug=theatre.slug,
                        date="2026-07-01",
                        showtime_id=f"{theatre.amc_theatre_id}",
                        when="2026-07-01T19:00:00",
                        movie_name="Sample One",
                        movie_id="movie-1",
                        showtime_url=f"https://www.amctheatres.com/showtimes/{theatre.amc_theatre_id}",
                        attribute_names="",
                    )
                ],
            )
        sample_set = sample_service.ensure_default_theatre_sample(
            self.conn,
            sample_key="weighted-view",
            sample_size=2,
            certainty_count=0,
            seed="sample",
        )
        sampled_ids = [
            member.amc_theatre_id
            for member in db.select_theatre_sample_members(self.conn, sample_set.sample_set_id)
        ]
        for theatre_id in sampled_ids:
            showtime = db.select_showtime_by_id(self.conn, str(theatre_id))
            self.assertIsNotNone(showtime)
            db.upsert_seat_snapshot(
                self.conn,
                showtime=showtime,
                seat_fill=SeatFill(
                    theatre_slug=showtime.theatre_slug,
                    date="2026-07-01",
                    showtime_id=showtime.showtime_id,
                    showtime_url=f"https://www.amctheatres.com/showtimes/{showtime.showtime_id}",
                    total_seats=100,
                    available_seats=80,
                    filled_or_unavailable_seats=20,
                    fill_rate=0.2,
                ),
                snapshot_utc_at=dt.datetime(2026, 7, 1, 23, 30, tzinfo=dt.timezone.utc),
                minutes_before_showtime=30,
            )
        self.conn.commit()

        row = self.conn.execute(
            """
            SELECT s3_occupied_proxy, c3_capacity, s3_snapshot_count, full_day_showtime_count
            FROM analytics.amc_movie_day_blocks_v1
            WHERE amc_movie_id = 'movie-1' AND exhibition_date = '2026-07-01'
            """
        ).fetchone()

        self.assertEqual(80, row[0])
        self.assertEqual(400, row[1])
        self.assertEqual(2, row[2])
        self.assertEqual(4, row[3])

    def test_collect_theatre_showtimes_uses_embedded_dates_to_cover_window(self) -> None:
        theatre = parse_theatre_sitemap(SITEMAP_XML)[0]
        db.upsert_theatres(self.conn, [theatre])
        stored_theatre = db.select_active_theatres_basic(self.conn)[0]
        start_date = dt.date(2026, 7, 1)
        first_page_payload = (
            showtime_payload(
                showtime_id="100",
                movie_id="movie-1",
                movie_name="Sample One",
                when="2026-07-01T19:00:00-04:00",
            )
            | showtime_payload(
                showtime_id="101",
                movie_id="movie-2",
                movie_name="Sample Two",
                when="2026-07-02T18:00:00-04:00",
            )
        )
        third_page_payload = showtime_payload(
            showtime_id="102",
            movie_id="movie-3",
            movie_name="Sample Three",
            when="2026-07-03T20:00:00-04:00",
        )
        fetcher = FakeShowtimeFetcher(
            {
                showtimes_url(start_date, stored_theatre.slug): apollo_html(first_page_payload),
                showtimes_url(start_date + dt.timedelta(days=2), stored_theatre.slug): apollo_html(third_page_payload),
            }
        )

        count = showtime_service.collect_theatre_showtimes(
            self.conn,
            fetcher,  # type: ignore[arg-type]
            theatre=stored_theatre,
            exhibition_date=start_date,
            inventory_days=3,
        )

        self.assertEqual(3, count)
        self.assertEqual(
            [
                showtimes_url(start_date, stored_theatre.slug),
                showtimes_url(start_date + dt.timedelta(days=2), stored_theatre.slug),
            ],
            fetcher.urls,
        )
        rows = self.conn.execute(
            """
            SELECT showtime_id, exhibition_date
            FROM amc_showtimes
            ORDER BY showtime_id
            """
        ).fetchall()
        self.assertEqual(
            [
                ("100", start_date),
                ("101", start_date + dt.timedelta(days=1)),
                ("102", start_date + dt.timedelta(days=2)),
            ],
            [(str(row[0]), row[1]) for row in rows],
        )

    def test_campaign_queue_health_counts_due_and_late_tasks(self) -> None:
        campaign_id = db.ensure_campaign(self.conn, dt.date(2026, 7, 1))
        run_id = db.create_run(self.conn, campaign_id=campaign_id, run_type="seat_collection")
        now = db.utc_now()
        self.conn.executemany(
            """
            INSERT INTO collection_tasks (
                run_id, task_type, scheduled_for, status, priority, available_after, completed_at
            ) VALUES (%s, 'collect_seat_snapshot', %s, %s, 5, %s, %s)
            """,
            [
                (run_id, now - dt.timedelta(minutes=3), "queued", now - dt.timedelta(minutes=3), None),
                (run_id, now - dt.timedelta(seconds=10), "queued", now - dt.timedelta(seconds=10), None),
                (run_id, now + dt.timedelta(minutes=30), "queued", now + dt.timedelta(minutes=30), None),
                (run_id, now, "running", now, None),
                (run_id, now, "succeeded", now, now),
            ],
        )
        self.conn.commit()

        health = db.campaign_queue_health(self.conn, campaign_id)

        self.assertEqual(3, health["queued"])
        self.assertEqual(1, health["running"])
        self.assertEqual(2, health["due_now"])
        self.assertGreaterEqual(health["late"], 1)
        self.assertEqual(1, health["succeeded_last_5m"])
        self.assertAlmostEqual(10.0, health["eta_minutes"])
        self.assertAlmostEqual(10.0, health["due_eta_minutes"])


if __name__ == "__main__":
    unittest.main()
