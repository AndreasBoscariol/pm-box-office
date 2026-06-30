from __future__ import annotations

import datetime as dt
import unittest

from scripts.ingest.amc import collect
from scripts.ingest.amc import db
from scripts.ingest.amc.parsers import SeatFill, ShowtimeRecord
from scripts.ingest.amc.sampling import stratified_sample
from scripts.ingest.amc.scheduler import DEFAULT_OFFSETS_MINUTES, scheduled_snapshots
from scripts.ingest.amc.sitemap import parse_theatre_sitemap
from scripts.ingest.amc.timezones import infer_us_timezone, parse_showtime_to_local_and_utc
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

    def test_movie_target_upsert_is_idempotent(self) -> None:
        db.upsert_movie_target(
            self.conn,
            target_date="2026-07-01",
            amc_movie_id="movie-1",
            amc_movie_name="Sample One",
            source_theatre_id=123,
            source_theatre_slug="amc-empire-25",
            source_theatre_name="AMC Empire 25",
            notes="first save",
        )
        db.upsert_movie_target(
            self.conn,
            target_date="2026-07-01",
            amc_movie_id="movie-1",
            amc_movie_name="Sample One",
            source_theatre_id=123,
            source_theatre_slug="amc-empire-25",
            source_theatre_name="AMC Empire 25",
            notes="second save",
        )
        self.conn.commit()

        row = self.conn.execute(
            """
            SELECT COUNT(*), MAX(notes)
            FROM amc_movie_targets
            WHERE target_date = %s AND amc_movie_id = %s
            """,
            ("2026-07-01", "movie-1"),
        ).fetchone()
        self.assertEqual(1, row[0])
        self.assertEqual("second save", row[1])


if __name__ == "__main__":
    unittest.main()
