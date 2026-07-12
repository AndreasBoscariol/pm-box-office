from __future__ import annotations

import datetime as dt
import json
import os
import unittest
from unittest.mock import Mock, patch

from pm_box_office.sources.amc import collect
from pm_box_office.sources.amc import db
from pm_box_office.sources.amc.client import FetchResult
from pm_box_office.sources.amc.parsers import SeatFill, ShowtimeRecord
from pm_box_office.sources.amc.parsers import showtimes_url
from pm_box_office.sources.amc.sampling import fixed_theatre_sample, stratified_sample
from pm_box_office.sources.amc.scheduler import DEFAULT_OFFSETS_MINUTES, scheduled_snapshots
from pm_box_office.sources.amc.services import movie_service, sample_service, seat_service, showtime_service
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
        self.refresh_values: list[bool | None] = []

    def get_result(self, url: str, *, refresh: bool | None = None) -> FetchResult:
        self.urls.append(url)
        self.refresh_values.append(refresh)
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

    def test_collect_snapshot_enqueues_forecast_refresh_after_seat_write(self) -> None:
        observed = dt.datetime(2026, 7, 10, 20, 55, tzinfo=dt.timezone.utc)
        showtime = db.StoredShowtime(
            showtime_id="100",
            amc_theatre_id=1,
            theatre_slug="amc-sample-10",
            local_show_date="2026-07-10",
            local_start_at=observed,
            utc_start_at=observed + dt.timedelta(minutes=5),
            timezone="UTC",
            amc_movie_id="movie-1",
            amc_movie_name="Sample One",
        )
        fill = SeatFill(
            theatre_slug="amc-sample-10",
            date="2026-07-10",
            showtime_id="100",
            showtime_url="https://www.amctheatres.com/showtimes/100",
            total_seats=100,
            available_seats=60,
            filled_or_unavailable_seats=40,
            fill_rate=0.4,
            raw_cache_path="cache/seats/100.html",
        )

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(seat_service, "fetch_seat_fill", return_value=fill),
            patch.object(seat_service.db, "upsert_seat_snapshot") as upsert,
            patch.object(seat_service, "record_ingest_event") as record_event,
            patch.object(seat_service.refresh_queue, "enqueue_from_amc_showtime", return_value=1) as enqueue,
        ):
            seat_service.collect_snapshot(Mock(), Mock(), showtime=showtime, target_offset_minutes=5, observed_at=observed)

        upsert.assert_called_once()
        record_event.assert_called_once()
        enqueue.assert_called_once_with(
            upsert.call_args.args[0],
            showtime_id="100",
            model_version="latest",
            debounce_seconds=20,
            source_updated_at=observed,
        )

    def test_smooth_seat_scan_schedule_spreads_same_minute_bursts(self) -> None:
        start = dt.datetime(2026, 7, 1, 23, 0, tzinfo=dt.timezone.utc)
        showtimes = [
            db.StoredShowtime(
                showtime_id=f"{index}",
                amc_theatre_id=index,
                theatre_slug=f"amc-{index}",
                local_show_date="2026-07-01",
                local_start_at=start,
                utc_start_at=start,
                timezone="UTC",
                amc_movie_id="movie-1",
                amc_movie_name="Sample One",
            )
            for index in range(30)
        ]

        rows = db.seat_scan_task_rows(
            showtimes,
            offsets=range(20, 0, -1),
            schedule_strategy="smooth_once",
        )
        loads: dict[dt.datetime, int] = {}
        for row in rows:
            loads[row["scheduled_for"]] = loads.get(row["scheduled_for"], 0) + 1

        self.assertEqual(30, len(rows))
        self.assertEqual(20, len(loads))
        self.assertLessEqual(max(loads.values()), 2)
        self.assertEqual(1, len([row for row in rows if row["showtime"].showtime_id == "0"]))

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

    def test_collect_parser_keeps_durable_collection_surface(self) -> None:
        parser = collect.build_parser()

        inventory_args = parser.parse_args(["create-inventory-run", "2026-07-01"])
        ingest_args = parser.parse_args(["--refresh", "ingest-theatres", "--offline"])

        self.assertEqual("create-inventory-run", inventory_args.command)
        self.assertEqual(dt.date(2026, 7, 1), inventory_args.target_date)
        self.assertEqual("ingest-theatres", ingest_args.command)
        self.assertTrue(ingest_args.refresh)
        self.assertTrue(ingest_args.offline)


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

    def test_utc_showtime_date_is_stored_as_theatre_local_business_date(self) -> None:
        theatre = AmcTheatre(
            amc_theatre_id=200,
            slug="amc-pacific-12",
            theatre_url="https://www.amctheatres.com/movie-theatres/los-angeles/amc-pacific-12",
            name="AMC Pacific 12",
            address_line1="",
            city="LOS ANGELES",
            state="CA",
            postal_code="90001",
            latitude=34.0,
            longitude=-118.2,
            timezone="America/Los_Angeles",
            inferred_screen_count=12,
        )
        db.upsert_theatres(self.conn, [theatre])
        stored_theatre = db.select_active_theatres(self.conn)[0]
        showtimes = [
            ShowtimeRecord(
                theatre_slug=stored_theatre.slug,
                date="2026-07-03",
                showtime_id="utc-evening",
                when="2026-07-03T00:00:00Z",
                movie_name="Sample One",
                movie_id="movie-1",
                showtime_url="https://www.amctheatres.com/showtimes/utc-evening",
                attribute_names="",
            ),
            ShowtimeRecord(
                theatre_slug=stored_theatre.slug,
                date="2026-07-04",
                showtime_id="late-night",
                when="2026-07-04T07:30:00Z",
                movie_name="Sample One",
                movie_id="movie-1",
                showtime_url="https://www.amctheatres.com/showtimes/late-night",
                attribute_names="",
            ),
        ]

        db.upsert_showtimes(self.conn, theatre=stored_theatre, showtimes=showtimes)
        rows = self.conn.execute(
            """
            SELECT showtime_id, exhibition_date, local_calendar_start_at
            FROM amc_showtimes
            ORDER BY showtime_id
            """
        ).fetchall()

        self.assertEqual(
            [
                ("late-night", dt.date(2026, 7, 3), dt.datetime(2026, 7, 4, 0, 30)),
                ("utc-evening", dt.date(2026, 7, 2), dt.datetime(2026, 7, 2, 17, 0)),
            ],
            [(str(row[0]), row[1], row[2]) for row in rows],
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
                (101, 'https://www.the-numbers.com/movie/Sample-One-(2026)', 'Sample One'),
                (102, 'https://www.the-numbers.com/movie/Old-Movie-(2025)', 'Old Movie'),
                (103, 'https://www.the-numbers.com/movie/Citizen-Kane-(1941)', 'Citizen Kane (Special Engagement, re-release)')
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

    def test_select_the_numbers_active_movies_includes_opening_weekend_movies(self) -> None:
        theatre = parse_theatre_sitemap(SITEMAP_XML)[0]
        db.upsert_theatres(self.conn, [theatre])
        stored_theatre = db.select_active_theatres(self.conn)[0]
        db.upsert_showtimes(
            self.conn,
            theatre=stored_theatre,
            showtimes=[
                ShowtimeRecord(
                    theatre_slug=stored_theatre.slug,
                    date="2026-07-10",
                    showtime_id="300",
                    when="2026-07-10T19:00:00-04:00",
                    movie_name="Opening Example",
                    movie_id="opening-example",
                    showtime_url="https://www.amctheatres.com/showtimes/300",
                    attribute_names="",
                )
            ],
        )
        self.conn.execute("ALTER TABLE movies ADD COLUMN IF NOT EXISTS release_date DATE")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS daily_chart_pages (chart_date TEXT, movie_url TEXT, title TEXT, gross_usd INTEGER, days_in_release INTEGER)"
        )
        self.conn.execute("CREATE TABLE IF NOT EXISTS movie_source_ids (source TEXT, source_movie_id TEXT, movie_id INTEGER)")
        self.conn.execute(
            "INSERT INTO movies (movie_id, title, release_date) VALUES (104, 'Opening Example', '2026-07-10')"
        )

        matches = movie_service.select_the_numbers_active_movies(
            self.conn,
            exhibition_date=dt.date(2026, 7, 10),
        )

        self.assertEqual(["opening-example"], [match.amc_movie_id for match in matches])

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
                    attribute_names="IMAX|Reserved Seating",
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
            target_offsets_minutes=(120, 30, 5, -15),
        )
        self.conn.commit()

        rows = self.conn.execute(
            """
            SELECT priority, target_offset_minutes, scheduled_for
            FROM collection_tasks
            WHERE run_id = %s
            ORDER BY target_offset_minutes DESC
            """,
            (run_id,),
        ).fetchall()
        self.assertEqual(4, task_count)
        self.assertEqual([120, 30, 5, -15], [row[1] for row in rows])
        self.assertEqual(showtime.utc_start_at + dt.timedelta(minutes=15), db.ensure_utc(rows[-1][2]))

    def test_create_seat_scan_tasks_allows_unmatched_movie_source_id(self) -> None:
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
                    attribute_names="Reserved Seating",
                )
            ],
        )
        self.conn.execute(
            """
            INSERT INTO movie_source_ids (movie_id, source, source_movie_id, source_title, match_status)
            VALUES (NULL, 'amc', 'movie-1', 'Sample One', 'unmatched')
            """
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
            target_offsets_minutes=(30,),
        )
        self.conn.commit()

        row = self.conn.execute(
            "SELECT movie_id FROM collection_tasks WHERE run_id = %s",
            (run_id,),
        ).fetchone()
        self.assertEqual(1, task_count)
        self.assertIsNone(row[0])

    def test_seat_collection_quality_view_uses_planned_task_grid(self) -> None:
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
                    attribute_names="IMAX|Reserved Seating",
                )
            ],
        )
        campaign_id = db.ensure_campaign(self.conn, dt.date(2026, 7, 1))
        sample_set = sample_service.ensure_default_theatre_sample(
            self.conn,
            sample_key="quality-grid",
            sample_size=1,
            certainty_count=0,
            seed="sample",
        )
        run_id = db.create_run(
            self.conn,
            campaign_id=campaign_id,
            run_type="seat_collection",
            sample_set_id=sample_set.sample_set_id,
            sample_key=sample_set.sample_key,
            target_offsets_minutes=(120, 30),
            schedule_strategy="all_offsets",
        )
        showtime = db.select_showtimes_for_target(
            self.conn,
            target_date="2026-07-01",
            target_amc_movie_id="movie-1",
            target_amc_movie_name=None,
        )[0]
        db.create_seat_scan_tasks(
            self.conn,
            run_id=run_id,
            showtimes=[showtime],
            target_offsets_minutes=(120, 30),
            schedule_strategy="all_offsets",
        )
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
            snapshot_utc_at=dt.datetime(2026, 7, 1, 22, 30, tzinfo=dt.timezone.utc),
            minutes_before_showtime=30,
        )
        self.conn.commit()

        task_offsets = self.conn.execute(
            """
            SELECT target_offset_minutes
            FROM collection_tasks
            WHERE run_id = %s
            ORDER BY target_offset_minutes DESC
            """,
            (run_id,),
        ).fetchall()
        quality = self.conn.execute(
            """
            SELECT
                expected_snapshots,
                observed_snapshots,
                snapshot_success_rate,
                observed_capacity,
                expected_known_capacity,
                capacity_coverage_rate
            FROM analytics.amc_seat_collection_quality_v1
            WHERE run_id = %s AND amc_movie_id = 'movie-1'
            """,
            (run_id,),
        ).fetchone()
        blocks = self.conn.execute(
            """
            SELECT snapshot_coverage, capacity_coverage
            FROM analytics.amc_movie_day_blocks_v1
            WHERE amc_movie_id = 'movie-1' AND exhibition_date = '2026-07-01'
            """
        ).fetchone()

        self.assertEqual([120, 30], [int(row[0]) for row in task_offsets])
        self.assertEqual(2, int(quality[0]))
        self.assertEqual(1, int(quality[1]))
        self.assertAlmostEqual(0.5, float(quality[2]))
        self.assertEqual(100, int(quality[3]))
        self.assertEqual(200, int(quality[4]))
        self.assertAlmostEqual(0.5, float(quality[5]))
        self.assertAlmostEqual(0.5, float(blocks[0]))
        self.assertAlmostEqual(0.5, float(blocks[1]))

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
                    attribute_names="IMAX|Reserved Seating",
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

    def test_seat_collection_includes_unselected_live_movies(self) -> None:
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
                    movie_name="Selected Movie",
                    movie_id="movie-1",
                    showtime_url="https://www.amctheatres.com/showtimes/100",
                    attribute_names="Reserved Seating",
                ),
                ShowtimeRecord(
                    theatre_slug=stored_theatre.slug,
                    date="2026-07-01",
                    showtime_id="200",
                    when="2026-07-01T20:00:00-04:00",
                    movie_name="Unselected Movie",
                    movie_id="movie-2",
                    showtime_url="https://www.amctheatres.com/showtimes/200",
                    attribute_names="Reserved Seating",
                ),
            ],
        )
        db.upsert_amc_movie(self.conn, amc_movie_id="movie-1", amc_movie_name="Selected Movie")
        db.upsert_amc_movie(self.conn, amc_movie_id="movie-2", amc_movie_name="Unselected Movie")
        campaign_id = db.ensure_campaign(self.conn, dt.date(2026, 7, 1))
        db.set_campaign_movie_selected(
            self.conn,
            campaign_id=campaign_id,
            amc_movie_id="movie-1",
            selected=True,
        )
        sample_service.ensure_default_theatre_sample(
            self.conn,
            sample_key="all-live-movies",
            sample_size=1,
            certainty_count=1,
            seed="all-live-movies",
        )
        self.conn.commit()

        run_id, task_count = movie_service.create_seat_collection_run(
            self.conn,
            exhibition_date=dt.date(2026, 7, 1),
            sample_key="all-live-movies",
        )

        movie_ids = {
            str(row[0])
            for row in self.conn.execute(
                """
                SELECT s.amc_movie_id
                FROM collection_tasks t
                JOIN amc_showtimes s ON s.showtime_id = t.showtime_id
                WHERE t.run_id = %s
                """,
                (run_id,),
            ).fetchall()
        }
        self.assertEqual(2, task_count)
        self.assertEqual({"movie-1", "movie-2"}, movie_ids)

    def test_create_inventory_run_force_refresh_cancels_active_run(self) -> None:
        theatre = parse_theatre_sitemap(SITEMAP_XML)[0]
        db.upsert_theatres(self.conn, [theatre])
        campaign_id = db.ensure_campaign(self.conn, dt.date(2026, 7, 1))
        old_run_id = db.create_run(self.conn, campaign_id=campaign_id, run_type="showtime_inventory")
        db.create_inventory_tasks(self.conn, run_id=old_run_id, theatres=db.select_active_theatres_basic(self.conn))
        self.conn.commit()

        new_run_id, task_count = showtime_service.create_inventory_run(
            self.conn,
            exhibition_date=dt.date(2026, 7, 1),
            force_refresh=True,
        )

        rows = self.conn.execute(
            """
            SELECT run_id, status, tasks_total, tasks_cancelled
            FROM collection_runs
            WHERE campaign_id = %s AND run_type = 'showtime_inventory'
            ORDER BY started_at, run_id
            """,
            (campaign_id,),
        ).fetchall()

        self.assertNotEqual(str(old_run_id), new_run_id)
        self.assertEqual(1, task_count)
        self.assertEqual("cancelled", rows[0][1])
        self.assertEqual(1, int(rows[0][3]))
        self.assertEqual(new_run_id, str(rows[1][0]))
        self.assertEqual("queued", rows[1][1])

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
                        attribute_names="Reserved Seating",
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

    def test_default_theatre_sample_uses_top_hybrid_universe_order(self) -> None:
        theatres = [
            AmcTheatre(
                amc_theatre_id=2000 + index,
                slug=f"amc-top-hybrid-{index}",
                theatre_url=f"https://www.amctheatres.com/movie-theatres/test/amc-top-hybrid-{index}",
                name=name,
                address_line1="1 Sample Way",
                city="Sample",
                state=state,
                postal_code=f"{index:05d}",
                latitude=None,
                longitude=None,
                timezone="America/New_York",
                inferred_screen_count=10,
            )
            for index, (name, state) in enumerate(sample_service.TOP_HYBRID_THEATRES, start=1)
        ]
        extra_theatre = AmcTheatre(
            amc_theatre_id=2999,
            slug="amc-extra-24",
            theatre_url="https://www.amctheatres.com/movie-theatres/test/amc-extra-24",
            name="AMC Extra 24",
            address_line1="1 Sample Way",
            city="Sample",
            state="CA",
            postal_code="99999",
            latitude=None,
            longitude=None,
            timezone="America/Los_Angeles",
            inferred_screen_count=24,
        )
        db.upsert_theatres(self.conn, [*theatres, extra_theatre])

        sample_set = sample_service.ensure_default_theatre_sample(self.conn)
        members = db.select_theatre_sample_members(self.conn, sample_set.sample_set_id)

        self.assertEqual("top_hybrid_30", sample_set.sample_key)
        self.assertEqual(30, sample_set.sample_size)
        self.assertEqual(30, len(members))
        self.assertEqual(
            [theatre.amc_theatre_id for theatre in theatres],
            [member.amc_theatre_id for member in members],
        )
        self.assertNotIn(extra_theatre.amc_theatre_id, {member.amc_theatre_id for member in members})

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
                        attribute_names="Reserved Seating",
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

    def test_seat_collection_run_excludes_non_reserved_showtimes(self) -> None:
        theatres = [
            synthetic_theatre(index, state="CA", timezone="America/Los_Angeles", screens=10 + index)
            for index in range(1, 4)
        ]
        db.upsert_theatres(self.conn, theatres)
        for theatre in db.select_active_theatres_basic(self.conn):
            is_reserved = theatre.amc_theatre_id != theatres[0].amc_theatre_id
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
                        attribute_names="Reserved Seating" if is_reserved else "Closed Caption|Audio Description",
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
        sample_service.ensure_default_theatre_sample(
            self.conn,
            sample_key="reserved-only-seat-run",
            sample_size=3,
            certainty_count=0,
            seed="sample",
        )
        self.conn.commit()

        run_id, task_count = movie_service.create_seat_collection_run(
            self.conn,
            exhibition_date=dt.date(2026, 7, 1),
            sample_key="reserved-only-seat-run",
        )

        task_rows = self.conn.execute(
            """
            SELECT s.attribute_names, t.priority, t.deadline_at, t.target_offset_minutes, t.scheduled_for
            FROM collection_tasks t
            JOIN amc_showtimes s ON s.showtime_id = t.showtime_id
            WHERE t.run_id = %s
            ORDER BY s.showtime_id
            """,
            (run_id,),
        ).fetchall()
        self.assertEqual(2, task_count)
        self.assertTrue(all("Reserved Seating" in row[0] for row in task_rows))
        self.assertTrue(all(1 <= int(row[1]) <= 20 for row in task_rows))
        self.assertTrue(all(row[2] is None for row in task_rows))
        self.assertTrue(all(int(row[3]) == 10 for row in task_rows))

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
        self.assertEqual([True, True], fetcher.refresh_values)
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

    def test_mark_task_failed_can_extend_attempt_budget_for_backoff(self) -> None:
        campaign_id = db.ensure_campaign(self.conn, dt.date(2026, 7, 1))
        run_id = db.create_run(self.conn, campaign_id=campaign_id, run_type="seat_collection")
        now = db.utc_now()
        row = self.conn.execute(
            """
            INSERT INTO collection_tasks (
                run_id, task_type, scheduled_for, status, priority, attempt_count,
                max_attempts, available_after
            ) VALUES (%s, 'collect_seat_snapshot', %s, 'running', 5, 3, 3, %s)
            RETURNING task_id
            """,
            (run_id, now, now),
        ).fetchone()

        db.mark_task_failed(
            self.conn,
            int(row[0]),
            exc=ValueError("Could not find showtime.seatingLayout.seats in AMC RSC payload"),
            retry_delay_seconds=900,
            minimum_max_attempts=6,
        )

        task_row = self.conn.execute(
            """
            SELECT status, attempt_count, max_attempts, available_after, completed_at
            FROM collection_tasks
            WHERE task_id = %s
            """,
            (int(row[0]),),
        ).fetchone()

        self.assertEqual("retry", task_row[0])
        self.assertEqual(3, int(task_row[1]))
        self.assertEqual(6, int(task_row[2]))
        self.assertGreaterEqual(db.ensure_utc(task_row[3]), now + dt.timedelta(seconds=890))
        self.assertIsNone(task_row[4])

    def test_active_seat_throttle_blocks_seat_claims_across_workers(self) -> None:
        campaign_id = db.ensure_campaign(self.conn, dt.date(2026, 7, 1))
        run_id = db.create_run(self.conn, campaign_id=campaign_id, run_type="seat_collection")
        now = db.utc_now()
        self.conn.executemany(
            """
            INSERT INTO collection_tasks (
                run_id, task_type, scheduled_for, status, priority, available_after
            ) VALUES (%s, %s, %s, 'queued', 5, %s)
            """,
            [
                (run_id, "collect_seat_snapshot", now, now),
                (run_id, "collect_theatre_showtimes", now, now),
            ],
        )
        blocked_until = db.extend_throttle(
            self.conn,
            "seat_collection",
            delay_seconds=75,
            reason="test shared backoff",
            now=now,
        )
        self.conn.commit()

        claimed = db.claim_due_tasks(self.conn, worker_id="worker-1", limit=10, now=now + dt.timedelta(seconds=10))

        self.assertEqual(["collect_theatre_showtimes"], [task.task_type for task in claimed])
        self.assertEqual(blocked_until, db.active_throttle_until(self.conn, "seat_collection", now=now))

        claimed_after_backoff = db.claim_due_tasks(
            self.conn,
            worker_id="worker-2",
            limit=10,
            now=blocked_until + dt.timedelta(seconds=1),
        )

        self.assertEqual(["collect_seat_snapshot"], [task.task_type for task in claimed_after_backoff])

    def test_historical_deadline_does_not_block_seat_claim(self) -> None:
        campaign_id = db.ensure_campaign(self.conn, dt.date(2026, 7, 1))
        run_id = db.create_run(self.conn, campaign_id=campaign_id, run_type="seat_collection")
        now = db.utc_now()
        row = self.conn.execute(
            """
            INSERT INTO collection_tasks (
                run_id, task_type, scheduled_for, deadline_at, status, priority, available_after
            ) VALUES (%s, 'collect_seat_snapshot', %s, %s, 'queued', 5, %s)
            RETURNING task_id
            """,
            (run_id, now - dt.timedelta(minutes=10), now - dt.timedelta(minutes=1), now - dt.timedelta(minutes=10)),
        ).fetchone()
        self.conn.commit()

        claimed = db.claim_due_tasks(self.conn, worker_id="worker-1", limit=10, now=now)

        task_row = self.conn.execute(
            """
            SELECT status, last_error_type, completed_at
            FROM collection_tasks
            WHERE task_id = %s
            """,
            (int(row[0]),),
        ).fetchone()
        self.assertEqual([int(row[0])], [task.task_id for task in claimed])
        self.assertEqual("running", task_row[0])
        self.assertIsNone(task_row[1])
        self.assertIsNone(task_row[2])

    def test_seat_tasks_without_deadline_are_claimable_after_showtime_start(self) -> None:
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
                    showtime_id="expired-showtime",
                    when="2026-07-01T19:00:00-04:00",
                    movie_name="Sample One",
                    movie_id="movie-1",
                    showtime_url="https://www.amctheatres.com/showtimes/expired-showtime",
                    attribute_names="Reserved Seating",
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
        now = showtime.utc_start_at + dt.timedelta(minutes=1)
        row = self.conn.execute(
            """
            INSERT INTO collection_tasks (
                run_id, task_type, showtime_id, scheduled_for, deadline_at, status, priority, available_after
            ) VALUES (%s, 'collect_seat_snapshot', %s, %s, NULL, 'queued', 5, %s)
            RETURNING task_id
            """,
            (
                run_id,
                showtime.showtime_id,
                showtime.utc_start_at - dt.timedelta(minutes=5),
                showtime.utc_start_at - dt.timedelta(minutes=5),
            ),
        ).fetchone()
        self.conn.commit()

        claimed = db.claim_due_tasks(self.conn, worker_id="worker-1", limit=10, now=now)

        task_row = self.conn.execute(
            """
            SELECT status, last_error_type, completed_at
            FROM collection_tasks
            WHERE task_id = %s
            """,
            (int(row[0]),),
        ).fetchone()
        self.assertEqual([int(row[0])], [task.task_id for task in claimed])
        self.assertEqual("running", task_row[0])
        self.assertIsNone(task_row[1])
        self.assertIsNone(task_row[2])

    def test_global_live_request_limiter_returns_wait_after_cap(self) -> None:
        now = dt.datetime(2026, 7, 1, 12, 0, tzinfo=dt.timezone.utc)

        first_wait = db.acquire_live_request_slot(
            self.conn,
            max_requests_per_minute=2,
            worker_id="worker-1",
            url_kind="rsc",
            source_url="https://www.amctheatres.com/showtimes/100/seats?_rsc=1",
            now=now,
        )
        second_wait = db.acquire_live_request_slot(
            self.conn,
            max_requests_per_minute=2,
            worker_id="worker-2",
            url_kind="rsc",
            source_url="https://www.amctheatres.com/showtimes/101/seats?_rsc=1",
            now=now + dt.timedelta(seconds=1),
        )
        third_wait = db.acquire_live_request_slot(
            self.conn,
            max_requests_per_minute=2,
            worker_id="worker-3",
            url_kind="rsc",
            source_url="https://www.amctheatres.com/showtimes/102/seats?_rsc=1",
            now=now + dt.timedelta(seconds=2),
        )

        self.assertIsNone(first_wait)
        self.assertIsNone(second_wait)
        self.assertEqual(now + dt.timedelta(seconds=60), third_wait)


if __name__ == "__main__":
    unittest.main()
