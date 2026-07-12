from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

from pm_box_office.sources.boxofficeguru import ingest as boxofficeguru_ingest
from pm_box_office.sources.boxofficepro import ingest as boxofficepro_ingest
from pm_box_office.sources.boxofficereport import ingest as boxofficereport_ingest
from pm_box_office.sources.boxofficetheory import ingest as boxofficetheory_ingest
from pm_box_office.sources.boxofficetheory_substack import ingest as boxofficetheory_substack_ingest
from pm_box_office.sources.edwarddouglas_substack import ingest as edwarddouglas_substack_ingest
from pm_box_office.sources.joblo import ingest as joblo_ingest
from pm_box_office.sources.polymarket import ingest as polymarket_ingest
from pm_box_office.sources.the_numbers import ingest as the_numbers_ingest
from pm_box_office.sources.toddmthatcher import ingest as toddmthatcher_ingest
from pm_box_office.orchestration import repository
from pm_box_office.orchestration.registry import (
    BOX_OFFICE_PREDICTION_SOURCE_KEYS,
    RUN_ALL_SOURCE_KEYS,
    SOURCE_DEFINITIONS,
)
from tests.postgres_test_utils import drop_isolated_postgres_schema, make_isolated_postgres_schema


def test_project_ingest_scripts_are_registered_for_web_orchestration_without_database() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    script_entries = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))["project"]["scripts"]
    script_modules = {
        entry_point.split(":", 1)[0]
        for script_name, entry_point in script_entries.items()
        if script_name.endswith("-ingest")
        or script_name in {
            "pm-box-office-scrape-the-numbers",
            "pm-box-office-the-numbers-predictions",
        }
    }
    registered_modules = {source.command for source in SOURCE_DEFINITIONS}

    assert set() == script_modules - registered_modules


def test_manual_only_prediction_source_is_excluded_from_run_all_without_database() -> None:
    source_by_key = {source.source_key: source for source in SOURCE_DEFINITIONS}

    assert source_by_key["the_numbers_predictions"].command == "pm_box_office.sources.the_numbers.predictions"
    assert "the_numbers_predictions" not in RUN_ALL_SOURCE_KEYS


def test_box_office_prediction_ingests_are_registered_for_web_without_database() -> None:
    source_by_key = {source.source_key: source for source in SOURCE_DEFINITIONS}

    for source_key in BOX_OFFICE_PREDICTION_SOURCE_KEYS:
        assert source_key in source_by_key
        assert source_by_key[source_key].command.startswith("pm_box_office.sources.")
        assert "Predictions" in source_by_key[source_key].display_name
        assert source_key in RUN_ALL_SOURCE_KEYS


def test_polymarket_metadata_ingest_is_a_scheduled_web_source_without_database() -> None:
    source_by_key = {source.source_key: source for source in SOURCE_DEFINITIONS}

    assert source_by_key["polymarket_metadata"].command == "pm_box_office.sources.polymarket.ingest"
    assert source_by_key["polymarket_metadata"].requires_movies is True
    assert "polymarket_metadata" not in RUN_ALL_SOURCE_KEYS


def test_run_all_source_schema_initializers_take_advisory_lock_before_movie_schema_without_database() -> None:
    modules = [
        the_numbers_ingest,
        boxofficepro_ingest,
        boxofficereport_ingest,
        boxofficetheory_ingest,
        boxofficetheory_substack_ingest,
        edwarddouglas_substack_ingest,
        boxofficeguru_ingest,
        toddmthatcher_ingest,
        joblo_ingest,
    ]

    for module in modules:
        source = Path(module.__file__).read_text(encoding="utf-8")
        lock_index = source.index("acquire_schema_init_lock(conn)")
        movie_schema_index = source.index("movie_identity.ensure_movie_identity_schema(conn)")
        assert lock_index < movie_schema_index


def test_polymarket_metadata_parser_validates_complete_bucket_set_without_database() -> None:
    event = {
        "id": "event-1",
        "title": "Example Movie Opening Weekend Box Office",
        "description": "Domestic 3-day opening weekend, per The Numbers. If exactly between brackets, use the higher bracket.",
        "markets": [
            {"id": "m1", "question": "Will Example Movie make under $10M?"},
            {"id": "m2", "question": "Will Example Movie make $10M-$20M?"},
            {"id": "m3", "question": "Will Example Movie make $20M-$30M?"},
            {"id": "m4", "question": "Will Example Movie make $30M-$40M?"},
            {"id": "m5", "question": "Will Example Movie make $40M or more?"},
        ],
    }

    buckets, errors = polymarket_ingest.parse_event_buckets(event)
    validation = polymarket_ingest.validate_bucket_set(buckets)

    assert errors == []
    assert validation.valid
    assert [bucket.market_id for bucket in buckets] == ["m1", "m2", "m3", "m4", "m5"]


class OrchestrationRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn, self.schema = make_isolated_postgres_schema()
        repository.initialize_orchestration_database(self.conn)
        repository.seed_sources(self.conn)
        self.conn.commit()

    def tearDown(self) -> None:
        drop_isolated_postgres_schema(self.conn, self.schema)

    def test_seed_sources_and_create_run_prevents_duplicate_active_run(self) -> None:
        run_id = repository.create_run(self.conn, source_key="the_numbers", trigger="manual")

        with self.assertRaises(repository.SourceAlreadyRunningError):
            repository.create_run(self.conn, source_key="the_numbers", trigger="manual")

        row = self.conn.execute(
            "SELECT source_key, status FROM ingest_runs WHERE run_id = %s",
            (str(run_id),),
        ).fetchone()
        self.assertEqual(("the_numbers", "queued"), tuple(row))

    def test_create_run_fails_stale_queued_runs_before_concurrency_check(self) -> None:
        stale_run_id = repository.create_run(self.conn, source_key="the_numbers", trigger="manual")
        self.conn.execute(
            """
            UPDATE ingest_runs
            SET requested_at = CURRENT_TIMESTAMP - INTERVAL '10 minutes',
                heartbeat_at = NULL
            WHERE run_id = %s
            """,
            (str(stale_run_id),),
        )

        new_run_id = repository.create_run(self.conn, source_key="the_numbers", trigger="manual")

        rows = self.conn.execute(
            """
            SELECT run_id, status, error_summary
            FROM ingest_runs
            ORDER BY requested_at
            """
        ).fetchall()
        self.assertEqual(str(stale_run_id), str(rows[0][0]))
        self.assertEqual("failed", rows[0][1])
        self.assertIn("queued past supervisor startup", rows[0][2])
        self.assertEqual(str(new_run_id), str(rows[1][0]))
        self.assertEqual("queued", rows[1][1])

    def test_movie_dependent_sources_require_movies(self) -> None:
        with self.assertRaises(repository.SourceDependencyError):
            repository.create_run(self.conn, source_key="wikipedia", trigger="manual")
        with self.assertRaises(repository.SourceDependencyError):
            repository.create_run(self.conn, source_key="rotten_tomatoes", trigger="manual")

        self.conn.execute("CREATE TABLE movies (movie_id BIGINT PRIMARY KEY)")
        self.conn.execute("INSERT INTO movies (movie_id) VALUES (1)")
        run_id = repository.create_run(self.conn, source_key="wikipedia", trigger="manual")
        rt_run_id = repository.create_run(self.conn, source_key="rotten_tomatoes", trigger="manual")

        self.assertIsNotNone(run_id)
        self.assertIsNotNone(rt_run_id)

    def test_seed_sources_registers_rotten_tomatoes(self) -> None:
        row = self.conn.execute(
            """
            SELECT display_name, command, requires_movies
            FROM ingest_sources
            WHERE source_key = 'rotten_tomatoes'
            """
        ).fetchone()

        self.assertEqual(
            ("Rotten Tomatoes Critics", "pm_box_office.sources.rotten_tomatoes.ingest", True),
            tuple(row),
        )

    def test_polymarket_metadata_sync_upserts_valid_market_tables(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE movies (
                movie_id BIGINT PRIMARY KEY,
                title TEXT NOT NULL,
                release_date DATE,
                release_year INTEGER
            )
            """
        )
        self.conn.execute(
            "INSERT INTO movies (movie_id, title, release_date, release_year) VALUES (1, 'Example Movie', '2026-07-10', 2026)"
        )
        event = {
            "id": "event-1",
            "slug": "example-movie-opening-weekend",
            "title": "Example Movie Opening Weekend Box Office",
            "description": "Domestic 3-day opening weekend, per The Numbers. If exactly between brackets, use the higher bracket.",
            "active": True,
            "closed": False,
            "markets": [
                {"id": "m1", "question": "Will Example Movie make under $10M?", "clobTokenIds": '["101","102"]', "outcomes": '["Yes","No"]'},
                {"id": "m2", "question": "Will Example Movie make $10M-$20M?"},
                {"id": "m3", "question": "Will Example Movie make $20M-$30M?"},
                {"id": "m4", "question": "Will Example Movie make $30M-$40M?"},
                {"id": "m5", "question": "Will Example Movie make $40M or more?"},
            ],
        }

        summary = polymarket_ingest.sync_metadata(self.conn, [event])

        self.assertEqual(1, summary.events_upserted)
        self.assertEqual(5, summary.markets_upserted)
        self.assertEqual(5, summary.semantics_upserted)
        self.assertEqual(1, summary.valid_events)
        self.assertEqual((1,), summary.matched_movie_ids)
        validation = self.conn.execute(
            """
            SELECT validation_status, validated_bucket_count, validation_errors
            FROM prediction_market_backtest.event_bucket_validations
            WHERE event_id = 'event-1'
            """
        ).fetchone()
        self.assertEqual("valid", validation[0])
        self.assertEqual(5, validation[1])
        self.assertEqual([], validation[2])
        boundaries = self.conn.execute(
            """
            SELECT bucket_upper
            FROM prediction_market_backtest.contract_semantics
            WHERE market_id IN ('m1', 'm2', 'm3', 'm4')
            ORDER BY bucket_upper
            """
        ).fetchall()
        self.assertEqual([10_000_000, 20_000_000, 30_000_000, 40_000_000], [int(row[0]) for row in boundaries])

    def test_seed_sources_registers_the_numbers_predictions_for_manual_runs(self) -> None:
        row = self.conn.execute(
            """
            SELECT display_name, command, enabled
            FROM ingest_sources
            WHERE source_key = 'the_numbers_predictions'
            """
        ).fetchone()

        self.assertEqual(
            ("The Numbers Predictions", "pm_box_office.sources.the_numbers.predictions", True),
            tuple(row),
        )
        self.assertNotIn("the_numbers_predictions", RUN_ALL_SOURCE_KEYS)

    def test_seed_sources_registers_boxofficetheory_substack_for_web_runs(self) -> None:
        row = self.conn.execute(
            """
            SELECT display_name, command, enabled, requires_movies
            FROM ingest_sources
            WHERE source_key = 'boxofficetheory_substack'
            """
        ).fetchone()

        self.assertEqual(
            (
                "Box Office Theory Substack Predictions",
                "pm_box_office.sources.boxofficetheory_substack.ingest",
                True,
                False,
            ),
            tuple(row),
        )
        self.assertIn("boxofficetheory_substack", RUN_ALL_SOURCE_KEYS)

    def test_autorun_state_tracks_next_daily_run(self) -> None:
        state = repository.get_autorun_state(self.conn)

        self.assertTrue(state["enabled"])
        self.assertEqual(24, state["interval_hours"])
        self.assertGreater(state["seconds_until_next_run"], 0)
        self.assertLessEqual(state["seconds_until_next_run"], 24 * 60 * 60)

        repository.record_autorun_trigger(self.conn)
        updated = repository.get_autorun_state(self.conn)

        self.assertGreater(updated["seconds_until_next_run"], 0)
        self.assertLessEqual(updated["seconds_until_next_run"], 24 * 60 * 60)
        self.assertIsNotNone(updated["last_triggered_at"])

    def test_log_tail_returns_oldest_to_newest_with_limit(self) -> None:
        run_id = repository.create_run(self.conn, source_key="the_numbers", trigger="manual")
        for index in range(5):
            repository.append_log(self.conn, run_id=run_id, stream="stdout", line=f"line {index}")

        logs = repository.list_log_tail(self.conn, run_id, limit=3)

        self.assertEqual(["line 2", "line 3", "line 4"], [row["line"] for row in logs])


if __name__ == "__main__":
    unittest.main()
