from __future__ import annotations

import unittest

from pm_box_office.orchestration import repository
from tests.postgres_test_utils import drop_isolated_postgres_schema, make_isolated_postgres_schema


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
