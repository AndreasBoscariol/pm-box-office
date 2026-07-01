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

        self.conn.execute("CREATE TABLE movies (movie_id BIGINT PRIMARY KEY)")
        self.conn.execute("INSERT INTO movies (movie_id) VALUES (1)")
        run_id = repository.create_run(self.conn, source_key="wikipedia", trigger="manual")

        self.assertIsNotNone(run_id)

    def test_log_tail_returns_oldest_to_newest_with_limit(self) -> None:
        run_id = repository.create_run(self.conn, source_key="the_numbers", trigger="manual")
        for index in range(5):
            repository.append_log(self.conn, run_id=run_id, stream="stdout", line=f"line {index}")

        logs = repository.list_log_tail(self.conn, run_id, limit=3)

        self.assertEqual(["line 2", "line 3", "line 4"], [row["line"] for row in logs])


if __name__ == "__main__":
    unittest.main()

