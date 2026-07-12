from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pm_box_office.db.connection import database_url_from_env
from pm_box_office.orchestration import repository, supervisor
from tests.postgres_test_utils import (
    drop_isolated_postgres_schema,
    make_isolated_postgres_schema,
    url_with_search_path,
)


class OrchestrationSupervisorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn, self.schema = make_isolated_postgres_schema()
        repository.initialize_orchestration_database(self.conn)
        repository.seed_sources(self.conn)
        self.conn.execute(
            """
            INSERT INTO ingest_sources (source_key, display_name, command, default_args)
            VALUES ('fake_success', 'Fake Success', 'fake_success_ingest', '[]'::jsonb)
            """
        )
        self.run_id = repository.create_run(self.conn, source_key="fake_success", trigger="test")
        self.conn.commit()

    def tearDown(self) -> None:
        drop_isolated_postgres_schema(self.conn, self.schema)

    def test_supervisor_records_subprocess_logs_and_success(self) -> None:
        base_url = os.environ.get("TEST_DATABASE_URL") or database_url_from_env()
        if not base_url:
            raise unittest.SkipTest("Set TEST_DATABASE_URL or DATABASE_URL to run PostgreSQL integration tests.")
        database_url = url_with_search_path(base_url, self.schema)
        with tempfile.TemporaryDirectory() as temp_dir:
            module_path = Path(temp_dir) / "fake_success_ingest.py"
            module_path.write_text(
                "print('fake scraper started')\nprint('rows=3')\n",
                encoding="utf-8",
            )
            previous_pythonpath = os.environ.get("PYTHONPATH")
            os.environ["PYTHONPATH"] = (
                temp_dir if previous_pythonpath is None else f"{temp_dir}{os.pathsep}{previous_pythonpath}"
            )
            try:
                exit_code = supervisor.supervise_run(str(self.run_id), database_url=database_url)
            finally:
                if previous_pythonpath is None:
                    os.environ.pop("PYTHONPATH", None)
                else:
                    os.environ["PYTHONPATH"] = previous_pythonpath

        self.assertEqual(0, exit_code)
        row = self.conn.execute(
            "SELECT status, exit_code FROM ingest_runs WHERE run_id = %s",
            (str(self.run_id),),
        ).fetchone()
        self.assertEqual(("succeeded", 0), tuple(row))
        logs = repository.list_log_tail(self.conn, self.run_id, limit=10)
        self.assertIn("fake scraper started", [log["line"] for log in logs])

    def test_supervisor_starts_forecast_worker_when_refreshes_are_queued(self) -> None:
        base_url = os.environ.get("TEST_DATABASE_URL") or database_url_from_env()
        if not base_url:
            raise unittest.SkipTest("Set TEST_DATABASE_URL or DATABASE_URL to run PostgreSQL integration tests.")
        database_url = url_with_search_path(base_url, self.schema)
        with tempfile.TemporaryDirectory() as temp_dir:
            module_path = Path(temp_dir) / "fake_success_ingest.py"
            module_path.write_text("print('fake scraper started')\n", encoding="utf-8")
            previous_pythonpath = os.environ.get("PYTHONPATH")
            os.environ["PYTHONPATH"] = (
                temp_dir if previous_pythonpath is None else f"{temp_dir}{os.pathsep}{previous_pythonpath}"
            )
            try:
                with (
                    patch.object(
                        supervisor.movie_identity,
                        "reconcile_prediction_movie_identities",
                        return_value={"repointed_predictions": [], "affected_movie_ids": []},
                    ),
                    patch.object(supervisor, "enqueue_upcoming_candidate_refreshes", return_value=2),
                    patch.object(supervisor, "activate_today_opening_amc_campaign", return_value=0),
                    patch.object(supervisor, "ensure_forecast_refresh_worker_started", return_value=1234) as start_worker,
                ):
                    exit_code = supervisor.supervise_run(str(self.run_id), database_url=database_url)
            finally:
                if previous_pythonpath is None:
                    os.environ.pop("PYTHONPATH", None)
                else:
                    os.environ["PYTHONPATH"] = previous_pythonpath

        self.assertEqual(0, exit_code)
        start_worker.assert_called_once_with()
        logs = repository.list_log_tail(self.conn, self.run_id, limit=10)
        self.assertIn("Forecast refresh worker running pid=1234", [log["line"] for log in logs])


if __name__ == "__main__":
    unittest.main()
