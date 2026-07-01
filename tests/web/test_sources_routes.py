from __future__ import annotations

import unittest
import uuid
from unittest.mock import patch

from pm_box_office.orchestration import repository
from pm_box_office.web.routes import sources


class SourceRouteTests(unittest.TestCase):
    def test_run_source_redirects_with_started_message(self) -> None:
        run_id = uuid.uuid4()
        with patch.object(sources.runner, "start_source_run", return_value=run_id) as start_source_run:
            response = sources.run_source("the_numbers")

        self.assertEqual(303, response.status_code)
        self.assertIn("Started%20the_numbers", response.headers["location"])
        start_source_run.assert_called_once_with("the_numbers", trigger="manual")

    def test_run_source_redirects_with_orchestration_error(self) -> None:
        with patch.object(
            sources.runner,
            "start_source_run",
            side_effect=repository.SourceAlreadyRunningError("already running"),
        ):
            response = sources.run_source("the_numbers")

        self.assertEqual(303, response.status_code)
        self.assertEqual("/sources?error=already%20running", response.headers["location"])

    def test_cancel_source_run_rejects_invalid_uuid(self) -> None:
        with patch.object(sources.runner, "cancel_run") as cancel_run:
            response = sources.cancel_source_run("not-a-uuid")

        self.assertEqual(303, response.status_code)
        self.assertEqual("/sources?error=Invalid%20run%20id", response.headers["location"])
        cancel_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
