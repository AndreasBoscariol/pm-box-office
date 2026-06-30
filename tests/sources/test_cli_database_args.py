from __future__ import annotations

import unittest

from pm_box_office.models import train
from pm_box_office.research.papers import recreate_competitive_dynamics
from pm_box_office.sources.amc import collect
from pm_box_office.sources.amc.jobs import worker
from pm_box_office.sources.the_numbers import ingest as the_numbers
from pm_box_office.sources.wikipedia import ingest as wikipedia


class CliDatabaseArgTests(unittest.TestCase):
    def test_db_backed_cli_help_includes_database_url(self) -> None:
        parsers = [
            collect.build_parser(),
            worker.build_parser(),
            train.build_parser(),
            recreate_competitive_dynamics.build_parser(),
            the_numbers.build_arg_parser(),
            wikipedia.build_parser(),
        ]
        for parser in parsers:
            with self.subTest(prog=parser.prog):
                self.assertIn("--database-url", parser.format_help())

    def test_amc_collect_accepts_database_url_before_or_after_subcommand(self) -> None:
        parser = collect.build_parser()

        before = parser.parse_args(["--database-url", "postgresql://user@host/db", "init-db"])
        after = parser.parse_args(["init-db", "--database-url", "postgresql://other@host/db"])

        self.assertEqual("postgresql://user@host/db", before.database_url)
        self.assertEqual("postgresql://other@host/db", after.database_url)


if __name__ == "__main__":
    unittest.main()
