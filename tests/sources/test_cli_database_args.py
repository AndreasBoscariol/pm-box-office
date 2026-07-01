from __future__ import annotations

import unittest

from pm_box_office.models import evaluate_boxofficepro
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
            evaluate_boxofficepro.build_parser(),
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

    def test_amc_collect_create_seat_run_is_sample_only(self) -> None:
        parser = collect.build_parser()

        args = parser.parse_args(["create-seat-run", "2026-07-01", "--sample-key", "balanced_175"])

        self.assertEqual("balanced_175", args.sample_key)
        with self.assertRaises(SystemExit):
            parser.parse_args(["create-seat-run", "2026-07-01", "--full-coverage"])

    def test_amc_collect_hides_legacy_ad_hoc_commands(self) -> None:
        help_text = collect.build_parser().format_help()

        self.assertNotIn("select-movie", help_text)
        self.assertNotIn("morning-showtimes", help_text)
        self.assertNotIn("seat-snapshots", help_text)
        self.assertIn("select-the-numbers-active", help_text)
        self.assertIn("reset-collection-state", help_text)

    def test_amc_collect_parses_the_numbers_active_selector(self) -> None:
        args = collect.build_parser().parse_args(
            ["select-the-numbers-active", "2026-07-02", "--lookback-days", "10"]
        )

        self.assertEqual("select-the-numbers-active", args.command)
        self.assertEqual(10, args.lookback_days)

    def test_amc_worker_defaults_are_conservative(self) -> None:
        args = worker.build_parser().parse_args([])

        self.assertEqual(1, args.limit)
        self.assertEqual(3.0, args.delay_seconds)


if __name__ == "__main__":
    unittest.main()
