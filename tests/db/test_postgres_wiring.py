#!/usr/bin/env python3
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from pm_box_office.db import connection as db
from pm_box_office.sources.wikipedia import ingest as ingest_wikipedia_boxoffice
from pm_box_office.sources.the_numbers import ingest as scrape_the_numbers


class PostgresWiringTests(unittest.TestCase):
    def test_rejects_non_postgres_database_urls(self) -> None:
        with self.assertRaises(SystemExit):
            db.validate_postgres_url("data/processed/the_numbers_box_office.sqlite")
        with self.assertRaises(SystemExit):
            db.validate_postgres_url("sqlite:///data/processed/the_numbers_box_office.sqlite")

    def test_accepts_postgres_database_urls(self) -> None:
        db.validate_postgres_url("postgresql://user:pass@localhost:5432/pm_box_office")
        db.validate_postgres_url("postgres://user:pass@localhost/pm_box_office")

    def test_dotenv_supplies_database_url_without_overriding_env(self) -> None:
        original = os.environ.pop("DATABASE_URL", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                env_path = Path(tmp) / ".env"
                env_path.write_text(
                    "DATABASE_URL=postgresql://localhost/from_dotenv\n",
                    encoding="utf-8",
                )
                db.load_dotenv(env_path)
                self.assertEqual("postgresql://localhost/from_dotenv", os.environ.get("DATABASE_URL"))

                env_path.write_text(
                    "DATABASE_URL=postgresql://localhost/ignored\n",
                    encoding="utf-8",
                )
                db.load_dotenv(env_path)
                self.assertEqual("postgresql://localhost/from_dotenv", os.environ.get("DATABASE_URL"))
        finally:
            if original is None:
                os.environ.pop("DATABASE_URL", None)
            else:
                os.environ["DATABASE_URL"] = original

    def test_ingest_clis_are_postgres_url_only(self) -> None:
        scraper_options = {
            option
            for action in scrape_the_numbers.build_arg_parser()._actions
            for option in action.option_strings
        }
        wiki_options = {
            option
            for action in ingest_wikipedia_boxoffice.build_parser()._actions
            for option in action.option_strings
        }
        self.assertIn("--database-url", scraper_options)
        self.assertIn("--database-url", wiki_options)
        self.assertNotIn("--db", scraper_options)
        self.assertNotIn("--db", wiki_options)


if __name__ == "__main__":
    unittest.main()
