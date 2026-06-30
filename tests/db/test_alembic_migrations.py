from __future__ import annotations

import os
import unittest
import urllib.parse
from pathlib import Path

from alembic import command
from alembic.config import Config

from pm_box_office.db.connection import connect_database, database_url_from_env, table_names
from tests.postgres_test_utils import drop_isolated_postgres_schema, make_isolated_postgres_schema


def url_with_search_path(database_url: str, schema: str) -> str:
    parsed = urllib.parse.urlsplit(database_url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query.append(("options", f"-csearch_path={schema}"))
    return urllib.parse.urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urllib.parse.urlencode(query),
            parsed.fragment,
        )
    )


class AlembicMigrationTests(unittest.TestCase):
    def test_upgrade_head_creates_core_tables_and_views(self) -> None:
        base_url = os.environ.get("TEST_DATABASE_URL") or database_url_from_env()
        if not base_url:
            raise unittest.SkipTest("Set TEST_DATABASE_URL or DATABASE_URL to run Alembic migration tests.")

        setup_conn, schema = make_isolated_postgres_schema()
        setup_conn.close()
        migration_url = url_with_search_path(base_url, schema)
        previous_database_url = os.environ.get("DATABASE_URL")
        migrated_conn = None
        try:
            os.environ["DATABASE_URL"] = migration_url
            config = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
            command.upgrade(config, "head")

            migrated_conn = connect_database(migration_url)
            names = set(table_names(migrated_conn))
            self.assertIn("movies", names)
            self.assertIn("amc_showtimes", names)
            self.assertIn("collection_runs", names)
            self.assertTrue(
                migrated_conn.execute("SELECT to_regclass('analytics.amc_movie_day_blocks_v1')").fetchone()[0]
            )
        finally:
            if previous_database_url is None:
                os.environ.pop("DATABASE_URL", None)
            else:
                os.environ["DATABASE_URL"] = previous_database_url
            cleanup_conn = migrated_conn or connect_database(base_url)
            drop_isolated_postgres_schema(cleanup_conn, schema)


if __name__ == "__main__":
    unittest.main()

