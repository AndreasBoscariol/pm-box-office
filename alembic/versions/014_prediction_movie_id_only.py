"""prediction tables use movie_id only

Revision ID: 014_prediction_movie_id_only
Revises: 013_drop_legacy_movie_mapping_tables
Create Date: 2026-07-06
"""

from __future__ import annotations

from alembic import op


revision = "014_prediction_movie_id_only"
down_revision = "013_drop_legacy_movie_mapping_tables"
branch_labels = None
depends_on = None


PREDICTION_TABLES = (
    ("boxofficepro_weekend_predictions", "idx_boxofficepro_weekend_predictions_movie", "idx_boxofficepro_weekend_predictions_movie_id"),
    ("boxofficereport_weekend_predictions", "idx_boxofficereport_predictions_movie", "idx_boxofficereport_predictions_movie_id"),
    ("boxofficetheory_predictions", "idx_boxofficetheory_predictions_movie", "idx_boxofficetheory_predictions_movie_id"),
    ("boxofficeguru_predictions", None, "idx_boxofficeguru_predictions_movie_id"),
)


def upgrade() -> None:
    bind = op.get_bind()
    for table, old_index, new_index in PREDICTION_TABLES:
        if not _relation_exists(bind, table):
            continue
        op.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS movie_id BIGINT REFERENCES movies(movie_id)")
        if _column_exists(bind, table, "matched_movie_id"):
            op.execute(
                f"""
                UPDATE {table}
                SET movie_id = matched_movie_id
                WHERE movie_id IS NULL
                  AND matched_movie_id IS NOT NULL
                """
            )
        if old_index:
            op.execute(f"DROP INDEX IF EXISTS {old_index}")
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS matched_movie_id")
        op.execute(f"CREATE INDEX IF NOT EXISTS {new_index} ON {table}(movie_id)")


def downgrade() -> None:
    bind = op.get_bind()
    for table, old_index, _new_index in PREDICTION_TABLES:
        if not _relation_exists(bind, table):
            continue
        op.execute(f"ALTER TABLE IF EXISTS {table} ADD COLUMN IF NOT EXISTS matched_movie_id BIGINT REFERENCES movies(movie_id)")
        op.execute(
            f"""
            UPDATE {table}
            SET matched_movie_id = movie_id
            WHERE matched_movie_id IS NULL
              AND movie_id IS NOT NULL
            """
        )
        if old_index:
            op.execute(f"CREATE INDEX IF NOT EXISTS {old_index} ON {table}(matched_movie_id)")


def _relation_exists(bind: object, relation_name: str) -> bool:
    row = bind.exec_driver_sql("SELECT to_regclass(%s)", (relation_name,)).fetchone()  # type: ignore[attr-defined]
    return bool(row and row[0])


def _column_exists(bind: object, table_name: str, column_name: str) -> bool:
    row = bind.exec_driver_sql(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = %s
          AND column_name = %s
        """,
        (table_name, column_name),
    ).fetchone()  # type: ignore[attr-defined]
    return row is not None
