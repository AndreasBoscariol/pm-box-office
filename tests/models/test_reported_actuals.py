from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from models.boxoffice.artifacts import ModelArtifacts
from models.boxoffice.reported_actuals import fetch_reported_actuals, overlay_reported_actuals
from models.boxoffice.schema import MovieOpening


def test_overlay_reported_actuals_replaces_frozen_missing_values() -> None:
    movie = MovieOpening(
        movie_id=11,
        release_run_id=22,
        title="Example",
        opening_weekend_start=date(2026, 7, 10),
    )
    artifacts = ModelArtifacts(
        model_version="test",
        artifact_dir=Path("."),
        manifest={},
        daily_baseline=pd.DataFrame(
            [
                {
                    "release_run_id": 22,
                    "actual_fri_usd": float("nan"),
                    "actual_sat_usd": float("nan"),
                    "actual_sun_usd": float("nan"),
                    "actual_ow_usd": float("nan"),
                }
            ]
        ),
    )
    actuals = pd.DataFrame(
        [
            {"release_run_id": 22, "box_office_date": "2026-07-10", "gross_usd": 30, "fetched_at": "2026-07-11T01:00:00Z"},
            {"release_run_id": 22, "box_office_date": "2026-07-11", "gross_usd": 40, "fetched_at": "2026-07-12T01:00:00Z"},
        ]
    )

    merged = overlay_reported_actuals(artifacts, movies=[movie], actuals=actuals)

    row = merged.daily_baseline.iloc[0]
    assert row["actual_fri_usd"] == 30
    assert row["actual_sat_usd"] == 40
    assert pd.isna(row["actual_sun_usd"])
    assert pd.isna(row["actual_ow_usd"])
    assert pd.isna(artifacts.daily_baseline.iloc[0]["actual_fri_usd"])


def test_overlay_reported_actuals_sets_weekend_actual_only_when_complete() -> None:
    movie = MovieOpening(11, 22, "Example", date(2026, 7, 10))
    artifacts = ModelArtifacts(
        model_version="test",
        artifact_dir=Path("."),
        manifest={},
        daily_baseline=pd.DataFrame([{"release_run_id": 22}]),
    )
    actuals = pd.DataFrame(
        [
            {"release_run_id": 22, "box_office_date": date(2026, 7, 10), "gross_usd": 30},
            {"release_run_id": 22, "box_office_date": date(2026, 7, 11), "gross_usd": 40},
            {"release_run_id": 22, "box_office_date": date(2026, 7, 12), "gross_usd": 20},
        ]
    )

    row = overlay_reported_actuals(artifacts, movies=[movie], actuals=actuals).daily_baseline.iloc[0]

    assert row["actual_ow_usd"] == 90


def test_fetch_reported_actuals_uses_vintage_primary_key_when_vintages_exist() -> None:
    class Cursor:
        description = [("actual_record_id",), ("release_run_id",), ("box_office_date",), ("gross_usd",), ("actual_source",), ("actual_ingested_at",), ("actual_published_at",)]

        def fetchall(self) -> list[tuple[object, ...]]:
            return []

    class Connection:
        def __init__(self) -> None:
            self.sql: list[str] = []

        def execute(self, sql: str, _params: object = None) -> object:
            self.sql.append(sql)
            if "to_regclass('daily_box_office_vintages')" in sql:
                return type("Exists", (), {"fetchone": lambda _self: (True,)})()
            return Cursor()

    conn = Connection()
    frame = fetch_reported_actuals(
        conn,
        movies=[MovieOpening(11, 22, "Example", date(2026, 7, 10))],
        as_of_utc=pd.Timestamp("2026-07-11T00:00:00Z").to_pydatetime(),
    )

    assert frame.empty
    assert "daily_box_office_vintage_id AS actual_record_id" in conn.sql[-1]
