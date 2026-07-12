from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from models.boxoffice.amc_shadow import (
    PREVIEW_TREATMENT,
    SHADOW_COLUMNS,
    THURSDAY_PREVIEW_TREATMENT,
    build_shadow_rows,
    build_thursday_preview_shadow_rows,
    write_shadow_rows,
)
from models.boxoffice.artifacts import ModelArtifacts
from models.boxoffice.origins import build_forecast_origins
from models.boxoffice.schema import ForecastResult, MovieOpening


def test_build_shadow_rows_keeps_candidates_and_production_point_separate() -> None:
    movie = MovieOpening(1, 10, "Example", date(2026, 7, 10))
    origin = next(item for item in build_forecast_origins(movie, mode="historical") if item.origin_key == "FRI_16:00")
    common = dict(
        movie=movie, origin=origin, lo80_usd=20.0, hi80_usd=40.0, lo95_usd=10.0,
        hi95_usd=50.0, point_model="test", interval_model="test", component_source="test",
        model_version="boxoffice_test", run_id="run_1",
    )
    daily = ForecastResult(target="friday", point_usd=30.0, **common)
    weekend = ForecastResult(target="opening_weekend", point_usd=80.0, **common)
    artifacts = ModelArtifacts(
        model_version="boxoffice_test", artifact_dir=Path("."), manifest={},
        daily_baseline=pd.DataFrame([{"release_run_id": 10, "movie_id": 1, "pre_fri_usd": 20.0}]),
    )
    plugin = pd.DataFrame([{
        "movie_id": 1, "regime": "live_friday", "forecast_origin": "16:00", "sample_key": "top_hybrid_30",
        "pred_daily_gross_multiplicative_usd": 30.0, "pred_daily_gross_additive_usd": 25.0,
        "pred_daily_gross_hybrid_usd": 27.5, "amc_coverage": 0.8, "amc_snapshot_count": 12,
        "amc_staleness_p50_minutes": 4.0, "feature_quality_bucket": "medium",
    }])

    rows = build_shadow_rows(movie=movie, results=[daily, weekend], live_plugin=plugin, artifacts=artifacts)

    assert len(rows) == 1
    assert rows[0]["production_daily_usd"] == 30.0
    assert rows[0]["additive_daily_usd"] == 25.0
    assert rows[0]["amc_candidate_available"] is True
    assert rows[0]["provisional_blend_daily_usd"] != rows[0]["production_daily_usd"]
    assert rows[0]["thursday_preview_treatment"] == PREVIEW_TREATMENT


def test_build_shadow_rows_records_thursday_preview_nowcast_when_available() -> None:
    movie = MovieOpening(1, 10, "Example", date(2026, 7, 10))
    origin = next(item for item in build_forecast_origins(movie, mode="historical") if item.origin_key == "FRI_16:00")
    common = dict(
        movie=movie, origin=origin, lo80_usd=20.0, hi80_usd=40.0, lo95_usd=10.0,
        hi95_usd=50.0, point_model="test", interval_model="test", component_source="test",
        model_version="boxoffice_test", run_id="run_1",
    )
    artifacts = ModelArtifacts(
        model_version="boxoffice_test", artifact_dir=Path("."), manifest={},
        daily_baseline=pd.DataFrame([{"release_run_id": 10, "movie_id": 1, "pre_fri_usd": 20.0}]),
    )
    thursday = pd.DataFrame([{
        "movie_id": 1,
        "forecast_origin": "16:00",
        "thursday_amc_seats_collected": True,
        "predicted_thursday_previews_usd": 8_000_000.0,
        "amc_observed_preview_seats": 1234.0,
        "shadow_ow_prior_usd": 108_000_000.0,
        "shadow_ow_prior_source": "thursday_amc_preview_nowcast",
        "thursday_preview_actual_usd": 8_500_000.0,
    }])

    rows = build_shadow_rows(
        movie=movie,
        results=[
            ForecastResult(target="friday", point_usd=30.0, **common),
            ForecastResult(target="opening_weekend", point_usd=80.0, **common),
        ],
        live_plugin=pd.DataFrame(),
        artifacts=artifacts,
        thursday_preview_nowcasts=thursday,
    )

    assert rows[0]["thursday_amc_seats_collected"] is True
    assert rows[0]["predicted_thursday_previews_usd"] == 8_000_000.0
    assert rows[0]["thursday_amc_observed_preview_seats"] == 1234.0
    assert rows[0]["thursday_amc_shadow_ow_prior_usd"] == 108_000_000.0
    assert rows[0]["thursday_amc_shadow_ow_prior_source"] == "thursday_amc_preview_nowcast"
    assert rows[0]["thursday_preview_actual_usd"] == 8_500_000.0


def test_build_thursday_preview_shadow_rows_persists_intraday_origins_without_friday_results() -> None:
    movie = MovieOpening(1, 10, "Example", date(2026, 7, 10))
    artifacts = ModelArtifacts(
        model_version="boxoffice_test", artifact_dir=Path("."), manifest={},
        daily_baseline=pd.DataFrame([{"release_run_id": 10, "movie_id": 1, "actual_ow_usd": 90.0}]),
    )
    nowcasts = pd.DataFrame(
        [
            {
                "movie_id": 1,
                "forecast_origin": "16:00",
                "forecast_origin_utc": "2026-07-09T20:00:00Z",
                "as_of_utc": "2026-07-09T20:00:00Z",
                "sample_key": "top_hybrid_30",
                "predicted_thursday_previews_usd": 8_000_000.0,
                "predicted_thursday_previews_multiplicative_usd": 8_100_000.0,
                "predicted_thursday_previews_additive_usd": 7_900_000.0,
                "predicted_thursday_previews_hybrid_usd": 8_000_000.0,
                "shadow_ow_prior_usd": 108_000_000.0,
                "shadow_ow_prior_source": "thursday_amc_preview_nowcast",
                "thursday_amc_seats_collected": True,
                "amc_observed_preview_seats": 1234.0,
                "amc_coverage": 0.75,
                "amc_snapshot_count": 12,
                "amc_staleness_p50_minutes": 4.0,
                "feature_quality_bucket": "medium",
            }
        ]
    )

    rows = build_thursday_preview_shadow_rows(
        movie=movie,
        thursday_preview_nowcasts=nowcasts,
        artifacts=artifacts,
        run_id="run_1",
    )

    assert len(rows) == 1
    assert rows[0]["origin_key"] == "THU_16:00"
    assert rows[0]["opening_weekend_usd"] == 108_000_000.0
    assert rows[0]["thursday_preview_treatment"] == THURSDAY_PREVIEW_TREATMENT
    assert rows[0]["predicted_thursday_previews_usd"] == 8_000_000.0


def test_shadow_insert_is_immutable_on_conflict() -> None:
    class FakeConnection:
        sql = ""

        def execute(self, _sql: str, _params: object) -> object:
            return type("Cursor", (), {"fetchall": lambda self: [(column,) for column in SHADOW_COLUMNS]})()

        def executemany(self, sql: str, _rows: object) -> object:
            self.sql = sql
            return type("Cursor", (), {"rowcount": 0})()

    conn = FakeConnection()
    write_shadow_rows(conn, [{"release_run_id": 10, "origin_key": "FRI_16:00", "model_version": "v1"}])

    assert "ON CONFLICT (release_run_id, origin_key, model_version) DO UPDATE SET" in conn.sql
    assert "WHERE LEFT(EXCLUDED.origin_key, 4) = 'THU_'" in conn.sql


def test_shadow_insert_uses_existing_columns_when_migration_lags() -> None:
    class FakeConnection:
        sql = ""
        values = []

        def execute(self, _sql: str, _params: object) -> object:
            existing = [
                "release_run_id",
                "movie_id",
                "title",
                "opening_weekend_start",
                "origin_key",
                "forecast_origin",
                "forecast_origin_utc",
                "as_of_utc",
                "model_version",
                "run_id",
                "opening_weekend_usd",
                "thursday_amc_seats_collected",
                "predicted_thursday_previews_usd",
            ]
            return type("Cursor", (), {"fetchall": lambda self: [(column,) for column in existing]})()

        def executemany(self, sql: str, rows: object) -> object:
            self.sql = sql
            self.values = list(rows)
            return type("Cursor", (), {"rowcount": 1})()

    conn = FakeConnection()
    write_shadow_rows(
        conn,
        [
            {
                "release_run_id": 10,
                "movie_id": 1,
                "title": "Example",
                "opening_weekend_start": date(2026, 7, 10),
                "origin_key": "THU_16:00",
                "forecast_origin": "16:00",
                "forecast_origin_utc": "2026-07-09T20:00:00Z",
                "as_of_utc": "2026-07-09T20:00:00Z",
                "model_version": "v1",
                "run_id": "run_1",
                "opening_weekend_usd": 108_000_000.0,
                "thursday_amc_seats_collected": True,
                "predicted_thursday_previews_usd": 8_000_000.0,
                "thursday_amc_shadow_ow_prior_usd": 108_000_000.0,
            }
        ],
    )

    assert "thursday_amc_shadow_ow_prior_usd" not in conn.sql
    assert "opening_weekend_usd" in conn.sql
    assert conn.values[0][-1] == 8_000_000.0
