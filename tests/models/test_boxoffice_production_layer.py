from __future__ import annotations

import math
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from models.boxoffice.artifacts import ModelArtifacts
from models.boxoffice.artifacts import load_model_artifacts
from models.boxoffice.intervals import log_normal_interval
from models.boxoffice.live_composition import compose_live_weekend_forecast
from models.boxoffice.forecast_distribution import ForecastDistribution
from models.boxoffice.persistence import attach_actuals_to_forecast_emissions, create_forecast_tables_sql, remove_obsolete_future_forecasts, write_forecast_rows
from models.boxoffice.origins import build_forecast_origins
from models.boxoffice.pre_release import forecast_pre_release_opening_weekend
from models.boxoffice.pre_release_distribution_policy import base_policy_for_origin, validate_policy
from models.boxoffice.schema import MovieOpening
from models.boxoffice.train_box_office_forecast_artifacts import build_daily_interval_policy, build_interval_policy


def test_build_historical_origins_has_full_timeline_order() -> None:
    movie = MovieOpening(
        movie_id=1,
        release_run_id=10,
        title="Example",
        opening_weekend_start=date(2026, 7, 10),
    )

    origins = build_forecast_origins(movie, mode="historical")

    assert len(origins) == 35
    assert origins[0].origin_key == "P_-14"
    assert origins[13].origin_key == "P_-1"
    assert origins[14].origin_key == "FRI_10:00"
    assert origins[-1].origin_key == "SUN_EOD"
    assert all(origin.as_of_utc <= origin.forecast_origin_utc for origin in origins)


def test_active_artifact_promotes_tail_safe_distribution_policy() -> None:
    artifacts = load_model_artifacts("boxoffice_local_007_retrained_from_db")

    policy = artifacts.pre_release_distribution_policy
    assert policy["policy_name"] == "production_distribution_v2_t4_tail_safe"
    assert policy["status"] == "production"
    assert policy["base_distribution_policy"] == "production_distribution_v1_raw"
    assert policy["tail_safety_enabled"] is True
    assert policy["tail_contamination_weight"] == 0.02
    assert policy["tail_reference_family"] == "student_t"
    assert policy["tail_reference_df"] == 4


def test_v3_bounded_d1_transition_policy_has_exact_boundary_and_unchanged_tail() -> None:
    policy = load_model_artifacts("boxoffice_local_007_retrained_from_db_amc_selector_fix1").pre_release_distribution_policy
    validate_policy(policy)
    assert policy["status"] == "production"
    assert base_policy_for_origin(-4) == "production_distribution_v2_base"
    assert base_policy_for_origin(-3) == "D1_point_size"
    assert base_policy_for_origin(-1) == "D1_point_size"


def test_live_origins_stop_at_as_of() -> None:
    movie = MovieOpening(
        movie_id=1,
        release_run_id=10,
        title="Example",
        opening_weekend_start=date(2026, 7, 10),
    )
    as_of = datetime(2026, 7, 10, 15, 59, tzinfo=timezone.utc)

    origins = build_forecast_origins(movie, mode="live", as_of_utc=as_of)
    keys = [origin.origin_key for origin in origins]

    assert "P_-1" in keys
    assert "FRI_10:00" in keys
    assert "FRI_12:00" not in keys
    assert "SAT_10:00" not in keys


def test_future_reconciliation_deletes_only_stale_virtual_rows() -> None:
    class Cursor:
        rowcount = 3

    class Connection:
        def __init__(self) -> None:
            self.sql = ""
            self.params: tuple[object, ...] = ()

        def execute(self, sql: str, params: tuple[object, ...]) -> Cursor:
            self.sql = sql
            self.params = params
            return Cursor()

    conn = Connection()
    removed = remove_obsolete_future_forecasts(
        conn,
        model_version="boxoffice_test",
        opening_weekend_start=date(2026, 7, 1),
        opening_weekend_end=date(2026, 7, 31),
        active_release_run_ids=[-12, -7],
    )

    assert removed == 3
    assert "release_run_id < 0" in conn.sql
    assert "release_run_id NOT IN (%s, %s)" in conn.sql
    assert conn.params == ("boxoffice_test", date(2026, 7, 1), date(2026, 7, 31), -12, -7)


def test_forecast_schema_includes_append_only_emission_tables() -> None:
    sql = create_forecast_tables_sql()

    assert "analytics.movie_forecast_emissions" in sql
    assert "analytics.movie_forecast_emission_components" in sql
    assert "emitted_forecast_id text PRIMARY KEY" in sql
    assert "forecast_role text NOT NULL DEFAULT 'production'" in sql
    assert "production_policy_label text" in sql
    assert "forecast_generated_at timestamptz NOT NULL" in sql
    assert "first_final_actual_usd numeric" in sql
    assert "latest_final_actual_usd numeric" in sql
    assert "distribution_payload jsonb" in sql
    assert "UNIQUE (model_version, release_run_id, origin_key, target, as_of_utc, forecast_role)" in sql
    assert "idx_movie_forecast_emissions_awaiting_actuals" in sql
    assert "PRIMARY KEY (emitted_forecast_id, forecast_id, component_day)" in sql


def test_attach_actuals_to_forecast_emissions_updates_outcomes_only() -> None:
    class Cursor:
        rowcount = 4

    class Connection:
        def __init__(self) -> None:
            self.sql = ""

        def execute(self, sql: str) -> Cursor:
            self.sql = sql
            return Cursor()

    conn = Connection()
    updated = attach_actuals_to_forecast_emissions(conn)

    assert updated == 4
    assert "UPDATE analytics.movie_forecast_emissions emission" in conn.sql
    assert "SET\n            actual_usd" in conn.sql
    assert "first_final_actual_usd = COALESCE" in conn.sql
    assert "latest_final_actual_usd = latest.actual_usd" in conn.sql
    assert "point_usd =" not in conn.sql
    assert "lo80_usd =" not in conn.sql
    assert "hi80_usd =" not in conn.sql


def test_write_forecast_rows_writes_idempotent_emission_records() -> None:
    class Cursor:
        def __init__(self, rows: list[tuple[object, ...]] | None = None) -> None:
            self._rows = rows or []

        def fetchall(self) -> list[tuple[object, ...]]:
            return self._rows

        def fetchone(self) -> tuple[object, ...] | None:
            return self._rows[0] if self._rows else None

    class Connection:
        def __init__(self) -> None:
            self.executed: list[tuple[str, tuple[object, ...] | None]] = []
            self.batches: list[tuple[str, list[tuple[object, ...]]]] = []

        def execute(self, sql: str, params: tuple[object, ...] | None = None) -> Cursor:
            self.executed.append((sql, params))
            return Cursor([])

        def executemany(self, sql: str, rows: list[tuple[object, ...]]) -> None:
            self.batches.append((sql, rows))

    forecast_row = {
        "forecast_id": "forecast_1",
        "run_id": "run_1",
        "model_version": "boxoffice_test",
        "movie_id": 1,
        "release_run_id": 10,
        "title": "Example",
        "opening_weekend_start": date(2026, 7, 10),
        "regime": "live_friday",
        "origin_key": "FRI_20:00",
        "origin_day": 0,
        "forecast_origin_local": datetime(2026, 7, 10, 20, tzinfo=timezone.utc),
        "forecast_origin_utc": datetime(2026, 7, 10, 20, tzinfo=timezone.utc),
        "as_of_utc": datetime(2026, 7, 10, 20, tzinfo=timezone.utc),
        "target": "opening_weekend",
        "point_usd": 100,
        "lo80_usd": 80,
        "hi80_usd": 120,
        "lo95_usd": 70,
        "hi95_usd": 140,
        "point_model": "latest_pre_release_carry_forward",
        "interval_model": "latest_pre_release_carry_forward",
        "component_source": "latest_pre_release_carry_forward",
        "actual_usd": None,
        "is_live": True,
        "is_backtest": False,
    }
    component_row = {
        "forecast_id": "forecast_1",
        "component_day": "Friday",
        "component_type": "baseline",
        "component_point_usd": 40,
        "component_sigma_log": 0.2,
        "component_model": "daily_shape_model",
        "component_source": "latest_pre_release_carry_forward",
    }
    conn = Connection()

    write_forecast_rows(conn, [forecast_row], [component_row])

    emission_batches = [item for item in conn.batches if "movie_forecast_emissions" in item[0]]
    component_batches = [item for item in conn.batches if "movie_forecast_emission_components" in item[0]]
    assert len(emission_batches) == 1
    assert len(component_batches) == 1
    assert "ON CONFLICT (model_version, release_run_id, origin_key, target, as_of_utc, forecast_role)" in emission_batches[0][0]
    assert emission_batches[0][1][0][3] == "production"
    assert len(str(emission_batches[0][1][0][4])) == 64
    assert emission_batches[0][1][0][5] == "friday_no_amc_pre_release_carry_forward"
    assert component_batches[0][1][0][0] == emission_batches[0][1][0][0]


def test_write_forecast_rows_rejects_same_logical_key_with_changed_payload() -> None:
    class Cursor:
        def __init__(self, row: tuple[object, ...] | None = None) -> None:
            self.row = row

        def fetchall(self) -> list[tuple[object, ...]]:
            return []

        def fetchone(self) -> tuple[object, ...] | None:
            return self.row

    class Connection:
        def __init__(self) -> None:
            self.conflict_recorded = False

        def execute(self, sql: str, params: tuple[object, ...] | None = None) -> Cursor:
            if "SELECT payload_hash" in sql:
                return Cursor(("different_existing_hash",))
            if "movie_forecast_emission_conflicts" in sql:
                self.conflict_recorded = True
                return Cursor()
            return Cursor([])

        def executemany(self, sql: str, rows: list[tuple[object, ...]]) -> None:
            raise AssertionError("conflicting payload should fail before writes")

    forecast_row = {
        "forecast_id": "forecast_1",
        "run_id": "run_1",
        "model_version": "boxoffice_test",
        "movie_id": 1,
        "release_run_id": 10,
        "title": "Example",
        "opening_weekend_start": date(2026, 7, 10),
        "regime": "live_friday",
        "origin_key": "FRI_20:00",
        "forecast_origin_local": datetime(2026, 7, 10, 20, tzinfo=timezone.utc),
        "forecast_origin_utc": datetime(2026, 7, 10, 20, tzinfo=timezone.utc),
        "as_of_utc": datetime(2026, 7, 10, 20, tzinfo=timezone.utc),
        "target": "opening_weekend",
        "point_usd": 101,
        "lo80_usd": 80,
        "hi80_usd": 120,
        "lo95_usd": 70,
        "hi95_usd": 140,
        "component_source": "daily_baseline",
        "is_live": True,
    }
    conn = Connection()

    with pytest.raises(RuntimeError, match="idempotency conflict"):
        write_forecast_rows(conn, [forecast_row], [])

    assert conn.conflict_recorded


def test_log_normal_interval_uses_textbook_multipliers() -> None:
    interval = log_normal_interval(100.0, 0.2)

    assert math.isclose(interval["lo80_usd"], 100.0 * math.exp(-1.28155 * 0.2))
    assert math.isclose(interval["hi95_usd"], 100.0 * math.exp(1.95996 * 0.2))


def test_pre_release_forecast_outputs_canonical_rows() -> None:
    movie = MovieOpening(1, 10, "Example", date(2026, 7, 10))
    origin = build_forecast_origins(movie, mode="historical", include_live=False)[-1]
    artifacts = ModelArtifacts(
        model_version="boxoffice_test",
        artifact_dir=Path("models/boxoffice/boxoffice_test"),
        manifest={},
        pre_release_point_policy={"default": "primary_point_forecast_usd"},
        pre_release_interval_policy={"default_sigma_log": 0.4},
        pre_release_panel=pd.DataFrame(
            [
                {
                    "movie_id": 1,
                    "release_run_id": 10,
                    "origin_day": -1,
                    "primary_point_forecast_usd": 100_000_000,
                    "actual_opening_weekend_gross_usd": 110_000_000,
                    "friday_share": 0.4,
                    "saturday_share": 0.35,
                    "sunday_share": 0.25,
                    "source_count": 3,
                    "estimate_sources": "a|b|c",
                }
            ]
        ),
    )

    results = forecast_pre_release_opening_weekend(
        movie=movie,
        origin=origin,
        artifacts=artifacts,
        run_id="run_1",
        is_backtest=True,
    )
    forecast_row = results[0].to_forecast_row()
    component_rows = results[0].to_component_rows()

    assert [result.target for result in results] == ["opening_weekend", "friday", "saturday", "sunday"]
    assert forecast_row["origin_key"] == "P_-1"
    assert forecast_row["target"] == "opening_weekend"
    assert forecast_row["log_error"] is not None
    assert len(component_rows) == 3
    distribution = ForecastDistribution.from_payload(forecast_row["distribution_payload"])
    assert distribution is not None
    assert len(distribution.quantile_levels) == 201
    assert distribution.quantile(0.5) == pytest.approx(100_000_000, abs=100_000)


def test_pre_release_forecast_prefers_selected_interval_panel_bounds() -> None:
    movie = MovieOpening(1, 10, "Example", date(2026, 7, 10))
    origin = build_forecast_origins(movie, mode="historical", include_live=False)[-1]
    artifacts = ModelArtifacts(
        model_version="boxoffice_test",
        artifact_dir=Path("models/boxoffice/boxoffice_test"),
        manifest={},
        pre_release_point_policy={"default": "primary_point_forecast_usd"},
        pre_release_interval_policy={"default_sigma_log": 3.0},
        pre_release_panel=pd.DataFrame(
            [
                {
                    "movie_id": 1,
                    "release_run_id": 10,
                    "origin_day": -1,
                    "primary_point_forecast_usd": 100_000_000,
                    "selected_interval_model": "global_sigma",
                    "selected_lo_80": 60_000_000,
                    "selected_hi_80": 150_000_000,
                    "selected_lo_95": 40_000_000,
                    "selected_hi_95": 210_000_000,
                    "actual_opening_weekend_gross_usd": 110_000_000,
                }
            ]
        ),
    )

    results = forecast_pre_release_opening_weekend(
        movie=movie,
        origin=origin,
        artifacts=artifacts,
        run_id="run_1",
        is_backtest=True,
    )
    forecast_row = results[0].to_forecast_row()

    assert forecast_row["lo95_usd"] == 40_000_000
    assert forecast_row["hi95_usd"] == 210_000_000
    assert forecast_row["interval_model"] == "selected_interval_global_sigma"


def test_interval_policy_uses_selected_interval_panel_sigmas(tmp_path: Path) -> None:
    panel = tmp_path / "interval_panel.csv"
    pd.DataFrame(
        [
            {
                "origin_day": -14,
                "primary_point_forecast_usd": 100_000_000,
                "selected_lo_95": 44_000_000,
                "selected_hi_95": 228_000_000,
                "sigma_global": 0.42,
                "actual_opening_weekend_gross_usd": 1_000,
            }
        ]
    ).to_csv(panel, index=False)

    policy = build_interval_policy(panel)

    assert policy["method"] == "movie_weighted_conditional_centered_empirical_log_residual_quantile_shrunk"
    assert policy["required_interval_model"] == "movie_weighted_conditional_centered_empirical_log_residual_quantile_shrunk"
    assert "conditional_log_residual_quantiles" in policy
    assert math.isclose(policy["default_sigma_log"], 0.42)
    assert 0.4 < policy["sigma_log_by_origin_day"]["-14"] < 0.5


def test_interval_policy_keeps_raw_pre_release_upper_tail(tmp_path: Path) -> None:
    panel = tmp_path / "interval_panel.csv"
    rows = []
    for idx, ratio in enumerate([0.4, 0.6, 0.8, 1.0, 1.2, 3.0]):
        rows.append(
            {
                "origin_day": -14,
                "primary_point_forecast_usd": 100_000_000,
                "selected_lo_95": 40_000_000,
                "selected_hi_95": 220_000_000,
                "actual_opening_weekend_gross_usd": 100_000_000 * ratio,
                "release_run_id": idx,
            }
        )
    pd.DataFrame(rows).to_csv(panel, index=False)

    policy = build_interval_policy(panel)
    quantiles = policy["log_residual_quantiles_by_origin_day"]["-14"]

    assert quantiles["method"] == "empirical_log_residual_quantile"
    assert math.exp(quantiles["hi80_log"]) > 1.30
    assert math.exp(quantiles["hi95_log"]) > 1.60
    assert "statistical_interval_note" in policy


def test_pre_release_forecast_prefers_empirical_quantile_policy() -> None:
    movie = MovieOpening(1, 10, "Example", date(2026, 7, 10))
    origin = build_forecast_origins(movie, mode="historical", include_live=False)[-1]
    artifacts = ModelArtifacts(
        model_version="boxoffice_test",
        artifact_dir=Path("models/boxoffice/boxoffice_test"),
        manifest={},
        pre_release_point_policy={"default": "primary_point_forecast_usd"},
        pre_release_interval_policy={
            "log_residual_quantiles_by_origin_day": {
                "-1": {"lo80_log": -0.2, "hi80_log": 0.1, "lo95_log": -0.5, "hi95_log": 0.4}
            }
        },
        pre_release_panel=pd.DataFrame(
            [
                {
                    "movie_id": 1,
                    "release_run_id": 10,
                    "origin_day": -1,
                    "primary_point_forecast_usd": 100_000_000,
                    "selected_interval_model": "global_sigma",
                    "selected_lo_80": 10_000_000,
                    "selected_hi_80": 300_000_000,
                    "selected_lo_95": 5_000_000,
                    "selected_hi_95": 500_000_000,
                }
            ]
        ),
    )

    results = forecast_pre_release_opening_weekend(
        movie=movie,
        origin=origin,
        artifacts=artifacts,
        run_id="run_1",
        is_backtest=True,
    )

    assert math.isclose(results[0].hi95_usd, 100_000_000 * math.exp(0.4))
    assert results[0].interval_model == "pre_release_empirical_quantile"


def test_pre_release_conditional_policy_uses_raw_empirical_interval_without_upper_cap() -> None:
    movie = MovieOpening(1, 10, "Example", date(2026, 7, 10))
    origin = build_forecast_origins(movie, mode="historical", include_live=False)[-1]
    artifacts = ModelArtifacts(
        model_version="boxoffice_test",
        artifact_dir=Path("models/boxoffice/boxoffice_test"),
        manifest={},
        pre_release_point_policy={"default": "primary_point_forecast_usd"},
        pre_release_interval_policy={
            "conditional_log_residual_quantiles": {
                "method": "movie_weighted_conditional_centered_empirical_log_residual_quantile_shrunk",
                "scopes": [
                    {
                        "group_cols": ["origin_bucket", "source_count_bucket", "point_bucket", "release_scope_bucket"],
                        "cells": {
                            "P_-1|one_source|50m_plus|wide_or_large_wide": {
                                "center_log": 0.0,
                                "lo80_centered_log": -0.20,
                                "hi80_centered_log": math.log(1.80),
                                "lo95_centered_log": -0.40,
                                "hi95_centered_log": math.log(2.20),
                            }
                        },
                    }
                ],
            },
            "required_interval_model": "movie_weighted_conditional_centered_empirical_log_residual_quantile_shrunk",
        },
        pre_release_panel=pd.DataFrame(
            [
                {
                    "movie_id": 1,
                    "release_run_id": 10,
                    "origin_day": -1,
                    "primary_point_forecast_usd": 100_000_000,
                    "source_count": 1,
                    "release_width_bucket": "wide",
                    "release_type": "movie_page_full_run",
                }
            ]
        ),
    )

    results = forecast_pre_release_opening_weekend(
        movie=movie,
        origin=origin,
        artifacts=artifacts,
        run_id="run_1",
        is_backtest=True,
    )

    assert math.isclose(results[0].hi80_usd, 180_000_000)
    assert math.isclose(results[0].hi95_usd, 220_000_000)
    assert results[0].interval_model == "pre_release_raw_conditional_centered_quantile"


def test_daily_interval_policy_builds_empirical_quantiles(tmp_path: Path) -> None:
    baseline = tmp_path / "daily.csv"
    pd.DataFrame(
        [
            {
                "pre_fri_usd": 100,
                "actual_fri_usd": 110,
                "pre_sat_usd": 100,
                "actual_sat_usd": 90,
                "pre_sun_usd": 100,
                "actual_sun_usd": 120,
                "after_fri_sat_usd": 100,
                "after_fri_sun_usd": 100,
                "after_sat_sun_usd": 100,
                "actual_ow_usd": 300,
            }
            for _ in range(5)
        ]
    ).to_csv(baseline, index=False)

    policy = build_daily_interval_policy(baseline)

    assert policy["method"] == "empirical_daily_log_residual_quantile"
    assert policy["by_regime_day"]["live_friday"]["Friday"]["n"] == 5
    assert policy["by_regime_day"]["live_friday"]["Friday"]["hi95_log"] > 0
    joint = policy["joint_by_regime_days"]["live_saturday"]["Saturday|Sunday"]
    assert joint["method"] == "joint_empirical_daily_log_residual_bootstrap"
    assert joint["days"] == ["Saturday", "Sunday"]
    assert joint["n"] == 5
    assert joint["residual_samples_log"][0]["Saturday"] == pytest.approx(math.log(0.9))
    assert joint["residual_samples_log"][0]["Sunday"] == pytest.approx(math.log(1.2))
    conditional = policy["conditional_by_regime_day"]["live_sunday"]["Sunday"]
    assert conditional["method"] == "signed_saturday_surprise_weighted_empirical_log_residual_bootstrap"
    assert conditional["n"] == 5
    assert conditional["residual_samples"][0]["conditioning_value"] == pytest.approx(math.log(0.9))
    assert conditional["residual_samples"][0]["log_residual"] == pytest.approx(math.log(1.2))


def test_live_composer_uses_known_actuals_and_plugin_current_day() -> None:
    movie = MovieOpening(1, 10, "Example", date(2026, 7, 10))
    origin = [item for item in build_forecast_origins(movie, mode="historical") if item.origin_key == "SAT_14:00"][0]
    artifacts = ModelArtifacts(
        model_version="boxoffice_test",
        artifact_dir=Path("models/boxoffice/boxoffice_test"),
        manifest={"composition": {"n_sim": 1000, "fallback_daily_sigma_log": 0.25}},
        daily_baseline=pd.DataFrame(
            [
                {
                    "movie_id": 1,
                    "release_run_id": 10,
                    "title": "Example",
                    "release_date": "2026-07-10",
                    "pre_fri_usd": 30_000_000,
                    "pre_sat_usd": 35_000_000,
                    "pre_sun_usd": 25_000_000,
                    "after_fri_sat_usd": 40_000_000,
                    "after_fri_sun_usd": 20_000_000,
                    "after_sat_sun_usd": 18_000_000,
                    "actual_fri_usd": 32_000_000,
                    "actual_sat_usd": 41_000_000,
                    "actual_sun_usd": 19_000_000,
                    "actual_ow_usd": 92_000_000,
                }
            ]
        ),
        live_plugin_nowcasts=pd.DataFrame(
            [
                {
                    "movie_id": 1,
                    "regime": "live_saturday",
                    "forecast_origin": "14:00",
                    "target_day": "Saturday",
                    "pred_daily_gross_usd": 42_000_000,
                    "sigma_log_daily": 0.2,
                    "source": "AMC_plugin",
                }
            ]
        ),
    )

    results = compose_live_weekend_forecast(
        movie=movie,
        origin=origin,
        artifacts=artifacts,
        run_id="run_1",
        is_live=False,
        is_backtest=True,
        seed=3,
    )
    weekend = results[0]
    component_types = {component.component_day: component.component_type for component in weekend.components}

    assert weekend.target == "opening_weekend"
    assert component_types == {"Friday": "actual", "Saturday": "AMC_nowcast", "Sunday": "baseline"}
    assert weekend.point_usd > 0


def test_live_composer_uses_empirical_daily_interval_policy() -> None:
    movie = MovieOpening(1, 10, "Example", date(2026, 7, 10))
    origin = [item for item in build_forecast_origins(movie, mode="historical") if item.origin_key == "SUN_14:00"][0]
    artifacts = ModelArtifacts(
        model_version="boxoffice_test",
        artifact_dir=Path("models/boxoffice/boxoffice_test"),
        manifest={"composition": {"n_sim": 5000, "fallback_daily_sigma_log": 0.45}},
        daily_baseline=pd.DataFrame(
            [
                {
                    "movie_id": 1,
                    "release_run_id": 10,
                    "title": "Example",
                    "release_date": "2026-07-10",
                    "actual_fri_usd": 30_000_000,
                    "actual_sat_usd": 40_000_000,
                    "actual_sun_usd": 20_000_000,
                    "actual_ow_usd": 90_000_000,
                    "after_sat_sun_usd": 20_000_000,
                }
            ]
        ),
        daily_interval_policy={
            "by_regime_day": {
                "live_sunday": {
                    "Sunday": {
                        "sigma_log": 0.12,
                        "lo80_log": -0.10,
                        "hi80_log": 0.10,
                        "lo95_log": -0.20,
                        "hi95_log": 0.20,
                        "residual_samples_log": [-0.2, -0.1, 0.0, 0.1, 0.2],
                    }
                }
            }
        },
    )

    results = compose_live_weekend_forecast(
        movie=movie,
        origin=origin,
        artifacts=artifacts,
        run_id="run_1",
        is_live=False,
        is_backtest=True,
        seed=3,
    )
    sunday = [result for result in results if result.target == "sunday"][0]

    assert sunday.interval_model == "empirical_daily_quantile"
    assert math.isclose(sunday.hi95_usd, 20_000_000 * math.exp(0.2))
    assert results[0].interval_model == "empirical_component_residual_simulation"
    assert results[0].lo80_usd <= results[0].point_usd <= results[0].hi80_usd
    assert results[0].point_model == "live_weekend_simulated_median"


def test_live_composer_uses_signed_saturday_conditioned_sunday_residuals() -> None:
    movie = MovieOpening(1, 10, "Example", date(2026, 7, 10))
    origin = [item for item in build_forecast_origins(movie, mode="historical") if item.origin_key == "SUN_14:00"][0]
    artifacts = ModelArtifacts(
        model_version="boxoffice_test",
        artifact_dir=Path("models/boxoffice/boxoffice_test"),
        manifest={"composition": {"n_sim": 5000, "fallback_daily_sigma_log": 0.45}},
        daily_baseline=pd.DataFrame(
            [
                {
                    "movie_id": 1,
                    "release_run_id": 10,
                    "title": "Example",
                    "release_date": "2026-07-10",
                    "actual_fri_usd": 30_000_000,
                    "actual_sat_usd": 45_000_000,
                    "after_fri_sat_usd": 40_000_000,
                    "after_sat_sun_usd": 20_000_000,
                    "actual_sun_usd": 22_000_000,
                    "actual_ow_usd": 97_000_000,
                }
            ]
        ),
        daily_interval_policy={
            "conditional_by_regime_day": {
                "live_sunday": {
                    "Sunday": {
                        "method": "signed_saturday_surprise_weighted_empirical_log_residual_bootstrap",
                        "bandwidth": 0.35,
                        "shrink_k": 20.0,
                        "residual_samples": [
                            {"conditioning_value": math.log(45_000_000 / 40_000_000), "log_residual": math.log(1.2)},
                            {"conditioning_value": math.log(45_000_000 / 40_000_000), "log_residual": math.log(1.1)},
                            {"conditioning_value": math.log(30_000_000 / 40_000_000), "log_residual": math.log(0.7)},
                            {"conditioning_value": math.log(60_000_000 / 40_000_000), "log_residual": math.log(1.5)},
                            {"conditioning_value": math.log(40_000_000 / 40_000_000), "log_residual": math.log(1.0)},
                        ],
                    }
                }
            },
            "by_regime_day": {
                "live_sunday": {
                    "Sunday": {
                        "sigma_log": 0.12,
                        "lo80_log": -0.10,
                        "hi80_log": 0.10,
                        "lo95_log": -0.20,
                        "hi95_log": 0.20,
                        "residual_samples_log": [math.log(0.5), math.log(0.6), math.log(0.7), math.log(0.8), math.log(0.9)],
                    }
                }
            },
        },
    )

    results = compose_live_weekend_forecast(
        movie=movie,
        origin=origin,
        artifacts=artifacts,
        run_id="run_1",
        is_live=False,
        is_backtest=True,
        seed=3,
    )
    weekend = results[0]

    assert weekend.interval_model == "signed_saturday_conditioned_sunday_residual_simulation"
    assert weekend.point_usd > 30_000_000 + 45_000_000 + 20_000_000


def test_live_composer_keeps_after_friday_joint_pairs_shadow_only() -> None:
    movie = MovieOpening(1, 10, "Example", date(2026, 7, 10))
    origin = [item for item in build_forecast_origins(movie, mode="historical") if item.origin_key == "SAT_10:00"][0]
    artifacts = ModelArtifacts(
        model_version="boxoffice_test",
        artifact_dir=Path("models/boxoffice/boxoffice_test"),
        manifest={"composition": {"n_sim": 5000, "fallback_daily_sigma_log": 0.45}},
        daily_baseline=pd.DataFrame(
            [
                {
                    "movie_id": 1,
                    "release_run_id": 10,
                    "title": "Example",
                    "release_date": "2026-07-10",
                    "actual_fri_usd": 30_000_000,
                    "after_fri_sat_usd": 40_000_000,
                    "after_fri_sun_usd": 20_000_000,
                    "actual_ow_usd": 90_000_000,
                }
            ]
        ),
        daily_interval_policy={
            "joint_by_regime_days": {
                "live_saturday": {
                    "Saturday|Sunday": {
                        "method": "joint_empirical_daily_log_residual_bootstrap",
                        "days": ["Saturday", "Sunday"],
                        "residual_samples_log": [
                            {"Saturday": math.log(0.5), "Sunday": math.log(0.5)},
                            {"Saturday": math.log(1.5), "Sunday": math.log(1.5)},
                        ],
                    }
                }
            }
        },
    )

    results = compose_live_weekend_forecast(
        movie=movie,
        origin=origin,
        artifacts=artifacts,
        run_id="run_1",
        is_live=False,
        is_backtest=True,
        seed=3,
    )
    weekend = results[0]

    assert weekend.interval_model == "empirical_component_residual_simulation"
    assert weekend.lo80_usd < weekend.point_usd < weekend.hi80_usd


def test_live_composer_uses_opening_friday_amc_nowcast_when_available() -> None:
    movie = MovieOpening(1, 10, "Example", date(2026, 7, 10))
    origin = [item for item in build_forecast_origins(movie, mode="historical") if item.origin_key == "FRI_16:00"][0]
    artifacts = ModelArtifacts(
        model_version="boxoffice_test",
        artifact_dir=Path("models/boxoffice/boxoffice_test"),
        manifest={"composition": {"n_sim": 5000, "fallback_daily_sigma_log": 0.45}},
        daily_baseline=pd.DataFrame(
            [
                {
                    "movie_id": 1,
                    "release_run_id": 10,
                    "title": "Example",
                    "release_date": "2026-07-10",
                    "pre_fri_usd": 30_000_000,
                    "pre_sat_usd": 35_000_000,
                    "pre_sun_usd": 25_000_000,
                }
            ]
        ),
        live_plugin_nowcasts=pd.DataFrame(
            [
                {
                    "movie_id": 1,
                    "regime": "live_friday",
                    "forecast_origin": "16:00",
                    "target_day": "Friday",
                    "sample_key": "top_hybrid_30",
                    "pred_daily_gross_usd": 32_000_000,
                    "sigma_log_daily": 0.9,
                    "source": "AMC_live_db",
                    "feature_quality_bucket": "high",
                }
            ]
        ),
        daily_interval_policy={
            "by_regime_day": {
                "live_friday": {
                    "Friday": {
                        "sigma_log": 1.5,
                        "lo80_log": -1.0,
                        "hi80_log": 1.0,
                        "lo95_log": -2.0,
                        "hi95_log": 2.0,
                        "residual_samples_log": [-2.0, -1.0, 0.0, 1.0, 2.0],
                    }
                }
            }
        },
        amc_interval_policy={
            "sample_key": "top_hybrid_30",
            "cells": {
                "sample_key=top_hybrid_30|regime=live_friday|forecast_origin=16:00|target_day=Friday|feature_quality_bucket=high": {
                    "sigma_log": 0.1,
                    "lo80_log": -0.1,
                    "hi80_log": 0.1,
                    "lo95_log": -0.2,
                    "hi95_log": 0.2,
                    "residual_samples_log": [-0.2, -0.1, 0.0, 0.1, 0.2],
                }
            },
        },
    )

    results = compose_live_weekend_forecast(
        movie=movie,
        origin=origin,
        artifacts=artifacts,
        run_id="run_1",
        is_live=False,
        is_backtest=True,
        seed=3,
    )
    friday = [result for result in results if result.target == "friday"][0]

    assert friday.point_usd == 32_000_000
    assert friday.component_source == "AMC_live_db"
    assert friday.interval_model == "empirical_amc_quantile_with_generic_floor"
    assert results[0].lo80_usd <= results[0].point_usd <= results[0].hi80_usd
    assert results[0].point_model == "live_weekend_simulated_median"
    assert results[0].interval_model == "amc_empirical_component_residual_simulation"


def _friday_fallback_artifacts(*, include_d1: bool = True, include_d2: bool = True) -> ModelArtifacts:
    rows = []
    if include_d2:
        rows.append(
            {
                "movie_id": 1,
                "release_run_id": 10,
                "origin_day": -2,
                "primary_point_forecast_usd": 80_000_000,
                "selected_interval_model": "d2",
                "selected_lo_80": 60_000_000,
                "selected_hi_80": 120_000_000,
                "selected_lo_95": 45_000_000,
                "selected_hi_95": 150_000_000,
                "actual_opening_weekend_gross_usd": 90_000_000,
                "friday_share": 0.4,
                "saturday_share": 0.35,
                "sunday_share": 0.25,
            }
        )
    if include_d1:
        rows.append(
            {
                "movie_id": 1,
                "release_run_id": 10,
                "origin_day": -1,
                "primary_point_forecast_usd": 100_000_000,
                "selected_interval_model": "d1",
                "selected_lo_80": 70_000_000,
                "selected_hi_80": 140_000_000,
                "selected_lo_95": 50_000_000,
                "selected_hi_95": 180_000_000,
                "actual_opening_weekend_gross_usd": 90_000_000,
                "friday_share": 0.4,
                "saturday_share": 0.35,
                "sunday_share": 0.25,
            }
        )
    return ModelArtifacts(
        model_version="boxoffice_test",
        artifact_dir=Path("models/boxoffice/boxoffice_test"),
        manifest={"composition": {"n_sim": 1000, "fallback_daily_sigma_log": 0.3}},
        pre_release_point_policy={"default": "primary_point_forecast_usd"},
        pre_release_interval_policy={"default_sigma_log": 0.4},
        pre_release_panel=pd.DataFrame(rows),
        daily_baseline=pd.DataFrame(
            [
                {
                    "movie_id": 1,
                    "release_run_id": 10,
                    "pre_fri_usd": 30_000_000,
                    "pre_sat_usd": 35_000_000,
                    "pre_sun_usd": 25_000_000,
                    "actual_ow_usd": 90_000_000,
                }
            ]
        ),
    )


def test_live_friday_no_amc_carries_forward_latest_pre_release_opening_weekend() -> None:
    movie = MovieOpening(1, 10, "Example", date(2026, 7, 10))
    origin = [item for item in build_forecast_origins(movie, mode="historical") if item.origin_key == "FRI_20:00"][0]

    weekend = compose_live_weekend_forecast(
        movie=movie,
        origin=origin,
        artifacts=_friday_fallback_artifacts(),
        run_id="run_1",
        is_live=False,
        is_backtest=True,
        seed=3,
    )[0]

    assert weekend.component_source == "latest_pre_release_carry_forward"
    assert weekend.point_usd == 100_000_000
    assert weekend.lo80_usd == 70_000_000
    assert weekend.hi80_usd == 140_000_000
    assert weekend.lo95_usd == 50_000_000
    assert weekend.hi95_usd == 180_000_000
    assert weekend.audit["carried_origin_key"] == "P_-1"


def test_live_friday_no_amc_falls_back_to_d2_when_d1_missing() -> None:
    movie = MovieOpening(1, 10, "Example", date(2026, 7, 10))
    origin = [item for item in build_forecast_origins(movie, mode="historical") if item.origin_key == "FRI_20:00"][0]

    weekend = compose_live_weekend_forecast(
        movie=movie,
        origin=origin,
        artifacts=_friday_fallback_artifacts(include_d1=False),
        run_id="run_1",
        is_live=False,
        is_backtest=True,
        seed=3,
    )[0]

    assert weekend.point_usd == 80_000_000
    assert weekend.audit["carried_origin_key"] == "P_-2"


def test_live_friday_no_amc_without_pre_release_fails_explicitly() -> None:
    movie = MovieOpening(1, 10, "Example", date(2026, 7, 10))
    origin = [item for item in build_forecast_origins(movie, mode="historical") if item.origin_key == "FRI_20:00"][0]

    with pytest.raises(ValueError, match="requires eligible D-1 or D-2"):
        compose_live_weekend_forecast(
            movie=movie,
            origin=origin,
            artifacts=_friday_fallback_artifacts(include_d1=False, include_d2=False),
            run_id="run_1",
            is_live=False,
            is_backtest=True,
            seed=3,
        )


def test_live_friday_actual_available_uses_live_composition_not_pre_release_carry_forward() -> None:
    movie = MovieOpening(1, 10, "Example", date(2026, 7, 10))
    origin = [item for item in build_forecast_origins(movie, mode="historical") if item.origin_key == "FRI_20:00"][0]
    artifacts = _friday_fallback_artifacts()
    artifacts.daily_baseline.loc[0, "actual_fri_usd"] = 32_000_000
    artifacts.daily_baseline.loc[0, "actual_fri_usd_available_as_of"] = True

    weekend = compose_live_weekend_forecast(
        movie=movie,
        origin=origin,
        artifacts=artifacts,
        run_id="run_1",
        is_live=False,
        is_backtest=True,
        seed=3,
    )[0]

    assert weekend.component_source == "daily_baseline"
    assert weekend.point_model == "live_weekend_simulated_median"
    assert any(component.component_day == "Friday" and component.component_type == "actual" for component in weekend.components)


def test_all_reported_actuals_collapse_weekend_distribution() -> None:
    movie = MovieOpening(1, 10, "Example", date(2026, 7, 10))
    origin = [item for item in build_forecast_origins(movie, mode="historical") if item.origin_key == "SUN_EOD"][0]
    artifacts = ModelArtifacts(
        model_version="boxoffice_test",
        artifact_dir=Path("models/boxoffice/boxoffice_test"),
        manifest={"composition": {"n_sim": 1000}},
        daily_baseline=pd.DataFrame(
            [{
                "movie_id": 1, "release_run_id": 10,
                "actual_fri_usd": 30, "actual_sat_usd": 40,
                "actual_sun_usd": 20, "actual_ow_usd": 90,
                "actual_fri_usd_available_as_of": True,
                "actual_sat_usd_available_as_of": True,
                "actual_sun_usd_available_as_of": True,
            }]
        ),
    )

    weekend = compose_live_weekend_forecast(
        movie=movie, origin=origin, artifacts=artifacts, run_id="run_1",
        is_live=False, is_backtest=True, seed=3,
    )[0]

    assert {component.component_type for component in weekend.components} == {"actual"}
    assert weekend.lo95_usd == weekend.lo80_usd == weekend.point_usd == weekend.hi80_usd == weekend.hi95_usd == 90


def test_live_simulation_is_deterministic_for_identical_inputs() -> None:
    movie = MovieOpening(1, 10, "Example", date(2026, 7, 10))
    origin = [item for item in build_forecast_origins(movie, mode="historical") if item.origin_key == "SAT_10:00"][0]
    artifacts = ModelArtifacts(
        model_version="boxoffice_test",
        artifact_dir=Path("models/boxoffice/boxoffice_test"),
        manifest={"composition": {"n_sim": 1000, "fallback_daily_sigma_log": 0.3}},
        daily_baseline=pd.DataFrame(
            [{
                "movie_id": 1, "release_run_id": 10,
                "actual_fri_usd": 30, "after_fri_sat_usd": 40, "after_fri_sun_usd": 20,
            }]
        ),
    )
    kwargs = dict(movie=movie, origin=origin, artifacts=artifacts, run_id="run_1", is_live=False, is_backtest=True)

    first = compose_live_weekend_forecast(**kwargs)[0]
    second = compose_live_weekend_forecast(**kwargs)[0]

    assert (first.point_usd, first.lo80_usd, first.hi80_usd, first.lo95_usd, first.hi95_usd) == (
        second.point_usd, second.lo80_usd, second.hi80_usd, second.lo95_usd, second.hi95_usd
    )


def test_live_cdf_policy_emits_a_reproducible_payload_and_after_friday_state() -> None:
    movie = MovieOpening(1, 10, "Example", date(2026, 7, 10))
    origin = [item for item in build_forecast_origins(movie, mode="historical") if item.origin_key == "FRI_20:00"][0]
    artifacts = ModelArtifacts(
        model_version="boxoffice_test",
        artifact_dir=Path("."),
        manifest={"composition": {"fallback_daily_sigma_log": 0.1}},
        live_weekend_distribution_policy={"simulation_draw_count": 1000, "policy_version": "test-v1"},
        daily_baseline=pd.DataFrame([{
            "movie_id": 1, "release_run_id": 10,
            "pre_fri_usd": 30_000_000, "pre_sat_usd": 35_000_000, "pre_sun_usd": 25_000_000,
            "actual_fri_usd": 31_000_000, "actual_fri_usd_available_as_of": True,
            "after_fri_sat_usd": 42_000_000, "after_fri_sun_usd": 21_000_000,
        }]),
    )
    first = compose_live_weekend_forecast(movie=movie, origin=origin, artifacts=artifacts, run_id="run_1", is_live=False, is_backtest=True)[0]
    second = compose_live_weekend_forecast(movie=movie, origin=origin, artifacts=artifacts, run_id="run_2", is_live=False, is_backtest=True)[0]
    payload = first.distribution_payload

    assert payload is not None
    assert payload["information_state"] == "after_friday"
    assert len(payload["quantile_levels"]) == len(payload["quantile_values_usd"]) == 201
    assert first.point_usd == payload["quantile_values_usd"][100]
    assert first.lo80_usd == payload["quantile_values_usd"][19]
    assert first.hi80_usd == payload["quantile_values_usd"][181]
    assert payload["payload_hash"] == second.distribution_payload["payload_hash"]
    distribution = ForecastDistribution.from_payload(payload)
    assert distribution is not None
    assert {component.component_day: component.component_point_usd for component in first.components} == pytest.approx(
        {"Friday": 31_000_000, "Saturday": 42_000_000, "Sunday": 21_000_000}
    )
