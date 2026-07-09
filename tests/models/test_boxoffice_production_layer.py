from __future__ import annotations

import math
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from models.boxoffice.artifacts import ModelArtifacts
from models.boxoffice.intervals import log_normal_interval
from models.boxoffice.live_composition import compose_live_weekend_forecast
from models.boxoffice.origins import build_forecast_origins
from models.boxoffice.pre_release import forecast_pre_release_opening_weekend
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

    assert policy["method"] == "selected_interval_panel_with_sigma_fallback"
    assert math.isclose(policy["default_sigma_log"], 0.42)
    assert 0.4 < policy["sigma_log_by_origin_day"]["-14"] < 0.5


def test_interval_policy_caps_pre_release_upper_tail_for_operator_display(tmp_path: Path) -> None:
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

    assert quantiles["method"] == "operational_capped_empirical_log_residual_quantile"
    assert math.isclose(math.exp(quantiles["hi80_log"]), 1.30)
    assert math.isclose(math.exp(quantiles["hi95_log"]), 1.60)


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
