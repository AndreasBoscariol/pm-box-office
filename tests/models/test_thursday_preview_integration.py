from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from models.boxoffice.artifacts import ModelArtifacts
from models.boxoffice.live_composition import build_live_components, compose_live_weekend_forecast
from models.boxoffice.origins import build_forecast_origins
from models.boxoffice.schema import MovieOpening
from models.boxoffice.thursday_amc_preview import build_thursday_amc_preview_shadow
from models.boxoffice.thursday_preview import update_ow_prior_from_reported_preview


def _movie() -> MovieOpening:
    return MovieOpening(1, 10, "Example", date(2026, 7, 10))


def _origin(key: str):
    return [item for item in build_forecast_origins(_movie(), mode="historical") if item.origin_key == key][0]


def _policy() -> dict[str, object]:
    return {
        "policy_version": "test_preview_v1",
        "alpha": math.log(1.2),
        "beta": 0.0,
        "old_scale_tolerance_pct": 0.05,
        "training_cutoff": "2026-06-26",
        "interval_calibration_status": "legacy_unrecalibrated_for_thursday_preview_update",
    }


def _opening_thursday_policy() -> dict[str, object]:
    return {
        "policy_name": "opening_thursday_actual_ratio_update_prod",
        "policy_version": "test_opening_thursday_actual_v1",
        "production_ow_update_enabled": True,
        "selected_update_model": "ratio_update",
        "training_cutoff": "2026-06-26",
        "parameters": {"alpha": math.log(1.2), "beta": 0.0},
        "residual_policy": {
            "baseline_residuals_log": [-0.2, -0.1, 0.0, 0.1, 0.2],
            "ratio_update_residuals_log": [-0.15, -0.05, 0.0, 0.05, 0.15],
            "bucket_smoothing_alpha": 0.5,
        },
    }


def _artifacts(
    row: dict[str, object],
    *,
    plugin: pd.DataFrame | None = None,
    opening_thursday_policy: dict[str, object] | None = None,
) -> ModelArtifacts:
    return ModelArtifacts(
        model_version="boxoffice_test",
        artifact_dir=Path("models/boxoffice/boxoffice_test"),
        manifest={"composition": {"n_sim": 1000, "fallback_daily_sigma_log": 0.0}},
        thursday_preview_policy=_policy(),
        opening_thursday_actual_policy=opening_thursday_policy or {},
        daily_baseline=pd.DataFrame([{"movie_id": 1, "release_run_id": 10, **row}]),
        live_plugin_nowcasts=plugin if plugin is not None else pd.DataFrame(),
    )


def _preview_row(**overrides: object) -> dict[str, object]:
    row = {
        "pre_fri_usd": 30_000_000,
        "pre_sat_usd": 35_000_000,
        "pre_sun_usd": 25_000_000,
        "total_forecast_usd": 90_000_000,
        "thursday_preview_gross_usd": 8_000_000,
        "forecast_origin_date": "2026-07-09",
        "thursday_preview_cutoff_date": "2026-07-10",
    }
    row.update(overrides)
    return row


def _opening_thursday_row(**overrides: object) -> dict[str, object]:
    row = _preview_row()
    row.update(
        {
            "opening_weekend_start": "2026-07-10",
            "opening_thursday_date": "2026-07-09",
            "opening_thursday_daily_gross_usd": 8_000_000,
            "opening_thursday_source": "the_numbers",
            "opening_thursday_fetched_at": "2026-07-10T12:00:00Z",
        }
    )
    row.update(overrides)
    return row


def test_reported_thursday_preview_policy_rejects_leaky_baseline_timestamp() -> None:
    update = update_ow_prior_from_reported_preview(
        baseline_ow_usd=90_000_000,
        row=pd.Series(_preview_row(forecast_origin_date="2026-07-10")),
        policy=_policy(),
    )

    assert not update.preview_update_applied
    assert update.ow_prior_source == "baseline_consensus"
    assert update.fallback_reason == "baseline_not_pre_preview"


def test_live_friday_no_amc_uses_preview_updated_prior_instead_of_carry_forward() -> None:
    artifacts = _artifacts(_preview_row())

    results = compose_live_weekend_forecast(
        movie=_movie(),
        origin=_origin("FRI_20:00"),
        artifacts=artifacts,
        run_id="run_1",
        is_live=False,
        is_backtest=True,
        seed=3,
    )
    weekend = results[0]

    assert weekend.component_source == "thursday_preview_actual"
    assert weekend.point_model == "live_weekend_simulated_median"
    assert weekend.point_usd == pytest.approx(108_000_000)
    assert weekend.audit["thursday_preview_update_applied"] is True
    assert weekend.audit["ow_prior_baseline_usd"] == pytest.approx(90_000_000)
    assert weekend.audit["ow_prior_usd"] == pytest.approx(108_000_000)
    assert {component.component_day: component.component_point_usd for component in weekend.components} == pytest.approx(
        {"Friday": 36_000_000, "Saturday": 42_000_000, "Sunday": 30_000_000}
    )


def test_live_friday_uses_official_opening_thursday_actual_prior_before_components() -> None:
    artifacts = _artifacts(_opening_thursday_row(), opening_thursday_policy=_opening_thursday_policy())

    weekend = compose_live_weekend_forecast(
        movie=_movie(),
        origin=_origin("FRI_20:00"),
        artifacts=artifacts,
        run_id="run_1",
        is_live=False,
        is_backtest=True,
        seed=3,
    )[0]

    assert weekend.component_source == "opening_thursday_actual_ratio_update_prod"
    assert weekend.audit["opening_thursday_actual_update_applied"] is True
    assert weekend.audit["thursday_preview_update_applied"] is False
    assert weekend.audit["ow_prior_usd"] == pytest.approx(108_000_000)
    assert {component.component_day: component.component_point_usd for component in weekend.components} == pytest.approx(
        {"Friday": 36_000_000, "Saturday": 42_000_000, "Sunday": 30_000_000}
    )


def test_live_friday_amc_replaces_friday_only_after_opening_thursday_actual_prior() -> None:
    plugin = pd.DataFrame(
        [
            {
                "movie_id": 1,
                "regime": "live_friday",
                "forecast_origin": "16:00",
                "target_day": "Friday",
                "pred_daily_gross_usd": 32_000_000,
                "sigma_log_daily": 0.0,
                "source": "AMC_live_db",
            }
        ]
    )
    artifacts = _artifacts(_opening_thursday_row(), plugin=plugin, opening_thursday_policy=_opening_thursday_policy())

    weekend = compose_live_weekend_forecast(
        movie=_movie(),
        origin=_origin("FRI_16:00"),
        artifacts=artifacts,
        run_id="run_1",
        is_live=False,
        is_backtest=True,
        seed=3,
    )[0]

    points = {component.component_day: component.component_point_usd for component in weekend.components}
    types = {component.component_day: component.component_type for component in weekend.components}
    assert types == {"Friday": "AMC_nowcast", "Saturday": "baseline", "Sunday": "baseline"}
    assert points == pytest.approx({"Friday": 32_000_000, "Saturday": 42_000_000, "Sunday": 30_000_000})
    assert weekend.point_usd == pytest.approx(104_000_000)


def test_preview_scale_validation_falls_back_when_pre_components_do_not_match_total() -> None:
    artifacts = _artifacts(_preview_row(total_forecast_usd=120_000_000))

    components = build_live_components(movie=_movie(), origin=_origin("FRI_20:00"), artifacts=artifacts)

    assert {component.component_day: component.component_point_usd for component in components} == {
        "Friday": 30_000_000,
        "Saturday": 35_000_000,
        "Sunday": 25_000_000,
    }


def test_live_friday_amc_replaces_friday_only_after_preview_rescales_weekend_prior() -> None:
    plugin = pd.DataFrame(
        [
            {
                "movie_id": 1,
                "regime": "live_friday",
                "forecast_origin": "16:00",
                "target_day": "Friday",
                "pred_daily_gross_usd": 32_000_000,
                "sigma_log_daily": 0.0,
                "source": "AMC_live_db",
            }
        ]
    )
    artifacts = _artifacts(_preview_row(), plugin=plugin)

    weekend = compose_live_weekend_forecast(
        movie=_movie(),
        origin=_origin("FRI_16:00"),
        artifacts=artifacts,
        run_id="run_1",
        is_live=False,
        is_backtest=True,
        seed=3,
    )[0]

    points = {component.component_day: component.component_point_usd for component in weekend.components}
    types = {component.component_day: component.component_type for component in weekend.components}
    assert types == {"Friday": "AMC_nowcast", "Saturday": "baseline", "Sunday": "baseline"}
    assert points == pytest.approx({"Friday": 32_000_000, "Saturday": 42_000_000, "Sunday": 30_000_000})
    assert weekend.point_usd == pytest.approx(104_000_000)


def test_live_saturday_preview_prior_rescales_remaining_after_friday_actual() -> None:
    artifacts = _artifacts(
        _preview_row(
            actual_fri_usd=30_000_000,
            after_fri_sat_usd=40_000_000,
            after_fri_sun_usd=20_000_000,
        )
    )

    weekend = compose_live_weekend_forecast(
        movie=_movie(),
        origin=_origin("SAT_14:00"),
        artifacts=artifacts,
        run_id="run_1",
        is_live=False,
        is_backtest=True,
        seed=3,
    )[0]

    points = {component.component_day: component.component_point_usd for component in weekend.components}
    assert points == pytest.approx({"Friday": 30_000_000, "Saturday": 52_000_000, "Sunday": 26_000_000})
    assert weekend.point_usd == pytest.approx(108_000_000)


def test_thursday_amc_preview_shadow_uses_policy_without_production_side_effects() -> None:
    shadow = build_thursday_amc_preview_shadow(
        baseline_ow_usd=90_000_000,
        predicted_preview_gross_usd=8_000_000,
        policy=_policy(),
        thursday_amc_seats_collected=True,
    )

    assert shadow["shadow_ow_prior_source"] == "thursday_amc_preview_nowcast"
    assert shadow["shadow_ow_prior_usd"] == pytest.approx(108_000_000)
    assert shadow["shadow_update_applied"] is True
