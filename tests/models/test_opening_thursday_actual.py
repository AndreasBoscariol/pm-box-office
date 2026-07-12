from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd
import pytest

from models.boxoffice.opening_thursday_actual import (
    fit_ratio_update_policy,
    update_ow_from_opening_thursday_actual,
)


def _policy() -> dict[str, object]:
    return {
        "policy_name": "opening_thursday_actual_ratio_update_prod",
        "policy_version": "test",
        "production_ow_update_enabled": True,
        "selected_update_model": "ratio_update",
        "training_cutoff": "2026-06-26",
        "parameters": {"alpha": math.log(1.2), "beta": 0.5},
        "residual_policy": {
            "baseline_residuals_log": [-0.2, -0.1, 0.0, 0.1, 0.2],
            "ratio_update_residuals_log": [-0.15, -0.05, 0.0, 0.05, 0.15],
            "bucket_smoothing_alpha": 0.5,
        },
    }


def _row(**overrides: object) -> pd.Series:
    payload = {
        "release_run_id": 10,
        "opening_weekend_start": "2026-07-10",
        "opening_thursday_date": "2026-07-09",
        "opening_thursday_source": "the_numbers",
        "opening_thursday_fetched_at": "2026-07-10T12:00:00Z",
    }
    payload.update(overrides)
    return pd.Series(payload)


def test_update_matches_frozen_ratio_policy_formula() -> None:
    update = update_ow_from_opening_thursday_actual(
        baseline_ow_usd=90_000_000,
        opening_thursday_daily_gross_usd=10_000_000,
        frozen_policy=_policy(),
        row=_row(),
        execution_time=datetime(2026, 7, 10, 13, tzinfo=timezone.utc),
    )

    expected = 90_000_000 * math.exp(math.log(1.2) + 0.5 * math.log(10_000_000 / 90_000_000))
    assert update.production_update_applied is True
    assert update.updated_ow_usd == pytest.approx(expected)
    assert update.update_multiplier == pytest.approx(expected / 90_000_000)


def test_strict_date_eligibility_and_holdover_rejection() -> None:
    mismatched = update_ow_from_opening_thursday_actual(
        baseline_ow_usd=90_000_000,
        opening_thursday_daily_gross_usd=10_000_000,
        frozen_policy=_policy(),
        row=_row(opening_thursday_date="2026-07-08"),
        execution_time=datetime(2026, 7, 10, 13, tzinfo=timezone.utc),
    )
    holdover = update_ow_from_opening_thursday_actual(
        baseline_ow_usd=90_000_000,
        opening_thursday_daily_gross_usd=10_000_000,
        frozen_policy=_policy(),
        row=_row(opening_weekend_start="2026-07-03", opening_thursday_date="2026-07-09"),
        execution_time=datetime(2026, 7, 10, 13, tzinfo=timezone.utc),
    )

    assert mismatched.production_update_applied is False
    assert mismatched.exclusion_reason == "daily_gross_not_opening_thursday"
    assert holdover.production_update_applied is False
    assert holdover.exclusion_reason == "daily_gross_not_opening_thursday"


def test_missing_invalid_or_unavailable_actual_retains_baseline() -> None:
    missing = update_ow_from_opening_thursday_actual(
        baseline_ow_usd=90_000_000,
        opening_thursday_daily_gross_usd=None,
        frozen_policy=_policy(),
        row=_row(),
        execution_time=datetime(2026, 7, 10, 13, tzinfo=timezone.utc),
    )
    unavailable = update_ow_from_opening_thursday_actual(
        baseline_ow_usd=90_000_000,
        opening_thursday_daily_gross_usd=10_000_000,
        frozen_policy=_policy(),
        row=_row(opening_thursday_fetched_at="2026-07-10T14:00:00Z"),
        execution_time=datetime(2026, 7, 10, 13, tzinfo=timezone.utc),
    )

    assert missing.updated_ow_usd == 90_000_000
    assert missing.production_update_applied is False
    assert unavailable.updated_ow_usd == 90_000_000
    assert unavailable.exclusion_reason == "source_record_not_available_by_execution_time"


def test_amc_source_cannot_pass_production_updater() -> None:
    update = update_ow_from_opening_thursday_actual(
        baseline_ow_usd=90_000_000,
        opening_thursday_daily_gross_usd=10_000_000,
        frozen_policy=_policy(),
        row=_row(opening_thursday_source="amc_nowcast"),
        execution_time=datetime(2026, 7, 10, 13, tzinfo=timezone.utc),
    )

    assert update.production_update_applied is False
    assert update.exclusion_reason == "non_the_numbers_source"


def test_distribution_integrity() -> None:
    update = update_ow_from_opening_thursday_actual(
        baseline_ow_usd=90_000_000,
        opening_thursday_daily_gross_usd=10_000_000,
        frozen_policy=_policy(),
        row=_row(),
        execution_time=datetime(2026, 7, 10, 13, tzinfo=timezone.utc),
    )

    dist = update.updated_distribution
    assert dist["lo95_usd"] <= dist["lo80_usd"] <= dist["point_usd"] <= dist["hi80_usd"] <= dist["hi95_usd"]
    assert sum(dist["bucket_probabilities"]) == pytest.approx(1.0)
    cdf = pd.Series(dist["bucket_probabilities"]).cumsum()
    assert cdf.is_monotonic_increasing


def test_freeze_and_inference_identity_for_training_rows() -> None:
    rows = []
    for i in range(25):
        baseline = 100.0 + i
        thursday = 10.0 + i
        actual = baseline * math.exp(0.2 + 0.4 * math.log(thursday / baseline))
        rows.append(
            {
                "release_run_id": i,
                "movie_id": i,
                "opening_weekend_start": pd.Timestamp("2024-01-05") + pd.Timedelta(days=7 * i),
                "baseline_forecast_usd": baseline,
                "opening_thursday_daily_gross_usd": thursday,
                "actual_opening_weekend_gross_usd": actual,
            }
        )
    panel = pd.DataFrame(rows)
    policy = fit_ratio_update_policy(panel, min_train_n=20)

    for row in panel.itertuples(index=False):
        update = update_ow_from_opening_thursday_actual(
            baseline_ow_usd=row.baseline_forecast_usd,
            opening_thursday_daily_gross_usd=row.opening_thursday_daily_gross_usd,
            frozen_policy=policy,
            row={
                "release_run_id": row.release_run_id,
                "opening_weekend_start": row.opening_weekend_start,
                "opening_thursday_date": row.opening_weekend_start - pd.Timedelta(days=1),
                "opening_thursday_source": "the_numbers",
            },
        )
        assert update.updated_ow_usd == pytest.approx(row.actual_opening_weekend_gross_usd)
