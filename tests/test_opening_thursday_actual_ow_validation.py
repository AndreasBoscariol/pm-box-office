from __future__ import annotations

import math

import pandas as pd
import pytest

from eda.During.opening_thursday_actual_ow_validation import (
    build_panel,
    build_rolling_predictions,
    validate_opening_thursday_daily,
)
from models.boxoffice.opening_thursday_actual import update_ow_from_opening_thursday_actual


def _daily_row(
    release_run_id: int,
    movie_id: int,
    *,
    opening_weekend_start: str,
    gross: float = 10.0,
    actual: float = 100.0,
    daily_id: int | None = None,
) -> dict[str, object]:
    return {
        "release_run_id": release_run_id,
        "movie_id": movie_id,
        "title": f"Movie {movie_id}",
        "opening_date": opening_weekend_start,
        "opening_weekend_start": opening_weekend_start,
        "opening_thursday_date": str(pd.Timestamp(opening_weekend_start) - pd.Timedelta(days=1))[:10],
        "release_year": int(opening_weekend_start[:4]),
        "release_width_bucket": "wide",
        "release_type": "movie_page_full_run",
        "opening_weekend_theaters": 3000,
        "actual_opening_weekend_gross_usd": actual,
        "daily_box_office_id": daily_id or release_run_id,
        "opening_thursday_daily_gross_usd": gross,
        "opening_thursday_source": "the_numbers",
        "opening_thursday_source_url": None,
        "opening_thursday_fetched_at": None,
    }


def _baseline_row(
    release_run_id: int,
    movie_id: int,
    *,
    opening_weekend_start: str,
    baseline: float = 100.0,
    forecast_origin_date: str | None = None,
    latest_estimate_date: str | None = None,
) -> dict[str, object]:
    thursday = pd.Timestamp(opening_weekend_start) - pd.Timedelta(days=1)
    safe_date = str(thursday - pd.Timedelta(days=1))[:10]
    return {
        "release_run_id": release_run_id,
        "movie_id": movie_id,
        "origin_day": -2,
        "forecast_origin_date": forecast_origin_date or safe_date,
        "latest_estimate_date": latest_estimate_date or safe_date,
        "primary_point_forecast_usd": baseline,
        "primary_point_method": "test",
        "source_count": 2,
        "selected_lo_80": baseline * 0.7,
        "selected_hi_80": baseline * 1.3,
        "selected_lo_95": baseline * 0.5,
        "selected_hi_95": baseline * 1.8,
    }


def test_opening_thursday_daily_target_uses_day_before_opening_weekend_without_preview_tag() -> None:
    rows = pd.DataFrame([_daily_row(1, 10, opening_weekend_start="2026-07-03", gross=12.0, actual=120.0)])

    summary = validate_opening_thursday_daily(rows)

    assert summary.iloc[0]["target_validation_status"] == "validated_opening_thursday_daily"
    assert summary.iloc[0]["target_definition"] == "opening_thursday_daily_gross"
    assert bool(summary.iloc[0]["target_date_matches_opening_weekend_minus_one"]) is True
    assert summary.iloc[0]["opening_thursday_daily_gross_usd"] == 12.0


def test_panel_selects_only_pre_thursday_baseline() -> None:
    rows = pd.DataFrame([_daily_row(1, 10, opening_weekend_start="2026-07-03", gross=12.0, actual=120.0)])
    panel = pd.DataFrame(
        [
            _baseline_row(1, 10, opening_weekend_start="2026-07-03", baseline=100.0, forecast_origin_date="2026-07-01", latest_estimate_date="2026-07-01"),
            _baseline_row(1, 10, opening_weekend_start="2026-07-03", baseline=200.0, forecast_origin_date="2026-07-02", latest_estimate_date="2026-07-02"),
        ]
    )

    analysis, audit = build_panel(rows, panel)

    assert len(analysis) == 1
    assert analysis.iloc[0]["baseline_forecast_usd"] == 100.0
    assert bool(audit.iloc[0]["included"]) is True
    assert math.isclose(analysis.iloc[0]["x_log_thursday_ratio"], math.log(12.0 / 100.0))


def test_rolling_predictions_train_only_on_prior_opening_weekends() -> None:
    rows = []
    for i in range(23):
        weekend = pd.Timestamp("2025-01-03") + pd.Timedelta(days=7 * (i // 2))
        rows.append(
            {
                "release_run_id": i + 1,
                "movie_id": i + 100,
                "title": f"Movie {i}",
                "opening_weekend_start": weekend,
                "release_year": int(weekend.year),
                "opening_thursday_date": weekend - pd.Timedelta(days=1),
                "opening_thursday_daily_gross_usd": 10.0 + i,
                "baseline_forecast_usd": 100.0,
                "actual_opening_weekend_gross_usd": 100.0 + i,
                "e_log": math.log((100.0 + i) / 100.0),
                "x_log_thursday_ratio": math.log((10.0 + i) / 100.0),
            }
        )
    panel = pd.DataFrame(rows)

    predictions = build_rolling_predictions(panel, min_train_n=20)
    scored = predictions.loc[predictions["scored"]]
    first_scored_weekend = scored["opening_weekend_start"].min()
    first_rows = scored.loc[scored["opening_weekend_start"].eq(first_scored_weekend)]

    assert first_scored_weekend == pd.Timestamp("2025-03-14")
    assert first_rows["rolling_train_n"].eq(20).all()


def test_validation_ratio_update_rows_are_reproducible_by_production_inference() -> None:
    rows = []
    for i in range(23):
        weekend = pd.Timestamp("2025-01-03") + pd.Timedelta(days=7 * i)
        baseline = 100.0
        thursday = 10.0 + i
        actual = baseline * math.exp(0.2 + 0.5 * math.log(thursday / baseline))
        rows.append(
            {
                "release_run_id": i + 1,
                "movie_id": i + 100,
                "title": f"Movie {i}",
                "opening_weekend_start": weekend,
                "release_year": int(weekend.year),
                "opening_thursday_date": weekend - pd.Timedelta(days=1),
                "opening_thursday_daily_gross_usd": thursday,
                "baseline_forecast_usd": baseline,
                "actual_opening_weekend_gross_usd": actual,
                "e_log": math.log(actual / baseline),
                "x_log_thursday_ratio": math.log(thursday / baseline),
            }
        )
    panel = pd.DataFrame(rows)
    predictions = build_rolling_predictions(panel, min_train_n=20)
    ratio_rows = predictions.loc[predictions["scored"] & predictions["ow_model"].eq("ratio_update")]

    assert len(ratio_rows) == 3
    for row in ratio_rows.itertuples(index=False):
        policy = {
            "policy_name": "opening_thursday_actual_ratio_update_prod",
            "policy_version": "fold_identity",
            "production_ow_update_enabled": True,
            "selected_update_model": "ratio_update",
            "parameters": {"alpha": row.ratio_alpha, "beta": row.ratio_beta},
            "residual_policy": {"ratio_update_residuals_log": [-0.1, -0.05, 0, 0.05, 0.1]},
        }
        update = update_ow_from_opening_thursday_actual(
            baseline_ow_usd=row.baseline_forecast_usd,
            opening_thursday_daily_gross_usd=row.opening_thursday_daily_gross_usd,
            frozen_policy=policy,
            row={
                "release_run_id": row.release_run_id,
                "opening_weekend_start": row.opening_weekend_start,
                "opening_thursday_date": row.opening_thursday_date,
                "opening_thursday_source": "the_numbers",
            },
        )
        assert update.updated_ow_usd == pytest.approx(row.forecast_ow_usd)
