from __future__ import annotations

import datetime as dt

import pandas as pd

from eda.During.thursday_amc_preview_nowcast_diagnostic import (
    build_policy,
    build_rolling_predictions,
    evaluate_models,
)
from models.boxoffice.thursday_amc_transfer import (
    HYBRID_PARTIAL_POOL,
    STAGE2_POOLS,
    donor_daily_panel,
    fit_pooled_transfer,
)


def _row(i: int, *, origin: str = "16:00") -> dict[str, object]:
    obs = 50.0 + i
    final = obs * 2
    actual = (final + 1) * 10_000
    return {
        "release_run_id": i,
        "movie_id": 100 + i,
        "title": f"Movie {i}",
        "exhibition_date": dt.date(2026, 1, 1) + dt.timedelta(days=7 * i),
        "opening_weekend_start": pd.Timestamp(dt.date(2026, 1, 2) + dt.timedelta(days=7 * i)),
        "release_year": 2026,
        "forecast_origin": origin,
        "is_preview": 1,
        "s_obs": obs,
        "s_final_eod": final,
        "c_obs": 100.0,
        "c_scheduled_known": 200.0,
        "actual_gross_usd": actual,
        "n_snapshots": 10,
        "coverage": 0.8,
    }


def test_thursday_amc_preview_rolling_predictions_and_policy() -> None:
    panel = pd.DataFrame([_row(i) for i in range(8)])

    predictions = build_rolling_predictions(panel, min_train_rows=3)

    assert predictions.loc[:2, "scored"].eq(False).all()
    assert predictions.loc[3:, "scored"].eq(True).all()
    assert predictions.loc[3, "training_n"] == 3
    assert predictions.loc[3, "forecast_preview_gross_hybrid_usd"] > 0

    metrics = evaluate_models(predictions)
    assert set(metrics["point_model"]) == {"multiplicative", "additive", "hybrid"}
    assert metrics["n"].min() == 5

    policy = build_policy(predictions, metrics, min_train_rows=3)
    assert policy["enabled"] is True
    assert policy["selected_point_model"] in {"multiplicative", "additive", "hybrid"}


def test_amc_transfer_pools_include_release_stage_hybrid_candidate() -> None:
    rows = []
    base = dt.date(2026, 1, 1)
    days = ["Thursday", "Friday", "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday"]
    for movie_idx in range(4):
        release_date = base + dt.timedelta(days=7 * movie_idx + 1)
        for day_offset, _ in enumerate(days):
            exhibition = release_date + dt.timedelta(days=day_offset - 1)
            rows.append(
                {
                    "movie_id": movie_idx + 1,
                    "release_run_id": movie_idx + 10,
                    "release_date": release_date,
                    "exhibition_date": exhibition,
                    "forecast_origin": "16:00",
                    "s_obs": 100 + movie_idx + day_offset,
                    "s_final_eod": 200 + 2 * movie_idx + day_offset,
                    "actual_gross_usd": 1_000_000 + 50_000 * movie_idx + 10_000 * day_offset,
                    "actual_theaters": 3000,
                    "n_snapshots": 10,
                    "coverage": 0.8,
                }
            )
    panel = pd.DataFrame(rows)

    assert set(STAGE2_POOLS) == {
        "ALL_DAILY_CONTEXT",
        "OPENING_EVENT_POOL",
        "WEEKDAY_POOL",
        "HYBRID_PARTIAL_POOL",
        "FSS_POOL",
    }
    opening = donor_daily_panel(panel, donor_pool="OPENING_EVENT_POOL")
    assert opening["is_opening_event"].eq(1.0).all()

    policy = fit_pooled_transfer(panel, selected_stage2_pool=HYBRID_PARTIAL_POOL, min_movies=2)
    assert policy["selected_stage2_pool"] == HYBRID_PARTIAL_POOL
    assert policy["partial_pooling_enabled"] is True
    assert policy["release_stage_context_enabled"] is True
    assert policy["production_ow_update_enabled"] is False
