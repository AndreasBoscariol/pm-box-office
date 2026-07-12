from __future__ import annotations

import math

import pandas as pd

from scripts.diagnose_amc_live_models import add_clustered_blends, add_rolling_holdover_bias_corrections, constrained_beta


def test_constrained_beta_recovers_partial_log_update() -> None:
    frame = pd.DataFrame(
        {
            "baseline_gross_usd": [100.0, 100.0, 100.0],
            "amc_pred_usd": [400.0, 25.0, 225.0],
        }
    )
    frame["actual_gross_usd"] = frame["baseline_gross_usd"] * (
        frame["amc_pred_usd"] / frame["baseline_gross_usd"]
    ) ** 0.5

    assert math.isclose(constrained_beta(frame), 0.5)


def test_clustered_blend_leaves_rows_without_baseline_unset() -> None:
    frame = pd.DataFrame(
        {
            "movie_day_key": ["1|a", "2|b", "3|c"],
            "origin_group": ["late", "late", "late"],
            "baseline_gross_usd": [100.0, 100.0, float("nan")],
            "amc_pred_usd": [200.0, 50.0, 75.0],
            "actual_gross_usd": [150.0, 80.0, 70.0],
        }
    )

    result = add_clustered_blends(frame)

    assert result.loc[:1, "blend_pooled_usd"].notna().all()
    assert pd.isna(result.loc[2, "blend_pooled_usd"])


def test_holdover_bias_correction_uses_only_earlier_dates() -> None:
    rows = []
    for idx in range(12):
        rows.append({
            "movie_day_key": f"{idx}|day", "forecast_origin": "16:00", "exhibition_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=idx),
            "run_day": 10, "actual_gross_usd": 200.0, "pred_gross_hybrid": 100.0,
            "pred_gross_additive": 100.0,
        })
    result = add_rolling_holdover_bias_corrections(pd.DataFrame(rows))

    assert result.loc[9, "pred_gross_hybrid_bias_corrected"] == 100.0
    assert result.loc[11, "pred_gross_hybrid_bias_corrected"] > 190.0
