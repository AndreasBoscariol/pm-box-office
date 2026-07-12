from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import t as student_t

from prediction_market_backtest.consensus_cdf import (BaseCdf, QCOLS, QUANTILE_GRID, TailLayeredCdf, blend_base,
                                                       run_consensus_cdf_study, sequential_diagnostics)


def _base(point: float = 100.0) -> BaseCdf:
    # This deliberately has a median at the point, as every challenger must.
    quantiles = point * np.exp(np.linspace(-1.0, 1.0, len(QUANTILE_GRID)))
    return BaseCdf.from_quantiles(quantiles, point)


def test_tail_layer_uses_shared_fixed_median_and_normalized_bucket_mass() -> None:
    base = _base()
    layered = TailLayeredCdf(base.cdf, base.ppf, 100.0, .30)
    assert layered.cdf(100.0) == pytest.approx(.5, abs=1e-9)
    probabilities = layered.bucket_probabilities((80.0, 100.0, 120.0))
    assert probabilities.sum() == pytest.approx(1.0)
    assert np.all(probabilities >= 0)


def test_base_mixture_happens_before_the_shared_tail_layer() -> None:
    left, right = _base(100.0), _base(100.0).scaled(1.1)
    mixed, inverse = blend_base(left, right, .75)
    layered = TailLayeredCdf(mixed, inverse, 100.0, .30)
    assert layered.cdf(100.0) == pytest.approx(.5, abs=1e-9)
    expected = .98 * (.75 * left.cdf(130) + .25 * right.cdf(130)) + .02 * student_t.cdf(np.log(1.3) / .30, 4)
    assert layered.cdf(130.0) == pytest.approx(expected)


def test_sequential_diagnostic_flags_unsupported_turnover() -> None:
    rows = [
        {"candidate": "base", "listing_origin": -10, "grid_id": "g", "movie_id": 1, "origin_day": -2, "probability_vector": [.8, .2], "source_state": "other_sparse", "source_count": 1, "frozen_point_forecast_usd": 100, "log_loss": 1.0, "realized_bucket_probability": .2, "bucket_leader": 0},
        {"candidate": "base", "listing_origin": -10, "grid_id": "g", "movie_id": 1, "origin_day": -1, "probability_vector": [.2, .8], "source_state": "other_sparse", "source_count": 1, "frozen_point_forecast_usd": 100, "log_loss": .2, "realized_bucket_probability": .8, "bucket_leader": 1},
    ]
    result = sequential_diagnostics(__import__("pandas").DataFrame(rows))
    assert result.iloc[0].tv == pytest.approx(.6)
    assert bool(result.iloc[0].unsupported_probability_movement)
    assert int(result.iloc[0].bucket_leader_reversal) == 1


def test_study_writes_fixed_listing_and_sequential_outputs(tmp_path) -> None:
    import pandas as pd

    rows = []
    for origin in range(-10, 0):
        for candidate, width in [("production_policy_v1", 1.0), ("D0_origin_empirical", 1.1), ("D1_point_size", .9), ("D2_quantile_equal_centered", .8), ("D2_source_shrunk_k40_equal_centered", .85)]:
            values = 20_000_000 * np.exp(np.linspace(-width, width, len(QUANTILE_GRID)))
            rows.append({"movie_id": 1, "release_run_id": 1, "origin_day": origin, "holdout_year": 2025, "candidate": candidate,
                         "frozen_point_forecast_usd": 20_000_000, "actual_opening_weekend_gross_usd": 21_000_000,
                         "source_count": 2, "estimate_sources": "boxofficereport, boxofficepro", **dict(zip(QCOLS, values))})
    panel = tmp_path / "panel.parquet"; pd.DataFrame(rows).to_parquet(panel)
    history = pd.DataFrame([{"movie_id": i, "origin_day": origin, "holdout_year": 2024, "signed_log_error": .1,
                             "cross_source_log_disagreement": .05, "aggregate_range_asymmetry_log": 0,
                             "source_count": 2, "estimate_sources": "boxofficereport, boxofficepro"} for i in range(30) for origin in range(-10, 0)])
    oof = tmp_path / "oof.csv"; history.to_csv(oof, index=False)
    output = tmp_path / "output"
    manifest = run_consensus_cdf_study(panel, output, oof_path=oof, actual_market_path=None, listing_origins=(-10,))
    assert manifest["records"] > 0
    assert (output / "01_fixed_listing_grid_scores.parquet").exists()
    assert (output / "05_sequential_probability_migration.csv").exists()
    scored = pd.read_parquet(output / "01_fixed_listing_grid_scores.parquet")
    assert scored.grid_source.eq("synthetic").all()
    assert scored.assign(boundary_key=scored.boundaries_usd.map(tuple)).groupby("grid_id").boundary_key.nunique().eq(1).all()
