from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from prediction_market_backtest.evaluation import ProbabilityCase, expanding_blend_evaluation, multiclass_brier
from prediction_market_backtest.forecast import ForecastDistribution, stable_seed
from prediction_market_backtest.market_history import PriceObservation, synchronize_prices
from prediction_market_backtest.probabilities import bucket_probability_details
from prediction_market_backtest.semantics import Bucket


def test_distribution_rejects_invalid_draws_and_seed_is_stable() -> None:
    now = datetime.now(timezone.utc)
    kwargs = dict(movie_id=1, regime="pre_release", origin="-1", information_cutoff_utc=now,
        forecast_created_utc=now, forecast_available_utc=now, point_forecast=10, model_artifact_version="m",
        distribution_artifact_version="d", simulator_version="s", seed=1, metadata={})
    with pytest.raises(ValueError):
        ForecastDistribution(draws=np.array([1, np.nan]), **kwargs)
    assert stable_seed(1, "-1", "d", 42) == stable_seed(1, "-1", "d", 42)


def test_exact_boundaries_and_monte_carlo_precision() -> None:
    buckets = [Bucket("low", None, 20), Bucket("middle", 20, 25), Bucket("high", 25, None)]
    details = bucket_probability_details([19, 20, 24, 25], buckets)
    assert [row.count for row in details] == [1, 2, 1]
    assert all(row.standard_error >= 0 for row in details)


def test_price_sync_never_uses_future_and_rejects_incomplete_vectors() -> None:
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = [PriceObservation("a", now - timedelta(minutes=2), .4),
            PriceObservation("a", now + timedelta(seconds=1), .9),
            PriceObservation("b", now - timedelta(minutes=1), .7)]
    vector = synchronize_prices(["a", "b"], rows, now, timedelta(minutes=15))
    assert vector.raw == (.4, .7)
    assert sum(vector.coherent) == pytest.approx(1)
    with pytest.raises(ValueError, match="incomplete"):
        synchronize_prices(["a", "missing"], rows, now, timedelta(minutes=15))


def test_multiclass_score_and_expanding_blend_are_chronological() -> None:
    assert multiclass_brier((.2, .8), 1) == pytest.approx(.08)
    cases = [ProbabilityCase(i, i, (.8, .2), (.6, .4), 0) for i in range(1, 12)]
    cases.append(ProbabilityCase(12, 12, (.7, .3), (.4, .6), 0))
    rows = expanding_blend_evaluation(cases, minimum_training_movies=10)
    assert [row["movie_id"] for row in rows] == [11, 12]
    assert rows[-1]["weight"] == 1.0
