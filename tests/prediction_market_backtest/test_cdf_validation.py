import numpy as np
import pytest

from prediction_market_backtest.cdf_validation import (ANCHOR_PROBABILITIES, CDFCandidate, build_candidates,
                                                        crps_from_quantiles, quantile_loss)
from prediction_market_backtest.policy007 import WeightedPool, shrunk_quantile_function


def _pool():
    return WeightedPool(np.array([-.5,0,.5]),np.array([1,2,1]),(1,2,3),1,("origin_bucket",),"P_-1",.6,
                        np.array([-.8,-.2,.1,.7]),np.ones(4))


def test_anchor_candidates_preserve_locked_quantiles_and_monotonicity():
    pool=_pool();locked=shrunk_quantile_function(pool,ANCHOR_PROBABILITIES)
    candidates=build_candidates(pool)
    for candidate in candidates[:2]:
        actual=np.log(candidate.quantile(ANCHOR_PROBABILITIES))
        assert actual==pytest.approx(locked)
        assert np.all(np.diff(candidate.residual_quantiles)>=0)
        assert 0<=candidate.cdf(1)<=1


def test_quantile_loss_and_degenerate_crps():
    assert quantile_loss(10,8,.25)==pytest.approx(.5)
    candidate=CDFCandidate("point",np.array([.01,.99]),np.array([0.,0.]),True)
    assert crps_from_quantiles(candidate,1,1)==pytest.approx(0)
