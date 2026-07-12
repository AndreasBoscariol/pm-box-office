import numpy as np
import pytest

from prediction_market_backtest.asymdist import SplitStudentT,gross_cdf


def test_threshold_metrics_hand_calculation_and_gross_bounds():
    probability=.25;outcome=1
    assert (probability-outcome)**2==pytest.approx(.5625)
    assert -(outcome*np.log(probability)+(1-outcome)*np.log1p(-probability))==pytest.approx(-np.log(.25))
    distribution=SplitStudentT(0,.3,.6,5)
    assert gross_cdf(distribution,0,20_000_000)==0
    assert 0<gross_cdf(distribution,20_000_000,20_000_000)<1
