import numpy as np
import pytest

from prediction_market_backtest.cdf_validation import CDFCandidate
from prediction_market_backtest.probcal import (BetaCalibration,CalibratedDistribution,fit_beta_calibration,
    fit_isotonic_calibration,location_shift,movie_balanced_weights)


def test_beta_identity_and_monotonic_inverse():
    identity=BetaCalibration(1,1);grid=np.linspace(0,1,101)
    assert identity.transform(grid)==pytest.approx(grid)
    fitted=fit_beta_calibration([.2,.3,.7,.8],[1,2,3,4],regularization=1)
    mapped=fitted.transform(grid)
    assert np.all(np.diff(mapped)>=0)
    assert fitted.inverse(mapped)==pytest.approx(grid,abs=1e-6)


def test_isotonic_shrink_and_movie_weights():
    weights=movie_balanced_weights([1,1,2]);assert weights[:2].sum()==pytest.approx(weights[2])
    identity=fit_isotonic_calibration([.2,.8],[1,2],0)
    grid=np.linspace(0,1,11);assert identity.transform(grid)==pytest.approx(grid)
    calibrated=fit_isotonic_calibration([.2,.8],[1,2],.5);assert np.all(np.diff(calibrated.transform(grid))>=0)


def test_location_and_calibrated_distribution_are_valid():
    assert location_shift([-.2,.4],[1,2]) in (-.2,.4)
    base=CDFCandidate("base",np.array([.01,.5,.99]),np.array([-1,0,1]),False)
    distribution=CalibratedDistribution("location",base,location_shift_log=.2)
    grid=distribution.quantile(np.linspace(.01,.99,20),10)
    assert np.all(np.diff(grid)>=0) and 0<=distribution.cdf(10,10)<=1
