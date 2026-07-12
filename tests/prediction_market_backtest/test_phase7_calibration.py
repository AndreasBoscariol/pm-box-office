import numpy as np
import pytest

from prediction_market_backtest.probcal import (OriginGroupPowerCalibration,PowerCalibration,
    fit_origin_group_power,fit_power_calibration)


def test_power_identity_monotonicity_inverse_and_bounds():
    grid=np.linspace(0,1,101);identity=PowerCalibration(1)
    assert identity.transform(grid)==pytest.approx(grid)
    tilted=PowerCalibration(1.4);assert np.all(np.diff(tilted.transform(grid))>=0)
    assert tilted.inverse(tilted.transform(grid))==pytest.approx(grid)
    with pytest.raises(ValueError):PowerCalibration(3)


def test_fitted_power_and_partial_pooling_are_deterministic():
    pits=[.4,.5,.7,.8,.9,.6];movies=list(range(6));groups=[0,0,1,1,2,2]
    first=fit_power_calibration(pits,movies,1);second=fit_power_calibration(pits,movies,1)
    assert first==second and first.exponent>=1
    exponents=fit_origin_group_power(pits,movies,groups,10)
    assert len(exponents)==3 and all(.5<=value<=2 for value in exponents)
    mapping=OriginGroupPowerCalibration(exponents,2);assert 0<=mapping.transform(.5)<=1
