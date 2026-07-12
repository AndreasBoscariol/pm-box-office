import numpy as np
import pytest
from scipy.integrate import quad

from prediction_market_backtest.asymdist import AsymmetricEmpirical,SplitStudentT


def test_split_student_t_normalization_monotonicity_and_inverse():
    distribution=SplitStudentT(-.1,.3,.7,5)
    integral=quad(lambda x:float(distribution.pdf(x)),-100,100)[0]
    assert integral==pytest.approx(1,abs=1e-7)
    grid=np.linspace(.001,.999,200);quantiles=distribution.ppf(grid)
    assert np.all(np.diff(quantiles)>0)
    assert distribution.cdf(quantiles)==pytest.approx(grid,abs=1e-9)
    assert np.array_equal(distribution.sample(100,7),distribution.sample(100,7))


def test_asymmetric_scale_and_empirical_transform():
    distribution=SplitStudentT(0,.2,.8,4)
    assert distribution.ppf(.95)>abs(distribution.ppf(.05))
    empirical=AsymmetricEmpirical(np.array([-1.,0.,1.]),np.array([.2,.3,.5]),(1,2,3),delta=-.1,lower_scale=.5,upper_scale=2,upper_tail_stretch=1.5)
    assert empirical.transformed[0]==pytest.approx(-.6)
    assert empirical.transformed[-1]>1
    assert empirical.cdf(100)==pytest.approx(1)
    assert np.array_equal(empirical.sample(20,4),empirical.sample(20,4))
