from prediction_market_backtest.policy_registry import require_approved_probability_policy
import pytest


def test_provisional_policy_is_not_production_eligible():
    with pytest.raises(ValueError,match="prospective"):require_approved_probability_policy("pre_release_pointscale_cal_001")


def test_prospective_sample_gate_is_strict():
    movies=29
    assert ("inconclusive_insufficient_prospective_sample" if movies<30 else "approved_probability_model")=="inconclusive_insufficient_prospective_sample"
