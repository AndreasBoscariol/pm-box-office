import pytest
from prediction_market_backtest.policy_registry import require_approved_probability_policy,validate_probability_panel


def test_rejected_missing_unapproved_and_mixed_policies_fail_closed():
    for policy in (None,"007_probcal_001","pre_release_asymdist_001"):
        with pytest.raises(ValueError):require_approved_probability_policy(policy)
    with pytest.raises(ValueError,match="mixed"):validate_probability_panel(["a","b"])
