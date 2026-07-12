"""Versioned pre-release consensus-distribution regime selection."""

from __future__ import annotations


POLICY_NAME = "production_distribution_v3_d1_from_p3_t4_tail_safe"
D1_START_ORIGIN = -3
TAIL_WEIGHT = 0.02
TAIL_DF = 4


def base_policy_for_origin(origin_day: int) -> str:
    """Select D1 only from P-3 onward; P-4 deliberately remains v2 base."""
    return "D1_point_size" if int(origin_day) >= D1_START_ORIGIN else "production_distribution_v2_base"


def validate_policy(policy: dict[str, object]) -> None:
    if policy.get("policy_name") != POLICY_NAME:
        raise ValueError("unexpected pre-release distribution policy")
    if int(policy.get("d1_start_origin", 0)) != D1_START_ORIGIN:
        raise ValueError("D1 activation boundary must be P_-3")
    if float(policy.get("tail_contamination_weight", 0)) != TAIL_WEIGHT or int(policy.get("tail_reference_df", 0)) != TAIL_DF:
        raise ValueError("v3 must retain the fixed 2% Student-t(4) tail layer")
