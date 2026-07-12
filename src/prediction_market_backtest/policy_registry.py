"""Fail-closed registry for production probability eligibility."""

from __future__ import annotations

REJECTED_POLICIES=frozenset({"007_cdf_extension_001","007_anchor_preserving_empirical_warp_001",
    "007_raw_weighted_empirical_benchmark","007_probcal_001","beta__007_cdf_extension_001",
    "beta__007_raw_weighted_empirical_benchmark","isotonic__007_cdf_extension_001",
    "isotonic__007_raw_weighted_empirical_benchmark","location__007_cdf_extension_001",
    "location__007_raw_weighted_empirical_benchmark"})
DEVELOPMENT_BASES=frozenset({"pre_release_pointscale_splitt_001"})
PROVISIONAL_POLICIES=frozenset({"pre_release_pointscale_cal_001"})
APPROVED_POLICIES:frozenset[str]=frozenset()


def require_approved_probability_policy(policy:str|None)->str:
    if not policy:raise ValueError("probability policy is required")
    if policy in REJECTED_POLICIES:raise ValueError(f"probability policy {policy} is rejected")
    if policy in DEVELOPMENT_BASES:raise ValueError(f"probability policy {policy} is a development base, not production eligible")
    if policy in PROVISIONAL_POLICIES:raise ValueError(f"probability policy {policy} awaits prospective validation")
    if policy not in APPROVED_POLICIES:raise ValueError(f"probability policy {policy} has not passed the approval gate")
    return policy


def validate_probability_panel(policies:list[str])->None:
    if len(set(policies))!=1:raise ValueError("mixed probability policies are forbidden")
    require_approved_probability_policy(policies[0] if policies else None)
