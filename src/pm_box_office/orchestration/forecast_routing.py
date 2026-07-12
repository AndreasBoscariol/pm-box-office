"""Declarative mapping from completed source runs to forecast refresh actions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceRefreshRule:
    reconcile_identities: bool = True
    refresh_estimate_candidates: bool = True
    refresh_polymarket_matches: bool = False
    activate_today_amc_campaign: bool = True


DEFAULT_REFRESH_RULE = SourceRefreshRule()

# A source gets a named rule rather than command-string conditionals scattered
# through the supervisor. Adding a collector now requires one reviewable entry.
SOURCE_REFRESH_RULES: dict[str, SourceRefreshRule] = {
    "polymarket_metadata": SourceRefreshRule(refresh_polymarket_matches=True),
    "amc_worker": SourceRefreshRule(
        reconcile_identities=False,
        refresh_estimate_candidates=False,
        activate_today_amc_campaign=False,
    ),
}


def refresh_rule_for(source_key: str) -> SourceRefreshRule:
    return SOURCE_REFRESH_RULES.get(source_key, DEFAULT_REFRESH_RULE)
