from pm_box_office.orchestration.forecast_routing import refresh_rule_for


def test_polymarket_refresh_rule_routes_validated_matches() -> None:
    rule = refresh_rule_for("polymarket_metadata")
    assert rule.refresh_polymarket_matches is True
    assert rule.refresh_estimate_candidates is True


def test_amc_worker_does_not_reconcile_every_seat_batch() -> None:
    rule = refresh_rule_for("amc_worker")
    assert rule.reconcile_identities is False
    assert rule.refresh_estimate_candidates is False
