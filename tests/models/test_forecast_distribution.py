from __future__ import annotations

import pytest

from models.boxoffice.forecast_distribution import ForecastDistribution


def payload() -> dict[str, object]:
    levels = [index / 102 for index in range(1, 102)]
    values = [10_000_000 + index * 1_000_000 for index in range(101)]
    return {
        "quantile_levels": levels,
        "quantile_values_usd": values,
        "distribution_policy": "production_distribution_v2",
        "distribution_policy_version": "v2",
        "distribution_emission_hash": "abc123",
        "tail_policy": {"name": "t4"},
    }


def test_full_payload_interpolates_monotonically_and_conserves_bucket_mass() -> None:
    distribution = ForecastDistribution.from_payload(payload())
    assert distribution is not None
    assert distribution.cdf(0) == 0
    assert distribution.cdf(200_000_000) == 1
    assert distribution.cdf(50_000_000) < distribution.cdf(60_000_000)
    assert distribution.quantile(0.25) < distribution.quantile(0.75)
    probabilities = distribution.bucket_probabilities([30_000_000, 50_000_000, 70_000_000, 90_000_000])
    assert all(value >= 0 for value in probabilities)
    assert sum(probabilities) == pytest.approx(1.0)


def test_five_quantile_interval_summary_is_not_accepted_as_production_distribution() -> None:
    short = payload()
    short["quantile_levels"] = [0.025, 0.1, 0.5, 0.9, 0.975]
    short["quantile_values_usd"] = [10_000_000, 20_000_000, 30_000_000, 40_000_000, 50_000_000]
    assert ForecastDistribution.from_payload(short) is None


def test_invalid_payload_is_rejected() -> None:
    invalid = payload()
    invalid["quantile_values_usd"] = list(reversed(invalid["quantile_values_usd"]))
    assert ForecastDistribution.from_payload(invalid) is None


def test_degenerate_distribution_uses_half_open_market_bucket_semantics() -> None:
    complete = payload()
    complete["quantile_values_usd"] = [50_000_000] * 101
    distribution = ForecastDistribution.from_payload(complete)

    assert distribution is not None
    # $50m belongs to [50m, 70m), not the lower bucket ending at $50m.
    assert distribution.bucket_probabilities([30_000_000, 50_000_000, 70_000_000, 90_000_000]) == [0, 0, 1, 0, 0]


def test_direct_draw_bucket_probabilities_take_precedence_for_matching_grid() -> None:
    direct = payload()
    direct["market_bucket_probabilities"] = {"grid-1": [0.1, 0.2, 0.3, 0.15, 0.25]}
    distribution = ForecastDistribution.from_payload(direct)

    assert distribution is not None
    assert distribution.bucket_probabilities([30_000_000, 50_000_000, 70_000_000, 90_000_000], grid_id="grid-1") == pytest.approx(
        [0.1, 0.2, 0.3, 0.15, 0.25]
    )
