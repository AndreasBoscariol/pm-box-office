from __future__ import annotations

import pytest

from models.boxoffice.market_buckets import (
    generate_fixed_grid_ensemble,
    generate_rounding_variants,
    get_allowed_widths,
    round_start_to_clean_strikes,
)


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        (999_999, ()),
        (1_000_000, (500_000,)),
        (5_000_000, (1_000_000, 1_500_000)),
        (10_000_000, (2_000_000,)),
        (18_000_000, (3_000_000,)),
        (30_000_000, (4_000_000, 5_000_000)),
        (45_000_000, (5_000_000,)),
        (75_000_000, (5_000_000, 7_000_000)),
        (90_000_000, (10_000_000,)),
        (125_000_000, (12_000_000, 13_000_000)),
        (190_000_000, (20_000_000,)),
    ],
)
def test_allowed_width_thresholds(anchor: int, expected: tuple[int, ...]) -> None:
    assert get_allowed_widths(anchor) == expected


@pytest.mark.parametrize(
    ("anchor", "width", "boundaries"),
    [
        (3_000_000, 500_000, (2_000_000, 2_500_000, 3_000_000, 3_500_000)),
        (15_000_000, 2_000_000, (12_000_000, 14_000_000, 16_000_000, 18_000_000)),
        (32_000_000, 4_000_000, (26_000_000, 30_000_000, 34_000_000, 38_000_000)),
        (105_000_000, 10_000_000, (90_000_000, 100_000_000, 110_000_000, 120_000_000)),
        (160_000_000, 12_000_000, (142_000_000, 154_000_000, 166_000_000, 178_000_000)),
        (240_000_000, 20_000_000, (210_000_000, 230_000_000, 250_000_000, 270_000_000)),
    ],
)
def test_canonical_grid_uses_clean_start_rounding(anchor: int, width: int, boundaries: tuple[int, ...]) -> None:
    grid = _variants(anchor)[0]
    assert grid.selected_width_usd == width
    assert grid.boundaries_usd == boundaries
    assert grid.boundaries_usd[0] > 0
    assert all(left < right for left, right in zip(grid.boundaries_usd, grid.boundaries_usd[1:]))


def test_true_rounding_tie_exposes_only_clean_strike_variants() -> None:
    assert round_start_to_clean_strikes(47_500_000, 5_000_000) == (45_000_000, 50_000_000)
    variants = _variants(55_000_000)
    assert [grid.rounding_variant for grid in variants] == ["canonical", "higher_strike"]
    assert [grid.rounded_start_usd for grid in variants] == [45_000_000, 50_000_000]


def test_nonpositive_start_is_shifted_without_duplicate_variant() -> None:
    variants = _variants(1_000_000)
    assert len(variants) == 1
    assert variants[0].boundaries_usd[0] == 500_000


def test_sub_million_anchor_is_out_of_domain() -> None:
    with pytest.raises(ValueError, match="outside"):
        _variants(999_999)


def test_fixed_research_ensemble_has_4_5_6_bucket_definitions() -> None:
    grids = generate_fixed_grid_ensemble(release_run_id=42, listing_origin="P_-10", anchor_forecast_usd=15_000_000)
    assert {grid.bucket_count for grid in grids} == {4, 5, 6}
    assert all(len(grid.boundaries_usd) == grid.bucket_count - 1 for grid in grids)
    assert all(grid.listing_origin == "P_-10" for grid in grids)
    assert all(left < right for grid in grids for left, right in zip(grid.boundaries_usd, grid.boundaries_usd[1:]))


def _variants(anchor: int):
    return generate_rounding_variants(
        release_run_id=42,
        effective_listing_origin="P_-10",
        anchor_forecast_usd=anchor,
        anchor_forecast_emission_hash="immutable-hash",
        point_policy_version="frozen-point-v1",
    )
