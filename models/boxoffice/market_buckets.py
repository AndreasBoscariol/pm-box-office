"""Deterministic synthetic box-office market bucket definitions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Mapping


TARGET_LISTING_ORIGIN = "P_-10"
WIDTH_RULE_VERSION = "movie_opening_widths_v1"
ROUNDING_RULE_VERSION = "clean_start_strikes_v1"


@dataclass(frozen=True)
class MarketGrid:
    release_run_id: int
    target_listing_origin: str
    effective_listing_origin: str
    anchor_forecast_usd: int
    anchor_forecast_emission_hash: str
    point_policy_version: str
    width_rule_version: str
    rounding_rule_version: str
    selected_width_usd: int
    ideal_start_usd: int
    rounded_start_usd: int
    boundaries_usd: tuple[int, int, int, int]
    rounding_variant: str
    width_selection_rule: str
    grid_id: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SyntheticMarketGrid:
    """A fixed synthetic market definition used only for CDF research.

    ``MarketGrid`` above is the five-bucket definition emitted by the live
    product.  Research needs a small, pre-declared 4/5/6 bucket ensemble
    without changing that runtime contract, hence this deliberately separate
    type.
    """

    release_run_id: int
    listing_origin: str
    anchor_forecast_usd: int
    selected_width_usd: int
    bucket_count: int
    boundaries_usd: tuple[int, ...]
    alignment: str
    rounding_variant: str
    grid_id: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def get_allowed_widths(anchor_forecast_usd: int | float) -> tuple[int, ...]:
    value = int(round(float(anchor_forecast_usd)))
    if value < 1_000_000:
        return ()
    if value < 5_000_000:
        return (500_000,)
    if value < 10_000_000:
        return (1_000_000, 1_500_000)
    if value < 18_000_000:
        return (2_000_000,)
    if value < 30_000_000:
        return (3_000_000,)
    if value < 45_000_000:
        return (4_000_000, 5_000_000)
    if value < 75_000_000:
        return (5_000_000,)
    if value < 90_000_000:
        return (5_000_000, 7_000_000)
    if value < 125_000_000:
        return (10_000_000,)
    if value < 190_000_000:
        return (12_000_000, 13_000_000)
    return (20_000_000,)


def select_canonical_width(
    anchor_forecast_usd: int | float,
    *,
    empirical_modes: Mapping[tuple[int, ...], int] | None = None,
) -> tuple[int, str]:
    """Choose an empirical mode when supplied, otherwise the lower width."""

    allowed = get_allowed_widths(anchor_forecast_usd)
    if not allowed:
        raise ValueError("anchor forecast is outside the synthetic market domain")
    empirical = (empirical_modes or {}).get(allowed)
    if empirical in allowed:
        return int(empirical), "empirical_mode_v1"
    return allowed[0], "lower_allowed_width_no_empirical_mode_v1"


def get_clean_strike_increment(width_usd: int) -> int:
    increments = {
        500_000: 500_000,
        1_000_000: 500_000,
        1_500_000: 500_000,
        2_000_000: 1_000_000,
        3_000_000: 1_000_000,
        4_000_000: 2_000_000,
        5_000_000: 5_000_000,
        7_000_000: 1_000_000,
        10_000_000: 5_000_000,
        12_000_000: 1_000_000,
        13_000_000: 1_000_000,
        20_000_000: 10_000_000,
    }
    try:
        return increments[int(width_usd)]
    except KeyError as exc:
        raise ValueError(f"unsupported market bucket width {width_usd}") from exc


def round_start_to_clean_strikes(ideal_start_usd: int | float, width_usd: int) -> tuple[int, ...]:
    """Return nearest clean starts, retaining both starts only for a true tie."""

    increment = get_clean_strike_increment(width_usd)
    ideal = float(ideal_start_usd)
    lower = math.floor(ideal / increment) * increment
    upper = math.ceil(ideal / increment) * increment
    if lower == upper:
        return (int(lower),)
    lower_distance = ideal - lower
    upper_distance = upper - ideal
    if math.isclose(lower_distance, upper_distance, abs_tol=1e-9):
        return int(lower), int(upper)
    return (int(lower if lower_distance < upper_distance else upper),)


def generate_rounding_variants(
    *,
    release_run_id: int,
    effective_listing_origin: str,
    anchor_forecast_usd: int | float,
    anchor_forecast_emission_hash: str,
    point_policy_version: str,
    empirical_modes: Mapping[tuple[int, ...], int] | None = None,
) -> tuple[MarketGrid, ...]:
    anchor = int(round(float(anchor_forecast_usd)))
    width, width_selection_rule = select_canonical_width(anchor, empirical_modes=empirical_modes)
    ideal_start = int(round(anchor - 1.5 * width))
    starts = round_start_to_clean_strikes(ideal_start, width)
    variants: list[MarketGrid] = []
    for start in starts:
        increment = get_clean_strike_increment(width)
        while start <= 0:
            start += increment
        if any(existing.rounded_start_usd == start for existing in variants):
            continue
        variant = "canonical" if not variants else "higher_strike"
        boundaries = (start, start + width, start + 2 * width, start + 3 * width)
        identity = {
            "release_run_id": int(release_run_id),
            "target_listing_origin": TARGET_LISTING_ORIGIN,
            "effective_listing_origin": effective_listing_origin,
            "anchor_forecast_usd": anchor,
            "anchor_forecast_emission_hash": anchor_forecast_emission_hash,
            "point_policy_version": point_policy_version,
            "width_rule_version": WIDTH_RULE_VERSION,
            "rounding_rule_version": ROUNDING_RULE_VERSION,
            "selected_width_usd": width,
            "ideal_start_usd": ideal_start,
            "rounded_start_usd": start,
            "boundaries_usd": boundaries,
            "rounding_variant": variant,
        }
        grid_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
        variants.append(
            MarketGrid(
                **identity,
                width_selection_rule=width_selection_rule,
                grid_id=f"movie-grid-{grid_id}",
            )
        )
    return tuple(variants)


def generate_canonical_grid(**kwargs: object) -> MarketGrid:
    return generate_rounding_variants(**kwargs)[0]


def generate_fixed_grid_ensemble(
    *,
    release_run_id: int,
    listing_origin: str,
    anchor_forecast_usd: int | float,
    bucket_counts: tuple[int, ...] = (4, 5, 6),
    empirical_modes: Mapping[tuple[int, ...], int] | None = None,
) -> tuple[SyntheticMarketGrid, ...]:
    """Create a small fixed 4/5/6-bucket robustness ensemble.

    Every member is derived once from the listing point.  Callers must retain
    the returned boundaries for later forecast origins; regenerating from a
    later point would silently recenter the market and invalidate sequential
    probability evaluation.
    """

    anchor = int(round(float(anchor_forecast_usd)))
    width, _ = select_canonical_width(anchor, empirical_modes=empirical_modes)
    grids: list[SyntheticMarketGrid] = []
    for bucket_count in bucket_counts:
        if bucket_count not in (4, 5, 6):
            raise ValueError("synthetic bucket counts must be 4, 5, or 6")
        boundary_count = bucket_count - 1
        # Odd bucket counts can place the point at a bucket centre.  For even
        # counts, retain both adjacent half-width alignments as a robustness
        # check instead of making an arbitrary one-sided choice.
        offsets = (0.0,) if bucket_count % 2 else (-0.5, 0.5)
        for offset in offsets:
            ideal_start = anchor - ((boundary_count - 1) / 2 + offset) * width
            for index, start in enumerate(round_start_to_clean_strikes(ideal_start, width)):
                while start <= 0:
                    start += get_clean_strike_increment(width)
                boundaries = tuple(int(start + step * width) for step in range(boundary_count))
                alignment = "point_centered" if bucket_count % 2 else (
                    "point_left_of_center" if offset < 0 else "point_right_of_center"
                )
                identity = {
                    "release_run_id": int(release_run_id),
                    "listing_origin": str(listing_origin),
                    "anchor_forecast_usd": anchor,
                    "selected_width_usd": width,
                    "bucket_count": bucket_count,
                    "boundaries_usd": boundaries,
                    "alignment": alignment,
                    "rounding_variant": "canonical" if index == 0 else "higher_strike",
                }
                digest = hashlib.sha256(
                    json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()[:20]
                grids.append(SyntheticMarketGrid(**identity, grid_id=f"synthetic-grid-{digest}"))
    return tuple(grids)
