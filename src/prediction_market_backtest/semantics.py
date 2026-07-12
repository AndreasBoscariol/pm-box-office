"""Contract parsing and complete bucket-set validation."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Iterable


MILLION = Decimal("1000000")


@dataclass(frozen=True, slots=True)
class Bucket:
    market_id: str
    lower: Decimal | None
    upper: Decimal | None
    include_lower: bool = True
    include_upper: bool = False
    geographic_scope: str = "domestic_us_canada"
    currency: str = "USD"
    weekend_duration: int = 3
    resolution_source: str | None = None

    def contains(self, value: Decimal | int | str) -> bool:
        value = Decimal(value)
        lower_ok = self.lower is None or value > self.lower or (self.include_lower and value == self.lower)
        upper_ok = self.upper is None or value < self.upper or (self.include_upper and value == self.upper)
        return lower_ok and upper_ok


@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...]


def _money(value: str, unit: str | None) -> Decimal:
    amount = Decimal(value.replace(",", ""))
    return amount * MILLION if unit and unit.lower().startswith("m") else amount


def parse_bucket(market_id: str, wording: str, **metadata: object) -> Bucket:
    """Parse common Polymarket bucket wording without inventing missing semantics."""
    text = wording.lower().replace("–", "-").replace("—", "-")
    if any(term in text for term in ("worldwide", "international", "lifetime", "production budget")):
        raise ValueError("ineligible target metric or geography")
    number = r"\$?\s*([0-9]+(?:\.[0-9]+)?(?:,[0-9]{3})*)\s*(million|m)?"
    between = re.search(number + r"\s*(?:-|to|and)\s*" + number, text)
    under = re.search(r"(?:under|less than|below)\s*" + number, text)
    at_most = re.search(r"(?:at most|no more than)\s*" + number, text)
    over = re.search(r"(?:over|more than|above|greater than)\s*" + number, text)
    at_least = re.search(r"(?:at least)\s*" + number, text)
    number_or_more = re.search(number + r"\s*(?:or more|and above|or above)", text)
    kwargs = {k: v for k, v in metadata.items() if k in Bucket.__dataclass_fields__}
    if between:
        lower = _money(between.group(1), between.group(2))
        upper = _money(between.group(3), between.group(4) or between.group(2))
        include_upper = bool(re.search(r"inclusive|through", text))
        return Bucket(market_id, lower, upper, True, include_upper, **kwargs)
    if under:
        return Bucket(market_id, None, _money(under.group(1), under.group(2)), False, False, **kwargs)
    if at_most:
        return Bucket(market_id, None, _money(at_most.group(1), at_most.group(2)), False, True, **kwargs)
    if at_least:
        return Bucket(market_id, _money(at_least.group(1), at_least.group(2)), None, True, False, **kwargs)
    if number_or_more:
        return Bucket(market_id, _money(number_or_more.group(1), number_or_more.group(2)), None, True, False, **kwargs)
    if over:
        return Bucket(market_id, _money(over.group(1), over.group(2)), None, False, False, **kwargs)
    raise ValueError(f"unrecognized bucket wording: {wording!r}")


def validate_bucket_set(buckets: Iterable[Bucket]) -> ValidationResult:
    items = list(buckets)
    errors: list[str] = []
    if not items:
        return ValidationResult(False, ("empty bucket set",))
    ids = [b.market_id for b in items]
    if len(ids) != len(set(ids)):
        errors.append("duplicate market")
    for field in ("geographic_scope", "currency", "weekend_duration", "resolution_source"):
        if len({getattr(b, field) for b in items}) > 1:
            errors.append(f"inconsistent {field}")
    ordered = sorted(items, key=lambda b: (b.lower is not None, b.lower or Decimal(0)))
    if ordered[0].lower is not None:
        errors.append("missing lower tail")
    if ordered[-1].upper is not None:
        errors.append("missing upper tail")
    for left, right in zip(ordered, ordered[1:]):
        if left.upper is None or right.lower is None:
            errors.append("overlapping unbounded bucket")
            continue
        if left.upper < right.lower:
            errors.append(f"gap before {right.market_id}")
        elif left.upper > right.lower:
            errors.append(f"overlap at {right.market_id}")
        elif left.include_upper == right.include_lower:
            errors.append(("overlap" if left.include_upper else "gap") + f" at boundary {left.upper}")
    return ValidationResult(not errors, tuple(errors))


def with_override(bucket: Bucket, **changes: object) -> Bucket:
    return replace(bucket, **changes)
