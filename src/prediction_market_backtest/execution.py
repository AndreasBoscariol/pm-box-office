"""Fee-aware quoted-liquidity IOC simulation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_UP
from typing import Callable, Iterable


D = Decimal


@dataclass(frozen=True, slots=True)
class Fill:
    price: Decimal
    shares: Decimal
    premium: Decimal
    fee: Decimal


@dataclass(frozen=True, slots=True)
class Execution:
    fills: tuple[Fill, ...]
    requested_shares: Decimal
    filled_shares: Decimal
    unfilled_shares: Decimal
    premium: Decimal
    fees: Decimal
    vwap: Decimal | None
    status: str


def polymarket_fee(shares: Decimal, price: Decimal, rate: Decimal, quantum: Decimal = D("0.0001")) -> Decimal:
    return (shares * rate * price * (D(1) - price)).quantize(quantum, rounding=ROUND_UP)


def walk_asks(
    asks: Iterable[tuple[object, object]], requested_shares: object, *, max_price: object = 1,
    fee_rate: object = 0, risk_budget: object | None = None,
    fee_fn: Callable[[Decimal, Decimal, Decimal], Decimal] = polymarket_fee,
) -> Execution:
    requested, cap, rate = D(str(requested_shares)), D(str(max_price)), D(str(fee_rate))
    budget = D(str(risk_budget)) if risk_budget is not None else None
    remaining, spent = requested, D(0)
    fills: list[Fill] = []
    for raw_price, raw_size in sorted(asks, key=lambda level: D(str(level[0]))):
        price, available = D(str(raw_price)), D(str(raw_size))
        if remaining <= 0 or price > cap or available <= 0:
            continue
        quantity = min(remaining, available)
        if budget is not None:
            per_share = price + rate * price * (D(1) - price)
            quantity = min(quantity, max(D(0), (budget - spent) / per_share))
        if quantity <= 0:
            break
        premium = quantity * price
        fee = fee_fn(quantity, price, rate)
        if budget is not None and spent + premium + fee > budget:
            quantity = max(D(0), quantity - (spent + premium + fee - budget) / (price or D(1)))
            premium, fee = quantity * price, fee_fn(quantity, price, rate)
        if quantity <= 0:
            break
        fills.append(Fill(price, quantity, premium, fee))
        spent += premium + fee
        remaining -= quantity
    filled = sum((f.shares for f in fills), D(0))
    premium = sum((f.premium for f in fills), D(0))
    fees = sum((f.fee for f in fills), D(0))
    status = "no_fill" if not fills else ("full_fill" if remaining <= D("0.00000001") else "partial_fill")
    return Execution(tuple(fills), requested, filled, max(D(0), requested - filled), premium, fees,
                     premium / filled if filled else None, status)


def expected_value(probability: object, execution: Execution) -> Decimal:
    if not execution.filled_shares:
        return D(0)
    return D(str(probability)) * execution.filled_shares - execution.premium - execution.fees

