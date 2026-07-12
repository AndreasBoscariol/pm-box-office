"""Deterministic public CLOB order-book reconstruction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterable, Mapping


@dataclass(slots=True)
class OrderBook:
    token_id: str
    bids: dict[Decimal, Decimal] = field(default_factory=dict)
    asks: dict[Decimal, Decimal] = field(default_factory=dict)
    sequence: int | None = None
    tick_size: Decimal | None = None
    valid: bool = True

    def snapshot(self, bids: Iterable[tuple[object, object]], asks: Iterable[tuple[object, object]], sequence: int | None = None) -> None:
        self.bids = _levels(bids)
        self.asks = _levels(asks)
        self.sequence = sequence
        self.valid = True
        self._check()

    def update(self, side: str, price: object, size: object, sequence: int | None = None) -> bool:
        if sequence is not None and self.sequence is not None:
            if sequence <= self.sequence:
                return False
            if sequence != self.sequence + 1:
                self.valid = False
                raise SequenceGap(f"expected {self.sequence + 1}, got {sequence}")
        levels = self.bids if side.lower() in {"buy", "bid", "bids"} else self.asks
        price_d, size_d = Decimal(str(price)), Decimal(str(size))
        if size_d <= 0:
            levels.pop(price_d, None)
        else:
            levels[price_d] = size_d
        self.sequence = sequence if sequence is not None else self.sequence
        self._check()
        return True

    @property
    def best_bid(self) -> Decimal | None:
        return max(self.bids, default=None)

    @property
    def best_ask(self) -> Decimal | None:
        return min(self.asks, default=None)

    @property
    def book_hash(self) -> str:
        payload = {"token_id": self.token_id, "bids": _serialized(self.bids, True), "asks": _serialized(self.asks, False)}
        return hashlib.sha256(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()).hexdigest()

    def _check(self) -> None:
        if self.best_bid is not None and self.best_ask is not None and self.best_bid >= self.best_ask:
            self.valid = False


class SequenceGap(RuntimeError):
    pass


def _levels(values: Iterable[tuple[object, object]]) -> dict[Decimal, Decimal]:
    return {Decimal(str(p)): Decimal(str(s)) for p, s in values if Decimal(str(s)) > 0}


def _serialized(values: Mapping[Decimal, Decimal], reverse: bool) -> list[list[str]]:
    return [[str(p), str(values[p])] for p in sorted(values, reverse=reverse)]

