"""No-lookahead forecast/book pairing rules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable


@dataclass(frozen=True, slots=True)
class BookTime:
    snapshot_id: int | str
    exchange_time: datetime
    received_time: datetime


def first_eligible_book(books: Iterable[BookTime], forecast_available: datetime, *, latency: timedelta = timedelta(),
                        max_book_age: timedelta | None = None) -> BookTime | None:
    threshold = forecast_available + latency
    for book in sorted(books, key=lambda b: b.received_time):
        if book.received_time < threshold or book.exchange_time > book.received_time:
            continue
        if max_book_age is not None and book.received_time - book.exchange_time > max_book_age:
            continue
        return book
    return None

