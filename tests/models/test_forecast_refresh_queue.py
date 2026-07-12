from __future__ import annotations

import datetime as dt

from models.boxoffice.refresh_queue import opening_friday_for_exhibition_date


def test_opening_friday_for_exhibition_date_maps_preview_and_weekend_days() -> None:
    assert opening_friday_for_exhibition_date(dt.date(2026, 7, 9)) == dt.date(2026, 7, 10)
    assert opening_friday_for_exhibition_date(dt.date(2026, 7, 10)) == dt.date(2026, 7, 10)
    assert opening_friday_for_exhibition_date(dt.date(2026, 7, 11)) == dt.date(2026, 7, 10)
    assert opening_friday_for_exhibition_date(dt.date(2026, 7, 12)) == dt.date(2026, 7, 10)
