from __future__ import annotations

from datetime import date, datetime

from pm_box_office.sources.amc.blocks import belongs_to_exhibition_business_day


def test_post_midnight_business_day_cutoff() -> None:
    thursday = date(2026, 7, 9)
    assert belongs_to_exhibition_business_day(datetime(2026, 7, 10, 2, 59), exhibition_date=thursday)
    assert not belongs_to_exhibition_business_day(datetime(2026, 7, 10, 3, 0), exhibition_date=thursday)
