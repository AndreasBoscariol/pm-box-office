from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.boxoffice.reported_preview_actuals import (
    ReportedPreviewActual,
    THURSDAY_ONLY,
    THURSDAY_PLUS_EARLY_ACCESS,
    normalize_preview_target_type,
)


def test_primary_preview_target_must_be_clean_wide_thursday() -> None:
    actual = ReportedPreviewActual(
        release_run_id=1,
        preview_business_date=datetime(2026, 1, 1).date(),
        preview_gross_usd=5_000_000,
        preview_target_type=THURSDAY_ONLY,
        includes_wednesday_early_access=False,
        preview_days_included=1,
        preview_source="studio",
        preview_published_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        received_at=datetime(2026, 1, 2, 0, 1, tzinfo=timezone.utc),
        is_primary_training_target=True,
        is_wide_release=True,
    )
    actual.validate()


def test_early_access_cannot_be_primary_target() -> None:
    actual = ReportedPreviewActual(
        release_run_id=1,
        preview_business_date=datetime(2026, 1, 1).date(),
        preview_gross_usd=5_000_000,
        preview_target_type=THURSDAY_ONLY,
        includes_wednesday_early_access=True,
        preview_days_included=1,
        preview_source="studio",
        preview_published_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        received_at=datetime(2026, 1, 2, 0, 1, tzinfo=timezone.utc),
        is_primary_training_target=True,
        is_wide_release=True,
    )
    with pytest.raises(ValueError, match="primary targets"):
        actual.validate()


def test_preview_target_type_aliases_are_explicit_not_thursday_only() -> None:
    assert normalize_preview_target_type("wednesday_plus_thursday") == THURSDAY_PLUS_EARLY_ACCESS
    assert normalize_preview_target_type("thursday_only") == THURSDAY_ONLY
