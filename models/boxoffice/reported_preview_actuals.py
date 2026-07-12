"""Canonical preview-gross targets with publication-time availability."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd


THURSDAY_ONLY = "thursday_only"
THURSDAY_PLUS_EARLY_ACCESS = "thursday_plus_early_access"
MULTI_DAY = "multi_day_preview_total"
NO_CONVENTIONAL_PREVIEW = "no_conventional_thursday_preview"
TARGET_TYPE_ALIASES = {
    "wednesday_plus_thursday": THURSDAY_PLUS_EARLY_ACCESS,
    "wednesday_early_access_plus_thursday": THURSDAY_PLUS_EARLY_ACCESS,
    "early_access_plus_thursday": THURSDAY_PLUS_EARLY_ACCESS,
    "multi_day": MULTI_DAY,
}
KNOWN_TARGET_TYPES = {
    THURSDAY_ONLY,
    THURSDAY_PLUS_EARLY_ACCESS,
    MULTI_DAY,
    NO_CONVENTIONAL_PREVIEW,
    "legacy_unclassified",
}


def normalize_preview_target_type(value: Any) -> str:
    raw = str(value or "").strip().lower()
    normalized = TARGET_TYPE_ALIASES.get(raw, raw)
    if normalized not in KNOWN_TARGET_TYPES:
        raise ValueError(f"unknown preview_target_type: {value}")
    return normalized


@dataclass(frozen=True)
class ReportedPreviewActual:
    release_run_id: int
    preview_business_date: date
    preview_gross_usd: float
    preview_target_type: str
    includes_wednesday_early_access: bool
    preview_days_included: int
    preview_source: str
    preview_published_at: datetime
    received_at: datetime
    is_primary_training_target: bool = False
    is_wide_release: bool | None = None
    preview_source_url: str | None = None
    revision: int = 1
    supersedes_id: int | None = None

    def validate(self) -> None:
        if self.preview_gross_usd <= 0:
            raise ValueError("preview_gross_usd must be positive")
        if normalize_preview_target_type(self.preview_target_type) not in KNOWN_TARGET_TYPES:
            raise ValueError(f"unknown preview_target_type: {self.preview_target_type}")
        if self.preview_days_included < 1:
            raise ValueError("preview_days_included must be positive")
        if self.received_at < self.preview_published_at:
            raise ValueError("received_at cannot precede preview_published_at")
        if self.is_primary_training_target and not self.is_clean_thursday_target:
            raise ValueError("primary targets must be wide, Thursday-only previews")

    @property
    def is_clean_thursday_target(self) -> bool:
        return (
            self.preview_target_type == THURSDAY_ONLY
            and not self.includes_wednesday_early_access
            and self.preview_days_included == 1
            and self.is_wide_release is True
        )


def upsert_reported_preview_actual(conn: Any, actual: ReportedPreviewActual) -> None:
    actual.validate()
    conn.execute(
        """
        INSERT INTO reported_preview_actuals (
            release_run_id, preview_business_date, preview_gross_usd, preview_target_type,
            includes_wednesday_early_access, preview_days_included, preview_source,
            preview_source_url, preview_published_at, received_at,
            is_primary_training_target, is_wide_release, revision, supersedes_id
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (release_run_id, preview_source, preview_published_at, revision)
        DO UPDATE SET
            preview_gross_usd = EXCLUDED.preview_gross_usd,
            preview_target_type = EXCLUDED.preview_target_type,
            includes_wednesday_early_access = EXCLUDED.includes_wednesday_early_access,
            preview_days_included = EXCLUDED.preview_days_included,
            preview_source_url = EXCLUDED.preview_source_url,
            received_at = EXCLUDED.received_at,
            is_primary_training_target = EXCLUDED.is_primary_training_target,
            is_wide_release = EXCLUDED.is_wide_release,
            supersedes_id = EXCLUDED.supersedes_id
        """,
        (
            actual.release_run_id, actual.preview_business_date, actual.preview_gross_usd, actual.preview_target_type,
            actual.includes_wednesday_early_access, actual.preview_days_included,
            actual.preview_source, actual.preview_source_url, actual.preview_published_at,
            actual.received_at, actual.is_primary_training_target, actual.is_wide_release,
            actual.revision, actual.supersedes_id,
        ),
    )


def fetch_reported_preview_actuals(
    conn: Any,
    *,
    as_of_utc: datetime | None = None,
    primary_only: bool = False,
) -> pd.DataFrame:
    clauses = ["superseded.reported_preview_actual_id IS NULL"]
    params: list[object] = []
    if as_of_utc is not None:
        clauses.extend(["r.preview_published_at <= %s", "r.received_at <= %s"])
        params.extend([as_of_utc, as_of_utc])
    if primary_only:
        clauses.append("r.is_primary_training_target = TRUE")
    cursor = conn.execute(
        f"""
        SELECT r.*
        FROM reported_preview_actuals r
        LEFT JOIN reported_preview_actuals superseded ON superseded.supersedes_id = r.reported_preview_actual_id
        WHERE {' AND '.join(clauses)}
        ORDER BY r.preview_published_at, r.release_run_id, r.revision
        """,
        params,
    )
    return pd.DataFrame(cursor.fetchall(), columns=[item[0] for item in cursor.description])


def legacy_preview_backfill_rows(conn: Any) -> int:
    """Import legacy preview rows conservatively; classification is required for training."""
    cursor = conn.execute(
        """
        INSERT INTO reported_preview_actuals (
            release_run_id, preview_business_date, preview_gross_usd, preview_target_type,
            includes_wednesday_early_access, preview_days_included, preview_source,
            preview_source_url, preview_published_at, received_at,
            is_primary_training_target, revision
        )
        SELECT dbo.release_run_id, dbo.box_office_date::date, dbo.gross_usd, 'legacy_unclassified', FALSE, 1,
               dbo.source, dbo.source_url, dbo.fetched_at::timestamptz,
               dbo.fetched_at::timestamptz, FALSE, 1
        FROM daily_box_office dbo
        WHERE dbo.is_preview = 1 AND dbo.gross_usd > 0
        ON CONFLICT (release_run_id, preview_source, preview_published_at, revision) DO NOTHING
        """
    )
    return int(getattr(cursor, "rowcount", 0) or 0)
