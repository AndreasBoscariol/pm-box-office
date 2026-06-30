"""Helpers for database migrations.

Alembic owns schema changes. Runtime code can import this module when it needs a
single canonical place to locate migration configuration.
"""

from __future__ import annotations

from pathlib import Path

from pm_box_office.config import REPO_ROOT


ALEMBIC_INI = REPO_ROOT / "alembic.ini"
ALEMBIC_VERSIONS = REPO_ROOT / "alembic" / "versions"

