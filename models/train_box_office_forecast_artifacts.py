#!/usr/bin/env python3
"""CLI wrapper for freezing box-office forecast artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.boxoffice.train_box_office_forecast_artifacts import main


if __name__ == "__main__":
    raise SystemExit(main())
