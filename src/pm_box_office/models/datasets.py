"""Dataset-loading entrypoints for model code."""

from __future__ import annotations

from typing import Any


def load_amc_training_frame(conn: Any, pd: Any) -> Any:
    from pm_box_office.models.train import load_training_frame

    return load_training_frame(conn, pd)

