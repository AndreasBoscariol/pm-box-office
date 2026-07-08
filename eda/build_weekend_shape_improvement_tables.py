"""Build materialized tables for weekend-shape baseline-improvement EDA.

The baseline-improvement notebook can load the residual diagnostic table from
``data/diagnostics/weekend_shape_residual_diagnostic_table.csv`` instead of
recomputing the rolling baselines on every run. Use this script when the
upstream shape baseline or input table changes and the cache should be rebuilt.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).with_name("weekend_shape_baseline_improvement.ipynb")
RESIDUAL_BUILD_CELL_MARKER = "def median_logratio_by_segment"


def code_cells_through_residual_build(notebook_path: Path) -> list[str]:
    notebook = json.loads(notebook_path.read_text())
    cells: list[str] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        cells.append(source)
        if source.lstrip().startswith(RESIDUAL_BUILD_CELL_MARKER):
            break
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the cached weekend-shape residual diagnostic table used by "
            "eda/weekend_shape_baseline_improvement.ipynb."
        )
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=NOTEBOOK_PATH,
        help="Notebook to source setup and residual-build cells from.",
    )
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    if not notebook_path.exists():
        raise FileNotFoundError(notebook_path)

    os.environ["PM_WEEKEND_SHAPE_FORCE_REBUILD"] = "1"
    os.environ["PM_WEEKEND_SHAPE_USE_CACHE"] = "0"

    namespace: dict[str, object] = {"__name__": "__weekend_shape_table_build__"}
    for index, source in enumerate(code_cells_through_residual_build(notebook_path), start=1):
        print(f"Running notebook code cell {index}")
        exec(compile(source, f"{notebook_path.name}:cell_{index}", "exec"), namespace)

    cache_path = namespace.get("RESIDUAL_DIAGNOSTIC_CACHE_PATH")
    residual_table = namespace.get("residual_table")
    row_count = len(residual_table) if residual_table is not None else "unknown"
    print(f"Built {cache_path} ({row_count} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
