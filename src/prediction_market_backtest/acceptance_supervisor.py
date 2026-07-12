"""Wait for the immutable formal run, close it out, then continue capture."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from .acceptance_closeout import closeout_capture
from .capture_runtime import run_capture_acceptance


async def supervise_acceptance(token_ids:list[str],formal_dir:str|Path,forward_dir:str|Path,poll_seconds:float=30)->None:
    manifest_path=Path(formal_dir)/"01_run_manifest.json"
    while True:
        if manifest_path.exists():
            manifest=json.loads(manifest_path.read_text())
            if manifest.get("status")!="running":break
        await asyncio.sleep(poll_seconds)
    decision=closeout_capture(formal_dir)
    if decision["decision"]!="passed":return
    # A long-lived, separately labelled sink preserves the formal run as fixed evidence.
    await run_capture_acceptance(token_ids,forward_dir,duration_hours=24*3650,diagnostic_only=True,controlled_faults=False)
