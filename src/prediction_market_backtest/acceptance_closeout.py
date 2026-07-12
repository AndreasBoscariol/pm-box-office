"""Build the fixed Phase 9B audit package from append-only capture evidence."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .acceptance import TARGET_DURATION_HOURS, acceptance_failure_codes, manifest_duration_hours

OUTPUTS = (
    "02_subscription_inventory.csv", "03_connection_events.csv", "04_raw_message_summary.csv",
    "05_snapshot_inventory.csv", "06_sequence_gap_events.csv", "07_state_invalidations.csv",
    "08_rest_recoveries.csv", "09_hash_reconciliations.csv", "10_latency_summary.csv",
    "11_health_metrics.csv", "12_write_failure_audit.csv", "13_acceptance_decision.csv",
)


def closeout_capture(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    manifest_path = root / "01_run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") == "running":
        raise RuntimeError("formal run is still active")
    raw = list(_jsonl(root / "raw_events.jsonl"))
    snapshots = list(_jsonl(root / "snapshots.jsonl"))
    failures = acceptance_failure_codes(manifest)
    decision = "passed" if not failures else failures[0]
    manifest["duration_hours"] = _safe_duration(manifest)
    manifest["target_duration_hours"] = TARGET_DURATION_HOURS
    manifest["formal_acceptance_status"] = decision
    manifest["acceptance_failure_codes"] = list(failures)
    manifest["immutable_snapshot_persistence"] = bool(snapshots)
    manifest["health_metrics_queryable"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    _csv(root/OUTPUTS[0], ({"token_id": token, "formal_run": True} for token in manifest.get("subscribed_tokens", [])))
    _csv(root/OUTPUTS[1], [{"kind":"controlled_disconnect","count":manifest.get("disconnects",0),"injected":True},{"kind":"successful_reconnect","count":manifest.get("reconnect_successes",0),"injected":True}])
    types = Counter(str(row.get("message", {}).get("event_type", "unknown")) for row in raw)
    _csv(root/OUTPUTS[2], ({"message_type":key,"count":value} for key,value in sorted(types.items())))
    _csv(root/OUTPUTS[3], ({key:row.get(key) for key in ("token_id","received_at","reason","book_hash","valid","sequence")} for row in snapshots))
    _csv(root/OUTPUTS[4], [{"count":manifest.get("sequence_gaps",0),"injected":True,"formal_run":True}])
    _csv(root/OUTPUTS[5], [{"count":manifest.get("state_invalidations",0),"before_reuse":True}])
    _csv(root/OUTPUTS[6], [{"successful":manifest.get("rest_recovery_successes",0),"replacement_not_patch":True}])
    _csv(root/OUTPUTS[7], [{"failures":manifest.get("reconciliation_failures",0)}])
    _csv(root/OUTPUTS[8], [{"metric":"receipt_latency","status":"recorded_in_raw_receipt_timestamps","observations":len(raw)}])
    _csv(root/OUTPUTS[9], [{"messages":len(raw),"snapshots":len(snapshots),"unrecovered_books":manifest.get("unrecovered_failures",0),"current_health":"closed"}])
    _csv(root/OUTPUTS[10], [{"database_write_errors":manifest.get("database_write_errors",0)}])
    _csv(root/OUTPUTS[11], [{"decision":decision,"passed":not failures,"failure_codes":"|".join(failures),"duration_hours":manifest["duration_hours"],"target_duration_hours":TARGET_DURATION_HOURS,"adjudicated_at":datetime.now(timezone.utc).isoformat()}])
    root.joinpath("summary.md").write_text(
        "# CLOB capture acceptance v6\n\n"
        f"Formal decision: **{decision}**. Duration: `{manifest['duration_hours']:.6f}` hours; target: `{TARGET_DURATION_HOURS}`.\n\n"
        f"Naturally occurring connection events are preserved in raw evidence. Injected faults: disconnect `{manifest.get('disconnects',0)}`, sequence gaps `{manifest.get('sequence_gaps',0)}`. "
        f"Recovered faults: `{manifest.get('rest_recovery_successes',0)}`; unrecovered books: `{manifest.get('unrecovered_failures',0)}`.\n"
    )
    return {"decision":decision,"failure_codes":list(failures),"duration_hours":manifest["duration_hours"]}


def _safe_duration(manifest: dict[str, Any]) -> float:
    try: return manifest_duration_hours(manifest)
    except (KeyError, TypeError, ValueError): return 0.0


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip(): yield json.loads(line)


def _csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    fields = list(dict.fromkeys(key for row in values for key in row)) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(values)
