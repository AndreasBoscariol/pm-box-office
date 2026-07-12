"""Independent operational gates for Phase 9B."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .acceptance import TARGET_DURATION_HOURS
from .phase8_status import build_phase8_status
from .prospective_actuals import scheduled_look


def build_phase9_status(root: str | Path = "data/diagnostics") -> dict[str, Any]:
    root=Path(root);base=build_phase8_status(root)
    review_dir=root/"prediction_market_reviewed_event_universe_v6"
    decisions=_rows(review_dir/"03_reviewed_event_decisions.csv")
    approved=[row for row in decisions if row.get("decision",row.get("reviewer_decision"))=="approved"]
    rejected=[row for row in decisions if str(row.get("decision",row.get("reviewer_decision",""))).startswith("rejected_")]
    followup=[row for row in decisions if row.get("decision",row.get("reviewer_decision"))=="needs_followup"]
    capture=_json(root/"prediction_market_clob_capture_acceptance_v6"/"01_run_manifest.json")
    capture_dir=root/"prediction_market_clob_capture_acceptance_v6"
    raw_evidence=_jsonl(capture_dir/"raw_events.jsonl");snapshot_evidence=_jsonl(capture_dir/"snapshots.jsonl")
    injected_gaps=sum(row.get("message",{}).get("injected_fault")=="controlled_sequence_gap" for row in raw_evidence)
    gap_recoveries=sum(row.get("reason")=="sequence_gap" and bool(row.get("valid")) for row in snapshot_evidence)
    connect_snapshots=sum(row.get("reason")=="connect_or_reconnect" for row in snapshot_evidence)
    token_count=max(1,len(capture.get("subscribed_tokens",[])));derived_reconnects=max(0,connect_snapshots//token_count-1)
    elapsed=_elapsed(capture);primary=int(base["completed_primary_validation_movies"]);look=scheduled_look(primary)
    panel=_coverage(root/"prediction_market_historical_price_panel_v6"/"10_panel_coverage.csv")
    independent_approved=len({row.get("approved_movie_id") for row in approved if row.get("approved_movie_id")})
    paired_movies=int(panel.get("independent_movies",0));readiness=_sample_label(paired_movies)
    blockers=[]
    if primary<30:blockers.append("prospective_probability_not_approved")
    if capture.get("formal_acceptance_status",capture.get("status"))!="passed":blockers.append("clob_acceptance_not_passed")
    if not approved:blockers.extend(["no_approved_event_semantics","no_locked_movie_matches"])
    return {
      "capture":{"current_run_id":capture.get("run_id",capture.get("start_utc")),"elapsed_duration_hours":elapsed,"target_duration_hours":TARGET_DURATION_HOURS,"formal_acceptance_status":capture.get("formal_acceptance_status",capture.get("status","not_started")),"disconnect_evidence":max(int(capture.get("disconnects",0)),derived_reconnects),"reconnect_evidence":max(int(capture.get("reconnect_successes",0)),derived_reconnects),"sequence_gap_evidence":max(int(capture.get("sequence_gaps",0)),injected_gaps),"recovery_evidence":max(int(capture.get("rest_recovery_successes",0)),gap_recoveries),"write_failures":capture.get("database_write_errors",0),"current_health":"running" if capture.get("status")=="running" else "closed"},
      "review":{"total_candidates":199,"reviewed":len(decisions),"approved":len(approved),"rejected":len(rejected),"followup":len(followup),"remaining":max(0,199-len(decisions)),"approved_independent_movies":independent_approved},
      "historical_panel":{"approved_events_queried":int(panel.get("approved_events_queried",0)),"complete_price_histories":int(panel.get("complete_price_histories",0)),"complete_vectors":int(panel.get("complete_vectors",0)),"timestamp_valid_pairings":int(panel.get("timestamp_valid_pairings",0)),"independent_movies":paired_movies,"diagnostic_readiness":readiness},
      "prospective_cohort":{"post_freeze_movies":base["post_freeze_movies_discovered"],"eligible_movies":base["eligible_prospective_movies"],"forecasts_generated":base.get("forecasts_awaiting_actuals",0)+primary,"actuals_finalized":primary,"primary_validation_count":primary,"progress_to_30":primary/30,"progress_to_50":primary/50,"artifact_checksum_status":base["artifact_checksum_status"],"artifact_checksum":base["artifact_checksum"],"formal_look_allowed":primary>=30,"next_formal_look_status":look[2]},
      "trading":"disabled","trading_blockers":blockers,
    }


def _elapsed(manifest:dict[str,Any])->float:
    if not manifest.get("start_utc"):return 0.0
    end=manifest.get("end_utc")
    if not end and manifest.get("status")=="running":end=datetime.now(timezone.utc).isoformat()
    if not end:return float(manifest.get("duration_hours",0))
    try:return max(0,(datetime.fromisoformat(end)-datetime.fromisoformat(manifest["start_utc"])).total_seconds()/3600)
    except (TypeError,ValueError):return 0.0


def _sample_label(n:int)->str:
    if n<10:return "blocked_small_sample"
    if n<20:return "exploratory"
    if n<30:return "diagnostic_small_sample"
    return "diagnostic"


def _rows(path:Path)->list[dict[str,str]]:
    return list(csv.DictReader(path.open())) if path.exists() and path.read_text().strip() else []
def _json(path:Path)->dict[str,Any]:return json.loads(path.read_text()) if path.exists() and path.read_text().strip() else {}
def _jsonl(path:Path)->list[dict[str,Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
def _coverage(path:Path)->dict[str,str]:
    rows=_rows(path);return rows[-1] if rows else {}
