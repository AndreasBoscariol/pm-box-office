"""Write Phase 7 prospective and external-evidence gate statuses."""

from __future__ import annotations

import json
from pathlib import Path


def write_phase7_status(root:str|Path,artifact_checksum:str)->dict[str,str]:
    root=Path(root);review=root/"prediction_market_reviewed_event_universe_v4";prices=root/"prediction_market_historical_price_panel_v4";market=root/"prediction_market_probability_diagnostic_pointscale_cal";capture=root/"prediction_market_clob_capture_acceptance_v4"
    for path in (review,prices,market,capture):path.mkdir(parents=True,exist_ok=True)
    statuses={"historical_development_status":"provisional_beta","prospective_probability_status":"inconclusive_insufficient_prospective_sample","event_panel_status":"pending_manual_review","historical_price_panel_status":"blocked_no_approved_events","historical_market_diagnostic_status":"blocked_no_approved_events","collector_status":"not_passed_24h","production_probability_eligible":"false","trading_status":"disabled"}
    source=Path("data/diagnostics/prediction_market_reviewed_event_universe_v3/01_prioritized_review_queue.csv")
    if source.exists():(review/"01_prioritized_review_queue.csv").write_bytes(source.read_bytes())
    (review/"summary.md").write_text("# Event universe v4\n\nStatus: **pending_manual_review**. Explicit human decisions remain required for all 199 candidates.\n")
    (prices/"summary.md").write_text("# Historical price panel v4\n\nStatus: **blocked_no_approved_events**.\n")
    (market/"01_study_manifest.json").write_text(json.dumps({"study_status":"diagnostic_not_approval","status":"blocked_no_approved_events","frozen_artifact_checksum":artifact_checksum,"model_changes_permitted":False},indent=2)+"\n");(market/"summary.md").write_text("# Locked historical market diagnostic\n\nStatus: **blocked** until reviewed events and timestamp-valid prices exist. Any eventual result is diagnostic, not approval.\n")
    (capture/"01_run_manifest.json").write_text(json.dumps({"status":"not_passed_24h","required_hours":24,"observed_hours":0},indent=2)+"\n");(capture/"summary.md").write_text("# Capture acceptance v4\n\nStatus: **not passed**. No 24-hour wall-clock run has completed.\n")
    (root/"phase7_summary.md").write_text("# Phase 7 status\n\n"+"\n".join(f"- {key}: `{value}`" for key,value in statuses.items())+"\n")
    return statuses
