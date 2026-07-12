"""Generate separated, fail-closed Phase 5 downstream status artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def write_phase5_status(root:str|Path)->dict[str,str]:
    root=Path(root);price=root/"prediction_market_historical_price_panel_v2";capture=root/"prediction_market_clob_capture_acceptance_v2";study=root/"prediction_market_probability_study_007_probcal_real";review=root/"prediction_market_reviewed_event_universe_v2"
    for path in (price,capture,study,review):path.mkdir(parents=True,exist_ok=True)
    statuses={"probability_model_status":"rejected","event_panel_status":"pending_manual_review","historical_price_panel_status":"blocked_no_approved_events","collector_status":"not_passed_24h","market_comparison_decision":"not_run_prerequisites_failed","trading_status":"disabled"}
    (study/"01_study_manifest.json").write_text(json.dumps({**statuses,"point_policy":"production_pre_release_point","interval_policy":"007","probability_policy":"007_probcal_001","blend_grid":[i/10 for i in range(11)]},indent=2)+"\n")
    (study/"summary.md").write_text("# Phase 5 formal status\n\n"+"\n".join(f"- {key}: `{value}`" for key,value in statuses.items())+"\n\nThe real market study was not run because both required approval gates did not pass.\n")
    (price/"summary.md").write_text("# Historical market price panel v2\n\nStatus: **blocked_no_approved_events**. Price download and pairing must follow locked human review; no unapproved event may enter this panel.\n")
    for name in ("01_token_price_history.csv","02_forecast_price_pairings.csv","03_pairing_rejections.csv","04_raw_market_vectors.csv","05_projected_market_vectors.csv","06_projection_diagnostics.csv","07_price_staleness_summary.csv","08_origin_coverage.csv"):(price/name).write_text("")
    capture_manifest={"status":"not_passed_24h","observed_duration_hours":0,"required_duration_hours":24,"reconnect_evidence":False,"sequence_gap_unit_test":True,"rest_recovery_unit_test":True}
    (capture/"01_run_manifest.json").write_text(json.dumps(capture_manifest,indent=2)+"\n");(capture/"summary.md").write_text("# CLOB capture acceptance v2\n\nStatus: **not passed**. A wall-clock 24-hour run has not been completed.\n")
    source=Path("data/diagnostics/prediction_market_reviewed_event_universe/06_manual_review_queue.csv")
    if source.exists():(review/"01_prioritized_review_queue.csv").write_bytes(source.read_bytes())
    (review/"summary.md").write_text("# Reviewed event universe v2\n\nStatus: **pending_manual_review**. The validated import workflow is available through `prediction-market import-review-decisions`; approvals require reviewer identity, movie ID, resolved warnings, and a lock.\n")
    return statuses
