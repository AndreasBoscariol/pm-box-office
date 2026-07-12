"""Write separated Phase 6 gate and downstream status artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def write_phase6_status(root:str|Path)->dict[str,str]:
    root=Path(root);review=root/"prediction_market_reviewed_event_universe_v3";prices=root/"prediction_market_historical_price_panel_v3";capture=root/"prediction_market_clob_capture_acceptance_v3";study=root/"prediction_market_probability_study_asymdist_real"
    for path in (review,prices,capture,study):path.mkdir(parents=True,exist_ok=True)
    rejected={"status":"rejected","production_probability_eligible":False,"policies":["007_cdf_extension_001","007_anchor_preserving_empirical_warp_001","007_raw_weighted_empirical_benchmark","007_probcal_001","all Phase 5 beta/isotonic/location candidates","pre_release_asymdist_001"]}
    (study/"rejected_probability_registry.json").write_text(json.dumps(rejected,indent=2)+"\n")
    statuses={"probability_model_status":"rejected","event_panel_status":"pending_manual_review","historical_price_panel_status":"blocked_no_approved_events","collector_status":"not_passed_24h","market_study_status":"not_run_prerequisites_failed","trading_status":"disabled"}
    (study/"01_study_manifest.json").write_text(json.dumps({**statuses,"point_policy":"production_pre_release_point","interval_policy":"007","probability_policy":"pre_release_asymdist_001"},indent=2)+"\n");(study/"summary.md").write_text("# Phase 6 formal status\n\n"+"\n".join(f"- {key}: `{value}`" for key,value in statuses.items())+"\n\nThe native models improved several proper scores but failed the frozen calibration gate. No real market study or trading logic was run.\n")
    source=Path("data/diagnostics/prediction_market_reviewed_event_universe_v2/01_prioritized_review_queue.csv")
    if source.exists():(review/"01_prioritized_review_queue.csv").write_bytes(source.read_bytes())
    (review/"summary.md").write_text("# Event universe v3\n\nStatus: **pending_manual_review**. All 199 candidates still require explicit human decisions through the validated import workflow.\n")
    (prices/"summary.md").write_text("# Historical price panel v3\n\nStatus: **blocked_no_approved_events**. No prices were admitted before locked event approval.\n")
    (capture/"01_run_manifest.json").write_text(json.dumps({"status":"not_passed_24h","required_hours":24,"observed_hours":0},indent=2)+"\n");(capture/"summary.md").write_text("# Capture acceptance v3\n\nStatus: **not passed**. Automated transport/recovery tests pass, but no 24-hour wall-clock acceptance run has completed.\n")
    return statuses
