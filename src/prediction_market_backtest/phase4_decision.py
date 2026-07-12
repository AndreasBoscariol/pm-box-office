"""Fail-closed formal Phase 4 decision artifact generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def write_inconclusive_decision(root: str | Path) -> dict[str, object]:
    root=Path(root);study=root/"prediction_market_probability_study_007_real";capture=root/"prediction_market_clob_capture_acceptance"
    study.mkdir(parents=True,exist_ok=True);capture.mkdir(parents=True,exist_ok=True)
    blockers=["cdf_extension_not_approved","zero_manually_approved_events","no_timestamp_valid_historical_market_panel","24_hour_capture_not_completed"]
    manifest={"created_at_utc":datetime.now(timezone.utc).isoformat(),"interval_policy":"007",
        "interval_policy_version":"boxoffice_local_007_weighted_interval_calibration","cdf_extension":None,
        "cdf_status":"not_approved","probability_decision":"inconclusive","trade_recommendations_enabled":False,
        "staleness_grid_minutes":[15,60,360,1440],"blend_weight_grid":[i/10 for i in range(11)],"blockers":blockers}
    (study/"01_study_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    (study/"04_frozen_cdf_extension.json").write_text(json.dumps({"selected":None,"status":"not_approved","selection_used_market_prices":False},indent=2)+"\n")
    for name in ("02_approved_event_universe.csv","03_split_membership.csv","09_score_summary.csv","10_score_by_origin.csv",
        "11_score_by_boundary.csv","12_score_by_bucket_structure.csv","13_threshold_calibration.csv","14_blend_weight_selection.csv",
        "15_price_staleness_sensitivity.csv","16_projection_sensitivity.csv","17_movie_cluster_bootstrap.csv","18_leave_one_movie_out.csv",
        "19_concentration_audit.csv"):(study/name).write_text("")
    (study/"20_rejection_audit.csv").write_text("reason\n"+"\n".join(blockers)+"\n")
    (study/"summary.md").write_text("# Phase 4 probability decision\n\nDecision: **inconclusive**.\n\n"+"\n".join(f"- {item}" for item in blockers)+"\n\nNo trade signals or profitability claims are authorized.\n")
    capture_manifest={"status":"not_passed","required_duration_hours":24,"observed_duration_hours":0,
        "transport_unit_tests_passed":True,"acceptance_window_completed":False}
    (capture/"01_run_manifest.json").write_text(json.dumps(capture_manifest,indent=2)+"\n")
    for name in ("02_subscription_summary.csv","03_connection_events.csv","04_sequence_gap_events.csv","05_rest_reconciliations.csv",
        "06_snapshot_inventory.csv","07_latency_summary.csv","08_health_metrics.csv","09_failure_audit.csv"):(capture/name).write_text("")
    (capture/"summary.md").write_text("# CLOB capture acceptance\n\nStatus: **not passed**. No 24-hour acceptance window has completed. Transport, sequence-gap invalidation, and REST-recovery behavior are covered by automated tests only.\n")
    return manifest
