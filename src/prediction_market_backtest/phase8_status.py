"""Operational Phase 8 status, reported in independent-movie units."""

from __future__ import annotations
import csv,json
from pathlib import Path
from typing import Any

from .prospective import EXPECTED_CHECKSUM,verify_frozen_artifact


def build_phase8_status(root:str|Path="data/diagnostics",artifact_dir:str|Path="models/boxoffice/pre_release_pointscale_cal_001")->dict[str,Any]:
    root=Path(root);_,verification=verify_frozen_artifact(artifact_dir);prospective=root/"pre_release_pointscale_cal_prospective_v2";interim=_json(prospective/"11_interim_status.json")
    review_queue=root/"prediction_market_reviewed_event_universe_v5"/"01_prioritized_review_queue.csv";reviewed=root/"prediction_market_reviewed_event_universe_v5"/"02_reviewed_decisions.csv";queue_count=_csv_count(review_queue);decisions=list(csv.DictReader(reviewed.open())) if reviewed.exists() and reviewed.read_text().strip() else [];approved=sum(row.get("decision")=="approved" for row in decisions)
    capture=_json(root/"prediction_market_clob_capture_acceptance_v5"/"01_run_manifest.json")
    return {"artifact_checksum_status":"verified" if verification.valid else "failed","artifact_checksum":verification.artifact_checksum,
        "post_freeze_movies_discovered":interim.get("post_freeze_movies_discovered",0),"eligible_prospective_movies":interim.get("eligible_movies",0),
        "completed_primary_validation_movies":interim.get("primary_validation_movies",0),"progress_to_30":interim.get("progress_to_30",0),"progress_to_50":interim.get("progress_to_50",0),
        "forecasts_awaiting_actuals":interim.get("movies_awaiting_actuals",0),"event_review_candidates":queue_count,"event_review_decisions":len(decisions),
        "approved_historical_events":approved,"historical_price_panel_status":"blocked_no_approved_events" if not approved else "pending_sync",
        "collector_acceptance_status":capture.get("status","not_started"),"trading_status":"disabled"}


def initialize_phase8_external_outputs(root:str|Path="data/diagnostics")->None:
    root=Path(root);review=root/"prediction_market_reviewed_event_universe_v5";prices=root/"prediction_market_historical_price_panel_v5";market=root/"prediction_market_probability_diagnostic_pointscale_beta";capture=root/"prediction_market_clob_capture_acceptance_v5"
    for path in (review,prices,market,capture):path.mkdir(parents=True,exist_ok=True)
    source=root/"prediction_market_reviewed_event_universe_v4"/"01_prioritized_review_queue.csv"
    if source.exists():(review/"01_prioritized_review_queue.csv").write_bytes(source.read_bytes())
    (review/"summary.md").write_text("# Event review v5\n\nStatus: **pending_manual_review**. Every event requires an explicit reviewer decision; none is auto-approved.\n")
    (prices/"summary.md").write_text("# Historical price panel v5\n\nStatus: **blocked_no_approved_events**.\n")
    (market/"01_study_manifest.json").write_text(json.dumps({"study_status":"diagnostic_not_probability_approval","model_changes_permitted":False,"status":"blocked_no_approved_events","artifact_checksum":EXPECTED_CHECKSUM},indent=2)+"\n");(market/"summary.md").write_text("# Historical point-scale beta diagnostic\n\nStatus: **blocked** pending locked event approvals and price synchronization.\n")
    if not (capture/"01_run_manifest.json").exists():(capture/"01_run_manifest.json").write_text(json.dumps({"status":"not_passed_24h","required_hours":24,"observed_hours":0},indent=2)+"\n")
    (capture/"summary.md").write_text("# CLOB acceptance v5\n\nStatus: **not passed**. The required 24-hour wall-clock run has not completed.\n")
def _json(path:Path)->dict[str,Any]:return json.loads(path.read_text()) if path.exists() and path.read_text().strip() else {}
def _csv_count(path:Path)->int:return sum(1 for _ in csv.DictReader(path.open())) if path.exists() and path.read_text().strip() else 0
