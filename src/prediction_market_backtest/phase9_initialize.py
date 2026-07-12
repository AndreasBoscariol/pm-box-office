"""Initialize versioned external-evidence outputs without inventing reviews."""

from __future__ import annotations
import csv,hashlib,json,re
from pathlib import Path
from typing import Any

from .historical_panel import initialize_blocked_panel


def initialize_phase9(root:str|Path="data/diagnostics")->dict[str,Any]:
    root=Path(root);review=root/"prediction_market_reviewed_event_universe_v6";review.mkdir(parents=True,exist_ok=True);source=root/"prediction_market_reviewed_event_universe_v5"/"01_prioritized_review_queue.csv";rows=list(csv.DictReader(source.open())) if source.exists() and source.read_text().strip() else []
    for row in rows:
        warnings=str(row.get("automated_warnings") or "");buckets=str(row.get("bucket_table_json") or "");row["priority_score"]=str(8-int(bool(warnings))*3+int(bool(row.get("proposed_movie_id")))*2+int('"lower": null' in buckets and '"upper": null' in buckets)*2+int(bool(row.get("full_resolution_rules"))))
        row["rule_template_id"]=_template_id(str(row.get("full_resolution_rules") or ""))
    rows.sort(key=lambda row:(-int(row["priority_score"]),row.get("event_id","")));_write(review/"01_prioritized_review_queue.csv",rows)
    for name in ("02_review_import_log.csv","03_reviewed_event_decisions.csv","04_approved_events.csv","05_rejected_events.csv","06_followup_events.csv","07_locked_movie_matches.csv","08_bucket_semantics_audit.csv","09_review_summary.csv"):
        path=review/name
        if not path.exists():path.write_text("")
    existing=list(csv.DictReader((review/"03_reviewed_event_decisions.csv").open())) if (review/"03_reviewed_event_decisions.csv").exists() and (review/"03_reviewed_event_decisions.csv").read_text().strip() else []
    approved=sum(row.get("decision")=="approved" for row in existing);rejected=sum(str(row.get("decision","")).startswith("rejected_") for row in existing);followup=sum(row.get("decision")=="needs_followup" for row in existing)
    summary={"total_candidates":len(rows),"reviewed_count":len(existing),"remaining_count":max(0,len(rows)-len(existing)),"approved_count":approved,"rejected_count":rejected,"followup_count":followup};(review/"summary.md").write_text("# Reviewed event universe v6\n\n"+"\n".join(f"- {k}: `{v}`" for k,v in summary.items())+"\n\nHuman decisions are required; template grouping never changes eligibility.\n")
    price=initialize_blocked_panel(root/"prediction_market_historical_price_panel_v6",0);diagnostic=root/"prediction_market_probability_diagnostic_pointscale_beta_v2";diagnostic.mkdir(parents=True,exist_ok=True);manifest={"study_role":"historical_diagnostic_only","probability_approval_eligible":False,"trading_approval_eligible":False,"status":"blocked_prerequisites","frozen_artifact_checksum":"a4748ccc38e2e7af7593cde0fbfd9db748c0594f5c716a39d79bd8564a1b24d9","blend_grid":[i/10 for i in range(11)]};(diagnostic/"01_study_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    for name in ("02_approved_event_panel.csv","03_model_probability_vectors.csv","04_market_probability_vectors.csv","05_complete_diagnostic_panel.csv","06_score_summary.csv","07_score_by_origin.csv","08_score_by_boundary.csv","09_blend_grid_results.csv","10_staleness_sensitivity.csv","11_projection_sensitivity.csv","12_movie_cluster_bootstrap.csv","13_leave_one_movie_out.csv","14_concentration_audit.csv","15_diagnostic_decision.csv"):
        path=diagnostic/name
        if not path.exists():path.write_text("")
    (diagnostic/"summary.md").write_text("# Locked historical model-versus-market diagnostic\n\nStatus: **blocked_prerequisites**. This study is diagnostic only and can never approve probabilities or trading.\n")
    capture=root/"prediction_market_clob_capture_acceptance_v6";capture.mkdir(parents=True,exist_ok=True);capture_manifest=capture/"01_run_manifest.json"
    if not capture_manifest.exists():capture_manifest.write_text(json.dumps({"status":"not_passed_24h","duration_hours":0,"controlled_preflight":"prediction_market_clob_capture_acceptance_v6_controlled_smoke","reconnect_successes":0,"sequence_gaps":0},indent=2)+"\n")
    (capture/"summary.md").write_text("# CLOB capture acceptance v6\n\nStatus: **not passed**. Controlled reconnect/gap recovery passed in the short preflight; the supervised 24-hour window remains outstanding.\n")
    return {"review":summary,"prices":price,"diagnostic":manifest}
def _write(path:Path,rows:list[dict[str,str]])->None:
    if not rows:path.write_text("");return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w",newline="",encoding="utf-8") as handle:writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
def _template_id(rules:str)->str:
    normalized=re.sub(r'"[^"]+"','"<TITLE>"',rules.lower())
    normalized=re.sub(r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:\s*-\s*(?:[a-z]+\s+)?\d{1,2})?', '<DATE>', normalized)
    normalized=re.sub(r'\b\d{4}\b|\b\d{1,2}:\d{2}\b','<N>',normalized)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]
