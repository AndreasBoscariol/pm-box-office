"""Immutable provisional artifact and prospective validation initialization."""

from __future__ import annotations

import hashlib,json,subprocess
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .asymdist import fit_conditional_split_t
from .probcal import fit_beta_calibration


def freeze_beta_artifact(panel_path:str|Path,development_dir:str|Path,artifact_dir:str|Path)->dict[str,Any]:
    panel_path=Path(panel_path);development_dir=Path(development_dir);artifact_dir=Path(artifact_dir)
    if artifact_dir.exists() and (artifact_dir/"manifest.json").exists():
        existing=json.loads((artifact_dir/"manifest.json").read_text());expected=existing.get("artifact_checksum");actual=_artifact_checksum({k:v for k,v in existing.items() if k!="artifact_checksum"})
        if expected!=actual:raise RuntimeError("frozen artifact was modified in place")
        return existing
    panel=pd.read_csv(panel_path);panel["opening_weekend_start"]=pd.to_datetime(panel.opening_weekend_start);valid=panel.loc[panel.primary_point_forecast_usd.gt(0)&panel.actual_opening_weekend_gross_usd.gt(0)&panel.release_width_bucket.astype(str).isin(["wide","large_wide"])]
    parameter_history=pd.read_csv(development_dir/"05_calibration_parameters.csv");base_penalty=float(parameter_history.iloc[-1].base_penalty);beta_penalty=float(parameter_history.iloc[-1].beta_penalty)
    base=fit_conditional_split_t(__import__('numpy').log(valid.actual_opening_weekend_gross_usd/valid.primary_point_forecast_usd),valid.primary_point_forecast_usd,valid.origin_day,valid.movie_id.astype(int),"point_scale_split_student_t",base_penalty)
    phase6=pd.read_csv("data/diagnostics/pre_release_asymdist_validation/06_pit_values.csv");pits=phase6.loc[phase6.candidate.eq("point_scale_split_student_t")]
    beta=fit_beta_calibration(pits.pit,pits.movie_id.astype(int),beta_penalty);now=datetime.now(timezone.utc).isoformat();commit=_git_commit()
    manifest={"artifact":"pre_release_pointscale_cal_001","status":"provisional_awaiting_prospective_validation","production_probability_eligible":False,
        "point_policy":"production_pre_release_point","interval_policy":"007","base_distribution_policy":"pre_release_pointscale_splitt_001","probability_policy":"pre_release_pointscale_cal_001","calibration_candidate":"beta",
        "base_parameters":asdict(base),"calibration_parameters":{"alpha":beta.alpha,"beta":beta.beta,"regularization":beta_penalty},"minimum_prior_movies":50,"minimum_prior_movies_per_group":25,"fallback":"global beta calibration","threshold_grid_usd":[10e6,15e6,20e6,25e6,30e6,35e6,40e6,50e6,60e6,75e6,100e6,125e6,150e6],
        "training_data_checksum":hashlib.sha256(panel_path.read_bytes()).hexdigest(),"source_commit":commit,"source_dirty_at_freeze":True,"latest_included_movie_id":int(valid.sort_values("opening_weekend_start").iloc[-1].movie_id),"latest_included_release_date":str(valid.opening_weekend_start.max().date()),"artifact_created_utc":now,"artifact_freeze_utc":now,"minimum_prospective_movies":30,"target_prospective_movies":50,"primary_origin_policy":"latest_valid_pre_release_origin","historical_development_metrics":{"outcome":"provisional_beta","outer_test_movies":51}}
    manifest["artifact_checksum"]=_artifact_checksum(manifest);artifact_dir.mkdir(parents=True,exist_ok=False);(artifact_dir/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True,default=list)+"\n");return manifest


def initialize_prospective(artifact_manifest:dict[str,Any],panel_path:str|Path,output_dir:str|Path)->dict[str,Any]:
    panel=pd.read_csv(panel_path);panel["opening_weekend_start"]=pd.to_datetime(panel.opening_weekend_start);freeze=pd.Timestamp(artifact_manifest["artifact_freeze_utc"]);eligible=panel.loc[panel.opening_weekend_start.dt.tz_localize("UTC")>freeze].copy();output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    validation={"artifact_checksum":artifact_manifest["artifact_checksum"],"freeze_utc":artifact_manifest["artifact_freeze_utc"],"minimum_prospective_movies":30,"target_prospective_movies":50,"primary_origin_policy":"latest_valid_pre_release_origin","model_updates_allowed":False,"market_availability_affects_eligibility":False}
    (output/"01_validation_manifest.json").write_text(json.dumps(validation,indent=2)+"\n");eligible.to_csv(output/"02_eligible_movie_inventory.csv",index=False)
    for name in ("03_forecast_distribution_index.csv","04_primary_latest_origin_panel.csv","05_all_origin_panel.csv","06_pit_summary.csv","07_threshold_calibration.csv","08_crps_comparison.csv","09_threshold_log_comparison.csv","10_rps_comparison.csv","11_tail_balance.csv","12_movie_cluster_bootstrap.csv","13_leave_one_movie_out.csv","14_concentration_audit.csv"):(output/name).write_text("")
    movies=int(eligible.movie_id.nunique());decision="inconclusive_insufficient_prospective_sample" if movies<30 else "pending_evaluation";(output/"15_formal_decision.csv").write_text(f"decision,prospective_movies\n{decision},{movies}\n");summary={"decision":decision,"prospective_movies":movies,"required_movies":30,"artifact_frozen":True,"production_probability_eligible":False};(output/"summary.md").write_text("# Prospective point-scale calibration validation\n\n"+"\n".join(f"- {k}: `{v}`" for k,v in summary.items())+"\n");return summary


def _artifact_checksum(payload:dict[str,Any])->str:return hashlib.sha256(json.dumps(payload,sort_keys=True,default=list,separators=(",",":")).encode()).hexdigest()
def _git_commit()->str:
    try:return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except Exception:return "unknown"
