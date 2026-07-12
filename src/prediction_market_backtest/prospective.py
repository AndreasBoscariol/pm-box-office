"""Fail-closed prospective shadow forecasting for the frozen beta artifact."""

from __future__ import annotations

import csv,hashlib,json
from dataclasses import asdict,dataclass
from datetime import datetime,time,timezone
from pathlib import Path
from typing import Any,Iterable

import numpy as np
import pandas as pd

from .asymdist import ConditionalSplitStudentT
from .probcal import BetaCalibration


EXPECTED_ARTIFACT="pre_release_pointscale_cal_001"
EXPECTED_CHECKSUM="a4748ccc38e2e7af7593cde0fbfd9db748c0594f5c716a39d79bd8564a1b24d9"
THRESHOLDS=np.array([10,15,20,25,30,35,40,50,60,75,100,125,150],float)*1_000_000


@dataclass(frozen=True,slots=True)
class ArtifactVerification:
    checked_at_utc:str
    artifact_checksum:str
    expected_checksum:str
    valid:bool
    reason:str


@dataclass(frozen=True,slots=True)
class CohortDecision:
    movie_id:int
    title:str
    opening_weekend_start:str
    eligibility_status:str
    exclusion_reason:str|None
    first_supported_origin:int|None
    latest_valid_origin:int|None
    polymarket_available:bool|None


def verify_frozen_artifact(artifact_dir:str|Path,expected_checksum:str=EXPECTED_CHECKSUM)->tuple[dict[str,Any],ArtifactVerification]:
    path=Path(artifact_dir)/"manifest.json";manifest=json.loads(path.read_text());stored=str(manifest.get("artifact_checksum") or "");payload={key:value for key,value in manifest.items() if key!="artifact_checksum"};calculated=hashlib.sha256(json.dumps(payload,sort_keys=True,default=list,separators=(",",":")).encode()).hexdigest();reasons=[]
    if manifest.get("artifact")!=EXPECTED_ARTIFACT:reasons.append("artifact identifier mismatch")
    if manifest.get("probability_policy")!=EXPECTED_ARTIFACT:reasons.append("probability policy mismatch")
    if manifest.get("status")!="provisional_awaiting_prospective_validation":reasons.append("artifact status mismatch")
    if manifest.get("production_probability_eligible") is not False:reasons.append("unexpected production eligibility")
    if stored!=calculated:reasons.append("manifest content checksum mismatch")
    if stored!=expected_checksum:reasons.append("frozen checksum mismatch")
    verification=ArtifactVerification(datetime.now(timezone.utc).isoformat(),stored,expected_checksum,not reasons,"; ".join(reasons) or "verified")
    if reasons:raise RuntimeError(verification.reason)
    return manifest,verification


def cohort_inventory(panel:pd.DataFrame,manifest:dict[str,Any],as_of_utc:datetime)->list[CohortDecision]:
    cutoff=pd.Timestamp(manifest["latest_included_release_date"]);frame=panel.copy();frame["opening_weekend_start"]=pd.to_datetime(frame.opening_weekend_start);post=frame.loc[frame.opening_weekend_start>cutoff].sort_values(["opening_weekend_start","movie_id","origin_day"]);decisions=[]
    for movie_id,rows in post.groupby("movie_id"):
        row=rows.iloc[0];reason=None
        if str(row.get("release_width_bucket","")).lower() not in {"wide","large_wide"}:reason="outside_locked_wide_release_universe"
        elif pd.isna(row.get("opening_weekend_start")):reason="missing_opening_weekend_date"
        origins=sorted({int(value) for value in pd.to_numeric(rows.origin_day,errors="coerce").dropna() if -14<=int(value)<=-1})
        if not origins and reason is None:reason="no_supported_pre_release_origin"
        decisions.append(CohortDecision(int(movie_id),str(row.get("title") or ""),str(row.opening_weekend_start.date()),"excluded" if reason else "eligible",reason,origins[0] if origins else None,origins[-1] if origins else None,None))
    return decisions


def generate_shadow_forecasts(panel_path:str|Path,artifact_dir:str|Path,output_dir:str|Path,*,as_of_utc:datetime|None=None,configured_seed:int=42)->dict[str,Any]:
    as_of_utc=as_of_utc or datetime.now(timezone.utc);manifest,verification=verify_frozen_artifact(artifact_dir);panel=pd.read_csv(panel_path);decisions=cohort_inventory(panel,manifest,as_of_utc);output=Path(output_dir);records=output/"records";records.mkdir(parents=True,exist_ok=True)
    _append_csv(output/"01_frozen_artifact_verification.csv",[asdict(verification)],("artifact_checksum","checked_at_utc"));_write_csv(output/"02_post_freeze_movie_inventory.csv",[asdict(row) for row in decisions]);_write_csv(output/"03_eligibility_decisions.csv",[asdict(row) for row in decisions])
    base=ConditionalSplitStudentT(**manifest["base_parameters"]);beta=BetaCalibration(float(manifest["calibration_parameters"]["alpha"]),float(manifest["calibration_parameters"]["beta"]));logs=[];index=[];failures=[]
    eligible={row.movie_id for row in decisions if row.eligibility_status=="eligible"};frame=panel.loc[panel.movie_id.isin(eligible)].copy()
    for _,row in frame.sort_values(["opening_weekend_start","origin_day"]).iterrows():
        movie_id=int(row.movie_id);origin=int(row.origin_day);forecast_date=pd.Timestamp(row.forecast_origin_date).date();cutoff=datetime.combine(forecast_date,time.max,timezone.utc);opening=pd.Timestamp(row.opening_weekend_start).tz_localize("UTC")
        key=f"{movie_id}:{origin}:{manifest['artifact_checksum']}"
        try:
            if cutoff>as_of_utc:raise ValueError("origin_not_reached")
            if as_of_utc>=opening.to_pydatetime() and not _existing_key(output/"05_forecast_distribution_index.csv",key):raise ValueError("late_generation_after_release")
            point_column="recency_weighted_log_consensus_lambda_1_usd" if origin==-1 else "dollar_median_consensus_usd";point=float(row[point_column])
            if not np.isfinite(point) or point<=0:raise ValueError("missing_production_point_forecast")
            distribution=base.distribution(point,origin);raw=np.array([float(distribution.cdf(np.log(threshold/point))) for threshold in THRESHOLDS]);probabilities=beta.transform(raw);seed=_seed(movie_id,origin,manifest["artifact_checksum"],configured_seed);draws=point*np.exp(distribution.ppf(beta.inverse(np.random.default_rng(seed).random(50_000))));draw_checksum=hashlib.sha256(draws.astype("<f8").tobytes()).hexdigest();created=datetime.now(timezone.utc);record={"record_key":key,"record_version":created.isoformat(),"movie_id":movie_id,"title":row.title,"release_date":str(pd.Timestamp(row.opening_weekend_start).date()),"origin":origin,"information_cutoff_utc":cutoff.isoformat(),"forecast_created_utc":created.isoformat(),"forecast_available_utc":created.isoformat(),"point_forecast":point,"point_policy":"production_pre_release_point","interval_policy":"007","base_distribution_policy":"pre_release_pointscale_splitt_001","probability_policy":EXPECTED_ARTIFACT,"artifact_checksum":manifest["artifact_checksum"],"base_parameters":asdict(distribution),"beta_parameters":manifest["calibration_parameters"],"thresholds":THRESHOLDS.tolist(),"threshold_probabilities":probabilities.tolist(),"seed":seed,"draw_checksum":draw_checksum,"input_data_checksum":_row_checksum(row),"code_commit":manifest["source_commit"],"fallback_status":"none","generation_status":"generated"}
            record_path=records/f"movie_{movie_id}_origin_{origin}_{created.strftime('%Y%m%dT%H%M%S%fZ')}.json";record_path.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n");index.append({"record_key":key,"record_version":record["record_version"],"movie_id":movie_id,"origin":origin,"record_path":str(record_path),"draw_checksum":draw_checksum,"generation_status":"generated"});logs.append({"movie_id":movie_id,"origin":origin,"status":"generated","reason":""})
        except Exception as exc:logs.append({"movie_id":movie_id,"origin":origin,"status":"not_generated","reason":str(exc)});failures.append({"movie_id":movie_id,"origin":origin,"failure":str(exc),"recorded_at_utc":datetime.now(timezone.utc).isoformat()})
    _append_csv(output/"04_forecast_generation_log.csv",logs,("movie_id","origin","status","reason"));_append_csv(output/"05_forecast_distribution_index.csv",index,("record_key","record_version"));_write_primary_panels(output);_append_csv(output/"09_data_quality_failures.csv",failures,("movie_id","origin","failure","recorded_at_utc"));summary=_progress(decisions,output);_write_csv(output/"10_cohort_progress.csv",[summary]);interim={**summary,"report_status":"interim_not_for_model_selection","formal_look_allowed":summary["primary_validation_movies"]>=30,"trading_status":"disabled"};(output/"11_interim_status.json").write_text(json.dumps(interim,indent=2)+"\n");(output/"summary.md").write_text("# Prospective shadow operation\n\n"+"\n".join(f"- {key}: `{value}`" for key,value in interim.items())+"\n");return interim


def _write_primary_panels(output:Path)->None:
    path=output/"05_forecast_distribution_index.csv"
    if not path.exists() or not path.read_text().strip():
        (output/"06_primary_latest_origin_panel.csv").write_text("");(output/"07_all_origin_panel.csv").write_text("");return
    frame=pd.read_csv(path);frame.to_csv(output/"07_all_origin_panel.csv",index=False);latest=frame.sort_values(["movie_id","origin"]).groupby("movie_id",as_index=False).tail(1);latest.to_csv(output/"06_primary_latest_origin_panel.csv",index=False)
def _progress(decisions:list[CohortDecision],output:Path)->dict[str,Any]:
    primary=output/"06_primary_latest_origin_panel.csv";count=pd.read_csv(primary).movie_id.nunique() if primary.exists() and primary.stat().st_size else 0
    return {"post_freeze_movies_discovered":len(decisions),"eligible_movies":sum(row.eligibility_status=="eligible" for row in decisions),"excluded_movies":sum(row.eligibility_status!="eligible" for row in decisions),"movies_with_forecasts":count,"movies_awaiting_actuals":count,"primary_validation_movies":0,"progress_to_30":0,"progress_to_50":0}
def _append_csv(path:Path,rows:list[dict[str,Any]],keys:tuple[str,...])->None:
    if not rows:return
    existing=[]
    if path.exists() and path.read_text().strip():existing=list(csv.DictReader(path.open()))
    seen={tuple(str(row.get(key,"")) for key in keys) for row in existing};combined=existing+[row for row in rows if tuple(str(row.get(key,"")) for key in keys) not in seen];_write_csv(path,combined)
def _write_csv(path:Path,rows:list[dict[str,Any]])->None:
    if not rows:path.write_text("");return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w",newline="",encoding="utf-8") as handle:writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
def _existing_key(path:Path,key:str)->bool:
    return path.exists() and path.read_text().strip() and key in {row["record_key"] for row in csv.DictReader(path.open())}
def _seed(movie:int,origin:int,checksum:str,configured:int)->int:return int.from_bytes(hashlib.sha256(f"{movie}|{origin}|{checksum}|{configured}".encode()).digest()[:8],"big")
def _row_checksum(row:pd.Series)->str:return hashlib.sha256(json.dumps(row.to_dict(),sort_keys=True,default=str).encode()).hexdigest()
