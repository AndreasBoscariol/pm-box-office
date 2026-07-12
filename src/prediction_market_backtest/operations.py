"""Persistent idempotent Phase 9 shadow supervisor and actual synchronization."""

from __future__ import annotations
import asyncio,hashlib,json,time
from datetime import datetime,timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from .phase8_status import build_phase8_status,initialize_phase8_external_outputs
from .prospective import generate_shadow_forecasts
from .prospective_actuals import ActualRevision,append_actual


def sync_prospective_actuals(panel_path:str|Path,output_dir:str|Path,*,as_of_utc:datetime|None=None)->dict[str,int]:
    as_of_utc=as_of_utc or datetime.now(timezone.utc);panel=pd.read_csv(panel_path);output=Path(output_dir);output.mkdir(parents=True,exist_ok=True);actual_path=output/"08_actual_sync_log.csv";inserted=skipped=failures=0
    if "actual_available_at" not in panel.columns:return {"inserted":0,"skipped":len(panel),"failures":0,"reason":"actual_available_at_missing_fail_closed"}  # type: ignore[return-value]
    cutoff=pd.Timestamp("2026-07-03");panel["opening_weekend_start"]=pd.to_datetime(panel.opening_weekend_start);panel["actual_available_at"]=pd.to_datetime(panel.actual_available_at,utc=True,errors="coerce")
    for _,row in panel.loc[panel.opening_weekend_start.gt(cutoff)].iterrows():
        try:
            if pd.isna(row.actual_available_at) or row.actual_available_at.to_pydatetime()>as_of_utc:skipped+=1;continue
            gross=Decimal(str(row.actual_opening_weekend_gross_usd));before=actual_path.stat().st_size if actual_path.exists() else 0
            append_actual(actual_path,ActualRevision(int(row.movie_id),gross,str(row.get("actual_source") or "approved_box_office_source"),row.actual_available_at.isoformat(),as_of_utc.isoformat(),3,"USD","domestic_us_canada",final_approved=bool(row.get("actual_final_approved",False))))
            inserted+=int(actual_path.stat().st_size>before);skipped+=int(actual_path.stat().st_size==before)
        except Exception:failures+=1
    return {"inserted":inserted,"skipped":skipped,"failures":failures}


async def run_shadow_supervisor(*,panel_path:str="models/boxoffice/boxoffice_local_007_weighted_interval_calibration/pre_release_panel.csv",artifact_dir:str="models/boxoffice/pre_release_pointscale_cal_001",output_dir:str="data/diagnostics/pre_release_pointscale_cal_prospective_v2",forecast_interval_seconds:int=3600,actual_interval_seconds:int=21600,status_interval_seconds:int=300)->None:
    config={"panel":panel_path,"artifact":artifact_dir,"output":output_dir,"forecast_interval":forecast_interval_seconds,"actual_interval":actual_interval_seconds,"status_interval":status_interval_seconds};config_checksum=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();log=Path(output_dir)/"service_jobs.jsonl";next_forecast=next_actual=next_status=0.
    while True:
        now=time.monotonic()
        if now>=next_forecast:
            await asyncio.to_thread(_record_job,log,"generate-prospective-forecasts",config_checksum,lambda:generate_shadow_forecasts(panel_path,artifact_dir,output_dir));next_forecast=now+forecast_interval_seconds
        if now>=next_actual:
            await asyncio.to_thread(_record_job,log,"sync-prospective-actuals",config_checksum,lambda:sync_prospective_actuals(panel_path,output_dir));next_actual=now+actual_interval_seconds
        if now>=next_status:
            await asyncio.to_thread(_record_job,log,"phase9-status",config_checksum,lambda:(initialize_phase8_external_outputs(),build_phase8_status())[1]);next_status=now+status_interval_seconds
        await asyncio.sleep(min(60,max(1,min(next_forecast,next_actual,next_status)-time.monotonic())))


def _record_job(path:Path,name:str,config_checksum:str,operation:Any)->None:
    started=datetime.now(timezone.utc);record={"job":name,"started_at":started.isoformat(),"config_checksum":config_checksum}
    try:result=operation();record.update({"status":"succeeded","exit_code":0,"result":result})
    except Exception as exc:record.update({"status":"failed","exit_code":1,"error":str(exc)})
    record["completed_at"]=datetime.now(timezone.utc).isoformat();record["runtime_seconds"]=(datetime.fromisoformat(record["completed_at"])-started).total_seconds();path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as handle:handle.write(json.dumps(record,sort_keys=True,default=str)+"\n")
