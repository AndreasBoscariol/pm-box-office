"""Frozen point-scale calibration historical-development study."""

from __future__ import annotations

import csv,json,hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .asymdist import fit_conditional_split_t
from .cdf_validation import CDFCandidate,EVALUATION_GRID
from .probcal import (BetaCalibration,CalibratedDistribution,OriginGroupPowerCalibration,PowerCalibration,
    fit_beta_calibration,fit_isotonic_calibration,fit_origin_group_power,fit_power_calibration)
from .asymdist_study import THRESHOLDS


MINIMUM_PRIOR_MOVIES=50
MINIMUM_GROUP_MOVIES=25


@dataclass(slots=True)
class BaseRecord:
    movie_id:int;origin:int;date:pd.Timestamp;year:int;point:float;actual:float;base:CDFCandidate;pit:float


def run_phase7_development(panel_path:str|Path,output_dir:str|Path,*,bootstrap_iterations:int=5000,seed:int=42)->dict[str,Any]:
    panel=pd.read_csv(panel_path);panel["opening_weekend_start"]=pd.to_datetime(panel.opening_weekend_start);panel=panel.sort_values(["opening_weekend_start","movie_id","origin_day"]);output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    manifest={"base_distribution_policy":"pre_release_pointscale_splitt_001","probability_policy":"pre_release_pointscale_cal_001","historical_status_only":"provisional_at_most","minimum_prior_movies":50,"minimum_prior_movies_per_group":25,"power_bounds":[.5,2],"empirical_lambda_grid":[0,.1,.2,.3,.4,.5],"market_data_used":False,"primary_prospective_origin":"latest_valid_pre_release_origin"};(output/"01_candidate_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    history:list[BaseRecord]=[];results=[];folds=[];inner=[];params=[]
    for date in sorted(panel.opening_weekend_start.unique()):
        train=_valid(panel.loc[panel.opening_weekend_start<date]);test=panel.loc[panel.opening_weekend_start.eq(date)];prior=train.movie_id.nunique()
        if prior<MINIMUM_PRIOR_MOVIES:
            for row in test.itertuples():folds.append({"movie_id":row.movie_id,"origin":row.origin_day,"date":date,"status":"base_insufficient_history","prior_movies":prior})
            continue
        penalty=_select_base_penalty(train);base_model=_fit_base(train,penalty)
        current=[]
        for _,row in test.iterrows():
            point=float(row.primary_point_forecast_usd);actual=float(row.actual_opening_weekend_gross_usd);distribution=base_model.distribution(point,int(row.origin_day));residual_q=distribution.ppf(EVALUATION_GRID);base=CDFCandidate("pre_release_pointscale_splitt_001",EVALUATION_GRID,residual_q,False);current.append(BaseRecord(int(row.movie_id),int(row.origin_day),date,int(row.release_year),point,actual,base,base.cdf(actual,point)))
        calibration_movies=len({record.movie_id for record in history})
        if calibration_movies<MINIMUM_PRIOR_MOVIES:
            for record in current:folds.append({"movie_id":record.movie_id,"origin":record.origin,"date":date,"status":"calibration_insufficient_history","prior_movies":calibration_movies})
            history.extend(current);continue
        selected=_select_calibration_parameters(history);power=fit_power_calibration([r.pit for r in history],[r.movie_id for r in history],selected["power_penalty"]);beta=fit_beta_calibration([r.pit for r in history],[r.movie_id for r in history],selected["beta_penalty"]);empirical=fit_isotonic_calibration([r.pit for r in history],[r.movie_id for r in history],selected["empirical_lambda"]);groups=[_group(r.origin) for r in history];counts={g:len({r.movie_id for r in history if _group(r.origin)==g}) for g in range(3)};origin_ok=all(value>=MINIMUM_GROUP_MOVIES for value in counts.values());origin_exp=fit_origin_group_power([r.pit for r in history],[r.movie_id for r in history],groups,selected["origin_penalty"]) if origin_ok else (power.exponent,)*3
        params.append({"test_date":date,"base_penalty":penalty,"power_exponent":power.exponent,"power_penalty":selected["power_penalty"],"beta_alpha":beta.alpha,"beta_beta":beta.beta,"beta_penalty":selected["beta_penalty"],"empirical_lambda":selected["empirical_lambda"],"origin_exponents":origin_exp,"origin_fallback_global":not origin_ok,"calibration_movies":calibration_movies})
        inner.extend({"test_date":date,"parameter":key,"selected":value} for key,value in selected.items())
        for record in current:
            maps=[("uncalibrated",None),("power",power),("beta",beta),("empirical",empirical),("origin_group_power",OriginGroupPowerCalibration(origin_exp,_group(record.origin)))]
            for name,mapping in maps:results.append(_evaluate(name,CalibratedDistribution(name,record.base,mapping),record,calibration_movies))
            folds.append({"movie_id":record.movie_id,"origin":record.origin,"date":date,"status":"outer_test","prior_movies":calibration_movies})
        history.extend(current)
    diagnostic=_diagnostic(history);_csv(output/"02_pointscale_pit_diagnostic.csv",diagnostic);_csv(output/"03_outer_fold_membership.csv",folds);_csv(output/"04_inner_selection.csv",inner);_csv(output/"05_calibration_parameters.csv",params);pit=_pit(results);scores=_scores(results);_csv(output/"06_pit_summary.csv",pit);_csv(output/"07_crps_summary.csv",scores);_csv(output/"08_threshold_log_summary.csv",scores);_csv(output/"09_rps_summary.csv",scores);_csv(output/"10_tail_balance.csv",pit);_csv(output/"11_score_by_origin.csv",_group_scores(results));bootstrap=_bootstrap(results,bootstrap_iterations,seed);_csv(output/"12_movie_cluster_bootstrap.csv",bootstrap);_csv(output/"13_leave_one_movie_out.csv",_leave_one(results));outcome,selection=_select_candidate(scores,pit,bootstrap);_csv(output/"14_candidate_selection.csv",selection)
    selected_name={"provisional_power":"power","provisional_beta":"beta","provisional_empirical":"empirical","provisional_origin_group":"origin_group_power"}.get(outcome)
    outer_movies=len({r["movie_id"] for r in results});summary={"historical_outcome":outcome,"selected_candidate":selected_name,"outer_test_movies":outer_movies,"evaluation_rows":len(results),"market_data_used":False,"production_probability_eligible":False};(output/"summary.md").write_text("# Point-scale calibration historical development\n\n"+"\n".join(f"- {k}: `{v}`" for k,v in summary.items())+"\n");return summary


def _valid(frame:pd.DataFrame)->pd.DataFrame:return frame.loc[frame.primary_point_forecast_usd.gt(0)&frame.actual_opening_weekend_gross_usd.gt(0)&frame.release_width_bucket.astype(str).isin(["wide","large_wide"])].copy()
def _fit_base(frame:pd.DataFrame,penalty:float):return fit_conditional_split_t(np.log(frame.actual_opening_weekend_gross_usd/frame.primary_point_forecast_usd),frame.primary_point_forecast_usd,frame.origin_day,frame.movie_id.astype(int),"point_scale_split_student_t",penalty)
def _select_base_penalty(frame:pd.DataFrame)->float:
    dates=sorted(frame.opening_weekend_start.unique());cut=dates[max(1,int(len(dates)*.8))-1];train=frame.loc[frame.opening_weekend_start<cut];validation=frame.loc[frame.opening_weekend_start>=cut];scores={}
    for penalty in (1.,10.):
        model=_fit_base(train,penalty);values=[]
        for row in validation.itertuples():values.append(-np.log(max(float(model.distribution(row.primary_point_forecast_usd,row.origin_day).pdf(np.log(row.actual_opening_weekend_gross_usd/row.primary_point_forecast_usd))),1e-15)))
        scores[penalty]=np.mean(values)
    return min(scores,key=scores.get)
def _select_calibration_parameters(history:list[BaseRecord])->dict[str,float]:
    dates=sorted({r.date for r in history});cut=dates[max(1,int(len(dates)*.8))-1];train=[r for r in history if r.date<cut];validation=[r for r in history if r.date>=cut];result={}
    grids={"power_penalty":[1.,5.,20.],"beta_penalty":[1.,5.,20.],"empirical_lambda":[0.,.1,.2,.3,.4,.5],"origin_penalty":[1.,5.,20.]}
    for key,grid in grids.items():
        scores={}
        for value in grid:
            if key=="power_penalty":factory=lambda r,m=fit_power_calibration([x.pit for x in train],[x.movie_id for x in train],value):m
            elif key=="beta_penalty":factory=lambda r,m=fit_beta_calibration([x.pit for x in train],[x.movie_id for x in train],value):m
            elif key=="empirical_lambda":factory=lambda r,m=fit_isotonic_calibration([x.pit for x in train],[x.movie_id for x in train],value):m
            else:
                exponents=fit_origin_group_power([x.pit for x in train],[x.movie_id for x in train],[_group(x.origin) for x in train],value);factory=lambda r,e=exponents:OriginGroupPowerCalibration(e,_group(r.origin))
            scores[value]=np.mean([_crps(CalibratedDistribution("x",r.base,factory(r)),r) for r in validation])
        result[key]=min(scores,key=scores.get)
    return result
def _crps(distribution:CalibratedDistribution,record:BaseRecord)->float:
    p=np.linspace(.001,.999,199);q=distribution.quantile(p,record.point);loss=np.where(record.actual>=q,p*(record.actual-q),(1-p)*(q-record.actual));return float(2*np.trapezoid(loss,p))
def _evaluate(name:str,distribution:CalibratedDistribution,record:BaseRecord,prior:int)->dict[str,Any]:
    probs=np.clip(np.array([distribution.cdf(x,record.point) for x in THRESHOLDS]),1e-6,1-1e-6);obs=(record.actual<=THRESHOLDS).astype(float);bucket=np.searchsorted(THRESHOLDS,record.actual);cum=probs;obs_cum=(np.arange(len(THRESHOLDS))>=bucket).astype(float)
    return {"candidate":name,"movie_id":record.movie_id,"origin":record.origin,"date":record.date,"year":record.year,"point":record.point,"actual":record.actual,"pit":distribution.cdf(record.actual,record.point),"crps":_crps(distribution,record),"threshold_log_score":float(np.mean(-(obs*np.log(probs)+(1-obs)*np.log1p(-probs)))),"rps":float(np.sum((cum-obs_cum)**2)),"prior_movies":prior}
def _diagnostic(history:list[BaseRecord])->list[dict[str,Any]]:
    frame=pd.DataFrame([{"movie_id":r.movie_id,"origin":r.origin,"origin_group":_group(r.origin),"pit":r.pit,"point_bucket":_point_bucket(r.point),"year":r.year} for r in history]);out=[]
    for key in ("origin_group","point_bucket","year"):
        for value,g in frame.groupby(key):out.append({"dimension":key,"value":value,"movies":g.movie_id.nunique(),"pit_mean":g.groupby("movie_id").pit.mean().mean(),"pit_median":g.pit.median(),"pit_variance":g.pit.var(ddof=0),"below_10":(g.pit<.1).mean(),"above_90":(g.pit>.9).mean(),"below_025":(g.pit<.025).mean(),"above_975":(g.pit>.975).mean()})
    return out
def _scores(rows:list[dict[str,Any]])->list[dict[str,Any]]:
    f=pd.DataFrame(rows);return [{"candidate":n,"crps":g.groupby("movie_id").crps.mean().mean(),"threshold_log_score":g.groupby("movie_id").threshold_log_score.mean().mean(),"rps":g.groupby("movie_id").rps.mean().mean(),"movies":g.movie_id.nunique()} for n,g in f.groupby("candidate")]
def _pit(rows:list[dict[str,Any]])->list[dict[str,Any]]:
    f=pd.DataFrame(rows);return [{"candidate":n,"pit_mean":g.groupby("movie_id").pit.mean().mean(),"pit_variance":g.pit.var(ddof=0),"below_10":(g.pit<.1).mean(),"above_90":(g.pit>.9).mean(),"below_025":(g.pit<.025).mean(),"above_975":(g.pit>.975).mean(),"movies":g.movie_id.nunique()} for n,g in f.groupby("candidate")]
def _bootstrap(rows:list[dict[str,Any]],iterations:int,seed:int)->list[dict[str,Any]]:
    f=pd.DataFrame(rows);rng=np.random.default_rng(seed);out=[]
    for n,g in f.groupby("candidate"):
        means=g.groupby("movie_id")[["pit","crps","threshold_log_score","rps"]].mean().to_numpy();idx=rng.integers(0,len(means),(iterations,len(means)));sample=means[idx].mean(1);out.append({"candidate":n,"pit_lower_95":np.quantile(sample[:,0],.025),"pit_upper_95":np.quantile(sample[:,0],.975),"crps_lower_95":np.quantile(sample[:,1],.025),"crps_upper_95":np.quantile(sample[:,1],.975),"iterations":iterations})
    return out
def _group_scores(rows:list[dict[str,Any]])->list[dict[str,Any]]:
    f=pd.DataFrame(rows);return [{"candidate":n,"origin":origin,"crps":g.crps.mean(),"threshold_log_score":g.threshold_log_score.mean(),"rps":g.rps.mean(),"pit_mean":g.pit.mean(),"movies":g.movie_id.nunique()} for (n,origin),g in f.groupby(["candidate","origin"])]
def _leave_one(rows:list[dict[str,Any]])->list[dict[str,Any]]:
    f=pd.DataFrame(rows);return [{"candidate":n,"left_out_movie":movie,"crps":g.loc[g.movie_id.ne(movie)].crps.mean(),"pit_mean":g.loc[g.movie_id.ne(movie)].pit.mean()} for n,g in f.groupby("candidate") for movie in g.movie_id.unique()]
def _select_candidate(scores:list[dict[str,Any]],pits:list[dict[str,Any]],bootstrap:list[dict[str,Any]])->tuple[str,list[dict[str,Any]]]:
    s={r["candidate"]:r for r in scores};p={r["candidate"]:r for r in pits};b={r["candidate"]:r for r in bootstrap};base=s["uncalibrated"];rows=[]
    for name in ("power","beta","empirical","origin_group_power"):
        calibration=b[name]["pit_lower_95"]<=.5<=b[name]["pit_upper_95"] and .48<=p[name]["pit_mean"]<=.52 and abs(p[name]["above_90"]-p[name]["below_10"])<abs(p["uncalibrated"]["above_90"]-p["uncalibrated"]["below_10"])
        noninferior=all(s[name][metric]/base[metric]<=1.02 for metric in ("crps","threshold_log_score","rps"));rows.append({"candidate":name,"calibration_gate":calibration,"noninferiority_gate":noninferior,"eligible":calibration and noninferior,"crps_ratio":s[name]["crps"]/base["crps"],"threshold_log_ratio":s[name]["threshold_log_score"]/base["threshold_log_score"],"rps_ratio":s[name]["rps"]/base["rps"]})
    eligible=[r for r in rows if r["eligible"]];mapping={"power":"provisional_power","beta":"provisional_beta","empirical":"provisional_empirical","origin_group_power":"provisional_origin_group"}
    if not eligible:return "no_candidate",rows
    chosen=min(eligible,key=lambda r:(s[r["candidate"]]["threshold_log_score"],s[r["candidate"]]["rps"],s[r["candidate"]]["crps"]));return mapping[chosen["candidate"]],rows
def _group(origin:int)->int:return 0 if origin<=-8 else (1 if origin<=-3 else 2)
def _point_bucket(point:float)->str:return "lt_10m" if point<1e7 else ("10m_25m" if point<2.5e7 else ("25m_50m" if point<5e7 else ("50m_100m" if point<1e8 else "100m_plus")))
def _csv(path:Path,rows:list[dict[str,Any]])->None:
    if not rows:path.write_text("");return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w",newline="",encoding="utf-8") as handle:writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
