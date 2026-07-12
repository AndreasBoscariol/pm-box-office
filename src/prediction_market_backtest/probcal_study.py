"""Nested chronological, movie-balanced 007 probability calibration study."""

from __future__ import annotations

import csv,json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cdf_validation import CDFCandidate,build_candidates,crps_from_quantiles,quantile_loss
from .policy007 import build_chronological_pool
from .probcal import (CalibratedDistribution,fit_beta_calibration,fit_isotonic_calibration,
                      location_shift,movie_balanced_weights)


MINIMUM_PRIOR_MOVIES=40


@dataclass(slots=True)
class BaseRow:
    movie_id:int;origin:int;date:pd.Timestamp;year:int;point:float;actual:float;base_name:str;base:CDFCandidate;pit:float;log_error:float


def run_probcal_study(panel_path:str|Path,output_dir:str|Path,*,bootstrap_iterations:int=5000,seed:int=42)->dict[str,Any]:
    panel=pd.read_csv(panel_path);panel["opening_weekend_start"]=pd.to_datetime(panel.opening_weekend_start);base_rows=[]
    for _,row in panel.sort_values("opening_weekend_start").iterrows():
        try:pool=build_chronological_pool(panel,row)
        except RuntimeError:continue
        point=float(row.primary_point_forecast_usd);actual=float(row.actual_opening_weekend_gross_usd)
        candidates=build_candidates(pool)
        for base in (candidates[0],candidates[2]):base_rows.append(BaseRow(int(row.movie_id),int(row.origin_day),row.opening_weekend_start,int(row.release_year),point,actual,base.name,base,base.cdf(actual,point),float(np.log(actual/point))))
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    manifest={"probability_policy":"007_probcal_001","point_policy":"production_pre_release_point","interval_policy":"007",
        "minimum_prior_movies":MINIMUM_PRIOR_MOVIES,"market_data_used":False,"bases":["007_cdf_extension_001","007_raw_weighted_empirical_benchmark"],
        "recalibrators":["beta","shrunk_isotonic","location_only"],"beta_regularization_grid":[1,5,20],"isotonic_shrinkage_grid":[.25,.5,.75,1.0]}
    (output/"01_candidate_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    evaluations=[];folds=[];inner_rows=[];parameters=[]
    dates=sorted({row.date for row in base_rows})
    for date in dates:
        current=[row for row in base_rows if row.date==date];prior=[row for row in base_rows if row.date<date]
        prior_movies={row.movie_id for row in prior}
        if len(prior_movies)<MINIMUM_PRIOR_MOVIES:
            for row in current:folds.append({"movie_id":row.movie_id,"date":date,"origin":row.origin,"status":"excluded_insufficient_history","prior_movies":len(prior_movies)})
            continue
        for base_name in sorted({row.base_name for row in current}):
            train=[row for row in prior if row.base_name==base_name];test=[row for row in current if row.base_name==base_name]
            beta_reg,beta_scores=_select_beta(train);iso_lambda,iso_scores=_select_isotonic(train)
            beta=fit_beta_calibration([r.pit for r in train],[r.movie_id for r in train],beta_reg)
            iso=fit_isotonic_calibration([r.pit for r in train],[r.movie_id for r in train],iso_lambda)
            shift=location_shift([r.log_error for r in train],[r.movie_id for r in train])
            parameters.append({"test_date":date,"base":base_name,"prior_movies":len(prior_movies),"beta_regularization":beta_reg,"beta_alpha":beta.alpha,"beta_beta":beta.beta,"isotonic_shrinkage":iso_lambda,"location_shift_log":shift})
            inner_rows.extend([{"test_date":date,"base":base_name,"type":"beta","parameter":key,"validation_crps":value} for key,value in beta_scores.items()]);inner_rows.extend([{"test_date":date,"base":base_name,"type":"isotonic","parameter":key,"validation_crps":value} for key,value in iso_scores.items()])
            for row in test:
                distributions=[CalibratedDistribution(f"beta__{base_name}",row.base,beta),CalibratedDistribution(f"isotonic__{base_name}",row.base,iso),CalibratedDistribution(f"location__{base_name}",row.base,location_shift_log=shift),CalibratedDistribution(f"uncalibrated__{base_name}",row.base)]
                folds.append({"movie_id":row.movie_id,"date":date,"origin":row.origin,"status":"outer_test","prior_movies":len(prior_movies)})
                for distribution in distributions:evaluations.append(_evaluate(distribution,row))
    _csv(output/"02_outer_fold_membership.csv",folds);_csv(output/"03_inner_fold_selection.csv",inner_rows);_csv(output/"04_calibration_parameters.csv",parameters);_csv(output/"05_probability_distribution_index.csv",[{k:r[k] for k in ("candidate","movie_id","origin","date","base","prior_movies")} for r in evaluations])
    summary=_score_summary(evaluations);_csv(output/"06_crps_summary.csv",summary);_csv(output/"07_log_score_summary.csv",summary);_csv(output/"08_pit_values.csv",[{k:r[k] for k in ("candidate","movie_id","origin","pit")} for r in evaluations]);pit_summary=_pit_summary(evaluations);_csv(output/"09_pit_summary.csv",pit_summary);_csv(output/"10_coverage_curve.csv",_coverage(evaluations));_csv(output/"11_tail_balance.csv",pit_summary);_csv(output/"12_threshold_calibration.csv",_thresholds(evaluations));_csv(output/"13_score_by_origin.csv",_score_group(evaluations,"origin"));_csv(output/"14_score_by_year.csv",_score_group(evaluations,"year"));bootstrap=_bootstrap(evaluations,bootstrap_iterations,seed);_csv(output/"15_movie_cluster_bootstrap.csv",bootstrap);_csv(output/"16_leave_one_movie_out.csv",_leave_one_movie(evaluations))
    selected,selection=_select(summary,pit_summary,bootstrap);_csv(output/"17_candidate_selection.csv",selection)
    outer_movies=len({r["movie_id"] for r in evaluations});status="approved" if selected else ("inconclusive_small_sample" if outer_movies<30 else "rejected")
    result={"probability_policy":"007_probcal_001","outer_test_movies":outer_movies,"outer_rows":len(evaluations),"selected_candidate":selected,"probability_model_status":status,"market_data_used":False}
    (output/"summary.md").write_text("# 007 probability recalibration validation\n\n"+"\n".join(f"- {k}: `{v}`" for k,v in result.items())+"\n")
    return result


def _validation_split(rows:list[BaseRow])->tuple[list[BaseRow],list[BaseRow]]:
    dates=sorted({r.date for r in rows});cut=dates[max(1,int(len(dates)*.8))-1];return [r for r in rows if r.date<cut],[r for r in rows if r.date>=cut]
def _select_beta(rows:list[BaseRow])->tuple[float,dict[float,float]]:
    train,val=_validation_split(rows);scores={}
    for reg in (1.,5.,20.):
        model=fit_beta_calibration([r.pit for r in train],[r.movie_id for r in train],reg);scores[reg]=_mean_crps([CalibratedDistribution("x",r.base,model) for r in val],val)
    return min(scores,key=scores.get),scores
def _select_isotonic(rows:list[BaseRow])->tuple[float,dict[float,float]]:
    train,val=_validation_split(rows);scores={}
    for value in (.25,.5,.75,1.):
        model=fit_isotonic_calibration([r.pit for r in train],[r.movie_id for r in train],value);scores[value]=_mean_crps([CalibratedDistribution("x",r.base,model) for r in val],val)
    return min(scores,key=scores.get),scores
def _mean_crps(distributions:list[CalibratedDistribution],rows:list[BaseRow])->float:return float(np.mean([_crps(d,r.actual,r.point) for d,r in zip(distributions,rows)]))
def _crps(d:CalibratedDistribution,outcome:float,point:float)->float:
    p=np.linspace(.001,.999,199);q=d.quantile(p,point);loss=np.where(outcome>=q,p*(outcome-q),(1-p)*(q-outcome));return float(2*np.trapezoid(loss,p))
def _evaluate(d:CalibratedDistribution,row:BaseRow)->dict[str,Any]:
    pit=d.cdf(row.actual,row.point);dx=max(1000,row.actual*1e-4);density=max((d.cdf(row.actual+dx,row.point)-d.cdf(max(1,row.actual-dx),row.point))/(2*dx),1e-15)
    return {"candidate":d.name,"base":row.base_name,"movie_id":row.movie_id,"origin":row.origin,"date":row.date,"year":row.year,"point":row.point,"actual":row.actual,"pit":pit,"crps":_crps(d,row.actual,row.point),"log_score":-np.log(density),"prior_movies":None,"q10":float(d.quantile(.1,row.point)),"q90":float(d.quantile(.9,row.point)),"q025":float(d.quantile(.025,row.point)),"q975":float(d.quantile(.975,row.point))}
def _score_summary(rows:list[dict[str,Any]])->list[dict[str,Any]]:
    f=pd.DataFrame(rows);return [{"candidate":name,"movie_averaged_crps":g.groupby("movie_id").crps.mean().mean(),"movie_averaged_log_score":g.groupby("movie_id").log_score.mean().mean(),"movies":g.movie_id.nunique(),"rows":len(g)} for name,g in f.groupby("candidate")]
def _pit_summary(rows:list[dict[str,Any]])->list[dict[str,Any]]:
    f=pd.DataFrame(rows);return [{"candidate":name,"pit_mean":g.groupby("movie_id").pit.mean().mean(),"pit_variance":g.pit.var(ddof=0),"below_10":(g.pit<.1).mean(),"above_90":(g.pit>.9).mean(),"below_025":(g.pit<.025).mean(),"above_975":(g.pit>.975).mean(),"movies":g.movie_id.nunique()} for name,g in f.groupby("candidate")]
def _coverage(rows:list[dict[str,Any]])->list[dict[str,Any]]:
    f=pd.DataFrame(rows);out=[]
    for name,g in f.groupby("candidate"):
        for nominal,lo,hi in ((.8,"q10","q90"),(.95,"q025","q975")):out.append({"candidate":name,"nominal":nominal,"coverage":((g.actual>=g[lo])&(g.actual<=g[hi])).mean(),"lower_miss":(g.actual<g[lo]).mean(),"upper_miss":(g.actual>g[hi]).mean()})
    return out
def _thresholds(rows:list[dict[str,Any]])->list[dict[str,Any]]:return []
def _score_group(rows:list[dict[str,Any]],key:str)->list[dict[str,Any]]:
    f=pd.DataFrame(rows);return [{"candidate":name,key:value,"mean_crps":g.crps.mean(),"mean_log_score":g.log_score.mean(),"movies":g.movie_id.nunique()} for (name,value),g in f.groupby(["candidate",key])]
def _bootstrap(rows:list[dict[str,Any]],iterations:int,seed:int)->list[dict[str,Any]]:
    f=pd.DataFrame(rows);rng=np.random.default_rng(seed);out=[]
    for name,g in f.groupby("candidate"):
        movie_means=g.groupby("movie_id")[["pit","crps"]].mean().to_numpy();indices=rng.integers(0,len(movie_means),size=(iterations,len(movie_means)))
        sampled=movie_means[indices].mean(axis=1);pit=sampled[:,0];crps=sampled[:,1]
        out.append({"candidate":name,"pit_mean_median":np.median(pit),"pit_mean_lower_95":np.quantile(pit,.025),"pit_mean_upper_95":np.quantile(pit,.975),"crps_median":np.median(crps),"crps_lower_95":np.quantile(crps,.025),"crps_upper_95":np.quantile(crps,.975),"iterations":iterations})
    return out
def _leave_one_movie(rows:list[dict[str,Any]])->list[dict[str,Any]]:
    f=pd.DataFrame(rows);out=[]
    for name,g in f.groupby("candidate"):
        for movie in g.movie_id.unique():out.append({"candidate":name,"left_out_movie":movie,"mean_crps":g.loc[g.movie_id.ne(movie)].crps.mean(),"pit_mean":g.loc[g.movie_id.ne(movie)].pit.mean()})
    return out
def _select(scores:list[dict[str,Any]],pits:list[dict[str,Any]],bootstrap:list[dict[str,Any]])->tuple[str|None,list[dict[str,Any]]]:
    score={r["candidate"]:r for r in scores};pit={r["candidate"]:r for r in pits};boot={r["candidate"]:r for r in bootstrap};rows=[]
    for name in sorted(score):
        if name.startswith("uncalibrated"):continue
        base="uncalibrated__"+name.split("__",1)[1];p=pit[name];b=boot[name];eligible=b["pit_mean_lower_95"]<=.5<=b["pit_mean_upper_95"] and abs(p["above_90"]-p["below_10"])<=.05 and score[name]["movie_averaged_crps"]<score[base]["movie_averaged_crps"] and score[name]["movie_averaged_log_score"]<=score[base]["movie_averaged_log_score"]+.1
        rows.append({"candidate":name,"eligible":eligible,"crps":score[name]["movie_averaged_crps"],"base_crps":score[base]["movie_averaged_crps"],"pit_ci_includes_half":b["pit_mean_lower_95"]<=.5<=b["pit_mean_upper_95"],"tail_difference":p["above_90"]-p["below_10"]})
    eligible=[r for r in rows if r["eligible"]];return (min(eligible,key=lambda r:r["crps"])["candidate"] if eligible else None),rows
def _csv(path:Path,rows:list[dict[str,Any]])->None:
    if not rows:path.write_text("");return
    with path.open("w",newline="",encoding="utf-8") as handle:writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
