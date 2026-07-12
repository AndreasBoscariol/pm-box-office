"""Market-independent nested evaluation of native asymmetric residual models."""

from __future__ import annotations

import csv,json,hashlib
from pathlib import Path
from typing import Any,Callable

import numpy as np
import pandas as pd

from .asymdist import (AsymmetricEmpirical,ConditionalSplitStudentT,fit_conditional_split_t,gross_cdf,gross_quantile)
from .cdf_validation import build_candidates
from .policy007 import build_chronological_pool,weighted_quantile
from .probcal import movie_balanced_weights


THRESHOLDS=np.array([10,15,20,25,30,35,40,50,60,75,100,125,150],float)*1_000_000
MINIMUM_PRIOR_MOVIES=50
PARAMETRIC=("global_split_student_t","point_scale_split_student_t","origin_point_scale_split_student_t")


def run_asymdist_study(panel_path:str|Path,output_dir:str|Path,*,bootstrap_iterations:int=5000,seed:int=42)->dict[str,Any]:
    panel=pd.read_csv(panel_path);panel["opening_weekend_start"]=pd.to_datetime(panel.opening_weekend_start);panel=panel.sort_values(["opening_weekend_start","movie_id","origin_day"]);output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    manifest={"point_policy":"production_pre_release_point","interval_policy":"007","probability_policy":"pre_release_asymdist_001","minimum_prior_movies":50,"thresholds_usd":THRESHOLDS.tolist(),"market_data_used":False,"candidates":["007_cdf_extension_001",*PARAMETRIC,"asymmetric_empirical","asymmetric_empirical_tail"],"parameter_bounds":{"nu":[2.5,50],"scale_ratio":[.5,5],"empirical_scales":[.5,3],"tail_stretch":[1,3]}}
    (output/"01_candidate_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    results=[];folds=[];inner=[];parameters=[];dates=sorted(panel.opening_weekend_start.unique())
    for date in dates:
        train=panel.loc[panel.opening_weekend_start<date];test=panel.loc[panel.opening_weekend_start.eq(date)];prior_movies=train.movie_id.nunique()
        if prior_movies<MINIMUM_PRIOR_MOVIES:
            for row in test.itertuples():folds.append({"movie_id":row.movie_id,"origin":row.origin_day,"date":date,"status":"insufficient_history","prior_movies":prior_movies})
            continue
        valid=_valid_training(train);cut_dates=sorted(valid.opening_weekend_start.unique());cut=cut_dates[max(1,int(len(cut_dates)*.8))-1];inner_train=valid.loc[valid.opening_weekend_start<cut];inner_val=valid.loc[valid.opening_weekend_start>=cut]
        fitted={}
        for candidate in PARAMETRIC:
            scores={}
            for penalty in (1.,10.):
                model=_fit(inner_train,candidate,penalty);scores[penalty]=_nll(model,inner_val)
            chosen=min(scores,key=scores.get);inner.extend({"test_date":date,"candidate":candidate,"penalty":key,"validation_log_score":value} for key,value in scores.items());fitted[candidate]=_fit(valid,candidate,chosen);parameters.append(_parameter_row(date,candidate,chosen,fitted[candidate],prior_movies))
        residuals=np.log(valid.actual_opening_weekend_gross_usd/valid.primary_point_forecast_usd);weights=movie_balanced_weights(valid.movie_id.astype(int));delta=float(weighted_quantile(residuals,weights,.5));global_q=weighted_quantile(residuals,weights,[.1,.5,.9]);parameters.extend([{"test_date":date,"candidate":"asymmetric_empirical","delta":delta,"lower_scale":1,"upper_scale":1,"tail_stretch":1,"prior_movies":prior_movies},{"test_date":date,"candidate":"asymmetric_empirical_tail","delta":delta,"lower_scale":1,"upper_scale":1,"tail_stretch":1.5,"prior_movies":prior_movies}])
        for _,row in test.iterrows():
            folds.append({"movie_id":int(row.movie_id),"origin":int(row.origin_day),"date":date,"status":"outer_test","prior_movies":prior_movies})
            point=float(row.primary_point_forecast_usd);actual=float(row.actual_opening_weekend_gross_usd);origin=int(row.origin_day)
            try:pool=build_chronological_pool(panel,row)
            except RuntimeError:continue
            baseline=build_candidates(pool)[0];results.append(_evaluate("007_cdf_extension_001",lambda x:baseline.cdf(x,point),lambda p:baseline.quantile(p,point),row,prior_movies))
            for name,model in fitted.items():
                distribution=model.distribution(point,origin);results.append(_evaluate(name,lambda x,d=distribution:gross_cdf(d,x,point),lambda p,d=distribution:gross_quantile(d,p,point),row,prior_movies))
            pool_q=weighted_quantile(pool.residuals,pool.weights,[.1,.5,.9]);lower=float(np.clip((global_q[1]-global_q[0])/max(pool_q[1]-pool_q[0],1e-6),.5,3));upper=float(np.clip((global_q[2]-global_q[1])/max(pool_q[2]-pool_q[1],1e-6),.5,3))
            for name,tail in (("asymmetric_empirical",1.),("asymmetric_empirical_tail",1.5)):
                distribution=AsymmetricEmpirical.from_pool(pool,delta=delta,lower_scale=lower,upper_scale=upper,upper_tail_stretch=tail);results.append(_evaluate(name,lambda x,d=distribution:gross_cdf(d,x,point),lambda p,d=distribution:gross_quantile(d,p,point),row,prior_movies))
    _csv(output/"02_outer_fold_membership.csv",folds);_csv(output/"03_inner_fold_selection.csv",inner);_csv(output/"04_parameter_estimates.csv",parameters);_csv(output/"05_distribution_index.csv",[{k:r[k] for k in ("candidate","movie_id","origin","date","prior_movies")} for r in results]);_csv(output/"06_pit_values.csv",[{k:r[k] for k in ("candidate","movie_id","origin","pit")} for r in results]);pit=_pit(results);scores=_scores(results);_csv(output/"07_pit_summary.csv",pit);_csv(output/"08_crps_summary.csv",scores);_csv(output/"09_threshold_brier.csv",_metric(results,"threshold_brier"));_csv(output/"10_threshold_log_score.csv",_metric(results,"threshold_log_score"));_csv(output/"11_canonical_rps.csv",_metric(results,"canonical_rps"));_csv(output/"12_tail_balance.csv",pit);_csv(output/"13_score_by_origin.csv",_group(results,"origin"));_csv(output/"14_score_by_point_bucket.csv",_group(results,"point_bucket"));_csv(output/"15_interval_comparison_007.csv",_intervals(results));bootstrap=_bootstrap(results,bootstrap_iterations,seed);_csv(output/"16_movie_cluster_bootstrap.csv",bootstrap);_csv(output/"17_leave_one_movie_out.csv",_leave(results,"movie_id"));_csv(output/"18_leave_one_year_out.csv",_leave(results,"year"));selected,selection=_select(scores,pit,bootstrap);_csv(output/"19_candidate_selection.csv",selection)
    outer_movies=len({r["movie_id"] for r in results});status="approved" if selected else ("inconclusive_small_sample" if outer_movies<50 else "rejected");summary={"probability_policy":"pre_release_asymdist_001","outer_test_movies":outer_movies,"evaluated_rows":len(results),"selected_candidate":selected,"probability_model_status":status,"market_data_used":False}
    (output/"summary.md").write_text("# Native asymmetric probability validation\n\n"+"\n".join(f"- {k}: `{v}`" for k,v in summary.items())+"\n");return summary


def _valid_training(frame:pd.DataFrame)->pd.DataFrame:
    point=pd.to_numeric(frame.primary_point_forecast_usd,errors="coerce");actual=pd.to_numeric(frame.actual_opening_weekend_gross_usd,errors="coerce");return frame.loc[point.gt(0)&actual.gt(0)&frame.release_width_bucket.astype(str).isin(["wide","large_wide"])].copy()
def _fit(frame:pd.DataFrame,candidate:str,penalty:float)->ConditionalSplitStudentT:
    residual=np.log(frame.actual_opening_weekend_gross_usd/frame.primary_point_forecast_usd);return fit_conditional_split_t(residual,frame.primary_point_forecast_usd,frame.origin_day,frame.movie_id.astype(int),candidate,penalty)
def _nll(model:ConditionalSplitStudentT,frame:pd.DataFrame)->float:
    values=[]
    for row in frame.itertuples():values.append(-np.log(max(float(model.distribution(float(row.primary_point_forecast_usd),int(row.origin_day)).pdf(np.log(row.actual_opening_weekend_gross_usd/row.primary_point_forecast_usd))),1e-15)))
    weights=movie_balanced_weights(frame.movie_id.astype(int));return float(np.sum(weights*values))
def _parameter_row(date:object,name:str,penalty:float,model:ConditionalSplitStudentT,movies:int)->dict[str,Any]:return {"test_date":date,"candidate":name,"penalty":penalty,"location":model.location,"sigma_lower":np.exp(model.log_sigma_lower),"sigma_upper":np.exp(model.log_sigma_upper),"scale_ratio":np.exp(model.log_sigma_upper-model.log_sigma_lower),"degrees_of_freedom":model.degrees_of_freedom,"lower_point_slope":model.lower_point_slope,"upper_point_slope":model.upper_point_slope,"location_group_offsets":model.location_group_offsets,"lower_group_offsets":model.lower_group_offsets,"upper_group_offsets":model.upper_group_offsets,"prior_movies":movies,"parameter_checksum":hashlib.sha256(repr(model).encode()).hexdigest()}
def _evaluate(name:str,cdf:Callable[[float],float],quantile:Callable[[np.ndarray],np.ndarray],row:pd.Series,prior:int)->dict[str,Any]:
    actual=float(row.actual_opening_weekend_gross_usd);p=np.linspace(.001,.999,199);q=quantile(p);loss=np.where(actual>=q,p*(actual-q),(1-p)*(q-actual));probs=np.clip(np.array([cdf(x) for x in THRESHOLDS]),1e-6,1-1e-6);observed=(actual<=THRESHOLDS).astype(float);brier=float(np.mean((probs-observed)**2));log=float(np.mean(-(observed*np.log(probs)+(1-observed)*np.log1p(-probs))));bucket=np.searchsorted(THRESHOLDS,actual);bucket_probs=np.diff(np.r_[0,probs,1]);cum=np.cumsum(bucket_probs)[:-1];obs_cum=(np.arange(len(THRESHOLDS))>=bucket).astype(float);rps=float(np.sum((cum-obs_cum)**2));pit=float(cdf(actual));point=float(row.primary_point_forecast_usd);return {"candidate":name,"movie_id":int(row.movie_id),"origin":int(row.origin_day),"date":row.opening_weekend_start,"year":int(row.release_year),"point_bucket":_point_bucket(point),"prior_movies":prior,"pit":pit,"crps":float(2*np.trapezoid(loss,p)),"threshold_brier":brier,"threshold_log_score":log,"canonical_rps":rps,"q10":float(quantile(np.array([.1]))[0]),"q90":float(quantile(np.array([.9]))[0]),"q025":float(quantile(np.array([.025]))[0]),"q975":float(quantile(np.array([.975]))[0]),"actual":actual}
def _scores(rows:list[dict[str,Any]])->list[dict[str,Any]]:
    f=pd.DataFrame(rows);return [{"candidate":n,"movie_crps":g.groupby("movie_id").crps.mean().mean(),"threshold_brier":g.groupby("movie_id").threshold_brier.mean().mean(),"threshold_log_score":g.groupby("movie_id").threshold_log_score.mean().mean(),"canonical_rps":g.groupby("movie_id").canonical_rps.mean().mean(),"movies":g.movie_id.nunique()} for n,g in f.groupby("candidate")]
def _pit(rows:list[dict[str,Any]])->list[dict[str,Any]]:
    f=pd.DataFrame(rows);return [{"candidate":n,"pit_mean":g.groupby("movie_id").pit.mean().mean(),"pit_variance":g.pit.var(ddof=0),"below_10":(g.pit<.1).mean(),"above_90":(g.pit>.9).mean(),"below_025":(g.pit<.025).mean(),"above_975":(g.pit>.975).mean(),"movies":g.movie_id.nunique()} for n,g in f.groupby("candidate")]
def _metric(rows:list[dict[str,Any]],metric:str)->list[dict[str,Any]]:return [{"candidate":r["candidate"],metric:r[metric]} for r in _scores(rows)]
def _group(rows:list[dict[str,Any]],key:str)->list[dict[str,Any]]:
    f=pd.DataFrame(rows);return [{"candidate":n,key:v,"crps":g.crps.mean(),"threshold_log_score":g.threshold_log_score.mean(),"canonical_rps":g.canonical_rps.mean(),"movies":g.movie_id.nunique()} for (n,v),g in f.groupby(["candidate",key])]
def _intervals(rows:list[dict[str,Any]])->list[dict[str,Any]]:
    f=pd.DataFrame(rows);return [{"candidate":n,"coverage80":((g.actual>=g.q10)&(g.actual<=g.q90)).mean(),"coverage95":((g.actual>=g.q025)&(g.actual<=g.q975)).mean(),"lower_miss80":(g.actual<g.q10).mean(),"upper_miss80":(g.actual>g.q90).mean(),"width80":(g.q90-g.q10).mean()} for n,g in f.groupby("candidate")]
def _bootstrap(rows:list[dict[str,Any]],iterations:int,seed:int)->list[dict[str,Any]]:
    f=pd.DataFrame(rows);rng=np.random.default_rng(seed);out=[]
    for n,g in f.groupby("candidate"):
        means=g.groupby("movie_id")[["pit","crps","threshold_log_score","canonical_rps"]].mean().to_numpy();idx=rng.integers(0,len(means),(iterations,len(means)));sample=means[idx].mean(1);out.append({"candidate":n,"pit_lower_95":np.quantile(sample[:,0],.025),"pit_upper_95":np.quantile(sample[:,0],.975),"crps_lower_95":np.quantile(sample[:,1],.025),"crps_upper_95":np.quantile(sample[:,1],.975),"iterations":iterations})
    return out
def _leave(rows:list[dict[str,Any]],key:str)->list[dict[str,Any]]:
    f=pd.DataFrame(rows);out=[]
    for n,g in f.groupby("candidate"):
        for value in g[key].unique():out.append({"candidate":n,f"left_out_{key}":value,"crps":g.loc[g[key].ne(value)].crps.mean(),"threshold_log_score":g.loc[g[key].ne(value)].threshold_log_score.mean(),"canonical_rps":g.loc[g[key].ne(value)].canonical_rps.mean()})
    return out
def _select(scores:list[dict[str,Any]],pits:list[dict[str,Any]],bootstrap:list[dict[str,Any]])->tuple[str|None,list[dict[str,Any]]]:
    s={r["candidate"]:r for r in scores};p={r["candidate"]:r for r in pits};b={r["candidate"]:r for r in bootstrap};base=s["007_cdf_extension_001"];out=[]
    for name in s:
        if name=="007_cdf_extension_001":continue
        calibration=b[name]["pit_lower_95"]<=.5<=b[name]["pit_upper_95"] and abs(p[name]["above_90"]-p[name]["below_10"])<=.05
        scores_ok=s[name]["movie_crps"]<base["movie_crps"] and s[name]["threshold_log_score"]<=base["threshold_log_score"] and s[name]["canonical_rps"]<base["canonical_rps"]
        out.append({"candidate":name,"calibration_gate":calibration,"score_gate":scores_ok,"eligible":calibration and scores_ok,"crps":s[name]["movie_crps"],"threshold_log_score":s[name]["threshold_log_score"],"canonical_rps":s[name]["canonical_rps"]})
    eligible=[r for r in out if r["eligible"]];return (min(eligible,key=lambda r:(r["crps"],r["canonical_rps"]))["candidate"] if eligible else None),out
def _point_bucket(v:float)->str:return "lt_10m" if v<1e7 else ("10m_25m" if v<2.5e7 else ("25m_50m" if v<5e7 else ("50m_100m" if v<1e8 else "100m_plus")))
def _csv(path:Path,rows:list[dict[str,Any]])->None:
    if not rows:path.write_text("");return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
