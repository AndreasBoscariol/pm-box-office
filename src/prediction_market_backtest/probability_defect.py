"""Diagnose location, scale, and tail defects without market data."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cdf_validation import build_candidates
from .policy007 import build_chronological_pool


def run_defect_diagnostic(panel_path:str|Path,output_dir:str|Path,*,bootstrap_iterations:int=5000,seed:int=42)->dict[str,Any]:
    panel=pd.read_csv(panel_path);panel["opening_weekend_start"]=pd.to_datetime(panel.opening_weekend_start);rows=[];coverage=[];shrinkage=[]
    levels=(.1,.2,.3,.4,.5,.6,.7,.8,.9,.95)
    for _,row in panel.sort_values("opening_weekend_start").iterrows():
        try:pool=build_chronological_pool(panel,row)
        except RuntimeError:continue
        candidates=build_candidates(pool);current=candidates[0];raw=candidates[2];point=float(row.primary_point_forecast_usd);actual=float(row.actual_opening_weekend_gross_usd);error=float(np.log(actual/point));pit=current.cdf(actual,point)
        base={"movie_id":int(row.movie_id),"origin":int(row.origin_day),"release_year":int(row.release_year),"point":point,"actual":actual,"log_error":error,"absolute_log_error":abs(error),"squared_log_error":error**2,"actual_above_point":int(actual>point),"pit":pit,"fallback_level":pool.fallback_level,"effective_n":pool.effective_sample_size,"source_count":row.get("source_count"),"is_franchise":row.get("is_franchise"),"point_bucket":_point_bucket(point)};rows.append(base)
        for level in levels:
            alpha=1-level;lower,upper=current.quantile([alpha/2,1-alpha/2],point);coverage.append({"movie_id":int(row.movie_id),"origin":int(row.origin_day),"nominal":level,"covered":int(lower<=actual<=upper),"lower_miss":int(actual<lower),"upper_miss":int(actual>upper),"width":upper-lower,"relative_width":(upper-lower)/point})
        shrinkage.append({"movie_id":int(row.movie_id),"origin":int(row.origin_day),"raw_pit":raw.cdf(actual,point),"shrunken_pit":pit,"pit_change":pit-raw.cdf(actual,point),"raw_median":float(raw.quantile(.5,point)),"shrunken_median":float(current.quantile(.5,point))})
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True);frame=pd.DataFrame(rows);cov=pd.DataFrame(coverage)
    _csv(output/"01_point_bias.csv",rows);_csv(output/"02_bias_by_origin.csv",_group(frame,["origin"]));_csv(output/"03_bias_by_point_bucket.csv",_group(frame,["point_bucket"]));_csv(output/"04_coverage_curve.csv",_coverage(cov));_csv(output/"05_tail_miss_balance.csv",_coverage(cov));_csv(output/"06_pit_by_origin.csv",_pit_group(frame,["origin"]));_csv(output/"07_pit_by_size.csv",_pit_group(frame,["point_bucket"]));_csv(output/"08_pit_by_fallback.csv",_pit_group(frame,["fallback_level"]));frame["effective_n_band"]=pd.cut(frame.effective_n,[0,10,20,40,80,np.inf]).astype(str);_csv(output/"09_pit_by_effective_n.csv",_pit_group(frame,["effective_n_band"]));_csv(output/"10_shrinkage_effect.csv",shrinkage)
    movie=frame.groupby("movie_id").agg({"log_error":"mean","absolute_log_error":"mean","squared_log_error":"mean","actual_above_point":"mean","pit":"mean"}).reset_index();_csv(output/"11_movie_balanced_summary.csv",movie.to_dict("records"));_csv(output/"12_movie_cluster_bootstrap.csv",_bootstrap(movie,bootstrap_iterations,seed))
    mean_error=float(movie.log_error.mean());above=float(movie.actual_above_point.mean());upper=float((frame.pit>.9).mean());lower=float((frame.pit<.1).mean());defects=[]
    if mean_error>.05 or above>.55:defects.append("location")
    empirical=cov.groupby("nominal").covered.mean();
    if float(np.mean(np.abs(empirical.index.to_numpy()-empirical.to_numpy())))>.05:defects.append("scale")
    if upper-lower>.05:defects.append("upper_tail")
    if lower-upper>.05:defects.append("lower_tail")
    result={"rows":len(frame),"movies":int(frame.movie_id.nunique()),"movie_balanced_mean_log_error":mean_error,"proportion_actual_above_point":above,"lower_tail_rate":lower,"upper_tail_rate":upper,"primary_defects":defects or ["insufficient_sample"]}
    (output/"summary.md").write_text("# 007 probability defect diagnostic\n\n"+"\n".join(f"- {k}: `{v}`" for k,v in result.items())+"\n")
    return result


def _group(frame:pd.DataFrame,keys:list[str])->list[dict[str,Any]]:
    out=[]
    for key,g in frame.groupby(keys,dropna=False):
        key=key if isinstance(key,tuple) else (key,);item=dict(zip(keys,key));item.update({"n_rows":len(g),"n_movies":g.movie_id.nunique(),"mean_log_error":g.log_error.mean(),"median_log_error":g.log_error.median(),"mean_absolute_log_error":g.absolute_log_error.mean(),"rmse_log":np.sqrt(g.squared_log_error.mean()),"proportion_actual_above_point":g.actual_above_point.mean()});out.append(item)
    return out
def _pit_group(frame:pd.DataFrame,keys:list[str])->list[dict[str,Any]]:
    out=[]
    for key,g in frame.groupby(keys,dropna=False):
        key=key if isinstance(key,tuple) else (key,);item=dict(zip(keys,key));item.update({"n_rows":len(g),"n_movies":g.movie_id.nunique(),"pit_mean":g.pit.mean(),"pit_median":g.pit.median(),"pit_variance":g.pit.var(ddof=0),"below_10":(g.pit<.1).mean(),"above_90":(g.pit>.9).mean(),"below_025":(g.pit<.025).mean(),"above_975":(g.pit>.975).mean()});out.append(item)
    return out
def _coverage(frame:pd.DataFrame)->list[dict[str,Any]]:return frame.groupby("nominal").agg({"covered":"mean","lower_miss":"mean","upper_miss":"mean","width":"mean","relative_width":"mean"}).reset_index().to_dict("records")
def _bootstrap(movie:pd.DataFrame,iterations:int,seed:int)->list[dict[str,Any]]:
    rng=np.random.default_rng(seed);values=[]
    for _ in range(iterations):values.append(movie.iloc[rng.integers(0,len(movie),len(movie))].mean(numeric_only=True).to_dict())
    return [{"metric":column,"median":np.median([row[column] for row in values]),"lower_95":np.quantile([row[column] for row in values],.025),"upper_95":np.quantile([row[column] for row in values],.975),"iterations":iterations} for column in ("log_error","actual_above_point","pit")]
def _point_bucket(value:float)->str:return "lt_10m" if value<1e7 else ("10m_25m" if value<2.5e7 else ("25m_50m" if value<5e7 else ("50m_100m" if value<1e8 else "100m_plus")))
def _csv(path:Path,rows:list[dict[str,Any]])->None:
    if not rows:path.write_text("");return
    with path.open("w",newline="",encoding="utf-8") as handle:writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
