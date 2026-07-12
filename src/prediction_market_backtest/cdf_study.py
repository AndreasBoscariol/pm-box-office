"""Reproducible market-blind Phase 4 CDF candidate study."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cdf_validation import ANCHOR_PROBABILITIES, build_candidates, evaluate_candidate, select_candidate
from .policy007 import build_chronological_pool, shrunk_quantile_function


DEVELOPMENT_CUTOFF = pd.Timestamp("2024-01-01")


def run_cdf_study(panel_path: str | Path, output_dir: str | Path, *, bootstrap_iterations: int = 5000,
                  seed: int = 42) -> dict[str, Any]:
    panel=pd.read_csv(panel_path);panel["opening_weekend_start"]=pd.to_datetime(panel.opening_weekend_start)
    development=panel.loc[panel.opening_weekend_start<DEVELOPMENT_CUTOFF].sort_values(["opening_weekend_start","movie_id","origin_day"])
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    definitions={"selection_data":"movies released before 2024-01-01; no Polymarket prices",
        "candidates":[{"name":"007_cdf_extension_001","definition":"pointwise 007 shrink over full probability grid"},
        {"name":"007_anchor_preserving_empirical_warp_001","definition":"raw weighted empirical shape with monotone anchor-locked residual warp"},
        {"name":"007_raw_weighted_empirical_benchmark","definition":"unmodified selected-cell weighted empirical quantile function"}]}
    (output/"01_candidate_definitions.json").write_text(json.dumps(definitions,indent=2)+"\n")
    rows=[];anchors=[];losses=[];thresholds=[];intervals=[]
    fixed_thresholds=np.array([10,15,20,25,30,40,50,75,100])*1_000_000
    for _,row in development.iterrows():
        try:pool=build_chronological_pool(panel,row)
        except RuntimeError:continue
        point=float(row.primary_point_forecast_usd);outcome=float(row.actual_opening_weekend_gross_usd)
        locked=shrunk_quantile_function(pool,ANCHOR_PROBABILITIES)
        for candidate in build_candidates(pool):
            metrics=evaluate_candidate(candidate,outcome,point)
            base={"candidate":candidate.name,"movie_id":int(row.movie_id),"origin":int(row.origin_day),
                  "release_year":int(row.release_year),"preserves_anchors":candidate.preserves_anchors,**metrics}
            rows.append(base)
            actual=np.log(candidate.quantile(ANCHOR_PROBABILITIES,point)/point)
            for probability,expected,observed in zip(ANCHOR_PROBABILITIES,locked,actual):
                anchors.append({"candidate":candidate.name,"movie_id":int(row.movie_id),"origin":int(row.origin_day),
                    "probability":probability,"locked_residual":expected,"candidate_residual":observed,"absolute_difference":abs(expected-observed)})
            for probability in candidate.probabilities:
                quantile=float(candidate.quantile(probability,point))
                error=outcome-quantile;loss=probability*error if error>=0 else (1-probability)*-error
                losses.append({"candidate":candidate.name,"movie_id":int(row.movie_id),"origin":int(row.origin_day),"probability":probability,"loss":loss})
            for threshold in fixed_thresholds:
                probability=candidate.cdf(float(threshold),point)
                thresholds.append({"candidate":candidate.name,"movie_id":int(row.movie_id),"origin":int(row.origin_day),
                    "threshold":threshold,"probability":probability,"outcome_below":int(outcome<threshold),"brier":(probability-int(outcome<threshold))**2})
            intervals.append({"candidate":candidate.name,"movie_id":int(row.movie_id),"origin":int(row.origin_day),
                "weighted_interval_score":metrics["weighted_interval_score"]})
    selected,summary=select_candidate(rows)
    _csv(output/"02_quantile_anchor_identity.csv",anchors);_csv(output/"03_crps_by_candidate.csv",summary)
    _csv(output/"04_crps_by_origin.csv",_group(rows,["candidate","origin"],["crps"]))
    _csv(output/"05_pit_values.csv",[{k:r[k] for k in ("candidate","movie_id","origin","pit")} for r in rows])
    _csv(output/"06_pit_summary.csv",_group(rows,["candidate"],["pit","lower_tail","upper_tail"]))
    _csv(output/"07_quantile_loss.csv",_group(losses,["candidate","probability"],["loss"]))
    _csv(output/"08_threshold_calibration.csv",_group(thresholds,["candidate","threshold"],["probability","outcome_below","brier"]))
    _csv(output/"09_interval_score_identity.csv",_group(intervals,["candidate"],["weighted_interval_score"]))
    bootstrap=_bootstrap(rows,bootstrap_iterations,seed);_csv(output/"10_movie_cluster_bootstrap.csv",bootstrap)
    _csv(output/"11_leave_one_year_out.csv",_leave_year(rows));_csv(output/"12_candidate_selection.csv",summary)
    decision="approved" if selected else "not_approved"
    result={"development_movies":int(development.movie_id.nunique()),"development_rows":len(development),
        "evaluated_rows":len(rows),"selected_extension":selected,"cdf_status":decision,"market_prices_used":False}
    (output/"summary.md").write_text("# 007 CDF extension validation\n\n"+"\n".join(f"- {k}: `{v}`" for k,v in result.items())+"\n")
    return result


def _group(rows:list[dict[str,Any]],keys:list[str],values:list[str])->list[dict[str,Any]]:
    frame=pd.DataFrame(rows);result=[]
    for group_key,group in frame.groupby(keys,dropna=False):
        group_key=group_key if isinstance(group_key,tuple) else (group_key,)
        item=dict(zip(keys,group_key));item.update({f"mean_{value}":float(group[value].mean()) for value in values});item["n"]=len(group);result.append(item)
    return result
def _bootstrap(rows:list[dict[str,Any]],iterations:int,seed:int)->list[dict[str,Any]]:
    frame=pd.DataFrame(rows);pivot=frame.pivot_table(index=["movie_id","origin"],columns="candidate",values="crps").dropna();movies=pivot.index.get_level_values("movie_id").unique().to_numpy();rng=np.random.default_rng(seed)
    candidates=list(pivot.columns);pairs=[(a,b) for i,a in enumerate(candidates) for b in candidates[i+1:]];samples={pair:[] for pair in pairs}
    by_movie={movie:pivot.loc[movie] for movie in movies}
    for _ in range(iterations):
        chosen=rng.choice(movies,len(movies),replace=True);sample=pd.concat([by_movie[movie] for movie in chosen])
        for pair in pairs:samples[pair].append(float((sample[pair[0]]-sample[pair[1]]).mean()))
    return [{"candidate_a":a,"candidate_b":b,"mean_difference":float(np.mean(v)),"lower_95":float(np.quantile(v,.025)),"upper_95":float(np.quantile(v,.975)),"probability_a_better":float(np.mean(np.asarray(v)<0)),"iterations":iterations} for (a,b),v in samples.items()]
def _leave_year(rows:list[dict[str,Any]])->list[dict[str,Any]]:
    frame=pd.DataFrame(rows);out=[]
    for year in sorted(frame.release_year.unique()):
        for candidate,group in frame.loc[frame.release_year.ne(year)].groupby("candidate"):
            out.append({"left_out_year":year,"candidate":candidate,"mean_crps":float(group.crps.mean()),"movies":group.movie_id.nunique()})
    return out
def _csv(path:Path,rows:list[dict[str,Any]])->None:
    if not rows:return
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
