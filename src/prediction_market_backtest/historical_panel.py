"""Raw historical price preservation and no-lookahead complete-vector pairing."""

from __future__ import annotations
import csv,hashlib,json,math
from datetime import datetime,timedelta,timezone
from decimal import Decimal
from pathlib import Path
from typing import Any,Iterable

import numpy as np
import pandas as pd

from .clients import ClobClient,GammaClient
from .market_history import PriceObservation,synchronize_prices
from .metrics import clustered_bootstrap,ranked_probability_score
from .probabilities import fixed_weight_blend,project_simplex


STALENESS_LIMITS=(timedelta(minutes=15),timedelta(hours=1),timedelta(hours=6),timedelta(hours=24))
EXPECTED_POLICY="pre_release_pointscale_cal_001"
EXPECTED_CHECKSUM="a4748ccc38e2e7af7593cde0fbfd9db748c0594f5c716a39d79bd8564a1b24d9"
PRIMARY_STALENESS=timedelta(hours=24)


def normalize_price_history(token_id:str,raw:Iterable[dict[str,Any]])->list[PriceObservation]:
    if not isinstance(token_id,str):raise TypeError("token ID must remain a string")
    dedup={}
    for item in raw:
        timestamp=item.get("t",item.get("timestamp"));price=Decimal(str(item.get("p",item.get("price"))))
        observed=datetime.fromtimestamp(float(timestamp),timezone.utc) if not isinstance(timestamp,str) or timestamp.replace(".","",1).isdigit() else datetime.fromisoformat(timestamp.replace("Z","+00:00"))
        key=(observed,price);dedup[key]=PriceObservation(token_id,observed,float(price))
    return sorted(dedup.values(),key=lambda row:row.observed_at)


def pair_complete_event(market_ids:list[str],observations:Iterable[PriceObservation],forecast_available:datetime,*,staleness:timedelta,market_open:datetime|None=None,resolved_at:datetime|None=None)->dict[str,Any]:
    if market_open and forecast_available<market_open:raise ValueError("market had not opened")
    if resolved_at and forecast_available>=resolved_at:raise ValueError("forecast is post-resolution")
    vector=synchronize_prices(market_ids,observations,forecast_available,staleness);raw=np.asarray(vector.raw);coherent=np.asarray(vector.coherent);state_payload=[(row.market_id,row.observed_at.isoformat(),row.yes_price) for row in vector.observations]
    return {"market_ids":market_ids,"raw_vector":raw.tolist(),"raw_sum":float(raw.sum()),"projected_vector":coherent.tolist(),"projection_distance":float(np.linalg.norm(coherent-raw)),"maximum_adjustment":float(np.max(np.abs(coherent-raw))),"maximum_price_age_seconds":max((forecast_available-row.observed_at).total_seconds() for row in vector.observations),"shared_market_state_id":hashlib.sha256(json.dumps(state_payload,sort_keys=True).encode()).hexdigest()}


def build_phase9c_historical_panel(review_dir:str|Path="data/diagnostics/prediction_market_reviewed_event_universe_v6",
                                  price_dir:str|Path="data/diagnostics/prediction_market_historical_price_panel_v6",
                                  diagnostic_dir:str|Path="data/diagnostics/prediction_market_probability_diagnostic_pointscale_beta_v3",
                                  forecast_panel:str|Path="models/boxoffice/boxoffice_local_007_weighted_interval_calibration/pre_release_panel.csv",
                                  *,gamma:GammaClient|None=None,clob:ClobClient|None=None,
                                  fetch_prices:bool=True,bootstrap_iterations:int=5000,seed:int=0)->dict[str,Any]:
    """Build the locked Phase 9C historical panel and diagnostic outputs.

    The function is intentionally fail-closed: unavailable token mappings, price
    history, forecast rows, or actuals are retained as rejection audit rows and
    never become scored panel rows.
    """
    review_root=Path(review_dir);price_root=Path(price_dir);diag_root=Path(diagnostic_dir)
    price_root.mkdir(parents=True,exist_ok=True);diag_root.mkdir(parents=True,exist_ok=True)
    queue=_read_csv(review_root/"01_prioritized_review_queue.csv");approved=_read_csv(review_root/"04_approved_events.csv")
    _validate_locked_reviews(review_root)
    queue_by_event={row["event_id"]:row for row in queue}
    approved=[{**row,**{f"queue_{k}":v for k,v in queue_by_event.get(row["event_id"],{}).items()}} for row in approved]
    gamma=gamma or GammaClient("https://gamma-api.polymarket.com")
    clob=clob or ClobClient("https://clob.polymarket.com")
    retrieval=[];raw_prices=[];token_map={};now=datetime.now(timezone.utc).isoformat()
    for event in approved:
        buckets=_buckets(event)
        resolved=_resolve_tokens(event,gamma) if fetch_prices else {}
        token_map[event["event_id"]]=resolved
        for bucket in buckets:
            market_id=str(bucket.get("market_id",""));tokens=resolved.get(market_id,{})
            for side in ("yes","no"):
                token_id=str(tokens.get(side,""))
                if not token_id:
                    retrieval.append(_retrieval_row(event,market_id,"",side,now,"missing_token_mapping"))
                    continue
                try:
                    raw=clob.price_history(token_id,interval="max",fidelity=1) if fetch_prices else []
                    observations=normalize_price_history(token_id,raw)
                    if observations:
                        for obs in observations:
                            raw_prices.append({"event_id":event["event_id"],"market_id":market_id,"token_id":token_id,"side":side,"observed_at":obs.observed_at.isoformat(),"price":f"{obs.yes_price:.10f}","yes_price_proxy":f"{(obs.yes_price if side=='yes' else 1-obs.yes_price):.10f}","proxy_source":"direct_yes" if side=="yes" else "one_minus_no"})
                        status="ok"
                        earliest=observations[0].observed_at.isoformat();latest=observations[-1].observed_at.isoformat()
                    else:
                        status="empty_history";earliest=latest=""
                    retrieval.append(_retrieval_row(event,market_id,token_id,side,now,status,earliest,latest,len(observations)))
                except Exception as exc:
                    retrieval.append(_retrieval_row(event,market_id,token_id,side,now,"error",error=str(exc)[:500]))
    forecasts,rejections=_forecast_inventory(approved,forecast_panel)
    audit=_event_recovery_audit(approved,forecasts,raw_prices,token_map)
    raw_obs=_recovered_yes_observations(raw_prices)
    pairings=[];raw_vectors=[];projected=[];proj_diag=[];complete=[];shared={}
    for forecast in forecasts:
        event_id=forecast["event_id"];market_ids=[str(b["market_id"]) for b in json.loads(forecast["event_bucket_definitions"])]
        try:
            result=pair_complete_event(market_ids,raw_obs,_parse_dt(forecast["forecast_availability_timestamp"]),staleness=PRIMARY_STALENESS)
            min_age=min((_parse_dt(forecast["forecast_availability_timestamp"])-obs.observed_at).total_seconds() for obs in synchronize_prices(market_ids,raw_obs,_parse_dt(forecast["forecast_availability_timestamp"]),PRIMARY_STALENESS).observations)
            pairing={**forecast,"pairing_status":"paired","rejection_reason":"","selected_market_timestamp":max(obs.observed_at.isoformat() for obs in synchronize_prices(market_ids,raw_obs,_parse_dt(forecast["forecast_availability_timestamp"]),PRIMARY_STALENESS).observations),"maximum_price_age_seconds":result["maximum_price_age_seconds"],"minimum_price_age_seconds":min_age,"shared_market_state_id":result["shared_market_state_id"]}
            pairings.append(pairing);shared[result["shared_market_state_id"]]=shared.get(result["shared_market_state_id"],0)+1
            model=json.loads(forecast["model_probability_vector"]);market=result["projected_vector"];winning=int(forecast["actual_winning_bucket"])
            raw_vectors.append({"event_id":event_id,"movie_id":forecast["movie_id"],"origin":forecast["origin"],"raw_probability_vector":json.dumps(result["raw_vector"]),"raw_probability_sum":result["raw_sum"],"shared_market_state_id":result["shared_market_state_id"]})
            projected.append({"event_id":event_id,"movie_id":forecast["movie_id"],"origin":forecast["origin"],"projected_probability_vector":json.dumps(market),"projection_method":"simplex","shared_market_state_id":result["shared_market_state_id"]})
            proj_diag.append({"event_id":event_id,"movie_id":forecast["movie_id"],"origin":forecast["origin"],"raw_probability_sum":result["raw_sum"],"projection_distance":result["projection_distance"],"largest_component_adjustment":result["maximum_adjustment"],"adjusted_components":sum(abs(a-b)>1e-12 for a,b in zip(result["raw_vector"],market)),"maximum_price_age_seconds":result["maximum_price_age_seconds"],"minimum_price_age_seconds":min_age,"shared_market_state_id":result["shared_market_state_id"],"flags":_projection_flags(result,min_age)})
            complete.append({**pairing,"model_probability_vector":json.dumps(model),"raw_market_price_vector":json.dumps(result["raw_vector"]),"projected_market_probability_vector":json.dumps(market),"winning_bucket":winning,"model_brier":_multi_brier(model,winning),"market_brier":_multi_brier(market,winning),"model_log_loss":_multi_log_loss(model,winning),"market_log_loss":_multi_log_loss(market,winning),"model_rps":ranked_probability_score(model,winning),"market_rps":ranked_probability_score(market,winning)})
        except Exception as exc:
            rejections.append({**forecast,"pairing_status":"rejected","rejection_reason":str(exc)})
    primary_panel=_latest_valid_rows(complete)
    paper_rows,paper_summary,paper_by_setting=_paper_backtest(primary_panel,raw_prices)
    sensitivity=_staleness_sensitivity(forecasts,raw_obs)
    shared_rows=[{"shared_market_state_id":state,"forecast_origin_rows":count,"shared_state":count>1} for state,count in sorted(shared.items())]
    independent=len({row["movie_id"] for row in complete});gate=_sample_label(independent)
    scores=_score_rows(complete,gate,bootstrap_iterations,seed)
    decision=_diagnostic_decision(gate,complete,scores)
    coverage=_coverage_rows(approved,forecasts,complete,rejections)
    storage=_write_panel_outputs(price_root,raw_prices,forecasts,pairings,rejections,raw_vectors,projected,proj_diag,sensitivity,shared_rows,complete,coverage,audit,primary_panel,paper_rows,paper_summary,paper_by_setting)
    _write_diagnostic_outputs(diag_root,complete,scores,decision,gate,storage,bootstrap_iterations,paper_rows,paper_summary,paper_by_setting)
    summary={"approved_events_queried":len(approved),"events_with_usable_price_history":len({r["event_id"] for r in raw_prices}),"complete_event_vectors":len({(r["event_id"],r["shared_market_state_id"]) for r in complete}),"paired_movie_origin_rows":len(complete),"primary_movie_rows":len(primary_panel),"independent_movies":independent,"recoverable_independent_movies":len({r["movie_id"] for r in audit if r.get("included_in_panel")=="true"}),"sample_gate":gate,"diagnostic_decision":decision,"paper_backtest_label":"historical_price_based_paper_backtest","paper_movies_traded":paper_summary.get("movies_traded",0),"primary_staleness_rule":"24 hours","storage_format":storage}
    _write_csv(price_root/"12_token_price_retrieval_metadata.csv",retrieval)
    (price_root/"summary.md").write_text(_summary_markdown(summary,coverage,proj_diag,rejections,paper_summary))
    return summary


def initialize_blocked_panel(output_dir:str|Path,approved_events:int)->dict[str,Any]:
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    names=("01_raw_token_prices.csv","02_historical_forecast_inventory.csv","03_forecast_price_pairings.csv","04_pairing_rejections.csv","05_raw_market_vectors.csv","06_projected_market_vectors.csv","07_projection_diagnostics.csv","08_staleness_sensitivity.csv","09_shared_market_states.csv","10_panel_coverage.csv")
    for name in names:
        path=output/name
        if not path.exists():path.write_text("")
    status="blocked_no_approved_events" if approved_events==0 else "pending_price_sync";summary={"approved_events":approved_events,"panel_status":status,"primary_staleness_policy":"must_be_frozen_before_scores"};(output/"summary.md").write_text("# Historical price panel v6\n\n"+"\n".join(f"- {k}: `{v}`" for k,v in summary.items())+"\n");return summary


def _read_csv(path:Path)->list[dict[str,str]]:
    if not path.exists() or not path.read_text().strip():return []
    with path.open(newline="",encoding="utf-8") as handle:return list(csv.DictReader(handle))


def _write_csv(path:Path,rows:Iterable[dict[str,Any]])->None:
    values=list(rows);fields=list(dict.fromkeys(k for row in values for k in row)) or ["status"]
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(values)


def _write_table(path:Path,rows:list[dict[str,Any]])->str:
    frame=pd.DataFrame(rows)
    if path.suffix==".parquet":
        frame.to_parquet(path,index=False)
        return "parquet"
    frame.to_csv(path,index=False);return "csv"


def _validate_locked_reviews(root:Path)->None:
    decisions=_read_csv(root/"03_reviewed_event_decisions.csv")
    if any(not row.get("decision") for row in decisions):raise ValueError("blank review decision")
    for row in decisions:
        if not row.get("reviewer") or not row.get("reviewed_at") or str(row.get("locked","")).lower()!="true":raise ValueError(f"review row is not locked: {row.get('event_id')}")
        if row["decision"]=="approved" and (not row.get("approved_movie_id") or row.get("source_warnings")):raise ValueError(f"invalid approved review: {row.get('event_id')}")


def _buckets(event:dict[str,str])->list[dict[str,Any]]:
    payload=event.get("queue_bucket_table_json") or event.get("bucket_table_json") or "[]"
    return json.loads(payload)


def _resolve_tokens(event:dict[str,str],gamma:GammaClient)->dict[str,dict[str,str]]:
    slug=event.get("queue_event_slug") or event.get("event_slug") or ""
    rows=list(gamma.events(slug=slug,limit=5)) if slug else []
    mapping={}
    for row in rows:
        for market in row.get("markets",[]) or []:
            market_id=str(market.get("id") or market.get("market_id") or "")
            tokens=market.get("clobTokenIds") or market.get("clob_token_ids") or market.get("tokens") or []
            if isinstance(tokens,str):
                try:tokens=json.loads(tokens)
                except json.JSONDecodeError:tokens=[tokens]
            outcomes=market.get("outcomes") or []
            if isinstance(outcomes,str):
                try:outcomes=json.loads(outcomes)
                except json.JSONDecodeError:outcomes=[]
            yes="";no=""
            for token in tokens:
                if isinstance(token,dict) and str(token.get("outcome","")).lower()=="yes":yes=str(token.get("token_id") or token.get("id") or "")
                elif isinstance(token,dict) and str(token.get("outcome","")).lower()=="no":no=str(token.get("token_id") or token.get("id") or "")
                elif isinstance(token,(str,int)) and not yes:yes=str(token)
                elif isinstance(token,(str,int)) and yes and not no:no=str(token)
            if outcomes and len(tokens)>=len(outcomes):
                for outcome,token in zip(outcomes,tokens):
                    if isinstance(token,dict):token=str(token.get("token_id") or token.get("id") or "")
                    if str(outcome).lower()=="yes":yes=str(token)
                    if str(outcome).lower()=="no":no=str(token)
            if market_id:mapping[market_id]={"yes":yes,"no":no}
    return mapping


def _forecast_inventory(approved:list[dict[str,str]],panel_path:str|Path)->tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    panel=_read_csv(Path(panel_path));by_movie={}
    for row in panel:
        if row.get("origin_day") in {str(i) for i in range(-14,0)}:by_movie.setdefault(str(row.get("movie_id")),[]).append(row)
    forecasts=[];rejections=[]
    for event in approved:
        movie_id=str(event.get("approved_movie_id",""));buckets=_buckets(event)
        rows=by_movie.get(movie_id,[])
        if not rows:
            rejections.append({"event_id":event["event_id"],"movie_id":movie_id,"pairing_status":"rejected","rejection_reason":"missing_historical_forecast"})
            continue
        for row in rows:
            point=_float(row.get("primary_point_forecast_usd"));actual=_float(row.get("actual_opening_weekend_gross_usd"))
            if point is None or actual is None:
                rejections.append({"event_id":event["event_id"],"movie_id":movie_id,"origin":row.get("origin_day"),"pairing_status":"rejected","rejection_reason":"missing_point_or_actual"})
                continue
            probs=_pointscale_probabilities(point,buckets)
            if abs(sum(probs)-1)>1e-9:raise ValueError("model probability vector does not sum to one")
            winner=_winning_bucket(actual,buckets)
            if winner is None:
                rejections.append({"event_id":event["event_id"],"movie_id":movie_id,"origin":row.get("origin_day"),"pairing_status":"rejected","rejection_reason":"actual_outside_bucket_vector"})
                continue
            forecasts.append({"event_id":event["event_id"],"event_title":event.get("event_title",""),"movie_id":movie_id,"origin":row.get("origin_day"),"information_cutoff":row.get("forecast_origin_date"),"forecast_availability_timestamp":row.get("forecast_origin_date")+"T12:00:00+00:00","production_point_forecast_usd":point,"frozen_probability_policy":EXPECTED_POLICY,"artifact_checksum":EXPECTED_CHECKSUM,"event_bucket_definitions":json.dumps(buckets,sort_keys=True),"model_probability_vector":json.dumps(probs),"actual_winning_bucket":winner,"release_year":row.get("release_year"),"opening_weekend_start":row.get("opening_weekend_start")})
    return forecasts,rejections


def _pointscale_probabilities(point:float,buckets:list[dict[str,Any]])->list[float]:
    sigma=max(point*.35,1_000_000.0);raw=[]
    for bucket in buckets:
        lo=None if bucket.get("lower") in (None,"") else float(bucket["lower"]);hi=None if bucket.get("upper") in (None,"") else float(bucket["upper"])
        cdf_hi=1.0 if hi is None else _normal_cdf((hi-point)/sigma)
        cdf_lo=0.0 if lo is None else _normal_cdf((lo-point)/sigma)
        raw.append(max(0.0,cdf_hi-cdf_lo))
    return project_simplex(raw)


def _winning_bucket(actual:float,buckets:list[dict[str,Any]])->int|None:
    for i,bucket in enumerate(buckets):
        lo=None if bucket.get("lower") in (None,"") else float(bucket["lower"]);hi=None if bucket.get("upper") in (None,"") else float(bucket["upper"])
        if (lo is None or actual>=lo) and (hi is None or actual<hi):return i
    return None


def _normal_cdf(value:float)->float:return .5*(1+math.erf(value/math.sqrt(2)))
def _float(value:object)->float|None:
    try:return float(value) if value not in (None,"") else None
    except ValueError:return None
def _parse_dt(value:str)->datetime:return datetime.fromisoformat(value.replace("Z","+00:00"))
def _multi_brier(probs:list[float],winner:int)->float:return sum((p-(1 if i==winner else 0))**2 for i,p in enumerate(probs))
def _multi_log_loss(probs:list[float],winner:int)->float:return -math.log(max(1e-9,probs[winner]))


def _retrieval_row(event:dict[str,str],market_id:str,token_id:str,side:str,requested_at:str,status:str,earliest:str="",latest:str="",count:int=0,error:str="")->dict[str,Any]:
    return {"event_id":event["event_id"],"market_id":market_id,"token_id":token_id,"side":side,"request_timestamp":requested_at,"requested_interval":"max","earliest_returned_price":earliest,"latest_returned_price":latest,"observation_count":count,"retrieval_status":status,"error_details":error}


def _recovered_yes_observations(raw_prices:list[dict[str,Any]])->list[PriceObservation]:
    best:dict[tuple[str,str],dict[str,Any]]={}
    for row in raw_prices:
        key=(str(row["market_id"]),str(row["observed_at"]))
        previous=best.get(key)
        if previous is None or (previous.get("proxy_source")!="direct_yes" and row.get("proxy_source")=="direct_yes"):
            best[key]=row
    return [PriceObservation(market_id,_parse_dt(observed_at),float(row["yes_price_proxy"])) for (market_id,observed_at),row in best.items()]


def _event_recovery_audit(approved:list[dict[str,str]],forecasts:list[dict[str,Any]],raw_prices:list[dict[str,Any]],token_map:dict[str,dict[str,dict[str,str]]])->list[dict[str,Any]]:
    forecasts_by_event:dict[str,list[dict[str,Any]]]={}
    for row in forecasts:forecasts_by_event.setdefault(str(row["event_id"]),[]).append(row)
    prices_by_event_market_side={(str(r["event_id"]),str(r["market_id"]),str(r["side"])) for r in raw_prices}
    obs=_recovered_yes_observations(raw_prices)
    rows=[]
    for event in approved:
        event_id=str(event["event_id"]);movie_id=str(event.get("approved_movie_id",""));buckets=_buckets(event);market_ids=[str(b.get("market_id","")) for b in buckets]
        event_forecasts=forecasts_by_event.get(event_id,[])
        token_complete=all(token_map.get(event_id,{}).get(m,{}).get("yes") and token_map.get(event_id,{}).get(m,{}).get("no") for m in market_ids)
        yes_available=all((event_id,m,"yes") in prices_by_event_market_side for m in market_ids)
        no_available=all((event_id,m,"no") in prices_by_event_market_side for m in market_ids)
        any_price_before=False;complete=False;within=False
        for forecast in event_forecasts:
            at=_parse_dt(forecast["forecast_availability_timestamp"])
            if any(o.market_id in market_ids and o.observed_at<=at for o in obs):any_price_before=True
            try:
                pair_complete_event(market_ids,obs,at,staleness=PRIMARY_STALENESS);complete=True;within=True
            except Exception:pass
        reason=""
        if not event_forecasts:reason="missing_historical_forecast"
        elif not token_complete:reason="token_mapping_incomplete"
        elif not yes_available and not no_available:reason="no_yes_or_no_history"
        elif not yes_available and no_available:reason="yes_recovered_from_no_history" if complete else "only_no_history_but_incomplete"
        elif not any_price_before:reason="market_opened_after_forecast"
        elif not complete:reason="incomplete_bucket_vector"
        elif not within:reason="outside_24h_staleness"
        rows.append({"event_id":event_id,"movie_id":movie_id,"approved":"true","token_mapping_complete":str(token_complete).lower(),"yes_history_available":str(yes_available).lower(),"no_history_available":str(no_available).lower(),"market_open_before_forecast":str(any_price_before).lower(),"internal_forecast_available":str(bool(event_forecasts)).lower(),"complete_bucket_vector":str(complete).lower(),"within_24h_staleness":str(within).lower(),"included_in_panel":str(complete).lower(),"primary_failure_reason":reason or "included"})
    return rows


def _projection_flags(result:dict[str,Any],min_age:float)->str:
    flags=[]
    if abs(result["raw_sum"]-1)>.1:flags.append("raw_sum_far_from_one")
    if result["projection_distance"]>.1:flags.append("large_projection_adjustment")
    if result["maximum_price_age_seconds"]-min_age>3600:flags.append("asynchronous_prices")
    return ";".join(flags)


def _staleness_sensitivity(forecasts:list[dict[str,Any]],observations:list[PriceObservation])->list[dict[str,Any]]:
    rows=[]
    for limit in STALENESS_LIMITS:
        paired=0
        for forecast in forecasts:
            try:
                pair_complete_event([str(b["market_id"]) for b in json.loads(forecast["event_bucket_definitions"])],observations,_parse_dt(forecast["forecast_availability_timestamp"]),staleness=limit);paired+=1
            except Exception:pass
        rows.append({"staleness_limit":str(limit),"paired_rows":paired,"coverage_rate":paired/len(forecasts) if forecasts else 0})
    return rows


def _sample_label(n:int)->str:
    if n<10:return "blocked_small_sample"
    if n<20:return "exploratory"
    if n<30:return "diagnostic_small_sample"
    return "diagnostic"


def _score_rows(rows:list[dict[str,Any]],gate:str,iterations:int,seed:int)->list[dict[str,Any]]:
    if gate=="blocked_small_sample":return []
    out=[]
    for weight in [i/10 for i in range(11)]:
        deltas=[];movies=[]
        for row in rows:
            model=json.loads(row["model_probability_vector"]);market=json.loads(row["projected_market_probability_vector"]);blend=fixed_weight_blend(model,market,weight);winner=int(row["winning_bucket"])
            deltas.append(_multi_brier(blend,winner)-_multi_brier(market,winner));movies.append(row["movie_id"])
        ci=clustered_bootstrap(movies,deltas,iterations=iterations,seed=seed) if deltas else {}
        out.append({"blend_weight_model":weight,"rows":len(deltas),"mean_blend_minus_market_brier":float(np.mean(deltas)) if deltas else "","ci_lower_95":ci.get("lower_95",""),"ci_upper_95":ci.get("upper_95",""),"probability_blend_worse_than_market":ci.get("probability_positive","")})
    return out


def _diagnostic_decision(gate:str,complete:list[dict[str,Any]],scores:list[dict[str,Any]])->str:
    if not complete:return "diagnostic_blocked_no_panel"
    if gate=="blocked_small_sample":return "diagnostic_inconclusive_small_sample"
    model=np.mean([row["model_brier"]-row["market_brier"] for row in complete])
    if model<-0.02:return "diagnostic_model_promising"
    if model>0.02:return "diagnostic_market_dominates"
    return "diagnostic_inconclusive_uncertainty"


def _coverage_rows(approved:list[dict[str,str]],forecasts:list[dict[str,Any]],complete:list[dict[str,Any]],rejections:list[dict[str,Any]])->list[dict[str,Any]]:
    return [{"metric":"approved_events","value":len(approved)},{"metric":"forecast_rows","value":len(forecasts)},{"metric":"paired_rows","value":len(complete)},{"metric":"independent_movies","value":len({r["movie_id"] for r in complete})},*({"metric":"rejected_"+str(reason),"value":sum(r.get("rejection_reason")==reason for r in rejections)} for reason in sorted({r.get("rejection_reason","") for r in rejections}))]


def _latest_valid_rows(complete:list[dict[str,Any]])->list[dict[str,Any]]:
    latest={}
    for row in complete:
        key=str(row["movie_id"])
        current=latest.get(key)
        candidate_key=(_parse_dt(row["forecast_availability_timestamp"]),str(row["event_id"]))
        current_key=(_parse_dt(current["forecast_availability_timestamp"]),str(current["event_id"])) if current else None
        if current is None or candidate_key>current_key:latest[key]=row
    return [latest[key] for key in sorted(latest,key=lambda value:int(value) if str(value).isdigit() else str(value))]


def _selected_price(raw_prices:list[dict[str,Any]],event_id:str,market_id:str,side:str,at:datetime)->float|None:
    candidates=[row for row in raw_prices if str(row["event_id"])==event_id and str(row["market_id"])==market_id and row["side"]==side and _parse_dt(row["observed_at"])<=at and at-_parse_dt(row["observed_at"])<=PRIMARY_STALENESS]
    if not candidates:return None
    latest=max(candidates,key=lambda row:_parse_dt(row["observed_at"]))
    return float(latest["price"])


def _paper_backtest(primary_rows:list[dict[str,Any]],raw_prices:list[dict[str,Any]])->tuple[list[dict[str,Any]],dict[str,Any],list[dict[str,Any]]]:
    rows=[];fee=0.0
    for threshold in (.02,.05,.10):
        for adverse in (0.0,.01,.02):
            cumulative=0.0;peak=0.0;max_drawdown=0.0
            for row in primary_rows:
                at=_parse_dt(row["forecast_availability_timestamp"]);event_id=str(row["event_id"]);model=json.loads(row["model_probability_vector"]);buckets=json.loads(row["event_bucket_definitions"]);winner=int(row["winning_bucket"])
                best=None
                for index,bucket in enumerate(buckets):
                    market_id=str(bucket["market_id"]);yes_price=_selected_price(raw_prices,event_id,market_id,"yes",at)
                    no_price=_selected_price(raw_prices,event_id,market_id,"no",at)
                    if yes_price is None and no_price is not None:yes_price=1-no_price
                    if no_price is None and yes_price is not None:no_price=1-yes_price
                    candidates=[]
                    if yes_price is not None:candidates.append(("yes",yes_price,model[index]-yes_price-fee,index==winner))
                    if no_price is not None:candidates.append(("no",no_price,(1-model[index])-no_price-fee,index!=winner))
                    for side,price,edge,won in candidates:
                        if edge>=threshold and (best is None or edge>best["estimated_net_edge"]):
                            best={"event_id":event_id,"movie_id":row["movie_id"],"event_title":row["event_title"],"origin":row["origin"],"bucket_index":index,"side":side,"historical_price":price,"estimated_net_edge":edge,"won":won}
                if best is None:
                    rows.append({"label":"historical_price_based_paper_backtest","edge_threshold":threshold,"adverse_execution_cents":adverse,"movie_id":row["movie_id"],"event_id":event_id,"traded":"false","pnl":0.0,"money_risked":0.0})
                    continue
                executed=min(.999999,best["historical_price"]+adverse);shares=10.0/executed if executed>0 else 0.0
                pnl=shares*((1.0 if best["won"] else 0.0)-executed-fee);cumulative+=pnl;peak=max(peak,cumulative);max_drawdown=min(max_drawdown,cumulative-peak)
                rows.append({**best,"label":"historical_price_based_paper_backtest","edge_threshold":threshold,"adverse_execution_cents":adverse,"executed_price":executed,"shares":shares,"money_risked":10.0,"pnl":pnl,"cumulative_pnl":cumulative,"max_drawdown_to_date":max_drawdown,"traded":"true","live_trading_eligible":"false"})
    primary=[r for r in rows if r.get("traded")=="true" and r["edge_threshold"]==.05 and r["adverse_execution_cents"]==0.0]
    all_primary=[r for r in rows if r["edge_threshold"]==.05 and r["adverse_execution_cents"]==0.0]
    total=sum(float(r.get("pnl",0)) for r in primary);risk=sum(float(r.get("money_risked",0)) for r in primary)
    movie_pnls={r["movie_id"]:float(r.get("pnl",0)) for r in primary}
    summary={"label":"historical_price_based_paper_backtest","primary_edge_threshold":0.05,"independent_movies":len(primary_rows),"movies_traded":len(primary),"total_net_pnl":total,"average_pnl_per_movie":total/len(primary_rows) if primary_rows else 0.0,"return_on_money_risked":total/risk if risk else 0.0,"win_rate":sum(float(r.get("pnl",0))>0 for r in primary)/len(primary) if primary else 0.0,"maximum_drawdown":min([float(r.get("max_drawdown_to_date",0)) for r in all_primary] or [0.0]),"largest_winning_movie":max(movie_pnls,key=movie_pnls.get) if movie_pnls else "","largest_losing_movie":min(movie_pnls,key=movie_pnls.get) if movie_pnls else "","profit_concentration":max(movie_pnls.values())/total if total>0 and movie_pnls else 0.0,"model_brier_score":float(np.mean([r["model_brier"] for r in primary_rows])) if primary_rows else "","market_brier_score":float(np.mean([r["market_brier"] for r in primary_rows])) if primary_rows else "","model_ranked_probability_score":float(np.mean([r["model_rps"] for r in primary_rows])) if primary_rows else "","market_ranked_probability_score":float(np.mean([r["market_rps"] for r in primary_rows])) if primary_rows else "","model_log_loss":float(np.mean([r["model_log_loss"] for r in primary_rows])) if primary_rows else "","market_log_loss":float(np.mean([r["market_log_loss"] for r in primary_rows])) if primary_rows else "","sample_interpretation":_paper_sample_label(len(primary_rows))}
    by_setting=[]
    for threshold in (.02,.05,.10):
        for adverse in (0.0,.01,.02):
            subset=[r for r in rows if r["edge_threshold"]==threshold and r["adverse_execution_cents"]==adverse]
            traded=[r for r in subset if r.get("traded")=="true"];setting_pnl=sum(float(r.get("pnl",0)) for r in traded);setting_risk=sum(float(r.get("money_risked",0)) for r in traded)
            by_setting.append({"label":"historical_price_based_paper_backtest","edge_threshold":threshold,"adverse_execution_cents":adverse,"independent_movies":len(primary_rows),"movies_traded":len(traded),"total_net_pnl":setting_pnl,"average_pnl_per_movie":setting_pnl/len(primary_rows) if primary_rows else 0.0,"return_on_money_risked":setting_pnl/setting_risk if setting_risk else 0.0,"win_rate":sum(float(r.get("pnl",0))>0 for r in traded)/len(traded) if traded else 0.0})
    return rows,summary,by_setting


def _paper_sample_label(n:int)->str:
    if n<10:return "inconclusive"
    if n<20:return "exploratory"
    if n<30:return "useful_but_preliminary"
    return "meaningful_historical_diagnostic"


def _write_panel_outputs(root:Path,raw_prices,forecasts,pairings,rejections,raw_vectors,projected,proj_diag,sensitivity,shared,complete,coverage,audit,primary_panel,paper_rows,paper_summary,paper_by_setting)->str:
    for legacy in ("01_raw_token_prices.csv","02_historical_forecast_inventory.csv","03_forecast_price_pairings.csv","05_raw_market_vectors.csv","06_projected_market_vectors.csv","10_panel_coverage.csv"):
        path=root/legacy
        if path.exists() and path.stat().st_size==0:path.unlink()
    formats=[_write_table(root/"01_raw_token_prices.parquet",raw_prices),_write_table(root/"02_historical_forecast_inventory.parquet",forecasts),_write_table(root/"03_forecast_price_pairings.parquet",pairings),_write_table(root/"05_raw_market_vectors.parquet",raw_vectors),_write_table(root/"06_projected_market_vectors.parquet",projected),_write_table(root/"10_complete_historical_panel.parquet",complete)]
    _write_csv(root/"04_pairing_rejections.csv",rejections);_write_csv(root/"07_projection_diagnostics.csv",proj_diag);_write_csv(root/"08_staleness_sensitivity.csv",sensitivity);_write_csv(root/"09_shared_market_states.csv",shared);_write_csv(root/"11_panel_coverage.csv",coverage)
    _write_csv(root/"13_event_recovery_audit.csv",audit);_write_table(root/"14_primary_latest_origin_panel.parquet",primary_panel);_write_csv(root/"15_historical_price_based_paper_backtest.csv",paper_rows);_write_csv(root/"16_historical_price_based_paper_backtest_summary.csv",[paper_summary]);_write_csv(root/"17_pnl_by_edge_threshold.csv",paper_by_setting)
    return "parquet" if all(item=="parquet" for item in formats) else "csv_fallback_at_parquet_path"


def _write_diagnostic_outputs(root:Path,complete:list[dict[str,Any]],scores:list[dict[str,Any]],decision:str,gate:str,storage:str,iterations:int,paper_rows:list[dict[str,Any]],paper_summary:dict[str,Any],paper_by_setting:list[dict[str,Any]])->None:
    manifest={"study_role":"historical_diagnostic_only","probability_approval_eligible":False,"trading_approval_eligible":False,"probability_policy":EXPECTED_POLICY,"artifact_checksum":EXPECTED_CHECKSUM,"model_changes_permitted":False,"sample_gate":gate,"storage_format":storage,"bootstrap_iterations":iterations}
    (root/"01_study_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    _write_csv(root/"02_score_by_blend_weight.csv",scores)
    _write_csv(root/"03_decision.csv",[{"diagnostic_decision":decision,"sample_gate":gate,"scored_rows":len(complete),"independent_movies":len({r["movie_id"] for r in complete})}])
    _write_csv(root/"04_scored_panel_audit.csv",complete)
    _write_csv(root/"05_historical_price_based_paper_backtest.csv",paper_rows);_write_csv(root/"06_historical_price_based_paper_backtest_summary.csv",[paper_summary]);_write_csv(root/"07_pnl_by_edge_threshold.csv",paper_by_setting)
    (root/"summary.md").write_text(f"# Locked historical model-versus-market diagnostic\n\n- decision: `{decision}`\n- sample_gate: `{gate}`\n- scored_rows: `{len(complete)}`\n- independent_movies: `{len({r['movie_id'] for r in complete})}`\n- paper_backtest_label: `historical_price_based_paper_backtest`\n- paper_movies_traded_at_5c: `{paper_summary.get('movies_traded',0)}`\n- paper_total_net_pnl_at_5c: `{paper_summary.get('total_net_pnl',0)}`\n- paper_sample_interpretation: `{paper_summary.get('sample_interpretation','')}`\n\nThis diagnostic cannot approve probabilities, alter the frozen artifact, or enable trading.\n")


def _summary_markdown(summary:dict[str,Any],coverage:list[dict[str,Any]],proj_diag:list[dict[str,Any]],rejections:list[dict[str,Any]],paper_summary:dict[str,Any])->str:
    rejection_counts={r["metric"].removeprefix("rejected_"):r["value"] for r in coverage if str(r["metric"]).startswith("rejected_")}
    ages=[float(r["maximum_price_age_seconds"]) for r in proj_diag];dist=[float(r["projection_distance"]) for r in proj_diag]
    extra={"rows_rejected_by_reason":rejection_counts,"price_age_seconds_p50":float(np.median(ages)) if ages else "","price_age_seconds_p95":float(np.quantile(ages,.95)) if ages else "","projection_distance_p50":float(np.median(dist)) if dist else "","projection_distance_p95":float(np.quantile(dist,.95)) if dist else "","paper_backtest_summary":paper_summary}
    return "# Historical price panel v6\n\n"+"\n".join(f"- {k}: `{v}`" for k,v in {**summary,**extra}.items())+"\n"
