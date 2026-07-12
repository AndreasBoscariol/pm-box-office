"""Deterministic Gamma candidate filtering and human-review export."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .matching import MovieCandidate, match_movie
from .semantics import Bucket, parse_bucket, validate_bucket_set, with_override


REJECTION_TERMS={"worldwide":"worldwide gross","international":"international gross","total domestic":"total domestic run",
    "lifetime":"lifetime gross","production budget":"production budget","second weekend":"later weekend","2nd weekend":"later weekend",
    "third weekend":"later weekend","3rd weekend":"later weekend","5-day":"unsupported extended duration","4-day":"unsupported extended duration"}


def automated_filter(event: dict[str,Any]) -> tuple[bool,str]:
    title=str(event.get("title") or "");text=(title+" "+str(event.get("description") or "")).lower();markets=event.get("markets") or []
    for term,reason in REJECTION_TERMS.items():
        if term in text:return False,reason
    if "opening weekend" not in text:return False,"not opening weekend"
    if any(term in title.lower() for term in ("which movie"," vs"," or ","highest grossing","bigger")):return False,"comparison or non-bucket question"
    if len(markets)<2:return False,"single threshold or incomplete bucket set"
    if not all((market.get("description") or event.get("description")) for market in markets):return False,"incomplete resolution terms"
    return True,"candidate"


def parse_event_buckets(event:dict[str,Any])->tuple[list[Bucket],list[str]]:
    rules=" ".join([str(event.get("description") or "")]+[str(m.get("description") or "") for m in event.get("markets") or []])
    scope="domestic_us_canada" if "domestic" in rules.lower() else "ambiguous"
    source="the_numbers" if "the-numbers.com" in rules.lower() else ("box_office_mojo" if "boxofficemojo.com" in rules.lower() else None)
    buckets=[];errors=[]
    for market in event.get("markets") or []:
        try:buckets.append(parse_bucket(str(market.get("id")),str(market.get("question")),geographic_scope=scope,resolution_source=source))
        except ValueError as exc:errors.append(f"{market.get('id')}: {exc}")
    if "higher range bracket" in rules.lower() or "higher bracket" in rules.lower():
        ordered=sorted(buckets,key=lambda b:(b.lower is not None,b.lower or 0))
        buckets=[with_override(bucket,include_lower=True if bucket.lower is not None else bucket.include_lower,
            include_upper=False if bucket.upper is not None else bucket.include_upper) for bucket in ordered]
    validation=validate_bucket_set(buckets)
    errors.extend(validation.errors)
    return buckets,errors


def build_review_universe(events:Iterable[dict[str,Any]],production_panel:str|Path,output_dir:str|Path)->dict[str,Any]:
    events=list({str(event.get("id")):event for event in events}.values());output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    panel=pd.read_csv(production_panel);movie_rows=panel.sort_values("opening_weekend_start").drop_duplicates("movie_id",keep="last")
    candidates=[MovieCandidate(int(row.movie_id),str(row.title),pd.Timestamp(row.opening_weekend_start).date()) for row in movie_rows.itertuples()]
    discovered=[];filtered=[];semantics=[];validations=[];matches=[];queue=[];rejected=[]
    for event in events:
        eid=str(event.get("id"));title=str(event.get("title") or "");discovered.append({"event_id":eid,"title":title,"slug":event.get("slug"),"market_count":len(event.get("markets") or []),"active":event.get("active"),"closed":event.get("closed")})
        accepted,reason=automated_filter(event);filtered.append({"event_id":eid,"accepted":accepted,"reason":reason})
        if not accepted:rejected.append({"event_id":eid,"title":title,"reason":reason});continue
        buckets,errors=parse_event_buckets(event);rules=str(event.get("description") or "")
        semantics.append({"event_id":eid,"title":title,"geography":"domestic_us_canada" if "domestic" in rules.lower() else "ambiguous",
            "duration":3 if "3-day" in rules.lower() else "unspecified","currency":"USD","resolution_source":"the_numbers" if "the-numbers.com" in rules.lower() else ("box_office_mojo" if "boxofficemojo.com" in rules.lower() else "unknown"),"rules":rules})
        valid=not errors;validations.append({"event_id":eid,"valid":valid,"bucket_count":len(buckets),"errors":"; ".join(errors),"buckets_json":json.dumps([{"market_id":b.market_id,"lower":str(b.lower) if b.lower is not None else None,"upper":str(b.upper) if b.upper is not None else None,"include_lower":b.include_lower,"include_upper":b.include_upper} for b in buckets])})
        clean=re.sub(r"[\"']|\s+Opening Weekend.*$","",title,flags=re.I).strip();match=match_movie(clean,None,candidates)
        matches.append({"event_id":eid,"proposed_movie_id":match.movie_id,"match_status":match.status,"match_score":match.score})
        warnings=list(errors)
        if "3-day" not in rules.lower():warnings.append("three-day duration not explicit")
        queue.append({"event_id":eid,"event_title":title,"event_slug":event.get("slug"),"full_resolution_rules":rules,
            "bucket_table_json":validations[-1]["buckets_json"],"proposed_movie_id":match.movie_id,"match_score":match.score,
            "automated_warnings":"; ".join(warnings),"reviewer_decision":"","reviewer_notes":""})
    for name,rows in (("01_all_discovered_events.csv",discovered),("02_automated_candidate_filter.csv",filtered),("03_parsed_event_semantics.csv",semantics),
        ("04_bucket_set_validation.csv",validations),("05_proposed_movie_matches.csv",matches),("06_manual_review_queue.csv",queue),
        ("07_reviewed_event_decisions.csv",[{"event_id":row["event_id"],"reviewer_decision":"","reviewer_notes":"","locked":False} for row in queue]),
        ("08_approved_event_universe.csv",[]),("09_rejected_event_audit.csv",rejected)):_csv(output/name,rows)
    counts=Counter(row["reason"] for row in rejected);summary={"discovered_events":len(events),"automated_candidates":len(queue),"manually_reviewed_events":0,"approved_events":0,"approved_independent_movies":0,"rejections":dict(counts)}
    (output/"summary.md").write_text("# Reviewed event universe\n\n"+"\n".join(f"- {k}: `{v}`" for k,v in summary.items())+"\n\nNo event is approved until a reviewer completes and locks `07_reviewed_event_decisions.csv`.\n")
    return summary


def _csv(path:Path,rows:list[dict[str,Any]])->None:
    if not rows:path.write_text("",encoding="utf-8");return
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
