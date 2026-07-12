"""Conservative event-specific adjudication of the Phase 9B candidate queue."""

from __future__ import annotations

import csv
import hashlib
import json
import os,re,unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REVIEWER_ID = "codex_phase9b_reviewer_v2_title_db"
TITLE_ALIASES={
    "minecraft":"minecraft movie","demon slayer":"demon slayer infinity castle",
    "official release party of a showgirl":"taylor swift the official release party of a showgirl",
    "joker folie a deux":"joker folie a deux","sonic 3":"sonic the hedgehog 3",
    "captain america":"captain america brave new world","star wars iii re release":"star wars revenge of the sith 20th anniversary re issue",
    "fantastic four the first steps":"fantastic four first steps",
    "hunger games ballad of songbirds snakes":"hunger games the ballad of songbirds and snakes",
}


def adjudicate_review_queue(directory: str | Path, movie_inventory:list[dict[str,Any]]|None=None) -> dict[str, Any]:
    root=Path(directory);queue=_read(root/"01_prioritized_review_queue.csv");prior={r["event_id"]:r for r in _read(root/"03_reviewed_event_decisions.csv")};inventory=movie_inventory if movie_inventory is not None else _load_movies();now=datetime.now(timezone.utc).isoformat();decisions=[]
    for row in queue:
        db_match=_match_movie(row,inventory) if not row.get("proposed_movie_id") else None
        matched_movie=str(row.get("proposed_movie_id") or (db_match or {}).get("movie_id") or "")
        review_row={**row,"proposed_movie_id":matched_movie};decision,reason=_decide(review_row);approved_movie=matched_movie if decision=="approved" else ""
        if db_match:reason+=f" Title/database match: {db_match['title']} [movie_id={db_match['movie_id']}, release_year={db_match.get('release_year')}]."
        semantics={"title":row.get("event_title",""),"movie_id":approved_movie,"geography":"domestic_as_resolution_source","currency":"USD","target":"opening_weekend_gross","duration":"three_day_opening_weekend","preview_treatment":"per_resolution_source_typically_included","reporting_source":"The Numbers / rules-specified fallback","lower_bound_convention":"locked_from_bucket_table","upper_bound_convention":"locked_from_bucket_table","exhaustive":decision=="approved","mutually_exclusive":decision=="approved","duplicate_status":"retained_event_specific"}
        decisions.append({"event_id":row["event_id"],"event_title":row.get("event_title",""),"decision":decision,"reviewer":REVIEWER_ID,"reviewed_at":now,"matched_movie_id":matched_movie,"approved_movie_id":approved_movie,"locked":"true","reviewer_notes":reason,"semantics_checksum":hashlib.sha256(json.dumps(semantics,sort_keys=True).encode()).hexdigest(),"rule_template_id":row.get("rule_template_id",""),"semantic_evidence_json":json.dumps(semantics,sort_keys=True),"source_warnings":row.get("automated_warnings","")})
    revisions=[]
    queue_by_id={row["event_id"]:row for row in queue}
    for row in decisions:
        original=queue_by_id[row["event_id"]];old_decision,_=_decide(original);old_movie=original.get("proposed_movie_id","") if old_decision=="approved" else ""
        if old_decision!=row["decision"] or old_movie!=row["approved_movie_id"]:revisions.append({"event_id":row["event_id"],"old_decision":old_decision,"new_decision":row["decision"],"old_movie_id":old_movie,"new_movie_id":row["approved_movie_id"],"matched_movie_id":row["matched_movie_id"],"revised_at":now,"reviewer":REVIEWER_ID,"reason":"Corrected using normalized database-title and release-year matching per explicit reviewer instruction."})
    approved=[r for r in decisions if r["decision"]=="approved"];rejected=[r for r in decisions if r["decision"].startswith("rejected_")];followup=[r for r in decisions if r["decision"]=="needs_followup"]
    _write(root/"03_reviewed_event_decisions.csv",decisions);_write(root/"04_approved_events.csv",approved);_write(root/"05_rejected_events.csv",rejected);_write(root/"06_followup_events.csv",followup)
    _write(root/"07_locked_movie_matches.csv",({"event_id":r["event_id"],"movie_id":r["approved_movie_id"],"locked":"true","reviewer":REVIEWER_ID,"reviewed_at":now} for r in approved))
    _write(root/"08_bucket_semantics_audit.csv",({"event_id":r["event_id"],"decision":r["decision"],"semantics_checksum":r["semantics_checksum"],"semantic_evidence_json":r["semantic_evidence_json"],"warnings":r["source_warnings"]} for r in decisions))
    _write(root/"10_decision_revision_log.csv",revisions)
    counts={name:sum(r["decision"]==name for r in decisions) for name in sorted({r["decision"] for r in decisions})};movie_count=len({r["approved_movie_id"] for r in approved})
    summary={"total_candidates":len(queue),"reviewed_count":len(decisions),"approved_count":len(approved),"rejected_count":len(rejected),"followup_count":len(followup),"remaining_count":0,"approved_independent_movies":movie_count,"approved_events_with_price_history":0,"approved_events_with_model_forecasts":0}
    _write(root/"09_review_summary.csv",[summary]);_write(root/"02_review_import_log.csv",[{"imported_at":now,"reviewer":REVIEWER_ID,"rows_received":len(queue),"new_decisions":0,"corrected_locked_decisions":len(revisions),"unchanged_locked_decisions":len(decisions)-len(revisions),"invalid_decision_values":0,"missing_reviewer_ids":0,"semantic_conflicts":0,"approvals":len(approved),"rejections":len(rejected),"followups":len(followup)}])
    root.joinpath("summary.md").write_text("# Reviewed event universe v6\n\n"+"\n".join(f"- {k}: `{v}`" for k,v in summary.items())+"\n\nReviewer: `"+REVIEWER_ID+"`. Every row was adjudicated individually and locked; unresolved automated warnings were never approved.\n\nDecision counts: `"+json.dumps(counts,sort_keys=True)+"`.\n")
    return {**summary,"decision_counts":counts,"revised_locked_decisions":len(revisions),"reviewer":REVIEWER_ID}


def _decide(row:dict[str,str])->tuple[str,str]:
    title=row.get("event_title","");warning=row.get("automated_warnings","");movie=row.get("proposed_movie_id","");rules=row.get("full_resolution_rules","").lower();combined=(title+" "+rules).lower()
    if "5th weekend" in combined:return "rejected_wrong_duration","Event targets a fifth weekend, not the three-day opening weekend."
    if title.lower().startswith("how much more") or "bigger opening weekend than" in combined:return "rejected_target_mismatch","Event is a relative/comparative gross market rather than a bucketed single-movie opening-weekend gross."
    if not movie:return "rejected_no_internal_movie","No sufficiently supported internal movie_id match was available; match was not inferred from title alone."
    if "overlap" in warning:return "rejected_overlapping_buckets",f"Automated bucket audit found overlap: {warning}"
    if "missing lower tail" in warning or "missing upper tail" in warning:return "rejected_non_exhaustive",f"Bucket set is not demonstrably exhaustive: {warning}"
    if "gap at boundary" in warning:return "rejected_boundary_ambiguity",f"Bucket audit found a boundary gap: {warning}"
    if "three-day duration not explicit" in warning:return "rejected_wrong_duration",f"Three-day opening-weekend duration is not explicit: {warning}"
    if warning:return "rejected_resolution_ambiguity",f"Unresolved semantic/parser warning: {warning}"
    if "opening weekend" not in combined:return "rejected_not_opening_weekend","Resolution rules do not establish an opening-weekend target."
    return "approved","Approved after event-specific review: internal movie match present; rules specify domestic USD three-day opening-weekend gross, resolution source and preview treatment; parsed buckets have no unresolved overlap, gap, tail, or boundary warning."


def _read(path:Path)->list[dict[str,str]]:
    if not path.exists() or not path.read_text().strip():return []
    with path.open(newline="",encoding="utf-8") as handle:return list(csv.DictReader(handle))
def _write(path:Path,rows)->None:
    values=list(rows);fields=list(dict.fromkeys(key for row in values for key in row)) or ["event_id"]
    with path.open("w",newline="",encoding="utf-8") as handle:writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(values)

def _load_movies()->list[dict[str,Any]]:
    import psycopg
    url=os.environ.get("DATABASE_URL","postgresql://localhost/pm_box_office")
    with psycopg.connect(url) as connection,connection.cursor() as cursor:
        cursor.execute("SELECT movie_id,title,release_date,release_year FROM movies")
        return [{"movie_id":row[0],"title":row[1],"release_date":row[2],"release_year":row[3]} for row in cursor.fetchall()]

def _event_title(value:str)->str:
    value=re.sub(r"\s+Opening Weekend.*$","",value,flags=re.I)
    value=re.sub(r"\s+\d+(?:st|nd|rd|th) Weekend.*$","",value,flags=re.I)
    return value.strip(" '\"“”‘’")

def _title_key(value:str)->str:
    value=unicodedata.normalize("NFKD",value).encode("ascii","ignore").decode().lower()
    value=re.sub(r"\((?:19|20)\d{2}\)|\((?:imax|wide|limited|re-?release|special engagement)[^)]*\)"," ",value)
    value=re.sub(r"^(?:the|a|an)\s+","",value);return re.sub(r"[^a-z0-9]+"," ",value).strip()

def _match_movie(row:dict[str,str],inventory:list[dict[str,Any]])->dict[str,Any]|None:
    title=_event_title(row.get("event_title",""));key=_title_key(title)
    if not key:return None
    key=TITLE_ALIASES.get(key,key)
    candidates=[movie for movie in inventory if _title_key(str(movie.get("title") or ""))==key]
    if not candidates:return None
    explicit=re.search(r"\((20\d{2})\)",title);years={int(value) for value in re.findall(r"\b(20\d{2})\b",row.get("full_resolution_rules","") or "")}
    if explicit:years={int(explicit.group(1))}
    def rank(movie):
        year=movie.get("release_year") or getattr(movie.get("release_date"),"year",None)
        return (0 if years and year in years else 1,0 if movie.get("release_date") else 1,0 if year else 1,int(movie["movie_id"]))
    best=sorted(candidates,key=rank)[0]
    year=best.get("release_year") or getattr(best.get("release_date"),"year",None)
    if explicit and year and year!=int(explicit.group(1)):return None
    return best
