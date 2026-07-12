"""Validated, lock-preserving CSV workflow for event review decisions."""

from __future__ import annotations

import csv,hashlib,json
from dataclasses import dataclass
from datetime import datetime,timezone
from pathlib import Path
from typing import Mapping


DECISIONS={"approved","rejected_not_opening_weekend","rejected_wrong_geography","rejected_wrong_duration",
    "rejected_non_exhaustive","rejected_overlapping_buckets","rejected_boundary_ambiguity","rejected_resolution_ambiguity","rejected_no_internal_movie","rejected_target_mismatch",
    "rejected_duplicate","needs_followup"}


@dataclass(frozen=True,slots=True)
class ReviewDecision:
    event_id:str;decision:str;reviewer:str;reviewed_at:str;approved_movie_id:int|None;locked:bool;notes:str;semantics_checksum:str


def import_review_csv(path:str|Path,existing:Mapping[str,ReviewDecision]|None=None)->dict[str,ReviewDecision]:
    existing=dict(existing or {});result=dict(existing)
    with Path(path).open(newline="",encoding="utf-8") as handle:
        for line,row in enumerate(csv.DictReader(handle),2):
            event_id=str(row.get("event_id") or "").strip();decision=str(row.get("reviewer_decision") or "").strip();reviewer=str(row.get("reviewer") or "").strip()
            if not event_id:raise ValueError(f"line {line}: event_id required")
            prior=existing.get(event_id)
            if prior and prior.locked:
                candidate_checksum=_semantics_checksum(row)
                if decision!=prior.decision or candidate_checksum!=prior.semantics_checksum or _movie(row)!=prior.approved_movie_id:raise ValueError(f"line {line}: locked review cannot be changed")
                result[event_id]=prior;continue
            if decision not in DECISIONS:raise ValueError(f"line {line}: invalid reviewer decision {decision!r}")
            if not reviewer:raise ValueError(f"line {line}: reviewer required")
            locked=_boolean(row.get("locked"));movie_id=_movie(row);checksum=_semantics_checksum(row)
            if decision=="approved":
                if not locked:raise ValueError(f"line {line}: approval must be locked")
                if movie_id is None:raise ValueError(f"line {line}: approval requires approved_movie_id")
                if str(row.get("automated_warnings") or "").strip():raise ValueError(f"line {line}: approval has unresolved semantic warnings")
            result[event_id]=ReviewDecision(event_id,decision,reviewer,str(row.get("reviewed_at") or datetime.now(timezone.utc).isoformat()),movie_id,locked,str(row.get("reviewer_notes") or ""),checksum)
    return result


def export_decisions(decisions:Mapping[str,ReviewDecision],path:str|Path)->None:
    fields=list(ReviewDecision.__dataclass_fields__)
    with Path(path).open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows([{field:getattr(value,field) for field in fields} for value in decisions.values()])


def _semantics_checksum(row:Mapping[str,str|None])->str:
    payload={key:row.get(key) for key in ("full_resolution_rules","bucket_table_json","parsed_target","approved_movie_id")};return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
def _movie(row:Mapping[str,str|None])->int|None:
    value=str(row.get("approved_movie_id") or "").strip();return int(value) if value else None
def _boolean(value:object)->bool:return str(value).strip().lower() in {"1","true","yes","locked"}
