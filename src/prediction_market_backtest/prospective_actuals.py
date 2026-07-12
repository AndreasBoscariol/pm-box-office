"""Append-only prospective actual revisions and scheduled decision rules."""

from __future__ import annotations
import csv
from dataclasses import asdict,dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path


@dataclass(frozen=True,slots=True)
class ActualRevision:
    movie_id:int;gross_usd:Decimal;source:str;source_published_at:str;received_at:str;duration_days:int;currency:str;geography:str;revision_of:str|None=None;final_approved:bool=False


def append_actual(path:str|Path,revision:ActualRevision)->None:
    if revision.gross_usd<0 or revision.duration_days!=3 or revision.currency!="USD" or revision.geography!="domestic_us_canada":raise ValueError("actual target does not match locked target")
    path=Path(path);rows=[]
    if path.exists() and path.read_text().strip():rows=list(csv.DictReader(path.open()))
    record={key:str(value) for key,value in asdict(revision).items()};key=(record["movie_id"],record["source_published_at"],record["received_at"])
    if key in {(row["movie_id"],row["source_published_at"],row["received_at"]) for row in rows}:return
    rows.append(record)
    with path.open("w",newline="",encoding="utf-8") as handle:writer=csv.DictWriter(handle,fieldnames=list(record));writer.writeheader();writer.writerows(rows)


def scheduled_look(movie_count:int,prior_decision:str|None=None)->tuple[int|None,float|None,str]:
    if movie_count<30:return None,None,"inconclusive_insufficient_prospective_sample"
    if movie_count==30:return 30,.975,"formal_first_look"
    if movie_count<50:return None,None,"continue_to_50" if prior_decision=="continue_to_50" else "no_unscheduled_look"
    if movie_count==50 and prior_decision=="continue_to_50":return 50,.95,"formal_final_look"
    return None,None,"no_additional_checkpoint"
