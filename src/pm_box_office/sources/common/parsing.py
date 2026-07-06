"""Shared parsing helpers for source ingests."""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Any


def clean_text(value: object | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", html.unescape(str(value)).replace("\xa0", " ")).strip()


def normalize_title(value: str, *, remove_studio_possessives: bool = False) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s*\(\d{4}\)\s*$", "", text)
    text = text.lower()
    if remove_studio_possessives:
        text = re.sub(r"\b(disney|marvel|warner bros|universal|paramount|sony)'?s\b", " ", text)
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def parse_int(value: Any, *, suffix_multipliers: bool = False) -> int | None:
    if value in (None, ""):
        return None
    text = clean_text(value).lower()
    if not text or text in {"-", "n/a", "\\n", "\\N".lower()}:
        return None
    multiplier = 1
    if suffix_multipliers:
        if text.endswith("k"):
            multiplier = 1_000
            text = text[:-1]
        elif text.endswith("m"):
            multiplier = 1_000_000
            text = text[:-1]
        elif text.endswith("b"):
            multiplier = 1_000_000_000
            text = text[:-1]
    number = re.sub(r"[^0-9.-]", "", text)
    if not number or number in {"-", "."}:
        return None
    try:
        return int(float(number) * multiplier)
    except ValueError:
        return None


def parse_money(value: Any) -> int | None:
    text = clean_text(value)
    if not text or text.lower() in {"-", "n/a"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    digits = re.sub(r"[^0-9]", "", text)
    if not digits:
        return None
    amount = int(digits)
    return -amount if negative else amount
