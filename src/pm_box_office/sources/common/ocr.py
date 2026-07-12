"""Shared OCR helpers for image-backed source parsers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

from pm_box_office.sources.common.parsing import clean_text


@dataclass(frozen=True)
class OcrToken:
    text: str
    confidence: float | None
    left: int | None
    top: int | None
    width: int | None
    height: int | None


@dataclass(frozen=True)
class OcrResult:
    text: str
    mean_confidence: float | None
    tokens: list[OcrToken]


class TesseractOcr:
    def __init__(self, command: str = "tesseract", psm: int = 6, language: str = "eng") -> None:
        self.command = command
        self.psm = psm
        self.language = language

    def read(self, image_path: Path) -> OcrResult:
        executable = shutil.which(self.command)
        if executable is None:
            raise RuntimeError(
                f"{self.command!r} is not installed. Install Tesseract or use --ocr-text-dir "
                "with precomputed OCR text fixtures."
            )
        process = subprocess.run(
            [
                executable,
                str(image_path),
                "stdout",
                "--psm",
                str(self.psm),
                "-l",
                self.language,
                "-c",
                "preserve_interword_spaces=1",
                "tsv",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return ocr_result_from_tesseract_tsv(process.stdout)


def ocr_result_from_text(text: str, *, confidence: float | None = None) -> OcrResult:
    return OcrResult(text=normalize_ocr_text(text), mean_confidence=confidence, tokens=[])


def ocr_result_from_tesseract_tsv(tsv_text: str) -> OcrResult:
    reader = csv.DictReader(tsv_text.splitlines(), delimiter="\t")
    line_words: dict[tuple[int, int, int], list[tuple[int, str]]] = {}
    tokens: list[OcrToken] = []
    confidences: list[float] = []
    for row in reader:
        text = clean_text(row.get("text", ""))
        if not text:
            continue
        confidence = parse_float(row.get("conf"))
        if confidence is not None and confidence >= 0:
            confidences.append(confidence)
        left = parse_optional_int(row.get("left"))
        top = parse_optional_int(row.get("top"))
        width = parse_optional_int(row.get("width"))
        height = parse_optional_int(row.get("height"))
        tokens.append(OcrToken(text=text, confidence=confidence, left=left, top=top, width=width, height=height))
        key = (
            parse_optional_int(row.get("block_num")) or 0,
            parse_optional_int(row.get("par_num")) or 0,
            parse_optional_int(row.get("line_num")) or 0,
        )
        line_words.setdefault(key, []).append((left or 0, text))
    lines = [" ".join(word for _, word in sorted(words)) for _, words in sorted(line_words.items()) if words]
    mean_confidence = sum(confidences) / len(confidences) if confidences else None
    return OcrResult(
        text=normalize_ocr_text("\n".join(lines)),
        mean_confidence=mean_confidence,
        tokens=tokens,
    )


def normalize_ocr_text(text: str) -> str:
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u00a0", " ")
    return "\n".join(clean_text(line) for line in text.splitlines() if clean_text(line))


def parse_optional_int(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None
