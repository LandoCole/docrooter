"""Tesseract through its command line (the system binary; no pip dependency)."""

from __future__ import annotations

import re
import shutil
import subprocess
from collections import defaultdict

from docrooter.elements.models import Box
from docrooter.errors import DocrooterError
from docrooter.ocr.base import OcrLine, scale_box

__all__ = ["TesseractEngine"]

# Tesseract wants ISO 639-2 codes; map the common bare ones. Others pass through ("deu+eng").
_LANG = {
    "en": "eng", "de": "deu", "fr": "fra", "es": "spa", "it": "ita", "pt": "por", "nl": "nld",
    "ru": "rus", "zh": "chi_sim", "ja": "jpn", "ko": "kor", "ar": "ara",
}  # fmt: skip
_TIMEOUT = 300  # seconds per image
_LANG_OK = re.compile(r"[A-Za-z0-9_]+(\+[A-Za-z0-9_]+)*")  # "eng", "deu+eng", "chi_sim"


class TesseractEngine:
    """``tesseract stdin stdout -l <lang> --psm 3 tsv``: words are grouped into lines by
    ``(block, paragraph, line)`` and a line's confidence is the mean of its words' / 100."""

    name = "tesseract"

    def __init__(self, *, lang: str = "en") -> None:
        language = _LANG.get(lang, lang)
        if not _LANG_OK.fullmatch(language):
            raise ValueError(
                f"invalid tesseract language {lang!r}: use codes such as 'eng' or 'deu+eng'"
            )
        binary = shutil.which("tesseract")
        if binary is None:
            raise DocrooterError(
                "the tesseract binary was not found on PATH: install it "
                "(macOS: brew install tesseract; Debian/Ubuntu: apt install tesseract-ocr)"
            )
        self._binary = binary
        self._lang = language

    def recognize(self, image: bytes, *, width_px: int, height_px: int, page: Box) -> list[OcrLine]:
        try:
            proc = subprocess.run(
                [self._binary, "stdin", "stdout", "-l", self._lang, "--psm", "3", "tsv"],
                input=image,
                capture_output=True,
                timeout=_TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise DocrooterError(f"tesseract timed out after {_TIMEOUT} s") from None
        if proc.returncode != 0:
            detail = proc.stderr.decode("utf-8", "replace").strip()
            raise DocrooterError(f"tesseract failed (exit {proc.returncode}): {detail}")
        return parse_tsv(proc.stdout.decode("utf-8", "replace"), width_px, height_px, page)

    def close(self) -> None:
        pass


def parse_tsv(tsv: str, width_px: int, height_px: int, page: Box) -> list[OcrLine]:
    """OCR lines from Tesseract's TSV output, in the engine's order."""
    words: dict[tuple[int, int, int], list[tuple[str, float, int, int, int, int]]] = defaultdict(
        list
    )
    for row in tsv.splitlines()[1:]:  # the first row is the header
        cols = row.split("\t", 11)
        if len(cols) < 12 or cols[0] != "5":  # level 5 = word
            continue
        text = cols[11].strip()
        try:
            conf = float(cols[10])
            left, top, w, h = (int(c) for c in cols[6:10])
            key = (int(cols[2]), int(cols[3]), int(cols[4]))
        except ValueError:
            continue
        if text and conf >= 0:  # conf -1 marks non-word rows
            words[key].append((text, conf, left, top, left + w, top + h))
    lines = []
    for items in words.values():  # insertion order = the engine's reading order
        box = scale_box(
            page, width_px, height_px,
            min(i[2] for i in items), min(i[3] for i in items),
            max(i[4] for i in items), max(i[5] for i in items),
        )  # fmt: skip
        lines.append(OcrLine(
            text=" ".join(i[0] for i in items), box=box,
            confidence=sum(i[1] for i in items) / len(items) / 100.0,
        ))  # fmt: skip
    return lines
