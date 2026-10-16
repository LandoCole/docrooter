"""Native PDF line extraction with two engines: ``pymupdf`` (extra ``pdf``) and ``pdfplumber``
(extra ``pdf-mit``). No OCR; pages with no text layer simply yield no lines."""

from __future__ import annotations

import importlib
import math
import statistics
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from docrooter.errors import DocrooterError, MissingExtraError

__all__ = ["Line", "extract_lines"]

Engine = Literal["pymupdf", "pdfplumber"]

_BOLD_FLAG = 16
_BOLD_NAMES = ("bold", "black", "heavy", "semibold")
_BOLD_FRACTION = 0.5
_LINE_TOLERANCE = 2.0  # pdfplumber: words whose tops differ by at most this share a line
_COLUMN_GAP_FACTOR = 3.0  # pdfplumber: a gap above this many median char widths splits a line
_BLOCK_GAP_FACTOR = 0.5  # pdfplumber: a gap above this fraction of the line height starts a block


class Line(BaseModel):
    """One visual line of text with its typography and position (points, origin top-left)."""

    page: int  # 1-based
    text: str
    size: float  # character-weighted dominant span size, rounded to 0.1
    bold: bool  # at least half of the characters are bold
    font: str  # font of the dominant span
    x0: float
    y0: float
    x1: float
    y1: float
    page_height: float
    page_width: float
    gap_above: float  # vertical gap to the previous line on the page; inf at the page top
    gap_below: float  # vertical gap to the next line on the page; inf at the page bottom
    block: int  # 0-based block index within the page, in reading order


def _import_pymupdf() -> Any:
    try:
        return importlib.import_module("pymupdf")
    except ImportError:
        pass
    try:
        return importlib.import_module("fitz")
    except ImportError as exc:
        raise MissingExtraError(
            "pymupdf is required for this feature: pip install 'docrooter[pdf]'"
        ) from exc


def _import_pdfplumber() -> Any:
    try:
        return importlib.import_module("pdfplumber")
    except ImportError as exc:
        raise MissingExtraError(
            "pdfplumber is required for this feature: pip install 'docrooter[pdf-mit]'"
        ) from exc


def _has_bold_name(font: str) -> bool:
    lowered = font.casefold()
    return any(word in lowered for word in _BOLD_NAMES)


def _summarise(spans: list[tuple[str, float, str, bool]]) -> tuple[float, str, bool]:
    """Dominant size, dominant font and bold verdict from ``(text, size, font, bold)`` spans."""
    sizes: Counter[float] = Counter()
    fonts: Counter[str] = Counter()
    total = 0
    bold_chars = 0
    for text, size, font, bold in spans:
        n = len(text)
        total += n
        sizes[round(size, 1)] += n
        fonts[font] += n
        if bold:
            bold_chars += n
    size = max(sizes.items(), key=lambda kv: kv[1])[0]
    font = max(fonts.items(), key=lambda kv: kv[1])[0]
    return size, font, total > 0 and bold_chars / total >= _BOLD_FRACTION


def _has_alnum(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


def _finalise(rows: list[dict[str, Any]]) -> list[Line]:
    """Add vertical gaps (reading order, clamped at 0, inf at page edges) and build ``Line``s."""
    lines: list[Line] = []
    start = 0
    while start < len(rows):
        end = start
        while end < len(rows) and rows[end]["page"] == rows[start]["page"]:
            end += 1
        for i in range(start, end):
            row = rows[i]
            row["gap_above"] = max(0.0, row["y0"] - rows[i - 1]["y1"]) if i > start else math.inf
            row["gap_below"] = max(0.0, rows[i + 1]["y0"] - row["y1"]) if i + 1 < end else math.inf
            lines.append(Line(**row))
        start = end
    return lines


def _pymupdf_rows(doc: Any, pymupdf: Any) -> Iterator[dict[str, Any]]:
    # Drop glyphs outside the media box (pymupdf >= 1.2x also clips to the page rect by default).
    flags = pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP
    for page_index, page in enumerate(doc):
        width, height = float(page.rect.width), float(page.rect.height)
        block_no = -1
        for block in page.get_text("dict", flags=flags, sort=True)["blocks"]:
            if block.get("type") != 0:
                continue
            block_no += 1
            for raw in block["lines"]:
                dx, dy = raw.get("dir", (1.0, 0.0))
                if abs(dy) > abs(dx):  # rotated text (margin stamps, vertical labels)
                    continue
                spans = [
                    (
                        s["text"],
                        float(s["size"]),
                        s["font"],
                        bool(s["flags"] & _BOLD_FLAG) or _has_bold_name(s["font"]),
                    )
                    for s in raw["spans"]
                    if s["text"].strip()
                ]
                text = "".join(s["text"] for s in raw["spans"]).strip()
                if not spans or not _has_alnum(text):
                    continue
                size, font, bold = _summarise(spans)
                x0, y0, x1, y1 = (float(v) for v in raw["bbox"])
                yield {
                    "page": page_index + 1, "text": text, "size": size, "bold": bold,
                    "font": font, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "page_height": height, "page_width": width, "block": block_no,
                }  # fmt: skip


def _split_columns(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split x-sorted words of one line wherever the gap exceeds 3x the median char width."""
    widths = [(w["x1"] - w["x0"]) / len(w["text"]) for w in words if w["text"]]
    limit = _COLUMN_GAP_FACTOR * statistics.median(widths) if widths else math.inf
    groups: list[list[dict[str, Any]]] = [[words[0]]]
    for word in words[1:]:
        if word["x0"] - groups[-1][-1]["x1"] > limit:
            groups.append([word])
        else:
            groups[-1].append(word)
    return groups


def _pdfplumber_rows(pdf: Any) -> Iterator[dict[str, Any]]:
    for page_index, page in enumerate(pdf.pages):
        width, height = float(page.width), float(page.height)
        words = page.extract_words(extra_attrs=["size", "fontname"])
        words.sort(key=lambda w: (w["top"], w["x0"]))
        grouped: list[list[dict[str, Any]]] = []
        for word in words:
            if grouped and word["top"] - grouped[-1][0]["top"] <= _LINE_TOLERANCE:
                grouped[-1].append(word)
            else:
                grouped.append([word])
        page_rows: list[dict[str, Any]] = []
        for line_words in grouped:
            line_words.sort(key=lambda w: w["x0"])
            for segment in _split_columns(line_words):
                text = " ".join(w["text"] for w in segment)
                if not _has_alnum(text):
                    continue
                spans = [
                    (w["text"], float(w["size"]), w["fontname"], _has_bold_name(w["fontname"]))
                    for w in segment
                ]
                size, font, bold = _summarise(spans)
                page_rows.append({
                    "page": page_index + 1, "text": text, "size": size, "bold": bold,
                    "font": font, "x0": float(min(w["x0"] for w in segment)),
                    "y0": float(min(w["top"] for w in segment)),
                    "x1": float(max(w["x1"] for w in segment)),
                    "y1": float(max(w["bottom"] for w in segment)),
                    "page_height": height, "page_width": width, "block": 0,
                })  # fmt: skip
        block_no = 0
        for i, row in enumerate(page_rows):
            if i:
                prev = page_rows[i - 1]
                gap = row["y0"] - prev["y1"]
                if gap > _BLOCK_GAP_FACTOR * max(row["y1"] - row["y0"], 1.0):
                    block_no += 1
            row["block"] = block_no
        yield from page_rows


def extract_lines(path: str | Path, *, engine: Engine) -> tuple[list[Line], int]:
    """Extract text lines (reading order) and the page count from ``path``.

    Raises ``DocrooterError`` for encrypted, corrupt or non-PDF files and ``MissingExtraError``
    when the engine's package is not installed.
    """
    if engine == "pymupdf":
        return _extract_pymupdf(path)
    if engine == "pdfplumber":
        return _extract_pdfplumber(path)
    raise DocrooterError(f"unknown PDF engine {engine!r}; use 'pymupdf' or 'pdfplumber'")


def _extract_pymupdf(path: str | Path) -> tuple[list[Line], int]:
    pymupdf = _import_pymupdf()
    try:
        doc = pymupdf.open(str(path), filetype="pdf")
    except Exception as exc:
        raise DocrooterError(f"cannot open PDF {path}: {exc}") from exc
    try:
        if doc.needs_pass:
            raise DocrooterError(f"cannot open PDF {path}: file is encrypted")
        n_pages = doc.page_count
        try:
            rows = list(_pymupdf_rows(doc, pymupdf))
        except Exception as exc:
            raise DocrooterError(f"cannot open PDF {path}: {exc}") from exc
    finally:
        doc.close()
    return _finalise(rows), n_pages


def _extract_pdfplumber(path: str | Path) -> tuple[list[Line], int]:
    pdfplumber = _import_pdfplumber()
    try:
        pdf = pdfplumber.open(str(path))
    except Exception as exc:
        raise DocrooterError(f"cannot open PDF {path}: {exc}") from exc
    try:
        n_pages = len(pdf.pages)
        try:
            rows = list(_pdfplumber_rows(pdf))
        except Exception as exc:
            raise DocrooterError(f"cannot open PDF {path}: {exc}") from exc
    except DocrooterError:
        raise
    except Exception as exc:
        raise DocrooterError(f"cannot open PDF {path}: {exc}") from exc
    finally:
        pdf.close()
    return _finalise(rows), n_pages
