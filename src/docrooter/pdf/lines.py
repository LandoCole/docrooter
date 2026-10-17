"""Native PDF line extraction with two engines: ``pymupdf`` (extra ``pdf``) and ``pdfplumber``
(extra ``pdf-mit``). No OCR; pages with no text layer simply yield no lines."""

from __future__ import annotations

import importlib
import math
import re
import statistics
from collections import Counter
from collections.abc import Callable, Iterator
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
_COLUMN_GAP_FACTOR = 3.0  # a gap above this many char widths splits a line (both engines)
_BLOCK_GAP_FACTOR = 0.5  # pdfplumber: a gap above this fraction of the line height starts a block
_SKEW_TOLERANCE = 1e-3  # pdfplumber: |b| + |c| below this share of the scale is float noise
_SIZE_NOISE = 1e-3  # pdfplumber: pt; a smaller size change along a line is float noise
_SAME_LINE_TOP = 1.0  # pdfplumber: pt; consecutive glyphs this close in top share a line
_ROW_OVERLAP = 0.5  # pymupdf: lines overlapping vertically by half the shorter height share a row
_WIDE_GAP_FACTOR = 2.0  # pymupdf: past this many char widths only a short label joins its row
_SHORT_LABEL_CHARS = 12
_LABEL = re.compile(r"^\(?[\dA-Za-z]{1,4}[.)]?$")  # "1", "3.", "(a)", "IV)"
_NO_SPACE_FACTOR = 0.15  # pymupdf: a gap under this many char widths joins without a space
PROGRESS_EVERY = 10  # pages between ``progress`` callbacks

Progress = Callable[[int, int], None]
"""``progress(pages_done, pages_total)``, called after every ``PROGRESS_EVERY``-th page."""


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


def _pymupdf_rows(
    doc: Any, pymupdf: Any, max_pages: int | None = None, progress: Progress | None = None
) -> Iterator[dict[str, Any]]:
    total = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
    # Drop glyphs outside the media box (pymupdf >= 1.2x also clips to the page rect by default).
    flags = pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP
    for page_index, page in enumerate(doc):
        if max_pages is not None and page_index >= max_pages:
            break
        width, height = float(page.rect.width), float(page.rect.height)
        block_no = -1
        for block in page.get_text("dict", flags=flags, sort=True)["blocks"]:
            if block.get("type") != 0:
                continue
            block_no += 1
            fragments: list[dict[str, Any]] = []
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
                if spans:
                    x0, y0, x1, y1 = (float(v) for v in raw["bbox"])
                    fragments.append(
                        {"spans": spans, "text": text, "x0": x0, "y0": y0, "x1": x1, "y1": y1}
                    )
            for frag in _join_row_fragments(fragments):
                if not _has_alnum(frag["text"]):
                    continue
                size, font, bold = _summarise(frag["spans"])
                yield {
                    "page": page_index + 1, "text": frag["text"], "size": size, "bold": bold,
                    "font": font, "x0": frag["x0"], "y0": frag["y0"], "x1": frag["x1"],
                    "y1": frag["y1"], "page_height": height, "page_width": width,
                    "block": block_no,
                }  # fmt: skip
        if progress is not None and (page_index + 1) % PROGRESS_EVERY == 0:
            progress(page_index + 1, total)


def _join_row_fragments(fragments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join consecutive pymupdf lines of one block that sit on the same row (a section number
    and its title, a run-in bold label and its text, a drop cap) so a ``Line`` is one visual
    line as with the pdfplumber engine.

    With ``w`` the average character width of the pair: a gap over ``3w`` never joins; a gap
    over ``2w`` joins only after a short left part (at most 12 characters, or a label such as
    "3." or "(a)"), since longer text there is usually an adjacent column or box; a gap under
    ``0.15w`` joins without a space ("T" + "his").
    """
    out: list[dict[str, Any]] = []
    for frag in fragments:
        if out and _same_row(out[-1], frag):
            prev = out[-1]
            gap = frag["x0"] - prev["x1"]
            chars = len(prev["text"]) + len(frag["text"])
            char_width = ((prev["x1"] - prev["x0"]) + (frag["x1"] - frag["x0"])) / max(chars, 1)
            left = prev["text"].strip()
            short_left = len(left) <= _SHORT_LABEL_CHARS or bool(_LABEL.match(left))
            if -char_width <= gap <= _COLUMN_GAP_FACTOR * char_width and (
                gap <= _WIDE_GAP_FACTOR * char_width or short_left
            ):
                sep = "" if gap < _NO_SPACE_FACTOR * char_width else " "
                prev["text"] = f"{prev['text']}{sep}{frag['text']}"
                prev["spans"] = prev["spans"] + frag["spans"]
                prev["x0"], prev["y0"] = min(prev["x0"], frag["x0"]), min(prev["y0"], frag["y0"])
                prev["x1"], prev["y1"] = max(prev["x1"], frag["x1"]), max(prev["y1"], frag["y1"])
                continue
        out.append(dict(frag))
    return out


def _same_row(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Vertical overlap of at least half the shorter fragment's height."""
    overlap = min(a["y1"], b["y1"]) - max(a["y0"], b["y0"])
    return overlap >= _ROW_OVERLAP * min(a["y1"] - a["y0"], b["y1"] - b["y0"])


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


def _pdfplumber_rows(
    pdf: Any, max_pages: int | None = None, progress: Progress | None = None
) -> Iterator[dict[str, Any]]:
    pages = pdf.pages[:max_pages]
    for page_index, page in enumerate(pages):
        try:
            page_rows = _pdfplumber_page_rows(page, page_index)
        finally:
            # Free the page's parsed objects and layout caches; pdfplumber keeps every page
            # it has opened, so a long PDF would otherwise hold all of them at once.
            close = getattr(page, "close", None) or getattr(page, "flush_cache", None)
            if close is not None:
                close()
        yield from page_rows
        if progress is not None and (page_index + 1) % PROGRESS_EVERY == 0:
            progress(page_index + 1, len(pages))


def _pdfplumber_words(page: Any) -> list[dict[str, Any]]:
    """The page's words. pdfplumber cuts the glyph stream into runs of equal ``upright``, size
    and font and finds the words of each run separately, so a size change always ends a word.

    pdfminer derives a glyph's size from its transformed box, so a text matrix with a tiny
    skew (``b``/``c`` around 1e-6, as some producers write) gives every glyph a slightly
    different float size, and with ``b * c > 0`` it also calls the glyph not upright. Every
    glyph then became a word of its own and the line text came out letter-spaced ("T o P r i
    m i t i v e"), which silently removed almost every heading of a long specification. See
    :func:`_denoised_chars`."""
    utils = _import_pdfplumber().utils
    return utils.extract_words(_denoised_chars(page.chars), extra_attrs=["size", "fontname"])


def _denoised_chars(chars: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copies of ``chars`` without float noise. A glyph that continues the previous one's line
    (same font, top within 1 pt) at a size within 0.001 pt of it takes that size; other
    size changes, such as nearby labels of a diagram, stay as they are. A glyph whose matrix is
    axis-aligned up to noise counts as upright, or pdfplumber would order the letters of its
    words top to bottom. Rotated text is left alone."""
    out: list[dict[str, Any]] = []
    prev: dict[str, Any] | None = None
    for char in chars:
        fixed = dict(char)
        if (
            prev is not None
            and fixed["fontname"] == prev["fontname"]
            and abs(fixed["size"] - prev["size"]) <= _SIZE_NOISE
            and abs(fixed["top"] - prev["top"]) <= _SAME_LINE_TOP
        ):
            fixed["size"] = prev["size"]
        a, b, c, d = (float(v) for v in fixed["matrix"][:4])
        if a * d > 0 and abs(b) + abs(c) <= _SKEW_TOLERANCE * max(abs(a), abs(d)):
            fixed["upright"] = True
        out.append(fixed)
        prev = fixed
    return out


def _pdfplumber_page_rows(page: Any, page_index: int) -> list[dict[str, Any]]:
    """The rows of one pdfplumber page, with block numbers."""
    width, height = float(page.width), float(page.height)
    words = _pdfplumber_words(page)
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
    return page_rows


def extract_lines(
    path: str | Path,
    *,
    engine: Engine,
    max_pages: int | None = None,
    progress: Progress | None = None,
) -> tuple[list[Line], int]:
    """Extract text lines (reading order) and the page count from ``path``.

    With ``max_pages`` only the first ``max_pages`` pages are read; the page count is still
    the whole document's. ``progress(done, total)`` is called after every 10th page read
    (``total`` is the number of pages read).

    Raises ``DocrooterError`` for encrypted, corrupt or non-PDF files and ``MissingExtraError``
    when the engine's package is not installed.
    """
    if engine == "pymupdf":
        return _extract_pymupdf(path, max_pages, progress)
    if engine == "pdfplumber":
        return _extract_pdfplumber(path, max_pages, progress)
    raise DocrooterError(f"unknown PDF engine {engine!r}; use 'pymupdf' or 'pdfplumber'")


def _extract_pymupdf(
    path: str | Path, max_pages: int | None = None, progress: Progress | None = None
) -> tuple[list[Line], int]:
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
            rows = list(_pymupdf_rows(doc, pymupdf, max_pages, progress))
        except Exception as exc:
            raise DocrooterError(f"cannot open PDF {path}: {exc}") from exc
    finally:
        doc.close()
    return _finalise(rows), n_pages


def _extract_pdfplumber(
    path: str | Path, max_pages: int | None = None, progress: Progress | None = None
) -> tuple[list[Line], int]:
    pdfplumber = _import_pdfplumber()
    try:
        pdf = pdfplumber.open(str(path))
    except Exception as exc:
        raise DocrooterError(f"cannot open PDF {path}: {exc}") from exc
    try:
        n_pages = len(pdf.pages)
        try:
            rows = list(_pdfplumber_rows(pdf, max_pages, progress))
        except Exception as exc:
            raise DocrooterError(f"cannot open PDF {path}: {exc}") from exc
    except DocrooterError:
        raise
    except Exception as exc:
        raise DocrooterError(f"cannot open PDF {path}: {exc}") from exc
    finally:
        pdf.close()
    return _finalise(rows), n_pages
