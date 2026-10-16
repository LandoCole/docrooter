"""Document-wide signals computed from extracted lines: body size, line gap, running headers
and footers, page numbers, printed TOC pages and PDF bookmarks."""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from difflib import SequenceMatcher
from pathlib import Path

from pydantic import BaseModel, Field

from docrooter.errors import MissingExtraError
from docrooter.pdf.lines import Line, _import_pymupdf

__all__ = [
    "DocSignals",
    "compute_signals",
    "fuzzy_match",
    "is_fuzzy_match",
    "normalize_title",
    "read_bookmarks",
]

_EDGE_FRACTION = 0.08  # top/bottom share of the page where running headers/footers sit
_RECURRENCE_FRACTION = 0.30
_RECURRENCE_MINIMUM = 3
# A section running line (the current section's title printed in the edge band of each of its
# pages) recurs on far fewer than 30% of a long document's pages. Matched on its exact text
# minus a leading/trailing page number, so "Item 1." and "Item 2." stay distinct (P5b).
_SECTION_RECURRENCE_MINIMUM = 3
_TOC_MIN_LINES = 5
_TOC_HEADED_MIN_LINES = 3
_MIN_POSITIVE_GAP = 0.5  # gaps at or below this (pt) are line-box overlap, not spacing

_PAGE_NUMBER = re.compile(
    r"^(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$|^[-–—]\s*\d{1,4}\s*[-–—]$|^[ivxlcdm]{1,7}$",
    re.IGNORECASE,
)
# The roman-numeral branch above also matches words like "civil"; require a well-formed numeral.
_ROMAN = re.compile(r"^m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$", re.IGNORECASE)
_TOC_LINE = re.compile(r"^(.*?)[\s.·…_-]{2,}(\d{1,4})$")
_TOC_TRAILING = re.compile(r"[\s.·…_-]+$")
_TOC_HEADINGS = frozenset({"contents", "table of contents"})
_DIGITS = re.compile(r"\d+")
_EDGE_PAGE_NUMBER = re.compile(r"^\d{1,4}\s+|\s+\d{1,4}$")
_SPACE = re.compile(r"\s+")
_LEADING_NUMBER = re.compile(
    r"^(?:(?:chapter|section|part|item|article|appendix)\s+)?"
    r"(?:\d+(?:\.\d+)*[.)]?|[ivxlc]+[.)]|[a-z][.)])\s+",
    re.IGNORECASE,
)
_TRAILING_PUNCT = re.compile(r"[\s.,;:!?\-–—·…_]+$")
_FUZZY_THRESHOLD = 0.85


class DocSignals(BaseModel):
    """Document-wide typography and layout signals. ``excluded`` and ``toc_page_lines`` hold
    indexes into the ``lines`` list given to :func:`compute_signals`."""

    body_size: float
    median_gap: float  # median positive line gap among non-excluded lines (0.0 if none)
    excluded: set[int] = Field(default_factory=set)  # running headers/footers, page numbers
    printed_toc: list[tuple[str, str]] = Field(default_factory=list)  # (title, page label)
    toc_page_lines: set[int] = Field(default_factory=set)  # every non-excluded line on a TOC page
    bookmarks: list[tuple[int, str, int]] = Field(default_factory=list)  # (level, title, page)


def normalize_title(s: str) -> str:
    """Casefold, strip leading numbering and trailing punctuation, collapse whitespace."""
    text = _SPACE.sub(" ", s.casefold()).strip()
    text = _LEADING_NUMBER.sub("", text, count=1)
    return _TRAILING_PUNCT.sub("", text).strip()


def fuzzy_match(a: str, b: str) -> float:
    """``SequenceMatcher`` ratio of the two titles after :func:`normalize_title`."""
    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def is_fuzzy_match(a: str, b: str) -> bool:
    """True when :func:`fuzzy_match` reaches the 0.85 threshold."""
    return fuzzy_match(a, b) >= _FUZZY_THRESHOLD


def _is_page_number(text: str) -> bool:
    stripped = text.strip()
    if not _PAGE_NUMBER.match(stripped):
        return False
    if stripped.isalpha():
        return bool(_ROMAN.match(stripped))
    return True


def _running_key(text: str) -> str:
    return _SPACE.sub(" ", _DIGITS.sub("#", text.casefold())).strip()


def _section_running_key(text: str) -> str:
    return _EDGE_PAGE_NUMBER.sub("", _SPACE.sub(" ", text.casefold()).strip())


def _in_edge_zone(line: Line) -> bool:
    centre = (line.y0 + line.y1) / 2
    return (
        centre < _EDGE_FRACTION * line.page_height
        or centre > (1 - _EDGE_FRACTION) * line.page_height
    )


def _body_size(lines: Sequence[Line]) -> float:
    weights: Counter[float] = Counter()
    for line in lines:
        weights[line.size] += len(line.text)
    return max(weights.items(), key=lambda kv: kv[1])[0] if weights else 0.0


def _find_excluded(lines: Sequence[Line], n_pages: int) -> set[int]:
    excluded = {i for i, line in enumerate(lines) if _is_page_number(line.text)}
    pages_by_key: defaultdict[str, set[int]] = defaultdict(set)
    pages_by_section_key: defaultdict[str, set[int]] = defaultdict(set)
    edge: list[tuple[int, str, str]] = []
    for i, line in enumerate(lines):
        if i in excluded or not _in_edge_zone(line):
            continue
        key, section_key = _running_key(line.text), _section_running_key(line.text)
        pages_by_key[key].add(line.page)
        pages_by_section_key[section_key].add(line.page)
        edge.append((i, key, section_key))
    needed = max(_RECURRENCE_MINIMUM, math.ceil(_RECURRENCE_FRACTION * n_pages))
    excluded.update(
        i
        for i, key, section_key in edge
        if len(pages_by_key[key]) >= needed
        or len(pages_by_section_key[section_key]) >= _SECTION_RECURRENCE_MINIMUM
    )
    return excluded


def _find_printed_toc(
    lines: Sequence[Line], excluded: set[int]
) -> tuple[list[tuple[str, str]], set[int]]:
    by_page: defaultdict[int, list[int]] = defaultdict(list)
    for i, line in enumerate(lines):
        if i not in excluded:
            by_page[line.page].append(i)
    entries: list[tuple[str, str]] = []
    toc_lines: set[int] = set()
    for page in sorted(by_page):
        indexes = by_page[page]
        matches = [(i, _TOC_LINE.match(lines[i].text.strip())) for i in indexes]
        matches = [(i, m) for i, m in matches if m and any(c.isalpha() for c in m.group(1))]
        headed = _SPACE.sub(" ", lines[indexes[0]].text.casefold()).strip(" :.") in _TOC_HEADINGS
        if len(matches) >= _TOC_MIN_LINES or (headed and len(matches) >= _TOC_HEADED_MIN_LINES):
            toc_lines.update(indexes)
            entries.extend(
                (_TOC_TRAILING.sub("", m.group(1)).strip(), m.group(2)) for _, m in matches
            )
    return entries, toc_lines


def compute_signals(
    lines: Sequence[Line],
    n_pages: int,
    *,
    bookmarks: Sequence[tuple[int, str, int]] | None = None,
) -> DocSignals:
    """Derive body size, median gap, excluded lines, printed TOC and bookmarks.

    Body size is the character-weighted mode of line size over all lines. The median gap is
    taken over the non-excluded lines' positive finite ``gap_above`` values.
    """
    excluded = _find_excluded(lines, n_pages)
    gaps = [
        line.gap_above
        for i, line in enumerate(lines)
        if i not in excluded and _MIN_POSITIVE_GAP < line.gap_above < math.inf
    ]
    entries, toc_lines = _find_printed_toc(lines, excluded)
    return DocSignals(
        body_size=_body_size(lines),
        median_gap=statistics.median(gaps) if gaps else 0.0,
        excluded=excluded,
        printed_toc=entries,
        toc_page_lines=toc_lines,
        bookmarks=[(int(lvl), str(title), int(page)) for lvl, title, page in bookmarks or []],
    )


def read_bookmarks(path: str | Path) -> list[tuple[int, str, int]]:
    """PDF outline as ``(level, title, page)``; ``[]`` when pymupdf is missing or unreadable.

    Items without a target page (page < 1) are dropped."""
    try:
        pymupdf = _import_pymupdf()
    except MissingExtraError:
        return []
    try:
        doc = pymupdf.open(str(path), filetype="pdf")
    except Exception:
        return []
    try:
        if doc.needs_pass:
            return []
        return [
            (int(level), str(title).strip(), int(page))
            for level, title, page, *_ in doc.get_toc()
            if str(title).strip() and int(page) >= 1  # page -1: outline item with no target
        ]
    except Exception:
        return []
    finally:
        doc.close()
