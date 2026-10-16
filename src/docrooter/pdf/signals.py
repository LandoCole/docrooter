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
# minus a leading/trailing page number, so "Item 1." and "Item 2." stay distinct (P5b). It must
# recur on 3 pages; a heading-like line (bold, or >= 1.15 x body, e.g. a repeated slide title)
# counts only when it sits at the same height (+/- 2 pt) on 3 pages, as running lines do. No
# share-of-pages floor: a datasheet's sections span 3-10 of 642 pages, and max(3, 10% of
# pages) cut TOC micro F1 from 0.702 to 0.580 (P5b fix round 1).
_SECTION_RECURRENCE_MINIMUM = 3
_SECTION_SAME_Y_TOLERANCE = 2.0  # pt
_SIZE_FACTOR = 1.15  # size >= 1.15 x body makes a line a heading candidate on its own
_TOC_MIN_LINES = 5
_TOC_HEADED_MIN_LINES = 3
# Contents pages that the leader regex misses (P5b): the page number is a separate line on the
# title's row, the line ends in leaders without a number, or (RFC style) the page is a list of
# short section-numbered lines with no page numbers at all. Tables and code listings look alike,
# so same-row page numbers must run in order (>= 80% of consecutive pairs), be mostly distinct
# and not exceed the page count, and RFC-style lists must be mostly dotted ("2.1.") numbers.
_TOC_ROW_WINDOW = 3  # following lines searched for a same-row page number
_TOC_ORDERED_FRACTION = 0.8
_TOC_DISTINCT_FRACTION = 0.5
_TOC_DOTTED_FRACTION = 0.5
_TOC_NUMBERED_MIN_LINES = 8
_TOC_NUMBERED_PAGE_SHARE = 0.6
_TOC_NUMBERED_MAX_WORDS = 12
_MIN_POSITIVE_GAP = 0.5  # gaps at or below this (pt) are line-box overlap, not spacing

_PAGE_NUMBER = re.compile(
    r"^(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$|^[-–—]\s*\d{1,4}\s*[-–—]$|^[ivxlcdm]{1,7}$",
    re.IGNORECASE,
)
# The roman-numeral branch above also matches words like "civil"; require a well-formed numeral.
_ROMAN = re.compile(r"^m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$", re.IGNORECASE)
_TOC_LINE = re.compile(r"^(.*?)[\s.·…_-]{2,}(\d{1,4})$")
_TOC_TRAILING = re.compile(r"[\s.·…_-]+$")
_TOC_LEADERS = re.compile(r"^(.*?[^\W\d_].*?)\s*(?:[.·…_]\s?){5,}$")  # not an ellipsis
_TOC_NUMBERED = re.compile(r"^(\d{1,3}(?:\.\d{1,3})*)\.?\s+\S")
_TOC_HEADINGS = frozenset({"contents", "table of contents"})
_EDGE_LEADING_NUMBER = re.compile(r"^\d{1,4}\s+")
_EDGE_TRAILING_NUMBER = re.compile(r"\s+\d{1,4}$")
# A trailing number after one of these words is the title's own number ("Chapter 2"), not a
# page number.
_NUMBERING_WORDS = r"chapter|part|section|item|appendix|article|schedule|exhibit|annex|table|figure"
_NUMBERING_WORD = re.compile(rf"\b(?:{_NUMBERING_WORDS})$", re.IGNORECASE)
_NUMBERED_OR_DIGITS = re.compile(rf"(\b(?:{_NUMBERING_WORDS})\s+)?\d+", re.IGNORECASE)
# "Page 5", "Page 5 of 80", "5 of 80", "5/80": always a page number, whatever precedes it.
_PAGE_EXPRESSION = re.compile(
    r"\bpage\s*\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?\b|\b\d{1,4}\s*(?:of|/)\s*\d{1,4}\b",
    re.IGNORECASE,
)
_DIGIT_RUN = re.compile(r"\d+")
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
    """Text with numbers generalised to ``#`` for the document-wide running-line test. A line
    that carries a page number ("Page 5 of 80", or a bare leading/trailing number) is a running
    line candidate and has every number generalised ("Section # · Page # of #"); otherwise a
    number right after a numbering word ("Chapter 2", "Item 1A") is the title's own and keeps
    its digits, so chapter openers stay distinct."""
    text = _SPACE.sub(" ", text.casefold()).strip()
    if _strip_page_number(text) != text:
        return _DIGIT_RUN.sub("#", text)
    return _NUMBERED_OR_DIGITS.sub(_mask_digits, text)


def _mask_digits(match: re.Match[str]) -> str:
    return match.group(0) if match.group(1) else "#"


def _strip_page_number(text: str) -> str:
    """``text`` (whitespace collapsed) without its page number: a page expression ("Page 5",
    "Page 5 of 80", "5/80") anywhere, then a bare leading or trailing number. A trailing number
    that follows a numbering word ("Chapter 2", "Item 3") belongs to the title and stays."""
    text = _SPACE.sub(" ", _PAGE_EXPRESSION.sub("", _SPACE.sub(" ", text).strip())).strip()
    text = _EDGE_LEADING_NUMBER.sub("", text)
    stripped = _EDGE_TRAILING_NUMBER.sub("", text)
    return text if _NUMBERING_WORD.search(stripped) else stripped


def _section_running_key(text: str) -> str:
    return _strip_page_number(text.casefold())


def _edge_copy_key(text: str) -> str:
    """Title of a possible running copy of a heading: page number and numbering stripped."""
    return normalize_title(_strip_page_number(text))


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
    copies_by_section_key: defaultdict[str, list[Line]] = defaultdict(list)
    edge: list[tuple[int, str, str]] = []
    for i, line in enumerate(lines):
        if i in excluded or not _in_edge_zone(line):
            continue
        key, section_key = _running_key(line.text), _section_running_key(line.text)
        pages_by_key[key].add(line.page)
        copies_by_section_key[section_key].append(line)
        edge.append((i, key, section_key))
    needed = max(_RECURRENCE_MINIMUM, math.ceil(_RECURRENCE_FRACTION * n_pages))
    body = _body_size(lines)
    excluded.update(
        i
        for i, key, section_key in edge
        if len(pages_by_key[key]) >= needed
        or _is_section_running_line(lines[i], copies_by_section_key[section_key], body)
    )
    return excluded


def _is_section_running_line(line: Line, copies: Sequence[Line], body: float) -> bool:
    """Whether an edge-band line whose text recurs as ``copies`` is a section running line: on
    at least 3 pages, and either not heading-like (neither bold nor >= 1.15 x body) or at the
    same height (+/- 2 pt) on at least 3 pages, as running lines are (P5b)."""
    if len({copy.page for copy in copies}) < _SECTION_RECURRENCE_MINIMUM:
        return False
    if not (line.bold or (body > 0 and line.size >= _SIZE_FACTOR * body)):
        return True
    same_y = {copy.page for copy in copies if abs(copy.y0 - line.y0) <= _SECTION_SAME_Y_TOLERANCE}
    return len(same_y) >= _SECTION_RECURRENCE_MINIMUM


def _same_row(a: Line, b: Line) -> bool:
    overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
    return overlap >= 0.5 * min(a.y1 - a.y0, b.y1 - b.y0)


def _row_page_number(lines: Sequence[Line], i: int) -> str | None:
    """A page-number line to the right of line ``i`` on its row (a contents entry whose number
    is a separate text object), if any."""
    line = lines[i]
    for j in range(i + 1, min(len(lines), i + 1 + _TOC_ROW_WINDOW)):
        other = lines[j]
        if other.page != line.page:
            break
        if other.x0 >= line.x1 - 1.0 and _same_row(line, other) and _is_page_number(other.text):
            label = other.text.strip()
            return label if label.isdigit() else None
    return None


def _mostly_ordered(values: Sequence[tuple[int, ...]]) -> bool:
    pairs = list(zip(values, values[1:], strict=False))
    if not pairs:
        return True
    return sum(a <= b for a, b in pairs) >= _TOC_ORDERED_FRACTION * len(pairs)


def _toc_entries(
    lines: Sequence[Line], indexes: Sequence[int], n_pages: int
) -> dict[int, tuple[str, str]]:
    """Contents entries on one page: leader lines with a trailing page number, a separate
    same-row page number (kept only when those numbers run in order), or bare leaders."""
    entries: dict[int, tuple[str, str]] = {}
    row_entries: dict[int, tuple[str, str]] = {}
    for i in indexes:
        text = lines[i].text.strip()
        match = _TOC_LINE.match(text)
        if match and any(c.isalpha() for c in match.group(1)):
            entries[i] = (_TOC_TRAILING.sub("", match.group(1)).strip(), match.group(2))
        elif (label := _row_page_number(lines, i)) is not None and any(c.isalpha() for c in text):
            row_entries[i] = (_TOC_TRAILING.sub("", text).strip(), label)
        elif leaders := _TOC_LEADERS.match(text):
            entries[i] = (leaders.group(1).strip(), "")
    labels = [int(label) for _, label in row_entries.values()]
    if (
        _mostly_ordered([(label,) for label in labels])
        and len(set(labels)) >= max(_TOC_MIN_LINES, _TOC_DISTINCT_FRACTION * len(labels))
        and max(labels, default=0) <= n_pages
    ):
        entries.update(row_entries)
    return dict(sorted(entries.items()))


def _numbered_contents(lines: Sequence[Line], indexes: Sequence[int]) -> list[int]:
    """Numbered entries of an RFC-style contents page (short section-numbered lines in order
    making up most of the page), or ``[]``."""
    numbered = [
        (i, m)
        for i in indexes
        if (m := _TOC_NUMBERED.match(lines[i].text.strip()))
        and len(lines[i].text.split()) <= _TOC_NUMBERED_MAX_WORDS
    ]
    numbers = [tuple(map(int, m.group(1).split("."))) for _, m in numbered]
    if (
        len(numbered) < _TOC_NUMBERED_MIN_LINES
        or len(numbered) < _TOC_NUMBERED_PAGE_SHARE * len(indexes)
        or sum(len(number) > 1 for number in numbers) < _TOC_DOTTED_FRACTION * len(numbers)
        or not _mostly_ordered(numbers)
    ):
        return []
    return [i for i, _ in numbered]


def _find_printed_toc(
    lines: Sequence[Line], excluded: set[int], n_pages: int
) -> tuple[list[tuple[str, str]], set[int]]:
    by_page: defaultdict[int, list[int]] = defaultdict(list)
    for i, line in enumerate(lines):
        if i not in excluded:
            by_page[line.page].append(i)
    entries: list[tuple[str, str]] = []
    toc_lines: set[int] = set()
    for page in sorted(by_page):
        indexes = by_page[page]
        found = _toc_entries(lines, indexes, n_pages)
        headed = _is_contents_heading(lines[indexes[0]].text)
        if len(found) >= _TOC_MIN_LINES or (headed and len(found) >= _TOC_HEADED_MIN_LINES):
            toc_lines.update(i for i in indexes if not _is_contents_heading(lines[i].text))
            entries.extend(found.values())
        elif numbered := _numbered_contents(lines, indexes):
            end = _contents_end(lines, indexes, numbered)
            toc_lines.update(
                i for i in indexes if i < end and not _is_contents_heading(lines[i].text)
            )
            entries.extend((lines[i].text.strip(), "") for i in numbered if i < end)
    return entries, toc_lines


def _contents_end(lines: Sequence[Line], indexes: Sequence[int], numbered: Sequence[int]) -> int:
    """Index where a numbered contents list stops on its page: the first entry numbered lower
    than an earlier one (the document's own "1. Introduction" after the last contents entry),
    or past the page's last line."""
    highest: tuple[int, ...] = ()
    for i in numbered:
        match = _TOC_NUMBERED.match(lines[i].text.strip())
        number = tuple(map(int, match.group(1).split("."))) if match else ()
        if number < highest:
            return i
        highest = max(highest, number)
    return indexes[-1] + 1


def _is_contents_heading(text: str) -> bool:
    """ "Contents" / "Table of Contents": the TOC page's own heading, which may be a heading."""
    return _SPACE.sub(" ", text.casefold()).strip(" :.") in _TOC_HEADINGS


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
    entries, toc_lines = _find_printed_toc(lines, excluded, n_pages)
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
