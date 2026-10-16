"""Heading detection on native PDF lines (spec §2.3): candidate pre-filter, multi-line title
merge, scoring, and level assignment."""

from __future__ import annotations

import math
import re
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from pydantic import BaseModel, Field

from docrooter.document.headings import _DOTTED, _NUMBERING, repair_levels, score_heading
from docrooter.document.models import RawHeading
from docrooter.pdf.lines import Line
from docrooter.pdf.signals import (
    _FUZZY_THRESHOLD,
    DocSignals,
    _edge_copy_key,
    _in_edge_zone,
    is_fuzzy_match,
    normalize_title,
)

__all__ = ["HeadingCandidate", "find_headings"]

_MAX_WORDS = 20
_SIZE_FACTOR = 1.15  # size >= 1.15 x body makes a line a candidate on its own
_ISOLATION_GAP_FACTOR = 1.2  # x median gap
_ISOLATION_SIZE_FACTOR = 0.5  # x line size; the median gap alone is ordinary leading
_MERGE_GAP_FACTOR = 1.5  # x median gap
_STYLE_TOLERANCE = 0.5  # pt
_MAX_LEVEL = 6
# A style used by fewer headings than min(3, 10% of them) is "one-off" (cover, front matter)
# and takes the level of the next recurring style instead of a level of its own (P5b).
_EDGE_COPY_PAGES = 2  # a running copy sits at most this many pages after its heading
_LEVEL_STYLE_MIN_COUNT = 3
_LEVEL_STYLE_MIN_SHARE = 0.1
_NUMBERED_FRACTION = 0.5
_BOOKMARK_PAGE_SLACK = 1
_LOW_CONFIDENCE = 0.7  # same cut as v0.1 validate_headings
_ROW_WINDOW = 4  # lines either side searched for same-row neighbours
_MIN_LETTERS = 2  # number-only lines ("12%", "21,268", "3.") are never headings
# A leading section number, uppercase roman numeral or single-letter label ("2.1", "IV.", "a)").
_NUMBER_PREFIX = re.compile(r"^\(?(?:\d+(?:\.\d+)*[.)]?|[IVXLC]+[.)]|[A-Za-z][.)])(?=\s|$)")

# Score weights (spec §2.3).
_W_TITLE = 0.4
_W_SIZE = 0.25
_SIZE_SPAN = 0.6  # size ratio 1.6 x body earns the full size weight
_W_BOLD = 0.1
_W_ISOLATED = 0.1
_W_NUMBERED = 0.1
_W_PRINTED_TOC = 0.25
_W_BOOKMARK = 0.25
_P_NO_BODY = 0.15


class HeadingCandidate(BaseModel):
    """A detected heading. ``line_index`` is the index (into the ``lines`` given to
    :func:`find_headings`) of its first line; a multi-line title spans ``line_count`` consecutive
    lines and is joined with spaces."""

    line_index: int
    line_count: int = 1
    text: str
    page: int
    size: float
    bold: bool
    score: float
    level: int | None = None
    flags: list[str] = Field(default_factory=list)


@dataclass
class _Run:
    """A candidate line plus the same-style lines merged into it (``start``..``end``)."""

    start: int
    end: int
    text: str


def _same_style(a: Line, b: Line) -> bool:
    return abs(a.size - b.size) <= _STYLE_TOLERANCE and a.bold == b.bold


def _style_key(size: float, bold: bool) -> tuple[float, bool]:
    return round(size * 2) / 2, bold


def _same_row(a: Line, b: Line) -> bool:
    """Vertical overlap of at least half the shorter line's height."""
    overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
    return overlap >= 0.5 * min(a.y1 - a.y0, b.y1 - b.y0)


def _has_text_to_the_left(lines: Sequence[Line], i: int) -> bool:
    """Whether a nearby line (reading order may interleave rows) ends left of line ``i`` on
    the same row."""
    line = lines[i]
    for j in range(max(0, i - _ROW_WINDOW), min(len(lines), i + _ROW_WINDOW + 1)):
        other = lines[j]
        if j != i and other.page == line.page and other.x1 <= line.x0 + 1.0:
            if _same_row(other, line):
                return True
    return False


def _is_isolated(lines: Sequence[Line], i: int, gap_floor: float) -> bool:
    """First line of its page or block, or ``gap_above >= max(1.2 x median gap, 0.5 x size)``.

    A line with text to its left on the same row (an inline fragment such as a link that
    pymupdf put in a block of its own) is never isolated.
    """
    line = lines[i]
    if _has_text_to_the_left(lines, i):
        return False
    if i == 0 or lines[i - 1].page != line.page:
        return True
    if lines[i - 1].block != line.block:
        return True
    return line.gap_above >= max(gap_floor, _ISOLATION_SIZE_FACTOR * line.size)


def _is_body_style(line: Line, body: float) -> bool:
    return not line.bold and abs(line.size - body) <= _STYLE_TOLERANCE


def _is_candidate(line: Line, isolated: bool, body: float) -> bool:
    text = line.text.strip()
    if len(text.split()) > _MAX_WORDS:
        return False
    if sum(c.isalpha() for c in _NUMBER_PREFIX.sub("", text, count=1)) < _MIN_LETTERS:
        return False
    if body > 0 and line.size >= _SIZE_FACTOR * body:
        return True
    return isolated and (line.bold or bool(_NUMBERING.match(text)))


def _extends_run(
    lines: Sequence[Line], first: int, last: int, nxt: int, candidate: list[bool], body: float,
    merge_gap: float,
) -> bool:  # fmt: skip
    """Whether line ``nxt`` continues the title ``first..last``: same page and style as the first
    line, never body-styled, and either a candidate itself within ``merge_gap`` or the rest of
    the last line's row (a title set apart from its number, e.g. "Item 1B." | "Unresolved…")."""
    line = lines[nxt]
    if line.page != lines[first].page or not _same_style(lines[first], line):
        return False
    if _is_body_style(line, body):
        return False
    return (candidate[nxt] and line.gap_above < merge_gap) or _same_row(lines[last], line)


def _merge_runs(
    lines: Sequence[Line], candidate: list[bool], skip: set[int], body: float, merge_gap: float
) -> list[_Run]:
    """Join each candidate with the title lines that follow it (see :func:`_extends_run`).
    A run longer than the word limit is a paragraph and is dropped."""
    runs: list[_Run] = []
    i = 0
    while i < len(lines):
        if not candidate[i]:
            i += 1
            continue
        end = i
        while (
            end + 1 < len(lines)
            and end + 1 not in skip
            and _extends_run(lines, i, end, end + 1, candidate, body, merge_gap)
        ):
            end += 1
        text = " ".join(lines[k].text.strip() for k in range(i, end + 1))
        if len(text.split()) <= _MAX_WORDS:
            runs.append(_Run(i, end, text))
        i = end + 1
    return runs


def _drop_edge_copies(lines: Sequence[Line], runs: list[_Run]) -> list[_Run]:
    """Drop runs in the top/bottom edge band whose title (page number and numbering stripped)
    repeats a run outside the band on the same page or up to two pages before: the section
    title printed as a running head or footer of a section too short (one or two pages) for its
    copies to be excluded as a running line (P5b)."""
    edge = [_in_edge_zone(lines[run.start]) for run in runs]
    inner: defaultdict[str, set[int]] = defaultdict(set)
    for run, at_edge in zip(runs, edge, strict=True):
        if not at_edge:
            inner[_edge_copy_key(run.text)].add(lines[run.start].page)
    kept = []
    for run, at_edge in zip(runs, edge, strict=True):
        page = lines[run.start].page
        pages = inner.get(_edge_copy_key(run.text), set()) if at_edge else set()
        if not any(0 <= page - other <= _EDGE_COPY_PAGES for other in pages):
            kept.append(run)
    return kept


def _next_line(i: int, n: int, excluded: set[int]) -> int | None:
    j = i + 1
    while j < n and j in excluded:
        j += 1
    return j if j < n else None


class _TocMatcher:
    """``any(is_fuzzy_match(text, title) for title in titles)``, fast for long printed TOCs.

    Titles are normalised once and an exact match short-circuits. A title is only compared
    when two upper bounds of the ``SequenceMatcher`` ratio reach the threshold: the length
    bound (``real_quick_ratio``, applied as a window over the length-sorted titles) and the
    character-multiset bound (``quick_ratio``, computed from precomputed counts). Survivors get
    the exact ``ratio()`` from a per-title matcher (title as ``b``, as in
    :func:`is_fuzzy_match`). Results are memoised by normalised text.
    """

    def __init__(self, titles: Iterable[str]) -> None:
        norms = sorted({normalize_title(t) for t in titles} - {""}, key=len)
        self._exact = set(norms)
        self._lengths = [len(norm) for norm in norms]
        self._counts = [dict(Counter(norm)) for norm in norms]
        self._matchers = [SequenceMatcher(None, "", norm) for norm in norms]
        self._memo: dict[str, bool] = {}

    def matches(self, text: str) -> bool:
        norm = normalize_title(text)
        if not norm or not self._matchers:
            return False
        if norm not in self._memo:
            self._memo[norm] = norm in self._exact or self._compare(norm)
        return self._memo[norm]

    def _compare(self, norm: str) -> bool:
        t, size = _FUZZY_THRESHOLD, len(norm)
        # real_quick_ratio = 2 * min(la, lb) / (la + lb) >= t bounds the title length.
        lo = bisect_left(self._lengths, size * t / (2 - t) - 1e-9)
        hi = bisect_right(self._lengths, size * (2 - t) / t + 1e-9)
        counts = Counter(norm).items()
        for k in range(lo, hi):
            total = size + self._lengths[k]
            if 2.0 * min(size, self._lengths[k]) / total < t:
                continue
            title_counts = self._counts[k].get
            shared = sum(min(n, title_counts(c, 0)) for c, n in counts)  # as quick_ratio
            if 2.0 * shared / total < t:
                continue
            matcher = self._matchers[k]
            matcher.set_seq1(norm)
            if matcher.ratio() >= t:
                return True
        return False


def _bookmark_level(
    text: str, page: int, by_page: dict[int, list[tuple[int, str, int]]]
) -> int | None:
    """Level of the nearest-page bookmark within one page whose title fuzzy-matches ``text``."""
    best: tuple[int, int] | None = None  # (page distance, level)
    for near in range(page - _BOOKMARK_PAGE_SLACK, page + _BOOKMARK_PAGE_SLACK + 1):
        for level, title, bm_page in by_page.get(near, ()):
            distance = abs(bm_page - page)
            if (best is None or distance < best[0]) and is_fuzzy_match(text, title):
                best = (distance, level)
    return best[1] if best else None


def _normalise(text: str) -> str:
    return " ".join(text.casefold().split())


def find_headings(
    lines: Sequence[Line],
    signals: DocSignals,
    *,
    min_score: float = 0.55,
    use_bookmarks: bool = True,
    use_printed_toc: bool = True,
) -> list[HeadingCandidate]:
    """Detect, score and level headings among ``lines`` (spec §2.3), in reading order.

    Lines in ``signals.excluded`` (running headers/footers, page numbers) and on printed TOC
    pages are never headings. Candidates scoring at least ``min_score`` are kept. Levels come
    from typography, then dotted numbering (when at least half the kept headings carry it), then
    a matching bookmark (flag ``level_from_bookmark``), and gaps are closed with ``repair_levels``.
    """
    n = len(lines)
    body = signals.body_size
    excluded = signals.excluded
    skip = excluded | signals.toc_page_lines
    gap_floor = _ISOLATION_GAP_FACTOR * signals.median_gap
    isolated = [_is_isolated(lines, i, gap_floor) for i in range(n)]
    candidate = [i not in skip and _is_candidate(lines[i], isolated[i], body) for i in range(n)]
    runs = _drop_edge_copies(
        lines, _merge_runs(lines, candidate, skip, body, _MERGE_GAP_FACTOR * signals.median_gap)
    )
    repeats = Counter(_normalise(run.text) for run in runs)
    toc = _TocMatcher(title for title, _ in signals.printed_toc) if use_printed_toc else None
    bookmarks: defaultdict[int, list[tuple[int, str, int]]] = defaultdict(list)
    for bookmark in signals.bookmarks if use_bookmarks else []:
        bookmarks[bookmark[2]].append(bookmark)

    kept: list[tuple[HeadingCandidate, int | None]] = []
    for run in runs:
        line = lines[run.start]
        title_score, reasons = score_heading(run.text, repeat_count=repeats[_normalise(run.text)])
        score = _W_TITLE * title_score
        if body > 0:
            score += _W_SIZE * max(0.0, min((line.size / body - 1) / _SIZE_SPAN, 1.0))
        score += _W_BOLD * line.bold
        score += _W_ISOLATED * isolated[run.start]
        score += _W_NUMBERED * ("numbering_prefix" in reasons)
        if toc is not None and toc.matches(run.text):
            score += _W_PRINTED_TOC
        bookmark_level = _bookmark_level(run.text, line.page, bookmarks)
        if bookmark_level is not None:
            score += _W_BOOKMARK
        nxt = _next_line(run.end, n, excluded)
        if nxt is not None and candidate[nxt] and _same_style(line, lines[nxt]):
            score -= _P_NO_BODY  # a run of same-style candidates is a list or table
        score = round(min(1.0, max(0.0, score)), 4)
        if score < min_score:
            continue
        heading = HeadingCandidate(
            line_index=run.start, line_count=run.end - run.start + 1, text=run.text,
            page=line.page, size=line.size, bold=line.bold, score=score,
        )  # fmt: skip
        kept.append((heading, bookmark_level))
    _assign_levels(kept)
    return [heading for heading, _ in kept]


def _style_levels(keys: Sequence[tuple[float, bool]]) -> dict[tuple[float, bool], int]:
    """Level of each heading style: recurring styles are ranked by size (bold first at equal
    size); a one-off style shares the level of the next recurring style below it, or sits one
    level below the last recurring style."""
    counts = Counter(keys)
    needed = min(_LEVEL_STYLE_MIN_COUNT, math.ceil(_LEVEL_STYLE_MIN_SHARE * len(keys)))
    ordered = sorted(counts, key=lambda key: (-key[0], not key[1]))
    recurring = [key for key in ordered if counts[key] >= needed] or ordered
    level = {key: index + 1 for index, key in enumerate(recurring)}
    below = len(recurring) + 1
    for key in reversed(ordered):
        if key in level:
            below = level[key]
        else:
            level[key] = below
    return level


def _assign_levels(kept: list[tuple[HeadingCandidate, int | None]]) -> None:
    """Set ``level`` and level flags on the kept headings in place."""
    if not kept:
        return
    rank = _style_levels([_style_key(h.size, h.bold) for h, _ in kept])
    dotted = [_DOTTED.match(h.text.strip()) for h, _ in kept]
    use_numbering = sum(1 for m in dotted if m) / len(kept) >= _NUMBERED_FRACTION
    for (heading, bookmark_level), match in zip(kept, dotted, strict=True):
        level = rank[_style_key(heading.size, heading.bold)]
        if use_numbering and match:
            level = len(match.group(1).split("."))
        if bookmark_level is not None:
            level = bookmark_level
            heading.flags.append("level_from_bookmark")
        heading.level = max(1, min(level, _MAX_LEVEL))
        if heading.score < _LOW_CONFIDENCE:
            heading.flags.append("low_heading_confidence")
    # Titles are left empty so repair_levels only closes gaps; numbering was applied above
    # and must not override a bookmark level.
    repaired = repair_levels(
        [RawHeading(title="", level=h.level or 1, text="", line=h.line_index) for h, _ in kept]
    )
    for (heading, _), fixed in zip(kept, repaired, strict=True):
        heading.level = fixed.level
        heading.flags.extend(flag for flag in fixed.flags if flag not in heading.flags)
