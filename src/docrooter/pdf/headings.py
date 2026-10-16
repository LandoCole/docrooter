"""Heading detection on native PDF lines (spec §2.3): candidate pre-filter, multi-line title
merge, scoring, and level assignment."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field

from docrooter.document.headings import _DOTTED, _NUMBERING, repair_levels, score_heading
from docrooter.document.models import RawHeading
from docrooter.pdf.lines import Line
from docrooter.pdf.signals import DocSignals, is_fuzzy_match

__all__ = ["HeadingCandidate", "find_headings"]

_MAX_WORDS = 20
_SIZE_FACTOR = 1.15  # size >= 1.15 x body makes a line a candidate on its own
_ISOLATION_GAP_FACTOR = 1.2  # x median gap
_ISOLATION_SIZE_FACTOR = 0.5  # x line size; the median gap alone is ordinary leading
_MERGE_GAP_FACTOR = 1.5  # x median gap
_STYLE_TOLERANCE = 0.5  # pt
_MAX_LEVEL = 6
_NUMBERED_FRACTION = 0.5
_BOOKMARK_PAGE_SLACK = 1
_LOW_CONFIDENCE = 0.7  # same cut as v0.1 validate_headings

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
    :func:`find_headings`) of its first line; multi-line titles are joined with spaces."""

    line_index: int
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


def _is_isolated(lines: Sequence[Line], i: int, gap_floor: float) -> bool:
    """First line of its page or block, or ``gap_above >= max(1.2 x median gap, 0.5 x size)``."""
    line = lines[i]
    if i == 0 or lines[i - 1].page != line.page or lines[i - 1].block != line.block:
        return True
    return line.gap_above >= max(gap_floor, _ISOLATION_SIZE_FACTOR * line.size)


def _is_candidate(line: Line, isolated: bool, body: float) -> bool:
    text = line.text.strip()
    if len(text.split()) > _MAX_WORDS:
        return False
    if body > 0 and line.size >= _SIZE_FACTOR * body:
        return True
    return isolated and (line.bold or bool(_NUMBERING.match(text)))


def _merge_runs(
    lines: Sequence[Line], candidate: list[bool], skip: set[int], merge_gap: float
) -> list[_Run]:
    """Join each candidate with the following lines on its page that share its style and sit
    closer than ``merge_gap``. A run longer than the word limit is a paragraph and is dropped."""
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
            and lines[end + 1].page == lines[i].page
            and _same_style(lines[i], lines[end + 1])
            and lines[end + 1].gap_above < merge_gap
        ):
            end += 1
        text = " ".join(lines[k].text.strip() for k in range(i, end + 1))
        if len(text.split()) <= _MAX_WORDS:
            runs.append(_Run(i, end, text))
        i = end + 1
    return runs


def _next_line(i: int, n: int, excluded: set[int]) -> int | None:
    j = i + 1
    while j < n and j in excluded:
        j += 1
    return j if j < n else None


def _bookmark_level(text: str, page: int, bookmarks: Sequence[tuple[int, str, int]]) -> int | None:
    """Level of the nearest-page bookmark within one page whose title fuzzy-matches ``text``."""
    best: tuple[int, int] | None = None  # (page distance, level)
    for level, title, bm_page in bookmarks:
        distance = abs(bm_page - page)
        if distance <= _BOOKMARK_PAGE_SLACK and is_fuzzy_match(text, title):
            if best is None or distance < best[0]:
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
    runs = _merge_runs(lines, candidate, skip, _MERGE_GAP_FACTOR * signals.median_gap)
    repeats = Counter(_normalise(run.text) for run in runs)
    toc_titles = [title for title, _ in signals.printed_toc] if use_printed_toc else []
    bookmarks = signals.bookmarks if use_bookmarks else []

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
        if any(is_fuzzy_match(run.text, title) for title in toc_titles):
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
            line_index=run.start, text=run.text, page=line.page, size=line.size,
            bold=line.bold, score=score,
        )  # fmt: skip
        kept.append((heading, bookmark_level))
    _assign_levels(kept)
    return [heading for heading, _ in kept]


def _assign_levels(kept: list[tuple[HeadingCandidate, int | None]]) -> None:
    """Set ``level`` and level flags on the kept headings in place."""
    if not kept:
        return
    styles = sorted(
        {_style_key(h.size, h.bold) for h, _ in kept}, key=lambda key: (-key[0], not key[1])
    )
    rank = {key: index + 1 for index, key in enumerate(styles)}
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
