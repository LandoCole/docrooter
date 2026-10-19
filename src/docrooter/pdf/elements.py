"""Element typing rules (spec §5.3): document lines and page objects become typed elements.

The rules run in spec order and each claims the lines it consumes; a line is never claimed
twice, and the paragraph rule takes every line still free. Elements come out in reading order
with ids ``e0``, ``e1``, … and ``section_id=None`` (sections are assigned by the caller).

All line indexes (``order``, ``signals``, ``headings``, ``title_index``, ``line_source``) refer
to ``lines``; ``pages`` supply the images, vector regions, ruled tables and page sizes.

Additions to the spec rules (task E5a rulings, tunable in E11a like the spec's thresholds):
- an excluded page-number-like line between body lines and outside the edge band is not
  furniture (``compute_signals`` marks every bare number, table cells included);
- content is never lost to a figure (controller ruling): an image or vector region is not a
  figure when the text inside it holds a heading or the title, forms a table (statement rows
  or aligned columns), reads as prose, is a list or a code listing, or covers at least
  ``FIGURE_MAX_TEXT_COVERAGE`` of the box, nor when the box covers at least
  ``FIGURE_BACKGROUND_PAGE_SHARE`` of the page with any text on it (a background or frame);
  its lines are left to the later rules. A figure does not take a caption set inside its
  region (the caption rule links it) or the end of a paragraph running into its edge, and
  its ``text`` is its label text (spec §4), as is ``FigureData.text``;
- a roman-numeral caption number must not run into a word ("Figure illustrates");
- Greek letters count as math symbols only outside words, so Greek prose is not a formula;
- ``Line`` has no superscript flag, so a footnote is recognised by its leading marker token
  (digits, ``*``, ``†``, ``‡``, ``§``) with the size and position rules;
- a ruled table whose engine cells miss letters or digits of the lines it consumes keeps its
  cells in ``TableData`` but takes its lines' text as ``text`` and the flag
  ``cell_text_incomplete`` (task E5b: element mode never loses text-layer content).

With layout-model ``regions`` (spec §6, task E6), rules 1 and 2 run first; then every region
of at least 0.5 confidence takes the free lines whose centres fall inside it (see
:func:`docrooter.layout.base.apply_regions`) as one element of its kind, flagged
``from_layout_model`` with the region's confidence, and rules 3 to 10 type the lines left over.
A Table region takes the ruled (or region "text"-strategy) table of ``pages`` it overlaps by at
least half, else cells from its lines' rows; a Picture region is a figure
(``source="layout_model"``) even without text; a Section-header region is a heading at the
level of the heading candidate among its lines (the caller adds candidates for regions without
one; with none it stays a paragraph, so its text is never lost); a Title region without a
candidate is a ``title`` with no level; a Page-footer or Page-header region of bare page
numbers is ``page_number``. On page 1, the first Section-header region holding the detected
title line (or, with no title detected, set in the largest type on page 1) is the ``title``.
Caption regions are linked to a figure or table as rule 5 links.
Ruled tables and image or vector figures holding lines that a region took are left to the model.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from docrooter.elements.models import Box, Element, ElementKind, FigureData, TableData
from docrooter.layout.base import MIN_CONFIDENCE, LayoutRegion, apply_regions, region_kind
from docrooter.pdf.element_rules import (
    CAPTION_RE,
    MIN_GAP,
    Rect,
    area,
    caption_match,
    clip_box,
    ends_sentence,
    footnote_marker,
    h_overlap,
    has_math_content,
    inside_share,
    is_code,
    is_equation_number,
    is_math_font,
    is_prose,
    is_text_grid,
    join_text,
    line_rect,
    list_marker,
    overlap_area,
    same_row,
    statement_rows,
    symbol_share,
    text_box,
    union,
)
from docrooter.pdf.headings import HeadingCandidate
from docrooter.pdf.lines import Line
from docrooter.pdf.objects import PageObjects
from docrooter.pdf.signals import DocSignals, _in_edge_zone, _is_page_number

__all__ = [
    "CAPTION_FIRST_SIDE",
    "CAPTION_MAX_DISTANCE_LINES",
    "CAPTION_MAX_LINES",
    "CAPTION_RE",
    "FIGURE_BACKGROUND_PAGE_SHARE",
    "FIGURE_CODE_LINE_SHARE",
    "FIGURE_GRID_ALIGN_TOLERANCE",
    "FIGURE_GRID_LINE_SHARE",
    "FIGURE_GRID_MIN_COLUMNS",
    "FIGURE_GRID_MIN_LABELLED_ROWS",
    "FIGURE_GRID_MIN_ROWS",
    "FIGURE_LIST_MIN_ITEMS",
    "FIGURE_MAX_TEXT_COVERAGE",
    "FIGURE_MAX_TEXT_DENSITY",
    "FIGURE_MIN_AREA",
    "FIGURE_PROSE_LINE_SHARE",
    "FIGURE_PROSE_MIN_SENTENCES",
    "FIGURE_PROSE_MIN_WORDS",
    "FIGURE_PROSE_RUN_LINES",
    "FIGURE_PROSE_SENTENCE_SHARE",
    "FIRST_LINE_INDENT",
    "FOOTNOTE_BOTTOM_SHARE",
    "FOOTNOTE_MAX_SIZE_RATIO",
    "FORMULA_MIN_SYMBOL_SHARE",
    "FRAME_MIN_LINES_PER_CELL",
    "LIST_INDENT_STEP",
    "PARAGRAPH_GAP_RATIO",
    "PARAGRAPH_SIZE_DELTA",
    "build_elements",
    "build_elements_with_lines",
    "furniture_lines",
]

# Spec §5.3 starting estimates, tuned on 400 DocLayNet train pages in E11a (the tuning log, with
# the starting values, is in docs/benchmarks-elements.md).
FIGURE_MIN_AREA = 0.02  # share of the page area
FIGURE_MAX_TEXT_DENSITY = 0.5  # text lines per 1,000 pt² inside a vector region
CAPTION_MAX_DISTANCE_LINES = 5.0  # caption line heights between a caption and its target
FOOTNOTE_BOTTOM_SHARE = 0.3  # bottom share of the page where footnotes sit
FOOTNOTE_MAX_SIZE_RATIO = 0.85  # x body size
LIST_INDENT_STEP = 8.0  # pt per list depth
FORMULA_MIN_SYMBOL_SHARE = 0.30
PARAGRAPH_GAP_RATIO = 1.75  # x the region's median line gap
PARAGRAPH_SIZE_DELTA = 0.5  # pt
FIRST_LINE_INDENT = 8.0  # pt
# Task E5a additions (see the module docstring).
FIGURE_MAX_TEXT_COVERAGE = 0.3  # share of a figure box covered by text lines
FIGURE_BACKGROUND_PAGE_SHARE = 0.5  # an image or vector region this large with text on it
# is a page background or frame
FIGURE_GRID_MIN_ROWS = 3  # rows of two or more lines sharing a column edge ...
FIGURE_GRID_MIN_COLUMNS = 2  # ... in this many columns make the text a table, not labels
FIGURE_GRID_ALIGN_TOLERANCE = 3.0  # pt between edges of one column
FIGURE_PROSE_MIN_WORDS = 5  # a line this long is running text, not a label ...
FIGURE_PROSE_LINE_SHARE = 0.5  # ... and this share of such lines makes the text prose,
FIGURE_PROSE_MIN_SENTENCES = 2  # as do this many such lines ending a sentence ...
FIGURE_PROSE_SENTENCE_SHARE = 0.25  # ... when they are also this share of the lines,
FIGURE_PROSE_RUN_LINES = 3  # as do this many such lines stacked as a paragraph
FIGURE_GRID_MIN_LABELLED_ROWS = 2  # rows of a word label then numbers: a statement table
FIGURE_GRID_LINE_SHARE = 0.7  # share of lines in worded multi-cell rows a table needs
FIGURE_CODE_LINE_SHARE = 0.5  # this share of monospace lines inside: a code listing
FIGURE_LIST_MIN_ITEMS = 2  # this many list items inside: a list set on a shape
FRAME_MIN_LINES_PER_CELL = 8.0  # a page-sized ruled "table" of prose is a frame when its
# cells hold this many lines each on average (final review; dev: frames 8-195, tables 1-7)
CAPTION_MAX_LINES = 6  # a caption is a line or a short paragraph
CAPTION_FIRST_SIDE = {"table": "above", "figure": "below"}  # preferred when both sides fit

_MIN_REGION_GAPS = 5  # a page with fewer gaps uses the document's median gap
_EDGE_TOLERANCE = 1.0  # pt
_HANGING_INDENT = 1.0  # pt; a list continuation line starts right of its marker
_FLOATS = frozenset({"figure", "table", "caption", "footnote", "page_header", "page_footer",
                     "page_number"})  # fmt: skip


@dataclass(eq=False)
class _Pending:
    """An element under construction: its kind, the lines it consumed and its place in order."""

    kind: ElementKind
    lines: list[int]
    page: int
    anchor: float  # position in the reading order (fractional for geometric anchors)
    boxes: list[Box]
    text: str = ""
    level: int | None = None
    list_depth: int | None = None
    marker: str | None = None
    table: TableData | None = None
    figure: FigureData | None = None
    caption_for: _Pending | None = None
    caption: _Pending | None = None
    flags: list[str] = field(default_factory=list)
    confidence: float | None = None


def build_elements(
    pages: Sequence[PageObjects],
    lines: Sequence[Line],
    order: Sequence[int],
    signals: DocSignals,
    headings: Sequence[HeadingCandidate],
    *,
    title_index: int | None,
    line_source: Sequence[Literal["text", "ocr"]] | None = None,
    regions: Sequence[LayoutRegion] = (),
    region_min_confidence: float = MIN_CONFIDENCE,
) -> list[Element]:
    """Type ``lines`` and the page objects into elements, in reading order (spec §5.3).

    ``order`` is a permutation of ``range(len(lines))`` (``order_lines``); ``title_index`` is
    the document title's line, which becomes the ``title`` element when it is on page 1;
    ``line_source`` marks OCR lines, whose elements get ``source="ocr"``; ``regions`` are a
    layout model's regions (see the module docstring).
    """
    pairs = build_elements_with_lines(
        pages, lines, order, signals, headings, title_index=title_index, line_source=line_source,
        regions=regions, region_min_confidence=region_min_confidence,
    )  # fmt: skip
    return [element for element, _ in pairs]


def build_elements_with_lines(
    pages: Sequence[PageObjects],
    lines: Sequence[Line],
    order: Sequence[int],
    signals: DocSignals,
    headings: Sequence[HeadingCandidate],
    *,
    title_index: int | None,
    line_source: Sequence[Literal["text", "ocr"]] | None = None,
    regions: Sequence[LayoutRegion] = (),
    region_min_confidence: float = MIN_CONFIDENCE,
) -> list[tuple[Element, list[int]]]:
    """:func:`build_elements`, with the indexes of the lines each element consumed (in
    reading order). Every line with text is consumed by exactly one element."""
    if sorted(order) != list(range(len(lines))):
        raise ValueError("order must be a permutation of range(len(lines))")
    if line_source is not None and len(line_source) != len(lines):
        raise ValueError("line_source must have one entry per line")
    builder = _Builder(pages, lines, order, signals, line_source)
    builder.furniture()
    builder.contents()
    if regions:
        builder.layout_regions(regions, headings, title_index, region_min_confidence)
    protected = _protected(headings, title_index, len(lines))
    builder.tables(protected)
    builder.figures(protected)
    builder.link_layout_captions()
    builder.captions()
    builder.headings(headings, title_index)
    builder.footnotes()
    builder.lists()
    builder.formulas()
    builder.paragraphs()
    return builder.finish()


def furniture_lines(lines: Sequence[Line], excluded: set[int]) -> set[int]:
    """The lines of ``excluded`` (``DocSignals.excluded``) that are page furniture. A bare
    number outside the edge band whose centre lies between the page's body lines is not: the
    signals mark every bare-number line, table cells included. Reading order
    (``order_lines``) and the furniture rule both use this set."""
    extent: dict[int, tuple[float, float]] = {}
    for i, line in enumerate(lines):
        if i in excluded or not line.text.strip():
            continue
        centre = (line.y0 + line.y1) / 2
        top, bottom = extent.get(line.page, (centre, centre))
        extent[line.page] = (min(top, centre), max(bottom, centre))
    out: set[int] = set()
    for i in excluded:
        if not 0 <= i < len(lines):
            continue
        line = lines[i]
        if _is_page_number(line.text) and not _in_edge_zone(line):
            centre = (line.y0 + line.y1) / 2
            top, bottom = extent.get(line.page, (centre, centre))
            if top < centre < bottom:
                continue  # a bare number among body lines (a table cell), not furniture
        out.add(i)
    return out


def _protected(headings: Sequence[HeadingCandidate], title_index: int | None, n: int) -> set[int]:
    """The lines of the heading candidates and the title line, which no figure may take."""
    lines = {k for h in headings for k in range(h.line_index, h.line_index + max(1, h.line_count))}
    if title_index is not None:
        lines.add(title_index)
    return {k for k in lines if 0 <= k < n}


# --- the builder ---------------------------------------------------------------------------


class _Builder:
    def __init__(
        self,
        pages: Sequence[PageObjects],
        lines: Sequence[Line],
        order: Sequence[int],
        signals: DocSignals,
        line_source: Sequence[str] | None,
    ) -> None:
        self.lines = lines
        self.order = list(order)
        self.pos = {i: k for k, i in enumerate(self.order)}
        self.signals = signals
        self.source = line_source
        self.pages = {p.page: p for p in pages}
        self.owner: dict[int, _Pending] = {}
        self.pending: list[_Pending] = []
        self.objects: defaultdict[int, list[_Pending]] = defaultdict(list)  # figures and tables
        self.on_page: defaultdict[int, list[int]] = defaultdict(list)  # in reading order
        for i in self.order:
            if lines[i].text.strip():
                self.on_page[lines[i].page].append(i)
        self.gap = {page: self._region_gap(idxs) for page, idxs in self.on_page.items()}
        self.layout_lines: set[int] = set()  # lines a layout region took
        self.layout_rects: defaultdict[int, list[Rect]] = defaultdict(list)  # figures, tables
        self.layout_captions: list[_Pending] = []
        self.used_tables: set[tuple[int, int]] = set()  # (page, index in PageObjects.tables)
        self.frames: set[tuple[int, int]] = set()  # ruled tables that are page frames

    # -- helpers ----------------------------------------------------------------------------
    def free(self, i: int) -> bool:
        return i not in self.owner and bool(self.lines[i].text.strip())

    def _size(self, page: int) -> tuple[float, float]:
        obj = self.pages.get(page)
        if obj is not None:
            return obj.width, obj.height
        line = next((self.lines[i] for i in self.on_page.get(page, [])), None)
        return (line.page_width, line.page_height) if line is not None else (0.0, 0.0)

    def _box(self, page: int, r: Rect) -> Box:
        return text_box(page, r, *self._size(page))

    def _clip(self, box: Box) -> Rect | None:
        return clip_box(box, *self._size(box.page))

    def _region_gap(self, idxs: Sequence[int]) -> float:
        """Median vertical gap between consecutive lines of a page in reading order (lines
        below and overlapping the previous one); the document's median with too few."""
        gaps = []
        excluded = self.signals.excluded
        body = [i for i in idxs if i not in excluded]
        for a, b in zip(body, body[1:], strict=False):
            la, lb = self.lines[a], self.lines[b]
            if lb.y0 >= la.y0 and not same_row(la, lb) and h_overlap(line_rect(la), line_rect(lb)):
                gaps.append(max(0.0, lb.y0 - la.y1))
        if len(gaps) >= _MIN_REGION_GAPS:
            return statistics.median(gaps)
        return self.signals.median_gap

    def _gap_limit(self, page: int) -> float:
        return PARAGRAPH_GAP_RATIO * max(self.gap.get(page, 0.0), MIN_GAP)

    def add(
        self,
        kind: ElementKind,
        idxs: Sequence[int],
        *,
        text: str | None = None,
        boxes: list[Box] | None = None,
        anchor: float | None = None,
        page: int | None = None,
        **fields: object,
    ) -> _Pending:
        """Claim ``idxs`` (sorted into reading order) for a new element. By default its text is
        the lines joined, it has one box per page around its lines, and it sits at its first
        line's reading position."""
        idxs = sorted(idxs, key=self.pos.__getitem__)
        if boxes is None:
            by_page: dict[int, list[Rect]] = {}
            for i in idxs:
                by_page.setdefault(self.lines[i].page, []).append(line_rect(self.lines[i]))
            boxes = [self._box(p, union(rects)) for p, rects in by_page.items()]
        item = _Pending(
            kind=kind,
            lines=idxs,
            page=boxes[0].page if page is None else page,
            anchor=float(self.pos[idxs[0]]) if anchor is None else anchor,
            boxes=boxes,
            text=join_text([self.lines[i].text for i in idxs]) if text is None else text,
            **fields,  # type: ignore[arg-type]
        )
        for i in idxs:
            self.owner[i] = item
        self.pending.append(item)
        if kind in ("figure", "table"):
            self.objects[item.page].append(item)
        return item

    def _geometric_anchor(self, page: int, r: Rect) -> float:
        """Reading position of a box: before the first line (in reading order) at or below its
        top in its columns, else after the last line above its bottom, else the page end."""
        idxs = self.on_page.get(page, [])
        for i in idxs:
            line = self.lines[i]
            if line.y0 >= r[1] - _EDGE_TOLERANCE and h_overlap(line_rect(line), r):
                return self.pos[i] - 0.5
        above = [i for i in idxs if self.lines[i].y1 <= r[3] + _EDGE_TOLERANCE
                 and h_overlap(line_rect(self.lines[i]), r)]  # fmt: skip
        if above:
            return self.pos[above[-1]] + 0.5
        body = [i for i in idxs if i not in self.signals.excluded]
        return self.pos[body[-1]] + 0.5 if body else (self.pos[idxs[0]] - 0.5 if idxs else 0.0)

    def _inside(self, page: int, r: Rect) -> list[int]:
        idxs = self.on_page.get(page, [])
        return [
            i for i in idxs if self.free(i) and inside_share(line_rect(self.lines[i]), r) >= 0.5
        ]

    # -- rule 1: furniture ------------------------------------------------------------------
    def furniture(self) -> None:
        furniture = furniture_lines(self.lines, self.signals.excluded)
        for i in self.order:
            if i not in furniture or not self.free(i):
                continue
            line = self.lines[i]
            kind: ElementKind
            if _is_page_number(line.text):
                kind = "page_number"
            else:
                centre = (line.y0 + line.y1) / 2
                kind = "page_header" if centre < line.page_height / 2 else "page_footer"
            self.add(kind, [i])

    # -- rule 2: printed contents pages -----------------------------------------------------
    def contents(self) -> None:
        by_page: defaultdict[int, list[int]] = defaultdict(list)
        for i in self.order:
            if i in self.signals.toc_body_lines and self.free(i):
                by_page[self.lines[i].page].append(i)
        for idxs in by_page.values():
            self.add(
                "contents", idxs, text="\n".join(join_text([self.lines[i].text]) for i in idxs)
            )

    # -- rule 3: ruled tables ---------------------------------------------------------------
    def tables(self, protected: set[int]) -> None:
        """Ruled tables, each with the free lines inside it. A table that owns no line is not
        emitted (its cells would repeat text other elements hold), and neither is a page
        frame (see :meth:`_is_frame`), whose lines stay free for the later rules."""
        for obj in self.pages.values():
            for k, (box, rows) in enumerate(obj.tables):
                if (obj.page, k) in self.used_tables:
                    continue  # a layout Table region took it
                r = self._clip(box)
                if r is None or self._holds_layout_lines(obj.page, r):
                    continue  # the layout model typed its lines
                idxs = self._inside(obj.page, r)
                if not idxs:
                    continue  # its text is another element's (an enclosing table, contents)
                if self._is_frame(obj, r, rows, idxs, protected):
                    self.frames.add((obj.page, k))
                    continue
                text, flags = self._table_text(rows, idxs)
                if not text:
                    continue
                self.add("table", idxs, text=text, page=obj.page,
                         boxes=[self._box(obj.page, r)],
                         anchor=self._geometric_anchor(obj.page, r),
                         table=TableData(rows=[list(row) for row in rows]),
                         flags=flags)  # fmt: skip

    def _is_frame(
        self, obj: PageObjects, r: Rect, rows: list[list[str]], idxs: list[int], protected: set[int]
    ) -> bool:
        """Whether a ruled table is a frame drawn around page content rather than a table: it
        holds a heading candidate or the title, or it covers at least
        ``FIGURE_BACKGROUND_PAGE_SHARE`` of the page, its lines read as prose (the figure
        rule's prose test) and its cells hold whole paragraphs (``FRAME_MIN_LINES_PER_CELL``
        lines per non-empty cell). A heading candidate that labels a statement row (a word
        label, then two or more numbers: "2024 Notes | 37 | 443") is a cell, not a heading.
        Prose alone does not make a frame: a statement table's long row labels, or a register
        table's descriptions, stack like a paragraph, so ``FIGURE_GRID_MIN_LABELLED_ROWS``
        statement rows, or fewer lines per cell, keep a page-sized table a table."""
        lines = [self.lines[i] for i in idxs]
        statements = statement_rows(lines)
        labels = {id(line) for row in statements for line in row}
        if any(i in protected and id(self.lines[i]) not in labels for i in idxs):
            return True
        if area(r) < FIGURE_BACKGROUND_PAGE_SHARE * obj.width * obj.height:
            return False
        if len(statements) >= FIGURE_GRID_MIN_LABELLED_ROWS:
            return False
        cells = sum(1 for row in rows for cell in row if cell.strip())
        if len(lines) < FRAME_MIN_LINES_PER_CELL * max(cells, 1):
            return False
        return is_prose(lines, min_words=FIGURE_PROSE_MIN_WORDS,
                        line_share=FIGURE_PROSE_LINE_SHARE,
                        min_sentences=FIGURE_PROSE_MIN_SENTENCES,
                        sentence_share=FIGURE_PROSE_SENTENCE_SHARE,
                        run_lines=FIGURE_PROSE_RUN_LINES)  # fmt: skip

    def _table_text(self, rows: list[list[str]], idxs: Sequence[int]) -> tuple[str, list[str]]:
        """A table's ``text`` (its cells, row by row) and flags."""
        text = "\n".join(
            " | ".join(" ".join(c.split()) for c in row)
            for row in rows
            if any(c.strip() for c in row)
        )
        held = Counter(ch for i in idxs for ch in self.lines[i].text if ch.isalnum())
        if text and held - Counter(ch for ch in text if ch.isalnum()):
            # The engine's cells miss letters or digits of the table's own lines (text clipped
            # by a cell, prose in a ruled box). Compared by character, as cells split tokens
            # ("(13" and ")"). Keep the cells in TableData, but the text is the lines', so no
            # content is lost.
            ordered = sorted(idxs, key=self.pos.__getitem__)
            text = "\n".join(join_text([self.lines[i].text]) for i in ordered)
            return text, ["cell_text_incomplete"]
        return text, []

    def _holds_layout_lines(self, page: int, r: Rect) -> bool:
        """Whether a line a layout region took lies (half or more) inside ``r``."""
        if not self.layout_lines:
            return False
        return any(
            i in self.layout_lines and inside_share(line_rect(self.lines[i]), r) >= 0.5
            for i in self.on_page.get(page, [])
        )

    # -- layout regions (spec §6) -----------------------------------------------------------
    def layout_regions(
        self,
        regions: Sequence[LayoutRegion],
        headings: Sequence[HeadingCandidate],
        title_index: int | None,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> None:
        """One element per region, from the free lines inside it (see the module docstring)."""
        _, used = apply_regions(
            self.lines, regions, min_confidence=min_confidence, skip=set(self.owner)
        )
        title_region = self._title_region(used, headings, title_index)
        level_of: dict[int, int] = {}
        for h in headings:
            for k in range(h.line_index, h.line_index + max(1, h.line_count)):
                if h.level is not None:
                    level_of.setdefault(k, h.level)
        lists = [r for r, _ in used if r.label == "list_item"]
        for region, idxs in used:
            mapped = region_kind(region.label)
            r = self._clip(region.box)
            if mapped is None or r is None:
                continue
            idxs = [i for i in idxs if self.free(i)]  # a Table region may take more lines
            kind, flags = mapped
            flags.append("from_layout_model")
            conf = round(region.confidence, 4)
            page = region.page
            self.layout_lines.update(idxs)
            if kind == "table":
                self._layout_table(page, r, idxs, flags, conf)
                continue
            if kind == "figure":
                label = "\n".join(
                    join_text([self.lines[i].text]) for i in sorted(idxs, key=self.pos.__getitem__)
                )
                self.layout_rects[page].append(r)
                self.add("figure", idxs, text=label, page=page, boxes=[self._box(page, r)],
                         anchor=self._geometric_anchor(page, r),
                         figure=FigureData(source="layout_model", text=label), flags=flags,
                         confidence=conf)  # fmt: skip
                continue
            if not idxs:
                continue
            ordered = sorted(idxs, key=self.pos.__getitem__)
            first = self.lines[ordered[0]].text.strip()
            if region is title_region:
                kind = "title"
            if kind in ("heading", "title"):
                level = next((level_of[i] for i in ordered if i in level_of), None)
                if kind == "heading" and level is None:
                    kind = "paragraph"  # no level to place it at: keep its text as text
                self.add(kind, idxs, level=level, flags=flags, confidence=conf)
            elif kind in ("page_header", "page_footer"):
                if all(_is_page_number(self.lines[i].text) for i in idxs):
                    kind = "page_number"
                self.add(kind, idxs, flags=flags, confidence=conf)
            elif kind == "list_item":
                match = list_marker(first)
                rest = first[match.end() :] if match else first
                texts = [rest] + [self.lines[i].text for i in ordered[1:]]
                base = min((o.box.x0 for o in lists if o.page == page and h_overlap(
                    (o.box.x0, o.box.y0, o.box.x1, o.box.y1), r)), default=r[0])  # fmt: skip
                self.add("list_item", idxs, text=join_text(texts),
                         marker=match.group(0).strip() if match else None,
                         list_depth=int(max(0.0, r[0] - base) // LIST_INDENT_STEP),
                         flags=flags, confidence=conf)  # fmt: skip
            elif kind == "footnote":
                match = footnote_marker(first)
                rest = first[match.end() :] if match else first
                texts = [rest] + [self.lines[i].text for i in ordered[1:]]
                self.add("footnote", idxs, text=join_text(texts),
                         marker=match.group(1) if match else None, flags=flags,
                         confidence=conf)  # fmt: skip
            elif kind == "contents":
                text = "\n".join(join_text([self.lines[i].text]) for i in ordered)
                self.add("contents", idxs, text=text, flags=flags, confidence=conf)
            else:
                item = self.add(kind, idxs, flags=flags, confidence=conf)
                if kind == "caption":
                    self.layout_captions.append(item)

    def _title_region(
        self,
        used: Sequence[tuple[LayoutRegion, list[int]]],
        headings: Sequence[HeadingCandidate],
        title_index: int | None,
    ) -> LayoutRegion | None:
        """The Section-header region on page 1 that is the document title: the first (in
        reading order) holding the detected title line; with no title detected, the first
        set in the largest type on page 1, unless a heading candidate there is larger."""
        first: list[tuple[LayoutRegion, list[int]]] = sorted(
            ((r, [i for i in idxs if self.free(i)]) for r, idxs in used
             if r.page == 1 and r.label == "section_header"),
            key=lambda ri: min((self.pos[i] for i in ri[1]), default=len(self.pos)),
        )  # fmt: skip
        first = [(r, idxs) for r, idxs in first if idxs]
        if not first:
            return None
        if title_index is not None:
            return next((r for r, idxs in first if title_index in idxs), None)
        sizes = [max(self.lines[i].size for i in idxs) for _, idxs in first]
        largest = max(sizes)
        detected = [h.size for h in headings if h.page == 1 and "from_layout_model" not in h.flags]
        if detected and max(detected) > largest + PARAGRAPH_SIZE_DELTA:
            return None  # the largest type on page 1 is a heading the detector found
        return next(r for (r, _), size in zip(first, sizes, strict=True) if size >= largest)

    def _layout_table(
        self, page: int, r: Rect, idxs: list[int], flags: list[str], conf: float
    ) -> None:
        """A Table region: the engine table it overlaps most (by at least half of the smaller
        box), with that table's lines too; else cells from its lines' rows."""
        obj = self.pages.get(page)
        match: int | None = None
        best = 0.5
        for k, (box, _) in enumerate(obj.tables if obj is not None else []):
            t = self._clip(box)
            if t is None or (page, k) in self.used_tables:
                continue
            share = max(inside_share(t, r), inside_share(r, t))
            if share >= best:
                match, best = k, share
        rect = r
        rows: list[list[str]] = []
        if obj is not None and match is not None:
            self.used_tables.add((page, match))
            box, rows = obj.tables[match]
            t = self._clip(box) or r
            rect = union([r, t])
            extra = [i for i in self._inside(page, t) if i not in idxs]
            self.layout_lines.update(extra)
            idxs = idxs + extra
        if not idxs:
            return  # its text is another element's: an engine table would repeat it
        if not rows:
            rows = self._rows_from_lines(idxs)
        text, more = self._table_text(rows, idxs)
        if not text:
            return
        self.layout_rects[page].append(rect)
        self.add("table", idxs, text=text, page=page, boxes=[self._box(page, rect)],
                 anchor=self._geometric_anchor(page, rect), table=TableData(rows=rows),
                 flags=flags + more, confidence=conf)  # fmt: skip

    def _rows_from_lines(self, idxs: Sequence[int]) -> list[list[str]]:
        """Table rows from lines: lines on one row (``same_row`` with its first line) are its
        cells, left to right."""
        rows: list[list[Line]] = []
        for i in sorted(idxs, key=lambda k: (self.lines[k].y0, self.lines[k].x0)):
            line = self.lines[i]
            if rows and same_row(rows[-1][0], line):
                rows[-1].append(line)
            else:
                rows.append([line])
        return [[" ".join(c.text.split()) for c in sorted(row, key=lambda c: c.x0)] for row in rows]

    def link_layout_captions(self) -> None:
        """Link each Caption region's element to a figure or table, as rule 5 does."""
        for caption in self.layout_captions:
            lines = [self.lines[i] for i in caption.lines]
            r = union([line_rect(line) for line in lines])
            height = lines[0].y1 - lines[0].y0
            wanted = "table" if lines[0].text.strip().lower().startswith("table") else "figure"
            target = self._caption_target(caption.page, r, height, wanted)
            if target is not None:
                caption.caption_for = target
                target.caption = caption

    # -- rule 4: figures --------------------------------------------------------------------
    def figures(self, protected: set[int]) -> None:
        """Images and vector regions as figures, unless that would hide text that is content
        (``protected``: the heading and title lines)."""
        for obj in self.pages.values():
            page_area = obj.width * obj.height
            if page_area <= 0:
                continue
            tables = [
                r
                for k, (box, _) in enumerate(obj.tables)
                if (obj.page, k) not in self.frames and (r := self._clip(box)) is not None
            ]
            kinds: tuple[tuple[list[Box], Literal["image", "vector"]], ...] = (
                (obj.images, "image"),
                (obj.drawings, "vector"),
            )
            candidates = [
                (r, source)
                for boxes, source in kinds
                for box in boxes
                if (r := self._clip(box)) is not None
            ]
            candidates.sort(key=lambda c: -area(c[0]))  # stable: images first at equal size
            accepted: list[Rect] = list(self.layout_rects.get(obj.page, []))
            for r, source in candidates:
                size = area(r)
                if size < FIGURE_MIN_AREA * page_area:
                    continue
                if any(inside_share(r, t) >= 0.5 for t in tables + accepted):
                    continue  # table rulings, a picture in a cell, or part of a larger figure
                if self._holds_layout_lines(obj.page, r):
                    continue  # the layout model typed the text on it
                inside = self._without_captions(self._inside(obj.page, r))
                if source == "vector" and len(inside) / (size / 1000) >= FIGURE_MAX_TEXT_DENSITY:
                    continue
                background = size >= FIGURE_BACKGROUND_PAGE_SHARE * page_area  # a page frame
                if inside and (background or self._is_content(inside, r, protected)):
                    continue  # its lines stay free for the later rules
                accepted.append(r)
                inside = self._without_text_from_above(inside)
                label = "\n".join(join_text([self.lines[i].text]) for i in inside)
                self.add("figure", inside, text=label, page=obj.page,
                         boxes=[self._box(obj.page, r)],
                         anchor=self._geometric_anchor(obj.page, r),
                         figure=FigureData(source=source, text=label))  # fmt: skip

    def _without_captions(self, inside: list[int]) -> list[int]:
        """``inside`` minus caption lines (a "Figure 3:" title set within the figure's own
        region), which the caption rule then links to the figure."""
        caption: set[int] = set()
        for i in inside:
            if i not in caption and caption_match(self.lines[i].text.strip()):
                caption.update(self._caption_block(i))
        return [i for i in inside if i not in caption]

    def _without_text_from_above(self, inside: list[int]) -> list[int]:
        """``inside`` minus the end of a paragraph that starts above the figure's region and
        runs into its edge: each line stacked under the previous one in reading order, at its
        left edge and the same size, after a line that does not end a sentence."""
        held = set(inside)
        dropped: set[int] = set()
        idxs = self.on_page[self.lines[inside[0]].page] if inside else []
        for prev, i in zip(idxs, idxs[1:], strict=False):
            if i in held and (prev not in held or prev in dropped):
                a, b = self.lines[prev], self.lines[i]
                if (
                    0 <= b.y0 - a.y1 <= a.y1 - a.y0
                    and abs(b.x0 - a.x0) < FIRST_LINE_INDENT
                    and abs(a.size - b.size) <= PARAGRAPH_SIZE_DELTA
                    and not ends_sentence(a.text)
                ):
                    dropped.add(i)  # it continues the text above
        return [i for i in inside if i not in dropped]

    def _is_content(self, inside: list[int], r: Rect, protected: set[int]) -> bool:
        """Whether the text inside a figure candidate is content rather than figure labels: a
        heading or the title, a table, prose, a list, a code listing, or text set on a shaded
        box or photo. (Any text on a page-sized box makes it a background; see ``figures``.)"""
        lines = [self.lines[i] for i in inside]
        covered = sum(overlap_area(line_rect(line), r) for line in lines)
        return (
            any(i in protected for i in inside)
            or covered >= FIGURE_MAX_TEXT_COVERAGE * area(r)
            or is_code(lines, line_share=FIGURE_CODE_LINE_SHARE)
            or sum(1 for line in lines if list_marker(line.text.strip())) >= FIGURE_LIST_MIN_ITEMS
            or is_text_grid(lines, min_rows=FIGURE_GRID_MIN_ROWS,
                            min_columns=FIGURE_GRID_MIN_COLUMNS,
                            tolerance=FIGURE_GRID_ALIGN_TOLERANCE,
                            min_labelled_rows=FIGURE_GRID_MIN_LABELLED_ROWS,
                            line_share=FIGURE_GRID_LINE_SHARE)
            or is_prose(lines, min_words=FIGURE_PROSE_MIN_WORDS,
                        line_share=FIGURE_PROSE_LINE_SHARE,
                        min_sentences=FIGURE_PROSE_MIN_SENTENCES,
                        sentence_share=FIGURE_PROSE_SENTENCE_SHARE,
                        run_lines=FIGURE_PROSE_RUN_LINES)
        )  # fmt: skip

    # -- rule 5: captions -------------------------------------------------------------------
    def captions(self) -> None:
        if not self.objects:
            return
        for i in self.order:
            if not self.free(i) or not caption_match(self.lines[i].text.strip()):
                continue
            block = self._caption_block(i)
            r = union([line_rect(self.lines[k]) for k in block])
            height = self.lines[i].y1 - self.lines[i].y0
            wanted = "table" if self.lines[i].text.strip().lower().startswith("table") else "figure"
            target = self._caption_target(self.lines[i].page, r, height, wanted)
            if target is not None:
                caption = self.add("caption", block, caption_for=target)
                target.caption = caption

    def _caption_target(self, page: int, r: Rect, height: float, wanted: str) -> _Pending | None:
        """The figure or table a caption at ``r`` belongs to, if any (rule 5)."""
        best: tuple[tuple[bool, bool, float], _Pending] | None = None
        for target in self.objects.get(page, []):
            t = target.boxes[0]
            if target.caption is not None:
                continue
            if not h_overlap(r, (t.x0, t.y0, t.x1, t.y1)):
                continue
            sides = [("below", r[1] - t.y1), ("above", t.y0 - r[3])]
            box = (t.x0, t.y0, t.x1, t.y1)
            if inside_share(r, box) >= 0.5:  # set within the figure's own region
                upper = (r[1] + r[3]) < (t.y0 + t.y1)
                sides = [("above" if upper else "below", 0.0)]
            for side, distance in sides:
                if -height <= distance <= CAPTION_MAX_DISTANCE_LINES * height:
                    key = (target.kind != wanted, side != CAPTION_FIRST_SIDE[target.kind],
                           abs(distance))  # fmt: skip
                    if best is None or key < best[0]:
                        best = (key, target)
        return best[1] if best is not None else None

    def _caption_block(self, i: int) -> list[int]:
        """The caption line and the free lines continuing it as a short paragraph."""
        block = [i]
        line = self.lines[i]
        idxs = self.on_page[line.page]
        limit = self._gap_limit(line.page)
        for k in idxs[idxs.index(i) + 1 :]:
            prev, nxt = self.lines[block[-1]], self.lines[k]
            if (
                len(block) >= CAPTION_MAX_LINES
                or not self.free(k)
                or caption_match(nxt.text.strip())
                or abs(nxt.size - line.size) > PARAGRAPH_SIZE_DELTA
                or not 0 <= nxt.y0 - prev.y1 <= limit
                or not h_overlap(line_rect(prev), line_rect(nxt))
            ):
                break
            block.append(k)
        return block

    # -- rule 6: headings and the title -----------------------------------------------------
    def headings(self, headings: Sequence[HeadingCandidate], title_index: int | None) -> None:
        n = len(self.lines)
        title = title_index
        if title is not None and not (0 <= title < n and self.lines[title].page == 1):
            title = None
        for heading in sorted(headings, key=lambda h: self.pos.get(h.line_index, n)):
            span = range(
                heading.line_index, min(n, heading.line_index + max(1, heading.line_count))
            )
            idxs: list[int] = []
            for k in span:
                if k in self.owner:
                    break  # an earlier rule claimed it: the heading ends there
                if self.free(k):
                    idxs.append(k)
            if not idxs:
                continue
            whole = len(idxs) == len(span)
            text = (
                " ".join(heading.text.split())
                if whole
                else join_text([self.lines[k].text for k in idxs])
            )
            kind: ElementKind = "title" if title is not None and title in span else "heading"
            self.add(kind, idxs, text=text, level=heading.level)
        if title is not None and self.free(title):
            self.add("title", [title])

    # -- rule 7: footnotes ------------------------------------------------------------------
    def footnotes(self) -> None:
        body = self.signals.body_size
        if body <= 0:
            return
        current: list[int] = []
        marker = ""

        def flush() -> None:
            nonlocal current
            if current:
                first = self.lines[current[0]].text.strip()
                match = footnote_marker(first)
                rest = first[match.end() :] if match else first
                texts = [rest] + [self.lines[k].text for k in current[1:]]
                self.add("footnote", current, text=join_text(texts), marker=marker)
            current = []

        for i in self.order:
            line = self.lines[i]
            if not line.text.strip():
                continue
            in_zone = (
                self.free(i)
                and line.page_height > 0
                and line.size <= FOOTNOTE_MAX_SIZE_RATIO * body + 1e-9
                and (line.y0 + line.y1) / 2 >= (1 - FOOTNOTE_BOTTOM_SHARE) * line.page_height
            )
            if not in_zone or (current and self.lines[current[-1]].page != line.page):
                flush()
                if not in_zone:
                    continue
            match = footnote_marker(line.text.strip())
            if match is not None:
                flush()
                current, marker = [i], match.group(1)
            elif current:
                current.append(i)  # a continuation line, until the next marker
        flush()

    # -- rule 8: lists ----------------------------------------------------------------------
    def lists(self) -> None:
        stack: list[float] = []  # x0 of each open depth
        current: list[int] = []
        found: tuple[str, int] = ("", 0)

        def flush() -> None:
            nonlocal current
            if current:
                first = self.lines[current[0]].text.strip()
                match = list_marker(first)
                rest = first[match.end() :] if match else first
                texts = [rest] + [self.lines[k].text for k in current[1:]]
                self.add("list_item", current, text=join_text(texts), marker=found[0],
                         list_depth=found[1])  # fmt: skip
            current = []

        last: Line | None = None
        for i in self.order:
            line = self.lines[i]
            if not line.text.strip():
                continue
            if not self.free(i):
                flush()
                stack, last = [], None
                continue
            match = list_marker(line.text.strip())
            if match is not None:
                flush()
                if last is None or last.page != line.page or line.y0 < last.y0 - _EDGE_TOLERANCE:
                    stack = []  # a new list, page or column
                while stack and stack[-1] - line.x0 >= LIST_INDENT_STEP:
                    stack.pop()
                if not stack or line.x0 - stack[-1] >= LIST_INDENT_STEP:
                    stack.append(line.x0)
                current, found, last = [i], (match.group(0).strip(), len(stack) - 1), line
                continue
            if current and self._continues_item(current, line):
                current.append(i)
                last = line
                continue
            flush()
            stack, last = [], None
        flush()

    def _continues_item(self, item: list[int], line: Line) -> bool:
        first, prev = self.lines[item[0]], self.lines[item[-1]]
        return (
            line.page == prev.page
            and line.x0 >= first.x0 + _HANGING_INDENT
            and 0 <= line.y0 - prev.y1 <= self._gap_limit(line.page)
            and abs(line.size - prev.size) <= PARAGRAPH_SIZE_DELTA
        )

    # -- rule 9: formulas -------------------------------------------------------------------
    def _is_formula_line(self, line: Line) -> bool:
        text = line.text.strip()
        if symbol_share(text) >= FORMULA_MIN_SYMBOL_SHARE:
            return True
        return is_math_font(line.font) and has_math_content(text)

    def formulas(self) -> None:
        for i in self.order:
            if not self.free(i) or not self._is_formula_line(self.lines[i]):
                continue
            line = self.lines[i]
            row = [k for k in self.on_page[line.page]
                   if k != i and self.free(k) and same_row(line, self.lines[k])]  # fmt: skip
            numbers = [k for k in row if is_equation_number(self.lines[k].text.strip())]
            if any(k not in numbers and not self._is_formula_line(self.lines[k]) for k in row):
                continue  # running text beside it: not standing alone
            self.add("formula", [i, *numbers])

    # -- rule 10: paragraphs ----------------------------------------------------------------
    def paragraphs(self) -> None:
        fragments: list[list[int]] = []  # of the paragraph being built
        passed: set[str] = set()  # kinds of the elements passed since its last line

        def close() -> None:
            nonlocal fragments
            if fragments:
                self._add_paragraph(fragments)
            fragments = []

        for i in self.order:
            line = self.lines[i]
            if not line.text.strip():
                continue
            if not self.free(i):
                passed.add(self.owner[i].kind)
                continue
            if not fragments:
                fragments = [[i]]
                passed = set()
                continue
            prev = self.lines[fragments[-1][-1]]
            verdict = self._boundary(prev, line, fragments[-1], passed)
            passed = set()
            if verdict == "same":
                fragments[-1].append(i)
            elif verdict == "continue":
                fragments.append([i])
            else:
                close()
                fragments = [[i]]
        close()

    def _boundary(self, prev: Line, line: Line, fragment: list[int], passed: set[str]) -> str:
        """ "same" (the line extends the fragment), "continue" (a new fragment of the same
        paragraph, across a column, band or page boundary) or "break"."""
        if passed - _FLOATS:
            return "break"  # a heading, list item, formula or contents page in between
        size_change = abs(line.size - prev.size) > PARAGRAPH_SIZE_DELTA
        if line.page != prev.page:
            boundary = line.page == prev.page + 1
        elif same_row(prev, line) and not passed:
            return "same"
        elif (
            passed
            or line.y0 < prev.y0 - _EDGE_TOLERANCE
            or not h_overlap(line_rect(prev), line_rect(line))
        ):
            boundary = True  # an element in between, or a new column
        elif self._float_between(line.page, prev, line):
            boundary = True  # a figure or table between them (a band)
        else:
            gap = line.y0 - prev.y1
            indent = line.x0 - min(self.lines[k].x0 for k in fragment)
            ends = ends_sentence(prev.text)
            if (
                gap > self._gap_limit(line.page)
                or size_change
                or (indent >= FIRST_LINE_INDENT and ends)
            ):
                return "break"
            return "same"
        if (
            boundary
            and not size_change
            and not ends_sentence(prev.text)
            and line.text.strip()[0].islower()
        ):
            return "continue"
        return "break"

    def _float_between(self, page: int, upper: Line, lower: Line) -> bool:
        for item in self.objects.get(page, []):
            b = item.boxes[0]
            r = (b.x0, b.y0, b.x1, b.y1)
            if (
                b.y0 >= upper.y1 - _EDGE_TOLERANCE
                and b.y1 <= lower.y0 + _EDGE_TOLERANCE
                and (h_overlap(r, line_rect(upper)) or h_overlap(r, line_rect(lower)))
            ):
                return True
        return False

    def _add_paragraph(self, fragments: list[list[int]]) -> _Pending:
        boxes = [
            self._box(self.lines[f[0]].page, union([line_rect(self.lines[i]) for i in f]))
            for f in fragments
        ]
        flags = ["continued"] if len(fragments) > 1 else []
        idxs = [i for f in fragments for i in f]
        return self.add("paragraph", idxs, boxes=boxes, flags=flags)

    # -- output -----------------------------------------------------------------------------
    def finish(self) -> list[tuple[Element, list[int]]]:
        ranked = sorted(
            enumerate(self.pending),
            key=lambda e: (e[1].page, e[1].anchor, e[1].boxes[0].y0, e[1].boxes[0].x0, e[0]),
        )
        ids = {id(item): f"e{k}" for k, (_, item) in enumerate(ranked)}
        out: list[tuple[Element, list[int]]] = []
        for k, (_, item) in enumerate(ranked):
            ocr = self.source is not None and any(self.source[i] == "ocr" for i in item.lines)
            element = Element(
                id=ids[id(item)], kind=item.kind, text=item.text, boxes=item.boxes, order=k,
                section_id=None, level=item.level, list_depth=item.list_depth,
                marker=item.marker, table=item.table, figure=item.figure,
                caption_for=ids[id(item.caption_for)] if item.caption_for else None,
                caption_id=ids[id(item.caption)] if item.caption else None,
                source="ocr" if ocr else "text", flags=list(item.flags),
                confidence=item.confidence,
            )  # fmt: skip
            out.append((element, item.lines))
        return out
