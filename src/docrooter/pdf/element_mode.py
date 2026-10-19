"""Element mode for PDFs (spec §5.3, §5.4, §7.4): ``load_pdf(elements=True)``.

The pipeline:

1. extract the engine-neutral page objects;
2. route every page (text layer, whole-page OCR, text layer plus OCR, or region OCR for
   text-free images);
3. OCR what needs it. A whole-page OCR page loses its text-layer lines (garbled text is
   never kept) and its ruled tables, and gets the OCR lines instead. A ``text+ocr`` page (a
   scan with some text layer) keeps every text-layer line and its tables, and gains the OCR
   lines that the text layer does not already cover (less than ``MERGE_MAX_COVERED`` of the
   OCR box under text-layer lines). Region OCR is kept aside for the figures;
4. with a layout engine (spec §6), detect the regions of every page; a Table region that
   overlaps no ruled table gets the engine's "text" table strategy inside it (text-layer pages);
5. compute the signals on the combined lines, order them by XY-cut (the lines of each layout
   region moving as one unit, ordered by the region's box), and compute the signals again on
   the ordered list when the order changed;
6. find headings on the ordered lines, plus a heading for each Section-header region that holds
   no heading candidate (levelled by the candidates' typography);
7. type the elements (layout regions first, after furniture and contents; the rules for the
   lines outside them), and put the region OCR text into the figures it belongs to;
8. build the section tree from the heading elements, with the heading detector's levels
   (typography, numbering, bookmark overrides) and gaps closed again if a heading candidate
   did not become a heading element.

Sections are built from elements, so they always agree: an element's ``section_id`` is the
section of the latest heading element before it (the preamble ``s0`` before the first one),
``Section.element_ids`` lists every element of the section in order, furniture and contents
included, and ``Section.text`` joins with blank lines the texts of its paragraphs, list items
and footnotes (with their markers), tables (as Markdown), captions, formulas and figure label
text from the text layer (``Element.text``; region OCR text is only in ``FigureData.text``).
Furniture, ``contents``, headings, and text read by region OCR are not section text. The
preamble exists when a non-furniture element precedes the first heading, or when there is no
heading at all (a PDF without text still yields one empty ``s0``).
"""

from __future__ import annotations

import logging
import os
import statistics
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from docrooter import events
from docrooter.document.build import build_document
from docrooter.document.headings import repair_levels
from docrooter.document.models import Document, RawHeading
from docrooter.elements.models import Box, Element
from docrooter.errors import DocrooterError, MissingExtraError
from docrooter.layout.base import MIN_CONFIDENCE, LayoutEngine, LayoutRegion, apply_regions
from docrooter.ocr.base import OcrEngine, OcrLine, ocr_lines_to_lines
from docrooter.pdf import lines as _lines_module
from docrooter.pdf.elements import build_elements_with_lines, furniture_lines
from docrooter.pdf.headings import HeadingCandidate, find_headings
from docrooter.pdf.lines import Line, _has_alnum
from docrooter.pdf.load import (
    _DEFAULT_SCORE,
    _LETTER_SPACED,
    _auto_engine,
    _heading_counts,
    _metadata_title,
)
from docrooter.pdf.objects import PageObjects, extract_page_objects
from docrooter.pdf.reading_order import order_lines
from docrooter.pdf.routing import Route, RouteDecision, _union_share, route_page
from docrooter.pdf.signals import compute_signals, read_bookmarks
from docrooter.pdf.text_tables import find_text_tables

__all__ = ["LAYOUT_ENGINES", "load_elements", "section_text"]

LAYOUT_ENGINES = ("heuristic",)  # built into the rules; models: docrooter.layout registry
TIMING_KEYS = (
    "extract", "route", "ocr", "layout", "signals", "headings", "elements", "assemble",
)  # fmt: skip
_MAX_HEADING_LEVEL = 6
_STYLE_TOLERANCE = 0.5  # pt, as the heading detector's styles
_FURNITURE = frozenset({"page_header", "page_footer", "page_number"})
_SECTION_TEXT_KINDS = frozenset(
    {"paragraph", "list_item", "table", "caption", "footnote", "formula", "figure"}
)
_MIN_REGION_OVERLAP = 0.5  # share of an OCR region that must lie inside its figure
MERGE_MAX_COVERED = 0.5  # text+ocr: an OCR line this much under text-layer lines is a duplicate

_log = logging.getLogger(__name__)


@dataclass
class _Ocr:
    """What the OCR stage produced: replacement lines per page (with their confidences), lines
    to add to ``text+ocr`` pages, and the text of OCR'd image regions."""

    pages: dict[int, tuple[list[Line], list[float | None]]] = field(default_factory=dict)
    added: dict[int, tuple[list[Line], list[float | None]]] = field(default_factory=dict)
    regions: list[tuple[Box, str, float | None]] = field(default_factory=list)
    skipped: bool = False  # a page needed whole-page OCR and there was no engine
    failed: bool = False  # an engine call raised


def load_elements(
    path: str | os.PathLike[str],
    *,
    engine: str = "auto",
    ocr: str | OcrEngine | None = None,
    layout: str | LayoutEngine = "heuristic",
    ocr_dpi: int = 200,
    min_heading_score: float | None = None,
    use_bookmarks: bool = True,
    use_printed_toc: bool = True,
    max_pages: int | None = None,
    tables: Literal["lines", "none"] = "lines",
) -> Document:
    """Load a PDF with typed elements; see :func:`docrooter.pdf.load.load_pdf`.

    ``ocr`` is ``None`` or ``"none"`` (route only: pages that need OCR keep their text layer,
    the route log says ``needs_ocr`` and ``meta["warnings"]`` gets ``"ocr_skipped"``),
    ``"auto"`` (the first installed engine, else as ``None``), an engine name (see
    :func:`docrooter.ocr.get_ocr_engine`) or an :class:`~docrooter.ocr.OcrEngine` instance,
    which the caller keeps and closes. An engine named here is created only if a page needs
    it, and closed before returning.

    ``layout`` is ``"heuristic"``, a registered layout engine name (created before any page
    work, so a missing extra fails fast, and closed after detection) or a
    :class:`~docrooter.layout.LayoutEngine` instance, which the caller keeps and closes.

    ``tables="none"`` skips table finding, for speed: no ruled-table search in the object
    pass and no "text" table strategy in layout Table regions.
    """
    if engine in ("both", "pymupdf4llm"):
        raise DocrooterError(
            f"engine {engine!r} has no element mode; use 'auto', 'pymupdf' or 'pdfplumber'"
        )
    if engine not in ("auto", "pymupdf", "pdfplumber"):
        raise DocrooterError(
            f"unknown PDF engine {engine!r}; use one of auto, pymupdf, pdfplumber, both, "
            "pymupdf4llm"
        )
    if ocr_dpi <= 0:
        raise ValueError(f"ocr_dpi must be positive, got {ocr_dpi}")
    if tables not in ("lines", "none"):
        raise ValueError(f"tables must be 'lines' or 'none', got {tables!r}")
    if max_pages is not None and max_pages < 1:
        raise ValueError(f"max_pages must be >= 1, got {max_pages}")
    if ocr is not None and not isinstance(ocr, str) and not isinstance(ocr, OcrEngine):
        raise TypeError("ocr must be None, an engine name or an OcrEngine")
    if isinstance(ocr, str) and ocr not in ("none", "auto"):
        from docrooter.ocr.registry import available_ocr_engines

        known = available_ocr_engines()
        if ocr not in known:  # checked before any page work, needed or not
            raise DocrooterError(f"unknown OCR engine {ocr!r}; available: {', '.join(known)}")
    layout_engine = _layout_engine(layout)  # raises before any page work
    owned_layout = isinstance(layout, str) and layout_engine is not None
    closed = False

    def release() -> None:
        """Close an engine created here, once: after detection, or on any failure."""
        nonlocal closed
        if owned_layout and not closed and layout_engine is not None:
            closed = True
            layout_engine.close()

    try:
        return _load_elements(
            Path(path), engine=engine, ocr=ocr, layout_engine=layout_engine, release=release,
            ocr_dpi=ocr_dpi, min_heading_score=min_heading_score, use_bookmarks=use_bookmarks,
            use_printed_toc=use_printed_toc, max_pages=max_pages, tables=tables,
        )  # fmt: skip
    finally:
        release()


def _load_elements(
    path: Path,
    *,
    engine: str,
    ocr: str | OcrEngine | None,
    layout_engine: LayoutEngine | None,
    release: Callable[[], None],
    ocr_dpi: int,
    min_heading_score: float | None,
    use_bookmarks: bool,
    use_printed_toc: bool,
    max_pages: int | None,
    tables: Literal["lines", "none"],
) -> Document:
    """:func:`load_elements` after its argument checks; ``release`` closes an owned layout
    engine (called once detection is done)."""
    if engine == "auto":
        engine = _auto_engine()
    min_score = _DEFAULT_SCORE if min_heading_score is None else min_heading_score
    timings = dict.fromkeys(TIMING_KEYS, 0.0)

    clock = time.perf_counter()
    stage = events.start("extract", engine=engine)
    pages = extract_page_objects(
        path,
        engine=engine,  # type: ignore[arg-type]
        max_pages=max_pages,
        progress=(lambda page, total: stage.progress(pages=total, page=page))
        if stage.active
        else None,
        tables=tables,
    )
    n_read = len(pages)
    n_pages = _page_count(path, engine) if max_pages is not None and n_read >= max_pages else n_read
    stage.end(lines=sum(len(p.lines) for p in pages), pages=n_read)
    clock = _tick(timings, "extract", clock)

    stage = events.start("route")
    routes: list[tuple[Route, list[Box], RouteDecision]] = []
    for objs in pages:
        routes.append(route_page(objs))
        if stage.active:
            stage.progress(**routes[-1][2].model_dump())
    needs_ocr = _needs_ocr(routes)
    if stage.active:
        stage.end(
            pages=n_read,
            ocr=needs_ocr["pages"],
            merge=needs_ocr["merge_pages"],
            regions=needs_ocr["regions"],
        )
    clock = _tick(timings, "route", clock)

    found = _run_ocr(path, pages, routes, ocr, ocr_dpi)
    clock = _tick(timings, "ocr", clock)

    layout_name = "heuristic" if layout_engine is None else _engine_name(layout_engine)
    stage = events.start("layout", engine=layout_name, pages=n_read if layout_engine else 0)
    regions: list[LayoutRegion] = []
    min_conf = _min_confidence(layout_engine)
    if layout_engine is not None:
        try:
            regions = layout_engine.detect(path, [objs.page for objs in pages], dpi=None)
        finally:
            release()
        regions = [r for r in regions if r.confidence >= min_conf]
    text_tables = _text_table_requests(pages, regions, found) if tables == "lines" else {}
    region_tables = find_text_tables(path, engine, text_tables)  # type: ignore[arg-type]
    if stage.active:
        labels = Counter(r.label for r in regions)
        stage.end(regions=len(regions), labels=dict(sorted(labels.items())),
                  text_tables=len(region_tables))  # fmt: skip
    clock = _tick(timings, "layout", clock)

    lines: list[Line] = []
    source: list[Literal["text", "ocr"]] = []
    confidence: list[float | None] = []
    build_pages: list[PageObjects] = []
    for objs in pages:
        if objs.page in found.pages:
            page_lines, page_conf = found.pages[objs.page]
            lines += page_lines
            source += ["ocr"] * len(page_lines)
            confidence += page_conf
            build_pages.append(objs.model_copy(update={"lines": page_lines, "tables": []}))
        elif objs.page in found.added:
            merged, merged_source, merged_conf = _merge_page_lines(
                objs.lines, *found.added[objs.page]
            )
            lines += merged
            source += merged_source
            confidence += merged_conf
            build_pages.append(objs.model_copy(update={"lines": merged}))
        else:
            lines += objs.lines
            source += ["text"] * len(objs.lines)
            confidence += [None] * len(objs.lines)
            build_pages.append(objs)
        extra_tables = [
            (box, region_tables[(objs.page, k)])
            for k, box in enumerate(text_tables.get(objs.page, []))
            if (objs.page, k) in region_tables
        ]
        if extra_tables:
            last = build_pages[-1]
            build_pages[-1] = last.model_copy(update={"tables": last.tables + extra_tables})

    stage = events.start("signals", engine=engine)
    bookmarks = read_bookmarks(path)
    if n_read < n_pages:
        bookmarks = [b for b in bookmarks if b[2] <= n_read]
    signals = compute_signals(lines, n_read, bookmarks=bookmarks)
    groups: dict[int, tuple[float, float, float, float]] = {}
    for region, idxs in (
        apply_regions(lines, regions, min_confidence=min_conf)[1] if regions else []
    ):
        groups.update(dict.fromkeys(idxs, (region.box.x0, region.box.y0, region.box.x1,
                                           region.box.y1)))  # fmt: skip
    order = order_lines(lines, furniture_lines(lines, signals.excluded), groups)
    if order != list(range(len(lines))):
        lines = [lines[i] for i in order]
        source = [source[i] for i in order]
        confidence = [confidence[i] for i in order]
        signals = compute_signals(lines, n_read, bookmarks=bookmarks)
    if stage.active:
        stage.end(bookmarks=len(signals.bookmarks), printed_toc_entries=len(signals.printed_toc))
    clock = _tick(timings, "signals", clock)

    stage = events.start("headings", engine=engine)
    headings = find_headings(
        lines,
        signals,
        min_score=min_score,
        use_bookmarks=use_bookmarks,
        use_printed_toc=use_printed_toc,
    )
    if regions:
        skip = furniture_lines(lines, signals.excluded) | signals.toc_body_lines
        headings = _layout_headings(lines, headings, regions, skip, min_conf)
    if stage.active:
        stage.end(**_heading_counts(headings))
    clock = _tick(timings, "headings", clock)

    stage = events.start("elements")
    title = next(
        (
            h
            for h in headings
            if h.level == 1
            and not _LETTER_SPACED.search(h.text)
            and "from_layout_model" not in h.flags  # the title is the heading detector's
        ),
        None,
    )
    typed = build_elements_with_lines(
        build_pages,
        lines,
        list(range(len(lines))),
        signals,
        headings,
        title_index=title.line_index if title is not None and title.page == 1 else None,
        line_source=source,
        regions=regions,
        region_min_confidence=min_conf,
    )
    _set_confidence(typed, confidence)
    _fill_region_text([element for element, _ in typed], found.regions)
    if stage.active:
        kinds = Counter(element.kind for element, _ in typed)
        stage.end(elements=len(typed), kinds=dict(sorted(kinds.items())))
    clock = _tick(timings, "elements", clock)

    stage = events.start("assemble")
    doc = _assemble(path, typed, headings, n_read)
    clock = _tick(timings, "assemble", clock)
    stage.end(sections=len(doc.sections))

    warnings = [] if lines else ["no_text_layer"]
    if found.skipped:
        warnings.append("ocr_skipped")
    if found.failed:
        warnings.append("ocr_failed")
    if n_read < n_pages:
        warnings.append("truncated_pages")
    doc.meta = {
        "engine": engine,
        "n_pages": n_pages,
        "timings": {key: round(timings[key], 4) for key in TIMING_KEYS},
        "bookmarks": len(signals.bookmarks),
        "printed_toc_entries": len(signals.printed_toc),
        "body_size": signals.body_size,
        "warnings": warnings,
        "routes": [decision.model_dump() for _, _, decision in routes],
        "needs_ocr": needs_ocr,
        "layout": layout_name,
        "layout_regions": len(regions),
        "element_mode": True,
    }
    return doc


# --- layout ----------------------------------------------------------------------------------


def _layout_engine(layout: str | LayoutEngine) -> LayoutEngine | None:
    """The engine for ``layout`` (``None`` for the heuristic rules)."""
    if isinstance(layout, str):
        if layout in LAYOUT_ENGINES:
            return None
        from docrooter.layout.registry import get_layout_engine

        return get_layout_engine(layout)  # names the known engines for an unknown one
    if not isinstance(layout, LayoutEngine):
        raise TypeError("layout must be 'heuristic', a layout engine name or a LayoutEngine")
    return layout


def _min_confidence(engine: LayoutEngine | None) -> float:
    """The engine's own ``min_confidence`` if it has one, else ``MIN_CONFIDENCE`` (spec §6)."""
    value = getattr(engine, "min_confidence", None)
    return float(value) if isinstance(value, int | float) else MIN_CONFIDENCE


def _engine_name(engine: LayoutEngine) -> str:
    return str(getattr(engine, "name", None) or type(engine).__name__)


def _text_table_requests(
    pages: Sequence[PageObjects], regions: Sequence[LayoutRegion], found: _Ocr
) -> dict[int, list[Box]]:
    """Table regions that overlap no ruled table by half of the smaller box, by page, on pages
    that keep their text layer (whole-page OCR pages have no text for the "text" strategy)."""
    by_page = {objs.page: objs for objs in pages}
    out: dict[int, list[Box]] = {}
    for region in regions:
        objs = by_page.get(region.page)
        if region.label != "table" or objs is None or region.page in found.pages:
            continue
        if not any(_box_overlap(region.box, box) >= 0.5 for box, _ in objs.tables):
            out.setdefault(region.page, []).append(region.box)
    return out


def _box_overlap(a: Box, b: Box) -> float:
    """The intersection of two boxes over the smaller one's area."""
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    smaller = min((a.x1 - a.x0) * (a.y1 - a.y0), (b.x1 - b.x0) * (b.y1 - b.y0))
    return max(w, 0.0) * max(h, 0.0) / smaller if smaller > 0 else 0.0


def _layout_headings(
    lines: Sequence[Line],
    headings: list[HeadingCandidate],
    regions: Sequence[LayoutRegion],
    skip: set[int],
    min_confidence: float = MIN_CONFIDENCE,
) -> list[HeadingCandidate]:
    """``headings`` plus one candidate (flag ``from_layout_model``, score = the region's
    confidence) for each Section-header region whose lines hold no candidate, so the region
    becomes a heading element and a section (spec §5.3). Its level is the most common level of
    the candidates in its style (size within 0.5 pt, same weight); without one, one below the
    deepest candidate set larger than it, or 1. Lines in ``skip`` (furniture, contents pages)
    are never headings."""
    covered = {
        k for h in headings for k in range(h.line_index, h.line_index + max(1, h.line_count))
    }
    added: list[HeadingCandidate] = []
    for region, idxs in apply_regions(lines, regions, min_confidence=min_confidence, skip=skip)[1]:
        if region.label != "section_header" or not idxs or covered.intersection(idxs):
            continue
        first = lines[idxs[0]]
        size = max(lines[i].size for i in idxs)
        same = Counter(
            h.level
            for h in headings
            if h.level is not None
            and abs(h.size - size) <= _STYLE_TOLERANCE
            and h.bold == first.bold
        )
        if same:
            level = same.most_common(1)[0][0]
        else:
            larger = [h.level for h in headings if h.level and h.size > size + _STYLE_TOLERANCE]
            level = min(max(larger) + 1, _MAX_HEADING_LEVEL) if larger else 1
        contiguous = idxs == list(range(idxs[0], idxs[-1] + 1))
        added.append(
            HeadingCandidate(
                line_index=idxs[0],
                line_count=len(idxs) if contiguous else 1,
                text=" ".join(" ".join(lines[i].text for i in idxs).split()),
                page=first.page,
                size=size,
                bold=first.bold,
                score=round(region.confidence, 4),
                level=level,
                flags=["from_layout_model"],
            )  # fmt: skip
        )
    return sorted(headings + added, key=lambda h: h.line_index) if added else headings


def _needs_ocr(routes: Sequence[tuple[Route, list[Box], RouteDecision]]) -> dict[str, int]:
    """What routing asked OCR for, page level apart from region level: pages to replace
    (``pages``), pages to merge (``merge_pages``, ``text+ocr``) and text-free image regions
    (``regions``). Whether an engine did it is in each decision's ``engine``."""
    return {
        "pages": sum(1 for task, _, _ in routes if task == "ocr"),
        "merge_pages": sum(1 for task, _, _ in routes if task == "text+ocr"),
        "regions": sum(len(regions) for _, regions, _ in routes),
    }


def _tick(timings: dict[str, float], key: str, since: float) -> float:
    now = time.perf_counter()
    timings[key] += now - since
    return now


def _page_count(path: Path, engine: str) -> int:
    """The whole document's page count (``extract_page_objects`` reads only ``max_pages``)."""
    if engine == "pymupdf":
        with _lines_module._import_pymupdf().open(str(path), filetype="pdf") as doc:
            return int(doc.page_count)
    with _lines_module._import_pdfplumber().open(str(path)) as pdf:
        return len(pdf.pages)


# --- OCR -------------------------------------------------------------------------------------


def _run_ocr(
    path: Path,
    pages: Sequence[PageObjects],
    routes: list[tuple[Route, list[Box], RouteDecision]],
    ocr: str | OcrEngine | None,
    dpi: int,
) -> _Ocr:
    """OCR the routed pages and regions with ``ocr``; update each decision's engine and reason
    in place. Emits the ``ocr`` stage (always, so the event order is fixed)."""
    out = _Ocr()
    todo = [k for k, (task, regions, _) in enumerate(routes) if task != "text" or regions]
    n_pages = sum(1 for k in todo if routes[k][0] != "text")
    n_regions = sum(len(routes[k][1]) for k in todo)
    owned = isinstance(ocr, str)
    engine: OcrEngine | None = None
    name: str | None = None
    if todo and ocr is not None and ocr != "none":
        if isinstance(ocr, str):
            name = ocr
            if ocr == "auto":
                from docrooter.ocr.registry import resolve_auto_ocr

                name = resolve_auto_ocr()
            if name is not None:
                from docrooter.ocr.registry import get_ocr_engine

                engine = get_ocr_engine(name)
        else:
            engine = ocr
        if engine is not None:
            name = str(getattr(engine, "name", None) or name or type(engine).__name__)
    stage = events.start("ocr", engine=name, pages=n_pages, regions=n_regions)
    n_lines = 0
    try:
        for k in todo:
            task, regions, decision = routes[k]
            objs = pages[k]
            if engine is None:
                # A page that needed whole-page OCR (to replace or to merge) flags the document
                # (spec §7.4); a text-free image without OCR is just a figure without label text.
                out.skipped = out.skipped or task != "text"
                decision.engine = "none"
                decision.reason = f"needs_ocr: {decision.reason}; no OCR engine, text layer kept"
                continue
            decision.engine = name or "ocr"
            capped = _render_dpi(objs.width, objs.height, dpi)
            if capped < dpi:  # a page over MAX_RENDER_PIXELS renders at a lower dpi
                decision.metrics["dpi_capped"] = round(capped, 1)
            try:
                if task != "text":
                    whole = Box(page=objs.page, x0=0.0, y0=0.0, x1=objs.width, y1=objs.height)
                    found = _recognize(engine, path, whole, dpi, clip=False)
                    page_lines, page_conf = _to_lines(found, objs)
                    decision.metrics["ocr_lines"] = float(len(page_lines))
                    known = [c for c in page_conf if c is not None]
                    if known:
                        decision.metrics["ocr_confidence"] = round(statistics.fmean(known), 4)
                    if task == "ocr":
                        out.pages[objs.page] = (page_lines, page_conf)
                        n_lines += len(page_lines)
                    else:
                        extra, extra_conf = _new_lines(found, objs)
                        out.added[objs.page] = (extra, extra_conf)
                        n_lines += len(extra)
                        decision.metrics["ocr_added"] = float(len(extra))
                else:
                    for region in regions:
                        found = _recognize(engine, path, region, dpi, clip=True)
                        text = "\n".join(" ".join(o.text.split()) for o in found if o.text.strip())
                        conf = statistics.fmean(o.confidence for o in found) if found else None
                        if text:
                            out.regions.append((region, text, conf))
                            n_lines += len(found)
            except MissingExtraError:
                raise  # the engine's package is missing: an error, not a page failure
            except Exception as exc:  # one failing page must not fail the document
                _log.warning("OCR failed on page %d", objs.page, exc_info=True)
                out.failed = True
                out.pages.pop(objs.page, None)
                out.added.pop(objs.page, None)
                failure = f"OCR failed ({type(exc).__name__}), text layer kept"
                decision.reason = f"{decision.reason}; {failure}"
            if stage.active:
                stage.progress(page=objs.page, task=task)
    finally:
        if owned and engine is not None:
            engine.close()
    stage.end(
        pages=len(out.pages) + len(out.added),
        regions=len(out.regions),
        lines=n_lines,
        skipped=out.skipped,
    )
    return out


def _render_dpi(width_pt: float, height_pt: float, dpi: int) -> float:
    from docrooter.ocr.render import render_dpi  # numpy and the renderers load on first use

    return render_dpi(width_pt, height_pt, dpi)


def _recognize(engine: OcrEngine, path: Path, box: Box, dpi: int, *, clip: bool) -> list[OcrLine]:
    """Render ``box`` (the whole page unless ``clip``) and recognise it; boxes in page points."""
    from docrooter.ocr.render import render_page

    png, width_px, height_px = render_page(path, box.page, dpi=dpi, clip=box if clip else None)
    return engine.recognize(png, width_px=width_px, height_px=height_px, page=box)


def _to_lines(found: list[OcrLine], objs: PageObjects) -> tuple[list[Line], list[float | None]]:
    """OCR output as ``Line`` objects of the page, with the confidence of each line (``None``
    for all of them if the lines cannot be matched to the engine's output)."""
    page_lines = ocr_lines_to_lines(found, objs.page, objs.width, objs.height)
    kept: list[float | None] = [o.confidence for o in found if _has_alnum(" ".join(o.text.split()))]
    return page_lines, kept if len(kept) == len(page_lines) else [None] * len(page_lines)


def _new_lines(found: list[OcrLine], objs: PageObjects) -> tuple[list[Line], list[float | None]]:
    """The OCR lines of a ``text+ocr`` page that the text layer lacks: those with less than
    ``MERGE_MAX_COVERED`` of their box under the union of the text-layer line boxes. A union,
    so a row the text layer splits into short lines (words, cells) still covers the OCR line
    that reads the whole row; a share of the OCR box, so an OCR line that runs on past a short
    text-layer line keeps its extra words. Their blocks are numbered after the text layer's,
    so a paragraph never spans both sources."""
    fresh = [o for o in found if _covered(o.box, objs.lines) < MERGE_MAX_COVERED]
    extra, conf = _to_lines(fresh, objs)
    if extra and objs.lines:
        offset = max(line.block for line in objs.lines) + 1
        extra = [line.model_copy(update={"block": line.block + offset}) for line in extra]
    return extra, conf


def _merge_page_lines(
    text_lines: Sequence[Line], extra: Sequence[Line], extra_conf: Sequence[float | None]
) -> tuple[list[Line], list[Literal["text", "ocr"]], list[float | None]]:
    """A ``text+ocr`` page's text-layer and added OCR lines merged top to bottom, then left to
    right, with the vertical gaps recomputed across the two sources (the concatenation is no
    extractor order, and reading order keeps a page without a gutter as it is). Returns the
    lines with the source and the OCR confidence of each."""
    tagged: list[tuple[Line, Literal["text", "ocr"], float | None]] = [
        *((line, "text", None) for line in text_lines),
        *((line, "ocr", conf) for line, conf in zip(extra, extra_conf, strict=True)),
    ]
    tagged.sort(key=lambda t: (t[0].y0, t[0].x0))
    lines = _lines_module._finalise([line.model_dump() for line, _, _ in tagged])
    return lines, [src for _, src, _ in tagged], [conf for _, _, conf in tagged]


def _covered(box: Box, lines: Sequence[Line]) -> float:
    """The share of ``box`` under the union of the ``lines``' boxes (0 for an empty box)."""
    width, height = box.x1 - box.x0, box.y1 - box.y0
    if width <= 0 or height <= 0:
        return 0.0
    inside = [
        Box(page=box.page, x0=line.x0 - box.x0, y0=line.y0 - box.y0,
            x1=line.x1 - box.x0, y1=line.y1 - box.y0)
        for line in lines
        if line.x1 > box.x0 and line.x0 < box.x1 and line.y1 > box.y0 and line.y0 < box.y1
    ]  # fmt: skip
    return _union_share(inside, width, height)


def _set_confidence(
    typed: list[tuple[Element, list[int]]], confidence: Sequence[float | None]
) -> None:
    """An element made of OCR lines gets the mean confidence of those lines."""
    for element, owned in typed:
        values = [c for c in (confidence[i] for i in owned) if c is not None]
        if values:
            element.confidence = round(statistics.fmean(values), 4)


def _fill_region_text(
    elements: list[Element], regions: list[tuple[Box, str, float | None]]
) -> None:
    """Region OCR text goes into ``FigureData.text`` of the figure holding most of the region
    (at least half of it), after the figure's text-layer labels, with ``source="ocr_region"``.
    ``Element.text`` keeps the text-layer labels only, so OCR text never becomes paragraph or
    section text (spec §7.4) and no text-layer label leaves ``Section.text``. The element is
    marked ``source="ocr"`` with the mean confidence of its regions. Text without a figure is
    dropped."""
    confidences: dict[int, list[float]] = {}
    for region, text, conf in regions:
        best: Element | None = None
        best_share = _MIN_REGION_OVERLAP
        area = max((region.x1 - region.x0) * (region.y1 - region.y0), 1e-9)
        for element in elements:
            if element.kind != "figure" or element.figure is None:
                continue
            for box in element.boxes:
                if box.page != region.page:
                    continue
                w = min(box.x1, region.x1) - max(box.x0, region.x0)
                h = min(box.y1, region.y1) - max(box.y0, region.y0)
                share = max(w, 0.0) * max(h, 0.0) / area
                if share >= best_share:
                    best, best_share = element, share
        if best is None or best.figure is None:
            continue
        old = best.figure.text
        joined = f"{old}\n{text}" if old else text
        best.figure = best.figure.model_copy(update={"text": joined, "source": "ocr_region"})
        best.source = "ocr"
        if conf is not None:
            confidences.setdefault(id(best), []).append(conf)
            best.confidence = round(statistics.fmean(confidences[id(best)]), 4)


# --- sections ------------------------------------------------------------------------------


def section_text(element: Element) -> str:
    """The text an element contributes to its section's ``Section.text`` (may be ``""``)."""
    kind = element.kind
    if kind == "table" and element.table is not None:
        if "cell_text_incomplete" in element.flags:
            return element.text  # the cells lost words; the text has them all
        return element.table.to_markdown()
    if kind == "figure":
        return element.text  # text-layer labels; region OCR text is only in FigureData.text
    if kind == "title" and element.level is None:
        return element.text  # a title line that is not a heading stays text
    if kind not in _SECTION_TEXT_KINDS:
        return ""
    if kind in ("list_item", "footnote") and element.marker:
        return f"{element.marker} {element.text}".strip()
    return element.text


def _starts_section(element: Element) -> bool:
    return element.kind in ("heading", "title") and element.level is not None


def _assemble(
    path: Path,
    typed: list[tuple[Element, list[int]]],
    headings: Sequence[HeadingCandidate],
    n_pages: int,
) -> Document:
    """The section tree built from the heading elements, with ``section_id``,
    ``element_ids`` and ``Section.text`` filled in."""
    by_line = {h.line_index: h for h in headings}
    elements = [element for element, _ in typed]
    heads = [k for k, element in enumerate(elements) if _starts_section(element)]
    first = heads[0] if heads else len(elements)
    has_preamble = not heads or any(e.kind not in _FURNITURE for e in elements[:first])

    blocks: list[tuple[int | None, int, int]] = []  # (heading element or None, start, stop)
    if has_preamble:
        blocks.append((None, 0, first))
    for position, k in enumerate(heads):
        stop = heads[position + 1] if position + 1 < len(heads) else len(elements)
        blocks.append((k, k + 1, stop))

    raw: list[RawHeading] = []
    for head, begin, stop in blocks:
        pieces = [section_text(elements[k]) for k in range(begin, stop)]
        text = "\n\n".join(piece for piece in pieces if piece)
        if head is None:
            page = next((e.boxes[0].page for e in elements[:first] if e.boxes), None)
            raw.append(
                RawHeading(
                    title="Preamble",
                    level=0,
                    text=text,
                    line=0,
                    page=page or (1 if n_pages else None),
                )  # fmt: skip
            )
            continue
        element, owned = typed[head]
        candidate = by_line.get(owned[0]) if owned else None
        raw.append(
            RawHeading(
                title=element.text,
                level=element.level or 1,
                text=text,
                line=owned[0] if owned else 0,
                page=element.boxes[0].page if element.boxes else None,
                heading_score=candidate.score if candidate is not None else 1.0,
                flags=list(candidate.flags) if candidate is not None else [],
            )
        )
    raw = _close_level_gaps(raw)

    title = next(
        (h.title for h in raw if h.level == 1 and not _LETTER_SPACED.search(h.title)), None
    ) or _metadata_title(path)
    doc = build_document(raw, source=str(path), title=title or path.stem)
    section_ids = list(doc.sections)
    for (head, begin, stop), section_id in zip(blocks, section_ids, strict=True):
        section = doc.sections[section_id]
        members = range(begin, stop) if head is None else range(head, stop)
        for k in members:
            elements[k].section_id = section_id
            section.element_ids.append(elements[k].id)
    doc.elements = elements
    return doc


def _close_level_gaps(raw: list[RawHeading]) -> list[RawHeading]:
    """Levels with gaps closed, as ``find_headings`` leaves them. A heading candidate that did
    not become an element (an earlier rule took its first line) can open a gap. Titles are
    left out so dotted numbering, already applied, cannot override a bookmark level."""
    fixed = repair_levels([h.model_copy(update={"title": ""}) for h in raw])
    out = []
    for heading, repaired in zip(raw, fixed, strict=True):
        if repaired.level != heading.level:
            flags = heading.flags + [f for f in repaired.flags if f not in heading.flags]
            heading = heading.model_copy(update={"level": repaired.level, "flags": flags})
        out.append(heading)
    return out
