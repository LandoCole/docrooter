"""Element mode for PDFs (spec §5.3, §5.4, §7.4): ``load_pdf(elements=True)``.

The pipeline:

1. extract the engine-neutral page objects;
2. route every page (text layer, whole-page OCR, or region OCR for text-free images);
3. OCR what needs it. A whole-page OCR page loses its text-layer lines (garbled text is
   never kept) and its ruled tables, and gets the OCR lines instead. Region OCR is kept aside
   for the figures;
4. compute the signals on the combined lines, order them by XY-cut, and compute the signals
   again on the ordered list when the order changed;
5. find headings on the ordered lines;
6. type the elements, and put the region OCR text into the figures it belongs to;
7. build the section tree from the heading elements, with the heading detector's levels
   (typography, numbering, bookmark overrides) and gaps closed again if a heading candidate
   did not become a heading element.

Sections are built from elements, so they always agree: an element's ``section_id`` is the
section of the latest heading element before it (the preamble ``s0`` before the first one),
``Section.element_ids`` lists every element of the section in order, furniture and contents
included, and ``Section.text`` joins with blank lines the texts of its paragraphs, list items
and footnotes (with their markers), tables (as Markdown), captions, formulas and figure label
text from the text layer. Furniture, ``contents``, headings, and text read by region OCR are
not section text. The preamble exists when a non-furniture element precedes the first heading,
or when there is no heading at all (a PDF without text still yields one empty ``s0``).
"""

from __future__ import annotations

import logging
import os
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from docrooter import events
from docrooter.document.build import build_document
from docrooter.document.headings import repair_levels
from docrooter.document.models import Document, RawHeading
from docrooter.elements.models import Box, Element
from docrooter.errors import DocrooterError, MissingExtraError
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
from docrooter.pdf.routing import RouteDecision, route_page
from docrooter.pdf.signals import compute_signals, read_bookmarks

__all__ = ["LAYOUT_ENGINES", "load_elements", "section_text"]

LAYOUT_ENGINES = ("heuristic",)
TIMING_KEYS = ("extract", "route", "ocr", "signals", "headings", "elements", "assemble")
_FURNITURE = frozenset({"page_header", "page_footer", "page_number"})
_SECTION_TEXT_KINDS = frozenset(
    {"paragraph", "list_item", "table", "caption", "footnote", "formula", "figure"}
)
_MIN_REGION_OVERLAP = 0.5  # share of an OCR region that must lie inside its figure

_log = logging.getLogger(__name__)


@dataclass
class _Ocr:
    """What the OCR stage produced: replacement lines per page (with their confidences) and
    the text of OCR'd image regions."""

    pages: dict[int, tuple[list[Line], list[float | None]]] = field(default_factory=dict)
    regions: list[tuple[Box, str, float | None]] = field(default_factory=list)
    skipped: bool = False  # a page needed whole-page OCR and there was no engine
    failed: bool = False  # an engine call raised


def load_elements(
    path: str | os.PathLike[str],
    *,
    engine: str = "auto",
    ocr: str | OcrEngine | None = None,
    layout: str = "heuristic",
    ocr_dpi: int = 200,
    min_heading_score: float | None = None,
    use_bookmarks: bool = True,
    use_printed_toc: bool = True,
    max_pages: int | None = None,
) -> Document:
    """Load a PDF with typed elements; see :func:`docrooter.pdf.load.load_pdf`.

    ``ocr`` is ``None`` or ``"none"`` (route only: pages that need OCR keep their text layer,
    the route log says ``needs_ocr`` and ``meta["warnings"]`` gets ``"ocr_skipped"``),
    ``"auto"`` (the first installed engine, else as ``None``), an engine name (see
    :func:`docrooter.ocr.get_ocr_engine`) or an :class:`~docrooter.ocr.OcrEngine` instance,
    which the caller keeps and closes. An engine named here is created only if a page needs
    it, and closed before returning.
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
    if layout not in LAYOUT_ENGINES:
        raise DocrooterError(
            f"unknown layout engine {layout!r}; use one of {', '.join(LAYOUT_ENGINES)}"
        )
    if ocr_dpi <= 0:
        raise ValueError(f"ocr_dpi must be positive, got {ocr_dpi}")
    if max_pages is not None and max_pages < 1:
        raise ValueError(f"max_pages must be >= 1, got {max_pages}")
    if ocr is not None and not isinstance(ocr, str) and not isinstance(ocr, OcrEngine):
        raise TypeError("ocr must be None, an engine name or an OcrEngine")
    path = Path(path)
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
    )
    n_read = len(pages)
    n_pages = _page_count(path, engine) if max_pages is not None and n_read >= max_pages else n_read
    stage.end(lines=sum(len(p.lines) for p in pages), pages=n_read)
    clock = _tick(timings, "extract", clock)

    stage = events.start("route")
    routes: list[tuple[Literal["text", "ocr"], list[Box], RouteDecision]] = []
    for objs in pages:
        routes.append(route_page(objs))
        if stage.active:
            stage.progress(**routes[-1][2].model_dump())
    if stage.active:
        stage.end(
            pages=n_read,
            ocr=sum(1 for task, _, _ in routes if task == "ocr"),
            regions=sum(len(regions) for _, regions, _ in routes),
        )
    clock = _tick(timings, "route", clock)

    found = _run_ocr(path, pages, routes, ocr, ocr_dpi)
    clock = _tick(timings, "ocr", clock)

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
        else:
            lines += objs.lines
            source += ["text"] * len(objs.lines)
            confidence += [None] * len(objs.lines)
            build_pages.append(objs)

    stage = events.start("signals", engine=engine)
    bookmarks = read_bookmarks(path)
    if n_read < n_pages:
        bookmarks = [b for b in bookmarks if b[2] <= n_read]
    signals = compute_signals(lines, n_read, bookmarks=bookmarks)
    order = order_lines(lines, furniture_lines(lines, signals.excluded))
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
    if stage.active:
        stage.end(**_heading_counts(headings))
    clock = _tick(timings, "headings", clock)

    stage = events.start("elements")
    title = next((h for h in headings if h.level == 1 and not _LETTER_SPACED.search(h.text)), None)
    typed = build_elements_with_lines(
        build_pages,
        lines,
        list(range(len(lines))),
        signals,
        headings,
        title_index=title.line_index if title is not None and title.page == 1 else None,
        line_source=source,
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
        "element_mode": True,
    }
    return doc


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
    routes: list[tuple[Literal["text", "ocr"], list[Box], RouteDecision]],
    ocr: str | OcrEngine | None,
    dpi: int,
) -> _Ocr:
    """OCR the routed pages and regions with ``ocr``; update each decision's engine and reason
    in place. Emits the ``ocr`` stage (always, so the event order is fixed)."""
    out = _Ocr()
    todo = [k for k, (task, regions, _) in enumerate(routes) if task == "ocr" or regions]
    n_pages = sum(1 for k in todo if routes[k][0] == "ocr")
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
                # Only a page that needed whole-page OCR flags the document (spec §7.4); a
                # text-free image without OCR is just a figure without label text.
                out.skipped = out.skipped or task == "ocr"
                decision.engine = "none"
                decision.reason = f"needs_ocr: {decision.reason}; no OCR engine, text layer kept"
                continue
            decision.engine = name or "ocr"
            try:
                if task == "ocr":
                    whole = Box(page=objs.page, x0=0.0, y0=0.0, x1=objs.width, y1=objs.height)
                    found = _recognize(engine, path, whole, dpi, clip=False)
                    page_lines, page_conf = _to_lines(found, objs)
                    out.pages[objs.page] = (page_lines, page_conf)
                    n_lines += len(page_lines)
                    decision.metrics["ocr_lines"] = float(len(page_lines))
                    known = [c for c in page_conf if c is not None]
                    if known:
                        decision.metrics["ocr_confidence"] = round(statistics.fmean(known), 4)
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
                failure = f"OCR failed ({type(exc).__name__}), text layer kept"
                decision.reason = f"{decision.reason}; {failure}"
            if stage.active:
                stage.progress(page=objs.page, task=task)
    finally:
        if owned and engine is not None:
            engine.close()
    stage.end(pages=len(out.pages), regions=len(out.regions), lines=n_lines, skipped=out.skipped)
    return out


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
    """Region OCR text becomes the label text of the figure holding most of the region (at
    least half of it); it never becomes paragraph text. Text without a figure is dropped."""
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
        joined = f"{best.text}\n{text}" if best.text else text
        best.text = joined
        best.figure = best.figure.model_copy(update={"text": joined, "source": "ocr_region"})
        best.source = "ocr"
        best.confidence = None if conf is None else round(conf, 4)


# --- sections ------------------------------------------------------------------------------


def section_text(element: Element) -> str:
    """The text an element contributes to its section's ``Section.text`` (may be ``""``)."""
    kind = element.kind
    if kind == "table" and element.table is not None:
        if "cell_text_incomplete" in element.flags:
            return element.text  # the cells lost words; the text has them all
        return element.table.to_markdown()
    if kind == "figure":
        if element.figure is not None and element.figure.source == "ocr_region":
            return ""
        return element.text
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
