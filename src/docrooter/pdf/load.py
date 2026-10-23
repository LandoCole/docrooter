"""``load_pdf``: assemble a :class:`~docrooter.document.models.Document` from a PDF.

The section tree comes from PyMuPDF4LLM's Markdown (:mod:`docrooter.document.pdf`): its layout
model finds the headings and their levels, and docrooter only repairs level jumps and applies
the PDF's bookmarks. Element mode (``elements=True``, :mod:`docrooter.pdf.element_mode`) reads
the native text lines with the ``pymupdf`` or ``pdfplumber`` engine instead."""

from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from docrooter import events
from docrooter._extras import install_hint
from docrooter.document.models import Document
from docrooter.errors import DocrooterError, MissingExtraError
from docrooter.pdf import lines as _lines_module

if TYPE_CHECKING:
    from docrooter.layout.base import LayoutEngine
    from docrooter.ocr.base import OcrEngine
    from docrooter.pdf.headings import HeadingCandidate

__all__ = ["ENGINES", "load_pdf"]

ENGINES = ("auto", "pymupdf", "pdfplumber", "pymupdf4llm")
"""``auto``, ``pymupdf`` and ``pymupdf4llm`` read the section tree with PyMuPDF4LLM; in element
mode ``auto`` and ``pymupdf`` read native lines with PyMuPDF, and ``pdfplumber`` (element mode
only) with pdfplumber."""

_DEFAULT_SCORE = 0.55  # element mode's heading cut
# Metadata titles that are really file names ("Microsoft Word - draft.docx", "report.pdf").
_FILENAME_TITLE = re.compile(
    r"\.(pdf|docx?|pptx?|xlsx?|indd|qxd|tex|rtf|odt|txt|ai|eps)\s*$|^microsoft (word|powerpoint)\b"
    r"|[/\\]",
    re.IGNORECASE,
)


# "A N N U A L  R E P O R T": a cover title set with letter-spacing, unusable as a title.
_LETTER_SPACED = re.compile(r"(?<![A-Za-z])(?:[A-Za-z] ){3,}[A-Za-z](?![A-Za-z])")


def load_pdf(
    path: str | os.PathLike[str],
    *,
    engine: str = "auto",
    min_heading_score: float | None = None,
    use_bookmarks: bool = True,
    use_printed_toc: bool = True,
    repair: bool = True,
    max_pages: int | None = None,
    on_event: events.EventSink | None = None,
    elements: bool = False,
    ocr: str | OcrEngine | None = None,
    layout: str | LayoutEngine = "heuristic",
    ocr_dpi: int = 200,
    tables: Literal["lines", "none"] = "lines",
    pages: Sequence[int] | None = None,
) -> Document:
    """Load a PDF into a :class:`Document`.

    **The section tree** (the default) is read with PyMuPDF4LLM (extra ``pdf``, AGPL-3.0):
    ``pymupdf4llm.to_markdown`` with its layout model, page headers and footers left out. Its
    Markdown headings and their levels are the sections; docrooter only cleans the titles
    (emphasis, HTML tags, bullets), repairs level jumps (``repair``) and, with
    ``use_bookmarks``, gives a heading that matches a PDF bookmark the bookmark's level.
    ``engine`` is ``auto``, ``pymupdf`` or ``pymupdf4llm`` (all the same here); ``pdfplumber``
    is for element mode only. ``ocr="auto"`` lets PyMuPDF4LLM OCR pages without a text layer
    (with an OCR engine it can find, such as RapidOCR or Tesseract); by default it does not.
    ``min_heading_score`` and ``use_printed_toc`` are element-mode options.

    ``Document.meta`` holds ``engine`` (``"pymupdf4llm"``), ``n_pages``, ``timings`` (seconds
    for ``extract`` and ``assemble``), ``bookmarks`` and ``warnings`` (``"no_text_layer"`` when
    no text could be extracted, ``"truncated_pages"`` when ``max_pages`` cut the reading).
    ``max_pages`` caps the pages read, which bounds time and memory on very long files;
    ``n_pages`` stays the whole document's count and bookmarks past the pages read are ignored.

    A PDF without text becomes one empty section ``s0``. Encrypted or unreadable files raise
    :class:`~docrooter.errors.DocrooterError`. ``on_event`` receives an
    :class:`~docrooter.events.Event` for each stage (``extract``, ``headings``, ``assemble``)
    and a final ``done``, or an ``error`` before the exception is re-raised.

    **Element mode** (``elements=True``, engines ``auto``, ``pymupdf`` and ``pdfplumber``) also
    types every block into ``Document.elements`` and builds the sections from them (see
    :mod:`docrooter.pdf.element_mode`). Pages are routed (``meta["routes"]``, one
    ``RouteDecision`` dump per page) and, with an ``ocr`` engine (a name, ``"auto"`` or an
    :class:`~docrooter.ocr.OcrEngine`), scanned pages without a text layer and garbled pages
    are read by OCR, rendered at ``ocr_dpi``; a scanned page that has some text layer keeps it
    and gains the OCR lines it lacks (route ``"text+ocr"``). Without an engine pages keep their
    text layer, and ``meta["warnings"]`` gets ``"ocr_skipped"`` when a page needed whole-page
    OCR (to replace or to merge). ``meta["needs_ocr"]`` counts what routing asked OCR for:
    ``pages`` to replace, ``merge_pages`` and image ``regions``. ``layout`` is ``"heuristic"``
    (the rules only), a layout engine name (``"heron"``, extra ``layout``, or one registered
    with :func:`docrooter.layout.register_layout_engine`; created, used and closed here) or a
    :class:`~docrooter.layout.LayoutEngine` instance, which the caller keeps and closes; its
    regions type the lines inside them (spec §6) and ``meta["layout"]`` and
    ``meta["layout_regions"]`` say which engine ran and how many regions it gave.
    ``meta["element_mode"]`` is ``True``, the timings add ``route``, ``ocr``, ``layout`` and
    ``elements``, and the events are ``extract``, ``route`` (progress per page), ``ocr``,
    ``layout``, ``signals``, ``headings``, ``elements`` (with counts by kind) and ``assemble``.
    ``tables="none"`` skips table finding in element mode, for speed (most of element mode's
    extra cost): ruled tables are not searched and their lines are typed by the other rules.
    ``pages`` (element mode only) reads just those 1-based pages, in page order, at most
    ``max_pages`` of them; ``n_pages`` stays the whole document's count.
    ``repair`` does not apply. Without ``elements``, ``layout``, ``ocr_dpi``, ``tables`` and
    ``pages`` must keep their defaults, and ``ocr`` is ``None`` or ``"auto"``.
    """
    if not elements and (
        ocr not in (None, "auto")
        or not _is_heuristic(layout)
        or ocr_dpi != 200
        or tables != "lines"
    ):
        raise TypeError(
            "layout, ocr_dpi, tables and an OCR engine: only valid with elements=True "
            "(the section tree takes ocr='auto' or None)"
        )
    if not elements and pages is not None:
        raise TypeError("pages: only valid with elements=True")
    if not elements and (min_heading_score is not None or not use_printed_toc):
        raise TypeError("min_heading_score and use_printed_toc: only valid with elements=True")
    with events.pipeline_run(on_event):
        if elements:
            from docrooter.pdf.element_mode import load_elements

            return load_elements(
                path,
                engine=engine,
                ocr=ocr,
                layout=layout,
                ocr_dpi=ocr_dpi,
                min_heading_score=min_heading_score,
                use_bookmarks=use_bookmarks,
                use_printed_toc=use_printed_toc,
                max_pages=max_pages,
                tables=tables,
                pages=pages,
            )
        return _load_pdf(
            path,
            engine=engine,
            use_bookmarks=use_bookmarks,
            repair=repair,
            max_pages=max_pages,
            ocr=ocr == "auto",
        )


def _is_heuristic(layout: object) -> bool:
    return isinstance(layout, str) and layout == "heuristic"


def _load_pdf(
    path: str | os.PathLike[str],
    *,
    engine: str,
    use_bookmarks: bool,
    repair: bool,
    max_pages: int | None,
    ocr: bool,
) -> Document:
    if engine not in ENGINES:
        raise DocrooterError(f"unknown PDF engine {engine!r}; use one of {', '.join(ENGINES)}")
    if engine == "pdfplumber":
        raise DocrooterError(
            "the pdfplumber engine reads native lines for element mode (elements=True); the "
            "section tree is read with PyMuPDF4LLM (engine 'auto')"
        )
    if max_pages is not None and max_pages < 1:
        raise ValueError(f"max_pages must be >= 1, got {max_pages}")
    from docrooter.document.pdf import load_pdf_markdown

    return load_pdf_markdown(
        Path(path), use_bookmarks=use_bookmarks, repair=repair, max_pages=max_pages, ocr=ocr
    )


def _auto_engine() -> str:
    try:
        _lines_module._import_pymupdf()
        return "pymupdf"
    except MissingExtraError:
        pass
    try:
        _lines_module._import_pdfplumber()
        return "pdfplumber"
    except MissingExtraError as exc:
        raise MissingExtraError(
            f"a PDF engine is required: {install_hint('pdf')} (or the pdf-mit extra)"
        ) from exc


def _heading_counts(headings: Sequence[HeadingCandidate]) -> dict[str, Any]:
    """``{"headings": n, "levels": {"1": n1, ...}}``, the data of a ``headings`` event."""
    levels = Counter(h.level or 1 for h in headings)
    return {"headings": len(headings), "levels": {str(k): levels[k] for k in sorted(levels)}}


def _metadata_title(path: Path) -> str | None:
    """The PDF's metadata title unless it is empty or looks like a file name; never raises."""
    try:
        pymupdf = _lines_module._import_pymupdf()
        doc = pymupdf.open(str(path), filetype="pdf")
    except Exception:
        return None
    try:
        title = " ".join(str((doc.metadata or {}).get("title") or "").split())
    except Exception:
        return None
    finally:
        doc.close()
    return None if not title or _FILENAME_TITLE.search(title) else title
