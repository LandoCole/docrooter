"""Load a document from a path or Markdown text."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from docrooter import events
from docrooter.document.markdown import MARKDOWN_MIN_SCORE, parse_markdown
from docrooter.document.models import Document
from docrooter.document.pdf import load_pdf

_TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
_KNOWN_SUFFIXES = _TEXT_SUFFIXES | {".pdf"}
_TEXT_SOURCE = "<string>"


def _is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def load(
    source: str | os.PathLike[str],
    *,
    min_heading_score: float | None = None,
    repair: bool = True,
    elements: bool = False,
    on_event: events.EventSink | None = None,
    **pdf_options: Any,
) -> Document:
    """Load a document and build its section tree.

    A :class:`~pathlib.Path` (or any ``os.PathLike``) is always a file path. A ``str`` is treated
    as Markdown text when it contains a newline; otherwise it is a path if it names an existing
    file, a missing file (``FileNotFoundError``) if it has a known extension (``.md``,
    ``.markdown``, ``.txt``, ``.pdf``), and Markdown text in every other case.
    ``.pdf`` files go to :func:`docrooter.pdf.load.load_pdf`, which also takes ``pdf_options``
    (``engine``, ``use_bookmarks``, ``use_printed_toc``, ``max_pages``, and with
    ``elements=True`` the element-mode options ``ocr``, ``layout``, ``ocr_dpi``, ``tables`` and
    ``pages``);
    other files are read as UTF-8 Markdown. Headings are extracted, validated (scores below
    ``min_heading_score`` are merged into the preceding text), and, if ``repair`` is true, have
    their levels repaired.
    With ``elements=True`` the document also carries typed block elements
    (``Document.elements`` and ``Section.element_ids``); for a PDF this is element mode, whose
    sections are built from the elements (pages routed, OCR when an ``ocr`` engine is given).
    By default the result is exactly as in v0.3.
    ``min_heading_score`` defaults to 0.5 for Markdown and 0.55 for PDF.

    ``on_event`` receives pipeline :class:`~docrooter.events.Event` objects: the PDF stages (see
    :func:`~docrooter.load_pdf`), or just ``assemble`` for Markdown, then ``done``.
    """
    with events.pipeline_run(on_event):
        return _load(source, min_heading_score, repair, elements, on_event, pdf_options)


def _load(
    source: str | os.PathLike[str],
    min_heading_score: float | None,
    repair: bool,
    elements: bool,
    on_event: events.EventSink | None,
    pdf_options: dict[str, Any],
) -> Document:
    path: Path | None
    if isinstance(source, str):
        if "\n" in source:
            path = None
        else:
            candidate = Path(source)
            if _is_file(candidate):
                path = candidate
            elif candidate.suffix.lower() in _KNOWN_SUFFIXES:
                raise FileNotFoundError(f"No such file: {source}")
            else:
                path = None
    else:
        path = Path(source)

    is_pdf = path is not None and path.suffix.lower() == ".pdf"
    if pdf_options and not is_pdf:
        raise TypeError(f"{', '.join(sorted(pdf_options))}: only valid when loading a PDF")
    score = MARKDOWN_MIN_SCORE if min_heading_score is None else min_heading_score
    if path is None:
        return _build(str(source), _TEXT_SOURCE, None, score, repair, elements)

    if not _is_file(path):
        raise FileNotFoundError(f"No such file: {path}")
    if is_pdf:
        if min_heading_score is not None:
            pdf_options["min_heading_score"] = min_heading_score
        if elements:
            pdf_options["elements"] = True
        return load_pdf(path, repair=repair, on_event=on_event, **pdf_options)
    text = path.read_text(encoding="utf-8-sig")
    return _build(text, str(source), path.stem, score, repair, elements)


def _build(
    text: str,
    source: str,
    fallback_title: str | None,
    min_heading_score: float,
    repair: bool,
    elements: bool,
) -> Document:
    stage = events.start("assemble")
    doc = parse_markdown(
        text,
        source=source,
        elements=elements,
        min_heading_score=min_heading_score,
        repair=repair,
        fallback_title=fallback_title,
    )
    stage.end(sections=len(doc.sections))
    return doc
