"""PDF loading: the section tree from PyMuPDF4LLM's Markdown (:func:`load_pdf_markdown`), the
default reading of :func:`~docrooter.pdf.load.load_pdf`."""

from __future__ import annotations

import contextlib
import io
import re
import time
from bisect import bisect_right
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from docrooter import events
from docrooter._extras import require
from docrooter.document.build import build_document
from docrooter.document.headings import repair_levels
from docrooter.document.markdown import extract_headings
from docrooter.document.models import Document, RawHeading
from docrooter.errors import DocrooterError
from docrooter.pdf.load import _LETTER_SPACED, _metadata_title, load_pdf

__all__ = ["apply_bookmarks", "load_pdf", "load_pdf_markdown"]

_BOOKMARK_FLAG = "level_from_bookmark"
_NON_WORD = re.compile(r"[\W_]+")
# Bold/italic markers PyMuPDF4LLM wraps around heading text ("# **Title**").
_EMPHASIS = re.compile(r"\*{1,3}|(?<!\w)_{1,3}|_{1,3}(?!\w)")
# HTML it uses for underline and highlight ("<u>Highlights</u>", "<mark>...</mark>").
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
# A list bullet left at the start of a heading ("• Summary of 2021").
_BULLET = re.compile(r"^[\u2022\u25aa\u25cf\u2013\u2014\-\s]+")
_HAS_WORD = re.compile(r"[^\W_]")


def _clean_title(title: str) -> str:
    """A heading's text without Markdown emphasis, HTML tags or a leading bullet."""
    text = _BULLET.sub("", _HTML_TAG.sub("", _EMPHASIS.sub("", title)))
    return " ".join(text.split()) or " ".join(title.split())


def _normalise(title: str) -> str:
    """Casefolded title with punctuation (including numbering dots) and extra spaces removed."""
    return _NON_WORD.sub(" ", title.casefold()).strip()


def apply_bookmarks(
    headings: Sequence[RawHeading], toc: Sequence[Sequence[Any]]
) -> list[RawHeading]:
    """Give every heading that matches a bookmark the bookmark's level.

    ``toc`` is PyMuPDF's ``[level, title, page]`` list. Titles match when equal after
    normalisation (casefold, punctuation and whitespace collapsed). Each bookmark is used at
    most once, in document order, preferring one that points at the heading's own page when
    several share a title. Matched headings get the ``level_from_bookmark`` flag; the others
    are returned unchanged. Inputs are not mutated.
    """
    pending: dict[str, list[tuple[int, Any]]] = {}
    for entry in toc:
        level, title = int(entry[0]), str(entry[1])
        page = entry[2] if len(entry) > 2 else None
        if level > 0:
            pending.setdefault(_normalise(title), []).append((level, page))

    out: list[RawHeading] = []
    for heading in headings:
        candidates = pending.get(_normalise(heading.title)) if heading.level > 0 else None
        if not candidates:
            out.append(heading)
            continue
        pick = next((i for i, (_, page) in enumerate(candidates) if page == heading.page), 0)
        level, _ = candidates.pop(pick)
        flags = list(heading.flags)
        if _BOOKMARK_FLAG not in flags:
            flags.append(_BOOKMARK_FLAG)
        out.append(heading.model_copy(update={"level": level, "flags": flags}))
    return out


def _join_chunks(chunks: Sequence[dict[str, Any]]) -> tuple[str, list[int], list[int]]:
    """Join page chunks into one Markdown text.

    Returns the text, the 1-based line each chunk starts on and each chunk's page number.
    """
    parts: list[str] = []
    starts: list[int] = []
    pages: list[int] = []
    line = 1
    for index, chunk in enumerate(chunks):
        text = str(chunk.get("text") or "") + "\n\n"
        parts.append(text)
        starts.append(line)
        line += len(text.splitlines())
        number = (chunk.get("metadata") or {}).get("page_number")
        pages.append(int(number) if number else index + 1)
    return "".join(parts), starts, pages


def load_pdf_markdown(
    path: Path,
    *,
    use_bookmarks: bool = True,
    repair: bool = True,
    max_pages: int | None = None,
    ocr: bool = False,
) -> Document:
    """The section tree of a PDF from PyMuPDF4LLM's Markdown. Requires the ``pdf`` extra.

    ``pymupdf4llm.to_markdown(path, page_chunks=True, header=False, footer=False)`` reads the
    pages with its layout model (page headers and footers left out; OCR only with ``ocr``).
    Its Markdown headings and their levels are the sections, each with the page it starts on
    (1-based). Titles lose Markdown emphasis, HTML tags and leading bullets; a heading with no
    letter or digit left is dropped (its text stays with the section before). ``repair`` fixes
    level jumps (:func:`~docrooter.document.headings.repair_levels`), and with ``use_bookmarks``
    a heading that matches a PDF bookmark takes the bookmark's level (:func:`apply_bookmarks`).
    The title is the first level-1 heading, else the PDF's metadata title, else the file name.
    PyMuPDF and PyMuPDF4LLM are AGPL-3.0 licensed.
    """
    pymupdf4llm = require("pymupdf4llm", "pdf")
    pymupdf = require("pymupdf", "pdf")
    timings: dict[str, float] = {}

    clock = time.perf_counter()
    stage = events.start("extract", engine="pymupdf4llm")
    try:
        with pymupdf.open(str(path)) as pdf:
            if pdf.needs_pass:
                raise DocrooterError(f"cannot open PDF {path}: file is encrypted")
            n_pages = pdf.page_count
            toc = pdf.get_toc() if use_bookmarks else []
    except DocrooterError:
        raise
    except Exception as exc:
        raise DocrooterError(f"cannot open PDF {path}: {exc}") from None
    n_read = n_pages if max_pages is None else min(max_pages, n_pages)
    options: dict[str, Any] = {
        "page_chunks": True,
        "header": False,
        "footer": False,
        "use_ocr": ocr,
        "show_progress": False,
    }
    if n_read < n_pages:
        options["pages"] = list(range(n_read))
        toc = [entry for entry in toc if entry[2] <= n_read]
    with contextlib.redirect_stdout(io.StringIO()):  # it prints parser and OCR messages
        chunks = pymupdf4llm.to_markdown(str(path), **options) if n_read else []
    markdown, starts, pages = _join_chunks(chunks)
    timings["extract"] = time.perf_counter() - clock
    stage.end(pages=n_read)

    clock = time.perf_counter()
    stage = events.start("headings", engine="pymupdf4llm")
    raw: list[RawHeading] = []
    for heading in extract_headings(markdown):
        page = pages[bisect_right(starts, heading.line) - 1] if starts else None
        if heading.level == 0:
            raw.append(heading.model_copy(update={"page": page}))
            continue
        clean = _clean_title(heading.title)
        if not _HAS_WORD.search(clean):  # an emptied heading: its text joins the one before
            if raw:
                joined = "\n".join(t for t in (raw[-1].text, heading.text) if t)
                raw[-1] = raw[-1].model_copy(update={"text": joined})
            continue
        raw.append(heading.model_copy(update={"title": clean, "page": page}))
    if repair:
        raw = repair_levels(raw)
    if toc:
        raw = apply_bookmarks(raw, toc)
    if not raw:
        raw = [RawHeading(title="Preamble", level=0, text="", line=0, page=1 if n_pages else None)]
    if stage.active:
        levels: dict[str, int] = {}
        for heading in raw:
            if heading.level:
                levels[str(heading.level)] = levels.get(str(heading.level), 0) + 1
        stage.end(headings=sum(levels.values()), levels=dict(sorted(levels.items())))
    timings["headings"] = time.perf_counter() - clock

    clock = time.perf_counter()
    stage = events.start("assemble")
    title: str | None = next(
        (h.title for h in raw if h.level == 1 and not _LETTER_SPACED.search(h.title)), None
    ) or _metadata_title(path)
    doc = build_document(raw, source=str(path), title=title or path.stem)
    timings["assemble"] = time.perf_counter() - clock
    stage.end(sections=len(doc.sections))

    warnings = [] if markdown.strip() else ["no_text_layer"]
    if n_read < n_pages:
        warnings.append("truncated_pages")
    doc.meta = {
        "engine": "pymupdf4llm",
        "n_pages": n_pages,
        "timings": {key: round(value, 4) for key, value in timings.items()},
        "bookmarks": len(toc),
        "warnings": warnings,
    }
    return doc
