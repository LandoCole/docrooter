"""PDF loading. :func:`load_pdf` is the native engine (:mod:`docrooter.pdf.load`); this module
also holds the v0.1 PyMuPDF4LLM Markdown path (``engine="pymupdf4llm"``)."""

from __future__ import annotations

import os
import re
from bisect import bisect_right
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from docrooter._extras import require
from docrooter.document.build import build_document
from docrooter.document.headings import repair_levels, validate_headings
from docrooter.document.markdown import extract_headings
from docrooter.document.models import Document, RawHeading
from docrooter.pdf.load import load_pdf

__all__ = ["apply_bookmarks", "load_pdf"]

_BOOKMARK_FLAG = "level_from_bookmark"
_NON_WORD = re.compile(r"[\W_]+")
# Bold/italic markers PyMuPDF4LLM wraps around heading text ("# **Title**").
_EMPHASIS = re.compile(r"\*{1,3}|(?<!\w)_{1,3}|_{1,3}(?!\w)")


def _clean_title(title: str) -> str:
    return " ".join(_EMPHASIS.sub("", title).split()) or title.strip()


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


def _load_pdf_pymupdf4llm(
    path: str | os.PathLike[str],
    *,
    min_heading_score: float = 0.5,
    repair: bool = True,
    max_pages: int | None = None,
) -> Document:
    """Load a PDF through PyMuPDF4LLM Markdown (the v0.1 path). Requires the ``pdf`` extra.

    The Markdown comes from ``pymupdf4llm.to_markdown(path, page_chunks=True)`` and each
    heading records the page it appears on (1-based). When the PDF has bookmarks, headings whose
    title matches a bookmark take its level and the ``level_from_bookmark`` flag; headings
    without a match keep their (repaired) level. PyMuPDF is AGPL-3.0 licensed.
    """
    pymupdf4llm = require("pymupdf4llm", "pdf")
    pymupdf = require("pymupdf", "pdf")
    path = Path(path)

    with pymupdf.open(str(path)) as pdf:
        n_pages = pdf.page_count
        toc = pdf.get_toc()
    truncated = max_pages is not None and max_pages < n_pages
    if truncated:
        pages = list(range(max_pages))  # type: ignore[arg-type]
        chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True, pages=pages)
        toc = [entry for entry in toc if entry[2] <= max_pages]
    else:
        chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    markdown, starts, pages = _join_chunks(chunks)

    raw = []
    for heading in extract_headings(markdown):
        page = pages[bisect_right(starts, heading.line) - 1] if starts else None
        update: dict[str, Any] = {"page": page}
        if heading.level > 0:
            update["title"] = _clean_title(heading.title)
        raw.append(heading.model_copy(update=update))

    raw = validate_headings(raw, min_score=min_heading_score)
    if repair:
        raw = repair_levels(raw)

    if toc:
        raw = apply_bookmarks(raw, toc)

    title = next((h.title for h in raw if h.level == 1), path.stem)
    doc = build_document(raw, source=str(path), title=title)
    doc.meta = {
        "engine": "pymupdf4llm",
        "n_pages": n_pages,
        "warnings": ["truncated_pages"] if truncated else [],
    }
    return doc
