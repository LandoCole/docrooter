"""``load_pdf``: assemble a :class:`~docrooter.document.models.Document` from native PDF lines
(spec §2.4). Engines: ``auto``, ``pymupdf``, ``pdfplumber``, ``both`` and the legacy
``pymupdf4llm`` path."""

from __future__ import annotations

import os
import re
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docrooter.document.build import build_document
from docrooter.document.headings import repair_levels
from docrooter.document.models import Document, RawHeading
from docrooter.errors import DocrooterError, MissingExtraError
from docrooter.pdf import lines as _lines_module
from docrooter.pdf.headings import HeadingCandidate, find_headings
from docrooter.pdf.lines import Line, extract_lines
from docrooter.pdf.signals import DocSignals, compute_signals, is_fuzzy_match, read_bookmarks

__all__ = ["ENGINES", "load_pdf"]

ENGINES = ("auto", "pymupdf", "pdfplumber", "both", "pymupdf4llm")

_KEEP_SCORE = 0.8  # `both`: a heading this confident in either engine survives alone
_TIMING_KEYS = ("extract", "signals", "headings", "assemble")
# Metadata titles that are really file names ("Microsoft Word - draft.docx", "report.pdf").
_FILENAME_TITLE = re.compile(
    r"\.(pdf|docx?|pptx?|xlsx?|indd|qxd|tex|rtf|odt|txt|ai|eps)\s*$|^microsoft (word|powerpoint)\b"
    r"|[/\\]",
    re.IGNORECASE,
)


@dataclass
class _Analysis:
    """Everything one engine found: its lines, signals and scored headings."""

    lines: list[Line]
    n_pages: int
    signals: DocSignals
    headings: list[HeadingCandidate]
    timings: dict[str, float] = field(default_factory=dict)


def load_pdf(
    path: str | os.PathLike[str],
    *,
    engine: str = "auto",
    min_heading_score: float = 0.55,
    use_bookmarks: bool = True,
    use_printed_toc: bool = True,
    repair: bool = True,
) -> Document:
    """Load a PDF into a :class:`Document` using native text extraction (no OCR).

    ``engine`` is ``auto`` (pymupdf if installed, else pdfplumber), ``pymupdf`` (extra ``pdf``),
    ``pdfplumber`` (extra ``pdf-mit``), ``both`` (pymupdf headings kept when pdfplumber agrees or
    either engine is at least 0.8 confident) or ``pymupdf4llm`` (the v0.1 Markdown path).
    Headings already carry repaired levels, so ``repair`` only affects the ``pymupdf4llm`` path.

    ``Document.meta`` holds ``engine``, ``n_pages``, ``timings`` (seconds for ``extract``,
    ``signals``, ``headings``, ``assemble``), ``bookmarks``, ``printed_toc_entries``,
    ``body_size`` and ``warnings`` (``"no_text_layer"`` when no text could be extracted).
    A PDF without text becomes one empty section ``s0``. Encrypted or unreadable files raise
    :class:`~docrooter.errors.DocrooterError`.
    """
    if engine not in ENGINES:
        raise DocrooterError(f"unknown PDF engine {engine!r}; use one of {', '.join(ENGINES)}")
    path = Path(path)
    if engine == "pymupdf4llm":
        from docrooter.document.pdf import _load_pdf_pymupdf4llm

        return _load_pdf_pymupdf4llm(path, min_heading_score=min_heading_score, repair=repair)
    if engine == "auto":
        engine = _auto_engine()

    options: dict[str, Any] = {
        "min_score": min_heading_score,
        "use_bookmarks": use_bookmarks,
        "use_printed_toc": use_printed_toc,
    }
    if engine == "both":
        primary = _analyse(path, "pymupdf", None, **options)
        other = _analyse(path, "pdfplumber", primary.signals.bookmarks, **options)
        start = time.perf_counter()
        skip = primary.signals.excluded | primary.signals.toc_page_lines
        headings = _merge_engines(
            primary.headings, other.headings, primary.lines, other.lines, skip=skip
        )
        merge_seconds = time.perf_counter() - start
        timings = {key: primary.timings[key] + other.timings[key] for key in _TIMING_KEYS}
        timings["headings"] += merge_seconds
    else:
        primary = _analyse(path, engine, None, **options)
        headings, timings = primary.headings, dict(primary.timings)

    start = time.perf_counter()
    raw = _assemble(primary.lines, primary.signals, headings, primary.n_pages)
    title = next((h.title for h in raw if h.level == 1), None) or _metadata_title(path)
    doc = build_document(raw, source=str(path), title=title or path.stem)
    timings["assemble"] = time.perf_counter() - start

    doc.meta = {
        "engine": engine,
        "n_pages": primary.n_pages,
        "timings": {key: round(timings[key], 4) for key in _TIMING_KEYS},
        "bookmarks": len(primary.signals.bookmarks),
        "printed_toc_entries": len(primary.signals.printed_toc),
        "body_size": primary.signals.body_size,
        "warnings": [] if primary.lines else ["no_text_layer"],
    }
    return doc


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
            "a PDF engine is required: pip install 'docrooter[pdf]' (or 'docrooter[pdf-mit]')"
        ) from exc


def _analyse(
    path: Path,
    engine: str,
    bookmarks: list[tuple[int, str, int]] | None,
    *,
    min_score: float,
    use_bookmarks: bool,
    use_printed_toc: bool,
) -> _Analysis:
    """Extract lines, compute signals and find headings with one engine. ``bookmarks`` skips
    re-reading the outline when a previous engine already did."""
    start = time.perf_counter()
    lines, n_pages = extract_lines(path, engine=engine)  # type: ignore[arg-type]
    extracted = time.perf_counter()
    if bookmarks is None:
        bookmarks = read_bookmarks(path)
    signals = compute_signals(lines, n_pages, bookmarks=bookmarks)
    signalled = time.perf_counter()
    headings = find_headings(
        lines,
        signals,
        min_score=min_score,
        use_bookmarks=use_bookmarks,
        use_printed_toc=use_printed_toc,
    )
    done = time.perf_counter()
    timings = {
        "extract": extracted - start,
        "signals": signalled - extracted,
        "headings": done - signalled,
        "assemble": 0.0,
    }
    return _Analysis(lines, n_pages, signals, headings, timings)


def _merge_engines(
    primary: Sequence[HeadingCandidate],
    other: Sequence[HeadingCandidate],
    primary_lines: Sequence[Line],
    other_lines: Sequence[Line],
    *,
    skip: set[int] | frozenset[int] = frozenset(),
) -> list[HeadingCandidate]:
    """Keep a primary heading when ``other`` has a fuzzy-matching heading on the same page or its
    score is at least 0.8. An ``other`` heading with no match and a score of at least 0.8 is
    added at the primary line it matches (same page, not in ``skip``), if any. The result is in
    reading order with level gaps closed; inputs are not mutated."""
    by_page: defaultdict[int, list[HeadingCandidate]] = defaultdict(list)
    for heading in other:
        by_page[heading.page].append(heading)

    matched: set[int] = set()
    kept: list[HeadingCandidate] = []
    for heading in primary:
        found = False
        for candidate in by_page[heading.page]:
            if is_fuzzy_match(heading.text, candidate.text):
                matched.add(id(candidate))
                found = True
        if found or heading.score >= _KEEP_SCORE:
            kept.append(heading.model_copy(deep=True))

    taken = {h.line_index for h in primary}
    for heading in other:
        if id(heading) in matched or heading.score < _KEEP_SCORE:
            continue
        index = _locate(heading, primary_lines, other_lines, taken | set(skip))
        if index is not None:
            taken.add(index)
            kept.append(heading.model_copy(update={"line_index": index}, deep=True))

    kept.sort(key=lambda h: h.line_index)
    repaired = repair_levels(
        [RawHeading(title="", level=h.level or 1, text="", line=h.line_index) for h in kept]
    )
    for heading, fixed in zip(kept, repaired, strict=True):
        heading.level = fixed.level
        heading.flags.extend(flag for flag in fixed.flags if flag not in heading.flags)
    return kept


def _locate(
    heading: HeadingCandidate,
    primary_lines: Sequence[Line],
    other_lines: Sequence[Line],
    taken: set[int],
) -> int | None:
    """Index of the first primary line on the heading's page whose text matches the heading's
    first line in the other engine."""
    if heading.line_index >= len(other_lines):
        return None
    first = other_lines[heading.line_index].text
    for index, line in enumerate(primary_lines):
        if line.page == heading.page and index not in taken and is_fuzzy_match(first, line.text):
            return index
    return None


def _span(lines: Sequence[Line], heading: HeadingCandidate) -> int:
    """Number of lines the heading's title was joined from (``find_headings`` joins stripped line
    texts with single spaces); 1 when they cannot be reconstructed."""
    joined = ""
    for index in range(heading.line_index, len(lines)):
        piece = lines[index].text.strip()
        joined = f"{joined} {piece}" if joined else piece
        if joined == heading.text:
            return index - heading.line_index + 1
        if len(joined) > len(heading.text):
            break
    return 1


def _assemble(
    lines: Sequence[Line],
    signals: DocSignals,
    headings: Sequence[HeadingCandidate],
    n_pages: int,
) -> list[RawHeading]:
    """Heading blocks with their body text, preceded by the preamble (level 0).

    A heading's body is the non-excluded lines (running headers/footers, page numbers, printed
    TOC pages) after its title lines and before the next heading, joined with newlines. The
    preamble holds the lines before the first heading; it is omitted when empty unless the PDF
    has no headings at all (a text-less PDF still yields one empty ``s0``)."""
    skip = signals.excluded | signals.toc_page_lines

    def body(start: int, stop: int) -> tuple[str, int | None]:
        kept = [lines[k] for k in range(start, stop) if k not in skip]
        return "\n".join(line.text for line in kept), (kept[0].page if kept else None)

    ordered = sorted(headings, key=lambda h: h.line_index)
    raw: list[RawHeading] = []
    first = ordered[0].line_index if ordered else len(lines)
    text, page = body(0, first)
    if text or not ordered:
        raw.append(
            RawHeading(
                title="Preamble", level=0, text=text, line=0, page=page or (1 if n_pages else None)
            )
        )
    for position, heading in enumerate(ordered):
        stop = ordered[position + 1].line_index if position + 1 < len(ordered) else len(lines)
        begin = min(heading.line_index + _span(lines, heading), stop)
        raw.append(
            RawHeading(
                title=heading.text,
                level=heading.level or 1,
                text=body(begin, stop)[0],
                line=heading.line_index,
                page=heading.page,
                heading_score=heading.score,
                flags=list(heading.flags),
            )
        )
    return raw


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
