"""PDF loading: the section tree from PyMuPDF4LLM's Markdown (:func:`load_pdf_markdown`), the
default reading of :func:`~docrooter.pdf.load.load_pdf`."""

from __future__ import annotations

import contextlib
import io
import re
import time
from bisect import bisect_right
from collections.abc import Iterator, Sequence
from difflib import SequenceMatcher
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
_FROM_BOOKMARK_FLAG = "heading_from_bookmark"
# A heading with no exact bookmark takes a leftover one this similar on a page this close.
_FUZZY_RATIO = 0.9
_FUZZY_PAGES = 1
# Both titles need this many characters (normalised) to be fuzzy matched: "Revenue" and
# "Revenues", or "Income Taxes" and "Income Tax", are different headings, not a typo.
_FUZZY_MIN_CHARS = 15
# Words that number what follows them ("Part II", "Item 7A", "Note 1", "Appendix B").
_NUMBERED_WORDS = frozenset(
    "part item note section article chapter appendix exhibit schedule table figure".split()
)
_ROMAN = re.compile(r"(?=[ivxl])l?x{0,3}(?:ix|iv|v?i{0,3})")
# In a body, between a bookmark title's words: any run of non-word characters (spaces,
# punctuation, Markdown emphasis) and single line breaks, but not a blank line.
_TITLE_GAP = r"(?:[^\w\n]|_|\n(?![^\S\n]*(?:\n|$)))+"
# Before and after the title on its line: spaces and emphasis markers only.
_TITLE_EDGE = r"(?:[^\S\n]|[*_])*"
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
    headings: Sequence[RawHeading],
    toc: Sequence[Sequence[Any]],
    *,
    line_pages: Sequence[Sequence[tuple[int, int | None]]] | None = None,
) -> list[RawHeading]:
    """Give every heading that matches a bookmark the bookmark's level, and add the bookmarked
    headings the Markdown has only as body text.

    ``toc`` is PyMuPDF's ``[level, title, page]`` list. First, titles match when equal after
    normalisation (casefold, punctuation and whitespace collapsed), preferring a bookmark that
    points at the heading's own page when several share a title. Then a heading still without
    one takes a bookmark left over whose normalised title is at least 90% similar
    (``difflib.SequenceMatcher`` ratio; both titles at least 15 characters, so "Revenue" never
    takes "Revenues"), whose numbering is the same ("Item 7" never takes
    "Item 7A", nor "Part I" "Part II") and whose page is within one page of the heading's.
    Each bookmark is used at most once, in document order. Matched headings get the
    ``level_from_bookmark`` flag.

    A bookmark (level and page 1 or more) still unused is looked for in the body text on its
    page: its title's words in order, case-insensitively, with only spaces, punctuation,
    emphasis or single line breaks between them, alone on its line or lines. The first such
    place splits the body: the text before stays, and a heading follows with the bookmark's
    title, level and page, the text after, and the ``heading_from_bookmark`` flag. A bookmark
    not found is skipped. ``line_pages`` gives, for each heading, the ``(line, page)`` of each
    line of its text (1-based Markdown line); without it a body counts as on its heading's
    page. The other headings are returned unchanged. Inputs are not mutated. Raises
    ``ValueError`` when ``line_pages`` has not one entry per heading.
    """
    if line_pages is not None and len(line_pages) != len(headings):
        raise ValueError("line_pages needs one entry per heading")
    bookmarks: list[tuple[int, str, Any]] = [
        (int(entry[0]), str(entry[1]), entry[2] if len(entry) > 2 else None) for entry in toc
    ]
    keys = [_normalise(title) for _, title, _ in bookmarks]
    used = [level <= 0 for level, _, _ in bookmarks]
    picks: list[int | None] = [None] * len(headings)

    by_key: dict[str, list[int]] = {}
    for index, key in enumerate(keys):
        if not used[index]:
            by_key.setdefault(key, []).append(index)
    for position, heading in enumerate(headings):
        candidates = by_key.get(_normalise(heading.title)) if heading.level > 0 else None
        if not candidates:
            continue
        pick = next((i for i in candidates if bookmarks[i][2] == heading.page), candidates[0])
        candidates.remove(pick)
        used[pick] = True
        picks[position] = pick

    for position, heading in enumerate(headings):
        if picks[position] is None and heading.level > 0 and heading.page is not None:
            near = _fuzzy_pick(_normalise(heading.title), heading.page, bookmarks, keys, used)
            if near is not None:
                used[near] = True
                picks[position] = near

    out: list[RawHeading] = []
    for heading, chosen in zip(headings, picks, strict=True):
        if chosen is not None:
            flags = list(heading.flags)
            if _BOOKMARK_FLAG not in flags:
                flags.append(_BOOKMARK_FLAG)
            heading = heading.model_copy(update={"level": bookmarks[chosen][0], "flags": flags})
        out.append(heading)

    leftover = [
        bookmark
        for bookmark, done in zip(bookmarks, used, strict=True)
        if not done and isinstance(bookmark[2], int) and bookmark[2] >= 1
    ]
    if not leftover:
        return out
    pages = (
        [list(lines) for lines in line_pages]
        if line_pages is not None
        else [
            [(heading.line + 1 + k, heading.page) for k in range(len(heading.text.splitlines()))]
            for heading in headings
        ]
    )
    bodies: dict[int, _Body] = {}
    for level, title, page in leftover:
        _insert_heading(out, pages, bodies, level, title, page)
    return out


def _fuzzy_pick(
    key: str,
    page: int,
    bookmarks: Sequence[tuple[int, str, Any]],
    keys: list[str],
    used: list[bool],
) -> int | None:
    """The unused bookmark most similar to ``key`` (ratio at least 0.9) within a page of it,
    with the same numbering (:func:`_numbering`)."""
    if len(key) < _FUZZY_MIN_CHARS:
        return None
    numbering = _numbering(key)
    best: tuple[float, int, int] | None = None  # (ratio, -page distance, -index)
    for index, (_, _, bookmark_page) in enumerate(bookmarks):
        if used[index] or not isinstance(bookmark_page, int):
            continue
        distance = abs(bookmark_page - page)
        if distance > _FUZZY_PAGES:
            continue
        if len(keys[index]) < _FUZZY_MIN_CHARS or _numbering(keys[index]) != numbering:
            continue
        matcher = SequenceMatcher(None, key, keys[index])
        if matcher.real_quick_ratio() < _FUZZY_RATIO or matcher.quick_ratio() < _FUZZY_RATIO:
            continue
        ratio = matcher.ratio()
        if ratio >= _FUZZY_RATIO and (best is None or (ratio, -distance, -index) > best):
            best = (ratio, -distance, -index)
    return None if best is None else -best[2]


def _numbering(key: str) -> list[str]:
    """The numbering in a normalised title: every word with a digit, and a roman numeral or
    single letter right after a numbering word ("part ii") or a number ("7 a")."""
    out: list[str] = []
    previous = ""
    for word in key.split():
        if any(c.isdigit() for c in word):
            out.append(word)
        elif (previous in _NUMBERED_WORDS or any(c.isdigit() for c in previous)) and (
            len(word) == 1 or _ROMAN.fullmatch(word)
        ):
            out.append(word)
        previous = word
    return out


def _title_pattern(title: str) -> re.Pattern[str] | None:
    """Matches ``title`` alone on its line or lines in a body (see :func:`apply_bookmarks`)."""
    words = [word for word in _NON_WORD.split(title) if word]
    if not words:
        return None
    body = _TITLE_GAP.join(re.escape(word) for word in words)
    return re.compile(rf"^{_TITLE_EDGE}{body}{_TITLE_EDGE}$", re.IGNORECASE | re.MULTILINE)


class _Body:
    """A heading's body split into lines once, as ``str.splitlines`` counts them (as the reader
    and ``line_pages`` do). The lines are joined by "\\n", so a match's line is its "\\n" count."""

    def __init__(self, text: str, lines: list[tuple[int, int | None]]) -> None:
        self.kept = text.splitlines(keepends=True)
        self.plain = [line.rstrip("\r\n\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029") for line in self.kept]
        self.joined = "\n".join(self.plain)
        self.starts: list[int] = []  # where each line starts in ``joined``
        offset = 0
        for line in self.plain:
            self.starts.append(offset)
            offset += len(line) + 1
        # per page, the runs of consecutive lines on it as (first line, one past the last)
        self.runs: dict[int | None, list[tuple[int, int]]] = {}
        for index, (_, line_page) in enumerate(lines[: len(self.plain)]):
            runs = self.runs.setdefault(line_page, [])
            if runs and runs[-1][1] == index:
                runs[-1] = (runs[-1][0], index + 1)
            else:
                runs.append((index, index + 1))

    def find(self, pattern: re.Pattern[str], page: int, wrap: int) -> Iterator[tuple[int, int]]:
        """The first and last line of each match of ``pattern`` that starts on ``page``. Only
        those lines and the ``wrap`` after them are searched, so a title may wrap over lines."""
        count = len(self.plain)
        for run_start, run_end in self.runs.get(page, []):
            stop = min(run_end + wrap, count)
            end = len(self.joined) if stop == count else self.starts[stop] - 1
            for match in pattern.finditer(self.joined, self.starts[run_start], end):
                first = self.joined.count("\n", 0, match.start())
                if run_start <= first < run_end:
                    yield first, self.joined.count("\n", 0, match.end())


def _insert_heading(
    headings: list[RawHeading],
    pages: list[list[tuple[int, int | None]]],
    bodies: dict[int, _Body],
    level: int,
    title: str,
    page: int,
) -> None:
    """Split the first body holding ``title`` on ``page`` at it, in place (``headings`` and
    ``pages``). ``bodies`` keeps each heading's :class:`_Body` by ``id``."""
    pattern = _title_pattern(title)
    if pattern is None:
        return
    wrap = len(title.split()) + 1
    for position, heading in enumerate(headings):
        lines = pages[position]
        body = bodies.get(id(heading))
        if body is None:
            body = bodies[id(heading)] = _Body(heading.text, lines)
        for first, last in body.find(pattern, page, wrap):
            start = last + 1
            while start < len(body.plain) and not body.plain[start].strip():
                start += 1
            before = "".join(body.kept[:first]).rstrip()
            after = "".join(body.kept[start:]).strip()
            new = RawHeading(
                title=" ".join(title.split()),
                level=level,
                text=after,
                line=lines[first][0],
                page=page,
                flags=[_FROM_BOOKMARK_FLAG],
            )
            del bodies[id(heading)]
            headings[position] = heading.model_copy(update={"text": before})
            headings.insert(position + 1, new)
            pages[position] = lines[: len(before.splitlines())]
            pages.insert(position + 1, lines[start : start + len(after.splitlines())])
            return


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


def _text_line_pages(
    heading: RawHeading, lines: list[str], starts: list[int], pages: list[int]
) -> list[tuple[int, int | None]]:
    """The ``(line, page)`` of each line of ``heading.text``, a stripped slice of ``lines``.

    A heading's body starts after its line (a preamble's on line 1, a setext heading's after
    the underline); the text starts at the body's first non-blank line.
    """
    text_lines = heading.text.splitlines()
    if not text_lines:
        return []
    first = text_lines[0].strip()
    index = heading.line if heading.level > 0 else 0  # 0-based index of the line after it
    while index < len(lines) and lines[index].strip() != first:
        index += 1
    count = len(text_lines)
    return [
        (index + k + 1, pages[bisect_right(starts, index + k + 1) - 1] if starts else None)
        for k in range(count)
    ]


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
    level jumps (:func:`~docrooter.document.headings.repair_levels`). With ``use_bookmarks`` a
    heading that matches a PDF bookmark, exactly or nearly, takes the bookmark's level, and a
    bookmarked heading found only in body text becomes a heading (:func:`apply_bookmarks`);
    the repair then keeps the bookmark levels and nests the other headings around them.
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
    lines = markdown.splitlines()
    raw: list[RawHeading] = []
    line_pages: list[list[tuple[int, int | None]]] = []  # (line, page) of each line of a text
    for heading in extract_headings(markdown):
        page = pages[bisect_right(starts, heading.line) - 1] if starts else None
        text_pages = _text_line_pages(heading, lines, starts, pages)
        if heading.level == 0:
            raw.append(heading.model_copy(update={"page": page}))
            line_pages.append(text_pages)
            continue
        clean = _clean_title(heading.title)
        if not _HAS_WORD.search(clean):  # an emptied heading: its text joins the one before
            if raw:
                joined = "\n".join(t for t in (raw[-1].text, heading.text) if t)
                raw[-1] = raw[-1].model_copy(update={"text": joined})
                line_pages[-1] = line_pages[-1] + text_pages
            continue
        raw.append(heading.model_copy(update={"title": clean, "page": page}))
        line_pages.append(text_pages)
    if toc:
        raw = apply_bookmarks(raw, toc, line_pages=line_pages)
    if repair:  # after the bookmarks, so the headings around a bookmarked one nest by it
        raw = repair_levels(raw, keep_flags=(_BOOKMARK_FLAG, _FROM_BOOKMARK_FLAG))
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
