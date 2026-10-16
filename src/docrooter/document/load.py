"""Load a document from a path or Markdown text."""

from __future__ import annotations

import os
from pathlib import Path

from docrooter.document.build import build_document
from docrooter.document.headings import repair_levels, validate_headings
from docrooter.document.markdown import extract_headings
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
    source: str | os.PathLike[str], *, min_heading_score: float = 0.5, repair: bool = True
) -> Document:
    """Load a document and build its section tree.

    A :class:`~pathlib.Path` (or any ``os.PathLike``) is always a file path. A ``str`` is treated
    as Markdown text when it contains a newline; otherwise it is a path if it names an existing
    file, a missing file (``FileNotFoundError``) if it has a known extension (``.md``,
    ``.markdown``, ``.txt``, ``.pdf``), and Markdown text in every other case.
    ``.pdf`` files go to :func:`docrooter.document.pdf.load_pdf`; other files are read as UTF-8
    Markdown. Headings are extracted, validated (scores below ``min_heading_score`` are merged
    into the preceding text), and, if ``repair`` is true, have their levels repaired.
    """
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

    if path is None:
        return _build(str(source), _TEXT_SOURCE, None, min_heading_score, repair)

    if not _is_file(path):
        raise FileNotFoundError(f"No such file: {path}")
    if path.suffix.lower() == ".pdf":
        return load_pdf(path, min_heading_score=min_heading_score, repair=repair)
    text = path.read_text(encoding="utf-8-sig")
    return _build(text, str(source), path.stem, min_heading_score, repair)


def _build(
    text: str, source: str, fallback_title: str | None, min_heading_score: float, repair: bool
) -> Document:
    raw = validate_headings(extract_headings(text), min_score=min_heading_score)
    if repair:
        raw = repair_levels(raw)
    title = next((h.title for h in raw if h.level == 1), fallback_title)
    return build_document(raw, source=source, title=title)
