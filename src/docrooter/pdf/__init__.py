"""Native PDF TOC engine: line extraction and document signals."""

from docrooter.pdf.lines import Line, extract_lines
from docrooter.pdf.load import load_pdf
from docrooter.pdf.signals import (
    DocSignals,
    compute_signals,
    fuzzy_match,
    is_fuzzy_match,
    normalize_title,
    read_bookmarks,
)

__all__ = [
    "DocSignals",
    "Line",
    "compute_signals",
    "extract_lines",
    "fuzzy_match",
    "is_fuzzy_match",
    "load_pdf",
    "normalize_title",
    "read_bookmarks",
]
