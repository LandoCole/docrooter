"""PDF loader (stub until the PDF task lands)."""

from __future__ import annotations

import os

from docrooter._extras import require
from docrooter.document.models import Document


def load_pdf(
    path: str | os.PathLike[str], *, min_heading_score: float = 0.5, repair: bool = True
) -> Document:
    """Load a PDF into a :class:`Document`. Requires the ``pdf`` extra."""
    require("pymupdf4llm", "pdf")
    raise NotImplementedError("PDF loading is not implemented yet")
