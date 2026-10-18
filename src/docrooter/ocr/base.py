"""The OCR engine protocol and the conversion of OCR output into ``Line`` objects."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from docrooter.elements.models import Box
from docrooter.pdf.lines import Line, _finalise, _has_alnum

__all__ = ["OcrEngine", "OcrLine", "ocr_lines_to_lines"]

_BLOCK_GAP_FACTOR = 0.5  # a gap above this fraction of the line height starts a new block
_CAP_TO_FONT = 0.75  # box height (ascender to descender) to font size


class OcrLine(BaseModel):
    """One recognised line of text, its box in page points (origin top left) and a confidence
    in ``0..1``."""

    text: str
    box: Box
    confidence: float


@runtime_checkable
class OcrEngine(Protocol):
    """A text recogniser.

    ``recognize`` takes an encoded image (PNG) of ``width_px`` x ``height_px`` pixels that
    shows the page rectangle ``page`` (in points), and returns the lines it read with boxes
    already scaled to page points. Engines return lines in their own reading order.
    ``close()`` releases models and handles; it is safe to call twice.
    """

    name: str

    def recognize(
        self, image: bytes, *, width_px: int, height_px: int, page: Box
    ) -> list[OcrLine]: ...

    def close(self) -> None: ...


def scale_box(
    page: Box, width_px: int, height_px: int, x0: float, y0: float, x1: float, y1: float
) -> Box:
    """A pixel rectangle of the rendered image as a page-points ``Box``. The image shows the
    rectangle ``page``; the pixel origin is its top left."""
    sx = (page.x1 - page.x0) / max(width_px, 1)
    sy = (page.y1 - page.y0) / max(height_px, 1)
    return Box(
        page=page.page,
        x0=page.x0 + min(x0, x1) * sx,
        y0=page.y0 + min(y0, y1) * sy,
        x1=page.x0 + max(x0, x1) * sx,
        y1=page.y0 + max(y0, y1) * sy,
    )


def ocr_lines_to_lines(ocr: list[OcrLine], page: int, width: float, height: float) -> list[Line]:
    """Turn OCR output into ``Line`` objects for ``page`` (1-based) of ``width`` x ``height`` pt.

    ``size`` is the box height x 0.75, ``font`` is ``"ocr"`` and ``bold`` is ``False``. The
    engine's order is kept (the XY-cut orders the page later); vertical gaps and block numbers
    are computed as the pdfplumber extractor does, a gap above half the line height starting a
    new block. Lines without a letter or digit are dropped, as the extractors do. ``Line``
    carries no source: the caller tracks that these lines came from OCR.
    """
    rows: list[dict[str, Any]] = []
    for item in ocr:
        text = " ".join(item.text.split())
        if not _has_alnum(text):
            continue
        b = item.box
        rows.append({
            "page": page, "text": text, "size": round(max(b.y1 - b.y0, 0.0) * _CAP_TO_FONT, 1),
            "bold": False, "font": "ocr", "x0": b.x0, "y0": b.y0, "x1": b.x1, "y1": b.y1,
            "page_height": height, "page_width": width, "block": 0,
        })  # fmt: skip
    block = 0
    for i in range(1, len(rows)):
        gap = rows[i]["y0"] - rows[i - 1]["y1"]
        if gap > _BLOCK_GAP_FACTOR * max(rows[i]["y1"] - rows[i]["y0"], 1.0):
            block += 1
        rows[i]["block"] = block
    return _finalise(rows)
