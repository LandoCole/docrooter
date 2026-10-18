"""Render a PDF page (or a part of it) to a PNG for OCR."""

from __future__ import annotations

import importlib
import struct
import zlib
from pathlib import Path
from typing import Any

from docrooter._extras import require
from docrooter.elements.models import Box
from docrooter.errors import DocrooterError

__all__ = ["render_page"]


def render_page(
    path: str | Path, page: int, *, dpi: int = 200, clip: Box | None = None
) -> tuple[bytes, int, int]:
    """Render ``page`` (1-based) of ``path`` at ``dpi`` and return ``(png, width_px, height_px)``.

    ``clip``, a box in the displayed page's points (``/Rotate`` applied, origin top left), renders
    only that rectangle. Uses ``pymupdf`` when it is installed and ``pypdfium2`` otherwise, so the
    MIT-only install (``pdf-mit`` plus ``ocr``) never imports pymupdf.
    """
    if dpi <= 0:
        raise DocrooterError(f"dpi must be positive, got {dpi}")
    try:
        pymupdf = importlib.import_module("pymupdf")
    except ImportError:
        return _render_pdfium(path, page, dpi, clip)
    return _render_pymupdf(pymupdf, path, page, dpi, clip)


def _render_pymupdf(
    pymupdf: Any, path: str | Path, page: int, dpi: int, clip: Box | None
) -> tuple[bytes, int, int]:
    with pymupdf.open(str(path)) as doc:
        if not 1 <= page <= len(doc):
            raise DocrooterError(f"page {page} is out of range (the PDF has {len(doc)} pages)")
        pg = doc[page - 1]
        rect = pymupdf.Rect(clip.x0, clip.y0, clip.x1, clip.y1) if clip is not None else None
        scale = dpi / 72.0
        pix = pg.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=rect, alpha=False)
        return pix.tobytes("png"), int(pix.width), int(pix.height)


def _render_pdfium(
    path: str | Path, page: int, dpi: int, clip: Box | None
) -> tuple[bytes, int, int]:
    pdfium = require("pypdfium2", "ocr")
    pdf = pdfium.PdfDocument(str(path))
    try:
        if not 1 <= page <= len(pdf):
            raise DocrooterError(f"page {page} is out of range (the PDF has {len(pdf)} pages)")
        scale = dpi / 72.0
        pixels = pdf[page - 1].render(scale=scale).to_numpy()  # height x width x channels, BGR(A)
    finally:
        pdf.close()
    if clip is not None:
        h, w = pixels.shape[:2]
        x0, x1 = sorted((round(clip.x0 * scale), round(clip.x1 * scale)))
        y0, y1 = sorted((round(clip.y0 * scale), round(clip.y1 * scale)))
        pixels = pixels[max(y0, 0) : min(max(y1, y0 + 1), h), max(x0, 0) : min(max(x1, x0 + 1), w)]
    height, width = pixels.shape[:2]
    rgb = pixels[:, :, 2::-1] if pixels.shape[2] >= 3 else pixels  # BGR(A) to RGB
    return _png(rgb, width, height), width, height


def _png(rgb: Any, width: int, height: int) -> bytes:
    """Encode an 8-bit RGB (or grey) array as PNG without needing Pillow."""
    channels = 1 if rgb.ndim == 2 or rgb.shape[2] == 1 else 3
    color_type = 0 if channels == 1 else 2
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
