"""OCR engines behind one protocol, page rendering and OCR-to-``Line`` conversion.

Engines are imported lazily: ``import docrooter.ocr`` loads no OCR package.
"""

from docrooter.ocr.base import OcrEngine, OcrLine, ocr_lines_to_lines
from docrooter.ocr.registry import (
    available_ocr_engines,
    get_ocr_engine,
    register_ocr_engine,
    resolve_auto_ocr,
)
from docrooter.ocr.render import render_page

__all__ = [
    "OcrEngine",
    "OcrLine",
    "available_ocr_engines",
    "get_ocr_engine",
    "ocr_lines_to_lines",
    "register_ocr_engine",
    "render_page",
    "resolve_auto_ocr",
]
