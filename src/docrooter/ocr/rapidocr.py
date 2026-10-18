"""RapidOCR (PaddleOCR models on onnxruntime), ``pip install 'docrooter[ocr]'``."""

from __future__ import annotations

from typing import Any

from docrooter._extras import require
from docrooter.elements.models import Box
from docrooter.ocr.base import OcrLine, scale_box

__all__ = ["RapidOcrEngine"]


class RapidOcrEngine:
    """``RapidOCR()`` with its default models, created on first use and dropped by ``close()``.

    The default models read English and Chinese. ``lang`` is accepted for the common interface
    but does not select other models: they would be downloaded at run time, which docrooter
    does not do silently.
    """

    name = "rapidocr"

    def __init__(self, *, lang: str = "en") -> None:
        self._mod = require("rapidocr", "ocr")
        self._lang = lang
        self._engine: Any = None

    def recognize(self, image: bytes, *, width_px: int, height_px: int, page: Box) -> list[OcrLine]:
        if self._engine is None:
            # default models; only the logging is changed (it prints every model path)
            self._engine = self._mod.RapidOCR(params={"Global.log_level": "warning"})
        out = self._engine(image)
        if out.boxes is None or out.txts is None:
            return []
        scores = out.scores if out.scores is not None else [1.0] * len(out.txts)
        lines = []
        for quad, text, score in zip(out.boxes, out.txts, scores, strict=False):
            if not text.strip():
                continue
            xs = [float(p[0]) for p in quad]
            ys = [float(p[1]) for p in quad]
            box = scale_box(page, width_px, height_px, min(xs), min(ys), max(xs), max(ys))
            lines.append(OcrLine(text=text, box=box, confidence=float(score)))
        return lines

    def close(self) -> None:
        self._engine = None
