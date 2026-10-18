"""Apple Vision OCR through ``ocrmac`` (macOS only, ``pip install 'docrooter[ocr-mac]'``)."""

from __future__ import annotations

import importlib
import io

from docrooter._extras import require
from docrooter.elements.models import Box
from docrooter.ocr.base import OcrLine, scale_box

__all__ = ["OcrmacEngine"]

# Vision wants BCP 47 tags; map the common bare codes.
_LANG = {
    "en": "en-US", "de": "de-DE", "fr": "fr-FR", "es": "es-ES", "it": "it-IT", "pt": "pt-BR",
    "zh": "zh-Hans", "ja": "ja-JP", "ko": "ko-KR", "ru": "ru-RU", "uk": "uk-UA",
}  # fmt: skip


class OcrmacEngine:
    """``recognition_level="accurate"`` Apple Vision. It needs no model download. Vision
    reports a confidence of 1.0 for most lines."""

    name = "ocrmac"

    def __init__(self, *, lang: str = "en") -> None:
        require("ocrmac", "ocr-mac")
        self._mod = importlib.import_module("ocrmac.ocrmac")
        self._pil = require("PIL.Image", "ocr-mac")
        self._lang = _LANG.get(lang, lang)

    def recognize(self, image: bytes, *, width_px: int, height_px: int, page: Box) -> list[OcrLine]:
        pil = self._pil.open(io.BytesIO(image))
        result = self._mod.OCR(
            pil, recognition_level="accurate", language_preference=[self._lang]
        ).recognize(px=True)  # (text, confidence, (x0, y0, x1, y1)) in pixels, origin top left
        lines = []
        for text, confidence, (x0, y0, x1, y1) in result:
            if text.strip():
                box = scale_box(page, width_px, height_px, x0, y0, x1, y1)
                lines.append(OcrLine(text=text, box=box, confidence=float(confidence)))
        return lines

    def close(self) -> None:
        pass
