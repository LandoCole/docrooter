"""The OCR engine registry: built-in names, user engines and ``auto`` resolution."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from collections.abc import Callable

from docrooter._extras import install_hint
from docrooter.errors import DocrooterError
from docrooter.ocr.base import OcrEngine

__all__ = ["available_ocr_engines", "get_ocr_engine", "register_ocr_engine", "resolve_auto_ocr"]

OcrFactory = Callable[..., OcrEngine]  # called as factory(lang=...)


def _ocrmac(*, lang: str = "en") -> OcrEngine:
    from docrooter.ocr.ocrmac import OcrmacEngine

    return OcrmacEngine(lang=lang)


def _rapidocr(*, lang: str = "en") -> OcrEngine:
    from docrooter.ocr.rapidocr import RapidOcrEngine

    return RapidOcrEngine(lang=lang)


def _tesseract(*, lang: str = "en") -> OcrEngine:
    from docrooter.ocr.tesseract import TesseractEngine

    return TesseractEngine(lang=lang)


_FACTORIES: dict[str, OcrFactory] = {
    "ocrmac": _ocrmac,
    "rapidocr": _rapidocr,
    "tesseract": _tesseract,
}


def register_ocr_engine(name: str, factory: OcrFactory) -> None:
    """Register an OCR engine under ``name`` (replacing any engine of that name).
    ``factory(lang=...)`` returns an :class:`~docrooter.ocr.base.OcrEngine`."""
    if not name or name in ("auto", "none"):
        raise DocrooterError(f"{name!r} is not a valid OCR engine name")
    _FACTORIES[name] = factory


def available_ocr_engines() -> list[str]:
    """The registered engine names (built in and user-registered), in registration order.
    A name can still fail to load when its package or binary is missing."""
    return list(_FACTORIES)


def get_ocr_engine(name: str, *, lang: str = "en") -> OcrEngine:
    """A new engine for ``name``; ``"auto"`` picks one with :func:`resolve_auto_ocr`.

    Raises ``DocrooterError`` for an unknown name (the message lists the registered ones) and
    ``MissingExtraError`` when the engine's package is not installed.
    """
    if name == "auto":
        resolved = resolve_auto_ocr()
        if resolved is None:
            raise DocrooterError(
                f"no OCR engine is available: {install_hint('ocr')} (the ocr-mac extra on "
                "macOS) or install the tesseract binary"
            )
        name = resolved
    try:
        factory = _FACTORIES[name]
    except KeyError:
        known = ", ".join(available_ocr_engines())
        raise DocrooterError(f"unknown OCR engine {name!r}; available: {known}") from None
    return factory(lang=lang)


def _has_module(*names: str) -> bool:
    try:
        return all(importlib.util.find_spec(n) is not None for n in names)
    except (ImportError, ValueError):
        return False


def resolve_auto_ocr() -> str | None:
    """The first usable engine: ``tesseract`` (the binary on PATH), then ``ocrmac`` (macOS), then
    ``rapidocr``; ``None`` when there is none. Tesseract is first for accuracy (character error
    rate 0.058, against 0.089 for rapidocr and 0.102 for ocrmac; ``docs/benchmarks-elements.md``).
    ocrmac comes before rapidocr for speed and memory (0.73 against 3.04 s per page, 1.2 against
    3.2 GiB), although rapidocr is the more accurate of the two."""
    if shutil.which("tesseract"):
        return "tesseract"
    if sys.platform == "darwin" and _has_module("ocrmac"):
        return "ocrmac"
    if _has_module("rapidocr", "onnxruntime"):
        return "rapidocr"
    return None
