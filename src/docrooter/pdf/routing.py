"""Page routing: does a page keep its text layer, or need OCR? (spec §7.4)

Everything here is pure: the metrics and the decision rules work on the engine-neutral
:class:`docrooter.pdf.objects.PageObjects`, so they are testable without a PDF and without an
OCR engine. Engine selection and profiles live in ``docrooter.rooter``; the route log
(``Document.meta["routes"]``) is a list of :class:`RouteDecision` dumps.

All thresholds are the spec's starting values, tuned only on train-split pages.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from docrooter.elements.models import Box
from docrooter.pdf.lines import Line
from docrooter.pdf.objects import PageObjects

__all__ = ["PageMetrics", "RouteDecision", "page_metrics", "route_page"]

_PT2_PER_IN2 = 72.0 * 72.0

# Whole-page OCR thresholds (spec §7.4, rule 1).
SCAN_IMAGE_COV = 0.75  # image_cov >= this ...
SCAN_CHAR_DENSITY = 2.0  # ... and char_density < this: a scanned page
GARBLED_MAX = 0.10  # garbled > this
INVISIBLE_MAX = 0.5  # invisible_text > this ...
INVISIBLE_GARBLED = 0.02  # ... and garbled > this: a hidden layer that is junk (see below)
# Region OCR (rule 2).
REGION_MIN_AREA = 0.05  # an image of at least this share of the page
MAX_IMAGES = 500  # image placements considered per page, largest first (the checks are quadratic)
# Page-level garble heuristics (a page also counts as garbled when one of them fires).
SYMBOL_MIN_CHARS = 50
SYMBOL_MAX_SHARE = 0.5  # more than this share of >= 50 chars non-alphanumeric
WORD_MIN_TOKENS = 20
WORD_MIN_SHARE = 0.5  # fewer than this share of >= 20 tokens dictionary-like
LATIN_MIN_SHARE = 0.5  # the dictionary rule needs at least this share of Latin letters

_VOWELS = frozenset("aeiouyàáâãäåæèéêëìíîïòóôõöøùúûüýÿœ")


class PageMetrics(BaseModel):
    char_density: float  # printable characters per square inch
    image_cov: float  # union area of the image placements over the page area, 0..1
    garbled: float  # share of garbled characters; 1.0 when a page-level heuristic fires
    invisible_text: float  # share of characters in render mode 3 (pymupdf only)
    latin_share: float  # share of the letters that are Latin; 0 when there are no letters


class RouteDecision(BaseModel):
    scope: Literal["document", "page", "region"]
    target: int | list[float] | None  # a page number, or a region box [x0, y0, x1, y1]
    task: str  # "text" or "ocr"
    engine: str  # "text" for the text layer; "auto" for OCR until the rooter resolves it
    reason: str
    metrics: dict[str, float] = Field(default_factory=dict)


def page_metrics(objs: PageObjects) -> PageMetrics:
    """The §7.4 metrics of one page."""
    return _measure(objs)[0]


def route_page(objs: PageObjects) -> tuple[Literal["text", "ocr"], list[Box], RouteDecision]:
    """Decide how to read one page.

    Returns ``("ocr", [], decision)`` for whole-page OCR and ``("text", regions, decision)``
    otherwise, where ``regions`` are the image placements to OCR for figure text (empty if
    none). A region decision has ``scope="region"`` and the union box of the regions as its
    target; its ``reason`` gives their number.
    """
    metrics, garble_reason, images, dropped = _measure(objs)
    log = metrics.model_dump()
    log["images_capped"] = float(dropped)  # image placements ignored beyond MAX_IMAGES
    whole = _whole_page_reason(metrics, garble_reason)
    if whole is not None:
        return (
            "ocr",
            [],
            RouteDecision(
                scope="page", target=objs.page, task="ocr", engine="auto", reason=whole, metrics=log
            ),
        )
    regions = _text_free_images(objs, images)
    if regions:
        union = [
            min(b.x0 for b in regions),
            min(b.y0 for b in regions),
            max(b.x1 for b in regions),
            max(b.y1 for b in regions),
        ]
        reason = f"region OCR: {len(regions)} image region(s) of >= 5% of the page with no text"
        return (
            "text",
            regions,
            RouteDecision(
                scope="region", target=union, task="ocr", engine="auto", reason=reason, metrics=log
            ),
        )
    return (
        "text",
        [],
        RouteDecision(
            scope="page",
            target=objs.page,
            task="text",
            engine="text",
            reason="text layer",
            metrics=log,
        ),
    )


def _whole_page_reason(metrics: PageMetrics, garble_reason: str) -> str | None:
    if metrics.image_cov >= SCAN_IMAGE_COV and metrics.char_density < SCAN_CHAR_DENSITY:
        return (
            f"scanned page: image coverage {metrics.image_cov:.2f}, "
            f"{metrics.char_density:.2f} chars per square inch"
        )
    if metrics.garbled > GARBLED_MAX:
        return f"garbled text layer: {garble_reason or f'{metrics.garbled:.2f} of characters'}"
    # A hidden text layer is an old OCR pass. The spec leaves "the invisible text is garbled"
    # open, and the 10% bar above would make this rule redundant, so a hidden layer is
    # distrusted at a lower garbled share.
    if metrics.invisible_text > INVISIBLE_MAX and metrics.garbled > INVISIBLE_GARBLED:
        return (
            f"invisible text layer ({metrics.invisible_text:.2f} of characters) "
            f"that is garbled ({metrics.garbled:.2f})"
        )
    return None


def _measure(objs: PageObjects) -> tuple[PageMetrics, str, list[Box], int]:
    """The metrics, the reason for a page-level garble heuristic that fired (else ""), the
    image placements considered and the number dropped by the ``MAX_IMAGES`` cap."""
    area = max(objs.width * objs.height, 0.0)
    chars = objs.char_count
    density = chars / (area / _PT2_PER_IN2) if area > 0 else 0.0
    garbled = objs.garbled_chars / chars if chars else 0.0
    text = "\n".join(line.text for line in objs.lines)
    latin_share = _latin_share(text)
    reason = _garble_heuristic(text, latin_share)
    if reason:
        garbled = 1.0
    images = sorted(objs.images, key=lambda b: (b.x1 - b.x0) * (b.y1 - b.y0), reverse=True)
    dropped = max(len(images) - MAX_IMAGES, 0)
    images = images[:MAX_IMAGES]
    metrics = PageMetrics(
        char_density=density,
        image_cov=_union_share(images, objs.width, objs.height),
        garbled=min(garbled, 1.0),
        invisible_text=min(objs.invisible_chars / chars, 1.0) if chars else 0.0,
        latin_share=latin_share,
    )
    return metrics, reason, images, dropped


# --- garble heuristics ---------------------------------------------------------------------


def _latin_share(text: str) -> float:
    letters = latin = 0
    for ch in text:
        if ch.isalpha():
            letters += 1
            o = ord(ch)
            latin += o < 0x250 or 0x1E00 <= o <= 0x1EFF
    return latin / letters if letters else 0.0


_LEADERS = re.compile(r"[.·…_\-\s]{3,}")


def _garble_heuristic(text: str, latin_share: float) -> str:
    """The reason a page's text counts as garbled beyond bad characters, or ``""``.

    1. More than half of >= 50 characters are neither alphanumeric nor space, after runs of
       three or more leader characters (``[.·…_-]`` and spaces) are removed. CJK, Cyrillic and
       Arabic letters are alphanumeric, so non-Latin pages pass.
    2. Fewer than half of >= 20 tokens are dictionary-like (alphabetic with a vowel, or
       numeric). No word list is used, and the rule applies only when at least half of the
       letters are Latin: scripts without vowels or spaces would always fail it.
    """
    # Leader runs (dots, middle dots, underscores, hyphens, spaced) fill contents pages; they are
    # layout, not garbage, so they leave the symbol test.
    chars = [ch for ch in _LEADERS.sub("", text) if not ch.isspace()]
    if len(chars) >= SYMBOL_MIN_CHARS:
        symbols = sum(1 for ch in chars if not ch.isalnum())
        if symbols / len(chars) > SYMBOL_MAX_SHARE:
            return f"{symbols / len(chars):.2f} of characters are symbols"
    if latin_share >= LATIN_MIN_SHARE:
        tokens = [core for core in (_core(tok) for tok in text.split()) if core]
        if len(tokens) >= WORD_MIN_TOKENS:
            good = sum(1 for core in tokens if _dictionary_like(core))
            if good / len(tokens) < WORD_MIN_SHARE:
                return f"only {good / len(tokens):.2f} of tokens look like words"
    return ""


_PUNCT = re.compile(r"[\W_]+")


def _core(token: str) -> str:
    """The token without punctuation: ``(1,204.5)`` -> ``12045``."""
    return _PUNCT.sub("", token)


def _dictionary_like(core: str) -> bool:
    if core.isdigit():
        return True
    return core.isalpha() and any(ch in _VOWELS for ch in core.lower())


# --- geometry ------------------------------------------------------------------------------


def _clip(box: Box, width: float, height: float) -> tuple[float, float, float, float] | None:
    x0, y0 = max(box.x0, 0.0), max(box.y0, 0.0)
    x1, y1 = min(box.x1, width), min(box.y1, height)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _union_share(boxes: list[Box], width: float, height: float) -> float:
    """The area of the union of ``boxes`` (clipped to the page) over the page area."""
    if width <= 0 or height <= 0:
        return 0.0
    rects = [r for r in (_clip(b, width, height) for b in boxes) if r is not None]
    if not rects:
        return 0.0
    xs = sorted({r[0] for r in rects} | {r[2] for r in rects})
    total = 0.0
    for left, right in zip(xs, xs[1:], strict=False):
        spans = sorted((r[1], r[3]) for r in rects if r[0] <= left and r[2] >= right)
        covered, top = 0.0, 0.0
        for y0, y1 in spans:
            y0 = max(y0, top)
            if y1 > y0:
                covered += y1 - y0
                top = y1
        total += covered * (right - left)
    return min(total / (width * height), 1.0)


def _text_free_images(objs: PageObjects, images: list[Box]) -> list[Box]:
    """Image placements of >= 5% of the page that contain no text-layer line. A line is
    inside when its centre is."""
    page_area = objs.width * objs.height
    if page_area <= 0:
        return []
    regions: list[Box] = []
    for image in images:
        rect = _clip(image, objs.width, objs.height)
        if rect is None or (rect[2] - rect[0]) * (rect[3] - rect[1]) < REGION_MIN_AREA * page_area:
            continue
        if any(_centre_inside(line, rect) for line in objs.lines):
            continue
        regions.append(image)
    return regions


def _centre_inside(line: Line, rect: tuple[float, float, float, float]) -> bool:
    cx, cy = (line.x0 + line.x1) / 2, (line.y0 + line.y1) / 2
    return rect[0] <= cx <= rect[2] and rect[1] <= cy <= rect[3]
