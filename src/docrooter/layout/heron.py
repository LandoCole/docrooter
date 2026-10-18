"""Docling's heron layout model (RT-DETRv2, Apache-2.0) on onnxruntime: ``docrooter[layout]``.

No torch and no remote code: the ONNX weights are data, downloaded once through
``huggingface_hub`` at the pinned revision (see :mod:`docrooter.layout.recipes` for the I/O
contract). Pages are rendered at ``HERON_RENDER_DPI``, resized to 640 x 640 with a bilinear
filter (Pillow's, which the Hugging Face image processor uses, reimplemented in numpy), and
passed as ``uint8``; the graph rescales and decodes, so post-processing is a confidence cut and
the scaling of boxes from pixels to page points.

The inference session is created on the first ``detect`` and kept until ``close()``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from docrooter import runtime
from docrooter._extras import require
from docrooter.elements.models import Box
from docrooter.errors import MissingExtraError
from docrooter.layout.base import MIN_CONFIDENCE, LayoutRegion
from docrooter.layout.recipes import (
    HERON_FILE,
    HERON_INPUT_SIZE,
    HERON_LABELS,
    HERON_RENDER_DPI,
    HERON_REPO,
    HERON_REVISION,
)

__all__ = ["HeronLayout", "decode", "preprocess", "resize_bilinear"]

_EXTRA = "layout"


class HeronLayout:
    """The heron layout engine. ``providers`` are onnxruntime execution providers (default
    ``["CPUExecutionProvider"]``); the thread count follows ``docrooter.runtime`` settings
    (``$DOCROOTER_THREADS``) when set.

    Raises ``MissingExtraError`` when ``onnxruntime`` or ``huggingface_hub`` is not installed
    (at creation) or the weights cannot be fetched (on the first ``detect``).
    """

    name = "heron"

    def __init__(
        self,
        revision: str = HERON_REVISION,
        min_confidence: float = MIN_CONFIDENCE,
        *,
        providers: Sequence[str] | None = None,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}")
        self._ort = require("onnxruntime", _EXTRA)
        self._hub = require("huggingface_hub", _EXTRA)
        self.revision = revision
        self.min_confidence = min_confidence
        self.providers = list(providers) if providers is not None else ["CPUExecutionProvider"]
        self._session: Any = None

    def _load(self) -> Any:
        if self._session is None:
            try:
                weights = self._hub.hf_hub_download(HERON_REPO, HERON_FILE, revision=self.revision)
            except Exception as exc:
                raise MissingExtraError(
                    f"the heron layout weights ({HERON_REPO}@{self.revision}) could not be "
                    f"fetched ({type(exc).__name__}: {exc}); they are downloaded on first use "
                    f"with pip install 'docrooter[{_EXTRA}]'"
                ) from exc
            options = self._ort.SessionOptions()
            threads = runtime.settings().threads
            if threads:
                options.intra_op_num_threads = threads
            self._session = self._ort.InferenceSession(
                str(weights), sess_options=options, providers=self.providers
            )
        return self._session

    def detect(
        self, path: str | Path, pages: Sequence[int], *, dpi: int = HERON_RENDER_DPI
    ) -> list[LayoutRegion]:
        """The regions of ``pages`` (1-based) of ``path`` with at least ``min_confidence``,
        boxes in displayed-page points."""
        from docrooter.ocr.render import render_page_rgb

        session = self._load()
        out: list[LayoutRegion] = []
        for page in pages:
            rgb, width_pt, height_pt = render_page_rgb(path, page, dpi=dpi, extra=_EXTRA)
            height_px, width_px = rgb.shape[:2]
            labels, boxes, scores = session.run(
                ["labels", "boxes", "scores"],
                {
                    "images": preprocess(rgb),
                    "orig_target_sizes": np.array([[width_px, height_px]], dtype=np.int64),
                },
            )
            out += decode(
                labels[0], boxes[0], scores[0], page=page, width_pt=width_pt,
                height_pt=height_pt, width_px=width_px, height_px=height_px,
                min_confidence=self.min_confidence,
            )  # fmt: skip
        return out

    def close(self) -> None:
        """Release the inference session (safe to call twice; ``detect`` reloads it)."""
        self._session = None


def _filter_weights(n_in: int, n_out: int) -> np.ndarray:
    """``(n_out, n_in)`` weights of Pillow's bilinear (triangle) filter, widened when
    shrinking so every input pixel counts (antialiasing)."""
    scale = n_in / n_out
    support = max(scale, 1.0)
    centres = (np.arange(n_out) + 0.5) * scale
    pixels = np.arange(n_in) + 0.5
    weights = np.clip(1.0 - np.abs(pixels[None, :] - centres[:, None]) / support, 0.0, None)
    return (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)


def resize_bilinear(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """``rgb`` (``uint8``, height x width x channels) resized to ``height`` x ``width``."""
    h, w, c = rgb.shape
    rows = _filter_weights(h, height)
    cols = _filter_weights(w, width)
    # two matrix products (BLAS): rows over (h, w*c), then columns over each output row
    out = (rows @ rgb.reshape(h, w * c).astype(np.float32)).reshape(height, w, c)
    out = (out.transpose(0, 2, 1) @ cols.T).transpose(0, 2, 1)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """The model input for one page image: ``uint8[1, 3, 640, 640]`` RGB."""
    width, height = HERON_INPUT_SIZE
    resized = resize_bilinear(rgb[:, :, :3], width, height)
    return np.ascontiguousarray(resized.transpose(2, 0, 1)[None])


def decode(
    labels: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    page: int,
    width_pt: float,
    height_pt: float,
    width_px: int,
    height_px: int,
    min_confidence: float,
) -> list[LayoutRegion]:
    """One image's outputs as regions: scores of at least ``min_confidence``, known labels,
    boxes scaled from pixels to points and clipped to the page, empty boxes dropped; most
    confident first."""
    sx, sy = width_pt / max(width_px, 1), height_pt / max(height_px, 1)
    out: list[LayoutRegion] = []
    for k in np.argsort(-scores, kind="stable"):
        score, label = float(scores[k]), int(labels[k])
        if score < min_confidence:
            break
        if not 0 <= label < len(HERON_LABELS):
            continue
        x0, y0, x1, y1 = (float(v) for v in boxes[k])
        x0, x1 = sorted((min(max(x0 * sx, 0.0), width_pt), min(max(x1 * sx, 0.0), width_pt)))
        y0, y1 = sorted((min(max(y0 * sy, 0.0), height_pt), min(max(y1 * sy, 0.0), height_pt)))
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            continue
        box = Box(page=page, x0=round(x0, 2), y0=round(y0, 2), x1=round(x1, 2), y1=round(y1, 2))
        out.append(LayoutRegion(page=page, box=box, label=HERON_LABELS[label],
                                confidence=round(score, 4)))  # fmt: skip
    return out
