"""The pinned heron ONNX export and its I/O contract, from the repository's own files.

``docling-project/docling-layout-heron-onnx`` (Apache-2.0) at ``HERON_REVISION``:

- ``config.json``: ``RTDetrV2ForObjectDetection`` with the 17 classes of ``HERON_LABELS``
  (``id2label`` order);
- ``preprocessor_config.json``: ``RTDetrImageProcessor``, resize to 640 x 640 (``resample`` 2,
  bilinear, aspect ratio not kept), ``do_rescale`` and ``do_normalize`` both false: the graph
  takes the ``uint8`` RGB pixels as they are and scales them itself, and no mean or std is
  applied (the ``image_mean``/``image_std`` listed there are unused);
- ``model.onnx`` (171,220,471 bytes): inputs ``images`` ``uint8[batch, 3, 640, 640]`` and
  ``orig_target_sizes`` ``int64[batch, 2]`` as ``(width, height)`` of the source image;
  outputs ``labels`` ``int64[batch, 300]`` (indexes into ``HERON_LABELS``), ``boxes``
  ``float32[batch, 300, 4]`` (``x0, y0, x1, y1`` in the pixels of ``orig_target_sizes``) and
  ``scores`` ``float32[batch, 300]`` (sigmoid class scores; the top 300 query-class pairs, so
  one box can come back under two labels). The decoding is in the graph.

The ``(width, height)`` order and the label indexes were checked on a generated page (task E6
report): with ``(height, width)`` the boxes come out transposed.
"""

from __future__ import annotations

__all__ = [
    "HERON_FILE",
    "HERON_INPUT_SIZE",
    "HERON_LABELS",
    "HERON_RENDER_DPI",
    "HERON_REPO",
    "HERON_REVISION",
]

HERON_REPO = "docling-project/docling-layout-heron-onnx"
# HfApi().model_info(HERON_REPO).sha on 2026-10-18
HERON_REVISION = "40bde044036bb181c130ddf6c51792187268748f"
HERON_FILE = "model.onnx"
HERON_INPUT_SIZE = (640, 640)  # (width, height) the page image is resized to
HERON_RENDER_DPI = 72  # render at scale 1 (a letter page is 612 x 792 px), then resize to 640
HERON_LABELS: tuple[str, ...] = (
    "caption",
    "footnote",
    "formula",
    "list_item",
    "page_footer",
    "page_header",
    "picture",
    "section_header",
    "table",
    "text",
    "title",
    "document_index",
    "code",
    "checkbox_selected",
    "checkbox_unselected",
    "form",
    "key_value_region",
)
