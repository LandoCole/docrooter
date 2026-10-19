"""The element model: the typed blocks a document is made of, in reading order."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, SerializerFunctionWrapHandler, model_serializer

ElementKind = Literal[
    "title",
    "heading",
    "paragraph",
    "list_item",
    "table",
    "figure",
    "caption",
    "footnote",
    "page_header",
    "page_footer",
    "page_number",
    "contents",
    "formula",
]


class Box(BaseModel):
    """A rectangle on one page, in points with the origin at the top left (like ``Line``)."""

    page: int  # 1-based
    x0: float
    y0: float
    x1: float
    y1: float


class TableData(BaseModel):
    rows: list[list[str]]  # cell text, row-major; ragged rows are padded with "" on output
    header_rows: int = 0

    def to_markdown(self) -> str:
        """A pipe table. The delimiter row follows the header (an empty header if there is none)."""
        if not self.rows:
            return ""
        width = max(len(row) for row in self.rows)
        lines = [
            "| " + " | ".join(_cell(c) for c in (row + [""] * (width - len(row)))) + " |"
            for row in self.rows
        ]
        if self.header_rows <= 0:  # Markdown tables need a header row, so add an empty one
            lines.insert(0, "| " + " | ".join([""] * width) + " |")
        lines.insert(1, "| " + " | ".join(["---"] * width) + " |")
        return "\n".join(lines)


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


class FigureData(BaseModel):
    source: Literal["image", "vector", "ocr_region", "layout_model"]
    text: str = ""  # labels and other text found inside the figure region


Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class TagResult(BaseModel):
    """One facet's tags on an element: the chosen labels and the evidence behind them."""

    labels: list[str]  # topic: the best leaf; function: every tag over its threshold
    probs: dict[str, Probability]  # per-label probability, calibrated when the bundle is
    confidence: Probability  # the tagger's confidence in ``labels`` as a whole
    source: str  # what produced the tags: a bundle, "embedding:<spec>", "teacher:<spec>"
    # Topic only: the unconstrained family distribution, before restricting to the
    # document's family (so doc-type evidence built on it is not circular).
    family_probs: dict[str, Probability] | None = None
    # The content fingerprint of the saved tagger bundle that produced the tags
    # (docrooter.tagger.element_tags.tagger_fingerprint); None when unknown, as for the
    # zero-shot and teacher taggers. select() guarantees recall only on tags whose fingerprint
    # is the one its selector was fitted with.
    fingerprint: str | None = None

    @model_serializer(mode="wrap")
    def _omit_empty_optionals(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        for key in ("family_probs", "fingerprint"):
            if getattr(self, key) is None:
                data.pop(key, None)
        return data


class Element(BaseModel):
    id: str  # "e0", "e1", … in reading order; stable for the same input and config
    kind: ElementKind
    text: str  # tables: cells joined row by row; figures: the caption-free label text
    boxes: list[Box]  # one box per page fragment; empty for Markdown
    order: int  # global reading-order index
    section_id: str | None  # the Section this element belongs to
    level: int | None = None  # heading level (headings and title only)
    list_depth: int | None = None
    marker: str | None = None  # list bullet or enumerator, footnote marker
    table: TableData | None = None
    figure: FigureData | None = None
    caption_for: str | None = None  # caption -> the figure or table element id
    caption_id: str | None = None  # figure or table -> its caption element id
    source: Literal["text", "ocr"] = "text"
    confidence: float | None = None  # OCR or layout-model confidence; None for text heuristics
    flags: list[str] = Field(default_factory=list)  # "continued", "low_confidence", "code", …
    tags: dict[str, TagResult] | None = None  # facet ("topic", "function") -> its tags

    @model_serializer(mode="wrap")
    def _omit_empty_tags(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # Keeps the JSON of untagged elements identical to v0.4's.
        data: dict[str, Any] = handler(self)
        if self.tags is None:
            data.pop("tags", None)
        return data
