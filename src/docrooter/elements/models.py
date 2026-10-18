"""The element model: the typed blocks a document is made of, in reading order."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

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
