"""Layout regions, the layout-engine protocol, and how regions type the lines (spec §6).

A layout engine finds labelled regions on rendered pages. Each line whose centre falls inside a
region (with at least ``MIN_CONFIDENCE``) takes the region's element kind; a line inside several
regions goes to the most confident one, then the smallest. Lines outside every region keep the
heuristic rules.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Container, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from docrooter.elements.models import Box, ElementKind
from docrooter.pdf.lines import Line

__all__ = [
    "HERON_FLAGS",
    "HERON_TO_KIND",
    "MIN_CONFIDENCE",
    "LayoutEngine",
    "LayoutRegion",
    "apply_regions",
    "region_kind",
]

MIN_CONFIDENCE = 0.5  # spec §6: regions below this are not used

# Spec §6 (with the DocLayNet mapping of spec §4). Labels are the export's ``id2label`` names.
HERON_TO_KIND: dict[str, ElementKind] = {
    "caption": "caption",
    "footnote": "footnote",
    "formula": "formula",
    "list_item": "list_item",
    "page_footer": "page_footer",
    "page_header": "page_header",
    "picture": "figure",
    "section_header": "heading",
    "table": "table",
    "text": "paragraph",
    "title": "title",
    "document_index": "contents",
    "code": "paragraph",
    "checkbox_selected": "paragraph",
    "checkbox_unselected": "paragraph",
    "form": "paragraph",
    "key_value_region": "paragraph",
}
# The flag an element of these labels carries (spec §6: Code, Checkbox, Form, Key-Value Region).
HERON_FLAGS: dict[str, str] = {
    "code": "code",
    "checkbox_selected": "form",
    "checkbox_unselected": "form",
    "form": "form",
    "key_value_region": "form",
}


@dataclass(frozen=True)
class LayoutRegion:
    """A region found by a layout engine: its page (1-based), its box in the displayed page's
    points (origin top left, like ``Line``), the engine's label and its confidence."""

    page: int
    box: Box
    label: str
    confidence: float

    @property
    def area(self) -> float:
        return max(self.box.x1 - self.box.x0, 0.0) * max(self.box.y1 - self.box.y0, 0.0)


@runtime_checkable
class LayoutEngine(Protocol):
    """Finds layout regions on PDF pages. ``detect`` renders ``pages`` (1-based) of ``path`` at
    ``dpi`` and returns their regions; ``close`` releases the model."""

    name: str

    def detect(self, path: str | Path, pages: Sequence[int], *, dpi: int) -> list[LayoutRegion]: ...

    def close(self) -> None: ...


def region_kind(label: str) -> tuple[ElementKind, list[str]] | None:
    """The element kind and flags of a region label (``None`` for a label with no mapping)."""
    kind = HERON_TO_KIND.get(label)
    if kind is None:
        return None
    flag = HERON_FLAGS.get(label)
    return kind, [flag] if flag else []


def apply_regions(
    lines: Sequence[Line],
    regions: Sequence[LayoutRegion],
    *,
    min_confidence: float = MIN_CONFIDENCE,
    skip: Container[int] = (),
) -> tuple[dict[int, ElementKind], list[tuple[LayoutRegion, list[int]]]]:
    """Assign lines to regions.

    Returns ``kind_by_line`` (line index to the kind of the region that took it) and
    ``regions_used``: every region with a known label and at least ``min_confidence``, with the
    indexes of the lines it took (ascending, possibly none), ordered by page, top, then left.
    A line is taken by a region when its centre lies inside the region's box; with several such
    regions the most confident wins, then the smallest. Lines in ``skip`` and lines without
    text are left alone.
    """
    used = [r for r in regions if r.confidence >= min_confidence and r.label in HERON_TO_KIND]
    used.sort(key=lambda r: (r.page, r.box.y0, r.box.x0, -r.confidence))
    by_page: defaultdict[int, list[int]] = defaultdict(list)
    for k, region in enumerate(used):
        by_page[region.page].append(k)
    members: list[list[int]] = [[] for _ in used]
    kind_by_line: dict[int, ElementKind] = {}
    for i, line in enumerate(lines):
        if i in skip or not line.text.strip():
            continue
        cx, cy = (line.x0 + line.x1) / 2, (line.y0 + line.y1) / 2
        best: int | None = None
        for k in by_page.get(line.page, ()):
            b = used[k].box
            if not (b.x0 <= cx <= b.x1 and b.y0 <= cy <= b.y1):
                continue
            if best is None or _beats(used[k], used[best]):
                best = k
        if best is not None:
            members[best].append(i)
            kind_by_line[i] = HERON_TO_KIND[used[best].label]
    return kind_by_line, list(zip(used, members, strict=True))


def _beats(a: LayoutRegion, b: LayoutRegion) -> bool:
    """Overlap resolution (spec §6): higher confidence, then smaller area."""
    if a.confidence != b.confidence:
        return a.confidence > b.confidence
    return a.area < b.area
