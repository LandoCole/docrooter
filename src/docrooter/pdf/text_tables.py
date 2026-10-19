"""Unruled tables inside layout-model Table regions (spec §6): the engine's "text" table
strategy, restricted to the region.

The heuristic engine never runs the "text" strategy on a whole page (it finds tables in prose);
it runs only where a layout model has already said "table".
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from docrooter.elements.models import Box
from docrooter.pdf.lines import Engine, _import_pdfplumber, _import_pymupdf
from docrooter.pdf.objects import _is_table, _table_kwargs, _table_rows, visible_page

__all__ = ["find_text_tables"]

_log = logging.getLogger(__name__)

_Found = dict[tuple[int, int], list[list[str]]]
_PDFPLUMBER_TEXT = {"vertical_strategy": "text", "horizontal_strategy": "text"}


def find_text_tables(
    path: str | Path, engine: Engine, regions: Mapping[int, Sequence[Box]]
) -> _Found:
    """Cell rows of the table found with the "text" strategy inside each box of ``regions``
    (page to boxes in displayed-page points), keyed by ``(page, index of the box)``. A region
    where the search finds no table of at least two rows and two columns, or fails, is left out.
    """
    if not regions:
        return {}
    if engine == "pymupdf":
        return _pymupdf(path, regions)
    return _pdfplumber(path, regions)


def _best(found: Sequence[Any]) -> list[list[str]] | None:
    """The rows of the table with the most non-empty cells, if it is a table. The "text"
    strategy reads the leading between rows as rows of its own; they are dropped."""
    best: list[list[str]] | None = None
    best_cells = 0
    for table in found:
        rows = [row for row in _table_rows(table.extract()) if any(row)]  # no blank rows
        cells = sum(1 for row in rows for cell in row if cell.strip())
        if _is_table(rows) and cells > best_cells:
            best, best_cells = rows, cells
    return best


def _pymupdf(path: str | Path, regions: Mapping[int, Sequence[Box]]) -> _Found:
    pymupdf = _import_pymupdf()
    kwargs = _table_kwargs(pymupdf)
    out: _Found = {}
    with pymupdf.open(str(path), filetype="pdf") as doc:
        for number, boxes in regions.items():
            if not 1 <= number <= doc.page_count:
                continue
            page = doc[number - 1]
            for k, box in enumerate(boxes):
                try:
                    # find_tables works on the displayed page (it un-rotates internally), so
                    # the clip is in display space like the boxes it reports
                    clip = pymupdf.Rect(box.x0, box.y0, box.x1, box.y1)
                    rows = _best(page.find_tables(clip=clip, strategy="text", **kwargs).tables)
                except Exception:  # one odd region must not fail the document
                    _log.debug("text table search failed on page %d", number, exc_info=True)
                    continue
                if rows is not None:
                    out[(number, k)] = rows
    return out


def _pdfplumber(path: str | Path, regions: Mapping[int, Sequence[Box]]) -> _Found:
    pdfplumber = _import_pdfplumber()
    out: _Found = {}
    with pdfplumber.open(str(path)) as pdf:
        for number, boxes in regions.items():
            if not 1 <= number <= len(pdf.pages):
                continue
            # Region boxes are in display space (the visible box, as pymupdf); pdfplumber's
            # objects are offset from it on cropped or offset-origin pages.
            page, dx, dy = visible_page(pdf.pages[number - 1])
            for k, box in enumerate(boxes):
                try:
                    width, height = float(page.width), float(page.height)
                    bbox = (max(box.x0, 0.0) + dx, max(box.y0, 0.0) + dy,
                            min(box.x1, width) + dx, min(box.y1, height) + dy)  # fmt: skip
                    region = page.crop(bbox, strict=False)
                    rows = _best(region.find_tables(table_settings=_PDFPLUMBER_TEXT))
                except Exception:
                    _log.debug("text table search failed on page %d", number, exc_info=True)
                    continue
                if rows is not None:
                    out[(number, k)] = rows
    return out
