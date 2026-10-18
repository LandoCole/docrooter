"""Reading order for a page's lines: iterative recursive XY-cut (spec §5.2).

Full-width boxes split the page into horizontal bands; inside a band, vertical whitespace
valleys (column gutters) are cut first, then horizontal gaps, and the pieces are visited
left to right or top to bottom. The work is driven by an explicit stack (no recursion).

A gutter only counts as a column boundary when each side holds at least two boxes that are
20% of the text width or wider: a label column beside a column of figures (a ledger, a contents
list) is a table, not two text columns, and is left in the extractor's order. This guard is an
addition to the §5.2 rules, made so that single-column pages with tables stay unchanged.

A page with no column gutter anywhere keeps the extractor's order exactly: the cut only
reorders when it finds columns, so single-column pages never change.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence

from docrooter.pdf.lines import Line

__all__ = ["order_lines", "xy_cut_order"]

Box = tuple[float, float, float, float]  # x0, y0, x1, y1; points, origin top-left

_FULL_WIDTH = 0.75  # a box at least this share of the text width separates bands
_GUTTER_MIN = 8.0  # pt
_GUTTER_CHARS = 3.0  # a gutter is at least this many median character widths wide
_GUTTER_COVERAGE = 0.6  # share of the band height the two sides must share
_GAP_FACTOR = 1.5  # a horizontal cut needs a gap of this many median line gaps
_COLUMN_WIDTH = 0.2  # a column holds text boxes at least this share of the text width wide
_COLUMN_MIN_BOXES = 2  # ... at least this many of them, on each side of a gutter


def xy_cut_order(
    boxes: Sequence[Box],
    *,
    page_width: float,
    text_width: float | None = None,
    median_char_width: float,
    median_line_gap: float,
) -> list[int]:
    """A permutation of ``range(len(boxes))`` in reading order.

    Boxes spanning at least 75% of ``text_width`` (default: the union width of the boxes,
    capped at ``page_width``) split the page into bands. Inside a band, a vertical gutter is a
    gap of at least ``max(8, 3 * median_char_width)`` between boxes whose two sides share at
    least 60% of the band's height and that each hold at least two boxes 20% of the text width
    or wider (so label/figure tables are not columns); failing that, a horizontal gap of at least
    ``1.5 * median_line_gap`` is cut. Within a cell that cannot be cut, the input order is kept.
    If no gutter is found anywhere the input order is returned unchanged.
    """
    n = len(boxes)
    if n < 2:
        return list(range(n))
    if text_width is None:
        text_width = max(b[2] for b in boxes) - min(b[0] for b in boxes)
        if page_width > 0:
            text_width = min(text_width, page_width)
    min_gutter = max(_GUTTER_MIN, _GUTTER_CHARS * median_char_width)
    min_gap = _GAP_FACTOR * median_line_gap

    order: list[int] = []
    cut_columns = False
    stack: list[list[int]] = [list(range(n))]
    runs = _bands(boxes, text_width)
    if len(runs) > 1:
        stack = runs[::-1]
    while stack:
        group = stack.pop()
        if len(group) < 2:
            order.extend(group)
            continue
        parts = _column_cut(boxes, group, min_gutter, text_width)
        if parts is not None:
            cut_columns = True
        else:
            parts = _row_cut(boxes, group, min_gap)
        if parts is None:
            order.extend(group)
        else:
            stack.extend(parts[::-1])
    return order if cut_columns else list(range(n))


def _bands(boxes: Sequence[Box], text_width: float) -> list[list[int]]:
    """Runs of full-width and of narrower boxes, top to bottom (input order within a run)."""
    limit = _FULL_WIDTH * text_width
    by_height = sorted(range(len(boxes)), key=lambda i: (boxes[i][1] + boxes[i][3], i))
    runs: list[list[int]] = []
    wide: bool | None = None
    for i in by_height:
        is_wide = text_width > 0 and boxes[i][2] - boxes[i][0] >= limit
        if is_wide != wide:
            runs.append([])
            wide = is_wide
        runs[-1].append(i)
    return [sorted(run) for run in runs]


def _column_cut(
    boxes: Sequence[Box], group: list[int], min_gutter: float, text_width: float
) -> list[list[int]] | None:
    """Cut ``group`` at every qualifying vertical gutter, or ``None`` if there is none."""
    by_x = sorted(group, key=lambda i: (boxes[i][0], boxes[i][2], i))
    n = len(by_x)
    top = min(boxes[i][1] for i in group)
    bottom = max(boxes[i][3] for i in group)
    height = bottom - top
    # y-extent of everything left of position k (prefix) and right of it (suffix)
    pre_lo, pre_hi = [0.0] * n, [0.0] * n
    suf_lo, suf_hi = [0.0] * n, [0.0] * n
    lo, hi = float("inf"), float("-inf")
    for k, i in enumerate(by_x):
        lo, hi = min(lo, boxes[i][1]), max(hi, boxes[i][3])
        pre_lo[k], pre_hi[k] = lo, hi
    lo, hi = float("inf"), float("-inf")
    for k in range(n - 1, -1, -1):
        i = by_x[k]
        lo, hi = min(lo, boxes[i][1]), max(hi, boxes[i][3])
        suf_lo[k], suf_hi[k] = lo, hi
    wide = _COLUMN_WIDTH * text_width
    pre_wide, suf_wide = [0] * n, [0] * n
    count = 0
    for k, i in enumerate(by_x):
        count += boxes[i][2] - boxes[i][0] >= wide
        pre_wide[k] = count
    count = 0
    for k in range(n - 1, -1, -1):
        i = by_x[k]
        count += boxes[i][2] - boxes[i][0] >= wide
        suf_wide[k] = count
    cuts: list[int] = []
    reach = boxes[by_x[0]][2]
    for k in range(1, n):
        i = by_x[k]
        if (
            boxes[i][0] - reach >= min_gutter
            and pre_wide[k - 1] >= _COLUMN_MIN_BOXES
            and suf_wide[k] >= _COLUMN_MIN_BOXES
        ):
            shared = min(pre_hi[k - 1], suf_hi[k]) - max(pre_lo[k - 1], suf_lo[k])
            if height <= 0 or shared >= _GUTTER_COVERAGE * height:
                cuts.append(k)
        reach = max(reach, boxes[i][2])
    if not cuts:
        return None
    bounds = [0, *cuts, n]
    return [sorted(by_x[a:b]) for a, b in zip(bounds, bounds[1:], strict=False)]


def _row_cut(boxes: Sequence[Box], group: list[int], min_gap: float) -> list[list[int]] | None:
    """Cut ``group`` at every horizontal gap of at least ``min_gap`` (and above zero)."""
    by_y = sorted(group, key=lambda i: (boxes[i][1], boxes[i][3], i))
    parts: list[list[int]] = [[by_y[0]]]
    reach = boxes[by_y[0]][3]
    for i in by_y[1:]:
        gap = boxes[i][1] - reach
        if gap > 0 and gap >= min_gap:
            parts.append([])
        parts[-1].append(i)
        reach = max(reach, boxes[i][3])
    if len(parts) == 1:
        return None
    return [sorted(part) for part in parts]


def order_lines(
    lines: Sequence[Line], excluded: set[int], groups: Mapping[int, Box] | None = None
) -> list[int]:
    """A document-wide permutation of ``range(len(lines))``, page by page.

    On each page the excluded lines (running headers, footers, page numbers) keep their
    original relative order at the page start (upper half of the page) or end (lower half),
    and the other lines follow :func:`xy_cut_order`.

    ``groups`` maps lines to the box of a layout region holding them (spec §6): the lines of
    one region move as one unit, ordered by the region's box, and keep their own order inside
    it; the other lines are units of their own. Without ``groups`` nothing changes.
    """
    pages: defaultdict[int, list[int]] = defaultdict(list)
    for i, line in enumerate(lines):
        pages[line.page].append(i)
    result: list[int] = []
    for page in sorted(pages):
        indexes = pages[page]
        head = [i for i in indexes if i in excluded and _in_top_half(lines[i])]
        tail = [i for i in indexes if i in excluded and not _in_top_half(lines[i])]
        body = [i for i in indexes if i not in excluded]
        result.extend(head)
        result.extend(_order_page(lines, body, groups or {}))
        result.extend(tail)
    return result


def _in_top_half(line: Line) -> bool:
    return (line.y0 + line.y1) / 2 < line.page_height / 2


def _order_page(lines: Sequence[Line], indexes: list[int], groups: Mapping[int, Box]) -> list[int]:
    if len(indexes) < 2:
        return indexes
    if any(i in groups for i in indexes):
        return _order_units(lines, indexes, groups)
    page = [lines[i] for i in indexes]
    widths = [ln.page_width for ln in page if ln.page_width > 0]
    char_widths = [(ln.x1 - ln.x0) / len(ln.text) for ln in page if ln.text and ln.x1 > ln.x0]
    gaps = [ln.gap_above for ln in page if 0 < ln.gap_above < float("inf")]
    local = xy_cut_order(
        [(ln.x0, ln.y0, ln.x1, ln.y1) for ln in page],
        page_width=max(widths, default=0.0),
        median_char_width=statistics.median(char_widths) if char_widths else 0.0,
        median_line_gap=statistics.median(gaps) if gaps else 0.0,
    )
    return [indexes[k] for k in local]


def _order_units(lines: Sequence[Line], indexes: list[int], groups: Mapping[int, Box]) -> list[int]:
    """:func:`_order_page` over units: each region's lines (first seen where its first line
    is) and each line outside the regions."""
    units: dict[object, list[int]] = {}
    boxes: list[Box] = []
    for i in indexes:
        key: object = ("region", groups[i]) if i in groups else ("line", i)
        if key not in units:
            units[key] = []
            line = lines[i]
            boxes.append(groups[i] if i in groups else (line.x0, line.y0, line.x1, line.y1))
        units[key].append(i)
    members = list(units.values())
    page = [lines[i] for i in indexes]
    widths = [ln.page_width for ln in page if ln.page_width > 0]
    char_widths = [(ln.x1 - ln.x0) / len(ln.text) for ln in page if ln.text and ln.x1 > ln.x0]
    gaps = [ln.gap_above for ln in page if 0 < ln.gap_above < float("inf")]
    local = xy_cut_order(
        boxes,
        page_width=max(widths, default=0.0),
        median_char_width=statistics.median(char_widths) if char_widths else 0.0,
        median_line_gap=statistics.median(gaps) if gaps else 0.0,
    )
    return [i for k in local for i in members[k]]
