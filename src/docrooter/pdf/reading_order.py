"""Reading order for a page's lines: column-first XY-cut (spec §5.2, E12 controller ruling).

The work is driven by an explicit stack (no recursion). A *region* starts as the whole page and
is handled in this order:

1. **Spanning boxes.** A box at least :data:`_SPANNING` of the text width wide is a separator.
   The separators, ordered by y, cut the region into bands (what is above the first one, between
   two of them, below the last), and each band is a region of its own. A line of a single-column
   page is spanning, so the short boxes between two lines never make columns.
2. **A section break under columns.** The topmost horizontal gap that no box crosses, that is
   at least :data:`_STRUCTURAL_GAP` median line heights (and not under ``1.5 x median line gap``)
   and that has a gutter above it and, below it, a box across that gutter, cuts the region in
   two first, so a table or a section under two columns is not taken for the continuation of
   one of them. (A gap below which everything stays inside the columns is inside them, as
   between their sections.)
3. **A gutter.** The boxes are projected onto x. A vertical valley between them is a gutter
   when it is at least ``max(8 pt, 3 x median character width)`` wide, each side holds at least
   two boxes (and two that are 20% of the text width or wider: a label column beside a column
   of figures is a table, not two text columns), the two sides' y-ranges overlap by at least
   50% of the shorter one, and the sides are not the label and figure columns of an unruled
   table (the row-aligned guard, E11b: rows must stay together; a side of figures, or any
   column of cells among three, is a table). A title or heading that crosses the gutter (at
   most :data:`_CROSSING` boxes, none beside the columns' content) is a separator. The left
   side is read before the right one, and each is a region again, so the left column comes
   first whatever the lengths of the columns and the headings above and below them.
4. **No gutter:** the region is cut horizontally at its largest gap of at least
   ``1.5 x median line gap`` (and above zero), and the two halves are regions. A region that
   cannot be cut keeps the input order.

A page with no gutter anywhere keeps the extractor's order exactly: the cut only reorders when
it finds columns, so pages without a gutter never change.
"""

from __future__ import annotations

import bisect
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence

from docrooter.pdf.lines import Line

__all__ = ["order_lines", "xy_cut_order"]

Box = tuple[float, float, float, float]  # x0, y0, x1, y1; points, origin top-left
Item = tuple[bool, list[int], int, int]  # a region (else a box to emit), its boxes, id, parent

_SPANNING = 0.6  # a box at least this share of the text width separates bands
_GUTTER_MIN = 8.0  # pt
_GUTTER_CHARS = 3.0  # a gutter is at least this many median character widths wide
_GUTTER_OVERLAP = 0.5  # the sides' y-ranges overlap by this share of the shorter one
_GAP_FACTOR = 1.5  # a horizontal cut needs a gap of this many median line gaps
_COLUMN_WIDTH = 0.2  # a column holds text boxes at least this share of the text width wide
_COLUMN_MIN_BOXES = 2  # ... at least this many of them, on each side of a gutter
_ROW_ALIGN = 0.2  # a partner's vertical centre is within this share of the median line height
_ROW_ALIGNED_SHARE = 0.5  # ... for at least this share of the boxes of the smaller side
_NUMERIC_SHARE = (
    0.6  # a side with at least this share of numeric boxes holds the figures of a table
)
_CELL_WIDTH = 0.1  # a side whose median box is narrower than this share of the text width is cells
_PRECHECK_BOXES = 10  # regions this big are first checked once, setting aside the most boxes
_STRUCTURAL_GAP = 1.25  # a gap of this many median line heights under columns is a section break
_TABLE_COLUMNS = 3  # a projection with this many columns is also checked column by column
_MAX_BREAKS = 8  # section breaks tried per region, from the top
_CROSSING = 3  # boxes that cross a gutter (titles, headings) and still leave it a gutter


def xy_cut_order(
    boxes: Sequence[Box],
    *,
    page_width: float,
    text_width: float | None = None,
    median_char_width: float,
    median_line_gap: float,
    numeric: Sequence[bool] | None = None,
) -> list[int]:
    """A permutation of ``range(len(boxes))`` in reading order (see the module docstring).

    ``text_width`` (default: the union width of the boxes, capped at ``page_width``) is the
    width of the page's text: spanning boxes, the column-width guard and the table guard are
    measured against it. ``numeric`` flags the boxes that hold figures. If no gutter is found
    anywhere the input order is returned unchanged.
    """
    return xy_cut(
        boxes,
        page_width=page_width,
        text_width=text_width,
        median_char_width=median_char_width,
        median_line_gap=median_line_gap,
        numeric=numeric,
    )[0]


def xy_cut(
    boxes: Sequence[Box],
    *,
    page_width: float,
    text_width: float | None = None,
    median_char_width: float,
    median_line_gap: float,
    numeric: Sequence[bool] | None = None,
    trace: list[tuple[int, int, list[int], str]] | None = None,
) -> tuple[list[int], bool]:
    """:func:`xy_cut_order` and whether a gutter was found anywhere (when none was, the order
    is the identity). ``trace``, if given, receives ``(region id, parent id, boxes, how)`` for
    every region in the order they are handled; ``how`` is ``"columns"`` (cut at a gutter),
    ``"rows"`` (cut at a gap), ``"leaf"``, or ``"separators"`` (the spanning and crossing boxes
    of a gutter region, ids negative). The first region is 0 with parent -1."""
    n = len(boxes)
    if n < 2:
        return list(range(n)), False
    if text_width is None:
        text_width = max(b[2] for b in boxes) - min(b[0] for b in boxes)
        if page_width > 0:
            text_width = min(text_width, page_width)
    min_gutter = max(_GUTTER_MIN, _GUTTER_CHARS * median_char_width)
    min_gap = _GAP_FACTOR * median_line_gap
    heights = [b[3] - b[1] for b in boxes]
    structural = max(min_gap, _STRUCTURAL_GAP * statistics.median(heights))

    limit = _SPANNING * text_width
    order: list[int] = []
    cut_columns = False
    next_id = 1
    stack: list[Item] = [(True, list(range(n)), 0, -1)]
    while stack:
        is_region, group, rid, parent = stack.pop()
        if not is_region or len(group) < 2:
            order.extend(group)
            if trace is not None and is_region:
                trace.append((rid, parent, group, "leaf"))
            continue
        spanning = [i for i in group if text_width > 0 and boxes[i][2] - boxes[i][0] >= limit]
        children: list[Item] = []
        split = None
        if spanning and len(spanning) < len(group):  # the spanning boxes cut the region in bands
            how = "bands"
            rest = [i for i in group if i not in set(spanning)]
            children = _bands(boxes, rest, [], spanning)
        elif (
            halves := _break_under_columns(
                boxes, group, structural, text_width, min_gutter, numeric
            )
        ) is not None:  # columns, then a gap that is a section break
            how = "rows"
            children = [(True, half, 0, rid) for half in halves]
        elif (split := _find_gutter(boxes, group, text_width, min_gutter, numeric)) is not None:
            cut_columns, how = True, "columns"
            children = _bands(boxes, *split)
        else:
            halves = _row_cut(boxes, group, min_gap)
            how = "rows" if halves is not None else "leaf"
            children = [(True, half, 0, rid) for half in halves or []]
        if trace is not None:
            trace.append((rid, parent, group, how))
            separators = spanning if how == "bands" else (split[2] if split else [])
            if separators:
                trace.append((-next_id, rid, sorted(separators), "separators"))
        if not children:
            order.extend(sorted(group))
            continue
        stack.extend(reversed(_numbered(children, rid, next_id)))
        next_id += len(children)
    return (order if cut_columns else list(range(n))), cut_columns


def _numbered(children: list[Item], parent: int, first_id: int) -> list[Item]:
    """``children`` with ids from ``first_id`` and the parent ``parent``."""
    return [
        (is_region, group, first_id + k, parent)
        for k, (is_region, group, _, _) in enumerate(children)
    ]


def _bands(
    boxes: Sequence[Box], left: list[int], right: list[int], separators: list[int]
) -> list[Item]:
    """The stack items, in reading order, of a region with a gutter: the bands between its
    separators and the separators themselves; with no separator, the two sides."""
    if not separators:
        return [(True, left, 0, 0), (True, right, 0, 0)]
    seps = sorted(separators, key=lambda i: ((boxes[i][1] + boxes[i][3]) / 2, boxes[i][0], i))
    centres = [(boxes[i][1] + boxes[i][3]) / 2 for i in seps]
    bands: list[list[int]] = [[] for _ in range(len(seps) + 1)]
    for i in sorted([*left, *right]):
        bands[bisect.bisect_left(centres, (boxes[i][1] + boxes[i][3]) / 2)].append(i)
    items: list[Item] = []
    for k, sep in enumerate(seps):
        if bands[k]:
            items.append((True, bands[k], 0, 0))
        items.append((False, [sep], 0, 0))
    if bands[-1]:
        items.append((True, bands[-1], 0, 0))
    return items


def _find_gutter(
    boxes: Sequence[Box],
    group: list[int],
    text_width: float,
    min_gutter: float,
    numeric: Sequence[bool] | None = None,
) -> tuple[list[int], list[int], list[int]] | None:
    """``(left, right, separators)`` for the first valid gutter of the region, or ``None``.

    The spanning boxes are separators and are left out of the x-projection. Titles and headings
    that cross a gutter are found by setting aside up to :data:`_CROSSING` of the topmost and of
    the bottommost boxes before projecting; a set-aside box that does cross the valley becomes a
    separator, one that does not stays on its side.
    """
    limit = _SPANNING * text_width
    spanning = [i for i in group if text_width > 0 and boxes[i][2] - boxes[i][0] >= limit]
    spanning_set = set(spanning)
    body = [i for i in group if i not in spanning_set]
    if len(body) < 2 * _COLUMN_MIN_BOXES:
        return None
    by_y = sorted(body, key=lambda i: ((boxes[i][1] + boxes[i][3]) / 2, i))
    # Valleys of what is left once up to three of the topmost and of the bottommost boxes are set
    # aside (a title or heading that crosses a gutter no longer fills it), least set aside first.
    core = (
        _valleys(boxes, by_y[_CROSSING : len(by_y) - _CROSSING], min_gutter)
        if len(by_y) > 2 * _CROSSING
        else []
    )
    if len(by_y) > _PRECHECK_BOXES and not core:
        return None  # a big region without a valley once the most boxes are set aside has none
    combos = sorted(
        ((p, q) for p in range(_CROSSING + 1) for q in range(_CROSSING + 1)),
        key=lambda pq: (sum(pq), pq),
    )
    tried: set[tuple[float, float]] = set()
    for p, q in combos:
        if len(by_y) - p - q < 2:
            continue
        valleys = _valleys(boxes, by_y[p : len(by_y) - q], min_gutter)
        columns = core if len(core) > len(valleys) else valleys  # the table check's columns
        for valley in valleys:
            if valley in tried:
                continue
            tried.add(valley)
            found = _validate(boxes, body, valley, columns, text_width, numeric)
            if found is not None:
                left, right, crossing = found
                return left, right, [*spanning, *crossing]
    return None


def _valleys(
    boxes: Sequence[Box], members: list[int], min_gutter: float
) -> list[tuple[float, float]]:
    """The x-intervals of at least ``min_gutter`` that no box of ``members`` covers, left to
    right."""
    by_x = sorted(members, key=lambda i: (boxes[i][0], boxes[i][2], i))
    out: list[tuple[float, float]] = []
    if not by_x:
        return out
    reach = boxes[by_x[0]][2]
    for i in by_x[1:]:
        if boxes[i][0] - reach >= min_gutter:
            out.append((reach, boxes[i][0]))
        reach = max(reach, boxes[i][2])
    return out


def _validate(
    boxes: Sequence[Box],
    body: list[int],
    valley: tuple[float, float],
    valleys: list[tuple[float, float]],
    text_width: float,
    numeric: Sequence[bool] | None,
) -> tuple[list[int], list[int], list[int]] | None:
    """The sides and the crossing boxes if ``valley`` is a gutter, else ``None``. ``valleys`` are
    all the valleys of the projection: a table with a narrow column of cells between two wide
    ones (a cover page: name, ticker, exchange) is caught by that column, whatever valley is
    tried."""
    lo, hi = valley
    left = [i for i in body if boxes[i][2] <= lo]
    right = [i for i in body if boxes[i][0] >= hi]
    crossing = [i for i in body if boxes[i][2] > lo and boxes[i][0] < hi]
    if len(crossing) > _CROSSING:
        return None
    wide = _COLUMN_WIDTH * text_width
    for side in (left, right):
        if len(side) < _COLUMN_MIN_BOXES:
            return None
        if sum(boxes[i][2] - boxes[i][0] >= wide for i in side) < _COLUMN_MIN_BOXES:
            return None
    l_top, l_bottom = min(boxes[i][1] for i in left), max(boxes[i][3] for i in left)
    r_top, r_bottom = min(boxes[i][1] for i in right), max(boxes[i][3] for i in right)
    shared_top, shared_bottom = max(l_top, r_top), min(l_bottom, r_bottom)
    shorter = min(l_bottom - l_top, r_bottom - r_top)
    if shared_bottom <= shared_top or shared_bottom - shared_top < _GUTTER_OVERLAP * shorter:
        return None
    for i in crossing:  # a title or heading, not a line between the columns' lines
        centre = (boxes[i][1] + boxes[i][3]) / 2
        if shared_top < centre < shared_bottom:
            return None
    if _table_rows(boxes, left, right, text_width, numeric):
        return None
    if len(valleys) >= _TABLE_COLUMNS - 1:
        centres = [(a + b) / 2 for a, b in valleys]
        bands: list[list[int]] = [[] for _ in range(len(valleys) + 1)]
        for i in (*left, *right):
            bands[bisect.bisect_left(centres, (boxes[i][0] + boxes[i][2]) / 2)].append(i)
        everyone = {*left, *right}
        for band in bands:
            members = set(band)
            others = [i for i in everyone if i not in members]
            if len(band) >= _COLUMN_MIN_BOXES and _table_rows(
                boxes, band, others, text_width, numeric
            ):
                return None
    return left, right, crossing


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _table_rows(
    boxes: Sequence[Box],
    left: Sequence[int],
    right: Sequence[int],
    text_width: float,
    numeric: Sequence[bool] | None = None,
) -> bool:
    """Whether a gutter between ``left`` and ``right`` runs through an unruled table: the sides
    are row-aligned (at least :data:`_ROW_ALIGNED_SHARE` of the smaller side's boxes have a box
    on the other side whose vertical centre is within :data:`_ROW_ALIGN` x the median line
    height) and one side is made of cells: its median box is under 10% of the text width, or
    (``numeric`` flags the boxes that hold figures) at least :data:`_NUMERIC_SHARE` of its boxes
    are figures. A row of figures can be as wide as a line of text, so width alone misses the
    tables of a financial statement once they are cut out of the page."""
    cell = _CELL_WIDTH * text_width
    cells = (
        _median([boxes[i][2] - boxes[i][0] for i in left]) < cell
        or _median([boxes[i][2] - boxes[i][0] for i in right]) < cell
    )
    if numeric is not None and not cells:
        cells = any(
            sum(numeric[i] for i in side) >= _NUMERIC_SHARE * len(side) for side in (left, right)
        )
    if not cells:
        return False
    small, big = (left, right) if len(left) <= len(right) else (right, left)
    line_height = _median([boxes[i][3] - boxes[i][1] for i in (*left, *right)])
    tolerance = _ROW_ALIGN * line_height
    centres = sorted((boxes[i][1] + boxes[i][3]) / 2 for i in big)
    aligned = 0
    for i in small:
        c = (boxes[i][1] + boxes[i][3]) / 2
        j = bisect.bisect_left(centres, c - tolerance)
        aligned += j < len(centres) and centres[j] <= c + tolerance
    return aligned >= _ROW_ALIGNED_SHARE * len(small)


def _break_under_columns(
    boxes: Sequence[Box],
    group: list[int],
    structural: float,
    text_width: float,
    min_gutter: float,
    numeric: Sequence[bool] | None = None,
) -> list[list[int]] | None:
    """The region cut in two at its topmost section break with columns above it and a box
    across their gutter below it, or ``None``. A section break is a gap of at least
    ``structural`` that no box crosses."""
    members = sorted(group, key=lambda i: (boxes[i][1], boxes[i][3], i))
    core = members[_CROSSING : len(members) - _CROSSING]
    if not (_valleys(boxes, members, min_gutter) or _valleys(boxes, core, min_gutter)):
        return None  # nothing like columns: skip the scan
    reach = boxes[members[0]][3]
    tried = 0
    for k in range(1, len(members)):
        i = members[k]
        if boxes[i][1] - reach >= structural > 0 and tried < _MAX_BREAKS:
            tried += 1
            head, tail = sorted(members[:k]), sorted(members[k:])
            found = _find_gutter(boxes, head, text_width, min_gutter, numeric)
            if found is not None:
                lo = max(boxes[i][2] for i in found[0])
                hi = min(boxes[i][0] for i in found[1])
                if any(boxes[i][2] > lo and boxes[i][0] < hi for i in tail):
                    return [head, tail]  # what follows is not confined to the columns
        reach = max(reach, boxes[i][3])
    return None


def _row_cut(boxes: Sequence[Box], group: list[int], min_gap: float) -> list[list[int]] | None:
    """Cut ``group`` in two at its largest horizontal gap of at least ``min_gap`` (and above
    zero), or ``None`` if there is none. The topmost of equal gaps wins."""
    by_y = sorted(group, key=lambda i: (boxes[i][1], boxes[i][3], i))
    reach = boxes[by_y[0]][3]
    best_gap, best_at = 0.0, 0
    for k in range(1, len(by_y)):
        i = by_y[k]
        gap = boxes[i][1] - reach
        if gap > best_gap and gap >= min_gap:
            best_gap, best_at = gap, k
        reach = max(reach, boxes[i][3])
    if best_at == 0:
        return None
    return [sorted(by_y[:best_at]), sorted(by_y[best_at:])]


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
    """The page's lines in reading order: :func:`xy_cut_order` over units, each layout
    region's lines (``groups``; placed where its first line is, kept in their order) or a
    single line. Without groups every line is its own unit."""
    if len(indexes) < 2:
        return indexes
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
    gaps = _line_gaps(page)
    local = xy_cut_order(
        boxes,
        page_width=max(widths, default=0.0),
        median_char_width=statistics.median(char_widths) if char_widths else 0.0,
        median_line_gap=statistics.median(gaps) if gaps else 0.0,
        numeric=[all(_is_figures(lines[i].text) for i in unit) for unit in members],
    )
    return [i for k in local for i in members[k]]


def _is_figures(text: str) -> bool:
    """Whether a line holds figures, not words: it has a digit and at most one letter in five
    of its letters and digits (``$ (13,870)``, ``41% 31% 13%``, ``2024``)."""
    digits = sum(ch.isdigit() for ch in text)
    letters = sum(ch.isalpha() for ch in text)
    return digits > 0 and letters <= 0.2 * (digits + letters)


def _line_gaps(page: Sequence[Line]) -> list[float]:
    """The gaps between consecutive lines of one extractor block, zeros included.

    Lines of a block are in one column, so these are the line-to-line gaps of running text. At
    tight leading the line boxes touch or overlap and the gap is 0: leaving the zeros out made
    the median a structural gap (title to heading, heading to text), above every gap that
    separates a heading from its columns. A gap across blocks jumps between columns and is left
    out. A page without two lines in one block falls back to its positive gaps.
    """
    inside = [
        ln.gap_above
        for prev, ln in zip(page, page[1:], strict=False)
        if ln.block == prev.block and ln.gap_above < float("inf")
    ]
    return inside or [ln.gap_above for ln in page if 0 < ln.gap_above < float("inf")]
