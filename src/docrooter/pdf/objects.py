"""Engine-neutral page objects: the text lines, raster images, vector-graphics regions, ruled
tables and character statistics of every page, from ``pymupdf`` or ``pdfplumber``.

All boxes are in points in the *displayed* page's coordinates (origin top left, ``/Rotate``
applied) and are clipped to the page, so downstream code never has to know the engine or the
page rotation. On an unrotated page ``lines`` equal :func:`docrooter.pdf.lines.extract_lines`
for the same engine, in content and order; on a rotated page they are mapped into display
space as well (``extract_lines`` leaves pymupdf lines in unrotated page space, and its output
is frozen). No text-layer line is lost to the rotation: text that runs vertically on screen
because the page is rotated (horizontal on the unrotated page, which is what ``extract_lines``
keeps) is kept as a tall display-space line, and the page gets the flag ``"vertical_text"``.

Vector-heavy pages (maps, plots with tens of thousands of paths) are bounded: above
``MAX_DRAWING_PATHS`` paths the drawing area becomes a single region and no table search runs,
and the page gets the flag ``"drawings_capped"``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from docrooter.elements.models import Box
from docrooter.errors import DocrooterError
from docrooter.pdf.lines import (
    PROGRESS_EVERY,
    Engine,
    Line,
    Progress,
    _finalise,
    _fragment_row,
    _import_pdfplumber,
    _import_pymupdf,
    _is_horizontal,
    _join_row_fragments,
    _line_fragment,
    _pdfplumber_page_rows,
    _pymupdf_flags,
    _pymupdf_page_rows,
)

__all__ = ["MAX_DRAWING_PATHS", "PageObjects", "extract_page_objects"]

MAX_DRAWING_PATHS = 5000  # per page; above it the whole drawing area becomes one cluster
MAX_DRAWING_CLUSTERS = 200  # per page; the largest are kept
MAX_TABLE_PATHS = 1500  # per page; the table search is skipped on pages with more paths
# The ruled-table pre-check (``_may_hold_ruled_table``) uses the table finders' own settings
# (pymupdf and pdfplumber share them): edges shorter than 1 pt are dropped first, positions
# within 3 pt snap together, collinear segments within 3 pt join, and edges within 3 pt
# intersect.
_RULE_MIN_LENGTH = 1.0
_RULE_SNAP = 3.0
_RULE_JOIN = 3.0
_RULE_INTERSECT = 3.0
_MAX_RULE_VERTICES = 3000  # more edge crossings than this: let the finder decide
_PYMUPDF_GAP = 3.0  # pt; drawing items this close join one cluster (``cluster_drawings`` default)
_PDFPLUMBER_GAP = 6.0  # pt; the same for pdfplumber rects, lines and curves
_MIN_SIDE = 0.5  # pt; a box thinner than this (after clipping) is degenerate and dropped
_MIN_TABLE_ROWS = 2
_MIN_TABLE_COLS = 2

_log = logging.getLogger(__name__)

_CID = re.compile(r"\(cid:\d+\)")  # pdfminer's rendering of a glyph without a Unicode mapping

_RawBox = tuple[float, float, float, float]


class PageObjects(BaseModel):
    """What one page contains, independent of the engine that found it."""

    page: int  # 1-based
    width: float  # displayed page size, points
    height: float
    rotation: int  # the page's /Rotate (0, 90, 180, 270); boxes are already rotated
    lines: list[Line]
    images: list[Box]  # raster image placements
    drawings: list[Box]  # clustered vector-graphics regions
    tables: list[tuple[Box, list[list[str]]]]  # ruled tables: box and row-major cell text
    char_count: int  # characters of the text layer, whitespace excluded
    invisible_chars: int  # render mode 3 (pymupdf only; 0 with pdfplumber)
    garbled_chars: int  # U+FFFD, ``(cid:N)``, Private Use Area characters, C0/C1 controls
    n_drawing_paths: int  # vector paths on the page, before clustering
    flags: list[str] = Field(default_factory=list)  # "drawings_capped", "tables_skipped", …


def extract_page_objects(
    path: str | Path,
    *,
    engine: Engine,
    max_pages: int | None = None,
    progress: Progress | None = None,
    tables: Literal["lines", "none"] = "lines",
) -> list[PageObjects]:
    """Extract the :class:`PageObjects` of every page (of the first ``max_pages``) of ``path``.

    ``progress(done, total)`` is called after every 10th page, as in ``extract_lines``.
    ``tables="none"`` skips the ruled-table search (the slowest part of the pass): every page
    then has no ``tables``.

    Raises ``DocrooterError`` for encrypted, corrupt or non-PDF files and ``MissingExtraError``
    when the engine's package is not installed.
    """
    if tables not in ("lines", "none"):
        raise ValueError(f"tables must be 'lines' or 'none', got {tables!r}")
    find = tables == "lines"
    if engine == "pymupdf":
        return _extract_pymupdf(path, max_pages, progress, find)
    if engine == "pdfplumber":
        return _extract_pdfplumber(path, max_pages, progress, find)
    raise DocrooterError(f"unknown PDF engine {engine!r}; use 'pymupdf' or 'pdfplumber'")


# --- shared helpers ------------------------------------------------------------------------


def _count_chars(text: str) -> tuple[int, int]:
    """``(characters, garbled characters)`` of ``text``, whitespace not counted. A pdfminer
    ``(cid:N)`` token counts as one garbled character."""
    total = garbled = 0
    pos = 0
    for match in _CID.finditer(text):
        t, g = _count_plain(text[pos : match.start()])
        total, garbled = total + t + 1, garbled + g + 1
        pos = match.end()
    t, g = _count_plain(text[pos:])
    return total + t, garbled + g


# Garbled characters: U+FFFD, the Private Use Area and private use planes 15 and 16, C0
# controls other than tab, newline, vertical tab, form feed and carriage return, DEL and the
# C1 controls. ``_GARBLED_SPACE`` are those of them that ``str.isspace`` also accepts (the
# file separators and NEL); they still count as characters.
_GARBLED = re.compile(
    "[\ufffd\ue000-\uf8ff\U000f0000-\U000ffffd\U00100000-\U0010fffd\x00-\x08\x0e-\x1f\x7f-\x9f]"
)
_GARBLED_SPACE = re.compile("[\x1c-\x1f\x85]")


def _count_plain(text: str) -> tuple[int, int]:
    """``_count_chars`` without ``(cid:N)`` tokens: garbled characters count even when
    ``str.isspace`` accepts them, other whitespace does not."""
    garbled = len(_GARBLED.findall(text))
    visible = len("".join(text.split()))
    return visible + len(_GARBLED_SPACE.findall(text)), garbled


def _to_box(page: int, raw: Iterable[float], width: float, height: float) -> Box | None:
    """``raw`` (x0, y0, x1, y1) normalised and clipped to the page, or ``None`` when nothing
    of it is left (fully off the page, zero area or thinner than ``_MIN_SIDE``)."""
    x0, y0, x1, y1 = (float(v) for v in raw)
    x0, x1 = max(0.0, min(x0, x1)), min(width, max(x0, x1))
    y0, y1 = max(0.0, min(y0, y1)), min(height, max(y0, y1))
    if x1 - x0 < _MIN_SIDE or y1 - y0 < _MIN_SIDE:
        return None
    return Box(page=page, x0=round(x0, 2), y0=round(y0, 2), x1=round(x1, 2), y1=round(y1, 2))


def _boxes(page: int, raws: Iterable[Iterable[float]], width: float, height: float) -> list[Box]:
    """Clipped boxes of ``raws``, degenerate ones dropped, de-duplicated, top to bottom."""
    out: dict[tuple[float, float, float, float], Box] = {}
    for raw in raws:
        box = _to_box(page, raw, width, height)
        if box is not None:
            out.setdefault((round(box.x0), round(box.y0), round(box.x1), round(box.y1)), box)
    return sorted(out.values(), key=lambda b: (b.y0, b.x0))


def _near(a: list[float], b: list[float], gap: float) -> bool:
    return a[0] <= b[2] + gap and b[0] <= a[2] + gap and a[1] <= b[3] + gap and b[1] <= a[3] + gap


def _cluster(boxes: Iterable[_RawBox], gap: float) -> list[_RawBox]:
    """Merge boxes closer than ``gap`` into their bounding boxes, repeating until no two
    clusters are close (``pymupdf.Page.cluster_drawings`` semantics, as a sweep over x so that
    a few thousand items cost milliseconds instead of seconds)."""
    clusters = [[min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])] for b in boxes]
    while True:
        clusters.sort(key=lambda c: c[0])
        done: list[list[float]] = []
        active: list[list[float]] = []
        merged = False
        for cur in clusters:
            keep: list[list[float]] = []
            for other in active:
                if other[2] + gap < cur[0]:
                    done.append(other)  # nothing later in the sweep can reach it
                elif _near(other, cur, gap):
                    cur[0], cur[1] = min(cur[0], other[0]), min(cur[1], other[1])
                    cur[2], cur[3] = max(cur[2], other[2]), max(cur[3], other[3])
                    merged = True
                else:
                    keep.append(other)
            keep.append(cur)
            active = keep
        clusters = done + active
        if not merged:
            return [(c[0], c[1], c[2], c[3]) for c in clusters]


def _drawing_clusters(
    page: int, raws: list[_RawBox], gap: float, width: float, height: float
) -> tuple[list[Box], list[str]]:
    """The vector-graphics regions of a page from the boxes of its paths, and flags.

    Paths are clipped to the page, merged by proximity, and regions not larger than the
    merge distance in both directions (a rule, a bullet) are dropped, as
    ``cluster_drawings`` does. Above ``MAX_DRAWING_PATHS`` paths the union of all of them is
    the one region."""
    flags: list[str] = []
    if len(raws) > MAX_DRAWING_PATHS:
        flags.append("drawings_capped")
        x0, y0 = min(r[0] for r in raws), min(r[1] for r in raws)
        x1, y1 = max(r[2] for r in raws), max(r[3] for r in raws)
        box = _to_box(page, (x0, y0, x1, y1), width, height)
        return ([box] if box is not None else []), flags
    inside = [
        r
        for r in raws
        if r[2] >= 0 and r[0] <= width and r[3] >= 0 and r[1] <= height  # not fully off page
    ]
    regions = [c for c in _cluster(inside, gap) if (c[2] - c[0]) > gap and (c[3] - c[1]) > gap]
    boxes = [b for c in regions if (b := _to_box(page, c, width, height)) is not None]
    if len(boxes) > MAX_DRAWING_CLUSTERS:
        boxes.sort(key=lambda b: (b.x1 - b.x0) * (b.y1 - b.y0), reverse=True)
        boxes = boxes[:MAX_DRAWING_CLUSTERS]
        flags.append("drawings_capped")
    boxes.sort(key=lambda b: (b.y0, b.x0))
    return boxes, flags


def _table_rows(raw: Iterable[Iterable[Any]]) -> list[list[str]]:
    return [[(cell or "").strip() for cell in row] for row in raw]


def _is_table(rows: list[list[str]]) -> bool:
    return (
        len(rows) >= _MIN_TABLE_ROWS
        and max((len(r) for r in rows), default=0) >= _MIN_TABLE_COLS
        and any(cell for row in rows for cell in row)
    )


# --- ruled-table pre-check -----------------------------------------------------------------
#
# The table finders (``find_tables`` with the "lines" strategy, in pymupdf and pdfplumber alike)
# cost far more than everything else on a page: they rebuild the page's characters in Python
# before looking at a single line. A ruled table needs cells, and cells need crossing rules, so
# a page whose rules cannot form one table of at least two rows and two columns never reaches
# the finder. The check below runs the finder's own grid search (snap, join, intersect, cells,
# tables of cells sharing a corner) on the rules alone, without the text.

_Rule = tuple[float, float, float]  # (position, start, end): y, x0, x1 or x, y0, y1


def _snap_rules(rules: list[_Rule]) -> list[list[float]]:
    """Positions within ``_RULE_SNAP`` of their neighbour (chained) moved to their mean."""
    ordered = sorted(rules)
    out: list[list[float]] = []
    group: list[_Rule] = []
    for rule in ordered:
        if group and rule[0] > group[-1][0] + _RULE_SNAP:
            mean = sum(r[0] for r in group) / len(group)
            out += [[mean, r[1], r[2]] for r in group]
            group = []
        group.append(rule)
    if group:
        mean = sum(r[0] for r in group) / len(group)
        out += [[mean, r[1], r[2]] for r in group]
    return out


def _join_rules(rules: list[list[float]]) -> list[list[float]]:
    """Collinear rules that touch or nearly touch (``_RULE_JOIN``) merged into one."""
    out: list[list[float]] = []
    for rule in sorted(rules):
        last = out[-1] if out else None
        if last is not None and last[0] == rule[0] and rule[1] <= last[2] + _RULE_JOIN:
            last[2] = max(last[2], rule[2])
        else:
            out.append(list(rule))
    return out


def _clip_rules(rules: list[_Rule], along: float, across: float) -> list[_Rule]:
    """Rules clipped to the page, as the pymupdf finder clips its lines; ``along`` is the page
    extent in the rules' direction, ``across`` the other one."""
    out = []
    for pos, start, end in rules:
        if start > along or end < 0 or pos > across or pos < 0:
            continue
        out.append((pos, max(start, 0.0), min(end, along)))
    return out


def _may_hold_ruled_table(
    horizontal: list[_Rule], vertical: list[_Rule], bounds: tuple[float, float] | None = None
) -> bool:
    """Whether the rules of a page (``horizontal`` as (y, x0, x1), ``vertical`` as
    (x, y0, y1)) can form a ruled table of at least two rows and two columns; ``bounds`` is
    the (width, height) the finder clips to, if it clips.

    It errs towards ``True``: after merging it keeps the edges the finder drops as shorter
    than 3 pt, and above ``_MAX_RULE_VERTICES`` crossings it does not look further."""
    if bounds is not None:
        horizontal = _clip_rules(horizontal, bounds[0], bounds[1])
        vertical = _clip_rules(vertical, bounds[1], bounds[0])
    # the finder drops edges under 1 pt before snapping; they would drag the snap positions
    horizontal = [r for r in horizontal if r[2] - r[1] >= _RULE_MIN_LENGTH]
    vertical = [r for r in vertical if r[2] - r[1] >= _RULE_MIN_LENGTH]
    if len(horizontal) < 2 or len(vertical) < 2:
        return False
    hs = _join_rules(_snap_rules(horizontal))
    vs = _join_rules(_snap_rules(vertical))
    h = np.array(hs)
    v = np.array(vs)
    tol = _RULE_INTERSECT
    crossing = (
        (v[None, :, 1] <= h[:, None, 0] + tol)
        & (v[None, :, 2] >= h[:, None, 0] - tol)
        & (v[None, :, 0] >= h[:, None, 1] - tol)
        & (v[None, :, 0] <= h[:, None, 2] + tol)
    )
    rows, cols = np.nonzero(crossing)
    if len(rows) < 4:  # one cell needs four corners
        return False
    if len(rows) > _MAX_RULE_VERTICES:
        return True
    # vertex (x, y) -> the vertical and the horizontal rules through it
    vertices: dict[tuple[float, float], tuple[set[int], set[int]]] = {}
    for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
        through = vertices.setdefault((vs[j][0], hs[i][0]), (set(), set()))
        through[0].add(j)
        through[1].add(i)
    return _has_grid(vertices)


def _has_grid(vertices: dict[tuple[float, float], tuple[set[int], set[int]]]) -> bool:
    """The finder's cell search on the rule crossings: each vertex opens the smallest cell
    whose four corners are joined by rules; cells sharing a corner form a table. True when a
    table has two distinct cell tops and two distinct cell lefts."""

    def joined(a: tuple[float, float], b: tuple[float, float]) -> bool:
        if a[0] == b[0] and vertices[a][0] & vertices[b][0]:
            return True
        return a[1] == b[1] and bool(vertices[a][1] & vertices[b][1])

    points = sorted(vertices)
    by_x: dict[float, list[tuple[float, float]]] = {}
    by_y: dict[float, list[tuple[float, float]]] = {}
    for point in points:
        by_x.setdefault(point[0], []).append(point)
        by_y.setdefault(point[1], []).append(point)
    cells: list[tuple[float, float, float, float]] = []
    for point in points:
        cell = None
        below = [q for q in by_x[point[0]] if q > point]
        right = [q for q in by_y[point[1]] if q > point]
        for down in below:
            if not joined(point, down):
                continue
            for across in right:
                corner = (across[0], down[1])
                if (
                    joined(point, across)
                    and corner in vertices
                    and joined(corner, across)
                    and joined(corner, down)
                ):
                    cell = (point[0], point[1], corner[0], corner[1])
                    break
            if cell is not None:
                break
        if cell is not None:
            cells.append(cell)
    if len(cells) < 2:
        return False
    parent = list(range(len(cells)))

    def root(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    owner: dict[tuple[float, float], int] = {}
    for k, (x0, y0, x1, y1) in enumerate(cells):
        for corner in ((x0, y0), (x0, y1), (x1, y0), (x1, y1)):
            if corner in owner:
                parent[root(k)] = root(owner[corner])
            else:
                owner[corner] = k
    tables: dict[int, list[tuple[float, float, float, float]]] = {}
    for k, cell in enumerate(cells):
        tables.setdefault(root(k), []).append(cell)
    return any(
        len({c[1] for c in group}) >= _MIN_TABLE_ROWS
        and len({c[0] for c in group}) >= _MIN_TABLE_COLS
        for group in tables.values()
    )


def _pymupdf_rulings(paths: list[dict[str, Any]]) -> tuple[list[_Rule], list[_Rule]]:
    """The rules pymupdf's table finder builds from a page's vector paths, in unrotated page
    space: axis-parallel lines (within the snap tolerance), the sides of rectangles and quads
    (a rectangle thinner than 3 pt is one line), and the box of every cluster of touching
    paths, which the finder adds when the cluster holds text (here always, so nothing is
    missed). A line counts as horizontal only when exactly level, as in the finder."""
    horizontal: list[_Rule] = []
    vertical: list[_Rule] = []

    def line(x0: float, y0: float, x1: float, y1: float) -> None:
        if abs(x0 - x1) > _RULE_SNAP and abs(y0 - y1) > _RULE_SNAP:
            return  # not axis-parallel
        if y0 == y1:
            horizontal.append((y0, min(x0, x1), max(x0, x1)))
        else:
            vertical.append((min(x0, x1), min(y0, y1), max(y0, y1)))

    def rect(x0: float, y0: float, x1: float, y1: float) -> None:
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        if x1 - x0 <= _RULE_SNAP and x1 - x0 < y1 - y0:  # a vertical line drawn as a box
            line((x0 + x1) / 2, y0, (x0 + x1) / 2, y1)
        elif y1 - y0 <= _RULE_SNAP and y1 - y0 < x1 - x0:  # a horizontal line
            line(x0, (y0 + y1) / 2, x1, (y0 + y1) / 2)
        else:
            line(x0, y0, x0, y1)
            line(x0, y1, x1, y1)
            line(x1, y1, x1, y0)
            line(x1, y0, x0, y0)

    for path in paths:
        items = path["items"]
        if path.get("closePath") and items and items[0][0] == "l" and items[-1][0] == "l":
            items = [*items, ("l", items[-1][2], items[0][1])]
        for item in items:
            kind = item[0]
            if kind == "l":
                (x0, y0), (x1, y1) = item[1], item[2]
                line(x0, y0, x1, y1)
            elif kind == "re":
                rect(*item[1][:4])
            elif kind == "qu":
                ul, ur, ll, lr = item[1]
                for (x0, y0), (x1, y1) in ((ul, ll), (ll, lr), (lr, ur), (ur, ul)):
                    line(x0, y0, x1, y1)
    for x0, y0, x1, y1 in _cluster((tuple(p["rect"]) for p in paths), _PYMUPDF_GAP):
        rect(x0, y0, x1, y1)
    return horizontal, vertical


def _unrotated_size(page: Any) -> tuple[float, float]:
    """The page size in its unrotated space, where pymupdf's finder works."""
    width, height = float(page.rect.width), float(page.rect.height)
    return (height, width) if int(page.rotation) in (90, 270) else (width, height)


def _pdfplumber_rulings(page: Any) -> tuple[list[_Rule], list[_Rule]]:
    """The horizontal and vertical edges pdfplumber's table finder starts from."""
    horizontal: list[_Rule] = []
    vertical: list[_Rule] = []
    for edge in page.edges:
        if edge.get("orientation") == "h":
            horizontal.append((float(edge["top"]), float(edge["x0"]), float(edge["x1"])))
        elif edge.get("orientation") == "v":
            vertical.append((float(edge["x0"]), float(edge["top"]), float(edge["bottom"])))
    return horizontal, vertical


# --- pymupdf -------------------------------------------------------------------------------


def _extract_pymupdf(
    path: str | Path, max_pages: int | None, progress: Progress | None, find_tables: bool = True
) -> list[PageObjects]:
    pymupdf = _import_pymupdf()
    try:
        doc = pymupdf.open(str(path), filetype="pdf")
    except Exception as exc:
        raise DocrooterError(f"cannot open PDF {path}: {exc}") from exc
    try:
        if doc.needs_pass:
            raise DocrooterError(f"cannot open PDF {path}: file is encrypted")
        total = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
        flags = _pymupdf_flags(pymupdf)
        table_kwargs = _table_kwargs(pymupdf)
        out: list[PageObjects] = []
        try:
            for page_index, page in enumerate(doc):
                if max_pages is not None and page_index >= max_pages:
                    break
                out.append(
                    _pymupdf_page(page, page_index, pymupdf, flags, table_kwargs, find_tables)
                )
                if progress is not None and (page_index + 1) % PROGRESS_EVERY == 0:
                    progress(page_index + 1, total)
        except Exception as exc:
            raise DocrooterError(f"cannot open PDF {path}: {exc}") from exc
    finally:
        doc.close()
    return out


def _table_kwargs(pymupdf: Any) -> dict[str, Any]:
    """Extra ``find_tables`` arguments. Newer pymupdf gates the table search by a layout model
    once ``pymupdf4llm`` or ``pymupdf.layout`` has been imported anywhere in the process; ruled
    tables here must not depend on that and must never run a model."""
    import inspect

    try:
        params = inspect.signature(pymupdf.table.find_tables).parameters
    except (AttributeError, TypeError, ValueError):
        return {}
    return {"use_layout": False} if "use_layout" in params else {}


def _pymupdf_page(
    page: Any,
    page_index: int,
    pymupdf: Any,
    text_flags: int,
    table_kwargs: dict[str, Any],
    find_tables: bool = True,
) -> PageObjects:
    number = page_index + 1
    width, height = float(page.rect.width), float(page.rect.height)
    rotation = int(page.rotation)
    matrix = page.rotation_matrix if rotation else None  # unrotated space -> display space

    def display(raw: Iterable[float]) -> _RawBox:
        if matrix is None:
            x0, y0, x1, y1 = (float(v) for v in raw)
            return x0, y0, x1, y1
        return _transform(raw, matrix)

    # One text extraction serves the image blocks, the lines and the character statistics.
    # It keeps images too; an image block splits the text blocks around it, so a page that has
    # one is read again without images for its lines (``extract_lines`` reads them that way).
    textpage = page.get_textpage(flags=text_flags | pymupdf.TEXT_PRESERVE_IMAGES)
    image_blocks = [b[:4] for b in textpage.extractBLOCKS() if b[6] == 1]
    if image_blocks:
        textpage = page.get_textpage(flags=text_flags)
    # ``extract_lines`` reads the blocks sorted; the rotated path sorts them by their displayed
    # boxes itself.
    blocks = page.get_text("dict", textpage=textpage, sort=matrix is None)["blocks"]
    if matrix is None:
        text_rows = list(_pymupdf_page_rows(page, page_index, text_flags, blocks))
        vertical = False
    else:
        text_rows, vertical = _pymupdf_display_rows(page, page_index, matrix, blocks)
    lines = _finalise(text_rows)
    char_count, garbled, invisible = _pymupdf_char_stats(page, blocks)

    image_raws = [display(b) for b in image_blocks]
    image_raws += [display(info["bbox"]) for info in page.get_image_info()]
    images = _boxes(number, image_raws, width, height)

    paths = page.get_cdrawings()
    drawings, flags = _drawing_clusters(
        number, [display(p["rect"]) for p in paths], _PYMUPDF_GAP, width, height
    )
    tables: list[tuple[Box, list[list[str]]]] = []
    if not find_tables:
        pass
    elif len(paths) > MAX_TABLE_PATHS:
        flags.append("tables_skipped")
    elif _may_hold_ruled_table(*_pymupdf_rulings(paths), bounds=_unrotated_size(page)):
        try:
            found = page.find_tables(strategy="lines", **table_kwargs).tables
            for table in found:
                rows = _table_rows(table.extract())
                # pymupdf already reports table boxes in display space (unlike text and paths)
                box = _to_box(number, table.bbox, width, height)
                if box is not None and _is_table(rows):
                    tables.append((box, rows))
        except Exception:  # one odd page must not fail the document
            _log.debug("table search failed on page %d", number, exc_info=True)
            flags.append("tables_failed")
    tables.sort(key=lambda t: (t[0].y0, t[0].x0))

    if vertical:
        flags.append("vertical_text")
    return PageObjects(
        page=number, width=width, height=height, rotation=rotation, lines=lines, images=images,
        drawings=drawings, tables=tables, char_count=char_count, invisible_chars=invisible,
        garbled_chars=garbled, n_drawing_paths=len(paths), flags=flags,
    )  # fmt: skip


_FILLED_OR_STROKED = 16 | 32  # span ``char_flags``: FZ_STEXT_FILLED, FZ_STEXT_STROKED


def _pymupdf_char_stats(page: Any, blocks: list[dict[str, Any]]) -> tuple[int, int, int]:
    """``(characters, garbled, invisible)`` of a page from its ``dict`` text blocks, whitespace
    not counted: the same characters as ``get_text("text")`` with the same flags. A span is
    invisible (text render mode 3, an OCR layer) when its ``char_flags`` say it is neither
    filled nor stroked; spans split where those flags change. Older pymupdf without
    ``char_flags`` falls back to the page's text trace."""
    total = garbled = invisible = 0
    traced = False
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            t, g = _count_chars("".join(span["text"] for span in line["spans"]))
            total, garbled = total + t, garbled + g
            for span in line["spans"]:
                flags = span.get("char_flags")
                if flags is None:
                    traced = True
                elif not flags & _FILLED_OR_STROKED:
                    invisible += sum(1 for ch in span["text"] if not ch.isspace())
    if traced:
        invisible = sum(
            1
            for span in page.get_texttrace()
            if span.get("type") == 3  # render mode 3: neither filled nor stroked
            for ch in span["chars"]
            if not chr(ch[0]).isspace()
        )
    return total, garbled, invisible


def _pymupdf_display_rows(
    page: Any, page_index: int, matrix: Any, blocks: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], bool]:
    """The rows of a rotated pymupdf page in display space, and whether any of them runs
    vertically on screen.

    Blocks are ordered by their displayed boxes (pymupdf's own sort key, bottom then left).
    In each block, every line that is horizontal on the unrotated page is joined into rows in
    unrotated space, exactly as ``extract_lines`` joins it, and the row's box is then mapped to
    display space; so every ``extract_lines`` line is kept with the same text, whether it reads
    across the screen (0 and 180 degrees) or runs down it (90 and 270, the text of a portrait
    page turned by ``/Rotate``). Lines vertical on the unrotated page that read across the
    screen (upright text authored for the rotation, which ``extract_lines`` skips) are joined
    in display space. Lines vertical in both spaces are skipped.
    """
    width, height = float(page.rect.width), float(page.rect.height)
    keyed = sorted(
        ((_transform(b["bbox"], matrix), b) for b in blocks if b.get("type") == 0),
        key=lambda kb: (kb[0][3], kb[0][0]),
    )
    rows: list[dict[str, Any]] = []
    any_vertical = False
    for block_no, (_, block) in enumerate(keyed):
        unrotated: list[dict[str, Any]] = []  # horizontal on the unrotated page
        upright: list[dict[str, Any]] = []  # vertical on the page, across the screen
        for raw in block["lines"]:
            frag = _line_fragment(raw)
            if frag is None:
                continue
            dx, dy = raw.get("dir", (1.0, 0.0))
            on_screen = _is_horizontal(
                (matrix.a * dx + matrix.c * dy, matrix.b * dx + matrix.d * dy)
            )
            if _is_horizontal((dx, dy)):
                unrotated.append(frag)
                any_vertical = any_vertical or not on_screen
            elif on_screen:
                upright.append(_moved(frag, matrix))
        joined = [_moved(frag, matrix) for frag in _join_row_fragments(unrotated)]
        joined += _join_row_fragments(upright)
        for frag in joined:
            row = _fragment_row(frag, page_index, width, height, block_no)
            if row is not None:
                rows.append(row)
    return rows, any_vertical


def _transform(raw: Iterable[float], matrix: Any) -> _RawBox:
    """``raw`` (x0, y0, x1, y1) mapped through ``matrix`` (a rotation by a multiple of 90
    degrees), normalised so that x0 <= x1 and y0 <= y1."""
    x0, y0, x1, y1 = (float(v) for v in raw)
    pts = [
        (matrix.a * x + matrix.c * y + matrix.e, matrix.b * x + matrix.d * y + matrix.f)
        for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))
    ]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _moved(frag: dict[str, Any], matrix: Any) -> dict[str, Any]:
    """A copy of a line fragment with its box in display space."""
    x0, y0, x1, y1 = _transform((frag["x0"], frag["y0"], frag["x1"], frag["y1"]), matrix)
    return {**frag, "x0": x0, "y0": y0, "x1": x1, "y1": y1}


# --- pdfplumber ----------------------------------------------------------------------------


def _extract_pdfplumber(
    path: str | Path, max_pages: int | None, progress: Progress | None, find_tables: bool = True
) -> list[PageObjects]:
    pdfplumber = _import_pdfplumber()
    try:
        pdf = pdfplumber.open(str(path))
    except Exception as exc:
        raise DocrooterError(f"cannot open PDF {path}: {exc}") from exc
    out: list[PageObjects] = []
    try:
        pages = pdf.pages[:max_pages]
        for page_index, page in enumerate(pages):
            try:
                out.append(_pdfplumber_page(page, page_index, find_tables))
            finally:
                # Free the page's parsed objects and layout caches (see ``extract_lines``).
                close = getattr(page, "close", None) or getattr(page, "flush_cache", None)
                if close is not None:
                    close()
            if progress is not None and (page_index + 1) % PROGRESS_EVERY == 0:
                progress(page_index + 1, len(pages))
    except DocrooterError:
        raise
    except Exception as exc:
        raise DocrooterError(f"cannot open PDF {path}: {exc}") from exc
    finally:
        pdf.close()
    return out


def _visible_frame(page: Any) -> tuple[float, float, float, float]:
    """The visible page (CropBox ∩ MediaBox, as viewers and pymupdf show it) in pdfplumber's
    object coordinates, as ``(dx, dy, width, height)``: subtracting ``(dx, dy)`` from a
    pdfplumber box gives display coordinates (top-left origin, /Rotate applied), the space of
    pymupdf's boxes. Derived from pdfminer's page matrix (its ``process_page``) and
    pdfplumber's flip, for each rotation; pdfplumber's own ``cropbox`` is not used because it
    ignores a MediaBox origin under rotation."""
    obj = page.page_obj
    mx0, mx1 = sorted(float(v) for v in (obj.mediabox[0], obj.mediabox[2]))
    my0, my1 = sorted(float(v) for v in (obj.mediabox[1], obj.mediabox[3]))
    cx0, cx1 = sorted(float(v) for v in (obj.cropbox[0], obj.cropbox[2]))
    cy0, cy1 = sorted(float(v) for v in (obj.cropbox[1], obj.cropbox[3]))
    vx0, vy0, vx1, vy1 = max(mx0, cx0), max(my0, cy0), min(mx1, cx1), min(my1, cy1)
    if vx1 - vx0 < _MIN_SIDE or vy1 - vy0 < _MIN_SIDE:  # no overlap: the MediaBox
        vx0, vy0, vx1, vy1 = mx0, my0, mx1, my1
    rotation = int(getattr(page, "rotation", 0) or 0) % 360
    if rotation == 90:
        return vy0, vx0 - 2 * mx0, vy1 - vy0, vx1 - vx0
    if rotation == 180:
        return mx1 + mx0 - vx1, vy0 - 2 * my0, vx1 - vx0, vy1 - vy0
    if rotation == 270:
        return my1 + my0 - vy1, mx1 - mx0 - vx1, vy1 - vy0, vx1 - vx0
    return vx0, (my1 - my0) - vy1, vx1 - vx0, vy1 - vy0


def visible_page(page: Any) -> tuple[Any, float, float]:
    """The pdfplumber page cut to its visible box (objects outside it, such as printer marks,
    are dropped), and the ``(dx, dy)`` to subtract from its boxes for display coordinates.
    A page whose visible box is its whole pdfplumber page is not cropped, but may still need
    the shift (a MediaBox whose origin is not 0). Used by element mode only; the fast path
    (``extract_lines``) is not changed by this."""
    try:
        dx, dy, width, height = _visible_frame(page)
    except Exception:  # malformed page boxes: keep pdfplumber's page
        _log.debug("cannot read the page boxes of page %s", page.page_number, exc_info=True)
        return page, 0.0, 0.0
    frame = (dx, dy, dx + width, dy + height)
    if all(abs(a - b) < 0.01 for a, b in zip(frame, page.bbox, strict=True)):
        return page, dx, dy
    try:
        return page.crop(frame, relative=False, strict=False), dx, dy
    except Exception:
        _log.debug("cannot crop page %s to %s", page.page_number, frame, exc_info=True)
        return page, 0.0, 0.0


def _shifted(raw: Iterable[float], dx: float, dy: float) -> _RawBox:
    x0, y0, x1, y1 = (float(v) for v in raw)
    return (x0 - dx, y0 - dy, x1 - dx, y1 - dy)


def _pdfplumber_page(page: Any, page_index: int, find_tables: bool = True) -> PageObjects:
    number = page_index + 1
    page, dx, dy = visible_page(page)
    width, height = float(page.width), float(page.height)  # displayed size; /Rotate applied
    text_rows = _pdfplumber_page_rows(page, page_index)
    for row in text_rows:
        row["x0"], row["y0"], row["x1"], row["y1"] = _shifted(
            (row["x0"], row["y0"], row["x1"], row["y1"]), dx, dy
        )
    lines = _finalise(text_rows)

    char_count = garbled = 0
    for char in page.chars:
        t, g = _count_chars(str(char["text"]))
        char_count, garbled = char_count + t, garbled + g

    images = _boxes(
        number,
        (_shifted((i["x0"], i["top"], i["x1"], i["bottom"]), dx, dy) for i in page.images),
        width,
        height,
    )
    shapes = [*page.rects, *page.lines, *page.curves]
    drawings, flags = _drawing_clusters(
        number, [_shifted((s["x0"], s["top"], s["x1"], s["bottom"]), dx, dy) for s in shapes],
        _PDFPLUMBER_GAP, width, height,
    )  # fmt: skip
    tables: list[tuple[Box, list[list[str]]]] = []
    if not find_tables:
        pass
    elif len(shapes) > MAX_TABLE_PATHS:
        flags.append("tables_skipped")
    elif shapes and _may_hold_ruled_table(*_pdfplumber_rulings(page)):
        try:
            settings = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
            for table in page.find_tables(table_settings=settings):
                rows = _table_rows(table.extract())
                box = _to_box(number, _shifted(table.bbox, dx, dy), width, height)
                if box is not None and _is_table(rows):
                    tables.append((box, rows))
        except Exception:  # one odd page must not fail the document
            _log.debug("table search failed on page %d", number, exc_info=True)
            flags.append("tables_failed")
    tables.sort(key=lambda t: (t[0].y0, t[0].x0))

    return PageObjects(
        page=number, width=width, height=height, rotation=int(getattr(page, "rotation", 0) or 0),
        lines=lines, images=images, drawings=drawings, tables=tables, char_count=char_count,
        invisible_chars=0, garbled_chars=garbled, n_drawing_paths=len(shapes), flags=flags,
    )  # fmt: skip
