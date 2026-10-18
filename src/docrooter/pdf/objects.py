"""Engine-neutral page objects: the text lines, raster images, vector-graphics regions, ruled
tables and character statistics of every page, from ``pymupdf`` or ``pdfplumber``.

All boxes are in points in the *displayed* page's coordinates (origin top left, ``/Rotate``
applied) and are clipped to the page, so downstream code never has to know the engine or the
page rotation. On an unrotated page ``lines`` equal :func:`docrooter.pdf.lines.extract_lines`
for the same engine, in content and order; on a rotated page they are mapped into display
space as well (``extract_lines`` leaves pymupdf lines in unrotated page space, and its output
is frozen).

Vector-heavy pages (maps, plots with tens of thousands of paths) are bounded: above
``MAX_DRAWING_PATHS`` paths the drawing area becomes a single region and no table search runs,
and the page gets the flag ``"drawings_capped"``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from docrooter.elements.models import Box
from docrooter.errors import DocrooterError
from docrooter.pdf.lines import (
    PROGRESS_EVERY,
    Engine,
    Line,
    Progress,
    _finalise,
    _import_pdfplumber,
    _import_pymupdf,
    _pdfplumber_page_rows,
    _pymupdf_flags,
    _pymupdf_page_rows,
)

__all__ = ["MAX_DRAWING_PATHS", "PageObjects", "extract_page_objects"]

MAX_DRAWING_PATHS = 5000  # per page; above it the whole drawing area becomes one cluster
MAX_DRAWING_CLUSTERS = 200  # per page; the largest are kept
MAX_TABLE_PATHS = 1500  # per page; the table search is skipped on pages with more paths
_MIN_TABLE_ITEMS = 3  # fewer drawing items than this (lines, rectangles) cannot form a grid
_PYMUPDF_GAP = 3.0  # pt; drawing items this close join one cluster (``cluster_drawings`` default)
_PDFPLUMBER_GAP = 6.0  # pt; the same for pdfplumber rects, lines and curves
_MIN_SIDE = 0.5  # pt; a box thinner than this (after clipping) is degenerate and dropped
_MIN_TABLE_ROWS = 2
_MIN_TABLE_COLS = 2

_CID = re.compile(r"\(cid:\d+\)")  # pdfminer's rendering of a glyph without a Unicode mapping
_PLAIN_SPACE = "\t\n\r\x0b\x0c"

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
) -> list[PageObjects]:
    """Extract the :class:`PageObjects` of every page (of the first ``max_pages``) of ``path``.

    ``progress(done, total)`` is called after every 10th page, as in ``extract_lines``.

    Raises ``DocrooterError`` for encrypted, corrupt or non-PDF files and ``MissingExtraError``
    when the engine's package is not installed.
    """
    if engine == "pymupdf":
        return _extract_pymupdf(path, max_pages, progress)
    if engine == "pdfplumber":
        return _extract_pdfplumber(path, max_pages, progress)
    raise DocrooterError(f"unknown PDF engine {engine!r}; use 'pymupdf' or 'pdfplumber'")


# --- shared helpers ------------------------------------------------------------------------


def _is_garbled(ch: str) -> bool:
    o = ord(ch)
    return (
        o == 0xFFFD
        or 0xE000 <= o <= 0xF8FF  # Private Use Area
        or (o >= 0xF0000 and (o & 0xFFFF) <= 0xFFFD)  # private use planes 15 and 16
        or (o < 0x20 and ch not in _PLAIN_SPACE)  # C0 controls (tab, newline, … are spacing)
        or 0x7F <= o <= 0x9F  # DEL and the C1 controls
    )


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


def _count_plain(text: str) -> tuple[int, int]:
    total = garbled = 0
    for ch in text:
        if _is_garbled(ch):
            total += 1
            garbled += 1
        elif not ch.isspace():
            total += 1
    return total, garbled


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


# --- pymupdf -------------------------------------------------------------------------------


def _extract_pymupdf(
    path: str | Path, max_pages: int | None, progress: Progress | None
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
                out.append(_pymupdf_page(page, page_index, pymupdf, flags, table_kwargs))
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
    page: Any, page_index: int, pymupdf: Any, text_flags: int, table_kwargs: dict[str, Any]
) -> PageObjects:
    number = page_index + 1
    width, height = float(page.rect.width), float(page.rect.height)
    rotation = int(page.rotation)
    matrix = page.rotation_matrix if rotation else None  # unrotated space -> display space

    def display(raw: Iterable[float]) -> _RawBox:
        x0, y0, x1, y1 = (float(v) for v in raw)
        if matrix is None:
            return x0, y0, x1, y1
        pts = [
            (matrix.a * x + matrix.c * y + matrix.e, matrix.b * x + matrix.d * y + matrix.f)
            for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))
        ]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    lines = _finalise(list(_pymupdf_page_rows(page, page_index, text_flags, matrix)))

    char_count, garbled = _count_chars(page.get_text("text", flags=text_flags))
    invisible = sum(
        1
        for span in page.get_texttrace()
        if span.get("type") == 3  # text render mode 3: neither filled nor stroked (an OCR layer)
        for ch in span["chars"]
        if not chr(ch[0]).isspace()
    )

    image_flags = pymupdf.TEXT_PRESERVE_IMAGES | pymupdf.TEXT_MEDIABOX_CLIP
    image_raws = [
        display(b[:4]) for b in page.get_text("blocks", flags=image_flags) if b[6] == 1
    ] + [display(info["bbox"]) for info in page.get_image_info()]
    images = _boxes(number, image_raws, width, height)

    paths = page.get_cdrawings()
    drawings, flags = _drawing_clusters(
        number, [display(p["rect"]) for p in paths], _PYMUPDF_GAP, width, height
    )
    tables: list[tuple[Box, list[list[str]]]] = []
    if len(paths) > MAX_TABLE_PATHS:
        flags.append("tables_skipped")
    elif sum(len(p["items"]) for p in paths) >= _MIN_TABLE_ITEMS:  # no rules, no ruled table
        try:
            found = page.find_tables(strategy="lines", **table_kwargs).tables
            for table in found:
                rows = _table_rows(table.extract())
                box = _to_box(number, display(table.bbox), width, height)
                if box is not None and _is_table(rows):
                    tables.append((box, rows))
        except Exception:  # one odd page must not fail the document
            flags.append("tables_failed")
    tables.sort(key=lambda t: (t[0].y0, t[0].x0))

    return PageObjects(
        page=number, width=width, height=height, rotation=rotation, lines=lines, images=images,
        drawings=drawings, tables=tables, char_count=char_count, invisible_chars=invisible,
        garbled_chars=garbled, n_drawing_paths=len(paths), flags=flags,
    )  # fmt: skip


# --- pdfplumber ----------------------------------------------------------------------------


def _extract_pdfplumber(
    path: str | Path, max_pages: int | None, progress: Progress | None
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
                out.append(_pdfplumber_page(page, page_index))
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


def _pdfplumber_page(page: Any, page_index: int) -> PageObjects:
    number = page_index + 1
    width, height = float(page.width), float(page.height)  # displayed size; /Rotate applied
    lines = _finalise(_pdfplumber_page_rows(page, page_index))

    char_count = garbled = 0
    for char in page.chars:
        t, g = _count_chars(str(char["text"]))
        char_count, garbled = char_count + t, garbled + g

    images = _boxes(
        number, ((i["x0"], i["top"], i["x1"], i["bottom"]) for i in page.images), width, height
    )
    shapes = [*page.rects, *page.lines, *page.curves]
    drawings, flags = _drawing_clusters(
        number, [(s["x0"], s["top"], s["x1"], s["bottom"]) for s in shapes],
        _PDFPLUMBER_GAP, width, height,
    )  # fmt: skip
    tables: list[tuple[Box, list[list[str]]]] = []
    if len(shapes) > MAX_TABLE_PATHS:
        flags.append("tables_skipped")
    elif len(shapes) >= _MIN_TABLE_ITEMS:  # no rules, no ruled table
        try:
            settings = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
            for table in page.find_tables(table_settings=settings):
                rows = _table_rows(table.extract())
                box = _to_box(number, table.bbox, width, height)
                if box is not None and _is_table(rows):
                    tables.append((box, rows))
        except Exception:  # one odd page must not fail the document
            flags.append("tables_failed")
    tables.sort(key=lambda t: (t[0].y0, t[0].x0))

    return PageObjects(
        page=number, width=width, height=height, rotation=int(getattr(page, "rotation", 0) or 0),
        lines=lines, images=images, drawings=drawings, tables=tables, char_count=char_count,
        invisible_chars=0, garbled_chars=garbled, n_drawing_paths=len(shapes), flags=flags,
    )  # fmt: skip
