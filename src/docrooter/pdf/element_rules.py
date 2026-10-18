"""Geometry and text predicates for the element typing rules in :mod:`docrooter.pdf.elements`.

The tunable thresholds live in ``elements``; the patterns here are fixed by spec §5.3
(``CAPTION_RE`` is re-exported from ``elements``).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from docrooter.elements.models import Box
from docrooter.pdf.lines import Line
from docrooter.pdf.signals import _ROMAN

Rect = tuple[float, float, float, float]  # x0, y0, x1, y1; points, origin top-left

MIN_GAP = 0.5  # pt; smaller gaps and sides are box overlap and rounding, not layout

CAPTION_RE = re.compile(
    r"^(Figure|Fig\.|Table|Exhibit|Chart|Graph|Diagram)\s*[\dIVXLC]+[.:—–-]?", re.IGNORECASE
)

# A marker, then words: "1 See", "1The", "(2) The", "* Unaudited". A number before figures
# ("688 $ (4,638)") or a decimal ("12.5%") is a table row or a value, not a marker.
_FOOTNOTE_MARKER = re.compile(
    r"^(\(\d{1,3}\)|\d{1,3}(?![.,]\d)|[*†‡§]{1,3})[.)]?\s*(?=[^\W\d_]|[\"'“‘(\[])"
)
_BULLET = re.compile(r"^(?:[•◦▪]\s*|[-–*]\s+)(?=\S)")
_ENUMERATOR = re.compile(
    r"^(\d{1,3}[.)]|\((?:\d{1,3}|[a-z]|[ivxlc]{1,6})\)|[a-z][.)]|[ivxlc]{1,6}[.)])\s+(?=\S)"
)
_EQUATION_NUMBER = re.compile(r"^\(\d{1,3}(?:\.\d{1,3})?[a-z]?\)$")
_MATH_FONT = re.compile(r"math|cmmi|cmsy|symbol", re.IGNORECASE)
_MONO_FONT = re.compile(
    r"mono|courier|consol|menlo|inconsolata|sourcecode|typewriter|cmtt|code", re.IGNORECASE
)
_MATH_CHARS = frozenset(
    "=+−×÷±∓<>≤≥≠≈≡∼∝∞√∂∇∑∏∫∮∈∉∋⊂⊃⊆⊇∪∩∧∨¬∀∃→←↔⇒⇐⇔·∘′″^|⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
)
_NOTE_LABEL = re.compile(r"^(?:sources?|notes?)\b", re.IGNORECASE)  # a chart's own notes
_TERMINAL = re.compile(r"[.!?:;][\"'”’»)\]]*$")  # closing quotes or brackets after punctuation


# --- geometry ------------------------------------------------------------------------------


def line_rect(line: Line) -> Rect:
    return line.x0, line.y0, line.x1, line.y1


def area(r: Rect) -> float:
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def overlap_area(a: Rect, b: Rect) -> float:
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def inside_share(inner: Rect, outer: Rect) -> float:
    """Share of ``inner`` inside ``outer``; a box without area counts by its centre."""
    size = area(inner)
    if size <= 0:
        cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
        return float(outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3])
    return overlap_area(inner, outer) / size


def h_overlap(a: Rect, b: Rect) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0])


def same_row(a: Line, b: Line) -> bool:
    """Vertical overlap of at least half the shorter line's height."""
    overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
    return overlap >= 0.5 * min(a.y1 - a.y0, b.y1 - b.y0)


def union(rects: Sequence[Rect]) -> Rect:
    return (min(r[0] for r in rects), min(r[1] for r in rects),
            max(r[2] for r in rects), max(r[3] for r in rects))  # fmt: skip


def text_box(page: int, r: Rect, width: float, height: float) -> Box:
    """``r`` clipped to a ``width`` x ``height`` page (unclipped if the size is unknown); a box
    squeezed to nothing keeps zero size, so a text element always has its box."""
    x0, y0, x1, y1 = r
    if width > 0 and height > 0:
        x0, x1 = min(max(x0, 0.0), width), min(max(x1, 0.0), width)
        y0, y1 = min(max(y0, 0.0), height), min(max(y1, 0.0), height)
    return Box(page=page, x0=round(x0, 2), y0=round(y0, 2),
               x1=round(max(x0, x1), 2), y1=round(max(y0, y1), 2))  # fmt: skip


def clip_box(box: Box, width: float, height: float) -> Rect | None:
    """An object box clipped to its page, or ``None`` when nothing with area is left."""
    x0, y0 = max(0.0, min(box.x0, box.x1)), max(0.0, min(box.y0, box.y1))
    x1, y1 = min(width, max(box.x0, box.x1)), min(height, max(box.y0, box.y1))
    if x1 - x0 < MIN_GAP or y1 - y0 < MIN_GAP:
        return None
    return x0, y0, x1, y1


# --- text ----------------------------------------------------------------------------------


def join_text(texts: Sequence[str]) -> str:
    """Line texts joined with single spaces, whitespace collapsed, blank lines dropped."""
    return " ".join(" ".join(t.split()) for t in texts if t.strip())


def caption_match(text: str) -> bool:
    """``CAPTION_RE``, except that a numeral made of letters must end the word
    ("Table C." is a caption, "Figure illustrates" is not)."""
    match = CAPTION_RE.match(text)
    if match is None:
        return False
    numeral = text[match.end(1) : match.end()].strip().rstrip(".:—–-")
    following = text[match.end() : match.end() + 1]
    return any(c.isdigit() for c in numeral) or not following.isalpha()


def ends_sentence(text: str) -> bool:
    """Ends in ``.``, ``!``, ``?``, ``:`` or ``;``, possibly followed by closing quotes or
    brackets."""
    return bool(_TERMINAL.search(text.rstrip()))


def footnote_marker(text: str) -> re.Match[str] | None:
    """A leading footnote marker (1-3 digits, bare or in parentheses, ``*``, ``†``, ``‡``,
    ``§``) followed by words; group 1 is the marker. Years, decimals and numbers followed by
    more figures are not markers."""
    return _FOOTNOTE_MARKER.match(text)


def list_marker(text: str) -> re.Match[str] | None:
    """A leading bullet (``•``, ``◦``, ``▪``, ``-``, ``–``, ``*``) or enumerator (``1.``,
    ``1)``, ``(a)``, ``a.``, ``i.``, ``(iv)``) followed by text; ``group(0)`` is the marker
    with its trailing space. Letter enumerators are lowercase; a multi-letter one must be a
    roman numeral ("civil." is a word)."""
    bullet = _BULLET.match(text)
    if bullet is not None:
        return bullet
    match = _ENUMERATOR.match(text)
    if match is None:
        return None
    label = match.group(1).strip("().")
    if label.isalpha() and len(label) > 1 and not _ROMAN.match(label):
        return None
    return match


def is_equation_number(text: str) -> bool:
    return bool(_EQUATION_NUMBER.match(text))


def is_math_font(font: str) -> bool:
    return bool(_MATH_FONT.search(font))


def has_math_content(text: str) -> bool:
    return any(c.isalnum() or c in _MATH_CHARS for c in text)


def symbol_share(text: str) -> float:
    """Share of the non-space characters that are math symbols: operators, superscript and
    subscript digits, and Greek or math-alphanumeric letters that stand outside words (so
    Greek prose does not count)."""
    total = sum(1 for c in text if not c.isspace())
    if not total:
        return 0.0
    count = 0
    for k, c in enumerate(text):
        if c in _MATH_CHARS:
            count += 1
        elif "Ͱ" <= c <= "Ͽ" or "\U0001d400" <= c <= "\U0001d7ff":
            before = text[k - 1] if k else " "
            after = text[k + 1] if k + 1 < len(text) else " "
            count += not (before.isalpha() or after.isalpha())
    return count / total


# --- what the text inside a figure candidate is ------------------------------------------


def _is_numeric(text: str) -> bool:
    alnum = [c for c in text if c.isalnum()]
    return bool(alnum) and sum(c.isdigit() for c in alnum) >= 0.5 * len(alnum)


def _is_worded(text: str) -> bool:
    return sum(c.isalpha() for c in text) >= 3


def _is_labelled_row(row: Sequence[Line]) -> bool:
    """A statement row: a word label first, then two or more numeric cells, as statements
    compare periods ("Net income | 112,010 | 93,736"). A chart legend entry beside an axis
    tick has one number; an axis row starts with a number."""
    return _is_worded(row[0].text) and sum(_is_numeric(c.text) for c in row[1:]) >= 2


def is_text_grid(
    lines: Sequence[Line],
    *,
    min_rows: int,
    min_columns: int,
    tolerance: float,
    min_labelled_rows: int,
    line_share: float,
) -> bool:
    """Whether ``lines`` form a table. At least ``line_share`` of them must sit in rows of two
    or more lines with a worded cell (a chart's lines are mostly lone ticks, axis rows and
    legend entries). Then either ``min_labelled_rows`` statement rows (a word label, then two
    or more numeric cells) or ``min_columns`` columns make a table: a column is a left or
    right edge (within ``tolerance`` pt) shared by at least ``min_rows`` such rows, and cells
    aligned on both edges count once."""
    rows: list[list[Line]] = []
    for line in sorted(lines, key=lambda ln: (ln.y0, ln.x0)):
        if rows and same_row(rows[-1][0], line):
            rows[-1].append(line)
        else:
            rows.append([line])
    # rows of numbers alone are a chart's ticks (two y axes give "35 | 350"), not a table
    multi = [
        sorted(row, key=lambda ln: ln.x0)
        for row in rows
        if len(row) >= 2 and any(_is_worded(cell.text) for cell in row)
    ]
    if sum(len(row) for row in multi) < line_share * len(lines):
        return False  # mostly lone ticks and axis labels, as charts are
    if sum(_is_labelled_row(row) for row in multi) >= min_labelled_rows:
        return True
    if len(multi) < min_rows:
        return False
    clusters: list[set[tuple[int, int]]] = []  # (row, cell) members of each aligned edge
    for edge in (0, 1):  # left edges, then right edges
        points = sorted(
            ((cell.x1 if edge else cell.x0), r, c)
            for r, row in enumerate(multi)
            for c, cell in enumerate(row)
        )
        start = None
        for x, r, c in points:
            if start is None or x - start > tolerance:
                start = x
                clusters.append(set())
            clusters[-1].add((r, c))
    columns: list[set[tuple[int, int]]] = []
    for members in sorted(clusters, key=len, reverse=True):
        if len({r for r, _ in members}) < min_rows:
            continue
        taken = sum(1 for m in members if any(m in col for col in columns))
        if taken < len(members) / 2:  # a column both left and right aligned counts once
            columns.append(members)
    return len(columns) >= min_columns


def is_code(lines: Sequence[Line], *, line_share: float) -> bool:
    """Whether at least ``line_share`` of ``lines`` are set in a monospace font (a listing)."""
    mono = sum(1 for line in lines if _MONO_FONT.search(line.font))
    return bool(lines) and mono >= line_share * len(lines)


def is_prose(
    lines: Sequence[Line],
    *,
    min_words: int,
    line_share: float,
    min_sentences: int,
    sentence_share: float,
    run_lines: int,
) -> bool:
    """Whether ``lines`` hold running text: at least ``line_share`` of them have ``min_words``
    words or more; or at least ``min_sentences`` such lines, and ``sentence_share`` of all
    lines, end a sentence; or ``run_lines`` such lines are stacked as a paragraph. A chart's
    "Source: …" or "Note: …" lines among its labels are not running text."""
    if not lines:
        return False
    long = [line for line in lines if len(line.text.split()) >= min_words]
    sentences = sum(1 for line in long if ends_sentence(line.text))
    if len(long) >= line_share * len(lines) or (
        sentences >= min_sentences and sentences >= sentence_share * len(lines)
    ):
        return True
    run: list[Line] = []
    for line in sorted(long, key=lambda ln: (ln.y0, ln.x0)):
        prev = run[-1] if run else None
        stacked = (
            prev is not None
            and 0 <= line.y0 - prev.y1 <= prev.y1 - prev.y0
            and h_overlap(line_rect(prev), line_rect(line))
        )
        run = [*run, line] if stacked else [line]
        if len(run) >= run_lines and not _NOTE_LABEL.match(run[0].text.strip()):
            return True
    return False
