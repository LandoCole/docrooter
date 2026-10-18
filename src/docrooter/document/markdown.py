"""Markdown parsing: headings (ATX and setext, fenced code aware) and block elements."""

from __future__ import annotations

import re
from collections.abc import Iterator

from docrooter.document.build import build_document
from docrooter.document.headings import repair_levels, validate_headings
from docrooter.document.models import Document, RawHeading
from docrooter.elements.models import Element, ElementKind, FigureData, TableData

MARKDOWN_MIN_SCORE = 0.5

_ATX = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_SETEXT_H1 = re.compile(r"^ {0,3}=+[ \t]*$")
_SETEXT_H2 = re.compile(r"^ {0,3}-+[ \t]*$")
_YAML_KEY = re.compile(r"^[A-Za-z_][\w-]*\s*:")
_FRONT_MATTER_MAX_LINES = 50
# A line that cannot be the text of a setext heading (it starts another block type).
_NOT_SETEXT_TEXT = re.compile(r"^ {0,3}(?:[-*+][ \t]|\d+[.)][ \t]|>|\||#{1,6}(?:[ \t]|$))")

# Block patterns for the element pass.
_LIST_ITEM = re.compile(
    r"^(?P<indent>[ \t]*)(?P<marker>[-*+]|(?P<number>\d{1,9})[.)])[ \t]+(?P<text>\S.*)$"
)
_THEMATIC_BREAK = re.compile(r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
_TABLE_DELIMITER = re.compile(r"^[ \t]*\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$")
_IMAGE = re.compile(r"^[ \t]*!\[(?P<alt>[^\]]*)\]\([^)]*\)[ \t]*$")
_QUOTE = re.compile(r"^ {0,3}(?:>[ \t]?)+")
_UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")


def _skip_front_matter(lines: list[str]) -> int:
    """Return the index of the first line after a leading YAML front-matter block.

    Deliberately conservative, so content is never lost: the document must open with ``---``,
    a closing ``---``/``...`` must follow within ``_FRONT_MATTER_MAX_LINES`` lines, the block must
    contain a top-level ``key:`` line, and every non-blank line must be a ``key:`` line, indented,
    or a ``- `` list item. A line whose stripped form starts with ``#`` disqualifies the block
    (YAML comments in front matter are not supported). Otherwise the leading ``---`` is ordinary
    text.
    """
    if not lines or lines[0].rstrip() != "---":
        return 0
    for end in range(1, min(len(lines), _FRONT_MATTER_MAX_LINES + 1)):
        if lines[end].rstrip() in ("---", "..."):
            break
    else:
        return 0
    block = lines[1:end]
    if not any(_YAML_KEY.match(line) for line in block):
        return 0
    for line in block:
        if line.lstrip().startswith("#"):
            return 0  # may be an (indented) ATX heading
        if line.strip() and not (
            _YAML_KEY.match(line) or line[0] in " \t" or line.startswith("- ")
        ):
            return 0
    return end + 1


def _scan(lines: list[str], start: int) -> Iterator[tuple[str, int, str, int]]:
    """The line scanner shared by heading and element extraction.

    Yields ``(kind, index, text, level)`` in document order, where ``index`` is the 0-based index
    of the first line the item covers:

    - ``"heading"``: ``text`` is the title and ``level`` its level (1-6); a setext heading covers
      two lines;
    - ``"fence_open"`` / ``"fence_close"``: a code-fence marker line (``level`` is 0);
    - ``"code"``: a line inside a fenced code block (``level`` is 0);
    - ``"text"``: any other line (``level`` is 0).
    """
    fence: tuple[str, int] | None = None  # (char, length)
    index = start
    while index < len(lines):
        line = lines[index]
        fence_match = _FENCE.match(line)

        if fence is not None:
            if (
                fence_match
                and fence_match.group(1)[0] == fence[0]
                and len(fence_match.group(1)) >= fence[1]
                and not line.strip().strip(fence[0])
            ):
                fence = None
                yield "fence_close", index, line, 0
            else:
                yield "code", index, line, 0
            index += 1
            continue

        if fence_match:
            fence = (fence_match.group(1)[0], len(fence_match.group(1)))
            yield "fence_open", index, line, 0
            index += 1
            continue

        atx = _ATX.match(line)
        if atx:
            yield "heading", index, atx.group(2).strip(), len(atx.group(1))
            index += 1
            continue

        if (
            index + 1 < len(lines)
            and line.strip()
            and not _NOT_SETEXT_TEXT.match(line)
            and not _SETEXT_H1.match(line)
            and not _SETEXT_H2.match(line)
        ):
            underline = lines[index + 1]
            level = 1 if _SETEXT_H1.match(underline) else 2 if _SETEXT_H2.match(underline) else 0
            if level:
                yield "heading", index, line.strip(), level
                index += 2
                continue

        yield "text", index, line, 0
        index += 1


def _extract_headings(lines: list[str], start: int) -> list[RawHeading]:
    headings: list[RawHeading] = []
    body: list[str] = []
    current: tuple[str, int, int] | None = None  # (title, level, line)

    def flush() -> None:
        text = "\n".join(body).strip()
        if current is not None:
            title, level, line = current
            headings.append(RawHeading(title=title, level=level, text=text, line=line))
        elif text:
            headings.append(RawHeading(title="Preamble", level=0, text=text, line=1))
        body.clear()

    for kind, index, text, level in _scan(lines, start):
        if kind == "heading":
            flush()
            current = (text, level, index + 1)
        else:
            body.append(text)

    flush()
    return headings


def extract_headings(markdown: str) -> list[RawHeading]:
    """Split Markdown into headings with the body text that follows each one.

    Headings inside fenced code blocks are ignored. Text before the first heading becomes a
    ``level=0`` preamble block (only when it is non-blank).
    """
    lines = markdown.splitlines()
    return _extract_headings(lines, _skip_front_matter(lines))


def parse_markdown(
    text: str,
    *,
    source: str,
    elements: bool = False,
    min_heading_score: float = MARKDOWN_MIN_SCORE,
    repair: bool = True,
    fallback_title: str | None = None,
) -> Document:
    """Parse Markdown text into a :class:`Document`.

    Headings are extracted, validated (scores below ``min_heading_score`` are merged into the
    preceding text) and, if ``repair`` is true, have their levels repaired. The title is the first
    level-1 heading, else ``fallback_title``. With ``elements=True`` the document also carries
    typed block elements (``Document.elements``, ``Section.element_ids``); without it the result
    is identical to v0.3's.
    """
    lines = text.splitlines()
    start = _skip_front_matter(lines)
    raw = validate_headings(_extract_headings(lines, start), min_score=min_heading_score)
    if repair:
        raw = repair_levels(raw)
    title = next((h.title for h in raw if h.level == 1), fallback_title)
    doc = build_document(raw, source=source, title=title)
    if elements:
        _attach_elements(doc, raw, lines, start)
    return doc


def _split_row(line: str) -> list[str]:
    row = line.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|") and not row.endswith("\\|"):
        row = row[:-1]
    return [cell.strip().replace("\\|", "|") for cell in _UNESCAPED_PIPE.split(row)]


def _is_table_start(line: str, following: str | None) -> bool:
    if "|" not in line or following is None or "|" not in following:
        return False
    if not _TABLE_DELIMITER.match(following):
        return False
    return len(_split_row(line)) == len(_split_row(following))


class _ElementBuilder:
    """Turns the scanner's events into elements, tracking the current section."""

    def __init__(self, heading_sections: dict[int, tuple[int, str]], first_section: str | None):
        self.elements: list[Element] = []
        self.owned: dict[str, list[str]] = {}
        self._heading_sections = heading_sections  # 1-based line -> (repaired level, section id)
        self._section = first_section
        # An open paragraph or list item: (kind, lines, marker, depth).
        self._open: tuple[ElementKind, list[str], str | None, int | None] | None = None
        self._table: list[list[str]] | None = None
        self._formula: list[str] | None = None
        self._code: list[str] | None = None
        self._indents: list[int] = []  # list indentation stack, to derive item depth

    # -- emission ---------------------------------------------------------------------------
    def _emit(self, kind: ElementKind, text: str, **fields: object) -> None:
        index = len(self.elements)
        element = Element(
            id=f"e{index}",
            kind=kind,
            text=text,
            boxes=[],
            order=index,
            section_id=self._section,
            **fields,  # type: ignore[arg-type]
        )
        self.elements.append(element)
        if self._section is not None:
            self.owned.setdefault(self._section, []).append(element.id)

    def _flush_open(self) -> None:
        if self._open is None:
            return
        kind, parts, marker, depth = self._open
        self._open = None
        text = " ".join(parts).strip()
        if kind == "list_item":
            self._emit(kind, text, marker=marker, list_depth=depth)
        else:
            self._indents.clear()
            self._emit(kind, text)

    def _flush_table(self) -> None:
        if self._table is None:
            return
        rows, self._table = self._table, None
        self._indents.clear()
        self._emit(
            "table",
            "\n".join(" ".join(cell for cell in row if cell) for row in rows),
            table=TableData(rows=rows, header_rows=1),
        )

    def _flush_formula(self) -> None:
        if self._formula is None:
            return
        parts, self._formula = self._formula, None
        text = "\n".join(p.strip() for p in parts if p.strip())
        if text:
            self._indents.clear()
            self._emit("formula", text)

    def _flush_code(self) -> None:
        if self._code is None:
            return
        lines, self._code = self._code, None
        text = "\n".join(lines).strip("\n")
        if text.strip():
            self._indents.clear()
            self._emit("paragraph", text, flags=["code"])

    def flush(self) -> None:
        self._flush_open()
        self._flush_table()
        self._flush_formula()
        self._flush_code()

    # -- events -----------------------------------------------------------------------------
    def heading(self, index: int, title: str) -> None:
        self.flush()
        self._indents.clear()
        found = self._heading_sections.get(index + 1)
        if found is None:  # rejected by validation: its text stays ordinary text
            self._emit("paragraph", title)
            return
        level, self._section = found
        self._emit("heading", title, level=level)

    def fence_open(self) -> None:
        self.flush()
        self._code = []

    def fence_close(self) -> None:
        self._flush_code()

    def code(self, line: str) -> None:
        if self._code is not None:
            self._code.append(line)

    def text(self, line: str, following: str | None, pending_delimiter: bool) -> bool:
        """Handle one text line. Returns True when the next line is a table delimiter to skip."""
        if pending_delimiter:
            return False
        if self._formula is not None:
            if "$$" in line:
                self._formula.append(line.split("$$", 1)[0])
                self._flush_formula()
            else:
                self._formula.append(line)
            return False

        if self._table is not None:
            if "|" in line and line.strip():
                self._table.append(_split_row(line))
                return False
            self._flush_table()

        line = _QUOTE.sub("", line)
        stripped = line.strip()
        if not stripped:
            self._flush_open()
            return False

        if _THEMATIC_BREAK.match(line):
            self._flush_open()
            self._indents.clear()
            return False

        image = _IMAGE.match(line)
        if image:
            self._flush_open()
            self._indents.clear()
            self._emit("figure", image.group("alt").strip(), figure=FigureData(source="image"))
            return False

        if stripped.startswith("$$"):
            self._flush_open()
            rest = stripped[2:]
            if "$$" in rest:
                self._formula = [rest.split("$$", 1)[0]]
                self._flush_formula()
            else:
                self._formula = [rest]
            return False

        if _is_table_start(line, following):
            self._flush_open()
            self._table = [_split_row(line)]
            return True

        item = _LIST_ITEM.match(line)
        if item and not (
            self._open is not None
            and self._open[0] == "paragraph"
            and item.group("number") is not None
            and int(item.group("number")) != 1
        ):
            self._flush_open()
            indent = len(item.group("indent").expandtabs(4))
            while self._indents and self._indents[-1] > indent:
                self._indents.pop()
            if not self._indents or self._indents[-1] < indent:
                self._indents.append(indent)
            self._open = (
                "list_item",
                [item.group("text").strip()],
                item.group("marker"),
                len(self._indents) - 1,
            )
            return False

        if self._open is None:
            self._open = ("paragraph", [stripped], None, None)
        else:
            self._open[1].append(stripped)
        return False


def _attach_elements(doc: Document, raw: list[RawHeading], lines: list[str], start: int) -> None:
    """Build the element list and fill ``Document.elements`` and ``Section.element_ids``."""
    # build_document creates one section per raw block, in order, so they zip.
    heading_sections = {
        block.line: (block.level, section.id)
        for block, section in zip(raw, doc.sections.values(), strict=True)
        if block.level > 0
    }
    builder = _ElementBuilder(heading_sections, "s0" if "s0" in doc.sections else None)
    skip_delimiter = False
    for kind, index, text, _level in _scan(lines, start):
        if kind == "heading":
            builder.heading(index, text)
        elif kind == "fence_open":
            builder.fence_open()
        elif kind == "fence_close":
            builder.fence_close()
        elif kind == "code":
            builder.code(text)
        else:
            following = lines[index + 1] if index + 1 < len(lines) else None
            skip_delimiter = builder.text(text, following, skip_delimiter)
    builder.flush()

    doc.elements = builder.elements
    for section_id, element_ids in builder.owned.items():
        doc.sections[section_id].element_ids = element_ids
