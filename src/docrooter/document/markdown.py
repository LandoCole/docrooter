"""Markdown heading extraction (ATX and setext, fenced code aware)."""

from __future__ import annotations

import re

from docrooter.document.models import RawHeading

_ATX = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_SETEXT_H1 = re.compile(r"^ {0,3}=+[ \t]*$")
_SETEXT_H2 = re.compile(r"^ {0,3}-+[ \t]*$")
_YAML_KEY = re.compile(r"^[\w-]+[ \t]*:")
# A line that cannot be the text of a setext heading (it starts another block type).
_NOT_SETEXT_TEXT = re.compile(r"^ {0,3}(?:[-*+][ \t]|\d+[.)][ \t]|>|\||#{1,6}(?:[ \t]|$))")


def _skip_front_matter(lines: list[str]) -> int:
    """Return the index of the first line after a leading YAML front-matter block.

    The block only counts when it looks like YAML: it opens with a ``key:`` line and every
    non-blank line is a ``key:`` line, indented, a ``- `` list item or a ``#`` comment.
    Otherwise the leading ``---`` is an ordinary horizontal rule and nothing is skipped.
    """
    if not lines or lines[0].strip() != "---":
        return 0
    for end in range(1, len(lines)):
        if lines[end].strip() in ("---", "..."):
            block = lines[1:end]
            break
    else:
        return 0
    if not block or not _YAML_KEY.match(block[0]):
        return 0
    for line in block:
        if line.strip() and not (
            _YAML_KEY.match(line) or line[0] in " \t" or line.startswith(("- ", "#"))
        ):
            return 0
    return end + 1


def extract_headings(markdown: str) -> list[RawHeading]:
    """Split Markdown into headings with the body text that follows each one.

    Headings inside fenced code blocks are ignored. Text before the first heading becomes a
    ``level=0`` preamble block (only when it is non-blank).
    """
    lines = markdown.splitlines()
    start = _skip_front_matter(lines)

    headings: list[RawHeading] = []
    body: list[str] = []
    current: tuple[str, int, int] | None = None  # (title, level, line)
    fence: tuple[str, int] | None = None  # (char, length)

    def flush() -> None:
        text = "\n".join(body).strip()
        if current is not None:
            title, level, line = current
            headings.append(RawHeading(title=title, level=level, text=text, line=line))
        elif text:
            headings.append(RawHeading(title="Preamble", level=0, text=text, line=1))
        body.clear()

    index = start
    while index < len(lines):
        line = lines[index]
        fence_match = _FENCE.match(line)

        if fence is not None:
            body.append(line)
            if (
                fence_match
                and fence_match.group(1)[0] == fence[0]
                and len(fence_match.group(1)) >= fence[1]
                and not line.strip().strip(fence[0])
            ):
                fence = None
            index += 1
            continue

        if fence_match:
            fence = (fence_match.group(1)[0], len(fence_match.group(1)))
            body.append(line)
            index += 1
            continue

        atx = _ATX.match(line)
        if atx:
            flush()
            current = (atx.group(2).strip(), len(atx.group(1)), index + 1)
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
                flush()
                current = (line.strip(), level, index + 1)
                index += 2
                continue

        body.append(line)
        index += 1

    flush()
    return headings
