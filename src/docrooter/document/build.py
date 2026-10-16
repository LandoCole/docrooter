"""Build a :class:`Document` tree from a flat list of headings."""

from __future__ import annotations

from pathlib import Path

from docrooter.document.models import Document, RawHeading, Section


def build_document(raw: list[RawHeading], *, source: str, title: str | None = None) -> Document:
    """Nest headings into a section tree.

    A heading becomes a child of the nearest preceding heading with a smaller level. A
    ``level=0`` block is a preamble: it becomes the root section ``s0`` and never a parent. When
    the document has no headings at all, the preamble takes the source file's stem as its title.
    """
    sections: dict[str, Section] = {}
    root_ids: list[str] = []
    stack: list[Section] = []
    child_counts: dict[str | None, int] = {}  # parent id (None = top level) -> children so far
    has_headings = any(h.level > 0 for h in raw)

    for heading in raw:
        if heading.level <= 0:
            section = Section(
                id="s0",
                title="Preamble" if has_headings else (Path(source).stem or "Document"),
                level=0,
                text=heading.text,
                page_start=heading.page,
                heading_score=heading.heading_score,
                flags=list(heading.flags),
            )
            sections[section.id] = section
            root_ids.append(section.id)
            continue

        while stack and stack[-1].level >= heading.level:
            stack.pop()
        parent = stack[-1] if stack else None
        parent_id = parent.id if parent else None

        ordinal = child_counts.get(parent_id, 0) + 1
        child_counts[parent_id] = ordinal
        section_id = f"{parent.id}.{ordinal}" if parent else f"s{ordinal}"

        section = Section(
            id=section_id,
            title=heading.title,
            level=heading.level,
            text=heading.text,
            parent_id=parent_id,
            page_start=heading.page,
            heading_score=heading.heading_score,
            flags=list(heading.flags),
        )
        sections[section_id] = section
        if parent:
            parent.children.append(section_id)
        else:
            root_ids.append(section_id)
        stack.append(section)

    return Document(source=source, title=title, sections=sections, root_ids=root_ids)
