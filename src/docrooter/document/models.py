"""Document data model: sections, table of contents, and the document tree."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, Field, SerializerFunctionWrapHandler, model_serializer

from docrooter.elements.models import Element


class RawHeading(BaseModel):
    """A heading as found by a loader, before the tree is built.

    ``text`` is the body that follows the heading up to the next heading. A preamble block
    (text before the first heading) uses ``level=0``.
    """

    title: str
    level: int
    text: str
    line: int
    page: int | None = None
    heading_score: float = 1.0
    flags: list[str] = Field(default_factory=list)


class Section(BaseModel):
    id: str
    title: str
    level: int
    text: str
    parent_id: str | None = None
    children: list[str] = Field(default_factory=list)
    page_start: int | None = None
    heading_score: float = 1.0
    flags: list[str] = Field(default_factory=list)
    element_ids: list[str] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def _omit_empty_element_ids(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # Keeps the JSON of documents loaded without elements identical to v0.3's.
        data: dict[str, Any] = handler(self)
        if not self.element_ids:
            data.pop("element_ids", None)
        return data


class TocEntry(BaseModel):
    id: str
    title: str
    level: int
    depth: int


class Document(BaseModel):
    source: str
    title: str | None = None
    sections: dict[str, Section] = Field(default_factory=dict)
    root_ids: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
    elements: list[Element] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def _omit_empty_elements(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # Keeps the JSON of documents loaded without elements identical to v0.3's.
        data: dict[str, Any] = handler(self)
        if not self.elements:
            data.pop("elements", None)
        return data

    def walk(self) -> Iterator[Section]:
        """Yield every section in pre-order (reading order). Iterative, no recursion."""
        stack = list(reversed(self.root_ids))
        while stack:
            section = self.sections[stack.pop()]
            yield section
            stack.extend(reversed(section.children))

    def toc(self) -> list[TocEntry]:
        """Table of contents in reading order. The preamble (level 0) is not a heading."""
        entries: list[TocEntry] = []
        stack = [(sid, 0) for sid in reversed(self.root_ids)]
        while stack:
            sid, depth = stack.pop()
            section = self.sections[sid]
            if section.level > 0:
                entries.append(
                    TocEntry(id=section.id, title=section.title, level=section.level, depth=depth)
                )
            stack.extend((child, depth + 1) for child in reversed(section.children))
        return entries

    def section(self, key: str) -> Section:
        """Look a section up by id, else by title (first in reading order)."""
        if key in self.sections:
            return self.sections[key]
        for section in self.walk():
            if section.title == key:
                return section
        raise KeyError(key)

    def leaves(self) -> list[Section]:
        """Sections without children, in reading order."""
        return [s for s in self.walk() if not s.children]

    @classmethod
    def from_sections(cls, entries: list[dict], *, source: str = "<sections>") -> Document:
        """Build a document from ``{"title", "level", "text"?}`` dicts in reading order."""
        from docrooter.document.build import build_document

        raw = [
            RawHeading(
                title=entry["title"],
                level=entry["level"],
                text=entry.get("text", ""),
                line=index + 1,
            )
            for index, entry in enumerate(entries)
        ]
        return build_document(raw, source=source)
