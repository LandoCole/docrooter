"""Parse a document into one self-contained JSON response, at one of three levels.

:func:`parse` reads a document, classifies it once and returns a :class:`ParseResult`. The
application says up front how much it needs (``include``):

- ``"type"``: the document type, from the doc-type call alone (no section tags);
- ``"sections"``: the type, plus every tagged heading with its section tag;
- ``"text"``: the type and the tags, plus the full text of each tagged section, grouped by tag.
  The text costs no model calls.

The response stores as JSON (``model_dump_json``) and a reader needs only a JSON parser; with
doc-rooter, :meth:`ParseResult.get_section` answers "give me tag X" from the stored text, with an
explicit ``found``, ``not_found`` or ``unknown_tag`` and no model call.

- A section's text is its own body, then every subsection under it. In ``section_text`` each
  leaf tag (``none`` excluded) lists the sections carrying it; a section inside another with the
  same tag is dropped, since its text is already in the outer one, and same-tag siblings stay
  separate. Text can repeat across tags (Item 8 holds the balance sheet).
- A tag matches its own key, and a group tag (``"financial"``) every key under it.
- A tag the catalogue has but the document does not is ``not_found``; a tag the catalogue lacks
  is ``unknown_tag``. Neither raises, since an application asks for tags speculatively.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, SerializerFunctionWrapHandler, model_serializer

from docrooter.backends.base import Backend
from docrooter.classify.classifier import DEFAULT_BACKEND
from docrooter.document import Document, Section
from docrooter.document.load import load
from docrooter.flow import (
    CATALOGUE,
    DOCUMENT_STAGE,
    NO_TAG,
    SECTION_STAGE,
    Classification,
    ConfidenceSource,
    Flow,
    SectionTag,
    classify,
)
from docrooter.ladder import Level
from docrooter.taxonomies import builtin_taxonomy
from docrooter.taxonomy import Taxonomy
from docrooter.views import estimate_tokens

MAX_HEADING_HASHES = 6
"""Markdown has six heading levels; a deeper section is written with six."""

Include = Literal["type", "sections", "text"]
"""How much :func:`parse` returns."""

INCLUDES: tuple[str, ...] = ("type", "sections", "text")
"""The ``include`` values, least to most."""

STAGES = (DOCUMENT_STAGE, SECTION_STAGE)
"""The keys of ``ParseResult.tokens``."""


class SectionMatch(BaseModel):
    """One section that carries the asked tag.

    - ``tag`` is the section's own leaf label, which can differ from the asked tag on a group
      query; ``path`` is the ancestors' titles from the outermost down, without the section
      itself or the preamble.
    - ``text`` is the section's body followed by every descendant (each under a Markdown heading
      of its own); the section's own heading is ``heading``, not part of ``text``.
    - ``page_start`` is the section's page and ``page_end`` an upper bound: the page where the
      next section after it starts, else the last page start within it. Both are ``None``
      without pages (Markdown input).
    - ``confidence`` and ``confidence_source`` are copied from the section's tag.
    - ``tokens`` is ``estimate_tokens(heading + "\\n\\n" + text)``, for budgeting a prompt.
    """

    model_config = ConfigDict(extra="forbid")

    section_id: str
    tag: str
    heading: str
    path: list[str]
    level: int
    text: str
    page_start: int | None
    page_end: int | None
    confidence: float | None
    confidence_source: ConfidenceSource
    tokens: int


class SectionContext(BaseModel):
    """The answer for one asked ``tag``: ``found`` with its ``sections``, ``not_found`` (the
    catalogue has the tag, the document does not) or ``unknown_tag`` (the catalogue lacks it).
    ``message`` explains anything but ``found``."""

    model_config = ConfigDict(extra="forbid")

    tag: str
    status: Literal["found", "not_found", "unknown_tag"]
    message: str | None = None
    sections: list[SectionMatch]


class TaggedSection(BaseModel):
    """One tagged heading in the table of contents of a :class:`ParseResult`: ``path`` is the
    ancestors' titles from the outermost down (without the section itself or the preamble) and
    ``tag`` the section's leaf label, which may be ``none``."""

    model_config = ConfigDict(extra="forbid")

    section_id: str
    heading: str
    level: int
    path: list[str]
    tag: str
    confidence: float | None
    page_start: int | None


class ParseResult(BaseModel):
    """What :func:`parse` returns: a self-contained, JSON-ready answer at one level.

    - ``include`` is the level asked for. ``sections`` (every tagged heading in reading order,
      ``none`` included) is ``None`` at ``"type"``; ``section_text`` (the text of each tagged
      section, by leaf tag in reading order of first appearance, without ``none``) is ``None``
      below ``"text"``. A field left ``None`` for its level is absent from the JSON.
    - ``tokens`` is the tokens spent per stage (``"document"``, ``"sections"``), prompt plus
      completion; a stage that made no call is ``0``.

    ``model_dump_json()`` and ``model_validate_json()`` round-trip, so a stored response can be
    read back and asked with :meth:`get_section`.
    """

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    source: str
    include: Include
    backend: str
    doc_type: str | None
    doc_type_confidence: float | None
    tokens: dict[str, int]
    sections: list[TaggedSection] | None = None
    section_text: dict[str, list[SectionMatch]] | None = None

    @model_serializer(mode="wrap")
    def _omit_unrequested_levels(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        for name in ("sections", "section_text"):
            if getattr(self, name) is None:
                data.pop(name, None)
        return data

    def get_section(self, tag: str, *, catalogue: Taxonomy | None = None) -> SectionContext:
        """The context of ``tag`` (surrounding whitespace is ignored, case is exact), read from
        ``section_text``. A group tag such as ``"financial"`` matches every ``financial.*`` key;
        a match inside another match is dropped. ``catalogue`` (default: the built-in
        ``section_catalogue``) says which tags exist. ``ValueError`` below ``include="text"``."""
        return self._context(tag, catalogue)

    def get_sections(
        self, tags: Sequence[str], *, catalogue: Taxonomy | None = None
    ) -> list[SectionContext]:
        """One context per tag, in the order asked; a repeated tag is answered each time."""
        if isinstance(tags, str):
            raise TypeError("tags must be a sequence of tags, not a single string")
        return [self._context(tag, catalogue) for tag in tags]

    def _context(self, asked: str, catalogue: Taxonomy | None) -> SectionContext:
        if self.section_text is None:
            raise ValueError(
                f'section text was not requested (include="{self.include}"); '
                'parse with include="text" to look up a section'
            )
        tag = asked.strip()
        hits = [
            match
            for key, matches in self.section_text.items()
            if key == tag or key.startswith(tag + ".")
            for match in matches
        ]
        ids = {match.section_id for match in hits}
        hits = [
            match
            for match in hits
            if not any(match.section_id.startswith(other + ".") for other in ids)
        ]
        if not hits:
            known = builtin_taxonomy(CATALOGUE) if catalogue is None else catalogue
            if tag not in known:
                name = CATALOGUE if catalogue is None else "the catalogue"
                return SectionContext(
                    tag=tag,
                    status="unknown_tag",
                    message=f"{tag} is not a tag in {name}.",
                    sections=[],
                )
            return SectionContext(
                tag=tag,
                status="not_found",
                message=f"No section is tagged {tag} in this document.",
                sections=[],
            )
        hits.sort(key=lambda match: _reading_order(match.section_id))
        return SectionContext(tag=tag, status="found", sections=hits)


def parse(
    source: Document | str | os.PathLike[str],
    *,
    include: Include = "sections",
    flow: Flow = "doc_first",
    backend: str | Backend = DEFAULT_BACKEND,
    level: Level | None = None,
    **classify_kwargs: Any,
) -> ParseResult:
    """Read ``source`` (a :class:`~docrooter.document.Document`, a path or Markdown text) and
    return a :class:`ParseResult` at the ``include`` level: ``"type"`` (the document type, from
    the doc-type call alone), ``"sections"`` (also every tagged heading with its tag) or ``"text"``
    (also the full text of each tagged section). ``flow`` defaults to ``"doc_first"``; the
    other arguments go to :func:`~docrooter.flow.classify`.

    Raises ``ValueError`` for an unknown ``include``, or for ``include="type"`` with
    ``flow="sections_first"`` (that flow decides the type from the tags) or with
    ``max_sections``."""
    if include not in INCLUDES:
        raise ValueError(f"unknown include {include!r}; use {', '.join(INCLUDES)}")
    if include == "type":
        if flow != "doc_first":
            raise ValueError(
                'include="type" needs flow="doc_first": the other flow decides the type '
                "from the section tags"
            )
        if "max_sections" in classify_kwargs:
            raise ValueError('max_sections does not apply with include="type"')
        classify_kwargs["max_sections"] = 0
    document = source if isinstance(source, Document) else load(source)
    classification = classify(document, flow=flow, backend=backend, level=level, **classify_kwargs)
    return ParseResult(
        source=classification.source,
        include=include,
        backend=classification.backend,
        doc_type=classification.doc_type,
        doc_type_confidence=classification.doc_type_confidence,
        tokens=_stage_tokens(classification),
        sections=None if include == "type" else tagged_sections(document, classification),
        section_text=section_text(document, classification) if include == "text" else None,
    )


def _stage_tokens(classification: Classification) -> dict[str, int]:
    usages = classification.stage_tokens
    return {
        stage: usages[stage].prompt + usages[stage].completion if stage in usages else 0
        for stage in STAGES
    }


def tagged_sections(document: Document, classification: Classification) -> list[TaggedSection]:
    """Every tagged heading in reading order, ``none`` included. ``ValueError`` when the
    classification tags a section the document does not have."""
    labels = _labels(document, classification)
    return [
        TaggedSection(
            section_id=section.id,
            heading=section.title,
            level=section.level,
            path=[a.title for a in reversed(list(_ancestors(document, section)))],
            tag=tag.label,
            confidence=tag.confidence,
            page_start=section.page_start,
        )
        for section in document.walk()
        if (tag := labels.get(section.id)) is not None and tag.label is not None
    ]


def section_text(
    document: Document, classification: Classification
) -> dict[str, list[SectionMatch]]:
    """The text of each tagged section, keyed by leaf tag in reading order of the tag's first
    section, without ``none``. A section inside another section with the same tag is dropped
    (its text is already in the outer one); text can repeat across tags. ``ValueError`` when the
    classification tags a section the document does not have."""
    labels = _labels(document, classification)
    order = list(document.walk())
    position = {section.id: index for index, section in enumerate(order)}
    tagged = [
        (section, tag)
        for section in order
        if (tag := labels.get(section.id)) is not None and tag.label not in (None, NO_TAG)
    ]
    ids_by_tag: dict[str, set[str]] = {}
    for section, tag in tagged:
        ids_by_tag.setdefault(tag.label or "", set()).add(section.id)
    result: dict[str, list[SectionMatch]] = {}
    for section, tag in tagged:
        label = tag.label or ""
        if any(a.id in ids_by_tag[label] for a in _ancestors(document, section)):
            continue
        result.setdefault(label, []).append(_match(document, section, tag, order, position))
    return result


def _labels(document: Document, classification: Classification) -> dict[str, SectionTag]:
    missing = sorted({t.id for t in classification.sections} - document.sections.keys())
    if missing:
        raise ValueError(
            f"the classification tags sections the document does not have: {', '.join(missing)}"
        )
    return {tag.id: tag for tag in classification.sections}


def _reading_order(section_id: str) -> tuple[tuple[int, int, str], ...]:
    """A sort key for hierarchical ids: ``s4.3`` sorts after ``s4`` and before ``s4.10``."""
    parts = section_id.removeprefix("s").split(".")
    return tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in parts)


def _match(
    document: Document,
    section: Section,
    tag: SectionTag,
    order: list[Section],
    position: dict[str, int],
) -> SectionMatch:
    subtree = list(_subtree(document, section))
    body = [section.text] + [
        part
        for child in subtree[1:]
        for part in (f"{'#' * min(child.level, MAX_HEADING_HASHES)} {child.title}", child.text)
    ]
    text = "\n\n".join(part.strip() for part in body if part.strip()).strip()
    page_start = section.page_start
    page_end = None
    if page_start is not None:
        following = position[section.id] + len(subtree)
        if following < len(order) and order[following].page_start is not None:
            page_end = order[following].page_start
        else:
            page_end = max(s.page_start for s in subtree if s.page_start is not None)
    return SectionMatch(
        section_id=section.id,
        tag=tag.label or "",
        heading=section.title,
        path=[a.title for a in reversed(list(_ancestors(document, section)))],
        level=section.level,
        text=text,
        page_start=page_start,
        page_end=page_end,
        confidence=tag.confidence,
        confidence_source=tag.confidence_source,
        tokens=estimate_tokens(section.title + "\n\n" + text),
    )


def _ancestors(document: Document, section: Section) -> Iterator[Section]:
    """The sections above ``section``, nearest first, without the preamble."""
    parent_id = section.parent_id
    while parent_id is not None:
        parent = document.sections[parent_id]
        if parent.level > 0:
            yield parent
        parent_id = parent.parent_id


def _subtree(document: Document, section: Section) -> Iterator[Section]:
    """``section`` and every descendant, in reading order. Iterative, no recursion."""
    stack = [section.id]
    while stack:
        current = document.sections[stack.pop()]
        yield current
        stack.extend(reversed(current.children))
