"""Section context: hand an application a section's full text by its tag.

:class:`TaggedDocument` holds a :class:`~docrooter.document.Document` with the
:class:`~docrooter.flow.Classification` :func:`~docrooter.flow.classify` made of it. Asking for a
tag returns a :class:`SectionContext`: the heading and the full text (the section's own body,
then every subsection under it) of each section carrying that tag. Nothing is read or classified
again, and no model is called.

- A section matches a tag when its label equals the tag or is under it in the catalogue (a group
  query: ``"financial"`` matches every ``financial.*`` leaf). A match inside another match for
  the same tag is dropped, since its text is already in the outer one; same-tag siblings stay
  separate.
- A tag the catalogue has but the document does not is ``not_found``; a tag the catalogue lacks
  is ``unknown_tag``. Neither raises, since an application asks for tags speculatively.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, PrivateAttr

from docrooter.document import Document, Section
from docrooter.document.load import load
from docrooter.errors import UnknownTaxonomyError
from docrooter.flow import NO_TAG, Classification, ConfidenceSource, SectionTag, classify
from docrooter.taxonomies import builtin_taxonomy
from docrooter.taxonomy import Taxonomy
from docrooter.views import estimate_tokens

MAX_HEADING_HASHES = 6
"""Markdown has six heading levels; a deeper section is written with six."""

FORMAT = "docrooter.tagged_document"
"""The ``format`` of a saved :class:`TaggedDocument`."""

FORMAT_VERSION = 1
"""The ``version`` of a saved :class:`TaggedDocument`."""


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


class TaggedDocument(BaseModel):
    """A document with its classification, asked for section text by tag.

    ``taxonomy`` (a :class:`~docrooter.taxonomy.Taxonomy` or a built-in name) says which
    catalogue the tags come from; by default it is ``classification.section_taxonomy`` as a
    built-in. When that cannot be resolved (a custom catalogue that was not passed), an unknown
    tag cannot be told from a missing one and both are ``not_found``. It is not a field and is
    not serialised.

    Raises ``ValueError`` when the classification tags a section the document does not have
    (it belongs to another reading).
    """

    model_config = ConfigDict(extra="forbid")

    document: Document
    classification: Classification
    _taxonomy: Taxonomy | None = PrivateAttr(default=None)
    _taxonomy_name: str | None = PrivateAttr(default=None)

    def __init__(self, /, *, taxonomy: Taxonomy | str | None = None, **data: Any) -> None:
        # ``taxonomy`` is not a field: it reaches ``model_post_init`` as the validation context.
        self.__pydantic_validator__.validate_python(
            data, self_instance=self, context={"taxonomy": taxonomy}
        )

    def model_post_init(self, context: Any, /) -> None:
        missing = sorted(
            {t.id for t in self.classification.sections} - self.document.sections.keys()
        )
        if missing:
            raise ValueError(
                f"the classification tags sections the document does not have: {', '.join(missing)}"
            )
        given = context.get("taxonomy") if isinstance(context, dict) else None
        name = given if isinstance(given, str) else self.classification.section_taxonomy
        self._taxonomy_name = name
        if isinstance(given, Taxonomy):
            self._taxonomy = given
        elif name is not None:
            try:
                self._taxonomy = builtin_taxonomy(name)
            except UnknownTaxonomyError:
                self._taxonomy = None

    @classmethod
    def from_source(
        cls,
        source: Document | str | os.PathLike[str],
        *,
        taxonomy: Taxonomy | str | None = None,
        **classify_kwargs: Any,
    ) -> TaggedDocument:
        """Load ``source`` (a :class:`~docrooter.document.Document` is used as it is) and
        classify it once; ``classify_kwargs`` (``backend``, ``flow``, ``level``…) go to
        :func:`~docrooter.flow.classify`."""
        document = source if isinstance(source, Document) else load(source)
        classification = classify(document, **classify_kwargs)
        return cls(document=document, classification=classification, taxonomy=taxonomy)

    def to_json(self) -> str:
        """The document and classification as JSON, in a versioned envelope."""
        envelope = {
            "format": FORMAT,
            "version": FORMAT_VERSION,
            **self.model_dump(mode="json"),
        }
        return json.dumps(envelope, ensure_ascii=False)

    @classmethod
    def from_json(
        cls, data: str | bytes, *, taxonomy: Taxonomy | str | None = None
    ) -> TaggedDocument:
        """Rebuild from :meth:`to_json`. ``ValueError`` when the JSON is not a tagged document
        of this format and version. ``taxonomy`` is as in the constructor."""
        envelope = json.loads(data)
        version = envelope.get("version") if isinstance(envelope, dict) else None
        if (
            not isinstance(envelope, dict)
            or envelope.get("format") != FORMAT
            or type(version) is not int
            or version != FORMAT_VERSION
        ):
            raise ValueError(f"not a tagged document: expected format {FORMAT} version 1")
        return cls(
            document=Document.model_validate(envelope.get("document")),
            classification=Classification.model_validate(envelope.get("classification")),
            taxonomy=taxonomy,
        )

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write :meth:`to_json` to ``path`` as UTF-8."""
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(
        cls, path: str | os.PathLike[str], *, taxonomy: Taxonomy | str | None = None
    ) -> TaggedDocument:
        """Read a file written by :meth:`save`."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"), taxonomy=taxonomy)

    @property
    def doc_type(self) -> str | None:
        return self.classification.doc_type

    def available_tags(self) -> list[str]:
        """The distinct leaf tags the document has, excluding ``none``, in reading order of
        first appearance."""
        labels = self._labels()
        seen: dict[str, None] = {}
        for section in self.document.walk():
            tag = labels.get(section.id)
            if tag is not None and tag.label not in (None, NO_TAG):
                seen.setdefault(tag.label)
        return list(seen)

    def get_section(self, tag: str, *, min_confidence: float | None = None) -> SectionContext:
        """The context of ``tag`` (surrounding whitespace is ignored, case is exact).
        ``min_confidence`` drops matches below it; a match with no confidence is kept."""
        return self._context(tag, min_confidence, list(self.document.walk()), self._labels())

    def get_sections(
        self, tags: Sequence[str], *, min_confidence: float | None = None
    ) -> list[SectionContext]:
        """One context per tag, in the order asked; a repeated tag is answered each time."""
        if isinstance(tags, str):
            raise TypeError("tags must be a sequence of tags, not a single string")
        order, labels = list(self.document.walk()), self._labels()
        return [self._context(tag, min_confidence, order, labels) for tag in tags]

    def get_all_sections(self, *, min_confidence: float | None = None) -> list[SectionContext]:
        """One context per available tag, in reading order of each tag's first section. Each
        context is complete on its own, so text repeats across contexts when a tagged section
        sits inside another tagged section."""
        order, labels = list(self.document.walk()), self._labels()
        return [self._context(tag, min_confidence, order, labels) for tag in self.available_tags()]

    def _labels(self) -> dict[str, SectionTag]:
        return {tag.id: tag for tag in self.classification.sections}

    def _covers(self, label: str, tag: str) -> bool:
        """Whether a section labelled ``label`` is under ``tag`` (or is ``tag``)."""
        if label == tag:
            return True
        if self._taxonomy is not None and label in self._taxonomy:
            return any(node.id == tag for node in self._taxonomy.path(label))
        return label.startswith(tag + ".")

    def _context(
        self,
        asked: str,
        min_confidence: float | None,
        order: list[Section],
        labels: dict[str, SectionTag],
    ) -> SectionContext:
        tag = asked.strip()
        if self._taxonomy is not None and tag not in self._taxonomy:
            catalogue = self._taxonomy_name or "the taxonomy"
            return SectionContext(
                tag=tag,
                status="unknown_tag",
                message=f"{tag} is not a tag in {catalogue}.",
                sections=[],
            )
        hits = [
            (section, labels[section.id])
            for section in order
            if section.id in labels
            and labels[section.id].label is not None
            and self._covers(labels[section.id].label or "", tag)
        ]
        if min_confidence is not None:
            kept = [
                (s, t) for s, t in hits if t.confidence is None or t.confidence >= min_confidence
            ]
            dropped = len(kept) < len(hits)
            hits = kept
        else:
            dropped = False
        ids = {section.id for section, _ in hits}
        hits = [
            (section, t)
            for section, t in hits
            if not any(ancestor.id in ids for ancestor in _ancestors(self.document, section))
        ]
        if not hits:
            message = f"No section is tagged {tag} in this document."
            if dropped:
                message = f"No section tagged {tag} has a confidence of at least {min_confidence}."
            return SectionContext(tag=tag, status="not_found", message=message, sections=[])
        position = {section.id: index for index, section in enumerate(order)}
        matches = [self._match(section, t, order, position) for section, t in hits]
        return SectionContext(tag=tag, status="found", sections=matches)

    def _match(
        self, section: Section, tag: SectionTag, order: list[Section], position: dict[str, int]
    ) -> SectionMatch:
        subtree = list(_subtree(self.document, section))
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
            path=[a.title for a in reversed(list(_ancestors(self.document, section)))],
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
