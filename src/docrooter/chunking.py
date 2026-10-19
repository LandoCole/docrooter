"""Section-aware chunks of a document's elements, for indexing (spec §9).

A chunk never crosses a section boundary and never splits an element. Elements are packed in
reading order up to ``max_tokens`` (:func:`docrooter.views.estimate_tokens`, the heading path
included); a table is always a chunk of its own, and a single element that does not fit
becomes a chunk of its own, flagged ``"oversize"``.

The text of a chunk is its heading path joined with `` > ``, a blank line, and the texts of its
elements joined with blank lines. A heading or title element that opens its section is the last
item of the heading path, so it is not repeated in the body: it is listed in ``element_ids`` of
the section's first chunk. A section with no body (a heading followed at once by its first
subsection, or a leaf heading such as "Item 6. [Reserved]") gets one chunk of its own whose text
is just the heading path, flagged ``"heading_only"``, so that every heading is in exactly one
chunk. An element with no text at all is in no chunk.

Packing is linear: word counts are taken once per element and summed as the chunk grows. If the
heading path alone is longer than ``max_tokens``, every chunk of that section is flagged
``"oversize"``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from docrooter.document.models import Document
from docrooter.elements.models import Element
from docrooter.errors import DocrooterError
from docrooter.summarize import heading_titles, table_description
from docrooter.views import estimate_tokens, tokens_for_words

if TYPE_CHECKING:
    from docrooter.rooter.analysis import Analysis

__all__ = ["Chunk", "chunk"]

FURNITURE_KINDS = frozenset({"page_header", "page_footer", "page_number"})
SKIPPED_KINDS = FURNITURE_KINDS | {"contents"}
OVERSIZE = "oversize"
HEADING_ONLY = "heading_only"
PATH_SEPARATOR = " > "


class Chunk(BaseModel):
    """A run of elements of one section, ready to index.

    ``element_ids`` are the elements whose text is in ``text`` (plus the section's opening
    heading in its first chunk); with ``overlap_elements`` an element can be in two chunks.
    ``tokens`` is ``estimate_tokens(text)``. ``flags`` holds ``"oversize"`` for a chunk of one
    element that is longer than ``max_tokens`` on its own, and ``"heading_only"`` for the
    path-only chunk of a section with no body (its ``element_ids`` are its heading).
    """

    id: str  # "c0", "c1", … in reading order
    section_id: str | None
    element_ids: list[str]
    text: str
    heading_path: list[str]
    tokens: int
    flags: list[str] = Field(default_factory=list)


def chunk(
    source: Analysis | Document,
    *,
    max_tokens: int = 512,
    overlap_elements: int = 0,
    include_furniture: bool = False,
) -> list[Chunk]:
    """Split the elements of ``source`` (an :class:`~docrooter.Analysis` or a
    :class:`~docrooter.Document` loaded in element mode) into chunks.

    ``overlap_elements`` repeats up to that many elements from the end of a chunk at the start of
    the next one in the same section (never a table, and only while the result still fits
    ``max_tokens``). Page furniture and ``contents`` elements are skipped unless
    ``include_furniture``.
    """
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
    if overlap_elements < 0:
        raise ValueError(f"overlap_elements must be >= 0, got {overlap_elements}")
    doc = document_of(source)
    skipped = frozenset() if include_furniture else SKIPPED_KINDS
    parts = _fold_captions(element_parts(doc, lambda el: el.kind not in skipped))
    chunks: list[Chunk] = []
    for section_id, group in _by_section(parts):
        path = heading_titles(doc, section_id)
        packer = _Packer(path, max_tokens, overlap_elements)
        for part in group:
            packer.add(part)
        packer.finish()
        for element_ids, text, tokens, flags in packer.out:
            chunks.append(
                Chunk(
                    id=f"c{len(chunks)}",
                    section_id=section_id,
                    element_ids=element_ids,
                    text=text,
                    heading_path=path,
                    tokens=tokens,
                    flags=flags,
                )
            )
    return chunks


# --- shared with docrooter.embed ---------------------------------------------------------------


def document_of(source: Analysis | Document) -> Document:
    """The element-mode document of ``source``; a document without elements that has text is an
    error (it was loaded without ``elements=True``)."""
    doc = source if isinstance(source, Document) else source.document
    if not doc.elements and any(section.text.strip() for section in doc.sections.values()):
        raise DocrooterError(
            "the document has no elements: load it with elements=True (or use "
            "process() with a profile other than 'fast')"
        )
    return doc


@dataclass(frozen=True)
class Part:
    """An element and its body text; ``opener`` for the heading that starts its section (its
    text is the last item of the heading path, so the body is empty)."""

    element: Element
    body: str
    opener: bool = False
    table: bool = False
    words: int = 0  # whitespace-separated words of ``body``
    folded: tuple[Element, ...] = ()  # a table's caption, whose text is in ``body`` already


def compose(path: Iterable[str], body: str) -> str:
    """``"A > B\\n\\nbody"``; just the path or just the body when the other is empty."""
    prefix = PATH_SEPARATOR.join(path)
    return f"{prefix}\n\n{body}" if prefix and body else prefix or body


def element_parts(doc: Document, keep: Callable[[Element], bool]) -> list[Part]:
    """The parts of the elements ``keep`` accepts, in reading order. An element with no text
    gives none."""
    captions = _captions(doc)
    parts: list[Part] = []
    for el in doc.elements:
        if not keep(el):
            continue
        if _opens_section(el):
            if el.text.strip():
                parts.append(Part(el, "", opener=True))
            continue
        body = _body(el, captions)
        if body.strip():
            parts.append(Part(el, body, table=el.kind == "table", words=len(body.split())))
    return parts


def _fold_captions(parts: list[Part]) -> list[Part]:
    """Fold the caption of each table into the table's part. The table's description line
    already holds the caption text, so a chunk of its own would repeat it. The caption element
    stays in exactly one chunk: the table's. A caption in another section keeps its own."""
    by_id = {part.element.id: part for part in parts}
    caption_of: dict[str, Element] = {}  # table id -> its caption
    used: set[str] = set()

    def link(table: Part, caption: Part | None) -> None:
        if (
            caption is not None
            and caption.element.kind == "caption"
            and caption.element.section_id == table.element.section_id
            and table.element.id not in caption_of
            and caption.element.id not in used
        ):
            caption_of[table.element.id] = caption.element
            used.add(caption.element.id)

    for part in parts:  # a caption that names its table...
        if part.element.kind == "caption" and part.element.caption_for in by_id:
            table = by_id[part.element.caption_for]
            if table.table:
                link(table, part)
    for part in parts:  # ... or a table that names its caption
        if part.table and part.element.caption_id in by_id:
            link(part, by_id[part.element.caption_id])
    return [
        replace(part, folded=(caption_of[part.element.id],))
        if part.element.id in caption_of
        else part
        for part in parts
        if part.element.id not in used
    ]


def _opens_section(el: Element) -> bool:
    return el.kind in ("heading", "title") and el.level is not None


def _captions(doc: Document) -> dict[str, str]:
    """Caption text by the id of the table or figure it belongs to."""
    by_id = {el.id: el for el in doc.elements}
    out: dict[str, str] = {}
    for el in doc.elements:
        if el.kind == "caption" and el.caption_for:
            out.setdefault(el.caption_for, el.text)
        if el.caption_id and el.caption_id in by_id:
            out[el.id] = by_id[el.caption_id].text
    return out


def _body(el: Element, captions: dict[str, str]) -> str:
    if el.kind == "table":
        caption = " ".join(captions.get(el.id, "").split())
        description = table_description(el, caption=caption or None)
        incomplete = "cell_text_incomplete" in el.flags  # the cells lost words; the text has all
        markdown = "" if el.table is None or incomplete else el.table.to_markdown()
        return f"{description}\n{markdown or el.text.strip()}".rstrip()
    if el.kind in ("list_item", "footnote") and el.marker:
        return f"{el.marker} {el.text}".strip()
    return el.text.strip()


def _by_section(parts: list[Part]) -> list[tuple[str | None, list[Part]]]:
    """Runs of consecutive parts of one section (a section's elements are contiguous)."""
    groups: list[tuple[str | None, list[Part]]] = []
    for part in parts:
        section_id = part.element.section_id
        if groups and groups[-1][0] == section_id:
            groups[-1][1].append(part)
        else:
            groups.append((section_id, [part]))
    return groups


class _Packer:
    """Packs the parts of one section into chunks (``out``: ids, text, tokens, flags)."""

    def __init__(self, path: list[str], max_tokens: int, overlap: int) -> None:
        self.path = path
        self.path_words = len(PATH_SEPARATOR.join(path).split())
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.out: list[tuple[list[str], str, int, list[str]]] = []
        self.openers: list[Part] = []  # openers waiting for the first chunk with a body
        self.current: list[Part] = []  # overlap parts first, then the new ones
        self.words = 0  # the words of ``current``
        self.n_overlap = 0
        self.tail: list[Part] = []  # what the next chunk may repeat

    def _fits(self, words: int) -> bool:
        return tokens_for_words(self.path_words + words) <= self.max_tokens

    def add(self, part: Part) -> None:
        if part.opener:
            self.openers.append(part)
        elif part.table:
            self._flush()
            self._emit([*self.openers, part])
            self.openers, self.tail = [], []
        elif len(self.current) > self.n_overlap and self._fits(self.words + part.words):
            self.current.append(part)
            self.words += part.words
        else:
            self._flush()
            repeated = self.tail[-self.overlap :] if self.overlap else []
            words = sum(p.words for p in repeated)
            while repeated and not self._fits(words + part.words):
                words -= repeated[0].words  # drop the oldest until the new element fits
                repeated = repeated[1:]
            self.current, self.n_overlap = [*repeated, part], len(repeated)
            self.words = words + part.words

    def finish(self) -> None:
        self._flush()
        if self.openers:  # a section with no body: its heading is a chunk of the path alone
            self._emit(self.openers, heading_only=True)
            self.openers = []

    def _flush(self) -> None:
        if len(self.current) > self.n_overlap:  # it has a new element, not only overlap
            self._emit([*self.openers, *self.current])
            self.tail = self.current[-self.overlap :] if self.overlap else []
            self.openers = []
        self.current, self.n_overlap, self.words = [], 0, 0

    def _emit(self, parts: list[Part], *, heading_only: bool = False) -> None:
        text = compose(self.path, "\n\n".join(p.body for p in parts if p.body))
        tokens = estimate_tokens(text)
        flags = [HEADING_ONLY] if heading_only else []
        if tokens > self.max_tokens:
            flags.append(OVERSIZE)
        ids = [
            element.id
            for p in parts
            for element in sorted([p.element, *p.folded], key=lambda e: e.order)
        ]
        self.out.append((ids, text, tokens, flags))
