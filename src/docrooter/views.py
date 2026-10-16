"""Render sections into compact, budgeted text views for classification."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Literal, NamedTuple

from pydantic import BaseModel, field_validator

from docrooter.document import Document, Section
from docrooter.keyphrases import document_frequencies, keyphrases_from_frequencies

_FIELD_NAMES = ("path", "title", "keyphrases", "lead", "text")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def estimate_tokens(text: str) -> int:
    """Cheap token estimate: 1.3 tokens per whitespace-separated word."""
    return math.ceil(len(text.split()) * 1.3)


class RenderedView(NamedTuple):
    text: str
    truncated: bool
    n_tokens: int


class SectionView(BaseModel):
    fields: tuple[str, ...] = ("path", "title", "keyphrases", "lead", "text")
    max_tokens: int = 512
    lead_sentences: int = 3
    keyphrases: int = 8
    empty_section: Literal["children_titles", "title"] = "children_titles"

    @field_validator("fields")
    @classmethod
    def _known_fields(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unknown = [name for name in value if name not in _FIELD_NAMES]
        if unknown:
            raise ValueError(f"unknown view fields {unknown}; valid fields: {list(_FIELD_NAMES)}")
        return value

    def render_all(
        self,
        doc: Document,
        section_ids: Sequence[str],
        count_tokens: Callable[[str], int] | None = None,
    ) -> dict[str, RenderedView]:
        """Render each requested section. Unknown ids raise ``KeyError``."""
        count = count_tokens or estimate_tokens
        sections = [doc.sections[sid] for sid in section_ids]
        df = n_docs = None
        if "keyphrases" in self.fields:
            corpus = [s.text for s in doc.sections.values()]
            df, n_docs = document_frequencies(corpus), len(corpus)
        return {section.id: self._render(doc, section, df, n_docs, count) for section in sections}

    def _render(self, doc, section: Section, df, n_docs, count) -> RenderedView:
        entries = (self._field(doc, section, name, df, n_docs) for name in self.fields)
        return render_budgeted(entries, section.title, self.max_tokens, count)

    def _field(
        self, doc: Document, section: Section, name: str, df, n_docs
    ) -> tuple[str, str] | None:
        text = section.text.strip()
        if name == "path":
            return "Path:", " > ".join(_path_titles(doc, section))
        if name == "title":
            return "Title:", section.title
        if name == "keyphrases":
            phrases = keyphrases_from_frequencies(text, df, n_docs, self.keyphrases)
            return ("Key phrases:", ", ".join(phrases)) if phrases else None
        if name == "lead":
            sentences = _SENTENCE_SPLIT.split(" ".join(text.split()))
            lead = " ".join(sentences[: self.lead_sentences]).strip()
            return ("Lead:", lead) if lead else None
        # text
        if text:
            return "Text:", text
        if self.empty_section == "children_titles" and section.children:
            titles = "; ".join(doc.sections[child].title for child in section.children)
            return "Subsections:", titles
        return "Text:", section.title


def render_budgeted(
    entries: Iterable[tuple[str, str] | None],
    fallback_title: str,
    max_tokens: int,
    count: Callable[[str], int],
) -> RenderedView:
    """Join ``(label, content)`` lines in order while they fit ``max_tokens``.

    ``None`` entries are skipped. The first line that does not fit is cut to its longest
    fitting word prefix (dropped if not even one word fits) and rendering stops there. The
    result is never empty: with no line at all it is ``Title: <fallback_title>``, cut to fit.
    """
    lines: list[str] = []
    truncated = False
    for entry in entries:
        if entry is None:
            continue
        label, content = entry
        line = f"{label} {content}" if label else content
        if count("\n".join([*lines, line])) <= max_tokens:
            lines.append(line)
            continue
        truncated = True
        fitted = _fit(lines, label, content, max_tokens, count)
        if fitted is not None:
            lines.append(fitted)
        break
    if not lines:
        fallback, cut = _title_fallback(fallback_title, max_tokens, count)
        lines.append(fallback)
        truncated = truncated or cut
    text = "\n".join(lines)
    return RenderedView(text=text, truncated=truncated, n_tokens=count(text))


def _fit(
    lines: list[str], label: str, content: str, max_tokens: int, count: Callable[[str], int]
) -> str | None:
    """Longest word-prefix line of ``content`` that fits the budget, else None."""
    words = content.split()

    def line_of(n: int) -> str:
        body = " ".join(words[:n])
        return f"{label} {body}" if label else body

    def fits(n: int) -> bool:
        return count("\n".join([*lines, line_of(n)])) <= max_tokens

    low, high = 0, len(words) - 1  # the full content is already known not to fit
    while low < high:
        mid = (low + high + 1) // 2
        if fits(mid):
            low = mid
        else:
            high = mid - 1
    return line_of(low) if low else None


def _title_fallback(title: str, max_tokens: int, count: Callable[[str], int]) -> tuple[str, bool]:
    """A never-empty view: ``Title: <title>``, cut to the budget, keeping at least one word.

    Returns the line and whether it was cut (or the label dropped) to fit.
    """
    words = title.split() or [title]
    labeled = f"Title: {' '.join(words)}"
    if count(labeled) <= max_tokens:
        return labeled, False
    for n in range(len(words), 0, -1):
        candidate = f"Title: {' '.join(words[:n])}"
        if count(candidate) <= max_tokens:
            return candidate, True
    for n in range(len(words), 1, -1):
        candidate = " ".join(words[:n])
        if count(candidate) <= max_tokens:
            return candidate, True
    return words[0], True


def _path_titles(doc: Document, section: Section) -> list[str]:
    titles = [section.title]
    parent_id = section.parent_id
    while parent_id is not None:
        parent = doc.sections[parent_id]
        if parent.level > 0:
            titles.append(parent.title)
        parent_id = parent.parent_id
    titles.reverse()
    return titles
