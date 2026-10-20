"""Render a whole document into one compact, budgeted text for document-type classification."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, field_validator

from docrooter.document import Document, Section
from docrooter.keyphrases import document_frequencies, drop_contained, scored_keyphrases
from docrooter.pdf.signals import normalize_title
from docrooter.views import WORDS_PER_TOKEN, RenderedView, estimate_tokens, render_budgeted

_FIELD_NAMES = ("title", "stats", "toc", "lead", "keyphrases", "text")

# Leading numbering already in a heading ("1.", "2.3", "IV."), replaced by the view's own. A
# bare number of four or more digits is a year ("2024 Annual Report"), not numbering.
_LEADING_NUMBER = re.compile(r"^\s*(?:\d{1,3}(?:\.\d+)*[.)]?|\d+(?:\.\d+)*[.)]|[IVXLC]+[.)])\s+")

# Normalised titles of front-matter and boilerplate sections that ``Lead`` skips.
_SKIPPED_TITLES = frozenset(
    {
        "cover",
        "cover page",
        "front cover",
        "notice",
        "important notice",
        "legal notice",
        "table of contents",
        "contents",
        "copyright",
        "copyright notice",
        "disclaimer",
        "important disclaimer",
        "forward-looking statements",
        "forward looking statements",
    }
)
_FORWARD_LOOKING = re.compile(r"forward[- ]looking statements$")


class DocumentView(BaseModel):
    """Which document-level fields to render, in order, and the token budget.

    Fields: ``title`` (document title, else the source file's stem), ``stats`` (page and
    section counts), ``toc`` (numbered headings down to ``toc_depth``, at most
    ``toc_entries``), ``lead`` (the first ``lead_words`` words of body text, skipping cover,
    notice, contents and similar boilerplate sections), ``keyphrases`` (section keyphrases
    pooled by summed score) and ``text`` (the whole body text, each word once: the ``lead``
    first, then the rest in reading order, boilerplate included; not a default field).
    Budgeting works as in ``SectionView``: the first field that does not fit is cut to the
    longest word prefix that does, and the view is never empty. The body text is gathered only
    up to the words the budget could ever hold, so a long document is never counted whole.
    """

    fields: tuple[str, ...] = ("title", "stats", "toc", "lead", "keyphrases")
    max_tokens: int = 512
    toc_depth: int = 2
    toc_entries: int = 40
    lead_words: int = 120
    keyphrases: int = 12

    @field_validator("fields")
    @classmethod
    def _known_fields(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unknown = [name for name in value if name not in _FIELD_NAMES]
        if unknown:
            raise ValueError(f"unknown view fields {unknown}; valid fields: {list(_FIELD_NAMES)}")
        return value

    def render(
        self, doc: Document, count_tokens: Callable[[str], int] | None = None
    ) -> RenderedView:
        """The document as ``Label: content`` lines within ``max_tokens``."""
        count = count_tokens or estimate_tokens
        title = document_title(doc)
        entries = (self._field(doc, name, title) for name in self.fields)
        return render_budgeted(entries, title, self.max_tokens, count)

    def _field(self, doc: Document, name: str, title: str) -> tuple[str, str] | None:
        if name == "title":
            return "Title:", title
        if name == "stats":
            return "Stats:", _stats(doc)
        if name == "toc":
            contents = _contents(doc, self.toc_depth, self.toc_entries)
            return ("Contents:", contents) if contents else None
        if name == "lead":
            lead = _lead(doc, self.lead_words)
            return ("Lead:", lead) if lead else None
        if name == "text":
            # One word past the cap, so render_budgeted sees (and flags) a cut.
            text = _full_text(doc, self.lead_words, WORDS_PER_TOKEN * self.max_tokens + 1)
            return ("Text:", text) if text else None
        phrases = _pooled_keyphrases(doc, self.keyphrases)
        return ("Key phrases:", ", ".join(phrases)) if phrases else None


def document_title(doc: Document) -> str:
    """``doc.title``, else the source file's stem (else the source itself)."""
    if doc.title and doc.title.strip():
        return doc.title.strip()
    return Path(doc.source).stem or doc.source


def _stats(doc: Document) -> str:
    n_sections = sum(1 for s in doc.sections.values() if s.level > 0)
    parts = [_plural(n_sections, "section")]
    n_pages = doc.meta.get("n_pages")
    if isinstance(n_pages, int) and not isinstance(n_pages, bool):
        parts.insert(0, _plural(n_pages, "page"))
    return ", ".join(parts)


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _contents(doc: Document, depth: int, limit: int) -> str:
    """``"1 Intro; 1.1 Scope; 2 Results"``: headings numbered by their place in the tree."""
    counters: list[int] = []
    shown: list[str] = []
    for entry in doc.toc():
        del counters[entry.depth + 1 :]
        counters.extend([0] * (entry.depth + 1 - len(counters)))
        counters[entry.depth] += 1
        if entry.depth >= depth:
            continue
        number = ".".join(str(c) for c in counters)
        title = _LEADING_NUMBER.sub("", " ".join(entry.title.split()), count=1) or entry.title
        shown.append(f"{number} {title}".strip())
        if len(shown) >= limit:
            break
    return "; ".join(shown)


def _is_boilerplate(section: Section) -> bool:
    title = normalize_title(section.title)
    return title in _SKIPPED_TITLES or bool(_FORWARD_LOOKING.search(title))


def _lead_words(doc: Document, n_words: int) -> tuple[list[str], dict[str, int]]:
    """The first ``n_words`` words of non-boilerplate body text, and how many words each
    section gave."""
    words: list[str] = []
    used: dict[str, int] = {}
    for section in doc.walk():
        if len(words) >= n_words:
            break
        if section.text.strip() and not _is_boilerplate(section):
            taken = section.text.split()[: n_words - len(words)]
            words.extend(taken)
            used[section.id] = len(taken)
    return words, used


def _lead(doc: Document, n_words: int) -> str:
    return " ".join(_lead_words(doc, n_words)[0])


def _full_text(doc: Document, lead_words: int, max_words: int) -> str:
    """The first ``max_words`` words of the body text: the lead (``lead_words`` words), then the
    remaining words of every section in reading order, so each word appears once."""
    words, used = _lead_words(doc, min(lead_words, max_words))
    for section in doc.walk():
        if len(words) >= max_words:
            break
        skip = used.get(section.id, 0)
        words.extend(section.text.split()[skip : skip + max_words - len(words)])
    return " ".join(words)


def _pooled_keyphrases(doc: Document, k: int) -> list[str]:
    """Each section's top-``k`` keyphrases (corpus: all section texts), pooled by summed score."""
    if k <= 0:
        return []
    corpus = [s.text for s in doc.walk()]
    df, n_docs = document_frequencies(corpus), len(corpus)
    totals: dict[str, float] = {}
    for text in corpus:
        for phrase, score in scored_keyphrases(text, df, n_docs, k):
            totals[phrase] = totals.get(phrase, 0.0) + score
    # sorted() is stable: equal totals keep first-seen (reading) order.
    ranked = sorted(totals.items(), key=lambda item: -item[1])
    return [phrase for phrase, _ in drop_contained(ranked, k)]
