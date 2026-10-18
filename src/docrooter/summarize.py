"""Key sentences and key phrases for sections and long elements (no ML, no extra dependencies).

``centroid`` scores each sentence by its TF-IDF cosine to the whole text, a bonus for similarity
to the heading path, a position prior and a length penalty, then picks the best sentences that
do not repeat each other. ``lead`` just takes the first sentences.
"""

from __future__ import annotations

import functools
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Literal

from pydantic import BaseModel, Field

from docrooter.document.models import Document
from docrooter.elements.models import Element
from docrooter.keyphrases import (
    STOPWORDS,
    document_frequencies,
    keyphrases_from_frequencies,
)
from docrooter.keyphrases import _tokenize as tokenize

HEADING_WEIGHT = 0.2
POSITION_WEIGHT = 0.1
LENGTH_PENALTY = 0.3
MIN_WORDS, MAX_WORDS = 6, 80
REDUNDANCY_COSINE = 0.7
SUMMARISED_KINDS = frozenset({"paragraph", "list_item"})

# --- sentence splitting ---------------------------------------------------------------------

# Abbreviations after which a full stop never ends a sentence, whatever follows. Corporate
# suffixes ("Inc.", "Co.", "Ltd."), "etc." and "et al." are deliberately not here: they are
# protected by the lower-case rule below when the sentence goes on, and split before a capital.
_ABBREVIATIONS = frozenset(
    """mr mrs ms dr prof sr jr st vs no nos fig figs eq eqs sec sect ch vol pp cf viz approx
    dept gen gov sen rep mt e.g i.e u.s u.k""".split()
)
_BLOCK_BREAK = re.compile(r"\n\s*\n|\n(?=\s*[-*•▪◦]\s)")  # blank line, bullet
_BOUNDARY = re.compile(r"([.!?…]+[\"'”’)\]]*)\s+(?=\S)")
_OPENERS, _CLOSERS = "\"'“‘([", "\"'”’)]"
_INITIAL = re.compile(r"[A-Za-z]\.")
_LIST_MARKER = re.compile(r"\(?\d{1,3}[.)]")


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, protecting abbreviations, initials, list markers, ellipses.

    A full stop followed by a lower-case letter never splits. Decimals and section numbers
    ("3.14", "1.2.3") have no space after the dot, so they are never candidates. Blank lines and
    bullet lines start a new sentence; other line breaks are ordinary spaces.
    """
    sentences: list[str] = []
    for block in _BLOCK_BREAK.split(text):
        flat = " ".join(block.split())
        start = 0
        for match in _BOUNDARY.finditer(flat):
            if _ends_sentence(flat, match):
                sentences.append(flat[start : match.end(1)])
                start = match.end()
        if flat[start:]:
            sentences.append(flat[start:])
    return sentences


def _ends_sentence(flat: str, match: re.Match[str]) -> bool:
    if flat[match.end()].islower():
        return False
    token = flat[: match.end(1)].rsplit(" ", 1)[-1].lstrip(_OPENERS).rstrip(_CLOSERS)
    if token.endswith(".") and not token.endswith(".."):
        return not (
            token.rstrip(".").lower() in _ABBREVIATIONS
            or _INITIAL.fullmatch(token)
            or _LIST_MARKER.fullmatch(token)
        )
    return True


# --- vectors ----------------------------------------------------------------------------------


class _Vec:
    """A sparse TF-IDF vector with its norm."""

    __slots__ = ("norm", "weights")

    def __init__(self, text: str, idf: Callable[[str], float]):
        tf = Counter(t for t in tokenize(text) if t not in STOPWORDS)
        self.weights = {t: c * idf(t) for t, c in tf.items()}
        self.norm = math.sqrt(sum(w * w for w in self.weights.values()))

    def cosine(self, other: _Vec) -> float:
        if not self.norm or not other.norm:
            return 0.0
        small, big = sorted((self.weights, other.weights), key=len)
        return sum(w * big.get(t, 0.0) for t, w in small.items()) / (self.norm * other.norm)


# --- summaries --------------------------------------------------------------------------------


class Summary(BaseModel):
    key_sentences: list[str]
    keyphrases: list[str] = Field(default_factory=list)
    method: Literal["lead", "centroid"]


def summarize(
    doc: Document,
    *,
    method: Literal["lead", "centroid"] = "centroid",
    k: int = 3,
    keyphrases: int = 8,
    element_min_sentences: int = 4,
) -> dict[str, Summary]:
    """Summaries keyed by id: every section with at least 2 sentences, then every ``paragraph``
    or ``list_item`` element with at least ``element_min_sentences`` (section and element ids do
    not collide: ``s1`` against ``e0``). Tables get :func:`table_description` instead.

    Document frequencies are taken over the document's sections, as :mod:`docrooter.views` does.
    """
    if method not in ("lead", "centroid"):
        raise ValueError(f"unknown summary method {method!r}; use 'lead' or 'centroid'")
    corpus = [section.text for section in doc.sections.values()]
    df, n_docs = document_frequencies(corpus), len(corpus)

    @functools.cache
    def idf(token: str) -> float:  # smoothed, so a word in every section still carries weight
        return math.log((1 + n_docs) / (1 + df.get((token,), 0))) + 1

    def one(text: str, section_id: str | None) -> Summary:
        sentences = split_sentences(text)
        if method == "lead":
            picked = sentences[:k]
        else:
            picked = _centroid_pick(sentences, _heading_path(doc, section_id), idf, k)
        phrases = keyphrases_from_frequencies(text, df, n_docs, keyphrases)
        return Summary(key_sentences=picked, keyphrases=phrases, method=method)

    out: dict[str, Summary] = {}
    for section in doc.sections.values():
        if len(split_sentences(section.text)) >= 2:
            out[section.id] = one(section.text, section.id)
    for el in doc.elements:
        if el.kind in SUMMARISED_KINDS and len(split_sentences(el.text)) >= element_min_sentences:
            out[el.id] = one(el.text, el.section_id)
    return out


def _heading_path(doc: Document, section_id: str | None) -> str:
    """Titles from the root down to the section, joined with " > " (iterative; no preamble)."""
    titles: list[str] = []
    seen: set[str] = set()
    while section_id is not None and section_id in doc.sections and section_id not in seen:
        seen.add(section_id)
        section = doc.sections[section_id]
        if section.level > 0:
            titles.append(section.title)
        section_id = section.parent_id
    return " > ".join(reversed(titles))


def _centroid_pick(
    sentences: Sequence[str], path: str, idf: Callable[[str], float], k: int
) -> list[str]:
    vecs = [_Vec(s, idf) for s in sentences]
    centroid = _Vec(" ".join(sentences), idf)
    heading = _Vec(path, idf)
    scores: list[float] = []
    for index, (sentence, vec) in enumerate(zip(sentences, vecs, strict=True)):
        score = vec.cosine(centroid) + HEADING_WEIGHT * vec.cosine(heading)
        score += POSITION_WEIGHT / (1 + index)
        if not MIN_WORDS <= len(sentence.split()) <= MAX_WORDS:
            score -= LENGTH_PENALTY
        scores.append(score)
    picked: list[int] = []
    for index in sorted(range(len(sentences)), key=lambda i: (-scores[i], i)):
        if len(picked) >= k:
            break
        if all(vecs[index].cosine(vecs[j]) <= REDUNDANCY_COSINE for j in picked):
            picked.append(index)
    return [sentences[i] for i in sorted(picked)]


# --- tables -----------------------------------------------------------------------------------


def table_description(el: Element, *, caption: str | None = None) -> str:
    """``"Table: <caption or first header row>; <n_rows>×<n_cols>"``.

    The caption is a separate element, so the caller passes its text. Without one, the label is
    the table's first row (its header row when it has one) with empty cells dropped.
    """
    rows = el.table.rows if el.table else []
    label = (caption or "").strip()
    if not label and rows:
        label = " | ".join(" ".join(cell.split()) for cell in rows[0] if cell.strip())
    n_cols = max((len(row) for row in rows), default=0)
    return f"Table: {label or 'untitled'}; {len(rows)}×{n_cols}"
