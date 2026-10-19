"""Zero-shot shortlists for the teacher (spec §7.3, §4.0 ruling 5).

A cheap embedding model narrows each element's choices before the teacher LLM sees it:

- **topics:** the element (as ``heading path + text``) is scored against every leaf of its
  document family's topic taxonomy; the top ``k`` are kept, plus ``<family>.other`` when it is
  not among them (:func:`shortlist`);
- **functions:** each element of the sample is scored against each of the 12 function
  definitions on its own (:func:`function_cosines`). An element's prescreen score for a tag is
  the percentile rank of its cosine among every sampled element's cosine to that tag
  (:func:`percentile_scores`), so the tags never compete: two tags that both apply both score
  high, and ruling 6's "< 0.2 = implicit negative" means "among the 20% of the sample least
  like this tag". The top ``k`` tags by this score are the questions the teacher is asked
  (:func:`prescreen_functions`).

The backend is anything with ``score(texts, candidates, *, instruction)`` returning cosine
similarities: :class:`~docrooter.backends.embedding.EmbeddingBackend` (UltraBERT-Embed by
default), whose label vectors average each node's ``label: description`` with its examples.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any

import numpy as np

from docrooter.backends.base import Candidate, candidate_text
from docrooter.corpus import CorpusElement, ManifestRecord
from docrooter.taxonomies import ELEMENT_FAMILIES, builtin_taxonomy, family_for_doc_type
from docrooter.taxonomy import Taxonomy, TaxonomyNode
from docrooter.teach.prompts import table_text

__all__ = [
    "FUNCTION_INSTRUCTION",
    "MAX_QUERY_CHARS",
    "PRESCREEN_K",
    "SHORTLIST_K",
    "TOPIC_INSTRUCTION",
    "doc_family",
    "element_query",
    "element_text",
    "family_leaves",
    "family_options",
    "function_cosines",
    "function_scores",
    "function_tags",
    "percentile_scores",
    "prescreen_functions",
    "shortlist",
    "shortlist_scores",
    "top_functions",
    "topic_taxonomy",
]

SHORTLIST_K = 5  # topic leaves kept per element (spec §7.3), plus ``other``
PRESCREEN_K = 3  # function questions per element (§4.0 ruling 5)
# The embedder's input is cut to this many characters (about 500 tokens): the shortlist only
# needs the gist, and UltraBERT's 1024-token window would cost four times the attention.
MAX_QUERY_CHARS = 2000
TOPIC_INSTRUCTION = "Which section of a {family} does this document element belong to?"
FUNCTION_INSTRUCTION = "What does this document element do?"


@functools.cache
def _topics() -> Taxonomy:
    return builtin_taxonomy("element_topics")


@functools.cache
def _functions() -> Taxonomy:
    return builtin_taxonomy("element_functions")


def topic_taxonomy() -> Taxonomy:
    """The built-in ``element_topics`` taxonomy, parsed once (do not modify it)."""
    return _topics()


def doc_family(record: ManifestRecord) -> str:
    """The element topic family of a corpus document: its ``family_hint`` when that names a
    family, else the family of its ``doc_type_hint`` (``generic`` when there is none)."""
    if record.family_hint in ELEMENT_FAMILIES:
        return str(record.family_hint)
    return family_for_doc_type(record.doc_type_hint or "")


def family_leaves(family: str, taxonomy: Taxonomy | None = None) -> list[TaxonomyNode]:
    """The topic leaves of ``family`` in taxonomy order (``other`` last in the built-ins)."""
    taxonomy = taxonomy if taxonomy is not None else _topics()
    if family not in taxonomy:
        raise ValueError(f"unknown element family {family!r}; use one of {ELEMENT_FAMILIES}")
    return taxonomy.children(family)


def family_options(family: str, taxonomy: Taxonomy | None = None) -> list[tuple[str, str]]:
    """``(leaf id, "label: definition")`` for every leaf of ``family``: the teacher's options,
    lettered A.. in this order in the family's cached prefix."""
    return [(leaf.id, candidate_text(leaf)) for leaf in family_leaves(family, taxonomy)]


def element_text(element: CorpusElement, caption: str | None = None) -> str:
    """What the teacher reads: the element text, or for a table its description (``caption``)
    then its Markdown (:func:`~docrooter.teach.prompts.table_text`)."""
    if element.kind == "table" and element.table_markdown:
        return table_text(caption, element.table_markdown)
    return element.text


def element_query(element: CorpusElement, caption: str | None = None) -> str:
    """What the embedder reads (spec §7.3): the heading path, a newline, then the element text,
    cut to :data:`MAX_QUERY_CHARS` characters."""
    path = " > ".join(element.heading_path)
    text = element_text(element, caption)
    query = f"{path}\n{text}" if path else text
    return query[:MAX_QUERY_CHARS]


def _candidates(nodes: Sequence[TaxonomyNode]) -> list[Candidate]:
    return [Candidate(n.id, candidate_text(n), tuple(n.examples)) for n in nodes]


def _queries(elements: Sequence[CorpusElement], captions: Sequence[str | None] | None) -> list[str]:
    if captions is None:
        captions = [None] * len(elements)
    return [element_query(e, c) for e, c in zip(elements, captions, strict=True)]


def shortlist_scores(
    elements: Sequence[CorpusElement],
    family: str,
    backend: Any,
    k: int = SHORTLIST_K,
    *,
    taxonomy: Taxonomy | None = None,
    captions: Sequence[str | None] | None = None,
) -> list[list[tuple[str, float]]]:
    """Per element, the top ``k`` leaves of ``family`` with their cosine similarities, best
    first (ties in taxonomy order), then ``<family>.other`` with its own score if it was not
    among them."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    leaves = family_leaves(family, taxonomy)
    if not elements:
        return []
    ids = [leaf.id for leaf in leaves]
    other = f"{family}.other"
    label = (taxonomy if taxonomy is not None else _topics()).node(family).label
    scores = backend.score(
        _queries(elements, captions),
        _candidates(leaves),
        instruction=TOPIC_INSTRUCTION.format(family=label.lower()),
    )
    values = np.asarray(scores.values, dtype=float)
    out = []
    for row in values:
        order = sorted(range(len(ids)), key=lambda j: (-row[j], j))
        top = [(ids[j], float(row[j])) for j in order[:k]]
        if other in ids and all(tag != other for tag, _ in top):
            top.append((other, float(row[ids.index(other)])))
        out.append(top)
    return out


def shortlist(
    elements: Sequence[CorpusElement],
    family: str,
    backend: Any,
    k: int = SHORTLIST_K,
    *,
    taxonomy: Taxonomy | None = None,
    captions: Sequence[str | None] | None = None,
) -> list[list[str]]:
    """Per element, the ids of its top ``k`` topic leaves within ``family`` (best first), always
    including ``<family>.other`` (spec §7.3)."""
    return [
        [tag for tag, _ in row]
        for row in shortlist_scores(
            elements, family, backend, k, taxonomy=taxonomy, captions=captions
        )
    ]


def function_cosines(
    elements: Sequence[CorpusElement],
    backend: Any,
    *,
    taxonomy: Taxonomy | None = None,
    captions: Sequence[str | None] | None = None,
) -> np.ndarray:
    """``(elements x tags)`` cosine similarities of each element to each function tag (its
    ``label: description`` and examples), tags in taxonomy order. Each cell depends only on
    its element and its tag."""
    nodes = (taxonomy if taxonomy is not None else _functions()).leaves()
    if not elements:
        return np.zeros((0, len(nodes)))
    scores = backend.score(
        _queries(elements, captions), _candidates(nodes), instruction=FUNCTION_INSTRUCTION
    )
    return np.asarray(scores.values, dtype=float).reshape(len(elements), len(nodes))


def percentile_scores(cosines: np.ndarray) -> np.ndarray:
    """Per column (tag), each row's percentile rank among the column's values, in [0, 1]: 0 for
    the lowest, 1 for the highest, ties sharing their average rank. A single row scores 0.5."""
    values = np.asarray(cosines, dtype=float)
    n = values.shape[0]
    if n == 0:
        return values.copy()
    if n == 1:
        return np.full_like(values, 0.5)
    out = np.empty_like(values)
    for j in range(values.shape[1]):
        column = values[:, j]
        ordered = np.sort(column)
        below = np.searchsorted(ordered, column, side="left")
        through = np.searchsorted(ordered, column, side="right")
        out[:, j] = (below + through - 1) / 2 / (n - 1)
    return out


def function_scores(
    elements: Sequence[CorpusElement],
    backend: Any,
    *,
    taxonomy: Taxonomy | None = None,
    captions: Sequence[str | None] | None = None,
) -> list[dict[str, float]]:
    """Per element, its prescreen score for every function tag (taxonomy order): the
    percentile of its cosine to the tag among ``elements`` (the sample)."""
    ids = function_tags(taxonomy)
    ranks = percentile_scores(
        function_cosines(elements, backend, taxonomy=taxonomy, captions=captions)
    )
    return [{tag: float(p) for tag, p in zip(ids, row, strict=True)} for row in ranks]


def function_tags(taxonomy: Taxonomy | None = None) -> list[str]:
    """The function tag ids in taxonomy order."""
    return [n.id for n in (taxonomy if taxonomy is not None else _functions()).leaves()]


def top_functions(scores: dict[str, float], k: int = PRESCREEN_K) -> list[tuple[str, float]]:
    """The ``k`` highest-scoring tags of one element, best first (ties in taxonomy order)."""
    order = list(scores)
    ranked = sorted(order, key=lambda tag: (-scores[tag], order.index(tag)))
    return [(tag, scores[tag]) for tag in ranked[:k]]


def prescreen_functions(
    elements: Sequence[CorpusElement],
    backend: Any,
    k: int = PRESCREEN_K,
    *,
    taxonomy: Taxonomy | None = None,
    captions: Sequence[str | None] | None = None,
) -> list[list[tuple[str, float]]]:
    """Per element, its ``k`` best function tags by prescreen score (percentile among
    ``elements``), best first: the questions the teacher is asked (§4.0 ruling 5)."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    return [
        top_functions(row, k)
        for row in function_scores(elements, backend, taxonomy=taxonomy, captions=captions)
    ]
