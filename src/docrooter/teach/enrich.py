"""Taxonomy enrichment (spec §8, TELEClass §3.2 adapted): class-indicative terms mined from the
teacher-labelled train elements, written to the taxonomy's optional ``keywords`` field.

For a class ``c`` (a topic leaf, or a family) with siblings ``S`` (the other children of its
parent) and a term ``t``:

- ``pop(t, c) = log(1 + df(t, D_c))``, where ``D_c`` holds the train elements whose teacher
  choice is ``c`` or one of its descendants, and ``df`` counts the elements mentioning ``t``;
- ``dist(t, c) = exp(BM25(t, D_c)) / (1 + Σ_{c' ∈ S} exp(BM25(t, D_c')))``. BM25 treats each
  class of the sibling group (``c`` and ``S``) as one document, the concatenation of its
  elements, with the group as the collection: ``k1 = 1.5``, ``b = 0.75`` and the non-negative
  idf ``log(1 + (N - n + 0.5) / (n + 0.5))``;
- ``aff(t, c) = pop^0.5 · dist^0.5 · max(0, cos(t, label of c))``, the cosine between the
  embeddings of the term and of the class label, both encoded bare (no prompt).

Candidates are n-grams of 1 to 3 tokens with document frequency ≥ 5 among the elements mined,
whose first and last tokens are neither stop words nor bare numbers nor single characters; an
n-gram never spans punctuation. Counting streams the corpus in four passes, and the n-gram
tables it holds are bounded by the (nearly) frequent n-grams: a fixed-size count-min sketch
screens the longer n-grams before they are counted exactly (:func:`count_terms`).
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol

import numpy as np

from docrooter.corpus.store import Corpus, LabelKey, LabelRow
from docrooter.keyphrases import STOPWORDS
from docrooter.taxonomy import Taxonomy
from docrooter.teach.base import refuse_jev

__all__ = [
    "BM25_B",
    "BM25_K1",
    "MAX_NGRAM",
    "MIN_DF",
    "TOPIC_FACET",
    "Choice",
    "Encoder",
    "TermCounts",
    "bm25",
    "count_terms",
    "element_ngrams",
    "enrich_taxonomy",
    "mine_terms",
    "normalise_term",
    "teacher_choices",
    "term_affinity",
]

logger = logging.getLogger("docrooter.teach")

TOPIC_FACET = "topic"
MAX_NGRAM = 3
MIN_DF = 5
BM25_K1 = 1.5
BM25_B = 0.75
SKETCH_WIDTH = 1 << 22  # buckets per row of the n-gram count-min sketch (2 rows: 32 MiB)
TERM_PROMPT = ""  # terms and class labels are encoded alike, bare (TELEClass §3.2)

_SEGMENT = re.compile(r"[.,;:!?()\[\]{}<>\"“”«»|•·…\n\r\t]+")
_TOKEN = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")


class Encoder(Protocol):
    """What enrichment needs of an embedder: L2-normalised rows, one per text
    (:meth:`docrooter.backends.embedding.EmbeddingBackend.encode`)."""

    def encode(self, texts: list[str], *, prompt: str) -> np.ndarray: ...


class Choice(NamedTuple):
    """The teacher's choice for one element's topic: the leaf with the highest probability."""

    label: str
    prob: float


# --- teacher labels ---------------------------------------------------------------------------


def _refuse_jev_row(row: LabelRow) -> None:
    """No Jev output may enter training (TypeSafe MCA §2.3(b)): refuse a row whose provenance
    names Jev as its teacher, engine or model."""
    for field in ("teacher", "engine", "model", "name"):
        value = row.provenance.get(field)
        if isinstance(value, str):
            refuse_jev(value)
            refuse_jev(":" + value)  # a bare model reference


def teacher_choices(
    labels: Mapping[LabelKey, LabelRow] | Iterable[LabelRow], taxonomy: Taxonomy
) -> dict[LabelKey, Choice]:
    """The teacher's topic choice per element key: the argmax of each ``topic`` row's ``probs``
    (a distribution over the shortlisted leaves) and its probability. Rows of other facets are
    ignored, and so are rows with no probabilities; for a key given twice, the last row wins.

    A row naming a node that is not a leaf of ``taxonomy`` is a ``ValueError`` (labels must
    match the tree they were scored on); a row whose provenance names Jev raises
    :class:`~docrooter.errors.TeacherNotAllowedError`.
    """
    rows = labels.values() if isinstance(labels, Mapping) else labels
    out: dict[LabelKey, Choice] = {}
    for row in rows:
        if row.facet != TOPIC_FACET or not row.probs:
            continue
        _refuse_jev_row(row)
        label, prob = max(row.probs.items(), key=lambda item: item[1])
        node = taxonomy.nodes.get(label)
        if node is None or node.children:
            raise ValueError(
                f"topic label {label!r} of element {row.doc_id}/{row.element_id} is not a leaf "
                "of the taxonomy"
            )
        out[row.key] = Choice(label, float(prob))
    return out


# --- tokens and n-grams -----------------------------------------------------------------------


def _segments(text: str) -> list[list[str]]:
    """Lower-cased tokens, split into runs that no n-gram crosses (at punctuation)."""
    lowered = text.lower().replace("’", "'")
    out = []
    for part in _SEGMENT.split(lowered):
        tokens = _TOKEN.findall(part)
        if tokens:
            out.append(tokens)
    return out


def _edge_ok(token: str) -> bool:
    return len(token) > 1 and not token.isdigit() and token not in STOPWORDS


def _has_clean_edges(term: str) -> bool:
    words = term.split(" ")
    return _edge_ok(words[0]) and _edge_ok(words[-1])


def _grams(segments: list[list[str]], n: int) -> Iterator[tuple[str, ...]]:
    for tokens in segments:
        for i in range(len(tokens) - n + 1):
            yield tuple(tokens[i : i + n])


def element_ngrams(text: str) -> set[str]:
    """Every n-gram of ``text`` (1 to 3 tokens, space-joined, none crossing punctuation),
    whatever its edges: what a keyword is matched against."""
    segments = _segments(text)
    return {" ".join(g) for n in range(1, MAX_NGRAM + 1) for g in _grams(segments, n)}


def normalise_term(term: str) -> str:
    """A term in the tokenizer's form (lower case, single spaces), as mined terms are."""
    return " ".join(token for segment in _segments(term) for token in segment)


# --- counting ---------------------------------------------------------------------------------


@dataclass
class TermCounts:
    """Per-class counts of the candidate terms. Rows follow ``classes``, columns ``terms``.

    ``tf[c, t]``: occurrences of ``t`` in the elements of class ``c``; ``df[c, t]``: elements of
    ``c`` mentioning ``t``; ``length[c]``: tokens in the elements of ``c``.
    """

    terms: list[str]
    classes: list[str]
    tf: np.ndarray
    df: np.ndarray
    length: np.ndarray


class _Sketch:
    """A count-min sketch of document frequencies, two rows of ``width`` buckets: an estimate is
    never below the true count, so it can only let through n-grams that are not frequent, which
    the exact count then drops. Hashes are Python's (fixed within a process)."""

    def __init__(self, width: int) -> None:
        if width < 2 or width & (width - 1):
            raise ValueError(f"sketch width must be a power of two, got {width}")
        self.mask = np.uint64(width - 1)
        self.rows = np.zeros((2, width), dtype=np.int32)

    def _index(self, grams: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
        h = np.fromiter((hash(g) for g in grams), dtype=np.int64).view(np.uint64)
        return (h & self.mask).astype(np.intp), ((h >> np.uint64(32)) & self.mask).astype(np.intp)

    def add(self, grams: Iterable[str]) -> None:
        first, second = self._index(grams)
        np.add.at(self.rows[0], first, 1)
        np.add.at(self.rows[1], second, 1)

    def estimate(self, grams: Iterable[str]) -> np.ndarray:
        first, second = self._index(grams)
        return np.minimum(self.rows[0][first], self.rows[1][second])


def _longer_grams(text: str, vocabulary: set[str]) -> set[str]:
    """The distinct n-grams of 2 to :data:`MAX_NGRAM` tokens of ``text`` whose tokens are all in
    ``vocabulary`` (an n-gram is no more frequent than any of its tokens)."""
    segments = _segments(text)
    return {
        " ".join(g)
        for n in range(2, MAX_NGRAM + 1)
        for g in _grams(segments, n)
        if all(token in vocabulary for token in g)
    }


def count_terms(
    source: Callable[[], Iterable[tuple[str, Sequence[str]]]],
    classes: Sequence[str],
    *,
    min_df: int = MIN_DF,
    sketch_width: int = SKETCH_WIDTH,
) -> TermCounts:
    """Count the candidate terms of the texts ``source()`` yields, with the classes each text
    belongs to (``source`` is called once per pass and must yield the same texts each time).

    A candidate is an n-gram of 1 to :data:`MAX_NGRAM` tokens in at least ``min_df`` texts whose
    edge tokens are not stop words, bare numbers or single characters. Classes not in
    ``classes`` are ignored.

    Four streaming passes: token document frequencies; a count-min sketch of the longer
    n-grams made of frequent tokens; their exact document frequencies, for those the sketch
    puts at ``min_df`` or more; the per-class counts of the candidates. Only the last two hold
    n-gram tables, both bounded by the (nearly) frequent n-grams; the sketch is fixed in size
    (two rows of ``sketch_width`` ``int32``).
    """
    tokens: Counter[str] = Counter()
    for text, _ in source():
        tokens.update({t for segment in _segments(text) for t in segment})
    vocabulary = {t for t, count in tokens.items() if count >= min_df}
    del tokens

    sketch = _Sketch(sketch_width)
    for text, _ in source():
        sketch.add(_longer_grams(text, vocabulary))
    longer: Counter[str] = Counter()
    for text, _ in source():
        grams = list(_longer_grams(text, vocabulary))
        if grams:
            estimates = sketch.estimate(grams)
            longer.update(g for g, e in zip(grams, estimates, strict=True) if e >= min_df)
    del sketch

    frequent = list(vocabulary) + [g for g, count in longer.items() if count >= min_df]
    del longer, vocabulary
    candidates = sorted(g for g in frequent if _has_clean_edges(g))
    del frequent
    index = {term: i for i, term in enumerate(candidates)}
    row = {c: i for i, c in enumerate(classes)}
    tf = np.zeros((len(classes), len(candidates)), dtype=np.int32)
    df = np.zeros_like(tf)
    length = np.zeros(len(classes), dtype=np.int64)
    for text, member_of in source():
        rows = [row[c] for c in dict.fromkeys(member_of) if c in row]
        if not rows:
            continue
        segments = _segments(text)
        hits: Counter[int] = Counter()
        for n in range(1, MAX_NGRAM + 1):
            for g in _grams(segments, n):
                i = index.get(" ".join(g))
                if i is not None:
                    hits[i] += 1
        ids = np.fromiter(hits.keys(), dtype=np.intp, count=len(hits))
        counts = np.fromiter(hits.values(), dtype=np.int32, count=len(hits))
        n_tokens = sum(len(s) for s in segments)
        for r in rows:
            tf[r, ids] += counts
            df[r, ids] += 1
            length[r] += n_tokens
    return TermCounts(candidates, list(classes), tf, df, length)


# --- scoring ----------------------------------------------------------------------------------


def bm25(
    tf: np.ndarray, length: np.ndarray, *, k1: float = BM25_K1, b: float = BM25_B
) -> np.ndarray:
    """BM25 of each term (column) as a one-term query against each document (row) of a
    collection: ``idf · tf · (k1 + 1) / (tf + k1 · (1 - b + b · len / avglen))``, with
    ``idf = log(1 + (N - n + 0.5) / (n + 0.5))``, ``N`` the documents and ``n`` those holding
    the term."""
    tf = np.asarray(tf, dtype=np.float64)
    length = np.asarray(length, dtype=np.float64)
    n_docs = tf.shape[0]
    holding = (tf > 0).sum(axis=0)
    idf = np.log1p((n_docs - holding + 0.5) / (holding + 0.5))
    avg = length.mean() if n_docs else 0.0
    norm = 1.0 - b + b * (length / avg if avg > 0 else np.ones_like(length))
    return idf * tf * (k1 + 1.0) / (tf + k1 * norm[:, None])


def term_affinity(counts: TermCounts, cos: np.ndarray, taxonomy: Taxonomy) -> np.ndarray:
    """``aff[c, t] = pop^0.5 · dist^0.5 · max(0, cos[c, t])`` for every class row of ``counts``
    (module docstring); ``cos`` has the same shape as ``counts.tf``. The siblings of a class are
    the other children of its parent that are classes of ``counts``."""
    groups: dict[str | None, list[int]] = {}
    for i, c in enumerate(counts.classes):
        groups.setdefault(taxonomy.node(c).parent_id, []).append(i)
    cos = np.asarray(cos)
    aff = np.zeros(counts.tf.shape, dtype=np.float64)
    for members in groups.values():  # one sibling group at a time: small temporaries
        e = np.exp(bm25(counts.tf[members], counts.length[members]))
        dist = e / (1.0 + e.sum(axis=0) - e)  # the siblings' sum leaves the class itself out
        pop = np.log1p(counts.df[members], dtype=np.float64)
        aff[members] = np.sqrt(pop) * np.sqrt(dist) * np.maximum(cos[members], 0.0)
    return aff


def _cosines(encoder: Encoder, terms: list[str], labels: list[str], batch_size: int) -> np.ndarray:
    """``cos[c, t]`` between label ``c`` and term ``t``; term embeddings are dropped batch by
    batch."""
    classes = _unit(np.asarray(encoder.encode(labels, prompt=TERM_PROMPT), dtype=np.float32))
    out = np.zeros((len(labels), len(terms)), dtype=np.float32)
    for start in range(0, len(terms), batch_size):
        chunk = terms[start : start + batch_size]
        vectors = _unit(np.asarray(encoder.encode(chunk, prompt=TERM_PROMPT), dtype=np.float32))
        out[:, start : start + len(chunk)] = classes @ vectors.T
    return out


def _unit(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norm > 0, norm, 1.0)


def mine_terms(
    corpus: Corpus,
    labels: Mapping[LabelKey, LabelRow] | Iterable[LabelRow],
    taxonomy: Taxonomy,
    embedder: Encoder,
    *,
    k: int = 10,
    min_df: int = MIN_DF,
    split: str = "train",
    batch_size: int = 512,
) -> dict[str, list[str]]:
    """The top ``k`` terms by affinity for every node of the topic ``taxonomy`` (leaves and
    families), mined from the ``split`` elements that have a teacher ``topic`` label (spec §8;
    module docstring). Every node is a key; a node whose elements yield no term with a positive
    affinity gets ``[]``. Ties keep alphabetical order. calib-crc documents are never read
    (:func:`~docrooter.corpus.calib_half`).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    choices = teacher_choices(labels, taxonomy)
    classes = list(taxonomy.nodes)

    def source() -> Iterator[tuple[str, list[str]]]:
        for element in corpus.iter_fit_elements(split=split):
            choice = choices.get(
                LabelKey(element.doc_id, element.element_id, element.text_sha1, TOPIC_FACET)
            )
            if choice is not None:
                yield element.text, [n.id for n in taxonomy.path(choice.label)]

    counts = count_terms(source, classes, min_df=min_df)
    logger.info("term mining: %d candidate terms over %d classes", len(counts.terms), len(classes))
    out: dict[str, list[str]] = {c: [] for c in classes}
    if not counts.terms:
        return out
    labels_text = [taxonomy.nodes[c].label for c in classes]
    cos = _cosines(embedder, counts.terms, labels_text, batch_size)
    aff = term_affinity(counts, cos, taxonomy)
    del cos
    for r, c in enumerate(classes):
        order = np.argsort(-aff[r], kind="stable")[:k]  # terms are sorted: ties stay alphabetical
        out[c] = [counts.terms[i] for i in order if aff[r, i] > 0]
    return out


def enrich_taxonomy(taxonomy: Taxonomy, terms: Mapping[str, Sequence[str]]) -> Taxonomy:
    """A copy of ``taxonomy`` whose nodes named in ``terms`` carry those terms as ``keywords``
    (replacing any they had; an empty list clears them). An id not in the taxonomy is a
    ``KeyError``."""
    unknown = sorted(set(terms) - set(taxonomy.nodes))
    if unknown:
        raise KeyError(f"terms for nodes not in the taxonomy: {unknown}")
    data: dict[str, Any] = taxonomy.to_dict()
    for node in data["nodes"]:
        if node["id"] in terms:
            node["keywords"] = [str(t) for t in terms[node["id"]]]
    return Taxonomy.from_dict(data)
