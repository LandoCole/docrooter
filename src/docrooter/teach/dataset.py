"""The student's training set (spec §8, §9; §4.0 ruling 7): the TELEClass steps after the
teacher, in order, and the :class:`~docrooter.train.Dataset` they give.

:func:`build_training_set` runs, on the teacher-labelled elements of the train split:

1. term mining (:func:`~docrooter.teach.enrich.mine_terms`), then the enriched taxonomy;
2. class centroids, from the elements that contain one of their class's keywords;
3. each element's core classes and gap, among the leaves of its teacher choice's family (the
   topic is chosen within the document's family, spec §4.1);
4. the keep rule and the trim (:func:`~docrooter.teach.confidence.confident_set`).

:func:`to_dataset` turns the result into examples whose text is the student input
:func:`input_text`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

import numpy as np

from docrooter.corpus.store import Corpus, CorpusElement, LabelKey, LabelRow
from docrooter.taxonomy import Taxonomy
from docrooter.teach.confidence import (
    KEEP,
    MIN_PER_CLASS,
    PROB_OVERRIDE_QUANTILE,
    TrainingSet,
    class_centroids,
    confident_set,
    core_classes,
)
from docrooter.teach.enrich import (
    TOPIC_FACET,
    Encoder,
    element_ngrams,
    enrich_taxonomy,
    mine_terms,
    normalise_term,
    teacher_choices,
)
from docrooter.teach.prompts import table_text
from docrooter.train import Dataset, Example

__all__ = ["ELEMENT_PROMPT", "build_training_set", "input_text", "to_dataset"]

logger = logging.getLogger("docrooter.teach")

ELEMENT_PROMPT = ""  # elements are documents: encoded bare, as docrooter.embed does
HEADING_SEP = " > "


def input_text(element: CorpusElement) -> str:
    """The student's input for an element: ``"{heading path}\\n\\n[{kind}] {text}"``, the
    heading path joined by ``" > "`` (empty in the preamble). A table's text is its cells as
    Markdown when the corpus has them (spec Review Focus 3), else its text. Truncation to the
    student's token limit is the trainer's."""
    text = element.text
    if element.kind == "table" and element.table_markdown:
        text = table_text(None, element.table_markdown)
    return f"{HEADING_SEP.join(element.heading_path)}\n\n[{element.kind}] {text}"


def to_dataset(training_set: TrainingSet, corpus: Corpus) -> Dataset:
    """The training set as a :class:`~docrooter.train.Dataset`, in the training set's order:
    text :func:`input_text`, path the example's (``[family, leaf]``, or ``[family]`` for an
    untrained leaf), ``doc_id`` the document and ``section_id`` the element id. An example whose
    element (with that text) is not in the corpus is a ``ValueError``."""
    wanted = {(e.doc_id, e.element_id, e.text_sha1) for e in training_set.examples}
    texts: dict[tuple[str, str, str], str] = {}
    for element in corpus.iter_elements():
        key = (element.doc_id, element.element_id, element.text_sha1)
        if key in wanted:
            texts[key] = input_text(element)
    missing = [k for k in wanted if k not in texts]
    if missing:
        shown = ", ".join(f"{d}/{e}" for d, e, _ in sorted(missing)[:5])
        raise ValueError(
            f"{len(missing)} training example(s) not in the corpus with the labelled text: {shown}"
        )
    return Dataset(
        [
            Example(
                text=texts[(e.doc_id, e.element_id, e.text_sha1)],
                path=list(e.path),
                doc_id=e.doc_id,
                section_id=e.element_id,
            )
            for e in training_set.examples
        ]
    )


def _encode(encoder: Encoder, texts: list[str], batch_size: int) -> np.ndarray:
    out: np.ndarray | None = None
    for start in range(0, len(texts), batch_size):
        chunk = np.asarray(
            encoder.encode(texts[start : start + batch_size], prompt=ELEMENT_PROMPT),
            dtype=np.float32,
        )
        if out is None:
            out = np.zeros((len(texts), chunk.shape[1]), dtype=np.float32)
        out[start : start + len(chunk)] = chunk
    return out if out is not None else np.zeros((0, 0), dtype=np.float32)


def _family_leaves(taxonomy: Taxonomy, leaf: str) -> list[str]:
    parent = taxonomy.nodes[leaf].parent_id or Taxonomy.ROOT_ID
    return [n.id for n in taxonomy.children(parent) if not n.children]


def build_training_set(
    corpus: Corpus,
    labels: Mapping[LabelKey, LabelRow] | Iterable[LabelRow],
    taxonomy: Taxonomy,
    embedder: Encoder,
    *,
    k: int = 10,
    keep: float = KEEP,
    min_per_class: int = MIN_PER_CLASS,
    prob_override_quantile: float = PROB_OVERRIDE_QUANTILE,
    split: str = "train",
    batch_size: int = 256,
) -> tuple[Taxonomy, TrainingSet]:
    """``(enriched taxonomy, confident set)`` from the teacher's ``topic`` rows on the ``split``
    elements, in the order of spec §4.0 ruling 7 (module docstring). Holds one embedding per
    labelled element (``float32``)."""
    if not isinstance(labels, Mapping):
        labels = list(labels)  # read more than once
    terms = mine_terms(corpus, labels, taxonomy, embedder, k=k, split=split)
    enriched = enrich_taxonomy(taxonomy, terms)
    keywords = {c: {normalise_term(w) for w in node.keywords} for c, node in enriched.nodes.items()}

    choices = teacher_choices(labels, taxonomy)
    keys: list[LabelKey] = []
    texts: list[str] = []
    chosen: list[str] = []
    has_keyword: list[bool] = []
    for element in corpus.iter_elements(split=split):
        key = LabelKey(element.doc_id, element.element_id, element.text_sha1, TOPIC_FACET)
        choice = choices.get(key)
        if choice is None:
            continue
        keys.append(key)
        texts.append(input_text(element))
        chosen.append(choice.label)
        has_keyword.append(bool(keywords[choice.label] & element_ngrams(element.text)))
    logger.info(
        "similarity gap: %d labelled %s elements, %d contain a class keyword",
        len(keys),
        split,
        sum(has_keyword),
    )
    vectors = _encode(embedder, texts, batch_size)
    del texts
    centroids = class_centroids(vectors, chosen, has_keyword)
    families = {label: _family_leaves(taxonomy, label) for label in set(chosen)}
    candidates = [families[label] for label in chosen]
    cores = dict(zip(keys, core_classes(vectors, candidates, centroids), strict=True))
    del vectors
    training_set = confident_set(
        labels,
        cores,
        keep=keep,
        min_per_class=min_per_class,
        prob_override_quantile=prob_override_quantile,
        taxonomy=taxonomy,
    )
    return enriched, training_set
