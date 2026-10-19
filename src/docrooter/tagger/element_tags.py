"""The rooter's ``element_tags`` task (spec §10): what its spec means, the taggers behind it,
and tagging the content elements of a document.

Spec forms (``RouterConfig.element_tags``):

- ``none``: no tags (the default in every profile);
- a bundle directory, as :meth:`ElementTagger.save` writes it: the trained student;
- ``embedding:<model>``: a zero-shot baseline (:class:`EmbeddingTagger`). The topic is the
  leaf of the document's family whose definition is most similar to the element (a softmax
  over the family's leaves at the backend's temperature); the functions are a softmax over the
  12 function definitions, and a tag is a label from :data:`FUNCTION_THRESHOLD` on, so at
  most one is;
- ``teacher:<engine spec>``: a teacher LLM (``mlx:``, ``hf:`` or ``openai:``,
  :func:`~docrooter.teach.get_teacher`) chooses the topic among all the leaves of the family
  and answers a yes/no question per function tag (13 forward passes per element: slow, for
  checking the student). Jev is refused, as everywhere (TypeSafe MCA §2.3(b)).

The family is the document's, from its type (:func:`~docrooter.taxonomies.family_for_doc_type`).
Only :data:`TAGGED_KINDS` are tagged; headings, figures and page furniture keep ``tags=None``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from docrooter.backends.base import Scores, to_probs
from docrooter.elements.models import TagResult
from docrooter.errors import DocrooterError

if TYPE_CHECKING:
    from docrooter.corpus.store import CorpusElement
    from docrooter.document.models import Document

__all__ = [
    "EMBEDDING_PREFIX",
    "FUNCTION_THRESHOLD",
    "TAGGED_KINDS",
    "TEACHER_PREFIX",
    "EmbeddingTagger",
    "ElementTags",
    "TeacherTagger",
    "check_spec",
    "content_elements",
    "load_bundle",
    "refuse_jev_spec",
    "resolve_tagger",
    "tag_document",
]

TAGGED_KINDS: tuple[str, ...] = (
    "paragraph",
    "list_item",
    "table",
    "caption",
    "footnote",
    "formula",
)
"""The element kinds that get tags (spec §10)."""
EMBEDDING_PREFIX = "embedding:"
TEACHER_PREFIX = "teacher:"
FUNCTION_THRESHOLD = 0.5  # as the student's (docrooter.tagger.bundle.FUNCTION_THRESHOLD)
BUNDLE_MANIFEST = "tagger.json"  # docrooter.tagger.bundle.TAGGER_MANIFEST
TEACHER_ENGINES = ("mlx", "hf", "openai")
_DIGITS = 6


@runtime_checkable
class ElementTags(Protocol):
    """What tags elements: :class:`~docrooter.tagger.ElementTagger` and the taggers here.
    ``tag`` returns ``{"topic": TagResult, "function": TagResult}`` (either may be missing)
    per element; ``family`` is the document's element family."""

    def tag(self, elements: Sequence[CorpusElement], family: str) -> list[dict[str, TagResult]]: ...


def refuse_jev_spec(spec: str) -> None:
    """Raise :class:`~docrooter.errors.TeacherNotAllowedError` for an ``element_tags`` spec
    that names Jev: ``teacher:jev:…``, a teacher naming a Jev model, or ``jev:…`` itself
    (TypeSafe MCA §2.3(b))."""
    from docrooter.teach.base import refuse_jev

    value = spec.strip()
    refuse_jev(value[len(TEACHER_PREFIX) :] if value.startswith(TEACHER_PREFIX) else value)


def check_spec(spec: str) -> None:
    """Check an ``element_tags`` spec without loading anything: Jev is refused, an
    ``embedding:`` spec must name a known backend kind, a ``teacher:`` spec a known engine,
    and anything else must be a bundle directory (holding ``tagger.json``)."""
    refuse_jev_spec(spec)
    value = spec.strip()
    if value.startswith(EMBEDDING_PREFIX):
        from docrooter.backends import registry

        registry.check_spec(value)
    elif value.startswith(TEACHER_PREFIX):
        engine, sep, target = value[len(TEACHER_PREFIX) :].partition(":")
        if engine.strip().lower() not in TEACHER_ENGINES or not sep or not target.strip():
            raise DocrooterError(
                f"bad element_tags teacher spec {spec!r}: use teacher:mlx:<repo>@<revision>, "
                "teacher:hf:<repo>@<revision> or teacher:openai:<model>"
            )
    elif not (Path(value) / BUNDLE_MANIFEST).is_file():
        raise DocrooterError(
            f"element_tags={spec!r}: no element-tagger bundle there (no {BUNDLE_MANIFEST}); "
            "use none, a bundle directory, embedding:<model> or teacher:<engine spec>"
        )


def load_bundle(path: str) -> ElementTags:
    """The :class:`~docrooter.tagger.ElementTagger` saved at ``path`` (its body loads on the
    first element)."""
    from docrooter.tagger.bundle import ElementTagger

    return ElementTagger.load(path)


def resolve_tagger(spec: str) -> ElementTags:
    """The tagger of an ``element_tags`` spec other than ``none`` (module docstring). The
    caller closes it."""
    check_spec(spec)
    value = spec.strip()
    if value.startswith(EMBEDDING_PREFIX):
        return EmbeddingTagger(value)
    if value.startswith(TEACHER_PREFIX):
        return TeacherTagger(value[len(TEACHER_PREFIX) :])
    return load_bundle(value)


def content_elements(doc: Document, doc_id: str = "document") -> list[CorpusElement]:
    """The document's :data:`TAGGED_KINDS` elements as the student reads them (heading path,
    kind, text; a table's cells as Markdown), in reading order."""
    from docrooter.corpus.build import corpus_elements

    return [e for e in corpus_elements(doc_id, doc) if e.kind in TAGGED_KINDS]


def tag_document(doc: Document, tagger: ElementTags, family: str) -> int:
    """Set ``tags`` on the document's :data:`TAGGED_KINDS` elements (in one call to
    ``tagger.tag``, which batches); returns how many were tagged."""
    elements = content_elements(doc)
    if not elements:
        return 0
    results = tagger.tag(elements, family)
    if len(results) != len(elements):
        raise DocrooterError(
            f"the element tagger returned {len(results)} results for {len(elements)} elements"
        )
    by_id = {el.id: el for el in doc.elements}
    for element, tags in zip(elements, results, strict=True):
        by_id[element.element_id].tags = dict(tags)
    return len(elements)


def _p(value: float) -> float:
    return round(min(1.0, max(0.0, float(value))), _DIGITS)


def _topic_result(probs: dict[str, float], source: str) -> TagResult:
    best = max(probs, key=lambda k: probs[k])  # first of equals: taxonomy order
    return TagResult(
        labels=[best],
        probs={k: _p(v) for k, v in probs.items()},
        confidence=_p(probs[best]),
        source=source,
    )


class EmbeddingTagger:
    """The zero-shot baseline of ``element_tags="embedding:<model>"`` (module docstring):
    cosine similarities of each element (heading path and text, as the teacher's shortlist
    reads it) to the leaves of its family and to the function definitions, turned into
    probabilities by a softmax at the backend's temperature. ``family_probs`` is not set."""

    def __init__(self, spec: str, *, backend: Any = None, batch_size: int = 256) -> None:
        from docrooter.backends import registry

        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.name = spec
        self.backend = backend if backend is not None else registry.resolve_backend(spec)
        self.batch_size = batch_size

    @property
    def source(self) -> str:
        return self.name

    def _probs(self, values: np.ndarray) -> np.ndarray:
        temperature = float(getattr(self.backend, "temperature", 1.0))
        return to_probs(Scores(values, "similarity"), temperature=temperature, multi_label=False)

    def tag(self, elements: Sequence[CorpusElement], family: str) -> list[dict[str, TagResult]]:
        from docrooter.teach.shortlist import (
            family_leaves,
            function_cosines,
            function_tags,
            shortlist_scores,
        )

        leaves = [leaf.id for leaf in family_leaves(family)]
        tags = function_tags()
        out: list[dict[str, TagResult]] = []
        for start in range(0, len(elements), self.batch_size):
            batch = list(elements[start : start + self.batch_size])
            scored = shortlist_scores(batch, family, self.backend, k=len(leaves))
            cosines = np.array([[dict(row)[leaf] for leaf in leaves] for row in scored])
            topics = self._probs(cosines.reshape(len(batch), len(leaves)))
            functions = self._probs(function_cosines(batch, self.backend))
            for topic_row, function_row in zip(topics, functions, strict=True):
                values = dict(zip(tags, function_row.tolist(), strict=True))
                out.append(
                    {
                        "topic": _topic_result(
                            dict(zip(leaves, topic_row.tolist(), strict=True)), self.source
                        ),
                        "function": TagResult(
                            labels=[t for t in tags if values[t] >= FUNCTION_THRESHOLD],
                            probs={t: _p(v) for t, v in values.items()},
                            confidence=_p(max(values.values())),
                            source=self.source,
                        ),
                    }
                )
        return out

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()


class TeacherTagger:
    """``element_tags="teacher:<engine spec>"`` (module docstring): the teacher's choice among
    every leaf of the family (one cached prefix per family, as the labelling run's), and its
    P(yes) for each function tag; a function is a label from :data:`FUNCTION_THRESHOLD` on.
    Nothing loads until the first element."""

    def __init__(self, engine_spec: str, *, teacher: Any = None) -> None:
        from docrooter.teach.base import get_teacher, refuse_jev

        refuse_jev(engine_spec)
        self.teacher = teacher if teacher is not None else get_teacher(engine_spec)
        refuse_jev(self.teacher.name)

    @property
    def source(self) -> str:
        return f"{TEACHER_PREFIX}{self.teacher.name}"

    def tag(self, elements: Sequence[CorpusElement], family: str) -> list[dict[str, TagResult]]:
        from docrooter.taxonomies import builtin_taxonomy
        from docrooter.teach import prompts
        from docrooter.teach.base import TeachItem
        from docrooter.teach.label import TEACHER_MAX_CHARS, TOPIC_TEACHER_INSTRUCTION
        from docrooter.teach.shortlist import element_text, family_options, topic_taxonomy

        if not elements:
            return []
        label = topic_taxonomy().node(family).label
        options = family_options(family)
        items = [
            TeachItem(
                text=element_text(e)[:TEACHER_MAX_CHARS],
                context=prompts.context_text(label, e.heading_path, e.kind),
                options=options,
            )
            for e in elements
        ]
        topics = self.teacher.choose(
            items,
            instruction=TOPIC_TEACHER_INSTRUCTION.format(family=label.lower()),
            prefix_key=f"topic:{family}",
            prefix_options=options,
        )
        bare = [item.model_copy(update={"options": []}) for item in items]
        answers: dict[str, list[float]] = {}
        for node in builtin_taxonomy("element_functions").leaves():
            question = prompts.function_question(node.label, node.description or "")
            answers[node.id] = self.teacher.yes_no(bare, question=question)
        out = []
        for i, topic in enumerate(topics):
            values = {tag: float(p[i]) for tag, p in answers.items()}
            out.append(
                {
                    "topic": _topic_result(dict(topic), self.source),
                    "function": TagResult(
                        labels=[t for t, v in values.items() if v >= FUNCTION_THRESHOLD],
                        probs={t: _p(v) for t, v in values.items()},
                        confidence=_p(min(max(v, 1.0 - v) for v in values.values())),
                        source=self.source,
                    ),
                }
            )
        return out

    def close(self) -> None:
        self.teacher.close()
