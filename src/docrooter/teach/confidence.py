"""The similarity-gap filter and the confident set (spec §8, §4.0 ruling 7; TELEClass §3.3).

- **Centroids:** a class's centroid is the mean embedding of its teacher-labelled train elements
  that contain one of its keywords (all of them when none does), L2-normalised.
- **Core classes:** an element's classes sorted by cosine to their centroids; the core classes
  are those ranked above the largest drop between neighbours, and that drop is its gap.
- **Keep rule:** an element stays when the teacher's choice is one of its core classes, or the
  teacher's probability is at least the ``prob_override_quantile`` (0.9) quantile of the teacher
  probabilities of the elements given that class (linear interpolation): only about the most
  confident decile of each class bypasses the gap check. This amends spec §4.0 ruling 7 (an
  absolute 0.8) by controller ruling: the teacher is overconfident (mean top probability 0.974
  in the L7 smoke run), so an absolute threshold would keep nearly everything.
- **Combined confidence:** the teacher's probability times the element's gap rank percentile
  (the share of all scored elements whose gap is at most its own).
- **Trim:** per class, the top ``keep`` (75%) by combined confidence, rounded up, and at least
  ``min_per_class`` (20) where available. A class with fewer than ``min_per_class`` elements
  passing the keep rule is left untrained: its elements are kept with their family as the path,
  so the class is predicted only via its family (Review Focus 4), and it is reported.

This filter is for the topic facet only.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, Field

from docrooter.corpus.store import LabelKey, LabelRow
from docrooter.taxonomy import Taxonomy
from docrooter.teach.enrich import teacher_choices

__all__ = [
    "KEEP",
    "MIN_PER_CLASS",
    "PROB_OVERRIDE_QUANTILE",
    "ClassReport",
    "TrainingExample",
    "TrainingSet",
    "class_centroids",
    "confident_set",
    "core_classes",
    "gap_core_classes",
]

logger = logging.getLogger("docrooter.teach")

KEEP = 0.75
MIN_PER_CLASS = 20
PROB_OVERRIDE_QUANTILE = 0.9

Core = tuple[list[str], float]


def _unit(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norm > 0, norm, 1.0)


def class_centroids(
    vectors: np.ndarray, labels: Sequence[str], has_keyword: Sequence[bool]
) -> dict[str, np.ndarray]:
    """Per class, the L2-normalised mean of the rows of ``vectors`` labelled with it whose
    ``has_keyword`` is true: the elements that contain one of the class keywords. A class none
    of whose elements contains a keyword (it has no keywords) falls back to all its elements."""
    vectors = np.asarray(vectors)
    if len(labels) != len(vectors) or len(has_keyword) != len(vectors):
        raise ValueError("vectors, labels and has_keyword must have the same length")
    rows: dict[str, list[int]] = {}
    with_keyword: dict[str, list[int]] = {}
    for i, (label, hit) in enumerate(zip(labels, has_keyword, strict=True)):
        rows.setdefault(label, []).append(i)
        if hit:
            with_keyword.setdefault(label, []).append(i)
    return {
        label: _unit(vectors[with_keyword.get(label) or members].mean(axis=0, dtype=np.float64))
        for label, members in rows.items()
    }


def gap_core_classes(scores: Mapping[str, float]) -> Core:
    """``(core, gap)`` for one element's class scores: the classes ranked above the largest
    drop between neighbours in the descending order, and that drop. Of equal drops the first
    counts; equal scores keep their order in ``scores``. One class is its own core with gap 0;
    no class gives ``([], 0.0)``."""
    ranked = sorted(scores.items(), key=lambda item: -item[1])
    if len(ranked) < 2:
        return [c for c, _ in ranked], 0.0
    values = np.array([s for _, s in ranked], dtype=np.float64)
    drops = values[:-1] - values[1:]
    m = int(np.argmax(drops))
    return [c for c, _ in ranked[: m + 1]], float(drops[m])


def core_classes(
    vectors: np.ndarray,
    candidates: Sequence[Sequence[str]],
    centroids: Mapping[str, np.ndarray],
) -> list[Core]:
    """``gap_core_classes`` for each row of ``vectors``, scored by cosine against the centroids
    of its ``candidates`` (a class with no centroid is not scored)."""
    out: list[Core] = []
    for row, classes in zip(np.asarray(vectors), candidates, strict=True):
        vector = _unit(row.astype(np.float64))  # row by row: no copy of the whole matrix
        scored = [c for c in classes if c in centroids]
        if not scored:
            out.append(([], 0.0))
            continue
        matrix = np.stack([centroids[c] for c in scored])
        out.append(gap_core_classes(dict(zip(scored, (matrix @ vector).tolist(), strict=True))))
    return out


class TrainingExample(BaseModel):
    """One confident element: its key, the teacher's leaf (``label``) and the path it trains
    (``[family, leaf]``, or ``[family]`` when the leaf is left untrained)."""

    doc_id: str
    element_id: str
    text_sha1: str
    label: str
    path: list[str]
    teacher_prob: float
    gap: float
    in_core: bool
    confidence: float


class ClassReport(BaseModel):
    """Counts for one topic leaf: elements the teacher gave it (among those scored), those
    passing the keep rule, those kept after the trim, and whether it is trained.
    ``override_prob`` is the class's teacher probability at or above which an element outside
    its core classes is kept (``None`` when no element was given the class)."""

    labelled: int = 0
    confident: int = 0
    kept: int = 0
    trained: bool = False
    override_prob: float | None = None


class TrainingSet(BaseModel):
    """The confident topic set: examples in the order of the label rows, a report per leaf of
    the taxonomy, and the parameters used."""

    examples: list[TrainingExample] = Field(default_factory=list)
    classes: dict[str, ClassReport] = Field(default_factory=dict)
    params: dict[str, float] = Field(default_factory=dict)

    @property
    def untrained(self) -> list[str]:
        """The leaves left untrained (fewer than ``min_per_class`` confident elements), in
        taxonomy order: predicted only via their family."""
        return [c for c, report in self.classes.items() if not report.trained]


def _default_taxonomy() -> Taxonomy:
    from docrooter.taxonomies import builtin_taxonomy

    return builtin_taxonomy("element_topics")


def confident_set(
    labels: Mapping[LabelKey, LabelRow] | Iterable[LabelRow],
    cores: Mapping[LabelKey, Core],
    *,
    keep: float = KEEP,
    min_per_class: int = MIN_PER_CLASS,
    prob_override_quantile: float = PROB_OVERRIDE_QUANTILE,
    taxonomy: Taxonomy | None = None,
) -> TrainingSet:
    """The confident training set from the teacher's ``topic`` rows and each element's
    ``(core classes, gap)`` (module docstring). Only elements in ``cores`` are candidates, so
    ``cores`` decides the split. ``taxonomy`` is the topic tree (default: the built-in
    ``element_topics``). Rows naming Jev raise
    :class:`~docrooter.errors.TeacherNotAllowedError`.
    """
    if not 0 < keep <= 1:
        raise ValueError(f"keep must be in (0, 1], got {keep}")
    if min_per_class < 1:
        raise ValueError(f"min_per_class must be >= 1, got {min_per_class}")
    if not 0 <= prob_override_quantile <= 1:
        raise ValueError(f"prob_override_quantile must be in [0, 1], got {prob_override_quantile}")
    taxonomy = taxonomy if taxonomy is not None else _default_taxonomy()
    choices = {
        key: choice for key, choice in teacher_choices(labels, taxonomy).items() if key in cores
    }
    gaps = np.sort(np.array([cores[key][1] for key in choices], dtype=np.float64))
    probs: dict[str, list[float]] = {}
    for label, prob in choices.values():
        probs.setdefault(label, []).append(prob)
    override = {
        label: float(np.quantile(np.array(values), prob_override_quantile))
        for label, values in probs.items()
    }

    by_class: dict[str, list[TrainingExample]] = {}
    classes = {leaf.id: ClassReport() for leaf in taxonomy.leaves()}
    for key, (label, prob) in choices.items():
        core, gap = cores[key]
        report = classes[label]
        report.labelled += 1
        report.override_prob = override[label]
        in_core = label in core
        if not (in_core or prob >= override[label]):
            continue
        report.confident += 1
        percentile = np.searchsorted(gaps, gap, side="right") / len(gaps)
        by_class.setdefault(label, []).append(
            TrainingExample(
                doc_id=key.doc_id,
                element_id=key.element_id,
                text_sha1=key.text_sha1,
                label=label,
                path=[n.id for n in taxonomy.path(label)],
                teacher_prob=prob,
                gap=float(gap),
                in_core=in_core,
                confidence=float(prob * percentile),
            )
        )

    for label, members in by_class.items():
        report = classes[label]
        report.trained = len(members) >= min_per_class
        if report.trained:
            n_keep = max(math.ceil(keep * len(members)), min_per_class)
            ranked = sorted(members, key=lambda e: -e.confidence)  # stable: row order on ties
            members[:] = ranked[:n_keep]
        else:
            for example in members:
                example.path = example.path[:-1]
        report.kept = len(members)

    examples = [e for members in by_class.values() for e in members]
    order = {(k.doc_id, k.element_id, k.text_sha1): i for i, k in enumerate(choices)}
    examples.sort(key=lambda e: order[(e.doc_id, e.element_id, e.text_sha1)])
    result = TrainingSet(
        examples=examples,
        classes=classes,
        params={
            "keep": keep,
            "min_per_class": min_per_class,
            "prob_override_quantile": prob_override_quantile,
        },
    )
    if result.untrained:
        logger.warning(
            "topic classes with fewer than %d confident elements, left untrained (predicted via "
            "their family): %s",
            min_per_class,
            ", ".join(f"{c} ({classes[c].confident})" for c in result.untrained),
        )
    return result
