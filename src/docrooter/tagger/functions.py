"""Function heads (spec §9; §4.0 ruling 6): one-vs-rest logistic regression per function tag
on the student's embeddings.

Targets per tag, from the teacher's yes/no answers and the zero-shot prescreen:

- the tag was **asked**: P(yes) >= ``pos`` (0.7) is a positive, <= ``neg`` (0.2) a negative,
  anything between is left out;
- the tag was **not asked**: the element is left out of that tag's data, unless its prescreen
  score for the tag is below ``implicit_neg_prescreen`` (0.2), which makes it an implicit
  negative.

Only elements in the function sample (those with a teacher ``function`` row) are candidates:
"not asked" is said of the elements the teacher was asking about. The prescreen is a
percentile per tag, so the elements outside the sample would add a fifth of the whole labelled
set as implicit negatives of every tag and swamp the teacher's answers.

A tag needs :data:`MIN_EXAMPLES` positives and as many negatives for a head; without them it
has none (reported), and the tagger does not report it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from docrooter._extras import require
from docrooter.corpus.store import Corpus, LabelKey, LabelRow
from docrooter.teach.base import refuse_jev

__all__ = [
    "FUNCTION_FACET",
    "IMPLICIT_NEG_PRESCREEN",
    "MIN_EXAMPLES",
    "NEG",
    "POS",
    "FunctionRow",
    "function_rows",
    "function_targets",
    "train_function_heads",
]

FUNCTION_FACET = "function"
POS = 0.7
NEG = 0.2
IMPLICIT_NEG_PRESCREEN = 0.2
MIN_EXAMPLES = 5  # positives and negatives a tag needs for a head


class FunctionRow(BaseModel):
    """One element of the function sample: its student input, the teacher's P(yes) for each
    tag it was asked about, and the prescreen score of every tag."""

    text: str
    answers: dict[str, float]
    prescreen: dict[str, float] = Field(default_factory=dict)


def function_targets(
    rows: Sequence[FunctionRow],
    tags: Sequence[str],
    *,
    pos: float = POS,
    neg: float = NEG,
    implicit_neg_prescreen: float = IMPLICIT_NEG_PRESCREEN,
) -> dict[str, list[tuple[int, int]]]:
    """``tag -> [(row index, 1 or 0), ...]``: the rows that train each tag's head and their
    targets (module docstring). A row naming a tag outside ``tags`` is a ``ValueError``."""
    if not 0.0 <= neg < pos <= 1.0:
        raise ValueError(f"need 0 <= neg < pos <= 1, got neg={neg}, pos={pos}")
    known = set(tags)
    out: dict[str, list[tuple[int, int]]] = {tag: [] for tag in tags}
    for i, row in enumerate(rows):
        unknown = sorted(set(row.answers) - known)
        if unknown:
            raise ValueError(f"function row {i} answers unknown tag(s) {unknown}")
        for tag in tags:
            if tag in row.answers:
                p = row.answers[tag]
                if p >= pos:
                    out[tag].append((i, 1))
                elif p <= neg:
                    out[tag].append((i, 0))
            else:
                score = row.prescreen.get(tag)
                if score is not None and score < implicit_neg_prescreen:
                    out[tag].append((i, 0))
    return out


def target_counts(targets: Mapping[str, list[tuple[int, int]]]) -> dict[str, dict[str, int]]:
    """``tag -> {"positive": n, "negative": n}``."""
    return {
        tag: {
            "positive": sum(y for _, y in pairs),
            "negative": sum(1 - y for _, y in pairs),
        }
        for tag, pairs in targets.items()
    }


def train_function_heads(
    embeddings: np.ndarray,
    rows: Sequence[FunctionRow],
    *,
    pos: float = POS,
    neg: float = NEG,
    implicit_neg_prescreen: float = IMPLICIT_NEG_PRESCREEN,
    tags: Sequence[str] | None = None,
    min_examples: int = MIN_EXAMPLES,
    seed: int = 0,
) -> dict[str, Any]:
    """``tag -> LogisticRegression`` (classes ``[0, 1]``) on ``embeddings`` (one row per
    ``rows`` entry), for each tag with ``min_examples`` positives and negatives. ``tags``
    default to the built-in ``element_functions`` leaves; the result keeps their order."""
    linear_model = require("sklearn.linear_model", "models")
    embeddings = np.asarray(embeddings)
    if len(embeddings) != len(rows):
        raise ValueError(f"{len(embeddings)} embeddings for {len(rows)} rows")
    tags = list(tags) if tags is not None else default_tags()
    targets = function_targets(
        rows, tags, pos=pos, neg=neg, implicit_neg_prescreen=implicit_neg_prescreen
    )
    heads: dict[str, Any] = {}
    for tag, counts in target_counts(targets).items():
        if min(counts.values()) < min_examples:
            continue
        index = [i for i, _ in targets[tag]]
        head = linear_model.LogisticRegression(max_iter=1000, random_state=seed)
        head.fit(embeddings[index], [y for _, y in targets[tag]])
        heads[tag] = head
    return heads


def default_tags() -> list[str]:
    from docrooter.taxonomies import builtin_taxonomy

    return [n.id for n in builtin_taxonomy("element_functions").leaves()]


def _refuse_jev_row(row: LabelRow) -> None:
    """No Jev output may enter training or calibration (TypeSafe MCA §2.3(b))."""
    for field in ("teacher", "engine", "model", "name"):
        value = row.provenance.get(field)
        if isinstance(value, str):
            refuse_jev(value)
            refuse_jev(":" + value)  # a bare model reference


def _by_key(
    rows: Mapping[LabelKey, LabelRow] | Iterable[LabelRow],
) -> dict[LabelKey, LabelRow]:
    if isinstance(rows, Mapping):
        rows = rows.values()
    return {row.key: row for row in rows if row.facet == FUNCTION_FACET}


def function_rows(
    corpus: Corpus,
    teacher: Mapping[LabelKey, LabelRow] | Iterable[LabelRow],
    prescreen: Mapping[LabelKey, LabelRow] | Iterable[LabelRow],
    *,
    split: str = "train",
    max_chars: int,
) -> list[FunctionRow]:
    """The function sample of ``split`` as :class:`FunctionRow`, in corpus order: every element
    with a teacher ``function`` row, its prescreen row's scores (none when it has no prescreen
    row) and its student input (:func:`~docrooter.tagger.bundle.student_text`). Teacher rows
    naming Jev raise :class:`~docrooter.errors.TeacherNotAllowedError`."""
    from docrooter.tagger.bundle import student_text

    answers = _by_key(teacher)
    for row in answers.values():
        _refuse_jev_row(row)
    scores = _by_key(prescreen)
    out = []
    for element in corpus.iter_elements(split=split):
        key = LabelKey(element.doc_id, element.element_id, element.text_sha1, FUNCTION_FACET)
        answer = answers.get(key)
        if answer is None:
            continue
        screen = scores.get(key)
        out.append(
            FunctionRow(
                text=student_text(element, max_chars),
                answers=dict(answer.probs),
                prescreen=dict(screen.probs) if screen is not None else {},
            )
        )
    return out
