"""The relevance model: P(relevant | element, need) from tag, similarity and structure features
(spec §5).

:func:`features` builds one row per element:

- ``p_topic``: the topic tagger's probability mass on the need's topics (0 when untagged);
- ``p_func``: the highest function-tag probability among the need's functions (0 likewise);
- ``sim``: the cosine between the element's embedding and the need text, given by the caller;
- ``section_hit``: 1 when the element's section's normalised label is one of the need's
  topics (a section label is a topic's leaf name, e.g. ``risk_factors`` for
  ``annual_report.risk_factors``; the full topic id matches too);
- ``kind_*``: one-hot over the six content kinds (:data:`CONTENT_KINDS`);
- ``log_tokens``: ``log(1 + estimate_tokens(text))``.

A free-text query has no topics or functions, so its ``p_topic``, ``p_func`` and
``section_hit`` are 0 and only similarity, kind and length carry it.

:class:`Combiner` is an L2-regularised logistic regression over standardised features, fitted
with Newton's method in numpy (no scikit-learn). It is fitted on calib-fit documents only
(spec §0.1): :meth:`Combiner.fit` refuses any row from another half. It is saved as JSON.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from docrooter.corpus.store import calib_half
from docrooter.elements.models import TagResult
from docrooter.select.fields import Field
from docrooter.views import estimate_tokens

__all__ = ["CONTENT_KINDS", "FEATURE_NAMES", "Combiner", "features"]

CONTENT_KINDS: tuple[str, ...] = (
    "paragraph",
    "list_item",
    "table",
    "caption",
    "footnote",
    "formula",
)
"""The six content kinds: the element tagger's ``TAGGED_KINDS``."""

FEATURE_NAMES: tuple[str, ...] = (
    "p_topic",
    "p_func",
    "sim",
    "section_hit",
    *(f"kind_{kind}" for kind in CONTENT_KINDS),
    "log_tokens",
)
"""The columns of :func:`features`, in order."""

COMBINER_FORMAT = "docrooter-relevance-combiner"
COMBINER_VERSION = 1
_MAX_NEWTON_STEPS = 100
_STEP_TOL = 1e-10
_MIN_SCALE = 1e-12


class _ElementLike(Protocol):
    @property
    def kind(self) -> str: ...

    @property
    def text(self) -> str: ...


def _need_tags(need: Field | Sequence[Field] | str) -> tuple[set[str], set[str]]:
    """The topics and functions of a need: a field, the union over a list of fields, or none
    for a free-text query."""
    if isinstance(need, str):
        return set(), set()
    fields = [need] if isinstance(need, Field) else list(need)
    topics = {t for f in fields for t in f.topics}
    functions = {fn for f in fields for fn in f.functions}
    return topics, functions


def features(
    elements: Sequence[_ElementLike],
    tags: Sequence[Mapping[str, TagResult] | None],
    sims: Sequence[float] | np.ndarray,
    sections: Sequence[str | None],
    need: Field | Sequence[Field] | str,
) -> np.ndarray:
    """The spec §5 feature matrix, ``(len(elements), len(FEATURE_NAMES))`` float64.

    ``tags[i]`` is element i's facet -> :class:`~docrooter.elements.models.TagResult` map
    (``None`` when untagged), ``sims[i]`` its cosine to the need text, and ``sections[i]`` the
    normalised label of its section (``None`` when unlabelled). All are aligned with
    ``elements``.
    """
    m = len(elements)
    sim_values = np.asarray(sims, dtype=np.float64)
    if len(tags) != m or sim_values.shape != (m,) or len(sections) != m:
        raise ValueError(
            f"features: {m} elements but {len(tags)} tag maps, {sim_values.shape} sims and "
            f"{len(sections)} section labels"
        )
    topics, functions = _need_tags(need)
    section_keys = topics | {t.split(".", 1)[1] for t in topics if "." in t}
    kind_column = {kind: 4 + i for i, kind in enumerate(CONTENT_KINDS)}

    X = np.zeros((m, len(FEATURE_NAMES)), dtype=np.float64)
    X[:, 2] = sim_values
    for i, (element, element_tags, section) in enumerate(
        zip(elements, tags, sections, strict=True)
    ):
        if element_tags:
            topic = element_tags.get("topic")
            if topic is not None and topics:
                X[i, 0] = min(1.0, sum(topic.probs.get(t, 0.0) for t in topics))
            function = element_tags.get("function")
            if function is not None and functions:
                X[i, 1] = max(function.probs.get(fn, 0.0) for fn in functions)
        if section is not None and section in section_keys:
            X[i, 3] = 1.0
        column = kind_column.get(element.kind)
        if column is not None:
            X[i, column] = 1.0
        X[i, -1] = math.log1p(estimate_tokens(element.text))
    return X


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # overflow-free


class Combiner:
    """P(relevant | features): logistic regression on standardised features, with an L2
    penalty ``l2`` on the weights (not the intercept)."""

    def __init__(self, *, l2: float = 1.0, feature_names: Sequence[str] = FEATURE_NAMES) -> None:
        if not l2 > 0.0:
            raise ValueError(f"l2 must be positive, got {l2}")
        self.l2 = float(l2)
        self.feature_names = tuple(feature_names)
        self.mean = np.zeros(len(self.feature_names))
        self.scale = np.ones(len(self.feature_names))
        self.coef = np.zeros(len(self.feature_names))  # on the standardised features
        self.intercept = 0.0
        self.n_train = 0
        self.n_docs = 0
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, *, doc_ids: Sequence[str]) -> Combiner:
        """Fit on rows ``X`` with labels ``y`` (0/1). ``doc_ids[i]`` is row i's document, and
        every one must be in calib-fit (:func:`~docrooter.corpus.calib_half`): the combiner
        never sees a calib-crc document, which the dial calibrates on (spec §0.1)."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        d = len(self.feature_names)
        if X.ndim != 2 or X.shape[1] != d:
            raise ValueError(f"X must be (rows, {d}) for {self.feature_names}, got {X.shape}")
        if y.shape != (X.shape[0],) or len(doc_ids) != X.shape[0]:
            raise ValueError(
                f"{X.shape[0]} rows but {y.shape} labels and {len(doc_ids)} document ids"
            )
        if not np.isin(y, (0.0, 1.0)).all() or y.min() == y.max():
            raise ValueError("y must be 0/1 labels with both classes present")
        if not np.isfinite(X).all():
            raise ValueError("X has non-finite values")
        documents = set(doc_ids)
        outside = sorted(doc for doc in documents if calib_half(doc) != "fit")
        if outside:
            raise ValueError(
                f"the combiner fits on calib-fit documents only (spec §0.1); "
                f"{len(outside)} are calib-crc, e.g. {outside[:3]}"
            )

        self.mean = X.mean(axis=0)
        std = X.std(axis=0)
        self.scale = np.where(std > _MIN_SCALE, std, 1.0)
        Z = (X - self.mean) / self.scale
        A = np.hstack([np.ones((Z.shape[0], 1)), Z])
        penalty = np.full(d + 1, self.l2)
        penalty[0] = 0.0
        w = np.zeros(d + 1)

        def objective(v: np.ndarray) -> float:
            z = A @ v
            return float(np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * np.sum(penalty * v * v))

        current = objective(w)
        for _ in range(_MAX_NEWTON_STEPS):
            p = _sigmoid(A @ w)
            gradient = A.T @ (p - y) + penalty * w
            hessian = (A.T * (p * (1.0 - p))) @ A + np.diag(penalty)
            step = np.linalg.solve(hessian + 1e-12 * np.eye(d + 1), gradient)
            t = 1.0
            candidate = objective(w - step)
            while candidate > current and t > 1e-8:  # backtrack: the objective never rises
                t *= 0.5
                candidate = objective(w - t * step)
            w = w - t * step
            current = candidate
            if np.abs(t * step).max() < _STEP_TOL:
                break
        self.intercept = float(w[0])
        self.coef = w[1:].copy()
        self.n_train = int(X.shape[0])
        self.n_docs = len(documents)
        self.fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """P(relevant) per row of ``X``, shape ``(rows,)``."""
        if not self.fitted:
            raise ValueError("the combiner is not fitted")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError(f"X must be (rows, {len(self.feature_names)}), got {X.shape}")
        return _sigmoid(((X - self.mean) / self.scale) @ self.coef + self.intercept)

    def to_dict(self) -> dict[str, Any]:
        if not self.fitted:
            raise ValueError("the combiner is not fitted")
        return {
            "format": COMBINER_FORMAT,
            "version": COMBINER_VERSION,
            "feature_names": list(self.feature_names),
            "l2": self.l2,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
            "n_train": self.n_train,
            "n_docs": self.n_docs,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Combiner:
        if data.get("format") != COMBINER_FORMAT or data.get("version") != COMBINER_VERSION:
            raise ValueError(
                f"not a {COMBINER_FORMAT} v{COMBINER_VERSION}: "
                f"{data.get('format')!r} v{data.get('version')!r}"
            )
        model = cls(l2=float(data["l2"]), feature_names=data["feature_names"])
        d = len(model.feature_names)
        model.mean = np.asarray(data["mean"], dtype=np.float64)
        model.scale = np.asarray(data["scale"], dtype=np.float64)
        model.coef = np.asarray(data["coef"], dtype=np.float64)
        if not model.mean.shape == model.scale.shape == model.coef.shape == (d,):
            raise ValueError("combiner weights do not match its feature names")
        model.intercept = float(data["intercept"])
        model.n_train = int(data["n_train"])
        model.n_docs = int(data["n_docs"])
        model.fitted = True
        return model

    def save(self, path: str | Path) -> None:
        """Write the combiner as JSON (``selector.json`` when no tagger bundle holds it)."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Combiner:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
