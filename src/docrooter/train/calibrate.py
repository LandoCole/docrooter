"""Temperature calibration: pick, per backend, the temperature that best fits gold decisions."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from docrooter.backends.base import Candidate, Scores, candidate_text, to_probs
from docrooter.classify import engine
from docrooter.taxonomy import Taxonomy

if TYPE_CHECKING:
    from docrooter.classify.classifier import Classifier
    from docrooter.train.dataset import Dataset

DEFAULT_GRID: tuple[float, ...] = tuple(float(t) for t in np.logspace(-3, 1, 50))
MIN_DECISIONS = 5
_PROB_FLOOR = 1e-12
_FLAT_NLL = 1e-9


@dataclass
class _Chunk:
    """Raw scores of one scored batch at one node, and the gold child's column per row."""

    scores: Scores
    gold: np.ndarray
    multi_label: bool


@dataclass
class _BackendData:
    base_temperature: float
    chunks: list[_Chunk] = field(default_factory=list)

    @property
    def n(self) -> int:
        return sum(len(c.gold) for c in self.chunks)


def _collect(clf: Classifier, dev: Dataset) -> dict[str, _BackendData]:
    """Raw scores at the gold internal nodes, grouped by the name of the backend that decides.

    Routing, candidates, instruction, batching and the (truncated) text are those of
    classification, so the scores are what the classifier would see at those nodes.
    """
    resolved = dev.resolve(clf.taxonomy)
    prepared = clf.prepare_texts([e.text for e in resolved])
    by_node: dict[str, tuple[list[str], list[str]]] = {}
    for example, text in zip(resolved, prepared, strict=True):
        if not text.strip():  # blank text never reaches a backend
            continue
        parents = [Taxonomy.ROOT_ID, *example.path[:-1]]
        for parent, child in zip(parents, example.path, strict=True):
            texts, golds = by_node.setdefault(parent, ([], []))
            texts.append(text)
            golds.append(child)

    settings = clf.settings
    data: dict[str, _BackendData] = {}
    for node_id in (n for n in (Taxonomy.ROOT_ID, *clf.taxonomy.nodes) if n in by_node):
        children = clf.taxonomy.children(node_id)
        if len(children) < 2:  # a single child is chosen without a backend: nothing to calibrate
            continue
        backend = engine.pick_backend(clf.router, node_id, len(children))
        if backend is None:
            continue
        texts, golds = by_node[node_id]
        index = {c.id: i for i, c in enumerate(children)}
        node = clf.taxonomy.node(node_id)
        candidates = [Candidate(c.id, candidate_text(c), tuple(c.examples)) for c in children]
        entry = data.setdefault(backend.name, _BackendData(engine.backend_temperature(backend)))
        for start in range(0, len(texts), settings.batch_size):
            batch = texts[start : start + settings.batch_size]
            scores = engine.score_batch(
                backend, batch, candidates, settings.instruction, node.multi_label
            )
            gold = np.array([index[g] for g in golds[start : start + settings.batch_size]])
            entry.chunks.append(_Chunk(scores, gold, node.multi_label))
    return data


def _nll(entry: _BackendData, temperature: float) -> float:
    """Mean negative log probability of the gold child over all of a backend's decisions."""
    total = 0.0
    for chunk in entry.chunks:
        probs = to_probs(chunk.scores, temperature=temperature, multi_label=chunk.multi_label)
        picked = probs[np.arange(len(chunk.gold)), chunk.gold]
        total += float(-np.log(np.clip(picked, _PROB_FLOOR, None)).sum())
    return total / entry.n


def mean_nll(clf: Classifier, dev: Dataset) -> dict[str, float]:
    """Mean NLL of the gold child at the gold internal nodes, per backend name.

    Uses each backend's temperature as the classifier would (``settings.temperatures`` first).
    """
    return {
        name: _nll(entry, clf.settings.temperatures.get(name, entry.base_temperature))
        for name, entry in _collect(clf, dev).items()
    }


def calibrate(clf: Classifier, dev: Dataset, *, grid: Sequence[float] = DEFAULT_GRID) -> Classifier:
    """A copy of ``clf`` whose per-backend temperatures minimise the NLL of ``dev``'s gold paths.

    For every backend, the raw scores at the gold internal nodes it decides are collected (same
    routing and text as classification) and the temperature from ``grid`` with the lowest
    mean NLL of the gold child, through :func:`~docrooter.backends.base.to_probs`, goes to
    ``settings.temperatures[backend.name]``. A backend that decides fewer than
    ``MIN_DECISIONS`` of the gold decisions, or whose probabilities do not depend on the
    temperature, keeps its temperature. The outcome per backend is in the copy's
    ``calibration_report``: ``n``, and either ``temperature``, ``nll_before`` and ``nll_after``
    or ``skipped`` with the reason.
    """
    candidates = [float(t) for t in grid]
    if not candidates or any(t <= 0 for t in candidates):
        raise ValueError("grid must hold at least one temperature, all greater than 0")
    out = copy.copy(clf)
    out.settings = clf.settings.model_copy(deep=True)
    out._resolved = dict(clf._resolved)
    report: dict[str, dict[str, Any]] = {}
    for name, entry in _collect(clf, dev).items():
        if entry.n < MIN_DECISIONS:
            report[name] = {"n": entry.n, "skipped": f"fewer than {MIN_DECISIONS} decisions"}
            continue
        losses = np.array([_nll(entry, t) for t in candidates])
        if np.ptp(losses) < _FLAT_NLL:
            report[name] = {"n": entry.n, "skipped": "probabilities do not depend on temperature"}
            continue
        best = candidates[int(np.argmin(losses))]
        before = clf.settings.temperatures.get(name, entry.base_temperature)
        out.settings.temperatures[name] = best
        report[name] = {
            "n": entry.n,
            "temperature": best,
            "nll_before": _nll(entry, before),
            "nll_after": float(losses.min()),
        }
    out.calibration_report = report
    return out
