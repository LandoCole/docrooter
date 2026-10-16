"""Document-type classification: one label per document from its view, its sections, or both."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from docrooter.backends.base import Backend
from docrooter.classify._beams import TreeResult
from docrooter.classify.classifier import DEFAULT_BACKEND, Classifier, _is_trained
from docrooter.classify.engine import TOP_K_PROBS
from docrooter.classify.results import DocTypeResult, PathResult, Step
from docrooter.classify.strategies import Beam, Flat, Greedy
from docrooter.doc_view import DocumentView
from docrooter.document import Document
from docrooter.taxonomy import Taxonomy
from docrooter.views import SectionView, estimate_tokens

Mode = Literal["view", "sections", "hybrid"]
MODES: tuple[str, ...] = ("view", "sections", "hybrid")

VIEW_WEIGHT = 0.6
"""Weight of the view distribution in ``mode="hybrid"``; the sections get the rest."""

LEAF_PROBS_TOP = 10
SECTION_VIEW = SectionView(fields=("title", "lead"), max_tokens=128)
AGGREGATE_BACKEND = "(aggregate)"
"""``Step.backend`` of a path chosen from an aggregated (sections or hybrid) distribution."""

DEFAULT_INSTRUCTION = "What type of document is this?"


def classify_document(
    doc: Document,
    taxonomy: Taxonomy,
    *,
    backend: str | Backend = DEFAULT_BACKEND,
    strategy: Greedy | Beam | Flat | None = None,
    view: DocumentView | None = None,
    instruction: str | None = DEFAULT_INSTRUCTION,
    mode: Mode = "view",
    threshold: float = 0.0,
    **classifier_kwargs: Any,
) -> DocTypeResult:
    """Classify one document's type against ``taxonomy`` (strategy ``Flat()`` by default).

    ``threshold`` defaults to 0, so a document with content always gets a label; gate on
    ``result.confidence`` or pass a ``threshold`` to get ``"low_confidence"`` instead.
    ``classifier_kwargs`` go to :class:`Classifier` (``fallback``, ``batch_size``, ...). See
    :meth:`Classifier.classify_documents` for the modes.
    """
    classifier = Classifier(
        taxonomy,
        backend,
        strategy=strategy if strategy is not None else Flat(),
        instruction=instruction,
        threshold=threshold,
        **classifier_kwargs,
    )
    return classifier.classify_documents([doc], view=view, mode=mode)[0]


def classify_documents(
    classifier: Classifier, docs: Sequence[Document], view: DocumentView, mode: str
) -> list[DocTypeResult]:
    """The implementation of :meth:`Classifier.classify_documents`."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}; got {mode!r}")
    if mode != "view" and _is_trained(classifier.backend):
        raise ValueError(
            f"mode={mode!r} scores sections with Flat, which needs a zero-shot backend"
        )
    docs = list(docs)
    has_content = [_has_content(doc) for doc in docs]
    count = classifier._token_counter() if any(has_content) else estimate_tokens

    view_texts: dict[str, str] = {}
    rendered: dict[int, str] = {}
    weights: dict[int, dict[str, int]] = {}  # doc index -> section key -> token weight
    section_texts: dict[str, str] = {}
    for i, doc in enumerate(docs):
        if not has_content[i]:
            continue
        rendered[i] = view.render(doc, count).text
        if mode != "sections":
            view_texts[f"v{i}"] = rendered[i]
        if mode != "view":
            leaves = [
                s for s in doc.leaves() if s.text.strip() or (s.level > 0 and s.title.strip())
            ]
            section_views = SECTION_VIEW.render_all(doc, [s.id for s in leaves], count)
            for section in leaves:
                key = f"s{i}:{section.id}"
                section_texts[key] = section_views[section.id].text
                weights.setdefault(i, {})[key] = max(1, estimate_tokens(section.text))

    start = time.perf_counter()
    results = _run(classifier, view_texts, section_texts)
    elapsed = time.perf_counter() - start
    classifier._warn_no_backend(results.values())

    n_classified = sum(1 for i in rendered if mode != "sections" or i in weights)
    per_doc = elapsed / n_classified if n_classified else 0.0
    root_threshold = classifier.taxonomy.node(Taxonomy.ROOT_ID).threshold
    threshold = root_threshold if root_threshold is not None else classifier.settings.threshold
    out: list[DocTypeResult] = []
    for i, doc in enumerate(docs):
        if i not in rendered or (mode == "sections" and i not in weights):
            out.append(_empty(doc, mode, rendered.get(i, "")))
            continue
        view_result = results.get(f"v{i}")
        section_results = [(results[k], w) for k, w in weights.get(i, {}).items()]
        view_dist = _view_distribution(view_result, classifier.strategy) if view_result else {}
        if mode == "view":
            dist = view_dist
        else:
            dist = _sections_distribution(section_results)
            if mode == "hybrid":
                dist = mix_distributions(view_dist, dist)
        reasons = [r[0].stopped_reason for r in ([view_result] if view_result else [])]
        reasons += [r[0].stopped_reason for r, _ in section_results]
        best, confidence = decide_doc_type(
            dist,
            classifier.taxonomy,
            threshold,
            reasons,
            view_best=view_result[0] if mode == "view" and view_result else None,
        )
        out.append(
            DocTypeResult(
                source=doc.source,
                best=best,
                confidence=confidence,
                leaf_probs=_top(dist, LEAF_PROBS_TOP),
                view_text=rendered[i],
                timings={"classify": per_doc},
                mode=mode,  # type: ignore[arg-type]
            )
        )
    return out


def mix_distributions(
    view: Mapping[str, float], sections: Mapping[str, float], view_weight: float = VIEW_WEIGHT
) -> dict[str, float]:
    """``view_weight * view + (1 - view_weight) * sections`` per node (missing = 0), summing to 1.

    Empty when both are empty (or all zero).
    """
    mixed: dict[str, float] = {}
    for dist, weight in ((view, view_weight), (sections, 1.0 - view_weight)):
        for node_id, p in dist.items():
            mixed[node_id] = mixed.get(node_id, 0.0) + weight * p
    total = sum(mixed.values())
    return {node_id: p / total for node_id, p in mixed.items()} if total > 0 else {}


# --- internals ----------------------------------------------------------------------------


def _has_content(doc: Document) -> bool:
    """Whether anything but the source name could be shown: a title, a heading or body text.

    A preamble's title is synthetic (``"Preamble"`` or the file stem), and so is a title equal
    to the file stem (PDF loaders set one), so neither counts. A PDF without a text layer
    (``"no_text_layer"`` in ``meta["warnings"]``) has no content.
    """
    if "no_text_layer" in (doc.meta.get("warnings") or []):
        return False
    title = (doc.title or "").strip()
    if title and title != Path(doc.source).stem:
        return True
    return any(s.text.strip() or (s.level > 0 and s.title.strip()) for s in doc.sections.values())


def _run(
    classifier: Classifier, view_texts: dict[str, str], section_texts: dict[str, str]
) -> dict[str, TreeResult]:
    """Score the views with the classifier's strategy and the sections with full-distribution
    Flat; one traversal when both are plain Flat."""
    settings = classifier.settings.model_copy(update={"full_probs": True})
    if view_texts and section_texts and classifier.strategy == Flat():
        return classifier._run({**view_texts, **section_texts}, Flat(), settings)
    results: dict[str, TreeResult] = {}
    if view_texts:
        results.update(classifier._run(view_texts, classifier.strategy, settings))
    if section_texts:
        results.update(classifier._run(section_texts, Flat(), settings))
    return results


def _empty(doc: Document, mode: str, view_text: str) -> DocTypeResult:
    return DocTypeResult(
        source=doc.source,
        best=PathResult(node_ids=[], stopped_reason="empty_text"),
        view_text=view_text,
        timings={"classify": 0.0},
        mode=mode,  # type: ignore[arg-type]
    )


def _flat_distribution(result: TreeResult) -> dict[str, float]:
    """Every candidate's probability, from a full-probs Flat result (even a low-confidence one)."""
    best, alternatives, _ = result
    for path in (best, *alternatives):
        if path is not None and path.steps:
            return dict(path.steps[0].probs)
    return {}


def _path_distribution(result: TreeResult) -> dict[str, float]:
    """``{last node: score}`` of the best path and its alternatives, scaled to sum at most 1."""
    best, alternatives, _ = result
    dist: dict[str, float] = {}
    for path in (best, *alternatives):
        if path is not None and path.node_ids:
            dist.setdefault(path.node_ids[-1], path.score)
    total = sum(dist.values())
    return {n: p / total for n, p in dist.items()} if total > 1 else dist


def _view_distribution(result: TreeResult, strategy: Greedy | Beam | Flat) -> dict[str, float]:
    return _flat_distribution(result) if isinstance(strategy, Flat) else _path_distribution(result)


def _sections_distribution(results: list[tuple[TreeResult, int]]) -> dict[str, float]:
    """Token-weighted mean of the sections' Flat distributions (sections without one skipped)."""
    totals: dict[str, float] = {}
    weight_sum = 0
    for result, weight in results:
        dist = _flat_distribution(result)
        if not dist:
            continue
        weight_sum += weight
        for node_id, p in dist.items():
            totals[node_id] = totals.get(node_id, 0.0) + weight * p
    return {n: p / weight_sum for n, p in totals.items()} if weight_sum else {}


def _top(dist: Mapping[str, float], k: int) -> dict[str, float]:
    # sorted() is stable: ties keep the distribution's (taxonomy) order.
    return dict(sorted(dist.items(), key=lambda item: -item[1])[:k])


def decide_doc_type(
    dist: Mapping[str, float],
    taxonomy: Taxonomy,
    threshold: float,
    reasons: Sequence[str | None] = (),
    view_best: PathResult | None = None,
) -> tuple[PathResult, float]:
    """The decision rule of every mode: the argmax node of ``dist`` and its probability.

    Below ``threshold`` the path is empty and stopped as ``"low_confidence"`` (the probability is
    still returned). Otherwise it is ``view_best`` (the traversal's own path, steps trimmed) when
    that path ends at the argmax, else the argmax node's taxonomy path with one root ``Step``
    over ``dist`` (backend ``"(aggregate)"``). An empty ``dist`` gives an empty path stopped for
    the first of ``reasons`` that is set (``"low_confidence"`` if none) and probability 0.
    """
    if not dist:
        reason = next((r for r in reasons if r is not None), "low_confidence")
        return PathResult(node_ids=[], stopped_reason=reason), 0.0
    shown = _top(dist, TOP_K_PROBS)
    node_id = next(iter(shown))
    confidence = dist[node_id]
    if confidence < threshold:
        return PathResult(node_ids=[], stopped_reason="low_confidence"), confidence
    if view_best is not None and view_best.node_ids and view_best.node_ids[-1] == node_id:
        return _trim_steps(view_best), confidence
    nodes = taxonomy.path(node_id)
    step = Step(node_id=Taxonomy.ROOT_ID, chosen=[node_id], probs=shown, backend=AGGREGATE_BACKEND)
    path = PathResult(
        node_ids=[n.id for n in nodes],
        labels=[n.label for n in nodes],
        score=confidence,
        steps=[step],
        stopped_reason=None,
    )
    return path, confidence


def _trim_steps(path: PathResult) -> PathResult:
    """``path`` with each step's ``probs`` cut back to the top ``TOP_K_PROBS`` plus its choices."""
    steps = []
    for step in path.steps:
        ranked = sorted(step.probs.items(), key=lambda item: -item[1])
        probs = {c: p for i, (c, p) in enumerate(ranked) if i < TOP_K_PROBS or c in step.chosen}
        steps.append(step.model_copy(update={"probs": probs}))
    return path.model_copy(update={"steps": steps})
