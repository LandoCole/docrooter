"""Level-synchronous taxonomy traversal for many texts at once.

Every round advances all unfinished paths of all texts by one decision. The pending
``(text, node)`` decisions are grouped by ``(node, backend)`` and each group is scored with one
``backend.score`` call per mini-batch, so a taxonomy of depth d costs about d batched rounds.
The traversal is iterative: no recursion over the tree.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from docrooter.backends.base import Backend, Candidate, Scores, candidate_text, to_probs
from docrooter.classify._beams import PartialPath, TreeResult, collect, prune
from docrooter.classify.results import PathResult, Step
from docrooter.classify.strategies import Beam, Flat, Greedy
from docrooter.errors import BackendError
from docrooter.taxonomy import Taxonomy, TaxonomyNode

Router = Callable[[str], Iterable[Backend]]
"""Candidate backends for a decision at a node id, in priority order.

The iterable may be lazy; ``pick_backend`` stops consuming at the first usable backend."""

AUTO_BACKEND = "(auto)"
"""``Step.backend`` for a single-child node, which is chosen with p=1.0 and no backend call."""

TOP_K_PROBS = 5
"""How many of the most probable children a ``Step`` records (chosen children are added)."""


class EngineSettings(BaseModel):
    """Decision settings shared by every node unless the node overrides them.

    ``threshold`` applies where the taxonomy node has no ``threshold`` of its own.
    ``temperatures`` maps a backend name to the temperature used instead of its own.
    ``full_probs`` makes ``run_flat`` record every candidate's probability in its step, not
    just the top ``TOP_K_PROBS``.
    """

    model_config = ConfigDict(extra="forbid")

    threshold: float = 0.35
    min_margin: float = 0.0
    max_fanout: int = Field(default=3, ge=1)
    max_depth: int = Field(default=64, ge=1)
    batch_size: int = Field(default=32, ge=1)
    instruction: str | None = None
    temperatures: dict[str, float] = Field(default_factory=dict)
    full_probs: bool = False


def run_tree(
    texts: dict[str, str],
    taxonomy: Taxonomy,
    router: Router,
    strategy: Greedy | Beam,
    settings: EngineSettings,
) -> dict[str, TreeResult]:
    """Walk every text down ``taxonomy`` from the root; return ``(best, alternatives, selected)``.

    ``best`` is the most probable finished path. If it never crossed a multi-label node,
    ``selected == [best]``. Otherwise ``selected`` holds the most probable path of each branch
    of ``best``'s first multi-label fan-out (the labels assigned together with ``best``), most
    probable first, with ``best`` first. ``alternatives`` are the other finished paths, the
    rival hypotheses, most probable first and at most ``Beam.width`` of them (always empty for
    ``Greedy``).
    """
    if not isinstance(strategy, (Greedy, Beam)):
        raise TypeError(f"run_tree takes Greedy or Beam, not {type(strategy).__name__}")
    width = 1 if isinstance(strategy, Greedy) else strategy.width
    backends: dict[str, Backend | None] = {}

    beams = [PartialPath(key) for key in texts]
    for _ in range(settings.max_depth):
        if all(b.done for b in beams):
            break
        beams = _advance(beams, texts, taxonomy, router, backends, width, settings)
        beams = prune(beams, width)
    beams = [
        b if b.done else b.stop("max_depth" if taxonomy.children(b.node_id) else None)
        for b in beams
    ]
    return collect(beams, list(texts), taxonomy, width)


def run_flat(
    texts: dict[str, str],
    taxonomy: Taxonomy,
    backend: Backend | None,
    strategy: Flat,
    settings: EngineSettings,
) -> dict[str, TreeResult]:
    """Score every leaf (or every node, with ``strategy.allow_internal``) of each text at once.

    Candidates carry their full path (``"A > Other"``), so the same label in different branches
    stays distinct. The argmax node, if its probability reaches the threshold (the root's own
    ``threshold`` when it has one), is ``best``: its path from the top level down with a single
    ``Step`` at the root, and ``selected == [best]``. Otherwise ``best`` is an empty path
    stopped as ``"low_confidence"``, ``selected == []``, and ``alternatives`` start from the
    argmax. ``alternatives`` are the next ``TOP_K_PROBS`` candidates by probability. Each
    step's ``probs`` holds the top ``TOP_K_PROBS`` candidates, or all of them with
    ``settings.full_probs``. A ``None`` backend (nothing can score the root) gives every text
    an empty ``"no_backend"`` path.
    """
    if backend is None:
        stopped = PathResult(node_ids=[], stopped_reason="no_backend")
        return {key: (stopped, [], []) for key in texts}
    nodes = list(taxonomy.nodes.values()) if strategy.allow_internal else taxonomy.leaves()
    candidates = [
        Candidate(n.id, candidate_text(n, taxonomy=taxonomy, full_path=True), tuple(n.examples))
        for n in nodes
    ]
    keys = list(texts)
    probs = _score_candidates(
        backend, [texts[k] for k in keys], candidates, settings, multi_label=False, where="the root"
    )
    root_threshold = taxonomy.node(Taxonomy.ROOT_ID).threshold
    threshold = root_threshold if root_threshold is not None else settings.threshold

    results: dict[str, TreeResult] = {}
    for key, row in zip(keys, probs, strict=True):
        # sorted() is stable, so equal probabilities keep the taxonomy's node order.
        ranked = sorted(((n.id, float(p)) for n, p in zip(nodes, row, strict=True)), key=_by_prob)
        if not ranked:
            results[key] = (PathResult(node_ids=[], stopped_reason="low_confidence"), [], [])
            continue
        shown = dict(ranked if settings.full_probs else ranked[:TOP_K_PROBS])
        paths = [
            _flat_path(node_id, p, shown, backend.name, taxonomy)
            for node_id, p in ranked[: 1 + TOP_K_PROBS]  # all a result can use
        ]
        if ranked[0][1] >= threshold:
            results[key] = (paths[0], paths[1 : 1 + TOP_K_PROBS], [paths[0]])
        else:
            stopped = PathResult(node_ids=[], stopped_reason="low_confidence")
            results[key] = (stopped, paths[:TOP_K_PROBS], [])
    return results


def _by_prob(item: tuple[str, float]) -> float:
    return -item[1]


def _flat_path(
    node_id: str, p: float, shown: dict[str, float], backend_name: str, taxonomy: Taxonomy
) -> PathResult:
    """The path to ``node_id`` as one ``Step`` at the root; ``shown`` always includes the pick."""
    probs = shown if node_id in shown else {**shown, node_id: p}
    nodes = taxonomy.path(node_id)
    return PathResult(
        node_ids=[n.id for n in nodes],
        labels=[n.label for n in nodes],
        score=p,
        steps=[
            Step(
                node_id=Taxonomy.ROOT_ID, chosen=[node_id], probs=dict(probs), backend=backend_name
            )
        ],
        stopped_reason=None,
    )


def decide(
    texts: Sequence[str],
    node_id: str,
    taxonomy: Taxonomy,
    router: Router,
    settings: EngineSettings,
) -> tuple[list[str], np.ndarray, str]:
    """Score the children of ``node_id`` for every text, as ``run_tree`` would at that node.

    Returns the child ids in taxonomy order, probabilities of shape ``(len(texts), n_children)``
    and the backend name. Used for teacher-forced evaluation and calibration.
    """
    node = taxonomy.node(node_id)
    children = taxonomy.children(node_id)
    child_ids = [c.id for c in children]
    if not children:
        raise ValueError(f"node {node_id!r} is a leaf; there is no decision to make")
    if len(children) == 1:
        return child_ids, np.ones((len(texts), 1)), AUTO_BACKEND
    backend = pick_backend(router, node_id, len(children))
    if backend is None:
        raise BackendError(
            f"no backend can decide at node {node_id!r} ({len(children)} children): none of "
            "the routed backends supports it with that many candidates"
        )
    return child_ids, _score(backend, list(texts), node, children, settings), backend.name


def _advance(
    beams: list[PartialPath],
    texts: dict[str, str],
    taxonomy: Taxonomy,
    router: Router,
    backends: dict[str, Backend | None],
    width: int,
    settings: EngineSettings,
) -> list[PartialPath]:
    """Advance every unfinished beam by one decision; finished beams pass through unchanged."""
    out: list[list[PartialPath]] = []
    groups: dict[tuple[str, int], tuple[Backend, TaxonomyNode, list[int]]] = {}
    for i, beam in enumerate(beams):
        if beam.done:
            out.append([beam])
            continue
        node = taxonomy.node(beam.node_id)
        children = taxonomy.children(node.id)
        if not children:
            out.append([beam.stop(None)])
        elif len(children) == 1:
            only = children[0].id
            step = Step(node_id=node.id, chosen=[only], probs={only: 1.0}, backend=AUTO_BACKEND)
            out.append([beam.extend(step, 1.0, fan_out=False)])
        else:
            if node.id not in backends:
                backends[node.id] = pick_backend(router, node.id, len(children))
            backend = backends[node.id]
            if backend is None:
                out.append([beam.stop("no_backend")])
            else:
                groups.setdefault((node.id, id(backend)), (backend, node, []))[2].append(i)
                out.append([])  # filled in once the group is scored

    for backend, node, members in groups.values():
        children = taxonomy.children(node.id)
        keys = list(dict.fromkeys(beams[i].key for i in members))
        probs = _score(backend, [texts[k] for k in keys], node, children, settings)
        row = {k: r for r, k in enumerate(keys)}
        for i in members:
            beam = beams[i]
            out[i] = _expand(
                beam, node, children, probs[row[beam.key]], backend.name, width, settings
            )
    return [b for expanded in out for b in expanded]


def _expand(
    beam: PartialPath,
    node: TaxonomyNode,
    children: list[TaxonomyNode],
    probs: np.ndarray,
    backend_name: str,
    width: int,
    settings: EngineSettings,
) -> list[PartialPath]:
    """Apply one scored decision to a beam: stop it, or replace it by its chosen children.

    A single-label decision keeps at most ``width`` children (Greedy: 1); more could not
    survive pruning anyway. A multi-label decision fans out to at most ``max_fanout``.
    """
    threshold = node.threshold if node.threshold is not None else settings.threshold
    # sorted() is stable, so equal probabilities keep the taxonomy's child order.
    ranked = sorted(
        ((c.id, float(p)) for c, p in zip(children, probs, strict=True)), key=lambda cp: -cp[1]
    )
    eligible = [(c, p) for c, p in ranked if p >= threshold]
    if not eligible:
        return [beam.stop("low_confidence")]
    if not node.multi_label and len(ranked) > 1:
        if ranked[0][1] - ranked[1][1] < settings.min_margin:
            return [beam.stop("low_margin")]

    picks = eligible[: settings.max_fanout if node.multi_label else width]
    picked = {child for child, _ in picks}
    # The top children plus every child picked here, so a step always shows its own choice.
    probs_shown = {c: p for i, (c, p) in enumerate(ranked) if i < TOP_K_PROBS or c in picked}
    return [
        beam.extend(
            Step(node_id=node.id, chosen=[child], probs=probs_shown, backend=backend_name),
            p,
            fan_out=node.multi_label,
        )
        for child, p in picks
    ]


def pick_backend(router: Router, node_id: str, n_candidates: int) -> Backend | None:
    """The first routed backend that supports the node and accepts ``n_candidates``."""
    for backend in router(node_id):
        if backend.supports(node_id) and (
            backend.max_candidates is None or n_candidates <= backend.max_candidates
        ):
            return backend
    return None


def _score(
    backend: Backend,
    texts: list[str],
    node: TaxonomyNode,
    children: list[TaxonomyNode],
    settings: EngineSettings,
) -> np.ndarray:
    """Probabilities of shape ``(len(texts), len(children))``, scored in mini-batches."""
    candidates = [Candidate(c.id, candidate_text(c), tuple(c.examples)) for c in children]
    where = f"node {node.id!r}"
    return _score_candidates(
        backend, texts, candidates, settings, multi_label=node.multi_label, where=where
    )


def _score_candidates(
    backend: Backend,
    texts: list[str],
    candidates: list[Candidate],
    settings: EngineSettings,
    *,
    multi_label: bool,
    where: str,
) -> np.ndarray:
    """Probabilities of shape ``(len(texts), len(candidates))``, one ``score`` call per batch."""
    temperature = settings.temperatures.get(backend.name, backend_temperature(backend))
    parts = []
    for start in range(0, len(texts), settings.batch_size):
        batch = texts[start : start + settings.batch_size]
        scores = score_batch(backend, batch, candidates, settings.instruction, multi_label)
        expected = (len(batch), len(candidates))
        if np.shape(scores.values) != expected:
            raise BackendError(
                f"backend {backend.name!r} returned scores of shape {np.shape(scores.values)} "
                f"at {where}; expected {expected}"
            )
        parts.append(to_probs(scores, temperature=temperature, multi_label=multi_label))
    return np.concatenate(parts) if parts else np.zeros((0, len(candidates)))


def backend_temperature(backend: Backend) -> float:
    """``backend.temperature``, or 1.0 for a custom backend that does not define one."""
    return float(getattr(backend, "temperature", 1.0))


def score_batch(
    backend: Backend,
    texts: Sequence[str],
    candidates: Sequence[Candidate],
    instruction: str | None,
    multi_label: bool,
) -> Scores:
    """``backend.score``; ``multi_label=True`` is passed only for a multi-label decision, so a
    custom backend whose ``score`` has no ``multi_label`` argument works at every other node."""
    if multi_label:
        return backend.score(texts, candidates, instruction=instruction, multi_label=True)
    return backend.score(texts, candidates, instruction=instruction)
