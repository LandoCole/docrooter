"""Level-synchronous taxonomy traversal for many texts at once.

Every round advances all unfinished paths of all texts by one decision. The pending
``(text, node)`` decisions are grouped by ``(node, backend)`` and each group is scored with one
``backend.score`` call per mini-batch, so a taxonomy of depth d costs about d batched rounds.
The traversal is iterative: no recursion over the tree.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from docrooter.backends.base import Backend, Candidate, candidate_text, to_probs
from docrooter.classify._beams import PartialPath, TreeResult, collect, prune
from docrooter.classify.results import Step
from docrooter.classify.strategies import Beam, Greedy
from docrooter.errors import BackendError
from docrooter.taxonomy import Taxonomy, TaxonomyNode

Router = Callable[[str], list[Backend]]
"""Candidate backends for a decision at a node id, in priority order."""

AUTO_BACKEND = "(auto)"
"""``Step.backend`` for a single-child node, which is chosen with p=1.0 and no backend call."""

TOP_K_PROBS = 5
"""How many of the most probable children a ``Step`` records (chosen children are added)."""


class EngineSettings(BaseModel):
    """Decision settings shared by every node unless the node overrides them.

    ``threshold`` applies where the taxonomy node has no ``threshold`` of its own.
    ``temperatures`` maps a backend name to the temperature used instead of its own.
    """

    model_config = ConfigDict(extra="forbid")

    threshold: float = 0.35
    min_margin: float = 0.0
    max_fanout: int = Field(default=3, ge=1)
    max_depth: int = Field(default=64, ge=1)
    batch_size: int = Field(default=32, ge=1)
    instruction: str | None = None
    temperatures: dict[str, float] = Field(default_factory=dict)


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
    backend = _pick_backend(router, node_id, len(children))
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
                backends[node.id] = _pick_backend(router, node.id, len(children))
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


def _pick_backend(router: Router, node_id: str, n_candidates: int) -> Backend | None:
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
    temperature = settings.temperatures.get(backend.name, backend.temperature)
    parts = []
    for start in range(0, len(texts), settings.batch_size):
        batch = texts[start : start + settings.batch_size]
        scores = backend.score(
            batch, candidates, instruction=settings.instruction, multi_label=node.multi_label
        )
        expected = (len(batch), len(candidates))
        if np.shape(scores.values) != expected:
            raise BackendError(
                f"backend {backend.name!r} returned scores of shape {np.shape(scores.values)} "
                f"at node {node.id!r}; expected {expected}"
            )
        parts.append(to_probs(scores, temperature=temperature, multi_label=node.multi_label))
    return np.concatenate(parts) if parts else np.zeros((0, len(candidates)))
