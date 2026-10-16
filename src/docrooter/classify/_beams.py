"""Traversal state: partial paths, beam pruning, and turning finished paths into results."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from docrooter.classify.results import PathResult, Step
from docrooter.taxonomy import Taxonomy

TreeResult = tuple[PathResult | None, list[PathResult], list[PathResult]]
"""``(best, alternatives, selected)`` for one text."""


@dataclass(frozen=True)
class PartialPath:
    """One partial path of one text.

    ``branch`` holds one ``(parent_id, child_id)`` entry per multi-label fan-out along the
    path; beams are pruned only against beams of the same text in the same branch.
    """

    key: str
    node_ids: tuple[str, ...] = ()
    logp: float = 0.0
    steps: tuple[Step, ...] = ()
    branch: tuple[tuple[str, str], ...] = ()
    done: bool = False
    reason: str | None = None

    @property
    def node_id(self) -> str:
        return self.node_ids[-1] if self.node_ids else Taxonomy.ROOT_ID

    def extend(self, step: Step, p: float, *, fan_out: bool) -> PartialPath:
        """Append ``step``'s child; ``fan_out`` marks a multi-label choice (a new branch)."""
        (child,) = step.chosen
        return replace(
            self,
            node_ids=(*self.node_ids, child),
            logp=self.logp + (math.log(p) if p > 0 else -math.inf),
            steps=(*self.steps, step),
            branch=(*self.branch, (step.node_id, child)) if fan_out else self.branch,
        )

    def stop(self, reason: str | None) -> PartialPath:
        return replace(self, done=True, reason=reason)


def prune(beams: list[PartialPath], width: int) -> list[PartialPath]:
    """Keep the ``width`` most probable beams per (text, branch); ties keep the earlier beam."""
    pools: dict[tuple[str, tuple[tuple[str, str], ...]], list[PartialPath]] = {}
    for beam in beams:
        pools.setdefault((beam.key, beam.branch), []).append(beam)
    return [b for pool in pools.values() for b in sorted(pool, key=_by_logp)[:width]]


def collect(
    beams: list[PartialPath], keys: list[str], taxonomy: Taxonomy, width: int
) -> dict[str, TreeResult]:
    """Turn finished beams into ``(best, alternatives, selected)`` per text."""
    by_key: dict[str, list[PartialPath]] = {key: [] for key in keys}
    for beam in beams:
        by_key[beam.key].append(beam)

    results: dict[str, TreeResult] = {}
    for key in keys:
        ranked = sorted(by_key[key], key=_by_logp)  # stable: ties keep the earlier beam
        chosen = _co_selected(ranked, taxonomy)
        chosen_ids = {id(b) for b in chosen}
        rivals = [b for b in ranked if id(b) not in chosen_ids][:width]
        selected = [_path_result(b, taxonomy) for b in chosen]
        results[key] = (selected[0], [_path_result(b, taxonomy) for b in rivals], selected)
    return results


def _co_selected(ranked: list[PartialPath], taxonomy: Taxonomy) -> list[PartialPath]:
    """The best beam plus the best beam of each other branch of its first multi-label fan-out.

    ``ranked`` is one text's finished beams, most probable first. Beams that share the best
    beam's trunk (the path down to that fan-out's parent node) and fanned out there first are
    co-selected labels; every other beam is a rival hypothesis.
    """
    best = ranked[0]
    if not best.branch:
        return [best]
    parent = best.branch[0][0]
    trunk = best.node_ids[: taxonomy.node_depth(parent)]
    heads: dict[tuple[tuple[str, str], ...], PartialPath] = {}
    for beam in ranked:
        if beam.branch and beam.branch[0][0] == parent and beam.node_ids[: len(trunk)] == trunk:
            heads.setdefault(beam.branch, beam)
    return list(heads.values())


def _by_logp(beam: PartialPath) -> float:
    return -beam.logp


def _path_result(beam: PartialPath, taxonomy: Taxonomy) -> PathResult:
    return PathResult(
        node_ids=list(beam.node_ids),
        labels=[taxonomy.node(n).label for n in beam.node_ids],
        score=math.exp(beam.logp),
        steps=list(beam.steps),
        stopped_reason=beam.reason,
    )
