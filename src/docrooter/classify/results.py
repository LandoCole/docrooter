"""Classification results: decision steps, taxonomy paths, sections and documents."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from docrooter._extras import require


class Step(BaseModel):
    """One decision: at parent ``node_id``, ``backend`` scored the children and chose some.

    ``probs`` holds the five most probable children plus every child chosen at this decision,
    most probable first, keyed by child id, so ``probs[chosen[0]]`` always exists. A node
    with a single child is chosen without a backend call; its step has
    ``probs == {child: 1.0}`` and ``backend == "(auto)"``.
    """

    model_config = ConfigDict(extra="forbid")

    node_id: str
    chosen: list[str]
    probs: dict[str, float]
    backend: str


class PathResult(BaseModel):
    """A path from the top level down; ``steps[i]`` is the decision that chose ``node_ids[i]``.

    ``score`` is the product of the chosen children's probabilities (1.0 for an empty path).
    ``stopped_reason`` is ``None`` when the path reached a leaf, otherwise one of
    ``"low_confidence"``, ``"low_margin"``, ``"max_depth"`` or ``"no_backend"``.
    """

    model_config = ConfigDict(extra="forbid")

    node_ids: list[str]
    labels: list[str] = Field(default_factory=list)
    score: float = 1.0
    steps: list[Step] = Field(default_factory=list)
    stopped_reason: str | None = None


class SectionResult(BaseModel):
    """The classification of one section.

    ``best`` is the highest-scoring path. ``selected`` holds the paths assigned together with
    it at its first multi-label node, one per branch, ``best`` first (just ``[best]`` when no
    multi-label node was crossed). ``alternatives`` are the rival paths that competed with
    them. ``n_tokens`` weights the section in summaries.
    """

    model_config = ConfigDict(extra="forbid")

    section_id: str
    title: str
    best: PathResult | None
    alternatives: list[PathResult] = Field(default_factory=list)
    selected: list[PathResult] = Field(default_factory=list)
    truncated: bool = False
    n_tokens: int


class DocumentResult(BaseModel):
    """Section results for one document, keyed by section id."""

    model_config = ConfigDict(extra="forbid")

    source: str
    sections: dict[str, SectionResult]

    def summary(self, depth: int = 1) -> dict[str, float]:
        """Share of the document (by tokens) in each label at 1-based ``depth``.

        Every section whose best path reaches ``depth`` adds its ``n_tokens`` to the label found
        there; shares are normalised to sum to 1. Empty when no section contributes any weight.
        """
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        weights: dict[str, float] = {}
        for result in self.sections.values():
            if result.best is not None and len(result.best.labels) >= depth:
                label = result.best.labels[depth - 1]
                weights[label] = weights.get(label, 0.0) + result.n_tokens
        total = sum(weights.values())
        if total <= 0:
            return {}
        return {label: weight / total for label, weight in weights.items()}

    def to_json(self, indent: int | None = 2) -> str:
        """The result as JSON; ``DocumentResult.model_validate_json`` reads it back."""
        return self.model_dump_json(indent=indent)

    def to_dataframe(self) -> Any:
        """One row per section: section_id, title, path (labels joined by ``" > "``), score and
        stopped_reason of the best path. Needs pandas (``pip install 'docrooter[bench]'``)."""
        pd = require("pandas", "bench")
        rows = [
            {
                "section_id": r.section_id,
                "title": r.title,
                "path": " > ".join(r.best.labels) if r.best else "",
                "score": r.best.score if r.best else None,
                "stopped_reason": r.best.stopped_reason if r.best else None,
            }
            for r in self.sections.values()
        ]
        return pd.DataFrame(
            rows, columns=["section_id", "title", "path", "score", "stopped_reason"]
        )
