"""Classification results: decision steps, taxonomy paths, sections and documents."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Step(BaseModel):
    """One decision: at parent ``node_id``, ``backend`` scored the children and chose some.

    ``probs`` holds up to the five most probable children, most probable first, keyed by
    child id. A node with a single child is chosen without a backend call; its step has
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

    ``best`` is the highest-scoring path, ``alternatives`` are the other beam paths that
    competed with it, and ``selected`` holds one path per multi-label branch (just ``best``
    when no multi-label node was crossed). ``n_tokens`` weights the section in summaries.
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
