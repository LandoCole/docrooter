"""Traversal strategies: how a section walks down the taxonomy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Greedy:
    """Follow the most probable child at each step while it passes the threshold. One path."""


@dataclass(frozen=True)
class Beam:
    """Keep the ``width`` best partial paths, ranked by summed log probabilities."""

    width: int = 3

    def __post_init__(self) -> None:
        if self.width < 1:
            raise ValueError(f"Beam width must be at least 1, got {self.width}")


@dataclass(frozen=True)
class Flat:
    """Score every leaf (or every node, with ``allow_internal``) in one call, by full path."""

    allow_internal: bool = False
