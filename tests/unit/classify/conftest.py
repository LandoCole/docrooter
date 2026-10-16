"""Shared helpers for classification tests: spy backends and small taxonomies.

The helpers are plain importable names (``from conftest import ...``) and some are also
exposed as fixtures.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from docrooter.backends import Candidate, KeywordBackend, Scores
from docrooter.taxonomy import Taxonomy, TaxonomyNode


class CountingBackend:
    """Wraps a backend and counts ``score`` calls."""

    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name
        self.max_candidates = inner.max_candidates
        self.temperature = inner.temperature
        self.calls = 0

    def score(
        self,
        texts: Sequence[str],
        candidates: Sequence[Candidate],
        *,
        instruction: str | None = None,
        multi_label: bool = False,
    ) -> Scores:
        self.calls += 1
        return self.inner.score(texts, candidates, instruction=instruction, multi_label=multi_label)

    def supports(self, node_id: str) -> bool:
        return self.inner.supports(node_id)

    def tokenizer(self) -> Callable[[str], int] | None:
        return self.inner.tokenizer()


class RecordingBackend(CountingBackend):
    """A :class:`CountingBackend` that also keeps the arguments of the last ``score`` call."""

    def __init__(self, inner):
        super().__init__(inner)
        self.last_texts: list[str] = []
        self.last_candidates: list[Candidate] = []
        self.last_instruction: str | None = None
        self.last_multi_label: bool | None = None

    def score(self, texts, candidates, *, instruction=None, multi_label=False) -> Scores:
        self.last_texts = list(texts)
        self.last_candidates = list(candidates)
        self.last_instruction = instruction
        self.last_multi_label = multi_label
        return super().score(texts, candidates, instruction=instruction, multi_label=multi_label)


class NeverSupports:
    """A backend that supports no node; scoring it is a test failure."""

    name = "never"
    max_candidates: int | None = None
    temperature = 1.0

    def score(self, texts, candidates, *, instruction=None, multi_label=False) -> Scores:
        raise AssertionError("NeverSupports.score must not be called")

    def supports(self, node_id: str) -> bool:
        return False

    def tokenizer(self) -> Callable[[str], int] | None:
        return None


def _fin_legal_nodes() -> list[TaxonomyNode]:
    return [
        TaxonomyNode(id="financial", label="Financial"),
        TaxonomyNode(id="financial.tax", label="Tax", parent_id="financial"),
        TaxonomyNode(id="financial.audit", label="Audit", parent_id="financial"),
        TaxonomyNode(id="legal", label="Legal"),
        TaxonomyNode(id="legal.contracts", label="Contracts", parent_id="legal"),
        TaxonomyNode(id="legal.litigation", label="Litigation", parent_id="legal"),
    ]


def fin_legal_taxonomy() -> Taxonomy:
    """Financial{Tax, Audit}, Legal{Contracts, Litigation}; labels only, no descriptions."""
    return Taxonomy(_fin_legal_nodes())


def taxonomy_with_multilabel_root() -> Taxonomy:
    """The :func:`fin_legal_taxonomy` tree with a multi-label root decision."""
    return Taxonomy(_fin_legal_nodes(), root={"multi_label": True})


def taxonomy_with_multilabel_financial(*, root_multi_label: bool = False) -> Taxonomy:
    """The :func:`fin_legal_taxonomy` tree where the decision under Financial is multi-label."""
    nodes = [
        n.model_copy(update={"multi_label": True}) if n.id == "financial" else n
        for n in _fin_legal_nodes()
    ]
    return Taxonomy(nodes, root={"multi_label": True} if root_multi_label else None)


def chain_taxonomy(n: int) -> Taxonomy:
    """A single chain ``l0 > l1 > ... > l{n-1}``: every node has exactly one child."""
    return Taxonomy(
        [
            TaxonomyNode(id=f"l{i}", label=f"L{i}", parent_id=f"l{i - 1}" if i else None)
            for i in range(n)
        ]
    )


@pytest.fixture
def tax() -> Taxonomy:
    return fin_legal_taxonomy()


@pytest.fixture
def router():
    """A router that sends every decision to one shared :class:`KeywordBackend`."""
    backend = KeywordBackend()
    return lambda node_id: [backend]
