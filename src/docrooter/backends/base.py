"""Backend protocol, score containers and probability normalisation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from docrooter.taxonomy import Taxonomy, TaxonomyNode

ScoreKind = Literal["softmax", "independent", "similarity"]

_PROB_FLOOR = 1e-12


class Candidate(NamedTuple):
    """One label a backend can choose: its display text and optional example snippets."""

    node_id: str
    text: str
    examples: tuple[str, ...] = ()


class Scores(NamedTuple):
    """Raw backend output, shape ``(n_texts, n_candidates)``, tagged with how to read it.

    ``softmax`` rows are probabilities over the candidates, ``independent`` entries are
    per-candidate probabilities, ``similarity`` entries are unbounded scores.
    """

    values: np.ndarray
    kind: ScoreKind


@runtime_checkable
class Backend(Protocol):
    """Scores texts against candidate labels.

    The classifier and ``calibrate`` treat three members as optional, so a custom backend may
    leave them out: ``temperature`` (1.0 when absent), the ``multi_label`` argument of
    ``score``, which is passed (as ``True``) only for decisions at multi-label nodes, and
    ``close()``, which releases the model and caches (see :func:`close_backend`). A closed
    backend reloads on its next use.
    """

    name: str
    max_candidates: int | None
    temperature: float

    def score(
        self,
        texts: Sequence[str],
        candidates: Sequence[Candidate],
        *,
        instruction: str | None = None,
        multi_label: bool = False,
    ) -> Scores: ...

    def supports(self, node_id: str) -> bool: ...

    def tokenizer(self) -> Callable[[str], int] | None: ...


@dataclass(frozen=True)
class TokenUsage:
    """Tokens spent on LLM calls: prompt and completion tokens over ``calls`` requests.

    ``exact`` is true only when every count came from the API's ``usage``; any estimated part
    makes the whole estimated. Usages add with ``+``.
    """

    prompt: int = 0
    completion: int = 0
    calls: int = 0
    exact: bool = True

    def __add__(self, other: TokenUsage) -> TokenUsage:
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            self.prompt + other.prompt,
            self.completion + other.completion,
            self.calls + other.calls,
            self.exact and other.exact,
        )


@runtime_checkable
class ReportsUsage(Protocol):
    """A backend that counts its LLM tokens. ``usage()`` is cumulative over the backend's life."""

    def usage(self) -> TokenUsage: ...


def usage_of(backend: object) -> TokenUsage | None:
    """The backend's cumulative :class:`TokenUsage`, or ``None`` when it reports none.

    Works for any backend: those without a ``usage()`` method (embedding, NLI, keyword, ...)
    return ``None``, and the caller decides whether to estimate.
    """
    usage = getattr(backend, "usage", None)
    if not callable(usage):
        return None
    result = usage()
    return result if isinstance(result, TokenUsage) else None


def close_backend(backend: object) -> None:
    """Call ``backend.close()`` if the backend has one (``close`` is optional)."""
    close = getattr(backend, "close", None)
    if callable(close):
        close()


def _row_softmax(x: np.ndarray) -> np.ndarray:
    if x.shape[-1] == 0:
        return x
    shifted = x - x.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def to_probs(scores: Scores, *, temperature: float, multi_label: bool) -> np.ndarray:
    """Turn raw scores into probabilities.

    ``similarity``: row softmax of ``values / temperature``. ``softmax``: returned as is when
    ``temperature == 1``, else re-tempered through log space. ``independent``: returned as is
    for ``multi_label``, otherwise rows are renormalised to sum to 1.
    """
    values = np.asarray(scores.values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"scores must be 2-D (texts x candidates), got shape {values.shape}")
    if scores.kind == "similarity":
        return _row_softmax(values / temperature)
    if scores.kind == "softmax":
        if temperature == 1:
            return values
        return _row_softmax(np.log(np.clip(values, _PROB_FLOOR, None)) / temperature)
    if scores.kind == "independent":
        if multi_label or values.shape[-1] == 0:
            return values
        totals = values.sum(axis=-1, keepdims=True)
        uniform = np.full_like(values, 1.0 / values.shape[-1])
        return np.where(totals > 0, values / np.where(totals > 0, totals, 1.0), uniform)
    raise ValueError(f"unknown score kind {scores.kind!r}")


def candidate_text(
    node: TaxonomyNode, *, taxonomy: Taxonomy | None = None, full_path: bool = False
) -> str:
    """Text a backend sees for a node: ``label`` or ``label: description``.

    With ``full_path`` (and a ``taxonomy``) the labels from the top level down are joined by
    ``" > "`` and the node's own description is appended.
    """
    if full_path and taxonomy is not None:
        label = " > ".join(n.label for n in taxonomy.path(node.id)) or node.label
    else:
        label = node.label
    return f"{label}: {node.description}" if node.description else label
