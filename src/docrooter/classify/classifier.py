"""The public ``Classifier``: backend routing, section selection and result assembly."""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from docrooter.backends import registry
from docrooter.backends.base import Backend
from docrooter.classify import engine
from docrooter.classify._beams import TreeResult
from docrooter.classify.engine import EngineSettings
from docrooter.classify.results import DocumentResult, SectionResult
from docrooter.classify.strategies import Beam, Flat, Greedy
from docrooter.document import Document, Section
from docrooter.taxonomy import Taxonomy
from docrooter.views import SectionView, estimate_tokens

DEFAULT_BACKEND = "embedding:dmis-lab/UltraBERT-Embed-150M"
DEFAULT_STRATEGY = Beam(3)

SectionSelector = str | list[int] | Callable[[Section], bool]

_UNSET = object()
_DEPTH_KEY = re.compile(r"^depth(>=|<=|==)(\d+)$")
_DEPTH_TESTS: dict[str, Callable[[int, int], bool]] = {
    ">=": lambda depth, k: depth >= k,
    "<=": lambda depth, k: depth <= k,
    "==": lambda depth, k: depth == k,
}


class Classifier:
    """Classify document sections (or plain texts) against a taxonomy.

    Backends are given as instances or ``kind[:model[@revision]]`` specs. Specs are only turned
    into backends on first use (and once per spec string), so constructing a ``Classifier``
    never loads a model.

    The backends tried for a decision at node ``N`` (depth ``d = taxonomy.node_depth(N)``), in
    order: ``overrides["node:N"]``, the node's own ``backend``, the first matching
    ``overrides["depth>=k" | "depth<=k" | "depth==k"]`` in dict order, the primary ``backend``,
    then ``fallback``. The first one that supports the node and accepts its number of
    candidates decides. With a trained primary model and no ``fallback``, the fallback is the
    model's ``as_zero_shot()``.

    ``sections`` picks what a document contributes: ``"leaves"``, ``"all"`` (every section,
    but not a blank preamble), a list of heading levels, or a predicate on ``Section``.
    """

    def __init__(
        self,
        taxonomy: Taxonomy,
        backend: str | Backend = DEFAULT_BACKEND,
        *,
        fallback: str | Backend | None = None,
        overrides: dict[str, str | Backend] | None = None,
        strategy: Greedy | Beam | Flat = DEFAULT_STRATEGY,
        view: SectionView | None = None,
        instruction: str | None = None,
        threshold: float = 0.35,
        min_margin: float = 0.0,
        max_fanout: int = 3,
        sections: SectionSelector = "leaves",
        batch_size: int = 32,
        temperatures: dict[str, float] | None = None,
    ):
        if not isinstance(strategy, (Greedy, Beam, Flat)):
            raise TypeError(f"strategy must be Greedy, Beam or Flat, got {type(strategy).__name__}")
        self.taxonomy = taxonomy
        self.backend = backend
        self.fallback = fallback
        self.overrides = dict(overrides or {})
        self.strategy = strategy
        self.view = view or SectionView()
        self.sections = _check_sections(sections)
        self.settings = EngineSettings(
            threshold=threshold,
            min_margin=min_margin,
            max_fanout=max_fanout,
            batch_size=batch_size,
            instruction=instruction,
            temperatures=dict(temperatures or {}),
        )
        self._node_overrides: dict[str, str | Backend] = {}
        self._depth_overrides: list[tuple[str, int, str | Backend]] = []
        self._parse_overrides()
        self._resolved: dict[str, Backend] = {}
        self._zero_shot: Backend | None | object = _UNSET

    # --- routing --------------------------------------------------------------------------

    def _parse_overrides(self) -> None:
        for key, value in self.overrides.items():
            if key.startswith("node:"):
                node_id = key[len("node:") :]
                if node_id != Taxonomy.ROOT_ID and node_id not in self.taxonomy:
                    raise ValueError(f"override {key!r}: no node {node_id!r} in the taxonomy")
                self._node_overrides[node_id] = value
            elif match := _DEPTH_KEY.match(key):
                self._depth_overrides.append((match.group(1), int(match.group(2)), value))
            else:
                raise ValueError(
                    f"unknown override key {key!r}; use 'node:<id>', 'depth>=k', 'depth<=k' "
                    "or 'depth==k'"
                )

    def _resolve(self, spec: str | Backend) -> Backend:
        """A backend for ``spec``, loading a spec string at most once."""
        if not isinstance(spec, str):
            return spec
        if spec not in self._resolved:
            self._resolved[spec] = registry.resolve_backend(spec)
        return self._resolved[spec]

    def _fallback(self) -> Backend | None:
        """The fallback backend: ``fallback``, else a trained primary's zero-shot version."""
        if self.fallback is not None:
            return self._resolve(self.fallback)
        if self._zero_shot is _UNSET:
            as_zero_shot = getattr(self._resolve(self.backend), "as_zero_shot", None)
            self._zero_shot = as_zero_shot() if callable(as_zero_shot) else None
        return self._zero_shot  # type: ignore[return-value]

    def router(self, node_id: str) -> list[Backend]:
        """The backends to try for a decision at ``node_id``, best match first."""
        specs: list[str | Backend] = []
        if node_id in self._node_overrides:
            specs.append(self._node_overrides[node_id])
        if (node_backend := self.taxonomy.node(node_id).backend) is not None:
            specs.append(node_backend)
        depth = self.taxonomy.node_depth(node_id)
        for op, k, spec in self._depth_overrides:
            if _DEPTH_TESTS[op](depth, k):
                specs.append(spec)
                break
        specs.append(self.backend)
        backends = [self._resolve(spec) for spec in specs]
        if (fallback := self._fallback()) is not None:
            backends.append(fallback)
        unique: dict[int, Backend] = {}
        for backend in backends:
            unique.setdefault(id(backend), backend)
        return list(unique.values())

    def decide(self, texts: Sequence[str], node_id: str) -> tuple[list[str], np.ndarray, str]:
        """Child ids, probabilities ``(len(texts), n_children)`` and backend name at a node."""
        return engine.decide(texts, node_id, self.taxonomy, self.router, self.settings)

    # --- classification -------------------------------------------------------------------

    def classify(self, doc: Document) -> DocumentResult:
        """Classify the selected sections of one document."""
        return self.classify_many([doc])[0]

    def classify_many(self, docs: Sequence[Document]) -> list[DocumentResult]:
        """Classify several documents in one traversal, so they share each round's backend calls."""
        texts: dict[str, str] = {}
        meta: dict[str, tuple[int, Section, bool]] = {}
        selected = [self._select(doc) for doc in docs]
        count = self._token_counter() if any(selected) else estimate_tokens
        for index, (doc, sections) in enumerate(zip(docs, selected, strict=True)):
            rendered = self.view.render_all(doc, [s.id for s in sections], count)
            for section in sections:
                key = f"{index}:{section.id}"
                texts[key] = rendered[section.id].text
                meta[key] = (index, section, rendered[section.id].truncated)

        results = self._run(texts)
        out = [DocumentResult(source=doc.source, sections={}) for doc in docs]
        for key, (index, section, truncated) in meta.items():
            best, alternatives, chosen = results[key]
            out[index].sections[section.id] = SectionResult(
                section_id=section.id,
                title=section.title,
                best=best,
                alternatives=alternatives,
                selected=chosen,
                truncated=truncated,
                n_tokens=estimate_tokens(section.text),
            )
        self._warn_no_backend(results.values())
        return out

    def classify_texts(self, texts: list[str]) -> list[SectionResult]:
        """Classify plain texts; the results have section ids ``"0"``, ``"1"``, ..."""
        count = self._token_counter() if texts else estimate_tokens
        keyed: dict[str, str] = {}
        truncated: dict[str, bool] = {}
        for i, text in enumerate(texts):
            keyed[str(i)], truncated[str(i)] = _truncate(text, self.view.max_tokens, count)
        results = self._run(keyed)
        out = []
        for i, text in enumerate(texts):
            best, alternatives, chosen = results[str(i)]
            out.append(
                SectionResult(
                    section_id=str(i),
                    title=_first_line(text),
                    best=best,
                    alternatives=alternatives,
                    selected=chosen,
                    truncated=truncated[str(i)],
                    n_tokens=estimate_tokens(text),
                )
            )
        self._warn_no_backend(results.values())
        return out

    # --- internals ------------------------------------------------------------------------

    def _run(self, texts: dict[str, str]) -> dict[str, TreeResult]:
        if not texts:
            return {}
        if isinstance(self.strategy, Flat):
            n_candidates = len(
                self.taxonomy.nodes if self.strategy.allow_internal else self.taxonomy.leaves()
            )
            backend = engine.pick_backend(self.router, Taxonomy.ROOT_ID, n_candidates)
            return engine.run_flat(texts, self.taxonomy, backend, self.strategy, self.settings)
        return engine.run_tree(texts, self.taxonomy, self.router, self.strategy, self.settings)

    def _token_counter(self) -> Callable[[str], int]:
        tokenizer = self._resolve(self.backend).tokenizer()
        return tokenizer if tokenizer is not None else estimate_tokens

    def _select(self, doc: Document) -> list[Section]:
        which = self.sections
        if which == "leaves":
            return doc.leaves()
        if which == "all":
            return [s for s in doc.walk() if s.level > 0 or s.text.strip()]
        if callable(which):
            return [s for s in doc.walk() if which(s)]
        levels = set(which)
        return [s for s in doc.walk() if s.level in levels]

    def _warn_no_backend(self, results: Any) -> None:
        nodes: dict[str, None] = {}
        count = 0
        for best, _, chosen in results:
            stopped = [p for p in (chosen or [best]) if p.stopped_reason == "no_backend"]
            if stopped:
                count += 1
                for path in stopped:
                    nodes[path.node_ids[-1] if path.node_ids else Taxonomy.ROOT_ID] = None
        if count:
            warnings.warn(
                f"{count} section(s) stopped with no_backend: no backend supports node(s) "
                f"{', '.join(map(repr, nodes))}",
                UserWarning,
                stacklevel=3,
            )


def _check_sections(sections: SectionSelector) -> SectionSelector:
    if isinstance(sections, str):
        if sections not in ("leaves", "all"):
            raise ValueError(
                f"sections must be 'leaves', 'all', a list of levels or a callable, "
                f"got {sections!r}"
            )
        return sections
    if callable(sections):
        return sections
    levels = list(sections)
    if not all(isinstance(level, int) and not isinstance(level, bool) for level in levels):
        raise ValueError(f"sections as a list must hold int levels, got {levels!r}")
    return levels


def _first_line(text: str, limit: int = 80) -> str:
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _truncate(text: str, max_tokens: int, count: Callable[[str], int]) -> tuple[str, bool]:
    """``text`` cut to the longest word prefix within ``max_tokens`` (at least one word)."""
    if count(text) <= max_tokens:
        return text, False
    words = text.split()
    low, high = 1, len(words) - 1  # the whole text does not fit
    while low < high:
        mid = (low + high + 1) // 2
        if count(" ".join(words[:mid])) <= max_tokens:
            low = mid
        else:
            high = mid - 1
    return " ".join(words[:low]), True
