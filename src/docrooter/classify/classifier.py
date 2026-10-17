"""The public ``Classifier``: backend routing, section selection and result assembly."""

from __future__ import annotations

import functools
import json
import re
import warnings
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from docrooter import events
from docrooter.backends import registry
from docrooter.backends.base import Backend, close_backend
from docrooter.classify import engine
from docrooter.classify._beams import TreeResult
from docrooter.classify.engine import EngineSettings
from docrooter.classify.results import DocTypeResult, DocumentResult, PathResult, SectionResult
from docrooter.classify.strategies import Beam, Flat, Greedy
from docrooter.doc_view import DocumentView
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

    :meth:`close` (also on leaving a ``with`` block) releases the models this classifier
    loaded from spec strings; backend instances you passed in are yours to close.
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
        if isinstance(strategy, Flat) and _is_trained(backend):
            raise ValueError(
                "Flat strategy needs a zero-shot backend; trained backends score per-node heads"
            )
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
        self._taxonomy_hash: str | None = None
        self.calibration_report: dict[str, dict[str, Any]] | None = None
        for spec in (backend, fallback, *self.overrides.values()):
            if spec is not None and not isinstance(spec, str):
                self._check_taxonomy_hash(spec, stacklevel=4)

    # --- lifecycle ------------------------------------------------------------------------

    def close(self) -> None:
        """Close every backend this classifier loaded from a spec string, and the zero-shot
        fallback of such a backend. Backend instances passed in are left open. The classifier
        stays usable: specs are loaded again on next use."""
        resolved, self._resolved = list(self._resolved.values()), {}
        zero_shot, self._zero_shot = self._zero_shot, _UNSET
        if isinstance(self.backend, str) and zero_shot is not None and zero_shot is not _UNSET:
            resolved.append(cast("Backend", zero_shot))  # a trained spec's body: ours too
        seen: set[int] = set()
        for backend in resolved:
            if id(backend) not in seen:
                seen.add(id(backend))
                close_backend(backend)

    def __enter__(self) -> Classifier:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

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
            self._check_taxonomy_hash(self._resolved[spec], stacklevel=5)
        return self._resolved[spec]

    def _check_taxonomy_hash(self, backend: Backend, *, stacklevel: int) -> None:
        """Warn when a trained backend was trained on a different taxonomy."""
        trained_on = getattr(backend, "taxonomy_hash", None)
        if trained_on is None:
            return
        if self._taxonomy_hash is None:
            self._taxonomy_hash = self.taxonomy.content_hash()
        if trained_on != self._taxonomy_hash:
            warnings.warn(
                f"{backend.name} was trained on a different taxonomy (content hash "
                f"{trained_on[:12]}, this one is {self._taxonomy_hash[:12]}); its heads may "
                "not match the nodes they are used for",
                UserWarning,
                stacklevel=stacklevel,
            )

    def _fallback(self) -> Backend | None:
        """The fallback backend: ``fallback``, else a trained primary's zero-shot version."""
        if self.fallback is not None:
            return self._resolve(self.fallback)
        if self._zero_shot is _UNSET:
            as_zero_shot = getattr(self._resolve(self.backend), "as_zero_shot", None)
            self._zero_shot = as_zero_shot() if callable(as_zero_shot) else None
        return self._zero_shot  # type: ignore[return-value]

    def router(self, node_id: str) -> Sequence[Backend]:
        """The backends to try for a decision at ``node_id``, best match first.

        The result is lazy: a backend (the fallback in particular) is only resolved when the
        caller reaches it, so a fallback model is never loaded while earlier ones suffice.
        """
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
        thunks: list[Callable[[], Backend | None]] = [
            functools.partial(self._resolve, spec) for spec in specs
        ]
        thunks.append(self._fallback)
        return _LazyBackends(thunks)

    def decide(self, texts: Sequence[str], node_id: str) -> tuple[list[str], np.ndarray, str]:
        """Child ids, probabilities ``(len(texts), n_children)`` and backend name at a node."""
        return engine.decide(texts, node_id, self.taxonomy, self.router, self.settings)

    # --- configuration --------------------------------------------------------------------

    def save_config(self, path: str | Path) -> None:
        """Write the configuration as JSON; :meth:`from_config` rebuilds the classifier from it.

        Backends are saved as spec strings; a ``TrainedModel`` as ``"trained:<bundle path>"``, so
        it must have been saved (or loaded) from a bundle. Any other backend object, and a
        callable ``sections``, cannot be written and raise ``ValueError``.
        """
        if callable(self.sections):
            raise ValueError("cannot save a classifier whose sections is a callable")
        settings = self.settings
        config = {
            "backend": _backend_spec(self.backend, "backend"),
            "fallback": None if self.fallback is None else _backend_spec(self.fallback, "fallback"),
            "overrides": {
                key: _backend_spec(value, f"override {key!r}")
                for key, value in self.overrides.items()
            },
            "strategy": _strategy_to_dict(self.strategy),
            "view": self.view.model_dump(),
            "settings": {
                "threshold": settings.threshold,
                "min_margin": settings.min_margin,
                "max_fanout": settings.max_fanout,
                "batch_size": settings.batch_size,
                "instruction": settings.instruction,
                "temperatures": dict(settings.temperatures),
            },
            "sections": self.sections,
            "taxonomy_hash": self.taxonomy.content_hash(),
        }
        Path(path).write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", "utf-8")

    @classmethod
    def from_config(cls, path: str | Path, taxonomy: Taxonomy) -> Classifier:
        """A classifier from a :meth:`save_config` file, for ``taxonomy``.

        Warns when ``taxonomy`` differs from the one the configuration was saved with.
        """
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        saved_hash = config.get("taxonomy_hash")
        if saved_hash is not None and saved_hash != taxonomy.content_hash():
            warnings.warn(
                f"{path} was saved for a different taxonomy (content hash {saved_hash[:12]}, "
                f"this one is {taxonomy.content_hash()[:12]})",
                UserWarning,
                stacklevel=2,
            )
        settings = config.get("settings", {})
        return cls(
            taxonomy,
            config["backend"],
            fallback=config.get("fallback"),
            overrides=config.get("overrides") or None,
            strategy=_strategy_from_dict(config.get("strategy")),
            view=SectionView(**config["view"]) if config.get("view") else None,
            sections=config.get("sections", "leaves"),
            **settings,
        )

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

    def classify_documents(
        self,
        docs: Sequence[Document],
        *,
        view: DocumentView | None = None,
        mode: str = "view",
        strategy: Greedy | Beam | Flat | None = None,
        threshold: float | None = None,
        instruction: str | None = None,
        on_event: events.EventSink | None = None,
    ) -> list[DocTypeResult]:
        """One document-type result per document, against this classifier's taxonomy.

        Document-type defaults, independent of the section settings this classifier was built
        with: ``strategy=None`` is ``Flat()``, ``instruction=None`` is ``"What type of document
        is this?"`` and ``threshold=None`` is this classifier's ``threshold``. The root's own
        ``threshold``, when set, wins as at every node. :func:`~docrooter.classify_document`
        is this method with ``threshold=0.0``.

        ``mode="view"`` classifies each document's rendered ``view`` (default
        ``DocumentView()``) with ``strategy``. ``mode="sections"`` scores every leaf section's
        title and lead with ``Flat`` and takes the token-weighted mean of the leaf
        distributions; the argmax is ``best``. ``mode="hybrid"`` mixes the two as
        ``0.6 * view + 0.4 * sections``. Every mode then decides the same way: the argmax
        node, or an empty ``"low_confidence"`` path when its probability (``confidence``) is
        below the threshold. All documents share one traversal per kind of text. A document
        with nothing to show (no text beyond the file name, or a PDF without a text layer) gets
        ``best.stopped_reason == "empty_text"`` and is never sent to a backend.

        ``leaf_probs`` under ``Flat`` is the leaf distribution. Under ``Greedy`` or ``Beam`` it
        holds the end node of the best path and of each alternative with its path score (the
        product of its step probabilities), scaled to sum to at most 1; a path that stopped
        early ends at an internal node. When no path got past the root, ``leaf_probs`` holds
        the root decision's probabilities over the top-level nodes and ``confidence`` is their
        maximum, so it is never 0 just because the traversal stopped.

        ``on_event`` receives pipeline :class:`~docrooter.events.Event` objects (``view``,
        ``model_load``, ``classify``, then ``done``); see :func:`~docrooter.classify_document`.
        """
        from docrooter.classify.documents import classify_documents

        with events.pipeline_run(on_event):
            return classify_documents(
                self,
                docs,
                view or DocumentView(),
                mode,
                strategy=strategy,
                threshold=threshold,
                instruction=instruction,
            )

    def prepare_texts(self, texts: Sequence[str]) -> list[str]:
        """The texts as the backends see them: cut to ``view.max_tokens`` (blank ones unchanged).

        :meth:`classify_texts` scores exactly these; use them with :meth:`decide` to get the
        same decisions.
        """
        count = self._token_counter() if any(t.strip() for t in texts) else estimate_tokens
        return [_truncate(t, self.view.max_tokens, count)[0] if t.strip() else t for t in texts]

    def classify_texts(self, texts: list[str]) -> list[SectionResult]:
        """Classify plain texts; the results have section ids ``"0"``, ``"1"``, ..."""
        prepared = self.prepare_texts(texts)
        keyed: dict[str, str] = {}
        truncated: dict[str, bool] = {}
        for i, (text, cut) in enumerate(zip(texts, prepared, strict=True)):
            if text.strip():  # blank text is never sent to a backend
                keyed[str(i)], truncated[str(i)] = cut, cut != text
        results = self._run(keyed)
        out = []
        for i, text in enumerate(texts):
            if str(i) in results:
                best, alternatives, chosen = results[str(i)]
            else:
                best, alternatives, chosen = (
                    PathResult(node_ids=[], stopped_reason="empty_text"),
                    [],
                    [],
                )
            out.append(
                SectionResult(
                    section_id=str(i),
                    title=_first_line(text),
                    best=best,
                    alternatives=alternatives,
                    selected=chosen,
                    truncated=truncated.get(str(i), False),
                    n_tokens=estimate_tokens(text),
                )
            )
        self._warn_no_backend(results.values())
        return out

    # --- internals ------------------------------------------------------------------------

    def _run(
        self,
        texts: dict[str, str],
        strategy: Greedy | Beam | Flat | None = None,
        settings: EngineSettings | None = None,
    ) -> dict[str, TreeResult]:
        """Traverse for every text; ``strategy`` and ``settings`` default to the classifier's."""
        if not texts:
            return {}
        strategy = strategy if strategy is not None else self.strategy
        settings = settings if settings is not None else self.settings
        if isinstance(strategy, Flat):
            n_candidates = len(
                self.taxonomy.nodes if strategy.allow_internal else self.taxonomy.leaves()
            )
            backend = engine.pick_backend(self.router, Taxonomy.ROOT_ID, n_candidates)
            return engine.run_flat(texts, self.taxonomy, backend, strategy, settings)
        return engine.run_tree(texts, self.taxonomy, self.router, strategy, settings)

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
            stopped = [
                p for p in (chosen or [best]) if p is not None and p.stopped_reason == "no_backend"
            ]
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


class _LazyBackends(Sequence[Backend]):
    """Backends produced one at a time from thunks, de-duplicated by identity, skipping ``None``.

    Iterating or indexing resolves only as many thunks as needed; ``len`` resolves them all.
    """

    def __init__(self, thunks: list[Callable[[], Backend | None]]):
        self._pending = iter(thunks)
        self._items: list[Backend] = []
        self._seen: set[int] = set()

    def _advance(self) -> bool:
        for thunk in self._pending:
            backend = thunk()
            if backend is not None and id(backend) not in self._seen:
                self._seen.add(id(backend))
                self._items.append(backend)
                return True
        return False

    def __iter__(self) -> Iterator[Backend]:
        i = 0
        while i < len(self._items) or self._advance():
            yield self._items[i]
            i += 1

    def __getitem__(self, index):
        if isinstance(index, int) and index >= 0:
            while len(self._items) <= index and self._advance():
                pass
        else:
            while self._advance():
                pass
        return self._items[index]

    def __len__(self) -> int:
        while self._advance():
            pass
        return len(self._items)


def _is_trained(backend: str | Backend) -> bool:
    """Whether ``backend`` is a ``TrainedModel`` or a ``"trained:"`` spec."""
    if isinstance(backend, str):
        return registry.parse_spec(backend)[0] == "trained"
    from docrooter.train.bundle import TrainedModel

    return isinstance(backend, TrainedModel)


def _backend_spec(backend: str | Backend, where: str) -> str:
    """The spec string that rebuilds ``backend``; backend objects other than saved bundles fail."""
    if isinstance(backend, str):
        return backend
    from docrooter.train.bundle import TrainedModel

    if isinstance(backend, TrainedModel):
        if backend.path is None:
            raise ValueError(
                f"the {where} is a TrainedModel that was never saved; call model.save(path) "
                "first so the configuration can point to its bundle"
            )
        return f"trained:{backend.path}"
    raise ValueError(
        f"cannot save the {where}: a backend object ({type(backend).__name__}) has no spec; "
        "use a spec string or a saved TrainedModel"
    )


def _strategy_to_dict(strategy: Greedy | Beam | Flat) -> dict[str, Any]:
    if isinstance(strategy, Beam):
        return {"type": "beam", "width": strategy.width}
    if isinstance(strategy, Flat):
        return {"type": "flat", "allow_internal": strategy.allow_internal}
    return {"type": "greedy"}


def _strategy_from_dict(spec: dict[str, Any] | None) -> Greedy | Beam | Flat:
    if spec is None:
        return DEFAULT_STRATEGY
    kind = spec.get("type")
    if kind == "greedy":
        return Greedy()
    if kind == "beam":
        return Beam(int(spec.get("width", 3)))
    if kind == "flat":
        return Flat(allow_internal=bool(spec.get("allow_internal", False)))
    raise ValueError(f"unknown strategy {spec!r} in the configuration; use greedy, beam or flat")


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
