"""Backend registry: ``kind[:model[@revision]]`` specs resolved to backend instances."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from docrooter.backends.base import Backend
from docrooter.errors import BackendError

# Built-in kinds map to ``"module:attr"`` import paths and are imported only when resolved.
_BUILTIN: dict[str, str] = {
    "keyword": "docrooter.backends.fake:KeywordBackend",
    "embedding": "docrooter.backends.embedding:EmbeddingBackend",
    "nli": "docrooter.backends.nli:NLIBackend",
    "mlm-choice": "docrooter.backends.mlm_choice:MLMChoiceBackend",
    "trained": "docrooter.train.bundle:TrainedModel.load",
    "openai": "docrooter.backends.openai:OpenAIBackend",
    "jev": "docrooter.backends.jev:JevBackend",
}
_KINDS: dict[str, str | Callable[..., Backend]] = dict(_BUILTIN)


def register_backend_kind(kind: str, factory: Callable[..., Backend]) -> None:
    """Register (or replace) a backend kind; ``factory`` gets ``(model, revision=..., **kw)``."""
    _KINDS[kind] = factory


def parse_spec(spec: str) -> tuple[str, str, str | None]:
    """Split ``kind[:model[@revision]]`` into ``(kind, model, revision)``."""
    kind, _, rest = spec.partition(":")
    model, sep, revision = rest.rpartition("@")
    if not sep:
        model, revision = rest, ""
    return kind.strip(), model, revision or None


def _load_factory(kind: str) -> Callable[..., Backend]:
    target = _KINDS[kind]
    if not isinstance(target, str):
        return target
    module_name, _, attr_path = target.partition(":")
    try:
        obj: Any = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name and module_name.startswith(exc.name):
            raise BackendError(f"backend kind {kind!r} is not available: {exc}") from exc
        raise
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def resolve_backend(spec: str | Backend, **kwargs: Any) -> Backend:
    """Return a backend for ``spec``; backend instances pass through unchanged."""
    if not isinstance(spec, str):
        return spec
    kind, model, revision = parse_spec(spec)
    if kind not in _KINDS:
        known = ", ".join(sorted(_KINDS))
        raise BackendError(f"unknown backend kind {kind!r}; known kinds: {known}")
    factory = _load_factory(kind)
    if not model:
        return factory(**kwargs)
    if revision is not None:
        kwargs["revision"] = revision
    return factory(model, **kwargs)
