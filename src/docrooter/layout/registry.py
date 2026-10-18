"""The layout engine registry: ``heron`` built in, user engines by name."""

from __future__ import annotations

from collections.abc import Callable

from docrooter.errors import DocrooterError
from docrooter.layout.base import LayoutEngine

__all__ = ["available_layout_engines", "get_layout_engine", "register_layout_engine"]

LayoutFactory = Callable[[], LayoutEngine]

_RESERVED = ("heuristic", "none", "auto")


def _heron() -> LayoutEngine:
    from docrooter.layout.heron import HeronLayout

    return HeronLayout()


_FACTORIES: dict[str, LayoutFactory] = {"heron": _heron}


def register_layout_engine(name: str, factory: LayoutFactory) -> None:
    """Register a layout engine under ``name`` (replacing any engine of that name).
    ``factory()`` returns a :class:`~docrooter.layout.base.LayoutEngine`."""
    if not name or name in _RESERVED:
        raise DocrooterError(f"{name!r} is not a valid layout engine name")
    _FACTORIES[name] = factory


def available_layout_engines() -> list[str]:
    """The registered engine names, built in first. A name can still fail to load when its
    package or weights are missing."""
    return list(_FACTORIES)


def get_layout_engine(name: str) -> LayoutEngine:
    """A new engine for ``name``. Raises ``DocrooterError`` for an unknown name and
    ``MissingExtraError`` when the engine's packages are not installed."""
    try:
        factory = _FACTORIES[name]
    except KeyError:
        known = ", ".join(["heuristic", *available_layout_engines()])
        raise DocrooterError(f"unknown layout engine {name!r}; use one of {known}") from None
    return factory()
