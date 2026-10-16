"""Lazy import helper for optional dependencies."""

from __future__ import annotations

import importlib
import types

from docrooter.errors import MissingExtraError

__all__ = ["MissingExtraError", "require"]


def require(module: str, extra: str) -> types.ModuleType:
    """Import ``module``, or raise ``MissingExtraError`` naming the extra that provides it."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingExtraError(
            f"{module} is required for this feature: pip install 'docrooter[{extra}]'"
        ) from exc
