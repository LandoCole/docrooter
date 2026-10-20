"""Lazy import helper for optional dependencies."""

from __future__ import annotations

import importlib
import types

from docrooter.errors import MissingExtraError

__all__ = ["REPO_URL", "MissingExtraError", "install_hint", "require"]

REPO_URL = "git+https://github.com/LandoCole/docrooter"
"""Where docrooter installs from: it is not on PyPI."""


def install_hint(extra: str) -> str:
    """The command that installs ``extra``, in the GitHub form docrooter installs from."""
    return f'pip install "docrooter[{extra}] @ {REPO_URL}"'


def require(module: str, extra: str) -> types.ModuleType:
    """Import ``module``, or raise ``MissingExtraError`` naming the extra that provides it."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingExtraError(
            f"{module} is required for this feature: {install_hint(extra)}"
        ) from exc
