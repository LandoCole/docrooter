"""Exception hierarchy for docrooter."""

from __future__ import annotations

from typing import Any


class DocrooterError(Exception):
    """Base class for all docrooter errors."""


class TaxonomyError(DocrooterError):
    """Raised when a taxonomy is invalid. Validation problems are listed in ``issues``."""

    def __init__(self, message: str, issues: list[Any] | None = None) -> None:
        super().__init__(message)
        self.issues: list[Any] = list(issues) if issues is not None else []


class BackendError(DocrooterError):
    """Raised when a classification or extraction backend fails."""


class DatasetError(DocrooterError):
    """Raised when a dataset is missing or malformed."""
