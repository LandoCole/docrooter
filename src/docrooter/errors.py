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


class UnknownTaxonomyError(TaxonomyError, KeyError):
    """Raised for a built-in taxonomy name that does not exist; the message lists the valid
    names. It is a ``KeyError`` too, so ``except KeyError`` keeps working."""

    def __str__(self) -> str:  # KeyError would show the message quoted, as a repr
        return str(self.args[0]) if self.args else ""


class FieldError(DocrooterError, ValueError):
    """Raised for a field set that cannot be loaded or does not fit the taxonomies. Each
    problem found is listed in ``problems``. It is a ``ValueError`` too."""

    def __init__(self, message: str, problems: list[str] | None = None) -> None:
        super().__init__(message)
        self.problems: list[str] = list(problems) if problems is not None else []


class BackendError(DocrooterError):
    """Raised when a classification or extraction backend fails."""


class DatasetError(DocrooterError):
    """Raised when a dataset is missing or malformed."""


class MissingExtraError(DocrooterError, ImportError):
    """Raised when an optional dependency is required but not installed.

    It is an ``ImportError`` too, so ``except ImportError`` keeps working.
    """


class TeacherNotAllowedError(DocrooterError, ValueError):
    """Raised for a teacher model that may not label training data: TypeSafe's Jev, whose
    output may never be used to train, distil or calibrate another model (TypeSafe MCA
    §2.3(b)). It is a ``ValueError`` too, like other bad teacher specs."""
