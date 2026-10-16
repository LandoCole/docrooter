"""Classification backends: scoring protocol, probability normalisation and the registry."""

from docrooter.backends.base import (
    Backend,
    Candidate,
    Scores,
    candidate_text,
    to_probs,
)
from docrooter.backends.fake import KeywordBackend
from docrooter.backends.registry import parse_spec, register_backend_kind, resolve_backend

__all__ = [
    "Backend",
    "Candidate",
    "KeywordBackend",
    "Scores",
    "candidate_text",
    "parse_spec",
    "register_backend_kind",
    "resolve_backend",
    "to_probs",
]
