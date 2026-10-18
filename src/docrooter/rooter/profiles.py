"""The three profiles of the rooter (spec §7.2): the engine of every task, by profile name.

``fast`` is today's pipeline (no elements, no OCR, no summaries); ``balanced`` adds elements
with the heuristic layout rules, OCR when an engine is installed and centroid summaries;
``accurate`` swaps the rules for the heron layout model (extra ``layout``) and escalates
low-confidence labels to Jev when ``TYPESAFE_API_KEY`` is set.
"""

from __future__ import annotations

import os
from typing import Any, Final

from docrooter.classify.classifier import DEFAULT_BACKEND

__all__ = ["DEFAULT_PROFILE", "JEV_ESCALATION", "PROFILES", "profile_defaults"]

DEFAULT_PROFILE: Final = "balanced"
# docrooter.backends.jev's API_KEY_ENV and DEFAULT_MODEL, repeated so that importing docrooter
# does not import the backend (and asyncio); a test keeps them equal.
API_KEY_ENV = "TYPESAFE_API_KEY"
JEV_ESCALATION = "jev:jev-1.13.0"
"""``accurate``'s escalation backend, used only when ``TYPESAFE_API_KEY`` is set."""

PROFILES: dict[str, dict[str, Any]] = {
    "fast": {
        "text": "auto",
        "ocr": "none",
        "layout": "none",
        "doc_type": DEFAULT_BACKEND,
        "sections": DEFAULT_BACKEND,
        "summaries": "none",
        "embeddings": "none",
        "escalate": "none",
        "escalate_below": 0.5,
    },
    "balanced": {
        "text": "auto",
        "ocr": "auto",
        "layout": "heuristic",
        "doc_type": DEFAULT_BACKEND,
        "sections": DEFAULT_BACKEND,
        "summaries": "centroid",
        "embeddings": "none",
        "escalate": "none",
        "escalate_below": 0.5,
    },
    "accurate": {
        "text": "auto",
        "ocr": "auto",
        "layout": "heron",
        "doc_type": DEFAULT_BACKEND,
        "sections": DEFAULT_BACKEND,
        "summaries": "centroid",
        "embeddings": "none",
        "escalate": JEV_ESCALATION,  # only with TYPESAFE_API_KEY set; see profile_defaults
        "escalate_below": 0.5,
    },
}
"""Each profile's task engines, as spec §7.2 writes them. ``accurate``'s ``escalate`` holds
when ``TYPESAFE_API_KEY`` is set; :func:`profile_defaults` applies that condition."""


def profile_defaults(name: str) -> dict[str, Any]:
    """A copy of profile ``name``'s settings as they apply now: ``accurate`` escalates to Jev
    only when the ``TYPESAFE_API_KEY`` environment variable is set (non-blank), else
    ``escalate`` is ``"none"``. Raises ``ValueError`` for an unknown profile."""
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; use one of {', '.join(PROFILES)}")
    values = dict(PROFILES[name])
    if values["escalate"] == JEV_ESCALATION and not os.environ.get(API_KEY_ENV, "").strip():
        values["escalate"] = "none"
    return values
