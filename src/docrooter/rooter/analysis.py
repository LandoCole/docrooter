"""``Analysis``: everything one :func:`~docrooter.process` run produced (spec §7.6)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from docrooter.classify.normalize import NormalizationResult
from docrooter.classify.results import DocTypeResult
from docrooter.document.models import Document
from docrooter.embed import EmbeddingSet
from docrooter.rooter.config import RouterConfig
from docrooter.summarize import Summary

__all__ = ["Analysis"]


class Analysis(BaseModel):
    """The result of :func:`~docrooter.process`.

    ``document`` carries the elements (except under ``layout="none"``, the ``fast`` path) and
    the route log in ``meta["routes"]``. ``doc_type`` and ``sections`` are ``None`` when their
    task is ``"none"``. ``summaries`` is keyed by section id and by element id. ``embeddings``
    (one vector per element) is set only when the ``embeddings`` task is. ``config`` is the
    resolved configuration that ran (a backend or engine the caller passed in appears by its
    name), and ``timings`` holds seconds per stage (``load``, ``doc_type``, ``escalate``,
    ``sections``, ``summaries``, ``embeddings``, and ``total``) for the stages that ran.
    """

    document: Document
    doc_type: DocTypeResult | None = None
    sections: NormalizationResult | None = None
    summaries: dict[str, Summary] = Field(default_factory=dict)
    embeddings: EmbeddingSet | None = None
    config: RouterConfig
    timings: dict[str, float] = Field(default_factory=dict)

    def to_json(self, *, include_embeddings: bool = False, indent: int | None = None) -> str:
        """The analysis as JSON; the embedding vectors only with ``include_embeddings``."""
        exclude = None if include_embeddings else {"embeddings"}
        return self.model_dump_json(exclude=exclude, indent=indent)
