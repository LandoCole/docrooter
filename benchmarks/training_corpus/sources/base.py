"""What every source module provides: a :class:`Source` that lists :class:`Candidate` PDFs.

A source only *lists* documents (with the listing requests it needs, made through the fetcher's
:class:`~benchmarks.training_corpus._net.Net`, which rate-limits, honours robots.txt and caches
listings). Downloading, checking and recording are the fetcher's job.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchmarks.training_corpus._net import Net

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_id(prefix: str, raw: str) -> str:
    """A file-name-safe document id: ``<prefix>-<raw>`` with other characters as ``_``."""
    tail = _UNSAFE.sub("_", raw).strip("._-")
    return f"{prefix}-{tail}"[:128]


@dataclass(frozen=True)
class Candidate:
    """One PDF a source offers. ``id`` is file-name-safe and unique across sources;
    ``doc_type_hint`` is a docrooter doc type id for the source's own type (Form ARS ->
    ``financial.annual_report``); ``family_hint`` overrides the element family that type maps to;
    ``extra`` holds source fields kept in the manifest (arXiv authors, an EDGAR accession)."""

    id: str
    url: str
    source: str
    licence: str
    doc_type_hint: str
    title: str = ""
    publisher: str = ""
    family_hint: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Source:
    """A corpus source. ``target`` and ``cap`` are document counts at the 1,100-document scale
    (spec §5): the fetcher scales both to ``--target``, fills ``target`` first and makes up
    shortfalls elsewhere up to ``cap``. ``probe_url`` is a URL on the host the PDFs come from;
    the fetcher checks it against robots.txt (one request per host) before listing anything."""

    name: str
    target: int
    cap: int
    probe_url: str
    candidates: Callable[[Net], Iterable[Candidate]]
    note: str = ""


def env_key(*names: str) -> str | None:
    """The first non-empty API key among environment variables ``names``. Keys come from the
    process environment only; ``.env`` is never read."""
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def round_robin(*iterables: Iterable[Candidate]) -> Iterator[Candidate]:
    """Interleave lazily: one item from each iterable in turn until all are exhausted."""
    iterators = [iter(it) for it in iterables]
    while iterators:
        alive = []
        for iterator in iterators:
            try:
                yield next(iterator)
            except StopIteration:
                continue
            alive.append(iterator)
        iterators = alive
