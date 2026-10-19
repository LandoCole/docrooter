"""What every source module provides: a :class:`Source` that lists :class:`Candidate` PDFs.

A source only *lists* documents (with the listing requests it needs, made through the fetcher's
:class:`~benchmarks.training_corpus._net.Net`, which rate-limits, honours robots.txt and caches
listings). Downloading, checking and recording are the fetcher's job.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchmarks.training_corpus._net import Net

_UNSAFE = re.compile(r"[^a-z0-9_-]+")
MAX_ID = 80


def safe_id(prefix: str, raw: str) -> str:
    """A file-name-safe document id matching ``^[a-z0-9_-]{1,80}$``: ``<prefix>-<raw>``,
    lower-cased, other characters as ``_``. An id that would be longer is shortened and given
    a hash of the full raw value, so shortening never merges two documents."""
    tail = _UNSAFE.sub("_", raw.lower()).strip("_-")
    doc_id = f"{prefix}-{tail}"
    if len(doc_id) > MAX_ID:
        digest = hashlib.sha1(raw.encode()).hexdigest()[:10]
        doc_id = f"{doc_id[: MAX_ID - 11].rstrip('_-')}-{digest}"
    return doc_id


@dataclass(frozen=True)
class Candidate:
    """One PDF a source offers. ``id`` is file-name-safe and unique across sources;
    ``doc_type_hint`` is a docrooter doc type id for the source's own type (Form ARS ->
    ``financial.annual_report``), from which the fetcher derives the manifest's ``family_hint``;
    ``extra`` holds source fields kept in the manifest (arXiv authors, an EDGAR accession).
    ``archive`` and ``archive_member`` name a PDF inside a cached dataset archive (CUAD), which
    the fetcher reads instead of downloading ``url``; neither goes into the manifest."""

    id: str
    url: str
    source: str
    licence: str
    doc_type_hint: str
    title: str = ""
    publisher: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    archive: Path | None = None
    archive_member: str | None = None


@dataclass(frozen=True)
class Source:
    """A corpus source. ``target`` and ``cap`` are document counts at the 1,100-document scale
    (spec §5): the fetcher scales both to ``--target``, fills ``target`` first and makes up
    shortfalls elsewhere up to ``cap``. ``probe_url`` is a URL on the host the PDFs come from;
    the fetcher checks it against robots.txt (one request per host) before listing anything.
    A source whose PDFs come from many hosts has none: each host is checked per download."""

    name: str
    target: int
    cap: int
    probe_url: str | None
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
