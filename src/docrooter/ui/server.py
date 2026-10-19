"""The playground server: a FastAPI app that runs uploaded documents through the pipeline and
streams its events to the page as Server-Sent Events. A run's ``profile`` (``fast``,
``balanced`` or ``accurate``) picks the rooter profile; the worker keeps one set of OCR and
layout engines for it (:class:`EnginePool`) and closes them when the profile or backend changes.

One worker thread takes jobs from a :class:`queue.Queue`, so runs are processed one at a time
and memory stays bounded. Each job enters :func:`~docrooter.events.pipeline_run` inside the
worker; its sink hands every event to the event loop with ``loop.call_soon_threadsafe``. All
run state is changed on the loop thread only, so the HTTP handlers need no locks.

Endpoints: ``POST /api/runs`` (multipart ``file`` plus ``backend``, ``engine``, ``mode``,
``normalize`` and ``profile``), ``GET /api/runs``, ``GET /api/runs/{id}``,
``GET /api/runs/{id}/events`` (SSE, ending with a ``{"stage": "job", "status": "end"}`` event
the server always sends), ``GET /api/backends``, and the page itself at ``/``.

Started with ``element_tags`` (``docrooter serve --element-tags BUNDLE``), the worker loads
that tagger once, on the first run in element mode, tags every such run's content elements in
the family of its document type, and closes the tagger on shutdown; the elements then carry
their tags (``tags``) and the result names the tagger and family (``element_tags``).

The completeness dial (spec v0.6 §7): the last :data:`MAX_KEPT_ANALYSES` runs in element mode
keep their analysis, and ``POST /api/runs/{id}/select`` reads it with
:func:`docrooter.select.selection.select` for a field or a query at a completeness, returning
the :func:`~docrooter.select.selection.readout` line, the selection's numbers and the selected
element ids (never their text). ``GET /api/fields`` lists the built-in fields. ``docrooter
serve --selector`` (or a ``selector.json`` in the ``--element-tags`` bundle) supplies the
fitted selector; without one only the whole document (completeness 1.0) can be selected. A
selection that needs a model (the selector's embedder, the tagger) runs on the worker thread,
between runs, so one model is loaded at a time.
"""

from __future__ import annotations

import asyncio
import codecs
import importlib.util
import ipaddress
import json
import logging
import os
import queue
import re
import socket
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, MutableMapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from docrooter._extras import require

require("fastapi", "ui")
require("python_multipart", "ui")

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import PlainTextResponse, Response, StreamingResponse  # noqa: E402
from starlette.datastructures import Headers, UploadFile  # noqa: E402
from starlette.types import ASGIApp, Message, Receive, Scope, Send  # noqa: E402

from docrooter import events  # noqa: E402
from docrooter.backends import registry  # noqa: E402
from docrooter.backends.base import Backend  # noqa: E402
from docrooter.classify.classifier import Classifier  # noqa: E402
from docrooter.classify.documents import MODES, classify_document  # noqa: E402
from docrooter.classify.normalize import (  # noqa: E402
    DOC_TYPES,
    GENERIC_TAXONOMY,
    NormalizationResult,
    normalize_sections,
)
from docrooter.classify.results import DocTypeResult  # noqa: E402
from docrooter.document import Document  # noqa: E402
from docrooter.document.load import load  # noqa: E402
from docrooter.errors import DocrooterError  # noqa: E402
from docrooter.layout import registry as layout_registry  # noqa: E402
from docrooter.ocr import registry as ocr_registry  # noqa: E402
from docrooter.pdf.load import ENGINES  # noqa: E402
from docrooter.rooter import DEFAULT_PROFILE, PROFILES, RouterConfig, process  # noqa: E402
from docrooter.rooter.analysis import Analysis  # noqa: E402
from docrooter.rooter.process import _LazyOcr, _tag_elements  # noqa: E402
from docrooter.rooter.profiles import jev_key_set  # noqa: E402
from docrooter.summarize import Summary  # noqa: E402
from docrooter.taxonomies import builtin_taxonomy, family_for_doc_type  # noqa: E402
from docrooter.taxonomy import Taxonomy  # noqa: E402

if TYPE_CHECKING:
    import numpy as np

    from docrooter.select.fields import Field as SelectField
    from docrooter.select.selection import DocContext, Selection, Selector

__all__ = [
    "MAX_TEXT_UPLOAD_BYTES",
    "MAX_TOC_ENTRIES",
    "MAX_UPLOAD_BYTES",
    "EnginePool",
    "check_backend_spec",
    "create_app",
    "load_env_file",
]

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
"""Largest accepted upload; read on every request, so tests can patch it."""
MAX_TEXT_UPLOAD_BYTES = 1 * 1024 * 1024
"""Largest accepted ``.md`` or ``.txt`` upload. PDFs are bounded by :data:`MAX_PAGES`; text has
no pages, and its work and memory grow with its headings: a heading on every other line peaks
at about 270 MB per MB of file, and about 1.6 GB per MB with section labels on (measured on a
worst-case 1 MiB file). 1 MB is close to 200,000 words, beyond a real filing as text, and
keeps a run well inside the default 8 GB memory budget."""
MAX_TOC_ENTRIES = 5000
"""TOC (and section-label) entries a run keeps and returns; the rest are dropped and the
result says ``truncated``. A long filing has a few hundred; 5,000 entries are about 0.6 MB of
JSON per run (1.2 MB with section labels), so :data:`HISTORY` runs stay small and the page
builds a bounded tree."""
MAX_ELEMENT_TEXT_CHARS = 200
"""Characters of an element's text the page gets; ``text_chars`` says how long it was."""
MAX_SUMMARY_SENTENCE_CHARS = 300
"""Characters kept of each key sentence in the result (the sentence splitter can return a very
long one from text without full stops)."""
MAX_TAG_PROBS = 5
"""Probabilities sent per element and facet: the labels' and the next highest, up to five."""
MAX_KEYPHRASES = 20
MAX_KEYPHRASE_CHARS = 80
"""Key phrases kept per summary, and characters kept of each (a text of long tokens gives long
phrases)."""
MAX_PAGES = 2000
HISTORY = 20
"""Runs kept in memory; the oldest is dropped when a new one arrives."""
MAX_KEPT_ANALYSES = 3
"""Runs whose analysis is kept for the dial (a document and its tags are far larger than a run's
JSON); the oldest is dropped first, and selecting on it says to run the file again."""
MAX_QUERY_CHARS = 500
"""Longest free-text need the select endpoint accepts."""
MAX_SELECT_BODY_BYTES = 16 * 1024
"""Largest select request body; a request is a need, a number and a budget."""
KEEPALIVE_SECONDS = 15.0
MAX_PENDING = 8
"""Jobs that may wait behind the running one; more get 503."""
STOP_TIMEOUT_SECONDS = 10.0
STALE_TEMP_SECONDS = 3600.0
TEMP_PREFIX = "docrooter-"
_LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})
_ANY_ADDRESS = frozenset({"", "0.0.0.0", "::"})
_FORM_OVERHEAD = 64 * 1024  # multipart framing and option fields on top of the file
_MAX_DIGITS = 18  # a longer header number is no byte count or event index (and int() may refuse it)
_CHUNK = 1024 * 1024
_ACCEPTED = (".pdf", ".md", ".txt")
_TOP_DOC_TYPES = 5
_ERROR_CHARS = 500
_STATIC = {
    "index.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
    "favicon.svg": "image/svg+xml",
}
_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; "
        "frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-store",
}


# --- backends -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendInfo:
    spec: str
    label: str
    description: str
    modules: tuple[str, ...]  # what its extra provides; all importable means installed

    def entry(self) -> dict[str, Any]:
        installed = all(importlib.util.find_spec(m) is not None for m in self.modules)
        return {
            "spec": self.spec,
            "label": self.label,
            "description": self.description,
            "installed": installed,
        }


_LOCAL_MODEL = ("torch", "sentence_transformers")
CATALOGUE: tuple[BackendInfo, ...] = (
    BackendInfo(
        "keyword",
        "Keyword",
        "Word overlap with the label descriptions. Instant and model-free: checks the pipeline, "
        "not the labels.",
        (),
    ),
    BackendInfo(
        "embedding:dmis-lab/UltraBERT-Embed-150M",
        "UltraBERT Embed 150M",
        "The default local embedding model.",
        _LOCAL_MODEL,
    ),
    BackendInfo(
        "embedding:BAAI/bge-small-en-v1.5",
        "bge-small",
        "A small, fast local embedding model (33M parameters).",
        _LOCAL_MODEL,
    ),
    BackendInfo(
        "nli:MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        "DeBERTa-v3 NLI",
        "A local zero-shot NLI model; the slowest of the local models.",
        ("torch", "transformers"),
    ),
)
JEV = BackendInfo(
    "jev:jev-1.13.0",
    "Jev 1.13",
    "TypeSafe's hosted model: the document text is sent to the TypeSafe API.",
    ("typesafe_sdk",),
)


def available_backends(default_spec: str) -> list[BackendInfo]:
    """The catalogue, Jev only when ``TYPESAFE_API_KEY`` is set, and ``default_spec`` (the
    ``--backend`` of ``docrooter serve``) when it is none of those."""
    found = list(CATALOGUE)
    if jev_key_set():  # set and not blank, as the profiles check it
        found.append(JEV)
    if all(info.spec != default_spec for info in found):
        found.append(BackendInfo(default_spec, default_spec, "Given with --backend.", ()))
    return found


class BackendPool:
    """The backend of the most recent run, built through a :class:`Classifier` on its spec so
    a model loads once and is reused. Asking for another spec closes the previous classifier
    (and the model it loaded) first. Used from the worker thread only."""

    def __init__(self, taxonomy: Taxonomy) -> None:
        self._taxonomy = taxonomy
        self.spec: str | None = None
        self._classifier: Classifier | None = None

    def get(self, spec: str) -> Backend:
        if spec != self.spec or self._classifier is None:
            self.close()
            self._classifier = Classifier(self._taxonomy, spec)
            self.spec = spec
        return self._classifier._resolve(spec)

    def close(self) -> None:
        classifier, self._classifier, self.spec = self._classifier, None, None
        if classifier is not None:
            classifier.close()


# --- profiles -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfileInfo:
    name: str
    label: str
    description: str
    modules: tuple[str, ...]  # what its extra provides; all importable means available
    extra: str | None = None

    def entry(self) -> dict[str, Any]:
        installed = all(importlib.util.find_spec(m) is not None for m in self.modules)
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "installed": installed,
            "extra": self.extra,
        }


PROFILE_INFO: tuple[ProfileInfo, ...] = (
    ProfileInfo(
        "fast",
        "Fast",
        "Headings and labels only, as in v0.3: no elements, no OCR, no summaries.",
        (),
    ),
    ProfileInfo(
        "balanced",
        "Balanced",
        "Elements by the layout rules, OCR for scanned pages when an engine is installed, "
        "and summaries.",
        (),
    ),
    ProfileInfo(
        "accurate",
        "Accurate",
        "Elements typed by the heron layout model (fetches its weights on first use), OCR and "
        "summaries.",
        ("onnxruntime", "pypdfium2", "huggingface_hub"),
        "layout",
    ),
)
# PDF engines that have no element mode: only the fast profile can use them.
_FAST_ONLY_ENGINES = frozenset({"both", "pymupdf4llm"})


class EnginePool:
    """The OCR and layout engines of the most recent element-mode run, keyed by profile. A run
    with the same profile reuses them, whatever the classification backend: the layout model
    does not depend on it. A different profile closes them first, and so does a run that needs
    none (the ``fast`` profile, see :meth:`close`), so a loaded layout model is released as soon
    as it is not wanted. The OCR engine is created on the first page that needs it. Used from
    the worker thread only."""

    def __init__(self) -> None:
        self.key: str | None = None
        self._engines: dict[str, Any] = {}

    def get(self, profile: str, config: RouterConfig) -> dict[str, Any]:
        """The engines for ``config`` as :func:`~docrooter.process` keywords (``ocr``,
        ``layout``); a task with no engine to share is left out and ``process`` handles it
        (``ocr="auto"`` with none installed becomes a warning in the route log)."""
        if self.key != profile:
            self.close()
            self._engines = self._build(config)
            self.key = profile
        return dict(self._engines)

    @staticmethod
    def _build(config: RouterConfig) -> dict[str, Any]:
        engines: dict[str, Any] = {}
        ocr = config.ocr
        if ocr == "auto":
            ocr = ocr_registry.resolve_auto_ocr() or "none"
        if ocr != "none":
            engines["ocr"] = _LazyOcr(ocr, config.ocr_lang)
        if config.layout not in ("none", "heuristic"):
            try:
                engines["layout"] = layout_registry.get_layout_engine(config.layout)
            except BaseException:
                if "ocr" in engines:
                    engines["ocr"].close()
                raise
        return engines

    def close(self) -> None:
        engines, self._engines, self.key = self._engines, {}, None
        for engine in engines.values():
            try:
                engine.close()
            except Exception:  # one failing close must not keep the other engine open
                logger.warning("closing %s failed", type(engine).__name__, exc_info=True)


# --- runs ---------------------------------------------------------------------------------


def stage_timings(log: list[dict[str, Any]]) -> dict[str, float]:
    """Seconds per stage from an event log, summed when a stage runs twice (``engine="both"``
    extracts with two engines)."""
    started: dict[str, float] = {}
    out: dict[str, float] = {}
    for event in log:
        stage = event["stage"]
        if event["status"] == "start":
            started[stage] = event["t"]
        elif event["status"] == "end" and stage in started:
            out[stage] = round(out.get(stage, 0.0) + event["t"] - started.pop(stage), 4)
    return out


@dataclass(eq=False)
class Run:
    """One upload and what became of it. Changed on the event loop thread only."""

    id: str
    filename: str
    kind: str  # "pdf", "md" or "txt"
    size: int
    options: dict[str, Any]
    created: float = field(default_factory=time.time)
    status: str = "queued"  # queued, running, done or error
    events: list[dict[str, Any]] = field(default_factory=list)
    listeners: set[asyncio.Queue[dict[str, Any]]] = field(default_factory=set)
    result: dict[str, Any] | None = None
    error: str | None = None
    seconds: float | None = None

    @property
    def finished(self) -> bool:
        return self.status in ("done", "error")

    def publish(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        for listener in self.listeners:
            listener.put_nowait(event)

    def started(self) -> None:
        self.status = "running"

    def finish(
        self, status: str, result: dict[str, Any] | None, error: str | None, seconds: float
    ) -> None:
        """Store the outcome, then send the closing ``job`` event (so a client that sees it
        can already fetch the result)."""
        self.status, self.result, self.error = status, result, error
        self.seconds = round(seconds, 4)
        data: dict[str, Any] = {"status": status, "seconds": self.seconds}
        if error is not None:
            data["error"] = error
        self.publish({"stage": "job", "status": "end", "t": self.seconds, "data": data})

    def summary(self) -> dict[str, Any]:
        doc_type = (self.result or {}).get("doc_type") or {}
        return {
            "id": self.id,
            "filename": self.filename,
            "kind": self.kind,
            "size": self.size,
            "created": self.created,
            "status": self.status,
            "options": self.options,
            "seconds": self.seconds,
            "error": self.error,
            "doc_type": doc_type.get("label"),
            "confidence": doc_type.get("confidence"),
        }

    def detail(self) -> dict[str, Any]:
        result = self.result or {}
        return {
            **self.summary(),
            "timings": stage_timings(self.events),
            "events": self.events,
            "document": result.get("document"),
            "meta": result.get("meta"),
            "toc": result.get("toc"),
            "toc_total": result.get("toc_total"),
            "truncated": result.get("truncated", False),
            "doc_type": result.get("doc_type"),
            "sections": result.get("sections"),
            "summaries": result.get("summaries"),
            "elements": result.get("elements"),
            "elements_total": result.get("elements_total"),
            "elements_truncated": result.get("elements_truncated", False),
            "element_counts": result.get("element_counts"),
            "element_tags": result.get("element_tags"),
            "field_family": result.get("field_family"),
            "routes": result.get("routes"),
        }


@dataclass(eq=False)
class Job:
    run: Run
    path: str
    loop: asyncio.AbstractEventLoop
    received_s: float
    queued_at: float = field(default_factory=time.perf_counter)

    def post(self, callback: Callable[..., None], *args: Any) -> None:
        """Run ``callback(*args)`` on the event loop; dropped once the loop has closed."""
        with suppress(RuntimeError):
            self.loop.call_soon_threadsafe(callback, *args)

    def push(self, event: events.Event) -> None:
        """The pipeline sink: hand the event to the loop thread. An error message names the
        upload, never the server's temp file."""
        data = event.model_dump(mode="json")
        if event.status == "error" and isinstance(data["data"].get("message"), str):
            data["data"]["message"] = self.redact(data["data"]["message"])
        self.loop.call_soon_threadsafe(self.run.publish, data)

    def redact(self, message: str) -> str:
        """``message`` with the temp file's path (as given, and resolved: macOS reports
        ``/private/var/...`` for ``/var/...``) replaced by the upload's name."""
        for path in sorted({os.path.realpath(self.path), self.path}, key=len, reverse=True):
            message = message.replace(path, self.run.filename)
        return message


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _summary_out(summary: Summary) -> dict[str, Any]:
    return {
        "key_sentences": [
            _cut(sentence, MAX_SUMMARY_SENTENCE_CHARS) for sentence in summary.key_sentences
        ],
        "keyphrases": [
            _cut(phrase, MAX_KEYPHRASE_CHARS) for phrase in summary.keyphrases[:MAX_KEYPHRASES]
        ],
    }


def _tags_out(tags: dict[str, Any]) -> dict[str, Any]:
    """An element's tags as the page shows them: per facet the labels, the confidence and the
    highest probabilities, best first (every label's, then the next, :data:`MAX_TAG_PROBS` at
    least)."""
    out = {}
    for facet, result in tags.items():
        ranked = sorted(result.probs.items(), key=lambda item: -item[1])
        keep = max(MAX_TAG_PROBS, len(result.labels))
        out[facet] = {
            "labels": list(result.labels),
            "confidence": round(result.confidence, 4),
            "probs": [{"id": k, "prob": round(v, 4)} for k, v in ranked[:keep]],
        }
    return out


def _elements_out(
    doc: Document, summaries: dict[str, Summary]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """The first :data:`MAX_TOC_ENTRIES` elements as the page shows them, and the count of
    every kind in the whole document."""
    counts: dict[str, int] = {}
    for el in doc.elements:
        counts[el.kind] = counts.get(el.kind, 0) + 1
    out: list[dict[str, Any]] = []
    for el in doc.elements[:MAX_TOC_ENTRIES]:
        summary = summaries.get(el.id)
        out.append(
            {
                "id": el.id,
                "kind": el.kind,
                "order": el.order,
                "page": el.boxes[0].page if el.boxes else None,
                "section_id": el.section_id,
                "text": _cut(el.text, MAX_ELEMENT_TEXT_CHARS),
                "text_chars": len(el.text),
                "source": el.source,
                "confidence": round(el.confidence, 4) if el.confidence is not None else None,
                "flags": list(el.flags),
                "level": el.level,
                "summary": _summary_out(summary)["key_sentences"] if summary else [],
            }
        )
        if el.tags is not None:  # only runs with element tags have the key
            out[-1]["tags"] = _tags_out(el.tags)
    return out, counts


def _routes_out(doc: Document) -> dict[str, Any] | None:
    """The route log in short, or ``None`` when the run routed nothing (the fast profile).

    Every page read has exactly one decision: ``page`` scope, or ``region`` scope when image
    regions were found on a page read from its text layer. ``pages`` counts both, so it matches
    the ``route`` stage's own end event. ``tasks`` says how each page was read (a region page
    counts as ``text``), ``regions`` is the number of image regions (``meta["needs_ocr"]``) and
    ``region_pages`` the pages that have any. ``decisions`` are the configuration decisions
    (the engines chosen).
    """
    log = doc.meta.get("routes")
    if log is None:
        return None
    tasks: dict[str, int] = {}
    region_pages = 0
    decisions: list[dict[str, str]] = []
    for item in log:
        if item["scope"] == "page":
            tasks[item["task"]] = tasks.get(item["task"], 0) + 1
        elif item["scope"] == "region":
            region_pages += 1
            tasks["text"] = tasks.get("text", 0) + 1
        else:
            decisions.append(
                {"task": item["task"], "engine": item["engine"], "reason": item["reason"]}
            )
    needs_ocr = doc.meta.get("needs_ocr")
    regions = needs_ocr["regions"] if needs_ocr else region_pages
    return {
        "pages": sum(tasks.values()),
        "tasks": tasks,
        "regions": regions,
        "region_pages": region_pages,
        "decisions": decisions,
        "needs_ocr": needs_ocr,
    }


def build_result(
    doc: Document,
    filename: str,
    doc_type: DocTypeResult,
    norm: NormalizationResult | None,
    doc_types: Taxonomy,
    summaries: dict[str, Summary] | None = None,
    element_tags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The JSON the page shows: TOC entries, the document type, the section labels, and, for
    runs in element mode, the elements, the summaries of the listed sections and the routes.
    ``element_tags`` (``{"source", "family"}``) names the tagger of a tagged run."""
    summaries = summaries or {}

    def label(node_id: str) -> str:
        return doc_types.node(node_id).label if node_id in doc_types else node_id

    all_entries = doc.toc()
    toc: list[dict[str, Any]] = []
    for entry in all_entries[:MAX_TOC_ENTRIES]:
        section = doc.sections[entry.id]
        toc.append(
            {
                "id": entry.id,
                "title": entry.title,
                "level": entry.level,
                "depth": entry.depth,
                "page": section.page_start,
                "heading_score": round(section.heading_score, 4),
                "flags": list(section.flags),
            }
        )
    best = doc_type.best
    type_out = {
        "id": best.node_ids[-1] if best.node_ids else None,
        "label": best.labels[-1] if best.labels else None,
        "path": list(best.labels),
        "node_ids": list(best.node_ids),
        "confidence": round(doc_type.confidence, 4),
        "stopped_reason": best.stopped_reason,
        "mode": doc_type.mode,
        "leaf_probs": [
            {"id": node_id, "label": label(node_id), "prob": round(prob, 4)}
            for node_id, prob in list(doc_type.leaf_probs.items())[:_TOP_DOC_TYPES]
        ],
    }
    sections = None
    if norm is not None:
        entries = []
        for item in toc:
            found = norm.sections.get(item["id"])
            entries.append(
                {
                    "id": item["id"],
                    "title": item["title"],
                    "level": item["level"],
                    "depth": item["depth"],
                    "canonical": found.canonical if found else None,
                    "label": found.label if found else None,
                    "score": round(found.score, 4) if found and found.canonical else None,
                }
            )
        sections = {"taxonomy": norm.taxonomy, "doc_type": norm.doc_type, "entries": entries}
    routes = _routes_out(doc)
    # JSON-safe whatever a loader put in; the per-page route log is summarised under "routes"
    meta = json.loads(json.dumps({k: v for k, v in doc.meta.items() if k != "routes"}, default=str))
    elements, element_counts = _elements_out(doc, summaries)
    return {
        "document": {"title": doc.title, "source": filename, "sections": len(doc.sections)},
        "meta": meta,
        "toc": toc,
        "toc_total": len(all_entries),
        "truncated": len(all_entries) > len(toc),
        "doc_type": type_out,
        "sections": sections,
        "summaries": {
            item["id"]: _summary_out(summaries[item["id"]])
            for item in toc
            if item["id"] in summaries
        },
        "elements": elements,
        "elements_total": len(doc.elements),
        "elements_truncated": len(doc.elements) > len(elements),
        "element_counts": element_counts,
        "element_tags": element_tags,
        "field_family": family_for_doc_type(best.node_ids[-1] if best.node_ids else ""),
        "routes": routes,
    }


@dataclass(eq=False)
class _Task:
    """A call for the worker thread, queued between runs: its result (or exception) is handed
    to ``future`` on the event loop. Used for selections that load a model, so models are only
    ever touched from the worker."""

    fn: Callable[[], Any]
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future[Any]

    def run(self) -> None:
        if self.future.done():  # the caller gave up (cancelled) or was answered: no work to do
            return
        try:
            value = self.fn()
        except Exception as exc:  # handed to the awaiting handler, which decides
            self._settle(exc=exc)
        else:
            self._settle(value=value)

    def fail(self, exc: BaseException) -> None:
        self._settle(exc=exc)

    def _settle(self, value: Any = None, exc: BaseException | None = None) -> None:
        def deliver() -> None:
            if self.future.done():
                return
            if exc is not None:
                self.future.set_exception(exc)
            else:
                self.future.set_result(value)

        with suppress(RuntimeError):
            self.loop.call_soon_threadsafe(deliver)


@dataclass(frozen=True)
class _Scored:
    """What a selection computed for a kept run, for the next one: the prepared document and the
    features and key of the last need. Replaced whole, never changed, so the event loop may
    read it while the worker writes."""

    ctx: DocContext
    encoder_spec: str | None
    need_key: tuple[Any, ...] | None = None
    X: np.ndarray | None = None


class _LazyTagger:
    """The playground's element tagger, loaded when :func:`prepare` first asks it to tag
    something: a tagged analysis never loads it."""

    def __init__(self, load: Callable[[], Any]) -> None:
        self._load = load

    def tag(self, elements: Any, family: str) -> Any:
        return self._load().tag(elements, family)


def _need_key(need: Any) -> tuple[str | None, tuple[Any, ...]]:
    """``(family, key)`` of a need: the family its tags must be in (``None`` for a query, the
    document's own) and what names it."""
    if isinstance(need, str):
        return None, ("query", need)
    fields = need if isinstance(need, list) else [need]
    return fields[0].family, ("fields", *(f.id for f in fields))


class Playground:
    """Run history, the job queue and its worker thread, and the backend pool."""

    def __init__(
        self,
        default_backend: str,
        element_tags: str | None = None,
        selector: Selector | None = None,
        selector_name: str | None = None,
    ) -> None:
        self.default_backend = default_backend
        self.element_tags = element_tags
        self.selector = selector
        self.selector_name = selector_name
        self.analyses: OrderedDict[str, Analysis] = OrderedDict()
        self.scored: dict[str, _Scored] = {}
        self._tagger: Any = None
        self.doc_types = builtin_taxonomy(DOC_TYPES)
        self.backends = BackendPool(self.doc_types)
        self.engines = EnginePool()
        self.runs: OrderedDict[str, Run] = OrderedDict()
        self.jobs: queue.Queue[Job | _Task | None] = queue.Queue(maxsize=MAX_PENDING)
        self._worker: threading.Thread | None = None
        self._current: Job | None = None  # the job the worker is running
        self.streams_ended = False

    def end_streams(self) -> None:
        """End every open event stream (each closes without its ``job`` event) and any opened
        later, so a server that is stopping need not wait for the runs pages follow. Call it
        on the event loop thread."""
        self.streams_ended = True
        for run in self.runs.values():
            for listener in run.listeners:
                listener.put_nowait(_END_STREAM)

    # lifecycle

    def start(self) -> None:
        sweep_stale_temp_files()
        self._worker = threading.Thread(target=self._work, name="docrooter-playground", daemon=True)
        self._worker.start()

    def stop(self, timeout: float | None = None) -> None:
        """Drop the jobs still waiting (deleting their files), then stop the worker after the
        job it is running; it closes the backend on its way out. If that job outlasts
        ``timeout`` (default :data:`STOP_TIMEOUT_SECONDS`), its file is deleted anyway."""
        while True:
            try:
                job = self.jobs.get_nowait()
            except queue.Empty:
                break
            if isinstance(job, _Task):
                job.fail(HTTPException(503, "The server is stopping."))
            elif job is not None:
                _unlink(job.path)
        self.jobs.put(None)  # room for it: the queue was just emptied, and only we add
        if self._worker is not None:
            self._worker.join(STOP_TIMEOUT_SECONDS if timeout is None else timeout)
            running = self._current
            if self._worker.is_alive() and running is not None:
                _unlink(running.path)

    # runs

    @property
    def full(self) -> bool:
        return self.jobs.full()

    def add(self, run: Run, job: Job) -> None:
        """Queue the job (``queue.Full`` when :data:`MAX_PENDING` already wait) and record
        the run, dropping the oldest beyond :data:`HISTORY`."""
        self.jobs.put_nowait(job)
        self.runs[run.id] = run
        while len(self.runs) > HISTORY:
            dropped, _ = self.runs.popitem(last=False)
            self.analyses.pop(dropped, None)
            self.scored.pop(dropped, None)

    def keep(self, run_id: str, analysis: Analysis) -> None:
        """Keep the analysis of a run for the dial, dropping the oldest beyond
        :data:`MAX_KEPT_ANALYSES`. Event loop thread only."""
        self.analyses[run_id] = analysis
        self.scored.pop(run_id, None)  # features of a replaced analysis are not this one's
        while len(self.analyses) > MAX_KEPT_ANALYSES:
            dropped, _ = self.analyses.popitem(last=False)
            self.scored.pop(dropped, None)

    async def on_worker(self, fn: Callable[[], Any]) -> Any:
        """The result of ``fn()`` run on the worker thread after the jobs ahead of it
        (``queue.Full`` when :data:`MAX_PENDING` already wait)."""
        loop = asyncio.get_running_loop()
        task = _Task(fn, loop, loop.create_future())
        self.jobs.put_nowait(task)
        return await task.future

    def select_cached(
        self,
        run_id: str,
        analysis: Analysis,
        need: SelectField | list[SelectField] | str,
        completeness: float,
        budget: int | None,
    ) -> Selection | None:
        """The selection when the run's features for ``need`` are already computed (only a
        re-threshold: no model, any thread), else ``None``."""
        from docrooter.select.selection import select_scored

        entry = self.scored.get(run_id)
        family, key = _need_key(need)
        if (
            entry is None
            or self.selector is None
            or entry.ctx.analysis is not analysis
            or entry.ctx.family != (family or entry.ctx.doc_family)
            or entry.need_key != key
            or entry.X is None
        ):
            return None
        return select_scored(
            analysis,
            entry.ctx,
            entry.X,
            need,
            completeness=completeness,
            budget=budget,
            selector=self.selector,
        )

    def select(
        self,
        run_id: str,
        analysis: Analysis,
        need: SelectField | list[SelectField] | str,
        completeness: float,
        budget: int | None,
    ) -> Selection:
        """:func:`~docrooter.select.selection.select` with this server's selector, tagger and
        embedder. The embedder is the selector's recorded spec, taken from the backend pool: a
        pooled backend of that spec is reused, otherwise the pooled model is closed first, so
        one model is held. The tagger loads only if an element needs tags. The prepared
        document and the features of the need are kept for the run, so the next completeness
        for it is :meth:`select_cached`. Worker thread only, unless ``completeness`` is 1.0 (no
        model; ``budget`` is refused there)."""
        from docrooter.select.selection import need_features, prepare, select, select_scored

        if completeness >= 1.0:
            return select(analysis, need, completeness=completeness, budget=budget)
        selector = self.selector
        assert selector is not None  # the endpoint checked
        family, key = _need_key(need)
        spec = selector.embedder
        encoder = None if spec is None else self.backends.get(spec)
        entry = self.scored.get(run_id)
        if (
            entry is None
            or entry.ctx.analysis is not analysis
            or entry.ctx.family != (family or entry.ctx.doc_family)
            or entry.encoder_spec != spec
        ):
            tagger = _LazyTagger(self.tagger) if self.element_tags is not None else None
            ctx = prepare(
                analysis, tagger=tagger, encoder=encoder, family=family, encoder_spec=spec
            )
            entry = _Scored(ctx, spec)
        else:
            entry.ctx.encoder = encoder  # the pool may have swapped the model since
        _, X = need_features(entry.ctx, need)
        entry = _Scored(entry.ctx, spec, key, X)
        if run_id in self.analyses:  # not evicted while this ran
            self.scored[run_id] = entry
        return select_scored(
            analysis,
            entry.ctx,
            X,
            need,
            completeness=completeness,
            budget=budget,
            selector=selector,
        )

    def _work(self) -> None:
        try:
            while (job := self.jobs.get()) is not None:
                if isinstance(job, _Task):
                    job.run()
                    continue
                self._current = job
                try:
                    self._execute(job)
                finally:
                    self._current = None
        finally:
            self.engines.close()
            self.backends.close()
            self._close_tagger()

    def tagger(self) -> Any:
        """The element tagger, loaded on first use (worker thread only)."""
        if self._tagger is None:
            from docrooter.tagger.element_tags import resolve_tagger

            assert self.element_tags is not None
            self._tagger = resolve_tagger(self.element_tags)
        return self._tagger

    def _close_tagger(self) -> None:
        tagger, self._tagger = self._tagger, None
        close = getattr(tagger, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.warning("closing the element tagger failed", exc_info=True)

    @property
    def tagger_name(self) -> str | None:
        """The tagger as the page names it: a bundle's directory name, else the spec."""
        spec = self.element_tags
        if spec is None or spec.startswith(("embedding:", "teacher:")):
            return spec
        return Path(spec).name

    def _load(self, job: Job, options: dict[str, Any]) -> tuple[Document, dict[str, Summary]]:
        """Read the document as the profile says and, in element mode, summarise it. The
        ``fast`` profile is :func:`~docrooter.load` as before; the others run the rooter's
        ``process`` for the loading, routing and summaries, and the playground then classifies
        with its pooled backend, so ``mode`` still applies. Document types are not escalated
        here (``escalate="none"``)."""
        is_pdf = job.run.kind == "pdf"
        if options["profile"] == "fast":
            self.engines.close()  # a fast run uses no engine: free a loaded layout model now
            pdf_options = {"engine": options["engine"], "max_pages": MAX_PAGES} if is_pdf else {}
            return load(Path(job.path), on_event=job.push, **pdf_options), {}
        overrides: dict[str, Any] = {
            "doc_type": "none",
            "sections": "none",
            "escalate": "none",
            "embeddings": "none",
        }
        if is_pdf:
            overrides.update(text=options["engine"], max_pages=MAX_PAGES)
        config = RouterConfig.resolve(profile=options["profile"], **overrides)
        engines = self.engines.get(options["profile"], config)
        analysis = process(Path(job.path), config=config, on_event=job.push, **engines)
        return analysis.document, analysis.summaries

    def _execute(self, job: Job) -> None:
        run, options = job.run, job.run.options
        started = time.perf_counter()
        job.post(run.started)
        status, result, error = "error", None, None
        try:
            with events.pipeline_run(job.push):
                stage = events.start("upload", filename=run.filename, bytes=run.size)
                stage.end(
                    bytes=run.size,
                    kind=run.kind,
                    received_s=round(job.received_s, 4),
                    queued_s=round(started - job.queued_at, 4),
                )
                doc, summaries = self._load(job, options)
                backend = self.backends.get(options["backend"])
                doc_type = classify_document(
                    doc, self.doc_types, backend=backend, mode=options["mode"], on_event=job.push
                )
                norm = None
                leaf = doc_type.best.node_ids[-1] if doc_type.best.node_ids else None
                tagged: dict[str, Any] | None = None
                if self.element_tags is not None and options["profile"] != "fast":
                    # the event names the bundle by its directory, never its full path
                    _tag_elements(doc, self.tagger(), leaf, self.tagger_name or "")
                    family = family_for_doc_type(leaf or "")
                    tagged = {"source": self.tagger_name, "family": family}
                if options["normalize"]:
                    norm = normalize_sections(
                        doc,
                        doc_type=leaf,
                        # without a type, the generic sections rather than classifying again
                        taxonomy=None if leaf else GENERIC_TAXONOMY,
                        backend=backend,
                        on_event=job.push,
                    )
            result = build_result(
                doc, run.filename, doc_type, norm, self.doc_types, summaries, tagged
            )
            if doc.elements:  # the fast profile builds none: nothing for the dial to read
                # the backends that typed it and labelled its sections, which select()
                # compares with the selector's fit; the playground never escalates
                config = RouterConfig.resolve(
                    profile=options["profile"],
                    doc_type=options["backend"],
                    sections=options["backend"] if norm is not None else "none",
                    escalate="none",
                )
                analysis = Analysis(
                    document=doc,
                    doc_type=doc_type,
                    sections=norm,
                    summaries=summaries,
                    config=config,
                )
                job.post(self.keep, run.id, analysis)
            status = "done"
        except Exception as exc:
            message = job.redact(str(exc)) or type(exc).__name__
            error = f"{type(exc).__name__}: {message}"[:_ERROR_CHARS]
            logger.warning("run %s (%s) failed", run.id, run.filename, exc_info=True)
        finally:
            _unlink(job.path)
            job.post(run.finish, status, result, error, time.perf_counter() - started)


def _unlink(path: str) -> None:
    with suppress(FileNotFoundError):
        os.unlink(path)


def sweep_stale_temp_files(max_age: float = STALE_TEMP_SECONDS) -> None:
    """Delete upload temp files (``docrooter-*``) older than ``max_age`` seconds, left behind
    by a server that was killed mid-run."""
    folder = Path(tempfile.gettempdir())
    cutoff = time.time() - max_age
    for path in folder.glob(f"{TEMP_PREFIX}*"):
        with suppress(OSError):
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()


# --- HTTP ---------------------------------------------------------------------------------


_END_STREAM: dict[str, Any] = {"stage": "end_stream"}
"""Put in a listener queue by :meth:`Playground.end_streams`; never sent to the client."""


def _sse(index: int, event: dict[str, Any]) -> str:
    return f"id: {index}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n"


async def _stream(run: Run, skip: int, playground: Playground) -> AsyncIterator[str]:
    """The run's events from ``skip`` on: the ones so far, then live ones until the closing
    ``job`` event, or until :meth:`Playground.end_streams`. Snapshot and subscription happen
    without an ``await`` in between, so no event is lost or repeated."""
    listener: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    backlog = run.events[skip:]
    live = not run.finished and not playground.streams_ended
    if live:
        run.listeners.add(listener)
    try:
        index = skip
        for event in backlog:
            yield _sse(index, event)
            index += 1
        while live:
            try:
                event = await asyncio.wait_for(listener.get(), KEEPALIVE_SECONDS)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
                continue
            if event is _END_STREAM:
                break
            yield _sse(index, event)
            index += 1
            live = event["stage"] != "job"
    finally:
        run.listeners.discard(listener)


def _host_and_port(value: str) -> tuple[str, int] | None:
    """``("127.0.0.1", 8765)`` from a Host header such as ``127.0.0.1:8765`` (port 80 when
    absent, IPv6 brackets dropped, lower case); None when it does not parse."""
    try:
        parts = urlsplit(f"//{value}")
        name, port = parts.hostname, parts.port
    except ValueError:
        return None
    if not name or parts.username is not None or parts.path or parts.query:
        return None
    return name.lower(), port if port is not None else 80


def allowed_host_names(host: str) -> frozenset[str]:
    """The Host names accepted for a server bound to ``host``: the loopback names for a
    loopback bind; the bound host otherwise, plus, for an any-address bind, this machine's
    name and addresses. Loopback names are always accepted."""
    bound = host.strip("[]").lower()
    names = set(_LOOPBACK_NAMES)
    if bound in _ANY_ADDRESS:
        machine = socket.gethostname().lower()
        names.update({machine, f"{machine}.local"})
        with suppress(OSError):
            names.update(str(info[4][0]).lower() for info in socket.getaddrinfo(machine, None))
    elif not _is_loopback_name(bound):
        names.add(bound)
    return frozenset(names)


def _is_loopback_name(name: str) -> bool:
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


class HostGuard:
    """Refuse (400) any HTTP request whose ``Host`` is not one of ``names`` on ``port``. This
    stops DNS rebinding: a hostile page whose domain resolves to 127.0.0.1 still sends its own
    domain as Host, so it can neither post documents nor read results."""

    def __init__(self, app: ASGIApp, names: frozenset[str], port: int) -> None:
        self.app, self.names, self.port = app, names, port

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            found = _host_and_port(Headers(scope=scope).get("host", ""))
            if found is None or found[0] not in self.names or found[1] != self.port:
                await PlainTextResponse("Invalid host header", status_code=400)(
                    scope, receive, send
                )
                return
        await self.app(scope, receive, send)


def _check_origin(request: Request) -> None:
    """Refuse a cross-site form post: a browser sends ``Origin``, and it must be this host."""
    origin = request.headers.get("origin")
    if origin is not None and urlsplit(origin).netloc != request.headers.get("host"):
        raise HTTPException(403, "Uploads from other sites are refused.")


def _flag(value: object) -> bool:
    return str(value).strip().lower() in ("1", "true", "on", "yes")


def _check_profile(profile: str, engine: str, kind: str) -> None:
    """Refuse (400) a profile that is unknown, whose extra is not installed, or that cannot
    read this upload with this PDF engine."""
    if profile not in PROFILES:
        raise HTTPException(400, f"Unknown profile {profile!r}; use one of {', '.join(PROFILES)}.")
    info = next(i for i in PROFILE_INFO if i.name == profile)
    if not info.entry()["installed"]:
        raise HTTPException(
            400,
            f"The {profile} profile needs the {info.extra} extra: "
            f"pip install 'docrooter[{info.extra}]'.",
        )
    if profile != "fast" and kind == "pdf" and engine in _FAST_ONLY_ENGINES:
        raise HTTPException(
            400,
            f"The {engine} PDF engine builds no elements; use it with the fast profile, or "
            "choose auto, pymupdf or pdfplumber.",
        )


def _options(form: Any, allowed: list[str], default_backend: str, kind: str) -> dict[str, Any]:
    backend = str(form.get("backend") or default_backend)
    engine = str(form.get("engine") or "auto")
    mode = str(form.get("mode") or "view")
    profile = str(form.get("profile") or DEFAULT_PROFILE)
    if backend not in allowed:
        raise HTTPException(400, f"Unknown backend {backend!r}; use one of {', '.join(allowed)}.")
    if engine not in ENGINES:
        raise HTTPException(400, f"Unknown engine {engine!r}; use one of {', '.join(ENGINES)}.")
    if mode not in MODES:
        raise HTTPException(400, f"Unknown mode {mode!r}; use one of {', '.join(MODES)}.")
    _check_profile(profile, engine, kind)
    return {
        "backend": backend,
        "engine": engine,
        "mode": mode,
        "normalize": _flag(form.get("normalize", "false")),
        "profile": profile,
    }


def _megabytes(limit: int) -> str:
    return f"{limit / (1024 * 1024):g} MB"


def _too_large(limit: int) -> HTTPException:
    return HTTPException(413, f"The file is larger than {_megabytes(limit)}.")


def _text_too_large(limit: int) -> HTTPException:
    return HTTPException(
        413, f"Text uploads (.md, .txt) are limited to {_megabytes(limit)}; this file is larger."
    )


def _counting(receive: Receive, cap: int, limit: int) -> Receive:
    """``receive`` that raises 413 once the body passes ``cap`` bytes, whatever the declared
    ``Content-Length`` said."""
    seen = 0

    async def counted() -> Message:
        nonlocal seen
        message = await receive()
        if message["type"] == "http.request":
            seen += len(message.get("body", b""))
            if seen > cap:
                raise _too_large(limit)
        return message

    return counted


async def _receive(upload: UploadFile, suffix: str, limit: int) -> tuple[str, int]:
    """Copy the upload to a new temp file and check it: ``(path, size)``. Text has the lower
    :data:`MAX_TEXT_UPLOAD_BYTES` cap. The file is deleted again when a check fails."""
    fd, path = tempfile.mkstemp(prefix=TEMP_PREFIX, suffix=suffix)
    # Text is checked as it streams past (the decoder holds at most a partial character).
    decoder = None if suffix == ".pdf" else codecs.getincrementaldecoder("utf-8-sig")()
    text_limit = MAX_TEXT_UPLOAD_BYTES if decoder is not None else None
    try:
        size, head = 0, b""
        with os.fdopen(fd, "wb") as out:
            while chunk := await upload.read(_CHUNK):
                size += len(chunk)
                if size > limit:
                    raise _too_large(limit)
                if text_limit is not None and size > text_limit:
                    raise _text_too_large(text_limit)
                if len(head) < 4:
                    head += chunk[: 4 - len(head)]
                if decoder is not None:
                    _decode(decoder, chunk, suffix)
                out.write(chunk)
        if size == 0:
            raise HTTPException(400, "The file is empty.")
        if suffix == ".pdf" and head != b"%PDF":
            raise HTTPException(415, "This .pdf file is not a PDF (it does not start with %PDF).")
        if decoder is not None:
            _decode(decoder, b"", suffix, final=True)
        return path, size
    except BaseException:
        _unlink(path)
        raise


def _decode(
    decoder: codecs.IncrementalDecoder, chunk: bytes, suffix: str, final: bool = False
) -> None:
    try:
        decoder.decode(chunk, final)
    except UnicodeDecodeError as exc:
        raise HTTPException(415, f"This {suffix} file is not UTF-8 text.") from exc


def _static(name: str) -> Response:
    data = resources.files("docrooter.ui").joinpath("static").joinpath(name).read_bytes()
    return Response(data, media_type=_STATIC[name], headers=_SECURITY_HEADERS)


def check_backend_spec(spec: str) -> None:
    """Raise ``BackendError`` when ``spec`` names an unknown backend kind (nothing loads)."""
    registry.check_spec(spec)


def create_app(
    backend_spec: str = "keyword",
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    element_tags: str | None = None,
    selector: str | os.PathLike[str] | None = None,
) -> FastAPI:
    """The playground app; ``backend_spec`` is the backend selected by default. ``host`` and
    ``port`` are where it is served: only requests addressed to them are answered (see
    :class:`HostGuard`). ``element_tags`` (a tagger bundle directory, ``embedding:<model>`` or
    ``teacher:<engine spec>``; checked here, loaded on the first run in element mode) tags the
    elements of every run that has them. ``selector`` (a ``selector.json`` or a folder holding
    one, read here; by default the ``selector.json`` of the ``element_tags`` bundle, if it has
    one) enables the completeness dial below 1.0."""
    check_backend_spec(backend_spec)
    if element_tags is not None:
        from docrooter.tagger.element_tags import check_spec

        check_spec(element_tags)
    loaded, selector_name = _load_selector(selector, element_tags)
    playground = Playground(backend_spec, element_tags, loaded, selector_name)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        playground.start()
        try:
            yield
        finally:
            playground.stop()

    app = FastAPI(title="docrooter playground", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.playground = playground
    app.add_middleware(HostGuard, names=allowed_host_names(host), port=port)

    def get_run(run_id: str) -> Run:
        run = playground.runs.get(run_id)
        if run is None:
            raise HTTPException(404, f"No run {run_id!r} (only the last {HISTORY} are kept).")
        return run

    @app.get("/", include_in_schema=False)
    async def index() -> Response:
        return _static("index.html")

    @app.get("/static/{name}", include_in_schema=False)
    async def static(name: str) -> Response:
        if name not in _STATIC:
            raise HTTPException(404, "Not found.")
        return _static(name)

    @app.get("/api/backends")
    async def backends() -> dict[str, Any]:
        return {
            "default": playground.default_backend,
            "loaded": playground.backends.spec,
            "backends": [info.entry() for info in available_backends(backend_spec)],
            "engines": list(ENGINES),
            "modes": list(MODES),
            "default_profile": DEFAULT_PROFILE,
            "profiles": [info.entry() for info in PROFILE_INFO],
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "max_text_upload_bytes": MAX_TEXT_UPLOAD_BYTES,
            "element_tags": playground.tagger_name,
            "selector": {
                "loaded": playground.selector is not None,
                "name": playground.selector_name,
                "embedder": playground.selector.embedder if playground.selector else None,
            },
        }

    @app.get("/api/fields")
    async def fields() -> dict[str, list[dict[str, str]]]:
        return _builtin_field_listing()

    @app.post("/api/runs", status_code=202)
    async def create_run(request: Request) -> dict[str, str]:
        _check_origin(request)
        if playground.full:
            raise HTTPException(503, f"{MAX_PENDING} documents are already waiting; try later.")
        limit = MAX_UPLOAD_BYTES
        cap = limit + _FORM_OVERHEAD
        length = request.headers.get("content-length")
        if length is None:
            raise HTTPException(411, "Send the upload with a Content-Length.")
        if not _is_decimal(length):
            raise HTTPException(400, "Content-Length must be a whole number of bytes.")
        if int(length) > cap:
            raise _too_large(limit)
        received = time.perf_counter()
        # Count what actually arrives too: the body may run past what it declared.
        request = Request(request.scope, receive=_counting(request.receive, cap, limit))
        form = await request.form(max_files=1, max_fields=10)
        try:
            upload = form.get("file")
            if not isinstance(upload, UploadFile):
                raise HTTPException(400, "Send the document as the multipart field 'file'.")
            filename = Path(upload.filename or "document").name
            # The part's Content-Type is ignored: browsers send "", application/octet-stream
            # or text/markdown for the same .md file. The suffix and the bytes are checked.
            suffix = Path(filename).suffix.lower()
            if suffix not in _ACCEPTED:
                raise HTTPException(415, "Upload a .pdf, .md or .txt file.")
            allowed = [info.spec for info in available_backends(backend_spec)]
            options = _options(form, allowed, backend_spec, suffix[1:])
            path, size = await _receive(upload, suffix, limit)
        finally:
            await form.close()
        run = Run(uuid.uuid4().hex[:12], filename, suffix[1:], size, options)
        job = Job(run, path, asyncio.get_running_loop(), time.perf_counter() - received)
        try:
            playground.add(run, job)
        except queue.Full:
            _unlink(path)
            raise HTTPException(
                503, f"{MAX_PENDING} documents are already waiting; try later."
            ) from None
        return {"run_id": run.id}

    @app.get("/api/runs")
    async def list_runs() -> list[dict[str, Any]]:
        return [run.summary() for run in reversed(playground.runs.values())]

    @app.get("/api/runs/{run_id}")
    async def run_detail(run_id: str) -> dict[str, Any]:
        return get_run(run_id).detail()

    @app.post("/api/runs/{run_id}/select")
    async def select_elements(run_id: str, request: Request) -> dict[str, Any]:
        _check_origin(request)
        run = get_run(run_id)
        need, completeness, budget = _select_request(await _read_json(request))
        analysis = playground.analyses.get(run_id)
        if analysis is None:
            if not run.finished:
                raise HTTPException(409, "This run is still running: select once it is done.")
            if run.options.get("profile") == "fast":
                raise HTTPException(
                    409, "The fast profile builds no elements: run the file again with Balanced."
                )
            raise HTTPException(
                404,
                f"No analysis kept for run {run_id!r} (only the last {MAX_KEPT_ANALYSES} runs "
                "with elements are kept): run it again.",
            )
        selector = playground.selector
        if completeness < 1.0:
            if selector is None:
                raise HTTPException(409, NO_SELECTOR)
            if isinstance(need, str) and selector.embedder is None:
                raise HTTPException(409, NO_QUERY_EMBEDDER)
        try:
            if completeness >= 1.0:  # no model: answer now, whatever the worker is doing
                selection = playground.select(run_id, analysis, need, completeness, budget)
            else:  # a re-threshold of features already computed is as cheap
                cached = playground.select_cached(run_id, analysis, need, completeness, budget)
                selection = cached or await playground.on_worker(
                    lambda: playground.select(run_id, analysis, need, completeness, budget)
                )
        except queue.Full:
            raise HTTPException(
                503, f"{MAX_PENDING} jobs are already waiting; try later."
            ) from None
        except (ValueError, DocrooterError, ImportError) as exc:  # a missing extra says how
            raise HTTPException(400, str(exc)[:_ERROR_CHARS]) from None
        return _selection_out(selection)

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str, request: Request) -> StreamingResponse:
        run = get_run(run_id)
        last = request.headers.get("last-event-id", "")
        skip = int(last) + 1 if _is_decimal(last) else 0
        return StreamingResponse(
            _stream(run, skip, playground),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


NO_SELECTOR = "no selector: fit one with docrooter fit-selector"
NO_QUERY_EMBEDDER = "free-text queries need an embedder; this selector was fitted without one"


async def _read_json(request: Request) -> Any:
    """The request's JSON body, 413 past :data:`MAX_SELECT_BODY_BYTES` (declared or sent), 400
    when it is not JSON."""
    length = request.headers.get("content-length", "")
    if _is_decimal(length) and int(length) > MAX_SELECT_BODY_BYTES:
        raise HTTPException(413, "The request is too large.")
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_SELECT_BODY_BYTES:
            raise HTTPException(413, "The request is too large.")
    try:
        return json.loads(body)
    except ValueError:
        raise HTTPException(400, "Send a JSON body.") from None


def _load_selector(
    path: str | os.PathLike[str] | None, element_tags: str | None
) -> tuple[Selector | None, str | None]:
    """The fitted selector and its file name: ``path`` (``selector.json`` or a folder holding
    one), else the ``selector.json`` of the ``element_tags`` bundle when there is one, else
    none. Raises ``DocrooterError`` when ``path`` holds none."""
    from docrooter.select.selection import SELECTOR_FILE, Selector

    if path is None and element_tags is not None:
        bundle = Path(element_tags)
        if bundle.is_dir() and (bundle / SELECTOR_FILE).is_file():
            path = bundle / SELECTOR_FILE
    if path is None:
        return None, None
    chosen = Path(path)
    file = chosen / SELECTOR_FILE if chosen.is_dir() else chosen
    return Selector.load(chosen), file.name


def _builtin_field_listing() -> dict[str, list[dict[str, str]]]:
    """The built-in fields as the picker shows them: family to ``id``, ``label``, ``description``
    of each field."""
    from docrooter.select.fields import builtin_fields, list_builtin_field_sets

    return {
        family: [
            {"id": f.id, "label": f.label, "description": f.description}
            for f in builtin_fields(family)
        ]
        for family in list_builtin_field_sets()
    }


def _select_request(body: Any) -> tuple[Any, float, int | None]:
    """The need (a built-in field, a list of fields of one family, or a free-text query),
    completeness and token budget of a select request; 400 for anything malformed."""
    if not isinstance(body, dict):
        raise HTTPException(400, "Send a JSON object with need and completeness.")
    completeness = body.get("completeness")
    if (
        isinstance(completeness, bool)
        or not isinstance(completeness, (int, float))
        or not 0.0 <= completeness <= 1.0  # also false for NaN
    ):
        raise HTTPException(400, "completeness must be a number from 0 to 1.")
    budget = body.get("budget")
    if budget is not None and (
        isinstance(budget, bool) or not isinstance(budget, int) or budget < 1
    ):
        raise HTTPException(400, "budget must be a whole number of tokens, 1 or more.")
    return _need(body.get("need")), float(completeness), budget


def _need(value: Any) -> Any:
    """A need from its JSON form: a field id, a list of field ids (of one family), or any
    other string as a free-text query of at most :data:`MAX_QUERY_CHARS` characters."""
    from docrooter.select.fields import builtin_fields, list_builtin_field_sets

    known = {f.id: f for family in list_builtin_field_sets() for f in builtin_fields(family)}
    if isinstance(value, list):
        unknown = [v for v in value if not isinstance(v, str) or v.strip() not in known]
        if not value or unknown:
            raise HTTPException(400, "A list of needs must hold built-in field ids only.")
        fields = [known[v.strip()] for v in value]
        if len({f.family for f in fields}) > 1:
            raise HTTPException(400, "The fields of one need must be of one family.")
        return fields[0] if len(fields) == 1 else fields
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(400, "need must be a field id, a list of field ids or a query.")
    if value.strip() in known:
        return known[value.strip()]
    if len(value) > MAX_QUERY_CHARS:
        raise HTTPException(400, f"A query may be at most {MAX_QUERY_CHARS} characters.")
    return value


def _selection_out(selection: Selection) -> dict[str, Any]:
    """The selection as the page gets it: the readout, the numbers, and the first
    :data:`MAX_TOC_ENTRIES` selected ids (the page lists no more elements than that). No chunk
    text, so the answer stays small however large the selection."""
    from docrooter.select.selection import readout

    out = selection.model_dump(mode="json", exclude={"chunks", "element_ids"})
    ids = selection.element_ids
    return {
        "readout": readout(selection),
        **out,
        "element_ids": ids[:MAX_TOC_ENTRIES],
        "selected_total": len(ids),
        "truncated": len(ids) > MAX_TOC_ENTRIES,
    }


def _is_decimal(value: str) -> bool:
    """Whether ``value`` is a non-negative ASCII decimal of at most 18 digits. ``str.isdigit``
    alone also accepts superscript digits, which ``int()`` rejects, and ``int()`` refuses more
    than 4,300 digits; no byte count or event index needs 19."""
    return value.isascii() and value.isdigit() and len(value) <= _MAX_DIGITS


_INLINE_COMMENT = re.compile(r"\s+#")


def load_env_file(
    path: str | os.PathLike[str], environ: MutableMapping[str, str] | None = None
) -> None:
    """Set the ``KEY=value`` pairs of a ``.env`` file that the environment lacks. Silent:
    nothing is printed or logged. Blank lines, comments and lines without ``=`` are skipped;
    ``export`` is stripped; a quoted value is taken between its quotes, and an unquoted one
    loses a trailing `` # comment`` (so ``KEY= # comment`` is empty, while ``KEY=#x`` keeps
    ``#x``: a comment needs whitespace before it)."""
    environ = os.environ if environ is None else environ
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw = line.split("=", 1)
        key, value = key.removeprefix("export ").strip(), raw.strip()
        if value[:1] in ("'", '"') and (end := value.find(value[0], 1)) > 0:
            value = value[1:end]
        elif _INLINE_COMMENT.match(raw):  # `KEY= # comment`: an empty value
            value = ""
        else:
            value = _INLINE_COMMENT.split(value, maxsplit=1)[0].strip()
        if key and " " not in key and key not in environ:
            environ[key] = value
