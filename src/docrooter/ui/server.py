"""The playground server: a FastAPI app that runs uploaded documents through the pipeline and
streams its events to the page as Server-Sent Events.

One worker thread takes jobs from a :class:`queue.Queue`, so runs are processed one at a time
and memory stays bounded. Each job enters :func:`~docrooter.events.pipeline_run` inside the
worker; its sink hands every event to the event loop with ``loop.call_soon_threadsafe``. All
run state is changed on the loop thread only, so the HTTP handlers need no locks.

Endpoints: ``POST /api/runs`` (multipart ``file`` plus ``backend``, ``engine``, ``mode`` and
``normalize``), ``GET /api/runs``, ``GET /api/runs/{id}``, ``GET /api/runs/{id}/events``
(SSE, ending with a ``{"stage": "job", "status": "end"}`` event the server always sends),
``GET /api/backends``, and the page itself at ``/``.
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
from typing import Any
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
from docrooter.pdf.load import ENGINES  # noqa: E402
from docrooter.taxonomies import builtin_taxonomy  # noqa: E402
from docrooter.taxonomy import Taxonomy  # noqa: E402

__all__ = ["MAX_UPLOAD_BYTES", "check_backend_spec", "create_app", "load_env_file"]

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
"""Largest accepted upload; read on every request, so tests can patch it."""
MAX_PAGES = 2000
HISTORY = 20
"""Runs kept in memory; the oldest is dropped when a new one arrives."""
KEEPALIVE_SECONDS = 15.0
MAX_PENDING = 8
"""Jobs that may wait behind the running one; more get 503."""
STOP_TIMEOUT_SECONDS = 10.0
STALE_TEMP_SECONDS = 3600.0
TEMP_PREFIX = "docrooter-"
_LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})
_ANY_ADDRESS = frozenset({"", "0.0.0.0", "::"})
_FORM_OVERHEAD = 64 * 1024  # multipart framing and option fields on top of the file
_CHUNK = 1024 * 1024
_ACCEPTED = (".pdf", ".md", ".txt")
_TOP_DOC_TYPES = 5
_ERROR_CHARS = 500
_STATIC = {
    "index.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
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
    if os.environ.get("TYPESAFE_API_KEY"):
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
            "doc_type": result.get("doc_type"),
            "sections": result.get("sections"),
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
        """The pipeline sink: hand the event to the loop thread."""
        self.loop.call_soon_threadsafe(self.run.publish, event.model_dump(mode="json"))


def build_result(
    doc: Document,
    filename: str,
    doc_type: DocTypeResult,
    norm: NormalizationResult | None,
    doc_types: Taxonomy,
) -> dict[str, Any]:
    """The JSON the page shows: TOC entries, the document type and the section labels."""

    def label(node_id: str) -> str:
        return doc_types.node(node_id).label if node_id in doc_types else node_id

    toc: list[dict[str, Any]] = []
    for entry in doc.toc():
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
    meta = json.loads(json.dumps(doc.meta, default=str))  # JSON-safe whatever a loader put in
    return {
        "document": {"title": doc.title, "source": filename, "sections": len(doc.sections)},
        "meta": meta,
        "toc": toc,
        "doc_type": type_out,
        "sections": sections,
    }


class Playground:
    """Run history, the job queue and its worker thread, and the backend pool."""

    def __init__(self, default_backend: str) -> None:
        self.default_backend = default_backend
        self.doc_types = builtin_taxonomy(DOC_TYPES)
        self.backends = BackendPool(self.doc_types)
        self.runs: OrderedDict[str, Run] = OrderedDict()
        self.jobs: queue.Queue[Job | None] = queue.Queue(maxsize=MAX_PENDING)
        self._worker: threading.Thread | None = None
        self._current: Job | None = None  # the job the worker is running

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
            if job is not None:
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
            self.runs.popitem(last=False)

    def _work(self) -> None:
        try:
            while (job := self.jobs.get()) is not None:
                self._current = job
                try:
                    self._execute(job)
                finally:
                    self._current = None
        finally:
            self.backends.close()

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
                pdf_options = (
                    {"engine": options["engine"], "max_pages": MAX_PAGES}
                    if run.kind == "pdf"
                    else {}
                )
                doc = load(Path(job.path), on_event=job.push, **pdf_options)
                backend = self.backends.get(options["backend"])
                doc_type = classify_document(
                    doc, self.doc_types, backend=backend, mode=options["mode"], on_event=job.push
                )
                norm = None
                if options["normalize"]:
                    leaf = doc_type.best.node_ids[-1] if doc_type.best.node_ids else None
                    norm = normalize_sections(
                        doc,
                        doc_type=leaf,
                        # without a type, the generic sections rather than classifying again
                        taxonomy=None if leaf else GENERIC_TAXONOMY,
                        backend=backend,
                        on_event=job.push,
                    )
            result = build_result(doc, run.filename, doc_type, norm, self.doc_types)
            status = "done"
        except Exception as exc:
            message = str(exc).replace(job.path, run.filename) or type(exc).__name__
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


def _sse(index: int, event: dict[str, Any]) -> str:
    return f"id: {index}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n"


async def _stream(run: Run, skip: int) -> AsyncIterator[str]:
    """The run's events from ``skip`` on: the ones so far, then live ones until the closing
    ``job`` event. Snapshot and subscription happen without an ``await`` in between, so no
    event is lost or repeated."""
    listener: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    backlog = run.events[skip:]
    live = not run.finished
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


def _options(form: Any, allowed: list[str], default_backend: str) -> dict[str, Any]:
    backend = str(form.get("backend") or default_backend)
    engine = str(form.get("engine") or "auto")
    mode = str(form.get("mode") or "view")
    if backend not in allowed:
        raise HTTPException(400, f"Unknown backend {backend!r}; use one of {', '.join(allowed)}.")
    if engine not in ENGINES:
        raise HTTPException(400, f"Unknown engine {engine!r}; use one of {', '.join(ENGINES)}.")
    if mode not in MODES:
        raise HTTPException(400, f"Unknown mode {mode!r}; use one of {', '.join(MODES)}.")
    return {
        "backend": backend,
        "engine": engine,
        "mode": mode,
        "normalize": _flag(form.get("normalize", "false")),
    }


def _too_large(limit: int) -> HTTPException:
    return HTTPException(413, f"The file is larger than {limit // (1024 * 1024)} MB.")


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
    """Copy the upload to a new temp file and check it: ``(path, size)``. The file is deleted
    again when a check fails."""
    fd, path = tempfile.mkstemp(prefix=TEMP_PREFIX, suffix=suffix)
    # Text is checked as it streams past (the decoder holds at most a partial character).
    decoder = None if suffix == ".pdf" else codecs.getincrementaldecoder("utf-8-sig")()
    try:
        size, head = 0, b""
        with os.fdopen(fd, "wb") as out:
            while chunk := await upload.read(_CHUNK):
                size += len(chunk)
                if size > limit:
                    raise _too_large(limit)
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
    backend_spec: str = "keyword", *, host: str = "127.0.0.1", port: int = 8765
) -> FastAPI:
    """The playground app; ``backend_spec`` is the backend selected by default. ``host`` and
    ``port`` are where it is served: only requests addressed to them are answered (see
    :class:`HostGuard`)."""
    check_backend_spec(backend_spec)
    playground = Playground(backend_spec)

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
            "max_upload_bytes": MAX_UPLOAD_BYTES,
        }

    @app.post("/api/runs", status_code=202)
    async def create_run(request: Request) -> dict[str, str]:
        _check_origin(request)
        if playground.full:
            raise HTTPException(503, f"{MAX_PENDING} documents are already waiting; try later.")
        limit = MAX_UPLOAD_BYTES
        cap = limit + _FORM_OVERHEAD
        length = request.headers.get("content-length", "")
        if not length.isdigit():
            raise HTTPException(411, "Send the upload with a Content-Length.")
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
            options = _options(form, allowed, backend_spec)
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

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str, request: Request) -> StreamingResponse:
        run = get_run(run_id)
        last = request.headers.get("last-event-id", "")
        skip = int(last) + 1 if last.isdigit() else 0
        return StreamingResponse(
            _stream(run, skip),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


_INLINE_COMMENT = re.compile(r"\s+#")


def load_env_file(
    path: str | os.PathLike[str], environ: MutableMapping[str, str] | None = None
) -> None:
    """Set the ``KEY=value`` pairs of a ``.env`` file that the environment lacks. Silent:
    nothing is printed or logged. Blank lines, comments and lines without ``=`` are skipped;
    ``export`` is stripped; a quoted value is taken between its quotes, and an unquoted one
    loses a trailing `` # comment``."""
    environ = os.environ if environ is None else environ
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.removeprefix("export ").strip(), value.strip()
        if value[:1] in ("'", '"') and (end := value.find(value[0], 1)) > 0:
            value = value[1:end]
        else:
            value = _INLINE_COMMENT.split(value, maxsplit=1)[0].strip()
        if key and " " not in key and key not in environ:
            environ[key] = value
