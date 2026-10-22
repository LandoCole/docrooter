"""The live demo's server function: classify one upload with docrooter and Claude Haiku 5.5.

``POST /api/docrooter_demo?flow=<flow>&level=<level>&name=<file name>`` (flow ``doc_first`` or
``sections_first``, level ``headings``, ``terms``, ``lead`` or ``full``), with the file as the
raw request body (a PDF, Markdown or plain text, at most 4 MB). The reply is JSON in the shape
of the demo's sample runs: the four rung views, the sections, and one run for the requested flow
and level. Errors reply ``{"error": "..."}``.

It runs as a Vercel Python function (the ``handler`` class) and under ``demo/serve.py`` locally.
The Anthropic key comes from ``ANTHROPIC_API_KEY`` in the environment and is never logged or
returned. Each visitor is limited in memory (per instance) to 10 runs a minute and 100 a day, a
PDF is read to at most 30 pages, and document text is never logged or kept.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

MODEL = "claude-haiku-5-5"
BASE_URL = "https://api.anthropic.com/v1"
PRICE = {"prompt": 0.10e-6, "completion": 0.50e-6}  # US$ per token, list price
MAX_BYTES = 4 * 1024 * 1024
MAX_PAGES = 30  # the layout model reads about 0.3 to 0.4 s per page
SECTIONS_SHOWN = 24
VIEW_PREVIEW_CHARS = 2400
FLOWS = ("doc_first", "sections_first")
LEVELS = ("headings", "terms", "lead", "full")
PER_MINUTE, PER_DAY = 10, 100

_hits: dict[str, deque[float]] = defaultdict(deque)
_lock = threading.Lock()


def allowed(visitor: str, now: float | None = None) -> bool:
    """Record one run for ``visitor`` and say whether it is within the limits."""
    now = time.time() if now is None else now
    with _lock:
        hits = _hits[visitor]
        while hits and now - hits[0] > 86400:
            hits.popleft()
        if len(hits) >= PER_DAY or sum(1 for t in hits if now - t < 60) >= PER_MINUTE:
            return False
        hits.append(now)
        return True


def _view(doc, level: str) -> dict:
    from docrooter.ladder import document_view

    text = document_view(level).render(doc).text
    shown = text if len(text) <= VIEW_PREVIEW_CHARS else text[:VIEW_PREVIEW_CHARS].rstrip() + " …"
    return {"text": shown, "chars": len(text)}


def classify_bytes(data: bytes, name: str, flow: str, level: str) -> dict:
    """docrooter on one file's bytes: the demo's JSON for one run."""
    from docrooter import classify, load
    from docrooter.backends.openai import OpenAIBackend
    from docrooter.flow import select_sections
    from docrooter.taxonomies import builtin_taxonomy

    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("The live demo isn't configured on this server.")
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf" or data[:5] == b"%PDF-":
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "upload.pdf"
            path.write_bytes(data)
            doc = load(path, max_pages=MAX_PAGES)
    elif suffix in (".md", ".markdown", ".txt", ""):
        text = data.decode("utf-8", errors="replace")
        if not text.strip():
            raise ValueError("That file has no text.")
        doc = load(text if "\n" in text else text + "\n")  # a str with a newline is Markdown
    else:
        raise ValueError("Choose a PDF, Markdown or text file.")

    backend = OpenAIBackend(
        MODEL,
        base_url=BASE_URL,
        api_key=key,
        timeout=30,
        concurrency=4,
        extra_body={"thinking": {"type": "disabled"}},
    )
    result = classify(doc, flow=flow, level=level, backend=backend, calls="batched")
    sections = select_sections(doc)
    tokens = result.tokens
    labels = {}
    taxonomies = [builtin_taxonomy("doc_types")]
    if result.section_taxonomy:
        taxonomies.append(builtin_taxonomy(result.section_taxonomy))
    for node_id in [result.doc_type, *(t.label for t in result.sections)]:
        for taxonomy in taxonomies:
            if node_id and node_id in taxonomy:
                labels[node_id] = taxonomy.node(node_id).label
    n_pages = doc.meta.get("n_pages")
    return {
        "title": doc.title or Path(name).stem or "Your document",
        "pages": n_pages if isinstance(n_pages, int) else None,
        "publisher": None,
        "gold": None,
        "views": {lv: _view(doc, lv) for lv in LEVELS},
        "sections": [
            {"id": s.id, "title": s.title, "level": s.level} for s in sections[:SECTIONS_SHOWN]
        ],
        "sections_total": len(sections),
        "labels": labels,
        "runs": {
            f"{flow}/{level}": {
                "pred": result.doc_type,
                "correct": None,
                "decided_by": result.decided_by,
                "flags": list(result.flags),
                "candidates": list(result.candidates),
                "tokens": {
                    "prompt": tokens.prompt,
                    "completion": tokens.completion,
                    "calls": tokens.calls,
                    "exact": tokens.exact,
                },
                "cost_usd": round(
                    tokens.prompt * PRICE["prompt"] + tokens.completion * PRICE["completion"], 6
                ),
                "tags": {t.id: {"pred": t.label} for t in result.sections},
            }
        },
    }


class handler(BaseHTTPRequestHandler):  # noqa: N801 - Vercel looks for this name
    def do_POST(self) -> None:  # noqa: N802
        query = parse_qs(urlsplit(self.path).query)
        flow = (query.get("flow") or ["doc_first"])[0]
        level = (query.get("level") or ["headings"])[0]
        name = (query.get("name") or ["upload"])[0][:200]
        if flow not in FLOWS or level not in LEVELS:
            return self._reply(400, {"error": "Unknown flow or level."})
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            return self._reply(400, {"error": "The upload was empty."})
        if length > MAX_BYTES:
            return self._reply(413, {"error": "The live demo takes files up to 4 MB."})
        forwarded = self.headers.get("x-forwarded-for") or self.client_address[0]
        if not allowed(forwarded.split(",")[0].strip()):
            limit = "That's the limit for now: 10 runs a minute, 100 a day."
            return self._reply(429, {"error": limit})
        data = self.rfile.read(length)
        try:
            body = classify_bytes(data, name, flow, level)
        except ValueError as exc:
            return self._reply(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - the visitor gets a short reason, never a trace
            generic = "docrooter couldn't read or classify that file."
            reason = str(exc) if isinstance(exc, RuntimeError) else generic
            return self._reply(502, {"error": reason})
        return self._reply(200, body)

    def do_GET(self) -> None:  # noqa: N802
        self._reply(405, {"error": "POST a file to classify it."})

    def _reply(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """No request logging: query strings carry file names."""
