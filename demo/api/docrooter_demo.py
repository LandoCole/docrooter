"""The live demo's server function: classify one upload with doc-rooter and Jev.

``POST /api/docrooter_demo?name=<file name>``, with the file as the raw request body (a PDF,
Markdown or plain text, at most 4 MB). doc-rooter reads the file and classifies it at all four
rungs of the context ladder with Jev, TypeSafe's decision model: Jev reads the document view and
names its type, then tags every section (``doc_first``). The reply is JSON in the shape of the
demo's samples (see :func:`describe`): what the model saw at each rung, the sections, and one run
per rung. Errors reply ``{"error": "..."}``.

It runs as a Vercel Python function (the ``handler`` class) and under ``demo/serve.py`` locally;
``demo/build_samples.py`` uses :func:`describe` and :func:`run` for the samples. The TypeSafe key
comes from ``TYPESAFE_API_KEY`` in the environment and is never logged or returned. Each visitor
is limited in memory (per instance) to 10 runs a minute and 100 a day, a PDF is read to at most
30 pages, and document text is never logged or kept.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

MODEL = "jev-1.13.0"
PRICE = {"prompt": 0.042e-6, "completion": 0.0}  # Jev, US$ per token: input only, output free
MAX_BYTES = 4 * 1024 * 1024
MAX_PAGES = 30  # the layout model reads about 0.3 to 0.4 s per page
SECTIONS_SHOWN = 200
SECTION_VIEW_CHARS = 600  # a section's own view, as sent (the full rung's is cut for display)
DOC_VIEW_CHARS = 20000
TOP_TYPES = 6
FLOWS = ("doc_first",)  # read, then tag: the demo shows no section rules
LEVELS = ("headings", "terms", "lead", "full")
PER_MINUTE, PER_DAY = 10, 100
NO_TAG = "none"

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


def _cut(text: str, chars: int) -> str:
    return text if len(text) <= chars else text[:chars].rstrip() + " …"


def _fields(text: str) -> list[list[str]]:
    """A rendered document view as ``[label, content]`` pairs (its ``Label: content`` lines)."""
    fields: list[list[str]] = []
    for line in text.splitlines():
        name, sep, rest = line.partition(": ")
        if sep and name and name[0].isupper() and len(name) <= 16:
            fields.append([name, rest])
        elif fields:
            fields[-1][1] += "\n" + line
    return fields


def describe(doc: Any, gold_tags: dict[str, str] | None = None) -> dict[str, Any]:
    """What doc-rooter would send for ``doc`` at every rung: the document view (``views``,
    cut to :data:`DOC_VIEW_CHARS` for display) and, per section in reading order, its own view
    (``sections[].views``). ``gold_tags`` maps section ids to gold tags, where known."""
    from docrooter.flow import select_sections
    from docrooter.ladder import document_view, section_view

    tagged = {s.id for s in select_sections(doc)}
    ordered = [s for s in doc.walk() if s.id in tagged][:SECTIONS_SHOWN]
    ids = [s.id for s in ordered]
    per_rung = {lv: section_view(lv).render_all(doc, ids) for lv in LEVELS}
    views = {}
    for lv in LEVELS:
        rendered = document_view(lv).render(doc)
        views[lv] = {
            "fields": _fields(_cut(rendered.text, DOC_VIEW_CHARS)),
            "chars": len(rendered.text),
            "truncated": rendered.truncated,
        }
    sections = []
    for s in ordered:
        row: dict[str, Any] = {"id": s.id, "title": s.title, "level": s.level}
        if gold_tags and s.id in gold_tags:
            row["gold"] = gold_tags[s.id]
        row["views"] = {lv: _cut(per_rung[lv][s.id].text, SECTION_VIEW_CHARS) for lv in LEVELS}
        sections.append(row)
    n_pages = doc.meta.get("n_pages")
    return {
        "title": doc.title,
        "pages": n_pages if isinstance(n_pages, int) else None,
        "views": views,
        "sections": sections,
        "sections_total": len(tagged),
    }


def section_score(tags: dict[str, list], gold_tags: dict[str, str]) -> dict[str, Any] | None:
    """How a run's tags did against the gold tags it has, ``none`` abstaining: ``right`` of the
    ``gold`` domain sections tagged right, and the share of domain tags given that were right."""
    pairs = [(g, tags.get(sid, [None])[0]) for sid, g in gold_tags.items() if sid in tags]
    if not pairs:
        return None
    domain = [p for g, p in pairs if g != NO_TAG]
    given = [(g, p) for g, p in pairs if p not in (None, NO_TAG)]
    right = sum(1 for g, p in pairs if g == p and g != NO_TAG)
    return {
        "right": right,
        "gold": len(domain),
        "precision": right / len(given) if given else None,
        "none_right": sum(1 for g, p in pairs if g == NO_TAG and p in (None, NO_TAG)),
        "none_gold": sum(1 for g, _ in pairs if g == NO_TAG),
    }


def run(
    doc: Any,
    flow: str,
    level: str,
    backend: Any,
    gold: str | None = None,
    gold_tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """One ``classify`` run as the demo shows it: the type, Jev's probabilities for the most
    likely types, how it was decided, the tokens and cost, and every section's tag."""
    from docrooter import classify

    result = classify(doc, flow=flow, level=level, backend=backend, calls="per_item")
    tokens = result.tokens
    tags = {t.id: [t.label, round(t.confidence or 0.0, 3)] for t in result.sections}
    probs = sorted(result.doc_type_probs.items(), key=lambda kv: -kv[1])[:TOP_TYPES]
    return {
        "pred": result.doc_type,
        "correct": (result.doc_type == gold) if gold else None,
        "confidence": result.doc_type_confidence,
        "probs": [[k, round(v, 4)] for k, v in probs],
        "decided_by": result.decided_by,
        "flags": list(result.flags),
        "candidates": list(result.candidates),
        "tokens": {
            "prompt": tokens.prompt,
            "completion": tokens.completion,
            "calls": tokens.calls,
            "exact": tokens.exact,
            "sections": result.stage_tokens.get("sections", tokens).prompt,
            "document": result.stage_tokens["document"].prompt
            if "document" in result.stage_tokens
            else 0,
        },
        "cost_usd": round(
            tokens.prompt * PRICE["prompt"] + tokens.completion * PRICE["completion"], 6
        ),
        "tags": tags,
        "score": section_score(tags, gold_tags) if gold_tags else None,
    }


def labels_for(ids: list[str | None]) -> dict[str, str]:
    """The display labels of doc types and section tags."""
    from docrooter.taxonomies import builtin_taxonomy

    taxonomies = [builtin_taxonomy("doc_types"), builtin_taxonomy("section_catalogue")]
    out = {}
    for node_id in ids:
        for taxonomy in taxonomies:
            if node_id and node_id in taxonomy:
                node = taxonomy.node(node_id)
                out[node_id] = node.label
                if node_id.endswith(".other") and node.parent_id:  # "Other" alone says nothing
                    out[node_id] = f"{taxonomy.node(node.parent_id).label}: other"
    return out


def jev_backend() -> Any:
    from docrooter.backends.jev import JevBackend

    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not key:
        raise RuntimeError("The live demo isn't configured on this server.")
    return JevBackend(MODEL, api_key=key, concurrency=16, timeout=30)


def classify_bytes(data: bytes, name: str, flow: str) -> dict:
    """doc-rooter on one file's bytes, at every rung of ``flow``: the demo's JSON."""
    from docrooter import load

    jev_backend()  # no key: say so before reading the file
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

    out = describe(doc)
    with ThreadPoolExecutor(len(LEVELS)) as pool:
        # a backend each: a run's tokens are read from its backend's cumulative usage
        found = pool.map(lambda lv: run(doc, flow, lv, jev_backend()), LEVELS)
        runs = dict(zip(LEVELS, found, strict=True))
    out["runs"] = {f"{flow}/{lv}": r for lv, r in runs.items()}
    out["title"] = out["title"] or Path(name).stem or "Your document"
    used = [r["pred"] for r in runs.values()]
    used += [k for r in runs.values() for k, _ in r["probs"]]
    used += [t[0] for r in runs.values() for t in r["tags"].values()]
    out.update(gold=None, publisher=None, labels=labels_for(used))
    return out


class handler(BaseHTTPRequestHandler):  # noqa: N801 - Vercel looks for this name
    def do_POST(self) -> None:  # noqa: N802
        query = parse_qs(urlsplit(self.path).query)
        flow = (query.get("flow") or ["doc_first"])[0]
        name = (query.get("name") or ["upload"])[0][:200]
        if flow not in FLOWS:
            return self._reply(400, {"error": "Unknown flow."})
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
            body = classify_bytes(data, name, flow)
        except ValueError as exc:
            return self._reply(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - the visitor gets a short reason, never a trace
            generic = "doc-rooter couldn't read or classify that file."
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
