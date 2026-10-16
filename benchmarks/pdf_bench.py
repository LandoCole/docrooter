"""Real-PDF document-type benchmark: docrooter backends vs TypeSafe Jev on the labelled corpus.

``PYTHONPATH=src uv run --extra models --extra pdf --extra jev --extra bench \
    python -m benchmarks.pdf_bench``

Every corpus PDF is parsed once with the default engine and cached as Document JSON (with its
parse seconds) in ``benchmarks/.cache/docs/``. Each system in :data:`SYSTEMS` then labels every
document with one of the 18 leaf types (threshold 0, so there is always a label). Results go to
``benchmarks/results/<date>/pdf.json`` after each system, and ``--docs`` rewrites the generated
block of the ``## Document-type classification`` section of ``docs/benchmarks-pdf.md``.

Timing: local-model systems run one untimed warm-up classification of a tiny synthetic document
first (model load + label encoding, reported as ``load_s``). A batch system's per-document
classify time is the batch's wall time divided by the number of documents.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import platform
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks import jev_chunks
from benchmarks.toc import git_revision
from docrooter.document import Document
from docrooter.evaluate.metrics import leaf_prf
from docrooter.taxonomies import builtin_taxonomy
from docrooter.taxonomy import Taxonomy

ROOT = Path(__file__).parent.parent
CORPUS = Path(__file__).parent / "corpus"
CACHE = Path(__file__).parent / ".cache"
DOC_CACHE = CACHE / "docs"
RESULTS_ROOT = Path(__file__).parent / "results"
DOCS_PATH = ROOT / "docs" / "benchmarks-pdf.md"
ENV_FILE = ROOT / ".env"

JEV_MODEL = "jev-1.13.0"
JEV_CONCURRENCY = 8
JEV_USD_PER_M_INPUT = 0.042
ULTRABERT = "dmis-lab/UltraBERT-Embed-150M"
INSTRUCTION = "What type of document is this?"
DEVICE_PROBE_S = 10.0
BOOTSTRAP_RESAMPLES = 1000
NONE_LABEL = "(none)"


@dataclass(frozen=True)
class System:
    id: str
    kind: str  # "docrooter" (classify_documents), "jev-chunks" or "loo"
    what: str
    backend: str | None = None
    mode: str = "view"

    @property
    def is_jev(self) -> bool:
        return self.id.startswith("jev")


SYSTEMS: dict[str, System] = {
    s.id: s
    for s in (
        System(
            "ultrabert-view",
            "docrooter",
            "UltraBERT-Embed-150M, DocumentView, Flat (zero-shot)",
            f"embedding:{ULTRABERT}",
        ),
        System(
            "ultrabert-hybrid",
            "docrooter",
            "UltraBERT-Embed-150M, 0.6 view + 0.4 sections (zero-shot)",
            f"embedding:{ULTRABERT}",
            "hybrid",
        ),
        System(
            "bge-view",
            "docrooter",
            "bge-small-en-v1.5, DocumentView, Flat (zero-shot)",
            "embedding:BAAI/bge-small-en-v1.5",
        ),
        System(
            "nli-view",
            "docrooter",
            "deberta-v3-base-zeroshot-v2.0 NLI, DocumentView, Flat (zero-shot)",
            "nli:MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        ),
        System(
            "jev-view",
            "docrooter",
            f"Jev {JEV_MODEL}, DocumentView as state, one Choice over 18 leaves",
            f"jev:{JEV_MODEL}",
        ),
        System(
            "jev-chunks",
            "jev-chunks",
            f"Jev {JEV_MODEL}, raw page text, <=8 chunks of ~1,500 tokens, "
            "mean of per-chunk Choices (no docrooter)",
        ),
        System(
            "ultrabert-loo",
            "loo",
            "UltraBERT-Embed-150M view embeddings + logistic regression, trained (leave-one-out)",
        ),
    )
}
DEFAULT_ORDER = (
    "jev-view",
    "jev-chunks",
    "bge-view",
    "ultrabert-view",
    "ultrabert-loo",
    "nli-view",
    "ultrabert-hybrid",
)


def log(message: str) -> None:
    print(f"[{datetime.datetime.now():%H:%M:%S}] {message}", flush=True)


# --- secrets -----------------------------------------------------------------------------


def load_env(path: Path = ENV_FILE, environ: MutableMapping[str, str] | None = None) -> list[str]:
    """Set the ``KEY=value`` pairs of a ``.env`` file that ``environ`` does not have yet.

    Silent: nothing is printed or logged, and only the names that were set are returned. Blank
    lines, comments and lines without ``=`` are skipped; ``export`` and matching quotes are
    stripped. A missing file sets nothing.
    """
    environ = os.environ if environ is None else environ
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    loaded: list[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if not key or " " in key or key in environ:
            continue
        environ[key] = value
        loaded.append(key)
    return loaded


# --- corpus and parsing ------------------------------------------------------------------


def manifest(corpus: Path = CORPUS) -> list[dict[str, Any]]:
    with (corpus / "manifest.jsonl").open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def parse_corpus(
    engine: str = "auto",
    *,
    entries: Sequence[dict[str, Any]] | None = None,
    corpus: Path = CORPUS,
    cache: Path = DOC_CACHE,
) -> dict[str, tuple[Document, float]]:
    """``{doc_id: (Document, parse_seconds)}``, parsing each PDF once and caching it as JSON.

    A cached file is reused when its engine matches. Before the first uncached parse, the
    smallest PDF is parsed once untimed so import and first-call costs do not land on a document.
    """
    from docrooter.pdf.load import load_pdf

    entries = list(entries) if entries is not None else manifest(corpus)
    cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, tuple[Document, float]] = {}
    warmed = False
    for entry in entries:
        path = cache / f"{entry['id']}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("engine") == engine:
                out[entry["id"]] = (Document.model_validate(saved["document"]), saved["parse_s"])
                continue
        if not warmed:
            smallest = min(entries, key=lambda e: e["pages"])
            load_pdf(corpus / smallest["file"], engine=engine)
            warmed = True
        start = time.perf_counter()
        doc = load_pdf(corpus / entry["file"], engine=engine)
        seconds = time.perf_counter() - start
        record = {
            "doc_id": entry["id"],
            "engine": engine,
            "parse_s": seconds,
            "loadavg": list(os.getloadavg()),
            "document": doc.model_dump(mode="json"),
        }
        path.write_text(json.dumps(record), encoding="utf-8")
        out[entry["id"]] = (doc, seconds)
        log(f"parsed {entry['id']} ({entry['pages']} pages) in {seconds:.2f} s")
    return out


# --- device ------------------------------------------------------------------------------


def probe_device(timeout: float = DEVICE_PROBE_S) -> tuple[str, str]:
    """``("mps", note)`` when a 2000x2000 MPS matmul finishes within ``timeout`` s, else CPU.

    The probe runs in a daemon thread, so a hung MPS cannot block the benchmark.
    """
    import torch

    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return "cpu", "MPS not available"
    result: dict[str, Any] = {}

    def work() -> None:
        try:
            start = time.perf_counter()
            x = torch.randn(2000, 2000, device="mps")
            (x @ x).sum().item()
            result["seconds"] = time.perf_counter() - start
        except Exception as exc:  # noqa: BLE001 - any MPS failure means "use CPU"
            result["error"] = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    thread.join(timeout)
    if "seconds" in result:
        return "mps", f"MPS probe matmul took {result['seconds']:.2f} s"
    if "error" in result:
        return "cpu", f"MPS probe failed ({result['error'][:200]})"
    return "cpu", f"MPS probe did not finish within {timeout:.0f} s"


def free_memory() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()


# --- metrics -----------------------------------------------------------------------------


def bootstrap_ci(
    hits: Sequence[bool], n: int = BOOTSTRAP_RESAMPLES, seed: int = 0
) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean of ``hits`` (``n`` resamples, fixed seed)."""
    if not len(hits):
        return 0.0, 0.0
    values = np.asarray(hits, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _path(taxonomy: Taxonomy, node_id: str | None) -> list[str]:
    return [n.id for n in taxonomy.path(node_id)] if node_id else []


def system_metrics(rows: Sequence[Mapping[str, Any]], taxonomy: Taxonomy) -> dict[str, Any]:
    """Accuracy (leaf, level 1, bootstrap CI), macro and per-class P/R/F1 over the gold leaf
    classes, the confusion matrix, mean seconds and Jev token totals for one system's rows."""
    gold = [_path(taxonomy, r["gold"]) for r in rows]
    pred = [_path(taxonomy, r["pred"]) for r in rows]
    hits = [bool(p) and p[-1] == g[-1] for p, g in zip(pred, gold, strict=True)]
    l1 = [bool(p) and p[0] == g[0] for p, g in zip(pred, gold, strict=True)]
    macro_p, macro_r, macro_f1 = leaf_prf(pred, gold)
    per_class: dict[str, dict[str, Any]] = {}
    for c in sorted({r["gold"] for r in rows}):
        p, r, f = _class_prf(rows, c)
        per_class[c] = {"p": p, "r": r, "f1": f, "n": sum(row["gold"] == c for row in rows)}
    confusion: dict[str, dict[str, int]] = {}
    for row in rows:
        cell = confusion.setdefault(row["gold"], {})
        key = row["pred"] or NONE_LABEL
        cell[key] = cell.get(key, 0) + 1
    confusion = {g: dict(sorted(c.items())) for g, c in sorted(confusion.items())}
    n = len(rows)
    tokens = [r.get("tokens") for r in rows if r.get("tokens") is not None]
    total_tokens = sum(tokens) if tokens else None

    def mean(key: str) -> float:
        return sum(float(r.get(key) or 0.0) for r in rows) / n if n else 0.0

    return {
        "n": n,
        "leaf_accuracy": sum(hits) / n if n else 0.0,
        "leaf_accuracy_ci95": bootstrap_ci(hits),
        "level1_accuracy": sum(l1) / n if n else 0.0,
        "macro_p": macro_p,
        "macro_r": macro_r,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion": confusion,
        "avg_parse_s": mean("parse_s"),
        "avg_classify_s": mean("classify_s"),
        "avg_total_s": mean("parse_s") + mean("classify_s"),
        "jev_input_tokens": total_tokens,
        "jev_tokens_estimated": any(r.get("tokens_estimated") for r in rows) if tokens else None,
        "jev_cost_usd": total_tokens * JEV_USD_PER_M_INPUT / 1e6 if total_tokens else None,
    }


def _class_prf(rows: Sequence[Mapping[str, Any]], c: str) -> tuple[float, float, float]:
    tp = sum(r["pred"] == c and r["gold"] == c for r in rows)
    n_pred = sum(r["pred"] == c for r in rows)
    n_gold = sum(r["gold"] == c for r in rows)
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


# --- leave-one-out -----------------------------------------------------------------------

FitPredict = Callable[[Sequence[Any], Sequence[str], Any], tuple[str, dict[str, float]]]


def loo_predict(
    features: Sequence[Any], labels: Sequence[str], fit_predict: FitPredict
) -> list[tuple[str, dict[str, float], float]]:
    """For each document, ``fit_predict`` on all the others and predict it.

    Returns ``(label, probs, seconds)`` per document in input order; the held-out document is
    never part of its own training set.
    """
    out = []
    for i in range(len(features)):
        train_x = [x for j, x in enumerate(features) if j != i]
        train_y = [y for j, y in enumerate(labels) if j != i]
        start = time.perf_counter()
        label, probs = fit_predict(train_x, train_y, features[i])
        out.append((label, probs, time.perf_counter() - start))
    return out


def logistic_fit_predict(
    train_x: Sequence[Any], train_y: Sequence[str], test_x: Any
) -> tuple[str, dict[str, float]]:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=1000)
    clf.fit(np.asarray(train_x), list(train_y))
    proba = clf.predict_proba(np.asarray([test_x]))[0]
    probs = {str(c): float(p) for c, p in zip(clf.classes_, proba, strict=True)}
    return max(probs, key=lambda k: probs[k]), probs


# --- running systems ---------------------------------------------------------------------


def _top3(probs: Mapping[str, float]) -> dict[str, float]:
    return {k: round(v, 4) for k, v in sorted(probs.items(), key=lambda kv: -kv[1])[:3]}


def _warmup_doc() -> Document:
    return Document.from_sections(
        [{"title": "Introduction", "level": 1, "text": "This short note describes a product."}],
        source="warmup.txt",
    )


def _backend(system: System) -> Any:
    if system.is_jev:
        return jev_chunks.UsageJevBackend(JEV_MODEL, concurrency=JEV_CONCURRENCY)
    from docrooter.backends.registry import resolve_backend

    return resolve_backend(system.backend or "")


def run_docrooter(
    system: System,
    ids: Sequence[str],
    docs: Mapping[str, tuple[Document, float]],
    taxonomy: Taxonomy,
) -> dict[str, Any]:
    from docrooter.classify import Classifier
    from docrooter.classify.strategies import Flat
    from docrooter.doc_view import DocumentView

    backend = _backend(system)
    classifier = Classifier(
        taxonomy, backend, strategy=Flat(), instruction=INSTRUCTION, threshold=0.0
    )
    view = DocumentView()
    start = time.perf_counter()
    classifier.classify_documents([_warmup_doc()], view=view, mode=system.mode)
    load_s = time.perf_counter() - start
    if isinstance(backend, jev_chunks.UsageJevBackend):
        backend.usage.clear()
    log(f"{system.id}: warm-up (load) {load_s:.1f} s; classifying {len(ids)} documents")
    start = time.perf_counter()
    results = classifier.classify_documents([docs[i][0] for i in ids], view=view, mode=system.mode)
    batch_s = time.perf_counter() - start
    rows = []
    for doc_id, result in zip(ids, results, strict=True):
        row: dict[str, Any] = {
            "doc_id": doc_id,
            "pred": result.best.node_ids[-1] if result.best.node_ids else None,
            "confidence": round(result.confidence, 4),
            "leaf_probs_top3": _top3(result.leaf_probs),
            "parse_s": docs[doc_id][1],
            "classify_s": batch_s / len(ids),
        }
        if isinstance(backend, jev_chunks.UsageJevBackend):
            row["tokens"] = backend.tokens_for([result.view_text])
            row["tokens_estimated"] = False
            if row["tokens"] is None:
                row["tokens"] = jev_chunks.estimate_tokens(result.view_text)
                row["tokens_estimated"] = True
        rows.append(row)
    del classifier, backend
    free_memory()
    return {"rows": rows, "load_s": load_s, "batch_s": batch_s}


def run_jev_chunks(
    system: System,
    ids: Sequence[str],
    entries: Mapping[str, dict[str, Any]],
    taxonomy: Taxonomy,
    corpus: Path,
) -> dict[str, Any]:
    backend = _backend(system)
    rows = []
    for doc_id in ids:
        res = jev_chunks.classify(
            corpus / entries[doc_id]["file"], taxonomy, backend=backend, instruction=INSTRUCTION
        )
        rows.append(
            {
                "doc_id": doc_id,
                "pred": res.leaf,
                "confidence": round(res.probs.get(res.leaf, 0.0), 4) if res.leaf else 0.0,
                "leaf_probs_top3": _top3(res.probs),
                "parse_s": res.extract_s,
                "classify_s": res.classify_s,
                "tokens": res.n_tokens,
                "tokens_estimated": res.tokens_estimated,
                "n_chunks": res.n_chunks,
            }
        )
        log(
            f"jev-chunks {doc_id}: {res.leaf} ({res.n_chunks} chunks, {res.n_tokens} tokens, "
            f"{res.classify_s:.2f} s)"
        )
    return {"rows": rows, "load_s": 0.0}


def run_loo(
    system: System,
    ids: Sequence[str],
    docs: Mapping[str, tuple[Document, float]],
    gold: Mapping[str, str],
) -> dict[str, Any]:
    from docrooter.backends.embedding import EmbeddingBackend
    from docrooter.doc_view import DocumentView

    backend = EmbeddingBackend(ULTRABERT)
    prompt = backend.recipe.query_prompt.format(instruction=INSTRUCTION)
    start = time.perf_counter()
    count = backend.tokenizer()  # loads the model
    backend.encode(["warm-up"], prompt=prompt)
    load_s = time.perf_counter() - start
    log(f"{system.id}: load {load_s:.1f} s; embedding {len(ids)} views")
    start = time.perf_counter()
    view = DocumentView()
    texts = [view.render(docs[i][0], count).text for i in ids]
    features = backend.encode(texts, prompt=prompt)
    embed_s = time.perf_counter() - start
    del backend
    free_memory()
    preds = loo_predict(list(features), [gold[i] for i in ids], logistic_fit_predict)
    rows = [
        {
            "doc_id": doc_id,
            "pred": label,
            "confidence": round(probs[label], 4),
            "leaf_probs_top3": _top3(probs),
            "parse_s": docs[doc_id][1],
            "classify_s": embed_s / len(ids),
            "fit_s": seconds,
        }
        for doc_id, (label, probs, seconds) in zip(ids, preds, strict=True)
    ]
    return {"rows": rows, "load_s": load_s, "embed_s": embed_s}


def collect_env() -> dict[str, Any]:
    import importlib.metadata as md

    versions = {}
    for pkg in (
        "docrooter",
        "torch",
        "transformers",
        "sentence-transformers",
        "scikit-learn",
        "typesafe-sdk",
        "pymupdf",
    ):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "versions": versions,
    }


# --- Markdown report ---------------------------------------------------------------------

SECTION = "## Document-type classification"
BEGIN, END = "<!-- doctype-benchmark:begin -->", "<!-- doctype-benchmark:end -->"
CHECK, CROSS = "✓", "✗"


def _f(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _short(node_id: str | None) -> str:
    return node_id.split(".", 1)[-1] if node_id else NONE_LABEL


def _ordered(result: Mapping[str, Any]) -> list[str]:
    present = result.get("systems", {})
    return [s for s in SYSTEMS if s in present and present[s].get("metrics")]


def best_docrooter(result: Mapping[str, Any], *, zero_shot_only: bool) -> str | None:
    candidates = [
        s
        for s in _ordered(result)
        if not SYSTEMS[s].is_jev and not (zero_shot_only and SYSTEMS[s].kind == "loo")
    ]
    metrics = {s: result["systems"][s]["metrics"] for s in candidates}
    return max(
        candidates,
        key=lambda s: (metrics[s]["macro_f1"], metrics[s]["leaf_accuracy"]),
        default=None,
    )


def render_section(result: Mapping[str, Any], taxonomy: Taxonomy) -> str:
    systems = _ordered(result)
    env = result.get("env", {})
    out: list[str] = []
    leaves = [n.id for n in taxonomy.leaves()]
    n_docs = len(result.get("documents", []))
    counts = Counter(d["gold"] for d in result.get("documents", []))
    versions = env.get("versions", {})
    lo_n, hi_n = min(counts.values(), default=0), max(counts.values(), default=0)
    out.append(
        f"Corpus: {n_docs} real PDFs from `benchmarks/corpus/` with gold labels from the "
        f"manifest, {len(leaves)} leaf types in 6 groups (`builtin:doc_types`; every label has a "
        "definition, see `benchmarks/corpus/README.md`). Classes have "
        f"{lo_n}-{hi_n} documents each, so per-class numbers are anecdotal and one document "
        f"moves leaf accuracy by {100 / max(n_docs, 1):.1f} points. The 95% intervals are "
        f"percentile bootstraps over documents ({BOOTSTRAP_RESAMPLES} resamples, seed 0). Every "
        "system labels every document (threshold 0)."
    )
    out.append("")
    out.append(
        f"Environment: Python {env.get('python')}, {env.get('machine')}, "
        f"{env.get('cpu_count')} CPUs, device `{env.get('device')}` ({env.get('device_note')}), "
        f"torch {versions.get('torch')}, sentence-transformers "
        f"{versions.get('sentence-transformers')}, scikit-learn {versions.get('scikit-learn')}, "
        f"typesafe-sdk {versions.get('typesafe-sdk')}, pymupdf {versions.get('pymupdf')}, Jev "
        f"`{JEV_MODEL}`. The load average at the start of each system is listed below; the "
        "machine was shared with another benchmark, so seconds are noisy upper bounds."
    )
    out.append("")
    out.append(
        "Timing: *parse* is one `load_pdf` call with the default engine (pymupdf), cached and "
        "shared by every system except `jev-chunks`, whose parse is pymupdf `page.get_text()` "
        "over all pages. *Classify* for the `classify_documents` systems (`*-view`, "
        "`ultrabert-hybrid`) is the wall time of one call over all documents divided by the "
        "number of documents (DocumentView rendering included). For `ultrabert-loo` it is view "
        "rendering plus embedding divided by the number of documents; the logistic-regression "
        "fit per fold is listed separately. For `jev-chunks` it is the wall time of each "
        "document's concurrent chunk requests. *Load* is a warm-up excluded from the averages: "
        "model load plus one classification of a tiny synthetic document (the label vectors are "
        f"encoded there). Jev cost is input tokens x ${JEV_USD_PER_M_INPUT}/M (output tokens are "
        "free), with tokens as reported by the API (`usage.input_tokens`)."
    )
    out.append("")
    out.append("### Aggregate")
    out.append("")
    out.append(
        "| System | What | Leaf acc | 95% CI | Level-1 acc | Macro P | Macro R | Macro F1 | "
        "Parse s/doc | Classify s/doc | Total s/doc | Load s | Jev input tokens | Jev cost |"
    )
    out.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for s in systems:
        entry = result["systems"][s]
        m = entry["metrics"]
        lo, hi = m["leaf_accuracy_ci95"]
        tokens = m["jev_input_tokens"]
        tok = (
            "-"
            if tokens is None
            else f"{tokens:,}" + (" (est.)" if m["jev_tokens_estimated"] else "")
        )
        cost = "-" if m["jev_cost_usd"] is None else f"${m['jev_cost_usd']:.4f}"
        out.append(
            f"| `{s}` | {SYSTEMS[s].what} | {_f(m['leaf_accuracy'])} | {_f(lo, 2)}-{_f(hi, 2)} | "
            f"{_f(m['level1_accuracy'])} | {_f(m['macro_p'])} | {_f(m['macro_r'])} | "
            f"{_f(m['macro_f1'])} | {_f(m['avg_parse_s'], 2)} | {_f(m['avg_classify_s'], 3)} | "
            f"{_f(m['avg_total_s'], 2)} | {_f(entry.get('load_s'), 1)} | {tok} | {cost} |"
        )
    out.append("")
    out.append("Run metadata per system:")
    out.append("")
    for s in systems:
        entry = result["systems"][s]
        extra = ""
        if SYSTEMS[s].kind == "loo":
            fits = [r.get("fit_s", 0.0) for r in entry["rows"]]
            extra = f"; mean logistic-regression fit {sum(fits) / len(fits):.2f} s per fold"
        if s == "jev-chunks":
            chunks = [r.get("n_chunks", 0) for r in entry["rows"]]
            extra = f"; {sum(chunks)} chunks sent ({sum(chunks) / len(chunks):.1f} per document)"
        loadavg = ", ".join(f"{x:.1f}" for x in entry.get("loadavg_start", []))
        out.append(
            f"- `{s}`: {entry.get('date')} at `{entry.get('commit')}`, device "
            f"`{entry.get('device')}`, load average {loadavg} at start{extra}."
        )
    out.append("")

    shown = []
    for name in (
        best_docrooter(result, zero_shot_only=True),
        best_docrooter(result, zero_shot_only=False),
        "jev-view",
    ):
        if name and name in systems and name not in shown:
            shown.append(name)
    if shown:
        out.append("### Per class")
        out.append("")
        out.append(
            "Best docrooter system by macro F1 (zero-shot: `"
            + str(best_docrooter(result, zero_shot_only=True))
            + "`; any: `"
            + str(best_docrooter(result, zero_shot_only=False))
            + "`) and `jev-view`. P / R / F1 per gold class."
        )
        out.append("")
        out.append("| Class | n | " + " | ".join(f"`{s}` P / R / F1" for s in shown) + " |")
        out.append("| --- | --- | " + " | ".join("---" for _ in shown) + " |")
        classes = sorted({d["gold"] for d in result["documents"]}, key=leaves.index)
        for c in classes:
            cells = []
            for s in shown:
                pc = result["systems"][s]["metrics"]["per_class"][c]
                cells.append(f"{_f(pc['p'], 2)} / {_f(pc['r'], 2)} / {_f(pc['f1'], 2)}")
            n = result["systems"][shown[0]]["metrics"]["per_class"][c]["n"]
            out.append(f"| {c} | {n} | " + " | ".join(cells) + " |")
        out.append("")
        out.append("### Confusion matrices")
        out.append("")
        out.append(
            "Rows are gold, columns are predictions, numbered as the rows. Empty cells are 0; the "
            "diagonal is in bold."
        )
        for s in shown:
            out.append("")
            out.append(f"`{s}`:")
            out.append("")
            out.extend(_confusion_table(result["systems"][s]["metrics"]["confusion"], leaves))
        out.append("")

    out.append("### Per document")
    out.append("")
    out.append(
        f"{CHECK} correct leaf, {CROSS} wrong. Predictions show the leaf id without its group."
    )
    out.append("")
    out.append("| Doc | Title | Pages | Gold | " + " | ".join(f"`{s}`" for s in systems) + " |")
    out.append("| --- | --- | --- | --- | " + " | ".join("---" for _ in systems) + " |")
    by_system = {s: {r["doc_id"]: r for r in result["systems"][s]["rows"]} for s in systems}
    for d in result["documents"]:
        cells = []
        for s in systems:
            row = by_system[s].get(d["id"])
            if row is None:
                cells.append("")
                continue
            mark = CHECK if row["pred"] == d["gold"] else CROSS
            cells.append(f"{_short(row['pred'])} {mark}")
        title = d["title"] if len(d["title"]) <= 48 else d["title"][:45] + "..."
        out.append(
            f"| {d['id']} | {title.replace('|', '/')} | {d['pages']} | {_short(d['gold'])} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(out)


def _confusion_table(
    confusion: Mapping[str, Mapping[str, int]], leaves: Sequence[str]
) -> list[str]:
    index = {leaf: i + 1 for i, leaf in enumerate(leaves)}
    has_none = any(NONE_LABEL in row for row in confusion.values())
    cols = [*leaves, *([NONE_LABEL] if has_none else [])]
    head = "| Gold \\ Pred | " + " | ".join(str(index.get(c, "-")) for c in cols) + " |"
    lines = [head, "| --- | " + " | ".join("---" for _ in cols) + " |"]
    for leaf in leaves:
        row = confusion.get(leaf)
        if row is None:
            continue
        cells = []
        for c in cols:
            v = row.get(c, 0)
            cells.append("" if not v else (f"**{v}**" if c == leaf else str(v)))
        lines.append(f"| {index[leaf]} {_short(leaf)} | " + " | ".join(cells) + " |")
    return lines


def update_docs(path: Path, block: str) -> None:
    """Replace the generated block of the ``## Document-type classification`` section of
    ``path`` (appending the section when missing); everything else is kept."""
    wrapped = f"{BEGIN}\n{block}\n{END}"
    text = path.read_text(encoding="utf-8") if path.exists() else "# PDF benchmarks\n"
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        text = f"{head}{wrapped}{rest.split(END, 1)[1]}"
    elif SECTION in text:
        head, rest = text.split(SECTION, 1)
        text = f"{head}{SECTION}\n\n{wrapped}\n{rest}"
    else:
        text = f"{text.rstrip()}\n\n{SECTION}\n\n{wrapped}\n"
    path.write_text(text, encoding="utf-8")


# --- main --------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Real-PDF document-type benchmark")
    parser.add_argument("--systems", default=",".join(DEFAULT_ORDER))
    parser.add_argument("--out", type=Path, default=None, help="results dir (default: dated)")
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--only", default=None, help="comma-separated doc ids (smoke runs)")
    parser.add_argument("--device", default=None, help="skip the MPS probe and use this device")
    parser.add_argument("--docs", type=Path, default=None, help="also update this Markdown file")
    parser.add_argument("--report-only", action="store_true", help="render from existing pdf.json")
    args = parser.parse_args(argv)

    taxonomy = builtin_taxonomy("doc_types")
    out_dir = args.out or RESULTS_ROOT / datetime.date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pdf.json"
    if args.report_only:
        saved = json.loads(path.read_text(encoding="utf-8"))
        update_docs(args.docs or DOCS_PATH, render_section(saved, taxonomy))
        return

    load_env()  # silent: never prints names or values
    entries = manifest(args.corpus)
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        entries = [e for e in entries if e["id"] in wanted]
    by_id = {e["id"]: e for e in entries}
    ids = [e["id"] for e in entries]
    gold = {e["id"]: e["label"] for e in entries}
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = [s for s in systems if s not in SYSTEMS]
    if unknown:
        parser.error(f"unknown systems {unknown}; choose from {list(SYSTEMS)}")

    result: dict[str, Any] = (
        json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"systems": {}}
    )
    result["env"] = collect_env()
    result["documents"] = [
        {"id": e["id"], "title": e["title"], "pages": e["pages"], "gold": e["label"]}
        for e in entries
    ]
    if any(SYSTEMS[s].kind != "jev-chunks" and not SYSTEMS[s].is_jev for s in systems):
        if args.device:
            device, note = args.device, "set with --device"
        else:
            device, note = probe_device()
        os.environ["DOCROOTER_DEVICE"] = device
    else:
        device, note = "n/a", "API only"
    result["env"] |= {"device": device, "device_note": note}
    log(f"device {device}: {note}")

    log(f"parsing {len(ids)} PDFs (cached in {DOC_CACHE})")
    docs = parse_corpus("auto", entries=entries, corpus=args.corpus)
    result["parse"] = {
        "engine": "auto",
        "avg_parse_s": sum(docs[i][1] for i in ids) / len(ids),
        "per_doc_s": {i: round(docs[i][1], 4) for i in ids},
    }

    for name in systems:
        system = SYSTEMS[name]
        loadavg_start = list(os.getloadavg())
        log(f"== {name} (load average {loadavg_start[0]:.1f})")
        try:
            if system.kind == "docrooter":
                run = run_docrooter(system, ids, docs, taxonomy)
            elif system.kind == "jev-chunks":
                run = run_jev_chunks(system, ids, by_id, taxonomy, args.corpus)
            else:
                run = run_loo(system, ids, docs, gold)
        except Exception as exc:  # noqa: BLE001 - one failing system must not lose the others
            log(f"{name} FAILED: {type(exc).__name__}: {str(exc)[:500]}")
            result["systems"][name] = {"error": f"{type(exc).__name__}: {str(exc)[:500]}"}
            path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            free_memory()
            continue
        for row in run["rows"]:
            row["gold"] = gold[row["doc_id"]]
            row["correct"] = row["pred"] == row["gold"]
        run |= {
            "what": system.what,
            "backend": system.backend,
            "mode": system.mode,
            "device": "n/a" if system.is_jev else device,
            "date": datetime.date.today().isoformat(),
            "commit": git_revision(),
            "loadavg_start": loadavg_start,
            "loadavg_end": list(os.getloadavg()),
            "metrics": system_metrics(run["rows"], taxonomy),
        }
        result["systems"][name] = run
        m = run["metrics"]
        log(
            f"{name}: leaf acc {m['leaf_accuracy']:.3f}, macro F1 {m['macro_f1']:.3f}, "
            f"classify {m['avg_classify_s']:.3f} s/doc"
        )
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.docs:
        update_docs(args.docs, render_section(result, taxonomy))
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
