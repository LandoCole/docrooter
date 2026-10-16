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
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks import jev_chunks
from docrooter.document import Document
from docrooter.evaluate.metrics import leaf_prf
from docrooter.taxonomies import builtin_taxonomy
from docrooter.taxonomy import Taxonomy
from docrooter.views import RenderedView, estimate_tokens

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


LEAD_FIRST = ("title", "lead", "toc", "keyphrases", "stats")
TUNED_VIEW: dict[str, Any] = {"fields": LEAD_FIRST, "max_tokens": 1024, "lead_words": 400}
"""The best zero-shot DocumentView for UltraBERT on this corpus (P9b; chosen on these 51 docs)."""
BGE = "BAAI/bge-small-en-v1.5"
LOGISTIC_C_GRID = (1.0, 10.0, 100.0, 1000.0)
KNN_GRID = (1, 3, 5)
MIX_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class System:
    id: str
    kind: str  # "docrooter" (classify_documents), "jev-chunks" or "loo"
    what: str
    backend: str | None = None
    mode: str = "view"
    view: Mapping[str, Any] | None = None  # DocumentView(**view); None is the default view
    method: str | None = None  # "loo" systems: "logistic-cv" or "ensemble-cv"
    dev_tuned: bool = False  # settings chosen by looking at results on this corpus

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
            "ultrabert-view-tuned",
            "docrooter",
            "UltraBERT-Embed-150M, lead-first DocumentView (1,024 tokens, 400 lead words), Flat "
            "(zero-shot; view chosen on this corpus, dev-tuned)",
            f"embedding:{ULTRABERT}",
            view=TUNED_VIEW,
            dev_tuned=True,
        ),
        System(
            "ultrabert-loo",
            "loo",
            "UltraBERT-Embed-150M view embeddings + logistic regression, C in {1, 10, 100, 1000} "
            "by inner leave-one-out (trained, leave-one-out)",
            method="logistic-cv",
        ),
        System(
            "ultrabert-bge-loo",
            "loo",
            "Zero-shot UltraBERT-view and bge-view leaf distributions mixed, weight in "
            "{0, .25, .5, .75, 1} by inner leave-one-out (leave-one-out)",
            method="ensemble-cv",
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
    "ultrabert-view-tuned",
    "ultrabert-bge-loo",
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
        _SECRET_VALUES.add(value)
    return loaded


_SECRET_VALUES: set[str] = set()
"""Values set by :func:`load_env`; :func:`redact` removes them from any text."""
_KEY_LIKE = re.compile(r"apikey_\S+")
_SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET", re.IGNORECASE)
REDACTED = "[REDACTED]"


def redact(text: str, environ: Mapping[str, str] | None = None) -> str:
    """``text`` with key-like strings (``apikey_...``), every value :func:`load_env` set and
    every secret-looking environment value (name contains KEY, TOKEN or SECRET; at least 8
    characters) replaced by ``[REDACTED]``. Use it on exception text before it is logged or
    saved."""
    environ = os.environ if environ is None else environ
    secrets = set(_SECRET_VALUES)
    secrets |= {v for k, v in environ.items() if _SECRET_NAME.search(k)}
    for value in sorted((v for v in secrets if len(v) >= 8), key=len, reverse=True):
        text = text.replace(value, REDACTED)
    return _KEY_LIKE.sub(REDACTED, text)


def bench_revision(root: Path = ROOT) -> str | None:
    """Short HEAD commit, with ``+dirty`` when tracked files under ``src/`` or ``benchmarks/``
    (results excluded) have uncommitted changes, so a run of uncommitted code is marked."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True,
            check=True,
        ).stdout.strip()  # fmt: skip
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", "src", "benchmarks",
             ":(exclude)benchmarks/results"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{sha}+dirty" if dirty else sha


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


@dataclass(frozen=True)
class LooDoc:
    """One document's leave-one-out features.

    ``vec`` is the L2-normalised view embedding; ``zero_shot`` the zero-shot leaf distribution
    of the same model (all leaves, package temperature); ``other`` a second model's zero-shot
    distribution (for ensembles; empty when unused).
    """

    vec: Any
    zero_shot: Mapping[str, float]
    other: Mapping[str, float] = field(default_factory=dict)


def _argmax(probs: Mapping[str, float]) -> str:
    """The most probable label; ties go to the first in ``probs``' order."""
    return max(probs, key=lambda k: probs[k])


def _softmax(scores: Mapping[str, float], temperature: float) -> dict[str, float]:
    if not scores:
        return {}
    values = np.asarray(list(scores.values()), dtype=float) / temperature
    exp = np.exp(values - values.max())
    return {k: float(v) for k, v in zip(scores, exp / exp.sum(), strict=True)}


def _vecs(docs: Sequence[Any]) -> np.ndarray:
    return np.stack([np.asarray(d.vec if isinstance(d, LooDoc) else d, dtype=float) for d in docs])


def _unit(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norm > 0, norm, 1.0)


def logistic(c: float = 1.0) -> FitPredict:
    """Multinomial logistic regression on the view embeddings with inverse regularisation ``c``."""

    def fit_predict(
        train_x: Sequence[Any], train_y: Sequence[str], test_x: Any
    ) -> tuple[str, dict[str, float]]:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(C=c, max_iter=5000)
        clf.fit(_vecs(train_x), list(train_y))
        proba = clf.predict_proba(_vecs([test_x]))[0]
        probs = {str(k): float(p) for k, p in zip(clf.classes_, proba, strict=True)}
        return _argmax(probs), probs

    return fit_predict


logistic_fit_predict = logistic(1.0)
"""The P9 head: ``LogisticRegression(max_iter=...)`` at the default ``C=1``."""


def class_centroids(train_x: Sequence[Any], train_y: Sequence[str]) -> dict[str, np.ndarray]:
    """``{label: unit-length mean of its training embeddings}``, in first-seen label order."""
    vecs = _vecs(train_x)
    labels = list(dict.fromkeys(train_y))
    return {c: _unit(vecs[[y == c for y in train_y]].mean(axis=0)) for c in labels}


def centroid_probs(
    train_x: Sequence[Any], train_y: Sequence[str], test_x: Any, temperature: float
) -> dict[str, float]:
    """Softmax over the cosine similarity of ``test_x`` to each training class centroid."""
    vec = _vecs([test_x])[0]
    sims = {c: float(vec @ m) for c, m in class_centroids(train_x, train_y).items()}
    return _softmax(sims, temperature)


def centroid(temperature: float = 0.05) -> FitPredict:
    """Nearest class centroid (cosine); classes absent from training cannot be predicted."""

    def fit_predict(
        train_x: Sequence[Any], train_y: Sequence[str], test_x: Any
    ) -> tuple[str, dict[str, float]]:
        probs = centroid_probs(train_x, train_y, test_x, temperature)
        return _argmax(probs), probs

    return fit_predict


def knn(k: int) -> FitPredict:
    """Cosine k-nearest neighbours; each neighbour votes with its similarity."""

    def fit_predict(
        train_x: Sequence[Any], train_y: Sequence[str], test_x: Any
    ) -> tuple[str, dict[str, float]]:
        sims = _vecs(train_x) @ _vecs([test_x])[0]
        votes: dict[str, float] = {}
        for j in np.argsort(-sims, kind="stable")[:k]:
            votes[train_y[j]] = votes.get(train_y[j], 0.0) + max(float(sims[j]), 1e-9)
        total = sum(votes.values())
        probs = {c: v / total for c, v in votes.items()}
        return _argmax(probs), probs

    return fit_predict


def prior_centroid_mix(weight: float, temperature: float = 0.05) -> FitPredict:
    """``(1 - weight) * zero-shot + weight * centroid softmax`` (see ``mix_distributions``).

    The zero-shot distribution covers every leaf, so a class with no training document can still
    be predicted; the centroid part only covers classes seen in training.
    """
    from docrooter.classify.documents import mix_distributions

    def fit_predict(
        train_x: Sequence[LooDoc], train_y: Sequence[str], test_x: LooDoc
    ) -> tuple[str, dict[str, float]]:
        cent = centroid_probs(train_x, train_y, test_x, temperature) if weight > 0 else {}
        probs = mix_distributions(test_x.zero_shot, cent, view_weight=1.0 - weight)
        return _argmax(probs), probs

    return fit_predict


def ensemble(weight: float) -> FitPredict:
    """``(1 - weight) * zero_shot + weight * other``: two zero-shot models, no training."""
    from docrooter.classify.documents import mix_distributions

    def fit_predict(
        train_x: Sequence[LooDoc], train_y: Sequence[str], test_x: LooDoc
    ) -> tuple[str, dict[str, float]]:
        probs = mix_distributions(test_x.zero_shot, test_x.other, view_weight=1.0 - weight)
        return _argmax(probs), probs

    return fit_predict


def examples_prototype(label_vecs: Mapping[str, Any], temperature: float = 0.05) -> FitPredict:
    """The package's own few-shot mechanism (``TaxonomyNode.examples``): each leaf's vector is
    the unit mean of its label vector and its training documents' embeddings."""

    def fit_predict(
        train_x: Sequence[Any], train_y: Sequence[str], test_x: Any
    ) -> tuple[str, dict[str, float]]:
        vecs = _vecs(train_x) if len(train_x) else np.zeros((0, 0))
        test = _vecs([test_x])[0]
        sims = {}
        for c, label_vec in label_vecs.items():
            rows = [np.asarray(label_vec, dtype=float)]
            rows += [vecs[j] for j, y in enumerate(train_y) if y == c]
            sims[c] = float(test @ _unit(np.mean(rows, axis=0)))
        probs = _softmax(sims, temperature)
        return _argmax(probs), probs

    return fit_predict


def pick_config(
    preds: Sequence[Mapping[str, str]], labels: Sequence[str], configs: Sequence[str]
) -> str:
    """The config whose predictions in ``preds`` (one ``{config: label}`` per document) are most
    often right on ``labels``; ties go to the first in ``configs``."""
    hits = {c: sum(p[c] == y for p, y in zip(preds, labels, strict=True)) for c in configs}
    return max(configs, key=lambda c: hits[c])


class ConfigChoice:
    """A "fit" that picks the zero-shot config most accurate on the training documents and
    returns its prediction for the test document (features: ``{config: predicted label}``).
    Each call's chosen config is appended to ``chosen``."""

    def __init__(self, configs: Sequence[str]) -> None:
        self.configs = list(configs)
        self.chosen: list[str] = []

    def __call__(
        self,
        train_x: Sequence[Mapping[str, str]],
        train_y: Sequence[str],
        test_x: Mapping[str, str],
    ) -> tuple[str, dict[str, float]]:
        best = pick_config(train_x, train_y, self.configs)
        self.chosen.append(best)
        return test_x[best], {test_x[best]: 1.0}


config_choice = ConfigChoice


_NLL_FLOOR = 1e-6


class InnerLoo:
    """A ``FitPredict`` that tunes one hyperparameter by leave-one-out on its training set only.

    For each candidate value in ``grid`` (in order), ``make(value)`` is evaluated by inner
    leave-one-out over the training documents it is given: never the outer held-out document.
    The value with the best inner accuracy wins; ties go to the lower inner mean negative
    log-likelihood, then to the earlier value. The winner is refitted on all training documents.
    Each call's chosen value is appended to ``chosen``.
    """

    def __init__(self, make: Callable[[Any], FitPredict], grid: Sequence[Any]) -> None:
        if not grid:
            raise ValueError("grid must not be empty")
        self.make = make
        self.grid = list(grid)
        self.chosen: list[Any] = []

    def select(self, train_x: Sequence[Any], train_y: Sequence[str]) -> Any:
        if len(self.grid) == 1:
            return self.grid[0]
        best_key: tuple[float, float] | None = None
        best = self.grid[0]
        for value in self.grid:
            preds = loo_predict(train_x, train_y, self.make(value))
            acc = sum(p[0] == y for p, y in zip(preds, train_y, strict=True)) / len(train_y)
            nll = float(
                np.mean(
                    [
                        -np.log(max(p[1].get(y, 0.0), _NLL_FLOOR))
                        for p, y in zip(preds, train_y, strict=True)
                    ]
                )
            )
            key = (acc, -nll)
            if best_key is None or key > best_key:
                best_key, best = key, value
        return best

    def __call__(
        self, train_x: Sequence[Any], train_y: Sequence[str], test_x: Any
    ) -> tuple[str, dict[str, float]]:
        value = self.select(train_x, train_y)
        self.chosen.append(value)
        return self.make(value)(train_x, train_y, test_x)


# --- shared views and incremental results -----------------------------------------------

SHARED_COUNTER = "estimate_tokens"
"""Every ``*-view`` system and every embedding experiment sees the same view text, rendered once
per view config with ``docrooter.views.estimate_tokens`` (1.3 tokens per word) rather than
each backend's own tokenizer."""


def view_key(view: Mapping[str, Any] | None) -> str:
    """A stable name for a DocumentView config (``"default"`` for ``DocumentView()``)."""
    return json.dumps(dict(view or {}), sort_keys=True, default=list) if view else "default"


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


class SharedViews:
    """Each document's DocumentView, rendered once per view config with ``estimate_tokens``."""

    def __init__(self, ids: Sequence[str], docs: Mapping[str, tuple[Document, float]]) -> None:
        self.ids = list(ids)
        self.docs = docs
        self._cache: dict[str, dict[str, RenderedView]] = {}

    def rendered(self, view: Mapping[str, Any] | None = None) -> dict[str, RenderedView]:
        key = view_key(view)
        if key not in self._cache:
            from docrooter.doc_view import DocumentView

            doc_view = DocumentView(**(view or {}))
            self._cache[key] = {
                i: doc_view.render(self.docs[i][0], estimate_tokens) for i in self.ids
            }
        return self._cache[key]

    def texts(self, view: Mapping[str, Any] | None = None) -> list[str]:
        rendered = self.rendered(view)
        return [rendered[i].text for i in self.ids]

    def n_differing(self, view: Mapping[str, Any] | None, count: Callable[[str], int]) -> int:
        """How many documents' views would differ if rendered with ``count`` instead."""
        from docrooter.doc_view import DocumentView

        doc_view = DocumentView(**(view or {}))
        shared = self.rendered(view)
        return sum(doc_view.render(self.docs[i][0], count).text != shared[i].text for i in self.ids)

    def manifest(self) -> dict[str, Any]:
        """``{view key: {doc id: sha256[:12] of the shared text}}`` for ``pdf.json``."""
        return {
            key: {i: text_sha(r.text) for i, r in rendered.items()}
            for key, rendered in self._cache.items()
        }


def fixed_view(view: Mapping[str, Any] | None, rendered: Mapping[str, RenderedView]) -> Any:
    """A ``DocumentView`` whose ``render`` returns the pre-rendered text of a known document
    (matched by ``doc.source``) and ignores the token counter, so ``classify_documents`` sends
    exactly the shared text and keeps all its other behaviour. Unknown documents (the warm-up)
    are rendered normally with ``estimate_tokens``."""
    from docrooter.doc_view import DocumentView

    class FixedView(DocumentView):
        def render(
            self, doc: Document, count_tokens: Callable[[str], int] | None = None
        ) -> RenderedView:
            if doc.source in by_source:
                return by_source[doc.source]
            return super().render(doc, estimate_tokens)

    by_source: dict[str, RenderedView] = {}
    return FixedView(**(view or {})), by_source


class RowWriter:
    """One system's entry in ``result``, rewritten to ``path`` after every batch of rows.

    Rows already computed (and, for Jev, already paid for) survive a failure later in the
    run: the entry then keeps ``rows`` and gains ``error``.
    """

    def __init__(
        self, path: Path, result: dict[str, Any], name: str, gold: Mapping[str, str]
    ) -> None:
        self.path = path
        self.result = result
        self.gold = gold
        self.entry: dict[str, Any] = {"rows": [], "partial": True}
        result.setdefault("systems", {})[name] = self.entry

    def add(self, rows: Sequence[dict[str, Any]]) -> None:
        for row in rows:
            row["gold"] = self.gold[row["doc_id"]]
            row["correct"] = row["pred"] == row["gold"]
            self.entry["rows"].append(row)
        self.save()

    def fail(self, error: str) -> None:
        self.entry["error"] = redact(error)
        self.save()

    def save(self) -> None:
        self.path.write_text(
            json.dumps(self.result, indent=2, ensure_ascii=False), encoding="utf-8"
        )


AddRows = Callable[[Sequence[dict[str, Any]]], None]


def _no_rows(rows: Sequence[dict[str, Any]]) -> None:
    pass


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
    shared: SharedViews,
    add_rows: AddRows = _no_rows,
) -> dict[str, Any]:
    """``classify_documents`` on the shared view texts. Jev runs in chunks of
    ``JEV_CONCURRENCY`` documents so each chunk's rows are saved before the next is paid for;
    local backends run as one batch."""
    from docrooter.classify import Classifier
    from docrooter.classify.strategies import Flat

    backend = _backend(system)
    classifier = Classifier(
        taxonomy, backend, strategy=Flat(), instruction=INSTRUCTION, threshold=0.0
    )
    rendered = shared.rendered(system.view)
    view, by_source = fixed_view(system.view, rendered)
    start = time.perf_counter()
    classifier.classify_documents([_warmup_doc()], view=view, mode=system.mode)
    load_s = time.perf_counter() - start
    if isinstance(backend, jev_chunks.UsageJevBackend):
        backend.usage.clear()
    differing = shared.n_differing(system.view, classifier._token_counter())
    by_source.update({docs[i][0].source: rendered[i] for i in ids})
    log(
        f"{system.id}: warm-up (load) {load_s:.1f} s; {differing} of {len(ids)} views would "
        f"differ with the backend's own counter; classifying"
    )
    chunk = JEV_CONCURRENCY if system.is_jev else len(ids)
    rows: list[dict[str, Any]] = []
    batch_s = 0.0
    for lo in range(0, len(ids), chunk):
        part = list(ids[lo : lo + chunk])
        start = time.perf_counter()
        results = classifier.classify_documents(
            [docs[i][0] for i in part], view=view, mode=system.mode
        )
        part_s = time.perf_counter() - start
        batch_s += part_s
        new = []
        for doc_id, result in zip(part, results, strict=True):
            if result.view_text != rendered[doc_id].text:
                raise RuntimeError(f"{doc_id}: the classifier saw a different view text")
            row: dict[str, Any] = {
                "doc_id": doc_id,
                "pred": result.best.node_ids[-1] if result.best.node_ids else None,
                "confidence": round(result.confidence, 4),
                "leaf_probs_top3": _top3(result.leaf_probs),
                "parse_s": docs[doc_id][1],
                "classify_s": part_s / len(part),
                "view_sha": text_sha(result.view_text),
            }
            if isinstance(backend, jev_chunks.UsageJevBackend):
                row["tokens"] = backend.tokens_for([result.view_text])
                row["tokens_estimated"] = False
                if row["tokens"] is None:
                    row["tokens"] = jev_chunks.estimate_tokens(result.view_text)
                    row["tokens_estimated"] = True
            new.append(row)
        rows.extend(new)
        add_rows(new)
    del classifier, backend
    free_memory()
    return {
        "rows": rows,
        "load_s": load_s,
        "batch_s": batch_s,
        "view_counter": SHARED_COUNTER,
        "views_differing_from_own_counter": differing,
    }


def run_jev_chunks(
    system: System,
    ids: Sequence[str],
    entries: Mapping[str, dict[str, Any]],
    taxonomy: Taxonomy,
    corpus: Path,
    add_rows: AddRows = _no_rows,
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
        add_rows(rows[-1:])
        log(
            f"jev-chunks {doc_id}: {res.leaf} ({res.n_chunks} chunks, {res.n_tokens} tokens, "
            f"{res.classify_s:.2f} s)"
        )
    return {"rows": rows, "load_s": 0.0}


ZS_VIEWS: dict[str, dict[str, Any]] = {
    "default512": {},
    "lead1024": dict(TUNED_VIEW),
    "toc1024": {"max_tokens": 1024, "toc_depth": 3, "toc_entries": 80},
    "lead512": {"fields": LEAD_FIRST, "lead_words": 400},
    "default1024-lead400": {"max_tokens": 1024, "lead_words": 400},
}
"""DocumentView variants tried for zero-shot (P9b). ``default512`` is ``DocumentView()``."""
ZS_INSTRUCTIONS = {
    "what": INSTRUCTION,
    "identify": "Identify the type of document from its title, contents and opening text",
}
LABEL_STYLES = ("label-def", "def-first")
ZERO_SHOT_GRID: tuple[tuple[str, str, str], ...] = (
    *(
        (v, i, lab)
        for v in ("default512", "lead1024", "toc1024")
        for i in ZS_INSTRUCTIONS
        for lab in LABEL_STYLES
    ),
    *((v, "what", lab) for v in ("lead512", "default1024-lead400") for lab in LABEL_STYLES),
)
"""UltraBERT configs (view, instruction, label style): a 3x2x2 factorial, then two ablations of
the lead-first view (budget, then field order). The first is the P9 ``ultrabert-view`` setup."""
BGE_GRID: tuple[tuple[str, str, str], ...] = tuple(
    (v, "what", lab) for v in ("default512", "lead512") for lab in LABEL_STYLES
)
"""bge-small configs (its query prompt has no instruction slot, so only view and labels vary)."""


def label_text(taxonomy: Taxonomy, node_id: str, style: str) -> str:
    """A leaf's candidate text: ``"label-def"`` is what ``Flat`` sends (full label path, then the
    definition); ``"def-first"`` is the definition followed by the path in parentheses."""
    from docrooter.backends.base import candidate_text

    node = taxonomy.node(node_id)
    if style == "label-def":
        return candidate_text(node, taxonomy=taxonomy, full_path=True)
    if style == "def-first":
        path = " > ".join(n.label for n in taxonomy.path(node_id))
        return f"{node.description} ({path})" if node.description else path
    raise ValueError(f"unknown label style {style!r}")


class ZeroShotEmbedder:
    """View embeddings and zero-shot leaf distributions from one embedding model.

    Given the shared view texts, equivalent to ``classify_documents`` with ``Flat`` and the
    given instruction (same query prompt, label prompt and temperature), but it keeps the
    embeddings.
    """

    def __init__(self, model: str, taxonomy: Taxonomy) -> None:
        from docrooter.backends.embedding import EmbeddingBackend

        self.backend = EmbeddingBackend(model)
        self.taxonomy = taxonomy
        self.leaves = [n.id for n in taxonomy.leaves()]
        start = time.perf_counter()
        self.count = self.backend.tokenizer()  # loads the model
        self.backend.encode(["warm-up"], prompt="")
        self.load_s = time.perf_counter() - start

    def label_vecs(self, style: str) -> np.ndarray:
        prefix = self.backend.recipe.label_prompt
        texts = [prefix + label_text(self.taxonomy, leaf, style) for leaf in self.leaves]
        return self.backend.encode(texts, prompt="")

    def embed(self, texts: Sequence[str], instruction: str) -> np.ndarray:
        prompt = self.backend.recipe.query_prompt.format(instruction=instruction)
        return self.backend.encode(list(texts), prompt=prompt)

    def distributions(self, vecs: np.ndarray, label_vecs: np.ndarray) -> list[dict[str, float]]:
        from docrooter.backends.base import Scores, to_probs

        probs = to_probs(
            Scores(vecs @ label_vecs.T, "similarity"),
            temperature=self.backend.temperature,
            multi_label=False,
        )
        return [dict(zip(self.leaves, map(float, row), strict=True)) for row in probs]


def _loo_rows(
    ids: Sequence[str],
    docs: Mapping[str, tuple[Document, float]],
    preds: Sequence[tuple[str, dict[str, float], float]],
    classify_s: float,
) -> list[dict[str, Any]]:
    return [
        {
            "doc_id": doc_id,
            "pred": label,
            "confidence": round(probs.get(label, 0.0), 4),
            "leaf_probs_top3": _top3(probs),
            "parse_s": docs[doc_id][1],
            "classify_s": classify_s,
            "fit_s": seconds,
        }
        for doc_id, (label, probs, seconds) in zip(ids, preds, strict=True)
    ]


def run_loo(
    system: System,
    ids: Sequence[str],
    docs: Mapping[str, tuple[Document, float]],
    gold: Mapping[str, str],
    taxonomy: Taxonomy,
    shared: SharedViews,
    add_rows: AddRows = _no_rows,
) -> dict[str, Any]:
    """A leave-one-out system: features from the shared default DocumentView (as P9), then
    ``system.method`` with its hyperparameter tuned by inner leave-one-out."""
    labels = [gold[i] for i in ids]
    models = [ULTRABERT] + ([BGE] if system.method == "ensemble-cv" else [])
    load_s, embed_s = 0.0, 0.0
    vecs: dict[str, np.ndarray] = {}
    dists: dict[str, list[dict[str, float]]] = {}
    differing: dict[str, int] = {}
    for model in models:
        embedder = ZeroShotEmbedder(model, taxonomy)
        load_s += embedder.load_s
        differing[model] = shared.n_differing(None, embedder.count)
        log(f"{system.id}: {model} load {embedder.load_s:.1f} s; embedding {len(ids)} views")
        start = time.perf_counter()
        vecs[model] = embedder.embed(shared.texts(None), INSTRUCTION)
        dists[model] = embedder.distributions(vecs[model], embedder.label_vecs("label-def"))
        embed_s += time.perf_counter() - start
        del embedder
        free_memory()
    features = [
        LooDoc(vecs[ULTRABERT][k], dists[ULTRABERT][k], dists[BGE][k] if BGE in dists else {})
        for k in range(len(ids))
    ]
    if system.method == "logistic-cv":
        fit_predict = InnerLoo(logistic, LOGISTIC_C_GRID)
    elif system.method == "ensemble-cv":
        fit_predict = InnerLoo(ensemble, MIX_GRID)
    else:
        raise ValueError(f"unknown leave-one-out method {system.method!r}")
    preds = loo_predict(features, labels, fit_predict)
    rows = _loo_rows(ids, docs, preds, embed_s / len(ids))
    view_text = shared.rendered(None)
    for row, value in zip(rows, fit_predict.chosen, strict=True):
        row["chosen"] = value
        row["view_sha"] = text_sha(view_text[row["doc_id"]].text)
    add_rows(rows)
    return {
        "rows": rows,
        "load_s": load_s,
        "embed_s": embed_s,
        "chosen": dict(Counter(str(v) for v in fit_predict.chosen)),
        "view_counter": SHARED_COUNTER,
        "views_differing_from_own_counter": differing,
    }


def _experiment(
    exp_id: str,
    group: str,
    what: str,
    ids: Sequence[str],
    labels: Sequence[str],
    preds: Sequence[str],
    taxonomy: Taxonomy,
    chosen: Sequence[Any] | None = None,
) -> dict[str, Any]:
    rows = [{"doc_id": i, "gold": g, "pred": p} for i, g, p in zip(ids, labels, preds, strict=True)]
    m = system_metrics(rows, taxonomy)
    keep = ("n", "leaf_accuracy", "leaf_accuracy_ci95", "level1_accuracy", "macro_f1")
    out: dict[str, Any] = {
        "id": exp_id,
        "group": group,
        "what": what,
        "metrics": {k: m[k] for k in keep},
        "preds": dict(zip(ids, preds, strict=True)),
    }
    if chosen is not None:
        out["chosen"] = dict(Counter(str(v) for v in chosen))
    return out


EXP_ZERO_SHOT = "zero-shot, dev-tuned"
EXP_LOO = "leave-one-out (honest)"
EXP_DIAGNOSTIC = "diagnostic (fixed hyperparameter, not tuned honestly)"


def run_experiments(
    ids: Sequence[str],
    docs: Mapping[str, tuple[Document, float]],
    gold: Mapping[str, str],
    taxonomy: Taxonomy,
    shared: SharedViews,
) -> list[dict[str, Any]]:
    """P9b: the zero-shot grids (every config evaluated on all documents, so any choice among
    them is dev-tuned) and the leave-one-out systems (honest: each held-out document's
    prediction never depends on its own label)."""
    labels = [gold[i] for i in ids]
    leaves = [n.id for n in taxonomy.leaves()]
    out: list[dict[str, Any]] = []
    zs_preds: dict[str, list[str]] = {}
    feats: dict[str, dict[str, Any]] = {}
    differing: dict[str, int] = {}
    for model, grid in ((ULTRABERT, ZERO_SHOT_GRID), (BGE, BGE_GRID)):
        short = "ultrabert" if model == ULTRABERT else "bge"
        embedder = ZeroShotEmbedder(model, taxonomy)
        label_vecs = {style: embedder.label_vecs(style) for style in LABEL_STYLES}
        texts = {}
        for view, instr, style in grid:
            if view not in texts:
                texts[view] = shared.texts(ZS_VIEWS[view])
                differing[f"{short}/{view}"] = shared.n_differing(ZS_VIEWS[view], embedder.count)
            vecs = embedder.embed(texts[view], ZS_INSTRUCTIONS[instr])
            dists = embedder.distributions(vecs, label_vecs[style])
            preds = [_argmax(d) for d in dists]
            exp_id = f"{short}/{view}/{instr}/{style}"
            zs_preds[exp_id] = preds
            feats[exp_id] = {"vecs": vecs, "dists": dists}
            what = f"{short}, view {view}, instruction {instr}, labels {style}"
            out.append(_experiment(exp_id, EXP_ZERO_SHOT, what, ids, labels, preds, taxonomy))
            out[-1]["views_differing_from_own_counter"] = differing[f"{short}/{view}"]
            log(f"experiment {exp_id}: {out[-1]['metrics']['leaf_accuracy']:.3f}")
        if model == ULTRABERT:
            feats["ultrabert/label-vecs"] = dict(zip(leaves, label_vecs["label-def"], strict=True))
        del embedder
        free_memory()

    base = feats[f"ultrabert/{'/'.join(ZERO_SHOT_GRID[0])}"]
    other = feats[f"bge/{'/'.join(BGE_GRID[0])}"]["dists"]
    features = [LooDoc(base["vecs"][k], base["dists"][k], other[k]) for k in range(len(ids))]

    def loo(exp_id: str, group: str, what: str, fit_predict: FitPredict, xs: Any = None) -> None:
        preds = loo_predict(features if xs is None else xs, labels, fit_predict)
        chosen = getattr(fit_predict, "chosen", None)
        out.append(
            _experiment(exp_id, group, what, ids, labels, [p[0] for p in preds], taxonomy, chosen)
        )
        log(f"experiment {exp_id}: {out[-1]['metrics']['leaf_accuracy']:.3f}")

    ub_configs = [f"ultrabert/{'/'.join(c)}" for c in ZERO_SHOT_GRID]
    loo(
        "zero-shot-config-loo",
        EXP_LOO,
        f"zero-shot UltraBERT config picked from the {len(ub_configs)} above by accuracy on the "
        "other 50 documents (honest estimate of the dev-tuned choice)",
        config_choice(ub_configs),
        [{c: zs_preds[c][k] for c in ub_configs} for k in range(len(ids))],
    )
    loo(
        "logistic-cv",
        EXP_LOO,
        "logistic regression, C in {1, 10, 100, 1000} by inner LOO",
        InnerLoo(logistic, LOGISTIC_C_GRID),
    )
    loo("centroid", EXP_LOO, "nearest class centroid (cosine)", centroid())
    loo("knn-cv", EXP_LOO, "cosine kNN, k in {1, 3, 5} by inner LOO", InnerLoo(knn, KNN_GRID))
    loo(
        "prior-centroid-cv",
        EXP_LOO,
        "(1 - w) zero-shot + w centroid softmax, w in {0, .25, .5, .75, 1} by inner LOO",
        InnerLoo(prior_centroid_mix, MIX_GRID),
    )
    loo(
        "examples",
        EXP_LOO,
        "package examples mechanism: leaf vector = unit mean of label vector and training views",
        examples_prototype(feats["ultrabert/label-vecs"]),
    )
    loo(
        "ensemble-cv",
        EXP_LOO,
        "(1 - w) UltraBERT zero-shot + w bge zero-shot, w in {0, .25, .5, .75, 1} by inner LOO",
        InnerLoo(ensemble, MIX_GRID),
    )
    for c in LOGISTIC_C_GRID:
        loo(f"logistic-C{c:g}", EXP_DIAGNOSTIC, f"logistic regression, fixed C={c:g}", logistic(c))
    for k in KNN_GRID:
        loo(f"knn-k{k}", EXP_DIAGNOSTIC, f"cosine kNN, fixed k={k}", knn(k))
    for w in MIX_GRID[1:-1]:
        loo(f"ensemble-w{w:g}", EXP_DIAGNOSTIC, f"ensemble, fixed bge weight {w:g}", ensemble(w))
    return out


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


def best_docrooter(
    result: Mapping[str, Any], *, zero_shot_only: bool, dev_tuned: bool = False
) -> str | None:
    """The local (non-Jev) system with the best macro F1, then leaf accuracy. Honest systems
    only by default; ``dev_tuned=True`` picks among the dev-tuned ones instead. With
    ``zero_shot_only`` leave-one-out systems are excluded."""
    candidates = [
        s
        for s in _ordered(result)
        if not SYSTEMS[s].is_jev
        and SYSTEMS[s].dev_tuned == dev_tuned
        and not (zero_shot_only and SYSTEMS[s].kind == "loo")
    ]
    metrics = {s: result["systems"][s]["metrics"] for s in candidates}
    return max(
        candidates,
        key=lambda s: (metrics[s]["macro_f1"], metrics[s]["leaf_accuracy"]),
        default=None,
    )


def _differing(entry: Mapping[str, Any]) -> str:
    """``"; views differing ..."`` for a run that recorded it, else ``""``."""
    diff = entry.get("views_differing_from_own_counter")
    if diff is None:
        return ""
    if isinstance(diff, Mapping):
        parts = ", ".join(f"{_model_short(k)} {v}" for k, v in diff.items())
        return f"; views differing from the backends' own counters: {parts} of {len(entry['rows'])}"
    return f"; views differing from the backend's own counter: {diff} of {len(entry['rows'])}"


def _model_short(model: str) -> str:
    return {ULTRABERT: "UltraBERT", BGE: "bge"}.get(model, model)


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
        "`ultrabert-hybrid`) is batch throughput: the wall time of the classify calls divided "
        "by the number of documents (local backends: one call over all documents; Jev: chunks "
        f"of {JEV_CONCURRENCY} documents with {JEV_CONCURRENCY} concurrent requests, so each "
        "chunk's rows are saved before the next is paid for). For the leave-one-out systems "
        "(`*-loo`) it is view rendering, embedding and zero-shot scoring divided by the number "
        "of documents; the fit per fold (with its inner leave-one-out tuning) is listed "
        "separately. For `jev-chunks` it is per-document latency: the wall time of each "
        "document's concurrent chunk requests. The table marks each figure as `(batch)` or "
        "`(latency)`. *Load* is a warm-up excluded from the averages: "
        "model load plus one classification of a tiny synthetic document (the label vectors are "
        f"encoded there). Jev cost is input tokens x ${JEV_USD_PER_M_INPUT}/M (output tokens are "
        "free), with tokens as reported by the API (`usage.input_tokens`)."
    )
    out.append("")
    out.append(
        "Views: every `*-view` system, `ultrabert-hybrid` (its view half), the leave-one-out "
        "systems and the P9b embedding experiments receive the same DocumentView text per "
        "document and view config, rendered once with the shared word-based counter "
        "(`docrooter.views.estimate_tokens`, 1.3 tokens per word) instead of each backend's own "
        "tokenizer. Each row records the SHA-256 prefix of the text the system saw "
        "(`view_sha`), and `pdf.json` lists the shared texts' hashes under `views`. Whatever a "
        "model's own tokenizer cannot fit is cut by that model as usual (bge-small and the NLI "
        "model at 512 tokens). For each system, *Views differing* counts the documents whose "
        "view would have been different had it been rendered with the backend's own counter, "
        "which was the P9 behaviour."
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
            f"{_f(m['macro_f1'])} | {_f(m['avg_parse_s'], 2)} | "
            f"{_f(m['avg_classify_s'], 3)} {'(latency)' if s == 'jev-chunks' else '(batch)'} | "
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
            extra = f"; mean fit {sum(fits) / len(fits):.2f} s per fold (inner LOO included)"
            if entry.get("chosen"):
                picks = ", ".join(f"{k} x{v}" for k, v in entry["chosen"].items())
                extra += f"; hyperparameter chosen per fold: {picks}"
        if s == "jev-chunks":
            chunks = [r.get("n_chunks", 0) for r in entry["rows"]]
            extra = f"; {sum(chunks)} chunks sent ({sum(chunks) / len(chunks):.1f} per document)"
        loadavg = ", ".join(f"{x:.1f}" for x in entry.get("loadavg_start", []))
        out.append(
            f"- `{s}`: {entry.get('date')} at `{entry.get('commit')}`, device "
            f"`{entry.get('device')}`, load average {loadavg} at start{extra}{_differing(entry)}."
        )
    out.append("")

    out.extend(render_experiments(result))

    honest_zs = best_docrooter(result, zero_shot_only=True)
    honest_any = best_docrooter(result, zero_shot_only=False)
    tuned = best_docrooter(result, zero_shot_only=False, dev_tuned=True)
    shown = []
    for name in (honest_zs, honest_any, "jev-view", tuned):
        if name and name in systems and name not in shown:
            shown.append(name)
    headers = {s: f"`{s}`" + (" (dev-tuned)" if SYSTEMS[s].dev_tuned else "") for s in shown}
    if shown:
        out.append("### Per class")
        out.append("")
        out.append(
            f"Best honest docrooter system by macro F1 (zero-shot: `{honest_zs}`; any, "
            f"leave-one-out included: `{honest_any}`), `jev-view`"
            + (
                f", and the best dev-tuned system `{tuned}` (its settings were chosen on this "
                "corpus, so its numbers are optimistic)"
                if tuned in shown
                else ""
            )
            + ". P / R / F1 per gold class."
        )
        out.append("")
        out.append("| Class | n | " + " | ".join(f"{headers[s]} P / R / F1" for s in shown) + " |")
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
            out.append(f"{headers[s]}:")
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


def render_experiments(result: Mapping[str, Any]) -> list[str]:
    """The ``### Configurations tried (P9b)`` tables, one per experiment group (empty when the
    result has no experiments)."""
    exps = result.get("experiments") or {}
    runs = exps.get("runs") or []
    if not runs:
        return []
    out = [
        "### Configurations tried (P9b)",
        "",
        f"Every configuration evaluated in P9b ({exps.get('date')} at `{exps.get('commit')}`, "
        f"device `{exps.get('device')}`), with its leaf accuracy, 95% bootstrap CI and macro F1 "
        "on the same 51 documents. *Zero-shot, dev-tuned* configs need no labels, but each was "
        "scored on this corpus, so the best of them is an optimistic estimate. *Leave-one-out "
        "(honest)* systems predict each document from the other 50 only; every hyperparameter is "
        "chosen by an inner leave-one-out over those 50 (ties: lower inner log-loss, then the "
        "first value), so the held-out document never influences its own prediction. Their "
        "features are the default `DocumentView()` embeddings with the P9 instruction, fixed "
        "before P9b. *Diagnostic* rows fix a hyperparameter and are shown only to explain the "
        "tuned ones. All of them use the shared `estimate_tokens` view texts (see *Views* above).",
    ]
    groups = list(dict.fromkeys(r["group"] for r in runs))
    for group in groups:
        out += ["", f"**{group[0].upper()}{group[1:]}**", ""]
        out.append("| Config | What | Leaf acc | 95% CI | Level-1 acc | Macro F1 | Chosen |")
        out.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in runs:
            if r["group"] != group:
                continue
            m = r["metrics"]
            lo, hi = m["leaf_accuracy_ci95"]
            chosen = ", ".join(
                f"{k}: {v}" for k, v in sorted(r.get("chosen", {}).items(), key=lambda kv: -kv[1])
            )
            out.append(
                f"| `{r['id']}` | {r['what']} | {_f(m['leaf_accuracy'])} | "
                f"{_f(lo, 2)}-{_f(hi, 2)} | {_f(m['level1_accuracy'])} | {_f(m['macro_f1'])} | "
                f"{chosen or '-'} |"
            )
    out.append("")
    return out


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
    parser.add_argument(
        "--experiments",
        action="store_true",
        help="also run the P9b zero-shot grid and leave-one-out experiments",
    )
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
    local = args.experiments or any(
        SYSTEMS[s].kind != "jev-chunks" and not SYSTEMS[s].is_jev for s in systems
    )
    if local:
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

    shared = SharedViews(ids, docs)
    for name in systems:
        system = SYSTEMS[name]
        loadavg_start = list(os.getloadavg())
        commit = bench_revision()
        log(f"== {name} (load average {loadavg_start[0]:.1f})")
        writer = RowWriter(path, result, name, gold)
        try:
            if system.kind == "docrooter":
                run = run_docrooter(system, ids, docs, taxonomy, shared, writer.add)
            elif system.kind == "jev-chunks":
                run = run_jev_chunks(system, ids, by_id, taxonomy, args.corpus, writer.add)
            else:
                run = run_loo(system, ids, docs, gold, taxonomy, shared, writer.add)
        except Exception as exc:  # noqa: BLE001 - one failing system must not lose the others
            error = redact(f"{type(exc).__name__}: {str(exc)[:500]}")
            log(f"{name} FAILED after {len(writer.entry['rows'])} rows: {error}")
            writer.fail(error)
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
            "commit": commit,
            "loadavg_start": loadavg_start,
            "loadavg_end": list(os.getloadavg()),
            "metrics": system_metrics(run["rows"], taxonomy),
        }
        result["systems"][name] = run
        result["views"] = {"counter": SHARED_COUNTER, "sha256_12": shared.manifest()}
        m = run["metrics"]
        log(
            f"{name}: leaf acc {m['leaf_accuracy']:.3f}, macro F1 {m['macro_f1']:.3f}, "
            f"classify {m['avg_classify_s']:.3f} s/doc"
        )
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.experiments:
        log("== experiments (P9b)")
        result["experiments"] = {
            "date": datetime.date.today().isoformat(),
            "commit": bench_revision(),
            "device": device,
            "view_counter": SHARED_COUNTER,
            "runs": run_experiments(ids, docs, gold, taxonomy, shared),
        }
        result["views"] = {"counter": SHARED_COUNTER, "sha256_12": shared.manifest()}
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.docs:
        update_docs(args.docs, render_section(result, taxonomy))
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
