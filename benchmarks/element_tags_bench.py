"""Element tagger evaluation on the independently annotated gold set (spec §11, §4.0 ruling 4;
task L11).

**Gold.** ``benchmarks/training_corpus/gold.jsonl`` holds keys and labels only (900 elements,
Claude-annotated, disclosed as such). Each element's text is read from the corpus store's
``gold`` split by ``(doc_id, element_id)`` and its ``text_sha1`` is checked: a missing element,
a document outside the gold split or a changed text stops the run (:func:`gold_items`).

**Systems** (``run --system NAME``; ``--tagger SPEC`` for a name not in :data:`SYSTEMS`):

- ``keyword``: keyword overlap (``KeywordBackend``) through L10's zero-shot tagger;
- ``zero-shot``: ``embedding:dmis-lab/UltraBERT-Embed-150M``, L10's ``EmbeddingTagger``;
- ``zero-shot-enriched``: the same with each topic leaf's mined keywords appended to its
  definition (:class:`EnrichedEmbeddingTagger`; ``--enriched``, by default the store's
  ``train-elements/element_topics.enriched.json``);
- ``teacher-4b`` and ``teacher-8b``: L10's ``TeacherTagger`` on the pinned MLX models; the 8B
  runs on a 300-element subset stratified by family (:func:`gold_subset`);
- students: ``--tagger <bundle dir>``, under any name (``stage-a``, ``stage-b``,
  ``stage-b-al``): ``ElementTagger.tag()``.

**Modes** (§4.0 ruling 4). *Primary*: the family comes from the gold document's known type
(the gold row's ``family``); the targets apply here. *End to end*: the family of the rooter's
predicted doc type, from ``gold-doctypes`` (``--doc-types``). Elements whose predicted family
is their gold family reuse the primary tags.

**Metrics.**

- topic: accuracy and macro P/R/F1 over the classes present in gold (overall, and per gold
  family over that family's gold classes), so the class set depends on the gold set only and is
  the same for every system; a prediction outside it counts as a miss of its gold class, never
  as a class of its own (``macro_over: "gold_classes"``). Hierarchical P/R/F1 over family ->
  leaf (each label's ancestor set without the root; a family-level prediction earns the family
  node). The headline is the overall macro F1 in primary mode;
- function: micro and macro F1 and per-tag P/R/F1 over the 12 tags. A tag nobody labelled or
  predicted has no F1 and is left out of the macro (``tags_scored`` says how many count);
- calibration: ECE over :data:`ECE_BINS` equal-width bins of the top-label confidence (topic:
  the reported leaf's probability; function: max(p, 1 - p) of each tag's P(yes) decision).
  A student is scored before (temperatures 1.0) and after its calibrated temperatures, its
  embeddings computed once.

**Throughput and memory.** elements/s over the primary pass after a one-element warm-up (the
load); for a teacher, also the items it was asked in that pass, counted, per second; and the
process's peak footprint (:mod:`benchmarks.memory_check`). One system per process, under
:mod:`benchmarks._guard`.

**No source hints.** Every PDF a model reads (``gold-doctypes``, ``doctype-evidence``) is
passed as a neutral copy, ``doc_<sha256[:12]>.pdf`` in a temporary directory
(:func:`neutral_copies`): the doc-type view's title falls back to the file stem, and gold
document ids name their source, hence their family.

**Results** go to ``benchmarks/results/<date>/element-tags*.json`` with provenance: commit,
gold sha256, bundle sha256, the evaluated teacher's name and dtype, and for a student the
labelling teacher's engine name(s) with dtype, the label files' sha256 and the confident set's
fingerprint (:func:`training_provenance`), doc-type file sha256. No element text is written.

Commands (each loads at most one model set; run them one at a time)::

    PYTHONPATH=src .venv/bin/python -m benchmarks.element_tags_bench gold-doctypes
    PYTHONPATH=src .venv/bin/python -m benchmarks.element_tags_bench run --system keyword \\
        --doc-types benchmarks/results/<date>/element-tags-gold-doctypes.json
    PYTHONPATH=src .venv/bin/python -m benchmarks.element_tags_bench run --system stage-b \\
        --tagger benchmarks/.cache/bundles/element-tagger-v1 --device mps --doc-types ...
    PYTHONPATH=src .venv/bin/python -m benchmarks.element_tags_bench learning-curve --stage A
    PYTHONPATH=src .venv/bin/python -m benchmarks.element_tags_bench doctype-evidence \\
        --bundle benchmarks/.cache/bundles/element-tagger-v1

``doctype-evidence`` scores the dev corpus only (the held-out corpus is not run in v0.5): the
rooter's own doc type (``best``, with its abstention: the v0.3/v0.4 baseline), and the fixed
mix ``0.7 · view + 0.3 · element evidence`` (:func:`mix_doc_type`), never tuned, of the view
renormalised over the doc-type leaves it holds (``leaf_probs`` keeps the top 10 nodes only).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks import _guard
from docrooter.tagger.element_tags import EmbeddingTagger

BENCHMARKS = Path(__file__).parent
GOLD_PATH = BENCHMARKS / "training_corpus" / "gold.jsonl"
STORE = BENCHMARKS / ".cache" / "corpus_store"
GOLD_PDFS = BENCHMARKS / ".cache" / "training_corpus" / "pdfs"
BUNDLES = BENCHMARKS / ".cache" / "bundles"
RESULTS_ROOT = BENCHMARKS / "results"
WORK_DIR = "train-elements"  # docrooter.tagger.pipeline.WORK_DIR
ENRICHED_FILE = "element_topics.enriched.json"  # docrooter.tagger.pipeline.ENRICHED_FILE

RESULTS_FILE = "element-tags.json"
GOLD_DOCTYPES_FILE = "element-tags-gold-doctypes.json"
CURVE_FILE = "element-tags-learning-curve.json"
EVIDENCE_FILE = "element-tags-doctype-evidence.json"
DISCLOSURE = "Claude-annotated gold"
GOLD_FIELDS = (
    "doc_id",
    "element_id",
    "text_sha1",
    "family",
    "topic",
    "functions",
    "annotator",
    "adjudicated",
)

ECE_BINS = 15
VIEW_WEIGHT = 0.7  # spec §11: fixed, never tuned on dev
ELEMENT_WEIGHT = 0.3
LEARNING_CURVE_FRACTIONS = (0.125, 0.25, 0.5, 1.0)
TEACHER_SUBSET = 300
FUNCTION_THRESHOLD = 0.5  # docrooter.tagger.bundle.FUNCTION_THRESHOLD
MACRO_OVER = "gold_classes"

ULTRABERT = "dmis-lab/UltraBERT-Embed-150M"
TEACHER_4B = (
    "mlx:mlx-community/Qwen3-4B-Instruct-2507-4bit@50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
)
TEACHER_8B = "mlx:mlx-community/Qwen3-8B-4bit@545dc4251c05440727734bcd94334791f6ab0192"
KEYWORD = "keyword"
ENRICHED_PREFIX = "enriched:"
SYSTEMS: dict[str, str] = {
    "keyword": KEYWORD,
    "zero-shot": f"embedding:{ULTRABERT}",
    "zero-shot-enriched": f"{ENRICHED_PREFIX}{ULTRABERT}",
    "teacher-4b": f"teacher:{TEACHER_4B}",
    "teacher-8b": f"teacher:{TEACHER_8B}",
}
"""Systems with a fixed spec; a student is ``--tagger <bundle dir>`` under its own name."""
SYSTEM_SUBSETS: dict[str, int] = {"teacher-8b": TEACHER_SUBSET}
GOLD_DOCTYPE_PROFILE = "balanced"  # the rooter's default profile
EVIDENCE_PROFILE = "accurate"  # heron elements, as the corpus (and so the student) was built
GB = 1024**3


def log(message: str) -> None:
    print(f"[{datetime.datetime.now():%H:%M:%S}] {message}", flush=True)


# --- gold ----------------------------------------------------------------------------------------


class GoldTextError(ValueError):
    """Gold rows that cannot be joined to their text; the message lists every problem."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        shown = "\n  ".join(self.errors[:20])
        more = f"\n  (+{len(self.errors) - 20} more)" if len(self.errors) > 20 else ""
        super().__init__(f"{len(self.errors)} problem(s) with the gold set:\n  {shown}{more}")


@dataclass(frozen=True)
class GoldItem:
    """One gold element: its labels and its corpus element (text, heading path, kind)."""

    doc_id: str
    element_id: str
    family: str
    topic: str
    functions: tuple[str, ...]
    element: Any  # docrooter.corpus.CorpusElement


def read_gold(path: str | os.PathLike[str] = GOLD_PATH) -> list[dict[str, Any]]:
    """The gold rows of ``path``, each checked for the :data:`GOLD_FIELDS`."""
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            errors.append(f"{path} line {number}: not JSON: {exc}")
            continue
        if not isinstance(row, dict):
            errors.append(f"{path} line {number}: not a JSON object")
            continue
        missing = [name for name in GOLD_FIELDS if name not in row]
        if missing:
            errors.append(f"{path} line {number}: missing {missing}")
            continue
        rows.append(row)
    if errors:
        raise GoldTextError(errors)
    return rows


def gold_items(rows: Sequence[Mapping[str, Any]], corpus: Any) -> list[GoldItem]:
    """The gold rows joined to their elements in ``corpus`` (a corpus store), in row order.

    Every row's document must be in the store's ``gold`` split, its element there, and the
    element's ``text_sha1`` the row's; anything else raises :class:`GoldTextError` listing every
    problem, before any model is loaded."""
    keys = [(str(r["doc_id"]), str(r["element_id"])) for r in rows]
    errors = [f"gold element {d}/{e} given twice" for (d, e), n in Counter(keys).items() if n > 1]
    records = {r.id: r for r in corpus.manifest()}
    for doc_id in dict.fromkeys(d for d, _ in keys):
        record = records.get(doc_id)
        if record is None:
            errors.append(f"gold document {doc_id!r} is not in the corpus store {corpus.dir}")
        elif record.split != "gold":
            errors.append(
                f"gold document {doc_id!r} is in the {record.split!r} split, not the gold split"
            )
    if errors:
        raise GoldTextError(errors)
    wanted = set(keys)
    found = {}
    for element in corpus.iter_elements(split="gold"):
        key = (element.doc_id, element.element_id)
        if key in wanted:
            found[key] = element
    items = []
    for row, key in zip(rows, keys, strict=True):
        element = found.get(key)
        if element is None:
            errors.append(f"gold element {key[0]}/{key[1]} is not in the corpus store")
        elif element.text_sha1 != row["text_sha1"]:
            errors.append(
                f"gold element {key[0]}/{key[1]}: text_sha1 {row['text_sha1']} in the gold file, "
                f"{element.text_sha1} in the store (its text changed)"
            )
        else:
            items.append(
                GoldItem(
                    doc_id=key[0],
                    element_id=key[1],
                    family=str(row["family"]),
                    topic=str(row["topic"]),
                    functions=tuple(row["functions"]),
                    element=element,
                )
            )
    if errors:
        raise GoldTextError(errors)
    return items


def _key_hash(*parts: str) -> str:
    return hashlib.sha1(":".join(parts).encode("utf-8")).hexdigest()


def gold_subset(items: Sequence[GoldItem], n: int) -> list[GoldItem]:
    """``n`` gold items stratified by family (largest-remainder quotas), chosen within each
    family by a hash of their key, so the subset does not depend on the input order; kept in
    input order. All items when ``n`` is at least their number."""
    if n < 1:
        raise ValueError(f"a subset needs at least one element, got {n}")
    if n >= len(items):
        return list(items)
    by_family: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        by_family[it.family].append(i)
    quotas = {f: n * len(ix) / len(items) for f, ix in by_family.items()}
    counts = {f: math.floor(q) for f, q in quotas.items()}
    extra = sorted(quotas, key=lambda f: (-(quotas[f] - counts[f]), f))
    for f in extra[: n - sum(counts.values())]:
        counts[f] += 1
    chosen: set[int] = set()
    for f, ix in by_family.items():
        ranked = sorted(ix, key=lambda i: _key_hash("subset", items[i].doc_id, items[i].element_id))
        chosen.update(ranked[: counts[f]])
    return [items[i] for i in sorted(chosen)]


# --- metrics -------------------------------------------------------------------------------------


def _topics() -> Any:
    from docrooter.teach.shortlist import topic_taxonomy

    return topic_taxonomy()


def function_tags() -> list[str]:
    """The 12 function tags, in taxonomy order."""
    from docrooter.teach.shortlist import function_tags as tags

    return tags()


def _f1(tp: int, fp: int, fn: int) -> float | None:
    denominator = 2 * tp + fp + fn
    return None if denominator == 0 else 2 * tp / denominator


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    return {
        "p": tp / (tp + fp) if tp + fp else 0.0,
        "r": tp / (tp + fn) if tp + fn else 0.0,
        "f1": _f1(tp, fp, fn) or 0.0,
    }


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _ancestors(label: str | None, taxonomy: Any) -> set[str]:
    if label is None:
        return set()
    if label not in taxonomy:
        return {label}  # an unknown label is a predicted node that matches nothing
    return {node.id for node in taxonomy.path(label)}


def hierarchical_prf(
    gold: Sequence[str], pred: Sequence[str | None], taxonomy: Any = None
) -> dict[str, float]:
    """Hierarchical P/R/F1 (Kiritchenko et al.): each label stands for its ancestor set in the
    topic tree (family and leaf, no root), and the overlaps are pooled over all elements."""
    if len(gold) != len(pred):
        raise ValueError(f"{len(gold)} gold labels but {len(pred)} predictions")
    taxonomy = taxonomy if taxonomy is not None else _topics()
    overlap = n_pred = n_gold = 0
    for g, p in zip(gold, pred, strict=True):
        gs, ps = _ancestors(g, taxonomy), _ancestors(p, taxonomy)
        overlap += len(gs & ps)
        n_pred += len(ps)
        n_gold += len(gs)
    hp = overlap / n_pred if n_pred else 0.0
    hr = overlap / n_gold if n_gold else 0.0
    return {"p": hp, "r": hr, "f1": 2 * hp * hr / (hp + hr) if hp + hr else 0.0}


def topic_metrics(
    gold: Sequence[str], pred: Sequence[str | None], taxonomy: Any = None
) -> dict[str, Any]:
    """Accuracy, macro P/R/F1 over the classes present in ``gold`` (an undefined ratio counts
    0; a predicted label outside them is only a miss of its gold class), per-class scores and
    hierarchical P/R/F1. A ``None`` prediction is a miss."""
    if len(gold) != len(pred):
        raise ValueError(f"{len(gold)} gold labels but {len(pred)} predictions")
    n = len(gold)
    pairs = Counter(zip(gold, pred, strict=True))
    gold_n, pred_n = Counter(gold), Counter(p for p in pred if p is not None)
    classes = sorted(gold_n)
    per_class = {}
    for c in classes:
        tp = pairs[(c, c)]
        per_class[c] = {**_prf(tp, pred_n[c] - tp, gold_n[c] - tp), "support": gold_n[c]}
    return {
        "n": n,
        "accuracy": sum(pairs[(c, c)] for c in classes) / n if n else None,
        "macro_over": MACRO_OVER,
        "classes": len(classes),
        "macro_p": _mean([v["p"] for v in per_class.values()]),
        "macro_r": _mean([v["r"] for v in per_class.values()]),
        "macro_f1": _mean([v["f1"] for v in per_class.values()]),
        "hierarchical": hierarchical_prf(gold, pred, taxonomy),
        "per_class": per_class,
    }


def function_metrics(
    gold: Sequence[Iterable[str]], pred: Sequence[Iterable[str]], tags: Sequence[str] | None = None
) -> dict[str, Any]:
    """Micro and macro F1 and per-tag P/R/F1 of multi-label function tags. A tag nobody
    labelled or predicted has no F1 (``None``) and is left out of the macro."""
    if len(gold) != len(pred):
        raise ValueError(f"{len(gold)} gold label sets but {len(pred)} predictions")
    tags = list(tags) if tags is not None else function_tags()
    gold_sets = [set(g) for g in gold]
    pred_sets = [set(p) for p in pred]
    per_tag: dict[str, dict[str, Any]] = {}
    total = [0, 0, 0]
    for tag in tags:
        tp = sum(tag in g and tag in p for g, p in zip(gold_sets, pred_sets, strict=True))
        fp = sum(tag in p for p in pred_sets) - tp
        fn = sum(tag in g for g in gold_sets) - tp
        total = [total[0] + tp, total[1] + fp, total[2] + fn]
        defined = _f1(tp, fp, fn) is not None
        scores: dict[str, float | None] = (
            dict(_prf(tp, fp, fn)) if defined else dict.fromkeys(("p", "r", "f1"))
        )
        per_tag[tag] = {**scores, "support": tp + fn, "predicted": tp + fp}
    scored = [v["f1"] for v in per_tag.values() if v["f1"] is not None]
    return {
        "n": len(gold_sets),
        "micro": _prf(*total),
        "macro_f1": _mean(scored),
        "tags_scored": len(scored),
        "per_tag": per_tag,
    }


def ece(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = ECE_BINS
) -> float | None:
    """Expected calibration error: the bin-size-weighted mean |accuracy - confidence| over
    ``n_bins`` equal-width confidence bins ``(k/n, (k+1)/n]`` (0 goes in the first). ``None``
    without predictions."""
    conf = np.asarray(confidences, dtype=float)
    hits = np.asarray(correct, dtype=float)
    if conf.shape != hits.shape:
        raise ValueError(f"{conf.size} confidences but {hits.size} outcomes")
    if conf.size == 0:
        return None
    # Rounded first, so a confidence on a bin edge stays in its bin despite float noise
    # (0.2 x 15 = 3.0000000000000004).
    bins = np.clip(np.ceil(np.round(conf * n_bins, 9)).astype(int) - 1, 0, n_bins - 1)
    total = 0.0
    for b in np.unique(bins):
        mask = bins == b
        total += mask.sum() / conf.size * abs(hits[mask].mean() - conf[mask].mean())
    return float(total)


# --- scoring a tagger on gold --------------------------------------------------------------------

TagRow = Mapping[str, Any]  # {"topic": TagResult, "function": TagResult}


def _topic_label(tags: TagRow) -> str | None:
    topic = tags.get("topic")
    return topic.labels[0] if topic is not None and topic.labels else None


def _function_labels(tags: TagRow) -> list[str]:
    function = tags.get("function")
    return list(function.labels) if function is not None else []


def _eces(items: Sequence[GoldItem], results: Sequence[TagRow]) -> dict[str, float | None]:
    confidences, hits = [], []
    for it, tags in zip(items, results, strict=True):
        topic = tags.get("topic")
        if topic is not None:
            confidences.append(float(topic.confidence))
            hits.append(_topic_label(tags) == it.topic)
    f_conf, f_hits = [], []
    for it, tags in zip(items, results, strict=True):
        function = tags.get("function")
        if function is None:
            continue
        for tag, p in function.probs.items():
            f_conf.append(max(p, 1.0 - p))
            f_hits.append((p >= FUNCTION_THRESHOLD) == (tag in it.functions))
    return {"topic": ece(confidences, hits), "function": ece(f_conf, f_hits)}


def score_mode(
    items: Sequence[GoldItem], results: Sequence[TagRow], taxonomy: Any = None
) -> dict[str, Any]:
    """The metrics of one mode's tags (module docstring): topic overall and per gold family,
    functions (``None`` when the tagger gives none) and ECE."""
    taxonomy = taxonomy if taxonomy is not None else _topics()
    gold = [it.topic for it in items]
    pred = [_topic_label(r) for r in results]
    by_family = {}
    for family in sorted({it.family for it in items}):
        idx = [i for i, it in enumerate(items) if it.family == family]
        by_family[family] = topic_metrics([gold[i] for i in idx], [pred[i] for i in idx], taxonomy)
    has_functions = bool(results) and all(r.get("function") is not None for r in results)
    return {
        "topic": topic_metrics(gold, pred, taxonomy),
        "topic_by_family": by_family,
        "family_mean_macro_f1": _mean([m["macro_f1"] for m in by_family.values()]),
        "function": (
            function_metrics([it.functions for it in items], [_function_labels(r) for r in results])
            if has_functions
            else None
        ),
        "ece": _eces(items, results),
    }


def _tag_all(
    tagger: Any,
    items: Sequence[GoldItem],
    families: Sequence[str],
    cache: dict[tuple[int, str], TagRow],
) -> list[TagRow]:
    """Each item's tags within ``families[i]``: one ``tagger.tag`` call per family for the
    items not already in ``cache`` (keyed by item index and family)."""
    todo: dict[str, list[int]] = defaultdict(list)
    for i, family in enumerate(families):
        if (i, family) not in cache:
            todo[family].append(i)
    for family, idx in todo.items():
        results = tagger.tag([items[i].element for i in idx], family)
        if len(results) != len(idx):
            raise RuntimeError(
                f"the tagger returned {len(results)} results for {len(idx)} elements"
            )
        for i, tags in zip(idx, results, strict=True):
            cache[(i, family)] = tags
    return [cache[(i, family)] for i, family in enumerate(families)]


class _EmbedMemo:
    """Stands in for a student's ``embed`` while it is scored: each batch of texts is embedded
    once, so the untempered and end-to-end passes reuse the primary pass's vectors."""

    def __init__(self, embed: Callable[[list[str]], Any]) -> None:
        self.embed = embed
        self.vectors: dict[tuple[str, ...], Any] = {}

    def __call__(self, texts: list[str]) -> Any:
        key = tuple(texts)
        if key not in self.vectors:
            self.vectors[key] = self.embed(texts)
        return self.vectors[key]


@contextlib.contextmanager
def _memoised_embed(tagger: Any) -> Any:
    embed = getattr(tagger, "embed", None)
    if not callable(embed):
        yield
        return
    own = vars(tagger).get("embed")
    tagger.embed = _EmbedMemo(embed)
    try:
        yield
    finally:
        if own is None:
            del tagger.embed
        else:
            tagger.embed = own


class _CountingTeacher:
    """Stands in for a ``TeacherTagger``'s engine while it is scored: counts the items it is
    asked (topic choices and yes/no questions), so the teacher's items/s is measured, not
    derived."""

    def __init__(self, teacher: Any) -> None:
        self._teacher = teacher
        self.items = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._teacher, name)

    def choose(self, items: Sequence[Any], **options: Any) -> Any:
        self.items += len(items)
        return self._teacher.choose(items, **options)

    def yes_no(self, items: Sequence[Any], **options: Any) -> Any:
        self.items += len(items)
        return self._teacher.yes_no(items, **options)


@contextlib.contextmanager
def _counted_teacher(tagger: Any) -> Iterator[_CountingTeacher | None]:
    teacher = getattr(tagger, "teacher", None)
    if teacher is None or not callable(getattr(teacher, "choose", None)):
        yield None
        return
    counter = _CountingTeacher(teacher)
    tagger.teacher = counter
    try:
        yield counter
    finally:
        tagger.teacher = teacher


def _calibration(
    tagger: Any, items: Sequence[GoldItem], calibrated: Sequence[TagRow]
) -> dict[str, Any]:
    """ECE of a student's primary tags at temperature 1.0 (before) and at its calibrated
    temperatures (after)."""
    temperatures = tagger.temperatures
    tagger.temperatures = dict.fromkeys(temperatures, 1.0)
    try:
        before = _tag_all(tagger, items, [it.family for it in items], {})
    finally:
        tagger.temperatures = temperatures
    return {
        "temperatures": dict(temperatures),
        "ece_before": _eces(items, before),
        "ece_after": _eces(items, calibrated),
    }


def evaluate_tagger(
    tagger: Any,
    items: Sequence[GoldItem],
    *,
    predicted_families: Mapping[str, str] | None = None,
    taxonomy: Any = None,
) -> dict[str, Any]:
    """Score ``tagger`` (anything with ``tag(elements, family)``) on ``items``: primary mode,
    end to end when ``predicted_families`` (doc id -> family) is given, the calibration of a
    student (a tagger with ``temperatures``), throughput and the per-element predictions (labels
    only, no text)."""
    items = list(items)
    if predicted_families is not None:
        missing = sorted({it.doc_id for it in items} - set(predicted_families))
        if missing:
            raise ValueError(f"no predicted doc type for gold document(s) {missing[:10]}")
    taxonomy = taxonomy if taxonomy is not None else _topics()
    gold_families = [it.family for it in items]
    teacher_items = None
    with _memoised_embed(tagger), _counted_teacher(tagger) as counter:
        started = time.perf_counter()
        if items:
            tagger.tag([items[0].element], items[0].family)  # the model loads here
        load_s = time.perf_counter() - started
        cache: dict[tuple[int, str], TagRow] = {}
        asked_before = counter.items if counter is not None else 0
        started = time.perf_counter()
        primary = _tag_all(tagger, items, gold_families, cache)
        tag_s = time.perf_counter() - started
        if counter is not None:
            teacher_items = counter.items - asked_before
        e2e = e2e_families = None
        if predicted_families is not None:
            e2e_families = [predicted_families[it.doc_id] for it in items]
            e2e = _tag_all(tagger, items, e2e_families, cache)
        calibration = (
            _calibration(tagger, items, primary)
            if isinstance(getattr(tagger, "temperatures", None), dict)
            else None
        )
    end_to_end = None
    if e2e is not None and e2e_families is not None:
        end_to_end = score_mode(items, e2e, taxonomy)
        agree = sum(f == g for f, g in zip(e2e_families, gold_families, strict=True))
        end_to_end["families"] = {"agree": agree, "n": len(items)}
    predictions = []
    for i, it in enumerate(items):
        topic = primary[i].get("topic")
        row: dict[str, Any] = {
            "doc_id": it.doc_id,
            "element_id": it.element_id,
            "gold_family": it.family,
            "gold_topic": it.topic,
            "gold_functions": list(it.functions),
            "topic": _topic_label(primary[i]),
            "topic_confidence": round(float(topic.confidence), 6) if topic is not None else None,
            "functions": _function_labels(primary[i]),
        }
        if e2e is not None and e2e_families is not None:
            row["e2e_family"] = e2e_families[i]
            row["e2e_topic"] = _topic_label(e2e[i])
        predictions.append(row)
    return {
        "n": len(items),
        "load_s": round(load_s, 3),
        "tag_s": round(tag_s, 3),
        "elements_per_s": round(len(items) / max(tag_s, 1e-9), 3),
        "teacher_items": teacher_items,
        "teacher_items_per_s": (
            round(teacher_items / max(tag_s, 1e-9), 3) if teacher_items is not None else None
        ),
        "primary": score_mode(items, primary, taxonomy),
        "end_to_end": end_to_end,
        "calibration": calibration,
        "predictions": predictions,
    }


# --- systems -------------------------------------------------------------------------------------


def _enriched_text(node: Any) -> str:
    from docrooter.backends.base import candidate_text

    text = candidate_text(node)
    return f"{text} Key terms: {', '.join(node.keywords)}." if node.keywords else text


def _zero_shot_tags(
    leaves: Sequence[str],
    topic_row: Sequence[float],
    tags: Sequence[str],
    function_row: Any,
    source: str,
) -> dict[str, Any]:
    """Tags as L10's ``EmbeddingTagger`` builds them from its two probability rows."""
    from docrooter.elements.models import TagResult

    def p(value: float) -> float:
        return round(min(1.0, max(0.0, float(value))), 6)

    topics = dict(zip(leaves, (float(v) for v in topic_row), strict=True))
    best = max(topics, key=lambda k: topics[k])
    values = dict(zip(tags, (float(v) for v in function_row), strict=True))
    return {
        "topic": TagResult(
            labels=[best], probs={k: p(v) for k, v in topics.items()}, confidence=p(topics[best]),
            source=source,
        ),
        "function": TagResult(
            labels=[t for t in tags if values[t] >= FUNCTION_THRESHOLD],
            probs={t: p(v) for t, v in values.items()},
            confidence=p(max(values.values())),
            source=source,
        ),
    }  # fmt: skip


class EnrichedEmbeddingTagger(EmbeddingTagger):
    """Zero-shot with the enriched taxonomy (spec §11): L10's ``EmbeddingTagger``, except that
    each topic leaf is described by its definition followed by its mined keywords (``Key terms:
    a, b, c.``; spec §8's ``keywords`` field), with its examples as before. Functions are scored
    as by the plain zero-shot tagger (the enrichment is of the topic tree only)."""

    def __init__(self, spec: str, enriched: Any, *, backend: Any = None, batch_size: int = 256):
        builtin = _topics()
        ids = sorted(n.id for n in enriched.leaves())
        if ids != sorted(n.id for n in builtin.leaves()):
            raise ValueError(
                "the enriched taxonomy's leaves are not those of element_topics; pass the "
                f"{ENRICHED_FILE} that train-elements wrote"
            )
        super().__init__(spec, backend=backend, batch_size=batch_size)
        self.enriched = enriched
        self.name = f"{spec}+enriched"

    def tag(self, elements: Sequence[Any], family: str) -> list[dict[str, Any]]:
        from docrooter.backends.base import Candidate
        from docrooter.teach.shortlist import (
            TOPIC_INSTRUCTION,
            element_query,
            family_leaves,
            function_cosines,
        )

        nodes = family_leaves(family, self.enriched)
        leaves = [n.id for n in nodes]
        candidates = [Candidate(n.id, _enriched_text(n), tuple(n.examples)) for n in nodes]
        instruction = TOPIC_INSTRUCTION.format(family=self.enriched.node(family).label.lower())
        tags = function_tags()
        out = []
        for start in range(0, len(elements), self.batch_size):
            batch = list(elements[start : start + self.batch_size])
            scores = self.backend.score(
                [element_query(e) for e in batch], candidates, instruction=instruction
            )
            cosines = np.asarray(scores.values, dtype=float).reshape(len(batch), len(leaves))
            topics = self._probs(cosines)
            functions = self._probs(function_cosines(batch, self.backend))
            for topic_row, function_row in zip(topics, functions, strict=True):
                out.append(_zero_shot_tags(leaves, topic_row, tags, function_row, self.source))
        return out


def load_enriched(path: str | os.PathLike[str]) -> Any:
    from docrooter.taxonomy import Taxonomy

    return Taxonomy.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_tagger(
    spec: str, *, enriched_path: str | os.PathLike[str] | None = None, device: str | None = None
) -> Any:
    """The tagger of a system spec: ``keyword``, ``embedding:<model>``, ``enriched:<model>``
    (with ``enriched_path``), ``teacher:<engine spec>`` (Jev refused) or a bundle directory.
    Nothing heavy loads until the first element."""
    from docrooter.tagger.element_tags import (
        EMBEDDING_PREFIX,
        TEACHER_PREFIX,
        TeacherTagger,
        refuse_jev_spec,
    )

    value = spec.strip()
    refuse_jev_spec(value)
    if value == KEYWORD:
        from docrooter.backends.fake import KeywordBackend

        return EmbeddingTagger(KEYWORD, backend=KeywordBackend())
    if value.startswith(ENRICHED_PREFIX):
        if enriched_path is None:
            raise ValueError("an enriched: system needs the enriched taxonomy (--enriched)")
        model = value[len(ENRICHED_PREFIX) :]
        return EnrichedEmbeddingTagger(f"{EMBEDDING_PREFIX}{model}", load_enriched(enriched_path))
    if value.startswith(EMBEDDING_PREFIX):
        return EmbeddingTagger(value)
    if value.startswith(TEACHER_PREFIX):
        return TeacherTagger(value[len(TEACHER_PREFIX) :])
    from docrooter.tagger.bundle import ElementTagger

    return ElementTagger.load(value, device=device)


# --- provenance, memory, files -------------------------------------------------------------------


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bundle_sha256(path: str | os.PathLike[str]) -> str:
    """One hash over every file of a bundle directory: relative path and bytes, in path order
    (temporary ``*.tmp`` files left out)."""
    root = Path(path)
    digest = hashlib.sha256()
    for file in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix != ".tmp"):
        digest.update(file.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(file_sha256(file).encode("ascii"))
    return digest.hexdigest()


def teacher_parts(name: str) -> dict[str, str | None]:
    """``engine:model@revision#dtype`` split up (an openai name has no revision or dtype)."""
    engine, _, target = name.partition(":")
    target, _, dtype = target.partition("#")
    model, at, revision = target.rpartition("@")
    if not at:
        model, revision = target, ""
    return {"engine": engine, "model": model, "revision": revision or None, "dtype": dtype or None}


def _commit() -> str | None:
    from benchmarks.pdf_bench import bench_revision

    return bench_revision()


def _display(path: str | os.PathLike[str]) -> str:
    resolved = Path(path).resolve()
    root = BENCHMARKS.parent.resolve()
    return str(resolved.relative_to(root)) if resolved.is_relative_to(root) else str(resolved)


def provenance(
    spec: str,
    tagger: Any,
    *,
    gold_path: str | os.PathLike[str],
    doc_types_path: str | os.PathLike[str] | None = None,
    enriched_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Commit, gold and its hash, the bundle (path, hash, training info), the teacher (name with
    dtype), the enriched taxonomy's and the predicted doc types' hashes."""
    bundle = getattr(tagger, "path", None)
    teacher = getattr(tagger, "teacher", None)
    info = getattr(tagger, "info", None) if bundle is not None else None
    return {
        "commit": _commit(),
        "spec": spec,
        "gold": _display(gold_path),
        "gold_sha256": file_sha256(gold_path),
        "disclosure": DISCLOSURE,
        "bundle": _display(bundle) if bundle is not None else None,
        "bundle_sha256": bundle_sha256(bundle) if bundle is not None else None,
        "bundle_info": (
            {k: info.get(k) for k in BUNDLE_INFO_KEYS} if isinstance(info, dict) else None
        ),
        "temperatures": dict(tagger.temperatures) if bundle is not None else None,
        "teacher": (
            {"name": teacher.name, **teacher_parts(teacher.name)} if teacher is not None else None
        ),
        "enriched_sha256": (
            file_sha256(enriched_path)
            if enriched_path is not None and spec.startswith(ENRICHED_PREFIX)
            else None
        ),
        "doc_types_sha256": file_sha256(doc_types_path) if doc_types_path is not None else None,
    }


LABEL_FILES = ("teacher", "prescreen", "weak")  # the store's labellers (docrooter.teach.label)
BUNDLE_INFO_KEYS = (
    "stage",
    "base_model",
    "examples",
    "untrained_leaves",
    "calibration",
    "train_elements",
    "learning_curve",
    "active_learning",
)


def training_provenance(store: str | os.PathLike[str]) -> dict[str, Any]:
    """Where a student's training data came from, read from the corpus store: the engine
    name(s) of the teacher that labelled it, with dtype (each teacher label row's provenance),
    how many rows came from the active-learning round, the sha256 of each label file, the
    labelling settings (``train-elements/label.json``) and the confident set's fingerprint
    (``train-elements/dataset.json``)."""
    root = Path(store)
    labels = root / "labels"
    files = {
        name: file_sha256(labels / f"{name}.jsonl")
        for name in LABEL_FILES
        if (labels / f"{name}.jsonl").is_file()
    }
    teachers: Counter[str] = Counter()
    active_learning = 0
    for row in _teacher_rows(root):
        provenance = row.get("provenance") or {}
        teachers[str(provenance.get("teacher"))] += 1
        active_learning += provenance.get("source") == "active_learning"
    work = root / WORK_DIR
    marker = _read_json_file(work / "label.json") or {}
    dataset = _read_json_file(work / "dataset.json") or {}
    return {
        "teachers": [
            {"name": name, "rows": n, **teacher_parts(name)} for name, n in teachers.items()
        ],
        "active_learning_rows": active_learning,
        "label_files_sha256": files,
        "label_settings": marker.get("settings"),
        "dataset_fingerprint": dataset.get("fingerprint"),
    }


def _teacher_rows(store: Path) -> Iterator[dict[str, Any]]:
    """The rows of the store's ``labels/teacher.jsonl`` (none when it is absent)."""
    path = store / "labels" / "teacher.jsonl"
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                continue  # a torn last line, as the store's readers skip it
            if isinstance(row, dict):
                yield row


def active_learning_keys(store: str | os.PathLike[str]) -> set[tuple[str, str]]:
    """``(doc_id, element_id)`` of the elements the active-learning round sent to the teacher
    (rows whose provenance ``source`` is ``active_learning``)."""
    return {
        (str(row.get("doc_id")), str(row.get("element_id")))
        for row in _teacher_rows(Path(store))
        if (row.get("provenance") or {}).get("source") == "active_learning"
    }


def check_no_gold(
    datasets: Mapping[str, Sequence[Any]], corpus: Any, items: Sequence[GoldItem]
) -> dict[str, int]:
    """Refuse training data holding a gold-split element: an example whose ``doc_id`` is a
    gold-split document (or a gold element's key). Returns how many examples were checked and
    how many had no ``doc_id`` to check."""
    gold_docs = {r.id for r in corpus.manifest() if r.split == "gold"}
    gold_docs |= {it.doc_id for it in items}
    errors = []
    checked = unchecked = 0
    for name, examples in datasets.items():
        for example in examples:
            if example.doc_id is None:
                unchecked += 1
                continue
            checked += 1
            if example.doc_id in gold_docs:
                errors.append(
                    f"{name}: {example.doc_id}/{example.section_id} is a gold-split element"
                )
    if errors:
        raise GoldTextError(errors)
    return {"checked": checked, "without_doc_id": unchecked}


def check_full_bundle(
    info: Mapping[str, Any], *, stage: str, dataset: str | None, active_learning_examples: int
) -> None:
    """Refuse ``--full-bundle`` as the curve's 1/1 point unless it was trained, at the curve's
    stage, on the current confident set (its recorded fingerprint is ``dataset``), before the
    active-learning round."""
    recorded = info.get("train_elements") or {}
    problems = []
    if dataset is None or recorded.get("dataset") != dataset:
        problems.append(
            f"its confident-set fingerprint {recorded.get('dataset')!r} is not the current "
            f"one {dataset!r}"
        )
    if recorded.get("stage", info.get("stage")) != stage:
        problems.append(f"it is a stage {recorded.get('stage', info.get('stage'))} bundle")
    if "active_learning" in info or active_learning_examples:
        problems.append(
            "it is not pre-active-learning (the bundle or the confident set holds "
            f"active-learning labels: {active_learning_examples} examples)"
        )
    if problems:
        raise SystemExit(
            f"--full-bundle cannot be the stage {stage} curve's 1/1 point: "
            + "; ".join(problems)
            + ". Leave out --full-bundle to train the 1/1 point on the curve's confident set."
        )


def _read_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@contextlib.contextmanager
def neutral_copies(
    paths: Sequence[Path], sha256s: Sequence[str] | None = None
) -> Iterator[list[Path]]:
    """Each PDF under a neutral name, ``doc_<sha256[:12]>.pdf``, in a temporary directory
    removed afterwards (a hard link where possible, else a copy). A model must never see a
    source-revealing file name: the doc-type view's title falls back to the file stem."""
    with tempfile.TemporaryDirectory(prefix="docrooter-neutral-") as tmp:
        out = []
        for k, path in enumerate(paths):
            sha = sha256s[k] if sha256s is not None else file_sha256(path)
            target = Path(tmp) / f"doc_{sha[:12]}.pdf"
            if not target.exists():
                try:
                    os.link(path, target)
                except OSError:
                    shutil.copyfile(path, target)
            out.append(target)
        yield out


NEUTRAL_NAME = re.compile(r"doc_[0-9a-f]{12}\.pdf")


def _set_device(device: str | None, spec: str = "") -> str:
    """The torch device for this run, exported as ``DOCROOTER_DEVICE`` so every backend loads
    there (``mlx`` for an MLX teacher, ``cpu`` for the keyword system: they use no torch)."""
    if device:
        os.environ["DOCROOTER_DEVICE"] = device
        return device
    if spec == KEYWORD:
        return "cpu"
    if spec.startswith("teacher:mlx:"):
        return "mlx"
    from docrooter.backends.loading import select_device

    chosen = select_device(None)
    os.environ["DOCROOTER_DEVICE"] = chosen
    return chosen


def _memory(sampler: Any, device: str) -> dict[str, Any]:
    """The process's peak memory, as :func:`benchmarks.memory_check.run_case` records it."""
    from benchmarks.memory_check import footprint_bytes, peak_rss_bytes
    from docrooter import runtime

    peak = peak_rss_bytes()
    footprint = footprint_bytes()
    if footprint is not None:
        total, method = footprint[1], "peak phys_footprint (task_info)"
    else:
        total, method = peak + sampler.peak_driver, "peak RSS + peak MPS driver"
    target = runtime.settings().memory_budget_gb
    row: dict[str, Any] = {
        "peak_total_gb": round(total / GB, 3),
        "total_method": method,
        "peak_rss_gb": round(peak / GB, 3),
        "target_gb": target,
        "within_target": total / GB <= target,
    }
    if device.startswith("mps"):
        row["peak_mps_driver_gb"] = round(sampler.peak_driver / GB, 3)
        row["peak_mps_allocated_gb"] = round(sampler.peak_allocated / GB, 3)
    return row


def _environment() -> dict[str, Any]:
    names = [*_guard.ENV_CAPS, *_guard.MPS_CAP_VARS, "DOCROOTER_DEVICE"]
    return {
        "env": {name: os.environ.get(name) for name in names},
        "platform": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "system": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _merge_row(path: Path, row: dict[str, Any], key: Sequence[str], header: dict[str, Any]) -> None:
    """Put ``row`` into ``path``'s ``runs``, replacing the run with the same ``key`` values."""
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else dict(header)
    runs = data.setdefault("runs", [])
    wanted = tuple(row[k] for k in key)
    for k, existing in enumerate(runs):
        if tuple(existing.get(name) for name in key) == wanted:
            runs[k] = row
            break
    else:
        runs.append(row)
    _write_json(path, data)


def _out_dir(value: Path | None) -> Path:
    return value if value is not None else RESULTS_ROOT / datetime.date.today().isoformat()


def read_predicted_families(path: Path, gold_path: Path) -> dict[str, str]:
    """``{doc_id: family}`` from a ``gold-doctypes`` file, refused when it was made for another
    gold file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("gold_sha256") != file_sha256(gold_path):
        raise SystemExit(
            f"{path} was made for another gold file (its gold_sha256 differs from {gold_path}'s); "
            "run gold-doctypes again"
        )
    return {doc_id: str(record["family"]) for doc_id, record in data["documents"].items()}


# --- commands ------------------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    spec = args.tagger or SYSTEMS.get(args.system)
    if spec is None:
        raise SystemExit(
            f"unknown system {args.system!r}: pass --tagger SPEC (a bundle directory, "
            f"embedding:<model>, enriched:<model>, teacher:<engine spec> or keyword); "
            f"known names: {', '.join(SYSTEMS)}"
        )
    subset = args.subset if args.subset is not None else SYSTEM_SUBSETS.get(args.system)
    device = _set_device(args.device, spec)
    items = gold_items(read_gold(args.gold), _corpus(args.store))
    if subset is not None:
        items = gold_subset(items, subset)
    predicted = read_predicted_families(args.doc_types, args.gold) if args.doc_types else None
    enriched = args.enriched or args.store / WORK_DIR / ENRICHED_FILE
    log(f"{args.system}: {spec} on {len(items)} gold elements ({device})")
    from benchmarks.memory_check import MpsSampler

    started = time.perf_counter()
    with MpsSampler() as sampler:
        tagger = build_tagger(spec, enriched_path=enriched, device=device)
        try:
            result = evaluate_tagger(tagger, items, predicted_families=predicted)
            prov = provenance(
                spec,
                tagger,
                gold_path=args.gold,
                doc_types_path=args.doc_types,
                enriched_path=enriched,
            )
            if prov["bundle"] is not None:
                prov["training"] = training_provenance(args.store)
                recorded = ((prov["bundle_info"] or {}).get("train_elements") or {}).get("dataset")
                prov["bundle_matches_dataset"] = (
                    recorded is not None and recorded == prov["training"]["dataset_fingerprint"]
                )
        finally:
            close = getattr(tagger, "close", None)
            if callable(close):
                close()
    row: dict[str, Any] = {
        "system": args.system,
        "spec": spec,
        "device": device,
        "subset": subset,
        "date": datetime.date.today().isoformat(),
        **result,
        **_memory(sampler, device),
        "seconds": round(time.perf_counter() - started, 1),
        "provenance": prov,
        **_environment(),
    }
    path = _out_dir(args.out) / RESULTS_FILE
    header = {"disclosure": DISCLOSURE, "macro_over": MACRO_OVER, "runs": []}
    _merge_row(path, row, ("system", "device", "subset"), header)
    topic = result["primary"]["topic"]
    log(
        f"{args.system}: topic macro F1 {topic['macro_f1']:.3f}, accuracy {topic['accuracy']:.3f}; "
        f"{result['elements_per_s']:.1f} elements/s; peak {row['peak_total_gb']} GB -> {path}"
    )
    return 0


def _corpus(store: Path) -> Any:
    from docrooter.corpus.store import Corpus

    return Corpus(store)


def cmd_gold_doctypes(args: argparse.Namespace) -> int:
    """The rooter's doc type for every gold document (the end-to-end mode's families)."""
    from docrooter.rooter import process_many
    from docrooter.taxonomies import family_for_doc_type

    device = _set_device(args.device)
    rows = read_gold(args.gold)
    records = {r.id: r for r in _corpus(args.store).manifest()}
    gold_family = {str(r["doc_id"]): str(r["family"]) for r in rows}
    doc_ids = list(gold_family)
    errors, paths = [], []
    for doc_id in doc_ids:
        record = records.get(doc_id)
        pdf = args.pdf_dir / f"{doc_id}.pdf"
        if record is None:
            errors.append(f"gold document {doc_id!r} is not in the corpus store")
        elif not pdf.is_file():
            errors.append(f"gold document {doc_id!r}: no PDF at {pdf}")
        elif file_sha256(pdf) != record.sha256:
            errors.append(
                f"gold document {doc_id!r}: {pdf} is not the PDF the store was built from"
            )
        paths.append(pdf)
    if errors:
        raise GoldTextError(errors)
    log(f"gold-doctypes: {len(paths)} gold documents, profile {args.profile}, {args.doc_type}")
    from benchmarks.memory_check import MpsSampler

    documents: dict[str, dict[str, Any]] = {}
    sources: set[str] = set()
    started = time.perf_counter()
    shas = [records[d].sha256 for d in doc_ids]
    with MpsSampler() as sampler, neutral_copies(paths, shas) as neutral:
        analyses = process_many(
            neutral,
            profile=args.profile,
            doc_type=args.doc_type,
            sections="none",
            summaries="none",
            escalate="none",
            embeddings="none",
            element_tags="none",
        )
        with contextlib.closing(analyses):
            for doc_id, analysis in zip(doc_ids, analyses, strict=True):
                result = analysis.doc_type
                leaf = result.best.node_ids[-1] if result and result.best.node_ids else None
                if result is not None:
                    sources.add(result.source)
                documents[doc_id] = {
                    "doc_type": leaf,
                    "confidence": round(result.confidence, 4) if result else None,
                    "top3": _top3(result.leaf_probs) if result else {},
                    "family": family_for_doc_type(leaf or ""),
                    "gold_family": gold_family[doc_id],
                }
                log(f"{doc_id}: {leaf} -> {documents[doc_id]['family']}")
    agree = sum(d["family"] == d["gold_family"] for d in documents.values())
    data = {
        "gold_sha256": file_sha256(args.gold),
        "profile": args.profile,
        "doc_type_backend": args.doc_type,
        "doc_type_sources": sorted(sources),
        "inputs": "neutral copies, doc_<sha256[:12]>.pdf (no document id reaches the model)",
        "device": device,
        "commit": _commit(),
        "date": datetime.date.today().isoformat(),
        "family_agreement": {"agree": agree, "n": len(documents)},
        "seconds": round(time.perf_counter() - started, 1),
        **_memory(sampler, device),
        **_environment(),
        "documents": documents,
    }
    path = _out_dir(args.out) / GOLD_DOCTYPES_FILE
    _write_json(path, data)
    log(f"gold-doctypes: family agrees with gold on {agree}/{len(documents)} -> {path}")
    return 0


def _top3(probs: Mapping[str, float]) -> dict[str, float]:
    ranked = sorted(((k, v) for k, v in probs.items() if v > 0), key=lambda kv: -kv[1])
    return {k: round(v, 4) for k, v in ranked[:3]}


# --- learning curve ------------------------------------------------------------------------------


def parse_fraction(value: str) -> float:
    """``"1/8"``, ``"0.125"`` or ``"1"`` as a float in (0, 1]."""
    fraction = float(Fraction(value.strip()))
    if not 0 < fraction <= 1:
        raise ValueError(f"a fraction must be in (0, 1], got {value!r}")
    return fraction


def fraction_label(fraction: float) -> str:
    return str(Fraction(fraction).limit_denominator(1000))


def _share(n: int, fraction: float) -> int:
    return n if fraction >= 1 else max(1, math.floor(fraction * n + 0.5))


def nested_fraction_indices(
    groups: Sequence[str], fractions: Iterable[float], seed: int = 0
) -> dict[float, list[int]]:
    """For each fraction, the indices of a class-stratified subset (``groups[i]`` is example
    ``i``'s class): per class, a prefix of one seeded permutation, of ``round(fraction · n)``
    examples and at least one. Prefixes of the same permutation, so a smaller fraction's subset
    is inside every larger one's, whichever fractions are asked for together."""
    fractions = list(fractions)
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError(f"a fraction must be in (0, 1], got {fraction}")
    rng = np.random.default_rng(seed)
    by_group: dict[str, list[int]] = defaultdict(list)
    for i, group in enumerate(groups):
        by_group[group].append(i)
    orders = [[ix[k] for k in rng.permutation(len(ix))] for _, ix in sorted(by_group.items())]
    return {
        f: sorted(i for order in orders for i in order[: _share(len(order), f)]) for f in fractions
    }


def _function_rows(path: Path) -> list[Any]:
    from docrooter.tagger.functions import FunctionRow

    lines = path.read_text(encoding="utf-8").splitlines()
    return [FunctionRow.model_validate_json(line) for line in lines if line.strip()]


def _bundle_info(path: Path) -> dict[str, Any]:
    try:
        return dict(json.loads((path / "tagger.json").read_text(encoding="utf-8"))["info"])
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def run_curve_point(args: argparse.Namespace, fraction: float) -> dict[str, Any]:
    """Train (or reuse) the student on ``fraction`` of the confident set and score it on gold
    (primary mode). The calib split and the gold set are the same at every size."""
    from docrooter.tagger.bundle import ElementTagger
    from docrooter.tagger.train import calibrate_tagger, train_tagger
    from docrooter.train import Dataset

    work: Path = args.work
    train = list(Dataset.load(work / "train.jsonl"))
    calib = Dataset.load(work / "calib.jsonl")
    function_train = _function_rows(work / "functions-train.jsonl")
    function_calib = _function_rows(work / "functions-calib.jsonl")
    dataset = json.loads((work / "dataset.json").read_text(encoding="utf-8"))
    topic_idx = nested_fraction_indices([">".join(e.path) for e in train], [fraction], args.seed)
    function_idx = nested_fraction_indices([""] * len(function_train), [fraction], args.seed)
    label = fraction_label(fraction)
    wanted = {
        "stage": args.stage,
        "fraction": label,
        "examples": len(topic_idx[fraction]),
        "function_rows": len(function_idx[fraction]),
        "seed": args.seed,
        "dataset": dataset.get("fingerprint"),
    }
    out = args.bundle_root / f"stage-{args.stage.lower()}-{label.replace('/', 'of')}"
    corpus = _corpus(args.store)
    items = gold_items(read_gold(args.gold), corpus)
    leakage = check_no_gold({"train": train, "calib": list(calib)}, corpus, items)
    al_keys = active_learning_keys(args.store)
    al_examples = sum((e.doc_id, e.section_id) in al_keys for e in train)
    training = training_provenance(args.store)
    started = time.perf_counter()
    if fraction >= 1 and args.full_bundle is not None:
        check_full_bundle(
            _bundle_info(args.full_bundle),
            stage=args.stage,
            dataset=wanted["dataset"],
            active_learning_examples=al_examples,
        )
        tagger, how, out = (
            ElementTagger.load(args.full_bundle, device=args.device),
            "reused",
            args.full_bundle,
        )
    elif _bundle_info(out).get("learning_curve") == wanted:
        tagger, how = ElementTagger.load(out, device=args.device), "cached"
    else:
        n = wanted["examples"]
        log(f"learning curve: training stage {args.stage} on {label} ({n} examples)")
        tagger = train_tagger(
            Dataset([train[i] for i in topic_idx[fraction]]),
            [function_train[i] for i in function_idx[fraction]],
            stage=args.stage,
            device=args.device,
            seed=args.seed,
            time_budget_hours=args.time_budget_hours,
            seconds_per_step=args.seconds_per_step,
        )
        calibrate_tagger(tagger, calib, function_calib)
        tagger.info["learning_curve"] = wanted
        Path(out).mkdir(parents=True, exist_ok=True)
        tagger.save(out)
        how = "trained"
    train_s = time.perf_counter() - started
    try:
        result = evaluate_tagger(tagger, items)
        info = dict(tagger.info)
    finally:
        tagger.close()
    primary = result["primary"]
    function = primary["function"] or {}
    return {
        **wanted,
        "fraction_value": fraction,
        "how": how,
        "train_s": round(train_s, 1),
        "macro_f1": primary["topic"]["macro_f1"],
        "accuracy": primary["topic"]["accuracy"],
        "hierarchical_f1": primary["topic"]["hierarchical"]["f1"],
        "family_mean_macro_f1": primary["family_mean_macro_f1"],
        "function_micro_f1": (function.get("micro") or {}).get("f1"),
        "function_macro_f1": function.get("macro_f1"),
        "ece": primary["ece"],
        "calibration": result["calibration"],
        "elements_per_s": result["elements_per_s"],
        "untrained_leaves": info.get("untrained_leaves"),
        "budget": info.get("budget"),
        "gold_leakage_check": leakage,
        "active_learning_examples": al_examples,
        "training": training,
        "bundle": _display(out),
        "bundle_sha256": bundle_sha256(out),
        "gold_sha256": file_sha256(args.gold),
        "commit": _commit(),
        "date": datetime.date.today().isoformat(),
    }


def cmd_learning_curve(args: argparse.Namespace) -> int:
    """Each fraction in its own process (its peak memory and model are its own), one after the
    other; a single fraction runs here."""
    fractions = [parse_fraction(f) for f in args.fractions.split(",")]
    if len(fractions) > 1:
        for fraction in fractions:
            command = [sys.executable, "-m", "benchmarks.element_tags_bench", "learning-curve"]
            command += ["--fractions", fraction_label(fraction), *_curve_options(args)]
            code = subprocess.run(command, check=False).returncode
            if code:
                return code
        return 0
    from benchmarks.memory_check import MpsSampler

    device = _set_device(args.device)
    args.device = device
    started = time.perf_counter()
    with MpsSampler() as sampler:
        row = run_curve_point(args, fractions[0])
    row.update(_memory(sampler, device), seconds=round(time.perf_counter() - started, 1))
    row.update(device=device, **_environment())
    path = _out_dir(args.out) / CURVE_FILE
    header = {"disclosure": DISCLOSURE, "framing": "performance depends on corpus size", "runs": []}
    _merge_row(path, row, ("stage", "fraction"), header)
    log(
        f"learning curve {row['stage']} {row['fraction']}: macro F1 {row['macro_f1']:.3f} -> {path}"
    )
    return 0


def _curve_options(args: argparse.Namespace) -> list[str]:
    options = ["--stage", args.stage, "--work", str(args.work), "--seed", str(args.seed)]
    options += ["--bundle-root", str(args.bundle_root), "--store", str(args.store)]
    options += ["--gold", str(args.gold), "--out", str(_out_dir(args.out))]
    options += ["--time-budget-hours", str(args.time_budget_hours)]
    options += ["--seconds-per-step", str(args.seconds_per_step)]
    if args.full_bundle is not None:
        options += ["--full-bundle", str(args.full_bundle)]
    if args.device:
        options += ["--device", args.device]
    return options


# --- doc type with element evidence --------------------------------------------------------------


def _doc_type_leaves() -> list[str]:
    from docrooter.taxonomies import builtin_taxonomy

    return [n.id for n in builtin_taxonomy("doc_types").leaves()]


def implied_doc_types(
    family_probs: Mapping[str, float],
    view: Mapping[str, float],
    doc_types: Sequence[str] | None = None,
) -> dict[str, float]:
    """The doc-type distribution the elements imply: each family's probability (the evidence)
    shared among its doc types (the doc-type -> family map) in proportion to the view's
    probabilities, evenly when the view gives the family nothing."""
    from docrooter.taxonomies import family_for_doc_type

    doc_types = list(doc_types) if doc_types is not None else _doc_type_leaves()
    members: dict[str, list[str]] = defaultdict(list)
    for doc_type in doc_types:
        members[family_for_doc_type(doc_type)].append(doc_type)
    out = dict.fromkeys(doc_types, 0.0)
    for family, types in members.items():
        mass = float(family_probs.get(family, 0.0))
        view_mass = sum(float(view.get(t, 0.0)) for t in types)
        for t in types:
            share = float(view.get(t, 0.0)) / view_mass if view_mass > 0 else 1 / len(types)
            out[t] = mass * share
    return out


def normalise_view(view: Mapping[str, float], doc_types: Sequence[str]) -> dict[str, float]:
    """``view`` over ``doc_types`` (other keys dropped), scaled to sum to 1 (all zeros when it
    has no mass there): the rooter's ``leaf_probs`` keep only the top 10 nodes, so they sum to
    less than 1, which would otherwise tilt the fixed 0.7/0.3 mix towards the elements."""
    kept = {t: max(0.0, float(view.get(t, 0.0))) for t in doc_types}
    mass = sum(kept.values())
    return {t: v / mass for t, v in kept.items()} if mass > 0 else kept


def mix_doc_type(
    view: Mapping[str, float],
    family_probs: Mapping[str, float],
    doc_types: Sequence[str] | None = None,
) -> dict[str, float]:
    """``VIEW_WEIGHT · view + ELEMENT_WEIGHT · implied`` (spec §11: 0.7 / 0.3, fixed, never
    tuned), over the doc-type leaves; the view is first renormalised over them
    (:func:`normalise_view`), so both parts sum to 1 and the weights are what they say."""
    doc_types = list(doc_types) if doc_types is not None else _doc_type_leaves()
    view = normalise_view(view, doc_types)
    implied = implied_doc_types(family_probs, view, doc_types)
    return {
        t: VIEW_WEIGHT * float(view.get(t, 0.0)) + ELEMENT_WEIGHT * implied[t] for t in doc_types
    }


def element_family_evidence(topic_tags: Iterable[Any]) -> dict[str, float] | None:
    """The mean of the elements' unconstrained family distributions (``TagResult.family_probs``,
    §4.0 ruling 2: not the family the doc type imposed), normalised; ``None`` when no element
    has one."""
    rows = [t.family_probs for t in topic_tags if t is not None and t.family_probs]
    if not rows:
        return None
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        for family, p in row.items():
            totals[family] += float(p) / len(rows)
    norm = sum(totals.values())
    return {f: v / norm for f, v in totals.items()} if norm > 0 else None


def _argmax(probs: Mapping[str, float], order: Sequence[str]) -> str | None:
    ranked = [t for t in order if probs.get(t, 0.0) > 0]
    return max(ranked, key=lambda t: probs[t]) if ranked else None


def cmd_doctype_evidence(args: argparse.Namespace) -> int:
    """Dev documents scored by the view alone and by the fixed view + element-evidence mix."""
    from benchmarks import corpora
    from benchmarks.memory_check import MpsSampler
    from docrooter.rooter import process_many

    device = _set_device(args.device)
    entries = corpora.read_manifest(corpora.DEV)  # dev only: no held-out run in v0.5
    leaves = _doc_type_leaves()
    log(f"doctype-evidence: {len(entries)} dev documents, profile {args.profile}, {args.bundle}")
    rows = []
    started = time.perf_counter()
    paths = [corpora.DEV / e["file"] for e in entries]
    with MpsSampler() as sampler, neutral_copies(paths) as neutral:
        analyses = process_many(
            neutral,
            profile=args.profile,
            doc_type=args.doc_type,
            sections="none",
            summaries="none",
            escalate="none",
            embeddings="none",
            element_tags=str(args.bundle),
        )
        with contextlib.closing(analyses):
            for entry, analysis in zip(entries, analyses, strict=True):
                result = analysis.doc_type
                raw = {
                    k: v for k, v in (result.leaf_probs if result else {}).items() if k in leaves
                }
                view = normalise_view(raw, leaves)
                tags = [(el.tags or {}).get("topic") for el in analysis.document.elements]
                evidence = element_family_evidence(tags)
                mixed = mix_doc_type(view, evidence, leaves) if evidence else view
                best = result.best.node_ids[-1] if result and result.best.node_ids else None
                row = {
                    "doc_id": entry["id"],
                    "gold": entry["label"],
                    "pred_view": best,  # the rooter's own doc type: the v0.3/v0.4 baseline
                    "pred_view_argmax": _argmax(view, leaves),
                    "pred_mix": _argmax(mixed, leaves),
                    "view_leaves": len(raw),
                    "view_mass": round(sum(raw.values()), 6),
                    "n_tagged": sum(t is not None for t in tags),
                    "evidence": {k: round(v, 4) for k, v in (evidence or {}).items()},
                    "view_top3": _top3(view),
                    "mix_top3": _top3(mixed),
                }
                rows.append(row)
                view_pred, mix_pred = row["pred_view"], row["pred_mix"]
                log(f"{entry['id']}: view {view_pred}, mix {mix_pred} (gold {entry['label']})")
    n = len(rows)
    view_hits = [r["pred_view"] == r["gold"] for r in rows]
    argmax_hits = [r["pred_view_argmax"] == r["gold"] for r in rows]
    mix_hits = [r["pred_mix"] == r["gold"] for r in rows]
    from benchmarks.pdf_bench import bootstrap_ci

    data = {
        "corpus": "dev",
        "weights": {"view": VIEW_WEIGHT, "elements": ELEMENT_WEIGHT, "tuned": False},
        "view": (
            "the doc-type leaves among the rooter's leaf_probs (its top 10 nodes), renormalised "
            "to sum to 1 before the mix; pred_view is the rooter's own best doc type (with its "
            "low-confidence abstention), pred_view_argmax and pred_mix are argmaxes"
        ),
        "inputs": "neutral copies, doc_<sha256[:12]>.pdf",
        "dev_manifest_sha256": file_sha256(corpora.DEV / "manifest.jsonl"),
        "profile": args.profile,
        "doc_type_backend": args.doc_type,
        "bundle": _display(args.bundle),
        "bundle_sha256": bundle_sha256(args.bundle),
        "device": device,
        "commit": _commit(),
        "date": datetime.date.today().isoformat(),
        "summary": {
            "n": n,
            "view_accuracy": sum(view_hits) / n if n else None,
            "view_ci95": bootstrap_ci(view_hits),
            "view_argmax_accuracy": sum(argmax_hits) / n if n else None,
            "mix_accuracy": sum(mix_hits) / n if n else None,
            "mix_ci95": bootstrap_ci(mix_hits),
            "changed": sum(r["pred_view"] != r["pred_mix"] for r in rows),
            "fixed": sum(m and not v for v, m in zip(view_hits, mix_hits, strict=True)),
            "broken": sum(v and not m for v, m in zip(view_hits, mix_hits, strict=True)),
            "no_evidence": sum(not r["evidence"] for r in rows),
        },
        "seconds": round(time.perf_counter() - started, 1),
        **_memory(sampler, device),
        **_environment(),
        "rows": rows,
    }
    path = _out_dir(args.out) / EVIDENCE_FILE
    _write_json(path, data)
    s = data["summary"]
    log(f"doctype-evidence: view {s['view_accuracy']}, mix {s['mix_accuracy']} -> {path}")
    return 0


# --- command line --------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    from docrooter.classify.classifier import DEFAULT_BACKEND

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--store", type=Path, default=STORE, help="corpus store (gold split)")
        p.add_argument("--gold", type=Path, default=GOLD_PATH)
        p.add_argument("--out", type=Path, default=None, help="results dir (default: dated)")
        p.add_argument("--device", default=None, help="torch device (default: cuda, mps, cpu)")
        _guard.add_guard_argument(p)

    p = sub.add_parser("run", help="score one system on gold (primary and end-to-end modes)")
    p.add_argument("--system", required=True, help=f"result name; known: {', '.join(SYSTEMS)}")
    p.add_argument("--tagger", default=None, help="system spec (a bundle dir for a student)")
    p.add_argument("--subset", type=int, default=None, help="gold elements, family-stratified")
    p.add_argument("--doc-types", type=Path, default=None, help="gold-doctypes output (end to end)")
    p.add_argument("--enriched", type=Path, default=None, help="enriched topic taxonomy JSON")
    common(p)

    p = sub.add_parser("gold-doctypes", help="the rooter's doc type of every gold document")
    p.add_argument("--profile", default=GOLD_DOCTYPE_PROFILE)
    p.add_argument("--doc-type", default=DEFAULT_BACKEND, help="doc-type backend spec")
    p.add_argument("--pdf-dir", type=Path, default=GOLD_PDFS)
    common(p)

    p = sub.add_parser("learning-curve", help="student trained on nested fractions, on gold")
    p.add_argument("--stage", choices=("A", "B"), required=True)
    p.add_argument(
        "--fractions", default=",".join(fraction_label(f) for f in LEARNING_CURVE_FRACTIONS)
    )
    p.add_argument("--work", type=Path, default=STORE / WORK_DIR, help="train-elements dir")
    p.add_argument("--bundle-root", type=Path, default=BUNDLES / "learning-curve")
    p.add_argument("--full-bundle", type=Path, default=None, help="reuse this bundle for 1/1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-budget-hours", type=float, default=6.0)
    p.add_argument(  # docrooter.tagger.train.SECONDS_PER_STEP, an estimate: pass the measured one
        "--seconds-per-step", type=float, default=41.0
    )
    common(p)

    p = sub.add_parser("doctype-evidence", help="dev doc type: view alone and the fixed mix")
    p.add_argument("--bundle", type=Path, required=True, help="the student bundle")
    p.add_argument("--profile", default=EVIDENCE_PROFILE)
    p.add_argument("--doc-type", default=DEFAULT_BACKEND, help="doc-type backend spec")
    common(p)
    return parser


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "run": cmd_run,
    "gold-doctypes": cmd_gold_doctypes,
    "learning-curve": cmd_learning_curve,
    "doctype-evidence": cmd_doctype_evidence,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _guard.single_model_process(
        "benchmarks.element_tags_bench", allow_concurrent=args.allow_concurrent
    )
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
