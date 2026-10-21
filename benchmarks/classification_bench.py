"""The classification benchmark (spec G5, rulings §6.1-6.2): how much accuracy each rung of the
context ladder buys, and at what token cost, against sending the whole document.

    PYTHONPATH=src python -m benchmarks.classification_bench gold
    PYTHONPATH=src python -m benchmarks.classification_bench run --backend SPEC [--label NAME] \
        --flow F[,F] (--level L[,L] | --section-level L[,L] --doc-level L[,L]) \
        [--calls C[,C]] [--docs gold|heldout] [--out DIR] [--limit N] [--no-cache]
    PYTHONPATH=src python -m benchmarks.classification_bench report --results DIR [--out PATH]

``gold`` writes ``classification/doc_type_gold.json`` from the manifest: the source-assigned
``doc_type_hint`` of every ``gold``-split document, ids and types only, with a ``corrections``
list (kept when the file is rewritten) that :func:`load_doc_type_gold` applies.

``run`` classifies every document with ``docrooter.classify`` once per configuration (flow x
section level x doc level x calls). ``--flow``, the levels and ``--calls`` take comma lists or
``all``; ``--level`` pairs the section and document levels (the main grid), ``--section-level``
with ``--doc-level`` crosses them (the cross-grid). ``--calls`` defaults to all three patterns
(``per_item``, ``batched``, ``batched_flat``) for an ``openai:`` backend and to ``per_item`` and
``batched`` for the others, which classify per item either way (their ``batched`` run is the
token estimate of what an LLM would be sent). Documents are read once per process with the
reading the section gold was made with (profile ``fast``; a different reading raises) and the
Document is reused across configurations (``--no-cache`` re-reads it instead). Each
configuration writes one JSON file into ``--out`` (default
``benchmarks/results/<date>/classification/``) with the code identity (git sha and a dirty
flag); a configuration whose file exists with the same settings hash (which includes a
fingerprint of the package code) is skipped, so a crashed run resumes. ``--docs heldout``
runs the frozen held-out corpus and scores the doc type only: the controller runs it once, at
the end.

Per configuration the file records, per document, the gold and predicted doc type, how it was
decided, its flags, the tokens (total and per stage, exact or estimated), the wall time and,
on the gold split, the predicted catalogue leaf of each gold section; and the metrics:
doc-type accuracy and macro-F1, section macro- and micro-F1 at leaf and at group, the share
decided by rules, prompt tokens per document, the estimated split of the prompt into options,
content and instructions, the truncated shares (any view, and the document view), the share of
``full`` documents whose largest exact document call exceeded 16,000 prompt tokens, and wall
time. Results hold ids and numbers
only, never document text (:func:`check_ids_and_numbers` refuses a file that would).

``report`` writes ``docs/benchmarks-classification.md``: the accuracy-against-tokens table,
the token share against the same backend, calls and flow's ``full``/``full`` run, the cheapest
level within 2 points of ``full`` per backend, calls and flow (document ladder by doc-type accuracy,
section ladder by section leaf macro-F1), the cross-grid, the held-out run and the disclosures.

Backends: ``keyword``; ``embedding:dmis-lab/UltraBERT-Embed-150M``; a local LLM as
``openai:<model>`` with ``DOCROOTER_LLM_BASE_URL`` pointing at ``mlx_lm.server``. Jev runs only
with ``--allow-jev``. Embedding and LLM runs take the one-model-process guard; an LLM run
ignores the local server it sends its requests to.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import functools
import hashlib
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from benchmarks import _guard
from benchmarks.classification import section_sample
from benchmarks.classification.section_gold import catalogue_leaves
from benchmarks.training_corpus.gold_sample import write_atomic

BENCHMARKS = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARKS.parent
CLASSIFICATION_DIR = BENCHMARKS / "classification"
MANIFEST_PATH = section_sample.MANIFEST_PATH
PDF_DIR = section_sample.PDF_DIR  # the explicit nosync path, never the ``.cache`` symlink
DOC_TYPE_GOLD_PATH = CLASSIFICATION_DIR / "doc_type_gold.json"
SECTION_GOLD_PATH = CLASSIFICATION_DIR / "section_gold.jsonl"
AGREEMENT_PATH = CLASSIFICATION_DIR / "section_gold_agreement.json"
SAMPLE_SUMMARY_PATH = section_sample.SUMMARY_PATH
HELDOUT_DIR = BENCHMARKS / "heldout"
RESULTS_ROOT = BENCHMARKS / "results"
DOCS_PATH = REPO_ROOT / "docs" / "benchmarks-classification.md"

SCHEMA = 1
LEVELS: tuple[str, ...] = ("headings", "terms", "lead", "full")
FLOWS: tuple[str, ...] = ("doc_first", "sections_first")
CALLS: tuple[str, ...] = ("per_item", "batched", "batched_flat")
"""How ``classify`` asks the backend (V4b): one item per call, or a stage per call with each
option listed once (two-step groups then leaves, or flat)."""
MAX_SECTIONS = 40  # spec §6.2: classify tags at most 40 level-1/2 sections per document
DOCS_SPLITS: tuple[str, ...] = ("gold", "heldout")
FULL_CAP = 16000  # spec §6.1: the measured prompt size of a ``full`` document call
CALL_VIEWS = 40  # docrooter.batch.MAX_BATCH: the most section views one batched call carries
WITHIN_POINTS = 2.0  # spec §6.2: "within 2 points of full"
MAX_ERRORS = 5
# The reading fields that must equal the section gold's (the docrooter version may differ).
READING_KEYS = ("profile", "settings", "elements", "levels")
_LABEL = re.compile(r"^[A-Za-z0-9_.-]+$")
# What a results string may be: an id, a spec, a level, a timestamp. Never text.
_ID = re.compile(r"^[A-Za-z0-9_.:/@+~\-]{0,256}$")


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# --- gold ------------------------------------------------------------------------------------


def doc_type_gold_from_manifest(manifest: Path = MANIFEST_PATH) -> dict[str, Any]:
    """The doc-type gold of the manifest's ``gold`` split: ``{id: doc_type_hint}``, sorted, with
    an empty ``corrections`` list. Ids and types only."""
    documents: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("split") == "gold":
                documents[record["id"]] = record["doc_type_hint"]
    return {
        "source": "benchmarks/training_corpus/manifest.jsonl: doc_type_hint of split gold",
        "documents": dict(sorted(documents.items())),
        "corrections": [],
    }


def load_doc_type_gold(path: Path = DOC_TYPE_GOLD_PATH) -> dict[str, str]:
    """``{doc id: doc type}`` with the file's ``corrections`` (``{id, from, to, ...}``) applied.
    A correction whose ``from`` is not the recorded type raises ``ValueError``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    gold = dict(data["documents"])
    for correction in data.get("corrections", []):
        doc_id = correction["id"]
        if gold.get(doc_id) != correction["from"]:
            raise ValueError(
                f"{path}: correction for {doc_id!r} expects {correction['from']!r}, "
                f"the gold has {gold.get(doc_id)!r}"
            )
        gold[doc_id] = correction["to"]
    return gold


def load_section_gold(path: Path = SECTION_GOLD_PATH) -> dict[str, list[dict[str, Any]]]:
    """The section gold rows grouped by document id, in file order."""
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            by_doc.setdefault(row["doc_id"], []).append(row)
    return by_doc


def current_reading() -> dict[str, Any]:
    """The reading this benchmark uses (the section sampler's)."""
    return section_sample.reading_metadata()


def check_reading(agreement: Path = AGREEMENT_PATH) -> dict[str, Any]:
    """The reading recorded with the section gold (``sample.reading``). Raises ``ValueError``
    unless this benchmark reads documents the same way (ruling P1): same profile, settings,
    elements and section levels."""
    recorded = json.loads(agreement.read_text(encoding="utf-8"))["sample"]["reading"]
    current = current_reading()
    differ = [k for k in READING_KEYS if recorded.get(k) != current.get(k)]
    if differ:
        raise ValueError(
            f"the benchmark's reading differs from the reading the section gold was made with "
            f"({agreement}) in {differ}: section ids would not match the gold"
        )
    return dict(recorded)


# --- scoring ---------------------------------------------------------------------------------


def _class_f1(pairs: Sequence[tuple[str, str | None]]) -> dict[str, dict[str, float]]:
    """Per gold class: n, precision, recall and F1 (0 when undefined)."""
    stats: dict[str, dict[str, float]] = {}
    for c in sorted({g for g, _ in pairs}):
        tp = sum(1 for g, p in pairs if g == c and p == c)
        fp = sum(1 for g, p in pairs if p == c and g != c)
        fn = sum(1 for g, p in pairs if g == c and p != c)
        stats[c] = {
            "n": tp + fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        }
    return stats


def f1_scores(pairs: Sequence[tuple[str, str | None]]) -> dict[str, Any]:
    """Macro-F1 (over the gold classes) and micro-F1 of ``(gold, prediction)`` pairs. A missing
    prediction (``None``) counts against recall only."""
    if not pairs:
        return {"n": 0, "macro_f1": None, "micro_f1": None}
    correct = sum(1 for g, p in pairs if g == p)
    predicted = sum(1 for _, p in pairs if p is not None)
    precision = correct / predicted if predicted else 0.0
    recall = correct / len(pairs)
    per_class = _class_f1(pairs)
    return {
        "n": len(pairs),
        "macro_f1": sum(s["f1"] for s in per_class.values()) / len(per_class),
        "micro_f1": 2 * precision * recall / (precision + recall) if correct else 0.0,
    }


def doc_type_metrics(gold: Mapping[str, str], pred: Mapping[str, str | None]) -> dict[str, Any]:
    """Accuracy and macro-F1 over the doc types present in ``gold`` (a type that is only
    predicted adds no class; predicting it still costs the true type its recall)."""
    pairs = [(gold[doc_id], pred.get(doc_id)) for doc_id in sorted(gold)]
    if not pairs:
        return {"n": 0, "accuracy": None, "macro_f1": None, "per_type": {}}
    per_type = _class_f1(pairs)
    return {
        "n": len(pairs),
        "accuracy": sum(1 for g, p in pairs if g == p) / len(pairs),
        "macro_f1": sum(s["f1"] for s in per_type.values()) / len(per_type),
        "per_type": per_type,
    }


def group_of(label: str) -> str:
    """A catalogue label's group: the id's prefix before the first dot."""
    return label.split(".", 1)[0]


@functools.cache
def catalogue_sources() -> dict[str, str]:
    """``{"<section taxonomy>.<leaf>": catalogue leaf}`` from each catalogue leaf's
    ``metadata["from"]``: how a doc_first label (from the doc type's own section taxonomy) is
    scored against the catalogue gold."""
    from docrooter.taxonomies import builtin_taxonomy

    taxonomy = builtin_taxonomy("section_catalogue")
    sources: dict[str, str] = {}
    for group in taxonomy.children():
        for leaf in taxonomy.children(group.id):
            for source in (leaf.metadata or {}).get("from", []):
                sources.setdefault(source, leaf.id)
    return sources


CATALOGUE = "section_catalogue"


def section_taxonomy_of(classification: Any, *, flow: str, doc_type: str | None) -> str:
    """The section taxonomy a Classification's tags are labelled in: its ``section_taxonomy``,
    else the catalogue in ``sections_first`` and the doc type's mapped one in ``doc_first``."""
    recorded = _get(classification, "section_taxonomy") if classification is not None else None
    if recorded:
        return str(recorded)
    if flow == "sections_first":
        return CATALOGUE
    from docrooter.taxonomies import DOC_TYPE_TO_SECTION_TAXONOMY

    return DOC_TYPE_TO_SECTION_TAXONOMY.get(doc_type or "", "section_types/generic")


def catalogue_label(raw: str | None, *, section_taxonomy: str) -> str | None:
    """The catalogue id a predicted section label stands for, or ``None``. A label of a built-in
    per-type section taxonomy (``section_types/<family>``, the doc_first labels) is looked up in
    :func:`catalogue_sources`; a catalogue leaf or group stands for itself."""
    if raw is None:
        return None
    if section_taxonomy.startswith("section_types/"):
        family = section_taxonomy.removeprefix("section_types/")
        return catalogue_sources().get(f"{family}.{raw}")
    leaves = catalogue_leaves()
    return raw if raw in leaves or raw in set(leaves.values()) else None


def score_sections(
    doc: Any,
    gold_rows: Sequence[Mapping[str, Any]],
    tags: Iterable[Any],
    *,
    section_taxonomy: str = CATALOGUE,
) -> list[dict[str, Any]]:
    """One record per gold section of ``doc``: ``section_id``, ``gold``, ``pred`` (a catalogue
    id or ``None``) and ``status``. A gold section whose id is not in this reading is
    ``absent``, one whose title's SHA-1 differs is ``title_mismatch``; both are excluded from
    scoring. ``raw`` keeps a predicted label that is not itself the catalogue id, and
    ``unmapped`` marks one with no catalogue id. ``section_taxonomy`` is the taxonomy the tags'
    labels belong to (see :func:`catalogue_label`). ``doc`` is ``None`` for a document that
    failed to read: its gold sections are scored as untagged misses."""
    predicted = {str(tag.id): tag.label for tag in tags}
    records: list[dict[str, Any]] = []
    for row in gold_rows:
        section_id = row["section_id"]
        section = doc.sections.get(section_id) if doc is not None else None
        if doc is None:  # the document failed to read: every gold section is a miss
            status = "ok"
        elif section is None:
            status = "absent"
        elif section_sample.title_sha1(section.title) != row["title_sha1"]:
            status = "title_mismatch"
        else:
            status = "ok"
        raw = predicted.get(section_id)
        label = catalogue_label(raw, section_taxonomy=section_taxonomy)
        record: dict[str, Any] = {
            "section_id": section_id,
            "gold": row["label"],
            "pred": label,
            "status": status,
        }
        if raw is not None and raw != label:
            record["raw"] = raw
        if raw is not None and label is None:
            record["unmapped"] = True
        records.append(record)
    return records


def section_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Section macro- and micro-F1 at leaf and at group over the ``ok`` records, with the count
    excluded (reading mismatches), missing (no tag) and unmapped."""
    scored = [r for r in records if r["status"] == "ok"]
    leaf = [(r["gold"], r["pred"]) for r in scored]
    group = [(group_of(g), group_of(p) if p is not None else None) for g, p in leaf]
    leaf_scores, group_scores = f1_scores(leaf), f1_scores(group)
    return {
        "n": len(scored),
        "excluded": len(records) - len(scored),
        "missing": sum(1 for r in scored if r["pred"] is None and not r.get("unmapped")),
        "unmapped": sum(1 for r in scored if r.get("unmapped")),
        "leaf": {k: leaf_scores[k] for k in ("macro_f1", "micro_f1")},
        "group": {k: group_scores[k] for k in ("macro_f1", "micro_f1")},
    }


# --- tokens and the pick ---------------------------------------------------------------------


def _prompts(result: Mapping[str, Any]) -> dict[str, int]:
    return {d["id"]: d["tokens"]["prompt"] for d in result["documents"] if d.get("tokens")}


def token_share(result: Mapping[str, Any], baseline: Mapping[str, Any] | None) -> float | None:
    """Prompt tokens of ``result`` over those of ``baseline`` (the same backend and flow's
    ``full``/``full`` run), summed over the documents both ran; ``None`` without a baseline."""
    if baseline is None:
        return None
    mine, base = _prompts(result), _prompts(baseline)
    common = sorted(mine.keys() & base.keys())
    denominator = sum(base[d] for d in common)
    if not common or denominator <= 0:
        return None
    return sum(mine[d] for d in common) / denominator


def _key(result: Mapping[str, Any]) -> tuple[str, str, str, str, str, str]:
    """(backend label, docs, calls, flow, section level, doc level) of a run."""
    config = result["config"]
    return (
        result["backend"]["label"],
        result["docs"],
        config.get("calls", "per_item"),
        config["flow"],
        config["section_level"],
        config["doc_level"],
    )


def baseline_for(
    result: Mapping[str, Any], results: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    """The ``full``/``full`` run of the same backend, documents, calls and flow, if any."""
    label, docs, calls, flow, _, _ = _key(result)
    wanted = (label, docs, calls, flow, "full", "full")
    return next((r for r in results if _key(r) == wanted), None)


def cheapest_within(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    points: float = WITHIN_POINTS,
    level_key: str = "level",
    tokens_key: str = "tokens",
) -> Mapping[str, Any] | None:
    """The row with the fewest tokens whose ``metric`` is at most ``points`` (percentage points)
    below the ``full`` row's; ties go to the higher metric, then the lower rung. ``None``
    without a ``full`` row that has the metric."""
    full = next((r for r in rows if r[level_key] == "full" and r.get(metric) is not None), None)
    if full is None:
        return None
    floor = full[metric] - points / 100 - 1e-9
    eligible = [
        r
        for r in rows
        if r.get(metric) is not None and r.get(tokens_key) is not None and r[metric] >= floor
    ]
    return min(eligible, key=lambda r: (r[tokens_key], -r[metric], LEVELS.index(r[level_key])))


# --- one configuration -----------------------------------------------------------------------


class Config(NamedTuple):
    """One benchmark configuration."""

    flow: str
    section_level: str
    doc_level: str
    calls: str = "per_item"

    def as_dict(self) -> dict[str, str]:
        return self._asdict()


class DocInput(NamedTuple):
    """A document to classify: its id, its gold doc type and how to get the Document."""

    id: str
    gold: str
    load: Callable[[], Any]


Classify = Callable[..., Any]


def _get(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _usage(usage: Any) -> dict[str, Any] | None:
    """A TokenUsage-like value as ``{prompt, completion, calls, exact}``."""
    if usage is None:
        return None
    return {
        "prompt": int(_get(usage, "prompt") or 0),
        "completion": int(_get(usage, "completion") or 0),
        "calls": int(_get(usage, "calls") or 0),
        "exact": bool(_get(usage, "exact")),
    }


def _stage_usage(classification: Any, stage: str) -> Any:
    """The tokens of one stage (``sections`` or ``document``) of a Classification."""
    stages = _get(classification, "stage_tokens")
    return _get(stages, stage) if stages is not None else None


def _call(record: Any) -> dict[str, Any]:
    """A call record (``CallRecord``, of either stage) as ids and numbers."""
    return {
        "level": _get(record, "level"),
        "prompt_tokens": int(_get(record, "prompt_tokens") or 0),
        "completion_tokens": int(_get(record, "completion_tokens") or 0),
        "exact": bool(_get(record, "exact")),
        "truncated": bool(_get(record, "truncated")),
        "options_count": int(_get(record, "options_count") or 0),
        # the backend's exact/estimated token ratio the call's budget was divided by
        "token_ratio": float(_get(record, "token_ratio") or 1.0),
    }


def _flags(flags: Any) -> list[str]:
    if flags is None:
        return []
    if hasattr(flags, "model_dump"):
        flags = flags.model_dump()
    if isinstance(flags, Mapping):
        return sorted(str(k) for k, v in flags.items() if v)
    return sorted(str(f) for f in flags)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def check_ids_and_numbers(obj: Any, where: str = "result") -> None:
    """Raise ``ValueError`` if any key or string in ``obj`` could be text rather than an id,
    a spec, a level or a timestamp: results hold ids and numbers only."""
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            check_ids_and_numbers(key, where)
            check_ids_and_numbers(value, f"{where}.{key}")
    elif isinstance(obj, (list, tuple)):
        for k, value in enumerate(obj):
            check_ids_and_numbers(value, f"{where}[{k}]")
    elif isinstance(obj, str):
        if not _ID.match(obj):
            raise ValueError(f"{where} holds text, not an id or a number; results never hold text")
    elif obj is not None and not isinstance(obj, (bool, int, float)):
        raise ValueError(f"{where} holds a {type(obj).__name__}, not an id or a number")


def config_settings(
    settings: Mapping[str, Any], config: Config, *, spec: str, label: str, docs: str
) -> dict[str, Any]:
    """Everything a configuration's results depend on (hashed for resuming)."""
    return {**settings, "spec": spec, "label": label, "docs": docs, **config.as_dict()}


def settings_hash(settings: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()


def _cost(backend: Any) -> float | None:
    """The backend's cumulative cost estimate in US$, if it exposes one (``cost_estimate()``)."""
    cost = getattr(backend, "cost_estimate", None)
    value = cost() if callable(cost) else None
    return float(value) if isinstance(value, (int, float)) else None


def _document_record(
    item: DocInput,
    config: Config,
    *,
    backend: Any,
    classify: Classify,
    section_gold: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    error = None
    doc = None
    try:
        doc = item.load()  # a document that fails to read is one error, not the whole grid
        result = classify(
            doc,
            flow=config.flow,
            section_level=config.section_level,
            doc_level=config.doc_level,
            backend=backend,
            calls=config.calls,
        )
    except Exception as exc:  # recorded by class name only: a message may quote the document
        result, error = None, type(exc).__name__
        log(f"  {item.id}: {error}: {str(exc)[:200]}")  # stderr only, never the results
    wall = time.perf_counter() - started
    doc_type = _get(result, "doc_type") if result is not None else None
    tags = list(_get(result, "sections") or []) if result is not None else []
    document_usage = _usage(_stage_usage(result, "document")) if result is not None else None
    document_calls = [_call(c) for c in _get(result, "document_calls") or []] if result else []
    section_calls = [_call(c) for c in _get(result, "section_calls") or []] if result else []
    split = _get(result, "tokens_split") if result is not None else None
    record: dict[str, Any] = {
        "id": item.id,
        "gold": item.gold,
        "pred": doc_type,
        "correct": doc_type == item.gold,
        "decided_by": _get(result, "decided_by") if result is not None else None,
        "flags": _flags(_get(result, "flags")) if result is not None else [],
        "candidates": len(_get(result, "candidates") or []) if result is not None else 0,
        "sections_tagged": len(tags),
        "tokens": _usage(_get(result, "tokens")) if result is not None else None,
        "stage_tokens": {
            "sections": _usage(_stage_usage(result, "sections")) if result is not None else None,
            "document": document_usage,
        },
        "document_calls": document_calls,
        "section_calls": section_calls,
        # The largest exact prompt of any call, of either stage: what the 16,000 cap is about.
        "max_call_prompt": max(
            (c["prompt_tokens"] for c in section_calls + document_calls if c["exact"]),
            default=None,
        ),
        "doc_truncated": bool(_get(result, "doc_truncated")) if result is not None else False,
        "tokens_split": {str(k): int(v) for k, v in split.items()} if split else None,
        "calls_used": _get(result, "calls") if result is not None else None,
        "wall_s": round(wall, 4),
        "error": error,
    }
    if section_gold is not None:
        record["sections"] = score_sections(
            doc,
            section_gold.get(item.id, []),
            tags,
            section_taxonomy=section_taxonomy_of(result, flow=config.flow, doc_type=doc_type),
        )
    return record


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _share(flags: Sequence[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def run_metrics(records: Sequence[Mapping[str, Any]], config: Config) -> dict[str, Any]:
    """The configuration's metrics from its per-document records."""
    gold = {r["id"]: r["gold"] for r in records}
    pred = {r["id"]: r["pred"] for r in records}
    counted = [r["tokens"] for r in records if r["tokens"] is not None]
    # every call of both stages, at any level; only exact (measured) prompts count
    largest = [r["max_call_prompt"] for r in records if r.get("max_call_prompt") is not None]
    over_cap: dict[str, Any] = {
        "over_cap_share": _share([p > FULL_CAP for p in largest]),
        "over_cap_n": len(largest),
        "max_call_prompt": max(largest, default=None),
    }
    splits = [r["tokens_split"] for r in records if r["tokens_split"]]
    split_keys = sorted({k for split in splits for k in split})
    with_sections = all("sections" in r for r in records) and bool(records)
    walls = [r["wall_s"] for r in records]
    return {
        "documents": len(records),
        "errors": sum(1 for r in records if r["error"]),
        "doc_type": doc_type_metrics(gold, pred),
        "sections": (
            section_metrics([s for r in records for s in r["sections"]]) if with_sections else None
        ),
        "decided_by_rules": _share([r["decided_by"] == "rules" for r in records]),
        "prompt_tokens_mean": _mean([t["prompt"] for t in counted]),
        "completion_tokens_mean": _mean([t["completion"] for t in counted]),
        "calls_mean": _mean([t["calls"] for t in counted]),
        "tokens_exact": bool(counted) and all(t["exact"] for t in counted),
        "exact_share": _share([t["exact"] for t in counted]),
        "truncated_share": _share(["truncated" in r["flags"] for r in records]),
        "doc_truncated_share": _share([r["doc_truncated"] for r in records]),
        "tokens_split_mean": (
            {k: _mean([split.get(k, 0) for split in splits]) for k in split_keys}
            if splits
            else None
        ),
        "calls_used": sorted({str(r["calls_used"]) for r in records if r["calls_used"]}),
        **over_cap,
        "wall_s_total": round(sum(walls), 3),
        "wall_s_mean": _mean(walls),
    }


class RunAborted(RuntimeError):
    """Too many documents failed: the configuration is not written, so a re-run retries it."""


def run_configuration(
    docs_in: Sequence[DocInput],
    config: Config,
    *,
    backend: Any,
    spec: str,
    label: str,
    docs: str,
    section_gold: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    classify: Classify,
    settings: Mapping[str, Any],
    max_errors: int = MAX_ERRORS,
    code: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify every document under ``config`` and return the results (ids and numbers).
    ``code`` (see :func:`code_identity`) is recorded as is, outside the settings hash.
    ``section_gold`` is ``None`` for a doc-type-only run (held-out). More than ``max_errors``
    failed documents, or all of them, raise :class:`RunAborted`."""
    full_settings = config_settings(settings, config, spec=spec, label=label, docs=docs)
    check_ids_and_numbers(full_settings, "settings")
    started, clock = _now(), time.perf_counter()
    cost_before = _cost(backend)
    records: list[dict[str, Any]] = []
    for n, item in enumerate(docs_in, 1):
        records.append(
            _document_record(
                item, config, backend=backend, classify=classify, section_gold=section_gold
            )
        )
        errors = sum(1 for r in records if r["error"])
        if errors > max_errors:
            raise RunAborted(f"{label} {config}: {errors} documents failed; not written")
        if n % 10 == 0 or n == len(docs_in):
            log(
                f"  [{n}/{len(docs_in)}] {label} {config.flow} {config.section_level}/"
                f"{config.doc_level}"
            )
    if records and all(r["error"] for r in records):
        raise RunAborted(f"{label} {config}: every document failed; not written")
    cost_after = _cost(backend)
    result = {
        "schema": SCHEMA,
        "settings": full_settings,
        "settings_hash": settings_hash(full_settings),
        "backend": {
            "spec": spec,
            "label": label,
            "endpoint_host": getattr(backend, "endpoint_host", None) or None,
        },
        "config": config.as_dict(),
        "docs": docs,
        "code": dict(code) if code is not None else None,
        "started_at": started,
        "finished_at": _now(),
        "wall_s": round(time.perf_counter() - clock, 3),
        "cost_usd": (
            cost_after - cost_before if cost_after is not None and cost_before is not None else None
        ),
        "metrics": run_metrics(records, config),
        "documents": records,
    }
    check_ids_and_numbers(result)
    return result


def result_name(label: str, docs: str, config: Config, limit: int | None = None) -> str:
    """The results file of a configuration; a ``--limit`` smoke run gets its own name, so it
    never replaces a full run."""
    c = config
    smoke = f"__limit{limit}" if limit else ""
    return f"{label}__{docs}__{c.calls}__{c.flow}__{c.section_level}__{c.doc_level}{smoke}.json"


def _existing_hash(path: Path) -> str | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("settings_hash")
    except (OSError, ValueError):
        return None


def run_grid(
    docs_in: Sequence[DocInput],
    configs: Sequence[Config],
    *,
    out: Path,
    backend: Any,
    spec: str,
    label: str,
    docs: str,
    section_gold: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    classify: Classify,
    settings: Mapping[str, Any],
    max_errors: int = MAX_ERRORS,
    code: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Run each configuration into ``out/<label>__<docs>__<calls>__<flow>__<sl>__<dl>.json``,
    skipping one whose file exists with the same settings hash. Returns every configuration's
    path."""
    paths = []
    for config in configs:
        path = out / result_name(label, docs, config, settings.get("limit"))
        paths.append(path)
        wanted = settings_hash(config_settings(settings, config, spec=spec, label=label, docs=docs))
        if path.is_file() and _existing_hash(path) == wanted:
            log(f"skip {path.name}: done with these settings")
            continue
        log(f"run {path.name}")
        result = run_configuration(
            docs_in,
            config,
            backend=backend,
            spec=spec,
            label=label,
            docs=docs,
            section_gold=section_gold,
            classify=classify,
            settings=settings,
            max_errors=max_errors,
            code=code,
        )
        write_atomic(path, json.dumps(result, indent=1) + "\n")
    return paths


def parse_list(value: str, allowed: Sequence[str]) -> list[str]:
    """A comma list of ``allowed`` values, or ``all``."""
    if value == "all":
        return list(allowed)
    items = [v.strip() for v in value.split(",") if v.strip()]
    unknown = [v for v in items if v not in allowed]
    if unknown or not items:
        raise ValueError(f"unknown {unknown or value!r}; choose from {', '.join(allowed)} or all")
    return items


def grid_configs(
    flows: Sequence[str],
    *,
    levels: Sequence[str] | None = None,
    section_levels: Sequence[str] | None = None,
    doc_levels: Sequence[str] | None = None,
    calls: Sequence[str] = ("per_item",),
) -> list[Config]:
    """Paired levels (``levels``: the main grid) or section x doc levels (the cross-grid), each
    under every ``calls`` pattern."""
    if levels is not None:
        if section_levels is not None or doc_levels is not None:
            raise ValueError("give --level, or --section-level with --doc-level, not both")
        pairs = [(level, level) for level in levels]
    elif section_levels is None or doc_levels is None:
        raise ValueError("give --level, or both --section-level and --doc-level")
    else:
        pairs = [(s, d) for s in section_levels for d in doc_levels]
    return [Config(f, s, d, c) for f in flows for s, d in pairs for c in calls]


def default_calls(kind: str) -> list[str]:
    """The ``calls`` patterns a backend kind runs by default. An OpenAI-compatible LLM batches
    for real, so all three. Every other backend classifies one item at a time whatever it is
    told; ``batched`` changes only its token estimate (what an LLM would be sent), so it runs
    ``per_item`` and the two-step ``batched`` estimate."""
    return list(CALLS) if kind == "openai" else ["per_item", "batched"]


def code_identity(run: Callable[[list[str]], str] | None = None) -> dict[str, Any]:
    """The commit the benchmark runs from and whether the code (``src/``, ``benchmarks/``) has
    uncommitted changes; ``None`` values when git is unavailable."""

    def git(args: list[str]) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=30
        ).stdout

    run = run or git
    try:
        sha = run(["rev-parse", "HEAD"]).strip()
        dirty = bool(
            run(
                ["status", "--porcelain", "--untracked-files=no", "--", "src", "benchmarks"]
            ).strip()
        )
    except (OSError, subprocess.SubprocessError, KeyError):
        return {"git_sha": None, "dirty": None}
    return {"git_sha": sha, "dirty": dirty}


def code_paths() -> list[Path]:
    """The code a configuration's results depend on: the package and this benchmark."""
    return sorted(
        [
            *(REPO_ROOT / "src" / "docrooter").rglob("*.py"),
            *(REPO_ROOT / "src" / "docrooter" / "taxonomies").rglob("*.json"),
            Path(__file__).resolve(),
            *CLASSIFICATION_DIR.glob("*.py"),
        ]
    )


def code_fingerprint(paths: Iterable[Path]) -> str:
    """SHA-1 over the paths and contents: part of the settings hash, so a code fix mid-grid
    re-runs the configurations it could change (a docs-only commit does not)."""
    digest = hashlib.sha1()
    for path in paths:
        digest.update(path.name.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


# --- documents -------------------------------------------------------------------------------


def _loader(path: Path, *, cache: bool = True) -> Callable[[], Any]:
    """Reads the document on first use and keeps it (one reading per process), or, without
    ``cache``, reads it again on every use (less memory, one reading per configuration)."""

    def read() -> Any:
        return section_sample.read_document(path)

    return functools.cache(read) if cache else read


def gold_inputs(
    gold: Mapping[str, str], pdf_dir: Path = PDF_DIR, *, cache: bool = True
) -> list[DocInput]:
    missing = [d for d in gold if not (pdf_dir / f"{d}.pdf").is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} gold PDF(s) not in {pdf_dir}: {missing[:5]}")
    return [DocInput(d, gold[d], _loader(pdf_dir / f"{d}.pdf", cache=cache)) for d in sorted(gold)]


def heldout_inputs(corpus: Path = HELDOUT_DIR, *, cache: bool = True) -> list[DocInput]:
    """The frozen held-out corpus: gold is each record's ``label``."""
    from benchmarks.corpora import read_manifest

    return [
        DocInput(r["id"], r["label"], _loader(corpus / r["file"], cache=cache))
        for r in read_manifest(corpus)
    ]


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def public_spec(spec: str) -> str:
    """``spec`` with the home directory written as ``~``: results never hold it."""
    home = str(Path.home())
    return spec.replace(home, "~") if home and home != "/" else spec


def default_label(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", public_spec(spec)).strip("-.") or "backend"


# --- report ----------------------------------------------------------------------------------

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{100 * value:.1f}"


def _tokens(value: float | None) -> str:
    return "–" if value is None else f"{value:,.0f}"


def _metric(result: Mapping[str, Any], *path: str) -> Any:
    value: Any = result["metrics"]
    for key in path:
        if value is None:
            return None
        value = value.get(key)
    return value


def _sorted(results: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    def key(r: Mapping[str, Any]) -> tuple[Any, ...]:
        label, docs, calls, flow, sl, dl = _key(r)
        order = CALLS.index(calls) if calls in CALLS else len(CALLS)
        return (label, docs, order, FLOWS.index(flow), LEVELS.index(sl), LEVELS.index(dl))

    return sorted(results, key=key)


def _kind(result: Mapping[str, Any]) -> str:
    return "exact" if result["metrics"].get("tokens_exact") else "estimated"


def _model_name(spec: str) -> str:
    """A readable model name from an ``openai:`` spec (a Hugging Face snapshot path or an id)."""
    model = spec.partition(":")[2]
    found = re.search(r"models--([^/]+)", model)
    return found.group(1).replace("--", "/") if found else model.rstrip("/").rsplit("/", 1)[-1]


def _llm_disclosure(results: Sequence[Mapping[str, Any]]) -> list[str]:
    """One line per LLM backend label in the results: what model, and whether it was local."""
    seen: dict[str, Mapping[str, Any]] = {}
    for r in results:
        if str(r["backend"].get("spec", "")).startswith("openai:"):
            seen.setdefault(r["backend"]["label"], r["backend"])
    lines = []
    for label, backend in sorted(seen.items()):
        host = backend.get("endpoint_host")
        where = (
            "a local stand-in on a local server, not a hosted model"
            if host is None or host in _LOCAL_HOSTS
            else f"the hosted endpoint `{host}`"
        )
        lines.append(
            f"- **LLM rows `{label}`** are `{_model_name(str(backend['spec']))}` through the "
            f"OpenAI-compatible backend, on {where} (the same code path as a hosted model)."
        )
    return lines


def _highlights_conflict(doc_gold: Mapping[str, str]) -> str:
    """The gold sections of non-financial reports labelled ``financial.financial_highlights``
    under the catalogue's earlier wording, which the sharpened definition now sends to
    ``general.summary`` (ids only)."""
    refs = [
        f"`{doc_id}#{row['section_id']}`"
        for doc_id, rows in load_section_gold().items()
        if doc_gold.get(doc_id) == "government.government_report"
        for row in rows
        if row["label"] == "financial.financial_highlights"
    ]
    if not refs:
        return ""
    return (
        f" {len(refs)} government-report gold sections ({', '.join(refs)}) were labelled "
        "`financial.financial_highlights` under the earlier wording and now disagree with the "
        "definition; a model that follows it is scored wrong on them."
    )


def _heldout_types(results: Sequence[Mapping[str, Any]]) -> str:
    """How many doc types the held-out rows cover, when there are any."""
    heldout = [r for r in results if r.get("docs") == "heldout"]
    if not heldout:
        return ""
    counts = collections.Counter(d["gold"] for d in heldout[0].get("documents", []))
    if not counts:
        return ""
    return (
        f" The held-out corpus covers {len(counts)} types, {min(counts.values())} to "
        f"{max(counts.values())} documents each."
    )


def _code_versions(results: Sequence[Mapping[str, Any]]) -> list[str]:
    """A warning when the rows were run at more than one code version, or on a dirty checkout
    (empty when they all share one clean version)."""
    versions = {
        (
            (r.get("settings") or {}).get("code_fingerprint"),
            (r.get("code") or {}).get("git_sha"),
            bool((r.get("code") or {}).get("dirty")),
        )
        for r in results
    }
    dirty = any(v[2] for v in versions)
    if len(versions) <= 1 and not dirty:
        return []
    shas = sorted({(v[1] or "unknown")[:12] for v in versions})
    return [
        f"**Code versions.** These rows were run at {len(versions)} code versions "
        f"(git {', '.join(shas)}){'; some on a dirty checkout' if dirty else ''}: compare rows "
        f"from different versions with care.",
        "",
    ]


def _disclosures(results: Sequence[Mapping[str, Any]] = ()) -> list[str]:
    from docrooter.taxonomies import DOC_TYPE_TO_SECTION_TAXONOMY, builtin_taxonomy

    agreement = json.loads(AGREEMENT_PATH.read_text(encoding="utf-8"))
    summary = json.loads(SAMPLE_SUMMARY_PATH.read_text(encoding="utf-8"))
    gold_data = json.loads(DOC_TYPE_GOLD_PATH.read_text(encoding="utf-8"))
    doc_gold = load_doc_type_gold()
    types = sorted(set(doc_gold.values()))
    all_types = [n.id for n in builtin_taxonomy("doc_types").leaves()]
    without = summary["documents_without_sections"]
    reachable = len(set(catalogue_sources().values()))
    generic = "section_types/generic"
    generic_docs = sum(
        1 for t in doc_gold.values() if DOC_TYPE_TO_SECTION_TAXONOMY.get(t, generic) == generic
    )
    generic_leaves = len(list(builtin_taxonomy(generic).leaves()))
    return [
        f"- **Section gold is Claude-annotated, for evaluation only.** "
        f"{agreement['sections']} level-1 and level-2 sections of the "
        f"{len(gold_data['documents'])} documents, labelled blind against the section catalogue; "
        f"{agreement['double_annotated']} were double-annotated: Cohen's κ "
        f"{agreement['leaf']['kappa']:.3f} at leaf and {agreement['group']['kappa']:.3f} at group, "
        f"{agreement['adjudicated']} disagreements adjudicated. The annotators were all Claude "
        f"(the same underlying model), so κ measures consistency, not correctness.",
        "- **Doc-type rules were revised once after a reviewer checked them against the section "
        "gold. Gold-split doc-type numbers are therefore development numbers; the held-out run "
        "is the clean measure.** The revision was made from the doc-type definitions (an "
        "ordinary contract clause no longer makes a terms policy, and a contract no longer needs "
        "a titled parties or signatures section), not tuned to the gold labels; two catalogue "
        "definitions were sharpened with it (`contract.acceptable_use` now covers the user "
        "terms of published terms, and `financial.financial_highlights` sends a non-financial "
        "report's highlights to `general.summary`), after the section gold was labelled."
        f"{_highlights_conflict(doc_gold)}",
        "- **Rules N1/N2 were changed after viewing gold-split results; gold doc-type numbers "
        "are development numbers; the held-out set is the clean measure.** N1: the rules narrow "
        "or decide only when at least max(2, 20% of the tagged sections) sections were tagged "
        "with confidence, otherwise every type is offered (`thin_evidence`). N2: a weakly "
        "supported lone type (`low_support`) is offered with every type, not with its siblings.",
        f"- **Doc-type gold comes from the source**: the manifest's source-assigned type "
        f"(`benchmarks/classification/doc_type_gold.json`), with "
        f"{len(gold_data['corrections'])} correction(s).",
        f"- **Only {len(types)} of the {len(all_types)} doc types are in the corpus** "
        f"({', '.join(f'`{t}`' for t in types)}); the rules and the classifier are not validated "
        f"on the other {len(all_types) - len(types)}. Macro-F1 is over the gold types; "
        f"section macro-F1 is over the gold leaves.{_heldout_types(results)}",
        *_llm_disclosure(results),
        "- **Tokens are exact or estimated, as the Tokens column says.** Exact counts come from "
        "the API's `usage` (LLM backends). Keyword and embedding backends make no LLM call: "
        "their tokens estimate what the same views and options would cost an LLM. Section "
        "tagging tokens count in each document's tokens. The split column (options / content / "
        "instructions) is always an estimate, even next to exact totals: it shows how much of "
        "the prompt the option lists take.",
        "- **Calls.** `per_item` asks one item per call; `batched` asks a whole stage per call "
        "with each option listed once (two steps: groups, then the leaves of each group); "
        "`batched_flat` lists every leaf once. Keyword and embedding backends always classify "
        "one item at a time: their `batched` rows differ from `per_item` only in the token "
        "estimate (what an LLM would be sent).",
        f"- **Over the cap.** Every call is budgeted to a {FULL_CAP:,}-token target prompt: "
        f"`full` cuts the document view head-first to fit it, and batched section calls are "
        f"packed (at most {CALL_VIEWS} views, and an estimated 85% of the cap, per call), a "
        f"view too large for any call being cut and sent alone. The over-cap table covers every "
        f"call of both stages, for each run with a `full` level: the share of documents with an "
        f"exact call over {FULL_CAP:,} tokens, the largest exact call (estimated runs have none), "
        f"the share whose deciding document view was cut, and the share with any view (section "
        f"or document) cut.",
        f"- **At most {MAX_SECTIONS} sections** (level 1 first, then level 2, in reading order) "
        f"are tagged per document. A gold section past the cap is untagged and counts as a miss: "
        f"the Sections column gives scored / excluded (reading mismatch) / untagged.",
        f"- **{len(without)} documents have no level-1/2 sections** in the reading (profile "
        f"`fast`), so they hold no section gold; `sections_first` falls back to `doc_first` "
        f"for them (flag `no_sections`).",
        f"- **doc_first section labels** come from the predicted doc type's own section taxonomy "
        f"and are scored through the catalogue's `from` mapping, which reaches only "
        f"{reachable} of the {len(catalogue_leaves())} catalogue leaves (from "
        f"{len(catalogue_sources())} per-type leaves). The {generic_docs} gold documents whose "
        f"type maps to `{generic}` ({generic_leaves} leaves) are capped hardest, so doc_first's "
        f"section F1 is structurally below sections_first's. A label with no catalogue leaf "
        f"counts as no prediction (it costs recall, not precision).",
        "- **Token share** is a run's prompt tokens over those of the same backend, calls and "
        "flow's `full`/`full` run, summed over the same documents.",
        "- **Failed documents** (the Errors column) count as wrong in accuracy and carry no "
        "tokens, so they are left out of the token means.",
    ]


def _split(result: Mapping[str, Any]) -> str:
    split = _metric(result, "tokens_split_mean")
    if not split:
        return "–"
    return " / ".join(_tokens(split.get(k)) for k in ("options", "content", "instructions"))


def _section_counts(result: Mapping[str, Any]) -> str:
    sections = _metric(result, "sections")
    if not sections:
        return "–"
    return f"{sections['n']} / {sections['excluded']} / {sections['missing']}"


def _main_table(
    runs: Sequence[Mapping[str, Any]], all_runs: Sequence[Mapping[str, Any]]
) -> list[str]:
    lines = [
        "| Backend | Calls | Flow | Level | Doc acc | Doc macro-F1 | Sec leaf macro-F1 "
        "| Sec leaf micro-F1 | Sec group macro-F1 | Sec group micro-F1 "
        "| Sections scored / excl. / untagged | Prompt tokens/doc | Token share vs full "
        "| Est. split options / content / instr. | Decided by rules | Any view truncated "
        "| Tokens | Wall s/doc | Errors |",
        "|---|---|---|---|" + "---:|" * 6 + "---|---:|---:|---|---:|---:|---|---:|---:|",
    ]
    for r in runs:
        label, _, calls, flow, level, _ = _key(r)
        share = token_share(r, baseline_for(r, all_runs))
        wall = _metric(r, "wall_s_mean")
        lines.append(
            f"| {label} | {calls} | {flow} | {level} "
            f"| {_pct(_metric(r, 'doc_type', 'accuracy'))} "
            f"| {_pct(_metric(r, 'doc_type', 'macro_f1'))} "
            f"| {_pct(_metric(r, 'sections', 'leaf', 'macro_f1'))} "
            f"| {_pct(_metric(r, 'sections', 'leaf', 'micro_f1'))} "
            f"| {_pct(_metric(r, 'sections', 'group', 'macro_f1'))} "
            f"| {_pct(_metric(r, 'sections', 'group', 'micro_f1'))} "
            f"| {_section_counts(r)} "
            f"| {_tokens(_metric(r, 'prompt_tokens_mean'))} | {_pct(share)} | {_split(r)} "
            f"| {_pct(_metric(r, 'decided_by_rules'))} | {_pct(_metric(r, 'truncated_share'))} "
            f"| {_kind(r)} | {'–' if wall is None else f'{wall:.2f}'} "
            f"| {_metric(r, 'errors') or 0} |"
        )
    return lines


def _ladder_rows(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "level": r["config"]["doc_level"],
            "doc_accuracy": _metric(r, "doc_type", "accuracy"),
            "section_macro_f1": _metric(r, "sections", "leaf", "macro_f1"),
            "tokens": _metric(r, "prompt_tokens_mean"),
        }
        for r in runs
    ]


def _pick_cell(rows: Sequence[Mapping[str, Any]], metric: str) -> str:
    pick = cheapest_within(rows, metric)
    if pick is None:
        return "no `full` run"
    full = next(r for r in rows if r["level"] == "full")
    return (
        f"`{pick['level']}`: {_pct(pick[metric])} at {_tokens(pick['tokens'])} tokens "
        f"(`full`: {_pct(full[metric])} at {_tokens(full['tokens'])})"
    )


def _groups(
    runs: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], list[Mapping[str, Any]]]:
    """Runs by (backend label, calls, flow), in order."""
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for r in runs:
        label, _, calls, flow, _, _ = _key(r)
        groups.setdefault((label, calls, flow), []).append(r)
    return groups


def _picks(runs: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| Backend | Calls | Flow | Document ladder (doc-type accuracy) "
        "| Section ladder (section leaf macro-F1) |",
        "|---|---|---|---|---|",
    ]
    for (label, calls, flow), group in _groups(runs).items():
        rows = _ladder_rows(group)
        lines.append(
            f"| {label} | {calls} | {flow} | {_pick_cell(rows, 'doc_accuracy')} "
            f"| {_pick_cell(rows, 'section_macro_f1')} |"
        )
    return lines


def _cross_tables(runs: Sequence[Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    for (label, calls, flow), group in _groups(runs).items():
        if all(r["config"]["section_level"] == r["config"]["doc_level"] for r in group):
            continue
        cells = {(r["config"]["section_level"], r["config"]["doc_level"]): r for r in group}
        lines += [
            f"**{label}, {calls}, {flow}**: doc acc / section leaf macro-F1 / prompt tokens per "
            f"document; rows are the section level, columns the document level.",
            "",
            "| Section \\ document | " + " | ".join(LEVELS) + " |",
            "|---|" + "---:|" * len(LEVELS),
        ]
        for s in LEVELS:
            row = []
            for d in LEVELS:
                cell = cells.get((s, d))
                row.append(
                    "–"
                    if cell is None
                    else f"{_pct(_metric(cell, 'doc_type', 'accuracy'))} / "
                    f"{_pct(_metric(cell, 'sections', 'leaf', 'macro_f1'))} / "
                    f"{_tokens(_metric(cell, 'prompt_tokens_mean'))}"
                )
            lines.append(f"| {s} | " + " | ".join(row) + " |")
        lines.append("")
    return lines


def _over_cap(runs: Sequence[Mapping[str, Any]]) -> list[str]:
    full = [r for r in runs if "full" in (r["config"]["section_level"], r["config"]["doc_level"])]
    if not full:
        return []
    lines = [
        "| Backend | Calls | Docs | Flow | Section level | Doc level | Documents with an exact "
        f"call | Any call over {FULL_CAP:,} | Largest exact call | Document view truncated "
        "| Any view truncated |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in full:
        label, docs, calls, flow, section_level, doc_level = _key(r)
        n = _metric(r, "over_cap_n") or 0
        largest = _metric(r, "max_call_prompt")
        lines.append(
            f"| {label} | {calls} | {docs} | {flow} | {section_level} | {doc_level} | {n} "
            f"| {_pct(_metric(r, 'over_cap_share')) if n else 'n/a (estimated)'} "
            f"| {f'{largest:,}' if n and largest is not None else 'n/a'} "
            f"| {_pct(_metric(r, 'doc_truncated_share'))} "
            f"| {_pct(_metric(r, 'truncated_share'))} |"
        )
    return lines


def _heldout_table(runs: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| Backend | Calls | Flow | Section level | Doc level | Doc acc | Doc macro-F1 "
        "| Prompt tokens/doc | Tokens | Documents |",
        "|---|---|---|---|---|---:|---:|---:|---|---:|",
    ]
    for r in runs:
        label, _, calls, flow, sl, dl = _key(r)
        lines.append(
            f"| {label} | {calls} | {flow} | {sl} | {dl} "
            f"| {_pct(_metric(r, 'doc_type', 'accuracy'))} "
            f"| {_pct(_metric(r, 'doc_type', 'macro_f1'))} "
            f"| {_tokens(_metric(r, 'prompt_tokens_mean'))} | {_kind(r)} "
            f"| {_metric(r, 'documents')} |"
        )
    return lines


def render_report(results: Sequence[Mapping[str, Any]]) -> str:
    """The markdown of ``docs/benchmarks-classification.md`` from the results files' contents."""
    complete = [r for r in results if not (r.get("settings") or {}).get("limit")]
    partial = len(results) - len(complete)
    gold = _sorted(r for r in complete if r["docs"] == "gold")
    heldout = _sorted(r for r in complete if r["docs"] == "heldout")
    main = [r for r in gold if r["config"]["section_level"] == r["config"]["doc_level"]]
    lines = [
        "# Classification benchmark: accuracy against tokens",
        "",
        "How much accuracy each rung of the context ladder (`headings`, `terms`, `lead`, `full`) "
        "buys, and at what token cost, against sending the whole document (spec G5). The set is "
        "the 113 public documents of the gold split; the frozen held-out corpus is scored for doc "
        "type only, once, at the end. Generated by "
        "`python -m benchmarks.classification_bench report` from the results files; do not edit "
        "by hand.",
        "",
    ]
    if not complete:
        lines.append("**Results pending.** The grid has not been run yet.")
    else:
        lines += _code_versions(complete)
        lines += ["## Accuracy against tokens (main grid: section level = document level)", ""]
        lines += _main_table(main, gold) if main else ["No main-grid runs."]
        lines += [
            "",
            "Doc acc is doc-type accuracy; Sec are the section-tag F1 scores at catalogue leaf "
            "and group. Percentages, except counts, tokens and seconds.",
            "",
            "## Cheapest level within 2 points of full",
            "",
            "The rung with the fewest prompt tokens per document whose score is at most 2 points "
            "below the same backend, calls and flow's `full` run (spec §6.2), chosen "
            "independently for the document ladder (doc-type accuracy) and the section ladder "
            "(section leaf macro-F1).",
            "",
        ]
        lines += _picks(main) if main else ["No main-grid runs."]
        cross = _cross_tables(gold)
        if cross:
            lines += ["", "## Cross-grid: section level x document level", "", *cross]
        over = _over_cap(gold + heldout)
        if over:
            lines += ["", f"## Over the {FULL_CAP:,}-token cap at `full`", "", *over]
        if heldout:
            lines += [
                "",
                "## Held-out corpus (doc type only, run once)",
                "",
                *_heldout_table(heldout),
            ]
    if partial:
        lines += ["", f"{partial} partial run(s) (`--limit`) are left out."]
    lines += ["", "## Disclosures", "", *_disclosures(complete), ""]
    return "\n".join(lines)


def load_results(directory: Path) -> list[dict[str, Any]]:
    """Every results file under ``directory`` (files with a ``config`` and ``metrics``)."""
    results = []
    for path in sorted(directory.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "config" in data and "metrics" in data:
            results.append(data)
    return results


# --- CLI -------------------------------------------------------------------------------------


def _classify() -> Classify:
    from docrooter import classify  # V4; imported here so the module loads without it

    return classify


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from docrooter import __version__
    from docrooter.backends.registry import parse_spec

    kind = parse_spec(args.backend)[0]
    if kind == "jev" and not args.allow_jev:
        parser.error("Jev runs only with --allow-jev (it is optional and costs money)")
    label = args.label or default_label(args.backend)
    if not _LABEL.match(label):
        parser.error(f"--label must match {_LABEL.pattern}")
    try:
        flows = parse_list(args.flow, FLOWS)
        configs = grid_configs(
            flows,
            levels=parse_list(args.level, LEVELS) if args.level else None,
            section_levels=parse_list(args.section_level, LEVELS) if args.section_level else None,
            doc_levels=parse_list(args.doc_level, LEVELS) if args.doc_level else None,
            calls=parse_list(args.calls, CALLS) if args.calls else default_calls(kind),
        )
    except ValueError as exc:
        parser.error(str(exc))
    if kind == "keyword":
        _guard.apply_env_caps()
    else:
        _guard.single_model_process(
            "classification_bench",
            allow_concurrent=args.allow_concurrent,
            ignore=_guard.LLM_SERVER if kind == "openai" else None,
        )
    reading = check_reading()
    if args.docs == "gold":
        doc_type_gold = load_doc_type_gold()
        docs_in = gold_inputs(doc_type_gold, cache=not args.no_cache)
        section_gold: dict[str, list[dict[str, Any]]] | None = load_section_gold()
        gold_hashes = {
            "doc_type_gold_sha1": _sha1(DOC_TYPE_GOLD_PATH),
            "section_gold_sha1": _sha1(SECTION_GOLD_PATH),
        }
    else:
        docs_in = heldout_inputs(cache=not args.no_cache)
        section_gold = None
        gold_hashes = {"heldout_manifest_sha1": _sha1(HELDOUT_DIR / "manifest.jsonl")}
    if args.limit:
        docs_in = docs_in[: args.limit]
    settings = {
        "schema": SCHEMA,
        "reading": {k: reading[k] for k in READING_KEYS},
        "docrooter_version": __version__,
        "limit": args.limit,
        "code_fingerprint": code_fingerprint(code_paths()),
        **gold_hashes,
    }
    code = code_identity()
    from docrooter.backends.base import close_backend
    from docrooter.backends.registry import resolve_backend

    classify = _classify()
    backend = resolve_backend(args.backend)
    try:
        paths = run_grid(
            docs_in,
            configs,
            out=args.out,
            backend=backend,
            spec=public_spec(args.backend),
            label=label,
            docs=args.docs,
            section_gold=section_gold,
            classify=classify,
            settings=settings,
            max_errors=args.max_errors,
            code=code,
        )
    finally:
        close_backend(backend)
    print(f"{len(paths)} configuration(s) in {args.out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    gold = sub.add_parser("gold", help="write the doc-type gold from the manifest")
    gold.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    gold.add_argument("--out", type=Path, default=DOC_TYPE_GOLD_PATH)
    run = sub.add_parser("run", help="run configurations")
    run.add_argument("--backend", required=True, help="backend spec, e.g. keyword")
    run.add_argument("--label", help="short backend name for the results (e.g. qwen3-4b-local)")
    run.add_argument("--flow", default="all", help="doc_first, sections_first, a list or all")
    run.add_argument("--level", help="paired section and document level(s), or all")
    run.add_argument("--section-level", help="section level(s) for the cross-grid, or all")
    run.add_argument("--doc-level", help="document level(s) for the cross-grid, or all")
    run.add_argument("--docs", choices=DOCS_SPLITS, default="gold")
    run.add_argument(
        "--out",
        type=Path,
        default=RESULTS_ROOT / datetime.date.today().isoformat() / "classification",
    )
    run.add_argument(
        "--calls",
        help="per_item, batched, batched_flat, a list or all (default: all three for openai "
        "backends; per_item,batched for the others, whose batched run is only a token estimate)",
    )
    run.add_argument("--limit", type=int, help="only the first N documents (a smoke run)")
    run.add_argument(
        "--no-cache",
        action="store_true",
        help="read each document again per configuration instead of keeping it in memory",
    )
    run.add_argument("--max-errors", type=int, default=MAX_ERRORS)
    run.add_argument("--allow-jev", action="store_true", help="allow a jev backend")
    _guard.add_guard_argument(run)
    report = sub.add_parser("report", help="write the markdown report")
    report.add_argument("--results", type=Path, required=True, help="the results directory")
    report.add_argument("--out", type=Path, default=DOCS_PATH)
    args = parser.parse_args(argv)
    if args.command == "gold":
        data = doc_type_gold_from_manifest(args.manifest)
        if args.out.is_file():  # keep the controller's corrections
            data["corrections"] = json.loads(args.out.read_text(encoding="utf-8"))["corrections"]
        check_ids_and_numbers(data["documents"])
        write_atomic(args.out, json.dumps(data, indent=2) + "\n")
        print(f"{len(data['documents'])} documents in {args.out}")
        return 0
    if args.command == "run":
        return _run(args, parser)
    if not args.results.is_dir():
        parser.error(f"--results {args.results} is not a directory")
    write_atomic(args.out, render_report(load_results(args.results)))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
