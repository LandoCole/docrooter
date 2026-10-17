"""Section-normalisation benchmark: ``normalize_sections`` vs agent-assigned silver labels.

``PYTHONPATH=src uv run --extra models --extra pdf --extra jev --extra bench \
    python -m benchmarks.normalize_bench --docs docs/benchmarks-pdf.md``

The silver set is ``benchmarks/corpus/section_gold.jsonl``: a JSON header line (``{"_meta":
{...}}``, with the labelling time and the src commit the documents were loaded at), then one line
per level-1/2 section ``{doc_id, section_title, page, level, canonical, note?}``. A null
``canonical`` marks parser junk (not a real section): it is counted, never scored.

Each gold document is loaded fresh with the default engine and the gold rows are matched to its
sections by (normalised title, page); rows that no longer match are reported, not fatal. Every
system runs ``normalize_sections(doc, doc_type=<manifest label>, backend=...)`` per document; one
extra run uses ``doc_type=None`` (the auto doc-type path) with the default backend. Results go to
``benchmarks/results/<date>/normalize.json`` and, with ``--docs``, to the generated block of the
``## Section normalisation`` section of ``docs/benchmarks-pdf.md``.

``--configs`` picks settings from :data:`CONFIGS` (section view x built-in heading synonyms on or
off). Each run's summary goes to ``variants``; only :data:`FINAL_CONFIG` (the package default)
fills ``systems`` and gets the auto doc-type run. ``baseline_p10`` keeps the original P10 run
(title + lead at 96 tokens, no synonyms) untouched, for the before/after table.

Timing: each system first normalises one tiny synthetic document untimed (model load; reported
as ``load_s``). Seconds per section is the wall time of each document's ``normalize_sections``
call (label encoding and view rendering included) summed over documents, divided by the number
of sections it labelled (junk included, since the system labels those too).
"""

from __future__ import annotations

import argparse
import datetime
import functools
import json
import os
import re
import time
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks import _guard, corpora, jev_chunks, pdf_bench
from benchmarks.toc import git_revision
from docrooter.document import Document
from docrooter.evaluate.metrics import leaf_prf
from docrooter.taxonomies import DOC_TYPE_TO_SECTION_TAXONOMY, builtin_taxonomy
from docrooter.taxonomy import Taxonomy
from docrooter.views import SectionView

GOLD_PATH = pdf_bench.CORPUS / "section_gold.jsonl"
GOLD_FIELDS = ("doc_id", "section_title", "page", "level", "canonical")
LEVELS = (1, 2)
SYSTEMS: dict[str, str] = {
    "ultrabert": f"embedding:{pdf_bench.ULTRABERT}",
    "bge-small": f"embedding:{pdf_bench.BGE}",
    "jev": f"jev:{pdf_bench.JEV_MODEL}",
}
AUTO_SYSTEM = "ultrabert"  # backend of the doc_type=None run (the library default)
TABLE_DOCS = ("doc_027", "doc_002")  # per-entry tables: one research paper, one annual report
SECTION = "## Section normalisation"
BEGIN, END = "<!-- normalize-benchmark:begin -->", "<!-- normalize-benchmark:end -->"
log = pdf_bench.log


@dataclass(frozen=True)
class Config:
    """One ``normalize_sections`` setting: the section view and whether the built-in section
    taxonomies keep their heading-synonym ``examples`` (False strips them: the P10 setting)."""

    view: SectionView
    synonyms: bool


VIEWS: dict[str, SectionView] = {
    "title+lead@96": SectionView(fields=("title", "lead"), max_tokens=96),  # P10 default
    "title": SectionView(fields=("title",), max_tokens=96),
    "path+title": SectionView(fields=("path", "title"), max_tokens=96),
    "title+lead@48": SectionView(fields=("title", "lead"), max_tokens=48),
}
CONFIGS: dict[str, Config] = {
    f"{view}{'+syn' if synonyms else ''}": Config(VIEWS[view], synonyms)
    for synonyms in (False, True)
    for view in VIEWS
}
BASELINE_CONFIG = "title+lead@96"  # the P10 setting: no synonyms, title + lead at 96 tokens
FINAL_CONFIG = "path+title+syn"  # the package default (NORMALIZE_VIEW + built-in synonyms)


# --- gold --------------------------------------------------------------------------------


def read_gold(path: Path = GOLD_PATH) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``(header, rows)`` of a silver-label file. The first non-blank line must be the
    ``{"_meta": {...}}`` header; every row needs :data:`GOLD_FIELDS` (else ``ValueError``)."""
    header: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if header is None:
                if not isinstance(record, dict) or "_meta" not in record:
                    raise ValueError(f"{path}: line {number} is not a {{'_meta': ...}} header")
                header = record["_meta"]
                continue
            missing = [f for f in GOLD_FIELDS if f not in record]
            if missing:
                raise ValueError(f"{path}: line {number} lacks {', '.join(missing)}")
            rows.append(record)
    if header is None:
        raise ValueError(f"{path}: empty file, no header")
    return header, rows


_NUMBERING = re.compile(
    r"^(?:(?:item|note|part|section|article)\s+)?(?:\d+[a-z]?(?:\.\d+)*|[ivxl]+|[a-z])[.:)]?\s+"
)


def heading_key(title: str) -> str:
    """:func:`title_key` without one leading number ("3.1", "Item 1A.", "Note 2.", "A.") and
    with punctuation turned into spaces: the key for heading-vs-example overlap."""
    key = _NUMBERING.sub("", title_key(title))
    return " ".join(re.sub(r"[^\w\s]", " ", key).split())


@functools.cache
def _example_keys(taxonomy: str) -> frozenset[str]:
    nodes = builtin_taxonomy(taxonomy).nodes.values()
    return frozenset(heading_key(e) for node in nodes for e in node.examples)


def in_examples(row: Mapping[str, Any]) -> bool:
    """Whether the row's heading equals (by :func:`heading_key`) a heading-synonym example of
    any node of its gold taxonomy, so the synonyms may have handed the model its title."""
    return heading_key(row["section_title"]) in _example_keys(row["taxonomy"])


def title_key(title: str) -> str:
    """Casefolded title with whitespace collapsed (numbering kept: it disambiguates)."""
    return " ".join(title.split()).casefold()


@dataclass
class Match:
    pairs: list[tuple[dict[str, Any], str]] = field(default_factory=list)  # (gold row, section id)
    unmatched_gold: list[dict[str, Any]] = field(default_factory=list)
    unlabelled: list[str] = field(default_factory=list)  # section ids with no gold row


def match_gold(gold: Sequence[Mapping[str, Any]], sections: Sequence[Any]) -> Match:
    """Pair gold rows with sections (objects with ``id``, ``title``, ``page_start``) by
    ``(title_key, page)``; repeats of one key pair up in reading order."""
    free: dict[tuple[str, int | None], deque[str]] = defaultdict(deque)
    for s in sections:
        free[(title_key(s.title), s.page_start)].append(s.id)
    out = Match()
    used: set[str] = set()
    for row in gold:
        queue = free.get((title_key(row["section_title"]), row["page"]))
        if queue:
            sid = queue.popleft()
            used.add(sid)
            out.pairs.append((dict(row), sid))
        else:
            out.unmatched_gold.append(dict(row))
    out.unlabelled = [s.id for s in sections if s.id not in used]
    return out


# --- metrics -----------------------------------------------------------------------------


def score(rows: Sequence[Mapping[str, Any]], *, qualify: bool = False) -> dict[str, Any]:
    """Accuracy and macro F1 over rows with a gold label; junk rows (gold None) are counted.

    A prediction is correct when it names the gold node of the gold ``taxonomy``: a row whose
    ``pred_taxonomy`` differs (the auto doc-type run chose another taxonomy) is always wrong.
    Macro F1 averages over the gold classes (``leaf_prf``). With ``qualify`` each class is
    prefixed by its taxonomy, so pooled documents of different types never share one.
    """
    scored = [r for r in rows if r["gold"] is not None]

    def gold_of(r: Mapping[str, Any]) -> tuple[str | None, str]:
        return r.get("taxonomy"), r["gold"]

    def pred_of(r: Mapping[str, Any]) -> tuple[str | None, str | None]:
        return r.get("pred_taxonomy", r.get("taxonomy")), r["pred"]

    out: dict[str, Any] = {
        "n_scored": len(scored),
        "n_junk": len(rows) - len(scored),
        "n_correct": sum(pred_of(r) == gold_of(r) for r in scored),
        "accuracy": None,
        "macro_f1": None,
    }
    if scored:

        def path(taxonomy: str | None, label: str | None) -> list[str]:
            if label is None:
                return []
            return [f"{taxonomy}:{label}" if qualify else label]

        gold = [path(*gold_of(r)) for r in scored]
        # Unqualified, a node of another taxonomy must still never count as the gold class.
        pred = [
            path(*pred_of(r)) if qualify or pred_of(r)[0] == gold_of(r)[0] else ["<other>"]
            for r in scored
        ]
        out["accuracy"] = out["n_correct"] / len(scored)
        out["macro_f1"] = leaf_prf(pred, gold)[2]
    return out


def jev_cost(tokens: int | None) -> float | None:
    """US dollars for ``tokens`` Jev input tokens (output tokens are free)."""
    return None if tokens is None else tokens * pdf_bench.JEV_USD_PER_M_INPUT / 1e6


# --- running -----------------------------------------------------------------------------


def load_documents(doc_ids: Sequence[str], corpus: Path) -> dict[str, Document]:
    """Fresh ``load_pdf`` (default engine) of each document, so matching sees the current src."""
    from docrooter.pdf.load import load_pdf

    entries = {e["id"]: e for e in pdf_bench.manifest(corpus)}
    return {i: load_pdf(corpus / entries[i]["file"]) for i in doc_ids}


def without_synonyms(name: str) -> Taxonomy:
    """The built-in taxonomy ``name`` with every node's ``examples`` removed."""
    data = builtin_taxonomy(name).to_dict()
    for node in data["nodes"]:
        node["examples"] = []
    return Taxonomy.from_dict(data)


def _backend(name: str) -> Any:
    if name == "jev":
        return jev_chunks.UsageJevBackend(
            pdf_bench.JEV_MODEL, concurrency=pdf_bench.JEV_CONCURRENCY
        )
    from docrooter.backends.registry import resolve_backend

    return resolve_backend(SYSTEMS[name])


def _warmup(backend: Any, config: Config, doc_types: Sequence[str | None]) -> float:
    """Untimed: load the model and encode the label vectors of every section taxonomy the run
    can use, under each doc type's instruction. ``None`` (auto) warms the doc-type step and
    every leaf doc type. Returns the seconds taken."""
    from docrooter.classify.normalize import normalize_sections

    doc = Document.from_sections(
        [{"title": "Introduction", "level": 1, "text": "This short note describes a method."}],
        source="warmup.txt",
    )
    types: list[str | None] = list(dict.fromkeys(doc_types))
    if None in types:
        types += [leaf.id for leaf in builtin_taxonomy("doc_types").leaves()]
    start = time.perf_counter()
    for doc_type in dict.fromkeys(types):
        taxonomy = None
        if doc_type is not None and not config.synonyms:
            taxonomy = without_synonyms(DOC_TYPE_TO_SECTION_TAXONOMY[doc_type])
        normalize_sections(
            doc, doc_type=doc_type, taxonomy=taxonomy, backend=backend, view=config.view
        )
    return time.perf_counter() - start


def _usage_tokens(backend: Any) -> tuple[int | None, bool]:
    """SDK-reported input tokens since the last ``usage.clear()``; estimated from the state
    texts (words x 1.3) when any request reported none."""
    usage = backend.usage
    if all(t is not None for _, t in usage):
        return sum(t for _, t in usage if t is not None), False
    return sum(t if t is not None else jev_chunks.estimate_tokens(s) for s, t in usage), True


def run_system(
    name: str,
    docs: Mapping[str, Document],
    doc_types: Mapping[str, str | None],
    gold: Mapping[str, Sequence[Mapping[str, Any]]],
    true_types: Mapping[str, str],
    config: Config = CONFIGS[FINAL_CONFIG],
) -> dict[str, Any]:
    """Normalise every document with backend ``name`` under ``config``; ``doc_types[id]`` is
    passed as ``doc_type`` (None = auto, which needs a config with synonyms). Rows carry gold
    and prediction for every matched section."""
    from docrooter.classify.normalize import CUSTOM_TAXONOMY, normalize_sections

    if not config.synonyms and any(t is None for t in doc_types.values()):
        raise ValueError("a config without synonyms needs a known doc type for every document")
    backend = _backend(name)
    load_s = _warmup(backend, config, list(doc_types.values()))
    is_jev = isinstance(backend, jev_chunks.UsageJevBackend)
    log(f"{name}: warm-up (load) {load_s:.1f} s")
    rows: list[dict[str, Any]] = []
    per_doc: dict[str, Any] = {}
    tokens_total, estimated = 0, False
    for doc_id, doc in docs.items():
        if is_jev:
            backend.usage.clear()
        doc_type = doc_types[doc_id]
        stripped = None if config.synonyms else DOC_TYPE_TO_SECTION_TAXONOMY[doc_type]
        start = time.perf_counter()
        result = normalize_sections(
            doc,
            doc_type=doc_type,
            taxonomy=None if stripped is None else without_synonyms(stripped),
            backend=backend,
            levels=LEVELS,
            view=config.view,
        )
        seconds = time.perf_counter() - start
        # the stripped copy is a custom taxonomy; it stands for the built-in it came from
        pred_taxonomy = stripped if result.taxonomy == CUSTOM_TAXONOMY else result.taxonomy
        match = match_gold(gold[doc_id], [doc.sections[sid] for sid in result.sections])
        gold_taxonomy = DOC_TYPE_TO_SECTION_TAXONOMY[true_types[doc_id]]
        doc_rows = []
        for g, sid in match.pairs:
            norm = result.sections[sid]
            doc_rows.append(
                {
                    "doc_id": doc_id,
                    "section_title": g["section_title"],
                    "page": g["page"],
                    "level": g["level"],
                    "taxonomy": gold_taxonomy,
                    "pred_taxonomy": pred_taxonomy,
                    "gold": g["canonical"],
                    "pred": norm.canonical,
                    "score": round(norm.score, 4),
                }
            )
        n_labelled = sum(n.canonical is not None for n in result.sections.values())
        entry: dict[str, Any] = {
            "seconds": seconds,
            "n_labelled": n_labelled,
            "doc_type": result.doc_type,
            "doc_type_confidence": result.doc_type_confidence,
            "doc_type_matched": result.doc_type == true_types[doc_id],
            "taxonomy": pred_taxonomy,
            "unmatched_gold": [
                {k: g[k] for k in ("section_title", "page", "level")} for g in match.unmatched_gold
            ],
            "unlabelled_sections": [
                {"title": doc.sections[s].title, "page": doc.sections[s].page_start}
                for s in match.unlabelled
            ],
            **score(doc_rows),
        }
        if is_jev:
            tokens, est = _usage_tokens(backend)
            entry |= {"tokens": tokens, "tokens_estimated": est}
            tokens_total += tokens
            estimated = estimated or est
        per_doc[doc_id] = entry
        rows.extend(doc_rows)
        acc = entry["accuracy"]
        log(
            f"{name} {doc_id}: {entry['n_correct']}/{entry['n_scored']} "
            f"({'n/a' if acc is None else f'{acc:.3f}'}), {seconds:.2f} s, "
            f"doc type {result.doc_type}"
        )
        if match.unmatched_gold:
            log(f"{name} {doc_id}: {len(match.unmatched_gold)} gold rows did not match")
    n_labelled = sum(d["n_labelled"] for d in per_doc.values())
    seconds = sum(d["seconds"] for d in per_doc.values())
    out: dict[str, Any] = {
        "backend": SYSTEMS[name],
        "config": next(k for k, c in CONFIGS.items() if c == config),
        "load_s": load_s,
        "seconds": seconds,
        "n_labelled": n_labelled,
        "s_per_section": seconds / n_labelled if n_labelled else None,
        "per_doc": per_doc,
        "overall": score(rows, qualify=True),
        "rows": rows,
    }
    if is_jev:
        out |= {
            "tokens": tokens_total,
            "tokens_estimated": estimated,
            "cost_usd": jev_cost(tokens_total),
        }
    del backend
    pdf_bench.free_memory()
    return out


def non_overlap(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Accuracy over the scored rows whose heading is not a heading-synonym example
    (:func:`in_examples`)."""
    s = score([r for r in rows if r["gold"] is not None and not in_examples(r)], qualify=True)
    return {k: s[k] for k in ("accuracy", "n_correct", "n_scored")}


def summarize(run: Mapping[str, Any], top: int = 3) -> dict[str, Any]:
    """The headline numbers of one run, the same on the rows outside the synonym overlap, and
    its ``top`` most frequent predicted labels over the scored rows (a collapse onto few labels
    shows up here)."""
    o = run["overall"]
    preds = Counter(r["pred"] for r in run["rows"] if r["gold"] is not None)
    out = {k: o[k] for k in ("accuracy", "macro_f1", "n_correct", "n_scored")} | {
        "non_overlap": non_overlap(run["rows"]),
        "s_per_section": run["s_per_section"],
        "top_preds": preds.most_common(top),
    }
    for k in ("tokens", "tokens_estimated", "cost_usd"):
        if k in run:
            out[k] = run[k]
    return out


# --- Markdown report ---------------------------------------------------------------------


def _f(value: float | None, digits: int = 3) -> str:
    return pdf_bench._f(value, digits)


def _cost(run: Mapping[str, Any]) -> str:
    if "cost_usd" not in run:
        return "-"
    return f"${run['cost_usd']:.4f}" + (" (est.)" if run.get("tokens_estimated") else "")


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _config_text(result: Mapping[str, Any], name: str) -> str:
    c = result["configs"][name]
    fields = " + ".join(c["fields"])
    syn = "with" if c["synonyms"] else "without"
    return f"view {fields} at {c['max_tokens']} tokens, {syn} the taxonomies' heading synonyms"


def _acc(o: Mapping[str, Any]) -> str:
    return f"{_f(o['accuracy'])} ({o['n_correct']}/{o['n_scored']})"


def _acc_both(overall: Mapping[str, Any], rest: Mapping[str, Any]) -> str:
    return f"{_acc(overall)}; {_acc(rest)}"


def _overlap_note(result: Mapping[str, Any]) -> list[str]:
    runs = [r for r in result["systems"].values() if "rows" in r]
    if not runs:
        return []
    scored = [r for r in runs[0]["rows"] if r["gold"] is not None]
    hits = [r for r in scored if in_examples(r)]
    titles = {(r["taxonomy"], heading_key(r["section_title"])) for r in hits}
    return [
        "",
        f"**Synonym overlap.** {len(hits)} of the {len(scored)} scored sections have a heading "
        f"equal to a heading-synonym example of their taxonomy ({len(titles)} distinct "
        "headings; compared case- and punctuation-insensitively after dropping one leading "
        'number such as "3.1", "Item 1A." or "Note 2."). For those, the synonyms can hand the '
        "model the answer, so every accuracy below is also given on the other "
        f"{len(scored) - len(hits)} (*non-overlap*), as *all; non-overlap*.",
    ]


def _before_after(result: Mapping[str, Any], names: Sequence[str]) -> list[str]:
    base = result.get("baseline_p10")
    if not base:
        return []
    lines = [
        "",
        "### Before / after",
        "",
        f"*Baseline (P10)* is the original run ({base['date']} at `{base['commit']}`: "
        f"{_config_text(result, base['config'])}); *after* is this run. Accuracy cells are "
        "*all; non-overlap*.",
        "",
        "| System | Baseline (P10) accuracy | After accuracy | Change (all) | Change (non-overlap) "
        "| Baseline macro F1 | After macro F1 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    runs = [(n, base["systems"].get(n), result["systems"][n]) for n in names]
    if "auto_doc_type" in base and "overall" in result.get("auto_doc_type", {}):
        runs.append(("auto_doc_type", base["auto_doc_type"], result["auto_doc_type"]))
    for n, before, after in runs:
        if not before or "overall" not in before:
            continue
        b, a = before["overall"], after["overall"]
        bn, an = non_overlap(before["rows"]), non_overlap(after["rows"])

        def pts(x: Mapping[str, Any], y: Mapping[str, Any]) -> str:
            if x["accuracy"] is None or y["accuracy"] is None:
                return "-"
            return f"{100 * (y['accuracy'] - x['accuracy']):+.1f} pts"

        lines.append(
            f"| `{n}` | {_acc_both(b, bn)} | {_acc_both(a, an)} | {pts(b, a)} | {pts(bn, an)} "
            f"| {_f(b['macro_f1'])} | {_f(a['macro_f1'])} |"
        )
    return lines


def _variants(result: Mapping[str, Any]) -> list[str]:
    variants = result.get("variants") or {}
    if not variants:
        return []
    names = [n for n in SYSTEMS if any(n in v for v in variants.values())]
    lines = [
        "",
        "### Configurations tried (the silver set used as a dev set)",
        "",
        "Every configuration that was run, all on the same scored sections, so the chosen "
        f"default (`{result['final_config']}`) is selected on the evaluation data and its margin "
        "over the others is optimistic. Accuracy cells are *all; non-overlap*. *Top prediction* "
        "is the most frequent predicted label over the scored sections and its count (a "
        "collapse onto one label shows up here).",
        "",
        "| Config | View | Synonyms | "
        + " | ".join(f"`{n}` accuracy" for n in names)
        + " | "
        + " | ".join(f"`{n}` top prediction" for n in names)
        + " |",
        "| --- | --- | --- | "
        + " | ".join("---" for _ in names)
        + " | "
        + " | ".join("---" for _ in names)
        + " |",
    ]
    for config in CONFIGS:
        if config not in variants:
            continue
        c, runs = result["configs"][config], variants[config]
        acc, top = [], []
        for n in names:
            v = runs.get(n)
            if v is None or "error" in v:
                acc.append("-" if v is None else "error")
                top.append("-")
                continue
            acc.append(_acc_both(v, v["non_overlap"]))
            top.append(
                f"`{v['top_preds'][0][0]}` ({v['top_preds'][0][1]})" if v["top_preds"] else "-"
            )
        mark = " (default)" if config == result["final_config"] else ""
        lines.append(
            f"| `{config}`{mark} | {' + '.join(c['fields'])} @ {c['max_tokens']} "
            f"| {'yes' if c['synonyms'] else 'no'} | " + " | ".join(acc + top) + " |"
        )
    return lines


def render_section(result: Mapping[str, Any], table_docs: Sequence[str] = TABLE_DOCS) -> str:
    """The generated block; ``table_docs`` get a per-entry table each."""
    meta, systems = result["gold"]["meta"], result["systems"]
    names = [n for n in SYSTEMS if n in systems and "error" not in systems[n]]
    docs = result["documents"]
    lines = [
        f"Silver set: `{result['gold']['path']}`, {result['gold']['n_entries']} "
        f"level-1/2 sections of {len(docs)} documents ({result['gold']['n_junk']} marked as "
        "parser junk, i.e. not a real section, and excluded from accuracy). **These are "
        "agent-assigned silver labels, not human gold:** an agent (Claude Opus 5.5) read each "
        "heading and the first ~300 characters of its body and chose the best node of the doc "
        "type's section taxonomy from the node definitions, before any normalisation run "
        f"(labelled {meta['labelled_at']}, documents loaded at `{meta['src_commit']}`). "
        "Ambiguous cases carry a `note` in the file. Expect a few points of label noise.",
        "",
        f"Run {result['date']} at `{result['commit']}`, device `{result['env']['device']}` "
        f"({result['env']['device_note']}). Each system calls `normalize_sections(doc, "
        "doc_type=<gold type>, backend=...)` once per document with the package defaults: Flat, "
        f"{_config_text(result, result['final_config'])}. "
        "*Macro F1* averages over the gold classes (per document over that document's classes; "
        "overall over all classes, qualified by taxonomy). *s/section* is the summed wall time "
        "of the per-document calls (view rendering and label encoding included) divided by the "
        "sections labelled, junk included. Model load and the label encoding of every section "
        "taxonomy the run can use (one untimed warm-up per doc type) are excluded (*Load s*). "
        "Jev cost is input "
        f"tokens x ${pdf_bench.JEV_USD_PER_M_INPUT}/M as reported by the API.",
        *_overlap_note(result),
        *_before_after(result, names),
        *_variants(result),
        "",
        "### Aggregate",
        "",
        "| System | Backend | Accuracy | Correct | Macro F1 | s/section | Load s | Jev tokens "
        "| Jev cost |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for n in names:
        run, o = systems[n], systems[n]["overall"]
        tokens = f"{run['tokens']:,}" if "tokens" in run else "-"
        lines.append(
            f"| `{n}` | `{run['backend']}` | {_f(o['accuracy'])} "
            f"| {o['n_correct']}/{o['n_scored']} | {_f(o['macro_f1'])} "
            f"| {_f(run['s_per_section'], 4)} | {run['load_s']:.1f} "
            f"| {tokens} | {_cost(run)} |"
        )
    lines += ["", "### Per document (accuracy / macro F1)", ""]
    lines.append(
        "| Document | Type | Sections | Junk | Unmatched | "
        + " | ".join(f"`{n}`" for n in names)
        + " |"
    )
    lines.append("| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in names) + " |")
    for d in docs:
        first = systems[names[0]]["per_doc"][d["id"]] if names else {}
        cells = []
        for n in names:
            p = systems[n]["per_doc"][d["id"]]
            cells.append(f"{_f(p['accuracy'])} / {_f(p['macro_f1'])}")
        lines.append(
            f"| {d['id']} {_cell(d['title'])} | `{d['doc_type']}` | {d['n_gold']} | {d['n_junk']} "
            f"| {len(first.get('unmatched_gold', []))} | " + " | ".join(cells) + " |"
        )
    auto = result.get("auto_doc_type")
    if auto and "error" not in auto:
        o = auto["overall"]
        lines += [
            "",
            "### Auto doc type (`doc_type=None`)",
            "",
            f"One run of `normalize_sections(doc)` with the default backend (`{auto['backend']}`), "
            "so `classify_document` picks the doc type and hence the section taxonomy. The doc "
            f"type matched the gold type on **{auto['matched']}/{auto['n']}** documents; section "
            f"accuracy {_f(o['accuracy'])} ({o['n_correct']}/{o['n_scored']}), macro F1 "
            f"{_f(o['macro_f1'])} (a wrong doc type usually means a different taxonomy, so its "
            "sections all count as wrong).",
            "",
            "| Document | Gold type | Auto type | Confidence | Taxonomy | Accuracy |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for d in docs:
            p = auto["per_doc"][d["id"]]
            mark = pdf_bench.CHECK if p["doc_type_matched"] else pdf_bench.CROSS
            lines.append(
                f"| {d['id']} | `{d['doc_type']}` | {mark} `{p['doc_type']}` "
                f"| {_f(p['doc_type_confidence'])} | `{p['taxonomy']}` | {_f(p['accuracy'])} |"
            )
    for doc_id in table_docs:
        d = next((x for x in docs if x["id"] == doc_id), None)
        if d is None or not names:
            continue
        lines += [
            "",
            f"### Per entry: {doc_id} {_cell(d['title'])}",
            "",
            "| Page | Level | Section | Gold | " + " | ".join(f"`{n}`" for n in names) + " |",
            "| --- | --- | --- | --- | " + " | ".join("---" for _ in names) + " |",
        ]
        by_system = {n: [r for r in systems[n]["rows"] if r["doc_id"] == doc_id] for n in names}
        for i, r in enumerate(by_system[names[0]]):
            gold = "*(junk)*" if r["gold"] is None else f"`{r['gold']}`"
            cells = []
            for n in names:
                p = by_system[n][i]["pred"]
                mark = (
                    ""
                    if r["gold"] is None
                    else (f"{pdf_bench.CHECK} " if p == r["gold"] else f"{pdf_bench.CROSS} ")
                )
                cells.append(f"{mark}`{p}`")
            lines.append(
                f"| {r['page']} | {r['level']} | {_cell(r['section_title'][:70])} | {gold} | "
                + " | ".join(cells)
                + " |"
            )
    return "\n".join(lines)


def update_docs(path: Path, block: str) -> None:
    """Replace the generated block of the ``## Section normalisation`` section of ``path``,
    appending the section at the end when it is missing; everything else is kept."""
    wrapped = f"{BEGIN}\n{block}\n{END}"
    text = path.read_text(encoding="utf-8")
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
    parser = argparse.ArgumentParser(description="Section-normalisation silver benchmark")
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument(
        "--configs",
        default=FINAL_CONFIG,
        help=f"comma-separated configs from {list(CONFIGS)}; only {FINAL_CONFIG} gets full "
        "per-document results and the auto doc-type run, the others a summary",
    )
    parser.add_argument("--no-auto", action="store_true", help="skip the doc_type=None run")
    parser.add_argument(
        "--gold", type=Path, default=None, help="default: <corpus>/section_gold.jsonl"
    )
    parser.add_argument(
        "--corpus",
        type=corpora.corpus_path,
        default=pdf_bench.CORPUS,
        help="dev, heldout or a corpus directory (results of a non-dev corpus go to "
        "<name>-normalize.json)",
    )
    parser.add_argument("--out", type=Path, default=None, help="results dir (default: dated)")
    parser.add_argument("--device", default=None, help="skip the MPS probe and use this device")
    parser.add_argument("--docs", type=Path, default=None, help="also update this Markdown file")
    parser.add_argument("--report-only", action="store_true", help="render from normalize.json")
    _guard.add_guard_argument(parser)
    args = parser.parse_args(argv)

    out_dir = args.out or pdf_bench.RESULTS_ROOT / datetime.date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / corpora.results_name(args.corpus, "normalize.json")
    args.gold = args.gold or args.corpus / GOLD_PATH.name
    if args.report_only:
        update_docs(args.docs or pdf_bench.DOCS_PATH, render_section(json.loads(path.read_text())))
        return
    _guard.single_model_process(
        "benchmarks.normalize_bench", allow_concurrent=args.allow_concurrent
    )
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = [s for s in systems if s not in SYSTEMS]
    if unknown:
        parser.error(f"unknown systems {unknown}; choose from {list(SYSTEMS)}")
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        parser.error(f"unknown configs {unknown}; choose from {list(CONFIGS)}")

    pdf_bench.load_env()  # silent: never prints names or values
    meta, gold_rows = read_gold(args.gold)
    doc_ids = list(meta.get("documents") or dict.fromkeys(r["doc_id"] for r in gold_rows))
    entries = {e["id"]: e for e in pdf_bench.manifest(args.corpus)}
    true_types = {i: entries[i]["label"] for i in doc_ids}
    gold: dict[str, list[dict[str, Any]]] = {i: [] for i in doc_ids}
    for r in gold_rows:
        gold[r["doc_id"]].append(r)

    if args.device:
        device, note = args.device, "set with --device"
    elif any(s != "jev" for s in systems) or not args.no_auto:
        device, note = pdf_bench.probe_device()
    else:
        device, note = "n/a", "API only"
    os.environ["DOCROOTER_DEVICE"] = device
    log(f"device {device}: {note}")
    log(f"loading {len(doc_ids)} PDFs")
    docs = load_documents(doc_ids, args.corpus)

    result: dict[str, Any] = (
        json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"systems": {}}
    )
    result |= {
        "final_config": FINAL_CONFIG,
        "configs": {
            k: {"fields": list(c.view.fields), "max_tokens": c.view.max_tokens}
            | {"synonyms": c.synonyms}
            for k, c in CONFIGS.items()
        },
        "date": datetime.date.today().isoformat(),
        "commit": git_revision(),
        "corpus": corpora.relative(args.corpus),
        "env": pdf_bench.collect_env() | {"device": device, "device_note": note},
        "gold": {
            "path": str(args.gold.resolve().relative_to(pdf_bench.ROOT.resolve())),
            "meta": meta,
            "n_entries": len(gold_rows),
            "n_junk": sum(r["canonical"] is None for r in gold_rows),
        },
        "documents": [
            {
                "id": i,
                "title": entries[i]["title"],
                "doc_type": true_types[i],
                "n_sections": sum(s.level in LEVELS for s in docs[i].walk()),
                "n_gold": len(gold[i]),
                "n_junk": sum(r["canonical"] is None for r in gold[i]),
            }
            for i in doc_ids
        ],
    }
    runs: list[tuple[str, str, str, dict[str, str | None]]] = [
        (c, s, s, true_types) for c in configs for s in systems
    ]
    if FINAL_CONFIG in configs and not args.no_auto:
        runs.append((FINAL_CONFIG, "auto_doc_type", AUTO_SYSTEM, dict.fromkeys(doc_ids)))
    variants = result.setdefault("variants", {})
    for config, key, name, doc_types in runs:
        log(f"== {config} {key} (load average {os.getloadavg()[0]:.1f})")
        try:
            run = run_system(name, docs, doc_types, gold, true_types, CONFIGS[config])
        except Exception as exc:  # noqa: BLE001 - one failing system must not lose the others
            log(f"{key} FAILED: {type(exc).__name__}: {str(exc)[:500]}")
            run = {"error": f"{type(exc).__name__}: {str(exc)[:500]}"}
        else:
            run["loadavg_end"] = list(os.getloadavg())
            if key == "auto_doc_type":
                run["matched"] = sum(d["doc_type_matched"] for d in run["per_doc"].values())
                run["n"] = len(run["per_doc"])
            o = run["overall"]
            log(f"{config} {key}: accuracy {_f(o['accuracy'])}, macro F1 {_f(o['macro_f1'])}")
        if key == "auto_doc_type":
            result["auto_doc_type"] = run
        else:
            variants.setdefault(config, {})[key] = run if "error" in run else summarize(run)
            if config == FINAL_CONFIG:
                result["systems"][key] = run
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.docs:
        update_docs(args.docs, render_section(result))
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
