"""Selection benchmark (spec §8, §0.2, §0.5; task S7): does ``select()`` keep the promised share
of relevant content, at what token cost, and how do simple baselines do with the same tokens?

**Documents.** Every gold-split document of the corpus store that has ``teacher-full`` labels or
gold rows, read again with :class:`~docrooter.select.fit.CorpusPdfReader` (the profile and page
cap the selector was fitted with, from its ``meta``, unless given; the pages the build read;
sections normalised in the document's family), tagged by the trained tagger and embedded once
(:func:`~docrooter.select.selection.prepare`). Teacher relevance comes from
:func:`~docrooter.select.fit.labels_by_document` and :func:`~docrooter.select.fit.label_leaves`,
the reading and labels the selector was fitted on, whose per-document check counts are kept. A
document whose labels do not fit its reading (an incomplete labelling run, a diverged reading)
gives no main units, but its gold subset is still evaluated. Every document is accounted for:
read, with units, read without units, or skipped with its reason (:func:`run_gold`).

**Units.** A unit is a (document, built-in field of the document's family) pair with at least
one teacher-relevant content element (an element whose teacher-full topic is one of the field's
topics).

**Main number** (teacher relevance, spec §0.2). For each c in :data:`C_GRID`, the selection is
:func:`~docrooter.select.selection.select_scored` on the document's features (the tail of
``select()``: the same cut, dial and expansion, without tagging and embedding again for every
c). Recall is ``|selected ∩ relevant| / |relevant|`` over content elements AFTER expansion (what
the user gets); per c, per family and overall: mean, 10th percentile and min recall, mean token
share, and the shares of units with ``guarantee`` and with ``capped``. The program target
(spec §0.5) is mean recall >= c at c = 0.9, judged twice (:func:`program_target`): the headline
counts a capped unit (the whole document) at recall 1, as the user gets it, and the
uncapped-only verdict judges the dial without them. Both, the capped share, the skipped
documents and the gold hash mismatches are printed together (:func:`target_line`).

**The curve.** For each unit, the selector's probabilities (the field combiner select() uses)
are cut at every threshold of :data:`THRESHOLDS` and expanded; the mean recall and mean token
share over units at each threshold make the curve. Tokens-to-95 is the smallest mean token share
at which the mean recall reaches 0.95, on the curve and on the dial's own c grid.

**Secondary numbers** (spec §0.2). *Gold subset*: the gold elements of the document whose gold
topic is one of the field's topics are relevant; recall is over gold-annotated elements only,
tokens over the whole document, at the same c grid. A gold element whose text hash differs in
the reading (or that is not a content element of it) is counted and excluded. *Teacher–gold
disagreement*: on the gold elements, the teacher-full topic against the gold topic (accuracy,
per family, the 10 most frequent confusions); no model is needed for this.

**Baselines** (spec §8.3), each with the SAME tokens as the selector: for each unit and c < 1,
the budget is the selector's tokens at that c. Each picks content elements only:

- ``lead_k``: content elements in reading order, as long as the next one fits (a prefix);
- ``bm25``: BM25 (k1 = 1.5, b = 0.75; lower-cased ``\\w+`` tokens without
  :data:`docrooter.keyphrases.STOPWORDS`) of each element's text against the field's query (else
  its description); the top elements, each one that still fits, no expansion;
- ``similarity``: the same fill ranked by the ``sim`` feature;
- ``structure``: the elements whose section matches the field (``section_hit``), in reading
  order, each one that still fits, then nothing.

The fills are greedy: an element that does not fit is skipped and smaller ones are still tried
(slightly generous to the baselines). The whole document (recall 1, share 1) is the reference
row.

**Results** go to ``benchmarks/results/<date>/selection.json`` (ids and numbers only, never
element text) and ``selection.md``. One model set per process, under :mod:`benchmarks._guard`::

    PYTHONPATH=src .venv/bin/python -m benchmarks.selection_bench gold \\
        --tagger <bundle> --selector <selector.json> \\
        --store benchmarks/.cache.nosync/corpus_store \\
        --pdf-dir benchmarks/.cache.nosync/training_corpus/pdfs
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks import _guard
from docrooter.keyphrases import STOPWORDS
from docrooter.select.relevance import FEATURE_NAMES
from docrooter.select.selection import Selection

BENCHMARKS = Path(__file__).resolve().parent
CACHE = BENCHMARKS / ".cache.nosync"
STORE = CACHE / "corpus_store"
PDF_DIR = CACHE / "training_corpus" / "pdfs"
GOLD_PATH = BENCHMARKS / "training_corpus" / "gold.jsonl"
RESULTS_ROOT = BENCHMARKS / "results"
RESULTS_FILE = "selection.json"
MARKDOWN_FILE = "selection.md"

C_GRID: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0)
TARGET_C = 0.9  # spec §0.5: mean teacher-relevance recall >= c at c = 0.9
TOKENS_TO_RECALL = 0.95
THRESHOLDS: tuple[float, ...] = tuple(
    sorted(
        {round(float(t), 6) for t in np.linspace(0.0, 1.0, 101)}
        | {round(float(t), 6) for t in np.geomspace(1e-4, 1e-2, 9)},
        reverse=True,
    )
)
"""The curve's thresholds on P(relevant): every 0.01, and finer below 0.01 (high recall)."""
BM25_K1 = 1.5
BM25_B = 0.75
BASELINES: tuple[str, ...] = ("lead_k", "bm25", "similarity", "structure")
WHOLE_DOCUMENT = {"recall": 1.0, "token_share": 1.0}
SIM = FEATURE_NAMES.index("sim")
SECTION_HIT = FEATURE_NAMES.index("section_hit")
DISCLOSURE = (
    "teacher relevance (the CRC target) is the main number; gold is Claude-annotated; "
    "pairs from one document are correlated"
)
DEFAULT_PROFILE = "accurate"  # the corpus build's
DEFAULT_MAX_PAGES = 30
_TOL = 1e-9
_WORD = re.compile(r"\w+")


def log(message: str) -> None:
    print(f"[{datetime.datetime.now():%H:%M:%S}] {message}", flush=True)


def c_key(c: float) -> str:
    """The results key of a completeness: ``"0.90"``."""
    return f"{c:.2f}"


# --- recall and token share -----------------------------------------------------------------------


def recall(selected: Collection[str], relevant: Collection[str]) -> float | None:
    """``|selected ∩ relevant| / |relevant|``; ``None`` (undefined) when nothing is relevant."""
    if not relevant:
        return None
    chosen = set(selected)
    return sum(1 for item in relevant if item in chosen) / len(relevant)


def evaluate_selection(sel: Selection, relevant: Collection[str]) -> dict[str, Any]:
    """One unit at one c: recall over ``relevant`` after expansion, and the selection's tokens."""
    return {
        "recall": recall(sel.element_ids, relevant),
        "token_share": sel.token_share,
        "tokens": sel.tokens,
        "guarantee": sel.guarantee,
        "capped": sel.capped,
    }


def summary_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """n, recall mean, 10th percentile and min, mean token share, the shares of units with
    ``guarantee`` and ``capped``, and the mean recall over the uncapped units alone (a capped
    unit returns the whole document, so its recall of 1 says nothing about the dial)."""
    recalls = [float(r["recall"]) for r in rows if r["recall"] is not None]
    uncapped = [float(r["recall"]) for r in rows if r["recall"] is not None and not r["capped"]]
    if not recalls:
        return {
            "n": 0,
            "recall_mean": None,
            "recall_p10": None,
            "recall_min": None,
            "token_share_mean": None,
            "guarantee_share": None,
            "capped_share": None,
            "n_uncapped": 0,
            "recall_mean_uncapped": None,
        }
    n = len(rows)
    return {
        "n": n,
        "recall_mean": float(np.mean(recalls)),
        "recall_p10": float(np.percentile(recalls, 10)),
        "recall_min": float(min(recalls)),
        "token_share_mean": float(np.mean([float(r["token_share"]) for r in rows])),
        "guarantee_share": sum(bool(r["guarantee"]) for r in rows) / n,
        "capped_share": sum(bool(r["capped"]) for r in rows) / n,
        "n_uncapped": len(uncapped),
        "recall_mean_uncapped": float(np.mean(uncapped)) if uncapped else None,
    }


def aggregate(units: Sequence[Mapping[str, Any]], c_grid: Sequence[float]) -> dict[str, Any]:
    """Per c (:func:`c_key`), overall and per family: :func:`summary_stats`."""
    families = sorted({str(u["family"]) for u in units})
    out: dict[str, Any] = {"overall": {}, "families": {f: {} for f in families}}
    for c in c_grid:
        key = c_key(c)
        out["overall"][key] = summary_stats([u["by_c"][key] for u in units])
        for family in families:
            rows = [u["by_c"][key] for u in units if u["family"] == family]
            out["families"][family][key] = summary_stats(rows)
    return out


def program_target(agg: Mapping[str, Any], c: float = TARGET_C) -> dict[str, Any]:
    """Spec §0.5: mean recall >= c at c = 0.9, overall and per family, judged twice.

    The headline (``pass``) keeps capped units at recall 1, which is what the user gets; the
    uncapped-only verdict (``pass_uncapped``, ``None`` when every unit is capped) judges the dial
    itself. The capped and guaranteed shares, the 10th percentile and the token share are kept
    beside them."""
    key = c_key(c)

    def verdict(stats: Mapping[str, Any]) -> dict[str, Any]:
        mean = stats.get("recall_mean")
        uncapped = stats.get("recall_mean_uncapped")
        return {
            "n": stats.get("n", 0),
            "mean": mean,
            "p10": stats.get("recall_p10"),
            "token_share_mean": stats.get("token_share_mean"),
            "guarantee_share": stats.get("guarantee_share"),
            "capped_share": stats.get("capped_share"),
            "pass": mean is not None and mean >= c - _TOL,
            "n_uncapped": stats.get("n_uncapped", 0),
            "mean_uncapped": uncapped,
            "pass_uncapped": None if uncapped is None else uncapped >= c - _TOL,
        }

    overall = verdict(agg["overall"].get(key, {}))
    return {
        "c": c,
        **overall,
        "families": {f: verdict(by_c.get(key, {})) for f, by_c in agg["families"].items()},
    }


SKIP_REASONS = (
    "no_teacher_labels",
    "no_fields",
    "nothing_to_evaluate",
    "read_failed",
    "label_check_failed",
)


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value:.0%}"


def _verdict_text(value: bool | None) -> str:
    return "n/a" if value is None else ("PASS" if value else "FAIL")


def target_line(
    target: Mapping[str, Any],
    documents: Mapping[str, Any],
    gold_elements: Mapping[str, int],
) -> str:
    """The program-target readout: both verdicts, the capped share, and every exclusion."""
    uncapped = (
        f"uncapped only {_verdict_text(target['pass_uncapped'])} "
        f"(mean {_fmt(target['mean_uncapped'])}, n = {target['n_uncapped']})"
        if target["pass_uncapped"] is not None
        else "uncapped only n/a (every unit capped)"
    )
    skipped = documents.get("skipped", {})
    reasons = [f"{r} {len(skipped[r])}" for r in SKIP_REASONS if skipped.get(r)]
    return (
        f"program target at c = {target['c']:.2f}: headline {_verdict_text(target['pass'])} "
        f"(mean recall {_fmt(target['mean'])}, p10 {_fmt(target['p10'])}, token share "
        f"{_fmt(target['token_share_mean'])}; {_pct(target['capped_share'])} of units capped, "
        f"{_pct(target['guarantee_share'])} guaranteed; n = {target['n']}), {uncapped}; "
        f"documents: {documents.get('gold_split', 0)} gold, {documents.get('read', 0)} read, "
        f"{documents.get('evaluated', 0)} with units, "
        f"{len(documents.get('read_without_units', []))} read without units; "
        f"skipped: {', '.join(reasons) or 'none'}; gold elements: "
        f"{gold_elements.get('matched', 0)} matched, {gold_elements.get('hash_mismatch', 0)} "
        f"hash mismatch, {gold_elements.get('missing', 0)} missing"
    )


def reading_settings(
    profile: str | None, max_pages: int | None, meta: Mapping[str, Any]
) -> tuple[str, int, list[str]]:
    """The profile and page cap to read with: the ones given, else the selector's fit (its
    ``meta``), else the corpus build's defaults; plus a warning for each one given that differs
    from the fit."""
    fit_profile, fit_pages = meta.get("profile"), meta.get("max_pages")
    warnings: list[str] = []
    if profile is not None and fit_profile is not None and profile != fit_profile:
        warnings.append(
            f"--profile {profile} differs from the profile the selector was fitted with "
            f"({fit_profile}): readings may diverge from the labels"
        )
    if max_pages is not None and fit_pages is not None and max_pages != fit_pages:
        warnings.append(
            f"--max-pages {max_pages} differs from the {fit_pages} pages the selector was "
            "fitted with: readings may diverge from the labels"
        )
    chosen_profile = profile or fit_profile or DEFAULT_PROFILE
    chosen_pages = max_pages if max_pages is not None else fit_pages or DEFAULT_MAX_PAGES
    return str(chosen_profile), int(chosen_pages), warnings


# --- the curve ------------------------------------------------------------------------------------


def threshold_curve(
    p: np.ndarray,
    expansions: Sequence[Sequence[str]],
    tokens: Mapping[str, int],
    total: int,
    relevant: Collection[str],
    thresholds: Sequence[float],
) -> list[tuple[float, float]]:
    """``(token share, recall)`` of the cut ``p >= threshold`` with expansion, per threshold (in
    the given order). ``expansions[j]`` is the element ids content row ``j`` brings (itself
    included); ``tokens`` covers every one of them. One pass from the highest threshold down,
    each row expanded once."""
    if not relevant:
        raise ValueError("a curve needs at least one relevant element")
    wanted = set(relevant)
    order = np.argsort(-np.asarray(p, dtype=np.float64), kind="stable")
    picked: set[str] = set()
    used = hits = k = 0
    points: dict[float, tuple[float, float]] = {}
    for threshold in sorted(set(thresholds), reverse=True):
        while k < len(order) and p[order[k]] >= threshold:
            for element_id in expansions[int(order[k])]:
                if element_id not in picked:
                    picked.add(element_id)
                    used += tokens[element_id]
                    hits += element_id in wanted
            k += 1
        points[threshold] = (used / total if total else 0.0, hits / len(wanted))
    return [points[t] for t in thresholds]


def mean_curve(
    curves: Sequence[Sequence[tuple[float, float]]], thresholds: Sequence[float]
) -> list[dict[str, float]]:
    """The mean token share and mean recall over units at each threshold."""
    if not curves:
        return []
    array = np.asarray(curves, dtype=np.float64)  # units x thresholds x (share, recall)
    shares, recalls = array[:, :, 0].mean(axis=0), array[:, :, 1].mean(axis=0)
    return [
        {"threshold": float(t), "token_share": float(s), "recall": float(r)}
        for t, s, r in zip(thresholds, shares, recalls, strict=True)
    ]


def tokens_to_recall(
    points: Iterable[Mapping[str, Any]], target: float = TOKENS_TO_RECALL
) -> float | None:
    """The smallest token share among points whose recall reaches ``target`` (``None`` if
    none does)."""
    shares = [
        float(point["token_share"])
        for point in points
        if point["recall"] is not None and float(point["recall"]) >= target - _TOL
    ]
    return min(shares) if shares else None


# --- baselines ------------------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """BM25 tokens: lower-cased ``\\w+`` words that are not stopwords."""
    return [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


def bm25_scores(
    texts: Sequence[str], query: str, *, k1: float = BM25_K1, b: float = BM25_B
) -> np.ndarray:
    """Okapi BM25 of each text against ``query``, the texts being the collection. The idf is
    ``log(1 + (N - df + 0.5) / (df + 0.5))``, never negative."""
    docs = [tokenize(t) for t in texts]
    n = len(docs)
    scores = np.zeros(n)
    terms = set(tokenize(query))
    if not n or not terms:
        return scores
    lengths = np.array([len(d) for d in docs], dtype=np.float64)
    mean_length = float(lengths.mean()) or 1.0
    counts = [Counter(d) for d in docs]
    for term in terms:
        df = sum(1 for c in counts if term in c)
        if not df:
            continue
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        tf = np.array([c.get(term, 0) for c in counts], dtype=np.float64)
        scores += idf * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * lengths / mean_length))
    return scores


def _fill(order: Iterable[int], tokens: Sequence[int], budget: int) -> list[int]:
    """Each row of ``order`` that still fits the budget, in reading order."""
    picked: list[int] = []
    used = 0
    for j in order:
        if used + tokens[j] <= budget:
            picked.append(int(j))
            used += tokens[j]
    return sorted(picked)


def lead_k(tokens: Sequence[int], budget: int) -> list[int]:
    """Rows in reading order while the next one fits (a prefix of the document)."""
    picked: list[int] = []
    used = 0
    for j, cost in enumerate(tokens):
        if used + cost > budget:
            break
        picked.append(j)
        used += cost
    return picked


def ranked(scores: np.ndarray, tokens: Sequence[int], budget: int) -> list[int]:
    """The highest-scoring rows (ties in reading order), each one that still fits."""
    return _fill(np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable"), tokens, budget)


def structure_only(section_hit: np.ndarray, tokens: Sequence[int], budget: int) -> list[int]:
    """The rows whose section matches the need, in reading order, each one that still fits;
    nothing else."""
    return _fill(np.flatnonzero(np.asarray(section_hit) > 0.5), tokens, budget)


def baseline_rows(
    texts: Sequence[str],
    tokens: Sequence[int],
    sim: np.ndarray,
    section_hit: np.ndarray,
    query: str,
    budget: int,
) -> dict[str, list[int]]:
    """The content rows each baseline picks within ``budget`` tokens (:data:`BASELINES`)."""
    return {
        "lead_k": lead_k(tokens, budget),
        "bm25": ranked(bm25_scores(texts, query), tokens, budget),
        "similarity": ranked(sim, tokens, budget),
        "structure": structure_only(section_hit, tokens, budget),
    }


def rows_score(
    picked: Sequence[int], relevant: Collection[int], tokens: Sequence[int], total: int
) -> tuple[float | None, float]:
    """``(recall, token share)`` of a set of content rows."""
    share = sum(tokens[j] for j in picked) / total if total else 0.0
    return recall({str(j) for j in picked}, {str(j) for j in relevant}), share


def aggregate_baselines(
    units: Sequence[Mapping[str, Any]], c_grid: Sequence[float]
) -> dict[str, Any]:
    """Per c < 1, overall and per family: the selector's and each baseline's mean recall and
    mean token share over the same units."""
    families = sorted({str(u["family"]) for u in units})
    out: dict[str, Any] = {"overall": {}, "families": {f: {} for f in families}}

    def means(group: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
        row: dict[str, Any] = {"n": len(group)}
        for name in ("selector", *BASELINES):
            scores = [
                u["by_c"][key] if name == "selector" else u["baselines"][key][name] for u in group
            ]
            recalls = [float(s["recall"]) for s in scores if s["recall"] is not None]
            row[name] = {
                "recall_mean": float(np.mean(recalls)) if recalls else None,
                "token_share_mean": (
                    float(np.mean([float(s["token_share"]) for s in scores])) if scores else None
                ),
            }
        return row

    for c in c_grid:
        if c >= 1.0:
            continue
        key = c_key(c)
        out["overall"][key] = means(units, key)
        for family in families:
            out["families"][family][key] = means([u for u in units if u["family"] == family], key)
    out["whole_document"] = dict(WHOLE_DOCUMENT)
    return out


# --- gold -----------------------------------------------------------------------------------------


def read_gold(path: str | os.PathLike[str] = GOLD_PATH) -> list[dict[str, Any]]:
    """The gold rows (keys and labels only, no text)."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def gold_matches(
    gold_rows: Iterable[Mapping[str, Any]], reread: Mapping[str, str]
) -> tuple[dict[str, str], dict[str, int]]:
    """The gold elements of one document found in its reading (``element id -> text sha1`` of
    its content elements) with the same text: ``({element id: gold topic}, counts)``. One whose
    hash differs is a ``hash_mismatch``, one not in the reading ``missing``; both are excluded."""
    matched: dict[str, str] = {}
    counts = {"gold": 0, "matched": 0, "hash_mismatch": 0, "missing": 0}
    for row in gold_rows:
        counts["gold"] += 1
        element_id = str(row["element_id"])
        sha = reread.get(element_id)
        if sha is None:
            counts["missing"] += 1
        elif sha != row["text_sha1"]:
            counts["hash_mismatch"] += 1
        else:
            counts["matched"] += 1
            matched[element_id] = str(row["topic"])
    return matched, counts


def teacher_gold_disagreement(
    gold_rows: Iterable[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Mapping[str, str]]],
    top: int = 10,
) -> dict[str, Any]:
    """The teacher-full topic against the gold topic on the gold elements: accuracy overall and
    per family, and the ``top`` most frequent (gold, teacher) confusions. ``labels`` is
    :func:`~docrooter.select.fit.labels_by_document`'s ``doc -> element -> text sha1 -> leaf``;
    an element with no teacher label is ``unlabelled``, one labelled for another text a
    ``hash_mismatch``; both are left out."""
    n = correct = unlabelled = mismatch = 0
    by_family: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    confusions: Counter[tuple[str, str]] = Counter()
    for row in gold_rows:
        by_hash = labels.get(str(row["doc_id"]), {}).get(str(row["element_id"]))
        if by_hash is None:
            unlabelled += 1
            continue
        leaf = by_hash.get(str(row["text_sha1"]))
        if leaf is None:
            mismatch += 1
            continue
        gold = str(row["topic"])
        n += 1
        hit = leaf == gold
        correct += hit
        family = by_family[str(row["family"])]
        family[0] += 1
        family[1] += hit
        if not hit:
            confusions[(gold, leaf)] += 1
    ranked_pairs = sorted(confusions.items(), key=lambda item: (-item[1], item[0]))[:top]
    return {
        "n": n,
        "accuracy": correct / n if n else None,
        "unlabelled": unlabelled,
        "hash_mismatch": mismatch,
        "families": {
            f: {"n": total, "accuracy": hits / total if total else None}
            for f, (total, hits) in sorted(by_family.items())
        },
        "confusions": [{"gold": g, "teacher": t, "n": k} for (g, t), k in ranked_pairs],
    }


# --- one document ---------------------------------------------------------------------------------


def evaluate_document(
    analysis: Any,
    ctx: Any,
    leaves: Sequence[str | None] | None,
    fields: Sequence[Any],
    gold_rows: Sequence[Mapping[str, Any]],
    selector: Any,
    *,
    doc_id: str,
    family: str,
    c_grid: Sequence[float] = C_GRID,
    thresholds: Sequence[float] = THRESHOLDS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int], list[list[Any]]]:
    """The units of one prepared document: ``(units, gold units, gold counts, curves)``; a
    unit's curve is :func:`threshold_curve` on the probabilities select() cuts. ``leaves`` is
    each content element's teacher topic; ``None`` (no usable teacher labels) gives gold-subset
    units only."""
    from docrooter.corpus.store import text_sha1
    from docrooter.select.selection import need_features, need_texts, select_scored

    index = ctx.index
    elements = index.doc.elements
    content_ids: list[str] = list(ctx.element_ids)
    texts = [elements[pos].text for pos in index.content]
    tokens = [int(index.tokens[pos]) for pos in index.content]
    token_by_id = {elements[pos].id: int(index.tokens[pos]) for pos in index.selectable}
    expansions = [[elements[q].id for q in index.expansion(pos)] for pos in index.content]
    total = int(index.total_tokens)
    reread = {elements[pos].id: text_sha1(elements[pos].text) for pos in index.content}
    matched, gold_counts = gold_matches(gold_rows, reread)
    full = ctx.tagged and ctx.has_sims
    units: list[dict[str, Any]] = []
    gold_units: list[dict[str, Any]] = []
    curves: list[list[Any]] = []
    for field in fields:
        topics = set(field.topics)
        relevant = {content_ids[j] for j, leaf in enumerate(leaves or ()) if leaf in topics}
        gold_relevant = {element_id for element_id, t in matched.items() if t in topics}
        if not relevant and not gold_relevant:
            continue
        _, X = need_features(ctx, field)
        selections = {
            c: select_scored(analysis, ctx, X, field, completeness=c, selector=selector)
            for c in c_grid
        }
        base = {"doc_id": doc_id, "family": family, "field_id": field.id}
        if relevant:
            relevant_rows = {
                j for j, element_id in enumerate(content_ids) if element_id in relevant
            }
            query = need_texts(field)[0]
            baselines: dict[str, Any] = {}
            for c in c_grid:
                if c >= 1.0:
                    continue
                picks = baseline_rows(
                    texts, tokens, X[:, SIM], X[:, SECTION_HIT], query, selections[c].tokens
                )
                baselines[c_key(c)] = {
                    name: dict(
                        zip(
                            ("recall", "token_share"),
                            rows_score(rows, relevant_rows, tokens, total),
                            strict=True,
                        )
                    )
                    for name, rows in picks.items()
                }
            combiner = selector.field if full else selector.field_fallback
            p = combiner.predict_proba(X)
            curve = threshold_curve(p, expansions, token_by_id, total, relevant, thresholds)
            curves.append([family, curve])
            units.append(
                {
                    **base,
                    "n_relevant": len(relevant),
                    "n_content": len(content_ids),
                    "full_features": bool(full),
                    "by_c": {
                        c_key(c): evaluate_selection(sel, relevant) for c, sel in selections.items()
                    },
                    "baselines": baselines,
                }
            )
        if gold_relevant:
            gold_units.append(
                {
                    **base,
                    "n_relevant": len(gold_relevant),
                    "by_c": {
                        c_key(c): evaluate_selection(sel, gold_relevant)
                        for c, sel in selections.items()
                    },
                }
            )
    return units, gold_units, gold_counts, curves


def summarise_curves(
    curves: Sequence[Sequence[Any]], thresholds: Sequence[float] = THRESHOLDS
) -> dict[str, Any]:
    """The mean curve and tokens-to-95, overall and per family, from ``[family, curve]`` pairs."""
    families = sorted({str(family) for family, _ in curves})
    out: dict[str, Any] = {}
    for name in ("overall", *families):
        group = [curve for family, curve in curves if name in ("overall", family)]
        points = mean_curve(group, thresholds)
        out[name] = {"n": len(group), "tokens_to_95": tokens_to_recall(points), "points": points}
    return out


def dial_tokens_to_95(agg: Mapping[str, Any]) -> dict[str, float | None]:
    """Tokens-to-95 on the dial's own c grid: the smallest mean token share at a c whose mean
    recall reaches 0.95, overall and per family."""

    def points(by_c: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"recall": s["recall_mean"], "token_share": s["token_share_mean"]}
            for s in by_c.values()
            if s["recall_mean"] is not None
        ]

    out = {"overall": tokens_to_recall(points(agg["overall"]))}
    out.update({f: tokens_to_recall(points(by_c)) for f, by_c in agg["families"].items()})
    return out


# --- results --------------------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _main_table(by_group: Mapping[str, Mapping[str, Any]], title: str) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| group | c | n | recall mean | p10 | min | token share | guarantee | capped |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for group, by_c in by_group.items():
        for key, s in by_c.items():
            lines.append(
                f"| {group} | {key} | {s['n']} | {_fmt(s['recall_mean'])} | "
                f"{_fmt(s['recall_p10'])} | {_fmt(s['recall_min'])} | "
                f"{_fmt(s['token_share_mean'])} | {_fmt(s['guarantee_share'], 2)} | "
                f"{_fmt(s['capped_share'], 2)} |"
            )
    return [*lines, ""]


def _coverage(results: Mapping[str, Any]) -> list[str]:
    documents = results.get("documents", {})
    skipped = documents.get("skipped", {})
    gold = results.get("gold_elements", {})
    totals = results.get("label_checks", {}).get("totals", {})
    return [
        "### Coverage",
        "",
        f"{documents.get('gold_split', 0)} gold-split documents, {documents.get('read', 0)} "
        f"read, {documents.get('evaluated', 0)} with units, "
        f"{len(documents.get('read_without_units', []))} read without units.",
        "",
        "| skipped because | documents |",
        "|---|---|",
        *(f"| {reason} | {len(skipped.get(reason, []))} |" for reason in SKIP_REASONS),
        "",
        "| gold elements | n |",
        "|---|---|",
        f"| matched | {gold.get('matched', 0)} |",
        f"| hash mismatch | {gold.get('hash_mismatch', 0)} |",
        f"| missing | {gold.get('missing', 0)} |",
        "",
        "Teacher labels of the evaluated documents: "
        + (", ".join(f"{k} {v}" for k, v in totals.items()) or "none")
        + " (unlabelled elements count as not relevant).",
        "",
    ]


def markdown(results: Mapping[str, Any]) -> str:
    """The results as Markdown tables (for ``docs/benchmarks-selection.md``)."""
    main, gold = results["main"], results["gold_subset"]
    target = results["program_target"]
    lines = [
        "# Selection benchmark results",
        "",
        f"Date {results['date']}, commit {results['provenance'].get('commit')}. "
        f"{results['disclosure']}.",
        "",
        f"**Program target** (mean teacher-relevance recall >= {target['c']} at c = "
        f"{target['c']}; the headline counts a capped unit at recall 1, the uncapped-only "
        "verdict leaves capped units out):",
        "",
        target_line(target, results.get("documents", {}), results.get("gold_elements", {})),
        "",
    ]
    for family, v in target["families"].items():
        lines.append(
            f"- {family}: headline {_verdict_text(v['pass'])} (mean {_fmt(v['mean'])}, "
            f"{_pct(v['capped_share'])} capped, n = {v['n']}), uncapped only "
            f"{_verdict_text(v['pass_uncapped'])} (mean {_fmt(v['mean_uncapped'])}, "
            f"n = {v['n_uncapped']})"
        )
    lines.append("")
    lines += _coverage(results)
    lines += _main_table(
        {"overall": main["overall"], **main["families"]}, "Main: teacher relevance"
    )
    lines += _main_table(
        {"overall": gold["overall"], **gold["families"]},
        "Secondary: gold-annotated subset (recall over gold elements, tokens over the document)",
    )
    d = results["disagreement"]
    lines += [
        "### Teacher–gold disagreement",
        "",
        f"Accuracy {_fmt(d['accuracy'])} on n = {d['n']} (unlabelled {d['unlabelled']}, "
        f"hash mismatch {d['hash_mismatch']}).",
        "",
        "| family | n | accuracy |",
        "|---|---|---|",
        *(f"| {f} | {v['n']} | {_fmt(v['accuracy'])} |" for f, v in d["families"].items()),
        "",
        "| gold topic | teacher topic | n |",
        "|---|---|---|",
        *(f"| {c['gold']} | {c['teacher']} | {c['n']} |" for c in d["confusions"]),
        "",
        "### Baselines (same tokens as the selector)",
        "",
        "| c | n | " + " | ".join(("selector", *BASELINES)) + " |",
        "|---|---|" + "---|" * (1 + len(BASELINES)),
    ]
    for key, row in results["baselines"]["overall"].items():
        cells = [
            f"{_fmt(row[name]['recall_mean'])} / {_fmt(row[name]['token_share_mean'])}"
            for name in ("selector", *BASELINES)
        ]
        lines.append(f"| {key} | {row['n']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Cells are mean recall / mean token share. Whole document: recall 1.000 / share 1.000.",
        "",
        "### Tokens to 95% recall",
        "",
        "| group | curve | dial grid |",
        "|---|---|---|",
    ]
    for group, curve in results["curve"].items():
        dial = results["dial_tokens_to_95"].get(group)
        lines.append(f"| {group} | {_fmt(curve['tokens_to_95'])} | {_fmt(dial)} |")
    return "\n".join([*lines, ""])


def _commit() -> str | None:
    from benchmarks.pdf_bench import bench_revision

    return bench_revision()


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _out_dir(value: Path | None) -> Path:
    return value if value is not None else RESULTS_ROOT / datetime.date.today().isoformat()


def _environment() -> dict[str, Any]:
    names = [*_guard.ENV_CAPS, *_guard.MPS_CAP_VARS, "DOCROOTER_DEVICE"]
    return {
        "env": {name: os.environ.get(name) for name in names},
        "platform": {"python": platform.python_version(), "machine": platform.machine()},
    }


def load_selector(selector: Path | None, tagger: str) -> Any:
    """The selector: ``selector``, else the tagger bundle's ``selector.json``."""
    from docrooter.select.selection import SELECTOR_FILE, Selector

    path = selector if selector is not None else Path(tagger) / SELECTOR_FILE
    return Selector.load(path)


# --- commands -------------------------------------------------------------------------------------


def run_gold(
    records: Sequence[Any],
    read: Callable[[Any], tuple[Any, Any]],
    labels: Mapping[str, Mapping[str, Mapping[str, str]]],
    fields: Mapping[str, Sequence[Any]],
    gold_by_doc: Mapping[str, Sequence[Mapping[str, Any]]],
    selector: Any,
) -> dict[str, Any]:
    """Every gold-split record: read it (``read(record) -> (analysis, prepared context)``) when it
    has teacher labels or gold rows, evaluate its units, and account for every document.

    The gold subset is evaluated whenever the reading succeeds, whatever the teacher labels
    say; the main units need the labels to pass :func:`~docrooter.select.fit.label_leaves`'s
    checks, whose counts are kept per document. A reading that fails, for any reason, is
    recorded with the exception's type."""
    from docrooter.errors import DocrooterError
    from docrooter.select.fit import label_leaves
    from docrooter.teach.shortlist import doc_family

    skipped: dict[str, list[Any]] = {reason: [] for reason in SKIP_REASONS}
    units: list[dict[str, Any]] = []
    gold_units: list[dict[str, Any]] = []
    curves: list[list[Any]] = []
    gold_counts: Counter[str] = Counter()
    checks: dict[str, dict[str, Any]] = {}
    read_ok: list[str] = []
    without_units: list[str] = []
    evaluated: list[str] = []
    for done, record in enumerate(records, 1):
        family = doc_family(record)
        has_labels = record.id in labels
        gold_rows = gold_by_doc.get(record.id, [])
        if not has_labels:
            skipped["no_teacher_labels"].append(record.id)
        if not fields.get(family):
            skipped["no_fields"].append(record.id)
            continue
        if not has_labels and not gold_rows:
            skipped["nothing_to_evaluate"].append(record.id)
            continue
        try:
            analysis, ctx = read(record)
        except Exception as exc:  # one bad PDF never takes the run down
            error = f"{type(exc).__name__}: {exc}"[:300]
            skipped["read_failed"].append({"doc_id": record.id, "error": error})
            log(f"[{done}/{len(records)}] {record.id}: read failed ({type(exc).__name__})")
            continue
        read_ok.append(record.id)
        leaves = None
        if has_labels:
            try:
                leaves, checks[record.id] = label_leaves(ctx, labels[record.id], record.id)
            except DocrooterError as exc:
                error = f"{type(exc).__name__}: {exc}"[:300]
                skipped["label_check_failed"].append({"doc_id": record.id, "error": error})
        doc_units, doc_gold, counts, doc_curves = evaluate_document(
            analysis,
            ctx,
            leaves,
            fields[family],
            gold_rows,
            selector,
            doc_id=record.id,
            family=family,
        )
        units += doc_units
        gold_units += doc_gold
        curves += doc_curves
        gold_counts.update(counts)
        (evaluated if doc_units or doc_gold else without_units).append(record.id)
        log(
            f"[{done}/{len(records)}] {record.id}: {len(doc_units)} units, "
            f"{len(doc_gold)} gold-subset units"
        )
    totals: Counter[str] = Counter()
    for check in checks.values():
        totals.update({k: v for k, v in check.items() if k != "rate"})
    return {
        "units": units,
        "gold_units": gold_units,
        "curves": curves,
        "gold_elements": {
            "gold": gold_counts["gold"],
            "matched": gold_counts["matched"],
            "hash_mismatch": gold_counts["hash_mismatch"],
            "missing": gold_counts["missing"],
        },
        "label_checks": {"totals": dict(totals), "per_document": checks},
        "documents": {
            "gold_split": len(records),
            "read": len(read_ok),
            "evaluated": len(evaluated),
            "read_without_units": without_units,
            "skipped": skipped,
        },
    }


def cmd_gold(args: argparse.Namespace) -> int:
    from docrooter.backends.base import close_backend
    from docrooter.backends.registry import resolve_backend
    from docrooter.corpus.store import Corpus
    from docrooter.rooter.profiles import profile_defaults
    from docrooter.select.fields import builtin_fields, list_builtin_field_sets
    from docrooter.select.fit import CorpusPdfReader, labels_by_document
    from docrooter.select.selection import prepare
    from docrooter.tagger.element_tags import resolve_tagger
    from docrooter.teach.shortlist import doc_family

    started = time.perf_counter()
    corpus = Corpus(args.store)
    gold_rows = read_gold(args.gold)
    labels = labels_by_document(corpus)
    disagreement = teacher_gold_disagreement(gold_rows, labels)
    selector = load_selector(args.selector, args.tagger)
    profile, max_pages, warnings = reading_settings(args.profile, args.max_pages, selector.meta)
    for warning in warnings:
        log(f"warning: {warning}")
    embedder = args.embedder or selector.embedder
    if embedder is None:
        raise SystemExit("the selector records no embedder; pass --embedder")
    fields = {family: builtin_fields(family) for family in list_builtin_field_sets()}
    gold_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gold_rows:
        gold_by_doc[str(row["doc_id"])].append(row)
    records = [r for r in corpus.manifest() if r.split == "gold"]
    log(f"{len(records)} gold-split documents, {sum(r.id in labels for r in records)} labelled")

    tagger = resolve_tagger(args.tagger)
    encoder = resolve_backend(embedder)
    same = embedder == profile_defaults(profile)["sections"]
    reader = CorpusPdfReader(
        corpus,
        args.pdf_dir,
        profile=profile,
        max_pages=max_pages,
        sections_backend=encoder if same else None,
    )

    def read(record: Any) -> tuple[Any, Any]:
        analysis = reader(record)
        family = doc_family(record)
        return analysis, prepare(analysis, tagger=tagger, encoder=encoder, family=family)

    try:
        run = run_gold(records, read, labels, fields, gold_by_doc, selector)
    finally:
        reader.close()
        close_backend(tagger)
        close_backend(encoder)

    units = run["units"]
    main = aggregate(units, C_GRID)
    target = program_target(main)
    results: dict[str, Any] = {
        "date": datetime.date.today().isoformat(),
        "disclosure": DISCLOSURE,
        "c_grid": list(C_GRID),
        "documents": run["documents"],
        "main": main,
        "program_target": target,
        "gold_subset": aggregate(run["gold_units"], C_GRID),
        "gold_elements": run["gold_elements"],
        "label_checks": run["label_checks"],
        "disagreement": disagreement,
        "baselines": aggregate_baselines(units, C_GRID),
        "curve": summarise_curves(run["curves"]),
        "dial_tokens_to_95": dial_tokens_to_95(main),
        "units": units,
        "gold_units": run["gold_units"],
        "provenance": {
            "commit": _commit(),
            "tagger": args.tagger,
            "selector": str(args.selector) if args.selector else None,
            "selector_meta": {k: selector.meta.get(k) for k in ("profile", "max_pages", "dials")},
            "embedder": embedder,
            "profile": profile,
            "max_pages": max_pages,
            "reading_warnings": warnings,
            "gold_sha256": _sha256(args.gold),
        },
        "seconds": round(time.perf_counter() - started, 1),
        **_environment(),
    }
    out = _out_dir(args.out)
    _write_json(out / RESULTS_FILE, results)
    (out / MARKDOWN_FILE).write_text(markdown(results), encoding="utf-8")
    log(f"{target_line(target, run['documents'], run['gold_elements'])} -> {out / RESULTS_FILE}")
    return 0


def cmd_markdown(args: argparse.Namespace) -> int:
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    sys.stdout.write(markdown(results))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("gold", help="the main evaluation on the gold split (loads models)")
    p.add_argument("--tagger", required=True, help="element tagger bundle directory")
    p.add_argument("--selector", type=Path, default=None, help="selector.json (default: bundle's)")
    p.add_argument("--embedder", default=None, help="embedding spec (default: the selector's)")
    p.add_argument("--store", type=Path, default=STORE, help="corpus store")
    p.add_argument("--pdf-dir", type=Path, default=PDF_DIR, help="folder of <id>.pdf")
    p.add_argument("--gold", type=Path, default=GOLD_PATH)
    p.add_argument("--profile", default=None, help="reading profile (default: the selector fit's)")
    p.add_argument(
        "--max-pages", type=int, default=None, help="page cap (default: the selector fit's)"
    )
    p.add_argument("--out", type=Path, default=None, help="results dir (default: dated)")
    _guard.add_guard_argument(p)
    p = sub.add_parser("markdown", help="print a results file as Markdown (no models)")
    p.add_argument("results", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "markdown":
        return cmd_markdown(args)
    _guard.single_model_process(
        "benchmarks.selection_bench", allow_concurrent=args.allow_concurrent
    )
    return cmd_gold(args)


if __name__ == "__main__":
    sys.exit(main())
