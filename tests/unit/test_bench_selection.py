"""Selection benchmark (spec §8, task S7): recall and token share, the curve and tokens-to-95,
the baselines at the selector's budget, the evidence-QA alignment and its cache-only writes,
and the extraction harness's scoring and refusals.

Fakes only: hand-built selections, scores and token counts, a fake HTTP transport. No model is
loaded, nothing touches the network and no real benchmark data is read.
"""

from __future__ import annotations

import io
import json
import os
import urllib.error
from collections import Counter
from email.message import Message
from pathlib import Path

import numpy as np
import pytest

from benchmarks import _guard
from benchmarks import evidence_qa as qa
from benchmarks import selection_bench as bench
from benchmarks.extraction import run as extraction
from benchmarks.training_corpus._net import Net, RateLimiter
from docrooter import Document, RouterConfig
from docrooter.classify.normalize import NormalizationResult
from docrooter.classify.results import DocTypeResult, PathResult
from docrooter.corpus.store import text_sha1
from docrooter.elements.models import Box, Element, TagResult
from docrooter.rooter.analysis import Analysis
from docrooter.select import Field, calibration_of
from docrooter.select.crc import DialTable
from docrooter.select.relevance import FEATURE_NAMES, Combiner
from docrooter.select.selection import FitProvenance, Selection, Selector, prepare


def selection(ids, tokens, share, *, c=0.9, guarantee=True, capped=False) -> Selection:
    return Selection(
        need="annual_report.risk_factors",
        completeness=c,
        threshold=0.5,
        element_ids=list(ids),
        tokens=tokens,
        token_share=share,
        expected_recall=c,
        guarantee=guarantee,
        chunks=[],
        capped=capped,
    )


# --- recall and token share ----------------------------------------------------------------------


def unit(family, field_id, recalls, shares, guarantee=True, capped=False, doc_id=None):
    """A unit result at c = 0.8 and 0.9."""
    return {
        "doc_id": doc_id or f"d-{field_id}",
        "family": family,
        "field_id": field_id,
        "by_c": {
            bench.c_key(c): {
                "recall": r,
                "token_share": s,
                "tokens": 1,
                "guarantee": guarantee,
                "capped": capped,
            }
            for c, r, s in zip((0.8, 0.9), recalls, shares, strict=True)
        },
    }


def test_recall_and_token_share():
    # Recall is over the relevant content elements, after expansion: what the user gets.
    assert bench.recall({"e1", "e2", "h0"}, {"e1", "e3"}) == 0.5
    assert bench.recall({"e1"}, set()) is None  # undefined, never 0 or 1

    row = bench.evaluate_selection(selection(["h0", "e1", "e2"], 30, 0.3), {"e1", "e3"})
    assert row == {
        "recall": 0.5,
        "token_share": 0.3,
        "tokens": 30,
        "guarantee": True,
        "capped": False,
    }

    units = [
        unit("annual_report", "a", (0.8, 1.0), (0.2, 0.3)),
        unit("annual_report", "b", (0.6, 0.8), (0.4, 0.5)),
        unit("fund_document", "c", (1.0, 1.0), (1.0, 1.0), capped=True),
    ]
    agg = bench.aggregate(units, (0.8, 0.9))
    overall = agg["overall"][bench.c_key(0.9)]
    assert overall["n"] == 3
    assert overall["recall_mean"] == pytest.approx((1.0 + 0.8 + 1.0) / 3)
    assert overall["recall_min"] == 0.8
    assert overall["recall_p10"] == pytest.approx(np.percentile([1.0, 0.8, 1.0], 10))
    assert overall["token_share_mean"] == pytest.approx((0.3 + 0.5 + 1.0) / 3)
    assert overall["guarantee_share"] == 1.0
    assert overall["capped_share"] == pytest.approx(1 / 3)
    ar = agg["families"]["annual_report"][bench.c_key(0.8)]
    assert ar["n"] == 2 and ar["recall_mean"] == pytest.approx(0.7)
    assert ar["capped_share"] == 0.0

    # The program target (spec §0.5): mean recall >= c at c = 0.9, reported honestly.
    verdict = bench.program_target(agg)
    assert verdict["c"] == 0.9 and verdict["pass"] is True
    low = bench.aggregate([unit("annual_report", "x", (0.5, 0.85), (0.1, 0.2))], (0.8, 0.9))
    assert bench.program_target(low)["pass"] is False


def test_program_target_shows_capped_units_and_exclusions():
    # A capped unit returns the whole document: recall 1 in the headline, as the user gets it,
    # but the uncapped-only mean and the capped share are shown beside it, with their verdicts.
    units = [
        unit("annual_report", "a", (1.0, 1.0), (1.0, 1.0), capped=True),
        unit("annual_report", "b", (0.6, 0.85), (0.3, 0.4)),
    ]
    agg = bench.aggregate(units, (0.8, 0.9))
    stats = agg["overall"][bench.c_key(0.9)]
    assert stats["n_uncapped"] == 1 and stats["recall_mean_uncapped"] == pytest.approx(0.85)
    verdict = bench.program_target(agg)
    assert verdict["mean"] == pytest.approx(0.925) and verdict["pass"] is True
    assert verdict["capped_share"] == 0.5 and verdict["guarantee_share"] == 1.0
    assert verdict["mean_uncapped"] == pytest.approx(0.85) and verdict["n_uncapped"] == 1
    assert verdict["pass_uncapped"] is False
    assert verdict["families"]["annual_report"]["pass_uncapped"] is False
    all_capped = bench.program_target(
        bench.aggregate([unit("fund_document", "c", (1.0, 1.0), (1.0, 1.0), capped=True)], (0.9,))
    )
    assert all_capped["pass"] is True and all_capped["pass_uncapped"] is None  # undefined

    documents = {
        "gold_split": 10,
        "read": 8,
        "evaluated": 6,
        "read_without_units": ["d7", "d8"],
        "without_teacher_labels": ["d9"],
        "skipped": {
            "no_fields": [],
            "read_failed": [{"doc_id": "d10", "error": "DocrooterError: x"}],
            "label_check_failed": [{"doc_id": "d6", "error": "DocrooterError: y"}],
        },
    }
    gold_elements = {"gold": 900, "matched": 880, "hash_mismatch": 15, "missing": 5}
    line = bench.target_line(verdict, documents, gold_elements)
    assert "headline PASS" in line and "uncapped only FAIL" in line
    assert "50% of units capped" in line and "mean 0.850" in line
    assert "1 without teacher labels (gold subset only)" in line and "read_failed 1" in line
    assert "no_teacher_labels" not in line
    assert "label_check_failed 1" in line and "2 read without units" in line
    assert "15 hash mismatch" in line and "5 missing" in line

    results = {
        "date": "2026-10-19",
        "disclosure": bench.DISCLOSURE,
        "provenance": {"commit": "abc"},
        "documents": documents,
        "gold_elements": gold_elements,
        "label_checks": {"totals": {"content": 10, "no_label": 1, "missing": 0}},
        "main": agg,
        "program_target": verdict,
        "gold_subset": agg,
        "disagreement": bench.teacher_gold_disagreement([], {}),
        "baselines": bench.aggregate_baselines([], (0.9,)),
        "curve": {},
        "dial_tokens_to_95": {},
    }
    text = bench.markdown(results)
    assert line in text and "### Coverage" in text
    assert "| read_failed | 1 |" in text and "| hash mismatch | 15 |" in text


def test_reading_settings_default_to_the_selector_fit():
    meta = {"profile": "balanced", "max_pages": 20}
    assert bench.reading_settings(None, None, meta) == ("balanced", 20, [])
    profile, pages, warnings = bench.reading_settings("accurate", 20, meta)
    assert (profile, pages) == ("accurate", 20)
    assert len(warnings) == 1 and "balanced" in warnings[0] and "accurate" in warnings[0]
    assert bench.reading_settings(None, 40, meta)[2][0].count("max-pages") == 1
    assert bench.reading_settings(None, None, {}) == ("accurate", 30, [])


# --- the curve and tokens-to-95 ------------------------------------------------------------------


def test_tokens_to_95_recall():
    # Three content rows; row 0 brings a heading with it (expansion), row 2 is relevant.
    p = np.array([0.9, 0.5, 0.2])
    expansions = [["h0", "e0"], ["e1"], ["e2"]]
    tokens = {"h0": 10, "e0": 20, "e1": 30, "e2": 40}
    curve = bench.threshold_curve(p, expansions, tokens, 100, {"e0", "e2"}, (1.0, 0.8, 0.3, 0.0))
    assert curve == [
        (0.0, 0.0),  # nothing reaches 1.0
        (pytest.approx(0.3), 0.5),  # row 0 and its heading
        (pytest.approx(0.6), 0.5),
        (pytest.approx(1.0), 1.0),
    ]

    thresholds = (0.6, 0.4, 0.2)
    curves = [
        [(0.1, 0.5), (0.3, 0.9), (0.5, 1.0)],
        [(0.2, 0.7), (0.4, 1.0), (0.6, 1.0)],
    ]
    mean = bench.mean_curve(curves, thresholds)
    assert [point["threshold"] for point in mean] == [0.6, 0.4, 0.2]
    assert mean[1]["token_share"] == pytest.approx(0.35)
    assert mean[1]["recall"] == pytest.approx(0.95)
    # The smallest token share at which the mean recall reaches 0.95.
    assert bench.tokens_to_recall(mean, 0.95) == pytest.approx(0.35)
    assert bench.tokens_to_recall(mean, 0.999) == pytest.approx(0.55)
    assert bench.tokens_to_recall(mean[:1], 0.95) is None  # never reached


# --- baselines -----------------------------------------------------------------------------------


def test_baselines_same_budget():
    texts = [
        "Revenue grew strongly in the year.",
        "Risk factors: currency risk and interest rate risk.",
        "The board met six times.",
        "Liquidity risk is managed by treasury.",
        "Our offices are in Stockholm.",
    ]
    tokens = [10, 12, 8, 9, 7]
    sim = np.array([0.1, 0.9, 0.2, 0.7, 0.0])
    section_hit = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    budget = 21

    rows = bench.baseline_rows(texts, tokens, sim, section_hit, "What is the risk?", budget)
    assert set(rows) == set(bench.BASELINES) == {"lead_k", "bm25", "similarity", "structure"}
    for name, picked in rows.items():
        assert sum(tokens[j] for j in picked) <= budget, name
    assert rows["lead_k"] == [0]  # a prefix in reading order: 10 + 12 > 21, so it stops
    assert rows["similarity"] == [1, 3]  # by sim, each one that still fits (reading order out)
    assert rows["structure"] == [1, 3]  # section hits in reading order, then nothing
    assert set(rows["bm25"]) == {1, 3}  # the two elements that mention "risk"

    # BM25: lower-cased \w+ tokens without stopwords, k1 = 1.5, b = 0.75.
    assert bench.tokenize("The Risks, and RISK!") == ["risks", "risk"]
    scores = bench.bm25_scores(texts, "risk")
    assert scores[1] > scores[3] > 0.0 and scores[0] == scores[2] == scores[4] == 0.0
    assert bench.bm25_scores(texts, "the and").tolist() == [0.0] * 5  # stopwords only

    # Every baseline is scored like the selector: recall over relevant rows, share of all tokens.
    assert bench.rows_score([1, 3], {1}, tokens, 100) == (1.0, pytest.approx(0.21))
    assert bench.rows_score([], {1}, tokens, 100) == (0.0, 0.0)
    # Structure only takes nothing beyond the section hits, however large the budget.
    assert bench.structure_only(section_hit, tokens, 1000) == [1, 3]
    # The whole document is the reference row.
    assert bench.WHOLE_DOCUMENT == {"recall": 1.0, "token_share": 1.0}


def test_gold_subset_and_teacher_gold_disagreement():
    gold = [
        {"doc_id": "d", "element_id": "e1", "text_sha1": "s1", "family": "f", "topic": "f.a"},
        {"doc_id": "d", "element_id": "e2", "text_sha1": "s2", "family": "f", "topic": "f.b"},
        {"doc_id": "d", "element_id": "e3", "text_sha1": "s3", "family": "f", "topic": "f.b"},
        {"doc_id": "d", "element_id": "e4", "text_sha1": "s4", "family": "f", "topic": "f.a"},
    ]
    matched, counts = bench.gold_matches(gold, {"e1": "s1", "e2": "s2", "e3": "changed"})
    assert matched == {"e1": "f.a", "e2": "f.b"}
    assert counts == {"gold": 4, "matched": 2, "hash_mismatch": 1, "missing": 1}

    labels = {"d": {"e1": {"s1": "f.a"}, "e2": {"s2": "f.a"}, "e3": {"other": "f.b"}}}
    result = bench.teacher_gold_disagreement(gold, labels)
    assert result["n"] == 2 and result["accuracy"] == 0.5
    assert result["hash_mismatch"] == 1 and result["unlabelled"] == 1
    assert result["families"]["f"] == {"n": 2, "accuracy": 0.5}
    assert result["confusions"] == [{"gold": "f.b", "teacher": "f.a", "n": 1}]


# --- one document, end to end with fakes ---------------------------------------------------------

RISK = Field(
    id="annual_report.risk_factors",
    label="Risk factors",
    description="The risks the company faces.",
    topics=["annual_report.risk_factors"],
    query="What are the main risks?",
)
LEGAL = Field(
    id="annual_report.legal_proceedings",
    label="Legal proceedings",
    description="Lawsuits.",
    topics=["annual_report.legal_proceedings"],
)


def _combiner(names, weights) -> Combiner:
    names = list(names)
    return Combiner.from_dict(
        {
            "format": "docrooter-relevance-combiner",
            "version": 1,
            "feature_names": names,
            "l2": 1.0,
            "max_iter": 100,
            "mean": [0.0] * len(names),
            "scale": [1.0] * len(names),
            "coef": [weights.get(name, 0.0) for name in names],
            "intercept": -5.0,
            "n_train": 10,
            "n_docs": 2,
            "converged": True,
        }
    )


TAGGER_FP = "fp-of-the-bench-bundle"
EMBEDDER_SPEC = "embedding:bench-fake"
PROVENANCE = FitProvenance(
    tagger=TAGGER_FP,
    embedder=EMBEDDER_SPEC,
    profile="accurate",
    layout="heron",
    sections="embedding:dmis-lab/UltraBERT-Embed-150M",
)


class FakeTagger:
    """Tags as a bundle of fingerprint :data:`TAGGER_FP` would (the fake selector's fit)."""

    def __init__(self, fingerprint=TAGGER_FP):
        self.fingerprint = fingerprint

    def tag(self, elements, family):
        leaves = [
            f"{family}.{'risk_factors' if 'risk' in e.text.lower() else 'other'}" for e in elements
        ]
        return [
            {
                "topic": TagResult(
                    labels=[leaf],
                    probs={leaf: 1.0},
                    confidence=1.0,
                    source="fake",
                    fingerprint=self.fingerprint,
                )
            }
            for leaf in leaves
        ]


class FakeEmbedder:
    spec = EMBEDDER_SPEC

    def encode(self, texts, prompt=""):
        return np.array([[1.0, 0.0] if "risk" in t.lower() else [0.0, 1.0] for t in texts])


def _fake_document():
    field = _combiner(FEATURE_NAMES, {"p_topic": 10.0})
    dial = DialTable(
        family="annual_report",
        n=100,
        n_docs=100,
        n_excluded=0,
        grid=(0.5, 0.9),
        curve_lambdas=[0.0, 0.5, 1.0],
        curve_risk=[0.0, 0.0, 1.0],
        combiner_fingerprint=field.fingerprint(),
    )
    fallback = _combiner(("sim", "section_hit"), {"sim": 10.0})
    selector = Selector(
        field=field,
        field_fallback=fallback,
        query=_combiner(FEATURE_NAMES, {"sim": 10.0}),
        query_fallback=fallback,
        dials={"annual_report": dial},
        calibrated_fields=calibration_of([RISK, LEGAL]),
        provenance=PROVENANCE,
    )
    doc = Document.from_sections([{"title": "Factors", "level": 1}])
    texts = [
        ("heading", "Factors"),
        ("paragraph", "This section opens the discussion."),
        ("paragraph", "Filler about our offices and operations."),
        ("paragraph", "Interest rate risk could harm our results."),
        ("paragraph", "Closing filler."),
    ]
    doc.elements = [
        Element(
            id=f"e{i}",
            kind=kind,
            text=text,
            boxes=[Box(page=1, x0=0, y0=0, x1=1, y1=1)],
            order=i,
            section_id="s1",
            **({"level": 1} if kind == "heading" else {}),
        )
        for i, (kind, text) in enumerate(texts)
    ]
    doc.sections["s1"].element_ids.extend(e.id for e in doc.elements)
    analysis = Analysis(
        document=doc,
        doc_type=DocTypeResult(
            source="test", best=PathResult(node_ids=["financial", "financial.annual_report"])
        ),
        sections=NormalizationResult(doc_type=None, taxonomy="section_types/annual_report"),
        # the reading the fake selector was fitted on (escalation off whatever the environment)
        config=RouterConfig(profile="accurate", escalate="none"),
    )
    ctx = prepare(analysis, tagger=FakeTagger(), encoder=FakeEmbedder(), family="annual_report")
    return analysis, ctx, selector


def test_evaluate_document_with_fakes():
    analysis, ctx, selector = _fake_document()
    leaves = [tags["topic"].labels[0] for tags in ctx.tags]
    doc = analysis.document
    gold = [
        {
            "doc_id": "d",
            "element_id": "e3",
            "text_sha1": text_sha1(doc.elements[3].text),
            "family": "annual_report",
            "topic": "annual_report.risk_factors",
        }
    ]
    units, gold_units, counts, curves = bench.evaluate_document(
        analysis, ctx, leaves, [RISK, LEGAL], gold, selector, doc_id="d", family="annual_report"
    )
    # LEGAL has no relevant element: no unit (recall would be undefined).
    assert [u["field_id"] for u in units] == [RISK.id] == [u["field_id"] for u in gold_units]
    assert counts["matched"] == 1
    unit = units[0]
    assert unit["by_c"]["1.00"] == {
        "recall": 1.0,
        "token_share": 1.0,
        "tokens": unit["by_c"]["1.00"]["tokens"],
        "guarantee": True,
        "capped": False,
    }
    assert unit["by_c"]["0.90"]["recall"] == 1.0 and unit["by_c"]["0.90"]["guarantee"] is True
    budget = unit["by_c"]["0.90"]["tokens"]
    assert set(unit["baselines"]["0.90"]) == set(bench.BASELINES)
    for score in unit["baselines"]["0.90"].values():
        assert score["token_share"] <= unit["by_c"]["0.90"]["token_share"] + 1e-12
    assert budget > 0 and "1.00" not in unit["baselines"]
    assert len(curves) == 1 and len(curves[0][1]) == len(bench.THRESHOLDS)
    assert curves[0][1][-1] == (pytest.approx(1.0), 1.0)  # threshold 0: everything

    # Evidence QA on the same document: the question as a need and its closest field.
    evidence = (qa.Evidence(1, "Interest rate risk could harm our results."),)
    question = qa.Question("q1", "d", "Which interest rate risk?", "x", evidence)
    matcher = qa.FieldMatcher([RISK, LEGAL], FakeEmbedder())
    rows, tally = qa.evaluate_document(
        analysis, ctx, [question], matcher, selector, {1}, other_base_shift=-1
    )
    assert tally["aligned"] == 1 and tally["aligned_if_other_base"] == 0
    assert rows[0]["field_id"] == RISK.id and rows[0]["n_relevant"] == 1
    assert set(rows[0]["query"]) == {bench.c_key(c) for c in qa.QUERY_GRID}
    assert all(r["guarantee"] is False for r in rows[0]["query"].values())
    assert "Which" not in json.dumps(rows)  # ids and numbers only
    off_page = qa.Question("q2", "d", "q", "x", (qa.Evidence(7, "anything"),))
    _, tally = qa.evaluate_document(
        analysis, ctx, [off_page], matcher, selector, {1}, other_base_shift=None
    )
    assert tally["pages_not_read"] == 1

    # Without teacher labels, the gold subset is still evaluated.
    units, gold_units, _, curves = bench.evaluate_document(
        analysis, ctx, None, [RISK, LEGAL], gold, selector, doc_id="d", family="annual_report"
    )
    assert units == [] and curves == [] and [u["field_id"] for u in gold_units] == [RISK.id]


def test_run_gold_counts_every_document():
    from docrooter.corpus.store import ManifestRecord
    from docrooter.errors import DocrooterError

    analysis, ctx, selector = _fake_document()
    doc = analysis.document
    content = [e for e in doc.elements if e.kind != "heading"]

    def leaves(topic_of, sha=text_sha1):
        return {e.id: {sha(e.text): topic_of(e)} for e in content}

    def risky(e):
        return "annual_report.risk_factors" if "risk" in e.text.lower() else "annual_report.other"

    labels = {
        "ok": leaves(risky),
        "nounits": leaves(lambda e: "annual_report.other"),
        "badcheck": leaves(risky, sha=lambda text: "0" * 40),  # labels for other texts
    }
    gold_row = {
        "element_id": "e3",
        "text_sha1": text_sha1(doc.elements[3].text),
        "family": "annual_report",
        "topic": "annual_report.risk_factors",
    }
    gold_by_doc = {d: [{**gold_row, "doc_id": d}] for d in ("ok", "nolabels", "badcheck", "broken")}

    def record(doc_id):
        return ManifestRecord(id=doc_id, sha256="0" * 64, family_hint="annual_report", split="gold")

    def read(rec):
        if rec.id == "broken":
            raise DocrooterError("cannot open PDF")
        return analysis, ctx

    ids = ["ok", "nounits", "nolabels", "badcheck", "broken", "nothing"]
    run = bench.run_gold(
        [record(d) for d in ids], read, labels, {"annual_report": [RISK, LEGAL]},
        gold_by_doc, selector,
    )  # fmt: skip
    docs = run["documents"]
    assert docs["gold_split"] == 6 and docs["read"] == 4
    assert sorted({u["doc_id"] for u in run["units"]}) == ["ok"]
    assert sorted({u["doc_id"] for u in run["gold_units"]}) == ["badcheck", "nolabels", "ok"]
    assert docs["evaluated"] == 3  # with any unit, main or gold subset
    assert docs["read_without_units"] == ["nounits"]
    skipped = docs["skipped"]
    # a document without teacher labels is not skipped: its gold subset is still evaluated
    assert docs["without_teacher_labels"] == ["nolabels", "broken", "nothing"]
    assert "no_teacher_labels" not in skipped
    assert [f["doc_id"] for f in skipped["read_failed"]] == ["broken"]
    assert skipped["read_failed"][0]["error"].startswith("DocrooterError")
    assert [f["doc_id"] for f in skipped["label_check_failed"]] == ["badcheck"]
    assert skipped["nothing_to_evaluate"] == ["nothing"]  # no labels and no gold: never read
    checks = run["label_checks"]
    assert set(checks["per_document"]) == {"ok", "nounits"}
    assert checks["totals"]["content"] == 2 * len(content)
    assert run["gold_elements"]["matched"] == 3


def test_selection_bench_and_evidence_qa_are_model_processes():
    root = "/x/.venv/bin/python3"
    table = [
        (30, 1, "python3.12", f"{root} -m benchmarks.selection_bench gold --tagger b"),
        (31, 1, "python3.12", f"{root} -m benchmarks.evidence_qa run --tagger b"),
        (32, 1, "python3.12", f"{root} -m benchmarks.extraction.run --mlx-model m"),
        (33, 1, "python3.12", f"{root} -m pytest tests/unit/test_bench_selection.py"),
    ]
    found = _guard.other_model_processes(table, pid=99, ppid=98)
    assert [pid for pid, _ in found] == [30, 31, 32]


# --- evidence QA ---------------------------------------------------------------------------------


def test_evidence_alignment_char_overlap():
    evidence_text = (
        "Total revenue was $2,468 million in 2031, up 4%.   Net income was $300 million."
    )
    elements = [
        qa.PageText("e1", frozenset({5}), "Total revenue was\n$2,468 million in 2031, up 4%."),
        qa.PageText("e2", frozenset({5}), "The weather in Ruritania was cold that winter."),
        qa.PageText("e3", frozenset({6}), "Total revenue was $2,468 million in 2031, up 4%."),
        qa.PageText("e4", frozenset({5}), ""),
        qa.PageText(
            "e5",
            frozenset({5}),
            "Net income was $300 million. Other unrelated sentences follow here, long enough "
            "that most of this element is not in the evidence at all, so it stays out.",
        ),
    ]
    # Whitespace is normalised; the share is of the element's characters.
    assert qa.overlap_share(elements[0].text, evidence_text) == pytest.approx(1.0)
    assert qa.overlap_share(elements[1].text, evidence_text) < 0.5
    assert qa.overlap_share("", evidence_text) == 0.0
    relevant = qa.evidence_elements(elements, [qa.Evidence(page=5, text=evidence_text)])
    assert relevant == {"e1"}  # e3 is off the evidence page, e5 under half overlapping
    # Several evidence items: the union of each page's matches.
    both = qa.evidence_elements(
        elements, [qa.Evidence(5, evidence_text), qa.Evidence(6, evidence_text)]
    )
    assert both == {"e1", "e3"}

    # FinanceBench rows: 0-based evidence pages by default, ids and evidence kept.
    row = {
        "financebench_id": "financebench_id_00001",
        "doc_name": "ACME_2031_10K",
        "question": "What was revenue?",
        "answer": "$2,468 million",
        "evidence": [{"evidence_text": evidence_text, "evidence_page_num": 4}],
    }
    question = qa.parse_financebench([json.dumps(row)])[0]
    assert question.id == "financebench_id_00001" and question.doc_id == "ACME_2031_10K"
    assert question.evidence == (qa.Evidence(5, evidence_text),)
    assert qa.parse_financebench([json.dumps(row)], page_base=1)[0].evidence[0].page == 4


def test_schema_failures_are_loud():
    good = qa.Question("q1", "d", "q", "a", (qa.Evidence(1, "t"),))
    bare = qa.Question("q2", "d", "q", "a", ())
    qa.check_questions([good, good, bare], "financebench")  # one in three: fine
    with pytest.raises(SystemExit, match="no question loaded.*schema"):
        qa.check_questions([], "tat-dqa")
    with pytest.raises(SystemExit, match="3 of 4 questions have no evidence.*schema"):
        qa.check_questions([good, bare, bare, bare], "tat-dqa")

    assert qa.check_alignment({"questions_evaluated": 4, "aligned": 3}) == []
    with pytest.raises(SystemExit, match="no question aligned.*page base.*schema"):
        qa.check_alignment({"questions_evaluated": 4, "aligned": 0, "unaligned": 4})
    with pytest.raises(SystemExit, match="no evidence.*schema"):
        qa.check_alignment({"questions_evaluated": 4, "aligned": 1, "no_evidence": 3})
    warnings = qa.check_alignment(
        {"questions_evaluated": 4, "aligned": 1, "aligned_if_other_base": 3}
    )
    assert len(warnings) == 1 and "page base" in warnings[0]


def test_field_texts_are_embedded_once():
    class Counting(FakeEmbedder):
        def __init__(self):
            self.batches = []

        def encode(self, texts, prompt=""):
            self.batches.append(len(texts))
            return super().encode(texts, prompt)

    encoder = Counting()
    matcher = qa.FieldMatcher([LEGAL, RISK], encoder)
    assert matcher.closest("Which risks matter?") is RISK
    assert matcher.closest("Who sued whom?").id in (LEGAL.id, RISK.id)
    assert encoder.batches == [2, 1, 1]  # the fields once, then one question at a time


def test_field_matcher_scores_as_similarities_does():
    """The matcher splits similarities() in two halves; it must give the same numbers."""
    from docrooter.select.selection import _need_sims, need_texts, similarities

    encoder = FakeEmbedder()
    fields = [LEGAL, RISK]
    matcher = qa.FieldMatcher(fields, encoder)
    for question in ("Which risks matter?", "Who sued whom?"):
        split = _need_sims(matcher.vectors, question, encoder)
        whole = similarities([need_texts(f)[0] for f in fields], question, encoder)
        assert np.allclose(split, whole)


def test_a_partial_flag_says_whether_every_document_was_processed():
    """A document with no aligned question has no rows but was processed: not partial."""
    assert qa.is_partial(Counter({"documents_processed": 3}), 3) is False
    assert qa.is_partial(Counter({"documents_processed": 2}), 3) is True
    assert qa.is_partial(Counter(), 1) is True


def test_a_financebench_schema_failure_falls_back_to_tat_dqa(tmp_path, monkeypatch):
    def misread(cache, **kwargs):
        qa.check_questions([], qa.FINANCEBENCH)  # a schema the loader does not match

    fallback = qa.Dataset(qa.TAT_DQA, qa.LICENCES[qa.TAT_DQA], [], tmp_path, {"documents": 0})
    monkeypatch.setattr(qa, "fetch_financebench", misread)
    monkeypatch.setattr(qa, "load_tat_dqa", lambda cache, **kwargs: fallback)
    cache = tmp_path / "evidence_qa"
    assert qa.fetch(cache) is fallback
    record = json.loads((cache / "dataset.json").read_text())
    assert record["dataset"] == qa.TAT_DQA and "schema" in record["fallback_reason"]
    # on its own, a schema failure still exits loudly
    with pytest.raises(SystemExit, match="schema"):
        qa.check_questions([], qa.TAT_DQA)


def test_run_documents_survives_a_crash_and_writes_as_it_goes():
    written = []

    def evaluate(doc_id, payload):
        if doc_id == "bad":
            raise RuntimeError("engine crashed")
        return [{"id": f"{doc_id}-q"}], Counter({"aligned": 1})

    rows, counts, failures = qa.run_documents(
        [("a", None), ("bad", None), ("c", None)],
        evaluate,
        lambda rows, counts, failures: written.append(len(rows)),
    )
    assert [r["id"] for r in rows] == ["a-q", "c-q"]
    assert counts["aligned"] == 2 and counts["documents_failed"] == 1
    assert counts["documents_processed"] == 3  # failed or not, every one was processed
    assert failures == [{"doc_id": "bad", "error": "RuntimeError"}]
    assert written == [1, 1, 2]  # after every document, the failed one included
    with pytest.raises(KeyboardInterrupt):
        qa.run_documents([("x", None)], lambda d, p: (_ for _ in ()).throw(KeyboardInterrupt()),
                         lambda *a: None)  # fmt: skip


def test_fetch_builds_its_own_net_in_the_cache_and_records_the_dataset(tmp_path, monkeypatch):
    from benchmarks.training_corpus import _net

    row = {
        "financebench_id": "financebench_id_1",
        "doc_name": "DOC_A",
        "question": "q",
        "answer": "a",
        "evidence": [{"evidence_text": "t", "evidence_page_num": 0}],
    }
    built = []

    class FakeNet:
        def __init__(self, cache_dir=None, **kwargs):
            built.append(cache_dir)

        def listing(self, url):
            return json.dumps(row).encode()

        def download_pdf(self, url, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"%PDF-1.4 fake")

    monkeypatch.setattr(_net, "Net", FakeNet)
    cache = tmp_path / "evidence_qa"
    dataset = qa.fetch(cache, max_documents=5)
    assert dataset.name == "financebench" and len(dataset.questions) == 1
    assert built == [cache / "listings"]
    record = json.loads((cache / "dataset.json").read_text())
    assert record["dataset"] == "financebench" and record["fallback_reason"] is None
    assert sorted(p.name for p in tmp_path.iterdir()) == ["evidence_qa"]


class _Response(io.BytesIO):
    def __init__(self, body: bytes) -> None:
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body))}
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def test_nc_dataset_never_written_outside_cache(tmp_path, monkeypatch):
    rows = [
        {
            "financebench_id": f"financebench_id_{k}",
            "doc_name": f"DOC_{k % 3}",
            "question": "q",
            "answer": "a",
            "evidence": [{"evidence_text": "t", "evidence_page_num": 0}],
        }
        for k in range(6)
    ]
    body = "\n".join(json.dumps(r) for r in rows).encode()

    def transport(request, timeout):
        url = request.full_url
        if url.endswith("robots.txt"):
            raise urllib.error.HTTPError(url, 404, "Not Found", Message(), None)
        if url.endswith(".jsonl"):
            return _Response(body)
        if url.endswith("DOC_1.pdf") and "raw.githubusercontent" in url:
            return _Response(b"version https://git-lfs.github.com/spec/v1\n")  # an LFS pointer
        return _Response(b"%PDF-1.4 fake")

    cache = tmp_path / "cache" / "evidence_qa"
    cache.parent.mkdir()  # the git-ignored cache root exists; the fetch makes the rest
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    writes: list[Path] = []
    real_open, real_replace = Path.open, os.replace
    real_write_bytes, real_write_text = Path.write_bytes, Path.write_text

    def spy_open(self, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            writes.append(Path(self))
        return real_open(self, mode, *args, **kwargs)

    def spy_replace(src, dst, *args, **kwargs):
        writes.append(Path(dst))
        return real_replace(src, dst, *args, **kwargs)

    def spy_write_bytes(self, data):
        writes.append(Path(self))
        return real_write_bytes(self, data)

    def spy_write_text(self, data, *args, **kwargs):
        writes.append(Path(self))
        return real_write_text(self, data, *args, **kwargs)

    real_mkdir = Path.mkdir

    def spy_mkdir(self, *args, **kwargs):
        writes.append(Path(self))
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy_open)
    monkeypatch.setattr(os, "replace", spy_replace)
    monkeypatch.setattr(Path, "write_bytes", spy_write_bytes)
    monkeypatch.setattr(Path, "write_text", spy_write_text)
    monkeypatch.setattr(Path, "mkdir", spy_mkdir)

    monkeypatch.chdir(elsewhere)
    clock = iter(range(10_000))
    net = Net(
        cache / "listings",
        transport=transport,
        limiter=RateLimiter(clock=lambda: float(next(clock)), sleep=lambda s: None),
        log=lambda message: None,
    )
    dataset = qa.fetch_financebench(cache, net=net, max_documents=2)
    assert dataset.name == "financebench"
    assert sorted({q.doc_id for q in dataset.questions}) == ["DOC_0", "DOC_1"]  # capped at 2
    assert (cache / "financebench" / "pdfs" / "DOC_1.pdf").read_bytes().startswith(b"%PDF-")

    assert writes, "the fetch wrote nothing"
    root = cache.resolve()
    outside = [p for p in writes if not p.resolve().is_relative_to(root)]
    assert outside == []
    assert list(elsewhere.iterdir()) == []

    # The guard refuses a path outside the cache, whatever asks for it.
    with pytest.raises(ValueError, match="outside"):
        qa.in_cache(tmp_path / "gold.jsonl", cache)
    # The default cache is the explicit nosync path, never the .cache symlink.
    assert qa.CACHE.parts[-2:] == (".cache.nosync", "evidence_qa")


# --- the extraction harness ----------------------------------------------------------------------


def test_extraction_scoring_exact_or_numeric_tolerance():
    assert extraction.score("$2,468.00", "2468") is True  # numeric, same value
    assert extraction.score("The answer is 2,488 million.", "$2,468 million") is True  # 0.8%
    assert extraction.score("2,505", "$2,468") is False  # 1.5% off
    assert extraction.score("(12.5)", "-12.5") is True  # accounting negative
    assert extraction.score("  Yes. ", "yes") is True  # exact match, normalised
    assert extraction.score("No", "yes") is False
    assert extraction.score("", "0") is False


def test_extraction_scoring_ignores_years_and_keys_on_the_main_number():
    gold = "The FY2031 fixed asset turnover ratio for Example Games is 31.47."
    assert extraction.score("2031", gold) is False  # the year is not the answer
    assert extraction.score("FY2031 was strong", gold) is False
    assert extraction.score("31.5", gold) is True  # 0.1% off
    assert extraction.score("In fiscal 2031 it was 31.47", gold) is True
    # Several non-year numbers: the prediction must match the last one.
    several = "Revenue rose from $1,200 million in 2030 to $2,468 million in 2031."
    assert extraction.score("1,200", several) is False
    assert extraction.score("$2,468 million", several) is True
    # A list answer: any of its items.
    assert extraction.score("12", ["10", "12"]) is True
    assert extraction.score("11", ["10", "12"]) is False
    assert extraction.main_numbers("FY 2021 and fiscal year 2020: 3.5% in 2050") == [3.5]


def test_extraction_refuses_jev_and_typesafe(monkeypatch, capsys):
    for name in ("DOCROOTER_EXTRACT_BASE_URL", "DOCROOTER_EXTRACT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("DOCROOTER_EXTRACT_API_KEY", raising=False)
    assert extraction.main([]) == 0
    assert "not configured" in capsys.readouterr().out

    monkeypatch.setenv("DOCROOTER_EXTRACT_BASE_URL", "https://api.TypeSafe.ai/v1")
    monkeypatch.setenv("DOCROOTER_EXTRACT_MODEL", "some-model")
    monkeypatch.setenv("DOCROOTER_EXTRACT_API_KEY", "k")
    with pytest.raises(SystemExit, match="typesafe"):
        extraction.endpoint_from_env()
    monkeypatch.setenv("DOCROOTER_EXTRACT_BASE_URL", "http://localhost:8080/v1")
    monkeypatch.setenv("DOCROOTER_EXTRACT_MODEL", "jev-1")
    with pytest.raises(SystemExit, match="Jev"):
        extraction.endpoint_from_env()
    monkeypatch.setenv("DOCROOTER_EXTRACT_MODEL", "qwen")
    monkeypatch.setenv("TYPESAFE_API_KEY", "never-read")
    endpoint = extraction.endpoint_from_env()
    assert endpoint is not None and endpoint.model == "qwen"
    assert "never-read" not in repr(endpoint)


# --- the runtime path (G3) -----------------------------------------------------------------------


def _runtime_labels(doc, wrong=("e2",), missing=("e4",)):
    """Teacher-full labels of the fake document: ``wrong`` labelled for another text, ``missing``
    unlabelled, the rest by keyword."""
    out = {}
    for e in doc.elements:
        if e.kind == "heading" or e.id in missing:
            continue
        leaf = "annual_report.risk_factors" if "risk" in e.text.lower() else "annual_report.other"
        sha = "0" * 40 if e.id in wrong else text_sha1(e.text)
        out[e.id] = {sha: leaf}
    return out


def test_runtime_units_match_labels_by_id_and_hash():
    analysis, ctx, selector = _fake_document()
    labels = _runtime_labels(analysis.document)
    units, counts = bench.runtime_units(
        analysis, ctx, labels, [RISK, LEGAL], selector, doc_id="d", family="annual_report"
    )
    # e1 and e3 match; e2 is labelled for another text and e4 not at all: both excluded, counted
    assert counts == {"content": 4, "with_text": 4, "matched": 2, "unmatched": 2}
    assert [u["field_id"] for u in units] == [RISK.id]  # LEGAL: no relevant matched element
    unit = units[0]
    assert unit["n_relevant"] == 1 and unit["fit_mismatch"] == []
    assert unit["doc_type_mismatch"] is False
    at_90 = unit["by_c"]["0.90"]
    assert at_90["recall"] == 1.0 and at_90["guarantee"] is True

    # a runtime reading other than the fit's (here the balanced layout) is an estimate
    balanced = analysis.model_copy(update={"config": RouterConfig(profile="balanced")})
    balanced_ctx = prepare(balanced, tagger=FakeTagger(), encoder=FakeEmbedder())
    units, _ = bench.runtime_units(
        balanced, balanced_ctx, labels, [RISK], selector, doc_id="d", family="annual_report"
    )
    assert units[0]["fit_mismatch"] == ["layout"]
    assert units[0]["by_c"]["0.90"]["guarantee"] is False


def test_run_gold_measures_the_runtime_path_beside_the_fit_reading():
    from docrooter.corpus.store import ManifestRecord
    from docrooter.errors import DocrooterError

    analysis, ctx, selector = _fake_document()
    labels = {d: _runtime_labels(analysis.document, wrong=(), missing=()) for d in ("a", "b")}
    runtime_analysis = analysis.model_copy(update={"config": RouterConfig(profile="balanced")})
    runtime_ctx = prepare(runtime_analysis, tagger=FakeTagger(), encoder=FakeEmbedder())

    def record(doc_id):
        return ManifestRecord(id=doc_id, sha256="0" * 64, family_hint="annual_report", split="gold")

    def read_runtime(rec):
        if rec.id == "b":
            raise DocrooterError("cannot open PDF")
        return runtime_analysis, runtime_ctx

    run = bench.run_gold(
        [record("a"), record("b")],
        lambda rec: (analysis, ctx),
        labels,
        {"annual_report": [RISK, LEGAL]},
        {},
        selector,
        read_runtime=read_runtime,
    )
    runtime = run["runtime"]
    assert runtime["documents"]["read"] == 1
    assert [f["doc_id"] for f in runtime["documents"]["read_failed"]] == ["b"]
    assert [u["doc_id"] for u in runtime["units"]] == ["a"]
    assert runtime["labels"]["matched"] == 4 and runtime["labels"]["unmatched"] == 0
    assert runtime["labels"]["match_rate"] == pytest.approx(
        runtime["labels"]["matched"] / runtime["labels"]["with_text"]
    )
    assert runtime["fit_mismatch"] == {"layout": 1}


def test_markdown_shows_the_runtime_row_and_the_dials():
    units = [unit("annual_report", "a", (0.9, 0.95), (0.2, 0.3))]
    agg = bench.aggregate(units, (0.8, 0.9))
    verdict = bench.program_target(agg)
    runtime_units = [unit("annual_report", "a", (0.7, 0.8), (0.2, 0.3), guarantee=False)]
    runtime_agg = bench.aggregate(runtime_units, (0.8, 0.9))
    documents = {"gold_split": 1, "read": 1, "evaluated": 1, "read_without_units": []}
    results = {
        "date": "2026-10-19",
        "disclosure": bench.DISCLOSURE,
        "provenance": {
            "commit": "abc",
            "selector_meta": {"dials": {"annual_report": {"n": 99, "n_docs": 40}}},
            "overrides": ["--embedder x differs from the selector's fit (y)"],
        },
        "documents": documents,
        "gold_elements": {},
        "label_checks": {"totals": {}},
        "main": agg,
        "program_target": verdict,
        "gold_subset": agg,
        "disagreement": bench.teacher_gold_disagreement([], {}),
        "baselines": bench.aggregate_baselines([], (0.9,)),
        "curve": {},
        "dial_tokens_to_95": {},
        "runtime": {
            "main": runtime_agg,
            "program_target": bench.program_target(runtime_agg),
            "labels": {"content": 10, "with_text": 9, "matched": 8, "unmatched": 1,
                       "match_rate": 8 / 9},
            "documents": {"read": 1, "evaluated": 1, "read_failed": []},
            "fit_mismatch": {"layout": 1},
            "doc_type_mismatch": 0,
        },
    }  # fmt: skip
    text = bench.markdown(results)
    assert "### Fit-identical reading and runtime path" in text
    assert "| 0.90 | 0.950 | 0.300 | 1.00 | 0.800 | 0.300 | 0.00 |" in text
    assert "label match rate 0.889 (8 of 9" in text and "layout 1" in text
    assert "### Dials" in text and "| annual_report | 99 | 40 | 0.010 |" in text
    assert "--embedder x differs" in text


def test_overrides_of_the_fit_are_named():
    _, _, selector = _fake_document()
    assert bench.provenance_overrides(selector, tagger=TAGGER_FP, embedder=EMBEDDER_SPEC) == []
    both = bench.provenance_overrides(selector, tagger="fp-other", embedder="embedding:other")
    assert len(both) == 2 and "--tagger" in both[0] and "--embedder" in both[1]
    assert "guarantee" in both[0]
    unknown = bench.provenance_overrides(selector, tagger=None, embedder=EMBEDDER_SPEC)
    assert len(unknown) == 1 and "--tagger" in unknown[0]
    bare = Selector(
        field=selector.field,
        field_fallback=selector.field_fallback,
        query=selector.query,
        query_fallback=selector.query_fallback,
        dials=selector.dials,
    )
    none = bench.provenance_overrides(bare, tagger=TAGGER_FP, embedder=EMBEDDER_SPEC)
    assert len(none) == 1 and "provenance" in none[0]


def test_paths_in_results_are_relative_to_the_repository():
    inside = str(bench.REPO_ROOT / "benchmarks" / ".cache.nosync" / "x.pdf")
    assert bench.relative(f"no PDF for d at {inside}") == (
        "no PDF for d at benchmarks/.cache.nosync/x.pdf"
    )
    # a path outside the repository never reaches the results: only its last part does
    assert bench.relative("/elsewhere/bundles/tagger-a") == "<external>/tagger-a"
    assert bench.relative("DocrooterError: no PDF for d at /home/u/pdfs/d.pdf; pass --pdf-dir") == (
        "DocrooterError: no PDF for d at <external>/d.pdf; pass --pdf-dir"
    )
    assert (
        bench.relative("slack 1/(n+1), and/or a ratio 3/4") == "slack 1/(n+1), and/or a ratio 3/4"
    )
