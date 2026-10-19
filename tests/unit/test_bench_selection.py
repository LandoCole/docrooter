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
from docrooter.classify.results import DocTypeResult, PathResult
from docrooter.corpus.store import text_sha1
from docrooter.elements.models import Box, Element, TagResult
from docrooter.rooter.analysis import Analysis
from docrooter.select import Field
from docrooter.select.crc import DialTable
from docrooter.select.relevance import FEATURE_NAMES, Combiner
from docrooter.select.selection import Selection, Selector, prepare


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

    def unit(family, field_id, recalls, shares, guarantee=True, capped=False):
        return {
            "doc_id": f"d-{field_id}",
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


class FakeTagger:
    def tag(self, elements, family):
        leaves = [
            f"{family}.{'risk_factors' if 'risk' in e.text.lower() else 'other'}" for e in elements
        ]
        return [
            {"topic": TagResult(labels=[leaf], probs={leaf: 1.0}, confidence=1.0, source="fake")}
            for leaf in leaves
        ]


class FakeEmbedder:
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
        config=RouterConfig(profile="fast"),
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
    rows, tally = qa.evaluate_document(
        analysis, ctx, [question], [RISK, LEGAL], selector, FakeEmbedder(), {1},
        other_base_shift=-1,
    )  # fmt: skip
    assert tally["aligned"] == 1 and tally["aligned_if_other_base"] == 0
    assert rows[0]["field_id"] == RISK.id and rows[0]["n_relevant"] == 1
    assert set(rows[0]["query"]) == {bench.c_key(c) for c in qa.QUERY_GRID}
    assert all(r["guarantee"] is False for r in rows[0]["query"].values())
    assert "Which" not in json.dumps(rows)  # ids and numbers only
    off_page = qa.Question("q2", "d", "q", "x", (qa.Evidence(7, "anything"),))
    _, tally = qa.evaluate_document(
        analysis, ctx, [off_page], [RISK], selector, FakeEmbedder(), {1}, other_base_shift=None
    )
    assert tally["pages_not_read"] == 1


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
        "Total revenue was $1,577 million in 2018, up 4%.   Net income was $300 million."
    )
    elements = [
        qa.PageText("e1", frozenset({5}), "Total revenue was\n$1,577 million in 2018, up 4%."),
        qa.PageText("e2", frozenset({5}), "The weather in Minnesota was cold that winter."),
        qa.PageText("e3", frozenset({6}), "Total revenue was $1,577 million in 2018, up 4%."),
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
        "doc_name": "ACME_2018_10K",
        "question": "What was revenue?",
        "answer": "$1,577 million",
        "evidence": [{"evidence_text": evidence_text, "evidence_page_num": 4}],
    }
    question = qa.parse_financebench([json.dumps(row)])[0]
    assert question.id == "financebench_id_00001" and question.doc_id == "ACME_2018_10K"
    assert question.evidence == (qa.Evidence(5, evidence_text),)
    assert qa.parse_financebench([json.dumps(row)], page_base=1)[0].evidence[0].page == 4

    # The closest field is the one most similar to the question.
    fields = ["annual_report.a", "annual_report.b", "annual_report.c"]
    assert qa.closest(fields, np.array([0.1, 0.7, 0.3])) == "annual_report.b"


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
    assert extraction.score("$1,577.00", "1577") is True  # numeric, same value
    assert extraction.score("The answer is 1,590 million.", "$1,577 million") is True  # 0.8%
    assert extraction.score("1,600", "$1,577") is False  # 1.5% off
    assert extraction.score("(12.5)", "-12.5") is True  # accounting negative
    assert extraction.score("  Yes. ", "yes") is True  # exact match, normalised
    assert extraction.score("No", "yes") is False
    assert extraction.score("", "0") is False


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
