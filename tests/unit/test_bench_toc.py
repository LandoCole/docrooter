"""TOC benchmark scoring: one-to-one greedy matching and per-document metrics. No PDFs needed."""

import json

import pytest

from benchmarks import toc as bench_toc
from benchmarks.toc import gold_from_bookmarks, match_headings, score_doc


def test_match_is_one_to_one():
    pred = [("Introduction", 1, 1), ("Introduction", 1, 1)]
    gold = [("Introduction", 1, 1)]
    assert match_headings(pred, gold) == [(0, 0)]


def test_match_respects_page_tolerance():
    gold = [("Results", 10, 1)]
    assert match_headings([("Results", 11, 1)], gold) == [(0, 0)]
    assert match_headings([("Results", 12, 1)], gold) == []
    assert match_headings([("Results", 12, 1)], gold, page_tol=2) == [(0, 0)]


def test_match_prefers_highest_ratio():
    # Both predictions clear 0.85 against the gold title; the exact one must win it.
    gold = [("Financial Statements", 5, 1)]
    pred = [("Financial Statement", 5, 1), ("Financial Statements", 5, 1)]
    assert match_headings(pred, gold) == [(1, 0)]


def test_match_greedy_leaves_loser_to_other_gold():
    gold = [("Risk Factors", 3, 1), ("Risk Factor", 3, 1)]
    pred = [("Risk Factors", 3, 1), ("Risk Factor", 3, 1)]
    assert sorted(match_headings(pred, gold)) == [(0, 0), (1, 1)]


def test_match_threshold_and_normalisation():
    gold = [("1. Business Overview.", 2, 1)]
    assert match_headings([("BUSINESS OVERVIEW", 2, 1)], gold) == [(0, 0)]
    assert match_headings([("Something else entirely", 2, 1)], gold) == []
    assert match_headings([("Business Overview", 2, 1)], gold, ratio=1.0) == [(0, 0)]


def test_match_empty_inputs():
    assert match_headings([], [("A", 1, 1)]) == []
    assert match_headings([("A", 1, 1)], []) == []


def test_score_doc_by_hand():
    gold = [("Alpha", 1, 1), ("Beta", 2, 2), ("Gamma", 3, 1), ("Delta", 4, 1)]
    # Alpha ok level, Beta level wrong, Gamma missing, Delta missing; one spurious prediction.
    pred = [("Alpha", 1, 1), ("Beta", 2, 1), ("Zebra Crossing", 9, 1)]
    out = score_doc(pred, gold)
    assert out["n_pred"] == 3 and out["n_gold"] == 4
    assert out["p"] == pytest.approx(2 / 3)
    assert out["r"] == pytest.approx(2 / 4)
    assert out["f1"] == pytest.approx(2 * (2 / 3) * 0.5 / ((2 / 3) + 0.5))
    assert out["level_acc"] == pytest.approx(0.5)
    assert out["n_match"] == 2 and out["n_level_ok"] == 1


def test_score_doc_depth_filter_applies_to_both_sides():
    gold = [("Alpha", 1, 1), ("Beta", 2, 2), ("Deep", 3, 3)]
    pred = [("Alpha", 1, 1), ("Deep", 3, 3), ("Extra", 5, 4)]
    out = score_doc(pred, gold, max_gold_depth=2)
    assert out["n_gold"] == 2 and out["n_pred"] == 1  # Deep and Extra (level > 2) dropped
    assert out["p"] == 1.0 and out["r"] == 0.5


def test_score_doc_empty_cases():
    out = score_doc([], [("A", 1, 1)])
    assert (out["p"], out["r"], out["f1"], out["level_acc"]) == (0.0, 0.0, 0.0, None)
    out = score_doc([("A", 1, 1)], [])
    assert out["n_gold"] == 0 and out["r"] == 0.0 and out["f1"] == 0.0


def test_gold_from_bookmarks_shifts_levels_and_drops_unpaged():
    marks = [(2, "A", 1), (3, "B", 2), (2, "C", 0), (4, "D", -1)]
    assert gold_from_bookmarks(marks) == [("A", 1, 1), ("B", 2, 2)]
    assert gold_from_bookmarks([]) == []


def test_aggregate_micro_and_macro():
    docs = [
        {"n_pred": 2, "n_gold": 4, "n_match": 2, "n_level_ok": 1, "f1": 0.667, "seconds": 1.0},
        {"n_pred": 8, "n_gold": 4, "n_match": 4, "n_level_ok": 4, "f1": 0.667, "seconds": 3.0},
    ]
    agg = bench_toc.aggregate(docs)
    assert agg["p"] == pytest.approx(6 / 10)
    assert agg["r"] == pytest.approx(6 / 8)
    assert agg["level_acc"] == pytest.approx(5 / 6)
    assert agg["macro_f1"] == pytest.approx(0.667)
    assert agg["seconds_per_doc"] == pytest.approx(2.0)
    assert agg["n_docs"] == 2
    json.dumps(agg)


def _result():
    def doc(i, f1):
        score = {"p": f1, "r": f1, "f1": f1, "level_acc": 1.0, "n_pred": 2, "n_gold": 2,
                 "n_match": 1, "n_level_ok": 1}  # fmt: skip
        return {"id": f"doc_{i}", "title": f"T{i}", "pages": i, "seconds": 1.0,
                "all": score, "depth2": score}  # fmt: skip

    def engine(name, ids, note=None):
        rows = [doc(i, 0.5) for i in ids]
        flat = [r["all"] | {"seconds": r["seconds"]} for r in rows]
        return {"engine": name, "n_docs": len(rows), "doc_ids": [r["id"] for r in rows],
                "all": bench_toc.aggregate(flat), "depth2": bench_toc.aggregate(flat),
                "docs": rows, "note": note}  # fmt: skip

    return {
        "env": {"python": "3.12", "machine": "arm64", "cpu_count": 8, "loadavg": [1.0, 2.0, 3.0],
                "versions": {}},
        "min_bookmarks": 5,
        "match": {"ratio": 0.85, "page_tol": 1},
        "engines": {
            "pymupdf": engine("pymupdf", [1, 2, 3]),
            "pymupdf4llm": engine("pymupdf4llm", [1], note="limited to the 1 smallest"),
        },
    }  # fmt: skip


def test_render_tables_has_aggregate_subset_and_per_doc():
    text = bench_toc.render_tables(_result())
    assert "| pymupdf | 3 |" in text
    assert "| pymupdf4llm* | 1 |" in text
    assert "limited to the 1 smallest" in text
    assert "Same 1 documents" in text
    assert "| doc_3 | T3 |" in text


def test_update_docs_creates_then_replaces_only_its_block(tmp_path):
    path = tmp_path / "benchmarks-pdf.md"
    bench_toc.update_docs(path, "first tables")
    text = path.read_text()
    assert text.startswith("# PDF benchmarks\n\n## PDF TOC extraction\n")
    assert "first tables" in text

    path.write_text(text + "\nhand-written analysis\n\n## Other section\n\nkeep me\n")
    bench_toc.update_docs(path, "second tables")
    out = path.read_text()
    assert "second tables" in out and "first tables" not in out
    assert "hand-written analysis" in out and "keep me" in out
    assert out.count("## PDF TOC extraction") == 1


def test_update_docs_appends_section_to_foreign_file(tmp_path):
    path = tmp_path / "benchmarks-pdf.md"
    path.write_text("# PDF benchmarks\n\n## Another\n\ntext\n")
    bench_toc.update_docs(path, "tables")
    out = path.read_text()
    assert "## Another\n\ntext" in out and out.count("## PDF TOC extraction") == 1


def test_compare_modes_reports_micro_f1_and_per_doc_deltas():
    def entry(f1s):
        docs = [
            {"id": f"d{k}", "depth2": {"f1": f, "n_match": 1, "n_pred": 2, "n_gold": 2}}
            for k, f in enumerate(f1s)
        ]
        return {"depth2": {"f1": sum(f1s) / len(f1s)}, "all": {"seconds_per_doc": 1.0},
                "docs": docs}  # fmt: skip

    engines = {"pymupdf": entry([0.5, 0.8]), "pymupdf+elements": entry([0.6, 0.8])}
    out = bench_toc.compare_modes(engines)
    assert list(out) == ["pymupdf"]
    row = out["pymupdf"]
    assert row["fast_depth2_micro_f1"] == pytest.approx(0.65)
    assert row["elements_depth2_micro_f1"] == pytest.approx(0.7)
    assert row["delta"] == pytest.approx(0.05)
    assert [(d["id"], d["delta"]) for d in row["per_doc_deltas"]] == [("d0", 0.1)]
