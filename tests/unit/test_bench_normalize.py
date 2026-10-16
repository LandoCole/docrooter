"""Section-normalisation benchmark helpers: gold header parsing, gold-to-section matching and the
metric math. No PDFs, no network, no torch."""

import json

import pytest

from benchmarks import normalize_bench as nb
from docrooter.document.models import Section


def _section(sid, title, page, level=1):
    return Section(id=sid, title=title, level=level, text="", page_start=page)


def _gold(title, page, canonical="method", doc_id="d", level=1):
    return {
        "doc_id": doc_id,
        "section_title": title,
        "page": page,
        "level": level,
        "canonical": canonical,
    }


# --- header parsing ----------------------------------------------------------------------


def _write(tmp_path, lines):
    path = tmp_path / "gold.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n\n", encoding="utf-8")
    return path


def test_read_gold_returns_header_and_rows(tmp_path):
    meta = {"kind": "silver", "src_commit": "abc1234", "labelled_at": "2026-10-16T14:41:44-04:00"}
    path = _write(tmp_path, [{"_meta": meta}, _gold("Intro", 1), _gold("Refs", 9, None)])
    header, rows = nb.read_gold(path)
    assert header == meta
    assert [r["section_title"] for r in rows] == ["Intro", "Refs"]
    assert rows[1]["canonical"] is None


def test_read_gold_requires_a_header_line(tmp_path):
    path = _write(tmp_path, [_gold("Intro", 1)])
    with pytest.raises(ValueError, match="header"):
        nb.read_gold(path)


def test_read_gold_rejects_rows_missing_fields(tmp_path):
    bad = _gold("Intro", 1)
    del bad["canonical"]
    path = _write(tmp_path, [{"_meta": {}}, bad])
    with pytest.raises(ValueError, match="line 2.*canonical"):
        nb.read_gold(path)


def test_the_committed_gold_file_parses():
    header, rows = nb.read_gold(nb.GOLD_PATH)
    assert header["kind"] == "silver"
    assert header["src_commit"] and header["labelled_at"]
    assert set(header["documents"]) == {r["doc_id"] for r in rows}


# --- matching ----------------------------------------------------------------------------


def test_title_key_ignores_case_and_whitespace():
    assert nb.title_key("  1  Introduction\n") == nb.title_key("1 introduction")
    assert nb.title_key("1 Introduction") != nb.title_key("Introduction")


def test_match_pairs_by_title_and_page():
    sections = [_section("s1", "Abstract", 1), _section("s2", "1 Introduction", 2, level=1)]
    gold = [_gold("1  INTRODUCTION", 2, "introduction"), _gold("Abstract", 1, "abstract")]
    m = nb.match_gold(gold, sections)
    assert [(g["canonical"], sid) for g, sid in m.pairs] == [
        ("introduction", "s2"),
        ("abstract", "s1"),
    ]
    assert m.unmatched_gold == [] and m.unlabelled == []


def test_match_reports_page_or_title_changes_instead_of_crashing():
    sections = [_section("s1", "Results", 8), _section("s2", "New heading", 3)]
    gold = [_gold("Results", 7), _gold("Gone", 1)]
    m = nb.match_gold(gold, sections)
    assert m.pairs == []
    assert [g["section_title"] for g in m.unmatched_gold] == ["Results", "Gone"]
    assert m.unlabelled == ["s1", "s2"]


def test_repeated_title_on_the_same_page_matches_in_order():
    sections = [_section("a", "Railroad", 67), _section("b", "Railroad", 67)]
    gold = [_gold("Railroad", 67, "x"), _gold("Railroad", 67, "y"), _gold("Railroad", 67, "z")]
    m = nb.match_gold(gold, sections)
    assert [(g["canonical"], sid) for g, sid in m.pairs] == [("x", "a"), ("y", "b")]
    assert [g["canonical"] for g in m.unmatched_gold] == ["z"]


# --- metrics -----------------------------------------------------------------------------


def test_score_excludes_junk_and_macro_averages_over_gold_classes():
    rows = [
        {"gold": "a", "pred": "a"},
        {"gold": "a", "pred": "b"},
        {"gold": "b", "pred": "b"},
        {"gold": "c", "pred": "a"},
        {"gold": None, "pred": "c"},  # junk: not scored, counted
    ]
    s = nb.score(rows)
    assert s["n_scored"] == 4 and s["n_junk"] == 1 and s["n_correct"] == 2
    assert s["accuracy"] == pytest.approx(0.5)
    # a: P 1/2, R 1/2 -> F1 .5; b: P 1/2, R 1 -> 2/3; c: no hits -> 0.
    assert s["macro_f1"] == pytest.approx((0.5 + 2 / 3 + 0.0) / 3)


def test_score_of_no_scored_rows_is_none():
    s = nb.score([{"gold": None, "pred": "a"}])
    assert s["n_scored"] == 0 and s["accuracy"] is None and s["macro_f1"] is None


def test_pooled_classes_are_qualified_by_taxonomy():
    rows = [
        {"gold": "other", "pred": "other", "taxonomy": "section_types/annual_report"},
        {"gold": "other", "pred": "x", "taxonomy": "section_types/generic"},
    ]
    s = nb.score(rows, qualify=True)
    # Two distinct classes: F1 1.0 and 0.0.
    assert s["macro_f1"] == pytest.approx(0.5)


def test_jev_cost_uses_input_tokens_only():
    assert nb.jev_cost(1_000_000) == pytest.approx(nb.pdf_bench.JEV_USD_PER_M_INPUT)
    assert nb.jev_cost(None) is None


def test_a_prediction_from_another_taxonomy_is_wrong():
    rows = [
        {"gold": "other", "pred": "other", "taxonomy": "t/a", "pred_taxonomy": "t/generic"},
        {"gold": "other", "pred": "other", "taxonomy": "t/a", "pred_taxonomy": "t/a"},
    ]
    for qualify in (False, True):
        s = nb.score(rows, qualify=qualify)
        assert s["n_correct"] == 1
        # One gold class: P 1/1 (the foreign "other" is not a guess of it), R 1/2.
        assert s["macro_f1"] == pytest.approx(2 / 3)


# --- configurations ----------------------------------------------------------------------


def test_baseline_config_is_the_p10_setting_and_final_is_the_package_default():
    from docrooter.classify.normalize import NORMALIZE_VIEW
    from docrooter.views import SectionView

    base = nb.CONFIGS[nb.BASELINE_CONFIG]
    assert base == nb.Config(SectionView(fields=("title", "lead"), max_tokens=96), False)
    final = nb.CONFIGS[nb.FINAL_CONFIG]
    assert final.view == NORMALIZE_VIEW and final.synonyms


def test_without_synonyms_strips_only_the_examples():
    from docrooter.taxonomies import builtin_taxonomy

    name = "section_types/research_paper"
    builtin, stripped = builtin_taxonomy(name), nb.without_synonyms(name)
    assert any(n.examples for n in builtin.nodes.values())
    assert all(not n.examples for n in stripped.nodes.values())
    assert [(n.id, n.label, n.description) for n in stripped.nodes.values()] == [
        (n.id, n.label, n.description) for n in builtin.nodes.values()
    ]


def _run(accuracy, n_correct, preds, **extra):
    rows = [{"gold": "a", "pred": p} for p in preds] + [{"gold": None, "pred": "junk"}] * 3
    overall = {"accuracy": accuracy, "macro_f1": 0.5, "n_correct": n_correct, "n_scored": 4}
    return {"overall": overall, "s_per_section": 0.01, "rows": rows, **extra}


def test_summarize_counts_top_predictions_over_scored_rows_only():
    s = nb.summarize(_run(0.5, 2, ["a", "b", "b", "c"], tokens=10, cost_usd=0.1))
    assert s["top_preds"] == [("b", 2), ("a", 1), ("c", 1)]
    assert (s["accuracy"], s["n_correct"], s["tokens"], s["cost_usd"]) == (0.5, 2, 10, 0.1)
    assert "rows" not in s


def test_before_after_and_variant_tables():
    configs = {
        k: {"fields": list(c.view.fields), "max_tokens": c.view.max_tokens, "synonyms": c.synonyms}
        for k, c in nb.CONFIGS.items()
    }
    before, after = _run(0.25, 1, ["a", "b", "b", "b"]), _run(0.75, 3, ["a", "a", "a", "b"])
    result = {
        "final_config": nb.FINAL_CONFIG,
        "configs": configs,
        "systems": {"ultrabert": after},
        "baseline_p10": {
            "date": "2026-10-16",
            "commit": "abc1234",
            "config": nb.BASELINE_CONFIG,
            "systems": {"ultrabert": before},
        },
        "variants": {
            nb.BASELINE_CONFIG: {"ultrabert": nb.summarize(before)},
            nb.FINAL_CONFIG: {"ultrabert": nb.summarize(after)},
        },
    }
    table = "\n".join(nb._before_after(result, ["ultrabert"]))
    assert "| `ultrabert` | 0.250 (1/4) | 0.750 (3/4) | +50.0 pts |" in table
    assert "without the taxonomies' heading synonyms" in table
    variants = "\n".join(nb._variants(result))
    assert (
        f"| `{nb.BASELINE_CONFIG}` | title + lead @ 96 | no | 0.250 (1/4) | `b` (3) |" in variants
    )
    assert f"| `{nb.FINAL_CONFIG}` (default) |" in variants
    assert "dev set" in variants
