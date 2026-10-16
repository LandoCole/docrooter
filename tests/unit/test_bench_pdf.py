"""Real-PDF document-type benchmark helpers: metrics, chunk selection, leave-one-out splits and
the ``.env`` loader. No PDFs, no network, no torch."""

import pytest

from benchmarks import jev_chunks, pdf_bench
from docrooter.taxonomies import builtin_taxonomy

TAXONOMY = builtin_taxonomy("doc_types")


def _row(doc_id, gold, pred, **extra):
    return {"doc_id": doc_id, "gold": gold, "pred": pred, **extra}


# --- metrics -----------------------------------------------------------------------------


def test_metrics_on_synthetic_rows():
    rows = [
        _row(
            "a", "financial.annual_report", "financial.annual_report", parse_s=1.0, classify_s=0.5
        ),
        _row(
            "b",
            "financial.annual_report",
            "financial.quarterly_report",
            parse_s=3.0,
            classify_s=0.5,
        ),
        _row("c", "legal.contract", "legal.contract", parse_s=2.0, classify_s=1.0, tokens=100),
        _row(
            "d", "legal.contract", "research.white_paper", parse_s=2.0, classify_s=1.0, tokens=300
        ),
    ]
    m = pdf_bench.system_metrics(rows, TAXONOMY)
    assert m["n"] == 4
    assert m["leaf_accuracy"] == pytest.approx(0.5)
    # b stays in "financial"; d leaves "legal".
    assert m["level1_accuracy"] == pytest.approx(0.75)
    # Gold classes: annual_report (P 1, R 0.5) and contract (P 1, R 0.5).
    assert m["macro_p"] == pytest.approx(1.0)
    assert m["macro_r"] == pytest.approx(0.5)
    assert m["macro_f1"] == pytest.approx(2 / 3)
    assert m["avg_parse_s"] == pytest.approx(2.0)
    assert m["avg_classify_s"] == pytest.approx(0.75)
    assert m["avg_total_s"] == pytest.approx(2.75)
    assert m["jev_input_tokens"] == 400
    assert m["jev_cost_usd"] == pytest.approx(400 * 0.042 / 1e6)
    per_class = m["per_class"]
    assert set(per_class) == {"financial.annual_report", "legal.contract"}
    assert per_class["financial.annual_report"] == {
        "p": 1.0,
        "r": 0.5,
        "f1": pytest.approx(2 / 3),
        "n": 2,
    }
    assert m["confusion"]["financial.annual_report"] == {
        "financial.annual_report": 1,
        "financial.quarterly_report": 1,
    }
    lo, hi = m["leaf_accuracy_ci95"]
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0


def test_metrics_without_tokens_report_none_and_missing_prediction_is_wrong():
    rows = [
        _row("a", "legal.contract", None, parse_s=1.0, classify_s=0.0),
        _row("b", "legal.contract", "legal.contract", parse_s=1.0, classify_s=0.0),
    ]
    m = pdf_bench.system_metrics(rows, TAXONOMY)
    assert m["leaf_accuracy"] == pytest.approx(0.5)
    assert m["level1_accuracy"] == pytest.approx(0.5)
    assert m["jev_input_tokens"] is None
    assert m["jev_cost_usd"] is None
    assert m["confusion"]["legal.contract"] == {"(none)": 1, "legal.contract": 1}


def test_bootstrap_ci_is_deterministic_and_degenerate_on_constant_input():
    hits = [True, False, True, True, False, True, False, True]
    assert pdf_bench.bootstrap_ci(hits) == pdf_bench.bootstrap_ci(hits)
    assert pdf_bench.bootstrap_ci([True] * 5) == (1.0, 1.0)
    assert pdf_bench.bootstrap_ci([]) == (0.0, 0.0)


# --- chunk selection ---------------------------------------------------------------------


def test_chunk_indices_evenly_spaced_and_bounded():
    assert jev_chunks.chunk_indices(3, 8) == [0, 1, 2]
    assert jev_chunks.chunk_indices(0, 8) == []
    assert jev_chunks.chunk_indices(20, 1) == [0]
    picked = jev_chunks.chunk_indices(29, 8)
    assert len(picked) == 8
    assert picked[0] == 0 and picked[-1] == 28
    assert picked == sorted(set(picked))
    gaps = [b - a for a, b in zip(picked, picked[1:], strict=False)]
    assert max(gaps) - min(gaps) <= 1
    assert jev_chunks.chunk_indices(29, 8) == picked


def test_select_chunks_sizes_and_limit():
    text = " ".join(f"w{i}" for i in range(10_000))
    chunks = jev_chunks.select_chunks(text, chunk_tokens=130, max_chunks=8)
    # 130 tokens = 100 words at 1.3 tokens per word; 100 chunks exist, 8 are kept.
    assert len(chunks) == 8
    assert all(len(c.split()) == 100 for c in chunks)
    assert chunks[0].split()[0] == "w0"
    assert chunks[-1].split()[-1] == "w9999"
    assert chunks == jev_chunks.select_chunks(text, chunk_tokens=130, max_chunks=8)


def test_select_chunks_short_and_empty_text():
    assert jev_chunks.select_chunks("one two\nthree", chunk_tokens=1500, max_chunks=8) == [
        "one two\nthree"
    ]
    assert jev_chunks.select_chunks("  \n ", chunk_tokens=1500, max_chunks=8) == []


def test_estimate_tokens_is_words_times_1_3():
    assert jev_chunks.estimate_tokens("a b c d e f g h i j") == 13


def test_mean_probs_averages_rows():
    probs = jev_chunks.mean_probs([{"x": 1.0, "y": 0.0}, {"x": 0.5, "y": 0.5}])
    assert probs == pytest.approx({"x": 0.75, "y": 0.25})
    assert jev_chunks.mean_probs([]) == {}


# --- leave-one-out -----------------------------------------------------------------------


def test_loo_never_trains_on_held_out_doc():
    features = [[float(i)] for i in range(6)]
    labels = ["a", "a", "b", "b", "c", "c"]
    seen = []

    def fit_predict(train_x, train_y, test_x):
        seen.append(([x[0] for x in train_x], list(train_y), test_x[0]))
        return "a", {"a": 1.0}

    preds = pdf_bench.loo_predict(features, labels, fit_predict)
    assert len(preds) == 6
    assert [p[0] for p in preds] == ["a"] * 6
    for i, (train_x, train_y, test_x) in enumerate(seen):
        assert test_x == float(i)
        assert float(i) not in train_x
        assert len(train_x) == len(train_y) == 5


# --- .env loading ------------------------------------------------------------------------


def test_load_env_sets_missing_keys_silently(tmp_path, capsys):
    secret = "sk-test-0123456789"
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"# comment\nTYPESAFE_API_KEY='{secret}'\nexport OTHER=\"x y\"\nKEEP=new\n\nBROKEN LINE\n",
        encoding="utf-8",
    )
    environ = {"KEEP": "old"}
    loaded = pdf_bench.load_env(env_file, environ)
    assert environ["TYPESAFE_API_KEY"] == secret
    assert environ["OTHER"] == "x y"
    assert environ["KEEP"] == "old"
    assert sorted(loaded) == ["OTHER", "TYPESAFE_API_KEY"]
    out, err = capsys.readouterr()
    assert secret not in out + err
    assert out == err == ""


def test_load_env_missing_file_is_a_no_op(tmp_path):
    environ: dict[str, str] = {}
    assert pdf_bench.load_env(tmp_path / "nope.env", environ) == []
    assert environ == {}
