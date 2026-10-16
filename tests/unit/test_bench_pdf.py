"""Real-PDF document-type benchmark helpers: metrics, chunk selection, leave-one-out splits and
the ``.env`` loader. No PDFs, no network, no torch."""

import numpy as np
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


# --- P9b: inner-CV tuning, few-shot heads and mixtures ----------------------------------


def _unit_vec(*xs):
    v = np.asarray(xs, dtype=float)
    return v / np.linalg.norm(v)


def test_inner_loo_never_sees_the_outer_held_out_doc():
    features = [[float(i)] for i in range(6)]
    labels = ["a", "a", "a", "b", "b", "b"]
    calls = []  # (value, train ids, test id) of every fit, inner and final

    def make(value):
        def fit_predict(train_x, train_y, test_x):
            calls.append((value, [x[0] for x in train_x], test_x[0]))
            return "a", {"a": 1.0}

        return fit_predict

    tuned = pdf_bench.InnerLoo(make, [1, 2])
    for held_out in range(6):
        calls.clear()
        train = [x for j, x in enumerate(features) if j != held_out]
        train_y = [y for j, y in enumerate(labels) if j != held_out]
        tuned(train, train_y, features[held_out])
        inner, final = calls[:-1], calls[-1]
        # 2 values x 5 inner folds, each training on 4 of the 5 outer-training docs.
        assert len(inner) == 10
        for _, train_ids, test_id in inner:
            assert float(held_out) not in train_ids
            assert test_id != float(held_out)
            assert len(train_ids) == 4 and test_id not in train_ids
        assert final[2] == float(held_out)
        assert float(held_out) not in final[1]
    assert len(tuned.chosen) == 6


def test_inner_loo_picks_best_accuracy_then_lower_log_loss_then_first():
    labels = ["a", "a", "b", "b"]
    features = [[0.0], [0.0], [1.0], [1.0]]

    def make(value):
        def fit_predict(train_x, train_y, test_x):
            if value == "always-a":
                return "a", {"a": 0.9, "b": 0.1}
            if value == "right-unsure":
                guess = "a" if test_x[0] == 0.0 else "b"
                return guess, {guess: 0.6, ("b" if guess == "a" else "a"): 0.4}
            guess = "a" if test_x[0] == 0.0 else "b"  # "right-sure"
            return guess, {guess: 0.99, ("b" if guess == "a" else "a"): 0.01}

        return fit_predict

    tuned = pdf_bench.InnerLoo(make, ["always-a", "right-unsure", "right-sure"])
    assert tuned.select(features, labels) == "right-sure"
    first = pdf_bench.InnerLoo(make, ["right-sure", "right-sure-copy"])
    first.make = lambda v: make("right-sure")
    assert first.select(features, labels) == "right-sure"
    with pytest.raises(ValueError):
        pdf_bench.InnerLoo(make, [])


def test_ensemble_mixes_two_zero_shot_distributions():
    doc = pdf_bench.LooDoc(vec=[1.0], zero_shot={"a": 0.8, "b": 0.2}, other={"a": 0.2, "b": 0.8})
    label, probs = pdf_bench.ensemble(0.25)([], [], doc)
    assert probs == pytest.approx({"a": 0.65, "b": 0.35})
    assert label == "a"
    assert pdf_bench.ensemble(0.0)([], [], doc)[1] == pytest.approx(doc.zero_shot)
    assert pdf_bench.ensemble(1.0)([], [], doc)[1] == pytest.approx(doc.other)
    assert pdf_bench.ensemble(0.75)([], [], doc)[0] == "b"


def test_prior_centroid_mix_endpoints_and_unseen_classes():
    train = [
        pdf_bench.LooDoc(_unit_vec(1, 0), {}),
        pdf_bench.LooDoc(_unit_vec(0, 1), {}),
    ]
    train_y = ["a", "b"]
    test = pdf_bench.LooDoc(_unit_vec(1, 0.1), {"a": 0.2, "b": 0.3, "c": 0.5})
    label, probs = pdf_bench.prior_centroid_mix(0.0)(train, train_y, test)
    assert label == "c" and probs == pytest.approx(test.zero_shot)
    cent = pdf_bench.centroid_probs(train, train_y, test, 0.05)
    label, probs = pdf_bench.prior_centroid_mix(1.0)(train, train_y, test)
    assert label == "a" and probs["c"] == 0.0
    assert {k: probs[k] for k in cent} == pytest.approx(cent)
    _, half = pdf_bench.prior_centroid_mix(0.5)(train, train_y, test)
    # "c" has no training document: only the zero-shot half reaches it.
    assert half["c"] == pytest.approx(0.25)
    assert half["a"] == pytest.approx(0.5 * 0.2 + 0.5 * cent["a"])
    assert sum(half.values()) == pytest.approx(1.0)


def test_centroid_and_knn_on_toy_vectors():
    train = [_unit_vec(1, 0), _unit_vec(0.9, 0.1), _unit_vec(0, 1)]
    train_y = ["a", "a", "b"]
    centroids = pdf_bench.class_centroids(train, train_y)
    assert list(centroids) == ["a", "b"]
    assert np.linalg.norm(centroids["a"]) == pytest.approx(1.0)
    assert pdf_bench.centroid()(train, train_y, _unit_vec(0.2, 1))[0] == "b"
    assert pdf_bench.knn(1)(train, train_y, _unit_vec(0.1, 1))[0] == "b"
    # k=3: two "a" neighbours outvote the nearest "b".
    label, probs = pdf_bench.knn(3)(train, train_y, _unit_vec(0.6, 1))
    assert label == "a"
    assert sum(probs.values()) == pytest.approx(1.0)


def test_logistic_c_controls_regularisation():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    centres = {c: rng.normal(size=8) for c in "abc"}
    train, train_y = [], []
    for c, centre in centres.items():
        for _ in range(2):
            v = centre + 0.1 * rng.normal(size=8)
            train.append(v / np.linalg.norm(v))
            train_y.append(c)
    test = centres["b"] / np.linalg.norm(centres["b"])
    _, weak = pdf_bench.logistic(1.0)(train, train_y, test)
    label, strong = pdf_bench.logistic(1000.0)(train, train_y, test)
    assert label == "b"
    assert strong["b"] > weak["b"]


def test_examples_prototype_without_examples_is_zero_shot():
    label_vecs = {"a": _unit_vec(1, 0), "b": _unit_vec(0, 1)}
    fit = pdf_bench.examples_prototype(label_vecs)
    assert fit([], [], _unit_vec(0.6, 1))[0] == "b"
    # One "a" example next to the test doc pulls "a"'s vector over.
    assert fit([_unit_vec(0.6, 1)], ["a"], _unit_vec(0.6, 1))[0] == "a"


def test_config_choice_uses_only_training_accuracy():
    configs = ["x", "y"]
    labels = ["a", "a", "a", "a"]
    # Only "x" is right on doc 0; only "y" is right on docs 1-3.
    preds = [{"x": "a", "y": "b"}] + [{"x": "b", "y": "a"}] * 3
    assert pdf_bench.pick_config(preds, labels, configs) == "y"
    assert pdf_bench.pick_config(preds[:1], labels[:1], configs) == "x"
    assert pdf_bench.pick_config([{"x": "a", "y": "a"}], ["a"], configs) == "x"  # tie: first
    choice = pdf_bench.config_choice(configs)
    out = pdf_bench.loo_predict(preds, labels, choice)
    # Doc 0's own label never helps it: "y" wins on the other three, so doc 0 gets "b".
    assert [o[0] for o in out] == ["b", "a", "a", "a"]
    assert choice.chosen == ["y", "y", "y", "y"]
    assert all(set(o[1]) == {o[0]} for o in out)  # probabilities hold labels only


def test_label_text_styles():
    leaf = "legal.contract"
    node = TAXONOMY.node(leaf)
    assert pdf_bench.label_text(TAXONOMY, leaf, "label-def") == (
        f"Legal > Contract: {node.description}"
    )
    assert pdf_bench.label_text(TAXONOMY, leaf, "def-first") == (
        f"{node.description} (Legal > Contract)"
    )
    with pytest.raises(ValueError):
        pdf_bench.label_text(TAXONOMY, leaf, "nope")


def test_zero_shot_grid_is_small_and_starts_with_the_p9_setup():
    assert pdf_bench.ZERO_SHOT_GRID[0] == ("default512", "what", "label-def")
    assert len(pdf_bench.ZERO_SHOT_GRID) == len(set(pdf_bench.ZERO_SHOT_GRID)) == 16
    assert pdf_bench.ZS_VIEWS["default512"] == {}
    assert pdf_bench.ZS_INSTRUCTIONS["what"] == pdf_bench.INSTRUCTION


def test_render_experiments_groups_rows():
    metrics = {
        "n": 2,
        "leaf_accuracy": 0.5,
        "leaf_accuracy_ci95": [0.0, 1.0],
        "level1_accuracy": 1.0,
        "macro_f1": 0.333,
    }
    result = {
        "experiments": {
            "date": "2026-10-16",
            "commit": "abc",
            "device": "cpu",
            "runs": [
                {"id": "z1", "group": pdf_bench.EXP_ZERO_SHOT, "what": "zs", "metrics": metrics},
                {
                    "id": "l1",
                    "group": pdf_bench.EXP_LOO,
                    "what": "loo",
                    "metrics": metrics,
                    "chosen": {"10.0": 3, "1.0": 1},
                },
            ],
        }
    }
    text = "\n".join(pdf_bench.render_experiments(result))
    assert text.index("**Zero-shot, dev-tuned**") < text.index("**Leave-one-out (honest)**")
    assert "| `l1` | loo | 0.500 | 0.00-1.00 | 1.000 | 0.333 | 10.0: 3, 1.0: 1 |" in text
    assert pdf_bench.render_experiments({}) == []


# --- shared views, best system, incremental rows, redaction --------------------------------


def _doc(source, words):
    from docrooter.document import Document

    text = " ".join(f"word{i}" for i in range(words))
    return Document.from_sections(
        [{"title": "Introduction", "level": 1, "text": text}], source=source
    )


def test_shared_views_render_once_with_estimate_tokens_and_count_differences():
    from docrooter.views import estimate_tokens

    docs = {"a": (_doc("a.pdf", 3), 0.0), "b": (_doc("b.pdf", 600), 0.0)}
    shared = pdf_bench.SharedViews(["a", "b"], docs)
    rendered = shared.rendered(None)
    assert shared.rendered({}) is rendered  # same config, rendered once
    from docrooter.doc_view import DocumentView

    assert rendered["b"].text == DocumentView().render(docs["b"][0], estimate_tokens).text
    # A counter 10x as strict cuts the long view but not the short one: only "b" differs.
    assert shared.n_differing(None, lambda t: 10 * estimate_tokens(t)) == 1
    assert shared.n_differing(None, estimate_tokens) == 0
    manifest = shared.manifest()
    assert manifest["default"]["a"] == pdf_bench.text_sha(rendered["a"].text)


def test_fixed_view_returns_the_shared_text_whatever_the_counter():
    docs = {"a": (_doc("a.pdf", 600), 0.0)}
    shared = pdf_bench.SharedViews(["a"], docs)
    rendered = shared.rendered(None)
    view, by_source = pdf_bench.fixed_view(None, rendered)
    by_source["a.pdf"] = rendered["a"]
    assert view.render(docs["a"][0], lambda t: len(t)).text == rendered["a"].text
    other = _doc("warmup.txt", 5)
    assert view.render(other, lambda t: 10**9).text  # unknown docs still render


def _system_entry(f1, acc=0.5):
    return {"metrics": {"macro_f1": f1, "leaf_accuracy": acc}, "rows": []}


def test_best_docrooter_excludes_jev_and_dev_tuned_systems():
    result = {
        "systems": {
            "jev-view": _system_entry(0.99),
            "ultrabert-view-tuned": _system_entry(0.9),
            "ultrabert-bge-loo": _system_entry(0.8),
            "ultrabert-view": _system_entry(0.7),
            "bge-view": _system_entry(0.6),
            "nli-view": {"error": "boom"},
        }
    }
    assert pdf_bench.SYSTEMS["ultrabert-view-tuned"].dev_tuned
    assert pdf_bench.best_docrooter(result, zero_shot_only=True) == "ultrabert-view"
    assert pdf_bench.best_docrooter(result, zero_shot_only=False) == "ultrabert-bge-loo"
    assert (
        pdf_bench.best_docrooter(result, zero_shot_only=False, dev_tuned=True)
        == "ultrabert-view-tuned"
    )
    assert pdf_bench.best_docrooter({"systems": {}}, zero_shot_only=True) is None


def test_row_writer_saves_after_each_batch_and_keeps_rows_on_failure(tmp_path):
    import json

    path = tmp_path / "pdf.json"
    result = {"systems": {"jev-view": _system_entry(0.9)}}
    writer = pdf_bench.RowWriter(path, result, "jev-view", {"d1": "x", "d2": "y"})
    writer.add([{"doc_id": "d1", "pred": "x"}])
    saved = json.loads(path.read_text())
    assert saved["systems"]["jev-view"]["rows"] == [
        {"doc_id": "d1", "pred": "x", "gold": "x", "correct": True}
    ]
    assert saved["systems"]["jev-view"]["partial"] is True
    writer.add([{"doc_id": "d2", "pred": "x"}])
    writer.fail("RuntimeError: network down, key apikey_abc123")
    saved = json.loads(path.read_text())
    entry = saved["systems"]["jev-view"]
    assert [r["doc_id"] for r in entry["rows"]] == ["d1", "d2"]
    assert entry["rows"][1]["correct"] is False
    assert "apikey_abc123" not in entry["error"] and "[REDACTED]" in entry["error"]
    # A partial entry has no metrics, so the report skips it.
    assert pdf_bench._ordered(saved) == []


def test_redact_removes_key_like_and_env_secret_values():
    secret = "sk-live-0123456789abcdef"
    environ = {"TYPESAFE_API_KEY": secret, "HOME": "/Users/someone", "SHORT_KEY": "abc"}
    text = f"401 for {secret} (apikey_XYZ987, home /Users/someone, abc)"
    out = pdf_bench.redact(text, environ)
    assert secret not in out and "apikey_XYZ987" not in out
    assert out.count("[REDACTED]") == 2
    assert "/Users/someone" in out and "abc)" in out  # not secrets


def test_redact_removes_values_loaded_from_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("SOME_VALUE=hunter2-hunter2\n", encoding="utf-8")
    pdf_bench.load_env(env_file, {})
    assert pdf_bench.redact("oops hunter2-hunter2", {}) == "oops [REDACTED]"


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
