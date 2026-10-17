"""Held-out evaluation plumbing: corpus selection, the dev -> held-out transfer system, the
empty-text path of the doc-type harness, the freeze record and the dev-vs-held-out table.
No PDFs, no models, no torch."""

import json

import pytest

from benchmarks import corpora, heldout_eval, pdf_bench
from benchmarks import toc as bench_toc
from docrooter.document import Document
from docrooter.taxonomies import builtin_taxonomy

TAXONOMY = builtin_taxonomy("doc_types")


def _write_manifest(directory, records):
    directory.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r) for r in records]
    (directory / "manifest.jsonl").write_text("\n".join(lines) + "\n\n", encoding="utf-8")


# --- corpora -----------------------------------------------------------------------------


def test_corpus_names_and_results_files():
    assert corpora.corpus_path("dev") == corpora.DEV
    assert corpora.corpus_path("heldout") == corpora.HELDOUT
    assert corpora.corpus_path("some/dir") == corpora.Path("some/dir")
    assert corpora.results_name(corpora.DEV, "pdf.json") == "pdf.json"
    assert corpora.results_name(corpora.HELDOUT, "pdf.json") == "heldout-pdf.json"
    assert corpora.relative(corpora.HELDOUT) == "benchmarks/heldout"
    assert pdf_bench.CORPUS == corpora.DEV and bench_toc.CORPUS == corpora.DEV


def test_manifest_skips_the_meta_record(tmp_path):
    _write_manifest(
        tmp_path,
        [{"_meta": {"n_docs": 2}}, {"id": "x1", "file": "pdfs/x1.pdf"}, {"id": "x2", "file": "b"}],
    )
    assert [r["id"] for r in pdf_bench.manifest(tmp_path)] == ["x1", "x2"]


def test_committed_heldout_manifest_has_40_documents_and_no_dev_overlap():
    held = pdf_bench.manifest(corpora.HELDOUT)
    dev = pdf_bench.manifest(corpora.DEV)
    assert len(held) == 40 and all("_meta" not in r for r in held)
    assert not {r["id"] for r in held} & {r["id"] for r in dev}
    assert {r["label"] for r in held} <= {n.id for n in TAXONOMY.leaves()}


def test_toc_corpus_docs_skips_the_meta_record_and_missing_pdfs(tmp_path):
    _write_manifest(tmp_path, [{"_meta": {}}, {"id": "x1", "file": "pdfs/missing.pdf"}])
    assert bench_toc.corpus_docs(tmp_path) == []


# --- trained on dev, tested on held-out ----------------------------------------------------


def test_transfer_systems_are_registered_and_not_zero_shot():
    for name, method in (
        ("ultrabert-dev-logistic", "logistic-cv"),
        ("ultrabert-bge-dev-ensemble", "ensemble-cv"),
    ):
        system = pdf_bench.SYSTEMS[name]
        assert system.kind == "transfer" and system.method == method and not system.dev_tuned
        assert name not in pdf_bench.DEFAULT_ORDER  # the dev benchmark is unchanged
    entry = {"metrics": {"macro_f1": 0.9, "leaf_accuracy": 0.9}, "rows": []}
    low = {"metrics": {"macro_f1": 0.1, "leaf_accuracy": 0.1}, "rows": []}
    result = {"systems": {"ultrabert-dev-logistic": entry, "bge-view": low}}
    assert pdf_bench.best_docrooter(result, zero_shot_only=True) == "bge-view"
    assert pdf_bench.best_docrooter(result, zero_shot_only=False) == "ultrabert-dev-logistic"


def test_transfer_predict_tunes_on_training_docs_only_then_predicts_every_test_doc():
    train_x = [[0.0], [1.0], [2.0], [3.0]]
    train_y = ["a", "a", "b", "b"]
    test_x = [[10.0], [11.0], [12.0]]
    calls = []

    def make(value):
        def fit_predict(xs, ys, x):
            calls.append((value, [v[0] for v in xs], x[0]))
            guess = "a" if (x[0] < 2) == (value == "right") else "b"
            return guess, {guess: 1.0}

        return fit_predict

    tuned = pdf_bench.InnerLoo(make, ["wrong", "right"])
    chosen, preds = pdf_bench.transfer_predict(train_x, train_y, test_x, tuned)
    assert chosen == "right"
    selection, final = calls[:-3], calls[-3:]
    assert len(selection) == 2 * 4  # inner LOO over the 4 training docs per value
    for _, xs, x in selection:
        assert x < 10 and all(v < 10 for v in xs)  # test docs never seen while tuning
    assert [c[2] for c in final] == [10.0, 11.0, 12.0]
    assert all(c[0] == "right" and c[1] == [0.0, 1.0, 2.0, 3.0] for c in final)
    assert [p[0] for p in preds] == ["b", "b", "b"]
    assert tuned.chosen == []  # chosen once, not per test document


def test_run_transfer_refuses_test_docs_from_the_training_corpus(tmp_path):
    _write_manifest(tmp_path, [{"id": "d1", "file": "x.pdf", "label": "legal.contract"}])
    system = pdf_bench.SYSTEMS["ultrabert-dev-logistic"]
    with pytest.raises(ValueError, match="training corpus"):
        pdf_bench.run_transfer(system, ["d1"], {}, TAXONOMY, None, train_corpus=tmp_path)


# --- empty text (image-only PDFs) ----------------------------------------------------------


def _doc(source, text, *, no_text_layer=False):
    doc = Document.from_sections([{"title": "Notice", "level": 1, "text": text}], source=source)
    if no_text_layer:
        doc.meta["warnings"] = ["no_text_layer"]
    return doc


def test_run_docrooter_counts_an_image_only_doc_as_a_miss(monkeypatch):
    from docrooter.backends.fake import KeywordBackend

    monkeypatch.setattr(pdf_bench, "_backend", lambda system: KeywordBackend())
    docs = {
        "t": (_doc("t.pdf", "This agreement between the parties sets out terms."), 0.1),
        "e": (_doc("e.pdf", "", no_text_layer=True), 0.2),
    }
    shared = pdf_bench.SharedViews(["t", "e"], docs)
    system = pdf_bench.System("kw", "docrooter", "keyword", "keyword")
    added = []
    run = pdf_bench.run_docrooter(system, ["t", "e"], docs, TAXONOMY, shared, added.extend)
    rows = {r["doc_id"]: r for r in run["rows"]}
    assert rows["t"]["pred"] is not None and "stopped_reason" not in rows["t"]
    assert rows["e"]["pred"] is None and rows["e"]["stopped_reason"] == "empty_text"
    assert len(added) == 2


# --- freeze record -----------------------------------------------------------------------


def _fake_git(status="", diff="", head="abc1234"):
    calls = []

    def run(args):
        calls.append(list(args))
        if args[0] == "status":
            return status
        if args[0] == "diff":
            return diff
        if args[0] == "rev-parse":
            return head
        if args[0] == "log":
            return "lab0001"
        raise AssertionError(args)

    return run, calls


def test_code_changes_lists_uncommitted_and_committed_changes():
    run, calls = _fake_git(status=" M benchmarks/toc.py\n", diff="src/docrooter/x.py\n")
    assert heldout_eval.code_changes("abc1234", run=run) == [
        "benchmarks/toc.py",
        "src/docrooter/x.py",
    ]
    status_call = calls[0]
    assert ":(exclude)benchmarks/results" in status_call
    assert ":(exclude)src/docrooter/ui" in status_call
    assert calls[1][:4] == ["diff", "--name-only", "abc1234", "HEAD"]


def test_freeze_record_refuses_uncommitted_code_and_records_commits():
    dirty, _ = _fake_git(status=" M benchmarks/pdf_bench.py\n")
    with pytest.raises(RuntimeError, match="uncommitted"):
        heldout_eval.freeze_record(run=dirty)
    clean, _ = _fake_git()
    record = heldout_eval.freeze_record(run=clean)
    assert record["frozen_commit"] == "abc1234"
    assert record["label_commits"] == {
        "benchmarks/heldout/manifest.jsonl": "lab0001",
        "benchmarks/heldout/section_gold.jsonl": "lab0001",
    }
    assert record["steps"] == {}


def test_stamp_adds_the_freeze_record_and_keeps_results(tmp_path):
    path = tmp_path / "heldout-toc.json"
    path.write_text(json.dumps({"engines": {"pymupdf": {}}}))
    heldout_eval.stamp(path, {"frozen_commit": "abc1234"})
    saved = json.loads(path.read_text())
    assert saved == {"engines": {"pymupdf": {}}, "freeze": {"frozen_commit": "abc1234"}}
    heldout_eval.stamp(tmp_path / "missing.json", {})  # no file, no error


# --- dev vs held-out ---------------------------------------------------------------------


def _toc(f1):
    agg = {"f1": f1, "macro_f1": f1 / 2}
    return {"engines": {"pymupdf": {"n_docs": 3, "depth2": agg, "all": agg}}}


def _pdf(**systems):
    return {
        "systems": {
            name: {"metrics": {"leaf_accuracy": acc, "macro_f1": acc - 0.1, "n": n}}
            for name, (acc, n) in systems.items()
        }
    }


def _norm(acc):
    overall = {"accuracy": acc, "macro_f1": acc, "n_scored": 10}
    return {"systems": {"jev": {"overall": overall}}, "auto_doc_type": {"error": "boom"}}


def test_comparison_rows_pair_loo_systems_with_their_transfer_counterparts():
    dev = {
        "toc": _toc(0.7),
        "pdf": _pdf(**{"jev-view": (0.9, 51), "ultrabert-loo": (0.7, 51)}),
        "normalize": _norm(0.7),
    }
    held = {
        "toc": _toc(0.6),
        "pdf": _pdf(**{"jev-view": (0.8, 40), "ultrabert-dev-logistic": (0.5, 40)}),
        "normalize": _norm(0.6),
    }
    rows = heldout_eval.comparison_rows(dev, held)
    keys = [(r["task"], r["system"], r["metric"]) for r in rows]
    assert ("TOC", "pymupdf", "F1 micro (d<=2)") in keys
    assert ("Doc type", "jev-view", "leaf accuracy") in keys
    assert ("Doc type", "ultrabert-loo -> ultrabert-dev-logistic", "macro F1") in keys
    assert ("Normalisation", "jev", "accuracy") in keys
    assert not any(r["system"] == "auto_doc_type" for r in rows)  # errored run skipped
    jev = next(r for r in rows if r["system"] == "jev-view" and r["metric"] == "leaf accuracy")
    assert (jev["dev"], jev["heldout"], jev["n_dev"], jev["n_heldout"]) == (0.9, 0.8, 51, 40)
    table = heldout_eval.render_comparison(rows)
    expected = "| Doc type | `jev-view` | leaf accuracy | 0.900 | 0.800 | -10.0 pts | 51 / 40 |"
    assert expected in table
    assert heldout_eval.comparison_rows({}, {}) == []


def test_replace_block_appends_once_then_replaces_in_place():
    text = "# PDF benchmarks\n\n## Other\n\nkept\n"
    once = heldout_eval.replace_block(text, "<!-- b -->", "<!-- e -->", "v1", "## Dev vs held-out")
    assert once.startswith(text.rstrip())
    assert once.endswith("## Dev vs held-out\n\n<!-- b -->\nv1\n<!-- e -->\n")
    twice = heldout_eval.replace_block(once, "<!-- b -->", "<!-- e -->", "v2", "## Dev vs held-out")
    assert twice.count("## Dev vs held-out") == 1 and "v2" in twice and "v1" not in twice
    assert "kept" in twice


def test_git_output_keeps_the_status_column_of_the_first_line(monkeypatch):
    class Done:
        def __init__(self, stdout):
            self.stdout = stdout

    status = " M benchmarks/toc.py\n?? src/docrooter/new.py\n"
    outputs = {"status": status, "rev-parse": "abc1234\n"}
    monkeypatch.setattr(
        heldout_eval.subprocess, "run", lambda argv, **kwargs: Done(outputs[argv[1]])
    )
    assert heldout_eval.git(["rev-parse", "--short", "HEAD"]) == "abc1234"
    # stripping the whole output used to drop the first line's leading space: "rc/..."
    assert heldout_eval.code_changes(run=heldout_eval.git) == [
        "benchmarks/toc.py",
        "src/docrooter/new.py",
    ]


def test_run_steps_records_repo_relative_arguments(monkeypatch):
    launched = []

    class Done:
        returncode = 0

    def fake_run(argv, **kwargs):
        launched.append((argv, kwargs["cwd"]))
        return Done()

    monkeypatch.setattr(heldout_eval.subprocess, "run", fake_run)
    record = {"frozen_commit": "abc", "steps": {}}
    monkeypatch.setattr(heldout_eval, "freeze_record", lambda: record)
    monkeypatch.setattr(heldout_eval, "code_changes", lambda since=None: [])
    monkeypatch.setattr(heldout_eval, "git", lambda args: "abc")
    monkeypatch.setattr(heldout_eval, "stamp", lambda path, freeze: None)
    out_dir = heldout_eval.ROOT / "benchmarks" / "results" / "2099-01-01"
    freeze = heldout_eval.run_steps(["toc"], out_dir)
    argv = freeze["steps"]["toc"]["argv"]
    assert argv[argv.index("--out") + 1] == "benchmarks/results/2099-01-01"
    assert not any(str(heldout_eval.ROOT) in a for a in argv)
    ((ran, cwd),) = launched
    assert ran[1:] == argv and cwd == heldout_eval.ROOT  # what ran is what was recorded
    assert heldout_eval.repo_relative(heldout_eval.ROOT.parent / "x") == str(
        heldout_eval.ROOT.parent / "x"
    )  # outside the repository: kept as given
