import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from conftest import (
    CountingBackend,
    NeverSupports,
    RecordingBackend,
    fin_legal_taxonomy,
    taxonomy_with_multilabel_root,
)
from typer.testing import CliRunner

from docrooter import Classifier as TopLevelClassifier
from docrooter import Document, SectionView, Taxonomy, TaxonomyNode, load
from docrooter.backends import KeywordBackend, Scores
from docrooter.classify import (
    DocumentResult,
    EngineSettings,
    Flat,
    Greedy,
    PathResult,
    SectionResult,
    Step,
    run_flat,
)
from docrooter.classify.classifier import Classifier
from docrooter.cli import app
from docrooter.errors import BackendError

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
runner = CliRunner()


@pytest.fixture
def tax():
    return fin_legal_taxonomy()


@pytest.fixture
def doc():
    return Document.from_sections(
        [
            {"title": "Taxes", "level": 1, "text": "income tax deferred tax tax"},
            {"title": "Agreements", "level": 1, "text": "contracts legal contracts"},
        ]
    )


def two_other_taxonomy() -> Taxonomy:
    return Taxonomy(
        [
            TaxonomyNode(id="a", label="A"),
            TaxonomyNode(id="a.other", label="Other", parent_id="a"),
            TaxonomyNode(id="a.x", label="X", parent_id="a"),
            TaxonomyNode(id="b", label="B"),
            TaxonomyNode(id="b.other", label="Other", parent_id="b"),
            TaxonomyNode(id="b.y", label="Y", parent_id="b"),
        ]
    )


# --- the brief's tests ------------------------------------------------------------------------


def test_classify_doc_leaves_only(tax, doc):
    res = Classifier(tax, backend=KeywordBackend(), threshold=0).classify(doc)
    assert set(res.sections) == {s.id for s in doc.leaves()}
    assert res.source == doc.source


def test_override_precedence(tax):
    a, b, c = KeywordBackend(name="a"), KeywordBackend(name="b"), KeywordBackend(name="c")
    clf = Classifier(tax, backend=a, overrides={"depth>=1": b, "node:legal": c})
    assert clf.router("financial")[0].name == "b" and clf.router("legal")[0].name == "c"
    assert clf.router(Taxonomy.ROOT_ID)[0].name == "a"


def test_flat_uses_full_path_candidates_for_duplicate_labels():
    spy = RecordingBackend(KeywordBackend())
    Classifier(two_other_taxonomy(), backend=spy, strategy=Flat()).classify_texts(["x"])
    texts = {c.text.split(":")[0] for c in spy.last_candidates}
    assert {"A > Other", "B > Other"} <= texts
    assert {c.node_id for c in spy.last_candidates} == {"a.other", "a.x", "b.other", "b.y"}


def test_summary_weighted_by_tokens(tax):
    doc = Document.from_sections(
        [
            {"title": "T", "level": 1, "text": "tax " * 30},
            {"title": "C", "level": 1, "text": "contracts"},
        ]
    )
    res = Classifier(tax, backend=KeywordBackend(), threshold=0).classify(doc)
    s = res.summary(depth=1)
    assert s["Financial"] > s["Legal"] and abs(sum(s.values()) - 1) < 1e-9


def test_classify_many_single_traversal(tax):
    one_branch = Document.from_sections(
        [{"title": "T", "level": 1, "text": "tax"}, {"title": "U", "level": 1, "text": "audit"}]
    )
    spy = CountingBackend(KeywordBackend())
    out = Classifier(tax, backend=spy, strategy=Greedy(), threshold=0).classify_many(
        [one_branch, one_branch]
    )
    assert len(out) == 2 and out[0].sections.keys() == out[1].sections.keys()
    assert spy.calls == 2  # both docs share each round's call


def test_no_backend_warns_once_per_run(tax, doc):
    with pytest.warns(UserWarning, match="no_backend") as w:
        Classifier(tax, backend=NeverSupports(), threshold=0).classify(doc)
    assert len([x for x in w if "no_backend" in str(x.message)]) == 1


def test_truncated_flag_passed_through(tax):
    long_doc = Document.from_sections([{"title": "T", "level": 1, "text": "tax " * 200}])
    res = Classifier(tax, backend=KeywordBackend(), view=SectionView(max_tokens=5)).classify(
        long_doc
    )
    assert any(r.truncated for r in res.sections.values())


def test_construction_does_not_resolve_backend(tax, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("resolved at construction")

    monkeypatch.setattr("docrooter.backends.registry.resolve_backend", boom)
    Classifier(tax, backend="embedding:whatever", overrides={"depth>=1": "nli:x"})
    Classifier(tax)  # the default spec is not resolved either


def test_cli_classify_keyword(tmp_path):
    out = tmp_path / "o.json"
    r = runner.invoke(
        app,
        [
            "classify",
            str(FIXTURES / "docs" / "simple.md"),
            "-t",
            str(FIXTURES / "taxonomies" / "nested.json"),
            "-b",
            "keyword",
            "-o",
            str(out),
        ],
    )
    assert r.exit_code == 0, r.output
    assert json.loads(out.read_text())["sections"]


# --- additional behaviour ---------------------------------------------------------------------


def test_top_level_export():
    assert TopLevelClassifier is Classifier


def test_no_heading_document_is_classified(tax):
    doc = load("financial tax income tax\nmore text about audit")
    res = Classifier(tax, backend=KeywordBackend(), threshold=0).classify(doc)
    assert set(res.sections) == {"s0"}
    assert res.sections["s0"].best.node_ids[0] == "financial"


def test_empty_document_makes_no_backend_call(tax):
    spy = CountingBackend(KeywordBackend())
    empty = load("")
    assert empty.sections == {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = Classifier(tax, backend=spy).classify(empty)
        assert Classifier(tax, backend=spy).classify_many([empty, empty])[1].sections == {}
        assert Classifier(tax, backend=spy).classify_texts([]) == []
    assert res.sections == {} and spy.calls == 0


def test_sections_selection(tax):
    doc = load("pre text\n\n# One\ntax\n\n## Sub\naudit\n\n# Two\ncontracts\n")
    assert [s.id for s in doc.walk()] == ["s0", "s1", "s1.1", "s2"]

    def ids(sections):
        clf = Classifier(tax, backend=KeywordBackend(), threshold=0, sections=sections)
        return list(clf.classify(doc).sections)

    assert ids("leaves") == ["s0", "s1.1", "s2"]
    assert ids("all") == ["s0", "s1", "s1.1", "s2"]
    assert ids([1]) == ["s1", "s2"]
    assert ids([0, 2]) == ["s0", "s1.1"]
    assert ids(lambda s: s.title == "Two") == ["s2"]


def test_sections_all_skips_blank_preamble(tax):
    doc = Document.from_sections([{"title": "One", "level": 1, "text": "tax"}])
    clf = Classifier(tax, backend=KeywordBackend(), threshold=0, sections="all")
    assert list(clf.classify(doc).sections) == ["s1"]


def test_bad_sections_value_rejected(tax):
    with pytest.raises(ValueError, match="sections"):
        Classifier(tax, sections="everything")


@pytest.mark.parametrize("key", ["depth>1", "depth>=x", "foo", "node:", "depth=2", "node:nope"])
def test_bad_override_key_rejected(tax, key):
    with pytest.raises(ValueError, match="override"):
        Classifier(tax, overrides={key: KeywordBackend()})


def test_override_key_forms_and_priority_order():
    t = fin_legal_taxonomy()
    n = {x: KeywordBackend(name=x) for x in "pqrstuv"}
    clf = Classifier(
        t,
        backend=n["p"],
        fallback=n["v"],
        overrides={
            "depth==2": n["q"],
            "depth<=1": n["r"],
            "node:financial.tax": n["s"],
            "depth>=0": n["u"],
        },
    )
    assert [b.name for b in clf.router("financial.tax")] == ["s", "q", "p", "v"]
    assert [b.name for b in clf.router("financial.audit")] == ["q", "p", "v"]
    assert [b.name for b in clf.router("financial")] == ["r", "p", "v"]  # depth<=1 before depth>=0
    assert [b.name for b in clf.router(Taxonomy.ROOT_ID)] == ["r", "p", "v"]


def test_node_backend_beats_depth_override():
    nodes = [
        TaxonomyNode(id="a", label="A", backend="keyword:node"),
        TaxonomyNode(id="a.x", label="X", parent_id="a"),
    ]
    clf = Classifier(
        Taxonomy(nodes),
        backend=KeywordBackend(name="p"),
        overrides={"depth>=0": KeywordBackend(name="d"), "node:a": KeywordBackend(name="n")},
    )
    # node override, the node's own spec ("keyword:node" names its backend "node"), depth, primary
    assert [b.name for b in clf.router("a")] == ["n", "node", "d", "p"]
    assert [b.name for b in clf.router("a.x")] == ["d", "p"]
    clf2 = Classifier(
        Taxonomy(nodes), backend=KeywordBackend(name="p"), overrides={"depth>=0": "keyword"}
    )
    # a.backend is a spec string resolved lazily; the router puts it ahead of depth overrides
    assert [b.name for b in clf2.router("a")] == ["node", "keyword", "p"]


def test_specs_resolved_once_per_spec_string(tax, monkeypatch):
    calls = []

    def fake_resolve(spec):
        calls.append(spec)
        return KeywordBackend(name=spec)

    monkeypatch.setattr("docrooter.backends.registry.resolve_backend", fake_resolve)
    clf = Classifier(tax, backend="x", overrides={"depth>=1": "y", "node:legal": "y"})
    for node_id in ("legal", "financial", Taxonomy.ROOT_ID):
        list(clf.router(node_id))
    assert sorted(calls) == ["x", "y"]


def test_fallback_is_appended_last_and_used_when_primary_unsupported(tax, doc):
    fallback = CountingBackend(KeywordBackend(name="fb"))
    clf = Classifier(tax, backend=NeverSupports(), fallback=fallback, threshold=0)
    assert [b.name for b in clf.router("financial")] == ["never", "fb"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = clf.classify(doc)
    assert fallback.calls > 0
    assert all(r.best.stopped_reason is None for r in res.sections.values())


def test_trained_model_gets_zero_shot_fallback(tax):
    zero = KeywordBackend(name="zero")

    class Trained(KeywordBackend):
        def as_zero_shot(self):
            return zero

    trained = Trained(name="trained")
    assert [b.name for b in Classifier(tax, backend=trained).router("financial")] == [
        "trained",
        "zero",
    ]
    explicit = KeywordBackend(name="mine")
    clf = Classifier(tax, backend=trained, fallback=explicit)
    assert [b.name for b in clf.router("financial")] == ["trained", "mine"]


def test_decide_delegates_to_engine(tax):
    clf = Classifier(tax, backend=KeywordBackend())
    ids, probs, backend = clf.decide(["financial tax", "legal contracts"], Taxonomy.ROOT_ID)
    assert ids == ["financial", "legal"] and backend == "keyword"
    assert probs.shape == (2, 2) and probs[0, 0] > probs[0, 1] and probs[1, 1] > probs[1, 0]


def test_settings_reflect_constructor(tax):
    clf = Classifier(
        tax,
        threshold=0.2,
        min_margin=0.1,
        max_fanout=2,
        batch_size=4,
        instruction="inst",
        temperatures={"keyword": 0.5},
    )
    assert clf.settings == EngineSettings(
        threshold=0.2,
        min_margin=0.1,
        max_fanout=2,
        batch_size=4,
        instruction="inst",
        temperatures={"keyword": 0.5},
    )


def test_strategy_is_validated(tax):
    with pytest.raises(TypeError, match="strategy"):
        Classifier(tax, strategy="beam")  # type: ignore[arg-type]


def test_instruction_reaches_backend(tax):
    spy = RecordingBackend(KeywordBackend())
    Classifier(tax, backend=spy, instruction="Pick one", strategy=Greedy()).classify_texts(["tax"])
    assert spy.last_instruction == "Pick one"


def test_classify_texts_ids_and_tokens(tax):
    clf = Classifier(tax, backend=KeywordBackend(), threshold=0, strategy=Greedy())
    out = clf.classify_texts(["financial tax", "legal contracts and litigation"])
    assert [r.section_id for r in out] == ["0", "1"]
    assert out[0].best.node_ids == ["financial", "financial.tax"]
    assert out[1].best.node_ids[0] == "legal"
    assert out[0].n_tokens == 3 and out[0].truncated is False


def test_classify_texts_truncates_to_view_budget(tax):
    spy = RecordingBackend(KeywordBackend())
    clf = Classifier(tax, backend=spy, view=SectionView(max_tokens=10), strategy=Greedy())
    (res,) = clf.classify_texts(["tax " * 100])
    assert res.truncated is True and res.n_tokens == 130
    sent = spy.last_texts[0]
    assert 0 < len(sent.split()) < 100
    assert (
        Classifier(tax, backend=spy, view=SectionView(max_tokens=10))
        .classify_texts(["tax tax"])[0]
        .truncated
        is False
    )


def test_classify_texts_warns_for_no_backend(tax):
    with pytest.warns(UserWarning, match="no_backend"):
        Classifier(tax, backend=NeverSupports()).classify_texts(["tax"])


def test_no_backend_warning_names_count_and_nodes(tax, doc):
    with pytest.warns(UserWarning) as w:
        Classifier(tax, backend=NeverSupports()).classify(doc)
    (msg,) = [str(x.message) for x in w if "no_backend" in str(x.message)]
    assert "2 section" in msg and Taxonomy.ROOT_ID in msg


def test_no_warning_when_all_decided(tax, doc):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Classifier(tax, backend=KeywordBackend(), threshold=0).classify(doc)


def test_classify_many_matches_individual_classify(tax, doc):
    other = Document.from_sections([{"title": "Z", "level": 1, "text": "litigation contracts"}])
    clf = Classifier(tax, backend=KeywordBackend(), threshold=0)
    many = clf.classify_many([doc, other])
    assert many[0] == clf.classify(doc) and many[1] == clf.classify(other)
    assert clf.classify_many([]) == []


def test_section_result_fields(tax, doc):
    res = Classifier(tax, backend=KeywordBackend(), threshold=0).classify(doc)
    r = res.sections["s1"]
    assert r.section_id == "s1" and r.title == "Taxes" and r.n_tokens == 7
    assert r.selected == [r.best]


# --- Flat -------------------------------------------------------------------------------------


def run(texts, taxonomy, backend, strategy=None, **settings):
    return run_flat(
        texts, taxonomy, backend, strategy or Flat(), EngineSettings(**{"threshold": 0, **settings})
    )


def test_flat_picks_best_leaf_with_full_path(tax):
    out = run({"x": "income tax tax", "y": "litigation"}, tax, KeywordBackend())
    best, alts, selected = out["x"]
    assert best.node_ids == ["financial", "financial.tax"]
    assert best.labels == ["Financial", "Tax"] and best.stopped_reason is None
    assert len(best.steps) == 1 and best.steps[0].node_id == Taxonomy.ROOT_ID
    assert best.steps[0].chosen == ["financial.tax"] and best.steps[0].backend == "keyword"
    assert selected == [best]
    assert [a.node_ids[-1] for a in alts] and best.node_ids[-1] not in {
        a.node_ids[-1] for a in alts
    }
    assert out["y"][0].node_ids == ["legal", "legal.litigation"]


def test_flat_alternatives_are_ranked_and_capped_at_five():
    nodes = [TaxonomyNode(id=f"n{i}", label=f"word{i}") for i in range(9)]
    t = Taxonomy(nodes)
    best, alts, _ = run({"x": "word3 word5"}, t, KeywordBackend())["x"]
    assert len(alts) == 5
    assert best.node_ids == ["n3"]
    assert all(a.score <= best.score for a in alts)
    assert [a.score for a in alts] == sorted((a.score for a in alts), reverse=True)


def test_flat_low_confidence(tax):
    best, alts, selected = run({"x": "financial"}, tax, KeywordBackend(), threshold=0.99)["x"]
    assert best.node_ids == [] and best.stopped_reason == "low_confidence"
    assert selected == [] and alts


def test_flat_uses_node_threshold_of_root():
    t = Taxonomy(
        [TaxonomyNode(id="a", label="A"), TaxonomyNode(id="b", label="B")],
        root={"threshold": 0.99},
    )
    best, _, _ = run({"x": "a b"}, t, KeywordBackend(), threshold=0.0)["x"]
    assert best.stopped_reason == "low_confidence"


def test_flat_batches_by_batch_size(tax):
    spy = CountingBackend(KeywordBackend())
    run({str(i): "tax" for i in range(5)}, tax, spy, batch_size=2)
    assert spy.calls == 3


def test_flat_allow_internal_scores_all_nodes(tax):
    spy = RecordingBackend(KeywordBackend())
    best, _, _ = run({"x": "financial"}, tax, spy, Flat(allow_internal=True))["x"]
    assert len(spy.last_candidates) == 6
    assert best.node_ids == ["financial"]


def test_flat_leaves_only_by_default(tax):
    spy = RecordingBackend(KeywordBackend())
    run({"x": "tax"}, tax, spy)
    assert [c.node_id for c in spy.last_candidates] == [
        "financial.tax",
        "financial.audit",
        "legal.contracts",
        "legal.litigation",
    ]


def test_flat_no_backend(tax):
    best, alts, selected = run({"x": "tax"}, tax, None)["x"]
    assert best.node_ids == [] and best.stopped_reason == "no_backend"
    assert alts == [] and selected == []


def test_flat_rejects_wrong_score_shape(tax):
    class Bad(KeywordBackend):
        def score(self, texts, candidates, **kw):
            return Scores(np.zeros((len(texts), 1)), "similarity")

    with pytest.raises(BackendError, match="shape"):
        run({"x": "tax"}, tax, Bad())


def test_flat_applies_temperature_override(tax):
    cool = run({"x": "tax"}, tax, KeywordBackend())["x"][0].score
    warm = run({"x": "tax"}, tax, KeywordBackend(), temperatures={"keyword": 10.0})["x"][0].score
    assert warm < cool


def test_flat_classifier_skips_unsupported_primary_for_fallback(tax):
    clf = Classifier(
        tax, backend=NeverSupports(), fallback=KeywordBackend(), strategy=Flat(), threshold=0
    )
    (res,) = clf.classify_texts(["income tax"])
    assert res.best.node_ids == ["financial", "financial.tax"]


def test_flat_classifier_respects_max_candidates(tax):
    class Small(KeywordBackend):
        max_candidates = 2

    clf = Classifier(tax, backend=Small(name="small"), strategy=Flat())
    with pytest.warns(UserWarning, match="no_backend"):
        (res,) = clf.classify_texts(["income tax"])
    assert res.best.stopped_reason == "no_backend"


def test_flat_with_multilabel_taxonomy_is_still_single_label():
    t = taxonomy_with_multilabel_root()
    clf = Classifier(t, backend=KeywordBackend(), strategy=Flat(), threshold=0)
    (res,) = clf.classify_texts(["tax audit contracts"])
    assert len(res.selected) == 1


def test_flat_classifier_on_document(tax, doc):
    res = Classifier(tax, backend=KeywordBackend(), strategy=Flat(), threshold=0).classify(doc)
    assert res.sections["s1"].best.node_ids == ["financial", "financial.tax"]
    assert res.sections["s2"].best.node_ids[0] == "legal"


# --- DocumentResult ---------------------------------------------------------------------------


def _result(sid, n_tokens, labels, score=0.5, reason=None):
    best = PathResult(
        node_ids=[f"id-{label}" for label in labels],
        labels=labels,
        score=score,
        steps=[Step(node_id="r", chosen=["c"], probs={"c": 1.0}, backend="b")],
        stopped_reason=reason,
    )
    return SectionResult(section_id=sid, title=sid.upper(), best=best, n_tokens=n_tokens)


def test_summary_depths_and_edge_cases():
    res = DocumentResult(
        source="d",
        sections={
            "a": _result("a", 30, ["Financial", "Tax"]),
            "b": _result("b", 10, ["Legal"]),
            "c": _result("c", 20, []),  # no path: ignored
            "d": SectionResult(section_id="d", title="D", best=None, n_tokens=99),
        },
    )
    assert res.summary() == pytest.approx({"Financial": 0.75, "Legal": 0.25})
    assert res.summary(depth=2) == {"Financial > Tax": 1.0}
    assert res.summary(depth=3) == {}
    with pytest.raises(ValueError, match="depth"):
        res.summary(depth=0)


def test_summary_zero_weight_is_empty():
    res = DocumentResult(source="d", sections={"a": _result("a", 0, ["Financial"])})
    assert res.summary() == {}
    assert DocumentResult(source="d", sections={}).summary() == {}


def test_to_json_round_trips():
    res = DocumentResult(source="d", sections={"a": _result("a", 3, ["Financial"])})
    text = res.to_json()
    assert json.loads(text)["sections"]["a"]["title"] == "A"
    assert DocumentResult.model_validate_json(text) == res
    assert "\n  " in text and "\n    " not in res.to_json(indent=None)


def test_to_dataframe_columns_and_rows():
    pd = pytest.importorskip("pandas")
    res = DocumentResult(
        source="d",
        sections={
            "a": _result("a", 3, ["Financial", "Tax"], score=0.6),
            "b": _result("b", 3, [], reason="low_confidence"),
        },
    )
    df = res.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["section_id", "title", "path", "score", "stopped_reason"]
    assert df.loc[0, "path"] == "Financial > Tax" and df.loc[0, "score"] == 0.6
    assert df.loc[1, "stopped_reason"] == "low_confidence" and df.loc[1, "path"] == ""
    empty = DocumentResult(source="d", sections={}).to_dataframe()
    assert list(empty.columns) == ["section_id", "title", "path", "score", "stopped_reason"]


def test_to_dataframe_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "pandas", None)  # makes `import pandas` fail
    with pytest.raises(ImportError, match=r"pip install 'docrooter\[bench\]'"):
        DocumentResult(source="d", sections={}).to_dataframe()


# --- CLI --------------------------------------------------------------------------------------

SIMPLE = str(FIXTURES / "docs" / "simple.md")
NESTED = str(FIXTURES / "taxonomies" / "nested.json")


def test_cli_classify_stdout_is_json():
    r = runner.invoke(app, ["classify", SIMPLE, "-t", NESTED, "-b", "keyword", "--threshold", "0"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.stdout)
    assert set(data["sections"]) == {"s1.1", "s1.2", "s2"}


@pytest.mark.parametrize("strategy", ["greedy", "beam:2", "flat", "beam:3"])
def test_cli_classify_strategies(strategy):
    r = runner.invoke(
        app,
        ["classify", SIMPLE, "-t", NESTED, "-b", "keyword", "--strategy", strategy,
         "--threshold", "0"],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["sections"]


@pytest.mark.parametrize("strategy", ["beam", "beam:x", "beam:0", "deep"])
def test_cli_classify_bad_strategy(strategy):
    r = runner.invoke(
        app, ["classify", SIMPLE, "-t", NESTED, "-b", "keyword", "--strategy", strategy]
    )
    assert r.exit_code != 0 and "strategy" in r.output.lower()


def test_cli_classify_table_when_writing_file(tmp_path):
    out = tmp_path / "o.json"
    r = runner.invoke(
        app,
        ["classify", SIMPLE, "-t", NESTED, "-b", "keyword", "--threshold", "0", "-o", str(out)],
    )
    assert r.exit_code == 0, r.output
    assert "Background" in r.output and "Financial" in r.output
    assert DocumentResult.model_validate_json(out.read_text()).sections


def test_cli_classify_escapes_markup_in_table(tmp_path):
    md = tmp_path / "d.md"
    md.write_text("# [bold]Tax[/bold] [red]\ntax tax\n")
    out = tmp_path / "o.json"
    r = runner.invoke(
        app,
        ["classify", str(md), "-t", NESTED, "-b", "keyword", "--threshold", "0", "-o", str(out)],
    )
    assert r.exit_code == 0, r.output
    assert "[bold]Tax[/bold] [red]" in r.output


def test_cli_classify_instruction_and_unknown_backend():
    r = runner.invoke(
        app, ["classify", SIMPLE, "-t", NESTED, "-b", "nope:x", "--instruction", "hi"]
    )
    assert r.exit_code == 1 and "nope" in r.output


def test_cli_classify_bad_taxonomy(tmp_path):
    r = runner.invoke(app, ["classify", SIMPLE, "-t", str(FIXTURES / "taxonomies" / "broken.csv")])
    assert r.exit_code == 1


def test_cli_classify_default_backend_is_ultrabert():
    from docrooter.cli import DEFAULT_BACKEND

    assert DEFAULT_BACKEND == "embedding:dmis-lab/UltraBERT-Embed-150M"


# --- fix round 1 ------------------------------------------------------------------------------


def test_cli_classify_table_shows_stop_reason_instead_of_score(tmp_path):
    out = tmp_path / "o.json"
    r = runner.invoke(
        app,
        ["classify", SIMPLE, "-t", NESTED, "-b", "keyword", "--threshold", "0.99", "-o", str(out)],
    )
    assert r.exit_code == 0, r.output
    assert "low_confidence" in r.output and "1.00" not in r.output
    data = json.loads(out.read_text())
    assert all(s["best"]["node_ids"] == [] for s in data["sections"].values())


def test_fallback_not_resolved_while_primary_handles_every_node(tax, doc, monkeypatch):
    def boom(spec):
        raise AssertionError(f"resolved {spec!r}")

    monkeypatch.setattr("docrooter.backends.registry.resolve_backend", boom)
    clf = Classifier(tax, backend=KeywordBackend(), fallback="nli:never-loaded", threshold=0)
    res = clf.classify(doc)
    assert all(r.best.stopped_reason is None for r in res.sections.values())
    assert clf.router("financial")[0].name == "keyword"  # indexing stays lazy too
    Classifier(
        tax, backend=KeywordBackend(), fallback="nli:never-loaded", strategy=Flat(), threshold=0
    ).classify(doc)


def test_zero_shot_fallback_not_built_while_trained_primary_suffices(tax, doc):
    class Trained(KeywordBackend):
        def as_zero_shot(self):
            raise AssertionError("zero-shot fallback built needlessly")

    Classifier(tax, backend=Trained(), threshold=0).classify(doc)


def test_fallback_resolved_when_primary_unsupported(tax, doc, monkeypatch):
    seen = []

    def fake(spec):
        seen.append(spec)
        return KeywordBackend(name=spec)

    monkeypatch.setattr("docrooter.backends.registry.resolve_backend", fake)
    Classifier(tax, backend=NeverSupports(), fallback="fb", threshold=0).classify(doc)
    assert seen == ["fb"]


def test_blank_texts_are_not_scored(tax):
    spy = RecordingBackend(KeywordBackend())
    clf = Classifier(tax, backend=spy, strategy=Greedy(), threshold=0)
    out = clf.classify_texts(["", "financial tax", "  \n\t "])
    assert [r.section_id for r in out] == ["0", "1", "2"]
    for i in (0, 2):
        assert out[i].best == PathResult(node_ids=[], stopped_reason="empty_text")
        assert out[i].selected == [] and out[i].alternatives == [] and out[i].truncated is False
    assert out[1].best.node_ids == ["financial", "financial.tax"]
    assert spy.last_texts == ["financial tax"]


def test_all_blank_texts_make_no_backend_call(tax):
    spy = CountingBackend(KeywordBackend())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = Classifier(tax, backend=spy).classify_texts(["", " "])
    assert spy.calls == 0 and {r.best.stopped_reason for r in out} == {"empty_text"}


def test_summary_keys_deeper_levels_by_full_path():
    """Same leaf label under different parents stays apart below the top level."""
    res = DocumentResult(
        source="d",
        sections={
            "a": _result("a", 30, ["A", "Other"]),
            "b": _result("b", 10, ["B", "Other"]),
        },
    )
    assert res.summary(depth=2) == pytest.approx({"A > Other": 0.75, "B > Other": 0.25})
    assert res.summary(depth=1) == pytest.approx({"A": 0.75, "B": 0.25})


def _trained_instance():
    from docrooter.train.bundle import TrainedModel

    class Stub(TrainedModel):
        def __init__(self):
            self.name = "trained:stub"

    return Stub()


@pytest.mark.parametrize("backend", ["trained:some/bundle", "instance"])
def test_flat_strategy_rejects_trained_backends(tax, backend):
    spec = _trained_instance() if backend == "instance" else backend
    message = "Flat strategy needs a zero-shot backend; trained backends score per-node heads"
    with pytest.raises(ValueError) as info:
        Classifier(tax, spec, strategy=Flat())
    assert str(info.value) == message
    Classifier(tax, spec, strategy=Greedy())  # other strategies accept it
