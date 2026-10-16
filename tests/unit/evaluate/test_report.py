import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

import docrooter
from docrooter import Classifier, Dataset, Taxonomy, TaxonomyNode
from docrooter.backends import KeywordBackend
from docrooter.cli import app
from docrooter.errors import BackendError
from docrooter.evaluate import Report, compare, evaluate
from docrooter.train import Example

FIXTURES = Path(__file__).parents[2] / "fixtures"
SMALL = FIXTURES / "datasets" / "small.jsonl"
NESTED = FIXTURES / "taxonomies" / "nested.json"
runner = CliRunner()


@pytest.fixture
def tax() -> Taxonomy:
    return Taxonomy.load(NESTED)


def test_exported_from_package():
    assert docrooter.evaluate is evaluate and docrooter.compare is compare


def test_evaluate_with_keyword_backend_end_to_end(tax):
    rep = evaluate(Classifier(tax, backend=KeywordBackend(), threshold=0), Dataset.load(SMALL))
    assert rep.n == 8 and 0 <= rep.hf1 <= 1 and set(rep.node_acc) >= {Taxonomy.ROOT_ID, "financial"}
    assert "| hF1 |" in rep.to_markdown()
    assert rep.device == "cpu" and rep.sections_per_sec > 0
    assert rep.meta["skipped_nodes"] == []
    assert 0 <= rep.macro_f1 <= 1 and set(rep.depth_acc) == {1, 2}
    assert "torch" not in sys.modules


def test_evaluate_without_node_decisions(tax):
    rep = evaluate(
        Classifier(tax, backend=KeywordBackend(), threshold=0),
        Dataset.load(SMALL),
        node_decisions=False,
    )
    assert rep.node_acc == {} and rep.confusions == {}


def test_evaluate_counts_decide_calls_once_per_node(tax):
    clf = Classifier(tax, backend=KeywordBackend(), threshold=0)
    calls = []
    real = clf.decide

    def spy(texts, node_id):
        calls.append((node_id, len(texts)))
        return real(texts, node_id)

    clf.decide = spy
    evaluate(clf, Dataset.load(SMALL))
    assert sorted(calls) == sorted([("financial", 4), ("legal", 4), (Taxonomy.ROOT_ID, 8)])


def test_skipped_nodes_are_listed(tax):
    clf = Classifier(tax, backend=KeywordBackend(), threshold=0)
    real = clf.decide

    def decide(texts, node_id):
        if node_id == "legal":
            raise BackendError("nothing can decide here")
        return real(texts, node_id)

    clf.decide = decide
    rep = evaluate(clf, Dataset.load(SMALL))
    assert rep.meta["skipped_nodes"] == ["legal"]
    assert "legal" not in rep.node_acc and "financial" in rep.node_acc


def test_confusions_report_gold_and_predicted_children():
    nodes = [
        TaxonomyNode(id="a", label="Alpha"),
        TaxonomyNode(id="b", label="Beta"),
    ]
    tax = Taxonomy(nodes)
    # no text shares a word with a label, so the keyword backend sees all-zero scores and picks
    # the first child: "a" is always predicted, and every "b" example is confused with it
    ds = Dataset([Example(text="zzz", path=["a"]), Example(text="qqq", path=["b"])] * 2)
    rep = evaluate(Classifier(tax, backend=KeywordBackend(), threshold=0), ds)
    assert rep.node_acc[Taxonomy.ROOT_ID] == 0.5
    assert rep.confusions == {Taxonomy.ROOT_ID: [("b", "a", 2)]}


def test_evaluate_empty_dataset(tax):
    rep = evaluate(Classifier(tax, backend=KeywordBackend()), Dataset([]))
    assert rep.n == 0 and rep.exact == 0.0 and rep.sections_per_sec == 0.0


def test_device_is_selected_only_for_model_backends(tax, monkeypatch):
    import docrooter.backends.loading as loading

    monkeypatch.setattr(loading, "select_device", lambda *a, **k: "mps")
    clf = Classifier(tax, backend=KeywordBackend(), threshold=0)
    assert evaluate(clf, Dataset.load(SMALL)).device == "cpu"
    # a model spec in the config is enough to report the device, even if routing never uses it
    clf = Classifier(tax, backend=KeywordBackend(), fallback="embedding:some/model", threshold=0)
    assert evaluate(clf, Dataset.load(SMALL), node_decisions=False).device == "mps"


def test_report_json_roundtrip_and_print(tax, capsys):
    rep = evaluate(Classifier(tax, backend=KeywordBackend(), threshold=0), Dataset.load(SMALL))
    again = Report.model_validate_json(rep.to_json())
    assert again == rep
    rep.print()
    out = capsys.readouterr().out
    assert "hF1" in out and "financial" in out


def test_print_keeps_bracketed_ids_literal(capsys):
    tax = Taxonomy([TaxonomyNode(id="[red]a", label="A"), TaxonomyNode(id="[red]b", label="B")])
    ds = Dataset([Example(text="x", path=["[red]a"]), Example(text="y", path=["[red]b"])])
    evaluate(Classifier(tax, backend=KeywordBackend(), threshold=0), ds).print()
    assert "[red]a" in capsys.readouterr().out


def test_compare_prints_one_column_per_classifier(tax, capsys):
    ds = Dataset.load(SMALL)
    cmp = compare(
        {
            "kw": Classifier(tax, backend=KeywordBackend(), threshold=0),
            "kw-strict": Classifier(tax, backend=KeywordBackend(), threshold=0.99),
        },
        ds,
    )
    md = cmp.to_markdown()
    assert md.splitlines()[0] == "| Metric | kw | kw-strict |"
    assert "| hF1 |" in md
    cmp.print()
    out = capsys.readouterr().out
    assert "kw-strict" in out and "hF1" in out


def test_cli_eval_formats():
    base = ["eval", "-t", str(NESTED), "-d", str(SMALL), "-b", "keyword", "--threshold", "0"]
    table = runner.invoke(app, base)
    assert table.exit_code == 0, table.output
    assert "hF1" in table.output
    md = runner.invoke(app, [*base, "--format", "md"])
    assert md.exit_code == 0 and "| hF1 |" in md.output
    js = runner.invoke(app, [*base, "--format", "json", "--strategy", "greedy"])
    assert js.exit_code == 0, js.output
    assert json.loads(js.output)["n"] == 8


def test_cli_eval_bad_format_and_missing_extra():
    base = ["eval", "-t", str(NESTED), "-d", str(SMALL), "-b", "keyword"]
    assert runner.invoke(app, [*base, "--format", "xml"]).exit_code != 0
    bad = runner.invoke(app, ["eval", "-t", str(NESTED), "-d", str(SMALL), "-b", "nope:x"])
    assert bad.exit_code == 1 and "error:" in bad.output
