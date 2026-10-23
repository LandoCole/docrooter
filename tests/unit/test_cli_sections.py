"""``docrooter sections SOURCE``: section text by tag from the command line (model-free)."""

from __future__ import annotations

import json

import pytest
from test_context import BALANCE, TENK, build
from typer.testing import CliRunner

from docrooter.cli import app
from docrooter.context import TaggedDocument

runner = CliRunner()
WIDE = {"COLUMNS": "400"}
SCHEDULE = "financial.schedule_of_investments"


@pytest.fixture
def tagged():
    document, classification = build(TENK)
    return TaggedDocument(document=document, classification=classification)


@pytest.fixture
def saved(tagged, tmp_path):
    path = tmp_path / "saved.json"
    tagged.save(path)
    return path


def test_sections_from_saved_json_tag(saved):
    result = runner.invoke(app, ["sections", str(saved), "--tag", BALANCE, "--tag", SCHEDULE])
    assert result.exit_code == 0, result.output
    out = json.loads(result.stdout)
    assert out["doc_type"] == "financial.annual_report"
    assert out["source"] == "doc.pdf"
    assert [r["status"] for r in out["results"]] == ["found", "not_found"]
    assert out["results"][0]["sections"][0]["heading"] == "CONSOLIDATED BALANCE SHEETS"


def test_sections_list_is_default(saved, tagged):
    result = runner.invoke(app, ["sections", str(saved)])
    assert result.exit_code == 0, result.output
    out = json.loads(result.stdout)
    assert out["tags"] == tagged.available_tags()
    assert "results" not in out

    listed = runner.invoke(app, ["sections", str(saved), "--list"])
    assert json.loads(listed.stdout) == out

    every = runner.invoke(app, ["sections", str(saved), "--all"])
    assert every.exit_code == 0, every.output
    results = json.loads(every.stdout)["results"]
    assert [r["tag"] for r in results] == tagged.available_tags()


def test_sections_min_confidence(saved):
    result = runner.invoke(app, ["sections", str(saved), "--all", "--min-confidence", "0.95"])
    assert result.exit_code == 0, result.output
    found = {
        r["tag"]: [m["confidence"] for m in r["sections"]]
        for r in json.loads(result.stdout)["results"]
        if r["status"] == "found"
    }
    assert found == {BALANCE: [0.97], "disclosure.risk_factors": [None]}  # unknown is kept


def test_sections_document_source_classifies_and_saves(tagged, tmp_path, monkeypatch):
    seen = {}

    def fake(source, **kwargs):
        seen["source"], seen["kwargs"] = source, kwargs
        return tagged

    monkeypatch.setattr(TaggedDocument, "from_source", staticmethod(fake))
    doc = tmp_path / "report.md"
    doc.write_text("# Title\n\nBody.\n", encoding="utf-8")
    out_path = tmp_path / "out.json"
    result = runner.invoke(
        app,
        ["sections", str(doc), "--save", str(out_path), "-b", "jev", "--flow", "doc_first"],
    )
    assert result.exit_code == 0, result.output
    assert seen["source"] == doc
    assert seen["kwargs"] == {"backend": "jev", "flow": "doc_first"}  # no level: not given
    assert json.loads(result.stdout)["tags"] == tagged.available_tags()
    assert TaggedDocument.load(out_path).available_tags() == tagged.available_tags()


def test_sections_rejects_tag_with_all(saved):
    result = runner.invoke(app, ["sections", str(saved), "--tag", "x", "--all"], env=WIDE)
    assert result.exit_code != 0
    assert "--tag" in result.output and "--all" in result.output


def test_sections_rejects_save_for_a_json_source(saved, tmp_path):
    out_path = tmp_path / "again.json"
    result = runner.invoke(app, ["sections", str(saved), "--save", str(out_path)], env=WIDE)
    assert result.exit_code != 0
    assert "--save" in result.output
    assert not out_path.exists()
