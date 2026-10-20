"""``docrooter classify FILE``: the v1 flows from the command line (keyword backend)."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from docrooter import Classification
from docrooter.cli import app

runner = CliRunner()
WIDE = {"COLUMNS": "400"}

REPORT = """# Notes to the financial statements

Note one sets out the accounting policies and revenue recognition.

# Management discussion

Results of operations improved as revenue grew.

## Liquidity

Cash flows from operations funded capital expenditure.
"""


@pytest.fixture
def report(tmp_path):
    path = tmp_path / "report.md"
    path.write_text(REPORT, encoding="utf-8")
    return path


def test_cli_classify_json(report, tmp_path):
    log = tmp_path / "log.jsonl"
    args = ["classify", str(report), "-b", "keyword", "--json", "--level", "headings"]
    result = runner.invoke(app, [*args, "--log", str(log)])
    assert result.exit_code == 0, result.output
    parsed = Classification.model_validate_json(result.stdout)
    assert parsed.flow == "sections_first" and parsed.backend == "keyword"
    assert parsed.levels["section"] == "headings"
    assert [t.title for t in parsed.sections] == [
        "Notes to the financial statements",
        "Management discussion",
        "Liquidity",
    ]
    assert not parsed.tokens.exact and parsed.tokens.prompt > 0  # keyword: estimated
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    section_rows = [r for r in rows if r["stage"] == "sections"]
    assert len(section_rows) == 2 * len(parsed.sections)  # keyword asks per item: group, leaf
    assert all(r["level"] == "headings" for r in rows)

    doc_first = runner.invoke(
        app,
        ["classify", str(report), "-b", "keyword", "--json", "--flow", "doc_first",
         "--doc-level", "lead", "--section-level", "headings"],
    )  # fmt: skip
    assert doc_first.exit_code == 0, doc_first.output
    parsed = Classification.model_validate_json(doc_first.stdout)
    assert parsed.flow == "doc_first" and parsed.decided_by == "model"
    assert parsed.levels == {"section": "headings", "document": "lead"}


def test_cli_classify_calls_option(report):
    for calls, used in (("per_item", "per_item"), ("batched_flat", "per_item")):
        args = ["classify", str(report), "-b", "keyword", "--json", "--calls", calls]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        parsed = Classification.model_validate_json(result.stdout)
        assert parsed.calls == used  # keyword scores per item; the pattern sets the estimate
    flat = Classification.model_validate_json(
        runner.invoke(app, [*args[:-1], "batched_flat"]).stdout
    )
    per_item = Classification.model_validate_json(
        runner.invoke(app, [*args[:-1], "per_item"]).stdout
    )
    assert flat.tokens.calls < per_item.tokens.calls


def test_cli_classify_human_output(report):
    result = runner.invoke(app, ["classify", str(report), "-b", "keyword"], env=WIDE)
    assert result.exit_code == 0, result.output
    out = result.stdout
    assert "doc type:" in out and "decided by" in out
    assert "candidates:" in out
    assert "Notes to the financial statements" in out and "Confidence" in out
    assert "tokens (estimated):" in out


@pytest.mark.parametrize(
    ("args", "hint"),
    [
        (["--flow", "vibes"], "--flow"),
        (["--level", "auto"], "--level"),
        (["--doc-level", "all"], "--doc-level"),
        (["--strategy", "greedy"], "--strategy"),
        (["--calls", "sometimes"], "--calls"),
        (["-t", "TAX", "--level", "terms"], "--level"),
    ],
)
def test_cli_classify_rejects_options_that_do_not_apply(report, tmp_path, args, hint):
    taxonomy = tmp_path / "tax.json"
    taxonomy.write_text(json.dumps({"nodes": [{"id": "a", "label": "A"}]}), encoding="utf-8")
    args = [str(taxonomy) if a == "TAX" else a for a in args]
    result = runner.invoke(app, ["classify", str(report), "-b", "keyword", *args], env=WIDE)
    assert result.exit_code == 2
    assert hint in result.output


def text_answers():
    """A Classification whose confidences are text answers (an endpoint without logprobs)."""
    return Classification.model_validate(
        {
            "source": "court.md",
            "flow": "sections_first",
            "doc_type": "legal.court_opinion",
            "doc_type_confidence": None,
            "decided_by": "rules",
            "candidates": ["legal.court_opinion"],
            "sections": [
                {
                    "id": "s1",
                    "title": "Opinion",
                    "level": 1,
                    "label": "law.opinion",
                    "confidence": None,
                    "level_used": "terms",
                    "confidence_source": "text",
                }
            ],
            "backend": "openai:test",
        }
    )


def test_cli_prints_text_answers_without_a_confidence(capsys):
    from docrooter.cli import _print_classification

    _print_classification(text_answers())
    out = capsys.readouterr().out
    assert "Court opinion" in out and "no probability" in out
    assert "law.opinion" in out and "letter" in out


@pytest.mark.parametrize(
    ("args", "phrase"),
    [
        (["--strategy", "greedy"], "--strategy only applies with"),
        (["--strategy", "greedy", "--threshold", "0.4"], "only apply with"),
        (["-t", "TAX", "--level", "terms"], "--level does not apply with"),
        (["-t", "TAX", "--level", "terms", "--json"], "do not apply with"),
    ],
)
def test_cli_usage_errors_agree_in_number(report, tmp_path, args, phrase):
    taxonomy = tmp_path / "tax.json"
    taxonomy.write_text(json.dumps({"nodes": [{"id": "a", "label": "A"}]}), encoding="utf-8")
    args = [str(taxonomy) if a == "TAX" else a for a in args]
    result = runner.invoke(app, ["classify", str(report), "-b", "keyword", *args], env=WIDE)
    assert result.exit_code == 2
    assert phrase in " ".join(result.output.split())
