"""``docrooter parse FILE``: the document, its type and its sections as JSON (keyword backend)."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from docrooter import ParseResult
from docrooter.cli import app

runner = CliRunner()
WIDE = {"COLUMNS": "400"}

REPORT = """# Notes to the financial statements

Note one sets out the accounting policies and revenue recognition. Café résumé.

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


def _parse(report, *args):
    return runner.invoke(app, ["parse", str(report), "-b", "keyword", *args], env=WIDE)


def test_parse_cli_prints_json_at_each_level(report):
    default = _parse(report)
    assert default.exit_code == 0, default.output
    assert ParseResult.model_validate_json(default.stdout).include == "sections"

    only_type = _parse(report, "--include", "type")
    assert only_type.exit_code == 0, only_type.output
    parsed = ParseResult.model_validate_json(only_type.stdout)
    assert parsed.include == "type" and parsed.backend == "keyword"
    assert "sections" not in json.loads(only_type.stdout)
    assert "section_text" not in json.loads(only_type.stdout)

    sections = _parse(report, "--include", "sections", "--level", "headings")
    assert sections.exit_code == 0, sections.output
    data = json.loads(sections.stdout)
    assert len(data["sections"]) == 3 and "section_text" not in data

    text = _parse(report, "--include", "text", "--flow", "sections_first")
    assert text.exit_code == 0, text.output
    parsed = ParseResult.model_validate_json(text.stdout)
    assert parsed.sections is not None and parsed.section_text is not None
    assert "Café résumé" in text.stdout  # ensure_ascii is off


def test_parse_cli_out_writes_file(report, tmp_path):
    out = tmp_path / "parsed.json"
    result = _parse(report, "--include", "text", "--out", str(out))
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == f"wrote {out}"
    written = out.read_text(encoding="utf-8")
    assert "Café résumé" in written
    assert ParseResult.model_validate_json(written).include == "text"


def test_parse_cli_rejects_bad_include(report):
    result = _parse(report, "--include", "everything")
    assert result.exit_code != 0
    assert "everything" in result.output and "type, sections, text" in result.output


def test_parse_cli_type_with_sections_first_is_an_error(report):
    result = _parse(report, "--include", "type", "--flow", "sections_first")
    assert result.exit_code == 2
    assert "doc_first" in result.output


def test_parse_cli_rejects_bad_flow_and_level(report):
    assert _parse(report, "--flow", "sideways").exit_code == 2
    assert _parse(report, "--level", "everything").exit_code == 2
