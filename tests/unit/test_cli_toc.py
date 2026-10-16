import json
from pathlib import Path

from typer.testing import CliRunner

from docrooter.cli import app

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "docs"
runner = CliRunner()


def test_cli_toc_json():
    result = runner.invoke(app, ["toc", str(FIXTURES / "simple.md"), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data[0] == {"id": "s1", "title": "Intro", "level": 1, "depth": 0}
    assert [e["title"] for e in data] == ["Intro", "Background", "Scope", "Results"]


def test_cli_toc_tree():
    result = runner.invoke(app, ["toc", str(FIXTURES / "simple.md")])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert "Intro" in lines
    assert "  Scope" in lines
    assert "    Scope" not in lines


def test_cli_toc_missing_file():
    result = runner.invoke(app, ["toc", str(FIXTURES / "nope.md")])
    assert result.exit_code != 0


def test_cli_no_args_shows_help():
    result = runner.invoke(app, [])
    assert "toc" in result.output
