from pathlib import Path

from typer.testing import CliRunner

from docrooter.cli import app

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "taxonomies"
runner = CliRunner()


def test_cli_taxonomy_validate_ok():
    result = runner.invoke(app, ["taxonomy", "validate", str(FIX / "nested.json")])
    assert result.exit_code == 0, result.output
    assert "valid" in result.output and "6 nodes" in result.output


def test_cli_taxonomy_validate_exit_code():
    result = runner.invoke(app, ["taxonomy", "validate", str(FIX / "broken.csv")])
    assert result.exit_code == 1
    for token in ("missing_parent", "cycle", "row 4", "error"):
        assert token in result.output


def test_cli_taxonomy_validate_warnings_still_exit_zero(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("id,parent_id,label\na,,A\nb,a,B\n")
    result = runner.invoke(app, ["taxonomy", "validate", str(p)])
    assert result.exit_code == 0, result.output
    assert "valid" in result.output and "single_child" in result.output


def test_cli_taxonomy_validate_unknown_format(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("foo,bar\n1,2\n")
    result = runner.invoke(app, ["taxonomy", "validate", str(p)])
    assert result.exit_code == 1 and "unknown_csv_format" in result.output


def test_cli_taxonomy_validate_missing_file():
    result = runner.invoke(app, ["taxonomy", "validate", str(FIX / "nope.csv")])
    assert result.exit_code != 0
