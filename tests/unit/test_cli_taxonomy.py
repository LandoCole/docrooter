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


def test_cli_taxonomy_validate_shows_bracketed_labels_verbatim(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text(
        "id,parent_id,label\n"
        "a,,Notes [see appendix]\n"
        "b,,notes [see appendix]\n"
        "c,,Odd [/x] label\n"
        "d,,odd [/x] label\n"
    )
    result = runner.invoke(app, ["taxonomy", "validate", str(p)], env={"COLUMNS": "300"})
    assert result.exit_code == 1, result.output
    assert "label 'notes [see appendix]' duplicates" in result.output
    assert "label 'odd [/x] label' duplicates" in result.output
    assert "duplicate_sibling_label" in result.output


def test_cli_taxonomy_validate_json_with_root_settings():
    result = runner.invoke(app, ["taxonomy", "validate", str(FIX / "root_settings.json")])
    assert result.exit_code == 0, result.output
    assert "valid" in result.output


def test_cli_taxonomy_validate_invalid_root(tmp_path):
    p = tmp_path / "t.json"
    p.write_text('{"root": {"colour": "red"}, "nodes": [{"id": "a", "label": "A"}]}')
    result = runner.invoke(app, ["taxonomy", "validate", str(p)], env={"COLUMNS": "200"})
    assert result.exit_code == 1 and "invalid_root" in result.output
