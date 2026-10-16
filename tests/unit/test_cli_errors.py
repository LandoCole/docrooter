"""Every command reports expected failures as ``error: ...`` with exit status 1."""

from pathlib import Path

from typer.testing import CliRunner

from docrooter.cli import app
from docrooter.document import pdf as pdf_module
from docrooter.errors import MissingExtraError

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
DOC = str(FIXTURES / "docs" / "simple.md")
TAX = str(FIXTURES / "taxonomies" / "nested.json")
runner = CliRunner()


def _assert_error(result, *fragments: str) -> None:
    assert result.exit_code == 1, result.output
    assert "error: " in result.output
    assert "Traceback" not in result.output
    flat = " ".join(result.output.split())  # rich may wrap long lines
    for fragment in fragments:
        assert fragment in flat, flat


def test_toc_pdf_without_the_pdf_extra(monkeypatch, tmp_path):
    def missing(module, extra):
        raise MissingExtraError(f"{module} is required: pip install 'docrooter[{extra}]'")

    monkeypatch.setattr(pdf_module, "require", missing)
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    result = runner.invoke(app, ["toc", str(pdf), "--engine", "pymupdf4llm"])
    _assert_error(result, "pip install 'docrooter[pdf]'")


def test_classify_openai_without_a_model():
    result = runner.invoke(app, ["classify", DOC, "-t", TAX, "-b", "openai"])
    _assert_error(result, "openai needs a model: openai:<model>")


def test_classify_out_into_a_missing_directory(tmp_path):
    out = tmp_path / "missing" / "out.json"
    result = runner.invoke(app, ["classify", DOC, "-t", TAX, "-b", "keyword", "-o", str(out)])
    _assert_error(result, "missing")
    assert not out.exists()


def test_error_text_is_not_read_as_markup(monkeypatch, tmp_path):
    def missing(module, extra):
        raise MissingExtraError("[bold]not markup[/bold] pip install 'docrooter[pdf]'")

    monkeypatch.setattr(pdf_module, "require", missing)
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    result = runner.invoke(app, ["toc", str(pdf), "--engine", "pymupdf4llm"])
    _assert_error(result, "[bold]not markup[/bold]", "docrooter[pdf]")


def test_eval_flat_with_a_trained_spec_is_a_value_error(tmp_path):
    data = tmp_path / "d.jsonl"
    data.write_text('{"text": "t", "path": ["financial"]}\n', encoding="utf-8")
    result = runner.invoke(
        app, ["eval", "-t", TAX, "-d", str(data), "-b", "trained:x", "--strategy", "flat"]
    )
    _assert_error(result, "Flat strategy needs a zero-shot backend")
