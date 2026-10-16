from pathlib import Path

import pytest

import docrooter.document.pdf as pdf_module
from docrooter import Document, Section, load
from docrooter._extras import MissingExtraError

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "docs"


def test_load_md_path_equals_text():
    path = FIXTURES / "simple.md"
    assert load(path).toc() == load(path.read_text(encoding="utf-8")).toc()


def test_load_str_path_and_source():
    path = FIXTURES / "simple.md"
    doc = load(str(path))
    assert doc.source == str(path)
    assert load("# A\nbody").source == "<string>"


def test_load_title_from_h1_else_stem(tmp_path):
    assert load(FIXTURES / "simple.md").title == "Intro"
    plain = tmp_path / "notes.txt"
    plain.write_text("just text\n", encoding="utf-8")
    assert load(plain).title == "notes"
    assert load("just text\nmore").title is None


def test_load_runs_validation_and_repair():
    text = "# A\nbody\n### Deep\nx\n# Page 3\ny\n"
    doc = load(text)
    assert [e.title for e in doc.toc()] == ["A", "Deep"]
    assert doc.section("Deep").level == 2
    assert "level_repaired" in doc.section("Deep").flags
    raw = load(text, repair=False, min_heading_score=0.0)
    assert [e.title for e in raw.toc()] == ["A", "Deep", "Page 3"]
    assert raw.section("Deep").level == 3


def test_load_missing_known_extension_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(str(tmp_path / "nope.md"))
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "nope.pdf")


def test_load_single_line_text_is_markdown():
    doc = load("# Only heading")
    assert [e.title for e in doc.toc()] == ["Only heading"]


def test_load_unknown_path_object_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "missing")


def test_load_exports():
    assert isinstance(load("# A\nb"), Document)
    assert isinstance(load("# A\nb").section("s1"), Section)


def test_load_pdf_without_extra(monkeypatch, tmp_path):
    def boom(module, extra):
        raise MissingExtraError(f"pip install 'docrooter[{extra}]'")

    monkeypatch.setattr(pdf_module, "require", boom)
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    with pytest.raises(MissingExtraError, match=r"docrooter\[pdf\]"):
        load(pdf)
