"""``load_pdf`` on small generated PDFs: engines, timings, meta, no-text and encrypted files."""

import pytest
from typer.testing import CliRunner

from docrooter.cli import app
from docrooter.errors import DocrooterError, MissingExtraError
from docrooter.pdf.load import load_pdf

pymupdf = pytest.importorskip("pymupdf")
runner = CliRunner()

BODY = "This sentence is ordinary body text that stays small and keeps the body size at eleven."


def _write(doc, path):
    doc.save(str(path))
    doc.close()
    return path


def _page(doc, items):
    page = doc.new_page(width=612, height=792)
    y = 90
    for kind, text in items:
        if kind == "h1":
            page.insert_text((72, y), text, fontsize=22, fontname="hebo")
            y += 40
        elif kind == "h2":
            page.insert_text((72, y), text, fontsize=16, fontname="hebo")
            y += 32
        else:
            for k in range(4):
                page.insert_text((72, y), f"{text} Line {k}.", fontsize=11, fontname="helv")
                y += 15
            y += 22
    return page


@pytest.fixture
def toc_pdf(tmp_path):
    doc = pymupdf.open()
    _page(doc, [("p", "Preface words."), ("h1", "Introduction"), ("p", BODY),
                ("h2", "Background"), ("p", BODY)])  # fmt: skip
    _page(doc, [("h1", "Methods"), ("p", BODY), ("h2", "Data Sources"), ("p", BODY)])
    _page(doc, [("h1", "Results"), ("p", BODY)])
    return _write(doc, tmp_path / "toc.pdf")


def test_native_toc_levels_pages_and_meta(toc_pdf):
    doc = load_pdf(toc_pdf, engine="pymupdf")
    assert [(e.title, e.level, e.depth) for e in doc.toc()] == [
        ("Introduction", 1, 0),
        ("Background", 2, 1),
        ("Methods", 1, 0),
        ("Data Sources", 2, 1),
        ("Results", 1, 0),
    ]
    pages = {s.title: s.page_start for s in doc.walk()}
    assert pages["Introduction"] == 1 and pages["Methods"] == 2 and pages["Results"] == 3
    assert doc.title == "Introduction" and doc.source == str(toc_pdf)
    assert doc.meta["engine"] == "pymupdf" and doc.meta["n_pages"] == 3
    assert set(doc.meta["timings"]) == {"extract", "signals", "headings", "assemble"}
    assert all(isinstance(v, float) and v >= 0 for v in doc.meta["timings"].values())
    assert doc.meta["bookmarks"] == 0 and doc.meta["printed_toc_entries"] == 0
    assert doc.meta["body_size"] == 11.0 and doc.meta["warnings"] == []


def test_body_text_and_preamble(toc_pdf):
    doc = load_pdf(toc_pdf, engine="pymupdf")
    preamble = doc.sections["s0"]
    assert preamble.level == 0 and preamble.text.startswith("Preface words. Line 0.")
    assert "Introduction" not in preamble.text
    intro = doc.section("Introduction")
    assert intro.text.splitlines()[0] == f"{BODY} Line 0."
    assert "Background" not in intro.text and "Methods" not in intro.text
    assert doc.section("Background").text.count("\n") == 3
    assert doc.section("Results").text.startswith(BODY)


def test_preamble_is_omitted_when_empty(tmp_path):
    pdf = pymupdf.open()
    _page(pdf, [("h1", "Alpha"), ("p", BODY), ("h1", "Beta"), ("p", BODY)])
    doc = load_pdf(_write(pdf, tmp_path / "a.pdf"), engine="pymupdf")
    assert "s0" not in doc.sections and doc.root_ids == ["s1", "s2"]


def test_headers_footers_and_page_numbers_leave_the_body(tmp_path):
    pdf = pymupdf.open()
    for n in range(1, 5):
        page = _page(pdf, [("h1", f"Chapter {n}"), ("p", BODY)])
        page.insert_text((72, 40), "ACME Annual Report", fontsize=9, fontname="helv")
        page.insert_text((300, 760), str(n), fontsize=9, fontname="helv")
    doc = load_pdf(_write(pdf, tmp_path / "h.pdf"), engine="pymupdf")
    assert [e.title for e in doc.toc()] == [f"Chapter {n}" for n in range(1, 5)]
    assert all("ACME" not in s.text for s in doc.walk())


def test_bookmarks_set_levels_and_are_counted(tmp_path):
    pdf = pymupdf.open()
    _page(pdf, [("h1", "Alpha"), ("p", BODY), ("h1", "Beta"), ("p", BODY)])
    pdf.set_toc([[1, "Alpha", 1], [2, "Beta", 1]])
    path = _write(pdf, tmp_path / "b.pdf")
    doc = load_pdf(path, engine="pymupdf")
    assert [(e.title, e.level) for e in doc.toc()] == [("Alpha", 1), ("Beta", 2)]
    assert doc.meta["bookmarks"] == 2
    assert "level_from_bookmark" in doc.section("Beta").flags
    plain = load_pdf(path, engine="pymupdf", use_bookmarks=False)
    assert [e.level for e in plain.toc()] == [1, 1]


def test_title_falls_back_to_metadata_then_stem(tmp_path):
    pdf = pymupdf.open()
    page = pdf.new_page()
    for k in range(6):
        page.insert_text((72, 100 + 15 * k), f"just body text number {k}", fontsize=11)
    pdf.set_metadata({"title": "The Real Title"})
    path = _write(pdf, tmp_path / "meta.pdf")
    doc = load_pdf(path, engine="pymupdf")
    assert doc.title == "The Real Title"

    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 100), "just body text", fontsize=11)
    pdf.set_metadata({"title": "Microsoft Word - draft.docx"})
    doc = load_pdf(_write(pdf, tmp_path / "stem.pdf"), engine="pymupdf")
    assert doc.title == "stem"


def test_image_only_pdf_is_a_single_empty_section(tmp_path):
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.draw_rect(pymupdf.Rect(100, 100, 300, 300), fill=(0.5, 0.5, 0.5))
    path = _write(pdf, tmp_path / "scan.pdf")
    doc = load_pdf(path)
    assert list(doc.sections) == ["s0"] and doc.sections["s0"].text == ""
    assert doc.toc() == []
    assert doc.meta["warnings"] == ["no_text_layer"] and doc.meta["n_pages"] == 1


def test_encrypted_pdf_raises_docrooter_error(tmp_path):
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 100), "secret", fontsize=11)
    path = tmp_path / "locked.pdf"
    pdf.save(str(path), encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="x", owner_pw="y")
    pdf.close()
    with pytest.raises(DocrooterError, match="locked.pdf"):
        load_pdf(path)


def test_broken_pdf_raises_docrooter_error(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4\nnot really a pdf")
    with pytest.raises(DocrooterError, match="broken.pdf"):
        load_pdf(path)


def test_unknown_engine(toc_pdf):
    with pytest.raises(DocrooterError, match="engine"):
        load_pdf(toc_pdf, engine="nope")


def test_auto_falls_back_to_pdfplumber_then_errors(monkeypatch, toc_pdf):
    pytest.importorskip("pdfplumber")
    import docrooter.pdf.lines as lines_module

    def missing():
        raise MissingExtraError("pymupdf is required: pip install 'docrooter[pdf]'")

    monkeypatch.setattr(lines_module, "_import_pymupdf", missing)
    doc = load_pdf(toc_pdf)
    assert doc.meta["engine"] == "pdfplumber"
    assert [e.title for e in doc.toc()][:2] == ["Introduction", "Background"]

    monkeypatch.setattr(lines_module, "_import_pdfplumber", missing)
    with pytest.raises(MissingExtraError, match=r"docrooter\[pdf\]"):
        load_pdf(toc_pdf)


def test_engine_both_is_a_subset_of_pymupdf(toc_pdf):
    pytest.importorskip("pdfplumber")
    ref = load_pdf(toc_pdf, engine="pymupdf")
    both = load_pdf(toc_pdf, engine="both")
    assert both.meta["engine"] == "both" and both.meta["n_pages"] == 3
    assert {e.title for e in both.toc()} <= {e.title for e in ref.toc()}
    assert [e.title for e in both.toc()] == [e.title for e in ref.toc()]
    assert both.section("Introduction").text == ref.section("Introduction").text


def test_engine_both_drops_low_scoring_single_engine_headings():
    from docrooter.pdf.headings import HeadingCandidate
    from docrooter.pdf.load import _merge_engines

    def cand(i, text, page, score, level=1):
        return HeadingCandidate(
            line_index=i, text=text, page=page, size=12, bold=True, score=score, level=level
        )

    primary = [cand(1, "Shared Heading", 1, 0.6), cand(5, "Only Here Low", 1, 0.6),
               cand(9, "Only Here High", 2, 0.85)]  # fmt: skip
    other = [cand(2, "shared heading", 1, 0.6), cand(7, "Elsewhere", 2, 0.6)]
    kept = _merge_engines(primary, other, [], [])
    assert [h.text for h in kept] == ["Shared Heading", "Only Here High"]


def test_pymupdf4llm_engine_is_the_legacy_path(toc_pdf):
    pytest.importorskip("pymupdf4llm")
    doc = load_pdf(toc_pdf, engine="pymupdf4llm")
    assert doc.meta["engine"] == "pymupdf4llm" and doc.meta["n_pages"] == 3
    assert doc.toc()


def test_cli_toc_engine_pymupdf(toc_pdf):
    result = runner.invoke(app, ["toc", str(toc_pdf), "--engine", "pymupdf"])
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        "Introduction", "  Background", "Methods", "  Data Sources", "Results",
    ]  # fmt: skip
    assert "3 pages, 5 sections" in result.stderr and "engine pymupdf" in result.stderr


def test_cli_toc_json_has_no_summary(toc_pdf):
    result = runner.invoke(app, ["toc", str(toc_pdf), "--engine", "pymupdf", "--json"])
    assert result.exit_code == 0, result.output
    assert result.stderr == "" and result.stdout.lstrip().startswith("[")


def test_cli_toc_rejects_unknown_engine(toc_pdf):
    result = runner.invoke(app, ["toc", str(toc_pdf), "--engine", "nope"])
    assert result.exit_code != 0


def test_cli_toc_encrypted_pdf_is_an_error_message(tmp_path):
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 100), "secret", fontsize=11)
    path = tmp_path / "locked.pdf"
    pdf.save(str(path), encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="x", owner_pw="y")
    pdf.close()
    result = runner.invoke(app, ["toc", str(path)])
    assert result.exit_code == 1 and "error: " in result.output and "Traceback" not in result.output
