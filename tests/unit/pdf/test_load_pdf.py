"""``load_pdf``'s section tree (PyMuPDF4LLM) on small generated PDFs: levels, cleaning, bookmarks,
titles, meta, no-text and encrypted files."""

import pytest
from typer.testing import CliRunner

from docrooter.cli import app
from docrooter.errors import DocrooterError
from docrooter.pdf.load import load_pdf

pymupdf = pytest.importorskip("pymupdf")
pymupdf4llm = pytest.importorskip("pymupdf4llm")
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


def test_section_tree_levels_pages_and_meta(toc_pdf):
    doc = load_pdf(toc_pdf)
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
    assert doc.meta["engine"] == "pymupdf4llm" and doc.meta["n_pages"] == 3
    assert set(doc.meta["timings"]) == {"extract", "headings", "assemble"}
    assert all(isinstance(v, float) and v >= 0 for v in doc.meta["timings"].values())
    assert doc.meta["bookmarks"] == 0 and doc.meta["warnings"] == []


def test_body_text_and_preamble(toc_pdf):
    doc = load_pdf(toc_pdf)
    preamble = doc.sections["s0"]
    assert preamble.level == 0 and preamble.text.startswith("Preface words. Line 0.")
    assert "Introduction" not in preamble.text
    intro = doc.section("Introduction")
    assert intro.text.startswith(f"{BODY} Line 0.")
    assert "Background" not in intro.text and "Methods" not in intro.text
    assert doc.section("Results").text.startswith(BODY)


@pytest.mark.parametrize("engine", ["auto", "pymupdf", "pymupdf4llm"])
def test_every_section_tree_engine_reads_with_pymupdf4llm(toc_pdf, engine):
    assert load_pdf(toc_pdf, engine=engine).meta["engine"] == "pymupdf4llm"


def test_pdfplumber_is_element_mode_only(toc_pdf):
    with pytest.raises(DocrooterError, match="element mode"):
        load_pdf(toc_pdf, engine="pdfplumber")


def test_element_mode_options_are_rejected_without_it(toc_pdf):
    with pytest.raises(TypeError, match="elements=True"):
        load_pdf(toc_pdf, min_heading_score=0.4)
    with pytest.raises(TypeError, match="elements=True"):
        load_pdf(toc_pdf, use_printed_toc=False)
    with pytest.raises(TypeError, match="elements=True"):
        load_pdf(toc_pdf, ocr="tesseract")


def _fake_markdown(monkeypatch, pages, seen=None):
    """Make pymupdf4llm.to_markdown return these page texts (and record its options)."""

    def to_markdown(path, **options):
        if seen is not None:
            seen.update(options)
        wanted = options.get("pages") or range(len(pages))
        return [{"text": pages[i], "metadata": {"page_number": i + 1}} for i in wanted]

    monkeypatch.setattr(pymupdf4llm, "to_markdown", to_markdown)


def test_titles_lose_markup_and_emptied_headings_join_the_text_before(monkeypatch, toc_pdf):
    seen = {}
    _fake_markdown(
        monkeypatch,
        [
            "# **<u>Highlights</u>**\n\nWhy it matters.\n\n###### **\u2022 Summary of 2021**\n\nok",
            "## **\u2014**\n\nstray text\n\n## _Results_\n\nNumbers.",
            "",
        ],
        seen,
    )
    doc = load_pdf(toc_pdf)
    assert [(e.title, e.level) for e in doc.toc()] == [
        ("Highlights", 1),
        ("Summary of 2021", 2),  # a level jump (1 to 6) is repaired
        ("Results", 2),
    ]
    assert "stray text" in doc.section("Summary of 2021").text  # the emptied "## —" heading
    assert seen["header"] is False and seen["footer"] is False and seen["use_ocr"] is False
    assert seen["page_chunks"] is True


def test_ocr_auto_lets_pymupdf4llm_ocr(monkeypatch, toc_pdf):
    seen = {}
    _fake_markdown(monkeypatch, ["# A\n\ntext", "", ""], seen)
    load_pdf(toc_pdf, ocr="auto")
    assert seen["use_ocr"] is True


def test_repair_false_keeps_pymupdf4llm_levels(monkeypatch, toc_pdf):
    _fake_markdown(monkeypatch, ["# A\n\nx\n\n#### B\n\ny", "", ""])
    assert [e.level for e in load_pdf(toc_pdf, repair=False).toc()] == [1, 4]
    assert [e.level for e in load_pdf(toc_pdf).toc()] == [1, 2]


def test_bookmarks_set_levels_and_are_counted(tmp_path):
    pdf = pymupdf.open()
    _page(pdf, [("h1", "Alpha"), ("p", BODY), ("h1", "Beta"), ("p", BODY)])
    pdf.set_toc([[1, "Alpha", 1], [2, "Beta", 1]])
    path = _write(pdf, tmp_path / "b.pdf")
    doc = load_pdf(path)
    assert [(e.title, e.level) for e in doc.toc()] == [("Alpha", 1), ("Beta", 2)]
    assert doc.meta["bookmarks"] == 2
    assert "level_from_bookmark" in doc.section("Beta").flags
    plain = load_pdf(path, use_bookmarks=False)
    assert [e.level for e in plain.toc()] == [1, 1] and plain.meta["bookmarks"] == 0


def test_title_falls_back_to_metadata_then_stem(monkeypatch, tmp_path):
    _fake_markdown(monkeypatch, ["just body text, no heading"])
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 100), "just body text", fontsize=11)
    pdf.set_metadata({"title": "The Real Title"})
    assert load_pdf(_write(pdf, tmp_path / "meta.pdf")).title == "The Real Title"

    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 100), "just body text", fontsize=11)
    pdf.set_metadata({"title": "Microsoft Word - draft.docx"})
    assert load_pdf(_write(pdf, tmp_path / "stem.pdf")).title == "stem"


def _cover_pdf(tmp_path, name, *, metadata_title=None):
    pdf = pymupdf.open()
    page = _page(pdf, [("h1", "A N N U A L R E P O R T"), ("p", BODY), ("h2", "Overview"),
                       ("p", BODY)])  # fmt: skip
    if metadata_title:
        pdf.set_metadata({"title": metadata_title})
    assert page is not None
    return _write(pdf, tmp_path / name)


def test_letter_spaced_heading_is_not_the_document_title(monkeypatch, tmp_path):
    _fake_markdown(monkeypatch, ["# A N N U A L R E P O R T\n\ntext\n\n## Overview\n\nmore"])
    doc = load_pdf(_cover_pdf(tmp_path, "cover.pdf", metadata_title="Acme 1997"))
    assert doc.toc()[0].title == "A N N U A L R E P O R T"  # still a section
    assert doc.title == "Acme 1997"
    assert load_pdf(_cover_pdf(tmp_path, "stemmed.pdf")).title == "stemmed"


def test_image_only_pdf_is_a_single_empty_section(tmp_path):
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.draw_rect(pymupdf.Rect(100, 100, 300, 300), fill=(0.5, 0.5, 0.5))
    doc = load_pdf(_write(pdf, tmp_path / "scan.pdf"))
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


def test_max_pages_skips_later_pages_and_warns(toc_pdf):
    doc = load_pdf(toc_pdf, max_pages=2)
    titles = [e.title for e in doc.toc()]
    assert titles == ["Introduction", "Background", "Methods", "Data Sources"]
    assert "Results" not in " ".join(s.text for s in doc.walk())
    assert doc.meta["n_pages"] == 3 and doc.meta["warnings"] == ["truncated_pages"]
    whole = load_pdf(toc_pdf, max_pages=3)
    assert whole.meta["warnings"] == [] and whole.toc()[-1].title == "Results"


def test_max_pages_drops_bookmarks_past_the_cap(toc_pdf, tmp_path):
    doc = pymupdf.open(str(toc_pdf))
    doc.set_toc([[1, "Introduction", 1], [1, "Methods", 2], [1, "Results", 3]])
    path = _write(doc, tmp_path / "bookmarked.pdf")
    assert load_pdf(path, max_pages=2).meta["bookmarks"] == 2


def test_max_pages_must_be_positive(toc_pdf):
    with pytest.raises(ValueError, match="max_pages"):
        load_pdf(toc_pdf, max_pages=0)


def test_cli_toc(toc_pdf):
    result = runner.invoke(app, ["toc", str(toc_pdf)])
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        "Introduction", "  Background", "Methods", "  Data Sources", "Results",
    ]  # fmt: skip
    assert "3 pages, 5 sections" in result.stderr and "engine pymupdf4llm" in result.stderr


def test_cli_toc_json_has_no_summary(toc_pdf):
    result = runner.invoke(app, ["toc", str(toc_pdf), "--json"])
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
