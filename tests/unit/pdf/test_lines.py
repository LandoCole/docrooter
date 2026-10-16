import math

import pytest

from docrooter.errors import DocrooterError
from docrooter.pdf import extract_lines, read_bookmarks

pymupdf = pytest.importorskip("pymupdf")


@pytest.fixture
def pdf_path(tmp_path):
    doc = pymupdf.open()
    for n in (1, 2):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 100), f"Chapter {n} Overview", fontsize=20, fontname="hebo")
        y = 140
        for k in range(3):
            page.insert_text((72, y), f"Body sentence {k} of page {n} stays small.",
                             fontsize=11, fontname="helv")  # fmt: skip
            y += 16
        page.insert_text((72, y + 40), "Isolated paragraph after a gap.", fontsize=11,
                         fontname="helv")  # fmt: skip
    doc.set_toc([[1, "Chapter 1 Overview", 1], [1, "Chapter 2 Overview", 2]])
    path = tmp_path / "t.pdf"
    doc.save(path)
    doc.close()
    return path


def test_pymupdf_lines(pdf_path):
    lines, n_pages = extract_lines(pdf_path, engine="pymupdf")
    assert n_pages == 2
    assert [ln.page for ln in lines] == [1] * 5 + [2] * 5
    title, first_body = lines[0], lines[1]
    assert title.text == "Chapter 1 Overview"
    assert title.size == 20.0 and title.bold and "Bold" in title.font
    assert first_body.size == 11.0 and not first_body.bold
    assert title.page_width == 612 and title.page_height == 792
    assert title.gap_above == math.inf and lines[4].gap_below == math.inf
    assert lines[5].gap_above == math.inf  # page 2 restarts
    assert all(math.isfinite(ln.gap_above) for ln in lines[1:5])
    assert lines[4].gap_above > 10 > lines[2].gap_above  # the paragraph gap is visible
    assert lines[1].gap_above == pytest.approx(lines[0].gap_below)
    assert all(ln.gap_above >= 0 and ln.gap_below >= 0 for ln in lines)
    assert title.y1 < first_body.y0 + 5 and title.x0 == pytest.approx(72, abs=1)


def test_pdfplumber_engine_matches_pymupdf_texts(pdf_path):
    pytest.importorskip("pdfplumber")
    ref, _ = extract_lines(pdf_path, engine="pymupdf")
    lines, n_pages = extract_lines(pdf_path, engine="pdfplumber")
    assert n_pages == 2
    assert [ln.text for ln in lines] == [ln.text for ln in ref]
    assert [ln.page for ln in lines] == [ln.page for ln in ref]
    assert [(ln.size, ln.bold) for ln in lines] == [(ln.size, ln.bold) for ln in ref]
    assert lines[0].gap_above == math.inf and lines[0].gap_below > 0
    assert lines[4].gap_above > lines[2].gap_above  # block break before the isolated line
    assert lines[4].block > lines[3].block


def test_pdfplumber_splits_columns(tmp_path):
    pytest.importorskip("pdfplumber")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "left column text", fontsize=11)
    page.insert_text((350, 100), "right column text", fontsize=11)
    path = tmp_path / "cols.pdf"
    doc.save(path)
    lines, _ = extract_lines(path, engine="pdfplumber")
    assert [ln.text for ln in lines] == ["left column text", "right column text"]


def test_symbol_only_lines_are_dropped(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Real text", fontsize=11)
    page.insert_text((72, 130), "*** --- ***", fontsize=11)
    path = tmp_path / "s.pdf"
    doc.save(path)
    lines, _ = extract_lines(path, engine="pymupdf")
    assert [ln.text for ln in lines] == ["Real text"]


@pytest.mark.parametrize("engine", ["pymupdf", "pdfplumber"])
def test_non_pdf_raises_docrooter_error(tmp_path, engine):
    if engine == "pdfplumber":
        pytest.importorskip("pdfplumber")
    bad = tmp_path / "x.pdf"
    bad.write_text("this is not a pdf")
    with pytest.raises(DocrooterError, match="cannot open PDF"):
        extract_lines(bad, engine=engine)
    with pytest.raises(DocrooterError, match="cannot open PDF"):
        extract_lines(tmp_path / "missing.pdf", engine=engine)


def test_encrypted_pdf_raises(tmp_path):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 100), "secret", fontsize=11)
    path = tmp_path / "enc.pdf"
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
    with pytest.raises(DocrooterError, match="cannot open PDF"):
        extract_lines(path, engine="pymupdf")


def test_unknown_engine_raises(pdf_path):
    with pytest.raises(DocrooterError, match="engine"):
        extract_lines(pdf_path, engine="nope")  # type: ignore[arg-type]


def test_read_bookmarks(pdf_path, tmp_path):
    assert read_bookmarks(pdf_path) == [(1, "Chapter 1 Overview", 1), (1, "Chapter 2 Overview", 2)]
    assert read_bookmarks(tmp_path / "missing.pdf") == []


def test_read_bookmarks_drops_items_without_target_page(tmp_path):
    doc = pymupdf.open()
    doc.new_page()
    doc.new_page()
    doc.set_toc([[1, "Kept", 1], [1, "No target", -1], [1, "Also kept", 2]])
    path = tmp_path / "toc.pdf"
    doc.save(path)
    doc.close()
    assert read_bookmarks(path) == [(1, "Kept", 1), (1, "Also kept", 2)]


def test_text_outside_the_media_box_is_not_extracted(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "Inside text", fontsize=11)
    xref = page.get_contents()[0]
    outside = b"\nq BT /helv 11 Tf 1 0 0 1 700 500 Tm (Outside text) Tj ET Q\n"
    stream = doc.xref_stream(xref) + outside
    doc.update_stream(xref, stream)
    path = tmp_path / "outside.pdf"
    doc.save(path)
    doc.close()
    lines, _ = extract_lines(path, engine="pymupdf")
    assert [line.text for line in lines] == ["Inside text"]
