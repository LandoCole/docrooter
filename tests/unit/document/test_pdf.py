"""PDF loader: bookmark matching (pure) and a generated PDF through PyMuPDF4LLM."""

import pytest

from docrooter import load
from docrooter.document.models import RawHeading
from docrooter.document.pdf import _clean_title, _join_chunks, _normalise, apply_bookmarks

BODY = (
    "This paragraph is ordinary body text that runs long enough to dominate the page and set "
    "the body font size for the whole document."
)


def heading(title, level=1, page=1):
    return RawHeading(title=title, level=level, text="", line=1, page=page)


def test_normalise_ignores_case_punctuation_and_spacing():
    assert _normalise("  1.2.  Risk   FACTORS: ") == _normalise("1.2 risk factors")
    assert _normalise("Item 1A. Risk Factors") == _normalise("ITEM 1A - Risk Factors")
    assert _normalise("Methods") != _normalise("Methods and Data")


def test_clean_title_strips_emphasis_markers():
    assert _clean_title("**Introduction**") == "Introduction"
    assert _clean_title("**1.** **Overview**") == "1. Overview"
    assert _clean_title("_Notes_ on snake_case") == "Notes on snake_case"
    assert _clean_title("***") == "***"


def test_apply_bookmarks_overrides_matching_levels_only():
    headings = [heading("1. Intro", 1), heading("Scope", 1), heading("Unlisted", 2)]
    toc = [[1, "1 Intro", 1], [2, "scope", 1]]
    out = apply_bookmarks(headings, toc)
    assert [(h.title, h.level) for h in out] == [("1. Intro", 1), ("Scope", 2), ("Unlisted", 2)]
    assert out[0].flags == ["level_from_bookmark"] and out[1].flags == ["level_from_bookmark"]
    assert out[2].flags == []
    assert headings[1].level == 1 and headings[1].flags == []  # inputs untouched


def test_apply_bookmarks_duplicate_titles_prefer_matching_page():
    toc = [[1, "Summary", 1], [2, "Summary", 3]]
    out = apply_bookmarks([heading("Summary", 1, page=3), heading("Summary", 1, page=1)], toc)
    assert [h.level for h in out] == [2, 1]
    # more headings than bookmarks: the extra one keeps its level
    out = apply_bookmarks([heading("Summary", 3, page=9)] * 3, toc)
    assert [h.level for h in out] == [1, 2, 3]


def test_apply_bookmarks_skips_preamble_and_empty_toc():
    pre = RawHeading(title="Preamble", level=0, text="x", line=1)
    assert apply_bookmarks([pre], [[1, "Preamble", 1]]) == [pre]
    assert apply_bookmarks([heading("A")], []) == [heading("A")]


def test_join_chunks_tracks_start_lines_and_pages():
    chunks = [
        {"text": "# A\n\nbody\n\n", "metadata": {"page_number": 1}},
        {"text": "more", "metadata": {"page_number": 2}},
        {"text": "# B", "metadata": {}},
    ]
    text, starts, pages = _join_chunks(chunks)
    lines = text.splitlines()
    assert [lines[s - 1] for s in starts] == ["# A", "more", "# B"]
    assert pages == [1, 2, 3]


@pytest.fixture(scope="module")
def pymupdf():
    pytest.importorskip("pymupdf4llm")
    return pytest.importorskip("pymupdf")


def make_pdf(pymupdf, path, *, bookmarks):
    """Three pages: every heading is drawn the same size, so bookmarks decide the levels."""
    doc = pymupdf.open()

    def page(items):
        p = doc.new_page()
        y = 72
        for kind, text in items:
            if kind == "h":
                p.insert_text((72, y), text, fontsize=24, fontname="hebo")
                y += 44
            else:
                for _ in range(4):
                    p.insert_text((72, y), text, fontsize=10, fontname="helv")
                    y += 16
                y += 14

    page([("h", "Introduction"), ("b", BODY), ("h", "Background"), ("b", BODY)])
    page([("h", "Methods"), ("b", BODY)])
    page([("h", "Appendix Notes"), ("b", BODY)])
    if bookmarks:
        doc.set_toc([[1, "Introduction", 1], [2, "Background", 1], [1, "Methods", 2]])
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def bookmarked_pdf(pymupdf, tmp_path):
    return make_pdf(pymupdf, tmp_path / "bookmarked.pdf", bookmarks=True)


def test_pdf_bookmark_levels(bookmarked_pdf):
    doc = load(bookmarked_pdf, engine="pymupdf4llm")
    toc = {e.title: e for e in doc.toc()}
    assert [(e.title, e.level) for e in doc.toc()][:3] == [
        ("Introduction", 1),
        ("Background", 2),
        ("Methods", 1),
    ]
    for title in ("Introduction", "Background", "Methods"):
        assert "level_from_bookmark" in doc.section(toc[title].id).flags
    assert doc.section(toc["Background"].id).parent_id == toc["Introduction"].id
    # not bookmarked: keeps its detected, repaired level and no bookmark flag
    appendix = doc.section(toc["Appendix Notes"].id)
    assert "level_from_bookmark" not in appendix.flags
    assert appendix.level == 1


def test_pdf_pages_and_text(bookmarked_pdf):
    doc = load(bookmarked_pdf, engine="pymupdf4llm")
    pages = {s.title: s.page_start for s in doc.walk()}
    assert pages["Introduction"] == 1 and pages["Background"] == 1
    assert pages["Methods"] == 2 and pages["Appendix Notes"] == 3
    methods = next(e.id for e in doc.toc() if e.title == "Methods")
    assert "ordinary body text" in doc.section(methods).text
    assert doc.source == str(bookmarked_pdf) and doc.title == "Introduction"


def test_pdf_without_bookmarks_uses_font_levels(pymupdf, tmp_path):
    path = make_pdf(pymupdf, tmp_path / "plain.pdf", bookmarks=False)
    doc = load(path, engine="pymupdf4llm")
    titles = ["Introduction", "Background", "Methods", "Appendix Notes"]
    assert [e.title for e in doc.toc()] == titles
    assert {e.level for e in doc.toc()} == {1}  # the bookmark test's level 2 came from bookmarks
    assert all("level_from_bookmark" not in s.flags for s in doc.walk())
    assert doc.section(doc.toc()[0].id).page_start == 1


def test_pymupdf4llm_max_pages(bookmarked_pdf):
    doc = load(bookmarked_pdf, engine="pymupdf4llm", max_pages=2)
    titles = [e.title for e in doc.toc()]
    assert "Methods" in titles and "Appendix Notes" not in titles
    assert doc.meta["n_pages"] == 3 and doc.meta["warnings"] == ["truncated_pages"]
    assert load(bookmarked_pdf, engine="pymupdf4llm").meta["warnings"] == []
