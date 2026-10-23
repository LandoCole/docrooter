"""PDF loader: bookmark matching (pure) and a generated PDF through PyMuPDF4LLM."""

import time

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


# --- D9: fuzzy bookmarks, headings from bookmarks, no level jumps ----------------------------

ITEM_10 = "Item 10. Directors, Executive Officers and Corporate Governance"
COMPARISONS = "Comparisons of the Tax Liabilities of FCCs and USCCs Depend on the Measure Used"


def body(title, text, *, level=1, page=1, line=1):
    return RawHeading(title=title, level=level, text=text, line=line, page=page)


def pages_of(text, *page_runs, first_line=2):
    """``(line, page)`` for each line of ``text``: ``page_runs`` are ``(page, n_lines)`` runs."""
    pages = [page for page, count in page_runs for _ in range(count)]
    assert len(pages) == len(text.splitlines())
    return [(first_line + k, page) for k, page in enumerate(pages)]


@pytest.mark.parametrize(
    ("title", "bookmark"),
    [
        (ITEM_10, "Item 10. Directors, Executive Officers and Corporate Goverance"),
        (
            "Report of Independent Registered Public Accounting Firm",
            "Reports of Independent Registered Public Accounting Firm",
        ),
        (
            "ITEM 12. SECURITY OWNERSHIP OF CERTAIN BENEFICIAL OWNERS AND MANAGEMENT AND "
            "RELATED STOCKHOLDER MATTERS",
            "ITEM 12. SECURITY OWNERSHIP OF CERTAIN BENEFICIAL OWNERS AND MANAGEMENT AND "
            "RELATED STOCKHOLDERMATTERS",
        ),
    ],
)
def test_bookmark_typo_still_matches(title, bookmark):
    out = apply_bookmarks(
        [heading("PART III", 1, page=55), heading(title, 4, page=56)],
        [[1, "PART III", 55], [2, bookmark, 56]],
    )
    assert [(h.title, h.level) for h in out] == [("PART III", 1), (title, 2)]
    assert out[1].flags == ["level_from_bookmark"]


def test_fuzzy_match_needs_nearby_page():
    toc = [[2, "Item 10. Directors, Executive Officers and Corporate Goverance", 70]]
    out = apply_bookmarks([heading(ITEM_10, 4, page=56)], toc)
    assert out[0].level == 4 and out[0].flags == []
    # one page away still matches
    toc = [[2, "Item 10. Directors, Executive Officers and Corporate Goverance", 57]]
    assert apply_bookmarks([heading(ITEM_10, 4, page=56)], toc)[0].level == 2


def test_exact_matches_win_before_fuzzy():
    # The near-duplicate comes first in reading order, but the bookmark "Basis for Opinions"
    # belongs to the exact heading later on; the first heading must not take it fuzzily.
    headings = [heading("Basis for Opinion", 4, page=52), heading("Basis for Opinions", 4, page=52)]
    out = apply_bookmarks(headings, [[3, "Basis for Opinions", 52]])
    assert [h.level for h in out] == [4, 3]
    assert out[0].flags == [] and out[1].flags == ["level_from_bookmark"]


def test_missing_bookmark_heading_is_inserted():
    text = (
        "Scope text on page nine.\n"
        "\n"
        "Comparisons of the Tax Liabilities of FCCs and USCCs\n"
        "Depend on the Measure Used\n"
        "\n"
        "Comparisons of the tax liabilities vary."
    )
    scope = body("Scope and Methodology", text, level=2, page=9, line=40)
    toc = [[2, "Scope and Methodology", 9], [2, COMPARISONS, 10]]
    out = apply_bookmarks([scope], toc, line_pages=[pages_of(text, (9, 2), (10, 4), first_line=42)])
    assert [(h.title, h.level, h.page) for h in out] == [
        ("Scope and Methodology", 2, 9),
        (COMPARISONS, 2, 10),
    ]
    assert out[0].text == "Scope text on page nine."
    assert out[1].text == "Comparisons of the tax liabilities vary."
    assert out[1].flags == ["heading_from_bookmark"] and out[1].line == 44
    assert scope.text == text  # input untouched


def test_inserted_heading_allows_emphasis_and_punctuation():
    text = "Intro.\n**Item 10. Directors, Executive Officers and Corporate Governance**\nBody."
    out = apply_bookmarks([body("PART III", text, page=56)], [[2, ITEM_10, 56]])
    assert [(h.title, h.level) for h in out] == [("PART III", 1), (ITEM_10, 2)]
    assert out[1].text == "Body." and out[0].text == "Intro."


def test_insertion_skips_contents_and_mid_sentence():
    # A "Contents" page (p3) lists the title at a line start; the bookmark points at p10.
    contents = f"{COMPARISONS}\nBackground"
    text = (
        "Earlier text.\n"
        f"As shown in {COMPARISONS}, the measures differ.\n"
        f"{COMPARISONS} differently, the note says.\n"
        "More text."
    )
    headings = [body("Contents", contents, page=3), body("Scope", text, level=2, page=9)]
    line_pages = [pages_of(contents, (3, 2)), pages_of(text, (9, 1), (10, 3))]
    toc = [[2, COMPARISONS, 10]]
    assert apply_bookmarks(headings, toc, line_pages=line_pages) == headings
    # the same body with the title on a line of its own on p10 is split
    text = f"{contents}\nOn page ten.\n{COMPARISONS}\nFindings."
    single = [body("Scope", text, level=2, page=3)]
    out = apply_bookmarks(single, toc, line_pages=[pages_of(text, (3, 2), (10, 3))])
    assert [h.title for h in out] == ["Scope", COMPARISONS]
    assert out[0].text == f"{contents}\nOn page ten." and out[1].text == "Findings."
    # without line pages a body counts as on its heading's page only
    assert apply_bookmarks(single, toc) == single


def make_blank_pdf(pymupdf, path, n_pages, toc):
    doc = pymupdf.open()
    for _ in range(n_pages):
        doc.new_page()
    if toc:
        doc.set_toc(toc)
    doc.save(str(path))
    doc.close()
    return path


def fake_markdown(monkeypatch, chunks):
    """Make PyMuPDF4LLM return ``chunks`` (one Markdown string per page)."""
    import pymupdf4llm

    def to_markdown(path, **options):
        return [{"text": text, "metadata": {"page_number": i + 1}} for i, text in enumerate(chunks)]

    monkeypatch.setattr(pymupdf4llm, "to_markdown", to_markdown)


def test_no_level_jump_after_bookmarks(pymupdf, tmp_path, monkeypatch):
    from docrooter.document.pdf import load_pdf_markdown

    # font levels 1-4 then 4 and 5: the bookmark lifts PART III to level 1, which leaves
    # "Executive Compensation" three levels below it unless the cap runs again
    chunks = [
        "# Annual Report\n\n## Part II\n\n### Item 9\n\nx\n\n#### PART III\n\nintro\n\n"
        "#### Executive Compensation\n\nbody\n\n##### Detail\n\nx\n"
    ]
    fake_markdown(monkeypatch, chunks)
    path = make_blank_pdf(pymupdf, tmp_path / "jump.pdf", 1, [[1, "PART III", 1]])
    doc = load_pdf_markdown(path)
    assert [(s.title, s.level) for s in doc.walk()][3:] == [
        ("PART III", 1),
        ("Executive Compensation", 2),
        ("Detail", 3),
    ]
    assert "level_repaired" in doc.section("s2.1").flags
    assert doc.section("s2.1").parent_id == "s2"
    # repair=False leaves the levels alone
    raw = load_pdf_markdown(path, repair=False)
    assert [s.level for s in raw.walk()] == [1, 2, 3, 1, 4, 5]


def test_heading_from_bookmark_through_the_reader(pymupdf, tmp_path, monkeypatch):
    from docrooter.document.pdf import load_pdf_markdown

    chunks = [
        "# Contents\n\nScope and Methodology\n\nComparisons of the Tax Liabilities of FCCs and "
        "USCCs Depend on the Measure Used\n",
        "## Scope and Methodology\n\nWe used SOI data.\n\n## ***\n",  # emptied: text joins
        "concluded that the data were reliable.\n\nComparisons of the Tax Liabilities of FCCs and "
        "USCCs\nDepend on the Measure Used \n\nComparisons vary.\n\n### Sub finding\n\nmore\n",
    ]
    fake_markdown(monkeypatch, chunks)
    toc = [[1, "Contents", 1], [2, "Scope and Methodology", 2], [2, COMPARISONS, 3]]
    doc = load_pdf_markdown(make_blank_pdf(pymupdf, tmp_path / "gao.pdf", 3, toc))
    assert [(s.title, s.level, s.page_start) for s in doc.walk()] == [
        ("Contents", 1, 1),
        ("Scope and Methodology", 2, 2),
        (COMPARISONS, 2, 3),
        ("Sub finding", 3, 3),
    ]
    scope = doc.section("s1.1")
    assert scope.text == "We used SOI data.\nconcluded that the data were reliable."
    finding = doc.section("s1.2")
    assert finding.text == "Comparisons vary." and "heading_from_bookmark" in finding.flags
    assert doc.section("s1").text.endswith("Depend on the Measure Used")  # contents untouched


def test_no_bookmarks_only_repairs_levels(pymupdf, tmp_path, monkeypatch):
    from docrooter.document.build import build_document
    from docrooter.document.headings import repair_levels
    from docrooter.document.markdown import extract_headings
    from docrooter.document.pdf import load_pdf_markdown

    chunks = [
        "lead\n\n# Report\n\nintro\n\n### Deep\n\nComparisons of the Tax Liabilities\n",
        "## Next\n\nbody\n\n#### Deeper\n\ntext\n",
    ]
    fake_markdown(monkeypatch, chunks)
    path = make_blank_pdf(pymupdf, tmp_path / "plain.pdf", 2, [])
    doc = load_pdf_markdown(path)
    markdown, starts, pages = _join_chunks(
        [{"text": t, "metadata": {"page_number": i + 1}} for i, t in enumerate(chunks)]
    )
    raw = [
        h.model_copy(update={"page": pages[0] if h.line < starts[1] else pages[1]})
        for h in extract_headings(markdown)
    ]
    expected = build_document(repair_levels(raw), source=str(path), title="Report")
    assert doc.sections == expected.sections and doc.root_ids == expected.root_ids
    assert doc.meta["bookmarks"] == 0


def test_line_pages_needs_one_entry_per_heading():
    with pytest.raises(ValueError, match="one entry per heading"):
        apply_bookmarks([heading("A")], [[1, "B", 1]], line_pages=[])


@pytest.mark.parametrize("separator", [" ", "\x0c", "\r", "\x85"])
def test_insertion_counts_lines_as_splitlines_does(separator):
    # str.splitlines (the reader's line scheme) also breaks on these; "\n" counting does not
    text = f"alpha{separator}beta\nTitle Here\nafter text"
    lines = pages_of(text, (1, 2), (2, 2))
    out = apply_bookmarks([body("Start", text, page=1)], [[2, "Title Here", 2]], line_pages=[lines])
    assert [(h.title, h.page, h.line) for h in out] == [("Start", 1, 1), ("Title Here", 2, 4)]
    assert out[0].text == f"alpha{separator}beta" and out[1].text == "after text"
    # on the page before, the title is not looked for
    out = apply_bookmarks([body("Start", text, page=1)], [[2, "Title Here", 1]], line_pages=[lines])
    assert [h.title for h in out] == ["Start"]
    # all on one page: the title is not repeated in the new section's text
    lines = pages_of(text, (1, 4))
    out = apply_bookmarks([body("Start", text, page=1)], [[2, "Title Here", 1]], line_pages=[lines])
    assert [h.text for h in out] == [f"alpha{separator}beta", "after text"]


@pytest.mark.parametrize(
    ("title", "bookmark"),
    [
        ("Part I Financial Information", "Part II Financial Information"),
        ("Item 7 Management Discussion", "Item 7A Management Discussion"),
        ("Note 1 Summary of Significant Policies", "Note 2 Summary of Significant Policies"),
        ("Exhibit 3.1 Certificate of Incorporation", "Exhibit 3.2 Certificate of Incorporation"),
    ],
)
def test_fuzzy_match_keeps_numbering_apart(title, bookmark):
    out = apply_bookmarks([heading(title, 4, page=30)], [[2, bookmark, 31]])
    assert out[0].level == 4 and out[0].flags == []


def test_same_level_headings_stay_siblings_under_a_bookmark(pymupdf, tmp_path, monkeypatch):
    from docrooter.document.pdf import load_pdf_markdown

    statements = ["Balance Sheets", "Cash Flows", "Equity", "Notes"]
    chunks = [
        "# Annual Report\n\n## Part II\n\n### Item 7\n\nx\n\n#### Item 8\n\ny\n\n"
        + "".join(f"###### {title}\n\nbody\n\n" for title in statements)
    ]
    fake_markdown(monkeypatch, chunks)
    toc = [[1, "Annual Report", 1], [2, "Part II", 1], [3, "Item 7", 1], [3, "Item 8", 1]]
    doc = load_pdf_markdown(make_blank_pdf(pymupdf, tmp_path / "att.pdf", 1, toc))
    item_8 = next(s for s in doc.walk() if s.title == "Item 8")
    assert item_8.level == 3
    assert [doc.section(c).title for c in item_8.children] == statements
    assert {doc.section(c).level for c in item_8.children} == {4}


def test_many_leftover_bookmarks_in_a_large_body_stay_fast():
    # 3 MB body over 100 pages and 500 bookmarks that are nowhere in it: each bookmark must
    # only search the lines of its own page, not the whole body.
    line = "x" * 74
    text = "\n".join([line] * 40_000)
    lines = [(2 + k, 1 + k // 400) for k in range(40_000)]
    toc = [[2, f"Missing Heading Number {k}", 1 + k % 100] for k in range(500)]
    started = time.perf_counter()
    out = apply_bookmarks([body("Start", text, page=1)], toc, line_pages=[lines])
    assert time.perf_counter() - started < 3
    assert len(out) == 1 and out[0].text == text


def test_insertion_finds_a_title_wrapped_over_several_lines():
    text = "one\ntwo\nA Long Title That\nWraps Over Lines\nThree\nafter"
    lines = pages_of(text, (1, 2), (2, 4))
    toc = [[2, "A Long Title That Wraps Over Lines", 2]]
    out = apply_bookmarks([body("Start", text, page=1)], toc, line_pages=[lines])
    assert [(h.title, h.text) for h in out] == [
        ("Start", "one\ntwo"),
        ("A Long Title That Wraps Over Lines", "Three\nafter"),
    ]


@pytest.mark.parametrize(
    ("title", "bookmark"),
    [("Revenue", "Revenues"), ("Income Taxes", "Income Tax"), ("Risk Factor", "Risk Factors")],
)
def test_short_titles_are_not_fuzzy_matched(title, bookmark):
    # a short title one letter off is more likely a different heading than a typo
    out = apply_bookmarks([heading(title, 4, page=30)], [[2, bookmark, 30]])
    assert out[0].level == 4 and out[0].flags == []
    # equal after normalisation still matches
    out = apply_bookmarks([heading(title, 4, page=30)], [[2, title.upper(), 30]])
    assert out[0].level == 2


def test_a_short_heading_leaves_its_bookmark_for_the_missed_heading():
    text = "Intro.\nRevenues\nSales grew."
    out = apply_bookmarks(
        [body("Revenue", text, level=3, page=30)], [[2, "Revenues", 30]], line_pages=None
    )
    assert [(h.title, h.level) for h in out] == [("Revenue", 3), ("Revenues", 2)]
    assert out[0].text == "Intro." and out[1].text == "Sales grew."
