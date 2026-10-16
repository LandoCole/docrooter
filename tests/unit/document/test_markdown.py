from pathlib import Path

from docrooter.document import Document, build_document, extract_headings

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "docs"


def read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_atx_headings_build_tree():
    doc = build_document(extract_headings(read("simple.md")), source="simple.md")
    assert [e.title for e in doc.toc()] == ["Intro", "Background", "Scope", "Results"]
    assert doc.section("s1.2").title == "Scope" and doc.section("s1.2").parent_id == "s1"
    assert doc.section("Intro").text.startswith("Intro body")  # body excludes child sections
    assert "Background body" not in doc.section("Intro").text


def test_no_headings_single_section():
    doc = build_document(extract_headings("just text\nmore"), source="notes.md")
    assert list(doc.sections) == ["s0"] and doc.sections["s0"].title == "notes"
    doc = build_document(extract_headings(read("no_headings.md")), source="notes.md")
    assert list(doc.sections) == ["s0"]


def test_preamble_before_first_heading():
    doc = build_document(extract_headings("lead text\n# A\nbody"), source="x.md")
    assert doc.sections["s0"].title == "Preamble" and doc.root_ids == ["s0", "s1"]
    assert doc.sections["s1"].parent_id is None
    assert [e.title for e in doc.toc()] == ["A"]


def test_setext_and_fenced_code_ignored():
    titles = [h.title for h in extract_headings(read("setext_and_code.md"))]
    assert titles == ["Setext Title", "Sub"]  # '# comment' inside ``` is not a heading


def test_setext_levels_and_atx_closing_hashes():
    headings = extract_headings("A\n===\nB\n---\n## C ##\n#hashtag\n# C#\n")
    assert [(h.title, h.level) for h in headings] == [("A", 1), ("B", 2), ("C", 2), ("C#", 1)]


def test_front_matter_skipped():
    headings = extract_headings("---\ntitle: x\n---\n# A\nbody")
    assert [h.title for h in headings] == ["A"]


def test_duplicate_titles_unique_ids():
    doc = build_document(extract_headings(read("duplicate_titles.md")), source="d.md")
    ids = [s.id for s in doc.walk() if s.title == "Overview"]
    assert len(ids) == 2 and len(set(ids)) == 2
    assert doc.section("Overview").id == ids[0]


def test_deep_doc_no_recursion():
    md = "\n".join(f"{'#' * min(d, 6)} H{d}\nx" for d in range(1, 31))  # repeated H6s stay siblings
    doc = build_document(extract_headings(md), source="deep.md")
    assert len(list(doc.walk())) == 30
    assert len(doc.toc()) == 30


def test_deeply_nested_tree_no_recursion():
    entries = [{"title": f"T{d}", "level": d} for d in range(1, 31)]
    doc = Document.from_sections(entries)
    assert len(list(doc.walk())) == 30
    assert doc.toc()[-1].depth == 29
    assert [s.title for s in doc.leaves()] == ["T30"]


def test_section_lookup_and_leaves():
    doc = build_document(extract_headings(read("simple.md")), source="simple.md")
    assert [s.id for s in doc.leaves()] == ["s1.1", "s1.2", "s2"]
    try:
        doc.section("Nope")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")


def test_from_sections():
    doc = Document.from_sections(
        [{"title": "A", "level": 1, "text": "x"}, {"title": "B", "level": 2}]
    )
    assert doc.source == "<sections>"
    assert doc.section("s1.1").text == "" and doc.section("s1.1").parent_id == "s1"
