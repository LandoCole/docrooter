"""``chunk()``: section-aware chunks of an element list (spec §9). Documents are built by hand, so
no PDF, no backend and no model is involved."""

import time

import pytest

from docrooter import Document, RouterConfig, chunking
from docrooter.chunking import Chunk, chunk
from docrooter.elements.models import Element, TableData
from docrooter.errors import DocrooterError
from docrooter.rooter.analysis import Analysis
from docrooter.views import estimate_tokens

WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa"  # ten words


def words(n: int) -> str:
    return " ".join(["word"] * n)


def make_doc(items: list[tuple], sections=None) -> Document:
    """A document with sections A (s1), B inside A (s1.1) and C (s2), and ``items`` as its
    elements: ``(kind, text, section_id, **fields)`` in reading order, ids e0, e1, …"""
    doc = Document.from_sections(
        sections
        or [
            {"title": "A", "level": 1},
            {"title": "B", "level": 2},
            {"title": "C", "level": 1},
        ]
    )
    elements = []
    for index, (kind, text, section_id, *rest) in enumerate(items):
        fields = rest[0] if rest else {}
        if kind in ("heading", "title") and "level" not in fields:
            fields = {"level": 1, **fields}
        elements.append(
            Element(
                id=f"e{index}",
                kind=kind,
                text=text,
                boxes=[],
                order=index,
                section_id=section_id,
                **fields,
            )
        )
    doc.elements = elements
    for element in elements:
        if element.section_id in doc.sections:
            doc.sections[element.section_id].element_ids.append(element.id)
    return doc


def standard_doc() -> Document:
    revenue = TableData(
        rows=[["Region", "Revenue"], ["North", "10"], ["South", "12"]], header_rows=1
    )
    return make_doc(
        [
            ("heading", "A", "s1"),  # e0
            ("paragraph", "Intro of A.", "s1"),  # e1
            ("heading", "B", "s1.1", {"level": 2}),  # e2
            ("paragraph", "First of B.", "s1.1"),  # e3
            ("paragraph", "Second of B.", "s1.1"),  # e4
            ("table", "Region Revenue North 10 South 12", "s1.1", {"table": revenue}),  # e5
            ("caption", "Table 1. Revenue by region", "s1.1", {"caption_for": "e5"}),  # e6
            ("heading", "C", "s2"),  # e7
            ("paragraph", "Body of C.", "s2"),  # e8
            ("page_footer", "Page 3", "s2"),  # e9
        ]
    )


def test_chunks_never_cross_sections():
    doc = standard_doc()
    for max_tokens in (512, 8, 1):
        chunks = chunk(doc, max_tokens=max_tokens)
        by_id = {e.id: e for e in doc.elements}
        for c in chunks:
            assert {by_id[eid].section_id for eid in c.element_ids} == {c.section_id}
        assert {c.section_id for c in chunks} == {"s1", "s1.1", "s2"}
    # a roomy budget still gives one chunk per section here (the table is its own chunk)
    assert [c.section_id for c in chunk(doc)] == ["s1", "s1.1", "s1.1", "s1.1", "s2"]


def test_table_is_one_chunk_with_description():
    doc = standard_doc()
    chunks = chunk(doc)
    (table,) = [c for c in chunks if "e5" in c.element_ids]
    assert table.element_ids == ["e5"]
    markdown = doc.elements[5].table.to_markdown()
    assert table.text == f"A > B\n\nTable: Table 1. Revenue by region; 3×2\n{markdown}"
    # the neighbouring paragraphs are not merged into it, whatever the budget
    assert [c.element_ids for c in chunks if c.section_id == "s1.1"] == [
        ["e2", "e3", "e4"],
        ["e5"],
        ["e6"],
    ]
    # without a caption, the description names the first row
    doc.elements[5].caption_id = None
    doc.elements[6] = doc.elements[6].model_copy(update={"caption_for": None})
    (plain,) = [c for c in chunk(doc) if "e5" in c.element_ids]
    assert "Table: Region | Revenue; 3×2" in plain.text


def test_table_caption_is_found_through_the_table_too():
    doc = standard_doc()
    doc.elements[5].caption_id = "e6"
    doc.elements[6] = doc.elements[6].model_copy(update={"caption_for": None})
    (table,) = [c for c in chunk(doc) if "e5" in c.element_ids]
    assert "Table: Table 1. Revenue by region; 3×2" in table.text


def test_table_without_cells_uses_its_text():
    doc = make_doc([("table", "a b c", "s1")])
    (c,) = chunk(doc)
    assert c.text == "A\n\nTable: untitled; 0×0\na b c"
    incomplete = TableData(rows=[["x"]], header_rows=1)
    doc = make_doc(
        [
            (
                "table",
                "x and the lost words",
                "s1",
                {"table": incomplete, "flags": ["cell_text_incomplete"]},
            )
        ]
    )
    assert chunk(doc)[0].text.endswith("\nx and the lost words")


def test_oversize_element_flagged():
    doc = make_doc(
        [
            ("paragraph", words(5), "s1"),
            ("paragraph", words(100), "s1"),
            ("paragraph", words(5), "s1"),
        ]
    )
    chunks = chunk(doc, max_tokens=20)
    assert [c.element_ids for c in chunks] == [["e0"], ["e1"], ["e2"]]
    assert [c.flags for c in chunks] == [[], ["oversize"], []]
    assert chunks[1].tokens > 20 and all(c.tokens <= 20 for c in (chunks[0], chunks[2]))


def rich_doc() -> Document:
    """Headings with a body, leaf headings with none (C, D) and a table, in five sections:
    A (s1) > B (s1.1), C (s1.2); D (s2); E (s3)."""
    table = TableData(rows=[["x", "y"], ["1", "2"]], header_rows=1)
    return make_doc(
        [
            ("heading", "A", "s1"),  # e0
            ("paragraph", "Intro of A.", "s1"),  # e1
            ("heading", "B", "s1.1", {"level": 2}),  # e2
            ("paragraph", "First of B.", "s1.1"),  # e3
            ("table", "x y 1 2", "s1.1", {"table": table}),  # e4
            ("heading", "C", "s1.2", {"level": 2}),  # e5: no body
            ("heading", "D", "s2"),  # e6: no body
            ("heading", "E", "s3"),  # e7
            ("paragraph", "Body of E.", "s3"),  # e8
            ("page_footer", "Page 3", "s3"),  # e9
        ],
        sections=[
            {"title": "A", "level": 1},
            {"title": "B", "level": 2},
            {"title": "C", "level": 2},
            {"title": "D", "level": 1},
            {"title": "E", "level": 1},
        ],
    )


def test_every_content_element_in_exactly_one_chunk():
    for doc in (standard_doc(), rich_doc()):
        for max_tokens in (512, 16, 4, 1):
            ids = [eid for c in chunk(doc, max_tokens=max_tokens) for eid in c.element_ids]
            assert sorted(ids) == sorted(e.id for e in doc.elements if e.kind != "page_footer")
        with_furniture = [eid for c in chunk(doc, include_furniture=True) for eid in c.element_ids]
        assert sorted(with_furniture) == sorted(e.id for e in doc.elements)
    assert chunk(standard_doc(), include_furniture=True)[-1].text == "C\n\nBody of C.\n\nPage 3"


def test_furniture_and_contents_are_skipped_by_default():
    doc = make_doc(
        [
            ("page_header", "ACME 10-K", "s1"),
            ("contents", "Item 1 .... 3", "s1"),
            ("page_number", "3", "s1"),
            ("paragraph", "Real text.", "s1"),
        ]
    )
    (c,) = chunk(doc)
    assert c.element_ids == ["e3"] and c.text == "A\n\nReal text."
    (everything,) = chunk(doc, include_furniture=True)
    assert everything.element_ids == ["e0", "e1", "e2", "e3"]


def test_heading_path_prefix_format():
    doc = standard_doc()
    first, second = chunk(doc)[:2]
    assert first.heading_path == ["A"] and first.text == "A\n\nIntro of A."
    assert second.heading_path == ["A", "B"]
    assert second.text == "A > B\n\nFirst of B.\n\nSecond of B."
    assert second.tokens == estimate_tokens(second.text)


def test_the_opening_heading_is_in_element_ids_but_not_repeated_in_the_text():
    doc = standard_doc()
    first = chunk(doc)[0]
    assert first.element_ids == ["e0", "e1"] and first.text == "A\n\nIntro of A."


def test_a_section_with_no_body_gets_a_path_only_chunk():
    chunks = chunk(rich_doc())
    assert [(c.section_id, c.element_ids, c.flags) for c in chunks] == [
        ("s1", ["e0", "e1"], []),
        ("s1.1", ["e2", "e3"], []),
        ("s1.1", ["e4"], []),  # the table
        ("s1.2", ["e5"], ["heading_only"]),
        ("s2", ["e6"], ["heading_only"]),
        ("s3", ["e7", "e8"], []),
    ]
    c, d = chunks[3], chunks[4]
    assert (c.text, c.heading_path, c.tokens) == ("A > C", ["A", "C"], 4)
    assert (d.text, d.heading_path) == ("D", ["D"])
    # a heading that opens a section with a body stays in that section's first chunk
    assert chunks[1].text == "A > B\n\nFirst of B."


def test_heading_only_chunk_stands_for_a_heading_before_a_table():
    # the table chunk takes the pending heading, so nothing else is emitted for it
    doc = make_doc(
        [("heading", "A", "s1"), ("table", "a b", "s1")],
    )
    (c,) = chunk(doc)
    assert c.element_ids == ["e0", "e1"] and c.flags == []


def test_packing_is_linear(monkeypatch):
    doc = make_doc([("paragraph", words(20), "s1") for _ in range(4000)])
    calls = []
    real = chunking.estimate_tokens
    monkeypatch.setattr(chunking, "estimate_tokens", lambda text: calls.append(1) or real(text))
    started = time.perf_counter()
    chunks = chunk(doc, max_tokens=100_000, overlap_elements=2)
    elapsed = time.perf_counter() - started
    assert len(chunks) == 2 and chunks[0].element_ids[0] == "e0"
    assert len(calls) == len(chunks)  # one estimate per emitted chunk, none per append
    assert elapsed < 1.0
    assert all(c.tokens <= 100 for c in chunk(doc, max_tokens=100, overlap_elements=2))


def test_preamble_has_no_path():
    doc = make_doc(
        [("paragraph", "Before any heading.", "s0"), ("paragraph", "Under A.", "s1")],
        sections=[
            {"title": "Preamble", "level": 0, "text": "Before any heading."},
            {"title": "A", "level": 1},
        ],
    )
    pre, under = chunk(doc)
    assert pre.heading_path == [] and pre.text == "Before any heading."
    assert under.text == "A\n\nUnder A."


def test_list_markers_are_kept():
    doc = make_doc(
        [
            ("list_item", "first point", "s1", {"marker": "1."}),
            ("list_item", "second point", "s1", {"marker": "•"}),
            ("footnote", "see page 4", "s1", {"marker": "*"}),
        ]
    )
    assert chunk(doc)[0].text == "A\n\n1. first point\n\n• second point\n\n* see page 4"


def test_a_title_line_that_is_not_a_heading_stays_text():
    doc = make_doc(
        [
            ("title", "Annual Report", "s1", {"level": None}),
            ("paragraph", "Text.", "s1"),
        ]
    )
    assert chunk(doc)[0].text == "A\n\nAnnual Report\n\nText."


def test_packing_fills_chunks_up_to_the_budget():
    doc = make_doc([("paragraph", words(10), "s1") for _ in range(6)])
    chunks = chunk(doc, max_tokens=40)
    assert all(c.tokens <= 40 for c in chunks)
    assert [len(c.element_ids) for c in chunks] == [2, 2, 2]  # 1 + 10 + 10 words = 27 tokens
    assert [eid for c in chunks for eid in c.element_ids] == [f"e{i}" for i in range(6)]
    assert [c.id for c in chunks] == ["c0", "c1", "c2"]


def test_overlap_elements():
    doc = make_doc(
        [("paragraph", words(10), "s1") for _ in range(5)]
        + [("paragraph", words(10), "s2"), ("paragraph", words(10), "s2")]
    )
    plain = chunk(doc, max_tokens=40)
    assert [c.element_ids for c in plain] == [["e0", "e1"], ["e2", "e3"], ["e4"], ["e5", "e6"]]
    overlapped = chunk(doc, max_tokens=40, overlap_elements=1)
    assert [c.element_ids for c in overlapped] == [
        ["e0", "e1"],
        ["e1", "e2"],
        ["e2", "e3"],
        ["e3", "e4"],
        ["e5", "e6"],  # a new section starts clean
    ]
    assert all(c.tokens <= 40 for c in overlapped)
    # the overlap is the same element text, in the same form
    assert overlapped[1].text.split("\n\n")[1] == overlapped[0].text.split("\n\n")[-1]


def test_overlap_is_dropped_when_it_would_not_fit():
    doc = make_doc([("paragraph", words(10), "s1"), ("paragraph", words(25), "s1")])
    chunks = chunk(doc, max_tokens=40, overlap_elements=3)
    assert [c.element_ids for c in chunks] == [["e0"], ["e1"]]


def test_overlap_does_not_repeat_a_table_or_enter_one():
    doc = standard_doc()
    chunks = chunk(doc, max_tokens=512, overlap_elements=2)
    (table,) = [c for c in chunks if "e5" in c.element_ids]
    assert table.element_ids == ["e5"]
    assert not any("e5" in c.element_ids for c in chunks if c is not table)
    # the caption after the table does not take the paragraphs before it as overlap
    assert [c.element_ids for c in chunks if "e6" in c.element_ids] == [["e6"]]


def test_empty_and_textless_inputs():
    assert chunk(make_doc([])) == []
    assert chunk(make_doc([("figure", "", "s1"), ("paragraph", "  ", "s1")])) == []
    assert chunk(Document.from_sections([{"title": "A", "level": 1}])) == []


def test_a_document_without_elements_is_an_error():
    doc = Document.from_sections([{"title": "A", "level": 1, "text": "Some body text."}])
    with pytest.raises(DocrooterError, match="elements=True"):
        chunk(doc)


def test_an_analysis_chunks_like_its_document():
    doc = standard_doc()
    analysis = Analysis(document=doc, config=RouterConfig(profile="fast"))
    assert chunk(analysis) == chunk(doc)
    assert all(isinstance(c, Chunk) for c in chunk(analysis))


def test_argument_validation():
    doc = standard_doc()
    with pytest.raises(ValueError, match="max_tokens"):
        chunk(doc, max_tokens=0)
    with pytest.raises(ValueError, match="overlap_elements"):
        chunk(doc, overlap_elements=-1)
