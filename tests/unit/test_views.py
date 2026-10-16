from pathlib import Path

import pytest

from docrooter import Document, SectionView, load
from docrooter.views import estimate_tokens

SIMPLE = Path(__file__).parent.parent / "fixtures" / "docs" / "simple.md"


@pytest.fixture
def doc():
    return load(SIMPLE)


@pytest.fixture
def long_doc():
    body = " ".join(f"word{i}" for i in range(200)) + "."
    return Document.from_sections([{"title": "Long", "level": 1, "text": body}])


def test_render_field_order_and_labels(doc):
    v = SectionView(fields=("path", "title", "text")).render_all(doc, ["s1.2"])["s1.2"]
    assert (
        v.text.splitlines()[0] == "Path: Intro > Scope" and v.text.splitlines()[1] == "Title: Scope"
    )
    assert v.text.splitlines()[2] == "Text: Scope body."
    assert not v.truncated


def test_budget_truncates_last_fitting_field(long_doc):
    v = SectionView(fields=("title", "text"), max_tokens=20).render_all(long_doc, ["s1"])["s1"]
    assert v.truncated and v.n_tokens <= 20
    assert v.text.splitlines()[0] == "Title: Long" and v.text.splitlines()[1].startswith(
        "Text: word0"
    )


def test_n_tokens_uses_custom_counter(long_doc):
    v = SectionView(fields=("title", "text"), max_tokens=10).render_all(
        long_doc, ["s1"], count_tokens=lambda t: len(t.split())
    )["s1"]
    assert v.truncated and v.n_tokens == len(v.text.split()) <= 10


def test_budget_too_small_for_next_field_stops(long_doc):
    v = SectionView(fields=("title", "text"), max_tokens=3).render_all(long_doc, ["s1"])["s1"]
    assert v.truncated and v.text == "Title: Long" and v.n_tokens == estimate_tokens(v.text)


def test_heading_only_leaf_renders_title():
    d = Document.from_sections([{"title": "Signatures", "level": 1}])
    v = SectionView(fields=("text",)).render_all(d, ["s1"])["s1"]
    assert v.text.strip() != "" and "Signatures" in v.text


def test_empty_parent_uses_children_titles():
    d = Document.from_sections(
        [{"title": "Part II", "level": 1}, {"title": "Item 7", "level": 2, "text": "x"}]
    )
    assert "Subsections: Item 7" in SectionView(fields=("text",)).render_all(d, ["s1"])["s1"].text


def test_empty_parent_title_mode():
    d = Document.from_sections(
        [{"title": "Part II", "level": 1}, {"title": "Item 7", "level": 2, "text": "x"}]
    )
    v = SectionView(fields=("text",), empty_section="title").render_all(d, ["s1"])["s1"]
    assert v.text == "Text: Part II"


def test_lead_and_keyphrases():
    d = Document.from_sections(
        [
            {"title": "A", "level": 1, "text": "First one. Second one! Third one? Fourth one."},
            {"title": "B", "level": 1, "text": "Unrelated filler content."},
        ]
    )
    v = SectionView(fields=("keyphrases", "lead"), lead_sentences=2).render_all(d, ["s1"])["s1"]
    lines = v.text.splitlines()
    assert lines[0].startswith("Key phrases: ") and lines[1] == "Lead: First one. Second one!"


def test_keyphrases_line_omitted_when_none():
    d = Document.from_sections([{"title": "A", "level": 1}])
    v = SectionView(fields=("title", "keyphrases")).render_all(d, ["s1"])["s1"]
    assert v.text == "Title: A"


def test_path_skips_preamble_and_unknown_id_raises():
    d = Document.from_sections(
        [{"title": "Pre", "level": 0, "text": "p"}, {"title": "H", "level": 1}]
    )
    pre = d.root_ids[0]
    view = SectionView(fields=("path",))
    assert view.render_all(d, [pre])[pre].text == "Path: Preamble"
    with pytest.raises(KeyError):
        view.render_all(d, ["nope"])


def test_unknown_field_rejected():
    with pytest.raises(ValueError):
        SectionView(fields=("title", "bogus"))


def test_deep_document_path():
    entries = [{"title": f"T{i}", "level": min(i + 1, 6), "text": "x"} for i in range(30)]
    d = Document.from_sections(entries)
    last = list(d.sections)[-1]
    out = SectionView(fields=("path",)).render_all(d, [last])[last].text
    assert out.startswith("Path: T0 > ") and out.endswith("T29")


@pytest.fixture
def signatures():
    return Document.from_sections([{"title": "Signatures", "level": 1}])


@pytest.mark.parametrize("fields", [("keyphrases",), ("lead",), ("keyphrases", "lead")])
def test_view_never_empty_when_fields_render_nothing(signatures, fields):
    v = SectionView(fields=fields).render_all(signatures, ["s1"])["s1"]
    assert v.text == "Title: Signatures" and not v.truncated


def test_view_never_empty_under_tiny_budget(signatures):
    v = SectionView(fields=("text",), max_tokens=1).render_all(signatures, ["s1"])["s1"]
    assert v.text == "Signatures" and v.truncated


def test_title_fallback_cut_to_budget_keeps_first_word():
    d = Document.from_sections([{"title": "Alpha Beta Gamma Delta Epsilon", "level": 1}])
    view = SectionView(fields=("lead",), max_tokens=4)
    v = view.render_all(d, ["s1"])["s1"]
    assert v.truncated and v.text == "Title: Alpha Beta" and v.n_tokens <= 4
    assert SectionView(fields=("lead",), max_tokens=1).render_all(d, ["s1"])["s1"].text == "Alpha"


def test_multiline_earlier_field_does_not_confuse_truncation():
    d = Document.from_sections(
        [{"title": "one two three four five six seven eight", "level": 1, "text": "l1\nl2\nl3"}]
    )
    v = SectionView(fields=("text", "title"), max_tokens=14).render_all(d, ["s1"])["s1"]
    assert v.truncated and v.n_tokens <= 14
    assert v.text.splitlines()[:3] == ["Text: l1", "l2", "l3"]
    assert v.text.splitlines()[3].startswith("Title: one two")
    assert v.text.count("l2") == 1
