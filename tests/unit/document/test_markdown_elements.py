from pathlib import Path

from docrooter import load
from docrooter.document.markdown import parse_markdown

MIXED = """\
# Title

Lead paragraph spans
two lines.

- item one
  - nested item

| h1 | h2 |
| --- | --- |
| a | b |

```python
# not a heading
x = 1
```

$$
E = mc^2
$$

![A diagram](img/d.png)
"""


def test_markdown_elements_kinds():
    doc = load(MIXED, elements=True)
    kinds = [e.kind for e in doc.elements]
    assert kinds == [
        "heading",
        "paragraph",
        "list_item",
        "list_item",
        "table",
        "paragraph",
        "formula",
        "figure",
    ]
    heading, para, item0, item1, table, code, formula, figure = doc.elements
    assert heading.text == "Title" and heading.level == 1
    assert para.text == "Lead paragraph spans two lines."
    assert (item0.list_depth, item1.list_depth) == (0, 1)
    assert (item0.text, item1.text, item0.marker) == ("item one", "nested item", "-")
    assert table.table is not None and table.table.header_rows == 1
    assert table.table.rows == [["h1", "h2"], ["a", "b"]]
    assert code.flags == ["code"] and code.text == "# not a heading\nx = 1"
    assert formula.text == "E = mc^2"
    assert figure.text == "A diagram" and figure.figure is not None
    assert all(e.boxes == [] and e.source == "text" for e in doc.elements)


def test_element_ids_follow_reading_order():
    doc = load(MIXED, elements=True)
    assert [e.id for e in doc.elements] == [f"e{i}" for i in range(len(doc.elements))]
    assert [e.order for e in doc.elements] == list(range(len(doc.elements)))
    assert load(MIXED, elements=True) == doc  # stable for the same input


def test_markdown_section_element_ids():
    text = "lead\n\n# A\n\npara a\n\n## B\n\n- b1\n- b2\n\n# C\n\nc text\n"
    doc = load(text, elements=True)
    by_id = {e.id: e for e in doc.elements}
    assert doc.sections["s0"].element_ids == ["e0"]
    assert doc.elements[0].section_id == "s0"
    assert [by_id[i].text for i in doc.sections["s1"].element_ids] == ["A", "para a"]
    assert [by_id[i].text for i in doc.sections["s1.1"].element_ids] == ["B", "b1", "b2"]
    assert [by_id[i].text for i in doc.sections["s2"].element_ids] == ["C", "c text"]
    owned = [i for s in doc.sections.values() for i in s.element_ids]
    assert sorted(owned) == sorted(by_id) and len(owned) == len(set(owned))
    for section in doc.sections.values():
        assert all(by_id[i].section_id == section.id for i in section.element_ids)


def test_markdown_elements_off_by_default(tmp_path: Path):
    path = tmp_path / "x.md"
    path.write_text("# A\n\nbody\n", encoding="utf-8")
    doc = load(path)
    assert doc.elements == []
    assert doc.sections["s1"].element_ids == []
    assert "elements" not in doc.model_dump()


def test_load_file_with_elements(tmp_path: Path):
    path = tmp_path / "x.md"
    path.write_text("# A\n\nbody\n", encoding="utf-8")
    assert [e.kind for e in load(path, elements=True).elements] == ["heading", "paragraph"]


def test_parse_markdown_matches_load():
    assert parse_markdown(MIXED, source="<string>", elements=True) == load(MIXED, elements=True)
    assert parse_markdown(MIXED, source="<string>").elements == []


def test_repaired_levels_and_rejected_headings():
    doc = load("# Intro\n\ntext\n\n#### Deep Heading\n\nmore\n", elements=True)
    levels = [(e.text, e.level) for e in doc.elements if e.kind == "heading"]
    assert levels == [("Intro", 1), ("Deep Heading", 2)]  # repaired, as in the section tree
    doc = load("# 12\n\ntext\n", elements=True)  # page-number title: rejected
    assert [e.kind for e in doc.elements] == ["paragraph", "paragraph"]
    assert doc.elements[0].section_id == "s0" and "s0" in doc.sections


def test_setext_headings_front_matter_and_tilde_fences():
    text = "---\ntitle: x\n---\nHead\n====\n\nbody\n\n~~~\n# no\n~~~\n"
    doc = load(text, elements=True)
    assert [(e.kind, e.text) for e in doc.elements] == [
        ("heading", "Head"),
        ("paragraph", "body"),
        ("paragraph", "# no"),
    ]


def test_list_variants_and_thematic_break():
    text = "# L\n\n1. one\n2. two\n   continued\n\n* * *\n\n+ plus\n    - deep\n"
    doc = load(text, elements=True)
    items = [e for e in doc.elements if e.kind == "list_item"]
    assert [(i.marker, i.text, i.list_depth) for i in items] == [
        ("1.", "one", 0),
        ("2.", "two continued", 0),
        ("+", "plus", 0),
        ("-", "deep", 1),
    ]
    assert [e.kind for e in doc.elements].count("paragraph") == 0


def test_table_without_leading_pipes_and_ragged_rows():
    text = "# T\n\nx | y\n--|--\n1 | 2 | 3\n\nafter\n"
    doc = load(text, elements=True)
    table = next(e for e in doc.elements if e.kind == "table")
    assert table.table is not None and table.table.rows == [["x", "y"], ["1", "2", "3"]]
    assert doc.elements[-1].text == "after"


def test_paragraph_interrupted_by_blocks_and_inline_image_stays_text():
    text = "# P\n\nsome text\n- a\nmore ![i](x.png) text\n"
    doc = load(text, elements=True)
    assert [(e.kind, e.text) for e in doc.elements[1:]] == [
        ("paragraph", "some text"),
        ("list_item", "a more ![i](x.png) text"),
    ]


def test_empty_and_blank_documents_have_no_elements():
    assert load("", elements=True).elements == []
    assert load("\n\n  \n", elements=True).elements == []
