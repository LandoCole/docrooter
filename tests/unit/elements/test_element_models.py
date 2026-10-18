import json
from pathlib import Path

from docrooter import Document, Section
from docrooter.elements import Box, Element, FigureData, TableData

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "docs" / "v03_document.json"


def test_v03_document_json_round_trips_unchanged():
    text = FIXTURE.read_text(encoding="utf-8")
    doc = Document.model_validate_json(text)
    assert doc.elements == []
    assert all(s.element_ids == [] for s in doc.sections.values())
    assert doc.model_dump_json(indent=2) == text
    assert json.loads(json.dumps(doc.model_dump())) == json.loads(text)
    assert "elements" not in doc.model_dump()
    assert "element_ids" not in doc.sections["s1"].model_dump()


def test_new_fields_serialise_when_set():
    element = Element(id="e0", kind="paragraph", text="hi", boxes=[], order=0, section_id="s1")
    doc = Document(
        source="x",
        sections={"s1": Section(id="s1", title="T", level=1, text="hi", element_ids=["e0"])},
        root_ids=["s1"],
        elements=[element],
    )
    dumped = doc.model_dump()
    assert dumped["elements"][0]["id"] == "e0"
    assert dumped["sections"]["s1"]["element_ids"] == ["e0"]
    assert Document.model_validate_json(doc.model_dump_json()) == doc


def test_table_to_markdown_pads_ragged_rows():
    table = TableData(rows=[["a", "b"], ["c"]], header_rows=1)
    assert table.to_markdown() == "| a | b |\n| --- | --- |\n| c |  |"


def test_table_to_markdown_without_header_and_escapes():
    table = TableData(rows=[["a|b", "x\ny"]])
    assert table.to_markdown() == "|  |  |\n| --- | --- |\n| a\\|b | x y |"
    assert TableData(rows=[]).to_markdown() == ""


def test_element_defaults():
    element = Element(
        id="e3",
        kind="figure",
        text="",
        boxes=[Box(page=1, x0=0, y0=0, x1=10, y1=10)],
        order=3,
        section_id=None,
        figure=FigureData(source="image"),
    )
    assert element.source == "text" and element.flags == [] and element.confidence is None
    assert element.table is None and element.caption_for is None and element.level is None
    first = Element(id="e4", kind="paragraph", text="t", boxes=[], order=4, section_id=None)
    first.flags.append("x")
    second = Element(id="e5", kind="paragraph", text="t", boxes=[], order=5, section_id=None)
    assert second.flags == []
