"""The real heron layout model on a generated page (title, text, ruled table). Run with
``pytest -m layout``; skipped without the ``layout`` extra. The first run downloads the pinned
weights (~171 MB) into the Hugging Face cache."""

import pytest

from docrooter.elements import Box
from docrooter.layout import HeronLayout
from docrooter.pdf.load import load_pdf

pytestmark = pytest.mark.layout

pytest.importorskip("onnxruntime")
pytest.importorskip("huggingface_hub")
pymupdf = pytest.importorskip("pymupdf")

TITLE = "Quarterly Results Overview"
BODY = "The company reported steady growth in revenue across all of its regions this year."
TABLE = Box(page=1, x0=72, y0=300, x1=540, y1=444)  # 6 rows x 3 columns, fully ruled


def iou(a: Box, b: Box) -> float:
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    inter = max(w, 0.0) * max(h, 0.0)
    union = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter
    return inter / union if union > 0 else 0.0


@pytest.fixture(scope="module")
def pdf_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("heron") / "page.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 90), TITLE, fontsize=24, fontname="hebo")
    y = 140
    for _ in range(7):
        page.insert_text((72, y), BODY, fontsize=11, fontname="helv")
        y += 14
    cw, rh = (TABLE.x1 - TABLE.x0) / 3, (TABLE.y1 - TABLE.y0) / 6
    for r in range(6):
        for c in range(3):
            x, top = TABLE.x0 + c * cw, TABLE.y0 + r * rh
            page.draw_rect(pymupdf.Rect(x, top, x + cw, top + rh), color=(0, 0, 0), width=0.8)
            text = ("Region", "Revenue", "Growth")[c] if r == 0 else f"{(r * 37 + c * 11) % 97}"
            page.insert_text((x + 6, top + 16), text, fontsize=10, fontname="helv")
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture(scope="module")
def engine():
    heron = HeronLayout()
    yield heron
    heron.close()


def test_heron_finds_the_drawn_table(pdf_path, engine):
    regions = engine.detect(pdf_path, [1], dpi=72)
    assert regions and all(r.confidence >= 0.5 and r.page == 1 for r in regions)
    tables = [r for r in regions if r.label == "table"]
    assert tables, [r.label for r in regions]
    assert max(iou(r.box, TABLE) for r in tables) >= 0.5
    # the title is a title or section header, the body lines are text
    title = [r for r in regions if r.box.y0 < 100 and r.box.y1 > 70]
    assert {r.label for r in title} & {"title", "section_header"}
    assert any(r.label == "text" and 130 < r.box.y0 < 240 for r in regions)


def test_heron_regions_type_the_elements(pdf_path, engine):
    doc = load_pdf(pdf_path, elements=True, layout=engine)
    assert doc.meta["layout"] == "heron" and doc.meta["layout_regions"] > 0
    tables = [e for e in doc.elements if e.kind == "table"]
    assert len(tables) == 1 and "from_layout_model" in tables[0].flags
    assert tables[0].table is not None and len(tables[0].table.rows) == 6
    assert any(TITLE in s.title for s in doc.walk())
    assert all("Region" not in e.text for e in doc.elements if e.kind != "table")
