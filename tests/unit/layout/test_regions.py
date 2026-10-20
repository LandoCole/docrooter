"""Layout regions (spec §6) with a fake engine: how regions type lines, overlap resolution,
the confidence cut, Section-header headings, unruled tables, the registry and heron's pure
pre- and post-processing. No onnxruntime here (the unit-test import guard forbids it)."""

import sys

import numpy as np
import pytest

from docrooter.elements.models import Box
from docrooter.errors import DocrooterError, MissingExtraError
from docrooter.layout import (
    HERON_TO_KIND,
    LayoutEngine,
    LayoutRegion,
    apply_regions,
    get_layout_engine,
    register_layout_engine,
)
from docrooter.layout import registry as layout_registry
from docrooter.layout.heron import decode, preprocess, resize_bilinear
from docrooter.layout.recipes import HERON_LABELS, HERON_REVISION
from docrooter.pdf import element_mode
from docrooter.pdf.lines import Line
from docrooter.pdf.load import load_pdf
from docrooter.pdf.objects import extract_page_objects
from docrooter.pdf.reading_order import order_lines

pymupdf = pytest.importorskip("pymupdf")

BODY = "This sentence is ordinary body text that keeps the body size at eleven points"


class FakeLayout:
    """A :class:`LayoutEngine` returning fixed regions."""

    name = "fake-layout"

    def __init__(self, regions, min_confidence=None):
        self.regions = list(regions)
        if min_confidence is not None:
            self.min_confidence = min_confidence
        self.calls = []
        self.closed = 0

    def detect(self, path, pages, *, dpi=None):
        self.calls.append((list(pages), dpi))
        return [r for r in self.regions if r.page in pages]

    def close(self):
        self.closed += 1


def region(label, x0, y0, x1, y1, *, conf=0.9, page=1):
    return LayoutRegion(page=page, box=Box(page=page, x0=x0, y0=y0, x1=x1, y1=y1),
                        label=label, confidence=conf)  # fmt: skip


def around(path, text, *, pad=2.0):
    """A box around the text-layer line(s) starting with ``text`` (union when several)."""
    lines = [ln for p in extract_page_objects(path, engine="pymupdf") for ln in p.lines
             if ln.text.strip().startswith(text)]  # fmt: skip
    assert lines, text
    return (min(ln.x0 for ln in lines) - pad, min(ln.y0 for ln in lines) - pad,
            max(ln.x1 for ln in lines) + pad, max(ln.y1 for ln in lines) + pad)  # fmt: skip


def line(text, x0, y0, x1, y1, page=1):
    return Line(page=page, text=text, size=11.0, bold=False, font="Helvetica", x0=x0, y0=y0,
                x1=x1, y1=y1, page_height=792.0, page_width=612.0, gap_above=4.0,
                gap_below=4.0, block=0)  # fmt: skip


def _pdf(tmp_path, name="doc.pdf"):
    """A page: a 22 pt heading, a body paragraph, a body-size "Scope of work" line inside the
    flow, a two-item bullet list, a code line and an unruled three-column table."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 80), "Introduction", fontsize=22, fontname="hebo")
    y = 120
    for k in range(3):
        page.insert_text((72, y), f"{BODY} {k}", fontsize=11, fontname="helv")
        y += 15
    page.insert_text((72, y), "Scope of work", fontsize=11, fontname="helv")
    y += 15
    for k in range(2):
        page.insert_text((72, y), f"{BODY} again {k}", fontsize=11, fontname="helv")
        y += 15
    y += 20
    for k in range(2):
        page.insert_text((80, y), f"- Bullet item number {k} of the list", fontsize=11)
        y += 15
    y += 20
    page.insert_text((72, y), "result = compute(alpha, beta)", fontsize=11, fontname="cour")
    y += 40
    for r, cells in enumerate([("Region", "Revenue", "Growth"), ("North", "120", "4%"),
                               ("South", "95", "2%"), ("East", "143", "7%")]):  # fmt: skip
        for c, cell in enumerate(cells):
            page.insert_text((72 + 160 * c, y + 18 * r), cell, fontsize=11, fontname="helv")
    page.insert_text((72, y + 120), f"{BODY} closing", fontsize=11, fontname="helv")
    path = tmp_path / name
    doc.save(str(path))
    doc.close()
    return path


def _by_text(doc, start):
    found = [e for e in doc.elements if e.text.startswith(start)]
    assert found, start
    return found[0]


# --- the brief's tests -----------------------------------------------------------------------


def test_region_kinds_override_heuristics(tmp_path):
    path = _pdf(tmp_path)
    plain = load_pdf(path, elements=True)
    assert _by_text(plain, "Bullet item number 0").kind == "list_item"
    assert _by_text(plain, BODY).kind == "paragraph"

    engine = FakeLayout([
        region("text", *around(path, "- Bullet item number 0")),
        region("footnote", *around(path, BODY + " 0")),
        region("code", *around(path, "result = compute")),
        region("page_footer", *around(path, "- Bullet item number 1"), conf=0.7),
    ])  # fmt: skip
    doc = load_pdf(path, elements=True, layout=engine)
    item = _by_text(doc, "- Bullet item number 0")
    assert item.kind == "paragraph" and "from_layout_model" in item.flags
    assert item.confidence == pytest.approx(0.9)
    assert _by_text(doc, BODY + " 0").kind == "footnote"
    code = _by_text(doc, "result = compute")
    assert code.kind == "paragraph" and "code" in code.flags
    assert _by_text(doc, "- Bullet item number 1").kind == "page_footer"
    assert doc.meta["layout"] == "fake-layout" and doc.meta["layout_regions"] == 4
    assert engine.calls == [([1], None)]  # the engine renders at its own resolution
    assert engine.closed == 0  # an instance stays the caller's


def test_overlap_resolution_confidence_then_area():
    lines = [line("inside both", 100, 100, 200, 110), line("only big", 100, 300, 200, 310)]
    big = region("text", 50, 50, 500, 500, conf=0.8)
    small = region("table", 90, 90, 210, 120, conf=0.6)
    kinds, used = apply_regions(lines, [big, small])
    assert kinds == {0: "paragraph", 1: "paragraph"}  # the more confident region wins
    assert dict((r.label, idxs) for r, idxs in used) == {"text": [0, 1], "table": []}

    small = region("table", 90, 90, 210, 120, conf=0.8)  # equal confidence: smaller wins
    kinds, used = apply_regions(lines, [big, small])
    assert kinds == {0: "table", 1: "paragraph"}
    assert dict((r.label, idxs) for r, idxs in used) == {"text": [1], "table": [0]}


def test_low_confidence_regions_ignored(tmp_path):
    lines = [line("a line", 100, 100, 200, 110)]
    kinds, used = apply_regions(lines, [region("table", 50, 50, 500, 500, conf=0.49)])
    assert kinds == {} and used == []
    kinds, _ = apply_regions(lines, [region("table", 50, 50, 500, 500, conf=0.5)])
    assert kinds == {0: "table"}  # 0.5 itself is used (spec: >= 0.5)

    path = _pdf(tmp_path)
    engine = FakeLayout([region("text", *around(path, "Introduction"), conf=0.3)])
    doc = load_pdf(path, elements=True, layout=engine)
    heading = _by_text(doc, "Introduction")
    assert heading.kind == "title" and heading.level == 1  # the heuristics' title heading
    assert "from_layout_model" not in heading.flags and doc.meta["layout_regions"] == 0


def test_engine_min_confidence_sets_the_cut(tmp_path):
    path = _pdf(tmp_path)
    low = region("text", *around(path, "Introduction"), conf=0.4)
    doc = load_pdf(path, elements=True, layout=FakeLayout([low]))  # default cut 0.5
    assert _by_text(doc, "Introduction").kind == "title" and doc.meta["layout_regions"] == 0
    doc = load_pdf(path, elements=True, layout=FakeLayout([low], min_confidence=0.3))
    element = _by_text(doc, "Introduction")
    assert element.kind == "paragraph" and element.confidence == pytest.approx(0.4)
    assert doc.meta["layout_regions"] == 1
    strict = FakeLayout([region("text", *around(path, "Introduction"), conf=0.6)], 0.7)
    assert _by_text(load_pdf(path, elements=True, layout=strict), "Introduction").kind == "title"


def test_section_header_on_the_detected_title_is_the_title(tmp_path):
    path = _pdf(tmp_path)
    engine = FakeLayout([region("section_header", *around(path, "Introduction")),
                         region("section_header", *around(path, "Scope of work"))])  # fmt: skip
    doc = load_pdf(path, elements=True, layout=engine)
    title = _by_text(doc, "Introduction")
    assert title.kind == "title" and title.level == 1 and "from_layout_model" in title.flags
    assert _by_text(doc, "Scope of work").kind == "heading"
    assert [s.title for s in doc.toc()][:2] == ["Introduction", "Scope of work"]


def test_largest_section_header_on_page_one_is_the_title_without_a_detected_one(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    y = 90
    for text, size in [(f"{BODY} a", 11), ("Annual overview", 12), (f"{BODY} b", 11),
                       ("Scope of work", 11), (f"{BODY} c", 11), (f"{BODY} d", 11)]:  # fmt: skip
        page.insert_text((72, y), text, fontsize=size, fontname="helv")
        y += 15
    path = tmp_path / "plain.pdf"
    doc.save(str(path))
    doc.close()
    plain = load_pdf(path, elements=True)
    assert not [e for e in plain.elements if e.kind in ("title", "heading")]  # none detected
    engine = FakeLayout([region("section_header", *around(path, "Scope of work")),
                         region("section_header", *around(path, "Annual overview"))])  # fmt: skip
    out = load_pdf(path, elements=True, layout=engine)
    assert _by_text(out, "Annual overview").kind == "title"
    assert _by_text(out, "Scope of work").kind == "heading"
    assert {s.title for s in out.toc()} == {"Annual overview", "Scope of work"}


def test_section_header_region_creates_section(tmp_path):
    path = _pdf(tmp_path)
    assert "Scope of work" not in [s.title for s in load_pdf(path, elements=True).walk()]

    engine = FakeLayout([region("section_header", *around(path, "Scope of work"), conf=0.8)])
    doc = load_pdf(path, elements=True, layout=engine)
    element = _by_text(doc, "Scope of work")
    assert element.kind == "heading" and "from_layout_model" in element.flags
    section = doc.section("Scope of work")
    assert section.level == 2  # smaller than the level-1 "Introduction"
    assert element.id in section.element_ids and element.section_id == section.id
    assert (BODY + " again 0") in section.text
    assert (BODY + " 0") in doc.section("Introduction").text


def test_section_header_region_on_a_running_header_is_not_a_section(tmp_path):
    doc = pymupdf.open()
    for k in range(3):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 40), "ACME Annual Report", fontsize=9)
        page.insert_text((72, 100), f"Chapter {k + 1}", fontsize=20, fontname="hebo")
        for j in range(4):
            page.insert_text((72, 140 + 15 * j), f"{BODY} {j}", fontsize=11)
    path = tmp_path / "running.pdf"
    doc.save(str(path))
    doc.close()
    box = around(path, "ACME Annual Report")  # union over the three pages: page 1's box fits
    engine = FakeLayout([region("section_header", *box, page=p) for p in (1, 2, 3)])
    out = load_pdf(path, elements=True, layout=engine)
    assert [s.title for s in out.toc()] == ["Chapter 1", "Chapter 2", "Chapter 3"]
    assert {e.kind for e in out.elements if e.text == "ACME Annual Report"} == {"page_header"}


def test_heading_candidate_inside_text_region_is_not_heading(tmp_path):
    path = _pdf(tmp_path)
    assert "Introduction" in [s.title for s in load_pdf(path, elements=True).walk()]

    for label in ("text", "table"):
        engine = FakeLayout([region(label, *around(path, "Introduction"))])
        doc = load_pdf(path, elements=True, layout=engine)
        assert "Introduction" not in [s.title for s in doc.walk()]
        element = _by_text(doc, "Introduction")
        assert element.kind == ("paragraph" if label == "text" else "table")
        assert any("Introduction" in s.text for s in doc.walk())  # its text is kept


@pytest.mark.parametrize("engine_name", ["pymupdf", "pdfplumber"])
def test_unruled_table_region_uses_text_strategy(tmp_path, monkeypatch, engine_name):
    if engine_name == "pdfplumber":
        pytest.importorskip("pdfplumber")
    path = _pdf(tmp_path)
    plain = load_pdf(path, elements=True)
    assert not [e for e in plain.elements if e.kind == "table"]  # no rulings: no table

    calls = []
    real = element_mode.find_text_tables

    def spy(p, engine, requests):
        found = real(p, engine, requests)
        calls.append((engine, {page: list(boxes) for page, boxes in requests.items()}, found))
        return found

    monkeypatch.setattr(element_mode, "find_text_tables", spy)
    x0, y0, _, _ = around(path, "Region")
    _, _, x1, y1 = around(path, "East")
    x1 = around(path, "Growth")[2]
    engine = FakeLayout([region("table", x0, y0, x1 + 4, y1 + 2)])
    doc = load_pdf(path, elements=True, engine=engine_name, layout=engine)
    (call,) = [c for c in calls if c[1]]
    assert call[0] == engine_name and len(call[1][1]) == 1 and (1, 0) in call[2]
    (table,) = [e for e in doc.elements if e.kind == "table"]
    assert "from_layout_model" in table.flags
    rows = [[c for c in row if c] for row in table.table.rows]
    assert rows == [["Region", "Revenue", "Growth"], ["North", "120", "4%"],
                    ["South", "95", "2%"], ["East", "143", "7%"]]  # fmt: skip
    assert not [e for e in doc.elements if e.kind != "table" and "North" in e.text]


def test_lines_outside_regions_use_heuristics(tmp_path):
    path = _pdf(tmp_path)
    engine = FakeLayout([region("text", *around(path, BODY + " 0"))])
    doc = load_pdf(path, elements=True, layout=engine)
    assert _by_text(doc, BODY + " 0").kind == "paragraph"
    assert "from_layout_model" in _by_text(doc, BODY + " 0").flags
    heading = _by_text(doc, "Introduction")
    assert heading.kind == "title" and "from_layout_model" not in heading.flags
    item = _by_text(doc, "Bullet item number 0")
    assert item.kind == "list_item" and item.flags == [] and item.confidence is None
    plain = load_pdf(path, elements=True)
    assert [e.kind for e in plain.elements if not e.text.startswith(BODY)] == [
        e.kind for e in doc.elements if not e.text.startswith(BODY)
    ]


def test_missing_onnxruntime_raises_layout_extra_message(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # import fails as if not installed
    with pytest.raises(MissingExtraError, match=r'pip install "docrooter\[layout\] @ git\+'):
        get_layout_engine("heron")

    def no_page_work(*args, **kwargs):
        raise AssertionError("pages were read before the layout engine was checked")

    monkeypatch.setattr(element_mode, "extract_page_objects", no_page_work)
    with pytest.raises(MissingExtraError, match=r"docrooter\[layout\]"):
        load_pdf(_pdf(tmp_path), elements=True, layout="heron")


# --- more region behaviour -------------------------------------------------------------------


def test_picture_region_is_a_figure_and_caption_region_links(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 80), BODY, fontsize=11)
    page.insert_text((150, 200), "axis label", fontsize=8)
    page.insert_text((72, 330), "Figure 1: Sales by region", fontsize=9)
    path = tmp_path / "fig.pdf"
    doc.save(str(path))
    doc.close()
    engine = FakeLayout([
        region("picture", 72, 120, 400, 310, conf=0.95),
        region("caption", *around(path, "Figure 1"), conf=0.85),
    ])  # fmt: skip
    out = load_pdf(path, elements=True, layout=engine)
    (figure,) = [e for e in out.elements if e.kind == "figure"]
    assert figure.figure.source == "layout_model" and figure.text == "axis label"
    assert (figure.boxes[0].y0, figure.boxes[0].y1) == (120, 310)
    (caption,) = [e for e in out.elements if e.kind == "caption"]
    assert caption.caption_for == figure.id and figure.caption_id == caption.id

    empty = FakeLayout([region("picture", 300, 400, 500, 600)])
    (figure,) = [e for e in load_pdf(path, elements=True, layout=empty).elements
                 if e.kind == "figure"]  # fmt: skip
    assert figure.text == "" and figure.figure.source == "layout_model"


def test_table_region_takes_its_ruled_table_and_lines_once(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 80), BODY, fontsize=11)
    x0, y0, cw, rh = 72, 120, 150, 20
    names = ("North", "South", "East", "West")
    for r in range(5):
        page.draw_line((x0, y0 + r * rh), (x0 + 3 * cw, y0 + r * rh))
    for c in range(4):
        page.draw_line((x0 + c * cw, y0), (x0 + c * cw, y0 + 4 * rh))
    for r in range(4):
        for c in range(3):
            text = names[r] if c == 0 else f"{r * 10 + c}"
            page.insert_text((x0 + c * cw + 5, y0 + r * rh + 14), text, fontsize=10)
    page.insert_text((72, 260), f"{BODY} below", fontsize=11)
    path = tmp_path / "ruled.pdf"
    doc.save(str(path))
    doc.close()
    # the Table region covers the top three rows; a later Text region covers the last one
    engine = FakeLayout([region("table", 70, 118, 524, 181),
                         region("text", 70, 182, 524, 201)])  # fmt: skip
    out = load_pdf(path, elements=True, engine="pymupdf", layout=engine)
    (table,) = [e for e in out.elements if e.kind == "table"]
    assert "from_layout_model" in table.flags and len(table.table.rows) == 4
    assert (table.boxes[0].y0, table.boxes[0].y1) == pytest.approx((118, 200), abs=1)
    assert [e.kind for e in out.elements if "West" in e.text] == ["table"]  # taken once


def test_page_footer_region_of_a_number_is_page_number(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 80), BODY, fontsize=11)
    page.insert_text((300, 500), "17", fontsize=9)
    path = tmp_path / "num.pdf"
    doc.save(str(path))
    doc.close()
    engine = FakeLayout([region("page_footer", *around(path, "17"))])
    out = load_pdf(path, elements=True, layout=engine)
    assert _by_text(out, "17").kind == "page_number"


def test_region_lines_order_as_one_unit():
    left = [line(f"left {k}", 72, 100 + 20 * k, 290, 110 + 20 * k) for k in range(2)]
    right = [line(f"right {k}", 320, 100 + 20 * k, 540, 110 + 20 * k) for k in range(2)]
    lines = [left[0], right[0], left[1], right[1]]  # extractor order: row by row
    groups = {0: (70, 98, 292, 132), 2: (70, 98, 292, 132), 1: (318, 98, 542, 132),
              3: (318, 98, 542, 132)}  # fmt: skip
    assert order_lines(lines, set(), groups) == [0, 2, 1, 3]
    assert order_lines(lines, set(), {}) == order_lines(lines, set())


def test_named_layout_engine_is_created_and_closed(tmp_path, monkeypatch):
    path = _pdf(tmp_path)
    made = []

    def factory():
        made.append(FakeLayout([region("text", *around(path, "Introduction"))]))
        return made[-1]

    monkeypatch.setitem(layout_registry._FACTORIES, "fake", factory)
    doc = load_pdf(path, elements=True, layout="fake")
    assert len(made) == 1 and made[0].closed == 1
    assert _by_text(doc, "Introduction").kind == "paragraph"


def test_named_layout_engine_is_closed_when_loading_fails_before_detect(tmp_path, monkeypatch):
    # Final review M1: an engine created by name is closed on any failure before detect
    made = []

    def factory():
        made.append(FakeLayout([]))
        return made[-1]

    monkeypatch.setitem(layout_registry._FACTORIES, "fake", factory)
    good = _pdf(tmp_path)
    bad = tmp_path / "corrupt.pdf"
    bad.write_bytes(b"%PDF-1.7 this is not a PDF")
    failing = [
        dict(path=good, ocr_dpi=0),  # argument errors
        dict(path=good, tables="some"),
        dict(path=good, max_pages=0),
        dict(path=good, ocr="no-such-ocr"),
        dict(path=bad),  # extraction fails
    ]
    for kwargs in failing:
        with pytest.raises(Exception):  # noqa: B017, PT011 - each case raises its own error
            load_pdf(kwargs.pop("path"), elements=True, layout="fake", **kwargs)
    assert all(engine.closed == 1 for engine in made), [e.closed for e in made]
    assert len(made) == 1  # only the extraction failure got past the argument checks


def test_registry():
    assert isinstance(FakeLayout([]), LayoutEngine)
    with pytest.raises(DocrooterError, match="unknown layout engine 'nope'"):
        get_layout_engine("nope")
    for name in ("", "heuristic"):
        with pytest.raises(DocrooterError, match="not a valid layout engine name"):
            register_layout_engine(name, lambda: FakeLayout([]))
    try:
        register_layout_engine("mine", lambda: FakeLayout([]))
        assert isinstance(get_layout_engine("mine"), FakeLayout)
    finally:
        layout_registry._FACTORIES.pop("mine", None)


def test_heron_mapping_covers_every_label():
    assert set(HERON_TO_KIND) == set(HERON_LABELS) and len(HERON_LABELS) == 17
    assert HERON_TO_KIND["document_index"] == "contents"
    assert HERON_TO_KIND["section_header"] == "heading"
    assert HERON_TO_KIND["picture"] == "figure"
    assert len(HERON_REVISION) == 40


# --- heron pre- and post-processing (pure numpy) ---------------------------------------------


def test_preprocess_gives_uint8_chw_at_input_size():
    rgb = np.zeros((792, 612, 3), dtype=np.uint8)
    rgb[:, :, 0] = 200
    out = preprocess(rgb)
    assert out.shape == (1, 3, 640, 640) and out.dtype == np.uint8
    assert (out[0, 0] == 200).all() and (out[0, 1:] == 0).all()
    # a black band keeps its place after resizing
    rgb = np.full((800, 600, 3), 255, dtype=np.uint8)
    rgb[200:400] = 0
    small = resize_bilinear(rgb, 300, 400)
    assert (small[110:190] == 0).all() and (small[:90] == 255).all()


def test_decode_scales_filters_and_clips():
    labels = np.array([8, 9, 7, 99, 6])
    boxes = np.array([[61.2, 79.2, 306.0, 396.0], [0, 0, 10, 10], [-5, 700, 700, 800],
                      [0, 0, 5, 5], [10, 10, 10, 20]], dtype=np.float32)  # fmt: skip
    scores = np.array([0.95, 0.4, 0.7, 0.9, 0.8], dtype=np.float32)
    out = decode(labels, boxes, scores, page=3, width_pt=612, height_pt=792, width_px=1224,
                 height_px=1584, min_confidence=0.5)  # fmt: skip
    assert [(r.label, r.confidence) for r in out] == [("table", 0.95), ("section_header", 0.7)]
    table, header = out
    assert table.page == 3 and (table.box.x0, table.box.y0, table.box.x1, table.box.y1) == (
        30.6, 39.6, 153.0, 198.0)  # fmt: skip
    assert (header.box.x0, header.box.y1) == (0.0, 400.0)  # clipped at the left edge


class _FakeSession:
    def __init__(self, path, sess_options=None, providers=None):
        self.path, self.options, self.providers = path, sess_options, providers
        self.feeds = []

    def run(self, names, feeds):
        assert names == ["labels", "boxes", "scores"]
        self.feeds.append(feeds)
        w, h = (int(v) for v in feeds["orig_target_sizes"][0])
        boxes = np.array([[[0.1 * w, 0.1 * h, 0.5 * w, 0.2 * h], [0, 0, w, h]]], np.float32)
        return np.array([[10, 6]]), boxes, np.array([[0.9, 0.3]], np.float32)


def _fake_runtime(monkeypatch, *, download=None):
    """Stand-ins for onnxruntime and huggingface_hub (never the real packages here)."""
    import types

    ort = types.ModuleType("onnxruntime")
    ort.SessionOptions = lambda: types.SimpleNamespace(intra_op_num_threads=0)
    ort.InferenceSession = _FakeSession
    hub = types.ModuleType("huggingface_hub")
    downloads = []

    def hf_hub_download(repo, filename, revision):
        downloads.append((repo, filename, revision))
        if download is not None:
            raise download
        return "/cache/model.onnx"

    hub.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return downloads


def test_heron_loads_lazily_once_and_scales_boxes(tmp_path, monkeypatch):
    from docrooter.layout import HeronLayout
    from docrooter.layout.recipes import HERON_REPO

    downloads = _fake_runtime(monkeypatch)
    monkeypatch.setenv("DOCROOTER_THREADS", "3")
    path = _pdf(tmp_path)
    heron = HeronLayout()
    assert heron._session is None and downloads == []  # nothing loaded at creation
    regions = heron.detect(path, [1], dpi=72)
    regions += heron.detect(path, [1], dpi=144)
    assert downloads == [(HERON_REPO, "model.onnx", HERON_REVISION)]  # one download, reused
    session = heron._session
    assert (
        session.providers == ["CPUExecutionProvider"] and session.options.intra_op_num_threads == 3
    )
    assert [f["images"].shape for f in session.feeds] == [(1, 3, 640, 640)] * 2
    assert [list(f["orig_target_sizes"][0]) for f in session.feeds] == [[612, 792], [1224, 1584]]
    assert [(r.label, r.confidence) for r in regions] == [("title", 0.9)] * 2  # 0.3 cut
    for r in regions:  # the same page box at both resolutions, in points
        assert (r.box.x0, r.box.y0, r.box.x1, r.box.y1) == pytest.approx((61.2, 79.2, 306, 158.4))
    heron.close()
    assert heron._session is None


def test_heron_weights_unavailable_is_a_missing_extra_error(tmp_path, monkeypatch):
    from docrooter.layout import HeronLayout

    _fake_runtime(monkeypatch, download=OSError("offline"))
    with pytest.raises(MissingExtraError, match=r"heron layout weights") as raised:
        HeronLayout().detect(_pdf(tmp_path), [1], dpi=72)
    message = str(raised.value)
    assert "docling-project/docling-layout-heron-onnx" in message and HERON_REVISION in message
    assert "OSError: offline" in message and "network" in message
    assert "pip install" not in message  # the extra is installed; installing is not the fix
    with pytest.raises(ValueError, match="min_confidence"):
        HeronLayout(min_confidence=1.5)
