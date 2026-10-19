"""Element mode in ``load_pdf(elements=True)``: routing, OCR with a fake engine, XY-cut order,
elements and the sections built from them."""

import itertools
import math
from collections import Counter
from pathlib import Path

import pytest

from docrooter import load
from docrooter.elements.models import Box
from docrooter.errors import DocrooterError
from docrooter.ocr import registry as ocr_registry
from docrooter.ocr.base import OcrEngine, OcrLine
from docrooter.pdf.element_mode import section_text
from docrooter.pdf.lines import extract_lines
from docrooter.pdf.load import load_pdf
from docrooter.pdf.objects import extract_page_objects

pymupdf = pytest.importorskip("pymupdf")

GOLDEN = Path(__file__).parent / "golden"
BODY = "This sentence is ordinary body text that stays small and keeps the body size at eleven."
FURNITURE = {"page_header", "page_footer", "page_number"}
BODY_PX = 11 / 0.75  # an OCR box this tall gives an 11 pt line
HEADING_PX = 22 / 0.75


class FakeOcr:
    """An :class:`OcrEngine` that reads known text: ``pages`` maps a page number to its
    ``(text, box height)`` lines, ``regions`` maps a page number to the text of any clipped
    region on it."""

    name = "fake"

    def __init__(self, pages=None, regions=None, fail=False, region_conf=(0.8,)):
        self.pages = pages or {}
        self.regions = regions or {}
        self.region_conf = list(region_conf)
        self.fail = fail
        self.calls = []
        self.closed = 0

    def recognize(self, image, *, width_px, height_px, page):
        assert image.startswith(b"\x89PNG") and width_px > 0 and height_px > 0
        self.calls.append(page)
        if self.fail:
            raise RuntimeError("engine broke")
        if page.x0 > 0 or page.y0 > 0:  # a clipped region
            text = self.regions.get(page.page)
            box = Box(page=page.page, x0=page.x0 + 5, y0=page.y0 + 5, x1=page.x1 - 5,
                      y1=page.y0 + 5 + BODY_PX)  # fmt: skip
            conf = self.region_conf[min(len(self.calls), len(self.region_conf)) - 1]
            return [OcrLine(text=text, box=box, confidence=conf)] if text else []
        out, y = [], 80.0
        for text, height in self.pages.get(page.page, []):
            box = Box(page=page.page, x0=72, y0=y, x1=72 + 5.5 * len(text), y1=y + height)
            out.append(OcrLine(text=text, box=box, confidence=0.9))
            y += height + (20 if height > BODY_PX else 4)
        return out

    def close(self):
        self.closed += 1


def _save(doc, path):
    doc.save(str(path))
    doc.close()
    return path


def _text_page(doc, heading=None, *, lines=4, start=90.0, body=BODY):
    page = doc.new_page(width=612, height=792)
    y = start
    if heading:
        page.insert_text((72, y), heading, fontsize=22, fontname="hebo")
        y += 40
    for k in range(lines):
        page.insert_text((72, y), f"{body} Line {k}.", fontsize=11, fontname="helv")
        y += 15
    return page


def _scan_page(doc):
    page = doc.new_page(width=612, height=792)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 61, 79), False)
    pix.set_rect(pix.irect, (190, 190, 190))
    page.insert_image(page.rect, stream=pix.tobytes("png"), keep_proportion=False)
    return page


def _ocr_page(heading, n=4, body="Scanned body text that the engine read from the image"):
    lines = [(heading, HEADING_PX)] if heading else []
    return lines + [(f"{body} line {k}.", BODY_PX) for k in range(n)]


def _members(doc):
    """Section id of every element, from ``Section.element_ids``."""
    owner = {}
    for section in doc.walk():
        for eid in section.element_ids:
            assert eid not in owner, f"{eid} is in two sections"
            owner[eid] = section.id
    return owner


def _richer_pdf(tmp_path):
    doc = pymupdf.open()
    page = _text_page(doc, "Quarterly Review", lines=3)
    page.insert_text((72, 40), "ACME Quarterly", fontsize=9, fontname="helv")
    page.insert_text((300, 770), "1", fontsize=9, fontname="helv")
    for k in range(3):
        page.insert_text((80, 200 + 14 * k), f"- Bullet point number {k}", fontsize=11)
    page.insert_text((72, 270), "Table 1. Results", fontsize=9)
    x0, y0, cw, rh = 72, 280, 150, 20
    for r in range(4):
        page.draw_line((x0, y0 + r * rh), (x0 + 3 * cw, y0 + r * rh))
    for c in range(4):
        page.draw_line((x0 + c * cw, y0), (x0 + c * cw, y0 + 3 * rh))
    for r in range(3):
        for c in range(3):
            page.insert_text((x0 + c * cw + 4, y0 + r * rh + 14), f"r{r}c{c}", fontsize=9)
    page.insert_text((72, 380), "Outlook", fontsize=16, fontname="hebo")
    for k in range(3):
        page.insert_text((72, 405 + 15 * k), f"{BODY} Outlook {k}.", fontsize=11)
    page.insert_text((72, 720), "1 A footnote set in small type.", fontsize=7)
    for n in (2, 3):
        page = _text_page(doc, f"Chapter {n}")
        page.insert_text((72, 40), "ACME Quarterly", fontsize=9, fontname="helv")
        page.insert_text((300, 770), str(n), fontsize=9, fontname="helv")
    return _save(doc, tmp_path / "rich.pdf")


# --- sections from elements --------------------------------------------------------------


def test_element_mode_sections_built_from_elements(tmp_path):
    doc = load_pdf(_richer_pdf(tmp_path), elements=True)
    assert doc.meta["element_mode"] is True and len(doc.meta["routes"]) == 3
    kinds = Counter(e.kind for e in doc.elements)
    for kind in ("title", "heading", "paragraph", "list_item", "table", "caption", "footnote",
                 "page_header", "page_number"):  # fmt: skip
        assert kinds[kind] >= 1, (kind, kinds)

    owner = _members(doc)
    for element in doc.elements:
        if element.kind not in FURNITURE:
            assert element.section_id is not None and owner[element.id] == element.section_id
        if element.id in owner:
            assert owner[element.id] == element.section_id
    by_id = {e.id: e for e in doc.elements}
    for section in doc.walk():
        members = [by_id[eid] for eid in section.element_ids]
        assert [e.order for e in members] == sorted(e.order for e in members)
        expected = "\n\n".join(t for t in (section_text(e) for e in members) if t)
        assert section.text == expected

    heads = [e for e in doc.elements if e.kind in ("heading", "title") and e.level is not None]
    assert [(s.title, s.level) for s in doc.walk() if s.level > 0] == [
        (e.text, e.level) for e in heads
    ]
    assert [s.title for s in doc.walk()] == ["Quarterly Review", "Outlook", "Chapter 2",
                                             "Chapter 3"]  # fmt: skip
    review = doc.section("Quarterly Review")
    assert "- Bullet point number 0" in review.text  # list items keep their marker
    assert "| r0c0 | r0c1 | r0c2 |" in review.text  # tables as Markdown
    assert "Table 1. Results" in review.text
    assert "1 A footnote set in small type." in doc.section("Outlook").text
    assert all("ACME" not in s.text for s in doc.walk())  # furniture is not section text


def test_ruled_page_frame_keeps_headings_and_sections(tmp_path):
    """A page drawn in a frame with a header rule and a centre rule is not one table that
    takes the headings: element mode finds the sections the fast path finds."""
    doc = pymupdf.open()
    for n in (1, 2):
        page = doc.new_page(width=612, height=792)
        page.draw_rect(pymupdf.Rect(30, 30, 582, 762), color=(0, 0, 0))
        page.draw_line((30, 90), (582, 90))
        page.draw_line((306, 90), (306, 762))
        page.insert_text((50, 70), f"{n}. Introduction to the Results", fontsize=18,
                         fontname="hebo")  # fmt: skip
        for x in (50, 320):
            for k in range(20):
                page.insert_text((x, 120 + 15 * k), f"Column text line {k} has words.",
                                 fontsize=10, fontname="helv")  # fmt: skip
    path = _save(doc, tmp_path / "framed.pdf")
    fast = load_pdf(path)
    out = load_pdf(path, elements=True)
    titles = [s.title for s in fast.walk()]
    assert titles == ["1. Introduction to the Results", "2. Introduction to the Results"]
    assert [s.title for s in out.walk()] == titles
    assert not [e for e in out.elements if e.kind == "table"]
    assert "Column text line 19 has words." in out.section(titles[1]).text


def test_figure_label_text_is_section_text(tmp_path):
    doc = pymupdf.open()
    page = _text_page(doc, "Results")
    page.draw_rect(pymupdf.Rect(72, 250, 400, 450), color=(0, 0, 0), fill=(0.6, 0.7, 0.9))
    page.insert_text((90, 300), "Q1", fontsize=8)
    page.insert_text((300, 420), "Q4", fontsize=8)
    loaded = load_pdf(_save(doc, tmp_path / "fig.pdf"), elements=True)
    (figure,) = [e for e in loaded.elements if e.kind == "figure"]
    assert figure.figure.source == "vector" and "Q1" in figure.text
    assert "Q1" in loaded.section("Results").text and "Q4" in loaded.section("Results").text


def test_two_column_toc_order_fixed():
    path = GOLDEN / "two_column.pdf"
    fast = [e.title for e in load_pdf(path).toc()]
    assert "1.1 Column Topic 1.1 2.1 Column Topic 2.1" in fast  # rows across columns, merged
    doc = load_pdf(path, elements=True)
    assert [e.title for e in doc.toc()] == [
        "A Study of Two Columns", "1.1 Column Topic 1.1", "1.2 Column Topic 1.2",
        "2.1 Column Topic 2.1", "2.2 Column Topic 2.2", "3.1 Column Topic 3.1",
        "3.2 Column Topic 3.2", "4.1 Column Topic 4.1", "4.2 Column Topic 4.2",
    ]  # fmt: skip
    assert doc.section("1.2 Column Topic 1.2").text.startswith("Column text 1.2 line 0")
    assert "2.1" not in doc.section("1.2 Column Topic 1.2").text


def test_single_column_toc_and_text_match_fast():
    path = GOLDEN / "single_column.pdf"
    fast, elements = load_pdf(path), load_pdf(path, elements=True)
    assert [(e.title, e.level) for e in elements.toc()] == [(e.title, e.level) for e in fast.toc()]
    for a, b in zip(fast.walk(), elements.walk(), strict=True):
        assert a.title == b.title and a.page_start == b.page_start
        # same words; element mode joins a paragraph's lines with spaces, not newlines
        assert a.text.split() == b.text.split()


# --- routing and OCR ---------------------------------------------------------------------


def test_mixed_scanned_and_text_pages_keep_page_order(tmp_path):
    doc = pymupdf.open()
    _text_page(doc, "1 Alpha")
    _scan_page(doc)
    page = _text_page(doc, None, lines=3, body="continued on the third page")
    page.insert_text((72, 200), "3 Gamma", fontsize=22, fontname="hebo")
    for k in range(3):
        page.insert_text((72, 240 + 15 * k), f"{BODY} Gamma {k}.", fontsize=11)
    path = _save(doc, tmp_path / "mixed.pdf")
    engine = FakeOcr(pages={2: _ocr_page("2 Beta")})

    loaded = load_pdf(path, elements=True, ocr=engine)
    routes = loaded.meta["routes"]
    assert [(r["target"], r["task"], r["engine"]) for r in routes] == [
        (1, "text", "text"), (2, "ocr", "fake"), (3, "text", "text"),
    ]  # fmt: skip
    assert [c.page for c in engine.calls] == [2] and engine.closed == 0  # caller's engine
    assert routes[1]["metrics"]["ocr_lines"] == 5.0
    pages = [e.boxes[0].page for e in loaded.elements]
    assert pages == sorted(pages)
    assert [e.order for e in loaded.elements] == list(range(len(loaded.elements)))
    ocr = [e for e in loaded.elements if e.source == "ocr"]
    assert ocr and {e.boxes[0].page for e in ocr} == {2}
    assert all(e.confidence == pytest.approx(0.9) for e in ocr)
    assert [(s.title, s.page_start) for s in loaded.walk()] == [
        ("1 Alpha", 1), ("2 Beta", 2), ("3 Gamma", 3),
    ]  # fmt: skip
    beta = loaded.section("2 Beta")
    assert beta.text.startswith("Scanned body text that the engine read from the image line 0.")
    assert "continued on the third page" in beta.text  # the section runs on across pages
    assert "Gamma 0" in loaded.section("3 Gamma").text
    assert "ocr_skipped" not in loaded.meta["warnings"]


def test_scanned_pdf_with_fake_ocr_has_sections():
    engine = FakeOcr(pages={1: _ocr_page("Inspection Report"), 2: _ocr_page("Findings", 6)})
    doc = load_pdf(GOLDEN / "scanned.pdf", elements=True, ocr=engine)
    assert [s.title for s in doc.walk()] == ["Inspection Report", "Findings"]
    assert all(s.text for s in doc.walk())
    assert doc.meta["warnings"] == []
    assert not [e for e in doc.elements if e.kind == "figure"]  # the scan is a background
    assert {e.source for e in doc.elements} == {"ocr"}


def test_huge_scanned_page_is_ocred_at_a_capped_dpi(tmp_path):
    # Final review R-I2: the render is capped at MAX_RENDER_PIXELS and the route says so
    from docrooter.ocr.render import MAX_RENDER_PIXELS

    doc = pymupdf.open()
    page = doc.new_page(width=3600, height=3600)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 8, 8), False)
    pix.set_rect(pix.irect, (190, 190, 190))
    page.insert_image(page.rect, stream=pix.tobytes("png"), keep_proportion=False)
    path = _save(doc, tmp_path / "huge.pdf")
    sizes = []

    class Sizes(FakeOcr):
        def recognize(self, image, *, width_px, height_px, page):
            sizes.append(width_px * height_px)
            return super().recognize(image, width_px=width_px, height_px=height_px, page=page)

    out = load_pdf(path, elements=True, ocr=Sizes(pages={1: _ocr_page("Survey")}))
    (route,) = out.meta["routes"]
    assert route["task"] == "ocr" and sizes and max(sizes) <= MAX_RENDER_PIXELS
    assert 0 < route["metrics"]["dpi_capped"] < 200
    normal = load_pdf(GOLDEN / "scanned.pdf", elements=True, ocr=FakeOcr(pages={1: _ocr_page("A")}))
    assert all("dpi_capped" not in r["metrics"] for r in normal.meta["routes"])


def test_scanned_pdf_without_ocr_warns():
    doc = load_pdf(GOLDEN / "scanned.pdf", elements=True)
    assert list(doc.sections) == ["s0"] and doc.sections["s0"].text == ""
    assert doc.toc() == []
    assert doc.meta["warnings"] == ["no_text_layer", "ocr_skipped"]
    routes = doc.meta["routes"]
    assert [(r["scope"], r["target"], r["task"], r["engine"]) for r in routes] == [
        ("page", 1, "ocr", "none"), ("page", 2, "ocr", "none"),
    ]  # fmt: skip
    assert all(r["reason"].startswith("needs_ocr: scanned page") for r in routes)


def test_garbled_text_layer_is_replaced_by_ocr():
    path = GOLDEN / "garbled.pdf"
    kept = load_pdf(path, elements=True)
    assert "�" in kept.sections["s0"].text and kept.meta["warnings"] == ["ocr_skipped"]
    engine = FakeOcr(pages={1: _ocr_page("Readable Title")})
    doc = load_pdf(path, elements=True, ocr=engine)
    assert doc.meta["routes"][0]["reason"].startswith("garbled text layer")
    assert all("�" not in e.text and "" not in e.text for e in doc.elements)
    assert [s.title for s in doc.walk()] == ["Readable Title"]


def test_region_ocr_fills_figure_text_only(tmp_path):
    doc = pymupdf.open()
    page = _text_page(doc, "Overview", lines=4)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 30), False)
    pix.set_rect(pix.irect, (30, 120, 200))
    page.insert_image(pymupdf.Rect(100, 300, 500, 600), stream=pix.tobytes("png"))
    for k in range(3):
        page.insert_text((72, 640 + 15 * k), f"{BODY} After {k}.", fontsize=11)
    path = _save(doc, tmp_path / "region.pdf")
    engine = FakeOcr(regions={1: "Chart label 42"})

    loaded = load_pdf(path, elements=True, ocr=engine)
    (route,) = loaded.meta["routes"]
    assert route["scope"] == "region" and route["task"] == "ocr" and route["engine"] == "fake"
    (call,) = engine.calls
    assert call.x0 > 0 and call.y0 > 0  # only the image region was rendered
    (figure,) = [e for e in loaded.elements if e.kind == "figure"]
    assert figure.figure.text == "Chart label 42" and figure.text == ""  # no text-layer labels
    assert figure.figure.source == "ocr_region" and figure.source == "ocr"
    assert figure.confidence == pytest.approx(0.8)
    others = [e for e in loaded.elements if e is not figure]
    assert all("Chart label" not in e.text for e in others)
    assert all("Chart label" not in s.text for s in loaded.walk())

    plain = load_pdf(path, elements=True)  # no engine: logged, but no document warning
    (route,) = plain.meta["routes"]
    assert route["engine"] == "none" and route["reason"].startswith("needs_ocr: region OCR")
    assert plain.meta["warnings"] == []
    (figure,) = [e for e in plain.elements if e.kind == "figure"]
    assert figure.text == "" and figure.figure.source == "image"


def test_missing_ocr_package_is_an_error_not_a_page_failure():
    from docrooter.errors import MissingExtraError

    class Missing(FakeOcr):
        def recognize(self, image, *, width_px, height_px, page):
            raise MissingExtraError("rapidocr is required: pip install 'docrooter[ocr]'")

    with pytest.raises(MissingExtraError):
        load_pdf(GOLDEN / "scanned.pdf", elements=True, ocr=Missing())


def test_ocr_engine_by_name_is_created_lazily_and_closed(monkeypatch):
    made = []

    def factory(*, lang="en"):
        made.append(FakeOcr(pages={1: _ocr_page("Named Engine"), 2: _ocr_page("Second")}))
        return made[-1]

    monkeypatch.setattr(ocr_registry, "_FACTORIES", dict(ocr_registry._FACTORIES))
    ocr_registry.register_ocr_engine("fake", factory)
    load_pdf(GOLDEN / "single_column.pdf", elements=True, ocr="fake")
    assert made == []  # nothing needed OCR
    doc = load_pdf(GOLDEN / "scanned.pdf", elements=True, ocr="fake")
    assert len(made) == 1 and made[0].closed == 1
    assert [s.title for s in doc.walk()] == ["Named Engine", "Second"]


def test_auto_ocr_without_an_engine_skips(monkeypatch):
    monkeypatch.setattr(ocr_registry, "resolve_auto_ocr", lambda: None)
    doc = load_pdf(GOLDEN / "scanned.pdf", elements=True, ocr="auto")
    assert doc.meta["warnings"] == ["no_text_layer", "ocr_skipped"]
    assert {r["engine"] for r in doc.meta["routes"]} == {"none"}


def test_failing_ocr_engine_keeps_the_text_layer():
    engine = FakeOcr(fail=True)
    doc = load_pdf(GOLDEN / "garbled.pdf", elements=True, ocr=engine)
    assert "ocr_failed" in doc.meta["warnings"]
    assert "OCR failed (RuntimeError)" in doc.meta["routes"][0]["reason"]
    assert "�" in doc.sections["s0"].text


def test_fake_engine_is_an_ocr_engine():
    assert isinstance(FakeOcr(), OcrEngine)


# --- events, arguments, entry points -----------------------------------------------------


def test_element_events_order(tmp_path):
    seen = []
    load_pdf(GOLDEN / "scanned.pdf", elements=True, ocr=FakeOcr(pages={1: _ocr_page("T")}),
             on_event=seen.append)  # fmt: skip
    starts = [e.stage for e in seen if e.status == "start"]
    assert starts == ["extract", "route", "ocr", "layout", "signals", "headings", "elements",
                      "assemble"]  # fmt: skip
    assert seen[-1].stage == "done"
    routes = [e for e in seen if e.stage == "route" and e.status == "progress"]
    assert [e.data["target"] for e in routes] == [1, 2] and routes[0].data["task"] == "ocr"
    (ocr_start,) = [e for e in seen if e.stage == "ocr" and e.status == "start"]
    assert ocr_start.data == {"engine": "fake", "pages": 2, "regions": 0}
    (elements_end,) = [e for e in seen if e.stage == "elements" and e.status == "end"]
    assert elements_end.data["elements"] == sum(elements_end.data["kinds"].values())
    ends = [e.stage for e in seen if e.status == "end" and e.stage != "done"]
    assert ends == starts  # every stage closed, in order


def test_meta_timings_and_max_pages(tmp_path):
    doc = load_pdf(GOLDEN / "single_column.pdf", elements=True, max_pages=2)
    assert set(doc.meta["timings"]) == {"extract", "route", "ocr", "layout", "signals",
                                        "headings", "elements", "assemble"}  # fmt: skip
    assert doc.meta["layout"] == "heuristic" and doc.meta["layout_regions"] == 0
    assert doc.meta["n_pages"] == 3 and doc.meta["warnings"] == ["truncated_pages"]
    assert len(doc.meta["routes"]) == 2
    assert max(e.boxes[0].page for e in doc.elements) == 2


def test_pages_reads_a_page_subset(tmp_path):
    doc = pymupdf.open()
    for k in range(1, 7):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 100), f"Part {k}", fontsize=16, fontname="hebo")
        page.insert_text((72, 130), f"Body text of part {k} in plain words.", fontsize=11)
    doc.set_toc([[1, f"Part {k}", k] for k in range(1, 7)])
    path = _save(doc, tmp_path / "parts.pdf")
    got = load_pdf(path, elements=True, pages=[5, 1, 3])
    assert sorted({e.boxes[0].page for e in got.elements}) == [1, 3, 5]
    assert [s.title for s in got.walk() if s.level > 0] == ["Part 1", "Part 3", "Part 5"]
    assert got.meta["n_pages"] == 6 and got.meta["warnings"] == ["truncated_pages"]
    assert len(got.meta["routes"]) == 3 and got.meta["bookmarks"] == 3  # bookmarks of pages read
    capped = load_pdf(path, elements=True, pages=[2, 4, 6], max_pages=2)
    assert sorted({e.boxes[0].page for e in capped.elements}) == [2, 4]
    whole = load_pdf(path, elements=True, pages=range(1, 7))
    assert whole.meta["warnings"] == [] and whole.meta["n_pages"] == 6
    assert whole.model_dump(exclude={"meta"}) == load_pdf(path, elements=True).model_dump(
        exclude={"meta"}
    )
    with pytest.raises(TypeError, match="pages: only valid with elements=True"):
        load_pdf(path, pages=[1])
    with pytest.raises(ValueError, match="pages"):
        load_pdf(path, elements=True, pages=[0])


def test_arguments_are_checked():
    path = GOLDEN / "single_column.pdf"
    with pytest.raises(TypeError, match="only valid with elements=True"):
        load_pdf(path, ocr="auto")
    with pytest.raises(TypeError):
        load_pdf(path, ocr_dpi=300)
    with pytest.raises(DocrooterError, match="no element mode"):
        load_pdf(path, elements=True, engine="both")
    with pytest.raises(DocrooterError, match="unknown layout engine"):
        load_pdf(path, elements=True, layout="no-such-engine")
    with pytest.raises(TypeError, match="only valid with elements=True"):
        load_pdf(path, layout="heron")
    with pytest.raises(TypeError, match="LayoutEngine"):
        load_pdf(path, elements=True, layout=object())
    with pytest.raises(ValueError, match="ocr_dpi"):
        load_pdf(path, elements=True, ocr_dpi=0)


def test_load_passes_element_options_for_pdfs():
    engine = FakeOcr(pages={1: _ocr_page("Through Load"), 2: _ocr_page("Page Two")})
    doc = load(GOLDEN / "scanned.pdf", elements=True, ocr=engine, ocr_dpi=100)
    assert [s.title for s in doc.walk()] == ["Through Load", "Page Two"]
    assert doc.meta["element_mode"] is True


@pytest.mark.parametrize("engine", ["pymupdf", "pdfplumber"])
def test_pdfplumber_and_pymupdf_both_build_sections(engine):
    if engine == "pdfplumber":
        pytest.importorskip("pdfplumber")
    doc = load_pdf(GOLDEN / "bookmarks.pdf", elements=True, engine=engine)
    assert doc.meta["engine"] == engine
    assert [(e.title, e.level) for e in doc.toc()] == [
        ("Alpha", 1), ("Alpha Detail", 2), ("Beta", 2), ("Beta Detail", 3),
    ]  # fmt: skip
    assert "level_from_bookmark" in doc.section("Beta").flags


# --- rotated pages keep their text (E2 carry) ----------------------------------------------


def _numbered_heading(page, number, title, y):
    """A section number and its title as two lines of one block on one row (the title sits
    1.5 pt lower, so pymupdf does not merge them itself); ``extract_lines`` joins them."""
    page.insert_text((72, y), number, fontsize=14, fontname="hebo")
    page.insert_text((100, y + 1.5), title, fontsize=14, fontname="hebo")


def _rotated_pdf(tmp_path, name, rotations):
    """For each rotation: a page with text drawn before ``set_rotation``, one with text drawn
    after it, and one with both; each has a numbered heading split into same-row lines."""
    doc = pymupdf.open()
    for rot in rotations:
        page = doc.new_page(width=612, height=792)
        _numbered_heading(page, "3.", "Introduction", 60)
        for k in range(5):
            text = f"Before r{rot} line {k} of body text"
            page.insert_text((72, 100 + 20 * k), text, fontsize=11)
        page.set_rotation(rot)
        page = doc.new_page(width=612, height=792)
        page.set_rotation(rot)
        _numbered_heading(page, "4.", "Methods", 60)
        for k in range(5):
            text = f"After r{rot} line {k} of body text"
            page.insert_text((72, 100 + 20 * k), text, fontsize=11)
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 100), f"Section r{rot}", fontsize=18, fontname="hebo")
        page.insert_text((72, 140), f"Mixed before r{rot} and more words", fontsize=11)
        page.set_rotation(rot)
        page.insert_text((72, 400), f"Mixed after r{rot} and more words", fontsize=11)
        _numbered_heading(page, "5.", "Results", 450)
    return _save(doc, tmp_path / name)


@pytest.mark.parametrize("engine", ["pymupdf", "pdfplumber"])
@pytest.mark.parametrize(
    "rotations",
    [(0, 90, 180, 270), (90, 90), (270, 0, 180), (180,)],
    ids=["all", "r90", "r270", "r180"],
)
def test_rotated_pages_keep_every_extract_lines_text(tmp_path, engine, rotations):
    if engine == "pdfplumber":
        pytest.importorskip("pdfplumber")
    path = _rotated_pdf(tmp_path, "rot.pdf", rotations)
    expected, _ = extract_lines(path, engine=engine)
    pages = extract_page_objects(path, engine=engine)
    assert len(pages) == 3 * len(rotations)
    for objs in pages:
        want = Counter(ln.text for ln in expected if ln.page == objs.page)
        assert want, objs.page
        got = [ln.text for ln in objs.lines]
        assert Counter(got) == want, (objs.page, objs.rotation)
        assert sorted(got) == sorted(want.elements()), (objs.page, objs.rotation)
        if engine == "pymupdf":  # pdfplumber's own lines on rotated pages are letter soup,
            # mirrored at 180 degrees; element mode still equals them exactly (above)
            heading = {1: "3. Introduction", 2: "4. Methods", 0: "5. Results"}[objs.page % 3]
            assert heading in got and want[heading] == 1, (objs.page, objs.rotation, got)
        for ln in objs.lines:  # in display space, on the page
            assert 0 <= ln.x0 <= ln.x1 <= objs.width + 0.5
            assert 0 <= ln.y0 <= ln.y1 <= objs.height + 0.5
    if engine == "pymupdf":
        vertical = {p.page for p in pages if "vertical_text" in p.flags}
        assert vertical == {p.page for p in pages if p.rotation in (90, 270)}

    doc = load_pdf(path, elements=True, engine=engine)
    # a list or footnote marker is kept apart from the element's text
    everything = " ".join(f"{e.marker} {e.text}" if e.marker else e.text for e in doc.elements)
    for line in expected:
        assert line.text in everything


def test_upright_text_on_a_rotated_page_is_kept_too(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.set_rotation(90)
    page.insert_text(
        pymupdf.Point(100, 150) * page.derotation_matrix, "Upright on screen", fontsize=14,
        rotate=90,
    )  # fmt: skip
    page.insert_text((72, 100), "Sideways on screen", fontsize=11)
    path = _save(doc, tmp_path / "upright.pdf")
    (objs,) = extract_page_objects(path, engine="pymupdf")
    texts = {ln.text: ln for ln in objs.lines}
    assert set(texts) == {"Upright on screen", "Sideways on screen"}
    upright, sideways = texts["Upright on screen"], texts["Sideways on screen"]
    assert upright.x1 - upright.x0 > upright.y1 - upright.y0  # reads across
    assert sideways.y1 - sideways.y0 > sideways.x1 - sideways.x0  # runs down the screen
    assert [ln.text for ln in extract_lines(path, engine="pymupdf")[0]] == ["Sideways on screen"]


def test_table_cells_that_lose_words_keep_the_lines_text(tmp_path):
    doc = pymupdf.open()
    page = _text_page(doc, "Terms", lines=3)
    x0, y0, cw, rh = 72, 200, 150, 20
    for r in range(4):
        page.draw_line((x0, y0 + r * rh), (x0 + 2 * cw, y0 + r * rh))
    for c in range(3):
        page.draw_line((x0 + c * cw, y0), (x0 + c * cw, y0 + 3 * rh))
    for r in range(3):
        page.insert_text((x0 + 4, y0 + r * rh + 14), f"term {r}", fontsize=9)
    page.insert_text((x0 + cw + 4, y0 + 14), "short value", fontsize=9)
    # runs past the table's right edge: the engine's cell holds only the part inside
    page.insert_text((x0 + cw + 4, y0 + rh + 14), "a value that overflows the ruled cell edge",
                     fontsize=9)  # fmt: skip
    page.insert_text((x0 + cw + 4, y0 + 2 * rh + 14), "last value", fontsize=9)
    loaded = load_pdf(_save(doc, tmp_path / "overflow.pdf"), elements=True)
    (table,) = [e for e in loaded.elements if e.kind == "table"]
    assert "cell_text_incomplete" in table.flags
    assert "a value that overflows the ruled cell edge" in table.text
    assert table.table.rows and "edge" not in str(table.table.rows)  # cells as the engine read
    assert "overflows the ruled cell edge" in loaded.section("Terms").text


def test_region_ocr_keeps_the_figures_text_layer_labels(tmp_path):
    doc = pymupdf.open()
    page = _text_page(doc, "Overview", lines=3)
    page.draw_rect(pymupdf.Rect(72, 250, 520, 560), color=(0, 0, 0), fill=(0.9, 0.9, 0.95))
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 30), False)
    pix.set_rect(pix.irect, (30, 120, 200))
    for rect in ((90, 270, 290, 420), (310, 270, 500, 420)):  # each >= 5% of the page
        page.insert_image(pymupdf.Rect(*rect), stream=pix.tobytes("png"), keep_proportion=False)
    page.insert_text((100, 500), "Q1", fontsize=8)  # text-layer labels of the chart
    page.insert_text((450, 540), "Q4", fontsize=8)
    path = _save(doc, tmp_path / "labels.pdf")
    engine = FakeOcr(regions={1: "Scanned axis"}, region_conf=(0.6, 0.9))

    loaded = load_pdf(path, elements=True, ocr=engine)
    assert len(engine.calls) == 2
    (figure,) = [e for e in loaded.elements if e.kind == "figure"]
    assert figure.text == "Q1\nQ4"  # the text-layer labels stay the element's text
    assert figure.figure.text == "Q1\nQ4\nScanned axis\nScanned axis"
    assert figure.figure.source == "ocr_region" and figure.source == "ocr"
    assert figure.confidence == pytest.approx(0.75)  # the mean over both regions
    overview = loaded.section("Overview").text
    assert "Q1" in overview and "Q4" in overview  # labels are never lost from the section
    assert "Scanned axis" not in overview  # region OCR is figure text only


def test_unknown_ocr_engine_name_fails_before_any_page_work(monkeypatch):
    import docrooter.pdf.element_mode as element_mode

    def boom(*args, **kwargs):
        raise AssertionError("pages were extracted")

    monkeypatch.setattr(element_mode, "extract_page_objects", boom)
    with pytest.raises(DocrooterError, match="unknown OCR engine 'no-such-engine'.*tesseract"):
        load_pdf(GOLDEN / "single_column.pdf", elements=True, ocr="no-such-engine")


def test_table_stays_after_the_title_on_a_single_column_page(monkeypatch):
    """Bare numbers inside a table are excluded by the signals but are not furniture; reading
    order must not move them (and with them the table) to the top of the page."""
    import docrooter.pdf.element_mode as element_mode

    path = GOLDEN / "tables.pdf"
    kinds = [e.kind for e in load_pdf(path, elements=True).elements]
    assert kinds == ["title", "paragraph", "caption", "table", "heading", "paragraph"]
    doc = load_pdf(path, elements=True)
    table = next(e for e in doc.elements if e.kind == "table")
    assert table.section_id == "s1" and "s0" not in doc.sections

    # the same order with every excluded line treated as furniture puts the table first
    monkeypatch.setattr(element_mode, "furniture_lines", lambda lines, excluded: set(excluded))
    broken = [e.kind for e in load_pdf(path, elements=True).elements]
    assert broken.index("table") < broken.index("title")


def test_furniture_lines_releases_interior_bare_numbers():
    from docrooter.pdf.elements import furniture_lines
    from docrooter.pdf.lines import Line

    def line(text, y):
        return Line(page=1, text=text, size=10, bold=False, font="helv", x0=72, y0=y,
                    x1=300, y1=y + 10, page_height=792, page_width=612, gap_above=2,
                    gap_below=2, block=0)  # fmt: skip

    lines = [line("ACME Report", 30), line("Body text above", 100), line("2024", 200),
             line("Body text below", 300), line("7", 760)]  # fmt: skip
    assert furniture_lines(lines, {0, 2, 4}) == {0, 4}


def _scan_with_text_layer(tmp_path):
    """A scanned page (full-page image) that also carries a few text-layer lines: a heading
    the OCR engine reads again, a caption and a page number. 31 characters over the page is
    well under 2 per square inch."""
    doc = pymupdf.open()
    page = _scan_page(doc)
    page.insert_text((72, 102), "Plant Survey", fontsize=22, fontname="hebo")
    page.insert_text((72, 700), "Figure 3 plant layout", fontsize=11, fontname="helv")
    page.insert_text((300, 770), "12", fontsize=9, fontname="helv")
    return _save(doc, tmp_path / "scan_text.pdf")


def _covered(box, lines):
    """Share of ``box`` under the union of the lines' boxes, on a 0.5 pt grid (independent of
    the implementation's sweep)."""
    step = 0.5
    nx, ny = int((box.x1 - box.x0) / step), int((box.y1 - box.y0) / step)
    hit = 0
    for i in range(nx):
        for j in range(ny):
            x, y = box.x0 + (i + 0.5) * step, box.y0 + (j + 0.5) * step
            hit += any(ln.x0 <= x <= ln.x1 and ln.y0 <= y <= ln.y1 for ln in lines)
    return hit / (nx * ny)


def test_scanned_rule_with_text_layer_merges(tmp_path):
    path = _scan_with_text_layer(tmp_path)
    (objs,) = extract_page_objects(path, engine="pymupdf")
    text_layer = [ln.text for ln in objs.lines]
    assert text_layer == ["Plant Survey", "Figure 3 plant layout", "12"]
    engine = FakeOcr(pages={1: _ocr_page("Plant Survey")})  # heading again, then 4 body lines
    read = engine.recognize(b"\x89PNG", width_px=1, height_px=1, page=Box(page=1, x0=0, y0=0,
                            x1=objs.width, y1=objs.height))  # fmt: skip
    engine.calls.clear()
    covered = [o.text for o in read if _covered(o.box, objs.lines) >= 0.5]
    assert covered == ["Plant Survey"]  # the fixture really has one duplicate line

    doc = load_pdf(path, elements=True, ocr=engine)
    (route,) = doc.meta["routes"]
    assert (route["scope"], route["task"], route["engine"]) == ("page", "text+ocr", "fake")
    assert route["reason"].startswith("scanned page with a text layer")
    assert route["metrics"]["ocr_lines"] == 5.0 and route["metrics"]["ocr_added"] == 4.0
    assert len(engine.calls) == 1
    texts = [e.text for e in doc.elements]
    joined = "\n".join(texts)
    for line in text_layer:  # every text-layer line is kept
        assert line in joined
    assert joined.count("Plant Survey") == 1  # the overlapping OCR line was not added
    for o in read[1:]:  # the OCR lines that overlap nothing were added
        assert o.text in joined
    by_text = {e.text: e for e in doc.elements}
    assert by_text["Plant Survey"].source == "text"
    # the two sources are merged top to bottom: the OCR body (y 129..) is read before the
    # text-layer caption at y 688 and the page number
    assert [(e.kind, e.source, e.text[:17]) for e in doc.elements] == [
        ("title", "text", "Plant Survey"),
        ("paragraph", "ocr", "Scanned body text"),
        ("paragraph", "text", "Figure 3 plant la"),
        ("page_number", "text", "12"),
    ]
    ocr = [e for e in doc.elements if e.source == "ocr"]
    assert ocr and all("Scanned body text" in e.text for e in ocr)
    assert all(e.confidence == pytest.approx(0.9) for e in ocr)
    assert doc.meta["warnings"] == []
    assert doc.meta["needs_ocr"] == {"pages": 0, "merge_pages": 1, "regions": 0}

    # Without an engine the page keeps its text layer; the route log says what it needed and
    # the document is flagged ocr_skipped (spec §7.4: the page needed OCR).
    plain = load_pdf(path, elements=True)
    (route,) = plain.meta["routes"]
    assert route["engine"] == "none" and route["task"] == "text+ocr"
    assert route["reason"].startswith("needs_ocr: scanned page with a text layer")
    assert plain.meta["warnings"] == ["ocr_skipped"]
    assert plain.meta["needs_ocr"] == {"pages": 0, "merge_pages": 1, "regions": 0}
    assert [e.text for e in plain.elements if e.text] == [
        e.text for e in doc.elements if e.source == "text" and e.text
    ]


def test_needs_ocr_counts_separate_pages_from_regions(tmp_path):
    # Review minor (E5b): page-level and region-level OCR needs are counted apart
    scanned = load_pdf(GOLDEN / "scanned.pdf", elements=True)
    assert scanned.meta["needs_ocr"] == {"pages": 2, "merge_pages": 0, "regions": 0}
    doc = pymupdf.open()
    page = _text_page(doc, "Overview", lines=4)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 30), False)
    pix.set_rect(pix.irect, (30, 120, 200))
    page.insert_image(pymupdf.Rect(100, 300, 500, 600), stream=pix.tobytes("png"))
    photo = load_pdf(_save(doc, tmp_path / "photo.pdf"), elements=True)
    assert photo.meta["needs_ocr"] == {"pages": 0, "merge_pages": 0, "regions": 1}
    assert photo.meta["warnings"] == []
    plain = load_pdf(GOLDEN / "single_column.pdf", elements=True)
    assert plain.meta["needs_ocr"] == {"pages": 0, "merge_pages": 0, "regions": 0}


def _merge_page(*texts_and_boxes):
    """A page whose text layer is the given (text, x0, y0, x1, y1) lines."""
    from docrooter.pdf.lines import Line
    from docrooter.pdf.objects import PageObjects

    lines = [
        Line(page=1, text=t, size=11.0, bold=False, font="helv", x0=x0, y0=y0, x1=x1, y1=y1,
             page_height=792.0, page_width=612.0, gap_above=4.0, gap_below=4.0, block=k)
        for k, (t, x0, y0, x1, y1) in enumerate(texts_and_boxes)
    ]  # fmt: skip
    return PageObjects(page=1, width=612.0, height=792.0, rotation=0, lines=lines, images=[],
                       drawings=[], tables=[], char_count=sum(len(t) for t, *_ in texts_and_boxes),
                       invisible_chars=0, garbled_chars=0, n_drawing_paths=0)  # fmt: skip


def _ocr(text, x0, y0, x1, y1):
    return OcrLine(text=text, box=Box(page=1, x0=x0, y0=y0, x1=x1, y1=y1), confidence=0.9)


def test_merged_page_lines_are_sorted_top_to_bottom_with_gaps_recomputed():
    from docrooter.pdf.element_mode import _merge_page_lines

    page = _merge_page(("Title", 72, 80, 300, 100), ("Caption low on the page", 72, 690, 300, 702))
    extra, _ = _to_lines_for_test(
        page,
        [
            _ocr("Body read by OCR", 72, 130, 400, 144),
            _ocr("More body read by OCR", 72, 148, 400, 162),
        ],
    )
    lines, source, conf = _merge_page_lines(page.lines, extra, [0.8, 0.7])
    assert [ln.text for ln in lines] == ["Title", "Body read by OCR", "More body read by OCR",
                                         "Caption low on the page"]  # fmt: skip
    assert source == ["text", "ocr", "ocr", "text"] and conf == [None, 0.8, 0.7, None]
    assert [ln.gap_above for ln in lines] == [math.inf, 30.0, 4.0, 528.0]
    assert [ln.gap_below for ln in lines] == [30.0, 4.0, 528.0, math.inf]


def _to_lines_for_test(page, found):
    from docrooter.pdf.element_mode import _new_lines

    return _new_lines(found, page)


def test_merge_drops_an_ocr_row_the_text_layer_splits_into_short_lines():
    from docrooter.pdf.element_mode import _new_lines

    # the text layer has a table row as four cells; OCR reads the row as one line. Each cell
    # alone overlaps the OCR line little (IoU < 0.3), but together they cover all of it.
    row = _ocr("Revenue 1,204 1,317 +9%", 72, 99, 480, 113)
    cells = _merge_page(("Revenue", 72, 99, 180, 113), ("1,204", 180, 99, 290, 113),
                        ("1,317", 290, 99, 400, 113), ("+9%", 400, 99, 480, 113))  # fmt: skip
    assert _covered(row.box, cells.lines) == pytest.approx(1.0, abs=0.01)
    extra, _ = _new_lines([row], cells)
    assert extra == []  # a row tiled by short text-layer lines is a duplicate
    new = _ocr("Handwritten note in the margin", 72, 300, 300, 314)
    extra, conf = _new_lines([row, new], cells)
    assert [ln.text for ln in extra] == ["Handwritten note in the margin"] and conf == [0.9]
    assert extra[0].block == 4  # numbered after the text layer's blocks


def test_merge_keeps_an_ocr_line_that_runs_past_a_short_text_line():
    from docrooter.pdf.element_mode import _new_lines

    # the text layer has only the first words of a title; OCR reads the whole title. The text
    # line is mostly inside the OCR line (IoU ~0.34 under the old rule) but covers well under
    # half of it, so the OCR line and its extra words are kept.
    page = _merge_page(("Plant Survey", 72, 80, 202, 108))
    title = _ocr("Plant Survey 2024 Annual Report", 72, 80, 452, 108)
    assert _covered(title.box, page.lines) < 0.5
    extra, _ = _new_lines([title], page)
    assert [ln.text for ln in extra] == ["Plant Survey 2024 Annual Report"]
    # the same line, read only as far as the text layer goes, is a duplicate
    extra, _ = _new_lines([_ocr("Plant Survey", 72, 81, 200, 107)], page)
    assert extra == []


HEADINGS = {
    "short": "1. Business",
    "medium": "1. Business and operating results of the group",
    "wide": "1. Business and operating results of the group during the financial year",
}
COLUMN_WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima".split()


def _short_columns_pdf(path, *, leading, left, right, heading):
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 80), HEADINGS[heading], fontsize=16, fontname="hebo")
    for tag, x, n in (("LEFT", 72, left), ("RIGHT", 330, right)):
        for k in range(n):
            words = " ".join(COLUMN_WORDS[(3 * k + j) % len(COLUMN_WORDS)] for j in range(5))
            page.insert_text((x, 110 + k * leading), f"{tag} {words}", fontsize=10)
    return _save(doc, path)


@pytest.mark.parametrize("heading", HEADINGS)
@pytest.mark.parametrize("columns", [(5, 4), (6, 5), (4, 3), (7, 5), (5, 6), (6, 6)])
@pytest.mark.parametrize("leading", [12, 13, 14, 15])
def test_short_columns_under_a_heading_read_left_then_right(tmp_path, leading, columns, heading):
    """Unequal short columns at tight leading used to read the right column first (the heading
    stretched the left side, the sides shared under 60% of the band, and the extractor's
    shorter-column-first order was kept)."""
    path = _short_columns_pdf(
        tmp_path / "cols.pdf", leading=leading, left=columns[0], right=columns[1], heading=heading
    )
    doc = load_pdf(path, elements=True)
    body = " ".join(e.text for e in doc.elements if e.kind not in ("title", "heading"))
    assert 0 <= body.find("LEFT") < body.find("RIGHT")
    assert body.count("LEFT") == columns[0] and body.count("RIGHT") == columns[1]


GRID_WORDS = (
    "our sensors measure pressure temperature and flow in factories and power plants demand "
    "grew in every region during the year and we added two production lines to keep up orders "
    "from new customers rose by a third"
).split()
GRID_HEADINGS = {
    "short": "1. Business",
    "medium": "1. Business and operating results",  # ends inside the gutter
    "wide": "1. Business and operating results of the group in the year",
}


def _grid_pdf(path, *, title, heading, left, right, leading, gap, lower):
    """The page of the example: an optional title, a heading, two 10 pt columns (``left`` and
    ``right`` lines) and optionally the next section's heading and a line under them."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    y = 80
    if title:
        page.insert_text((72, y), "Annual Report 2025", fontsize=24, fontname="hebo")
        y += 45
    page.insert_text((72, y), GRID_HEADINGS[heading], fontsize=16, fontname="hebo")
    top = y + gap
    for tag, x, n in (("Left", 72, left), ("Right", 330, right)):
        for k in range(n):
            words = " ".join(GRID_WORDS[(5 * k + j) % len(GRID_WORDS)] for j in range(6))
            page.insert_text(
                (x, top + k * leading), (f"{tag} " if k == 0 else "") + words, fontsize=10
            )
    if lower:
        below = top + max(left, right) * leading + 30
        page.insert_text((72, below), "2. Financial results", fontsize=16, fontname="hebo")
        page.insert_text(
            (72, below + 25), "Revenue rose in every segment during the year.", fontsize=10
        )
    return _save(doc, path)


def test_columns_read_left_then_right_over_a_grid_of_layouts(tmp_path):
    """960 layouts: with and without a title, three heading widths (one ends inside the gutter),
    column lengths from 5/4 to 12/3, leading from 11 to 15 pt, two heading-to-column spacings,
    with and without a section under the columns. At most 2% may read the right column first;
    at the time of writing none do (the previous release failed 300 of them)."""
    grid = itertools.product(
        [True, False],
        GRID_HEADINGS,
        [(5, 4), (6, 5), (4, 3), (6, 4), (5, 6), (7, 5), (6, 6), (12, 3)],
        [11, 12, 13, 14, 15],
        [18, 30],
        [False, True],
    )
    failures = []
    for n, (title, heading, (left, right), leading, gap, lower) in enumerate(grid):
        path = _grid_pdf(
            tmp_path / f"grid{n % 4}.pdf",
            title=title, heading=heading, left=left, right=right, leading=leading, gap=gap,
            lower=lower,
        )  # fmt: skip
        doc = load_pdf(path, elements=True)
        body = " ".join(e.text for e in doc.elements if e.kind not in ("title", "heading"))
        if not 0 <= body.find("Left") < body.find("Right"):
            failures.append((title, heading, left, right, leading, gap, lower))
    assert n + 1 == 960
    assert len(failures) <= 19, failures[:5]  # 2% of 960
