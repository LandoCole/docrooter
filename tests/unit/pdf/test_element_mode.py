"""Element mode in ``load_pdf(elements=True)``: routing, OCR with a fake engine, XY-cut order,
elements and the sections built from them."""

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

    def __init__(self, pages=None, regions=None, fail=False):
        self.pages = pages or {}
        self.regions = regions or {}
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
            return [OcrLine(text=text, box=box, confidence=0.8)] if text else []
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
    assert figure.text == figure.figure.text == "Chart label 42"
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
    assert starts == ["extract", "route", "ocr", "signals", "headings", "elements", "assemble"]
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
    assert set(doc.meta["timings"]) == {"extract", "route", "ocr", "signals", "headings",
                                        "elements", "assemble"}  # fmt: skip
    assert doc.meta["n_pages"] == 3 and doc.meta["warnings"] == ["truncated_pages"]
    assert len(doc.meta["routes"]) == 2
    assert max(e.boxes[0].page for e in doc.elements) == 2


def test_arguments_are_checked():
    path = GOLDEN / "single_column.pdf"
    with pytest.raises(TypeError, match="only valid with elements=True"):
        load_pdf(path, ocr="auto")
    with pytest.raises(TypeError):
        load_pdf(path, ocr_dpi=300)
    with pytest.raises(DocrooterError, match="no element mode"):
        load_pdf(path, elements=True, engine="both")
    with pytest.raises(DocrooterError, match="unknown layout engine"):
        load_pdf(path, elements=True, layout="heron")
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


def _rotated_pdf(tmp_path, name, rotations):
    """For each rotation: a page with text drawn before ``set_rotation``, one with text drawn
    after it, and one with both."""
    doc = pymupdf.open()
    for rot in rotations:
        page = doc.new_page(width=612, height=792)
        for k in range(5):
            text = f"Before r{rot} line {k} of body text"
            page.insert_text((72, 100 + 20 * k), text, fontsize=11)
        page.set_rotation(rot)
        page = doc.new_page(width=612, height=792)
        page.set_rotation(rot)
        for k in range(5):
            text = f"After r{rot} line {k} of body text"
            page.insert_text((72, 100 + 20 * k), text, fontsize=11)
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 100), f"Section r{rot}", fontsize=18, fontname="hebo")
        page.insert_text((72, 140), f"Mixed before r{rot} and more words", fontsize=11)
        page.set_rotation(rot)
        page.insert_text((72, 400), f"Mixed after r{rot} and more words", fontsize=11)
    return _save(doc, tmp_path / name)


@pytest.mark.parametrize("engine", ["pymupdf", "pdfplumber"])
@pytest.mark.parametrize(
    "rotations", [(0, 90, 180, 270), (90, 90), (270, 0, 180)], ids=["all", "r90", "r270"]
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
        assert Counter(ln.text for ln in objs.lines) == want, (objs.page, objs.rotation)
        for ln in objs.lines:  # in display space, on the page
            assert 0 <= ln.x0 <= ln.x1 <= objs.width + 0.5
            assert 0 <= ln.y0 <= ln.y1 <= objs.height + 0.5
    if engine == "pymupdf":
        vertical = {p.page for p in pages if "vertical_text" in p.flags}
        assert vertical == {p.page for p in pages if p.rotation in (90, 270)}

    doc = load_pdf(path, elements=True, engine=engine)
    everything = " ".join(e.text for e in doc.elements)
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
