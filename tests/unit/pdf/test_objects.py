import time

import pytest

from docrooter.errors import DocrooterError
from docrooter.pdf import objects
from docrooter.pdf.lines import extract_lines
from docrooter.pdf.objects import MAX_DRAWING_PATHS, extract_page_objects

pymupdf = pytest.importorskip("pymupdf")


@pytest.fixture(params=["pymupdf", "pdfplumber"])
def engine(request):
    if request.param == "pdfplumber":
        pytest.importorskip("pdfplumber")
    return request.param


def _save(doc, tmp_path, name="t.pdf"):
    path = tmp_path / name
    doc.save(path)
    doc.close()
    return path


def _png(width=40, height=30):
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), False)
    pix.set_rect(pix.irect, (200, 30, 30))
    return pix.tobytes("png")


def _text_pdf(tmp_path, rotation=0):
    doc = pymupdf.open()
    for n in (1, 2):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 100), f"Chapter {n} Overview", fontsize=20, fontname="hebo")
        y = 140
        for k in range(3):
            page.insert_text((72, y), f"Body sentence {k} of page {n} stays small.", fontsize=11)
            y += 16
        page.insert_text((72, y + 40), "Isolated paragraph after a gap.", fontsize=11)
        if rotation:
            page.set_rotation(rotation)
    return _save(doc, tmp_path)


def test_objects_lines_match_extract_lines(engine, tmp_path):
    path = _text_pdf(tmp_path)
    expected, n_pages = extract_lines(path, engine=engine)
    pages = extract_page_objects(path, engine=engine)
    assert n_pages == len(pages) == 2
    assert [p.page for p in pages] == [1, 2]
    got = [ln for p in pages for ln in p.lines]
    assert len(got) == len(expected) == 10
    assert [ln.model_dump() for ln in got] == [ln.model_dump() for ln in expected]
    assert (pages[0].width, pages[0].height, pages[0].rotation) == (612, 792, 0)


# Lines of the generated PDF above as the v0.3 ``extract_lines`` (feat/v0.3, 28fbf9d) wrote
# them: text, size, bold, block, then the box rounded to 1 pt.
_FROZEN = {
    "pymupdf": [
        ("Chapter 1 Overview", 20.0, True, 0, (72, 79, 259, 106)),
        ("Body sentence 0 of page 1 stays small.", 11.0, False, 1, (72, 128, 263, 143)),
        ("Body sentence 1 of page 1 stays small.", 11.0, False, 1, (72, 144, 263, 159)),
        ("Body sentence 2 of page 1 stays small.", 11.0, False, 1, (72, 160, 263, 175)),
        ("Isolated paragraph after a gap.", 11.0, False, 2, (72, 216, 222, 231)),
    ],
}


def test_extract_lines_output_is_unchanged(tmp_path):
    """``extract_lines`` is frozen; exposing a per-page iterator must not change it."""
    path = _text_pdf(tmp_path)
    lines, _ = extract_lines(path, engine="pymupdf")
    got = [
        (ln.text, ln.size, ln.bold, ln.block, tuple(round(v) for v in (ln.x0, ln.y0, ln.x1, ln.y1)))
        for ln in lines
        if ln.page == 1
    ]
    assert got == _FROZEN["pymupdf"]


def test_image_placement_box(engine, tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(pymupdf.Rect(100, 200, 260, 320), stream=_png())
    path = _save(doc, tmp_path)
    (p,) = extract_page_objects(path, engine=engine)
    assert len(p.images) == 1
    box = p.images[0]
    assert box.page == 1
    assert (box.x0, box.y0, box.x1, box.y1) == pytest.approx((100, 200, 260, 320), abs=1)
    assert p.drawings == [] and p.tables == []


def _bar_chart(page):
    shape = page.new_shape()
    for i in range(30):
        x = 100 + i * 10
        top = 400 - 20 - (i * 7) % 90  # varied heights, 2 pt between bars
        shape.draw_rect(pymupdf.Rect(x, top, x + 8, 400))
        shape.finish(color=(0, 0, 0), fill=(0.2, 0.4, 0.8), width=0.5)
    for points in (
        [(98, 400), (98, 280), (95, 280)],  # y axis with a tick
        [(98, 400), (400, 400)],  # x axis
        [(98, 340), (400, 340)],  # grid lines
        [(98, 310), (400, 310)],
        [(98, 370), (400, 370)],
    ):
        shape.draw_polyline(points)
        shape.finish(color=(0.3, 0.3, 0.3), width=0.5, closePath=False)
    shape.commit()


def test_vector_chart_is_one_drawing_cluster(engine, tmp_path):
    doc = pymupdf.open()
    _bar_chart(doc.new_page(width=612, height=792))
    path = _save(doc, tmp_path)
    (p,) = extract_page_objects(path, engine=engine)
    assert p.n_drawing_paths >= 35
    assert len(p.drawings) == 1
    box = p.drawings[0]
    assert (box.x0, box.x1) == pytest.approx((95, 398), abs=2)
    assert (box.y0, box.y1) == pytest.approx((280, 400), abs=2)
    assert "drawings_capped" not in p.flags


def test_ruled_table_cells(engine, tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    rows = [["A", "B", "C"], ["1", "2", "3"], ["4", "5", "6"]]
    shape = page.new_shape()
    for r in range(4):  # 3 x 3 cells of 80 x 30 pt, top left at (100, 200)
        shape.draw_line((100, 200 + 30 * r), (340, 200 + 30 * r))
    for c in range(4):
        shape.draw_line((100 + 80 * c, 200), (100 + 80 * c, 290))
    shape.finish(color=(0, 0, 0), width=1)
    shape.commit()
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            page.insert_text((135 + 80 * c, 220 + 30 * r), text, fontsize=12)
    path = _save(doc, tmp_path)
    (p,) = extract_page_objects(path, engine=engine)
    assert len(p.tables) == 1
    box, cells = p.tables[0]
    assert cells == rows
    assert (box.x0, box.y0, box.x1, box.y1) == pytest.approx((100, 200, 340, 290), abs=2)


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_rotated_ruled_table_box_is_in_display_space(engine, tmp_path, rotation):
    """A ruled grid on a /Rotate page: the table box encloses the grid as displayed."""
    W, H = 600.0, 800.0
    grid = (100.0, 200.0, 340.0, 290.0)  # unrotated: 3 x 3 cells of 80 x 30 pt
    doc = pymupdf.open()
    page = doc.new_page(width=W, height=H)
    shape = page.new_shape()
    for r in range(4):
        shape.draw_line((100, 200 + 30 * r), (340, 200 + 30 * r))
    for c in range(4):
        shape.draw_line((100 + 80 * c, 200), (100 + 80 * c, 290))
    shape.finish(color=(0, 0, 0), width=1)
    shape.commit()
    for r in range(3):
        for c in range(3):
            page.insert_text((135 + 80 * c, 220 + 30 * r), "x", fontsize=12)
    page.set_rotation(rotation)
    path = _save(doc, tmp_path)

    def to_display(x, y):
        return {90: (H - y, x), 180: (W - x, H - y), 270: (y, W - x)}[rotation]

    (ax, ay), (bx, by) = to_display(grid[0], grid[1]), to_display(grid[2], grid[3])
    want = (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))
    (p,) = extract_page_objects(path, engine=engine)
    assert len(p.tables) == 1
    box = p.tables[0][0]
    got = (box.x0, box.y0, box.x1, box.y1)
    iw = min(got[2], want[2]) - max(got[0], want[0])
    ih = min(got[3], want[3]) - max(got[1], want[1])
    inter = max(iw, 0) * max(ih, 0)
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])  # noqa: E731
    assert inter / (area(got) + area(want) - inter) >= 0.9, (got, want)


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_rotated_page_boxes_are_in_display_space(engine, tmp_path, rotation):
    """A /Rotate page: every box is where the viewer shows it, not where the content stream
    puts it. The page is 600 x 800 unrotated; content is authored so that it reads upright
    after the rotation."""
    W, H = 600.0, 800.0

    def to_display(x, y):  # unrotated point -> displayed point
        return {90: (H - y, x), 180: (W - x, H - y), 270: (y, W - x)}[rotation]

    def display_box(x0, y0, x1, y1):
        (ax, ay), (bx, by) = to_display(x0, y0), to_display(x1, y1)
        return min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)

    doc = pymupdf.open()
    page = doc.new_page(width=W, height=H)
    page.set_rotation(rotation)
    # Upright text with its baseline origin at display (100, 150).
    page.insert_text(
        pymupdf.Point(100, 150) * page.derotation_matrix, "Rotated heading text",
        fontsize=14, rotate=rotation,
    )  # fmt: skip
    image = (100, 200, 220, 260)  # unrotated rect
    page.insert_image(pymupdf.Rect(*image), stream=_png(), keep_proportion=False)
    chart = (300, 300, 460, 380)
    page.draw_rect(pymupdf.Rect(*chart), color=(0, 0, 0), fill=(0.5, 0.5, 0.9))
    path = _save(doc, tmp_path)

    (p,) = extract_page_objects(path, engine=engine)
    dw, dh = (H, W) if rotation in (90, 270) else (W, H)
    assert (p.width, p.height, p.rotation) == (dw, dh, rotation)

    (ibox,) = p.images
    assert (ibox.x0, ibox.y0, ibox.x1, ibox.y1) == pytest.approx(display_box(*image), abs=1)
    (dbox,) = p.drawings
    assert (dbox.x0, dbox.y0, dbox.x1, dbox.y1) == pytest.approx(display_box(*chart), abs=1)
    (line,) = p.lines
    assert line.text == "Rotated heading text"
    assert line.x0 == pytest.approx(100, abs=1)
    assert 130 < line.y0 < line.y1 < 160  # baseline at 150: ascent above, descent below
    assert (line.page_width, line.page_height) == (dw, dh)
    assert 0 <= line.x0 < line.x1 <= dw and 0 <= line.y0 < line.y1 <= dh


def _heavy_pdf(tmp_path, n_paths):
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "A map with many paths", fontsize=14)
    shape = page.new_shape()
    for i in range(n_paths):
        x, y = 20 + (i * 37) % 570, 80 + (i * 91) % 690
        shape.draw_rect(pymupdf.Rect(x, y, x + 0.8, y + 0.8))
        shape.finish(fill=(0, 0, 0), width=0.1)
    shape.commit()
    return _save(doc, tmp_path)


def test_vector_heavy_page_is_bounded(engine, tmp_path):
    path = _heavy_pdf(tmp_path, 20_000)
    started = time.perf_counter()
    (p,) = extract_page_objects(path, engine=engine)
    elapsed = time.perf_counter() - started
    assert elapsed < 6.0
    assert p.n_drawing_paths > MAX_DRAWING_PATHS
    assert "drawings_capped" in p.flags
    assert 1 <= len(p.drawings) <= 200
    assert [ln.text for ln in p.lines] == ["A map with many paths"]  # still processed


def test_just_under_the_cap_is_clustered_quickly(engine, tmp_path):
    path = _heavy_pdf(tmp_path, MAX_DRAWING_PATHS - 100)
    started = time.perf_counter()
    (p,) = extract_page_objects(path, engine=engine)
    assert time.perf_counter() - started < 3.0
    assert "drawings_capped" not in p.flags
    assert len(p.drawings) <= 200


def test_degenerate_boxes_are_clipped_or_dropped(engine, tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    image = dict(stream=_png(), keep_proportion=False)
    page.insert_image(pymupdf.Rect(-60, 100, 100, 200), **image)  # partly off the left
    page.insert_image(pymupdf.Rect(560, 700, 700, 900), **image)  # off right and bottom
    path = _save(doc, tmp_path)
    (p,) = extract_page_objects(path, engine=engine)
    assert len(p.images) == 2
    for b in p.images:
        assert 0 <= b.x0 < b.x1 <= 612 and 0 <= b.y0 < b.y1 <= 792
    left, right = sorted(p.images, key=lambda b: b.x0)
    assert (left.x0, left.x1) == pytest.approx((0, 100), abs=1)
    assert (right.x0, right.x1, right.y0, right.y1) == pytest.approx((560, 612, 700, 792), abs=1)


def test_to_box_drops_zero_area_and_off_page_boxes():
    assert objects._to_box(1, (10, 10, 10, 50), 612, 792) is None  # zero width
    assert objects._to_box(1, (10, 10, 50, 10), 612, 792) is None  # zero height
    assert objects._to_box(1, (700, 10, 800, 50), 612, 792) is None  # fully off the page
    assert objects._to_box(1, (-40, -5, 30, 20), 612, 792).model_dump() == {
        "page": 1, "x0": 0.0, "y0": 0.0, "x1": 30.0, "y1": 20.0,
    }  # fmt: skip
    flipped = objects._to_box(2, (50, 60, 10, 20), 612, 792)  # corners in the wrong order
    assert (flipped.x0, flipped.y0, flipped.x1, flipped.y1) == (10, 20, 50, 60)


def test_empty_pages_do_not_crash(engine, tmp_path):
    doc = pymupdf.open()
    doc.new_page(width=612, height=792)
    doc.new_page(width=300, height=400)
    path = _save(doc, tmp_path)
    pages = extract_page_objects(path, engine=engine)
    assert [(p.width, p.height) for p in pages] == [(612, 792), (300, 400)]
    for p in pages:
        assert not (p.lines or p.images or p.drawings or p.tables or p.flags)
        assert p.char_count == p.invisible_chars == p.garbled_chars == p.n_drawing_paths == 0


def test_count_chars_rules():
    assert objects._count_chars("Hello  world\n") == (10, 0)
    assert objects._count_chars("a�b") == (3, 1)  # replacement character
    assert objects._count_chars("xy\U000f0000") == (5, 3)  # private use areas
    assert objects._count_chars("a\x01b\x7f\x85c\t") == (6, 3)  # C0 / C1 controls; tab is space
    assert objects._count_chars("ab (cid:12)(cid:7) cd") == (6, 2)  # one character per token
    assert objects._count_chars("日本語 Привет") == (9, 0)


_TO_UNICODE = (
    b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
    b"1 begincodespacerange <00> <FF> endcodespacerange\n"
    b"4 beginbfchar <01> <FFFD> <02> <E000> <03> <0001> <04> <0007> endbfchar\n"
    b"endcmap end end"
)


def _garbled_pdf(tmp_path):
    """A one-line page whose font maps codes 1-4 to U+FFFD, a Private Use character and two
    C0 controls; the line reads "AB" plus those four codes."""
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
        b" /Resources << /Font << /F1 5 0 R >> >> >>",
        None,  # content stream, below
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /ToUnicode 6 0 R >>",
        None,  # ToUnicode stream
    ]
    content = b"BT /F1 12 Tf 72 700 Td <414201020304> Tj ET"
    streams = {3: content, 5: _TO_UNICODE}
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs):
        offsets.append(len(out))
        if body is None:
            data = streams[i]
            body = b"<< /Length %d >>\nstream\n" % len(data) + data + b"\nendstream"
        out += b"%d 0 obj\n" % (i + 1) + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    path = tmp_path / "garbled.pdf"
    path.write_bytes(bytes(out))
    return path


def test_char_counts(tmp_path):
    path = _garbled_pdf(tmp_path)
    (p,) = extract_page_objects(path, engine="pymupdf")
    assert (p.char_count, p.garbled_chars, p.invisible_chars) == (6, 4, 0)

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "Visible words here", fontsize=12)
    page.insert_text((72, 130), "Hidden ocr layer", fontsize=12, render_mode=3)
    path = _save(doc, tmp_path, "invisible.pdf")
    (p,) = extract_page_objects(path, engine="pymupdf")
    assert p.char_count == 16 + 14 and p.invisible_chars == 14 and p.garbled_chars == 0
    pytest.importorskip("pdfplumber")
    (q,) = extract_page_objects(path, engine="pdfplumber")
    assert q.invisible_chars == 0 and q.garbled_chars == 0 and q.char_count == 16 + 14


def test_pdfplumber_counts_garbled_characters(tmp_path):
    pytest.importorskip("pdfplumber")
    path = _garbled_pdf(tmp_path)
    (q,) = extract_page_objects(path, engine="pdfplumber")
    assert q.char_count >= 2 and q.invisible_chars == 0


def test_max_pages_progress_and_errors(engine, tmp_path):
    doc = pymupdf.open()
    for _ in range(12):
        doc.new_page(width=200, height=200)
    path = _save(doc, tmp_path)
    calls = []
    pages = extract_page_objects(
        path, engine=engine, max_pages=11, progress=lambda done, total: calls.append((done, total))
    )
    assert [p.page for p in pages] == list(range(1, 12))
    assert calls == [(10, 11)]

    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf at all")
    with pytest.raises(DocrooterError):
        extract_page_objects(bad, engine=engine)
    with pytest.raises(DocrooterError):
        extract_page_objects(path, engine="nope")  # type: ignore[arg-type]
