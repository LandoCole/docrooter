import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

import docrooter
from docrooter.elements import Box
from docrooter.errors import DocrooterError, MissingExtraError
from docrooter.ocr import (
    OcrEngine,
    OcrLine,
    available_ocr_engines,
    get_ocr_engine,
    ocr_lines_to_lines,
    register_ocr_engine,
    resolve_auto_ocr,
)
from docrooter.ocr import registry as registry_mod
from docrooter.ocr.tesseract import TesseractEngine


def _ocr(text, x0, y0, x1, y1, conf=0.9, page=1):
    return OcrLine(text=text, box=Box(page=page, x0=x0, y0=y0, x1=x1, y1=y1), confidence=conf)


def test_ocr_lines_to_lines_geometry():
    ocr = [
        _ocr("Title of the page", 72, 60, 300, 84),
        _ocr("first body line", 72, 100, 400, 112),
        _ocr("second body line", 72, 114, 420, 126),
        _ocr("  ", 72, 130, 100, 140),  # blank: dropped
        _ocr("... ---", 72, 130, 100, 140),  # no letter or digit: dropped
        _ocr("last line", 72, 300, 200, 312),
    ]
    lines = ocr_lines_to_lines(ocr, page=3, width=612.0, height=792.0)
    assert [ln.text for ln in lines] == [
        "Title of the page",
        "first body line",
        "second body line",
        "last line",
    ]
    title = lines[0]
    assert title.size == pytest.approx(24 * 0.75)
    assert (title.font, title.bold, title.page) == ("ocr", False, 3)
    assert (title.page_width, title.page_height) == (612.0, 792.0)
    assert (title.x0, title.y0, title.x1, title.y1) == (72, 60, 300, 84)
    # gaps are to the neighbouring line, clamped at 0, infinite at the page edges
    assert math.isinf(lines[0].gap_above) and math.isinf(lines[-1].gap_below)
    assert lines[0].gap_below == pytest.approx(16.0)
    assert lines[1].gap_above == pytest.approx(16.0)
    assert lines[1].gap_below == pytest.approx(2.0)
    assert lines[2].gap_above == pytest.approx(2.0)
    assert lines[2].gap_below == pytest.approx(174.0)
    # a gap above half the line height starts a block
    assert [ln.block for ln in lines] == [0, 1, 1, 2]


def test_ocr_lines_to_lines_empty_and_overlap():
    assert ocr_lines_to_lines([], 1, 612.0, 792.0) == []
    # overlapping boxes (a column jump) clamp the gap to zero and start a block
    lines = ocr_lines_to_lines(
        [
            _ocr("left column end", 72, 500, 280, 512),
            _ocr("right column start", 320, 100, 540, 112),
        ],
        1,
        612.0,
        792.0,
    )
    assert lines[0].gap_below == 0.0 and lines[1].gap_above == 0.0


def test_registry_unknown_engine_error_lists_available():
    with pytest.raises(DocrooterError) as exc:
        get_ocr_engine("no-such-engine")
    message = str(exc.value)
    assert "no-such-engine" in message
    for name in ("ocrmac", "rapidocr", "tesseract"):
        assert name in message


def test_register_custom_engine(monkeypatch):
    monkeypatch.setattr(registry_mod, "_FACTORIES", dict(registry_mod._FACTORIES))

    class Fake:
        name = "fake"

        def __init__(self, lang):
            self.lang = lang

        def recognize(self, image, *, width_px, height_px, page):
            return []

        def close(self):
            pass

    register_ocr_engine("fake", lambda *, lang="en": Fake(lang))
    assert "fake" in available_ocr_engines()
    engine = get_ocr_engine("fake", lang="de")
    assert isinstance(engine, OcrEngine) and engine.lang == "de"
    with pytest.raises(DocrooterError):
        register_ocr_engine("auto", lambda **kw: Fake("x"))


def test_auto_ocr_resolution_order(monkeypatch):
    installed = {"ocrmac", "rapidocr", "onnxruntime"}
    binaries = {"tesseract"}
    monkeypatch.setattr(
        registry_mod.importlib.util,
        "find_spec",
        lambda name, *a: object() if name in installed else None,
    )
    monkeypatch.setattr(
        registry_mod.shutil, "which", lambda name, *a, **k: "/bin/x" if name in binaries else None
    )
    monkeypatch.setattr(sys, "platform", "darwin")
    assert resolve_auto_ocr() == "tesseract"  # measured CER 0.058: the best engine wins
    binaries.clear()
    assert resolve_auto_ocr() == "ocrmac"
    monkeypatch.setattr(sys, "platform", "linux")  # ocrmac is darwin-only
    assert resolve_auto_ocr() == "rapidocr"
    binaries.add("tesseract")
    assert resolve_auto_ocr() == "tesseract"
    binaries.clear()
    installed.discard("onnxruntime")  # rapidocr needs it
    assert resolve_auto_ocr() is None
    with pytest.raises(DocrooterError, match="no OCR engine"):
        get_ocr_engine("auto")


def test_missing_extra_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "rapidocr", None)  # makes the import fail
    with pytest.raises(MissingExtraError) as exc:
        get_ocr_engine("rapidocr")
    assert "pip install 'docrooter[ocr]'" in str(exc.value)
    monkeypatch.setitem(sys.modules, "ocrmac", None)
    with pytest.raises(MissingExtraError, match=r"docrooter\[ocr-mac\]"):
        get_ocr_engine("ocrmac")


def test_missing_tesseract_binary_message(monkeypatch):
    monkeypatch.setattr("docrooter.ocr.tesseract.shutil.which", lambda name: None)
    with pytest.raises(DocrooterError, match="tesseract binary was not found"):
        get_ocr_engine("tesseract")


TSV = (
    "\t".join(
        "level page_num block_num par_num line_num word_num left top width height conf text".split()
    )
    + "\n"
    + "\n".join(
        "\t".join(str(c) for c in row)
        for row in [
            (1, 1, 0, 0, 0, 0, 0, 0, 1000, 500, -1, ""),
            (2, 1, 1, 0, 0, 0, 100, 50, 400, 100, -1, ""),
            (4, 1, 1, 1, 1, 0, 100, 50, 400, 30, -1, ""),
            (5, 1, 1, 1, 1, 1, 100, 50, 120, 30, 96.0, "Hello"),
            (5, 1, 1, 1, 1, 2, 240, 52, 260, 28, 90.0, "world"),
            (4, 1, 1, 1, 2, 0, 100, 100, 400, 30, -1, ""),
            (5, 1, 1, 1, 2, 1, 100, 100, 200, 30, 80.0, "second"),
            (5, 1, 1, 1, 2, 2, 320, 100, 90, 30, 60.0, "line"),
            (5, 1, 1, 1, 2, 3, 420, 100, 40, 30, -1, ""),  # empty word: skipped
            (5, 1, 2, 1, 1, 1, 600, 300, 100, 40, 99.0, "other-block"),
        ]
    )
    + "\n"
)


def test_tesseract_tsv_parsing(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout=TSV.encode(), stderr=b"")

    monkeypatch.setattr("docrooter.ocr.tesseract.shutil.which", lambda name: "/opt/tess")
    monkeypatch.setattr("docrooter.ocr.tesseract.subprocess.run", fake_run)
    engine = TesseractEngine(lang="en")
    # the 1000 x 500 px image shows a 500 x 250 pt page: half a point per pixel
    lines = engine.recognize(
        b"PNG", width_px=1000, height_px=500, page=Box(page=2, x0=0, y0=0, x1=500, y1=250)
    )
    cmd, kwargs = calls[0]
    assert cmd == ["/opt/tess", "stdin", "stdout", "-l", "eng", "--psm", "3", "tsv"]
    assert kwargs["input"] == b"PNG"
    assert kwargs["timeout"] == 300
    assert not kwargs.get("shell")
    assert [ln.text for ln in lines] == ["Hello world", "second line", "other-block"]
    first = lines[0]
    assert first.confidence == pytest.approx((96 + 90) / 2 / 100)
    assert lines[1].confidence == pytest.approx((80 + 60) / 2 / 100)
    assert (first.box.page, first.box.x0, first.box.y0) == (2, 50.0, 25.0)
    assert (first.box.x1, first.box.y1) == (250.0, 40.0)  # right edge 240+260=500 px; 50+30=80 px


def test_tesseract_failure_is_reported(monkeypatch):
    monkeypatch.setattr("docrooter.ocr.tesseract.shutil.which", lambda name: "/opt/tess")
    monkeypatch.setattr(
        "docrooter.ocr.tesseract.subprocess.run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"no language"),
    )
    engine = TesseractEngine()
    with pytest.raises(DocrooterError, match="no language"):
        engine.recognize(b"x", width_px=10, height_px=10, page=Box(page=1, x0=0, y0=0, x1=1, y1=1))


def test_importing_ocr_loads_no_engine_package():
    # in a fresh interpreter: the unit run may have loaded these through pymupdf4llm
    src = str(Path(docrooter.__file__).parents[1])
    code = (
        "import sys, docrooter.ocr, docrooter.pdf.routing\n"
        "bad = [m for m in ('rapidocr', 'onnxruntime', 'ocrmac', 'torch') if m in sys.modules]\n"
        "assert not bad, bad"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": src},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


PAGE = Box(page=2, x0=0, y0=0, x1=500, y1=250)  # shown by a 1000 x 500 px image


def test_rapidocr_adapter_scales_quads_to_page_points(monkeypatch):
    import types

    seen = {}

    class Out:
        boxes = [[[100, 50], [300, 52], [298, 80], [100, 78]], [[0, 0], [10, 0], [10, 10], [0, 10]]]
        txts = ("Hello world", "  ")
        scores = (0.9, 0.5)

    class FakeRapidOCR:
        def __init__(self, params=None):
            seen["params"] = params

        def __call__(self, image):
            seen["image"] = image
            return Out()

    fake = types.ModuleType("rapidocr")
    fake.RapidOCR = FakeRapidOCR
    monkeypatch.setitem(sys.modules, "rapidocr", fake)
    engine = get_ocr_engine("rapidocr")
    lines = engine.recognize(b"PNG", width_px=1000, height_px=500, page=PAGE)
    assert seen["image"] == b"PNG"
    assert [ln.text for ln in lines] == ["Hello world"]  # the blank one is dropped
    box = lines[0].box
    assert (box.page, box.x0, box.y0, box.x1, box.y1) == (2, 50.0, 25.0, 150.0, 40.0)
    assert lines[0].confidence == 0.9
    engine.close()
    engine.close()  # twice is fine

    class Empty:
        boxes = txts = scores = None

    FakeRapidOCR.__call__ = lambda self, image: Empty()
    assert get_ocr_engine("rapidocr").recognize(b"", width_px=1, height_px=1, page=PAGE) == []


def test_ocrmac_adapter_scales_pixel_boxes(monkeypatch):
    import types

    calls = {}

    class FakeOCR:
        def __init__(self, image, **kwargs):
            calls["kwargs"] = kwargs

        def recognize(self, px=False):
            calls["px"] = px
            return [("Title", 1.0, (100, 50, 300, 80)), (" ", 1.0, (0, 0, 1, 1))]

    package = types.ModuleType("ocrmac")
    inner = types.ModuleType("ocrmac.ocrmac")
    inner.OCR = FakeOCR
    package.ocrmac = inner
    pil = types.ModuleType("PIL")
    pil_image = types.ModuleType("PIL.Image")
    pil_image.open = lambda fp: object()
    pil.Image = pil_image
    for name, mod in (
        ("ocrmac", package),
        ("ocrmac.ocrmac", inner),
        ("PIL", pil),
        ("PIL.Image", pil_image),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    engine = get_ocr_engine("ocrmac", lang="de")
    lines = engine.recognize(b"PNG", width_px=1000, height_px=500, page=PAGE)
    assert calls["kwargs"] == {"recognition_level": "accurate", "language_preference": ["de-DE"]}
    assert calls["px"] is True
    assert [ln.text for ln in lines] == ["Title"]
    assert (lines[0].box.x0, lines[0].box.y0, lines[0].box.x1, lines[0].box.y1) == (
        50.0,
        25.0,
        150.0,
        40.0,
    )
    engine.close()


# --- rendering -----------------------------------------------------------------------------


def _pdf(tmp_path, rotation=0):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "Render me", fontsize=20)
    if rotation:
        page.set_rotation(rotation)
    path = tmp_path / "r.pdf"
    doc.save(path)
    doc.close()
    return path


def _png_size(png):
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    return int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")


@pytest.mark.parametrize("backend", ["pymupdf", "pdfium"])
@pytest.mark.parametrize("rotation", [0, 90])
def test_render_page_and_clip(tmp_path, monkeypatch, backend, rotation):
    from docrooter.ocr import render

    path = _pdf(tmp_path, rotation)
    if backend == "pdfium":
        pytest.importorskip("pypdfium2")
        monkeypatch.setitem(sys.modules, "pymupdf", None)  # force the fallback
    width_pt, height_pt = (612, 792) if rotation == 0 else (792, 612)
    png, w, h = render.render_page(path, 1, dpi=144)
    assert (w, h) == (width_pt * 2, height_pt * 2) and _png_size(png) == (w, h)
    clip = Box(page=1, x0=0, y0=0, x1=300, y1=100)
    png, w, h = render.render_page(path, 1, dpi=144, clip=clip)
    assert (w, h) == (600, 200) and _png_size(png) == (w, h)
    with pytest.raises(DocrooterError, match="out of range"):
        render.render_page(path, 2)
    with pytest.raises(DocrooterError, match="dpi"):
        render.render_page(path, 1, dpi=0)


def _huge_pdf(tmp_path, side=3600):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page(width=side, height=side)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 8, 8), False)
    pix.set_rect(pix.irect, (200, 200, 200))
    page.insert_image(page.rect, stream=pix.tobytes("png"), keep_proportion=False)
    path = tmp_path / "huge.pdf"
    doc.save(path)
    doc.close()
    return path


def test_render_dpi_caps_the_pixel_count():
    from docrooter.ocr.render import MAX_RENDER_PIXELS, render_dpi

    assert MAX_RENDER_PIXELS == 25_000_000
    assert render_dpi(612, 792, 300) == 300  # a letter page at 300 dpi is 8.4 MP: no cap
    capped = render_dpi(3600, 3600, 200)  # 10000 x 10000 px at 200 dpi
    assert capped < 200
    assert (3600 * capped / 72) ** 2 <= MAX_RENDER_PIXELS
    assert (3600 * capped / 72) ** 2 >= 0.99 * MAX_RENDER_PIXELS
    assert render_dpi(14400, 14400, 600) > 0  # the largest PDF page still renders


@pytest.mark.parametrize("backend", ["pymupdf", "pdfium"])
def test_huge_page_renders_within_the_pixel_cap(tmp_path, monkeypatch, backend):
    # Final review R-I2: a 3.6 KB PDF with a 3600 pt page asked 10000 x 10000 px at 200 dpi
    from docrooter.ocr import render

    path = _huge_pdf(tmp_path)
    if backend == "pdfium":
        pytest.importorskip("pypdfium2")
        monkeypatch.setitem(sys.modules, "pymupdf", None)  # force the fallback
    png, w, h = render.render_page(path, 1, dpi=200)
    assert w * h <= render.MAX_RENDER_PIXELS and _png_size(png) == (w, h)
    assert w * h >= 0.98 * render.MAX_RENDER_PIXELS  # capped, not shrunk further
    clip = Box(page=1, x0=0, y0=0, x1=3600, y1=1800)
    _, w, h = render.render_page(path, 1, dpi=600, clip=clip)
    assert w * h <= render.MAX_RENDER_PIXELS / 2 + w + h
    rgb, width_pt, height_pt = render.render_page_rgb(path, 1, dpi=200)
    assert rgb.shape[0] * rgb.shape[1] <= render.MAX_RENDER_PIXELS
    assert (width_pt, height_pt) == (3600, 3600)


@pytest.mark.parametrize("lang", ["-l", "--version", "eng;rm", "eng deu", "", "eng+", "../eng"])
def test_tesseract_rejects_invalid_language(monkeypatch, lang):
    monkeypatch.setattr("docrooter.ocr.tesseract.shutil.which", lambda name: "/opt/tess")
    with pytest.raises(ValueError, match="invalid tesseract language"):
        TesseractEngine(lang=lang)


@pytest.mark.parametrize(
    ("lang", "code"), [("deu+eng", "deu+eng"), ("chi_sim", "chi_sim"), ("fr", "fra")]
)
def test_tesseract_accepts_valid_languages(monkeypatch, lang, code):
    monkeypatch.setattr("docrooter.ocr.tesseract.shutil.which", lambda name: "/opt/tess")
    assert TesseractEngine(lang=lang)._lang == code


def test_tesseract_timeout_is_a_docrooter_error(monkeypatch):
    def hang(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr("docrooter.ocr.tesseract.shutil.which", lambda name: "/opt/tess")
    monkeypatch.setattr("docrooter.ocr.tesseract.subprocess.run", hang)
    engine = TesseractEngine()
    with pytest.raises(DocrooterError, match="timed out after 300 s"):
        engine.recognize(b"x", width_px=10, height_px=10, page=Box(page=1, x0=0, y0=0, x1=1, y1=1))
