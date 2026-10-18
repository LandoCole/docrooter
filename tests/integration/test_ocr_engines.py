"""Real OCR engines on a generated text page. Run with ``pytest -m ocr``; each engine that is
not installed is skipped."""

import importlib.util
import re
import shutil
import sys

import pytest

from docrooter.elements import Box
from docrooter.ocr import get_ocr_engine, ocr_lines_to_lines, render_page

pytestmark = pytest.mark.ocr

pymupdf = pytest.importorskip("pymupdf")

PARAGRAPHS = [
    "Annual Report of the Company",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Revenue increased by twelve percent during the fiscal year ended June.",
    "Management expects continued growth across all of its operating segments.",
    "Risk factors include currency movements and changes in consumer demand.",
]


def _engine_available(name: str) -> bool:
    if name == "tesseract":
        return shutil.which("tesseract") is not None
    if name == "ocrmac":
        return sys.platform == "darwin" and importlib.util.find_spec("ocrmac") is not None
    return importlib.util.find_spec("rapidocr") is not None


ENGINES = ["ocrmac", "rapidocr", "tesseract"]


@pytest.fixture(scope="module")
def pdf_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("ocr") / "page.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    y = 100
    for k, text in enumerate(PARAGRAPHS):
        page.insert_text((72, y), text, fontsize=18 if k == 0 else 12, fontname="helv")
        y += 40
    doc.save(path)
    doc.close()
    return path


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _recall(expected: str, got: str) -> float:
    want, have = _words(expected), _words(got)
    hits = 0
    for word in want:
        if word in have:
            have.remove(word)
            hits += 1
    return hits / len(want)


@pytest.fixture(params=ENGINES)
def engine(request):
    if not _engine_available(request.param):
        pytest.skip(f"{request.param} is not installed")
    eng = get_ocr_engine(request.param)
    yield eng
    eng.close()


def test_engine_reads_a_rendered_page(engine, pdf_path):
    png, w, h = render_page(pdf_path, 1, dpi=200)
    page = Box(page=1, x0=0, y0=0, x1=612, y1=792)
    lines = engine.recognize(png, width_px=w, height_px=h, page=page)
    assert lines, f"{engine.name} read nothing"
    assert _recall(" ".join(PARAGRAPHS), " ".join(ln.text for ln in lines)) >= 0.90
    for ln in lines:
        assert 0.0 <= ln.confidence <= 1.0
        assert 0 <= ln.box.x0 < ln.box.x1 <= 612 + 1
        assert 0 <= ln.box.y0 < ln.box.y1 <= 792 + 1
    # the title is the first line, near y = 100 - 18 pt (its baseline is at 100)
    title = min(lines, key=lambda ln: ln.box.y0)
    assert "annual" in title.text.lower()
    assert 70 <= title.box.y0 <= 100 and 95 <= title.box.y1 <= 115
    # and the lines convert into Line objects with sizes in the right range
    converted = ocr_lines_to_lines(lines, 1, 612.0, 792.0)
    assert converted[0].size > converted[-1].size  # the 18 pt title beats the 12 pt body


def test_engine_reads_a_clipped_region_in_page_points(engine, pdf_path):
    clip = Box(page=1, x0=50, y0=170, x1=600, y1=240)  # the third paragraph's neighbourhood
    png, w, h = render_page(pdf_path, 1, dpi=200, clip=clip)
    lines = engine.recognize(png, width_px=w, height_px=h, page=clip)
    text = " ".join(ln.text for ln in lines)
    assert _recall(PARAGRAPHS[2], text) >= 0.90
    for ln in lines:
        assert clip.x0 - 1 <= ln.box.x0 and ln.box.x1 <= clip.x1 + 1
        assert clip.y0 - 1 <= ln.box.y0 and ln.box.y1 <= clip.y1 + 1


def test_pypdfium2_render_feeds_an_engine(pdf_path, monkeypatch):
    pytest.importorskip("pypdfium2")
    name = next((n for n in ENGINES if _engine_available(n)), None)
    if name is None:
        pytest.skip("no OCR engine is installed")
    monkeypatch.setitem(sys.modules, "pymupdf", None)  # the MIT-only path: no pymupdf
    png, w, h = render_page(pdf_path, 1, dpi=200)
    monkeypatch.undo()
    eng = get_ocr_engine(name)
    try:
        lines = eng.recognize(
            png, width_px=w, height_px=h, page=Box(page=1, x0=0, y0=0, x1=612, y1=792)
        )
    finally:
        eng.close()
    assert _recall(" ".join(PARAGRAPHS), " ".join(ln.text for ln in lines)) >= 0.90
