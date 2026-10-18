"""The fast path is frozen: ``load_pdf()`` with default arguments gives v0.3's output.

The goldens in ``golden/`` were written by ``feat/v0.3`` (28fbf9d) from a git worktree with
``golden/generate.py``; only ``source`` and the timing values are normalised.
"""

import json
from pathlib import Path

import pytest

from docrooter.pdf.load import load_pdf

pytest.importorskip("pymupdf")

GOLDEN = Path(__file__).parent / "golden"
NAMES = ("single_column", "two_column", "tables", "scanned", "garbled", "bookmarks")


def _normalise(doc, name):
    data = json.loads(doc.model_dump_json())
    data["source"] = name
    data["meta"]["timings"] = {key: 0.0 for key in data["meta"]["timings"]}
    return data


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("engine", [None, "pdfplumber"])
def test_load_pdf_default_output_equals_v03(name, engine):
    if engine == "pdfplumber":
        pytest.importorskip("pdfplumber")
    path = GOLDEN / f"{name}.pdf"
    doc = load_pdf(path) if engine is None else load_pdf(path, engine=engine)
    golden = GOLDEN / (f"{name}.json" if engine is None else f"{name}.{engine}.json")
    expected = golden.read_text(encoding="utf-8")
    got = json.dumps(_normalise(doc, path.name), indent=2, ensure_ascii=False) + "\n"
    assert got == expected
    assert "routes" not in doc.meta and "element_mode" not in doc.meta
    assert doc.elements == [] and "elements" not in json.loads(doc.model_dump_json())
