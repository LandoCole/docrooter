"""Build the golden PDFs the element-mode and rooter tests read (only when missing, so a rerun
keeps the committed inputs)::

    PYTHONPATH=src .venv/bin/python tests/unit/pdf/golden/generate.py

They once also pinned v0.3's heuristic fast path as JSON; that path is gone (the section tree is
read with PyMuPDF4LLM), and so are those goldens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
NAMES = ("single_column", "two_column", "tables", "scanned", "garbled", "bookmarks")

BODY = "This sentence is ordinary body text that stays small and keeps the body size at eleven."


def _paragraph(page: Any, y: float, text: str, *, x: float = 72, n: int = 4) -> float:
    for k in range(n):
        page.insert_text((x, y), f"{text} Line {k}.", fontsize=11, fontname="helv")
        y += 15
    return y + 22


def _single_column(pymupdf: Any) -> Any:
    doc = pymupdf.open()
    for n, (h1, h2) in enumerate(
        [("Introduction", "Background"), ("Methods", "Data Sources"), ("Results", "Discussion")],
        start=1,
    ):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 40), "ACME Technical Report", fontsize=9, fontname="helv")
        page.insert_text((300, 760), str(n), fontsize=9, fontname="helv")
        y = 90.0
        if n == 1:
            y = _paragraph(page, y, "Preface words come first.", n=2)
        page.insert_text((72, y), f"{n} {h1}", fontsize=22, fontname="hebo")
        y = _paragraph(page, y + 40, BODY)
        page.insert_text((72, y), f"{n}.1 {h2}", fontsize=16, fontname="hebo")
        _paragraph(page, y + 32, BODY)
    return doc


def _two_column(pymupdf: Any) -> Any:
    doc = pymupdf.open()
    for n in (1, 2):
        page = doc.new_page(width=612, height=792)
        if n == 1:
            page.insert_text((72, 70), "A Study of Two Columns", fontsize=24, fontname="hebo")
        for col, x in enumerate((72, 320)):
            y = 120.0
            for part in (1, 2):
                number = f"{2 * (n - 1) + col + 1}.{part}"
                page.insert_text((x, y), f"{number} Column Topic {number}", fontsize=14,
                                 fontname="hebo")  # fmt: skip
                y += 24
                for k in range(10):
                    page.insert_text((x, y), f"Column text {number} line {k} runs on", fontsize=10,
                                     fontname="helv")  # fmt: skip
                    y += 13
                y += 20
    return doc


def _tables(pymupdf: Any) -> Any:
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 80), "Quarterly Figures", fontsize=22, fontname="hebo")
    y = _paragraph(page, 120, BODY, n=3)
    page.insert_text((72, y), "Table 1. Results by region", fontsize=9, fontname="helv")
    x0, y0, cw, rh = 72.0, y + 10, 150.0, 20.0
    for r in range(5):
        page.draw_line((x0, y0 + r * rh), (x0 + 3 * cw, y0 + r * rh))
    for c in range(4):
        page.draw_line((x0 + c * cw, y0), (x0 + c * cw, y0 + 4 * rh))
    for r in range(4):
        for c in range(3):
            text = ("Region", "Revenue", "Margin")[c] if r == 0 else f"{r * 100 + c * 7}"
            page.insert_text((x0 + c * cw + 4, y0 + r * rh + 14), text, fontsize=9)
    page.insert_text((72, y0 + 4 * rh + 40), "Commentary", fontsize=16, fontname="hebo")
    _paragraph(page, y0 + 4 * rh + 72, BODY, n=3)
    return doc


def _scanned(pymupdf: Any) -> Any:
    doc = pymupdf.open()
    for shade in (200, 150):
        page = doc.new_page(width=612, height=792)
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 85, 110), False)
        pix.set_rect(pix.irect, (shade, shade, shade))
        page.insert_image(page.rect, stream=pix.tobytes("png"), keep_proportion=False)
    return doc


_TO_UNICODE = (
    b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
    b"1 begincodespacerange <00> <FF> endcodespacerange\n"
    b"3 beginbfchar <01> <FFFD> <02> <E000> <03> <E001> endbfchar\n"
    b"endcmap end end"
)


def _garbled_bytes() -> bytes:
    """Twenty lines in a font whose ToUnicode maps codes 1-3 to U+FFFD and Private Use
    characters, with a few plain letters among them."""
    rows = []
    for k in range(20):
        rows.append(b"BT /F1 12 Tf 72 %d Td <41010203420102030141> Tj ET" % (720 - 18 * k))
    content = b"\n".join(rows)
    objs: list[bytes | None] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
        b" /Resources << /Font << /F1 5 0 R >> >> >>",
        None,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /ToUnicode 6 0 R >>",
        None,
    ]
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
    return bytes(out)


def _bookmarks(pymupdf: Any) -> Any:
    doc = pymupdf.open()
    for title, sub in (("Alpha", "Alpha Detail"), ("Beta", "Beta Detail")):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 90), title, fontsize=22, fontname="hebo")
        y = _paragraph(page, 130, BODY)
        page.insert_text((72, y), sub, fontsize=16, fontname="hebo")
        _paragraph(page, y + 32, BODY)
    doc.set_toc([[1, "Alpha", 1], [2, "Alpha Detail", 1], [2, "Beta", 2], [3, "Beta Detail", 2]])
    return doc


def build_pdfs(directory: Path = HERE) -> list[Path]:
    """Write the six PDFs that are missing; return all six paths."""
    import pymupdf

    builders = {
        "single_column": _single_column, "two_column": _two_column, "tables": _tables,
        "scanned": _scanned, "bookmarks": _bookmarks,
    }  # fmt: skip
    paths = []
    for name in NAMES:
        path = directory / f"{name}.pdf"
        paths.append(path)
        if path.exists():
            continue
        if name == "garbled":
            path.write_bytes(_garbled_bytes())
            continue
        doc = builders[name](pymupdf)
        doc.save(str(path), garbage=3, deflate=True, no_new_id=True)
        doc.close()
    return paths


def main() -> None:
    for path in build_pdfs():
        print(path)


if __name__ == "__main__":
    main()
