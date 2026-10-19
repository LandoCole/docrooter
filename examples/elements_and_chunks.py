"""Read a PDF as typed elements with the rooter, then cut it into chunks for retrieval (RAG).

Usage: ``python examples/elements_and_chunks.py [report.pdf]``. Without a path it first writes a
small sample report (``sample_elements.pdf``, 2 pages: a two-column page with a ruled table and a
caption, then a short page) to a scratch directory with PyMuPDF, so the output is reproducible.
Needs ``docrooter[pdf]`` (PyMuPDF, AGPL-3.0); for your own files only, ``docrooter[pdf-mit]``
(pdfplumber, MIT) also works. The keyword backends need no models and ``ocr="none"`` needs no
OCR engine; use ``ocr="auto"`` to OCR scanned pages when one is installed (see the OCR guide).
"""

import sys
import tempfile
import textwrap
from pathlib import Path

from docrooter import chunk, process

LEFT = (
    "Our sensors measure pressure, temperature and flow in factories and power plants. "
    "Demand grew in every region during the fiscal year, "
    "and we added two production lines to keep up. "
    "Orders from new customers rose by a third."
)
RIGHT = (
    "Competition remains strong, and our suppliers face shortages of some components. "
    "We keep a second source for each critical part and hold extra stock of the rest. "
    "Prices for raw materials were stable."
)
OUTLOOK = (
    "We expect revenue to keep growing as factories add sensors to older machines. "
    "Margins should hold steady."
)
TABLE = [
    ["Segment", "2024", "2025"],
    ["Sensors", "412", "468"],
    ["Controls", "198", "215"],
    ["Services", "96", "121"],
]


def write_sample(path: Path) -> None:
    """A two-page PDF: bold headings, two text columns, a ruled table with a caption, and a
    page number at the foot of each page."""
    import pymupdf

    def text(page, x, y, lines, *, size=10.0, font="helv"):
        for line in lines:
            page.insert_text((x, y), line, fontsize=size, fontname=font)
            y += 15

    pdf = pymupdf.open()
    first = pdf.new_page(width=612, height=792)
    text(first, 72, 80, ["Annual Report 2025"], size=24, font="hebo")
    text(first, 72, 125, ["1. Business"], size=16, font="hebo")
    text(first, 72, 155, textwrap.wrap(LEFT, 36))
    text(first, 330, 155, textwrap.wrap(RIGHT, 36))
    text(first, 72, 285, ["2. Financial results"], size=16, font="hebo")
    for r, row in enumerate(TABLE):
        for c, cell in enumerate(row):
            first.draw_rect(pymupdf.Rect(72 + c * 100, 310 + r * 18, 172 + c * 100, 328 + r * 18))
            first.insert_text((76 + c * 100, 323 + r * 18), cell, fontsize=10, fontname="helv")
    text(first, 72, 398, ["Table 1. Revenue by segment (millions)."], size=9, font="heit")
    text(first, 290, 750, ["Page 1"], size=9)
    second = pdf.new_page(width=612, height=792)
    text(second, 72, 80, ["3. Outlook"], size=16, font="hebo")
    text(second, 72, 105, textwrap.wrap(OUTLOOK, 80))
    text(second, 290, 750, ["Page 2"], size=9)
    pdf.save(str(path))
    pdf.close()


def main(path: str) -> None:
    analysis = process(path, profile="balanced", ocr="none", doc_type="keyword", sections="keyword")
    doc = analysis.document
    print(f"{analysis.config.profile}: {len(doc.elements)} elements, {len(doc.sections)} sections")
    print("type:", " > ".join(analysis.doc_type.best.labels))
    for element in doc.elements:
        label = element.text[:36].replace("\n", " ")
        print(f"{element.order:>2} p{element.boxes[0].page} {element.kind:<12}{label}")

    table = next(element for element in doc.elements if element.kind == "table")
    print(table.table.to_markdown())

    chunks = chunk(analysis, max_tokens=100)
    for piece in chunks:
        print(f"{piece.id}: {piece.tokens} tokens, {' > '.join(piece.heading_path)} {piece.flags}")
    print(chunks[2].text)  # a table is never split: a description line, then its rows


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        with tempfile.TemporaryDirectory() as scratch:  # deleted afterwards
            sample = Path(scratch) / "sample_elements.pdf"
            write_sample(sample)
            main(str(sample))
