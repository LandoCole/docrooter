"""Read a PDF's table of contents, then name its type and its sections (keyword backend).

Usage: ``python examples/pdf_quickstart.py [report.pdf]``. Without a path it first writes a small
sample annual report (``sample_report.pdf``, 3 pages) to a scratch directory with PyMuPDF, so the
output is reproducible. Needs ``docrooter[pdf]`` (PyMuPDF4LLM, AGPL-3.0). The keyword backend
needs no models; swap in ``backend="embedding"`` (``docrooter[models]``) or ``"jev"`` for real
accuracy.
"""

import sys
import tempfile
from pathlib import Path

from docrooter import builtin_taxonomy, classify_document, load, normalize_sections

# The sample: one list of (heading, body lines) per page.
SAMPLE_PAGES = [
    [
        ("Annual Report 2025", []),
        ("Item 1. Business", ["We design, manufacture and sell industrial sensors."] * 3),
        ("Item 1A. Risk Factors", ["Our business faces risks from competition and supply."] * 3),
    ],
    [
        ("Item 7. Management's Discussion and Analysis", ["Revenue grew and margins rose."] * 4),
        ("Liquidity and Capital Resources", ["Cash and investments funded our operations."] * 3),
    ],
    [
        (
            "Item 8. Financial Statements",
            ["The consolidated balance sheet and income statement."] * 4,
        ),
        ("Item 9A. Controls and Procedures", ["Management assessed internal control."] * 3),
    ],
]


def write_sample(path: Path) -> None:
    """A small text-based PDF: bold 18 pt headings (14 pt for a subsection) over 10 pt body."""
    import pymupdf

    pdf = pymupdf.open()
    for blocks in SAMPLE_PAGES:
        page = pdf.new_page(width=612, height=792)
        y = 80.0
        for title, lines in blocks:
            size = 14 if title.startswith("Liquidity") else 18
            page.insert_text((72, y), title, fontsize=size, fontname="hebo")
            y += 30
            for line in lines:
                page.insert_text((72, y), line, fontsize=10)
                y += 14
            y += 24
    pdf.save(str(path))
    pdf.close()


def main(path: str) -> None:
    doc = load(path)  # the section tree from PyMuPDF4LLM (its layout model); no OCR
    print(f"{doc.meta['engine']}: {doc.meta['n_pages']} pages, {len(doc.toc())} headings")
    for entry in doc.toc():
        page = doc.sections[entry.id].page_start
        print(f"{'  ' * entry.depth}{entry.title}  (p. {page})")

    doc_types = builtin_taxonomy("doc_types")
    result = classify_document(doc, doc_types, backend="keyword")
    if not result.best.node_ids:  # e.g. "empty_text" for a scan without a text layer
        print("no type:", result.best.stopped_reason)
        return
    print("type:", " > ".join(result.best.labels), f"(confidence {result.confidence:.2f})")

    doc_type = result.best.node_ids[-1]
    norm = normalize_sections(doc, doc_type=doc_type, backend="keyword")
    print("section taxonomy:", norm.taxonomy)
    for section in norm.sections.values():
        print(f"  {section.title} -> {section.canonical}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        with tempfile.TemporaryDirectory() as scratch:  # deleted afterwards
            sample = Path(scratch) / "sample_report.pdf"
            write_sample(sample)
            main(str(sample))
