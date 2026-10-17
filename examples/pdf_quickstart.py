"""Read a PDF's table of contents, then name its type and its sections (keyword backend).

Usage: ``python examples/pdf_quickstart.py report.pdf``. Needs a PDF engine: install
``docrooter[pdf]`` (PyMuPDF, AGPL-3.0) or ``docrooter[pdf-mit]`` (pdfplumber, MIT). The keyword
backend needs no models; swap in ``backend="embedding"`` (``docrooter[models]``) or ``"jev"`` for
real accuracy.
"""

import sys

from docrooter import builtin_taxonomy, classify_document, load, normalize_sections


def main(path: str) -> None:
    doc = load(path)  # engine="auto": PyMuPDF if installed, else pdfplumber; no OCR
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
    if len(sys.argv) != 2:
        sys.exit("usage: python examples/pdf_quickstart.py FILE.pdf")
    main(sys.argv[1])
