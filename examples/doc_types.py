"""Document type and canonical section labels with the built-in taxonomies (keyword backend).

Run it as is for a Markdown paper, or pass a PDF path: ``python examples/doc_types.py x.pdf``
loads it with PyMuPDF4LLM (``pip install 'docrooter[pdf]'``, AGPL-3.0).
"""

import sys

from docrooter import (
    DocumentView,
    builtin_taxonomy,
    classify_document,
    list_builtin_taxonomies,
    load,
    normalize_sections,
)

MARKDOWN = """\
# Sparse Attention for Long Documents

## Abstract
We propose a sparse attention method and report experiments, results and a dataset.

## Methods
Our method and experimental setup.

## Results
Results of the experiments against the baselines.

## References
Cited works and prior research.
"""


def main(path: str | None = None) -> None:
    doc = load(path) if path else load(MARKDOWN)
    print(", ".join(list_builtin_taxonomies()))
    doc_types = builtin_taxonomy("doc_types")
    view = DocumentView(fields=("title", "toc", "lead"), max_tokens=512)
    result = classify_document(doc, doc_types, backend="keyword", view=view)
    print(" > ".join(result.best.labels), f"{result.confidence:.2f}")
    doc_type = result.best.node_ids[-1]
    norm = normalize_sections(doc, doc_type=doc_type, backend="keyword", levels=[2])
    for section in norm.sections.values():
        print(f"{section.title}: {section.canonical}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
