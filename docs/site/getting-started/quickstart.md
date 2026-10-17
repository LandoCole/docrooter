# Quickstart: a PDF in five minutes

This walks one PDF through the whole pipeline: table of contents, document type, and a
canonical label for every section. It uses the `keyword` backend so that nothing needs to be
downloaded; the last step swaps in a real model.

## 1. Install

```bash
pip install 'docrooter[pdf]'       # PyMuPDF engine (AGPL-3.0)
# or: pip install 'docrooter[pdf-mit]'   # pdfplumber engine (MIT)
```

Any text-based PDF works: an annual report, a paper, a contract, a manual. Scanned PDFs
without a text layer do not (there is no OCR).

## 2. Run the script

Save [`examples/pdf_quickstart.py`](../../../examples/pdf_quickstart.py) and run it on your PDF:

```bash
python examples/pdf_quickstart.py report.pdf
```

```python
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
```

On Apple's Form 10-K for fiscal 2025 (80 pages) it prints, shortened:

```text
pymupdf: 80 pages, 94 headings
...
PART I  (p. 4)
  Item 1. Business  (p. 4)
    Company Background  (p. 4)
    Products  (p. 4)
...
  Item 1A. Risk Factors  (p. 8)
...
type: Financial > Annual report (confidence 0.08)
section taxonomy: section_types/annual_report
  Item 1. Business -> business_overview
  Item 1A. Risk Factors -> risk_factors
  Item 2. Properties -> properties
  Item 3. Legal Proceedings -> legal_proceedings
  Item 9A. Controls and Procedures -> controls_and_procedures
...
```

## 3. What happened

- **`load(path)`** read the PDF's text and layout and found headings from font size, weight,
  isolation, numbering, the printed contents page and the PDF's bookmarks (this filing has
  102). Every section has a `level`, a `page_start`, a `heading_score` and `flags` explaining
  unusual decisions. `doc.meta` records the engine, the page count and the time per stage. See
  [The TOC engine](../concepts/toc-engine.md).
- **`classify_document`** rendered the document into a short
  [view](../concepts/views.md) (title, statistics, contents, opening text, key phrases) and
  classified it against the built-in `doc_types` taxonomy. `confidence` is the probability of
  the chosen type.
- **`normalize_sections`** picked the section taxonomy for that type
  (`section_types/annual_report`) and labelled each level-1 and level-2 heading.

The keyword backend only counts shared words, which is why its confidence is so low and why
some labels are wrong (the cover page lines get section labels too). It is good for trying the
API and for tests, not for real documents.

## 4. Use a real model

Install the models and change `backend="keyword"` to the default:

```bash
pip install 'docrooter[pdf,models]'
```

The first run downloads `dmis-lab/UltraBERT-Embed-150M` (about 600 MB) and then runs on your
machine (CUDA, Apple MPS or CPU). For the highest accuracy, use `backend="jev"` with a
`TYPESAFE_API_KEY`; see [Jev and OpenAI-compatible LLMs](../guides/llm-backends.md). The
[benchmarks](../benchmarks/index.md) compare them.

## Next

- [Command line and playground](cli-and-playground.md): the same steps without writing code.
- [Document-type gating](../guides/doctype-gating.md): sort a folder of PDFs by type.
- [Custom taxonomies](../guides/custom-taxonomies.md): classify sections against your own
  categories.
