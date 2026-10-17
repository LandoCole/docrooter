# docrooter

<div class="dr-hero" markdown>
![docrooter logo: a document page whose ruled lines branch into roots](../assets/logo.svg#only-light)
![docrooter logo: a document page whose ruled lines branch into roots](../assets/logo-dark.svg#only-dark)

**Find the structure of a document, then say what each part is.** Table-of-contents extraction,
document-type classification and section normalisation for PDF and Markdown, against a
taxonomy you supply.
</div>

## What it does

docrooter takes a PDF or Markdown file and gives you three things:

1. **A table of contents.** It reads PDFs natively (no OCR) and finds the headings from
   typography, numbering, the printed contents page and the PDF's bookmarks, then builds a
   section tree with levels and page numbers.
2. **The document's type.** A built-in taxonomy of 18 types in 6 groups (annual report,
   contract, research paper, datasheet, ...) or your own. You get the type, a confidence you can
   gate a pipeline on, and the probabilities of the runners-up.
3. **What each section is.** Each heading is mapped to a canonical section of its document type
   ("Item 1A. Risk Factors" becomes `risk_factors`), or classified against any taxonomy you
   write in JSON or CSV, with the full path through the tree and the decision at every step.

It runs zero-shot out of the box: small embedding models run on a laptop, and LLM backends
(TypeSafe Jev or any OpenAI-compatible server) are optional. When zero-shot is not enough it
trains per-node classifiers from a few labelled examples, and it measures itself with built-in
evaluation reports.

## Why it exists

Document pipelines usually need structure before they need answers: *which* filings are annual
reports, *where* the risk factors start, *which* clauses are termination clauses. Generic text
extractors flatten that structure, and sending whole documents to a large model is slow and
expensive. docrooter recovers the structure cheaply and locally, labels it against categories
you control, and tells you how sure it is, so the expensive steps only see what they need.

## Results at a glance

Measured on a 51-PDF development corpus and then once, frozen, on a separate 40-PDF held-out
corpus that no tuning step ever saw. The held-out column is the honest estimate; see the
[benchmarks](benchmarks/index.md) for the method, intervals and per-document tables.

| Task | System | Dev | Held-out |
| --- | --- | ---: | ---: |
| TOC extraction, F1 at depth ≤ 2 | default engine (`pymupdf`) | 0.703 | 0.576 |
| Document type, leaf accuracy | Jev on a `DocumentView` (`jev-view`) | 0.961 | 0.925 |
| Document type, leaf accuracy | UltraBERT zero-shot, the local default (`ultrabert-view`) | 0.765 | 0.825 |
| Section normalisation, accuracy | Jev | 0.725 | 0.692 |
| Section normalisation, accuracy | UltraBERT, the local default | 0.514 | 0.579 |

- With 40 documents, one document is 2.5 points; the 95% intervals of document-type accuracy
  are 25-30 points wide for the local models (7.5-17.5 for Jev).
- TOC extraction is where the held-out set exposed real weaknesses (earnings releases,
  bilingual statutes, slide decks). Document typing generalised.
- Loading a PDF with the default engine takes about 1 s per 100 pages on an Apple M1.

## 30-second example

No models, no API key: this uses the dependency-free `keyword` backend, so it runs right after
`pip install docrooter`. It is [`examples/quickstart.py`](../../examples/quickstart.py).

```python
from docrooter import Classifier, Taxonomy, load

MARKDOWN = """\
# Report

## Income tax
The tax rate rose.

## Auditor's report
An audit opinion.
"""

taxonomy = Taxonomy.from_dict(
    {
        "nodes": [
            {
                "id": "financial",
                "label": "Financial",
                "description": "Tax rates, audit opinions and financial results",
                "children": [
                    {"id": "tax", "label": "Tax", "description": "Income tax and the tax rate"},
                    {"id": "audit", "label": "Audit", "description": "The audit opinion"},
                ],
            },
            {"id": "legal", "label": "Legal", "description": "Contracts, courts and disputes"},
        ]
    }
)

doc = load(MARKDOWN)
classifier = Classifier(taxonomy, "keyword", threshold=0.0)  # the default backend is a model
for section in classifier.classify(doc).sections.values():
    print(f"{section.title}: {' > '.join(section.best.labels)}")
```

```text
Income tax: Financial > Tax
Auditor's report: Financial > Audit
```

The `keyword` backend only counts shared words, so it is for trying the API, not for real
documents. Drop the `"keyword"` argument to use the default local model, or pass `"jev"` for the
most accurate backend.

## Where next

- [Installation](getting-started/installation.md): the extras and their licences.
- [Quickstart](getting-started/quickstart.md): a PDF from table of contents to section labels
  in five minutes.
- [Concepts](concepts/index.md): how the pieces fit together.
- [FAQ](faq.md): which backend to pick, PDF engine licences, scanned PDFs, limitations.
