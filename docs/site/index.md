# docrooter

<div class="dr-hero" markdown>
![docrooter logo: a document page whose ruled lines branch into roots](../assets/logo.svg#only-light)
![docrooter logo: a document page whose ruled lines branch into roots](../assets/logo-dark.svg#only-dark)

**The minimum context a hosted LLM needs to classify a document.** docrooter tags a document's
sections first, lets their tags decide the document type or narrow the types a model is
offered, sends only the context level you choose, and counts every token.
</div>

## What it does

- **Sections first.** The level-1 and level-2 sections are tagged against one shared
  [section catalogue](concepts/taxonomies.md) (112 section types in 12 groups). Each of the 19
  document types is defined by the sections it needs, so the tags decide the type with no
  document-level call, or narrow the choice to a few types: a document with a balance sheet and
  a shareholder letter is offered the financial types that need them, not all 19. See
  [Sections first](guides/sections-first.md).
- **A context ladder.** Four named levels, `headings`, `terms`, `lead` and `full`, set how much
  of each section and of the document the model sees; each rung adds one field and its tokens.
  See [The context ladder](concepts/context-ladder.md).
- **Batched calls.** With an LLM, the options are listed once per call, not once per section:
  the sections are sorted into groups in one call, then labelled within their group.
- **Exact token accounting.** Every call's prompt and completion tokens come from the API's
  `usage`, per document, per stage and per section. Backends that make no LLM call report what
  the same prompts would cost, marked estimated.

Hosted LLMs (any OpenAI-compatible API) are the deployment target. Local models are a free
stand-in for testing: the same code path runs against a small model served on your machine
([A free local stand-in](guides/llm-backends.md#a-free-local-stand-in)). The benchmark corpus
is for validation, not training; its accuracy-against-tokens results are pending
([Classification benchmark](benchmarks/classification.md)).

Underneath, docrooter still does what it did before 1.0: it reads PDFs natively and finds their
[table of contents](concepts/toc-engine.md), classifies sections against
[any taxonomy you write](guides/custom-taxonomies.md), and runs zero-shot with small local
embedding models or [trains per-node classifiers](guides/training-evaluation.md) from a few
labelled examples.

**Experimental in 1.0:** [element tags](concepts/element-tags.md) from a tagger you
[train on your own documents](guides/train-your-tagger.md), and
[selection](guides/select-for-extraction.md) with [the completeness dial](concepts/completeness-dial.md)
and `fit-selector`. They work, but sit outside the stability promise and their results are
pending.

## Classify a document

No models and no API key: the `keyword` backend runs right after installing. It is
[`examples/classify_sections_first.py`](../../examples/classify_sections_first.py):

```python
from docrooter import LEVELS, classify, load

doc = load(MARKDOWN)
result = classify(doc, level="terms", backend="keyword")
flags = ", ".join(result.flags) or "none"
print(f"doc type: {result.doc_type} (decided by {result.decided_by}; flags: {flags})")
```

Swap `backend="keyword"` for `"openai:<model>"` to ask a hosted or local LLM; the tokens are
then exact.

## Why it exists

Document pipelines need to know what a document is, and what its parts are, before they can
do anything else with it. Sending the whole document to a large model answers that, but most
of the tokens are wasted: the headings, a few key terms or the first sentences usually say
enough. docrooter sends the least that still decides, lets the section structure do the work
where it can, and tells you exactly what it spent.

## Earlier results at a glance

These come from the document-type and TOC benchmarks of the development versions, before the
v1 flow existed. Measured on a 51-PDF development corpus and then once, frozen, on a separate 40-PDF held-out
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

## Your own taxonomy in 30 seconds

No models, no API key: this uses the dependency-free `keyword` backend, so it runs right after
the core install. It is [`examples/quickstart.py`](../../examples/quickstart.py).

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
- [Sections first](guides/sections-first.md): `docrooter.classify`, step by step.
- [Quickstart](getting-started/quickstart.md): a PDF from table of contents to section labels
  in five minutes.
- [Concepts](concepts/index.md): how the pieces fit together.
- [FAQ](faq.md): which backend to pick, PDF engine licences, scanned PDFs, limitations.
