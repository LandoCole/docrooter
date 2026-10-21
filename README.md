<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img src="docs/assets/logo.svg" alt="docrooter logo: a document page whose ruled lines branch into roots" width="112" height="112">
  </picture>
</p>

<h1 align="center">docrooter</h1>

<p align="center"><strong>The minimum context a hosted LLM needs to classify a document.</strong><br>
Sections first, on a context ladder, in batched calls, with exact token accounting.</p>

<p align="center">
  <a href="https://github.com/LandoCole/docrooter/actions/workflows/ci.yml"><img src="https://github.com/LandoCole/docrooter/actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&amp;logoColor=white" alt="Python 3.10, 3.11, 3.12 and 3.13">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License: Apache-2.0"></a>
</p>

Sending a whole document to an LLM to ask what it is wastes most of the tokens: the headings, a
few key terms or the first sentences usually say enough. docrooter gives a hosted LLM the
minimum context it needs to classify a document and its sections:

- **A context ladder.** Four named levels, `headings`, `terms`, `lead` and `full`, set exactly
  how much of each section and of the document the model sees.
- **Sections first.** The sections are tagged against one shared section catalogue, and the
  document types are defined by the sections they need, so the tags decide the type with no
  document-level call, or narrow the types the model is offered.
- **Batched calls** list the options once per call, not once per section.
- **Exact token accounting.** Every call's tokens come from the API's `usage`; anything else is
  labelled an estimate.

Hosted LLMs, through any OpenAI-compatible API, are the deployment target. Local models are a
free stand-in for testing, on the same code path. The benchmark corpus is for validation, not
training. The core install is small (`pydantic`, `numpy`, `typer`, `rich`); LLM backends, local
models and PDF parsing are extras. docrooter still extracts tables of contents from PDFs and
classifies sections against any taxonomy you supply, as it did before 1.0.

**Experimental in 1.0:** [element tags and the learned tagger](#element-tags-v05-experimental)
(`teach`, `tagger`, `train-elements`, the corpus tooling) and
[selection](#select-what-to-read-v06-experimental) (`select`, the completeness dial,
`fit-selector`). They ship, but outside the stability promise, and their results are pending.

## How it works

`docrooter.classify(doc)` runs the **sections-first** flow:

1. **Tag the sections.** The level-1 and level-2 sections (at most 40) are tagged against
   `builtin_taxonomy("section_catalogue")`: 112 section types in 12 groups (financial
   statements, narrative, disclosures, fund, offering, contract, law, research, standards,
   technical, ESG and general), group first, then leaf.
2. **Match the rules.** Each of the 19 document types in `doc_types` has a `sections` rule
   block, written from its definition: `requires` (groups of section ids, each to be met),
   `typical` (an ordering) and `excludes`.
3. **Decide.** One candidate whose supporting tags are confident is decided by the rules alone,
   with no document-level call (`decided_by="rules"`); with batched calls, or letter answers
   without logprobs, it needs two supporting sections. Several candidates go to the model,
   which chooses among those only (`"model"`). A lone undecided candidate is offered with its
   near misses (and, when still alone, with every type, flag `low_support`); with no
   candidate, near misses are offered, and with nothing to offer the model chooses among every
   type (`"fallback"`, flag `rules_unmet`). The rules narrow or decide only when at least
   `max(2, 20% of the tagged sections)` sections were tagged with confidence; otherwise every
   type is offered (flags `rules_unmet`, `thin_evidence`).

`flow="doc_first"` is the other order: the type first, from the document view, then the
sections with that type's own section taxonomy.

**The context ladder** (`level=`, `terms` by default; `section_level=` and `doc_level=` set the
two steps apart). Each rung shows everything below it and adds one field:

| Level | A section shows | The document shows |
| --- | --- | --- |
| `headings` | heading path and title (32 tokens) | title, counts and contents (320) |
| `terms` | + key phrases (64) | + pooled key phrases (448) |
| `lead` | + first sentences (192) | + opening text (640) |
| `full` | the whole section text (2,048) | the whole body text, budgeted to a 16,000-token target prompt |

**Calls.** With an LLM backend, `calls="batched"` (the default) sorts the sections into groups
in one call and labels them within each group in one more call per group, listing each option
once; `calls="batched_flat"` lists every leaf in one call; `calls="per_item"` asks one item at a
time, with probabilities (and allows `cascade=True`, which asks low-confidence items again one
rung up). Every call's prompt is budgeted to `max_call_tokens` (16,000 by default, on
estimated tokens, packed to 85% of it for the estimate's error): batched sections are packed by
size as well as count (at most 40 per call), a view too large for any call is cut and flagged
`truncated`, and the document view is cut to fit beside its call's options.

**Tokens.** `result.tokens` has the `prompt`, `completion` and `calls` spent, with `exact` true
only when every call reported `usage`; `result.stage_tokens` splits it into sections and
document, `result.section_calls` and `result.document_calls` record each call, and every
section tag carries its share (an estimate when a call carried several sections). `log=path` appends one JSON line per decision
(level, view hash, options, probabilities, model, endpoint host, prompt hash, tokens, and a
`trainable` flag that is always false for Jev); API keys never reach it.

See [Sections first](docs/site/guides/sections-first.md) and
[The context ladder](docs/site/concepts/context-ladder.md).

## Benchmark: accuracy against tokens

**Results pending.** The classification benchmark measures how much accuracy each rung of the
ladder buys, and at what token cost, against sending the whole document: doc-type accuracy and
macro-F1, section macro- and micro-F1, the share decided by rules alone and prompt tokens per
document, for both flows, every level, and the keyword, UltraBERT and local-LLM backends, on the
113 public documents of the gold split (951 Claude-annotated sections, evaluation only; the
annotators were all Claude, the same underlying model, so κ measures consistency, not
correctness). The doc-type rules were revised once after a reviewer checked them against the
section gold, so the gold split's doc-type numbers will be development numbers; the held-out run
is the clean measure. It will
also set the measured default level per backend; until then the default is `terms`. The report
is [docs/benchmarks-classification.md](docs/benchmarks-classification.md), generated from the
results files; its numbers will be quoted from there, never written by hand.

### Earlier results (TOC and document type, before v1)

Measured on a 51-PDF development corpus, which the defaults were chosen on, and then once, at a
frozen commit, on a separate 40-PDF held-out corpus that no tuning step saw (no document or
publisher in common). The held-out column is the honest estimate.

| Task (metric) | System | Dev, 51 PDFs | Held-out, 40 PDFs |
| --- | --- | ---: | ---: |
| Document type, 18 types (leaf accuracy) | `jev-view`: TypeSafe Jev on docrooter's `DocumentView` | 0.961 | 0.925 |
| | `jev-chunks`: Jev on raw page text, no docrooter view | 0.922 | 0.975 |
| | `ultrabert-view`: UltraBERT zero-shot, the local default | 0.765 | 0.825 |
| | `ultrabert-hybrid`: UltraBERT on the view and the sections | 0.824 | 0.775 |
| | `bge-view`: bge-small zero-shot | 0.608 | 0.700 |
| TOC extraction (F1 at depth ≤ 2 against PDF bookmarks) | default engine (`pymupdf`) | 0.703 (dev-tuned) | 0.576 |

- *18 types*: both corpora were labelled with the 18 types that existed then. v0.5 adds `financial.fact_sheet`
  (19 types), so the document-type numbers predate it, and the held-out corpus has not been run
  again.
- *Dev-tuned*: the heading rules were tuned on the dev documents (the 24 with at least 5
  bookmarks are scored; 23 on held-out), so 0.703 is optimistic; it was 0.478 before tuning. No dev-tuned
  document-type system is in the table (the dev-tuned view, `ultrabert-view-tuned`, scored 0.843
  on dev and was not part of the held-out run).
- With 40 documents one document is 2.5 points, and the 95% intervals of leaf accuracy are
  25-30 points wide for the local models (7.5-17.5 for Jev).
- **Public datasets** (section classification, hierarchical F1): zero-shot UltraBERT-Embed scores
  0.752 on SEC 10-K items, 0.381 on LEDGAR clauses and 0.172 on DBpedia classes; with 16
  labelled examples per class, trained heads reach 0.888, 0.719 and 0.922.

Details: [held-out evaluation](docs/benchmarks-heldout.md), [PDF benchmarks on the dev
corpus](docs/benchmarks-pdf.md), [public datasets](docs/benchmarks.md) (with the decisions they
led to) and [performance and memory](docs/performance.md).

## Install

docrooter 1.0 installs from GitHub (it is not on PyPI yet), or from the wheel attached to the
[GitHub release](https://github.com/LandoCole/docrooter/releases):

```bash
pip install "git+https://github.com/LandoCole/docrooter@v1.0.0"                        # core
pip install "docrooter[llm] @ git+https://github.com/LandoCole/docrooter@v1.0.0"       # + LLM backend
pip install "docrooter[models,pdf] @ git+https://github.com/LandoCole/docrooter@v1.0.0" # + local models, PDFs
```

The rest of this README writes extras as `pip install 'docrooter[models]'`; use the form above
with the same extras until docrooter is on PyPI.

| Extra | Adds | Needed for |
| --- | --- | --- |
| *(none)* | `pydantic`, `numpy`, `typer`, `rich` | Markdown/text loading, taxonomies, the `keyword` backend, the CLI |
| `models` | `torch`, `transformers>=5`, `sentence-transformers>=6`, `scikit-learn`, `joblib`, `datasets` | `embedding`, `nli`, `mlm-choice` and `trained` backends; `docrooter train`; `docrooter models` |
| `pdf` | `pymupdf4llm` (which installs PyMuPDF) | Loading `.pdf` files with PyMuPDF, the default engine (AGPL; see the licence note below) |
| `pdf-mit` | `pdfplumber` | Loading `.pdf` files with the MIT-licensed pdfplumber engine |
| `llm` | `httpx` | The `openai` backend (any OpenAI-compatible server) |
| `jev` | `typesafe-sdk` | The `jev` backend (TypeSafe Jev) |
| `ui` | `fastapi`, `uvicorn`, `python-multipart` | `docrooter serve`, the local [playground](#playground) |
| `ocr` | `rapidocr`, `onnxruntime`, `pypdfium2` | The `rapidocr` engine for scanned pages (element mode) |
| `ocr-mac` | `ocrmac` (macOS only) | The `ocrmac` engine (Apple Vision) |
| `layout` | `onnxruntime`, `pypdfium2`, `huggingface_hub` | The heron layout model of the `accurate` profile (no torch) |
| `teach-mlx` | `mlx-lm` (Apple silicon, Python 3.11+) | The default teacher (Qwen3-4B-Instruct-2507, Apache-2.0, on MLX) for [training your own element tagger](#element-tags-v05-experimental) (experimental); not in `all` |
| `bench` | `datasets`, `pandas` | The benchmarks, `DocumentResult.to_dataframe()` |
| `all` | `models` + `pdf` + `pdf-mit` + `llm` + `jev` + `ui` + `ocr` + `ocr-mac` + `layout` | Everything above except `bench` |

Python 3.10 or newer. Asking for something an extra provides without installing it fails with
the exact command to run, in the GitHub form, e.g.
`pip install "docrooter[models] @ git+https://github.com/LandoCole/docrooter"`.

**PDF engines and the AGPL.** The default PDF engine is PyMuPDF, which is licensed under the
AGPL-3.0, unlike docrooter itself (Apache-2.0). It comes with the `[pdf]` extra (and so with
`[all]`) and is imported only when a `.pdf` file is loaded. Read the AGPL terms before you
distribute software or run a network service that includes it. For a permissive stack, install
`[pdf-mit]` instead: it adds `pdfplumber` (MIT), which `load(path)` uses automatically when
PyMuPDF is not installed, or explicitly with `engine="pdfplumber"`. The `pymupdf4llm` engine is
the legacy v0.1 Markdown path, kept for comparison (`engine="pymupdf4llm"`, AGPL).

**`trust_remote_code` and UltraBERT.** The default model, `dmis-lab/UltraBERT-Embed-150M`
(and `UltraBERT-380M`, used by `mlm-choice`), ships its own modelling code, which
transformers only runs with `trust_remote_code=True`. docrooter turns that on only for models
whose built-in recipe pins a commit (`367d5343b5a0164d57b810cedf5de4b8983d5984` for
UltraBERT-Embed-150M, `65bbbfeefb03bc93b9965d1d511708dabcee8511` for UltraBERT-380M), so the
code that runs is the code at that commit and not whatever the repository holds later. For any
other model that needs remote code you have to opt in yourself, after reading the code:
`EmbeddingBackend("org/model", trust_remote_code=True)`. The first run downloads about 600 MB
for the default model.

## Quickstart

`docrooter.classify` with the dependency-free `keyword` backend runs straight after the core
install. The full script is
[`examples/classify_sections_first.py`](examples/classify_sections_first.py):

```python
from docrooter import LEVELS, classify, load

doc = load(MARKDOWN)
result = classify(doc, level="terms", backend="keyword")
flags = ", ".join(result.flags) or "none"
print(f"doc type: {result.doc_type} (decided by {result.decided_by}; flags: {flags})")
for tag in result.sections:
    print(f"  {tag.title}: {tag.label}")
kind = "exact" if result.tokens.exact else "estimated"
print(f"tokens ({kind}): {result.tokens.prompt:,} prompt over {result.tokens.calls} calls")
```

The keyword backend only counts shared words: it shows the API and the token accounting, not
the accuracy, and its tokens are estimates of what an LLM would be sent. For a real run, point
the OpenAI-compatible backend at a hosted model, or at a free local stand-in:

```bash
docrooter classify report.pdf -b openai:gpt-4o-mini --level terms --log decisions.jsonl
export DOCROOTER_LLM_BASE_URL=http://127.0.0.1:18080/v1   # mlx_lm.server with Qwen3-4B-Instruct-2507
docrooter classify report.pdf -b "openai:$MODEL" --json
```

The stand-in recipe (the pinned Qwen3-4B-Instruct-2507 4-bit snapshot served by `mlx_lm.server`
on port 18080) is in
[the LLM guide](docs/site/guides/llm-backends.md#a-free-local-stand-in).

### Your own taxonomy

Sections can also be classified against a taxonomy you write. This example lives in
[`examples/quickstart.py`](examples/quickstart.py).

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

`load` takes a path (`.md`, `.markdown`, `.txt`, `.pdf`) or Markdown text. `doc.toc()` lists the
headings with their depth, and `classify` returns a `DocumentResult` with a `best` path (node
ids, labels, score, and the decision made at each step) and the rival `alternatives` for every
section. `result.to_json()` serialises it and `result.summary()` gives the share of the document
in each top-level category.

Leave the backend out to use the default, UltraBERT embeddings
([`examples/embedding_quickstart.py`](examples/embedding_quickstart.py), needs `docrooter[models]`):

```python
from docrooter import Beam, Classifier

classifier = Classifier(taxonomy)  # zero-shot
classifier = Classifier(taxonomy, strategy=Beam(3), threshold=0.35)
```

The first line uses `"embedding:dmis-lab/UltraBERT-Embed-150M"` and the second spells out the
other defaults.

`Greedy()` follows the best child at each level, `Beam(n)` keeps the `n` best paths and `Flat()`
scores every leaf in a single call, by its full path. A path stops early, with a `stopped_reason`, when no child reaches the
threshold.

## Document types, section labels and PDFs (v0.2)

Two built-in taxonomies ship with the package: `doc_types` (19 document types in 6 groups) and
one section taxonomy per family (`list_builtin_taxonomies()` lists them; `builtin_taxonomy(name)`
returns a fresh copy). `classify_document` labels a whole document from a `DocumentView` (its
title, page and section counts, headings, opening text and keyphrases, within a token budget),
and `normalize_sections` maps each heading to a canonical section of its type. This runs with
the `keyword` backend; it lives in [`examples/doc_types.py`](examples/doc_types.py):

```python
from docrooter import DocumentView, builtin_taxonomy, classify_document, load, normalize_sections

doc = load(path, engine="pdfplumber") if path else load(MARKDOWN)
doc_types = builtin_taxonomy("doc_types")
view = DocumentView(fields=("title", "toc", "lead"), max_tokens=512)
result = classify_document(doc, doc_types, backend="keyword", view=view)
print(" > ".join(result.best.labels), f"{result.confidence:.2f}")
doc_type = result.best.node_ids[-1]
norm = normalize_sections(doc, doc_type=doc_type, backend="keyword", levels=[2])
for section in norm.sections.values():
    print(f"{section.title}: {section.canonical}")
```

```text
Research > Research paper 0.11
Abstract: abstract
Methods: method
Results: results
References: references
```

- **PDF engines.** `load(path, engine=...)` (or `load_pdf`) reads PDFs natively, without OCR:
  `"auto"` (the default: PyMuPDF if installed, else pdfplumber), `"pymupdf"` (extra `pdf`),
  `"pdfplumber"` (extra `pdf-mit`), `"both"` (PyMuPDF headings that pdfplumber agrees with) or
  `"pymupdf4llm"` (the legacy v0.1 path). `doc.meta` records the engine, page count and timings.
- **`classify_document`** uses `Flat()` and threshold 0 by default, so a document with text always
  gets a label; gate on `result.confidence`. For many documents, build one classifier and call
  `Classifier(taxonomy, backend).classify_documents(docs, threshold=0.0)`, which gives the same
  results in one batch. `mode="sections"` or `"hybrid"` also scores the section headings.
- **`normalize_sections(doc)`** without `doc_type` classifies the document type first, with the
  same backend.

The `doctype` and `normalize` commands do the same from the shell.

### Gating a pipeline

`--keep` filters by type id (a group such as `financial` keeps all its leaves, and it can be
repeated) and `--paths-only` prints just the kept files, so the document type decides what the
next step sees. For example, to keep the shareholder letters in a folder of filings and extract
their tables of contents:

```bash
docrooter doctype filings/ --keep financial.shareholder_letter --paths-only > letters.txt
while read -r f; do docrooter toc "$f" --json > "${f%.*}.toc.json"; done < letters.txt
```

This uses the default local model (`docrooter[models]`, and `docrooter[pdf]` for PDFs); add
`-b keyword` to try it without one, or `-b jev` for the most accurate backend. With
`--threshold T`, a file whose confidence is below `T` gets no type and is never kept.

## Elements and profiles (v0.4)

`process()` is one pipeline with three profiles. `fast` is the v0.3 behaviour, byte for byte.
`balanced` (the default) also builds **elements**: the paragraphs, list items, tables, figures,
captions, footnotes and page furniture of a document, in reading order (two-column pages read
column by column), each with its page box and its section. It OCRs scanned pages when an engine
is installed, and writes summaries. `accurate` swaps the heuristic rules for the heron layout
model (extra `layout`). Then `chunk()` cuts the elements into section-aware chunks for retrieval.
This runs with the keyword backend and no OCR engine; it lives in
[`examples/elements_and_chunks.py`](examples/elements_and_chunks.py):

```python
from docrooter import chunk, process

analysis = process(path, profile="balanced", ocr="none", doc_type="keyword", sections="keyword")
doc = analysis.document
for element in doc.elements:
    label = element.text[:36].replace("\n", " ")
    print(f"{element.order:>2} p{element.boxes[0].page} {element.kind:<12}{label}")
chunks = chunk(analysis, max_tokens=100)
print(chunks[2].text)  # a table is never split: a description line, then its rows
```

```text
 0 p1 title       Annual Report 2025
 1 p1 heading     1. Business
 2 p1 paragraph   Our sensors measure pressure, temper
 3 p1 paragraph   Competition remains strong, and our 
 4 p1 heading     2. Financial results
 5 p1 table       Segment | 2024 | 2025 Sensors | 412 
 6 p1 caption     Table 1. Revenue by segment (million
 7 p1 page_number Page 1
 8 p2 heading     3. Outlook
 9 p2 paragraph   We expect revenue to keep growing as
10 p2 page_number Page 2
2. Financial results

Table: Table 1. Revenue by segment (millions); 4×3
|  |  |  |
| --- | --- | --- |
| Segment | 2024 | 2025 |
| Sensors | 412 | 468 |
| Controls | 198 | 215 |
| Services | 96 | 121 |
```

- **Heuristic or heron.** On 400 DocLayNet test pages the heuristic layout scores a macro F1 of
  0.257 against heron's 0.736, and its text coverage for body text is 0.927. `balanced` is fast
  with good text coverage; `accurate` is for structure quality.
- **Speed**, per 100 pages: `fast` 0.99 s, `balanced` 4.64 s (PyMuPDF), `accurate` 64.6 s.
  `tables="none"` skips the search for ruled tables and is the speed switch.
- **Page furniture.** In element mode, running headers, footers and page numbers move to the start
  or end of their page, by design.
- **OCR.** `ocr="auto"` picks Tesseract (if the binary is on PATH), then Apple Vision (macOS),
  then RapidOCR: tesseract first for accuracy (character error rate 0.058, against 0.089 for
  RapidOCR and 0.102 for Apple Vision), Apple Vision before RapidOCR for speed and memory (0.73
  against 3.04 s per page, 1.2 against 3.2 GiB).

From the shell: `docrooter elements report.pdf`, `docrooter process report.pdf --profile
accurate --chunks --out analysis.json` and `docrooter config show --profile balanced`. See the
[element benchmarks](docs/benchmarks-elements.md) for the numbers and their commits.

## Element tags (v0.5, experimental)

> **Experimental.** The learned element tagger (`docrooter.teach`, `docrooter.tagger`,
> `docrooter.corpus`, `train-elements`) ships in 1.0 outside the stability promise; its results
> are pending.

[Elements](#elements-and-profiles-v04) say what a block is; **element tags** say what it is
about (its **topic**, one leaf of a tree per document family, such as
`fund_document.fees_and_expenses`) and what it does (its **functions**, any of twelve such as
`risk_disclosure`, `obligation` or `legal_boilerplate`). Every paragraph, list item, table,
caption, footnote and formula can carry both in `Element.tags`, so an application can read only
the elements it needs, or gate and route by tag. Tags are off by default (`element_tags="none"`
in every profile) and a v0.4 document serialises as before.

The tagger is a BERT-class student that you train on your own documents, labelled by a local
teacher LLM, so no model weights are published. `docrooter train-elements` runs the whole chain
and resumes after an interruption:

```bash
docrooter corpus build corpus/ --from-manifest manifest.jsonl --pdf-dir pdfs/
docrooter train-elements corpus/ --stage A --out bundles/tagger-a --teacher mlx:mlx-community/Qwen3-4B-Instruct-2507-4bit@50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b
docrooter tag report.pdf --bundle bundles/tagger-a
```

- **Teachers and terms.** The default teacher is Qwen3-4B-Instruct-2507 (Apache-2.0) on MLX.
  **Jev cannot be a teacher**: TypeSafe's MCA §2.3(b) forbids distillation, and docrooter refuses
  it. A teacher you bring (`hf:` or an OpenAI-compatible `openai:` endpoint) is under its
  provider's terms. CUAD contracts (CC BY 4.0) need attribution wherever you use them; the
  [guide](docs/site/guides/train-your-tagger.md#licensing-and-terms) has the citation.
- **Our corpus.** 1,100 public native-PDF documents with 81,822 pages (925 train, 62 calibration,
  113 gold), as a manifest and fetch scripts under `benchmarks/training_corpus/`; no PDF is
  committed. EDGAR could not be used (the SEC refuses the project's User-Agent), so the annual
  reports come from issuers' own investor-relations sites, mostly non-US.
- **Disclosures.** The 900-element gold set is Claude-annotated, and all annotators share one
  underlying model, so the topic kappa of 0.953 (180 double-annotated elements) measures
  consistency, not human-verified correctness. Performance depends on corpus size; the learning
  curve is on the benchmark page.
- **Results** are pending: [docs/benchmarks-element-tags.md](docs/benchmarks-element-tags.md).

See [Element tags](docs/site/concepts/element-tags.md) for the taxonomies and
[Train your own element tagger](docs/site/guides/train-your-tagger.md) for the steps;
[`examples/train_element_tagger.py`](examples/train_element_tagger.py) runs the stages on a tiny
corpus with no model.

## Select what to read (v0.6, experimental)

> **Experimental.** Selection (`select`, the completeness dial, `fit-selector`) ships in 1.0
> outside the stability promise; its results are pending.

A downstream LLM rarely needs the whole document. `select()` returns the elements to read for a
**field** (the risk factors of an annual report, a fund's fees) or a free-text query, at a
**completeness** `c` from 0 to 1: `c = 1.0` is the whole document, and a lower `c` sends fewer
tokens. docrooter selects; your application extracts.

```python
from docrooter import builtin_fields, process, select
from docrooter.select.selection import readout

risk = {f.id: f for f in builtin_fields("annual_report")}["annual_report.risk_factors"]
analysis = process(
    "report.pdf", profile="accurate", element_tags="bundles/tagger-a", escalate="none"
)
selection = select(analysis, risk, completeness=0.9, selector="bundles/tagger-a")
print(readout(selection))
print("\n\n".join(chunk.text for chunk in selection.chunks))
```

- **What `c` promises.** On documents like its calibration set, the *average* share of relevant
  elements kept is at least `c` (conformal risk control), averaged over documents **and over the
  fields of the family**. It is not a promise about any one document or any one field, and
  "relevant" is what the **teacher** model labelled, not a human.
- **Only for what was calibrated.** The guarantee holds for **one field** at a time, on a
  document **read and tagged as the selector was fitted**: the same profile (`accurate`, the
  corpus build's) with its layout engine and sections backend and escalation off, the bundle's
  own tagger and the selector's embedder. Anything else is reported
  in `fit_mismatch` (for example `["layout"]` for the default `balanced` profile) and gives an
  estimate.
- **When it promises nothing.** Below `c = 1.0`, free-text queries, a list of several fields
  (call `select()` once per field), a missing tagger or embedder, a tagger, embedder or reading
  other than the fit's, a document type that does not match the field's family, a token budget,
  and a field the dial was not calibrated on (your own, until you refit) give an estimate only
  (`guarantee=False`). Where the calibration set is too small for `c` (only above `n/(n+1)`),
  the whole document comes back (`capped=True`). `c = 1.0` is always the whole document.
- **Your own corpus.** A selector is fitted, not shipped: `docrooter teach label --full`, then
  `docrooter fit-selector`. Without one only `c = 1.0` works.
- **Results** are pending: [docs/benchmarks-selection.md](docs/benchmarks-selection.md).

See [Select what to send to an LLM](docs/site/guides/select-for-extraction.md) and
[The completeness dial](docs/site/concepts/completeness-dial.md);
[`examples/select_for_extraction.py`](examples/select_for_extraction.py) runs `select()` with no
model. `docrooter select report.pdf --need annual_report.risk_factors --profile accurate --tagger
BUNDLE` does it from the shell, and `docrooter serve --selector` adds the dial to the playground.

## Command line

```bash
docrooter classify report.md -b keyword                  # type and section tags, sections first
docrooter classify report.pdf --level headings --json    # the cheapest rung, as JSON
docrooter classify report.pdf --flow doc_first --calls per_item --log decisions.jsonl

docrooter toc report.md                                  # indented table of contents
docrooter toc report.pdf --json                          # as JSON (needs docrooter[pdf])
docrooter --version                                      # the installed version

docrooter taxonomy validate taxonomy.csv                 # check a taxonomy file
docrooter taxonomy validate paths.csv --sep ">"          # path-style CSV with another separator

docrooter classify report.md -t taxonomy.json            # JSON on stdout, default backend
docrooter classify report.md -t taxonomy.json -b keyword --strategy greedy
docrooter classify report.md -t taxonomy.json -b jev -o result.json   # table + result.json

docrooter doctype docs/ -b keyword                       # document type per file (builtin:doc_types)
docrooter doctype docs/ --keep financial --paths-only    # only financial docs, one path per line
docrooter doctype docs/ --jsonl types.jsonl --workers 4  # parse PDFs in 4 processes
docrooter doctype big/ --chunk-size 16 --max-pages 200   # bound memory on large corpora
docrooter normalize report.md                            # canonical label for each section title
docrooter normalize report.md --doc-type research.research_paper --json

docrooter elements report.pdf --kinds heading,table     # typed elements in reading order
docrooter process report.pdf --profile accurate --chunks -o analysis.json   # the rooter
docrooter config show --profile balanced                 # the resolved configuration, as TOML

docrooter corpus build corpus/ --from-manifest manifest.jsonl --pdf-dir pdfs/   # elements into a store
docrooter train-elements corpus/ --out bundle/ --teacher mlx:<repo>@<rev>       # label, train, calibrate
docrooter tag report.pdf --bundle bundle/                # topic and function tag per element
docrooter tag-corpus corpus/ --bundle bundle/ --split train --split calib
docrooter select report.pdf --need annual_report.risk_factors --tagger bundle/   # experimental: what to read
docrooter teach label corpus/ --full --splits calib      # the teacher labels every calibration element
docrooter fit-selector corpus/ --bundle bundle/ --out bundle/selector.json   # fit the dial

docrooter eval -t taxonomy.json -d labelled.jsonl -b keyword          # accuracy, hierarchical F1
docrooter eval -t taxonomy.json -d labelled.jsonl --format md

docrooter train -t taxonomy.json -d labelled.jsonl -o bundle/         # train heads and fine-tune
docrooter eval  -t taxonomy.json -d labelled.jsonl -b trained:bundle/ # then evaluate the bundle

docrooter models pull dmis-lab/UltraBERT-Embed-150M      # download for offline use
docrooter models list                                    # what is in the Hugging Face cache
```

`-b` takes a backend spec (see [Backends](#backends)); `--strategy` takes `greedy`, `beam:N` or
`flat`. Errors are printed as `error: ...` with exit status 1.

## Playground

```bash
pip install 'docrooter[ui]'
docrooter serve                    # http://127.0.0.1:8765
docrooter serve -b keyword --open  # pick the preselected backend, open the browser
```

![The docrooter playground after a run: the pipeline stages as a timed waterfall, then the
table of contents](docs/assets/playground.png)

*The Federal Reserve's July 2024 Monetary Policy Report (`doc_045` of the dev corpus, 79 pages)
with the `keyword` backend: 1.53 s end to end, most of it reading the pages. The keyword backend
needs no model and only checks the pipeline; it calls this report a proxy statement.*

`docrooter serve` starts a local web page. Drop in a PDF, Markdown or text file, choose a backend,
and watch the pipeline run stage by stage (reading pages, finding headings, loading the model,
classifying) with the time each stage takes. The result shows the table of contents as a tree
with each heading's score, the document type with the probabilities of the five likeliest
types, and, optionally, a canonical label for every section. Recent runs stay in the
sidebar while the server runs.

It is a local tool: it binds `127.0.0.1` by default, rejects requests whose `Host` is not that
address, loads no external assets, caps uploads at 50 MB and deletes each upload as soon as it has
been processed. `--env-file .env` reads missing variables such as `TYPESAFE_API_KEY` for the
`jev` backend. The model-based backends obey the same memory budget as the library: set
`DOCROOTER_MEMORY_BUDGET_GB` (8 by default; see [Memory](#memory)) before starting it. See
[SECURITY.md](SECURITY.md) for the guard rails before you expose it any further.

## Taxonomies

A taxonomy is a tree of categories, each with a label and optionally a description and
examples that zero-shot backends use. It can be written as nested JSON, flat JSON, or one of
three CSV layouts, and `Taxonomy.load` detects which. All five below describe the same two-level
taxonomy (shortened here); [docs/taxonomy-formats.md](docs/taxonomy-formats.md) has the full reference
(optional columns, per-node `threshold`, `multi_label` and `backend`, root settings, the
validation codes).

Nested JSON:

```json
{"nodes": [
  {"id": "financial", "label": "Financial", "description": "Money, accounting, results",
   "children": [
     {"id": "financial.tax", "label": "Tax", "description": "Income tax and deferred tax"},
     {"id": "financial.audit", "label": "Audit", "description": "Audit reports and findings"}
   ]},
  {"id": "legal", "label": "Legal", "description": "Legal matters"}
]}
```

Flat JSON (nodes linked by `parent_id`):

```json
{"nodes": [
  {"id": "financial", "label": "Financial"},
  {"id": "financial.tax", "label": "Tax", "parent_id": "financial"},
  {"id": "financial.audit", "label": "Audit", "parent_id": "financial"},
  {"id": "legal", "label": "Legal"}
]}
```

Adjacency CSV (`parent_id` column):

```csv
id,parent_id,label,description
financial,,Financial,"Money, accounting, results"
financial.tax,financial,Tax,Income tax and deferred tax
financial.audit,financial,Audit,Audit reports and findings
legal,,Legal,Legal matters
```

Levels CSV (`level_1`, `level_2`, ... columns):

```csv
level_1,level_2,description
Financial,,"Money, accounting, results"
Financial,Tax,Income tax and deferred tax
Financial,Audit,Audit reports and findings
Legal,,Legal matters
```

Path CSV (labels joined by `>`):

```csv
path,description
Financial,"Money, accounting, results"
Financial > Tax,Income tax and deferred tax
Financial > Audit,Audit reports and findings
Legal,Legal matters
```

Run `docrooter taxonomy validate FILE` before using a new file; it names the row or JSON pointer
of every problem.

## Training and evaluation

Zero-shot is a good start. If you have labelled sections, docrooter fits one small classifier
head per taxonomy node on top of a shared embedding model, and can fine-tune that model first.
Nodes that do not have enough examples fall back to the zero-shot model. The full script is
[`examples/train_and_eval.py`](examples/train_and_eval.py) (needs `docrooter[models]`).

**1. Labelled data.** A dataset is JSONL (`{"text": ..., "path": [...]}` per line) or CSV (`text`
and `path` columns, the path joined by `>`). A path lists node ids or labels from the top down.

```jsonl
{"text": "The effective tax rate was 21.4 percent.", "path": ["Financial", "Tax"]}
{"text": "The auditor issued an unqualified opinion.", "path": ["financial", "financial.audit"]}
```

**2. Bootstrap labels from a zero-shot run.** Classify some documents, keep the rendered text of
each section with its best path, then fix the mistakes by hand and train on the result:

```python
from docrooter import Classifier, Dataset

classifier = Classifier(taxonomy)  # zero-shot
result = classifier.classify(doc)
pseudo = Dataset.from_results(result, doc, classifier.view)
pseudo.save("pseudo.jsonl")  # review and correct, then load it back
```

**3. Split, train, save.**

```python
from docrooter import Dataset, train

dataset = Dataset.load("labelled.jsonl")
train_set, test_set = dataset.split(test_size=0.2)  # stratified by top-level category
model = train(taxonomy, train_set, finetune_body=False, min_examples_per_child=2)
model.save("bundle/")
```

This fits only the heads, which is fast. Leave out `finetune_body=False` to fine-tune the body
first (the default, and usually better with enough data). The CLI equivalent is
`docrooter train -t taxonomy.json -d labelled.jsonl -o bundle/ [--no-finetune-body]`. A node gets a
head only when each of its children has at least `min_examples_per_child` examples (default 4).

**4. Evaluate against zero-shot.**

```python
from docrooter import Classifier, TrainedModel, compare, evaluate

trained = Classifier(taxonomy, TrainedModel.load("bundle/"))  # or backend "trained:bundle/"
compare({"zero-shot": Classifier(taxonomy), "trained": trained}, test_set).print()
evaluate(trained, test_set).print()  # adds per-node accuracy and sibling confusions
```

Reports give exact-path accuracy, hierarchical precision/recall/F1, leaf macro-F1, the share of
sections that stopped early, selective accuracy and throughput. `docrooter eval` does the same
from the command line.

**5. Calibrate (optional).** `calibrate` returns a copy of the classifier whose backend
temperatures are tuned on a development set (hold one out of the training data), so the
probabilities, and with them the `threshold`, mean what they say.

```python
from docrooter import calibrate

calibrated = calibrate(trained, dev_set)
print(calibrated.calibration_report)
```

> **Security: trained bundles contain pickles.** A bundle stores its heads as `joblib` files,
> which are Python pickles, and loading a pickle can run arbitrary code. Only load bundles you
> trained yourself or got from a source you trust. The same goes for `trained:<path>` specs and
> `docrooter eval -b trained:...`.

## Backends

A backend is chosen by a spec string, `kind[:model[@revision]]`, or by passing an instance. The
classifier can use different backends at different nodes or depths (`overrides=`), per-node
`backend` settings in the taxonomy, and a `fallback`.

| Spec | How it scores | Needs | Notes |
| --- | --- | --- | --- |
| `keyword` | Word overlap with the label, description and examples | core | A baseline for tests and demos, not for real use |
| `embedding[:model]` | Embedding similarity | `models` | The default, `dmis-lab/UltraBERT-Embed-150M`; any sentence-embedding model works, e.g. `embedding:BAAI/bge-small-en-v1.5` |
| `nli[:model]` | Textual entailment | `models` | Default `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` |
| `mlm-choice[:model]` | A masked LM fills in the letter of the right option | `models` | Experimental; default `dmis-lab/UltraBERT-380M` |
| `trained:<dir>` | Per-node heads on a fine-tuned body | `models` | From `docrooter train` / `train()`; see the pickle warning above |
| `openai:<model>` | Chat model answers with an option letter (A-Z) | `llm` | Any OpenAI-compatible endpoint, e.g. `openai:gpt-4o-mini`; the model is required |
| `jev[:model]` | TypeSafe Jev typed choice and yes/no probabilities | `jev` | Default `jev-1.13.0`; needs `TYPESAFE_API_KEY` |

Model-backed backends pick the device in the order CUDA, MPS, CPU. FlashAttention 2 is used only on
CUDA when it is installed; otherwise attention runs through `sdpa`.

### Custom backends

Any object with these members can be passed as a backend (or registered as a spec kind with
`docrooter.backends.register_backend_kind`); there is nothing to subclass:

- `name`: a string that identifies the backend, e.g. in results and calibrated temperatures.
- `max_candidates`: the most candidates one `score` call accepts, or `None` for no limit. A
  decision with more candidates goes to the next backend in the routing order.
- `temperature` (optional, default 1.0): applied to `similarity` scores and used to re-temper
  `softmax` scores. `calibrate()` tunes it per backend `name`.
- `score(texts, candidates, *, instruction=None[, multi_label])`: returns `Scores(values, kind)`
  with `values` of shape `(len(texts), len(candidates))`. `kind` is `"softmax"` (rows are
  probabilities), `"independent"` (one probability per candidate) or `"similarity"` (unbounded
  scores). Each candidate has `node_id`, `text` and `examples`. `multi_label=True` is passed only
  for decisions at multi-label nodes, so a backend without that argument works everywhere else.
- `supports(node_id)`: whether the backend can decide at this node.
- `tokenizer()`: a function that counts the tokens of a text, or `None` to use an estimate.

```python
from docrooter.backends import Scores


class OverlapBackend:
    name = "overlap"  # keys calibrated temperatures; keep it unique per backend
    max_candidates = None  # or a limit: larger candidate sets go to the next backend

    def score(self, texts, candidates, *, instruction=None):
        rows = []
        for text in texts:
            words = set(text.lower().split())
            rows.append([len(words & set(c.text.lower().split())) for c in candidates])
        return Scores(np.array(rows, dtype=float), "similarity")

    def supports(self, node_id):
        return True

    def tokenizer(self):
        return None  # token counts are estimated


classifier = Classifier(taxonomy, OverlapBackend(), threshold=0.0)
```

The full script is [`examples/custom_backend.py`](examples/custom_backend.py).

### API keys

The `jev` backend reads `TYPESAFE_API_KEY`. The `openai` backend reads `DOCROOTER_LLM_API_KEY`
or `OPENAI_API_KEY`, and the server address from `DOCROOTER_LLM_BASE_URL` or `OPENAI_BASE_URL`
(the OpenAI API when unset; a local server that needs no key works without one). Keys are only
read from the environment (or an explicit `api_key=`) and are redacted from error messages.

docrooter does not read `.env` files itself. Keep keys in a git-ignored `.env` and hand them to
the process:

```bash
# .env  (never commit this file)
TYPESAFE_API_KEY=...

uv run --env-file .env python examples/jev_backend.py
# or: set -a; source .env; set +a
```

Never put a key in code, a taxonomy file or a committed config. See
[`examples/jev_backend.py`](examples/jev_backend.py).

### Offline use

Models come from the Hugging Face cache. Download them while you have a connection, then forbid
network access:

```bash
docrooter models pull dmis-lab/UltraBERT-Embed-150M     # the pinned revision, unless --revision
HF_HUB_OFFLINE=1 docrooter classify report.md -t taxonomy.json
```

A trained bundle is self-contained (the fine-tuned body is inside it) and loads offline.

### Memory

docrooter keeps its models within a memory budget, **8 GiB by default**. Set it, and the thread
count, before the first model loads, with `DOCROOTER_MEMORY_BUDGET_GB` and `DOCROOTER_THREADS`
or in code; release models when you are done:

```python
import docrooter

# or set DOCROOTER_MEMORY_BUDGET_GB=8 and DOCROOTER_THREADS=4
docrooter.configure(memory_budget_gb=8, threads=4)

with docrooter.Classifier(taxonomy) as clf:  # closes the models it loaded
    results = clf.classify(doc)
```

On Apple silicon, Metal gets a hard cap of the budget minus 1.5 GiB, and a model that still runs
out of MPS memory moves to the CPU once, with a warning. On the CPU the budget only scales the
batch size. The cap is process-wide and applies whether or not you call `configure()`: the first
model load sets `PYTORCH_MPS_HIGH_WATERMARK_RATIO` and `PYTORCH_MPS_LOW_WATERMARK_RATIO` in
`os.environ`, so subprocesses you start afterwards inherit them, and caps Metal memory for the
process, your own MPS models included. To opt out, set either variable yourself before the first
model load (your values win; docrooter then sets neither and applies no fraction), or raise the
budget. Every backend has `close()`; a `TrainedModel` closes its body too, so save a body
fine-tuned in memory first. `docrooter doctype --chunk-size N --max-pages N` bounds memory on
large corpora. The [memory guide](docs/site/guides/memory-performance.md) has every setting, and
[docs/performance.md](docs/performance.md) the measured peaks (for example, classifying a
642-page datasheet and fine-tuning on EDGAR items).

## Benchmarks

The benchmark suite (`benchmarks/`) produces every number in this README, from results files
committed under `benchmarks/results/`:

- [docs/benchmarks-classification.md](docs/benchmarks-classification.md): accuracy against
  tokens for every flow, level and backend of `docrooter.classify` (results pending);
- [docs/benchmarks-pdf.md](docs/benchmarks-pdf.md): TOC extraction, document type and section
  normalisation on the 51-PDF dev corpus;
- [docs/benchmarks-heldout.md](docs/benchmarks-heldout.md): the frozen run on the 40-PDF
  held-out corpus;
- [docs/benchmarks-element-tags.md](docs/benchmarks-element-tags.md): the element tagger on a
  Claude-annotated gold set (results pending);
- [docs/benchmarks-selection.md](docs/benchmarks-selection.md): the completeness dial: realised
  recall against the promised one, and tokens against baselines (results pending);
- [docs/benchmarks.md](docs/benchmarks.md): section classification, zero-shot and few-shot, on
  SEC 10-K items (EDGAR-CORPUS), DBpedia classes and LEDGAR contract clauses, with the default
  backend, training base and strategy decided under rules written down before the runs.

See [benchmarks/README.md](benchmarks/README.md) for the setup and how to run them.

## Development

```bash
git clone https://github.com/LandoCole/docrooter && cd docrooter
uv sync                         # core + dev tools; add --extra models for the model extras
uv run pytest                   # unit tests: fast, no model downloads, never import torch
uv run pytest -m models         # model tests: download models and need --extra models
uv run ruff check .             # lint
uv run ruff format --check .    # formatting
uv run mypy                     # types (src/docrooter)
uv run pytest --cov=docrooter   # coverage; needs the pdf extras for docrooter.pdf to reach 80%
uv build                        # sdist and wheel in dist/
PYTHONPATH=src uv run mkdocs serve   # the documentation site (pages in docs/site/)
```

Tests marked `models` are deselected by default. The live `jev` test skips unless
`TYPESAFE_API_KEY` is set. For the benchmarks, add `--extra bench`.

[CONTRIBUTING.md](CONTRIBUTING.md) has the full setup (including `PYTHONPATH=src` on macOS), the
memory-budget and benchmark rules and the pull request checklist. See also the
[CHANGELOG](CHANGELOG.md), the [security policy](SECURITY.md) and the
[code of conduct](CODE_OF_CONDUCT.md).

## Licence

Apache-2.0 for docrooter. See [LICENSE](LICENSE). The optional `[pdf]` extra pulls in the
AGPL-3.0 PyMuPDF, as noted above; `[pdf-mit]` (pdfplumber) is MIT.
