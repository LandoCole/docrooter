# docrooter v0.2: PDF TOC engine, document-type classification, real-PDF benchmark

- **Date:** 2026-10-16
- **Status:** Approved under the user's standing instruction of 2026-10-16 to "write additional specs and implementation plans and continue on your own"
- **Builds on:** `2026-10-15-docrooter-section-classification-design.md` (v0.1)

## 1. Purpose

v0.1 classifies sections against any taxonomy. v0.2 adds the use case the user named as primary: **classifying and categorising whole documents by type, driven by their TOC**. It also normalises TOC sections into canonical section labels.

Typical use is as a cheap gate in front of expensive systems, for example "only embed documents that are investor letters".

Two pieces of v0.1 were weak on real PDFs:

- The `pymupdf4llm` path took 33 s for 92 pages, because OCR ran.
- It produced level drift: 165 headings at level 5 on an annual report.

v0.2 replaces it with a native multi-signal TOC engine and proves the whole pipeline on the 51-PDF labelled corpus in `benchmarks/corpus/`.

### Success criteria

1. **PDF loading is fast and accurate.**
   - `dr.load("x.pdf")` with the default engine takes under 1.5 s per 100 pages on the M1, with no OCR.
   - On the 24 corpus PDFs that have ≥5 bookmarks, heading F1 is at least 0.70 against bookmark titles. The bookmarks are hidden from the extractor during scoring.
2. **Document types are one call.** `dr.classify_document(doc, taxonomy)` and `docrooter doctype <files|dirs> -t doc_types.json` return a document type with probabilities and timings. The CLI has `--keep` and `--jsonl` options for pipeline gating.
3. **Section normalisation.** `dr.normalize_sections(doc, doc_type=...)` maps each TOC entry to a canonical section label from a built-in section-type taxonomy.
4. **The real-PDF benchmark** covers all 51 PDFs and 18 leaf types, comparing docrooter backends with Jev.
   - **Metrics:**
     - Accuracy at leaf and at level 1.
     - Macro precision, recall and F1 at leaf level.
     - Per-class P/R/F1.
     - Confusion matrix.
     - Average time per document, split into PDF parsing and classification.
     - Jev token cost.
   - **Results:** published in `docs/benchmarks-pdf.md`.
5. Everything ships with tests, docs and CLI help. The core install stays torch-free.

### Non-goals (v0.2)

- OCR of scanned PDFs.
- Table extraction.
- DOCX/HTML loaders.
- The UI. That is phase 4 and gets its own spec.

## 2. PDF TOC engine (`docrooter.pdf`)

### 2.1 Line extraction (`pdf/lines.py`)

```python
class Line(BaseModel):
    page: int                # 1-based
    text: str
    size: float              # dominant span size (by characters)
    bold: bool               # span flag bit 16 or "Bold"/"Black"/"Heavy" in the font name, for most characters
    font: str
    x0: float
    y0: float
    x1: float
    y1: float
    page_height: float
    page_width: float
    gap_above: float         # vertical gap to the previous line on the same page (inf at page top)
    gap_below: float
    block: int
```

Two engines:

- **`pymupdf`** (extra `pdf`): `page.get_text("dict", flags=TEXT_PRESERVE_WHITESPACE)`, iterating blocks → lines → spans. No OCR.
- **`pdfplumber`** (extra `pdf-mit`; MIT licensed, built on pdfminer): `page.extract_words(extra_attrs=["size","fontname"])`, grouped into lines by `top` within a tolerance of 2 pt.

Both engines produce `list[Line]`. Lines that contain no alphanumerics are dropped.

### 2.2 Document signals (`pdf/signals.py`)

- **Body size:** the character-weighted mode of `size`.
- **Median line gap.**
- **Running headers and footers:** a line in the top or bottom 8% of the page whose normalised text (digits replaced by `#`, casefolded) recurs on at least 30% of pages, with a minimum of 3. These lines are excluded from both headings and body.
- **Page-number lines:** for example `12`, `Page 3 of 9`, `- 4 -`, `iv`. Excluded.
- **Printed TOC pages:** a page with at least 5 lines matching `^(.*?)[\s.·…_-]{2,}(\d{1,4})$`, or a heading of "Contents" or "Table of Contents" followed by such lines. The parsed entries become `(title, page_label)`.
- **Bookmarks:** `get_toc()` (pymupdf only) gives `[level, title, page]`.

### 2.3 Heading candidates and scoring (`pdf/headings.py`)

**Pre-filter.** A line is a candidate when all of these hold:

- It is at most 20 words.
- It is not a header, footer or page number.
- It does not appear on a printed TOC page.
- At least one of the following:
  - `size ≥ 1.15 × body`
  - bold and isolated (`gap_above ≥ 1.2 × median gap`, or it starts a block)
  - a numbering prefix and isolated

Consecutive candidate lines on one page that share a style (size within 0.5 pt, same bold) and have a small gap are merged into one multi-line title.

**Score.** The score is in [0, 1] and is clamped:

| Signal | Contribution |
|---|---|
| v0.1 `score_heading(title, repeat_count)` | weight 0.4 |
| Size ratio | `min((size/body − 1) / 0.6, 1)`, weight 0.25 |
| Bold | 0.1 |
| Isolation | 0.1 |
| Numbering | 0.1 |
| Fuzzy match (ratio ≥ 0.85) against a printed TOC entry | +0.25 |
| Fuzzy match against a bookmark title | +0.25 |
| Candidate is followed by no body text before the next candidate of the same style | −0.15 |

Candidates scoring at least `min_heading_score` (default 0.55) are kept.

**Levels.**

1. Cluster the kept headings by style key `(round(size*2)/2, bold)`. Rank the clusters by size, descending, with bold before regular on ties. The rank becomes the level, capped at 6.
2. If at least 50% of headings carry dotted numbering, the numbering depth overrides the level.
3. A bookmark match overrides both, and the heading gets flag `level_from_bookmark`.
4. `repair_levels` (v0.1) closes gaps.

The v0.1 numbering regex is fixed as part of this work:

- Roman numerals `IV.` followed by a space now match.
- Matching is case-insensitive (`ITEM 7`).
- The title-case tokenizer keeps apostrophes and hyphens inside words.

### 2.4 Assembly (`pdf/load.py`)

```python
load_pdf(path, *, engine="auto", min_heading_score=0.55, use_bookmarks=True,
         use_printed_toc=True, repair=True) -> Document
```

- **Engines:**
  - `auto` picks `pymupdf` if it is installed, else `pdfplumber`.
  - `both` runs both engines and keeps a heading when it is found by both (fuzzy title match on the same page), or when one engine scored it at least 0.8.
  - `pymupdf4llm` is the v0.1 path, kept as a legacy option.
- **Body text:** each section's body is the concatenation of non-excluded lines up to the next heading, with lines joined by `"\n"`. Page boundaries are kept as `"\n"`.
- **`Document.meta`** is a new dict field, default `{}`. It holds:
  - `engine`, `n_pages`
  - `timings`: `{"extract": s, "signals": s, "headings": s, "assemble": s}`
  - `bookmarks: int`, `printed_toc_entries: int`, `body_size: float`

`dr.load()` routes `.pdf` to `load_pdf` with the default engine. `docrooter toc x.pdf` gains `--engine`.

### 2.5 TOC benchmark (`benchmarks/toc.py`)

- **Gold:** the bookmark list of each corpus PDF with at least 5 bookmarks (24 documents).
- **Extraction:** runs with `use_bookmarks=False`.
- **Matching:** a predicted heading matches a gold bookmark when `difflib` ratio ≥ 0.85 on normalised titles and the pages are within ±1. Matching is one-to-one and greedy by ratio.
- **Metrics per document and in aggregate:**
  - heading precision, recall and F1 (micro and macro over documents)
  - level accuracy on matched pairs, where the gold levels are bookmark levels shifted so the minimum is 1
  - seconds per document
- **Engines compared:** `pymupdf`, `pdfplumber`, `both`, `pymupdf4llm`.

Because many bookmark sets list only top-level sections, precision is reported but **recall and F1 at bookmark depth ≤ 2** are the headline numbers.

## 3. Document-type classification

### 3.1 DocumentView (`docrooter/doc_view.py`)

```python
DocumentView(fields=("title","stats","toc","lead","keyphrases"), max_tokens=512,
             toc_depth=2, toc_entries=40, lead_words=120, keyphrases=12)
render(doc, count_tokens=None) -> RenderedView
```

Rendered as:

```
Title: <doc.title or source stem>
Stats: <n_pages> pages, <n_sections> sections
Contents: 1 Intro; 1.1 Scope; 2 Results; ...   (entries up to toc_depth, in reading order)
Lead: <first lead_words words of body text, skipping sections titled like cover/notice/table of contents>
Key phrases: <doc-level keyphrases: section keyphrases pooled, top-k by summed score>
```

The token-budget behaviour is the same as `SectionView`: it never renders empty text, and it truncates the last field that fits.

### 3.2 API

```python
dr.classify_document(doc, taxonomy, *, backend=..., strategy=Flat(), view=DocumentView(),
                     instruction="What type of document is this?", mode="view") -> DocTypeResult
Classifier.classify_documents(docs, *, view=DocumentView(), mode="view") -> list[DocTypeResult]
```

- **`mode="view"`:** one classification of the rendered DocumentView.
- **`mode="sections"`:** classify every leaf section's view against the doc-type taxonomy, then aggregate. The aggregate is the token-weighted mean of the leaf-probability vectors from Flat scoring, and the doc type is the argmax.
- **`mode="hybrid"`:** `0.6 × view + 0.4 × sections`, averaged as probabilities over leaves.

**`DocTypeResult`** contains:

- `source`
- `best: PathResult`
- `leaf_probs: dict[str, float]` for the top 10
- `view_text`
- `timings: {"classify": s}`
- `mode`

**Default strategy is `Flat()`.** Document-type taxonomies are shallow. In v0.1 B2, Flat with full-path candidate text beat hierarchical decoding for zero-shot embeddings by a wide margin (0.69 vs 0.14–0.21 hF1 in the smoke run). Candidate text includes descriptions, so the definitions matter.

### 3.3 Built-in taxonomies (package data `docrooter/taxonomies/`)

- **`doc_types.json`:** the corpus taxonomy (6 groups, 18 types, each with a definition).
- **`section_types/`:**
  - `annual_report.json`: shareholder letter, business overview, risk factors, MD&A, financial statements, notes, governance, executive compensation, controls, legal proceedings, properties, market for equity, exhibits, sustainability, other
  - `research_paper.json`: abstract, introduction, related work, background, method, experiments, results, discussion, limitations, conclusion, acknowledgements, references, appendix
  - `contract.json`: parties/recitals, definitions, scope/services, term, payment, confidentiality, intellectual property, warranties, indemnification, limitation of liability, termination, governing law/disputes, general provisions, signatures/exhibits
  - `technical_document.json`: overview, installation/getting started, configuration, usage, reference, specifications/electrical characteristics, troubleshooting, appendix
  - `generic.json`: front matter, summary, introduction, body, conclusion, references, appendix

  Every node has a one-sentence definition. A mapping `DOC_TYPE_TO_SECTION_TAXONOMY` sends leaf doc types to one of these, with `generic` as the fallback.
- **Access:** `dr.builtin_taxonomy(name) -> Taxonomy`, where name is `"doc_types"` or `"section_types/annual_report"`. Data is loaded with `importlib.resources`.

### 3.4 Section normalisation

```python
dr.normalize_sections(doc, *, doc_type=None, taxonomy=None, backend=..., levels=(1, 2)) -> NormalizationResult
```

- **Choosing the section taxonomy:** use `taxonomy` if given, else the section taxonomy mapped from `doc_type`. If `doc_type` is None, call `classify_document` first.
- **Classification:** each TOC entry at the given levels is classified with `SectionView(fields=("title","lead"), max_tokens=96)` and `Flat()`.
- **Result:** maps `section_id` to `{canonical: node_id, label, score}`, plus the `doc_type` used.

### 3.5 CLI

```
docrooter doctype PATH... -t TAXONOMY|builtin:doc_types [-b BACKEND] [--mode view|sections|hybrid]
                  [--engine auto|pymupdf|pdfplumber|both] [--keep ID ...] [--jsonl OUT] [--workers N]
docrooter normalize FILE [--doc-type ID] [-b BACKEND] [--json]
```

- `doctype` expands directories recursively to `*.pdf`, `*.md` and `*.txt`.
- It prints a rich table: file, type, probability, parse seconds, classify seconds.
- `--keep` prints or writes only the files whose doc type, or any ancestor of it, is in the keep list. This is the pipeline gate.
- `--jsonl` writes one record per file with `{source, type, label, prob, leaf_probs, timings}`.
- PDF parsing runs in a process pool of `--workers` processes (default `min(4, cpu_count)`). Classification is batched in the main process.

## 4. Real-PDF benchmark (`benchmarks/pdf_bench.py`)

- **Corpus:** `benchmarks/corpus/manifest.jsonl` (51 PDFs). Gold labels come from the manifest.
- **Parsing:** each PDF is parsed once per engine and cached as Document JSON in `benchmarks/.cache/docs/`. Parse time is recorded per document.

### Systems

| Id | What |
|---|---|
| `ultrabert-view` | `embedding:dmis-lab/UltraBERT-Embed-150M`, mode view, Flat over the 18 leaves |
| `ultrabert-hybrid` | same backend, mode hybrid |
| `bge-view` | `embedding:BAAI/bge-small-en-v1.5`, mode view |
| `nli-view` | `nli:MoritzLaurer/deberta-v3-base-zeroshot-v2.0`, mode view |
| `jev-view` | `jev:jev-1.13.0`, docrooter's DocumentView sent as state; one Choice over the 18 leaves, with definitions as criteria |
| `jev-chunks` | the user's suggested baseline, with no docrooter structure: take raw page text (pymupdf), split it into ~1,500-token chunks, and pick up to 8 evenly spaced chunks; one Jev Choice per chunk; aggregate by mean probability |
| `ultrabert-loo` | trained heads-only on UltraBERT-Embed view embeddings, leave-one-out over 51 documents (train on 50, predict 1). Embeddings are computed once and reused |

### Output

`benchmarks/results/<date>/pdf.json` holds per-document predictions and timings. `docs/benchmarks-pdf.md` contains:

- an aggregate table, one row per system: accuracy at leaf and level 1, macro P/R/F1, average parse s/doc, average classify s/doc, total s/doc, and Jev input tokens with estimated cost at $0.042/M
- per-class P/R/F1 for the best docrooter system and for Jev
- a confusion matrix for those two systems
- a per-document table: file, pages, gold, prediction per system

It also includes the TOC benchmark (§2.5) and a normalisation spot-check:

- **Silver labels:** an agent labels the TOC entries of 6 corpus documents with their canonical section type: 3 research papers, 2 annual reports and 1 contract. Labels are reviewed against definitions, are committed in `benchmarks/corpus/section_gold.jsonl`, and are disclosed as silver.
- **Reported:** accuracy of `normalize_sections`.

**Fairness:** every system receives the same documents. Jev and docrooter backends receive the same DocumentView text in `*-view` systems. Timing excludes model download but includes model load on the first document. Load time is reported separately.

## 5. Packaging

- **New extras:**
  - `pdf-mit = ["pdfplumber"]`
  - `pdf` stays pymupdf plus pymupdf4llm. pymupdf is the default engine; pymupdf4llm is legacy.
- **Package data:** included via hatch, so the wheel includes `docrooter/taxonomies/**/*.json`.
- **Unit tests:**
  - PDF unit tests generate small PDFs with pymupdf (skipped without the extra).
  - Signal and heading logic is tested on synthetic `Line` lists with no PDF dependency.
  - Unit tests remain torch-free.

## 6. Decisions and risks

- **PyMuPDF licensing:** PyMuPDF is AGPL. It is the fast default only when installed. `pdf-mit` gives a permissive path. The README states the trade-off.
- **Heading quality:** heading-score weights are fixed constants in v0.2. A learned scorer trained on bookmark labels is deferred, but the TOC benchmark produces exactly the data it would need.
- **Small corpus:** 51 documents and 18 classes means about 3 per class. The benchmark reports this sample size, and the leave-one-out trained system is labelled as indicative.
