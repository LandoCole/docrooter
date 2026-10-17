# Held-out generalisation evaluation

Does docrooter generalise beyond the documents it was tuned on? Every number in
`docs/benchmarks-pdf.md` comes from the 51-PDF dev corpus (`benchmarks/corpus/`), and every
default (views, thresholds, heading rules, section-taxonomy synonyms, the tuned view) was
chosen while looking at those numbers. This page reports **one** frozen run on a separate
40-PDF corpus that no tuning step ever saw.

## Method

**Corpus.** `benchmarks/heldout/` (see its `README.md`): 40 PDFs, the same 18 leaf types
(2-3 per type), no document and no publisher in common with the dev corpus. It is deliberately
wider: UK, EU, Canadian and Australian sources, documents from 1896 to 2026, scans and old
typesetting, one image-only PDF, and a 1-page letter up to an 816-page standard. The
document-type labels were committed in `ae6b037` before any run.

**Silver section labels.** `benchmarks/heldout/section_gold.jsonl` (commit `9d195cb`, before
any model run on this corpus): 157 level-1/2 sections of 4 documents (24 marked as parser
junk). They follow the P10 procedure exactly: load with the default engine, take the level-1/2
entries, read each heading plus ~300 characters of its body against the node definitions of the
doc type's mapped section taxonomy. Like the dev silver set, these labels were assigned by an
agent (Claude Opus 5.5), not checked by a human. Ambiguous rows carry a `note`.
The documents: the held-out set's only two research papers (`ho_023` *Generative Adversarial
Nets*; `ho_024` Shannon 1948), one annual report (`ho_001` Rio Tinto 2023; chosen over `ho_002`
because it has fewer sections: 98 vs 151) and one contract (`ho_013` HMRC model services
contract; `ho_014` parses to only 2 sections). They were chosen by type and section count only.

**Freeze.** The run started on 2026-10-16 at 22:17 EDT at commit **`72f6135`**, which is the
frozen commit. It finished at 22:29 with every harness exiting 0. Each results file
(`benchmarks/results/2026-10-16/heldout-{toc,pdf,normalize}.json`) carries a `freeze` record
with the frozen commit, the label commits (`ae6b037` for the manifest and `9d195cb` for the
silver sections), per-step start and end times, HEAD at the end of each step (`72f6135` every
time) and the list of benchmark-code paths changed since the freeze (empty every time). There
were no code or configuration changes between the freeze and the run, and none after it. The run
happened once and nothing was re-run.

Disclosures:

- **Harness bug fixed before the freeze, not after it.** The doc-type harness raised
  "the classifier saw a different view text" for a document with no text layer, which would have
  aborted every docrooter system at `ho_018`. This was found by a unit test, before any held-out
  run. Since the fix, such a document is a recorded `empty_text` miss.
- **Smoke test.** The new transfer systems were smoke-tested before the freeze on 2 dev PDFs
  under renamed ids, never on held-out data.
- **Commits behind the dev numbers.** The dev numbers in the comparison come from the dev
  results files at the commits recorded in them:
  - TOC: `e72fca6`.
  - Doc type: `53090c0`, `3bd85f2` and `0c22383+dirty`. The `+dirty` is a heading-level edit of
    the report renderer that was uncommitted when the dev re-run started.
  - Normalisation: `0c22383`.

  The `src/` changes between those commits and `72f6135` are pipeline events, memory caps,
  `max_pages` and taxonomy caching. None of them is meant to change any output, but the dev TOC
  and doc-type numbers were not re-measured at `72f6135`.

**What ran** (`python -m benchmarks.heldout_eval run`, one model process at a time):

1. TOC extraction with the `pymupdf`, `pdfplumber` and `both` engines on the held-out PDFs
   with at least 5 bookmarks. These are the same metrics as on dev, and no models are involved.
2. Document type on all 40 PDFs:
   - the dev-benchmark systems `jev-view`, `jev-chunks`, `ultrabert-view`, `ultrabert-hybrid`
     and `bge-view`, with the same shared-view rendering as the dev benchmark;
   - two "trained on dev, tested on held-out" systems, which replace the dev leave-one-out
     systems:
     - `ultrabert-dev-logistic`: logistic regression on UltraBERT default-view embeddings of
       all 51 dev documents. C is chosen from {1, 10, 100, 1000} by leave-one-out on dev, then
       the model predicts the held-out documents.
     - `ultrabert-bge-dev-ensemble`: the UltraBERT+bge zero-shot mixture, with the weight from
       {0, .25, .5, .75, 1} chosen by accuracy on the 51 dev documents.
3. Section normalisation on the 4 silver documents with UltraBERT, bge-small and Jev, using the
   shipped defaults (`path+title` view with heading synonyms). There is also one `doc_type=None`
   run with the default backend.

**Known noisy items.**

- `ho_018` (*Donoghue v Stevenson*) is an image-only scan with no text layer. docrooter does
  not OCR, so every docrooter system stops it as `empty_text`, and `jev-chunks` gets no text.
  This is expected and **counted as a miss** for every system, so 39/40 is the ceiling.
- `ho_006` (ADP letters to shareholders) is labelled `shareholder_letter` by content, but its
  cover carries a proxy-statement banner, so `proxy_statement` is a defensible error.
- The silver section labels are single-annotator agent labels. Some sections have no natural
  node, for example the contract's audit, insurance and remedies clauses, and the annual
  report's mining-reserves pages. Those rows carry a `could be ...` note, so expect a few points
  of label noise.

## Analysis

**Headline:**

- Document-type classification generalises: no system drops by more than about 5 points, and
  the local zero-shot view systems (`ultrabert-view`, `bge-view`) do better on held-out than on
  dev.
- Section normalisation holds up for the local models and drops slightly for Jev.
- TOC extraction is where the held-out set exposes real weaknesses: d<=2 F1 falls by about 12
  points for the default engine. The `pdfplumber` engine also fails outright on one large
  document.

With 40 documents, one document is 2.5 points. The 95% bootstrap intervals of leaf accuracy are
25-30 points wide for the local systems and 7.5-17.5 points for Jev, whose scores sit near the
ceiling, so only the large changes below mean anything.

### Document type: generalises

- **`jev-view` 0.925 (dev 0.961) and `jev-chunks` 0.975 (dev 0.922).**
  - Jev's misses are `ho_018`, `ho_006` and `ho_040`:
    - `ho_018` is image-only, so every system misses it and 0.975 is the ceiling.
    - `ho_006` is the known noisy shareholder letter with a proxy banner.
    - `ho_040` is an investor presentation, predicted as a white paper.
  - `jev-chunks` reaches the ceiling. It sees up to 8 chunks of raw page text, which helps on
    covers that mislead (ho_006, ho_040).
- **Zero-shot local models do not drop.**
  - `ultrabert-view` scores 0.825 (dev 0.765) and `bge-view` 0.700 (dev 0.608).
  - Both use the default `DocumentView()`, not the lead-first view chosen on dev in P9b
    (`ultrabert-view-tuned` was not part of this run). Their dev numbers were not inflated by
    tuning, and the held-out set is, if anything, a little easier for them.
- **`ultrabert-hybrid` drops slightly: 0.775 (dev 0.824).** The section half pulls `ho_001`
  (Rio Tinto, whose strategic report is heavy on ESG) to `sustainability_report` and `ho_017`
  (Plessy v. Ferguson, a scan) to `legislation`.
- **Trained on dev, tested on held-out: the honest transfer test is the weakest result.**
  - `ultrabert-dev-logistic` scores 0.675, against 0.706 for dev leave-one-out (C=1000, the same
    value the dev folds chose).
  - It is **15 points below the zero-shot `ultrabert-view` on the same documents** (0.825). A
    head fitted on 51 documents (2-4 per class) learns publisher and layout cues that do not
    transfer:
    - both earnings releases and one half-year report go to `investor_presentation`;
    - CDW's proxy statement and ADP's letters go to `annual_report`;
    - Shannon 1948 goes to `shareholder_letter`.
  - The dev+bge ensemble (weight 0.5, chosen on dev) scores 0.800 (dev 0.824), which is again
    **below zero-shot UltraBERT alone** (0.825). The dev-chosen weight does not transfer as a
    gain.
  - Conclusion: at this corpus size, few-shot training on dev does not beat the zero-shot
    default for unseen publishers. The zero-shot default is the right shipping choice.
- **Weak classes for local models.**
  - `investor_presentation`: every local system misclassifies both the Moody's and the Urban
    Edge decks.
  - `white_paper`: the zero-shot local systems miss both Uniswap v3 and the Ethereum explainer.
    The UltraBERT systems send them to `research_paper` or `government_report`, bge-small to
    `earnings_release` and `technical_standard`. Only the dev-trained logistic head gets both.
  - *An Introduction to R* (`ho_030`, a user manual): the UltraBERT systems say
    `research_paper`, bge-small `datasheet`; the dev-trained head gets it right.
  - The first two weaknesses are already there on dev, where `ultrabert-view` scores
    white-paper F1 0.00 and investor-presentation F1 0.50, so they are not something the
    held-out set introduced. (On dev its user-manual F1 is 0.86, so `ho_030` is a new miss.)
    With 2-3 documents per class, the data cannot say how much of this comes from the genres
    themselves.

### TOC extraction: drops, partly real and partly bookmark artefacts

**Default engine (`pymupdf`).** d<=2 F1 falls from 0.703 to 0.576 micro and from 0.591 to
0.480 macro. The all-depth micro F1 rises (0.685 to 0.754), but only because ECMA-262 (`ho_025`)
has 2,159 of the 4,363 bookmarks and pymupdf matches it at 0.96. Without `ho_025`, all-depth
micro F1 is 0.553.

**Real weaknesses the held-out set exposes:**

1. **`pdfplumber` returns 3 headings for ECMA-262 (816 pages, 2,159 bookmarks)**, without
   raising an error. That one document takes `pdfplumber` from roughly level with `pymupdf` to
   0.360 all-depth micro F1. It also drags `both` down at all depths (272 predictions against
   pymupdf's 2,065). Excluding `ho_025`, the three engines are within about 2 points of each
   other (d<=2 micro 0.505-0.526). The cause was not investigated, because the held-out set must not be
   used for debugging. It should be reproduced on a dev or synthetic document: a silent collapse
   on a large, well-formed PDF is a robustness bug, not a tuning matter.

   **After the frozen run, a candidate cause was reproduced on dev and synthetic PDFs and fixed
   (v0.3 Q5, `f3baa43`); whether it is `ho_025`'s cause is not confirmed.** The fix did not
   touch this set. On those PDFs, a text matrix with a tiny skew gives every glyph a slightly different float size in
   pdfminer, and pdfplumber then split every glyph into its own word, so titles came out
   letter-spaced and stopped being heading candidates. A synthetic specification went from 0
   to 12 of 12 headings under `pdfplumber` with the fix. Whether this was `ho_025`'s cause
   was not checked, because that would mean debugging on the held-out PDF; see the tuning log
   in [benchmarks-pdf.md](benchmarks-pdf.md#tuning-log-p5b). **Every held-out number in this
   document is pre-fix:** it was measured at the frozen commit `72f6135`, before `f3baa43`
   landed, and is not re-run, so the `pdfplumber` and `both` rows still include the collapse.
2. **Earnings releases and bilingual statutes.**
   - Pinterest's earnings release (`ho_007`, 16 bookmarks) yields one heading. Its section
     headings ("Q2 2024 Financial Highlights") are bold at body size (10 pt Arial) and are not
     detected.
   - PIPEDA (`ho_021`, English and French in parallel columns) produces merged bilingual
     headings ("Short Title Titre abrégé") that match no bookmark.
3. **Level assignment.** Microsoft's sustainability report (`ho_038`) has a two-line cover title
   that becomes two headings, which pushes the later headings one level deeper than the
   bookmarks. The d<=2 cut then removes them (d<=2 F1 0.00, all-depth 0.22).

**Bookmark sets that are not tables of contents** (gold artefacts, not extractor errors):

- `ho_029`'s 72 bookmarks are design-tool page labels ("01 (1)", "02 (2)").
- `ho_039`'s bookmarks are "Slide N: title".
- `ho_018` has no text layer.

Excluding those three (post hoc, shown only for scale), pymupdf d<=2 is 0.649 micro and 0.551
macro, against 0.703 and 0.591 on dev. That remaining gap of about 5 points is within what
23 documents can show.

### Section normalisation: holds for local models, Jev slightly lower

| Backend | Dev accuracy (n=142) | Held-out accuracy (n=133) | Held-out, noted alternative counted as right* |
| --- | --- | --- | --- |
| `ultrabert` (default) | 0.514 | 0.579 | 0.632 |
| `bge-small` | 0.408 | 0.496 | 0.571 |
| `jev` | 0.725 | 0.692 | 0.782 |
| auto doc type (UltraBERT) | 0.479 | 0.406 | - |

\* Post hoc and for scale only: a prediction also counts when it is the alternative named in the
row's `could be ...` note. 50 of the 133 held-out rows carry such a note, against 21 of the 142
dev rows. The held-out documents have more sections with no natural node (contract audit,
insurance and remedies clauses; mining reserves pages in the annual report).

- **The dev numbers here are optimistic and the held-out ones are not.** The defaults (the
  `path+title` view and the heading synonyms) were chosen on the dev silver set. On held-out
  data, UltraBERT and bge-small do *better*, so that choice did not overfit. Only 14 of the 133
  held-out headings equal a synonym example.
- **Jev drops 3 points in accuracy and 5.5 in macro F1.**
  - Its errors are mostly annual-report sections:
    - strategy and product-group pages labelled `other`;
    - risk-management pages labelled `corporate_governance`;
    - board pages labelled `sustainability`.
  - There are also Shannon's theory parts, labelled `background` rather than `method`.
  - 12 of its 41 errors are the alternative already named in the row's `could be ...` note.
- **Shannon 1948 (`ho_024`) is the hardest document for every backend.** Its 6 scored sections
  are the paper's Parts I-V, a 1948 structure that the modern research-paper taxonomy
  (abstract, method, experiments...) does not fit.
- **The auto doc-type path drops 7 points.** The library's default doc-type step (UltraBERT,
  confidence 0.21) types the HMRC contract (`ho_013`) as `terms_policy`. That selects the
  generic taxonomy, so all 40 of its sections count as wrong. `ultrabert-view` in the doc-type
  benchmark gets this document right. The auto path takes one low-confidence decision, so passing
  `doc_type=` explicitly remains the recommendation when the type is known.

### Where it generalises and where it does not

- **Generalises:**
  - Zero-shot document typing, both local (UltraBERT) and Jev.
  - Section normalisation with the shipped defaults.
  - TOC extraction on conventional, typeset documents: standards, manuals, datasheets,
    prospectuses and government reports mostly score 0.65-0.94 d<=2 F1. The exception is the
    LR8 datasheet at 0.50.
- **Does not generalise, or is fragile:**
  - Few-shot heads trained on the 51 dev documents. They lose to zero-shot on new publishers.
  - `pdfplumber` on at least one very large PDF, failing silently.
  - Headings in earnings releases, bilingual statutes and slide decks.
  - Heading levels after a split cover title.
  - Image-only PDFs, which are out of scope without OCR.

**Recommended follow-ups (on dev or synthetic data, never on this set):**

- reproduce the `pdfplumber` collapse (a candidate cause was reproduced and fixed in v0.3 Q5 on dev
  and synthetic PDFs, `f3baa43`; `ho_025`'s own cause is not confirmed);
- add a guard so that `both` does not lose a good `pymupdf` parse when the second engine
  returns almost nothing;
- add an `empty_text` or `no_text_layer` hint to the CLI output;
- record the doc-type confidence in `normalize_sections` results, so a low-confidence auto type
  can be flagged.

A held-out set used to choose these fixes is no longer held-out. A new set must be collected
before the next generalisation claim.

## Results

The tables below are generated by `python -m benchmarks.heldout_eval render` from the results
files. The *Configurations tried* table in the normalisation section lists only the shipped
default: nothing was tried on this set.

<!-- heldout-benchmark:begin -->
## Dev vs held-out

| Task | System | Metric | Dev | Held-out | Change | n dev / held-out |
| --- | --- | --- | --- | --- | --- | --- |
| TOC | `pymupdf` | F1 micro (d<=2) | 0.703 | 0.576 | -12.7 pts | 24 / 23 |
| TOC | `pymupdf` | F1 macro (d<=2) | 0.591 | 0.480 | -11.2 pts | 24 / 23 |
| TOC | `pymupdf` | F1 micro (all) | 0.685 | 0.754 | +6.9 pts | 24 / 23 |
| TOC | `pdfplumber` | F1 micro (d<=2) | 0.711 | 0.458 | -25.3 pts | 24 / 23 |
| TOC | `pdfplumber` | F1 macro (d<=2) | 0.599 | 0.439 | -16.0 pts | 24 / 23 |
| TOC | `pdfplumber` | F1 micro (all) | 0.702 | 0.360 | -34.2 pts | 24 / 23 |
| TOC | `both` | F1 micro (d<=2) | 0.716 | 0.593 | -12.2 pts | 24 / 23 |
| TOC | `both` | F1 macro (d<=2) | 0.604 | 0.490 | -11.4 pts | 24 / 23 |
| TOC | `both` | F1 micro (all) | 0.717 | 0.448 | -26.8 pts | 24 / 23 |
| Doc type | `jev-view` | leaf accuracy | 0.961 | 0.925 | -3.6 pts | 51 / 40 |
| Doc type | `jev-view` | macro F1 | 0.944 | 0.930 | -1.4 pts | 51 / 40 |
| Doc type | `jev-chunks` | leaf accuracy | 0.922 | 0.975 | +5.3 pts | 51 / 40 |
| Doc type | `jev-chunks` | macro F1 | 0.911 | 0.989 | +7.8 pts | 51 / 40 |
| Doc type | `ultrabert-view` | leaf accuracy | 0.765 | 0.825 | +6.0 pts | 51 / 40 |
| Doc type | `ultrabert-view` | macro F1 | 0.709 | 0.778 | +6.8 pts | 51 / 40 |
| Doc type | `ultrabert-hybrid` | leaf accuracy | 0.824 | 0.775 | -4.9 pts | 51 / 40 |
| Doc type | `ultrabert-hybrid` | macro F1 | 0.765 | 0.731 | -3.4 pts | 51 / 40 |
| Doc type | `bge-view` | leaf accuracy | 0.608 | 0.700 | +9.2 pts | 51 / 40 |
| Doc type | `bge-view` | macro F1 | 0.541 | 0.654 | +11.3 pts | 51 / 40 |
| Doc type | `ultrabert-loo -> ultrabert-dev-logistic` | leaf accuracy | 0.706 | 0.675 | -3.1 pts | 51 / 40 |
| Doc type | `ultrabert-loo -> ultrabert-dev-logistic` | macro F1 | 0.647 | 0.665 | +1.8 pts | 51 / 40 |
| Doc type | `ultrabert-bge-loo -> ultrabert-bge-dev-ensemble` | leaf accuracy | 0.824 | 0.800 | -2.4 pts | 51 / 40 |
| Doc type | `ultrabert-bge-loo -> ultrabert-bge-dev-ensemble` | macro F1 | 0.784 | 0.753 | -3.1 pts | 51 / 40 |
| Normalisation | `ultrabert` | accuracy | 0.514 | 0.579 | +6.5 pts | 142 / 133 |
| Normalisation | `ultrabert` | macro F1 | 0.572 | 0.564 | -0.8 pts | 142 / 133 |
| Normalisation | `bge-small` | accuracy | 0.408 | 0.496 | +8.8 pts | 142 / 133 |
| Normalisation | `bge-small` | macro F1 | 0.554 | 0.555 | +0.1 pts | 142 / 133 |
| Normalisation | `jev` | accuracy | 0.725 | 0.692 | -3.4 pts | 142 / 133 |
| Normalisation | `jev` | macro F1 | 0.762 | 0.707 | -5.5 pts | 142 / 133 |
| Normalisation | `auto_doc_type` | accuracy | 0.479 | 0.406 | -7.3 pts | 142 / 133 |
| Normalisation | `auto_doc_type` | macro F1 | 0.494 | 0.394 | -10.0 pts | 142 / 133 |

## TOC extraction (held-out)

Gold: bookmarks of the 23 corpus PDFs with at least 5 bookmarks (levels shifted so the minimum is 1; items without a target page dropped). Extraction runs with `use_bookmarks=False`. A prediction matches a bookmark at normalised-title ratio >= 0.85 and pages within +/-1, one-to-one and greedy by ratio. The depth <= 2 columns keep only gold and predicted headings of level <= 2, so precision is measured at that depth too. Micro pools the counts; macro is the mean of per-document F1. Level accuracy is measured on matched pairs. Bookmark sets often omit lower-level headings, so precision is pessimistic and recall / F1 at depth <= 2 are the headline numbers.

Environment: Python 3.12.13, arm64, 8 CPUs, pymupdf 1.28.2, pdfplumber 0.11.10, pymupdf4llm 1.28.2. `os.getloadavg()` was 3.9, 3.9, 3.3 (1/5/15 min) when the latest run started. The machine was shared with other jobs, so seconds per document are noisy upper bounds; each engine ran one untimed warm-up document first. Seconds are wall-clock for one `load_pdf` call. Each engine's row keeps the run that produced it: `pymupdf` 2026-10-16 at `72f6135` (load 3.9, 3.9, 3.3); `pdfplumber` 2026-10-16 at `72f6135` (load 4.6, 4.1, 3.4); `both` 2026-10-16 at `72f6135` (load 6.0, 4.4, 3.6).

### Aggregate by engine (depth <= 2 headline, all depths for reference)

| Engine | Docs | P (d<=2) | R (d<=2) | F1 micro (d<=2) | F1 macro (d<=2) | Level acc (d<=2) | P (all) | R (all) | F1 micro (all) | F1 macro (all) | Level acc (all) | s/doc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pymupdf | 23 | 0.624 | 0.534 | 0.576 | 0.480 | 0.911 | 0.765 | 0.743 | 0.754 | 0.515 | 0.923 | 1.41 |
| pdfplumber | 23 | 0.558 | 0.388 | 0.458 | 0.439 | 0.905 | 0.547 | 0.269 | 0.360 | 0.468 | 0.842 | 5.21 |
| both | 23 | 0.688 | 0.521 | 0.593 | 0.490 | 0.925 | 0.693 | 0.331 | 0.448 | 0.493 | 0.858 | 6.60 |


### Per document: `pymupdf` (the default engine)

| Doc | Title | Pages | Gold d<=2 | Pred d<=2 | P | R | F1 | Level acc | F1 (all) | s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ho_001 | Rio Tinto Annual Report 2023 | 360 | 96 | 73 | 0.66 | 0.50 | 0.57 | 0.58 | 0.44 | 6.96 |
| ho_002 | The Collinson Group Limited Annual report and... | 103 | 158 | 96 | 0.23 | 0.14 | 0.17 | 0.64 | 0.26 | 1.24 |
| ho_003 | Roche Half-Year Report 2024 | 91 | 9 | 38 | 0.24 | 1.00 | 0.38 | 0.11 | 0.99 | 1.08 |
| ho_007 | Pinterest Announces Second Quarter 2024 Results | 11 | 16 | 1 | 0.00 | 0.00 | 0.00 | n/a | 0.00 | 0.12 |
| ho_008 | International Paper Reports Second Quarter 20... | 11 | 6 | 1 | 1.00 | 0.17 | 0.29 | 1.00 | 0.29 | 0.19 |
| ho_009 | Vanguard Ultra-Short Bond ETF Prospectus (Apr... | 40 | 14 | 16 | 0.88 | 1.00 | 0.93 | 1.00 | 0.93 | 0.49 |
| ho_011 | CDW Corporation Notice of 2024 Annual Meeting... | 81 | 24 | 46 | 0.41 | 0.79 | 0.54 | 0.21 | 0.57 | 1.35 |
| ho_016 | CatalyST Privacy and cookie policy (May 2018) | 3 | 47 | 9 | 1.00 | 0.19 | 0.32 | 1.00 | 0.32 | 0.03 |
| ho_018 | M'Alister (or Donoghue) v Stevenson [1932] AC... | 62 | 62 | 0 | 0.00 | 0.00 | 0.00 | n/a | 0.00 | 0.01 |
| ho_021 | Personal Information Protection and Electroni... | 71 | 26 | 39 | 0.00 | 0.00 | 0.00 | n/a | 0.00 | 1.27 |
| ho_025 | ECMA-262, 15th edition, June 2024: ECMAScript... | 816 | 206 | 205 | 0.90 | 0.89 | 0.90 | 0.97 | 0.96 | 9.79 |
| ho_027 | ETSI EN 303 645 V2.1.1 (2020-06): Cyber Secur... | 34 | 33 | 35 | 0.91 | 0.97 | 0.94 | 0.94 | 0.94 | 0.44 |
| ho_028 | Uniswap v3 Core (March 2021) | 9 | 18 | 18 | 0.89 | 0.89 | 0.89 | 1.00 | 0.89 | 0.20 |
| ho_029 | Ethereum Basics for Governments and Institutions | 71 | 72 | 31 | 0.00 | 0.00 | 0.00 | n/a | 0.00 | 0.77 |
| ho_030 | An Introduction to R (Version 4.6.1) | 103 | 169 | 108 | 0.95 | 0.61 | 0.74 | 1.00 | 0.79 | 1.29 |
| ho_031 | Debian Reference | 285 | 105 | 134 | 0.75 | 0.95 | 0.84 | 1.00 | 0.94 | 3.14 |
| ho_032 | ESP32 Series Datasheet Version 5.3 | 78 | 47 | 69 | 0.64 | 0.94 | 0.76 | 1.00 | 0.83 | 0.63 |
| ho_033 | LR8 High-Input Voltage, Adjustable, 3-Termina... | 18 | 26 | 34 | 0.44 | 0.58 | 0.50 | 0.80 | 0.41 | 0.15 |
| ho_034 | BME280 Data sheet (revision 1.24, February 2024) | 60 | 59 | 71 | 0.77 | 0.93 | 0.85 | 1.00 | 0.89 | 0.59 |
| ho_035 | Monetary Policy Report - July 2024 | 29 | 26 | 26 | 0.65 | 0.65 | 0.65 | 1.00 | 0.67 | 0.34 |
| ho_036 | Police productivity (HC 1380, Session 2024-2026) | 51 | 10 | 17 | 0.59 | 1.00 | 0.74 | 1.00 | 0.53 | 0.49 |
| ho_038 | Microsoft 2024 Environmental Sustainability R... | 88 | 8 | 13 | 0.00 | 0.00 | 0.00 | n/a | 0.22 | 1.44 |
| ho_039 | Moody's Corporation 1Q 2024 Investor Presenta... | 70 | 71 | 40 | 0.03 | 0.01 | 0.02 | 0.00 | 0.01 | 0.52 |

## Document type (held-out)

Corpus: 40 real PDFs from `benchmarks/heldout/` with gold labels from the manifest, 18 leaf types in 6 groups (`builtin:doc_types`; every label has a definition, see `benchmarks/heldout/README.md`). Classes have 2-3 documents each, so per-class numbers are anecdotal and one document moves leaf accuracy by 2.5 points. The 95% intervals are percentile bootstraps over documents (1000 resamples, seed 0). Every system labels every document (threshold 0).

Environment: Python 3.12.13, arm64, 8 CPUs, device `mps` (MPS probe matmul took 0.19 s), torch 2.14.1, sentence-transformers 6.1.0, scikit-learn 1.9.1, typesafe-sdk 0.7.5, pymupdf 1.28.2, Jev `jev-1.13.0`. The load average at the start of each system is listed below; the machine was shared with another benchmark, so seconds are noisy upper bounds.

Timing: *parse* is one `load_pdf` call with the default engine (pymupdf), cached and shared by every system except `jev-chunks`, whose parse is pymupdf `page.get_text()` over all pages. *Classify* for the `classify_documents` systems (`*-view`, `ultrabert-hybrid`) is batch throughput: the wall time of the classify calls divided by the number of documents (local backends: one call over all documents; Jev: chunks of 8 documents with 8 concurrent requests, so each chunk's rows are saved before the next is paid for). For the leave-one-out systems (`*-loo`) it is view rendering, embedding and zero-shot scoring divided by the number of documents; the fit per fold (with its inner leave-one-out tuning) is listed separately. For `jev-chunks` it is per-document latency: the wall time of each document's concurrent chunk requests. The table marks each figure as `(batch)` or `(latency)`. *Load* is a warm-up excluded from the averages: model load plus one classification of a tiny synthetic document (the label vectors are encoded there). Jev cost is input tokens x $0.042/M (output tokens are free), with tokens as reported by the API (`usage.input_tokens`).

Views: every `*-view` system, `ultrabert-hybrid` (its view half), the leave-one-out systems and the P9b embedding experiments receive the same DocumentView text per document and view config, rendered once with the shared word-based counter (`docrooter.views.estimate_tokens`, 1.3 tokens per word) instead of each backend's own tokenizer. Each row records the SHA-256 prefix of the text the system saw (`view_sha`), and `pdf.json` lists the shared texts' hashes under `views`. Whatever a model's own tokenizer cannot fit is cut by that model as usual (bge-small and the NLI model at 512 tokens). For each system, *Views differing* counts the documents whose view would have been different had it been rendered with the backend's own counter, which was the P9 behaviour.

### Aggregate

| System | What | Leaf acc | 95% CI | Level-1 acc | Macro P | Macro R | Macro F1 | Parse s/doc | Classify s/doc | Total s/doc | Load s | Jev input tokens | Jev cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ultrabert-view` | UltraBERT-Embed-150M, DocumentView, Flat (zero-shot) | 0.825 | 0.70-0.95 | 0.875 | 0.787 | 0.815 | 0.778 | 1.21 | 0.243 (batch) | 1.45 | 10.1 | - | - |
| `ultrabert-hybrid` | UltraBERT-Embed-150M, 0.6 view + 0.4 sections (zero-shot) | 0.775 | 0.62-0.90 | 0.875 | 0.758 | 0.769 | 0.731 | 1.21 | 3.448 (batch) | 4.65 | 1.9 | - | - |
| `bge-view` | bge-small-en-v1.5, DocumentView, Flat (zero-shot) | 0.700 | 0.55-0.85 | 0.875 | 0.727 | 0.685 | 0.654 | 1.21 | 0.033 (batch) | 1.24 | 3.2 | - | - |
| `jev-view` | Jev jev-1.13.0, DocumentView as state, one Choice over 18 leaves | 0.925 | 0.82-1.00 | 0.950 | 0.963 | 0.926 | 0.930 | 1.21 | 0.061 (batch) | 1.27 | 0.5 | 79,424 (est.) | $0.0033 |
| `jev-chunks` | Jev jev-1.13.0, raw page text, <=8 chunks of ~1,500 tokens, mean of per-chunk Choices (no docrooter) | 0.975 | 0.93-1.00 | 0.975 | 1.000 | 0.981 | 0.989 | 0.35 | 0.429 (latency) | 0.78 | 0.0 | 850,373 | $0.0357 |
| `ultrabert-dev-logistic` | UltraBERT-Embed-150M view embeddings + logistic regression trained on all 51 dev docs, C in {1, 10, 100, 1000} by inner leave-one-out on dev (trained on dev, tested on another corpus) | 0.675 | 0.53-0.80 | 0.775 | 0.754 | 0.657 | 0.665 | 1.21 | 0.650 (batch) | 1.86 | 1.3 | - | - |
| `ultrabert-bge-dev-ensemble` | Zero-shot UltraBERT-view and bge-view leaf distributions mixed, weight in {0, .25, .5, .75, 1} chosen by leave-one-out on the 51 dev docs (tested on another corpus) | 0.800 | 0.65-0.93 | 0.900 | 0.773 | 0.796 | 0.753 | 1.21 | 0.721 (batch) | 1.93 | 4.2 | - | - |

Run metadata per system:

- `ultrabert-view`: 2026-10-16 at `72f6135`, device `mps`, load average 4.1, 4.1, 3.6 at start; views differing from the backend's own counter: 22 of 40.
- `ultrabert-hybrid`: 2026-10-16 at `72f6135`, device `mps`, load average 3.8, 4.0, 3.6 at start; views differing from the backend's own counter: 22 of 40.
- `bge-view`: 2026-10-16 at `72f6135`, device `mps`, load average 6.0, 4.7, 3.9 at start; views differing from the backend's own counter: 14 of 40.
- `jev-view`: 2026-10-16 at `72f6135`, device `n/a`, load average 4.0, 4.1, 3.6 at start; views differing from the backend's own counter: 0 of 40.
- `jev-chunks`: 2026-10-16 at `72f6135`, device `n/a`, load average 4.1, 4.1, 3.6 at start; 251 chunks sent (6.3 per document).
- `ultrabert-dev-logistic`: 2026-10-16 at `72f6135`, device `mps`, load average 5.6, 4.7, 3.9 at start; views differing from the backends' own counters: UltraBERT 22 of 40.
- `ultrabert-bge-dev-ensemble`: 2026-10-16 at `72f6135`, device `mps`, load average 4.6, 4.5, 3.9 at start; views differing from the backends' own counters: UltraBERT 22, bge 14 of 40.

### Per class

Best honest docrooter system by macro F1 (zero-shot: `ultrabert-view`; any, leave-one-out included: `ultrabert-view`), `jev-view`. P / R / F1 per gold class.

| Class | n | `ultrabert-view` P / R / F1 | `jev-view` P / R / F1 |
| --- | --- | --- | --- |
| financial.annual_report | 2 | 0.67 / 1.00 / 0.80 | 1.00 / 1.00 / 1.00 |
| financial.quarterly_report | 2 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| financial.shareholder_letter | 2 | 1.00 / 0.50 / 0.67 | 1.00 / 0.50 / 0.67 |
| financial.earnings_release | 2 | 0.67 / 1.00 / 0.80 | 1.00 / 1.00 / 1.00 |
| financial.prospectus | 2 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| financial.proxy_statement | 2 | 0.67 / 1.00 / 0.80 | 0.67 / 1.00 / 0.80 |
| legal.contract | 2 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| legal.terms_policy | 2 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| legal.court_opinion | 3 | 1.00 / 0.67 / 0.80 | 1.00 / 0.67 / 0.80 |
| legal.legislation | 3 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| research.research_paper | 2 | 0.50 / 1.00 / 0.67 | 1.00 / 1.00 / 1.00 |
| research.technical_standard | 3 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| research.white_paper | 2 | 0.00 / 0.00 / 0.00 | 0.67 / 1.00 / 0.80 |
| technical.user_manual | 2 | 1.00 / 0.50 / 0.67 | 1.00 / 1.00 / 1.00 |
| technical.datasheet | 3 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| government.government_report | 2 | 0.67 / 1.00 / 0.80 | 1.00 / 1.00 / 1.00 |
| corporate.sustainability_report | 2 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| corporate.investor_presentation | 2 | 0.00 / 0.00 / 0.00 | 1.00 / 0.50 / 0.67 |

### Confusion matrices

Rows are gold, columns are predictions, numbered as the rows. Empty cells are 0; the diagonal is in bold.

`ultrabert-view`:

| Gold \ Pred | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | - |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 annual_report | **2** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2 quarterly_report |  | **2** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 3 shareholder_letter |  |  | **1** |  |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 4 earnings_release |  |  |  | **2** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 5 prospectus |  |  |  |  | **2** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 6 proxy_statement |  |  |  |  |  | **2** |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 7 contract |  |  |  |  |  |  | **2** |  |  |  |  |  |  |  |  |  |  |  |  |
| 8 terms_policy |  |  |  |  |  |  |  | **2** |  |  |  |  |  |  |  |  |  |  |  |
| 9 court_opinion |  |  |  |  |  |  |  |  | **2** |  |  |  |  |  |  |  |  |  | 1 |
| 10 legislation |  |  |  |  |  |  |  |  |  | **3** |  |  |  |  |  |  |  |  |  |
| 11 research_paper |  |  |  |  |  |  |  |  |  |  | **2** |  |  |  |  |  |  |  |  |
| 12 technical_standard |  |  |  |  |  |  |  |  |  |  |  | **3** |  |  |  |  |  |  |  |
| 13 white_paper |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  |  | 1 |  |  |  |
| 14 user_manual |  |  |  |  |  |  |  |  |  |  | 1 |  |  | **1** |  |  |  |  |  |
| 15 datasheet |  |  |  |  |  |  |  |  |  |  |  |  |  |  | **3** |  |  |  |  |
| 16 government_report |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | **2** |  |  |  |
| 17 sustainability_report |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | **2** |  |  |
| 18 investor_presentation | 1 |  |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

`jev-view`:

| Gold \ Pred | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | - |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 annual_report | **2** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2 quarterly_report |  | **2** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 3 shareholder_letter |  |  | **1** |  |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 4 earnings_release |  |  |  | **2** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 5 prospectus |  |  |  |  | **2** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 6 proxy_statement |  |  |  |  |  | **2** |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 7 contract |  |  |  |  |  |  | **2** |  |  |  |  |  |  |  |  |  |  |  |  |
| 8 terms_policy |  |  |  |  |  |  |  | **2** |  |  |  |  |  |  |  |  |  |  |  |
| 9 court_opinion |  |  |  |  |  |  |  |  | **2** |  |  |  |  |  |  |  |  |  | 1 |
| 10 legislation |  |  |  |  |  |  |  |  |  | **3** |  |  |  |  |  |  |  |  |  |
| 11 research_paper |  |  |  |  |  |  |  |  |  |  | **2** |  |  |  |  |  |  |  |  |
| 12 technical_standard |  |  |  |  |  |  |  |  |  |  |  | **3** |  |  |  |  |  |  |  |
| 13 white_paper |  |  |  |  |  |  |  |  |  |  |  |  | **2** |  |  |  |  |  |  |
| 14 user_manual |  |  |  |  |  |  |  |  |  |  |  |  |  | **2** |  |  |  |  |  |
| 15 datasheet |  |  |  |  |  |  |  |  |  |  |  |  |  |  | **3** |  |  |  |  |
| 16 government_report |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | **2** |  |  |  |
| 17 sustainability_report |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | **2** |  |  |
| 18 investor_presentation |  |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  |  | **1** |  |

### Per document

✓ correct leaf, ✗ wrong. Predictions show the leaf id without its group.

| Doc | Title | Pages | Gold | `ultrabert-view` | `ultrabert-hybrid` | `bge-view` | `jev-view` | `jev-chunks` | `ultrabert-dev-logistic` | `ultrabert-bge-dev-ensemble` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ho_001 | Rio Tinto Annual Report 2023 | 360 | annual_report | annual_report ✓ | sustainability_report ✗ | annual_report ✓ | annual_report ✓ | annual_report ✓ | annual_report ✓ | annual_report ✓ |
| ho_002 | The Collinson Group Limited Annual report and... | 103 | annual_report | annual_report ✓ | annual_report ✓ | annual_report ✓ | annual_report ✓ | annual_report ✓ | annual_report ✓ | annual_report ✓ |
| ho_003 | Roche Half-Year Report 2024 | 91 | quarterly_report | quarterly_report ✓ | quarterly_report ✓ | quarterly_report ✓ | quarterly_report ✓ | quarterly_report ✓ | investor_presentation ✗ | quarterly_report ✓ |
| ho_004 | Planet 13 Holdings Inc. Form 10-Q for the qua... | 66 | quarterly_report | quarterly_report ✓ | quarterly_report ✓ | annual_report ✗ | quarterly_report ✓ | quarterly_report ✓ | quarterly_report ✓ | quarterly_report ✓ |
| ho_005 | Markel Financial Holdings Limited (now Fairfa... | 1 | shareholder_letter | shareholder_letter ✓ | shareholder_letter ✓ | shareholder_letter ✓ | shareholder_letter ✓ | shareholder_letter ✓ | shareholder_letter ✓ | shareholder_letter ✓ |
| ho_006 | ADP Letters to Shareholders 2025 (Board Chair... | 6 | shareholder_letter | proxy_statement ✗ | proxy_statement ✗ | proxy_statement ✗ | proxy_statement ✗ | shareholder_letter ✓ | annual_report ✗ | proxy_statement ✗ |
| ho_007 | Pinterest Announces Second Quarter 2024 Results | 11 | earnings_release | earnings_release ✓ | earnings_release ✓ | earnings_release ✓ | earnings_release ✓ | earnings_release ✓ | investor_presentation ✗ | earnings_release ✓ |
| ho_008 | International Paper Reports Second Quarter 20... | 11 | earnings_release | earnings_release ✓ | earnings_release ✓ | earnings_release ✓ | earnings_release ✓ | earnings_release ✓ | investor_presentation ✗ | earnings_release ✓ |
| ho_009 | Vanguard Ultra-Short Bond ETF Prospectus (Apr... | 40 | prospectus | prospectus ✓ | prospectus ✓ | prospectus ✓ | prospectus ✓ | prospectus ✓ | prospectus ✓ | prospectus ✓ |
| ho_010 | HSBC Holdings plc Debt Issuance Programme Bas... | 96 | prospectus | prospectus ✓ | prospectus ✓ | prospectus ✓ | prospectus ✓ | prospectus ✓ | prospectus ✓ | prospectus ✓ |
| ho_011 | CDW Corporation Notice of 2024 Annual Meeting... | 81 | proxy_statement | proxy_statement ✓ | proxy_statement ✓ | proxy_statement ✓ | proxy_statement ✓ | proxy_statement ✓ | annual_report ✗ | proxy_statement ✓ |
| ho_012 | Brixmor Property Group 2024 Proxy Statement a... | 96 | proxy_statement | proxy_statement ✓ | proxy_statement ✓ | proxy_statement ✓ | proxy_statement ✓ | proxy_statement ✓ | proxy_statement ✓ | proxy_statement ✓ |
| ho_013 | HMRC Standard Goods and Services Model Contra... | 91 | contract | contract ✓ | contract ✓ | proxy_statement ✗ | contract ✓ | contract ✓ | terms_policy ✗ | contract ✓ |
| ho_014 | Valencia College Contractor's Agreement template | 6 | contract | contract ✓ | contract ✓ | contract ✓ | contract ✓ | contract ✓ | contract ✓ | contract ✓ |
| ho_015 | Cookies on vmd.defra.gov.uk (cookie policy) | 2 | terms_policy | terms_policy ✓ | terms_policy ✓ | terms_policy ✓ | terms_policy ✓ | terms_policy ✓ | white_paper ✗ | terms_policy ✓ |
| ho_016 | CatalyST Privacy and cookie policy (May 2018) | 3 | terms_policy | terms_policy ✓ | terms_policy ✓ | proxy_statement ✗ | terms_policy ✓ | terms_policy ✓ | terms_policy ✓ | terms_policy ✓ |
| ho_017 | Plessy v. Ferguson, 163 U.S. 537 (1896) | 28 | court_opinion | court_opinion ✓ | legislation ✗ | contract ✗ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | legislation ✗ |
| ho_018 | M'Alister (or Donoghue) v Stevenson [1932] AC... | 62 | court_opinion | (none) ✗ | (none) ✗ | (none) ✗ | (none) ✗ | (none) ✗ | (none) ✗ | (none) ✗ |
| ho_019 | Brown v. Board of Education of Topeka, 347 U.... | 14 | court_opinion | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ |
| ho_020 | Data Protection Act 2018 (Number 7 of 2018, I... | 184 | legislation | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ |
| ho_021 | Personal Information Protection and Electroni... | 71 | legislation | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | terms_policy ✗ | legislation ✓ |
| ho_022 | Privacy Act 1988 (No. 119, 1988), Compilation... | 448 | legislation | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ |
| ho_023 | Generative Adversarial Nets (arXiv:1406.2661) | 9 | research_paper | research_paper ✓ | research_paper ✓ | research_paper ✓ | research_paper ✓ | research_paper ✓ | research_paper ✓ | research_paper ✓ |
| ho_024 | A Mathematical Theory of Communication (Bell ... | 55 | research_paper | research_paper ✓ | research_paper ✓ | technical_standard ✗ | research_paper ✓ | research_paper ✓ | shareholder_letter ✗ | research_paper ✓ |
| ho_025 | ECMA-262, 15th edition, June 2024: ECMAScript... | 816 | technical_standard | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ |
| ho_026 | ECMA-404, 2nd edition, December 2017: The JSO... | 16 | technical_standard | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ |
| ho_027 | ETSI EN 303 645 V2.1.1 (2020-06): Cyber Secur... | 34 | technical_standard | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ |
| ho_028 | Uniswap v3 Core (March 2021) | 9 | white_paper | research_paper ✗ | research_paper ✗ | earnings_release ✗ | white_paper ✓ | white_paper ✓ | white_paper ✓ | research_paper ✗ |
| ho_029 | Ethereum Basics for Governments and Institutions | 71 | white_paper | government_report ✗ | research_paper ✗ | technical_standard ✗ | white_paper ✓ | white_paper ✓ | white_paper ✓ | government_report ✗ |
| ho_030 | An Introduction to R (Version 4.6.1) | 103 | user_manual | research_paper ✗ | research_paper ✗ | datasheet ✗ | user_manual ✓ | user_manual ✓ | user_manual ✓ | research_paper ✗ |
| ho_031 | Debian Reference | 285 | user_manual | user_manual ✓ | user_manual ✓ | user_manual ✓ | user_manual ✓ | user_manual ✓ | user_manual ✓ | user_manual ✓ |
| ho_032 | ESP32 Series Datasheet Version 5.3 | 78 | datasheet | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ |
| ho_033 | LR8 High-Input Voltage, Adjustable, 3-Termina... | 18 | datasheet | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ |
| ho_034 | BME280 Data sheet (revision 1.24, February 2024) | 60 | datasheet | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ |
| ho_035 | Monetary Policy Report - July 2024 | 29 | government_report | government_report ✓ | government_report ✓ | government_report ✓ | government_report ✓ | government_report ✓ | government_report ✓ | government_report ✓ |
| ho_036 | Police productivity (HC 1380, Session 2024-2026) | 51 | government_report | government_report ✓ | government_report ✓ | government_report ✓ | government_report ✓ | government_report ✓ | white_paper ✗ | government_report ✓ |
| ho_037 | Siemens Sustainability Report fiscal year 2023 | 158 | sustainability_report | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ |
| ho_038 | Microsoft 2024 Environmental Sustainability R... | 88 | sustainability_report | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ |
| ho_039 | Moody's Corporation 1Q 2024 Investor Presenta... | 70 | investor_presentation | annual_report ✗ | annual_report ✗ | sustainability_report ✗ | investor_presentation ✓ | investor_presentation ✓ | annual_report ✗ | sustainability_report ✗ |
| ho_040 | Urban Edge Properties Investor Presentation, ... | 46 | investor_presentation | earnings_release ✗ | earnings_release ✗ | earnings_release ✗ | white_paper ✗ | investor_presentation ✓ | shareholder_letter ✗ | earnings_release ✗ |

## Section normalisation (held-out)

Silver set: `benchmarks/heldout/section_gold.jsonl`, 157 level-1/2 sections of 4 documents (24 marked as parser junk, i.e. not a real section, and excluded from accuracy). **These are agent-assigned silver labels, not human gold:** an agent (Claude Opus 5.5) read each heading and the first ~300 characters of its body and chose the best node of the doc type's section taxonomy from the node definitions, before any normalisation run (labelled 2026-10-16T19:55:59-04:00, documents loaded at `42c1f33`). Ambiguous cases carry a `note` in the file. Expect a few points of label noise.

Run 2026-10-16 at `72f6135`, device `mps` (MPS probe matmul took 0.18 s). Each system calls `normalize_sections(doc, doc_type=<gold type>, backend=...)` once per document with the package defaults: Flat, view path + title at 96 tokens, with the taxonomies' heading synonyms. *Macro F1* averages over the gold classes (per document over that document's classes; overall over all classes, qualified by taxonomy). *s/section* is the summed wall time of the per-document calls (view rendering and label encoding included) divided by the sections labelled, junk included. Model load and the label encoding of every section taxonomy the run can use (one untimed warm-up per doc type) are excluded (*Load s*). Jev cost is input tokens x $0.042/M as reported by the API.

**Synonym overlap.** 14 of the 133 scored sections have a heading equal to a heading-synonym example of their taxonomy (14 distinct headings; compared case- and punctuation-insensitively after dropping one leading number such as "3.1", "Item 1A." or "Note 2."). For those, the synonyms can hand the model the answer, so every accuracy below is also given on the other 119 (*non-overlap*), as *all; non-overlap*.

### Configurations tried

Only the package default (`path+title+syn`) was run on these sections, so no configuration was selected on them. Accuracy cells are *all; non-overlap*. *Top prediction* is the most frequent predicted label over the scored sections and its count (a collapse onto one label shows up here).

| Config | View | Synonyms | `ultrabert` accuracy | `bge-small` accuracy | `jev` accuracy | `ultrabert` top prediction | `bge-small` top prediction | `jev` top prediction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `path+title+syn` (default) | path + title @ 96 | yes | 0.579 (77/133); 0.538 (64/119) | 0.496 (66/133); 0.454 (54/119) | 0.692 (92/133); 0.664 (79/119) | `sustainability` (20) | `management_discussion` (24) | `sustainability` (15) |

### Aggregate

| System | Backend | Accuracy | Correct | Macro F1 | s/section | Load s | Jev tokens | Jev cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ultrabert` | `embedding:dmis-lab/UltraBERT-Embed-150M` | 0.579 | 77/133 | 0.564 | 0.0161 | 13.1 | - | - |
| `bge-small` | `embedding:BAAI/bge-small-en-v1.5` | 0.496 | 66/133 | 0.555 | 0.0034 | 4.3 | - | - |
| `jev` | `jev:jev-1.13.0` | 0.692 | 92/133 | 0.707 | 0.0402 | 1.2 | 213,468 | $0.0090 |

### Per document (accuracy / macro F1)

| Document | Type | Sections | Junk | Unmatched | `ultrabert` | `bge-small` | `jev` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ho_023 Generative Adversarial Nets (arXiv:1406.2661) | `research.research_paper` | 12 | 1 | 0 | 0.545 / 0.646 | 0.545 / 0.667 | 0.727 / 0.833 |
| ho_024 A Mathematical Theory of Communication (Bell System Technical Journal, 1948) | `research.research_paper` | 7 | 1 | 0 | 0.333 / 0.500 | 0.167 / 0.111 | 0.500 / 0.600 |
| ho_001 Rio Tinto Annual Report 2023 | `financial.annual_report` | 98 | 22 | 0 | 0.605 / 0.539 | 0.513 / 0.462 | 0.697 / 0.697 |
| ho_013 HMRC Standard Goods and Services Model Contract - Terms and Conditions | `legal.contract` | 40 | 0 | 0 | 0.575 / 0.564 | 0.500 / 0.633 | 0.700 / 0.655 |

### Auto doc type (`doc_type=None`)

One run of `normalize_sections(doc)` with the default backend (`embedding:dmis-lab/UltraBERT-Embed-150M`), so `classify_document` picks the doc type and hence the section taxonomy. The doc type matched the gold type on **3/4** documents; section accuracy 0.406 (54/133), macro F1 0.394 (a wrong doc type usually means a different taxonomy, so its sections all count as wrong).

| Document | Gold type | Auto type | Confidence | Taxonomy | Accuracy |
| --- | --- | --- | --- | --- | --- |
| ho_023 | `research.research_paper` | ✓ `research.research_paper` | 0.608 | `section_types/research_paper` | 0.545 |
| ho_024 | `research.research_paper` | ✓ `research.research_paper` | 0.365 | `section_types/research_paper` | 0.333 |
| ho_001 | `financial.annual_report` | ✓ `financial.annual_report` | 0.234 | `section_types/annual_report` | 0.605 |
| ho_013 | `legal.contract` | ✗ `legal.terms_policy` | 0.210 | `section_types/generic` | 0.000 |

### Per entry: ho_023 Generative Adversarial Nets (arXiv:1406.2661)

| Page | Level | Section | Gold | `ultrabert` | `bge-small` | `jev` |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | Generative Adversarial Nets | *(junk)* | `abstract` | `experiments` | `method` |
| 1 | 1 | Abstract | `abstract` | ✓ `abstract` | ✓ `abstract` | ✓ `abstract` |
| 1 | 1 | 1 Introduction | `introduction` | ✓ `introduction` | ✓ `introduction` | ✓ `introduction` |
| 2 | 1 | 2 Related work | `related_work` | ✓ `related_work` | ✓ `related_work` | ✓ `related_work` |
| 2 | 1 | 3 Adversarial nets | `method` | ✗ `abstract` | ✗ `experiments` | ✓ `method` |
| 3 | 1 | 4 Theoretical Results | `method` | ✗ `results` | ✗ `results` | ✗ `results` |
| 4 | 2 | 4.1 Global Optimality of pg = pdata | `method` | ✗ `background` | ✗ `background` | ✗ `discussion` |
| 5 | 2 | 4.2 Convergence of Algorithm 1 | `method` | ✗ `abstract` | ✗ `background` | ✓ `method` |
| 5 | 1 | 5 Experiments | `experiments` | ✓ `experiments` | ✓ `experiments` | ✓ `experiments` |
| 7 | 1 | 6 Advantages and disadvantages | `discussion` | ✗ `introduction` | ✗ `introduction` | ✗ `limitations` |
| 7 | 1 | 7 Conclusions and future work | `conclusion` | ✓ `conclusion` | ✓ `conclusion` | ✓ `conclusion` |
| 8 | 1 | References | `references` | ✓ `references` | ✓ `references` | ✓ `references` |

### Per entry: ho_024 A Mathematical Theory of Communication (Bell System Technical Journal, 1948)

| Page | Level | Section | Gold | `ultrabert` | `bge-small` | `jev` |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | A Mathematical Theory of Communication | `introduction` | ✓ `introduction` | ✗ `discussion` | ✓ `introduction` |
| 3 | 2 | PART I: DISCRETE NOISELESS SYSTEMS | `method` | ✗ `background` | ✗ `background` | ✗ `background` |
| 19 | 2 | PART II: THE DISCRETE CHANNEL WITH NOISE | `method` | ✗ `background` | ✗ `background` | ✗ `background` |
| 31 | 1 | ∑Pi | *(junk)* | `background` | `background` | `background` |
| 32 | 2 | PART III: MATHEMATICAL PRELIMINARIES | `background` | ✓ `background` | ✓ `background` | ✓ `background` |
| 41 | 2 | PART IV: THE CONTINUOUS CHANNEL | `method` | ✗ `conclusion` | ✗ `background` | ✗ `background` |
| 47 | 2 | PART V: THE RATE FOR A CONTINUOUS SOURCE | `method` | ✗ `appendix` | ✗ `background` | ✓ `method` |

### Per entry: ho_001 Rio Tinto Annual Report 2023

| Page | Level | Section | Gold | `ultrabert` | `bge-small` | `jev` |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 1 | Contents | `other` | ✓ `other` | ✓ `other` | ✓ `other` |
| 2 | 1 | 2023 year in review | *(junk)* | `other` | `other` | `shareholder_letter` |
| 3 | 2 | $54.0bn | *(junk)* | `management_discussion` | `management_discussion` | `other` |
| 3 | 1 | 2023 year in review | `other` | ✓ `other` | ✓ `other` | ✗ `shareholder_letter` |
| 3 | 2 | $10.1bn | *(junk)* | `business_overview` | `notes_to_financial_statements` | `shareholder_letter` |
| 4 | 1 | At a glance | `business_overview` | ✓ `business_overview` | ✗ `exhibits` | ✗ `other` |
| 4 | 2 | Our business | `business_overview` | ✓ `business_overview` | ✓ `business_overview` | ✓ `business_overview` |
| 6 | 1 | Chair’s statement | `shareholder_letter` | ✓ `shareholder_letter` | ✗ `other` | ✓ `shareholder_letter` |
| 8 | 2 | Q&A with Jakob Stausholm | `shareholder_letter` | ✓ `shareholder_letter` | ✗ `other` | ✓ `shareholder_letter` |
| 8 | 1 | $15.2bn | *(junk)* | `business_overview` | `market_for_equity` | `other` |
| 8 | 1 | $10.1bn | *(junk)* | `business_overview` | `notes_to_financial_statements` | `other` |
| 8 | 1 | $7.1bn | *(junk)* | `management_discussion` | `management_discussion` | `other` |
| 10 | 2 | Creating value by living our purpose | `business_overview` | ✗ `sustainability` | ✗ `management_discussion` | ✓ `business_overview` |
| 12 | 2 | Finding better ways™ in 2023 | `business_overview` | ✗ `sustainability` | ✗ `management_discussion` | ✗ `shareholder_letter` |
| 12 | 2 | Our purpose in action | `business_overview` | ✗ `management_discussion` | ✗ `management_discussion` | ✗ `sustainability` |
| 13 | 1 | 5Mt | *(junk)* | `sustainability` | `other` | `market_for_equity` |
| 14 | 1 | Our stakeholders | `sustainability` | ✓ `sustainability` | ✓ `sustainability` | ✓ `sustainability` |
| 14 | 1 | A$725m | *(junk)* | `management_discussion` | `management_discussion` | `other` |
| 14 | 1 | $76bn | *(junk)* | `business_overview` | `management_discussion` | `other` |
| 14 | 1 | $4.6bn² | *(junk)* | `properties` | `market_for_equity` | `other` |
| 14 | 1 | $4.1bn² | *(junk)* | `properties` | `notes_to_financial_statements` | `other` |
| 15 | 1 | $20.8bn $7.1bn | *(junk)* | `management_discussion` | `management_discussion` | `other` |
| 15 | 1 | $54.0bn | *(junk)* | `business_overview` | `management_discussion` | `other` |
| 15 | 1 | 26Mt CO2e | *(junk)* | `sustainability` | `sustainability` | `sustainability` |
| 15 | 1 | 543Mt CO2e | *(junk)* | `sustainability` | `sustainability` | `sustainability` |
| 16 | 1 | Strategic context | `business_overview` | ✓ `business_overview` | ✓ `business_overview` | ✓ `business_overview` |
| 18 | 2 | Our strategy | `business_overview` | ✓ `business_overview` | ✓ `business_overview` | ✓ `business_overview` |
| 19 | 1 | Progressing our strategy | `business_overview` | ✓ `business_overview` | ✗ `management_discussion` | ✓ `business_overview` |
| 20 | 1 | Progressing our four objectives | `business_overview` | ✓ `business_overview` | ✗ `other` | ✓ `business_overview` |
| 20 | 2 | Best operator | `business_overview` | ✓ `business_overview` | ✗ `management_discussion` | ✗ `management_discussion` |
| 22 | 1 | Key performance indicators | `management_discussion` | ✗ `sustainability` | ✓ `management_discussion` | ✗ `other` |
| 25 | 1 | Chief Financial Officer’s statement | `management_discussion` | ✗ `shareholder_letter` | ✗ `financial_statements` | ✓ `management_discussion` |
| 25 | 2 | $15.2 billion | *(junk)* | `shareholder_letter` | `financial_statements` | `management_discussion` |
| 26 | 1 | Financial review | `management_discussion` | ✓ `management_discussion` | ✓ `management_discussion` | ✓ `management_discussion` |
| 31 | 2 | 60% payout ratio on the ordinary dividend delivers an eight-year track | `market_for_equity` | ✗ `financial_statements` | ✓ `market_for_equity` | ✓ `market_for_equity` |
| 32 | 1 | Portfolio management | `management_discussion` | ✗ `market_for_equity` | ✓ `management_discussion` | ✗ `other` |
| 34 | 1 | Iron Ore | `business_overview` | ✓ `business_overview` | ✗ `management_discussion` | ✓ `business_overview` |
| 34 | 2 | $20.0bn $32.2bn | *(junk)* | `management_discussion` | `management_discussion` | `notes_to_financial_statements` |
| 36 | 1 | Aluminium | `business_overview` | ✓ `business_overview` | ✗ `management_discussion` | ✗ `other` |
| 36 | 2 | $2.3bn $12.3bn | *(junk)* | `financial_statements` | `market_for_equity` | `other` |
| 38 | 1 | Copper | `business_overview` | ✓ `business_overview` | ✗ `management_discussion` | ✗ `other` |
| 38 | 2 | $1.9bn $6.7bn | *(junk)* | `management_discussion` | `market_for_equity` | `other` |
| 40 | 1 | Minerals | `business_overview` | ✗ `management_discussion` | ✗ `properties` | ✗ `other` |
| 40 | 2 | $1.4bn $5.9bn | *(junk)* | `management_discussion` | `market_for_equity` | `other` |
| 42 | 1 | Our approach to ESG | `sustainability` | ✓ `sustainability` | ✓ `sustainability` | ✓ `sustainability` |
| 43 | 2 | Our ESG framework | `sustainability` | ✓ `sustainability` | ✓ `sustainability` | ✓ `sustainability` |
| 46 | 1 | Environmental performance | `sustainability` | ✓ `sustainability` | ✓ `sustainability` | ✓ `sustainability` |
| 46 | 2 | Climate change | `sustainability` | ✓ `sustainability` | ✓ `sustainability` | ✓ `sustainability` |
| 61 | 2 | Environmental stewardship | `sustainability` | ✓ `sustainability` | ✓ `sustainability` | ✓ `sustainability` |
| 66 | 2 | Mining and metals practices | `sustainability` | ✓ `sustainability` | ✓ `sustainability` | ✓ `sustainability` |
| 66 | 1 | $17.2bn | *(junk)* | `business_overview` | `market_for_equity` | `other` |
| 68 | 1 | Social performance | `sustainability` | ✓ `sustainability` | ✓ `sustainability` | ✓ `sustainability` |
| 68 | 2 | Community engagement and social investment | `sustainability` | ✓ `sustainability` | ✓ `sustainability` | ✓ `sustainability` |
| 73 | 2 | Health, safety and wellbeing | `sustainability` | ✓ `sustainability` | ✓ `sustainability` | ✓ `sustainability` |
| 75 | 2 | Talent, diversity and inclusion | `sustainability` | ✓ `sustainability` | ✓ `sustainability` | ✓ `sustainability` |
| 77 | 2 | Human rights | `sustainability` | ✓ `sustainability` | ✓ `sustainability` | ✓ `sustainability` |
| 78 | 1 | Governance performance | `corporate_governance` | ✓ `corporate_governance` | ✓ `corporate_governance` | ✓ `corporate_governance` |
| 78 | 2 | 25,187 employees | *(junk)* | `corporate_governance` | `corporate_governance` | `sustainability` |
| 80 | 2 | Our approach to risk | `risk_factors` | ✓ `risk_factors` | ✗ `management_discussion` | ✗ `corporate_governance` |
| 81 | 2 | Our approach to risk management | `risk_factors` | ✓ `risk_factors` | ✗ `management_discussion` | ✗ `corporate_governance` |
| 83 | 2 | Principal risks and uncertainties | `risk_factors` | ✓ `risk_factors` | ✓ `risk_factors` | ✓ `risk_factors` |
| 91 | 1 | Five-year review | `other` | ✗ `business_overview` | ✗ `management_discussion` | ✓ `other` |
| 93 | 2 | Chair’s introduction | `corporate_governance` | ✗ `shareholder_letter` | ✗ `management_discussion` | ✗ `shareholder_letter` |
| 94 | 2 | Board of Directors | `corporate_governance` | ✓ `corporate_governance` | ✓ `corporate_governance` | ✓ `corporate_governance` |
| 96 | 2 | Executive Committee | `corporate_governance` | ✓ `corporate_governance` | ✓ `corporate_governance` | ✓ `corporate_governance` |
| 98 | 2 | Our stakeholders | `corporate_governance` | ✗ `sustainability` | ✗ `sustainability` | ✗ `sustainability` |
| 103 | 2 | How the Board monitors culture | `corporate_governance` | ✓ `corporate_governance` | ✓ `corporate_governance` | ✓ `corporate_governance` |
| 104 | 2 | Board activities in 2023 | `corporate_governance` | ✓ `corporate_governance` | ✓ `corporate_governance` | ✓ `corporate_governance` |
| 105 | 2 | Governance framework | `corporate_governance` | ✓ `corporate_governance` | ✓ `corporate_governance` | ✓ `corporate_governance` |
| 106 | 2 | Evaluating our performance | `corporate_governance` | ✗ `management_discussion` | ✗ `management_discussion` | ✗ `management_discussion` |
| 107 | 2 | Nominations Committee report | `corporate_governance` | ✓ `corporate_governance` | ✓ `corporate_governance` | ✓ `corporate_governance` |
| 109 | 2 | Audit & Risk Committee report | `corporate_governance` | ✗ `risk_factors` | ✓ `corporate_governance` | ✓ `corporate_governance` |
| 113 | 2 | Sustainability Committee report | `corporate_governance` | ✗ `sustainability` | ✗ `sustainability` | ✗ `sustainability` |
| 114 | 2 | Sustainability Committee members | `corporate_governance` | ✗ `sustainability` | ✗ `sustainability` | ✓ `corporate_governance` |
| 115 | 2 | Remuneration report | `executive_compensation` | ✓ `executive_compensation` | ✓ `executive_compensation` | ✓ `executive_compensation` |
| 148 | 2 | Additional statutory disclosure | `other` | ✗ `sustainability` | ✗ `financial_statements` | ✓ `other` |
| 154 | 2 | Compliance with governance codes and standards | `corporate_governance` | ✓ `corporate_governance` | ✓ `corporate_governance` | ✓ `corporate_governance` |
| 159 | 2 | 2023 Financial Statements | `financial_statements` | ✓ `financial_statements` | ✓ `financial_statements` | ✓ `financial_statements` |
| 160 | 2 | About Rio Tinto | `notes_to_financial_statements` | ✗ `business_overview` | ✗ `management_discussion` | ✗ `business_overview` |
| 160 | 2 | About the presentation of our financial statements | `notes_to_financial_statements` | ✗ `financial_statements` | ✗ `financial_statements` | ✓ `notes_to_financial_statements` |
| 170 | 2 | Group Income Statement | `financial_statements` | ✓ `financial_statements` | ✓ `financial_statements` | ✓ `financial_statements` |
| 171 | 2 | Group Statement of Comprehensive Income | `financial_statements` | ✓ `financial_statements` | ✓ `financial_statements` | ✓ `financial_statements` |
| 172 | 2 | Group Cash Flow Statement | `financial_statements` | ✓ `financial_statements` | ✓ `financial_statements` | ✓ `financial_statements` |
| 173 | 2 | Group Balance Sheet | `financial_statements` | ✓ `financial_statements` | ✓ `financial_statements` | ✓ `financial_statements` |
| 174 | 2 | Group Statement of Changes in Equity | `financial_statements` | ✗ `market_for_equity` | ✗ `market_for_equity` | ✓ `financial_statements` |
| 267 | 2 | Australian Corporations Act – Summary of ASIC Relief | `other` | ✗ `notes_to_financial_statements` | ✗ `corporate_governance` | ✓ `other` |
| 268 | 2 | Directors’ declaration | `financial_statements` | ✗ `corporate_governance` | ✗ `corporate_governance` | ✗ `other` |
| 288 | 2 | Rio Tinto Financial Information by Business Unit | `other` | ✗ `business_overview` | ✗ `management_discussion` | ✗ `notes_to_financial_statements` |
| 297 | 2 | Financial summary 2014-2023 | `other` | ✗ `financial_statements` | ✗ `notes_to_financial_statements` | ✗ `financial_statements` |
| 298 | 2 | Production, Ore Reserves, Mineral Resources and Operations | `properties` | ✗ `management_discussion` | ✗ `management_discussion` | ✗ `business_overview` |
| 299 | 2 | Metals and minerals production | `properties` | ✗ `management_discussion` | ✗ `management_discussion` | ✗ `business_overview` |
| 301 | 2 | Mineral Resources and Ore Reserves | `properties` | ✗ `management_discussion` | ✗ `management_discussion` | ✓ `properties` |
| 302 | 2 | Ore Reserves | `properties` | ✗ `management_discussion` | ✗ `management_discussion` | ✓ `properties` |
| 314 | 2 | Mineral Resources | `properties` | ✗ `management_discussion` | ✗ `management_discussion` | ✓ `properties` |
| 325 | 2 | Competent Persons | `properties` | ✗ `sustainability` | ✗ `corporate_governance` | ✗ `corporate_governance` |
| 326 | 2 | Mines and production facilities | `properties` | ✗ `management_discussion` | ✗ `management_discussion` | ✓ `properties` |
| 345 | 1 | Additional information | `other` | ✓ `other` | ✓ `other` | ✓ `other` |
| 349 | 2 | Shareholder information | `market_for_equity` | ✓ `market_for_equity` | ✓ `market_for_equity` | ✓ `market_for_equity` |

### Per entry: ho_013 HMRC Standard Goods and Services Model Contract - Terms and Conditions

| Page | Level | Section | Gold | `ultrabert` | `bge-small` | `jev` |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 1 | SECTION A - PRELIMINARIES | `definitions` | ✗ `parties_recitals` | ✗ `parties_recitals` | ✗ `parties_recitals` |
| 8 | 1 | SECTION B – THE SERVICES | `scope_services` | ✓ `scope_services` | ✓ `scope_services` | ✓ `scope_services` |
| 15 | 1 | SECTION C – PAYMENT, TAXATION AND VALUE FOR MONEY PROVISIONS | `payment` | ✓ `payment` | ✓ `payment` | ✓ `payment` |
| 20 | 1 | SECTION D - CONTRACT GOVERNANCE 11 GOVERNANCE | `general_provisions` | ✗ `governing_law_disputes` | ✗ `governing_law_disputes` | ✗ `governing_law_disputes` |
| 20 | 1 | 12 RECORDS, REPORTS, AUDITS & OPEN BOOK DATA Audit Rights | `general_provisions` | ✗ `signatures_exhibits` | ✗ `signatures_exhibits` | ✓ `general_provisions` |
| 24 | 1 | 13 CHANGE Change Control Procedure | `general_provisions` | ✓ `general_provisions` | ✗ `signatures_exhibits` | ✓ `general_provisions` |
| 24 | 1 | SECTION E – SUPPLIER PERSONNEL AND SUPPLY CHAIN 14 SUPPLIER PERSONNEL | `scope_services` | ✓ `scope_services` | ✓ `scope_services` | ✓ `scope_services` |
| 26 | 1 | 15 SUPPLY CHAIN RIGHTS AND PROTECTIONS | `general_provisions` | ✗ `intellectual_property` | ✗ `intellectual_property` | ✓ `general_provisions` |
| 28 | 1 | SECTION F - INTELLECTUAL PROPERTY, DATA AND CONFIDENTIALITY 16 INTELLE | `intellectual_property` | ✓ `intellectual_property` | ✓ `intellectual_property` | ✓ `intellectual_property` |
| 28 | 2 | 17.1 The Supplier hereby grants to the Authority and the Government Co | `intellectual_property` | ✓ `intellectual_property` | ✓ `intellectual_property` | ✓ `intellectual_property` |
| 33 | 1 | 21 CONFIDENTIALITY | `confidentiality` | ✓ `confidentiality` | ✓ `confidentiality` | ✓ `confidentiality` |
| 35 | 1 | 22 TRANSPARENCY AND FREEDOM OF INFORMATION | `confidentiality` | ✓ `confidentiality` | ✓ `confidentiality` | ✗ `general_provisions` |
| 36 | 1 | 23 PROTECTION OF PERSONAL DATA | `confidentiality` | ✓ `confidentiality` | ✓ `confidentiality` | ✓ `confidentiality` |
| 36 | 2 | 23.1 The Parties shall comply with Schedule 2.8 (Data Processing and L | `confidentiality` | ✓ `confidentiality` | ✓ `confidentiality` | ✓ `confidentiality` |
| 37 | 1 | SECTION G - LIABILITY, INDEMNITIES AND INSURANCE | `limitation_of_liability` | ✗ `indemnification` | ✗ `indemnification` | ✗ `indemnification` |
| 38 | 2 | 25.2 Subject to Clause 25.1 (Unlimited Liability) and Clauses 25.5 and | `limitation_of_liability` | ✓ `limitation_of_liability` | ✓ `limitation_of_liability` | ✓ `limitation_of_liability` |
| 39 | 2 | 25.4 Subject to Clause 25.1 (Unlimited Liability) and Clause 25.5 (Con | `limitation_of_liability` | ✓ `limitation_of_liability` | ✓ `limitation_of_liability` | ✓ `limitation_of_liability` |
| 40 | 1 | 26 INSURANCE | `general_provisions` | ✓ `general_provisions` | ✗ `signatures_exhibits` | ✓ `general_provisions` |
| 40 | 1 | SECTION H – REMEDIES AND RELIEF | `scope_services` | ✗ `general_provisions` | ✗ `general_provisions` | ✗ `limitation_of_liability` |
| 41 | 2 | 27.3 If: | `scope_services` | ✗ `general_provisions` | ✗ `general_provisions` | ✗ `limitation_of_liability` |
| 42 | 2 | 28.2 If in any Service Period: | `payment` | ✗ `general_provisions` | ✗ `general_provisions` | ✗ `governing_law_disputes` |
| 43 | 1 | 29 ESCALATION PROCESS | `governing_law_disputes` | ✗ `payment` | ✗ `signatures_exhibits` | ✓ `governing_law_disputes` |
| 45 | 1 | 31 AUTHORITY CAUSE | `general_provisions` | ✗ `signatures_exhibits` | ✗ `signatures_exhibits` | ✓ `general_provisions` |
| 46 | 1 | 32 FORCE MAJEURE | `general_provisions` | ✓ `general_provisions` | ✓ `general_provisions` | ✓ `general_provisions` |
| 48 | 1 | SECTION I – TERMINATION AND EXIT MANAGEMENT | `termination` | ✓ `termination` | ✓ `termination` | ✓ `termination` |
| 48 | 2 | 33.2 Where the Authority: | `termination` | ✗ `governing_law_disputes` | ✓ `termination` | ✓ `termination` |
| 50 | 2 | 34.1 The provisions of Clauses  10.9 and 10.17 (Promoting Tax Complian | `termination` | ✗ `payment` | ✓ `termination` | ✓ `termination` |
| 51 | 1 | SECTION J - MISCELLANEOUS AND GOVERNING LAW 35 COMPLIANCE Health and S | `general_provisions` | ✓ `general_provisions` | ✓ `general_provisions` | ✓ `general_provisions` |
| 52 | 1 | 36 ASSIGNMENT AND NOVATION | `general_provisions` | ✓ `general_provisions` | ✓ `general_provisions` | ✓ `general_provisions` |
| 55 | 1 | 40 SEVERANCE | `general_provisions` | ✓ `general_provisions` | ✗ `signatures_exhibits` | ✓ `general_provisions` |
| 55 | 1 | 41 FURTHER ASSURANCES | `general_provisions` | ✓ `general_provisions` | ✗ `signatures_exhibits` | ✓ `general_provisions` |
| 55 | 1 | 42 ENTIRE AGREEMENT | `general_provisions` | ✓ `general_provisions` | ✓ `general_provisions` | ✓ `general_provisions` |
| 56 | 1 | 44 NOTICES | `general_provisions` | ✓ `general_provisions` | ✗ `signatures_exhibits` | ✓ `general_provisions` |
| 58 | 1 | 46 DISPUTES | `governing_law_disputes` | ✓ `governing_law_disputes` | ✓ `governing_law_disputes` | ✓ `governing_law_disputes` |
| 58 | 2 | 47.2 Subject to Clause 45 (Disputes) and Schedule 8.4 (Dispute Resolut | `governing_law_disputes` | ✓ `governing_law_disputes` | ✓ `governing_law_disputes` | ✓ `governing_law_disputes` |
| 61 | 1 | “Authority Cause” | `definitions` | ✗ `general_provisions` | ✗ `signatures_exhibits` | ✗ `termination` |
| 75 | 1 | “Intellectual Property Rights” or | `definitions` | ✗ `intellectual_property` | ✗ `intellectual_property` | ✗ `intellectual_property` |
| 82 | 1 | “Rectification Plan Process” | `definitions` | ✗ `scope_services` | ✗ `signatures_exhibits` | ✗ `general_provisions` |
| 84 | 1 | “Services” | `definitions` | ✗ `scope_services` | ✗ `scope_services` | ✗ `scope_services` |
| 87 | 1 | “Supplier Personnel” | `definitions` | ✗ `scope_services` | ✗ `scope_services` | ✗ `scope_services` |
<!-- heldout-benchmark:end -->
