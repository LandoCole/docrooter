# PDF benchmarks

## PDF TOC extraction

<!-- toc-benchmark:begin -->
Gold: bookmarks of the 24 corpus PDFs with at least 5 bookmarks (levels shifted so the minimum is 1; items without a target page dropped). Extraction runs with `use_bookmarks=False`. A prediction matches a bookmark at normalised-title ratio >= 0.85 and pages within +/-1, one-to-one and greedy by ratio. The depth <= 2 columns keep only gold and predicted headings of level <= 2, so precision is measured at that depth too. Micro pools the counts; macro is the mean of per-document F1. Level accuracy is measured on matched pairs. Bookmark sets often omit lower-level headings, so precision is pessimistic and recall / F1 at depth <= 2 are the headline numbers.

Environment: Python 3.12.13, arm64, 8 CPUs, pymupdf 1.28.2, pdfplumber 0.11.10, pymupdf4llm 1.28.2. `os.getloadavg()` was 8.1, 6.4, 7.2 (1/5/15 min) when the latest run started. The machine was shared with other jobs, so seconds per document are noisy upper bounds; each engine ran one untimed warm-up document first. Seconds are wall-clock for one `load_pdf` call. Each engine's row keeps the run that produced it: `pymupdf` 2026-10-16 at `0e8c75e` (load 8.1, 6.4, 7.2); `pdfplumber` 2026-10-16 at `0e8c75e` (load 19.5, 9.8, 8.4); `both` 2026-10-16 at `0e8c75e` (load 15.3, 9.7, 8.5); `pymupdf4llm` 2026-10-16 at `5610ada` (load 8.1, 10.8, 14.2).

### Aggregate by engine (depth <= 2 headline, all depths for reference)

| Engine | Docs | P (d<=2) | R (d<=2) | F1 micro (d<=2) | F1 macro (d<=2) | Level acc (d<=2) | P (all) | R (all) | F1 micro (all) | F1 macro (all) | Level acc (all) | s/doc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pymupdf | 24 | 0.632 | 0.791 | 0.702 | 0.592 | 0.770 | 0.550 | 0.905 | 0.684 | 0.632 | 0.596 | 1.95 |
| pdfplumber | 24 | 0.643 | 0.788 | 0.708 | 0.596 | 0.796 | 0.575 | 0.900 | 0.702 | 0.642 | 0.606 | 9.15 |
| both | 24 | 0.656 | 0.785 | 0.715 | 0.604 | 0.770 | 0.595 | 0.900 | 0.716 | 0.656 | 0.597 | 11.75 |
| pymupdf4llm* | 8 | 0.667 | 0.564 | 0.611 | 0.517 | 0.974 | 0.615 | 0.692 | 0.651 | 0.550 | 0.912 | 15.21 |

\* `pymupdf4llm`: limited to the 8 smallest bookmarked documents (by page count); see the matched-subset table below.

### Same 8 documents for every engine (the `pymupdf4llm` subset)

| Engine | P (d<=2) | R (d<=2) | F1 micro (d<=2) | F1 macro (d<=2) | Level acc (d<=2) | F1 micro (all) | s/doc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pymupdf | 0.757 | 0.569 | 0.650 | 0.472 | 0.678 | 0.754 | 0.36 |
| pdfplumber | 0.743 | 0.559 | 0.638 | 0.467 | 0.673 | 0.723 | 1.64 |
| both | 0.776 | 0.564 | 0.653 | 0.481 | 0.675 | 0.749 | 2.18 |
| pymupdf4llm | 0.667 | 0.564 | 0.611 | 0.517 | 0.974 | 0.651 | 15.21 |

### Per document: `pymupdf` (the default engine)

| Doc | Title | Pages | Gold d<=2 | Pred d<=2 | P | R | F1 | Level acc | F1 (all) | s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| doc_001 | Berkshire Hathaway Inc. 2023 Annual Report | 152 | 9 | 32 | 0.03 | 0.11 | 0.05 | 0.00 | 0.19 | 3.49 |
| doc_003 | Amazon.com, Inc. 2023 Annual Report | 92 | 30 | 29 | 0.69 | 0.67 | 0.68 | 0.00 | 0.68 | 2.34 |
| doc_004 | Berkshire Hathaway Inc. Form 10-Q for the qua... | 56 | 22 | 14 | 0.71 | 0.45 | 0.56 | 0.00 | 0.53 | 1.20 |
| doc_005 | Berkshire Hathaway Inc. Form 10-Q for the qua... | 59 | 22 | 14 | 0.71 | 0.45 | 0.56 | 0.00 | 0.53 | 1.31 |
| doc_009 | Amazon.com Announces Second Quarter Results (... | 12 | 18 | 0 | 0.00 | 0.00 | 0.00 | n/a | 0.00 | 0.19 |
| doc_010 | Amazon.com Announces Third Quarter Results (O... | 12 | 18 | 0 | 0.00 | 0.00 | 0.00 | n/a | 0.00 | 0.19 |
| doc_027 | Attention Is All You Need | 15 | 19 | 23 | 0.83 | 1.00 | 0.90 | 1.00 | 0.92 | 0.29 |
| doc_029 | Language Models are Few-Shot Learners | 75 | 32 | 38 | 0.84 | 1.00 | 0.91 | 1.00 | 0.72 | 1.66 |
| doc_030 | On the Opportunities and Risks of Foundation ... | 214 | 39 | 40 | 0.95 | 0.97 | 0.96 | 1.00 | 0.90 | 5.68 |
| doc_031 | RFC 9110: HTTP Semantics | 194 | 30 | 158 | 0.18 | 0.93 | 0.30 | 0.00 | 0.86 | 3.32 |
| doc_032 | RFC 9000: QUIC: A UDP-Based Multiplexed and S... | 151 | 36 | 141 | 0.25 | 0.97 | 0.40 | 0.03 | 0.97 | 3.17 |
| doc_033 | RFC 9239: Updates to ECMAScript Media Types | 13 | 17 | 25 | 0.64 | 0.94 | 0.76 | 0.06 | 0.96 | 0.34 |
| doc_034 | RFC 9457: Problem Details for HTTP APIs | 16 | 19 | 25 | 0.72 | 0.95 | 0.82 | 0.06 | 0.97 | 0.25 |
| doc_038 | Bash Reference Manual, Edition 5.3 | 214 | 70 | 76 | 0.82 | 0.89 | 0.85 | 1.00 | 0.92 | 3.83 |
| doc_039 | GNU Coreutils manual | 326 | 179 | 188 | 0.95 | 0.99 | 0.97 | 1.00 | 0.98 | 4.76 |
| doc_040 | Getting Started with Raspberry Pi Pico-series | 36 | 55 | 61 | 0.90 | 1.00 | 0.95 | 1.00 | 0.93 | 1.07 |
| doc_041 | NE555 / SA555 / SE555 Precision Timers datasheet | 39 | 30 | 45 | 0.44 | 0.67 | 0.53 | 1.00 | 0.61 | 0.46 |
| doc_042 | RP2040 Datasheet | 642 | 96 | 98 | 0.97 | 0.99 | 0.98 | 0.77 | 0.44 | 7.30 |
| doc_045 | Monetary Policy Report, July 5, 2024 | 79 | 23 | 12 | 0.67 | 0.35 | 0.46 | 1.00 | 0.65 | 1.09 |
| doc_046 | Financial Stability Report, April 2024 | 69 | 13 | 14 | 0.93 | 1.00 | 0.96 | 1.00 | 0.96 | 0.72 |
| doc_048 | Google Environmental Report 2024 | 86 | 32 | 24 | 0.62 | 0.47 | 0.54 | 0.93 | 0.35 | 2.89 |
| doc_049 | NVIDIA Sustainability Report, Fiscal Year 2024 | 41 | 30 | 46 | 0.61 | 0.93 | 0.74 | 1.00 | 0.51 | 0.64 |
| doc_050 | Tesla Q2 2024 Update (investor slide deck) | 29 | 29 | 1 | 1.00 | 0.03 | 0.07 | 1.00 | 0.07 | 0.35 |
| doc_051 | NVIDIA Investor Presentation Q2 FY25 (Septemb... | 27 | 27 | 17 | 0.35 | 0.22 | 0.27 | 0.17 | 0.54 | 0.17 |
<!-- toc-benchmark:end -->

### Reading the numbers

After the P5b tuning (see the tuning log below), the default engine (`pymupdf`, what `engine="auto"` resolves to) scores 0.702 micro F1 at depth <= 2 (P 0.632, R 0.791), up from 0.478 (P 0.369, R 0.678) in P5. Macro F1 rose from 0.464 to 0.592. Level accuracy on matched pairs fell from 0.802 to 0.770 at depth <= 2, because many more headings now match (708 vs 607) and some of the new matches carry a gold level the extractor gets wrong. Over all depths it rose from 0.548 to 0.596. These figures were measured on the same 24 documents that were used to choose the rules, so they are optimistic; see the disclosure below. `pdfplumber` (0.708) and `both` (0.715) are slightly ahead on quality but take about 5x longer per document on this loaded machine. The `pymupdf4llm` row is the P5 run (2026-10-16, commit `5610ada`). It was not re-run, because that engine uses the v0.1 Markdown path and does not go through the tuned detector. On its 8-document subset the tuned `pymupdf` now leads on F1 (0.650 vs 0.611 micro) while `pymupdf4llm` keeps far better level accuracy (0.97 vs 0.68).

### Tuning log (P5b)

**Disclosure: the 24-document bookmark benchmark was used as a development set.** Each rule was chosen by inspecting this benchmark's worst documents and was kept only if it raised depth <= 2 micro F1 without lowering macro F1. Treat the numbers above as development-set scores, not held-out estimates. The rules are general (no document ids, titles or per-file thresholds), and each one has unit tests on synthetic `Line` lists in `tests/unit/pdf/`. They were still selected on this data, so some overfitting to these 24 documents is likely.

All rows are `pymupdf` with `use_bookmarks=False`. Each row is measured with the code at that commit and is cumulative. The first rule is measured against the P5 code (`5610ada`, whose `src/docrooter/pdf` is identical to `adf1c3d`).

| Step | Commit | P (d<=2) | R (d<=2) | F1 micro (d<=2) | F1 macro (d<=2) | Level acc (d<=2) | F1 micro (all) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P5 baseline | `5610ada` | 0.369 | 0.678 | 0.478 | 0.464 | 0.802 | 0.594 |
| Exclude section running heads/footers repeated on 3+ pages | `23f8cd2` | 0.590 | 0.666 | 0.626 | 0.480 | 0.799 | 0.668 |
| One-off heading styles share the next recurring style's level | `bf04ccb` | 0.590 | 0.779 | 0.671 | 0.565 | 0.768 | 0.668 |
| Drop edge-band copies of a heading from the next two pages | `94a5cbf` | 0.602 | 0.779 | 0.679 | 0.569 | 0.768 | 0.671 |
| Merge wrapped title lines within 0.6 x their size | `fbd900a` | 0.606 | 0.781 | 0.682 | 0.572 | 0.768 | 0.672 |
| Detect contents pages without inline page numbers | `29adddd` | 0.610 | 0.790 | 0.688 | 0.582 | 0.755 | 0.675 |
| *Rejected:* lower the score threshold by 0.1 when no line is larger than body text | (not committed) | 0.593 | 0.792 | 0.678 | 0.586 | 0.756 | 0.671 |
| Text set below body size is never a heading candidate | `5d3e621` | 0.631 | 0.786 | 0.700 | 0.587 | 0.768 | 0.682 |
| Repeated-title penalty counts only copies set at least as large | `50b43ba` | 0.632 | 0.791 | 0.702 | 0.592 | 0.770 | 0.684 |
| Review fix: numbered openers ("Chapter 2") keep their number; size-candidate running lines need the same height on 3 pages | `0e8c75e` | 0.632 | 0.791 | 0.702 | 0.592 | 0.770 | 0.684 |
| *Rejected:* section running lines must recur on max(3, 10% of pages) | (not committed) | 0.458 | 0.791 | 0.580 | 0.564 | 0.770 | 0.629 |

Notes on the rules:

- **Section running lines** (`23f8cd2`). An edge-band line whose text, with any leading or trailing page number removed, recurs on at least 3 pages is now excluded as a running line, even far below 30% of a long document. RP2040 (doc_042) dropped from 742 to 151 depth <= 2 predictions. This adds to the ">= 30% of pages" rule in the global constraints and does not replace it.
- **One-off styles** (`bf04ccb`). A style used by fewer than min(3, 10%) of the kept headings (cover titles, front-matter one-offs) takes the level of the next recurring style instead of a level of its own. This fixed the level-5 `Item N.` headings in doc_003, which went from F1 0.00 to 0.67. It is also the main cause of the fall in level accuracy. The 10% count threshold is unstable on documents with about 20 headings (doc_045 changes levels when a few candidates are removed).
- **Edge-band copies** (`94a5cbf`) and **wrapped titles** (`fbd900a`) are small precision and recall fixes.
- **Contents pages without page numbers** (`29adddd`) cover RFC-style numbered contents and contents rows whose page number is a separate text object. The page's own "Contents" title may now be a heading.
- **Rejected: flat-document threshold relief.** This was the P5 proposal for the earnings releases doc_009/010, where all text is body size. On those two documents it produced 18 predictions each for 1 match each, and micro F1 fell from 0.688 to 0.678. It was left uncommitted by an interrupted session, measured, and reverted.
- **Sub-body text** (`5d3e621`). No matched heading in the benchmark is set below 0.95 x body size, while 51 predictions were (table-cell fragments such as RP2040's `0 Register`, footnotes in doc_045). Recall at depth <= 2 fell by 4 matches in doc_045 only. Removing its footnotes changed the one-off style threshold there and moved four headings to level 3.
- **Size-aware repeat penalty** (`50b43ba`). Section titles that overview pages repeat in smaller type ("Net-zero carbon" in doc_048) were penalised as repeated titles at their own 50 pt opening heading.

- **Review fix round 1** (`0e8c75e`). This commit adds correctness guards that do not change the benchmark score. First, a trailing number after a numbering word (chapter, part, section, item, appendix, article, schedule, exhibit, annex, table, figure) is no longer stripped as a page number. Before, "Chapter 1/2/3" or "Item 1/2/3" set at page tops would collapse into one running-line key and be excluded. Second, a recurring edge line that is bold and at least 1.15 x body size is excluded only when its copies sit at the same height on 3 pages, so repeated slide titles survive. A proposed minimum of max(3, 10% of pages) was measured and rejected: RP2040's sections span 3-10 of its 642 pages, and with that minimum its depth <= 2 predictions rose from 98 to 523.

The spec §2.3 score weights and the 0.55 default threshold were not changed.

### Remaining failure patterns (default engine, after tuning)

- **RFC bookmarks nest every section under the title (doc_031-034).** Sections are gold level 2 and subsections gold level 3. The extractor puts sections at level 1 and subsections at level 2, so the subsections count as false positives at depth <= 2. Level accuracy is about 0 for these four documents. They account for 252 of the 413 depth <= 2 false positives. Without them, micro F1 on the other 20 documents is 0.781 (P 0.791, R 0.770). This comes from how the gold is built, not from the extractor, and it was left alone.
- **Earnings releases (doc_009/010, F1 0.00).** All text is body size. Most gold titles (`Page 1 Financials (QTD)`, `CEO Quote`) are labels of page regions and are not printed as headings, so the only rule tried (above) was rejected.
- **Slide decks (doc_050/051, F1 0.07/0.27).** Gold titles are `Slide N: ...`, which never occur in the text.
- **Shallow gold (doc_001, F1 0.05).** The Berkshire annual report has 9 bookmarks. The extractor finds the Form 10-K `Item` headings and the letter's section titles, which the bookmarks do not list.
- **Engineering drawings and "(continued)" titles (doc_041, doc_049).** Package drawing pages (`JG0008A`, `5.08 MAX`) and repeated box titles ending in "(continued)" are still predicted. Each is a small, single-document effect.

## Document-type classification

**Headline.** On 51 real PDFs and 18 defined leaf types, TypeSafe Jev given docrooter's DocumentView (`jev-view`) is the most accurate system: 49/51 correct (leaf accuracy 0.961, macro F1 0.944), at 1.74 s/doc end to end and $0.0044 for the whole corpus. The best local docrooter backend, `ultrabert-hybrid`, gets 40/51 (0.784, macro F1 0.733), and the plain `ultrabert-view` 39/51 (0.765, macro F1 0.719) at the same 1.78 s/doc with no API. Jev on naive raw-text chunks (`jev-chunks`, the suggested baseline) gets 47/51 (0.922) using 10.7x the input tokens. The trained leave-one-out head (`ultrabert-loo`) is the worst system at 12/51 (0.235); see the analysis below. These are single-run numbers on a small corpus.

<!-- doctype-benchmark:begin -->
Corpus: 51 real PDFs from `benchmarks/corpus/` with gold labels from the manifest, 18 leaf types in 6 groups (`builtin:doc_types`; every label has a definition, see `benchmarks/corpus/README.md`). Classes have 2-4 documents each, so per-class numbers are anecdotal and one document moves leaf accuracy by 2.0 points. The 95% intervals are percentile bootstraps over documents (1000 resamples, seed 0). Every system labels every document (threshold 0).

Environment: Python 3.12.13, arm64, 8 CPUs, device `mps` (MPS probe matmul took 1.85 s), torch 2.14.1, sentence-transformers 6.1.0, scikit-learn 1.9.1, typesafe-sdk 0.7.5, pymupdf 1.28.2, Jev `jev-1.13.0`. The load average at the start of each system is listed below; the machine was shared with another benchmark, so seconds are noisy upper bounds.

Timing: *parse* is one `load_pdf` call with the default engine (pymupdf), cached and shared by every system except `jev-chunks`, whose parse is pymupdf `page.get_text()` over all pages. *Classify* for the `classify_documents` systems (`*-view`, `ultrabert-hybrid`) is the wall time of one call over all documents divided by the number of documents (DocumentView rendering included). For `ultrabert-loo` it is view rendering plus embedding divided by the number of documents; the logistic-regression fit per fold is listed separately. For `jev-chunks` it is the wall time of each document's concurrent chunk requests. *Load* is a warm-up excluded from the averages: model load plus one classification of a tiny synthetic document (the label vectors are encoded there). Jev cost is input tokens x $0.042/M (output tokens are free), with tokens as reported by the API (`usage.input_tokens`).

### Aggregate

| System | What | Leaf acc | 95% CI | Level-1 acc | Macro P | Macro R | Macro F1 | Parse s/doc | Classify s/doc | Total s/doc | Load s | Jev input tokens | Jev cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ultrabert-view` | UltraBERT-Embed-150M, DocumentView, Flat (zero-shot) | 0.765 | 0.65-0.86 | 0.902 | 0.708 | 0.764 | 0.719 | 1.60 | 0.177 | 1.78 | 2.6 | - | - |
| `ultrabert-hybrid` | UltraBERT-Embed-150M, 0.6 view + 0.4 sections (zero-shot) | 0.784 | 0.67-0.88 | 0.922 | 0.711 | 0.778 | 0.733 | 1.60 | 2.655 | 4.26 | 2.3 | - | - |
| `bge-view` | bge-small-en-v1.5, DocumentView, Flat (zero-shot) | 0.608 | 0.47-0.75 | 0.784 | 0.576 | 0.611 | 0.538 | 1.60 | 0.134 | 1.74 | 14.3 | - | - |
| `nli-view` | deberta-v3-base-zeroshot-v2.0 NLI, DocumentView, Flat (zero-shot) | 0.569 | 0.43-0.69 | 0.765 | 0.510 | 0.532 | 0.485 | 1.60 | 14.713 | 16.32 | 2.8 | - | - |
| `jev-view` | Jev jev-1.13.0, DocumentView as state, one Choice over 18 leaves | 0.961 | 0.90-1.00 | 0.961 | 0.968 | 0.944 | 0.944 | 1.60 | 0.137 | 1.74 | 0.7 | 104,384 | $0.0044 |
| `jev-chunks` | Jev jev-1.13.0, raw page text, <=8 chunks of ~1,500 tokens, mean of per-chunk Choices (no docrooter) | 0.922 | 0.84-0.98 | 0.961 | 0.947 | 0.912 | 0.911 | 0.49 | 0.320 | 0.81 | 0.0 | 1,121,481 | $0.0471 |
| `ultrabert-loo` | UltraBERT-Embed-150M view embeddings + logistic regression, trained (leave-one-out) | 0.235 | 0.14-0.35 | 0.608 | 0.063 | 0.167 | 0.090 | 1.60 | 0.179 | 1.78 | 1.5 | - | - |

Run metadata per system:

- `ultrabert-view`: 2026-10-16 at `3bd85f2`, device `mps`, load average 3.9, 5.9, 13.1 at start.
- `ultrabert-hybrid`: 2026-10-16 at `3bd85f2`, device `mps`, load average 3.3, 4.9, 8.3 at start.
- `bge-view`: 2026-10-16 at `3bd85f2`, device `mps`, load average 3.7, 6.1, 13.4 at start.
- `nli-view`: 2026-10-16 at `3bd85f2`, device `mps`, load average 4.3, 5.8, 12.9 at start.
- `jev-view`: 2026-10-16 at `3bd85f2`, device `n/a`, load average 3.8, 6.5, 14.0 at start.
- `jev-chunks`: 2026-10-16 at `3bd85f2`, device `n/a`, load average 4.4, 6.5, 13.9 at start; 331 chunks sent (6.5 per document).
- `ultrabert-loo`: 2026-10-16 at `3bd85f2`, device `mps`, load average 4.2, 5.9, 13.1 at start; mean logistic-regression fit 0.01 s per fold.

### Per class

Best docrooter system by macro F1 (zero-shot: `ultrabert-hybrid`; any: `ultrabert-hybrid`) and `jev-view`. P / R / F1 per gold class.

| Class | n | `ultrabert-hybrid` P / R / F1 | `jev-view` P / R / F1 |
| --- | --- | --- | --- |
| financial.annual_report | 3 | 0.67 / 0.67 / 0.67 | 1.00 / 1.00 / 1.00 |
| financial.quarterly_report | 2 | 1.00 / 1.00 / 1.00 | 0.67 / 1.00 / 0.80 |
| financial.shareholder_letter | 3 | 0.50 / 0.33 / 0.40 | 1.00 / 1.00 / 1.00 |
| financial.earnings_release | 2 | 0.67 / 1.00 / 0.80 | 1.00 / 1.00 / 1.00 |
| financial.prospectus | 2 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| financial.proxy_statement | 2 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| legal.contract | 2 | 0.00 / 0.00 / 0.00 | 1.00 / 0.50 / 0.67 |
| legal.terms_policy | 3 | 0.75 / 1.00 / 0.86 | 1.00 / 1.00 / 1.00 |
| legal.court_opinion | 4 | 1.00 / 0.75 / 0.86 | 1.00 / 1.00 / 1.00 |
| legal.legislation | 3 | 0.75 / 1.00 / 0.86 | 1.00 / 1.00 / 1.00 |
| research.research_paper | 4 | 0.67 / 1.00 / 0.80 | 1.00 / 1.00 / 1.00 |
| research.technical_standard | 4 | 0.80 / 1.00 / 0.89 | 1.00 / 1.00 / 1.00 |
| research.white_paper | 3 | 0.00 / 0.00 / 0.00 | 1.00 / 1.00 / 1.00 |
| technical.user_manual | 3 | 0.75 / 1.00 / 0.86 | 0.75 / 1.00 / 0.86 |
| technical.datasheet | 4 | 1.00 / 0.75 / 0.86 | 1.00 / 1.00 / 1.00 |
| government.government_report | 3 | 0.75 / 1.00 / 0.86 | 1.00 / 1.00 / 1.00 |
| corporate.sustainability_report | 2 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| corporate.investor_presentation | 2 | 0.50 / 0.50 / 0.50 | 1.00 / 0.50 / 0.67 |

### Confusion matrices

Rows are gold, columns are predictions, numbered as the rows. Empty cells are 0; the diagonal is in bold.

`ultrabert-hybrid`:

| Gold \ Pred | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 annual_report | **2** |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2 quarterly_report |  | **2** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 3 shareholder_letter | 1 |  | **1** |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 |
| 4 earnings_release |  |  |  | **2** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 5 prospectus |  |  |  |  | **2** |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 6 proxy_statement |  |  |  |  |  | **2** |  |  |  |  |  |  |  |  |  |  |  |  |
| 7 contract |  |  |  |  |  |  |  | 1 |  |  |  | 1 |  |  |  |  |  |  |
| 8 terms_policy |  |  |  |  |  |  |  | **3** |  |  |  |  |  |  |  |  |  |  |
| 9 court_opinion |  |  |  |  |  |  |  |  | **3** | 1 |  |  |  |  |  |  |  |  |
| 10 legislation |  |  |  |  |  |  |  |  |  | **3** |  |  |  |  |  |  |  |  |
| 11 research_paper |  |  |  |  |  |  |  |  |  |  | **4** |  |  |  |  |  |  |  |
| 12 technical_standard |  |  |  |  |  |  |  |  |  |  |  | **4** |  |  |  |  |  |  |
| 13 white_paper |  |  |  |  |  |  |  |  |  |  | 2 |  |  |  |  | 1 |  |  |
| 14 user_manual |  |  |  |  |  |  |  |  |  |  |  |  |  | **3** |  |  |  |  |
| 15 datasheet |  |  |  |  |  |  |  |  |  |  |  |  |  | 1 | **3** |  |  |  |
| 16 government_report |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | **3** |  |  |
| 17 sustainability_report |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | **2** |  |
| 18 investor_presentation |  |  |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  | **1** |

`jev-view`:

| Gold \ Pred | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 annual_report | **3** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2 quarterly_report |  | **2** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 3 shareholder_letter |  |  | **3** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 4 earnings_release |  |  |  | **2** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 5 prospectus |  |  |  |  | **2** |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 6 proxy_statement |  |  |  |  |  | **2** |  |  |  |  |  |  |  |  |  |  |  |  |
| 7 contract |  |  |  |  |  |  | **1** |  |  |  |  |  |  | 1 |  |  |  |  |
| 8 terms_policy |  |  |  |  |  |  |  | **3** |  |  |  |  |  |  |  |  |  |  |
| 9 court_opinion |  |  |  |  |  |  |  |  | **4** |  |  |  |  |  |  |  |  |  |
| 10 legislation |  |  |  |  |  |  |  |  |  | **3** |  |  |  |  |  |  |  |  |
| 11 research_paper |  |  |  |  |  |  |  |  |  |  | **4** |  |  |  |  |  |  |  |
| 12 technical_standard |  |  |  |  |  |  |  |  |  |  |  | **4** |  |  |  |  |  |  |
| 13 white_paper |  |  |  |  |  |  |  |  |  |  |  |  | **3** |  |  |  |  |  |
| 14 user_manual |  |  |  |  |  |  |  |  |  |  |  |  |  | **3** |  |  |  |  |
| 15 datasheet |  |  |  |  |  |  |  |  |  |  |  |  |  |  | **4** |  |  |  |
| 16 government_report |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | **3** |  |  |
| 17 sustainability_report |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | **2** |  |
| 18 investor_presentation |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | **1** |

### Per document

✓ correct leaf, ✗ wrong. Predictions show the leaf id without its group.

| Doc | Title | Pages | Gold | `ultrabert-view` | `ultrabert-hybrid` | `bge-view` | `nli-view` | `jev-view` | `jev-chunks` | `ultrabert-loo` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| doc_001 | Berkshire Hathaway Inc. 2023 Annual Report | 152 | annual_report | annual_report ✓ | annual_report ✓ | shareholder_letter ✗ | investor_presentation ✗ | annual_report ✓ | annual_report ✓ | shareholder_letter ✗ |
| doc_002 | Tesla, Inc. Annual Report on Form 10-K for fi... | 130 | annual_report | annual_report ✓ | annual_report ✓ | proxy_statement ✗ | annual_report ✓ | annual_report ✓ | annual_report ✓ | datasheet ✗ |
| doc_003 | Amazon.com, Inc. 2023 Annual Report | 92 | annual_report | shareholder_letter ✗ | shareholder_letter ✗ | shareholder_letter ✗ | shareholder_letter ✗ | annual_report ✓ | annual_report ✓ | shareholder_letter ✗ |
| doc_004 | Berkshire Hathaway Inc. Form 10-Q for the qua... | 56 | quarterly_report | quarterly_report ✓ | quarterly_report ✓ | proxy_statement ✗ | quarterly_report ✓ | quarterly_report ✓ | quarterly_report ✓ | annual_report ✗ |
| doc_005 | Berkshire Hathaway Inc. Form 10-Q for the qua... | 59 | quarterly_report | quarterly_report ✓ | quarterly_report ✓ | proxy_statement ✗ | prospectus ✗ | quarterly_report ✓ | quarterly_report ✓ | annual_report ✗ |
| doc_006 | Berkshire Hathaway Chairman's Letter to Share... | 16 | shareholder_letter | investor_presentation ✗ | investor_presentation ✗ | shareholder_letter ✓ | shareholder_letter ✓ | shareholder_letter ✓ | shareholder_letter ✓ | annual_report ✗ |
| doc_007 | Berkshire Hathaway Chairman's Letter to Share... | 22 | shareholder_letter | annual_report ✗ | annual_report ✗ | earnings_release ✗ | annual_report ✗ | shareholder_letter ✓ | shareholder_letter ✓ | court_opinion ✗ |
| doc_008 | Amazon 2020 Shareholder Letter with the 1997 ... | 10 | shareholder_letter | shareholder_letter ✓ | shareholder_letter ✓ | shareholder_letter ✓ | prospectus ✗ | shareholder_letter ✓ | shareholder_letter ✓ | annual_report ✗ |
| doc_009 | Amazon.com Announces Second Quarter Results (... | 12 | earnings_release | earnings_release ✓ | earnings_release ✓ | earnings_release ✓ | quarterly_report ✗ | earnings_release ✓ | quarterly_report ✗ | shareholder_letter ✗ |
| doc_010 | Amazon.com Announces Third Quarter Results (O... | 12 | earnings_release | earnings_release ✓ | earnings_release ✓ | earnings_release ✓ | quarterly_report ✗ | earnings_release ✓ | earnings_release ✓ | shareholder_letter ✗ |
| doc_011 | Erste Abwicklungsanstalt Debt Issuance Progra... | 113 | prospectus | prospectus ✓ | prospectus ✓ | prospectus ✓ | prospectus ✓ | prospectus ✓ | prospectus ✓ | court_opinion ✗ |
| doc_012 | EnBW Debt Issuance Programme Prospectus 2024 | 222 | prospectus | prospectus ✓ | prospectus ✓ | prospectus ✓ | prospectus ✓ | prospectus ✓ | prospectus ✓ | datasheet ✗ |
| doc_013 | Postal Realty Trust 2025 Notice of Annual Mee... | 63 | proxy_statement | proxy_statement ✓ | proxy_statement ✓ | proxy_statement ✓ | investor_presentation ✗ | proxy_statement ✓ | proxy_statement ✓ | annual_report ✗ |
| doc_014 | WisdomTree, Inc. 2024 Notice of Annual Meetin... | 161 | proxy_statement | proxy_statement ✓ | proxy_statement ✓ | proxy_statement ✓ | investor_presentation ✗ | proxy_statement ✓ | proxy_statement ✓ | court_opinion ✗ |
| doc_015 | Sample Non-Disclosure Agreement for condition... | 4 | contract | government_report ✗ | terms_policy ✗ | proxy_statement ✗ | government_report ✗ | user_manual ✗ | contract ✓ | court_opinion ✗ |
| doc_016 | NIST Cooperative Research and Development Agr... | 10 | contract | technical_standard ✗ | technical_standard ✗ | technical_standard ✗ | contract ✓ | contract ✓ | contract ✓ | datasheet ✗ |
| doc_017 | Apple Privacy Policy (updated July 30, 2025) | 1 | terms_policy | terms_policy ✓ | terms_policy ✓ | terms_policy ✓ | terms_policy ✓ | terms_policy ✓ | terms_policy ✓ | technical_standard ✗ |
| doc_018 | Terms of Use (GoWhere support service) | 19 | terms_policy | terms_policy ✓ | terms_policy ✓ | terms_policy ✓ | contract ✗ | terms_policy ✓ | terms_policy ✓ | technical_standard ✗ |
| doc_019 | Terms of Service for the Fieldwork Performanc... | 8 | terms_policy | terms_policy ✓ | terms_policy ✓ | proxy_statement ✗ | contract ✗ | terms_policy ✓ | terms_policy ✓ | technical_standard ✗ |
| doc_020 | Dobbs v. Jackson Women's Health Organization,... | 213 | court_opinion | legislation ✗ | legislation ✗ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ |
| doc_021 | Trump v. Anderson, No. 23-719 (per curiam sli... | 20 | court_opinion | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ |
| doc_022 | O'Connor-Ratcliff v. Garnier, No. 22-324 (per... | 6 | court_opinion | court_opinion ✓ | court_opinion ✓ | earnings_release ✗ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ |
| doc_023 | Department of Education v. Louisiana, No. 24A... | 13 | court_opinion | legislation ✗ | court_opinion ✓ | proxy_statement ✗ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ |
| doc_024 | CHIPS and Science Act, Public Law 117-167 | 394 | legislation | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | court_opinion ✗ |
| doc_025 | Fiscal Responsibility Act of 2023, Public Law... | 41 | legislation | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | court_opinion ✗ |
| doc_026 | S. 623, Sunshine Protection Act of 2021 (as p... | 6 | legislation | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | court_opinion ✗ |
| doc_027 | Attention Is All You Need | 15 | research_paper | research_paper ✓ | research_paper ✓ | investor_presentation ✗ | research_paper ✓ | research_paper ✓ | research_paper ✓ | research_paper ✓ |
| doc_028 | Deep Residual Learning for Image Recognition | 12 | research_paper | research_paper ✓ | research_paper ✓ | investor_presentation ✗ | research_paper ✓ | research_paper ✓ | research_paper ✓ | technical_standard ✗ |
| doc_029 | Language Models are Few-Shot Learners | 75 | research_paper | research_paper ✓ | research_paper ✓ | earnings_release ✗ | research_paper ✓ | research_paper ✓ | research_paper ✓ | research_paper ✓ |
| doc_030 | On the Opportunities and Risks of Foundation ... | 214 | research_paper | research_paper ✓ | research_paper ✓ | sustainability_report ✗ | research_paper ✓ | research_paper ✓ | research_paper ✓ | technical_standard ✗ |
| doc_031 | RFC 9110: HTTP Semantics | 194 | technical_standard | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ |
| doc_032 | RFC 9000: QUIC: A UDP-Based Multiplexed and S... | 151 | technical_standard | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ |
| doc_033 | RFC 9239: Updates to ECMAScript Media Types | 13 | technical_standard | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ |
| doc_034 | RFC 9457: Problem Details for HTTP APIs | 16 | technical_standard | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ |
| doc_035 | Google Cloud's AI Adoption Framework | 37 | white_paper | research_paper ✗ | research_paper ✗ | technical_standard ✗ | investor_presentation ✗ | white_paper ✓ | white_paper ✓ | research_paper ✗ |
| doc_036 | Future of Jobs Report 2023 (Insight Report) | 296 | white_paper | government_report ✗ | government_report ✗ | white_paper ✓ | government_report ✗ | white_paper ✓ | white_paper ✓ | research_paper ✗ |
| doc_037 | Bitcoin: A Peer-to-Peer Electronic Cash System | 9 | white_paper | research_paper ✗ | research_paper ✗ | earnings_release ✗ | technical_standard ✗ | white_paper ✓ | research_paper ✗ | technical_standard ✗ |
| doc_038 | Bash Reference Manual, Edition 5.3 | 214 | user_manual | user_manual ✓ | user_manual ✓ | user_manual ✓ | user_manual ✓ | user_manual ✓ | user_manual ✓ | technical_standard ✗ |
| doc_039 | GNU Coreutils manual | 326 | user_manual | user_manual ✓ | user_manual ✓ | datasheet ✗ | technical_standard ✗ | user_manual ✓ | user_manual ✓ | technical_standard ✗ |
| doc_040 | Getting Started with Raspberry Pi Pico-series | 36 | user_manual | user_manual ✓ | user_manual ✓ | datasheet ✗ | datasheet ✗ | user_manual ✓ | user_manual ✓ | datasheet ✗ |
| doc_041 | NE555 / SA555 / SE555 Precision Timers datasheet | 39 | datasheet | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | technical_standard ✗ |
| doc_042 | RP2040 Datasheet | 642 | datasheet | user_manual ✗ | user_manual ✗ | datasheet ✓ | technical_standard ✗ | datasheet ✓ | datasheet ✓ | technical_standard ✗ |
| doc_043 | 1N4148 Small Signal Fast Switching Diode data... | 4 | datasheet | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ |
| doc_044 | 1N4001-1N4007 1.0A Rectifier datasheet | 5 | datasheet | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | terms_policy ✗ | datasheet ✓ |
| doc_045 | Monetary Policy Report, July 5, 2024 | 79 | government_report | government_report ✓ | government_report ✓ | government_report ✓ | annual_report ✗ | government_report ✓ | government_report ✓ | court_opinion ✗ |
| doc_046 | Financial Stability Report, April 2024 | 69 | government_report | government_report ✓ | government_report ✓ | government_report ✓ | quarterly_report ✗ | government_report ✓ | government_report ✓ | research_paper ✗ |
| doc_047 | The 9/11 Commission Report | 585 | government_report | government_report ✓ | government_report ✓ | government_report ✓ | research_paper ✗ | government_report ✓ | government_report ✓ | court_opinion ✗ |
| doc_048 | Google Environmental Report 2024 | 86 | sustainability_report | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ | research_paper ✗ |
| doc_049 | NVIDIA Sustainability Report, Fiscal Year 2024 | 41 | sustainability_report | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ | research_paper ✗ |
| doc_050 | Tesla Q2 2024 Update (investor slide deck) | 29 | investor_presentation | earnings_release ✗ | earnings_release ✗ | earnings_release ✗ | quarterly_report ✗ | quarterly_report ✗ | quarterly_report ✗ | datasheet ✗ |
| doc_051 | NVIDIA Investor Presentation Q2 FY25 (Septemb... | 27 | investor_presentation | investor_presentation ✓ | investor_presentation ✓ | investor_presentation ✓ | investor_presentation ✓ | investor_presentation ✓ | investor_presentation ✓ | research_paper ✗ |
<!-- doctype-benchmark:end -->

### Analysis

**Is the gap real?** On 51 documents one document is 2 accuracy points. The 95% bootstrap intervals of `jev-view` (0.90-1.00) and `ultrabert-hybrid` (0.67-0.88) do not overlap, so Jev's lead over the local backends is not noise. The difference between `jev-view` and `jev-chunks` is 2 documents (intervals 0.90-1.00 vs 0.84-0.98), which this corpus cannot separate. The same holds for `ultrabert-hybrid` vs `ultrabert-view` (1 document), and for `bge-view` vs `nli-view` (2 documents).

**Where `jev-view` fails (2 of 51).**

- **doc_015, a sample NDA, predicted `user_manual` (0.53, with `contract` at 0.33).** The PDF opens with a page of "Instructions for non-disclosure agreements". The view's lead is that instruction text, and the title is the first numbered clause, because the parser turns the agreement's numbered paragraphs into headings. Jev answered for what the view shows.
- **doc_050, the Tesla Q2 2024 Update deck, predicted `quarterly_report` (0.53, with `earnings_release` at 0.39).** Every system misses this document. It is a text-heavy shareholder "update" laid out as slides. It parses to a single section, and its view has letter-spaced keyphrases (`s u m`, `g h t`), so the slide layout that defines `investor_presentation` is not visible in the text.

**Where `jev-chunks` fails (4 of 51).**

- **doc_044, a two-page diode datasheet, was lost on a tie.** The second chunk is the vendor's legal "Important Notice". The per-chunk mean gave `terms_policy` and `datasheet` 0.50 each, and the tie went to the option listed first. Equal-weight averaging of raw chunks lets boilerplate vote; a document view does not have that failure mode.
- **doc_009 (earnings release, predicted `quarterly_report`) and doc_050.** Both are financial-type confusions.
- **doc_037 (the Bitcoin paper, `white_paper` gold, predicted `research_paper` at 0.63).** This is a borderline label that `jev-view` got right.

**Where the local zero-shot backends fail.** `ultrabert-*` is good at the group: level-1 accuracy is 0.90-0.92, and `ultrabert-hybrid` has recall 1.00 on 11 of 18 classes. Most errors fall inside a group, between siblings whose definitions differ by who published the document or why, rather than by vocabulary:

- **White papers.** `ultrabert-*` got 0 of 3. They are predicted as `research_paper` or `government_report`, because "not peer-reviewed" and "issued by an agency" are not similarity features.
- **Contracts.** `ultrabert-*` got 0 of 2. The NIST CRADA reads like a `technical_standard`, and the NDA like `terms_policy` or `government_report`.
- **Shareholder letters vs annual reports.** doc_003, Amazon's annual report, starts with the 1997 shareholder letter, and DocumentView takes its title from the first non-letter-spaced level-1 heading ("1997 LETTER TO SHAREHOLDERS"). The Berkshire letters go to `investor_presentation` or `annual_report`.
- **Other errors.**
  - Dobbs is predicted as `legislation`, at 0.37 against 0.29 for `court_opinion`.
  - The RP2040 datasheet is predicted as `user_manual`, at 0.277 against 0.268 for `datasheet`.
  - `hybrid` fixes exactly one document (doc_023, a court opinion) for 15x the classify time (2.66 vs 0.18 s/doc). It embeds every leaf section.

`bge-view` (0.608) suffers from bge-small's 512-token window. The 512-token view plus the query prompt overflows it, so the end of the view is truncated. It also sends all four research papers elsewhere (`investor_presentation`, `earnings_release`, `sustainability_report`). `nli-view` (0.569) is both the least accurate zero-shot system and by far the slowest: 14.7 s/doc, because it runs 18 premise-hypothesis pairs of up to 512 tokens per document through DeBERTa on MPS.

**The trained leave-one-out system is a negative result, and the reason is known.** `LogisticRegression(max_iter=1000)` uses the default `C=1`. On unit-norm 768-d embeddings with 50 training documents, that regularises almost everything away. The predicted probabilities are close to uniform (top class about 0.07-0.10), so the head falls back on class priors: 42 of 51 predictions are one of the four classes with four documents. Classes with 2 documents have a single training example when one of them is held out. A post-hoc diagnostic was run on the same embeddings and the same LOO splits. It is not a reported system, because the settings were chosen after seeing the result. It gave 0.59 at C=10, 0.61 at C=100, 0.65 at C=1000, and 0.65 for a nearest class centroid. Even tuned, 1-3 examples per class do not beat zero-shot with written definitions (0.765). Training only pays off with more labelled documents per class.

**Cost and speed.**

- **Parsing dominates the local pipeline.** The default `load_pdf` takes 1.60 s/doc on average (0.02-9.1 s, roughly proportional to pages). Classification takes 0.14-0.18 s/doc for `jev-view`, `ultrabert-view` and `bge-view`.
- **`jev-chunks` is the fastest end to end (0.81 s/doc)** because it skips structure: `page.get_text()` is 0.49 s/doc. It pays for that in tokens: 1,121,481 input tokens ($0.047, about $0.92 per 1,000 documents) against 104,384 for `jev-view` (about 2,050 per document, of which roughly 1,500 are the fixed instruction and 18 definitions, since the smallest request was 1,742 tokens; $0.0044, about $0.09 per 1,000 documents).
- **Both Jev costs are small in absolute terms.** The view's real advantages are 10.7x fewer tokens per document, a bounded request size whatever the page count, and robustness to boilerplate chunks. Its accuracy edge over chunks is not significant on this corpus.
- **Jev timings are network-bound.** They come from one run with 8 concurrent requests. `jev-view` classify time is the batch wall time over 51 documents, 7 s in total.

**Honest comparison.** For document-type gating with written definitions, Jev is clearly better than any local zero-shot model tested here, by 18-39 accuracy points. docrooter's contribution in that pairing is the compact, budgeted DocumentView: Jev reads it as well as or better than eight raw chunks at a tenth of the tokens. When no API is allowed, `ultrabert-view` is the local choice at 0.765 leaf accuracy and 0.90 group accuracy, with `confidence` available for routing low-confidence documents to review. `ultrabert-hybrid` is +1 document for 15x the classify time, and `bge-view` and `nli-view` are not competitive.

**Follow-ups this run exposed (not fixed here).**

- DocumentView keyphrases on letter-spaced text (doc_050).
- Title choice when a document starts with an embedded letter or an instruction page (doc_003, doc_015).
- A tie-break rule for exact ties in aggregated distributions.

**Provenance.**

- The run was on 2026-10-16. The commits listed above are the `src/` revision; the benchmark scripts were added in the commit that published these results.
- Parsed documents are cached in `benchmarks/.cache/docs/`. Four of them (doc_026-028, doc_041) were parsed during a smoke run a few minutes earlier, at a similar load.
- Another benchmark job was running on the machine but was light during this run (load average 3-5, against 20-30 earlier in the day).
- Raw rows, including the top-3 probabilities and per-document tokens, are in `benchmarks/results/2026-10-16/pdf.json`.

### Known limitations

- **Headings set below body size are never detected.** Since `5d3e621`, a line smaller than the body size by more than 0.5 pt is not a heading candidate. Small-caps headings, whose measured size is often the small-capital glyph size, and compact run-in headings in documents with large body text are lost even when they are bold and isolated. No matched heading in this benchmark is below 0.95 x body, so the rule costs nothing here, but other corpora may differ.
- **Section running lines need only 3 pages.** A non-candidate edge-band line, or a bold large one at a fixed height, that repeats on 3 pages is excluded, however long the document. A genuine heading printed in the top or bottom 8% of the page at the same height on 3 or more pages with identical text (for example, a recurring "Notes" page title) is therefore lost.

