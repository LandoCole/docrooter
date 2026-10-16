# PDF benchmarks

## PDF TOC extraction

<!-- toc-benchmark:begin -->
Gold: bookmarks of the 24 corpus PDFs with at least 5 bookmarks (levels shifted so the minimum is 1; items without a target page dropped). Extraction runs with `use_bookmarks=False`. A prediction matches a bookmark at normalised-title ratio >= 0.85 and pages within +/-1, one-to-one and greedy by ratio. The depth <= 2 columns keep only gold and predicted headings of level <= 2, so precision is measured at that depth too. Micro pools the counts; macro is the mean of per-document F1. Level accuracy is measured on matched pairs. Bookmark sets often omit lower-level headings, so precision is pessimistic and recall / F1 at depth <= 2 are the headline numbers.

Environment: Python 3.12.13, arm64, 8 CPUs, pymupdf 1.28.2, pdfplumber 0.11.10, pymupdf4llm 1.28.2. `os.getloadavg()` was 30.0, 27.3, 30.7 (1/5/15 min) when the latest run started. The machine was shared with other jobs, so seconds per document are noisy upper bounds; each engine ran one untimed warm-up document first. Seconds are wall-clock for one `load_pdf` call. Each engine's row keeps the run that produced it: `pymupdf` 2026-10-16 at `ee95ee5` (load 30.0, 27.3, 30.7); `pdfplumber` 2026-10-16 at `ee95ee5` (load 24.9, 26.3, 30.1); `both` 2026-10-16 at `ee95ee5` (load 15.9, 20.4, 26.9); `pymupdf4llm` 2026-10-16 at `5610ada` (load 8.1, 10.8, 14.2).

### Aggregate by engine (depth <= 2 headline, all depths for reference)

| Engine | Docs | P (d<=2) | R (d<=2) | F1 micro (d<=2) | F1 macro (d<=2) | Level acc (d<=2) | P (all) | R (all) | F1 micro (all) | F1 macro (all) | Level acc (all) | s/doc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pymupdf | 24 | 0.632 | 0.791 | 0.702 | 0.592 | 0.770 | 0.550 | 0.905 | 0.684 | 0.632 | 0.596 | 1.52 |
| pdfplumber | 24 | 0.643 | 0.788 | 0.708 | 0.596 | 0.796 | 0.575 | 0.900 | 0.702 | 0.642 | 0.606 | 7.32 |
| both | 24 | 0.656 | 0.785 | 0.715 | 0.604 | 0.770 | 0.595 | 0.900 | 0.716 | 0.656 | 0.597 | 8.68 |
| pymupdf4llm* | 8 | 0.667 | 0.564 | 0.611 | 0.517 | 0.974 | 0.615 | 0.692 | 0.651 | 0.550 | 0.912 | 15.21 |

\* `pymupdf4llm`: limited to the 8 smallest bookmarked documents (by page count); see the matched-subset table below.

### Same 8 documents for every engine (the `pymupdf4llm` subset)

| Engine | P (d<=2) | R (d<=2) | F1 micro (d<=2) | F1 macro (d<=2) | Level acc (d<=2) | F1 micro (all) | s/doc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pymupdf | 0.757 | 0.569 | 0.650 | 0.472 | 0.678 | 0.754 | 0.27 |
| pdfplumber | 0.743 | 0.559 | 0.638 | 0.467 | 0.673 | 0.723 | 1.43 |
| both | 0.776 | 0.564 | 0.653 | 0.481 | 0.675 | 0.749 | 1.58 |
| pymupdf4llm | 0.667 | 0.564 | 0.611 | 0.517 | 0.974 | 0.651 | 15.21 |

### Per document: `pymupdf` (the default engine)

| Doc | Title | Pages | Gold d<=2 | Pred d<=2 | P | R | F1 | Level acc | F1 (all) | s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| doc_001 | Berkshire Hathaway Inc. 2023 Annual Report | 152 | 9 | 32 | 0.03 | 0.11 | 0.05 | 0.00 | 0.19 | 3.05 |
| doc_003 | Amazon.com, Inc. 2023 Annual Report | 92 | 30 | 29 | 0.69 | 0.67 | 0.68 | 0.00 | 0.68 | 1.67 |
| doc_004 | Berkshire Hathaway Inc. Form 10-Q for the qua... | 56 | 22 | 14 | 0.71 | 0.45 | 0.56 | 0.00 | 0.53 | 0.99 |
| doc_005 | Berkshire Hathaway Inc. Form 10-Q for the qua... | 59 | 22 | 14 | 0.71 | 0.45 | 0.56 | 0.00 | 0.53 | 1.08 |
| doc_009 | Amazon.com Announces Second Quarter Results (... | 12 | 18 | 0 | 0.00 | 0.00 | 0.00 | n/a | 0.00 | 0.16 |
| doc_010 | Amazon.com Announces Third Quarter Results (O... | 12 | 18 | 0 | 0.00 | 0.00 | 0.00 | n/a | 0.00 | 0.17 |
| doc_027 | Attention Is All You Need | 15 | 19 | 23 | 0.83 | 1.00 | 0.90 | 1.00 | 0.92 | 0.24 |
| doc_029 | Language Models are Few-Shot Learners | 75 | 32 | 38 | 0.84 | 1.00 | 0.91 | 1.00 | 0.72 | 1.32 |
| doc_030 | On the Opportunities and Risks of Foundation ... | 214 | 39 | 40 | 0.95 | 0.97 | 0.96 | 1.00 | 0.90 | 4.28 |
| doc_031 | RFC 9110: HTTP Semantics | 194 | 30 | 158 | 0.18 | 0.93 | 0.30 | 0.00 | 0.86 | 2.32 |
| doc_032 | RFC 9000: QUIC: A UDP-Based Multiplexed and S... | 151 | 36 | 141 | 0.25 | 0.97 | 0.40 | 0.03 | 0.97 | 1.65 |
| doc_033 | RFC 9239: Updates to ECMAScript Media Types | 13 | 17 | 25 | 0.64 | 0.94 | 0.76 | 0.06 | 0.96 | 0.14 |
| doc_034 | RFC 9457: Problem Details for HTTP APIs | 16 | 19 | 25 | 0.72 | 0.95 | 0.82 | 0.06 | 0.97 | 0.14 |
| doc_038 | Bash Reference Manual, Edition 5.3 | 214 | 70 | 76 | 0.82 | 0.89 | 0.85 | 1.00 | 0.92 | 2.63 |
| doc_039 | GNU Coreutils manual | 326 | 179 | 188 | 0.95 | 0.99 | 0.97 | 1.00 | 0.98 | 3.88 |
| doc_040 | Getting Started with Raspberry Pi Pico-series | 36 | 55 | 61 | 0.90 | 1.00 | 0.95 | 1.00 | 0.93 | 0.92 |
| doc_041 | NE555 / SA555 / SE555 Precision Timers datasheet | 39 | 30 | 45 | 0.44 | 0.67 | 0.53 | 1.00 | 0.61 | 0.36 |
| doc_042 | RP2040 Datasheet | 642 | 96 | 98 | 0.97 | 0.99 | 0.98 | 0.77 | 0.44 | 6.27 |
| doc_045 | Monetary Policy Report, July 5, 2024 | 79 | 23 | 12 | 0.67 | 0.35 | 0.46 | 1.00 | 0.65 | 0.90 |
| doc_046 | Financial Stability Report, April 2024 | 69 | 13 | 14 | 0.93 | 1.00 | 0.96 | 1.00 | 0.96 | 0.68 |
| doc_048 | Google Environmental Report 2024 | 86 | 32 | 24 | 0.62 | 0.47 | 0.54 | 0.93 | 0.35 | 2.70 |
| doc_049 | NVIDIA Sustainability Report, Fiscal Year 2024 | 41 | 30 | 46 | 0.61 | 0.93 | 0.74 | 1.00 | 0.51 | 0.53 |
| doc_050 | Tesla Q2 2024 Update (investor slide deck) | 29 | 29 | 1 | 1.00 | 0.03 | 0.07 | 1.00 | 0.07 | 0.24 |
| doc_051 | NVIDIA Investor Presentation Q2 FY25 (Septemb... | 27 | 27 | 17 | 0.35 | 0.22 | 0.27 | 0.17 | 0.54 | 0.15 |
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

Notes on the rules:

- **Section running lines** (`23f8cd2`). An edge-band line whose text, with any leading or trailing page number removed, recurs on at least 3 pages is now excluded as a running line, even far below 30% of a long document. RP2040 (doc_042) dropped from 742 to 151 depth <= 2 predictions. This adds to the ">= 30% of pages" rule in the global constraints and does not replace it.
- **One-off styles** (`bf04ccb`). A style used by fewer than min(3, 10%) of the kept headings (cover titles, front-matter one-offs) takes the level of the next recurring style instead of a level of its own. This fixed the level-5 `Item N.` headings in doc_003, which went from F1 0.00 to 0.67. It is also the main cause of the fall in level accuracy. The 10% count threshold is unstable on documents with about 20 headings (doc_045 changes levels when a few candidates are removed).
- **Edge-band copies** (`94a5cbf`) and **wrapped titles** (`fbd900a`) are small precision and recall fixes.
- **Contents pages without page numbers** (`29adddd`) cover RFC-style numbered contents and contents rows whose page number is a separate text object. The page's own "Contents" title may now be a heading.
- **Rejected: flat-document threshold relief.** This was the P5 proposal for the earnings releases doc_009/010, where all text is body size. On those two documents it produced 18 predictions each for 1 match each, and micro F1 fell from 0.688 to 0.678. It was left uncommitted by an interrupted session, measured, and reverted.
- **Sub-body text** (`5d3e621`). No matched heading in the benchmark is set below 0.95 x body size, while 51 predictions were (table-cell fragments such as RP2040's `0 Register`, footnotes in doc_045). Recall at depth <= 2 fell by 4 matches in doc_045 only. Removing its footnotes changed the one-off style threshold there and moved four headings to level 3.
- **Size-aware repeat penalty** (`50b43ba`). Section titles that overview pages repeat in smaller type ("Net-zero carbon" in doc_048) were penalised as repeated titles at their own 50 pt opening heading.

The spec §2.3 score weights and the 0.55 default threshold were not changed.

### Remaining failure patterns (default engine, after tuning)

- **RFC bookmarks nest every section under the title (doc_031-034).** Sections are gold level 2 and subsections gold level 3. The extractor puts sections at level 1 and subsections at level 2, so the subsections count as false positives at depth <= 2. Level accuracy is about 0 for these four documents. They account for 252 of the 413 depth <= 2 false positives. Without them, micro F1 on the other 20 documents is 0.781 (P 0.791, R 0.770). This comes from how the gold is built, not from the extractor, and it was left alone.
- **Earnings releases (doc_009/010, F1 0.00).** All text is body size. Most gold titles (`Page 1 Financials (QTD)`, `CEO Quote`) are labels of page regions and are not printed as headings, so the only rule tried (above) was rejected.
- **Slide decks (doc_050/051, F1 0.07/0.27).** Gold titles are `Slide N: ...`, which never occur in the text.
- **Shallow gold (doc_001, F1 0.05).** The Berkshire annual report has 9 bookmarks. The extractor finds the Form 10-K `Item` headings and the letter's section titles, which the bookmarks do not list.
- **Engineering drawings and "(continued)" titles (doc_041, doc_049).** Package drawing pages (`JG0008A`, `5.08 MAX`) and repeated box titles ending in "(continued)" are still predicted. Each is a small, single-document effect.
