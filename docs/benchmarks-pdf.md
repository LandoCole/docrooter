# PDF benchmarks

## PDF TOC extraction

<!-- toc-benchmark:begin -->
Gold: bookmarks of the 24 corpus PDFs with at least 5 bookmarks (levels shifted so the minimum is 1; items without a target page dropped). Extraction runs with `use_bookmarks=False`. A prediction matches a bookmark at normalised-title ratio >= 0.85 and pages within +/-1, one-to-one and greedy by ratio. The depth <= 2 columns keep only gold and predicted headings of level <= 2, so precision is measured at that depth too. Micro pools the counts; macro is the mean of per-document F1. Level accuracy is measured on matched pairs. Bookmark sets often omit lower-level headings, so precision is pessimistic and recall / F1 at depth <= 2 are the headline numbers.

Environment: Python 3.12.13, arm64, 8 CPUs, pymupdf 1.28.2, pdfplumber 0.11.10, pymupdf4llm 1.28.2. `os.getloadavg()` was 9.8, 15.3, 12.9 (1/5/15 min) when the latest run started. The machine was shared with other jobs, so seconds per document are noisy upper bounds; each engine ran one untimed warm-up document first. Seconds are wall-clock for one `load_pdf` call. Each engine's row keeps the run that produced it: `pymupdf` 2026-10-16 at `e72fca6` (load 9.8, 15.3, 12.9); `pdfplumber` 2026-10-16 at `e72fca6` (load 9.5, 14.5, 12.7); `both` 2026-10-16 at `e72fca6` (load 6.8, 11.1, 11.6); `pymupdf4llm` 2026-10-16 at `5610ada` (load 8.1, 10.8, 14.2).

### Aggregate by engine (depth <= 2 headline, all depths for reference)

| Engine | Docs | P (d<=2) | R (d<=2) | F1 micro (d<=2) | F1 macro (d<=2) | Level acc (d<=2) | P (all) | R (all) | F1 micro (all) | F1 macro (all) | Level acc (all) | s/doc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pymupdf | 24 | 0.633 | 0.791 | 0.703 | 0.591 | 0.770 | 0.550 | 0.905 | 0.685 | 0.634 | 0.596 | 1.53 |
| pdfplumber | 24 | 0.648 | 0.788 | 0.711 | 0.599 | 0.796 | 0.576 | 0.900 | 0.702 | 0.642 | 0.606 | 7.31 |
| both | 24 | 0.657 | 0.785 | 0.716 | 0.604 | 0.770 | 0.595 | 0.900 | 0.717 | 0.656 | 0.597 | 8.58 |
| pymupdf4llm* | 8 | 0.667 | 0.564 | 0.611 | 0.517 | 0.974 | 0.615 | 0.692 | 0.651 | 0.550 | 0.912 | 15.21 |

\* `pymupdf4llm`: limited to the 8 smallest bookmarked documents (by page count); see the matched-subset table below.

### Same 8 documents for every engine (the `pymupdf4llm` subset)

| Engine | P (d<=2) | R (d<=2) | F1 micro (d<=2) | F1 macro (d<=2) | Level acc (d<=2) | F1 micro (all) | s/doc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pymupdf | 0.757 | 0.569 | 0.650 | 0.472 | 0.678 | 0.754 | 0.28 |
| pdfplumber | 0.743 | 0.559 | 0.638 | 0.467 | 0.673 | 0.723 | 1.23 |
| both | 0.776 | 0.564 | 0.653 | 0.481 | 0.675 | 0.749 | 1.56 |
| pymupdf4llm | 0.667 | 0.564 | 0.611 | 0.517 | 0.974 | 0.651 | 15.21 |

### Per document: `pymupdf` (the default engine)

| Doc | Title | Pages | Gold d<=2 | Pred d<=2 | P | R | F1 | Level acc | F1 (all) | s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| doc_001 | Berkshire Hathaway Inc. 2023 Annual Report | 152 | 9 | 29 | 0.03 | 0.11 | 0.05 | 0.00 | 0.24 | 3.03 |
| doc_003 | Amazon.com, Inc. 2023 Annual Report | 92 | 30 | 29 | 0.69 | 0.67 | 0.68 | 0.00 | 0.68 | 1.60 |
| doc_004 | Berkshire Hathaway Inc. Form 10-Q for the qua... | 56 | 22 | 14 | 0.71 | 0.45 | 0.56 | 0.00 | 0.53 | 0.94 |
| doc_005 | Berkshire Hathaway Inc. Form 10-Q for the qua... | 59 | 22 | 14 | 0.71 | 0.45 | 0.56 | 0.00 | 0.53 | 1.03 |
| doc_009 | Amazon.com Announces Second Quarter Results (... | 12 | 18 | 0 | 0.00 | 0.00 | 0.00 | n/a | 0.00 | 0.16 |
| doc_010 | Amazon.com Announces Third Quarter Results (O... | 12 | 18 | 0 | 0.00 | 0.00 | 0.00 | n/a | 0.00 | 0.17 |
| doc_027 | Attention Is All You Need | 15 | 19 | 23 | 0.83 | 1.00 | 0.90 | 1.00 | 0.92 | 0.25 |
| doc_029 | Language Models are Few-Shot Learners | 75 | 32 | 38 | 0.84 | 1.00 | 0.91 | 1.00 | 0.72 | 1.21 |
| doc_030 | On the Opportunities and Risks of Foundation ... | 214 | 39 | 40 | 0.95 | 0.97 | 0.96 | 1.00 | 0.90 | 4.11 |
| doc_031 | RFC 9110: HTTP Semantics | 194 | 30 | 158 | 0.18 | 0.93 | 0.30 | 0.00 | 0.86 | 2.20 |
| doc_032 | RFC 9000: QUIC: A UDP-Based Multiplexed and S... | 151 | 36 | 141 | 0.25 | 0.97 | 0.40 | 0.03 | 0.97 | 1.69 |
| doc_033 | RFC 9239: Updates to ECMAScript Media Types | 13 | 17 | 25 | 0.64 | 0.94 | 0.76 | 0.06 | 0.96 | 0.14 |
| doc_034 | RFC 9457: Problem Details for HTTP APIs | 16 | 19 | 25 | 0.72 | 0.95 | 0.82 | 0.06 | 0.97 | 0.15 |
| doc_038 | Bash Reference Manual, Edition 5.3 | 214 | 70 | 76 | 0.82 | 0.89 | 0.85 | 1.00 | 0.92 | 2.67 |
| doc_039 | GNU Coreutils manual | 326 | 179 | 188 | 0.95 | 0.99 | 0.97 | 1.00 | 0.98 | 3.92 |
| doc_040 | Getting Started with Raspberry Pi Pico-series | 36 | 55 | 61 | 0.90 | 1.00 | 0.95 | 1.00 | 0.93 | 0.92 |
| doc_041 | NE555 / SA555 / SE555 Precision Timers datasheet | 39 | 30 | 45 | 0.44 | 0.67 | 0.53 | 1.00 | 0.61 | 0.37 |
| doc_042 | RP2040 Datasheet | 642 | 96 | 98 | 0.97 | 0.99 | 0.98 | 0.77 | 0.44 | 6.37 |
| doc_045 | Monetary Policy Report, July 5, 2024 | 79 | 23 | 13 | 0.62 | 0.35 | 0.44 | 1.00 | 0.64 | 0.91 |
| doc_046 | Financial Stability Report, April 2024 | 69 | 13 | 14 | 0.93 | 1.00 | 0.96 | 1.00 | 0.96 | 0.67 |
| doc_048 | Google Environmental Report 2024 | 86 | 32 | 24 | 0.62 | 0.47 | 0.54 | 0.93 | 0.35 | 2.86 |
| doc_049 | NVIDIA Sustainability Report, Fiscal Year 2024 | 41 | 30 | 46 | 0.61 | 0.93 | 0.74 | 1.00 | 0.51 | 0.83 |
| doc_050 | Tesla Q2 2024 Update (investor slide deck) | 29 | 29 | 1 | 1.00 | 0.03 | 0.07 | 1.00 | 0.07 | 0.29 |
| doc_051 | NVIDIA Investor Presentation Q2 FY25 (Septemb... | 27 | 27 | 17 | 0.35 | 0.22 | 0.27 | 0.17 | 0.54 | 0.17 |
<!-- toc-benchmark:end -->

### Reading the numbers

After the P5b tuning and the v0.2 review fix for dot-leader tables (see the tuning log below), the default engine (`pymupdf`, what `engine="auto"` resolves to) scores 0.703 micro F1 at depth <= 2 (P 0.633, R 0.791), up from 0.478 (P 0.369, R 0.678) in P5. Macro F1 rose from 0.464 to 0.591. Level accuracy on matched pairs fell from 0.802 to 0.770 at depth <= 2, because many more headings now match (708 vs 607) and some of the new matches carry a gold level the extractor gets wrong. Over all depths it rose from 0.548 to 0.596. These figures were measured on the same 24 documents that were used to choose the rules, so they are optimistic; see the disclosure below. `pdfplumber` (0.711) and `both` (0.716) are slightly ahead on quality but take about 5x longer per document on this loaded machine. The `pymupdf4llm` row is the P5 run (2026-10-16, commit `5610ada`). It was not re-run, because that engine uses the v0.1 Markdown path and does not go through the tuned detector. On its 8-document subset the tuned `pymupdf` now leads on F1 (0.650 vs 0.611 micro) while `pymupdf4llm` keeps far better level accuracy (0.97 vs 0.68).

**Speed: the v0.2 criterion was not demonstrated.** The v0.2 spec asks for under 1.5 s per 100 pages with the default engine. The P5b run measured 1.76 s/100 pages (46.7 s over the 2,649 pages of the 24 documents, load average about 8). The re-run after the dot-leader fix (`e72fca6`) measured 1.38 s/100 pages (36.7 s), but it is one run on a shared machine whose load average was 9.8 at its start, so neither figure is a clean measurement. The speed will be re-measured on an idle machine in v0.3.

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
| Review fix 2: numbering-word exception also in the 30% rule; heading-like = bold OR >= 1.15 x body | `3e69550` | 0.632 | 0.791 | 0.702 | 0.592 | 0.770 | 0.684 |
| Review fix 3: page expressions ("Page 5 of 80") masked first; page-numbered lines fully generalised | `ed6d42a` | 0.632 | 0.791 | 0.702 | 0.592 | 0.770 | 0.684 |
| *Rejected:* section running lines must recur on max(3, 10% of pages) | (not committed) | 0.458 | 0.791 | 0.580 | 0.564 | 0.770 | 0.629 |
| v0.2 review fix: bare dot-leader lines count only on a contents-headed page or with >= 3 page-numbered entries; only high-confidence contents pages leave the body | `e72fca6` | 0.633 | 0.791 | 0.703 | 0.591 | 0.770 | 0.685 |

Notes on the rules:

- **Section running lines** (`23f8cd2`). An edge-band line whose text, with any leading or trailing page number removed, recurs on at least 3 pages is now excluded as a running line, even far below 30% of a long document. RP2040 (doc_042) dropped from 742 to 151 depth <= 2 predictions. This adds to the ">= 30% of pages" rule in the global constraints and does not replace it.
- **One-off styles** (`bf04ccb`). A style used by fewer than min(3, 10%) of the kept headings (cover titles, front-matter one-offs) takes the level of the next recurring style instead of a level of its own. This fixed the level-5 `Item N.` headings in doc_003, which went from F1 0.00 to 0.67. It is also the main cause of the fall in level accuracy. The 10% count threshold is unstable on documents with about 20 headings (doc_045 changes levels when a few candidates are removed).
- **Edge-band copies** (`94a5cbf`) and **wrapped titles** (`fbd900a`) are small precision and recall fixes.
- **Contents pages without page numbers** (`29adddd`) cover RFC-style numbered contents and contents rows whose page number is a separate text object. The page's own "Contents" title may now be a heading.
- **Rejected: flat-document threshold relief.** This was the P5 proposal for the earnings releases doc_009/010, where all text is body size. On those two documents it produced 18 predictions each for 1 match each, and micro F1 fell from 0.688 to 0.678. It was left uncommitted by an interrupted session, measured, and reverted.
- **Sub-body text** (`5d3e621`). No matched heading in the benchmark is set below 0.95 x body size, while 51 predictions were (table-cell fragments such as RP2040's `0 Register`, footnotes in doc_045). Recall at depth <= 2 fell by 4 matches in doc_045 only. Removing its footnotes changed the one-off style threshold there and moved four headings to level 3.
- **Size-aware repeat penalty** (`50b43ba`). Section titles that overview pages repeat in smaller type ("Net-zero carbon" in doc_048) were penalised as repeated titles at their own 50 pt opening heading.

- **Review fix round 1** (`0e8c75e`). This commit adds correctness guards that do not change the benchmark score. First, a trailing number after a numbering word (chapter, part, section, item, appendix, article, schedule, exhibit, annex, table, figure) is no longer stripped as a page number. Before, "Chapter 1/2/3" or "Item 1/2/3" set at page tops would collapse into one running-line key and be excluded. Second, a recurring edge line that is bold and at least 1.15 x body size is excluded only when its copies sit at the same height on 3 pages, so repeated slide titles survive. Review fix round 2 (`3e69550`) closes two gaps. The >= 30%-of-pages rule, which replaces digits with `#`, now also keeps a number that follows a numbering word, so "Chapter 1-4" opening 3-4 pages of a 9-12 page document is no longer excluded. A line now counts as heading-like if it is bold *or* at least 1.15 x body size, instead of both, so the same-height test now also covers large non-bold titles and bold titles at body size. Review fix round 3 (`ed6d42a`) restores footers such as "Section N · Page P of 80". Page expressions are masked before the numbering-word exception, and a line that carries a page number has all of its numbers generalised. A proposed minimum of max(3, 10% of pages) was measured and rejected: RP2040's sections span 3-10 of its 642 pages, and with that minimum its depth <= 2 predictions rose from 98 to 523.

- **v0.2 review fix: dot-leader tables** (`e72fca6`). A line ending in dot leaders without a page number ("Revenues . . . . .") counted as a contents entry, so financial tables with leader-dotted labels made whole pages "contents pages", and their text was deleted from the body: about 17.8% of doc_007's text (Berkshire tables) and 12.7% of doc_013's (ownership and pay tables; measured as the share of non-excluded line characters). Such bare-leader lines now count only on a page headed "Contents" / "Table of Contents" (within its first 5 lines) or one with at least 3 page-numbered entries. Only high-confidence contents pages (a contents heading, or at least 5 page-numbered entries) are removed from body text; other contents pages are only kept out of the headings. This is a correctness fix, not a tuning step, and was kept although macro F1 fell by 0.001: doc_045 gained one false-positive heading from a projections table (F1 0.46 to 0.44), while doc_001 lost three false positives (F1 0.049 to 0.053). Micro F1 rose from 0.702 to 0.703. All 298 lines of doc_007's and 348 of doc_013's former table "contents pages" are now in the body. Two RFC documents (doc_031, doc_032) have unheaded continuation pages of a numbered contents list; these are still never headings but now stay in the body text.

The spec §2.3 score weights and the 0.55 default threshold were not changed.

### Remaining failure patterns (default engine, after tuning)

- **RFC bookmarks nest every section under the title (doc_031-034).** Sections are gold level 2 and subsections gold level 3. The extractor puts sections at level 1 and subsections at level 2, so the subsections count as false positives at depth <= 2. Level accuracy is about 0 for these four documents. They account for 252 of the 413 depth <= 2 false positives. Without them, micro F1 on the other 20 documents is 0.781 (P 0.791, R 0.770). This comes from how the gold is built, not from the extractor, and it was left alone.
- **Earnings releases (doc_009/010, F1 0.00).** All text is body size. Most gold titles (`Page 1 Financials (QTD)`, `CEO Quote`) are labels of page regions and are not printed as headings, so the only rule tried (above) was rejected.
- **Slide decks (doc_050/051, F1 0.07/0.27).** Gold titles are `Slide N: ...`, which never occur in the text.
- **Shallow gold (doc_001, F1 0.05).** The Berkshire annual report has 9 bookmarks. The extractor finds the Form 10-K `Item` headings and the letter's section titles, which the bookmarks do not list.
- **Engineering drawings and "(continued)" titles (doc_041, doc_049).** Package drawing pages (`JG0008A`, `5.08 MAX`) and repeated box titles ending in "(continued)" are still predicted. Each is a small, single-document effect.

### Known limitations

- **Headings set below body size are never detected.** Since `5d3e621`, a line smaller than the body size by more than 0.5 pt is not a heading candidate. Small-caps headings, whose measured size is often the small-capital glyph size, and compact run-in headings in documents with large body text are lost even when they are bold and isolated. No matched heading in this benchmark is below 0.95 x body, so the rule costs nothing here, but other corpora may differ.
- **Section running lines need only 3 pages.** An edge-band line that is neither bold nor at least 1.15 x body size is excluded when its text repeats on 3 pages, however long the document. A bold or large one is excluded when it also sits at the same height on 3 pages. A genuine heading printed in the top or bottom 8% of the page at the same height on 3 or more pages with identical text (for example, a recurring "Notes" page title) is therefore lost.
- **Numbered openers in short documents.** A number after a numbering word (chapter, part, section, item, appendix, article, schedule, exhibit, annex, table, figure) is kept in both running-line keys when the line carries no page number. A line with a page number ("Section 3 · Page 12 of 80", "Chapter 3 · 12") is generalised to a running-line key as before. Two cases still collapse in the >= 30%-of-pages rule. One is openers whose number follows a word outside the list, such as "Lecture 1" / "Lecture 2" or "Unit 3". The other is plural numbering words, such as "Figures 1-2", "Tables 3" or "Items 1A". Both are excluded when they sit in the edge band of at least max(3, ceil(0.3 x pages)) pages. In practice that means short documents, or documents that put such a line at the top of 30% or more of their pages.

## Document-type classification

**Headline.** On 51 real PDFs and 18 defined leaf types, TypeSafe Jev given docrooter's DocumentView (`jev-view`) is the most accurate system: 49/51 correct (leaf accuracy 0.961, macro F1 0.944), at 1.52 s/doc end to end and $0.0044 for the whole corpus. Jev on naive raw-text chunks (`jev-chunks`) gets 47/51 (0.922) using 10.7x the input tokens. Every view-based system now receives the same view text, rendered once with a shared word-based token counter (see *Views* below). The `jev-view`, `ultrabert-view`, `ultrabert-hybrid` and `bge-view` rows were re-run after the v0.2 contents-page fix (dot-leader tables no longer delete body text); no accuracy changed (see *Provenance*).

With no API:

- **Best honest local system:** the P9 configuration `ultrabert-hybrid` (0.6 view + 0.4 sections), chosen before any P9b result, gets 42/51 (0.824, macro F1 0.765). The plain `ultrabert-view` gets 39/51 (0.765).
- **Best zero-shot, dev-tuned:** `ultrabert-view-tuned`, a lead-first DocumentView chosen on these same 51 documents, gets 43/51 (0.843, macro F1 0.808). The leave-one-out estimate of making that choice is also 43/51. That estimate is only a lower bound on the optimism, because the grid it chooses from was itself designed after seeing results on these documents.
- **Promoted leave-one-out system:** `ultrabert-bge-loo`, a mix of the UltraBERT and bge zero-shot distributions with the weight chosen by inner LOO, gets 38/51 (0.745). This is 4 documents *below* its P9b result under per-backend rendering (0.824). It is now no better than `ultrabert-view`. The pairing, the mixing method and the decision to promote it were all made after seeing results on these documents, so even its earlier figure was mildly optimistic.
- **Trained head:** the logistic head with C tuned by inner LOO (`ultrabert-loo`, which replaces P9's C=1 row at 0.235) gets 36/51 (0.706). No few-shot head beats zero-shot at 1-3 labelled documents per class.

The gap from the best local system to Jev is 6-7 documents (12-14 points). These are single-run numbers on a small corpus, and rendering alone moves UltraBERT systems by up to 4 documents.

<!-- doctype-benchmark:begin -->
Corpus: 51 real PDFs from `benchmarks/corpus/` with gold labels from the manifest, 18 leaf types in 6 groups (`builtin:doc_types`; every label has a definition, see `benchmarks/corpus/README.md`). Classes have 2-4 documents each, so per-class numbers are anecdotal and one document moves leaf accuracy by 2.0 points. The 95% intervals are percentile bootstraps over documents (1000 resamples, seed 0). Every system labels every document (threshold 0).

Environment: Python 3.12.13, arm64, 8 CPUs, device `mps` (MPS probe matmul took 0.64 s), torch 2.14.1, sentence-transformers 6.1.0, scikit-learn 1.9.1, typesafe-sdk 0.7.5, pymupdf 1.28.2, Jev `jev-1.13.0`. The load average at the start of each system is listed below; the machine was shared with another benchmark, so seconds are noisy upper bounds.

Timing: *parse* is one `load_pdf` call with the default engine (pymupdf), cached and shared by every system except `jev-chunks`, whose parse is pymupdf `page.get_text()` over all pages. *Classify* for the `classify_documents` systems (`*-view`, `ultrabert-hybrid`) is batch throughput: the wall time of the classify calls divided by the number of documents (local backends: one call over all documents; Jev: chunks of 8 documents with 8 concurrent requests, so each chunk's rows are saved before the next is paid for). For the leave-one-out systems (`*-loo`) it is view rendering, embedding and zero-shot scoring divided by the number of documents; the fit per fold (with its inner leave-one-out tuning) is listed separately. For `jev-chunks` it is per-document latency: the wall time of each document's concurrent chunk requests. The table marks each figure as `(batch)` or `(latency)`. *Load* is a warm-up excluded from the averages: model load plus one classification of a tiny synthetic document (the label vectors are encoded there). Jev cost is input tokens x $0.042/M (output tokens are free), with tokens as reported by the API (`usage.input_tokens`).

Views: every `*-view` system, `ultrabert-hybrid` (its view half), the leave-one-out systems and the P9b embedding experiments receive the same DocumentView text per document and view config, rendered once with the shared word-based counter (`docrooter.views.estimate_tokens`, 1.3 tokens per word) instead of each backend's own tokenizer. Each row records the SHA-256 prefix of the text the system saw (`view_sha`), and `pdf.json` lists the shared texts' hashes under `views`. Whatever a model's own tokenizer cannot fit is cut by that model as usual (bge-small and the NLI model at 512 tokens). For each system, *Views differing* counts the documents whose view would have been different had it been rendered with the backend's own counter, which was the P9 behaviour.

### Aggregate

| System | What | Leaf acc | 95% CI | Level-1 acc | Macro P | Macro R | Macro F1 | Parse s/doc | Classify s/doc | Total s/doc | Load s | Jev input tokens | Jev cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ultrabert-view` | UltraBERT-Embed-150M, DocumentView, Flat (zero-shot) | 0.765 | 0.65-0.86 | 0.902 | 0.689 | 0.759 | 0.709 | 1.47 | 0.616 (batch) | 2.08 | 12.2 | - | - |
| `ultrabert-hybrid` | UltraBERT-Embed-150M, 0.6 view + 0.4 sections (zero-shot) | 0.824 | 0.73-0.92 | 0.922 | 0.739 | 0.806 | 0.765 | 1.47 | 3.094 (batch) | 4.56 | 2.6 | - | - |
| `bge-view` | bge-small-en-v1.5, DocumentView, Flat (zero-shot) | 0.608 | 0.47-0.75 | 0.784 | 0.584 | 0.611 | 0.541 | 1.47 | 0.034 (batch) | 1.50 | 2.0 | - | - |
| `nli-view` | deberta-v3-base-zeroshot-v2.0 NLI, DocumentView, Flat (zero-shot) | 0.569 | 0.43-0.69 | 0.765 | 0.510 | 0.532 | 0.485 | 1.60 | 15.076 (batch) | 16.68 | 5.2 | - | - |
| `jev-view` | Jev jev-1.13.0, DocumentView as state, one Choice over 18 leaves | 0.961 | 0.90-1.00 | 0.961 | 0.968 | 0.944 | 0.944 | 1.47 | 0.058 (batch) | 1.52 | 0.6 | 104,372 | $0.0044 |
| `jev-chunks` | Jev jev-1.13.0, raw page text, <=8 chunks of ~1,500 tokens, mean of per-chunk Choices (no docrooter) | 0.922 | 0.84-0.98 | 0.961 | 0.947 | 0.912 | 0.911 | 0.49 | 0.320 (latency) | 0.81 | 0.0 | 1,121,481 | $0.0471 |
| `ultrabert-view-tuned` | UltraBERT-Embed-150M, lead-first DocumentView (1,024 tokens, 400 lead words), Flat (zero-shot; view chosen on this corpus, dev-tuned) | 0.843 | 0.73-0.94 | 0.941 | 0.818 | 0.833 | 0.808 | 1.60 | 1.039 (batch) | 2.64 | 2.8 | - | - |
| `ultrabert-loo` | UltraBERT-Embed-150M view embeddings + logistic regression, C in {1, 10, 100, 1000} by inner leave-one-out (trained, leave-one-out) | 0.706 | 0.59-0.82 | 0.902 | 0.615 | 0.690 | 0.647 | 1.60 | 0.474 (batch) | 2.08 | 1.6 | - | - |
| `ultrabert-bge-loo` | Zero-shot UltraBERT-view and bge-view leaf distributions mixed, weight in {0, .25, .5, .75, 1} by inner leave-one-out (leave-one-out) | 0.745 | 0.63-0.86 | 0.882 | 0.656 | 0.741 | 0.680 | 1.60 | 0.485 (batch) | 2.09 | 3.5 | - | - |

Run metadata per system:

- `ultrabert-view`: 2026-10-16 at `53090c0`, device `mps`, load average 19.8, 14.6, 11.8 at start; views differing from the backend's own counter: 25 of 51.
- `ultrabert-hybrid`: 2026-10-16 at `53090c0`, device `mps`, load average 12.6, 13.5, 11.5 at start; views differing from the backend's own counter: 25 of 51.
- `bge-view`: 2026-10-16 at `53090c0`, device `mps`, load average 4.1, 9.4, 10.2 at start; views differing from the backend's own counter: 16 of 51.
- `nli-view`: 2026-10-16 at `196d4d0+dirty`, device `mps`, load average 5.7, 6.3, 6.5 at start; views differing from the backend's own counter: 13 of 51.
- `jev-view`: 2026-10-16 at `53090c0`, device `n/a`, load average 23.6, 15.1, 11.9 at start; views differing from the backend's own counter: 0 of 51.
- `jev-chunks`: 2026-10-16 at `3bd85f2`, device `n/a`, load average 4.4, 6.5, 13.9 at start; 331 chunks sent (6.5 per document).
- `ultrabert-view-tuned`: 2026-10-16 at `aad5b3f`, device `mps`, load average 4.7, 7.0, 7.3 at start; views differing from the backend's own counter: 20 of 51.
- `ultrabert-loo`: 2026-10-16 at `aad5b3f`, device `mps`, load average 4.6, 6.5, 7.1 at start; mean fit 3.29 s per fold (inner LOO included); hyperparameter chosen per fold: 1000.0 x50, 100.0 x1; views differing from the backends' own counters: UltraBERT 25 of 51.
- `ultrabert-bge-loo`: 2026-10-16 at `6d0a6eb`, device `mps`, load average 8.3, 8.2, 7.7 at start; mean fit 0.00 s per fold (inner LOO included); hyperparameter chosen per fold: 0.5 x48, 0.25 x3; views differing from the backends' own counters: UltraBERT 25, bge 18 of 51.

### Configurations tried (P9b)

Every configuration evaluated in P9b (2026-10-16 at `6d0a6eb`, device `mps`), with its leaf accuracy, 95% bootstrap CI and macro F1 on the same 51 documents. *Zero-shot, dev-tuned* configs need no labels, but each was scored on this corpus, so the best of them is an optimistic estimate. *Leave-one-out (honest)* systems predict each document from the other 50 only; every hyperparameter is chosen by an inner leave-one-out over those 50 (ties: lower inner log-loss, then the first value), so the held-out document never influences its own prediction. Their features are the default `DocumentView()` embeddings with the P9 instruction, fixed before P9b. *Diagnostic* rows fix a hyperparameter and are shown only to explain the tuned ones. All of them use the shared `estimate_tokens` view texts (see *Views* above).

**Zero-shot, dev-tuned**

| Config | What | Leaf acc | 95% CI | Level-1 acc | Macro F1 | Chosen |
| --- | --- | --- | --- | --- | --- | --- |
| `ultrabert/default512/what/label-def` | ultrabert, view default512, instruction what, labels label-def | 0.765 | 0.65-0.86 | 0.902 | 0.709 | - |
| `ultrabert/default512/what/def-first` | ultrabert, view default512, instruction what, labels def-first | 0.804 | 0.69-0.90 | 0.922 | 0.749 | - |
| `ultrabert/default512/identify/label-def` | ultrabert, view default512, instruction identify, labels label-def | 0.745 | 0.63-0.86 | 0.863 | 0.689 | - |
| `ultrabert/default512/identify/def-first` | ultrabert, view default512, instruction identify, labels def-first | 0.765 | 0.65-0.86 | 0.882 | 0.714 | - |
| `ultrabert/lead1024/what/label-def` | ultrabert, view lead1024, instruction what, labels label-def | 0.843 | 0.73-0.94 | 0.941 | 0.808 | - |
| `ultrabert/lead1024/what/def-first` | ultrabert, view lead1024, instruction what, labels def-first | 0.824 | 0.71-0.92 | 0.941 | 0.769 | - |
| `ultrabert/lead1024/identify/label-def` | ultrabert, view lead1024, instruction identify, labels label-def | 0.784 | 0.67-0.88 | 0.882 | 0.731 | - |
| `ultrabert/lead1024/identify/def-first` | ultrabert, view lead1024, instruction identify, labels def-first | 0.824 | 0.71-0.92 | 0.922 | 0.771 | - |
| `ultrabert/toc1024/what/label-def` | ultrabert, view toc1024, instruction what, labels label-def | 0.784 | 0.67-0.88 | 0.902 | 0.725 | - |
| `ultrabert/toc1024/what/def-first` | ultrabert, view toc1024, instruction what, labels def-first | 0.824 | 0.71-0.92 | 0.922 | 0.765 | - |
| `ultrabert/toc1024/identify/label-def` | ultrabert, view toc1024, instruction identify, labels label-def | 0.765 | 0.65-0.88 | 0.863 | 0.703 | - |
| `ultrabert/toc1024/identify/def-first` | ultrabert, view toc1024, instruction identify, labels def-first | 0.784 | 0.67-0.88 | 0.882 | 0.730 | - |
| `ultrabert/lead512/what/label-def` | ultrabert, view lead512, instruction what, labels label-def | 0.804 | 0.69-0.90 | 0.941 | 0.782 | - |
| `ultrabert/lead512/what/def-first` | ultrabert, view lead512, instruction what, labels def-first | 0.804 | 0.69-0.90 | 0.941 | 0.769 | - |
| `ultrabert/default1024-lead400/what/label-def` | ultrabert, view default1024-lead400, instruction what, labels label-def | 0.784 | 0.67-0.88 | 0.941 | 0.720 | - |
| `ultrabert/default1024-lead400/what/def-first` | ultrabert, view default1024-lead400, instruction what, labels def-first | 0.784 | 0.67-0.88 | 0.922 | 0.722 | - |
| `bge/default512/what/label-def` | bge, view default512, instruction what, labels label-def | 0.608 | 0.47-0.75 | 0.784 | 0.538 | - |
| `bge/default512/what/def-first` | bge, view default512, instruction what, labels def-first | 0.569 | 0.43-0.71 | 0.725 | 0.503 | - |
| `bge/lead512/what/label-def` | bge, view lead512, instruction what, labels label-def | 0.569 | 0.43-0.71 | 0.686 | 0.529 | - |
| `bge/lead512/what/def-first` | bge, view lead512, instruction what, labels def-first | 0.549 | 0.41-0.69 | 0.686 | 0.493 | - |

**Leave-one-out (honest)**

| Config | What | Leaf acc | 95% CI | Level-1 acc | Macro F1 | Chosen |
| --- | --- | --- | --- | --- | --- | --- |
| `zero-shot-config-loo` | zero-shot UltraBERT config picked from the 16 above by accuracy on the other 50 documents (honest estimate of the dev-tuned choice) | 0.843 | 0.73-0.94 | 0.941 | 0.808 | ultrabert/lead1024/what/label-def: 51 |
| `logistic-cv` | logistic regression, C in {1, 10, 100, 1000} by inner LOO | 0.706 | 0.59-0.82 | 0.902 | 0.647 | 1000.0: 50, 100.0: 1 |
| `centroid` | nearest class centroid (cosine) | 0.667 | 0.53-0.78 | 0.882 | 0.633 | - |
| `knn-cv` | cosine kNN, k in {1, 3, 5} by inner LOO | 0.765 | 0.65-0.88 | 0.961 | 0.739 | 1: 51 |
| `prior-centroid-cv` | (1 - w) zero-shot + w centroid softmax, w in {0, .25, .5, .75, 1} by inner LOO | 0.725 | 0.61-0.84 | 0.941 | 0.672 | 0.25: 46, 0.5: 4, 0.0: 1 |
| `examples` | package examples mechanism: leaf vector = unit mean of label vector and training views | 0.686 | 0.55-0.80 | 0.922 | 0.653 | - |
| `ensemble-cv` | (1 - w) UltraBERT zero-shot + w bge zero-shot, w in {0, .25, .5, .75, 1} by inner LOO | 0.745 | 0.63-0.86 | 0.882 | 0.680 | 0.5: 48, 0.25: 3 |

**Diagnostic (fixed hyperparameter, not tuned honestly)**

| Config | What | Leaf acc | 95% CI | Level-1 acc | Macro F1 | Chosen |
| --- | --- | --- | --- | --- | --- | --- |
| `logistic-C1` | logistic regression, fixed C=1 | 0.255 | 0.14-0.37 | 0.608 | 0.097 | - |
| `logistic-C10` | logistic regression, fixed C=10 | 0.588 | 0.45-0.73 | 0.863 | 0.508 | - |
| `logistic-C100` | logistic regression, fixed C=100 | 0.667 | 0.53-0.80 | 0.882 | 0.618 | - |
| `logistic-C1000` | logistic regression, fixed C=1000 | 0.706 | 0.59-0.82 | 0.902 | 0.647 | - |
| `knn-k1` | cosine kNN, fixed k=1 | 0.765 | 0.65-0.88 | 0.961 | 0.739 | - |
| `knn-k3` | cosine kNN, fixed k=3 | 0.608 | 0.47-0.75 | 0.902 | 0.549 | - |
| `knn-k5` | cosine kNN, fixed k=5 | 0.529 | 0.39-0.67 | 0.824 | 0.411 | - |
| `ensemble-w0.25` | ensemble, fixed bge weight 0.25 | 0.784 | 0.67-0.88 | 0.902 | 0.725 | - |
| `ensemble-w0.5` | ensemble, fixed bge weight 0.5 | 0.804 | 0.69-0.90 | 0.922 | 0.757 | - |
| `ensemble-w0.75` | ensemble, fixed bge weight 0.75 | 0.725 | 0.61-0.84 | 0.902 | 0.650 | - |

### Per class

Best honest docrooter system by macro F1 (zero-shot: `ultrabert-hybrid`; any, leave-one-out included: `ultrabert-hybrid`), `jev-view`, and the best dev-tuned system `ultrabert-view-tuned` (its settings were chosen on this corpus, so its numbers are optimistic). P / R / F1 per gold class.

| Class | n | `ultrabert-hybrid` P / R / F1 | `jev-view` P / R / F1 | `ultrabert-view-tuned` (dev-tuned) P / R / F1 |
| --- | --- | --- | --- | --- |
| financial.annual_report | 3 | 0.67 / 0.67 / 0.67 | 1.00 / 1.00 / 1.00 | 0.50 / 0.33 / 0.40 |
| financial.quarterly_report | 2 | 1.00 / 1.00 / 1.00 | 0.67 / 1.00 / 0.80 | 1.00 / 1.00 / 1.00 |
| financial.shareholder_letter | 3 | 0.50 / 0.33 / 0.40 | 1.00 / 1.00 / 1.00 | 0.67 / 0.67 / 0.67 |
| financial.earnings_release | 2 | 0.67 / 1.00 / 0.80 | 1.00 / 1.00 / 1.00 | 0.67 / 1.00 / 0.80 |
| financial.prospectus | 2 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| financial.proxy_statement | 2 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 0.67 / 1.00 / 0.80 |
| legal.contract | 2 | 0.00 / 0.00 / 0.00 | 1.00 / 0.50 / 0.67 | 1.00 / 0.50 / 0.67 |
| legal.terms_policy | 3 | 0.75 / 1.00 / 0.86 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| legal.court_opinion | 4 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| legal.legislation | 3 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| research.research_paper | 4 | 0.67 / 1.00 / 0.80 | 1.00 / 1.00 / 1.00 | 0.67 / 1.00 / 0.80 |
| research.technical_standard | 4 | 0.80 / 1.00 / 0.89 | 1.00 / 1.00 / 1.00 | 0.80 / 1.00 / 0.89 |
| research.white_paper | 3 | 0.00 / 0.00 / 0.00 | 1.00 / 1.00 / 1.00 | 0.00 / 0.00 / 0.00 |
| technical.user_manual | 3 | 1.00 / 1.00 / 1.00 | 0.75 / 1.00 / 0.86 | 1.00 / 1.00 / 1.00 |
| technical.datasheet | 4 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| government.government_report | 3 | 0.75 / 1.00 / 0.86 | 1.00 / 1.00 / 1.00 | 0.75 / 1.00 / 0.86 |
| corporate.sustainability_report | 2 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| corporate.investor_presentation | 2 | 0.50 / 0.50 / 0.50 | 1.00 / 0.50 / 0.67 | 1.00 / 0.50 / 0.67 |

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
| 9 court_opinion |  |  |  |  |  |  |  |  | **4** |  |  |  |  |  |  |  |  |  |
| 10 legislation |  |  |  |  |  |  |  |  |  | **3** |  |  |  |  |  |  |  |  |
| 11 research_paper |  |  |  |  |  |  |  |  |  |  | **4** |  |  |  |  |  |  |  |
| 12 technical_standard |  |  |  |  |  |  |  |  |  |  |  | **4** |  |  |  |  |  |  |
| 13 white_paper |  |  |  |  |  |  |  |  |  |  | 2 |  |  |  |  | 1 |  |  |
| 14 user_manual |  |  |  |  |  |  |  |  |  |  |  |  |  | **3** |  |  |  |  |
| 15 datasheet |  |  |  |  |  |  |  |  |  |  |  |  |  |  | **4** |  |  |  |
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

`ultrabert-view-tuned` (dev-tuned):

| Gold \ Pred | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 annual_report | **1** |  | 1 |  |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |
| 2 quarterly_report |  | **2** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 3 shareholder_letter | 1 |  | **2** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 4 earnings_release |  |  |  | **2** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 5 prospectus |  |  |  |  | **2** |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 6 proxy_statement |  |  |  |  |  | **2** |  |  |  |  |  |  |  |  |  |  |  |  |
| 7 contract |  |  |  |  |  |  | **1** |  |  |  |  | 1 |  |  |  |  |  |  |
| 8 terms_policy |  |  |  |  |  |  |  | **3** |  |  |  |  |  |  |  |  |  |  |
| 9 court_opinion |  |  |  |  |  |  |  |  | **4** |  |  |  |  |  |  |  |  |  |
| 10 legislation |  |  |  |  |  |  |  |  |  | **3** |  |  |  |  |  |  |  |  |
| 11 research_paper |  |  |  |  |  |  |  |  |  |  | **4** |  |  |  |  |  |  |  |
| 12 technical_standard |  |  |  |  |  |  |  |  |  |  |  | **4** |  |  |  |  |  |  |
| 13 white_paper |  |  |  |  |  |  |  |  |  |  | 2 |  |  |  |  | 1 |  |  |
| 14 user_manual |  |  |  |  |  |  |  |  |  |  |  |  |  | **3** |  |  |  |  |
| 15 datasheet |  |  |  |  |  |  |  |  |  |  |  |  |  |  | **4** |  |  |  |
| 16 government_report |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | **3** |  |  |
| 17 sustainability_report |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | **2** |  |
| 18 investor_presentation |  |  |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  | **1** |

### Per document

✓ correct leaf, ✗ wrong. Predictions show the leaf id without its group.

| Doc | Title | Pages | Gold | `ultrabert-view` | `ultrabert-hybrid` | `bge-view` | `nli-view` | `jev-view` | `jev-chunks` | `ultrabert-view-tuned` | `ultrabert-loo` | `ultrabert-bge-loo` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| doc_001 | Berkshire Hathaway Inc. 2023 Annual Report | 152 | annual_report | annual_report ✓ | annual_report ✓ | proxy_statement ✗ | investor_presentation ✗ | annual_report ✓ | annual_report ✓ | annual_report ✓ | shareholder_letter ✗ | shareholder_letter ✗ |
| doc_002 | Tesla, Inc. Annual Report on Form 10-K for fi... | 130 | annual_report | proxy_statement ✗ | annual_report ✓ | proxy_statement ✗ | annual_report ✓ | annual_report ✓ | annual_report ✓ | proxy_statement ✗ | quarterly_report ✗ | proxy_statement ✗ |
| doc_003 | Amazon.com, Inc. 2023 Annual Report | 92 | annual_report | shareholder_letter ✗ | shareholder_letter ✗ | shareholder_letter ✗ | shareholder_letter ✗ | annual_report ✓ | annual_report ✓ | shareholder_letter ✗ | shareholder_letter ✗ | shareholder_letter ✗ |
| doc_004 | Berkshire Hathaway Inc. Form 10-Q for the qua... | 56 | quarterly_report | quarterly_report ✓ | quarterly_report ✓ | proxy_statement ✗ | quarterly_report ✓ | quarterly_report ✓ | quarterly_report ✓ | quarterly_report ✓ | quarterly_report ✓ | quarterly_report ✓ |
| doc_005 | Berkshire Hathaway Inc. Form 10-Q for the qua... | 59 | quarterly_report | quarterly_report ✓ | quarterly_report ✓ | proxy_statement ✗ | prospectus ✗ | quarterly_report ✓ | quarterly_report ✓ | quarterly_report ✓ | quarterly_report ✓ | quarterly_report ✓ |
| doc_006 | Berkshire Hathaway Chairman's Letter to Share... | 16 | shareholder_letter | investor_presentation ✗ | investor_presentation ✗ | shareholder_letter ✓ | shareholder_letter ✓ | shareholder_letter ✓ | shareholder_letter ✓ | shareholder_letter ✓ | annual_report ✗ | investor_presentation ✗ |
| doc_007 | Berkshire Hathaway Chairman's Letter to Share... | 22 | shareholder_letter | annual_report ✗ | annual_report ✗ | earnings_release ✗ | annual_report ✗ | shareholder_letter ✓ | shareholder_letter ✓ | annual_report ✗ | annual_report ✗ | annual_report ✗ |
| doc_008 | Amazon 2020 Shareholder Letter with the 1997 ... | 10 | shareholder_letter | shareholder_letter ✓ | shareholder_letter ✓ | shareholder_letter ✓ | prospectus ✗ | shareholder_letter ✓ | shareholder_letter ✓ | shareholder_letter ✓ | annual_report ✗ | shareholder_letter ✓ |
| doc_009 | Amazon.com Announces Second Quarter Results (... | 12 | earnings_release | earnings_release ✓ | earnings_release ✓ | earnings_release ✓ | quarterly_report ✗ | earnings_release ✓ | quarterly_report ✗ | earnings_release ✓ | earnings_release ✓ | earnings_release ✓ |
| doc_010 | Amazon.com Announces Third Quarter Results (O... | 12 | earnings_release | earnings_release ✓ | earnings_release ✓ | earnings_release ✓ | quarterly_report ✗ | earnings_release ✓ | earnings_release ✓ | earnings_release ✓ | earnings_release ✓ | earnings_release ✓ |
| doc_011 | Erste Abwicklungsanstalt Debt Issuance Progra... | 113 | prospectus | prospectus ✓ | prospectus ✓ | prospectus ✓ | prospectus ✓ | prospectus ✓ | prospectus ✓ | prospectus ✓ | prospectus ✓ | prospectus ✓ |
| doc_012 | EnBW Debt Issuance Programme Prospectus 2024 | 222 | prospectus | prospectus ✓ | prospectus ✓ | prospectus ✓ | prospectus ✓ | prospectus ✓ | prospectus ✓ | prospectus ✓ | prospectus ✓ | prospectus ✓ |
| doc_013 | Postal Realty Trust 2025 Notice of Annual Mee... | 63 | proxy_statement | proxy_statement ✓ | proxy_statement ✓ | proxy_statement ✓ | investor_presentation ✗ | proxy_statement ✓ | proxy_statement ✓ | proxy_statement ✓ | proxy_statement ✓ | proxy_statement ✓ |
| doc_014 | WisdomTree, Inc. 2024 Notice of Annual Meetin... | 161 | proxy_statement | proxy_statement ✓ | proxy_statement ✓ | proxy_statement ✓ | investor_presentation ✗ | proxy_statement ✓ | proxy_statement ✓ | proxy_statement ✓ | proxy_statement ✓ | proxy_statement ✓ |
| doc_015 | Sample Non-Disclosure Agreement for condition... | 4 | contract | government_report ✗ | terms_policy ✗ | proxy_statement ✗ | government_report ✗ | user_manual ✗ | contract ✓ | contract ✓ | terms_policy ✗ | proxy_statement ✗ |
| doc_016 | NIST Cooperative Research and Development Agr... | 10 | contract | technical_standard ✗ | technical_standard ✗ | technical_standard ✗ | contract ✓ | contract ✓ | contract ✓ | technical_standard ✗ | datasheet ✗ | technical_standard ✗ |
| doc_017 | Apple Privacy Policy (updated July 30, 2025) | 1 | terms_policy | terms_policy ✓ | terms_policy ✓ | terms_policy ✓ | terms_policy ✓ | terms_policy ✓ | terms_policy ✓ | terms_policy ✓ | terms_policy ✓ | terms_policy ✓ |
| doc_018 | Terms of Use (GoWhere support service) | 19 | terms_policy | terms_policy ✓ | terms_policy ✓ | terms_policy ✓ | contract ✗ | terms_policy ✓ | terms_policy ✓ | terms_policy ✓ | terms_policy ✓ | terms_policy ✓ |
| doc_019 | Terms of Service for the Fieldwork Performanc... | 8 | terms_policy | terms_policy ✓ | terms_policy ✓ | proxy_statement ✗ | contract ✗ | terms_policy ✓ | terms_policy ✓ | terms_policy ✓ | terms_policy ✓ | terms_policy ✓ |
| doc_020 | Dobbs v. Jackson Women's Health Organization,... | 213 | court_opinion | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ |
| doc_021 | Trump v. Anderson, No. 23-719 (per curiam sli... | 20 | court_opinion | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ |
| doc_022 | O'Connor-Ratcliff v. Garnier, No. 22-324 (per... | 6 | court_opinion | court_opinion ✓ | court_opinion ✓ | earnings_release ✗ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | earnings_release ✗ |
| doc_023 | Department of Education v. Louisiana, No. 24A... | 13 | court_opinion | legislation ✗ | court_opinion ✓ | proxy_statement ✗ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | court_opinion ✓ | legislation ✗ |
| doc_024 | CHIPS and Science Act, Public Law 117-167 | 394 | legislation | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ |
| doc_025 | Fiscal Responsibility Act of 2023, Public Law... | 41 | legislation | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ |
| doc_026 | S. 623, Sunshine Protection Act of 2021 (as p... | 6 | legislation | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ | legislation ✓ |
| doc_027 | Attention Is All You Need | 15 | research_paper | research_paper ✓ | research_paper ✓ | investor_presentation ✗ | research_paper ✓ | research_paper ✓ | research_paper ✓ | research_paper ✓ | research_paper ✓ | research_paper ✓ |
| doc_028 | Deep Residual Learning for Image Recognition | 12 | research_paper | research_paper ✓ | research_paper ✓ | investor_presentation ✗ | research_paper ✓ | research_paper ✓ | research_paper ✓ | research_paper ✓ | research_paper ✓ | research_paper ✓ |
| doc_029 | Language Models are Few-Shot Learners | 75 | research_paper | research_paper ✓ | research_paper ✓ | earnings_release ✗ | research_paper ✓ | research_paper ✓ | research_paper ✓ | research_paper ✓ | research_paper ✓ | research_paper ✓ |
| doc_030 | On the Opportunities and Risks of Foundation ... | 214 | research_paper | research_paper ✓ | research_paper ✓ | sustainability_report ✗ | research_paper ✓ | research_paper ✓ | research_paper ✓ | research_paper ✓ | research_paper ✓ | research_paper ✓ |
| doc_031 | RFC 9110: HTTP Semantics | 194 | technical_standard | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ |
| doc_032 | RFC 9000: QUIC: A UDP-Based Multiplexed and S... | 151 | technical_standard | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ |
| doc_033 | RFC 9239: Updates to ECMAScript Media Types | 13 | technical_standard | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ |
| doc_034 | RFC 9457: Problem Details for HTTP APIs | 16 | technical_standard | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ | technical_standard ✓ |
| doc_035 | Google Cloud's AI Adoption Framework | 37 | white_paper | research_paper ✗ | research_paper ✗ | technical_standard ✗ | investor_presentation ✗ | white_paper ✓ | white_paper ✓ | research_paper ✗ | research_paper ✗ | research_paper ✗ |
| doc_036 | Future of Jobs Report 2023 (Insight Report) | 296 | white_paper | government_report ✗ | government_report ✗ | white_paper ✓ | government_report ✗ | white_paper ✓ | white_paper ✓ | government_report ✗ | government_report ✗ | government_report ✗ |
| doc_037 | Bitcoin: A Peer-to-Peer Electronic Cash System | 9 | white_paper | research_paper ✗ | research_paper ✗ | earnings_release ✗ | technical_standard ✗ | white_paper ✓ | research_paper ✗ | research_paper ✗ | user_manual ✗ | research_paper ✗ |
| doc_038 | Bash Reference Manual, Edition 5.3 | 214 | user_manual | user_manual ✓ | user_manual ✓ | user_manual ✓ | user_manual ✓ | user_manual ✓ | user_manual ✓ | user_manual ✓ | user_manual ✓ | user_manual ✓ |
| doc_039 | GNU Coreutils manual | 326 | user_manual | user_manual ✓ | user_manual ✓ | datasheet ✗ | technical_standard ✗ | user_manual ✓ | user_manual ✓ | user_manual ✓ | user_manual ✓ | user_manual ✓ |
| doc_040 | Getting Started with Raspberry Pi Pico-series | 36 | user_manual | user_manual ✓ | user_manual ✓ | datasheet ✗ | datasheet ✗ | user_manual ✓ | user_manual ✓ | user_manual ✓ | datasheet ✗ | user_manual ✓ |
| doc_041 | NE555 / SA555 / SE555 Precision Timers datasheet | 39 | datasheet | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ |
| doc_042 | RP2040 Datasheet | 642 | datasheet | user_manual ✗ | datasheet ✓ | datasheet ✓ | technical_standard ✗ | datasheet ✓ | datasheet ✓ | datasheet ✓ | user_manual ✗ | datasheet ✓ |
| doc_043 | 1N4148 Small Signal Fast Switching Diode data... | 4 | datasheet | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ |
| doc_044 | 1N4001-1N4007 1.0A Rectifier datasheet | 5 | datasheet | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | datasheet ✓ | terms_policy ✗ | datasheet ✓ | datasheet ✓ | datasheet ✓ |
| doc_045 | Monetary Policy Report, July 5, 2024 | 79 | government_report | government_report ✓ | government_report ✓ | government_report ✓ | annual_report ✗ | government_report ✓ | government_report ✓ | government_report ✓ | government_report ✓ | government_report ✓ |
| doc_046 | Financial Stability Report, April 2024 | 69 | government_report | government_report ✓ | government_report ✓ | government_report ✓ | quarterly_report ✗ | government_report ✓ | government_report ✓ | government_report ✓ | government_report ✓ | government_report ✓ |
| doc_047 | The 9/11 Commission Report | 585 | government_report | government_report ✓ | government_report ✓ | government_report ✓ | research_paper ✗ | government_report ✓ | government_report ✓ | government_report ✓ | government_report ✓ | government_report ✓ |
| doc_048 | Google Environmental Report 2024 | 86 | sustainability_report | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ |
| doc_049 | NVIDIA Sustainability Report, Fiscal Year 2024 | 41 | sustainability_report | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ | sustainability_report ✓ |
| doc_050 | Tesla Q2 2024 Update (investor slide deck) | 29 | investor_presentation | earnings_release ✗ | earnings_release ✗ | earnings_release ✗ | quarterly_report ✗ | quarterly_report ✗ | quarterly_report ✗ | earnings_release ✗ | annual_report ✗ | earnings_release ✗ |
| doc_051 | NVIDIA Investor Presentation Q2 FY25 (Septemb... | 27 | investor_presentation | investor_presentation ✓ | investor_presentation ✓ | investor_presentation ✓ | investor_presentation ✓ | investor_presentation ✓ | investor_presentation ✓ | investor_presentation ✓ | annual_report ✗ | investor_presentation ✓ |
<!-- doctype-benchmark:end -->

### Analysis

**Shared views, and what changed (review fix).** In P9 and the first P9b run, each system rendered its own view with its own token counter: the backend's tokenizer for local models, and the word-based `estimate_tokens` for Jev, which has none. The same document could therefore reach different systems as different text. All view-based systems now get one rendering per document and view config, made with `estimate_tokens` (see *Views* above), and all were re-run on all 51 documents.

- **How many views differed under the old rendering.** For the default view: 25 of 51 for UltraBERT, 18 for bge-small, 13 for the NLI model and 0 for Jev. For the P9b view configs with UltraBERT: 20 (`lead1024`, `default1024-lead400`), 11 (`toc1024`) and 51 (`lead512`); bge's `lead512` also differed on all 51. Every difference is in where the budget cut falls, so it changes how much contents, lead or keyphrases a view keeps, never which fields appear first.
- **Unchanged predictions.** `jev-view` (identical inputs, 104,384 tokens again), `bge-view` and `nli-view` give the same prediction on every document. bge and the NLI model cut at 512 tokens themselves, so the extra text never reached them.
- **Changed predictions.** UltraBERT reads up to 8,192 tokens, and its systems moved:

  | System | Change in correct documents | Fixed | Broken |
  |---|---|---|---|
  | `ultrabert-view` | 0 | doc_020 | doc_002 |
  | `ultrabert-hybrid` | +2 | doc_020, doc_042 | none |
  | `ultrabert-view-tuned` | 0 | doc_040 | doc_002 |
  | `ultrabert-loo` | +4 | 5 documents | 1 document |
  | `ultrabert-bge-loo` | -4 | none | doc_001, doc_006, doc_023, doc_036 |

  In the zero-shot grid, 7 of the 16 UltraBERT configs moved by 1-2 documents. Most notably, `lead512` dropped from 43 to 41 with the P9 labels.
- **What this implies.** Differences of 1-4 documents between UltraBERT configurations are within what the token-counting detail alone produces. The earlier P9b ensemble gain was not robust to it.

**Is the gap real?** On 51 documents one document is 2 accuracy points. Paired comparisons on the same documents use an exact McNemar test on the discordant documents and a paired bootstrap of the accuracy difference (1000 resamples, seed 0):

| Comparison | Only the first right | Only the second right | p | Difference (paired CI) |
|---|---|---|---|---|
| `jev-view` vs `ultrabert-hybrid` | 7 | 0 | 0.016 | +0.137 (+0.06 to +0.24) |
| `jev-view` vs `ultrabert-view-tuned` | 7 | 1 (doc_015) | 0.07 | +0.118 (+0.02 to +0.24) |
| `jev-view` vs `ultrabert-view` | 10 | 0 | 0.002 | |
| `jev-view` vs `ultrabert-bge-loo` | 11 | 0 | 0.001 | |
| `jev-view` vs `jev-chunks` | 3 | 1 | 0.63 | |
| `ultrabert-view-tuned` vs `ultrabert-view` | 4 | 0 | 0.125 | |
| `ultrabert-hybrid` vs `ultrabert-view` | 3 | 0 | 0.25 | |

- **Jev's lead is real** over every local system except the dev-tuned view. Against that view it is likely but not significant at 5%.
- **Jev view vs Jev chunks:** this corpus cannot separate them.
- **Local gains over `ultrabert-view`:** none is significant on its own.

**Where `jev-view` fails (2 of 51).**

- **doc_015, a sample NDA, predicted `user_manual` (0.58, with `contract` at 0.27).** The PDF opens with a page of "Instructions for non-disclosure agreements". The view's lead is that instruction text, and the title is the first numbered clause, because the parser turns the agreement's numbered paragraphs into headings. Jev answered for what the view shows.
- **doc_050, the Tesla Q2 2024 Update deck, predicted `quarterly_report` (0.47, with `earnings_release` at 0.44).** Every system misses this document. It is a text-heavy shareholder "update" laid out as slides. It parses to a single section, and its view has letter-spaced keyphrases (`s u m`, `g h t`), so the slide layout that defines `investor_presentation` is not visible in the text.

**Where `jev-chunks` fails (4 of 51).**

- **doc_044, a two-page diode datasheet, lost on a tie.** The second chunk is the vendor's legal "Important Notice". The per-chunk mean gave `terms_policy` and `datasheet` 0.50 each, and the tie went to the option listed first. Equal-weight averaging of raw chunks lets boilerplate vote; a document view does not have that failure mode.
- **doc_009 (earnings release, predicted `quarterly_report`) and doc_050.** Both are financial-type confusions.
- **doc_037 (the Bitcoin paper, gold `white_paper`, predicted `research_paper` at 0.63).** This is a borderline label that `jev-view` got right.

**Where the local zero-shot backends fail.** UltraBERT is good at the group: level-1 accuracy is 0.90 for `ultrabert-view`, 0.92 for `ultrabert-hybrid` and 0.94 for `ultrabert-view-tuned`. Most errors fall inside a group, between siblings whose definitions differ by who published the document or why rather than by vocabulary.

- **White papers.** `ultrabert-view`, `ultrabert-hybrid` and `ultrabert-view-tuned` each get 0 of 3. They predict `research_paper` or `government_report`, because "not peer-reviewed" and "issued by an agency" are not similarity features.
- **Contracts.**
  - `ultrabert-view` and `ultrabert-hybrid` get 0 of 2. The NIST CRADA (doc_016) reads like a `technical_standard`, and the NDA (doc_015) like `government_report` or `terms_policy`.
  - `ultrabert-view-tuned` gets the NDA right, narrowly: `contract` 0.17 against `terms_policy` 0.17.
- **Shareholder letters vs annual reports.**
  - doc_003, Amazon's annual report, starts with the 1997 shareholder letter, and DocumentView takes its title from the first non-letter-spaced level-1 heading ("1997 LETTER TO SHAREHOLDERS").
  - The Berkshire letters go to `investor_presentation` or `annual_report` in `ultrabert-view`.
  - doc_002 (Tesla's 10-K) goes to `proxy_statement` in both `ultrabert-view` (0.15 against 0.10 for `annual_report`) and `ultrabert-view-tuned` (0.18 against 0.15).
- **Other `ultrabert-view` errors.**
  - doc_023 (a per-curiam court opinion) is predicted as `legislation` (0.22 against 0.18 for `court_opinion`). Dobbs (doc_020) is now right (`court_opinion` 0.63); under the old rendering `ultrabert-view` predicted `legislation`.
  - The RP2040 datasheet (doc_042) is predicted as `user_manual` (0.32 against 0.26 for `datasheet`).
  - `ultrabert-hybrid` fixes doc_002, doc_023 and doc_042 and breaks nothing. The cost is about 5x the classify time (3.1 vs 0.62 s/doc in the contents-page re-run; 5.4 vs 0.75 in the shared-view run), because it embeds every leaf section.

`bge-view` (0.608) is limited by bge-small's 512-token window: the view plus the query prompt does not fit, so the end of the view is truncated. It also sends documents of several groups to `proxy_statement` and `earnings_release`. `nli-view` (0.569) is both the least accurate zero-shot system and by far the slowest (15.1 s/doc), because it runs 18 premise-hypothesis pairs of up to 512 tokens per document through DeBERTa on MPS.

**How close can local get (P9b).** Every configuration tried is in the *Configurations tried (P9b)* tables above. The numbers there are from the shared-view re-run, and nothing else was evaluated except the scratch prototypes noted below.

- **Zero-shot variants (16 UltraBERT configs and 4 bge configs, all scored on this corpus).** The grid is a 3x2x2 factorial (view x instruction x label text), plus two ablations of the lead-first view. The ablations were added after the first results.
  - *View.* Putting 400 words of lead first is what helps.
    - At 1,024 tokens (`lead1024`) it gains 1-4 documents over the default view in all four instruction/label combinations.
    - At the default 512 tokens (`lead512`), where it leaves little room for the contents, it gains 0-2.
    - Keeping the default field order and only raising the budget and lead (`default1024-lead400`) gains +1 or -1, so the order matters.
    - A contents-heavy view (`toc1024`) gains 1 document in each combination.
  - *Instruction.* "Identify the type of document from its title, contents and opening text" is worse than or equal to "What type of document is this?" in all 6 paired comparisons, by 0-3 documents.
  - *Label text.* Definition-first helps in 5 of the 8 UltraBERT pairs (+1 or +2), is neutral in 2 and hurts in 1 (-1 at `lead1024`). It hurts bge in both pairs. It is not a consistent win, and no gain exceeds 2 documents, which is within the rendering noise above.
  - *Selection bias.* The best config (43/51) was picked after seeing these 51 documents. Re-running the choice by leave-one-out (`zero-shot-config-loo`: the config most accurate on the other 50 documents predicts the held-out one) picks `lead1024` in all 51 folds and also gives 43/51. This estimate is mildly optimistic and is a lower bound on the optimism: it accounts for choosing among the 16 configs, but not for having designed the grid (the views, instructions and ablations) after seeing results on the same documents.
- **Few-shot, honest leave-one-out.** The features are the shared default view, as fixed in P9. Each held-out document is predicted from the other 50, with hyperparameters chosen by inner leave-one-out on those 50 only.
  - Logistic regression (C = 1000 in 50 of 51 folds): 0.706. Nearest centroid: 0.667. kNN (k = 1 in every fold): 0.765, the same as zero-shot `ultrabert-view`.
  - The package's own few-shot mechanism (`TaxonomyNode.examples`, where a leaf vector is the mean of the label and example vectors) gets 0.686, 4 documents *below* zero-shot. Whole-document embeddings of 1-3 examples pull each leaf towards the examples' publisher and layout rather than the type.
  - Mixing the zero-shot distribution with centroid evidence (`prior-centroid-cv`, weight 0.25 in 46 folds) gets 0.725, below zero-shot alone.
  - The supervised heads reach level-1 accuracy of 0.88-0.96. They fail at the same within-group distinctions as zero-shot, with less information than the written definitions.
- **Ensemble.** `ensemble-cv` mixes the UltraBERT-view and bge-view zero-shot distributions, with the weight chosen by inner LOO (0.5 in 48 folds, 0.25 in 3), and gets 0.745. A fixed weight of 0.5 would have given 0.804. Under the old rendering this system was 0.824 with 0.5 in every fold. It was promoted to the aggregate table (`ultrabert-bge-loo`) from that result. The choices of pairing (UltraBERT with bge), mixing method (a linear mix of distributions) and which LOO system to call "best honest" were all made after seeing results on these 51 documents, so it is mildly optimistic as a system. Re-running under shared views shows the gain was not robust. It stays in the table, not swapped for a post-hoc winner, so the record shows what was promoted.
- **Not reported as rows, disclosed here.** Prototype runs during the first P9b pass, on the dev-tuned `lead512` features with the old rendering, gave 0.765 for the UltraBERT+bge mix and 0.26-0.84 for the few-shot heads. None was above that view's own zero-shot score.
- **Remaining errors of `ultrabert-view-tuned` (8 of 51).**
  - All three white papers (doc_035-037).
  - The NIST CRADA (doc_016) as `technical_standard`.
  - Amazon's annual report (doc_003) and Tesla's 10-K (doc_002), as `shareholder_letter` and `proxy_statement`.
  - One Berkshire letter (doc_007) as `annual_report`.
  - doc_050, which every system misses.

  Jev gets 7 of these 8 right. They are distinctions of genre and intent ("not peer-reviewed", "issued to promote a view", "a letter bundled into a report") that cosine similarity to a definition does not capture.
- **What it would take to close the gap.**
  - *More labelled data.* At 1-3 documents per class no supervised head beats zero-shot, so the learning curve starts above this corpus. A trained head, or the package's fine-tuning, would plausibly need tens of documents per class, plus a separate held-out set to measure it.
  - *A larger or different local model.* The errors are judgements about genre, not missing text, so model capacity rather than the view looks like the bottleneck. Candidates are larger instruction-tuned embedding models, or a small local generative model asked the same Choice as Jev. Neither was tested here.
  - *A held-out corpus.* The P9b view should be confirmed on documents not used to choose it before it is trusted at 0.84.
- **Package defaults were not changed.** The lead-first view is the only zero-shot change worth at least 3 points, and it is not clearly general.
  - It costs bge 2 documents (0.608 to 0.569).
  - `jev-view`'s 0.961 was measured with the default view.
  - It was chosen on the evaluation set itself.
  - Its gain over the default (4 vs 0 discordant documents, p = 0.125) is not significant.

  `DocumentView(fields=("title", "lead", "toc", "keyphrases", "stats"), max_tokens=1024, lead_words=400)` is the documented option for UltraBERT users. The instruction and label-text variants did not help consistently.

**The P9 trained leave-one-out row was replaced.** P9's `ultrabert-loo` used `LogisticRegression(max_iter=1000)` at the default `C=1`. On unit-norm 768-d embeddings that regularises almost everything away: probabilities are near-uniform, and predictions fall back on the four classes with four documents (0.235; 0.255 under shared views). The row now uses the same features and splits, with C chosen per fold by inner leave-one-out over {1, 10, 100, 1000}, and gets 0.706. The fixed-C rows, including C=1, are kept as diagnostics in the P9b tables.

**Cost and speed.**

- **Parsing dominates the local pipeline.** The default `load_pdf` takes 1.47 s/doc on average (0.02-9.3 s, roughly proportional to pages) in the contents-page re-run; it was 1.60 s/doc in the shared-view run.
- **Classification times are noisy.** The machine was shared with other benchmark jobs (load average 5-8, peaking near 20 during the experiments), so these are upper bounds. In the contents-page re-run (load average 4-24 at the systems' starts), classification took 0.03 s/doc for `bge-view` and 0.62 s/doc for `ultrabert-view` (0.08 and 0.75 in the shared-view run; 0.18 s/doc for `ultrabert-view` in the quieter P9 run). The figures are batch throughput.
- **`jev-chunks` is the fastest end to end (0.81 s/doc)** because it skips structure: `page.get_text()` is 0.49 s/doc. It pays for that in tokens: 1,121,481 input tokens ($0.047, about $0.92 per 1,000 documents) against 104,372 for `jev-view`. That is about 2,050 per document, of which roughly 1,500 are the fixed instruction and 18 definitions, since the smallest request was 1,742 tokens ($0.0044, about $0.09 per 1,000 documents).
- **Both Jev costs are small in absolute terms.** The view's real advantages are 10.7x fewer tokens per document, a bounded request size whatever the page count, and robustness to boilerplate chunks. Its accuracy edge over chunks is not significant on this corpus.
- **Jev timings are network-bound.**
  - `jev-view` now runs in chunks of 8 documents with 8 concurrent requests, so each chunk's rows are saved before the next is paid for. Its classify time is batch throughput over 51 documents, 3.0 s in total (2.7 s in the shared-view run).
  - `jev-chunks` reports per-document latency.

**Honest comparison.** For document-type gating with written definitions, Jev is clearly better than any local system tested here: by 6-7 documents (12-14 points) over the best local systems and by up to 20 points over the plain `ultrabert-view`. docrooter's contribution in that pairing is the compact, budgeted DocumentView: Jev reads it as well as or better than eight raw chunks at a tenth of the tokens.

When no API is allowed:

- **The honest local choice is `ultrabert-hybrid`** (0.824, 0.92 group accuracy). It is a P9 configuration fixed before P9b, at about 5x the classify time of `ultrabert-view` (7x in the shared-view run).
- **`ultrabert-view-tuned`** is a single pass at 0.843, but it was tuned on these documents. Its leave-one-out estimate (0.843) is a lower bound on that optimism, not an unbiased figure, and it should be confirmed on held-out documents.
- **`ultrabert-bge-loo`** (0.745) was promoted from a result that did not survive the shared-view re-run. Its pairing, mixing method and "best honest" label were all chosen after seeing these 51 documents.
- **Routing:** `confidence` is available for sending low-confidence documents to review.

`bge-view` and `nli-view` are not competitive alone.

**Follow-ups this run exposed (not fixed here).**

- DocumentView keyphrases on letter-spaced text (doc_050).
- Title choice when a document starts with an embedded letter or an instruction page (doc_003, doc_015).
- A tie-break rule for exact ties in aggregated distributions.
- Whether `classify_documents` itself should render views with a backend-independent counter, so results do not depend on the tokenizer.

**Provenance.**

- **Runs.** The P9 runs, the first P9b run and the shared-view re-run were all on 2026-10-16. Parsed documents are cached in `benchmarks/.cache/docs/`, so every system within a run uses the same parses. The cache was deleted and rebuilt for the contents-page re-run (below), so its four systems saw the post-fix parses and the other rows the pre-fix ones. Raw rows, including the top-3 probabilities, per-document tokens and each row's `view_sha`, are in `benchmarks/results/2026-10-16/pdf.json`. The experiment rows, with per-document predictions and the hyperparameter chosen per fold, are under `experiments`.
- **The shared-view re-run.** The command was `python -m benchmarks.pdf_bench --systems jev-view,ultrabert-view,ultrabert-hybrid,bge-view,nli-view,ultrabert-view-tuned,ultrabert-loo,ultrabert-bge-loo --experiments`, with the harness at commit `196d4d0`. `jev-chunks` does not use DocumentView and was not re-run; its row is from P9 at `3bd85f2`.
- **Recorded commits.** Each system's commit is HEAD at its start. It is marked `+dirty` when tracked files under `src/` or `benchmarks/` (results excluded) had uncommitted changes. Other agents committed during the run, hence `aad5b3f` and `6d0a6eb` for the later systems.
  - The `+dirty` on the first five systems of the shared-view run (now only `nli-view`'s row still carries it) was another agent's uncommitted change to `render_budgeted` in `src/docrooter/views.py`, later committed as `8cc9f10`. It cuts each view line to 4 x `max_tokens` words before the token counter sees it, so a whole section is never tokenised. No other file under `src/` or `benchmarks/` was dirty.
  - All 255 shared view texts (5 view configs x 51 documents) were re-rendered with HEAD's `views.py` afterwards and are byte-identical.
- **Contents-page re-run (after the v0.2 dot-leader fix, `e72fca6`).** The fix changes the body text, and for doc_001 the headings, of some PDFs. So `jev-view`, `ultrabert-view`, `ultrabert-hybrid` and `bge-view` were re-run on all 51 documents at `53090c0`, after deleting the cached parses: `python -m benchmarks.pdf_bench --systems jev-view,ultrabert-view,ultrabert-hybrid,bge-view --docs docs/benchmarks-pdf.md`.
  - The default view text changed for 7 documents (doc_001, doc_013, doc_014, doc_031, doc_032, doc_036, doc_045).
  - No leaf or level-1 accuracy changed. The only prediction that changed is `bge-view` on doc_001 (gold `annual_report`): `shareholder_letter` became `proxy_statement`, still wrong. That moves bge's macro P from 0.576 to 0.584 and macro F1 from 0.538 to 0.541.
  - `jev-view` used 104,372 input tokens instead of 104,384, still $0.0044.
  - Parse and classify seconds moved with machine load.
  - **Not re-run:** `nli-view`, `ultrabert-view-tuned`, `ultrabert-loo`, `ultrabert-bge-loo`, the P9b experiment tables and the *Analysis* comparisons that use them. They are still on the pre-fix parses, and `pdf.json` keeps the pre-fix view hashes for their view configs. `jev-chunks` does not read parsed text.
- **Environment line.** It is from the re-run (MPS).

## Section normalisation

**Headline.** These are scores on 145 scored sections from 6 real PDFs. The labels are agent-assigned silver labels (see below), and the same set was used to choose the defaults. Two package changes in P10b improve the local backends:

- heading synonyms, added as `examples` on every node of the 5 built-in section taxonomies;
- a headings-only default view, `path + title`, used when the chosen taxonomy has examples.

If no node of the chosen taxonomy (for example, a custom one) has `examples`, `normalize_sections` keeps the previous `title + lead` view at 96 tokens, because without synonyms `path + title` was worse for UltraBERT (0.262 vs 0.345). An explicit `view=` always wins.

39 scored sections have a heading that equals one of the synonyms (see *Synonym overlap* below), so each cell gives *all* (out of 145) and *non-overlap* (out of 106):

| System | P10 | P10b | Change |
| --- | --- | --- | --- |
| UltraBERT (default backend) | 50/145 (0.345); 18/106 (0.170) | 71/145 (0.490); 34/106 (0.321) | +14.5; +15.1 points |
| bge-small | 33/145 (0.228); 17/106 (0.160) | 57/145 (0.393); 20/106 (0.189) | +16.6; +2.8 points |
| Jev | 102/145 (0.703); 65/106 | 103/145 (0.710); 65/106 | +0.7 points |
| Auto doc-type path | 0.303; 0.132 | 0.455; 0.292 | +15.2; +16.0 points |

- **For bge, most of the gain needs a heading the synonyms already list.** Only 3 of its 24 gained sections are on non-overlap rows. This does not hold for UltraBERT: 16 of its 21 gained sections are on non-overlap rows.
- **Jev's change is within noise.** Re-running the P10 configuration today gave 101/145 (0.697) against P10's 102.
- **Jev costs more.** $0.0085 instead of $0.0064, because the synonyms are sent with every candidate.

**How the default was chosen.** Read this before trusting the numbers above.

- **The silver set served as the dev set.**
  - The labels, the P10 runs and the P10 per-section tables for doc_027 and doc_002 were committed before the synonyms. Those tables show some silver titles.
  - Git proves only that the synonyms (`da38a07`) came before any run that used them, not that they were written blind to those tables.
- **The synonyms were changed once afterwards, at review.** `b9cf84b` removed six entries: "Model Architecture", "Training", "Chapter 2", "Item 1B. Unresolved Staff Comments", "Item 4. Mine Safety Disclosures" and "Item 16. Form 10-K Summary". These are removals only. After them, UltraBERT's overall score fell from 0.600 to 0.490, mostly on doc_027 (19 → 12 of 22). That document's level-2 sections sit under "3 Model Architecture" and "5 Training", which the path view puts into every one of them.
- **Eight configurations were run.** Each of 4 views was run with and without synonyms on both local backends; Jev was run on 3 of them. All results are in the table below.
- **`path+title+syn` is the best of the 8 for UltraBERT on both row sets.** For bge it is one section behind `title+syn` on all rows and level with it on non-overlap rows. Against synonyms with the old view, it gains:
  - UltraBERT: +5.5 points on all rows, +8.5 on non-overlap rows;
  - bge: +9.0 on all rows, but −0.9 (one section) on non-overlap rows.

  So for bge the view change helps only on headings the synonyms already list.
- **Synonyms alone, with the old view, give +9.0 (UltraBERT) and +7.6 (bge).**

**Synonym overlap.** The non-overlap rule compares each heading with its gold taxonomy's examples. The comparison ignores case and punctuation and drops one leading number, such as "3.1", "Item 1A." or "Note 2.". Under this rule, 39 scored sections (28 distinct headings) match an example. The review count of 47 rows was made before `b9cf84b` and may have used a different rule.

**Collapse.** The P10 collapse onto a few labels is smaller but still present.

- **Top predictions.** UltraBERT's most frequent prediction changes from `introduction` (39) to `experiments` (18). bge's stays `experiments`, but drops from 54 to 27.
- **Remaining errors.**
  - UltraBERT labels annual-report `other` sections as `business_overview` (7).
  - bge labels `method` as `experiments` (11) and `results` as `background` (9).
  - Research appendices go to `experiments`.

**Label sensitivity.** 124 of the scored rows have no "could be X" note. On those rows the scores are UltraBERT 66 (0.532), bge 53 (0.427) and Jev 90 (0.726). All numbers here are single runs on 6 documents.

**Speed.** Each system now runs one untimed warm-up that encodes every section taxonomy it can use, so *s/section* no longer includes the one-off encoding of the examples. Under the same rule, the P10 runs warmed only the research-paper taxonomy. This run's *s/section*: UltraBERT 0.019, bge 0.004 and Jev 0.028.

**View rendering.** `render_budgeted` now cuts each line to 4 × `max_tokens` words before the token counter sees it, so a whole section is never tokenised. P10 had logged 8521- and 5243-token sequences.

- **Same output.** Views were identical before and after for all 204 sections of the 6 documents, with both local tokenizers and 4 views. A seeded differential unit test guards this.
- **Remaining warning.** Garbled PDF text can still exceed bge's 512-token limit; the longest seen was 1084 tokens for 384 words. When that happens, transformers logs one harmless warning.

**Not re-run after the dot-leader fix.** The normalisation numbers were measured before the v0.2 contents-page fix (`e72fca6`). That fix changes doc_001's parse: page 13's earnings table is no longer a contents page, so "The Scorecard in 2023" is now a section. The rows of that table and of page 149's list of operating companies are also no longer printed-contents entries, and those entries had boosted the borderline headings "Pilot Travel Centers (PTC)", "Insurance—Investment Income" and "Railroad" (pages 35-68) over the score threshold; without the boost they are no longer detected. Against today's parse, 5 silver rows of doc_001 no longer match a section (3 scored, 2 marked as junk) and 1 section has no silver row. The other 5 documents match exactly as before. The section benchmark should be re-run, and doc_001's silver rows re-checked, in v0.3.

<!-- normalize-benchmark:begin -->
Silver set: `benchmarks/corpus/section_gold.jsonl`, 163 level-1/2 sections of 6 documents (18 marked as parser junk, i.e. not a real section, and excluded from accuracy). **These are agent-assigned silver labels, not human gold:** an agent (Claude Opus 5.5) read each heading and the first ~300 characters of its body and chose the best node of the doc type's section taxonomy from the node definitions, before any normalisation run (labelled 2026-10-16T14:41:44-04:00, documents loaded at `0dd70bc`). Ambiguous cases carry a `note` in the file. Expect a few points of label noise.

Run 2026-10-16 at `ae6b037`, device `mps` (MPS probe matmul took 0.21 s). Each system calls `normalize_sections(doc, doc_type=<gold type>, backend=...)` once per document with the package defaults: Flat, view path + title at 96 tokens, with the taxonomies' heading synonyms. *Macro F1* averages over the gold classes (per document over that document's classes; overall over all classes, qualified by taxonomy). *s/section* is the summed wall time of the per-document calls (view rendering and label encoding included) divided by the sections labelled, junk included. Model load and the label encoding of every section taxonomy the run can use (one untimed warm-up per doc type) are excluded (*Load s*). Jev cost is input tokens x $0.042/M as reported by the API.

**Synonym overlap.** 39 of the 145 scored sections have a heading equal to a heading-synonym example of their taxonomy (28 distinct headings; compared case- and punctuation-insensitively after dropping one leading number such as "3.1", "Item 1A." or "Note 2."). For those, the synonyms can hand the model the answer, so every accuracy below is also given on the other 106 (*non-overlap*), as *all; non-overlap*.

### Before / after

*Baseline (P10)* is the original run (2026-10-16 at `6d3ee70`: view title + lead at 96 tokens, without the taxonomies' heading synonyms); *after* is this run. Accuracy cells are *all; non-overlap*.

| System | Baseline (P10) accuracy | After accuracy | Change (all) | Change (non-overlap) | Baseline macro F1 | After macro F1 |
| --- | --- | --- | --- | --- | --- | --- |
| `ultrabert` | 0.345 (50/145); 0.170 (18/106) | 0.490 (71/145); 0.321 (34/106) | +14.5 pts | +15.1 pts | 0.456 | 0.540 |
| `bge-small` | 0.228 (33/145); 0.160 (17/106) | 0.393 (57/145); 0.189 (20/106) | +16.6 pts | +2.8 pts | 0.308 | 0.532 |
| `jev` | 0.703 (102/145); 0.613 (65/106) | 0.710 (103/145); 0.613 (65/106) | +0.7 pts | +0.0 pts | 0.755 | 0.734 |
| `auto_doc_type` | 0.303 (44/145); 0.132 (14/106) | 0.455 (66/145); 0.292 (31/106) | +15.2 pts | +16.0 pts | 0.344 | 0.465 |

### Configurations tried (the silver set used as a dev set)

Every configuration that was run, all on the same scored sections, so the chosen default (`path+title+syn`) is selected on the evaluation data and its margin over the others is optimistic. Accuracy cells are *all; non-overlap*. *Top prediction* is the most frequent predicted label over the scored sections and its count (a collapse onto one label shows up here).

| Config | View | Synonyms | `ultrabert` accuracy | `bge-small` accuracy | `jev` accuracy | `ultrabert` top prediction | `bge-small` top prediction | `jev` top prediction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `title+lead@96` | title + lead @ 96 | no | 0.345 (50/145); 0.170 (18/106) | 0.228 (33/145); 0.160 (17/106) | 0.697 (101/145); 0.604 (64/106) | `introduction` (39) | `experiments` (54) | `experiments` (19) |
| `title` | title @ 96 | no | 0.262 (38/145); 0.104 (11/106) | 0.310 (45/145); 0.160 (17/106) | - | `introduction` (62) | `experiments` (33) | - |
| `path+title` | path + title @ 96 | no | 0.262 (38/145); 0.113 (12/106) | 0.352 (51/145); 0.189 (20/106) | - | `introduction` (54) | `experiments` (32) | - |
| `title+lead@48` | title + lead @ 48 | no | 0.324 (47/145); 0.160 (17/106) | 0.262 (38/145); 0.151 (16/106) | - | `introduction` (42) | `experiments` (48) | - |
| `title+lead@96+syn` | title + lead @ 96 | yes | 0.434 (63/145); 0.236 (25/106) | 0.303 (44/145); 0.198 (21/106) | 0.676 (98/145); 0.566 (60/106) | `experiments` (32) | `experiments` (60) | `experiments` (20) |
| `title+syn` | title @ 96 | yes | 0.393 (57/145); 0.189 (20/106) | 0.400 (58/145); 0.189 (20/106) | - | `experiments` (20) | `experiments` (36) | - |
| `path+title+syn` (default) | path + title @ 96 | yes | 0.490 (71/145); 0.321 (34/106) | 0.393 (57/145); 0.189 (20/106) | 0.710 (103/145); 0.613 (65/106) | `experiments` (18) | `experiments` (27) | `other` (25) |
| `title+lead@48+syn` | title + lead @ 48 | yes | 0.441 (64/145); 0.255 (27/106) | 0.310 (45/145); 0.189 (20/106) | - | `experiments` (28) | `experiments` (54) | - |

### Aggregate

| System | Backend | Accuracy | Correct | Macro F1 | s/section | Load s | Jev tokens | Jev cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ultrabert` | `embedding:dmis-lab/UltraBERT-Embed-150M` | 0.490 | 71/145 | 0.540 | 0.0185 | 14.6 | - | - |
| `bge-small` | `embedding:BAAI/bge-small-en-v1.5` | 0.393 | 57/145 | 0.532 | 0.0037 | 5.9 | - | - |
| `jev` | `jev:jev-1.13.0` | 0.710 | 103/145 | 0.734 | 0.0277 | 1.1 | 202,823 | $0.0085 |

### Per document (accuracy / macro F1)

| Document | Type | Sections | Junk | Unmatched | `ultrabert` | `bge-small` | `jev` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| doc_027 Attention Is All You Need | `research.research_paper` | 23 | 1 | 0 | 0.545 / 0.604 | 0.409 / 0.504 | 0.818 / 0.724 |
| doc_028 Deep Residual Learning for Image Recognition | `research.research_paper` | 16 | 2 | 0 | 0.500 / 0.571 | 0.429 / 0.550 | 0.714 / 0.683 |
| doc_029 Language Models are Few-Shot Learners | `research.research_paper` | 38 | 2 | 0 | 0.694 / 0.745 | 0.361 / 0.618 | 0.722 / 0.711 |
| doc_001 Berkshire Hathaway Inc. 2023 Annual Report | `financial.annual_report` | 29 | 5 | 0 | 0.250 / 0.268 | 0.333 / 0.304 | 0.583 / 0.471 |
| doc_002 Tesla, Inc. Annual Report on Form 10-K for fiscal year 2023 | `financial.annual_report` | 45 | 7 | 0 | 0.421 / 0.614 | 0.368 / 0.536 | 0.632 / 0.726 |
| doc_016 NIST Cooperative Research and Development Agreement (CRADA) template, 2018 | `legal.contract` | 12 | 1 | 0 | 0.455 / 0.352 | 0.636 / 0.553 | 1.000 / 1.000 |

### Auto doc type (`doc_type=None`)

One run of `normalize_sections(doc)` with the default backend (`embedding:dmis-lab/UltraBERT-Embed-150M`), so `classify_document` picks the doc type and hence the section taxonomy. The doc type matched the gold type on **5/6** documents; section accuracy 0.455 (66/145), macro F1 0.465 (a wrong doc type usually means a different taxonomy, so its sections all count as wrong).

| Document | Gold type | Auto type | Confidence | Taxonomy | Accuracy |
| --- | --- | --- | --- | --- | --- |
| doc_027 | `research.research_paper` | ✓ `research.research_paper` | 0.627 | `section_types/research_paper` | 0.545 |
| doc_028 | `research.research_paper` | ✓ `research.research_paper` | 0.662 | `section_types/research_paper` | 0.500 |
| doc_029 | `research.research_paper` | ✓ `research.research_paper` | 0.372 | `section_types/research_paper` | 0.694 |
| doc_001 | `financial.annual_report` | ✓ `financial.annual_report` | 0.196 | `section_types/annual_report` | 0.250 |
| doc_002 | `financial.annual_report` | ✓ `financial.annual_report` | 0.121 | `section_types/annual_report` | 0.421 |
| doc_016 | `legal.contract` | ✗ `research.technical_standard` | 0.500 | `section_types/generic` | 0.000 |

### Per entry: doc_027 Attention Is All You Need

| Page | Level | Section | Gold | `ultrabert` | `bge-small` | `jev` |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | Attention Is All You Need | *(junk)* | `introduction` | `background` | `introduction` |
| 1 | 1 | Abstract | `abstract` | ✓ `abstract` | ✓ `abstract` | ✓ `abstract` |
| 2 | 1 | 1 Introduction | `introduction` | ✓ `introduction` | ✓ `introduction` | ✓ `introduction` |
| 2 | 1 | 2 Background | `related_work` | ✗ `background` | ✗ `background` | ✗ `background` |
| 2 | 1 | 3 Model Architecture | `method` | ✓ `method` | ✗ `experiments` | ✓ `method` |
| 3 | 2 | 3.1 Encoder and Decoder Stacks | `method` | ✗ `experiments` | ✗ `experiments` | ✓ `method` |
| 3 | 2 | 3.2 Attention | `method` | ✗ `background` | ✗ `experiments` | ✓ `method` |
| 5 | 2 | 3.3 Position-wise Feed-Forward Networks | `method` | ✗ `background` | ✗ `experiments` | ✓ `method` |
| 5 | 2 | 3.4 Embeddings and Softmax | `method` | ✗ `experiments` | ✗ `experiments` | ✓ `method` |
| 6 | 2 | 3.5 Positional Encoding | `method` | ✗ `background` | ✗ `experiments` | ✓ `method` |
| 6 | 1 | 4 Why Self-Attention | `method` | ✗ `introduction` | ✗ `introduction` | ✓ `method` |
| 7 | 1 | 5 Training | `experiments` | ✗ `method` | ✗ `background` | ✓ `experiments` |
| 7 | 2 | 5.1 Training Data and Batching | `experiments` | ✓ `experiments` | ✓ `experiments` | ✓ `experiments` |
| 7 | 2 | 5.2 Hardware and Schedule | `experiments` | ✓ `experiments` | ✓ `experiments` | ✓ `experiments` |
| 7 | 2 | 5.3 Optimizer | `experiments` | ✓ `experiments` | ✓ `experiments` | ✗ `method` |
| 7 | 2 | 5.4 Regularization | `experiments` | ✗ `background` | ✓ `experiments` | ✗ `method` |
| 8 | 1 | 6 Results | `results` | ✓ `results` | ✓ `results` | ✓ `results` |
| 8 | 2 | 6.1 Machine Translation | `results` | ✓ `results` | ✗ `background` | ✓ `results` |
| 8 | 2 | 6.2 Model Variations | `results` | ✓ `results` | ✗ `experiments` | ✓ `results` |
| 9 | 2 | 6.3 English Constituency Parsing | `results` | ✓ `results` | ✗ `background` | ✓ `results` |
| 10 | 1 | 7 Conclusion | `conclusion` | ✓ `conclusion` | ✓ `conclusion` | ✓ `conclusion` |
| 10 | 1 | References | `references` | ✓ `references` | ✓ `references` | ✓ `references` |
| 13 | 1 | Attention Visualizations | `appendix` | ✗ `experiments` | ✗ `experiments` | ✗ `results` |

### Per entry: doc_002 Tesla, Inc. Annual Report on Form 10-K for fiscal year 2023

| Page | Level | Section | Gold | `ultrabert` | `bge-small` | `jev` |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | UNITED STATES SECURITIES AND EXCHANGE COMMISSION | *(junk)* | `notes_to_financial_statements` | `management_discussion` | `other` |
| 1 | 1 | Washington, D.C. 20549 | *(junk)* | `properties` | `other` | `other` |
| 1 | 1 | FORM 10-K | `other` | ✗ `business_overview` | ✗ `notes_to_financial_statements` | ✓ `other` |
| 1 | 1 | Tesla, Inc. | `other` | ✗ `business_overview` | ✗ `business_overview` | ✓ `other` |
| 3 | 2 | PART I. | *(junk)* | `business_overview` | `business_overview` | `other` |
| 3 | 2 | PART II. | *(junk)* | `management_discussion` | `management_discussion` | `other` |
| 3 | 2 | PART III. | *(junk)* | `business_overview` | `management_discussion` | `other` |
| 3 | 2 | PART IV. | *(junk)* | `business_overview` | `management_discussion` | `other` |
| 15 | 2 | ITEM 1A. RISK FACTORS | `risk_factors` | ✓ `risk_factors` | ✓ `risk_factors` | ✓ `risk_factors` |
| 29 | 2 | ITEM 1B. UNRESOLVED STAFF COMMENTS | `other` | ✗ `notes_to_financial_statements` | ✗ `management_discussion` | ✓ `other` |
| 31 | 2 | ITEM 2. PROPERTIES | `properties` | ✓ `properties` | ✓ `properties` | ✓ `properties` |
| 31 | 2 | ITEM 3. LEGAL PROCEEDINGS | `legal_proceedings` | ✓ `legal_proceedings` | ✓ `legal_proceedings` | ✓ `legal_proceedings` |
| 31 | 2 | ITEM 4. MINE SAFETY DISCLOSURES | `other` | ✗ `management_discussion` | ✗ `management_discussion` | ✓ `other` |
| 32 | 2 | ITEM 5. MARKET FOR REGISTRANT’S COMMON EQUITY, RELATED STOCKHOLDER MAT | `market_for_equity` | ✓ `market_for_equity` | ✓ `market_for_equity` | ✓ `market_for_equity` |
| 34 | 2 | ITEM 7. MANAGEMENT’S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AN | `management_discussion` | ✓ `management_discussion` | ✓ `management_discussion` | ✓ `management_discussion` |
| 48 | 2 | Report of Independent Registered Public Accounting Firm | `financial_statements` | ✓ `financial_statements` | ✓ `financial_statements` | ✓ `financial_statements` |
| 50 | 2 | Consolidated Balance Sheets | `financial_statements` | ✓ `financial_statements` | ✓ `financial_statements` | ✓ `financial_statements` |
| 52 | 2 | Consolidated Statements of Comprehensive Income (in millions) | `financial_statements` | ✓ `financial_statements` | ✓ `financial_statements` | ✓ `financial_statements` |
| 53 | 2 | Consolidated Statements of Redeemable Noncontrolling Interests and Equ | `financial_statements` | ✓ `financial_statements` | ✗ `market_for_equity` | ✓ `financial_statements` |
| 56 | 2 | Notes to Consolidated Financial Statements | `notes_to_financial_statements` | ✓ `notes_to_financial_statements` | ✓ `notes_to_financial_statements` | ✓ `notes_to_financial_statements` |
| 77 | 2 | 2024 Notes | `notes_to_financial_statements` | ✓ `notes_to_financial_statements` | ✓ `notes_to_financial_statements` | ✓ `notes_to_financial_statements` |
| 84 | 2 | 2018 CEO Performance Award | `notes_to_financial_statements` | ✗ `shareholder_letter` | ✗ `shareholder_letter` | ✗ `executive_compensation` |
| 95 | 2 | ITEM 9. CHANGES IN AND DISAGREEMENTS WITH ACCOUNTANTS ON ACCOUNTING AN | `other` | ✗ `controls_and_procedures` | ✗ `management_discussion` | ✓ `other` |
| 95 | 2 | ITEM 9A. CONTROLS AND PROCEDURES | `controls_and_procedures` | ✓ `controls_and_procedures` | ✓ `controls_and_procedures` | ✓ `controls_and_procedures` |
| 96 | 2 | ITEM 9B. OTHER INFORMATION | `other` | ✗ `business_overview` | ✗ `business_overview` | ✓ `other` |
| 96 | 2 | ITEM 9C. DISCLOSURE REGARDING FOREIGN JURISDICTIONS THAT PREVENT INSPE | `other` | ✗ `controls_and_procedures` | ✗ `controls_and_procedures` | ✓ `other` |
| 97 | 2 | ITEM 10. DIRECTORS, EXECUTIVE OFFICERS AND CORPORATE GOVERNANCE | `corporate_governance` | ✓ `corporate_governance` | ✓ `corporate_governance` | ✓ `corporate_governance` |
| 97 | 2 | ITEM 11. EXECUTIVE COMPENSATION | `executive_compensation` | ✓ `executive_compensation` | ✓ `executive_compensation` | ✓ `executive_compensation` |
| 97 | 2 | ITEM 12. SECURITY OWNERSHIP OF CERTAIN BENEFICIAL OWNERS AND MANAGEMEN | `corporate_governance` | ✓ `corporate_governance` | ✗ `market_for_equity` | ✗ `market_for_equity` |
| 97 | 2 | ITEM 13. CERTAIN RELATIONSHIPS AND RELATED TRANSACTIONS AND DIRECTOR I | `corporate_governance` | ✗ `notes_to_financial_statements` | ✗ `market_for_equity` | ✓ `corporate_governance` |
| 97 | 2 | ITEM 14. PRINCIPAL ACCOUNTANT FEES AND SERVICES | `other` | ✗ `financial_statements` | ✗ `notes_to_financial_statements` | ✓ `other` |
| 98 | 2 | ITEM 15. EXHIBITS AND FINANCIAL STATEMENT SCHEDULES | `exhibits` | ✓ `exhibits` | ✓ `exhibits` | ✓ `exhibits` |
| 113 | 2 | ITEM 16. SUMMARY | `exhibits` | ✗ `business_overview` | ✗ `management_discussion` | ✗ `other` |
| 124 | 2 | SECTION 1350 CERTIFICATIONS | `exhibits` | ✗ `notes_to_financial_statements` | ✗ `corporate_governance` | ✗ `controls_and_procedures` |
| 125 | 1 | CLAWBACK POLICY | `exhibits` | ✗ `sustainability` | ✗ `market_for_equity` | ✗ `corporate_governance` |
| 125 | 1 | 1. Definitions | `exhibits` | ✗ `business_overview` | ✗ `business_overview` | ✗ `other` |
| 127 | 1 | 2. Recoupment of Erroneously Awarded Compensation | `exhibits` | ✗ `executive_compensation` | ✗ `executive_compensation` | ✗ `executive_compensation` |
| 127 | 1 | 3. Means of Repayment | `exhibits` | ✗ `notes_to_financial_statements` | ✗ `notes_to_financial_statements` | ✗ `other` |
| 128 | 1 | 4. No Indemnification | `exhibits` | ✗ `notes_to_financial_statements` | ✗ `other` | ✗ `other` |
| 128 | 1 | 5. Miscellaneous | `exhibits` | ✗ `notes_to_financial_statements` | ✗ `other` | ✗ `other` |
| 129 | 1 | 6. Amendment and Termination | `exhibits` | ✗ `management_discussion` | ✗ `management_discussion` | ✗ `other` |
| 129 | 1 | 7. Successors | `exhibits` | ✗ `corporate_governance` | ✗ `management_discussion` | ✗ `management_discussion` |
| 129 | 1 | 8. Agreement to Arbitrate; Governing Law; Venue | `exhibits` | ✗ `legal_proceedings` | ✗ `legal_proceedings` | ✗ `other` |
| 130 | 1 | CLAWBACK POLICY ACKNOWLEDGMENT, CONSENT AND AGREEMENT | `exhibits` | ✗ `notes_to_financial_statements` | ✗ `notes_to_financial_statements` | ✗ `corporate_governance` |
| 130 | 1 | Signed: Print Name: Date: | *(junk)* | `exhibits` | `exhibits` | `exhibits` |
<!-- normalize-benchmark:end -->

## Dev vs held-out

One frozen run at commit `72f6135` on the 40-PDF held-out corpus (`benchmarks/heldout/`), which
shares no documents or publishers with the dev corpus and was never used for tuning. Full
method, per-document tables and analysis are in [benchmarks-heldout.md](benchmarks-heldout.md).
Dev numbers come from `benchmarks/results/2026-10-16/{toc,pdf,normalize}.json`, read when the
held-out run finished. The dev leave-one-out systems and the normalisation rows had been re-run
just before. The dev leave-one-out systems become "trained on all 51 dev documents, tested on
held-out". One held-out document is 2.5 points.

<!-- heldout-compare:begin -->
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
<!-- heldout-compare:end -->

**Discussion.**

- **Document type generalises.**
  - Jev stays at 0.925-0.975. `ho_018` is image-only and counts as a miss for everyone.
  - The zero-shot local views do *better* than on dev.
  - The honest transfer test is the one that disappoints: the logistic head trained on dev
    (0.675) and the dev-weighted UltraBERT+bge ensemble (0.800) both score **below zero-shot
    `ultrabert-view` (0.825)** on new publishers. With 2-4 training documents per class, the
    trained head learns publisher cues.
- **Section normalisation holds.** UltraBERT and bge-small improve on held-out data even though
  the defaults were chosen on the dev silver set. Jev drops 3 points, and the auto doc-type path
  drops 7: one contract is typed as terms and policy at confidence 0.21.
- **TOC extraction is the real drop.** Default-engine d<=2 F1 falls by about 12 points:
  - About half of the drop comes from bookmark sets that are not tables of contents
    (design-tool page labels, "Slide N:" prefixes, a scan).
  - The rest is genuine: bold body-size headings in an earnings release, a bilingual statute,
    and a split cover title shifting heading levels.
  - `pdfplumber` silently returns 3 headings for the 816-page ECMA-262 specification, which is
    what drives its all-depth collapse and pulls `both` down with it.
