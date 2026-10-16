# PDF benchmarks

## PDF TOC extraction

<!-- toc-benchmark:begin -->
Gold: bookmarks of the 24 corpus PDFs with at least 5 bookmarks (levels shifted so the minimum is 1; items without a target page dropped). Extraction runs with `use_bookmarks=False`. A prediction matches a bookmark at normalised-title ratio >= 0.85 and pages within +/-1, one-to-one and greedy by ratio. The depth <= 2 columns keep only gold and predicted headings of level <= 2, so precision is measured at that depth too. Micro pools the counts; macro is the mean of per-document F1. Level accuracy is measured on matched pairs. Bookmark sets often omit lower-level headings, so precision is pessimistic and recall / F1 at depth <= 2 are the headline numbers.

Environment: Python 3.12.13, arm64, 8 CPUs, pymupdf 1.28.2, pdfplumber 0.11.10, pymupdf4llm 1.28.2. `os.getloadavg()` was 19.4, 14.6, 17.0 (1/5/15 min) when the run started. The machine was shared with other jobs, so seconds per document are noisy upper bounds; each engine ran one untimed warm-up document first. Seconds are wall-clock for one `load_pdf` call.

### Aggregate by engine (depth <= 2 headline, all depths for reference)

| Engine | Docs | P (d<=2) | R (d<=2) | F1 micro (d<=2) | F1 macro (d<=2) | Level acc (d<=2) | P (all) | R (all) | F1 micro (all) | F1 macro (all) | Level acc (all) | s/doc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pymupdf | 24 | 0.369 | 0.678 | 0.478 | 0.464 | 0.802 | 0.444 | 0.898 | 0.594 | 0.595 | 0.548 | 1.69 |
| pdfplumber | 24 | 0.367 | 0.654 | 0.470 | 0.449 | 0.805 | 0.462 | 0.885 | 0.607 | 0.600 | 0.545 | 14.44 |
| both | 24 | 0.372 | 0.665 | 0.477 | 0.468 | 0.812 | 0.475 | 0.886 | 0.618 | 0.612 | 0.552 | 14.14 |
| pymupdf4llm* | 8 | 0.667 | 0.564 | 0.611 | 0.517 | 0.974 | 0.615 | 0.692 | 0.651 | 0.550 | 0.912 | 15.21 |

\* `pymupdf4llm`: limited to the 8 smallest bookmarked documents (by page count); see the matched-subset table below.

### Same 8 documents for every engine (the `pymupdf4llm` subset)

| Engine | P (d<=2) | R (d<=2) | F1 micro (d<=2) | F1 macro (d<=2) | Level acc (d<=2) | F1 micro (all) | s/doc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pymupdf | 0.676 | 0.351 | 0.463 | 0.383 | 0.789 | 0.728 | 0.34 |
| pdfplumber | 0.696 | 0.351 | 0.467 | 0.389 | 0.789 | 0.709 | 2.33 |
| both | 0.710 | 0.351 | 0.470 | 0.395 | 0.789 | 0.731 | 2.56 |
| pymupdf4llm | 0.667 | 0.564 | 0.611 | 0.517 | 0.974 | 0.651 | 15.21 |

### Per document: `pymupdf` (the default engine)

| Doc | Title | Pages | Gold d<=2 | Pred d<=2 | P | R | F1 | Level acc | F1 (all) | s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| doc_001 | Berkshire Hathaway Inc. 2023 Annual Report | 152 | 9 | 6 | 0.17 | 0.11 | 0.13 | 0.00 | 0.17 | 3.58 |
| doc_003 | Amazon.com, Inc. 2023 Annual Report | 92 | 30 | 4 | 0.00 | 0.00 | 0.00 | n/a | 0.66 | 1.74 |
| doc_004 | Berkshire Hathaway Inc. Form 10-Q for the qua... | 56 | 22 | 14 | 0.71 | 0.45 | 0.56 | 0.00 | 0.53 | 0.87 |
| doc_005 | Berkshire Hathaway Inc. Form 10-Q for the qua... | 59 | 22 | 14 | 0.71 | 0.45 | 0.56 | 0.00 | 0.53 | 0.97 |
| doc_009 | Amazon.com Announces Second Quarter Results (... | 12 | 18 | 0 | 0.00 | 0.00 | 0.00 | n/a | 0.00 | 0.17 |
| doc_010 | Amazon.com Announces Third Quarter Results (O... | 12 | 18 | 0 | 0.00 | 0.00 | 0.00 | n/a | 0.00 | 0.18 |
| doc_027 | Attention Is All You Need | 15 | 19 | 23 | 0.83 | 1.00 | 0.90 | 1.00 | 0.92 | 0.24 |
| doc_029 | Language Models are Few-Shot Learners | 75 | 32 | 41 | 0.78 | 1.00 | 0.88 | 0.75 | 0.70 | 1.20 |
| doc_030 | On the Opportunities and Risks of Foundation ... | 214 | 39 | 43 | 0.84 | 0.92 | 0.88 | 1.00 | 0.84 | 3.92 |
| doc_031 | RFC 9110: HTTP Semantics | 194 | 30 | 149 | 0.19 | 0.93 | 0.31 | 0.32 | 0.85 | 2.75 |
| doc_032 | RFC 9000: QUIC: A UDP-Based Multiplexed and S... | 151 | 36 | 146 | 0.24 | 0.97 | 0.38 | 0.34 | 0.96 | 1.70 |
| doc_033 | RFC 9239: Updates to ECMAScript Media Types | 13 | 17 | 24 | 0.67 | 0.94 | 0.78 | 0.56 | 0.96 | 0.15 |
| doc_034 | RFC 9457: Problem Details for HTTP APIs | 16 | 19 | 25 | 0.72 | 0.95 | 0.82 | 0.61 | 0.95 | 0.15 |
| doc_038 | Bash Reference Manual, Edition 5.3 | 214 | 70 | 73 | 0.84 | 0.87 | 0.85 | 0.98 | 0.91 | 3.05 |
| doc_039 | GNU Coreutils manual | 326 | 179 | 180 | 0.96 | 0.96 | 0.96 | 0.99 | 0.97 | 4.46 |
| doc_040 | Getting Started with Raspberry Pi Pico-series | 36 | 55 | 26 | 0.58 | 0.27 | 0.37 | 1.00 | 0.88 | 0.43 |
| doc_041 | NE555 / SA555 / SE555 Precision Timers datasheet | 39 | 30 | 45 | 0.44 | 0.67 | 0.53 | 1.00 | 0.60 | 0.47 |
| doc_042 | RP2040 Datasheet | 642 | 96 | 742 | 0.13 | 0.98 | 0.22 | 0.78 | 0.31 | 6.58 |
| doc_045 | Monetary Policy Report, July 5, 2024 | 79 | 23 | 56 | 0.25 | 0.61 | 0.35 | 1.00 | 0.41 | 0.94 |
| doc_046 | Financial Stability Report, April 2024 | 69 | 13 | 14 | 0.86 | 0.92 | 0.89 | 1.00 | 0.80 | 0.63 |
| doc_048 | Google Environmental Report 2024 | 86 | 32 | 7 | 0.71 | 0.16 | 0.26 | 0.00 | 0.31 | 4.15 |
| doc_049 | NVIDIA Sustainability Report, Fiscal Year 2024 | 41 | 30 | 8 | 0.75 | 0.20 | 0.32 | 0.00 | 0.48 | 0.96 |
| doc_050 | Tesla Q2 2024 Update (investor slide deck) | 29 | 29 | 2 | 0.50 | 0.03 | 0.06 | 1.00 | 0.06 | 0.95 |
| doc_051 | NVIDIA Investor Presentation Q2 FY25 (Septemb... | 27 | 27 | 5 | 0.40 | 0.07 | 0.12 | 0.50 | 0.49 | 0.44 |
<!-- toc-benchmark:end -->

### Reading the numbers

The default engine (`pymupdf`, what `engine="auto"` resolves to) scores 0.478 micro F1 at depth <= 2, below the 0.70 target. Recall is decent (0.678 at depth <= 2, 0.898 over all depths) but precision is 0.369, and that is partly a property of the gold rather than the extractor: bookmark sets are often shallow, flat or synthetic (see doc_004/005, where every bookmark is level 1, and the slide decks below). Level accuracy on matched pairs is 0.80 at depth <= 2 but only 0.55 over all depths. `pdfplumber` and `both` are within noise of `pymupdf` on quality and about 8x slower per document, with the caveat that the machine was heavily loaded. `pymupdf4llm` (legacy path, runs OCR) is clearly the best on its 8-document subset (0.611 vs 0.463 micro F1 for `pymupdf` on the same documents, level accuracy 0.97), but it is 45x slower per document, and a different order of magnitude on larger files, which is why it was limited to the 8 smallest bookmarked documents. Its two scoring failures are the same ones the native engine has (doc_009/010 and the slide decks).

### The three worst documents (depth <= 2 F1 = 0.00, default engine)

Three documents tie at F1 0.00 at depth <= 2: doc_003, doc_009 and doc_010. doc_009 and doc_010 are two editions of the same Amazon earnings-release layout, so they share one cause.

1. **doc_003, Amazon 2023 annual report (F1 0.00 at depth <= 2, 0.66 over all depths).** The extractor finds almost every `Item N.` heading (31 predictions, all-depth recall is high), but assigns them level 5. The 10-K body uses bold text at body size (9.7 pt), so the typographic ranking puts the cover page (18 pt "ANNUAL REPORT"), the 1997 shareholder letter (12 pt) and the SEC form front page (15.5 pt and 10.7 pt lines) above them, and each of those distinct styles consumes a level. Four spurious front-matter lines end up at levels 2-3 and every real section sits at level 5. Proposal: when assigning levels from typography, ignore styles used by fewer than about 3 headings (or fewer than 2% of headings) when ranking, or rank styles by how many headings and pages they cover instead of by absolute size. Dropping page 13 sentence-like lines ("1997 was indeed an incredible year...", 0.55) would also help; a heading-length or terminal-punctuation penalty in the title score is the likely fix.
2. **doc_009 and doc_010, Amazon quarterly earnings releases (0 predictions).** All text, including the section labels, is 10 pt, the same as the body size, so the `size >= 1.15 x body` pre-filter removes every candidate (candidate scores peak at 0.48: bold and isolated but no size credit; the best are `AMAZON.COM ANNOUNCES SECOND QUARTER RESULTS` and `Financial Guidance`). The gold is also hard: the bookmarks (`Page 1 Financials (QTD)`, `CEO Quote`, `Highlights`) are labels of regions of a designed page, not text printed as headings. Proposal: allow a second candidate path for bold, isolated, short lines at body size when the document has no larger-than-body candidates at all, with a threshold lowered to about 0.45; expect only a partial win here because most gold titles do not appear in the text.

### Other patterns seen while inspecting

- **Slide decks (doc_050, doc_051; F1 0.06 and 0.12).** Gold titles are `Slide Number 2`, `Slide 2` and the like, which never occur in the page text, so recall is capped near zero by construction; for NVIDIA the extractor does find titles (`Data Center`, `Gaming`) but also stat captions such as `154% Y/Y`. A slide-deck mode (one heading per page from the largest line) would be needed, and the gold would still not match.
- **Section-name running footers (doc_042, 742 predictions for 96 gold; doc_045).** In the RP2040 datasheet the current section title (`1.4. Pinout Reference`, 8 pt, y = 0.947 of the page height) is printed in the footer band of every page of that section. It is in the bottom 8% but repeats on only a few pages (11-14), far below the 30% recurrence the exclusion rule requires, so each copy is scored as a heading. doc_045 has the same effect with a changing page number (`6 Part 1: Recent Economic...`), so the exact-text test also misses it. Proposal: in the edge bands, additionally exclude a line whose normalised text (page number stripped) equals the previous or next page's edge line, or equals a heading kept on an adjacent page.
- **Printed TOC pages in RFCs (doc_031/032).** Roughly 30 predictions come from pages 3-11, where each line of the printed contents (`3.7. Intermediaries`) scores as a heading. The extracted lines carry no trailing page numbers, so the `^(.*?)[\s.·…_-]{2,}(\d{1,4})$` detector never fires (`printed_toc` is 0). Proposal: also treat a page as a TOC page when at least 50% of its lines start with a section number and the numbers run in increasing order.
- **Gold levels do not always follow numbering.** doc_031 puts `1. Introduction` at level 2 under the document title, and the extractor puts it at level 1. Some level mismatches are therefore gold artefacts.
