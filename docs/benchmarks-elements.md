# Element benchmarks

## Element kinds on DocLayNet (spec §10.1)

**Data.** [DocLayNet](https://github.com/DS4SD/DocLayNet) (Pfitzmann et al., KDD 2022), licence [CDLA-Permissive-1.0](https://cdla.io/permissive-1-0/): 80,863 human-annotated pages from six document categories, with 11 layout classes. The benchmark reads the Hugging Face re-packaging `docling-project/DocLayNet-v1.2` at revision `0daf93102e2efce76c3e11a274a5e0d0969391d3`. It holds each page's born-digital single-page PDF in a binary column, beside the gold boxes and the PDF text cells inside each box. `benchmarks/doclaynet.py` pins every parquet shard by SHA-256 (the LFS object id at that revision). It downloads every shard of a split, because the shards are sorted by category, checks each one, samples, writes the sampled PDFs to the git-ignored `benchmarks/.cache/doclaynet/`, and then deletes the shards.

**Provenance: these results predate the XY-cut table fix.** Every DocLayNet number in this section (§10.1: the test and train tables, the tuning log and the disclosure) was produced at or before commit `213979e`, with `ee2a829` checked against it, and so before `c09b9f7` ("XY-cut keeps row-aligned tables together"). That fix changed element building: over the 51 dev PDFs, pymupdf's element count went from 90,908 to 89,410 (the lines are the same, 316,617; 89,516 after the later reading-order rewrite, `1a02fcb`). The DocLayNet scores may therefore move, and they were not re-scored, because the test split has already been scored twice (see the disclosure below) and a third scoring would no longer be a held-out measurement. Treat them as describing the code at `213979e`. The dev-corpus checks and the reading-order, OCR and speed sections below say which commit they were made at.

**Sample.** 400 test-split pages and 400 train-split pages, stratified over the six categories with an equal share each (67 pages for each of the first four categories in name order, 66 for patents and scientific articles), drawn with seed `20261018`. DocLayNet's train and test splits share no documents. Thresholds were tuned on the train pages only. The test pages have been scored twice, both times after tuning. The second scoring was made for correctness, not for score: see the disclosure below.

**Scoring.**
- Element kinds map to DocLayNet classes by the spec §4 table: `page_number` scores as Page-footer and `contents` as Text.
- On each page, predicted element boxes are matched one to one to gold boxes of the same class, greedily by IoU, at IoU >= 0.5.
- Per class, precision, recall and F1 pool the counts over all pages. Macro F1 is the mean over the classes present.
- Text coverage is free of IoU. For each class, it is the share of characters in the gold boxes' PDF text cells that land in a predicted element of that class (the element the cell overlaps most). Element boxes hug their lines, while DocLayNet boxes are padded regions, and one gold Text region may hold several predicted paragraphs. Coverage still counts all of those characters as correct.
- Gold boxes are COCO `[x, y, w, h]` in 1025 x 1025 page-image pixels, because DocLayNet stretches each page to a square. Each axis is scaled to PDF points by its own page side.

**Systems.**
- `heuristic-pymupdf` and `heuristic-pdfplumber`: `load_pdf(elements=True)` with the element rules of spec §5.3.
- `heron`: pymupdf with `layout="heron"` (spec §6), using one ONNX session for the whole run, under the one-model-process guard.

```
.venv/bin/python -I benchmarks/doclaynet.py prepare --split train   # and --split test
PYTHONPATH=src .venv/bin/python -m benchmarks.elements_bench --split train
PYTHONPATH=src .venv/bin/python -m benchmarks.elements_bench --split test \
    --systems heuristic-pymupdf,heuristic-pdfplumber,heron --docs docs/benchmarks-elements.md
```

### Test results (final: shipped constants, `elements-doclaynet.json`)

These are the numbers for the constants that ship (commit `213979e`): the tuned values below, with the spec figure thresholds and the pdfplumber CropBox fix. The later pdfplumber geometry fix (`ee2a829`) changes no test page (see the disclosure).

<!-- elements-benchmark:begin -->
Split: **test**, 400 pages (financial_reports 67, government_tenders 67, laws_and_regulations 67, manuals 67, patents 66, scientific_articles 66); sample digest `cdda52423f79`; commit `213979e`; 2026-10-18. Matching: one-to-one, greedy by IoU >= 0.5, within a class. Pages without a text layer (scans; no OCR engine runs in this benchmark): heuristic-pymupdf 9, heuristic-pdfplumber 9, heron 9; load errors: heuristic-pymupdf 0, heuristic-pdfplumber 0, heron 0.

### Per class (F1; precision / recall below)

| Class | Gold | heuristic-pymupdf | heuristic-pdfplumber | heron |
| --- | --- | --- | --- | --- |
| Caption | 121 | 0.453 (0.76 / 0.32) | 0.435 (0.76 / 0.31) | 0.851 (0.88 / 0.83) |
| Footnote | 50 | 0.237 (0.35 / 0.18) | 0.165 (0.20 / 0.14) | 0.971 (0.94 / 1.00) |
| Formula | 153 | 0.065 (0.06 / 0.08) | 0.016 (0.01 / 0.03) | 0.867 (0.81 / 0.93) |
| List-item | 1075 | 0.460 (0.51 / 0.42) | 0.397 (0.48 / 0.34) | 0.900 (0.93 / 0.87) |
| Page-footer | 330 | 0.503 (0.51 / 0.49) | 0.477 (0.46 / 0.50) | 0.715 (0.59 / 0.91) |
| Page-header | 314 | 0.000 (0.00 / 0.00) | 0.000 (0.00 / 0.00) | 0.664 (0.85 / 0.54) |
| Picture | 314 | 0.152 (0.13 / 0.19) | 0.165 (0.14 / 0.20) | 0.718 (0.59 / 0.93) |
| Section-header | 582 | 0.184 (0.53 / 0.11) | 0.152 (0.58 / 0.09) | 0.680 (0.82 / 0.58) |
| Table | 158 | 0.290 (0.61 / 0.19) | 0.288 (0.60 / 0.19) | 0.801 (0.89 / 0.73) |
| Text | 2205 | 0.407 (0.29 / 0.71) | 0.308 (0.21 / 0.60) | 0.836 (0.82 / 0.85) |
| Title | 26 | 0.080 (0.05 / 0.27) | 0.045 (0.03 / 0.15) | 0.092 (0.05 / 0.38) |
| **Macro F1** | | **0.257** | **0.223** | **0.736** |
| s/page | | 0.029 | 0.081 | 0.609 |

### Per category (macro F1)

| Category | Pages | heuristic-pymupdf | heuristic-pdfplumber | heron |
| --- | --- | --- | --- | --- |
| financial_reports | 67 | 0.217 | 0.206 | 0.640 |
| government_tenders | 67 | 0.330 | 0.312 | 0.688 |
| laws_and_regulations | 67 | 0.267 | 0.263 | 0.600 |
| manuals | 67 | 0.217 | 0.219 | 0.726 |
| patents | 66 | 0.095 | 0.062 | 0.621 |
| scientific_articles | 66 | 0.237 | 0.167 | 0.684 |

### Text coverage (share of gold characters in an element of the right kind)

| Class | Gold chars | heuristic-pymupdf | heuristic-pdfplumber | heron |
| --- | --- | --- | --- | --- |
| Caption | 11301 | 0.374 | 0.238 | 0.927 |
| Footnote | 7689 | 0.283 | 0.271 | 0.999 |
| Formula | 13875 | 0.159 | 0.115 | 0.938 |
| List-item | 123099 | 0.410 | 0.323 | 0.906 |
| Page-footer | 5009 | 0.119 | 0.115 | 0.872 |
| Page-header | 6673 | 0.000 | 0.000 | 0.732 |
| Picture | 17398 | 0.178 | 0.189 | 0.911 |
| Section-header | 20119 | 0.077 | 0.062 | 0.574 |
| Table | 104853 | 0.157 | 0.157 | 0.665 |
| Text | 478266 | 0.927 | 0.928 | 0.949 |
| Title | 940 | 0.220 | 0.162 | 0.352 |

### Spec §10.1 target: heuristic F1 >= 0.8 for Text, Table, Picture on financial_reports and scientific_articles

| System | Category | Text | Table | Picture | Met |
| --- | --- | --- | --- | --- | --- |
| heuristic-pymupdf | financial_reports | 0.371 | 0.318 | 0.351 | no |
| heuristic-pymupdf | scientific_articles | 0.402 | 0.000 | 0.423 | no |
| heuristic-pdfplumber | financial_reports | 0.308 | 0.308 | 0.464 | no |
| heuristic-pdfplumber | scientific_articles | 0.228 | 0.000 | 0.464 | no |
<!-- elements-benchmark:end -->

### Disclosure: the test split has been seen twice

The test pages were first scored once, after tuning, at commit `d951b11` plus the uncommitted tuned constants (`elements-doclaynet-first.json`, reproduced below). Review then found two problems, and the controller ruled on both:

1. **The tuned figure thresholds broke the E5a content ruling.** `FIGURE_MIN_AREA` 0.005 and `FIGURE_MAX_TEXT_COVERAGE` 0.4 turned shaded statement-table rows and banner rows into figures on the dev corpus. Both were reverted to the spec / E5a values (pass 5 of the tuning log).
2. **heuristic-pdfplumber boxes were in the wrong space on some pages.** The gold and pymupdf use the visible page, CropBox ∩ MediaBox, but pdfplumber element boxes were in its MediaBox frame. The fix was made at the source (`pdf/objects.py`) in two steps:
   - **`213979e`** handled a CropBox inside a MediaBox whose origin is 0. That fixed the 5 affected test pages (all financial reports) and 9 of the 13 affected train pages. It did not handle a MediaBox with a non-zero origin (4 train pages, 18–33 pt off), and it gave a wrong page size when the CropBox is larger than the MediaBox (dev doc_001, 126 pages).
   - **`ee2a829`** handles every case. It takes the visible box as CropBox ∩ MediaBox, uses it for the page size, and shifts every pdfplumber box by its top-left corner, derived per rotation. The layout-region text-table search (`pdf/text_tables.py`) gets the same transform.
   - **Effect on the published test numbers.** Under `ee2a829`, **0 of the 400 test pages** change their pdfplumber page box or element boxes. This was found by comparing objects only, without scoring. The published heuristic-pdfplumber test numbers therefore stand for the current code. On train, 5 pages change, and train was re-scored.

After those fixes the published first scoring no longer described the shipped code, so the test split was scored a second time with the final constants. This second scoring was for correctness, not for score: no threshold was changed after either test scoring, and the test pages were never used to choose a value. Both results are published.

**What changed between the two scorings:**
- **heuristic-pymupdf:** macro F1 0.263 → 0.257. Picture TP went from 89 to 59 and FP from 443 to 405, because the small images that `FIGURE_MIN_AREA` 0.005 accepted were 30 true and 38 false pictures on test. Caption and Text moved by one box each.
- **heuristic-pdfplumber:** macro F1 0.226 → 0.223. The Picture loss is the same; Text gains 43 TP from the CropBox fix and the figure revert.
- **heron:** macro F1 0.7345 → 0.7359, Picture FP 230 → 206. The heron system runs on pymupdf, so the CropBox fix does not touch it. The difference comes from `FIGURE_MIN_AREA`: images outside every heron region still go through the heuristic figure rule.

#### First test scoring (superseded: `FIGURE_MIN_AREA` 0.005, `FIGURE_MAX_TEXT_COVERAGE` 0.4, pdfplumber before the CropBox fix)

Split: **test**, 400 pages (financial_reports 67, government_tenders 67, laws_and_regulations 67, manuals 67, patents 66, scientific_articles 66); sample digest `cdda52423f79`; commit `d951b11+dirty`; 2026-10-18. Matching: one-to-one, greedy by IoU >= 0.5, within a class. Pages without a text layer (scans; no OCR engine runs in this benchmark): heuristic-pymupdf 9, heuristic-pdfplumber 9, heron 9; load errors: heuristic-pymupdf 0, heuristic-pdfplumber 0, heron 0.

#### Per class (F1; precision / recall below)

| Class | Gold | heuristic-pymupdf | heuristic-pdfplumber | heron |
| --- | --- | --- | --- | --- |
| Caption | 121 | 0.460 (0.75 / 0.33) | 0.442 (0.75 / 0.31) | 0.851 (0.88 / 0.83) |
| Footnote | 50 | 0.237 (0.35 / 0.18) | 0.165 (0.20 / 0.14) | 0.971 (0.94 / 1.00) |
| Formula | 153 | 0.065 (0.06 / 0.08) | 0.016 (0.01 / 0.03) | 0.867 (0.81 / 0.93) |
| List-item | 1075 | 0.460 (0.51 / 0.42) | 0.391 (0.48 / 0.33) | 0.900 (0.93 / 0.87) |
| Page-footer | 330 | 0.503 (0.51 / 0.49) | 0.468 (0.45 / 0.49) | 0.715 (0.59 / 0.91) |
| Page-header | 314 | 0.000 (0.00 / 0.00) | 0.000 (0.00 / 0.00) | 0.664 (0.85 / 0.54) |
| Picture | 314 | 0.210 (0.17 / 0.28) | 0.226 (0.18 / 0.31) | 0.702 (0.56 / 0.94) |
| Section-header | 582 | 0.184 (0.53 / 0.11) | 0.149 (0.57 / 0.09) | 0.680 (0.82 / 0.58) |
| Table | 158 | 0.290 (0.61 / 0.19) | 0.288 (0.60 / 0.19) | 0.801 (0.89 / 0.73) |
| Text | 2205 | 0.407 (0.29 / 0.71) | 0.299 (0.20 / 0.58) | 0.836 (0.82 / 0.85) |
| Title | 26 | 0.080 (0.05 / 0.27) | 0.045 (0.03 / 0.15) | 0.092 (0.05 / 0.38) |
| **Macro F1** | | **0.263** | **0.226** | **0.734** |
| s/page | | 0.029 | 0.080 | 0.587 |

#### Per category (macro F1)

| Category | Pages | heuristic-pymupdf | heuristic-pdfplumber | heron |
| --- | --- | --- | --- | --- |
| financial_reports | 67 | 0.224 | 0.189 | 0.638 |
| government_tenders | 67 | 0.359 | 0.342 | 0.683 |
| laws_and_regulations | 67 | 0.267 | 0.262 | 0.600 |
| manuals | 67 | 0.214 | 0.216 | 0.722 |
| patents | 66 | 0.095 | 0.062 | 0.621 |
| scientific_articles | 66 | 0.243 | 0.173 | 0.687 |

#### Text coverage (share of gold characters in an element of the right kind)

| Class | Gold chars | heuristic-pymupdf | heuristic-pdfplumber | heron |
| --- | --- | --- | --- | --- |
| Caption | 11301 | 0.387 | 0.240 | 0.927 |
| Footnote | 7689 | 0.283 | 0.271 | 0.999 |
| Formula | 13875 | 0.159 | 0.115 | 0.937 |
| List-item | 123099 | 0.410 | 0.322 | 0.906 |
| Page-footer | 5009 | 0.119 | 0.113 | 0.872 |
| Page-header | 6673 | 0.000 | 0.000 | 0.732 |
| Picture | 17398 | 0.178 | 0.189 | 0.911 |
| Section-header | 20119 | 0.077 | 0.059 | 0.574 |
| Table | 104853 | 0.157 | 0.157 | 0.665 |
| Text | 478266 | 0.925 | 0.925 | 0.949 |
| Title | 940 | 0.220 | 0.162 | 0.352 |

#### Spec §10.1 target: heuristic F1 >= 0.8 for Text, Table, Picture on financial_reports and scientific_articles

| System | Category | Text | Table | Picture | Met |
| --- | --- | --- | --- | --- | --- |
| heuristic-pymupdf | financial_reports | 0.371 | 0.318 | 0.419 | no |
| heuristic-pymupdf | scientific_articles | 0.402 | 0.000 | 0.468 | no |
| heuristic-pdfplumber | financial_reports | 0.266 | 0.308 | 0.515 | no |
| heuristic-pdfplumber | scientific_articles | 0.228 | 0.000 | 0.521 | no |

### Reading the numbers (final scoring)

- **The heuristic rules miss the spec §10.1 target by a wide margin.**
  - On the test pages, `heuristic-pymupdf` scores F1 0.37 / 0.32 / 0.35 for Text / Table / Picture on financial reports, and 0.40 / 0.00 / 0.42 on scientific articles, against a target of 0.8. The target is reported as missed.
  - `heron` reaches a macro F1 of 0.736 against the heuristics' 0.257, with 0.80 on Table and 0.84 on Text over all categories.
  - For tables, figures and region-level text, the layout model is the engine to use.
- **Text is found, but cut differently.**
  - The IoU-free text coverage of Text is 0.93 for both heuristic engines (heron: 0.95). Nearly every character of a gold Text region lands in a paragraph, so the low box F1 (0.41) comes mostly from region boundaries.
  - DocLayNet's padded regions often span what the rules split into several paragraphs, or join lines that the rules keep apart.
  - Text precision (0.29) also takes in the lines that should have been Page-header, Section-header, Table or Formula.
- **Single-page documents disable two signals.**
  - Running headers and footers are detected by recurrence across pages (spec §5.3 rule 1), which a one-page PDF cannot show, so heuristic Page-header F1 is 0. Page-footer scores 0.50 only because bare page numbers are still recognised.
  - Every page is page 1, so the first large heading on any page can become the document `title`. Title precision is 0.05, and heron's 0.09 has the same cause.
  - Both numbers understate what the rules do in real multi-page documents.
- **Tables are mostly unruled.** The heuristic engine detects only ruled tables (spec §5.3 rule 3). Table recall is 0.19 and text coverage is 0.16. Scientific-article tables, booktabs style with few rulings, score 0.
- **Pictures:** heuristic Picture precision is 0.55 on train but 0.13 on test (405 false pictures).
  - Small images are not the main cause: the first scoring, with smaller figures allowed, had 443.
  - Most of these are image and vector regions that DocLayNet does not box as Picture, such as decorations, backgrounds, logos and shaded panels. On the test sample they are much more frequent than on the train sample.
  - This is a property of the heuristic figure rule, not of a tuned value.
- **Scans:** 9 test pages and 10 train pages have no text layer. No OCR engine runs in this benchmark (OCR is measured in spec §10.4), so these pages yield a page-sized figure or nothing, as in `load_pdf(elements=True)` without OCR.
- **Macro F1 convention:** the mean over the classes present in gold *or* predictions on the scored pages. A class with predictions but no gold therefore counts as F1 0, where COCO-style mAP would skip it. This matters only per category.
- **Train vs test:** with the final constants, heuristic-pymupdf scores 0.350 macro F1 on train and 0.257 on test. The train number is optimistic, because it is the sample the thresholds were chosen on.

### Train results (`elements-doclaynet-train.json`, final constants)

| System | Train macro F1, spec starting values | Train macro F1, first tuned set | Train macro F1, final (shipped) |
| --- | --- | --- | --- |
| heuristic-pymupdf | 0.3305 | 0.3535 | 0.3496 |
| heuristic-pdfplumber | 0.2965 | 0.3131 | 0.3119 (with the visible-box fix, `ee2a829`) |

The starting values were re-scored with `--set` (see the log below). pdfplumber was not tuned on, but it gains too, because the constants are shared.

## Tuning log (E11a, DocLayNet train pages only)

**Method.** Coordinate descent on `heuristic-pymupdf`, the default engine, over the 400 train pages. Each candidate changes one constant of `docrooter.pdf.elements` from the current best set, through `--set NAME=VALUE`. A change was kept only if train macro F1 rose by at least 0.001, the resolution reported. Ties went to the value closest to the current one. One evaluation takes about 13 s. The test pages were not opened during tuning.

**Starting point:** spec §5.3 values plus the E5a guards, train macro F1 **0.3305**.

### Pass 1 (every tunable constant)

| Constant (start) | Tried: train macro F1 | Result | Macro F1 after |
| --- | --- | --- | --- |
| PARAGRAPH_GAP_RATIO (1.5) | 1.0: 0.3179, 1.25: 0.3288, 1.75: 0.3328, 2.0: 0.3318, 2.5: 0.3320, 3.0: 0.3317 | keep 1.75 | 0.3328 |
| PARAGRAPH_SIZE_DELTA (0.5) | 0.25: 0.3326, 1.0: 0.3301, 1.5: 0.3301, 2.0: 0.3299 | no gain | 0.3328 |
| FIRST_LINE_INDENT (8) | 4: 0.3330, 6: 0.3329, 12: 0.3322, 16: 0.3317 | no gain (+0.0002 < 0.001) | 0.3328 |
| FIGURE_MIN_AREA (0.02) | 0.005: 0.3363, 0.01: 0.3358, 0.015: 0.3332, 0.03: 0.3302, 0.05: 0.3284 | keep 0.005, **reverted in pass 5 (ruling)** | 0.3363 |
| FIGURE_MAX_TEXT_DENSITY (0.5) | 0.25: 0.3339, 1.0: 0.3359, 2.0: 0.3358, 4.0: 0.3357 | no gain | 0.3363 |
| FIGURE_MAX_TEXT_COVERAGE (0.3) | 0.15: 0.3368, 0.2: 0.3367, 0.4: 0.3385, 0.5: 0.3385 | keep (0.5 in the pass; 0.4 ties, and pass 2 restarted from 0.4, the smaller change), **reverted to 0.3 in pass 5 (ruling)** | 0.3385 |
| FIGURE_BACKGROUND_PAGE_SHARE (0.5) | 0.35: 0.3372, 0.4: 0.3369, 0.6: 0.3385, 0.7: 0.3385, 0.9: 0.3385 | no gain | 0.3385 |
| FIGURE_PROSE_MIN_WORDS (5) | 3: 0.3354, 4: 0.3387, 6: 0.3392, 8: 0.3401 | keep 8, **later reverted (ruling below)** | 0.3401 |
| FIGURE_PROSE_LINE_SHARE (0.5) | 0.3: 0.3401, 0.4: 0.3401, 0.6: 0.3410, 0.7: 0.3406 | no gain (+0.0009) | 0.3401 |
| FIGURE_PROSE_MIN_SENTENCES (2) | 1, 3, 4: 0.3401 | no effect | 0.3401 |
| FIGURE_PROSE_SENTENCE_SHARE (0.25) | 0.1, 0.4, 0.5: 0.3401 | no effect | 0.3401 |
| FIGURE_PROSE_RUN_LINES (3) | 2: 0.3396, 4: 0.3401, 5: 0.3406 | no gain | 0.3401 |
| FIGURE_GRID_MIN_ROWS (3) | 2, 4, 5: 0.3401 | no effect | 0.3401 |
| FIGURE_GRID_MIN_COLUMNS (2) | 3: 0.3401 | no effect | 0.3401 |
| FIGURE_GRID_ALIGN_TOLERANCE (3) | 1.5, 5, 8: 0.3401 | no effect | 0.3401 |
| FIGURE_GRID_MIN_LABELLED_ROWS (2) | 1, 3, 4: 0.3401 | no effect | 0.3401 |
| FIGURE_GRID_LINE_SHARE (0.7) | 0.5, 0.6, 0.8, 0.9: 0.3401 | no effect | 0.3401 |
| FIGURE_CODE_LINE_SHARE (0.5) | 0.3, 0.7: 0.3401 | no effect | 0.3401 |
| FIGURE_LIST_MIN_ITEMS (2) | 1: 0.3402, 3: 0.3401, 4: 0.3401 | no gain | 0.3401 |
| CAPTION_MAX_DISTANCE_LINES (1.5) | 1.0: 0.3355, 2.0: 0.3428, 3.0: 0.3447, 4.0: 0.3452 | keep 4.0 | 0.3452 |
| CAPTION_MAX_LINES (4) | 2: 0.3357, 3: 0.3434, 6: 0.3463, 8: 0.3463 | keep (8 in the pass; 6 ties, and pass 2 restarted from 6) | 0.3463 |
| FOOTNOTE_BOTTOM_SHARE (0.25) | 0.15: 0.3317, 0.2: 0.3404, 0.35: 0.3499, 0.5: 0.3459 | keep 0.35 | 0.3499 |
| FOOTNOTE_MAX_SIZE_RATIO (0.9) | 0.8: 0.3483, 0.85: 0.3518, 0.95: 0.3500, 1.0: 0.3236 | keep 0.85 | 0.3518 |
| FORMULA_MIN_SYMBOL_SHARE (0.3) | 0.15: 0.3509, 0.2: 0.3511, 0.25: 0.3513, 0.4: 0.3503, 0.5: 0.3505 | no gain | 0.3518 |
| LIST_INDENT_STEP (8) | 12: 0.3518 | no effect (it sets `list_depth` only, never a box) | 0.3518 |

### Pass 2 (around the pass-1 set, with the tie choices of 0.4 and 6; start 0.3518)

| Constant (current) | Tried: train macro F1 | Result | Macro F1 after |
| --- | --- | --- | --- |
| PARAGRAPH_GAP_RATIO (1.75) | 1.5: 0.3486, 1.6: 0.3487, 1.9: 0.3508, 2.0: 0.3507 | no gain | 0.3518 |
| PARAGRAPH_SIZE_DELTA (0.5) | 0.25: 0.3516, 0.75: 0.3516 | no gain | 0.3518 |
| FIRST_LINE_INDENT (8) | 4: 0.3519 | no gain | 0.3518 |
| FIGURE_MIN_AREA (0.005) | 0.0025: 0.3520, 0.0075: 0.3495, 0.01: 0.3509 | no gain | 0.3518 |
| FIGURE_MAX_TEXT_COVERAGE (0.4) | 0.3: 0.3515, 0.5: 0.3518, 0.6: 0.3517, 0.7: 0.3517 | no gain | 0.3518 |
| FIGURE_PROSE_MIN_WORDS (8) | 6: 0.3501, 10: 0.3523, 12: 0.3532 | keep 12, **later reverted (ruling below)** | 0.3532 |
| FIGURE_PROSE_LINE_SHARE (0.5) | 0.6: 0.3532 | no gain | 0.3532 |
| FIGURE_PROSE_RUN_LINES (3) | 5: 0.3532 | no gain | 0.3532 |
| CAPTION_MAX_DISTANCE_LINES (4.0) | 3.0: 0.3528, 5.0: 0.3547, 6.0: 0.3547, 8.0: 0.3547 | keep 5.0 | 0.3547 |
| CAPTION_MAX_LINES (6) | 4: 0.3536, 8: 0.3547, 10: 0.3547 | no gain | 0.3547 |
| FOOTNOTE_BOTTOM_SHARE (0.35) | 0.3: 0.3564, 0.4: 0.3532 | keep 0.3 | 0.3564 |
| FOOTNOTE_MAX_SIZE_RATIO (0.85) | 0.825: 0.3522, 0.875: 0.3564, 0.9: 0.3543 | no gain | 0.3564 |
| FORMULA_MIN_SYMBOL_SHARE (0.3) | 0.25: 0.3560 | no gain | 0.3564 |

### Pass 3 (grid edges; start 0.3564)

| Constant (current) | Tried: train macro F1 | Result |
| --- | --- | --- |
| FIGURE_PROSE_MIN_WORDS (12) | 16: 0.3565, 24: 0.3565 | no gain |
| FOOTNOTE_BOTTOM_SHARE (0.3) | 0.275: 0.3533, 0.325: 0.3562 | no gain |

### Ruling: the prose guard is not tuned away

Raising `FIGURE_PROSE_MIN_WORDS` from 5 to 12 gained 0.003 on train, but it effectively switches off the E5a controller ruling that prose set over a picture or shaded box is text, not a figure (`test_prose_over_a_picture_or_shape_is_not_a_figure` fails: its 10-word prose lines become figure labels). DocLayNet labels any text inside a Picture region as Picture, so its score rewards hiding text in figures, which the ruling exists to prevent. The change was reverted to 5.

### Pass 4 (with `FIGURE_PROSE_MIN_WORDS` back at 5; start **0.3535**)

| Constant (current) | Tried: train macro F1 | Result |
| --- | --- | --- |
| FIGURE_MIN_AREA (0.005) | 0.0025: 0.3537, 0.01: 0.3525, 0.02: 0.3495 | no gain |
| FIGURE_MAX_TEXT_COVERAGE (0.4) | 0.3: 0.3531, 0.5: 0.3535 | no gain |
| CAPTION_MAX_DISTANCE_LINES (5.0) | 4.0: 0.3519, 6.0: 0.3535 | no gain |

### Pass 5: controller ruling on the figure thresholds, then re-convergence

**Ruling (E11a review, binding).** `FIGURE_MIN_AREA` returns to the spec value 0.02, although that costs about 0.004 of train F1. The E5a content-typing ruling outranks the score. At 0.005, the dev-corpus figures rose from 740 to 1,026. doc_003 went from 2 figures to 27, mostly shaded statement-table rows such as "Loss carryforwards - Foreign 2,831 2,796", and the 10-K's banner rows became figures again.

`FIGURE_MAX_TEXT_COVERAGE` was then re-checked on the same dev pages (51 dev PDFs plus the local 10-K, pymupdf, `FIGURE_MIN_AREA` 0.02):

| `FIGURE_MAX_TEXT_COVERAGE` | Dev figures | doc_003 | 10-K |
| --- | --- | --- | --- |
| 0.3 (E5a) | 742 | 2 | 2 |
| 0.4 (tuned) | 744 | 3 | 2 |

At 0.4, doc_003 gains a statement-row figure ("Foreign government and agency securities 506 … 505") and doc_049 gains a figure of mixed text. Both are the mistyping that the ruling forbids, so it was reverted to 0.3 as well.

**Re-convergence.** The other kept constants were re-swept around the final set, with `FIGURE_MIN_AREA` 0.02, `FIGURE_MAX_TEXT_COVERAGE` 0.3 and `FIGURE_PROSE_MIN_WORDS` 5. The start was **0.3496**. No change rose by 0.001 or more, so the set has converged:

| Constant (current) | Tried: train macro F1 |
| --- | --- |
| PARAGRAPH_GAP_RATIO (1.75) | 1.5: 0.3476, 1.6: 0.3476, 1.9: 0.3487, 2.0: 0.3485 |
| CAPTION_MAX_DISTANCE_LINES (5.0) | 1.5: 0.3400, 3.0: 0.3452, 4.0: 0.3480, 6.0: 0.3496 |
| CAPTION_MAX_LINES (6) | 4: 0.3485, 5: 0.3485, 8: 0.3496 |
| FOOTNOTE_BOTTOM_SHARE (0.3) | 0.25: 0.3444, 0.275: 0.3466, 0.325: 0.3495, 0.35: 0.3480 |
| FOOTNOTE_MAX_SIZE_RATIO (0.85) | 0.825: 0.3455, 0.875: 0.3496, 0.9: 0.3476 |

Each kept change is worth +0.001 to +0.01 on 400 pages, so the smaller ones are close to sampling noise.

### Final values (train macro F1 0.3305 → 0.3496)

| Constant | Spec / E5a start | Shipped |
| --- | --- | --- |
| `PARAGRAPH_GAP_RATIO` | 1.5 | 1.75 |
| `CAPTION_MAX_DISTANCE_LINES` | 1.5 | 5.0 |
| `CAPTION_MAX_LINES` | 4 | 6 |
| `FOOTNOTE_BOTTOM_SHARE` | 0.25 | 0.3 |
| `FOOTNOTE_MAX_SIZE_RATIO` | 0.9 | 0.85 |

Every other constant keeps its starting value, including `FIGURE_MIN_AREA` 0.02, `FIGURE_MAX_TEXT_COVERAGE` 0.3 and `FIGURE_PROSE_MIN_WORDS` 5, by ruling. Each results JSON records the effective values under `thresholds`.

### Dev-corpus checks (final constants; commit given per line)

- **TOC no-regression (spec §10.3):** `benchmarks.toc --engines pymupdf --elements` on the 24 bookmarked dev PDFs gives depth <= 2 micro F1 **0.7083** in element mode, against 0.7031 for `fast`. That meets the 0.703 target (`benchmarks/results/2026-10-19/toc-elements.json`, commit `1a02fcb`). It was 0.7071 at `c09b9f7` and `a0830ea`, and 0.7074 at `213979e`, before the XY-cut table fix. Tuning did not change it.
- **Ownership invariant:** on all 51 dev PDFs, every line with text is in exactly one element. There were 0 failures and no empty non-figure element.
  - pymupdf, after the column-first rewrite (`1a02fcb`): 316,617 lines, 89,516 elements, 0 failures (it was 89,414 at `a0830ea`, 89,410 at `c09b9f7` and 90,908 at `213979e`).
  - pdfplumber: 331,104 lines, 95,779 elements, counted at `ee2a829`, before the XY-cut table fix. It was not re-counted.

## Reading order (spec §10.2)

```
PYTHONPATH=src .venv/bin/python -m benchmarks.elements_bench --reading-order
```

Each dev PDF is loaded twice with pymupdf and no OCR: in element mode, recording the line list and the permutation that `order_lines` returns, and as `fast`. Bookmarks are off for the TOC in both, as in the TOC benchmark.

- **Results:** `benchmarks/results/2026-10-19/elements-reading-order.json`, from commit `1a02fcb`: the column-first XY-cut of the last subsection of this section, with clean code. The runs at `c09b9f7` (the row-alignment fix below) and at `a0830ea` (the short-columns patch) were overwritten; their figures are quoted below as *after the row-alignment fix* and *after the short-columns fix*. **The tables, spot checks and per-document figures between here and the last subsection describe the XY-cut of `c09b9f7`**; the last subsection gives what changed.
- **Before the fix:** the same code measured `src` at `2f3825a`, unchanged since `aa729eb`. That run is quoted below as *before*; it was kept out of the repository.
- **Superseded:** `2026-10-18/elements-reading-order.json` is the first E11b run, made before the fix.

The run covers 5,036 pages with at least one text line.

### What the single-column check can and cannot show

`xy_cut_order` returns the input order exactly when it cuts no column gutter on a page (the identity guard).

- **Pages with no gutter (4,482 after the column-first rewrite; 4,520 at `c09b9f7`).** Their body order equals `fast`'s by construction. That is a property of the code, not a measurement: a page that a reader would call single-column but that XY-cut cuts anyway is simply not in this set.
- **The non-trivial checks on these pages:**
  - the element-mode line list equals `fast`'s `extract_lines` on all 5,036 pages (checksums of the line texts in order);
  - with furniture included, 4,216 of 4,482 no-gutter pages equal `fast`. The other 266 differ only in where furniture lines sit: element mode puts running headers, footers and page numbers at the page's start or end, by the E5b design. doc_030 has 106 such pages, doc_042 86 and doc_002 17.

### Non-circular check: lines moved on cut pages of single-column documents

A reader's notion of a single-column page cannot be checked from the cut decision itself. So the check is: in each dev document where fewer than half the pages have a cut, count the body lines that XY-cut moves on its pages that do have one. *Moved* is the number of lines that must move to restore the extraction order: the page's line count minus the longest increasing subsequence of the new order.

The first run of this benchmark (review E11b) found a false gutter on doc_004 p27, a single-column revenue table in a 10-Q:
- XY-cut cut between the row labels and the number columns, giving every label first and then every number. That breaks row association in exactly the financial tables docrooter is for.
- The 20%-of-text-width column guard did not stop it, because the table's labels and column headers are wide enough.

**The fix (controller ruling; `src/docrooter/pdf/reading_order.py`).** A vertical cut is rejected when both of these hold:
1. **The sides are row-aligned.** At least 50% of the boxes on the side with fewer boxes have a box on the other side whose vertical centre is within 0.2 x the median line height.
2. **One side is made of cells.** Its median box is narrower than 10% of the text width.

The second condition and the tight tolerance are refinements of the ruling (which said 0.5 x the line height), and the data showed both were needed:
- **Text columns are row-aligned too.** Text columns set on one baseline grid pass the alignment test: 67-96% of lines in doc_028 (ResNet) and doc_045's two-column pages. Rejecting those cuts would read real articles row by row.
- **The 0.5 tolerance finds chance partners.** Next to dense prose, almost any box has a line within half a line height. For example, 100% of doc_045 p13's chart labels "align" with the text beside them.
- **The first cell threshold was too high.** At 20% of the text width, the narrow three-column pages of doc_049 (columns at about 20% of the text width) were read row by row. At 10%, only tables stay together: number columns are 3-7% of the text width.

Unit tests pin the new rule:
- an unruled label/number table keeps its row order;
- a two-column article with independent baselines still reads column by column;
- text columns on one baseline grid, and three narrow brochure columns, are still columns;
- the existing reading-order tests pass.

Mostly single-column documents (gutters on less than 50% of their pages):

| | Before (`2f3825a` src) | After the row-alignment fix (`c09b9f7`) | After the short-columns fix (`a0830ea`) | After the column-first rewrite (`1a02fcb`) |
| --- | --- | --- | --- | --- |
| Documents | 46 | 47 (doc_045 drops to 36 of 73 column pages) | 47 | 45 (doc_011 and doc_045 now have a gutter on 58 of 113 and 42 of 73 pages) |
| Pages with a cut | 228 | 221 | 222 | 147 |
| Body lines moved on those pages | 5,755 (of 20,692) | 5,425 (of 19,352) | 5,429 (of 19,428) | 2,225 (of 10,544) |
| ... on the same 46 documents | 5,755 | 4,545 | 4,549 | not comparable (the document set changed) |

| Document | Pages with a cut, before -> after | Lines moved, before -> after |
| --- | --- | --- |
| doc_004 10-Q (p27 and p28: the revenue table) | 3 -> 1 | 34 -> **0** |
| doc_005 10-Q | 4 -> 2 | 32 -> 5 |
| doc_003 Amazon annual report | 4 -> 0 | 51 -> 0 |
| doc_001 Berkshire annual report | 9 -> 2 | 199 -> 5 |
| doc_009 / doc_010 earnings releases | 3 -> 1 each | 65 -> 3 each |
| doc_042 RP2040 datasheet | 15 -> 2 | 528 -> 54 |
| doc_041 NE555 datasheet | 5 -> 3 | 85 -> 21 |
| doc_046 Financial Stability Report | 16 -> 13 | 280 -> 164 |
| doc_048 Google environmental report | 19 -> 18 | 824 -> 737 |
| doc_011 bilingual prospectus | 56 -> 56 | 2,062 -> 2,062 |
| doc_039 GNU coreutils manual | 24 -> 22 | 704 -> 685 |
| doc_047 9/11 Commission Report | 11 -> 11 | 201 -> 201 |

Every financial filing (doc_001 to doc_005, doc_009, doc_010) now moves at most 5 lines per document. doc_004 p27 and p28 are no longer cut. Before the fix, 120 of p27's 137 body lines and 32 of p28's 150 changed position (the review's count); now 0 do, and both keep their extraction (row) order.

**Spot check of 10 pages that are still cut** (in mostly single-column documents; the order checked by eye):

| Page | What it is | Verdict |
| --- | --- | --- |
| doc_011 p50 | German and English text in parallel columns | genuine columns |
| doc_039 p310 | two-column index | genuine columns |
| doc_048 p31 | report page with two text columns | genuine columns |
| doc_047 p11 | staff list in two columns | genuine columns |
| doc_029 p49 | two generated poems side by side | genuine columns |
| doc_050 p7 | slide with text columns | genuine columns |
| doc_011 p42 | label/value terms table, wide value column | false gutter: table |
| doc_035 p11 | maturity-scale grid with wide text cells | false gutter: table |
| doc_046 p16 | chart: axis labels and legend | chart labels |
| doc_044 p3 | datasheet chart: axis labels | chart labels |

So 6 of 10 are true multi-column layouts. Two are tables whose cells are wide text, which the cell rule deliberately does not catch, and two are charts, where any order of axis labels is arbitrary. Tables with wide text cells remain a known limitation of the heuristic order; heron (`accurate`) orders by layout regions instead.

**The page-global identity guard (review E3).** Once XY-cut finds a gutter on a page, the whole page is re-sorted by bands and row cuts, including the page's single-column regions. The check: on each page with a gutter, take every band XY-cut forms (a run of full-width or of narrower lines) that has no gutter of its own. Such a band is *shifted* when its lines' relative order differs from the extraction order, which a band-local guard would have kept.

| Pages with a gutter | Reordered | Bands | Single-column bands | Shifted bands | Lines in shifted bands |
| --- | --- | --- | --- | --- | --- |
| 517 | 507 | 1,195 | 670 | **2** | 64 of 59,160 |

The row above is the XY-cut of `a0830ea`. For the column-first rewrite (`1a02fcb`) a band is a part of the cut: a region cut at a gutter, or a maximal part with no gutter (a title, a table, a section under columns). There are 554 pages with a gutter, 545 reordered, 3,797 bands, 2,945 single-column bands, and 58 shifted bands with 2,042 lines (of 62,969). A single-column band is cut at its horizontal gaps, which can re-sort its lines by y; the shifted bands are those where that moved a line (mostly figure labels and tables, which have no order to keep).

Both shifted bands are in doc_036 (Future of Jobs Report): a chart's source note on p28, and a chart's label column on p44. No prose band moved, so the guard stays page-global.

**Bookmark-pair order agreement.** This runs on the 24 dev PDFs with at least 5 bookmarks. Headings are matched to bookmarks as in the TOC benchmark (title ratio >= 0.85, page within 1). For each pair of consecutive bookmarks whose two headings both match, the pair agrees when the headings come out in the same order.

| Mode | Pairs agreeing | Share |
| --- | --- | --- |
| `fast` (before XY-cut) | 1,640 of 1,642 | 0.9988 |
| element mode (after XY-cut) | 1,652 of 1,654 | 0.9988 |

- The fix leaves these numbers exactly as they were.
- The two disagreeing pairs are the same in both modes, one each in doc_004 and doc_005.
- Element mode matches 12 more pairs (doc_045 +2, doc_048 +7, doc_049 +3) and adds no disagreement.

**Kendall's tau on two-column documents.** A document counts as two-column when a gutter is cut on at least half of its pages. Only one dev research paper qualifies: doc_028 (Deep Residual Learning, 12 of 12 pages). It has no bookmarks.

Ruling: its gold is its own section numbering. The numbered headings ("3.1 ...") are taken in predicted order, and tau is computed against the order of their numbers. The two modes find different heading sets (9 numbered headings in `fast`, 11 in element mode), so the two taus are not over the same items. One other document qualifies and has bookmarks: doc_049. Before the fix doc_045 qualified too (it has a gutter on 36 of 73 pages after it); its row stays in the table for comparison.

| Document | Column pages (after the fix) | Gold | `fast` tau | Element-mode tau |
| --- | --- | --- | --- | --- |
| doc_028 Deep Residual Learning | 12 / 12 | section numbers | 0.944 (9 headings) | **1.000** (11 headings) |
| doc_045 Monetary Policy Report | 36 / 73 | bookmarks | 0.736 | **0.800** |
| doc_049 NVIDIA Sustainability Report | 27 / 41 | bookmarks | 1.000 | 1.000 |

- doc_049 lost column cuts on four pages: its GRI index tables (pp. 27, 30, 31, 38), which now keep row order.
- doc_036 (no bookmarks) also crosses the 50% line. Its numbered "headings" are the numbered items of its guide panels, laid out in grids (pages 89-92 and 259-260), not section numbers, so they are no gold for it.

**Column-first reading order (E12 fix rounds 1 and 2, `a0830ea` then `70dd2ca` to `1a02fcb`).**

*The bug.* A heading over one of two short columns of unequal length (for example 16 pt over two 10 pt columns of 5 and 4 lines at 12 pt leading) read the right column first. The line-gap median left out the zero gaps of tight leading, so the row-cut threshold was above every gap that separates a heading from its columns; the heading stayed in the column block and stretched one side, so the sides shared under 60% of the band; no gutter was cut, and the extractor's order was kept, which for PyMuPDF puts the shorter column first. The first patch (`a0830ea`, set aside up to two boxes above the other side's top) closed one shape of it. The re-review's grid of 960 layouts (a title or not; a short, medium and wide heading, the medium one ending inside the gutter; column lengths 5/4 to 12/3; leading 11 to 15 pt; two heading spacings; a section under the columns or not) still read the right column first in **300** of them (381 at `5e7daaf`).

*The rewrite.* `xy_cut_order` is now column-first (controller ruling): spanning boxes (60% of the text width) cut a region into bands; in a band only the other boxes are projected onto x, a gutter is a valley of at least 8 pt or three character widths with two boxes (two that are 20% of the text width) on each side and y-ranges overlapping by half of the shorter side, and up to three titles or headings that cross it are separators; the left side is read before the right, band by band; a region with no gutter is cut at its largest gap; and the line-gap median counts the zero gaps. Two additions were needed to keep what the old coverage test did incidentally, now that every region is judged alone:
- a section break under columns (a gap of 1.25 line heights with a gutter above it and a box across the gutter below it) is cut first, so a table or a section under two columns is not read as the end of one of them;
- the table guard of `c09b9f7` also treats a side of figures (a line that is mostly digits) as cells, whatever the width of the line, and checks a projection with three or more columns column by column (a cover page of name, ticker and exchange). Without them, doc_003, doc_004 and doc_001 read the labels of a statement before its figures again.

A cheaper first try, lowering the row-cut threshold, was rejected: it cut the golden two-column page into rows before its gutter was seen. Two existing unit tests changed with the ruling: a short column beside a tall one is now a gutter (its y-range lies inside the tall one's: the 50% rule is of the shorter side), and the full-width test uses a 290 pt heading (spanning is 60%, not 75%).

*Tests.* `test_columns_read_left_then_right_over_a_grid_of_layouts` regenerates the 960 layouts and allows at most 19 failures (2%); **0 fail** (300 at the re-review's HEAD). The 72-PDF parametrised test of the first fix and the box-level tests also pass.

*Dev corpus, before (`a0830ea`) and after (`1a02fcb`).*
- Line invariant: every line with text in exactly one element, 0 failures; 316,617 lines, 89,414 -> 89,516 elements.
- `benchmarks.toc --elements`: element-mode depth <= 2 micro F1 0.7071 -> **0.7083** (target >= 0.703; `fast` 0.7031).
- Lines moved on the mostly single-column documents (the proxy for "single-column pages are left alone"): doc_003 0 -> 0, doc_004 0 -> 0, doc_001 5 -> **19** (target <= 20). doc_001 has 3 cut pages: p136 (9 lines moved: a two-column table of company and description) and p151 (10: the directors' biographies in two columns) move lines, a third moves none. doc_005 5 -> 0, doc_007 6 -> 0, doc_009 and doc_010 3 -> 0, doc_051 5 -> 0. Documents that gain lines moved are the ones with real column pages that were read across before (doc_012 172 -> 224, doc_014 24 -> 53, doc_042 54 -> 93, doc_041 21 -> 72 with chart and diagram labels, doc_031 0 -> 14 and doc_043 0 -> 10 with one page each).
- Pages with a gutter 517 -> 554; doc_011 (a bilingual prospectus in parallel columns) and doc_045 now have a gutter on 58 of 113 and 42 of 73 pages and leave the mostly single-column set.
- Kendall tau: doc_028 (section numbers) 1.000 (11 headings) -> 1.000; doc_049 1.000 -> 1.000; doc_045 0.800 -> 0.800; doc_036's numbered items 0.410 -> 0.600. Bookmark-pair agreement 1,652 of 1,654 -> 1,651 of 1,653 (the same two disagreements).
- Time: the order step over doc_042 (47,743 lines) took 0.14 s before the rewrite and 3.3 s after (15.3 s in the first column-first version, before the search was bounded); doc_001 0.03 -> 0.27 s. The TOC benchmark's element mode took 5.08 -> 5.24 s per document in the same runs (the load average differs).

**TOC and the line invariant after the fix.**
- `benchmarks.toc --engines pymupdf --elements` (`2026-10-19/toc-elements.json`) gives depth <= 2 micro F1 **0.7071** in element mode, against 0.7031 for `fast`. The target is >= 0.703, and the value before the fix was 0.7074.
- On all 51 dev PDFs, every line with text is in exactly one element: 316,617 lines, 89,414 elements, 0 failures (counted at `a0830ea`; the check wraps the element builder and asserts one owner per line).

## OCR and routing (spec §10.4)

```
PYTHONPATH=src .venv/bin/python -m benchmarks.ocr_bench
```

Results are in `benchmarks/results/2026-10-18/ocr.json`. Its stamp says `aa729eb`, but that is only the `src` revision: the stamp ignored uncommitted benchmark code at the time, and that has since been fixed. The measuring code is `benchmarks/ocr_bench.py` as committed in `2f3825a`. The element-kind agreement below predates the XY-cut table fix (`c09b9f7`), which can change the order, though not the kinds, of OCR'd table lines.

**Pages.**
- Dev PDFs are shuffled with seed `20261018:ocr`. The first 20 that have at least 3 pages with a clean text layer are taken, with 3 such pages each, also seeded: 60 pages from doc_002, 003, 005, 006, 007, 013, 016, 018, 019, 022, 023, 024, 031, 033, 035, 040, 044, 045, 047 and 050. doc_017 was skipped, because it has a single page.
- A page is clean when it is routed `text` with no OCR region, has at least 200 text-layer characters and no invisible text, and its images cover at most 10% of it. Text inside images is not in the text layer, so it could not serve as ground truth.
- Each page is copied into a one-page PDF (the ground truth: its text layer, without annotations or form fields). It is also rendered at 200 dpi into a one-page, image-only PDF of the same size.
- Everything is cached in the git-ignored `benchmarks/.cache/ocr/`. The selection digest is `880a6925c2b8`.

**Scoring.**
- CER and WER are computed after the spec normalisation: NFKC, words hyphenated at a line end joined before a lowercase letter, and whitespace collapsed.
- An engine returns lines in its own order, so both sides are put in one fixed order before scoring. Lines are grouped into rows by vertical centre, rows run top to bottom and lines run left to right within a row.
- CER and WER are pooled: total edits over total reference characters or words. Word recall and precision are order-free (the multiset intersection of words), so they show how much of an error rate comes from order rather than recognition.
- Each engine runs in its own worker process, one after the other. The worker recognises every page after a warm-up page, so model loading is excluded from the per-page time and reported as load.
- *Elements s/page* is `load_pdf(elements=True, ocr=engine)` on the image-only page: routing, rendering, OCR and element typing.

| Engine | CER | WER | Median page CER | Word recall | Word precision | OCR s/page | Elements s/page | Load s | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ocrmac 1.0.1 (Apple Vision, accurate) | 0.102 | 0.208 | 0.013 | 0.802 | 0.961 | 0.73 | 0.83 | 1.2 | 1,184 MiB |
| rapidocr 3.10.0 (onnxruntime, CPU) | 0.089 | 0.196 | 0.043 | 0.809 | 0.953 | 3.04 | 3.16 | 2.4 | 3,311 MiB |
| tesseract 5.5.1 (`--psm 3`, `eng`) | **0.058** | **0.168** | **0.012** | **0.842** | **0.965** | 1.72 | 1.83 | 0.5 | 803 MiB + 263 MiB (binary) |

Rendering a page at 200 dpi takes 0.10 s on top of the OCR time. Peak RSS is the worker's whole process, which is 126 MiB before the engine is created. For tesseract, the separate binary's peak is listed after the `+`. Every model process stayed far inside the 8 GiB budget.

**Reading the numbers.**
- The pooled CER is dominated by a few pages; the median page is at about 1% for ocrmac and tesseract.
  - doc_007 p1 (Berkshire's performance table) scores 0.69-0.79 CER for every engine. Its text layer is full of dot leaders that no engine reads, and word precision there is 0.94 (measured for ocrmac).
  - doc_007 p5 and doc_013 p49 (dense tables) score 0.22-0.34.
- ocrmac fails on the three CHIPS Act pages (doc_024, CER 0.51-0.65; the other engines get 0.04-0.06). Its word recall on those pages is 0.35 pooled (0.30-0.39 per page), and it outputs unrelated words. On doc_050 p23 (an income-statement slide), it reads 2 of 5 number columns.
- rapidocr's median page CER is about 4%, against about 1% for the others: its errors are small but spread over most pages. It is also the slowest engine and the largest in memory.
- `ocr=auto` picks tesseract when its binary is on PATH, then ocrmac on macOS, then rapidocr. Tesseract is first for accuracy: its CER is 0.058, against 0.089 for rapidocr and 0.102 for ocrmac. ocrmac precedes rapidocr for speed and memory (0.73 against 3.04 s per page, 1.2 against 3.2 GiB peak), although rapidocr is the more accurate of the two. ocrmac is the fastest engine and fine on prose, but it is the least reliable on legal text and tables. (Before this ordering, `auto` picked ocrmac first; the `balanced` document-type run in `docs/benchmarks-pdf.md` was made with ocrmac.)

**Routing confusion.** Expected labels: every page of the 51 dev PDFs is born-digital (`text`), and every image-only page needs `ocr`. A region decision keeps the page's text layer, so it counts as `text`. The dev rows are `routing-dev.json` from E5c (commit `8913b7a`). The pymupdf routing path has not changed since: the later `objects.py` changes are the pdfplumber geometry fix and the `tables` switch.

| Expected \ Decided | `text` | `text+ocr` | `ocr` |
| --- | --- | --- | --- |
| `text`: 51 dev PDFs (5,069 pages; 127 with OCR regions) | 5,055 | 13 | 1 |
| `ocr`: 60 image-only pages | 0 | 0 | **60** |

- The one dev page replaced by OCR, doc_051 p27, has no text layer at all: it is a full-page image.
- The 13 `text+ocr` pages are full-page images carrying 21-254 characters of text layer: covers and photo pages of doc_013, 036, 048, 049 and 051. They keep their text, and OCR adds only the lines that text lacks.
- No page with a usable text layer was replaced.

**Element-kind agreement.** Each of the 924 text-layer elements on the 60 pages is compared with the OCR element (from the image-only page) whose box overlaps it most.

| Engine | Same kind (elements) | Same kind (characters) | No overlapping OCR element |
| --- | --- | --- | --- |
| ocrmac | 778 / 924 (0.842) | 0.904 | 40 |
| rapidocr | 820 / 924 (0.887) | 0.923 | 9 |
| tesseract | 771 / 924 (0.834) | 0.899 | 21 |

Paragraphs agree in 94-98% of cases. Where the kinds differ, the cause is by design:
- An OCR page has no ruled tables, so 10-11 of the 11 tables become paragraphs (tesseract misses one).
- Its only image is the page scan, so 13-14 of the 14 figures become paragraphs or are missed (tesseract types one as a title).
- Headings come only from size and isolation (no bold, no font), so 12-17 of 32 headings and 10-13 of 31 titles are kept.
- Page numbers agree in 14-32 of 44 cases.

Full confusion tables are in the JSON.

## Speed and memory (spec §10.5)

On the ten-document Q5 profiling set (1,439 pages, no OCR), `fast` takes 0.99 s per 100 pages. Element mode takes 4.64 s with heuristic pymupdf, 7.65 s with heuristic pdfplumber and 64.6 s with heron. Peak RSS is 200, 263, 308 and 904 MiB.

The OCR engines take 0.73 s (ocrmac), 1.72 s (tesseract) and 3.04 s (rapidocr) per page at 200 dpi. `fast` stays within 5% of v0.3: +0.8% to +2.8%.

Method, raw files and the `fast` A/B: `docs/performance.md`, *Element mode, OCR and the `fast` path (v0.4)*.
