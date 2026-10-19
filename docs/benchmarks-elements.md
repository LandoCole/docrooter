# Element benchmarks

## Element kinds on DocLayNet (spec §10.1)

**Data.** [DocLayNet](https://github.com/DS4SD/DocLayNet) (Pfitzmann et al., KDD 2022), licence [CDLA-Permissive-1.0](https://cdla.io/permissive-1-0/): 80,863 human-annotated pages from six document categories, with 11 layout classes. The benchmark reads the Hugging Face re-packaging `docling-project/DocLayNet-v1.2` at revision `0daf93102e2efce76c3e11a274a5e0d0969391d3`. It holds each page's born-digital single-page PDF in a binary column, beside the gold boxes and the PDF text cells inside each box. `benchmarks/doclaynet.py` pins every parquet shard by SHA-256 (the LFS object id at that revision). It downloads every shard of a split, because the shards are sorted by category, checks each one, samples, writes the sampled PDFs to the git-ignored `benchmarks/.cache/doclaynet/`, and then deletes the shards.

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

These are the numbers for the constants that ship (commit `213979e`): the tuned values below, with the spec figure thresholds and the pdfplumber CropBox fix.

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
2. **heuristic-pdfplumber boxes were in the wrong space on cropped pages.** On pages whose CropBox differs from the MediaBox, pdfplumber element boxes were in MediaBox space, while the gold and pymupdf use the CropBox. That affected 5 test pages (all financial reports) and 13 train pages. This was fixed at the source (`pdf/objects.py`).

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
| heuristic-pdfplumber | 0.2965 | 0.3131 | 0.3110 (with the CropBox fix) |

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

### Dev-corpus checks (final constants, commit `213979e`)

- **TOC no-regression (spec §10.3):** `benchmarks.toc --engines pymupdf --elements` on the 24 bookmarked dev PDFs gives depth <= 2 micro F1 **0.7074** in element mode, against 0.7031 for `fast`. That is unchanged by tuning and meets the 0.703 target (`benchmarks/results/2026-10-18/toc-elements.json`).
- **Ownership invariant:** on all 51 dev PDFs, every line with text is in exactly one element. There were 0 failures and no empty non-figure element.
  - pymupdf: 316,617 lines, 90,908 elements.
  - pdfplumber, with the CropBox fix: 331,104 lines, 95,770 elements.
