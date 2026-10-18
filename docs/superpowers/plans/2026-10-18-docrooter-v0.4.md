# docrooter v0.4 Implementation Plan: elements and the rooter

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every visible part of a PDF or Markdown document becomes a typed, ordered element inside the section tree. A configurable rooter decides text versus OCR per page and the engine for each task. Profiles are `fast`, `balanced` and `accurate`.

**Architecture:**
- **Engine-neutral page objects:** lines, images, drawings and ruled tables from pymupdf or pdfplumber feed a pure, torch-free element builder.
- **Element-mode pipeline:** route, then OCR, then XY-cut, then signals and headings, then element typing, then sections built from elements. It sits beside the untouched `fast` path.
- **Optional extras:** OCR engines and the heron ONNX layout model, behind protocols.
- **`docrooter.rooter`:** resolves the profile or config into engines and returns an `Analysis`.

**Tech Stack:** Python ≥ 3.10, pydantic v2, pymupdf / pdfplumber (existing extras), onnxruntime + pypdfium2 + huggingface_hub (`layout`), rapidocr (`ocr`), ocrmac (`ocr-mac`), and the tesseract CLI (system binary).

**Spec:** `docs/superpowers/specs/2026-10-18-docrooter-v0.4-elements-rooter-design.md`. The program is `docs/superpowers/program/2026-10-18-docrooter-element-intelligence-program.md`; the research is `docs/superpowers/research/2026-10-18-element-classification-research.md`.

## Global Constraints

- **All v0.1–v0.3 global constraints still hold:**
  - the core is torch-free, with lazy extras, and no recursion over trees;
  - unit tests never import torch. The autouse guard in `tests/unit/conftest.py` enforces this. Add onnxruntime and the OCR engines to that guard;
  - secrets live only in `.env`;
  - use `PYTHONPATH=src` for all commands, never a plain `uv sync`, and use `uv sync --inexact` or `uv run --extra …`.
- **The fast path is frozen.** `load()`, `load_pdf()` with default arguments, `classify_document()` and `normalize_sections()` must produce byte-identical output to v0.3 (`feat/v0.3` at 28fbf9d), with no new `meta` keys, and run within 5% of v0.3's speed.
- **Element mode** is entered only through `load_pdf(elements=True)`, `load(elements=True)` or `process()` with a non-`fast` profile.
- **New extras:**
  - `layout = ["onnxruntime>=1.18", "pypdfium2>=4.30", "huggingface_hub>=0.23"]`;
  - `ocr = ["rapidocr>=3.0", "onnxruntime>=1.18", "pypdfium2>=4.30"]`;
  - `ocr-mac = ["ocrmac>=1.0; sys_platform == 'darwin'"]`;
  - `all` includes all three;
  - core gains `tomli>=2; python_version < '3.11'`.
- **Licences:** no GPL, AGPL or non-commercial dependency in any default or in the `layout`/`ocr` extras. Never import `pymupdf.layout` and never call pymupdf4llm outside `engine="pymupdf4llm"`.
- **Missing extras** raise through `docrooter._extras.require(module, extra)`, with the existing message format.
- **Memory:** at most one model-loading process at a time, with an 8 GB budget (`benchmarks._guard`). The heron session and OCR engines are created lazily, reused within a `process_many` run, and released by `close()`.
- **Held-out:** `benchmarks/heldout/` is never opened in v0.4.
- **Heuristic thresholds** are tuned only on DocLayNet train-split pages and dev-corpus documents. DocLayNet test-split pages are scored once, in E11a's final step.
- **Commits** are conventional and end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Rotated pages** (`/Rotate 90`) and landscape pages: boxes must be in the displayed page's coordinates, and XY-cut must order the displayed layout. Pinned by `test_rotated_page_boxes_are_in_display_space` (E2).
2. **Vector-heavy pages** (charts or maps with tens of thousands of paths) must not blow up time. Drawing collection is capped, and the page is still processed. Pinned by `test_vector_heavy_page_is_bounded` (E2).
3. **Mixed PDFs** (scanned pages between text pages): OCR lines must land in page order, and sections must continue across them. Pinned by `test_mixed_scanned_and_text_pages_keep_page_order` (E5b).
4. **Degenerate geometry:** empty pages, zero-area boxes, images placed partly off the page. Boxes are clipped to the page, with no crash and no empty elements. Pinned by `test_degenerate_boxes_are_clipped_or_dropped` (E2) and `test_empty_page_yields_no_elements` (E5a).
5. **Non-Latin text layers** (CJK, Cyrillic, Arabic) must not be routed to OCR as garbled. The dictionary-like token rule applies only when ≥ 50% of letters are Latin. Pinned by `test_cjk_page_is_not_garbled` (E4).

---

## File structure

| File | Responsibility |
| - | - |
| `src/docrooter/elements/__init__.py` | Re-exports `Element`, `Box`, `TableData`, `FigureData`, `ElementKind` |
| `src/docrooter/elements/models.py` | The element models (spec §4) |
| `src/docrooter/document/models.py` | + `Document.elements`, `Section.element_ids` |
| `src/docrooter/document/markdown.py` | + Markdown elements |
| `src/docrooter/pdf/objects.py` | `PageObjects`, and extraction for pymupdf and pdfplumber |
| `src/docrooter/pdf/reading_order.py` | Recursive XY-cut |
| `src/docrooter/pdf/routing.py` | Page metrics and text-vs-OCR decisions (pure) |
| `src/docrooter/ocr/{__init__,base,render,ocrmac,rapidocr,tesseract,registry}.py` | The OCR protocol, page rendering, adapters, registry, and OCR→`Line` |
| `src/docrooter/pdf/elements.py` | Element typing rules 1–10, caption linking, paragraph continuation |
| `src/docrooter/pdf/element_mode.py` | The element-mode pipeline: route → OCR → XY-cut → signals/headings → typing → sections |
| `src/docrooter/pdf/load.py` | `load_pdf(..., elements=False, ocr=None, layout="heuristic")` dispatch only |
| `src/docrooter/layout/{__init__,base,heron,registry}.py` | The layout-engine protocol, heron ONNX, and region merge |
| `src/docrooter/summarize.py` | Sentence splitter, `lead`, `centroid`, table descriptions |
| `src/docrooter/rooter/{__init__,config,profiles,process,analysis}.py` | `RouterConfig`, profiles, `process`/`process_many`, `Analysis`, escalation |
| `src/docrooter/chunking.py` | `chunk()`, `Chunk` |
| `src/docrooter/embed.py` | `embed()`, `EmbeddingSet` |
| `src/docrooter/cli.py` | `elements`, `process` and `config show` commands |
| `src/docrooter/ui/server.py`, `ui/static/*` | The profile selector, Elements tab and route stage |
| `benchmarks/elements_bench.py`, `benchmarks/doclaynet.py`, `benchmarks/ocr_bench.py` | Harnesses |
| `docs/benchmarks-elements.md`, `docs/site/concepts/{elements,rooter}.md`, `docs/site/guides/{ocr-and-layout,chunking-for-rag}.md` | Documentation |

---

### Task E1: Element model and Markdown elements

**Files:**
- Create: `src/docrooter/elements/__init__.py`, `src/docrooter/elements/models.py`
- Modify: `src/docrooter/document/models.py` (add fields), `src/docrooter/document/markdown.py`, `src/docrooter/document/load.py` (`elements: bool = False`), `src/docrooter/__init__.py` (exports)
- Test: `tests/unit/elements/test_models.py`, `tests/unit/document/test_markdown_elements.py`

**Interfaces:**
- Produces:
  - `Element`, `Box`, `TableData` (`to_markdown() -> str`), `FigureData` and `ElementKind`, exactly as in spec §4;
  - `Document.elements: list[Element] = []` and `Section.element_ids: list[str] = []`;
  - `parse_markdown(text, *, source, elements: bool = False) -> Document`, plus whatever the current Markdown entry point is named, given the same keyword;
  - `load(path, *, elements: bool = False, **kw)`.

- [ ] **Step 1: Write the failing tests.**
  - `test_v03_document_json_round_trips_unchanged`: a v0.3 `Document` JSON fixture, without the new keys, loads and dumps byte-identically. This includes `exclude_defaults` behaviour; check how `Document` serialises today and keep it.
  - `test_table_to_markdown_pads_ragged_rows`: `TableData(rows=[["a","b"],["c"]], header_rows=1).to_markdown() == "| a | b |\n| --- | --- |\n| c |  |"`.
  - `test_element_ids_follow_reading_order`.
  - `test_markdown_elements_kinds`: one doc with heading, paragraph, nested list (depths 0 and 1), pipe table, fenced code, `$$` formula and image. Kinds come out in order `["heading","paragraph","list_item","list_item","table","paragraph","formula","figure"]`; the code paragraph has flag `"code"`; the image's text is its alt text.
  - `test_markdown_section_element_ids`: each section lists exactly its own elements.
  - `test_markdown_elements_off_by_default`: `load("x.md").elements == []`.
- [ ] **Step 2:** Run `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/elements tests/unit/document -q`. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement the models and the Markdown element pass. Reuse the existing Markdown line scanner; do not add a Markdown dependency.
- [ ] **Step 4:** Run the same command, plus the whole unit suite. Expected: PASS, and the existing tests are unchanged.
- [ ] **Step 5:** Commit `feat(elements): element model and Markdown elements`.

### Task E2: Page objects from pymupdf and pdfplumber

**Files:**
- Create: `src/docrooter/pdf/objects.py`
- Modify: `src/docrooter/pdf/lines.py`, only to expose a per-page iterator if it is needed. Leave `extract_lines` output unchanged.
- Test: `tests/unit/pdf/test_objects.py`

**Interfaces:**
- Consumes: `Line` (`pdf/lines.py`).
- Produces:
  - `PageObjects(page, width, height, rotation, lines, images: list[Box], drawings: list[Box], tables: list[tuple[Box, list[list[str]]]], char_count, invisible_chars, garbled_chars, n_drawing_paths, flags: list[str])`;
  - `extract_page_objects(path, *, engine: Literal["pymupdf","pdfplumber"], max_pages=None, progress=None) -> list[PageObjects]`.
  - Lines must be identical, in content and order, to `extract_lines` for the same engine.

- [ ] **Step 1: Write the failing tests** on generated PDFs (pymupdf, `importorskip`). Each runs for both engines (parametrised) where the engine is installed.
  - `test_objects_lines_match_extract_lines`.
  - `test_image_placement_box`: an inserted PNG gives one image box within ±1 pt.
  - `test_vector_chart_is_one_drawing_cluster`: 30 rectangles plus 5 polylines forming a bar chart become one cluster box.
  - `test_ruled_table_cells`: a 3×3 ruled grid with text gives one table with rows `[["A","B","C"],["1","2","3"],["4","5","6"]]`.
  - `test_rotated_page_boxes_are_in_display_space` (Review Focus 1).
  - `test_vector_heavy_page_is_bounded`: 20,000 tiny paths on one page take < 3 s, the page gets the flag `"drawings_capped"`, and drawings ≤ 200 clusters. The collection cap is `MAX_DRAWING_PATHS = 5000` per page; above it, the whole drawing area becomes one cluster.
  - `test_degenerate_boxes_are_clipped_or_dropped` (Review Focus 4): an off-page image is clipped, and a zero-area box is dropped.
  - `test_char_counts`: `garbled_chars` counts U+FFFD, `(cid:N)`, Private Use Area characters and C0/C1 controls. `invisible_chars` counts render mode 3 (pymupdf; 0 on pdfplumber).
- [ ] **Step 2:** Run `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/pdf/test_objects.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement `extract_page_objects`.
  - **pymupdf:** `get_text("dict", flags=TEXT_PRESERVE_IMAGES)` image blocks, `get_image_info()`, `cluster_drawings()`, `find_tables(strategy="lines")`, and `get_texttrace()` for render mode 3. `page.rotation_matrix` maps everything to display space.
  - **pdfplumber:** `page.images`; rects, lines and curves clustered by box proximity ≤ 6 pt; `find_tables` with `lines` strategies; `(cid:` detection from char text.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat(pdf): engine-neutral page objects`.

### Task E3: Reading order (recursive XY-cut)

**Files:**
- Create: `src/docrooter/pdf/reading_order.py`
- Test: `tests/unit/pdf/test_reading_order.py`

**Interfaces:**
- Produces:
  - `xy_cut_order(boxes: Sequence[tuple[float,float,float,float]], *, page_width: float, text_width: float | None = None, median_char_width: float, median_line_gap: float) -> list[int]`, a permutation of input indexes;
  - `order_lines(lines: Sequence[Line], excluded: set[int]) -> list[int]`, a document-wide permutation, page by page, with excluded (furniture) lines kept at their original relative position at page start or end.

- [ ] **Step 1: Write the failing tests** on synthetic boxes.
  - `test_two_columns_read_left_then_right`.
  - `test_three_columns`.
  - `test_full_width_title_and_figure_split_bands`: a title above two columns, a full-width figure, then two more columns. Order: title, colA1, colB1, figure, colA2, colB2.
  - `test_single_column_order_unchanged`: for single-column input already in top-to-bottom order, the permutation is the identity.
  - `test_gutter_rules`: a gutter narrower than `max(8, 3 × median_char_width)`, or covering < 60% of the band height, is not a cut.
  - `test_no_recursion`: iterative implementation. A 2,000-box page completes and `sys.getrecursionlimit()` is untouched (no-recursion constraint).
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement with an explicit work stack. Bands are split by boxes spanning ≥ 75% of `text_width` (default: the union width of the boxes). Then cut vertical valleys, then horizontal gaps ≥ `1.5 × median_line_gap`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat(pdf): recursive XY-cut reading order`.

### Task E4: Page routing and OCR engines

**Files:**
- Create:
  - `src/docrooter/pdf/routing.py`
  - `src/docrooter/ocr/__init__.py`, `ocr/base.py`, `ocr/render.py`, `ocr/ocrmac.py`, `ocr/rapidocr.py`, `ocr/tesseract.py`, `ocr/registry.py`
- Modify: `pyproject.toml` (extras `ocr`, `ocr-mac`, and `all`), `tests/unit/conftest.py` (forbid importing `onnxruntime`, `rapidocr` and `ocrmac` in unit tests)
- Test: `tests/unit/pdf/test_routing.py`, `tests/unit/ocr/test_ocr.py`; `tests/integration/test_ocr_engines.py` (marker `ocr`)

**Interfaces:**
- Consumes: `PageObjects` (E2).
- Produces:
  - `PageMetrics(char_density, image_cov, garbled, invisible_text, latin_share)`;
  - `RouteDecision(scope: Literal["document","page","region"], target: int | list[float] | None, task: str, engine: str, reason: str, metrics: dict[str, float])`;
  - `page_metrics(objs: PageObjects) -> PageMetrics`;
  - `route_page(objs: PageObjects) -> tuple[Literal["text","ocr"], list[Box], RouteDecision]`, where the boxes are the regions for region OCR;
  - `OcrLine(text, box: Box, confidence: float)`;
  - `class OcrEngine(Protocol): name: str; def recognize(self, image: bytes, *, width_px: int, height_px: int, page: Box) -> list[OcrLine]; def close(self) -> None`;
  - `render_page(path, page: int, *, dpi: int = 200, clip: Box | None = None) -> tuple[bytes, int, int]` (PNG bytes; pymupdf if installed, else pypdfium2);
  - `ocr_lines_to_lines(ocr: list[OcrLine], page: int, width: float, height: float) -> list[Line]` (`size = height_of_box × 0.75`, `font="ocr"`, `bold=False`, block indexes in order);
  - `get_ocr_engine(name: str, *, lang: str = "en") -> OcrEngine`, `register_ocr_engine(name, factory)` and `available_ocr_engines() -> list[str]`;
  - `resolve_auto_ocr() -> str | None`, which tries `ocrmac` (darwin), then `rapidocr`, then `tesseract` (on PATH).

- [ ] **Step 1: Write the failing tests.** Thresholds are exactly spec §7.4.
  - `test_scanned_page_routes_to_ocr`: `image_cov` 0.9 and `char_density` 0.5 route to `"ocr"`.
  - `test_garbled_page_routes_to_ocr`: 15% U+FFFD routes to `"ocr"`.
  - `test_text_page_routes_to_text`.
  - `test_invisible_garbled_layer_routes_to_ocr`.
  - `test_region_ocr_for_textless_large_image`: a 6%-area image with no chars inside is a region; a 4% one is not.
  - `test_cjk_page_is_not_garbled` (Review Focus 5).
  - `test_ocr_lines_to_lines_geometry`.
  - `test_registry_unknown_engine_error_lists_available`.
  - `test_auto_ocr_resolution_order`, with monkeypatched `find_spec` and `shutil.which`.
  - `test_tesseract_tsv_parsing`, on a fixture TSV string with `subprocess.run` faked: words are grouped into lines by `(block, par, line)`, and confidence is the mean of word confidences / 100.
  - `test_missing_extra_message`: `get_ocr_engine("rapidocr")` without the package raises the `pip install 'docrooter[ocr]'` message.
- [ ] **Step 2:** Run `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/pdf/test_routing.py tests/unit/ocr -q`. Expected: FAIL.
- [ ] **Step 3:** Implement. The adapters normalise each engine's output to pixel boxes, then scale to page points. `ocrmac` uses `recognition_level="accurate"`. `rapidocr` uses `RapidOCR()` with default models. `tesseract` runs `tesseract stdin stdout -l {lang} --psm 3 tsv`.
- [ ] **Step 4:** Run the unit tests. Expected: PASS. Then run `PYTHONPATH=src .venv/bin/python -m pytest tests/integration/test_ocr_engines.py -m ocr -q`, which renders a generated text page and checks that each installed engine's OCR contains ≥ 90% of its words. The tesseract binary is installed here; install rapidocr and ocrmac with `uv pip install`, since they are not in uv.lock yet. Record which engines ran.
- [ ] **Step 5:** Commit `feat(ocr): page routing, OCR protocol and engines`.

### Task E5a: Element typing rules

**Files:**
- Create: `src/docrooter/pdf/elements.py`
- Test: `tests/unit/pdf/test_elements.py`

**Interfaces:**
- Consumes:
  - `PageObjects` (E2), `xy_cut_order` / `order_lines` (E3);
  - `DocSignals` (`signals.excluded`, the printed-TOC pages, `body_size`, `median_gap`) and `HeadingCandidate` (`line_index`, `line_count`, `level`), from `pdf/signals.py` and `pdf/headings.py`.
- Produces:
  - `build_elements(pages: Sequence[PageObjects], lines: Sequence[Line], order: Sequence[int], signals: DocSignals, headings: Sequence[HeadingCandidate], *, title_index: int | None, line_source: Sequence[Literal["text","ocr"]] | None = None) -> list[Element]`;
  - elements are in reading order, with ids `e0…`, and `section_id` still `None` (set in E5b);
  - `CAPTION_RE` and the module-level threshold constants, named in UPPER_SNAKE for E11a tuning:
    - `FIGURE_MIN_AREA = 0.02`
    - `FIGURE_MAX_TEXT_DENSITY = 0.5`
    - `CAPTION_MAX_DISTANCE_LINES = 1.5`
    - `FOOTNOTE_BOTTOM_SHARE = 0.25`
    - `FOOTNOTE_MAX_SIZE_RATIO = 0.9`
    - `LIST_INDENT_STEP = 8.0`
    - `FORMULA_MIN_SYMBOL_SHARE = 0.30`
    - `PARAGRAPH_GAP_RATIO = 1.5`
    - `PARAGRAPH_SIZE_DELTA = 0.5`
    - `FIRST_LINE_INDENT = 8.0`

- [ ] **Step 1: Write the failing tests.** One per spec §5.3 rule, on synthetic `PageObjects` and `Line` lists:
  - `test_furniture_kinds`;
  - `test_contents_page_one_element`;
  - `test_table_consumes_its_lines`;
  - `test_image_and_vector_figures_consume_label_text`;
  - `test_caption_links_both_ways_above_table_below_figure`;
  - `test_title_and_heading_levels`;
  - `test_footnote_merges_until_next_marker`;
  - `test_bullets_enumerators_and_depth` (all bullet and enumerator forms from the spec);
  - `test_formula_by_symbols_and_math_font`;
  - `test_paragraph_breaks_gap_size_indent`;
  - `test_paragraph_continues_across_page_with_flag` (two boxes, flag `"continued"`);
  - `test_paragraph_does_not_continue_after_terminal_punctuation`;
  - `test_each_line_in_at_most_one_element`, a property check over all fixtures;
  - `test_empty_page_yields_no_elements` (Review Focus 4);
  - `test_ocr_lines_produce_source_ocr`.
- [ ] **Step 2:** Run `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/pdf/test_elements.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement the rules in spec order. Each rule marks the lines it consumes, and paragraphs take the rest.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat(pdf): element typing rules`.

### Task E5b: The element-mode pipeline in `load_pdf`

**Files:**
- Create: `src/docrooter/pdf/element_mode.py`
- Modify: `src/docrooter/pdf/load.py`, which gains keyword-only `elements: bool = False`, `ocr: str | OcrEngine | None = None`, `layout: str = "heuristic"` (only `"heuristic"` until E6, which widens it to `str | LayoutEngine` and wires regions in) and `ocr_dpi: int = 200`, and dispatches to `element_mode.load_elements` when `elements=True`. The default path is untouched.
- Modify: `src/docrooter/document/load.py`, which passes the new keywords through for PDFs.
- Test: `tests/unit/pdf/test_element_mode.py`, `tests/unit/pdf/test_fast_path_frozen.py`

**Interfaces:**
- Consumes: E2, E3, E4 and E5a; `compute_signals`, `find_headings` and the existing assembly (level assignment, bookmark overrides).
- Produces:
  - `load_elements(path, *, engine, ocr, layout, ocr_dpi, min_heading_score, use_bookmarks, use_printed_toc, max_pages) -> Document`;
  - `Document.meta["routes"]` is a list of `RouteDecision.model_dump()`, `meta["element_mode"] = True`, and `meta["warnings"]` may include `"ocr_skipped"`;
  - new events: `route` (progress per page), `ocr` (start/progress/end) and `elements` (start/end, with counts by kind).

- [ ] **Step 1: Write the failing tests.**
  - `test_fast_path_frozen`: for 6 generated PDFs (single-column, two-column, tables, scanned image-only, garbled font, bookmarks), `load_pdf(p)` JSON equals a golden JSON produced from `feat/v0.3` (28fbf9d). Generate the goldens with a one-off script that runs the v0.3 code from a `git worktree` in the scratchpad, and commit them under `tests/unit/pdf/golden/`. There must be no `routes` key.
  - `test_element_mode_sections_built_from_elements`: every non-furniture element belongs to exactly one section; `Section.text` equals its elements' text joined with `"\n\n"`; and the headings equal the heading elements.
  - `test_two_column_toc_order_fixed`: two-column headings come out in column order.
  - `test_mixed_scanned_and_text_pages_keep_page_order` (Review Focus 3), with a fake OCR engine.
  - `test_scanned_pdf_with_fake_ocr_has_sections`: a scanned PDF with the fake engine gives headings and section text, not an empty `s0`.
  - `test_scanned_pdf_without_ocr_warns`: `ocr=None` gives `"ocr_skipped"` plus the `no_text_layer` behaviour, with routes recorded.
  - `test_region_ocr_fills_figure_text_only`.
  - `test_element_events_order`: route, then extract/ocr, then signals, headings, elements, assemble, done.
- [ ] **Step 2:** Run `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/pdf -q`. Expected: the new tests FAIL and the old ones PASS.
- [ ] **Step 3:** Implement the pipeline:
  1. extract objects;
  2. route each page;
  3. OCR the pages that need it (render, recognize, convert to lines, replace that page's lines);
  4. run region OCR into the figures;
  5. compute signals on the combined lines;
  6. order lines with `order_lines`;
  7. run `find_headings` on the ordered lines;
  8. `build_elements`;
  9. assemble sections from the heading elements and set `section_id`, `element_ids` and `Section.text`.
- [ ] **Step 4:** Run the unit suite. Expected: PASS. Then run the dev TOC benchmark in both modes:

  ```
  PYTHONPATH=src .venv/bin/python -m benchmarks.toc --engine pymupdf --out benchmarks/.cache/toc-fast.json
  ```

  and add `--elements` (implement this flag in `benchmarks/toc.py`) to get the element-mode file.
  - **Target** (spec §10.3): element-mode depth ≤ 2 micro F1 ≥ 0.703.
  - **If it is lower:** fix the rules on dev and re-run. If the gap remains after reasonable effort, report it with per-document deltas. Do not change `fast`.
- [ ] **Step 5:** Commit `feat(pdf): element mode with routing, OCR and sections from elements`. Commit the toc comparison JSON under `benchmarks/results/<date>/toc-elements.json`.

### Task E5c: Element-mode performance and routing precision (added during execution)

Added by controller ruling after E5b. Element mode measured 7.7 s/doc against fast's 1.18 s/doc on the 51 dev PDFs, with about 97% of the time in the page-object pass. Routing also sent 37 born-digital doc_042 pages to OCR as "garbled" and 6 doc_048 pages as "scanned".

**Files:**
- Modify: `src/docrooter/pdf/objects.py`, `src/docrooter/pdf/routing.py`, `src/docrooter/pdf/element_mode.py`
- Test: `tests/unit/pdf/test_objects.py`, `tests/unit/pdf/test_routing.py`, `tests/unit/pdf/test_element_mode.py`
- Create: `benchmarks/results/2026-10-18/element-speed.json`, `benchmarks/results/2026-10-18/routing-dev.json`

**Interfaces:**
- Unchanged signatures.
- `route_page` may return a new decision value `"text+ocr"`: merge OCR lines into the text layer.

- [ ] **Step 1: Profile.** Profile element mode on the 10 Q5 profiling PDFs with pyinstrument and record the top costs in the report. Expected suspects: `find_tables` on every page, `get_drawings` or the clusterer, and a second text extraction.
- [ ] **Step 2: Write the failing tests.**
  - `test_find_tables_skipped_without_rulings`: a page with no horizontal or vertical line segments never calls the engine's table finder.
  - `test_hex_and_identifier_tokens_are_word_like`: a datasheet register page full of tokens such as `0x4000_0000`, `GPIO_CTRL[3:0]` and `RW`, with normal English descriptions, routes to `text`.
  - `test_scanned_rule_with_text_layer_merges`: a page with `image_cov ≥ 0.75`, `char_density < 2` and some text-layer lines routes to `text+ocr`. With a fake engine, the result keeps every text-layer line and adds only OCR lines whose boxes overlap no text-layer line (IoU < 0.3).
  - `test_no_text_layer_page_still_replaced`: zero text-layer characters still routes to `ocr`.
- [ ] **Step 3: Implement.**
  - Skip table finding on pages without ruling segments.
  - Extract text once and derive both the lines and the objects from that one pass.
  - Avoid any repeated per-page work you find in the profile.
  - Token rule: a token containing a digit, or matching `^[A-Za-z_][A-Za-z0-9_\[\]:.]*$` with an underscore or bracket, counts as word-like.
  - Scanned rule: whole-page replacement only when the page has zero text-layer characters. Otherwise route to `text+ocr` and merge.
- [ ] **Step 4: Run the unit suite.** Expected: PASS. Then run both measurements and commit the JSON:
  - `routing-dev.json`: routing over the 51 dev PDFs. **Target:** 0 pages decided `ocr` (replace) for pages that have a usable text layer. Record every `text+ocr` page, with its reason.
  - `element-speed.json`: element mode against fast on the 51 dev PDFs. **Target:** element mode ≤ 2.5× fast's total seconds.

  Re-run `benchmarks.toc --elements` to confirm depth ≤ 2 micro F1 is still ≥ 0.703. Report misses as they are, with the profile.
- [ ] **Step 5: Commit** `perf(pdf): faster element mode; precise OCR routing`.

### Task E6: Heron ONNX layout engine

**Files:**
- Create: `src/docrooter/layout/__init__.py`, `layout/base.py`, `layout/heron.py`, `layout/registry.py`
- Modify:
  - `src/docrooter/backends/recipes.py`, or a new `layout/recipes.py`: the pinned heron repo id and revision, the file name, the input size, the class list and the normalisation, taken from the model card or config;
  - `src/docrooter/pdf/element_mode.py`: apply the regions;
  - `pyproject.toml`: the `layout` extra.
- Test: `tests/unit/layout/test_regions.py`, `tests/integration/test_heron.py` (marker `layout`)

**Interfaces:**
- Produces:
  - `LayoutRegion(page: int, box: Box, label: str, confidence: float)`;
  - `class LayoutEngine(Protocol): name: str; def detect(self, path, pages: Sequence[int], *, dpi: int) -> list[LayoutRegion]; def close(self) -> None`;
  - `HeronLayout(revision: str = <pinned>, min_confidence: float = 0.5)`;
  - `apply_regions(lines, regions, ...) -> (kind_by_line: dict[int, ElementKind], regions_used)`;
  - `HERON_TO_KIND`: the mapping from spec §6;
  - `get_layout_engine(name)`, `register_layout_engine(name, factory)`.

- [ ] **Step 1: Write the failing tests** with a fake engine returning fixed regions:
  - `test_region_kinds_override_heuristics`;
  - `test_overlap_resolution_confidence_then_area`;
  - `test_low_confidence_regions_ignored`;
  - `test_section_header_region_creates_section`;
  - `test_heading_candidate_inside_text_region_is_not_heading`;
  - `test_unruled_table_region_uses_text_strategy`;
  - `test_lines_outside_regions_use_heuristics`;
  - `test_missing_onnxruntime_raises_layout_extra_message`.
- [ ] **Step 2:** Run `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/layout -q`. Expected: FAIL.
- [ ] **Step 3:** Implement.
  - Download through `huggingface_hub.hf_hub_download(repo, filename, revision=<pinned sha>)`.
  - Create an `onnxruntime.InferenceSession` lazily with `CPUExecutionProvider` (CoreML optional, only if it runs without error).
  - Preprocess and post-process exactly as the model's config specifies. Verify on one generated page with a title, text and a table: the integration test must detect a Table region overlapping the drawn table at IoU ≥ 0.5.
- [ ] **Step 4:** Run the unit tests. Expected: PASS. Then run `uv pip install onnxruntime pypdfium2 huggingface_hub` and `PYTHONPATH=src .venv/bin/python -m pytest tests/integration/test_heron.py -m layout -q`. Expected: PASS. Record peak RSS and seconds per page on 5 dev pages under the memory guard (one model process).
- [ ] **Step 5:** Commit `feat(layout): optional heron ONNX layout engine`.

### Task E7: Summaries

**Files:**
- Create: `src/docrooter/summarize.py`
- Test: `tests/unit/test_summarize.py`

**Interfaces:**
- Consumes: `keyphrases.document_frequencies` and `keyphrases_from_frequencies`; `Document`, `Element`.
- Produces:
  - `split_sentences(text: str) -> list[str]`;
  - `Summary(key_sentences: list[str], keyphrases: list[str], method: Literal["lead","centroid"])`;
  - `summarize(doc: Document, *, method="centroid", k=3, keyphrases=8, element_min_sentences=4) -> dict[str, Summary]`, keyed by section id and by element id;
  - `table_description(el: Element) -> str` (`"Table: <caption or first header row>; <n_rows>×<n_cols>"`).

- [ ] **Step 1: Write the failing tests.**
  - `test_splitter_protects_abbreviations`, with sentences containing "Apple Inc. reported", "No. 5", "U.S. GAAP", "e.g. revenue", "Fig. 3 shows", "Smith et al. found", "J. P. Morgan", "3.14", "Section 1.2.3 applies" and "Wait... then". Each must stay intact.
  - `test_lead_takes_first_k`.
  - `test_centroid_prefers_central_sentence_and_dedups`: two near-duplicate central sentences give only one picked, and the picks come back in document order.
  - `test_centroid_heading_bonus`.
  - `test_length_penalty`.
  - `test_summary_coverage`: sections with ≥ 2 sentences and elements with ≥ 4.
  - `test_table_description_format`.
  - `test_deterministic`: two runs are identical.
- [ ] **Step 2:** Run `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_summarize.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement the scoring exactly as spec §8: centroid cosine, `+0.2 × heading similarity`, `+0.1 / (1 + index)`, `−0.3` for < 6 or > 80 words, and a redundancy cap of 0.7.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat: key sentences and key phrases per section and element`.

### Task E8: The rooter (`RouterConfig`, profiles, `process`, `Analysis`, escalation)

**Files:**
- Create: `src/docrooter/rooter/__init__.py`, `rooter/config.py`, `rooter/profiles.py`, `rooter/process.py`, `rooter/analysis.py`
- Modify: `src/docrooter/__init__.py` (exports), `pyproject.toml` (`tomli` marker)
- Test: `tests/unit/rooter/test_config.py`, `tests/unit/rooter/test_process.py`

**Interfaces:**
- Consumes: `load_pdf`/`load` (E1, E5b), `classify_document`, `normalize_sections`, `builtin_taxonomy("doc_types")`, `summarize` (E7), `embed` (E9, imported lazily), and the OCR and layout registries.
- Produces:
  - `RouterConfig` (pydantic, `extra="forbid"`) with fields `profile`, `text`, `ocr`, `layout`, `doc_type`, `sections`, `summaries`, `embeddings`, `escalate`, `escalate_below=0.5`, `max_pages=None`, `ocr_lang="en"` and `ocr_dpi=200`. Also `RouterConfig.from_file(path)` (TOML or JSON, by suffix) and `RouterConfig.resolve(profile=None, file=None, **overrides) -> RouterConfig` (precedence: overrides, then file, then profile).
  - `PROFILES: dict[str, dict]`, exactly spec §7.2.
  - `Analysis`, exactly spec §7.6, plus `to_json(include_embeddings=False)`.
  - `process(source, *, config=None, profile=None, on_event=None, **overrides) -> Analysis` and `process_many(sources, **same) -> Iterator[Analysis]`.

- [ ] **Step 1: Write the failing tests.**
  - `test_precedence_overrides_file_profile`.
  - `test_unknown_key_is_error`.
  - `test_toml_and_json_files`.
  - `test_profiles_match_spec`.
  - `test_fast_profile_matches_v03_functions`: the `process(p, profile="fast", doc_type="keyword", sections="keyword")` document, doc type and sections JSON equal `load` + `classify_document` + `normalize_sections` with `backend="keyword"`, and the document has no `routes`.
  - `test_balanced_has_elements_routes_summaries`, with the keyword backend.
  - `test_escalation_replaces_low_confidence_doc_type`, with fake backends: primary confidence 0.3, escalate result kept, both recorded in routes.
  - `test_escalation_changed_type_renormalises_all_sections`.
  - `test_no_escalation_above_threshold`.
  - `test_process_many_reuses_and_closes_backends`, with counting fakes: one load, one close.
  - `test_caller_owned_backend_not_closed`.
  - `test_accurate_without_jev_key_has_no_escalation`.
  - `test_ocr_auto_unavailable_route_warning`.
- [ ] **Step 2:** Run `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/rooter -q`. Expected: FAIL.
- [ ] **Step 3:** Implement. `process` emits a `route` event for each configuration decision, such as the resolved OCR engine. Timings go into `Analysis.timings`.
- [ ] **Step 4:** Run the tests and the full unit suite. Expected: PASS.
- [ ] **Step 5:** Commit `feat(rooter): profiles, config, process() and escalation`.

### Task E9: Chunks and embeddings

**Files:**
- Create: `src/docrooter/chunking.py`, `src/docrooter/embed.py`
- Modify: `rooter/process.py` (the `embeddings` task), `src/docrooter/__init__.py`
- Test: `tests/unit/test_chunking.py`, `tests/unit/test_embed.py`

**Interfaces:**
- Produces:
  - `Chunk(id, section_id, element_ids, text, heading_path: list[str], tokens: int, flags: list[str])`;
  - `chunk(source: Analysis | Document, *, max_tokens=512, overlap_elements=0, include_furniture=False) -> list[Chunk]`;
  - `EmbeddingSet(model: str, revision: str | None, dim: int, dtype: str, ids: list[str], vectors: np.ndarray)` with `to_npz(path)` and `from_npz(path)`;
  - `embed(source, backend: str | Backend, *, unit="element", dtype="float16", kinds=None) -> EmbeddingSet`. It uses the backend's existing `encode` path; check `backends/embedding.py` for the method name and use it, without adding a new model path.

- [ ] **Step 1: Write the failing tests.**
  - `test_chunks_never_cross_sections`.
  - `test_table_is_one_chunk_with_description`.
  - `test_oversize_element_flagged`.
  - `test_every_content_element_in_exactly_one_chunk`.
  - `test_heading_path_prefix_format`: `"A > B\n\n<text>"`.
  - `test_overlap_elements`.
  - `test_embed_records_model_and_revision`, with a fake embedding backend.
  - `test_float16_default`.
  - `test_npz_round_trip`.
  - `test_embed_kinds_filter`.
- [ ] **Step 2:** Run `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_chunking.py tests/unit/test_embed.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement. Tokens come from `views.estimate_tokens`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat: section-aware chunks and element embeddings`.

### Task E10: CLI and playground

**Files:**
- Modify: `src/docrooter/cli.py`, `src/docrooter/ui/server.py`, `src/docrooter/ui/static/{index.html,app.js,style.css}`
- Test: `tests/unit/test_cli_elements.py`, `tests/unit/ui/test_server.py` (extend)

**Interfaces:**
- Consumes: `process`, `RouterConfig`, `Analysis`, `chunk` and `load_pdf(elements=True)`.
- Produces:
  - `docrooter elements FILE [--profile balanced] [--config PATH] [--json] [--kinds k1,k2]`;
  - `docrooter process FILE [--profile] [--config] [--out PATH] [--chunks]`;
  - `docrooter config show [--profile]`;
  - server: a `profile` form field (default `balanced`); the result gains `elements` (capped by `MAX_TOC_ENTRIES`, with `truncated`), `summaries` (key sentences per TOC entry) and `routes` (per-page counts);
  - a `route` stage in the events.

- [ ] **Step 1: Write the failing tests.**
  - `test_elements_cli_table_and_json`.
  - `test_process_cli_writes_analysis_json`.
  - `test_config_show_resolves_profile`.
  - `test_server_profile_field_and_elements_payload` (keyword backend, generated PDF).
  - `test_server_elements_capped`.
  - `test_server_rejects_unknown_profile_400`.
  - `test_app_js_uses_textcontent_only`: a static check that app.js has no `innerHTML` assignments; the existing test may cover this, so extend it.
- [ ] **Step 2:** Run `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_cli_elements.py tests/unit/ui -q`. Expected: FAIL.
- [ ] **Step 3:** Implement. The UI adds:
  - a profile select;
  - an Elements tab with a list grouped by page: a kind chip, an OCR badge, the first 200 characters, and an expandable summary;
  - key sentences under each TOC entry;
  - a route row in the waterfall.

  Keep the existing visual language. All text goes in through `textContent`.
- [ ] **Step 4:** Run the tests. Expected: PASS. Then start `docrooter serve --backend keyword --port 18950`, upload `benchmarks/corpus/pdfs/doc_027.pdf` and `doc_045.pdf` with the in-app browser or Playwright, and screenshot the Elements tab into the task report. Use port 18950, never 8765.
- [ ] **Step 5:** Commit `feat(cli,ui): elements and process commands, playground Elements tab`.

### Task E11a: DocLayNet element benchmark and threshold tuning

**Files:**
- Create: `benchmarks/doclaynet.py` (fetch and sample), `benchmarks/elements_bench.py` (scoring)
- Modify: `benchmarks/.gitignore` or the root one, so the DocLayNet cache is ignored
- Create: `docs/benchmarks-elements.md`
- Test: `tests/unit/test_bench_elements.py` (matching and scoring on synthetic boxes)

**Interfaces:**
- Consumes: `load_pdf(elements=True)`, the E5a threshold constants and `HeronLayout`.
- Produces:
  - `match_boxes(pred: list[(Box, kind)], gold: list[(Box, cls)], iou=0.5) -> list[pairs]`, greedy by IoU;
  - `score(pages) -> {per_class, macro, per_category, text_coverage}`;
  - results go to `benchmarks/results/<date>/elements-doclaynet.json`.

- [ ] **Step 1: Write the failing tests.**
  - `test_greedy_one_to_one_matching`.
  - `test_class_mapping_per_spec_table`, including `page_number` mapping to Page-footer and `contents` to Text.
  - `test_text_coverage_score`.
- [ ] **Step 2:** Run `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_bench_elements.py -q`. Expected: FAIL. Implement, then run again. Expected: PASS.
- [ ] **Step 3:** Pick the DocLayNet source. It must include PDF pages, or text cells to rebuild them, plus the boxes, and be reproducible and checksum-pinned. Record the source URL, licence (CDLA-Permissive-1.0) and checksums in `benchmarks/doclaynet.py`. Sample 400 test pages and up to 400 train pages, stratified over the 6 categories, with a fixed seed.
- [ ] **Step 4: Tune** the E5a constants on train pages only. Log every change and its train macro F1 in `docs/benchmarks-elements.md` (a tuning log, as in `docs/benchmarks-pdf.md`). Keep a change only if train macro F1 rises. After tuning, re-run the dev TOC comparison (E5b Step 4) to confirm no regression.
- [ ] **Step 5: Score the test pages once,** for heuristic-pymupdf, heuristic-pdfplumber and heron (under the memory guard). Write per-class and per-category tables, and report the spec §10.1 target as met or missed.
- [ ] **Step 6:** Commit `bench: DocLayNet element benchmark and tuned element thresholds`.

### Task E11b: OCR, routing, reading-order, doc-type and speed measurements

**Files:**
- Create: `benchmarks/ocr_bench.py`
- Modify: `benchmarks/elements_bench.py` (reading order), `benchmarks/pdf_bench.py` (`--profile balanced`), `benchmarks/profile_pipeline.py` (`--elements`), `docs/benchmarks-elements.md`, `docs/performance.md`
- Test: `tests/unit/test_bench_ocr.py` (CER/WER normalisation)

- [ ] **Step 1: Write the failing tests.**
  - `test_cer_normalisation`: NFKC, collapsed whitespace, de-hyphenated line ends.
  - `test_wer`.
  - `test_routing_confusion_counts`.

  Run them: FAIL. Implement, then run: PASS.
- [ ] **Step 2: OCR.** Render 60 pages from 20 dev PDFs (fixed seed, 200 dpi) into image-only PDFs under `benchmarks/.cache/ocr/`. For each installed engine (ocrmac, rapidocr, tesseract), record CER, WER, seconds per page and peak RSS. Report routing confusion on the 51 dev PDFs and the 60 image pages. Record element-kind agreement between text-layer and OCR elements on the same pages. Results go in `benchmarks/results/<date>/ocr.json`.
- [ ] **Step 3: Reading order.** Compute bookmark-pair order agreement on the dev bookmark documents, `fast` vs element mode, and Kendall τ on the two-column dev papers. Write a checksum check that single-column dev pages' line order is unchanged. Results go in `elements-reading-order.json`.
- [ ] **Step 4: Doc type in `balanced`.** Run `ultrabert-view` and `ultrabert-hybrid` on the dev corpus with `--profile balanced`, under the memory guard, against the committed `fast` numbers. Report the difference in `docs/benchmarks-pdf.md`, in a v0.4 section.
- [ ] **Step 5: Speed and memory.** Measure seconds per 100 pages for element mode (heuristic pymupdf, heuristic pdfplumber, heron) on the Q5 profiling set, and `fast` against v0.3 (within 5%). Write to `docs/performance.md`.
- [ ] **Step 6:** Commit `bench: OCR, routing, reading order, balanced doc type and element speed`.

### Task E12: Docs, packaging and release notes

**Files:**
- Create: `docs/site/concepts/elements.md`, `docs/site/concepts/rooter.md`, `docs/site/guides/ocr-and-layout.md`, `docs/site/guides/chunking-for-rag.md`, `examples/elements_and_chunks.py`
- Modify: `mkdocs.yml` (nav), `docs/gen_cli.py` output (regenerate), `README.md` (a short "Elements and profiles" section with one example), `CHANGELOG.md` (Unreleased), `tests/unit/test_examples.py` (run the new example on a generated PDF)

- [ ] **Step 1:** Write the pages. Every code sample comes from `examples/` or is covered by `test_docs_snippets.py`, and every number traces to a committed results file.
- [ ] **Step 2: Run the checks:**
  - `PYTHONPATH=src .venv/bin/python -m pytest tests/unit -q`
  - `.venv/bin/ruff check .`
  - `.venv/bin/ruff format --check .`
  - `PYTHONPATH=src .venv/bin/mypy src/docrooter`
  - `.venv/bin/mkdocs build --strict`

  Expected: all clean.
- [ ] **Step 3: Wheel smoke.** Run `uv build`. In a fresh venv, install the wheel with `[pdf]` and run `docrooter elements` on a generated two-column PDF. Install `[ocr]` and run `docrooter process` on a generated scanned PDF with `--profile balanced`. Check `unzip -l` for the new packages.
- [ ] **Step 4:** Commit `docs: elements, the rooter, OCR and chunking guides`.
