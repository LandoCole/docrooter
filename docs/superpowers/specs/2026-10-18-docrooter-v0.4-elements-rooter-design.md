# docrooter v0.4: elements and the rooter

- **Date:** 2026-10-18
- **Status:** Approved under the user's standing instruction of 2026-10-18 ("write up the program plan … then the next spec and implementation plan … lean on your own judgement and research … keep looping until the more capable docrooter is complete").
- **Program:** `docs/superpowers/program/2026-10-18-docrooter-element-intelligence-program.md`, phase v0.4.
- **Research:** `docs/superpowers/research/2026-10-18-element-classification-research.md`, §1, §2 and §9.
- **Builds on:** v0.1–v0.3 specs, which all still apply.

## 1. Intent

Today a document becomes headings and section text. v0.4 makes every visible part of a document an **element**: a paragraph, list item, table, figure, caption, footnote, heading or piece of page furniture. Each element has a place on the page, in reading order and in the section tree.

It also adds the **rooter**: one configurable pipeline. Per page, the rooter decides whether to read the text layer or OCR the page. Per task, it decides which engine or backend runs. It ships three profiles (`fast`, `balanced`, `accurate`) and records why it made each decision.

This is the foundation for the v0.5 learned element tagger and the v0.6 completeness dial. v0.4 itself does not classify elements semantically, beyond inheriting their section's label.

## 2. Goals

1. **Element model.** An ordered list of typed elements per document, from PDFs (born-digital and scanned) and from Markdown.
2. **Reading order.** Multi-column pages read column by column.
3. **Rooter.** Text-versus-OCR decisions per page, engine choice per task, three profiles, user configuration from Python or a TOML/JSON file, a route log, and confidence-based escalation for document type and section labels.
4. **OCR** as optional engines behind one protocol, feeding the same element builder.
5. **Optional ML layout engine:** Docling's Apache-2.0 heron model, through its ONNX export on onnxruntime, so it stays torch-free.
6. **Key sentences and key phrases** per section and per long element. Torch-free.
7. **Section-aware chunks and opt-in element embeddings**, indexed as `heading path + element text`.
8. **Surfaces:** a `process()` API, `docrooter elements` and `docrooter process` CLI commands, and a playground Elements tab with routing shown on the timeline.
9. **Measured:** element-kind accuracy on DocLayNet, reading order, OCR character error rate (CER) and speed, routing accuracy, and no regression of the dev TOC benchmark.

## 3. Non-goals and anti-goals

- **Non-goals:**
  - semantic element tags beyond section inheritance (v0.5);
  - selection and the dial (v0.6);
  - reading values out of charts (a later optional vision-language-model add-on);
  - handwriting;
  - non-Latin OCR tuning;
  - table cell structure beyond rows and columns (no spanning-cell recovery; Docling's TableFormer is a later optional extra);
  - HTML input.
- **Anti-goals:**
  - The `fast` path, meaning today's `load()` / `load_pdf()` / `classify_document()`, must not get slower or change its output.
  - Elements must not make the core import torch, onnxruntime or any OCR engine.
  - No GPL/AGPL or non-commercial dependency may enter a default install. The ML layout default must not be PyMuPDF Layout (AGPL), DocLayout-YOLO (AGPL) or Surya weights (revenue cap).

## 4. Element model (`docrooter.elements`)

```python
ElementKind = Literal[
    "title", "heading", "paragraph", "list_item", "table", "figure", "caption",
    "footnote", "page_header", "page_footer", "page_number", "contents", "formula",
]

class Box(BaseModel):        # points, origin top-left, as Line
    page: int                # 1-based
    x0: float; y0: float; x1: float; y1: float

class TableData(BaseModel):
    rows: list[list[str]]    # cell text, row-major; ragged rows padded with ""
    header_rows: int = 0
    def to_markdown(self) -> str: ...

class FigureData(BaseModel):
    source: Literal["image", "vector", "ocr_region", "layout_model"]
    text: str = ""           # labels and other text found inside the figure region

class Element(BaseModel):
    id: str                  # "e0", "e1", … in reading order; stable for the same input and config
    kind: ElementKind
    text: str                # normalised text (tables: cells joined row by row; figures: caption-free label text)
    boxes: list[Box]         # one box per page fragment; a paragraph continued across a page break has two
    order: int               # global reading-order index
    section_id: str | None   # the Section this element belongs to (None before the first heading)
    level: int | None = None # heading level (headings and title only)
    list_depth: int | None = None
    marker: str | None = None          # list bullet or enumerator, footnote marker
    table: TableData | None = None
    figure: FigureData | None = None
    caption_for: str | None = None     # caption → the figure or table element id
    caption_id: str | None = None      # figure or table → its caption element id
    source: Literal["text", "ocr"] = "text"
    confidence: float | None = None    # OCR or layout-model confidence; None for text-layer heuristics
    flags: list[str] = []              # e.g. "continued", "low_confidence", "from_layout_model"
```

**Changes to existing models.** All are additive and default to empty, so v0.3 serialisation round-trips unchanged:
- `Document.elements: list[Element]`.
- `Section.element_ids: list[str]`.
- `Document.meta["routes"]`, holding the route log (§7.4).

**Kinds**, and how they relate to today's heuristics:
- `page_header` / `page_footer` / `page_number`: the running lines and page numbers that `pdf/signals.py` already detects.
- `contents`: lines of a printed table-of-contents page.
- `heading` / `title`: the existing heading detector's output, unchanged.

**Mapping to DocLayNet's 11 classes**, for evaluation:

| DocLayNet | docrooter |
| - | - |
| Caption | caption |
| Footnote | footnote |
| Formula | formula |
| List-item | list_item |
| Page-footer | page_footer, page_number |
| Page-header | page_header |
| Picture | figure |
| Section-header | heading |
| Table | table |
| Text | paragraph |
| Title | title |

`contents` is scored as Text, which is how DocLayNet labels contents lines.

**Markdown** produces elements too:
- ATX and setext headings → `heading`;
- paragraphs → `paragraph`;
- `-`, `*`, `+` and `1.` items → `list_item`, with depth from indentation;
- pipe tables → `table`;
- fenced code → `paragraph` with the flag `code`;
- `$$` blocks → `formula`;
- images `![alt](…)` → `figure`, with the alt text as its text.

Boxes are empty for Markdown, so `boxes` may be `[]`.

## 5. Building elements from PDFs (`docrooter.pdf.elements`)

### 5.1 Inputs

The element builder works on engine-neutral page objects, so pymupdf and pdfplumber produce the same element kinds:

```python
class PageObjects(BaseModel):
    page: int; width: float; height: float
    lines: list[Line]                 # existing Line model (pdf/lines.py), already has bboxes
    images: list[Box]                 # raster image placements
    drawings: list[Box]               # clustered vector-graphics regions (pymupdf cluster_drawings; pdfplumber: clustered rects/lines/curves)
    tables: list[tuple[Box, list[list[str]]]]   # ruled tables found by the engine's "lines" strategy
```

- **pymupdf:** `get_text("dict")` with image blocks preserved, `cluster_drawings()` and `find_tables(strategy="lines")`.
- **pdfplumber:** `page.images`, rects, lines and curves clustered by proximity, and `find_tables({"vertical_strategy": "lines", "horizontal_strategy": "lines"})`.
- **Cost:** the object pass runs only when elements are requested. `load_pdf()` without elements does exactly what it does today.

### 5.2 Reading order (recursive XY-cut)

Per page, after page furniture is removed:
1. **Full-width separators.** Elements spanning ≥ 75% of the text width (titles, wide tables and figures, full-width headings) split the page into horizontal bands.
2. **Column cuts.** Inside each band, recursive XY-cut finds vertical gutters. A gutter is a whitespace valley at least `max(8 pt, 3 × median character width)` wide, covering ≥ 60% of the band's height. The band is then cut horizontally at gaps ≥ `1.5 × median line gap`.
3. **Ordering.** Leaf cells are ordered top-to-bottom, then left-to-right.

The order applies to elements. It applies to TOC heading detection only if the dev TOC benchmark shows no regression (§10). The plan task decides this and records the numbers either way. Single-column pages must come out in exactly today's order.

### 5.3 Element typing rules (heuristic engine, the default)

These are applied in this order. Each line is consumed by at most one element.

1. **Furniture:** lines the signals mark as running headers, footers or page numbers become `page_header` / `page_footer` / `page_number`, split by top or bottom band.
2. **Contents:** lines of a high-confidence printed-contents page become one `contents` element per page.
3. **Tables:** each engine-detected ruled table becomes a `table` element with `TableData`, and lines inside its box are consumed. Unruled tables are not detected by default; they are detected only by the layout model (§6).
4. **Figures:**
   - image placements ≥ 2% of the page area; and
   - drawing clusters ≥ 2% of the page area that are not table rulings, whose text-line density is < 0.5 lines per 1,000 pt²;

   each become a `figure`. Text lines inside the figure box are consumed into `FigureData.text`.
5. **Captions:** a line or short paragraph matching `^(Figure|Fig\.|Table|Exhibit|Chart|Graph|Diagram)\s*[\dIVXLC]+[.:—–-]?` (case-insensitive), within 1.5 line heights above or below a figure or table and horizontally overlapping it, becomes a `caption` linked both ways. A table caption may sit above its table.
6. **Headings and title:** the existing heading detector's kept candidates. The document title, if detected on page 1, becomes `title`.
7. **Footnotes:** in the bottom 25% of the page, size ≤ `0.9 × body`, starting with a marker (digits, `*`, `†`, `‡`, `§`, or a superscript span on pymupdf) → `footnote`. Consecutive footnote lines merge until the next marker.
8. **Lists:** a line starting with a bullet (`•`, `◦`, `▪`, `-`, `–`, `*`) or an enumerator (`1.`, `1)`, `(a)`, `a.`, `i.`, `(iv)`) followed by text → `list_item`. Continuation lines with a hanging indent join it. `list_depth` comes from the indent, in steps of ≥ 8 pt.
9. **Formula:** a line with ≥ 30% math symbols (`=`, `∑`, `∫`, `±`, Greek letters, superscript or subscript digits), or set in a math font family (names containing `Math`, `CMMI`, `CMSY`, `Symbol`), standing alone → `formula`.
10. **Paragraphs:** all remaining lines, grouped in reading order. A new paragraph starts at any of:
    - a vertical gap > `1.5 ×` the median line gap of the region;
    - a size change > 0.5 pt;
    - a first-line indent of ≥ 8 pt after a line that ends a sentence;
    - a column or band boundary.

    A paragraph continues across a column, band or page boundary when its last line has no terminal punctuation (`.`, `!`, `?`, `:`, `;`, closing quote or bracket after punctuation) and the next fragment starts lowercase. The fragments become one element with two boxes and the flag `continued`.

**One source of truth: in element mode, the TOC is built from elements.** Ruling, after the builder check. In element mode (`load_pdf(elements=True)`, `balanced`, `accurate`), the pipeline is:
1. route pages (§7.4);
2. replace OCR pages' lines;
3. order lines by XY-cut (§5.2);
4. compute signals and run the heading detector on that ordered line list;
5. type all elements;
6. build the section tree from the heading elements, with the existing level assignment and bookmark overrides.

Then:
- Each element's `section_id` is the section of the most recent heading element in reading order, or `None` before the first one. `None` elements go to the preamble section `s0` if one exists.
- `Section.element_ids` lists a section's own elements, not its descendants'.
- `Section.text` is the section's element texts joined in order. It includes paragraphs, list items, table markdown, captions, footnotes and formulas. It excludes furniture and `contents`.

Sections, summaries and chunks are therefore always consistent with elements. When the heron engine (§6) labels a region Section-header, that becomes a heading element and therefore a Section. A heading candidate inside a region that heron labels Text or Table is not a heading. The `fast` path keeps today's line order, signals and section text exactly. Element mode is allowed to differ from `fast`, and the difference is measured (§10.3).

**Every threshold above is a starting estimate**, tuned on DocLayNet's train split only (§10.1). The final values and the tuning log go into `docs/benchmarks-elements.md`.

### 5.4 OCR-sourced pages

OCR returns lines with boxes and confidence. They become `Line` objects:
- `size` = box height × 0.75 (cap height to font size);
- `bold=False`, `font="ocr"`;
- `source="ocr"` carried through to the elements.

The same builder runs on them. Furniture detection, captions, lists and paragraphs work unchanged. Headings come from size and isolation only.

**OCR feeds everything in element mode.** Ruling. For a page routed to whole-page OCR, its text-layer lines are removed from the document-wide line list and its OCR lines are inserted in page order before signals and headings are computed. The section tree, section text, classification views and elements all see OCR text. This also applies to garbled born-digital pages, whose garbled text is never kept. Region OCR (§7.4) only fills figure text. A scanned PDF in `balanced` therefore no longer yields one empty `s0`. If OCR is unavailable, it yields the same `no_text_layer` result as today, plus the route log.

## 6. Optional ML layout engine (`docrooter[layout]`)

- **Model:** `docling-project/docling-layout-heron-onnx`, Apache-2.0, pinned to a revision in recipes. It runs on onnxruntime, with pages rendered by `pypdfium2` (Apache-2.0/BSD) at the model's input size. There is no torch and no remote code: ONNX weights are data, so the trust gate does not apply. The download goes through `huggingface_hub` with a pinned revision, like the other recipes.
- **Use:** the detected boxes, with class and confidence, become regions. Each region collects the text-layer lines whose centres fall inside it, or the OCR lines on OCR pages, and the region's class sets the element kind. This overrides heuristics 3–10 for lines inside a region. Lines outside every region fall back to the heuristics.
  - Regions with ≥ 0.5 confidence are used.
  - Overlapping regions resolve by higher confidence, then smaller area.
  - Heron's extra classes map as follows: Document Index → `contents`; Code → `paragraph` with the flag `code`; Checkbox, Form and Key-Value Region → `paragraph` with the flag `form`.
- **Reading order:** XY-cut over the region boxes.
- **Unruled tables:** a Table region with no engine-detected table is extracted with the engine's "text" table strategy, restricted to the region.
- **Memory and speed:** one model process. The session is created lazily, reused, and released by `close()`. Peak memory and seconds per page are measured and recorded (§10). The model runs only in the `accurate` profile, or when the user selects `layout="heron"`.
- **Fallback:** if `onnxruntime` or the weights are missing, raise the standard missing-extra error. Never fall back silently in a profile that asked for it.

## 7. The rooter (`docrooter.rooter`)

### 7.1 Tasks and engines

| Task | Engines (built-in names) | Default (`balanced`) |
| - | - | - |
| `text` | `pymupdf`, `pdfplumber`, `both`, `auto` (as today) | `auto` |
| `ocr` | `none`, `auto`, `ocrmac`, `rapidocr`, `tesseract` | `auto`: `ocrmac` on macOS if installed, else `rapidocr` if installed, else `tesseract` if the binary is on PATH, else `none` with a route warning |
| `layout` | `heuristic`, `heron` | `heuristic` |
| `doc_type` | any backend spec (`keyword`, `embedding:…`, `nli:…`, `openai:…`, `jev:…`, a trained bundle path) or `none` | `embedding:dmis-lab/UltraBERT-Embed-150M` |
| `sections` | any backend spec, or `none` | same as `doc_type` (normalisation runs if the type maps to a section taxonomy) |
| `summaries` | `none`, `lead`, `centroid` | `centroid` |
| `embeddings` | `none` or an embedding backend spec | `none` |
| `escalate` | `none` or a backend spec, with `escalate_below: float` | `none` |

Engines are looked up in a registry. Users can register their own:
- an OCR engine with `register_ocr_engine(name, factory)`;
- a layout engine with `register_layout_engine(name, factory)`.

Backends use the existing backend registry.

### 7.2 Profiles

```text
fast:      text=auto  ocr=none  layout=none(no elements)  doc_type=<default>  sections=<default>  summaries=none  embeddings=none
balanced:  text=auto  ocr=auto  layout=heuristic          doc_type=<default>  sections=<default>  summaries=centroid embeddings=none
accurate:  text=auto  ocr=auto  layout=heron              doc_type=<default>  sections=<default>  summaries=centroid embeddings=none
           escalate=jev:jev-1.13.0 if TYPESAFE_API_KEY is set, else none; escalate_below=0.5
```

`fast` reproduces today's `load()` + `classify_document()` + `normalize_sections()` exactly. Its outputs are byte-identical to calling those functions with the same arguments, and a test asserts this.

### 7.3 Configuration

- `RouterConfig` (pydantic) holds the task → engine mapping, profile, `max_pages`, the OCR language (`"en"` default), DPI for OCR rendering (200), and the escalation threshold.
- It can be built in Python (`RouterConfig(profile="balanced", ocr="rapidocr")`), loaded from TOML or JSON (`RouterConfig.from_file(path)`), or overridden per call. Precedence: call arguments, then the file, then the profile defaults.
- Unknown keys are errors.
- The CLI gains `--profile` and `--config` on `process` and `elements`.

### 7.4 Page routing (text versus OCR)

For each page, compute from the text engine's objects:
- `char_density`: printable characters per square inch of page.
- `image_cov`: union area of image placements over page area.
- `garbled`: the share of characters that are any of:
  - U+FFFD;
  - pdfminer `(cid:N)`;
  - Private Use Area characters;
  - C0/C1 controls.

  A page also counts as garbled if > 50% of ≥ 50 characters are non-alphanumeric and non-space, or if < 50% of its ≥ 20 tokens are dictionary-like. "Dictionary-like" means alphabetic with a vowel, or numeric. No word list is shipped.
- `invisible_text`: the share of characters in render mode 3. pymupdf only; 0 on pdfplumber.

Decide, in this order:
1. **Whole-page OCR** if any of:
   - `image_cov ≥ 0.75` and `char_density < 2`;
   - `garbled > 0.10`;
   - `invisible_text > 0.5` and the invisible text is garbled.
2. **Region OCR** for each image placement ≥ 5% of the page with no text-layer characters inside it. The result becomes that figure's `FigureData.text` with `source="ocr_region"`. It is never paragraph text.
3. **Otherwise**, use the text layer.

When `ocr=none` (or no engine is available) and a page needed OCR, it is kept as is. The route log records `needs_ocr` and the reason, and the result flags the document `ocr_skipped`.

**Where routing runs.** Ruling.
- **`fast`:** routing never runs. There is no object pass, no `meta["routes"]` key and no OCR, exactly as today.
- **`load_pdf(elements=True, ocr=None)`:** computes page metrics and decisions and writes the route log, but calls an OCR engine only when one is passed (`ocr="rapidocr"`, an engine instance, …).
- **`process()`:** passes the profile's OCR engine.
- **Code layout:** metric computation and the decision rules are pure functions in `docrooter.pdf.routing`. Engine selection and profiles live in `docrooter.rooter`.

**Route log.** Every decision is a `RouteDecision(scope, target, task, engine, reason, metrics)` in `Document.meta["routes"]`:
- `scope` is `"document"`, `"page"` or `"region"`.
- `target` is a page number or a region box.

It is also emitted as an event: stage `route`, status `progress`.

### 7.5 Escalation

- **Trigger:** after `doc_type` (and `sections`) classification, if the result's confidence is below `escalate_below` and an `escalate` backend is set, the same view is re-classified by the escalation backend.
- **Which result is kept:** the escalated result, whenever escalation fires. Ruling: confidences from different backends are not calibrated against each other, so they are not compared. The escalation backend is configured as the stronger one.
- **Recorded:** both results and the reason go in the route log.
- **Section labels:** if escalation changes the document type, and with it the section taxonomy, all sections are re-normalised with the new taxonomy by the primary `sections` backend first. Then, either way, sections still below the threshold are re-labelled by the escalation backend in one batch.

### 7.6 Entry point

```python
def process(source: str | Path, *, config: RouterConfig | None = None, profile: str | None = None,
            on_event: EventSink | None = None, **overrides) -> Analysis

class Analysis(BaseModel):
    document: Document               # with elements (except in fast) and meta["routes"]
    doc_type: DocTypeResult | None
    sections: NormalizationResult | None
    summaries: dict[str, Summary]    # keyed by section id and by element id
    embeddings: EmbeddingSet | None
    config: RouterConfig             # the resolved configuration actually used
    timings: dict[str, float]        # seconds per stage
    def to_json(self, *, include_embeddings: bool = False) -> str: ...
```

- **Backends:** loaded once per `process()` call and closed at the end, unless a `Classifier` or backend instance is passed in through `overrides`. In that case the caller owns it, as today.
- **`process_many(paths, …)`:** reuses one set of engines and backends across documents, closes them at the end, and yields results one at a time. It never holds all of them in memory.

## 8. Summaries (`docrooter.summarize`)

- **Sentence splitter:** a regex splitter that protects common abbreviations ("Inc.", "Co.", "No.", "U.S.", "e.g.", "i.e.", "Fig.", "et al.", initials, decimals and section numbers such as "1.2.3"), numbered list markers and ellipses. It has unit tests on financial, legal and scientific samples.
- **Methods:**
  - `lead`: the first `k` sentences.
  - `centroid`: score each sentence by:
    - TF-IDF cosine to the section centroid (document frequencies taken over the document's sections, as `keyphrases` already does);
    - `+ 0.2 ×` similarity to the heading path (the GoSum idea, done cheaply);
    - `+ 0.1 ×` a position prior `1 / (1 + index)`;
    - `− 0.3` if the sentence is < 6 words or > 80 words.

    Pick the top `k` (default 3) without redundancy: skip a sentence whose cosine to an already picked one is > 0.7. Return them in document order.
- **Key phrases:** the existing `keyphrases_from_frequencies`, top 8.
- **Output:** `Summary(key_sentences: list[str], keyphrases: list[str], method: str)`.
- **Coverage:** every section with ≥ 2 sentences, plus every `paragraph` / `list_item` element with ≥ 4 sentences (`summarize_elements_min_sentences`, default 4). Tables get a one-line description: `"Table: <caption or first header row>; <n_rows>×<n_cols>"`.

## 9. Chunks and embeddings

- **`chunk(analysis_or_document, *, max_tokens=512, overlap_elements=0, include_furniture=False)`** returns a list of `Chunk(id, section_id, element_ids, text, heading_path, tokens)`.
  - Chunks never cross a section boundary and never split an element. A single element longer than `max_tokens` becomes its own chunk, flagged `oversize`.
  - A table always becomes one chunk, whose text is its description line plus its markdown.
  - The text is `heading path` (joined with " > ") + blank line + element texts. Tokens use `views.estimate_tokens`.
  - Furniture and `contents` are skipped unless asked for.
- **`embed(analysis, backend, *, unit="element"|"chunk", dtype="float16"|"float32", kinds=None)`** returns `EmbeddingSet(model, revision, dim, dtype, ids, vectors)`.
  - It needs the `models` extra.
  - The text embedded is the same `heading path + text` form.
  - It records the backend's model and revision, so callers can embed queries consistently.
  - Vectors go in `Analysis.embeddings` only when the `embeddings` task is set. `to_json()` leaves them out unless asked.

## 10. Measurement (results under `benchmarks/results/<date>/`, write-ups in `docs/benchmarks-elements.md`)

### 10.1 Element kinds on DocLayNet
- **Data:** DocLayNet (CDLA-Permissive-1.0). The full download is too large, so the plan picks the smallest official or mirrored subset that includes born-digital PDF pages, or the text cells needed to rebuild them, plus the gold boxes. Criteria: reproducible, licence preserved, pinned by checksum.
  - **Sample:** 400 test-split pages stratified over the six categories (financial reports, manuals, scientific articles, laws and regulations, government tenders, patents).
  - **Tuning:** a disjoint set of up to 400 train-split pages for threshold tuning. Thresholds are tuned on train pages only; test pages are scored once, at the end.
- **Metric:** greedy one-to-one matching of predicted to gold boxes on the same page, at IoU ≥ 0.5, with classes mapped by §4. Report:
  - per-class precision, recall and F1, and macro F1 over the mapped classes;
  - per-category macro F1;
  - also an IoU-free text-coverage score per class: the share of gold-box text characters that land in a predicted element of the right kind. Box conventions differ between line-merged predictions and DocLayNet's region boxes, so both views are given.
- **Engines:** heuristic (pymupdf and pdfplumber) and heron.
- **Target:** heuristic F1 ≥ 0.8 for Text, Table and Picture on the financial-report and scientific-article categories. Report as is if missed.

### 10.2 Reading order
- **Synthetic:** unit tests with generated two- and three-column PDFs and a full-width title and figure. Exact order is required.
- **Dev corpus:** for the documents whose bookmark order is known, the share of consecutive bookmark pairs whose matched headings appear in the same order. Before vs after XY-cut.
- **Real two-column papers in the dev set:** Kendall τ of heading order vs bookmarks.
- **Hard constraint:** pages that were single-column must be unchanged, verified by checksum of the line order on all 51 dev documents.

### 10.3 TOC no-regression
- The dev TOC benchmark (`benchmarks/toc.py`) on `pymupdf` and `pdfplumber`, with elements and reading order on and off.
- Element mode builds the TOC from elements (§5.3), so this compares element mode's TOC against `fast` on the 24 bookmark documents.
- **Target:** depth ≤ 2 micro F1 in element mode is no lower than `fast`'s 0.703 (pymupdf).
- **If it is lower:** fix the element rules on dev until it is not, or report the gap. The two modes never silently diverge on single-column documents.
- **Document type:** also re-run the dev doc-type benchmark for `ultrabert-view` and `ultrabert-hybrid` in `balanced` against `fast`, and report the difference.

### 10.4 OCR
- **Data:** 60 pages rendered at 200 dpi from 20 dev-corpus PDFs, into image-only PDFs. Ground truth is the text layer of the same page.
- **Metrics:** CER and WER after normalisation (NFKC, collapsed whitespace, de-hyphenated line ends), seconds per page, and peak memory. Engines: ocrmac, rapidocr, tesseract, whichever are installed.
- **Routing accuracy:** on the 51 dev PDFs (expected: ~0 OCR pages) and on the 60 image-only pages (expected: all OCR), report the confusion counts.
- Also element-kind agreement between text-layer and OCR-sourced elements on the same pages.

### 10.5 Speed and memory
- Seconds per 100 pages for elements on the profiling set (the same 10 PDFs as Q5), for heuristic pymupdf, heuristic pdfplumber and heron.
- Peak memory for heron and OCR.
- `fast` must be no slower than v0.3's `load` (within 5%).

**Held-out:** the frozen held-out corpus is not used in v0.4. The program's final evaluation (v0.7) runs it once.

## 11. Surfaces

- **API:** `process`, `process_many`, `RouterConfig`, `Analysis`, `Element`, `chunk`, `embed`, `summarize`, `register_ocr_engine` and `register_layout_engine` are exported from `docrooter`. `load_pdf(..., elements=True)` and `load(..., elements=True)` build elements without classification.
- **CLI:**
  - `docrooter elements FILE [--profile] [--config] [--json] [--kinds …]` prints elements as a table: order, page, kind, the first 80 characters, and the section.
  - `docrooter process FILE [--profile] [--config] [--out analysis.json] [--chunks]`.
  - `docrooter config show [--profile]` prints the resolved configuration.
- **Playground:**
  - a profile selector;
  - an **Elements** tab: the list by page, with kind chips, OCR badges and expandable summaries;
  - route decisions on the timeline, as a "route" stage with per-page counts;
  - summaries shown under each TOC entry.

  Existing caps still apply: 5,000 entries, 1 MB of text, 50 MB of PDF, 2,000 pages.
- **Docs:** a concepts page each for elements and the rooter, a guide each for OCR and layout extras and for chunking for RAG, CLI reference regeneration, and a CHANGELOG entry.

## 12. Packaging

- **New extras:**
  - `layout = ["onnxruntime>=1.18", "pypdfium2>=4.30", "huggingface_hub>=0.23"]`;
  - `ocr = ["rapidocr>=3.0", "onnxruntime>=1.18", "pypdfium2>=4.30"]`;
  - `ocr-mac = ["ocrmac>=1.0; sys_platform == 'darwin'"]`.
  - Tesseract uses the system binary through `subprocess` (TSV output), so it needs no pip dependency; a missing binary is reported clearly.
  - `all` includes `layout`, `ocr` and `ocr-mac`.
- **Rendering pages for OCR:** with pymupdf when the `pdf` extra is installed (`Page.get_pixmap`); otherwise `pypdfium2`. The MIT-only path (`pdf-mit` + `ocr`) never imports pymupdf.
- All new imports are lazy. Missing-extra errors use the existing `require()` helper and message format.

## 13. Testing

- **Unit tests stay torch-free and onnxruntime-free.** Element rules are tested on synthetic `PageObjects` and `Line` lists, and on tiny generated PDFs (pymupdf, skipped if absent): two-column, tables, figures with captions, footnotes, lists, a paragraph continued across pages, and a garbled-font page for routing.
- **OCR and heron:** adapters are tested with fakes (an engine returning fixed boxes). Real-engine tests are integration tests marked `ocr` and `layout`, outside the default suite.
- **Rooter:**
  - config precedence and file parsing;
  - profile resolution;
  - `fast` byte-identity with the v0.3 functions;
  - the route log;
  - escalation with fake backends;
  - `process_many` releasing backends.
- **Summaries:** splitter cases, deterministic centroid output, redundancy filtering.
- **Chunk invariants:** no cross-section chunks, tables whole, every non-furniture element in exactly one chunk.
- **Coverage gate:** ≥ 80% on the new packages.

## 14. Left to the builder

- Exact data structures inside the builder.
- TOML parsing: `tomllib` on Python ≥ 3.11, and `tomli` (a tiny core dependency, `python_version < '3.11'`) on 3.10.
- The DocLayNet subset source, within the §10.1 criteria.
- The heron ONNX I/O details (input size, normalisation, output decoding). Take them from the model card and the export's config, and test them on a known page.
- Threshold values after tuning on train pages, with the log recorded.
- Playground layout, within the existing visual language.

All of these must be recorded in the task reports and the elements benchmark doc.
