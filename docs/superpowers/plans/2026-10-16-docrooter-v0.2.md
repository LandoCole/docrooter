# docrooter v0.2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fast native multi-signal PDF TOC engine, document-type classification, section normalisation, a `doctype` gating CLI, and a 51-PDF benchmark comparing docrooter with TypeSafe Jev.

**Architecture:** `docrooter.pdf` turns PDFs into `Line` records (pymupdf or pdfplumber), then derives document signals (body size, running headers, printed TOC, bookmarks), scores heading candidates, assigns levels by style clusters, and assembles a v0.1 `Document`. `DocumentView` renders a whole document compactly for any v0.1 backend. `classify_document` and `normalize_sections` reuse the v0.1 `Classifier` with `Flat` and built-in taxonomies shipped as package data.

**Tech Stack:** v0.1 stack, plus pymupdf (extra `pdf`) and pdfplumber (new extra `pdf-mit`).

**Spec:** `docs/superpowers/specs/2026-10-16-docrooter-v0.2-pdf-doctype-design.md` (v0.1 spec still applies)

## Global Constraints

- All v0.1 global constraints still hold:
  - The core is torch-free, with lazy extras.
  - Unit tests never import torch; an autouse guard in `tests/unit/conftest.py` enforces this.
  - No recursion over trees.
  - The trust gate and pinned revisions apply.
  - Secrets live only in `.env` and are never printed.
  - Missing-extra errors read `pip install 'docrooter[<extra>]'`.
- New extra `pdf-mit = ["pdfplumber"]`, also included in `all`. The `pdf` extra keeps `pymupdf4llm`, which pulls in pymupdf. Import pymupdf as `import pymupdf` (fall back to `fitz`).
- No OCR anywhere in the default path. Never call pymupdf4llm unless `engine="pymupdf4llm"`.
- `Document.meta: dict[str, Any] = {}` is a new field. It must not break v0.1 serialisation or tests.
- Default `min_heading_score` for PDFs is 0.55. Pre-filter limits: `size ≥ 1.15 × body`; isolation means `gap_above ≥ 1.2 × median gap`; titles are ≤ 20 words. Running headers and footers sit in the top or bottom 8% of the page and recur on ≥30% of pages (minimum 3). The printed-TOC line regex is `^(.*?)[\s.·…_-]{2,}(\d{1,4})$` with ≥5 lines per page. Fuzzy match means `difflib.SequenceMatcher` ratio ≥ 0.85 on normalised text (casefold, collapse whitespace, strip leading numbering and trailing punctuation).
- Heading score weights are fixed exactly as in spec §2.3.
- Built-in taxonomies are JSON package data under `src/docrooter/taxonomies/` and are loaded via `importlib.resources`. Every node has a non-empty `description`.
- PDF unit tests generate tiny PDFs with pymupdf inside the test and skip if pymupdf is absent. Signal and heading logic is unit-tested on synthetic `Line` lists.
- Benchmark artifacts go under `benchmarks/results/<date>/` (committed JSON) and `benchmarks/.cache/` (ignored). Results docs are `docs/benchmarks-pdf.md`.
- The Jev model is pinned to `jev-1.13.0`. Jev is called only when `TYPESAFE_API_KEY` is present in the environment; benchmark scripts load `.env` silently.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Scanned or image-only PDF with no text layer.** `load_pdf` returns a Document with a single empty `s0`, plus `meta["warnings"]` containing `"no_text_layer"`. It does not crash. classify_document then returns `stopped_reason="empty_text"`. (P4)
2. **PDF where every line has the same font size** (e.g. plain RFC text). Headings must still come from numbering, isolation or printed TOC, and the result must not be zero sections when numbered headings exist. (P3)
3. **Slide deck (investor presentation)** where most text is large. The body size is the mode, so large slide titles stay headings and bullet text does not. There must be no explosion to hundreds of level-1 headings: cap title merging and require isolation. (P3, checked in the P5 benchmark)
4. **Encrypted or broken PDF.** Raise `DocrooterError` naming the file. Never hang. (P4)
5. **`--keep` with an internal node id** (e.g. `financial`) keeps every document whose type descends from it. (P8)

---

### Task P1: Heading-scoring fixes, `Document.meta`, built-in taxonomies

**Files:**
- Modify: `src/docrooter/document/headings.py`, `src/docrooter/document/models.py`, `pyproject.toml` (package data)
- Create: `src/docrooter/taxonomies/__init__.py`, `src/docrooter/taxonomies/doc_types.json` (copied from `benchmarks/corpus/doc_types.json`), `src/docrooter/taxonomies/section_types/{annual_report,research_paper,contract,technical_document,generic}.json`, tests `tests/unit/test_builtin_taxonomies.py`, extend `tests/unit/document/test_headings.py`
- Modify: `src/docrooter/__init__.py` (export `builtin_taxonomy`)

**Interfaces:**
- Produces: `builtin_taxonomy(name: str) -> Taxonomy` (names `doc_types`, `section_types/<name>`), `list_builtin_taxonomies() -> list[str]`, `DOC_TYPE_TO_SECTION_TAXONOMY: dict[str, str]` (leaf doc-type id → section taxonomy name), and `Document.meta: dict[str, Any]`.

- [ ] **Step 1: Failing tests.**
  - `score_heading("IV. Results")[0] >= 0.85`
  - `score_heading("ITEM 7. MANAGEMENT'S DISCUSSION")` gets the numbering bonus
  - `"Company's Results Year-over-Year"` gets the title-case bonus
  - Every built-in taxonomy loads, every node has a description, and `doc_types` has 18 leaves
  - `DOC_TYPE_TO_SECTION_TAXONOMY` covers all 18 leaves
  - `Document().meta == {}`, and `meta` round-trips through `model_dump`/`model_validate`
- [ ] **Step 2:** Implement the fixes. Make the numbering regex case-insensitive. Change the roman-numeral alternative to `[IVXLC]+\.(?=\s|$)`. Use the title-case token pattern `[^\W\d_]+(?:['’-][^\W\d_]+)*`. Write the section taxonomies with the node lists from spec §3.3; ids are slugs and descriptions are one sentence each. Map doc types to section taxonomies:
  - `financial.annual_report`, `quarterly_report`, `proxy_statement` → annual_report
  - `research.*` → research_paper
  - `legal.contract` → contract
  - `technical.*` → technical_document
  - everything else → generic
- [ ] **Step 3:** Run `uv run pytest -q` and `uv run ruff check .`, then commit with message `feat: heading regex fixes, Document.meta, built-in taxonomies`.

### Task P2: PDF line extraction and document signals

**Files:**
- Create: `src/docrooter/pdf/__init__.py`, `src/docrooter/pdf/lines.py`, `src/docrooter/pdf/signals.py`, `tests/unit/pdf/test_lines.py`, `tests/unit/pdf/test_signals.py`
- Modify: `pyproject.toml` (extra `pdf-mit`, add to `all`), `uv.lock`

**Interfaces:**
- Produces:
  - `Line` (pydantic; fields per spec §2.1)
  - `extract_lines(path, *, engine: Literal["pymupdf","pdfplumber"]) -> tuple[list[Line], int]`, returning the lines and the page count. It raises `DocrooterError(f"cannot open PDF {path}: …")` on encrypted or corrupt files.
  - `DocSignals(BaseModel)`: `body_size: float`, `median_gap: float`, `excluded: set[int]` (indexes of header/footer/page-number lines), `printed_toc: list[tuple[str, str]]` (title, page label), `toc_page_lines: set[int]`, `bookmarks: list[tuple[int, str, int]]`
  - `compute_signals(lines, n_pages, *, bookmarks=None) -> DocSignals`
  - `normalize_title(s) -> str` (the shared fuzzy-match normaliser)
  - `fuzzy_match(a, b) -> float`
  - `read_bookmarks(path) -> list[tuple[int, str, int]]` (pymupdf; returns `[]` if unavailable)

- [ ] **Step 1: Failing tests.**
  - Synthetic `Line` lists cover:
    - The body size is the character-weighted mode.
    - A footer "Company Confidential" present on 10 of 12 pages within the bottom 8% is excluded.
    - "Page 3 of 12" and a lone "iv" are excluded.
    - A page with 6 dot-leader lines yields `printed_toc` entries and marks those lines.
    - `normalize_title("1.2  Risk Factors.") == "risk factors"`.
  - PDF tests, skipped without pymupdf:
    - A generated 2-page PDF with a 20pt bold title and 11pt body gives the right `size`/`bold`, 1-based `page`, and finite `gap_above`.
    - The pdfplumber engine gives the same line texts on that PDF (skipped without pdfplumber).
    - A non-PDF file raises `DocrooterError`.
- [ ] **Step 2:** Implement as specified. `uv sync --extra pdf --extra pdf-mit` is permitted.
- [ ] **Step 3:** Run tests and lint, then commit with message `feat: PDF line extraction and document signals`.

### Task P3: Heading candidates, scoring, level assignment

**Files:**
- Create: `src/docrooter/pdf/headings.py`, `tests/unit/pdf/test_headings.py`

**Interfaces:**
- Consumes: `Line`, `DocSignals`, `fuzzy_match`, v0.1 `score_heading`, `RawHeading`, `repair_levels`.
- Produces:
  - `HeadingCandidate(BaseModel)`: `line_index: int`, `text`, `page`, `size`, `bold`, `score`, `level: int | None`, `flags: list[str]`
  - `find_headings(lines, signals, *, min_score=0.55, use_bookmarks=True, use_printed_toc=True) -> list[HeadingCandidate]`, scored, filtered, with levels assigned, in reading order
- Algorithm: implement spec §2.3 exactly, including the multi-line title merge. Merge only when consecutive lines are on the same page, have the same style, have a gap under 1.5 × median gap, and have a combined length of 20 words or fewer.

- [ ] **Step 1: Failing tests on synthetic lines.**
  - Size-based headings: two heading sizes give levels 1 and 2.
  - The style tie-break puts bold before regular at equal size.
  - Equal font size everywhere: numbered isolated lines ("1 Introduction", "2.1 Scope") are found, with levels from numbering (Review Focus 2).
  - Slide-deck case: 30 pages with a large title per page and 18pt bullets over a 14pt body gives ≤ 40 headings and no bullet lines (Review Focus 3).
  - A printed TOC match boosts a borderline candidate above threshold.
  - A bookmark match sets `level_from_bookmark`.
  - A two-line title is merged.
  - A sentence-like 25-word bold line is rejected.
  - Excluded footer lines are never candidates.
- [ ] **Step 2:** Implement.
- [ ] **Step 3:** Run tests and lint, then commit with message `feat: PDF heading detection and level assignment`.

### Task P4: `load_pdf` assembly, engines, timings, CLI `--engine`

**Files:**
- Modify: `src/docrooter/document/pdf.py` (now delegates to `docrooter.pdf.load`; keep the pymupdf4llm path as `engine="pymupdf4llm"`), `src/docrooter/document/load.py`, `src/docrooter/cli.py`
- Create: `src/docrooter/pdf/load.py`, `tests/unit/pdf/test_load.py`

**Interfaces:**
- Produces: `load_pdf(path, *, engine="auto", min_heading_score=0.55, use_bookmarks=True, use_printed_toc=True, repair=True) -> Document`, filling `meta` per spec §2.4. `load()` passes `engine` through via `**kwargs` for PDFs. `docrooter toc FILE --engine auto|pymupdf|pdfplumber|both|pymupdf4llm`.
- Assembly:
  - Headings become `RawHeading(title, level, text=<body lines until next heading>, line=<line_index>, page, heading_score=score, flags)`.
  - Text before the first heading becomes the preamble (level 0).
  - Then `repair_levels` (if `repair`), then `build_document`.
- `engine="both"` keeps a heading if both engines found it (fuzzy title ≥ 0.85, page equal) or if either scored ≥ 0.8. The body comes from the pymupdf lines.

- [ ] **Step 1: Failing tests (generated PDFs).**
  - A 3-page PDF with H1/H2 styles yields the right TOC, pages and levels, plus a `meta` with an `engine`, `n_pages` and four timings.
  - An image-only PDF (a page with just a drawn rectangle) yields a single `s0` and `meta["warnings"] == ["no_text_layer"]` (Review Focus 1).
  - An encrypted PDF (`doc.save(..., encryption=PDF_ENCRYPT_AES_256, user_pw="x")`) raises `DocrooterError` (Review Focus 4).
  - `engine="both"` returns a subset or equal set of headings.
  - The CLI `toc --engine pymupdf` works.
- [ ] **Step 2:** Implement.
- [ ] **Step 3: Corpus sanity (report only).** Run `load()` with the default engine on `doc_003` (Amazon annual report, 92 pages) and `doc_027` (Attention paper) from `benchmarks/corpus/pdfs/`. Record seconds, section count, the level histogram and the title in the report. The target is under 1.5 s per 100 pages.
- [ ] **Step 4:** Run tests and lint, then commit with message `feat: native PDF TOC engine (load_pdf)`.

### Task P5: TOC benchmark on corpus bookmarks

**Files:**
- Create: `benchmarks/toc.py`, `tests/unit/test_bench_toc.py`
- Output: `benchmarks/results/<date>/toc.json` and a section in `docs/benchmarks-pdf.md` (create the file with a `## PDF TOC extraction` section; later tasks append)

**Interfaces:**
- `match_headings(pred: list[tuple[str,int,int]], gold: list[tuple[str,int,int]], *, ratio=0.85, page_tol=1) -> list[tuple[int,int]]` (one-to-one greedy by ratio). Each tuple is title, page, level.
- `score_doc(pred, gold, *, max_gold_depth=None) -> dict` with keys `p, r, f1, level_acc, n_pred, n_gold`.
- CLI: `python -m benchmarks.toc --engines pymupdf,pdfplumber,both,pymupdf4llm --out DIR`.
- Uses corpus docs with ≥ 5 bookmarks. Gold is the bookmarks; extraction runs with `use_bookmarks=False`. Report both all-depth and depth ≤ 2 metrics, as micro and macro means, plus seconds per doc.

- [ ] **Step 1: Failing unit tests.** `match_headings` must be one-to-one, respect page tolerance, and prefer the highest ratio. `score_doc` math is checked by hand.
- [ ] **Step 2:** Implement the harness.
- [ ] **Step 3: Run the benchmark.** pymupdf, pdfplumber and both run on all 24 docs. `pymupdf4llm` may be limited to ≤ 8 docs if it exceeds 10 minutes, and that must be disclosed. Write results and the markdown section, including a per-doc table.
- [ ] **Step 4:** If default-engine depth ≤ 2 F1 is below 0.70, inspect the 3 worst docs and report why. Do not tune weights in this task; propose changes in the report.
- [ ] **Step 5:** Commit with message `bench: PDF TOC extraction benchmark`.

### Task P6: DocumentView and `classify_document`

**Files:**
- Create: `src/docrooter/doc_view.py`, `src/docrooter/classify/documents.py`, `tests/unit/test_doc_view.py`, `tests/unit/classify/test_documents.py`
- Modify: `src/docrooter/classify/classifier.py` (add `classify_documents`), `src/docrooter/classify/results.py` (`DocTypeResult`), `src/docrooter/__init__.py` (export `DocumentView`, `classify_document`, `DocTypeResult`)

**Interfaces:**
- Produces:
  - `DocumentView` and `render(doc, count_tokens=None) -> RenderedView` per spec §3.1
  - `DocTypeResult` per spec §3.2
  - `Classifier.classify_documents(docs, *, view=None, mode="view") -> list[DocTypeResult]`, batched: one `classify_texts` over all views, and for sections mode one traversal over all docs' leaf sections
  - `classify_document(doc, taxonomy, *, backend="embedding:dmis-lab/UltraBERT-Embed-150M", strategy=None, view=None, instruction="What type of document is this?", mode="view", **classifier_kwargs) -> DocTypeResult`. When `strategy` is None, use `Flat()`.
- `leaf_probs`:
  - For Flat: the normalised probabilities over leaves, from the Flat step's `probs` plus the alternatives. Add a `Classifier`-internal way to get the full leaf distribution, e.g. `run_flat` returns all leaf probs in `PathResult.steps[0].probs` when a `full_probs=True` setting is on. Keep the top-5 default for other callers.
  - For tree strategies: the product along paths for the reported alternatives.
- `mode="sections"`: classify each leaf section's `SectionView(fields=("title","lead"), max_tokens=128)` with Flat, then take the token-weighted mean of leaf distributions. `mode="hybrid"`: `0.6·view + 0.4·sections`.

- [ ] **Step 1: Failing tests (KeywordBackend).**
  - Render order and labels, e.g. `Contents: 1 Intro; 1.1 Scope; 2 Results`.
  - The cover, notice and TOC sections are skipped for `Lead`.
  - The budget is respected and the view is never empty.
  - `classify_document` returns the right leaf for a keyword-crafted doc in all three modes.
  - `leaf_probs` sum ≤ 1 + 1e-6 and contain best.
  - `classify_documents` makes one backend call per strategy round for N docs.
  - An empty doc gives `stopped_reason="empty_text"`.
- [ ] **Step 2:** Implement.
- [ ] **Step 3:** Run tests and lint, then commit with message `feat: DocumentView and document-type classification`.

### Task P7: Section normalisation

**Files:**
- Create: `src/docrooter/classify/normalize.py`, `tests/unit/classify/test_normalize.py`
- Modify: `src/docrooter/__init__.py` (export `normalize_sections`, `NormalizationResult`)

**Interfaces:**
- Produces: `NormalizationResult(BaseModel)` with fields `doc_type: str | None`, `taxonomy: str`, `sections: dict[str, SectionNorm]`. `SectionNorm` holds `canonical: str | None`, `label: str | None`, `score: float`, `title: str`.
- `normalize_sections(doc, *, doc_type=None, taxonomy=None, backend="embedding:dmis-lab/UltraBERT-Embed-150M", levels=(1, 2), classifier=None) -> NormalizationResult`. `taxonomy` may be a `Taxonomy` or a built-in name.

- [ ] **Step 1: Failing tests (KeywordBackend).**
  - The right section taxonomy is chosen from `doc_type`.
  - A heading "References" maps to `references` in `research_paper`.
  - Only the requested levels are classified.
  - With `doc_type=None`, `classify_document` is called first (spy).
  - An unknown built-in name raises `KeyError` listing the valid names.
- [ ] **Step 2:** Implement.
- [ ] **Step 3:** Run tests and lint, then commit with message `feat: section normalisation`.

### Task P8: CLI `doctype` and `normalize`

**Files:**
- Modify: `src/docrooter/cli.py`
- Create: `tests/unit/test_cli_doctype.py`

**Interfaces:** per spec §3.5.
- `-t` accepts a path or `builtin:<name>`.
- `--workers` parses PDFs in a `concurrent.futures.ProcessPoolExecutor`. Fall back to serial when `workers == 1` or there is a single file.
- `--keep` matches the doc type or any ancestor id (Review Focus 5).
- `--jsonl` writes UTF-8 lines.
- The exit code is 0 even when some files fail. Failed files are listed with an `error` field in JSONL and in a table row.

- [ ] **Step 1: Failing tests (KeywordBackend, Markdown inputs in a tmp dir).**
  - A directory is expanded.
  - The table lists every file.
  - `--keep financial` keeps only financial docs.
  - The `--jsonl` schema is correct.
  - One unreadable file still gives exit 0 with an error row.
  - `normalize --json` output is correct.
- [ ] **Step 2:** Implement.
- [ ] **Step 3:** Run tests and lint, then commit with message `feat: doctype and normalize CLI`.

### Task P9: Real-PDF document-type benchmark (incl. Jev)

**Files:**
- Create: `benchmarks/pdf_bench.py`, `benchmarks/jev_chunks.py`, `tests/unit/test_bench_pdf.py`
- Output: `benchmarks/results/<date>/pdf.json` and the `## Document-type classification` section of `docs/benchmarks-pdf.md`

**Interfaces:**
- `parse_corpus(engine="auto") -> dict[doc_id, (Document, parse_seconds)]`, cached as JSON in `benchmarks/.cache/docs/`.
- `SYSTEMS` registry per spec §4.
- `run(systems, out)` writes per-document rows `{doc_id, gold, pred, leaf_probs_top3, parse_s, classify_s, tokens?}` incrementally.
- `jev_chunks.classify(pdf_path, taxonomy, *, chunk_tokens=1500, max_chunks=8) -> (leaf, probs, n_tokens)`:
  - Text comes from pymupdf `page.get_text()`.
  - Tokens are estimated as words × 1.3.
  - Chunks are chosen evenly spaced.
  - There is one Choice per chunk; requests run concurrently through `JevBackend`. Reuse the backend: candidate texts = `candidate_text(leaf, taxonomy=..., full_path=True)` including descriptions.
- `ultrabert-loo`:
  - Embed each doc's DocumentView once with the UltraBERT query prompt.
  - For each held-out doc, fit `LogisticRegression(max_iter=1000)` on the other 50 (leaf labels) and predict.
  - Report it as "trained (leave-one-out)".
- Metrics: accuracy (leaf and level 1), macro P/R/F1 over gold leaf classes (reuse `docrooter.evaluate.metrics.leaf_prf`), confusion matrix, per-class table, timing averages, and Jev tokens and cost at $0.042 per 1M input tokens.

- [ ] **Step 1: Failing unit tests** (no network, no torch): metric aggregation on synthetic rows, chunk selection (evenly spaced, ≤ max), and a LOO split that never trains on the held-out doc.
- [ ] **Step 2:** Implement.
- [ ] **Step 3: Run.**
  - Parse the corpus with the default engine.
  - Run all systems on CPU unless MPS is responsive. Check MPS with a 10-second probe first; the machine is memory constrained, and MPS hung earlier under swap pressure.
  - Jev runs need `.env` loaded silently.
  - Keep the full run in the background, log to `benchmarks/.cache/pdf_bench.log`, and write results incrementally.
- [ ] **Step 4:** Write the markdown section: aggregate table, per-class tables for the best docrooter system and for Jev, confusion matrices, a per-doc table, and a frank analysis of where each fails.
- [ ] **Step 5:** Commit with message `bench: real-PDF document-type benchmark vs Jev`.

### Task P10: Section normalisation silver set and evaluation

**Files:**
- Create: `benchmarks/corpus/section_gold.jsonl`, `benchmarks/normalize_bench.py`, `tests/unit/test_bench_normalize.py`
- Output: the `## Section normalisation` section of `docs/benchmarks-pdf.md`

- [ ] **Step 1: Write the silver labels.**
  - Documents: research papers `doc_027`, `doc_028` and `doc_029`; annual reports `doc_001` and `doc_002`; contract `doc_016`.
  - Load each with the default engine, list its level-1 and level-2 TOC entries, and assign each one a canonical node id from the mapped section taxonomy by reading the heading and its first lines against the node definitions.
  - Record one JSON line per entry: `{doc_id, section_title, page, canonical}`, matched to the loaded document by title and page.
  - State in the file header (README section) that these are agent-assigned silver labels.
- [ ] **Step 2:** Write a unit test of the matching logic, then implement `normalize_bench.py`. It runs `normalize_sections` with `doc_type` = the gold doc type, using the UltraBERT and Jev backends, and reports accuracy and macro F1 per document and overall.
- [ ] **Step 3:** Run it, write the markdown section, and commit with message `bench: section normalisation silver evaluation`.
