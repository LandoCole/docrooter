# Changelog

All notable changes to docrooter are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the version is below 1.0, a minor
release may still change the API; the changes are listed under *Changed*.

The repository URL in the comparison links at the bottom is a placeholder
(`docrooter/docrooter`). No git tags exist yet: `v0.1.0` and `v0.2.0` will be created at release
time, and until then the comparison links do not resolve. The package version is `0.2.0`, read
from `docrooter.__version__` (the one place it is set), and the `[Unreleased]` work below ships
in the release after it.

## [Unreleased]

Work towards the next release: memory containment, pipeline events, a local playground, a
held-out evaluation, the repository essentials, (v0.4) elements, OCR and the rooter, (v0.5)
element tags and a tagger you train yourself, and (v0.6) selection and the completeness dial.

### Added

- **Element tags (v0.5).** Each paragraph, list item, table, caption, footnote and formula can
  carry a **topic** (one leaf of `element_topics`: a tree of document family, then section) and
  **functions** (any of the twelve tags of `element_functions`) in `Element.tags`, a dictionary of
  `TagResult` (`labels`, `probs`, `confidence`, `source`, and for topics `family_probs`). Set the
  rooter's `element_tags` task to a trained bundle directory, `embedding:<model>` (a zero-shot
  baseline) or `teacher:<engine spec>`; it is `none` in every profile, and an untagged element
  serialises as in v0.4. `docrooter tag FILE --bundle PATH [--json]`,
  `docrooter tag-corpus DIR --bundle PATH` and `docrooter serve --element-tags BUNDLE` (tag chips
  in the playground's Elements tab) use it. See
  [docs/site/concepts/element-tags.md](docs/site/concepts/element-tags.md).
- **Train your own tagger.** `docrooter corpus build DIR` reads PDFs into a resumable corpus store
  (the elements of the first and some sampled pages); `docrooter teach label DIR` has a local
  teacher LLM label topics (by multiple-choice log-probability, among an embedding shortlist) and
  functions (yes or no); `docrooter train-elements DIR --teacher SPEC --out BUNDLE` runs the
  labelling, a confident training set (term mining, a similarity-gap filter, a keep rule) and the
  student's training and calibration, every stage resumable. The teacher is
  Qwen3-4B-Instruct-2507 (Apache-2.0, 4-bit) on MLX by default (extra `teach-mlx`, Apple silicon);
  `hf:` (extra `models`) and OpenAI-compatible `openai:` (extra `llm`) teachers work too.
  **Jev cannot be a teacher**: TypeSafe MCA §2.3(b) forbids distillation, and docrooter refuses it
  with a clear error; a teacher you bring is under its provider's terms. See
  [Train your own element tagger](docs/site/guides/train-your-tagger.md) and
  `examples/train_element_tagger.py` (a tiny corpus, no model).
- **`financial.fact_sheet`**, a nineteenth document type: a short periodic summary of one fund or
  share class. Fund documents have their own section taxonomy, `section_types/fund_document`.
- **The training corpus** (a manifest and fetch scripts, `benchmarks/training_corpus/`; no PDF is
  committed): 1,100 public native-PDF documents with 81,822 pages (925 train, 62 calibration, 113
  gold), and a gold set of 900 elements whose labels are **Claude-annotated** (topic Cohen's
  kappa 0.953 on 180 double-annotated elements; the annotators share one underlying model, so
  kappa measures consistency, not human-verified correctness). EDGAR could not be used (the SEC
  refuses the project's User-Agent), so annual reports come from issuers' own investor-relations
  sites, mostly non-US. CUAD contracts (CC BY 4.0) are attributed in the corpus README.
- **Element-tag benchmarks.** The harness and its method are in
  [docs/benchmarks-element-tags.md](docs/benchmarks-element-tags.md); the results are pending.

- **Selection and the completeness dial (v0.6).** `select(analysis, need, completeness=0.9)`
  says which elements a downstream LLM should read for a need, and your application does the
  extraction. See [The completeness dial](docs/site/concepts/completeness-dial.md) and
  [Select what to send to an LLM](docs/site/guides/select-for-extraction.md).
  - **Fields.** A `Field` names the element topics (and functions) that carry a target.
    Built-in sets: `annual_report` (15 fields), `fund_document` (12) and `contract` (15);
    `builtin_fields`, `load_fields` and `validate_fields` read and check your own, in JSON.
  - **`select()`** takes a field, a list of fields of one family or a free-text query, and
    returns a `Selection`: the element ids in reading order, the `chunks`, `tokens` and
    `token_share`, `expected_recall`, `relevant_expected`, and the flags `guarantee`, `capped`,
    `doc_type_mismatch` and `over_budget`. `readout()` is its one-line summary. `completeness=1.0`
    is always the whole document and needs no model. A `budget` (tokens) replaces the dial.
    Each selected element brings its context (the previous element, the section's heading and
    first paragraph, a table's caption), so expansion can only raise recall.
  - **The dial** is conformal risk control, per element family: on documents like the calibration
    set, the average share of relevant elements kept is at least `c`. **It is calibrated against
    teacher relevance, not human labels**, it is an average (some documents are below `c`), its
    1/(n+1) slack can cap it (the whole document is returned, `capped=True`), and its units come
    from shared documents, so they are correlated. It gives no guarantee, only an estimate, for
    free-text queries, without a tagger or embedder, on a document type that does not match the
    field's family, under a budget, or for a family without a dial.
  - **`docrooter fit-selector`** fits the relevance models (on calib-fit) and the dials (on
    calib-crc) from the teacher's full labels, which `docrooter teach label --full` writes
    (`labels/teacher-full.jsonl`, never used to train the tagger). There is no built-in selector:
    fit one on your corpus.
  - **CLI and playground.** `docrooter select FILE --need ... [--completeness] [--budget]
    [--tagger] [--selector] [--json]`, and `docrooter serve --selector` for a dial panel: field
    picker, query box, slider, the presets Lean (0.8), Balanced (0.9), Thorough (0.98) and
    Whole document (1.0), a readout with badges, and *Show selected only*.
  - **Benchmarks.** `benchmarks/selection_bench.py` (recall against the promise, tokens against
    baselines), `benchmarks/evidence_qa.py` (evaluation-only public QA evidence) and a demo
    extraction harness, `benchmarks/extraction/` (not in the wheel). The results are pending:
    [docs/benchmarks-selection.md](docs/benchmarks-selection.md).
  - **Example.** `examples/select_for_extraction.py` runs `select()` with stand-ins and no model.

- **Elements.** Every visible part of a document is a typed `Element`: title, heading,
  paragraph, list item, table, figure, caption, footnote, formula, contents and page furniture,
  each with page boxes, a place in reading order and its section. `load_pdf(path,
  elements=True)` and `load(path, elements=True)` build them (Markdown too), and
  `Document.elements`, `Section.element_ids` and `Document.meta["routes"]` carry them. Two-column
  pages are read column by column (XY-cut), and a gutter between the row-aligned labels and
  numbers of an unruled table is not cut. In element mode running headers, footers and page
  numbers move to the start or end of their page, by design. See
  [docs/site/concepts/elements.md](docs/site/concepts/elements.md).
- **The rooter.** `process()` and `process_many()` run one pipeline under a `RouterConfig` (Python,
  TOML or JSON, with the precedence call arguments, file, profile) and return an `Analysis`. Three
  profiles: `fast` (the v0.3 pipeline, byte-identical and within 5% of its speed), `balanced`
  (the default: heuristic elements, OCR, summaries) and `accurate` (the heron layout model, with
  escalation to Jev when `TYPESAFE_API_KEY` is set). Each page is routed to its text layer or to
  OCR, and every decision is recorded in the route log.
- **OCR.** Three engines behind one protocol: `tesseract` (the system binary), `ocrmac` (extra
  `ocr-mac`) and `rapidocr` (extra `ocr`); `register_ocr_engine` adds your own. Measured
  character error rates: tesseract 0.058, rapidocr 0.089, ocrmac 0.102.
- **Layout model.** Extra `layout`: Docling's heron model on onnxruntime, no torch
  (`layout="heron"`; `register_layout_engine` adds others). On DocLayNet test pages its macro F1
  is 0.736 against 0.257 for the heuristic rules.
- **Summaries, chunks and embeddings.** `summarize` (key sentences and key phrases, no model),
  `chunk()` (section-aware chunks that keep tables whole, with their caption, and start with their heading path) and
  `embed()` (element or chunk embeddings, with the model and revision recorded).
- **Surfaces.** `docrooter elements`, `docrooter process [--profile] [--config] [--chunks]` and
  `docrooter config show`; a profile selector and an Elements tab in the playground; guides for
  [OCR and layout models](docs/site/guides/ocr-and-layout.md) and
  [chunking for RAG](docs/site/guides/chunking-for-rag.md); `examples/elements_and_chunks.py`.
- **Element benchmarks.** DocLayNet element kinds, reading order, OCR and routing, and speed:
  [docs/benchmarks-elements.md](docs/benchmarks-elements.md). Per 100 pages, `fast` takes 0.99 s,
  `balanced` 4.64 s (PyMuPDF) and `accurate` 64.6 s; `tables="none"` is the speed switch.
- **Memory budget.** `docrooter.configure(memory_budget_gb=..., threads=..., device=...)` and the
  `DOCROOTER_MEMORY_BUDGET_GB`, `DOCROOTER_THREADS` and `DOCROOTER_DEVICE` variables set
  process-wide limits (8 GiB by default). On Apple silicon, Metal gets the budget minus 1.5 GiB of
  host headroom. The cap applies to the whole process, configured or not: the first model load
  sets `PYTORCH_MPS_HIGH_WATERMARK_RATIO` and `PYTORCH_MPS_LOW_WATERMARK_RATIO` in `os.environ`
  (subprocesses inherit them) unless you set either one yourself. A model that runs out of MPS
  memory moves to the CPU once, with the same weights, and logs a warning; fine-tuning first
  retries with half the batch.
- **Releasing models.** Every backend has `close()`; `Classifier` and `TrainedModel` close their
  models too, and `Classifier` works as a context manager. `classify_document` and
  `normalize_sections` close the backends they load from a spec string.
- **Large corpora.** `docrooter doctype --chunk-size N` parses, classifies and writes files `N` at a
  time, and `--max-pages N` (`load_pdf(max_pages=N)`) skips the pages past `N` and records
  `"truncated_pages"` in `meta["warnings"]`.
- **Pipeline events.** `load`, `load_pdf`, `classify_document`, `Classifier.classify_documents` and
  `normalize_sections` take `on_event`, a callback that receives an `Event` for each stage
  (`extract`, `signals`, `headings`, `assemble`, `view`, `model_load`, `classify`, `normalize`);
  `docrooter.events.pipeline_run` shares one clock across chained calls. A sink that raises never
  breaks processing.
- **Playground.** `docrooter serve` starts a local web page (extra `ui`) where you upload a
  document, pick a backend and watch the stages run. It binds `127.0.0.1` by default, has no
  external assets, caps uploads at 50 MB and deletes them after processing. `--env-file` reads
  missing variables such as `TYPESAFE_API_KEY` from a file. See `SECURITY.md` for its guard rails.
- **Held-out evaluation.** A frozen 40-PDF corpus (`benchmarks/heldout/`), labels committed before
  any run, and one recorded run of the TOC, document-type and normalisation benchmarks on it:
  [docs/benchmarks-heldout.md](docs/benchmarks-heldout.md). The dev-versus-held-out comparison is in
  [docs/benchmarks-pdf.md](docs/benchmarks-pdf.md). Headline results:
  - Document type generalises: Jev given the `DocumentView` scores 0.925 leaf accuracy on the 40
    held-out PDFs (0.961 on dev) and UltraBERT zero-shot 0.825 (0.765 on dev).
  - TOC extraction drops: the default engine's micro F1 at depth <= 2 is 0.576 (0.703 on dev),
    partly real and partly bookmark artefacts.
  - Section normalisation with the local models holds (UltraBERT accuracy 0.579 against 0.514);
    Jev is slightly lower (0.692 against 0.725).
- **Performance notes**, with peak process memory for classifying a 642-page datasheet and for
  fine-tuning on EDGAR items: [docs/performance.md](docs/performance.md).
- **Repository essentials.** `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `SECURITY.md`, issue and pull request templates, a pre-commit configuration, the logo, wordmark
  and favicon under `docs/assets/`, and a README header with badges.
- **CI.** The test matrix covers Python 3.10 to 3.13, `ruff format --check` and `mypy` run, and an
  `extras-smoke` job installs the `pdf`, `pdf-mit`, `ui` and `llm` extras (no torch) so their
  tests run. That job also measures coverage and fails the build under 80% line coverage on any of
  `docrooter.document`, `taxonomy`, `classify`, `views` or `pdf` (96% to 100% at the time of
  writing). The publish workflow runs the tests before it uploads.
- **Typing.** The package ships `py.typed` and `mypy src/docrooter` is clean (the optional extras
  are configured as untyped imports in `pyproject.toml`).
- **Documentation site** (`mkdocs.yml`, pages in `docs/site/`): getting started, concepts, guides,
  a CLI reference generated from `--help` (`docs/gen_cli.py`), an API reference generated from the
  docstrings, the benchmark reports, and an FAQ. Every Python sample on it is a line of a script in
  `examples/` (ten new ones). All but `memory.py` (UltraBERT) and `openai_backend.py` (an
  OpenAI-compatible server) run with the keyword backend, some need the `pdf` extra, and
  `tests/unit/test_docs_snippets.py` runs those eight and checks every CLI option the pages
  mention. A `docs` workflow builds it with `mkdocs build --strict` on pushes to
  `main` and on pull requests; nothing is deployed yet.
- **`docrooter --version`** prints the installed version.
- The playground uses the new mark (`favicon.svg`) as its tab icon.

### Changed

- **Prospectuses use the fund-document sections.** `financial.prospectus` maps to
  `section_types/fund_document` (it was `generic`), as `financial.fact_sheet` does, so
  `normalize_sections` on a prospectus labels its headings with fees, risks and performance
  rather than generic sections. The document-type benchmark numbers in this changelog and the
  docs were measured with 18 types and predate `financial.fact_sheet`; the held-out corpus has
  not been run again.
- **`ocr="auto"` resolves to `tesseract` (the binary on PATH), then `ocrmac` (macOS), then
  `rapidocr`.** Tesseract comes first for accuracy (character error rate 0.058, against 0.089
  for rapidocr and 0.102 for ocrmac, which also fails on legal pages and table columns). ocrmac
  comes before rapidocr for speed and memory (0.73 against 3.04 s per page, 1.2 against 3.2 GiB),
  although rapidocr is the more accurate of the two. The first draft of the rooter preferred
  ocrmac on macOS.
- New extras `layout`, `ocr` and `ocr-mac`, all included in `all`; Python 3.10 gains `tomli` in
  the core dependencies (router config files).
- The UltraBERT recipes read at most 1024 tokens (was 8192), and `train()` truncates to 512 tokens
  (`max_seq_len`) and fine-tunes with gradient checkpointing (`gradient_checkpointing=False`
  restores the old behaviour). Fine-tuning on MPS used to fail (it reached 19.5 GB in a
  development run).
- The embedding cache is an LRU of 50,000 texts (`cache_size=`); it used to grow without bound.
- The pdfplumber engine frees each page after reading it.
- Loading a PDF with the default engine is faster: its four load stages took 19% less time on
  the ten profiling PDFs (0.94 against 1.15 s per 100 pages), and the TOC benchmark's 24 dev
  PDFs load at 1.05 s per 100 pages (see [docs/performance.md](docs/performance.md)).
- The package version is read from `docrooter.__version__` (a hatch dynamic version), so
  `pyproject.toml` and the module can no longer disagree.
- A trained bundle's manifest keeps the recipe's batch size; a half batch from an
  out-of-memory retry is recorded only in `train_metrics`.
- The `ui` extra requires `fastapi>=0.100`, `uvicorn>=0.22` and `python-multipart>=0.0.13`;
  `httpx` joins the dev group; the publish workflow fails when the release tag is not
  `v<__version__>`.
- The `classify_document` / `classify_documents` event docs list `model_load` before `view`,
  and the `configure()` docstring describes its process-wide MPS side effects and the opt-out.

### Fixed

- The playground read its static files with `Traversable.joinpath("static", name)`; the
  multi-segment form needs Python 3.11 for packages that are not plain directories (such as a
  zipped install). It joins one segment at a time now.
- `Classifier.classify_documents` skips a decision that has no best path when it collects stop
  reasons and no-backend warnings, instead of raising `AttributeError`. No built-in engine
  returns one; the guard came with the typing pass.
- The playground answers 400 to a `Content-Length` that is not an ASCII number (it answered 411,
  or 500 for a superscript digit, which `str.isdigit` accepts; a missing one is still 411), and
  ignores such a `Last-Event-ID` instead of failing.
- `docrooter serve --env-file` reads `KEY= # comment` as an empty value instead of `# comment`.
- The pdfplumber engine no longer splits every glyph into its own word when a slightly skewed
  text matrix gives the glyphs sizes that differ by float noise; such titles came out
  letter-spaced and were never heading candidates. Found through the held-out run, fixed on dev
  and synthetic PDFs only; the held-out numbers predate the fix.
- Closing the `as_zero_shot()` backend of an unsaved fine-tuned `TrainedModel` no longer lets
  the next call silently rebuild the base model under the fine-tuned heads; it raises
  `BackendError` until the model is saved.
- The embedding cache stores its own copy of each row, so `cache_size` bounds memory as well
  as entries (a cached row used to keep its whole encode batch alive).
- An MPS out-of-memory error while an NLI or MLM-choice model moves to the GPU falls back to
  the CPU inside the load: one `model_load` start and end, one model build, and a "while
  loading" warning.
- `docrooter serve` stops promptly on Ctrl+C while a page follows a run: the event streams
  are ended, with a 2 s graceful-shutdown bound.
- Playground `.md` and `.txt` uploads are capped at 1 MB, and a run keeps and returns at most
  5,000 TOC and section entries (`truncated` and `toc_total` in the result, shown on the page).
  A heading-dense text file costs about 270 MB of memory per MB, or about 1.6 GB per MB with
  section labels on.
- The playground's pipeline `error` event no longer contains the server's temp-file path.
- A `Content-Length` or `Last-Event-ID` longer than 18 digits is rejected (400) or ignored
  instead of causing a 500.
- The playground page stops waiting when a run is gone after a server restart.

## [0.2.0] - 2026-10-16

The PDF release: a native table-of-contents engine, document-type classification and section
normalisation, measured on real PDFs.

### Added

- **Native PDF TOC engine.** `load(path)` and `load_pdf(path, engine=...)` find the headings of a PDF
  without OCR, from font size and weight, position, numbering, running headers and footers, printed
  contents pages and (unless `use_bookmarks=False`) the PDF's bookmarks. Engines: `auto`
  (PyMuPDF if installed, else pdfplumber), `pymupdf`, `pdfplumber`, `both` and the legacy
  `pymupdf4llm` path. `Document.meta` records the engine, page count and timings.
- **`pdf-mit` extra** (`pdfplumber`, MIT) as a permissive alternative to the AGPL PyMuPDF, and a
  README note on the licensing difference.
- **Built-in taxonomies.** `doc_types` (18 document types in 6 groups) and one section taxonomy per
  family, with `list_builtin_taxonomies()` and `builtin_taxonomy(name)`; every node has a
  description.
- **Document types.** `DocumentView` (title, page and section counts, headings, opening text and
  keyphrases within a token budget), `classify_document` and
  `Classifier.classify_documents`, which return a `DocTypeResult` with a confidence.
- **Section normalisation.** `normalize_sections` maps each heading to a canonical section of the
  document's type (classifying the type first when none is given).
- **CLI.** `docrooter doctype` (table or `--jsonl`, `--keep` and `--paths-only` to gate a pipeline,
  `--workers`, `--mode view|sections|hybrid`) and `docrooter normalize`; `docrooter toc` takes
  `--engine`.
- **Real-PDF benchmarks** on a 51-PDF development corpus: [docs/benchmarks-pdf.md](docs/benchmarks-pdf.md).
  Headline results, all on the development set that the rules were tuned on:
  - TOC extraction with the default engine: micro F1 0.703 at depth <= 2 against PDF bookmarks
    (24 documents; 0.478 before tuning).
  - Document type over 18 leaf types: TypeSafe Jev given the `DocumentView` 49/51 correct (0.961);
    the best local system, UltraBERT with the hybrid view, 42/51 (0.824).
  - Section normalisation against agent-assigned silver labels (142 sections, 6 documents): Jev 0.725 accuracy,
    UltraBERT 0.514.

### Changed

- `Document` gained a `meta` dictionary; v0.1 serialisation and tests are unaffected.

### Fixed

- Dot-leader tables are not mistaken for contents pages, and running headers and footers that
  repeat on three or more pages are excluded from the headings.
- Doc-type decisions use a single rule, scanned PDFs without a text layer report an empty view
  rather than a guess, and `docrooter doctype` survives a broken process pool.
- Built-in taxonomies are cached (`builtin_taxonomy(name)` still returns a fresh copy), and an
  unknown name raises `UnknownTaxonomyError`.

## [0.1.0] - 2026-10-15

First release: classify the sections of a document against a taxonomy you supply.

### Added

- **Loading.** `load()` reads Markdown, plain text and PDFs (through PDF bookmarks), validates the
  headings and repairs the hierarchy; `docrooter toc` prints or exports the result.
- **Taxonomies.** A tree of categories with labels, descriptions, examples and per-node settings,
  read from nested JSON, flat JSON or three CSV layouts, with validation that names the row or JSON
  pointer of each problem (`docrooter taxonomy validate`).
- **Classifier.** Traversal strategies `Greedy`, `Beam(n)` and `Flat`; probabilities and the path
  through the tree for every section, rival alternatives, early stops with a reason, multi-label
  nodes, routing of nodes to different backends, and a fallback backend.
- **Backends.** `keyword`, `embedding` (default `dmis-lab/UltraBERT-Embed-150M`), `nli`,
  `mlm-choice`, `trained`, `openai` (any OpenAI-compatible server) and `jev` (TypeSafe Jev), plus a
  protocol for custom backends. Models with their own modelling code run only from a pinned commit
  (`trust_remote_code` gate); API keys are read from the environment and redacted from errors.
- **Training.** Per-node classifier heads on a shared, optionally fine-tuned embedding body
  (`train()`, `docrooter train`), `Dataset` helpers, and temperature calibration (`calibrate`).
- **Evaluation.** Exact-path accuracy, hierarchical precision, recall and F1, selective accuracy and
  throughput (`evaluate`, `compare`, `docrooter eval`).
- **CLI.** `toc`, `classify`, `eval`, `train`, `taxonomy validate` and `models pull|list` (offline
  use with `HF_HUB_OFFLINE=1`).
- **Benchmarks** on EDGAR-CORPUS 10-K items, DBpedia classes and LEDGAR contract clauses
  (`benchmarks/`); zero-shot results in [docs/benchmarks.md](docs/benchmarks.md). On 10-K items
  (n=1000) the UltraBERT embedding backend scores 0.752 hierarchical F1.
- Apache-2.0 licence and a trusted-publishing workflow for PyPI.

[Unreleased]: https://github.com/docrooter/docrooter/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/docrooter/docrooter/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/docrooter/docrooter/releases/tag/v0.1.0
