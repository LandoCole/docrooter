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
held-out evaluation and the repository essentials.

### Added

- **Memory budget.** `docrooter.configure(memory_budget_gb=..., threads=..., device=...)` and the
  `DOCROOTER_MEMORY_BUDGET_GB`, `DOCROOTER_THREADS` and `DOCROOTER_DEVICE` variables set
  process-wide limits (8 GiB by default). On Apple silicon, Metal gets the budget minus 1.5 GiB of
  host headroom. A model that runs out of MPS memory moves to the CPU once, with the same weights,
  and logs a warning; fine-tuning first retries with half the batch.
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
  `examples/` (nine new ones; all but `openai_backend.py` run without models or keys), and
  `tests/unit/test_docs_snippets.py` runs them and checks every CLI option the pages mention. A `docs` workflow builds it with
  `mkdocs build --strict` on pushes to `main` and on pull requests; nothing is deployed yet.
- **`docrooter --version`** prints the installed version.
- The playground uses the new mark (`favicon.svg`) as its tab icon.

### Changed

- The UltraBERT recipes read at most 1024 tokens (was 8192), and `train()` truncates to 512 tokens
  (`max_seq_len`) and fine-tunes with gradient checkpointing (`gradient_checkpointing=False`
  restores the old behaviour). Fine-tuning on MPS used to reach 19.5 GB and fail.
- The embedding cache is an LRU of 50,000 texts (`cache_size=`); it used to grow without bound.
- The pdfplumber engine frees each page after reading it.
- The package version is read from `docrooter.__version__` (a hatch dynamic version), so
  `pyproject.toml` and the module can no longer disagree.

### Fixed

- The playground read its static files with `Traversable.joinpath("static", name)`; the
  multi-segment form needs Python 3.11 for packages that are not plain directories (such as a
  zipped install). It joins one segment at a time now.

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
