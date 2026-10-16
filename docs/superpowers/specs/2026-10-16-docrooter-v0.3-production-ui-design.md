# docrooter v0.3: production readiness and the local playground UI

- **Date:** 2026-10-16
- **Status:** Approved under the user's standing instruction ("write new specs and plans and loop until … ready for PROD … complete with documentation, getting started, logo/branding, license and all the essentials … simple clean local UI … to see processing time and backend processes as they happen"). Two later user requests are also in scope: contain memory usage, and validate generalisation on different document sets.
- **Builds on:** v0.1 and v0.2 specs.

## 1. Goals

1. **Bounded memory.** The package, tests and benchmarks run within a bounded memory footprint on a 16 GB laptop. No default code path should allocate multi-gigabyte buffers.
2. **Generalisation is measured.** A held-out corpus that was never used for tuning gets one frozen evaluation of TOC extraction, document types and section normalisation. The results are published next to the dev-set results.
3. **A professional, installable library.** That means:
   - a documentation site;
   - getting-started material;
   - logo and branding;
   - the standard repository files (changelog, contributing guide, code of conduct, security policy, templates);
   - typed public API (`py.typed`) and CI that runs lint, types and tests on 3.10 and 3.12;
   - a version bump to `0.2.0`.
4. **A local playground UI.** `docrooter serve` opens a browser page where the user uploads a PDF, watches every pipeline stage stream in live with timings, and inspects the TOC, the document type, section labels and raw JSON.

## 2. Memory containment

### Package defaults

- **UltraBERT recipe `max_seq_len`:** reduce from 8192 to **1024**. Document views are at most 1024 tokens, and longer inputs are truncated. This affects `dmis-lab/UltraBERT-Embed-150M` and `UltraBERT-380M`.
- **`train()` gains `max_seq_len: int = 512`.** It is applied to the SentenceTransformer during fine-tuning, so long training texts cannot allocate O(8k²) attention. This fixes the observed 19.5 GB MPS OOM.
- **Encoding batch size:** default 16 on CPU and MPS, 32 on CUDA. It is configurable per recipe and per call.
- **`docrooter.configure(...)`:** a new process-level setup function.
  - Signature: `configure(*, device=None, threads=None, mps_memory_fraction=None)`.
  - Each value is applied when torch is first used: `torch.set_num_threads`, and `torch.mps.set_per_process_memory_fraction`.
  - Matching environment variables: `DOCROOTER_DEVICE`, `DOCROOTER_THREADS`, `DOCROOTER_MPS_MEMORY_FRACTION`.
- **Releasing models:** backends get `close()`, which drops the model, empties the caches, calls `gc.collect()`, and calls `torch.mps.empty_cache()` or `torch.cuda.empty_cache()` where available. `Classifier` gets `close()` and context-manager support. Embedding caches become a bounded LRU, 50,000 entries by default.

### Tests and benchmarks

- Unit tests stay torch-free.
- `-m models` integration tests run in a single process. Each test module closes its backends.
- Benchmark entry points set conservative defaults unless the user overrides them: threads 4, MPS memory fraction 0.5, `TOKENIZERS_PARALLELISM=false`. They refuse to start when another docrooter benchmark process is already running, unless `--allow-concurrent` is passed.

### Verification

A memory test script, `benchmarks/memory_check.py`, records peak RSS and, where available, MPS-allocated memory for:

- `load` + `classify_document` on a 600-page PDF with UltraBERT;
- `train()` on B1 with k=16.

Targets: peak RSS under 3 GB for classification and under 6 GB for training, on CPU.

## 3. Held-out generalisation evaluation

- **Corpus:** `benchmarks/heldout/` (manifest and fetch script). It uses the same 18 leaf types, different documents, and different publishers where possible. Labels are fixed before any run.
- **Frozen evaluation:** run once at a recorded commit, with no changes to code or configuration afterwards. It covers:
  - TOC extraction on the held-out docs with at least 5 bookmarks: the same metrics as P5.
  - Document type for every system in the dev benchmark (`jev-view`, `jev-chunks`, `ultrabert-view`, `ultrabert-hybrid`, `bge-view`). The LOO-trained system is replaced by "train on all 51 dev docs, test on held-out", which is an honest transfer test.
  - Section normalisation: silver labels for 4 held-out documents (2 research papers, 1 annual report, 1 contract), written and committed before any model run.
- **Publication:** results go to `docs/benchmarks-heldout.md`. The dev-vs-held-out comparison table also goes into `docs/benchmarks-pdf.md`, with a frank discussion of any drop.

## 4. Documentation and branding

- **Docs site (`mkdocs-material` with `mkdocstrings`), in `docs/site/`.** Pages:
  - Home
  - Getting started: install matrix, a 5-minute quickstart with a PDF, the CLI
  - Concepts: the TOC engine, taxonomies, views, backends, strategies, results
  - Guides:
    - Document-type gating pipeline (`doctype --keep … --paths-only`)
    - Section normalisation
    - Custom taxonomies
    - Training and evaluation
    - Using Jev or an OpenAI-compatible LLM
    - Memory and performance
  - CLI reference, generated from typer
  - API reference, generated by mkdocstrings
  - Benchmarks: dev, held-out and public datasets
  - FAQ
  - Changelog
- **Docs publishing:** a GitHub Pages workflow, `docs.yml`, that builds on push to main. Do not push or deploy anything; only the workflow file is added.
- **Branding:**
  - `docs/assets/logo.svg`: an original mark, a document page whose ruled lines branch into a root or tree shape, in a 2-colour palette.
  - A favicon and a wordmark.
  - The README gets a header image and badges for Python versions, license and CI.
- **Repository essentials:** `CHANGELOG.md` (Keep a Changelog format), `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `SECURITY.md`, issue and PR templates, `.pre-commit-config.yaml` (ruff and ruff-format), and `py.typed`.
- **CI:** add a mypy step on the public modules, and pytest coverage reporting with a target of at least 80% on the core.
- **Version and release notes:** `0.2.0`, and `docrooter --version`.

## 5. Playground UI (`docrooter serve`)

### Packaging and server

- **Extra:** `ui = ["fastapi", "uvicorn", "python-multipart"]`.
- **Command:** `docrooter serve [--host 127.0.0.1] [--port 8765] [--backend SPEC] [--open]`.
- **Network exposure:** binds to localhost only by default. A non-local `--host` prints a warning.

### Pipeline events

Add an optional callback `on_event: Callable[[Event], None]`, with `Event(stage, status, t, data)`, to `load_pdf`, `classify_document` / `Classifier.classify_documents` and `normalize_sections`. The stages are:

- `upload`
- `extract` (with page progress every 10 pages)
- `signals`
- `headings`
- `assemble`
- `view`
- `model_load`
- `classify`
- `normalize`
- `done`
- `error`

### API

- `POST /api/runs` takes a multipart PDF plus options: backend, mode, engine, and whether to normalise. It returns `{run_id}`.
- `GET /api/runs/{id}/events` streams Server-Sent Events, each a JSON event.
- `GET /api/runs/{id}` returns the final result JSON.
- `GET /api/backends` lists the available backends, with Jev included only if `TYPESAFE_API_KEY` is set.

### Runs, limits and models

- One worker processes runs sequentially, keeping memory bounded.
- Uploads are capped at 50 MB, stored in a temp directory, and deleted after processing.
- Results are kept in memory for the session only: the last 20 runs.
- The backend is loaded once and reused. It can be switched from the UI, and switching closes the previous one.

### Frontend

The frontend is one static HTML file plus a small JS module, with no build step, served from package data. It contains:

- A drop zone and file picker, a backend selector, an engine selector, and a "normalize sections" toggle.
- A live stage timeline: each stage appears as it starts and finishes, with a duration bar (a waterfall) and per-stage details such as pages, lines, headings, tokens and the model device.
- A results tab with three parts:
  - a TOC tree (level indentation, page numbers, heading score chips, flags);
  - the document type with a probability bar chart of the top 5 and the confidence;
  - section canonical labels next to the TOC entries.
- A raw JSON tab with a download button.
- A run history list.

The look is clean and minimal: system fonts, a light theme by default and a dark theme via `prefers-color-scheme`, and no external CDN dependencies, so it works offline.

### Testing

- API tests use FastAPI's TestClient with the `keyword` backend and a generated PDF, so they need no models. They check the event order, the result schema, the upload limit and deletion of the temp file.
- A Playwright smoke test is optional.

## 6. Performance review (Rust decision)

Profile the pipeline on the dev corpus with `pyinstrument` and report the time split across PDF text extraction (pymupdf C code), our Python heading logic, view rendering and model inference.

Rust (PyO3) is justified only if our own Python code accounts for ≥ 30% of the time per document for the default local pipeline AND a hot loop is clearly isolable. Otherwise document the decision not to use Rust, with the numbers. The expectation, to be verified, is that model inference and pymupdf dominate.

## 7. Out of scope

- Hosted deployment.
- Authentication for the UI (it is localhost only).
- OCR.
- Training new base models.
