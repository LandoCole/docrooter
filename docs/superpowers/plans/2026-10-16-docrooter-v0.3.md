# docrooter v0.3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** bounded memory, a held-out generalisation evaluation, production polish (docs site, branding, repo essentials, typed API, CI), and a live local playground UI.

**Spec:** `docs/superpowers/specs/2026-10-16-docrooter-v0.3-production-ui-design.md` (v0.1 and v0.2 specs still apply)

## Global Constraints

- All v0.1 and v0.2 global constraints still apply. In particular:
  - the core stays torch-free;
  - unit tests stay torch-free;
  - secrets live only in `.env`;
  - `PYTHONPATH=src` is used for all commands (the editable `.pth` file gets a macOS hidden flag);
  - never run a plain `uv sync`.
- **MEMORY (user requirement):** at most one model-loading process may run at a time. Before running models, wait with `while pgrep -f "benchmarks\.|docrooter serve" >/dev/null; do sleep 30; done`, excluding your own PID. Use `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false`.
- The new extra is `ui = ["fastapi", "uvicorn", "python-multipart"]`, included in `all`.
- The version becomes `0.2.0`.
- The UI binds `127.0.0.1` by default. It has no external CDN assets. Uploads are capped at 50 MB and deleted after processing.

## Tasks

### Task Q1: Memory containment

**Files:**
- `backends/recipes.py`
- `backends/embedding.py`, `backends/nli.py`, `backends/mlm_choice.py`, `backends/jev.py`, `backends/openai.py`, `backends/fake.py`
- `train/trainer.py`
- `train/bundle.py`
- `classify/classifier.py`
- new `src/docrooter/runtime.py` (holds `configure`)
- `__init__.py`
- `benchmarks/_guard.py` (new)
- `benchmarks/memory_check.py` (new)
- tests

- [ ] **Recipes.** The UltraBERT recipes get `max_seq_len=1024`. Add a recipe field `batch_size`, default 16, with `_cuda_batch_size` 32.
- [ ] **Training.** `train(max_seq_len=512)` sets `model.max_seq_length` before fine-tuning and before encoding the heads. Test with a fake ST that the value is applied.
- [ ] **`docrooter.configure`.** `configure(device=None, threads=None, mps_memory_fraction=None)` stores the settings and also reads the env vars `DOCROOTER_THREADS` and `DOCROOTER_MPS_MEMORY_FRACTION`. They are applied lazily on the first model load in `backends/loading.py`. Unit-test with fakes (torch-free).
- [ ] **Releasing memory.** Add `Backend.close()`, optional in the protocol: drop the model and caches, `gc.collect()`, and `empty_cache` when torch is already imported. Add `Classifier.close()` and `__enter__`/`__exit__`. `TrainedModel.close()` also closes its zero-shot body.
- [ ] **Bounded cache.** The embedding cache becomes an LRU (`collections.OrderedDict`) with `cache_size=50_000`. Test the eviction.
- [ ] **Benchmark guard.** `benchmarks/_guard.py` provides `single_model_process(name)`, which refuses to start when another `benchmarks.` model process is running, unless `--allow-concurrent` is passed. It also sets the default env caps. Wire it into `run.py`, `pdf_bench.py`, `normalize_bench.py` and `toc.py`, where it only sets the env caps since that script has no models.
- [ ] **Memory check.** `benchmarks/memory_check.py` measures peak RSS (`resource.getrusage`) and MPS allocated memory for two cases: `load` + `classify_document` on doc_042 (642 pages, UltraBERT), and `train(k=16 on B1)` with `finetune_body=True` on CPU. Run it, record the results in `docs/performance.md`, and check them against the targets in spec §2.
- [ ] **Fine-tuning benchmark.** Re-run the few-shot fine-tuned stage (B1, k=16, both bases) and the strategies suite with the new caps. This replaces the earlier OOM. Then update `docs/benchmarks.md`.
- [ ] **Commit.** Use the conventional commit messages.

### Task Q2: Pipeline events (`on_event`)

**Files:**
- `src/docrooter/events.py` (new)
- `pdf/load.py`
- `classify/documents.py`
- `classify/classifier.py`
- `classify/normalize.py`
- tests

- [ ] **Event model.** `Event(BaseModel)` has the fields `stage`, `status` (`"start"|"progress"|"end"|"error"`), `t` (monotonic seconds since the run started), and `data: dict`. `EventSink = Callable[[Event], None]`.
- [ ] **Instrumentation.** The stages and their data are listed in spec §5. Extraction progress is emitted every 10 pages: pymupdf iterates pages, so the hook goes in `extract_lines` through an optional `progress` callback. `model_load` is emitted when a backend loads lazily. Use a context var so backends can report it without new parameters.
- [ ] **Zero overhead without a sink.** When no sink is given, nothing changes. Test with a recording sink: the stage order and the end-after-start invariants for every stage.

### Task Q3: Playground server and frontend

**Files:**
- `src/docrooter/ui/__init__.py`
- `ui/server.py` (FastAPI app factory `create_app(backend_spec, ...)`)
- `ui/static/index.html`
- `ui/static/app.js`
- `ui/static/style.css`
- `cli.py` (`serve` command)
- `pyproject.toml` (`ui` extra and package data)
- `tests/unit/ui/test_server.py`

- [ ] **API.** Per spec §5. A single worker thread with a `queue.Queue` runs jobs. An SSE stream uses a per-run `asyncio.Queue`, fed via `loop.call_soon_threadsafe`. Results are kept in memory for the last 20 runs. Uploads go through `tempfile.NamedTemporaryFile(delete=False)` and are removed in `finally`. Uploads over 50 MB return 413. Content type and the `%PDF` magic are both validated. `.md` and `.txt` uploads are accepted too.
- [ ] **Backends.** `/api/backends` lists `keyword`, `embedding:…UltraBERT…`, `embedding:BAAI/bge-small-en-v1.5`, `nli:…`, plus `jev:jev-1.13.0` when `TYPESAFE_API_KEY` is set. Switching backend closes the previous one (Q1).
- [ ] **Frontend.** Use the frontend-design skill. One page, with no build step, no CDN, a system font stack, a light theme by default and a dark theme via `prefers-color-scheme`. It contains:
  - a drop zone;
  - option selects for backend, engine and mode, and a normalise toggle;
  - a live stage timeline / waterfall with durations and stage data;
  - a results area with a TOC tree (levels, pages, score chips), the doc type with a top-5 probability bars chart and the confidence, and section canonical labels;
  - a raw JSON tab with a download button;
  - run history.

  Accessibility: semantic HTML, keyboard-operable controls, visible focus.
- [ ] **CLI.** `docrooter serve --host --port --backend --open`. A non-localhost host prints a warning. `--open` uses `webbrowser.open`.
- [ ] **Tests.** Use the TestClient with the keyword backend and a generated PDF:
  - the stream yields `upload … done` in order;
  - the result schema is correct;
  - a 413 is returned on oversize;
  - the temp file is deleted;
  - a non-PDF bytes upload with a `.pdf` name is rejected.
- [ ] **Manual verification.** Run `docrooter serve` with the keyword backend and drive it in the in-app browser: upload `benchmarks/corpus/pdfs/doc_027.pdf`, take a screenshot, and check that the events stream live. Then repeat once with the UltraBERT backend, following the memory rule.

### Task Q4: Held-out generalisation evaluation (frozen)

**Files:**
- `benchmarks/heldout_eval.py` (orchestrates the existing harnesses with `--corpus heldout`)
- the existing harnesses gain a `--corpus dev|heldout` flag
- `benchmarks/heldout/section_gold.jsonl`
- `docs/benchmarks-heldout.md`
- a comparison table in `docs/benchmarks-pdf.md`

- [ ] **Labels first.** Write the held-out section silver labels for 4 documents (2 research papers, 1 annual report, 1 contract) and commit them BEFORE any model run. Use the same procedure as P10.
- [ ] **Frozen run.** Record the commit, then run once:
  - TOC: all engines, on the held-out docs with ≥5 bookmarks;
  - doc type: `jev-view`, `jev-chunks`, `ultrabert-view`, `ultrabert-hybrid`, `bge-view`, and "trained-on-dev → held-out" (LogisticRegression with the C picked by inner CV on the dev set, plus the UltraBERT+bge ensemble with the dev-chosen weight);
  - normalisation: UltraBERT, bge, Jev.
- [ ] **No tuning after the run.** No code or config changes between the recorded commit and the run. If a bug is found, fix it, re-run everything, and disclose that.
- [ ] **Write-up.** Put the dev-vs-held-out comparison table and an honest analysis in the docs.

### Task Q5: Profiling and the Rust decision

**Files:**
- `benchmarks/profile_pipeline.py`
- `docs/performance.md`

- [ ] Use `pyinstrument` (dev dependency) on 10 dev-corpus PDFs spanning sizes, with the default local pipeline (`load` + `classify_document` with UltraBERT, and `normalize_sections`). Report the time split: pymupdf extraction (C), our line assembly and signals and headings (Python), view rendering, tokenisation and inference.
- [ ] Apply the decision rule from spec §6. If Rust is justified, write a short follow-up spec instead of implementing it here. Otherwise document why not, with the numbers.
- [ ] Apply any cheap pure-Python hot-path optimisations found (e.g. regex precompilation, avoiding repeated `normalize_title`), with tests, if they give ≥ 15% on the Python share.

### Task Q6: Repository essentials, branding, typing, CI

**Files:**
- `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`
- `.github/ISSUE_TEMPLATE/{bug_report,feature_request}.md`, `.github/pull_request_template.md`
- `.pre-commit-config.yaml`
- `src/docrooter/py.typed`
- `docs/assets/{logo.svg,logo-dark.svg,wordmark.svg,favicon.svg}`
- `README.md` header and badges
- `pyproject.toml` (version 0.2.0, dev deps mypy, pytest-cov, pyinstrument, mkdocs-material, mkdocstrings[python])
- `.github/workflows/ci.yml` (mypy + coverage; a 3.10 and 3.12 matrix)
- `cli.py` (`--version`)

- [ ] Write the logo as hand-authored SVG. It is an original mark: a page outline whose ruled lines branch into a small root or tree, in two colours, legible at 16 px.
- [ ] Fix mypy errors in public modules until `mypy src/docrooter` is clean, or narrowly configured with documented ignores.
- [ ] Coverage: add a CI report. The target is ≥ 80% line coverage on `document`, `taxonomy`, `classify`, `views` and `pdf`.

### Task Q7: Documentation site

**Files:**
- `mkdocs.yml`
- `docs/site/**` (pages per spec §4)
- `.github/workflows/docs.yml` (build on push to main; do not deploy now)

- [ ] **Pages.** Every page's code samples come from `examples/` or are covered by a doctest-style test, so the docs stay runnable.
- [ ] **CLI reference.** Generate it by running `docrooter <cmd> --help` into markdown via a small script, `docs/gen_cli.py`.
- [ ] **Build.** `mkdocs build --strict` passes.

### Task Q8: Release readiness pass

- [ ] Finalise `docs/benchmarks.md` (v0.1 public-dataset decisions): commit the zero-shot, few-shot and strategy results, and write the Decisions section per the v0.1 decision rules with links.
- [ ] Refresh the README:
  - headline results: dev + held-out document type and TOC, plus the public datasets;
  - the gating-pipeline example;
  - the UI screenshot (`docs/assets/playground.png`);
  - the memory guidance.
- [ ] Run the full test suite, `ruff`, `mypy`, `mkdocs build --strict`, `uv build`, and a wheel smoke test in a fresh temp venv: `pip install dist/*.whl` then `docrooter --version` and `docrooter toc` on a Markdown file.
- [ ] Run the final whole-branch review (most capable model) and its fix wave.
