# Performance

## Memory

docrooter works within a memory budget, 8 GiB by default (`docrooter.configure(memory_budget_gb=...)`
or `DOCROOTER_MEMORY_BUDGET_GB`). The budget is meant for the whole process.

- **On Apple silicon,** Metal and the host share memory, so Metal gets the budget minus 1.5 GiB
  of host headroom (at least half the budget). That share is applied with
  `torch.mps.set_per_process_memory_fraction(metal / recommended_max_memory)`. Before torch
  loads, the `PYTORCH_MPS_*_WATERMARK_RATIO` variables are set from an estimate of the same
  value.
  - The fraction caps Metal memory only, not the process total. The headroom is what keeps
    the total within the budget.
- **On the CPU,** nothing caps the resident set. The budget only scales the default batch
  sizes.
- **Out-of-memory fallback:** a model that still runs out of MPS memory moves to the CPU once,
  with the same weights, and logs a warning. Fine-tuning first retries with half the batch.

`python -m benchmarks.memory_check` measures two cases, each in a fresh process:

- **classify:** `load` + `classify_document` on doc_042, the 642-page RP2040 datasheet (1,239
  sections), with UltraBERT-Embed-150M against `builtin:doc_types`.
- **train:** `train()` on B1 (EDGAR 10-K items) with k=16 per leaf (306 examples, 596 triplets),
  `finetune_body=True`, UltraBERT-Embed-150M base, and the defaults: `max_seq_len=512`, batch
  16 and gradient checkpointing.

The target is a peak **process total** of at most the 8 GiB budget.

- **Process total on macOS:** the peak physical footprint,
  `task_info(TASK_VM_INFO).ledger_phys_footprint_peak`. It is the "Memory" column of Activity
  Monitor and counts Metal buffers as well as host pages.
- **Peak RSS** (`ru_maxrss`) counts host pages only.
- A sampler thread records the peaks of `torch.mps.current_allocated_memory()` (tensors) and
  `driver_allocated_memory()` (everything Metal holds, including the allocator's cache).

Measured on 2026-10-16: Apple M1, 16 GB, macOS 26.5, Python 3.12, torch 2.14.1,
sentence-transformers 6.1.0. Benchmark defaults applied: 4 threads, `TOKENIZERS_PARALLELISM=false`.

| Case | Device | Process total | Peak RSS | MPS allocated | MPS driver | MPS fraction | Time | ≤ 8 GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| train | MPS (default) | 7.59 GiB | 1.21 GiB | 4.52 GiB | 6.39 GiB | 0.549 | 17.8 min | yes |
| classify | MPS (default) | 1.91 GiB¹ | 0.77 GiB | 0.56 GiB | 1.14 GiB | 0.6757¹ | 17 s | yes |
| train | CPU | 6.23 GiB² | 6.23 GiB | n/a | n/a | n/a | 24.2 min | yes |
| classify | CPU | 1.13 GiB² | 1.13 GiB | n/a | n/a | n/a | 24 s | yes |

- **MPS train run:** the fraction 0.549 is 6.5 GiB of the 11.84 GiB that Metal recommends on
  this machine. Training stayed on MPS at batch 16, with no retry or fallback.
- **Before the headroom:** with the whole budget given to Metal (fraction 0.6757), the same
  run peaked at 7.02 GiB of Metal memory plus 1.13 GiB RSS, about 8.15 GiB in total. That was
  over the budget, so the 1.5 GiB headroom was added.
- **¹ classify on MPS** was measured under the earlier 0.6757 cap, before the footprint was
  recorded. Its total is RSS + Metal driver, and it is far under the budget either way.
- **² CPU rows** were measured with `--device cpu` before the budget existed. They hold no
  Metal memory, so the total is the RSS.
- **Predictions:** both classify runs predicted `technical.user_manual` for doc_042, at
  confidence 0.33.
- **Raw rows:** `benchmarks/results/2026-10-16/memory.json`.

### What the defaults changed

- **Fine-tuning activations dominate training memory.** One CPU fine-tuning step over 16 triplets
  (48 texts) at 512 tokens took the process to 9.7 GiB without gradient checkpointing and to
  5.8 GiB with it. Halving the batch or the sequence length without checkpointing did not help
  (9.9 and 10.1 GiB), because the machine was already compressing memory at that point. The full
  B1 run without checkpointing peaked at 10.2 GiB on CPU and took 53 minutes; with checkpointing
  it peaked at 6.2 GiB and took 24 minutes (less memory pressure more than paid for the
  recomputation). `train(gradient_checkpointing=False)` restores the old behaviour.
- **Sequence length.** The UltraBERT recipes now stop at 1024 tokens (was 8192), and `train()`
  truncates to 512. Before this, fine-tuning on MPS reached 19.5 GB and failed.
- **Releasing models.** `close()` on every backend, `Classifier.close()` / `with Classifier(...)`,
  and `TrainedModel.close()` drop the model and caches, run `gc.collect()` and empty the MPS or
  CUDA cache. `classify_document` and `normalize_sections` close what they load from a spec
  string.
- **Embedding cache.** The embedding cache is an LRU of 50,000 texts (`cache_size=`); before, it
  grew without bound.
- **Large corpora.** `docrooter doctype` parses, classifies and writes `--chunk-size` files at a
  time (default 32). `--max-pages N` (`load_pdf(max_pages=N)`) skips pages past N and adds
  `"truncated_pages"` to `meta["warnings"]`. The pdfplumber engine frees each page after it is
  read.

## Speed and the Rust decision

The v0.3 spec (section 6) allows a Rust (PyO3) rewrite only if our own Python code takes at
least 30% of the per-document time of the default local pipeline **and** a hot loop is clearly
isolable. The decision is **no Rust**: our own Python is 24% before the Q5 optimisations and
13% after them, and what remains is spread over several small loops.

### How it was measured

`python -m benchmarks.profile_pipeline` (pyinstrument 5.1, 1 ms sampling) runs the default local
pipeline on ten dev PDFs from 1 to 642 pages (1,439 pages in all):

1. `load()` with the default engine (pymupdf);
2. `classify_document` against `builtin:doc_types` with UltraBERT-Embed-150M on MPS;
3. `normalize_sections` for the predicted type.

- **Model handling.** The model is loaded once, before any timing (10.5 s). A warm-up pass over
  the ten documents fills the label-vector caches, and the text-embedding cache is cleared
  before every timed document. This is the steady state of a long-running process such as
  `docrooter serve`.
- **Two runs per document.** The first is unprofiled and gives the wall time and the stage
  times from the pipeline events. The second runs under pyinstrument. Each sample goes to one
  bucket: pymupdf, tokenisation or the model if its stack is inside them, otherwise the
  innermost docrooter stage.
- **Our own Python** is every sample outside pymupdf, a tokenizer, torch, transformers and
  sentence-transformers.
- **Profiler bias.** pyinstrument hooks every Python call, which inflates call-heavy Python.
  Profiled runs took 12% longer than unprofiled ones. The `headings` stage took 0.44 s by its
  events but 1.35 s under the profiler. The shares below therefore overstate our own code. From
  the unprofiled stage times (signals, headings, assembly and view, plus line assembly
  estimated from the profile), our own Python is roughly 9% of the time after the
  optimisations and roughly 19% before.
- **Environment.** Measured on 2026-10-17: Apple M1, 16 GB, macOS 26.5, Python 3.12.13,
  torch 2.14.1, sentence-transformers 6.1.0, transformers 5.20.0, pymupdf 1.28.2, with the
  benchmark defaults (4 threads, 8 GiB budget). The load average was 2.7 / 3.5 / 3.7 (1/5/15
  min) at the start of the "before" run and 2.5 / 3.2 / 3.5 at the start of the "after" run.
- **Concurrent server.** `docrooter serve` was running during both runs, because the user was
  testing the playground. Its model had been released (resident set about 85 MB), so the runs
  were started with `--allow-concurrent`.
- **Code versions.** "Before" is `src/` at `f3baa43` and "after" is `70a9a0f`. Raw rows are in
  `benchmarks/results/2026-10-17/profile-before-q5.json` and `profile.json`.

### Where the time goes (all ten documents)

| Bucket | Before: s | Before: share | After: s | After: share |
| --- | ---: | ---: | ---: | ---: |
| pymupdf (C, `get_text`) | 12.71 | 40.8% | 12.83 | 46.9% |
| our line assembly | 1.05 | 3.4% | 1.16 | 4.2% |
| signals | 3.57 | 11.5% | 0.54 | 2.0% |
| headings | 2.31 | 7.4% | 1.35 | 4.9% |
| tree assembly | 0.03 | 0.1% | 0.03 | 0.1% |
| view rendering, keyphrases | 0.28 | 0.9% | 0.28 | 1.0% |
| tokenisation | 0.33 | 1.0% | 0.37 | 1.3% |
| model inference | 10.66 | 34.3% | 10.65 | 38.9% |
| other | 0.18 | 0.6% | 0.18 | 0.7% |
| **total (profiled)** | 31.12 | 100% | 27.39 | 100% |
| **our own Python** | 7.42 | **23.9%** | 3.54 | **12.9%** |

Unprofiled wall time for the ten documents fell from 27.5 s to 24.5 s (-11%). Of the model
time, `normalize_sections` takes 8.9 s and the document-type step takes 1.9 s.

| Doc | Pages | Wall s before | Wall s after | pymupdf | model (inference + tokenisation) | our Python before | our Python after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| doc_017 | 1 | 0.55 | 0.51 | 23% | 69% | 14% | 7% |
| doc_043 | 4 | 0.39 | 0.28 | 18% | 74% | 9% | 8% |
| doc_037 | 9 | 0.44 | 0.43 | 23% | 70% | 12% | 7% |
| doc_027 | 15 | 0.79 | 0.77 | 26% | 66% | 15% | 8% |
| doc_041 | 39 | 1.47 | 1.44 | 21% | 72% | 12% | 7% |
| doc_045 | 79 | 1.54 | 1.41 | 47% | 34% | 26% | 19% |
| doc_002 | 130 | 3.52 | 2.87 | 51% | 43% | 24% | 6% |
| doc_031 | 194 | 2.98 | 2.55 | 58% | 28% | 29% | 14% |
| doc_039 | 326 | 7.64 | 6.98 | 40% | 49% | 20% | 11% |
| doc_042 | 642 | 8.21 | 7.23 | 59% | 23% | 30% | 18% |

(pymupdf and model shares are the "after" run's.)

### The optimisations (Q5, `70a9a0f`)

All four are pure Python. Each leaves every document unchanged: the `Document` JSON (timings
aside) of all 51 dev PDFs is byte-identical with both engines, and the TOC benchmark's
per-document results are identical for `pymupdf`, `pdfplumber` and `both`.

- **The contents-leader regex.** `_TOC_LEADERS` (`^(.*?[^\W\d_].*?)\s*(?:[.·…_]\s?){5,}$`)
  backtracks quadratically on every long line that fails it: 2.3 s on doc_042's 47,743 lines
  and 1.8 s on doc_039. It now runs only when the last 10 characters end in five leaders. That
  is a necessary condition, and a unit test checks it on 20,000 random strings. This was the
  only real hot spot: `signals` went from 3.38 s to 0.35 s in the unprofiled stage times.
- **Heading candidates.** `_is_candidate` tests size and isolation before it counts letters,
  so most body lines are rejected without the letter count.
- **Fuzzy matching.** `is_fuzzy_match` rejects on `SequenceMatcher.real_quick_ratio()` and
  `quick_ratio()` (upper bounds of `ratio()`) before computing the ratio.
- **Memoised titles.** `normalize_title` is memoised, so bookmark and contents titles are not
  re-normalised for every candidate.

Our own Python time fell by 52% (7.42 s to 3.54 s profiled), well above the 15% bar.

### Why not Rust

- **Below the threshold.** Our own Python is under the 30% threshold in aggregate (12.9%) and
  in every document (at most 19%, doc_045). Before the optimisations it was 23.9% in
  aggregate, and only doc_042 touched 30%. That figure came from the profiler, which overstates
  Python, and the regex fix removed most of it.
- **No isolable hot loop.** What remains is spread over line assembly from pymupdf's dict
  (1.2 s), heading detection (1.4 s, of which the contents matcher is about 0.6 s and isolation
  checks about 0.2 s) and signals (0.5 s). Each is under 5% of the time. Moving all of them to
  Rust would save at most about 13% (profiled), or about 9% of wall time, and would add a
  compiled extension to a package that is pure Python today.
- **The real costs are elsewhere.** pymupdf's C text extraction (47%) and the model (40%) take
  the time. Rust cannot speed up either from our side. The levers there are batching and model
  choice, not a rewrite.

### Load speed (the v0.2 criterion)

The v0.2 spec asks `load()` with the default engine to take under 1.5 s per 100 pages. On the
TOC benchmark's 24 bookmarked documents (2,649 pages, `use_bookmarks=False`, one untimed
warm-up) at `70a9a0f`:

- **Benchmark run:** **1.05 s/100 pages** (27.8 s), load average 3.2 / 2.4 / 2.4.
- **A/B passes:** two back-to-back passes per version measured 1.34 s/100 pages before the Q5
  optimisations (`f3baa43`) and 1.05-1.06 s/100 pages after them (load 2.4-3.4).

The criterion is met. On the profiling set, the four load stages took 13.5 s for 1,439 pages
(0.94 s/100 pages), and almost all of it is pymupdf's `get_text("dict")`.
