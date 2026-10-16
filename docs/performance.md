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
