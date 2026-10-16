# Performance

## Memory

docrooter works within a memory budget, 8 GiB by default (`docrooter.configure(memory_budget_gb=...)`
or `DOCROOTER_MEMORY_BUDGET_GB`). On Apple silicon the budget caps Metal memory through
`torch.mps.set_per_process_memory_fraction(budget / recommended_max_memory)`, with the
`PYTORCH_MPS_*_WATERMARK_RATIO` variables set from it before torch loads. On the CPU nothing caps
the resident set; the budget only scales the default batch sizes. A model that still runs out of
MPS memory moves to the CPU once, with a logged warning; fine-tuning first retries with half the
batch.

`python -m benchmarks.memory_check` measures two cases, each in a fresh process:

- **classify:** `load` + `classify_document` on doc_042, the 642-page RP2040 datasheet (1,239
  sections), with UltraBERT-Embed-150M against `builtin:doc_types`.
- **train:** `train()` on B1 (EDGAR 10-K items) with k=16 per leaf (306 examples, 596 triplets),
  `finetune_body=True`, UltraBERT-Embed-150M base, the defaults `max_seq_len=512`, batch 16 and
  gradient checkpointing.

Peak RSS is `ru_maxrss`. A sampler thread records the peaks of
`torch.mps.current_allocated_memory()` (tensors) and `driver_allocated_memory()` (everything
Metal holds, including the allocator's cache). Target: both peaks at most the 8 GiB budget.

Measured on 2026-10-16: Apple M1, 16 GB, macOS 26.5, Python 3.12, torch 2.14.1,
sentence-transformers 6.1.0. Benchmark defaults applied: 4 threads, `TOKENIZERS_PARALLELISM=false`.

| Case | Device | Peak RSS | Peak MPS allocated | Peak MPS driver | Time | Within 8 GiB |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| classify | MPS (default) | 0.77 GiB | 0.56 GiB | 1.14 GiB | 17 s | yes |
| train | MPS (default) | 1.13 GiB | 4.51 GiB | 7.02 GiB | 18.5 min | yes |
| classify | CPU | 1.13 GiB | n/a | n/a | 24 s | yes |
| train | CPU | 6.23 GiB | n/a | n/a | 24.2 min | yes |

The MPS cap for the runs was `0.6757` (8 GiB of the 11.84 GiB that Metal recommends on this
machine). Training on MPS stayed on MPS at batch 16, with no fallback. Both runs predicted
`technical.user_manual` for doc_042 at confidence 0.33. The CPU rows were measured with
`--device cpu` before the budget existed (under fixed 0.5 MPS caps, which do not affect a CPU
run). They were re-checked against the 8 GiB target. Raw rows are in
`benchmarks/results/2026-10-16/memory.json`.

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
