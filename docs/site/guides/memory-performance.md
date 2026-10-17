# Memory and performance

docrooter is built to run on a laptop next to other work. Model backends stay within a
**memory budget of 8 GiB** by default, use the GPU when there is one, and release their memory
when you are done. This page covers the settings; the measurements are in
[Performance and memory](../benchmarks/performance.md).

## Process-wide settings

Set limits once, before the first model loads:

```python
# or set DOCROOTER_MEMORY_BUDGET_GB=8 and DOCROOTER_THREADS=4
docrooter.configure(memory_budget_gb=8, threads=4)
```

| `configure(...)` | Environment variable | Default | Effect |
| --- | --- | --- | --- |
| `memory_budget_gb` | `DOCROOTER_MEMORY_BUDGET_GB` | 8 | The memory the whole process should stay within, in GiB |
| `device` | `DOCROOTER_DEVICE` | CUDA, then MPS, then CPU | The torch device for backends that are not given one |
| `threads` | `DOCROOTER_THREADS` | torch's default | `torch.set_num_threads` |
| `mps_memory_fraction` | `DOCROOTER_MPS_MEMORY_FRACTION` | derived from the budget | An explicit Metal memory fraction in (0, 1] |

`configure` beats the environment; a backend's own `device=` beats both. Settings apply at the
next model load.

## What the budget does

- **Apple silicon (MPS).** Metal and the host share memory, so Metal gets a hard cap of the
  budget minus 1.5 GiB of headroom (at least half the budget), applied with
  `torch.mps.set_per_process_memory_fraction`. If `configure` runs before torch is imported,
  the `PYTORCH_MPS_HIGH_WATERMARK_RATIO` / `PYTORCH_MPS_LOW_WATERMARK_RATIO` variables are set
  first so the cap holds from the first allocation. Watermark variables you set yourself are
  never overridden. The cap covers Metal memory, not the whole process; the headroom is what
  keeps the total within the budget.
- **CPU.** Nothing caps the resident set. The budget only scales the default batch size.
- **Batch size.** 32 on CUDA. Otherwise 16 at a budget of 8 GiB or more, halved for each halving
  below that, never under 4.
- **Sequence length.** UltraBERT reads at most 1,024 tokens per text, and `train()` truncates to
  `max_seq_len=512` and fine-tunes with gradient checkpointing.
- **Embedding cache.** An LRU of 50,000 texts per embedding backend.

## Falling back from MPS to the CPU

If a model still runs out of MPS memory while loading or encoding, it moves to the CPU once,
with the same weights (fine-tuned weights are kept), and logs a warning. Fine-tuning first
retries from the base weights with half the batch, then falls back to the CPU. The
`model_load` [event](../concepts/events.md) reports the device chosen and the device used.

To force the CPU, for example on a machine whose GPU is busy:

```bash
DOCROOTER_DEVICE=cpu docrooter doctype inbox/
```

## Releasing models

Every backend has `close()`. A `Classifier` used in a `with` block closes the models it loaded
from spec strings when the block ends:

```python
with docrooter.Classifier(taxonomy) as clf:  # the model is released when the block ends
    results = clf.classify(doc)
```

`classify_document` and `normalize_sections` close what they load from a spec string before
returning. That is convenient for one document and wasteful for many: for a batch, build one
`Classifier` (or pass a backend instance) and reuse it, as in
[Document-type gating](doctype-gating.md#from-python). A `TrainedModel` closes its body too;
save it first if it was fine-tuned in memory.

The script is [`examples/memory.py`](../../../examples/memory.py).

## Large corpora

```bash
docrooter doctype big/ --workers 4 --chunk-size 16 --max-pages 200
```

`--chunk-size` bounds how many parsed documents are held at once, `--max-pages` (or
`load_pdf(max_pages=...)`) skips the pages past N, and `--workers` parses PDFs in parallel
processes. The pdfplumber engine frees each page after reading it.

## Measured

On an Apple M1 with 16 GB (the report explains which rows predate the budget and how the
total is measured):

| Case | Device | Peak process memory | Time |
| --- | --- | ---: | ---: |
| Load and type a 642-page datasheet (UltraBERT) | MPS | 1.91 GiB | 17 s |
| Same | CPU | 1.13 GiB | 24 s |
| Fine-tune on 306 examples (`train()`, k=16 per leaf) | MPS | 7.59 GiB | 17.8 min |
| Same | CPU | 6.23 GiB | 24.2 min |

Every case stayed within the 8 GiB budget. Where the time goes in the default pipeline: PDF
text extraction in PyMuPDF's C code (about 47%) and model inference (about 39%); docrooter's own
Python is about 13%, which is why it is not being rewritten in Rust. Loading a PDF with the
default engine takes about 1.05 s per 100 pages. See
[Performance and memory](../benchmarks/performance.md) for the method and the per-document
tables.

## Tips

- Use one long-lived process (or the playground) for many documents: the model loads once
  (about 10 s for UltraBERT) and its label vectors are cached.
- Type documents from their views (`mode="view"`, the default); `sections` and `hybrid` encode
  every section too.
- On a shared machine, lower the budget (`DOCROOTER_MEMORY_BUDGET_GB=4`) rather than the number
  of documents: it shrinks the batches and the Metal cap.
