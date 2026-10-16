# docrooter benchmarks

Three public datasets, one shared view (`SectionView(fields=("text",), max_tokens=512)`), and a
set of named systems (backend specs). Results are JSON; `docs/benchmarks.md` is generated from them.

| Key | Dataset | Labels | Notes |
| --- | --- | --- | --- |
| `b1` | EDGAR-CORPUS 10-K filings, year 2020 | Part, then Item (24 nodes) | Items under 50 words are dropped; the leading `ITEM 7A. ...` heading is stripped so it cannot leak the label |
| `b2` | `DeveloperOats/DBPedia_Classes` test split | `l1` > `l2` > `l3` (219 leaves) | CamelCase labels split into words |
| `b3` | `coastalcph/lex_glue`, config `ledgar`, test split | 100 flat clause types | `b3-subset` is a fixed 300-example subset every system is also scored on |

EDGAR-CORPUS ships a loading script that current `datasets` releases refuse, so `b1` streams the raw
`2020/train.jsonl` instead (columns `filename`, `cik`, `year`, `section_1` ... `section_15`).

## Running

```bash
uv sync --extra models --extra bench
uv run python -m benchmarks.run --suite zero-shot --device mps --n-test 1000
uv run python -m benchmarks.report          # writes docs/benchmarks.md
```

`--only SYSTEM` (repeatable) reruns just those systems and updates their rows in place, e.g.
`--only deberta-nli --device cpu`. `--datasets b1,b2` limits the benchmarks. Results go to
`benchmarks/results/<date>/<suite>.json` and are rewritten after every system x dataset, so a
crashed run keeps its finished rows.

### Few-shot

```bash
uv run python -m benchmarks.run --suite few-shot --device mps --shots 4,8,16,64 \
    --bases ultrabert-embed,bge-small
```

For each benchmark, base and k, two models are trained on `pool_shots(train_pool, k, seed=0)`:
`<base>/heads` (`finetune_body=False`) and `<base>/finetuned`, then scored with Beam(3) and
threshold 0.0. Rows carry `shots`, `train_seconds` and `n_train` (the pools hold at most 20
examples per leaf, so k=64 trains on the whole pool). Bundles are kept under
`benchmarks/.cache/bundles/<dataset>/<base>/<k>/<variant>/` (git-ignored) and reused; delete a
directory to retrain it. Fine-tuning at the 512-token view is memory-hungry: on a 16 GB machine batch 32 (the recipe
default) runs out of memory even for bge-small, so use `--train-batch-size 4` (or less) and
`--train-max-triplets N` to cap the fine-tuning set. Either option gives the bundle its own
directory (`finetuned-bs4-mt2000`) and is recorded in each row's `env`.
Here `--only` takes a base name (`--only bge-small`) or one variant (`--only bge-small/heads`).

### Strategies

```bash
uv run python -m benchmarks.run --suite strategies --device mps
```

B2 only: Greedy, Beam(3) and Flat (embedding backends only) with the best zero-shot system, and
Greedy and Beam(3) with the best `--trained-shots` (default 16) trained model. "Best" is the
highest B2 hF1 in `zero-shot.json` / `few-shot.json` of the same `--out` directory; override with
`--zero-shot-system NAME` and `--trained-system BASE/VARIANT`.

Systems (`benchmarks/systems.py`): `ultrabert-embed`, `bge-small`, `deberta-nli` (B3 subset only),
`ultrabert-mlm` (B1 and B2 only), `openai` (needs `DOCROOTER_BENCH_LLM_URL` and
`DOCROOTER_BENCH_LLM_MODEL`) and `jev` (needs `TYPESAFE_API_KEY`). Systems whose backend module
is missing are skipped with a log line.

Each row stores the full `Report`, the top-level `macro_p/r/f1`, `sections_per_sec`, `seconds_total`
and `seconds_per_example` (the timed classification pass; model loading is excluded), plus
`wall_seconds` for the whole cell and an `env` block with library versions and model revisions.
A `decisions.md` next to the results file is appended to `docs/benchmarks.md`.
