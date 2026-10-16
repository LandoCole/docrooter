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
`--only deberta-nli --device cpu`. `--datasets b1,b2` limits the benchmarks. The few-shot and
strategies suites are not implemented yet. Results go to `benchmarks/results/<date>/<suite>.json`
and are rewritten after every system x dataset, so a crashed run keeps its finished rows.

Systems (`benchmarks/systems.py`): `ultrabert-embed`, `bge-small`, `deberta-nli` (B3 subset only),
`ultrabert-mlm` (B1 and B2 only), `openai` (needs `DOCROOTER_BENCH_LLM_URL` and
`DOCROOTER_BENCH_LLM_MODEL`) and `jev` (needs `TYPESAFE_API_KEY`). Systems whose backend module
is missing are skipped with a log line.

Each row stores the full `Report`, the top-level `macro_p/r/f1`, `sections_per_sec`, `seconds_total`
and `seconds_per_example` (the timed classification pass; model loading is excluded), plus
`wall_seconds` for the whole cell and an `env` block with library versions and model revisions.
A `decisions.md` next to the results file is appended to `docs/benchmarks.md`.
