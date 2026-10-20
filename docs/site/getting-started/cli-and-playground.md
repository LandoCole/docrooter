# Command line and playground

Everything in the [quickstart](quickstart.md) is also available without writing Python: a
`docrooter` command for scripts and pipelines, and a local web page for looking at one document
in detail. The [CLI reference](../reference/cli.md) lists every option.

## The command line

`-b/--backend` picks the backend on every command that classifies. The examples use `keyword`
so that they run without models; leave it out to use the default local model, or pass `jev`.

**Document type and section tags, sections first** (the v1 flow; see
[Sections first](../guides/sections-first.md)):

```bash
docrooter classify report.md -b keyword                  # type, how it was decided, tags, tokens
docrooter classify report.pdf --level headings --json    # the cheapest rung, as JSON
docrooter classify report.pdf -b openai:gpt-4o-mini --log decisions.jsonl
```

`--level` takes `headings`, `terms` (the default), `lead` or `full`; `--flow doc_first`
classifies the type first; `--calls` says how an LLM is asked (`batched`, `batched_flat` or
`per_item`).

**Table of contents.**

```bash
docrooter toc report.pdf                     # indented headings; a summary line on stderr
docrooter toc report.pdf --json              # id, title, level and depth per heading
docrooter toc report.pdf --engine pdfplumber # choose the PDF engine
```

```text
A sparse attention method
  Abstract
```

**Document type** for one file or a whole folder (searched recursively):

```bash
docrooter doctype inbox/ -b keyword
```

```text
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┓
┃ File             ┃ Type                         ┃ Confidence ┃ Parse s ┃ Classify s ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━┩
│ inbox/lease.md   │ Legal > Legislation          │ 0.07       │ 0.00    │ 0.000      │
│ inbox/paper.md   │ Research > Research paper    │ 0.07       │ 0.00    │ 0.000      │
│ inbox/results.md │ Financial > Earnings release │ 0.10       │ 0.00    │ 0.000      │
└──────────────────┴──────────────────────────────┴────────────┴─────────┴────────────┘
```

(The keyword backend calls the lease "legislation"; a real model is needed for real documents.)
`--keep financial --paths-only` turns this into a filter for other tools; see
[Document-type gating](../guides/doctype-gating.md).

**Canonical section labels:**

```bash
docrooter normalize paper.md -b keyword
docrooter normalize report.pdf --doc-type financial.annual_report --json
```

**Your own taxonomy:**

```bash
docrooter taxonomy validate taxonomy.csv              # check the file first
docrooter classify report.md -t taxonomy.csv -b keyword
docrooter classify report.md -t taxonomy.csv -o result.json   # table + JSON file
```

**Training and evaluation** (needs `docrooter[models]`):

```bash
docrooter eval -t taxonomy.json -d labelled.jsonl -b keyword --format md
docrooter train -t taxonomy.json -d labelled.jsonl -o bundle/
docrooter eval -t taxonomy.json -d labelled.jsonl -b trained:bundle/
```

**Element tags and your own tagger** (experimental; see [Train your own element tagger](../guides/train-your-tagger.md)):

```bash
docrooter corpus build corpus/ --from-manifest manifest.jsonl --pdf-dir pdfs/
docrooter train-elements corpus/ --stage A --out bundles/tagger-a --teacher mlx:mlx-community/Qwen3-4B-Instruct-2507-4bit@50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b
docrooter tag report.pdf --bundle bundles/tagger-a
```

Errors are printed as `error: ...` with exit status 1, so scripts can stop on them.

## The playground

```bash
pip install 'docrooter[ui]'
docrooter serve --open             # http://127.0.0.1:8765
```

Drop a PDF, Markdown or text file on the page, choose a backend and a PDF engine, and watch
each stage run with its timing: reading pages, signals, headings, building the tree, rendering
the view, loading the model, classifying and normalising. The result shows the table of contents
as a tree with heading scores, the document type with the probabilities of the five likeliest
types, and (unless you untick the option) a canonical label for each section. A raw JSON tab
has a download button, and recent runs stay in the sidebar while the server runs.

It is a local tool: it listens on `127.0.0.1` by default, loads no external assets, caps uploads at
50 MB and deletes each upload after processing. See [The playground](../guides/playground.md)
for the options, the API it is built on, and its limits.
