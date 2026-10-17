# Document-type gating

A common first step in a document pipeline is to sort what comes in: send annual reports to one
extractor, contracts to another, and drop the rest. `docrooter doctype` does this for a folder,
and its output is built to drive shell pipelines.

## From the shell

```bash
docrooter doctype inbox/                                 # a table: file, type, confidence, timings
docrooter doctype inbox/ --keep financial --paths-only   # only the financial files, one per line
docrooter doctype inbox/ --jsonl types.jsonl             # also one JSON record per file
```

- Folders are searched recursively for `.pdf`, `.md` and `.txt` files, skipping hidden folders,
  `node_modules` and `__pycache__`.
- `--keep` takes a type id or a group id and can be repeated: `--keep financial` keeps all six
  financial types, `--keep legal.contract --keep legal.terms_policy` keeps two leaves.
- `--paths-only` prints only the kept paths, so the type gates whatever comes next:

```bash
docrooter doctype inbox/ --keep financial --paths-only > financial.txt
while read -r f; do docrooter normalize "$f" --json; done < financial.txt
```

### The JSONL record

`--jsonl FILE` writes one record per file (with `--keep`, per kept file):

```json
{"source": "inbox/lease.md", "type": "legal.legislation", "label": "Legislation",
 "path": ["legal", "legal.legislation"], "confidence": 0.072,
 "leaf_probs": {"legal.legislation": 0.072, "legal.contract": 0.070, "...": 0.0},
 "timings": {"parse": 0.0003, "classify": 0.0002}, "error": null}
```

(Shortened and wrapped; each record is one line. This one is from the keyword backend, which
mistakes the lease for legislation.) A file that fails to parse gets `"error"` and no type
instead of stopping the run (with `--keep` it is reported on stderr and left out). A PDF
without a text layer gets no type either.

With `jq`, for example, to list the confident financial documents:

```bash
jq -r 'select(.path[0] == "financial" and .confidence >= 0.5) | .source' types.jsonl
```

### Large folders

```bash
docrooter doctype big/ --workers 4 --chunk-size 16 --max-pages 200
```

- `--workers N` parses PDFs in N processes (default: up to 4).
- `--chunk-size N` parses, classifies and writes N files at a time and then drops them, so
  memory does not grow with the number of files (default 32).
- `--max-pages N` reads only the first N pages of each PDF. The type is usually clear from the
  first pages, and this bounds time on 800-page filings.

### Choosing the backend and the threshold

The default backend is the local UltraBERT model. `-b jev` is the most accurate (see below);
`-b keyword` is only for trying the command. `--mode sections` or `--mode hybrid` also scores
the section headings; on the development corpus hybrid helped UltraBERT (0.824 vs 0.765) but
hurt it slightly on the held-out set (0.775 vs 0.825), so the default stays `view`.

`--threshold` (default 0) is the minimum confidence to assign a type at all; below it a file is
reported as `(low_confidence)` and never kept. For gating, it is usually better to keep the
threshold at 0, write `--jsonl`, and decide per type what confidence you trust.

## From Python

Build one classifier for all documents, so a model loads once and the documents are classified
in one batch. The script is [`examples/doctype_gate.py`](../../../examples/doctype_gate.py).

```python
paths = sorted(p for p in folder.rglob("*") if p.suffix in {".pdf", ".md", ".txt"})
docs = [load(path) for path in paths]

with Classifier(builtin_taxonomy("doc_types"), "keyword") as classifier:
    results = classifier.classify_documents(docs, threshold=0.0)

for path, result in zip(paths, results, strict=True):
    node_ids = result.best.node_ids  # e.g. ["financial", "financial.earnings_release"]
    keep = bool(node_ids) and node_ids[0] == "financial"
    print(f"{'keep' if keep else 'skip'}  {path.name}: {result.best.labels}")
```

```text
skip  lease.md: ['Legal', 'Legislation']
skip  paper.md: ['Research', 'Research paper']
keep  results.md: ['Financial', 'Earnings release']
```

For a single document, `classify_document(doc, builtin_taxonomy("doc_types"))` does the same
and releases the model it loaded before returning.

## How accurate is it?

On the 51-PDF development corpus and the 40-PDF held-out corpus (18 types, 2-4 documents
each):

| System | Dev | Held-out |
| --- | ---: | ---: |
| `jev` on the default `DocumentView` | 0.961 | 0.925 |
| UltraBERT zero-shot (the default) | 0.765 | 0.825 |
| bge-small zero-shot | 0.608 | 0.700 |

Weak spots for the local models are investor presentations and white papers; for every system,
image-only scans (no text, no OCR). Details are in the
[held-out evaluation](../benchmarks/heldout.md#document-type-generalises). With classes this
small, one document moves accuracy by 2-2.5 points.

## Your own document types

`-t` takes any taxonomy file instead of `builtin:doc_types`:

```bash
docrooter doctype inbox/ -t my_types.json --keep invoices --paths-only
```

Describe each type by what is *in* such a document (the headings, the vocabulary), since that
is what the view shows the model. See [Custom taxonomies](custom-taxonomies.md).
