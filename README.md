# docrooter

docrooter extracts a table of contents from Markdown and PDF documents and classifies every
section against a taxonomy you supply: a tree of categories described in JSON or CSV. It runs
zero-shot out of the box (small embedding models run on a laptop; LLM backends are optional),
returns probabilities and the full path through the tree for each section, and can
train per-node classifiers from a few labelled examples when zero-shot is not enough. The core
install is small (`pydantic`, `numpy`, `typer`, `rich`); models, PDF parsing and LLM backends
are extras.

## Install

```bash
pip install docrooter              # core: Markdown loading, taxonomies, the keyword backend
pip install 'docrooter[models]'    # + local models (embedding, NLI, MLM) and training
```

| Extra | Adds | Needed for |
| --- | --- | --- |
| *(none)* | `pydantic`, `numpy`, `typer`, `rich` | Markdown/text loading, taxonomies, the `keyword` backend, the CLI |
| `models` | `torch`, `transformers>=5`, `sentence-transformers>=6`, `scikit-learn`, `joblib`, `datasets` | `embedding`, `nli`, `mlm-choice` and `trained` backends; `docrooter train`; `docrooter models` |
| `pdf` | `pymupdf4llm` | Loading `.pdf` files (see the licence note below) |
| `llm` | `httpx` | The `openai` backend (any OpenAI-compatible server) |
| `jev` | `typesafe-sdk` | The `jev` backend (TypeSafe Jev) |
| `bench` | `datasets`, `pandas` | The benchmarks, `DocumentResult.to_dataframe()` |
| `all` | `models` + `pdf` + `llm` + `jev` | Everything above except `bench` |

Python 3.10 or newer. Asking for something an extra provides without installing it fails with
the exact command to run, e.g. `pip install 'docrooter[models]'`.

**PDF and the AGPL.** PDF support uses PyMuPDF through `pymupdf4llm`, which is licensed under
the AGPL-3.0, unlike docrooter itself (Apache-2.0). It is installed only by the `[pdf]` extra
(and so by `[all]`) and imported only when a `.pdf` file is loaded. Read the AGPL terms before
you distribute software or run a network service that includes it.

**`trust_remote_code` and UltraBERT.** The default model, `dmis-lab/UltraBERT-Embed-150M`
(and `UltraBERT-380M`, used by `mlm-choice`), ships its own modelling code, which
transformers only runs with `trust_remote_code=True`. docrooter turns that on only for models
whose built-in recipe pins a commit (`367d5343b5a0164d57b810cedf5de4b8983d5984` for
UltraBERT-Embed-150M, `65bbbfeefb03bc93b9965d1d511708dabcee8511` for UltraBERT-380M), so the
code that runs is the code at that commit and not whatever the repository holds later. For any
other model that needs remote code you have to opt in yourself, after reading the code:
`EmbeddingBackend("org/model", trust_remote_code=True)`. The first run downloads about 600 MB
for the default model.

## Quickstart

This example uses the dependency-free `keyword` backend, so it runs straight after
`pip install docrooter`. It lives in [`examples/quickstart.py`](examples/quickstart.py).

```python
from docrooter import Classifier, Taxonomy, load

MARKDOWN = """\
# Report

## Income tax
The tax rate rose.

## Auditor's report
An audit opinion.
"""

taxonomy = Taxonomy.from_dict(
    {
        "nodes": [
            {
                "id": "financial",
                "label": "Financial",
                "description": "Tax rates, audit opinions and financial results",
                "children": [
                    {"id": "tax", "label": "Tax", "description": "Income tax and the tax rate"},
                    {"id": "audit", "label": "Audit", "description": "The audit opinion"},
                ],
            },
            {"id": "legal", "label": "Legal", "description": "Contracts, courts and disputes"},
        ]
    }
)

doc = load(MARKDOWN)
classifier = Classifier(taxonomy, "keyword", threshold=0.0)  # the default backend is a model
for section in classifier.classify(doc).sections.values():
    print(f"{section.title}: {' > '.join(section.best.labels)}")
```

```text
Income tax: Financial > Tax
Auditor's report: Financial > Audit
```

`load` takes a path (`.md`, `.markdown`, `.txt`, `.pdf`) or Markdown text. `doc.toc()` lists the
headings with their depth, and `classify` returns a `DocumentResult` with a `best` path (node
ids, labels, score, and the decision made at each step) and the rival `alternatives` for every
section. `result.to_json()` serialises it and `result.summary()` gives the share of the document
in each top-level category.

Leave the backend out to use the default, UltraBERT embeddings
([`examples/embedding_quickstart.py`](examples/embedding_quickstart.py), needs `docrooter[models]`):

```python
from docrooter import Beam, Classifier

classifier = Classifier(taxonomy)  # zero-shot
classifier = Classifier(taxonomy, strategy=Beam(3), threshold=0.35)
```

The first line uses `"embedding:dmis-lab/UltraBERT-Embed-150M"` and the second spells out the
other defaults.

`Greedy()` follows the best child at each level, `Beam(n)` keeps the `n` best paths and `Flat()`
scores every leaf in a single call, by its full path. A path stops early, with a `stopped_reason`, when no child reaches the
threshold.

## Command line

```bash
docrooter toc report.md                                  # indented table of contents
docrooter toc report.pdf --json                          # as JSON (needs docrooter[pdf])

docrooter taxonomy validate taxonomy.csv                 # check a taxonomy file
docrooter taxonomy validate paths.csv --sep ">"          # path-style CSV with another separator

docrooter classify report.md -t taxonomy.json            # JSON on stdout, default backend
docrooter classify report.md -t taxonomy.json -b keyword --strategy greedy
docrooter classify report.md -t taxonomy.json -b jev -o result.json   # table + result.json

docrooter doctype docs/ -b keyword                       # document type per file (builtin:doc_types)
docrooter doctype docs/ --keep financial --paths-only    # only financial docs, one path per line
docrooter doctype docs/ --jsonl types.jsonl --workers 4  # parse PDFs in 4 processes
docrooter normalize report.md                            # canonical label for each section title
docrooter normalize report.md --doc-type research.research_paper --json

docrooter eval -t taxonomy.json -d labelled.jsonl -b keyword          # accuracy, hierarchical F1
docrooter eval -t taxonomy.json -d labelled.jsonl --format md

docrooter train -t taxonomy.json -d labelled.jsonl -o bundle/         # train heads and fine-tune
docrooter eval  -t taxonomy.json -d labelled.jsonl -b trained:bundle/ # then evaluate the bundle

docrooter models pull dmis-lab/UltraBERT-Embed-150M      # download for offline use
docrooter models list                                    # what is in the Hugging Face cache
```

`-b` takes a backend spec (see [Backends](#backends)); `--strategy` takes `greedy`, `beam:N` or
`flat`. Errors are printed as `error: ...` with exit status 1.

## Taxonomies

A taxonomy is a tree of categories, each with a label and optionally a description and
examples that zero-shot backends use. It can be written as nested JSON, flat JSON, or one of
three CSV layouts, and `Taxonomy.load` detects which. All five below describe the same two-level
taxonomy (shortened here); [docs/taxonomy-formats.md](docs/taxonomy-formats.md) has the full reference
(optional columns, per-node `threshold`, `multi_label` and `backend`, root settings, the
validation codes).

Nested JSON:

```json
{"nodes": [
  {"id": "financial", "label": "Financial", "description": "Money, accounting, results",
   "children": [
     {"id": "financial.tax", "label": "Tax", "description": "Income tax and deferred tax"},
     {"id": "financial.audit", "label": "Audit", "description": "Audit reports and findings"}
   ]},
  {"id": "legal", "label": "Legal", "description": "Legal matters"}
]}
```

Flat JSON (nodes linked by `parent_id`):

```json
{"nodes": [
  {"id": "financial", "label": "Financial"},
  {"id": "financial.tax", "label": "Tax", "parent_id": "financial"},
  {"id": "financial.audit", "label": "Audit", "parent_id": "financial"},
  {"id": "legal", "label": "Legal"}
]}
```

Adjacency CSV (`parent_id` column):

```csv
id,parent_id,label,description
financial,,Financial,"Money, accounting, results"
financial.tax,financial,Tax,Income tax and deferred tax
financial.audit,financial,Audit,Audit reports and findings
legal,,Legal,Legal matters
```

Levels CSV (`level_1`, `level_2`, ... columns):

```csv
level_1,level_2,description
Financial,,"Money, accounting, results"
Financial,Tax,Income tax and deferred tax
Financial,Audit,Audit reports and findings
Legal,,Legal matters
```

Path CSV (labels joined by `>`):

```csv
path,description
Financial,"Money, accounting, results"
Financial > Tax,Income tax and deferred tax
Financial > Audit,Audit reports and findings
Legal,Legal matters
```

Run `docrooter taxonomy validate FILE` before using a new file; it names the row or JSON pointer
of every problem.

## Training and evaluation

Zero-shot is a good start. If you have labelled sections, docrooter fits one small classifier
head per taxonomy node on top of a shared embedding model, and can fine-tune that model first.
Nodes that do not have enough examples fall back to the zero-shot model. The full script is
[`examples/train_and_eval.py`](examples/train_and_eval.py) (needs `docrooter[models]`).

**1. Labelled data.** A dataset is JSONL (`{"text": ..., "path": [...]}` per line) or CSV (`text`
and `path` columns, the path joined by `>`). A path lists node ids or labels from the top down.

```jsonl
{"text": "The effective tax rate was 21.4 percent.", "path": ["Financial", "Tax"]}
{"text": "The auditor issued an unqualified opinion.", "path": ["financial", "financial.audit"]}
```

**2. Bootstrap labels from a zero-shot run.** Classify some documents, keep the rendered text of
each section with its best path, then fix the mistakes by hand and train on the result:

```python
from docrooter import Classifier, Dataset

classifier = Classifier(taxonomy)  # zero-shot
result = classifier.classify(doc)
pseudo = Dataset.from_results(result, doc, classifier.view)
pseudo.save("pseudo.jsonl")  # review and correct, then load it back
```

**3. Split, train, save.**

```python
from docrooter import Dataset, train

dataset = Dataset.load("labelled.jsonl")
train_set, test_set = dataset.split(test_size=0.2)  # stratified by top-level category
model = train(taxonomy, train_set, finetune_body=False, min_examples_per_child=2)
model.save("bundle/")
```

This fits only the heads, which is fast. Leave out `finetune_body=False` to fine-tune the body
first (the default, and usually better with enough data). The CLI equivalent is
`docrooter train -t taxonomy.json -d labelled.jsonl -o bundle/ [--no-finetune-body]`. A node gets a
head only when each of its children has at least `min_examples_per_child` examples (default 4).

**4. Evaluate against zero-shot.**

```python
from docrooter import Classifier, TrainedModel, compare, evaluate

trained = Classifier(taxonomy, TrainedModel.load("bundle/"))  # or backend "trained:bundle/"
compare({"zero-shot": Classifier(taxonomy), "trained": trained}, test_set).print()
evaluate(trained, test_set).print()  # adds per-node accuracy and sibling confusions
```

Reports give exact-path accuracy, hierarchical precision/recall/F1, leaf macro-F1, the share of
sections that stopped early, selective accuracy and throughput. `docrooter eval` does the same
from the command line.

**5. Calibrate (optional).** `calibrate` returns a copy of the classifier whose backend
temperatures are tuned on a development set (hold one out of the training data), so the
probabilities, and with them the `threshold`, mean what they say.

```python
from docrooter import calibrate

calibrated = calibrate(trained, dev_set)
print(calibrated.calibration_report)
```

> **Security: trained bundles contain pickles.** A bundle stores its heads as `joblib` files,
> which are Python pickles, and loading a pickle can run arbitrary code. Only load bundles you
> trained yourself or got from a source you trust. The same goes for `trained:<path>` specs and
> `docrooter eval -b trained:...`.

## Backends

A backend is chosen by a spec string, `kind[:model[@revision]]`, or by passing an instance. The
classifier can use different backends at different nodes or depths (`overrides=`), per-node
`backend` settings in the taxonomy, and a `fallback`.

| Spec | How it scores | Needs | Notes |
| --- | --- | --- | --- |
| `keyword` | Word overlap with the label, description and examples | core | A baseline for tests and demos, not for real use |
| `embedding[:model]` | Embedding similarity | `models` | The default, `dmis-lab/UltraBERT-Embed-150M`; any sentence-embedding model works, e.g. `embedding:BAAI/bge-small-en-v1.5` |
| `nli[:model]` | Textual entailment | `models` | Default `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` |
| `mlm-choice[:model]` | A masked LM fills in the letter of the right option | `models` | Experimental; default `dmis-lab/UltraBERT-380M` |
| `trained:<dir>` | Per-node heads on a fine-tuned body | `models` | From `docrooter train` / `train()`; see the pickle warning above |
| `openai:<model>` | Chat model answers with an option number | `llm` | Any OpenAI-compatible endpoint, e.g. `openai:gpt-4o-mini`; the model is required |
| `jev[:model]` | TypeSafe Jev typed choice and yes/no probabilities | `jev` | Default `jev-1.13.0`; needs `TYPESAFE_API_KEY` |

Model-backed backends pick the device in the order CUDA, MPS, CPU. FlashAttention 2 is used only on
CUDA when it is installed; otherwise attention runs through `sdpa`.

### Custom backends

Any object with these members can be passed as a backend (or registered as a spec kind with
`docrooter.backends.register_backend_kind`); there is nothing to subclass:

- `name`: a string that identifies the backend, e.g. in results and calibrated temperatures.
- `max_candidates`: the most candidates one `score` call accepts, or `None` for no limit. A
  decision with more candidates goes to the next backend in the routing order.
- `temperature` (optional, default 1.0): applied to `similarity` scores and used to re-temper
  `softmax` scores. `calibrate()` tunes it per backend `name`.
- `score(texts, candidates, *, instruction=None[, multi_label])`: returns `Scores(values, kind)`
  with `values` of shape `(len(texts), len(candidates))`. `kind` is `"softmax"` (rows are
  probabilities), `"independent"` (one probability per candidate) or `"similarity"` (unbounded
  scores). Each candidate has `node_id`, `text` and `examples`. `multi_label=True` is passed only
  for decisions at multi-label nodes, so a backend without that argument works everywhere else.
- `supports(node_id)`: whether the backend can decide at this node.
- `tokenizer()`: a function that counts the tokens of a text, or `None` to use an estimate.

```python
from docrooter.backends import Scores


class OverlapBackend:
    name = "overlap"  # keys calibrated temperatures; keep it unique per backend
    max_candidates = None  # or a limit: larger candidate sets go to the next backend

    def score(self, texts, candidates, *, instruction=None):
        rows = []
        for text in texts:
            words = set(text.lower().split())
            rows.append([len(words & set(c.text.lower().split())) for c in candidates])
        return Scores(np.array(rows, dtype=float), "similarity")

    def supports(self, node_id):
        return True

    def tokenizer(self):
        return None  # token counts are estimated


classifier = Classifier(taxonomy, OverlapBackend(), threshold=0.0)
```

The full script is [`examples/custom_backend.py`](examples/custom_backend.py).

### API keys

The `jev` backend reads `TYPESAFE_API_KEY`. The `openai` backend reads `DOCROOTER_LLM_API_KEY`
or `OPENAI_API_KEY`, and the server address from `DOCROOTER_LLM_BASE_URL` or `OPENAI_BASE_URL`
(the OpenAI API when unset; a local server that needs no key works without one). Keys are only
read from the environment (or an explicit `api_key=`) and are redacted from error messages.

docrooter does not read `.env` files itself. Keep keys in a git-ignored `.env` and hand them to
the process:

```bash
# .env  (never commit this file)
TYPESAFE_API_KEY=...

uv run --env-file .env python examples/jev_backend.py
# or: set -a; source .env; set +a
```

Never put a key in code, a taxonomy file or a committed config. See
[`examples/jev_backend.py`](examples/jev_backend.py).

### Offline use

Models come from the Hugging Face cache. Download them while you have a connection, then forbid
network access:

```bash
docrooter models pull dmis-lab/UltraBERT-Embed-150M     # the pinned revision, unless --revision
HF_HUB_OFFLINE=1 docrooter classify report.md -t taxonomy.json
```

A trained bundle is self-contained (the fine-tuned body is inside it) and loads offline.

## Benchmarks

The repository includes a benchmark suite (`benchmarks/`) that scores the backends zero-shot and
with few-shot training on three public datasets: SEC 10-K filings (EDGAR-CORPUS), DBpedia
classes and LEDGAR contract clauses. Results are being finalised and will be published in
[docs/benchmarks.md](docs/benchmarks.md), which is generated by `python -m benchmarks.report`.
See [benchmarks/README.md](benchmarks/README.md) for the setup and how to run them.

## Development

```bash
git clone https://github.com/docrooter/docrooter && cd docrooter
uv sync                         # core + dev tools; add --extra models for the model extras
uv run pytest                   # unit tests: fast, no model downloads, never import torch
uv run pytest -m models         # model tests: download models and need --extra models
uv run ruff check .             # lint
uv build                        # sdist and wheel in dist/
```

Tests marked `models` are deselected by default. The live `jev` test skips unless
`TYPESAFE_API_KEY` is set. For the benchmarks, add `--extra bench`.

## Licence

Apache-2.0 for docrooter. See [LICENSE](LICENSE). The optional `[pdf]` extra pulls in the
AGPL-3.0 PyMuPDF, as noted above.
