# docrooter v0.1 — Section Classification Design

- **Date:** 2026-10-15
- **Status:** Draft, awaiting review
- **Package name:** `docrooter` (available on PyPI as of 2026-10-15)

## 1. Purpose

`docrooter` is a Python library that breaks a document into its table of contents (TOC) and classifies **each section** against a user-defined **taxonomy**. A taxonomy can be flat or many levels deep. Classification runs on small local models (BERT-class encoders by default, small LLMs optionally) instead of sending whole documents to large LLMs. The package also provides training and evaluation so users can measure, then improve, accuracy on their own taxonomy without writing fine-tuning code.

### Target users

Python developers building document pipelines: routing, indexing, compliance review, RAG preprocessing, and content auditing.

### Success criteria

1. A user can go from `pip install "docrooter[models]"` to per-section labels for a Markdown or PDF document in under 10 lines of code, or with one CLI command.
2. Taxonomies from 1 level up to at least 20 levels deep, with nodes of hundreds of children, work without code changes and without recursion limits.
3. Users can define taxonomies in JSON or CSV.
4. Users can swap the model backend with one argument and can bring any compatible Hugging Face model.
5. One call trains a classifier from labelled examples. One call produces an evaluation report.
6. A reproducible benchmark answers which zero-shot and few-shot setup should be the default. Section 9 covers the benchmark, including UltraBERT against established baselines.

### Non-goals for v0.1 (parked)

- Entity extraction, MCP server, token compression, hosted service.
- Cross-validating PDF structure across multiple parsers. v0.1 has a single PDF path, and the adapter interface leaves room for more parsers.
- LLM fine-tuning (LoRA). LLMs are inference backends and label generators only.
- DAG taxonomies where one node has several parents. v0.1 supports trees only.
- Labelling UI, multilingual support, OCR.

## 2. Architecture overview

```
               ┌──────────────┐     ┌──────────────┐     ┌────────────────────┐
 .md / .pdf ──►│  Document    │────►│ SectionView  │────►│  Classifier        │──► Results
 or TOC JSON   │  (TOC tree)  │     │ (text sent   │     │  traversal over    │   (path, scores,
               └──────────────┘     │  to model)   │     │  Taxonomy, calling │    alternatives)
                                    └──────────────┘     │  Backend per node  │
 taxonomy.json/.csv ──► Taxonomy ───────────────────────►└────────────────────┘
                                                                  ▲
 labelled JSONL/CSV ──► Dataset ──► train() ──► TrainedModel ─────┘ (is itself a Backend)
                                       │
                         Dataset ──► evaluate(classifier) ──► Report
```

Each unit has one job and a narrow interface:

| Unit | Responsibility | Depends on |
|---|---|---|
| `document` | Parse input into a tree of `Section`s; validate and repair headings | stdlib only (PDF needs an optional extra) |
| `views` | Render a `Section` into model input text within a token budget | `document` |
| `taxonomy` | Load, validate, and query the label tree | stdlib, pydantic |
| `backends` | Score texts against a set of candidate labels | optional ML extras |
| `classify` | Walk the taxonomy for each section, batching backend calls | `taxonomy`, `backends`, `views` |
| `train` | Build per-node training sets and produce a `TrainedModel` backend | `taxonomy`, `backends`, the train extra |
| `evaluate` | Compute metrics and reports from a dataset | `classify` |
| `cli` | Thin Typer wrapper over the public API | everything |

The core install contains no PyTorch. Every ML dependency sits behind an extra and is imported lazily. A missing extra raises an error that names the exact `pip install` command.

## 3. Documents and the TOC

### Data model

```python
class Section(BaseModel):
    id: str                  # stable: "s3.2" style or hash
    title: str
    level: int               # 1 = top level after repair
    text: str                # body text excluding child sections
    parent_id: str | None
    children: list[str]
    page_start: int | None   # PDFs only
    heading_score: float     # validator confidence that the title is a real heading
    flags: list[str]         # e.g. "level_repaired", "low_heading_confidence"

class Document(BaseModel):
    source: str
    title: str | None
    sections: dict[str, Section]   # flat map; tree via parent_id/children
    root_ids: list[str]
    def walk(self) -> Iterator[Section]  # iterative, pre-order
    def toc(self) -> list[TocEntry]
    def section(self, id_or_number: str) -> Section
```

### Loaders

`dr.load(path_or_text)` dispatches on extension:

- **Markdown (core):** parses ATX (`#`) and setext headings. It ignores headings inside fenced code blocks. Text up to the next heading of the same or higher level becomes the section body.
- **PDF (`[pdf]` extra):** uses `pymupdf4llm` to produce Markdown, then reuses the Markdown parser. If the PDF has embedded bookmarks (`doc.get_toc()`), those titles and levels take precedence and override the levels guessed from Markdown. **License note:** PyMuPDF is AGPL-3.0. It is opt-in only, and the README states the implication.
- **Pre-built TOC (core):** `Document.from_sections([...])` lets users who already have structure, for example from Docling, skip parsing.

### Heading validation (rule-based in v0.1)

Each candidate heading gets a score in [0, 1] from weighted features. Headings below `min_heading_score` (default 0.5) become body text of the enclosing section. Features:

- Length in words and characters. Titles longer than about 20 words get a penalty.
- Ends with sentence punctuation (`.`, `,`, `;`).
- Purely numeric or a page-number pattern (`Page 3`, `3 of 12`).
- URL, email, or mostly symbols.
- The same title appears 3 or more times in the document, which suggests a running header or footer.
- Starts with a list bullet or looks like a table row.
- Numbering prefix present (`1.2`, `IV.`, `Article 5`, `Appendix A`). This increases the score.
- Title case or all caps. This slightly increases the score.

### Hierarchy repair

1. If numbering prefixes exist, derive the level from their depth (`1.2.3` → 3). This overrides `#` count, because `pymupdf4llm` often emits flat levels.
2. Close level gaps: an H1 followed by an H4 becomes H1 followed by H2.
3. Record each change in `Section.flags`.

## 4. Section views

A `SectionView` decides what text a backend sees for a section. It is configurable because the best input differs by backend: embedding models do well with a short focused summary, while a 32K-context encoder can take the full text.

```python
SectionView(
    fields=("path", "title", "keyphrases", "lead", "text"),  # order = render order
    max_tokens=512,  # budget for the whole rendered view
    lead_sentences=3,
    keyphrases=8,
    empty_section="children_titles",  # sections with no body: use child titles instead
)
```

Rendered example:

```
Path: Part II > Item 7 > Liquidity and Capital Resources
Title: Liquidity and Capital Resources
Key phrases: revolving credit facility, cash equivalents, debt maturities
Lead: As of December 31, we had $1.2 billion in cash...
Text: ...
```

- **Keyphrases** use a built-in extractor with no dependencies. Candidates are stopword-delimited n-grams of 1 to 3 tokens. Each is scored by term frequency in the section × inverse frequency across the document's other sections. This surfaces what makes a section distinctive rather than generic document vocabulary.
- **Lead sentences** are the first N sentences, using a regex sentence splitter.
- **Token counting** uses the backend's tokenizer when one is available, otherwise a whitespace estimate × 1.3. The view fills fields in order and truncates the last field that fits. A view that was truncated sets `truncated=True` on the result.
- `fields` can exclude `path` when the benchmark needs to stop TOC titles from leaking the answer (Section 9).

## 5. Taxonomy

### Data model

```python
class TaxonomyNode(BaseModel):
    id: str                      # unique across the taxonomy
    label: str                   # display name; also used as label text if no description
    description: str | None      # strongly recommended for zero-shot
    examples: list[str] = []     # optional short examples (zero-shot prompts + few-shot seeds)
    parent_id: str | None
    children: list[str] = []
    multi_label: bool = False    # may more than one child be chosen here?
    threshold: float | None      # min child probability to descend; None = inherit default
    backend: str | None          # per-node backend override
    metadata: dict = {}

class Taxonomy:
    nodes: dict[str, TaxonomyNode]   # flat map, a virtual root "__root__" holds top-level ids
    def children(self, node_id) -> list[TaxonomyNode]
    def path(self, node_id) -> list[TaxonomyNode]     # iterative
    def depth(self) -> int
    def leaves(self) -> list[TaxonomyNode]
    def validate(self) -> list[TaxonomyIssue]
```

Storing the tree as a flat map with iterative traversal means depth costs nothing structurally. Tests cover a 30-level synthetic taxonomy.

### File formats (auto-detected)

**JSON, nested:**

```json
{"nodes": [
  {"id": "fin", "label": "Financial", "description": "Money, accounting, results",
   "children": [
     {"id": "fin.tax", "label": "Tax", "description": "Income tax, deferred tax, tax risk"}
   ]}
]}
```

**JSON, flat:** a list of objects with `parent_id`.

**CSV.** The format is detected from the header row:

| Header contains | Interpretation |
|---|---|
| `id`, `parent_id`, `label` (+ optional `description`, `examples`, `threshold`, `multi_label`) | Adjacency list. Best for large or deep taxonomies. |
| `level_1`, `level_2`, … `level_N` (+ optional `description`) | One row per path, where empty trailing cells mean a shorter path. `description` applies to the deepest node in the row. |
| `path` (+ optional `description`) | One row per node, with the path joined by `>` (separator configurable). |

IDs are generated from the labels (slugified path) when absent.

### Validation

`Taxonomy.validate()` checks for:

- Duplicate IDs
- Missing parents
- Cycles
- Nodes with exactly one child, which are allowed but warned because they add a step with no decision
- Empty labels
- Sibling labels that are identical after normalisation

Errors include the source file row or JSON pointer. `dr.Taxonomy.load()` raises on errors and logs warnings.

## 6. Backends

### Interface

```python
class Candidate(NamedTuple):
    node_id: str
    text: str          # rendered label text: label, or "label: description"

class Scores(NamedTuple):
    values: np.ndarray           # shape (n_texts, n_candidates)
    kind: Literal["softmax", "independent", "similarity"]

class Backend(Protocol):
    name: str
    max_candidates: int | None              # None = unlimited
    def score(self, texts: Sequence[str], candidates: Sequence[Candidate],
              *, instruction: str | None = None) -> Scores: ...
    def supports(self, node_id: str) -> bool  # fixed-label backends: only trained nodes
    def tokenizer(self) -> Callable[[str], int] | None
```

All candidates in one `score` call are the children of a single taxonomy node. The classifier converts raw scores to probabilities:

- **`similarity`:** softmax with temperature T. The default T is set per backend recipe and can be overridden by calibration (Section 8).
- **`softmax`:** used as-is.
- **`independent`:** used as-is for multi-label nodes. For single-label nodes the scores are renormalised.

### Built-in backends

Backend spec strings are `kind:model[@revision]`. A Python object implementing `Backend` also works.

| Spec | What it does | Takes labels at runtime? | Extra |
|---|---|---|---|
| `embedding:<model>` | Bi-encoder. Section view encoded with an instruction prompt, candidates with a label prompt; cosine similarity. Default `dmis-lab/UltraBERT-Embed-150M`, which uses the `Instruct: {task}\nQuery: ` and `Label: ` prompts. | Yes | `models` |
| `nli:<model>` | Zero-shot NLI ("This text is about {label}."), entailment probability. Baseline: `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`. | Yes | `models` |
| `mlm-choice:<model>` | Masked-LM multiple choice. Options are listed with single-token markers (`A`–`Z`); the backend reads mask-token logits over the markers. `max_candidates=26`. **Experimental until the benchmark confirms it.** Default `dmis-lab/UltraBERT-380M`. | Yes | `models` |
| `trained:<path>` / `TrainedModel` | Fine-tuned body + per-node heads (Section 7). | No, only trained nodes | `models` |
| `openai:<model>` | Any OpenAI-compatible chat endpoint (Ollama, llama.cpp server, vLLM, hosted APIs). The prompt lists numbered options and requests the option number; logprobs over the option tokens are used when available, otherwise the parsed answer gets probability 1. `base_url` and `api_key` come from arguments or env vars. | Yes | `llm` |

When a node has more children than a backend's `max_candidates`, the classifier uses that node's fallback backend.

### Model recipes

A small built-in registry maps known model IDs to defaults, so users don't have to learn per-model quirks:

```python
RECIPES = {
  "dmis-lab/UltraBERT-Embed-150M": Recipe(
      revision="<pinned sha at implementation time>",
      trust_remote_code=True,
      query_prompt="Instruct: {instruction}\nQuery: ",
      label_prompt="Label: ",
      max_seq_len=8192,           # default cap; model supports 32768
      temperature=0.05,           # starting point, replaced by benchmark result
      finetune_lr=5e-5,           # authors' reported best for 150M
  ),
  "dmis-lab/UltraBERT-380M": Recipe(revision=..., trust_remote_code=True, finetune_lr=3e-5, ...),
  "BAAI/bge-small-en-v1.5": Recipe(query_prompt="Represent this sentence for searching relevant passages: ", ...),
  ...
}
```

Unknown models get generic defaults. Users can register their own recipes or pass any field as a keyword argument.

### Security

`trust_remote_code=True` runs Python code from the model repository. docrooter enables it only when:

- the model is in the recipe registry with a pinned revision, or
- the user passes `trust_remote_code=True` explicitly.

Otherwise loading fails with an explanatory error.

### Devices

Autodetect in the order CUDA, then MPS, then CPU. Use FlashAttention 2 when installed on CUDA, otherwise SDPA. Use float32 on CPU and MPS unless overridden. The dev machine is an Apple M1, so MPS and CPU are first-class targets for the benchmark.

## 7. Classification

### Public API

```python
import docrooter as dr

doc = dr.load("10k.pdf")
tax = dr.Taxonomy.load("taxonomy.csv")

clf = dr.Classifier(
    tax,
    backend="embedding:dmis-lab/UltraBERT-Embed-150M",
    fallback=None,  # backend for nodes the primary can't handle
    overrides={"depth>=3": "nli:...", "node:fin.tax": "trained:./bundle"},
    strategy=dr.Beam(width=3),  # or dr.Greedy(), dr.Flat()
    view=dr.SectionView(),
    instruction="Classify this section of a corporate annual report",
    threshold=0.35,  # default per-node descend threshold
    min_margin=0.0,  # stop if top1 - top2 < margin
    sections="leaves",  # "all" | "leaves" | levels=[1,2] | callable
    batch_size=32,
)

result = clf.classify(doc)  # also: clf.classify_texts([...]), clf.classify_many(docs)
result.sections["s7.2"].path_labels  # ["Financial", "Liquidity"]
result.to_json()
result.to_dataframe()
result.summary(depth=1)
```

### Traversal strategies

The traversal for each section starts at the virtual root. Each step chooses among the current node's children.

- **`Greedy`:** take the argmax child if it is at or above `threshold`, otherwise stop. One path.
- **`Beam(width=k)`:** keep the top k partial paths, ranked by the sum of log conditional probabilities. Children below `threshold` are pruned. A path ends at a leaf or when no child passes. Return the best complete path plus the others as `alternatives`. This limits the damage from one early mistake.
- **`Flat`:** score every leaf (or every node, when `allow_internal=True`) in one call. Each candidate's text is its full path (`Financial > Tax > Deferred tax`). This is useful for shallow taxonomies and embedding backends, and it serves as a comparison in the benchmark. It is not used for `trained:` backends.

**Multi-label nodes:** every child at or above `threshold` is followed, up to `max_fanout` (default 3), so a section can end with several paths.

**Stopping early is a feature:** a section can end at an internal node with `stopped_reason` set to `"low_confidence"`, `"low_margin"`, `"max_depth"`, or `"no_backend"`.

### Batched execution

Traversal is **level-synchronous**:

1. At each round, gather every active `(section, node)` pair across all sections and beams.
2. Group the pairs by `(node, backend)`.
3. Call `backend.score` once per group, in mini-batches of `batch_size`.

Each section's rendered view is computed once and cached. Embedding backends also cache section embeddings and label embeddings, so each text is encoded once and every further level costs only a matrix multiply. A 20-level taxonomy therefore costs about 20 cheap rounds, not 20 model passes per section.

`max_depth` (default 64) is a safety guard.

### Result model

```python
class Step(BaseModel):
    node_id: str                       # decision made at this parent
    chosen: list[str]
    probs: dict[str, float]            # top-k children (k=5) with probabilities
    backend: str

class PathResult(BaseModel):
    node_ids: list[str]
    labels: list[str]
    score: float                       # product of step probabilities
    steps: list[Step]
    stopped_reason: str | None

class SectionResult(BaseModel):
    section_id: str
    title: str
    best: PathResult | None
    alternatives: list[PathResult]
    truncated: bool

class DocumentResult(BaseModel):
    source: str
    sections: dict[str, SectionResult]
    def summary(self, depth: int = 1) -> dict[str, float]   # label share weighted by section tokens
```

## 8. Training, calibration, evaluation

### Dataset format

JSONL, one example per line (CSV with the same columns is also accepted):

```json
{"text": "...section view or raw text...", "path": ["fin", "fin.tax"], "doc_id": "optional", "section_id": "optional"}
```

`path` may contain node IDs or labels; labels are resolved through the taxonomy. A path can stop at an internal node.

`dr.Dataset.from_results(result)` exports predictions as a dataset for human review. Any backend can act as the "teacher", so you can label with a slow model, correct, then train a fast one.

### Training method: shared body + per-node heads

Deep, wide taxonomies make "one fine-tuned transformer per node" too expensive: a 20-level taxonomy can have thousands of internal nodes. So v0.1 trains **one shared encoder** and **one tiny linear head per internal node**:

1. **Per-node sets:** for each internal node N, collect the examples whose path passes through N. The label is the child taken at the next step.
2. **Fine-tune the shared body (optional, on by default):** contrastive fine-tuning of the embedding model with sentence-transformers.
   - Positive pairs are examples sharing the same next-step label at the same node. Negatives are siblings, which are hard negatives by construction.
   - Loss: `MultipleNegativesRankingLoss` with hard negatives.
   - Learning rate and epochs come from the model recipe.
   - We use sentence-transformers directly rather than the `setfit` library, because UltraBERT requires transformers>=5 and setfit's version compatibility can lag behind it.
3. **Per-node heads:** for every node with at least `min_examples_per_child` examples (default 4) for at least 2 children, fit a scikit-learn `LogisticRegression` on the frozen fine-tuned embeddings. Each head is a few KB.
4. **Hybrid coverage:** nodes without enough data have no head. At inference, `TrainedModel.supports(node)` returns False and the classifier uses the zero-shot `fallback`, which by default is the embedding backend on the same body. A partly labelled taxonomy therefore still works end to end.

```python
train_ds, test_ds = dr.Dataset.load("labels.jsonl").split(test_size=0.2, stratify_depth=1, seed=0)
model = dr.train(
    tax,
    train_ds,
    base_model="dmis-lab/UltraBERT-Embed-150M",
    finetune_body=True,
    epochs=None,
    lr=None,
)  # None = from recipe
model.save("bundle/")  # manifest.json + body/ + heads/*.joblib + training metrics
clf = dr.Classifier(tax, backend=model, fallback="embedding:dmis-lab/UltraBERT-Embed-150M")
```

The bundle `manifest.json` records:

- the taxonomy content hash, which produces a warning when the model is loaded with a modified taxonomy
- the base model and its revision
- the docrooter version
- per-node example counts
- training metrics

### Calibration

`dr.calibrate(clf, dev_ds)` fits a temperature per backend by minimising negative log-likelihood of the gold child at each step. It returns a calibrated classifier, and the temperatures are saved with the classifier config. Per-node thresholds are out of scope for v0.1 (Section 11).

### Evaluation

```python
report = dr.evaluate(clf, test_ds)
report.print()  # rich table in terminal
report.to_json()
report.to_markdown()
dr.compare({"zero-shot": clf_a, "trained": clf_b}, test_ds).print()
```

Metrics:

| Metric | Meaning |
|---|---|
| Exact path accuracy | Predicted path equals the gold path |
| Accuracy at depth d | Correct node at each depth, for d = 1..D |
| Hierarchical P/R/F1 | Standard hierarchical metric: compare the ancestor-augmented sets of the predicted and gold nodes. Partial credit for the right branch. |
| Node decision accuracy | For each internal node, accuracy of the child choice given the correct parent. Isolates weak nodes. |
| Stop rate and selective accuracy | How often the classifier stopped early, and accuracy on the cases where it didn't |
| Top sibling confusions | Most-confused child pairs per node |
| Throughput | Sections per second, with device recorded |

## 9. Benchmark (the UltraBERT test, part of v0.1)

The benchmark answers questions that decide the package defaults:

1. Which **zero-shot** backend should be the default?
2. Does UltraBERT's **masked-LM multiple choice** work well enough to ship as a non-experimental backend?
3. Which **base model** should training default to?
4. Which **traversal strategy** should be the default?

It lives in `benchmarks/` in the repository. It is not shipped in the wheel. Datasets stream from Hugging Face and are subsampled with fixed seeds.

### Datasets

| ID | Dataset (HF) | License | Shape | Purpose |
|---|---|---|---|---|
| B1 `sections` | `eloukas/edgar-corpus` | Apache-2.0 | Real 10-K sections. 2-level taxonomy: Part (I–IV) → Item (1, 1A, 2, 3, 7, 7A, 8, …). Each Item's text is one example; the section view's token budget truncates long items. | Realistic document sections. View excludes `path` and `title` so the item name doesn't leak the answer. |
| B2 `deep` | `DeveloperOats/DBPedia_Classes` | CC0-1.0 | 3 levels: 9 → 70 → 219 classes | Traversal over a deeper tree: greedy vs beam vs flat, and how errors compound with depth |
| B3 `wide` | `coastalcph/lex_glue` (`ledgar`) | CC-BY-4.0 | Flat, 100 contract clause types | Wide flat taxonomy. MLM-choice is excluded because 100 > 26 options. |

Sample size: 1,000 test examples per dataset, plus training pools of 4, 8, 16, and 64 examples per class for the few-shot runs, with fixed seeds. NLI cost grows with the number of candidates (one forward pass per text–label pair), so on B3 (100 labels) the NLI baseline runs on a fixed 300-example subset of the test set, and every system is also reported on that subset for a like-for-like comparison.

### Systems compared

Zero-shot:

- `embedding:dmis-lab/UltraBERT-Embed-150M`
- `embedding:BAAI/bge-small-en-v1.5` (popular small embedder baseline)
- `nli:MoritzLaurer/deberta-v3-base-zeroshot-v2.0` (zero-shot classification baseline)
- `mlm-choice:dmis-lab/UltraBERT-380M` (B1 and B2 only)
- `openai:<model>` only if `DOCROOTER_BENCH_LLM_URL` is set. Off by default.

Few-shot trained (shared body + heads) at 4, 8, 16, and 64 examples per class:

- Base `dmis-lab/UltraBERT-Embed-150M`
- Base `BAAI/bge-small-en-v1.5`
- Each with frozen-body (heads only) and fine-tuned-body variants

Strategies on B2: Greedy, Beam(3), and Flat. Greedy and Beam(3) run with the best zero-shot backend and the best trained model. Flat runs only with embedding backends, since it needs one score per leaf (219 candidates).

### Outputs

- `benchmarks/results/<date>/<run>.json`, the raw metrics and environment (device, versions, model revisions)
- `docs/benchmarks.md`, generated tables with hierarchical F1, path accuracy, accuracy per depth, and throughput on the M1 (MPS and CPU)

### Decision rules (written down before running)

- **Default zero-shot backend:** highest mean hierarchical F1 across B1–B3, among backends with at least 20 sections per second on M1 MPS. Ties go to the smaller model.
- **`mlm-choice` status:** ships non-experimental only if, on both B1 and B2, it is within 3 hierarchical-F1 points of the best zero-shot backend.
- **Default training base model:** highest hierarchical F1 at 16 examples per class, averaged over B1–B3.
- **Default strategy:** Beam(3), unless Greedy is within 1 point of Beam(3) path accuracy on B2 and at least 1.5× faster. In that case Greedy becomes the default.
- Results that contradict expectations are recorded as they are, not tuned away. The chosen defaults link to the run that justified them.

### Reproducibility

`uv run python -m benchmarks.run --suite zero-shot|few-shot|strategies|all --device mps`. All seeds, revisions, and sample IDs go into the results file.

## 10. Packaging, CLI, project layout

### Layout

```
docrooter/
├── pyproject.toml
├── README.md
├── LICENSE                    # Apache-2.0
├── src/docrooter/
│   ├── __init__.py            # public API re-exports
│   ├── document/{models,markdown,headings,pdf}.py
│   ├── views.py
│   ├── keyphrases.py
│   ├── taxonomy/{models,io,validate}.py
│   ├── backends/{base,registry,recipes,embedding,nli,mlm_choice,trained,openai}.py
│   ├── classify/{classifier,strategies,results}.py
│   ├── train/{dataset,trainer,bundle,calibrate}.py
│   ├── evaluate/{metrics,report}.py
│   ├── _extras.py             # lazy imports + helpful install errors
│   └── cli.py
├── tests/
│   ├── fixtures/              # sample .md docs, taxonomies (JSON/CSV variants), datasets
│   ├── unit/                  # fast, no torch: uses FakeBackend
│   └── integration/           # @pytest.mark.models: downloads small models
├── benchmarks/
└── docs/
```

### Dependencies

| Install | Contents |
|---|---|
| core | `pydantic>=2`, `numpy`, `typer`, `rich` |
| `[models]` | `torch`, `transformers>=5`, `sentence-transformers>=6`, `scikit-learn`, `joblib` |
| `[pdf]` | `pymupdf4llm` (AGPL, opt-in) |
| `[llm]` | `httpx` |
| `[bench]` (dev only) | `datasets`, `pandas` |
| `[all]` | `models` + `pdf` + `llm` |

**Tooling:**

- Python `>=3.10`
- uv for environments, builds, and publishing
- ruff for lint and formatting
- pytest
- GitHub Actions with PyPI trusted publishing (workflow written in v0.1, first publish when the user decides)
- License: Apache-2.0, matching UltraBERT and the permissive dependencies

### CLI

```
docrooter toc FILE [--json]                         # print validated TOC
docrooter taxonomy validate FILE                    # check a taxonomy file
docrooter classify FILE -t TAXONOMY [-b BACKEND] [--strategy beam:3] [-o out.json]
docrooter train -t TAXONOMY -d DATA.jsonl [--base MODEL] -o BUNDLE_DIR
docrooter eval -t TAXONOMY -d TEST.jsonl -b BACKEND [--format md|json]
```

### Error handling

- Missing extra → `MissingExtraError("embedding backend needs: pip install 'docrooter[models]'")`
- Bad taxonomy → `TaxonomyError` with the full list of issues and their locations
- Fixed-label backend asked about an unsupported node with no fallback → the step stops with `stopped_reason="no_backend"` and a single summary warning per run (not per section)
- Sections exceeding the token budget are truncated and flagged, never an error
- Model download or network failures surface the original Hugging Face error with the model ID and revision

### Testing

- **Unit tests (no torch, run in CI on every push):**
  - Markdown parsing edge cases
  - Heading validator decisions on a labelled fixture set
  - Hierarchy repair
  - All taxonomy formats parsing to identical trees
  - Validation errors
  - Traversal strategies against a deterministic `FakeBackend` (keyword overlap scores), including a 30-level and a 500-child synthetic taxonomy
  - Result serialisation round trips
  - Metric functions against hand-computed values
- **Integration tests (`-m models`, run locally or nightly):** end to end with `bge-small` (fast), plus one UltraBERT smoke test for each backend type, and a train → save → load → classify round trip.
- **Benchmark:** run manually; results committed.

## 11. Implementation milestones

1. **Skeleton:** pyproject, layout, CI, `_extras`, FakeBackend.
2. **Documents and TOC:** Markdown loader, heading validator, hierarchy repair, `docrooter toc`.
3. **Taxonomy:** model, JSON and CSV loaders, validation, `docrooter taxonomy validate`.
4. **Views and keyphrases.**
5. **Classification core:** backend protocol, score normalisation, Greedy/Beam/Flat, batched level-synchronous execution, results, `docrooter classify` (tested with FakeBackend).
6. **Model backends:** recipes and security gate, embedding, NLI, MLM-choice, device handling.
7. **Benchmark zero-shot suite:** dataset adapters for B1–B3, runner, results doc. **First real UltraBERT results land here.**
8. **Training:** dataset, shared-body fine-tune, per-node heads, bundle save/load, hybrid fallback.
9. **Evaluation and calibration:** metrics, report, compare, temperature fitting; few-shot and strategy benchmark suites; set defaults from the decision rules.
10. **Extras and polish:** PDF loader, OpenAI-compatible backend, `Dataset.from_results`, README with a quickstart, PyPI workflow.

### Deferred (v0.2+)

- Per-node threshold tuning
- Cross-validating PDF structure across multiple parsers
- Learned heading classifier trained on PDF bookmarks
- DAG taxonomies
- Entity extraction
- MCP server
- Compression
- Hosted service

## 12. Assumptions and open points

- **Open source under Apache-2.0** is assumed, since the user has not said otherwise.
- **UltraBERT checkpoints were updated on 2026-10-14 and 2026-10-15.** Revisions are pinned at implementation time and recorded in recipes and benchmark results.
- **The MLM multiple-choice prompt format isn't documented by the authors.** The backend's prompt is our design and is validated only by the benchmark.
- **EDGAR-CORPUS item fields:** the B1 Part → Item mapping uses the standard 10-K structure. Items with empty text are dropped.
