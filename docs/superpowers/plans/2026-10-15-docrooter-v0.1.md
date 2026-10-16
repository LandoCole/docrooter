# docrooter v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `docrooter`, a Python package that turns Markdown/PDF documents into a validated TOC and classifies each section against flat-to-very-deep user taxonomies using small local models. It also trains, calibrates and evaluates those classifiers, and includes a benchmark that picks the defaults (including the UltraBERT test).

**Architecture:** A flat-map document tree and a flat-map taxonomy, both traversed iteratively. A `Backend` protocol scores texts against the children of one taxonomy node. A level-synchronous traversal engine batches all `(section, node)` decisions per round. Training produces a `TrainedModel` that is itself a `Backend`: a shared sentence-transformers body plus per-node logistic-regression heads.

**Tech Stack:** Python ≥3.10, uv, pydantic v2, numpy, typer, rich; optional torch, transformers≥5, sentence-transformers≥6, scikit-learn, datasets, pymupdf4llm, httpx; pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-15-docrooter-section-classification-design.md`

**Refinements to the spec made by this plan** (interface-level only; apply them as written):
- `SectionResult` gains `selected: list[PathResult]` (all accepted paths; length > 1 only for multi-label nodes) and `n_tokens: int` (used by `summary`).
- The `Backend` protocol gains a `temperature: float` attribute and a `multi_label: bool = False` keyword on `score`. `Candidate` gains `examples: tuple[str, ...] = ()`.
- New files: `document/load.py`, `document/build.py`, `classify/engine.py`, `backends/loading.py`, `backends/fake.py`, `errors.py`. Evaluation (Task 12) comes before the benchmark (Task 13), because the benchmark needs the metrics.
- `datasets` moves into the `[models]` extra, because sentence-transformers training consumes `datasets.Dataset`.

## Global Constraints

- Package import name `docrooter`; distribution name `docrooter`; console script `docrooter`; license Apache-2.0; `requires-python = ">=3.10"`; src layout.
- Core dependencies are exactly `pydantic>=2`, `numpy`, `typer`, `rich`. Every other import is lazy, through `docrooter._extras.require`.
- Extras: `models` = `torch`, `transformers>=5`, `sentence-transformers>=6`, `scikit-learn`, `joblib`, `datasets`; `pdf` = `pymupdf4llm`; `llm` = `httpx`; `bench` = `datasets`, `pandas`; `all` = models + pdf + llm.
- No recursion over document or taxonomy trees. All traversal is iterative, and tests cover depth 30.
- `trust_remote_code=True` only when the model's recipe has a pinned `revision` and `trust_remote_code=True`, or the caller passes `trust_remote_code=True` explicitly.
- Pinned revisions: `dmis-lab/UltraBERT-Embed-150M` @ `367d5343b5a0164d57b810cedf5de4b8983d5984`; `dmis-lab/UltraBERT-380M` @ `65bbbfeefb03bc93b9965d1d511708dabcee8511`.
- Device order: CUDA → MPS → CPU. float32 on CPU/MPS. FlashAttention 2 only on CUDA when importable, otherwise `sdpa`.
- PyMuPDF/pymupdf4llm is AGPL-3.0 and installed only via the `[pdf]` extra. The README states this.
- Unit tests never import torch. Model tests carry `@pytest.mark.models` and are deselected by default (`addopts = "-m 'not models'"`).
- Error messages for missing extras use the exact form: `pip install 'docrooter[<extra>]'`.
- Secrets live only in the git-ignored `.env` file (e.g. `TYPESAFE_API_KEY`). Never commit, log, or print API keys. Tests that need a key skip when it is absent.

## Review Focus

1. **Markdown with no headings at all** (plain text, or text before the first heading) must load as a single section `s0` titled from the file stem (or `"Preamble"` for leading text), and classification must work on it. Pinned in Task 2.
2. **Taxonomy CSV exported from Excel** (UTF-8 BOM, `\r\n` line endings, padded cells) must parse to the same tree as a clean file. Pinned in Task 5.
3. **A heading-only leaf section** (empty body, no children) must render a non-empty view (its title), never send `""` to a model. Pinned in Task 6.
4. **Duplicate section titles** ("Overview" under several parents) must get unique section IDs, and `Document.section("Overview")` must return the first in reading order. Pinned in Task 2.
5. **The same label in different branches** ("Other" under several parents) must resolve by parent context in datasets, and must render as distinct full-path candidates in `Flat`. Pinned in Tasks 11 and 9.

---

### Task 1: Project skeleton

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `LICENSE` (Apache-2.0 text), `README.md` (one-paragraph stub), `src/docrooter/__init__.py`, `src/docrooter/_extras.py`, `src/docrooter/errors.py`, `tests/unit/test_extras.py`, `.github/workflows/ci.yml`

**Interfaces:**
- Produces: `docrooter.__version__ = "0.1.0.dev0"`; `class MissingExtraError(ImportError)`; `require(module: str, extra: str) -> types.ModuleType`; `errors.DocrooterError(Exception)`, `errors.TaxonomyError(DocrooterError)` with `.issues: list`, `errors.BackendError(DocrooterError)`, `errors.DatasetError(DocrooterError)`.

- [ ] **Step 1:** `git init`. Then `uv init --lib --name docrooter --python 3.12` (keep `requires-python = ">=3.10"`, hatchling build backend, src layout). Add the dependencies and extras from Global Constraints, plus dev group `pytest`, `ruff`. Add `[project.scripts] docrooter = "docrooter.cli:app"`. Pytest config: `testpaths=["tests"]`, `markers=["models: downloads and runs ML models"]`, `addopts="-m 'not models'"`. Ruff: line length 100, rules `E,F,I,UP,B`.
- [ ] **Step 2: Write the failing test** `tests/unit/test_extras.py`:

```python
def test_require_returns_module():
    assert require("json", "models").dumps({}) == "{}"

def test_require_missing_names_extra():
    with pytest.raises(MissingExtraError, match=r"pip install 'docrooter\[models\]'"):
        require("definitely_not_a_module_xyz", "models")

def test_version():
    import docrooter
    assert docrooter.__version__ == "0.1.0.dev0"
```

- [ ] **Step 3:** Run `uv run pytest tests/unit/test_extras.py -v` and expect a failure (ImportError).
- [ ] **Step 4:** Implement `_extras.require` (importlib, re-raise as `MissingExtraError` with message `"{module} is required for this feature: pip install 'docrooter[{extra}]'"`), `errors.py`, and `__version__`.
- [ ] **Step 5:** Run `uv run pytest -v && uv run ruff check .` and expect a PASS and a clean lint.
- [ ] **Step 6:** Add CI workflow: on push/PR, `astral-sh/setup-uv`, Python 3.10 and 3.12 matrix, `uv sync`, `uv run ruff check .`, `uv run pytest`.
- [ ] **Step 7:** Commit `chore: project skeleton`.

### Task 2: Document model, Markdown extraction, tree builder

**Files:**
- Create: `src/docrooter/document/__init__.py`, `document/models.py`, `document/markdown.py`, `document/build.py`, `tests/unit/document/test_markdown.py`, `tests/fixtures/docs/{simple.md,no_headings.md,setext_and_code.md,duplicate_titles.md}`

**Interfaces:**
- Produces:
  - `RawHeading(BaseModel)`: `title: str`, `level: int`, `text: str`, `line: int`, `page: int | None = None`, `heading_score: float = 1.0`, `flags: list[str] = []`. A preamble block uses `level=0`. `build_document` copies `page` into `Section.page_start`.
  - `extract_headings(markdown: str) -> list[RawHeading]`
  - `build_document(raw: list[RawHeading], *, source: str, title: str | None = None) -> Document`
  - `Section` with the spec's fields: `id, title, level, text, parent_id, children, page_start, heading_score, flags`
  - `TocEntry(BaseModel)`: `id, title, level, depth: int`
  - `Document` with `source, title, sections: dict[str, Section], root_ids: list[str]`, `walk() -> Iterator[Section]` (pre-order, iterative), `toc() -> list[TocEntry]`, `section(key: str) -> Section` (by id, else first by title in reading order, else `KeyError`), `leaves() -> list[Section]`, `from_sections(entries: list[dict], *, source="<sections>") -> Document` (classmethod; each dict has `title`, `level`, optional `text`).
- Section IDs: preamble = `s0`. The others are `s` + dotted 1-based ordinal path in the tree (`s1`, `s1.2`, `s1.2.3`).

- [ ] **Step 1: Write failing tests:**

```python
def test_atx_headings_build_tree():
    doc = build_document(extract_headings(read("simple.md")), source="simple.md")
    assert [e.title for e in doc.toc()] == ["Intro", "Background", "Scope", "Results"]
    assert doc.section("s1.2").title == "Scope" and doc.section("s1.2").parent_id == "s1"
    assert doc.section("Intro").text.startswith("Intro body")   # body excludes child sections

def test_no_headings_single_section():
    doc = build_document(extract_headings("just text\nmore"), source="notes.md")
    assert list(doc.sections) == ["s0"] and doc.sections["s0"].title == "notes"

def test_preamble_before_first_heading():
    doc = build_document(extract_headings("lead text\n# A\nbody"), source="x.md")
    assert doc.sections["s0"].title == "Preamble" and doc.root_ids == ["s0", "s1"]

def test_setext_and_fenced_code_ignored():
    titles = [h.title for h in extract_headings(read("setext_and_code.md"))]
    assert titles == ["Setext Title", "Sub"]          # '# comment' inside ``` is not a heading

def test_duplicate_titles_unique_ids():
    doc = build_document(extract_headings(read("duplicate_titles.md")), source="d.md")
    ids = [s.id for s in doc.walk() if s.title == "Overview"]
    assert len(ids) == 2 and len(set(ids)) == 2
    assert doc.section("Overview").id == ids[0]

def test_deep_doc_no_recursion():
    md = "\n".join(f"{'#' * min(d, 6)} H{d}\nx" for d in range(1, 31))  # repeated H6s stay siblings
    doc = build_document(extract_headings(md), source="deep.md")
    assert len(list(doc.walk())) == 30
```

- [ ] **Step 2:** Run `uv run pytest tests/unit/document -v` and expect a FAIL.
- [ ] **Step 3:** Implement `extract_headings`. Line scanner: track fenced-code state (```` ``` ```` / `~~~`); ATX `^(#{1,6})\s+(.+?)\s*#*\s*$`; setext = non-empty line followed by `^=+\s*$` (level 1) or `^-+\s*$` (level 2). Body lines accumulate into the current heading's `text` (stripped).
- [ ] **Step 4:** Implement `build_document` with a stack of open sections: pop while `stack[-1].level >= new.level`. Ordinals come from a per-parent child counter. Title fallback for the `s0` no-heading case is `Path(source).stem`.
- [ ] **Step 5:** Run the tests and expect a PASS. Commit `feat: markdown TOC extraction and document tree`.

### Task 3: Heading validation and hierarchy repair

**Files:**
- Create: `src/docrooter/document/headings.py`, `tests/unit/document/test_headings.py`, `tests/fixtures/docs/noisy_headings.md`

**Interfaces:**
- Consumes: `RawHeading`.
- Produces: `score_heading(title: str, *, repeat_count: int = 1) -> tuple[float, list[str]]` (score in [0, 1] and the reasons); `validate_headings(raw: list[RawHeading], *, min_score: float = 0.5) -> list[RawHeading]` (rejected headings are merged into the previous block's text as a line; survivors get `heading_score`, and `"low_heading_confidence"` when the score is < 0.7); `repair_levels(raw: list[RawHeading]) -> list[RawHeading]`.

- [ ] **Step 1: Write failing tests:**

```python
@pytest.mark.parametrize("title,ok", [
    ("1.2 Liquidity and Capital Resources", True), ("RISK FACTORS", True), ("Appendix A", True),
    ("Page 3", False), ("3 of 12", False), ("https://example.com/x", False),
    ("This sentence is clearly a paragraph that someone formatted as a heading by mistake.", False),
    ("- bullet item", False), ("| a | b |", False),
])
def test_score_heading(title, ok):
    assert (score_heading(title)[0] >= 0.5) is ok

def test_repeated_title_penalized():
    assert score_heading("ACME Corp Annual Report", repeat_count=5)[0] < 0.5

def test_validate_merges_rejected_into_previous():
    out = validate_headings(extract_headings(read("noisy_headings.md")))
    assert "Page 3" not in [h.title for h in out]
    assert "Page 3" in out[0].text

def test_repair_numbering_overrides_hashes():
    raw = [RawHeading(title=t, level=2, text="", line=i)
           for i, t in enumerate(["1 Intro", "1.1 Scope", "1.1.1 Detail", "2 Next"])]
    assert [h.level for h in repair_levels(raw)] == [1, 2, 3, 1]

def test_repair_closes_gaps():
    raw = [RawHeading(title="A", level=1, text="", line=0), RawHeading(title="B", level=4, text="", line=1)]
    out = repair_levels(raw)
    assert [h.level for h in out] == [1, 2] and "level_repaired" in out[1].flags
```

- [ ] **Step 2:** Run and expect a FAIL.
- [ ] **Step 3:** Implement `score_heading` as a base score of 0.6 plus additive feature weights, clamped to [0, 1]. Weights:

| Condition | Weight |
|---|---|
| Numbering prefix `^(\d+(\.\d+)*\.?\|[IVXLC]+\.\|(Article\|Section\|Chapter\|Part\|Item\|Appendix)\s+\w+)\b` | +0.25 |
| Title case or all caps | +0.1 |
| More than 20 words | −0.4 |
| Ends with `.`, `,`, `;` | −0.3 |
| Matches `^(page\s*)?\d+(\s*(of\|/)\s*\d+)?$` (case-insensitive) | −0.7 |
| Contains a URL or email | −0.6 |
| Fewer than 40% alphanumeric characters | −0.5 |
| Starts with `-`, `*`, `+` or `\|` | −0.6 |
| `repeat_count >= 3` | −0.5 |

`validate_headings` computes `repeat_count` from normalised (casefold, stripped) title frequencies.
- [ ] **Step 4:** Implement `repair_levels`. If ≥50% of headings have a dotted-numeric prefix, those headings get `level = number of dot-separated parts`. Then a single pass caps each level at `previous_level + 1`. Changed headings get flag `"level_repaired"`.
- [ ] **Step 5:** Run the tests and expect a PASS. Commit `feat: heading validation and hierarchy repair`.

### Task 4: `load()`, CLI `toc`

**Files:**
- Create: `src/docrooter/document/load.py`, `src/docrooter/cli.py`, `tests/unit/document/test_load.py`, `tests/unit/test_cli_toc.py`
- Modify: `src/docrooter/__init__.py` (export `load`, `Document`, `Section`)

**Interfaces:**
- Consumes: Tasks 2 and 3.
- Produces: `load(source: str | os.PathLike, *, min_heading_score: float = 0.5, repair: bool = True) -> Document`. A path ending `.md`/`.markdown`/`.txt` is read from disk; `.pdf` calls `document.pdf.load_pdf` (Task 16; until then it raises `MissingExtraError` for `pdf`); any other `str` containing a newline is treated as Markdown text. Also the typer `app` with a `toc` command: `docrooter toc FILE [--json]`.

- [ ] **Step 1: Write failing tests:**
  - `test_load_md_path_equals_text` checks that `load(path).toc() == load(path.read_text()).toc()`.
  - `test_load_pdf_without_extra` monkeypatches `require` to raise, then checks that loading a PDF raises `MissingExtraError`.
  - `test_cli_toc_json` uses typer `CliRunner`: `docrooter toc simple.md --json` prints a JSON list whose first item has `{"id": "s1", "title": "Intro", "level": 1}`.
  - `test_cli_toc_tree` checks that the plain output indents "Scope" by two spaces per depth.
- [ ] **Step 2:** Run and expect a FAIL.
- [ ] **Step 3:** Implement `load` as extract, then validate, then repair (if enabled), then build. The CLI is a `typer.Typer(no_args_is_help=True)`, with subcommand groups added later in this file.
- [ ] **Step 4:** Run the tests and expect a PASS. Then commit with message `feat: load() and docrooter toc`.

### Task 5: Taxonomy model, validation, JSON/CSV loaders, CLI `taxonomy validate`

**Files:**
- Create: `src/docrooter/taxonomy/{__init__,models,validate,io}.py`, `tests/unit/taxonomy/{test_models,test_io,test_validate}.py`, `tests/fixtures/taxonomies/{nested.json,flat.json,adjacency.csv,levels.csv,path.csv,excel_export.csv,broken.csv}` (the first five describe the same 2-level tree: Financial{Tax, Audit}, Legal{Contracts, Litigation}; `excel_export.csv` = `levels.csv` saved with a BOM, `\r\n` and padded cells)
- Modify: `cli.py` (add the `taxonomy validate FILE` command, which exits with code 1 on errors and prints the issues as a rich table), `__init__.py` (export `Taxonomy`, `TaxonomyNode`)

**Interfaces:**
- Produces:
  - `TaxonomyNode`: fields as in spec Section 5.
  - `Taxonomy`:
    - `ROOT_ID = "__root__"`
    - `Taxonomy(nodes: list[TaxonomyNode])` builds the flat map and the virtual root, and raises `TaxonomyError` on errors
    - `node(id) -> TaxonomyNode`, `children(node_id=ROOT_ID) -> list[TaxonomyNode]`, `path(node_id) -> list[TaxonomyNode]` (root excluded), `depth() -> int`, `leaves() -> list[TaxonomyNode]`, `node_depth(node_id) -> int` (root = 0)
    - `resolve_path(parts: list[str]) -> list[str]`: each part is matched as a node id, else as a case-insensitive label among the children of the previous resolved node; raises `KeyError` naming the part and the valid options
    - `content_hash() -> str` (sha256 of canonical JSON)
    - `to_dict() -> dict`
    - `load(path) -> Taxonomy` (classmethod)
    - `from_dict(data) -> Taxonomy` (classmethod)
  - `TaxonomyIssue(severity: Literal["error","warning"], code: str, message: str, location: str | None)`
  - `validate_nodes(nodes: list[TaxonomyNode]) -> list[TaxonomyIssue]`, with codes `duplicate_id`, `missing_parent`, `cycle`, `empty_label`, `duplicate_sibling_label` (errors) and `single_child` (warning)
  - `io.from_csv(path_or_text: str | Path, *, sep: str = ">") -> Taxonomy`
  - Generated IDs: `".".join(slug(label) for each level)`, where `slug` = lowercase, non-alphanumerics → `-`, stripped.

- [ ] **Step 1: Write failing tests:**

```python
@pytest.mark.parametrize("name", ["nested.json", "flat.json", "adjacency.csv", "levels.csv", "path.csv", "excel_export.csv"])
def test_all_formats_same_tree(name):
    t = Taxonomy.load(FIX / name)
    assert {n.id: [c.id for c in t.children(n.id)] for n in t.children()} == \
           {"financial": ["financial.tax", "financial.audit"], "legal": ["legal.contracts", "legal.litigation"]}

def test_resolve_path_by_label_uses_parent_context():
    t = Taxonomy.from_dict({"nodes": [
        {"id": "a", "label": "A", "children": [{"id": "a.o", "label": "Other"}]},
        {"id": "b", "label": "B", "children": [{"id": "b.o", "label": "Other"}]}]})
    assert t.resolve_path(["B", "other"]) == ["b", "b.o"]

def test_validation_errors_have_locations():
    with pytest.raises(TaxonomyError) as e:
        Taxonomy.load(FIX / "broken.csv")       # contains a missing parent on row 4 and a cycle
    codes = {(i.code, i.location) for i in e.value.issues}
    assert ("missing_parent", "row 4") in codes and any(c == "cycle" for c, _ in codes)

def test_deep_and_wide():
    nodes = [TaxonomyNode(id=f"n{d}", label=f"L{d}", parent_id=f"n{d-1}" if d else None) for d in range(30)]
    nodes += [TaxonomyNode(id=f"w{i}", label=f"W{i}", parent_id="n0") for i in range(500)]
    t = Taxonomy(nodes)
    assert t.depth() == 30 and len(t.children("n0")) == 501 and len(t.path("n29")) == 30
```

- [ ] **Step 2:** Run `uv run pytest tests/unit/taxonomy -v` and expect a FAIL.
- [ ] **Step 3:** Implement the models and `validate_nodes`. Cycle detection walks parent pointers iteratively with a visited set.
- [ ] **Step 4:** Implement the IO. JSON: if the top-level `nodes` items contain `children` lists, flatten them iteratively; otherwise treat the file as flat with `parent_id`. CSV: open with `encoding="utf-8-sig"`, `csv.DictReader`, strip every cell, and choose the format by header (`parent_id`, then `level_1`, then `path`; anything else raises `TaxonomyError` code `unknown_csv_format`). `examples` cells are split on `|`. Locations are `"row N"` (1-based, header = row 1) or the JSON pointer `"/nodes/0/children/1"`.
- [ ] **Step 5:** Add the CLI command and a `test_cli_taxonomy_validate_exit_code` test (broken file → exit 1).
- [ ] **Step 6:** Run the tests and expect a PASS. Commit `feat: taxonomy model, loaders, validation`.

### Task 6: Keyphrases and SectionView

**Files:**
- Create: `src/docrooter/keyphrases.py`, `src/docrooter/views.py`, `tests/unit/test_keyphrases.py`, `tests/unit/test_views.py`
- Modify: `__init__.py` (export `SectionView`)

**Interfaces:**
- Consumes: `Document`, `Section`.
- Produces:
  - `extract_keyphrases(text: str, corpus: Sequence[str], k: int = 8) -> list[str]`
  - `RenderedView(NamedTuple)`: `text: str`, `truncated: bool`, `n_tokens: int`
  - `SectionView(BaseModel)`: `fields: tuple[str, ...] = ("path","title","keyphrases","lead","text")`, `max_tokens: int = 512`, `lead_sentences: int = 3`, `keyphrases: int = 8`, `empty_section: Literal["children_titles","title"] = "children_titles"`
  - `SectionView.render_all(doc: Document, section_ids: Sequence[str], count_tokens: Callable[[str], int] | None = None) -> dict[str, RenderedView]`
  - `estimate_tokens(text) -> int` = `ceil(len(text.split()) * 1.3)`

- [ ] **Step 1: Write failing tests:**

```python
def test_keyphrases_prefer_distinctive_terms():
    sec = "The revolving credit facility matures in 2027. The revolving credit facility has covenants."
    others = ["The company reported revenue growth.", "The company hired staff."]
    kp = extract_keyphrases(sec, [sec, *others], k=3)
    assert kp[0] == "revolving credit facility" and "the company" not in kp

def test_render_field_order_and_labels():
    v = SectionView(fields=("path", "title", "text")).render_all(doc, ["s1.2"])["s1.2"]
    assert v.text.splitlines()[0] == "Path: Intro > Scope" and v.text.splitlines()[1] == "Title: Scope"

def test_budget_truncates_last_fitting_field():
    v = SectionView(fields=("title", "text"), max_tokens=20).render_all(long_doc, ["s1"])["s1"]
    assert v.truncated and v.n_tokens <= 20

def test_heading_only_leaf_renders_title():
    d = Document.from_sections([{"title": "Signatures", "level": 1}])
    v = SectionView(fields=("text",)).render_all(d, ["s1"])["s1"]
    assert v.text.strip() != "" and "Signatures" in v.text

def test_empty_parent_uses_children_titles():
    d = Document.from_sections([{"title": "Part II", "level": 1}, {"title": "Item 7", "level": 2, "text": "x"}])
    assert "Subsections: Item 7" in SectionView(fields=("text",)).render_all(d, ["s1"])["s1"].text
```

- [ ] **Step 2:** Run and expect a FAIL.
- [ ] **Step 3:** Implement `extract_keyphrases`:
  1. Lowercase the text and tokenise with `[a-z0-9][a-z0-9\-']*`.
  2. Candidates are 1–3-grams that neither start nor end with a stopword (a built-in English list of about 180 words in `keyphrases.STOPWORDS`) and contain no stopword in the middle.
  3. Score = `tf(phrase in text) * log((1 + N) / (1 + df(phrase over corpus))) * (1 + 0.5 * (n_words - 1))`.
  4. Drop any phrase that is a substring of a higher-scored kept phrase, and return the top k.
- [ ] **Step 4:** Implement `render_all`. Compute the corpus (all section texts) once. Field lines:

| Field | Line |
|---|---|
| `path` | `Path: <ancestor titles + own, joined by " > ">` |
| `title` | `Title: …` |
| `keyphrases` | `Key phrases: a, b, c` (omitted when there are none) |
| `lead` | `Lead: <first N sentences, split on (?<=[.!?])\s+>` |
| `text` | `Text: …` |

Empty-body rules: if `text` is empty and the section has children, `children_titles` gives `Subsections: t1; t2`; otherwise use the title. Fill the fields in order. When a field exceeds the remaining budget, cut it word by word to fit, set `truncated=True` and stop.
- [ ] **Step 5:** Run the tests and expect a PASS. Commit `feat: keyphrases and section views`.

### Task 7: Backend protocol, probability normalisation, KeywordBackend, registry

**Files:**
- Create: `src/docrooter/backends/{__init__,base,fake,registry}.py`, `tests/unit/backends/test_base.py`, `tests/unit/backends/test_registry.py`

**Interfaces:**
- Produces:
  - `Candidate(NamedTuple)`: `node_id: str`, `text: str`, `examples: tuple[str, ...] = ()`
  - `Scores(NamedTuple)`: `values: np.ndarray`, `kind: Literal["softmax","independent","similarity"]`
  - `Backend(Protocol)`: `name: str`, `max_candidates: int | None`, `temperature: float`, `score(texts, candidates, *, instruction=None, multi_label=False) -> Scores`, `supports(node_id: str) -> bool`, `tokenizer() -> Callable[[str], int] | None`
  - `to_probs(scores: Scores, *, temperature: float, multi_label: bool) -> np.ndarray`:
    - `similarity`: row softmax of `values / T`
    - `softmax`: if T == 1, return as-is; else row softmax of `log(clip(p, 1e-12)) / T`
    - `independent`: as-is when `multi_label`; otherwise rows renormalised to sum to 1
  - `candidate_text(node: TaxonomyNode, *, taxonomy: Taxonomy | None = None, full_path: bool = False) -> str`: returns `label`, or `"label: description"`. With `full_path`, the labels are joined by `" > "` and the leaf's description is appended.
  - `KeywordBackend(name="keyword", temperature=0.1)`: similarity = Jaccard overlap of lowercase word sets between the text and `candidate.text + examples`. `supports` always returns True; `max_candidates=None`.
  - `parse_spec(spec: str) -> tuple[str, str, str | None]` (kind, model, revision)
  - `resolve_backend(spec: str | Backend, **kwargs) -> Backend`
  - `register_backend_kind(kind: str, factory: Callable[..., Backend]) -> None`
  - Built-in kinds are registered lazily by import path: `keyword`, `embedding`, `nli`, `mlm-choice`, `trained`, `openai`.

- [ ] **Step 1: Write failing tests:**

```python
def test_similarity_softmax_with_temperature():
    p = to_probs(Scores(np.array([[0.9, 0.1]]), "similarity"), temperature=0.1, multi_label=False)
    assert p.shape == (1, 2) and p[0, 0] > 0.99 and np.isclose(p.sum(), 1)

def test_independent_renormalised_for_single_label():
    p = to_probs(Scores(np.array([[0.6, 0.6]]), "independent"), temperature=1, multi_label=False)
    assert np.allclose(p, [[0.5, 0.5]])

def test_parse_spec():
    assert parse_spec("embedding:dmis-lab/UltraBERT-Embed-150M@abc") == ("embedding", "dmis-lab/UltraBERT-Embed-150M", "abc")
    assert parse_spec("keyword") == ("keyword", "", None)

def test_unknown_kind_lists_known():
    with pytest.raises(BackendError, match="embedding"):
        resolve_backend("bogus:x")

def test_keyword_backend_scores():
    s = KeywordBackend().score(["income tax expense"], [Candidate("t", "Tax: income tax"), Candidate("a", "Audit")])
    assert s.values[0, 0] > s.values[0, 1]
```

- [ ] **Step 2:** Run and expect a FAIL.
- [ ] **Step 3:** Implement as specified. `resolve_backend` returns `Backend` instances unchanged.
- [ ] **Step 4:** Run the tests and expect a PASS. Commit `feat: backend protocol and registry`.

### Task 8: Traversal engine (Greedy/Beam), results

**Files:**
- Create: `src/docrooter/classify/{__init__,strategies,results,engine}.py`, `tests/unit/classify/test_engine.py`

**Interfaces:**
- Consumes: `Taxonomy`, `Backend`, `to_probs`, `candidate_text`.
- Produces:
  - `Greedy()`, `Beam(width: int = 3)`, `Flat(allow_internal: bool = False)` as frozen dataclasses
  - Results models per the spec (`Step`, `PathResult`, `SectionResult`, `DocumentResult`), with `SectionResult.selected` and `n_tokens` added
  - `EngineSettings(BaseModel)`: `threshold: float = 0.35`, `min_margin: float = 0.0`, `max_fanout: int = 3`, `max_depth: int = 64`, `batch_size: int = 32`, `instruction: str | None = None`, `temperatures: dict[str, float] = {}`
  - `Router = Callable[[str], list[Backend]]` (candidate backends for a decision at `node_id`, in priority order)
  - `run_tree(texts: dict[str, str], taxonomy: Taxonomy, router: Router, strategy: Greedy | Beam, settings: EngineSettings) -> dict[str, tuple[PathResult | None, list[PathResult], list[PathResult]]]`, mapping each key to `(best, alternatives, selected)`
  - `decide(texts: list[str], node_id: str, taxonomy, router, settings) -> tuple[list[str], np.ndarray, str]` (child ids, probs `(n, k)`, backend name), for teacher-forced evaluation and calibration

- [ ] **Step 1: Write failing tests** using `KeywordBackend` and a small taxonomy with labels only, no descriptions (Financial{Tax, Audit}, Legal{Contracts, Litigation}). Ties in probability keep taxonomy child order (stable sort), so the tests are deterministic:

```python
def test_greedy_reaches_leaf():
    out = run_tree({"x": "income tax deferred tax financial"}, tax, router, Greedy(), EngineSettings(threshold=0.0))
    best, _, _ = out["x"]
    assert best.node_ids == ["financial", "financial.tax"] and best.stopped_reason is None
    assert best.steps[0].node_id == Taxonomy.ROOT_ID and len(best.steps) == 2

def test_low_confidence_stops_at_internal_node():
    best, _, _ = run_tree({"x": "financial"}, tax, router, Greedy(), EngineSettings(threshold=0.9))["x"]
    assert best.node_ids == ["financial"] and best.stopped_reason == "low_confidence"

def test_beam_returns_alternatives_sorted():
    best, alts, _ = run_tree({"x": "tax contracts"}, tax, router, Beam(3), EngineSettings(threshold=0.0))["x"]
    assert alts and all(a.score <= best.score for a in alts)

def test_multilabel_node_selects_several():
    t = taxonomy_with_multilabel_root()
    _, _, selected = run_tree({"x": "tax audit contracts litigation"}, t, router, Beam(2), EngineSettings(threshold=0.1))["x"]
    assert {p.node_ids[0] for p in selected} == {"financial", "legal"}

def test_low_margin_stops():
    best, _, _ = run_tree({"x": "tax contracts"}, tax, router, Greedy(), EngineSettings(threshold=0, min_margin=0.2))["x"]
    assert best.node_ids == [] and best.stopped_reason == "low_margin"

def test_no_backend_reason():
    best, _, _ = run_tree({"x": "tax"}, tax, lambda n: [], Greedy(), EngineSettings())["x"]
    assert best.node_ids == [] and best.stopped_reason == "no_backend"

def test_batching_one_call_per_node_per_round():
    spy = CountingBackend(KeywordBackend())
    run_tree({str(i): "income tax" for i in range(100)}, tax, lambda n: [spy], Greedy(), EngineSettings(threshold=0, batch_size=1000))
    assert spy.calls == 2          # root round + financial round

def test_depth_30_chain():
    best, _, _ = run_tree({"x": "l29"}, chain_taxonomy(30), router, Greedy(), EngineSettings(threshold=0))["x"]
    assert len(best.node_ids) == 30   # single-child nodes are auto-chosen with p=1.0, no backend call
```

- [ ] **Step 2:** Run and expect a FAIL.
- [ ] **Step 3:** Implement `run_tree` with this algorithm (the part the tests don't fully determine):

```
beam state: node_ids, logp, steps, done, reason, branch (tuple of multi-label choices)
init each key with one empty beam at ROOT
for round in range(max_depth):
    active = all (key, beam) not done; stop if none
    for each active beam: node = beam.node_ids[-1] if any else ROOT
        children = taxonomy.children(node)
        if not children: done (leaf)
        if len(children) == 1: extend with p=1.0, no backend call
    group remaining by (node, first backend b in router(node) with b.supports(node)
                       and (b.max_candidates is None or len(children) <= b.max_candidates))
        none found -> done, reason "no_backend"
    per group: unique keys -> score once per key in mini-batches of batch_size
        probs = to_probs(scores, temperature=settings.temperatures.get(b.name, b.temperature),
                         multi_label=node.multi_label)
    expand each beam:
        thr = node.threshold if set else settings.threshold
        ranked children by prob desc; eligible = p >= thr
        none eligible -> done, "low_confidence"
        single-label and len(ranked)>1 and p1 - p2 < min_margin -> done, "low_margin"
        multi-label: each of the top max_fanout eligible children -> new beam with branch + (child,)
        single-label: Greedy -> top eligible only; Beam -> every eligible child
        each Step records parent node, chosen [child], top-5 probs, backend name
    prune: per (key, branch) keep top `width` beams by logp (Greedy width = 1)
after loop: unfinished beams -> reason "max_depth"
per key: selected = best beam per branch; best = max(selected); alternatives = other beams
         in best's branch, sorted by logp desc; PathResult.score = exp(logp)
```

`decide` does a single teacher-forced scoring at `node_id` with the same backend selection and normalisation.
- [ ] **Step 4:** Run the tests and expect a PASS. Commit `feat: level-synchronous taxonomy traversal`.

### Task 9: Classifier, Flat strategy, overrides, CLI `classify`

**Files:**
- Create: `src/docrooter/classify/classifier.py`, `tests/unit/classify/test_classifier.py`
- Modify: `classify/engine.py` (add `run_flat`), `cli.py` (add `classify`), `__init__.py` (export `Classifier`, `Greedy`, `Beam`, `Flat`)

**Interfaces:**
- Consumes: Tasks 6, 7 and 8.
- Produces:
  - `run_flat(texts, taxonomy, backend, strategy: Flat, settings) -> same shape as run_tree`
  - `Classifier(taxonomy, backend="embedding:dmis-lab/UltraBERT-Embed-150M", *, fallback=None, overrides: dict[str, str | Backend] | None = None, strategy=Beam(3), view: SectionView | None = None, instruction=None, threshold=0.35, min_margin=0.0, max_fanout=3, sections: str | list[int] | Callable[[Section], bool] = "leaves", batch_size=32, temperatures: dict[str, float] | None = None)`
  - Methods: `classify(doc) -> DocumentResult`, `classify_texts(texts: list[str]) -> list[SectionResult]` (section ids `"0"`, `"1"`, …), `classify_many(docs) -> list[DocumentResult]` (one traversal over all docs), `router(node_id) -> list[Backend]`, `decide(texts, node_id)` (delegates to the engine), `.settings: EngineSettings`
  - `DocumentResult.summary(depth: int = 1) -> dict[str, float]`, `to_json(indent=2) -> str`, `to_dataframe()` (requires pandas; the hint names the `bench` extra)
  - Backends are resolved lazily on first use, so constructing a `Classifier` never downloads a model.
- Router priority for a decision at node N with decision depth d = `taxonomy.node_depth(N)`:
  1. `overrides["node:N"]`
  2. `N.backend`
  3. The first matching `overrides["depth>=k" | "depth<=k" | "depth==k"]` in dict order
  4. The primary backend
  5. `fallback` (appended last when not None)
- When `backend` is a `TrainedModel` and `fallback is None`, the fallback is `backend.as_zero_shot()` (Task 14 provides it; check with `hasattr`).
- After each `classify*` call, if any section stopped with `no_backend`, emit one `UserWarning` naming the count and the affected node ids. `SectionResult.truncated` comes from the rendered view.

- [ ] **Step 1: Write failing tests** (KeywordBackend throughout):

```python
def test_classify_doc_leaves_only():
    res = Classifier(tax, backend=KeywordBackend(), threshold=0).classify(doc)
    assert set(res.sections) == {s.id for s in doc.leaves()}

def test_override_precedence():
    a, b, c = KeywordBackend(name="a"), KeywordBackend(name="b"), KeywordBackend(name="c")
    clf = Classifier(tax, backend=a, overrides={"depth>=1": b, "node:legal": c})
    assert clf.router("financial")[0].name == "b" and clf.router("legal")[0].name == "c"
    assert clf.router(Taxonomy.ROOT_ID)[0].name == "a"

def test_flat_uses_full_path_candidates_for_duplicate_labels():
    t = taxonomy_with_other_under_each_parent()
    spy = RecordingBackend(KeywordBackend())
    Classifier(t, backend=spy, strategy=Flat()).classify_texts(["x"])
    assert {"A > Other", "B > Other"} <= {c.text.split(":")[0] for c in spy.last_candidates}

def test_summary_weighted_by_tokens():
    res = Classifier(tax, backend=KeywordBackend(), threshold=0).classify(doc_one_long_tax_one_short_legal)
    s = res.summary(depth=1)
    assert s["Financial"] > s["Legal"] and abs(sum(s.values()) - 1) < 1e-9

def test_classify_many_single_traversal():
    spy = CountingBackend(KeywordBackend())
    out = Classifier(tax, backend=spy, strategy=Greedy(), threshold=0).classify_many([doc, doc])
    assert len(out) == 2 and out[0].sections.keys() == out[1].sections.keys()
    assert spy.calls == 2      # both docs share each round's call

def test_no_backend_warns_once_per_run():
    with pytest.warns(UserWarning, match="no_backend") as w:
        Classifier(tax, backend=NeverSupports(), threshold=0).classify(doc)
    assert len([x for x in w if "no_backend" in str(x.message)]) == 1

def test_truncated_flag_passed_through():
    res = Classifier(tax, backend=KeywordBackend(), view=SectionView(max_tokens=5)).classify(long_doc)
    assert any(r.truncated for r in res.sections.values())

def test_construction_does_not_resolve_backend(monkeypatch):
    monkeypatch.setattr("docrooter.backends.registry.resolve_backend", boom)
    Classifier(tax, backend="embedding:whatever")   # no error until classify()

def test_cli_classify_keyword(tmp_path):
    r = runner.invoke(app, ["classify", "simple.md", "-t", "nested.json", "-b", "keyword", "-o", str(tmp_path / "o.json")])
    assert r.exit_code == 0 and json.loads((tmp_path / "o.json").read_text())["sections"]
```

- [ ] **Step 2:** Run and expect a FAIL.
- [ ] **Step 3:** Implement `run_flat`:
  1. Candidates are `leaves()`, or all non-root nodes when `allow_internal`.
  2. Each candidate's text is `candidate_text(n, taxonomy=t, full_path=True)`.
  3. Make one `score` call per mini-batch and take the argmax.
  4. If the best probability is below the threshold, return `PathResult(node_ids=[], stopped_reason="low_confidence")`; otherwise return the path to the argmax node with a single `Step` at `ROOT_ID`.
- [ ] **Step 4:** Implement `Classifier`. Render views with the primary backend's `tokenizer()` when available, otherwise `estimate_tokens`. `n_tokens` = `estimate_tokens(section.text)`. The CLI `classify` options are `-t/--taxonomy`, `-b/--backend` (default the UltraBERT embed spec), `--strategy greedy|beam:N|flat`, `--instruction`, `--threshold`, `-o/--out` (stdout when omitted), plus a rich table of section → path labels when writing to a file.
- [ ] **Step 5:** Run the tests and expect a PASS. Commit `feat: Classifier API, flat strategy, CLI classify`.

### Task 10: Recipes, model loading, embedding backend

**Files:**
- Create: `src/docrooter/backends/{recipes,loading,embedding}.py`, `tests/unit/backends/test_recipes.py`, `tests/integration/test_embedding.py`, `tests/unit/test_cli_models.py`
- Modify: `cli.py` (add the `models` command group)

**Interfaces:**
- Produces:
  - `Recipe(BaseModel)` with these fields and defaults:
    - `revision: str | None = None`
    - `trust_remote_code: bool = False`
    - `query_prompt: str = "{instruction}\n"`
    - `label_prompt: str = ""`
    - `default_instruction: str = "Classify this document section"`
    - `max_seq_len: int = 512`
    - `temperature: float = 0.05`
    - `batch_size: int = 32`
    - `finetune_lr: float = 2e-5`
    - `finetune_epochs: int = 1`
  - `get_recipe(model_id: str, **overrides) -> Recipe`; `register_recipe(model_id: str, recipe: Recipe) -> None`
  - Built-in recipes:
    - **UltraBERT-Embed-150M:** pinned revision, `trust_remote_code=True`, `query_prompt="Instruct: {instruction}\nQuery: "`, `label_prompt="Label: "`, `max_seq_len=8192`, `finetune_lr=5e-5`
    - **UltraBERT-380M:** pinned revision, `trust_remote_code=True`, `max_seq_len=8192`, `finetune_lr=3e-5`
    - **`BAAI/bge-small-en-v1.5`:** `query_prompt="Represent this sentence for searching relevant passages: "`, `max_seq_len=512`
    - **`MoritzLaurer/deberta-v3-base-zeroshot-v2.0`:** `max_seq_len=512`, `temperature=1.0`
  - `select_device(preference: str | None = None) -> str`
  - `attn_implementation(device: str) -> str`
  - `trust_decision(model_id: str, recipe: Recipe, explicit: bool | None) -> bool`: raises `BackendError` when the model needs remote code and neither condition from Global Constraints holds; the message mentions `trust_remote_code=True`
  - `EmbeddingBackend(model: str, *, revision=None, device=None, trust_remote_code=None, batch_size=None, **recipe_overrides)` with kind `similarity`, `max_candidates=None`, `encode(texts: list[str], *, prompt: str) -> np.ndarray` (L2-normalised, cached by `(prompt, text)`), `tokenizer()` (token count via the model tokenizer), `name = f"embedding:{model}"`
  - Label vector = normalised mean of `encode([label_prompt + candidate.text])` and `encode(examples, prompt=query_prompt)`. Query prompt = `recipe.query_prompt.format(instruction=instruction or recipe.default_instruction)`.

- [ ] **Step 1: Write failing unit tests (no torch):**
  - `test_trust_requires_pin_or_explicit`: an unknown model with `explicit=None` gives False. A recipe with `trust_remote_code=True` but `revision=None` raises `BackendError`. The UltraBERT recipe gives True.
  - `test_device_order` monkeypatches torch availability flags and checks `"cuda"`, then `"mps"`, then `"cpu"`.
  - `test_ultrabert_prompts` checks `get_recipe("dmis-lab/UltraBERT-Embed-150M").query_prompt.format(instruction="X") == "Instruct: X\nQuery: "`.
- [ ] **Step 2: Write the failing integration test** (`@pytest.mark.models`):

```python
@pytest.mark.parametrize("model", ["BAAI/bge-small-en-v1.5", "dmis-lab/UltraBERT-Embed-150M"])
def test_embedding_zero_shot_sanity(model):
    b = EmbeddingBackend(model)
    s = b.score(["The company recorded a deferred income tax liability."],
                [Candidate("tax", "Tax: income taxes"), Candidate("lit", "Litigation: lawsuits and legal proceedings")])
    assert s.kind == "similarity" and s.values[0, 0] > s.values[0, 1]
```

- [ ] **Step 3:** Run `uv run pytest tests/unit/backends -v` and expect a FAIL. Then implement. `select_device` imports torch lazily and returns `"cpu"` when torch is absent.
- [ ] **Step 4:** Run `uv sync --extra models && uv run pytest -m models tests/integration/test_embedding.py -v` and expect a PASS. This downloads the models; record the wall time in the commit message body.
- [ ] **Step 4b: `docrooter models` CLI.** Add `docrooter models pull MODEL_ID [--revision REV]` (calls `huggingface_hub.snapshot_download(repo_id, revision=rev or recipe.revision)` and prints the cache path) and `docrooter models list` (lists cached repos from `huggingface_hub.scan_cache_dir()` as a rich table: repo id, revision(s), size on disk). Both require the `models` extra. Unit test `tests/unit/test_cli_models.py` monkeypatches `snapshot_download`/`scan_cache_dir` and asserts the pinned recipe revision is passed for `dmis-lab/UltraBERT-Embed-150M` and that `list` prints the repo id.
- [ ] **Step 5:** Commit `feat: recipes, trust gate, embedding backend`.

### Task 11: NLI and MLM-choice backends, Dataset

**Files:**
- Create: `src/docrooter/backends/{nli,mlm_choice}.py`, `src/docrooter/train/{__init__,dataset}.py`, `tests/integration/test_nli_mlm.py`, `tests/unit/backends/test_mlm_prompt.py`, `tests/unit/train/test_dataset.py`, `tests/fixtures/datasets/{small.jsonl,small.csv}` (the same 8 examples, 2 per leaf of the Financial/Legal fixture taxonomy, with one path given as labels and the rest as ids)
- Modify: `__init__.py` (export `Dataset`)

**Interfaces:**
- Produces:
  - `NLIBackend(model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0", *, hypothesis_template="This text is about {}.", device=None, batch_size=None)`
    - Entailment index from `config.label2id` (key containing `"entail"`, case-insensitive). Contradiction or not-entailment is the other index.
    - With `multi_label=False`: kind `softmax`, a row softmax of the entailment logits across candidates.
    - With `multi_label=True`: kind `independent`, a per-pair softmax over [entail, other].
    - The hypothesis uses `candidate.text`.
  - `build_choice_prompt(text: str, options: list[str], *, instruction: str, answer: str, markers: Sequence[str] = LETTERS) -> str` (exact format below; `LETTERS` = `A`–`Z`; `answer` is the mask token for MLM and `""` for the LLM backend)
  - `MLMChoiceBackend(model="dmis-lab/UltraBERT-380M", *, revision=None, device=None, trust_remote_code=None)`
    - `max_candidates=26`, kind `softmax`
    - Markers `A`–`Z`. For each letter, use the first of `[" A", "A"]` that tokenises to exactly one token; raise `BackendError` if neither does.
    - Text is truncated (tokens) so the prompt fits `recipe.max_seq_len`.
    - Probabilities are a softmax over the mask-position logits restricted to the first `len(candidates)` marker ids.
  - `Example(BaseModel)`: `text: str`, `path: list[str]`, `doc_id: str | None = None`, `section_id: str | None = None`
  - `Dataset`:
    - `Dataset(examples: list[Example])`
    - `load(path) -> Dataset` (classmethod): JSONL, or CSV with columns `text,path`, where `path` is joined by `" > "`
    - `save(path)`, `__len__`, `__iter__`
    - `resolve(taxonomy) -> Dataset`: paths become node ids via `Taxonomy.resolve_path`. On failure it raises `DatasetError` naming the line number and the underlying `KeyError` message.
    - `split(test_size=0.2, stratify_depth=1, seed=0) -> tuple[Dataset, Dataset]`: stratify on `path[:stratify_depth]`
    - `node_examples(taxonomy, node_id) -> tuple[list[str], list[str]]`: texts and next-step child ids for examples whose path passes through the node; the root uses `path[0]`
    - `from_results(result: DocumentResult, doc: Document, view: SectionView) -> Dataset` (classmethod): text = rendered view, path = `best.node_ids`; sections with an empty best path are skipped

Exact MLM prompt (a fixed design choice, which the benchmark validates):

```
{instruction}
Text: {text}
Options:
A. {option_0}
B. {option_1}
...
Answer: {answer}
```

The option lines use `{marker}. {option}`; with `answer=""` the last line is `Answer:` with no trailing space.

- [ ] **Step 1: Write failing unit tests:**
  - `test_choice_prompt_format` asserts the exact string for 2 options.
  - `test_dataset_jsonl_csv_equal` loads both fixtures and compares the example lists.
  - `test_resolve_labels_by_parent_context` uses the "Other" taxonomy from Task 5: `["B","Other"]` → `["b","b.o"]`.
  - `test_resolve_error_has_line` checks that a bad path on line 3 produces `DatasetError` with `"line 3"` in the message.
  - `test_split_stratified_deterministic` checks that the same seed gives the same split and every depth-1 class appears in both splits.
  - `test_node_examples` checks that at `financial` it returns only the examples under financial, labelled `financial.tax` or `financial.audit`.
  - `test_from_results_roundtrip`.
- [ ] **Step 2: Write the failing integration tests** (`@pytest.mark.models`): the same two-candidate tax-vs-litigation sanity check as Task 10, for `NLIBackend()` and `MLMChoiceBackend()`. Assert the kind and that `values.sum(axis=1) ≈ 1` for single-label.
- [ ] **Step 3:** Run the tests and expect a FAIL. Then implement.
- [ ] **Step 4:** Run `uv run pytest -v` and then `uv run pytest -m models tests/integration/test_nli_mlm.py -v`, and expect a PASS. If the MLM sanity test fails, do not tune the prompt to pass it. Mark it `xfail(strict=False, reason="mlm-choice experimental; see benchmark")` and note the failure in the commit body; the benchmark decides its status.
- [ ] **Step 5:** Commit `feat: NLI and MLM-choice backends, Dataset`.

### Task 12: Metrics, evaluate, compare, CLI `eval`

**Files:**
- Create: `src/docrooter/evaluate/{__init__,metrics,report}.py`, `tests/unit/evaluate/test_metrics.py`, `tests/unit/evaluate/test_report.py`
- Modify: `cli.py` (add `eval`), `__init__.py` (export `evaluate`, `compare`)

**Interfaces:**
- Consumes: `Classifier.classify_texts`, `Classifier.decide`, `Dataset`.
- Produces:
  - Metric functions (all over lists of node-id paths):
    - `exact_path_accuracy(pred, gold) -> float`
    - `depth_accuracy(pred, gold) -> dict[int, float]`: depth d is scored over examples with `len(gold) >= d`; correct when `len(pred) >= d and pred[d-1] == gold[d-1]`
    - `hierarchical_prf(pred, gold) -> tuple[float, float, float]`: micro; sets = path node ids; 0 when a denominator is 0
    - `stop_rate(results: list[SectionResult]) -> float`: share where `best.stopped_reason` is not None
    - `selective_accuracy(pred, gold, stopped: list[bool]) -> float`
    - `sibling_confusions(decisions: dict[str, list[tuple[str, str]]], top: int = 3) -> dict[str, list[tuple[str, str, int]]]`
  - `Report(BaseModel)`: `n`, `exact`, `hp`, `hr`, `hf1`, `depth_acc: dict[int, float]`, `node_acc: dict[str, float]`, `stop_rate`, `selective_acc`, `confusions`, `sections_per_sec`, `device: str`, `meta: dict`; `print()`, `to_json()`, `to_markdown() -> str`
  - `evaluate(clf: Classifier, dataset: Dataset, *, node_decisions: bool = True) -> Report`
    - Resolve the dataset.
    - Time `classify_texts`; time excludes model loading, by calling one warm-up text first.
    - For node decisions, group gold `(text, node, gold_child)` triples by node and call `clf.decide` once per node.
  - `Comparison` with `print()`, `to_markdown()`; `compare(classifiers: dict[str, Classifier], dataset) -> Comparison`

- [ ] **Step 1: Write failing tests** with hand-computed values:

```python
def test_hierarchical_prf_partial_credit():
    p, r, f = hierarchical_prf([["a", "a.x"]], [["a", "a.y"]])
    assert (p, r) == (0.5, 0.5) and f == 0.5

def test_depth_accuracy_counts_only_deep_enough_gold():
    assert depth_accuracy([["a"], ["b", "b.x"]], [["a"], ["b", "b.y"]]) == {1: 1.0, 2: 0.0}

def test_stopped_short_prediction_wrong_at_depth2():
    assert depth_accuracy([["a"]], [["a", "a.x"]])[2] == 0.0

def test_evaluate_with_keyword_backend_end_to_end():
    rep = evaluate(Classifier(tax, backend=KeywordBackend(), threshold=0), Dataset.load(FIX / "small.jsonl"))
    assert rep.n == 8 and 0 <= rep.hf1 <= 1 and set(rep.node_acc) >= {Taxonomy.ROOT_ID, "financial"}
    assert "| hF1 |" in rep.to_markdown()
```

- [ ] **Step 2:** Run and expect a FAIL. Then implement. `eval` CLI options: `-t`, `-d`, `-b`, `--strategy`, `--format table|md|json`.
- [ ] **Step 3:** Run the tests and expect a PASS. Commit `feat: evaluation metrics, reports, compare`.

### Task 13: Benchmark harness and zero-shot run (UltraBERT test)

**Files:**
- Create: `benchmarks/__init__.py`, `benchmarks/data.py`, `benchmarks/systems.py`, `benchmarks/run.py`, `benchmarks/report.py`, `benchmarks/taxonomies/edgar.json`, `benchmarks/README.md`, `tests/unit/test_bench_data.py`, `docs/benchmarks.md` (generated)

**Interfaces:**
- Consumes: `Taxonomy`, `Dataset`, `Classifier`, `evaluate`, `compare`, all backends.
- Produces:
  - `BenchData(NamedTuple)`: `name`, `taxonomy`, `test: Dataset`, `train_pool: Dataset`, `subset_ids: list[int] | None`
  - `load_b1(n_test=1000, seed=0)`, `load_b2(n_test=1000, seed=0)`, `load_b3(n_test=1000, seed=0)`
  - `pool_shots(pool: Dataset, k: int, seed: int) -> Dataset` (k examples per leaf)
  - `run_suite(suite: Literal["zero-shot","few-shot","strategies"], device: str, n_test: int) -> list[dict]`
  - Result rows: `{suite, dataset, system, strategy, shots, report: report.model_dump(), env: {device, torch, transformers, sentence_transformers, docrooter, revisions}}`, written to `benchmarks/results/<YYYY-MM-DD>/<suite>.json`
  - `render_markdown(results_dir) -> str`, written to `docs/benchmarks.md`

Dataset specifics (spec Section 9):

| Benchmark | Source and labels | Text | Notes |
|---|---|---|---|
| **B1** | `load_dataset("eloukas/edgar-corpus", "year_2020", split="train", streaming=True)`. Explode the `section_*` fields into examples labelled Part → Item. Part I = 1, 1A, 1B, 2, 3, 4; Part II = 5, 6, 7, 7A, 8, 9, 9A, 9B; Part III = 10–14; Part IV = 15. | Item text with whitespace collapsed | Drop items under 50 words. `edgar.json` gives every Item a one-line description of the standard 10-K item. |
| **B2** | `DeveloperOats/DBPedia_Classes` test split, columns `text,l1,l2,l3` | `text` | Labels CamelCase-split into words. |
| **B3** | `coastalcph/lex_glue` config `ledgar` test split | `text` | Flat taxonomy from the `label` feature names. `subset_ids` = a fixed 300-example sample (seed 0). |

- **Benchmark view** for every system: `SectionView(fields=("text",), max_tokens=512)`. Instructions:
  - B1: `"Classify this section of a 10-K annual report"`
  - B2: `"Classify the topic of this Wikipedia article"`
  - B3: `"Classify the type of this contract clause"`
- **Zero-shot systems:**
  - `embedding:dmis-lab/UltraBERT-Embed-150M`
  - `embedding:BAAI/bge-small-en-v1.5`
  - `nli:MoritzLaurer/deberta-v3-base-zeroshot-v2.0`, on B3 run only on `subset_ids`
  - `mlm-choice:dmis-lab/UltraBERT-380M`, B1 and B2 only
  - `openai:$DOCROOTER_BENCH_LLM_MODEL`, only when `DOCROOTER_BENCH_LLM_URL` is set (this will be skipped until Task 16)
- **Strategy for the zero-shot suite:** `Beam(3)`, threshold 0.0 (forced to a leaf so accuracy is comparable).
- **B3 like-for-like:** every system is also reported on the B3 subset.

- [ ] **Step 1: Write failing unit tests** that need no network (feed the loaders small in-memory rows via an injectable `rows` parameter):
  - `test_b1_part_item_mapping` checks that a row with `section_7` produces the path `["part-ii", "part-ii.item-7"]` and a 10-word item is dropped.
  - `test_b2_camelcase_labels` checks `"AmusementParkAttraction"` → `"Amusement Park Attraction"`.
  - `test_pool_shots_k_per_leaf`.
  - `test_render_markdown_table` uses a fixture results file.
- [ ] **Step 2:** Run `uv run pytest tests/unit/test_bench_data.py -v` and expect a FAIL. Then implement.
- [ ] **Step 3:** Run the tests and expect a PASS. Commit `feat: benchmark harness`.
- [ ] **Step 4: Run the zero-shot suite** in the background, because it takes tens of minutes on the M1:
  `uv sync --extra models --extra bench && uv run python -m benchmarks.run --suite zero-shot --device mps --n-test 1000`
  Expected: `benchmarks/results/<date>/zero-shot.json` with 4 systems × 3 datasets (minus mlm-choice on B3), plus the B3 subset rows.
  If MPS errors on a model, rerun that system with `--device cpu --only <system>` and record the device in the results.
- [ ] **Step 5:** Run `uv run python -m benchmarks.report` to regenerate `docs/benchmarks.md`. Apply the spec's zero-shot decision rules and write the decisions under a "Decisions" heading, each linked to the results file:
  - **Default zero-shot backend:** best mean hF1 with ≥ 20 sections/s on MPS.
  - **`mlm-choice` status:** within 3 hF1 points of the best on B1 and B2.

  Update the `Classifier` default `backend`, and the `cli classify` default, if the winner is not UltraBERT-Embed. Remove or keep the `mlm-choice` xfail and the "experimental" docstring as decided.
- [ ] **Step 6:** Commit `bench: zero-shot results and default backend decision`, with the results JSON and the docs.

### Task 14: Training (shared body + per-node heads), TrainedModel bundle, CLI `train`

**Files:**
- Create: `src/docrooter/train/{trainer,bundle}.py`, `tests/unit/train/test_trainer_data.py`, `tests/integration/test_train_roundtrip.py`
- Modify: `cli.py` (add `train`), `__init__.py` (export `train`, `TrainedModel`), `backends/registry.py` (`trained:<path>` → `TrainedModel.load`)

**Interfaces:**
- Consumes: `EmbeddingBackend.encode`, `get_recipe`, `Dataset.node_examples`, `Taxonomy`.
- Produces:
  - `build_triplets(taxonomy, dataset, *, max_triplets=20000, seed=0) -> list[tuple[str, str, str]]` (pure, no torch). For each internal node and each example there, the triple is (anchor, positive with the same child, negative from a sibling child). Sample uniformly across nodes up to the cap. Skip nodes where any side is missing.
  - `trainable_nodes(taxonomy, dataset, min_examples_per_child=4) -> dict[str, dict[str, int]]`: node → child → count, for nodes with ≥2 children that meet the minimum.
  - `train(taxonomy, dataset, *, base_model="dmis-lab/UltraBERT-Embed-150M", finetune_body=True, epochs=None, lr=None, batch_size=None, min_examples_per_child=4, max_triplets=20000, device=None, seed=0, instruction=None) -> TrainedModel`
  - `TrainedModel` (implements `Backend`):
    - `name = "trained:<base_model>"`, kind `softmax`, `max_candidates=None`, `temperature=1.0`
    - `supports(node_id)` is True only for nodes that have a head
    - `score` raises `BackendError` when the candidate ids don't equal the head's classes
    - `save(path)`, `load(path) -> TrainedModel` (classmethod), `as_zero_shot() -> EmbeddingBackend` (same body directory, same recipe prompts)
    - `.manifest: dict`
- Manifest keys: `docrooter_version`, `base_model`, `base_revision`, `taxonomy_hash`, `instruction`, `heads: {node_id: {"file": "heads/<sha1(node_id)[:12]>.joblib", "classes": [...], "n": int}}`, `node_counts`, `train_metrics: {triplets, epochs, lr, seconds}`.
- `Classifier(...)` warns (`warnings.warn`, `UserWarning`) when given a `TrainedModel` whose `taxonomy_hash` differs from `taxonomy.content_hash()`.

- [ ] **Step 1: Write failing unit tests (no torch):**
  - `test_triplets_negatives_are_siblings` checks every negative's child id differs from the positive's at the same node.
  - `test_triplets_cap_and_determinism` checks the cap is respected and the same seed gives the same list.
  - `test_trainable_nodes_threshold` checks that a node with one child having 3 examples is excluded at min 4.
- [ ] **Step 2:** Run and expect a FAIL. Then implement the pure functions and run again for a PASS.
- [ ] **Step 3:** Implement `train`:
  1. Load an `EmbeddingBackend` for the base model and take its `SentenceTransformer`.
  2. If `finetune_body`, prefix every triplet string with the recipe query prompt (default instruction) and build a `datasets.Dataset` with columns `anchor, positive, negative`.
  3. Train with `SentenceTransformerTrainer`, `MultipleNegativesRankingLoss`, lr = `lr or recipe.finetune_lr`, epochs = `epochs or recipe.finetune_epochs`, `warmup_ratio=0.05`, `seed=seed`.
  4. Encode node examples with the query prompt, then fit `LogisticRegression(max_iter=1000)` per trainable node.
- [ ] **Step 4: Write and run the integration test** (`@pytest.mark.models`, bge-small for speed): train on `small.jsonl` with `min_examples_per_child=2`, `finetune_body=True`, `epochs=1`. Then save, load and classify through `Classifier(tax, backend=loaded)`. Assert:
  - Every head node has `supports()` True.
  - An untrained node routes to the `as_zero_shot()` fallback, which appears in the step's `backend` name.
  - A modified taxonomy triggers the hash `UserWarning`.
- [ ] **Step 5:** Add the `train` CLI command: `-t -d --base -o --no-finetune-body --epochs --lr --min-examples`. Test with `CliRunner` on the keyword-free path, by monkeypatching `train` to a stub that saves a dummy bundle.
- [ ] **Step 6:** Commit `feat: shared-body training with per-node heads`.

### Task 15: Calibration, classifier config, few-shot and strategy benchmarks, defaults

**Files:**
- Create: `src/docrooter/train/calibrate.py`, `tests/unit/train/test_calibrate.py`
- Modify: `classify/classifier.py` (`save_config`/`from_config`), `__init__.py` (export `calibrate`), `benchmarks/systems.py` (few-shot and strategy systems), `docs/benchmarks.md`

**Interfaces:**
- Consumes: `Classifier.decide`, `Dataset`, `to_probs`, `train`.
- Produces:
  - `calibrate(clf: Classifier, dev: Dataset, *, grid: Sequence[float] = tuple(np.logspace(-3, 1, 50))) -> Classifier`: a copy with `temperatures` set per backend name.
    - For each backend, collect the gold-node decisions it handles (raw `Scores` via `backend.score` at gold nodes).
    - Choose the T in the grid that minimises the mean NLL of the gold child under `to_probs(..., temperature=T)`.
  - `Classifier.save_config(path)` writes JSON: backend specs as strings (`TrainedModel` → `"trained:<its saved path>"`, and it raises `ValueError` if it was never saved), strategy, view, settings, temperatures, taxonomy hash.
  - `Classifier.from_config(path, taxonomy) -> Classifier`.

- [ ] **Step 1: Write failing tests:**
  - `test_calibrate_reduces_nll`: a KeywordBackend wrapper with deliberately bad temperature 10.0 on `small.jsonl`; after calibration the mean NLL computed by the same helper is lower and `clf2.settings.temperatures["keyword"] < 10`.
  - `test_config_roundtrip`: save, load, then `classify_texts` gives identical results.
  - `test_save_config_unsaved_trained_raises`.
- [ ] **Step 2:** Run and expect a FAIL. Then implement and run again for a PASS. Commit `feat: temperature calibration and classifier config`.
- [ ] **Step 3: Add the benchmark systems.**
  - **few-shot:** for k in {4, 8, 16, 64} and base in {UltraBERT-Embed-150M, bge-small}, two variants: heads-only (`finetune_body=False`) and fine-tuned. Train on `pool_shots(train_pool, k, seed=0)` and evaluate with Beam(3), threshold 0.
  - **strategies (B2 only):** Greedy, Beam(3) and Flat with the best zero-shot backend from Task 13, plus Greedy and Beam(3) with the best 16-shot trained model. Flat runs only for embedding backends.
- [ ] **Step 4: Run in the background:** `uv run python -m benchmarks.run --suite few-shot --device mps` and `--suite strategies`. Expected: two new results files.
- [ ] **Step 5:** Regenerate `docs/benchmarks.md` and apply the decision rules:
  - **Default `train(base_model=...)`:** best mean hF1 at 16 shots.
  - **Default strategy:** Beam(3), unless Greedy is within 1 point of path accuracy on B2 and at least 1.5× faster.
  - **Default `finetune_body`:** keep True unless heads-only is within 1 point at 16 shots.

  Update the defaults in `train`, `Classifier` and the CLI, and set the embedding recipes' `temperature` to the value from calibrating each one on its B1 train pool. Record each decision with a link.
- [ ] **Step 6:** Commit `bench: few-shot and strategy results; set defaults`.

### Task 16: PDF loader, OpenAI-compatible backend

**Files:**
- Create: `src/docrooter/document/pdf.py`, `src/docrooter/backends/openai.py`, `tests/unit/document/test_pdf.py`, `tests/unit/backends/test_openai.py`, `tests/fixtures/docs/bookmarked.pdf` (generated in a test fixture with PyMuPDF; skipped when the extra is absent)

**Interfaces:**
- Produces:
  - `load_pdf(path, *, min_heading_score=0.5, repair=True) -> Document`
    - Markdown comes from `pymupdf4llm.to_markdown(path, page_chunks=True)`.
    - Each page chunk's headings get `RawHeading.page` set to the chunk's page number.
    - If `fitz.open(path).get_toc()` is non-empty, every heading whose normalised title matches a bookmark title takes the bookmark level and flag `"level_from_bookmark"`; headings with no match keep their repaired level.
  - `OpenAIBackend(model: str, *, base_url: str | None = None, api_key: str | None = None, timeout: float = 60, concurrency: int = 8)`
    - `base_url`/`api_key` default to the `DOCROOTER_LLM_BASE_URL` / `DOCROOTER_LLM_API_KEY` env vars, then `OPENAI_BASE_URL` / `OPENAI_API_KEY`.
    - `max_candidates=50`, kind `softmax`, `temperature=1.0`.
    - The user message is `build_choice_prompt(..., answer="", markers=[str(i) for i in range(1, n + 1)])`, and the system message is `Answer with the option number only.`
    - Requests set `logprobs=true, top_logprobs=20, max_tokens=3, temperature=0`.
    - With logprobs: probabilities are the softmax over the first token's top_logprobs restricted to option numbers, with missing options set to `-inf`.
    - Otherwise: parse the first integer, which gets 1.0. An unparseable answer gives a uniform distribution and one logged warning per run.
    - Calls run concurrently via `httpx.AsyncClient` and a semaphore of `concurrency`.

- [ ] **Step 1: Write failing tests:**
  - `test_pdf_bookmark_levels` (skip if `pymupdf4llm` is missing): the fixture PDF has bookmarks at levels 1/2/1, and the resulting TOC levels match them.
  - `test_openai_logprob_parsing` uses `httpx.MockTransport` returning a canned completion with top_logprobs `{"2": -0.1, "1": -2.5}` and expects argmax = candidate index 1.
  - `test_openai_fallback_parse` (no logprobs, content `"2"`).
  - `test_openai_garbage_uniform`.
- [ ] **Step 2:** Run and expect a FAIL. Then implement and run again for a PASS.
- [ ] **Step 3:** Commit `feat: PDF loader with bookmarks; OpenAI-compatible backend`.

### Task 17: README, packaging, publish workflow

**Files:**
- Modify: `README.md`, `pyproject.toml` (classifiers, URLs, `all` extra), `src/docrooter/__init__.py` (final `__all__`)
- Create: `.github/workflows/publish.yml`, `tests/unit/test_public_api.py`, `docs/taxonomy-formats.md`

- [ ] **Step 1: Write the failing test** `test_public_api.py`:

```python
def test_public_api_surface():
    import docrooter as dr
    for name in ["load", "Document", "Section", "Taxonomy", "TaxonomyNode", "SectionView", "Classifier",
                 "Greedy", "Beam", "Flat", "Dataset", "train", "TrainedModel", "evaluate", "compare", "calibrate"]:
        assert hasattr(dr, name), name

def test_core_import_does_not_import_torch():
    import subprocess, sys
    out = subprocess.run([sys.executable, "-c", "import docrooter, sys; print('torch' in sys.modules)"],
                         capture_output=True, text=True)
    assert out.stdout.strip() == "False"
```

- [ ] **Step 2:** Run it and expect a FAIL if any export is missing. Fix the exports, then run again for a PASS.
- [ ] **Step 3:** Write the README:
  1. One-paragraph pitch.
  2. Install matrix (extras table, with the AGPL note on `[pdf]` and a `trust_remote_code` note for UltraBERT).
  3. A 10-line quickstart: load, Taxonomy, Classifier, classify, print.
  4. CLI examples for `toc`, `taxonomy validate`, `classify`, `train` and `eval`.
  5. Taxonomy formats (link to `docs/taxonomy-formats.md`, with one example of each of the five formats).
  6. Training and evaluation walkthrough, including bootstrapping labels with `Dataset.from_results`.
  7. Backends table.
  8. Benchmarks summary linking to `docs/benchmarks.md`.
  9. Development setup (`uv sync`, `uv run pytest`, `-m models`).
- [ ] **Step 4:** Write `publish.yml`: on release published, `uv build`, then `pypa/gh-action-pypi-publish` with trusted publishing (`id-token: write`). Do not create a release or publish; that is the user's decision.
- [ ] **Step 5:** Run `uv build && uv run pytest && uv run ruff check .` and expect a PASS, plus a wheel in `dist/` whose `.whl` contains no `benchmarks/`.
- [ ] **Step 6:** Commit `docs: README, taxonomy formats, publish workflow`.
