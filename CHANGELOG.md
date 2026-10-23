# Changelog

All notable changes to docrooter are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) from 1.0.0 on. Areas marked
*experimental* may still change in a minor release; everything else changes only in a major one.

1.0.0 is the first tagged release. 0.1.0 and 0.2.0 were development versions: they were never
tagged or published, and their entries are kept as history. The package version is read from
`docrooter.__version__`, the one place it is set.

## [Unreleased]

### Added

- **`docrooter.parse` and `docrooter parse`: a document's type, sections and section text as
  one JSON response.** `include="type"` returns the document type from the document-type call
  alone (no section tags); `"sections"` adds every tagged heading with its tag; `"text"` adds the full
  text of each tagged section, grouped by tag, at no extra model call. The application stores
  the JSON and its extraction step reads the text, with no call back to doc-rooter.
  `ParseResult.get_section` and `get_sections` look a tag up in a stored response and report
  `found`, `not_found` or `unknown_tag`. Exports `ParseResult`, `TaggedSection`,
  `SectionMatch` and `SectionContext`. See the guide "Parse a document for extraction" and
  `examples/parse_for_extraction.py`.
- **`section_text` joins neighbouring sections with the same tag.** Flat PDF headings used to
  split one section into dozens of entries (one annual report's notes came back as 42). Now
  sections with the same tag and no differently tagged section between them are one entry; a
  heading tagged `none` ends the run and an untagged one does not. `SectionMatch.section_ids`
  lists the sections joined, and the confidence is the lowest known one.
- **`extra_body` for the OpenAI-compatible backend.** `OpenAIBackend(extra_body=...)`, or the
  JSON object in `DOCROOTER_LLM_EXTRA_BODY`, adds provider fields to every request without
  replacing the request's own, e.g. `{"thinking": {"type": "disabled"}}` for a Claude model that
  would otherwise spend its small batched reply budget thinking.
- **Jev reports exact tokens.** `JevBackend.usage()` sums the input and output tokens every
  request reported, so a `classify` run on Jev has exact token counts.
- **Jev rows in the classification benchmark**, on the full main grid and the held-out set,
  per item with probabilities; Jev is the demo's model.
- **A `government` group in the section catalogue** for audit and oversight reports: highlights,
  objectives scope and methodology, findings, recommendations, agency comments. Also
  `research.references` and `financial.comprehensive_income`.
- **Hosted-LLM rows in the classification benchmark.** Claude Haiku 5.5 through Anthropic's
  OpenAI-compatible endpoint, on the full main grid (exact tokens) and once on the held-out set.
  The report gains a generated "Sections first against doc first" section.

### Removed

- **docrooter's heuristic PDF heading detector.** The section tree of a PDF (`load`,
  `load_pdf`, the CLI, the playground and the `fast` profile) is now read with PyMuPDF4LLM: its
  layout model finds the headings and their levels, page headers and footers are left out, and
  docrooter only cleans the titles, repairs level jumps and applies the PDF's bookmarks. Gone
  with it: the weighted heading scorer and the line analysis behind the section tree, the
  `engine="both"` merge, the `pdfplumber` engine outside element mode, and the
  `min_heading_score` and `use_printed_toc` options outside element mode. `engine` takes
  `auto`, `pymupdf` or `pymupdf4llm` (all the same reader); `ocr="auto"` lets PyMuPDF4LLM OCR
  pages without text. Element mode (`elements=True`) keeps its native line reader, with
  `pdfplumber` as its MIT option. Reading PDFs now always needs the AGPL `pdf` extra.

### Changed

- **Section tags normalise a heading's meaning (catalogue v2).** The generic `general` group
  (summary, introduction, background, body, findings, conclusion, references, appendix, ...)
  is gone: a heading with nothing to normalise is tagged `none`, a top-level leaf, and `none`
  is never evidence for the doc-type rules (their evidence share counts domain tags only). An
  unanswered item falls back to `none`, not `general.other`. The government-report rule needs
  a `government` tag; the white paper has no rule (the model decides it). Pass your own
  catalogue to tag structure too.
- **`classify` tags every heading at any depth**, level 1 first, then level 2 and so on, up to
  `max_sections=400` (was 40 sections at levels 1 and 2). The PyMuPDF4LLM reader often nests a
  balance sheet or the notes at level 3 to 6, where they were never tagged.
- **Headings past the cap inherit.** A section past `max_sections` takes its nearest tagged
  ancestor's domain tag and confidence (`confidence_source="inherited"`, no tokens): the notes
  under a tagged "Notes to the financial statements" are notes. Inherited tags are never
  evidence for the doc-type rules.
- **`doc_first` tags against the section catalogue**, as `sections_first` does: the two flows
  tag sections alike and differ only in how the type is decided.
- **A section view's path holds the section's ancestors** when the view also has a title line
  (it repeated the title), and when the budget is tight the outermost ancestors go first,
  marked `…`, so a deep heading's title always fits. At the `headings` rung (32 tokens) a deep
  heading's title was cut.
- **Section gold v2 and abstaining scores.** The benchmark's section gold is 2,194 headings at
  every depth of the PyMuPDF4LLM reading, labelled against catalogue v2 (Claude annotators,
  439 double-annotated, leaf κ 0.914). Section precision, recall and F1 treat `none` as an
  abstention.
- **The default `level` is measured per backend.** `classify(level=None)` (now the default), the
  CLI and the UI use `docrooter.flow.default_level`: the cheapest rung within 2 points of `full`
  in the benchmark. That is `headings` for the OpenAI-compatible backend (measured on Claude Haiku
  5.5) and the keyword backend, and `lead` for UltraBERT, the default backend; any other backend
  keeps `terms`. Pass
  `level="terms"` for the old behaviour.
- **A refused `temperature` is dropped.** A server that answers HTTP 400 naming `temperature`
  (Claude Haiku 5.5 does) is asked again without it, and never sent it again, as
  `response_format` already was.
- **`classify` narrows only on adequate section evidence.** In `sections_first` the doc-type
  rules may narrow or decide only when at least `max(min_evidence, ceil(min_evidence_share ×
  tagged sections))` sections were tagged with confidence (at least `rule_min_conf`; a valid
  batched or letter answer counts). `min_evidence` (2) and `min_evidence_share` (0.2) are new
  `classify` keywords, and `docrooter.rules.evidence_needed` computes the gate. With fewer, the
  model chooses among every type (`decided_by="fallback"`, flags `rules_unmet` and the new
  `thin_evidence`). A few confident tags among many sections narrowed the options to types that
  excluded the true one.
- **`low_support` widens to every doc type.** A lone undecided candidate or near miss, or a
  candidate with too little support to decide on, is now offered with every type
  (`decided_by="model"`, flag `low_support` kept; `candidates` keeps the rules' suggestion)
  instead of with its doc-type siblings, which excluded the true type on sparse evidence.
- **`classify` call budgets follow the backend's token ratio.** Every budget is divided by
  `token_ratio()`, the exact prompt tokens the OpenAI-compatible backend has spent per estimated
  token (1.0 to 3.0), recorded per call as `CallRecord.token_ratio`.

### Fixed

- **PDF headings get their bookmark levels despite small differences in the bookmark.** A
  bookmark with a typo, a plural or joined words ("Goverance", "Reports of …",
  "STOCKHOLDERMATTERS") now matches a heading on the same or next page, so Item 12 no longer
  sits inside Item 11. A bookmarked heading the reader saw only as body text becomes a heading
  (flag `heading_from_bookmark`), and the headings after a bookmarked one can no longer sit
  levels deeper than it. PDFs without bookmarks read as before.

## [1.0.0] - 2026-10-20

docrooter 1.0 gives a hosted LLM the minimum context it needs to classify a document. It tags
the sections first, against one shared section catalogue, and lets their tags decide the
document type or narrow the types the model is offered. What it sends is set by a context
ladder (`headings`, `terms`, `lead`, `full`), it asks the LLM in batched calls that list the
options once, and it reports the tokens it spent, exactly when the API returns `usage`. Local
models are a free stand-in for testing, and the corpus is for validation, not training.

This release also ships the work done after 0.2.0 that was never released on its own (memory
containment, events, the playground, the held-out evaluation, elements and the rooter, element
tags, selection); it is listed below the v1 entries.

### Added

- **`docrooter.classify(source_or_doc, *, flow="sections_first", level="terms", calls=None,
  backend=..., log=None)`**, the v1 headline API, returning a `Classification`. See
  [Sections first](docs/site/guides/sections-first.md).
  - **`sections_first`** (the default) tags the level-1 and level-2 sections (at most 40,
    `max_sections`: level 1 first, then level 2, in reading order) against the section
    catalogue, group first, then leaf. It matches the doc types' section rules against the tags.
    One candidate whose supporting tags are confident enough is decided by the rules with no
    document-level call (`decided_by="rules"`); several go to the model among those candidates
    only (`"model"`). A lone undecided candidate is offered with its near misses (types missing
    one required group, flag `near_miss`) and, when it is still alone, with its doc-type
    siblings that no `excludes` hit argues against (`low_support`). With no candidate the near
    misses are offered, and with nothing to offer the model chooses among every type
    (`"fallback"`, flagged `rules_unmet`). Waived excludes are flagged `excludes_waived`.
  - **Answers without probabilities are gated like batched ones.** Batched answers, and
    per-item answers read from a bare letter because the endpoint returned no logprobs
    (confidence `None`, `confidence_source="text"`), cannot pass the confidence gate: a lone
    candidate then needs two supporting sections. A reply that names no option is never read as
    the first option: the item gets no type (or the `other` section leaf at confidence 0) and the
    result is flagged `unparsed`.
  - **`doc_first`** classifies the type among every doc type from the document view, then labels
    the sections with that type's section taxonomy (the 0.2 behaviour, through the new API).
  - `Classification` carries the `doc_type` with its confidence and probabilities, `decided_by`,
    the `candidates` offered, the `flags`, the section tags (`SectionTag`: label, confidence,
    level used, tokens), one `CallRecord` per document call (`document_calls`) and per
    sections-stage call (`section_calls`), the `levels` used, `tokens` (in total and per stage),
    `tokens_split` (the estimated share of options, content and instructions) and how the
    backend was asked (`calls`).
  - `section_level` and `doc_level` set the two steps apart; `cascade=True` asks the
    low-confidence items again one rung up (per-item calls only); `doc_types=` and `catalogue=`
    replace the built-in taxonomies.
- **The context ladder** (`docrooter.ladder`, `LEVELS`): four named levels, `headings`, `terms`,
  `lead` and `full`, each an exact, documented `SectionView` and `DocumentView` preset; each rung
  shows everything the one below shows and more. `full` sends the whole document text, cut
  head-first to a view budgeted for a 16,000-token target prompt (`full_cap`); the benchmark
  measures how often a measured prompt still exceeds it. `DocumentView` gains a `text` field. See
  [The context ladder](docs/site/concepts/context-ladder.md).
- **`level=`** on `classify_document`, `Classifier.classify_documents` and `normalize_sections`
  (with `doc_level=`) picks a ladder rung instead of a view. `level=None`, the default, keeps the old
  views, and passing both `level=` and `view=` raises `ValueError`.
- **New taxonomies.**
  - **`section_catalogue`**: one shared section taxonomy of 112 leaves in 12 domain groups
    (financial, narrative, disclosure, fund, offering, contract, law, research, standard, technical, ESG,
    general), built from the per-type section taxonomies plus finer financial statements
    (balance sheet, income statement, cash-flow statement, changes in equity, schedule of
    investments, auditor's report, notes, account statement) and the shareholder letter. Every
    group has an `other` leaf, and every leaf a definition and examples.
  - **Doc-type section rules**: every leaf of `doc_types` has a `sections` block (`requires`, a
    list of groups of catalogue ids of which each must be met; `typical`, an ordering only; and
    `excludes`), written from the definitions, never from the corpus (`docrooter.rules`). Rules
    are left out of `Taxonomy.content_hash`, so trained bundles stay valid. **Doc-type rules
    were revised once after a reviewer checked them against the section gold. Gold-split
    doc-type numbers are therefore development numbers; the held-out run is the clean
    measure.** The revision was made from the definitions: `legal.terms_policy` now needs a
    section only published terms carry (privacy and data use, or acceptable use and user
    terms), so ordinary contract clauses no longer make a terms policy; `legal.contract` needs a
    core clause (obligations, term or termination, payment, or the operative provisions), not a
    titled parties or signatures section; an earnings release needs headline figures as well as
    statements or an outlook; `general.body` no longer meets a government report or white paper
    alone; and filing items, MD&A, notes and an auditor's report bar types that never carry
    them (investor presentation, sustainability report, earnings release). Two catalogue
    definitions were sharpened with it, after the section gold was labelled:
    `contract.acceptable_use` ("Acceptable use and user terms") covers acceptance of the terms,
    accounts and changes to the terms, and `financial.financial_highlights` sends a
    non-financial report's highlights to `general.summary`. Three government-report gold
    sections (`gao-gao-05-251#s2`, `gao-gao-06-681t#s2`, `gao-gao-05-572t#s1`) were labelled
    `financial.financial_highlights` under the earlier wording and now disagree with it.
- **Batched LLM calls** (`docrooter.batch`). With the OpenAI-compatible backend, `classify`
  lists the options once per call instead of once per section: `calls="batched"` (the default
  for a backend that can batch) sorts the sections into catalogue groups in one call, then asks
  one call per group among that group's leaves. Calls are bounded by size as well as count: at
  most 40 sections and `max_call_tokens` (16,000 by default, packed to an estimated 85% of it)
  per call, a section view too large for any call being cut, sent alone and flagged
  `truncated`; per-item calls are bounded the same way, and the document view of every level is
  cut to fit beside the document call's own instructions and options (`doc_truncated`), with a
  `ValueError` when the options leave no room. The bounds are on estimated tokens; the
  benchmark measures how often a measured prompt still exceeds 16,000. With per-item calls, a
  doc-type taxonomy with more types than the backend takes options per call (26 for the
  OpenAI-compatible backend) raises `ValueError` up front, where it used to fall back
  silently;
  `calls="batched_flat"` lists every leaf in one call; `calls="per_item"` asks one item at a time,
  with probabilities. A reply naming an id that is not an option is asked again once, then
  tagged with confidence 0 and flagged `unparsed`.
- **Exact token accounting.** Every LLM call records `prompt_tokens` and `completion_tokens` from
  the API's `usage`, and results carry `tokens.prompt`, `tokens.completion` and `tokens.calls`
  per document, per stage and per section. A total is `exact` only when every call was, and a
  section's share of a call that carried several sections is always an estimate; backends
  that make no LLM call (keyword, embedding, NLI) report what the same views and options would
  cost an LLM, marked estimated.
- **The results log.** `classify(..., log=path)` and `--log` append one JSON line per decision:
  the level and view hash, the options, the label and probabilities, the model, endpoint host and
  prompt hash, the tokens and a `trainable` flag (always false for Jev; otherwise the provider's
  terms decide). API keys never reach it: secret-named fields are hidden at any depth (also
  inside a mapping nested in the options, label or probs), `name=value` secrets in free text are
  redacted, and the backend's key and the key environment variables are scrubbed wherever they
  appear (from 8 characters; a shorter literal would corrupt the log).
- **`docrooter classify FILE`** runs the v1 flow: `--flow`, `--level`, `--section-level`,
  `--doc-level`, `--calls`, `--backend`, `--log` and `--json`. With `-t TAXONOMY` it classifies
  the sections against that taxonomy, as before. The playground can run either flow at a level.
- **A free local stand-in for a hosted LLM**: Qwen3-4B-Instruct-2507 (4-bit) served by
  `mlx_lm.server` through the OpenAI-compatible backend, the same code path as a hosted model;
  the recipe is in [the LLM guide](docs/site/guides/llm-backends.md#a-free-local-stand-in).
- **The classification benchmark** (`benchmarks/classification_bench.py`): accuracy against
  tokens for every flow, level and backend on the 113 documents of the gold split, against a
  Claude-annotated, evaluation-only section gold (951 sections, κ 0.923 at leaf). The annotators
  were all Claude (the same underlying model), so κ measures consistency, not correctness.
  Every call of both stages counts toward the over-cap measure. Results:
  [docs/benchmarks-classification.md](docs/benchmarks-classification.md) (held-out doc type with
  UltraBERT at `lead`: 82.5%; the local Qwen3-4B ran three configurations only).
- **Release packaging.** The `publish` workflow builds the sdist and wheel on a `v*` tag and
  attaches them to the GitHub release; uploading to PyPI is a manual workflow run with trusted
  publishing, so no PyPI token is stored (see [CONTRIBUTING.md](CONTRIBUTING.md#releasing)). `examples/classify_sections_first.py` runs the
  v1 flow with no model, and the docs gain [The context ladder](docs/site/concepts/context-ladder.md)
  and [Sections first](docs/site/guides/sections-first.md).

### Experimental

These areas ship in 1.0 but are not covered by its stability promise, and their numbers are
pending:

- **Selection and the completeness dial**: `select()`, `Selection`, `Selector`, `Field` and the
  field sets, `docrooter select` and `docrooter fit-selector`.
- **The learned element tagger**: element tags, `docrooter.teach`, `docrooter.tagger`,
  `docrooter.corpus`, `docrooter corpus`, `teach`, `train-elements`, `tag` and `tag-corpus`,
  and the training-corpus tooling under `benchmarks/training_corpus/`.

### Added (since 0.2.0)

- **Element tags (v0.5, experimental).** Each paragraph, list item, table, caption, footnote and formula can
  carry a **topic** (one leaf of `element_topics`: a tree of document family, then section) and
  **functions** (any of the twelve tags of `element_functions`) in `Element.tags`, a dictionary of
  `TagResult` (`labels`, `probs`, `confidence`, `source`, and for topics `family_probs`). Set the
  rooter's `element_tags` task to a trained bundle directory, `embedding:<model>` (a zero-shot
  baseline) or `teacher:<engine spec>`; it is `none` in every profile, and an untagged element
  serialises as in v0.4. `docrooter tag FILE --bundle PATH [--json]`,
  `docrooter tag-corpus DIR --bundle PATH` and `docrooter serve --element-tags BUNDLE` (tag chips
  in the playground's Elements tab) use it. See
  [docs/site/concepts/element-tags.md](docs/site/concepts/element-tags.md).
- **Train your own tagger (experimental).** `docrooter corpus build DIR` reads PDFs into a resumable corpus store
  (the elements of the first and some sampled pages); `docrooter teach label DIR` has a local
  teacher LLM label topics (by multiple-choice log-probability, among an embedding shortlist) and
  functions (yes or no); `docrooter train-elements DIR --teacher SPEC --out BUNDLE` runs the
  labelling, a confident training set (term mining, a similarity-gap filter, a keep rule) and the
  student's training and calibration, every stage resumable. The teacher is
  Qwen3-4B-Instruct-2507 (Apache-2.0, 4-bit) on MLX by default (extra `teach-mlx`, Apple silicon);
  `hf:` (extra `models`) and OpenAI-compatible `openai:` (extra `llm`) teachers work too.
  **Jev cannot be a teacher**: TypeSafe MCA §2.3(b) forbids distillation, and docrooter refuses it
  with a clear error; a teacher you bring is under its provider's terms. See
  [Train your own element tagger](docs/site/guides/train-your-tagger.md) and
  `examples/train_element_tagger.py` (a tiny corpus, no model).
- **`financial.fact_sheet`**, a nineteenth document type: a short periodic summary of one fund or
  share class. Fund documents have their own section taxonomy, `section_types/fund_document`.
- **The training corpus** (a manifest and fetch scripts, `benchmarks/training_corpus/`; no PDF is
  committed): 1,100 public native-PDF documents with 81,822 pages (925 train, 62 calibration, 113
  gold), and a gold set of 900 elements whose labels are **Claude-annotated** (topic Cohen's
  kappa 0.953 on 180 double-annotated elements; the annotators share one underlying model, so
  kappa measures consistency, not human-verified correctness). EDGAR could not be used (the SEC
  refuses the project's User-Agent), so annual reports come from issuers' own investor-relations
  sites, mostly non-US. CUAD contracts (CC BY 4.0) are attributed in the corpus README.
- **Element-tag benchmarks.** The harness and its method are in
  [docs/benchmarks-element-tags.md](docs/benchmarks-element-tags.md); the results are pending.

- **Selection and the completeness dial (v0.6, experimental).** `select(analysis, need, completeness=0.9)`
  says which elements a downstream LLM should read for a need, and your application does the
  extraction. See [The completeness dial](docs/site/concepts/completeness-dial.md) and
  [Select what to send to an LLM](docs/site/guides/select-for-extraction.md).
  - **Fields.** A `Field` names the element topics (and functions) that carry a target.
    Built-in sets: `annual_report` (15 fields), `fund_document` (12) and `contract` (15);
    `builtin_fields`, `load_fields` and `validate_fields` read and check your own, in JSON.
  - **`select()`** takes a field, a list of fields of one family or a free-text query, and
    returns a `Selection`: the element ids in reading order, the `chunks`, `tokens` and
    `token_share`, `expected_recall`, `relevant_expected`, `calibration_n`, and the flags
    `guarantee`, `capped`, `doc_type_mismatch`, `uncalibrated_field`, `multi_field`,
    `fit_mismatch` and `over_budget`. `readout()` is its one-line summary. `completeness=1.0`
    is always the whole document and needs no model. A `budget` (tokens) replaces the dial.
    Each selected element brings its context (the previous element, the section's heading and
    first paragraph, a table's caption), so expansion can only raise recall.
  - **The dial** is conformal risk control, per element family: on documents like the calibration
    set, the average share of relevant elements kept is at least `c`, averaged over documents and
    over the family's fields (units are pooled across fields, so one field can sit below `c`).
    **It is calibrated against teacher relevance, not human labels**, it is an average (some
    documents are below `c`), its 1/(n+1) slack caps it only above `c = n/(n+1)` (the whole
    document is returned, `capped=True`), and its units come from shared documents, so they are
    correlated. It holds only for **one field** on a document **read and tagged as the selector
    was fitted**: the selector records its fit provenance (the tagger bundle's fingerprint, which
    `TagResult.fingerprint` carries on every tag, the embedder spec, and the profile, layout
    engine and sections backend of the reading), and `select()` lists any difference in
    `fit_mismatch` (for example `["layout"]` for the default `balanced` profile against a selector
    fitted with `accurate`, or `["sections"]` when escalation is on). Backends are compared by
    their full spec, revision and recipe included (`EmbeddingBackend.spec`). Below `c = 1.0` it
    gives no guarantee, only an estimate, for free-text queries, for several fields at once
    (`multi_field=True`: select once per field), without a tagger or embedder, with another
    tagger, embedder or reading than the fit's, on a
    document type that does not match the field's family, under a budget, for a family without
    a dial, or for a field it was not calibrated on (`uncalibrated_field=True`: the selector
    records the fitted fields that gave the dial at least one unit, by id and a fingerprint of
    their topics, functions and need text; a `selector.json` without that record, or without
    provenance, guarantees nothing). `c = 1.0` is always the whole document and reports
    `guarantee=True`.
  - **`docrooter fit-selector`** fits the relevance models (on calib-fit) and the dials (on
    calib-crc) from the teacher's full labels, which `docrooter teach label --full` writes
    (`labels/teacher-full.jsonl`, never used to train the tagger). There is no built-in selector:
    fit one on your corpus.
  - **CLI and playground.** `docrooter select FILE --need ... [--completeness] [--budget]
    [--profile] [--tagger] [--selector] [--json]`, and `docrooter serve --selector` for a dial
    panel: field picker, query box, slider, the presets Lean (0.8), Balanced (0.9), Thorough
    (0.98) and Whole document (1.0), a readout with badges, and *Show selected only*. A corrupt
    `selector.json` is an `error:` line, and a mistyped field id a usage error.
  - **Benchmarks.** `benchmarks/selection_bench.py` (recall against the promise, tokens against
    baselines, on the fit-identical reading and on the runtime path users run),
    `benchmarks/evidence_qa.py` (evaluation-only public QA evidence) and a demo
    extraction harness, `benchmarks/extraction/` (not in the wheel). The results are pending:
    [docs/benchmarks-selection.md](docs/benchmarks-selection.md).
  - **Example.** `examples/select_for_extraction.py` runs `select()` with stand-ins and no model.

- **Elements.** Every visible part of a document is a typed `Element`: title, heading,
  paragraph, list item, table, figure, caption, footnote, formula, contents and page furniture,
  each with page boxes, a place in reading order and its section. `load_pdf(path,
  elements=True)` and `load(path, elements=True)` build them (Markdown too), and
  `Document.elements`, `Section.element_ids` and `Document.meta["routes"]` carry them. Two-column
  pages are read column by column (XY-cut), and a gutter between the row-aligned labels and
  numbers of an unruled table is not cut. In element mode running headers, footers and page
  numbers move to the start or end of their page, by design. See
  [docs/site/concepts/elements.md](docs/site/concepts/elements.md).
- **The rooter.** `process()` and `process_many()` run one pipeline under a `RouterConfig` (Python,
  TOML or JSON, with the precedence call arguments, file, profile) and return an `Analysis`. Three
  profiles: `fast` (the v0.3 pipeline, byte-identical and within 5% of its speed), `balanced`
  (the default: heuristic elements, OCR, summaries) and `accurate` (the heron layout model, with
  escalation to Jev when `TYPESAFE_API_KEY` is set). Each page is routed to its text layer or to
  OCR, and every decision is recorded in the route log.
- **OCR.** Three engines behind one protocol: `tesseract` (the system binary), `ocrmac` (extra
  `ocr-mac`) and `rapidocr` (extra `ocr`); `register_ocr_engine` adds your own. Measured
  character error rates: tesseract 0.058, rapidocr 0.089, ocrmac 0.102.
- **Layout model.** Extra `layout`: Docling's heron model on onnxruntime, no torch
  (`layout="heron"`; `register_layout_engine` adds others). On DocLayNet test pages its macro F1
  is 0.736 against 0.257 for the heuristic rules.
- **Summaries, chunks and embeddings.** `summarize` (key sentences and key phrases, no model),
  `chunk()` (section-aware chunks that keep tables whole, with their caption, and start with their heading path) and
  `embed()` (element or chunk embeddings, with the model and revision recorded).
- **Surfaces.** `docrooter elements`, `docrooter process [--profile] [--config] [--chunks]` and
  `docrooter config show`; a profile selector and an Elements tab in the playground; guides for
  [OCR and layout models](docs/site/guides/ocr-and-layout.md) and
  [chunking for RAG](docs/site/guides/chunking-for-rag.md); `examples/elements_and_chunks.py`.
- **Element benchmarks.** DocLayNet element kinds, reading order, OCR and routing, and speed:
  [docs/benchmarks-elements.md](docs/benchmarks-elements.md). Per 100 pages, `fast` takes 0.99 s,
  `balanced` 4.64 s (PyMuPDF) and `accurate` 64.6 s; `tables="none"` is the speed switch.
- **Memory budget.** `docrooter.configure(memory_budget_gb=..., threads=..., device=...)` and the
  `DOCROOTER_MEMORY_BUDGET_GB`, `DOCROOTER_THREADS` and `DOCROOTER_DEVICE` variables set
  process-wide limits (8 GiB by default). On Apple silicon, Metal gets the budget minus 1.5 GiB of
  host headroom. The cap applies to the whole process, configured or not: the first model load
  sets `PYTORCH_MPS_HIGH_WATERMARK_RATIO` and `PYTORCH_MPS_LOW_WATERMARK_RATIO` in `os.environ`
  (subprocesses inherit them) unless you set either one yourself. A model that runs out of MPS
  memory moves to the CPU once, with the same weights, and logs a warning; fine-tuning first
  retries with half the batch.
- **Releasing models.** Every backend has `close()`; `Classifier` and `TrainedModel` close their
  models too, and `Classifier` works as a context manager. `classify_document` and
  `normalize_sections` close the backends they load from a spec string.
- **Large corpora.** `docrooter doctype --chunk-size N` parses, classifies and writes files `N` at a
  time, and `--max-pages N` (`load_pdf(max_pages=N)`) skips the pages past `N` and records
  `"truncated_pages"` in `meta["warnings"]`.
- **Pipeline events.** `load`, `load_pdf`, `classify_document`, `Classifier.classify_documents` and
  `normalize_sections` take `on_event`, a callback that receives an `Event` for each stage
  (`extract`, `signals`, `headings`, `assemble`, `view`, `model_load`, `classify`, `normalize`);
  `docrooter.events.pipeline_run` shares one clock across chained calls. A sink that raises never
  breaks processing.
- **Playground.** `docrooter serve` starts a local web page (extra `ui`) where you upload a
  document, pick a backend and watch the stages run. It binds `127.0.0.1` by default, has no
  external assets, caps uploads at 50 MB and deletes them after processing. `--env-file` reads
  missing variables such as `TYPESAFE_API_KEY` from a file. See `SECURITY.md` for its guard rails.
- **Held-out evaluation.** A frozen 40-PDF corpus (`benchmarks/heldout/`), labels committed before
  any run, and one recorded run of the TOC, document-type and normalisation benchmarks on it:
  [docs/benchmarks-heldout.md](docs/benchmarks-heldout.md). The dev-versus-held-out comparison is in
  [docs/benchmarks-pdf.md](docs/benchmarks-pdf.md). Headline results:
  - Document type generalises: Jev given the `DocumentView` scores 0.925 leaf accuracy on the 40
    held-out PDFs (0.961 on dev) and UltraBERT zero-shot 0.825 (0.765 on dev).
  - TOC extraction drops: the default engine's micro F1 at depth <= 2 is 0.576 (0.703 on dev),
    partly real and partly bookmark artefacts.
  - Section normalisation with the local models holds (UltraBERT accuracy 0.579 against 0.514);
    Jev is slightly lower (0.692 against 0.725).
- **Performance notes**, with peak process memory for classifying a 642-page datasheet and for
  fine-tuning on EDGAR items: [docs/performance.md](docs/performance.md).
- **Repository essentials.** `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `SECURITY.md`, issue and pull request templates, a pre-commit configuration, the logo, wordmark
  and favicon under `docs/assets/`, and a README header with badges.
- **CI.** The test matrix covers Python 3.10 to 3.13, `ruff format --check` and `mypy` run, and an
  `extras-smoke` job installs the `pdf`, `pdf-mit`, `ui` and `llm` extras (no torch) so their
  tests run. That job also measures coverage and fails the build under 80% line coverage on any of
  `docrooter.document`, `taxonomy`, `classify`, `views` or `pdf` (96% to 100% at the time of
  writing). The publish workflow runs the tests before it uploads.
- **Typing.** The package ships `py.typed` and `mypy src/docrooter` is clean (the optional extras
  are configured as untyped imports in `pyproject.toml`).
- **Documentation site** (`mkdocs.yml`, pages in `docs/site/`): getting started, concepts, guides,
  a CLI reference generated from `--help` (`docs/gen_cli.py`), an API reference generated from the
  docstrings, the benchmark reports, and an FAQ. Every Python sample on it is a line of a script in
  `examples/` (ten new ones). All but `memory.py` (UltraBERT) and `openai_backend.py` (an
  OpenAI-compatible server) run with the keyword backend, some need the `pdf` extra, and
  `tests/unit/test_docs_snippets.py` runs those eight and checks every CLI option the pages
  mention. A `docs` workflow builds it with `mkdocs build --strict` on pushes to
  `main` and on pull requests; nothing is deployed yet.
- **`docrooter --version`** prints the installed version.
- The playground uses the new mark (`favicon.svg`) as its tab icon.

### Changed

- **Version 1.0.0**, the first tagged release, with the project URLs of
  <https://github.com/LandoCole/docrooter>. The package description, README and docs are
  rewritten around the minimum-context pitch.
- **`docrooter classify FILE` without `-t`** runs `docrooter.classify` (sections first, level
  `terms`) instead of failing for want of a taxonomy. `docrooter --help` gives the v1 pitch, and
  the experimental commands say "(experimental)" in their help.
- **`docrooter.classify` is now the function, and it shadows the `docrooter.classify`
  subpackage as an attribute** (as `select`, `summarize` and `embed` already did).
  `from docrooter.classify import Classifier` and `from docrooter.classify.engine import ...`
  still work; `import docrooter.classify as m` now gives the function,
  `import docrooter.classify.engine as e` raises `ImportError`, and
  `docrooter.classify.Classifier` raises `AttributeError`. Import from the subpackage with
  `from docrooter.classify import ...`.
- **`DocTypeResult` gains `truncated`** (whether the document view was cut to its budget). In
  `mode="sections"` the document view is no longer rendered: `view_text` is `""`, and the
  stage's token count no longer includes the view's tokens.
- **Missing-extra messages give the GitHub install form**, `pip install "docrooter[x] @
  git+https://github.com/LandoCole/docrooter"`, since docrooter is not on PyPI.

- **The `openai` backend letters its options A-Z (it numbered them), and takes at most 26
  options per call (it took 50).** Hierarchical strategies and the catalogue's groups keep
  calls under the limit.
  - The system prompt asks for the option letter.
  - Logprobs are read from the top 10 alternatives (was 20). `mlx_lm.server` allows at most 11.
    Tokenizer space markers (`ĠB`, `▁B`) count as the letter.
  - Without logprobs, only a bare letter reply counts as an answer.
  - What happens to a node with 27-50 children:
    - in a `Classifier`, it now goes to the next routed backend with a warning, or fails when
      there is none;
    - a direct `score` call raises `BackendError`.
  - Use a hierarchical strategy to keep such decisions on the LLM. The backend also reports
    its token usage (`usage()`, `on_call`), and each per-item call's record says how the reply
    was read (`answer`: `logprobs`, `letter` or `none`).
- **Prospectuses use the fund-document sections.** `financial.prospectus` maps to
  `section_types/fund_document` (it was `generic`), as `financial.fact_sheet` does, so
  `normalize_sections` on a prospectus labels its headings with fees, risks and performance
  rather than generic sections. The document-type benchmark numbers in this changelog and the
  docs were measured with 18 types and predate `financial.fact_sheet`; the held-out corpus has
  not been run again.
- **`ocr="auto"` resolves to `tesseract` (the binary on PATH), then `ocrmac` (macOS), then
  `rapidocr`.** Tesseract comes first for accuracy (character error rate 0.058, against 0.089
  for rapidocr and 0.102 for ocrmac, which also fails on legal pages and table columns). ocrmac
  comes before rapidocr for speed and memory (0.73 against 3.04 s per page, 1.2 against 3.2 GiB),
  although rapidocr is the more accurate of the two. The first draft of the rooter preferred
  ocrmac on macOS.
- New extras `layout`, `ocr` and `ocr-mac`, all included in `all`; Python 3.10 gains `tomli` in
  the core dependencies (router config files).
- The UltraBERT recipes read at most 1024 tokens (was 8192), and `train()` truncates to 512 tokens
  (`max_seq_len`) and fine-tunes with gradient checkpointing (`gradient_checkpointing=False`
  restores the old behaviour). Fine-tuning on MPS used to fail (it reached 19.5 GB in a
  development run).
- The embedding cache is an LRU of 50,000 texts (`cache_size=`); it used to grow without bound.
- The pdfplumber engine frees each page after reading it.
- Loading a PDF with the default engine is faster: its four load stages took 19% less time on
  the ten profiling PDFs (0.94 against 1.15 s per 100 pages), and the TOC benchmark's 24 dev
  PDFs load at 1.05 s per 100 pages (see [docs/performance.md](docs/performance.md)).
- The package version is read from `docrooter.__version__` (a hatch dynamic version), so
  `pyproject.toml` and the module can no longer disagree.
- A trained bundle's manifest keeps the recipe's batch size; a half batch from an
  out-of-memory retry is recorded only in `train_metrics`.
- The `ui` extra requires `fastapi>=0.100`, `uvicorn>=0.22` and `python-multipart>=0.0.13`;
  `httpx` joins the dev group; the publish workflow fails when the release tag is not
  `v<__version__>`.
- The `classify_document` / `classify_documents` event docs list `model_load` before `view`,
  and the `configure()` docstring describes its process-wide MPS side effects and the opt-out.

### Fixed

- The playground read its static files with `Traversable.joinpath("static", name)`; the
  multi-segment form needs Python 3.11 for packages that are not plain directories (such as a
  zipped install). It joins one segment at a time now.
- `Classifier.classify_documents` skips a decision that has no best path when it collects stop
  reasons and no-backend warnings, instead of raising `AttributeError`. No built-in engine
  returns one; the guard came with the typing pass.
- The playground answers 400 to a `Content-Length` that is not an ASCII number (it answered 411,
  or 500 for a superscript digit, which `str.isdigit` accepts; a missing one is still 411), and
  ignores such a `Last-Event-ID` instead of failing.
- `docrooter serve --env-file` reads `KEY= # comment` as an empty value instead of `# comment`.
- The pdfplumber engine no longer splits every glyph into its own word when a slightly skewed
  text matrix gives the glyphs sizes that differ by float noise; such titles came out
  letter-spaced and were never heading candidates. Found through the held-out run, fixed on dev
  and synthetic PDFs only; the held-out numbers predate the fix.
- Closing the `as_zero_shot()` backend of an unsaved fine-tuned `TrainedModel` no longer lets
  the next call silently rebuild the base model under the fine-tuned heads; it raises
  `BackendError` until the model is saved.
- The embedding cache stores its own copy of each row, so `cache_size` bounds memory as well
  as entries (a cached row used to keep its whole encode batch alive).
- An MPS out-of-memory error while an NLI or MLM-choice model moves to the GPU falls back to
  the CPU inside the load: one `model_load` start and end, one model build, and a "while
  loading" warning.
- `docrooter serve` stops promptly on Ctrl+C while a page follows a run: the event streams
  are ended, with a 2 s graceful-shutdown bound.
- Playground `.md` and `.txt` uploads are capped at 1 MB, and a run keeps and returns at most
  5,000 TOC and section entries (`truncated` and `toc_total` in the result, shown on the page).
  A heading-dense text file costs about 270 MB of memory per MB, or about 1.6 GB per MB with
  section labels on.
- The playground's pipeline `error` event no longer contains the server's temp-file path.
- A `Content-Length` or `Last-Event-ID` longer than 18 digits is rejected (400) or ignored
  instead of causing a 500.
- The playground page stops waiting when a run is gone after a server restart.

## 0.2.0 - 2026-10-16 (never tagged)

The PDF release: a native table-of-contents engine, document-type classification and section
normalisation, measured on real PDFs.

### Added

- **Native PDF TOC engine.** `load(path)` and `load_pdf(path, engine=...)` find the headings of a PDF
  without OCR, from font size and weight, position, numbering, running headers and footers, printed
  contents pages and (unless `use_bookmarks=False`) the PDF's bookmarks. Engines: `auto`
  (PyMuPDF if installed, else pdfplumber), `pymupdf`, `pdfplumber`, `both` and the legacy
  `pymupdf4llm` path. `Document.meta` records the engine, page count and timings.
- **`pdf-mit` extra** (`pdfplumber`, MIT) as a permissive alternative to the AGPL PyMuPDF, and a
  README note on the licensing difference.
- **Built-in taxonomies.** `doc_types` (18 document types in 6 groups) and one section taxonomy per
  family, with `list_builtin_taxonomies()` and `builtin_taxonomy(name)`; every node has a
  description.
- **Document types.** `DocumentView` (title, page and section counts, headings, opening text and
  keyphrases within a token budget), `classify_document` and
  `Classifier.classify_documents`, which return a `DocTypeResult` with a confidence.
- **Section normalisation.** `normalize_sections` maps each heading to a canonical section of the
  document's type (classifying the type first when none is given).
- **CLI.** `docrooter doctype` (table or `--jsonl`, `--keep` and `--paths-only` to gate a pipeline,
  `--workers`, `--mode view|sections|hybrid`) and `docrooter normalize`; `docrooter toc` takes
  `--engine`.
- **Real-PDF benchmarks** on a 51-PDF development corpus: [docs/benchmarks-pdf.md](docs/benchmarks-pdf.md).
  Headline results, all on the development set that the rules were tuned on:
  - TOC extraction with the default engine: micro F1 0.703 at depth <= 2 against PDF bookmarks
    (24 documents; 0.478 before tuning).
  - Document type over 18 leaf types: TypeSafe Jev given the `DocumentView` 49/51 correct (0.961);
    the best local system, UltraBERT with the hybrid view, 42/51 (0.824).
  - Section normalisation against agent-assigned silver labels (142 sections, 6 documents): Jev 0.725 accuracy,
    UltraBERT 0.514.

### Changed

- `Document` gained a `meta` dictionary; v0.1 serialisation and tests are unaffected.

### Fixed

- Dot-leader tables are not mistaken for contents pages, and running headers and footers that
  repeat on three or more pages are excluded from the headings.
- Doc-type decisions use a single rule, scanned PDFs without a text layer report an empty view
  rather than a guess, and `docrooter doctype` survives a broken process pool.
- Built-in taxonomies are cached (`builtin_taxonomy(name)` still returns a fresh copy), and an
  unknown name raises `UnknownTaxonomyError`.

## 0.1.0 - 2026-10-15 (never tagged)

First release: classify the sections of a document against a taxonomy you supply.

### Added

- **Loading.** `load()` reads Markdown, plain text and PDFs (through PDF bookmarks), validates the
  headings and repairs the hierarchy; `docrooter toc` prints or exports the result.
- **Taxonomies.** A tree of categories with labels, descriptions, examples and per-node settings,
  read from nested JSON, flat JSON or three CSV layouts, with validation that names the row or JSON
  pointer of each problem (`docrooter taxonomy validate`).
- **Classifier.** Traversal strategies `Greedy`, `Beam(n)` and `Flat`; probabilities and the path
  through the tree for every section, rival alternatives, early stops with a reason, multi-label
  nodes, routing of nodes to different backends, and a fallback backend.
- **Backends.** `keyword`, `embedding` (default `dmis-lab/UltraBERT-Embed-150M`), `nli`,
  `mlm-choice`, `trained`, `openai` (any OpenAI-compatible server) and `jev` (TypeSafe Jev), plus a
  protocol for custom backends. Models with their own modelling code run only from a pinned commit
  (`trust_remote_code` gate); API keys are read from the environment and redacted from errors.
- **Training.** Per-node classifier heads on a shared, optionally fine-tuned embedding body
  (`train()`, `docrooter train`), `Dataset` helpers, and temperature calibration (`calibrate`).
- **Evaluation.** Exact-path accuracy, hierarchical precision, recall and F1, selective accuracy and
  throughput (`evaluate`, `compare`, `docrooter eval`).
- **CLI.** `toc`, `classify`, `eval`, `train`, `taxonomy validate` and `models pull|list` (offline
  use with `HF_HUB_OFFLINE=1`).
- **Benchmarks** on EDGAR-CORPUS 10-K items, DBpedia classes and LEDGAR contract clauses
  (`benchmarks/`); zero-shot results in [docs/benchmarks.md](docs/benchmarks.md). On 10-K items
  (n=1000) the UltraBERT embedding backend scores 0.752 hierarchical F1.
- Apache-2.0 licence and a trusted-publishing workflow for PyPI.

[Unreleased]: https://github.com/LandoCole/docrooter/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/LandoCole/docrooter/releases/tag/v1.0.0
