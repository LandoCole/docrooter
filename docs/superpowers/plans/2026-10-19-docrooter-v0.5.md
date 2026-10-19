# docrooter v0.5 Implementation Plan: training corpus, teacher, learned element tagger

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:**
- Build a public training corpus of at least 1,000 PDFs.
- Label a sample of its elements with a local, permissively licensed LLM teacher.
- Fine-tune a BERT-class student that tags every element by topic and function.
- Calibrate the student, plug it into the rooter, and measure it on an independent gold set.

**Architecture:**
- **Corpus store:** `docrooter.corpus`, JSON lines.
- **Teacher:** `docrooter.teach`, with these engines:
  - MLX Qwen3-4B, using multiple-choice log-probabilities;
  - transformers;
  - OpenAI-compatible endpoints.
- **TELEClass loop:** shortlist → teacher → term mining → similarity-gap filter, building on the existing `train()`/`TrainedModel` machinery.
- **Element tagger bundle:** topic tree, function heads and temperatures.
- **Rooter:** a new `element_tags` task.
- **Long runs:** fetching, building the corpus, teacher labelling, training and evaluation are run by the controller in the background, one model process at a time. Implementer tasks build and smoke-test the tools on small slices.

**Tech Stack:** Python ≥ 3.10 (MLX engine: ≥ 3.11, macOS arm64), pydantic v2, numpy, sentence-transformers/transformers/torch (`models` extra), mlx-lm (`teach-mlx` extra), onnxruntime heron (`layout`, for corpus building).

**Spec:** `docs/superpowers/specs/2026-10-18-docrooter-v0.5-learned-tagger-design.md`. Its §4.0 holds the binding rulings from the builder check. The program document is `docs/superpowers/program/2026-10-18-docrooter-element-intelligence-program.md`. The research briefs are `docs/superpowers/research/2026-10-18-*.md`.

## Global Constraints

- All v0.1–v0.4 constraints still hold:
  - the core is torch-free;
  - unit tests are torch-, onnxruntime- and mlx-free;
  - no recursion;
  - secrets stay in `.env`;
  - commands run with `PYTHONPATH=src`;
  - no plain `uv sync`; run `uv lock` after dependency changes;
  - the fast path stays byte-identical to v0.3.
- **Never use Jev as a teacher.** No Jev output may enter training, calibration or teacher data. TypeSafe MCA §2.3(b).
- **Corpus isolation:** never open `benchmarks/heldout/`. No dev-corpus or held-out document may enter the training corpus: exclude by sha256, normalised URL, and (publisher, normalised title).
- **No PDFs are committed.** The corpus is a manifest plus fetch scripts. Corpus stores and bundles live in `benchmarks/.cache/` and are never committed or published.
- **Memory:** at most one model-loading process at a time, within the 8 GB budget (`benchmarks._guard`). This covers heron corpus builds, teacher runs, training and evaluation.
- **Gold set:** `benchmarks/training_corpus/gold.jsonl` is committed before any model scores a gold-split document. Gold labels are Claude-annotated and disclosed as such.
- **Polite fetching:**
  - EDGAR at most 5 requests/s, with User-Agent `docrooter-research/0.5 (research; contact via repository)`;
  - govinfo and api.congress.gov within their documented limits, with keys from the environment only; skip a source if no key is available;
  - arXiv OAI-PMH: 1 request per 3 s, CC-BY and CC0 only.
- **Teacher model pins:**
  - `mlx-community/Qwen3-4B-Instruct-2507-4bit` @ `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`;
  - fallback `mlx-community/Qwen3-8B-4bit` @ `545dc4251c05440727734bcd94334791f6ab0192`, with thinking disabled.
- **Element taxonomies:**
  - `element_topics.json`: one merged tree, root → family → leaf; families `annual_report`, `fund_document`, `contract`, `research_paper`, `technical_document`, `generic`; each has an `other` leaf;
  - `element_functions.json`: flat, multi-label, the 12 tags of spec §4.2.
- **Commits:** conventional, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Corpus leakage:** a dev or held-out PDF re-hosted at a different URL with identical bytes must be excluded by sha256. Pinned by `test_exclusion_by_hash_url_and_title` (L4).
2. **Teacher prompt drift:** changing the prompt or the taxonomy must invalidate cached labels. Pinned by `test_cache_key_includes_prompt_and_taxonomy_hash` (L7).
3. **Long elements and tables:** teacher and student inputs must be truncated at the stated token limits, with tables as description + markdown, and never crash. Pinned by `test_truncation_and_table_inputs` (L7, L10).
4. **Empty or degenerate classes:** a topic leaf with fewer than 20 confident examples must be left untrained and predicted only via its family, without crashing. Pinned by `test_sparse_class_falls_back_to_family` (L10).
5. **Interrupted long runs:** fetch, build, labelling and training must resume without redoing finished work or corrupting files: atomic writes, per-document checkpoints. Pinned by `test_resume_after_partial_write` (L3) and `test_label_run_resumes` (L8).

---

## File structure

| File | Responsibility |
| - | - |
| `src/docrooter/taxonomies/element_topics.json`, `element_functions.json`, `section_types/fund_document.json`, `doc_types.json` (+fact_sheet), `__init__.py` (map) | Taxonomies |
| `src/docrooter/taxonomy/models.py` | Optional `keywords` field on nodes |
| `src/docrooter/elements/models.py` | `TagResult`, `Element.tags` |
| `src/docrooter/corpus/{__init__,store,build}.py` | Corpus store and builder |
| `benchmarks/training_corpus/{fetch.py,sources/*.py,manifest.jsonl,README.md,gold.jsonl}` | Our corpus |
| `src/docrooter/teach/{__init__,base,prompts,mlx,hf,openai,label,shortlist,weak}.py` | Teacher engines and the labelling pipeline |
| `src/docrooter/teach/{enrich,confidence,dataset}.py` | Term mining, similarity-gap filter, training-set builder |
| `src/docrooter/tagger/{__init__,bundle,train,functions}.py` | `ElementTagger` bundle, stage A/B training, function heads, active learning, calibration |
| `src/docrooter/rooter/process.py`, `cli.py`, `ui/*` | Integration |
| `benchmarks/element_tags_bench.py`, `docs/benchmarks-element-tags.md` | Evaluation |

---

### Task L1: Taxonomies (topics tree, functions, fund documents, fact sheet)

**Files:**
- Create:
  - `src/docrooter/taxonomies/section_types/fund_document.json`
  - `src/docrooter/taxonomies/element_functions.json`
  - `src/docrooter/taxonomies/element_topics.json`
  - `scripts/build_element_topics.py`, which generates `element_topics.json` from the section taxonomies
- Modify:
  - `src/docrooter/taxonomies/doc_types.json`: add `financial.fact_sheet` with the definition from spec §4.3;
  - `src/docrooter/taxonomies/__init__.py`: map `financial.prospectus` and `financial.fact_sheet` to `section_types/fund_document`;
  - `src/docrooter/taxonomy/models.py`: optional `keywords: list[str] = []`, omitted from JSON when empty.
- Test: `tests/unit/taxonomy/test_element_taxonomies.py`

**Interfaces:**
- Produces:
  - `builtin_taxonomy("element_topics")` and `builtin_taxonomy("element_functions")`;
  - `ELEMENT_FAMILIES: tuple[str, ...]`;
  - `family_for_doc_type(doc_type: str) -> str`;
  - topic leaf ids are `f"{family}.{section_leaf}"`, plus `f"{family}.other"`.

- [ ] **Step 1: Write the failing tests.**
  - `test_element_topics_is_generated_from_section_taxonomies`: re-running the generator reproduces the file exactly.
  - `test_every_node_has_description_and_function_tags_have_examples_and_not_clause`.
  - `test_fund_document_leaves_per_spec`: the 10 leaves listed in spec §4.1.
  - `test_fact_sheet_doc_type_and_maps`: prospectus and fact_sheet → `fund_document`.
  - `test_keywords_field_optional_and_omitted`: v0.4 taxonomy JSON round-trips byte-identically.
  - `test_family_for_doc_type_covers_all_doc_types`.
- [ ] **Step 2:** Run `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/taxonomy -q`. Expected: FAIL.
- [ ] **Step 3:** Write the JSON files in the v0.2 definition style. The function examples are written for the taxonomy and not taken from any eval set. Write the generator.
- [ ] **Step 4:** Run the tests and the full unit suite. Expected: PASS. Existing doc-type and normalisation tests still pass.
- [ ] **Step 5:** Commit `feat(taxonomies): element topics and functions, fund documents, fact sheets`.

### Task L2: Element tags on the model

**Files:**
- Modify: `src/docrooter/elements/models.py`, `src/docrooter/elements/__init__.py`, `src/docrooter/__init__.py`
- Test: `tests/unit/elements/test_element_tags.py`

**Interfaces:**
- Produces:
  - `TagResult(labels: list[str], probs: dict[str, float], confidence: float, source: str, family_probs: dict[str, float] | None = None)`;
  - `Element.tags: dict[str, TagResult] | None = None`, keyed by facet `"topic"` or `"function"`, omitted from JSON when None.

- [ ] **Step 1:** Write the failing tests:
  - `test_v04_element_json_unchanged_without_tags`;
  - `test_tags_round_trip`;
  - `test_probs_and_confidence_validation`: probabilities lie in [0, 1], confidence in [0, 1].
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement the model changes.
- [ ] **Step 4:** Run the tests and the full unit suite. Expected: PASS.
- [ ] **Step 5:** Commit `feat(elements): topic and function tags`.

### Task L3: Corpus store and builder (`docrooter.corpus`)

**Files:**
- Create: `src/docrooter/corpus/__init__.py`, `corpus/store.py`, `corpus/build.py`
- Modify: `src/docrooter/cli.py` (`corpus build`, `corpus stats`)
- Test: `tests/unit/corpus/test_store.py`, `tests/unit/corpus/test_build.py`

**Interfaces:**
- Consumes: `process` / `process_many` and `RouterConfig`, from v0.4.
- Produces:
  - `ManifestRecord(id, url, source, licence, sha256, bytes, pages, fetched_at, doc_type_hint, family_hint, split)`;
  - `CorpusElement(doc_id, element_id, kind, text, text_sha1, heading_path: list[str], section_title: str | None, page: int | None)`;
  - `Corpus(dir)` with:
    - `.manifest() -> list[ManifestRecord]`
    - `.add_document(record, elements, pages_read)`
    - `.has(doc_id) -> bool`
    - `.iter_elements(split=None, kinds=None) -> Iterator[CorpusElement]`
    - `.write_labels(labeler: str, rows: Iterable[LabelRow])`
    - `.labels(labeler) -> dict[key, LabelRow]`
    - `.stats() -> dict`
  - `LabelRow(doc_id, element_id, text_sha1, facet, probs: dict[str, float], provenance: dict)`;
  - `build_corpus(corpus: Corpus, pdf_dir, *, profile="accurate", max_pages=30, first_pages=10, seed=0, on_progress=None) -> BuildReport`, which reads the first `first_pages` pages plus up to `max_pages - first_pages` sampled pages and skips furniture and `contents`.

- [ ] **Step 1: Write the failing tests.**
  - `test_add_is_idempotent_by_sha256`.
  - `test_resume_after_partial_write` (Review Focus 5): a crash between the doc file and the manifest line leaves no orphan visible to `iter_elements`, and re-adding repairs it.
  - `test_iter_elements_filters_split_and_kind`.
  - `test_labels_keyed_by_doc_element_textsha`.
  - `test_build_reads_first_and_sampled_pages_deterministically`, using the keyword backend, a fake layout engine and generated PDFs.
  - `test_build_skips_furniture_and_contents`.
- [ ] **Step 2:** Run `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/corpus -q`. Expected: FAIL.
- [ ] **Step 3:** Implement. Writes are atomic: write to a temp file, then `os.replace`. The manifest is appended only after the doc file is complete. The page subset is passed through `max_pages` plus a page list. If `load_pdf` has no page-list option, add `pages: Sequence[int] | None` to element mode only; the fast path stays untouched.
- [ ] **Step 4:** Run the tests and the full unit suite. Expected: PASS.
- [ ] **Step 5:** Commit `feat(corpus): corpus store and builder`.

### Task L4: Training corpus sources and fetcher (`benchmarks/training_corpus/`)

**Files:**
- Create:
  - `benchmarks/training_corpus/fetch.py`
  - `benchmarks/training_corpus/sources/{edgar_ars,edgar_ex10,fund_docs,govinfo_gao,crs,fed,arxiv,uscourts,rfc}.py`
  - `benchmarks/training_corpus/README.md`
  - `benchmarks/training_corpus/manifest.jsonl`, written by the fetcher and committed
- Test: `tests/unit/test_bench_training_corpus.py`

**Interfaces:**
- Produces: `python -m benchmarks.training_corpus.fetch --target 1100 [--sources ...] [--dry-run]`.
  - It writes PDFs to `benchmarks/.cache/training_corpus/pdfs/<id>.pdf` and the manifest with sha256, pages and licence.
  - It assigns splits by document, stratified by source with seed 0: about 85% train, 5% calib, 10% gold.
  - It checks the PDF magic bytes and a 50 MB cap, and parses with `-I` isolation for untrusted files.

- [ ] **Step 1: Write the failing tests**, offline, with fixture responses.
  - `test_exclusion_by_hash_url_and_title` (Review Focus 1): dev and held-out manifests are read only for sha256, url and title. Never open the held-out PDFs, only `benchmarks/heldout/manifest.jsonl`.
  - `test_rate_limiter_respects_per_source_limits`.
  - `test_arxiv_licence_filter_keeps_only_cc_by_and_cc0`.
  - `test_edgar_ars_keeps_pdf_main_documents_only`.
  - `test_split_assignment_deterministic_and_stratified`.
  - `test_resume_skips_fetched`.
- [ ] **Step 2:** Run `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_bench_training_corpus.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement each source with the endpoints from research brief §3. Fund documents come from public sponsor URLs listed explicitly in `sources/fund_docs.py`; never redistribute them. Sources needing an API key that is absent are skipped with a log line. Shortfalls are made up from other sources and recorded in the README.
- [ ] **Step 4:** Run the tests. Expected: PASS. Then do a dry run (`--dry-run --target 30`) that lists candidates without downloading, and a real smoke fetch of 10 documents.
- [ ] **Step 5:** Commit `bench: training corpus sources and fetcher`.
- [ ] **Controller run, after review:** the full fetch to ≥ 1,100 documents in the background. Commit `manifest.jsonl` and the README counts as `bench: training corpus manifest`.

### Task L5: Gold set (independent annotation)

**Files:**
- Create:
  - `benchmarks/training_corpus/gold_sample.py`, which picks the 900 elements
  - `benchmarks/training_corpus/annotation_guide.md`
  - `benchmarks/training_corpus/gold.jsonl`
  - `benchmarks/training_corpus/gold_agreement.json`
- Test: `tests/unit/test_bench_gold.py`

**Interfaces:**
- Consumes: the corpus store, after the controller's corpus build of the gold split.
- Produces: `gold.jsonl` rows with `doc_id`, `element_id`, `text_sha1`, `family`, `topic` (leaf id), `functions` (list), `annotator`, `adjudicated`.

- [ ] **Step 1:** Write the failing tests:
  - `test_gold_sample_stratified_per_spec`: 300, 150, 150, 150, 150 by family group, from gold-split documents only;
  - `test_gold_rows_valid_against_taxonomies`;
  - `test_kappa_computation`.

  Then implement the sampler and the agreement computation (Cohen's κ for topic, per-function κ).
- [ ] **Step 2:** Write the annotation guide from the taxonomy definitions only.
- [ ] **Step 3: Controller-run annotation.**
  - Annotator subagents label batches of 50 from the guide alone, with no model outputs visible.
  - 180 elements are double-annotated by a second subagent.
  - A third subagent adjudicates disagreements.
- [ ] **Step 4:** Commit `gold.jsonl` and `gold_agreement.json` BEFORE any model scores a gold document, as `bench: Claude-annotated element gold set (900)`.

### Task L6: Teacher engines (`docrooter.teach`)

**Files:**
- Create: `src/docrooter/teach/__init__.py`, `teach/base.py`, `teach/prompts.py`, `teach/mlx.py`, `teach/hf.py`, `teach/openai.py`
- Modify: `pyproject.toml`: extra `teach-mlx = ["mlx-lm>=0.32; sys_platform=='darwin' and platform_machine=='arm64' and python_version>='3.11'"]`, and mypy overrides for `mlx`, `mlx_lm`. Run `uv lock`.
- Test: `tests/unit/teach/test_teach.py`, `tests/integration/test_teacher_mlx.py` (marker `teach`)

**Interfaces:**
- Produces:
  - `TeachItem(text, context, options: list[tuple[str, str]])`;
  - `TeacherEngine` protocol: `name`, `choose(items, *, instruction, prefix_key) -> list[dict[str, float]]`, `yes_no(items, *, question) -> list[float]`, `close()`;
  - `get_teacher(spec: str) -> TeacherEngine`, accepting `mlx:<repo>@<rev>`, `hf:<repo>@<rev>` and `openai:<model>`. `jev:*` raises `TeacherNotAllowedError`, citing MCA §2.3(b);
  - `PROMPT_VERSION` and `prompt_hash(instruction, options_block) -> str`.

- [ ] **Step 1: Write the failing tests**, with a fake tokenizer and model.
  - `test_choose_softmax_over_option_letters`.
  - `test_yes_no_probability`.
  - `test_truncation_and_table_inputs` (Review Focus 3): 384 teacher tokens.
  - `test_prefix_cache_reused_across_items_with_same_prefix_key`.
  - `test_jev_teacher_refused`.
  - `test_cache_key_includes_prompt_and_taxonomy_hash` (Review Focus 2). It lives here as a pure function `label_key(...)`.
  - `test_openai_missing_letters_get_floor`.
- [ ] **Step 2:** Run `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/teach -q`. Expected: FAIL.
- [ ] **Step 3:** Implement following research brief §2.4: one forward pass per item, the last-position log-softmax at the letter token ids, and a cached prefix trimmed after each item. The hf engine uses the chat template on MPS, CUDA or CPU.
- [ ] **Step 4:** Run the unit tests. Expected: PASS. Then run the integration test under the guard with no other model process: the 4B MLX model labels 20 hand-written items. Record items/s and peak memory.
- [ ] **Step 5:** Commit `feat(teach): teacher engines with multiple-choice log-prob scoring`.

### Task L7: Labelling pipeline (shortlist, teacher, weak labels)

**Files:**
- Create: `src/docrooter/teach/shortlist.py`, `teach/weak.py`, `teach/label.py`
- Modify: `src/docrooter/cli.py` (`teach label`)
- Test: `tests/unit/teach/test_label.py`

**Interfaces:**
- Consumes: L1 taxonomies, L3 Corpus, L6 TeacherEngine, `EmbeddingBackend`.
- Produces:
  - `shortlist(elements, family, backend, k=5) -> list[list[str]]`, which always includes `<family>.other`;
  - `prescreen_functions(elements, backend, k=3) -> list[list[tuple[str, float]]]`;
  - `weak_labels(element, doc) -> dict` (`section_norm` when confidence ≥ 0.5, `edgar_item`);
  - `label_corpus(corpus, teacher, embedder, *, sample: SampleSpec, ceiling_hours=12.0, functions_share=0.5) -> LabelReport`, which writes LabelRows for `topic` and `function` with provenance;
  - `SampleSpec(per_doc=60, target=12000, floor=6000, splits=("train","calib"))`.

- [ ] **Step 1: Write the failing tests**, with fakes.
  - `test_shortlist_includes_other_and_top_k`.
  - `test_family_prefix_lists_all_leaves_and_suffix_restricts_letters` (spec §4.0 ruling 5).
  - `test_function_questions_top3_on_half_sample`.
  - `test_ceiling_stops_and_reports`.
  - `test_label_run_resumes` (Review Focus 5).
  - `test_never_labels_gold_split`.
  - `test_weak_labels_recorded_not_mixed`.
- [ ] **Step 2:** Run `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/teach -q`. Expected: FAIL.
- [ ] **Step 3:** Implement. Stratify the sample by kind and family within the per-document cap.
- [ ] **Step 4:** Run the tests. Expected: PASS. Then smoke-run on 200 elements with the real teacher under the guard.
- [ ] **Step 5:** Commit `feat(teach): corpus labelling with shortlist, teacher and weak labels`.
- [ ] **Controller run, after review:** the full labelling pass, in the background with the 12 h ceiling.

### Task L8: Enrichment, similarity-gap filter and training set

**Files:**
- Create: `src/docrooter/teach/enrich.py`, `teach/confidence.py`, `teach/dataset.py`
- Test: `tests/unit/teach/test_enrich_confidence.py`

**Interfaces:**
- Produces:
  - `mine_terms(corpus, labels, taxonomy, embedder, *, k=10) -> dict[str, list[str]]`, following spec §8: n-grams 1–3 with df ≥ 5, scored `aff = pop^0.5 · dist^0.5 · max(0, cos)`;
  - `enrich_taxonomy(taxonomy, terms) -> Taxonomy` (writes `keywords`);
  - `class_centroids(...)`, `gap_core_classes(scores) -> (core: list[str], gap: float)`;
  - `confident_set(labels, cores, *, keep=0.75, min_per_class=20, prob_override=0.8) -> TrainingSet`, plus a report of untrained classes;
  - `to_dataset(training_set, corpus) -> docrooter.train.Dataset`, with the input text `"{heading path}\n\n[{kind}] {text}"`.

- [ ] **Step 1: Write the failing tests:**
  - `test_term_scores_formula`
  - `test_core_classes_above_largest_gap`
  - `test_keep_rule_core_or_prob_override`
  - `test_top75_per_class_with_floor`
  - `test_sparse_class_reported_untrained`
  - `test_dataset_text_format`
- [ ] **Step 2:** Run `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/teach/test_enrich_confidence.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement, following the spec §4.0 ruling 7 step order.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat(teach): term mining, similarity-gap filter, training set`.

### Task L9: Student (stage A/B, function heads, active learning, calibration, bundle)

**Files:**
- Create: `src/docrooter/tagger/__init__.py`, `tagger/bundle.py`, `tagger/train.py`, `tagger/functions.py`
- Test: `tests/unit/tagger/test_tagger.py`, `tests/integration/test_tagger_train.py` (marker `models`)

**Interfaces:**
- Consumes: `train()`, `TrainedModel`, `calibrate()`, the L8 `Dataset` and the L1 taxonomies.
- Produces:
  - `ElementTagger` with:
    - `topic: TrainedModel`, scored over the whole tree with Beam(3);
    - `functions: dict[tag, LogReg]`;
    - `temperatures: {"topic": float, "function": float}`;
    - `.save(path)` / `.load(path)`;
    - `.tag(elements, family: str | None) -> list[dict[str, TagResult]]`, where topic labels are restricted to `family` when given and `family_probs` is unconstrained;
    - `.close()`.
  - `train_tagger(dataset, function_rows, *, stage: Literal["A","B"], base_model=DEFAULT, max_seq_len=256, max_triplets=40000, time_budget_hours=6.0) -> ElementTagger`.
  - `active_learning_round(tagger, corpus, teacher, *, n=3000) -> list[LabelRow]`.
  - `train_function_heads(embeddings, rows, *, pos=0.7, neg=0.2, implicit_neg_prescreen=0.2)`, where unasked functions are excluded and those with a prescreen score below 0.2 count as implicit negatives.

- [ ] **Step 1: Write the failing tests**, with fake embedders and a small taxonomy.
  - `test_sparse_class_falls_back_to_family` (Review Focus 4).
  - `test_tag_restricts_topic_to_family_and_keeps_family_probs`.
  - `test_function_head_training_excludes_unasked`.
  - `test_bundle_round_trip_with_temperatures`.
  - `test_truncation_and_table_inputs`, for student inputs (Review Focus 3).
  - `test_time_budget_subsamples`.
- [ ] **Step 2:** Run `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/tagger -q`. Expected: FAIL.
- [ ] **Step 3:** Implement stage B with `train(finetune_body=True, max_seq_len=256, gradient_checkpointing=True)`. If the integration smoke on MPS shows instability (NaN loss, OOM or a hang longer than 10 min with no progress), switch to a plain-`transformers` cross-entropy loop with the same bundle format, pad to multiples of 64, and record the switch.
- [ ] **Step 4:** Run the unit tests. Expected: PASS. Then run the integration smoke under the guard: stage A and B on a 500-example synthetic or corpus slice. Record s/step and peak memory.
- [ ] **Step 5:** Commit `feat(tagger): element tagger training, function heads, calibration`.
- [ ] **Controller runs, after review:** stage A on the full confident set, then stage B, then active learning, then retraining B, then calibration on `calib`. Results go to `benchmarks/results/<date>/tagger-train.json`.

### Task L10: Rooter, CLI and playground integration

**Files:**
- Modify: `src/docrooter/rooter/{config,profiles,process}.py`, `src/docrooter/cli.py`, `src/docrooter/ui/server.py`, `ui/static/*`
- Test: `tests/unit/rooter/test_element_tags.py`, `tests/unit/test_cli_tag.py`, `tests/unit/ui/test_server.py`

**Interfaces:**
- Produces:
  - `RouterConfig.element_tags`, which accepts `none`, a bundle path, `embedding:<spec>` (zero-shot over the topic tree and function definitions) or `teacher:<engine spec>`; the default is `none` in all profiles;
  - `process` fills `Element.tags` for paragraph, list_item, table, caption, footnote and formula elements, in batches, with one load per `process_many`;
  - CLI: `docrooter tag FILE --bundle PATH [--profile] [--json]`, `docrooter tag-corpus DIR --bundle PATH`, and `docrooter train-elements DIR --teacher SPEC --out BUNDLE [--stage A|B]`, which runs L7–L9 resumably;
  - playground: `serve --element-tags BUNDLE` adds tag chips with probabilities to the Elements tab.

- [ ] **Step 1: Write the failing tests**, with a fake tagger:
  - `test_element_tags_task_tags_content_kinds_only`;
  - `test_process_many_loads_tagger_once`;
  - `test_tag_cli_json`;
  - `test_train_elements_cli_resumes_stages`;
  - `test_server_tag_chips_textcontent`;
  - `test_teacher_jev_refused_in_config`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the full unit suite. Expected: PASS.
- [ ] **Step 5:** Commit `feat(rooter,cli,ui): element tags`.

### Task L11: Evaluation and write-up

**Files:**
- Create: `benchmarks/element_tags_bench.py`, `docs/benchmarks-element-tags.md`
- Modify: `docs/benchmarks-pdf.md` (the doc-type-with-element-evidence section), `mkdocs.yml`
- Test: `tests/unit/test_bench_element_tags.py`

- [ ] **Step 1:** Write the failing tests, then implement:
  - `test_topic_metrics_primary_and_end_to_end`;
  - `test_function_micro_macro_f1`;
  - `test_ece`;
  - `test_learning_curve_fractions_nested`;
  - `test_element_evidence_mix_fixed_weights` (0.7 / 0.3, never tuned).
- [ ] **Step 2: Controller runs**, under the guard, one model at a time:
  - every system in spec §11 on gold, in both modes;
  - the learning curve;
  - throughput and memory;
  - dev doc type with element evidence.

  Results go to `benchmarks/results/<date>/element-tags*.json`.
- [ ] **Step 3: Write the docs.** Give every number from the JSON, the program targets (met or missed), the disclosures (Claude-annotated gold, in-sample dev), and the learning curve framed as "performance depends on corpus size".
- [ ] **Step 4:** Commit `bench: element tagger evaluation`.

### Task L12: Docs, packaging, release notes

**Files:**
- Create: `docs/site/guides/train-your-tagger.md`, `docs/site/concepts/element-tags.md`, `examples/train_element_tagger.py` (a tiny-corpus example with fake or keyword fallbacks, tested)
- Modify: `README.md`, `CHANGELOG.md`, `mkdocs.yml`, `tests/unit/test_examples.py`

- [ ] **Step 1:** Write the pages. Explain that Jev cannot be a teacher and why, and that users bring their own teacher under their provider's terms.
- [ ] **Step 2:** Run the full unit suite, ruff, the format check, mypy and `mkdocs build --strict`. Expected: clean.
- [ ] **Step 3:** Build the wheel, install it in a fresh venv, and run the `docrooter tag --help` and `corpus --help` smoke.
- [ ] **Step 4:** Commit `docs: element tags and training your own tagger`.
