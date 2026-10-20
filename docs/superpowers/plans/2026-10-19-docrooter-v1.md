# docrooter v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

- **Goal:** Token-efficient, sections-first classification for hosted LLMs. It ships with a context ladder, a section catalogue and doc-type rules, exact token accounting, a 113-document benchmark, and a real installable 1.0.0 package published to GitHub.
- **Architecture:**
  - A `level=` preset layer over the existing `SectionView` and `DocumentView`.
  - A shared section catalogue, with doc-type `sections` rule blocks.
  - A new top-level `classify()` that runs the `sections_first` or `doc_first` flow on the existing `Classifier`, normalisation and strategies.
  - The OpenAI-compatible backend gains letter markers and usage counters. A local `mlx_lm.server` stands in for hosted models.
  - A benchmark harness over the v0.5 gold split, with new section gold.
- **Tech stack:** Python ≥ 3.10, pydantic, typer, httpx, numpy. Optional extras as before; `mlx_lm.server` is used for the local stand-in.
- **Spec:** `docs/superpowers/specs/2026-10-19-docrooter-v1-token-efficient-classification-design.md`. Its §6 rulings are binding.

## Global Constraints

- **Core and tests.** The core stays torch-free, and unit tests are torch-free, model-free and network-free. Bare `pytest` must stay clean.
- **Running tools.** Use `PYTHONPATH=src`, `.venv/bin/pytest`, `.venv/bin/ruff` and `.venv/bin/mypy`. Never run a plain `uv sync`.
- **Models.** One model process at a time, within an 8 GB budget. Implementers never load real models; controllers run all model jobs.
- **Secrets.** Never read or print `.env`. No keys in logs, results or commits.
- **What never gets committed:** corpus PDFs, the user's `tests/_10-K-2025-As-Filed.pdf`, and annotation or document text from the corpus.
- **Jev** is never a teacher. Its log rows are `trainable=false`.
- **Compatibility.** Existing public functions keep their defaults (`level=None` is the old behaviour). Passing both `level=` and `view=` raises `ValueError`.
- **Token numbers.** Numbers called "exact" come only from API `usage`. Everything else is labelled `estimated`.
- **Commits.** End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A document with no headings** (a flat PDF, or Markdown with no `#`). `sections_first` has nothing to tag. It must fall back to `doc_first` at the same level, flagged `no_sections`. The test is `test_classify_no_sections_falls_back` (V4).
2. **Overflow:** a level budget smaller than one field, or a document far beyond the 16k `full` cap. Views must cut, never overflow, and record `truncated`. The test is `test_full_level_caps_and_flags_truncation` (V2).
3. **An endpoint that returns no `usage` or no logprobs** (some local servers). Tokens then fall back to an `estimated` flag, and the answer is parsed from the text. The test is `test_usage_missing_marks_estimated` (V3).
4. **Rules with zero candidates or conflicting excludes.** These must take the fallback path with `rules_unmet` and never raise. The test is `test_rules_unmet_falls_back_to_all_types` (V4).
5. **More than 26 options in one call.** This must raise a clear error naming the hierarchical strategy as the fix, never silently truncate. The test is `test_more_than_26_options_raises` (V3).

---

### Task V1: Section catalogue and doc-type rule blocks

**Files:**
- Create: `src/docrooter/taxonomies/section_catalogue.json`, `src/docrooter/rules.py`
- Modify: `src/docrooter/taxonomies/doc_types.json` (a `sections` block per leaf), `src/docrooter/taxonomy/models.py` (if the node schema needs a `sections` field)
- Test: `tests/unit/test_section_catalogue.py`, `tests/unit/test_rules.py`

**Interfaces (produces):**
- `builtin_taxonomy("section_catalogue")`: groups with leaves, every group with an `other` leaf.
- `DocTypeRules` (pydantic) with `requires: list[list[str]]`, `typical: list[str]` and `excludes: list[str]`, loaded from each doc-type leaf's `sections` key.
- `doc_type_rules(taxonomy) -> dict[str, DocTypeRules]`.
- `match_rules(rules, tags: Mapping[str, float], *, min_conf=0.5) -> RuleMatch`, where `RuleMatch` has `candidates`, `near_misses` and `decided`, plus `support` (doc type → the section ids and confidences that met it).
  - `tags` maps a catalogue leaf id to the max confidence seen in the document.
  - A group id in a rule matches any of its leaves.

**Content rules:**
- The catalogue merges the leaves of `section_types/*.json`: their ids are remapped to `<group>.<leaf>`, and the ids, definitions and examples are kept.
- It adds the financial group of spec G2 (balance sheet, income statement, cash-flow statement, changes in equity, schedule of investments, auditor's report, notes, account statement) and the investor letter.
- Every one of the 19 doc-type leaves gets a rule block, written from its definition.

**Steps:**
- [ ] **Step 1: Failing tests.**
  - `test_catalogue_loads_and_every_group_has_other`.
  - `test_every_doc_type_leaf_has_rules_with_known_ids`.
  - `test_group_id_matches_any_leaf`.
  - `test_candidates_near_misses_and_excludes`: with excludes, a candidate is removed unless that would leave none.
  - `test_single_candidate_decided_needs_conf_0_7`.
  - `test_quarterly_report_example`: financial statements plus investor letter makes a candidate; without them, it does not.
- [ ] **Step 2:** Run `.venv/bin/pytest tests/unit/test_section_catalogue.py tests/unit/test_rules.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement. Write the definitions to the existing taxonomy standard: one sentence of scope plus a "not …" boundary where a neighbour is confusable.
- [ ] **Step 4:** Run again. Expected: PASS. Then run the full unit suite, ruff and mypy.
- [ ] **Step 5:** Commit `feat(taxonomy): section catalogue and doc-type section rules`.

### Task V2: The context ladder

**Files:**
- Create: `src/docrooter/ladder.py`
- Modify: `src/docrooter/classify/classifier.py`, `classify/documents.py`, `classify/normalize.py` (accept `level=`)
- Test: `tests/unit/test_ladder.py`

**Interfaces (produces):**
- `LEVELS = ("headings", "terms", "lead", "full")`.
- `section_view(level) -> SectionView` and `document_view(level, *, full_cap=16000) -> DocumentView`.
- `level=` keyword on `classify_document`, `normalize_sections` and `Classifier.classify_documents`. Passing both `level=` and `view=` raises `ValueError`.
- `next_level(level) -> str | None`.

**Presets** (the exact budgets are the builder's call within spec §2 G1; record them in the module docstring):
- headings: path + title, about 32 tokens;
- terms: + key phrases, about 64;
- lead: + 3 lead sentences, about 192;
- full: text at 2,048 per section; the document view's `full` is the whole text capped at `full_cap`.

**Steps:**
- [ ] **Step 1: Failing tests.**
  - `test_levels_are_monotone_in_tokens`: on a fixture document, each rung renders ≥ the tokens of the previous one.
  - `test_level_and_view_together_raise`.
  - `test_level_none_keeps_old_defaults`.
  - `test_full_level_caps_and_flags_truncation` (Review Focus 2).
  - `test_next_level_order`.
- [ ] **Step 2:** Run. Expected: FAIL. **Step 3:** Implement. **Step 4:** Run. Expected: PASS, with the full suite, ruff and mypy clean.
- [ ] **Step 5:** Commit `feat(views): the context ladder`.

### Task V3: LLM backend: letter markers, token usage, results log

**Files:**
- Modify: `src/docrooter/backends/openai.py`, `src/docrooter/backends/base.py` (a usage protocol)
- Create: `src/docrooter/results_log.py`
- Test: `tests/unit/backends/test_openai_v1.py`, `tests/unit/test_results_log.py`

**Interfaces (produces):**
- **Usage counters.**
  - `TokenUsage(prompt: int, completion: int, calls: int, exact: bool)`.
  - Backends may expose `usage() -> TokenUsage`, which is cumulative. `OpenAIBackend` fills it from the response `usage`.
  - `usage_of(backend) -> TokenUsage | None` works for any backend.
- **Letter markers.** `OpenAIBackend` uses letters A–Z as markers. The logprob distribution is over the letter tokens: pool whitespace variants, then parse the first letter in the text. More than 26 options raises `BackendError`.
- **Results log.**
  - `ResultsLog(path)` has `.write(record: dict)`. It appends JSONL, refuses any value that looks like an API key (redact), and sets `trainable=False` when the model or endpoint names Jev or typesafe.
  - Each row carries: level, view hash, options, label, probs, model, endpoint host, prompt hash, tokens, trainable and timestamp.
- **Docs.** `docs/site/guides/llm-backends.md` gets the local stand-in recipe: `mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --port 8080`, and `DOCROOTER_LLM_BASE_URL=http://127.0.0.1:8080/v1`.

**Steps:**
- [ ] **Step 1: Failing tests** (httpx MockTransport only).
  - `test_letter_markers_and_logprob_pooling`.
  - `test_usage_accumulates_from_response`.
  - `test_usage_missing_marks_estimated` (Review Focus 3).
  - `test_more_than_26_options_raises` (Review Focus 5).
  - `test_log_marks_jev_untrainable_and_redacts`.
- [ ] **Step 2:** Run. Expected: FAIL. **Step 3:** Implement, and update the existing openai tests for letters. **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit `feat(llm): letter markers, exact token usage, results log`.

### Task V4: `classify()`: sections-first and doc-first flows, CLI, playground

**Files:**
- Create: `src/docrooter/flow.py`
- Modify: `src/docrooter/__init__.py`, `src/docrooter/cli.py`, `src/docrooter/ui/server.py`, `ui/static/*`
- Test: `tests/unit/test_flow.py`, `tests/unit/test_cli_classify.py`, `tests/unit/ui/test_server.py`

**Interfaces:**
- **Consumes:** V1 (`match_rules`, the catalogue), V2 (levels), V3 (`usage_of`, `ResultsLog`).
- **Produces:**
  - `classify(source_or_doc, *, flow="sections_first", level="terms", section_level=None, doc_level=None, backend=DEFAULT_BACKEND, log=None, max_sections=40, rule_min_conf=0.5, decide_conf=0.7, cascade=False, threshold=0.5) -> Classification`.
  - `Classification` (pydantic) has these fields:
    - `doc_type`, `doc_type_confidence`, `decided_by` (`"rules"`, `"model"` or `"fallback"`);
    - `candidates` and `flags` (`no_sections`, `rules_unmet`, `near_miss`, `truncated`);
    - `sections: list[SectionTag]`, where each tag has `id`, `title`, `level`, `label`, `confidence`, `level_used` and `tokens`;
    - `tokens: TokenUsage`, plus per-stage tokens (`sections`, `document`);
    - `levels` (what was used) and `backend`.
  - **sections_first:**
    - tag up to `max_sections` level-1 and level-2 sections, level 1 first in reading order, against the catalogue with `Greedy` (group, then leaf);
    - match the rules;
    - then decide by rules, or classify among the candidates (with near misses if needed), or fall back.
  - **doc_first:** today's flow (doc type, then section taxonomy), made level-aware and token-counted.
  - **cascade=True:** items with confidence < `threshold` are re-asked one rung up, until `full`.
  - **CLI:** `docrooter classify FILE [--flow] [--level] [--section-level] [--doc-level] [--backend] [--log PATH] [--json]`.
  - **Playground:** a flow and level picker, plus a tokens readout. Section tags are shown in the Section labels tab, and `decided_by` and candidates in the Document type tab.

**Steps:**
- [ ] **Step 1: Failing tests** (a fake backend with scripted scores and fake usage).
  - `test_sections_first_rules_decide_single_candidate`.
  - `test_sections_first_narrows_options_to_candidates`: the doc-level call's options are only the candidates.
  - `test_rules_unmet_falls_back_to_all_types` (Review Focus 4).
  - `test_classify_no_sections_falls_back` (Review Focus 1).
  - `test_tokens_sum_sections_and_document`.
  - `test_cascade_escalates_only_low_confidence`.
  - `test_max_sections_order`.
  - `test_doc_first_level_aware`.
  - `test_cli_classify_json`.
  - `test_playground_flow_level_payload`.
- [ ] **Step 2:** Run. Expected: FAIL. **Step 3:** Implement. **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit `feat: classify() with sections-first and doc-first flows`.

### Task V5: Section gold for the 113 gold-split documents

**Files:**
- Create: `benchmarks/classification/section_sample.py`, `benchmarks/classification/section_guide.md`, `benchmarks/classification/section_gold.py` (merge, κ, adjudication)
- Committed output, created by the controller run: `benchmarks/classification/section_gold.jsonl` (ids and labels only, no text) and `section_gold_agreement.json`
- Test: `tests/unit/test_section_gold.py`

**Interfaces (produces):**
- `sample_sections(doc, *, max_per_doc=12) -> list[Section]`: level 1 and level 2, spread across the document, deterministic.
- Batch files go to `benchmarks/.cache.nosync/classification/annotation/` and are never committed.
- `section_gold.jsonl` rows: `{doc_id, section_id, title_sha1, level, label, annotator, adjudicated}`.
- Documents are read with the same reading the benchmark uses: profile `fast`, which has headings. Record that profile in the gold.

**Steps:**
- [ ] **Step 1: Failing tests.**
  - `test_sample_deterministic_and_capped`.
  - `test_kappa_and_merge`.
  - `test_gold_rows_hold_no_text`.
- [ ] **Step 2:** Run, then implement, then run. Expected: PASS. **Step 3:** Commit `bench: section gold tooling`.
- [ ] **Controller run:**
  1. Sample.
  2. Have annotators (Claude, blind) label every item, double-labelling 20%.
  3. Adjudicate disagreements.
  4. Commit the gold as `bench: section gold for the gold split`, BEFORE any model scores these documents' sections.

### Task V6: Classification benchmark

**Files:**
- Create: `benchmarks/classification_bench.py`, `docs/benchmarks-classification.md` (a skeleton with "results pending")
- Modify: `benchmarks/_guard.py` (add `classification_bench`)
- Test: `tests/unit/test_bench_classification.py`

**Interfaces:**
- **Consumes:** V4 `classify()`, the V5 gold, and the manifest doc types.
- **Produces:** `python -m benchmarks.classification_bench run --backend SPEC --flow F --level L [--section-level] [--doc-level] [--docs gold|heldout] --out DIR`, and `report` to write the markdown.
- **Per run:** doc-type accuracy and macro-F1; section macro-F1 and micro-F1 at leaf and at group; the share decided by rules; prompt tokens per document (exact or estimated); token share against `full`; truncated share; wall time.
- **Report:** the accuracy-vs-tokens table and the "cheapest within 2 points of `full`" pick per backend (spec §6.2).
- **Results** hold no text, only ids and numbers.

**Steps:**
- [ ] **Step 1: Failing tests.**
  - `test_doc_type_metrics`.
  - `test_section_f1_leaf_and_group`.
  - `test_token_share_vs_full`.
  - `test_cheapest_within_two_points`.
  - `test_results_hold_no_text`.
- [ ] **Step 2:** Run, then implement, then run. Expected: PASS. **Step 3:** Commit `bench: classification benchmark`.
- [ ] **Controller runs:** the grid of spec §6.2 on keyword, then UltraBERT, then the local Qwen3-4B via `mlx_lm.server`, one model process at a time. Then the cross-grid. Then held-out doc type once with the best configuration. Then write the docs numbers. Commit `bench: classification results`.

### Task V7: Packaging and docs for 1.0.0

**Files:**
- Modify: `pyproject.toml` (version 1.0.0, project URLs for the GitHub owner the user confirms), `README.md`, `CHANGELOG.md`, `mkdocs.yml`, `docs/site/**`, `.github/workflows/*`
- Create: `docs/site/concepts/context-ladder.md`, `docs/site/guides/sections-first.md`
- Test: `tests/unit/test_examples.py` (a new example `examples/classify_sections_first.py`, run with fakes)

**Steps:**
- [ ] **Step 1: README.** Rewrite it around the v1 pitch (minimum context for hosted LLMs), the ladder, sections-first, the benchmark table (pending until V6), install, and quickstart. Mark selection, the learned tagger and corpus tooling "experimental".
- [ ] **Step 2: CHANGELOG** 1.0.0.
- [ ] **Step 3: CI.** Check that `ci.yml` runs ruff, mypy and pytest on 3.10–3.12. Make `publish.yml` build the wheel and sdist and attach them to a GitHub release on tag. PyPI upload stays manual and documented.
- [ ] **Step 4: Packaging.** Run `uv build` (or `python -m build`). Install the wheel into a fresh venv in the scratchpad, then run `docrooter --help` and `docrooter classify` on a fixture Markdown file with the keyword backend. Run `mkdocs build --strict` with the site dir in the scratchpad.
- [ ] **Step 5:** Commit `release: 1.0.0`.

### Controller: final review and publish

- [ ] **Final review:** three parallel passes on the whole branch (correctness of the flows and tokens; API and packaging; benchmarks and docs as evidence). Then one fix wave and a scoped re-review.
- [ ] **History audit before any push:**
  - no `.env`, keys, corpus PDFs, the private 10-K, or annotation text anywhere in git history;
  - the absolute home path in 49d577b fixed or accepted by the user.
- [ ] **Publish:**
  - merge `feat/v1` into `main`, tag `v1.0.0`;
  - after the user confirms the owner and visibility: `gh repo create`, push `main` and the tag, and create a GitHub release with the wheel and sdist;
  - verify `pip install git+https://github.com/<owner>/docrooter@v1.0.0` in a fresh venv.
