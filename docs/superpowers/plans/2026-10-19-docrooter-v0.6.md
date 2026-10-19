# docrooter v0.6 Implementation Plan: selection and the completeness dial

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `select(analysis, need, completeness=c)` returns the elements a downstream LLM should read. A conformal-risk-control threshold behind it guarantees mean recall ≥ c on calibration-like documents (spec §0, §6). The selection is surfaced in the CLI and in a playground dial.

**Architecture:**
- **Fields** are named needs mapped to topic and function tags.
- **Relevance:** a per-element logistic combiner of tagger, embedding and structure features, fitted on calib-fit.
- **The dial:** a CRC threshold grid fitted on calib-crc, using (document, field) units with full teacher labels.
- **Selection** applies context expansion and an optional token budget.
- **Evaluation** uses gold-split documents (full teacher labels + the 900 gold elements) and an evaluation-only public evidence QA set.

**Tech Stack:** Python ≥ 3.10, numpy, pydantic. It reuses v0.5 `ElementTagger`, `docrooter.corpus` (`calib_half`), `docrooter.teach` (teacher engines) and the v0.4 `chunk()`.

**Spec:** `docs/superpowers/specs/2026-10-19-docrooter-v0.6-selection-dial-design.md`. Its §0 rulings are binding.

## Global Constraints

- **Earlier constraints still hold.** All v0.1–v0.5 constraints apply:
  - the core is torch-free;
  - unit tests are torch, mlx and onnxruntime-free, and no heavy imports happen at test collection;
  - no recursion;
  - secrets stay in `.env`;
  - use `PYTHONPATH=src`;
  - never run a plain `uv sync`;
  - the fast path stays byte-identical to v0.3.
- **Jev is never a teacher** (TypeSafe MCA §2.3(b)). Full labelling uses the local teacher.
- **Data separation (§0.1):**
  - calib-fit (`calib_half(doc_id) == "fit"`) fits the combiner;
  - calib-crc is used only for CRC;
  - gold-split documents are used only for evaluation;
  - the held-out corpus is never opened.
- **Evaluation-only datasets** (FinanceBench CC BY-NC 4.0, and others marked NC) are never redistributed, trained on or calibrated on. They are downloaded into `benchmarks/.cache.nosync/`.
- **One model process at a time** within the 8 GB budget, through `benchmarks._guard` (the guard counts `corpus build`, `teach label` and benchmark scripts).
- **`completeness = 1.0`** always returns every content element, with `guarantee=True`.
- **Commits** are conventional and end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A need with zero relevant elements in a document**, e.g. a fund document with no tax section. `select()` returns no field elements, only expansion context, and the expected recall is reported as undefined, not 0 or 1. Pinned by `test_no_relevant_elements_needs_defined_readout` (S4).
2. **Huge documents** (5,000+ elements): `select()` stays linear and fast. Pinned by `test_select_linear_time` (S4).
3. **A calibration set too small for the target**: a capped dial returns the whole document with `capped=True`. Pinned by `test_unreachable_target_caps_to_whole_document` (S3).
4. **Elements with no tags** (tagger absent or failed): fall back to similarity and structure, and set `guarantee=False`. Pinned by `test_no_tagger_fallback_no_guarantee` (S4).
5. **Budget smaller than one element**: return the single most relevant element and say so. Pinned by `test_tiny_budget` (S4).

---

### Task S1: Fields

**Files:**
- Create: `src/docrooter/select/__init__.py`, `select/fields.py`, `src/docrooter/fields/{annual_report,fund_document,contract}.json`
- Test: `tests/unit/select/test_fields.py`

**Interfaces:**
- Produces:
  - `Field(id, label, description, topics: list[str], functions: list[str] = [], query: str | None = None)`;
  - `builtin_fields(family: str) -> list[Field]`;
  - `load_fields(path) -> list[Field]`;
  - `validate_fields(fields, topics_taxonomy, functions_taxonomy)`.

- [ ] **Step 1:** Write the failing tests:
  - `test_builtin_field_sets_valid_against_taxonomies`: every topic is a leaf of the family, and functions exist;
  - `test_each_field_has_description_and_query`;
  - `test_custom_fields_json_roundtrip`;
  - `test_unknown_topic_rejected`.
- [ ] **Step 2:** Run `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/select -q`. Expected: FAIL.
- [ ] **Step 3:** Write the field sets. Use 8–15 fields per family, with descriptions in the taxonomy style, plus cross-cutting fields: fees and expenses with `financial_data`, risk with `risk_disclosure`, obligations with `obligation`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat(select): fields`.

### Task S2: Full teacher labelling of calib and gold documents

**Files:**
- Modify: `src/docrooter/teach/label.py` (a `full` sample mode), `src/docrooter/cli.py` (`teach label --full --splits calib gold`)
- Test: `tests/unit/teach/test_label_full.py`

**Interfaces:**
- Produces:
  - `SampleSpec(mode="full", splits=(...))`, which labels the topic of EVERY content element of the given splits' documents, with no function questions;
  - labels go to a separate labeler file, `teacher-full`, so they never feed v0.5 training.

- [ ] **Step 1:** Write the failing tests:
  - `test_full_mode_labels_every_content_element`;
  - `test_full_labels_in_separate_file_and_never_in_training` (the training dataset builder ignores `teacher-full`);
  - `test_gold_split_allowed_only_in_full_mode`;
  - `test_resume`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement. The ceiling is per pass, measured.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat(teach): full topic labelling for calibration and evaluation splits`.
- [ ] **Controller run, after review:** `docrooter teach label --full --splits calib gold`, in the background, after the v0.5 pipeline.

### Task S3: Relevance combiner and CRC calibration

**Files:**
- Create: `src/docrooter/select/relevance.py`, `select/crc.py`
- Test: `tests/unit/select/test_relevance.py`, `tests/unit/select/test_crc.py`

**Interfaces:**
- Produces:
  - `features(elements, tags, sims, sections, need) -> np.ndarray` (spec §5 features);
  - `Combiner.fit(X, y)` / `.predict_proba(X)`, a logistic regression via numpy (no sklearn), plus `save`/`load` to `selector.json`;
  - `crc_threshold(losses_by_lambda: dict[float, np.ndarray], alpha: float) -> tuple[float | None, int]`, implementing spec §6 exactly: max λ with (n/(n+1))·mean + 1/(n+1) ≤ α, or None if unreachable;
  - `calibrate_dial(units, grid=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99)) -> DialTable`, per family, with n stored;
  - `DialTable.threshold(c) -> (λ | None, capped: bool)`, monotone and interpolated.

- [ ] **Step 1:** Write the failing tests:
  - `test_crc_formula_on_synthetic_losses`;
  - `test_crc_monotone_in_target`;
  - `test_crc_guarantee_on_simulation`: draw 2,000 exchangeable synthetic documents; the realised mean recall on held-out simulations is ≥ c − 0.01;
  - `test_unreachable_target_caps_to_whole_document` (Review Focus 3);
  - `test_combiner_fit_predict_and_roundtrip`;
  - `test_combiner_fits_only_calib_fit` (uses `calib_half`);
  - `test_dial_uses_only_calib_crc`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat(select): relevance combiner and conformal dial`.

### Task S4: `select()`

**Files:**
- Create: `src/docrooter/select/selection.py`
- Modify: `src/docrooter/__init__.py`
- Test: `tests/unit/select/test_select.py`

**Interfaces:**
- Produces:
  - `Selection`, as spec §7, plus `capped: bool`, `doc_type_mismatch: bool` and `relevant_expected: float | None`;
  - `select(analysis, need, *, completeness=0.9, budget=None, selector=None, tagger=None) -> Selection`;
  - `Selector` (the loaded combiner plus dial tables), with `Selector.load(path)`.

- [ ] **Step 1:** Write the failing tests, with fake taggers and embeddings:
  - `test_completeness_one_returns_everything`;
  - `test_threshold_from_dial_and_guarantee_flag`;
  - `test_expansion_adds_previous_same_section_heading_first_paragraph_and_whole_tables`;
  - `test_budget_mode_no_guarantee`;
  - `test_query_need_never_guaranteed`;
  - `test_doc_type_mismatch_retags_and_drops_guarantee`;
  - `test_no_relevant_elements_needs_defined_readout` (Review Focus 1);
  - `test_select_linear_time` (Review Focus 2);
  - `test_no_tagger_fallback_no_guarantee` (Review Focus 4);
  - `test_tiny_budget` (Review Focus 5);
  - `test_chunks_from_selected_elements`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat(select): select() with expansion, budget and guarantee flags`.

### Task S5: Fitting the selector (pipeline) and the CLI

**Files:**
- Create: `src/docrooter/select/fit.py`
- Modify: `src/docrooter/cli.py` (`docrooter fit-selector DIR --bundle B --out selector.json`, and `docrooter select FILE --need ... [--completeness] [--budget] [--tagger] [--selector] [--json]`)
- Test: `tests/unit/select/test_fit.py`, `tests/unit/test_cli_select.py`

**Interfaces:**
- Produces: `fit_selector(corpus, tagger, embedder, *, fields_by_family) -> Selector`, which fits the combiner on calib-fit and the dial on calib-crc, using `teacher-full` labels as relevance.

- [ ] **Step 1:** Write the failing tests:
  - `test_fit_uses_teacher_full_labels_and_partitions`;
  - `test_cli_select_json`;
  - `test_cli_fit_selector_writes_file`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat(select): selector fitting and CLI`.

### Task S6: Playground dial

**Files:**
- Modify: `src/docrooter/ui/server.py`, `ui/static/*`
- Test: `tests/unit/ui/test_server.py`

- [ ] **Step 1:** Write the failing tests:
  - `test_select_endpoint_returns_selection`;
  - `test_dial_presets`;
  - `test_select_payload_bounded`;
  - `test_textcontent_only`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement:
  - a field picker and query box;
  - a completeness slider with presets;
  - the readout ("≈ c of relevant content kept · tokens (share) · guarantee or estimate or capped");
  - selected elements highlighted in the Elements tab.
- [ ] **Step 4:** Run the tests. Expected: PASS. Then do a manual check on port 18951 with the keyword backend and a fake or real selector, and take a screenshot to the scratchpad.
- [ ] **Step 5:** Commit `feat(ui): completeness dial`.

### Task S7: Selection benchmark and demo extraction harness

**Files:**
- Create: `benchmarks/selection_bench.py`, `benchmarks/evidence_qa.py` (FinanceBench or TAT-DQA fetch and alignment, evaluation only), `benchmarks/extraction/run.py` (optional demo), `docs/benchmarks-selection.md` (skeleton with "results pending")
- Test: `tests/unit/test_bench_selection.py`

- [ ] **Step 1:** Write the failing tests:
  - `test_recall_and_token_share`;
  - `test_tokens_to_95_recall`;
  - `test_baselines_same_budget`: whole document, lead-k, BM25 over elements, similarity-only and structure-only;
  - `test_evidence_alignment_char_overlap`;
  - `test_nc_dataset_never_written_outside_cache`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement. Gold evaluation reports teacher-relevance recall as the main number, the gold-annotated subset as the second number, and the measured teacher–gold disagreement. The extraction harness runs only if an endpoint or a local MLX model is configured. It never uses Jev.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `bench: selection benchmark and evaluation-only evidence QA`.
- [ ] **Controller runs:** fit the selector, then the gold evaluation, then evidence QA, then the docs numbers.

### Task S8: Docs and release notes

**Files:**
- Create: `docs/site/guides/select-for-extraction.md`, `docs/site/concepts/completeness-dial.md`, `examples/select_for_extraction.py` (tested with fakes)
- Modify: `README.md`, `CHANGELOG.md`, `mkdocs.yml`

- [ ] **Step 1:** Write the pages. Explain the guarantee, what it is calibrated against (teacher relevance), the capped dial, queries being estimate-only, and the doc-type mismatch.
- [ ] **Step 2:** Run the full bare `pytest`, ruff, the format check, mypy and `mkdocs build --strict`. Expected: clean.
- [ ] **Step 3:** Commit `docs: selection and the completeness dial`.
