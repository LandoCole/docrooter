# docrooter v1: token-efficient, sections-first classification

- **Date:** 2026-10-19
- **Status:** Approved in chat by the user on 2026-10-19 ("write up the spec/plan … continue with development and testing … publish a working v1"). It replaces the v0.7 phase of the program document.
- **Builds on:** v0.1–v0.3 (TOC, document type, section labels, views, backends), v0.4 (elements, rooter) and v0.6 (selection).
- **Re-anchors:** v0.5. The learned student tagger becomes an optional, experimental extra (§3).

## 0. Why this spec exists (the user's correction)

v0.5 drifted. It used the corpus as *training data* for a local student model, in the style of TELEClass. That is not the goal. In the user's words:

- **Hosted LLMs are the deployment target.** "In deployment … we can send it to … an intelligent large language model. But we're using the minimum amount of context to get exactly what we want."
- **Local models are for testing.** Running UltraBERT locally is "just testing it locally on a small model, just validating results", because the user does not want to pay provider tokens while testing.
- **The corpus is for validation.** It is not for building the taxonomy, and not mainly for training.
- **What we optimise:** "maximizing accuracy and … recall of the system and minimizing token cost". The benchmark compares the whole document against headings only, headings plus key terms, and so on.
- **Sections first.** Tag the sections ("balance sheet, income statement … notes to financials … investor letter"). Then use the section tags to determine the document type, or to narrow the options sent to the model: "only send the tax document types".
- **Saved LLM results** may someday train a small model. That is a by-product, not the goal.

## 1. Intent

docrooter turns a document into the smallest context that still lets a capable LLM classify it. It works at two levels:
- **sections:** what each section is;
- **document:** what the document is, decided from its sections where possible.

It returns the classifications, the evidence, and the exact tokens spent.

## 2. Goals

### G1. The context ladder

There is one named ladder for sections and one for documents. Each rung adds context, and therefore tokens.

| Level | Section view | Document view |
|---|---|---|
| `headings` | heading path + title | title + headings (TOC) |
| `terms` | + key phrases | + pooled key phrases |
| `lead` | + first sentences (lead) | + opening body text (lead) |
| `full` | full section text, budgeted | whole document text, budgeted to the model's context |

- **Built on existing views.** Each level is a preset over the existing `SectionView` and `DocumentView` fields and budgets. The presets are exact and documented.
- **API.** The public API takes `level=` wherever a view is used today.
- **Default level.** The default is the cheapest rung that the benchmark shows within 2 points of `full` accuracy, per backend. This is a measured default, recorded with its evidence. Until the benchmark exists, the default is `terms`.
- **Cascade.** `level="auto"` starts at `headings` and escalates one rung at a time, only for the items whose confidence is below the threshold. It stops at `max_level`.

### G2. The section catalogue

A section can be tagged without knowing the document type. The catalogue is one shared section taxonomy, `taxonomies/section_catalogue.json`, grouped by domain.

- **Origin.** It is built from the existing `section_types/*.json` leaves, merged and de-duplicated, plus a finer financial group:
  - balance sheet;
  - income statement;
  - cash-flow statement;
  - statement of changes in equity;
  - schedule of investments;
  - auditor's report;
  - notes to financial statements;
  - account statement.
- **Investor letter.** It joins the narrative group: shareholder or investor letter.
- **Group structure.** Every group has an `other` leaf. Every leaf has a definition and examples, written to the same standard as the existing taxonomies.
- **Hierarchical classification.** Sections are classified group first, then leaf, using the existing `Greedy` strategy, so each call carries few options.
- **Section taxonomies stay usable.** The existing per-doc-type section taxonomies remain usable for doc-type-first runs, which keeps backward compatibility.

### G3. Document types defined by their sections

`doc_types.json` leaves gain a `sections` rule block:
- `requires`: a list of groups of catalogue ids. The document must contain at least one section from each group.
- `typical`: catalogue ids that raise the score but are not required.
- `excludes`: optional.

Example: `financial.quarterly_report` requires {financial statements} and {investor letter or MD&A}. The builder writes rules for every doc-type leaf, from the definitions, never from the corpus.

**The `sections_first` flow.**
1. Tag the sections at the section level chosen (G1, G2).
2. Score every doc type against the tags. A required group is met when a section is tagged with one of its ids at confidence ≥ `rule_min_conf` (default 0.5).
3. **Candidates** are the types with all required groups met.
4. **Near misses** are types missing exactly one required group. They are kept only when no type is fully met, and flagged.
5. **One candidate** decides the type with no document-level call, and the result records `decided_by="rules"`.
6. **Several candidates** go to a document-level classification among those candidates only, at the document level chosen.
7. **No candidates and no near misses** fall back to classifying among all types, flagged `rules_unmet`.

**The `doc_first` flow** (today's flow) stays: classify the type, then label sections with its taxonomy. Both flows are benchmarked.

### G4. The LLM path: deployment and local stand-in

**One backend for both.**
- Hosted and local LLMs go through the existing OpenAI-compatible backend.
- The local stand-in is `mlx_lm.server` serving the pinned Qwen3-4B-Instruct-2507 4-bit. It is free and on the same code path as a hosted model.
- Jev stays an optional backend for comparison runs.

**Exact token accounting.**
- Every LLM call records `prompt_tokens` and `completion_tokens` from the API's `usage`.
- Results carry `tokens.prompt`, `tokens.completion` and `tokens.calls` per document and per section.
- Non-LLM backends report the estimated tokens the same view would cost an LLM, labelled as an estimate.

**Results log.**
- An optional JSONL log is written by `classify(..., log=path)` and `--log`. Each line holds one decision:
  - input level and view hash;
  - options offered;
  - chosen label and probabilities;
  - model, endpoint host and prompt hash;
  - tokens;
  - a `trainable` flag.
- `trainable` is false for Jev (TypeSafe MCA §2.3(b)) and true otherwise. The docs say provider terms decide.
- The log never holds API keys.

### G5. The benchmark: about 100 documents

**Set.**
- The v0.5 gold split: 113 public documents across 8 doc types (annual report 25, government report 26, contract 14, research paper 14, fact sheet 11, court opinion 11, prospectus 6, technical standard 6).
- It is used for development measurement.
- The frozen held-out corpus (40 documents) is run once, at the end, for doc type only.

**Gold.**
- **Doc type** comes from the source-assigned type in the manifest. The builder spot-checks it and records any corrections.
- **Section tags** are a new evaluation-only gold set: Claude annotators label the level-1 and level-2 sections of the 113 documents against the catalogue. They label blind and double-annotate a 20% subset for κ. They work from an annotation guide written first, and the gold is committed before any model scores these documents. Claude labels are evaluation-only (program rule).

**Grid.**
- Flow: `doc_first` or `sections_first`.
- Level: `headings`, `terms`, `lead` or `full` (section and document level).
- Backend: keyword; UltraBERT; local LLM (Qwen3-4B via mlx_lm.server); and optionally Jev on the best two levels, within the program's remaining US$25 cap.

**Metrics.**
- doc-type accuracy and macro-F1;
- section-tag macro-F1 and micro-F1 at leaf and at group;
- the share of documents decided by rules alone;
- prompt tokens per document (exact for LLMs) and the token share against `full`;
- wall time.

**Headline.** A table of accuracy against tokens for each level and flow. It answers "how much accuracy does each rung buy, and at what token cost", against the full-document baseline.

**Output.** `benchmarks/classification_bench.py`, with results in `benchmarks/results/<date>/` and `docs/benchmarks-classification.md`.

### G6. A real v1 package

- **Version 1.0.0.** The README is rewritten around the ladder and the sections-first flow, and the docs site is updated.
- **CHANGELOG** for 1.0.0.
- **CI:** a GitHub Actions workflow runs ruff, mypy and the unit tests on Python 3.10 to 3.12.
- **Packaging:** the wheel and sdist build, and a fresh-venv install from the wheel passes a CLI smoke test. The package installs with `pip install git+https://github.com/<owner>/docrooter` and from a GitHub release wheel.
- **PyPI publishing is out of scope:** it needs the user's token. The release notes say how to do it.
- **Experimental areas:** the learned tagger (teach, tagger, train, corpus) and selection (`select`, the dial) ship marked experimental in the docs. The selection guarantee is claimed only as measured.

## 3. Non-goals and anti-goals

**Non-goals:**
- deriving or reshaping the taxonomy from the corpus;
- training student models as a product goal;
- PyPI release;
- reading chart values.

**Anti-goals:**
- **Over-sending:** never sending more context than the chosen level.
- **Estimated tokens passed off as exact:** token numbers reported as exact must come from API usage.
- **Gold leakage:** any model scoring a gold or held-out document before its gold is committed.
- **Jev in training:** Jev outputs marked trainable.
- **Secrets:** keys in logs, results or commits.
- **Private or third-party files in git:** the user's private `tests/_10-K-2025-As-Filed.pdf` or any corpus PDF.

## 4. Cleanup (before v1 work)

- **Archive workspaces.** Move the v0.5 and v0.6 SDD workspaces to `.superpowers/archive/` (git-ignored), with their rulings summarised in `.superpowers/rulings-v0.5.md` and `-v0.6.md`.
- **Remove strays.** Remove stray scratch files and any untracked debris in the repo root.
- **Branches.** Fast-forward `main` through the stacked branches (feat/v0.1 → … → v0.6) locally. v1 work continues on `feat/v1`.
- **Keep stopped work.** The stopped v0.5 compute artefacts (teacher labels, stage A bundle) stay in `benchmarks/.cache.nosync/` and are documented as experimental. Nothing is deleted from the cache.

## 5. Constraints

- The torch-free core, one model process at a time, the 8 GB budget, `PYTHONPATH=src` and no plain `uv sync` all still hold.
- Never read or print `.env`.
- Never commit PDFs from the corpus, the user's private PDF, or annotation text.
- Publishing to GitHub happens once, at the end, after the user confirms the owner and visibility.

## 6. Left to the builder

- The catalogue's exact leaves and definitions, reviewed like taxonomies.
- The rule blocks for every doc type.
- The exact sentence and token budgets of each level preset.
- The cascade threshold default.
- The annotation guide for section gold.
- The benchmark's output format.
