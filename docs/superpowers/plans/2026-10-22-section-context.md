# Section context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An application can ask a classified document for a section's full text by its tag
(one tag, several, or all) and get JSON-ready results, with no re-reading and no model calls.

**Architecture:** A new module `src/docrooter/context.py` holds a pydantic `TaggedDocument`
(a `Document` plus its `Classification`). Its methods return `SectionContext` results made of
`SectionMatch` entries, and it saves and loads itself as JSON. A `docrooter sections` CLI
command wraps it. Nothing in `classify`, the views or the catalogue changes.

**Tech Stack:** Python 3.10+, pydantic v2, typer (CLI), pytest.

**Spec:** `docs/superpowers/specs/2026-10-22-section-context-design.md`. Read it first: the
decisions D1–D7 are the requirements, and this plan does not repeat them.

## Global Constraints

- **Running tools.** Always `PYTHONPATH=src`, with the repo venv:
  - tests: `PYTHONPATH=src .venv/bin/python -m pytest <path> -q`;
  - lint: `.venv/bin/ruff check . && .venv/bin/ruff format --check .`;
  - types: `PYTHONPATH=src .venv/bin/python -m mypy`.
- **What not to do.**
  - Never run `uv sync`.
  - Never read or print `.env`.
  - No network, no model calls, no torch in the new tests.
- **The working tree.**
  - Work on branch `feat/section-context` in the main checkout. Never switch branches, stash
    or reset.
  - The repo sits in iCloud. If a file named like `"<name> 2.<ext>"` appears, don't commit it.
- **Style.**
  - Match the surrounding code: docstrings in the package's plain style, and
    `model_config = ConfigDict(extra="forbid")` on the models.
  - Use `from __future__ import annotations`.
- **Names from the spec, verbatim.**
  - Format: `"docrooter.tagged_document"`, version `1`.
  - Statuses: `"found"`, `"not_found"`, `"unknown_tag"`.
  - The leaf `"none"` is excluded from `available_tags()` and `get_all_sections()`.
- **Commits.** End every commit message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A classification from another reading.** A section id in `classification.sections` that
   isn't in the document must raise `ValueError` naming the id, not a `KeyError` later (Task 1
   test).
2. **Sloppy tag input.** `" financial.balance_sheet "` is stripped and found. Case is exact,
   so `"Financial.Balance_Sheet"` is `unknown_tag` (Task 1 test).
3. **Empty documents.** A document with no tagged sections gives `available_tags() == []`,
   `get_all_sections() == []` and `get_section(x).status == "not_found"`, without crashing
   (Task 1 test).
4. **A custom catalogue that isn't passed.** When `section_taxonomy` is `None` or not a
   built-in name, an unknown tag is reported `not_found`, no exception is raised, and an exact
   label match still works (Task 1 test).
5. **Non-ASCII text, and more than six heading levels.**
   - Headings such as "Bilan consolidé" survive the `save`/`load` round trip.
   - A level-8 descendant renders as `######` (Tasks 1 and 2 tests).

---

### Task 1: `TaggedDocument` lookups

**Files:**
- Create: `src/docrooter/context.py`
- Test: `tests/unit/test_context.py`

**Interfaces:**
- Consumes:
  - `docrooter.document.models.Document`, `Section`, `Document.walk()`;
  - `docrooter.flow.Classification`, `SectionTag`, `NO_TAG`;
  - `docrooter.views.estimate_tokens(text: str) -> int`;
  - `docrooter.taxonomy.Taxonomy` (`__contains__`, `path(node_id) -> list[TaxonomyNode]`);
  - `docrooter.builtin_taxonomy(name)`, which raises `UnknownTaxonomyError` for an unknown
    name.
- Produces (later tasks rely on exactly these):
  - `class SectionMatch(BaseModel)`. Fields, in order:
    - `section_id: str`, `tag: str`, `heading: str`, `path: list[str]`, `level: int`;
    - `text: str`, `page_start: int | None`, `page_end: int | None`;
    - `confidence: float | None`, `confidence_source: str`, `tokens: int`.
  - `class SectionContext(BaseModel)`: `tag: str`,
    `status: Literal["found", "not_found", "unknown_tag"]`, `message: str | None`,
    `sections: list[SectionMatch]`.
  - `class TaggedDocument(BaseModel)`:
    - fields `document: Document` and `classification: Classification`;
    - a private `_taxonomy`, set from a constructor keyword:
      `TaggedDocument(document=..., classification=..., taxonomy: Taxonomy | str | None = None)`;
    - `doc_type -> str | None` (a property) and `available_tags() -> list[str]`;
    - `get_section(tag: str, *, min_confidence: float | None = None) -> SectionContext`;
    - `get_sections(tags: Sequence[str], *, min_confidence: float | None = None) -> list[SectionContext]`;
    - `get_all_sections(*, min_confidence: float | None = None) -> list[SectionContext]`.

The test fixture builds a 10-K-shaped document with `Document.from_sections` and a hand-built
`Classification` with `flow="doc_first"`, `decided_by="model"`, `backend="test"`,
`section_taxonomy="section_catalogue"`, and one `SectionTag(level_used="headings", ...)` per
heading. Then set `page_start` on the sections with `model_copy`, or build them directly. It
has:

- PART II (level 1, `none`), containing:
  - Item 8 (level 2, `financial.financial_statements`), containing:
    - CONSOLIDATED BALANCE SHEETS (level 3, `financial.balance_sheet`), with a Markdown table in
      its text;
    - a "Current assets" child (level 4, `financial.balance_sheet`).
  - Notes (level 2, `financial.notes_to_financial_statements`), with two children, "Note 1" and
    "Note 4", each tagged `financial.notes_to_financial_statements`, which collapse into Notes.
  - A sibling pair at level 2, "Note A" and "Note B", both
    `financial.notes_to_financial_statements`, which stay two matches.
- Item 1A Risk Factors (level 1, `disclosure.risk_factors`, confidence `None`,
  `confidence_source="text"`).

Use a second, smaller fixture where you need sibling-only matches. Keep the fixture in the test
file.

- [ ] **Step 1: Write the failing tests** in `tests/unit/test_context.py`.
  - `test_get_section_returns_heading_and_full_text`:
    - the balance sheet context has `status == "found"`, one match, and
      `heading == "CONSOLIDATED BALANCE SHEETS"`;
    - its `text` starts with the table, contains `"#### Current assets"`, and does not contain
      the balance-sheet heading line;
    - `path == ["PART II", "Item 8. Financial Statements"]`;
    - `tokens == estimate_tokens(heading + "\n\n" + text)`.
  - `test_nested_matches_collapse`: the notes give one match (Notes), whose text contains both
    "Note 1" and "Note 4".
  - `test_same_tag_siblings_stay_separate`: Note A and Note B are two matches, in reading order.
  - `test_group_tag_matches_leaves_under_it`:
    - `get_section("financial")` returns the outermost financial sections only (Item 8 and
      Notes…), each match's `tag` being its own leaf;
    - the balance sheet is not repeated, because it sits inside Item 8.
  - `test_missing_tag_is_not_found`:
    - `fund.schedule_of_investments` has `status == "not_found"`, `sections == []`, and a
      message containing the tag;
    - no exception is raised.
  - `test_unknown_tag`: `"financial.balence_sheet"` has `status == "unknown_tag"` and a message
    containing `"section_catalogue"`.
  - `test_tag_is_stripped_but_case_exact` (Review Focus 2).
  - `test_get_sections_keeps_order_and_duplicates`: asking `[b, a, b]` returns three contexts
    with tags `[b, a, b]`.
  - `test_get_all_sections_excludes_none`:
    - one context per leaf present, in reading order of first appearance;
    - `"none"` is absent;
    - `available_tags()` gives the same tags in the same order.
  - `test_min_confidence_drops_low_keeps_unknown`:
    - a match with confidence 0.2 is dropped at `min_confidence=0.5`, and status becomes
      `not_found` when nothing is left;
    - the risk-factors match (confidence `None`) is kept.
  - `test_page_end_is_next_section_start`: `page_start` and `page_end` follow D2, and both are
    `None` when the sections have no pages.
  - `test_deep_levels_cap_at_six_hashes`: a level-8 descendant renders as `"###### "`.
  - `test_mismatched_classification_raises`: a `SectionTag` id `"s99"` raises a `ValueError`
    whose message contains `"s99"`.
  - `test_empty_document` (Review Focus 3).
  - `test_unresolvable_taxonomy_reports_not_found` (Review Focus 4):
    - `section_taxonomy=None` and `"my_custom"` both work;
    - an exact label still matches;
    - a passed `Taxonomy` object enables `unknown_tag`.
  - `test_doc_type_property`.

- [ ] **Step 2: Run the tests and see them fail.**
  Run `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_context.py -q`. Expected: the
  import of `docrooter.context` fails.

- [ ] **Step 3: Implement `src/docrooter/context.py`** to the spec's D1–D5. Approach notes:
  - Index the labels by section id from `classification.sections`.
  - The reading order is `document.walk()`.
  - A match is "inside" another when the other's id is in its ancestor chain; walk
    `parent_id`.
  - "Under the tag in the catalogue" is `tag in {n.id for n in taxonomy.path(label)}`. Without
    a taxonomy, fall back to equality, or a `label.startswith(tag + ".")` prefix test.
  - Resolve the taxonomy once, in `model_post_init`. A `str` goes through `builtin_taxonomy`,
    and `UnknownTaxonomyError` becomes `None`.
  - `page_end` is the `page_start` of the first section in reading order after the match's
    subtree, else the largest `page_start` within the subtree.
  - The preamble (level 0) never appears in `path`.

- [ ] **Step 4: Run the tests and see them pass.** Run the same command. Expected: all pass.
  Then run `.venv/bin/ruff check src/docrooter/context.py tests/unit/test_context.py` and
  `PYTHONPATH=src .venv/bin/python -m mypy`, which must be clean.

- [ ] **Step 5: Commit.** Commit `src/docrooter/context.py` and `tests/unit/test_context.py`
  with the message `feat: section context lookups by tag (TaggedDocument)`.

### Task 2: Persistence, `from_source`, and the public export

**Files:**
- Modify: `src/docrooter/context.py`
- Modify: `src/docrooter/__init__.py` (the import and `__all__`)
- Modify: `tests/unit/test_public_api.py` (`PUBLIC_NAMES`)
- Test: `tests/unit/test_context.py`

**Interfaces:**
- Consumes: Task 1's `TaggedDocument`; `docrooter.load(source)`;
  `docrooter.classify(doc, **kwargs) -> Classification`.
- Produces:
  - `TaggedDocument.to_json() -> str` and
    `TaggedDocument.from_json(data: str | bytes, *, taxonomy: Taxonomy | str | None = None) -> TaggedDocument`;
  - `TaggedDocument.save(path: str | os.PathLike[str]) -> None` (UTF-8);
  - `TaggedDocument.load(path, *, taxonomy=None) -> TaggedDocument`;
  - `TaggedDocument.from_source(source: Document | str | os.PathLike[str], *, taxonomy=None, **classify_kwargs: Any) -> TaggedDocument`;
  - `TaggedDocument`, `SectionContext` and `SectionMatch` importable from `docrooter`.

- [ ] **Step 1: Write the failing tests.**
  - `test_json_round_trip`:
    - the JSON has top-level keys `{"format", "version", "document", "classification"}`, with
      `format == "docrooter.tagged_document"` and `version == 1`;
    - `from_json(to_json())` gives equal `get_all_sections()`.
  - `test_save_load_round_trip_unicode`: through `tmp_path`, a heading "Bilan consolidé"
    survives.
  - `test_from_json_rejects_wrong_format`: a bad `format` or `version` raises `ValueError`.
  - `test_from_source_loads_and_classifies`:
    - monkeypatch the `classify` that `context.py` calls to return a fixture classification,
      and pass a Markdown string;
    - assert the kwargs (`backend="jev"`, `flow="doc_first"`) reached it;
    - `from_source` on a `Document` skips `load`.
  - In `test_public_api.py`, add the three names to `PUBLIC_NAMES`.

- [ ] **Step 2: Run the tests and see them fail.** Run
  `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_context.py tests/unit/test_public_api.py -q`.
  Expected: the new tests fail with `AttributeError` or `ImportError`.

- [ ] **Step 3: Implement.**
  - Serialise with `model_dump(mode="json")` inside the envelope, using `json.dumps(...,
    ensure_ascii=False)`.
  - Import `load` and `classify` lazily inside `from_source`, if the top-level import order
    needs it.
  - Export from `docrooter/__init__.py`, alphabetically in `__all__`.

- [ ] **Step 4: Run the tests and see them pass.**
  - Run the same command, then the full unit suite:
    `PYTHONPATH=src .venv/bin/python -m pytest tests/unit -q -x`. Expected: everything passes.
  - Run ruff and mypy, which must be clean.

- [ ] **Step 5: Commit** with the message
  `feat: save, load and from_source for TaggedDocument; export it`.

### Task 3: `docrooter sections` CLI

**Files:**
- Modify: `src/docrooter/cli.py` (a new `@app.command("sections")`, next to `classify`)
- Test: `tests/unit/test_cli_sections.py`

**Interfaces:**
- Consumes: Task 2's `TaggedDocument.load`, `from_source`, `save`, and the `get_*` methods.
- Produces the CLI, to spec D6:
  - `docrooter sections SOURCE [--tag TAG]... [--all] [--list] [--save PATH] [--min-confidence F] [--backend/-b B] [--flow F] [--level L]`.

- [ ] **Step 1: Write the failing tests** with `typer.testing.CliRunner`, as the other
  `test_cli_*.py` files do.
  - `test_sections_from_saved_json_tag`:
    - save the fixture with `TaggedDocument.save`, then run `sections saved.json --tag
      financial.balance_sheet --tag fund.schedule_of_investments`;
    - exit 0;
    - the JSON has `doc_type`, `source`, and `results` with statuses `["found", "not_found"]`.
  - `test_sections_list_is_default`:
    - with no `--tag`/`--all`/`--list`, the output has `tags` equal to `available_tags()`;
    - with `--all`, `results` has one per tag.
  - `test_sections_document_source_classifies_and_saves`:
    - monkeypatch `TaggedDocument.from_source` to return the fixture;
    - run on a `.md` file with `--save out.json -b jev --flow doc_first`;
    - the kwargs reached `from_source`, and `out.json` loads back.
  - `test_sections_rejects_tag_with_all`: `--tag x --all` exits nonzero with a usage message.

- [ ] **Step 2: Run the tests and see them fail.** Run
  `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_cli_sections.py -q`. Expected:
  "No such command 'sections'".

- [ ] **Step 3: Implement the command** in `cli.py`.
  - It follows `classify`'s option style.
  - A source whose suffix is `.json` goes to `TaggedDocument.load`; anything else goes to
    `from_source`, with only the options the user gave.
  - It prints with `typer.echo(json.dumps(..., indent=2, ensure_ascii=False))`.

- [ ] **Step 4: Run the tests and see them pass.**
  - Run the same command, plus `tests/unit/test_cli_errors.py`. Expected: all pass.
  - Run ruff and mypy, which must be clean.

- [ ] **Step 5: Commit** with the message `feat: docrooter sections CLI`.

### Task 4: Guide, README, example, CHANGELOG

**Files:**
- Create: `docs/site/guides/section-context.md`
- Create: `examples/section_context.py`
- Modify: `mkdocs.yml` (the nav, under Guides, next to the select-for-extraction guide)
- Modify: `README.md` (a short "Get a section's text" section after the classify usage)
- Modify: `CHANGELOG.md` (Unreleased, then Added)
- Modify: `tests/unit/test_examples.py` (a test that runs the example)

**Interfaces:**
- Consumes: everything from Tasks 1–3.

- [ ] **Step 1: Write the failing test** `test_section_context_example_runs` in
  `test_examples.py`.
  - It runs `examples/section_context.py` with `runpy`, and asserts the exact printed lines
    the example produces. The first line is
    `financial.balance_sheet: found (CONSOLIDATED BALANCE SHEETS)`; one line is
    `fund.schedule_of_investments: not_found`.
  - The example builds its document with `Document.from_sections` and a hand-built
    `Classification`, so it runs offline. Its docstring shows the real call
    (`TaggedDocument.from_source("10k.pdf", backend="jev")`).

- [ ] **Step 2: Run it and see it fail** (the file is missing).

- [ ] **Step 3: Write the example, the guide, the README section, the nav entry and the
  CHANGELOG entry.**
  - The guide, in plain English, covers:
    - the platform flow (read and classify once, save, ask later);
    - the three calls and `available_tags`;
    - the JSON shape, copied from D2;
    - the statuses;
    - what the text includes (D3);
    - nested and group matches (D4);
    - storing the JSON in a database;
    - the CLI;
    - "doc-rooter supplies context; your application extracts".
  - Link it from the select-for-extraction guide's intro as the simple, tag-based route. If
    `test_docs_snippets.py` checks docs code blocks, keep the snippets valid.

- [ ] **Step 4: Run the tests and see them pass.**
  - Run `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_examples.py tests/unit/test_docs_snippets.py -q`.
  - Then run the full `tests/unit`, ruff and mypy, all clean.

- [ ] **Step 5: Commit** with the message
  `docs: section context guide, example, README and changelog`.
