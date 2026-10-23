# Parse for extraction Implementation Plan (revision 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `docrooter.parse(source, include="type" | "sections" | "text")` and
`docrooter parse` return one clean, self-contained JSON response. At `"text"`, it includes the
full text of every tagged section, grouped by tag.

**Architecture:**
- Revision 1's Tasks 1–3 are already on the branch: commits a9a8976, 1989af6 and 021abcc.
  They built `TaggedDocument` (lookups, save/load) and a `docrooter sections` CLI.
- This revision keeps their section-text engine in `src/docrooter/context.py` (`SectionMatch`,
  `SectionContext`, and the text, path, page and collapse logic). It removes `TaggedDocument`
  and the `sections` command, and adds `parse()`, `ParseResult` and `TaggedSection` on top.
- The tasks are numbered 4–6 so that their briefs and reports don't collide with revision 1's.

**Tech Stack:** Python 3.10+, pydantic v2, typer, pytest.

**Spec:** `docs/superpowers/specs/2026-10-22-section-context-design.md` (revision 2). Read it
first. D1–D7 are the requirements, and this plan does not repeat them.

## Global Constraints

- **Running tools.** Always use `PYTHONPATH=src` with the repo venv:
  - tests: `PYTHONPATH=src .venv/bin/python -m pytest <path> -q`;
  - lint: `.venv/bin/ruff check . && .venv/bin/ruff format --check .`;
  - types: `PYTHONPATH=src .venv/bin/python -m mypy`.
- **What not to do.**
  - Never run `uv sync`.
  - Never read or print `.env`.
  - No network, no model calls except through the offline `keyword` backend, and no torch in
    tests.
- **The working tree.**
  - Work on the branch `feat/section-context` in the main checkout. Never switch branches,
    stash or reset.
  - The repo sits in iCloud. If a file named like `"<name> 2.<ext>"` appears, don't commit it.
  - Commit only the files you changed.
- **Style.**
  - Match the surrounding code: plain docstrings, `ConfigDict(extra="forbid")`, and
    `from __future__ import annotations`.
- **Names from the spec, verbatim.**
  - `include` values: `"type"`, `"sections"`, `"text"`. The default is `"sections"`.
  - Statuses: `"found"`, `"not_found"`, `"unknown_tag"`.
  - The leaf `"none"` is in `sections` but never a key of `section_text`.
  - `ParseResult.version == 1`.
  - `parse`'s default flow is `"doc_first"`.
- **Commits.** End every commit message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **`include="type"` must not tag sections.** Prove it with a spy on `classify`
   (`flow="doc_first"`, `max_sections=0`), and with a keyword-backend run whose
   `tokens["sections"] == 0` and whose `sections is None` (Task 4 tests).
2. **A stored response round-trips.** `ParseResult.model_validate_json(r.model_dump_json())`
   equals `r`, with non-ASCII headings, at each of the three levels. The fields a level leaves
   out are absent from the JSON (Task 4 test).
3. **Asking the helper below `"text"`.** `get_section` on a `"sections"` result raises a
   `ValueError` that names `include="text"`; it does not return `not_found` (Task 4 test).
4. **Group queries across keys.** `get_section("financial")` on a result holding both
   `financial.financial_statements` (Item 8) and `financial.balance_sheet` (inside Item 8)
   returns Item 8 only (Task 4 test).
5. **Conflicting options.**
   - `include="type"` with `flow="sections_first"`, or with `max_sections`, raises
     `ValueError`.
   - A bad `include` raises `ValueError`.
   - On the CLI, a bad `--include` is a usage error (Tasks 4 and 5 tests).

---

### Task 4: `parse()` and `ParseResult`, replacing `TaggedDocument`

**Files:**
- Modify: `src/docrooter/context.py`
- Modify: `src/docrooter/__init__.py`, `tests/unit/test_public_api.py`,
  `docs/site/reference/api.md`
- Modify: `tests/unit/test_context.py`. Keep and adapt the text, collapse, page, path and deep
  level tests. Delete the `TaggedDocument`, save/load and `from_source` tests.
- Delete: `src/docrooter/cli.py`'s `sections` command and `tests/unit/test_cli_sections.py`.
  `docrooter sections` must not exist after this task, because it imports `TaggedDocument`.
  Task 5 adds `parse`.

**Interfaces:**
- Consumes: `docrooter.flow.classify` and `Classification`; `docrooter.document.load.load`;
  `docrooter.builtin_taxonomy`; and `Taxonomy`.
- Produces (Tasks 5 and 6 rely on these):
  - `Include = Literal["type", "sections", "text"]` and `INCLUDES: tuple[str, ...]`;
  - `class TaggedSection(BaseModel)` with the fields `section_id`, `heading`, `level`, `path`,
    `tag`, `confidence`, `page_start` (spec D2);
  - `class ParseResult(BaseModel)` with the fields `version: Literal[1] = 1`, `source`,
    `include`, `backend`, `doc_type`, `doc_type_confidence`, `tokens: dict[str, int]`,
    `sections: list[TaggedSection] | None`, `section_text: dict[str, list[SectionMatch]] | None`.
    Its methods are `get_section(tag, *, catalogue=None) -> SectionContext` and
    `get_sections(tags, *, catalogue=None) -> list[SectionContext]`;
  - `parse(source, *, include="sections", flow="doc_first", backend=DEFAULT_BACKEND, level=None, **classify_kwargs) -> ParseResult`;
  - module functions `tagged_sections(document, classification) -> list[TaggedSection]` and
    `section_text(document, classification) -> dict[str, list[SectionMatch]]`;
  - public exports `parse`, `ParseResult`, `TaggedSection`, `SectionMatch` and
    `SectionContext`, with `TaggedDocument` removed.

- [ ] **Step 1: Write the failing tests** in `tests/unit/test_context.py`.
  - Keep the existing fixtures: the 10-K-shaped `Document.from_sections` with a hand-built
    `Classification`. Call `section_text(doc, cls)` and `tagged_sections(doc, cls)` directly.
  - Port these tests: heading and full text; nested collapse; same-tag siblings; the
    six-hash cap; `page_end`; path without the preamble; tokens; the mismatched id
    (`section_text` raises a `ValueError` naming `"s99"`).
  - New tests:
    - `test_section_text_keys`: the keys are leaf tags in reading order of first appearance,
      without `"none"`. Item 8's entry contains the balance-sheet text, and the
      `financial.balance_sheet` key has it too (spec D4).
    - `test_tagged_sections_lists_every_tag`: every tagged heading is listed in reading order,
      `none` included.
    - `test_parse_type_only_skips_sections`:
      - spy on `context.classify`, and assert `flow="doc_first"` and `max_sections=0`;
      - the result has `sections is None` and `section_text is None`;
      - its JSON (`json.loads(model_dump_json())`) has no `sections` or `section_text` key.
    - `test_parse_levels_end_to_end_keyword`:
      - run on a Markdown string with `backend="keyword"` and `level="headings"` at each of
        `"type"`, `"sections"` and `"text"`;
      - `doc_type` is the same at all three;
      - `"type"` has `tokens["sections"] == 0`;
      - `"text"` has a `section_text` whose entries' `text` comes from the Markdown.
    - `test_parse_passes_classify_kwargs`: `calls="per_item"` reaches the spied classify.
    - `test_parse_rejects_conflicts` (Review Focus 5).
    - `test_parse_result_round_trip` (Review Focus 2): it uses a heading "Bilan consolidé".
    - `test_get_section_found_missing_unknown`:
      - `found` returns the balance sheet;
      - `financial.schedule_of_investments` is `not_found`;
      - `financial.balence_sheet` is `unknown_tag`, with a message naming `section_catalogue`;
      - `" financial.balance_sheet "` is stripped and found;
      - passing a custom `Taxonomy` as `catalogue` changes what counts as unknown.
    - `test_get_section_group_collapses_across_keys` (Review Focus 4).
    - `test_get_sections_order_and_duplicates`: asking `[b, a, b]` returns three in that
      order.
    - `test_get_section_needs_text_level` (Review Focus 3).

- [ ] **Step 2: Run the tests and see them fail.**
  - Run `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_context.py -q`.
  - Expected: `ImportError` on `parse` and `section_text`.

- [ ] **Step 3: Implement.**
  - Refactor the lookup internals of `TaggedDocument` into module functions, then delete
    `TaggedDocument`, `FORMAT` and `FORMAT_VERSION`.
  - Leave out the `None` level fields with a pydantic `model_serializer(mode="wrap")` that
    drops `sections` and `section_text` when they are `None`. `Section`'s
    `_omit_empty_element_ids` is the pattern to follow.
  - `tokens` sums `prompt + completion` of each `stage_tokens` entry.
  - Update the module docstring to the revision-2 picture.
  - Remove the `sections` CLI command and its test file.
  - Update the exports, `PUBLIC_NAMES` and `api.md`.

- [ ] **Step 4: Run the tests and see them pass.**
  - Run the same command, then the full `tests/unit`. Ruff and mypy must be clean.

- [ ] **Step 5: Commit** with the message
  `feat: parse() with type, sections and text response levels`.

### Task 5: `docrooter parse` CLI

**Files:**
- Modify: `src/docrooter/cli.py`, which gets a new `@app.command("parse")` next to `classify`.
- Test: `tests/unit/test_cli_parse.py`

**Interfaces:**
- Consumes: Task 4's `parse`, `INCLUDES` and `ParseResult`.
- Produces: `docrooter parse FILE [--include type|sections|text] [--backend/-b B] [--flow F] [--level L] [--out PATH]` (spec D6).

- [ ] **Step 1: Write the failing tests** with `CliRunner` and `-b keyword`, as
  `test_cli_classify.py` does.
  - `test_parse_cli_prints_json_at_each_level`: the output parses as a `ParseResult`, and at
    `--include type` it has no `sections` key.
  - `test_parse_cli_out_writes_file`: with `--out`, the file holds the JSON (UTF-8) and stdout
    is empty or a short confirmation. Match how `classify -t --out` reports.
  - `test_parse_cli_rejects_bad_include`: the exit code is nonzero with a usage message.
  - `test_parse_cli_type_with_sections_first_is_an_error`.

- [ ] **Step 2: Run the tests and see them fail.**
  - Run `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_cli_parse.py -q`.
  - Expected: "No such command 'parse'".

- [ ] **Step 3: Implement the command.**
  - Follow `classify`'s option style.
  - Map `parse`'s `ValueError` to a usage error.
  - Print with `json.dumps(..., indent=2, ensure_ascii=False)` on `model_dump(mode="json")`.

- [ ] **Step 4: Run the tests and see them pass.** Then run the full `tests/unit`, ruff and
  mypy, all clean.

- [ ] **Step 5: Commit** with the message `feat: docrooter parse CLI`.

### Task 6: Guide, README, example, CHANGELOG

**Files:**
- Create: `docs/site/guides/section-context.md`
- Create: `examples/parse_for_extraction.py`
- Modify: `mkdocs.yml`: the nav, under Guides, next to select-for-extraction.
- Modify: `README.md`: a short "Parse a document for extraction" section after the classify
  usage.
- Modify: `CHANGELOG.md`: under Unreleased, Added.
- Modify: `tests/unit/test_examples.py`: add a test that runs the example.

**Interfaces:**
- Consumes: Tasks 4–5.

- [ ] **Step 1: Write the failing test** `test_parse_for_extraction_example_runs`.
  - The example parses a small Markdown filing with `backend="keyword"` at `include="text"`.
  - It prints the `doc_type` line, then one line per key of `section_text`, as
    `<tag>: <heading>`.
  - The test asserts the exact lines the example produces.
  - The example's docstring shows the real call:
    `parse("q3.pdf", include="text", backend="jev")`.

- [ ] **Step 2: Run it and see it fail** (the file is missing).

- [ ] **Step 3: Write the example, the guide, the README section, the nav entry and the
  CHANGELOG entry.**
  - The guide, in plain English, covers:
    - the platform flow (parse once, store the JSON, the extraction engine reads it);
    - the three levels table and its cost note, from the spec;
    - the JSON shape, copied from D2;
    - what the text includes (D3), and nested and repeated text (D4);
    - the `get_section` helper and its statuses;
    - the CLI;
    - "doc-rooter supplies context; your application extracts".
  - Link the guide from the select-for-extraction guide's intro, as the simple, tag-based
    route.
  - Keep any docs code blocks valid for `test_docs_snippets.py`.

- [ ] **Step 4: Run the tests and see them pass.**
  - Run `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_examples.py tests/unit/test_docs_snippets.py -q`.
  - Then run the full `tests/unit`, ruff and mypy, all clean.

- [ ] **Step 5: Commit** with the message
  `docs: parse for extraction guide, example, README and changelog`.
