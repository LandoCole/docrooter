# doc-rooter parse: one call, three response levels, section text included

2026-10-22, revision 2. Asked for by the user, building on the demo's "Why" point that section
tags make data extraction more accurate. Revision 1 (in git history) stored the whole document
and had the application call doc-rooter again to cut out a section. The user replaced that with
this design: the application says up front what it needs, and doc-rooter puts it all in one
JSON response.

## Why

An extraction platform has doc-rooter built in. A PDF is uploaded, doc-rooter reads it, builds
the section tree and classifies it, and sends the result back. The application stores that JSON
in its database. Later its rules engine reads it:

1. "This is a quarterly report."
2. "It has a schedule of investments."
3. "Pull every company and its fair value from the schedule of investments."

The extraction engine takes the schedule of investments text straight from the stored JSON and
sends it to its model. There is no call back to doc-rooter, no re-feeding the document, and
nothing to hand-roll. doc-rooter supplies the context; the application (or another package)
does the extraction and question answering.

## The three response levels (the user's decision)

`include` says how much comes back:

| `include` | What's in the response | Model calls |
|---|---|---|
| `"type"` | the document type | the doc-type call only; no section tags (about 98% cheaper on Jev for a 10-K) |
| `"sections"` | the type, plus every heading with its section tag | type plus section tags |
| `"text"` | the type and the tags, plus the full text of each tagged section, grouped by tag | same as `"sections"`; the text costs no calls |

## Goals

- One function, `docrooter.parse(source, include=...)`, and one CLI command, `docrooter parse`,
  return a clean, self-contained JSON response at the chosen level.
- At `"text"`, the response has everything the extraction engine needs. A reader needs only a
  JSON parser, not doc-rooter.
- A tagged section's text runs from its heading to the next heading at the same or a higher
  level, with its subsections and tables included.
- A small Python helper on the result answers "give me tag X" with an explicit `found`,
  `not_found` or `unknown_tag`.

## Non-goals

- Extraction or question answering. doc-rooter never calls a model to answer anything here.
- An HTTP server, LLM tool schemas, or the demo site (possible follow-ups).
- Changing `classify`, `Classification`, the catalogue, the views or the benchmark.
- Asking doc-rooter later for text it didn't return. The application chooses `include` up
  front; text for an untagged or `none` section is not returned.
- Replacing the experimental `select()`.

## Anti-goals

- Returning a heading's own text only and missing its subsections or tables.
- Repeating text inside one tag's list, because a section and its subsection share a tag.
- A response full of internals (call records, token splits) that the application has to dig
  through.

## Decisions

### D1. `parse()`

```python
from docrooter import parse

result = parse("q3.pdf", include="text", backend="jev")
result.doc_type                                   # "financial.quarterly_report"
result.section_text["financial.schedule_of_investments"][0].text
result.model_dump_json()                          # store this; ParseResult.model_validate_json reads it back
```

- Signature: `parse(source, *, include="sections", flow="doc_first", backend=DEFAULT_BACKEND,
  level=None, **classify_kwargs) -> ParseResult`.
  - `source` is anything `classify` takes: a `Document`, a path, or Markdown text.
  - `classify_kwargs` pass straight through to `classify` (for example `calls`, `catalogue`,
    `doc_types`, `section_level`, `doc_level`).
  - An unknown `include` raises `ValueError`.
- `flow` defaults to `"doc_first"` (read the document, then tag). That is what the demo and the
  benchmark use; `classify` itself keeps its own default.
- `include="type"` calls `classify(..., flow="doc_first", max_sections=0)`: the doc-type call
  only, with no section tags. Passing `flow="sections_first"` with `include="type"` raises
  `ValueError`, because that flow decides the type from the tags. Passing `max_sections` with
  `include="type"` also raises.
- `"sections"` and `"text"` call `classify` once, with the given flow. `"text"` then builds the
  section text from the document and the classification, with no further calls.
- `parse` loads the source itself (`load`) when it isn't a `Document`, so that it has the
  document for the text.

### D2. `ParseResult`, the response

```json
{
  "version": 1,
  "source": "q3.pdf",
  "include": "text",
  "backend": "jev",
  "doc_type": "financial.quarterly_report",
  "doc_type_confidence": 0.94,
  "tokens": {"document": 1771, "sections": 88310},
  "sections": [
    {"section_id": "s3", "heading": "Schedule of Investments", "level": 1, "path": [],
     "tag": "financial.schedule_of_investments", "confidence": 0.97, "page_start": 12}
  ],
  "section_text": {
    "financial.schedule_of_investments": [
      {"section_id": "s3", "tag": "financial.schedule_of_investments",
       "heading": "Schedule of Investments", "path": [], "level": 1,
       "text": "| Company | Shares | Fair value |\n|---|---|---|\n...",
       "page_start": 12, "page_end": 15, "confidence": 0.97,
       "confidence_source": "probability", "tokens": 2210}
    ]
  }
}
```

- `ParseResult` fields:
  - `version` (always `1`), `source`, `include`, `backend`;
  - `doc_type` and `doc_type_confidence`, from the classification;
  - `tokens`: the tokens spent per stage (prompt plus completion), `{"document": int,
    "sections": int}`, from `classification.stage_tokens`. `sections` is `0` at `"type"`. A
    stage missing from `stage_tokens` counts as `0`.
  - `sections`: a list of `TaggedSection`, one per tagged heading in reading order, `none`
    included, so the application also gets the table of contents. It is `None` at `"type"`.
  - `section_text`: a `dict[str, list[SectionMatch]]`, keyed by leaf tag in reading order of
    each tag's first section, `none` excluded. It is `None` below `"text"`.
- In the JSON, a field that is `None` because the level doesn't include it (`sections`,
  `section_text`) is left out; `doc_type_confidence: null` stays. `model_dump_json()` and
  `model_validate_json()` round-trip.
- `TaggedSection`: `section_id`, `heading`, `level`, `path` (ancestor titles, outermost first,
  no preamble), `tag`, `confidence`, `page_start`.
- `SectionMatch`: as built in Task 1 (already in `context.py`). The fields are `section_id`,
  `tag`, `heading`, `path`, `level`, `text`, `page_start`, `page_end`, `confidence`,
  `confidence_source` and `tokens`.
- All models use `extra="forbid"`.

### D3. What a section's text is (unchanged from revision 1)

- `text` is the section's own body followed by every descendant in reading order. Each
  descendant is written as a Markdown heading (`"#" * min(level, 6)`, then the title) followed
  by its body. Parts are joined by a blank line.
- The section's own heading is in `heading`, not in `text`.
- `page_end` is an upper bound: the page where the next section after it starts, else the last
  page start within it.
- `tokens` is `estimate_tokens(heading + "\n\n" + text)`.

### D4. Which sections go under a tag in `section_text`

- Each leaf tag present (not `none`, not `None`) gets the sections carrying it, in reading
  order. Inherited tags count.
- **Nested matches collapse within a tag.** A section inside another section with the same tag
  is dropped, since its text is already there. Same-tag siblings ("Note 1" and "Note 4" as
  separate headings) stay separate entries.
- Across tags, text can repeat: "Item 8" (`financial.financial_statements`) contains the
  balance sheet, which is also under `financial.balance_sheet`. Each tag's entry is complete
  on its own.

### D5. The lookup helper on `ParseResult`

- `get_section(tag, *, catalogue=None) -> SectionContext`, and
  `get_sections(tags, *, catalogue=None) -> list[SectionContext]`, one per tag asked, in order.
  Both read only `section_text`; at a lower level they raise `ValueError` that says to use
  `include="text"`.
- A tag matches its own key, and a group tag (`"financial"`) matches every key under it
  (`"financial."` prefix). Across the keys, a match inside another match is dropped. Section
  ids are hierarchical (`s4.3` is under `s4`), so "inside" means the id starts with the other
  id plus `"."`. Matches come in reading order.
- `SectionContext` (already in `context.py`): `tag`, `status` (`"found"`, `"not_found"` or
  `"unknown_tag"`), `message`, `sections`. A tag is `unknown_tag` when it isn't in the
  catalogue, which is `catalogue` if given, else the built-in `section_catalogue`.
- Tags are stripped of surrounding spaces; case is exact. Neither call raises for a missing or
  unknown tag.

### D6. CLI: `docrooter parse`

```bash
docrooter parse q3.pdf --include text -b jev --out q3.json
docrooter parse q3.pdf --include type
```

- Options: `--include type|sections|text` (default `sections`), `--backend/-b` (default as for
  `classify`), `--flow` (default `doc_first`), `--level`, and `--out PATH`.
- It prints the `ParseResult` JSON (indented, `ensure_ascii=False`) to stdout, or writes it to
  `--out`.
- A wrong `--include` is a usage error.
- This replaces revision 1's `docrooter sections` command.

### D7. What changes from revision 1's code

- **Remove:**
  - `TaggedDocument`, with its save/load, `from_source` and JSON envelope;
  - the `docrooter sections` command;
  - their tests and the public names.
- **Keep:**
  - `SectionMatch` and `SectionContext`;
  - the text, path, page and collapse logic, refactored into module-level functions that take
    a `Document` and a `Classification`.
- **Add:** `parse`, `ParseResult` and `TaggedSection` in `src/docrooter/context.py`, exported
  from `docrooter` (and added to `__all__`, `PUBLIC_NAMES` and `docs/site/reference/api.md`).
- **Docs:**
  - a guide, `docs/site/guides/section-context.md` ("Parse a document for extraction"), in
    the nav;
  - a short README section;
  - `examples/parse_for_extraction.py`, which runs offline with the `keyword` backend on
    Markdown, plus a test that runs it;
  - a CHANGELOG entry.
- **Tests:** unit tests only. Use the `keyword` backend or a hand-built classification; no
  network and no torch.

## Revision 3 (2026-10-23): merged runs, and PDF heading levels

A live check (`parse --include text`, Jev) on four real PDFs (Apple 10-K, AT&T 2025 annual
report, iShares AGG prospectus, GAO-04-358) found boundaries exact wherever the reader's heading
tree was right, and two failure kinds where it wasn't. The user asked to fix both.

### What was found

- **Pieces.** Where a PDF's headings are flat, one logical section is several sibling headings.
  The AT&T notes came back as 42 entries (the "Notes to Consolidated Financial Statements" heading
  itself holds 7 tokens); the 10-K auditor's report as 8.
- **Over-runs**, from the PDF reader:
  - 10-K: bookmarks with typos ("Goverance", "Beceficial") or a plural ("Reports of Independent
    Registered Public Accounting Firm" against the heading "Report of …") fail the exact title
    match, so those headings keep their font-size level (4). Item 12 sits inside Item 11 and
    both auditor's reports inside the notes.
  - GAO: PyMuPDF4LLM missed six multi-line margin headings, including the level-2 findings
    chapters the bookmarks list ("Comparisons of the Tax Liabilities … Depend on the Measure
    Used", p. 10). Their level-3 subsections fall inside "Scope and Methodology".
  - AT&T: bookmarks have joined words ("STOCKHOLDERMATTERS", "ACCOUNTINGAND").

### D8. Merge runs of the same tag in `section_text`

- After nested matches collapse (D4), **consecutive matches of one tag merge into one entry**
  when no section between them in reading order carries a different label. `none` counts as a
  different label; an untagged section (`None`) does not break a run.
- A merged entry takes the first match's `section_id`, `heading`, `path`, `level` and
  `page_start`, and the last match's `page_end`. Its `text` is the first match's text, then each
  later match as a Markdown heading (`"#" * min(level, 6)`, then the title) followed by its
  text, joined by blank lines.
- `confidence` is the lowest known confidence of the merged matches (`None` only when all are
  `None`), and `confidence_source` is that match's. `tokens` is recomputed.
- **New field** `SectionMatch.section_ids: list[str]`: the ids of every match merged into the
  entry, in reading order (one id for an unmerged entry). The `get_section` collapse across keys
  treats an entry as covering all of its `section_ids`.
- This replaces D4's "same-tag siblings stay separate": "Note 1" and "Note 4" with nothing
  between them are now one entry.

### D9. PDF reader: fuzzy bookmarks, missing headings, no level jumps

In `load_pdf_markdown` (`src/docrooter/document/pdf.py`), with `use_bookmarks`:

1. **Fuzzy bookmark match.** A heading with no exact (normalised) bookmark match takes a
   bookmark whose normalised title has a `difflib.SequenceMatcher` ratio of at least 0.9 with
   the heading's (both titles at least 15 characters long, so "Revenue" never takes "Revenues"), and whose page is within one page of the heading's. Each bookmark is used at
   most once, in document order. The heading gets the `level_from_bookmark` flag, as for an
   exact match.
2. **Missing headings from bookmarks.** A bookmark (level ≥ 1, page ≥ 1) that no heading used is
   looked for in the body text of the heading whose text covers its page. The title's words are
   matched in order, case-insensitively, allowing any whitespace, Markdown emphasis or line
   breaks between them. When found at the start of a line, the body is split there: the text
   before stays, and a new heading follows with the bookmark's title and level, the page, the
   rest of the text, and the flag `heading_from_bookmark`. A bookmark that can't be found is
   skipped.
3. **Levels after bookmarks.** Bookmark levels are applied first; then, with `repair`, one gap repair runs (`repair_levels` with `keep_flags`). Bookmarked and bookmark-inserted headings keep their levels exactly. Every other heading goes one level below the nearest open heading with a smaller level (a stack keyed on the raw level, or on the kept level for a bookmarked heading), so no heading sits more than one level below the heading it nests under, and headings of one level stay siblings. A bookmark level itself may still jump. This replaces the repair-then-bookmarks order; a two-pass version flattened single-font headings.

Without bookmarks, only the gap repair changes: runs of same-level headings no longer nest (`# A / ### B / ### C` → B and C are siblings), in Markdown too.

### D10. The benchmark keeps its reading

The section gold (v2) is keyed by `section_id` and title hash on the reading it was annotated on,
`pymupdf4llm-1.28.2-7560c2e7a05a` in the readings cache. D9 changes the reader's source, which
changes `reading_key()`, so the benchmark would re-read every PDF and the gold would drift.

- `benchmarks/classification_bench.py` gets `GOLD_READING_KEY = "pymupdf4llm-1.28.2-7560c2e7a05a"`.
  Gold-grid and held-out runs read from that key's readings by default.
- `--fresh-reading` reads with the current reader under `reading_key()`, as before.
- Published numbers stay reproducible; measuring the new reader is a separate run.
