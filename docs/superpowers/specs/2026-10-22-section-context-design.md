# doc-rooter section context: hand an application a section's text by its tag

2026-10-22. Asked for by the user, building on the demo's "Why" point that section tags make data
extraction more accurate.

## Why

An extraction platform has doc-rooter built in. A PDF is uploaded, doc-rooter reads it, builds
the section tree and classifies it (the document type and a tag on every heading), and the
application gets the classification back. The application's rules engine then decides what to
extract: "this is a quarterly report and it has a schedule of investments, so pull every
company and its fair value from the schedule of investments."

Today the application has to hand-roll the next step: find the headings tagged
`financial.schedule_of_investments`, work out where each one ends, and stitch the text together.
This feature packages that step. The application says "give me the schedule of investments" and
gets back the tag, the heading and the full text of that section, ready to put in a prompt.
The application (or another package) does the extraction and question answering; doc-rooter
only supplies the context, exactly what is needed and nothing more.

## Goals

- Get one section's context by tag, several tags at once, or every tagged section.
- Each result carries the tag, the heading and the section's full text, from its heading up to
  the next heading at the same or a higher level, subsections included.
- A tag that is in the catalogue but not in the document returns an explicit "not found" result
  (no exception), with a message. A tag that is not in the catalogue at all returns "unknown
  tag", with a message.
- No re-reading and no model calls when asking for context. Reading and classifying happen once;
  the result can be saved (to a file or a database column) and asked later, from another process.
- JSON-ready results (pydantic models), so an HTTP service can return them as they are.

## Non-goals

- Extraction or question answering. doc-rooter never calls a model to answer anything here.
- An HTTP server. The package is a library and a CLI; the guide shows how an app stores the
  saved JSON and serves it.
- Changing `classify`, the catalogue, the views or the benchmark.
- LLM tool-calling schemas for these functions (a possible follow-up).
- The demo site (a possible follow-up: "show me the balance sheet text" on the demo).
- Replacing the experimental `select()` (element-level selection with a completeness dial).
  Section context is the simple, tag-based counterpart and needs no tagger or selector.

## Anti-goals

- Returning a heading's own text only and missing its subsections or tables.
- Returning the same text twice in one answer because a section and its subsection share a tag.
- Raising for a tag that simply isn't in this document: an app asks for tags speculatively.

## Decisions (all the builder's call, made in this design; the user handed them over)

### D1. One object: `TaggedDocument`

A pydantic model holding the `document` (`Document`) and its `classification`
(`Classification`). It is the thing an application keeps.

```python
from docrooter import TaggedDocument

tagged = TaggedDocument.from_source("10k.pdf", backend="jev")  # load() + classify(), once
tagged.doc_type                       # "financial.annual_report"
tagged.available_tags()               # tags found, in reading order of first appearance
tagged.get_section("financial.balance_sheet")                  # one SectionContext
tagged.get_sections(["financial.balance_sheet", "financial.income_statement"])
tagged.get_all_sections()                                       # every tag found
tagged.save("10k.tagged.json")        # later, anywhere: TaggedDocument.load("10k.tagged.json")
```

- `TaggedDocument(document=doc, classification=result)` wraps a classification the app already
  has. It checks that every section id in the classification exists in the document and raises
  `ValueError` if not (the classification belongs to another reading).
- `from_source(source, **classify_kwargs)` calls `load(source)` then `classify(doc, ...)` and
  passes keyword arguments through to `classify` (backend, flow, level, calls…).
- Persistence: `to_json() -> str`, `from_json(str | bytes)`, `save(path)`, `load(path)`. The JSON
  is `{"format": "docrooter.tagged_document", "version": 1, "document": {...},
  "classification": {...}}`. A wrong format or version raises `ValueError`.
- `doc_type` is a property: `classification.doc_type`.
- An optional `taxonomy` argument (a `Taxonomy` or built-in name) says which catalogue the tags
  come from. By default it is `classification.section_taxonomy` loaded with `builtin_taxonomy`.
  If that can't be resolved (a custom catalogue that wasn't passed), unknown tags can't be told
  from missing ones and both report `not_found`. The taxonomy is not saved in the JSON; a custom
  one is passed again on `load`/`from_json`.

### D2. Results: `SectionContext` and `SectionMatch`

```json
{
  "tag": "financial.balance_sheet",
  "status": "found",
  "message": null,
  "sections": [
    {
      "section_id": "s4.3.2",
      "tag": "financial.balance_sheet",
      "heading": "CONSOLIDATED BALANCE SHEETS",
      "path": ["PART II", "Item 8. Financial Statements and Supplementary Data"],
      "level": 3,
      "text": "| | 2025 | 2024 |\n|---|---|---|\n...",
      "page_start": 45,
      "page_end": 46,
      "confidence": 0.97,
      "confidence_source": "probability",
      "tokens": 812
    }
  ]
}
```

- `SectionContext`: `tag` (as asked), `status` (`"found" | "not_found" | "unknown_tag"`),
  `message` (`None` when found; otherwise a plain sentence such as "No section is tagged
  financial.balance_sheet in this document." or "financial.balence_sheet is not a tag in
  section_catalogue."), `sections` (a list of `SectionMatch`, empty unless found).
- `SectionMatch`:
  - `section_id`, `level`, and `heading` (the section's title);
  - `tag`, the section's own leaf label, which can differ from the asked tag on a group query;
  - `path`, the ancestor titles from the outermost down, without the section itself, and
    without the preamble;
  - `text`, the full context (D3);
  - `page_start` (the section's page) and `page_end`, an upper bound: the page where the next
    section after it starts, else the last page start within it. Both are `None` without pages
    (Markdown input);
  - `confidence` and `confidence_source`, copied from the `SectionTag`;
  - `tokens`: `estimate_tokens(heading + "\n\n" + text)` from `docrooter.views`. It is an
    estimate, for budgeting a prompt.
- Both models use `extra="forbid"`, like the rest of the package.

### D3. What a section's text is

- A section's `text` is its own body (`Section.text`) followed by every descendant in reading
  order. Each descendant contributes its heading as a Markdown heading (`"#" * min(level, 6) +
  " " + title`) and then its body. Parts are joined by a blank line and the result is stripped.
- The section's own heading is not in `text`; it is in `heading`. An app that wants one string
  puts the heading first.
- This is the text from the heading up to the next heading at the same or a higher level, which
  is the tree that `build_document` already builds. Tables come through as they are in the
  section bodies (PyMuPDF4LLM writes them as Markdown tables).

### D4. Which sections match a tag

- A section matches a query tag when its label equals the tag **or the label is under that tag
  in the catalogue** (a group query: `"financial"` matches every `financial.*` leaf).
- Sections with no label (`None`) never match. `none` is a real leaf: asking for `"none"` works,
  but `get_all_sections` leaves it out.
- Inherited tags (headings past the cap) match like any other.
- **Nested matches collapse.** If a match is a descendant of another match for the same query,
  it is dropped: its text is already inside the outer one. Item 1 Business and its subsections
  Overview, Segments, Strategy… all tagged `narrative.business_overview` give one match.
- Siblings with the same tag stay separate matches, in reading order ("Note 1" and "Note 4"
  under the notes are two matches).
- `min_confidence: float | None = None` on `get_section`, `get_sections` and
  `get_all_sections` drops matches whose confidence is below it. A `None` confidence (a per-item
  answer read from a letter) is unknown, not low, and is kept.

### D5. The three calls

- `get_section(tag, *, min_confidence=None) -> SectionContext`.
- `get_sections(tags, *, min_confidence=None) -> list[SectionContext]`, one per tag in the
  order asked. Duplicated tags are answered once each time they are asked (no dedupe).
- `get_all_sections(*, min_confidence=None) -> list[SectionContext]`, one per distinct leaf tag
  present (excluding `none`), in reading order of each tag's first section. Text can repeat
  across contexts here when a tagged section sits inside another tagged section (Item 8 and its
  balance sheet), since each tag's context is complete on its own.
- `available_tags() -> list[str]`: the distinct leaf labels present, excluding `none` and
  `None`, in reading order of first appearance.
- Reading order is `Document.walk()` order, not the order of `classification.sections`
  (tagging order is level 1 first).

### D6. CLI: `docrooter sections`

```bash
docrooter sections 10k.pdf --backend jev --save 10k.tagged.json --list
docrooter sections 10k.tagged.json --tag financial.balance_sheet --tag financial.income_statement
docrooter sections 10k.tagged.json --all --min-confidence 0.5
```

- `SOURCE` is a document (classified now, which costs model calls) or a saved `.json`
  tagged document (no calls).
- `--tag` (repeatable), `--all` or `--list`. `--list` prints the available tags as JSON. With
  none of them, `--list` is the default.
- `--save PATH` writes the tagged document (only meaningful for a document source).
- `--backend`, `--flow` and `--level` pass through to `classify` for a document source, with the
  same defaults as `docrooter classify`.
- Output is JSON on stdout: `{"source", "doc_type", "results": [SectionContext, ...]}` for
  `--tag`/`--all`, `{"source", "doc_type", "tags": [...]}` for `--list`. Exit 0 even when a
  tag is not found; the status is in the JSON.

### D7. Where it lives

- A new module `src/docrooter/context.py` with `TaggedDocument`, `SectionContext` and
  `SectionMatch`, exported from `docrooter` (and added to `__all__` and the public API test).
- Reuse `Document.walk`, `views.estimate_tokens`, `builtin_taxonomy` and the `Taxonomy` node
  lookups; do not copy them.
- Docs:
  - a guide, `docs/site/guides/section-context.md` ("Get a section's text"), in the
    `mkdocs.yml` nav;
  - a short README section;
  - `examples/section_context.py`, which runs offline on a hand-built classification, with a
    test that runs it;
  - a CHANGELOG entry under Unreleased.
- Tests are unit tests with a document built by `Document.from_sections` and a hand-built
  `Classification`. There are no model calls, no network and no torch.

## Left to the builder

- Helper names inside `context.py`, the exact message wording, and how the CLI command is laid
  out in `cli.py`, following the existing commands' style.
- Whether `TaggedDocument` caches the reading-order index (it should be cheap either way).
