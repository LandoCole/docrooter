# Parse a document for extraction

If your application extracts data from documents, doc-rooter can hand it the context it needs:
what kind of document this is, which sections it has, and the text of each section. `parse()`
returns all of that as one JSON response. Your application stores the JSON. Later, the step
that does the extraction reads the text of the section it wants straight from the stored JSON.
There is no second call to doc-rooter and no need to read the document again.

**Who does what.** doc-rooter supplies the context. Your application does the extraction: it
writes the prompt, calls the model and reads the answer. doc-rooter never calls a model to
answer anything here.

## The flow

```text
 q3.pdf ──► parse() ──► JSON, stored ──► your rules ──► extraction engine
            read and     in your          "a quarterly    reads section_text,
            classify     database         report with a   sends it to its model
            once                          schedule of
                                          investments"
```

An upload comes in. `parse()` reads it, finds the sections, tags them and sends back the JSON.
Your application stores that. Your rules then run on it:

1. "This is a quarterly report."
2. "It has a schedule of investments."
3. "Pull every company and its fair value from the schedule of investments."

The extraction engine takes the schedule's text from the stored JSON and sends it to its
model. It does not need doc-rooter, only a JSON parser.

## A first run

```python
from docrooter import ParseResult, parse

result = parse(MARKDOWN, include="text", backend="keyword")  # MARKDOWN: your text; see the example
print(f"document type: {result.doc_type}")
for tag, matches in result.section_text.items():
    for match in matches:
        print(f"{tag}: {match.heading}")
```

```text
document type: financial.earnings_release
financial.financial_highlights: Acme Corp Quarterly Report
financial.balance_sheet: Balance Sheet
financial.comprehensive_income: Income Statement
disclosure.risk_factors: Risk Factors
```

The full script is [`examples/parse_for_extraction.py`](../../../examples/parse_for_extraction.py).
It runs offline on a small Markdown filing, with the `keyword` backend standing in for a
model. That backend only counts shared words, so its tags show the shape of the response, not
how accurate a model is. For a real run, use `backend="jev"` (see
[Jev and OpenAI-compatible LLMs](llm-backends.md)) on a PDF:

```python
parse("q3.pdf", include="text", backend="jev")
```

`source` can be a path, Markdown text or a loaded `Document`, whatever
[`classify`](sections-first.md) takes.

## Three levels

`include` says how much comes back. Ask for no more than you need.

| `include` | What's in the response | Model calls |
|---|---|---|
| `"type"` | the document type | the document-type call only; no section tags (about 98% cheaper on Jev for a 10-K) |
| `"sections"` | the document type, plus every tagged heading with its section tag | the document type plus section tags |
| `"text"` | the document type and the tags, plus the full text of each tagged section, grouped by tag | same as `"sections"`; the text costs no calls |

**What it costs.** Tagging the sections is most of the cost. On Jev, a 10-K's document type
cost $0.00008, against $0.0038 for its section tags. If a rule only needs the document type,
for example to decide whether to keep a file, ask for `"type"`.

`"type"` reads the document first and does not tag any section, so it works only with
`flow="doc_first"`, the default. `flow="sections_first"` decides the document type from the tags, so it
raises `ValueError` with `"type"`.

**Long documents.** At most 400 headings are asked about. A deeper heading takes the tag of
its nearest tagged parent, and its text counts under that tag in `section_text`. A heading
under a parent tagged `none` is left out.

## The JSON

At `"text"`, the response looks like this. A section's text is under its tag in
`section_text`.

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

- `doc_type` and `doc_type_confidence` come from the classification.
- `tokens` is what the model read and wrote at each stage, for the document type and for the
  section tags. It is `0` for a stage that made no call.
- `sections` is the table of contents: every tagged heading in reading order, with the tag
  `none` for a heading that has no topic. Its `path` lists the headings above it, outermost
  first. It is left out at `"type"`.
- `section_text` is the text of each tagged section, by tag, in reading order. `none` is
  left out. It is left out below `"text"`.
- `page_start` and `page_end` are `null` for Markdown, which has no pages. `page_end` is an
  upper bound: the page where the next section starts.
- `tokens` in a section is an estimate of its size, for budgeting a prompt.

Store it with `result.model_dump_json()`. `ParseResult.model_validate_json()` reads it back.

## What the text includes

A section's `text` is its own body, followed by everything under it. Each subsection is
written as a Markdown heading with its body after it, with a blank line between parts. This
keeps the tables and notes under a heading. The section's own heading is in `heading`, not in
`text`.

In the example, the balance sheet's text runs on through its `### Current assets` subsection.

## Nested and repeated text

- **A section inside another section with the same tag is dropped.** Its text is already in
  the outer one, so the model never reads it twice. Two separate sections with the same tag,
  such as "Note 1" and "Note 4", stay as two entries.
- **Across different tags, text repeats.** "Item 8" is tagged `financial.financial_statements`
  and contains the balance sheet, which is also under `financial.balance_sheet`. Each tag's
  entry is complete on its own, so the extraction engine can read the one it wants.
- **Only tagged sections have text.** The text of a section tagged `none` is not returned. The
  application chooses `include` up front; doc-rooter does not return text it was not asked for.

## Look up a tag

You do not need doc-rooter to read the JSON. If you have it installed, `get_section` finds a
tag in a stored response, with no model call:

```python
stored = result.model_dump_json()  # keep this in your database
# Later, in the extraction step: read the JSON back and ask for a tag. No call to doc-rooter.
balance_sheet = ParseResult.model_validate_json(stored).get_section("financial.balance_sheet")
print(balance_sheet.status)
print(balance_sheet.sections[0].text)
```

```text
found
Total assets were 120 and total liabilities were 70.

### Current assets

Cash and receivables.
```

It returns a `SectionContext` with a `status`:

| `status` | Means |
|---|---|
| `found` | the document has the tag; `sections` holds its text |
| `not_found` | the tag is a real one, but no section in this document carries it |
| `unknown_tag` | the tag is not in the section catalogue (a typo, or a tag of your own) |

None of them raises, since an application often asks for tags that a document may not have;
check `status`. `message` explains anything but `found`.

- A group tag such as `"financial"` matches every tag under it. A match inside another match
  is dropped, and matches come in reading order.
- Surrounding spaces are ignored; upper and lower case must match exactly.
- `get_sections(["financial.balance_sheet", "disclosure.risk_factors"])` asks several tags at
  once and answers each in order.
- To use your own catalogue, pass `catalogue=`.
- Below `include="text"` there is no text to look in, so both raise `ValueError` that says to
  parse with `include="text"`.

## From the shell

```bash
docrooter parse q3.pdf --include text -b jev --out q3.json
docrooter parse q3.pdf --include type
```

`--include` takes `type`, `sections` (the default) or `text`. It prints the JSON, or writes it
to `--out`. `-b`, `--flow` and `--level` work as they do for `classify`. A wrong `--include`
is a usage error.

## What it does not do

- It does not extract or answer anything. That is your application's job.
- It does not give you text later that you did not ask for. Choose `include="text"` if you
  might need the text.
- It does not replace [`select()`](select-for-extraction.md), which is experimental and picks
  the parts of a document worth reading for a need. `parse` is the simple route: tags and the
  text under them.
