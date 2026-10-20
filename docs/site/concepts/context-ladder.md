# The context ladder

A hosted LLM charges for every token it reads, and most of a document is not needed to say
what the document is. The **context ladder** names four levels of context, from the headings
alone to the whole text, so you choose how much a model sees and the benchmark can say what
each extra rung buys.

| Level | A section shows | The document shows |
| --- | --- | --- |
| `headings` | its heading path and title | the title, counts and contents (headings) |
| `terms` | + its key phrases | + key phrases pooled over the sections |
| `lead` | + its first sentences | + the opening body text |
| `full` | its whole text, budgeted | the whole body text, budgeted to the prompt cap |

Each rung shows everything the rung below shows and adds one field, with a budget at least as
large, so a higher rung never loses anything a lower one showed. `terms` is the default of
`docrooter.classify` and `docrooter classify`, until the classification benchmark measures the
cheapest rung within 2 points of `full` for each backend (results pending; see
[the benchmark](../benchmarks/classification.md)). The older entry points
(`classify_document`, `Classifier.classify_documents`, `normalize_sections`) default to
`level=None`, their views from before 1.0 (see below).

## The presets

Each level is an exact preset over the existing [views](views.md). Budgets are view budgets in
estimated tokens (1.3 per word) unless the backend supplies a tokenizer.

**Sections** (`docrooter.ladder.section_view(level)`, a `SectionView`):

| Level | Fields, in order | `max_tokens` | Other settings |
| --- | --- | ---: | --- |
| `headings` | path, title | 32 | |
| `terms` | path, title, keyphrases | 64 | 8 key phrases |
| `lead` | path, title, keyphrases, lead | 192 | 8 key phrases, 3 lead sentences |
| `full` | path, title, keyphrases, text | 2,048 | 8 key phrases |

The section `text` starts with the section's lead, so `full` contains `lead`. A section with no
title shows its lead at every rung, also at `headings` and `terms` (within the rung's budget),
so it is never sent as a bare heading path.

**Documents** (`docrooter.ladder.document_view(level)`, a `DocumentView`). Every rung shows the
contents to depth 2, at most 40 entries:

| Level | Fields, in order | `max_tokens` | Other settings |
| --- | --- | ---: | --- |
| `headings` | title, stats, toc | 320 | |
| `terms` | title, stats, keyphrases, toc | 448 | 12 pooled key phrases |
| `lead` | title, stats, keyphrases, lead, toc | 640 | 12 pooled key phrases, 120 lead words |
| `full` | title, stats, keyphrases, toc, text | 12,576 | 12 pooled key phrases, 120 lead words |

The contents can be long, so the fields a rung adds go before them: the contents take what is
left, which is never less than the rung below left them.

**`full` and the prompt cap.** The document `text` is the whole body text, each word once: the
lead first (120 words, skipping cover and notice boilerplate), then the rest in reading order.
It is cut head-first, and the result records the cut (`doc_truncated`, flag `truncated`).
`full_cap` (16,000 by default) is the target size of the whole prompt, checked against the
API's `prompt_tokens`. The view gets `full_cap` less a reserve for the instruction, the options
and the error of the estimate: 16,000 - 1,024 - 15% of 16,000 = 12,576. Only the words a budget
could ever hold are gathered, so a long document is never counted whole. The 16,000 tokens are
a target, not a guarantee: the benchmark measures the share of calls that still exceed it.

**Every call has the same cap.** In `docrooter.classify`, `max_call_tokens` (16,000 by default,
the same cap) bounds every call's prompt, not only the document's: batched section calls are
packed to an estimated 85% of it (at most 40 sections each), a section view too large for any
call is cut head-first and sent alone (flag `truncated`), and the document `full` view is
budgeted from it as from `full_cap`.

## Seeing the rungs

[`examples/classify_sections_first.py`](../../../examples/classify_sections_first.py) renders
every section and the document at each level of a five-section paper and adds up the tokens:

```python
for level in LEVELS:
    rendered = section_view(level).render_all(doc, [s.id for s in doc.walk()])
    section_tokens = sum(view.n_tokens for view in rendered.values())
    doc_tokens = document_view(level).render(doc).n_tokens
    print(f"{level:<9} sections {section_tokens:>3} tokens, document {doc_tokens:>3} tokens")
```

```text
headings  sections  84 tokens, document  36 tokens
terms     sections 134 tokens, document  62 tokens
lead      sections 211 tokens, document 133 tokens
full      sections 220 tokens, document 133 tokens
```

On a document this small `lead` already shows nearly everything; on a real report the gap
between `lead` and `full` is what the ladder exists for.

## Where `level=` is accepted

- **`docrooter.classify(doc, level=...)`** sets both steps; `section_level=` and `doc_level=`
  set one each. With `cascade=True`, an item whose confidence is below `threshold` is asked
  again one rung up, until `full`, and every attempt's tokens count (per-item calls only).
- **`classify_document(doc, taxonomy, level=...)`**, **`Classifier.classify_documents(docs,
  level=...)`** and **`normalize_sections(doc, level=..., doc_level=...)`** take a rung instead
  of a `view=`.
  `level=None`, their default, keeps the views they used before 1.0; passing both `level=` and
  `view=` raises `ValueError`.
- **The command line**: `docrooter classify FILE --level terms`, or `--section-level` and
  `--doc-level`.

See [Sections first](../guides/sections-first.md) for how the levels are used to classify.
