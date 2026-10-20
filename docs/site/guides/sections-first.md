# Sections first

`docrooter.classify` is the v1 way to classify a document. It tags the document's sections
first, against one shared [section catalogue](../concepts/taxonomies.md), and then lets those
tags decide the document type, or narrow the types a model is asked to choose from. A
document whose sections include a balance sheet, an income statement and a letter to
shareholders is offered the financial types that need them, not all 19.

```python
result = classify(doc, level="terms", backend="keyword")
```

It returns a `Classification`: the type and how it was decided, the section tags, and every
token spent. How much of each section and of the document the backend sees is set by the
[context ladder](../concepts/context-ladder.md) (`level=`, `terms` by default).

## A first run

[`examples/classify_sections_first.py`](../../../examples/classify_sections_first.py) needs no
model: the `keyword` backend only counts shared words. It prints the ladder's token counts,
then classifies a five-section research paper:

```python
result = classify(doc, level="terms", backend="keyword")
flags = ", ".join(result.flags) or "none"
print(f"doc type: {result.doc_type} (decided by {result.decided_by}; flags: {flags})")
for tag in result.sections:
    print(f"  {tag.title}: {tag.label}")
kind = "exact" if result.tokens.exact else "estimated"
print(f"tokens ({kind}): {result.tokens.prompt:,} prompt over {result.tokens.calls} calls")
```

```text
doc type: research.research_paper (decided by fallback; flags: rules_unmet)
  Sparse retrieval for long documents: disclosure.executive_compensation
  Abstract: general.summary
  Related work: research.related_work
  Method: esg.other
  Results: research.results
  References: standard.normative_references
tokens (estimated): 3,996 prompt over 7 calls
```

The keyword backend's tags are weak and its confidences far below the rules' threshold, so no
doc-type rule is met (`rules_unmet`) and the type is chosen among all of them. That is the
honest result of a word-overlap baseline; it shows the API, the flags and the token accounting,
not the accuracy. Its tokens are **estimated**: what the same views and options would cost an
LLM, since the keyword backend makes no LLM call. With an LLM backend the tokens come from the
API's `usage` and are **exact**.

## How the type is decided

1. **Tag the sections.** The level-1 and level-2 sections (at most `max_sections=40`: level 1
   first, then level 2, each in reading order) are tagged against the catalogue at the section
   level. Per-item calls go group first, then leaf, so each call carries few options.
2. **Match the rules.** Every doc-type leaf of `builtin_taxonomy("doc_types")` has a `sections`
   block, written from the type's definition:
    - `requires`: a list of groups of catalogue ids. The document needs at least one section
      from each group, tagged with confidence at least `rule_min_conf` (0.5). An id may be a
      leaf (`financial.balance_sheet`) or a group (`financial`), which matches any of its leaves.
    - `typical`: ids that order the candidates (most matches first). They never gate.
    - `excludes`: ids that remove a candidate, unless that would leave none (then the candidates
      are kept, flagged `excludes_waived`, and never decided by the rules alone).
3. **Decide.**

| Rules give | What happens | `decided_by` | Flags |
| --- | --- | --- | --- |
| one candidate, decided (below) | the type, with no document-level call | `rules` | |
| several candidates | the model chooses among the candidates only | `model` | |
| one undecided candidate | the model chooses among it and the near misses (types missing exactly one required group); a type still alone is offered with its doc-type siblings | `model` | `near_miss`, `low_support` |
| no candidate, some near misses | the model chooses among the near misses (a lone one with its siblings) | `model` | `near_miss` |
| nothing to offer | the model chooses among every type | `fallback` | `rules_unmet` |
| no level-1 or level-2 section | the document step of `doc_first`, at the same document level | `fallback` | `no_sections` |

**When the rules decide alone.** A single candidate is decided by the rules when every tag
supporting it has confidence at least `decide_conf` (0.7), and it gets the lowest of those
confidences. With batched calls every valid tag has confidence 1.0, so that gate cannot tell
strong evidence from weak: a lone candidate is then decided only when at least two tagged
sections support its required groups. With one, it is offered to the model with its doc-type
siblings (`low_support`).

`result.candidates` lists the types offered, in order, and `result.document_calls` has one
record per document-level call (its level, tokens, whether its view was cut and how many types
it offered).

**`flow="doc_first"`** is the other order: classify the type among every doc type from the
document view, then label the sections with that type's own section taxonomy (as
`normalize_sections` does). Both flows are in the benchmark.

## How the LLM is asked

`calls=` says how a backend that can batch (the OpenAI-compatible one) is asked:

- **`"batched"`** (the default for such a backend) lists the options once per call instead of
  once per section. The sections go in batches of up to 40: one call sorts them into catalogue
  groups (each group with a one-line gloss), then one call per group that received sections
  chooses among that group's leaves. The document step is one call among the offered types. A
  reply naming an id that is not an option is asked again once, in one call for the failures
  only, and then tagged with confidence 0 and flagged `unparsed`.
- **`"batched_flat"`** lists every catalogue leaf in one call (kept for the benchmark's
  comparison).
- **`"per_item"`** asks one item at a time and returns probabilities. `cascade=True` (ask the
  low-confidence items again one rung up) needs it.

A backend that cannot batch (keyword, embedding, NLI) always asks per item, but its token
estimate follows the `calls` pattern, since that is what an LLM would be sent.
`result.tokens_split` estimates how much of the prompt was options, content and instructions.

## Tokens and the results log

- `result.tokens` is everything spent: `prompt`, `completion`, `calls` and `exact`.
  `result.stage_tokens` splits it into `sections` and `document`, and each `SectionTag` has its
  own `tokens`. A total is exact only when every call reported `usage`.
- `classify(..., log="decisions.jsonl")` appends one JSON line per decision: the level and view
  hash, the options, the label and probabilities, the model, endpoint host and prompt hash, the
  tokens and a `trainable` flag (always false for Jev). API keys never reach the log. See
  [the LLM guide](llm-backends.md#the-results-log).

## From the shell

```bash
docrooter classify report.md -b keyword                     # tables: type, tags, tokens
docrooter classify report.md -b keyword --json              # the Classification as JSON
docrooter classify report.pdf --level headings --log decisions.jsonl
docrooter classify report.pdf --section-level terms --doc-level lead
docrooter classify report.md --flow doc_first -b keyword
docrooter classify report.md --calls per_item -b keyword     # one item per call, with probabilities
```

The default backend is the local embedding model (`docrooter[models]`). `--calls` takes
`batched`, `batched_flat` or `per_item` (default: batched when the backend can). With `-t TAXONOMY` the command classifies the sections
against that taxonomy instead, as it did before 1.0.

## With a hosted or local LLM

```bash
export DOCROOTER_LLM_BASE_URL=http://127.0.0.1:18080/v1   # or leave unset for the OpenAI API
docrooter classify report.pdf -b openai:gpt-4o-mini --log decisions.jsonl
```

The key comes from `DOCROOTER_LLM_API_KEY` or `OPENAI_API_KEY`. Batched calls name the options
by id; per-item calls letter them A-Z, at most 26 per call. To test for free, serve
Qwen3-4B-Instruct-2507 locally with `mlx_lm.server`: the recipe is in
[A free local stand-in](llm-backends.md#a-free-local-stand-in). It runs the same code path as a
hosted model, and its token counts are exact.

## Your own types or catalogue

`classify(doc, doc_types=..., catalogue=...)` replaces the built-in doc-type taxonomy (with
its `sections` rules) and the section catalogue. A rule that names an id the catalogue lacks
raises `ValueError`. A doc-type leaf's rules are a `sections` object next to its `label` and
`description`, with the keys `requires`, `typical` and `excludes` described above; the built-in
`doc_types.json` has one for every type.

## Limits

- The rules are written from the type definitions, not fitted to a corpus, and only 8 of the 19
  doc types occur in the benchmark's gold split; the other 11 are not validated.
- The benchmark's accuracy and token numbers are pending:
  [Classification benchmark](../benchmarks/classification.md).
