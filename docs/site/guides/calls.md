# Batched or one at a time

`classify` asks a model two kinds of question: the document's type (one question per
document) and a tag for every heading (one per heading). `calls=` decides how those
questions are packed into requests:

- **Batched.** Many headings go in one request, and the option list is sent once per request.
  For an LLM that writes its answer as text.
- **One at a time** (`"per_item"`). One question per request, answered with a probability for
  every option. For a decision model, and for an LLM when you need probabilities.

Which to use depends on the kind of model.

## Which to use

| Model | Use | Why |
| --- | --- | --- |
| **Decision model** (`jev`) | One at a time (always) | docrooter asks Jev one question per request and gets a probability for every option back, so Jev is always asked one item at a time; `calls` is ignored. The probabilities are what make its confidences, the rules' thresholds and `cascade` meaningful. |
| **LLM** (`openai:<model>`: Claude, GPT, a local server) | **Batched** (the default) | The option list is sent once per request instead of once per heading, so tagging costs a fraction of the tokens. |
| LLM, when you need probabilities | One at a time | Probabilities come only from a server that returns logprobs (an OpenAI model, a local server; not Anthropic's endpoint); without them the answer is read from its letter, with no probability. Needed for `cascade=True`. Costs far more tokens. |
| Local models (`embedding:<model>`, `keyword`) | Either (no API cost) | They score one item at a time whatever you choose; `calls` only sets which pattern their token *estimate* describes (what an LLM would be sent). |

`calls=None` (the default) picks the right one: batched for a backend that can batch (the
OpenAI-compatible one), one at a time for every other.

## How to set it

```python
from docrooter import classify

batched = classify(path, backend="openai:claude-haiku-5-5")  # the default for an LLM
one_at_a_time = classify(path, backend="openai:claude-haiku-5-5", calls="per_item")
decision_model = classify(path, backend="jev")  # always one at a time
```

The script is [`examples/calls.py`](../../../examples/calls.py). For Claude, set the endpoint and
turn thinking off (see [Jev and OpenAI-compatible LLMs](llm-backends.md)):

```bash
export DOCROOTER_LLM_BASE_URL=https://api.anthropic.com/v1
export DOCROOTER_LLM_EXTRA_BODY='{"thinking": {"type": "disabled"}}'
```

```bash
docrooter classify report.pdf -b openai:claude-haiku-5-5 --calls batched
docrooter classify report.pdf -b openai:claude-haiku-5-5 --calls per_item
```

In the [playground](playground.md) it is the **LLM calls** menu: *Default*, *Batched, two
steps*, *Batched, every option at once* or *One item at a time*.

`result.calls` says how the run was actually asked (a Jev run reports `per_item` whatever you
passed), and `result.tokens.calls` how many requests it made.

## How a batched run asks

For an LLM, a batched `classify` makes a few requests per document:

1. **The type.** One request: the document view and every type, each with its definition. The
   model answers the type's id in JSON.
2. **The groups.** One request lists every heading's view, numbered, and the catalogue's groups
   (financial, disclosure, government, ... and `none`), each with a one-line gloss. The model
   answers a group for each heading.
3. **The tags.** One request per group that received headings, listing only that group's tags.
   A heading in `none` needs no second step.

Each request carries at most 40 headings and, with its instructions and options, at most
`max_call_tokens` (16,000 by default; requests are packed to an estimated 85% of it, the rest
covering the estimate's error). A heading's view too large for any request is cut head-first,
sent alone and flagged `truncated`. A long document simply takes more requests.

An answer that is not one of the options is asked again once, in one request for the failures
only; after that the heading is tagged `none` (or its group's `other` tag) at confidence 0 and
the result is flagged `unparsed`.

`calls="batched_flat"` lists every tag in a single request instead of groups then tags. It is
kept for the benchmark's comparison; the two-step default sends fewer option tokens.

## What batching costs you

- **No probabilities.** A batched answer has confidence 1.0 (0 when unparsed), so thresholds
  cannot tell a sure tag from a guess. (With `flow="sections_first"`, the doc-type rules
  compensate by needing two supporting sections before they decide alone.)
- **No cascade.** `cascade=True` re-asks single low-confidence answers one rung up, so it needs
  `calls="per_item"`; with batched calls it raises `ValueError`.
- **Output tokens.** The model writes JSON for every heading, which an LLM bills.

## What one at a time costs you

For a decision model the option list goes with every question. In the
[classification benchmark](../benchmarks/classification.md), Jev at the `headings` rung asked:

| Per document (113 public documents) | Requests | Input tokens | Cost |
| --- | ---: | ---: | ---: |
| The type | 1 | 1,771 | $0.00007 |
| Every heading's tag (about 168 headings) | 185 | 213,867 | $0.009 |

About 1,150 input tokens per tag question, most of it the options. Jev's input costs
$0.042 per million tokens and its output is free, so a 400-heading annual report is still
about a cent. To ask fewer questions:

- **Lower `max_sections`.** Headings past it are not asked about: each takes its nearest tagged
  parent's tag, with no request. `max_sections=40` asks about the top of the outline and lets
  the rest inherit.
- **Tag only what you need.** Pass your own `catalogue=` with fewer groups and tags: every
  question then carries a shorter list.

For comparison, Claude Haiku 5.5 named the type from the same view with about the same input
(2,344 tokens) for $0.00024, about three times Jev's price; batched, its tags cost far fewer
tokens per heading because the options were sent once per request.
