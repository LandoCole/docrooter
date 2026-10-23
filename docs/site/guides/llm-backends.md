# Jev and OpenAI-compatible LLMs

Two backends call a language model over HTTP instead of running one locally:

- **`jev`**: [TypeSafe](https://typesafe.ai) Jev, a model built for typed decisions. It returns
  a probability for each option, which is what the classifier's thresholds and beams need. It
  was the most accurate backend in every docrooter benchmark it ran in.
- **`openai:<model>`**: any server that speaks the OpenAI chat API (OpenAI itself, or a local
  server such as Ollama, vLLM, llama.cpp or `mlx_lm.server`). Per item, the model answers
  with an option letter; `docrooter.classify` asks it in batched calls by default, where it
  answers option ids in JSON (see [Sections first](sections-first.md#how-the-llm-is-asked)).

Both send the rendered [views](../concepts/views.md) (not whole documents) to the server, so
check that this is acceptable for your data before you use them.

How `classify` packs its questions into requests differs between the two: a decision model is
asked one question at a time, an LLM in batches by default. See
[Batched or one at a time](calls.md) for which to use and how to set it.

## Keys stay in the environment

Keys are read only from environment variables (or an explicit `api_key=`), and are redacted from
error messages. docrooter never reads a `.env` file by itself (the playground's `--env-file`
flag is the one exception). Keep keys in a git-ignored `.env` and hand them to the process:

```bash
# .env  (never commit this file)
TYPESAFE_API_KEY=...

uv run --env-file .env python examples/jev_backend.py
# or: set -a; source .env; set +a
```

| Backend | Variables |
| --- | --- |
| `jev` | `TYPESAFE_API_KEY` |
| `openai` | `DOCROOTER_LLM_API_KEY`, else `OPENAI_API_KEY`; the server from `DOCROOTER_LLM_BASE_URL`, else `OPENAI_BASE_URL`, else the OpenAI API |

Never put a key in code, in a taxonomy file or in a committed config.

## Jev

```bash
pip install 'docrooter[jev]'
```

```python
classifier = Classifier(taxonomy, "jev", strategy=Greedy())
for section in classifier.classify(load(MARKDOWN)).sections.values():
    print(f"{section.title}: {' > '.join(section.best.labels)} ({section.best.score:.2f})")
```

`"jev"` pins the model `jev-1.13.0`, so results do not change under you; `"jev:<model>"` picks
another. Requests run concurrently (16 at a time). From the shell:

```bash
docrooter doctype inbox/ -b jev --jsonl types.jsonl
docrooter normalize report.pdf -b jev --doc-type financial.annual_report
```

The script is [`examples/jev_backend.py`](../../../examples/jev_backend.py).

### Why Jev is more accurate

**What was measured.** On document types Jev got 49 of 51 development documents and 37 of 40
held-out documents right, against 39/51 and 33/40 for the default local model (UltraBERT
zero-shot on the same view; the best local configuration tried on dev reached 43/51). On the
development set the gap is significant (exact McNemar test, p = 0.002). On section
normalisation it led by 11-21 points.

**Why, from the error analysis.** A local embedding model scores how *similar* the view is to
each type's description. That gets the group right (UltraBERT picks the right one of the 6
groups for 90% of documents), but siblings inside a group often differ by who published the
document or why: a white paper is "not peer-reviewed", a government report is "issued by an
agency". Those are not similarity features, so the UltraBERT-only systems got 0 of 3 white
papers right (bge-small, alone or mixed with UltraBERT, got 1 of 3). Jev answers a typed choice
question over the view and the written definitions together, so it can apply such distinctions.
Its two development misses were documents whose view itself was misleading (an NDA that opens
with a page of instructions; a slide deck that parses to one section).

**What it costs.** A network call, a per-token price (the benchmark paid $0.0044 for typing 51
documents from their views, and $0.008 for normalising 159 sections), and sending your text to
a hosted service. The local default stays the right choice when data must not leave the
machine or when about 80% accuracy is enough.

## OpenAI-compatible servers

```bash
pip install 'docrooter[llm]'
```

```python
classifier = Classifier(taxonomy, f"openai:{model}", strategy=Greedy())
```

The model name is required (`openai:gpt-4o-mini`). For a local server, set
`DOCROOTER_LLM_BASE_URL` (e.g. `http://localhost:11434/v1`); a server that needs no key works
without one. Requests run concurrently (8 at a time). The script is
[`examples/openai_backend.py`](../../../examples/openai_backend.py).

### Claude

Anthropic's API speaks the OpenAI chat format, so Claude needs no extra code: point the backend
at it and name the model.

```bash
export DOCROOTER_LLM_BASE_URL=https://api.anthropic.com/v1
export DOCROOTER_LLM_API_KEY=...   # your Anthropic key, from a git-ignored .env
export DOCROOTER_LLM_EXTRA_BODY='{"thinking": {"type": "disabled"}}'
docrooter classify report.pdf -b openai:claude-haiku-5-5
```

`DOCROOTER_LLM_EXTRA_BODY` adds fields to every request: with thinking on, Claude spent the small
batched reply budget thinking and returned no answer. The endpoint refuses `temperature` (the
backend notices and stops sending it) and returns no logprobs, so per-item answers are read
from the letter, with no probability. Use batched calls (the default); see
[Batched or one at a time](calls.md).

In a per-item call (a `score` call, or `classify(..., calls="per_item")`) the options are
lettered A-Z and the model answers with one letter. `classify`'s default batched calls list the
options once per call by id and read a JSON reply instead.

- **With logprobs,** the probabilities come from the first token's top 10 alternatives,
  restricted to the option letters.
- **Without logprobs,** a reply that is a bare letter (`B`, `B.`, `(C)`) gets probability 1, so
  the "probabilities" are coarse. Any other reply ("I cannot tell") scores every option equally,
  with a warning. Each call's record says which it was (`answer`: `logprobs`, `letter` or
  `none`), and `docrooter.classify` uses it: a letter answer has confidence `None`
  (`confidence_source="text"`) and is gated like a batched answer, and a reply naming no
  option is flagged `unparsed`, never read as the first option.

Prefer `Greedy` (or `Beam`) with it, and evaluate it on your data before trusting a threshold.

One call offers at most 26 options, and options are never dropped. A `score` call with more
raises an error. Inside a `Classifier`, a node with more than 26 children goes to the next
routed backend (the `fallback`) with a warning, or fails when there is none. A hierarchical
strategy over a taxonomy whose nodes have at most 26 children keeps every decision on the LLM.

### Token usage

The backend counts the tokens of every call, from the `usage` the server returns, so the counts
are exact. A server that returns no `usage` gets an estimate instead (about 1.3 tokens per word
of the prompt), and the counts are then marked as estimated, never as exact. Each call can also
be reported as it happens, with its token counts and a hash of the exact prompt sent; the
`classify` flow uses this to attribute tokens to each section and document.

**Calibration.** The estimate undercounts dense text: against a real Qwen3-4B tokenizer, prompts
full of tables and numbers took far more tokens than 1.3 per word. So the backend keeps a running
ratio of the exact `prompt_tokens` to the estimate of the same messages, over every call that
reported `usage` (weighted by tokens), and exposes it as `backend.token_ratio()`: 1.0 until the
first exact call, and kept between 1.0 and 3.0. `classify` divides its call budgets by it, so
the calls after the first fit `max_call_tokens` in exact tokens; the first call of a run may
still overshoot. Backends that are not LLMs return 1.0, and each call record keeps the ratio it
was budgeted with (`CallRecord.token_ratio`).

### A free local stand-in

Hosted LLMs are what `docrooter.classify` is built for; a local model is a free stand-in for
testing. To develop against the same code path as a hosted model without paying for one, serve
the pinned Qwen3-4B-Instruct-2507 (4-bit, Apache-2.0) locally with
[`mlx_lm.server`](https://github.com/ml-explore/mlx-lm) (Apple silicon). Download it once, at the
pinned revision, and serve the snapshot directory from the Hugging Face cache, so the server
never fetches anything and the model name docrooter sends is the model that is loaded:

```bash
pip install mlx-lm
hf download mlx-community/Qwen3-4B-Instruct-2507-4bit --revision 50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b
MODEL=~/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b
mlx_lm.server --model "$MODEL" --port 18080 --decode-concurrency 1 --prompt-concurrency 1 --prompt-cache-size 2
```

(`hf download` prints the snapshot path; it differs when `HF_HOME` is set.) Then, in another
shell with the same `MODEL`, point docrooter at it (it needs no key):

```bash
export DOCROOTER_LLM_BASE_URL=http://127.0.0.1:18080/v1
export DOCROOTER_LLM_CONCURRENCY=1 DOCROOTER_LLM_TIMEOUT=600
docrooter classify report.pdf -b "openai:$MODEL" --log decisions.jsonl
uv run python examples/openai_backend.py "$MODEL"
```

The classification benchmark runs its LLM rows this way and labels them as a local stand-in,
not a hosted model.

Current `mlx_lm.server` releases return `usage` and OpenAI-style logprobs, so the counts are
exact. Their logprob tokens are raw vocabulary entries such as `ĠB`, which docrooter reads as
`B`. docrooter asks for 10 alternatives, within the server's limit of 11. An older or different
server that returns neither still works: docrooter reads the letter from the reply and marks
the token counts as estimated. The 4-bit model needs about 3 GB of memory; run one model
process at a time on a small machine.

On an 8 GB machine, serve one request at a time. With the server's default concurrency, a
few `lead` or `full` prompts processed together exhausted GPU memory on an M1 (Metal
"Insufficient Memory"), and queued requests timed out. The flags above, plus
`DOCROOTER_LLM_CONCURRENCY=1` and a long `DOCROOTER_LLM_TIMEOUT` (seconds), keep one prompt in
flight. Both variables set the OpenAI-compatible backend's defaults: `concurrency` (8) and
`timeout` (60 s).

### The results log

Each LLM decision can be logged as one JSON line: the input level and view hash, the options,
the chosen label and probabilities, the model, endpoint host and prompt hash, the tokens and a
`trainable` flag. API keys never reach the log: URLs are cut to their host, fields named like
secrets are masked, and the backend's own key is scrubbed wherever it appears. Whether a row may train another model is for
the provider's terms to decide: rows from Jev are always written with `trainable: false`
(TypeSafe's terms forbid training models on its outputs), and you should check your provider's
terms before training on any other rows.

## Mixing local and hosted

Send only some decisions to an LLM with `overrides` (for example `{"depth>=2": "jev"}` for the
finer levels), or set `backend` on the taxonomy nodes where the local model is weak. A common
pipeline types every document locally and re-checks only the ones below a confidence you choose
with Jev. See [Backends](../concepts/backends.md#routing-different-backends-for-different-decisions).
