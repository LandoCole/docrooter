# Jev and OpenAI-compatible LLMs

Two backends call a language model over HTTP instead of running one locally:

- **`jev`**: [TypeSafe](https://typesafe.ai) Jev, a model built for typed decisions. It returns
  a probability for each option, which is what the classifier's thresholds and beams need. It
  was the most accurate backend in every docrooter benchmark it ran in.
- **`openai:<model>`**: any server that speaks the OpenAI chat API (OpenAI itself, or a local
  server such as Ollama, vLLM or llama.cpp). The model answers with an option number.

Both send the rendered [views](../concepts/views.md) (not whole documents) to the server, so
check that this is acceptable for your data before you use them.

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

A chat model answers with one option number, so its "probabilities" are coarse. Prefer `Greedy`
or `Flat` with it, and evaluate it on your data before trusting a threshold: this backend is not
part of the published benchmarks.

## Mixing local and hosted

Send only some decisions to an LLM with `overrides` (for example `{"depth>=2": "jev"}` for the
finer levels), or set `backend` on the taxonomy nodes where the local model is weak. A common
pipeline types every document locally and re-checks only the ones below a confidence you choose
with Jev. See [Backends](../concepts/backends.md#routing-different-backends-for-different-decisions).
