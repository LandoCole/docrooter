# Backends

A backend scores a text against candidate nodes: "how well does this section match *Tax*, and
how well *Audit*?". Everything else (views, strategies, thresholds, results) is shared, so you
can swap backends without changing your code.

A backend is chosen by a spec string `kind[:model[@revision]]`, or by passing an object.

| Spec | How it scores | Extra | Runs | Notes |
| --- | --- | --- | --- | --- |
| `keyword` | Word overlap with the label, description and examples | core | locally | For tests and demos only |
| `embedding[:model]` | Embedding similarity | `models` | locally | **The default**, `dmis-lab/UltraBERT-Embed-150M`. Any sentence-embedding model works, e.g. `embedding:BAAI/bge-small-en-v1.5` |
| `nli[:model]` | Textual entailment | `models` | locally | Default `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`; slow |
| `mlm-choice[:model]` | A masked LM picks the letter of the right option | `models` | locally | Experimental; default `dmis-lab/UltraBERT-380M` |
| `trained:<dir>` | Per-node heads on a fine-tuned body | `models` | locally | From `docrooter train` or `train()`; see [Training](../guides/training-evaluation.md) |
| `jev[:model]` | TypeSafe Jev typed choice probabilities | `jev` | hosted API | Default `jev-1.13.0`; needs `TYPESAFE_API_KEY`. The most accurate backend in every docrooter benchmark it ran in |
| `openai:<model>` | A chat model answers with an option letter (A-Z) | `llm` | hosted or local server | Any OpenAI-compatible endpoint; the model is required |

```python
classifier = Classifier(taxonomy, "jev", strategy=Greedy())
```

## Local or hosted?

| | Local models (`embedding`, `trained`) | Hosted (`jev`, `openai`) |
| --- | --- | --- |
| Accuracy (document type, held-out) | 0.825 (UltraBERT zero-shot) | 0.925-0.975 (Jev) |
| Data | Never leaves the machine | Section or document views are sent to the API |
| Cost | Your hardware; about 600 MB download | Per token (Jev: $0.0044 for 51 documents' views in the benchmark) |
| Speed | Batch throughput, CPU or GPU | Network latency; requests run concurrently (16 at a time for Jev, 8 for `openai`) |
| Offline | Yes, after `docrooter models pull` | No |

A common pattern is local by default, hosted where accuracy matters: route only the hard
decisions to a hosted backend with `overrides` (below), or gate on `confidence` and send only
the doubtful documents to Jev. The [FAQ](../faq.md#local-or-hosted) has more on choosing.

## Devices and memory

Model backends pick the device in the order CUDA, MPS (Apple silicon), CPU, or the one set with
`configure(device=...)` or `DOCROOTER_DEVICE`. They stay within a memory budget (8 GiB by
default), batch their inputs, cache embeddings in an LRU of 50,000 texts, and move to the CPU
once if MPS runs out of memory. Every backend has `close()`, and a `Classifier` used in a `with`
block closes the models it loaded. See [Memory and performance](../guides/memory-performance.md).

## Remote code and pinned revisions {#trust_remote_code}

The default model, `dmis-lab/UltraBERT-Embed-150M` (and `UltraBERT-380M`, used by
`mlm-choice`), ships its own modelling code, which transformers runs only with
`trust_remote_code=True`. docrooter turns that on only for models whose built-in recipe needs it
**and** pins a commit:

| Model | Pinned commit |
| --- | --- |
| `dmis-lab/UltraBERT-Embed-150M` | `367d5343b5a0164d57b810cedf5de4b8983d5984` |
| `dmis-lab/UltraBERT-380M` | `65bbbfeefb03bc93b9965d1d511708dabcee8511` |

So the code that runs is the code at that commit, not whatever the repository holds later.

- A revision in the spec (`embedding:dmis-lab/UltraBERT-Embed-150M@<rev>`) replaces the pin for
  the download. It is trusted automatically only if it equals the pin. Any other revision of a
  remote-code model raises `BackendError` until you pass `trust_remote_code=True`. A spec string
  can't carry that, so construct `EmbeddingBackend(model, revision=..., trust_remote_code=True)`
  yourself after reviewing the model's code.
- Any other model that needs remote code fails with an error until you opt in yourself, after
  reading its code: `EmbeddingBackend("org/model", trust_remote_code=True)` from
  `docrooter.backends.embedding`.
- `docrooter models pull` downloads the pinned revision unless you pass `--revision`.

## Routing: different backends for different decisions

The classifier can use different backends at different places in the taxonomy. For a decision
at node `N`, it tries in order:

1. `overrides["node:N"]`;
2. the node's own `backend` setting in the taxonomy file;
3. the first matching depth rule in `overrides`, e.g. `{"depth>=2": "jev"}`
   (`depth>=k`, `depth<=k`, `depth==k`);
4. the primary `backend`;
5. the `fallback`.

The first backend that supports the node and accepts its number of candidates decides. With a
trained primary model and no `fallback`, nodes without a trained head fall back to the model's
zero-shot embedding body.

## Custom backends

Any object with these members can be a backend; there is nothing to subclass:

- `name`: a string that identifies the backend in results and calibrated temperatures.
- `max_candidates`: the most candidates one `score` call accepts, or `None`. A decision with more
  candidates goes to the next backend in the routing order.
- `temperature` (optional, default 1.0): applied to `similarity` scores and used to re-temper
  `softmax` scores; `calibrate()` tunes it.
- `score(texts, candidates, *, instruction=None[, multi_label])`: returns `Scores(values, kind)`
  with `values` of shape `(len(texts), len(candidates))`. `kind` is `"softmax"` (rows are
  probabilities), `"independent"` (one probability per candidate) or `"similarity"` (unbounded).
  Each candidate has `node_id`, `text` and `examples`.
- `supports(node_id)`: whether the backend can decide at this node.
- `tokenizer()`: a token-counting function, or `None` to use an estimate.

```python
from docrooter.backends import Scores


class OverlapBackend:
    name = "overlap"  # keys calibrated temperatures; keep it unique per backend
    max_candidates = None  # or a limit: larger candidate sets go to the next backend

    def score(self, texts, candidates, *, instruction=None):
        rows = []
        for text in texts:
            words = set(text.lower().split())
            rows.append([len(words & set(c.text.lower().split())) for c in candidates])
        return Scores(np.array(rows, dtype=float), "similarity")

    def supports(self, node_id):
        return True

    def tokenizer(self):
        return None  # token counts are estimated


classifier = Classifier(taxonomy, OverlapBackend(), threshold=0.0)
```

The full script is [`examples/custom_backend.py`](../../../examples/custom_backend.py). To use a
custom backend by name in specs (and on the CLI), register a factory with
`docrooter.backends.register_backend_kind("overlap", factory)`.
