# TypeSafe "Jev" (System One) API notes

Verified facts only. Each section cites the docs page it came from (all under
`https://docs.typesafe.ai`; append `.md` for the raw markdown). Items marked
**[live]** were also confirmed by calling the real API on 2026-10-16.
Docs index: <https://docs.typesafe.ai/llms.txt>.

## 1. Endpoint, auth, SDK

Source: <https://docs.typesafe.ai/api.md>, <https://docs.typesafe.ai/sdk/python.md>,
<https://docs.typesafe.ai/sdk/python/usage/basic.md>, <https://docs.typesafe.ai/sdk/python/api/constants.md>

| Item | Value |
| - | - |
| HTTP endpoint | `POST https://api.typesafe.ai/v1/systemone` **[live via SDK]** |
| Base URL (SDK default) | `https://api.typesafe.ai` (override: `base_url=` or env `TYPESAFE_BASE_URL`) |
| Auth header | `Authorization: Bearer <API_KEY>` (401 `authentication_error` on bad key **[live]**) |
| Content type | `application/json` |
| API key env var (SDK) | `TYPESAFE_API_KEY` (required). Leading/trailing whitespace is stripped; the SDK validates the key at client construction |
| Other env vars | `TYPESAFE_BASE_URL`, `TYPESAFE_DEFAULT_MODEL` (default `jev-latest`), `TYPESAFE_LOG_LEVEL` |
| Python package | `typesafe-sdk` (import name `typesafe_sdk`) |
| Install | `pip install typesafe-sdk` or `uv add typesafe-sdk` |
| Extras | `http2`, `pil`, `zstd` (request-body compression on Python <= 3.13, i.e. `backports-zstd`) |
| Latest version at test time | **0.7.5** (changelog 2026-10-14; initial public release 0.5.7 on 2026-09-14) |
| Min Python | `>=3.10` (from PyPI package metadata `Requires-Python`; not stated in docs) |
| Runtime deps (PyPI metadata) | `anyio`, `httpx2>=2.0.0`, `pydantic>=2.12.0`, `tenacity`, `truststore`, `packaging`, `typing-extensions` |
| Clients | `TypeSafeClient` (sync, context manager `with`), `AsyncTypeSafeClient` (async, `async with`) |
| Other SDKs | JavaScript `@typesafe-ai/sdk` |
| Console (keys, rate limits) | <https://console.typesafe.ai/> |

Note: SDK 0.7.0 switched its (de)serialization library from msgspec to pydantic, so the
SDK now pulls in `pydantic>=2.12`. Check this against docrooter's own pydantic pin.
`pip install typesafe-sdk` on Python 3.12 worked without the `zstd` extra **[live]**
(compression silently needs the extra; docs say bodies >= 1 KiB are zstd-compressed "when the
runtime supports it").

Models (<https://docs.typesafe.ai/models.md>):

| Name | Resolves to | Notes |
| - | - | - |
| `jev-latest` | `jev-1.13.0` | stable alias, SDK default; confirmed **[live]**: response `model` was `jev-1.13.0` |
| `jev-preview` | `jev-1.14.0` | newest (available 2026-10-14 per docs) |
| `jev-1.13.0`, `jev-1.14.0` | pinned IDs | pin these if you tune thresholds; aliases move silently |

`client.models.list()` returned only the two aliases **[live]**. Unknown model gives
`400 {"detail":{"error_type":"api_usage_error","message":"Unknown model: ..."}}` **[live]**.

## 2. Asking a Choice question (Python)

Source: <https://docs.typesafe.ai/primitives/choice.md>, <https://docs.typesafe.ai/sdk/python/api/clients/async.md>,
<https://docs.typesafe.ai/sdk/python/api/types/questions.md>

Exact call (sync). Verified working **[live]**:

```python
from typesafe_sdk import TypeSafeClient, Choice

with TypeSafeClient() as client:  # reads TYPESAFE_API_KEY
    response = client.system_one(
        state="Dear fellow shareholders, this year our operating earnings rose ...",
        questions={
            "doc_type": Choice(
                instructions="What type of document is this?",
                criteria={  # option -> description (or None)
                    "Investor letter": "A letter from company management to shareholders ...",
                    "Legal contract": "A binding agreement between parties ...",
                    "Research paper": "An academic paper presenting methods, results ...",
                    "Technical manual": "Step-by-step instructions or reference docs ...",
                },
            ),
        },
        # model="jev-latest",   # optional; default jev-latest
    )

ans = response.choices["doc_type"]  # or response.answers["doc_type"]
ans.choice  # str: highest-probability option
ans.probabilities  # dict[str, float], sums to 1
ans.confidence  # float 0..1 (see section 5)
response.model  # e.g. "jev-1.13.0"
response.usage  # input_tokens=..., output_tokens=...
response.request_id  # from x-typesafe-request-id header
response.raw_http_response.json()  # full raw JSON
```

Signature (SDK 0.7.5, both clients; async version is awaited):

```python
system_one(state: Content, questions: Mapping[str, Question], *, model: str|None=None,
           assets=None, retry: RetryPolicy|None=None, timeout: float|httpx2.Timeout|None=None,
           extra_headers=None, extra_body=None, response_model: type|None=None) -> SystemOneResponse
Choice(*, instructions: Content|None, criteria: Mapping[str, Content|None])
```

- `state`: str, dict or list (JSON). `instructions` and each criteria value may be a str,
  object, array, or `None`.
- Option names AND descriptions are both sent to the model (choice.md).
- Raw dicts are accepted in place of `Choice(...)`: `{"type": "choice", "instructions": ..., "criteria": {...}}`.
- Client-side, a Choice with empty criteria raises `TypeSafeError` ("at least one choice is required");
  a single-option Choice is accepted by the API **[live]**.
- Add an `other` / `none of the above` option when the list may not cover every input (choice.md).
- Option order bias: `jev-1.13` leans toward earlier options; reorder and re-check for consistency
  (<https://docs.typesafe.ai/model-jaggedness/jev-1.13.md>).

### Response JSON shape (Choice) **[live]**

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "doc_type": {
      "type": "choice",
      "choice": "Investor letter",
      "confidence": 1.0,
      "probabilities": {"Investor letter": 1.0, "Legal contract": 0.0,
                        "Research paper": 0.0, "Technical manual": 0.0}
    }
  },
  "usage": {"input_tokens": 449, "output_tokens": 55}
}
```

## 3. Asking a Noul question (Python) - multi-label use

Source: <https://docs.typesafe.ai/primitives/noul.md>

A Noul is a yes/no question returning P(yes) in [0,1]. There is **no `confidence`** field on Noul answers
(use `abs(2p-1)` if you want one; confidence.md). For multi-label classification, ask one Noul per label
in the same request and threshold each in code (phrase so that high = yes; one condition per Noul).

```python
from typesafe_sdk import TypeSafeClient, Noul, NoulCriteria

with TypeSafeClient() as client:
    response = client.system_one(
        state="...document text...",
        questions={
            "is_investor_letter": Noul(
                instructions="Is this text a letter from management to shareholders?"
            ),
            "has_financials": Noul(
                instructions="Does the text discuss financial results?",
                criteria=NoulCriteria(
                    true="Mentions earnings, revenue or profit figures",
                    false="No financial figures",
                ),  # criteria optional
            ),
        },
    )
p = response.nouls["is_investor_letter"].noul  # float 0..1  (also response.answers[id].noul)
```

Signature: `Noul(*, instructions: Content, criteria: NoulCriteria|None)` where `NoulCriteria(true=..., false=...)`.

Response **[live]** (4 Nouls + 1 Choice in ONE request):

```json
{"model": "jev-1.13.0",
 "answers": {
   "is_investor_letter": {"type": "noul", "noul": 0.84},
   "is_legal_contract":  {"type": "noul", "noul": 0.98},
   "is_research_paper":  {"type": "noul", "noul": 0.02},
   "noul_with_criteria": {"type": "noul", "noul": 0.99},
   "doc_type": {"type": "choice", "choice": "Investor letter", "confidence": 0.3,
                "probabilities": {"Investor letter": 0.53, "Legal contract": 0.15, "Other": 0.32}}},
 "usage": {"input_tokens": 462, "output_tokens": 127}}
```

(That test state intentionally mixed a shareholder sentence and an indemnification clause; the Noul
values are independent, i.e. multi-label, while the Choice splits.)

Also exists: `Score` (ordered levels, 2-10) and `Box` (images) - see api.md; not needed for docrooter.

## 4. Several questions in one request; batching; limits

Source: <https://docs.typesafe.ai/primitives.md>, <https://docs.typesafe.ai/patterns/fan-out.md>,
<https://docs.typesafe.ai/models.md>, <https://docs.typesafe.ai/api.md>

- `questions` is a map `{your_id: Question}`; answers return under the same ids. Ids are never sent to
  the model. Mix Choice/Noul/Score freely. The state is ingested once and all questions are evaluated
  **in parallel and independently** (one answer is not context for another).
- Official guidance: put every question that shares a state in ONE request ("speculative fan-out").
  Cookbook claim (parallel_questions.md): 13 questions batched = ~12x cheaper, ~10x faster than 13 calls.
- If a later question depends on an earlier answer, make a second request (primitives.md).
- **No limit on the number of questions.** The limit is size: request must fit both
  - per-request context (state + all questions): **64k tokens**
  - per-question context (state + any one question): **32k tokens**
  (`jev-1.13.0` and `jev-1.14.0`). Opt-in long context: header `TypeSafe-Danger-Zone-Long-Context: true`
  raises both to 262k tokens; same per-token price but slower and less accurate. Over-length returns
  `400` with `max_tokens_exceeded`.
- **Max options per Choice: 255.** **[live]** 255 options OK (5,566 input tokens, 0.50 s);
  256 options -> `422` `too_long ... Dictionary should have at most 255 items`.
- Score: 2-10 levels. Body limits (api.md): <= 32 MiB (413 over), UTF-8 JSON no BOM, <= 64 nesting levels
  in each of state/instructions/criteria, finite numbers only (else 400). Compression: `gzip`/`x-gzip`/`zstd`,
  one layer only, zstd window <= 8 MiB.
- Accuracy degrades as state gets large with irrelevant content: filter first (jev-1.13.md).
- Jev is English-first; other languages are lower accuracy (models.md).
- Jev is not for arithmetic, counting, date comparison, or text generation; keep those in code (jev-1.13.md).

## 5. Confidence

Source: <https://docs.typesafe.ai/confidence.md>

- Choice and Score answers carry `confidence` in [0,1], derived from `probabilities`; Noul has none.
- Choice formula: `(p_max - 1/n) / (1 - 1/n)` where n = number of options (so it depends on n).
  Only the top probability counts. Docs suggest also trying raw top probability and top/second ratio.
- Suggested three-way gating: high -> act, medium -> confirm/flag, low -> route to human/fallback.
  Thresholds should scale with risk (example uses 0.5 floor, 0.9 for high-stakes). Tune on own data.
- classification_using_confidence.md: 75-way Choice; on 60 filings a 0.9 cutoff split them 50/50
  (confident half 90% right, other half 40%); when unsure, report the parent category instead
  (one level up raised the 40% to 70%). No second call needed.

## 6. Recommended hierarchical / taxonomy classification

Source: <https://docs.typesafe.ai/cookbooks/hierarchical_classification.md>

- Each node of the hierarchy is one `Choice` whose options are that node's direct children. Classify
  level by level from the root to a leaf.
- **Greedy** (take argmax child each level) is cheap but cannot recover from an early mistake. In the
  cookbook it matched 2/4 expected leaves; **beam search with K=3** matched 4/4.
- **Beam search**: keep the best K=3 partial paths; each level, issue one Choice per live path
  concurrently (`ThreadPoolExecutor(max_workers=BEAM_WIDTH)`, so extra exploration adds little latency),
  expand each by its children, score each path by the geometric mean of its edge probabilities
  `prod(p_edge) ** (1/decisions)` (length-normalized so shallow and deep leaves compare fairly), keep top K.
  Only count levels with more than one child as decisions. Use `exp(mean(log p))` with an epsilon (1e-9)
  for very deep trees (>10 levels). `separation = top_score / second_score` is an ambiguity signal
  (near 1x = ambiguous). Max depth 12 in the cookbook.
- Cookbook trick: send opaque option keys (`c0`, `c1`, ...) mapped to the real labels as the
  *descriptions* (`criteria={f"c{i}": label}`) and map back after, to keep long/odd labels out of keys.
  A node with a single child needs no call.
- Cookbook client setup: `TypeSafeClient(retry=RetryPolicy(max_retries=5, backoff_initial=1.0, backoff_max=20.0))`,
  results cached per (state, labels). Used model id `jev-1.12` there (older than current).
- For flat taxonomies <= 255 labels a single Choice with all labels works (75-way tested in the docs).
- Related: because one request can carry many Choices, a document+section TOC pipeline can ask the
  document-level Choice and several section-level Choices in a single call when they share a state.

## 7. Rate limits, timeouts, errors, retries, pricing

Source: <https://docs.typesafe.ai/models.md>, <https://docs.typesafe.ai/api.md>,
<https://docs.typesafe.ai/sdk/python/api/retries.md>, <https://docs.typesafe.ai/sdk/python/api/exceptions.md>,
<https://docs.typesafe.ai/sdk/python/api/constants.md>

- **Rate limits**: per organization, measured in tokens/second AND requests/second; exact numbers are only in the
  console (<https://console.typesafe.ai/rate-limits>) - not published in the docs. Over limit -> `429`.
  Higher limits via sales@typesafe.ai.
- **Pricing**: per **input** token only; output tokens free. Jev 1.13 and 1.14: **$42 per billion input tokens
  ($0.042 per million)**. Long context costs the same per token. (The smoke test's ~450 tokens cost ~$0.00002.)
- **Errors** (JSON body, FastAPI-style `{"detail": ...}` **[live]**):
  `400` request rule violated / `max_tokens_exceeded` / unknown model; `401` bad/missing key; `413` > 32 MiB;
  `415` bad Content-Encoding; `422` validation (e.g. > 255 options; body lists offending field); `429` rate limit;
  `529` overloaded. SDK also maps 403/404 and 5xx.
- **SDK exceptions** (`typesafe_sdk`): `TypeSafeError` (base) > `TypeSafeAPIError` (`.status`, `.body`, `.headers`,
  `.request_id`) > `TypeSafeBadRequestError`(400), `TypeSafeAuthenticationError`(401), `TypeSafePermissionDeniedError`(403),
  `TypeSafeNotFoundError`(404), `TypeSafeUnprocessableEntityError`(422), `TypeSafeRateLimitError`(429, `.retry_after_ms`),
  `TypeSafeInternalServerError`(5xx), `TypeSafeAPIResponseValidationError`. Connection: `TypeSafeAPIConnectionError`,
  `TypeSafeAPITimeoutError`.
- **Retries** (SDK, on by default): `RetryPolicy(max_retries=2, backoff_initial=0.5, backoff_max=5.0, backoff_jitter=0.25,
  http_statuses={408, 429, 500-599}, respect_retry_after=True, api_connection_error=True, api_timeout_error=True,
  timeout=30.0)` where `timeout` is the total retry budget per SDK call. Set on the client (`retry=`) or per call
  (`system_one(..., retry=...)`). Docs: back off exponentially on 429/529; SDK honors `Retry-After` / `retry-after-ms`.
  Invalid API keys raise at client construction, before any request.
- **Timeouts**: SDK default `DEFAULT_TIMEOUT = 10.0` seconds per HTTP operation (override `timeout=` on client or call).
  The raw HTTP API's server-side timeout is not documented.

## 8. Async support and concurrency

Source: <https://docs.typesafe.ai/sdk/python/usage/advanced.md>, <https://docs.typesafe.ai/sdk/python.md>

- `AsyncTypeSafeClient` is first-class (`async with AsyncTypeSafeClient() as c: await c.system_one(...)`).
- **Reuse one client** across requests (avoids TLS handshake per call). Default clients keep up to 100 connections
  open and keep them alive 30 s; if you pass your own `http_client`, set limits yourself
  (`httpx2.Limits(max_connections=..., max_keepalive_connections=..., keepalive_expiry=30)`).
- HTTP/1.1 default (concurrency scales with connections); HTTP/2 (`typesafe-sdk[http2]`) multiplexes over one
  connection - only for low/moderate concurrency.
- High-rate async: `uv add "httpx-aiohttp[httpx2]"` and `AsyncTypeSafeClient(http_client=Httpx2AiohttpClient())`.
- Sync client can be shared across threads, but under heavy load use one client per thread.
- Install the `zstd` extra on Python < 3.14 for request-body compression (bodies >= 1 KiB).
- `TYPESAFE_LOG_LEVEL=debug` logs request/response bodies (not redacted) - do not enable with sensitive documents.
- Data handling: not trained on customer data; ZDR available for enterprise (<https://docs.typesafe.ai/legal.md>).
- Gateways: the SDK can target OpenRouter / Vercel AI Gateway / Pydantic AI Gateway via `base_url` + that gateway's key
  and model id (usage/basic.md).

## Smoke test (2026-10-16)

Environment: macOS, Python 3.12.13, `typesafe-sdk==0.7.5` in a throwaway venv (outside the repo; project
`pyproject.toml` untouched). Key loaded from `.env` inside the script; no key appears in output.
Model `jev-latest` -> response `model: "jev-1.13.0"`. Request: state = a shareholder-letter paragraph; one Choice,
4 options each with a one-line definition (identical to the example in section 2).

Result **[live]**: `choice = "Investor letter"`, `confidence = 1.0`,
`probabilities = {"Investor letter": 1.0, "Legal contract": 0.0, "Research paper": 0.0, "Technical manual": 0.0}`,
`usage = {input_tokens: 449, output_tokens: 55}`. Raw JSON shape is exactly the Choice response in section 2.
Response headers include `x-typesafe-request-id` (`req_...`). No rate-limit headers were observed on a success response.

Latency (client-side wall clock, reused client, same 449-token request):

| Scenario | Result |
| - | - |
| 3 sequential calls | 0.079 s, 0.164 s, 0.142 s (mean 0.128 s) |
| 5 concurrent calls (async, `asyncio.gather`, warmed client) | total wall 0.222 s; per-call 0.133-0.222 s |
| 5 concurrent calls (async, fresh client, cold connections) | total wall 0.455 s; per-call 0.424-0.454 s |
| 1 request, 4 Nouls + 1 Choice, new client (includes TLS setup) | 1.18 s (462 input / 127 output tokens) |
| 1 request, Choice with 255 options (5,566 input tokens) | 0.50 s |

All calls succeeded; no 429/529 was encountered at this volume (about 25 requests total across the probes).
Probe results: 255 options accepted, 256 -> 422; bad key -> 401 `authentication_error`; unknown model -> 400.
Latency numbers are single-run samples from one network location, not benchmarks.
