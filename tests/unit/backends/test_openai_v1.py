"""v1 OpenAI-compatible backend: letter markers and token usage (MockTransport, no network)."""

import json

import numpy as np
import pytest

httpx = pytest.importorskip("httpx")

from docrooter.backends.base import Candidate, TokenUsage, usage_of  # noqa: E402
from docrooter.backends.fake import KeywordBackend  # noqa: E402
from docrooter.backends.openai import SYSTEM_PROMPT, OpenAIBackend  # noqa: E402
from docrooter.errors import BackendError  # noqa: E402
from docrooter.views import estimate_tokens  # noqa: E402

CANDIDATES = [
    Candidate("a", "Tax: income taxes"),
    Candidate("b", "Litigation: lawsuits"),
    Candidate("c", "Leases: rent and leases"),
]


def completion(content="", top=None, usage=None):
    choice = {"message": {"role": "assistant", "content": content}}
    if top is not None:
        choice["logprobs"] = {
            "content": [
                {
                    "token": max(top, key=top.get),
                    "logprob": max(top.values()),
                    "top_logprobs": [{"token": t, "logprob": lp} for t, lp in top.items()],
                }
            ]
        }
    data = {"choices": [choice]}
    if usage is not None:
        data["usage"] = usage
    return data


def backend(handler, **kwargs):
    b = OpenAIBackend("gpt-test", base_url="http://llm.test/v1", api_key="k", **kwargs)
    b._transport = httpx.MockTransport(handler)
    return b


def test_letter_markers_and_logprob_pooling():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        # " B" and "B" pool; "1" (an old number marker) and "x" are not options; C is unlisted.
        top = {"B": -1.0, " B": -1.0, "A": -2.0, "1": -0.05, "x": -0.1}
        return httpx.Response(200, json=completion("B", top))

    row = backend(handler).score(["some text"], CANDIDATES).values[0]
    b_logit = np.logaddexp(-1.0, -1.0)
    expected = np.exp([-2.0, b_logit])
    assert np.allclose(row, [*(expected / expected.sum()), 0.0])
    assert int(row.argmax()) == 1

    body = seen[0]
    assert body["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "letter" in SYSTEM_PROMPT
    user = body["messages"][1]["content"]
    assert "A. Tax: income taxes" in user and "B. Litigation: lawsuits" in user
    assert "C. Leases: rent and leases" in user
    assert "1. Tax" not in user and user.endswith("Answer:")

    # Without logprobs the first standalone letter of the reply is the answer.
    def text_only(request):
        return httpx.Response(200, json=completion(" (C) Leases"))

    assert np.array_equal(backend(text_only).score(["t"], CANDIDATES).values, [[0, 0, 1]])

    # A letter beyond the offered options names no option: uniform.
    def out_of_range(request):
        return httpx.Response(200, json=completion("D"))

    assert np.allclose(backend(out_of_range).score(["t"], CANDIDATES).values, 1 / 3)


def test_usage_accumulates_from_response():
    def handler(request):
        usage = {"prompt_tokens": 40, "completion_tokens": 1, "total_tokens": 41}
        return httpx.Response(200, json=completion("A", {"A": -0.1}, usage=usage))

    b = backend(handler)
    assert b.usage() == TokenUsage(0, 0, 0, True)
    b.score(["one", "two"], CANDIDATES)
    assert b.usage() == TokenUsage(prompt=80, completion=2, calls=2, exact=True)
    b.score(["three"], CANDIDATES)
    assert b.usage() == TokenUsage(prompt=120, completion=3, calls=3, exact=True)
    assert usage_of(b) == b.usage()

    # Usage is summed and stays exact only while every part is exact.
    assert TokenUsage(1, 2, 1, True) + TokenUsage(3, 4, 2, True) == TokenUsage(4, 6, 3, True)
    assert (TokenUsage(1, 2, 1, True) + TokenUsage(3, 4, 2, False)).exact is False


def test_usage_missing_marks_estimated():
    """Review Focus 3: a server with no ``usage`` and no logprobs (e.g. mlx_lm.server)."""
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=completion("B"))

    b = backend(handler)
    scores = b.score(["a short section"], CANDIDATES)
    assert np.array_equal(scores.values, [[0, 1, 0]])  # parsed from the text
    messages = seen[0]["messages"]
    estimate = sum(estimate_tokens(m["content"]) for m in messages)
    assert b.usage() == TokenUsage(estimate, estimate_tokens("B"), 1, False)

    # One estimated call makes the running total estimated, and the exact counts still add up.
    calls = iter([{"prompt_tokens": 10, "completion_tokens": 1}, None])

    def mixed(request):
        return httpx.Response(200, json=completion("A", usage=next(calls)))

    b = backend(mixed, concurrency=1)
    b.score(["x"], CANDIDATES)
    assert b.usage() == TokenUsage(10, 1, 1, True)
    b.score(["y"], CANDIDATES)
    usage = b.usage()
    assert usage.calls == 2 and usage.exact is False and usage.prompt > 10

    # Partial usage, or a logprobs block in another server's format, is not trusted or fatal.
    def odd(request):
        data = completion("C", usage={"prompt_tokens": 12})
        data["choices"][0]["logprobs"] = {"tokens": ["C"], "top_logprobs": [{"34": -0.1}]}
        return httpx.Response(200, json=data)

    b = backend(odd)
    assert np.array_equal(b.score(["z"], CANDIDATES).values, [[0, 0, 1]])
    assert b.usage().exact is False

    # Backends that are not LLMs report no usage.
    assert usage_of(KeywordBackend()) is None
    assert usage_of(object()) is None


def test_more_than_26_options_raises():
    """Review Focus 5: never truncate silently; name the hierarchical strategy as the fix."""
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(200, json=completion("Z"))

    b = backend(handler)
    assert b.max_candidates == 26
    many = [Candidate(str(i), f"option {i}") for i in range(27)]
    with pytest.raises(BackendError, match="at most 26") as info:
        b.score(["t"], many)
    assert "hierarchical" in str(info.value) and "Greedy" in str(info.value)
    assert sent == []

    row = b.score(["t"], many[:26]).values[0]
    assert row[25] == 1.0
    assert "Z. option 25" in json.loads(sent[0].content)["messages"][1]["content"]
