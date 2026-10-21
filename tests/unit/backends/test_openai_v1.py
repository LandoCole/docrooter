"""v1 OpenAI-compatible backend: letter markers and token usage (MockTransport, no network)."""

import hashlib
import json
import logging

import numpy as np
import pytest

httpx = pytest.importorskip("httpx")

from docrooter.backends.base import Candidate, TokenUsage, usage_of  # noqa: E402
from docrooter.backends.fake import KeywordBackend  # noqa: E402
from docrooter.backends.openai import SYSTEM_PROMPT, OpenAIBackend, prompt_hash  # noqa: E402
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


# --- fix round 1 -------------------------------------------------------------------------------

TEN = [Candidate(str(i), f"option {i}") for i in range(10)]


def test_top_logprobs_is_mlx_safe():
    """mlx_lm.server rejects top_logprobs above 11; every request asks for at most 10."""
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=completion("A"))

    backend(handler).score(["a", "b"], CANDIDATES)
    assert seen and all(0 < body["top_logprobs"] <= 10 for body in seen)


def test_logprob_pooling_normalises_space_markers():
    def reply(top):
        return lambda request: httpx.Response(200, json=completion("x", top))

    # Byte-level BPE "ĠA" is option A: it must win, not be dropped in favour of a bare "B".
    row = backend(reply({"ĠA": -0.01, "B": -6.0})).score(["t"], CANDIDATES).values[0]
    expected = np.exp([-0.01, -6.0])
    assert np.allclose(row, [*(expected / expected.sum()), 0.0])

    # "ĠB", "B", " B" and SentencePiece "▁B" all pool into B.
    top = {"ĠB": -0.5, "B": -2.0, " B": -3.0, "▁B": -4.0, "ĠA": -1.0, "▁C": -2.5}
    row = backend(reply(top)).score(["t"], CANDIDATES).values[0]
    logits = [-1.0, np.logaddexp.reduce([-0.5, -2.0, -3.0, -4.0]), -2.5]
    expected = np.exp(logits)
    assert np.allclose(row, expected / expected.sum())

    # The current mlx_lm.server shape: {"id", "token", "logprob"} entries with raw vocab tokens.
    def mlx(request):
        data = completion("C")
        entries = [
            {"id": 356, "token": "ĠC", "logprob": -0.1},
            {"id": 425, "token": "ĠB", "logprob": -2.4},
        ]
        data["choices"][0]["logprobs"] = {
            "content": [{"id": 356, "token": "ĠC", "logprob": -0.1, "top_logprobs": entries}]
        }
        data["usage"] = {"prompt_tokens": 50, "completion_tokens": 1, "total_tokens": 51}
        return httpx.Response(200, json=data)

    b = backend(mlx)
    row = b.score(["t"], CANDIDATES).values[0]
    expected = np.exp([-2.4, -0.1])
    assert np.allclose(row, [0.0, *(expected / expected.sum())])
    assert b.usage() == TokenUsage(50, 1, 1, True)


def test_text_fallback_accepts_only_a_bare_letter(caplog):
    def answer(text):
        b = backend(lambda request: httpx.Response(200, json=completion(text)))
        return b.score(["t"], TEN).values[0]

    for prose in ("I cannot determine", "I'm", "A contract", "Answer: B", "", "b"):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="docrooter.backends.openai"):
            assert np.allclose(answer(prose), 0.1), prose
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1, prose

    for text, index in (("B.", 1), ("(C)", 2), (" D", 3), ("E) option 4", 4), ("F. option", 5)):
        assert int(answer(text).argmax()) == index and answer(text).max() == 1.0, text
    assert np.allclose(answer("K"), 0.1)  # a letter beyond the ten options


def test_token_usage_subtracts_and_counts_estimated_calls():
    exact = TokenUsage(10, 1, 1, True)
    estimated = TokenUsage(7, 2, 1, False)
    assert exact.estimated_calls == 0 and estimated.estimated_calls == 1
    total = exact + estimated + exact
    assert total == TokenUsage(27, 4, 3, False, estimated_calls=1)
    # A stage measured as after - before is exact when none of its own calls were estimated.
    assert (total + exact) - total == exact
    assert ((total + estimated) - total).exact is False
    assert TokenUsage(500, 0, 0, False).estimated_calls == 1  # an estimate with no LLM call
    with pytest.raises(ValueError):
        TokenUsage(1, 1, 1, True, estimated_calls=1)
    with pytest.raises(ValueError):
        exact - total  # not an earlier snapshot


def test_on_call_records_each_call():
    records = []
    calls = iter([{"prompt_tokens": 30, "completion_tokens": 1}, None])

    def handler(request):
        return httpx.Response(200, json=completion("B", usage=next(calls)))

    b = backend(handler, concurrency=1)
    b.on_call = records.append
    b.score(["first", "second"], CANDIDATES, instruction="Pick one.")

    assert sorted(r["index"] for r in records) == [0, 1]
    first = next(r for r in records if r["index"] == 0)
    messages = b.request_messages("first", [c.text for c in CANDIDATES], "Pick one.")
    assert first["prompt_hash"] == prompt_hash(messages)
    assert (
        first["prompt_hash"]
        == hashlib.sha256(
            json.dumps(messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
    )
    assert first["options"] == [c.text for c in CANDIDATES]
    assert first["model"] == "gpt-test" and first["endpoint_host"] == "llm.test"
    exact_record = next(r for r in records if r["exact"])
    assert exact_record["prompt_tokens"] == 30 and exact_record["completion_tokens"] == 1
    assert sum(not r["exact"] for r in records) == 1
    assert b.secret_values() == ("k",)


def test_too_many_options_for_a_node_warns_and_names_the_fix(caplog):
    from docrooter.classify.engine import pick_backend

    b = backend(lambda request: httpx.Response(200, json=completion("A")))
    fallback = KeywordBackend()
    with caplog.at_level(logging.WARNING, logger="docrooter.classify.engine"):
        assert pick_backend(lambda node: [b], "root", 27) is None
        assert pick_backend(lambda node: [b, fallback], "root", 30) is fallback
        assert pick_backend(lambda node: [b, fallback], "root", 26) is b
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert all("26" in w and "hierarchical" in w and b.name in w for w in warnings)


def test_on_call_says_how_each_answer_was_read():
    replies = {
        "first": completion("B", top={"B": -0.1, "A": -2.5}),
        "second": completion("C"),
        "third": completion("I cannot tell."),
    }

    def handler(request):
        user = json.loads(request.content)["messages"][1]["content"]
        text = user.split("Text: ", 1)[1].split("\n", 1)[0]
        return httpx.Response(200, json=replies[text])

    records = []
    b = backend(handler, concurrency=1)
    b.on_call = records.append
    b.score(["first", "second", "third"], CANDIDATES)
    answers = {r["index"]: r["answer"] for r in records}
    assert answers == {0: "logprobs", 1: "letter", 2: "none"}


def test_timeout_and_concurrency_from_env(monkeypatch):
    """A local stand-in serving one request at a time needs a long timeout and no fan-out."""
    from docrooter.backends.openai import OpenAIBackend

    monkeypatch.setenv("DOCROOTER_LLM_TIMEOUT", "600")
    monkeypatch.setenv("DOCROOTER_LLM_CONCURRENCY", "1")
    backend = OpenAIBackend("m", base_url="http://127.0.0.1:1/v1")
    assert backend.timeout == 600.0
    assert backend.concurrency == 1
    explicit = OpenAIBackend("m", base_url="http://127.0.0.1:1/v1", timeout=5, concurrency=3)
    assert (explicit.timeout, explicit.concurrency) == (5.0, 3)
    monkeypatch.setenv("DOCROOTER_LLM_CONCURRENCY", "zero")
    import pytest

    with pytest.raises(Exception, match="DOCROOTER_LLM_CONCURRENCY"):
        OpenAIBackend("m", base_url="http://127.0.0.1:1/v1")


# --- token calibration: the exact/estimated prompt ratio ----------------------------------


def scaled_usage(factor):
    """A handler whose usage reports ``factor`` times the estimate of the messages sent."""

    def handler(request):
        body = json.loads(request.content)
        estimate = sum(estimate_tokens(m["content"]) for m in body["messages"])
        usage = {"prompt_tokens": round(factor * estimate), "completion_tokens": 1}
        return httpx.Response(200, json=completion('{"1": "a"}', {"A": -0.1}, usage=usage))

    return handler


def test_token_ratio_defaults_to_one_until_an_exact_call():
    from docrooter.backends.base import token_ratio_of

    b = backend(lambda request: httpx.Response(200, json=completion("B")))
    assert b.token_ratio() == 1.0
    b.score(["some words to classify here"], CANDIDATES)  # no usage: estimated, not learned
    assert b.token_ratio() == 1.0
    assert token_ratio_of(KeywordBackend()) == 1.0  # not an LLM: nothing to calibrate
    assert token_ratio_of(object()) == 1.0


def test_token_ratio_learns_from_exact_calls():
    b = backend(scaled_usage(1.6))
    b.chat_json([{"role": "user", "content": " ".join(["figure"] * 400)}])
    assert b.token_ratio() == pytest.approx(1.6, abs=0.01)
    b.score(["word " * 300], CANDIDATES)
    assert b.token_ratio() == pytest.approx(1.6, abs=0.02)


def test_token_ratio_is_clamped():
    high = backend(scaled_usage(5.0))
    high.chat_json([{"role": "user", "content": "many numbers 1 2 3 4 5 6"}])
    assert high.token_ratio() == 3.0
    low = backend(scaled_usage(0.5))
    low.chat_json([{"role": "user", "content": " ".join(["figure"] * 100)}])
    assert low.token_ratio() == 1.0
