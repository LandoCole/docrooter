"""OpenAI-compatible backend against an httpx MockTransport (no network)."""

import json
import logging

import numpy as np
import pytest

httpx = pytest.importorskip("httpx")

from docrooter.backends.base import Candidate  # noqa: E402
from docrooter.backends.openai import OpenAIBackend  # noqa: E402
from docrooter.backends.registry import resolve_backend  # noqa: E402
from docrooter.errors import BackendError  # noqa: E402

CANDIDATES = [Candidate("a", "Tax: income taxes"), Candidate("b", "Litigation: lawsuits")]


def completion(content="", top=None):
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
    return {"choices": [choice]}


def backend(handler, **kwargs):
    b = OpenAIBackend("gpt-test", base_url="http://llm.test/v1", api_key="k", **kwargs)
    b._transport = httpx.MockTransport(handler)
    return b


def test_openai_logprob_parsing():
    seen = []

    def handler(request):
        seen.append((request, json.loads(request.content)))
        return httpx.Response(200, json=completion("2", {"2": -0.1, "1": -2.5, "x": -0.2}))

    scores = backend(handler).score(["some text"], CANDIDATES)
    assert scores.kind == "softmax"
    row = scores.values[0]
    expected = np.exp([-2.5, -0.1])
    assert np.allclose(row, expected / expected.sum())
    assert int(row.argmax()) == 1

    request, body = seen[0]
    assert str(request.url) == "http://llm.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer k"
    assert body["logprobs"] is True and body["top_logprobs"] == 20
    assert body["max_tokens"] == 3 and body["temperature"] == 0
    assert body["messages"][0] == {
        "role": "system",
        "content": "Answer with the option number only.",
    }
    user = body["messages"][1]["content"]
    assert "1. Tax: income taxes" in user and "2. Litigation: lawsuits" in user
    assert user.endswith("Answer:")


def test_openai_logprobs_pool_whitespace_variants_and_missing_options():
    def handler(request):
        return httpx.Response(200, json=completion("1", {" 1": -1.0, "1": -1.0}))

    row = backend(handler).score(["t"], CANDIDATES).values[0]
    assert np.allclose(row, [1.0, 0.0])  # option 2 was not listed: -inf


def test_openai_fallback_parse():
    def handler(request):
        return httpx.Response(200, json=completion(" 2."))

    scores = backend(handler).score(["a", "b"], CANDIDATES)
    assert np.array_equal(scores.values, [[0.0, 1.0], [0.0, 1.0]])
    assert scores.kind == "softmax"


def test_openai_garbage_uniform(caplog):
    def handler(request):
        return httpx.Response(200, json=completion("I cannot say"))

    with caplog.at_level(logging.WARNING, logger="docrooter.backends.openai"):
        scores = backend(handler).score(["a", "b", "c"], CANDIDATES)
    assert np.allclose(scores.values, 0.5)
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_openai_out_of_range_number_is_garbage():
    def handler(request):
        return httpx.Response(200, json=completion("7"))

    assert np.allclose(backend(handler).score(["a"], CANDIDATES).values, 0.5)


def test_openai_keeps_text_order_under_concurrency():
    def handler(request):
        user = json.loads(request.content)["messages"][1]["content"]
        return httpx.Response(200, json=completion("2" if "second" in user else "1"))

    scores = backend(handler, concurrency=2).score(
        ["first a", "second b", "first c", "second d"], CANDIDATES
    )
    assert scores.values.argmax(axis=1).tolist() == [0, 1, 0, 1]


def test_openai_http_error_is_backend_error():
    def handler(request):
        return httpx.Response(401, text='{"error": "bad key"}')

    with pytest.raises(BackendError, match="401"):
        backend(handler).score(["a"], CANDIDATES)


def test_openai_too_many_candidates_and_empty():
    b = backend(lambda request: httpx.Response(200, json=completion("1")))
    many = [Candidate(str(i), f"c{i}") for i in range(51)]
    with pytest.raises(BackendError, match="at most 50"):
        b.score(["a"], many)
    assert b.score([], CANDIDATES).values.shape == (0, 2)
    assert b.score(["a"], []).values.shape == (1, 0)


def test_openai_env_defaults_and_missing_key(monkeypatch):
    for name in ("DOCROOTER_LLM_BASE_URL", "DOCROOTER_LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://openai-env.test/v1/")
    monkeypatch.setenv("OPENAI_API_KEY", "from-openai")
    b = OpenAIBackend("m")
    assert b.base_url == "http://openai-env.test/v1" and b._api_key == "from-openai"
    monkeypatch.setenv("DOCROOTER_LLM_BASE_URL", "http://docrooter-env.test/v1")
    monkeypatch.setenv("DOCROOTER_LLM_API_KEY", "from-docrooter")
    b = OpenAIBackend("m")
    assert b.base_url == "http://docrooter-env.test/v1" and b._api_key == "from-docrooter"

    for name in ("DOCROOTER_LLM_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("DOCROOTER_LLM_API_KEY")
    with pytest.raises(BackendError, match="OPENAI_API_KEY"):
        OpenAIBackend("m")
    assert OpenAIBackend("m", base_url="http://localhost:8000/v1")._api_key is None


def test_openai_properties_and_registry(monkeypatch):
    monkeypatch.setenv("DOCROOTER_LLM_API_KEY", "k")
    b = resolve_backend("openai:gpt-4o-mini", concurrency=3)
    assert isinstance(b, OpenAIBackend)
    assert b.name == "openai:gpt-4o-mini" and b.max_candidates == 50
    assert b.temperature == 1.0 and b.concurrency == 3
    assert b.supports("x") and b.tokenizer() is None
    with pytest.raises(BackendError, match="revision"):
        resolve_backend("openai:gpt-4o-mini@abc")


def test_openai_api_key_is_stripped():
    seen = []

    def handler(request):
        seen.append(request.headers["authorization"])
        return httpx.Response(200, json=completion("1"))

    b = OpenAIBackend("m", base_url="http://llm.test/v1", api_key=" k\n")
    b._transport = httpx.MockTransport(handler)
    b.score(["t"], CANDIDATES)
    assert seen == ["Bearer k"]
    assert OpenAIBackend("m", base_url="http://x.test", api_key=" \n")._api_key is None


SECRET = "sk-SECRET-VALUE-123"


def test_openai_transport_error_does_not_leak_key():
    def handler(request):
        raise httpx.LocalProtocolError(f"Illegal header value b'Bearer {SECRET}\\n'")

    b = OpenAIBackend("m", base_url="http://llm.test/v1", api_key=SECRET + "\n")
    b._transport = httpx.MockTransport(handler)
    with pytest.raises(BackendError) as info:
        b.score(["t"], CANDIDATES)
    assert SECRET not in str(info.value)
    assert "LocalProtocolError" in str(info.value)
    assert info.value.__cause__ is None and info.value.__suppress_context__


def test_openai_http_error_body_does_not_leak_key():
    def handler(request):
        return httpx.Response(401, text=f'{{"error": "Incorrect API key provided: {SECRET}"}}')

    b = OpenAIBackend("m", base_url="http://llm.test/v1", api_key=SECRET)
    b._transport = httpx.MockTransport(handler)
    with pytest.raises(BackendError, match="401") as info:
        b.score(["t"], CANDIDATES)
    assert SECRET not in str(info.value)


def test_redact_helper():
    from docrooter.backends._secrets import redact

    assert redact(f"a {SECRET} b {SECRET}", SECRET) == "a *** b ***"
    assert redact(f"Bearer {SECRET}\n", f" {SECRET}\n") == "Bearer ***\n"
    assert redact("plain", None) == "plain" and redact("plain", "") == "plain"
    assert redact(ValueError("x"), "y") == "x"
