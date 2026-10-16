"""Jev backend against a fake ``typesafe_sdk`` module (no network, no SDK needed)."""

import asyncio
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from docrooter.backends.base import Candidate
from docrooter.backends.jev import JevBackend
from docrooter.backends.registry import resolve_backend
from docrooter.errors import BackendError

CANDIDATES = [
    Candidate("tax", "Tax: income taxes", ("deferred tax", "effective rate")),
    Candidate("lit", "Litigation: lawsuits"),
]


class FakeSDK(types.ModuleType):
    """Records what the backend sends and answers from ``choice_probs`` / ``noul_values``."""

    def __init__(self, choice_probs=None, noul_values=None):
        super().__init__("typesafe_sdk")
        self.calls = []
        self.client_kwargs = []
        self.choice_probs = choice_probs
        self.noul_values = noul_values
        self.in_flight = 0
        self.max_in_flight = 0
        sdk = self

        class Choice:
            def __init__(self, *, instructions, criteria):
                self.instructions, self.criteria = instructions, criteria

        class Noul:
            def __init__(self, *, instructions):
                self.instructions = instructions

        class AsyncTypeSafeClient:
            def __init__(self, **kwargs):
                sdk.client_kwargs.append(kwargs)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def system_one(self, state, questions, *, model=None):
                sdk.calls.append({"state": state, "questions": questions, "model": model})
                sdk.in_flight += 1
                sdk.max_in_flight = max(sdk.max_in_flight, sdk.in_flight)
                await asyncio.sleep(0.01)
                sdk.in_flight -= 1
                probs = sdk.choice_probs(state) if callable(sdk.choice_probs) else sdk.choice_probs
                choices = {
                    qid: SimpleNamespace(probabilities=dict(probs))
                    for qid, q in questions.items()
                    if isinstance(q, Choice)
                }
                nouls = {
                    qid: SimpleNamespace(noul=sdk.noul_values[qid])
                    for qid in questions
                    if isinstance(questions[qid], Noul) and qid in sdk.noul_values
                }
                return SimpleNamespace(choices=choices, nouls=nouls)

        self.Choice, self.Noul, self.AsyncTypeSafeClient = Choice, Noul, AsyncTypeSafeClient


@pytest.fixture
def install(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")

    def _install(**kwargs):
        fake = FakeSDK(**kwargs)
        monkeypatch.setitem(sys.modules, "typesafe_sdk", fake)
        return fake

    return _install


def test_jev_choice_mapping(install):
    fake = install(choice_probs={"c0": 0.1, "c1": 0.9})
    scores = JevBackend().score(["The deferred tax liability grew."], CANDIDATES)
    assert scores.kind == "softmax"
    assert np.allclose(scores.values, [[0.1, 0.9]])

    (call,) = fake.calls
    assert call["model"] == "jev-1.13.0"
    assert call["state"] == {"section": "The deferred tax liability grew."}
    (question,) = call["questions"].values()
    assert question.instructions == "Which option best describes this document section?"
    assert question.criteria == {
        "c0": "Tax: income taxes Examples: deferred tax; effective rate",
        "c1": "Litigation: lawsuits",
    }
    assert fake.client_kwargs == [{"api_key": "test-key", "timeout": 30}]


def test_jev_custom_instruction_and_missing_keys_renormalise(install):
    fake = install(choice_probs={"c1": 0.6})  # c0 missing -> 0.0, then renormalised
    scores = JevBackend().score(["t"], CANDIDATES, instruction="Pick one.")
    assert np.allclose(scores.values, [[0.0, 1.0]])
    assert next(iter(fake.calls[0]["questions"].values())).instructions == "Pick one."
    install(choice_probs={})  # nothing returned at all -> uniform
    assert np.allclose(JevBackend().score(["t"], CANDIDATES).values, 0.5)


def test_jev_rows_follow_text_order(install):
    install(choice_probs=lambda state: {"c0": 1.0} if state["section"] != "b" else {"c1": 1.0})
    scores = JevBackend(concurrency=2).score(["a", "b", "c", "b"], CANDIDATES)
    assert scores.values.argmax(axis=1).tolist() == [0, 1, 0, 1]


def test_jev_semaphore_limits_concurrency(install):
    fake = install(choice_probs={"c0": 1.0, "c1": 0.0})
    JevBackend(concurrency=2).score([f"t{i}" for i in range(8)], CANDIDATES)
    assert len(fake.calls) == 8 and fake.max_in_flight == 2


def test_jev_multilabel_nouls(install):
    fake = install(noul_values={"n0": 0.9, "n1": 0.2})
    scores = JevBackend().score(["t"], CANDIDATES, multi_label=True)
    assert scores.kind == "independent"
    assert np.allclose(scores.values, [[0.9, 0.2]])
    (call,) = fake.calls
    assert [q.instructions for q in call["questions"].values()] == [
        "Does this option apply to the section: Tax: income taxes?",
        "Does this option apply to the section: Litigation: lawsuits?",
    ]
    assert list(call["questions"]) == ["n0", "n1"]


def test_jev_missing_noul_is_zero(install):
    install(noul_values={"n1": 0.4})
    scores = JevBackend().score(["t"], CANDIDATES, multi_label=True)
    assert np.allclose(scores.values, [[0.0, 0.4]])


def test_jev_missing_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(BackendError, match="TYPESAFE_API_KEY"):
        JevBackend()
    monkeypatch.setenv("TYPESAFE_API_KEY", "   ")
    with pytest.raises(BackendError, match="TYPESAFE_API_KEY"):
        JevBackend()
    assert JevBackend(api_key="explicit").name == "jev:jev-1.13.0"


def test_jev_works_inside_a_running_loop(install):
    fake = install(choice_probs={"c0": 0.3, "c1": 0.7})

    async def caller():
        return JevBackend().score(["t1", "t2"], CANDIDATES)

    scores = asyncio.run(caller())
    assert np.allclose(scores.values, [[0.3, 0.7], [0.3, 0.7]])
    assert len(fake.calls) == 2


def test_jev_sdk_failure_is_backend_error(install, monkeypatch):
    fake = install(choice_probs={})

    async def boom(self, *args, **kwargs):
        raise RuntimeError("429 slow down")

    monkeypatch.setattr(fake.AsyncTypeSafeClient, "system_one", boom)
    with pytest.raises(BackendError, match="429 slow down"):
        JevBackend().score(["t"], CANDIDATES)


def test_jev_limits_and_empty(install):
    fake = install(choice_probs={"c0": 1.0})
    b = JevBackend()
    many = [Candidate(str(i), f"c{i}") for i in range(256)]
    with pytest.raises(BackendError, match="at most 255"):
        b.score(["t"], many)
    assert b.score([], CANDIDATES).values.shape == (0, 2)
    assert b.score(["t"], []).values.shape == (1, 0)
    assert fake.calls == []


def test_jev_missing_sdk(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)  # import raises ImportError
    with pytest.raises(ImportError, match=r"docrooter\[jev\]"):
        JevBackend().score(["t"], CANDIDATES)


def test_jev_properties_and_registry(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    b = resolve_backend("jev:jev-latest")
    assert isinstance(b, JevBackend) and b.name == "jev:jev-latest"
    assert b.max_candidates == 255 and b.temperature == 1.0
    assert b.supports("x") and b.tokenizer() is None
    assert resolve_backend("jev").name == "jev:jev-1.13.0"
    with pytest.raises(BackendError, match="revision"):
        resolve_backend("jev:jev-1.13.0@abc")


def test_jev_sdk_failure_does_not_leak_key(install, monkeypatch):
    fake = install(choice_probs={})
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-SECRET-VALUE-123\n")

    async def boom(self, *args, **kwargs):
        raise RuntimeError("401 bad credentials sk-SECRET-VALUE-123")

    monkeypatch.setattr(fake.AsyncTypeSafeClient, "system_one", boom)
    with pytest.raises(BackendError) as info:
        JevBackend().score(["t"], CANDIDATES)
    assert "sk-SECRET-VALUE-123" not in str(info.value)
    assert "RuntimeError" in str(info.value)
    assert info.value.__cause__ is None


@pytest.mark.parametrize("multi_label", [False, True])
def test_jev_malformed_response_is_backend_error(install, monkeypatch, multi_label):
    fake = install(choice_probs={})

    async def odd(self, state, questions, *, model=None):
        return SimpleNamespace(choices={}, nouls=None)  # no "choice" answer; nouls not a mapping

    monkeypatch.setattr(fake.AsyncTypeSafeClient, "system_one", odd)
    with pytest.raises(BackendError, match="unexpected response shape"):
        JevBackend().score(["t"], CANDIDATES, multi_label=multi_label)

    async def no_attrs(self, state, questions, *, model=None):
        return object()

    monkeypatch.setattr(fake.AsyncTypeSafeClient, "system_one", no_attrs)
    with pytest.raises(BackendError, match="unexpected response shape"):
        JevBackend().score(["t"], CANDIDATES, multi_label=multi_label)
