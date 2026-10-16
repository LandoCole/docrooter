import sys
import types

import numpy as np
import pytest

from docrooter.backends.base import Candidate
from docrooter.backends.embedding import EmbeddingBackend
from docrooter.errors import BackendError

ULTRA = "dmis-lab/UltraBERT-Embed-150M"


class FakeSentenceTransformer:
    """Embeds a string as a 3-d bag of 'a', 'b', 'c' counts, records every call."""

    instances: list = []

    def __init__(self, model_id, **kwargs):
        self.model_id = model_id
        self.kwargs = kwargs
        self.calls: list[tuple[str, list[str]]] = []
        self.batch_sizes: list[int] = []
        self.seq_lengths: list[int | None] = []
        self.max_seq_length = None
        self.tokenizer = types.SimpleNamespace(
            encode=lambda text, add_special_tokens=True: text.split()
        )
        FakeSentenceTransformer.instances.append(self)

    def encode(self, texts, prompt="", **kwargs):
        assert kwargs["normalize_embeddings"] is True
        self.calls.append((prompt, list(texts)))
        self.batch_sizes.append(kwargs["batch_size"])
        self.seq_lengths.append(self.max_seq_length)
        out = np.array([[(prompt + t).count(c) for c in "abc"] for t in texts], dtype=float)
        out = out + 0.01
        return out / np.linalg.norm(out, axis=1, keepdims=True)


@pytest.fixture
def fake_st(monkeypatch):
    FakeSentenceTransformer.instances = []
    module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    # an explicit device keeps ``select_device`` from importing the real torch
    monkeypatch.setenv("DOCROOTER_DEVICE", "cpu")
    return FakeSentenceTransformer


def test_construction_does_not_load(fake_st):
    backend = EmbeddingBackend("some/model")
    assert fake_st.instances == []
    assert backend.name == "embedding:some/model"
    assert backend.max_candidates is None and backend.supports("x")
    assert backend.temperature == 0.05


def test_load_arguments(fake_st):
    EmbeddingBackend(ULTRA, device="cpu").encode(["a"], prompt="")
    (model,) = fake_st.instances
    assert model.kwargs["device"] == "cpu"
    assert model.kwargs["revision"] == "367d5343b5a0164d57b810cedf5de4b8983d5984"
    assert model.kwargs["trust_remote_code"] is True
    assert model.kwargs["model_kwargs"] == {"attn_implementation": "sdpa", "dtype": "float32"}
    assert model.max_seq_length == 1024


@pytest.mark.parametrize(
    ("device", "dtype"), [("cpu", "float32"), ("mps", "float32"), ("cuda", None)]
)
def test_float32_enforced_off_cuda(fake_st, device, dtype):
    EmbeddingBackend("some/model", device=device).encode(["a"], prompt="")
    model_kwargs = fake_st.instances[0].kwargs["model_kwargs"]
    assert model_kwargs.get("dtype") == dtype


def test_encode_caches_by_prompt_and_text(fake_st):
    backend = EmbeddingBackend("some/model")
    first = backend.encode(["aa", "bb", "aa"], prompt="P")
    assert first.shape == (3, 3) and np.allclose(np.linalg.norm(first, axis=1), 1)
    backend.encode(["aa", "cc"], prompt="P")
    backend.encode(["aa"], prompt="Q")
    (model,) = fake_st.instances
    assert model.calls == [("P", ["aa", "bb"]), ("P", ["cc"]), ("Q", ["aa"])]


def test_score_is_cosine_similarity_with_label_prompt(fake_st):
    backend = EmbeddingBackend(ULTRA)
    scores = backend.score(["aaa", "bbb"], [Candidate("a", "aaa"), Candidate("b", "bbb")])
    assert scores.kind == "similarity" and scores.values.shape == (2, 2)
    assert scores.values[0, 0] > scores.values[0, 1]
    assert scores.values[1, 1] > scores.values[1, 0]
    assert (np.abs(scores.values) <= 1 + 1e-6).all()
    (model,) = fake_st.instances
    assert ("Instruct: Classify this document section\nQuery: ", ["aaa", "bbb"]) in model.calls
    assert ("", ["Label: aaa"]) in model.calls


def test_instruction_and_examples_shape_vectors(fake_st):
    backend = EmbeddingBackend("some/model")
    plain = backend.score(["aaa"], [Candidate("x", "bbb")], instruction="I")
    with_examples = backend.score(["aaa"], [Candidate("x", "bbb", ("aaa",))], instruction="I")
    assert with_examples.values[0, 0] > plain.values[0, 0]
    (model,) = fake_st.instances
    assert ("I\n", ["aaa"]) in model.calls  # default query prompt "{instruction}\n"


def test_label_vectors_are_cached(fake_st):
    backend = EmbeddingBackend("some/model")
    cands = [Candidate("x", "bbb", ("aaa",))]
    backend.score(["aaa"], cands)
    backend.score(["ccc"], cands)
    (model,) = fake_st.instances
    assert [c for c in model.calls if c[1] == ["bbb"]] == [("", ["bbb"])]
    assert [c for c in model.calls if c[1] == ["aaa"]] == [
        ("Classify this document section\n", ["aaa"])
    ]


def test_empty_inputs_do_not_load(fake_st):
    backend = EmbeddingBackend("some/model")
    assert backend.score([], [Candidate("x", "y")]).values.shape == (0, 1)
    assert backend.score(["t"], []).values.shape == (1, 0)
    assert fake_st.instances == []


def test_tokenizer(fake_st):
    count = EmbeddingBackend("some/model").tokenizer()
    assert count("one two three") == 3


def test_trust_gate(fake_st):
    assert EmbeddingBackend(ULTRA).trust_remote_code is True
    # a caller-chosen revision is not the vouched-for pin
    with pytest.raises(BackendError, match="trust_remote_code=True"):
        EmbeddingBackend(ULTRA, revision="main")
    assert EmbeddingBackend(ULTRA, revision="main", trust_remote_code=True).trust_remote_code
    assert EmbeddingBackend("some/model", revision="main").trust_remote_code is False
    assert EmbeddingBackend(ULTRA, revision="367d5343b5a0164d57b810cedf5de4b8983d5984")


@pytest.mark.parametrize(("device", "batch"), [("cpu", 16), ("mps", 16), ("cuda", 32)])
def test_default_batch_size_follows_the_device(fake_st, device, batch):
    EmbeddingBackend("some/model", device=device).encode(["a"], prompt="")
    assert fake_st.instances[0].batch_sizes == [batch]
    EmbeddingBackend("some/model", device=device, batch_size=4).encode(["a"], prompt="")
    assert fake_st.instances[1].batch_sizes == [4]


def test_cache_is_lru_bounded(fake_st):
    backend = EmbeddingBackend("some/model", cache_size=2)
    backend.encode(["a", "b"], prompt="")
    backend.encode(["a"], prompt="")  # hit: "a" becomes the most recent
    backend.encode(["c"], prompt="")  # evicts "b", the least recent
    assert list(backend._cache) == [("", "a"), ("", "c")]
    backend.encode(["b", "a"], prompt="")
    (model,) = fake_st.instances
    assert model.calls == [("", ["a", "b"]), ("", ["c"]), ("", ["b"])]
    assert EmbeddingBackend("some/model").cache_size == 50_000


def test_batch_larger_than_the_cache_still_returns_every_row(fake_st):
    backend = EmbeddingBackend("some/model", cache_size=1)
    out = backend.encode(["aa", "bb", "cc", "aa"], prompt="")
    assert out.shape == (4, 3) and np.allclose(out[0], out[3])
    assert len(backend._cache) == 1
    unbounded = EmbeddingBackend("some/model", cache_size=0)
    unbounded.encode(["aa"], prompt="")
    assert len(unbounded._cache) == 0
    with pytest.raises(ValueError, match="cache_size"):
        EmbeddingBackend("some/model", cache_size=-1)


def test_close_drops_model_and_caches_then_reloads(fake_st):
    backend = EmbeddingBackend("some/model")
    backend.score(["aaa"], [Candidate("x", "bbb")])
    assert backend._cache and backend._label_cache
    backend.close()
    assert backend._model is None and not backend._cache and not backend._label_cache
    backend.encode(["aaa"], prompt="")
    assert len(fake_st.instances) == 2


OOM = "MPS backend out of memory (MPS allocated: 7.9 GB, max allowed: 8.0 GB)"


def test_mps_oom_while_loading_falls_back_to_cpu(fake_st, monkeypatch, caplog):
    real_init = fake_st.__init__

    def init(self, model_id, **kwargs):
        if kwargs["device"] == "mps":
            raise RuntimeError(OOM)
        real_init(self, model_id, **kwargs)

    monkeypatch.setattr(fake_st, "__init__", init)
    backend = EmbeddingBackend("some/model", device="mps")
    with caplog.at_level("WARNING"):
        backend.encode(["a"], prompt="")
    (model,) = fake_st.instances
    assert model.kwargs["device"] == "cpu" and backend._device_in_use == "cpu"
    assert "falling back to the CPU" in caplog.text


def test_mps_oom_while_encoding_falls_back_to_cpu_once(fake_st, monkeypatch):
    real_encode = fake_st.encode

    def encode(self, texts, prompt="", **kwargs):
        if self.kwargs["device"] == "mps":
            raise RuntimeError(OOM)
        return real_encode(self, texts, prompt, **kwargs)

    monkeypatch.setattr(fake_st, "encode", encode)
    backend = EmbeddingBackend("some/model", device="mps")
    out = backend.encode(["aa", "bb"], prompt="")
    assert out.shape == (2, 3)
    assert [m.kwargs["device"] for m in fake_st.instances] == ["mps", "cpu"]
    assert backend._device == "cpu"


def test_other_errors_while_encoding_are_raised(fake_st, monkeypatch):
    def encode(self, texts, prompt="", **kwargs):
        raise RuntimeError("shape mismatch")

    monkeypatch.setattr(fake_st, "encode", encode)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        EmbeddingBackend("some/model", device="mps").encode(["a"], prompt="")
    assert len(fake_st.instances) == 1
