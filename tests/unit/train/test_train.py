"""``train()`` with a fake embedding body: heads, manifest and the no-triplet path, no torch."""

from pathlib import Path

import numpy as np
import pytest

import docrooter
from docrooter.backends import embedding
from docrooter.backends.recipes import get_recipe
from docrooter.taxonomy import Taxonomy
from docrooter.train import Dataset, Example
from docrooter.train.bundle import head_file
from docrooter.train.trainer import train

REAL_EMBEDDING = embedding.EmbeddingBackend

pytest.importorskip("sklearn.linear_model")

FIXTURES = Path(__file__).parents[2] / "fixtures"
WORDS = ("tax", "audit", "contract", "court")


class FakeEmbedding:
    instances: list["FakeEmbedding"] = []

    def __init__(self, model, *, device=None, batch_size=None, **kw):
        self.name = f"embedding:{model}"
        self.model_id = model
        self.recipe = get_recipe(model, batch_size=batch_size, **kw)
        self.prompts = []
        self._device = device
        self._device_in_use = device or "cpu"
        self.closed = 0
        FakeEmbedding.instances.append(self)

    @property
    def batch_size(self):
        return self.recipe.batch_size_for(self._device_in_use)

    def close(self):
        self.closed += 1

    def encode(self, texts, *, prompt):
        self.prompts.append(prompt)
        return np.array([[float(w in t) for w in WORDS] + [1.0] for t in texts])

    def _load(self):
        raise AssertionError("fine-tuning must not run")


@pytest.fixture
def nested() -> Taxonomy:
    return Taxonomy.load(FIXTURES / "taxonomies" / "nested.json")


@pytest.fixture(autouse=True)
def fake_body(monkeypatch):
    FakeEmbedding.instances.clear()
    monkeypatch.setattr(embedding, "EmbeddingBackend", FakeEmbedding)


def _data(spec: dict[tuple[str, ...], list[str]]) -> Dataset:
    return Dataset(
        [Example(text=t, path=list(path)) for path, texts in spec.items() for t in texts]
    )


DATA = {
    ("financial", "financial.tax"): ["tax one", "tax two"],
    ("financial", "financial.audit"): ["audit one", "audit two"],
    ("legal", "legal.contracts"): ["contract one", "contract two"],
    ("legal", "legal.litigation"): ["court one"],
}


def test_train_fits_heads_and_writes_the_manifest(nested):
    model = train(
        nested,
        _data(DATA),
        base_model="BAAI/bge-small-en-v1.5",
        finetune_body=False,
        min_examples_per_child=2,
        instruction="Pick the topic",
    )
    m = model.manifest
    assert set(m) == {
        "docrooter_version",
        "base_model",
        "base_revision",
        "taxonomy_hash",
        "instruction",
        "recipe",
        "heads",
        "node_counts",
        "train_metrics",
    }
    assert m["docrooter_version"] == docrooter.__version__
    assert m["base_model"] == "BAAI/bge-small-en-v1.5" and m["base_revision"] is None
    assert m["taxonomy_hash"] == nested.content_hash()
    assert m["instruction"] == "Pick the topic"
    assert m["heads"] == {
        Taxonomy.ROOT_ID: {
            "file": head_file(Taxonomy.ROOT_ID),
            "classes": ["financial", "legal"],
            "n": 7,
        },
        "financial": {
            "file": head_file("financial"),
            "classes": ["financial.audit", "financial.tax"],
            "n": 4,
        },
    }
    assert m["node_counts"]["legal"] == {"legal.contracts": 2, "legal.litigation": 1}
    assert m["train_metrics"]["triplets"] == 0 and m["train_metrics"]["epochs"] == 0
    assert m["train_metrics"]["lr"] is None and m["train_metrics"]["seconds"] >= 0
    (body,) = FakeEmbedding.instances
    assert set(body.prompts) == {get_recipe("BAAI/bge-small-en-v1.5").query_prompt}
    assert model.as_zero_shot() is body
    assert model.supports("financial") and not model.supports("legal")


def test_train_without_triplets_skips_finetuning(nested):
    # every node has a single child with data: no negatives anywhere
    data = _data({("financial", "financial.tax"): ["tax one", "tax two", "tax three"]})
    with pytest.warns(UserWarning) as record:
        model = train(nested, data, base_model="BAAI/bge-small-en-v1.5", finetune_body=True)
    messages = [str(w.message) for w in record]
    assert any("no triplets" in msg for msg in messages)
    assert any("no heads" in msg for msg in messages)
    assert model.manifest["train_metrics"]["triplets"] == 0
    assert model.manifest["heads"] == {}


def test_train_default_instruction_from_recipe(nested):
    model = train(nested, _data(DATA), finetune_body=False, min_examples_per_child=2)
    recipe = get_recipe("dmis-lab/UltraBERT-Embed-150M")
    assert model.manifest["instruction"] == recipe.default_instruction
    assert model.manifest["recipe"] == {**recipe.model_dump(), "max_seq_len": 512}
    (body,) = FakeEmbedding.instances
    assert body.prompts[0] == "Instruct: Classify this document section\nQuery: "


def test_max_seq_len_is_capped_by_the_recipe(nested):
    model = train(nested, _data(DATA), finetune_body=False, min_examples_per_child=2)
    assert model.manifest["recipe"]["max_seq_len"] == 512  # UltraBERT's recipe allows 1024
    model = train(
        nested, _data(DATA), finetune_body=False, min_examples_per_child=2, max_seq_len=4096
    )
    assert model.manifest["recipe"]["max_seq_len"] == 1024
    with pytest.raises(ValueError, match="max_seq_len"):
        train(nested, _data(DATA), max_seq_len=0)


def test_max_seq_len_applied_to_the_sentence_transformer(nested, monkeypatch):
    """A real EmbeddingBackend over a fake SentenceTransformer: the length is set before
    fine-tuning and holds while the heads' examples are encoded."""
    import sys
    import types

    from docrooter.train import trainer

    monkeypatch.setattr(embedding, "EmbeddingBackend", REAL_EMBEDDING)
    models = []

    class FakeST:
        def __init__(self, model_id, **kwargs):
            self.max_seq_length = None
            self.seen = []
            models.append(self)

        def encode(self, texts, **kwargs):
            self.seen.append(self.max_seq_length)
            return np.array([[float(w in t) for w in WORDS] + [1.0] for t in texts])

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(
        SentenceTransformer=FakeST
    ))  # fmt: skip
    monkeypatch.setenv("DOCROOTER_DEVICE", "cpu")
    finetuned = []

    def fake_finetune(body, triplets, **kwargs):
        model = body._load()
        finetuned.append((model.max_seq_length, body.batch_size))

    monkeypatch.setattr(trainer, "_finetune", fake_finetune)
    model = train(nested, _data(DATA), min_examples_per_child=2, max_seq_len=256)
    (st,) = models
    assert finetuned == [(256, 16)]
    assert st.seen and set(st.seen) == {256}
    assert model.manifest["recipe"]["max_seq_len"] == 256


def test_hf_model_only_when_checkpointing_is_supported():
    from types import SimpleNamespace

    from docrooter.train.trainer import _hf_model

    supported = SimpleNamespace(supports_gradient_checkpointing=True)
    assert _hf_model([SimpleNamespace(auto_model=supported)]) is supported
    unsupported = SimpleNamespace(supports_gradient_checkpointing=False)
    assert _hf_model([SimpleNamespace(auto_model=unsupported)]) is None
    assert _hf_model([SimpleNamespace()]) is None and _hf_model(object()) is None


@pytest.mark.parametrize("flag", [True, False])
def test_gradient_checkpointing_is_passed_to_finetuning(nested, monkeypatch, flag):
    from docrooter.train import trainer

    seen = []
    monkeypatch.setattr(
        trainer, "_finetune", lambda body, triplets, **kw: seen.append(kw["gradient_checkpointing"])
    )
    train(nested, _data(DATA), min_examples_per_child=2, gradient_checkpointing=flag)
    assert seen == [flag]


OOM = "MPS backend out of memory (MPS allocated: 7.9 GB, max allowed: 8.0 GB)"


@pytest.mark.parametrize(
    ("failures", "expected"),
    [
        (0, [("mps", 16)]),
        (1, [("mps", 16), ("mps", 8)]),
        (2, [("mps", 16), ("mps", 8), ("cpu", 8)]),
    ],
)
def test_finetuning_retries_half_batch_then_cpu_on_mps_oom(
    nested, monkeypatch, caplog, failures, expected
):
    from docrooter.train import trainer

    runs = []

    def flaky(body, triplets, **kw):
        if body._device == "cpu":
            body._device_in_use = "cpu"
        runs.append((body._device_in_use, body.batch_size))
        if len(runs) <= failures:
            raise RuntimeError(OOM)

    monkeypatch.setattr(trainer, "_finetune", flaky)
    with caplog.at_level("WARNING"):
        model = train(nested, _data(DATA), min_examples_per_child=2, device="mps")
    assert runs == expected
    (body,) = FakeEmbedding.instances
    assert body.closed == failures  # each retry starts again from the base weights
    metrics = model.manifest["train_metrics"]
    assert (metrics["device"], metrics["batch_size"]) == expected[-1]
    # the bundle keeps the recipe's batch; the half batch only applied to this run
    assert model.manifest["recipe"]["batch_size"] is None
    if failures:
        assert "retrying with batch 8" in caplog.text
    if failures == 2:
        assert "falling back to the CPU" in caplog.text


def test_an_explicit_batch_is_saved_unhalved_after_an_mps_oom(nested, monkeypatch):
    from docrooter.train import trainer

    runs = []

    def flaky(body, triplets, **kw):
        runs.append(body.batch_size)
        if len(runs) == 1:
            raise RuntimeError(OOM)

    monkeypatch.setattr(trainer, "_finetune", flaky)
    model = train(nested, _data(DATA), min_examples_per_child=2, device="mps", batch_size=12)
    assert runs == [12, 6]
    assert model.manifest["train_metrics"]["batch_size"] == 6
    assert model.manifest["recipe"]["batch_size"] == 12


def test_a_fine_tuned_body_is_marked_unrecoverable_until_saved(nested, monkeypatch):
    from docrooter.train import trainer

    monkeypatch.setattr(trainer, "_finetune", lambda body, triplets, **kw: None)
    fine_tuned = train(nested, _data(DATA), min_examples_per_child=2)
    assert "closed before it was saved" in fine_tuned.as_zero_shot()._unrecoverable
    FakeEmbedding.instances.clear()
    heads_only = train(nested, _data(DATA), min_examples_per_child=2, finetune_body=False)
    assert getattr(heads_only.as_zero_shot(), "_unrecoverable", None) is None


def test_finetuning_gives_up_after_the_cpu_attempt(nested, monkeypatch):
    from docrooter.train import trainer

    def always(body, triplets, **kw):
        body._device_in_use = "mps"  # pretend: every attempt hits the OOM
        raise RuntimeError(OOM)

    monkeypatch.setattr(trainer, "_finetune", always)
    with pytest.raises(RuntimeError, match="out of memory"):
        train(nested, _data(DATA), min_examples_per_child=2, device="mps")


def test_other_finetuning_errors_are_not_retried(nested, monkeypatch):
    from docrooter.train import trainer

    calls = []

    def broken(body, triplets, **kw):
        calls.append(1)
        raise RuntimeError("shape mismatch")

    monkeypatch.setattr(trainer, "_finetune", broken)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        train(nested, _data(DATA), min_examples_per_child=2, device="mps")
    assert calls == [1]
