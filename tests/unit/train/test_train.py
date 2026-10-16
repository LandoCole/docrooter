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

pytest.importorskip("sklearn.linear_model")

FIXTURES = Path(__file__).parents[2] / "fixtures"
WORDS = ("tax", "audit", "contract", "court")


class FakeEmbedding:
    instances: list["FakeEmbedding"] = []

    def __init__(self, model, *, device=None, batch_size=None, **kw):
        self.name = f"embedding:{model}"
        self.recipe = get_recipe(model, batch_size=batch_size)
        self.prompts = []
        FakeEmbedding.instances.append(self)

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
    assert model.manifest["recipe"] == recipe.model_dump()
    (body,) = FakeEmbedding.instances
    assert body.prompts[0] == "Instruct: Classify this document section\nQuery: "
