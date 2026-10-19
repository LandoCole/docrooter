"""``train(allow_partial=True)``: a head over the children with enough examples; the default
rule (every child needs them) is unchanged. Fake embedding body, no torch."""

from pathlib import Path

import numpy as np
import pytest

from docrooter.backends import embedding
from docrooter.backends.recipes import get_recipe
from docrooter.taxonomy import Taxonomy
from docrooter.train import Dataset, Example
from docrooter.train.trainer import train, trainable_nodes

pytest.importorskip("sklearn.linear_model")

FIXTURES = Path(__file__).parents[2] / "fixtures"
WORDS = ("tax", "audit", "contract", "court")
BASE = "BAAI/bge-small-en-v1.5"


class FakeEmbedding:
    def __init__(self, model, *, device=None, batch_size=None, **kw):
        self.name = f"embedding:{model}"
        self.model_id = model
        self.recipe = get_recipe(model, batch_size=batch_size, **kw)
        self._device = device
        self._device_in_use = device or "cpu"
        self.encoded: list[str] = []

    @property
    def batch_size(self):
        return self.recipe.batch_size_for(self._device_in_use)

    def close(self):
        pass

    def encode(self, texts, *, prompt):
        self.encoded.extend(texts)
        return np.array([[float(w in t) for w in WORDS] + [1.0] for t in texts])


@pytest.fixture(autouse=True)
def fake_body(monkeypatch):
    monkeypatch.setattr(embedding, "EmbeddingBackend", FakeEmbedding)


@pytest.fixture
def nested() -> Taxonomy:
    return Taxonomy.load(FIXTURES / "taxonomies" / "nested.json")


DATA = Dataset(
    [Example(text=f"tax {i}", path=["financial", "financial.tax"]) for i in range(3)]
    + [Example(text=f"audit {i}", path=["financial", "financial.audit"]) for i in range(3)]
    + [Example(text=f"contract {i}", path=["legal", "legal.contracts"]) for i in range(3)]
    + [Example(text="court 0", path=["legal", "legal.litigation"])]
)


def test_default_rule_is_unchanged(nested):
    model = train(nested, DATA, base_model=BASE, finetune_body=False, min_examples_per_child=2)
    assert set(model.manifest["heads"]) == {Taxonomy.ROOT_ID, "financial"}
    assert all("untrained" not in info for info in model.manifest["heads"].values())
    assert trainable_nodes(nested, DATA, 2) == {
        Taxonomy.ROOT_ID: {"financial": 6, "legal": 4},
        "financial": {"financial.tax": 3, "financial.audit": 3},
    }


def test_partial_head_leaves_out_sparse_children(nested):
    # legal has 3 children here, one of them sparse: a head over the two with examples
    tax = Taxonomy.from_dict(
        {
            "nodes": [
                *nested.to_dict()["nodes"],
                {"id": "legal.ip", "label": "IP", "parent_id": "legal"},
            ]
        }
    )
    data = Dataset(
        list(DATA) + [Example(text=f"ip court {i}", path=["legal", "legal.ip"]) for i in range(3)]
    )
    model = train(
        tax,
        data,
        base_model=BASE,
        finetune_body=False,
        min_examples_per_child=2,
        allow_partial=True,
    )
    legal = model.manifest["heads"]["legal"]
    assert legal["classes"] == ["legal.contracts", "legal.ip"]
    assert legal["untrained"] == ["legal.litigation"]
    assert legal["n"] == 6  # the sparse child's example is not used by the head
    assert "untrained" not in model.manifest["heads"]["financial"]
    assert model.manifest["node_counts"]["legal"]["legal.litigation"] == 1
    # one trained child is not enough for a head
    assert "legal" not in trainable_nodes(nested, DATA, 2, allow_partial=True)
