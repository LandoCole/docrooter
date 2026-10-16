import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import docrooter
from docrooter import Classifier
from docrooter.backends.base import Backend, Candidate
from docrooter.backends.embedding import EmbeddingBackend
from docrooter.backends.recipes import get_recipe
from docrooter.backends.registry import resolve_backend
from docrooter.errors import BackendError
from docrooter.taxonomy import Taxonomy
from docrooter.train.bundle import TrainedModel, head_file

linear_model = pytest.importorskip("sklearn.linear_model")
pytest.importorskip("joblib")

FIXTURES = Path(__file__).parents[2] / "fixtures"
WORDS = ("tax", "audit", "contract", "court")


class FakeBody:
    """Stands in for the embedding body: bag-of-keywords vectors, records the prompt."""

    name = "embedding:fake"

    def __init__(self):
        self.prompts = []
        self.saved_to = None

    def encode(self, texts, *, prompt):
        self.prompts.append(prompt)
        return np.array([[float(w in t.lower()) for w in WORDS] for t in texts])

    def _load(self):
        body = self

        class _Model:
            def save(self, path):
                body.saved_to = path
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "weights.bin").write_text("fake")

        return _Model()

    def tokenizer(self):
        return None


@pytest.fixture
def nested() -> Taxonomy:
    return Taxonomy.load(FIXTURES / "taxonomies" / "nested.json")


def _head(body, texts, labels):
    head = linear_model.LogisticRegression(max_iter=1000)
    head.fit(body.encode(texts, prompt=""), labels)
    return head


def _model(taxonomy, base="BAAI/bge-small-en-v1.5", instruction="Find the topic"):
    body = FakeBody()
    heads = {
        Taxonomy.ROOT_ID: _head(
            body,
            ["tax due", "audit opinion", "contract term", "court ruling"],
            ["financial", "financial", "legal", "legal"],
        ),
        "financial": _head(
            body,
            ["tax due", "tax rate", "audit opinion", "audit firm"],
            ["financial.tax"] * 2 + ["financial.audit"] * 2,
        ),
    }
    recipe = get_recipe(base)
    manifest = {
        "docrooter_version": docrooter.__version__,
        "base_model": base,
        "base_revision": recipe.revision,
        "taxonomy_hash": taxonomy.content_hash(),
        "instruction": instruction,
        "recipe": recipe.model_dump(),
        "heads": {
            node: {"file": head_file(node), "classes": [str(c) for c in h.classes_], "n": 4}
            for node, h in heads.items()
        },
        "node_counts": {},
        "train_metrics": {"triplets": 0, "epochs": 0, "lr": None, "seconds": 0.1},
    }
    body.prompts.clear()
    return TrainedModel(manifest, heads, body), body


def test_protocol_and_supports(nested):
    model, _ = _model(nested)
    assert isinstance(model, Backend)
    assert model.name == "trained:BAAI/bge-small-en-v1.5"
    assert model.max_candidates is None and model.temperature == 1.0
    assert model.supports(Taxonomy.ROOT_ID) and model.supports("financial")
    assert not model.supports("legal") and not model.supports("financial.tax")


def test_score_follows_candidate_order_and_training_prompt(nested):
    model, body = _model(nested)
    texts = ["the audit opinion", "tax due this year"]
    forward = model.score(
        texts, [Candidate("financial.tax", "Tax"), Candidate("financial.audit", "A")]
    )
    backward = model.score(
        texts,
        [Candidate("financial.audit", "A"), Candidate("financial.tax", "Tax")],
        instruction="ignored: the heads were trained on the manifest's instruction",
    )
    assert forward.kind == "softmax"
    assert np.allclose(forward.values.sum(axis=1), 1)
    assert np.allclose(forward.values, backward.values[:, ::-1])
    assert forward.values[0, 1] > 0.5 and forward.values[1, 0] > 0.5  # audit, then tax
    # bge's query prompt has no {instruction} slot; the prompt is the recipe's
    assert set(body.prompts) == {get_recipe("BAAI/bge-small-en-v1.5").query_prompt}


def test_score_uses_manifest_instruction(nested):
    model, body = _model(nested, base="dmis-lab/UltraBERT-Embed-150M", instruction="Route it")
    model.score(["tax"], [Candidate("legal", "L"), Candidate("financial", "F")])
    assert body.prompts == ["Instruct: Route it\nQuery: "]


@pytest.mark.parametrize(
    "ids",
    [
        ["financial.tax"],  # a subset
        ["financial.tax", "financial.audit", "legal"],  # a superset
        ["legal.contracts", "legal.litigation"],  # a node without a head
        ["financial.tax", "financial.tax"],  # duplicates
        [],
    ],
)
def test_score_rejects_candidates_that_differ_from_the_head(nested, ids):
    model, _ = _model(nested)
    with pytest.raises(BackendError):
        model.score(["tax"], [Candidate(i, i) for i in ids])


def test_score_no_texts(nested):
    model, body = _model(nested)
    s = model.score([], [Candidate("legal", "L"), Candidate("financial", "F")])
    assert s.values.shape == (0, 2) and body.prompts == []


def test_save_load_roundtrip(nested, tmp_path):
    model, body = _model(nested)
    model.save(tmp_path / "bundle")
    root = tmp_path / "bundle"
    assert body.saved_to == str(root / "body")
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest == model.manifest
    for node, info in manifest["heads"].items():
        digest = hashlib.sha1(node.encode()).hexdigest()[:12]
        assert info["file"] == f"heads/{digest}.joblib" and (root / info["file"]).is_file()

    loaded = TrainedModel.load(root)
    assert loaded.manifest == model.manifest and loaded.name == model.name
    assert loaded.supports("financial") and not loaded.supports("legal")
    zero_shot = loaded.as_zero_shot()
    assert zero_shot is loaded.as_zero_shot()  # cached
    assert isinstance(zero_shot, EmbeddingBackend)
    assert zero_shot.name == f"embedding:{root / 'body'}"
    assert zero_shot._model is None  # nothing loaded yet
    recipe = get_recipe("BAAI/bge-small-en-v1.5")
    assert zero_shot.recipe.query_prompt == recipe.query_prompt
    assert zero_shot.temperature == recipe.temperature
    assert zero_shot.trust_remote_code is False

    loaded._body = FakeBody()  # score through the saved heads without a real model
    candidates = [Candidate("financial.audit", "A"), Candidate("financial.tax", "T")]
    texts = ["audit opinion", "tax"]
    assert np.allclose(
        loaded.score(texts, candidates).values, model.score(texts, candidates).values
    )


def test_load_remote_code_body_trusts_only_when_the_base_needed_it(nested, tmp_path):
    model, _ = _model(nested, base="dmis-lab/UltraBERT-Embed-150M")
    model.save(tmp_path / "b")
    body = TrainedModel.load(tmp_path / "b").as_zero_shot()
    assert body.trust_remote_code is True
    assert body.recipe.revision is None  # the bundle's body is local, never fetched
    assert body.recipe.query_prompt == "Instruct: {instruction}\nQuery: "


def test_load_follows_the_device_order_unless_one_is_passed(nested, tmp_path, monkeypatch):
    from docrooter.backends.loading import select_device

    model, _ = _model(nested)
    model.save(tmp_path / "b")
    monkeypatch.setenv("DOCROOTER_DEVICE", "from-env")
    loaded = TrainedModel.load(tmp_path / "b")
    assert loaded._body._device is None  # nothing pinned: resolved by select_device() on first use
    assert select_device(loaded._body._device) == "from-env"
    assert loaded._device is None
    assert TrainedModel.load(tmp_path / "b", device="explicit")._body._device == "explicit"


def test_load_errors(tmp_path):
    with pytest.raises(BackendError, match="manifest.json not found"):
        TrainedModel.load(tmp_path)
    (tmp_path / "manifest.json").write_text("{not json")
    with pytest.raises(BackendError, match="cannot read"):
        TrainedModel.load(tmp_path)
    with pytest.raises(BackendError, match="no revisions"):
        TrainedModel.load(tmp_path, revision="abc")


def test_registry_resolves_trained_spec(nested, tmp_path):
    model, _ = _model(nested)
    model.save(tmp_path / "b")
    resolved = resolve_backend(f"trained:{tmp_path / 'b'}")
    assert isinstance(resolved, TrainedModel) and resolved.supports("financial")


def test_classifier_falls_back_to_zero_shot(nested):
    model, _ = _model(nested)
    clf = Classifier(nested, backend=model)
    assert [b.name for b in clf.router("legal")] == [model.name, "embedding:fake"]


def test_classifier_warns_on_taxonomy_hash_mismatch(nested, tmp_path):
    model, _ = _model(nested)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Classifier(nested, backend=model)  # same taxonomy: silent

    data = nested.to_dict()
    data["nodes"][0]["description"] = "changed"
    changed = Taxonomy.from_dict(data)
    with pytest.warns(UserWarning, match="different taxonomy"):
        Classifier(changed, backend=model)
    with pytest.warns(UserWarning, match="different taxonomy"):
        Classifier(changed, backend="keyword", fallback=model)

    model.save(tmp_path / "b")
    clf = Classifier(changed, backend=f"trained:{tmp_path / 'b'}")  # specs resolve lazily
    with pytest.warns(UserWarning, match="different taxonomy"):
        clf.router(Taxonomy.ROOT_ID)[0]


def test_exported_from_package():
    from docrooter.train import trainer

    assert docrooter.TrainedModel is TrainedModel
    assert docrooter.train is trainer.train
