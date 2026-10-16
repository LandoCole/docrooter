import time
from pathlib import Path

import numpy as np
import pytest

from docrooter import Classifier, Greedy, Taxonomy, TrainedModel, train
from docrooter.backends.base import Candidate
from docrooter.train import Dataset

pytestmark = pytest.mark.models

FIXTURES = Path(__file__).parents[1] / "fixtures"
BASE = "BAAI/bge-small-en-v1.5"
CONTRACT = "The lease obligates the company to minimum annual payments through 2031."


def test_train_save_load_classify(tmp_path):
    tax = Taxonomy.load(FIXTURES / "taxonomies" / "nested.json")
    data = Dataset.load(FIXTURES / "datasets" / "small.jsonl")
    # keep one litigation example, so 'legal' has too little data for a head
    data = Dataset([e for e in data if "court granted" not in e.text])

    started = time.perf_counter()
    model = train(
        tax,
        data,
        base_model=BASE,
        finetune_body=True,
        epochs=1,
        min_examples_per_child=2,
        max_seq_len=128,
    )
    trained_s = time.perf_counter() - started
    assert model.as_zero_shot()._load().max_seq_length == 128
    assert model.manifest["recipe"]["max_seq_len"] == 128
    metrics = model.manifest["train_metrics"]
    assert metrics["triplets"] == 7 + 4 + 2  # root, financial, legal (litigation has no positive)
    assert metrics["epochs"] == 1
    assert set(model.manifest["heads"]) == {Taxonomy.ROOT_ID, "financial"}

    model.save(tmp_path / "bundle")
    loaded = TrainedModel.load(tmp_path / "bundle")
    assert loaded.as_zero_shot()._load().max_seq_length == 128  # the saved recipe's limit
    for node_id in model.manifest["heads"]:
        assert loaded.supports(node_id)
    assert not loaded.supports("legal")

    # the saved body and heads reproduce the in-memory model
    candidates = [Candidate("legal", "Legal"), Candidate("financial", "Financial")]
    texts = [e.text for e in data]
    assert np.allclose(
        loaded.score(texts, candidates).values, model.score(texts, candidates).values, atol=1e-4
    )

    clf = Classifier(tax, backend=loaded, strategy=Greedy(), threshold=0.0)
    (result,) = clf.classify_texts([CONTRACT])
    steps = {step.node_id: step.backend for step in result.best.steps}
    assert result.best.node_ids[0] == "legal"
    assert steps[Taxonomy.ROOT_ID] == loaded.name == f"trained:{BASE}"
    assert steps["legal"] == loaded.as_zero_shot().name
    assert str(tmp_path / "bundle" / "body") in steps["legal"]

    data_dict = tax.to_dict()
    data_dict["nodes"][0]["description"] = "Money matters"
    with pytest.warns(UserWarning, match="different taxonomy"):
        Classifier(Taxonomy.from_dict(data_dict), backend=loaded)

    # closing frees the models; both reload the fine-tuned body from the bundle, not the hub
    before = loaded.score(texts, candidates).values
    loaded.close()
    assert loaded.as_zero_shot()._model is None
    assert np.allclose(loaded.score(texts, candidates).values, before, atol=1e-5)
    loaded.close()
    model.close()
    assert model.as_zero_shot()._model is None
    assert np.allclose(model.score(texts, candidates).values, before, atol=1e-4)
    model.close()
    print(f"\ntrain (incl. model load, 1 epoch, heads): {trained_s:.1f}s; manifest: {metrics}")
