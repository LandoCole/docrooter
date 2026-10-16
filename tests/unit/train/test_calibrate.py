import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from docrooter import (
    Beam,
    Classifier,
    Dataset,
    Flat,
    Greedy,
    SectionView,
    Taxonomy,
    TaxonomyNode,
    calibrate,
)
from docrooter.backends import KeywordBackend
from docrooter.backends.base import Scores
from docrooter.backends.recipes import get_recipe
from docrooter.train.bundle import TrainedModel
from docrooter.train.calibrate import mean_nll

FIXTURES = Path(__file__).parents[2] / "fixtures"


@pytest.fixture
def nested() -> Taxonomy:
    return Taxonomy.load(FIXTURES / "taxonomies" / "nested.json")


@pytest.fixture
def small(nested) -> Dataset:
    return Dataset.load(FIXTURES / "datasets" / "small.jsonl")


# --- calibration ------------------------------------------------------------------------


def test_calibrate_reduces_nll(nested, small):
    clf = Classifier(nested, backend=KeywordBackend(temperature=10.0), threshold=0.0)
    clf2 = calibrate(clf, small)
    assert clf2 is not clf
    assert clf.settings.temperatures == {}  # the original is untouched
    assert clf2.settings.temperatures["keyword"] < 10
    before, after = mean_nll(clf, small)["keyword"], mean_nll(clf2, small)["keyword"]
    assert after < before
    # the copy classifies like a classifier configured with that temperature from the start
    texts = [e.text for e in small]
    by_hand = Classifier(
        nested,
        backend=KeywordBackend(temperature=clf2.settings.temperatures["keyword"]),
        threshold=0.0,
    )
    assert [r.best.node_ids for r in clf2.classify_texts(texts)] == [
        r.best.node_ids for r in by_hand.classify_texts(texts)
    ]
    report = clf2.calibration_report["keyword"]
    assert report["n"] == 16 and report["temperature"] == clf2.settings.temperatures["keyword"]
    assert report["nll_after"] < report["nll_before"]


def test_calibrate_grid_is_respected(nested, small):
    clf = Classifier(nested, backend=KeywordBackend(temperature=10.0))
    clf2 = calibrate(clf, small, grid=[0.5, 2.0])
    assert clf2.settings.temperatures["keyword"] in (0.5, 2.0)


def test_calibrate_retempers_softmax_backends(nested, small):
    class Overconfident(KeywordBackend):
        """Always 99% sure of the first candidate, which is right for only half the decisions."""

        def score(self, texts, candidates, **kw):
            rest = 0.01 / (len(candidates) - 1)
            row = [0.99] + [rest] * (len(candidates) - 1)
            return Scores(np.tile(row, (len(texts), 1)), "softmax")

    clf = Classifier(nested, backend=Overconfident(name="sharp", temperature=1.0))
    clf2 = calibrate(clf, small)
    assert clf2.settings.temperatures["sharp"] > 1.0
    assert mean_nll(clf2, small)["sharp"] < mean_nll(clf, small)["sharp"]


def test_calibrate_skips_backends_with_few_decisions(nested):
    tiny = Dataset.load(FIXTURES / "datasets" / "small.jsonl")
    tiny = Dataset(tiny.examples[:2])  # 2 root decisions + 2 under financial = 4
    clf = Classifier(
        nested, backend=KeywordBackend(temperature=10.0), temperatures={"keyword": 7.0}
    )
    clf2 = calibrate(clf, tiny)
    assert clf2.settings.temperatures == {"keyword": 7.0}
    assert "skipped" in clf2.calibration_report["keyword"]
    assert clf2.calibration_report["keyword"]["n"] == 4


def test_calibrate_uses_the_routed_backend_per_node(nested, small):
    legal = KeywordBackend(name="legal-kw", temperature=10.0)
    clf = Classifier(
        nested,
        backend=KeywordBackend(temperature=10.0),
        overrides={"node:legal": legal},
        threshold=0.0,
    )
    clf2 = calibrate(clf, small)
    # root and financial decisions (12) go to "keyword"; the legal one (4) is below the minimum
    assert clf2.calibration_report["keyword"]["n"] == 12
    assert clf2.calibration_report["legal-kw"]["n"] == 4
    assert "skipped" in clf2.calibration_report["legal-kw"]
    assert "legal-kw" not in clf2.settings.temperatures


def test_calibrate_scores_the_truncated_text(nested):
    seen: list[str] = []

    class Spy(KeywordBackend):
        def score(self, texts, candidates, **kw):
            seen.extend(texts)
            return super().score(texts, candidates, **kw)

    long = "tax " * 400
    data = Dataset.load(FIXTURES / "datasets" / "small.jsonl")
    data = Dataset([data.examples[0].model_copy(update={"text": long})] * 6)
    clf = Classifier(nested, backend=Spy(), view=SectionView(fields=("text",), max_tokens=20))
    calibrate(clf, data)
    assert seen and all(len(t.split()) < 400 for t in seen)


# --- config -----------------------------------------------------------------------------


def _texts(small) -> list[str]:
    return [e.text for e in small]


def test_config_roundtrip(nested, small, tmp_path):
    clf = Classifier(
        nested,
        backend="keyword",
        fallback="keyword:other",
        overrides={"node:legal": "keyword", "depth>=1": "keyword"},
        strategy=Beam(2),
        view=SectionView(fields=("title", "text"), max_tokens=64),
        instruction="Classify it",
        threshold=0.1,
        min_margin=0.05,
        max_fanout=2,
        batch_size=4,
        sections=[1, 2],
        temperatures={"keyword": 0.2},
    )
    clf.save_config(tmp_path / "clf.json")
    data = json.loads((tmp_path / "clf.json").read_text())
    assert data["backend"] == "keyword" and data["fallback"] == "keyword:other"
    assert data["strategy"] == {"type": "beam", "width": 2}
    assert data["settings"]["temperatures"] == {"keyword": 0.2}
    assert data["sections"] == [1, 2] and data["taxonomy_hash"] == nested.content_hash()

    clf2 = Classifier.from_config(tmp_path / "clf.json", nested)
    assert clf2.strategy == Beam(2) and clf2.view == clf.view and clf2.sections == [1, 2]
    assert clf2.settings == clf.settings and clf2.overrides == clf.overrides
    assert clf2.fallback == "keyword:other"
    texts = _texts(small)
    assert [r.model_dump() for r in clf.classify_texts(texts)] == [
        r.model_dump() for r in clf2.classify_texts(texts)
    ]


@pytest.mark.parametrize("strategy", [Greedy(), Beam(5), Flat(), Flat(allow_internal=True)])
def test_config_roundtrips_every_strategy(nested, tmp_path, strategy):
    Classifier(nested, backend="keyword", strategy=strategy).save_config(tmp_path / "c.json")
    assert Classifier.from_config(tmp_path / "c.json", nested).strategy == strategy


def test_config_roundtrips_calibrated_temperatures(nested, small, tmp_path):
    clf = calibrate(Classifier(nested, backend="keyword", temperatures={"keyword": 10.0}), small)
    assert clf.settings.temperatures["keyword"] != 10.0
    clf.save_config(tmp_path / "c.json")
    spec = json.loads((tmp_path / "c.json").read_text())["settings"]["temperatures"]
    assert spec == clf.settings.temperatures


def test_save_config_rejects_unserialisable_parts(nested, tmp_path):
    with pytest.raises(ValueError, match="sections"):
        Classifier(nested, backend="keyword", sections=lambda s: True).save_config(tmp_path / "a")
    with pytest.raises(ValueError, match="backend object"):
        Classifier(nested, backend=KeywordBackend()).save_config(tmp_path / "b")
    with pytest.raises(ValueError, match="override"):
        Classifier(
            nested, backend="keyword", overrides={"node:legal": KeywordBackend()}
        ).save_config(tmp_path / "c")
    assert not list(tmp_path.iterdir())  # nothing half-written


def _trained_stub(nested) -> TrainedModel:
    recipe = get_recipe("BAAI/bge-small-en-v1.5")
    manifest = {
        "base_model": "BAAI/bge-small-en-v1.5",
        "taxonomy_hash": nested.content_hash(),
        "instruction": "x",
        "recipe": recipe.model_dump(),
        "heads": {},
    }

    class Body:
        name = "embedding:fake"

        def _load(self):
            class _Model:
                def save(self, path):
                    Path(path).mkdir(parents=True, exist_ok=True)

            return _Model()

    return TrainedModel(manifest, {}, Body())  # type: ignore[arg-type]


def test_save_config_unsaved_trained_raises(nested, tmp_path):
    model = _trained_stub(nested)
    clf = Classifier(nested, backend=model)
    with pytest.raises(ValueError, match="never saved"):
        clf.save_config(tmp_path / "c.json")


def test_save_config_writes_trained_path(nested, tmp_path):
    pytest.importorskip("joblib")
    model = _trained_stub(nested)
    model.save(tmp_path / "bundle")
    assert model.path == (tmp_path / "bundle").resolve()
    Classifier(nested, backend=model, fallback="keyword").save_config(tmp_path / "c.json")
    data = json.loads((tmp_path / "c.json").read_text())
    assert data["backend"] == f"trained:{(tmp_path / 'bundle').resolve()}"


def test_from_config_warns_on_a_different_taxonomy(nested, tmp_path):
    Classifier(nested, backend="keyword").save_config(tmp_path / "c.json")
    other = Taxonomy([TaxonomyNode(id="a", label="A"), TaxonomyNode(id="b", label="B")])
    with pytest.warns(UserWarning, match="taxonomy"):
        Classifier.from_config(tmp_path / "c.json", other)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Classifier.from_config(tmp_path / "c.json", nested)


def test_from_config_errors(nested, tmp_path):
    (tmp_path / "bad.json").write_text('{"backend": "keyword", "strategy": {"type": "nope"}}')
    with pytest.raises(ValueError, match="strategy"):
        Classifier.from_config(tmp_path / "bad.json", nested)
    with pytest.raises(FileNotFoundError):
        Classifier.from_config(tmp_path / "missing.json", nested)
