"""Backends that implement only the required protocol members still work end to end."""

from __future__ import annotations

import numpy as np
from conftest import fin_legal_taxonomy, taxonomy_with_multilabel_financial

from docrooter import Classifier, Dataset, calibrate
from docrooter.backends import KeywordBackend, Scores
from docrooter.train.calibrate import mean_nll
from docrooter.train.dataset import Example


class MinimalBackend:
    """No ``temperature`` attribute and a ``score`` without ``multi_label``."""

    name = "minimal"
    max_candidates = None

    def __init__(self):
        self.inner = KeywordBackend()
        self.calls = 0

    def score(self, texts, candidates, *, instruction=None):
        self.calls += 1
        return self.inner.score(texts, candidates, instruction=instruction)

    def supports(self, node_id):
        return True

    def tokenizer(self):
        return None


class KwargsBackend(MinimalBackend):
    """Records the keyword arguments of every ``score`` call."""

    def __init__(self):
        super().__init__()
        self.kwargs: list[dict] = []

    def score(self, texts, candidates, **kwargs):
        self.kwargs.append(kwargs)
        return Scores(np.full((len(texts), len(candidates)), 0.9), "independent")


def test_minimal_backend_classifies_with_temperature_one():
    backend = MinimalBackend()
    clf = Classifier(fin_legal_taxonomy(), backend, threshold=0.0)
    (result,) = clf.classify_texts(["income tax deferred tax financial"])
    assert result.best.node_ids == ["financial", "financial.tax"]
    assert backend.calls > 0
    # temperature 1.0: the same probabilities as a keyword backend at temperature 1.0
    ref = Classifier(fin_legal_taxonomy(), KeywordBackend(temperature=1.0), threshold=0.0)
    (expected,) = ref.classify_texts(["income tax deferred tax financial"])
    assert result.best.steps[0].probs == expected.best.steps[0].probs


def test_multi_label_is_passed_only_at_multi_label_nodes():
    backend = KwargsBackend()
    clf = Classifier(taxonomy_with_multilabel_financial(), backend, threshold=0.0)
    clf.classify_texts(["tax"])
    assert {"instruction": None} in backend.kwargs  # the single-label root decision
    assert {"instruction": None, "multi_label": True} in backend.kwargs  # under Financial
    assert all(kw.get("multi_label", True) is True for kw in backend.kwargs)


def _dev() -> Dataset:
    texts = ["income tax deferred tax", "audit opinion auditor", "contract clause", "lawsuit"]
    paths = [
        ["financial", "financial.tax"],
        ["financial", "financial.audit"],
        ["legal", "legal.contracts"],
        ["legal", "legal.litigation"],
    ]
    return Dataset([Example(text=t, path=p) for t, p in zip(texts * 3, paths * 3, strict=True)])


def test_minimal_backend_can_be_calibrated():
    clf = Classifier(fin_legal_taxonomy(), MinimalBackend(), threshold=0.0)
    assert "minimal" in mean_nll(clf, _dev())
    calibrated = calibrate(clf, _dev())
    assert calibrated.calibration_report["minimal"]["n"] == 24


def test_calibrate_passes_multi_label_only_at_multi_label_nodes():
    backend = KwargsBackend()
    calibrate(Classifier(taxonomy_with_multilabel_financial(), backend), _dev())
    assert {"instruction": None} in backend.kwargs
    assert {"instruction": None, "multi_label": True} in backend.kwargs
    assert all(kw.get("multi_label", True) is True for kw in backend.kwargs)
