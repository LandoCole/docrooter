import numpy as np
import pytest

from docrooter.backends.base import Candidate
from docrooter.backends.mlm_choice import MLMChoiceBackend
from docrooter.backends.nli import NLIBackend

pytestmark = pytest.mark.models

TEXT = "The company recorded a deferred income tax liability."
CANDIDATES = [
    Candidate("tax", "Tax: income taxes"),
    Candidate("lit", "Litigation: lawsuits and legal proceedings"),
]


def _check(backend):
    s = backend.score([TEXT], CANDIDATES)
    print(f"\n{backend.name}: tax={s.values[0, 0]:.4f} lit={s.values[0, 1]:.4f}")
    assert s.kind == "softmax"
    assert np.allclose(s.values.sum(axis=1), 1, atol=1e-5)
    assert s.values[0, 0] > s.values[0, 1]


def test_nli_zero_shot_sanity():
    backend = NLIBackend()
    _check(backend)
    independent = backend.score([TEXT], CANDIDATES, multi_label=True)
    assert independent.kind == "independent"
    assert independent.values[0, 0] > independent.values[0, 1]


def test_mlm_choice_zero_shot_sanity():
    _check(MLMChoiceBackend())
