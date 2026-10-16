import os
import time

import pytest

from docrooter.backends.base import Candidate
from docrooter.backends.jev import JevBackend

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(not os.environ.get("TYPESAFE_API_KEY"), reason="TYPESAFE_API_KEY not set"),
]

TEXT = "The company recorded a deferred income tax liability."
CANDIDATES = [
    Candidate("tax", "Tax: income taxes"),
    Candidate("lit", "Litigation: lawsuits and legal proceedings"),
]


def test_jev_zero_shot_sanity():
    pytest.importorskip("typesafe_sdk")
    backend = JevBackend()
    start = time.perf_counter()
    s = backend.score([TEXT], CANDIDATES)
    print(f"\n{backend.name}: {time.perf_counter() - start:.2f}s")
    print(f"{backend.name}: tax={s.values[0, 0]:.4f} lit={s.values[0, 1]:.4f}")
    assert s.kind == "softmax"
    assert abs(s.values.sum() - 1) < 1e-6
    assert s.values[0, 0] > s.values[0, 1]


def test_jev_multilabel_sanity():
    pytest.importorskip("typesafe_sdk")
    s = JevBackend().score([TEXT], CANDIDATES, multi_label=True)
    assert s.kind == "independent"
    assert s.values[0, 0] > s.values[0, 1]
