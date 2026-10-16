import numpy as np
import pytest

from docrooter.backends.base import Candidate
from docrooter.backends.embedding import EmbeddingBackend

pytestmark = pytest.mark.models

TEXT = "The company recorded a deferred income tax liability."
CANDIDATES = [
    Candidate("tax", "Tax: income taxes"),
    Candidate("lit", "Litigation: lawsuits and legal proceedings"),
]


@pytest.mark.parametrize("model", ["BAAI/bge-small-en-v1.5", "dmis-lab/UltraBERT-Embed-150M"])
def test_embedding_zero_shot_sanity(model):
    b = EmbeddingBackend(model)
    try:
        s = b.score([TEXT], CANDIDATES)
        print(f"\n{model}: tax={s.values[0, 0]:.4f} lit={s.values[0, 1]:.4f}")
        assert s.kind == "similarity" and s.values[0, 0] > s.values[0, 1]
    finally:
        b.close()


def test_ultrabert_is_capped_at_1024_tokens_and_close_reloads():
    b = EmbeddingBackend("dmis-lab/UltraBERT-Embed-150M", cache_size=2)
    try:
        assert b._load().max_seq_length == 1024
        assert b.batch_size == (32 if b._device_in_use.startswith("cuda") else 16)
        long_text = "deferred income tax liability " * 2000  # far past 1024 tokens
        first = b.score([long_text, TEXT], CANDIDATES).values
        assert len(b._cache) == 2  # the LRU holds two texts
        b.close()
        assert b._model is None and not b._cache and not b._label_cache
        again = b.score([long_text, TEXT], CANDIDATES).values  # reloads
        assert np.allclose(first, again, atol=1e-4)
    finally:
        b.close()
