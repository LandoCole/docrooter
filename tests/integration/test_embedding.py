import pytest

from docrooter.backends.base import Candidate
from docrooter.backends.embedding import EmbeddingBackend

pytestmark = pytest.mark.models


@pytest.mark.parametrize("model", ["BAAI/bge-small-en-v1.5", "dmis-lab/UltraBERT-Embed-150M"])
def test_embedding_zero_shot_sanity(model):
    b = EmbeddingBackend(model)
    s = b.score(
        ["The company recorded a deferred income tax liability."],
        [
            Candidate("tax", "Tax: income taxes"),
            Candidate("lit", "Litigation: lawsuits and legal proceedings"),
        ],
    )
    print(f"\n{model}: tax={s.values[0, 0]:.4f} lit={s.values[0, 1]:.4f}")
    assert s.kind == "similarity" and s.values[0, 0] > s.values[0, 1]
