import numpy as np

from docrooter.backends import (
    Backend,
    Candidate,
    KeywordBackend,
    Scores,
    candidate_text,
    to_probs,
)
from docrooter.taxonomy import Taxonomy, TaxonomyNode


def test_similarity_softmax_with_temperature():
    p = to_probs(Scores(np.array([[0.9, 0.1]]), "similarity"), temperature=0.1, multi_label=False)
    assert p.shape == (1, 2) and p[0, 0] > 0.99 and np.isclose(p.sum(), 1)


def test_similarity_is_numerically_stable():
    p = to_probs(Scores(np.array([[1e4, 0.0]]), "similarity"), temperature=0.001, multi_label=False)
    assert np.all(np.isfinite(p)) and np.allclose(p, [[1.0, 0.0]])


def test_empty_candidates():
    for kind in ("similarity", "softmax", "independent"):
        p = to_probs(Scores(np.zeros((3, 0)), kind), temperature=0.5, multi_label=False)
        assert p.shape == (3, 0)


def test_softmax_passthrough_at_temperature_one():
    p = np.array([[0.7, 0.3]])
    assert np.array_equal(to_probs(Scores(p, "softmax"), temperature=1, multi_label=False), p)


def test_softmax_retempered():
    p = np.array([[0.7, 0.3]])
    sharp = to_probs(Scores(p, "softmax"), temperature=0.5, multi_label=False)
    assert sharp[0, 0] > 0.7 and np.isclose(sharp.sum(), 1)
    zero = to_probs(Scores(np.array([[1.0, 0.0]]), "softmax"), temperature=0.5, multi_label=False)
    assert np.all(np.isfinite(zero)) and np.isclose(zero.sum(), 1)


def test_independent_renormalised_for_single_label():
    p = to_probs(Scores(np.array([[0.6, 0.6]]), "independent"), temperature=1, multi_label=False)
    assert np.allclose(p, [[0.5, 0.5]])


def test_independent_as_is_for_multi_label():
    v = np.array([[0.6, 0.6]])
    assert np.array_equal(to_probs(Scores(v, "independent"), temperature=1, multi_label=True), v)


def test_independent_all_zero_row_becomes_uniform():
    p = to_probs(Scores(np.zeros((1, 4)), "independent"), temperature=1, multi_label=False)
    assert np.allclose(p, 0.25)


def _taxonomy():
    return Taxonomy(
        [
            TaxonomyNode(id="fin", label="Financials"),
            TaxonomyNode(id="tax", label="Tax", description="income taxes", parent_id="fin"),
        ]
    )


def test_candidate_text():
    tax = _taxonomy()
    leaf, top = tax.nodes["tax"], tax.nodes["fin"]
    assert candidate_text(top) == "Financials"
    assert candidate_text(leaf) == "Tax: income taxes"
    assert candidate_text(leaf, taxonomy=tax) == "Tax: income taxes"
    assert candidate_text(leaf, taxonomy=tax, full_path=True) == "Financials > Tax: income taxes"
    assert candidate_text(top, taxonomy=tax, full_path=True) == "Financials"


def test_keyword_backend_scores():
    s = KeywordBackend().score(
        ["income tax expense"], [Candidate("t", "Tax: income tax"), Candidate("a", "Audit")]
    )
    assert s.kind == "similarity" and s.values.shape == (1, 2)
    assert s.values[0, 0] > s.values[0, 1]


def test_keyword_backend_examples_and_empty():
    b = KeywordBackend()
    s = b.score(["goodwill impairment"], [Candidate("g", "Other", ("goodwill impairment",))])
    assert s.values[0, 0] > 0
    assert b.score([""], [Candidate("x", "")]).values[0, 0] == 0
    assert b.score(["a"], []).values.shape == (1, 0)


def test_keyword_backend_satisfies_protocol():
    b = KeywordBackend()
    assert isinstance(b, Backend)
    assert b.supports("anything") and b.max_candidates is None
    assert b.tokenizer() is None and b.temperature == 0.1 and b.name == "keyword"
