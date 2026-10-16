import pytest

from docrooter.classify.results import PathResult, SectionResult
from docrooter.evaluate.metrics import (
    depth_accuracy,
    exact_path_accuracy,
    hierarchical_prf,
    leaf_prf,
    selective_accuracy,
    sibling_confusions,
    stop_rate,
)


def _result(best: PathResult | None) -> SectionResult:
    return SectionResult(section_id="0", title="t", best=best, n_tokens=1)


def test_exact_path_accuracy():
    pred = [["a", "a.x"], ["b"], ["a", "a.x"], []]
    gold = [["a", "a.x"], ["b", "b.x"], ["a", "a.y"], ["c"]]
    assert exact_path_accuracy(pred, gold) == 0.25
    assert exact_path_accuracy([], []) == 0.0


def test_hierarchical_prf_partial_credit():
    p, r, f = hierarchical_prf([["a", "a.x"]], [["a", "a.y"]])
    assert (p, r) == (0.5, 0.5) and f == 0.5


def test_hierarchical_prf_is_micro_and_penalises_short_and_empty_predictions():
    # hits: 2 + 1 + 0; predicted nodes: 2 + 1 + 0; gold nodes: 2 + 2 + 1
    p, r, f = hierarchical_prf([["a", "a.x"], ["b"], []], [["a", "a.x"], ["b", "b.x"], ["c"]])
    assert p == pytest.approx(3 / 3)
    assert r == pytest.approx(3 / 5)
    assert f == pytest.approx(2 * 1 * 0.6 / 1.6)
    assert hierarchical_prf([[]], [["a"]]) == (0.0, 0.0, 0.0)


def test_depth_accuracy_counts_only_deep_enough_gold():
    assert depth_accuracy([["a"], ["b", "b.x"]], [["a"], ["b", "b.y"]]) == {1: 1.0, 2: 0.0}


def test_stopped_short_prediction_wrong_at_depth2():
    assert depth_accuracy([["a"]], [["a", "a.x"]])[2] == 0.0


def test_depth_accuracy_empty():
    assert depth_accuracy([], []) == {}


def test_stop_rate_counts_missing_best_and_stopped_paths():
    results = [
        _result(PathResult(node_ids=["a", "a.x"])),
        _result(PathResult(node_ids=["a"], stopped_reason="low_confidence")),
        _result(None),
        _result(PathResult(node_ids=[], stopped_reason="empty_text")),
    ]
    assert stop_rate(results) == 0.75
    assert stop_rate([]) == 0.0


def test_selective_accuracy_only_over_unstopped():
    pred = [["a"], ["b"], ["c"]]
    gold = [["a"], ["x"], ["c"]]
    assert selective_accuracy(pred, gold, [False, False, True]) == 0.5
    assert selective_accuracy(pred, gold, [True, True, True]) == 0.0


def test_length_mismatch_is_an_error():
    with pytest.raises(ValueError, match="1 predictions for 2 gold"):
        exact_path_accuracy([["a"]], [["a"], ["b"]])


def test_leaf_prf_macro_hand_computed():
    # gold leaves: x x y y z; predicted leaves: x y y (none) x
    # x: tp1, predicted 2, true 2 -> p .5 r .5 f .5
    # y: tp1, predicted 2, true 2 -> p .5 r .5 f .5
    # z: tp0, predicted 0, true 1 -> p 0  r 0  f 0
    gold = [["g", "x"], ["g", "x"], ["g", "y"], ["g", "y"], ["h", "z"]]
    pred = [["g", "x"], ["g", "y"], ["g", "y"], [], ["g", "x"]]
    p, r, f = leaf_prf(pred, gold)
    assert p == pytest.approx(1.0 / 3)
    assert r == pytest.approx(1.0 / 3)
    assert f == pytest.approx(1.0 / 3)


def test_leaf_prf_perfect_and_empty():
    paths = [["a", "a.x"], ["b"]]
    assert leaf_prf(paths, paths) == (1.0, 1.0, 1.0)
    assert leaf_prf([], []) == (0.0, 0.0, 0.0)


def test_sibling_confusions_top_pairs_per_node():
    decisions = {
        "root": [("a", "b")] * 3 + [("b", "a")] * 2 + [("a", "c"), ("a", "d")] + [("a", "a")] * 5,
        "clean": [("x", "x")],
    }
    out = sibling_confusions(decisions)
    assert out == {"root": [("a", "b", 3), ("b", "a", 2), ("a", "c", 1)]}
    assert sibling_confusions(decisions, top=1) == {"root": [("a", "b", 3)]}
