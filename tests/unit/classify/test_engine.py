import dataclasses
import math

import numpy as np
import pytest
from conftest import (
    CountingBackend,
    NeverSupports,
    RecordingBackend,
    chain_taxonomy,
    fin_legal_taxonomy,
    taxonomy_with_multilabel_financial,
    taxonomy_with_multilabel_root,
)

from docrooter.backends import KeywordBackend, Scores
from docrooter.classify import (
    Beam,
    DocumentResult,
    EngineSettings,
    Flat,
    Greedy,
    PathResult,
    SectionResult,
    Step,
    decide,
    run_tree,
)
from docrooter.classify.engine import AUTO_BACKEND
from docrooter.errors import BackendError
from docrooter.taxonomy import Taxonomy, TaxonomyNode

# --- the brief's behaviour tests -------------------------------------------------------------


def test_greedy_reaches_leaf(tax, router):
    out = run_tree(
        {"x": "income tax deferred tax financial"},
        tax,
        router,
        Greedy(),
        EngineSettings(threshold=0.0),
    )
    best, _, _ = out["x"]
    assert best.node_ids == ["financial", "financial.tax"] and best.stopped_reason is None
    assert best.steps[0].node_id == Taxonomy.ROOT_ID and len(best.steps) == 2


def test_low_confidence_stops_at_internal_node(tax, router):
    best, _, _ = run_tree({"x": "financial"}, tax, router, Greedy(), EngineSettings(threshold=0.9))[
        "x"
    ]
    assert best.node_ids == ["financial"] and best.stopped_reason == "low_confidence"


def test_beam_returns_alternatives_sorted(tax, router):
    best, alts, _ = run_tree(
        {"x": "tax contracts"}, tax, router, Beam(3), EngineSettings(threshold=0.0)
    )["x"]
    assert alts and all(a.score <= best.score for a in alts)


def test_multilabel_node_selects_several(router):
    t = taxonomy_with_multilabel_root()
    _, _, selected = run_tree(
        {"x": "tax audit contracts litigation"}, t, router, Beam(2), EngineSettings(threshold=0.1)
    )["x"]
    assert {p.node_ids[0] for p in selected} == {"financial", "legal"}


def test_low_margin_stops(tax, router):
    best, _, _ = run_tree(
        {"x": "tax contracts"}, tax, router, Greedy(), EngineSettings(threshold=0, min_margin=0.2)
    )["x"]
    assert best.node_ids == [] and best.stopped_reason == "low_margin"


def test_no_backend_reason(tax):
    best, _, _ = run_tree({"x": "tax"}, tax, lambda n: [], Greedy(), EngineSettings())["x"]
    assert best.node_ids == [] and best.stopped_reason == "no_backend"


def test_batching_one_call_per_node_per_round(tax):
    spy = CountingBackend(KeywordBackend())
    run_tree(
        {str(i): "income tax" for i in range(100)},
        tax,
        lambda n: [spy],
        Greedy(),
        EngineSettings(threshold=0, batch_size=1000),
    )
    assert spy.calls == 2  # root round + financial round


def test_depth_30_chain(router):
    best, _, _ = run_tree(
        {"x": "l29"}, chain_taxonomy(30), router, Greedy(), EngineSettings(threshold=0)
    )["x"]
    # single-child nodes are auto-chosen with p=1.0, no backend call
    assert len(best.node_ids) == 30


# --- path contents ---------------------------------------------------------------------------


def test_path_labels_steps_and_score(tax, router):
    best, alts, selected = run_tree(
        {"x": "income tax deferred tax financial"}, tax, router, Greedy(), EngineSettings()
    )["x"]
    assert best.labels == ["Financial", "Tax"]
    root_step, fin_step = best.steps
    assert root_step.chosen == ["financial"] and fin_step.node_id == "financial"
    assert fin_step.chosen == ["financial.tax"]
    assert list(root_step.probs) == ["financial", "legal"]  # ranked, keyed by child id
    assert math.isclose(sum(root_step.probs.values()), 1.0)
    assert root_step.backend == "keyword"
    expected = root_step.probs["financial"] * fin_step.probs["financial.tax"]
    assert math.isclose(best.score, expected)
    assert alts == [] and selected == [best]


def test_stopped_at_root_has_empty_labels_and_no_steps(tax, router):
    best, _, _ = run_tree(
        {"x": "nothing here"}, tax, router, Greedy(), EngineSettings(threshold=0.9)
    )["x"]
    assert best.node_ids == [] and best.labels == [] and best.steps == []
    assert best.stopped_reason == "low_confidence" and best.score == 1.0


def test_ties_keep_taxonomy_child_order(tax, router):
    # "contracts" alone gives Financial and Legal the same root score; Financial comes first.
    best, _, _ = run_tree({"x": "contracts"}, tax, router, Greedy(), EngineSettings(threshold=0))[
        "x"
    ]
    assert best.node_ids[0] == "financial"
    assert list(best.steps[0].probs) == ["financial", "legal"]


def test_step_probs_hold_at_most_five_children(router):
    nodes = [TaxonomyNode(id=f"c{i}", label=f"C{i}") for i in range(8)]
    best, _, _ = run_tree({"x": "c3"}, Taxonomy(nodes), router, Greedy(), EngineSettings())["x"]
    assert best.node_ids == ["c3"] and len(best.steps[0].probs) == 5
    assert next(iter(best.steps[0].probs)) == "c3"


def test_runs_every_key(tax, router):
    out = run_tree(
        {"a": "income tax financial", "b": "legal contracts"},
        tax,
        router,
        Greedy(),
        EngineSettings(),
    )
    assert out["a"][0].node_ids == ["financial", "financial.tax"]
    assert out["b"][0].node_ids == ["legal", "legal.contracts"]
    assert run_tree({}, tax, router, Greedy(), EngineSettings()) == {}


# --- thresholds, margins, settings -----------------------------------------------------------


def test_node_threshold_overrides_setting(router):
    nodes = [
        TaxonomyNode(id="financial", label="Financial", threshold=0.999),
        TaxonomyNode(id="financial.tax", label="Tax", parent_id="financial"),
        TaxonomyNode(id="financial.audit", label="Audit", parent_id="financial"),
        TaxonomyNode(id="legal", label="Legal"),
    ]
    best, _, _ = run_tree(
        {"x": "financial tax"}, Taxonomy(nodes), router, Greedy(), EngineSettings(threshold=0)
    )["x"]
    assert best.node_ids == ["financial"] and best.stopped_reason == "low_confidence"


def test_root_threshold_from_taxonomy_root_settings(router):
    t = Taxonomy(list(fin_legal_taxonomy().nodes.values()), root={"threshold": 0.99})
    best, _, _ = run_tree({"x": "tax"}, t, router, Greedy(), EngineSettings(threshold=0))["x"]
    assert best.node_ids == [] and best.stopped_reason == "low_confidence"


def test_low_margin_applies_to_beam_too(tax, router):
    best, alts, _ = run_tree(
        {"x": "tax contracts"}, tax, router, Beam(3), EngineSettings(threshold=0, min_margin=0.2)
    )["x"]
    assert best.node_ids == [] and best.stopped_reason == "low_margin" and alts == []


def test_instruction_and_multi_label_passed_to_backend():
    spy = RecordingBackend(KeywordBackend())
    t = taxonomy_with_multilabel_root()
    run_tree(
        {"x": "tax"}, t, lambda n: [spy], Greedy(), EngineSettings(instruction="Classify 10-K")
    )
    assert spy.last_instruction == "Classify 10-K"
    assert spy.last_multi_label is False  # the last call was at single-label "financial"
    spy2 = RecordingBackend(KeywordBackend())
    run_tree({"x": "nothing"}, t, lambda n: [spy2], Greedy(), EngineSettings(threshold=0.9))
    assert spy2.last_multi_label is True and spy2.calls == 1


def test_candidates_are_children_in_taxonomy_order():
    spy = RecordingBackend(KeywordBackend())
    nodes = [
        TaxonomyNode(id="b", label="Beta", description="second letter", examples=["bb"]),
        TaxonomyNode(id="a", label="Alpha"),
    ]
    run_tree({"x": "beta"}, Taxonomy(nodes), lambda n: [spy], Greedy(), EngineSettings())
    assert [(c.node_id, c.text, c.examples) for c in spy.last_candidates] == [
        ("b", "Beta: second letter", ("bb",)),
        ("a", "Alpha", ()),
    ]


def test_temperature_override_by_backend_name(tax, router):
    sharp = run_tree({"x": "financial"}, tax, router, Greedy(), EngineSettings())["x"][0]
    flat = run_tree(
        {"x": "financial"}, tax, router, Greedy(), EngineSettings(temperatures={"keyword": 100.0})
    )["x"][0]
    assert sharp.steps[0].probs["financial"] > 0.99
    assert 0.5 < flat.steps[0].probs["financial"] < 0.51


def test_mini_batches_of_batch_size(tax):
    spy = CountingBackend(KeywordBackend())
    run_tree(
        {str(i): "financial" for i in range(5)},
        tax,
        lambda n: [spy],
        Greedy(),
        EngineSettings(threshold=0.9, batch_size=2),
    )
    assert spy.calls == 3 + 3  # 5 texts in batches of 2, at the root and at "financial"


# --- backend selection -----------------------------------------------------------------------


def test_router_skips_unsupporting_and_too_small_backends(tax):
    small = KeywordBackend(name="small")
    small.max_candidates = 1
    chosen = KeywordBackend(name="chosen")
    best, _, _ = run_tree(
        {"x": "income tax financial"},
        tax,
        lambda n: [NeverSupports(), small, chosen],
        Greedy(),
        EngineSettings(),
    )["x"]
    assert [s.backend for s in best.steps] == ["chosen", "chosen"]


def test_router_is_per_node(tax):
    a, b = KeywordBackend(name="a"), KeywordBackend(name="b")
    best, _, _ = run_tree(
        {"x": "income tax financial"},
        tax,
        lambda n: [a] if n == Taxonomy.ROOT_ID else [b],
        Greedy(),
        EngineSettings(),
    )["x"]
    assert [s.backend for s in best.steps] == ["a", "b"]


def test_no_backend_below_root_stops_there(tax):
    kw = KeywordBackend()
    best, _, _ = run_tree(
        {"x": "income tax financial"},
        tax,
        lambda n: [kw] if n == Taxonomy.ROOT_ID else [],
        Greedy(),
        EngineSettings(),
    )["x"]
    assert best.node_ids == ["financial"] and best.stopped_reason == "no_backend"


def test_single_child_needs_no_backend():
    best, _, _ = run_tree({"x": "?"}, chain_taxonomy(3), lambda n: [], Greedy(), EngineSettings())[
        "x"
    ]
    assert best.node_ids == ["l0", "l1", "l2"] and best.stopped_reason is None
    assert [s.backend for s in best.steps] == [AUTO_BACKEND] * 3
    assert best.steps[1].node_id == "l0" and best.steps[1].probs == {"l1": 1.0}
    assert best.score == 1.0


def test_backend_with_wrong_output_shape_raises(tax):
    class Broken(KeywordBackend):
        def score(self, texts, candidates, *, instruction=None, multi_label=False):
            return Scores(np.zeros((len(texts), len(candidates) + 1)), "similarity")

    with pytest.raises(BackendError, match="broken"):
        run_tree({"x": "tax"}, tax, lambda n: [Broken(name="broken")], Greedy(), EngineSettings())


# --- depth guard -----------------------------------------------------------------------------


def test_max_depth_stops_unfinished_paths(router):
    best, _, _ = run_tree(
        {"x": "?"}, chain_taxonomy(5), router, Greedy(), EngineSettings(max_depth=2)
    )["x"]
    assert best.node_ids == ["l0", "l1"] and best.stopped_reason == "max_depth"


def test_reaching_a_leaf_on_the_last_round_is_not_max_depth(router):
    best, _, _ = run_tree(
        {"x": "?"}, chain_taxonomy(3), router, Greedy(), EngineSettings(max_depth=3)
    )["x"]
    assert best.node_ids == ["l0", "l1", "l2"] and best.stopped_reason is None


# --- beam and multi-label --------------------------------------------------------------------


def test_beam_keeps_width_paths_sorted(tax, router):
    best, alts, selected = run_tree(
        {"x": "tax contracts"}, tax, router, Beam(3), EngineSettings(threshold=0)
    )["x"]
    assert best.node_ids == ["financial", "financial.tax"]
    assert [a.node_ids for a in alts] == [
        ["legal", "legal.contracts"],
        ["financial", "financial.audit"],
    ]
    assert alts[0].score >= alts[1].score and selected == [best]
    assert all(len(s.chosen) == 1 for p in [best, *alts] for s in p.steps)


def test_beam_width_one_matches_greedy(tax, router):
    text = {"x": "tax contracts litigation"}
    g = run_tree(text, tax, router, Greedy(), EngineSettings(threshold=0))["x"]
    b = run_tree(text, tax, router, Beam(1), EngineSettings(threshold=0))["x"]
    assert b[0] == g[0] and b[1] == [] and g[1] == []


def test_beam_prunes_below_threshold(tax, router):
    best, alts, _ = run_tree(
        {"x": "tax contracts"}, tax, router, Beam(3), EngineSettings(threshold=0.4)
    )["x"]
    paths = [p.node_ids for p in [best, *alts]]
    assert paths == [["financial", "financial.tax"], ["legal", "legal.contracts"]]


def test_greedy_multilabel_follows_each_eligible_child(router):
    t = taxonomy_with_multilabel_root()
    best, alts, selected = run_tree(
        {"x": "tax contracts"}, t, router, Greedy(), EngineSettings(threshold=0.1)
    )["x"]
    assert [p.node_ids for p in selected] == [
        ["financial", "financial.tax"],
        ["legal", "legal.contracts"],
    ]
    assert best == selected[0] and alts == []


def test_max_fanout_limits_multilabel_branches(router):
    t = taxonomy_with_multilabel_root()
    _, _, selected = run_tree(
        {"x": "tax contracts"}, t, router, Greedy(), EngineSettings(threshold=0, max_fanout=1)
    )["x"]
    assert [p.node_ids[0] for p in selected] == ["financial"]


def test_multilabel_ignores_min_margin(router):
    t = taxonomy_with_multilabel_root()
    _, _, selected = run_tree(
        {"x": "tax contracts"}, t, router, Greedy(), EngineSettings(threshold=0, min_margin=0.5)
    )["x"]
    assert {p.node_ids[0] for p in selected} == {"financial", "legal"}


def test_rival_of_a_multilabel_fan_out_is_an_alternative(router):
    # Legal > Contracts beats both Financial children, so the Financial fan-out is a rival.
    t = taxonomy_with_multilabel_financial()
    best, alts, selected = run_tree(
        {"x": "tax audit contracts"}, t, router, Beam(3), EngineSettings(threshold=0.1)
    )["x"]
    assert best.node_ids == ["legal", "legal.contracts"] and selected == [best]
    assert [a.node_ids for a in alts] == [
        ["financial", "financial.tax"],
        ["financial", "financial.audit"],
    ]


def test_winning_multilabel_fan_out_selects_each_branch(router):
    # Financial wins at the root and fans out to both children; Legal paths are rivals.
    t = taxonomy_with_multilabel_financial()
    best, alts, selected = run_tree(
        {"x": "financial tax audit"}, t, router, Beam(3), EngineSettings(threshold=0.01)
    )["x"]
    assert [p.node_ids for p in selected] == [
        ["financial", "financial.tax"],
        ["financial", "financial.audit"],
    ]
    assert best == selected[0]
    assert alts and all(a.node_ids[0] == "legal" for a in alts)
    assert all(a.score <= p.score for a in alts for p in selected)


def test_nested_fan_outs_are_selected_together(router):
    t = taxonomy_with_multilabel_financial(root_multi_label=True)
    best, alts, selected = run_tree(
        {"x": "tax audit contracts"}, t, router, Greedy(), EngineSettings(threshold=0.1)
    )["x"]
    assert [p.node_ids[-1] for p in selected] == [
        "legal.contracts",
        "financial.tax",
        "financial.audit",
    ]
    assert best == selected[0] and alts == []


def test_alternatives_capped_at_beam_width(router):
    t = taxonomy_with_multilabel_financial()
    # Rivals: both Financial fan-out branches plus Legal > Litigation; Beam(2) reports two.
    best, alts, _ = run_tree(
        {"x": "tax audit contracts"}, t, router, Beam(2), EngineSettings(threshold=0.01)
    )["x"]
    assert best.node_ids == ["legal", "legal.contracts"]
    assert [a.node_ids[-1] for a in alts] == ["financial.tax", "financial.audit"]


def test_every_step_shows_its_chosen_child():
    nodes = [TaxonomyNode(id=f"c{i}", label=f"C{i}") for i in range(8)]
    best, alts, _ = run_tree(
        {"x": "c3"},
        Taxonomy(nodes),
        lambda n: [KeywordBackend()],
        Beam(8),
        EngineSettings(threshold=0),
    )["x"]
    paths = [best, *alts]
    assert len(paths) == 8
    assert all(s.chosen[0] in s.probs for p in paths for s in p.steps)


def test_flat_is_not_a_tree_strategy(tax, router):
    with pytest.raises(TypeError, match="Flat"):
        run_tree({"x": "tax"}, tax, router, Flat(), EngineSettings())


# --- decide ----------------------------------------------------------------------------------


def test_decide_scores_children_at_node(tax, router):
    ids, probs, name = decide(["income tax", "audit"], "financial", tax, router, EngineSettings())
    assert ids == ["financial.tax", "financial.audit"] and name == "keyword"
    assert probs.shape == (2, 2) and np.allclose(probs.sum(axis=1), 1)
    assert probs[0, 0] > 0.9 and probs[1, 1] > 0.9


def test_decide_matches_run_tree_step(tax, router):
    best, _, _ = run_tree({"x": "income tax financial"}, tax, router, Greedy(), EngineSettings())[
        "x"
    ]
    ids, probs, _ = decide(
        ["income tax financial"], Taxonomy.ROOT_ID, tax, router, EngineSettings()
    )
    assert dict(zip(ids, probs[0].tolist(), strict=True)) == pytest.approx(best.steps[0].probs)


def test_decide_without_backend_raises(tax):
    with pytest.raises(BackendError, match="financial"):
        decide(["x"], "financial", tax, lambda n: [NeverSupports()], EngineSettings())


def test_decide_single_child_and_leaf(router):
    t = chain_taxonomy(2)
    ids, probs, name = decide(["a", "b"], "l0", t, router, EngineSettings())
    assert ids == ["l1"] and name == AUTO_BACKEND and np.array_equal(probs, np.ones((2, 1)))
    with pytest.raises(ValueError, match="leaf"):
        decide(["a"], "l1", t, router, EngineSettings())


def test_decide_batches(tax):
    spy = CountingBackend(KeywordBackend())
    _, probs, _ = decide(["t"] * 5, "legal", tax, lambda n: [spy], EngineSettings(batch_size=2))
    assert spy.calls == 3 and probs.shape == (5, 2)


# --- strategies and result models ------------------------------------------------------------


def test_strategies_are_frozen_and_validated():
    assert Beam().width == 3 and Flat().allow_internal is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        Beam().width = 5
    with pytest.raises(ValueError, match="width"):
        Beam(0)


def test_engine_settings_defaults():
    s = EngineSettings()
    assert (s.threshold, s.min_margin, s.max_fanout, s.max_depth, s.batch_size) == (
        0.35,
        0.0,
        3,
        64,
        32,
    )
    assert s.instruction is None and s.temperatures == {}


def test_results_round_trip_json():
    step = Step(node_id=Taxonomy.ROOT_ID, chosen=["a"], probs={"a": 0.9, "b": 0.1}, backend="k")
    path = PathResult(node_ids=["a"], labels=["A"], score=0.9, steps=[step], stopped_reason=None)
    section = SectionResult(
        section_id="s1",
        title="Intro",
        best=path,
        alternatives=[],
        selected=[path],
        truncated=False,
        n_tokens=12,
    )
    doc = DocumentResult(source="x.md", sections={"s1": section})
    assert DocumentResult.model_validate_json(doc.model_dump_json()) == doc
    empty = PathResult(node_ids=[], stopped_reason="low_confidence")
    assert empty.labels == [] and empty.steps == [] and empty.score == 1.0
