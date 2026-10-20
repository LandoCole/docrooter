"""Doc-type section rules: each doc-type leaf's ``sections`` block, and matching section tags
against them (spec G3 and ruling 6.3)."""

import json

import pytest
from pydantic import ValidationError

from docrooter.rules import DocTypeRules, RuleMatch, doc_type_rules, match_rules
from docrooter.taxonomies import builtin_taxonomy
from docrooter.taxonomy import Taxonomy

# doc_types.json without financial.fact_sheet, as pinned in test_element_taxonomies: the rule
# blocks must not change the hash that trained bundles store.
V04_DOC_TYPES_HASH = "bae09bbd9a31e78de2cc2ecff68b1e5401b0537cee3f7b836cc82957c839ea6a"


def _rules(**blocks: dict) -> dict[str, DocTypeRules]:
    return {name: DocTypeRules(**block) for name, block in blocks.items()}


def test_every_doc_type_leaf_has_rules_with_known_ids():
    doc_types = builtin_taxonomy("doc_types")
    catalogue = builtin_taxonomy("section_catalogue")
    rules = doc_type_rules(doc_types)
    assert list(rules) == [leaf.id for leaf in doc_types.leaves()]
    assert len(rules) == 19
    for doc_type, rule in rules.items():
        assert rule.requires, doc_type
        ids = [i for group in rule.requires for i in group] + rule.typical + rule.excludes
        for section_id in ids:
            assert section_id in catalogue, (doc_type, section_id)
            assert not section_id.endswith(".other"), (doc_type, section_id)
        required = {i for group in rule.requires for i in group}
        assert not required & set(rule.excludes), doc_type
    for group in doc_types.children():
        assert group.sections is None, group.id


def test_rule_blocks_leave_doc_types_hash_unchanged():
    doc_types = builtin_taxonomy("doc_types")
    v04 = Taxonomy([n for n in doc_types.nodes.values() if n.id != "financial.fact_sheet"])
    assert v04.content_hash() == V04_DOC_TYPES_HASH
    # the blocks still round-trip through the flat dump
    again = Taxonomy.from_dict(json.loads(json.dumps(doc_types.to_dict())))
    assert doc_type_rules(again) == doc_type_rules(doc_types)
    assert "sections" not in builtin_taxonomy("section_catalogue").to_dict()["nodes"][0]


def test_rules_block_is_validated():
    with pytest.raises(ValidationError):
        DocTypeRules(requires=[])
    with pytest.raises(ValidationError):
        DocTypeRules(requires=[["a.b"], []])
    with pytest.raises(ValidationError):
        DocTypeRules(requires=[["a.b"]], unknown=["x"])


def test_group_id_matches_any_leaf():
    rules = _rules(report={"requires": [["financial"]]})
    for leaf in ("financial.balance_sheet", "financial.notes_to_financial_statements"):
        result = match_rules(rules, {leaf: 0.9})
        assert result.candidates == ["report"]
        assert result.support == {"report": {leaf: 0.9}}
    # a group id is a prefix by whole segments only
    assert match_rules(rules, {"financialx.balance_sheet": 0.9}).candidates == []
    # a leaf id matches only itself
    leaf_rule = _rules(report={"requires": [["financial.balance_sheet"]]})
    assert match_rules(leaf_rule, {"financial.income_statement": 0.9}).candidates == []
    assert match_rules(leaf_rule, {"financial.balance_sheet": 0.9}).candidates == ["report"]


def test_candidates_near_misses_and_excludes():
    rules = _rules(
        both={"requires": [["x.a"], ["x.b"]], "typical": ["x.e"]},
        typical={"requires": [["x.a"], ["x.b"]], "typical": ["x.d", "x.f"]},
        near={"requires": [["x.a"], ["x.c"]]},
        unrelated={"requires": [["y"]]},
        barred={"requires": [["x.a"]], "excludes": ["x.d"]},
    )
    tags = {"x.a": 0.9, "x.b": 0.8, "x.d": 0.6, "x.c": 0.4, "y.z": 0.3}
    result = match_rules(rules, tags)
    assert isinstance(result, RuleMatch)
    # most typical matches first; tags below min_conf count for nothing
    assert result.candidates == ["typical", "both"]
    assert result.excluded == ["barred"]
    # a near miss met all but one required group, and at least one
    assert result.near_misses == ["near"]
    assert result.decided is None
    assert result.support["both"] == {"x.a": 0.9, "x.b": 0.8}
    assert result.support["near"] == {"x.a": 0.9}
    assert "unrelated" not in result.support

    # min_conf moves the line
    assert "near" in match_rules(rules, tags, min_conf=0.3).candidates

    # an excludes hit is ignored when removing it would leave no candidate
    alone = match_rules(rules, {"x.a": 0.9, "x.d": 0.9})
    assert alone.candidates == ["barred"]
    assert alone.excluded == []
    assert alone.decided == "barred"

    # every candidate excluded: all are kept, never an empty result
    clash = _rules(
        one={"requires": [["x.a"]], "excludes": ["x.b"]},
        two={"requires": [["x.b"]], "excludes": ["x.a"]},
    )
    result = match_rules(clash, {"x.a": 0.9, "x.b": 0.9})
    assert result.candidates == ["one", "two"]
    assert result.excluded == []

    # no tags at all: nothing, and no error
    empty = match_rules(rules, {})
    assert (empty.candidates, empty.near_misses, empty.decided) == ([], [], None)


def test_single_candidate_decided_needs_conf_0_7():
    rules = _rules(
        only={"requires": [["x.a"], ["x.b"]]},
        other={"requires": [["y.a"], ["y.b"]]},
    )
    assert match_rules(rules, {"x.a": 0.9, "x.b": 0.7}).decided == "only"
    weak = match_rules(rules, {"x.a": 0.9, "x.b": 0.69})
    assert weak.candidates == ["only"]
    assert weak.decided is None
    # each required group is supported by its strongest tag
    strong = match_rules(rules, {"x.a": 0.9, "x.b": 0.6, "x.b2": 0.1})
    assert strong.decided is None
    grouped = _rules(only={"requires": [["x.a"], ["x.b", "x.c"]]})
    result = match_rules(grouped, {"x.a": 0.9, "x.b": 0.55, "x.c": 0.8})
    assert result.support["only"] == {"x.a": 0.9, "x.c": 0.8}
    assert result.decided == "only"
    assert match_rules(rules, {"x.a": 0.9, "x.b": 0.69}, decide_conf=0.6).decided == "only"
    # two candidates are never decided by rules
    both = match_rules(rules, {"x.a": 0.9, "x.b": 0.9, "y.a": 0.9, "y.b": 0.9})
    assert both.candidates == ["only", "other"]
    assert both.decided is None


def test_quarterly_report_example():
    rules = doc_type_rules(builtin_taxonomy("doc_types"))
    statements_and_letter = {
        "financial.income_statement": 0.9,
        "financial.balance_sheet": 0.85,
        "narrative.shareholder_letter": 0.8,
    }
    assert "financial.quarterly_report" in match_rules(rules, statements_and_letter).candidates
    with_mdna = {"financial.cash_flow_statement": 0.9, "narrative.management_discussion": 0.8}
    assert "financial.quarterly_report" in match_rules(rules, with_mdna).candidates

    letter_only = {"narrative.shareholder_letter": 0.8}
    result = match_rules(rules, letter_only)
    assert "financial.quarterly_report" not in result.candidates
    assert result.candidates == ["financial.shareholder_letter"]
    statements_only = {"financial.income_statement": 0.9}
    assert "financial.quarterly_report" not in match_rules(rules, statements_only).candidates
