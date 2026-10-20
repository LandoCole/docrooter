"""Doc-type section rules: each doc-type leaf's ``sections`` block, and matching section tags
against them (spec G3 and ruling 6.3)."""

import json

import pytest
from pydantic import ValidationError

from docrooter.rules import (
    DocTypeRules,
    RuleMatch,
    doc_type_rules,
    match_rules,
    unknown_section_ids,
)
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
            assert not section_id.endswith(".other"), (doc_type, section_id)
        required = {i for group in rule.requires for i in group}
        assert not required & set(rule.excludes), doc_type
    assert unknown_section_ids(rules, catalogue) == {}
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

    # an excludes hit is waived when removing it would leave no candidate, and a waived
    # candidate is never decided: the evidence argued against it
    alone = match_rules(rules, {"x.a": 0.9, "x.d": 0.9})
    assert alone.candidates == ["barred"]
    assert alone.excluded == []
    assert alone.waived == ["barred"]
    assert alone.excludes_waived is True
    assert alone.decided is None
    assert result.excludes_waived is False and result.waived == []

    # every candidate excluded: all are kept, never an empty result
    clash = _rules(
        one={"requires": [["x.a"]], "excludes": ["x.b"]},
        two={"requires": [["x.b"]], "excludes": ["x.a"]},
    )
    result = match_rules(clash, {"x.a": 0.9, "x.b": 0.9})
    assert result.candidates == ["one", "two"]
    assert result.excluded == []
    assert result.waived == ["one", "two"]
    assert result.excludes_waived is True

    # no tags at all: nothing, and no error
    empty = match_rules(rules, {})
    assert (empty.candidates, empty.near_misses, empty.decided) == ([], [], None)


def test_waived_sole_candidate_is_not_decided():
    """A candidate kept only by the waiver is not decided, even with strong support."""
    rules = _rules(
        sole={"requires": [["x.a"]], "excludes": ["x.n"]},
        other={"requires": [["y.a"], ["y.b"]]},
    )
    result = match_rules(rules, {"x.a": 0.95, "x.n": 0.95})
    assert result.candidates == ["sole"]
    assert result.decided is None
    assert result.excludes_waived is True
    assert result.waived == ["sole"]
    # without the exclude hit the same support decides
    assert match_rules(rules, {"x.a": 0.95}).decided == "sole"


def test_waived_candidates_keep_near_misses():
    """Waived candidates do not crowd out near misses: both come back for the model."""
    rules = _rules(
        report={"requires": [["x.intro"]], "excludes": ["x.abstract"]},
        white={"requires": [["x.intro"]], "excludes": ["x.abstract"]},
        paper={"requires": [["x.abstract"], ["x.method"]]},
    )
    result = match_rules(rules, {"x.intro": 0.9, "x.abstract": 0.9})
    assert result.candidates == ["report", "white"]
    assert result.excludes_waived is True
    assert result.near_misses == ["paper"]
    assert result.support["paper"] == {"x.abstract": 0.9}
    assert result.decided is None


def test_excluded_near_miss_is_dropped_not_waived():
    rules = _rules(
        near={"requires": [["x.a"], ["x.b"]], "excludes": ["x.c"]},
        other={"requires": [["y.a"]]},
    )
    result = match_rules(rules, {"x.a": 0.9, "x.c": 0.9})
    assert result.near_misses == []
    assert result.excluded == ["near"]
    assert (result.waived, result.excludes_waived) == ([], False)


def test_unknown_section_ids_flags_typos():
    catalogue = builtin_taxonomy("section_catalogue")
    rules = _rules(
        good={"requires": [["financial", "financial.balance_sheet"]]},
        typo={"requires": [["financial.balance_shet"]], "typical": ["narative"]},
    )
    assert unknown_section_ids(rules, catalogue) == {"typo": ["financial.balance_shet", "narative"]}


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


# Realistic tag sets (hand-written from the filing forms, not from the corpus).
TEN_Q = {
    "general.front_matter": 0.9,
    "financial.financial_statements": 0.9,
    "financial.balance_sheet": 0.9,
    "financial.income_statement": 0.9,
    "financial.cash_flow_statement": 0.9,
    "financial.notes_to_financial_statements": 0.9,
    "narrative.management_discussion": 0.9,
    "disclosure.controls_and_procedures": 0.9,
    "disclosure.legal_proceedings": 0.85,
    "disclosure.risk_factors": 0.85,
    "disclosure.exhibits": 0.85,
}


def test_ten_q_with_use_of_proceeds_stays_quarterly_candidate():
    """Part II Item 2 of a 10-Q ("Unregistered Sales of Equity Securities and Use of
    Proceeds") must not turn the filing into a prospectus decided by rules."""
    rules = doc_type_rules(builtin_taxonomy("doc_types"))
    result = match_rules(rules, {**TEN_Q, "offering.use_of_proceeds": 0.9})
    assert "financial.quarterly_report" in result.candidates
    assert result.candidates[0] == "financial.quarterly_report"  # typical leaves rank it first
    assert result.decided != "financial.prospectus"
    assert "financial.prospectus" not in result.candidates


def test_fund_annual_report_is_not_decided_as_earnings_release():
    """A fund shareholder report with no letter: its management discussion of fund
    performance is tagged fund.performance."""
    rules = doc_type_rules(builtin_taxonomy("doc_types"))
    tags = {
        "fund.performance": 0.9,
        "financial.balance_sheet": 0.9,
        "financial.income_statement": 0.9,
        "financial.changes_in_equity": 0.9,
        "financial.schedule_of_investments": 0.9,
        "financial.notes_to_financial_statements": 0.9,
        "financial.auditors_report": 0.9,
    }
    result = match_rules(rules, tags)
    assert "financial.annual_report" in result.candidates
    assert result.decided != "financial.earnings_release"


def test_court_opinion_without_syllabus_is_candidate():
    rules = doc_type_rules(builtin_taxonomy("doc_types"))
    tags = {"law.caption": 0.9, "general.background": 0.8, "general.conclusion": 0.8}
    result = match_rules(rules, tags)
    assert "legal.court_opinion" in result.candidates


def test_scope_alone_is_not_a_technical_standard():
    rules = doc_type_rules(builtin_taxonomy("doc_types"))
    tags = {"standard.scope": 0.9, "general.findings": 0.9, "general.recommendations": 0.9}
    assert "research.technical_standard" not in match_rules(rules, tags).candidates


def test_barred_lists_every_type_an_exclude_argues_against():
    rules = {
        "a": DocTypeRules(requires=[["x"]]),
        "b": DocTypeRules(requires=[["y"]], excludes=["x"]),  # no required group met
    }
    match = match_rules(rules, {"x": 0.9})
    assert match.candidates == ["a"] and match.barred == ["b"]
    assert match.excluded == []  # b was neither a candidate nor a near miss


# --- F6: contract and terms_policy are told apart by what only each carries -------------

COMMERCIAL_CLAUSES = (
    "contract.limitation_of_liability",
    "contract.warranties",
    "contract.termination",
    "contract.intellectual_property",
    "contract.governing_law_disputes",
    "contract.general_provisions",
    "contract.definitions",
)


def builtin_rules():
    return doc_type_rules(builtin_taxonomy("doc_types"))


@pytest.mark.parametrize("clause", COMMERCIAL_CLAUSES)
def test_ordinary_commercial_clauses_never_make_a_terms_policy(clause):
    result = match_rules(builtin_rules(), {clause: 0.95, "contract.payment": 0.95})
    assert "legal.terms_policy" not in result.candidates


def test_a_contract_needs_no_parties_or_signatures_heading():
    """An agreement whose preamble is untitled and whose signature block is no heading."""
    tags = {
        "contract.definitions": 0.9,
        "contract.scope_services": 0.9,
        "contract.payment": 0.9,
        "contract.term": 0.9,
        "contract.warranties": 0.9,
        "contract.limitation_of_liability": 0.9,
        "contract.governing_law_disputes": 0.9,
    }
    result = match_rules(builtin_rules(), tags)
    assert result.candidates == ["legal.contract"] and result.decided == "legal.contract"


def test_terms_policy_needs_a_section_only_published_terms_carry():
    rules = builtin_rules()
    privacy = match_rules(rules, {"contract.privacy_data": 0.9, "general.other": 0.9})
    assert privacy.candidates == ["legal.terms_policy"]
    terms = match_rules(
        rules,
        {
            "contract.acceptable_use": 0.9,
            "contract.limitation_of_liability": 0.9,
            "contract.warranties": 0.9,
        },
    )
    # liability and warranty clauses also make a contract candidate: the model chooses
    assert set(terms.candidates) == {"legal.terms_policy", "legal.contract"}
    assert terms.candidates[0] == "legal.terms_policy" and terms.decided is None
    named = match_rules(rules, {"contract.acceptable_use": 0.9, "contract.parties_recitals": 0.9})
    # named parties argue against published terms: kept only as a waived, never decided type
    assert named.waived == ["legal.terms_policy"] and named.decided is None


def test_acceptable_use_covers_the_user_terms_of_published_terms():
    leaf = builtin_taxonomy("section_catalogue").node("contract.acceptable_use")
    text = leaf.description.lower()
    for clause in ("acceptance", "account", "change"):
        assert clause in text
    assert {"Acceptance of Terms", "Changes to These Terms"} <= set(leaf.examples)


# --- F6 re-check: other rules a generic or borrowed section could meet alone --------------


def test_financial_statements_alone_do_not_make_an_earnings_release():
    rules = builtin_rules()
    statements = {"financial.income_statement": 0.9, "financial.balance_sheet": 0.9}
    assert "financial.earnings_release" not in match_rules(rules, statements).candidates
    release = {**statements, "financial.financial_highlights": 0.9, "narrative.outlook": 0.9}
    assert "financial.earnings_release" in match_rules(rules, release).candidates
    audited = match_rules(rules, {**release, "financial.notes_to_financial_statements": 0.9})
    assert audited.decided is None and "financial.earnings_release" in audited.barred


def test_a_generic_body_section_does_not_make_a_report():
    result = match_rules(builtin_rules(), {"general.body": 0.95, "general.conclusion": 0.9})
    assert "government.government_report" not in result.candidates
    assert "research.white_paper" not in result.candidates
    findings = match_rules(builtin_rules(), {"general.findings": 0.9, "general.body": 0.9})
    assert "government.government_report" in findings.candidates


def test_filing_items_bar_an_investor_presentation():
    tags = {
        "narrative.management_discussion": 0.9,
        "disclosure.forward_looking_statements": 0.9,
        "disclosure.controls_and_procedures": 0.9,
    }
    filing = match_rules(builtin_rules(), tags)
    assert filing.decided is None and "corporate.investor_presentation" in filing.barred
    deck = {"narrative.outlook": 0.9, "disclosure.forward_looking_statements": 0.9}
    assert "corporate.investor_presentation" in match_rules(builtin_rules(), deck).candidates


def test_a_financial_report_with_an_esg_chapter_is_no_sustainability_report():
    tags = {
        "esg.climate_environment": 0.9,
        "esg.social": 0.9,
        "narrative.management_discussion": 0.9,
    }
    result = match_rules(builtin_rules(), tags)
    assert result.decided is None and "corporate.sustainability_report" in result.barred
    esg_only = {"esg.climate_environment": 0.9, "esg.social": 0.9}
    assert match_rules(builtin_rules(), esg_only).candidates == ["corporate.sustainability_report"]


def test_highlights_of_a_non_financial_document_are_a_summary():
    leaf = builtin_taxonomy("section_catalogue").node("financial.financial_highlights")
    assert "general.summary" in leaf.description
