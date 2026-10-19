"""v0.5 element taxonomies: the merged topic tree, the function tags, fund documents and the
fact-sheet doc type, and the optional ``keywords`` node field."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

from docrooter.classify.normalize import _section_taxonomy_name
from docrooter.taxonomies import (
    DOC_TYPE_TO_SECTION_TAXONOMY,
    ELEMENT_FAMILIES,
    builtin_taxonomy,
    family_for_doc_type,
)
from docrooter.taxonomy import Taxonomy, TaxonomyNode

REPO = Path(__file__).resolve().parents[3]
GENERATOR = REPO / "scripts" / "build_element_topics.py"
TAXONOMIES = REPO / "src" / "docrooter" / "taxonomies"

FUNCTION_TAGS = [
    "forward_looking",
    "risk_disclosure",
    "financial_data",
    "accounting_policy",
    "legal_boilerplate",
    "definition",
    "obligation",
    "procedure",
    "contact_or_admin",
    "signature",
    "citation",
    "marketing",
]

FUND_DOCUMENT_LEAVES = [
    "investment_objective",
    "fees_and_expenses",
    "principal_investment_strategies",
    "principal_risks",
    "performance",
    "holdings_and_allocation",
    "management",
    "purchase_and_sale",
    "tax_information",
    "other",
]

# content_hash() of every v0.4 built-in at afa07a3: trained bundles store it as taxonomy_hash,
# so adding the optional ``keywords`` field must leave each of them unchanged.
V04_HASHES = {
    "section_types/annual_report": (
        "7a8a0437979a7acb8752823c3fb976bd7977166a9512e33a6b37c6673334a2ed"
    ),
    "section_types/contract": "efd245333d655b8bce41d558257bbc6f295c9b00c0be997f91e383d0378b0e89",
    "section_types/generic": "3ad6d0d53c8efb4f8bace781c921608b7cba3b9190a3f947d5c37b46e922b3d9",
    "section_types/research_paper": (
        "89d028c87c27a707ed6e0765ce4f78c58361175930d6d206e37e78218539b6f9"
    ),
    "section_types/technical_document": (
        "25d514656f0f3585ac7e85c93c93344cd371be138b2e6bd0364055319f4966cf"
    ),
}
# doc_types.json at afa07a3; v0.5 adds financial.fact_sheet, so only the tree without it matches.
V04_DOC_TYPES_HASH = "bae09bbd9a31e78de2cc2ecff68b1e5401b0537cee3f7b836cc82957c839ea6a"

BRITISH = (
    *("licence", "capitalis", "organis", "behaviour", "recognis", "harbour", "normalis"),
    *("favour", "colour", "summaris", "characterise", "authoris", "utilis", "centre"),
)


def _generator():
    spec = importlib.util.spec_from_file_location("build_element_topics", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_element_topics_is_generated_from_section_taxonomies():
    generator = _generator()
    on_disk = (TAXONOMIES / "element_topics.json").read_text(encoding="utf-8")
    assert generator.render() == on_disk  # re-running the generator reproduces the file

    topics = builtin_taxonomy("element_topics")
    assert [n.id for n in topics.children()] == list(ELEMENT_FAMILIES)
    assert topics.depth() == 2
    for family in ELEMENT_FAMILIES:
        section = builtin_taxonomy(f"section_types/{family}")
        leaves = [n.id for n in topics.children(family)]
        expected = [f"{family}.{leaf}" for leaf in section.nodes]
        if f"{family}.other" not in expected:
            expected.append(f"{family}.other")
        assert leaves == expected, family
        for leaf_id, leaf in section.nodes.items():  # reused unchanged
            node = topics.node(f"{family}.{leaf_id}")
            assert (node.label, node.description, node.examples) == (
                leaf.label,
                leaf.description,
                leaf.examples,
            )


def test_generator_check_mode_flags_a_stale_file(tmp_path):
    generator = _generator()
    target = tmp_path / "element_topics.json"
    assert generator.main(["--check", "--output", str(target)]) == 1  # missing
    assert generator.main(["--output", str(target)]) == 0
    assert generator.main(["--check", "--output", str(target)]) == 0
    target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert generator.main(["--check", "--output", str(target)]) == 1


def test_every_node_has_description_and_function_tags_have_examples_and_not_clause():
    for name in ("element_topics", "element_functions", "section_types/fund_document"):
        taxonomy = builtin_taxonomy(name)
        for node in taxonomy.nodes.values():
            assert node.label.strip(), (name, node.id)
            assert node.description and node.description.strip(), (name, node.id)
            assert node.description == node.description.strip(), (name, node.id)
            text = " ".join([node.description, *node.examples]).lower()
            for british in BRITISH:
                assert british not in text, (name, node.id, british)

    functions = builtin_taxonomy("element_functions")
    assert list(functions.nodes) == FUNCTION_TAGS
    assert functions.depth() == 1
    assert functions.node(Taxonomy.ROOT_ID).multi_label is True
    for tag, node in functions.nodes.items():
        assert 3 <= len(node.examples) <= 6, tag
        assert len(set(node.examples)) == len(node.examples), tag
        # the definition ends with a "not this" clause naming what the tag excludes
        sentences = re.split(r"(?<=\.)\s+", node.description)
        assert any(s.startswith("Not ") for s in sentences[1:]), tag
        assert node.keywords == [], tag  # term mining fills these later, not the taxonomy

    topics = builtin_taxonomy("element_topics")
    for family in ELEMENT_FAMILIES:
        seen: dict[str, str] = {}
        for leaf in topics.children(family):
            assert 3 <= len(leaf.examples) <= 8, leaf.id
            for example in leaf.examples:  # one heading never names two leaves of a family
                assert seen.setdefault(example.casefold(), leaf.id) == leaf.id, example


def test_function_examples_are_not_copied_from_eval_documents():
    """The examples are written for the taxonomy: none may appear in the dev section gold."""
    gold = REPO / "benchmarks" / "corpus" / "section_gold.jsonl"
    if not gold.exists():
        pytest.skip("benchmark corpus not present")
    text = gold.read_text(encoding="utf-8").casefold()
    for node in builtin_taxonomy("element_functions").nodes.values():
        for example in node.examples:
            assert example.casefold() not in text, example


def test_fund_document_leaves_per_spec():
    fund = builtin_taxonomy("section_types/fund_document")
    assert list(fund.nodes) == FUND_DOCUMENT_LEAVES
    assert fund.depth() == 1
    labels = [n.label for n in fund.nodes.values()]
    assert len(set(labels)) == len(labels)
    seen: dict[str, str] = {}
    for node in fund.nodes.values():
        assert 3 <= len(node.examples) <= 8, node.id
        for example in node.examples:
            assert example.strip() == example and example, node.id
            assert seen.setdefault(example.casefold(), node.id) == node.id, example


def test_fact_sheet_doc_type_and_maps():
    doc_types = builtin_taxonomy("doc_types")
    fact_sheet = doc_types.node("financial.fact_sheet")
    assert fact_sheet.parent_id == "financial"
    assert fact_sheet.label == "Fact sheet"
    assert fact_sheet.description == (
        "A short (typically 1–4 page) periodic summary of a single fund or share class: "
        "objective, key facts, fees, performance table and chart, top holdings and allocation, "
        "as of a date. Not a full prospectus and not an annual report."
    )
    assert len(doc_types.leaves()) == 19
    for doc_type in ("financial.prospectus", "financial.fact_sheet"):
        assert DOC_TYPE_TO_SECTION_TAXONOMY[doc_type] == "section_types/fund_document"
        assert family_for_doc_type(doc_type) == "fund_document"

    # Apart from the new leaf, doc_types is the v0.4 tree, unchanged.
    v04 = Taxonomy([n for n in doc_types.nodes.values() if n.id != "financial.fact_sheet"])
    assert v04.content_hash() == V04_DOC_TYPES_HASH


def test_keywords_field_optional_and_omitted():
    for name, digest in V04_HASHES.items():
        taxonomy = builtin_taxonomy(name)
        assert taxonomy.content_hash() == digest, name
        for node in taxonomy.nodes.values():
            assert node.keywords == []
            assert "keywords" not in node.model_dump()
            assert "keywords" not in json.loads(node.model_dump_json())
        # the flat dump round-trips byte-identically, with no keywords key anywhere
        dumped = json.dumps(taxonomy.to_dict(), sort_keys=True)
        assert '"keywords"' not in dumped
        again = Taxonomy.from_dict(json.loads(dumped))
        assert json.dumps(again.to_dict(), sort_keys=True) == dumped
        assert again.content_hash() == digest

    plain = TaxonomyNode(id="a", label="A")
    tagged = TaxonomyNode(id="a", label="A", keywords=["net asset value", "nav"])
    assert "keywords" not in plain.model_dump()
    assert tagged.model_dump()["keywords"] == ["net asset value", "nav"]
    loaded = Taxonomy.from_dict({"nodes": [tagged.model_dump()]})
    assert loaded.node("a").keywords == ["net asset value", "nav"]
    assert loaded.content_hash() != Taxonomy([plain]).content_hash()  # keywords count when set


def test_family_for_doc_type_covers_all_doc_types():
    assert ELEMENT_FAMILIES == (
        "annual_report",
        "fund_document",
        "contract",
        "research_paper",
        "technical_document",
        "generic",
    )
    doc_types = builtin_taxonomy("doc_types")
    assert set(DOC_TYPE_TO_SECTION_TAXONOMY) == {leaf.id for leaf in doc_types.leaves()}
    used = set()
    for node_id in [*doc_types.nodes, "not.a.type", ""]:
        family = family_for_doc_type(node_id)
        assert family in ELEMENT_FAMILIES, node_id
        # the same section taxonomy normalize_sections picks for the doc type
        assert f"section_types/{family}" == _section_taxonomy_name(node_id or None, doc_types)
        used.add(family)
    assert used == set(ELEMENT_FAMILIES)
