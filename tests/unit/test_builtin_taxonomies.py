import pytest

import docrooter as dr
from docrooter.taxonomies import (
    DOC_TYPE_TO_SECTION_TAXONOMY,
    builtin_taxonomy,
    list_builtin_taxonomies,
)

SECTION_NAMES = [
    "section_types/annual_report",
    "section_types/research_paper",
    "section_types/contract",
    "section_types/technical_document",
    "section_types/generic",
]


def test_list_builtin_taxonomies():
    assert list_builtin_taxonomies() == sorted(["doc_types", *SECTION_NAMES])


def test_exported_from_package():
    assert dr.builtin_taxonomy is builtin_taxonomy


@pytest.mark.parametrize("name", ["doc_types", *SECTION_NAMES])
def test_every_builtin_loads_with_descriptions(name):
    taxonomy = builtin_taxonomy(name)
    assert len(taxonomy) > 0
    for node in taxonomy.nodes.values():
        assert node.description and node.description.strip(), node.id
        assert node.label.strip(), node.id


def test_doc_types_has_18_leaves_in_6_groups():
    taxonomy = builtin_taxonomy("doc_types")
    assert len(taxonomy.leaves()) == 18
    assert len(taxonomy.children()) == 6


def test_doc_types_matches_benchmark_corpus_taxonomy():
    from pathlib import Path

    corpus = Path(__file__).resolve().parents[2] / "benchmarks" / "corpus" / "doc_types.json"
    if not corpus.exists():
        pytest.skip("benchmark corpus not present")
    assert builtin_taxonomy("doc_types").content_hash() == dr.Taxonomy.load(corpus).content_hash()


@pytest.mark.parametrize("name", SECTION_NAMES)
def test_section_taxonomies_are_flat_slugs(name):
    taxonomy = builtin_taxonomy(name)
    assert taxonomy.depth() == 1
    for node_id in taxonomy.nodes:
        assert node_id == node_id.lower()
        assert node_id.replace("_", "").isalnum(), node_id


def test_doc_type_mapping_covers_every_leaf_and_resolves():
    leaves = {leaf.id for leaf in builtin_taxonomy("doc_types").leaves()}
    assert set(DOC_TYPE_TO_SECTION_TAXONOMY) == leaves
    for section_taxonomy in DOC_TYPE_TO_SECTION_TAXONOMY.values():
        assert section_taxonomy in SECTION_NAMES
        builtin_taxonomy(section_taxonomy)


def test_doc_type_mapping_families():
    m = DOC_TYPE_TO_SECTION_TAXONOMY
    for leaf in (
        "financial.annual_report",
        "financial.quarterly_report",
        "financial.proxy_statement",
    ):
        assert m[leaf] == "section_types/annual_report"
    for leaf, value in m.items():
        if leaf.startswith("research."):
            assert value == "section_types/research_paper"
        if leaf.startswith("technical."):
            assert value == "section_types/technical_document"
    assert m["legal.contract"] == "section_types/contract"
    assert m["legal.legislation"] == "section_types/generic"
    assert m["financial.prospectus"] == "section_types/generic"


@pytest.mark.parametrize(
    "name", ["nope", "section_types/nope", "other/generic", "../doc_types", ""]
)
def test_unknown_name_raises_keyerror_listing_options(name):
    with pytest.raises(KeyError, match="doc_types"):
        builtin_taxonomy(name)
