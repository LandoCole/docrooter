"""v1 section catalogue: one shared section taxonomy, grouped by domain, that tags a section
without knowing the document type."""

import re

from docrooter.taxonomies import builtin_taxonomy, list_builtin_taxonomies

FAMILIES = (
    "annual_report",
    "fund_document",
    "contract",
    "research_paper",
    "technical_document",
    "generic",
)
FINANCIAL_LEAVES = (
    "financial.balance_sheet",
    "financial.income_statement",
    "financial.cash_flow_statement",
    "financial.changes_in_equity",
    "financial.schedule_of_investments",
    "financial.auditors_report",
    "financial.notes_to_financial_statements",
    "financial.account_statement",
)
MAX_OPTIONS = 26  # the OpenAI-compatible backend's letter markers, A-Z


def test_catalogue_loads_and_every_group_has_other():
    assert "section_catalogue" in list_builtin_taxonomies()
    catalogue = builtin_taxonomy("section_catalogue")
    assert catalogue.depth() == 2
    groups = catalogue.children()
    assert 2 <= len(groups) <= MAX_OPTIONS
    for group in groups:
        leaf_ids = [leaf.id for leaf in catalogue.children(group.id)]
        assert f"{group.id}.other" in leaf_ids, group.id
        assert len(leaf_ids) <= MAX_OPTIONS, group.id
        assert group.description and group.description.strip(), group.id
        for leaf_id in leaf_ids:
            assert re.fullmatch(rf"{group.id}\.[a-z][a-z_]*", leaf_id), leaf_id

    seen: dict[str, str] = {}
    for leaf in catalogue.leaves():
        assert leaf.description and leaf.description.strip(), leaf.id
        assert 3 <= len(leaf.examples) <= 8, leaf.id
        for example in leaf.examples:
            assert example.strip() == example and example, leaf.id
            # one heading string never names two leaves of the catalogue
            assert seen.setdefault(example.casefold(), leaf.id) == leaf.id, example

    for leaf_id in FINANCIAL_LEAVES:
        assert leaf_id in catalogue, leaf_id
    letter = catalogue.node("narrative.shareholder_letter")
    assert "investor" in letter.label.lower()

    text = " ".join(n.description or "" for n in catalogue.nodes.values()).lower()
    for british in ("licence", "capitalised", "organised", "behaviour", "recognis"):
        assert british not in text


def test_catalogue_covers_every_family_leaf():
    """Each per-family section leaf maps to exactly one catalogue leaf (``metadata["from"]``),
    so doc-first labels can be scored against catalogue tags."""
    catalogue = builtin_taxonomy("section_catalogue")
    mapped: dict[str, str] = {}
    for leaf in catalogue.leaves():
        for source in leaf.metadata.get("from", []):
            assert source not in mapped, source
            mapped[source] = leaf.id
    for group in catalogue.children():
        assert "from" not in group.metadata, group.id
    expected = {
        f"{family}.{leaf.id}"
        for family in FAMILIES
        for leaf in builtin_taxonomy(f"section_types/{family}").leaves()
    }
    assert set(mapped) == expected
    # a leaf that keeps its family meaning keeps its id under the new group
    for source, target in mapped.items():
        if not source.endswith(".other"):
            family_leaf = source.split(".", 1)[1]
            if family_leaf != "principal_risks":
                assert target.split(".", 1)[1] == family_leaf, (source, target)


def test_family_section_taxonomies_are_unchanged():
    for family in FAMILIES:
        taxonomy = builtin_taxonomy(f"section_types/{family}")
        assert taxonomy.depth() == 1
        assert all(node.sections is None for node in taxonomy.nodes.values())
