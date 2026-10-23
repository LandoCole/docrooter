"""TaggedDocument: a section's text by tag, from a hand-built document and classification.

Model-free: the document comes from ``Document.from_sections`` and the classification is built
by hand, one ``SectionTag`` per heading.
"""

from __future__ import annotations

import json

import pytest

import docrooter.context as context_module
from docrooter import Classification, SectionTag
from docrooter.context import SectionContext, SectionMatch, TaggedDocument
from docrooter.document import Document
from docrooter.taxonomies import builtin_taxonomy
from docrooter.views import estimate_tokens

BALANCE = "financial.balance_sheet"
STATEMENTS = "financial.financial_statements"
NOTES = "financial.notes_to_financial_statements"
RISKS = "disclosure.risk_factors"
TABLE = "| | 2025 | 2024 |\n|---|---|---|\n| Cash | 10 | 8 |"

# (title, level, label, body, confidence, page)
TENK = [
    ("PART II", 1, "none", "", 0.99, 40),
    ("Item 8. Financial Statements", 2, STATEMENTS, "Statements follow.", 0.9, 44),
    ("CONSOLIDATED BALANCE SHEETS", 3, BALANCE, TABLE, 0.97, 45),
    ("Current assets", 4, BALANCE, "Cash and receivables.", 0.8, 46),
    ("Notes", 2, NOTES, "The notes.", 0.9, 48),
    ("Note 1", 3, NOTES, "Basis of presentation.", 0.9, 48),
    ("Note 4", 3, NOTES, "Revenue.", 0.9, 49),
    ("Note A", 2, NOTES, "First separate note.", 0.9, 50),
    ("Note B", 2, NOTES, "Second separate note.", 0.9, 51),
    ("Item 1A Risk Factors", 1, RISKS, "Risks.", None, 60),
]


def build(rows, *, paged=True, **classification):
    document = Document.from_sections(
        [{"title": t, "level": lv, "text": body} for t, lv, _, body, _, _ in rows], source="doc.pdf"
    )
    if paged:
        document = document.model_copy(
            update={
                "sections": {
                    sid: section.model_copy(update={"page_start": rows[i][5]})
                    for i, (sid, section) in enumerate(document.sections.items())
                }
            }
        )
    tags = [
        SectionTag(
            id=sid,
            title=title,
            level=level,
            label=label,
            confidence=confidence,
            level_used="headings",
            confidence_source="text" if confidence is None else "probability",
        )
        for sid, (title, level, label, _, confidence, _) in zip(
            document.sections, rows, strict=True
        )
    ]
    fields = {
        "source": "doc.pdf",
        "flow": "doc_first",
        "doc_type": "financial.annual_report",
        "decided_by": "model",
        "backend": "test",
        "section_taxonomy": "section_catalogue",
        "sections": tags,
    } | classification
    return document, Classification(**fields)


@pytest.fixture
def tagged():
    document, classification = build(TENK)
    return TaggedDocument(document=document, classification=classification)


def test_get_section_returns_heading_and_full_text(tagged):
    context = tagged.get_section(BALANCE)
    assert isinstance(context, SectionContext)
    assert context.status == "found"
    assert context.message is None
    [match] = context.sections
    assert isinstance(match, SectionMatch)
    assert match.heading == "CONSOLIDATED BALANCE SHEETS"
    assert match.text.startswith(TABLE)
    assert "#### Current assets" in match.text
    assert "Cash and receivables." in match.text
    assert "CONSOLIDATED BALANCE SHEETS" not in match.text
    assert match.path == ["PART II", "Item 8. Financial Statements"]
    assert (match.section_id, match.tag, match.level) == ("s1.1.1", BALANCE, 3)
    assert (match.confidence, match.confidence_source) == (0.97, "probability")
    assert match.tokens == estimate_tokens(match.heading + "\n\n" + match.text)


def test_nested_matches_collapse(tagged):
    context = tagged.get_section(NOTES)
    notes = context.sections[0]
    assert notes.heading == "Notes"
    assert "### Note 1" in notes.text and "Basis of presentation." in notes.text
    assert "### Note 4" in notes.text and "Revenue." in notes.text
    assert [m.heading for m in context.sections].count("Note 1") == 0


def test_same_tag_siblings_stay_separate(tagged):
    headings = [m.heading for m in tagged.get_section(NOTES).sections]
    assert headings == ["Notes", "Note A", "Note B"]


def test_group_tag_matches_leaves_under_it(tagged):
    context = tagged.get_section("financial")
    assert context.status == "found"
    assert [(m.heading, m.tag) for m in context.sections] == [
        ("Item 8. Financial Statements", STATEMENTS),
        ("Notes", NOTES),
        ("Note A", NOTES),
        ("Note B", NOTES),
    ]
    assert "CONSOLIDATED BALANCE SHEETS" in context.sections[0].text  # inside Item 8, not repeated


def test_missing_tag_is_not_found(tagged):
    context = tagged.get_section("financial.schedule_of_investments")
    assert context.status == "not_found"
    assert context.sections == []
    assert "financial.schedule_of_investments" in context.message


def test_unknown_tag(tagged):
    context = tagged.get_section("financial.balence_sheet")
    assert context.status == "unknown_tag"
    assert context.sections == []
    assert "financial.balence_sheet" in context.message
    assert "section_catalogue" in context.message


def test_tag_is_stripped_but_case_exact(tagged):
    stripped = tagged.get_section(f"  {BALANCE}\n")
    assert stripped.status == "found"
    assert stripped.tag == BALANCE
    wrong_case = tagged.get_section("Financial.Balance_Sheet")
    assert wrong_case.status == "unknown_tag"


def test_get_sections_keeps_order_and_duplicates(tagged):
    contexts = tagged.get_sections([RISKS, BALANCE, RISKS])
    assert [c.tag for c in contexts] == [RISKS, BALANCE, RISKS]
    assert all(c.status == "found" for c in contexts)


def test_get_sections_rejects_a_bare_string(tagged):
    with pytest.raises(TypeError):
        tagged.get_sections(BALANCE)


def test_get_all_sections_excludes_none(tagged):
    contexts = tagged.get_all_sections()
    assert [c.tag for c in contexts] == [STATEMENTS, BALANCE, NOTES, RISKS]
    assert "none" not in [c.tag for c in contexts]
    assert tagged.available_tags() == [c.tag for c in contexts]
    # each context is complete on its own, so Item 8's text is repeated inside it
    assert "CONSOLIDATED BALANCE SHEETS" in contexts[0].sections[0].text


def test_min_confidence_drops_low_keeps_unknown():
    rows = [
        ("Overview", 1, "narrative.business_overview", "Strong.", 0.2, 1),
        ("Risks", 1, RISKS, "Many.", None, 2),
    ]
    document, classification = build(rows)
    tagged = TaggedDocument(document=document, classification=classification)
    assert tagged.get_section("narrative.business_overview").status == "found"
    dropped = tagged.get_section("narrative.business_overview", min_confidence=0.5)
    assert dropped.status == "not_found"
    assert dropped.sections == []
    assert "0.5" in dropped.message
    kept = tagged.get_section(RISKS, min_confidence=0.5)
    assert kept.status == "found"
    assert kept.sections[0].confidence is None
    assert [c.status for c in tagged.get_all_sections(min_confidence=0.5)] == [
        "not_found",
        "found",
    ]


def test_min_confidence_filters_before_collapsing(tagged):
    # The outer balance sheet is below the bar, so the inner section is not hidden by it.
    rows = [
        (*row[:4], 0.1 if row[0] == "CONSOLIDATED BALANCE SHEETS" else row[4], row[5])
        for row in TENK
    ]
    document, classification = build(rows)
    low = TaggedDocument(document=document, classification=classification)
    [match] = low.get_section(BALANCE, min_confidence=0.5).sections
    assert match.heading == "Current assets"


def test_page_end_is_next_section_start(tagged):
    [balance] = tagged.get_section(BALANCE).sections
    assert (balance.page_start, balance.page_end) == (45, 48)  # next section: Notes
    [risks] = tagged.get_section(RISKS).sections
    assert (risks.page_start, risks.page_end) == (60, 60)  # last section: its own subtree
    [statements] = tagged.get_section(STATEMENTS).sections
    assert (statements.page_start, statements.page_end) == (44, 48)


def test_page_end_without_a_following_section_is_the_last_page_in_the_subtree():
    rows = [
        ("Item 8", 1, STATEMENTS, "Body.", 0.9, 10),
        ("Balance", 2, BALANCE, "Cash.", 0.9, 12),
    ]
    document, classification = build(rows)
    tagged = TaggedDocument(document=document, classification=classification)
    [match] = tagged.get_section(STATEMENTS).sections
    assert (match.page_start, match.page_end) == (10, 12)


def test_pages_are_none_without_pages():
    document, classification = build(TENK, paged=False)
    tagged = TaggedDocument(document=document, classification=classification)
    for context in tagged.get_all_sections():
        for match in context.sections:
            assert (match.page_start, match.page_end) == (None, None)


def test_deep_levels_cap_at_six_hashes():
    rows = [
        ("Item 8", 1, STATEMENTS, "Body.", 0.9, 1),
        ("Deep", 8, "none", "Deep body.", 0.9, 1),
    ]
    document, classification = build(rows)
    tagged = TaggedDocument(document=document, classification=classification)
    [match] = tagged.get_section(STATEMENTS).sections
    assert "\n###### Deep\n" in f"\n{match.text}\n"
    assert "####### Deep" not in match.text


def test_text_skips_empty_bodies_and_is_stripped():
    rows = [
        ("Item 8", 1, STATEMENTS, "", 0.9, 1),
        ("Empty", 2, "none", "", 0.9, 1),
        ("Full", 2, "none", "Words.\n", 0.9, 1),
    ]
    document, classification = build(rows)
    tagged = TaggedDocument(document=document, classification=classification)
    [match] = tagged.get_section(STATEMENTS).sections
    assert match.text == "## Empty\n\n## Full\n\nWords."


def test_mismatched_classification_raises(tagged):
    stray = SectionTag(id="s99", title="x", level=1, label=BALANCE, level_used="headings")
    classification = tagged.classification.model_copy(
        update={"sections": [*tagged.classification.sections, stray]}
    )
    with pytest.raises(ValueError, match="s99"):
        TaggedDocument(document=tagged.document, classification=classification)


def test_empty_document():
    document = Document(source="empty.md")
    classification = Classification(
        source="empty.md",
        flow="doc_first",
        doc_type=None,
        decided_by="fallback",
        backend="test",
        section_taxonomy="section_catalogue",
    )
    tagged = TaggedDocument(document=document, classification=classification)
    assert tagged.doc_type is None
    assert tagged.available_tags() == []
    assert tagged.get_all_sections() == []
    assert tagged.get_section(BALANCE).status == "not_found"


@pytest.mark.parametrize("name", [None, "my_custom"])
def test_unresolvable_taxonomy_reports_not_found(name):
    rows = [
        ("Item 8", 1, "custom.statements", "Body.", 0.9, 1),
        ("Balance", 2, "custom.statements.balance", "Cash.", 0.9, 2),
    ]
    document, classification = build(rows, section_taxonomy=name)
    tagged = TaggedDocument(document=document, classification=classification)
    assert tagged.get_section("whatever.nothing").status == "not_found"
    assert tagged.get_section("custom.statements.balance").status == "found"
    group = tagged.get_section("custom")  # a prefix still reaches the leaves under it
    assert [m.heading for m in group.sections] == ["Item 8"]


def test_passed_taxonomy_enables_unknown_tag():
    rows = [("Item 8", 1, "custom.statements", "Body.", 0.9, 1)]
    document, classification = build(rows, section_taxonomy="my_custom")
    catalogue = builtin_taxonomy("section_catalogue")
    tagged = TaggedDocument(document=document, classification=classification, taxonomy=catalogue)
    assert tagged.get_section("financial.balence_sheet").status == "unknown_tag"
    assert tagged.get_section(BALANCE).status == "not_found"
    by_name = TaggedDocument(
        document=document, classification=classification, taxonomy="section_catalogue"
    )
    assert by_name.get_section("financial.balence_sheet").status == "unknown_tag"


def test_taxonomy_is_not_a_serialised_field(tagged):
    assert set(tagged.model_dump()) == {"document", "classification"}


def test_doc_type_property(tagged):
    assert tagged.doc_type == "financial.annual_report"


def test_results_are_json_ready(tagged):
    dumped = tagged.get_section(BALANCE).model_dump(mode="json")
    assert dumped["status"] == "found"
    assert list(dumped["sections"][0]) == [
        "section_id",
        "tag",
        "heading",
        "path",
        "level",
        "text",
        "page_start",
        "page_end",
        "confidence",
        "confidence_source",
        "tokens",
    ]


def test_none_is_a_real_leaf(tagged):
    context = tagged.get_section("none")
    assert context.status == "found"
    assert [m.heading for m in context.sections] == ["PART II"]


def test_non_dict_validation_context_is_ignored(tagged):
    tagged.model_post_init("not a dict")
    assert tagged.get_section(BALANCE).status == "found"
    assert tagged.get_section("financial.balence_sheet").status == "unknown_tag"


def test_json_round_trip(tagged):
    raw = tagged.to_json()
    envelope = json.loads(raw)
    assert set(envelope) == {"format", "version", "document", "classification"}
    assert envelope["format"] == "docrooter.tagged_document"
    assert envelope["version"] == 1
    again = TaggedDocument.from_json(raw)
    assert again.get_all_sections() == tagged.get_all_sections()
    assert TaggedDocument.from_json(raw.encode("utf-8")).doc_type == tagged.doc_type
    # the taxonomy comes back from the classification, so an unknown tag is still told apart
    assert again.get_section("financial.balence_sheet").status == "unknown_tag"


def test_from_json_takes_a_taxonomy():
    rows = [("Item 8", 1, "custom.statements", "Body.", 0.9, 1)]
    document, classification = build(rows, section_taxonomy="my_custom")
    raw = TaggedDocument(document=document, classification=classification).to_json()
    plain = TaggedDocument.from_json(raw)
    assert plain.get_section("financial.balence_sheet").status == "not_found"
    given = TaggedDocument.from_json(raw, taxonomy=builtin_taxonomy("section_catalogue"))
    assert given.get_section("financial.balence_sheet").status == "unknown_tag"


def test_save_load_round_trip_unicode(tmp_path):
    rows = [("Bilan consolidé", 1, BALANCE, "Trésorerie : 10 €.", 0.9, 1)]
    document, classification = build(rows)
    path = tmp_path / "doc.tagged.json"
    TaggedDocument(document=document, classification=classification).save(path)
    assert "Bilan consolidé" in path.read_text(encoding="utf-8")
    loaded = TaggedDocument.load(path)
    [match] = loaded.get_section(BALANCE).sections
    assert match.heading == "Bilan consolidé"
    assert "Trésorerie : 10 €." in match.text
    assert TaggedDocument.load(str(path)).doc_type == loaded.doc_type


@pytest.mark.parametrize(
    "change", [{"format": "other"}, {"version": 2}, {"version": "1"}, {"format": None}]
)
def test_from_json_rejects_wrong_format(tagged, change):
    envelope = json.loads(tagged.to_json()) | change
    with pytest.raises(ValueError, match="tagged document"):
        TaggedDocument.from_json(json.dumps(envelope))


@pytest.mark.parametrize("raw", ["[]", "{}", "not json"])
def test_from_json_rejects_a_non_envelope(raw):
    with pytest.raises(ValueError):
        TaggedDocument.from_json(raw)


def test_from_source_loads_and_classifies(monkeypatch, tagged):
    seen = {}

    def fake_classify(document, **kwargs):
        seen["document"], seen["kwargs"] = document, kwargs
        return build([("Overview", 1, "narrative.business_overview", "Strong.", 0.9, 1)])[1]

    monkeypatch.setattr(context_module, "classify", fake_classify)
    result = TaggedDocument.from_source("# Overview\n\nStrong.\n", backend="jev", flow="doc_first")
    assert seen["kwargs"] == {"backend": "jev", "flow": "doc_first"}
    assert isinstance(seen["document"], Document)
    assert result.document is seen["document"]
    assert result.get_section("narrative.business_overview").status == "found"

    def no_load(source):
        raise AssertionError("a Document must not be loaded again")

    monkeypatch.setattr(context_module, "load", no_load)
    monkeypatch.setattr(context_module, "classify", lambda document, **kw: tagged.classification)
    again = TaggedDocument.from_source(tagged.document, taxonomy="section_catalogue")
    assert again.document is tagged.document
    assert again.get_section("financial.balence_sheet").status == "unknown_tag"
