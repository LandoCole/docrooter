"""parse(): section text and the ParseResult, from a hand-built document and classification.

Model-free: the document comes from ``Document.from_sections`` and the classification is built
by hand, one ``SectionTag`` per heading; the end-to-end tests use the offline keyword backend.
"""

from __future__ import annotations

import json

import pytest

import docrooter.context as context_module
from docrooter import Classification, SectionTag, load
from docrooter.backends.base import TokenUsage
from docrooter.context import (
    INCLUDES,
    ParseResult,
    SectionContext,
    SectionMatch,
    TaggedSection,
    parse,
    section_text,
    tagged_sections,
)
from docrooter.document import Document
from docrooter.taxonomy import Taxonomy, TaxonomyNode
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
def tenk():
    return build(TENK)


def text_of(rows, **classification):
    document, cls = build(rows, **classification)
    return section_text(document, cls)


def result_of(rows=TENK, include="text", **classification):
    document, cls = build(rows, **classification)
    return ParseResult(
        source=cls.source,
        include=include,
        backend=cls.backend,
        doc_type=cls.doc_type,
        doc_type_confidence=cls.doc_type_confidence,
        tokens={"document": 0, "sections": 0},
        sections=None if include == "type" else tagged_sections(document, cls),
        section_text=section_text(document, cls) if include == "text" else None,
    )


def test_section_heading_and_full_text(tenk):
    texts = section_text(*tenk)
    [match] = texts[BALANCE]
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


def test_nested_matches_collapse(tenk):
    matches = section_text(*tenk)[NOTES]
    assert matches[0].heading == "Notes"
    assert "### Note 1" in matches[0].text and "Basis of presentation." in matches[0].text
    assert "### Note 4" in matches[0].text and "Revenue." in matches[0].text
    assert "Note 1" not in [m.heading for m in matches]


def test_same_tag_siblings_stay_separate(tenk):
    assert [m.heading for m in section_text(*tenk)[NOTES]] == ["Notes", "Note A", "Note B"]


def test_section_text_keys(tenk):
    texts = section_text(*tenk)
    # reading order of each tag's first section; the "none" heading is not a key
    assert list(texts) == [STATEMENTS, BALANCE, NOTES, RISKS]
    # Item 8 holds the balance sheet, which is also under its own key (D4)
    assert "CONSOLIDATED BALANCE SHEETS" in texts[STATEMENTS][0].text
    assert texts[BALANCE][0].text.startswith(TABLE)


def test_tagged_sections_lists_every_tag(tenk):
    rows = tagged_sections(*tenk)
    assert all(isinstance(row, TaggedSection) for row in rows)
    assert [(r.heading, r.tag) for r in rows] == [(t, label) for t, _, label, *_ in TENK]
    part, item8, balance, current = rows[:4]
    assert (part.section_id, part.path, part.confidence, part.page_start) == ("s1", [], 0.99, 40)
    assert item8.path == ["PART II"]
    assert (balance.section_id, balance.level) == ("s1.1.1", 3)
    assert current.path == [
        "PART II",
        "Item 8. Financial Statements",
        "CONSOLIDATED BALANCE SHEETS",
    ]
    assert rows[-1].confidence is None


def test_tagged_sections_skip_untagged_labels():
    rows = [("A", 1, None, "a", 0.5, 1), ("B", 1, "narrative.business_overview", "b", 0.5, 2)]
    document, cls = build(rows)
    assert [r.heading for r in tagged_sections(document, cls)] == ["B"]
    assert list(section_text(document, cls)) == ["narrative.business_overview"]


def test_page_end_is_next_section_start(tenk):
    texts = section_text(*tenk)
    [balance] = texts[BALANCE]
    assert (balance.page_start, balance.page_end) == (45, 48)  # next section: Notes
    [risks] = texts[RISKS]
    assert (risks.page_start, risks.page_end) == (60, 60)  # last section: its own subtree
    [statements] = texts[STATEMENTS]
    assert (statements.page_start, statements.page_end) == (44, 48)


def test_page_end_without_a_following_section_is_the_last_page_in_the_subtree():
    rows = [
        ("Item 8", 1, STATEMENTS, "Body.", 0.9, 10),
        ("Balance", 2, BALANCE, "Cash.", 0.9, 12),
    ]
    [match] = text_of(rows)[STATEMENTS]
    assert (match.page_start, match.page_end) == (10, 12)


def test_pages_are_none_without_pages():
    document, cls = build(TENK, paged=False)
    for matches in section_text(document, cls).values():
        for match in matches:
            assert (match.page_start, match.page_end) == (None, None)
    assert {r.page_start for r in tagged_sections(document, cls)} == {None}


def test_deep_levels_cap_at_six_hashes():
    rows = [
        ("Item 8", 1, STATEMENTS, "Body.", 0.9, 1),
        ("Deep", 8, "none", "Deep body.", 0.9, 1),
    ]
    [match] = text_of(rows)[STATEMENTS]
    assert "\n###### Deep\n" in f"\n{match.text}\n"
    assert "####### Deep" not in match.text


def test_text_skips_empty_bodies_and_is_stripped():
    rows = [
        ("Item 8", 1, STATEMENTS, "", 0.9, 1),
        ("Empty", 2, "none", "", 0.9, 1),
        ("Full", 2, "none", "Words.\n", 0.9, 1),
    ]
    [match] = text_of(rows)[STATEMENTS]
    assert match.text == "## Empty\n\n## Full\n\nWords."


def test_mismatched_classification_raises(tenk):
    document, cls = tenk
    stray = SectionTag(id="s99", title="x", level=1, label=BALANCE, level_used="headings")
    cls = cls.model_copy(update={"sections": [*cls.sections, stray]})
    with pytest.raises(ValueError, match="s99"):
        section_text(document, cls)
    with pytest.raises(ValueError, match="s99"):
        tagged_sections(document, cls)


def test_empty_document():
    document = Document(source="empty.md")
    cls = Classification(
        source="empty.md",
        flow="doc_first",
        doc_type=None,
        decided_by="fallback",
        backend="test",
        section_taxonomy="section_catalogue",
    )
    assert section_text(document, cls) == {}
    assert tagged_sections(document, cls) == []


# --- parse() ---------------------------------------------------------------------------------

REPORT = """# Notes to the financial statements

Note one sets out the accounting policies and revenue recognition.

# Management discussion

Results of operations improved as revenue grew.

## Liquidity

Cash flows from operations funded capital expenditure.
"""


@pytest.fixture
def spy(monkeypatch):
    calls = []
    real = context_module.classify

    def spied(document, **kwargs):
        calls.append(kwargs)
        return real(document, **kwargs)

    monkeypatch.setattr(context_module, "classify", spied)
    return calls


def test_parse_type_only_skips_sections(spy):
    result = parse(REPORT, include="type", backend="keyword", level="headings")
    assert len(spy) == 1
    assert spy[0]["flow"] == "doc_first" and spy[0]["max_sections"] == 0
    assert result.include == "type"
    assert result.sections is None and result.section_text is None
    raw = json.loads(result.model_dump_json())
    assert "sections" not in raw and "section_text" not in raw
    assert "doc_type_confidence" in raw


def test_parse_levels_end_to_end_keyword():
    results = {
        level: parse(REPORT, include=level, backend="keyword", level="headings")
        for level in INCLUDES
    }
    assert len({r.doc_type for r in results.values()}) == 1
    assert results["type"].tokens["sections"] == 0
    assert results["type"].tokens["document"] > 0
    assert results["sections"].tokens["sections"] > 0
    assert [s.heading for s in results["sections"].sections] == [
        "Notes to the financial statements",
        "Management discussion",
        "Liquidity",
    ]
    assert results["sections"].section_text is None
    full = results["text"]
    assert full.sections == results["sections"].sections
    assert full.section_text
    texts = [m.text for matches in full.section_text.values() for m in matches]
    assert any("accounting policies" in t for t in texts)
    assert any("Cash flows from operations" in t for t in texts)
    assert (full.version, full.backend, full.include) == (1, "keyword", "text")


def test_parse_takes_a_document_and_a_path(tmp_path, spy):
    path = tmp_path / "report.md"
    path.write_text(REPORT, encoding="utf-8")
    from_path = parse(path, backend="keyword", level="headings")
    from_doc = parse(load(path), backend="keyword", level="headings")
    assert from_path.sections == from_doc.sections
    assert from_path.source == from_doc.source
    assert spy[0]["flow"] == "doc_first"  # the default flow


def test_parse_passes_classify_kwargs(spy):
    parse(REPORT, backend="keyword", level="headings", calls="per_item", flow="sections_first")
    assert spy[0]["calls"] == "per_item"
    assert spy[0]["flow"] == "sections_first"
    assert spy[0]["backend"] == "keyword" and spy[0]["level"] == "headings"
    assert "max_sections" not in spy[0]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"include": "everything"}, "include"),
        ({"include": "type", "flow": "sections_first"}, "doc_first"),
        ({"include": "type", "max_sections": 5}, "max_sections"),
    ],
)
def test_parse_rejects_conflicts(spy, kwargs, message):
    with pytest.raises(ValueError, match=message):
        parse(REPORT, backend="keyword", **kwargs)
    assert spy == []  # refused before any call


def test_parse_allows_max_sections_above_type():
    result = parse(REPORT, include="sections", backend="keyword", level="headings", max_sections=1)
    assert [s.section_id for s in result.sections] == ["s1"]  # max_sections reached classify


def test_stage_tokens_are_prompt_plus_completion(monkeypatch):
    document, cls = build(TENK)
    cls = cls.model_copy(update={"stage_tokens": {"document": TokenUsage(prompt=10, completion=2)}})
    monkeypatch.setattr(context_module, "classify", lambda doc, **kw: cls)
    result = parse(document, include="sections")
    assert result.tokens == {"document": 12, "sections": 0}


@pytest.mark.parametrize("include", INCLUDES)
def test_parse_result_round_trip(include):
    rows = [
        ("Bilan consolidé", 1, BALANCE, "Trésorerie : 10 €.", 0.9, 1),
        ("Annexe", 1, NOTES, "Détail.", None, 2),
    ]
    result = result_of(rows, include=include)
    raw = result.model_dump_json()
    assert "Bilan consolidé" in raw or include == "type"  # not escaped
    again = ParseResult.model_validate_json(raw)
    assert again == result
    assert (
        set(json.loads(raw))
        == {
            "type": {
                "version",
                "source",
                "include",
                "backend",
                "doc_type",
                "doc_type_confidence",
                "tokens",
            },
            "sections": {
                "version",
                "source",
                "include",
                "backend",
                "doc_type",
                "doc_type_confidence",
                "tokens",
                "sections",
            },
            "text": {
                "version",
                "source",
                "include",
                "backend",
                "doc_type",
                "doc_type_confidence",
                "tokens",
                "sections",
                "section_text",
            },
        }[include]
    )


def test_parse_result_json_has_the_documented_shape():
    dumped = json.loads(result_of().model_dump_json())
    assert list(dumped["sections"][0]) == [
        "section_id",
        "heading",
        "level",
        "path",
        "tag",
        "confidence",
        "page_start",
    ]
    assert list(dumped["section_text"][BALANCE][0]) == [
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


def test_get_section_found_missing_unknown():
    result = result_of()
    context = result.get_section(BALANCE)
    assert isinstance(context, SectionContext)
    assert context.status == "found" and context.message is None
    assert [m.heading for m in context.sections] == ["CONSOLIDATED BALANCE SHEETS"]

    missing = result.get_section("financial.schedule_of_investments")
    assert missing.status == "not_found" and missing.sections == []
    assert "financial.schedule_of_investments" in missing.message

    unknown = result.get_section("financial.balence_sheet")
    assert unknown.status == "unknown_tag" and unknown.sections == []
    assert "financial.balence_sheet" in unknown.message
    assert "section_catalogue" in unknown.message

    stripped = result.get_section(" financial.balance_sheet \n")
    assert stripped.status == "found" and stripped.tag == BALANCE
    assert result.get_section("Financial.Balance_Sheet").status == "unknown_tag"

    custom = Taxonomy(
        [
            TaxonomyNode(id="custom", label="Custom"),
            TaxonomyNode(id="custom.a", label="A", parent_id="custom"),
        ]
    )
    assert result.get_section(BALANCE, catalogue=custom).status == "unknown_tag"
    assert result.get_section("custom.a", catalogue=custom).status == "not_found"


def test_get_section_none_is_never_a_key():
    result = result_of()
    assert "none" not in result.section_text
    assert result.get_section("none").status == "not_found"


def test_get_section_group_collapses_across_keys():
    result = result_of()
    context = result.get_section("financial")
    assert context.status == "found"
    # the balance sheet sits inside Item 8, whose text already holds it
    assert [(m.heading, m.tag) for m in context.sections] == [
        ("Item 8. Financial Statements", STATEMENTS),
        ("Notes", NOTES),
        ("Note A", NOTES),
        ("Note B", NOTES),
    ]
    assert "CONSOLIDATED BALANCE SHEETS" in context.sections[0].text


def test_group_matches_in_reading_order_not_key_order():
    rows = [
        ("Overview", 1, "narrative.business_overview", "o", 0.9, 1),
        ("A", 1, NOTES, "a", 0.9, 2),
        ("B", 1, STATEMENTS, "b", 0.9, 3),
        ("C", 1, NOTES, "c", 0.9, 4),
    ]
    context = result_of(rows).get_section("financial")
    assert [m.heading for m in context.sections] == ["A", "B", "C"]  # not A, C, B


def test_reading_order_compares_ids_numerically():
    rows = [("Top", 1, "narrative.business_overview", "t", 0.9, 1)]
    rows += [(f"Sub {i}", 2, NOTES if i % 2 else STATEMENTS, str(i), 0.9, 1) for i in range(1, 12)]
    context = result_of(rows).get_section("financial")
    assert [m.heading for m in context.sections] == [f"Sub {i}" for i in range(1, 12)]


def test_get_sections_order_and_duplicates():
    contexts = result_of().get_sections([RISKS, BALANCE, RISKS])
    assert [c.tag for c in contexts] == [RISKS, BALANCE, RISKS]
    assert all(c.status == "found" for c in contexts)


def test_get_sections_rejects_a_bare_string():
    with pytest.raises(TypeError):
        result_of().get_sections(BALANCE)


@pytest.mark.parametrize("include", ["type", "sections"])
def test_get_section_needs_text_level(include):
    result = result_of(include=include)
    with pytest.raises(ValueError, match='include="text"'):
        result.get_section(BALANCE)
    with pytest.raises(ValueError, match='include="text"'):
        result.get_sections([BALANCE])


def test_a_stored_response_answers_after_a_round_trip():
    stored = result_of().model_dump_json()
    again = ParseResult.model_validate_json(stored)
    [match] = again.get_section(RISKS).sections
    assert (match.heading, match.text) == ("Item 1A Risk Factors", "Risks.")
