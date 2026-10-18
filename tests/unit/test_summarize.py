import pytest

from docrooter import Document
from docrooter.elements.models import Element, TableData
from docrooter.summarize import Summary, split_sentences, summarize, table_description


def _doc(*sections: tuple[str, str], elements: list[Element] | None = None) -> Document:
    doc = Document.from_sections([{"title": t, "level": 1, "text": x} for t, x in sections])
    doc.elements = elements or []
    return doc


def _para(eid: str, text: str, section_id: str | None = "s1", kind: str = "paragraph") -> Element:
    return Element(
        id=eid, kind=kind, text=text, boxes=[], order=int(eid[1:]), section_id=section_id
    )


@pytest.mark.parametrize(
    "sentence",
    [
        "Apple Inc. reported record revenue for the quarter.",
        "See Exhibit No. 5 for the full agreement terms.",
        "The results follow U.S. GAAP in every material respect.",
        "Growth came from several lines, e.g. revenue from services.",
        "Fig. 3 shows the distribution of the residuals.",
        "Smith et al. found a significant effect of the dose.",
        "The fund was advised by J. P. Morgan during the offering.",
        "The ratio rose to 3.14 over the period measured.",
        "Section 1.2.3 applies to every subsidiary of the group.",
        "Wait... then the whole plan changed without any warning.",
    ],
)
def test_splitter_protects_abbreviations(sentence):
    assert split_sentences(f"{sentence} A second sentence follows here.") == [
        sentence,
        "A second sentence follows here.",
    ]
    assert split_sentences(sentence) == [sentence]


def test_splitter_basics():
    assert split_sentences("") == []
    assert split_sentences("   \n ") == []
    assert split_sentences("Is it so? Yes! It is.") == ["Is it so?", "Yes!", "It is."]
    assert split_sentences('He said "stop." Then he left.') == ['He said "stop."', "Then he left."]
    assert split_sentences("Revenue rose in 2024. The board agreed.") == [
        "Revenue rose in 2024.",
        "The board agreed.",
    ]
    assert split_sentences("Line one\nstill one. Line two.") == ["Line one still one.", "Line two."]
    assert split_sentences("First para.\n\nSecond para") == ["First para.", "Second para"]
    assert split_sentences("1. Buy milk 2. Sell cars") == ["1. Buy milk 2. Sell cars"]


def test_lead_takes_first_k():
    text = (
        "Alpha one is here today. Beta two follows after it. Gamma three comes next. "
        "Delta four ends the run."
    )
    out = summarize(_doc(("A", text)), method="lead", k=2)
    assert out["s1"].key_sentences == ["Alpha one is here today.", "Beta two follows after it."]
    assert out["s1"].method == "lead"


_FACILITY = [
    "The revolving credit facility matures in 2027 and carries a floating interest rate.",
    "The revolving credit facility matures in 2027 and carries a variable interest rate.",
    "Our offices are located in several buildings across the region today.",
    "Covenants under the revolving credit facility limit additional borrowing and dividends.",
]
_OTHERS = [
    ("Offices", "Our offices are located in several buildings across the region today."),
    ("People", "The company employs engineers and designers in many different countries."),
]


def test_centroid_prefers_central_sentence_and_dedups():
    doc = _doc(("Debt", " ".join(_FACILITY)), *_OTHERS)
    out = summarize(doc, k=2)["s1"]
    assert out.method == "centroid"
    # near-duplicates 0 and 1: only one survives; the picks come back in document order
    assert (_FACILITY[0] in out.key_sentences) != (_FACILITY[1] in out.key_sentences)
    assert _FACILITY[2] not in out.key_sentences
    assert _FACILITY[3] in out.key_sentences
    assert out.key_sentences == sorted(out.key_sentences, key=_FACILITY.index)


def test_centroid_heading_bonus():
    body = [
        "Shipping costs depend on the carrier chosen for each order placed.",
        "Taxes are charged on every purchase made through the website store.",
    ]
    # Without the bonus the two read alike; the title decides.
    for title, want in (("Shipping costs", 0), ("Taxes charged", 1)):
        doc = _doc((title, " ".join(body)), ("Other", "Unrelated words fill this section."))
        assert summarize(doc, k=1)["s1"].key_sentences == [body[want]]


def test_length_penalty():
    # Same content words in each pair (stopwords carry no weight), so only the penalty differs;
    # the earlier sentence would win on the position prior alone.
    ok = "Revenue growth was strong across segments."
    other = ("B", "Nothing else matters here at all.")
    short = "Revenue growth strong."  # 3 words
    assert summarize(_doc(("A", f"{short} {ok}"), other), k=1)["s1"].key_sentences == [ok]
    long = "Revenue growth was strong across segments " + "and the of " * 25  # 81 words
    assert len(long.split()) == 81
    assert summarize(_doc(("A", f"{long.strip()}. {ok}"), other), k=1)["s1"].key_sentences == [ok]
    just_long_enough = " ".join(long.split()[:80]) + "."
    got = summarize(_doc(("A", f"{just_long_enough} {ok}"), other), k=1)["s1"].key_sentences
    assert got == [just_long_enough]  # 80 words is not penalised


def test_summary_coverage():
    four = "One is here now. Two is here now. Three is here now. Four is here now."
    three = "One is here now. Two is here now. Three is here now."
    doc = _doc(
        ("Many", four),
        ("Two", "Just two sentences sit here. And this is the second."),
        ("One", "Only a single sentence lives in this section."),
        elements=[
            _para("e0", four),
            _para("e1", three),
            _para("e2", four, kind="list_item"),
            _para("e3", four, kind="heading"),
            _para("e4", four, kind="caption"),
        ],
    )
    out = summarize(doc)
    assert set(out) == {"s1", "s2", "e0", "e2"}
    assert isinstance(out["e0"], Summary)
    assert set(summarize(doc, element_min_sentences=3)) == {"s1", "s2", "e0", "e1", "e2"}


def test_element_summary_uses_its_section_heading_path():
    sents = [
        "Plain filler sentence about nothing in particular at all.",
        "The covenant package restricts additional debt incurrence for the borrower.",
        "Weather patterns changed noticeably during the second half of that year.",
        "Engineers designed several improvements for the older factory machines.",
    ]
    for title, want in (("Covenant package", sents[1]), ("Weather patterns", sents[2])):
        doc = _doc((title, "x. y."), ("Other", "Different text entirely here."))
        doc.elements = [_para("e0", " ".join(sents), section_id="s1")]
        assert summarize(doc, k=1)["e0"].key_sentences == [want]


def test_keyphrases_and_deep_path():
    doc = Document.from_sections(
        [
            {"title": "Top", "level": 1, "text": ""},
            {"title": "Mid", "level": 2, "text": ""},
            {
                "title": "Leaf",
                "level": 3,
                "text": "The interest rate swap hedges debt. The interest rate swap expires soon.",
            },
        ]
    )
    out = summarize(doc, keyphrases=3)["s1.1.1"]
    assert out.keyphrases[0] == "interest rate swap" and len(out.keyphrases) <= 3


def test_table_description_format():
    table = Element(
        id="e0",
        kind="table",
        text="",
        boxes=[],
        order=0,
        section_id=None,
        table=TableData(
            rows=[["Year", "Revenue", ""], ["2023", "5", "6"], ["2024", "7"]], header_rows=1
        ),
    )
    assert table_description(table) == "Table: Year | Revenue; 3×3"
    assert table_description(table, caption="Table 2. Revenue by year") == (
        "Table: Table 2. Revenue by year; 3×3"
    )


def test_deterministic():
    text = " ".join(_FACILITY)
    doc = _doc(("Debt", text), *_OTHERS, elements=[_para("e0", text)])
    first = summarize(doc).copy()
    second = summarize(doc)
    assert first == second and list(first) == list(second)


def test_bad_method_rejected():
    with pytest.raises(ValueError, match="method"):
        summarize(_doc(("A", "x. y.")), method="magic")  # type: ignore[arg-type]
