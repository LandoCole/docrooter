from pathlib import Path

import pytest

from docrooter.document import RawHeading, extract_headings
from docrooter.document.headings import repair_levels, score_heading, validate_headings

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "docs"


def read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def make(*titles: str, level: int = 1) -> list[RawHeading]:
    return [RawHeading(title=t, level=level, text="", line=i + 1) for i, t in enumerate(titles)]


@pytest.mark.parametrize(
    "title,ok",
    [
        ("1.2 Liquidity and Capital Resources", True),
        ("RISK FACTORS", True),
        ("Appendix A", True),
        ("Page 3", False),
        ("3 of 12", False),
        ("https://example.com/x", False),
        (
            "This sentence is clearly a paragraph that someone formatted as a heading by mistake.",
            False,
        ),
        ("- bullet item", False),
        ("| a | b |", False),
    ],
)
def test_score_heading(title, ok):
    assert (score_heading(title)[0] >= 0.5) is ok


def test_score_in_unit_interval_with_reasons():
    for title in ["", "-", "1.1.1 A", "x" * 500, "RISK FACTORS"]:
        score, reasons = score_heading(title)
        assert 0.0 <= score <= 1.0 and isinstance(reasons, list)
    assert score_heading("1.2 Liquidity and Capital Resources")[0] == pytest.approx(0.95)
    assert score_heading("RISK FACTORS")[0] == pytest.approx(0.7)


def test_score_other_penalties():
    assert score_heading("Contact: jane@example.com")[0] < 0.5
    assert score_heading("Overview of the plan,")[0] < 0.5
    assert score_heading("Item 7")[0] >= 0.7
    long_title = " ".join(["Word"] * 21)
    assert score_heading(long_title)[0] < 0.5


def test_repeated_title_penalized():
    assert score_heading("ACME Corp Annual Report", repeat_count=5)[0] < 0.5
    assert score_heading("ACME Corp Annual Report", repeat_count=2)[0] >= 0.5


def test_validate_merges_rejected_into_previous():
    out = validate_headings(extract_headings(read("noisy_headings.md")))
    titles = [h.title for h in out]
    assert "Page 3" not in titles and "- bullet item" not in titles
    assert titles == ["1 Overview", "1.1 Scope", "2 Results"]
    assert "Page 3" in out[0].text and "Running page footer debris." in out[0].text
    assert out[0].text.startswith("Overview body text.")
    assert "- bullet item" in out[1].text and "Stray list marker" in out[1].text


def test_validate_scores_and_low_confidence_flag():
    raw = make("1 Intro", "something plainly lowercase here")
    out = validate_headings(raw, min_score=0.5)
    assert out[0].heading_score == pytest.approx(0.95) and out[0].flags == []
    assert out[1].heading_score == pytest.approx(0.6)
    assert out[1].flags == ["low_heading_confidence"]
    assert raw[1].flags == []  # input untouched


def test_validate_repeated_titles_rejected():
    raw = [
        RawHeading(title="ACME Corp Annual Report", level=1, text=f"t{i}", line=i) for i in range(4)
    ]
    out = validate_headings(raw)
    assert [(h.title, h.level) for h in out] == [("Preamble", 0)]
    assert out[0].text.count("ACME Corp Annual Report") == 4


def test_validate_first_block_rejected_becomes_preamble():
    raw = [
        RawHeading(title="Page 1", level=1, text="footer", line=1),
        RawHeading(title="Real Heading", level=1, text="body", line=3),
    ]
    out = validate_headings(raw)
    assert [(h.title, h.level) for h in out] == [("Preamble", 0), ("Real Heading", 1)]
    assert out[0].text == "Page 1\nfooter"


def test_validate_leaves_preamble_untouched():
    raw = [
        RawHeading(title="Preamble", level=0, text="lead", line=1),
        RawHeading(title="Page 2", level=1, text="x", line=3),
        RawHeading(title="Intro", level=1, text="body", line=5),
    ]
    out = validate_headings(raw)
    assert out[0].level == 0 and out[0].heading_score == 1.0 and out[0].flags == []
    assert out[0].text == "lead\nPage 2\nx"
    assert [h.title for h in out] == ["Preamble", "Intro"]


def test_repair_numbering_overrides_hashes():
    raw = [
        RawHeading(title=t, level=2, text="", line=i)
        for i, t in enumerate(["1 Intro", "1.1 Scope", "1.1.1 Detail", "2 Next"])
    ]
    assert [h.level for h in repair_levels(raw)] == [1, 2, 3, 1]


def test_repair_closes_gaps():
    raw = [
        RawHeading(title="A", level=1, text="", line=0),
        RawHeading(title="B", level=4, text="", line=1),
    ]
    out = repair_levels(raw)
    assert [h.level for h in out] == [1, 2] and "level_repaired" in out[1].flags
    assert out[0].flags == [] and raw[1].level == 4  # input untouched


def test_repair_skips_numbering_below_half():
    raw = make("1.1 Numbered", "Plain one", "Plain two", level=1)
    assert [h.level for h in repair_levels(raw)] == [1, 1, 1]


def test_repair_leaves_preamble_alone():
    raw = [
        RawHeading(title="Preamble", level=0, text="lead", line=1),
        RawHeading(title="A", level=3, text="", line=2),
    ]
    out = repair_levels(raw)
    assert [h.level for h in out] == [0, 1] and out[0].flags == []
    assert "level_repaired" in out[1].flags


def test_repair_deep_input_is_iterative():
    raw = [RawHeading(title=f"{i} H", level=6, text="", line=i) for i in range(1, 31)]
    assert [h.level for h in repair_levels(raw)] == [1] * 30


def test_roman_numeral_prefix_gets_numbering_bonus():
    score, reasons = score_heading("IV. Results")
    assert score >= 0.85
    assert "numbering_prefix" in reasons
    assert "numbering_prefix" not in score_heading("Civilization")[1]


def test_numbering_prefix_is_case_insensitive():
    assert "numbering_prefix" in score_heading("ITEM 7. MANAGEMENT'S DISCUSSION")[1]
    assert "numbering_prefix" in score_heading("ARTICLE IV")[1]


def test_title_case_tokens_keep_apostrophes_and_hyphens():
    assert "title_case" in score_heading("Company's Results Year-over-Year")[1]
    assert "title_case" in score_heading("Management’s Discussion")[1]
    assert "title_case" not in score_heading("the company's results over time")[1]
