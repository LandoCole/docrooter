import math
import random
from pathlib import Path

import pytest

from docrooter import Document, SectionView, load, views
from docrooter.views import estimate_tokens, render_budgeted

SIMPLE = Path(__file__).parent.parent / "fixtures" / "docs" / "simple.md"


@pytest.fixture
def doc():
    return load(SIMPLE)


@pytest.fixture
def long_doc():
    body = " ".join(f"word{i}" for i in range(200)) + "."
    return Document.from_sections([{"title": "Long", "level": 1, "text": body}])


def test_render_field_order_and_labels(doc):
    v = SectionView(fields=("path", "title", "text")).render_all(doc, ["s1.2"])["s1.2"]
    assert (
        v.text.splitlines()[0] == "Path: Intro > Scope" and v.text.splitlines()[1] == "Title: Scope"
    )
    assert v.text.splitlines()[2] == "Text: Scope body."
    assert not v.truncated


def test_budget_truncates_last_fitting_field(long_doc):
    v = SectionView(fields=("title", "text"), max_tokens=20).render_all(long_doc, ["s1"])["s1"]
    assert v.truncated and v.n_tokens <= 20
    assert v.text.splitlines()[0] == "Title: Long" and v.text.splitlines()[1].startswith(
        "Text: word0"
    )


def test_n_tokens_uses_custom_counter(long_doc):
    v = SectionView(fields=("title", "text"), max_tokens=10).render_all(
        long_doc, ["s1"], count_tokens=lambda t: len(t.split())
    )["s1"]
    assert v.truncated and v.n_tokens == len(v.text.split()) <= 10


def test_budget_too_small_for_next_field_stops(long_doc):
    v = SectionView(fields=("title", "text"), max_tokens=3).render_all(long_doc, ["s1"])["s1"]
    assert v.truncated and v.text == "Title: Long" and v.n_tokens == estimate_tokens(v.text)


def test_heading_only_leaf_renders_title():
    d = Document.from_sections([{"title": "Signatures", "level": 1}])
    v = SectionView(fields=("text",)).render_all(d, ["s1"])["s1"]
    assert v.text.strip() != "" and "Signatures" in v.text


def test_empty_parent_uses_children_titles():
    d = Document.from_sections(
        [{"title": "Part II", "level": 1}, {"title": "Item 7", "level": 2, "text": "x"}]
    )
    assert "Subsections: Item 7" in SectionView(fields=("text",)).render_all(d, ["s1"])["s1"].text


def test_empty_parent_title_mode():
    d = Document.from_sections(
        [{"title": "Part II", "level": 1}, {"title": "Item 7", "level": 2, "text": "x"}]
    )
    v = SectionView(fields=("text",), empty_section="title").render_all(d, ["s1"])["s1"]
    assert v.text == "Text: Part II"


def test_lead_and_keyphrases():
    d = Document.from_sections(
        [
            {"title": "A", "level": 1, "text": "First one. Second one! Third one? Fourth one."},
            {"title": "B", "level": 1, "text": "Unrelated filler content."},
        ]
    )
    v = SectionView(fields=("keyphrases", "lead"), lead_sentences=2).render_all(d, ["s1"])["s1"]
    lines = v.text.splitlines()
    assert lines[0].startswith("Key phrases: ") and lines[1] == "Lead: First one. Second one!"


def test_keyphrases_line_omitted_when_none():
    d = Document.from_sections([{"title": "A", "level": 1}])
    v = SectionView(fields=("title", "keyphrases")).render_all(d, ["s1"])["s1"]
    assert v.text == "Title: A"


def test_path_skips_preamble_and_unknown_id_raises():
    d = Document.from_sections(
        [{"title": "Pre", "level": 0, "text": "p"}, {"title": "H", "level": 1}]
    )
    pre = d.root_ids[0]
    view = SectionView(fields=("path",))
    assert view.render_all(d, [pre])[pre].text == "Path: Preamble"
    with pytest.raises(KeyError):
        view.render_all(d, ["nope"])


def test_unknown_field_rejected():
    with pytest.raises(ValueError):
        SectionView(fields=("title", "bogus"))


def test_deep_document_path():
    entries = [{"title": f"T{i}", "level": min(i + 1, 6), "text": "x"} for i in range(30)]
    d = Document.from_sections(entries)
    last = list(d.sections)[-1]
    out = SectionView(fields=("path",)).render_all(d, [last])[last].text
    assert out.startswith("Path: T0 > ") and out.endswith("T29")


@pytest.fixture
def signatures():
    return Document.from_sections([{"title": "Signatures", "level": 1}])


@pytest.mark.parametrize("fields", [("keyphrases",), ("lead",), ("keyphrases", "lead")])
def test_view_never_empty_when_fields_render_nothing(signatures, fields):
    v = SectionView(fields=fields).render_all(signatures, ["s1"])["s1"]
    assert v.text == "Title: Signatures" and not v.truncated


def test_view_never_empty_under_tiny_budget(signatures):
    v = SectionView(fields=("text",), max_tokens=1).render_all(signatures, ["s1"])["s1"]
    assert v.text == "Signatures" and v.truncated


def test_title_fallback_cut_to_budget_keeps_first_word():
    d = Document.from_sections([{"title": "Alpha Beta Gamma Delta Epsilon", "level": 1}])
    view = SectionView(fields=("lead",), max_tokens=4)
    v = view.render_all(d, ["s1"])["s1"]
    assert v.truncated and v.text == "Title: Alpha Beta" and v.n_tokens <= 4
    assert SectionView(fields=("lead",), max_tokens=1).render_all(d, ["s1"])["s1"].text == "Alpha"


def test_multiline_earlier_field_does_not_confuse_truncation():
    d = Document.from_sections(
        [{"title": "one two three four five six seven eight", "level": 1, "text": "l1\nl2\nl3"}]
    )
    v = SectionView(fields=("text", "title"), max_tokens=14).render_all(d, ["s1"])["s1"]
    assert v.truncated and v.n_tokens <= 14
    assert v.text.splitlines()[:3] == ["Text: l1", "l2", "l3"]
    assert v.text.splitlines()[3].startswith("Title: one two")
    assert v.text.count("l2") == 1


def _recording_word_counter(seen: list[int]):
    def count(text: str) -> int:
        seen.append(len(text.split()))
        return len(text.split())

    return count


def test_long_text_is_cut_before_the_counter_sees_it():
    body = " ".join(f"w{i}" for i in range(20_000))
    d = Document.from_sections([{"title": "Long", "level": 1, "text": body}])
    seen: list[int] = []
    v = SectionView(fields=("title", "text"), max_tokens=10).render_all(
        d, ["s1"], count_tokens=_recording_word_counter(seen)
    )["s1"]
    # same view as without the cut: the longest fitting word prefix
    assert v.text == "Title: Long\nText: " + " ".join(f"w{i}" for i in range(7))
    assert v.truncated and v.n_tokens == 10
    assert max(seen) <= 4 * 10 + 3  # never the whole section: at most ~4 words per token


def test_long_title_fallback_is_cut_before_the_counter_sees_it():
    title = " ".join(f"t{i}" for i in range(20_000))
    d = Document.from_sections([{"title": title, "level": 1}])
    seen: list[int] = []
    v = SectionView(fields=("lead",), max_tokens=4).render_all(
        d, ["s1"], count_tokens=_recording_word_counter(seen)
    )["s1"]
    assert v.text == "Title: t0 t1 t2" and v.truncated
    assert max(seen) <= 4 * 4 + 1


def test_cut_text_that_fits_is_still_marked_truncated():
    body = " ".join(f"w{i}" for i in range(1000))
    d = Document.from_sections([{"title": "Long", "level": 1, "text": body}])
    v = SectionView(fields=("text", "title"), max_tokens=5).render_all(
        d,
        ["s1"],
        count_tokens=lambda text: 1,  # a counter no real text can exceed
    )["s1"]
    assert v.truncated
    assert v.text == "Text: " + " ".join(f"w{i}" for i in range(20))  # 4 x 5 words, then stop


def _subword_count(text: str) -> int:
    return sum(math.ceil(len(word) / 3) for word in text.split())


@pytest.mark.parametrize(
    "count", [estimate_tokens, lambda t: len(t.split()), _subword_count], ids=["est", "w", "sub"]
)
def test_precut_matches_uncut_rendering_on_random_inputs(monkeypatch, count):
    """Differential: the word cap never changes a view (every counter costs >= 1/4 token per
    word), checked against the same function with the cap disabled."""
    rng = random.Random(1016)
    vocab = ["a", "of", "model", "results", "x1", "Item", "7A.", "supercalifragilistic", "été"]

    def random_text(max_words: int) -> str:
        words = [rng.choice(vocab) for _ in range(rng.randint(0, max_words))]
        return rng.choice([" ", "  ", "\n"]).join(words)

    cases = []
    for _ in range(300):
        entries = [
            None if rng.random() < 0.1 else (rng.choice(["Title:", "Lead:", ""]), random_text(60))
            for _ in range(rng.randint(0, 4))
        ]
        cases.append((entries, random_text(40), rng.randint(1, 12)))
    capped = [render_budgeted(e, t, m, count) for e, t, m in cases]
    monkeypatch.setattr(views, "_WORDS_PER_TOKEN", 10**9)
    uncapped = [render_budgeted(e, t, m, count) for e, t, m in cases]
    assert capped == uncapped
    over_cap = sum(
        any(e is not None and len(e[1].split()) > 4 * m for e in entries) for entries, _, m in cases
    )
    assert over_cap > 100  # the cap was exercised, not idle
