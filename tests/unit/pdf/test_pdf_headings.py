import math

from docrooter.pdf import Line, compute_signals
from docrooter.pdf.headings import HeadingCandidate, find_headings

H = 792.0
BODY = "This ordinary body sentence carries the running text of the section along"
BODY_GAP = 3.0  # leading between lines of a paragraph
PARA_GAP = 10.0  # gap between paragraphs
HEAD_GAP = 16.0  # space above a heading


def item(text, *, size=10.0, bold=False, gap=BODY_GAP):
    return (text, size, bold, gap)


def para(n=3, *, size=10.0, text=BODY):
    """``n`` lines of body text; the first opens a new paragraph."""
    return [item(text, size=size, gap=PARA_GAP if k == 0 else BODY_GAP) for k in range(n)]


def layout(pages):
    """Lay out ``pages`` (lists of ``(text, size, bold, gap_above)``) top to bottom as ``Line``s.

    A gap of at least ``PARA_GAP`` starts a new block, as pymupdf would for a new paragraph.
    """
    lines = []
    for page_no, items in enumerate(pages, start=1):
        y = 60.0
        block = -1
        page_lines = []
        for k, (text, size, bold, gap) in enumerate(items):
            if k == 0 or gap >= PARA_GAP:
                block += 1
            y0 = y if k == 0 else y + gap
            height = size * 1.2
            page_lines.append(dict(
                page=page_no, text=text, size=size, bold=bold, font="Helvetica",
                x0=72.0, y0=y0, x1=540.0, y1=y0 + height, page_height=H, page_width=612.0,
                gap_above=math.inf if k == 0 else gap, gap_below=math.inf, block=block,
            ))  # fmt: skip
            y = y0 + height
        for a, b in zip(page_lines, page_lines[1:], strict=False):
            a["gap_below"] = b["gap_above"]
        lines += [Line(**row) for row in page_lines]
    return lines


def run(pages, *, bookmarks=None, **kwargs) -> tuple[list[Line], list[HeadingCandidate]]:
    lines = layout(pages)
    signals = compute_signals(lines, len(pages), bookmarks=bookmarks)
    return lines, find_headings(lines, signals, **kwargs)


def titles(found):
    return [(h.text, h.level) for h in found]


def test_two_heading_sizes_give_levels_one_and_two():
    pages = [
        [item("Introduction", size=18, bold=True), *para(4),
         item("Background", size=14, bold=True, gap=HEAD_GAP), *para(3),
         item("Prior Work", size=14, bold=True, gap=HEAD_GAP), *para(3)],
        [item("Methods", size=18, bold=True), *para(4),
         item("Data Collection", size=14, bold=True, gap=HEAD_GAP), *para(5)],
    ]  # fmt: skip
    _, found = run(pages)
    assert titles(found) == [
        ("Introduction", 1), ("Background", 2), ("Prior Work", 2), ("Methods", 1),
        ("Data Collection", 2),
    ]  # fmt: skip
    assert [h.page for h in found] == [1, 1, 1, 2, 2]
    assert all(0.55 <= h.score <= 1.0 for h in found)


def test_bold_ranks_before_regular_at_equal_size():
    pages = [
        [item("Overview", size=16, bold=True), *para(4),
         item("Market Context", size=16, gap=HEAD_GAP), *para(3),
         item("Outlook", size=16, bold=True, gap=HEAD_GAP), *para(3),
         item("Risks Ahead", size=16, gap=HEAD_GAP), *para(3)],
    ]  # fmt: skip
    _, found = run(pages)
    assert titles(found) == [
        ("Overview", 1), ("Market Context", 2), ("Outlook", 1), ("Risks Ahead", 2),
    ]  # fmt: skip


def test_equal_font_size_numbered_isolated_lines_get_levels_from_numbering():
    pages = [
        [item("1 Introduction", gap=HEAD_GAP), *para(4),
         item("2 Background", gap=HEAD_GAP), *para(3),
         item("2.1 Scope", gap=HEAD_GAP), *para(3)],
        [item("2.2 Terminology", gap=HEAD_GAP), *para(3),
         item("3 Protocol Overview", gap=HEAD_GAP), *para(4)],
    ]  # fmt: skip
    _, found = run(pages)
    assert titles(found) == [
        ("1 Introduction", 1), ("2 Background", 1), ("2.1 Scope", 2), ("2.2 Terminology", 2),
        ("3 Protocol Overview", 1),
    ]  # fmt: skip


def test_slide_deck_titles_are_headings_and_bullets_are_not():
    bullets = [
        "• Revenue grew strongly across every region we serve",
        "• Operating margin expanded for the third straight quarter",
        "• Strong Growth in Cloud Subscriptions",
        "• Free cash flow reached a record level this year",
    ]
    pages = []
    for n in range(1, 31):
        slide = [item(f"Quarterly Highlights Part {n}", size=28)]
        slide += [item(b, size=18, gap=12.0) for b in bullets]
        slide += para(4, size=14)
        pages.append(slide)
    _, found = run(pages)
    assert len(found) <= 40
    assert not any(h.text.startswith("•") for h in found)
    assert [h.text for h in found] == [f"Quarterly Highlights Part {n}" for n in range(1, 31)]
    assert {h.level for h in found} == {1}


def _toc_page():
    entries = ["Executive Summary", "Overview of results", "Financial Statements",
               "Governance", "Appendix"]  # fmt: skip
    return [item("Contents", size=14, bold=True)] + [
        item(f"{t} " + "." * 20 + f" {p}", gap=6.0) for p, t in enumerate(entries, start=2)
    ]


def test_printed_toc_match_boosts_borderline_candidate():
    borderline = item("Overview of results", bold=True, gap=HEAD_GAP)  # sentence case, body size
    pages = [_toc_page(), [*para(3), borderline, *para(4)]]
    _, found = run(pages)
    assert [h.text for h in found] == ["Overview of results"]
    _, without = run(pages, use_printed_toc=False)
    assert without == []


def test_printed_toc_page_lines_are_never_headings():
    _, found = run([_toc_page(), para(5)])
    assert found == []


def test_bookmark_match_sets_level_and_flag():
    pages = [
        [item("Introduction", size=18, bold=True), *para(4),
         item("Scope", size=14, bold=True, gap=HEAD_GAP), *para(3)],
        [item("Terms Used", size=14, bold=True), *para(4),
         item("closing remarks", bold=True, gap=HEAD_GAP), *para(3)],
    ]  # fmt: skip
    bookmarks = [(1, "Introduction", 1), (1, "Terms Used", 2), (2, "Closing Remarks", 2)]
    _, found = run(pages, bookmarks=bookmarks)
    assert titles(found) == [
        ("Introduction", 1), ("Scope", 2), ("Terms Used", 1), ("closing remarks", 2),
    ]  # fmt: skip
    assert [("level_from_bookmark" in h.flags) for h in found] == [True, False, True, True]

    _, plain = run(pages, bookmarks=bookmarks, use_bookmarks=False)
    assert titles(plain) == [("Introduction", 1), ("Scope", 2), ("Terms Used", 2)]
    assert not any("level_from_bookmark" in h.flags for h in plain)


def test_bookmark_on_a_distant_page_does_not_match():
    pages = [[item("Introduction", size=18, bold=True), *para(4)], para(4), para(4)]
    _, found = run(pages, bookmarks=[(2, "Introduction", 3)])
    assert titles(found) == [("Introduction", 1)]
    assert "level_from_bookmark" not in found[0].flags


def test_two_line_title_is_merged():
    pages = [
        [item("Management's Discussion and Analysis of", size=18, bold=True),
         item("Financial Condition", size=18, bold=True, gap=2.0), *para(4),
         item("Results of Operations", size=14, bold=True, gap=HEAD_GAP), *para(4)],
    ]  # fmt: skip
    lines, found = run(pages)
    assert titles(found) == [
        ("Management's Discussion and Analysis of Financial Condition", 1),
        ("Results of Operations", 2),
    ]
    assert found[0].line_index == 0
    assert [h.line_count for h in found] == [2, 1]
    assert lines[found[1].line_index].text == "Results of Operations"


def test_wrapped_bold_paragraph_is_not_a_title():
    words = "Our Mission Is To Deliver Great Value To Every Customer We Serve Each Day"
    wrapped = [item(words, size=14, bold=True, gap=HEAD_GAP if k == 0 else BODY_GAP)
               for k in range(3)]  # fmt: skip
    pages = [[*para(3), *wrapped, *para(3)]]
    _, found = run(pages)
    assert found == []


def test_sentence_like_long_bold_line_is_rejected():
    sentence = " ".join(["word"] * 24) + " end."
    assert len(sentence.split()) == 25
    pages = [[*para(3), item(sentence, bold=True, gap=HEAD_GAP), *para(3)]]
    _, found = run(pages)
    assert found == []


def test_excluded_footer_lines_are_never_candidates():
    pages = []
    for n in range(1, 6):
        page = [item(f"Chapter Heading {n}", size=18, bold=True), *para(4)]
        pages.append(page)
    lines = layout(pages)
    footers = []
    for n in range(1, 6):  # a large bold running footer at the page bottom
        footers.append(lines[0].model_copy(update={
            "page": n, "text": "ACME CORPORATION CONFIDENTIAL", "y0": H - 40, "y1": H - 20,
            "gap_above": 300.0, "block": 9,
        }))  # fmt: skip
    ordered = sorted(lines + footers, key=lambda line: (line.page, line.y0))
    signals = compute_signals(ordered, 5)
    footer_idx = {i for i, line in enumerate(ordered) if line.text.startswith("ACME")}
    assert footer_idx <= signals.excluded
    found = find_headings(ordered, signals)
    assert [h.text for h in found] == [f"Chapter Heading {n}" for n in range(1, 6)]
    assert not {h.line_index for h in found} & footer_idx


def test_consecutive_same_style_candidates_are_penalised():
    pages = [
        [item("Key Figures", size=18, bold=True), *para(3),
         item("Revenue Summary", size=14, bold=True, gap=HEAD_GAP),
         item("Cost Summary", size=14, bold=True, gap=HEAD_GAP),
         item("Margin Summary", size=14, bold=True, gap=HEAD_GAP), *para(3)],
    ]  # fmt: skip
    _, found = run(pages, min_score=0.0)
    score = {h.text: h.score for h in found}
    # Only the last of the run is followed by body text; the others lose 0.15.
    assert score["Revenue Summary"] == score["Cost Summary"]
    assert math.isclose(score["Margin Summary"] - score["Revenue Summary"], 0.15, abs_tol=1e-3)
    _, kept = run(pages)
    assert [h.text for h in kept] == ["Key Figures", "Margin Summary"]


def test_min_score_filters_and_output_is_reading_order():
    pages = [
        [item("Alpha Section", size=18, bold=True), *para(3),
         item("Beta Section", size=14, bold=True, gap=HEAD_GAP), *para(3)],
    ]  # fmt: skip
    lines, found = run(pages, min_score=0.0)
    assert [h.line_index for h in found] == sorted(h.line_index for h in found)
    _, none = run(pages, min_score=1.01)
    assert none == []


def test_empty_input():
    signals = compute_signals([], 0)
    assert find_headings([], signals) == []


def test_tight_leading_equal_size_numbered_headings_do_not_swallow_their_body():
    pages = [
        [item("1 Introduction", gap=HEAD_GAP), *[item(BODY) for _ in range(4)],
         item("2 Background", gap=HEAD_GAP), *[item(BODY) for _ in range(3)],
         item("2.1 Scope", gap=HEAD_GAP), *[item(BODY) for _ in range(3)]],
        [item("2.2 Terminology", gap=HEAD_GAP), *[item(BODY) for _ in range(3)],
         item("3 Protocol Overview", gap=HEAD_GAP), *[item(BODY) for _ in range(4)]],
    ]  # fmt: skip
    _, found = run(pages)
    assert titles(found) == [
        ("1 Introduction", 1), ("2 Background", 1), ("2.1 Scope", 2), ("2.2 Terminology", 2),
        ("3 Protocol Overview", 1),
    ]  # fmt: skip


def test_short_body_line_under_a_heading_is_not_merged_into_it():
    pages = [[item("1 Introduction", gap=HEAD_GAP), item("See below."), *para(3)]]
    _, found = run(pages)
    assert [h.text for h in found] == ["1 Introduction"]


def test_title_on_the_same_row_as_its_number_is_merged():
    lines = layout([[*para(3), item("Item 1B.", bold=True, gap=HEAD_GAP), *para(3)]])
    number = lines[3]
    title = number.model_copy(update={"text": "Unresolved Staff Comments", "x0": 140.0,
                                      "gap_above": 0.0})  # fmt: skip
    lines = lines[:4] + [title] + lines[4:]
    found = find_headings(lines, compute_signals(lines, 1))
    assert [(h.text, h.line_index) for h in found] == [("Item 1B. Unresolved Staff Comments", 3)]


def test_number_only_lines_are_never_headings():
    numeric = ["12%", "21,268", "3.", "29,600 76% 50% 60%", "IV.", "(a)"]
    pages = [[*para(2)]]
    for text in numeric:
        pages[0] += [item(text, size=18, bold=True, gap=HEAD_GAP), *para(2)]
    pages[0] += [item("PART II", size=18, bold=True, gap=HEAD_GAP), *para(2),
                 item("IV. Results", size=18, bold=True, gap=HEAD_GAP), *para(2)]  # fmt: skip
    _, found = run(pages, min_score=0.0)
    assert [h.text for h in found] == ["PART II", "IV. Results"]


def _random_titles(rng, n):
    vocab = [f"{a}{b}" for a in ("risk", "cash", "tax", "plan", "data", "cost", "fund", "asset")
             for b in ("", "s", "ing", "ed", "ion", "al", "ity", "ment")]  # fmt: skip
    return [" ".join(rng.choice(vocab) for _ in range(rng.randint(2, 6))).title()
            for _ in range(n)]  # fmt: skip


def test_toc_matcher_agrees_with_is_fuzzy_match():
    import random

    from docrooter.pdf import is_fuzzy_match
    from docrooter.pdf.headings import _TocMatcher

    rng = random.Random(1)
    titles = _random_titles(rng, 60)
    probes = _random_titles(rng, 150) + [t[:-1] for t in titles[:30]] + titles[30:40]
    probes += ["", "12", "1.2 " + titles[0]]
    matcher = _TocMatcher(titles + [""])
    for probe in probes:
        assert matcher.matches(probe) == any(is_fuzzy_match(probe, t) for t in titles), probe


def test_printed_toc_matching_scales_to_long_tocs():
    import random
    import time

    from docrooter.pdf import DocSignals

    rng = random.Random(0)
    entries = [(t, str(k)) for k, t in enumerate(_random_titles(rng, 300), start=1)]
    pages = [[item(t, bold=True, gap=HEAD_GAP) for t in _random_titles(rng, 40)]
             for _ in range(50)]  # fmt: skip
    lines = layout(pages)
    signals = DocSignals(body_size=10.0, median_gap=BODY_GAP, printed_toc=entries)
    start = time.perf_counter()
    find_headings(lines, signals, min_score=0.0)
    assert time.perf_counter() - start < 2.0


def test_one_off_styles_do_not_consume_levels():
    # A cover title and a one-off front-matter style used once each must not push the
    # document's recurring chapter/section styles down to levels 3 and 4.
    pages = [
        [item("Annual Review", size=28, bold=True), *para(2),
         item("Letter From The Chair", size=18, bold=True, gap=HEAD_GAP), *para(4)],
    ]  # fmt: skip
    for n, name in enumerate(["Strategy", "Operations", "Finance", "Governance"], start=1):
        pages.append(
            [item(f"Chapter {name}", size=14, bold=True), *para(3),
             item(f"{name} Overview", size=12, bold=True, gap=HEAD_GAP), *para(3),
             item(f"{name} Outlook", size=12, bold=True, gap=HEAD_GAP), *para(3)]
        )  # fmt: skip
    _, found = run(pages)
    levels = dict(titles(found))
    assert levels["Annual Review"] == 1 and levels["Letter From The Chair"] == 1
    assert {levels[f"Chapter {n}"] for n in ["Strategy", "Finance"]} == {1}
    assert {levels[t] for t in ["Strategy Overview", "Finance Outlook"]} == {2}


def test_edge_band_copy_of_a_heading_is_not_a_heading():
    # A two-page section prints its title in the footer band of both pages (too few pages to
    # be a running line); the footer copies duplicate the real heading and are dropped.
    pages = [
        [item("Chapter One", size=18, bold=True), *para(4),
         item("2.2. Address Map", size=14, bold=True, gap=HEAD_GAP), *para(4)],
        [*para(5)],
        [item("Chapter Two", size=18, bold=True), *para(4)],
    ]  # fmt: skip
    lines = layout(pages)
    footers = [
        lines[0].model_copy(update={
            "page": n, "text": f"{n + 20} 2.2. Address Map", "size": 14.0, "bold": True,
            "y0": H - 40, "y1": H - 25, "gap_above": 300.0, "block": 9,
        })  # fmt: skip
        for n in (1, 2)
    ]
    ordered = sorted(lines + footers, key=lambda line: (line.page, line.y0))
    found = find_headings(ordered, compute_signals(ordered, 3))
    assert [(h.text, h.page) for h in found] == [
        ("Chapter One", 1), ("2.2. Address Map", 1), ("Chapter Two", 3),
    ]  # fmt: skip


def test_page_top_heading_listed_on_an_earlier_contents_page_is_kept():
    pages = [
        [item("Contents Overview", size=18, bold=True), *para(2),
         item("Item 3. Market Risk", size=14, bold=True, gap=HEAD_GAP), *para(2)],
        [*para(5)], [*para(5)], [*para(5)],
        [item("Item 3. Market Risk", size=14, bold=True), *para(4)],
    ]  # fmt: skip
    lines = layout(pages)
    top = next(i for i, line in enumerate(lines) if line.page == 5)
    lines[top] = lines[top].model_copy(update={"y0": 20.0, "y1": 37.0})  # in the top band
    found = find_headings(lines, compute_signals(lines, 5))
    assert ("Item 3. Market Risk", 5) in [(h.text, h.page) for h in found]
