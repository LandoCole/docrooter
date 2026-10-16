import math

from docrooter.pdf import Line, compute_signals, fuzzy_match, normalize_title

H = 800.0


def mk(page, text, *, size=11.0, y0=300.0, bold=False, gap_above=12.0, h=12.0, w=500.0):
    return Line(
        page=page, text=text, size=size, bold=bold, font="Helvetica", x0=72.0, y0=y0,
        x1=72.0 + w, y1=y0 + h, page_height=H, page_width=612.0, gap_above=gap_above,
        gap_below=12.0, block=0,
    )  # fmt: skip


def body_page(page, n=3, *, size=11.0):
    return [mk(page, f"Some ordinary body sentence number {k} here.", size=size,
               y0=200.0 + 30 * k) for k in range(n)]  # fmt: skip


def test_body_size_is_character_weighted_mode():
    lines = [mk(1, "x" * 400, size=11.0), mk(1, "Title", size=20.0)]
    lines += [mk(1, f"short {i}", size=9.0) for i in range(20)]  # many lines, few characters
    assert compute_signals(lines, 1).body_size == 11.0


def test_empty_input():
    s = compute_signals([], 0)
    assert s.body_size == 0.0 and s.median_gap == 0.0 and not s.excluded


def test_median_gap_ignores_overlap_and_page_edges():
    lines = [mk(1, "a b", gap_above=math.inf)]
    lines += [mk(1, f"line {i}", gap_above=g) for i, g in enumerate([0.0, 0.0, 4.0, 6.0, 8.0])]
    assert compute_signals(lines, 1).median_gap == 6.0


def test_running_footer_on_ten_of_twelve_pages_is_excluded():
    lines, footer_idx = [], set()
    for page in range(1, 13):
        lines += body_page(page)
        if page <= 10:
            footer_idx.add(len(lines))
            lines.append(mk(page, "Company Confidential", y0=H - 30))
    s = compute_signals(lines, 12)
    assert footer_idx <= s.excluded
    assert not any(i in s.excluded for i in range(len(lines)) if i not in footer_idx)


def test_running_header_digits_are_normalised():
    lines = []
    for page in range(1, 6):
        lines.append(mk(page, f"Annual Report 20{page}{page}", y0=10))
        lines += body_page(page)
    s = compute_signals(lines, 5)
    assert {i for i in s.excluded} == {0, 4, 8, 12, 16}


def test_edge_line_recurring_too_rarely_is_kept():
    lines = []
    for page in range(1, 11):
        lines += body_page(page)
        if page <= 2:  # below the 30% / minimum-3 threshold
            lines.append(mk(page, "Draft Notice", y0=H - 20))
    assert compute_signals(lines, 10).excluded == set()


def test_recurring_text_mid_page_is_not_a_running_line():
    lines = []
    for page in range(1, 8):
        lines.append(mk(page, "Summary", y0=400))
    assert compute_signals(lines, 7).excluded == set()


def test_page_numbers_are_excluded():
    lines = [
        mk(1, "Page 3 of 12", y0=H - 20), mk(1, "iv", y0=H - 20), mk(1, "12"),
        mk(1, "- 4 -"), mk(1, "3 / 9"), mk(1, "Real heading"),
    ]  # fmt: skip
    assert compute_signals(lines, 1).excluded == {0, 1, 2, 3, 4}


def test_words_that_look_like_roman_numerals_are_kept():
    lines = [mk(1, "Civil"), mk(1, "Mild"), mk(1, "Did")]
    assert compute_signals(lines, 1).excluded == set()


def test_printed_toc_page_with_dot_leaders():
    toc = [
        ("Introduction", "3"), ("Risk Factors", "7"), ("Business Overview", "12"),
        ("Financial Results", "20"), ("Governance", "31"), ("Appendix A", "40"),
    ]  # fmt: skip
    lines = [mk(2, "Some preface text on the page.")]
    lines += [mk(2, f"{t} " + "." * 30 + f" {p}") for t, p in toc]
    lines += body_page(3)
    s = compute_signals(lines, 3)
    assert s.printed_toc == toc
    assert set(range(1, 7)) <= s.toc_page_lines
    assert not any(i >= 7 for i in s.toc_page_lines)


def test_toc_leader_variants_and_raw_title_kept():
    lines = [
        mk(1, "1.2  Risk Factors ………… 14"),
        mk(1, "Methods ____ 15"), mk(1, "Results   16"), mk(1, "Discussion - - - 17"),
        mk(1, "Conclusion .  .  . 18"),
    ]  # fmt: skip
    s = compute_signals(lines, 1)
    assert s.printed_toc == [
        ("1.2  Risk Factors", "14"), ("Methods", "15"), ("Results", "16"),
        ("Discussion", "17"), ("Conclusion", "18"),
    ]  # fmt: skip


def test_four_toc_lines_without_heading_is_not_a_toc():
    lines = [mk(1, f"Chapter {i} .......... {i}") for i in range(4)]
    s = compute_signals(lines, 1)
    assert s.printed_toc == [] and s.toc_page_lines == set()


def test_contents_heading_lowers_threshold_to_three():
    lines = [mk(1, "Table of Contents"), mk(1, "Alpha .... 1"), mk(1, "Beta .... 5"),
             mk(1, "Gamma .... 9")]  # fmt: skip
    s = compute_signals(lines, 1)
    assert s.printed_toc == [("Alpha", "1"), ("Beta", "5"), ("Gamma", "9")]
    assert s.toc_page_lines == {1, 2, 3}  # the page's own heading may still be a heading


def test_toc_ignores_excluded_lines_and_digit_only_titles():
    lines = [mk(1, "Page 5 of 9", y0=H - 20)] + [mk(1, f"{n}   {n}") for n in range(10, 16)]
    assert compute_signals(lines, 1).printed_toc == []


def test_bookmarks_passed_through():
    s = compute_signals([], 0, bookmarks=[(1, "Intro", 1), (2, "Scope", 2)])
    assert s.bookmarks == [(1, "Intro", 1), (2, "Scope", 2)]
    assert compute_signals([], 0).bookmarks == []


def test_normalize_title():
    assert normalize_title("1.2  Risk Factors.") == "risk factors"
    assert normalize_title("ITEM 7. Management's Discussion") == "management's discussion"
    assert normalize_title("IV. Results:") == "results"
    assert normalize_title("A Brief History") == "a brief history"
    assert normalize_title("  Section 3.1)  Scope  ") == "scope"


def test_fuzzy_match():
    assert fuzzy_match("2.1 Risk Factors", "Risk factors.") == 1.0
    assert fuzzy_match("Risk Factors", "Risk Factor") >= 0.85
    assert fuzzy_match("Risk Factors", "Governance") < 0.5
    assert fuzzy_match("", "x") == 0.0


def test_section_running_footer_on_three_pages_of_a_long_document_is_excluded():
    # The current section's title in the footer band of each of its pages: far below 30% of a
    # long document, but an edge line repeated on three pages is a running line.
    lines, footer_idx = [], set()
    for page in range(1, 41):
        lines += body_page(page)
        if 11 <= page <= 13:
            footer_idx.add(len(lines))
            lines.append(mk(page, "1.4. Pinout Reference", size=8.0, y0=H - 40))
    s = compute_signals(lines, 40)
    assert s.excluded == footer_idx


def test_numbered_page_top_headings_are_not_section_running_lines():
    # "Item 1." .. "Item 4." at the top of pages differ only in their numbers: not a running line.
    lines = []
    for page in range(1, 41):
        if page % 10 == 1:
            lines.append(mk(page, f"Item {page // 10 + 1}.", y0=30))
        lines += body_page(page)
    assert compute_signals(lines, 40).excluded == set()


def test_section_running_header_with_changing_page_number_is_excluded():
    lines, header_idx = [], set()
    for page in range(1, 41):
        if 6 <= page <= 9:
            header_idx.add(len(lines))
            lines.append(mk(page, f"{page} Part 1: Recent Developments", y0=20))
        lines += body_page(page)
    assert compute_signals(lines, 40).excluded == header_idx


def _row(page, title, label, y0):
    """A contents row whose page number is a separate line on the same row, at the right."""
    return [
        mk(page, title, y0=y0, w=300.0),
        mk(page, label, y0=y0, w=20.0).model_copy(update={"x0": 500.0, "x1": 520.0}),
    ]


def test_toc_page_numbers_on_separate_lines_of_the_same_row():
    rows = [("Introduction", "3"), ("Approach", "6"), ("Training Dataset", "8"),
            ("Results", "10"), ("Limitations", "33"), ("Broader Impacts", "34")]  # fmt: skip
    lines = [mk(2, "Model and Architectures . . . . . . . . . .", y0=60.0)]
    for k, (title, label) in enumerate(rows):
        lines += _row(2, title, label, 100.0 + 20 * k)
    lines += body_page(3)
    s = compute_signals(lines, 40)
    assert ("Introduction", "3") in s.printed_toc and ("Broader Impacts", "34") in s.printed_toc
    assert ("Model and Architectures", "") in s.printed_toc
    assert {i for i, line in enumerate(lines) if line.page == 2} - s.excluded <= s.toc_page_lines
    assert not any(lines[i].page == 3 for i in s.toc_page_lines)


def test_table_rows_with_unordered_numbers_are_not_a_toc():
    rows = [("Office space", "655"), ("Physical stores", "22"), ("Fulfillment", "407"),
            ("Data centers", "18"), ("Other", "301"), ("Total", "90")]  # fmt: skip
    lines = []
    for k, (title, label) in enumerate(rows):
        lines += _row(1, title, label, 100.0 + 20 * k)
    s = compute_signals(lines, 1)
    assert s.printed_toc == [] and s.toc_page_lines == set()


def test_numbered_contents_without_page_numbers_is_a_toc_page():
    entries = ["1.\xa0\xa0Introduction", "1.1.\xa0\xa0Purpose", "1.2.\xa0\xa0History",
               "2.\xa0\xa0Conformance", "2.1.\xa0\xa0Syntax Notation", "2.2.\xa0\xa0Requirements",
               "3.\xa0\xa0Terminology", "3.1.\xa0\xa0Resources",
               "3.2.\xa0\xa0Representations"]  # fmt: skip
    lines = [mk(2, e, y0=100.0 + 20 * k) for k, e in enumerate(entries)] + body_page(3)
    s = compute_signals(lines, 3)
    assert s.toc_page_lines == set(range(len(entries)))
    assert ("1.1.\xa0\xa0Purpose", "") in s.printed_toc


def test_numbered_steps_mixed_with_prose_are_not_a_toc_page():
    lines = []
    for k in range(9):
        lines.append(mk(1, f"{k + 1}. Run the step", y0=100.0 + 40 * k))
        lines.append(mk(1, "An ordinary sentence that explains the step in detail here.",
                        y0=115.0 + 40 * k))  # fmt: skip
    assert compute_signals(lines, 1).toc_page_lines == set()


def test_code_listing_with_line_numbers_is_not_a_toc_page():
    code = ["int main() {", "stdio_init_all();", "int x = 0;", "while (true) {", "x++;",
            "sleep_ms(10);", "}", "return 0;", "}"]  # fmt: skip
    lines = [mk(1, f"{19 + k} {c}", y0=100.0 + 15 * k) for k, c in enumerate(code)]
    assert compute_signals(lines, 1).toc_page_lines == set()


def test_ellipsis_is_not_a_toc_leader():
    lines = [mk(1, f"Pass the value on to the{k}...", y0=100.0 + 15 * k) for k in range(6)]
    assert compute_signals(lines, 1).toc_page_lines == set()


def test_numbered_contents_stops_where_the_document_starts_on_the_same_page():
    entries = [f"{n}.{k}.\xa0\xa0Entry {n} {k}" for n in (21, 22) for k in range(1, 6)]
    lines = [mk(8, e, y0=60.0 + 18 * j) for j, e in enumerate(entries)]
    lines.append(mk(8, "1.\xa0\xa0Overview", y0=300.0))
    lines += [mk(8, "Body text of the overview section follows here.", y0=320.0 + 15 * k)
              for k in range(2)]  # fmt: skip
    s = compute_signals(lines, 20)
    assert s.toc_page_lines == set(range(len(entries)))
