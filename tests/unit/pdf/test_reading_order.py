import sys

from docrooter.pdf import Line
from docrooter.pdf.reading_order import order_lines, xy_cut_order

PAGE_W = 600.0
CW = 5.0  # median character width
LG = 4.0  # median line gap
LEFT = (50.0, 280.0)
RIGHT = (320.0, 550.0)


def column(x, y0, n, *, step=14.0, h=10.0):
    return [(x[0], y0 + k * step, x[1], y0 + k * step + h) for k in range(n)]


def cut(boxes, **kw):
    kw.setdefault("page_width", PAGE_W)
    kw.setdefault("median_char_width", CW)
    kw.setdefault("median_line_gap", LG)
    return xy_cut_order(boxes, **kw)


def test_two_columns_read_left_then_right():
    left, right = column(LEFT, 100, 5), column(RIGHT, 100, 5)
    boxes = [b for pair in zip(left, right, strict=True) for b in pair]  # row-interleaved
    order = cut(boxes)
    assert order == [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]


def test_three_columns():
    cols = [(40.0, 180.0), (210.0, 350.0), (380.0, 520.0)]
    boxes = [b for k in range(4) for c in cols for b in column(c, 100 + 60 * k, 1)]
    # rows interleave the columns; each column is a stack of 4 boxes
    order = cut(boxes)
    by_col = [[i for i in range(len(boxes)) if i % 3 == c] for c in range(3)]
    assert order == by_col[0] + by_col[1] + by_col[2]


def test_full_width_title_and_figure_split_bands():
    title = (50.0, 40.0, 550.0, 70.0)
    figure = (50.0, 200.0, 550.0, 300.0)
    a1, b1 = column(LEFT, 90, 6), column(RIGHT, 90, 6)  # y 90..174
    a2, b2 = column(LEFT, 320, 6), column(RIGHT, 320, 6)

    def rows(left, right):  # row-interleaved, as an extractor emits two columns
        return [b for pair in zip(left, right, strict=True) for b in pair]

    boxes = [title, *rows(a1, b1), figure, *rows(a2, b2)]
    # title, then column A (even offsets) before B in band 1, the figure, and the same in band 2
    expected = [0, 1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12, 13, 14, 16, 18, 20, 22, 24, 15, 17, 19]
    expected += [21, 23, 25]
    assert cut(boxes) == expected


def test_single_column_order_unchanged():
    boxes = column((50.0, 550.0), 100, 40)
    boxes[7] = (50.0, boxes[7][1], 300.0, boxes[7][3])  # a short line
    assert cut(boxes) == list(range(40))


def test_single_column_with_out_of_order_input_is_kept_as_is():
    # no column structure: the extractor's order is trusted, not re-sorted
    boxes = column((50.0, 550.0), 100, 10)
    boxes[3], boxes[4] = boxes[4], boxes[3]
    assert cut(boxes) == list(range(10))


def test_gutter_rules():
    # gutter of 11 pt: wide enough for 5 pt chars (needs max(8, 15)=15)? no -> not a cut
    a, b = column((50.0, 289.0), 100, 5), column((300.0, 550.0), 100, 5)
    boxes = [x for pair in zip(a, b, strict=True) for x in pair]
    assert cut(boxes, median_char_width=5.0) == list(range(10))  # gutter 11 < 15
    assert cut(boxes, median_char_width=2.0) == [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]  # 11 >= 8
    # gutter of 6 pt is below the 8 pt floor, whatever the character width
    a, b = column((50.0, 294.0), 100, 5), column((300.0, 550.0), 100, 5)
    boxes = [x for pair in zip(a, b, strict=True) for x in pair]
    assert cut(boxes, median_char_width=1.0) == list(range(10))
    # a right column covering only a third of the left column's height is a gutter too: its
    # y-range lies inside the left column's (E12 ruling: the sides overlap by 50% of the shorter)
    left, right = column(LEFT, 100, 12), column(RIGHT, 100, 4)
    boxes = [x for pair in zip(left[:4], right, strict=True) for x in pair] + left[4:]
    assert cut(boxes) == [*range(0, 8, 2), *range(8, 16), *range(1, 8, 2)]
    # ...and so is one covering most of it
    left, right = column(LEFT, 100, 12), column(RIGHT, 100, 10)
    boxes = [x for pair in zip(left[:10], right, strict=True) for x in pair] + left[10:]
    order = cut(boxes)
    assert order == [*range(0, 20, 2), 20, 21, *range(1, 20, 2)]


def test_label_and_number_table_is_not_split_into_columns():
    # a ledger: wide labels on the left, narrow figures on the right (a gutter, but no column)
    rows = [
        b
        for k in range(8)
        for b in (
            (50.0, 100.0 + 14 * k, 250.0, 110.0 + 14 * k),
            (470.0, 100.0 + 14 * k, 500.0, 110.0 + 14 * k),
        )
    ]
    assert cut(rows) == list(range(16))
    # ...but two columns of short lines (each at least 20% of the text width) are columns
    cols = [
        b
        for k in range(8)
        for b in (
            (50.0, 100.0 + 14 * k, 200.0, 110.0 + 14 * k),
            (350.0, 100.0 + 14 * k, 500.0, 110.0 + 14 * k),
        )
    ]
    assert cut(cols) == list(range(0, 16, 2)) + list(range(1, 16, 2))


def test_text_width_override_changes_what_is_full_width():
    # the 290 pt heading is full width (60%) only when the text width is <= 483 pt (default: 500)
    def rows(x, y0):
        return column(x, y0, 3)

    top = [b for pair in zip(rows(LEFT, 100), rows(RIGHT, 100), strict=True) for b in pair]
    heading = [(50.0, 140.0, 340.0, 150.0)]  # 2 pt below / above its neighbours: no gap cut
    bottom = [b for pair in zip(rows(LEFT, 152), rows(RIGHT, 152), strict=True) for b in pair]
    boxes = [*top, *heading, *bottom]
    # the heading crosses the gutter between the columns' lines: no gutter, no columns
    assert cut(boxes) == list(range(13))
    assert cut(boxes, text_width=400.0) == [0, 2, 4, 1, 3, 5, 6, 7, 9, 11, 8, 10, 12]


def test_no_recursion():
    limit = sys.getrecursionlimit()
    left, right = column(LEFT, 10, 1000, step=14), column(RIGHT, 10, 1000, step=14)
    boxes = [x for pair in zip(left, right, strict=True) for x in pair]
    order = cut(boxes)
    assert order == list(range(0, 2000, 2)) + list(range(1, 2000, 2))
    assert sys.getrecursionlimit() == limit


def test_degenerate_inputs():
    assert cut([]) == []
    assert cut([(1.0, 1.0, 1.0, 1.0)]) == [0]
    assert cut([(0.0, 0.0, 0.0, 0.0)] * 3) == [0, 1, 2]


def mk(page, i, x0, y0, x1, y1, *, text=None):
    return Line(
        page=page, text=text or "x" * max(1, int((x1 - x0) / 5)), size=10.0, bold=False,
        font="F", x0=x0, y0=y0, x1=x1, y1=y1, page_height=800.0, page_width=PAGE_W,
        gap_above=4.0, gap_below=4.0, block=0,
    )  # fmt: skip


def test_order_lines_orders_pages_and_places_furniture():
    lines = [
        mk(1, 0, 50, 20, 300, 30),  # 0: running header (excluded, top)
        *[mk(1, 0, x0, 100 + 14 * k, x0 + 230, 110 + 14 * k) for k in range(5) for x0 in (50, 320)],
        mk(1, 0, 280, 770, 320, 780),  # 11: page number (excluded, bottom)
        mk(2, 0, 50, 100, 550, 110),  # 12: page 2, single line
        mk(2, 0, 50, 20, 300, 30),  # 13: header of page 2
    ]
    got = order_lines(lines, {0, 11, 13})
    assert got[0] == 0
    assert got[1:6] == [1, 3, 5, 7, 9] and got[6:11] == [2, 4, 6, 8, 10]
    assert got[11] == 11
    assert got[12:] == [13, 12]  # header first on page 2
    assert sorted(got) == list(range(len(lines)))


def test_order_lines_keeps_furniture_in_original_relative_order():
    lines = [
        mk(1, 0, 50, 700, 300, 710),  # bottom, first
        mk(1, 0, 50, 780, 300, 790),  # bottom, second
        mk(1, 0, 50, 10, 300, 20),  # top
        mk(1, 0, 50, 100, 550, 110),
        mk(1, 0, 50, 120, 550, 130),
    ]
    assert order_lines(lines, {0, 1, 2}) == [2, 3, 4, 0, 1]


def test_order_lines_empty_and_single_column_identity():
    assert order_lines([], set()) == []
    lines = [mk(1, 0, 50, 100 + 14 * k, 550, 110 + 14 * k) for k in range(30)]
    assert order_lines(lines, set()) == list(range(30))


def unruled_table(rows=10):
    """A label column and two number columns in rows, under two column-header lines that span
    the number columns, emitted row by row as an extractor does (headers first)."""
    boxes = [(300.0, 90.0, 460.0, 100.0), (300.0, 104.0, 460.0, 114.0)]
    for k in range(rows):
        y = 120.0 + 14 * k
        boxes += [(50.0, y, 250.0, y + 10), (300.0, y, 330.0, y + 10), (430.0, y, 460.0, y + 10)]
    return boxes


def test_unruled_table_keeps_row_order():
    # without the row-alignment rule the label/number gutter is cut: every label, then every
    # number (doc_004 p27); the rows of a table must stay together
    boxes = unruled_table()
    assert cut(boxes) == list(range(len(boxes)))
    # the same table below a two-column article: the article is still read column by column
    article = [b for pair in zip(column(LEFT, 300, 6), column(RIGHT, 300, 6), strict=True)
               for b in pair]  # fmt: skip
    order = cut([*boxes, *article])
    n = len(boxes)
    assert order[:n] == list(range(n))
    assert order[n:] == [n + k for k in range(0, 12, 2)] + [n + k for k in range(1, 12, 2)]


def test_two_column_article_with_independent_baselines_reads_by_column():
    left = column(LEFT, 100, 8)
    right = column(RIGHT, 107, 8)  # baselines half a line pitch off the left column's
    boxes = [b for pair in zip(left, right, strict=True) for b in pair]
    assert cut(boxes) == list(range(0, 16, 2)) + list(range(1, 16, 2))


def test_text_columns_on_one_baseline_grid_are_still_columns():
    # row-aligned, but both sides hold full text lines: columns, not a table
    left, right = column(LEFT, 100, 8), column(RIGHT, 100, 8)
    boxes = [b for pair in zip(left, right, strict=True) for b in pair]
    assert cut(boxes) == list(range(0, 16, 2)) + list(range(1, 16, 2))


def test_narrow_text_columns_on_one_grid_are_still_columns():
    # three brochure columns of a fifth of the text width (680 pt), on one baseline grid, with
    # every other line short (median line 18% of the text width): narrow text, not table cells
    starts = (40.0, 310.0, 580.0)
    boxes = [
        (x, 100.0 + 14 * k, x + (140.0 if k % 2 == 0 else 110.0), 110.0 + 14 * k)
        for k in range(8)
        for x in starts
    ]
    order = cut(boxes, page_width=760.0)
    assert order == [i for c in range(3) for i in range(c, 24, 3)]


def test_columns_of_unequal_length_under_a_heading_read_left_then_right():
    # a heading over the left column only, then 5 lines left and 4 right at tight leading, with
    # the right column first in the input (an extractor sorts blocks by their bottom edge)
    heading = (50.0, 60.0, 140.0, 78.0)
    left, right = column(LEFT, 82, 5, step=12.0, h=13.8), column(RIGHT, 82, 4, step=12.0, h=13.8)
    assert cut([heading, *right, *left], median_line_gap=12.4) == [0, 5, 6, 7, 8, 9, 1, 2, 3, 4]
    assert cut([heading, *left, *right], median_line_gap=12.4) == list(range(10))


def test_a_short_column_beside_a_tall_one_is_read_after_it_whatever_is_around():
    title = (50.0, 20.0, 330.0, 50.0)  # ends inside the gutter: it crosses it
    heading = (50.0, 60.0, 140.0, 78.0)
    below = (50.0, 300.0, 340.0, 312.0)  # the next section's heading, across the gutter
    left, right = column(LEFT, 82, 12), column(RIGHT, 82, 3)
    boxes = [title, heading, *right, *left, below]
    # the title and heading, the long left column, the short right one, then the section below
    assert cut(boxes) == [0, 1, *range(5, 17), 2, 3, 4, 17]


def test_a_table_of_wide_figures_keeps_its_rows():
    # labels beside rows of percentages as wide as a line of text (doc_001 p122): the figures
    # are the cells of a table, whatever their width, so the rows stay together
    rows = [
        b
        for k in range(8)
        for b in (
            (50.0, 300.0 + 14 * k, 250.0, 310.0 + 14 * k),
            (300.0, 300.0 + 14 * k, 500.0, 310.0 + 14 * k),
        )
    ]
    assert cut(rows) == list(range(0, 16, 2)) + list(range(1, 16, 2))  # geometry alone: columns
    assert cut(rows, numeric=[k % 2 == 1 for k in range(16)]) == list(range(16))


def test_order_lines_keeps_the_rows_of_a_table_of_figures():
    def line(i, x0, x1, y, text):
        return mk(1, i, x0, y, x1, y + 10, text=text)

    rows = []
    for k in range(6):
        rows += [
            line(2 * k, 50, 250, 300 + 14 * k, f"Segment number {k} of the business"),
            line(2 * k + 1, 300, 500, 300 + 14 * k, "41% 31% 13% 7% 4% 2% 1% 1%"),
        ]
    assert order_lines(rows, set()) == list(range(12))
    words = [
        line(i, 300, 500, y, "Words that make a second column of text")
        if i % 2
        else line(i, 50, 250, y, "The first column of ordinary text")
        for i, y in zip(range(12), (300 + 14 * (k // 2) for k in range(12)), strict=True)
    ]
    assert order_lines(words, set()) == list(range(0, 12, 2)) + list(range(1, 12, 2))


def test_short_fragments_between_full_width_lines_are_not_columns():
    # a single-column page of full-width lines with a few short lines and a bold keyword set
    # apart on the right (doc_031 p110): the short boxes of different rows are no two columns
    boxes = []
    for k in range(10):
        y = 100.0 + 14 * k
        if k in (2, 5, 8):
            boxes += [(66.0, y, 297.0, y + 10), (398.0, y, 506.0, y + 10)]
        else:
            boxes.append((66.0, y, 520.0, y + 10))
    assert cut(boxes) == list(range(len(boxes)))


def test_a_table_with_a_narrow_column_of_cells_between_two_wide_ones_keeps_its_rows():
    # name, ticker, exchange (doc_004 p1): the ticker column makes it a table whichever valley
    # is tried, and the rows are in the extractor's order
    boxes = []
    for k in range(8):
        y = 300.0 + 12 * k
        boxes += [
            (36.0, y, 161.0, y + 10),
            (285.0, y, 315.0, y + 10),
            (369.0, y, 555.0, y + 10),
        ]
    assert cut(boxes, page_width=594.0) == list(range(len(boxes)))
