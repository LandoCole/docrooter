"""Element typing rules (spec §5.3) on synthetic page objects and lines, and on generated PDFs."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from docrooter.elements import Box, Element
from docrooter.pdf import elements as el
from docrooter.pdf.elements import CAPTION_RE, build_elements, build_elements_with_lines
from docrooter.pdf.headings import HeadingCandidate
from docrooter.pdf.lines import Line
from docrooter.pdf.objects import PageObjects
from docrooter.pdf.signals import DocSignals

W, H = 612.0, 792.0


# --- synthetic documents -------------------------------------------------------------------


def row(page, text, x0, y0, *, x1=None, size=10.0, h=None, font="Helvetica", bold=False):
    """One line as a dict; ``finish`` turns a list of them into ``Line`` objects."""
    h = size * 1.2 if h is None else h
    x1 = x0 + max(6.0, 0.5 * size * len(text)) if x1 is None else x1
    return dict(page=page, text=text, size=size, bold=bold, font=font, x0=x0, y0=y0, x1=x1,
                y1=y0 + h, page_height=H, page_width=W)  # fmt: skip


def column(page, texts, y0, *, x0=72.0, x1=540.0, step=14.0, size=10.0):
    return [row(page, t, x0, y0 + k * step, x1=x1, size=size) for k, t in enumerate(texts)]


def finish(rows):
    """``Line`` objects with gaps filled in, in the given order per page."""
    out = []
    prev = None
    for r in rows:
        gap = (
            math.inf
            if prev is None or prev["page"] != r["page"]
            else max(0.0, r["y0"] - prev["y1"])
        )
        out.append({**r, "gap_above": gap, "gap_below": math.inf, "block": 0})
        if prev is not None and prev["page"] == r["page"]:
            out[-2]["gap_below"] = gap
        prev = r
    return [Line(**r) for r in out]


@dataclass
class Doc:
    rows: list[dict]
    n_pages: int = 1
    images: dict[int, list[tuple[float, float, float, float]]] = field(default_factory=dict)
    drawings: dict[int, list[tuple[float, float, float, float]]] = field(default_factory=dict)
    tables: dict[int, list[tuple[tuple[float, float, float, float], list[list[str]]]]] = field(
        default_factory=dict
    )
    excluded: set[int] = field(default_factory=set)
    toc: set[int] = field(default_factory=set)
    headings: list[HeadingCandidate] = field(default_factory=list)
    title_index: int | None = None
    line_source: list[str] | None = None
    order: list[int] | None = None
    body: float = 10.0
    gap: float = 2.0

    @property
    def lines(self) -> list[Line]:
        return finish(self.rows)

    def pages(self) -> list[PageObjects]:
        lines = self.lines
        return [
            PageObjects(
                page=p,
                width=W,
                height=H,
                rotation=0,
                lines=[ln for ln in lines if ln.page == p],
                images=[
                    Box(page=p, x0=a, y0=b, x1=c, y1=d) for a, b, c, d in self.images.get(p, [])
                ],
                drawings=[
                    Box(page=p, x0=a, y0=b, x1=c, y1=d) for a, b, c, d in self.drawings.get(p, [])
                ],
                tables=[
                    (Box(page=p, x0=a, y0=b, x1=c, y1=d), rows)
                    for (a, b, c, d), rows in self.tables.get(p, [])
                ],
                char_count=0,
                invisible_chars=0,
                garbled_chars=0,
                n_drawing_paths=0,
            )  # fmt: skip
            for p in range(1, self.n_pages + 1)
        ]

    def args(self):
        lines = self.lines
        signals = DocSignals(
            body_size=self.body, median_gap=self.gap, excluded=set(self.excluded),
            toc_body_lines=set(self.toc), toc_page_lines=set(self.toc),
        )  # fmt: skip
        order = list(range(len(lines))) if self.order is None else self.order
        return (self.pages(), lines, order, signals, self.headings), dict(
            title_index=self.title_index, line_source=self.line_source
        )

    def build(self) -> list[Element]:
        args, kwargs = self.args()
        return build_elements(*args, **kwargs)

    def typed(self) -> list[tuple[Element, list[int]]]:
        args, kwargs = self.args()
        return build_elements_with_lines(*args, **kwargs)


def heading(rows, index, *, count=1, level=1, score=0.9):
    first = rows[index]
    text = " ".join(rows[k]["text"] for k in range(index, index + count))
    return HeadingCandidate(
        line_index=index, line_count=count, text=text, page=first["page"], size=first["size"],
        bold=first["bold"], score=score, level=level,
    )  # fmt: skip


def index_of(rows, text):
    return next(k for k, r in enumerate(rows) if r["text"] == text)


FIXTURES: dict[str, Callable[[], Doc]] = {}


def fixture(fn: Callable[[], Doc]) -> Callable[[], Doc]:
    FIXTURES[fn.__name__] = fn
    return fn


def kinds(elements):
    return [e.kind for e in elements]


def of_kind(elements, kind):
    return [e for e in elements if e.kind == kind]


@fixture
def furniture_doc():
    rows = [
        row(1, "ACME Corp Annual Report", 72, 20),
        row(1, "iv", 520, 20),
        *column(1, ["Body text line one of the page.", "Body text line two of the page."], 100),
        row(1, "12", 300, 300),
        *column(1, ["More body text below the number.", "And a last body line."], 400),
        row(1, "Confidential draft", 72, 760),
        row(1, "7", 300, 760),
    ]
    excluded = {
        index_of(rows, t)
        for t in ("ACME Corp Annual Report", "iv", "12", "Confidential draft", "7")
    }
    return Doc(rows, excluded=excluded)


@fixture
def contents_doc():
    entries = [f"Chapter {k} title ........ {k * 4}" for k in range(1, 6)]
    rows = [
        row(1, "Contents", 72, 60, size=14),
        *column(1, entries, 100),
        row(2, "Introduction", 72, 60, size=14),
        *column(2, ["The report opens with an introduction."], 100),
    ]
    toc = {index_of(rows, t) for t in entries}
    heads = [heading(rows, 0, level=1), heading(rows, index_of(rows, "Introduction"), level=1)]
    return Doc(rows, n_pages=2, toc=toc, headings=heads)


@fixture
def table_doc():
    rows = [
        row(1, "The table below lists revenue.", 72, 100),
        row(1, "Item", 80, 152), row(1, "2024", 300, 152), row(1, "2023", 450, 152),
        row(1, "Revenue", 80, 172), row(1, "12", 300, 172), row(1, "10", 450, 172),
        row(1, "Costs", 80, 192), row(1, "4", 450, 192),
        row(1, "Totals are unaudited.", 72, 240),
    ]  # fmt: skip
    cells = [["Item", "2024", "2023"], ["Revenue", "12", "10"], ["Costs", "", "4"]]
    # compute_signals marks every bare number as a page number; inside a table they are cells
    excluded = {index_of(rows, t) for t in ("2024", "2023", "12", "10", "4")}
    return Doc(rows, tables={1: [((72, 150, 540, 210), cells)]}, excluded=excluded)


@fixture
def figures_doc():
    rows = [
        row(1, "Axis A", 150, 200),
        row(1, "Q1", 150, 400), row(1, "Q2", 250, 400), row(1, "Q3", 350, 400),
        *[row(1, f"Note {k} text", 420, 352 + 12 * k, x1=530, h=10) for k in range(8)],
        row(1, "a", 80, 610), row(1, "b", 300, 610), row(1, "c", 80, 650), row(1, "d", 300, 650),
        row(1, "Body text under the figures.", 72, 720),
    ]  # fmt: skip
    return Doc(
        rows,
        images={1: [(100, 100, 400, 300), (500, 100, 510, 110)]},
        drawings={1: [(100, 350, 400, 550), (72, 600, 540, 700), (420, 350, 540, 450)]},
        tables={1: [((72, 600, 540, 700), [["a", "b"], ["c", "d"]])]},
    )


@fixture
def background_doc():
    rows = [
        *column(1, [f"Scanned body line {k} with ordinary words." for k in range(30)], 80),
        *[row(2, f"Callout line {k} inside a shaded box.", 110, 110 + 12 * k, x1=380, h=10)
          for k in range(6)],
        row(2, "Cover title over a photo", 100, 400, size=24),
    ]  # fmt: skip
    return Doc(
        rows,
        n_pages=2,
        images={1: [(0, 0, W, H)], 2: [(0, 300, W, H)]},
        drawings={2: [(100, 100, 400, 190)]},
    )


@fixture
def captions_doc():
    rows = [
        row(1, "Figure 1: Sales by region", 100, 305),
        row(1, "and by quarter.", 100, 319),
        row(1, "Table 2. Costs", 100, 430),
        row(1, "x", 110, 460), row(1, "y", 300, 460), row(1, "1", 110, 500), row(1, "2", 300, 500),
        row(1, "Figure 3 shows growth in every region.", 72, 650),
        row(2, "Figure illustrates the trend over time.", 100, 305),
    ]  # fmt: skip
    return Doc(
        rows,
        n_pages=2,
        images={1: [(100, 100, 400, 300)], 2: [(100, 100, 400, 300)]},
        tables={1: [((100, 450, 500, 550), [["x", "y"], ["1", "2"]])]},
    )


@fixture
def headings_doc():
    rows = [
        row(1, "Annual Report", 72, 60, size=24),
        row(1, "2024 Edition", 72, 90, size=24),
        *column(1, ["Opening words of the report."], 140),
        row(1, "1. Overview", 72, 200, size=14),
        *column(1, ["The overview section text."], 230),
        row(2, "2. Results", 72, 60, size=14),
        *column(2, ["The results section text."], 100),
    ]
    heads = [
        heading(rows, 0, count=2, level=1),
        heading(rows, index_of(rows, "1. Overview"), level=2),
        heading(rows, index_of(rows, "2. Results"), level=2),
    ]
    return Doc(rows, n_pages=2, headings=heads, title_index=0)


@fixture
def footnotes_doc():
    rows = [
        *column(1, ["Body text with a reference.1", "More body text follows."], 100),
        row(1, "1 not a footnote at the top", 72, 150, size=8),
        row(1, "3 apples were sold at body size.", 72, 600),
        row(1, "688 $ (4,638) $ 156", 72, 620, size=8),
        row(1, "1 First note text", 72, 640, size=8),
        row(1, "continues here.", 72, 650, size=8),
        row(1, "2 Second note", 72, 662, size=8),
        row(1, "* Star note", 72, 674, size=8),
        row(1, "†Dagger note", 72, 686, size=8),
        row(1, "(5) Parenthesised note", 72, 698, size=8),
    ]
    return Doc(rows)


@fixture
def lists_doc():
    items = [
        ("• Alpha item", 72), ("wraps onto a second line", 82), ("◦ Beta item", 90),
        ("▪ Gamma item", 108), ("- Delta item", 72), ("– Echo item", 72), ("* Foxtrot item", 72),
        ("1. One", 72), ("1) One b", 72), ("(a) Letter", 90), ("a. letter b", 90),
        ("i. roman", 108), ("(iv) roman paren", 108),
    ]  # fmt: skip
    rows = [row(1, t, x, 100 + 14 * k, x1=540) for k, (t, x) in enumerate(items)]
    rows += column(1, ["-5% decline in sales was reported.", "1.5 million units shipped."], 320)
    return Doc(rows)


@fixture
def formulas_doc():
    rows = [
        row(1, "The energy relation is given below.", 72, 100),
        row(1, "E = mc² + α·β", 200, 130, x1=330),
        row(1, "(1)", 500, 130, x1=520),
        row(1, "where m is the mass.", 72, 160),
        row(1, "x y z", 250, 190, x1=300, font="CMMI10"),
        row(1, "We set a = b in the first step of the proof.", 72, 220),
        row(1, "= = ±", 72, 260, x1=110),
        row(1, "text on the same row here", 200, 260),
    ]
    return Doc(rows)


@fixture
def paragraphs_doc():
    rows = [
        *column(1, ["First paragraph line one runs on", "and continues on the second line."], 100),
        *column(1, ["Second paragraph after a gap", "keeps going here."], 140),
        row(1, "Larger type paragraph", 72, 168, x1=540, size=12),
        *column(1, ["Back to body text that ends."], 184.4),
        row(1, "Indented first line of a new paragraph", 90, 198.4, x1=540),
        *column(1, ["continues at the margin"], 212.4),
        row(1, "and an indented line", 90, 226.4, x1=540),
    ]
    return Doc(rows)


@fixture
def continued_doc():
    rows = [
        *column(1, ["Opening paragraph ends here."], 600),
        *column(1, ["The results of the study were"], 640),
        row(1, "1", 300, 760),
        row(2, "Running head", 72, 20),
        *column(2, ["consistent with earlier work."], 100),
    ]
    excluded = {index_of(rows, "1"), index_of(rows, "Running head")}
    return Doc(rows, n_pages=2, excluded=excluded)


@fixture
def not_continued_doc():
    rows = [
        *column(1, ["The results of the study were final."], 640),
        *column(2, ["consistent with earlier work."], 100),
        *column(2, ["The second study was"], 300),
        *column(3, ["Consistent with the first one."], 100),
    ]
    return Doc(rows, n_pages=3)


@fixture
def columns_doc():
    left = column(1, ["Left column text starts", "and the left column text goes"], 100, x1=290)
    right = column(1, ["on into the right column.", "A second right paragraph."], 100, x0=320)
    right[1]["y0"] += 20
    right[1]["y1"] += 20
    return Doc(left + right)


@fixture
def ocr_doc():
    rows = [
        *column(1, ["A text layer page."], 100),
        *column(2, ["An OCR page line", "and its second line."], 100),
    ]
    return Doc(rows, n_pages=2, line_source=["text", "ocr", "ocr"])


@fixture
def degenerate_doc():
    rows = [row(1, "   ", 72, 100), *column(1, ["One real line."], 300)]
    return Doc(
        rows,
        n_pages=2,
        images={1: [(300, 300, 300, 400), (-50, -50, 200, 200)], 2: [(100, 100, 100.2, 100.2)]},
        drawings={2: [(50, 50, 50, 300)]},
    )


@fixture
def statement_doc():
    """Statements drawn with rules or shading (vector clusters the table search misses)."""
    rows = []
    for k in range(10):
        y = 105 + 15 * k
        rows += [row(1, f"Revenue line {k}", 80, y, x1=160), row(1, "12,345", 370, y, x1=400),
                 row(1, "11,111", 490, y, x1=520)]  # fmt: skip
    rows += [row(2, "Other assets", 80, 105, x1=150), row(2, "62,950", 370, 105, x1=400),
             row(2, "55,335", 490, 105, x1=520), row(2, "Total other assets", 80, 120, x1=170),
             row(2, "83,727", 370, 120, x1=400), row(2, "74,834", 490, 120, x1=520)]  # fmt: skip
    names = ["Acme Asia LLC", "Acme Canada Inc.", "Acme Europe Ltd", "Acme Japan KK"]
    places = ["Delaware, U.S.", "Canada", "Ireland", "Japan"]
    for k, (name, place) in enumerate(zip(names, places, strict=True)):
        rows += [row(3, name, 80, 105 + 15 * k, x1=170), row(3, place, 400, 105 + 15 * k, x1=460)]
    return Doc(
        rows,
        n_pages=3,
        drawings={1: [(72, 100, 540, 260)], 2: [(72, 100, 540, 135)], 3: [(72, 100, 540, 170)]},
    )


@fixture
def prose_box_doc():
    prose = [f"Running text line {k} of a paragraph set over a picture" for k in range(4)]
    rows = [
        *[row(1, t, 100, 150 + 14 * k, x1=400) for k, t in enumerate(prose)],
        *[row(2, t, 100, 150 + 14 * k, x1=400) for k, t in enumerate(prose)],
    ]
    return Doc(
        rows, n_pages=2, images={1: [(72, 100, 540, 400)]}, drawings={2: [(72, 100, 540, 400)]}
    )


@fixture
def banner_doc():
    rows = [
        row(1, "Risk Factors", 50, 112, x1=150, size=14),
        *column(1, ["The company faces several risks.", "They are described below."], 160),
    ]
    return Doc(rows, drawings={1: [(36, 100, 576, 140)]}, headings=[heading(rows, 0, level=1)])


@fixture
def shaded_code_and_list_doc():
    code = ["x = 1", "y = x + 2", "print(y)", "return y"]
    rows = [row(1, t, 90, 110 + 14 * k, x1=200, font="Courier") for k, t in enumerate(code)]
    rows += [
        row(1, t, 90, 410 + 14 * k, x1=250)
        for k, t in enumerate(["• First point", "• Second point"])
    ]
    return Doc(rows, drawings={1: [(72, 100, 540, 200), (72, 400, 540, 480)]})


@fixture
def chart_doc():
    """A chart whose legend entries share rows with both y axes' ticks, with its caption set
    inside its region, under a paragraph that runs into the region's top edge."""
    rows = [
        row(1, "This paragraph runs on with a long first line", 72, 82, x1=540),
        row(1, "and its last line ends inside the box", 72, 96, x1=540),
    ]
    for k, (left, right) in enumerate([("30", "300"), ("20", "200"), ("10", "100"), ("0", "0")]):
        y = 120 + 40 * k
        rows += [row(1, left, 105, y, x1=115), row(1, right, 470, y, x1=485)]
        if k < 2:
            rows.append(row(1, f"Series {'AB'[k]}", 300, y, x1=340))
    rows += [row(1, str(2020 + k), 150 + 80 * k, 270, x1=170 + 80 * k) for k in range(4)]
    rows += [
        row(1, "Percent", 105, 104, x1=140),
        row(1, "Figure 5: Rates over time", 150, 286, x1=300),
    ]
    return Doc(rows, drawings={1: [(100, 100, 500, 300)]})


@fixture
def long_caption_doc():
    rest = [
        "continues on a second line",
        "and a third line",
        "and a fourth line",
        "and a fifth line",
        "and a sixth line",
        "and a seventh line",
    ]
    rows = [row(1, "Figure 4: A long caption", 100, 305, x1=400)]
    rows += [row(1, t, 100, 319 + 14 * k, x1=400) for k, t in enumerate(rest)]
    return Doc(rows, images={1: [(100, 100, 400, 300)]})


@fixture
def caption_sides_doc():
    """A caption line halfway between two targets, on three pages."""
    rows = [
        row(1, "Figure 2: Between", 100, 214, x1=300),
        row(2, "Table 3. Between", 100, 214, x1=300),
        row(3, "Table 1. Mixed", 100, 214, x1=300),
    ]
    cells = [["a", "b"], ["c", "d"]]
    return Doc(
        rows,
        n_pages=3,
        images={1: [(100, 100, 400, 200), (100, 240, 400, 340)], 3: [(100, 100, 400, 200)]},
        tables={
            2: [((100, 100, 400, 200), cells), ((100, 240, 400, 340), cells)],
            3: [((100, 240, 400, 340), cells)],
        },
    )


# --- one test per rule ---------------------------------------------------------------------


def test_furniture_kinds():
    elements = furniture_doc().build()
    by_text = {e.text: e.kind for e in elements}
    assert by_text["ACME Corp Annual Report"] == "page_header"
    assert by_text["Confidential draft"] == "page_footer"
    assert by_text["7"] == "page_number"
    assert by_text["iv"] == "page_number"
    # a bare number between body lines is not furniture, whatever the signals say
    assert by_text["12"] == "paragraph"
    assert [e.kind for e in elements][:2] == ["page_header", "page_number"]
    assert [e.kind for e in elements][-2:] == ["page_footer", "page_number"]


def test_contents_page_one_element():
    elements = contents_doc().build()
    contents = of_kind(elements, "contents")
    assert len(contents) == 1
    assert contents[0].text.split("\n") == [
        f"Chapter {k} title ........ {k * 4}" for k in range(1, 6)
    ]
    assert contents[0].boxes[0].page == 1
    assert kinds(elements) == ["heading", "contents", "heading", "paragraph"]


def test_table_consumes_its_lines():
    doc = table_doc()
    typed = doc.typed()
    elements = [e for e, _ in typed]
    assert kinds(elements) == ["paragraph", "table", "paragraph"]
    table, table_lines = typed[1]
    assert table.text == "Item | 2024 | 2023\nRevenue | 12 | 10\nCosts |  | 4"
    assert table.table is not None and table.table.rows[1] == ["Revenue", "12", "10"]
    assert [b.model_dump() for b in table.boxes] == [
        {"page": 1, "x0": 72.0, "y0": 150.0, "x1": 540.0, "y1": 210.0}
    ]
    assert sorted(table_lines) == list(range(1, 9))
    assert not any("Revenue" in e.text for e in elements if e.kind != "table")


def test_image_and_vector_figures_consume_label_text():
    elements = figures_doc().build()
    figures = of_kind(elements, "figure")
    assert [(f.figure.source, f.figure.text, f.text) for f in figures] == [
        ("image", "Axis A", "Axis A"),  # spec §4: a figure's text is its label text
        ("vector", "Q1\nQ2\nQ3", "Q1\nQ2\nQ3"),
    ]
    assert [(f.boxes[0].x0, f.boxes[0].y0) for f in figures] == [(100, 100), (100, 350)]
    # the tiny image, the table rulings and the dense text box are not figures
    assert len(of_kind(elements, "table")) == 1
    notes = " ".join(e.text for e in of_kind(elements, "paragraph"))
    assert "Note 0 text" in notes and "Note 7 text" in notes
    assert not any("Axis A" in e.text or "Q2" in e.text for e in elements if e.kind != "figure")


def test_background_image_with_text_is_not_a_figure():
    elements = background_doc().build()
    assert of_kind(elements, "figure") == []
    text = " ".join(e.text for e in elements)
    assert "Scanned body line 29" in text and "Callout line 5" in text and "Cover title" in text


def test_caption_links_both_ways_above_table_below_figure():
    elements = captions_doc().build()
    page1 = [e for e in elements if e.boxes[0].page == 1]
    assert kinds(page1) == ["figure", "caption", "caption", "table", "paragraph"]
    figure, fig_caption, table_caption, table, _ = page1
    assert fig_caption.text == "Figure 1: Sales by region and by quarter."
    assert table_caption.text == "Table 2. Costs"
    assert (figure.caption_id, fig_caption.caption_for) == (fig_caption.id, figure.id)
    assert (table.caption_id, table_caption.caption_for) == (table_caption.id, table.id)
    # a "Figure" sentence far from any figure, and a word that only looks like a numeral
    assert page1[-1].text == "Figure 3 shows growth in every region."
    page2 = [e for e in elements if e.boxes[0].page == 2]
    assert kinds(page2) == ["figure", "paragraph"]
    assert page2[0].caption_id is None


def test_caption_regex_is_the_spec_pattern():
    assert CAPTION_RE.match("Fig. 3 — growth")
    assert CAPTION_RE.match("exhibit IV: map")
    assert CAPTION_RE.match("Diagram 2-")
    assert not CAPTION_RE.match("Figures show")


def test_title_and_heading_levels():
    doc = headings_doc()
    elements = doc.build()
    assert [(e.kind, e.level, e.text) for e in elements if e.kind in ("title", "heading")] == [
        ("title", 1, "Annual Report 2024 Edition"),
        ("heading", 2, "1. Overview"),
        ("heading", 2, "2. Results"),
    ]
    assert len(elements[0].boxes) == 1 and elements[0].boxes[0].y1 == pytest.approx(118.8)
    assert all(e.section_id is None for e in elements)
    doc.title_index = index_of(doc.rows, "2. Results")  # not on page 1: stays a heading
    assert of_kind(doc.build(), "title") == []
    doc.title_index = index_of(doc.rows, "Opening words of the report.")  # not a heading
    titles = of_kind(doc.build(), "title")
    assert [(t.text, t.level) for t in titles] == [("Opening words of the report.", None)]


def test_footnote_merges_until_next_marker():
    elements = footnotes_doc().build()
    notes = of_kind(elements, "footnote")
    assert [(n.marker, n.text) for n in notes] == [
        ("1", "First note text continues here."),
        ("2", "Second note"),
        ("*", "Star note"),
        ("†", "Dagger note"),
        ("(5)", "Parenthesised note"),
    ]
    other = [e.text for e in elements if e.kind != "footnote"]
    assert any(t.startswith("3 apples") for t in other)
    assert any("1 not a footnote" in t for t in other)
    assert any(t.startswith("688 $") for t in other)  # a table row in small type


def test_bullets_enumerators_and_depth():
    elements = lists_doc().build()
    items = of_kind(elements, "list_item")
    assert [(i.marker, i.list_depth) for i in items] == [
        ("•", 0), ("◦", 1), ("▪", 2), ("-", 0), ("–", 0), ("*", 0), ("1.", 0), ("1)", 0),
        ("(a)", 1), ("a.", 1), ("i.", 2), ("(iv)", 2),
    ]  # fmt: skip
    assert items[0].text == "Alpha item wraps onto a second line"
    assert items[-1].text == "roman paren"
    assert [e.text for e in of_kind(elements, "paragraph")] == [
        "-5% decline in sales was reported. 1.5 million units shipped."
    ]


def test_formula_by_symbols_and_math_font():
    elements = formulas_doc().build()
    assert [e.text for e in of_kind(elements, "formula")] == ["E = mc² + α·β (1)", "x y z"]
    paragraphs = " ".join(e.text for e in of_kind(elements, "paragraph"))
    assert "We set a = b" in paragraphs and "= = ±" in paragraphs


def test_formula_ignores_greek_words():
    doc = Doc(column(1, ["Ελληνικό κείμενο σε μια γραμμή."], 100))
    assert kinds(doc.build()) == ["paragraph"]


def test_paragraph_breaks_gap_size_indent():
    elements = paragraphs_doc().build()
    assert [e.text for e in elements] == [
        "First paragraph line one runs on and continues on the second line.",
        "Second paragraph after a gap keeps going here.",
        "Larger type paragraph",
        "Back to body text that ends.",
        "Indented first line of a new paragraph continues at the margin and an indented line",
    ]
    assert set(kinds(elements)) == {"paragraph"}


def test_paragraph_continues_across_page_with_flag():
    elements = continued_doc().build()
    paragraphs = of_kind(elements, "paragraph")
    assert [p.text for p in paragraphs] == [
        "Opening paragraph ends here.",
        "The results of the study were consistent with earlier work.",
    ]
    continued = paragraphs[1]
    assert continued.flags == ["continued"]
    assert [b.page for b in continued.boxes] == [1, 2]
    assert kinds(elements) == ["paragraph", "paragraph", "page_number", "page_header"]


def test_paragraph_continues_across_column():
    elements = columns_doc().build()
    assert [(e.text, len(e.boxes), e.flags) for e in elements] == [
        ("Left column text starts and the left column text goes on into the right column.", 2,
         ["continued"]),
        ("A second right paragraph.", 1, []),
    ]  # fmt: skip


def test_paragraph_does_not_continue_after_terminal_punctuation():
    elements = not_continued_doc().build()
    assert [e.text for e in elements] == [
        "The results of the study were final.",
        "consistent with earlier work.",
        "The second study was",
        "Consistent with the first one.",
    ]
    assert all(e.flags == [] and len(e.boxes) == 1 for e in elements)


def test_elements_are_numbered_in_reading_order():
    for name, make in FIXTURES.items():
        elements = make().build()
        assert [e.id for e in elements] == [f"e{k}" for k in range(len(elements))], name
        assert [e.order for e in elements] == list(range(len(elements))), name


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_each_line_in_at_most_one_element(name):
    _check_line_ownership(FIXTURES[name]())


def _check_line_ownership(doc):
    typed = doc.typed()
    lines = doc.lines
    counts = Counter(i for _, owned in typed for i in owned)
    assert all(n == 1 for n in counts.values()), counts
    # paragraphs take the rest: every line with text belongs to exactly one element
    assert set(counts) == {i for i, ln in enumerate(lines) if ln.text.strip()}
    for element, _ in typed:
        assert element.boxes, element
        assert element.kind == "figure" or element.text.strip(), element
        for box in element.boxes:
            assert 0 <= box.x0 <= box.x1 <= W and 0 <= box.y0 <= box.y1 <= H, element


def test_empty_page_yields_no_elements():
    assert Doc([], n_pages=3).build() == []
    assert build_elements([], [], [], DocSignals(body_size=0.0, median_gap=0.0), [],
                          title_index=None) == []  # fmt: skip
    elements = degenerate_doc().build()
    # the blank line, the zero-area image and the slivers on page 2 yield nothing; the image
    # partly off the page is clipped to it
    assert kinds(elements) == ["figure", "paragraph"]
    assert elements[0].boxes[0].model_dump() == {"page": 1, "x0": 0, "y0": 0, "x1": 200, "y1": 200}
    assert elements[1].text == "One real line."


def test_ocr_lines_produce_source_ocr():
    elements = ocr_doc().build()
    assert [(e.text, e.source) for e in elements] == [
        ("A text layer page.", "text"),
        ("An OCR page line and its second line.", "ocr"),
    ]


def test_rule_lined_statements_are_not_figures():
    elements = statement_doc().build()
    assert of_kind(elements, "figure") == []
    text = " ".join(e.text for e in elements)
    assert "Revenue line 3" in text and "83,727" in text and "Acme Japan KK" in text


def test_prose_over_a_picture_or_shape_is_not_a_figure():
    elements = prose_box_doc().build()
    assert of_kind(elements, "figure") == []
    assert [e.boxes[0].page for e in of_kind(elements, "paragraph")] == [1, 2]


def test_heading_on_a_banner_is_kept_and_the_banner_is_not_a_figure():
    elements = banner_doc().build()
    assert [(e.kind, e.text) for e in elements][:1] == [("heading", "Risk Factors")]
    assert of_kind(elements, "figure") == []


def test_code_and_list_on_a_shape_are_not_figures():
    elements = shaded_code_and_list_doc().build()
    assert of_kind(elements, "figure") == []
    assert [e.text for e in of_kind(elements, "list_item")] == ["First point", "Second point"]
    assert any("print(y)" in e.text for e in of_kind(elements, "paragraph"))


def test_chart_keeps_its_labels_caption_and_the_paragraph_above():
    elements = chart_doc().build()
    figures = of_kind(elements, "figure")
    assert len(figures) == 1
    labels = figures[0].figure.text.split("\n")
    assert {"30", "300", "Series A", "2023", "Percent"} <= set(labels)
    assert "Figure 5: Rates over time" not in labels
    assert not any("last line" in t for t in labels)
    caption = of_kind(elements, "caption")
    assert [c.text for c in caption] == ["Figure 5: Rates over time"]
    assert (caption[0].caption_for, figures[0].caption_id) == (figures[0].id, caption[0].id)
    assert of_kind(elements, "paragraph")[0].text == (
        "This paragraph runs on with a long first line and its last line ends inside the box"
    )


def test_caption_takes_at_most_six_lines():
    assert el.CAPTION_MAX_LINES == 6  # 4 before the E11a tuning
    elements = long_caption_doc().build()
    assert of_kind(elements, "caption")[0].text == (
        "Figure 4: A long caption continues on a second line and a third line and a fourth line"
        " and a fifth line and a sixth line"
    )
    assert [e.text for e in of_kind(elements, "paragraph")] == ["and a seventh line"]


def test_caption_side_and_kind_preferences():
    assert el.CAPTION_FIRST_SIDE == {"table": "above", "figure": "below"}
    elements = caption_sides_doc().build()
    targets = {e.id: e for e in elements}
    linked = {c.text: targets[c.caption_for] for c in of_kind(elements, "caption")}
    # equal distances: a figure takes the caption below it, a table the caption above it
    assert (linked["Figure 2: Between"].kind, linked["Figure 2: Between"].boxes[0].y0) == (
        "figure",
        100,
    )
    assert (linked["Table 3. Between"].kind, linked["Table 3. Between"].boxes[0].y0) == (
        "table",
        240,
    )
    # a "Table" caption prefers a table to a figure
    assert linked["Table 1. Mixed"].kind == "table"


def test_line_source_must_match_lines():
    doc = paragraphs_doc()
    doc.line_source = ["text"]
    with pytest.raises(ValueError, match="line_source"):
        doc.build()


def test_order_must_be_a_permutation():
    doc = paragraphs_doc()
    doc.order = [0, 0, 1]
    with pytest.raises(ValueError, match="permutation"):
        doc.build()


def test_threshold_constants():
    # Tuned on DocLayNet train pages in E11a (docs/benchmarks-elements.md has the log); the
    # spec §5.3 starting values were 1.5, 0.25, 0.9 and 1.5. The figure thresholds keep the
    # spec / E5a values (controller ruling: statement rows and banners must not be figures).
    assert (el.FIGURE_MIN_AREA, el.FIGURE_MAX_TEXT_DENSITY, el.CAPTION_MAX_DISTANCE_LINES) == (
        0.02, 0.5, 5.0,
    )  # fmt: skip
    assert (el.FOOTNOTE_BOTTOM_SHARE, el.FOOTNOTE_MAX_SIZE_RATIO, el.LIST_INDENT_STEP) == (
        0.3, 0.85, 8.0,
    )  # fmt: skip
    assert (el.FORMULA_MIN_SYMBOL_SHARE, el.PARAGRAPH_GAP_RATIO, el.PARAGRAPH_SIZE_DELTA) == (
        0.30, 1.75, 0.5,
    )  # fmt: skip
    assert el.FIRST_LINE_INDENT == 8.0
    assert (el.FIGURE_MAX_TEXT_COVERAGE, el.FIGURE_PROSE_MIN_WORDS) == (0.3, 5)


# --- generated PDFs through the real pipeline ----------------------------------------------


def _generated_pdf(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page(width=W, height=H)
    page.insert_text((72, 60), "Quarterly Review", fontsize=24, fontname="hebo")
    page.insert_text((72, 100), "1. Summary", fontsize=14, fontname="hebo")
    y = 125
    for k in range(4):
        page.insert_text(
            (72, y), f"Summary sentence {k} explains the quarter in plain words.", fontsize=10
        )
        y += 13
    for k in range(3):  # base-14 Helvetica has no "•" glyph
        page.insert_text((80, y + 8 + 13 * k), f"- Bullet point number {k}", fontsize=10)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 40), False)
    pix.set_rect(pix.irect, (30, 120, 200))
    page.insert_image(pymupdf.Rect(100, 260, 400, 420), stream=pix.tobytes("png"))
    page.insert_text((100, 434), "Figure 1: Revenue by quarter", fontsize=9)
    page.insert_text((72, 470), "Table 1. Results", fontsize=9)
    x0, y0, cw, rh = 72, 480, 150, 20
    for r in range(4):
        page.draw_line((x0, y0 + r * rh), (x0 + 3 * cw, y0 + r * rh))
    for c in range(4):
        page.draw_line((x0 + c * cw, y0), (x0 + c * cw, y0 + 3 * rh))
    for r in range(3):
        for c in range(3):
            page.insert_text((x0 + c * cw + 4, y0 + r * rh + 14), f"r{r}c{c}", fontsize=9)
    page.insert_text((72, 700), "1 A footnote set in small type.", fontsize=7)
    page.insert_text((300, 770), "1", fontsize=9)
    page = doc.new_page(width=W, height=H)
    for side, x in (("left", 72), ("right", 320)):
        for k in range(12):
            page.insert_text((x, 100 + 13 * k), f"The {side} column line {k} runs on", fontsize=10)
    page.insert_text((300, 770), "2", fontsize=9)
    path = tmp_path / "gen.pdf"
    doc.save(path)
    doc.close()
    return path


def _pipeline(path):
    from docrooter.pdf.headings import find_headings
    from docrooter.pdf.objects import extract_page_objects
    from docrooter.pdf.reading_order import order_lines
    from docrooter.pdf.signals import compute_signals

    pages = extract_page_objects(path, engine="pymupdf")
    lines = [ln for p in pages for ln in p.lines]
    signals = compute_signals(lines, len(pages))
    order = order_lines(lines, signals.excluded)
    heads = find_headings(lines, signals)
    title = heads[0].line_index if heads and heads[0].page == 1 else None
    return pages, lines, order, signals, heads, title


def test_generated_pdf_end_to_end(tmp_path):
    pages, lines, order, signals, heads, title = _pipeline(_generated_pdf(tmp_path))
    typed = build_elements_with_lines(pages, lines, order, signals, heads, title_index=title)
    counts = Counter(i for _, owned in typed for i in owned)
    assert all(n == 1 for n in counts.values())
    assert set(counts) == {i for i, ln in enumerate(lines) if ln.text.strip()}
    found = Counter(e.kind for e, _ in typed)
    for kind in ("title", "heading", "list_item", "figure", "caption", "table", "footnote",
                 "page_number", "paragraph"):  # fmt: skip
        assert found[kind] >= 1, (kind, found)
    table = next(e for e, _ in typed if e.kind == "table")
    assert table.text.startswith("r0c0 | r0c1 | r0c2")
    assert all(e.caption_id for e, _ in typed if e.kind in ("figure", "table"))
    page2 = [e for e, _ in typed if e.kind == "paragraph" and e.boxes[0].page == 2]
    assert page2 and "left column line 0" in page2[0].text
