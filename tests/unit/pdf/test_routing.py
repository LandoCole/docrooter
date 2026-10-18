import pytest

from docrooter.elements import Box
from docrooter.pdf.lines import Line
from docrooter.pdf.objects import PageObjects
from docrooter.pdf.routing import RouteDecision, page_metrics, route_page

W, H = 612.0, 792.0
SENTENCE = "The quick brown fox jumps over the lazy dog while reading"


def _line(text, y0=100.0, x0=72.0, x1=540.0, page=1):
    return Line(
        page=page,
        text=text,
        size=11.0,
        bold=False,
        font="Helvetica",
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y0 + 12,
        page_height=H,
        page_width=W,
        gap_above=4.0,
        gap_below=4.0,
        block=0,
    )


def _page(lines=(), images=(), *, char_count=None, garbled=0, invisible=0, width=W, height=H):
    lines = list(lines)
    if char_count is None:
        char_count = sum(len("".join(ln.text.split())) for ln in lines)
    return PageObjects(
        page=1,
        width=width,
        height=height,
        rotation=0,
        lines=lines,
        images=list(images),
        drawings=[],
        tables=[],
        char_count=char_count,
        invisible_chars=invisible,
        garbled_chars=garbled,
        n_drawing_paths=0,
    )


def _text_lines(n=40):
    return [_line(SENTENCE, y0=60.0 + 14 * k) for k in range(n)]


def _box(x0, y0, x1, y1):
    return Box(page=1, x0=x0, y0=y0, x1=x1, y1=y1)


def test_text_page_routes_to_text():
    route, regions, decision = route_page(_page(_text_lines()))
    assert route == "text"
    assert regions == []
    assert isinstance(decision, RouteDecision)
    assert (decision.scope, decision.target, decision.task) == ("page", 1, "text")
    assert set(decision.metrics) == {
        "char_density",
        "image_cov",
        "garbled",
        "invisible_text",
        "latin_share",
        "images_capped",
    }


def test_empty_page_routes_to_text():
    route, regions, _ = route_page(_page())
    assert route == "text"
    assert regions == []


def test_scanned_page_routes_to_ocr():
    # a 90% image over 47 characters: char_density 0.5 per square inch is below 2
    page = _page([_line("scan note")], [_box(0, 0, W, 0.9 * H)], char_count=47)
    metrics = page_metrics(page)
    assert metrics.image_cov == pytest.approx(0.9)
    assert metrics.char_density == pytest.approx(0.5, abs=0.01)
    route, regions, decision = route_page(page)
    assert route == "ocr"
    assert regions == []
    assert (decision.scope, decision.target, decision.task) == ("page", 1, "ocr")
    assert "image" in decision.reason


def test_dense_text_over_a_full_page_image_stays_text():
    # a scan that already carries a good text layer: char_density >= 2
    page = _page(_text_lines(), [_box(0, 0, W, H)])
    assert route_page(page)[0] == "text"


def test_image_coverage_is_a_union():
    page = _page([], [_box(0, 0, W, H / 2), _box(0, 0, W, H / 2), _box(0, H / 4, W, 3 * H / 4)])
    assert page_metrics(page).image_cov == pytest.approx(0.75)
    # beyond the page counts only inside it
    assert page_metrics(_page([], [_box(-100, -100, W + 100, H + 100)])).image_cov == 1.0


def test_garbled_page_routes_to_ocr():
    page = _page(_text_lines(), char_count=1000, garbled=150)  # 15% U+FFFD
    metrics = page_metrics(page)
    assert metrics.garbled == pytest.approx(0.15)
    route, _, decision = route_page(page)
    assert route == "ocr"
    assert "garbled" in decision.reason


def test_slightly_garbled_page_stays_text():
    assert route_page(_page(_text_lines(), char_count=1000, garbled=100))[0] == "text"


def test_invisible_garbled_layer_routes_to_ocr():
    # an OCR layer of render mode 3 text that is mostly junk, below the 10% garbled bar
    lines = _text_lines()
    page = _page(lines, char_count=1000, garbled=50, invisible=900)
    metrics = page_metrics(page)
    assert metrics.invisible_text == pytest.approx(0.9)
    assert metrics.garbled < 0.10
    route, _, decision = route_page(page)
    assert route == "ocr"
    assert "invisible" in decision.reason


def test_invisible_clean_layer_stays_text():
    page = _page(_text_lines(), char_count=1000, garbled=0, invisible=900)
    assert route_page(page)[0] == "text"


def test_symbol_soup_counts_as_garbled():
    junk = [_line("#@%^&*~|<>{}[]=+" * 3, y0=60.0 + 14 * k) for k in range(5)]
    assert page_metrics(_page(junk)).garbled > 0.10
    assert route_page(_page(junk))[0] == "ocr"


def test_consonant_soup_counts_as_garbled():
    junk = [_line("xkcd qwrt zzzz bcdf ghjk lmnp tvwx", y0=60.0 + 14 * k) for k in range(5)]
    metrics = page_metrics(_page(junk))
    assert metrics.latin_share == 1.0
    assert route_page(_page(junk))[0] == "ocr"


def test_numeric_page_is_not_garbled():
    rows = [_line("1,204.5 38.2 (917) 12 4,001 77.3 19 2,050", y0=60.0 + 14 * k) for k in range(5)]
    assert route_page(_page(rows))[0] == "text"


def test_region_ocr_for_textless_large_image():
    area = W * H
    big = _box(72, 300, 72 + 0.06 * area / 200, 500)  # 6% of the page, 200 pt tall
    small = _box(72, 520, 72 + 0.04 * area / 200, 720)  # 4%
    assert big.x1 - big.x0 > 0 and (big.x1 - big.x0) * 200 / area == pytest.approx(0.06)
    route, regions, decision = route_page(_page(_text_lines(8), [big, small]))
    assert route == "text"
    assert regions == [big]
    assert (decision.scope, decision.task) == ("region", "ocr")
    assert decision.target == [big.x0, big.y0, big.x1, big.y1]


def test_image_with_text_inside_is_not_a_region():
    image = _box(50, 50, 560, 400)  # 28% of the page, with lines inside it
    _, regions, _ = route_page(_page(_text_lines(), [image]))
    assert regions == []


def test_cjk_page_is_not_garbled():
    # Review Focus 5: a Chinese text layer has no vowels and no spaces; it is not garbled
    cjk = [
        _line("这是一个关于文档结构提取的中文段落内容很长" * 2, y0=60.0 + 14 * k) for k in range(30)
    ]
    metrics = page_metrics(_page(cjk))
    assert metrics.latin_share == 0.0
    assert metrics.garbled == 0.0
    assert route_page(_page(cjk))[0] == "text"


def test_cyrillic_and_arabic_pages_are_not_garbled():
    for text in (
        "Это обычный русский текст страницы документа " * 2,
        "هذا نص عربي عادي في صفحة " * 3,
    ):
        lines = [_line(text, y0=60.0 + 14 * k) for k in range(30)]
        assert page_metrics(_page(lines)).garbled == 0.0
        assert route_page(_page(lines))[0] == "text"


def test_dictionary_rule_needs_latin_majority():
    # 30 consonant-only tokens: Latin letters -> garbled; the same shape in Cyrillic -> not
    latin = [_line("bcd " * 10, y0=60.0 + 14 * k) for k in range(3)]
    cyrillic = [_line("бвг " * 10, y0=60.0 + 14 * k) for k in range(3)]
    assert page_metrics(_page(latin)).garbled > 0.10
    assert page_metrics(_page(cyrillic)).garbled == 0.0


def _contents_page(leader):
    rows = [
        _line(f"Item {k}. Business and properties {leader * 60} {k + 2}", y0=60.0 + 14 * k)
        for k in range(10)
    ]
    return _page(rows)


@pytest.mark.parametrize(
    "leader",
    [".", "·", "…", "_", "-", ". ", "- ", "._"],
    ids=[
        "dot",
        "middle-dot",
        "ellipsis",
        "underscore",
        "hyphen",
        "spaced-dot",
        "spaced-hyphen",
        "mixed",
    ],
)
def test_contents_page_with_leaders_routes_to_text(leader):
    page = _contents_page(leader)
    route, _, decision = route_page(page)
    assert route == "text", decision.reason
    assert page_metrics(page).garbled == 0.0


def test_symbol_garbled_page_with_leaders_still_routes_to_ocr():
    junk = [_line("#@%^&*~|<>{}[]=+ ..... ~|<>{}[]=+#@%^&*", y0=60.0 + 14 * k) for k in range(5)]
    route, _, decision = route_page(_page(junk))
    assert route == "ocr"
    assert "symbols" in decision.reason


def test_many_image_placements_are_capped_and_bounded():
    import time

    from docrooter.pdf.routing import MAX_IMAGES

    images = [
        _box(10 + (k % 50) * 11, 10 + (k // 50) * 7, 20 + (k % 50) * 11, 15 + (k // 50) * 7)
        for k in range(5000)
    ]
    big = _box(0, 0, 612, 400)  # the largest placement must survive the cap
    page = _page(_text_lines(), [*images, big])
    started = time.perf_counter()
    route, regions, decision = route_page(page)
    assert time.perf_counter() - started < 0.5
    assert decision.metrics["images_capped"] == 5001 - MAX_IMAGES
    assert page_metrics(page).image_cov >= 400 / 792  # the big image was kept
    assert route == "text" and regions == []  # text lines sit inside the big image


def test_few_images_are_not_capped():
    _, _, decision = route_page(_page(_text_lines(), [_box(0, 0, 10, 10)]))
    assert decision.metrics["images_capped"] == 0.0
