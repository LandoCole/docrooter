"""DocLayNet element benchmark scoring: box matching, class mapping, scores and text coverage.

Synthetic boxes only: no DocLayNet data, no PDFs, no models.
"""

import pytest

from benchmarks.elements_bench import (
    CLASSES,
    KIND_TO_CLASS,
    ScoredPage,
    box_iou,
    class_of,
    gold_to_points,
    match_boxes,
    score,
    text_coverage,
)
from docrooter.elements.models import Box


def box(x0, y0, x1, y1, page=1):
    return Box(page=page, x0=x0, y0=y0, x1=x1, y1=y1)


def test_box_iou():
    assert box_iou(box(0, 0, 10, 10), box(0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou(box(0, 0, 10, 10), box(5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert box_iou(box(0, 0, 10, 10), box(20, 20, 30, 30)) == 0.0
    assert box_iou(box(0, 0, 10, 10), box(0, 0, 10, 10, page=2)) == 0.0
    assert box_iou(box(0, 0, 0, 10), box(0, 0, 0, 10)) == 0.0  # zero area


def test_greedy_one_to_one_matching():
    gold = [(box(0, 0, 100, 20), "Text"), (box(0, 30, 100, 50), "Text")]
    pred = [
        (box(0, 0, 100, 22), "paragraph"),  # IoU 0.91 with gold 0
        (box(0, 0, 100, 20), "paragraph"),  # IoU 1.0 with gold 0: takes it first
        (box(0, 28, 100, 50), "paragraph"),  # IoU 0.91 with gold 1
    ]
    pairs = match_boxes(pred, gold)
    assert sorted(pairs) == [(1, 0), (2, 1)]  # pred 0 is left unmatched: one-to-one
    # Below the IoU threshold nothing matches; a lower threshold lets it.
    half = [(box(0, 0, 100, 40), "paragraph")]
    assert match_boxes(half, gold[:1]) == [(0, 0)]  # IoU exactly 0.5 (2000 / 4000) counts
    assert match_boxes([(box(0, 0, 100, 41), "paragraph")], gold[:1]) == []
    assert match_boxes([(box(0, 0, 100, 41), "paragraph")], gold[:1], iou=0.4) == [(0, 0)]


def test_matching_is_per_class():
    gold = [(box(0, 0, 100, 20), "Section-header")]
    assert match_boxes([(box(0, 0, 100, 20), "paragraph")], gold) == []
    assert match_boxes([(box(0, 0, 100, 20), "heading")], gold) == [(0, 0)]
    # A kind with no DocLayNet class never matches.
    assert match_boxes([(box(0, 0, 100, 20), "unknown")], gold) == []


def test_class_mapping_per_spec_table():
    assert set(CLASSES) == {
        "Caption", "Footnote", "Formula", "List-item", "Page-footer", "Page-header",
        "Picture", "Section-header", "Table", "Text", "Title",
    }  # fmt: skip
    expected = {
        "caption": "Caption", "footnote": "Footnote", "formula": "Formula",
        "list_item": "List-item", "page_footer": "Page-footer", "page_number": "Page-footer",
        "page_header": "Page-header", "figure": "Picture", "heading": "Section-header",
        "table": "Table", "paragraph": "Text", "contents": "Text", "title": "Title",
    }  # fmt: skip
    assert expected == KIND_TO_CLASS
    assert class_of("page_number") == "Page-footer"
    assert class_of("contents") == "Text"
    assert class_of("nonsense") is None


def test_gold_to_points_scales_each_axis():
    # DocLayNet boxes are COCO [x, y, w, h] in 1025 x 1025 image pixels; each axis scales by
    # its own page side.
    b = gold_to_points([102.5, 205.0, 102.5, 51.25], 612.0, 792.0, coco=(1025, 1025))
    assert (b.x0, b.y0, b.x1, b.y1) == pytest.approx((61.2, 158.4, 122.4, 198.0))
    assert b.page == 1


def test_score_per_class_and_macro():
    page = ScoredPage(
        page_id="p1",
        category="financial_reports",
        pred=[
            (box(0, 0, 100, 20), "paragraph"),  # TP Text
            (box(0, 100, 100, 200), "figure"),  # FP Picture (no gold)
            (box(0, 300, 100, 320), "page_number"),  # TP Page-footer
        ],
        gold=[
            (box(0, 0, 100, 20), "Text"),
            (box(0, 50, 100, 70), "Text"),  # FN Text
            (box(0, 300, 100, 320), "Page-footer"),
        ],
    )
    result = score([page])
    text = result["per_class"]["Text"]
    assert (text["tp"], text["fp"], text["fn"]) == (1, 0, 1)
    assert text["precision"] == pytest.approx(1.0)
    assert text["recall"] == pytest.approx(0.5)
    assert text["f1"] == pytest.approx(2 / 3)
    assert result["per_class"]["Picture"]["f1"] == 0.0
    assert result["per_class"]["Page-footer"]["f1"] == pytest.approx(1.0)
    # Macro over the classes present in gold or predictions (Text, Picture, Page-footer).
    assert result["macro"] == pytest.approx((2 / 3 + 0.0 + 1.0) / 3)
    assert result["per_class"]["Formula"]["f1"] is None  # absent: not in the macro
    assert result["per_category"]["financial_reports"]["macro"] == pytest.approx(result["macro"])
    assert result["pages"] == 1


def test_per_category_macro_is_separate():
    a = ScoredPage("a", "patents", pred=[(box(0, 0, 10, 10), "paragraph")],
                   gold=[(box(0, 0, 10, 10), "Text")])  # fmt: skip
    b = ScoredPage("b", "manuals", pred=[], gold=[(box(0, 0, 10, 10), "Text")])
    result = score([a, b])
    assert result["per_category"]["patents"]["macro"] == pytest.approx(1.0)
    assert result["per_category"]["manuals"]["macro"] == pytest.approx(0.0)
    assert result["per_class"]["Text"]["f1"] == pytest.approx(2 / 3)


def test_text_coverage_score():
    page = ScoredPage(
        page_id="p1",
        category="manuals",
        pred=[
            (box(0, 0, 200, 40), "paragraph"),
            (box(0, 50, 200, 70), "paragraph"),  # gold calls this a heading
            (box(0, 100, 200, 300), "figure"),
        ],
        gold=[
            (box(0, 0, 200, 70), "Text"),  # one region for two predicted paragraphs
            (box(0, 80, 200, 90), "Section-header"),
            (box(0, 100, 200, 300), "Picture"),
        ],
        # (cell box, characters, index of its gold box)
        cells=[
            (box(0, 0, 200, 10), 40, 0),
            (box(0, 55, 200, 65), 60, 0),
            (box(0, 80, 100, 90), 10, 1),  # in no predicted element
            (box(10, 110, 50, 120), 5, 2),
        ],
    )
    cov = text_coverage([page])
    # Text: both cells land in a paragraph, though IoU matching would split them.
    assert cov["Text"] == {"chars": 100, "covered": 100, "share": pytest.approx(1.0)}
    assert cov["Section-header"]["share"] == 0.0
    assert cov["Picture"]["share"] == pytest.approx(1.0)
    assert cov["Formula"] == {"chars": 0, "covered": 0, "share": None}
    assert score([page])["text_coverage"]["Text"]["share"] == pytest.approx(1.0)


def test_text_coverage_uses_the_most_overlapping_element():
    # A cell that straddles two predicted elements belongs to the one it overlaps most.
    page = ScoredPage(
        page_id="p",
        category="manuals",
        pred=[(box(0, 0, 100, 12), "heading"), (box(0, 12, 100, 100), "paragraph")],
        gold=[(box(0, 0, 100, 100), "Text")],
        cells=[(box(0, 8, 100, 18), 20, 0)],  # 4 pt in the heading, 6 pt in the paragraph
    )
    assert text_coverage([page])["Text"]["covered"] == 20


def test_threshold_overrides_are_restored():
    from benchmarks.elements_bench import overrides
    from docrooter.pdf import elements as rules

    before = rules.PARAGRAPH_GAP_RATIO
    with overrides({"PARAGRAPH_GAP_RATIO": 2.0, "CAPTION_MAX_LINES": 3.0}):
        assert rules.PARAGRAPH_GAP_RATIO == 2.0
        assert rules.CAPTION_MAX_LINES == 3 and isinstance(rules.CAPTION_MAX_LINES, int)
    assert before == rules.PARAGRAPH_GAP_RATIO
    with pytest.raises(SystemExit), overrides({"CAPTION_FIRST_SIDE": 1.0}):
        pass
