"""OCR benchmark scoring: text normalisation, CER, WER, page text order and routing confusion.

Strings and tuples only: no PDFs, no OCR engine.
"""

import pytest

from benchmarks.ocr_bench import (
    DECIDED,
    cer,
    edit_distance,
    kind_agreement,
    normalize_text,
    page_text,
    routing_confusion,
    wer,
    word_overlap,
)


def test_cer_normalisation():
    # NFKC: the "fi" ligature and a full-width digit become plain characters
    assert normalize_text("ﬁne １") == "fine 1"
    # whitespace runs (spaces, tabs, newlines) collapse to one space; ends are stripped
    assert normalize_text("  a \t b\n\n c  ") == "a b c"
    # a word hyphenated at a line end is joined; a hyphen before a capital or mid-line stays
    assert normalize_text("infor-\nmation") == "information"
    assert normalize_text("infor-  \n  mation") == "information"
    assert normalize_text("Anglo-\nSaxon well-known") == "Anglo- Saxon well-known"
    # so the same text, laid out differently, has no character errors
    assert cer("The infor-\nmation ﬁle", "The information  file") == 0.0
    assert cer("abcd", "abed") == pytest.approx(0.25)
    assert cer("abc", "") == pytest.approx(1.0)
    assert cer("", "") == 0.0
    assert cer("", "x") == pytest.approx(1.0)  # insertions against an empty reference


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("flaw", "lawn") == 2
    assert edit_distance(["a", "b", "c"], ["a", "c"]) == 1
    long_a, long_b = "ab" * 500, "ba" * 500
    assert edit_distance(long_a, long_b) == 2  # shift by one: delete the first, add the last


def test_wer():
    assert wer("the cat sat", "the cat sat") == 0.0
    assert wer("the cat sat", "the bat sat") == pytest.approx(1 / 3)
    assert wer("the cat sat on", "the cat") == pytest.approx(2 / 4)
    assert wer("infor-\nmation retrieval", "information retrieval") == 0.0
    assert wer("", "") == 0.0


def test_page_text_orders_rows_then_columns():
    # (text, x0, y0, x1, y1): two columns whose lines share rows, given out of order
    items = [
        ("right one", 300, 100, 500, 112),
        ("left two", 50, 120, 250, 132),
        ("left one", 50, 101, 250, 113),
        ("right two", 300, 121, 500, 133),
    ]
    assert page_text(items) == "left one right one\nleft two right two"
    assert page_text([]) == ""


def test_routing_confusion_counts():
    pairs = [("text", "text")] * 5 + [("text", "text+ocr"), ("ocr", "ocr"), ("ocr", "ocr")]
    pairs.append(("ocr", "text"))
    table = routing_confusion(pairs)
    assert table == {
        "text": {"text": 5, "text+ocr": 1, "ocr": 0},
        "ocr": {"text": 1, "text+ocr": 0, "ocr": 2},
    }
    assert tuple(table["text"]) == DECIDED
    with pytest.raises(ValueError):
        routing_confusion([("text", "region")])  # region decisions are counted as text first


def test_kind_agreement():
    # (kind, x0, y0, x1, y1, chars)
    text_layer = [
        ("heading", 50, 50, 300, 70, 10),
        ("paragraph", 50, 80, 500, 200, 300),
        ("table", 50, 220, 500, 400, 200),
        ("paragraph", 50, 600, 500, 650, 50),
    ]
    ocr = [
        ("heading", 52, 51, 290, 69, 10),
        ("paragraph", 50, 82, 500, 198, 290),
        ("paragraph", 50, 222, 500, 398, 180),  # the table read as a paragraph
    ]
    out = kind_agreement(text_layer, ocr)
    assert out["elements"] == 4
    assert out["agree"] == 2
    assert out["missed"] == 1  # nothing overlaps the last paragraph
    assert out["agree_chars"] == 310
    assert out["chars"] == 560
    assert out["confusion"]["table"] == {"paragraph": 1}
    assert out["confusion"]["paragraph"] == {"paragraph": 1, "-": 1}


def test_word_overlap_ignores_order():
    assert word_overlap("a b c d", "d c b a") == (4, 4, 4)
    assert word_overlap("a a b", "a b b x") == (2, 3, 4)  # multiset: one "a", one "b"
    assert word_overlap("infor-\nmation", "information") == (1, 1, 1)
    assert word_overlap("", "") == (0, 0, 0)
