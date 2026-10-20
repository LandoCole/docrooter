"""Calls bounded by size: batched and per-item prompts stay within ``max_call_tokens``.

Model-free and network-free: the keyword backend, whose tokens are estimated as the calls an
LLM would be sent.
"""

from __future__ import annotations

import pytest

from docrooter import classify
from docrooter.backends import KeywordBackend
from docrooter.batch import (
    MAX_BATCH,
    call_budget,
    cut_to_tokens,
    group_leaves,
    group_prompt,
    pack,
    section_prompt,
)
from docrooter.document import Document
from docrooter.ladder import FULL_CAP
from docrooter.taxonomies import builtin_taxonomy
from docrooter.views import estimate_tokens

CATALOGUE = builtin_taxonomy("section_catalogue")


def financial_leaves(views):
    return section_prompt(views, CATALOGUE, group_leaves(CATALOGUE, "financial"))


def words(n, word="figure"):
    return " ".join([word] * n)


def test_call_budget_keeps_a_reserve_for_the_estimate():
    assert call_budget() == call_budget(FULL_CAP) == 13600
    assert call_budget(1000) == 850
    with pytest.raises(ValueError):
        call_budget(0)


def test_cut_to_tokens_keeps_the_head_in_whole_words():
    assert cut_to_tokens("one  two three four", 3) == "one  two"
    assert cut_to_tokens("one two", 100) == "one two"
    assert cut_to_tokens("one two", 1) == ""
    text = words(500)
    cut = cut_to_tokens(text, 130)
    assert estimate_tokens(cut) <= 130 < estimate_tokens(cut + " figure")
    assert text.startswith(cut)


def test_pack_bounds_each_call_by_size_and_by_count():
    views = {str(i + 1): words(300) for i in range(30)}
    calls = pack(views, financial_leaves, budget=3000)
    assert [n for call in calls for n in call] == list(views)  # in order, each once
    assert all(call[n] == views[n] for call in calls for n in call)  # nothing cut
    assert len(calls) > 1
    for call in calls:
        assert financial_leaves(call).retry().estimate() <= 3000

    small = {str(i + 1): "x" for i in range(45)}
    assert [len(c) for c in pack(small, financial_leaves, budget=100_000)] == [MAX_BATCH, 5]


def test_pack_cuts_a_view_too_large_for_any_call_and_sends_it_alone():
    views = {"1": "short", "2": words(5000, "long"), "3": "short"}
    calls = pack(views, lambda v: group_prompt(v, CATALOGUE), budget=3000)
    assert [list(call) for call in calls] == [["1"], ["2"], ["3"]]
    cut = calls[1]["2"]
    assert cut != views["2"] and views["2"].startswith(cut) and cut
    assert group_prompt(calls[1], CATALOGUE).retry().estimate() <= 3000


def test_pack_refuses_a_budget_the_options_fill():
    with pytest.raises(ValueError, match="max_call_tokens"):
        pack({"1": "x"}, financial_leaves, budget=50)


def long_doc(n=40, n_words=2500):
    """``n`` level-1 sections whose ``full`` view reaches the 2,048-token section budget."""
    titles = ["Balance sheet", "Risk factors", "Outlook", "Properties"]
    return Document.from_sections(
        [
            {"title": f"{titles[i % 4]} {i}", "level": 1, "text": words(n_words, f"w{i}")}
            for i in range(n)
        ]
    )


@pytest.mark.parametrize("calls", ["batched", "batched_flat"])
def test_batched_section_calls_stay_within_the_cap(calls):
    result = classify(long_doc(n_words=1400), backend=KeywordBackend(), level="full", calls=calls)
    section_calls = result.section_calls
    assert section_calls and all(c.prompt_tokens <= FULL_CAP for c in section_calls)
    # 40 views of ~2,048 tokens cannot share one call: the stage takes several
    assert len(section_calls) > (2 if calls == "batched_flat" else 5)
    assert sum(c.prompt_tokens for c in section_calls) == result.stage_tokens["sections"].prompt
    assert len(section_calls) == result.stage_tokens["sections"].calls
    assert not any(c.truncated for c in section_calls)  # packing alone did it: nothing cut


def test_a_smaller_max_call_tokens_cuts_views_and_flags_them():
    result = classify(long_doc(4), backend=KeywordBackend(), level="full", max_call_tokens=2500)
    assert all(c.prompt_tokens <= 2500 for c in result.section_calls)
    assert any(c.truncated for c in result.section_calls) and "truncated" in result.flags
    assert all(c.prompt_tokens <= 2500 for c in result.document_calls)


def test_per_item_calls_follow_the_same_limit():
    result = classify(
        long_doc(3), backend=KeywordBackend(), level="full", calls="per_item",
        max_call_tokens=2000,
    )  # fmt: skip
    assert result.section_calls and all(c.prompt_tokens <= 2000 for c in result.section_calls)
    assert "truncated" in result.flags
    assert all(c.options_count >= 1 for c in result.section_calls)


def test_max_call_tokens_is_checked():
    with pytest.raises(ValueError, match="max_call_tokens"):
        classify(long_doc(1), backend=KeywordBackend(), max_call_tokens=0)
    with pytest.raises(ValueError, match="max_call_tokens"):
        classify(long_doc(1), backend=KeywordBackend(), max_call_tokens=100)
