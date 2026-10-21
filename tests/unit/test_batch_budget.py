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
from docrooter.taxonomy import Taxonomy
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


# --- the document call: its view is budgeted from the real prompt overhead ----------------


def big_doc_types():
    """The built-in doc types with descriptions four times as long (an options listing of
    about 4,000 tokens)."""
    base = builtin_taxonomy("doc_types")
    return Taxonomy(
        [
            n.model_copy(
                update={"description": " ".join([n.description or ""] * 4), "children": []}
            )
            for n in base.nodes.values()
        ]
    )


@pytest.mark.parametrize("calls", ["batched", "per_item"])
@pytest.mark.parametrize("flow", ["doc_first", "sections_first"])
def test_the_document_call_fits_a_small_cap(calls, flow):
    result = classify(
        long_doc(6, 800), backend=KeywordBackend(), flow=flow, level="lead", calls=calls,
        max_call_tokens=1500,
    )  # fmt: skip
    assert result.document_calls
    assert all(c.prompt_tokens <= call_budget(1500) for c in result.document_calls)
    assert result.doc_truncated == result.document_calls[-1].truncated
    if calls == "batched":  # its option listing leaves the view about 130 tokens: cut
        assert result.doc_truncated


@pytest.mark.parametrize("calls", ["batched", "per_item"])
def test_the_document_call_fits_the_cap_with_a_large_doc_type_taxonomy(calls):
    result = classify(
        long_doc(40, 1400), backend=KeywordBackend(), flow="doc_first", level="full",
        calls=calls, doc_types=big_doc_types(),
    )  # fmt: skip
    (call,) = result.document_calls
    assert call.prompt_tokens <= call_budget() and call.truncated and result.doc_truncated


def test_a_cap_the_document_options_fill_raises():
    for calls in ("batched", "per_item"):
        with pytest.raises(ValueError, match="document"):
            classify(long_doc(2, 50), backend=KeywordBackend(), flow="doc_first",
                     calls=calls, max_call_tokens=1200)  # fmt: skip


# --- calibration: the budget scaled by the backend's exact/estimated token ratio ----------


class Calibrated(KeywordBackend):
    """A non-LLM backend that reports a learned token ratio, as an LLM backend would."""

    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio

    def token_ratio(self):
        return self.ratio


def test_calls_are_packed_to_the_budget_over_the_token_ratio():
    plain = classify(long_doc(n_words=1400), backend=KeywordBackend(), level="full")
    scaled = classify(long_doc(n_words=1400), backend=Calibrated(1.6), level="full")
    assert all(c.token_ratio == 1.0 for c in plain.section_calls + plain.document_calls)
    assert all(c.token_ratio == 1.6 for c in scaled.section_calls + scaled.document_calls)
    budget = call_budget() / 1.6
    assert all(c.prompt_tokens <= budget for c in scaled.section_calls)
    assert max(c.prompt_tokens for c in scaled.section_calls) > 0.8 * budget  # about 1/1.6
    assert len(scaled.section_calls) > len(plain.section_calls)
    (doc_call,) = scaled.document_calls if scaled.document_calls else (None,)
    if doc_call is not None:
        assert doc_call.prompt_tokens <= budget


def test_the_document_view_follows_the_token_ratio():
    for calls in ("batched", "per_item"):
        result = classify(
            long_doc(40, 1400), backend=Calibrated(2.0), flow="doc_first", level="full",
            calls=calls,
        )  # fmt: skip
        (call,) = result.document_calls
        assert call.token_ratio == 2.0
        assert call_budget() / 2.0 * 0.8 < call.prompt_tokens <= call_budget() / 2.0


def test_the_token_ratio_is_clamped_when_budgeting():
    result = classify(long_doc(40, 1400), backend=Calibrated(9.0), flow="doc_first",
                      level="full")  # fmt: skip
    (call,) = result.document_calls
    assert call.token_ratio == 3.0 and call.prompt_tokens <= call_budget() / 3.0
    low = classify(long_doc(2, 50), backend=Calibrated(0.2), flow="doc_first")
    assert all(c.token_ratio == 1.0 for c in low.document_calls)


def test_an_exact_call_calibrates_the_next_packing():
    httpx = pytest.importorskip("httpx")
    import json

    from docrooter.backends.openai import OpenAIBackend

    sent = []

    def handler(request):
        body = json.loads(request.content)
        estimate = sum(estimate_tokens(m["content"]) for m in body["messages"])
        sent.append(estimate)
        usage = {"prompt_tokens": round(1.6 * estimate), "completion_tokens": 1}
        reply = {"choices": [{"message": {"role": "assistant", "content": "{}"}}], "usage": usage}
        return httpx.Response(200, json=reply)

    llm = OpenAIBackend("m", base_url="http://llm.test/v1", api_key="k")
    llm._transport = httpx.MockTransport(handler)
    llm.chat_json([{"role": "user", "content": words(500)}])  # the first exact call
    assert llm.token_ratio() == pytest.approx(1.6, abs=0.01)
    sent.clear()
    result = classify(long_doc(n_words=1400), backend=llm, level="full", calls="batched")
    calls = result.section_calls + result.document_calls
    assert all(c.exact and c.token_ratio == pytest.approx(1.6, abs=0.01) for c in calls)
    assert all(c.prompt_tokens <= FULL_CAP for c in calls)  # exact tokens within the cap
    assert max(sent) <= call_budget() / 1.6 and max(sent) > 0.8 * call_budget() / 1.6
