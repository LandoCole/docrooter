"""Batched LLM calls: one chat call per document stage, the options listed once.

Network-free: the OpenAI-compatible backend talks to an ``httpx.MockTransport`` that answers
from a script and reports usage.
"""

from __future__ import annotations

import json
import re

import pytest

httpx = pytest.importorskip("httpx")

from docrooter import classify  # noqa: E402
from docrooter.backends import KeywordBackend  # noqa: E402
from docrooter.backends.base import TokenUsage  # noqa: E402
from docrooter.backends.openai import OpenAIBackend  # noqa: E402
from docrooter.batch import document_prompt, section_prompt  # noqa: E402
from docrooter.document import Document  # noqa: E402
from docrooter.errors import BackendError  # noqa: E402
from docrooter.taxonomies import builtin_taxonomy  # noqa: E402

CATALOGUE = builtin_taxonomy("section_catalogue")
DOC_TYPES = builtin_taxonomy("doc_types")

SECTIONS = {
    "Notes to the financial statements": "financial.notes_to_financial_statements",
    "Management discussion": "narrative.management_discussion",
    "Executive compensation": "disclosure.executive_compensation",
    "Balance sheet": "financial.balance_sheet",
    "Income statement": "financial.income_statement",
    "Cash flows": "financial.cash_flow_statement",
    "Letter to shareholders": "narrative.shareholder_letter",
    "Outlook": "narrative.outlook",
    "Our business": "narrative.business_overview",
    "Risk factors": "disclosure.risk_factors",
    "Properties": "disclosure.properties",
    "Legal proceedings": "disclosure.legal_proceedings",
    "Opinion": "law.opinion",
    "Syllabus": "law.syllabus",
}


def financial_doc(*extra):
    return Document.from_sections(
        [
            {
                "title": "Notes to the financial statements",
                "level": 1,
                "text": "Note one sets out the accounting policies and revenue recognition.",
            },
            {
                "title": "Management discussion",
                "level": 1,
                "text": "Results of operations improved as revenue grew across segments.",
            },
            *extra,
        ],
        source="report.md",
    )


def numbered(user: str) -> dict[str, str]:
    """``{number: view}`` of a batched sections user turn."""
    parts = re.split(r"^\[(\d+)\] ", user, flags=re.MULTILINE)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}


class FakeLLM:
    """Answers batched calls: sections by their title (``answer`` for leaves, ``group`` for
    the groups of a two-step stage), the document with ``doc_type``. Every request body is
    kept."""

    def __init__(self, doc_type="financial.quarterly_report", answer=None, group=None):
        self.bodies: list[dict] = []
        self.doc_type = doc_type
        self.answer = answer or (lambda n, view, attempt: _by_title(view))
        self.group = group or (lambda n, view, attempt: _by_title(view).split(".")[0])

    def kind(self, body):
        system = body["messages"][0]["content"]
        if '"doc_type"' in system:
            return "document"
        return "groups" if "group id" in system else "sections"

    def of(self, kind):
        return [b for b in self.bodies if self.kind(b) == kind]

    def __call__(self, request):
        body = json.loads(request.content)
        self.bodies.append(body)
        kind = self.kind(body)
        if kind == "document":
            reply = {"doc_type": self.doc_type}
        else:
            attempt = len(self.of(kind))
            views = numbered(body["messages"][1]["content"])
            answer = self.group if kind == "groups" else self.answer
            reply = {n: answer(n, view, attempt) for n, view in views.items()}
        data = {
            "choices": [{"message": {"role": "assistant", "content": json.dumps(reply)}}],
            "usage": {"prompt_tokens": 1000 + len(self.bodies), "completion_tokens": 20},
        }
        return httpx.Response(200, json=data)


def _by_title(view):
    for title, leaf in SECTIONS.items():
        if f"Title: {title}" in view:
            return leaf
    return "general.body"


def llm(handler):
    backend = OpenAIBackend("gpt-test", base_url="http://llm.test/v1", api_key="k")
    backend._transport = httpx.MockTransport(handler)
    return backend


def test_batched_one_call_per_document_stage():
    fake = FakeLLM()
    result = classify(financial_doc(), backend=llm(fake), calls="batched_flat")
    assert result.calls == "batched_flat"
    assert len(fake.of("sections")) == 1 and len(fake.of("document")) == 1
    assert all(b["response_format"] == {"type": "json_object"} for b in fake.bodies)
    assert result.decided_by == "model" and result.doc_type == "financial.quarterly_report"
    assert result.doc_type_confidence == 1.0
    tags = {t.title: t for t in result.sections}
    assert tags["Management discussion"].label == "narrative.management_discussion"
    assert all(t.confidence == 1.0 and t.confidence_source == "batched" for t in tags.values())
    assert result.tokens.exact and result.tokens.calls == 2
    assert result.stage_tokens["sections"].prompt == 1001
    assert result.stage_tokens["document"].prompt == 1002
    assert sum((t.tokens for t in result.sections), TokenUsage()) == result.stage_tokens["sections"]
    (doc_call,) = result.document_calls
    assert doc_call.prompt_tokens == 1002 and doc_call.exact and doc_call.options_count == 2

    # more sections than one batch holds: one call per 40
    many = Document.from_sections(
        [{"title": f"Part {i}", "level": 1, "text": f"Body {i}."} for i in range(45)]
    )
    fake = FakeLLM()
    classify(many, backend=llm(fake), max_sections=45, calls="batched_flat")
    assert [len(numbered(b["messages"][1]["content"])) for b in fake.of("sections")] == [40, 5]


def test_batched_options_listed_once():
    fake = FakeLLM()
    classify(financial_doc(), backend=llm(fake), calls="batched_flat")
    (body,) = fake.of("sections")
    system, user = body["messages"][0]["content"], body["messages"][1]["content"]
    for leaf in CATALOGUE.leaves():
        assert len(re.findall(rf"^\s*{re.escape(leaf.id)}: ", system, re.MULTILINE)) == 1
        assert f"{leaf.id}: {leaf.label}: {leaf.description}" in system
    for group in CATALOGUE.children():
        assert re.search(rf"^{re.escape(group.id)}: {re.escape(group.label)}", system, re.M)
    assert "financial.balance_sheet" not in user  # the user turn holds only the views
    assert set(numbered(user)) == {"1", "2"}


def test_batched_invalid_ids_reasked_once_then_unparsed():
    def answer(n, view, attempt):
        if n == "1":
            return "financial.notes_to_financial_statements"
        return "not.an.id" if attempt == 1 else "still.wrong"

    fake = FakeLLM(answer=answer)
    result = classify(financial_doc(), backend=llm(fake), calls="batched_flat")
    first, again = fake.of("sections")
    assert set(numbered(again["messages"][1]["content"])) == {"2"}  # only the failure
    assert again["messages"][0] == first["messages"][0]
    tags = {t.title: t for t in result.sections}
    assert tags["Notes to the financial statements"].confidence == 1.0
    bad = tags["Management discussion"]
    assert bad.label == "general.other" and bad.confidence == 0.0
    assert "unparsed" in result.flags
    assert result.stage_tokens["sections"].calls == 2

    # a re-ask that succeeds clears the failure
    def fixed(n, view, attempt):
        return _by_title(view) if attempt == 2 or n == "1" else "nope"

    fake = FakeLLM(answer=fixed)
    result = classify(financial_doc(), backend=llm(fake), calls="batched_flat")
    assert "unparsed" not in result.flags
    assert {t.label for t in result.sections} == set(list(SECTIONS.values())[:2])

    # two-step: a group that is no group, twice, is unparsed too; its leaf call is never made
    fake = FakeLLM(group=lambda n, view, attempt: "financial" if n == "1" else "nonsense")
    result = classify(financial_doc(), backend=llm(fake))
    assert len(fake.of("groups")) == 2 and len(fake.of("sections")) == 1
    tags = {t.title: t for t in result.sections}
    assert tags["Management discussion"].label == "general.other"
    assert tags["Management discussion"].confidence == 0.0 and "unparsed" in result.flags


def test_batched_doc_stage_offers_only_candidates():
    fake = FakeLLM()
    classify(financial_doc(), backend=llm(fake))
    (body,) = fake.of("document")
    system = body["messages"][0]["content"]
    offered = {
        leaf.id for leaf in DOC_TYPES.leaves() if re.search(rf"^\s*{leaf.id}: ", system, re.M)
    }
    assert offered == {"financial.annual_report", "financial.quarterly_report"}
    assert "Stats:" in body["messages"][1]["content"]  # the document view


def test_non_llm_estimate_uses_batched_pattern():
    titles = list(SECTIONS)[:3] + ["Risk factors", "Properties", "Legal proceedings", "Exhibits"]
    doc = Document.from_sections(
        [{"title": t, "level": 1, "text": f"{t} in a sentence or two."} for t in titles * 2]
    )
    batched = classify(doc, backend=KeywordBackend(), level="headings")
    per_item = classify(doc, backend=KeywordBackend(), level="headings", calls="per_item")
    assert batched.calls == "per_item"  # a keyword backend scores per item, whatever the pattern
    assert not batched.tokens.exact
    sections = batched.stage_tokens["sections"]
    groups = {t.label.split(".")[0] for t in batched.sections}
    assert sections.calls == 1 + len(groups)  # the groups call, then one call per group
    assert per_item.stage_tokens["sections"].calls == 28  # per item: group + leaf each
    assert batched.tokens.prompt < per_item.tokens.prompt / 2  # the options once, not 28 times

    flat = classify(doc, backend=KeywordBackend(), level="headings", calls="batched_flat")
    assert flat.stage_tokens["sections"].calls == 1  # one flat call would carry all 14
    views = {str(i + 1): "" for i in range(14)}
    full_listing = section_prompt(views, CATALOGUE).parts()["options"]
    doc_options = (
        document_prompt("", DOC_TYPES, flat.candidates or None).parts()["options"]
        if flat.document_calls
        else 0
    )
    assert flat.tokens_split["options"] == full_listing + doc_options
    assert all(not c.exact for c in batched.document_calls)


def test_tokens_split_sums_to_estimate():
    for calls in ("batched", "batched_flat", "per_item"):
        result = classify(financial_doc(), backend=KeywordBackend(), calls=calls)
        split = result.tokens_split
        assert set(split) == {"options", "content", "instructions"}
        assert sum(split.values()) == result.tokens.prompt, calls
        assert split["options"] > split["content"]
    prompt = document_prompt("Title: x", DOC_TYPES, ["financial.annual_report"])
    assert sum(prompt.parts().values()) == prompt.estimate()


def test_cascade_with_batched_raises():
    with pytest.raises(ValueError, match="per_item"):
        classify(financial_doc(), backend=llm(FakeLLM()), cascade=True)
    with pytest.raises(ValueError, match="per_item"):
        classify(financial_doc(), backend=KeywordBackend(), calls="batched", cascade=True)
    with pytest.raises(ValueError, match="per_item"):
        classify(financial_doc(), backend=llm(FakeLLM()), calls="batched_flat", cascade=True)
    with pytest.raises(ValueError):
        classify(financial_doc(), backend=KeywordBackend(), calls="sometimes")
    # per-item cascade still works, also for an LLM backend
    fake_calls = []

    def handler(request):
        fake_calls.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "A"}}]})

    result = classify(
        financial_doc(), backend=llm(handler), calls="per_item", cascade=True, level="full"
    )
    assert result.tokens.calls == len(fake_calls)


def test_chat_json_falls_back_without_response_format():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": "response_format is not supported"})
        content = 'Sure: {"doc_type": "legal.contract"} done'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    backend = llm(handler)
    messages = [{"role": "user", "content": "hi"}]
    assert backend.chat_json(messages) == {"doc_type": "legal.contract"}  # parsed tolerantly
    assert backend.chat_json(messages) == {"doc_type": "legal.contract"}
    assert [("response_format" in b) for b in bodies] == [True, False, False]  # remembered
    assert backend.usage().calls == 2 and not backend.usage().exact  # no usage: estimated

    def broken(request):
        return httpx.Response(500, text="down")

    with pytest.raises(BackendError):
        llm(broken).chat_json(messages)


def twelve_sections():
    titles = [
        "Notes to the financial statements",
        "Balance sheet",
        "Income statement",
        "Cash flows",
        "Management discussion",
        "Letter to shareholders",
        "Outlook",
        "Our business",
        "Risk factors",
        "Executive compensation",
        "Properties",
        "Legal proceedings",
    ]
    return Document.from_sections(
        [{"title": t, "level": 1, "text": f"{t} in a sentence or two."} for t in titles]
    )


TWO_STEP_OPTIONS_BOUND = 2200
"""Option tokens of a two-step sections stage for 12 sections in 3 groups (estimated 1,625:
299 for the groups listing, then 585, 247 and 494 for the three groups' leaves), against 4,872
for the full catalogue listing."""


def test_two_step_batched_sections():
    fake = FakeLLM()
    result = classify(twelve_sections(), backend=llm(fake), level="headings")
    assert result.calls == "batched"
    (groups_call,) = fake.of("groups")
    assert len(numbered(groups_call["messages"][1]["content"])) == 12
    system = groups_call["messages"][0]["content"]
    for group in CATALOGUE.children():
        assert re.search(rf"^{re.escape(group.id)}: {re.escape(group.label)}", system, re.M)
    assert not any(leaf.id in system for leaf in CATALOGUE.leaves())  # groups only

    leaf_calls = fake.of("sections")
    assert len(leaf_calls) == 3
    for body in leaf_calls:
        listed = {
            leaf.id
            for leaf in CATALOGUE.leaves()
            if re.search(rf"^\s*{re.escape(leaf.id)}: ", body["messages"][0]["content"], re.M)
        }
        (group,) = {leaf_id.split(".")[0] for leaf_id in listed}  # one group's leaves only
        assert listed == {leaf.id for leaf in CATALOGUE.children(group)}
        assert len(numbered(body["messages"][1]["content"])) == 4
    assert {t.title: t.label for t in result.sections} == {
        title: SECTIONS[title] for title in (t.title for t in result.sections)
    }
    assert result.decided_by == "rules"  # no document call: the options are the sections'
    assert result.stage_tokens["sections"].calls == 4
    two_step = result.tokens_split["options"]

    flat = classify(twelve_sections(), backend=llm(FakeLLM()), level="headings",
                    calls="batched_flat")  # fmt: skip
    full_listing = section_prompt({"1": ""}, CATALOGUE).parts()["options"]
    assert flat.tokens_split["options"] == full_listing
    assert two_step < TWO_STEP_OPTIONS_BOUND < full_listing / 2


def test_batched_rule_gate_needs_two_supporting_sections():
    one = Document.from_sections([{"title": "Opinion", "level": 1, "text": "The court holds."}])
    fake = FakeLLM(doc_type="legal.court_opinion")
    result = classify(one, backend=llm(fake))
    assert result.sections[0].confidence == 1.0
    assert result.decided_by == "model" and "low_support" in result.flags
    siblings = [n.id for n in DOC_TYPES.children("legal")]
    assert result.candidates[0] == "legal.court_opinion"
    assert set(result.candidates) == set(siblings)
    (doc_call,) = fake.of("document")
    assert result.doc_type == "legal.court_opinion"

    two = Document.from_sections(
        [
            {"title": "Syllabus", "level": 1, "text": "The court reviews a statute."},
            {"title": "Opinion", "level": 1, "text": "The court holds."},
        ]
    )
    fake = FakeLLM()
    result = classify(two, backend=llm(fake))
    assert result.decided_by == "rules" and result.doc_type == "legal.court_opinion"
    assert fake.of("document") == [] and "low_support" not in result.flags
