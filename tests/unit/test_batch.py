"""Batched LLM calls: one chat call per document stage, the options listed once.

Network-free: the OpenAI-compatible backend talks to an ``httpx.MockTransport`` that answers
from a script and reports usage.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

httpx = pytest.importorskip("httpx")

from docrooter import classify  # noqa: E402
from docrooter.backends import KeywordBackend  # noqa: E402
from docrooter.backends.base import Candidate, Scores  # noqa: E402
from docrooter.backends.openai import OpenAIBackend  # noqa: E402
from docrooter.batch import (  # noqa: E402
    RETRY_NOTE,
    SECTION_INSTRUCTION,
    document_prompt,
    section_prompt,
)
from docrooter.document import Document  # noqa: E402
from docrooter.errors import BackendError  # noqa: E402
from docrooter.taxonomies import builtin_taxonomy  # noqa: E402
from docrooter.views import estimate_tokens  # noqa: E402

TEST_KEY = "test-key-not-real-0000"
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
    "Sustainability": "esg.sustainability",
}


EXEC_COMP_SECTION = {
    "title": "Executive compensation",
    "level": 1,
    "text": "Salaries and bonuses of the named executive officers.",
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
    # a key shaped like a real one: the results log scrubs the key wherever it appears
    backend = OpenAIBackend("gpt-test", base_url="http://llm.test/v1", api_key=TEST_KEY)
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
    assert sum(t.tokens.prompt for t in result.sections) == result.stage_tokens["sections"].prompt
    assert not any(t.tokens.exact for t in result.sections)  # shares of a call: estimates
    (section_call,) = result.section_calls
    assert section_call.prompt_tokens == 1001 and section_call.exact
    assert section_call.options_count == len(CATALOGUE.leaves())
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
    told_why = SECTION_INSTRUCTION + RETRY_NOTE  # the same prompt, plus why it is asked again
    assert again["messages"][0]["content"] == first["messages"][0]["content"].replace(
        SECTION_INSTRUCTION, told_why
    )
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


def test_calls_fall_back_without_temperature():
    # Some hosted models (Claude Haiku 5.5 through Anthropic's OpenAI-compatible endpoint) refuse
    # any temperature; the call is sent again without it, and later calls never send it.
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "temperature" in body:
            message = "`temperature` is deprecated for this model."
            return httpx.Response(400, json={"error": {"message": message}})
        content = '{"doc_type": "legal.contract"}'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    backend = llm(handler)
    messages = [{"role": "user", "content": "hi"}]
    assert backend.chat_json(messages) == {"doc_type": "legal.contract"}
    assert backend.chat_json(messages) == {"doc_type": "legal.contract"}
    assert [("temperature" in b) for b in bodies] == [True, False, False]  # remembered
    assert all("response_format" in b for b in bodies)  # JSON mode is untouched
    assert backend.usage().calls == 2

    def unrelated(request):
        return httpx.Response(400, json={"error": {"message": "prompt is too long"}})

    other = llm(unrelated)
    with pytest.raises(BackendError, match="too long"):
        other.chat_json(messages)
    assert other._send_temperature is True  # an unrelated 400 keeps the temperature


def test_extra_body_is_sent_with_every_call(monkeypatch):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        if "logprobs" in bodies[-1]:
            return httpx.Response(200, json={"choices": [{"message": {"content": "A"}}]})
        content = '{"doc_type": "legal.contract"}'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    monkeypatch.setenv("DOCROOTER_LLM_EXTRA_BODY", '{"thinking": {"type": "disabled"}}')
    backend = llm(handler)
    assert backend.extra_body == {"thinking": {"type": "disabled"}}
    backend.chat_json([{"role": "user", "content": "hi"}])
    backend.score(["a contract"], [Candidate("legal.contract", "Contract")])
    assert [b["thinking"] for b in bodies] == [{"type": "disabled"}] * 2
    assert bodies[0]["model"] == "gpt-test"  # extra fields never replace the request's own

    explicit = OpenAIBackend("m", base_url="http://llm.test/v1", extra_body={"seed": 7})
    assert explicit.extra_body == {"seed": 7}  # the argument wins over the environment
    for bad in ("[1]", "not json"):
        monkeypatch.setenv("DOCROOTER_LLM_EXTRA_BODY", bad)
        with pytest.raises(BackendError, match="DOCROOTER_LLM_EXTRA_BODY"):
            OpenAIBackend("m", base_url="http://llm.test/v1")


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
    one = Document.from_sections(
        [
            {"title": "Opinion", "level": 1, "text": "The court holds."},
            {"title": "Miscellany", "level": 1, "text": "Assorted remarks."},
        ]
    )
    fake = FakeLLM(doc_type="legal.court_opinion")
    result = classify(one, backend=llm(fake))
    assert result.sections[0].confidence == 1.0
    # one supporting section: too little to decide; low support offers every type
    assert result.decided_by == "model" and "low_support" in result.flags
    assert result.candidates == ["legal.court_opinion"]
    (doc_call,) = fake.of("document")
    system = doc_call["messages"][0]["content"]
    assert all(re.search(rf"^\s*{leaf.id}: ", system, re.M) for leaf in DOC_TYPES.leaves())
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


@pytest.mark.parametrize("calls", ["batched", "batched_flat"])
def test_batched_run_writes_the_results_log(tmp_path, calls):
    path = tmp_path / "log.jsonl"
    fake = FakeLLM()
    result = classify(financial_doc(), backend=llm(fake), calls=calls, log=path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == len(fake.bodies) == result.tokens.calls
    steps = [r["step"] for r in rows]
    expected = ["batch_group", "batch_leaf", "batch_leaf"] if calls == "batched" else ["batch"]
    assert steps == [*expected, "document"]
    assert TEST_KEY not in path.read_text()
    for row in rows:
        assert row["confidence_source"] == "batched" and row["probs"] == {}
        assert row["prompt_hash"] and row["tokens"]["exact"] is True
    ids = {t.id: t.label for t in result.sections}
    section_rows = [r for r in rows if r["stage"] == "sections"]
    answered = {
        k: v for r in section_rows if r["step"] != "batch_group" for k, v in r["label"].items()
    }
    assert answered == ids
    if calls == "batched":
        assert set(section_rows[0]["label"].values()) == {"financial", "narrative"}
    assert rows[-1]["label"] == result.doc_type


def test_batched_document_reask_says_why():
    class Wrong(FakeLLM):
        def __call__(self, request):
            self.doc_type = "not.a.type" if not self.of("document") else "financial.annual_report"
            return super().__call__(request)

    fake = Wrong()
    result = classify(financial_doc(), backend=llm(fake))
    first, again = fake.of("document")
    assert "not one of the listed ids" in again["messages"][0]["content"]
    assert result.doc_type == "financial.annual_report"
    assert len(result.document_calls) == 2


def test_json_mode_off_through_classify_only_for_a_response_format_refusal():
    seen = []

    def refusing(request):
        body = json.loads(request.content)
        seen.append("response_format" in body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": "response_format json_object unsupported"})
        return FakeLLM()(request)

    backend = llm(refusing)
    result = classify(financial_doc(), backend=backend)
    assert result.doc_type is not None and seen[0] is True and seen[1:] and not any(seen[1:])

    def too_long(request):
        return httpx.Response(400, json={"error": "context length exceeded"})

    backend = llm(too_long)
    with pytest.raises(BackendError):
        classify(financial_doc(), backend=backend)
    assert backend._json_mode is True  # an unrelated 400 keeps JSON mode


def test_two_step_chunks_and_exact_per_tag_sums():
    many = Document.from_sections(
        [
            {"title": t, "level": 1, "text": f"{t}, part {i}."}
            for i in range(15)
            for t in ("Balance sheet", "Risk factors", "Outlook")
        ]
    )
    fake = FakeLLM()
    result = classify(many, backend=llm(fake), max_sections=45)
    group_calls = [len(numbered(b["messages"][1]["content"])) for b in fake.of("groups")]
    leaf_calls = [len(numbered(b["messages"][1]["content"])) for b in fake.of("sections")]
    assert group_calls == [40, 5]
    assert sorted(leaf_calls) == [15, 15, 15]
    assert result.tokens.exact and result.stage_tokens["sections"].exact
    assert sum(t.tokens.prompt for t in result.sections) == result.stage_tokens["sections"].prompt
    assert not any(t.tokens.exact for t in result.sections)
    assert result.stage_tokens["sections"].calls == len(result.section_calls) == 5
    assert all(c.exact for c in result.section_calls)


def test_a_section_share_of_an_unmetered_call_is_never_exact():
    def no_usage(request):
        data = FakeLLM()(request).json()
        del data["usage"]
        return httpx.Response(200, json=data)

    for calls in ("batched", "batched_flat"):
        result = classify(financial_doc(EXEC_COMP_SECTION), backend=llm(no_usage), calls=calls)
        assert not result.tokens.exact
        assert not any(t.tokens.exact for t in result.sections), calls
        assert not any(c.exact for c in result.section_calls)


def test_estimates_say_their_pattern_and_count_calls_once():
    doc = financial_doc()
    for calls, pattern in ((None, "batched"), ("batched_flat", "batched_flat"),
                           ("per_item", "per_item")):  # fmt: skip
        result = classify(doc, backend=KeywordBackend(), calls=calls)
        assert result.tokens_pattern == pattern
        assert result.tokens.estimated_calls == result.tokens.calls
    exact = classify(doc, backend=llm(FakeLLM()))
    assert exact.tokens_pattern == "batched"
    # cascade with a backend that cannot batch: per-item calls, so a per-item estimate
    cascaded = classify(doc, backend=KeywordBackend(), cascade=True)
    assert cascaded.tokens_pattern == "per_item"
    assert cascaded.tokens.estimated_calls == cascaded.tokens.calls


class FirstOption:
    """A non-LLM backend (no usage) that always picks the first option, with certainty."""

    name = "first"
    max_candidates = None
    temperature = 1.0

    def score(self, texts, candidates, *, instruction=None, multi_label=False):
        values = np.zeros((len(texts), len(candidates)))
        values[:, 0] = 1.0
        return Scores(values, "softmax")

    def supports(self, node_id):
        return True

    def tokenizer(self):
        return None


def first_listed(system: str, indented: bool) -> str:
    """The first option id a batched system turn lists (the first leaf when ``indented``)."""
    lines = system.split("Options:\n", 1)[1].splitlines()
    pick = next((ln for ln in lines if ln.startswith("  ")), lines[0]) if indented else lines[0]
    return pick.strip().split(": ", 1)[0]


def sent_tokens(bodies) -> int:
    """What the requests carried, estimated the way the backend estimates a call."""
    return sum(estimate_tokens(m["content"]) for b in bodies for m in b["messages"])


def test_estimates_match_what_the_requests_carry():
    doc = financial_doc(EXEC_COMP_SECTION)

    # per item: the fake answers "A", the first option, as FirstOption picks it
    bodies: list[dict] = []

    def per_item(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "A"}}]})

    real = classify(doc, backend=llm(per_item), calls="per_item", level="headings")
    estimated = classify(doc, backend=FirstOption(), calls="per_item", level="headings")
    assert real.tokens.calls == estimated.tokens.calls == len(bodies)
    assert estimated.tokens.prompt == sent_tokens(bodies)
    assert estimated.tokens_split["instructions"] > 0  # system prompt and question, every call
    assert sum(estimated.tokens_split.values()) == estimated.tokens.prompt

    # batched: the fake answers the first group, then the first leaf, then the first type
    bodies = []

    def batched(request):
        body = json.loads(request.content)
        bodies.append(body)
        system, user = body["messages"][0]["content"], body["messages"][1]["content"]
        if '"doc_type"' in system:
            reply = {"doc_type": first_listed(system, indented=True)}
        else:
            pick = first_listed(system, indented="group id" not in system)
            reply = dict.fromkeys(numbered(user), pick)
        content = json.dumps(reply)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    for calls in ("batched", "batched_flat"):
        bodies.clear()
        real = classify(doc, backend=llm(batched), calls=calls, level="headings")
        estimated = classify(doc, backend=FirstOption(), calls=calls, level="headings")
        assert [t.label for t in real.sections] == [t.label for t in estimated.sections]
        assert estimated.tokens.calls == len(bodies), calls
        assert estimated.tokens.prompt == sent_tokens(bodies), calls
        assert sum(estimated.tokens_split.values()) == estimated.tokens.prompt


def test_batched_llm_calls_are_packed_by_size():
    long = " ".join(["figure"] * 1400)
    titles = ["Balance sheet", "Risk factors", "Outlook", "Properties"]
    doc = Document.from_sections(
        [{"title": titles[i % 4], "level": 1, "text": f"{long} {i}"} for i in range(40)]
    )
    fake = FakeLLM()
    result = classify(doc, backend=llm(fake), level="full")
    sections = [b for b in fake.bodies if fake.kind(b) != "document"]
    assert len(fake.of("groups")) > 1  # 40 views of ~1,850 tokens cannot share one call
    assert all(sent_tokens([b]) <= 13600 for b in sections)  # the estimated budget
    assert len(result.section_calls) == len(sections)
    assert {t.label for t in result.sections} == {SECTIONS[t] for t in titles}


def test_a_leaf_id_in_the_groups_step_counts_as_its_group():
    fake = FakeLLM(group=lambda n, view, attempt: _by_title(view))  # leaf ids, not group ids
    result = classify(financial_doc(), backend=llm(fake))
    assert len(fake.of("groups")) == 1  # no re-ask
    assert "unparsed" not in result.flags
    assert {t.label for t in result.sections} == set(list(SECTIONS.values())[:2])


# --- per item: answers without logprobs, and answers that name no option ------------------

LABELS = {
    "Opinion": ("Court opinions and legislation", "Opinion of the court"),
    "Syllabus": ("Court opinions and legislation", "Syllabus"),
}


def letter_llm(pick, *, logprobs=False):
    """A per-item endpoint: the letter of the option ``pick(text, options)`` names (prose for
    ``None``), with logprobs (that letter only) when ``logprobs``."""
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        user = body["messages"][1]["content"]
        text, listed = user.split("Text: ", 1)[1].split("\nOptions:\n", 1)
        options = re.findall(r"^([A-Z])\. (.*)$", listed, re.M)
        index = pick(text, [option for _, option in options])
        content = "I cannot tell." if index is None else options[index][0]
        choice = {"message": {"role": "assistant", "content": content}}
        if logprobs and index is not None:
            top = [{"token": content, "logprob": -0.01}]
            choice["logprobs"] = {
                "content": [{"token": content, "logprob": -0.01, "top_logprobs": top}]
            }
        usage = {"prompt_tokens": 50, "completion_tokens": 1}
        return httpx.Response(200, json={"choices": [choice], "usage": usage})

    backend = llm(handler)
    backend.bodies = bodies
    return backend


def by_label(doc_type="Court opinion", unparsed=()):
    """Picks the option that starts with the section's labels (its group's, then its own);
    for the document, the option naming ``doc_type``; ``None`` for the steps in
    ``unparsed`` (``"group"``, ``"leaf"`` or ``"document"``)."""

    def pick(text, options):
        if "Stats:" in text:
            if "document" in unparsed:
                return None
            return next(i for i, o in enumerate(options) if f"> {doc_type}:" in o)
        title = re.search(r"^Title: (.*)$", text, re.M).group(1)
        group, leaf = LABELS[title]
        for step, label in (("group", group), ("leaf", leaf)):
            hits = [i for i, o in enumerate(options) if o.startswith(f"{label}:") or o == label]
            if hits:
                return None if step in unparsed else hits[0]
        return None

    return pick


def court(*titles):
    return Document.from_sections(
        [{"title": t, "level": 1, "text": f"The court and the {t.lower()}."} for t in titles]
    )


def test_per_item_answers_with_logprobs_are_probabilities():
    result = classify(court("Syllabus", "Opinion"), backend=letter_llm(by_label(), logprobs=True),
                      calls="per_item")  # fmt: skip
    for tag in result.sections:
        assert tag.confidence_source == "probability" and tag.confidence == pytest.approx(1.0)
    assert result.decided_by == "rules" and result.doc_type == "legal.court_opinion"


def test_per_item_answers_without_logprobs_are_not_certainties():
    backend = letter_llm(by_label())
    result = classify(court("Opinion"), backend=backend, calls="per_item")
    (tag,) = result.sections
    assert tag.label == "law.opinion"
    assert tag.confidence is None and tag.confidence_source == "text"
    # one section with no probability: too little evidence for the rules at all
    assert result.decided_by == "fallback" and "thin_evidence" in result.flags
    assert result.document_calls and result.doc_type == "legal.court_opinion"
    assert result.doc_type_confidence is None and result.doc_type_probs == {}

    two = classify(court("Syllabus", "Opinion"), backend=letter_llm(by_label()), calls="per_item")
    assert two.decided_by == "rules" and two.doc_type == "legal.court_opinion"
    assert two.doc_type_confidence is None and not two.document_calls


def test_a_per_item_reply_naming_no_option_is_flagged_never_defaulted():
    result = classify(court("Opinion"), backend=letter_llm(by_label(unparsed=("document",))),
                      calls="per_item")  # fmt: skip
    assert result.document_calls  # it was asked
    assert result.doc_type is None and "unparsed" in result.flags
    assert result.doc_type_confidence == 0.0

    result = classify(court("Opinion"), backend=letter_llm(by_label(unparsed=("group",))),
                      calls="per_item")  # fmt: skip
    (tag,) = result.sections
    assert tag.label == "general.other" and tag.confidence == 0.0 and "unparsed" in result.flags

    result = classify(court("Opinion"), backend=letter_llm(by_label(unparsed=("leaf",))),
                      calls="per_item")  # fmt: skip
    (tag,) = result.sections
    assert tag.label == "law.other" and tag.confidence == 0.0 and "unparsed" in result.flags


def test_per_item_log_rows_say_how_the_answer_was_read(tmp_path):
    path = tmp_path / "log.jsonl"
    classify(court("Opinion"), backend=letter_llm(by_label(unparsed=("document",))),
             calls="per_item", log=path)  # fmt: skip
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    sections = [r for r in rows if r["stage"] == "sections"]
    assert sections and all(r["confidence_source"] == "text" and r["probs"] == {} for r in sections)
    document = rows[-1]
    assert document["stage"] == "document" and document["label"] is None


def test_batched_low_support_keeps_the_near_misses():
    doc = Document.from_sections(
        [
            {"title": "Sustainability", "level": 1, "text": "Our emissions fell this year."},
            {"title": "Our business", "level": 1, "text": "We make and sell widgets."},
        ]
    )
    fake = FakeLLM(doc_type="financial.annual_report")
    result = classify(doc, backend=llm(fake))
    assert result.decided_by == "model"
    assert {"low_support", "near_miss"} <= set(result.flags)
    assert result.candidates[0] == "corporate.sustainability_report"
    assert "financial.annual_report" in result.candidates  # a near miss (section 6.3)
    assert "corporate.investor_presentation" in result.candidates
    assert result.doc_type == "financial.annual_report"


def test_real_document_calls_fit_a_small_cap():
    from docrooter.batch import call_budget

    long = " ".join(["figure"] * 800)
    doc = Document.from_sections(
        [
            {"title": t, "level": 1, "text": long}
            for t in ("Balance sheet", "Outlook", "Risk factors")
        ]
    )
    fake = FakeLLM(doc_type="financial.annual_report")
    result = classify(doc, backend=llm(fake), flow="doc_first", level="lead", max_call_tokens=1500)
    (body,) = fake.of("document")
    assert sent_tokens([body]) <= call_budget(1500) and result.doc_truncated

    per_item = letter_llm(lambda text, options: 0)
    classify(doc, backend=per_item, flow="doc_first", level="lead", calls="per_item",
             max_call_tokens=1500)  # fmt: skip
    documents = [b for b in per_item.bodies if "Stats:" in b["messages"][1]["content"]]
    assert documents and all(sent_tokens([b]) <= call_budget(1500) for b in documents)


# --- N1/N2 with batched answers: valid answers count as confident tags ---------------------


def filler_doc(titles, n_total):
    sections = [{"title": t, "level": 1, "text": f"The {t.lower()} section."} for t in titles]
    sections += [
        {"title": f"Miscellany {i}", "level": 1, "text": "Assorted remarks."}
        for i in range(n_total - len(titles))
    ]
    return Document.from_sections(sections, source="report.md")


def known_only(n, view, attempt):
    """Valid ids for the known titles; an id that is not an option for every other section."""
    leaf = _by_title(view)
    return leaf if leaf != "general.body" else "not.an.option"


def known_group(n, view, attempt):
    return known_only(n, view, attempt).split(".")[0]


def test_batched_one_valid_tag_among_ten_is_thin_evidence():
    fake = FakeLLM(doc_type="financial.annual_report", answer=known_only, group=known_group)
    result = classify(filler_doc(["Notes to the financial statements"], 10), backend=llm(fake))
    assert {"rules_unmet", "thin_evidence"} <= set(result.flags)
    assert result.decided_by == "fallback" and result.candidates == []
    (body,) = fake.of("document")
    system = body["messages"][0]["content"]
    assert all(re.search(rf"^\s*{leaf.id}: ", system, re.M) for leaf in DOC_TYPES.leaves())


def test_batched_three_valid_tags_among_ten_narrow():
    fake = FakeLLM(answer=known_only, group=known_group)
    titles = ["Notes to the financial statements", "Management discussion", "Outlook"]
    result = classify(filler_doc(titles, 10), backend=llm(fake))
    assert "thin_evidence" not in result.flags and result.decided_by == "model"
    assert set(result.candidates) == {"financial.annual_report", "financial.quarterly_report"}


def test_per_item_text_answers_count_as_confident_tags():
    one = classify(court("Opinion"), backend=letter_llm(by_label()), calls="per_item")
    assert {"rules_unmet", "thin_evidence"} <= set(one.flags)
    two = classify(court("Syllabus", "Opinion"), backend=letter_llm(by_label()), calls="per_item")
    assert "thin_evidence" not in two.flags and two.decided_by == "rules"
