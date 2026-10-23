"""classify(): the sections-first and doc-first flows, levels, cascade, tokens and the log.

Model-free: a scripted backend favours one node per text at every step, and, when metered,
reports per-call usage through ``on_call`` as the OpenAI-compatible backend does.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from docrooter import Classification, SectionTag, classify
from docrooter.backends import KeywordBackend, Scores
from docrooter.backends.base import TokenUsage
from docrooter.document import Document
from docrooter.document.load import load
from docrooter.flow import select_sections
from docrooter.taxonomies import builtin_taxonomy
from docrooter.taxonomy import Taxonomy, TaxonomyNode

DOC_TYPE_LEAVES = {n.id for n in builtin_taxonomy("doc_types").leaves()}

SECTIONS = {
    "Notes to the financial statements": ("financial.notes_to_financial_statements", 0.95),
    "Management discussion": ("narrative.management_discussion", 0.95),
    "Executive compensation": ("disclosure.executive_compensation", 0.95),
    "Acknowledgements": ("general.acknowledgements", 0.95),
    "Performance": ("fund.performance", 0.95),
    "Top holdings": ("fund.holdings_and_allocation", 0.95),
    "Offering terms": ("offering.offering_terms", 0.95),
}


def by_title(table):
    def answer(text):
        for title, target in table.items():
            if re.search(rf"^Title: {re.escape(title)}$", text, re.MULTILINE):
                return target
        return None

    return answer


class Scripted:
    """Favours ``sections(text)`` (a ``(node id, p)``) for section texts and ``document(text)``
    for document views (they carry ``Stats:``) at every step: the node or the group that holds
    it gets ``p`` and the others share the rest. Every call is kept in ``calls``."""

    max_candidates = None
    temperature = 1.0

    def __init__(self, sections=None, document=None, *, metered=False, name="scripted"):
        self.name = name
        self._sections = sections or by_title(SECTIONS)
        self._document = document or (lambda text: None)
        self.metered = metered
        if metered:
            self.on_call = None  # set by classify() around each score call
        self.calls: list[tuple[str, str, list[str]]] = []  # (kind, text, option ids)

    def score(self, texts, candidates, *, instruction=None, multi_label=False):
        ids = [c.node_id for c in candidates]
        rows = []
        for i, text in enumerate(texts):
            kind = "document" if "Stats:" in text else "section"
            self.calls.append((kind, text, ids))
            answer = (self._document if kind == "document" else self._sections)(text)
            rows.append(_row(ids, answer))
            if self.metered and self.on_call is not None:
                self.on_call(
                    {
                        "index": i,
                        "prompt_hash": f"hash-{len(self.calls)}",
                        "prompt_tokens": 100 + len(text.split()),
                        "completion_tokens": 1,
                        "exact": True,
                        "options": ids,
                        "model": "scripted-model",
                        "endpoint_host": "llm.example",
                    }
                )
        return Scores(np.array(rows, dtype=float).reshape(len(texts), len(ids)), "softmax")

    def supports(self, node_id):
        return True

    def tokenizer(self):
        return None

    def of(self, kind):
        return [(text, ids) for k, text, ids in self.calls if k == kind]


def _row(ids, answer):
    n = len(ids)
    if answer is None or n == 1:
        return [1.0 / n] * n
    target, p = answer
    hit = [i for i, c in enumerate(ids) if target == c or target.startswith(c + ".")]
    if not hit:
        return [1.0 / n] * n
    row = [(1.0 - p) / (n - 1)] * n
    row[hit[0]] = p
    return row


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


EXEC_COMP = {
    "title": "Executive compensation",
    "level": 1,
    "text": "Salaries and bonuses of the named executive officers.",
}


# --- sections_first -----------------------------------------------------------------------


def test_sections_first_rules_decide_single_candidate():
    backend = Scripted()
    result = classify(financial_doc(EXEC_COMP), backend=backend)
    assert result.flow == "sections_first"
    assert result.decided_by == "rules"
    assert result.doc_type == "financial.annual_report"
    assert result.candidates == ["financial.annual_report"]  # quarterly is barred by exec comp
    assert result.doc_type_confidence == pytest.approx(0.95 * 0.95)
    assert backend.of("document") == []  # no document-level call
    assert result.stage_tokens["document"] == TokenUsage()
    assert result.levels == {"section": "terms", "document": None}
    assert result.section_taxonomy == "section_catalogue"
    labels = {t.title: t.label for t in result.sections}
    assert labels["Executive compensation"] == "disclosure.executive_compensation"


def test_sections_first_narrows_options_to_candidates():
    backend = Scripted(document=lambda text: ("financial.quarterly_report", 0.8))
    result = classify(financial_doc(), backend=backend)
    (doc_call,) = backend.of("document")
    _, options = doc_call
    assert set(options) == {"financial.annual_report", "financial.quarterly_report"}
    assert set(result.candidates) == set(options)
    assert result.decided_by == "model"
    assert result.doc_type == "financial.quarterly_report"
    assert result.doc_type_confidence == pytest.approx(0.8)
    assert "near_miss" not in result.flags and "rules_unmet" not in result.flags


def test_single_undecided_candidate_is_offered_with_its_near_misses():
    notes = ("financial.notes_to_financial_statements", 0.75)
    weak = {**SECTIONS, "Notes to the financial statements": notes}
    backend = Scripted(by_title(weak))
    result = classify(financial_doc(EXEC_COMP), backend=backend)  # 0.75**2 < decide_conf 0.7
    assert result.decided_by == "model"
    assert "near_miss" in result.flags
    assert result.candidates[0] == "financial.annual_report"
    assert "financial.proxy_statement" in result.candidates  # a near miss: no meeting section
    assert "corporate.investor_presentation" not in result.candidates  # barred by the notes
    (doc_call,) = backend.of("document")
    assert doc_call[1] == result.candidates  # offered in the candidates' order


def test_lone_near_miss_widens_to_every_type():
    doc = Document.from_sections(
        [
            {"title": "Abstract", "level": 1, "text": "We study retrieval for long documents."},
            {"title": "Abstract", "level": 2, "text": "A second summary of the findings."},
        ]
    )
    backend = Scripted(by_title({"Abstract": ("research.abstract", 0.95)}))
    result = classify(doc, backend=backend)
    assert {"near_miss", "low_support"} <= set(result.flags)
    assert result.decided_by == "model"
    assert result.candidates[0] == "research.research_paper"
    (doc_call,) = backend.of("document")
    assert set(doc_call[1]) == DOC_TYPE_LEAVES


def test_low_support_widens_a_custom_doc_type_taxonomy_too():
    doc_types = Taxonomy(
        [
            TaxonomyNode(id="a", label="A"),
            TaxonomyNode(
                id="a.memo",
                label="Memo",
                parent_id="a",
                sections={"requires": [["general.summary"]]},
            ),
            TaxonomyNode(id="b", label="B"),
            TaxonomyNode(
                id="b.letter",
                label="Letter",
                parent_id="b",
                sections={"requires": [["general.acknowledgements"]]},
            ),
        ]
    )
    doc = Document.from_sections(
        [
            {"title": "Summary", "level": 1, "text": "In short."},
            {"title": "Summary", "level": 2, "text": "In shorter."},
        ]
    )
    backend = Scripted(by_title({"Summary": ("general.summary", 0.75)}))
    result = classify(doc, backend=backend, doc_types=doc_types)
    assert result.decided_by == "model" and "low_support" in result.flags
    assert "rules_unmet" not in result.flags
    assert result.candidates == ["a.memo"]
    (doc_call,) = backend.of("document")
    assert set(doc_call[1]) == {"a.memo", "b.letter"}


def test_document_calls_record_every_attempt():
    def document(text):
        return ("financial.quarterly_report", 0.9) if "Key phrases:" in text else None

    backend = Scripted(document=document, metered=True)
    result = classify(financial_doc(), backend=backend, level="headings", cascade=True,
                      threshold=0.6)  # fmt: skip
    first, second = result.document_calls
    assert (first.level, second.level) == ("headings", "terms")
    assert all(c.exact and c.options_count == 2 and not c.truncated for c in (first, second))
    texts = [text for text, _ in backend.of("document")]
    assert [c.prompt_tokens for c in result.document_calls] == [100 + len(t.split()) for t in texts]
    assert sum(c.prompt_tokens for c in result.document_calls) == (
        result.stage_tokens["document"].prompt
    )
    assert result.doc_truncated is False

    rules = classify(financial_doc(EXEC_COMP), backend=Scripted())
    assert rules.document_calls == [] and rules.doc_truncated is False


def test_doc_truncated_is_the_document_view_only():
    long_text = " ".join(["revenue grew strongly this year"] * 4000)
    doc = Document.from_sections(
        [{"title": "Management discussion", "level": 1, "text": long_text}]
    )
    backend = Scripted(document=lambda text: ("financial.annual_report", 0.9))
    sections_cut = classify(doc, backend=backend, section_level="full", doc_level="headings")
    assert "truncated" in sections_cut.flags and not sections_cut.doc_truncated
    doc_cut = classify(doc, backend=backend, level="headings", doc_level="full")
    full_cap = doc_cut.document_calls[-1]
    assert doc_cut.doc_truncated and full_cap.truncated and "truncated" in doc_cut.flags


def test_rules_unmet_falls_back_to_all_types():
    doc = Document.from_sections(
        [{"title": "Acknowledgements", "level": 1, "text": "We thank our reviewers."}]
    )
    backend = Scripted(document=lambda text: ("research.white_paper", 0.6))
    result = classify(doc, backend=backend)
    assert result.decided_by == "fallback"
    assert "rules_unmet" in result.flags
    assert result.candidates == []
    (doc_call,) = backend.of("document")
    assert set(doc_call[1]) == DOC_TYPE_LEAVES
    assert result.doc_type == "research.white_paper"

    # Conflicting excludes: the only candidate (fact sheet) hits an exclude. It is waived, not
    # dropped, and offered with the near misses; nothing raises.
    conflicted = Document.from_sections(
        [
            {"title": "Performance", "level": 1, "text": "Returns over one year."},
            {"title": "Top holdings", "level": 1, "text": "The ten largest positions."},
            {"title": "Offering terms", "level": 1, "text": "Shares offered at par."},
        ]
    )
    backend = Scripted()
    result = classify(conflicted, backend=backend)
    assert {"excludes_waived", "near_miss"} <= set(result.flags)
    assert result.decided_by == "model"
    (doc_call,) = backend.of("document")
    assert "financial.fact_sheet" in doc_call[1] and "financial.prospectus" in doc_call[1]
    assert set(doc_call[1]) == set(result.candidates)
    assert result.doc_type in result.candidates


def test_classify_no_sections_falls_back():
    doc = load("Just some plain text without any heading.\nAnother line of plain text.\n")
    backend = Scripted(document=lambda text: ("government.government_report", 0.7))
    result = classify(doc, backend=backend, level="headings")
    assert result.flags == ["no_sections"]
    assert result.decided_by == "fallback"
    assert result.sections == []
    assert backend.of("section") == []
    (doc_call,) = backend.of("document")
    assert set(doc_call[1]) == DOC_TYPE_LEAVES  # the doc_first document step
    assert result.doc_type == "government.government_report"
    assert result.levels == {"section": "headings", "document": "headings"}


def test_tokens_sum_sections_and_document():
    backend = Scripted(document=lambda text: ("financial.annual_report", 0.9), metered=True)
    result = classify(financial_doc(), backend=backend)
    sections, document = result.stage_tokens["sections"], result.stage_tokens["document"]
    assert result.tokens == sections + document
    assert result.tokens.exact and result.tokens.calls == 5  # 2 sections x (group, leaf) + 1
    assert sections.calls == 4 and document.calls == 1
    assert sum((t.tokens for t in result.sections), TokenUsage()) == sections
    assert all(t.tokens.calls == 2 and t.tokens.exact for t in result.sections)
    expected = sum(100 + len(text.split()) for _, text, _ in backend.calls)
    assert result.tokens.prompt == expected and result.tokens.completion == 5

    # No usage at all (a keyword backend): what the views and options would cost an LLM.
    estimated = classify(financial_doc(), backend=KeywordBackend(), calls="per_item")
    assert not estimated.tokens.exact
    assert estimated.tokens.prompt > 0
    assert estimated.tokens.estimated_calls == estimated.tokens.calls >= 4


def test_cumulative_usage_is_split_over_the_texts_of_a_call():
    class UsageOnly(Scripted):
        def __init__(self):
            super().__init__(document=lambda text: ("financial.annual_report", 0.9))
            self._usage = TokenUsage()

        def score(self, texts, candidates, *, instruction=None, multi_label=False):
            self._usage = self._usage + TokenUsage(10 * len(texts), len(texts), len(texts), True)
            return super().score(texts, candidates, instruction=instruction)

        def usage(self):
            return self._usage

    result = classify(financial_doc(), backend=UsageOnly())
    assert result.tokens == TokenUsage(50, 5, 5, True)
    # a share of a call's usage is an estimate: exact only at call and document level
    assert all(t.tokens.prompt == 20 and not t.tokens.exact for t in result.sections)


def test_cascade_escalates_only_low_confidence():
    mystery = {
        "title": "Mystery",
        "level": 1,
        "text": "The board reviewed operating results and liquidity trends in detail.",
    }

    def sections(text):
        if "Title: Mystery" in text:
            p = 0.95 if "Key phrases:" in text else 0.6
            return ("narrative.management_discussion", p)
        return by_title(SECTIONS)(text)

    def document(text):
        return ("financial.quarterly_report", 0.9) if "Key phrases:" in text else None

    backend = Scripted(sections, document, metered=True)
    doc = Document.from_sections(
        [
            {
                "title": "Notes to the financial statements",
                "level": 1,
                "text": "Note one sets out the accounting policies.",
            },
            mystery,
        ]
    )
    result = classify(doc, backend=backend, level="headings", cascade=True, threshold=0.6)
    tags = {t.title: t for t in result.sections}
    assert tags["Notes to the financial statements"].level_used == "headings"
    assert tags["Notes to the financial statements"].tokens.calls == 2
    assert tags["Mystery"].level_used == "terms"
    assert tags["Mystery"].tokens.calls == 4  # both attempts count
    assert tags["Mystery"].confidence == pytest.approx(0.95 * 0.95)
    with_terms = [text for text, _ in backend.of("section") if "Key phrases:" in text]
    assert with_terms and all("Title: Mystery" in text for text in with_terms)

    # the document call: uniform over two candidates at headings, asked again at terms
    assert result.decided_by == "model"
    assert len(backend.of("document")) == 2
    assert result.levels == {"section": "headings", "document": "terms"}
    assert result.stage_tokens["document"].calls == 2
    assert result.doc_type == "financial.quarterly_report"

    # without cascade nothing is asked twice
    once = Scripted(sections, document)
    flat = classify(doc, backend=once, level="headings")
    assert all(t.level_used == "headings" for t in flat.sections)
    assert len(once.of("document")) == 1


def test_max_sections_order():
    doc = Document.from_sections(
        [
            {"title": "Alpha", "level": 1, "text": "a"},
            {"title": "Alpha one", "level": 2, "text": "a1"},
            {"title": "Beta", "level": 1, "text": "b"},
            {"title": "Beta one", "level": 2, "text": "b1"},
            {"title": "Beta one deep", "level": 3, "text": "b11"},
            {"title": "Gamma", "level": 1, "text": "c"},
        ]
    )
    assert [s.title for s in select_sections(doc, 40)] == [
        "Alpha",
        "Beta",
        "Gamma",
        "Alpha one",
        "Beta one",
    ]
    result = classify(doc, backend=KeywordBackend(), max_sections=4)
    assert [t.title for t in result.sections] == ["Alpha", "Beta", "Gamma", "Alpha one"]
    assert [t.level for t in result.sections] == [1, 1, 1, 2]
    assert all(isinstance(t, SectionTag) for t in result.sections)


def test_doc_first_level_aware():
    doc = Document.from_sections(
        [
            {"title": "Abstract", "level": 1, "text": "We study retrieval. It works well."},
            {"title": "Methods", "level": 1, "text": "We ran experiments. They were careful."},
        ],
        source="paper.md",
    )
    backend = Scripted(lambda text: None, lambda text: ("research.research_paper", 0.9))
    result = classify(
        doc, flow="doc_first", backend=backend, doc_level="headings", section_level="lead"
    )
    assert result.flow == "doc_first" and result.decided_by == "model"
    assert result.doc_type == "research.research_paper"
    assert result.candidates == []
    assert result.levels == {"section": "lead", "document": "headings"}
    assert result.section_taxonomy == "section_types/research_paper"
    (doc_call,) = backend.of("document")
    assert set(doc_call[1]) == DOC_TYPE_LEAVES
    assert "Key phrases:" not in doc_call[0] and "Lead:" not in doc_call[0]
    section_texts = [text for text, _ in backend.of("section")]
    assert len(section_texts) == 2 and all("Lead:" in text for text in section_texts)
    research = builtin_taxonomy("section_types/research_paper")
    assert all(t.label in research and t.level_used == "lead" for t in result.sections)
    assert result.stage_tokens["document"].calls == 1
    assert result.stage_tokens["sections"].calls == 1  # estimated as one batched call
    assert not result.tokens.exact  # no usage reported: estimated


# --- results log, validation, export ------------------------------------------------------


def test_log_writes_one_row_per_decision(tmp_path):
    backend = Scripted(document=lambda text: ("financial.annual_report", 0.9), metered=True)
    path = tmp_path / "log.jsonl"
    classify(financial_doc(), backend=backend, log=path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == len(backend.calls) == 5
    section_rows = [r for r in rows if r["stage"] == "sections"]
    assert len(section_rows) == 4 and all(r["section_id"] for r in section_rows)
    (doc_row,) = [r for r in rows if r["stage"] == "document"]
    assert doc_row["level"] == "terms" and doc_row["prompt_hash"].startswith("hash-")
    assert doc_row["model"] == "scripted-model" and doc_row["endpoint_host"] == "llm.example"
    assert doc_row["label"] == "financial.annual_report"
    assert set(doc_row["options"]) == set(doc_row["probs"])
    assert doc_row["tokens"]["exact"] is True and doc_row["trainable"] is True
    assert doc_row["step"] == "document"
    assert [r["step"] for r in section_rows] == ["group", "group", "leaf", "leaf"]


def test_custom_rules_with_unknown_section_ids_raise():
    doc_types = Taxonomy(
        [
            TaxonomyNode(
                id="memo",
                label="Memo",
                sections={"requires": [["general.summaryy"]]},
            ),
            TaxonomyNode(id="letter", label="Letter"),
        ]
    )
    with pytest.raises(ValueError, match=r"memo: general\.summaryy"):
        classify(financial_doc(), backend=KeywordBackend(), doc_types=doc_types)


@pytest.mark.parametrize(
    "kwargs",
    [{"flow": "vibes"}, {"level": "auto"}, {"doc_level": "everything"}, {"max_sections": -1}],
)
def test_bad_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        classify(financial_doc(), backend=KeywordBackend(), **kwargs)


def test_unowned_section_calls_still_count(monkeypatch):
    from docrooter.classify.classifier import Classifier

    real = Classifier.classify_texts

    def cut(self, texts):  # the backend sees other text than the rendered views
        return real(self, [t + " (cut)" for t in texts])

    monkeypatch.setattr(Classifier, "classify_texts", cut)
    result = classify(financial_doc(EXEC_COMP), backend=Scripted(metered=True))
    assert result.stage_tokens["sections"].calls == 6
    assert result.tokens.calls == 6


def test_a_split_usage_is_shared_as_estimates():
    from docrooter.flow import _split

    for exact in (True, False):
        parts = _split(TokenUsage(30, 3, 1, exact), 3)
        assert [p.prompt for p in parts] == [10, 10, 10] and sum(p.completion for p in parts) == 3
        assert not any(p.exact for p in parts)
    (whole,) = _split(TokenUsage(30, 3, 1, True), 1)
    assert whole == TokenUsage(30, 3, 1, True)  # a call of its own is not a share


def test_estimated_section_shares_are_never_exact():
    for calls in ("batched", "batched_flat", "per_item"):
        result = classify(financial_doc(EXEC_COMP), backend=KeywordBackend(), calls=calls)
        sections = result.stage_tokens["sections"]
        assert result.sections and not any(t.tokens.exact for t in result.sections), calls
        assert sum(t.tokens.prompt for t in result.sections) == sections.prompt
        assert sections.calls == len(result.section_calls)
        assert not any(c.exact for c in result.section_calls)
        assert sum(c.prompt_tokens for c in result.section_calls) == sections.prompt


def test_section_calls_record_every_per_item_call():
    backend = Scripted(document=lambda text: ("financial.annual_report", 0.9), metered=True)
    result = classify(financial_doc(), backend=backend, level="headings")
    assert len(result.section_calls) == result.stage_tokens["sections"].calls == 4
    assert all(c.exact and c.level == "headings" for c in result.section_calls)
    assert [c.prompt_tokens for c in result.section_calls] == [
        100 + len(text.split()) for text, _ in backend.of("section")
    ]


def test_classification_round_trips_as_json():
    result = classify(financial_doc(), backend=KeywordBackend(), level="headings")
    again = Classification.model_validate_json(result.model_dump_json())
    assert again == result
    assert isinstance(again.tokens, TokenUsage)


# --- minors: no call, no level; estimates; option limits ---------------------------------


@pytest.mark.parametrize("flow", ["sections_first", "doc_first"])
def test_an_empty_document_reports_no_document_call(flow):
    backend = Scripted()
    result = classify(Document.from_sections([]), backend=backend, flow=flow)
    assert backend.calls == [] and result.document_calls == []
    assert result.levels["document"] is None  # no document call, so no level
    assert result.doc_type is None and result.decided_by == "fallback"


def test_a_per_item_estimate_counts_the_answer_letter():
    result = classify(financial_doc(), backend=KeywordBackend(), calls="per_item")
    assert result.tokens.completion == result.tokens.calls  # one letter per call


def test_too_many_doc_types_for_per_item_calls_raise_up_front():
    many = Taxonomy(
        [TaxonomyNode(id="t", label="T")]
        + [TaxonomyNode(id=f"t.k{i}", label=f"Kind {i}", parent_id="t") for i in range(30)]
    )

    class Lettered(Scripted):
        max_candidates = 26

    with pytest.raises(ValueError, match="26"):
        classify(financial_doc(), backend=Lettered(), doc_types=many, calls="per_item")


# --- N1: the rules need adequate section evidence; N2: low support widens to every type ----


def with_filler(sections, n_total):
    """``sections`` plus untitled-for-the-rules filler sections, ``n_total`` in all: the
    scripted backend knows no filler title, so their tags are uniform (never confident)."""
    filler = [
        {"title": f"Miscellany {i}", "level": 1, "text": "Assorted remarks of no consequence."}
        for i in range(n_total - len(sections))
    ]
    return Document.from_sections([*sections, *filler], source="report.md")


NOTES = {"title": "Notes to the financial statements", "level": 1, "text": "Accounting policies."}
MDA = {"title": "Management discussion", "level": 1, "text": "Revenue grew across segments."}
THANKS = {"title": "Acknowledgements", "level": 1, "text": "We thank our auditors."}


def test_one_confident_tag_among_ten_is_thin_evidence_and_offers_every_type():
    backend = Scripted(document=lambda text: ("research.white_paper", 0.6))
    result = classify(with_filler([NOTES], 10), backend=backend)
    assert {"rules_unmet", "thin_evidence"} <= set(result.flags)
    assert result.decided_by == "fallback" and result.candidates == []
    (doc_call,) = backend.of("document")
    assert set(doc_call[1]) == DOC_TYPE_LEAVES


def test_three_confident_tags_among_ten_narrow():
    backend = Scripted(document=lambda text: ("financial.quarterly_report", 0.8))
    result = classify(with_filler([NOTES, MDA, THANKS], 10), backend=backend)
    assert "thin_evidence" not in result.flags and "rules_unmet" not in result.flags
    assert result.decided_by == "model"
    (doc_call,) = backend.of("document")
    assert set(doc_call[1]) == {"financial.annual_report", "financial.quarterly_report"}


def test_the_evidence_needed_grows_with_the_sections_tagged():
    # three confident tags among twenty: 20% of 20 is four, so too few
    result = classify(with_filler([NOTES, MDA, THANKS], 20), backend=Scripted())
    assert {"rules_unmet", "thin_evidence"} <= set(result.flags)
    # the keywords move the gate
    eased = classify(with_filler([NOTES, MDA, THANKS], 20), backend=Scripted(),
                     min_evidence_share=0.1)  # fmt: skip
    assert "thin_evidence" not in eased.flags and eased.decided_by == "model"
    strict = classify(with_filler([NOTES, MDA, THANKS], 10), backend=Scripted(), min_evidence=4)
    assert "thin_evidence" in strict.flags


def test_thin_evidence_stops_the_rules_deciding_too():
    one = Document.from_sections([{"title": "Opinion", "level": 1, "text": "The court holds."}])
    backend = Scripted(by_title({"Opinion": ("law.opinion", 0.99)}))
    result = classify(one, backend=backend)  # a lone confident tag: never enough to decide
    assert result.decided_by == "fallback" and "thin_evidence" in result.flags
    assert backend.of("document")


def test_evidence_keywords_are_checked():
    for kwargs in ({"min_evidence": -1}, {"min_evidence_share": 1.5}):
        with pytest.raises(ValueError, match="min_evidence"):
            classify(financial_doc(), backend=Scripted(), **kwargs)


def test_low_support_offers_every_type():
    doc = Document.from_sections(
        [
            {"title": "Opinion", "level": 1, "text": "The court holds that the appeal fails."},
            {"title": "Acknowledgements", "level": 1, "text": "We thank counsel."},
        ]
    )
    weak = {"Opinion": ("law.opinion", 0.75), "Acknowledgements": SECTIONS["Acknowledgements"]}
    backend = Scripted(by_title(weak))  # 0.75**2 < decide_conf: a lone undecided candidate
    result = classify(doc, backend=backend)
    assert result.decided_by == "model"
    assert "low_support" in result.flags and "thin_evidence" not in result.flags
    assert result.candidates == ["legal.court_opinion"]  # the rules' weak suggestion
    (doc_call,) = backend.of("document")
    assert set(doc_call[1]) == DOC_TYPE_LEAVES  # every type, siblings or not


def test_default_level_is_measured_per_backend():
    from docrooter.flow import FALLBACK_LEVEL, default_level

    assert default_level("openai:claude-haiku-5-5") == "headings"
    assert default_level("openai:any-model") == "headings"  # one hosted LLM measured
    assert default_level("embedding:dmis-lab/UltraBERT-Embed-150M") == "lead"
    assert default_level("embedding:some/other-model") == FALLBACK_LEVEL == "terms"
    assert default_level("keyword") == "terms"  # its pick lost 17.7 points to lead
    assert default_level(Scripted(name="openai:m")) == "headings"  # a backend by its name

    result = classify(financial_doc(EXEC_COMP), backend=Scripted(name="openai:m"))
    assert result.levels == {"section": "headings", "document": None}
    result = classify(financial_doc(EXEC_COMP), backend=Scripted())
    assert result.levels == {"section": "terms", "document": None}
    result = classify(financial_doc(EXEC_COMP), backend=Scripted(name="openai:m"), level="lead")
    assert result.levels == {"section": "lead", "document": None}  # an explicit level wins
