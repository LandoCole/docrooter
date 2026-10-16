import pytest
from conftest import CountingBackend, NeverSupports

import docrooter
from docrooter import Beam, Classifier, Document, DocumentView, Flat, Greedy, Taxonomy, TaxonomyNode
from docrooter.backends import KeywordBackend
from docrooter.classify import EngineSettings, run_flat
from docrooter.classify.documents import mix_distributions
from docrooter.classify.results import DocTypeResult


def doc_type_taxonomy() -> Taxonomy:
    return Taxonomy(
        [
            TaxonomyNode(id="financial", label="Financial", description="Company finances."),
            TaxonomyNode(
                id="financial.annual_report",
                label="Annual report",
                parent_id="financial",
                description="revenue earnings shareholders fiscal year audited balance sheet",
            ),
            TaxonomyNode(
                id="financial.prospectus",
                label="Prospectus",
                parent_id="financial",
                description="offering securities underwriters proceeds",
            ),
            TaxonomyNode(id="legal", label="Legal", description="Law."),
            TaxonomyNode(
                id="legal.contract",
                label="Contract",
                parent_id="legal",
                description="agreement parties termination indemnification governing law",
            ),
            TaxonomyNode(id="research", label="Research", description="Science."),
            TaxonomyNode(
                id="research.paper",
                label="Research paper",
                parent_id="research",
                description="abstract method experiments results references",
            ),
        ]
    )


def contract_doc() -> Document:
    return Document.from_sections(
        [
            {"title": "Parties", "level": 1, "text": "This agreement is made between the parties."},
            {
                "title": "Termination",
                "level": 1,
                "text": "Either party may terminate the agreement.",
            },
            {"title": "Indemnification", "level": 1, "text": "Parties indemnification terms."},
            {"title": "Governing law", "level": 1, "text": "This agreement follows governing law."},
        ],
        source="contract.md",
    )


def paper_doc() -> Document:
    return Document.from_sections(
        [
            {"title": "Abstract", "level": 1, "text": "We present experiments and results."},
            {"title": "Method", "level": 1, "text": "Our method is simple."},
            {"title": "Experiments", "level": 1, "text": "Experiments show results."},
            {"title": "References", "level": 1, "text": "References list."},
        ],
        source="paper.md",
    )


def annual_doc() -> Document:
    return Document.from_sections(
        [
            {"title": "Letter to shareholders", "level": 1, "text": "Revenue and earnings grew."},
            {"title": "Fiscal year review", "level": 1, "text": "The fiscal year revenue rose."},
            {
                "title": "Balance sheet",
                "level": 1,
                "text": "Audited balance sheet for shareholders.",
            },
        ],
        source="annual.md",
    )


EXPECTED = [
    (contract_doc, "legal.contract"),
    (paper_doc, "research.paper"),
    (annual_doc, "financial.annual_report"),
]
MODES = ["view", "sections", "hybrid"]


def check_leaf_probs(result: DocTypeResult) -> None:
    assert 0 < len(result.leaf_probs) <= 10
    assert sum(result.leaf_probs.values()) <= 1 + 1e-6
    assert result.best.node_ids[-1] in result.leaf_probs


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(("make_doc", "expected"), EXPECTED)
def test_classify_document_picks_right_leaf(mode, make_doc, expected):
    doc = make_doc()
    res = docrooter.classify_document(
        doc, doc_type_taxonomy(), backend=KeywordBackend(), mode=mode, threshold=0.0
    )
    assert isinstance(res, DocTypeResult)
    assert res.best.node_ids[-1] == expected
    assert res.best.node_ids == [expected.split(".")[0], expected]
    assert res.best.stopped_reason is None
    assert res.mode == mode and res.source == doc.source
    check_leaf_probs(res)
    assert res.timings["classify"] >= 0
    assert res.view_text == DocumentView().render(doc).text


@pytest.mark.parametrize("mode", ["sections", "hybrid"])
def test_aggregated_best_score_is_aggregated_prob(mode):
    res = docrooter.classify_document(
        contract_doc(), doc_type_taxonomy(), backend=KeywordBackend(), mode=mode
    )
    assert res.best.score == pytest.approx(res.leaf_probs["legal.contract"])
    assert res.best.score == max(res.leaf_probs.values())
    assert res.best.steps[0].chosen == ["legal.contract"]


def test_view_mode_leaf_probs_are_the_flat_distribution():
    res = docrooter.classify_document(
        paper_doc(), doc_type_taxonomy(), backend=KeywordBackend(), threshold=0.0
    )
    assert set(res.leaf_probs) == {n.id for n in doc_type_taxonomy().leaves()}
    assert sum(res.leaf_probs.values()) == pytest.approx(1.0)
    assert res.best.score == pytest.approx(res.leaf_probs["research.paper"])
    # The public step keeps the usual top-5 contract even though Flat scored every leaf.
    assert len(res.best.steps[0].probs) <= 5


def test_low_confidence_view_still_reports_distribution():
    res = docrooter.classify_document(
        paper_doc(), doc_type_taxonomy(), backend=KeywordBackend(), threshold=0.99
    )
    assert res.best.stopped_reason == "low_confidence" and res.best.node_ids == []
    assert sum(res.leaf_probs.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("strategy", [Beam(3), Greedy()])
def test_tree_strategy_leaf_probs_from_paths(strategy):
    res = docrooter.classify_document(
        contract_doc(),
        doc_type_taxonomy(),
        backend=KeywordBackend(),
        strategy=strategy,
        threshold=0.0,
    )
    assert res.best.node_ids[-1] == "legal.contract"
    check_leaf_probs(res)
    assert res.leaf_probs["legal.contract"] == pytest.approx(res.best.score)


@pytest.mark.parametrize("mode", MODES)
def test_classify_documents_one_backend_call_for_many_docs(mode):
    backend = CountingBackend(KeywordBackend())
    clf = Classifier(doc_type_taxonomy(), backend, strategy=Flat(), threshold=0.0)
    docs = [contract_doc(), paper_doc(), annual_doc()]
    results = clf.classify_documents(docs, mode=mode)
    assert backend.calls == 1
    assert [r.best.node_ids[-1] for r in results] == [
        "legal.contract",
        "research.paper",
        "financial.annual_report",
    ]


def test_hybrid_with_tree_strategy_runs_view_and_sections_separately():
    backend = CountingBackend(KeywordBackend())
    clf = Classifier(doc_type_taxonomy(), backend, strategy=Greedy(), threshold=0.0)
    results = clf.classify_documents([contract_doc(), contract_doc()], mode="hybrid")
    # Greedy view: one root call (Legal's only child needs none); sections: one Flat call.
    assert backend.calls == 2
    assert all(r.best.node_ids[-1] == "legal.contract" for r in results)


def test_sections_mode_weights_sections_by_tokens():
    long_contract = "agreement parties termination indemnification " * 30
    doc = Document.from_sections(
        [
            {"title": "Terms", "level": 1, "text": long_contract},
            {"title": "Abstract", "level": 1, "text": "abstract method experiments results"},
        ]
    )
    res = docrooter.classify_document(
        doc, doc_type_taxonomy(), backend=KeywordBackend(), mode="sections"
    )
    assert res.best.node_ids[-1] == "legal.contract"
    flipped = Document.from_sections(
        [
            {"title": "Terms", "level": 1, "text": "agreement parties termination"},
            {"title": "Abstract", "level": 1, "text": "abstract method experiments " * 30},
        ]
    )
    res = docrooter.classify_document(
        flipped, doc_type_taxonomy(), backend=KeywordBackend(), mode="sections"
    )
    assert res.best.node_ids[-1] == "research.paper"


@pytest.mark.parametrize("mode", MODES)
def test_empty_document_is_not_sent_to_a_backend(mode):
    clf = Classifier(doc_type_taxonomy(), NeverSupports(), strategy=Flat())
    (res,) = clf.classify_documents([Document(source="blank.pdf")], mode=mode)
    assert res.best.stopped_reason == "empty_text" and res.best.node_ids == []
    assert res.leaf_probs == {} and res.view_text == "" and res.mode == mode
    assert res.timings == {"classify": 0.0}


def test_empty_document_in_a_batch():
    backend = CountingBackend(KeywordBackend())
    clf = Classifier(doc_type_taxonomy(), backend, strategy=Flat(), threshold=0.0)
    blank = Document.from_sections([{"title": " ", "level": 1, "text": "  "}], source="b.md")
    results = clf.classify_documents([blank, paper_doc()])
    assert backend.calls == 1
    assert results[0].best.stopped_reason == "empty_text"
    assert results[1].best.node_ids[-1] == "research.paper"


def test_title_only_document_is_classified():
    doc = Document(source="x.pdf", title="Research paper abstract method experiments")
    res = docrooter.classify_document(doc, doc_type_taxonomy(), backend=KeywordBackend())
    assert res.best.node_ids[-1] == "research.paper"
    # No sections to aggregate: sections mode has nothing to classify.
    res = docrooter.classify_document(
        doc, doc_type_taxonomy(), backend=KeywordBackend(), mode="sections"
    )
    assert res.best.stopped_reason == "empty_text"


def test_custom_view_and_instruction_are_used():
    backend = CountingBackend(KeywordBackend())
    seen = {}

    def score(texts, candidates, *, instruction=None, multi_label=False):
        seen["texts"], seen["instruction"] = list(texts), instruction
        return KeywordBackend().score(texts, candidates)

    backend.score = score
    view = DocumentView(fields=("title", "toc"))
    res = docrooter.classify_document(paper_doc(), doc_type_taxonomy(), backend=backend, view=view)
    assert seen["instruction"] == "What type of document is this?"
    assert seen["texts"] == [view.render(paper_doc()).text] == [res.view_text]


def test_invalid_mode():
    clf = Classifier(doc_type_taxonomy(), KeywordBackend(), strategy=Flat())
    with pytest.raises(ValueError, match="mode"):
        clf.classify_documents([paper_doc()], mode="bogus")


def test_mix_distributions():
    mixed = mix_distributions({"a": 1.0}, {"a": 0.5, "b": 0.5})
    assert mixed == pytest.approx({"a": 0.8, "b": 0.2})
    assert mix_distributions({}, {"b": 0.4}) == pytest.approx({"b": 1.0})
    assert mix_distributions({}, {}) == {}


def test_run_flat_full_probs_setting():
    tax = Taxonomy([TaxonomyNode(id=f"n{i}", label=f"word{i}") for i in range(8)])
    texts = {"t": "word3"}
    default = run_flat(texts, tax, KeywordBackend(), Flat(), EngineSettings(threshold=0.0))
    assert len(default["t"][0].steps[0].probs) == 5
    full = run_flat(
        texts, tax, KeywordBackend(), Flat(), EngineSettings(threshold=0.0, full_probs=True)
    )
    best, alternatives, _ = full["t"]
    assert len(best.steps[0].probs) == 8 and sum(best.steps[0].probs.values()) == pytest.approx(1)
    assert best.node_ids == ["n3"] and len(alternatives) == 5


def test_public_exports():
    for name in ("DocumentView", "classify_document", "DocTypeResult"):
        assert name in docrooter.__all__ and hasattr(docrooter, name)
