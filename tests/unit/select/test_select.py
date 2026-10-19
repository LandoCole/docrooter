"""``select()``: which elements to read for a need, at a completeness (spec §6, §7).

Everything is fake and torch-free: documents are built by hand, the tagger tags "risk" texts
with the risk-factors topic, the embedder maps "risk" texts (and needs) to one direction and
everything else to another, and the combiners are hand-set logistic weights.
"""

import time

import numpy as np
import pytest

from docrooter import Document, RouterConfig
from docrooter.classify.normalize import NormalizationResult, SectionNorm
from docrooter.classify.results import DocTypeResult, PathResult
from docrooter.elements.models import Element, TableData, TagResult
from docrooter.errors import DocrooterError
from docrooter.rooter.analysis import Analysis
from docrooter.select import Field, calibration_of, field_fingerprint
from docrooter.select.crc import DialTable
from docrooter.select.relevance import FEATURE_NAMES, Combiner
from docrooter.select.selection import (
    SELECTOR_FILE,
    DocContext,
    FitProvenance,
    Selection,
    Selector,
    need_features,
    prepare,
    readout,
    select,
    select_scored,
)
from docrooter.views import estimate_tokens

RISK = Field(
    id="annual_report.risk_factors",
    label="Risk factors",
    description="The risks the company faces.",
    topics=["annual_report.risk_factors"],
    functions=["risk_disclosure"],
    query="What are the main risks the company faces?",
)
LEGAL = Field(
    id="annual_report.legal_proceedings",
    label="Legal proceedings",
    description="Lawsuits.",
    topics=["annual_report.legal_proceedings"],
)
FUND_FEES = Field(
    id="fund_document.fees",
    label="Fees",
    description="Fees and expenses.",
    topics=["fund_document.fees_and_expenses"],
)
ANNUAL_REPORT = "financial.annual_report"
PROSPECTUS = "financial.prospectus"  # a fund document


# --- fakes -------------------------------------------------------------------------------------


def _combiner(feature_names, weights: dict[str, float], intercept: float = -5.0) -> Combiner:
    names = list(feature_names)
    return Combiner.from_dict(
        {
            "format": "docrooter-relevance-combiner",
            "version": 1,
            "feature_names": names,
            "l2": 1.0,
            "max_iter": 100,
            "mean": [0.0] * len(names),
            "scale": [1.0] * len(names),
            "coef": [weights.get(name, 0.0) for name in names],
            "intercept": intercept,
            "n_train": 10,
            "n_docs": 2,
            "converged": True,
        }
    )


# The full field combiner listens to p_topic only, the fallbacks and the query combiner to sim
# only: p = sigmoid(10 * x - 5), about 0.993 for x = 1 and 0.0067 for x = 0.
FIELD = _combiner(FEATURE_NAMES, {"p_topic": 10.0})
FALLBACK = _combiner(("sim", "section_hit"), {"sim": 10.0})
QUERY = _combiner(FEATURE_NAMES, {"sim": 10.0})
HIGH = 1.0 / (1.0 + np.exp(-5.0))
LOW = 1.0 / (1.0 + np.exp(5.0))


def _dial(n: int = 100, family: str = "annual_report", combiner: Combiner = FIELD) -> DialTable:
    """λ̂(0.9) = 0.5 for n = 100; capped at 0.9 for n < 9 (1/(n+1) > 0.1)."""
    return DialTable(
        family=family,
        n=n,
        n_docs=n,
        n_excluded=0,
        grid=(0.5, 0.9),
        curve_lambdas=[0.0, 0.5, 1.0],
        curve_risk=[0.0, 0.0, 1.0],
        combiner_fingerprint=combiner.fingerprint(),
    )


TAGGER_FP = "fp-of-the-fitted-bundle"  # the tagger bundle fingerprint the fake fit recorded
EMBEDDER_SPEC = "embedding:fake-model"
# What the hand-set selector was "fitted" on: the accurate profile's reading (heron layout,
# section labels on), the fake tagger and the fake embedder. analysis_of() reads like this.
PROVENANCE = FitProvenance(
    tagger=TAGGER_FP,
    embedder=EMBEDDER_SPEC,
    profile="accurate",
    layout="heron",
    sections="embedding:dmis-lab/UltraBERT-Embed-150M",
)


def _selector(
    dials: dict[str, DialTable] | None = None,
    *,
    query: Combiner = QUERY,
    field_fallback: Combiner = FALLBACK,
    provenance: FitProvenance | None = PROVENANCE,
) -> Selector:
    return Selector(
        field=FIELD,
        field_fallback=field_fallback,
        query=query,
        query_fallback=FALLBACK,
        dials=dials or {"annual_report": _dial()},
        calibrated_fields=calibration_of([RISK, LEGAL, FUND_FEES]),
        provenance=provenance,
    )


def _topic(leaf: str, fingerprint: str | None = TAGGER_FP) -> TagResult:
    return TagResult(
        labels=[leaf], probs={leaf: 1.0}, confidence=1.0, source="fake", fingerprint=fingerprint
    )


class FakeTagger:
    """Tags "risk" texts with the family's risk leaf (annual reports) and anything else with
    the family's catch-all; records the families it was asked for. Its tags carry
    ``fingerprint``, by default the one the hand-set selector was fitted with."""

    def __init__(self, path=None, fingerprint: str | None = TAGGER_FP):
        self.calls: list[tuple[int, str]] = []
        self.path = path
        self.fingerprint = fingerprint

    def tag(self, elements, family):
        self.calls.append((len(elements), family))
        out = []
        for element in elements:
            risky = "risk" in element.text.lower() and family == "annual_report"
            leaf = f"{family}.risk_factors" if risky else f"{family}.other"
            out.append({"topic": _topic(leaf, self.fingerprint)})
        return out


class FakeEmbedder:
    """Two directions: texts that mention risk, and everything else (no recipe: bare prompts).
    Its ``name`` is the spec the hand-set selector was fitted with."""

    name = EMBEDDER_SPEC

    def __init__(self):
        self.calls = 0

    def encode(self, texts, prompt=""):
        self.calls += 1
        return np.array([[1.0, 0.0] if "risk" in t.lower() else [0.0, 1.0] for t in texts])


# --- documents ---------------------------------------------------------------------------------


def words(n: int) -> str:
    return " ".join(["word"] * n)


def make_doc(items, sections=None) -> Document:
    """Sections s1 and s2 (and s3 when given), and ``items`` as ``(kind, text, section_id,
    **fields)`` in reading order, ids e0, e1, …"""
    doc = Document.from_sections(
        sections or [{"title": "Factors", "level": 1}, {"title": "Statements", "level": 1}]
    )
    elements = []
    for index, (kind, text, section_id, *rest) in enumerate(items):
        fields = rest[0] if rest else {}
        if kind in ("heading", "title") and "level" not in fields:
            fields = {"level": 1, **fields}
        elements.append(
            Element(
                id=f"e{index}",
                kind=kind,
                text=text,
                boxes=[],
                order=index,
                section_id=section_id,
                **fields,
            )
        )
    doc.elements = elements
    for element in elements:
        if element.section_id in doc.sections:
            doc.sections[element.section_id].element_ids.append(element.id)
    return doc


RISK_TABLE = TableData(rows=[["Risk exposure", "Amount"], ["Rates", "10"]], header_rows=1)
PLAIN_TABLE = TableData(rows=[["Plain", "numbers"], ["1", "2"]], header_rows=1)


def expansion_doc() -> Document:
    return make_doc(
        [
            ("heading", "Factors", "s1"),  # e0
            ("paragraph", "This section opens the discussion.", "s1"),  # e1 first paragraph
            ("paragraph", "Some filler about operations.", "s1"),  # e2
            ("paragraph", "More filler about our offices.", "s1"),  # e3 previous of e4
            ("paragraph", "Interest rate risk could harm our results.", "s1"),  # e4 RELEVANT
            ("paragraph", "Filler after that one.", "s1"),  # e5
            ("heading", "Statements", "s2"),  # e6
            ("paragraph", "The statements follow.", "s2"),  # e7 first paragraph
            ("paragraph", "Filler before the table.", "s2"),  # e8 previous of e9
            (
                "table",
                "Risk exposure Amount Rates 10",
                "s2",
                {"table": RISK_TABLE, "caption_id": "e10"},
            ),  # e9 RELEVANT
            ("caption", "Table 1. Exposures", "s2", {"caption_for": "e9"}),  # e10
            ("paragraph", "Closing filler.", "s2"),  # e11
            ("page_footer", "Page 1", "s2"),  # e12
            ("heading", "Notes", "s3"),  # e13
            ("caption", "Table 2. Risk summary", "s3", {"caption_for": "e15"}),  # e14 RELEVANT
            ("table", "Plain numbers 1 2", "s3", {"table": PLAIN_TABLE, "caption_id": "e14"}),
        ],
        sections=[
            {"title": "Factors", "level": 1},
            {"title": "Statements", "level": 1},
            {"title": "Notes", "level": 1},
        ],
    )


NO_LABELS = NormalizationResult(doc_type=None, taxonomy="section_types/annual_report")
"""Section labelling ran and labelled nothing."""


def analysis_of(
    doc: Document,
    doc_type: str | None = ANNUAL_REPORT,
    sections: NormalizationResult | None = NO_LABELS,
    config: RouterConfig | None = None,
) -> Analysis:
    """An analysis read as the hand-set selector was fitted (the accurate profile, section
    labels on) unless ``sections`` or ``config`` say otherwise."""
    result = None
    if doc_type is not None:
        result = DocTypeResult(
            source="test", best=PathResult(node_ids=[doc_type.split(".")[0], doc_type])
        )
    return Analysis(
        document=doc,
        doc_type=result,
        sections=sections,
        config=config or RouterConfig(profile="accurate"),
    )


def selectable_ids(doc: Document) -> list[str]:
    return [e.id for e in doc.elements if e.kind not in ("page_footer", "figure")]


def tokens_of(doc: Document, ids) -> int:
    by_id = {e.id: e for e in doc.elements}
    return sum(estimate_tokens(by_id[i].text) for i in ids)


# --- tests -------------------------------------------------------------------------------------


def test_completeness_one_returns_everything():
    doc = expansion_doc()
    analysis = analysis_of(doc)
    for need in (RISK, [RISK, LEGAL], "anything at all"):
        result = select(analysis, need, completeness=1.0)  # no selector, tagger or embedder
        assert isinstance(result, Selection)
        assert result.element_ids == selectable_ids(doc)
        assert result.guarantee is True
        assert result.threshold is None
        assert result.expected_recall == 1.0
        assert result.capped is False
        assert result.relevant_expected is None
        assert result.token_share == 1.0
        assert result.tokens == tokens_of(doc, selectable_ids(doc))
    assert select(analysis, RISK, completeness=1.0).need == RISK.id
    assert select(analysis, "the query", completeness=1.0).need == "the query"
    # Even an empty document is "everything".
    empty = select(analysis_of(make_doc([])), RISK, completeness=1.0)
    assert empty.element_ids == [] and empty.tokens == 0 and empty.guarantee is True


def test_threshold_from_dial_and_guarantee_flag(tmp_path):
    doc = expansion_doc()
    analysis = analysis_of(doc)
    tagger, embedder = FakeTagger(), FakeEmbedder()
    result = select(
        analysis, RISK, completeness=0.9, selector=_selector(), tagger=tagger, embedder=embedder
    )
    assert result.threshold == 0.5  # λ̂(0.9) of the dial
    assert result.guarantee is True
    assert result.expected_recall == 0.9  # the CRC target
    assert result.capped is False and result.doc_type_mismatch is False
    assert result.relevant_expected == pytest.approx(3 * HIGH + 9 * LOW)
    assert {"e4", "e9", "e14"} <= set(result.element_ids)  # every relevant element
    assert not {"e2", "e5", "e11"} & set(result.element_ids)  # filler outside the expansion
    assert tagger.calls == [(12, "annual_report")]  # untagged: tagged in the field's family
    assert analysis.document.elements[4].tags is None  # the analysis is left as it was

    # A capped dial returns the whole document, with the trivial guarantee.
    capped = select(
        analysis,
        RISK,
        completeness=0.9,
        selector=_selector({"annual_report": _dial(n=5)}),
        tagger=FakeTagger(),
        embedder=FakeEmbedder(),
    )
    assert capped.capped is True and capped.guarantee is True
    assert capped.threshold is None and capped.expected_recall == 1.0
    assert capped.element_ids == selectable_ids(doc)

    # A list of fields of one family is one need, an estimate (the dial was calibrated on
    # single fields); mixed families are refused.
    both = select(
        analysis,
        [RISK, LEGAL],
        selector=_selector(),
        tagger=FakeTagger(),
        embedder=FakeEmbedder(),
    )
    assert both.guarantee is False and both.need == f"{RISK.id},{LEGAL.id}"
    assert both.multi_field is True and both.threshold is not None
    with pytest.raises(ValueError, match="famil"):
        select(analysis, [RISK, FUND_FEES], selector=_selector(), embedder=FakeEmbedder())

    # No selector below 1.0 is an error that says how to make one.
    with pytest.raises(ValueError, match="docrooter fit-selector"):
        select(analysis, RISK, completeness=0.9, tagger=FakeTagger(), embedder=FakeEmbedder())

    # The selector is found in the tagger's bundle directory, and loads from a file too.
    _selector().save(tmp_path)
    assert (tmp_path / SELECTOR_FILE).is_file()
    found = select(analysis, RISK, tagger=FakeTagger(path=tmp_path), embedder=FakeEmbedder())
    assert found.model_dump() == result.model_dump()
    loaded = Selector.load(tmp_path / SELECTOR_FILE)
    assert loaded.to_dict() == _selector().to_dict()
    by_path = select(
        analysis, RISK, selector=tmp_path, tagger=FakeTagger(), embedder=FakeEmbedder()
    )
    assert by_path.model_dump() == result.model_dump()

    # A dial calibrated on another combiner is refused at load.
    data = _selector().to_dict()
    data["dials"]["annual_report"]["combiner_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="another combiner"):
        Selector.from_dict(data)


def test_expansion_adds_previous_same_section_heading_first_paragraph_and_whole_tables():
    doc = expansion_doc()
    result = select(
        analysis_of(doc),
        RISK,
        selector=_selector(),
        tagger=FakeTagger(),
        embedder=FakeEmbedder(),
    )
    # core: e4, e9 (a table) and e14 (a caption)
    # e4  -> e3 (previous, same section), e0 (heading), e1 (first paragraph)
    # e9  -> e8 (previous), e6 (heading), e7 (first paragraph), e10 (its caption)
    # e14 -> e13 (previous and heading), e15 (its table, kept whole with its caption)
    expected = ["e0", "e1", "e3", "e4", "e6", "e7", "e8", "e9", "e10", "e13", "e14", "e15"]
    assert result.element_ids == expected  # in reading order
    assert result.tokens == tokens_of(doc, expected)  # headings and captions count
    assert result.token_share == pytest.approx(
        tokens_of(doc, expected) / tokens_of(doc, selectable_ids(doc))
    )

    # The previous element is only added within the same section.
    doc2 = make_doc(
        [
            ("heading", "Factors", "s1"),  # e0
            ("paragraph", "Filler in the first section.", "s1"),  # e1
            ("heading", "Statements", "s2"),  # e2
            ("paragraph", "Risk is everywhere.", "s2"),  # e3 RELEVANT, first of s2
        ]
    )
    result2 = select(
        analysis_of(doc2), RISK, selector=_selector(), tagger=FakeTagger(), embedder=FakeEmbedder()
    )
    assert result2.element_ids == ["e2", "e3"]


def test_budget_mode_no_guarantee():
    doc = expansion_doc()
    analysis = analysis_of(doc)
    group_e4 = ["e0", "e1", "e3", "e4"]
    group_e9 = ["e6", "e7", "e8", "e9", "e10"]
    group_e14 = ["e13", "e14", "e15"]
    # Room for the first group (by relevance, then reading order) and the third, not the second.
    budget = tokens_of(doc, group_e4) + tokens_of(doc, group_e14)
    assert tokens_of(doc, group_e9) > tokens_of(doc, group_e14)
    result = select(
        analysis,
        RISK,
        completeness=0.9,
        budget=budget,
        selector=_selector(),
        tagger=FakeTagger(),
        embedder=FakeEmbedder(),
    )
    assert result.guarantee is False and result.over_budget is False
    assert result.tokens <= budget
    assert result.element_ids == sorted(group_e4 + group_e14, key=lambda i: int(i[1:]))
    assert result.threshold is None
    content_p = {"e1": LOW, "e3": LOW, "e4": HIGH, "e14": HIGH, "e15": LOW}
    assert result.expected_recall == pytest.approx(sum(content_p.values()) / (3 * HIGH + 9 * LOW))
    assert result.expected_recall < 1.0

    # A larger budget takes everything relevant, and still makes no guarantee.
    big = select(
        analysis,
        RISK,
        budget=10_000,
        selector=_selector(),
        tagger=FakeTagger(),
        embedder=FakeEmbedder(),
    )
    assert big.guarantee is False and big.tokens <= 10_000
    assert {"e4", "e9", "e14"} <= set(big.element_ids)

    with pytest.raises(ValueError, match="budget"):
        select(analysis, RISK, budget=0, selector=_selector())
    with pytest.raises(ValueError, match="budget"):  # 1.0 always means the whole document
        select(analysis, RISK, completeness=1.0, budget=100)


def test_query_need_never_guaranteed():
    doc = expansion_doc()
    analysis = analysis_of(doc)
    result = select(
        analysis,
        "What are the main risks?",
        completeness=0.9,
        selector=_selector(),
        tagger=FakeTagger(),
        embedder=FakeEmbedder(),
    )
    assert result.guarantee is False
    assert result.need == "What are the main risks?"
    assert result.doc_type_mismatch is False
    # Model estimate: the shortest prefix by p whose share of sum(p) reaches 0.9. By
    # similarity, e15 is relevant too: its table text carries its caption, "Risk summary".
    total = 4 * HIGH + 8 * LOW
    assert result.relevant_expected == pytest.approx(total)
    assert result.expected_recall == pytest.approx(4 * HIGH / total)
    assert 0.9 <= result.expected_recall < 1.0
    assert result.threshold == pytest.approx(HIGH)  # the p of the last element taken
    assert {"e4", "e9", "e14", "e15"} <= set(result.element_ids)
    assert not {"e2", "e5", "e11"} & set(result.element_ids)

    # Ties are kept together: with every p equal, the whole set is the shortest prefix.
    flat = _selector(query=_combiner(FEATURE_NAMES, {}, intercept=0.0))  # p = 0.5 everywhere
    tied = select(analysis, "anything", selector=flat, tagger=FakeTagger(), embedder=FakeEmbedder())
    assert tied.threshold == pytest.approx(0.5) and tied.expected_recall == pytest.approx(1.0)
    assert tied.element_ids == selectable_ids(doc)

    # A blank query is refused.
    with pytest.raises(ValueError, match="query"):
        select(analysis, "   ", selector=_selector())


def test_doc_type_mismatch_retags_and_drops_guarantee():
    doc = expansion_doc()
    for element in doc.elements:  # tagged as a fund document, as process() would have
        if element.kind in ("paragraph", "table", "caption"):
            element.tags = {"topic": _topic("fund_document.other")}
    analysis = analysis_of(doc, doc_type=PROSPECTUS)
    tagger = FakeTagger()
    result = select(analysis, RISK, selector=_selector(), tagger=tagger, embedder=FakeEmbedder())
    assert tagger.calls == [(12, "annual_report")]  # re-tagged within the field's family
    assert result.doc_type_mismatch is True
    assert result.guarantee is False
    assert result.capped is False
    assert {"e4", "e9", "e14"} <= set(result.element_ids)  # the re-tagged topics count
    assert analysis.document.elements[4].tags["topic"].labels == ["fund_document.other"]

    # Without a tagger there is no re-tag: the similarity fallback, flagged, no guarantee.
    bare = select(analysis, RISK, selector=_selector(), embedder=FakeEmbedder())
    assert bare.doc_type_mismatch is True and bare.guarantee is False
    assert {"e4", "e9", "e14"} <= set(bare.element_ids)

    # A field of the document's family is no mismatch.
    fund = select(analysis, FUND_FEES, selector=_selector(), embedder=FakeEmbedder())
    assert fund.doc_type_mismatch is False


def test_no_relevant_elements_needs_defined_readout():
    doc = make_doc(
        [
            ("heading", "Fund overview", "s1"),
            ("paragraph", "The fund invests in bonds.", "s1"),
            ("paragraph", "It is managed by a team.", "s1"),
            ("heading", "Statements", "s2"),
            ("paragraph", "Net assets grew.", "s2"),
        ]
    )
    analysis = analysis_of(doc)
    kwargs = {"tagger": FakeTagger(), "embedder": FakeEmbedder()}
    # Guaranteed path: no element reaches λ̂, so nothing is selected. The guarantee is what
    # is promised (expected_recall = c); relevant_expected shows the small mass.
    result = select(analysis, RISK, selector=_selector(), **kwargs)
    assert result.relevant_expected == pytest.approx(3 * LOW)
    assert result.expected_recall == 0.9
    assert result.element_ids == [] and result.tokens == 0 and result.chunks == []
    assert result.guarantee is True  # the calibrated threshold was applied
    # A capped dial returns the whole document even then (R3 over R6), recall trivially 1.
    capped = select(analysis, RISK, selector=_selector({"annual_report": _dial(n=5)}), **kwargs)
    assert capped.capped is True and capped.guarantee is True
    assert capped.expected_recall == 1.0 and capped.element_ids == selectable_ids(doc)
    # Estimate path (a query) and budget mode: recall undefined, never the whole document.
    query = select(analysis, "risk", selector=_selector(), **kwargs)
    assert query.expected_recall is None and query.element_ids == []
    budget = select(analysis, RISK, budget=1_000, selector=_selector(), **kwargs)
    assert budget.expected_recall is None and budget.element_ids == []
    assert budget.over_budget is False


def _timed_select(n_sections: int, per_section: int = 20) -> tuple[float, Selection, int]:
    """Seconds for a guaranteed, a budgeted and a query select() on a document of
    ``n_sections`` labelled sections (the best of three runs), and the guaranteed result."""
    sections = [{"title": f"Part {i}", "level": 1} for i in range(n_sections)]
    items = []
    for s in range(n_sections):
        items.append(("heading", f"Part {s}", f"s{s + 1}"))
        for j in range(per_section):
            text = f"Paragraph {j} of part {s} with some ordinary words in it"
            if j % 7 == 3:
                text += " and a risk"
            items.append(("paragraph", text, f"s{s + 1}"))
    doc = make_doc(items, sections=sections)
    tags = {
        True: {"topic": _topic("annual_report.risk_factors")},
        False: {"topic": _topic("annual_report.other")},
    }
    for element in doc.elements:
        if element.kind == "paragraph":
            element.tags = tags["risk" in element.text]
    labels = NormalizationResult(
        doc_type=ANNUAL_REPORT,
        taxonomy="section_types/annual_report",
        sections={
            f"s{s + 1}": SectionNorm(canonical="risk_factors", label="Risk", title=f"Part {s}")
            for s in range(0, n_sections, 2)
        },
    )
    analysis = analysis_of(doc, sections=labels)
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        result = select(analysis, RISK, selector=_selector(), embedder=FakeEmbedder())
        budgeted = select(
            analysis, RISK, budget=20_000, selector=_selector(), embedder=FakeEmbedder()
        )
        query = select(analysis, "risk", selector=_selector(), embedder=FakeEmbedder())
        best = min(best, time.perf_counter() - start)
    assert result.guarantee is True and budgeted.tokens <= 20_000 and query.element_ids
    return best, result, len(doc.elements)


def test_select_linear_time():
    small, _, n_small = _timed_select(75)  # 1,575 elements
    large, result, n_large = _timed_select(300)  # 6,300 elements: four times as many
    assert n_large > 5_000 and n_large == 4 * n_small
    assert large < 2.0, f"select() took {large:.2f}s on {n_large} elements"
    # Linear: four times the elements cost about four times the time. A quadratic step would
    # cost sixteen; the bound leaves room for timer noise on the small document.
    assert large < 8.0 * small + 0.05, f"{small:.3f}s for {n_small}, {large:.3f}s for {n_large}"


def test_select_scored_is_the_tail_of_select():
    """The benchmarks and the playground score with prepare() + need_features() +
    select_scored(); that must be select() itself."""
    analysis = analysis_of(expansion_doc())
    for need in (RISK, [RISK, LEGAL], "What are the main risks?"):
        for c in (0.5, 0.9, 0.99):
            direct = select(
                analysis,
                need,
                completeness=c,
                selector=_selector(),
                tagger=FakeTagger(),
                embedder=FakeEmbedder(),
            )
            ctx = prepare(
                analysis, tagger=FakeTagger(), encoder=FakeEmbedder(), family="annual_report"
            )
            _, X = need_features(ctx, need)
            tail = select_scored(analysis, ctx, X, need, completeness=c, selector=_selector())
            assert tail.model_dump() == direct.model_dump()


def test_no_tagger_fallback_no_guarantee():
    doc = expansion_doc()
    analysis = analysis_of(doc)  # untagged, and no tagger given
    result = select(analysis, RISK, selector=_selector(), embedder=FakeEmbedder())
    assert result.guarantee is False
    assert result.expected_recall is not None and result.expected_recall < 1.0
    # The full combiner would see p_topic = 0 everywhere and select nothing; the fallback ranks
    # by similarity and finds every relevant element.
    assert {"e4", "e9", "e14"} <= set(result.element_ids)

    # Some elements tagged, not all: still the fallback.
    doc.elements[4].tags = {"topic": _topic("annual_report.risk_factors")}
    partial = select(analysis, RISK, selector=_selector(), embedder=FakeEmbedder())
    assert partial.guarantee is False

    # No embedder either: structure only (section labels), and no guarantee.
    labels = NormalizationResult(
        doc_type=ANNUAL_REPORT,
        taxonomy="section_types/annual_report",
        sections={"s1": SectionNorm(canonical="risk_factors", label="Risk", title="Factors")},
    )
    fallback = _selector(  # records no embedder, so none is loaded
        field_fallback=_combiner(("sim", "section_hit"), {"section_hit": 10.0}), provenance=None
    )
    structural = select(analysis_of(doc, sections=labels), RISK, selector=fallback)
    assert structural.guarantee is False
    assert {"e1", "e2", "e3", "e4", "e5"} <= set(structural.element_ids)  # section s1
    assert "e9" not in structural.element_ids


def test_tiny_budget():
    doc = expansion_doc()
    analysis = analysis_of(doc)
    kwargs = {"selector": _selector(), "tagger": FakeTagger(), "embedder": FakeEmbedder()}
    result = select(analysis, RISK, budget=1, **kwargs)
    # The most relevant element (the first of the tied best, in reading order) alone.
    assert result.element_ids == ["e4"]
    assert result.over_budget is True and result.guarantee is False
    assert result.tokens == tokens_of(doc, ["e4"]) > 1
    assert result.expected_recall == pytest.approx(HIGH / (3 * HIGH + 9 * LOW))

    # A budget that fits the best element but nothing with its expansion: the element alone.
    small = make_doc(
        [("heading", "Factors", "s1"), ("paragraph", "Risk could harm our results.", "s1")]
    )
    fits = select(analysis_of(small), RISK, budget=tokens_of(small, ["e1"]), **kwargs)
    assert fits.element_ids == ["e1"] and fits.over_budget is False

    # The best element alone exceeds the budget while low-relevance filler would fit: the best
    # element alone, flagged, never the filler (review probe I2).
    probe = make_doc(
        [
            ("heading", "Factors", "s1"),
            ("paragraph", "risk " + words(199), "s1"),
            ("heading", "Statements", "s2"),
            ("paragraph", "x", "s2"),
        ]
    )
    over = select(analysis_of(probe), RISK, budget=10, **kwargs)
    assert over.element_ids == ["e1"] and over.over_budget is True
    assert over.expected_recall == pytest.approx(HIGH / (HIGH + LOW))


def test_chunks_from_selected_elements():
    doc = expansion_doc()
    result = select(
        analysis_of(doc), RISK, selector=_selector(), tagger=FakeTagger(), embedder=FakeEmbedder()
    )
    chunked = [i for c in result.chunks for i in c.element_ids]
    assert sorted(chunked, key=lambda i: int(i[1:])) == result.element_ids  # each exactly once
    assert [c.id for c in result.chunks] == [f"c{i}" for i in range(len(result.chunks))]
    # Contiguous spans are merged, gaps are not bridged: e1 and e3 are apart (e2 is not read).
    by_element = {i: c for c in result.chunks for i in c.element_ids}
    assert by_element["e0"] is by_element["e1"]
    assert by_element["e3"] is by_element["e4"]
    assert by_element["e1"] is not by_element["e3"]
    assert "filler about operations" not in " ".join(c.text for c in result.chunks)
    # A table chunk holds its caption.
    assert by_element["e9"] is by_element["e10"]
    assert by_element["e14"] is by_element["e15"]
    assert all(c.heading_path for c in result.chunks)


def test_selector_validation():
    with pytest.raises(ValueError, match="fallback"):
        Selector(field=FIELD, field_fallback=FIELD, query=QUERY, query_fallback=FALLBACK, dials={})
    with pytest.raises(ValueError, match="all the features"):
        Selector(
            field=FALLBACK, field_fallback=FALLBACK, query=QUERY, query_fallback=FALLBACK, dials={}
        )
    with pytest.raises(ValueError, match="family"):
        Selector(
            field=FIELD,
            field_fallback=FALLBACK,
            query=QUERY,
            query_fallback=FALLBACK,
            dials={"contract": _dial()},
        )
    with pytest.raises(ValueError, match="completeness"):
        select(analysis_of(expansion_doc()), RISK, completeness=1.5, selector=_selector())


def test_expansion_keeps_tables_whole_with_captions():
    # Review probe M1: the caption arrives as the previous element of the relevant paragraph;
    # its table must come with it.
    doc = make_doc(
        [
            ("heading", "Factors", "s1"),  # e0
            ("paragraph", "Opening words.", "s1"),  # e1
            ("table", "Plain numbers 1 2", "s1", {"table": PLAIN_TABLE, "caption_id": "e3"}),
            ("caption", "Table 1. Numbers", "s1", {"caption_for": "e2"}),  # e3
            ("paragraph", "Risk could harm our results.", "s1"),  # e4 RELEVANT
        ]
    )
    kwargs = {"selector": _selector(), "tagger": FakeTagger(), "embedder": FakeEmbedder()}
    result = select(analysis_of(doc), RISK, **kwargs)
    assert result.element_ids == ["e0", "e1", "e2", "e3", "e4"]
    # A budget unit is whole too: the paragraph's unit brings the caption and so the table.
    budgeted = select(analysis_of(doc), RISK, budget=10_000, **kwargs)
    assert {"e2", "e3"} <= set(budgeted.element_ids)


def test_need_features_shared_by_select():
    doc = expansion_doc()
    analysis = analysis_of(doc, doc_type=PROSPECTUS)  # the family override wins over this
    embedder, tagger = FakeEmbedder(), FakeTagger()
    ctx = prepare(analysis, tagger=tagger, encoder=embedder, family="annual_report")
    assert isinstance(ctx, DocContext)
    assert tagger.calls == [(12, "annual_report")]
    content = [e.id for e in doc.elements if e.kind in ("paragraph", "table", "caption")]
    ids, X = need_features(ctx, RISK)
    assert ids == content and X.shape == (12, len(FEATURE_NAMES))
    p_topic = X[:, FEATURE_NAMES.index("p_topic")]
    assert [i for i, v in zip(ids, p_topic, strict=True) if v == 1.0] == ["e4", "e9", "e14"]
    # One context serves many needs: the document side is embedded once.
    ids2, X2 = need_features(ctx, "What are the main risks?")
    need_features(ctx, [RISK, LEGAL])
    assert ids2 == ids and embedder.calls == 1 + 3  # documents once, then one call per need
    sim = X2[:, FEATURE_NAMES.index("sim")]
    assert [i for i, v in zip(ids, sim, strict=True) if v > 0.99] == ["e4", "e9", "e14", "e15"]
    # select() scores exactly these features.
    result = select(
        analysis_of(doc), RISK, selector=_selector(), tagger=FakeTagger(), embedder=FakeEmbedder()
    )
    _, X3 = need_features(
        prepare(analysis_of(doc), tagger=FakeTagger(), encoder=FakeEmbedder()), RISK
    )
    assert result.relevant_expected == pytest.approx(float(FIELD.predict_proba(X3).sum()))


def test_selector_is_read_only_and_rechecked():
    selector = _selector()
    with pytest.raises(AttributeError):
        selector.field = FIELD
    selector.dials["annual_report"].combiner_fingerprint = "0" * 64  # a stale dial
    with pytest.raises(ValueError, match="another combiner"):
        select(
            analysis_of(expansion_doc()),
            RISK,
            selector=selector,
            tagger=FakeTagger(),
            embedder=FakeEmbedder(),
        )


def test_spec_tagger_closed_on_bad_result(monkeypatch):
    from docrooter.tagger import element_tags

    class Broken:
        closed = False

        def tag(self, elements, family):
            return []  # the wrong number of results

        def close(self):
            Broken.closed = True

    monkeypatch.setattr(element_tags, "resolve_tagger", lambda spec: Broken())
    with pytest.raises(DocrooterError, match="results"):
        select(
            analysis_of(expansion_doc()),
            RISK,
            selector=_selector(),
            tagger="some-bundle",
            embedder=FakeEmbedder(),
        )
    assert Broken.closed is True


# --- the guarantee only covers the fields the dial was calibrated on ---------------------------


def _select_risk(need, selector=None, completeness=0.9):
    return select(
        analysis_of(expansion_doc()),
        need,
        completeness=completeness,
        selector=selector or _selector(),
        tagger=FakeTagger(),
        embedder=FakeEmbedder(),
    )


def test_a_calibrated_field_is_guaranteed():
    result = _select_risk(RISK)
    assert result.guarantee is True and result.uncalibrated_field is False
    assert "field not calibrated" not in readout(result)


def test_a_user_field_with_a_calibrated_id_but_other_topics_is_not_guaranteed():
    edited = RISK.model_copy(update={"topics": ["annual_report.legal_proceedings"]})
    assert edited.id == RISK.id and field_fingerprint(edited) != field_fingerprint(RISK)
    result = _select_risk(edited)
    assert result.guarantee is False and result.uncalibrated_field is True
    assert result.capped is False and result.threshold is None  # no dial threshold
    assert "field not calibrated (estimate)" in readout(result)
    # the functions are part of the fingerprint too
    assert not _select_risk(RISK.model_copy(update={"functions": []})).guarantee


def test_an_unknown_field_is_not_guaranteed_and_a_list_needs_every_field_calibrated():
    mine = Field(
        id="annual_report.mine", label="Mine", description="x", topics=["annual_report.properties"]
    )
    result = _select_risk(mine)
    assert result.guarantee is False and result.uncalibrated_field is True
    both = _select_risk([RISK, mine])
    assert both.guarantee is False and both.uncalibrated_field is True
    # several calibrated fields are still an estimate: the dial was calibrated per field
    pair = _select_risk([RISK, LEGAL])
    assert pair.guarantee is False and pair.multi_field is True


def test_completeness_one_is_guaranteed_for_any_need_whatever_was_calibrated():
    mine = Field(
        id="annual_report.mine", label="Mine", description="x", topics=["annual_report.properties"]
    )
    uncalibrated = Selector(
        field=FIELD,
        field_fallback=FALLBACK,
        query=QUERY,
        query_fallback=FALLBACK,
        dials={"annual_report": _dial()},
    )
    for need in (mine, RISK, "a question"):
        result = _select_risk(need, uncalibrated, completeness=1.0)
        assert result.guarantee is True and result.uncalibrated_field is False


def test_a_selector_without_recorded_fields_guarantees_no_field():
    """An old selector.json has no calibrated fields: every field is then an estimate."""
    data = _selector().to_dict()
    del data["calibrated_fields"]
    old = Selector.from_dict(data)
    assert old.calibrated_fields == {}
    result = _select_risk(RISK, old)
    assert result.guarantee is False and result.uncalibrated_field is True


def test_calibrated_fields_round_trip_and_are_validated(tmp_path):
    selector = _selector()
    assert selector.calibrated_fields["annual_report"] == {
        RISK.id: field_fingerprint(RISK),
        LEGAL.id: field_fingerprint(LEGAL),
    }
    path = selector.save(tmp_path)
    loaded = Selector.load(path)
    assert loaded.calibrated_fields == selector.calibrated_fields
    assert _select_risk(RISK, loaded).guarantee is True
    data = selector.to_dict()
    data["calibrated_fields"] = {"annual_report": ["not", "a", "mapping"]}
    with pytest.raises(ValueError, match="calibrated_fields"):
        Selector.from_dict(data)
    with pytest.raises(ValueError, match="calibrated_fields"):
        Selector(
            field=FIELD,
            field_fallback=FALLBACK,
            query=QUERY,
            query_fallback=FALLBACK,
            dials={},
            calibrated_fields={"annual_report": {RISK.id: 7}},
        )


# --- a list of fields is an estimate (G2) ---------------------------------------------------


def test_a_list_of_fields_is_an_estimate():
    pair = _select_risk([RISK, LEGAL])
    assert pair.guarantee is False and pair.multi_field is True
    assert pair.uncalibrated_field is False and pair.fit_mismatch == []
    assert pair.capped is False
    assert "several fields (estimate)" in readout(pair)
    assert "guaranteed" not in readout(pair)
    # one field is not "several", and c = 1.0 is the whole document for any need
    single = _select_risk(RISK)
    assert single.multi_field is False and "several fields" not in readout(single)
    whole = _select_risk([RISK, LEGAL], completeness=1.0)
    assert whole.guarantee is True and whole.multi_field is False


# --- the guarantee needs the fit's tagger, embedder and reading (G1) -------------------------

READING_DIFFERS = "reading differs from calibration (estimate)"


def _tagged_doc(fingerprint: str | None) -> Document:
    """The expansion document, tagged as a tagger of ``fingerprint`` would tag it."""
    doc = expansion_doc()
    for element in doc.elements:
        if element.kind in ("paragraph", "table", "caption"):
            risky = "risk" in element.text.lower()
            leaf = "annual_report.risk_factors" if risky else "annual_report.other"
            element.tags = {"topic": _topic(leaf, fingerprint)}
    return doc


def test_the_fit_reading_tagger_and_embedder_earn_the_guarantee():
    result = _select_risk(RISK)
    assert result.guarantee is True and result.fit_mismatch == []
    assert READING_DIFFERS not in readout(result)
    # tags the analysis already carries, from the fitted bundle, count as the fit's
    tagged = select(
        analysis_of(_tagged_doc(TAGGER_FP)), RISK, selector=_selector(), embedder=FakeEmbedder()
    )
    assert tagged.guarantee is True and tagged.fit_mismatch == []
    assert tagged.model_dump() == result.model_dump()


def test_tags_from_another_tagger_drop_the_guarantee():
    for fingerprint in ("fp-of-another-bundle", None):  # None: tags of unknown provenance
        result = select(
            analysis_of(_tagged_doc(fingerprint)),
            RISK,
            selector=_selector(),
            embedder=FakeEmbedder(),
        )
        assert result.guarantee is False and result.fit_mismatch == ["tagger"]
        assert result.threshold is not None and result.threshold != 0.5  # the estimate cut
        assert result.expected_recall is not None and READING_DIFFERS in readout(result)
        assert {"e4", "e9", "e14"} <= set(result.element_ids)
    # an untagged analysis tagged here by another tagger
    other = select(
        analysis_of(expansion_doc()),
        RISK,
        selector=_selector(),
        tagger=FakeTagger(fingerprint="fp-of-another-bundle"),
        embedder=FakeEmbedder(),
    )
    assert other.guarantee is False and other.fit_mismatch == ["tagger"]


def test_another_embedder_drops_the_guarantee():
    class OtherEmbedder(FakeEmbedder):
        name = "embedding:another-model"

    class Unnamed(FakeEmbedder):
        name = None

    for embedder in (OtherEmbedder(), Unnamed()):
        result = select(
            analysis_of(expansion_doc()),
            RISK,
            selector=_selector(),
            tagger=FakeTagger(),
            embedder=embedder,
        )
        assert result.guarantee is False and result.fit_mismatch == ["embedder"]
        assert READING_DIFFERS in readout(result)
    # prepare() records the spec it is told, over the encoder's own name
    ctx = prepare(
        analysis_of(expansion_doc()),
        tagger=FakeTagger(),
        encoder=Unnamed(),
        encoder_spec=EMBEDDER_SPEC,
    )
    assert ctx.embedder_spec == EMBEDDER_SPEC
    _, X = need_features(ctx, RISK)
    same = select_scored(
        analysis_of(expansion_doc()), ctx, X, RISK, completeness=0.9, selector=_selector()
    )
    assert same.guarantee is True and same.fit_mismatch == []


def test_another_reading_drops_the_guarantee():
    doc = expansion_doc()
    kwargs = {"selector": _selector(), "tagger": FakeTagger(), "embedder": FakeEmbedder()}
    balanced = select(analysis_of(doc, config=RouterConfig(profile="balanced")), RISK, **kwargs)
    assert balanced.guarantee is False and balanced.fit_mismatch == ["layout"]
    unlabelled = select(analysis_of(doc, sections=None), RISK, **kwargs)
    assert unlabelled.guarantee is False and unlabelled.fit_mismatch == ["sections"]
    # every difference is listed, in a fixed order
    many = select(
        analysis_of(_tagged_doc("fp-other"), sections=None, config=RouterConfig()),
        RISK,
        selector=_selector(),
        embedder=type("E", (FakeEmbedder,), {"name": "embedding:other"})(),
    )
    assert many.fit_mismatch == ["tagger", "embedder", "layout", "sections"]
    assert readout(many).count(READING_DIFFERS) == 1


def test_a_selector_without_provenance_guarantees_nothing(tmp_path):
    old = _selector(provenance=None)
    result = _select_risk(RISK, old)
    assert result.guarantee is False and result.fit_mismatch == ["provenance"]
    # an old selector.json, without the provenance key, loads as one without provenance
    data = _selector().to_dict()
    del data["provenance"]
    assert Selector.from_dict(data).provenance is None
    # the provenance round-trips, and the recorded embedder is the default one
    loaded = Selector.load(_selector().save(tmp_path))
    assert loaded.provenance == PROVENANCE and loaded.embedder == EMBEDDER_SPEC
    assert _select_risk(RISK, loaded).guarantee is True
    with pytest.raises(ValueError, match="embedder"):
        Selector(
            field=FIELD,
            field_fallback=FALLBACK,
            query=QUERY,
            query_fallback=FALLBACK,
            dials={},
            embedder="embedding:else",
            provenance=PROVENANCE,
        )


def test_completeness_one_needs_no_provenance():
    for selector in (_selector(provenance=None), None):
        whole = select(
            analysis_of(_tagged_doc("fp-other"), sections=None),
            RISK,
            completeness=1.0,
            selector=selector,
        )
        assert whole.guarantee is True and whole.fit_mismatch == []


def test_tag_fingerprint_is_left_out_of_json_when_unknown():
    assert "fingerprint" not in _topic("annual_report.other", None).model_dump()
    assert _topic("annual_report.other").model_dump()["fingerprint"] == TAGGER_FP


def test_the_need_text_is_part_of_what_was_calibrated():
    """sim compares elements with the field's query (else its description), so a field whose
    need text changed was not calibrated, whatever its id, topics and functions."""
    reworded = RISK.model_copy(update={"query": "Which lawsuits is the company party to?"})
    assert field_fingerprint(reworded) != field_fingerprint(RISK)
    result = _select_risk(reworded)
    assert result.guarantee is False and result.uncalibrated_field is True
    # without a query, the description is the need text
    plain = LEGAL.model_copy(update={"description": "Something else entirely."})
    assert field_fingerprint(plain) != field_fingerprint(LEGAL)
    # a label is not part of it
    assert field_fingerprint(RISK.model_copy(update={"label": "Risks"})) == field_fingerprint(RISK)


# --- a corrupt selector.json is a ValueError, never a traceback ------------------------------


def _corrupt(data, change):
    change(data)
    return data


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.pop("combiners"),
        lambda d: d["combiners"].pop("query"),
        lambda d: d["combiners"]["field"].pop("coef"),
        lambda d: d.update(dials=["not", "a", "mapping"]),
        lambda d: d["dials"]["annual_report"].pop("curve"),
        lambda d: d.update(embedder=5),
        lambda d: d.update(meta="a string"),
        lambda d: d["combiners"]["field_fallback"].update(scale=[0.0, 1.0]),
        lambda d: d["combiners"]["query_fallback"].update(intercept=float("nan")),
        lambda d: d["combiners"]["query"].update(coef=["x"] * len(FEATURE_NAMES)),
        lambda d: d.update(provenance={"tagger": "x"}),
        lambda d: d.update(provenance="a string"),
    ],
)
def test_a_corrupt_selector_is_a_value_error(change):
    data = _corrupt(_selector().to_dict(), change)
    with pytest.raises(ValueError, match="corrupt selector|combiner|embedder|meta|provenance"):
        Selector.from_dict(data)


def test_a_corrupt_selector_file_names_the_file(tmp_path):
    for text in ("[1, 2]", "{not json", '{"format": "docrooter-selector", "version": 1}'):
        path = tmp_path / "selector.json"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="selector.json"):
            Selector.load(path)


# --- the API's edges -------------------------------------------------------------------------


def test_select_argument_edges():
    analysis = analysis_of(expansion_doc())
    kwargs = {"selector": _selector(), "tagger": FakeTagger(), "embedder": FakeEmbedder()}
    assert select(analysis, RISK, budget=np.int64(200), **kwargs).tokens <= 200
    with pytest.raises(ValueError, match="completeness"):
        select(analysis, RISK, completeness=True, **kwargs)
    with pytest.raises(ValueError, match="need"):
        select(analysis, None, **kwargs)
    with pytest.raises(TypeError, match=r"process\(\)"):
        select("report.pdf", RISK, **kwargs)

    class NoEncode:
        name = "keyword"

    with pytest.raises(ValueError, match="encode"):
        select(analysis, RISK, selector=_selector(), tagger=FakeTagger(), embedder=NoEncode())


def test_the_select_helpers_are_exported():
    import sys

    package = sys.modules["docrooter.select"]  # the function shadows the package attribute

    for name in ("readout", "prepare", "need_features", "select_scored", "FitProvenance"):
        assert name in package.__all__ and getattr(package, name) is not None
