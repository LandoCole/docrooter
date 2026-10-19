"""Fitting the selector (v0.6 S5): the combiners on calib-fit, the dial on calib-crc, relevance
from ``teacher-full`` labels.

Everything is fake and torch-free: a tmp corpus store with ``teacher-full`` rows, a fake
``analyse`` that rebuilds each stored document as an analysis (no PDF is read), a tagger that
tags "risk" texts with the risk-factors topic, and an embedder with two directions.
"""

import hashlib
import sys

import numpy as np
import pytest

from docrooter import Document, RouterConfig
from docrooter.backends import registry
from docrooter.classify.normalize import NormalizationResult
from docrooter.classify.results import DocTypeResult, PathResult
from docrooter.corpus import Corpus, CorpusElement, LabelRow, ManifestRecord
from docrooter.corpus.store import calib_half, text_sha1
from docrooter.elements.models import Element, TagResult
from docrooter.errors import DocrooterError
from docrooter.rooter.analysis import Analysis
from docrooter.select import Field, calibration_of
from docrooter.select.fit import CorpusPdfReader, fit_selector
from docrooter.select.selection import FitProvenance, Selector, select
from docrooter.tagger.element_tags import tagger_fingerprint

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
    description="Lawsuits the company is party to.",
    topics=["annual_report.legal_proceedings"],
)
FIELDS = {"annual_report": [RISK, LEGAL]}
RISK_LEAF = "annual_report.risk_factors"
OTHER_LEAF = "annual_report.other"


# --- fakes -------------------------------------------------------------------------------------


class FakeTagger:
    """Tags "risk" texts with the family's risk leaf (mostly) and anything else with its
    catch-all; counts its calls and closes. With a ``path`` it is a saved bundle there (a
    ``tagger.json``), and its tags carry the bundle's fingerprint, as a real bundle's do."""

    def __init__(self, path=None):
        self.calls = 0
        self.closed = 0
        self.path = path
        if path is not None:
            path.mkdir(parents=True, exist_ok=True)
            (path / "tagger.json").write_text('{"format": "fake"}', encoding="utf-8")
        self.fingerprint = tagger_fingerprint(self)

    def tag(self, elements, family):
        self.calls += 1
        out = []
        for element in elements:
            risky = "risk" in element.text.lower()
            probs = (
                {f"{family}.risk_factors": 0.8, f"{family}.other": 0.2}
                if risky
                else {f"{family}.risk_factors": 0.1, f"{family}.other": 0.9}
            )
            best = max(probs, key=probs.__getitem__)
            out.append(
                {
                    "topic": TagResult(
                        labels=[best],
                        probs=probs,
                        confidence=probs[best],
                        source="f",
                        fingerprint=self.fingerprint,
                    )
                }
            )
        return out

    def close(self):
        self.closed += 1


class FakeEmbedder:
    """Two directions: texts that mention risk, and everything else."""

    name = "embedding:fake"

    def encode(self, texts, prompt=""):
        return np.array([[1.0, 0.0] if "risk" in t.lower() else [0.0, 1.0] for t in texts])


# --- the corpus --------------------------------------------------------------------------------


def ids_in(half: str, n: int, prefix: str) -> list[str]:
    out, i = [], 0
    while len(out) < n:
        doc_id = f"{prefix}{i}"
        if calib_half(doc_id) == half:
            out.append(doc_id)
        i += 1
    return out


def items_for(
    doc_id: str, seed: int, paragraphs: int | None = None, empty: int = 0
) -> list[tuple[str, str, str]]:
    """``(element id, kind, text)`` of a small annual report: a heading, risk paragraphs and
    filler (``6 + seed % 3`` paragraphs unless ``paragraphs``), ``empty`` paragraphs without
    text, and a table."""
    items = [("h0", "heading", "Risk Factors")]
    items += [(f"z{k}", "paragraph", "") for k in range(empty)]
    for j in range(6 + seed % 3 if paragraphs is None else paragraphs):
        risky = j % 3 == 0
        text = (
            f"Interest rate risk number {j} could harm {doc_id}."
            if risky
            else f"Filler paragraph {j} about the offices of {doc_id}."
        )
        items.append((f"p{j}", "paragraph", text))
    items.append(("t0", "table", f"Revenue table of {doc_id}"))
    return items


def leaf_of(text: str) -> str:
    return RISK_LEAF if "risk" in text.lower() and "factors" not in text.lower() else OTHER_LEAF


def record(
    doc_id: str, split: str = "calib", family: str = "annual_report", doc_type=None
) -> ManifestRecord:
    return ManifestRecord(
        id=doc_id,
        sha256=hashlib.sha256(doc_id.encode()).hexdigest(),
        split=split,
        family_hint=family,
        doc_type_hint=doc_type,
    )


def label_row(doc_id: str, element_id: str, text: str, leaf: str) -> LabelRow:
    other = OTHER_LEAF if leaf != OTHER_LEAF else RISK_LEAF
    return LabelRow(
        doc_id=doc_id,
        element_id=element_id,
        text_sha1=text_sha1(text),
        facet="topic",
        probs={leaf: 0.9, other: 0.1},
        provenance={"teacher": "mlx:fake"},
    )


class Store:
    """A tmp corpus store and the items of each document, for the fake ``analyse``."""

    def __init__(self, path):
        self.corpus = Corpus(path)
        self.items: dict[str, list[tuple[str, str, str]]] = {}

    def add(
        self,
        doc_id: str,
        split: str = "calib",
        family: str = "annual_report",
        seed=0,
        *,
        paragraphs=None,
        empty=0,
        unlabelled=(),
        doc_type=None,
    ):
        """Store a document and its labels: ``teacher-full`` for every content element with
        text (the labeller skips empty ones) but ``unlabelled``, and a contrary ``teacher``."""
        items = items_for(doc_id, seed, paragraphs, empty)
        self.items[doc_id] = items
        self.corpus.add_document(
            record(doc_id, split, family, doc_type),
            [
                CorpusElement(
                    doc_id=doc_id,
                    element_id=eid,
                    kind=kind,
                    text=text,
                    heading_path=["Risk Factors"],
                    section_title="Risk Factors",
                    page=1,
                )
                for eid, kind, text in items
            ],
            [1],
        )
        content = [
            (eid, text)
            for eid, kind, text in items
            if kind != "heading" and text.strip() and eid not in unlabelled
        ]
        self.corpus.write_labels(
            "teacher-full", [label_row(doc_id, eid, text, leaf_of(text)) for eid, text in content]
        )
        # The v0.5 sample labeller says the opposite everywhere: fitting must never read it.
        self.corpus.write_labels(
            "teacher",
            [
                label_row(doc_id, eid, text, "annual_report.legal_proceedings")
                for eid, text in content
            ],
        )


def make_doc(items) -> Document:
    doc = Document.from_sections([{"title": "Risk Factors", "level": 1}])
    doc.elements = [
        Element(
            id=eid,
            kind=kind,
            text=text,
            boxes=[],
            order=i,
            section_id="s1",
            **({"level": 1} if kind == "heading" else {}),
        )
        for i, (eid, kind, text) in enumerate(items)
    ]
    doc.sections["s1"].element_ids.extend(e.id for e in doc.elements)
    return doc


class FakeAnalyse:
    """Rebuilds a stored document as an analysis; ``edit`` may change its items first."""

    def __init__(self, store: Store, edit=None):
        self.store = store
        self.edit = edit
        self.calls: list[str] = []

    def __call__(self, rec: ManifestRecord) -> Analysis:
        self.calls.append(rec.id)
        items = list(self.store.items[rec.id])
        if self.edit is not None:
            items = self.edit(rec.id, items)
        return Analysis(
            document=make_doc(items),
            sections=NormalizationResult(doc_type=None, taxonomy="section_types/annual_report"),
            config=RouterConfig(profile="balanced"),  # sections labelled by its backend
        )


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "store")
    for i, doc_id in enumerate(ids_in("fit", 4, "fit-")):
        s.add(doc_id, seed=i)
    for i, doc_id in enumerate(ids_in("crc", 4, "crc-")):
        s.add(doc_id, seed=i)
    return s


def fit(store: Store, analyse=None, tagger=None, **kwargs):
    analyse = analyse or FakeAnalyse(store)
    tagger = tagger or FakeTagger()
    selector = fit_selector(
        store.corpus,
        tagger,
        FakeEmbedder(),
        fields_by_family=kwargs.pop("fields_by_family", FIELDS),
        analyse=analyse,
        **kwargs,
    )
    return selector, analyse, tagger


def expected_positives(store: Store, half: str) -> int:
    return sum(
        1
        for doc_id, items in store.items.items()
        if calib_half(doc_id) == half and doc_id.startswith(half)
        for _, kind, text in items
        if kind != "heading" and leaf_of(text) == RISK_LEAF
    )


# --- tests -------------------------------------------------------------------------------------


def test_fit_uses_teacher_full_labels_and_partitions(store):
    # A calib document of a family without fields is skipped and counted.
    store.add("contract-doc", family="contract")
    selector, analyse, tagger = fit(store)
    fit_ids = ids_in("fit", 4, "fit-")
    crc_ids = ids_in("crc", 4, "crc-")

    # calib-fit trains the four combiners, calib-crc calibrates the dial: never the other way.
    for combiner in (
        selector.field,
        selector.field_fallback,
        selector.query,
        selector.query_fallback,
    ):
        assert combiner.fitted and combiner.n_docs == 4
    content_rows = sum(len(store.items[d]) - 1 for d in fit_ids)  # all but the heading
    assert selector.field.n_train == 2 * content_rows  # two fields
    assert selector.query.n_train == selector.field.n_train
    assert selector.field_fallback.feature_names == ("sim", "section_hit")
    dial = selector.dials["annual_report"]
    assert dial.n == 4 and dial.n_docs == 4  # one RISK unit per crc document
    assert dial.n_excluded == 4  # LEGAL is never relevant there: no CRC unit
    dial.check_combiner(selector.field)
    assert selector.embedder == "embedding:fake"
    # The calibrated fields are those that gave the dial at least one CRC unit: LEGAL gave
    # none, so the dial says nothing about it (a family without a dial has none either).
    assert selector.calibrated_fields == calibration_of([RISK])
    assert selector.is_calibrated(RISK) and not selector.is_calibrated(LEGAL)
    assert not selector.is_calibrated(RISK.model_copy(update={"topics": [LEGAL.topics[0]]}))
    assert selector.meta["crc_units"] == {"annual_report": {RISK.id: 4, LEGAL.id: 0}}

    # Relevance is teacher-full's: the "teacher" labeller would make every row a LEGAL one.
    meta = selector.meta
    balance = meta["class_balance"]["field"]
    assert balance["relevant"] == expected_positives(store, "fit")
    assert balance["rows"] == selector.field.n_train
    assert balance["share"] == pytest.approx(balance["relevant"] / balance["rows"])
    assert meta["class_balance"]["query"] == balance  # queries follow their field
    assert meta["labeler"] == "teacher-full"

    assert meta["documents"]["fit"] == {"annual_report": 4}
    assert meta["documents"]["crc"] == {"annual_report": 4}
    assert meta["documents"]["skipped_no_fields"] == {"contract": 1}
    assert meta["dials"]["annual_report"] == {"n": 4, "n_docs": 4, "n_excluded": 4}
    assert meta["profile"] == "accurate"
    manifest_sha = hashlib.sha256(store.corpus.manifest_path.read_bytes()).hexdigest()
    assert meta["corpus_manifest_sha256"] == manifest_sha
    assert meta["label_mismatch"]["hash_mismatch"] == 0
    assert meta["label_mismatch"]["no_label"] == 0
    assert meta["label_mismatch"]["missing"] == 0
    # every combiner converged; the fake tagger is no saved bundle, which is said
    assert [w for w in meta["warnings"] if "not a saved bundle" not in w] == []
    assert meta["fields_using_description"] == ["annual_report.legal_proceedings"]  # no query

    # Each calib document is read once, fit documents first; the skipped one is never read.
    assert analyse.calls == fit_ids + crc_ids
    assert tagger.calls == 8


def test_gold_and_train_documents_never_used(store):
    gold = [ids_in("fit", 1, "gold-")[0], ids_in("crc", 1, "gold-")[0]]
    train = [ids_in("fit", 1, "train-")[0], ids_in("crc", 1, "train-")[0]]
    for doc_id in gold:
        store.add(doc_id, split="gold")  # full mode labels gold too
    for doc_id in train:
        store.add(doc_id, split="train")
    selector, analyse, _ = fit(store)
    assert not set(analyse.calls) & set(gold + train)
    assert selector.field.n_docs == 4
    assert selector.dials["annual_report"].n_docs == 4
    assert selector.meta["documents"]["fit"] == {"annual_report": 4}
    assert selector.meta["documents"]["crc"] == {"annual_report": 4}


def test_hash_mismatch_raises_naming_the_document(store):
    victim = ids_in("crc", 4, "crc-")[1]

    def drift(doc_id, items):
        if doc_id != victim:
            return items
        return [(eid, kind, text + " (read differently)") for eid, kind, text in items]

    with pytest.raises(DocrooterError, match=victim):
        fit(store, analyse=FakeAnalyse(store, edit=drift))


def big_store(path, *, unlabelled=0, empty=0):
    """Two fit and two crc documents; the first fit one has 100 content elements with text
    (99 paragraphs and a table), ``empty`` more without, and its first ``unlabelled``
    paragraphs have no teacher-full label."""
    s = Store(path)
    fit_ids, crc_ids = ids_in("fit", 2, "fit-"), ids_in("crc", 2, "crc-")
    s.add(
        fit_ids[0],
        paragraphs=99,
        empty=empty,
        unlabelled={f"p{j}" for j in range(unlabelled)},
    )
    s.add(fit_ids[1], seed=1)
    for i, doc_id in enumerate(crc_ids):
        s.add(doc_id, seed=i)
    return s, fit_ids[0]


def test_empty_elements_may_lack_labels(tmp_path):
    store, target = big_store(tmp_path / "s", empty=5)
    selector, _, _ = fit(store)
    per_doc = selector.meta["label_mismatch"]["per_document"][target]
    assert per_doc["content"] == 105 and per_doc["no_label"] == 5 and per_doc["missing"] == 0
    assert selector.meta["class_balance"]["field"]["relevant"] == expected_positives(store, "fit")


@pytest.mark.parametrize(("unlabelled", "raises"), [(2, False), (3, True), (99, True)])
def test_unlabelled_text_elements_boundary(tmp_path, unlabelled, raises):
    """Only elements without text may lack a label: more than 2% of a document's elements
    with text unlabelled (a labelling run that stopped early) raises, naming it."""
    store, target = big_store(tmp_path / "s", unlabelled=unlabelled)
    if raises:
        with pytest.raises(DocrooterError, match=target):
            fit(store)
        return
    selector, _, _ = fit(store)
    per_doc = selector.meta["label_mismatch"]["per_document"][target]
    assert per_doc["missing"] == 2 and per_doc["no_label"] == 0
    assert per_doc["rate"] == pytest.approx(2 / 100)
    assert selector.meta["label_mismatch"]["missing"] == 2


@pytest.mark.parametrize(("changed", "raises"), [(2, False), (3, True)])
def test_hash_mismatch_boundary(tmp_path, changed, raises):
    store, target = big_store(tmp_path / "s")
    edited = {f"p{j}" for j in range(changed)}

    def drift(doc_id, items):
        if doc_id != target:
            return items
        return [(e, k, t + " (read differently)" if e in edited else t) for e, k, t in items]

    analyse = FakeAnalyse(store, edit=drift)
    if raises:
        with pytest.raises(DocrooterError, match=target):
            fit(store, analyse=analyse)
        return
    selector, _, _ = fit(store, analyse=analyse)
    assert selector.meta["label_mismatch"]["per_document"][target]["hash_mismatch"] == 2


def test_fitted_selector_round_trips_and_selects_with_guarantee(store, tmp_path):
    bundle = tmp_path / "bundle"
    selector, _, _ = fit(store, tagger=FakeTagger(bundle))
    path = selector.save(bundle)  # kept in the bundle, as fit-selector users do
    loaded = Selector.load(path)
    assert loaded.to_dict() == selector.to_dict()
    assert loaded.meta == selector.meta
    for dial in loaded.dials.values():
        dial.check_combiner(loaded.field)

    doc = make_doc(items_for("new-doc", 1))
    analysis = Analysis(
        document=doc,
        doc_type=DocTypeResult(
            source="test", best=PathResult(node_ids=["financial", "financial.annual_report"])
        ),
        sections=NormalizationResult(doc_type=None, taxonomy="section_types/annual_report"),
        config=RouterConfig(profile="balanced"),  # the reading the fit used
    )
    result = select(
        analysis,
        RISK,
        completeness=0.5,
        selector=loaded,
        tagger=FakeTagger(bundle),  # the bundle the selector was fitted with
        embedder=FakeEmbedder(),
    )
    assert result.guarantee is True and result.capped is False and result.fit_mismatch == []
    risky = {e.id for e in doc.elements if e.kind != "heading" and leaf_of(e.text) == RISK_LEAF}
    assert risky & set(result.element_ids)
    assert result.threshold is not None


def test_fit_needs_teacher_full_labels(tmp_path):
    s = Store(tmp_path / "bare")
    s.add(ids_in("fit", 1, "fit-")[0])
    (s.corpus.dir / "labels" / "teacher-full.jsonl").unlink()
    with pytest.raises(DocrooterError, match="teacher-full") as raised:
        fit(s)
    assert "teach label --full" in str(raised.value)  # the flag that exists
    assert "--mode" not in str(raised.value)


def test_fit_records_its_provenance(store, tmp_path):
    """The tagger bundle's fingerprint, the embedder spec and the reading are recorded, so
    select() can refuse the guarantee on features made otherwise."""
    bundle = tmp_path / "bundle"
    tagger = FakeTagger(bundle)
    selector, _, _ = fit(store, tagger=tagger)
    assert selector.provenance == FitProvenance(
        tagger=tagger_fingerprint(tagger),
        embedder="embedding:fake",
        profile="accurate",
        layout="heuristic",  # what the analyses were read with (FakeAnalyse: balanced)
        sections=RouterConfig(profile="balanced").sections,  # and labelled with
    )
    assert selector.provenance.tagger is not None
    assert selector.meta["tagger"]["fingerprint"] == selector.provenance.tagger

    # readings without section labels give no known sections backend: never a guarantee
    def unlabelled(rec):
        return FakeAnalyse(store)(rec).model_copy(update={"sections": None})

    bare, _, _ = fit(store, analyse=unlabelled, tagger=FakeTagger(bundle))
    assert bare.provenance.sections is None
    assert any("sections" in w for w in bare.meta["warnings"])
    # a tagger that is not a saved bundle gives no fingerprint: the selector never guarantees
    plain, _, _ = fit(store)
    assert plain.provenance is not None and plain.provenance.tagger is None
    assert any("not a saved bundle" in w for w in plain.meta["warnings"])

    # readings with different layouts are no single reading
    def mixed(rec):
        analysis = FakeAnalyse(store)(rec)
        layout = "heuristic" if calib_half(rec.id) == "fit" else "heron"
        return analysis.model_copy(update={"config": RouterConfig(layout=layout)})

    mixed_selector, _, _ = fit(store, analyse=mixed, tagger=FakeTagger(bundle))
    assert mixed_selector.provenance.layout is None
    assert any("layout" in w for w in mixed_selector.meta["warnings"])


def test_fields_must_sit_under_their_own_family(store):
    contract = Field(
        id="contract.termination",
        label="Termination",
        description="How the contract ends.",
        topics=["contract.termination"],
    )
    with pytest.raises(ValueError, match="contract.termination"):
        fit(store, fields_by_family={"annual_report": [RISK, contract]})


class FakeEngines:
    def __init__(self):
        self.closed = 0

    def layout(self):
        return "LAYOUT", None

    def ocr(self):
        return None, None

    def close(self):
        self.closed += 1


def test_default_reader_reads_as_the_build_did(store, tmp_path, monkeypatch):
    doc_id = ids_in("fit", 1, "fit-")[0]
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    # The fake store's sha256 is of the id, so the "PDF" holds the id's bytes.
    (pdfs / f"{doc_id}.pdf").write_bytes(doc_id.encode())
    seen = {}

    def fake_process(path, **kwargs):
        seen["process"] = (path, kwargs)
        return Analysis(document=make_doc(store.items[doc_id]), config=kwargs["config"])

    def fake_normalize(doc, **kwargs):
        seen["normalize"] = kwargs
        return None

    closed = []

    class Backend:
        def close(self):
            closed.append(True)

    monkeypatch.setattr(sys.modules["docrooter.rooter"], "process", fake_process)
    monkeypatch.setattr(
        sys.modules["docrooter.classify.normalize"], "normalize_sections", fake_normalize
    )
    resolved = []

    def resolve(spec):
        resolved.append(Backend())
        return resolved[-1]

    pages = {"n": 1}
    monkeypatch.setattr(registry, "resolve_backend", resolve)
    monkeypatch.setattr(
        sys.modules["docrooter.corpus.build"], "_page_count", lambda path, engine: pages["n"]
    )
    reader = CorpusPdfReader(store.corpus, pdfs, profile="accurate", max_pages=30)
    engines = FakeEngines()
    reader._engines = engines
    analysis = reader(store.corpus.manifest()[0])
    assert [e.id for e in analysis.document.elements] == [i for i, _, _ in store.items[doc_id]]
    # the analysis records the backend that labelled its sections, for the fit provenance
    assert analysis.config.sections == reader.sections_spec != "none"

    path, kwargs = seen["process"]
    assert path == pdfs / f"{doc_id}.pdf"
    assert kwargs["pages"] == [1]  # the pages the build read, from the store
    assert kwargs["layout"] == "LAYOUT" and kwargs["ocr"] == "none"
    cfg = kwargs["config"]
    assert (cfg.profile, cfg.max_pages, cfg.layout) == ("accurate", 30, "heron")
    assert (cfg.doc_type, cfg.sections, cfg.embeddings, cfg.escalate) == ("none",) * 4
    assert seen["normalize"]["taxonomy"] == "section_types/annual_report"  # the family's
    # No doc type in the manifest: the family's first doc type, so the section prompt reads
    # "... of a <doc type>" as it does for a correctly typed document at runtime.
    assert seen["normalize"]["doc_type"] == "financial.annual_report"
    assert seen["normalize"]["backend"] is resolved[0]
    reader.close()
    assert engines.closed == 1 and closed == [True]

    # A missing PDF, or one that is not the manifest's bytes, is refused before reading.
    other = store.corpus.manifest()[1]
    with pytest.raises(DocrooterError, match="no PDF"):
        reader(other)
    (pdfs / f"{other.id}.pdf").write_bytes(b"other bytes")
    with pytest.raises(DocrooterError, match="sha256"):
        reader(other)

    # The manifest's doc type is used when it maps to the document's family.
    quarterly = ids_in("fit", 1, "q-")[0]
    store.add(quarterly, doc_type="financial.quarterly_report")
    (pdfs / f"{quarterly}.pdf").write_bytes(quarterly.encode())
    reader(store.corpus.manifest()[-1])
    assert seen["normalize"]["doc_type"] == "financial.quarterly_report"

    # Stored pages that the page cap could not have produced: another --max-pages.
    pages["n"] = 40  # a 40-page PDF read with max_pages=30 gives 30 pages, not 1
    with pytest.raises(DocrooterError, match="max-pages"):
        reader(store.corpus.manifest()[0])
    reader.close()


def test_reader_shares_the_sections_backend_with_the_embedder(store, tmp_path, monkeypatch):
    """One embedding model, not two, when the profile's sections backend is the sim
    embedder's spec; a shared backend is the caller's to close."""
    made = []

    class Reader:
        def __init__(self, corpus, pdf_dir, **kwargs):
            self.kwargs = kwargs
            made.append(self)

        def __call__(self, record):
            return FakeAnalyse(store)(record)

        def close(self):
            pass

    embedders = []

    def resolve(spec):
        embedders.append(FakeEmbedder())
        return embedders[-1]

    from docrooter.rooter.profiles import profile_defaults
    from docrooter.select import fit as fit_module

    monkeypatch.setattr(fit_module, "CorpusPdfReader", Reader)
    monkeypatch.setattr(registry, "resolve_backend", resolve)
    sections = profile_defaults("accurate")["sections"]
    fit_selector(store.corpus, FakeTagger(), sections, fields_by_family=FIELDS)
    assert len(embedders) == 1 and made[-1].kwargs["sections_backend"] is embedders[0]
    fit_selector(store.corpus, FakeTagger(), "embedding:other", fields_by_family=FIELDS)
    assert made[-1].kwargs["sections_backend"] is None

    shared = object()
    reader = CorpusPdfReader(store.corpus, tmp_path, sections_backend=shared)
    reader._engines = FakeEngines()
    reader.close()  # leaves the shared backend alone (it has no close to call anyway)
    assert reader._sections is shared
