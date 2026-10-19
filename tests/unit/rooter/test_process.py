"""``process()`` / ``process_many()`` (spec §7): the fast path, element mode, the route log,
escalation, engine reuse and ownership, and the tables switch. Keyword and fake backends, fake
OCR and layout engines; no model is ever loaded."""

import contextlib
import json
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

import docrooter
from docrooter import classify_document, load, normalize_sections
from docrooter.backends import Scores
from docrooter.backends import registry as backend_registry
from docrooter.classify.normalize import GENERIC_TAXONOMY
from docrooter.embed import EmbeddingSet
from docrooter.errors import BackendError, DocrooterError, MissingExtraError
from docrooter.layout import registry as layout_registry
from docrooter.ocr import registry as ocr_registry
from docrooter.rooter import Analysis, RouterConfig, process, process_many
from docrooter.taxonomies import builtin_taxonomy

pymupdf = pytest.importorskip("pymupdf")

GOLDEN = Path(__file__).parents[1] / "pdf" / "golden"
MARKDOWN = (
    "# Annual Report\n\nThe company grew this year. Revenue rose by a third. Costs fell.\n\n"
    "## Risk Factors\n\nMarkets may fall. Rates may rise. Suppliers may fail.\n"
)


@pytest.fixture(autouse=True)
def _no_real_ocr(monkeypatch):
    """``ocr="auto"`` must never pick an installed engine here (ocrmac is forbidden in unit
    tests); tests that want one patch this again."""
    monkeypatch.setattr(ocr_registry, "resolve_auto_ocr", lambda: None)


@pytest.fixture
def registries(monkeypatch):
    """Private copies of the OCR, layout and backend registries."""
    monkeypatch.setattr(ocr_registry, "_FACTORIES", dict(ocr_registry._FACTORIES))
    monkeypatch.setattr(layout_registry, "_FACTORIES", dict(layout_registry._FACTORIES))
    monkeypatch.setattr(backend_registry, "_KINDS", dict(backend_registry._KINDS))


class PreferBackend:
    """A backend that gives probability ``prefer[node_id]`` to the first candidate listed in
    ``prefer`` and spreads the rest evenly (uniform when none is listed). It records each
    ``score`` call as ``(number of texts, candidate ids)`` and counts closes."""

    max_candidates = None
    temperature = 1.0

    def __init__(self, name="prefer", prefer=None):
        self.name = name
        self.prefer = dict(prefer or {})
        self.calls = []
        self.closed = 0

    def score(self, texts, candidates, *, instruction=None, multi_label=False):
        ids = [c.node_id for c in candidates]
        self.calls.append((len(texts), ids))
        hit = next((i for i, node in enumerate(ids) if node in self.prefer), None)
        if hit is None or len(ids) == 1:
            row = np.full(len(ids), 1.0 / len(ids))
        else:
            p = self.prefer[ids[hit]]
            row = np.full(len(ids), (1.0 - p) / (len(ids) - 1))
            row[hit] = p
        return Scores(np.tile(row, (len(texts), 1)), "softmax")

    def supports(self, node_id):
        return True

    def tokenizer(self):
        return None

    def close(self):
        self.closed += 1

    def section_calls(self, taxonomy):
        """Calls that scored sections of the built-in section ``taxonomy``."""
        ids = set(builtin_taxonomy(taxonomy).nodes)
        return [call for call in self.calls if set(call[1]) <= ids]


class FakeOcr:
    name = "fake-ocr"

    def __init__(self):
        self.calls = 0
        self.closed = 0

    def recognize(self, image, *, width_px, height_px, page):
        self.calls += 1
        return []

    def close(self):
        self.closed += 1


class FakeLayout:
    name = "fake-layout"

    def __init__(self):
        self.calls = 0
        self.closed = 0

    def detect(self, path, pages, *, dpi=None):
        self.calls += 1
        return []

    def close(self):
        self.closed += 1


class FakeEmbedding:
    """An embedding backend: deterministic unit vectors, every ``encode`` call recorded."""

    name = "embedding:fake/embed"
    model_id = "fake/embed"
    recipe = SimpleNamespace(revision="0123abcd", query_prompt="Q: {instruction}\n")

    def __init__(self, dim=4):
        self.dim = dim
        self.calls = []
        self.closed = 0

    def encode(self, texts, *, prompt):
        self.calls.append((list(texts), prompt))
        rows = []
        for text in texts:
            vec = np.random.default_rng(zlib.crc32(text.encode())).normal(size=self.dim)
            rows.append((vec / np.linalg.norm(vec)).astype(np.float32))
        return np.stack(rows) if rows else np.zeros((0, 0), dtype=np.float32)

    def close(self):
        self.closed += 1


class FailingBackend(PreferBackend):
    """A backend whose ``score`` raises ``error``."""

    def __init__(self, error, name="failing"):
        super().__init__(name)
        self.error = error

    def score(self, texts, candidates, *, instruction=None, multi_label=False):
        self.calls.append((len(texts), [c.node_id for c in candidates]))
        raise self.error


def _paper(tmp_path, name="paper.pdf"):
    """A one-page paper: a title, then three headed sections of four sentences each."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "A Study of Widgets", fontsize=24, fontname="hebo")
    y = 120.0
    for heading in ("Introduction", "Methods", "Results"):
        page.insert_text((72, y), heading, fontsize=16, fontname="hebo")
        y += 28
        for k in range(4):
            text = f"{heading} sentence {k} explains the widget experiment in plain words."
            page.insert_text((72, y), text, fontsize=11, fontname="helv")
            y += 15
        y += 12
    path = tmp_path / name
    doc.save(str(path))
    doc.close()
    return path


def _without_timings(doc_json):
    data = json.loads(doc_json)
    data["meta"].pop("timings", None)
    return data


def _routes(analysis, task=None):
    """The document-scope route decisions (configuration and escalation) of ``task``."""
    routes = analysis.document.meta.get("routes", [])
    return [r for r in routes if r["scope"] == "document" and (task is None or r["task"] == task)]


# --- the fast path ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["pdf", "markdown"])
def test_fast_profile_matches_v03_functions(tmp_path, kind):
    source = _paper(tmp_path) if kind == "pdf" else MARKDOWN
    analysis = process(source, profile="fast", doc_type="keyword", sections="keyword")

    doc = load(source)
    doc_type = classify_document(doc, builtin_taxonomy("doc_types"), backend="keyword")
    leaf = doc_type.best.node_ids[-1] if doc_type.best.node_ids else None
    sections = normalize_sections(
        doc, doc_type=leaf, taxonomy=None if leaf else GENERIC_TAXONOMY, backend="keyword"
    )

    assert _without_timings(analysis.document.model_dump_json()) == _without_timings(
        doc.model_dump_json()
    )
    assert "routes" not in analysis.document.meta and not analysis.document.elements
    assert "elements" not in json.loads(analysis.document.model_dump_json())
    assert analysis.doc_type is not None and analysis.sections is not None
    assert analysis.doc_type.model_dump(exclude={"timings"}) == doc_type.model_dump(
        exclude={"timings"}
    )
    assert analysis.sections.model_dump_json() == sections.model_dump_json()
    assert analysis.summaries == {} and analysis.embeddings is None
    assert analysis.config.profile == "fast" and analysis.config.layout == "none"
    assert {"load", "doc_type", "sections", "total"} <= set(analysis.timings)


def test_fast_with_tasks_off(tmp_path):
    analysis = process(_paper(tmp_path), profile="fast", doc_type="none", sections="none")
    assert analysis.doc_type is None and analysis.sections is None
    assert set(analysis.timings) == {"load", "total"}


# --- element mode ----------------------------------------------------------------------------


def test_balanced_has_elements_routes_summaries(tmp_path, registries, monkeypatch):
    ocr_registry.register_ocr_engine("fake-ocr", lambda *, lang="en": FakeOcr())
    monkeypatch.setattr(ocr_registry, "resolve_auto_ocr", lambda: "fake-ocr")
    seen = []
    analysis = process(
        _paper(tmp_path), profile="balanced", doc_type="keyword", sections="keyword",
        on_event=seen.append,
    )  # fmt: skip
    doc = analysis.document
    assert doc.meta["element_mode"] is True and doc.elements
    assert {"heading", "paragraph"} <= {e.kind for e in doc.elements}
    routes = doc.meta["routes"]
    head = [(r["scope"], r["task"], r["engine"]) for r in routes[:5]]
    assert head == [
        ("document", "text", "pymupdf"),
        ("document", "ocr", "fake-ocr"),
        ("document", "layout", "heuristic"),
        ("document", "doc_type", "keyword"),
        ("document", "sections", "keyword"),
    ]
    assert routes[1]["reason"].startswith("ocr=auto")
    assert [r["scope"] for r in routes[5:]] == ["page"]  # then the page decisions
    assert analysis.summaries and all(s.method == "centroid" for s in analysis.summaries.values())
    assert set(analysis.summaries) <= set(doc.sections) | {e.id for e in doc.elements}
    assert analysis.config.summaries == "centroid" and analysis.config.ocr == "auto"
    assert {"load", "doc_type", "sections", "summaries", "total"} <= set(analysis.timings)
    # the configuration decisions are route events, and the run has one ``done``
    route_progress = [e for e in seen if e.stage == "route" and e.status == "progress"]
    assert any(e.data.get("task") == "ocr" and e.data.get("engine") == "fake-ocr"
               for e in route_progress)  # fmt: skip
    assert [e.stage for e in seen].count("done") == 1
    assert any(e.stage == "summarize" for e in seen)


def test_markdown_in_element_mode(registries):
    analysis = process(MARKDOWN, doc_type="keyword", sections="keyword", summaries="lead")
    doc = analysis.document
    assert doc.elements
    assert [r["task"] for r in doc.meta["routes"]] == ["doc_type", "sections"]
    assert all(s.method == "lead" for s in analysis.summaries.values())


def test_ocr_auto_unavailable_route_warning():
    seen = []
    analysis = process(
        GOLDEN / "scanned.pdf", profile="balanced", doc_type="none", sections="none",
        on_event=seen.append,
    )  # fmt: skip
    (ocr,) = _routes(analysis, "ocr")
    assert ocr["scope"] == "document" and ocr["engine"] == "none"
    assert ocr["reason"].startswith("warning: ocr=auto and no OCR engine is installed")
    warnings = analysis.document.meta["warnings"]
    assert "ocr_unavailable" in warnings and "ocr_skipped" in warnings
    pages = [r for r in analysis.document.meta["routes"] if r["scope"] == "page"]
    assert pages and all(r["reason"].startswith("needs_ocr") for r in pages)
    assert any(
        e.stage == "route" and e.status == "progress" and e.data.get("task") == "ocr"
        and e.data["reason"].startswith("warning") for e in seen
    )  # fmt: skip


def test_explicit_ocr_none_has_no_warning(tmp_path):
    analysis = process(_paper(tmp_path), ocr="none", doc_type="none", sections="none")
    (ocr,) = _routes(analysis, "ocr")
    assert ocr["engine"] == "none" and not ocr["reason"].startswith("warning")
    assert "ocr_unavailable" not in analysis.document.meta["warnings"]


def test_ocr_lang_and_dpi_reach_the_engine(registries, monkeypatch):
    langs = []

    def factory(*, lang="en"):
        langs.append(lang)
        return FakeOcr()

    ocr_registry.register_ocr_engine("fake-ocr", factory)
    rendered = []
    import docrooter.ocr.render as render

    real = render.render_page

    def spy(path, page, *, dpi, clip=None):
        rendered.append(dpi)
        return real(path, page, dpi=dpi, clip=clip)

    monkeypatch.setattr(render, "render_page", spy)
    process(GOLDEN / "scanned.pdf", ocr="fake-ocr", ocr_lang="de", ocr_dpi=72,
            doc_type="none", sections="none", summaries="none")  # fmt: skip
    assert langs == ["de"] and rendered and set(rendered) == {72}


# --- escalation ------------------------------------------------------------------------------


def test_escalation_replaces_low_confidence_doc_type(tmp_path):
    primary = PreferBackend("primary", {"legal.contract": 0.3})
    strong = PreferBackend("strong", {"research.research_paper": 0.9})
    seen = []
    analysis = process(
        _paper(tmp_path), profile="fast", doc_type=primary, sections="none", escalate=strong,
        on_event=seen.append,
    )  # fmt: skip
    assert analysis.doc_type.best.node_ids[-1] == "research.research_paper"
    assert analysis.doc_type.confidence == pytest.approx(0.9)
    first, second = _routes(analysis, "doc_type")
    assert (first["engine"], second["engine"]) == ("primary", "strong")
    assert first["metrics"]["confidence"] == pytest.approx(0.3)
    assert "legal.contract escalated" in first["reason"]
    assert "below escalate_below 0.5" in first["reason"]
    assert second["metrics"]["confidence"] == pytest.approx(0.9)
    assert "research.research_paper" in second["reason"]
    assert analysis.config.escalate == "strong" and analysis.config.doc_type == "primary"
    assert "escalate" in analysis.timings
    assert any(e.stage == "escalate" and e.status == "end" for e in seen)


def test_escalation_changed_type_renormalises_all_sections(tmp_path):
    # the primary backend labels sections confidently, so only the type is escalated
    primary = PreferBackend("primary", {"legal.contract": 0.3, "method": 0.8, "term": 0.8})
    strong = PreferBackend("strong", {"research.research_paper": 0.9})
    analysis = process(
        _paper(tmp_path), profile="fast", doc_type=primary, sections=primary, escalate=strong
    )
    result = analysis.sections
    assert result.doc_type == "research.research_paper"
    assert result.taxonomy == "section_types/research_paper"
    assert result.sections and all(n.canonical == "method" for n in result.sections.values())
    # every section went through the primary backend with the new taxonomy, in one batch
    (call,) = primary.section_calls("section_types/research_paper")
    assert call[0] == len(result.sections)
    assert primary.section_calls("section_types/contract") == []
    assert strong.section_calls("section_types/research_paper") == []
    (renormalised,) = _routes(analysis, "sections")
    assert "section_types/research_paper" in renormalised["reason"]
    assert "section_types/contract" in renormalised["reason"]


def test_low_confidence_sections_relabelled_in_one_batch(tmp_path):
    primary = PreferBackend("primary", {"research.research_paper": 0.9})  # sections: uniform
    strong = PreferBackend("strong", {"results": 0.95})
    analysis = process(
        _paper(tmp_path), profile="fast", doc_type=primary, sections=primary, escalate=strong
    )
    result = analysis.sections
    assert result.taxonomy == "section_types/research_paper"
    assert all(n.canonical == "results" for n in result.sections.values())
    assert all(n.score == pytest.approx(0.95) for n in result.sections.values())
    (batch,) = strong.section_calls("section_types/research_paper")
    assert batch[0] == len(result.sections)
    assert _routes(analysis, "doc_type") == []  # the type was confident
    (relabelled,) = _routes(analysis, "sections")
    assert relabelled["engine"] == "strong"
    assert relabelled["metrics"]["sections"] == len(result.sections)


def test_no_escalation_above_threshold(tmp_path):
    primary = PreferBackend("primary", {"research.research_paper": 0.8, "results": 0.7})
    strong = PreferBackend("strong", {"legal.contract": 0.9})
    analysis = process(
        _paper(tmp_path), profile="fast", doc_type=primary, sections=primary, escalate=strong
    )
    assert strong.calls == []
    assert analysis.doc_type.best.node_ids[-1] == "research.research_paper"
    assert "routes" not in analysis.document.meta  # fast, and nothing escalated
    assert "escalate" not in analysis.timings
    # a lower threshold than the confidence never escalates; a higher one does
    process(_paper(tmp_path), profile="fast", doc_type=primary, sections="none",
            escalate=strong, escalate_below=0.85)  # fmt: skip
    assert len(strong.calls) == 1


def test_empty_document_is_not_escalated():
    primary = PreferBackend("primary")
    strong = PreferBackend("strong", {"legal.contract": 0.9})
    analysis = process(GOLDEN / "scanned.pdf", profile="fast", doc_type=primary,
                       sections="none", escalate=strong)  # fmt: skip
    assert analysis.doc_type.best.stopped_reason == "empty_text"
    assert strong.calls == []


SECRET = "sk-test-0123456789"


def _assert_no_secret(analysis, seen):
    """Neither the route log, the events nor the JSON may carry the error's message."""
    assert SECRET not in analysis.to_json()
    assert SECRET not in json.dumps([e.model_dump() for e in seen])


def test_failing_escalation_keeps_the_primary_doc_type(tmp_path):
    primary = PreferBackend("primary", {"legal.contract": 0.3})
    broken = FailingBackend(RuntimeError(f"401 for key {SECRET}"))
    seen = []
    analysis = process(
        _paper(tmp_path), profile="fast", doc_type=primary, sections="none", escalate=broken,
        on_event=seen.append,
    )  # fmt: skip
    assert analysis.doc_type.best.node_ids[-1] == "legal.contract"
    assert analysis.doc_type.confidence == pytest.approx(0.3)
    assert len(broken.calls) == 1
    (warning,) = _routes(analysis, "doc_type")
    assert warning["engine"] == "failing"
    assert warning["reason"].startswith("warning:")
    assert "failing" in warning["reason"] and "RuntimeError" in warning["reason"]
    assert "primary result kept" in warning["reason"]
    assert "escalate" in analysis.timings
    _assert_no_secret(analysis, seen)
    stage = [e for e in seen if e.stage == "escalate" and e.status == "end"]
    assert stage and stage[0].data.get("error") == "RuntimeError"
    assert [e.stage for e in seen].count("done") == 1  # the run itself did not fail


def test_failing_escalation_keeps_the_primary_section_labels(tmp_path):
    primary = PreferBackend("primary", {"research.research_paper": 0.9})  # sections: uniform
    broken = FailingBackend(ValueError(f"bad key {SECRET}"))
    ok = process(
        _paper(tmp_path), profile="fast", doc_type=primary, sections=primary,
        escalate=PreferBackend("strong", {"results": 0.95}),
    )  # fmt: skip
    seen = []
    analysis = process(
        _paper(tmp_path), profile="fast", doc_type=primary, sections=primary, escalate=broken,
        on_event=seen.append,
    )  # fmt: skip
    assert broken.calls and analysis.sections.sections
    assert {sid: n.canonical for sid, n in analysis.sections.sections.items()} != {
        sid: n.canonical for sid, n in ok.sections.sections.items()
    }
    assert all(n.score < 0.5 for n in analysis.sections.sections.values())
    (warning,) = _routes(analysis, "sections")
    assert warning["engine"] == "failing" and "ValueError" in warning["reason"]
    assert warning["reason"].startswith("warning:")
    _assert_no_secret(analysis, seen)


def test_failed_doc_type_escalation_is_not_retried_for_the_sections(tmp_path):
    # the primary labels sections with low scores, so the sections would be escalated too
    primary = PreferBackend("primary", {"legal.contract": 0.3})
    broken = FailingBackend(RuntimeError("boom"))
    analysis = process(
        _paper(tmp_path), profile="fast", doc_type=primary, sections=primary, escalate=broken
    )  # fmt: skip
    assert len(broken.calls) == 1  # the doc-type attempt only
    (warning,) = _routes(analysis)  # one warning in the whole route log
    assert warning["task"] == "doc_type" and "RuntimeError" in warning["reason"]
    assert analysis.sections is not None and analysis.doc_type.confidence == pytest.approx(0.3)


def test_escalation_backend_that_cannot_be_built_is_a_warning_too(tmp_path, registries):
    def broken(*args, **kwargs):
        raise BackendError(f"missing key {SECRET}")

    backend_registry.register_backend_kind("broken", broken)
    primary = PreferBackend("primary", {"legal.contract": 0.3})
    seen = []
    analysis = process(
        _paper(tmp_path), profile="fast", doc_type=primary, sections="none", escalate="broken",
        on_event=seen.append,
    )  # fmt: skip
    assert analysis.doc_type.best.node_ids[-1] == "legal.contract"
    (warning,) = _routes(analysis, "doc_type")
    assert warning["engine"] == "broken" and "BackendError" in warning["reason"]
    _assert_no_secret(analysis, seen)


def test_process_many_continues_after_a_failed_escalation(tmp_path):
    primary = PreferBackend("primary", {"legal.contract": 0.3})
    broken = FailingBackend(RuntimeError("boom"))
    sources = [_paper(tmp_path, "a.pdf"), _paper(tmp_path, "b.pdf")]
    results = list(
        process_many(sources, profile="fast", doc_type=primary, sections="none", escalate=broken)
    )
    assert len(results) == 2 and len(broken.calls) == 2
    assert all(len(_routes(a, "doc_type")) == 1 for a in results)


def test_accurate_without_jev_key_has_no_escalation(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    primary = PreferBackend("primary", {"legal.contract": 0.3})
    layout = FakeLayout()
    analysis = process(_paper(tmp_path), profile="accurate", layout=layout, doc_type=primary,
                       sections="none")  # fmt: skip
    assert analysis.config.escalate == "none"
    assert analysis.doc_type.best.node_ids[-1] == "legal.contract"
    assert _routes(analysis, "escalate") == [] and len(_routes(analysis, "doc_type")) == 1
    assert layout.calls == 1 and layout.closed == 0  # the caller's engine stays open
    assert analysis.document.meta["layout"] == "fake-layout"


def test_accurate_without_layout_extra_raises(tmp_path, monkeypatch):
    import docrooter.layout.heron as heron

    def missing(module, extra):
        raise MissingExtraError(f"{module} is required for this feature: pip install "
                                f"'docrooter[{extra}]'")  # fmt: skip

    monkeypatch.setattr(heron, "require", missing)
    with pytest.raises(MissingExtraError, match=r"docrooter\[layout\]"):
        process(_paper(tmp_path), profile="accurate", doc_type="keyword", sections="keyword")


# --- engines and backends: reuse and ownership -----------------------------------------------


def _counting(registries):
    made = {"ocr": [], "layout": [], "backend": []}

    def ocr_factory(*, lang="en"):
        made["ocr"].append(FakeOcr())
        return made["ocr"][-1]

    def layout_factory():
        made["layout"].append(FakeLayout())
        return made["layout"][-1]

    def backend_factory(*args, **kwargs):
        made["backend"].append(PreferBackend("count", {"research.research_paper": 0.9}))
        return made["backend"][-1]

    ocr_registry.register_ocr_engine("count-ocr", ocr_factory)
    layout_registry.register_layout_engine("count-layout", layout_factory)
    backend_registry.register_backend_kind("count", backend_factory)
    return made


def test_process_many_reuses_and_closes_backends(tmp_path, registries):
    made = _counting(registries)
    sources = [GOLDEN / "scanned.pdf", _paper(tmp_path), GOLDEN / "scanned.pdf"]
    results = process_many(
        sources, ocr="count-ocr", layout="count-layout", doc_type="count", sections="count",
        escalate="count", escalate_below=0.95, summaries="none",
    )  # fmt: skip
    first = next(results)
    assert isinstance(first, Analysis)
    assert [len(made[k]) for k in ("ocr", "layout", "backend")] == [1, 1, 1]
    assert made["ocr"][0].closed == made["layout"][0].closed == made["backend"][0].closed == 0
    rest = list(results)
    assert len(rest) == 2
    ocr, layout, backend = made["ocr"][0], made["layout"][0], made["backend"][0]
    assert [len(made[k]) for k in ("ocr", "layout", "backend")] == [1, 1, 1]
    assert (ocr.closed, layout.closed, backend.closed) == (1, 1, 1)
    assert ocr.calls >= 2 and layout.calls == 3  # both scans were OCR'd, every PDF laid out
    assert {a.document.meta["layout"] for a in [first, *rest]} == {"fake-layout"}


def test_process_closes_what_it_created(tmp_path, registries):
    made = _counting(registries)
    process(GOLDEN / "scanned.pdf", ocr="count-ocr", layout="count-layout", doc_type="count",
            sections="count", summaries="none")  # fmt: skip
    assert [len(made[k]) for k in ("ocr", "layout", "backend")] == [1, 1, 1]
    assert [made[k][0].closed for k in ("ocr", "layout", "backend")] == [1, 1, 1]


def test_process_many_closes_when_stopped_early(tmp_path, registries):
    made = _counting(registries)
    paths = [_paper(tmp_path, "a.pdf"), _paper(tmp_path, "b.pdf")]
    with contextlib.closing(process_many(paths, doc_type="count", sections="none")) as results:
        next(results)
    assert len(made["backend"]) == 1 and made["backend"][0].closed == 1


def test_ocr_engine_is_created_only_when_a_page_needs_it(tmp_path, registries):
    made = _counting(registries)
    analysis = process(_paper(tmp_path), ocr="count-ocr", doc_type="none", sections="none")
    assert made["ocr"] == []  # a born-digital page: no engine, nothing to close
    (ocr,) = _routes(analysis, "ocr")
    assert ocr["engine"] == "count-ocr"
    with pytest.raises(DocrooterError, match="unknown OCR engine 'nope'"):
        process(_paper(tmp_path), ocr="nope")


def test_process_many_checks_the_config_upfront():
    with pytest.raises(ValidationError):
        process_many([GOLDEN / "scanned.pdf"], bogus=1)


@pytest.fixture
def no_loading(monkeypatch):
    """Loading a document fails the test: the configuration must be rejected before it."""

    def load(*args, **kwargs):
        raise AssertionError("a document was loaded before the configuration was checked")

    # the module, not the ``process`` function that shadows it on the package
    monkeypatch.setattr(sys.modules["docrooter.rooter.process"], "load", load)


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        (dict(doc_type="bogus:x"), BackendError, "unknown backend kind 'bogus'"),
        (dict(sections="bogus:x"), BackendError, "unknown backend kind 'bogus'"),
        (dict(escalate="jevv:jev-1.13.0"), BackendError, "unknown backend kind 'jevv'"),
        (dict(embeddings="bogus:x"), BackendError, "unknown backend kind 'bogus'"),
        (dict(layout="heronn"), DocrooterError, "unknown layout engine 'heronn'"),
        (dict(ocr="nope"), DocrooterError, "unknown OCR engine 'nope'"),
        (dict(text="both"), DocrooterError, "text='both' has no element mode"),
    ],
)
def test_bad_specs_and_names_fail_before_any_document_is_read(no_loading, overrides, error,
                                                              match):  # fmt: skip
    # Final review R-I1: on a PDF or a Markdown text alike, and at process_many call time
    for source in (GOLDEN / "scanned.pdf", MARKDOWN):
        with pytest.raises(error, match=match):
            process(source, **overrides)
        with pytest.raises(error, match=match):
            process_many([source], **overrides)  # raised by the call, not by next()


def test_a_mistyped_escalate_kind_is_an_error_not_a_route_warning(no_loading):
    with pytest.raises(BackendError, match="unknown backend kind"):
        process(MARKDOWN, profile="fast", doc_type="keyword", escalate="bogus:x",
                escalate_below=0.0)  # fmt: skip


def test_good_specs_still_resolve(tmp_path, registries):
    # registered kinds and engines pass the up-front check; caller objects are not checked
    backend_registry.register_backend_kind("mine", lambda *a, **k: PreferBackend("mine"))
    layout_registry.register_layout_engine("fake", FakeLayout)
    analysis = process(MARKDOWN, doc_type="mine:any", sections="none", layout="fake", ocr="none",
                       text="pymupdf", escalate="none")  # fmt: skip
    assert analysis.config.layout == "fake" and analysis.doc_type is not None
    assert process(MARKDOWN, profile="fast", text="both", doc_type="none", sections="none")


def test_process_many_rejects_a_single_source(tmp_path):
    for single in ("paper.pdf", MARKDOWN, tmp_path / "paper.pdf", Path("paper.pdf"),
                   b"paper.pdf", bytearray(b"paper.pdf")):  # fmt: skip
        with pytest.raises(TypeError, match="pass a list"):
            process_many(single, profile="fast")  # raised by the call, not by the first next()
    # a list of one, and any other iterable, are fine
    assert len(list(process_many([_paper(tmp_path)], profile="fast", doc_type="none",
                                 sections="none"))) == 1  # fmt: skip
    assert len(list(process_many((p for p in [_paper(tmp_path, "b.pdf")]), profile="fast",
                                 doc_type="none", sections="none"))) == 1  # fmt: skip


def test_caller_owned_backend_not_closed(tmp_path):
    backend = PreferBackend("mine", {"research.research_paper": 0.9})
    ocr, layout = FakeOcr(), FakeLayout()
    sources = [GOLDEN / "scanned.pdf", _paper(tmp_path)]
    analysis = process(sources[0], doc_type=backend, sections=backend, ocr=ocr, layout=layout)
    assert list(process_many(sources, doc_type=backend, sections=backend, escalate=backend,
                             ocr=ocr, layout=layout))  # fmt: skip
    assert (backend.closed, ocr.closed, layout.closed) == (0, 0, 0)
    assert ocr.calls >= 2 and layout.calls == 3
    config = analysis.config
    assert (config.doc_type, config.sections, config.ocr, config.layout) == (
        "mine", "mine", "fake-ocr", "fake-layout",
    )  # fmt: skip
    head = {r["task"]: r for r in analysis.document.meta["routes"][:5]}
    assert head["ocr"]["reason"] == head["layout"]["reason"] == "engine passed by the caller"
    assert head["doc_type"]["reason"] == "passed by the caller"


def test_caller_classifier_is_used_and_kept_open(tmp_path):
    backend = PreferBackend("mine", {"research.research_paper": 0.9})
    classifier = docrooter.Classifier(builtin_taxonomy("doc_types"), backend)
    analysis = process(_paper(tmp_path), profile="fast", doc_type=classifier, sections="none")
    assert analysis.doc_type.best.node_ids[-1] == "research.research_paper"
    assert analysis.config.doc_type == "mine" and backend.closed == 0


def test_sections_classifier_on_another_taxonomy_uses_its_backend_with_a_warning(tmp_path):
    # Final review R-I3: the detected type maps to the contract taxonomy, the caller's
    # Classifier is built on the research-paper one: never an error, and the batch goes on
    detector = PreferBackend("detector", {"legal.contract": 0.9})
    labeller = PreferBackend("labeller")
    classifier = docrooter.Classifier(builtin_taxonomy("section_types/research_paper"), labeller)
    sources = [_paper(tmp_path, "a.pdf"), _paper(tmp_path, "b.pdf")]
    results = list(process_many(sources, profile="fast", doc_type=detector, sections=classifier))
    assert len(results) == 2
    for analysis in results:
        assert analysis.doc_type.best.node_ids[-1] == "legal.contract"
        assert analysis.sections.taxonomy == "section_types/contract"
        assert {n.canonical for n in analysis.sections.sections.values()} <= set(
            builtin_taxonomy("section_types/contract").nodes
        )
        (warning,) = [r for r in _routes(analysis, "sections") if "warning" in r["reason"]]
        assert "section_types/contract" in warning["reason"]
        assert "section_types/research_paper" in warning["reason"]
    assert labeller.section_calls("section_types/contract")  # its backend labelled them
    assert labeller.closed == 0  # the caller's backend stays open

    # the same taxonomy: the Classifier is used as it is, with no warning
    detector = PreferBackend("detector", {"research.research_paper": 0.9})
    analysis = process(sources[0], profile="fast", doc_type=detector, sections=classifier)
    assert analysis.sections.taxonomy == "section_types/research_paper"
    assert not [r for r in _routes(analysis, "sections") if "warning" in r["reason"]]


def test_bad_instances_are_type_errors(tmp_path):
    with pytest.raises(TypeError, match="ocr must be"):
        process(_paper(tmp_path), ocr=object())
    with pytest.raises(TypeError, match="escalate must be"):
        process(_paper(tmp_path), escalate=docrooter.Classifier(builtin_taxonomy("doc_types")))


# --- configuration through process ---------------------------------------------------------


def test_config_file_and_object(tmp_path):
    path = tmp_path / "router.toml"
    path.write_text('profile = "fast"\ndoc_type = "keyword"\nsections = "none"\n',
                    encoding="utf-8")  # fmt: skip
    paper = _paper(tmp_path)
    from_file = process(paper, config=path)
    assert from_file.config == RouterConfig.from_file(path)
    overridden = process(paper, config=from_file.config, doc_type="none")
    assert overridden.doc_type is None and overridden.config.profile == "fast"
    with pytest.raises(TypeError, match="not both"):
        process(paper, config=from_file.config, profile="fast")
    with pytest.raises(ValidationError, match="bogus"):
        process(paper, profile="fast", bogus=True)


def _fake_embedding_kind(registries):
    made = []

    def factory(model="", *, revision=None, **kwargs):
        made.append(FakeEmbedding())
        return made[-1]

    backend_registry.register_backend_kind("fake-embed", factory)
    return made


def test_embeddings_task_embeds_every_element(registries):
    made = _fake_embedding_kind(registries)
    seen = []
    analysis = process(MARKDOWN, doc_type="none", sections="none", embeddings="fake-embed",
                       on_event=seen.append)  # fmt: skip
    doc = analysis.document
    result = analysis.embeddings
    assert isinstance(result, EmbeddingSet)
    assert result.ids == [
        e.id
        for e in doc.elements
        if e.kind not in ("page_header", "page_footer", "page_number", "contents")
    ]
    assert (result.model, result.revision, result.dim, result.dtype) == (
        "fake/embed", "0123abcd", 4, "float16",
    )  # fmt: skip
    assert result.vectors.shape == (len(result.ids), 4)
    assert analysis.config.embeddings == "fake-embed" and "embeddings" in analysis.timings
    assert analysis.timings["total"] >= analysis.timings["embeddings"]
    assert len(made) == 1 and made[0].closed == 1  # created from its name, closed afterwards
    assert made[0].calls[0][1] == ""
    (route,) = _routes(analysis, "embeddings")
    assert route["engine"] == "fake-embed" and route["reason"] == "embeddings=fake-embed"
    assert any(e.stage == "embed" and e.status == "end" for e in seen)
    # the vectors are in the JSON only when asked for
    assert "embeddings" not in json.loads(analysis.to_json())
    asked = Analysis.model_validate_json(analysis.to_json(include_embeddings=True))
    assert asked.embeddings == result


def test_embeddings_of_a_pdf_follow_its_elements(tmp_path, registries):
    made = _fake_embedding_kind(registries)
    analysis = process(_paper(tmp_path), doc_type="none", sections="none", embeddings="fake-embed")
    doc = analysis.document
    assert analysis.embeddings.ids == [e.id for e in doc.elements]
    texts = made[0].calls[0][0]
    assert len(texts) == len(doc.elements)
    paragraph = next(t for e, t in zip(doc.elements, texts, strict=True) if e.kind == "paragraph")
    path, _, body = paragraph.partition("\n\n")
    assert path.endswith((" > Introduction", " > Methods", " > Results")) and "sentence" in body


def test_process_many_builds_the_embedding_backend_once(tmp_path, registries):
    made = _fake_embedding_kind(registries)
    sources = [MARKDOWN, MARKDOWN + "\n## Outlook\n\nThings may improve. Or not.\n"]
    results = process_many(sources, doc_type="none", sections="none", embeddings="fake-embed")
    first = next(results)
    assert len(made) == 1 and made[0].closed == 0
    rest = list(results)
    assert len(rest) == 1 and len(made) == 1 and made[0].closed == 1
    assert len(made[0].calls) == 2
    assert first.embeddings.ids != rest[0].embeddings.ids
    with contextlib.closing(process_many(sources, doc_type="none", sections="none",
                                         embeddings="fake-embed")) as early:  # fmt: skip
        next(early)
    assert len(made) == 2 and made[1].closed == 1


def test_embeddings_need_a_backend_that_encodes(registries):
    with pytest.raises(BackendError, match="cannot embed"):
        process(MARKDOWN, doc_type="none", sections="none", embeddings="keyword")


def test_embeddings_need_element_mode():
    with pytest.raises(ValueError, match="element mode.*layout"):
        process(MARKDOWN, profile="fast", embeddings="embedding:some/model")
    with pytest.raises(ValueError, match="element mode"):
        process_many([MARKDOWN], profile="fast", embeddings="embedding:some/model")


def test_analysis_json(tmp_path):
    analysis = process(_paper(tmp_path), doc_type="keyword", sections="keyword")
    data = json.loads(analysis.to_json())
    assert set(data) == {"document", "doc_type", "sections", "summaries", "config", "timings"}
    assert data["config"]["profile"] == "balanced"
    assert "embeddings" in json.loads(analysis.to_json(include_embeddings=True))
    assert Analysis.model_validate_json(analysis.to_json()).document == analysis.document


# --- the tables switch -------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ["pymupdf", "pdfplumber"])
def test_tables_none_never_calls_table_finder(engine, monkeypatch):
    if engine == "pdfplumber":
        page_class = pytest.importorskip("pdfplumber.page").Page
    else:
        page_class = pymupdf.Page
    calls = []
    real = page_class.find_tables

    def spy(self, *args, **kwargs):
        calls.append(1)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(page_class, "find_tables", spy)
    tables = GOLDEN / "tables.pdf"
    with_tables = docrooter.load_pdf(tables, engine=engine, elements=True)
    assert calls and any(e.kind == "table" for e in with_tables.elements)

    calls.clear()
    analysis = process(tables, text=engine, tables="none", doc_type="none", sections="none")
    assert calls == []
    assert not any(e.kind == "table" for e in analysis.document.elements)
    assert analysis.config.tables == "none"
    docrooter.load_pdf(tables, engine=engine, elements=True, tables="none")
    assert calls == []


def test_tables_option_needs_element_mode():
    with pytest.raises(TypeError, match="only valid with elements=True"):
        docrooter.load_pdf(GOLDEN / "tables.pdf", tables="none")
    with pytest.raises(ValueError, match="tables must be"):
        docrooter.load_pdf(GOLDEN / "tables.pdf", elements=True, tables="text")  # type: ignore[arg-type]
