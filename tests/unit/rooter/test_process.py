"""``process()`` / ``process_many()`` (spec §7): the fast path, element mode, the route log,
escalation, engine reuse and ownership, and the tables switch. Keyword and fake backends, fake
OCR and layout engines; no model is ever loaded."""

import contextlib
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import docrooter
from docrooter import classify_document, load, normalize_sections
from docrooter.backends import Scores
from docrooter.backends import registry as backend_registry
from docrooter.classify.normalize import GENERIC_TAXONOMY
from docrooter.errors import DocrooterError, MissingExtraError
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


def test_embeddings_task_not_available_yet(tmp_path):
    with pytest.raises(NotImplementedError, match="embeddings"):
        process(_paper(tmp_path), embeddings="embedding:some/model")


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
