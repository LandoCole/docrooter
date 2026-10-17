"""Pipeline events (``on_event``): stage order, start/end pairing, timing, progress, errors,
failing sinks and the no-sink path. Torch-free: keyword backend and generated PDFs."""

import asyncio
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import docrooter
from docrooter import Document, Taxonomy, TaxonomyNode, events
from docrooter.backends import KeywordBackend, loading
from docrooter.classify import Classifier, classify_document, normalize_sections
from docrooter.classify import normalize as normalize_module
from docrooter.errors import DocrooterError
from docrooter.events import Event, pipeline_run
from docrooter.pdf.load import load_pdf

pymupdf = pytest.importorskip("pymupdf")

BODY = "This sentence is ordinary body text that stays small and keeps the body size at eleven."


def _heading_page(doc, heading, body=BODY):
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 90), heading, fontsize=22, fontname="hebo")
    y = 130
    for k in range(4):
        page.insert_text((72, y), f"{body} Line {k}.", fontsize=11, fontname="helv")
        y += 15


def _pdf(tmp_path, pages=3, name="doc.pdf"):
    doc = pymupdf.open()
    for n in range(pages):
        _heading_page(doc, f"Chapter {n + 1}")
    path = tmp_path / name
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def pdf_path(tmp_path):
    return _pdf(tmp_path)


class Recorder:
    def __init__(self):
        self.events: list[Event] = []

    def __call__(self, event: Event) -> None:
        self.events.append(event)

    def stages(self, status=None):
        return [e.stage for e in self.events if status is None or e.status == status]


def doc_type_taxonomy() -> Taxonomy:
    return Taxonomy(
        [
            TaxonomyNode(
                id="legal.contract",
                label="Contract",
                description="agreement parties termination indemnification governing law",
            ),
            TaxonomyNode(
                id="research.paper",
                label="Research paper",
                description="abstract method experiments results references",
            ),
        ]
    )


def paper() -> Document:
    return Document.from_sections(
        [
            {"title": "Abstract", "level": 1, "text": "We present results."},
            {"title": "Method", "level": 1, "text": "Our method and experiments."},
            {"title": "References", "level": 1, "text": "Bibliography of cited works."},
        ],
        source="paper.md",
    )


def assert_well_formed(recorder: Recorder) -> None:
    """Every start has a matching end, in nesting order; ``t`` never goes back; one terminal."""
    open_stages: list[str] = []
    last_t = -1.0
    for event in recorder.events:
        assert event.t >= last_t >= -1.0
        last_t = event.t
        if event.status == "start":
            open_stages.append(event.stage)
        elif event.status == "end" and event.stage != "done":
            assert event.stage in open_stages, event
            open_stages.remove(event.stage)
        elif event.status == "progress":
            assert event.stage in open_stages, event
    assert not open_stages
    assert recorder.events[-1].stage == "done" and recorder.events[-1].status == "end"
    assert sum(e.stage == "done" for e in recorder.events) == 1
    json.dumps([e.model_dump() for e in recorder.events])  # JSON-safe


# --- load_pdf -------------------------------------------------------------------------------


def test_load_pdf_stage_order_and_data(pdf_path):
    rec = Recorder()
    doc = load_pdf(pdf_path, engine="pymupdf", on_event=rec)
    assert [(e.stage, e.status) for e in rec.events] == [
        ("extract", "start"),
        ("extract", "end"),
        ("signals", "start"),
        ("signals", "end"),
        ("headings", "start"),
        ("headings", "end"),
        ("assemble", "start"),
        ("assemble", "end"),
        ("done", "end"),
    ]
    assert_well_formed(rec)
    ends = {e.stage: e.data for e in rec.events if e.status == "end"}
    assert ends["extract"]["pages"] == 3 and ends["extract"]["lines"] > 0
    assert ends["headings"]["headings"] == 3
    assert ends["headings"]["levels"] == {"1": 3}
    assert ends["assemble"] == {"sections": len(doc.sections)}
    assert rec.events[0].data == {"engine": "pymupdf"}


def test_t_starts_near_zero_and_grows(pdf_path):
    rec = Recorder()
    load_pdf(pdf_path, on_event=rec)
    assert 0 <= rec.events[0].t < 1
    assert rec.events[-1].t > rec.events[0].t


def test_extract_progress_every_ten_pages(tmp_path):
    path = _pdf(tmp_path, pages=25, name="long.pdf")
    rec = Recorder()
    load_pdf(path, engine="pymupdf", on_event=rec)
    progress = [e for e in rec.events if e.status == "progress"]
    assert [e.data for e in progress] == [{"pages": 25, "page": 10}, {"pages": 25, "page": 20}]
    assert {e.stage for e in progress} == {"extract"}
    kinds = [(e.stage, e.status) for e in rec.events[:4]]
    assert kinds == [
        ("extract", "start"),
        ("extract", "progress"),
        ("extract", "progress"),
        ("extract", "end"),
    ]
    assert_well_formed(rec)


def test_extract_progress_counts_pages_read_with_max_pages(tmp_path):
    path = _pdf(tmp_path, pages=25, name="long.pdf")
    rec = Recorder()
    load_pdf(path, engine="pymupdf", max_pages=12, on_event=rec)
    progress = [e.data for e in rec.events if e.status == "progress"]
    assert progress == [{"pages": 12, "page": 10}]


def test_pdfplumber_engine_reports_progress_too(tmp_path):
    pytest.importorskip("pdfplumber")
    path = _pdf(tmp_path, pages=11, name="plumb.pdf")
    rec = Recorder()
    load_pdf(path, engine="pdfplumber", on_event=rec)
    assert [e.data for e in rec.events if e.status == "progress"] == [{"pages": 11, "page": 10}]
    assert_well_formed(rec)


def test_both_engines_report_each_engine_then_the_merge(pdf_path):
    pytest.importorskip("pdfplumber")
    rec = Recorder()
    load_pdf(pdf_path, engine="both", on_event=rec)
    assert_well_formed(rec)
    engines = [e.data.get("engine") for e in rec.events if e.status == "start"]
    assert engines == ["pymupdf"] * 3 + ["pdfplumber"] * 3 + ["both", None]
    assert rec.stages("start")[6] == "headings"


def test_markdown_load_emits_assemble_and_done():
    rec = Recorder()
    doc = docrooter.load("# Title\n\nBody text.\n\n## Sub\n\nMore.\n", on_event=rec)
    assert [(e.stage, e.status) for e in rec.events] == [
        ("assemble", "start"),
        ("assemble", "end"),
        ("done", "end"),
    ]
    assert rec.events[1].data == {"sections": len(doc.sections)}


def test_load_passes_on_event_to_load_pdf(pdf_path):
    rec = Recorder()
    docrooter.load(pdf_path, engine="pymupdf", on_event=rec)
    assert rec.stages("start")[:4] == ["extract", "signals", "headings", "assemble"]
    assert_well_formed(rec)


# --- classify / normalize -------------------------------------------------------------------


def test_classify_document_events():
    rec = Recorder()
    result = classify_document(paper(), doc_type_taxonomy(), backend="keyword", on_event=rec)
    assert [(e.stage, e.status) for e in rec.events] == [
        ("view", "start"),
        ("view", "end"),
        ("classify", "start"),
        ("classify", "end"),
        ("done", "end"),
    ]
    assert_well_formed(rec)
    view_end, classify_end = rec.events[1].data, rec.events[3].data
    assert view_end["tokens"] > 0
    assert classify_end["type"] == result.best.node_ids[-1] == "research.paper"
    assert classify_end["confidence"] == pytest.approx(result.confidence)


def test_classifier_classify_documents_events():
    rec = Recorder()
    classifier = Classifier(doc_type_taxonomy(), KeywordBackend())
    classifier.classify_documents([paper(), paper()], on_event=rec)
    assert rec.stages("start") == ["view", "classify"]
    assert rec.events[3].data == {"documents": 2}  # no single type for several documents
    assert_well_formed(rec)


def test_normalize_sections_with_known_doc_type():
    rec = Recorder()
    normalize_sections(
        paper(), doc_type="research.research_paper", backend=KeywordBackend(), on_event=rec
    )
    assert [(e.stage, e.status) for e in rec.events] == [
        ("normalize", "start"),
        ("normalize", "end"),
        ("done", "end"),
    ]
    assert rec.events[1].data == {"sections": 3}


def test_normalize_sections_classifies_the_doc_type_first():
    rec = Recorder()
    normalize_sections(paper(), backend=KeywordBackend(), on_event=rec)
    assert rec.stages("start") == ["view", "classify", "normalize"]
    assert_well_formed(rec)


def test_shared_run_has_one_clock_and_one_done(pdf_path):
    rec = Recorder()
    with pipeline_run(rec):
        doc = load_pdf(pdf_path, engine="pymupdf", on_event=rec)
        classify_document(doc, doc_type_taxonomy(), backend="keyword", on_event=rec)
        normalize_sections(doc, doc_type="research.research_paper", backend="keyword")
    assert_well_formed(rec)
    assert rec.stages("start") == [
        "extract",
        "signals",
        "headings",
        "assemble",  # fmt: skip
        "view",
        "classify",
        "normalize",
    ]


def test_inner_run_with_another_sink_reuses_the_clock(pdf_path):
    outer, inner = Recorder(), Recorder()
    with pipeline_run(outer):
        outer.events.clear()
        events.start("upload").end()
        time.sleep(0.05)
        load_pdf(pdf_path, engine="pymupdf", on_event=inner)
    assert outer.stages() == ["upload", "upload", "done"]
    assert inner.events[0].t >= 0.05  # continued the outer clock, not restarted at 0
    assert_well_formed(inner)


def test_model_load_events_wrap_a_backend_load():
    rec = Recorder()
    with pipeline_run(rec):
        device = loading.prepare_model_load("cpu", "embedding:test")
        loading.after_model_load(device, "embedding:test")
    assert [(e.stage, e.status, e.data) for e in rec.events[:2]] == [
        ("model_load", "start", {"backend": "embedding:test", "device": "cpu"}),
        ("model_load", "end", {"backend": "embedding:test", "device": "cpu"}),
    ]
    assert_well_formed(rec)


def test_model_load_without_a_run_reports_nothing():
    device = loading.prepare_model_load("cpu", "x")
    loading.after_model_load(device, "x")  # must not raise


# --- errors ---------------------------------------------------------------------------------


def test_error_event_then_reraise(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"this is not a pdf")
    rec = Recorder()
    with pytest.raises(DocrooterError) as caught:
        load_pdf(bad, engine="pymupdf", on_event=rec)
    last = rec.events[-1]
    assert (last.stage, last.status) == ("error", "error")
    assert last.data["message"] == str(caught.value)[:500]
    assert last.data["stage"] == "extract"
    assert "done" not in rec.stages()


def test_error_message_is_capped():
    rec = Recorder()
    with pytest.raises(ValueError, match="mode must be"):
        classify_document(
            paper(), doc_type_taxonomy(), backend="keyword", mode="x" * 900, on_event=rec
        )
    assert [e.status for e in rec.events] == ["error"]
    assert len(rec.events[0].data["message"]) == 500


def test_nested_calls_report_a_single_error(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"nope")
    rec = Recorder()
    with pytest.raises(DocrooterError), pipeline_run(rec):
        docrooter.load(bad, engine="pymupdf", on_event=rec)
    assert rec.stages("error") == ["error"]


# --- failing sinks --------------------------------------------------------------------------


def test_raising_sink_never_breaks_processing_and_is_disabled(pdf_path, caplog):
    calls: list[Event] = []

    def sink(event: Event) -> None:
        calls.append(event)
        raise RuntimeError("sink is broken")

    expected = load_pdf(pdf_path, engine="pymupdf")
    with caplog.at_level(logging.WARNING, logger="docrooter.events"):
        doc = load_pdf(pdf_path, engine="pymupdf", on_event=sink)
    assert len(calls) == 1  # disabled after the first failure
    assert len([r for r in caplog.records if r.name == "docrooter.events"]) == 1
    assert doc.toc() == expected.toc()


def test_sink_failing_late_keeps_earlier_events(pdf_path):
    seen: list[str] = []

    def sink(event: Event) -> None:
        seen.append(f"{event.stage}:{event.status}")
        if event.stage == "signals":
            raise RuntimeError("boom")

    load_pdf(pdf_path, engine="pymupdf", on_event=sink)
    assert seen == ["extract:start", "extract:end", "signals:start"]


# --- no sink --------------------------------------------------------------------------------


def _frozen(doc: Document) -> dict:
    data = doc.model_dump()
    data["meta"] = {k: v for k, v in data["meta"].items() if k != "timings"}
    return data


def test_no_sink_gives_identical_results(pdf_path):
    plain = load_pdf(pdf_path, engine="pymupdf")
    observed = load_pdf(pdf_path, engine="pymupdf", on_event=Recorder())
    assert _frozen(plain) == _frozen(observed)

    taxonomy = doc_type_taxonomy()
    a = classify_document(plain, taxonomy, backend="keyword")
    b = classify_document(plain, taxonomy, backend="keyword", on_event=Recorder())
    a_data, b_data = a.model_dump(), b.model_dump()
    a_data.pop("timings"), b_data.pop("timings")
    assert a_data == b_data

    n1 = normalize_sections(paper(), doc_type="research.research_paper", backend="keyword")
    n2 = normalize_sections(
        paper(), doc_type="research.research_paper", backend="keyword", on_event=Recorder()
    )
    assert n1 == n2


def test_without_a_sink_no_event_is_ever_built(pdf_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("an Event was built without a sink")

    monkeypatch.setattr(events, "Event", forbidden)
    doc = load_pdf(pdf_path, engine="pymupdf")
    classify_document(doc, doc_type_taxonomy(), backend="keyword")
    normalize_sections(paper(), doc_type="research.research_paper", backend="keyword")
    docrooter.load("# A\n\ntext\n")
    assert events._current.get() is None


def test_run_state_is_restored_after_a_call(pdf_path):
    load_pdf(pdf_path, engine="pymupdf", on_event=Recorder())
    assert events._current.get() is None
    with pytest.raises(DocrooterError):
        load_pdf(pdf_path, engine="bogus", on_event=Recorder())
    assert events._current.get() is None


# --- fix round 1: bound-method sinks, threads, interrupts, aborted stages -----------------------


class Job:
    """A sink that is a bound method: ``job.push`` is a new object on every access."""

    def __init__(self):
        self.events: list[Event] = []

    def push(self, event: Event) -> None:
        self.events.append(event)


def test_bound_method_sink_joins_the_enclosing_run(pdf_path):
    job = Job()
    assert job.push is not job.push
    with pipeline_run(job.push):
        doc = load_pdf(pdf_path, engine="pymupdf", on_event=job.push)
        classify_document(doc, doc_type_taxonomy(), backend="keyword", on_event=job.push)
    rec = Recorder()
    rec.events = job.events
    assert_well_formed(rec)  # exactly one done, every start ended
    assert rec.stages("start") == [
        "extract",
        "signals",
        "headings",
        "assemble",  # fmt: skip
        "view",
        "classify",
    ]


def test_bound_method_sink_reports_a_nested_error_once(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"nope")
    job = Job()
    with pytest.raises(DocrooterError), pipeline_run(job.push):
        load_pdf(bad, engine="pymupdf", on_event=job.push)
    assert [e.stage for e in job.events].count("error") == 1


def test_concurrent_runs_in_threads_only_see_their_own_events(tmp_path):
    paths = [_pdf(tmp_path, pages=pages, name=f"t{pages}.pdf") for pages in (2, 5)]
    recorders = [Recorder(), Recorder()]
    barrier = threading.Barrier(2)

    def work(path, rec):
        barrier.wait()
        for _ in range(3):
            load_pdf(path, engine="pymupdf", on_event=rec)

    threads = [threading.Thread(target=work, args=a) for a in zip(paths, recorders, strict=True)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for rec, pages in zip(recorders, (2, 5), strict=True):
        assert rec.stages().count("done") == 3
        assert {
            e.data["pages"] for e in rec.events if e.stage == "extract" and e.status == "end"
        } == {pages}


def test_run_entered_inside_a_worker_thread_works_end_to_end(pdf_path):
    rec = Recorder()

    def work():
        with pipeline_run(rec):
            doc = load_pdf(pdf_path, engine="pymupdf")
            classify_document(doc, doc_type_taxonomy(), backend="keyword")

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(work).result()
    assert_well_formed(rec)
    assert rec.stages("start")[-2:] == ["view", "classify"]


def test_the_run_does_not_cross_into_an_executor_thread(pdf_path):
    rec = Recorder()
    with pipeline_run(rec), ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(load_pdf, pdf_path, engine="pymupdf").result()
    assert rec.stages() == ["done"]  # documented: enter the run inside the worker


def test_asyncio_to_thread_carries_the_run(pdf_path):
    rec = Recorder()

    async def main():
        with pipeline_run(rec):
            await asyncio.to_thread(load_pdf, pdf_path, engine="pymupdf")

    asyncio.run(main())
    assert rec.stages("start") == ["extract", "signals", "headings", "assemble"]
    assert_well_formed(rec)


def test_keyboard_interrupt_emits_error_and_propagates():
    rec = Recorder()
    with pytest.raises(KeyboardInterrupt), pipeline_run(rec):
        events.start("extract")
        raise KeyboardInterrupt("stop")
    last = rec.events[-1]
    assert (last.stage, last.status) == ("error", "error")
    assert last.data == {"message": "stop", "stage": "extract"}


def test_cancelled_error_is_marked_cancelled():
    rec = Recorder()
    with pytest.raises(asyncio.CancelledError), pipeline_run(rec):
        raise asyncio.CancelledError()
    assert rec.events[-1].data == {"message": "", "cancelled": True}


def test_open_stages_are_ended_as_aborted_before_done():
    rec = Recorder()
    with pipeline_run(rec):
        events.start("model_load")
        events.start("classify")
    assert [(e.stage, e.status, e.data) for e in rec.events] == [
        ("model_load", "start", {}),
        ("classify", "start", {}),
        ("classify", "end", {"aborted": True}),
        ("model_load", "end", {"aborted": True}),
        ("done", "end", {}),
    ]
    assert_well_formed(rec)


def test_normalize_closes_the_backends_it_loaded_before_done(monkeypatch):
    rec = Recorder()
    seen: list[list[str]] = []
    real = normalize_module.close_backend

    def spy(item):
        seen.append(rec.stages())
        real(item)

    monkeypatch.setattr(normalize_module, "close_backend", spy)
    normalize_sections(paper(), backend="keyword", on_event=rec)
    assert seen and all("done" not in stages for stages in seen)
    assert rec.stages()[-1] == "done"


def test_classify_documents_loads_the_model_before_the_view_as_documented():
    """The view counts tokens with the backend's tokenizer, so a model loads before ``view``
    starts; the docstrings must give that order."""

    class LoadingBackend(KeywordBackend):
        def tokenizer(self):
            device = loading.prepare_model_load("cpu", "loading:test")
            loading.after_model_load(device, "loading:test")
            return lambda text: len(text.split())

    rec = Recorder()
    classify_document(paper(), doc_type_taxonomy(), backend=LoadingBackend(), on_event=rec)
    assert rec.stages("start") == ["model_load", "view", "classify"]
    assert_well_formed(rec)
    for doc in (classify_document.__doc__, Classifier.classify_documents.__doc__):
        events_part = doc[doc.index("``on_event``") :]
        order = [events_part.index(f"``{stage}``") for stage in ("model_load", "view", "classify")]
        assert order == sorted(order)
