"""The corpus builder (spec §5): page selection, element extraction through ``process`` with
the profile's layout engine (a fake one here, registered as ``heron``), furniture and contents
skipped, and resumable, per-document checkpoints. Generated PDFs; no model is loaded."""

import hashlib
import json
from collections import Counter

import pytest
from typer.testing import CliRunner

from docrooter.cli import app
from docrooter.corpus import Corpus, ManifestRecord, build_corpus, corpus_elements, select_pages
from docrooter.corpus import build as build_module
from docrooter.document.models import Document
from docrooter.elements.models import Box, Element, TableData
from docrooter.layout import registry as layout_registry
from docrooter.ocr import registry as ocr_registry
from docrooter.pdf.load import load_pdf
from docrooter.summarize import heading_titles

pymupdf = pytest.importorskip("pymupdf")

SKIPPED = {"page_header", "page_footer", "page_number", "contents"}


class FakeLayout:
    """A layout engine that finds no regions and records the pages it was asked about."""

    name = "fake-heron"
    instances = []

    def __init__(self):
        self.pages = []
        self.closed = 0
        FakeLayout.instances.append(self)

    def detect(self, path, pages, *, dpi=None):
        self.pages.append(list(pages))
        return []

    def close(self):
        self.closed += 1


@pytest.fixture(autouse=True)
def _fake_engines(monkeypatch):
    """``accurate``'s ``heron`` is the fake layout engine; ``ocr="auto"`` finds no engine."""
    monkeypatch.setattr(layout_registry, "_FACTORIES", {"heron": FakeLayout})
    monkeypatch.setattr(ocr_registry, "resolve_auto_ocr", lambda: None)
    FakeLayout.instances = []


def _report_pdf(path, n_pages):
    """A report: a running header and a page number on every page, a contents page (2), and on
    every other page a section heading, a sub-heading and a paragraph naming the page."""
    doc = pymupdf.open()
    for k in range(1, n_pages + 1):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 40), "Widget Corporation Annual Report", fontsize=9, fontname="helv")
        if k == 2:
            page.insert_text((72, 100), "Contents", fontsize=18, fontname="hebo")
            for j, title in enumerate(["Overview", "Strategy", "Risks", "Results", "Outlook"]):
                entry = f"{title} " + "." * 40 + f" {j + 3}"
                page.insert_text((72, 130 + 18 * j), entry, fontsize=11, fontname="helv")
        else:
            page.insert_text((72, 100), f"Section {k}", fontsize=18, fontname="hebo")
            page.insert_text((72, 130), f"Detail {k}", fontsize=13, fontname="hebo")
            for j in range(3):
                text = f"Page {k} sentence {j} describes the widget business in plain words."
                page.insert_text((72, 155 + 15 * j), text, fontsize=11, fontname="helv")
        page.insert_text((300, 770), str(k), fontsize=9, fontname="helv")
    doc.save(str(path))
    doc.close()
    return path


def _record(pdf, doc_id, split="train"):
    data = pdf.read_bytes()
    return ManifestRecord(
        id=doc_id,
        url=f"https://example.org/{doc_id}.pdf",
        source="test",
        licence="public-domain",
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
        doc_type_hint="financial.annual_report",
        family_hint="annual_report",
        split=split,
    )


@pytest.fixture
def pdfs(tmp_path):
    """A 24-page and a 4-page report in ``tmp_path/pdfs``, with their records."""
    folder = tmp_path / "pdfs"
    folder.mkdir()
    long = _report_pdf(folder / "long.pdf", 24)
    short = _report_pdf(folder / "short.pdf", 4)
    return folder, [_record(long, "long"), _record(short, "short", "gold")]


def _doc(corpus_dir, doc_id):
    return json.loads((corpus_dir / "docs" / f"{doc_id}.json").read_text())


def test_select_pages():
    assert select_pages(5, first_pages=10, max_pages=30, seed=0, doc_id="a") == [1, 2, 3, 4, 5]
    pages = select_pages(100, first_pages=10, max_pages=30, seed=0, doc_id="a")
    assert pages[:10] == list(range(1, 11)) and len(pages) == 30 and pages == sorted(set(pages))
    assert all(11 <= p <= 100 for p in pages[10:])
    assert pages == select_pages(100, first_pages=10, max_pages=30, seed=0, doc_id="a")
    assert pages != select_pages(100, first_pages=10, max_pages=30, seed=0, doc_id="b")
    assert pages != select_pages(100, first_pages=10, max_pages=30, seed=1, doc_id="a")
    assert select_pages(100, first_pages=30, max_pages=30, seed=0, doc_id="a") == list(range(1, 31))
    sampled = select_pages(100, first_pages=0, max_pages=5, seed=0, doc_id="a")
    assert len(sampled) == 5 and sampled != [1, 2, 3, 4, 5]
    assert select_pages(0, first_pages=10, max_pages=30, seed=0, doc_id="a") == []
    with pytest.raises(ValueError, match="first_pages"):
        select_pages(10, first_pages=31, max_pages=30, seed=0, doc_id="a")
    with pytest.raises(ValueError, match="max_pages"):
        select_pages(10, first_pages=0, max_pages=0, seed=0, doc_id="a")


def test_corpus_elements_of_a_document():
    doc = Document.from_sections(
        [{"title": "Preamble", "level": 0}, {"title": "Report", "level": 1},
         {"title": "Risks", "level": 2}]
    )  # fmt: skip
    box = [Box(page=3, x0=0, y0=0, x1=1, y1=1)]
    table = TableData(rows=[["Year", "Sales"], ["2025", "10"]], header_rows=1)
    doc.elements = [
        Element(id="e0", kind="paragraph", text="Before.", boxes=[], order=0, section_id="s0"),
        Element(id="e1", kind="page_header", text="ACME", boxes=box, order=1, section_id="s0"),
        Element(id="e2", kind="heading", text="Risks", boxes=box, order=2, section_id="s1.1"),
        Element(id="e3", kind="table", text="Year Sales 2025 10", boxes=box, order=3,
                section_id="s1.1", table=table),
        Element(id="e4", kind="contents", text="Risks ... 3", boxes=box, order=4, section_id=None),
    ]  # fmt: skip
    got = corpus_elements("d", doc)
    assert [e.element_id for e in got] == ["e0", "e2", "e3"]
    assert (got[0].heading_path, got[0].section_title, got[0].page) == ([], None, None)
    assert got[1].heading_path == ["Report", "Risks"] and got[1].section_title == "Risks"
    assert got[2].table_markdown == table.to_markdown() and got[2].page == 3
    assert got[0].table_markdown is None and got[2].text == "Year Sales 2025 10"


def test_build_reads_first_and_sampled_pages_deterministically(tmp_path, pdfs):
    folder, records = pdfs
    corpus = Corpus(tmp_path / "c")
    report = build_corpus(corpus, folder, records=records, max_pages=8, first_pages=3, seed=0)
    assert report.added == ["long", "short"] and report.failed == {}

    long, short = _doc(tmp_path / "c", "long"), _doc(tmp_path / "c", "short")
    expected = select_pages(24, first_pages=3, max_pages=8, seed=0, doc_id="long")
    assert long["pages_read"] == expected and expected[:3] == [1, 2, 3] and len(expected) == 8
    assert short["pages_read"] == [1, 2, 3, 4]
    assert report.pages_read == 12
    # Only the chosen pages were read: by the layout engine, made once for the whole build ...
    (engine,) = FakeLayout.instances
    assert engine.pages == [expected, [1, 2, 3, 4]] and engine.closed == 1
    # ... and in the elements, which come from every chosen page.
    pages = {e.page for e in corpus.iter_elements() if e.doc_id == "long"}
    assert pages == set(expected)
    assert {e.text for e in corpus.iter_elements() if e.doc_id == "long" and e.page == expected[-1]}

    # The same inputs give the same corpus, byte for byte; another seed samples other pages.
    again = Corpus(tmp_path / "again")
    build_corpus(again, folder, records=records, max_pages=8, first_pages=3, seed=0)
    for name in ("long", "short"):
        path = f"docs/{name}.json"
        assert (tmp_path / "again" / path).read_bytes() == (tmp_path / "c" / path).read_bytes()
    other = Corpus(tmp_path / "other")
    build_corpus(other, folder, records=records, max_pages=8, first_pages=3, seed=1)
    assert _doc(tmp_path / "other", "long")["pages_read"] != expected
    assert _doc(tmp_path / "other", "short")["pages_read"] == [1, 2, 3, 4]


def test_build_skips_furniture_and_contents(tmp_path, pdfs):
    folder, records = pdfs
    corpus = Corpus(tmp_path / "c")
    build_corpus(corpus, folder, records=records[1:])
    stored = list(corpus.iter_elements())
    loaded = load_pdf(folder / "short.pdf", elements=True)
    assert {"page_header", "page_number", "contents"} <= {e.kind for e in loaded.elements}
    assert not SKIPPED & {e.kind for e in stored}
    kept = [e for e in loaded.elements if e.kind not in SKIPPED]
    assert [e.element_id for e in stored] == [e.id for e in kept]  # ids as the rooter gives them
    assert Counter(e.kind for e in stored) == Counter(e.kind for e in kept)

    by_id = {e.id: e for e in loaded.elements}
    for element in stored:
        source = by_id[element.element_id]
        assert element.text == source.text
        assert element.heading_path == heading_titles(loaded, source.section_id)
        assert element.page == source.boxes[0].page
    (paragraph,) = [e for e in stored if e.text.startswith("Page 3 sentence 0")]
    assert paragraph.heading_path == ["Section 3", "Detail 3"]
    assert paragraph.section_title == "Detail 3"
    doc = _doc(tmp_path / "c", "short")
    assert doc["doc_type_hint"] == "financial.annual_report" and doc["id"] == "short"


def test_build_resumes_and_records_what_it_could_not_read(tmp_path, pdfs):
    folder, records = pdfs
    corrupt = folder / "corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.7\nnot really a pdf")
    swapped = folder / "swapped.pdf"
    swapped.write_bytes((folder / "short.pdf").read_bytes())
    elsewhere = _report_pdf(tmp_path / "elsewhere.pdf", 3)
    extra = [
        _record(corrupt, "corrupt"),
        _record(elsewhere, "missing"),  # no missing.pdf in the folder
        _record(folder / "long.pdf", "swapped"),  # swapped.pdf holds other bytes
    ]
    corpus = Corpus(tmp_path / "c")
    seen = []
    report = build_corpus(
        corpus,
        folder,
        records=[records[1], *extra],
        max_pages=6,
        first_pages=2,
        on_progress=lambda done, total, doc_id, status: seen.append((done, total, doc_id, status)),
    )
    assert report.added == ["short"] and report.missing == ["missing"]
    assert report.mismatched == ["swapped"] and list(report.failed) == ["corrupt"]
    assert report.failed["corrupt"].startswith("DocrooterError")
    assert seen == [
        (1, 4, "short", "reading"), (1, 4, "short", "added"),  # named before it is read
        (2, 4, "corrupt", "reading"), (2, 4, "corrupt", "failed"),
        (3, 4, "missing", "missing"), (4, 4, "swapped", "mismatched"),
    ]  # fmt: skip
    assert (tmp_path / "c" / "stats.json").exists()
    assert list(corpus.failures()) == ["corrupt"]  # persisted
    assert corpus.failures()["corrupt"]["reason"] == report.failed["corrupt"]
    assert corpus.pending_attempt() is None

    # A second run (after a crash, or with more records) does only the new work, and does not
    # read the document that failed again.
    reads = []

    def progress(done, total, doc_id, status):
        if status == "reading":
            reads.append(doc_id)

    report = build_corpus(
        corpus, folder, records=[*records, extra[0]], max_pages=6, first_pages=2,
        on_progress=progress,
    )  # fmt: skip
    assert report.added == ["long"] and report.skipped == ["short"]
    assert report.skipped_failed == ["corrupt"] and report.failed == {}
    assert reads == ["long"]
    report = build_corpus(corpus, folder, records=[extra[0]], retry_failed=True,
                          on_progress=progress)  # fmt: skip
    assert list(report.failed) == ["corrupt"] and reads == ["long", "corrupt"]
    assert [r.id for r in corpus.manifest()] == ["short", "long"]
    assert corpus.stats()["documents"] == 2 and corpus.stats()["failed"] == 1


def test_a_pdf_that_kills_the_build_is_not_read_again(tmp_path, pdfs, monkeypatch, caplog):
    """Review Focus 5: the document being read is noted durably before the read. A process
    killed mid-read (no Python cleanup) leaves the note; the next build records the document
    as failed and skips it, so one poison PDF cannot stop every resume."""
    folder, records = pdfs
    corpus = Corpus(tmp_path / "c")
    noted = []
    real_process = build_module.process

    def spy(source, **kwargs):
        noted.append(corpus.pending_attempt())
        return real_process(source, **kwargs)

    monkeypatch.setattr(build_module, "process", spy)
    build_corpus(corpus, folder, records=records[1:], max_pages=4, first_pages=2)
    assert noted == ["short"] and corpus.pending_attempt() is None  # noted, then cleared

    # The process dies while reading "long": its note is what a hard kill leaves behind.
    corpus.start_attempt("long")
    seen = []
    report = build_corpus(
        Corpus(tmp_path / "c"),
        folder,
        records=records,
        max_pages=4,
        first_pages=2,
        on_progress=lambda done, total, doc_id, status: seen.append((doc_id, status)),
    )
    assert report.crashed == ["long"] and report.skipped_failed == ["long"]
    assert seen == [("long", "skipped_failed"), ("short", "skipped")]
    assert Corpus(tmp_path / "c").failures()["long"]["reason"] == "crashed previous run"
    assert Corpus(tmp_path / "c").pending_attempt() is None
    assert "long was being read when the last build died" in caplog.text

    # Retrying reads it again (here it does not crash) and adds it.
    report = build_corpus(
        Corpus(tmp_path / "c"), folder, records=records, max_pages=4, first_pages=2,
        retry_failed=True,
    )  # fmt: skip
    assert report.added == ["long"] and report.crashed == []
    assert noted == ["short", "long"]

    # A note for a document that did get committed (killed after the manifest line) is dropped.
    corpus = Corpus(tmp_path / "c")
    corpus.start_attempt("short")
    report = build_corpus(corpus, folder, records=records, max_pages=4, first_pages=2)
    assert report.crashed == [] and corpus.pending_attempt() is None


@pytest.mark.parametrize("error", [KeyboardInterrupt, MemoryError])
def test_a_stopped_run_clears_its_note_and_memory_errors_are_recorded(
    tmp_path, pdfs, monkeypatch, error
):
    folder, records = pdfs
    corpus = Corpus(tmp_path / "c")

    def stop(source, **kwargs):
        raise error()

    monkeypatch.setattr(build_module, "process", stop)
    with pytest.raises(error):
        build_corpus(corpus, folder, records=records[1:], max_pages=4, first_pages=2)
    assert corpus.pending_attempt() is None
    expected = {"short": "MemoryError"} if error is MemoryError else {}
    assert {k: v["reason"] for k, v in corpus.failures().items()} == expected


def test_an_id_clash_is_reported_before_reading(tmp_path, pdfs, monkeypatch):
    folder, records = pdfs
    corpus = Corpus(tmp_path / "c")
    build_corpus(corpus, folder, records=records[1:], max_pages=4, first_pages=2)
    (folder / "SHORT.pdf").write_bytes((folder / "long.pdf").read_bytes())
    clash = _record(folder / "SHORT.pdf", "SHORT")
    monkeypatch.setattr(build_module, "process", lambda *a, **k: pytest.fail("read"))
    report = build_corpus(corpus, folder, records=[clash])
    assert list(report.failed) == ["SHORT"] and "differs only in case" in report.failed["SHORT"]


def test_missing_ocr_engine_is_warned_once(tmp_path, pdfs, caplog):
    folder, records = pdfs
    report = build_corpus(
        Corpus(tmp_path / "c"), folder, records=records, max_pages=4, first_pages=2
    )
    assert report.warnings == ["ocr_unavailable"]
    assert caplog.text.count("ocr=auto found no OCR engine") == 1


def _long_report_with_contents(path, n_pages=60):
    """Page 2 is a contents page whose page labels (on the same rows, no leaders) run to 57."""
    doc = pymupdf.open()
    titles = ["Overview", "Strategy", "Markets", "Risks", "Results", "Governance", "Outlook"]
    for k in range(1, n_pages + 1):
        page = doc.new_page(width=612, height=792)
        if k == 2:
            page.insert_text((72, 100), "Contents", fontsize=18, fontname="hebo")
            for j, title in enumerate(titles):
                page.insert_text((72, 140 + 20 * j), title, fontsize=11, fontname="helv")
                page.insert_text((520, 140 + 20 * j), str(3 + 9 * j), fontsize=11)
        else:
            page.insert_text((72, 100), f"Section {k}", fontsize=18, fontname="hebo")
            page.insert_text((72, 130), f"Page {k} text in plain words.", fontsize=11)
    doc.save(str(path))
    doc.close()
    return path


def test_build_skips_contents_on_sampled_long_documents(tmp_path):
    """Spec §5: printed contents are skipped even when their page labels point past the pages
    read (here 8 of 60)."""
    folder = tmp_path / "pdfs"
    folder.mkdir()
    pdf = _long_report_with_contents(folder / "long.pdf")
    corpus = Corpus(tmp_path / "c")
    build_corpus(corpus, folder, records=[_record(pdf, "long")], max_pages=8, first_pages=4)
    page_two = [e.text for e in corpus.iter_elements() if e.page == 2]
    assert page_two and not any("Overview" in text for text in page_two)


def test_build_uses_the_profile_engines_without_classifying_or_escalating(
    tmp_path, pdfs, monkeypatch
):
    """The build only extracts elements: no document-type or section backend (no embedding
    model), no summaries, and never the Jev escalation even with its key set (no Jev output
    may enter training data)."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "not-a-real-key")
    folder, records = pdfs
    configs = []
    real_process = build_module.process

    def spy(source, **kwargs):
        analysis = real_process(source, **kwargs)
        configs.append((kwargs, analysis.config))
        return analysis

    monkeypatch.setattr(build_module, "process", spy)
    build_corpus(Corpus(tmp_path / "c"), folder, records=records, max_pages=5, first_pages=2)
    assert len(configs) == 2
    for _, config in configs:
        assert config.profile == "accurate" and config.max_pages == 5
        assert (config.doc_type, config.sections, config.escalate) == ("none", "none", "none")
        assert config.summaries == "none" and config.embeddings == "none"
        assert config.layout == "fake-heron"  # the layout engine passed in, made once
    long = select_pages(24, first_pages=2, max_pages=5, seed=0, doc_id="long")
    assert [kwargs["pages"] for kwargs, _ in configs] == [long, [1, 2, 3, 4]]
    with pytest.raises(ValueError, match="first_pages"):
        build_corpus(Corpus(tmp_path / "x"), folder, records=records, max_pages=5, first_pages=6)


def test_build_without_records_reads_every_pdf_in_the_folder(tmp_path, pdfs):
    folder, _ = pdfs
    corpus = Corpus(tmp_path / "c")
    report = build_corpus(corpus, folder, max_pages=4, first_pages=4)
    assert report.added == ["long", "short"]
    (record,) = [r for r in corpus.manifest() if r.id == "short"]
    data = (folder / "short.pdf").read_bytes()
    assert record.sha256 == hashlib.sha256(data).hexdigest() and record.bytes == len(data)
    assert record.split == "train" and record.source == "local"


def test_cli_corpus_build_and_stats(tmp_path, pdfs):
    folder, records = pdfs
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(r.model_dump_json() + "\n" for r in records))
    runner = CliRunner()
    store = tmp_path / "store"
    args = ["corpus", "build", str(store), "--from-manifest", str(manifest),
            "--pdf-dir", str(folder), "--max-pages", "6", "--first-pages", "2"]  # fmt: skip
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "2 added" in result.output
    assert _doc(store, "long")["pages_read"] == select_pages(
        24, first_pages=2, max_pages=6, seed=0, doc_id="long"
    )
    result = runner.invoke(app, args)
    assert result.exit_code == 0 and "0 added, 2 skipped" in result.output

    result = runner.invoke(app, ["corpus", "stats", str(store)])
    assert result.exit_code == 0, result.output
    stats = json.loads(result.output)
    assert stats["documents"] == 2 and stats["by_split"]["gold"]["documents"] == 1

    result = runner.invoke(app, ["corpus", "stats", str(tmp_path / "nowhere")])
    assert result.exit_code == 1 and "no corpus" in result.output
    result = runner.invoke(app, args[:3] + ["--profile", "turbo"])
    assert result.exit_code != 0

    # The source manifest cannot be the store's own index.
    own = args[:3] + ["--from-manifest", str(store / "manifest.jsonl"), "--pdf-dir", str(folder)]
    result = runner.invoke(app, own)
    assert result.exit_code == 1 and "store's own index" in result.output

    # Failures are reported, skipped next time, and read again with --retry-failed.
    (folder / "bad.pdf").write_bytes(b"%PDF-1.7\nbroken")
    manifest.write_text(_record(folder / "bad.pdf", "bad").model_dump_json() + "\n")
    result = runner.invoke(app, args)
    assert result.exit_code == 0 and "1 failed, 0 failed before" in result.output
    assert "failed bad: DocrooterError" in result.output
    assert "1 failed before" in runner.invoke(app, args).output
    result = runner.invoke(app, [*args, "--retry-failed"])
    assert "[1/1] bad: reading" in result.output and "1 failed, 0 failed before" in result.output
