"""Build a corpus store from PDFs (spec §5): read part of every document with the rooter and
store its elements.

Each document is read with :func:`~docrooter.process` in element mode under a profile
(``accurate``: the heron layout model), on at most ``max_pages`` pages: the first
``first_pages`` and a sample of the rest, drawn with a generator seeded by ``seed`` and the
document id (:func:`select_pages`). The pages read are recorded. Furniture (page headers,
footers and numbers) and ``contents`` elements are not stored.

The build only extracts elements: document-type and section classification, summaries,
embeddings and escalation are switched off whatever the profile says. That keeps models other
than the layout engine out of the build, and Jev output out of the corpus (it may never enter
training data, TypeSafe MCA §2.3(b)).

The build is resumable (Review Focus 5):

- documents already in the store (by id or sha256) are skipped, and each document is committed
  as soon as it is read (see :mod:`docrooter.corpus.store`);
- a PDF that raises while being read is recorded in the store's ``failed.jsonl`` and the build
  moves on;
- before reading a document the build durably notes it (``attempting.json``) and clears the
  note afterwards. A PDF that kills the process (a native crash, the OOM killer) leaves the
  note behind, and the next build records that document as failed ("crashed previous run")
  instead of reading it again;
- recorded failures are skipped on later builds unless ``retry_failed`` is set.

The OCR and layout engines are created once, on the first document that needs reading, and
closed at the end. ``ocr="auto"`` with no OCR engine installed is logged once per build and
noted in the report's ``warnings``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from docrooter.corpus.store import Corpus, CorpusElement, ManifestRecord
from docrooter.document.models import Document
from docrooter.errors import DocrooterError, MissingExtraError
from docrooter.rooter import RouterConfig, process
from docrooter.rooter.process import _Engines
from docrooter.summarize import heading_titles

__all__ = [
    "CRASHED",
    "SKIPPED_KINDS",
    "BuildReport",
    "build_corpus",
    "corpus_elements",
    "reading_config",
    "select_pages",
]

SKIPPED_KINDS = frozenset({"page_header", "page_footer", "page_number", "contents"})
"""Element kinds the corpus does not store: page furniture and printed contents."""

_MAX_ERROR = 300  # characters of an error message kept in the report
CRASHED = "crashed previous run"
"""The failure reason of a document that was being read when an earlier build died."""
OCR_UNAVAILABLE = "ocr_unavailable"

_log = logging.getLogger(__name__)

Progress = Callable[[int, int, str, str], None]
"""``on_progress(done, total, doc_id, status)``. ``status`` is ``reading`` just before a
document is read (so the last progress line names a document that kills the process), then one
of ``added``, ``skipped``, ``skipped_failed``, ``missing``, ``mismatched`` or ``failed`` once
it is done with."""


class BuildReport(BaseModel):
    """What one :func:`build_corpus` run did, by document id: ``added`` to the store,
    ``skipped`` (already there, under this id or with the same sha256), ``missing`` (no
    ``<id>.pdf``), ``mismatched`` (the PDF's sha256 is not the record's), ``failed`` (the
    PDF could not be read in this run: the error's type and message) and ``skipped_failed``
    (a failure recorded by an earlier run, not retried). ``crashed`` names the document an
    earlier run died reading, recorded as failed when this run started. ``pages_read`` and
    ``elements`` count what was added; ``warnings`` holds ``"ocr_unavailable"`` when
    ``ocr="auto"`` found no engine; ``seconds`` is the wall-clock time."""

    added: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    skipped_failed: list[str] = Field(default_factory=list)
    crashed: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    mismatched: list[str] = Field(default_factory=list)
    failed: dict[str, str] = Field(default_factory=dict)
    pages_read: int = 0
    elements: int = 0
    warnings: list[str] = Field(default_factory=list)
    seconds: float = 0.0


def select_pages(
    n_pages: int, *, first_pages: int = 10, max_pages: int = 30, seed: int = 0, doc_id: str = ""
) -> list[int]:
    """The 1-based pages to read of a document of ``n_pages`` pages, in order: all of them if
    there are at most ``max_pages``, else the first ``first_pages`` and ``max_pages -
    first_pages`` of the others drawn at random. The draw depends only on ``seed`` and
    ``doc_id``, so it is the same on every run and machine."""
    if max_pages < 1:
        raise ValueError(f"max_pages must be >= 1, got {max_pages}")
    if not 0 <= first_pages <= max_pages:
        raise ValueError(
            f"first_pages must be between 0 and max_pages ({max_pages}), got {first_pages}"
        )
    if n_pages < 0:
        raise ValueError(f"n_pages must be >= 0, got {n_pages}")
    if n_pages <= max_pages:
        return list(range(1, n_pages + 1))
    rng = random.Random(f"{seed}:{doc_id}")  # str seeds hash with SHA-512: stable everywhere
    sampled = rng.sample(range(first_pages + 1, n_pages + 1), max_pages - first_pages)
    return list(range(1, first_pages + 1)) + sorted(sampled)


def corpus_elements(doc_id: str, document: Document) -> list[CorpusElement]:
    """The elements of ``document`` the corpus stores (all but furniture and ``contents``), with
    their heading paths, section titles, pages and, for tables, Markdown."""
    out: list[CorpusElement] = []
    for element in document.elements:
        if element.kind in SKIPPED_KINDS:
            continue
        section = document.sections.get(element.section_id or "")
        out.append(
            CorpusElement(
                doc_id=doc_id,
                element_id=element.id,
                kind=element.kind,
                text=element.text,
                heading_path=heading_titles(document, element.section_id),
                section_title=section.title if section is not None and section.level > 0 else None,
                page=element.boxes[0].page if element.boxes else None,
                table_markdown=element.table.to_markdown() if element.table is not None else None,
            )
        )
    return out


def build_corpus(
    corpus: Corpus,
    pdf_dir: str | os.PathLike[str],
    *,
    records: Iterable[ManifestRecord] | None = None,
    profile: str = "accurate",
    max_pages: int = 30,
    first_pages: int = 10,
    seed: int = 0,
    retry_failed: bool = False,
    on_progress: Progress | None = None,
) -> BuildReport:
    """Read the documents of ``records`` from ``pdf_dir/<id>.pdf`` into ``corpus``.

    ``records`` are the manifest's records (see :func:`~docrooter.corpus.read_manifest`); when
    ``None``, every ``*.pdf`` in ``pdf_dir`` is a ``train`` document whose id is its file name
    (without ``.pdf``). Each PDF must hash to its record's sha256. ``profile`` names the
    rooter profile whose text, OCR and layout engines read the pages; see the module docstring
    for what the build switches off, which pages it reads and how it resumes (``retry_failed``
    reads the documents that failed in earlier runs again). The store's ``stats.json`` is
    updated at the end.
    """
    select_pages(0, first_pages=first_pages, max_pages=max_pages)  # check the arguments
    cfg = reading_config(profile, max_pages)
    report = BuildReport()
    started = time.perf_counter()
    crashed = corpus.pending_attempt()
    if crashed is not None:
        if not corpus.has(crashed):
            _log.warning("%s was being read when the last build died; recorded as failed", crashed)
            corpus.record_failure(crashed, CRASHED)
            report.crashed.append(crashed)
        corpus.finish_attempt()
    folder = Path(pdf_dir)
    todo = list(records) if records is not None else _folder_records(folder, report)
    run = _Run(corpus, folder, cfg, report, first_pages, seed, on_progress, len(todo))
    known_failures = {} if retry_failed else corpus.failures()
    try:
        for done, record in enumerate(todo, 1):
            status = run.one(done, record, known_failures)
            if on_progress is not None:
                on_progress(done, len(todo), record.id, status)
    finally:
        run.engines.close()
    report.seconds = round(time.perf_counter() - started, 3)
    corpus.stats()
    return report


def reading_config(profile: str, max_pages: int) -> RouterConfig:
    """``profile``'s configuration with every task but loading switched off: how the build
    reads a document (and how fitting the selector reads it again, ``docrooter.select.fit``)."""
    cfg = RouterConfig.resolve(
        profile=profile,
        doc_type="none",
        sections="none",
        summaries="none",
        embeddings="none",
        escalate="none",
        max_pages=max_pages,
    )
    if cfg.layout == "none":
        raise ValueError(
            f"profile {profile!r} reads no elements (layout='none'); the corpus build needs "
            "element mode: use the balanced or accurate profile"
        )
    return cfg


class _Run:
    """One build: its settings, engines and report, and how it handles one document."""

    def __init__(
        self,
        corpus: Corpus,
        folder: Path,
        cfg: RouterConfig,
        report: BuildReport,
        first_pages: int,
        seed: int,
        on_progress: Progress | None,
        total: int,
    ) -> None:
        self.corpus, self.folder, self.cfg, self.report = corpus, folder, cfg, report
        self.first_pages, self.seed = first_pages, seed
        self.on_progress, self.total = on_progress, total
        self.engines = _Engines(cfg, {})

    def one(self, done: int, record: ManifestRecord, known_failures: dict[str, Any]) -> str:
        corpus, report = self.corpus, self.report
        if corpus.has(record.id) or corpus.has_sha256(record.sha256):
            report.skipped.append(record.id)
            return "skipped"
        if record.id in known_failures:
            report.skipped_failed.append(record.id)
            return "skipped_failed"
        path = self.folder / f"{record.id}.pdf"
        if not path.is_file():
            report.missing.append(record.id)
            return "missing"
        if _sha256(path) != record.sha256:
            report.mismatched.append(record.id)
            return "mismatched"
        conflict = corpus.id_conflict(record)
        if conflict is not None:  # found before the read, not after it
            self._fail(record.id, f"ValueError: {conflict}")
            return "failed"
        # Created on first use and reused; a missing extra stops the build here.
        layout, _ = self.engines.layout()
        ocr, _ = self.engines.ocr()
        if self.engines.ocr_unavailable and OCR_UNAVAILABLE not in report.warnings:
            _log.warning("ocr=auto found no OCR engine: pages that need OCR keep their text layer")
            report.warnings.append(OCR_UNAVAILABLE)
        if self.on_progress is not None:
            self.on_progress(done, self.total, record.id, "reading")
        corpus.start_attempt(record.id)
        try:
            pages = select_pages(
                _page_count(path, self.cfg.text),
                first_pages=self.first_pages,
                max_pages=self.cfg.max_pages or 1,
                seed=self.seed,
                doc_id=record.id,
            )
            analysis = process(
                path, config=self.cfg, pages=pages, layout=layout,
                ocr="none" if ocr is None else ocr,
            )  # fmt: skip
            elements = corpus_elements(record.id, analysis.document)
        except MemoryError:  # this document exhausted memory: record it, then stop
            self._fail(record.id, "MemoryError")
            raise
        except MissingExtraError:  # the install, not the PDF
            corpus.finish_attempt()
            raise
        except Exception as exc:  # an unreadable PDF must not stop a long build
            self._fail(record.id, f"{type(exc).__name__}: {exc}"[:_MAX_ERROR])
            return "failed"
        except BaseException:  # the run was stopped (Ctrl+C, exit): not the PDF's fault
            corpus.finish_attempt()
            raise
        try:
            corpus.add_document(record, elements, pages)
        finally:  # added, or the store failed (disk full): either way not a crashing PDF
            corpus.finish_attempt()
        report.added.append(record.id)
        report.pages_read += len(pages)
        report.elements += len(elements)
        return "added"

    def _fail(self, doc_id: str, reason: str) -> None:
        """Record a failure in the report and the store, and clear the attempt note."""
        self.report.failed[doc_id] = reason
        self.corpus.record_failure(doc_id, reason)
        self.corpus.finish_attempt()


def _folder_records(folder: Path, report: BuildReport) -> list[ManifestRecord]:
    """A ``train`` record for every PDF in ``folder``; a file name that is not a valid id is
    reported as failed."""
    records: list[ManifestRecord] = []
    for path in sorted(folder.glob("*.pdf")):
        try:
            records.append(
                ManifestRecord(
                    id=path.stem,
                    source="local",
                    sha256=_sha256(path),
                    bytes=path.stat().st_size,
                )
            )
        except ValidationError:
            report.failed[path.name] = "invalid document id (file name)"
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _page_count(path: Path, text_engine: str) -> int:
    """The document's page count, with the configured text engine (``auto``: as the loader
    picks it)."""
    from docrooter.pdf.element_mode import _page_count as count
    from docrooter.pdf.load import _auto_engine

    engine = text_engine if text_engine in ("pymupdf", "pdfplumber") else _auto_engine()
    try:
        return count(path, engine)
    except Exception as exc:
        raise DocrooterError(f"cannot open PDF {path}: {exc}") from exc
