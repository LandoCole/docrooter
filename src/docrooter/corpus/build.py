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

The build is resumable. Documents already in the store (by id or sha256) are skipped, each
document is committed as soon as it is read (see :mod:`docrooter.corpus.store`), and a PDF
that cannot be read is recorded in the report and the build moves on. The OCR and layout
engines are created once, on the first document that needs reading, and closed at the end.
"""

from __future__ import annotations

import hashlib
import os
import random
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from docrooter.corpus.store import Corpus, CorpusElement, ManifestRecord
from docrooter.document.models import Document
from docrooter.errors import DocrooterError, MissingExtraError
from docrooter.rooter import RouterConfig, process
from docrooter.rooter.process import _Engines
from docrooter.summarize import heading_titles

__all__ = ["SKIPPED_KINDS", "BuildReport", "build_corpus", "corpus_elements", "select_pages"]

SKIPPED_KINDS = frozenset({"page_header", "page_footer", "page_number", "contents"})
"""Element kinds the corpus does not store: page furniture and printed contents."""

_MAX_ERROR = 300  # characters of an error message kept in the report

Progress = Callable[[int, int, str, str], None]
"""``on_progress(done, total, doc_id, status)``, after each document; ``status`` is ``added``,
``skipped``, ``missing``, ``mismatched`` or ``failed``."""


class BuildReport(BaseModel):
    """What one :func:`build_corpus` run did, by document id: ``added`` to the store,
    ``skipped`` (already there, under this id or with the same sha256), ``missing`` (no
    ``<id>.pdf``), ``mismatched`` (the PDF's sha256 is not the record's) and ``failed`` (the
    PDF could not be read: the error's type and message). ``pages_read`` and ``elements``
    count what was added; ``seconds`` is the wall-clock time."""

    added: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    mismatched: list[str] = Field(default_factory=list)
    failed: dict[str, str] = Field(default_factory=dict)
    pages_read: int = 0
    elements: int = 0
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
    on_progress: Progress | None = None,
) -> BuildReport:
    """Read the documents of ``records`` from ``pdf_dir/<id>.pdf`` into ``corpus``.

    ``records`` are the manifest's records (see :func:`~docrooter.corpus.read_manifest`); when
    ``None``, every ``*.pdf`` in ``pdf_dir`` is a ``train`` document whose id is its file name
    (without ``.pdf``). Each PDF must hash to its record's sha256. ``profile`` names the
    rooter profile whose text, OCR and layout engines read the pages; see the module docstring
    for what the build switches off, which pages it reads and how it resumes. The store's
    ``stats.json`` is updated at the end.
    """
    select_pages(0, first_pages=first_pages, max_pages=max_pages)  # check the arguments
    cfg = _config(profile, max_pages)
    folder = Path(pdf_dir)
    report = BuildReport()
    started = time.perf_counter()
    todo = list(records) if records is not None else _folder_records(folder, report)
    engines = _Engines(cfg, {})
    try:
        for done, record in enumerate(todo, 1):
            status = _build_one(corpus, record, folder, cfg, engines, report, first_pages, seed)
            if on_progress is not None:
                on_progress(done, len(todo), record.id, status)
    finally:
        engines.close()
    report.seconds = round(time.perf_counter() - started, 3)
    corpus.stats()
    return report


def _config(profile: str, max_pages: int) -> RouterConfig:
    """``profile``'s configuration with every task but loading switched off."""
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


def _build_one(
    corpus: Corpus,
    record: ManifestRecord,
    folder: Path,
    cfg: RouterConfig,
    engines: _Engines,
    report: BuildReport,
    first_pages: int,
    seed: int,
) -> str:
    if corpus.has(record.id) or corpus.has_sha256(record.sha256):
        report.skipped.append(record.id)
        return "skipped"
    path = folder / f"{record.id}.pdf"
    if not path.is_file():
        report.missing.append(record.id)
        return "missing"
    if _sha256(path) != record.sha256:
        report.mismatched.append(record.id)
        return "mismatched"
    # Created on first use and reused; a missing extra stops the build here.
    layout, _ = engines.layout()
    ocr, _ = engines.ocr()
    try:
        pages = select_pages(
            _page_count(path, cfg.text),
            first_pages=first_pages,
            max_pages=cfg.max_pages or 1,
            seed=seed,
            doc_id=record.id,
        )
        analysis = process(
            path, config=cfg, pages=pages, layout=layout, ocr="none" if ocr is None else ocr
        )
        elements = corpus_elements(record.id, analysis.document)
    except (MemoryError, MissingExtraError):  # the machine or the install, not the PDF
        raise
    except Exception as exc:  # an unreadable PDF must not stop a long build
        report.failed[record.id] = f"{type(exc).__name__}: {exc}"[:_MAX_ERROR]
        return "failed"
    corpus.add_document(record, elements, pages)
    report.added.append(record.id)
    report.pages_read += len(pages)
    report.elements += len(elements)
    return "added"


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
