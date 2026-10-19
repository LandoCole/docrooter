"""Evidence QA for selection (spec §8.2, §0.4; task S7): evaluation only, descriptive only.

**Dataset.** FinanceBench's open-source split (Islam et al. 2023; GitHub
``patronus-ai/financebench``: ``data/financebench_open_source.jsonl``, questions with their
evidence pages and texts, and the PDFs in ``pdfs/``), licensed CC BY-NC 4.0. It is never
redistributed, trained on or calibrated on. If FinanceBench cannot be fetched, TAT-DQA (Zhu et
al. 2022, CC BY 4.0) is used under the same rules, read from files placed by hand under
``<cache>/tat-dqa/`` (its data is on Google Drive, which the polite fetcher cannot reach):
``tatdqa_dataset_<split>.json`` and ``tat_docs/<split>/<doc uid>.pdf`` with the OCR file
``<doc uid>.json`` beside it. Which dataset was used is recorded in ``dataset.json`` and in the
results.

**Where data lives.** Everything fetched or derived goes under :data:`CACHE`
(``benchmarks/.cache.nosync/evidence_qa/``, the explicit nosync path, git-ignored); every write
passes :func:`in_cache`. Downloads use :mod:`benchmarks.training_corpus._net` (fixed polite
User-Agent with no personal data, robots.txt, rate limits, capped PDF downloads), at most
:data:`MAX_DOCUMENTS` PDFs, the first documents in question-file order. Files already in the
cache are never fetched again, so when the fetcher may not reach GitHub (robots.txt), the
question file can be placed by hand at ``<cache>/financebench/financebench_open_source.jsonl``
and the PDFs at ``<cache>/financebench/pdfs/<doc_name>.pdf``. Results hold only ids and
numbers: no question, answer or evidence text is written or printed.

**Relevance.** An element is relevant to a question when one of its pages is an evidence page
and at least :data:`MIN_OVERLAP` of its characters overlap that page's evidence text
(:func:`overlap_share`): both texts are whitespace-normalised, and the overlap is the total size
of :class:`difflib.SequenceMatcher`'s matching blocks (``autojunk`` off) of at least
:data:`MIN_BLOCK_CHARS` characters, so that scattered single letters and short words do not
count. FinanceBench's ``evidence_page_num`` is 0-based (``--page-base``); the results also
count the questions that would align with the other base, to catch a wrong one. A question with
no aligned element, or with an evidence page beyond the pages read, is excluded and counted.

**Reading.** Each PDF is read as the corpus build reads a document
(:class:`~docrooter.select.fit.CorpusPdfReader`: the profile's engines, the first
``--max-pages`` pages, sections normalised as an annual report), tagged and embedded once.

**Metrics.** *Query needs*: the question text as a free-text need (``guarantee=False``, spec
§0.4); evidence recall and token share at every c of :data:`QUERY_GRID`. *Closest field*: the
built-in annual-report field whose text (its ``query``, else its description) is most similar
to the question (:func:`~docrooter.select.selection.similarities`: the field text embedded as a
passage once per run, each question with the query prompt: :class:`FieldMatcher`); evidence
recall and token share at c in :data:`FIELD_GRID`. Results are descriptive: no query-side
guarantee exists.

**Failing loudly.** A loaded dataset with no question, or with more than half its questions
without evidence, stops with a message naming the schema (:func:`check_questions`); a run where
no question aligns stops naming the page base or the schema, after writing its results
(:func:`check_alignment`). A document that fails to read, for any reason, is recorded with the
exception's type and the run goes on; the results file is rewritten after every document
(:func:`run_documents`), so a crash keeps what is done. The reading profile defaults to the
selector fit's; the page cap does not (whole filings are read).

Commands (fetch needs no model; run loads the tagger and embedder)::

    PYTHONPATH=src .venv/bin/python -m benchmarks.evidence_qa fetch
    PYTHONPATH=src .venv/bin/python -m benchmarks.evidence_qa run \\
        --tagger <bundle> --selector <selector.json>
"""

from __future__ import annotations

import argparse
import datetime
import difflib
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, TypeVar

import numpy as np

from benchmarks import _guard
from benchmarks.selection_bench import c_key, evaluate_selection, summary_stats

BENCHMARKS = Path(__file__).resolve().parent
CACHE = BENCHMARKS / ".cache.nosync" / "evidence_qa"
FINANCEBENCH = "financebench"
TAT_DQA = "tat-dqa"
LICENCES = {FINANCEBENCH: "CC BY-NC 4.0", TAT_DQA: "CC BY 4.0"}
REPO = "patronus-ai/financebench"
BRANCH = "main"
QUESTIONS_FILE = "financebench_open_source.jsonl"
QUESTIONS_URL = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/data/{QUESTIONS_FILE}"
PDF_URLS = (
    f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/pdfs/{{name}}.pdf",
    # The same file when the repository keeps its PDFs in Git LFS (raw then serves a pointer).
    f"https://media.githubusercontent.com/media/{REPO}/{BRANCH}/pdfs/{{name}}.pdf",
)
MAX_DOCUMENTS = 50
DATASET_FILE = "dataset.json"
RESULTS_FILE = "selection-evidence-qa.json"
FAMILY = "annual_report"
MIN_OVERLAP = 0.5
MIN_BLOCK_CHARS = 4
QUERY_GRID: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)
FIELD_GRID: tuple[float, ...] = (0.8, 0.9, 0.98)
DEFAULT_MAX_PAGES = 200  # whole filings: evidence pages lie anywhere in a 10-K
_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def log(message: str) -> None:
    print(f"[{datetime.datetime.now():%H:%M:%S}] {message}", flush=True)


# --- the data -------------------------------------------------------------------------------------


class Evidence(NamedTuple):
    page: int  # 1-based
    text: str


@dataclass(frozen=True)
class Question:
    """One evaluation question. Its text is held in memory only, never written out."""

    id: str
    doc_id: str
    question: str = field(repr=False)
    answer: str | tuple[str, ...] = field(repr=False)  # a tuple: any of them is right
    evidence: tuple[Evidence, ...] = field(repr=False)


@dataclass(frozen=True)
class Dataset:
    name: str
    licence: str
    questions: list[Question]
    pdf_dir: Path
    counts: dict[str, Any]


@dataclass(frozen=True)
class PageText:
    """A content element reduced to what alignment needs."""

    id: str
    pages: frozenset[int]
    text: str


def in_cache(path: Path, cache: Path) -> Path:
    """``path`` if it lies under ``cache``; ``ValueError`` otherwise. Every write of this module
    goes through it, so evaluation-only data never lands anywhere else."""
    resolved, root = Path(path).resolve(), Path(cache).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"refusing to write {path}: outside the evaluation-only cache {cache}")
    return Path(path)


def _write_bytes(path: Path, data: bytes, cache: Path) -> None:
    in_cache(path, cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = in_cache(path.with_name(path.name + ".tmp"), cache)
    tmp.write_bytes(data)
    os.replace(tmp, path)


def safe_id(name: str) -> str:
    """A document name as a corpus-safe id and file stem."""
    cleaned = _SAFE.sub("_", name)[:120]
    return cleaned if cleaned[:1].isalnum() else f"d{cleaned}"


def parse_financebench(lines: Iterable[str], page_base: int = 0) -> list[Question]:
    """FinanceBench rows as questions; ``page_base`` is the base of ``evidence_page_num``."""
    out = []
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        evidence = tuple(
            Evidence(int(item["evidence_page_num"]) + 1 - page_base, str(item["evidence_text"]))
            for item in row.get("evidence") or []
            if item.get("evidence_page_num") is not None and item.get("evidence_text")
        )
        out.append(
            Question(
                id=str(row["financebench_id"]),
                doc_id=str(row["doc_name"]),
                question=str(row["question"]),
                answer=str(row.get("answer") or ""),
                evidence=evidence,
            )
        )
    return out


def fetch_financebench(
    cache: Path = CACHE,
    *,
    net: Any = None,
    max_documents: int = MAX_DOCUMENTS,
    page_base: int = 0,
) -> Dataset:
    """The FinanceBench questions and up to ``max_documents`` PDFs, fetched into ``cache`` (or
    read from it). Raises the fetcher's errors when the question file cannot be had."""
    from benchmarks.training_corpus._net import FetchError, Net, Rejected

    root = cache / FINANCEBENCH
    if net is None:
        net = Net(in_cache(cache / "listings", cache), log=lambda message: None)
    questions_path = in_cache(root / QUESTIONS_FILE, cache)
    if not questions_path.is_file():
        _write_bytes(questions_path, net.listing(QUESTIONS_URL), cache)
    questions = parse_financebench(
        questions_path.read_text(encoding="utf-8").splitlines(), page_base=page_base
    )
    check_questions(questions, FINANCEBENCH)  # before any PDF is fetched
    names = list(dict.fromkeys(q.doc_id for q in questions))[:max_documents]
    pdf_dir = in_cache(root / "pdfs", cache)
    have: set[str] = set()
    failed = 0
    for name in names:
        dest = in_cache(pdf_dir / f"{safe_id(name)}.pdf", cache)
        if not dest.is_file():
            for template in PDF_URLS:
                url = template.format(name=urllib.parse.quote(name))
                try:
                    net.download_pdf(url, dest)
                    break
                except Rejected:
                    continue  # an LFS pointer, or not there: try the next source
                except FetchError:
                    break
        if dest.is_file():
            have.add(name)
        else:
            failed += 1
    kept = [q for q in questions if q.doc_id in have]
    return Dataset(
        name=FINANCEBENCH,
        licence=LICENCES[FINANCEBENCH],
        questions=kept,
        pdf_dir=pdf_dir,
        counts={
            "questions": len(questions),
            "documents_wanted": len(names),
            "documents": len(have),
            "documents_failed": failed,
            "questions_kept": len(kept),
            "page_base": page_base,
        },
    )


def load_tat_dqa(
    cache: Path = CACHE, *, split: str = "test", max_documents: int = MAX_DOCUMENTS
) -> Dataset:
    """TAT-DQA from files placed under ``<cache>/tat-dqa/`` (module docstring); evidence is the
    text of the OCR blocks its ``block_mapping`` names, grouped by page."""
    root = cache / TAT_DQA
    data_path = root / f"tatdqa_dataset_{split}.json"
    docs_dir = root / "tat_docs" / split
    if not data_path.is_file() or not docs_dir.is_dir():
        raise FileNotFoundError(
            f"TAT-DQA is not in place: expected {data_path} and {docs_dir}/<uid>.pdf|json"
        )
    entries = json.loads(data_path.read_text(encoding="utf-8"))
    questions: list[Question] = []
    have = 0
    for entry in entries:
        if have >= max_documents:
            break
        uid = str(entry["doc"]["uid"])
        pdf, ocr = docs_dir / f"{uid}.pdf", docs_dir / f"{uid}.json"
        if not pdf.is_file() or not ocr.is_file():
            continue
        have += 1
        blocks: dict[str, tuple[int, str]] = {}
        pages = json.loads(ocr.read_text(encoding="utf-8")).get("pages", [])
        for number, page in enumerate(pages, 1):
            for block in page.get("blocks", []):
                blocks[str(block.get("uid"))] = (number, str(block.get("text", "")))
        for q in entry.get("questions", []):
            by_page: dict[int, list[str]] = defaultdict(list)
            for mapping in q.get("block_mapping") or []:
                for block_uid in mapping:
                    if block_uid in blocks:
                        number, text = blocks[block_uid]
                        by_page[number].append(text)
            answer = q.get("answer")
            questions.append(
                Question(
                    id=str(q["uid"]),
                    doc_id=uid,
                    question=str(q["question"]),
                    answer=tuple(map(str, answer)) if isinstance(answer, list) else str(answer),
                    evidence=tuple(Evidence(n, " ".join(t)) for n, t in sorted(by_page.items())),
                )
            )
    check_questions(questions, TAT_DQA)
    return Dataset(
        name=TAT_DQA,
        licence=LICENCES[TAT_DQA],
        questions=questions,
        pdf_dir=docs_dir,
        counts={"documents": have, "questions_kept": len(questions), "split": split},
    )


def fetch(
    cache: Path = CACHE, *, max_documents: int = MAX_DOCUMENTS, page_base: int = 0
) -> Dataset:
    """FinanceBench, else TAT-DQA (module docstring); records the choice in ``dataset.json``."""
    from benchmarks.training_corpus._net import FetchError, Rejected

    reason = None
    try:
        dataset = fetch_financebench(cache, max_documents=max_documents, page_base=page_base)
        if not dataset.questions:
            reason = "no FinanceBench PDF could be fetched"
    except (FetchError, Rejected, OSError, ValueError) as exc:
        reason = f"FinanceBench fetch failed: {type(exc).__name__}: {exc}"
    if reason is not None:
        log(f"{reason}; falling back to TAT-DQA")
        try:
            dataset = load_tat_dqa(cache, max_documents=max_documents)
        except FileNotFoundError as exc:
            raise SystemExit(f"{reason}, and {exc}") from None
    record = {
        "dataset": dataset.name,
        "licence": dataset.licence,
        "fallback_reason": reason,
        "page_base": page_base,
        "max_documents": max_documents,
        "counts": dataset.counts,
    }
    _write_bytes(cache / DATASET_FILE, (json.dumps(record, indent=2) + "\n").encode(), cache)
    return dataset


def load_cached(cache: Path = CACHE) -> tuple[Dataset, dict[str, Any]]:
    """The dataset ``fetch`` chose, from the cache only (no network)."""
    path = cache / DATASET_FILE
    if not path.is_file():
        raise SystemExit(f"no fetched dataset at {cache}; run `benchmarks.evidence_qa fetch` first")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["dataset"] == FINANCEBENCH:
        root = cache / FINANCEBENCH
        questions = parse_financebench(
            (root / QUESTIONS_FILE).read_text(encoding="utf-8").splitlines(),
            page_base=int(record["page_base"]),
        )
        pdf_dir = root / "pdfs"
        kept = [q for q in questions if (pdf_dir / f"{safe_id(q.doc_id)}.pdf").is_file()]
        check_questions(kept, FINANCEBENCH)
        dataset = Dataset(FINANCEBENCH, LICENCES[FINANCEBENCH], kept, pdf_dir, record["counts"])
        return dataset, record
    return load_tat_dqa(cache, max_documents=int(record["max_documents"])), record


# --- alignment ------------------------------------------------------------------------------------


def normalise(text: str) -> str:
    return " ".join(text.split())


def overlap_share(element_text: str, evidence_text: str) -> float:
    """The share of the element's characters (whitespace-normalised) inside matching blocks of
    at least :data:`MIN_BLOCK_CHARS` with the evidence text."""
    a, b = normalise(element_text), normalise(evidence_text)
    if not a or not b:
        return 0.0
    blocks = difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()
    return sum(block.size for block in blocks if block.size >= MIN_BLOCK_CHARS) / len(a)


def evidence_elements(
    elements: Sequence[PageText], evidence: Sequence[Evidence], min_share: float = MIN_OVERLAP
) -> set[str]:
    """The elements relevant to a question: on an evidence page, and overlapping that page's
    evidence text by at least ``min_share`` of their characters."""
    relevant: set[str] = set()
    for item in evidence:
        for element in elements:
            if element.id in relevant or item.page not in element.pages:
                continue
            if overlap_share(element.text, item.text) >= min_share:
                relevant.add(element.id)
    return relevant


def page_texts(ctx: Any) -> list[PageText]:
    """A prepared document's content elements, with their pages, for alignment."""
    elements = ctx.index.doc.elements
    return [
        PageText(el.id, frozenset(box.page for box in el.boxes), el.text)
        for el in (elements[pos] for pos in ctx.index.content)
    ]


# --- reading --------------------------------------------------------------------------------------


class _FirstPages:
    """Stands in for the corpus store in :class:`~docrooter.select.fit.CorpusPdfReader`: a
    document's pages are its first ``max_pages``."""

    def __init__(self, pdf_dir: Path, max_pages: int, text_engine: str) -> None:
        self.pdf_dir = pdf_dir
        self.max_pages = max_pages
        self.text_engine = text_engine

    def pages_read(self, doc_id: str) -> list[int]:
        from docrooter.corpus.build import _page_count

        count = _page_count(self.pdf_dir / f"{doc_id}.pdf", self.text_engine)
        return list(range(1, min(count, self.max_pages) + 1))


def make_reader(pdf_dir: Path, *, profile: str, max_pages: int, sections_backend: Any) -> Any:
    """A :class:`~docrooter.select.fit.CorpusPdfReader` over the first ``max_pages`` pages."""
    from docrooter.corpus.build import reading_config
    from docrooter.select.fit import CorpusPdfReader

    pages = _FirstPages(pdf_dir, max_pages, reading_config(profile, max_pages).text)
    return CorpusPdfReader(
        pages,  # type: ignore[arg-type]  # only its pages_read is used
        pdf_dir,
        profile=profile,
        max_pages=max_pages,
        sections_backend=sections_backend,
    )


def record_for(doc_id: str, pdf_dir: Path) -> Any:
    """A manifest record for an evaluation PDF: an annual report, hashed as it is on disk."""
    from docrooter.corpus.store import ManifestRecord

    path = pdf_dir / f"{safe_id(doc_id)}.pdf"
    return ManifestRecord(
        id=safe_id(doc_id),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        family_hint=FAMILY,
    )


class FieldMatcher:
    """The closest built-in field to a question. The field texts (each field's ``query``, else
    its description) are embedded once, as passages, when the matcher is made; each question is
    embedded with the recipe's query prompt, as :func:`~docrooter.select.selection.similarities`
    does."""

    def __init__(self, fields: Sequence[Any], encoder: Any) -> None:
        from docrooter.select.selection import _text_vectors, need_texts

        self.fields = list(fields)
        self.encoder = encoder
        self.vectors = _text_vectors([need_texts(f)[0] for f in self.fields], encoder)

    def closest(self, question: str) -> Any:
        from docrooter.select.selection import _need_sims

        sims = _need_sims(self.vectors, question, self.encoder)
        return self.fields[int(np.argmax(sims))]


def check_questions(questions: Sequence[Question], name: str) -> None:
    """Exit when the loaded questions look like a misread dataset: none at all, or more than
    half without evidence (a schema the loader does not match)."""
    if not questions:
        raise SystemExit(
            f"{name}: no question loaded; the dataset schema likely differs from what the "
            "loader expects (or no document was fetched)"
        )
    bare = sum(1 for q in questions if not q.evidence)
    if bare * 2 > len(questions):
        raise SystemExit(
            f"{name}: {bare} of {len(questions)} questions have no evidence; the dataset schema "
            "likely differs from what the loader expects (its evidence fields)"
        )


def check_alignment(counts: Mapping[str, int]) -> list[str]:
    """Exit when a run's alignment looks wrong: no evaluated question, more than half without
    evidence (the schema), or none aligned (the page base or the schema). Returns a warning when
    the other page base would align more questions."""
    evaluated = counts.get("questions_evaluated", 0)
    if not evaluated:
        raise SystemExit("no question was evaluated; the dataset schema likely differs")
    bare = counts.get("no_evidence", 0)
    if bare * 2 > evaluated:
        raise SystemExit(
            f"{bare} of {evaluated} evaluated questions have no evidence; the dataset schema "
            "likely differs from what the loader expects"
        )
    aligned = counts.get("aligned", 0)
    if not aligned:
        raise SystemExit(
            f"no question aligned with its evidence (of {evaluated}); check the page base "
            "(fetch --page-base) or the dataset schema"
        )
    other = counts.get("aligned_if_other_base", 0)
    if other > aligned:
        return [
            f"{other} questions would align with the other page base against {aligned} with "
            "this one: check the page base (fetch --page-base)"
        ]
    return []


T = TypeVar("T")
Evaluate = Callable[[str, T], tuple[list[dict[str, Any]], Counter[str]]]
Progress = Callable[[list[dict[str, Any]], Counter[str], list[dict[str, str]]], None]


def run_documents(
    items: Sequence[tuple[str, T]], evaluate: Evaluate[T], on_progress: Progress
) -> tuple[list[dict[str, Any]], Counter[str], list[dict[str, str]]]:
    """``evaluate(doc_id, payload)`` for each item, so that one document's failure (any
    exception) is recorded with its type and the run goes on; ``on_progress(rows, counts,
    failures)`` after every document, for partial results. Interrupts still stop the run."""
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    for done, (doc_id, payload) in enumerate(items, 1):
        try:
            doc_rows, doc_counts = evaluate(doc_id, payload)
        except Exception as exc:
            failures.append({"doc_id": doc_id, "error": type(exc).__name__})
            counts["documents_failed"] += 1
            log(f"[{done}/{len(items)}] document failed ({type(exc).__name__}); skipped")
        else:
            rows += doc_rows
            counts.update(doc_counts)
            log(f"[{done}/{len(items)}] {len(doc_rows)} questions evaluated")
        on_progress(rows, counts, failures)
    return rows, counts, failures


# --- evaluation -----------------------------------------------------------------------------------


def evaluate_document(
    analysis: Any,
    ctx: Any,
    questions: Sequence[Question],
    matcher: FieldMatcher,
    selector: Any,
    pages_read: set[int],
    *,
    other_base_shift: int | None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Per question of one prepared document: evidence recall and token share for the question
    as a query need and for its closest field. Rows hold ids and numbers only."""
    from docrooter.select.selection import need_features, select_scored

    texts = page_texts(ctx)
    field_X: dict[str, np.ndarray] = {}
    counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for q in questions:
        counts["questions_evaluated"] += 1
        if other_base_shift is not None:
            shifted = [Evidence(e.page + other_base_shift, e.text) for e in q.evidence]
            counts["aligned_if_other_base"] += bool(evidence_elements(texts, shifted))
        if not q.evidence:
            counts["no_evidence"] += 1
            continue
        if any(e.page not in pages_read for e in q.evidence):
            counts["pages_not_read"] += 1
            continue
        relevant = evidence_elements(texts, q.evidence)
        if not relevant:
            counts["unaligned"] += 1
            continue
        counts["aligned"] += 1
        _, Xq = need_features(ctx, q.question)
        query = {
            c_key(c): evaluate_selection(
                select_scored(analysis, ctx, Xq, q.question, completeness=c, selector=selector),
                relevant,
            )
            for c in QUERY_GRID
        }
        field = matcher.closest(q.question)
        field_id = field.id
        if field_id not in field_X:
            field_X[field_id] = need_features(ctx, field)[1]
        by_field = {
            c_key(c): evaluate_selection(
                select_scored(
                    analysis, ctx, field_X[field_id], field, completeness=c,
                    selector=selector,
                ),
                relevant,
            )
            for c in FIELD_GRID
        }  # fmt: skip
        rows.append(
            {
                "id": q.id,
                "n_relevant": len(relevant),
                "field_id": field_id,
                "query": query,
                "field": by_field,
            }
        )
    return rows, counts


def summarise(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per c: recall (mean, p10, min), mean token share and guarantee share, for query needs and
    for the closest field."""
    return {
        "query": {
            c_key(c): summary_stats([r["query"][c_key(c)] for r in rows]) for c in QUERY_GRID
        },
        "closest_field": {
            c_key(c): summary_stats([r["field"][c_key(c)] for r in rows]) for c in FIELD_GRID
        },
        "closest_fields_used": dict(Counter(r["field_id"] for r in rows).most_common()),
    }


def cmd_fetch(args: argparse.Namespace) -> int:
    dataset = fetch(max_documents=args.max_documents, page_base=args.page_base)
    log(
        f"{dataset.name} ({dataset.licence}): {dataset.counts.get('documents')} documents, "
        f"{len(dataset.questions)} questions, under {CACHE}"
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from benchmarks.selection_bench import (
        _commit,
        _environment,
        _out_dir,
        _write_json,
        load_selector,
        reading_settings,
    )
    from docrooter.backends.base import close_backend
    from docrooter.backends.registry import resolve_backend
    from docrooter.rooter.profiles import profile_defaults
    from docrooter.select.fields import builtin_fields
    from docrooter.select.selection import prepare
    from docrooter.tagger.element_tags import resolve_tagger

    started = time.perf_counter()
    dataset, record = load_cached(CACHE)
    selector = load_selector(args.selector, args.tagger)
    # The profile defaults to the selector fit's; the page cap does not: whole filings are read
    # here, because evidence pages lie anywhere in them.
    profile, _, warnings = reading_settings(args.profile, None, selector.meta)
    for warning in warnings:
        log(f"warning: {warning}")
    embedder = args.embedder or selector.embedder
    if embedder is None:
        raise SystemExit("the selector records no embedder; pass --embedder")
    by_doc: dict[str, list[Question]] = defaultdict(list)
    for q in dataset.questions:
        by_doc[q.doc_id].append(q)
    shift = None
    if dataset.name == FINANCEBENCH:
        shift = -1 if int(record["page_base"]) == 0 else 1
    path = _out_dir(args.out) / RESULTS_FILE

    def results(
        rows: list[dict[str, Any]], counts: Counter[str], failures: list[dict[str, str]]
    ) -> dict[str, Any]:
        return {
            "date": datetime.date.today().isoformat(),
            "partial": len(failures) + len({r["doc_id"] for r in rows}) < len(by_doc),
            "dataset": dataset.name,
            "licence": dataset.licence,
            "evaluation_only": True,
            "descriptive_only": "queries are never guaranteed (spec §0.4)",
            "fallback_reason": record.get("fallback_reason"),
            "page_base": record.get("page_base"),
            "alignment": {"min_overlap": MIN_OVERLAP, "min_block_chars": MIN_BLOCK_CHARS},
            "counts": {**dataset.counts, **dict(counts)},
            "failures": failures,
            "summary": summarise(rows),
            "questions": rows,
            "provenance": {
                "commit": _commit(),
                "tagger": args.tagger,
                "selector": str(args.selector) if args.selector else None,
                "embedder": embedder,
                "profile": profile,
                "max_pages": args.max_pages,
                "reading_warnings": warnings,
            },
            "seconds": round(time.perf_counter() - started, 1),
            **_environment(),
        }

    tagger = resolve_tagger(args.tagger)
    encoder = resolve_backend(embedder)
    same = embedder == profile_defaults(profile)["sections"]
    reader = make_reader(
        dataset.pdf_dir,
        profile=profile,
        max_pages=args.max_pages,
        sections_backend=encoder if same else None,
    )
    matcher = FieldMatcher(builtin_fields(FAMILY), encoder)  # field texts embedded once

    def evaluate(
        doc_id: str, questions: list[Question]
    ) -> tuple[list[dict[str, Any]], Counter[str]]:
        doc = record_for(doc_id, dataset.pdf_dir)
        analysis = reader(doc)
        pages_read = set(reader.corpus.pages_read(doc.id))
        ctx = prepare(analysis, tagger=tagger, encoder=encoder, family=FAMILY)
        doc_rows, doc_counts = evaluate_document(
            analysis, ctx, questions, matcher, selector, pages_read, other_base_shift=shift
        )
        for row in doc_rows:
            row["doc_id"] = doc_id
        return doc_rows, doc_counts

    try:
        rows, counts, failures = run_documents(
            list(by_doc.items()),
            evaluate,
            lambda r, c, f: _write_json(path, results(r, c, f)),  # partial results as it goes
        )
    finally:
        reader.close()
        close_backend(tagger)
        close_backend(encoder)

    final = results(rows, counts, failures)
    _write_json(path, final)
    q90 = final["summary"]["query"][c_key(0.9)]
    log(
        f"{dataset.name}: {counts['aligned']} aligned questions, {len(failures)} documents "
        f"failed; query need at c = 0.90: mean recall {q90['recall_mean']}, token share "
        f"{q90['token_share_mean']} -> {path}"
    )
    for warning in check_alignment(counts):  # exits non-zero on a misread dataset
        log(f"warning: {warning}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fetch", help="fetch FinanceBench (else use TAT-DQA) into the cache")
    p.add_argument("--max-documents", type=int, default=MAX_DOCUMENTS)
    p.add_argument("--page-base", type=int, choices=(0, 1), default=0)
    p = sub.add_parser("run", help="evaluate selection on the fetched dataset (loads models)")
    p.add_argument("--tagger", required=True, help="element tagger bundle directory")
    p.add_argument("--selector", type=Path, default=None, help="selector.json (default: bundle's)")
    p.add_argument("--embedder", default=None, help="embedding spec (default: the selector's)")
    p.add_argument("--profile", default=None, help="reading profile (default: the selector fit's)")
    p.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    p.add_argument("--out", type=Path, default=None, help="results dir (default: dated)")
    _guard.add_guard_argument(p)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fetch":
        return cmd_fetch(args)
    _guard.single_model_process("benchmarks.evidence_qa", allow_concurrent=args.allow_concurrent)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
