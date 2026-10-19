"""Fetch the v0.5 training corpus: public native PDFs, finance-heavy, with no dev or held-out
overlap (spec §5).

    PYTHONPATH=src python -m benchmarks.training_corpus.fetch --target 1100
    PYTHONPATH=src python -m benchmarks.training_corpus.fetch --dry-run --target 30
    PYTHONPATH=src python -m benchmarks.training_corpus.fetch --sources fed arxiv --target 50

**Counts.** Each source's spec target and cap (at 1,100 documents) are scaled to ``--target``.
Sources are filled to their targets first; a shortfall (a source refused us, ran dry, or yielded
too few usable PDFs) is then made up from the other sources in :data:`MAKEUP_ORDER`, finance
first, each up to its cap. Both are recorded in the run summary and the README.

**Exclusion (Review Focus 1).** The dev manifest (``benchmarks/corpus/manifest.jsonl``) and the
held-out manifest (``benchmarks/heldout/manifest.jsonl``) are read for three fields only: sha256,
url and title. Nothing else under ``benchmarks/heldout/`` is opened. A candidate whose
normalised URL or normalised title matches is dropped before download; a download whose sha256
matches (the same bytes re-hosted elsewhere) is deleted and dropped. Every record is checked
again when the manifest is written, so a grown dev manifest also purges journalled documents.
Spec §5 matches on (publisher, normalised title); the controller ruling limits the manifests to
sha256, url and title, so titles are matched without the publisher, which can only drop more,
never less. Drops are recorded in the cache (``drops.jsonl``). A held-out drop is only counted:
its row holds a hash of the candidate id (so it is not retried) and no id, URL or title, and it
is never logged one by one.

**Checks.** Every candidate id must match ``^[a-z0-9_-]{1,80}$`` (it names a file). Every
download must start with ``%PDF-`` and be at most 50 MB. Its page count and text layer are read
by :mod:`benchmarks.training_corpus.pdf_meta` in a ``python -I`` child process with a timeout and
a 2 GB memory cap (``setrlimit`` where the OS enforces it, and a resident-size watchdog here); a
PDF with no usable text layer in its first pages (a scan) is dropped, since the corpus is native
PDFs only. Byte-identical documents are kept once. A candidate taken from a dataset archive
(CUAD) is read out of the cached archive with the same magic and size checks.

**Resume.** Each accepted document is appended (and fsynced) to ``fetched.jsonl`` in the cache
before the next one starts; a PDF that reached its final name but not the journal is re-checked
and adopted without a new download. Re-running skips everything fetched or permanently dropped;
listings come from the cache. The manifest is rewritten atomically at the end of every run.

**Splits.** Per document and stable: ``sha256("split:" + id)`` read as a number in [0, 1) puts
a document in ``gold`` below 0.10, ``calib`` below 0.15 and ``train`` otherwise (about 10/5/85).
A document that already has a split in the manifest keeps it. Adding or removing documents never
moves another document, so gold documents can never drift into training. Stratification by
source holds in expectation; the realised shares per source are reported in the summary and the
README.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.training_corpus._net import (
    MAX_PDF_BYTES,
    PDF_MAGIC,
    FetchError,
    HostBlocked,
    Net,
    Rejected,
    redact,
)
from benchmarks.training_corpus.sources import ALL_SOURCES, MAKEUP_ORDER, OPTIONAL_SOURCES
from benchmarks.training_corpus.sources.base import Candidate, Source
from docrooter.taxonomies import family_for_doc_type

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CACHE = REPO / "benchmarks" / ".cache" / "training_corpus"
MANIFEST = HERE / "manifest.jsonl"
README = HERE / "README.md"
DEV_MANIFEST = REPO / "benchmarks" / "corpus" / "manifest.jsonl"
HELDOUT_MANIFEST = REPO / "benchmarks" / "heldout" / "manifest.jsonl"
PDF_META = HERE / "pdf_meta.py"

SPEC_TOTAL = 1100
SPLITS = ("train", "calib", "gold")
SPLIT_BOUNDS = (("gold", 0.10), ("calib", 0.15))  # cumulative; the rest (about 85%) is train
SAFE_ID = re.compile(r"[a-z0-9_-]{1,80}")
MIN_TEXT_CHARS = 200  # in the first pages read by pdf_meta: fewer means a scan
MAX_CONSECUTIVE_FAILURES = 10
INSPECT_TIMEOUT = 180
INSPECT_MEMORY_CAP = 2 << 30  # bytes, for the pdf_meta child
EXCLUSION_FIELDS = ("sha256", "url", "title")


# --- exclusion -----------------------------------------------------------------------------


def normalise_url(url: str) -> str:
    """Host without ``www.`` plus the unquoted path without a trailing slash: scheme, port,
    query and fragment do not distinguish documents here."""
    parts = urllib.parse.urlsplit(url.strip())
    host = (parts.hostname or "").lower().removeprefix("www.")
    path = urllib.parse.unquote(parts.path).rstrip("/")
    return f"{host}{path}"


def normalise_title(title: str) -> str:
    """Case-, accent-width- and punctuation-insensitive title."""
    text = unicodedata.normalize("NFKC", title).casefold()
    return " ".join(re.sub(r"[\W_]+", " ", text).split())


@dataclass
class Exclusions:
    """sha256s, normalised URLs and normalised titles of the dev and held-out documents, each
    mapped to the corpus name it came from."""

    sha256: dict[str, str] = field(default_factory=dict)
    urls: dict[str, str] = field(default_factory=dict)
    titles: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_manifests(cls, manifests: Mapping[str, Path]) -> Exclusions:
        """Read each manifest file (name -> path) for sha256, url and title only."""
        out = cls()
        for name, path in manifests.items():
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{name} manifest line {number} is not JSON") from exc
                kept = {key: row.get(key) for key in EXCLUSION_FIELDS}
                del row  # nothing beyond the three fields is kept
                if isinstance(kept["sha256"], str) and kept["sha256"]:
                    out.sha256.setdefault(kept["sha256"].lower(), name)
                if isinstance(kept["url"], str) and normalise_url(kept["url"]):
                    out.urls.setdefault(normalise_url(kept["url"]), name)
                if isinstance(kept["title"], str) and normalise_title(kept["title"]):
                    out.titles.setdefault(normalise_title(kept["title"]), name)
        return out

    def match_candidate(self, candidate: Candidate) -> str | None:
        """``"<corpus>:url"`` or ``"<corpus>:title"`` when the candidate is excluded."""
        corpus = self.urls.get(normalise_url(candidate.url))
        if corpus:
            return f"{corpus}:url"
        title = normalise_title(candidate.title)
        corpus = self.titles.get(title) if title else None
        return f"{corpus}:title" if corpus else None

    def match_sha256(self, sha256: str) -> str | None:
        corpus = self.sha256.get(sha256.lower())
        return f"{corpus}:sha256" if corpus else None

    def match_record(self, record: Mapping[str, Any]) -> str | None:
        """Any match of a fetched record: sha256, URL or title."""
        by_hash = self.match_sha256(str(record.get("sha256", "")))
        if by_hash:
            return by_hash
        url, title = str(record.get("url", "")), str(record.get("title", ""))
        corpus = self.urls.get(normalise_url(url))
        if corpus:
            return f"{corpus}:url"
        corpus = self.titles.get(normalise_title(title)) if normalise_title(title) else None
        return f"{corpus}:title" if corpus else None


def default_exclusions() -> Exclusions:
    return Exclusions.from_manifests({"dev": DEV_MANIFEST, "heldout": HELDOUT_MANIFEST})


# --- splits and quotas ---------------------------------------------------------------------


def split_for(doc_id: str) -> str:
    """The split of one document, from ``sha256("split:" + id)`` read as a number in [0, 1)."""
    digest = hashlib.sha256(f"split:{doc_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return next((name for name, bound in SPLIT_BOUNDS if value < bound), "train")


def assign_splits(
    records: Iterable[Mapping[str, Any]], existing: Mapping[str, str] | None = None
) -> dict[str, str]:
    """``id -> split``: a split already in ``existing`` (the manifest's) is kept; any other
    document gets :func:`split_for`. One document's split never depends on another's."""
    existing = existing or {}
    out = {}
    for record in records:
        doc_id = str(record["id"])
        kept = existing.get(doc_id)
        out[doc_id] = kept if kept in SPLITS else split_for(doc_id)
    return out


def split_shares(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per source: document counts per split and the realised percentages."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        counts[str(record["source"])][str(record["split"])] += 1
    out = {}
    for source, counter in counts.items():
        total = sum(counter.values())
        out[source] = {
            **{name: counter.get(name, 0) for name in SPLITS},
            "percent": {name: round(100 * counter.get(name, 0) / total, 1) for name in SPLITS},
        }
    return out


def scaled_quotas(sources: Sequence[Source], target: int) -> dict[str, int]:
    """Each source's share of ``target``, proportional to its spec target (largest remainder,
    ties to the earlier source), summing to ``target`` exactly."""
    total = sum(s.target for s in sources)
    raw = {s.name: target * s.target / total for s in sources}
    quotas = {name: int(value) for name, value in raw.items()}
    order = sorted(sources, key=lambda s: -(raw[s.name] - quotas[s.name]))
    for source in order[: target - sum(quotas.values())]:
        quotas[source.name] += 1
    return quotas


def scaled_cap(source: Source, target: int, quota: int) -> int:
    return max(quota, math.ceil(source.cap * target / SPEC_TOTAL))


# --- the journal ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Complete JSON lines of ``path``; a torn last line (a crash mid-append) is ignored."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --- PDF inspection ------------------------------------------------------------------------


@dataclass(frozen=True)
class PdfInfo:
    pages: int
    text_chars: int
    encrypted: bool = False


class InspectError(Exception):
    pass


def _rss_bytes(pid: int) -> int:
    """Resident size of process ``pid`` (0 when it cannot be read)."""
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return 0
    return int(out) * 1024 if out.isdigit() else 0


def inspect_pdf(path: Path) -> PdfInfo:
    """Page count and text-layer size of an untrusted PDF, read in an isolated child process
    (``python -I``: no environment variables, no user site, no script or working directory on
    ``sys.path``), with a timeout and a memory cap: the child lowers its own limits where the OS
    enforces them, and this parent kills it if its resident size passes the cap anyway."""
    with subprocess.Popen(
        [sys.executable, "-I", str(PDF_META), str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(PDF_META.parent),
    ) as child:
        deadline = time.monotonic() + INSPECT_TIMEOUT
        while True:
            try:
                stdout, stderr = child.communicate(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() > deadline:
                    child.kill()
                    child.communicate()
                    raise InspectError(f"timed out after {INSPECT_TIMEOUT} s") from None
                if _rss_bytes(child.pid) > INSPECT_MEMORY_CAP:
                    child.kill()
                    child.communicate()
                    raise InspectError("over the 2 GB memory cap") from None
    lines = stdout.strip().splitlines()
    if child.returncode != 0 or not lines:
        tail = (stderr.strip().splitlines() or ["no output"])[-1]
        raise InspectError(f"cannot read PDF: {tail[:200]}")
    try:
        info = json.loads(lines[-1])
        return PdfInfo(int(info["pages"]), int(info["text_chars"]), bool(info["encrypted"]))
    except (ValueError, KeyError, TypeError) as exc:
        raise InspectError(f"unexpected pdf_meta output: {lines[-1][:200]}") from exc


# --- the run -------------------------------------------------------------------------------


@dataclass
class SourceRun:
    name: str
    quota: int
    cap: int
    have: int = 0
    listed: int = 0
    status: str = "ok"  # ok | unavailable
    reason: str = ""
    exhausted: bool = False
    iterator: Iterator[Candidate] | None = None

    @property
    def made_up(self) -> int:
        return max(0, self.have - self.quota)

    @property
    def shortfall(self) -> int:
        return max(0, self.quota - self.have)


@dataclass
class Report:
    target: int
    dry_run: bool
    sources: dict[str, SourceRun]
    drops: Counter[str] = field(default_factory=Counter)
    chosen: list[Candidate] = field(default_factory=list)  # dry run: what would be downloaded
    records: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "dry_run": self.dry_run,
            "documents": sum(s.have for s in self.sources.values()),
            "sources": {
                name: {
                    "quota": s.quota,
                    "cap": s.cap,
                    "documents": s.have,
                    "made_up": s.made_up,
                    "shortfall": s.shortfall,
                    "listed": s.listed,
                    "status": s.status,
                    "reason": s.reason,
                }
                for name, s in self.sources.items()
            },
            "drops": dict(sorted(self.drops.items())),
        }

    def table(self) -> str:
        rows = [
            "| Source | Scaled target | Documents | Made up | Shortfall | Status |",
            "| - | - | - | - | - | - |",
        ]
        for name, s in self.sources.items():
            status = s.status if not s.reason else f"{s.status}: {s.reason}"
            rows.append(
                f"| {name} | {s.quota} | {s.have} | {s.made_up} | {s.shortfall} | {status} |"
            )
        total = sum(s.have for s in self.sources.values())
        rows.append(f"| **total** | {self.target} | {total} | | | |")
        return "\n".join(rows)


class Fetcher:
    def __init__(
        self,
        sources: Sequence[Source],
        target: int,
        *,
        net: Net,
        exclusions: Exclusions,
        cache_dir: Path = CACHE,
        manifest_path: Path = MANIFEST,
        readme_path: Path | None = None,
        dry_run: bool = False,
        inspect: Callable[[Path], PdfInfo] = inspect_pdf,
        log: Callable[[str], None] = print,
        now: Callable[[], str] | None = None,
    ) -> None:
        self.sources = list(sources)
        self.target = target
        self.net = net
        self.exclusions = exclusions
        self.cache_dir = cache_dir
        self.pdf_dir = cache_dir / "pdfs"
        self.manifest_path = manifest_path
        self.readme_path = readme_path
        self.dry_run = dry_run
        self.inspect = inspect
        self.log = log
        self.now = now or (lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        self.fetched_path = cache_dir / "fetched.jsonl"
        self.drops_path = cache_dir / "drops.jsonl"

    # -- state

    def _load(self) -> None:
        self.fetched: dict[str, dict[str, Any]] = {}
        for row in _read_jsonl(self.fetched_path):
            pdf = self.pdf_dir / f"{row.get('id')}.pdf"
            if pdf.exists() and pdf.stat().st_size == row.get("bytes"):
                self.fetched[str(row["id"])] = row  # a missing or altered PDF is fetched again
        self.shas = {str(r["sha256"]): str(r["id"]) for r in self.fetched.values()}
        drops = [r for r in _read_jsonl(self.drops_path) if r.get("permanent")]
        self.permanent_drops = {str(r["id"]) for r in drops if "id" in r}
        self.heldout_drop_keys = {str(r["key"]) for r in drops if "key" in r}
        self.existing_splits = {
            str(r["id"]): str(r["split"])
            for r in _read_jsonl(self.manifest_path)
            if "id" in r and r.get("split") in SPLITS
        }

    def _dropped_before(self, candidate: Candidate) -> bool:
        return (
            candidate.id in self.permanent_drops
            or _drop_key(candidate.id) in self.heldout_drop_keys
        )

    def run(self) -> Report:
        self._load()
        quotas = scaled_quotas(self.sources, self.target)
        report = Report(
            self.target,
            self.dry_run,
            {
                s.name: SourceRun(
                    s.name, quotas[s.name], scaled_cap(s, self.target, quotas[s.name])
                )
                for s in self.sources
            },
        )
        for run in report.sources.values():
            run.have = sum(1 for r in self.fetched.values() if r.get("source") == run.name)
        by_name = {s.name: s for s in self.sources}
        try:
            for source in self.sources:
                state = report.sources[source.name]
                if self._preflight(source, state):
                    self._fill(source, state, state.quota, report)
            deficit = self.target - sum(s.have for s in report.sources.values())
            for name in MAKEUP_ORDER:
                if deficit <= 0:
                    break
                extra = report.sources.get(name)
                if extra is None or extra.status != "ok" or extra.exhausted:
                    continue
                before = extra.have
                self._fill(by_name[name], extra, min(extra.cap, extra.have + deficit), report)
                deficit -= extra.have - before
        finally:
            if not self.dry_run:
                report.records = self._write_manifest(report)
                self._write_summary(report)
        if self.dry_run:
            _write_atomic(
                self.cache_dir / "dry_run.jsonl",
                "".join(
                    json.dumps(_candidate_row(c), sort_keys=True) + "\n" for c in report.chosen
                ),
            )
        return report

    def _preflight(self, source: Source, state: SourceRun) -> bool:
        """robots.txt of the host the PDFs come from: one request per host. An unreadable
        robots.txt makes the source unavailable for this run only."""
        if source.probe_url is None:
            return True
        try:
            if self.net.allowed(source.probe_url):
                return True
            why = "robots.txt disallows it"
        except FetchError as exc:
            why = str(exc)
        state.status = "unavailable"
        state.reason = f"{redact(source.probe_url)}: {why}"
        self.log(f"SKIP     {source.name}: {state.reason}")
        return False

    def _fill(self, source: Source, state: SourceRun, want: int, report: Report) -> None:
        if state.iterator is None:
            state.iterator = iter(source.candidates(self.net))
        failures = 0
        while state.have < want:
            try:
                candidate = next(state.iterator)
            except StopIteration:
                state.exhausted = True
                return
            except HostBlocked as exc:
                state.status, state.reason = "unavailable", f"listing refused: {exc}"
                self.log(f"SKIP     {source.name}: {state.reason}")
                return
            except (FetchError, Rejected, ValueError, ET.ParseError, RecursionError) as exc:
                state.exhausted, state.reason = True, f"listing failed: {type(exc).__name__} {exc}"
                self.log(f"STOP     {source.name}: {state.reason}")
                return
            state.listed += 1
            if candidate.id in self.fetched or self._dropped_before(candidate):
                continue
            if not SAFE_ID.fullmatch(candidate.id):  # ids name files: no paths, no surprises
                report.drops["unsafe-id"] += 1
                self.log(f"DROP     {candidate.source}: unsafe id {candidate.id[:100]!r}")
                continue
            try:
                outcome = self._take(candidate, report)
            except RecursionError:  # defensive: one bad URL must never end the run
                (self.pdf_dir / f"{candidate.id}.pdf.part").unlink(missing_ok=True)
                outcome = self._drop(
                    candidate, "failed: RecursionError (transient)", report, permanent=False
                )
            if outcome == "ok":
                state.have += 1
                failures = 0
            elif outcome == "blocked":
                why = self.net.blocked(candidate.url) or "host refused downloads"
                if source.probe_url is None or _host(candidate.url) != _host(source.probe_url):
                    # Another of the source's hosts (a second fund sponsor) still works.
                    self.log(f"BLOCKED  {source.name} {candidate.id}: {why}")
                    continue
                state.status, state.reason = "unavailable", why
                self.log(f"SKIP     {source.name}: {state.reason}")
                return
            elif outcome == "failed":
                failures += 1
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    state.exhausted = True
                    state.reason = f"{failures} downloads failed in a row; retry later"
                    self.log(f"STOP     {source.name}: {state.reason}")
                    return

    def _drop(
        self, candidate: Candidate, reason: str, report: Report, *, permanent: bool = True
    ) -> str:
        if reason.startswith("heldout"):
            # Counted only: no id, URL, title or matching field, on disk or in the log.
            report.drops["heldout"] += 1
            self.heldout_drop_keys.add(_drop_key(candidate.id))
            if not self.dry_run:
                _append_jsonl(self.drops_path, {"key": _drop_key(candidate.id), "permanent": True})
            return "dropped"
        report.drops[reason if reason.startswith("dev:") else reason.split(":", 1)[0]] += 1
        row: dict[str, Any] = {
            "id": candidate.id,
            "source": candidate.source,
            "url": candidate.url,
            "reason": reason,
            "permanent": permanent,
            "at": self.now(),
        }
        if permanent:
            self.permanent_drops.add(candidate.id)
        if not self.dry_run:
            _append_jsonl(self.drops_path, row)
        self.log(f"DROP     {candidate.source} {candidate.id} {reason} {redact(candidate.url)}")
        return "dropped" if permanent else "failed"

    def _take(self, candidate: Candidate, report: Report) -> str:
        """Download, check and record one candidate: ``ok``, ``dropped``, ``failed`` (transient)
        or ``blocked`` (the host refused us)."""
        excluded = self.exclusions.match_candidate(candidate)
        if excluded:
            return self._drop(candidate, excluded, report)
        if self.dry_run:
            report.chosen.append(candidate)
            return "ok"
        dest = self.pdf_dir / f"{candidate.id}.pdf"
        if not dest.exists():
            try:
                if candidate.archive is not None:
                    extract_pdf(candidate.archive, candidate.archive_member or "", dest)
                else:
                    self.net.download_pdf(candidate.url, dest)
            except Rejected as exc:
                return self._drop(candidate, f"rejected: {exc}", report)
            except HostBlocked:
                return "blocked"
            except FetchError as exc:
                return self._drop(candidate, f"failed: {exc}", report, permanent=False)
        size = dest.stat().st_size
        with dest.open("rb") as f:
            head = f.read(len(PDF_MAGIC))
        if size > MAX_PDF_BYTES or not head.startswith(PDF_MAGIC):
            dest.unlink()
            return self._drop(candidate, "rejected: not a PDF under the size cap", report)
        sha256 = _sha256(dest)
        excluded = self.exclusions.match_sha256(sha256)
        if excluded:
            dest.unlink()
            return self._drop(candidate, excluded, report)
        if sha256 in self.shas:
            dest.unlink()
            return self._drop(candidate, f"duplicate: same bytes as {self.shas[sha256]}", report)
        try:
            info = self.inspect(dest)
        except InspectError as exc:
            dest.unlink()
            return self._drop(candidate, f"unreadable: {exc}", report)
        if info.encrypted or info.pages < 1 or info.text_chars < MIN_TEXT_CHARS:
            dest.unlink()
            why = "encrypted" if info.encrypted else f"{info.text_chars} text chars"
            return self._drop(candidate, f"no-text-layer: {why}", report)
        record = self._record(candidate, sha256, size, info.pages)
        _append_jsonl(self.fetched_path, record)  # the per-document checkpoint
        self.fetched[candidate.id] = record
        self.shas[sha256] = candidate.id
        self.log(f"OK       {candidate.source} {candidate.id} {info.pages} pages {size} bytes")
        return "ok"

    def _record(self, candidate: Candidate, sha256: str, size: int, pages: int) -> dict[str, Any]:
        record: dict[str, Any] = {
            "id": candidate.id,
            "url": candidate.url,
            "source": candidate.source,
            "licence": candidate.licence,
            "sha256": sha256,
            "bytes": size,
            "pages": pages,
            "fetched_at": self.now(),
            "doc_type_hint": candidate.doc_type_hint,
            "family_hint": candidate.family_hint or family_for_doc_type(candidate.doc_type_hint),
            "title": candidate.title,
            "publisher": candidate.publisher,
        }
        record.update({k: v for k, v in candidate.extra.items() if k not in record})
        return record

    def _write_manifest(self, report: Report) -> list[dict[str, Any]]:
        order = {s.name: i for i, s in enumerate(ALL_SOURCES + OPTIONAL_SOURCES)}
        records = []
        for record in self.fetched.values():
            excluded = self.exclusions.match_record(record)  # the exclusion lists may have grown
            if excluded:
                key = "heldout" if excluded.startswith("heldout") else f"manifest-{excluded}"
                report.drops[key] += 1
                continue
            records.append(record)
        records.sort(key=lambda r: (order.get(r["source"], len(order)), r["id"]))
        splits = assign_splits(records, self.existing_splits)
        rows = [{**r, "split": splits[r["id"]]} for r in records]
        _write_atomic(
            self.manifest_path,
            "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
        )
        return rows

    def _write_summary(self, report: Report) -> None:
        summary = report.summary()
        summary["manifest"] = {
            "documents": len(report.records),
            "by_source": dict(Counter(r["source"] for r in report.records)),
            "by_split": dict(Counter(r["split"] for r in report.records)),
            "split_shares_by_source": split_shares(report.records),
        }
        _write_atomic(self.cache_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
        if self.readme_path is not None and self.readme_path.exists():
            update_readme_counts(self.readme_path, report, summary)


def _host(url: str) -> str:
    return (urllib.parse.urlsplit(url).hostname or "").lower()


def _drop_key(doc_id: str) -> str:
    return hashlib.sha256(f"drop:{doc_id}".encode()).hexdigest()


def extract_pdf(archive: Path, member: str, dest: Path, max_bytes: int = MAX_PDF_BYTES) -> int:
    """Copy one PDF out of a cached dataset archive (untrusted data) to ``dest``: only the named
    member, streamed with the ``%PDF-`` magic and size checks of a download, written to
    ``.part`` and renamed. The archive's own paths never become file names here."""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as zf:
            info = zf.getinfo(member)
            if info.file_size > max_bytes:
                raise Rejected(f"archive member is over the {max_bytes}-byte cap")
            with zf.open(info) as src, part.open("wb") as out:
                head = src.read(len(PDF_MAGIC))
                if not head.startswith(PDF_MAGIC):
                    raise Rejected("archive member is not a PDF (no %PDF- magic)")
                size = len(head)
                out.write(head)
                while block := src.read(1 << 20):
                    size += len(block)
                    if size > max_bytes:
                        raise Rejected(f"archive member is over the {max_bytes}-byte cap")
                    out.write(block)
                out.flush()
                os.fsync(out.fileno())
    except (KeyError, zipfile.BadZipFile, OSError, EOFError) as exc:
        part.unlink(missing_ok=True)
        raise Rejected(f"cannot read archive member: {type(exc).__name__}") from None
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    os.replace(part, dest)
    return size


def _candidate_row(candidate: Candidate) -> dict[str, Any]:
    return {
        "id": candidate.id,
        "source": candidate.source,
        "url": candidate.url,
        "title": candidate.title,
        "doc_type_hint": candidate.doc_type_hint,
        "licence": candidate.licence,
    }


def split_table(shares: Mapping[str, Mapping[str, Any]]) -> str:
    rows = ["| Source | train | calib | gold | gold % |", "| - | - | - | - | - |"]
    for source, row in shares.items():
        rows.append(
            f"| {source} | {row['train']} | {row['calib']} | {row['gold']} "
            f"| {row['percent']['gold']} |"
        )
    return "\n".join(rows)


COUNTS_BEGIN = "<!-- counts:begin -->"
COUNTS_END = "<!-- counts:end -->"


def update_readme_counts(readme: Path, report: Report, summary: Mapping[str, Any]) -> None:
    """Replace the README's counts block with this run's table, splits and drops."""
    text = readme.read_text(encoding="utf-8")
    if COUNTS_BEGIN not in text or COUNTS_END not in text:
        return
    by_split = summary["manifest"]["by_split"]
    drops = ", ".join(f"{k} {v}" for k, v in sorted(report.drops.items())) or "none"
    block = "\n".join(
        [
            COUNTS_BEGIN,
            "",
            f"Last fetch: target {report.target}, {len(report.records)} documents in the "
            f"manifest (train {by_split.get('train', 0)}, calib {by_split.get('calib', 0)}, "
            f"gold {by_split.get('gold', 0)}).",
            "",
            report.table(),
            "",
            "Realised splits per source (hash-assigned, so stratified in expectation):",
            "",
            split_table(summary["manifest"]["split_shares_by_source"]),
            "",
            f"Drops this run: {drops}.",
            "",
            COUNTS_END,
        ]
    )
    head, rest = text.split(COUNTS_BEGIN, 1)
    tail = rest.split(COUNTS_END, 1)[1]
    _write_atomic(readme, head + block + tail)


def main(argv: Sequence[str] | None = None) -> int:
    names = [s.name for s in ALL_SOURCES + OPTIONAL_SOURCES]
    default = [s.name for s in ALL_SOURCES]
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--target", type=int, default=SPEC_TOTAL, help="documents to reach")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=names,
        default=default,
        metavar="SOURCE",
        help=f"default: {' '.join(default)}; also available: edgar_ars edgar_ex10",
    )
    parser.add_argument("--dry-run", action="store_true", help="list candidates, download none")
    parser.add_argument("--cache", type=Path, default=CACHE, help="PDFs, journal and listings")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args(argv)
    if args.target < 1:
        parser.error("--target must be positive")
    sources = [s for s in ALL_SOURCES + OPTIONAL_SOURCES if s.name in set(args.sources)]
    fetcher = Fetcher(
        sources,
        args.target,
        net=Net(args.cache / "listings", archive_dir=args.cache / "archives"),
        exclusions=default_exclusions(),
        cache_dir=args.cache,
        manifest_path=args.manifest,
        readme_path=README if args.manifest == MANIFEST else None,
        dry_run=args.dry_run,
    )
    report = fetcher.run()
    print()
    print(report.table())
    print(f"drops: {dict(sorted(report.drops.items()))}")
    if args.dry_run:
        print(f"dry run: {len(report.chosen)} candidates in {args.cache / 'dry_run.jsonl'}")
    else:
        print(f"manifest: {args.manifest} ({len(report.records)} documents)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
