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
matches (the same bytes re-hosted elsewhere) is deleted and dropped. Spec §5 matches on
(publisher, normalised title); the controller ruling limits the manifests to sha256, url and
title, so titles are matched without the publisher, which can only drop more, never less. Drops
are recorded in the cache (``drops.jsonl``); a held-out drop is recorded without its URL or title.

**Checks.** Every download must start with ``%PDF-`` and be at most 50 MB. Its page count and
text layer are read by :mod:`benchmarks.training_corpus.pdf_meta` in a ``python -I`` child
process; a PDF with no usable text layer in its first pages (a scan) is dropped, since the corpus
is native PDFs only. Byte-identical documents are kept once.

**Resume.** Each accepted document is appended (and fsynced) to ``fetched.jsonl`` in the cache
before the next one starts; a PDF that reached its final name but not the journal is re-checked
and adopted without a new download. Re-running skips everything fetched or permanently dropped;
listings come from the cache. The manifest is rewritten atomically at the end of every run.

**Splits.** Assigned by document over the whole manifest, stratified by source, with seed 0:
about 85% ``train``, 5% ``calib`` and 10% ``gold`` per source. They are a pure function of the
set of documents, so the full fetch fixes them; the gold set (L5) is drawn after it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
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
from benchmarks.training_corpus.sources import ALL_SOURCES, MAKEUP_ORDER
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
SEED = 0
SPLIT_SHARES = (("gold", 0.10), ("calib", 0.05))  # the rest is train (about 85%)
MIN_TEXT_CHARS = 200  # in the first pages read by pdf_meta: fewer means a scan
MAX_CONSECUTIVE_FAILURES = 10
INSPECT_TIMEOUT = 180
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


def default_exclusions() -> Exclusions:
    return Exclusions.from_manifests({"dev": DEV_MANIFEST, "heldout": HELDOUT_MANIFEST})


# --- splits and quotas ---------------------------------------------------------------------


def assign_splits(records: Iterable[Mapping[str, Any]], seed: int = SEED) -> dict[str, str]:
    """``id -> split`` by document, stratified by source: per source, ids sorted then shuffled
    with ``seed``; the first 10% (rounded) are ``gold``, the next 5% ``calib``, the rest
    ``train``. Independent of input order."""
    by_source: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_source[str(record["source"])].append(str(record["id"]))
    splits: dict[str, str] = {}
    for source in sorted(by_source):
        ids = sorted(by_source[source])
        random.Random(f"{seed}:{source}").shuffle(ids)
        bounds, start = [], 0
        for name, share in SPLIT_SHARES:
            count = int(len(ids) * share + 0.5)
            bounds.append((name, start, start + count))
            start += count
        for index, doc_id in enumerate(ids):
            splits[doc_id] = next((n for n, a, b in bounds if a <= index < b), "train")
    return splits


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


def inspect_pdf(path: Path) -> PdfInfo:
    """Page count and text-layer size of an untrusted PDF, read in an isolated child process
    (``python -I``: no environment variables, no user site, no script or working directory on
    ``sys.path``), with a timeout."""
    try:
        done = subprocess.run(
            [sys.executable, "-I", str(PDF_META), str(path)],
            capture_output=True,
            text=True,
            timeout=INSPECT_TIMEOUT,
            cwd=str(PDF_META.parent),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise InspectError(f"timed out after {INSPECT_TIMEOUT} s") from exc
    lines = done.stdout.strip().splitlines()
    if done.returncode != 0 or not lines:
        tail = (done.stderr.strip().splitlines() or ["no output"])[-1]
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
        self.permanent_drops = {
            str(r["id"]) for r in _read_jsonl(self.drops_path) if r.get("permanent")
        }

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
                report.records = self._write_manifest()
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
        """robots.txt of the host the PDFs come from: one request per host."""
        if self.net.allowed(source.probe_url):
            return True
        state.status = "unavailable"
        state.reason = f"{redact(source.probe_url)}: {self.net.robots_status(source.probe_url)}"
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
            except (FetchError, Rejected, ValueError, ET.ParseError) as exc:
                state.exhausted, state.reason = True, f"listing failed: {exc}"
                self.log(f"STOP     {source.name}: {state.reason}")
                return
            state.listed += 1
            if candidate.id in self.fetched or candidate.id in self.permanent_drops:
                continue
            outcome = self._take(candidate, report)
            if outcome == "ok":
                state.have += 1
                failures = 0
            elif outcome == "blocked":
                why = self.net.blocked(candidate.url) or "host refused downloads"
                if _host(candidate.url) != _host(source.probe_url):
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
        excluded = reason.startswith(("dev:", "heldout:"))
        report.drops[reason if excluded else reason.split(":", 1)[0]] += 1
        row: dict[str, Any] = {
            "id": candidate.id,
            "source": candidate.source,
            "reason": reason,
            "permanent": permanent,
            "at": self.now(),
        }
        if not reason.startswith("heldout"):
            row["url"] = candidate.url  # held-out matches are recorded without URL or title
        if permanent:
            self.permanent_drops.add(candidate.id)
        if not self.dry_run:
            _append_jsonl(self.drops_path, row)
        shown = "(held-out match)" if reason.startswith("heldout") else redact(candidate.url)
        self.log(f"DROP     {candidate.source} {candidate.id} {reason} {shown}")
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

    def _write_manifest(self) -> list[dict[str, Any]]:
        order = {s.name: i for i, s in enumerate(ALL_SOURCES)}
        records = sorted(
            self.fetched.values(), key=lambda r: (order.get(r["source"], len(order)), r["id"])
        )
        splits = assign_splits(records)
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
        }
        _write_atomic(self.cache_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
        if self.readme_path is not None and self.readme_path.exists():
            update_readme_counts(self.readme_path, report, summary)


def _host(url: str) -> str:
    return (urllib.parse.urlsplit(url).hostname or "").lower()


def _candidate_row(candidate: Candidate) -> dict[str, Any]:
    return {
        "id": candidate.id,
        "source": candidate.source,
        "url": candidate.url,
        "title": candidate.title,
        "doc_type_hint": candidate.doc_type_hint,
        "licence": candidate.licence,
    }


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
            f"Drops this run: {drops}.",
            "",
            COUNTS_END,
        ]
    )
    head, rest = text.split(COUNTS_BEGIN, 1)
    tail = rest.split(COUNTS_END, 1)[1]
    _write_atomic(readme, head + block + tail)


def main(argv: Sequence[str] | None = None) -> int:
    names = [s.name for s in ALL_SOURCES]
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--target", type=int, default=SPEC_TOTAL, help="documents to reach")
    parser.add_argument("--sources", nargs="+", choices=names, default=names, metavar="SOURCE")
    parser.add_argument("--dry-run", action="store_true", help="list candidates, download none")
    parser.add_argument("--cache", type=Path, default=CACHE, help="PDFs, journal and listings")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args(argv)
    if args.target < 1:
        parser.error("--target must be positive")
    sources = [s for s in ALL_SOURCES if s.name in set(args.sources)]
    fetcher = Fetcher(
        sources,
        args.target,
        net=Net(args.cache / "listings"),
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
