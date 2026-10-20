"""Pick the sections of the 113 gold-split documents for the section gold (spec G5, task V5).

    PYTHONPATH=src python -m benchmarks.classification.section_sample

Every document whose manifest split is ``gold`` is read with the reading the benchmark uses:
profile ``fast`` (no elements, no OCR, no layout model; headings and sections only). From each,
:func:`sample_sections` takes up to ``max_per_doc`` level-1 and level-2 sections, spread
through the document in reading order, with half the cap reserved for level 1. The choice is
a function of the document alone, so the same reading gives the same sample; the gold names a
section by document, section id and the SHA-1 of its title, so a reading that drifts is caught
when the gold is used. Items are numbered after a seeded shuffle across documents.

The sample (``benchmarks/.cache.nosync/classification/section_sample.jsonl`` by default) holds
section text from documents that are never redistributed: it stays in the git-ignored cache and
is never committed. Keep the file the annotations were made against: their item ids refer to it.
A summary of counts only (``section_sample_summary.json``) is committed with the gold, with the
reading settings the sample was made under.

The output rows hold ``item`` (an opaque id the annotators use), ``doc_id``, ``section_id``,
``title_sha1``, ``level``, ``title``, ``heading_path`` (the titles of the section's ancestors),
``text`` (the opening of the section's own text) and ``double`` (marked for double annotation,
20% of the items). Annotators see only the item, level, title, heading path and text
(``section_gold.annotator_view``): never the document id, its type or any model output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from benchmarks.training_corpus.gold_sample import jsonl, write_atomic
from docrooter import __version__, process
from docrooter.document import Document, Section
from docrooter.rooter import profile_defaults

HERE = Path(__file__).resolve().parent
BENCHMARKS = HERE.parent
MANIFEST_PATH = BENCHMARKS / "training_corpus" / "manifest.jsonl"
# The PDFs are never read through the ``benchmarks/.cache`` symlink: it may point elsewhere.
PDF_DIR = BENCHMARKS / ".cache.nosync" / "training_corpus" / "pdfs"
CACHE_DIR = BENCHMARKS / ".cache.nosync" / "classification"
SAMPLE_PATH = CACHE_DIR / "section_sample.jsonl"
SUMMARY_PATH = HERE / "section_sample_summary.json"

LEVELS = (1, 2)
MAX_PER_DOC = 12
OPENING_CHARS = 600  # how much of a section's own text the annotator sees
DOUBLE_FRACTION = 0.2  # spec G5: 20% double-annotated for kappa
SEED = 0
PROFILE = "fast"
# Reading only: the profile's classification tasks are switched off (they load no model here),
# which does not change the document that is read.
READ_OVERRIDES: dict[str, Any] = {"doc_type": "none", "sections": "none", "summaries": "none"}


def title_sha1(title: str) -> str:
    """The SHA-1 of a section's title: what the gold keeps in place of the title."""
    return hashlib.sha1(title.encode("utf-8")).hexdigest()


def reading_metadata() -> dict[str, Any]:
    """The reading the sample (and the benchmark) uses, to be recorded in the gold metadata."""
    return {
        "profile": PROFILE,
        "settings": {**profile_defaults(PROFILE), **READ_OVERRIDES},
        "elements": False,
        "docrooter_version": __version__,
        "levels": list(LEVELS),
        "opening_chars": OPENING_CHARS,
    }


def _spread(items: Sequence[Section], k: int, rng: random.Random) -> list[Section]:
    """``k`` of ``items``, one from each of ``k`` equal runs (a seeded position within the run,
    so a structure that repeats with the run's period is not hit at the same place every time);
    all of them if there are no more than ``k``."""
    n = len(items)
    if k >= n:
        return list(items)
    return [items[rng.randrange(i * n // k, (i + 1) * n // k)] for i in range(k)]


def sample_sections(doc: Document, *, max_per_doc: int = MAX_PER_DOC) -> list[Section]:
    """Up to ``max_per_doc`` level-1 and level-2 sections of ``doc``, in reading order.

    Sections without a title are skipped, and so are repeats of a title already seen in the
    document (a running page header read as a heading). Up to half the cap (at least one slot)
    is reserved for level 1, whose sections (Item 1A, Abstract, Article 4, ...) are the ones
    the doc-type rules decide on; the rest goes to level 2. A level with fewer sections than its
    share leaves the slots to the other. Within a level the picks are spread through the
    document, one from each of equal runs. Deterministic: a function of the document alone."""
    if max_per_doc < 1:
        raise ValueError(f"max_per_doc must be positive, got {max_per_doc}")
    seen: set[str] = set()
    levels: dict[int, list[Section]] = {level: [] for level in LEVELS}
    for s in doc.walk():
        if s.level in LEVELS and s.title.strip() and title_sha1(s.title) not in seen:
            seen.add(title_sha1(s.title))
            levels[s.level].append(s)
    first, second = levels[LEVELS[0]], levels[LEVELS[1]]
    take_first = min(len(first), max(1, max_per_doc // 2))
    take_second = min(len(second), max_per_doc - take_first)
    take_first = min(len(first), max_per_doc - take_second)
    key = "\n".join(f"{s.id}\t{s.title}" for s in (*first, *second))
    rng = random.Random(hashlib.sha1(key.encode("utf-8")).hexdigest())
    picked = _spread(first, take_first, rng) + _spread(second, take_second, rng)
    order = {s.id: k for k, s in enumerate(doc.walk())}
    return sorted(picked, key=lambda s: order[s.id])


def heading_path(doc: Document, section: Section) -> list[str]:
    """The titles of the section's ancestors, outermost first (not the section's own)."""
    path: list[str] = []
    parent_id = section.parent_id
    while parent_id is not None:
        parent = doc.sections[parent_id]
        if parent.level > 0 and parent.title.strip():
            path.append(parent.title)
        parent_id = parent.parent_id
    path.reverse()
    return path


def opening(section: Section, chars: int = OPENING_CHARS) -> str:
    """The first ``chars`` characters of the section's own text, whitespace collapsed."""
    return " ".join(section.text.split())[:chars]


def section_row(doc_id: str, doc: Document, section: Section) -> dict[str, Any]:
    """One sample row (without ``item`` and ``double``, which :func:`assign_items` adds)."""
    return {
        "doc_id": doc_id,
        "section_id": section.id,
        "title_sha1": title_sha1(section.title),
        "level": section.level,
        "title": section.title,
        "heading_path": heading_path(doc, section),
        "text": opening(section),
    }


def assign_items(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int = SEED,
    double_fraction: float = DOUBLE_FRACTION,
) -> list[dict[str, Any]]:
    """Shuffle ``rows`` by ``seed``, number them (``s0000``, ``s0001``, ...) and mark
    ``round(double_fraction * len(rows))`` of them, chosen by ``seed``, as ``double``."""
    if not 0 <= double_fraction <= 1:
        raise ValueError(f"double_fraction must be in [0, 1], got {double_fraction}")
    # Shuffled before numbering, so the order of the items (and of the batches) does not show
    # which sections come from one document, or in what order.
    shuffled = list(rows)
    random.Random(f"section_gold:{seed}:order").shuffle(shuffled)
    items = [f"s{k:04d}" for k in range(len(shuffled))]
    double = set(
        random.Random(f"section_gold:{seed}:double").sample(
            items, round(double_fraction * len(items))
        )
    )
    return [
        {"item": i, **row, "double": i in double} for i, row in zip(items, shuffled, strict=True)
    ]


def gold_documents(manifest: Path = MANIFEST_PATH) -> list[str]:
    """The ids of the manifest's gold-split documents, sorted."""
    ids: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("split") == "gold":
                ids.append(record["id"])
    return sorted(ids)


def read_document(path: Path) -> Document:
    """Read a PDF as the benchmark does: profile ``fast``, no classification (headings and
    sections only)."""
    analysis = process(path, profile=PROFILE, **READ_OVERRIDES)
    return analysis.document


def build_sample(
    docs: Iterable[tuple[str, Document]],
    *,
    max_per_doc: int = MAX_PER_DOC,
    seed: int = SEED,
    double_fraction: float = DOUBLE_FRACTION,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The sample rows and a counts-only summary for ``(doc_id, document)`` pairs."""
    rows: list[dict[str, Any]] = []
    per_doc: dict[str, dict[str, int]] = {}
    for doc_id, doc in sorted(docs, key=lambda pair: pair[0]):
        candidates = sum(1 for s in doc.walk() if s.level in LEVELS and s.title.strip())
        picked = sample_sections(doc, max_per_doc=max_per_doc)
        rows += [section_row(doc_id, doc, s) for s in picked]
        per_doc[doc_id] = {"candidates": candidates, "selected": len(picked)}
    items = assign_items(rows, seed=seed, double_fraction=double_fraction)
    summary = {
        "reading": reading_metadata(),
        "seed": seed,
        "max_per_doc": max_per_doc,
        "double_fraction": double_fraction,
        "documents": len(per_doc),
        "documents_without_sections": sorted(d for d, c in per_doc.items() if not c["selected"]),
        "selected": len(items),
        "double_annotated": sum(r["double"] for r in items),
        "by_level": {str(level): sum(r["level"] == level for r in items) for level in LEVELS},
        "per_document": per_doc,
    }
    return items, summary


def require_untracked(path: Path) -> None:
    """Refuse a path inside ``benchmarks/`` other than its git-ignored caches: files that hold
    document text must never be written where they could be committed."""
    resolved = path.resolve()
    if not resolved.is_relative_to(BENCHMARKS):
        return
    for cache in (".cache", ".cache.nosync"):
        if resolved.is_relative_to(BENCHMARKS / cache):
            return
    raise ValueError(
        f"{path} is inside benchmarks/ outside its git-ignored caches: files that hold document "
        f"text go under {CACHE_DIR}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--pdf-dir", type=Path, default=PDF_DIR)
    parser.add_argument("--out", type=Path, default=SAMPLE_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--max-per-doc", type=int, default=MAX_PER_DOC)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    if args.max_per_doc < 1:
        parser.error("--max-per-doc must be positive")
    try:
        require_untracked(args.out)
    except ValueError as exc:
        parser.error(str(exc))
    ids = gold_documents(args.manifest)
    missing = [i for i in ids if not (args.pdf_dir / f"{i}.pdf").is_file()]
    if missing:
        parser.error(f"{len(missing)} gold PDF(s) not in {args.pdf_dir}: {missing[:5]}")
    docs: list[tuple[str, Document]] = []
    for n, doc_id in enumerate(ids, 1):
        print(f"[{n}/{len(ids)}] {doc_id}", file=sys.stderr, flush=True)
        docs.append((doc_id, read_document(args.pdf_dir / f"{doc_id}.pdf")))
    rows, summary = build_sample(docs, max_per_doc=args.max_per_doc, seed=args.seed)
    write_atomic(args.out, jsonl(rows))
    write_atomic(args.summary, json.dumps(summary, indent=2) + "\n")
    print(
        f"{summary['selected']} sections from {summary['documents']} documents "
        f"({summary['double_annotated']} double); "
        f"{len(summary['documents_without_sections'])} without level-1/2 sections"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
