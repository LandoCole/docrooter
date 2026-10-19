"""Pick the 900 gold-set elements from the corpus store's gold split (spec §11, task L5).

    PYTHONPATH=src python -m benchmarks.training_corpus.gold_sample --corpus DIR

Only documents whose manifest split is ``gold`` are read, so no train or calib document can
enter the gold set. The quota is split into cells by family group (spec §11):

| Cell | Documents | Quota |
| - | - | - |
| ``annual_report`` | element family ``annual_report`` | 300 |
| ``fund_document`` | element family ``fund_document`` | 150 |
| ``research_paper`` | element family ``research_paper`` | 150 |
| ``government_legal.generic`` | family ``generic`` from GAO, CRS, Fed and USCOURTS | 100 |
| ``government_legal.contract`` | element family ``contract`` | 50 |
| ``technical`` | family ``technical_document``, and RFCs (family ``generic``) | 150 |

Within a cell, elements are stratified by kind (proportional to the cell's pool, with a floor
of ``MIN_PER_KIND`` per kind present) and spread over documents: each pick goes to the
document with the fewest picks so far, and no document gives more than ``MAX_DOC_SHARE`` of the
cell's quota. A cell with too few gold documents keeps what it has, and the shortfall is
reported; nothing is borrowed from another split or cell. Blank elements and repeated texts
within a cell are skipped. ``DOUBLE_FRACTION`` of each cell is marked for double annotation.

The output rows hold ``item`` (an opaque id the annotators use), ``doc_id``, ``element_id``,
``text_sha1``, ``kind``, ``text``, ``heading_path``, ``family`` (the element topic family the
topic must come from), ``group`` and ``double``. Annotators see only the item, text, kind,
heading path and family (``gold.annotator_view``); never any model output.

Spec §11 also names the teacher-free zero-shot topic as a stratum. It is not used: computing
it needs an embedding model, and this sampler loads none (ruling recorded in the L5 report).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docrooter.corpus import Corpus, CorpusElement, ManifestRecord
from docrooter.taxonomies import family_for_doc_type

HERE = Path(__file__).resolve().parent
SAMPLE_PATH = HERE / "gold_sample.jsonl"
SUMMARY_PATH = HERE / "gold_sample_summary.json"

SEED = 0
MAX_DOC_SHARE = 0.2  # at most this share of a cell's quota from one document
MIN_PER_KIND = 5  # floor per element kind present in a cell's pool
DOUBLE_FRACTION = 0.2  # spec §11: 20% (180 of 900) double-annotated

GOV_LEGAL_SOURCES = frozenset({"govinfo_gao", "crs", "fed", "uscourts"})
TECHNICAL_SOURCES = frozenset({"rfc"})


@dataclass(frozen=True)
class Cell:
    name: str
    group: str
    quota: int


CELLS: tuple[Cell, ...] = (
    Cell("annual_report", "annual_report", 300),
    Cell("fund_document", "fund_document", 150),
    Cell("research_paper", "research_paper", 150),
    Cell("government_legal.generic", "government_legal", 100),
    Cell("government_legal.contract", "government_legal", 50),
    Cell("technical", "technical", 150),
)
_CELL_INDEX = {cell.name: k for k, cell in enumerate(CELLS)}


def element_family(record: ManifestRecord) -> str:
    """The element topic family of a document: its ``family_hint``, else its doc type's."""
    return record.family_hint or family_for_doc_type(record.doc_type_hint or "")


def cell_for(record: ManifestRecord) -> str | None:
    """The cell a document's elements are sampled in, or ``None`` if it belongs to none."""
    family = element_family(record)
    if family in ("annual_report", "fund_document", "research_paper"):
        return family
    if family == "contract":
        return "government_legal.contract"
    if family == "technical_document":
        return "technical"
    if family == "generic":
        if record.source in TECHNICAL_SOURCES:
            return "technical"
        if record.source in GOV_LEGAL_SOURCES:
            return "government_legal.generic"
    return None


def cell_of_row(row: dict[str, Any]) -> str:
    """The cell of a sample row (from its group and family)."""
    if row["group"] == "government_legal":
        return f"government_legal.{'contract' if row['family'] == 'contract' else 'generic'}"
    return row["group"]


@dataclass
class GoldSample:
    rows: list[dict[str, Any]]
    summary: dict[str, Any] = field(default_factory=dict)


def sample_gold(
    corpus: Corpus,
    *,
    seed: int = SEED,
    max_doc_share: float = MAX_DOC_SHARE,
    min_per_kind: int = MIN_PER_KIND,
    double_fraction: float = DOUBLE_FRACTION,
) -> GoldSample:
    """Sample the gold elements from ``corpus``'s gold-split documents (see the module doc)."""
    records = {r.id: r for r in corpus.manifest() if r.split == "gold"}
    cell_of_doc = {doc_id: cell_for(r) for doc_id, r in records.items()}
    pools: dict[str, list[tuple[int, CorpusElement]]] = defaultdict(list)
    seen_text: dict[str, set[str]] = defaultdict(set)
    position: Counter[str] = Counter()
    for element in corpus.iter_elements(split="gold"):
        index = position[element.doc_id]
        position[element.doc_id] += 1
        cell = cell_of_doc.get(element.doc_id)
        if cell is None or not element.text.strip() or element.text_sha1 in seen_text[cell]:
            continue
        seen_text[cell].add(element.text_sha1)
        pools[cell].append((index, element))

    rows: list[tuple[int, str, int, dict[str, Any]]] = []
    cells_summary: dict[str, Any] = {}
    for cell in CELLS:
        rng = random.Random(f"gold:{seed}:{cell.name}")
        cap = max(1, int(cell.quota * max_doc_share))
        pool = pools.get(cell.name, [])
        chosen = _pick(pool, cell.quota, cap, min_per_kind, rng)
        double = set(rng.sample(sorted(chosen), round(double_fraction * len(chosen))))
        docs = sorted(d for d, c in cell_of_doc.items() if c == cell.name)
        for index, element in pool:
            key = (element.doc_id, index)
            if key not in chosen:
                continue
            row = {
                "doc_id": element.doc_id,
                "element_id": element.element_id,
                "text_sha1": element.text_sha1,
                "kind": element.kind,
                "text": element.text,
                "heading_path": list(element.heading_path),
                "family": element_family(records[element.doc_id]),
                "group": cell.group,
                "double": key in double,
            }
            rows.append((_CELL_INDEX[cell.name], element.doc_id, index, row))
        per_doc = Counter(doc_id for doc_id, _ in chosen)
        cells_summary[cell.name] = {
            "group": cell.group,
            "quota": cell.quota,
            "gold_documents": len(docs),
            "documents_used": len(per_doc),
            "candidate_elements": len(pool),
            "per_document_cap": cap,
            "max_from_one_document": max(per_doc.values(), default=0),
            "selected": len(chosen),
            "shortfall": cell.quota - len(chosen),
            "double_annotated": len(double),
            "by_kind": dict(sorted(Counter(chosen.values()).items())),
        }

    rows.sort(key=lambda t: t[:3])
    out = [{"item": f"g{k:04d}", **row} for k, (_, _, _, row) in enumerate(rows)]
    unassigned = Counter(
        records[d].source or "(none)" for d, cell in cell_of_doc.items() if cell is None
    )
    groups: dict[str, dict[str, int]] = {}
    for cell in CELLS:
        entry = groups.setdefault(cell.group, {"quota": 0, "selected": 0, "shortfall": 0})
        for key in entry:
            entry[key] += cells_summary[cell.name][key]
    summary = {
        "seed": seed,
        "max_doc_share": max_doc_share,
        "min_per_kind": min_per_kind,
        "double_fraction": double_fraction,
        "quota": sum(cell.quota for cell in CELLS),
        "selected": len(out),
        "shortfall": sum(c["shortfall"] for c in cells_summary.values()),
        "double_annotated": sum(c["double_annotated"] for c in cells_summary.values()),
        "gold_documents": len(records),
        "unassigned_gold_documents_by_source": dict(sorted(unassigned.items())),
        "groups": groups,
        "cells": cells_summary,
    }
    return GoldSample(out, summary)


def _pick(
    pool: list[tuple[int, CorpusElement]],
    quota: int,
    cap: int,
    min_per_kind: int,
    rng: random.Random,
) -> dict[tuple[str, int], str]:
    """Choose up to ``quota`` elements of ``pool``: ``{(doc_id, index): kind}``.

    Kind targets come from :func:`_kind_targets`. Each pick takes, among the documents that
    still have an element of the wanted kind and are under ``cap``, the one with the fewest
    picks so far (ties by a seeded document order). Picks a kind cannot fill (its documents
    are at the cap) are then made from any kind, the same way.
    """
    queues: dict[str, dict[str, deque[tuple[int, str]]]] = defaultdict(lambda: defaultdict(deque))
    for index, element in sorted(pool, key=lambda t: (t[1].doc_id, t[0])):
        queues[element.kind][element.doc_id].append((index, element.kind))
    for by_doc in queues.values():
        for doc_id in sorted(by_doc):
            items = list(by_doc[doc_id])
            rng.shuffle(items)
            by_doc[doc_id] = deque(items)
    doc_ids = sorted({element.doc_id for _, element in pool})
    rng.shuffle(doc_ids)
    rank = {doc_id: k for k, doc_id in enumerate(doc_ids)}
    picks: Counter[str] = Counter()
    chosen: dict[tuple[str, int], str] = {}

    def take(kinds: Sequence[str], n: int) -> int:
        taken = 0
        while taken < n:
            open_docs = [
                (picks[doc_id], rank[doc_id], kind, doc_id)
                for kind in kinds
                for doc_id, items in queues[kind].items()
                if items and picks[doc_id] < cap
            ]
            if not open_docs:
                break
            _, _, kind, doc_id = min(open_docs)
            index, _ = queues[kind][doc_id].popleft()
            chosen[(doc_id, index)] = kind
            picks[doc_id] += 1
            taken += 1
        return taken

    available = {kind: sum(len(q) for q in by_doc.values()) for kind, by_doc in queues.items()}
    targets = _kind_targets(available, quota, min_per_kind)
    for kind in sorted(targets):
        take([kind], targets[kind])
    take(sorted(queues), quota - len(chosen))
    return chosen


def _kind_targets(available: dict[str, int], total: int, floor: int) -> dict[str, int]:
    """Split ``min(total, available)`` picks over kinds: each kind gets ``min(floor, its
    count)``, and the rest is shared in proportion to the counts left (largest remainder,
    ties by kind name). If the floors alone exceed the total, it is shared in proportion to
    the counts instead."""
    n = min(total, sum(available.values()))
    base = {kind: min(floor, count) for kind, count in available.items()}
    if sum(base.values()) > n:
        base = dict.fromkeys(available, 0)
    spare = {kind: available[kind] - base[kind] for kind in available}
    remaining = n - sum(base.values())
    total_spare = sum(spare.values())
    if remaining <= 0 or total_spare == 0:
        return base
    shares = {kind: remaining * spare[kind] / total_spare for kind in available}
    targets = {kind: base[kind] + math.floor(shares[kind]) for kind in available}
    left = n - sum(targets.values())
    by_remainder = sorted(available, key=lambda k: (-(shares[k] - math.floor(shares[k])), k))
    for kind in by_remainder[:left]:
        targets[kind] += 1
    return targets


# --- files -----------------------------------------------------------------------------------


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a temporary file and ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def jsonl(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", type=Path, required=True, help="the corpus store directory")
    parser.add_argument("--out", type=Path, default=SAMPLE_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-doc-share", type=float, default=MAX_DOC_SHARE)
    args = parser.parse_args(argv)
    if not (0 < args.max_doc_share <= 1):
        parser.error("--max-doc-share must be in (0, 1]")
    if not (args.corpus / "manifest.jsonl").is_file():
        parser.error(f"no corpus store at {args.corpus}")
    sample = sample_gold(Corpus(args.corpus), seed=args.seed, max_doc_share=args.max_doc_share)
    write_atomic(args.out, jsonl(sample.rows))
    write_atomic(args.summary, json.dumps(sample.summary, indent=2) + "\n")
    s = sample.summary
    print(f"{s['selected']} of {s['quota']} elements ({s['double_annotated']} double)")
    for name, cell in s["cells"].items():
        note = f", shortfall {cell['shortfall']}" if cell["shortfall"] else ""
        print(
            f"  {name}: {cell['selected']}/{cell['quota']} from "
            f"{cell['documents_used']}/{cell['gold_documents']} gold documents{note}"
        )
    if s["unassigned_gold_documents_by_source"]:
        print(f"  unassigned gold documents: {s['unassigned_gold_documents_by_source']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
