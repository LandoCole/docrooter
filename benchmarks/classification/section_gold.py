"""Section gold annotation support (spec G5, task V5): annotator batches, label validation
against the section catalogue, inter-annotator agreement and the adjudication merge.

    PYTHONPATH=src python -m benchmarks.classification.section_gold batches --out-dir D
    PYTHONPATH=src python -m benchmarks.classification.section_gold validate FILE...
    PYTHONPATH=src python -m benchmarks.classification.section_gold adjudication-batch \
        --first FILE... --second FILE... --out PATH
    PYTHONPATH=src python -m benchmarks.classification.section_gold merge \
        --first FILE... --second FILE... --adjudication FILE...

Each command takes ``--sample S``, the sampler's ``section_sample.jsonl`` (default: in
``benchmarks/.cache.nosync/classification/``). Batches and adjudication batches hold section
text too: write them under ``benchmarks/.cache.nosync/classification/annotation/`` (the
default of ``--out-dir``), never into a tracked directory; the commands refuse one.

An annotation file is JSON lines, one row per item:
``{"item": "s0042", "label": "<catalogue leaf id>", "annotator": "<name>"}``. The label is a
leaf of ``builtin_taxonomy("section_catalogue")``; a group id is not a label. The first
annotation covers every item; the second covers exactly the ``double`` items, each by a
different annotator than the first; the adjudication covers at least the items on which the two
disagree, by an annotator who is neither of them. ``section_guide.md`` is what the annotators
are given first.

``merge`` writes ``section_gold.jsonl`` (``doc_id``, ``section_id``, ``title_sha1``, ``level``,
``label``, ``annotator``, ``adjudicated``: ids and labels, no text) and
``section_gold_agreement.json`` (Cohen's kappa at leaf and at group on the double-annotated
items, and the reading the sample was made with: profile ``fast``). A section's gold label is
the first annotator's when it was annotated once, the shared label when both agree
(``annotator`` is ``"<first>+<second>"``), and the adjudicator's when they disagree
(``adjudicated`` is true).
"""

from __future__ import annotations

import argparse
import functools
import json
import random
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from benchmarks.classification.section_sample import (
    CACHE_DIR,
    HERE,
    SAMPLE_PATH,
    SUMMARY_PATH,
    require_untracked,
)
from benchmarks.training_corpus.gold import AnnotationError, _pair_stats, read_jsonl
from benchmarks.training_corpus.gold_sample import jsonl, write_atomic
from docrooter.taxonomies import builtin_taxonomy

GOLD_PATH = HERE / "section_gold.jsonl"
AGREEMENT_PATH = HERE / "section_gold_agreement.json"
GUIDE_PATH = HERE / "section_guide.md"
ANNOTATION_DIR = CACHE_DIR / "annotation"
DISCLOSURE = "Claude-annotated gold, evaluation only"
BATCH_SIZE = 50

ANNOTATOR_FIELDS = ("item", "level", "title", "heading_path", "text")
GOLD_FIELDS = (
    "doc_id",
    "section_id",
    "title_sha1",
    "level",
    "label",
    "annotator",
    "adjudicated",
)
_SAMPLE_FIELDS = ("item", "doc_id", "section_id", "title_sha1", "level") + (
    "title",
    "heading_path",
    "text",
    "double",
)


class Label(NamedTuple):
    """One annotator's label for one item."""

    label: str
    annotator: str


# --- catalogue -------------------------------------------------------------------------------


@functools.cache
def catalogue_leaves() -> dict[str, str]:
    """``{leaf id: group id}`` of the section catalogue, in catalogue order. A top-level leaf
    (``none``) is its own group."""
    taxonomy = builtin_taxonomy("section_catalogue")
    return {
        leaf.id: group.id
        for group in taxonomy.children()
        for leaf in (taxonomy.children(group.id) or [group])
    }


def group_of(label: str) -> str:
    """The group of a catalogue leaf."""
    return catalogue_leaves()[label]


def catalogue_listing() -> list[dict[str, Any]]:
    """The catalogue as the annotators read it: each group with its leaves, descriptions and
    examples (no document text)."""
    taxonomy = builtin_taxonomy("section_catalogue")
    return [
        {
            "group": group.id,
            "label": group.label,
            "description": group.description,
            "leaves": [
                {
                    "id": leaf.id,
                    "label": leaf.label,
                    "description": leaf.description,
                    "examples": list(leaf.examples),
                }
                for leaf in (taxonomy.children(group.id) or [group])
            ],
        }
        for group in taxonomy.children()
    ]


def label_errors(label: Any) -> list[str]:
    """What is wrong with a label (empty if it is valid): it must be a catalogue leaf id."""
    if not isinstance(label, str):
        return [f"label must be a leaf id string, got {label!r}"]
    if label in catalogue_leaves():
        return []
    groups = set(catalogue_leaves().values())
    if label in groups:
        return [f"{label!r} is a group; label with one of its leaves"]
    return [f"unknown catalogue leaf {label!r}"]


# --- files -----------------------------------------------------------------------------------


def read_sample(path: Path = SAMPLE_PATH) -> list[dict[str, Any]]:
    """The sampler's rows, checked for their fields and unique items."""
    rows = read_jsonl(path)
    errors = [
        f"{path} row {k}: missing {sorted(set(_SAMPLE_FIELDS) - row.keys())}"
        for k, row in enumerate(rows, 1)
        if not set(_SAMPLE_FIELDS) <= row.keys()
    ]
    repeated = sorted(i for i, n in Counter(r.get("item") for r in rows).items() if n > 1)
    if repeated:
        errors.append(f"{path}: items given twice: {repeated[:10]}")
    if errors:
        raise AnnotationError(errors)
    return rows


def annotator_view(row: dict[str, Any]) -> dict[str, Any]:
    """What an annotator sees of a sample row: its item id, level, title, heading path and
    opening text. Nothing else (no document id, no document type, no model output)."""
    return {name: row[name] for name in ANNOTATOR_FIELDS}


# --- annotations -----------------------------------------------------------------------------


def read_annotations(
    rows: Iterable[dict[str, Any]], sample: Sequence[dict[str, Any]]
) -> dict[str, Label]:
    """Validate annotation rows against the sample and the catalogue: ``{item: Label}``.
    Raises :class:`AnnotationError` listing every bad row."""
    items = {row["item"] for row in sample}
    labels: dict[str, Label] = {}
    errors: list[str] = []
    for k, row in enumerate(rows, 1):
        item = row.get("item")
        where = f"row {k} (item {item!r})"
        if not isinstance(item, str) or item not in items:
            errors.append(f"{where}: not an item of the sample")
            continue
        if item in labels:
            errors.append(f"{where}: item annotated twice")
            continue
        annotator = row.get("annotator")
        problems = label_errors(row.get("label"))
        if not isinstance(annotator, str) or not annotator.strip():
            problems.append("needs a non-empty 'annotator'")
        if problems:
            errors.append(f"{where}: {'; '.join(problems)}")
            continue
        labels[item] = Label(row["label"], annotator)
    if errors:
        raise AnnotationError(errors)
    return labels


def read_annotation_files(
    paths: Iterable[Path], sample: Sequence[dict[str, Any]]
) -> dict[str, Label]:
    rows = [row for path in paths for row in read_jsonl(path)]
    return read_annotations(rows, sample)


def _check_roles(
    sample: Sequence[dict[str, Any]], first: dict[str, Label], second: dict[str, Label]
) -> list[str]:
    """The double-annotated items, after checking coverage and independence: the first
    annotation covers every item, the second exactly the double ones, by another annotator."""
    items = [row["item"] for row in sample]
    double = [row["item"] for row in sample if row["double"]]
    errors: list[str] = []
    missing = [i for i in items if i not in first]
    if missing:
        errors.append(f"first annotation is missing {len(missing)} item(s): {missing[:10]}")
    missing = [i for i in double if i not in second]
    if missing:
        errors.append(f"second annotation is missing {len(missing)} item(s): {missing[:10]}")
    extra = sorted(set(second) - set(double))
    if extra:
        errors.append(f"second annotation of item(s) not in the double subset: {extra[:10]}")
    same = [i for i in double if i in first and i in second]
    same = [i for i in same if first[i].annotator == second[i].annotator]
    if same:
        errors.append(f"not independent: same annotator twice on item(s) {same[:10]}")
    if errors:
        raise AnnotationError(errors)
    return double


def disagreements(
    sample: Sequence[dict[str, Any]], first: dict[str, Label], second: dict[str, Label]
) -> list[str]:
    """The double items on which the two annotators differ, in sample order: the
    adjudicator's work."""
    return [i for i in _check_roles(sample, first, second) if first[i].label != second[i].label]


# --- agreement -------------------------------------------------------------------------------


def agreement(
    sample: Sequence[dict[str, Any]], first: dict[str, Label], second: dict[str, Label]
) -> dict[str, Any]:
    """Agreement of the two annotators on the double items: Cohen's kappa of the leaf labels
    and of their groups."""
    double = _check_roles(sample, first, second)
    leaf = _pair_stats([first[i].label for i in double], [second[i].label for i in double])
    group = _pair_stats(
        [group_of(first[i].label) for i in double], [group_of(second[i].label) for i in double]
    )
    return {
        "double_annotated": len(double),
        "disagreements": sum(first[i].label != second[i].label for i in double),
        "leaf": leaf,
        "group": group,
    }


# --- merge -----------------------------------------------------------------------------------


def merge_gold(
    sample: Sequence[dict[str, Any]],
    first: dict[str, Label],
    second: dict[str, Label],
    adjudication: dict[str, Label],
) -> list[dict[str, Any]]:
    """The final gold rows, in sample order (see the module doc for which label wins)."""
    disputed = set(disagreements(sample, first, second))
    missing = [row["item"] for row in sample if row["item"] in disputed]
    missing = [i for i in missing if i not in adjudication]
    if missing:
        raise AnnotationError([f"no adjudication for disputed item(s) {missing[:10]}"])
    not_third = [
        i
        for i in sorted(disputed)
        if adjudication[i].annotator in (first[i].annotator, second[i].annotator)
    ]
    if not_third:
        raise AnnotationError([f"adjudicator is not a third annotator on item(s) {not_third[:10]}"])
    rows: list[dict[str, Any]] = []
    for row in sample:
        item = row["item"]
        if item in disputed:
            label, annotator, adjudicated = adjudication[item], adjudication[item].annotator, True
        elif item in second:
            label, adjudicated = first[item], False
            annotator = f"{first[item].annotator}+{second[item].annotator}"
        else:
            label, annotator, adjudicated = first[item], first[item].annotator, False
        rows.append(
            {
                "doc_id": row["doc_id"],
                "section_id": row["section_id"],
                "title_sha1": row["title_sha1"],
                "level": row["level"],
                "label": label.label,
                "annotator": annotator,
                "adjudicated": adjudicated,
            }
        )
    errors = gold_row_errors(rows, sample)
    if errors:
        raise AnnotationError(errors)
    return rows


def gold_row_errors(rows: Sequence[dict[str, Any]], sample: Sequence[dict[str, Any]]) -> list[str]:
    """What is wrong with gold rows (empty if nothing): each must have exactly the gold fields
    (so no text), name a sample section by its key with the sample's level, carry a catalogue
    leaf, an annotator and a boolean ``adjudicated``; every sampled section appears once."""
    keys = {(r["doc_id"], r["section_id"], r["title_sha1"]): r for r in sample}
    seen: set[tuple[Any, ...]] = set()
    errors: list[str] = []
    for k, row in enumerate(rows, 1):
        key = tuple(row.get(name) for name in GOLD_FIELDS[:3])
        if key not in keys:
            errors.append(f"gold row {k}: {key} is not a sampled section")
            continue
        if key in seen:
            errors.append(f"gold row {k}: {key} appears twice")
            continue
        seen.add(key)
        absent = [name for name in GOLD_FIELDS if name not in row]
        extra = sorted(set(row) - set(GOLD_FIELDS))
        if absent or extra:
            errors.append(f"gold row {k} {key}: missing {absent}, unexpected {extra}")
            continue
        problems = label_errors(row["label"])
        if row["level"] != keys[key]["level"]:
            problems.append(f"level {row['level']!r} is not the section's {keys[key]['level']!r}")
        if not isinstance(row["annotator"], str) or not row["annotator"].strip():
            problems.append("annotator must be a non-empty string")
        if not isinstance(row["adjudicated"], bool):
            problems.append("adjudicated must be true or false")
        if problems:
            errors.append(f"gold row {k} {key}: {'; '.join(problems)}")
    left_out = [row["item"] for key, row in keys.items() if key not in seen]
    if left_out:
        errors.append(f"{len(left_out)} sampled section(s) have no gold row: {left_out[:10]}")
    return errors


# --- CLI -------------------------------------------------------------------------------------


def _batches(sample: list[dict[str, Any]], out_dir: Path, size: int) -> dict[str, int]:
    """Write the annotator views in batches of ``size`` (``first/`` holds every item,
    ``second/`` the double ones) and the catalogue listing. Returns the batch counts."""
    counts: dict[str, int] = {}
    for role, rows in (("first", sample), ("second", [r for r in sample if r["double"]])):
        views = [annotator_view(r) for r in rows]
        chunks = [views[k : k + size] for k in range(0, len(views), size)]
        for n, chunk in enumerate(chunks, 1):
            write_atomic(out_dir / role / f"batch_{n:03d}.jsonl", jsonl(chunk))
        counts[role] = len(chunks)
    write_atomic(out_dir / "catalogue.json", json.dumps(catalogue_listing(), indent=2) + "\n")
    return counts


def _adjudication_rows(
    sample: list[dict[str, Any]], first: dict[str, Label], second: dict[str, Label]
) -> list[dict[str, Any]]:
    """The disputed items as the adjudicator sees them: the annotator view plus the two
    labels, without the annotators' names."""
    by_item = {row["item"]: row for row in sample}
    rows = []
    for item in disagreements(sample, first, second):
        labels = [first[item].label, second[item].label]
        if random.Random(f"adjudication:{item}").random() < 0.5:  # not first-then-second always
            labels.reverse()
        rows.append({**annotator_view(by_item[item]), "labels": labels})
    return rows


def _names(labels: dict[str, Label]) -> list[str]:
    return sorted({label.annotator for label in labels.values()})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("batches", help="write annotator batches (first and second)")
    p.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    p.add_argument("--out-dir", type=Path, default=ANNOTATION_DIR)
    p.add_argument("--size", type=int, default=BATCH_SIZE)
    p = sub.add_parser("validate", help="check annotation files against sample and catalogue")
    p.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    p.add_argument("files", type=Path, nargs="+")
    p = sub.add_parser("adjudication-batch", help="write the items the annotators disagree on")
    p.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    p.add_argument("--first", type=Path, nargs="+", required=True)
    p.add_argument("--second", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, default=ANNOTATION_DIR / "adjudication.jsonl")
    p = sub.add_parser("merge", help="write section_gold.jsonl and section_gold_agreement.json")
    p.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    p.add_argument("--first", type=Path, nargs="+", required=True)
    p.add_argument("--second", type=Path, nargs="+", required=True)
    p.add_argument("--adjudication", type=Path, nargs="+", required=True)
    p.add_argument("--sample-summary", type=Path, default=SUMMARY_PATH)
    p.add_argument("--gold", type=Path, default=GOLD_PATH)
    p.add_argument("--agreement", type=Path, default=AGREEMENT_PATH)
    args = parser.parse_args(argv)

    try:
        for text_path in (getattr(args, "out_dir", None), getattr(args, "out", None)):
            if text_path is not None:
                require_untracked(text_path)
        sample = read_sample(args.sample)
        if args.command == "batches":
            if args.size < 1:
                parser.error("--size must be positive")
            counts = _batches(sample, args.out_dir, args.size)
            print(f"batches: {counts['first']} first, {counts['second']} second")
        elif args.command == "validate":
            labels = read_annotation_files(args.files, sample)
            print(f"{len(labels)} valid annotation(s)")
        elif args.command == "adjudication-batch":
            first = read_annotation_files(args.first, sample)
            second = read_annotation_files(args.second, sample)
            rows = _adjudication_rows(sample, first, second)
            write_atomic(args.out, jsonl(rows))
            print(f"{len(rows)} item(s) to adjudicate")
        else:
            first = read_annotation_files(args.first, sample)
            second = read_annotation_files(args.second, sample)
            adjudication = read_annotation_files(args.adjudication, sample)
            rows = merge_gold(sample, first, second, adjudication)
            report = agreement(sample, first, second)
            disputed = set(disagreements(sample, first, second))
            sample_summary = json.loads(args.sample_summary.read_text(encoding="utf-8"))
            if not isinstance(sample_summary, dict) or "reading" not in sample_summary:
                raise AnnotationError(
                    [f"{args.sample_summary} has no 'reading' (not a sample summary)"]
                )
            summary = {
                "disclosure": DISCLOSURE,
                "reading": sample_summary["reading"],
                "sections": len(rows),
                "adjudicated": sum(r["adjudicated"] for r in rows),
                "adjudications_unused": len(set(adjudication) - disputed),
                **report,
                "annotators": {
                    "first": _names(first),
                    "second": _names(second),
                    "adjudicator": _names({i: adjudication[i] for i in disputed}),
                },
                "sample": sample_summary,
            }
            write_atomic(args.gold, jsonl(rows))
            write_atomic(args.agreement, json.dumps(summary, indent=2) + "\n")
            kappa = report["leaf"]["kappa"]
            shown = "undefined" if kappa is None else f"{kappa:.3f}"
            print(f"{len(rows)} gold rows; leaf kappa {shown} on {report['double_annotated']}")
    except (OSError, ValueError) as exc:  # AnnotationError is a ValueError
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
