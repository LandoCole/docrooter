"""Gold-set annotation support (spec §11, task L5): annotator batches, label validation
against the element taxonomies, inter-annotator agreement and the adjudication merge.

    PYTHONPATH=src python -m benchmarks.training_corpus.gold batches --sample S --out-dir D
    PYTHONPATH=src python -m benchmarks.training_corpus.gold validate --sample S FILE...
    PYTHONPATH=src python -m benchmarks.training_corpus.gold adjudication-batch --sample S \
        --first FILE... --second FILE... --out PATH
    PYTHONPATH=src python -m benchmarks.training_corpus.gold merge --sample S \
        --first FILE... --second FILE... --adjudication FILE... \
        [--sample-summary benchmarks/training_corpus/gold_sample_summary.json]

``S`` is the sampler's ``gold_sample.jsonl``. An annotation file is JSON lines, one row per
item: ``{"item": "g0042", "topic": "<leaf id>", "functions": [...], "annotator": "<name>"}``.
The first annotation covers every item; the second covers exactly the ``double`` items, each by
a different annotator than the first; the adjudication covers at least the items on which the
two disagree (topic, or the set of functions).

``merge`` writes ``gold.jsonl`` (``doc_id``, ``element_id``, ``text_sha1``, ``family``,
``topic``, ``functions``, ``annotator``, ``adjudicated``) and ``gold_agreement.json`` (Cohen's
kappa for the topic and per function tag on the double-annotated items). An element's gold
label is the first annotator's when it was annotated once, the shared label when both
annotators agree (``annotator`` is ``"<first>+<second>"``), and the adjudicator's when they
disagree (``adjudicated`` is true).
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from benchmarks.training_corpus.gold_sample import (
    HERE,
    SAMPLE_PATH,
    jsonl,
    write_atomic,
)
from docrooter.taxonomies import builtin_taxonomy

GOLD_PATH = HERE / "gold.jsonl"
AGREEMENT_PATH = HERE / "gold_agreement.json"
DISCLOSURE = "Claude-annotated gold"
BATCH_SIZE = 50

ANNOTATOR_FIELDS = ("item", "kind", "family", "heading_path", "text")
GOLD_FIELDS = (
    "doc_id",
    "element_id",
    "text_sha1",
    "family",
    "topic",
    "functions",
    "annotator",
    "adjudicated",
)
_SAMPLE_FIELDS = ("item", "doc_id", "element_id", "text_sha1", "kind", "text", "heading_path")
_SAMPLE_FIELDS += ("family", "group", "double")


class AnnotationError(ValueError):
    """Annotations or gold rows that cannot be used; the message lists every problem found."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        shown = "\n  ".join(self.errors[:20])
        more = f"\n  (+{len(self.errors) - 20} more)" if len(self.errors) > 20 else ""
        super().__init__(f"{len(self.errors)} problem(s):\n  {shown}{more}")


class Label(NamedTuple):
    """One annotator's label for one item; ``functions`` in taxonomy order."""

    topic: str
    functions: tuple[str, ...]
    annotator: str


# --- taxonomies ------------------------------------------------------------------------------


@functools.cache
def topic_leaves() -> dict[str, frozenset[str]]:
    """The topic leaf ids of each element family in ``element_topics``."""
    taxonomy = builtin_taxonomy("element_topics")
    return {
        family.id: frozenset(n.id for n in taxonomy.children(family.id) if not n.children)
        for family in taxonomy.children()
    }


@functools.cache
def function_tags() -> tuple[str, ...]:
    """The function tag ids of ``element_functions``, in taxonomy order."""
    return tuple(node.id for node in builtin_taxonomy("element_functions").leaves())


def label_errors(family: str, topic: Any, functions: Any) -> list[str]:
    """What is wrong with a label for an element of ``family`` (empty if it is valid): the
    topic must be a leaf of that family; the functions a list of distinct known tags."""
    leaves = topic_leaves()
    errors: list[str] = []
    if family not in leaves:
        return [f"unknown element family {family!r}"]
    if not isinstance(topic, str):
        errors.append(f"topic must be a leaf id string, got {topic!r}")
    elif topic not in leaves[family]:
        if any(topic in ids for ids in leaves.values()):
            errors.append(f"topic {topic!r} is not a leaf of family {family!r}")
        else:
            errors.append(f"unknown topic leaf {topic!r} (family {family!r})")
    if not isinstance(functions, list | tuple) or not all(isinstance(f, str) for f in functions):
        errors.append(f"functions must be a list of tags, got {functions!r}")
        return errors
    tags = set(function_tags())
    unknown = [f for f in functions if f not in tags]
    if unknown:
        errors.append(f"unknown function tag(s) {unknown!r}")
    repeated = sorted(f for f, n in Counter(functions).items() if n > 1)
    if repeated:
        errors.append(f"function tag(s) given twice: {repeated!r}")
    return errors


def _ordered(functions: Iterable[str]) -> tuple[str, ...]:
    wanted = set(functions)
    return tuple(tag for tag in function_tags() if tag in wanted)


# --- files -----------------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise AnnotationError([f"{path} line {number}: not JSON: {exc}"]) from exc
        if not isinstance(row, dict):
            raise AnnotationError([f"{path} line {number}: not a JSON object"])
        rows.append(row)
    return rows


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
    """What an annotator sees of a sample row: its item id, kind, family, heading path and
    text. Nothing else (no document id, no model output)."""
    return {name: row[name] for name in ANNOTATOR_FIELDS}


# --- annotations -----------------------------------------------------------------------------


def read_annotations(
    rows: Iterable[dict[str, Any]], sample: Sequence[dict[str, Any]]
) -> dict[str, Label]:
    """Validate annotation rows against the sample and the taxonomies: ``{item: Label}``.
    Raises :class:`AnnotationError` listing every bad row."""
    by_item = {row["item"]: row for row in sample}
    labels: dict[str, Label] = {}
    errors: list[str] = []
    for k, row in enumerate(rows, 1):
        item = row.get("item")
        where = f"row {k} (item {item!r})"
        if not isinstance(item, str) or item not in by_item:
            errors.append(f"{where}: not an item of the sample")
            continue
        if item in labels:
            errors.append(f"{where}: item annotated twice")
            continue
        annotator = row.get("annotator")
        problems = label_errors(by_item[item]["family"], row.get("topic"), row.get("functions"))
        if not isinstance(annotator, str) or not annotator.strip():
            problems.append("needs a non-empty 'annotator'")
        if problems:
            errors.append(f"{where}: {'; '.join(problems)}")
            continue
        labels[item] = Label(row["topic"], _ordered(row["functions"]), annotator)
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


def _agree(a: Label, b: Label) -> bool:
    return a.topic == b.topic and a.functions == b.functions


def disagreements(
    sample: Sequence[dict[str, Any]], first: dict[str, Label], second: dict[str, Label]
) -> list[str]:
    """The double items on which the two annotators differ (topic or functions), in sample
    order: the adjudicator's work."""
    return [i for i in _check_roles(sample, first, second) if not _agree(first[i], second[i])]


# --- agreement -------------------------------------------------------------------------------


def cohen_kappa(a: Sequence[Any], b: Sequence[Any]) -> float | None:
    """Cohen's kappa of two raters' nominal labels for the same items: (po - pe) / (1 - pe).
    ``None`` when undefined: no items, or chance agreement 1 (both used one same label)."""
    if len(a) != len(b):
        raise ValueError(f"raters labelled {len(a)} and {len(b)} items")
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    count_a, count_b = Counter(a), Counter(b)
    expected = sum(count_a[k] * count_b[k] for k in count_a) / (n * n)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def _pair_stats(a: Sequence[Any], b: Sequence[Any]) -> dict[str, Any]:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n if n else None
    return {"n": n, "kappa": cohen_kappa(a, b), "observed_agreement": observed}


def agreement(
    sample: Sequence[dict[str, Any]], first: dict[str, Label], second: dict[str, Label]
) -> dict[str, Any]:
    """Agreement of the two annotators on the double items: Cohen's kappa for the topic (in
    all and per family) and, per function tag, kappa on its presence."""
    double = _check_roles(sample, first, second)
    family = {row["item"]: row["family"] for row in sample}
    topic = _pair_stats([first[i].topic for i in double], [second[i].topic for i in double])
    by_family: dict[str, Any] = {}
    for name in sorted({family[i] for i in double}):
        items = [i for i in double if family[i] == name]
        by_family[name] = _pair_stats(
            [first[i].topic for i in items], [second[i].topic for i in items]
        )
    per_tag: dict[str, Any] = {}
    for tag in function_tags():
        a = [tag in first[i].functions for i in double]
        b = [tag in second[i].functions for i in double]
        per_tag[tag] = {**_pair_stats(a, b), "positives_first": sum(a), "positives_second": sum(b)}
    defined = [s["kappa"] for s in per_tag.values() if s["kappa"] is not None]
    exact = [first[i].functions == second[i].functions for i in double]
    return {
        "double_annotated": len(double),
        "disagreements": sum(not _agree(first[i], second[i]) for i in double),
        "topic": {**topic, "by_family": by_family},
        "functions": {
            "per_tag": per_tag,
            "mean_kappa_defined_tags": sum(defined) / len(defined) if defined else None,
            "exact_set_agreement": sum(exact) / len(exact) if exact else None,
        },
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
                "element_id": row["element_id"],
                "text_sha1": row["text_sha1"],
                "family": row["family"],
                "topic": label.topic,
                "functions": list(label.functions),
                "annotator": annotator,
                "adjudicated": adjudicated,
            }
        )
    errors = gold_row_errors(rows, sample)
    if errors:
        raise AnnotationError(errors)
    return rows


def gold_row_errors(rows: Sequence[dict[str, Any]], sample: Sequence[dict[str, Any]]) -> list[str]:
    """What is wrong with gold rows (empty if nothing): each must have the gold fields, name a
    sample element by its key with the sample's family, carry a valid label for that family,
    an annotator and a boolean ``adjudicated``; every sample element appears exactly once."""
    keys = {(r["doc_id"], r["element_id"], r["text_sha1"]): r for r in sample}
    seen: set[tuple[Any, ...]] = set()
    errors: list[str] = []
    for k, row in enumerate(rows, 1):
        key = tuple(row.get(name) for name in GOLD_FIELDS[:3])
        if key not in keys:
            errors.append(f"gold row {k}: {key} is not a sampled element")
            continue
        if key in seen:
            errors.append(f"gold row {k}: {key} appears twice")
            continue
        seen.add(key)
        absent = [name for name in GOLD_FIELDS if name not in row]
        if absent:
            errors.append(f"gold row {k} {key}: missing {absent}")
            continue
        family = keys[key]["family"]
        problems = label_errors(family, row["topic"], row["functions"])
        if row["family"] != family:
            problems.insert(0, f"family {row['family']!r} is not the element's {family!r}")
        if not isinstance(row["annotator"], str) or not row["annotator"].strip():
            problems.append("annotator must be a non-empty string")
        if not isinstance(row["adjudicated"], bool):
            problems.append("adjudicated must be true or false")
        if problems:
            errors.append(f"gold row {k} {key}: {'; '.join(problems)}")
    left_out = [row["item"] for key, row in keys.items() if key not in seen]
    if left_out:
        errors.append(f"{len(left_out)} sampled element(s) have no gold row: {left_out[:10]}")
    return errors


# --- CLI -------------------------------------------------------------------------------------


def _batches(sample: list[dict[str, Any]], out_dir: Path, size: int) -> dict[str, int]:
    """Write the annotator views in batches of ``size``: ``first/`` holds every item,
    ``second/`` the double ones. Returns the batch counts."""
    counts: dict[str, int] = {}
    for role, rows in (("first", sample), ("second", [r for r in sample if r["double"]])):
        views = [annotator_view(r) for r in rows]
        chunks = [views[k : k + size] for k in range(0, len(views), size)]
        for n, chunk in enumerate(chunks, 1):
            write_atomic(out_dir / role / f"batch_{n:03d}.jsonl", jsonl(chunk))
        counts[role] = len(chunks)
    return counts


def _adjudication_rows(
    sample: list[dict[str, Any]], first: dict[str, Label], second: dict[str, Label]
) -> list[dict[str, Any]]:
    """The disputed items as the adjudicator sees them: the annotator view plus the two
    labels, without the annotators' names."""
    by_item = {row["item"]: row for row in sample}
    return [
        {
            **annotator_view(by_item[item]),
            "labels": [
                {"topic": label.topic, "functions": list(label.functions)}
                for label in (first[item], second[item])
            ],
        }
        for item in disagreements(sample, first, second)
    ]


def _names(labels: dict[str, Label]) -> list[str]:
    return sorted({label.annotator for label in labels.values()})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("batches", help="write annotator batches (first and second)")
    p.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--size", type=int, default=BATCH_SIZE)
    p = sub.add_parser("validate", help="check annotation files against sample and taxonomies")
    p.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    p.add_argument("files", type=Path, nargs="+")
    p = sub.add_parser("adjudication-batch", help="write the items the annotators disagree on")
    p.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    p.add_argument("--first", type=Path, nargs="+", required=True)
    p.add_argument("--second", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("merge", help="write gold.jsonl and gold_agreement.json")
    p.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    p.add_argument("--first", type=Path, nargs="+", required=True)
    p.add_argument("--second", type=Path, nargs="+", required=True)
    p.add_argument("--adjudication", type=Path, nargs="+", required=True)
    p.add_argument("--sample-summary", type=Path, help="the sampler's summary, to include")
    p.add_argument("--gold", type=Path, default=GOLD_PATH)
    p.add_argument("--agreement", type=Path, default=AGREEMENT_PATH)
    args = parser.parse_args(argv)

    try:
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
            summary = {
                "disclosure": DISCLOSURE,
                "elements": len(rows),
                "adjudicated": sum(r["adjudicated"] for r in rows),
                "adjudications_unused": len(set(adjudication) - disputed),
                **report,
                "annotators": {
                    "first": _names(first),
                    "second": _names(second),
                    "adjudicator": _names({i: adjudication[i] for i in disputed}),
                },
            }
            if args.sample_summary is not None:
                summary["sample"] = json.loads(args.sample_summary.read_text(encoding="utf-8"))
            write_atomic(args.gold, jsonl(rows))
            write_atomic(args.agreement, json.dumps(summary, indent=2) + "\n")
            kappa = report["topic"]["kappa"]
            shown = "undefined" if kappa is None else f"{kappa:.3f}"
            print(f"{len(rows)} gold rows; topic kappa {shown} on {report['double_annotated']}")
    except AnnotationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
