"""Element kinds against DocLayNet's human layout labels (v0.4 task E11a; spec §10.1).

``PYTHONPATH=src .venv/bin/python -m benchmarks.elements_bench --split train``

Pages come from ``benchmarks/doclaynet.py prepare`` (a stratified sample of single-page
born-digital PDFs with their gold boxes). Each page is loaded with
``load_pdf(elements=True)`` per system:

- ``heuristic-pymupdf`` and ``heuristic-pdfplumber``: the element rules (spec §5.3);
- ``heron``: pymupdf with the heron layout model (spec §6), one session for the whole run,
  under the one-model-process guard.

Scoring (spec §10.1): element kinds map to DocLayNet's 11 classes (spec §4: ``page_number`` is
Page-footer, ``contents`` is Text). Predicted and gold boxes on the page are matched one to one,
greedy by IoU (>= 0.5), within a class; per class precision, recall and F1 pool the counts over
pages, and the macro F1 is the mean over the classes present (in gold or predictions). Per
category the same, over that category's pages. The IoU-free text coverage of a class is the
share of the characters of the PDF text cells inside its gold boxes that land in a predicted
element of that class: each cell belongs to the predicted element its box overlaps most.

Gold boxes are COCO ``[x, y, w, h]`` in the page image's pixels (1025 x 1025: DocLayNet
stretches each page to a square), so each axis is scaled by its own page side, taken from the
PDF's displayed page (``pymupdf``), the same space as element boxes.

``--set NAME=VALUE`` overrides an element-rule threshold of :mod:`docrooter.pdf.elements` for
the run (tuning on train pages only; see ``docs/benchmarks-elements.md``). Results go to
``benchmarks/results/<date>/elements-doclaynet[-train].json``; ``--docs`` writes the tables into
``docs/benchmarks-elements.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks import _guard, doclaynet
from benchmarks.toc import RESULTS_ROOT, collect_env, git_revision
from docrooter.elements.models import Box

CLASSES = (
    "Caption", "Footnote", "Formula", "List-item", "Page-footer", "Page-header", "Picture",
    "Section-header", "Table", "Text", "Title",
)  # fmt: skip
# Spec §4's mapping table.
KIND_TO_CLASS: dict[str, str] = {
    "caption": "Caption",
    "footnote": "Footnote",
    "formula": "Formula",
    "list_item": "List-item",
    "page_footer": "Page-footer",
    "page_number": "Page-footer",
    "page_header": "Page-header",
    "figure": "Picture",
    "heading": "Section-header",
    "table": "Table",
    "paragraph": "Text",
    "contents": "Text",
    "title": "Title",
}
SYSTEMS: dict[str, tuple[str, str]] = {  # name -> (engine, layout)
    "heuristic-pymupdf": ("pymupdf", "heuristic"),
    "heuristic-pdfplumber": ("pdfplumber", "heuristic"),
    "heron": ("pymupdf", "heron"),
}
IOU = 0.5
# Spec §10.1 target: heuristic F1 >= 0.8 for these classes on these categories.
TARGET_F1 = 0.8
TARGET_CLASSES = ("Text", "Table", "Picture")
TARGET_CATEGORIES = ("financial_reports", "scientific_articles")
DOCS = Path(__file__).resolve().parent.parent / "docs" / "benchmarks-elements.md"
BEGIN, END = "<!-- elements-benchmark:begin -->", "<!-- elements-benchmark:end -->"

Pred = tuple[Box, str]  # (box, element kind)
Gold = tuple[Box, str]  # (box, DocLayNet class)
Cell = tuple[Box, int, int]  # (cell box, characters, index of its gold box)


# --- matching and scoring ------------------------------------------------------------------


def class_of(kind: str) -> str | None:
    """The DocLayNet class an element kind is scored as (spec §4), or ``None``."""
    return KIND_TO_CLASS.get(kind)


def _area(b: Box) -> float:
    return max(0.0, b.x1 - b.x0) * max(0.0, b.y1 - b.y0)


def _intersection(a: Box, b: Box) -> float:
    if a.page != b.page:
        return 0.0
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if w > 0 and h > 0 else 0.0


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes (0 on different pages or for zero area)."""
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if inter > 0 and union > 0 else 0.0


_iou = iou  # ``match_boxes`` names its threshold ``iou``


def match_boxes(
    pred: Sequence[Pred], gold: Sequence[Gold], iou: float = IOU, *, same_class: bool = True
) -> list[tuple[int, int]]:
    """One-to-one greedy matching of predicted to gold boxes, as ``(pred, gold)`` indexes.

    A pair is a candidate when its IoU is at least ``iou`` and (with ``same_class``) the
    prediction's kind maps to the gold class; a kind with no class never matches then.
    Candidates are taken highest IoU first (ties: lower indexes), each box used once.
    """
    threshold = iou
    candidates = []
    for p, (pbox, kind) in enumerate(pred):
        mapped = class_of(kind)
        if same_class and mapped is None:
            continue
        for g, (gbox, cls) in enumerate(gold):
            if same_class and mapped != cls:
                continue
            value = _iou(pbox, gbox)
            if value >= threshold and value > 0:
                candidates.append((-value, p, g))
    candidates.sort()
    used_p: set[int] = set()
    used_g: set[int] = set()
    pairs = []
    for _, p, g in candidates:
        if p not in used_p and g not in used_g:
            used_p.add(p)
            used_g.add(g)
            pairs.append((p, g))
    return pairs


def gold_to_points(
    bbox: Sequence[float], width: float, height: float, *, coco: Sequence[int] = (1025, 1025),
    page: int = 1,
) -> Box:  # fmt: skip
    """A DocLayNet COCO box (``[x, y, w, h]`` in image pixels) in PDF points."""
    sx, sy = width / coco[0], height / coco[1]
    x, y, w, h = bbox
    return Box(page=page, x0=x * sx, y0=y * sy, x1=(x + w) * sx, y1=(y + h) * sy)


@dataclass
class ScoredPage:
    """One page: its predictions (box, element kind), gold (box, class) and gold text cells."""

    page_id: str
    category: str
    pred: list[Pred]
    gold: list[Gold]
    cells: list[Cell] = field(default_factory=list)


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    present = tp + fp + fn > 0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": (tp / (tp + fp) if tp + fp else 0.0) if present else None,
        "recall": (tp / (tp + fn) if tp + fn else 0.0) if present else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if present else None,
    }  # fmt: skip


def _counts(pages: Sequence[ScoredPage]) -> dict[str, dict[str, Any]]:
    tp: Counter[str] = Counter()
    n_pred: Counter[str] = Counter()
    n_gold: Counter[str] = Counter()
    for page in pages:
        for _, g in match_boxes(page.pred, page.gold):
            tp[page.gold[g][1]] += 1
        for _, kind in page.pred:
            if (cls := class_of(kind)) is not None:
                n_pred[cls] += 1
        for _, cls in page.gold:
            n_gold[cls] += 1
    return {c: _prf(tp[c], n_pred[c] - tp[c], n_gold[c] - tp[c]) for c in CLASSES}


def _macro(per_class: Mapping[str, Mapping[str, Any]]) -> float | None:
    f1s = [v["f1"] for v in per_class.values() if v["f1"] is not None]
    return sum(f1s) / len(f1s) if f1s else None


def text_coverage(pages: Sequence[ScoredPage]) -> dict[str, dict[str, Any]]:
    """Per class: the characters of its gold boxes' text cells and how many land in a
    predicted element of that class (the element the cell's box overlaps most; ties go to
    the smaller element)."""
    chars: Counter[str] = Counter()
    covered: Counter[str] = Counter()
    for page in pages:
        for cell, n, g in page.cells:
            cls = page.gold[g][1]
            chars[cls] += n
            best: tuple[float, float, str] | None = None
            for pbox, kind in page.pred:
                inter = _intersection(cell, pbox)
                if inter > 0 and (best is None or (inter, -_area(pbox)) > best[:2]):
                    best = (inter, -_area(pbox), kind)
            if best is not None and class_of(best[2]) == cls:
                covered[cls] += n
    return {
        c: {"chars": chars[c], "covered": covered[c],
            "share": covered[c] / chars[c] if chars[c] else None}
        for c in CLASSES
    }  # fmt: skip


def confusion(pages: Sequence[ScoredPage]) -> dict[str, dict[str, int]]:
    """Gold class -> predicted class (``"-"`` for none) over class-agnostic IoU matches;
    unmatched predictions are counted under gold ``"-"``. A diagnostic for tuning."""
    table: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for page in pages:
        pairs = match_boxes(page.pred, page.gold, same_class=False)
        matched_p = {p for p, _ in pairs}
        matched_g = {g: p for p, g in pairs}
        for g, (_, cls) in enumerate(page.gold):
            p = matched_g.get(g)
            kind = page.pred[p][1] if p is not None else None
            table[cls][(class_of(kind) or kind) if kind is not None else "-"] += 1
        for p, (_, kind) in enumerate(page.pred):
            if p not in matched_p:
                table["-"][class_of(kind) or kind] += 1
    return {g: dict(sorted(row.items())) for g, row in sorted(table.items())}


def score(pages: Sequence[ScoredPage]) -> dict[str, Any]:
    """Per-class P/R/F1, macro F1, per-category scores and text coverage (spec §10.1)."""
    per_class = _counts(pages)
    by_category: defaultdict[str, list[ScoredPage]] = defaultdict(list)
    for page in pages:
        by_category[page.category].append(page)
    per_category = {}
    for category, group in sorted(by_category.items()):
        counts = _counts(group)
        per_category[category] = {
            "pages": len(group),
            "macro": _macro(counts),
            "f1": {c: v["f1"] for c, v in counts.items()},
        }
    return {
        "pages": len(pages),
        "per_class": per_class,
        "macro": _macro(per_class),
        "per_category": per_category,
        "text_coverage": text_coverage(pages),
        "confusion": confusion(pages),
    }


# --- running the systems -------------------------------------------------------------------


@contextlib.contextmanager
def overrides(values: Mapping[str, float]) -> Iterator[None]:
    """Set element-rule thresholds of :mod:`docrooter.pdf.elements` for the block."""
    from docrooter.pdf import elements as rules

    unknown = [
        name
        for name in values
        if name not in rules.__all__ or not isinstance(getattr(rules, name), int | float)
    ]
    if unknown:
        raise SystemExit(f"not an element threshold: {', '.join(unknown)}")
    saved = {name: getattr(rules, name) for name in values}
    try:
        for name, value in values.items():
            setattr(rules, name, type(saved[name])(value))
        yield
    finally:
        for name, value in saved.items():
            setattr(rules, name, value)


def page_size(path: Path) -> tuple[float, float]:
    """The displayed size of the PDF's first page, in points."""
    import pymupdf

    with pymupdf.open(path) as pdf:
        rect = pdf[0].rect
        return rect.width, rect.height


def _chars(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def gold_page(
    record: Mapping[str, Any], width: float, height: float
) -> tuple[list[Gold], list[Cell]]:
    """A manifest record's gold boxes and text cells, in points."""
    coco = record["coco"]
    gold: list[Gold] = []
    cells: list[Cell] = []
    for g, item in enumerate(record["gold"]):
        gold.append((gold_to_points(item["bbox"], width, height, coco=coco), item["cls"]))
        for cell in item["cells"]:
            n = _chars(cell["text"])
            if n:
                cells.append((gold_to_points(cell["bbox"], width, height, coco=coco), n, g))
    return gold, cells


def predict(path: Path, engine: str, layout: Any) -> tuple[list[Pred], list[str]]:
    """The element boxes of a single-page PDF in element mode, with their kinds, and the
    document's warnings (``no_text_layer`` for a scan: no OCR engine runs here)."""
    from docrooter.pdf.load import load_pdf

    doc = load_pdf(path, engine=engine, elements=True, layout=layout)
    pred: list[Pred] = [(b, e.kind) for e in doc.elements for b in e.boxes if b.page == 1]
    return pred, list(doc.meta.get("warnings", []))


def run_system(
    name: str, split: str, records: Sequence[Mapping[str, Any]], *, verbose: bool = False
) -> tuple[list[ScoredPage], dict[str, Any]]:
    """Every page of ``records`` through one system; the scored pages and run notes."""
    from docrooter.errors import DocrooterError

    engine, layout_name = SYSTEMS[name]
    layout: Any = layout_name
    if layout_name != "heuristic":
        from docrooter.layout import get_layout_engine

        layout = get_layout_engine(layout_name)  # one session for the run
    pages: list[ScoredPage] = []
    errors: dict[str, str] = {}
    no_text: list[str] = []
    started = time.perf_counter()
    try:
        for k, record in enumerate(records):
            path = doclaynet.pdf_path(split, record["id"])
            width, height = page_size(path)
            gold, cells = gold_page(record, width, height)
            try:
                pred, warnings = predict(path, engine, layout)
                if "no_text_layer" in warnings:
                    no_text.append(record["id"])
            except DocrooterError as exc:  # unreadable page: scored with no predictions
                errors[record["id"]] = str(exc)[:200]
                pred = []
            pages.append(ScoredPage(record["id"], record["category"], pred, gold, cells))
            if verbose and (k + 1) % 50 == 0:
                print(f"  {name}: {k + 1}/{len(records)}", flush=True)
    finally:
        if hasattr(layout, "close"):
            layout.close()
    seconds = time.perf_counter() - started
    return pages, {"seconds": round(seconds, 2),
                   "seconds_per_page": round(seconds / max(1, len(records)), 4),
                   "errors": errors, "no_text_layer": no_text}  # fmt: skip


def target_check(result: Mapping[str, Any]) -> dict[str, Any]:
    """Spec §10.1 target: F1 >= 0.8 for Text, Table and Picture on the financial-report and
    scientific-article pages, per heuristic system."""
    out = {}
    for name, system in result["systems"].items():
        if not name.startswith("heuristic"):
            continue
        cells = {}
        for category in TARGET_CATEGORIES:
            f1 = system["per_category"].get(category, {}).get("f1", {})
            cells[category] = {c: f1.get(c) for c in TARGET_CLASSES}
        met = all(v is not None and v >= TARGET_F1 for row in cells.values() for v in row.values())
        out[name] = {"f1": cells, "met": met}
    return out


def manifest_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """SHA-256 over the sampled pages' ids and PDF checksums, in manifest order."""
    text = "\n".join(f"{r['id']} {r['pdf_sha256']}" for r in records)
    return hashlib.sha256(text.encode()).hexdigest()


def evaluate(
    split: str, systems: Sequence[str], *, values: Mapping[str, float] | None = None,
    limit: int | None = None, verbose: bool = False,
) -> dict[str, Any]:  # fmt: skip
    """Score ``systems`` on the sampled pages of ``split``, with threshold ``values``."""
    records = doclaynet.load_manifest(split)[:limit]
    result: dict[str, Any] = {
        "benchmark": "elements-doclaynet", "split": split,
        "date": datetime.date.today().isoformat(), "commit": git_revision(),
        "source": {"repo": doclaynet.REPO, "revision": doclaynet.REVISION,
                   "licence": doclaynet.LICENCE, "citation": doclaynet.CITATION},
        "sample": {"pages": len(records), "seed": doclaynet.SEED,
                   "digest": manifest_digest(records),
                   "per_category": dict(sorted(Counter(r["category"] for r in records).items()))},
        "iou": IOU, "overrides": dict(values or {}), "env": collect_env(), "systems": {},
    }  # fmt: skip
    with overrides(values or {}):
        for name in systems:
            pages, notes = run_system(name, split, records, verbose=verbose)
            result["systems"][name] = {**score(pages), **notes}
            if verbose:
                print(f"{name}: macro F1 {result['systems'][name]['macro']:.4f} "
                      f"({notes['seconds']:.0f}s)", flush=True)  # fmt: skip
    result["target"] = target_check(result)
    return result


# --- report --------------------------------------------------------------------------------


def _f(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render(result: Mapping[str, Any]) -> str:
    """Markdown tables for a result: per class, per category, text coverage, the target."""
    systems = result["systems"]
    names = list(systems)
    out = [
        f"Split: **{result['split']}**, {result['sample']['pages']} pages "
        f"({', '.join(f'{c} {n}' for c, n in result['sample']['per_category'].items())}); "
        f"sample digest `{result['sample']['digest'][:12]}`; commit `{result['commit']}`; "
        f"{result['date']}. Matching: one-to-one, greedy by IoU >= {result['iou']}, within a "
        "class. Pages without a text layer (scans; no OCR engine runs in this benchmark): "
        + ", ".join(f"{n} {len(systems[n].get('no_text_layer', []))}" for n in names)
        + "; load errors: "
        + ", ".join(f"{n} {len(systems[n].get('errors', {}))}" for n in names)
        + ".",
        "",
        "### Per class (F1; precision / recall below)",
        "",
        "| Class | Gold | " + " | ".join(names) + " |",
        "| --- | --- | " + " | ".join("---" for _ in names) + " |",
    ]
    for c in CLASSES:
        first = systems[names[0]]["per_class"][c]
        gold = first["tp"] + first["fn"]
        cells = []
        for n in names:
            v = systems[n]["per_class"][c]
            cells.append(f"{_f(v['f1'])} ({_f(v['precision'], 2)} / {_f(v['recall'], 2)})")
        out.append(f"| {c} | {gold} | " + " | ".join(cells) + " |")
    out.append("| **Macro F1** | | " + " | ".join(f"**{_f(systems[n]['macro'])}**" for n in names)
               + " |")  # fmt: skip
    out.append("| s/page | | " + " | ".join(_f(systems[n]["seconds_per_page"], 3) for n in names)
               + " |")  # fmt: skip
    out += ["", "### Per category (macro F1)", "",
            "| Category | Pages | " + " | ".join(names) + " |",
            "| --- | --- | " + " | ".join("---" for _ in names) + " |"]  # fmt: skip
    for category, row in systems[names[0]]["per_category"].items():
        out.append(f"| {category} | {row['pages']} | "
                   + " | ".join(_f(systems[n]["per_category"][category]["macro"]) for n in names)
                   + " |")  # fmt: skip
    out += ["", "### Text coverage (share of gold characters in an element of the right kind)",
            "", "| Class | Gold chars | " + " | ".join(names) + " |",
            "| --- | --- | " + " | ".join("---" for _ in names) + " |"]  # fmt: skip
    for c in CLASSES:
        chars = systems[names[0]]["text_coverage"][c]["chars"]
        out.append(f"| {c} | {chars} | "
                   + " | ".join(_f(systems[n]["text_coverage"][c]["share"]) for n in names)
                   + " |")  # fmt: skip
    if result.get("target"):
        out += ["", f"### Spec §10.1 target: heuristic F1 >= {TARGET_F1} for "
                f"{', '.join(TARGET_CLASSES)} on {' and '.join(TARGET_CATEGORIES)}", "",
                "| System | Category | " + " | ".join(TARGET_CLASSES) + " | Met |",
                "| --- | --- | " + "--- | " * len(TARGET_CLASSES) + "--- |"]  # fmt: skip
        for name, check in result["target"].items():
            for category, row in check["f1"].items():
                out.append(f"| {name} | {category} | "
                           + " | ".join(_f(row[c]) for c in TARGET_CLASSES)
                           + f" | {'yes' if check['met'] else 'no'} |")  # fmt: skip
    return "\n".join(out)


def update_docs(path: Path, tables: str) -> None:
    """Replace the marked results block of ``path`` (which must hold the markers)."""
    text = path.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise SystemExit(f"{path} has no {BEGIN} ... {END} block")
    head, rest = text.split(BEGIN, 1)
    tail = rest.split(END, 1)[1]
    path.write_text(f"{head}{BEGIN}\n{tables}\n{END}{tail}", encoding="utf-8")


def _parse_set(items: Sequence[str]) -> dict[str, float]:
    values = {}
    for item in items:
        name, _, value = item.partition("=")
        if not value:
            raise SystemExit(f"--set wants NAME=VALUE, got {item!r}")
        values[name.strip()] = float(value)
    return values


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Element kinds on DocLayNet (spec §10.1)")
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--systems", default="heuristic-pymupdf",
                        help=f"comma-separated, from {', '.join(SYSTEMS)}")  # fmt: skip
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="override an element threshold for this run (train only)")  # fmt: skip
    parser.add_argument("--limit", type=int, default=None, help="first N pages only")
    parser.add_argument("--out", type=Path, default=None, help="results dir (default: dated)")
    parser.add_argument("--docs", type=Path, default=None, help="also update this Markdown file")
    _guard.add_guard_argument(parser)
    args = parser.parse_args(argv)
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = [s for s in systems if s not in SYSTEMS]
    if unknown:
        raise SystemExit(f"unknown system: {', '.join(unknown)}")
    values = _parse_set(args.set)
    if values and args.split == "test":
        raise SystemExit("thresholds are tuned on train pages only: no --set with --split test")
    _guard.single_model_process("elements_bench", allow_concurrent=args.allow_concurrent)
    result = evaluate(args.split, systems, values=values, limit=args.limit, verbose=True)
    out_dir = args.out or RESULTS_ROOT / result["date"]
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.split == "test" else f"-{args.split}"
    path = out_dir / f"elements-doclaynet{suffix}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tables = render(result)
    print(tables)
    print(f"wrote {path}")
    if args.docs:
        update_docs(args.docs, tables)


if __name__ == "__main__":
    main()
