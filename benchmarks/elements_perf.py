"""Element-mode speed and page routing on the dev corpus (v0.4 task E5c; spec §7.4, §10.5).

``PYTHONPATH=src .venv/bin/python -m benchmarks.elements_perf --speed --routing``

- ``--speed``: every dev PDF is loaded with the default ``load_pdf`` (fast) and with
  ``load_pdf(elements=True)`` (pymupdf, no OCR engine), one after the other, after one warm-up
  document. Writes ``element-speed.json``: seconds per document in both modes, element-mode
  stage timings, the totals and their ratio (target: element mode <= 2.5x fast).
- ``--routing``: ``extract_page_objects`` and ``route_page`` on every page of every dev PDF.
  Writes ``routing-dev.json``: decision counts and every page decided ``ocr`` (replace) or
  ``text+ocr`` (merge), with its reason and its text-layer character count. Target: no page
  with a usable text layer is replaced.

``--baseline DIR`` embeds the totals of an earlier run (the same script against an older
``src``) in the result, as ``baseline``. No models, no OCR engine; the dev corpus only.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from benchmarks import _guard, corpora
from benchmarks.toc import RESULTS_ROOT, collect_env

SPEED_FILE = "element-speed.json"
ROUTING_FILE = "routing-dev.json"
SPEED_TARGET = 2.5  # element mode total seconds <= this x fast's


def src_revision() -> dict[str, Any]:
    """The commit of the ``docrooter`` package actually imported (a worktree for a baseline
    run), with ``dirty`` when its ``src`` has uncommitted changes."""
    import docrooter

    root = Path(docrooter.__file__).resolve().parent
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True,
            check=True,
        ).stdout.strip()  # fmt: skip
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "."], cwd=root, capture_output=True,
            text=True, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": sha, "dirty": bool(dirty)}


def run_speed(docs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    from docrooter.pdf.load import load_pdf

    load_pdf(corpora.DEV / docs[0]["file"], elements=True)  # warm-up: imports, font caches
    rows = []
    for record in docs:
        path = corpora.DEV / record["file"]
        started = time.perf_counter()
        fast = load_pdf(path)
        fast_s = time.perf_counter() - started
        started = time.perf_counter()
        doc = load_pdf(path, elements=True)
        element_s = time.perf_counter() - started
        rows.append({
            "id": record["id"], "pages": fast.meta["n_pages"],
            "fast_s": round(fast_s, 3), "elements_s": round(element_s, 3),
            "ratio": round(element_s / fast_s, 2) if fast_s else None,
            "element_timings": doc.meta["timings"],
        })  # fmt: skip
        print(f"{record['id']} {rows[-1]['pages']:4d}p fast {fast_s:6.2f}s "
              f"elements {element_s:6.2f}s", flush=True)  # fmt: skip
    fast_total = sum(r["fast_s"] for r in rows)
    element_total = sum(r["elements_s"] for r in rows)
    stages: Counter[str] = Counter()
    for r in rows:
        stages.update(r["element_timings"])
    return {
        "documents": len(rows),
        "pages": sum(r["pages"] for r in rows),
        "fast_total_s": round(fast_total, 2),
        "elements_total_s": round(element_total, 2),
        "fast_s_per_doc": round(fast_total / len(rows), 3),
        "elements_s_per_doc": round(element_total / len(rows), 3),
        "ratio": round(element_total / fast_total, 3),
        "target_ratio": SPEED_TARGET,
        "meets_target": element_total <= SPEED_TARGET * fast_total,
        "element_stage_totals_s": {k: round(v, 2) for k, v in stages.items()},
        "per_document": rows,
    }


def run_routing(docs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    from docrooter.pdf.objects import extract_page_objects
    from docrooter.pdf.routing import route_page

    counts: Counter[str] = Counter()
    replaced: list[dict[str, Any]] = []
    merged: list[dict[str, Any]] = []
    region_pages = regions = 0
    per_doc = {}
    for record in docs:
        pages = extract_page_objects(corpora.DEV / record["file"], engine="pymupdf")
        doc_counts: Counter[str] = Counter()
        for objs in pages:
            task, found, decision = route_page(objs)
            key = decision.task if decision.scope == "page" else "region"
            counts[key] += 1
            doc_counts[key] += 1
            if found:
                region_pages += 1
                regions += len(found)
            entry = {
                "id": record["id"], "page": objs.page, "chars": objs.char_count,
                "garbled_chars": objs.garbled_chars, "reason": decision.reason,
                "metrics": {k: round(v, 4) for k, v in decision.metrics.items()},
            }  # fmt: skip
            if task == "ocr":
                replaced.append(entry)
            elif task == "text+ocr":
                merged.append(entry)
        per_doc[record["id"]] = dict(doc_counts)
        print(record["id"], dict(doc_counts), flush=True)
    with_text = [e for e in replaced if e["chars"] > 0]
    return {
        "documents": len(docs),
        "pages": sum(counts.values()),
        "decisions": dict(counts),
        "replaced_pages": len(replaced),
        "replaced_pages_with_text_layer": len(with_text),
        "merged_pages": len(merged),
        "region_pages": region_pages,
        "regions": regions,
        "replaced": replaced,
        "merged": merged,
        "per_document": per_doc,
    }


def _summary(result: dict[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {k: result[k] for k in keys if k in result}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="element-mode speed and routing (dev corpus)")
    parser.add_argument("--speed", action="store_true")
    parser.add_argument("--routing", action="store_true")
    parser.add_argument("--out", type=Path, default=None, help="results dir (default: dated)")
    parser.add_argument("--baseline", type=Path, default=None, help="dir of an earlier run")
    parser.add_argument("--docs", default=None, help="comma-separated dev ids (default: all)")
    args = parser.parse_args(argv)
    if not (args.speed or args.routing):
        parser.error("pass --speed and/or --routing")
    _guard.apply_env_caps()  # no models here, so no single-process check

    docs = corpora.read_manifest(corpora.DEV)
    if args.docs:
        wanted = set(args.docs.split(","))
        docs = [d for d in docs if d["id"] in wanted]
    out_dir = args.out or RESULTS_ROOT / datetime.date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "date": datetime.date.today().isoformat(),
        "corpus": corpora.relative(corpora.DEV),
        "engine": "pymupdf",
        "ocr": None,
        **src_revision(),
        "env": collect_env(),
    }
    jobs = []
    if args.speed:
        keys = ("fast_total_s", "elements_total_s", "fast_s_per_doc", "elements_s_per_doc",
                "ratio", "element_stage_totals_s", "commit")  # fmt: skip
        jobs.append((SPEED_FILE, run_speed, keys))
    if args.routing:
        keys = ("decisions", "replaced_pages", "replaced_pages_with_text_layer",
                "merged_pages", "region_pages", "regions", "commit")  # fmt: skip
        jobs.append((ROUTING_FILE, run_routing, keys))
    for filename, run, keys in jobs:
        result = meta | run(docs)
        if args.baseline is not None:
            earlier = json.loads((args.baseline / filename).read_text(encoding="utf-8"))
            result["baseline"] = _summary(earlier, keys)
            if filename == ROUTING_FILE:
                result["baseline"]["replaced"] = [
                    {k: e[k] for k in ("id", "page", "chars", "reason")}
                    for e in earlier["replaced"]
                ]
        path = out_dir / filename
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
