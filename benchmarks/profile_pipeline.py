"""Where the time goes in the default local pipeline (v0.3 spec section 6, the Rust decision).

``PYTHONPATH=src uv run --extra all python -m benchmarks.profile_pipeline``

For each of ten dev-corpus PDFs (1 to 642 pages) the pipeline is ``load()`` (default engine),
``classify_document`` against ``builtin:doc_types`` and ``normalize_sections`` for the predicted
type, with UltraBERT-Embed-150M on the default device. The model is loaded once, before any
timing, and a warm-up pass over all ten documents fills the label-vector caches and compiles the
device kernels, so the timed passes measure the steady state of a long-running process (the
playground server). The text-embedding cache is cleared before every timed document.

Each document runs twice:

1. **Unprofiled.** Wall time per stage from the pipeline events (``extract``, ``signals``,
   ``headings``, ``assemble``, ``view``, ``classify``, ``normalize``).
2. **Under pyinstrument.** Every sample is put in one bucket from its call stack. In this order:
   - ``pymupdf``: inside the pymupdf package (``get_text`` above all);
   - ``tokenisation``: inside a tokenizer (``tokenizers``, ``transformers`` tokenization
     modules);
   - ``inference``: inside torch, transformers or sentence-transformers;
   - otherwise by the innermost docrooter stage on the stack: ``line_assembly`` (our code under
     ``extract_lines``), ``signals``, ``headings``, ``tree_assembly`` (``_assemble``,
     ``build_document``, metadata title), ``view`` (view rendering and keyphrases), else
     ``other`` (classifier logic, scoring, events, ...).

``own_python`` is every sample not inside pymupdf, a tokenizer or the model, the quantity the
spec's 30% rule is about. pyinstrument hooks every Python call, which inflates call-heavy pure
Python, so its shares overstate our own code a little; the unprofiled stage times are reported
next to them.

Results: ``benchmarks/results/<date>/profile.json``; a Markdown summary is printed.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from benchmarks import _guard, corpora

RESULTS_ROOT = Path(__file__).parent / "results"
CACHE = Path(__file__).parent / ".cache" / "profile"
BACKEND = "embedding:dmis-lab/UltraBERT-Embed-150M"
# Ten dev documents spanning 1 to 642 pages and most document families.
DOC_IDS = (
    "doc_017", "doc_043", "doc_037", "doc_027", "doc_041",
    "doc_045", "doc_002", "doc_031", "doc_039", "doc_042",
)  # fmt: skip
BUCKETS = (
    "pymupdf",
    "line_assembly",
    "signals",
    "headings",
    "tree_assembly",
    "view",
    "tokenisation",
    "inference",
    "other",
)
LIBRARY_BUCKETS = ("pymupdf", "tokenisation", "inference")
STAGES = ("extract", "signals", "headings", "assemble", "view", "classify", "normalize")
INTERVAL = 0.001

# Innermost docrooter function on the stack -> stage bucket.
_STAGE_FUNCTIONS = {
    "extract_lines": "line_assembly",
    "compute_signals": "signals",
    "read_bookmarks": "signals",
    "find_headings": "headings",
    "_assemble": "tree_assembly",
    "build_document": "tree_assembly",
    "_metadata_title": "tree_assembly",
    "render": "view",
    "render_all": "view",
}


def _frame(identifier: str) -> tuple[str, str]:
    """``(function, file)`` of a pyinstrument frame identifier."""
    parts = identifier.split("\x00")
    return parts[0], parts[1] if len(parts) > 1 else ""


def bucket_of(stack: Sequence[str]) -> str:
    """The bucket of one sample, from its pyinstrument frame identifiers (outermost first)."""
    frames = [_frame(identifier) for identifier in stack]
    files = [file.replace("\\", "/") for _, file in frames]
    if any("/pymupdf/" in f or "/fitz/" in f for f in files):
        return "pymupdf"
    if any(
        "/tokenizers/" in f or ("/transformers/" in f and "tokeniz" in f.rsplit("/", 1)[-1])
        for f in files
    ):
        return "tokenisation"
    if any(
        "/torch/" in f or "/transformers/" in f or "/sentence_transformers/" in f for f in files
    ):
        return "inference"
    for function, file in reversed(frames):
        if "/docrooter/" in file.replace("\\", "/") and function in _STAGE_FUNCTIONS:
            return _STAGE_FUNCTIONS[function]
    return "other"


def split_samples(records: Sequence[tuple[Sequence[str], float]]) -> dict[str, float]:
    """Seconds per bucket from pyinstrument ``frame_records``."""
    seconds = dict.fromkeys(BUCKETS, 0.0)
    for stack, duration in records:
        seconds[bucket_of(stack)] += duration
    return seconds


def stage_seconds(recorded: Sequence[Any]) -> dict[str, float]:
    """Wall seconds per stage from pipeline events (summed over repeats of a stage)."""
    open_at: dict[str, float] = {}
    out: defaultdict[str, float] = defaultdict(float)
    for event in recorded:
        if event.status == "start":
            open_at[event.stage] = event.t
        elif event.status == "end" and event.stage in open_at:
            out[event.stage] += event.t - open_at.pop(event.stage)
    return {stage: round(out.get(stage, 0.0), 4) for stage in STAGES}


def own_python(split: dict[str, float]) -> float:
    return sum(v for k, v in split.items() if k not in LIBRARY_BUCKETS)


def _pipeline(path: Path, backend: Any, doc_types: Any, sink: Any) -> dict[str, Any]:
    from docrooter import classify_document, events, load, normalize_sections

    with events.pipeline_run(sink):
        doc = load(path)
        result = classify_document(doc, doc_types, backend=backend)
        doc_type = result.best.node_ids[-1] if result.best.node_ids else None
        norm = normalize_sections(doc, doc_type=doc_type, backend=backend)
    return {
        "doc_type": doc_type,
        "sections": len(doc.sections),
        "normalised": len(norm.sections),
    }


def _clear_text_cache(backend: Any) -> None:
    """Forget the text embeddings (not the label vectors), so each run encodes its texts."""
    cache = getattr(backend, "_cache", None)
    if cache is not None:
        cache.clear()


def profile_doc(path: Path, backend: Any, doc_types: Any) -> dict[str, Any]:
    from pyinstrument import Profiler

    recorded: list[Any] = []
    _clear_text_cache(backend)
    start = time.perf_counter()
    outcome = _pipeline(path, backend, doc_types, recorded.append)
    wall = time.perf_counter() - start

    _clear_text_cache(backend)
    profiler = Profiler(interval=INTERVAL)
    profiler.start()
    _pipeline(path, backend, doc_types, lambda event: None)
    session = profiler.stop()
    split = split_samples(session.frame_records)
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"{path.stem}.txt").write_text(
        profiler.output_text(unicode=True, color=False), encoding="utf-8"
    )
    return {
        **outcome,
        "seconds": round(wall, 4),
        "stages": stage_seconds(recorded),
        "profiled_seconds": round(session.duration, 4),
        "split": {k: round(v, 4) for k, v in split.items()},
        "own_python_share": round(own_python(split) / max(sum(split.values()), 1e-9), 4),
    }


def upper_median(values: Sequence[float]) -> float:
    """The element at index ``n // 2`` of the sorted values: the median for an odd count, the
    upper of the two middle values for an even one (10 documents: the 6th smallest)."""
    return sorted(values)[len(values) // 2]


def imported_docrooter(root: Path) -> dict[str, str | None]:
    """Which docrooter this run imported and where it came from, since ``PYTHONPATH=src`` and an
    installed wheel can differ: the path of the imported package (relative to ``root`` when it
    lives in this checkout, else with the home directory shown as ``~``) and the short commit of
    the repository containing it, with ``+dirty`` when the package has uncommitted changes
    (None when it is not in a git checkout)."""
    import docrooter

    package = Path(docrooter.__file__).resolve().parent
    try:
        shown = str(package.relative_to(root.resolve()))
    except ValueError:
        home = str(Path.home())
        shown = (
            str(package).replace(home, "~", 1) if str(package).startswith(home) else str(package)
        )
    revision = None
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=package, capture_output=True, text=True,
            check=True,
        ).stdout.strip()  # fmt: skip
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "."], cwd=package, capture_output=True,
            text=True, check=True,
        ).stdout.strip()  # fmt: skip
        revision = f"{sha}+dirty" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"docrooter_path": shown, "docrooter_revision": revision}


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    split = {b: round(sum(r["split"][b] for r in rows), 4) for b in BUCKETS}
    sampled = sum(split.values())
    return {
        "pages": sum(r["pages"] for r in rows),
        "seconds": round(sum(r["seconds"] for r in rows), 4),
        "stages": {s: round(sum(r["stages"][s] for r in rows), 4) for s in STAGES},
        "profiled_seconds": round(sum(r["profiled_seconds"] for r in rows), 4),
        "split": split,
        "split_share": {b: round(v / max(sampled, 1e-9), 4) for b, v in split.items()},
        "own_python_share": round(own_python(split) / max(sampled, 1e-9), 4),
        # the upper median (see upper_median); the key is kept so recorded results still parse
        "own_python_share_median_doc": round(
            upper_median([r["own_python_share"] for r in rows]), 4
        ),
    }


def render(result: dict[str, Any]) -> str:
    header = "| Doc | Pages | Wall s | " + " | ".join(BUCKETS) + " | Own Python |"
    lines = [header, "| --- | ---: | ---: | " + " | ".join("---:" for _ in BUCKETS) + " | ---: |"]
    for row in [*result["docs"], {"id": "**all**", **result["aggregate"]}]:
        total = sum(row["split"].values()) or 1.0
        cells = " | ".join(f"{100 * row['split'][b] / total:.0f}%" for b in BUCKETS)
        lines.append(
            f"| {row['id']} | {row['pages']} | {row['seconds']:.2f} | {cells} | "
            f"{100 * row['own_python_share']:.0f}% |"
        )
    agg = result["aggregate"]
    lines.append("")
    lines.append(
        f"Own Python: {100 * agg['own_python_share']:.0f}% of all samples pooled, "
        f"{100 * agg['own_python_share_median_doc']:.0f}% upper median over documents."
    )
    if "docrooter_path" in result:
        lines.append(
            f"docrooter imported from `{result['docrooter_path']}` "
            f"at `{result.get('docrooter_revision') or 'no git checkout'}`."
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile the default local pipeline")
    parser.add_argument("--docs", default=",".join(DOC_IDS), help="comma-separated dev ids")
    parser.add_argument("--out", type=Path, default=None, help="results dir (default: dated)")
    parser.add_argument("--backend", default=BACKEND, help=f"backend spec (default {BACKEND})")
    _guard.add_guard_argument(parser)
    args = parser.parse_args(argv)
    _guard.single_model_process("profile_pipeline", allow_concurrent=args.allow_concurrent)

    from benchmarks.toc import collect_env, git_revision
    from docrooter.backends import registry
    from docrooter.taxonomies import builtin_taxonomy

    manifest = {row["id"]: row for row in corpora.read_manifest(corpora.DEV)}
    wanted = [manifest[i] for i in args.docs.split(",") if i]
    paths = [corpora.DEV / row["file"] for row in wanted]
    doc_types = builtin_taxonomy("doc_types")

    backend = registry.resolve_backend(args.backend)
    loaded = time.perf_counter()
    backend.tokenizer()  # loads the model, outside every timing
    model_load_seconds = time.perf_counter() - loaded
    for path in paths:  # warm-up: label vectors and device kernels
        _pipeline(path, backend, doc_types, None)

    loadavg_start = list(os.getloadavg())
    rows = []
    for row, path in zip(wanted, paths, strict=True):
        profiled = profile_doc(path, backend, doc_types)
        rows.append({"id": row["id"], "pages": row["pages"], **profiled})
        print(
            f"{row['id']} {row['pages']:4d}p {profiled['seconds']:7.2f}s "
            f"own Python {100 * profiled['own_python_share']:.0f}%",
            flush=True,
        )
    result = {
        "date": datetime.date.today().isoformat(),
        "commit": git_revision(),
        **imported_docrooter(Path(__file__).parent.parent),
        "env": collect_env(),
        "loadavg_start": loadavg_start,
        "loadavg_end": list(os.getloadavg()),
        "backend": args.backend,
        "device": getattr(backend, "_device_in_use", None),
        "model_load_seconds": round(model_load_seconds, 2),
        "interval": INTERVAL,
        "docs": rows,
        "aggregate": aggregate(rows),
    }
    backend.close()
    out_dir = args.out or RESULTS_ROOT / result["date"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "profile.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(render(result))
    print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
