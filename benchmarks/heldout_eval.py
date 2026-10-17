"""The frozen held-out generalisation evaluation (v0.3 spec section 3).

``PYTHONPATH=src uv run --extra models --extra pdf --extra pdf-mit --extra jev --extra bench \
    python -m benchmarks.heldout_eval run``

``run`` records the frozen commit (HEAD when the run starts; it refuses to start while the
benchmark code has uncommitted changes), then runs the existing harnesses on
``benchmarks/heldout/`` one after another, each in its own process so only one model process
exists at a time:

1. ``benchmarks.toc --corpus heldout`` with the pymupdf, pdfplumber and both engines (no models),
   on the held-out PDFs with at least 5 bookmarks;
2. ``benchmarks.pdf_bench --corpus heldout`` with :data:`PDF_SYSTEMS` on all 40 PDFs: the
   dev-benchmark systems plus the two "trained on dev, tested on held-out" systems that replace
   the dev leave-one-out ones;
3. ``benchmarks.normalize_bench --corpus heldout`` (UltraBERT, bge-small, Jev, package
   defaults) on the held-out silver set ``benchmarks/heldout/section_gold.jsonl``.

Results go to ``benchmarks/results/<date>/heldout-{toc,pdf,normalize}.json``; each gets a
``freeze`` record (frozen commit, label commits, start/end time, the harness's exit status and
whether the benchmark code was still unchanged when it finished). ``render`` writes the
generated block of ``docs/benchmarks-heldout.md`` and the dev-vs-held-out table of
``docs/benchmarks-pdf.md`` from those files and the dev results.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks import _guard

ROOT = Path(__file__).parent.parent
RESULTS_ROOT = Path(__file__).parent / "results"
HELDOUT_DOCS = ROOT / "docs" / "benchmarks-heldout.md"
PDF_DOCS = ROOT / "docs" / "benchmarks-pdf.md"

TOC_ENGINES = ("pymupdf", "pdfplumber", "both")
PDF_SYSTEMS = (
    "jev-view",
    "jev-chunks",
    "ultrabert-view",
    "ultrabert-hybrid",
    "bge-view",
    "ultrabert-dev-logistic",
    "ultrabert-bge-dev-ensemble",
)
NORMALIZE_SYSTEMS = ("ultrabert", "bge-small", "jev")
STEPS: dict[str, tuple[str, str, list[str]]] = {
    # step: (module, results file, arguments after --corpus heldout --out <dir>)
    "toc": ("benchmarks.toc", "heldout-toc.json", ["--engines", ",".join(TOC_ENGINES)]),
    "pdf": ("benchmarks.pdf_bench", "heldout-pdf.json", ["--systems", ",".join(PDF_SYSTEMS)]),
    "normalize": (
        "benchmarks.normalize_bench",
        "heldout-normalize.json",
        ["--systems", ",".join(NORMALIZE_SYSTEMS)],
    ),
}
MODEL_STEPS = ("pdf", "normalize")
# Code that can change a benchmark number. The playground UI and the CLI are not imported by any
# harness (another task edits them concurrently), and results are outputs.
CODE_PATHS = (
    "src",
    "benchmarks",
    ":(exclude)benchmarks/results",
    ":(exclude)src/docrooter/ui",
    ":(exclude)src/docrooter/cli.py",
)
LABEL_FILES = ("benchmarks/heldout/manifest.jsonl", "benchmarks/heldout/section_gold.jsonl")
# Dev systems and the held-out system that stands for each (the dev leave-one-out systems become
# "trained on all 51 dev docs, tested on held-out").
DOCTYPE_PAIRS = (
    ("jev-view", "jev-view"),
    ("jev-chunks", "jev-chunks"),
    ("ultrabert-view", "ultrabert-view"),
    ("ultrabert-hybrid", "ultrabert-hybrid"),
    ("bge-view", "bge-view"),
    ("ultrabert-loo", "ultrabert-dev-logistic"),
    ("ultrabert-bge-loo", "ultrabert-bge-dev-ensemble"),
)
COMPARE_SECTION = "## Dev vs held-out"
BEGIN, END = "<!-- heldout-benchmark:begin -->", "<!-- heldout-benchmark:end -->"
CMP_BEGIN, CMP_END = "<!-- heldout-compare:begin -->", "<!-- heldout-compare:end -->"

Git = Callable[[Sequence[str]], str]


def git(args: Sequence[str]) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def code_changes(since: str | None = None, run: Git = git) -> list[str]:
    """Benchmark-relevant paths (:data:`CODE_PATHS`) with uncommitted changes, plus, with
    ``since``, those committed after ``since``. Empty means the code is what ``since`` (or
    HEAD) holds."""
    status = run(["status", "--porcelain", "--", *CODE_PATHS])
    changed = [line[3:] for line in status.splitlines() if line.strip()]
    if since:
        diff = run(["diff", "--name-only", since, "HEAD", "--", *CODE_PATHS])
        changed += [line for line in diff.splitlines() if line.strip()]
    return sorted(set(changed))


def last_commit(path: str, run: Git = git) -> str:
    return run(["log", "-1", "--format=%h", "--", path])


def freeze_record(run: Git = git) -> dict[str, Any]:
    """The frozen commit and the commits that fixed the labels; raises ``RuntimeError`` while
    benchmark code has uncommitted changes (the run must be reproducible from the commit)."""
    dirty = code_changes(run=run)
    if dirty:
        raise RuntimeError(f"uncommitted benchmark code, commit it first: {dirty}")
    return {
        "frozen_commit": run(["rev-parse", "--short", "HEAD"]),
        "label_commits": {path: last_commit(path, run) for path in LABEL_FILES},
        "started_at": now(),
        "code_paths": list(CODE_PATHS),
        "steps": {},
    }


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def stamp(path: Path, freeze: Mapping[str, Any]) -> None:
    """Add (or replace) the ``freeze`` record of a results file, keeping everything else."""
    if not path.exists():
        return
    result = json.loads(path.read_text(encoding="utf-8"))
    result["freeze"] = dict(freeze)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def wait_for_other_model_processes(poll_s: float = 60.0) -> None:
    """Block while another model benchmark (or ``docrooter serve``) runs."""
    while others := _guard.other_model_processes():
        print(f"waiting for {len(others)} other model process(es): {others[0][1][:100]}")
        time.sleep(poll_s)


def run_steps(steps: Sequence[str], out_dir: Path) -> dict[str, Any]:
    freeze = freeze_record()
    print(f"frozen commit {freeze['frozen_commit']}", flush=True)
    for step in steps:
        module, filename, extra = STEPS[step]
        if step in MODEL_STEPS:
            wait_for_other_model_processes()
        started = now()
        argv = [sys.executable, "-m", module, "--corpus", "heldout", "--out", str(out_dir), *extra]
        print(f"== {step}: {' '.join(argv[1:])}", flush=True)
        status = subprocess.run(argv, cwd=ROOT).returncode
        changed = code_changes(freeze["frozen_commit"])
        freeze["steps"][step] = {
            "argv": argv[1:],
            "started_at": started,
            "finished_at": now(),
            "returncode": status,
            "head_at_end": git(["rev-parse", "--short", "HEAD"]),
            "code_changed_since_freeze": changed,
        }
        for name in (STEPS[s][1] for s in steps):
            stamp(out_dir / name, freeze)
        if changed:
            print(f"WARNING: benchmark code changed during the run: {changed}", flush=True)
    return freeze


# --- report ------------------------------------------------------------------------------


def _f(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _delta(dev: float | None, held: float | None) -> str:
    if dev is None or held is None:
        return "-"
    return f"{100 * (held - dev):+.1f} pts"


def _doctype_metrics(result: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    entry = result.get("systems", {}).get(name) or {}
    return entry.get("metrics")


def comparison_rows(
    dev: Mapping[str, Mapping[str, Any]], held: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """One row per compared quantity: ``{"task", "system", "metric", "dev", "heldout", "n_dev",
    "n_heldout"}``. ``dev`` and ``held`` map ``"toc"``, ``"pdf"`` and ``"normalize"`` to the
    parsed results files (a missing one is skipped)."""
    rows: list[dict[str, Any]] = []
    toc_d, toc_h = dev.get("toc", {}).get("engines", {}), held.get("toc", {}).get("engines", {})
    for engine in TOC_ENGINES:
        d, h = toc_d.get(engine), toc_h.get(engine)
        if not d or not h:
            continue
        for metric, key, field in (
            ("F1 micro (d<=2)", "depth2", "f1"),
            ("F1 macro (d<=2)", "depth2", "macro_f1"),
            ("F1 micro (all)", "all", "f1"),
        ):
            rows.append(
                {
                    "task": "TOC",
                    "system": engine,
                    "metric": metric,
                    "dev": d[key][field],
                    "heldout": h[key][field],
                    "n_dev": d["n_docs"],
                    "n_heldout": h["n_docs"],
                }
            )
    pdf_d, pdf_h = dev.get("pdf", {}), held.get("pdf", {})
    for dev_name, held_name in DOCTYPE_PAIRS:
        d, h = _doctype_metrics(pdf_d, dev_name), _doctype_metrics(pdf_h, held_name)
        if not d or not h:
            continue
        label = dev_name if dev_name == held_name else f"{dev_name} -> {held_name}"
        for metric, field in (("leaf accuracy", "leaf_accuracy"), ("macro F1", "macro_f1")):
            rows.append(
                {
                    "task": "Doc type",
                    "system": label,
                    "metric": metric,
                    "dev": d[field],
                    "heldout": h[field],
                    "n_dev": d["n"],
                    "n_heldout": h["n"],
                }
            )
    norm_d, norm_h = dev.get("normalize", {}), held.get("normalize", {})
    for name in (*NORMALIZE_SYSTEMS, "auto_doc_type"):
        if name == "auto_doc_type":
            d, h = norm_d.get(name), norm_h.get(name)
        else:
            d, h = norm_d.get("systems", {}).get(name), norm_h.get("systems", {}).get(name)
        if not d or not h or "overall" not in d or "overall" not in h:
            continue
        for metric, field in (("accuracy", "accuracy"), ("macro F1", "macro_f1")):
            rows.append(
                {
                    "task": "Normalisation",
                    "system": name,
                    "metric": metric,
                    "dev": d["overall"][field],
                    "heldout": h["overall"][field],
                    "n_dev": d["overall"]["n_scored"],
                    "n_heldout": h["overall"]["n_scored"],
                }
            )
    return rows


def render_comparison(rows: Sequence[Mapping[str, Any]]) -> str:
    """The dev-vs-held-out Markdown table."""
    out = [
        "| Task | System | Metric | Dev | Held-out | Change | n dev / held-out |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        out.append(
            f"| {r['task']} | `{r['system']}` | {r['metric']} | {_f(r['dev'])} | "
            f"{_f(r['heldout'])} | {_delta(r['dev'], r['heldout'])} | "
            f"{r['n_dev']} / {r['n_heldout']} |"
        )
    return "\n".join(out)


def replace_block(text: str, begin: str, end: str, block: str, section: str) -> str:
    """``text`` with the ``begin``/``end`` block replaced by ``block``; when there is none, the
    block is appended under a new ``section`` heading at the end. Nothing else changes."""
    wrapped = f"{begin}\n{block}\n{end}"
    if begin in text and end in text:
        head, rest = text.split(begin, 1)
        return f"{head}{wrapped}{rest.split(end, 1)[1]}"
    return f"{text.rstrip()}\n\n{section}\n\n{wrapped}\n"


def _load(directory: Path, name: str) -> dict[str, Any]:
    path = directory / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def render(out_dir: Path, dev_dir: Path) -> None:
    from benchmarks import normalize_bench, pdf_bench, toc
    from docrooter.taxonomies import builtin_taxonomy

    held = {s: _load(out_dir, STEPS[s][1]) for s in STEPS}
    dev = {s: _load(dev_dir, f"{s}.json") for s in STEPS}
    table = render_comparison(comparison_rows(dev, held))
    blocks = [f"## Dev vs held-out\n\n{table}"]
    if held["toc"]:
        blocks.append("## TOC extraction (held-out)\n\n" + toc.render_tables(held["toc"]))
    if held["pdf"]:
        taxonomy = builtin_taxonomy("doc_types")
        blocks.append(
            "## Document type (held-out)\n\n" + pdf_bench.render_section(held["pdf"], taxonomy)
        )
    if held["normalize"]:
        docs = [d["id"] for d in held["normalize"].get("documents", [])]
        blocks.append(
            "## Section normalisation (held-out)\n\n"
            + normalize_bench.render_section(held["normalize"], table_docs=docs)
        )
    text = HELDOUT_DOCS.read_text(encoding="utf-8") if HELDOUT_DOCS.exists() else ""
    HELDOUT_DOCS.write_text(
        replace_block(text, BEGIN, END, "\n\n".join(blocks), "## Results"), encoding="utf-8"
    )
    pdf_text = PDF_DOCS.read_text(encoding="utf-8")
    PDF_DOCS.write_text(
        replace_block(pdf_text, CMP_BEGIN, CMP_END, table, COMPARE_SECTION), encoding="utf-8"
    )
    print(f"wrote {HELDOUT_DOCS} and the {COMPARE_SECTION!r} block of {PDF_DOCS}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Frozen held-out generalisation evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="record the frozen commit and run every step once")
    run_p.add_argument("--steps", default=",".join(STEPS))
    run_p.add_argument("--out", type=Path, default=None, help="results dir (default: dated)")
    render_p = sub.add_parser("render", help="write the held-out docs from the results")
    render_p.add_argument("--out", type=Path, default=None, help="held-out results dir")
    render_p.add_argument("--dev", type=Path, default=None, help="dev results dir")
    args = parser.parse_args(argv)
    out_dir = args.out or RESULTS_ROOT / datetime.date.today().isoformat()
    if args.command == "run":
        steps = [s.strip() for s in args.steps.split(",") if s.strip()]
        unknown = [s for s in steps if s not in STEPS]
        if unknown:
            parser.error(f"unknown steps {unknown}; choose from {list(STEPS)}")
        out_dir.mkdir(parents=True, exist_ok=True)
        freeze = run_steps(steps, out_dir)
        failed = [s for s, v in freeze["steps"].items() if v["returncode"] != 0]
        if failed:
            raise SystemExit(f"steps failed: {failed}")
    else:
        render(out_dir, args.dev or out_dir)


if __name__ == "__main__":
    main()
