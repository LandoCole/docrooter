"""Render benchmark result files as Markdown (``docs/benchmarks.md``).

``uv run python -m benchmarks.report [RESULTS_DIR] [--out docs/benchmarks.md]``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = Path(__file__).parent / "results"
DOCS_OUT = REPO_ROOT / "docs" / "benchmarks.md"
SUITE_ORDER = ("zero-shot", "few-shot", "strategies")
DATASET_ORDER = ("b1", "b2", "b3", "b3-subset")
SUMMARY_DATASETS = ("b1", "b2", "b3-subset")
TITLES = {
    "b1": "B1: EDGAR-CORPUS 10-K items (Part, Item)",
    "b2": "B2: DBPedia Classes (L1, L2, L3)",
    "b3": "B3: LEDGAR contract clauses (100 flat classes)",
    "b3-subset": "B3 subset: LEDGAR, the fixed like-for-like sample",
}
DECISIONS_FILE = "decisions.md"


def latest_results_dir(root: Path = RESULTS_ROOT) -> Path:
    dirs = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
    if not dirs:
        raise FileNotFoundError(f"no results directories under {root}")
    return dirs[-1]


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _num(value: float | None, fmt: str = ".3f") -> str:
    return "n/a" if value is None else format(value, fmt)


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return lines


def _link(path: Path, link_base: Path) -> str:
    return Path(os.path.relpath(path, link_base)).as_posix()


def _summary(rows: list[dict[str, Any]]) -> list[str]:
    """hF1 per system on the like-for-like datasets, the mean over all of them, slowest speed."""
    systems = list(dict.fromkeys((r["system"], r["strategy"], r["shots"]) for r in rows))
    out = []
    for system, strategy, shots in systems:
        mine = {
            r["dataset"]: r
            for r in rows
            if (r["system"], r["strategy"], r["shots"]) == (system, strategy, shots)
        }
        scores = [mine[d]["report"]["hf1"] if d in mine else None for d in SUMMARY_DATASETS]
        complete = all(s is not None for s in scores)
        mean = sum(scores) / len(scores) if complete else None  # type: ignore[arg-type]
        speeds = [r["sections_per_sec"] for r in mine.values()]
        out.append((mean, system, strategy, shots, scores, min(speeds) if speeds else None))
    out.sort(key=lambda t: (t[0] is None, -(t[0] or 0.0), t[1]))
    header = ["System", "Strategy", "Shots", *(f"{d} hF1" for d in SUMMARY_DATASETS)]
    header += ["Mean hF1", "Min sections/s"]
    body = [
        [system, strategy, str(shots), *(_num(s) for s in scores), _num(mean), _num(speed, ".1f")]
        for mean, system, strategy, shots, scores, speed in out
    ]
    return _table(header, body)


def _dataset_table(rows: list[dict[str, Any]]) -> list[str]:
    header = ["System", "Strategy", "Shots", "n", "Exact", "hP", "hR", "hF1", "Macro-P", "Macro-R"]
    header += ["Macro-F1", "Stop rate", "Sections/s", "s/example", "Device"]
    ordered = sorted(rows, key=lambda r: (-r["report"]["hf1"], r["system"]))
    body = []
    for r in ordered:
        rep = r["report"]
        body.append(
            [
                r["system"],
                r["strategy"],
                str(r["shots"]),
                str(rep["n"]),
                _num(rep["exact"]),
                _num(rep["hp"]),
                _num(rep["hr"]),
                _num(rep["hf1"]),
                _num(r["macro_p"]),
                _num(r["macro_r"]),
                _num(r["macro_f1"]),
                _num(rep["stop_rate"]),
                _num(r["sections_per_sec"], ".1f"),
                _num(r["seconds_per_example"], ".4f"),
                rep["device"],
            ]
        )
    return _table(header, body)


def _env_line(rows: list[dict[str, Any]]) -> str:
    env = rows[0].get("env", {}) if rows else {}
    devices = sorted({r["report"]["device"] for r in rows})
    parts = [f"devices: {', '.join(devices)}"]
    for key in ("torch", "transformers", "sentence_transformers", "docrooter"):
        if env.get(key):
            parts.append(f"{key.replace('_', '-')} {env[key]}")
    revisions = {k: v for r in rows for k, v in r.get("env", {}).get("revisions", {}).items() if v}
    if revisions:
        parts.append(
            "revisions: " + ", ".join(f"{k} @ {v[:8]}" for k, v in sorted(revisions.items()))
        )
    return "; ".join(parts)


def render_markdown(results_dir: str | Path, link_base: str | Path | None = None) -> str:
    """Markdown for every ``<suite>.json`` in ``results_dir``, then its ``decisions.md`` if any.

    ``link_base`` is the directory the result links are relative to (default: ``docs/``).
    """
    results_dir = Path(results_dir)
    base = Path(link_base) if link_base is not None else DOCS_OUT.parent
    files = {p.stem: p for p in results_dir.glob("*.json")}
    ordered = [s for s in SUITE_ORDER if s in files] + sorted(set(files) - set(SUITE_ORDER))
    lines = [
        "# Benchmarks",
        "",
        "Generated by `python -m benchmarks.report`; do not edit by hand.",
        "Scores are hierarchical F1 (hF1), exact path accuracy and leaf macro-F1; speed is "
        "sections per second in the timed classification pass (model loading excluded).",
    ]
    for suite in ordered:
        rows = json.loads(files[suite].read_text(encoding="utf-8"))
        lines += ["", f"## {suite}", ""]
        lines.append(f"Results: [{files[suite].name}]({_link(files[suite], base)})")
        if not rows:
            lines += ["", "No rows."]
            continue
        lines += ["", _env_line(rows), "", "### Summary", ""]
        lines += _summary(rows)
        datasets = [d for d in DATASET_ORDER if any(r["dataset"] == d for r in rows)]
        datasets += sorted({r["dataset"] for r in rows} - set(DATASET_ORDER))
        for dataset in datasets:
            subset = [r for r in rows if r["dataset"] == dataset]
            lines += ["", f"### {TITLES.get(dataset, dataset)}", ""] + _dataset_table(subset)
    decisions = results_dir / DECISIONS_FILE
    if decisions.exists():
        lines += ["", decisions.read_text(encoding="utf-8").rstrip()]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m benchmarks.report", description=__doc__)
    p.add_argument("results_dir", nargs="?", type=Path, help="default: the latest under results/")
    p.add_argument("--out", type=Path, default=DOCS_OUT)
    args = p.parse_args(argv)
    results_dir = args.results_dir or latest_results_dir()
    text = render_markdown(results_dir, link_base=args.out.resolve().parent)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} from {results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
