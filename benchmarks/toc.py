"""PDF TOC extraction benchmark: predicted headings vs the PDF's own bookmarks.

``uv run --extra pdf --extra pdf-mit python -m benchmarks.toc --engines pymupdf,pdfplumber,both``

Gold is the bookmark list of every corpus PDF with enough bookmarks (levels shifted so the minimum
is 1). Extraction runs with ``use_bookmarks=False`` so the answer is never read from the file.
Metrics are reported for all depths and for gold depth <= 2, pooled (micro) and averaged over
documents (macro), plus wall-clock seconds per document.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from benchmarks import _guard, corpora
from docrooter.pdf.signals import normalize_title, read_bookmarks

Heading = tuple[str, int, int]  # (title, page, level)

CORPUS = corpora.DEV
RESULTS_ROOT = Path(__file__).parent / "results"
CACHE = Path(__file__).parent / ".cache"
MIN_BOOKMARKS = 5
HEADLINE_DEPTH = 2
LLM_ENGINE = "pymupdf4llm"
LLM_MAX_DOCS = 8
DEFAULT_ENGINES = ("pymupdf", "pdfplumber", "both", LLM_ENGINE)


# --- matching and scoring ---------------------------------------------------------------


def match_headings(
    pred: Sequence[Heading], gold: Sequence[Heading], *, ratio: float = 0.85, page_tol: int = 1
) -> list[tuple[int, int]]:
    """One-to-one greedy match of predicted to gold headings, returned as (pred, gold) indexes.

    A pair is a candidate when the pages differ by at most ``page_tol`` and the difflib ratio of
    the normalised titles is at least ``ratio``. Candidates are taken best ratio first (ties:
    closer page, then reading order), each heading used at most once.
    """
    gold_norm = [normalize_title(g[0]) for g in gold]
    by_page: dict[int, list[int]] = {}
    for j, (_, page, _) in enumerate(gold):
        by_page.setdefault(page, []).append(j)

    candidates: list[tuple[float, int, int, int]] = []
    for i, (title, page, _) in enumerate(pred):
        norm = normalize_title(title)
        if not norm:
            continue
        for p in range(page - page_tol, page + page_tol + 1):
            for j in by_page.get(p, ()):
                if not gold_norm[j]:
                    continue
                matcher = SequenceMatcher(None, norm, gold_norm[j])
                if (
                    matcher.real_quick_ratio() < ratio
                    or matcher.quick_ratio() < ratio
                    or (score := matcher.ratio()) < ratio
                ):
                    continue
                candidates.append((score, abs(p - page), i, j))
    candidates.sort(key=lambda c: (-c[0], c[1], c[2], c[3]))

    used_pred: set[int] = set()
    used_gold: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, _, i, j in candidates:
        if i in used_pred or j in used_gold:
            continue
        used_pred.add(i)
        used_gold.add(j)
        pairs.append((i, j))
    return sorted(pairs)


def _prf(n_match: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    p = n_match / n_pred if n_pred else 0.0
    r = n_match / n_gold if n_gold else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def score_doc(
    pred: Sequence[Heading],
    gold: Sequence[Heading],
    *,
    max_gold_depth: int | None = None,
    ratio: float = 0.85,
    page_tol: int = 1,
) -> dict[str, Any]:
    """Precision, recall, F1 and level accuracy (on matched pairs) for one document.

    With ``max_gold_depth`` both sides are cut to levels <= that depth before matching, so
    precision is also measured at that depth. ``level_acc`` is ``None`` when nothing matched.
    """
    if max_gold_depth is not None:
        pred = [h for h in pred if h[2] <= max_gold_depth]
        gold = [h for h in gold if h[2] <= max_gold_depth]
    pairs = match_headings(pred, gold, ratio=ratio, page_tol=page_tol)
    level_ok = sum(1 for i, j in pairs if pred[i][2] == gold[j][2])
    p, r, f1 = _prf(len(pairs), len(pred), len(gold))
    return {
        "p": p,
        "r": r,
        "f1": f1,
        "level_acc": level_ok / len(pairs) if pairs else None,
        "n_pred": len(pred),
        "n_gold": len(gold),
        "n_match": len(pairs),
        "n_level_ok": level_ok,
    }


def gold_from_bookmarks(bookmarks: Sequence[tuple[int, str, int]]) -> list[Heading]:
    """Bookmarks (level, title, page) as gold headings: no unpaged items, minimum level 1."""
    kept = [(level, title, page) for level, title, page in bookmarks if page >= 1]
    if not kept:
        return []
    shift = min(level for level, _, _ in kept) - 1
    return [(title, page, level - shift) for level, title, page in kept]


def aggregate(docs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Micro (pooled counts) and macro (mean per-document F1) scores over scored documents."""
    n_pred = sum(d["n_pred"] for d in docs)
    n_gold = sum(d["n_gold"] for d in docs)
    n_match = sum(d["n_match"] for d in docs)
    n_level_ok = sum(d["n_level_ok"] for d in docs)
    p, r, f1 = _prf(n_match, n_pred, n_gold)
    out: dict[str, Any] = {
        "n_docs": len(docs),
        "n_pred": n_pred,
        "n_gold": n_gold,
        "n_match": n_match,
        "p": p,
        "r": r,
        "f1": f1,
        "level_acc": n_level_ok / n_match if n_match else None,
        "macro_f1": sum(d["f1"] for d in docs) / len(docs) if docs else 0.0,
    }
    timed = [d["seconds"] for d in docs if "seconds" in d]
    out["seconds_per_doc"] = sum(timed) / len(timed) if timed else None
    return out


# --- corpus and engines -----------------------------------------------------------------


def corpus_docs(corpus: Path = CORPUS, min_bookmarks: int = MIN_BOOKMARKS) -> list[dict[str, Any]]:
    """Corpus documents with at least ``min_bookmarks`` bookmarks, each with its gold headings."""
    docs = []
    for row in corpora.read_manifest(corpus):
        path = corpus / row["file"]
        if not path.exists():
            continue
        gold = gold_from_bookmarks(read_bookmarks(path))
        if len(gold) >= min_bookmarks:
            docs.append(
                {"id": row["id"], "title": row["title"], "pages": row["pages"], "path": path,
                 "gold": gold}
            )  # fmt: skip
    return docs


def predicted_headings(doc: Any) -> list[Heading]:
    """Predicted (title, page_start, level) for every heading; the level-0 preamble is not one."""
    return [(s.title, s.page_start or 0, s.level) for s in doc.walk() if s.level > 0]


def run_engine(engine: str, docs: Sequence[dict[str, Any]], *, cache: Path | None = None) -> dict:
    """Load every document with ``engine`` (bookmarks off), time it and score against gold."""
    from docrooter.pdf.load import load_pdf

    if docs:  # warm-up: import the engine's libraries on the smallest document, untimed
        smallest = min(docs, key=lambda d: d["pages"])
        load_pdf(smallest["path"], engine=engine, use_bookmarks=False)

    rows = []
    for d in docs:
        error = None
        pred: list[Heading] = []
        start = time.perf_counter()
        try:
            loaded = load_pdf(d["path"], engine=engine, use_bookmarks=False)
            seconds = time.perf_counter() - start
            pred = predicted_headings(loaded)
        except Exception as exc:  # a failing document scores zero, it must not end the run
            seconds = time.perf_counter() - start
            error = f"{type(exc).__name__}: {exc}"
        row: dict[str, Any] = {
            "id": d["id"],
            "title": d["title"],
            "pages": d["pages"],
            "seconds": round(seconds, 3),
            "all": score_doc(pred, d["gold"]),
            f"depth{HEADLINE_DEPTH}": score_doc(pred, d["gold"], max_gold_depth=HEADLINE_DEPTH),
        }
        if error:
            row["error"] = error
        rows.append(row)
        if cache is not None:
            cache.mkdir(parents=True, exist_ok=True)
            (cache / f"toc-{engine}-{d['id']}.json").write_text(
                json.dumps({"pred": pred, "gold": d["gold"]}), encoding="utf-8"
            )
        print(
            f"{engine:12s} {d['id']} {d['pages']:4d}p {seconds:7.2f}s "
            f"d2 F1 {row[f'depth{HEADLINE_DEPTH}']['f1']:.3f}",
            flush=True,
        )

    def flat(key: str) -> list[dict[str, Any]]:
        return [{**r[key], "seconds": r["seconds"]} for r in rows]

    return {
        "engine": engine,
        "n_docs": len(rows),
        "doc_ids": [r["id"] for r in rows],
        "all": aggregate(flat("all")),
        f"depth{HEADLINE_DEPTH}": aggregate(flat(f"depth{HEADLINE_DEPTH}")),
        "docs": rows,
    }


def smallest(docs: Sequence[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    return sorted(docs, key=lambda d: (d["pages"], d["id"]))[:n]


def collect_env() -> dict[str, Any]:
    import importlib.metadata as md

    versions = {}
    for pkg in ("pymupdf", "pdfplumber", "pymupdf4llm", "docrooter"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "loadavg": list(os.getloadavg()),
        "versions": versions,
    }


def git_revision() -> str | None:
    """Short commit of the checkout, with ``+dirty`` when ``src/`` has uncommitted changes."""
    root = Path(__file__).parent.parent
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True,
            check=True,
        ).stdout.strip()  # fmt: skip
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "src"], cwd=root, capture_output=True,
            text=True, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{sha}+dirty" if dirty else sha


# --- Markdown report --------------------------------------------------------------------

DOCS_PATH = Path(__file__).parent.parent / "docs" / "benchmarks-pdf.md"
SECTION = "## PDF TOC extraction"
BEGIN, END = "<!-- toc-benchmark:begin -->", "<!-- toc-benchmark:end -->"
DEFAULT_ENGINE = "pymupdf"  # what engine="auto" resolves to when pymupdf is installed


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _subset_aggregate(entry: dict[str, Any], key: str, ids: Sequence[str]) -> dict[str, Any]:
    rows = [{**d[key], "seconds": d["seconds"]} for d in entry["docs"] if d["id"] in ids]
    return aggregate(rows)


def render_tables(result: dict[str, Any]) -> str:
    """Markdown for the generated part of the section: setup notes and the result tables."""
    depth = f"depth{HEADLINE_DEPTH}"
    engines: dict[str, Any] = result["engines"]
    env = result["env"]
    n_docs = next(iter(engines.values()))["n_docs"]
    match = result["match"]
    d = f"d<={HEADLINE_DEPTH}"
    intro = (
        f"Gold: bookmarks of the {n_docs} corpus PDFs with at least {result['min_bookmarks']} "
        "bookmarks (levels shifted so the minimum is 1; items without a target page dropped). "
        "Extraction runs with `use_bookmarks=False`. A prediction matches a bookmark at "
        f"normalised-title ratio >= {match['ratio']} and pages within +/-{match['page_tol']}, "
        f"one-to-one and greedy by ratio. The depth <= {HEADLINE_DEPTH} columns keep only gold and "
        f"predicted headings of level <= {HEADLINE_DEPTH}, so precision is measured at that depth "
        "too. Micro pools the counts; macro is the mean of per-document F1. Level accuracy is "
        "measured on matched pairs. Bookmark sets often omit lower-level headings, so precision "
        f"is pessimistic and recall / F1 at depth <= {HEADLINE_DEPTH} are the headline numbers."
    )
    versions = env["versions"]
    loadavg = ", ".join(f"{x:.1f}" for x in env["loadavg"])
    environment = (
        f"Environment: Python {env['python']}, {env['machine']}, {env['cpu_count']} CPUs, "
        f"pymupdf {versions.get('pymupdf')}, pdfplumber {versions.get('pdfplumber')}, "
        f"pymupdf4llm {versions.get('pymupdf4llm')}. `os.getloadavg()` was {loadavg} (1/5/15 min) "
        "when the latest run started. The machine was shared with other jobs, so seconds per "
        "document are noisy upper bounds; each engine ran one untimed warm-up document first. "
        "Seconds are wall-clock for one `load_pdf` call."
    )
    runs = [
        f"`{name}` {entry.get('date', 'n/a')} at `{entry.get('commit') or 'unknown'}`"
        + (
            f" (load {', '.join(f'{x:.1f}' for x in entry['loadavg_start'])})"
            if entry.get("loadavg_start")
            else ""
        )
        for name, entry in engines.items()
    ]
    if any("commit" in entry or "date" in entry for entry in engines.values()):
        environment += " Each engine's row keeps the run that produced it: " + "; ".join(runs) + "."
    out = [
        intro,
        "",
        environment,
        "",
        f"### Aggregate by engine (depth <= {HEADLINE_DEPTH} headline, all depths for reference)",
        "",
        f"| Engine | Docs | P ({d}) | R ({d}) | F1 micro ({d}) | F1 macro ({d}) | "
        f"Level acc ({d}) | P (all) | R (all) | F1 micro (all) | F1 macro (all) | "
        "Level acc (all) | s/doc |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    notes = []
    for name, entry in engines.items():
        shallow, full = entry[depth], entry["all"]
        label = f"{name}*" if entry.get("note") else name
        if entry.get("note"):
            notes.append(f"\\* `{name}`: {entry['note']}; see the matched-subset table below.")
        out.append(
            f"| {label} | {entry['n_docs']} | {_fmt(shallow['p'])} | {_fmt(shallow['r'])} | "
            f"{_fmt(shallow['f1'])} | {_fmt(shallow['macro_f1'])} | {_fmt(shallow['level_acc'])} | "
            f"{_fmt(full['p'])} | {_fmt(full['r'])} | {_fmt(full['f1'])} | "
            f"{_fmt(full['macro_f1'])} | {_fmt(full['level_acc'])} | "
            f"{_fmt(full['seconds_per_doc'], 2)} |"
        )
    out += [""] + notes

    limited = [(n, e) for n, e in engines.items() if e.get("note")]
    for name, entry in limited:
        ids = entry["doc_ids"]
        out += [
            "",
            f"### Same {len(ids)} documents for every engine (the `{name}` subset)",
            "",
            f"| Engine | P ({d}) | R ({d}) | F1 micro ({d}) | F1 macro ({d}) | "
            f"Level acc ({d}) | F1 micro (all) | s/doc |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for other, other_entry in engines.items():
            shallow = _subset_aggregate(other_entry, depth, ids)
            full = _subset_aggregate(other_entry, "all", ids)
            out.append(
                f"| {other} | {_fmt(shallow['p'])} | {_fmt(shallow['r'])} | "
                f"{_fmt(shallow['f1'])} | {_fmt(shallow['macro_f1'])} | "
                f"{_fmt(shallow['level_acc'])} | {_fmt(full['f1'])} | "
                f"{_fmt(shallow['seconds_per_doc'], 2)} |"
            )

    default = engines.get(DEFAULT_ENGINE)
    if default:
        out += [
            "",
            f"### Per document: `{DEFAULT_ENGINE}` (the default engine)",
            "",
            f"| Doc | Title | Pages | Gold {d} | Pred {d} | P | R | F1 | Level acc | "
            "F1 (all) | s |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in default["docs"]:
            s2, sa = row[depth], row["all"]
            title = row["title"] if len(row["title"]) <= 48 else row["title"][:45] + "..."
            out.append(
                f"| {row['id']} | {title.replace('|', '/')} | {row['pages']} | {s2['n_gold']} | "
                f"{s2['n_pred']} | {_fmt(s2['p'], 2)} | {_fmt(s2['r'], 2)} | {_fmt(s2['f1'], 2)} | "
                f"{_fmt(s2['level_acc'], 2)} | {_fmt(sa['f1'], 2)} | {row['seconds']:.2f} |"
            )
    return "\n".join(out)


def update_docs(path: Path, tables: str) -> None:
    """Write ``tables`` into the ``## PDF TOC extraction`` section of ``path``.

    Creates the file (with a ``# PDF benchmarks`` header) when missing. Only the marked block of
    this section is replaced, so hand-written analysis below it and other sections survive.
    """
    block = f"{BEGIN}\n{tables}\n{END}"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# PDF benchmarks\n\n{SECTION}\n\n{block}\n", encoding="utf-8")
        return
    text = path.read_text(encoding="utf-8")
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        tail = rest.split(END, 1)[1]
        text = f"{head}{block}{tail}"
    elif SECTION in text:
        head, rest = text.split(SECTION, 1)
        following = rest.find("\n## ")
        tail = rest[following:] if following != -1 else "\n"
        text = f"{head}{SECTION}\n\n{block}\n{tail}"
    else:
        text = f"{text.rstrip()}\n\n{SECTION}\n\n{block}\n"
    path.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PDF TOC extraction benchmark")
    parser.add_argument("--engines", default=",".join(DEFAULT_ENGINES))
    parser.add_argument("--out", type=Path, default=None, help="results dir (default: dated)")
    parser.add_argument(
        "--corpus",
        type=corpora.corpus_path,
        default=CORPUS,
        help="dev, heldout or a corpus directory (results of a non-dev corpus go to "
        "<name>-toc.json)",
    )
    parser.add_argument("--min-bookmarks", type=int, default=MIN_BOOKMARKS)
    parser.add_argument("--llm-max-docs", type=int, default=LLM_MAX_DOCS)
    parser.add_argument("--docs", type=Path, default=None, help="also update this Markdown file")
    parser.add_argument("--report-only", action="store_true", help="render from existing toc.json")
    args = parser.parse_args(argv)
    _guard.apply_env_caps()  # no models here, so no single-process check

    out_dir = args.out or RESULTS_ROOT / datetime.date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / corpora.results_name(args.corpus, "toc.json")
    if args.report_only:
        saved = json.loads(path.read_text(encoding="utf-8"))
        update_docs(args.docs or DOCS_PATH, render_tables(saved))
        return
    docs = corpus_docs(args.corpus, args.min_bookmarks)
    print(f"{len(docs)} corpus documents with >= {args.min_bookmarks} bookmarks", flush=True)

    if path.exists():  # keep engines from earlier invocations; --engines replaces only its own
        result = json.loads(path.read_text(encoding="utf-8"))
    else:
        result = {"engines": {}}
    result["env"] = collect_env()  # this invocation's; each engine keeps its own run metadata
    result["corpus"] = corpora.relative(args.corpus)
    result |= {"min_bookmarks": args.min_bookmarks, "match": {"ratio": 0.85, "page_tol": 1}}
    for engine in [e.strip() for e in args.engines.split(",") if e.strip()]:
        subset = docs
        note = None
        if engine == LLM_ENGINE and len(docs) > args.llm_max_docs:
            subset = smallest(docs, args.llm_max_docs)
            note = f"limited to the {len(subset)} smallest bookmarked documents (by page count)"
        loadavg_start = list(os.getloadavg())
        entry = run_engine(engine, subset, cache=CACHE)
        entry |= {
            "note": note,
            "date": datetime.date.today().isoformat(),
            "commit": git_revision(),
            "loadavg_start": loadavg_start,
            "loadavg_end": list(os.getloadavg()),
        }
        result["engines"][engine] = entry
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.docs:
        update_docs(args.docs, render_tables(result))
    print(f"wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
