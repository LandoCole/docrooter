"""OCR engines on image-only renders of dev pages, and OCR routing (v0.4 task E11b; spec §10.4).

``PYTHONPATH=src .venv/bin/python -m benchmarks.ocr_bench``

1. **Pages** (cached under ``benchmarks/.cache/ocr/``): 20 dev PDFs drawn with a fixed seed,
   3 pages each, from the pages whose text layer is clean (routed ``text`` with no OCR region,
   at least :data:`MIN_CHARS` characters, no invisible text, images under
   :data:`MAX_IMAGE_COV` of the page). Each page is copied into a one-page PDF (its text
   layer, the ground truth) and rendered at 200 dpi into an image-only one-page PDF of the same
   size.
2. **Routing.** ``route_page`` on the 60 image-only pages (expected: ``ocr``) and, from
   ``routing-dev.json`` (E5c, ``benchmarks.elements_perf --routing``), on every page of the
   51 dev PDFs (expected: ``text``, born-digital). A region decision keeps the page's text
   layer, so it counts as ``text``.
3. **Engines** (ocrmac, rapidocr, tesseract; whichever are installed), each in a fresh worker
   process, one after the other, so its peak RSS is its own:
   - CER and WER of the engine's lines against the text layer, after :func:`normalize_text`.
     Both sides are put in the same order first (:func:`page_text`: rows top to bottom, left
     to right within a row), because an engine's line order is its own; and the order-free
     word recall and precision (:func:`word_overlap`), which no ordering difference affects;
   - seconds per page: rendering, recognition, and the whole element-mode load of the
     image-only page (``load_pdf(elements=True, ocr=engine)``);
   - peak RSS of the worker (and of its children: the tesseract binary);
   - element-kind agreement between the text-layer elements and the OCR elements of the same
     page (:func:`kind_agreement`).

Results go to ``benchmarks/results/<date>/ocr.json``. OCR engines load models (rapidocr's ONNX
sessions), so the run is under the one-model-process guard.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata as md
import json
import random
import re
import resource
import shutil
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks import _guard, corpora
from benchmarks.elements_perf import ROUTING_FILE
from benchmarks.toc import CACHE, RESULTS_ROOT, collect_env, git_revision

OCR_CACHE = CACHE / "ocr"
SEED = "20261018:ocr"
N_DOCS = 20
PAGES_PER_DOC = 3
DPI = 200
MIN_CHARS = 200  # a ground-truth page has at least this many text-layer characters
MAX_IMAGE_COV = 0.1  # ... and images under this share of the page (their text is not in the layer)
ENGINES = ("ocrmac", "rapidocr", "tesseract")
DECIDED = ("text", "text+ocr", "ocr")
EXPECTED = ("text", "ocr")
RESULT_FILE = "ocr.json"
ROUTING_SOURCE = RESULTS_ROOT / "2026-10-18" / ROUTING_FILE

Item = tuple[str, float, float, float, float]  # text, x0, y0, x1, y1
Region = tuple[str, float, float, float, float, int]  # kind, x0, y0, x1, y1, characters


# --- scoring -------------------------------------------------------------------------------

_HYPHEN_BREAK = re.compile(r"(?<=\w)-[ \t]*\n\s*(?=[a-z])")


def normalize_text(text: str) -> str:
    """NFKC, words hyphenated at a line end joined (only before a lowercase letter), and every
    whitespace run collapsed to one space, ends stripped."""
    text = unicodedata.normalize("NFKC", text)
    text = _HYPHEN_BREAK.sub("", text)
    return " ".join(text.split())


def edit_distance(a: Sequence[Any], b: Sequence[Any]) -> int:
    """Levenshtein distance between two sequences (characters or words), one numpy row at a
    time: insertions along a row are a running minimum of ``row - index``."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    vocab: dict[Any, int] = {}
    av = np.fromiter((vocab.setdefault(x, len(vocab)) for x in a), dtype=np.int64, count=len(a))
    bv = np.fromiter((vocab.setdefault(x, len(vocab)) for x in b), dtype=np.int64, count=len(b))
    index = np.arange(len(b) + 1, dtype=np.int64)
    prev = index.copy()
    for k, token in enumerate(av, start=1):
        cur = np.empty_like(prev)
        cur[0] = k
        np.minimum(prev[:-1] + (bv != token), prev[1:] + 1, out=cur[1:])
        prev = index + np.minimum.accumulate(cur - index)
    return int(prev[-1])


def _rate(errors: int, n_ref: int) -> float:
    return errors / max(n_ref, 1)


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate after :func:`normalize_text`: edits over reference characters
    (an empty reference counts as one character, so insertions still score)."""
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    return _rate(edit_distance(ref, hyp), len(ref))


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate after :func:`normalize_text`: word edits over reference words."""
    ref, hyp = normalize_text(reference).split(), normalize_text(hypothesis).split()
    return _rate(edit_distance(ref, hyp), len(ref))


def word_overlap(reference: str, hypothesis: str) -> tuple[int, int, int]:
    """Order-free word counts after :func:`normalize_text`: ``(shared, reference words,
    hypothesis words)``, ``shared`` being the multiset intersection. Word recall and precision
    from these do not depend on the order the lines were read in (CER and WER do)."""
    ref, hyp = (
        Counter(normalize_text(reference).split()),
        Counter(normalize_text(hypothesis).split()),
    )
    return sum((ref & hyp).values()), sum(ref.values()), sum(hyp.values())


def page_text(items: Iterable[Item]) -> str:
    """A page's text in a fixed order, the same for the text layer and an engine: items sorted
    by vertical centre and grouped into rows (a centre within half the row's smallest height
    of the row's first item), rows top to bottom, items left to right in a row, joined by
    spaces; rows joined by newlines."""
    ordered = sorted(items, key=lambda it: ((it[2] + it[4]) / 2, it[1]))
    rows: list[list[Item]] = []
    anchor = 0.0
    tolerance = 0.0
    for item in ordered:
        centre, height = (item[2] + item[4]) / 2, max(item[4] - item[2], 0.0)
        if rows and abs(centre - anchor) <= tolerance:
            rows[-1].append(item)
            tolerance = min(tolerance, height / 2)
        else:
            rows.append([item])
            anchor, tolerance = centre, height / 2
    return "\n".join(" ".join(it[0] for it in sorted(row, key=lambda it: it[1])) for row in rows)


def routing_confusion(pairs: Iterable[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """``{expected: {decided: pages}}`` for expected ``text``/``ocr`` and decided ``text``,
    ``text+ocr`` or ``ocr`` (every cell present, in that order). A region decision keeps the
    text layer: callers count it as ``text`` before calling."""
    table = {e: dict.fromkeys(DECIDED, 0) for e in EXPECTED}
    for expected, decided in pairs:
        if expected not in table or decided not in DECIDED:
            raise ValueError(f"unknown routing pair {(expected, decided)!r}")
        table[expected][decided] += 1
    return table


def _overlap(a: Region, b: Region) -> float:
    w = min(a[3], b[3]) - max(a[1], b[1])
    h = min(a[4], b[4]) - max(a[2], b[2])
    return w * h if w > 0 and h > 0 else 0.0


def kind_agreement(text_layer: Sequence[Region], ocr: Sequence[Region]) -> dict[str, Any]:
    """Each text-layer element against the OCR element of the same page that overlaps its box
    most: the share that has the same kind, by element and by character, and the confusion
    ``{text-layer kind: {OCR kind or "-" for none: count}}``."""
    agree = agree_chars = chars = missed = 0
    confusion: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for region in text_layer:
        best = max(ocr, key=lambda o: _overlap(region, o), default=None)
        kind = best[0] if best is not None and _overlap(region, best) > 0 else "-"
        confusion[region[0]][kind] += 1
        chars += region[5]
        if kind == region[0]:
            agree += 1
            agree_chars += region[5]
        elif kind == "-":
            missed += 1
    return {
        "elements": len(text_layer), "agree": agree, "missed": missed,
        "chars": chars, "agree_chars": agree_chars,
        "share": agree / len(text_layer) if text_layer else None,
        "char_share": agree_chars / chars if chars else None,
        "confusion": {k: dict(sorted(v.items())) for k, v in sorted(confusion.items())},
    }  # fmt: skip


def _merge_agreement(parts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total: Counter[str] = Counter()
    confusion: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for part in parts:
        total.update({k: part[k] for k in ("elements", "agree", "missed", "chars", "agree_chars")})
        for kind, row in part["confusion"].items():
            confusion[kind].update(row)
    return {
        **dict(total),
        "share": total["agree"] / total["elements"] if total["elements"] else None,
        "char_share": total["agree_chars"] / total["chars"] if total["chars"] else None,
        "confusion": {k: dict(sorted(v.items())) for k, v in sorted(confusion.items())},
    }


# --- pages ---------------------------------------------------------------------------------


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean_pages(path: Path) -> list[int]:
    """Pages of ``path`` whose text layer can serve as OCR ground truth."""
    from docrooter.pdf.objects import extract_page_objects
    from docrooter.pdf.routing import page_metrics, route_page

    pages = []
    for objs in extract_page_objects(path, engine="pymupdf", tables="none"):
        task, regions, _ = route_page(objs)
        if task != "text" or regions or objs.char_count < MIN_CHARS or objs.invisible_chars:
            continue
        if page_metrics(objs).image_cov <= MAX_IMAGE_COV:
            pages.append(objs.page)
    return pages


def select_pages() -> list[dict[str, Any]]:
    """The benchmark pages: dev documents in a seeded shuffle, the first :data:`N_DOCS` with at
    least :data:`PAGES_PER_DOC` clean pages, and that many of their clean pages (seeded)."""
    rng = random.Random(SEED)
    docs = corpora.read_manifest(corpora.DEV)
    rng.shuffle(docs)
    chosen: list[dict[str, Any]] = []
    for record in docs:
        if len(chosen) == N_DOCS * PAGES_PER_DOC:
            break
        clean = _clean_pages(corpora.DEV / record["file"])
        if len(clean) < PAGES_PER_DOC:
            print(f"skip {record['id']}: {len(clean)} clean pages", flush=True)
            continue
        for page in sorted(rng.sample(clean, PAGES_PER_DOC)):
            chosen.append({"doc": record["id"], "file": record["file"], "page": page})
        print(f"{record['id']}: {len(clean)} clean pages", flush=True)
    return chosen


def prepare(cache: Path = OCR_CACHE) -> list[dict[str, Any]]:
    """Write (or reuse) the one-page text-layer and image-only PDFs and the ground truth;
    returns the page records (``pages.json``)."""
    import pymupdf

    from docrooter.pdf.objects import extract_page_objects

    manifest = cache / "pages.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved.get("seed") == SEED and saved.get("dpi") == DPI and all(
            _sha(cache / p["text_pdf"]) == p["text_sha256"]
            and _sha(cache / p["image_pdf"]) == p["image_sha256"]
            for p in saved["pages"]
        ):  # fmt: skip
            return saved["pages"]
    (cache / "text").mkdir(parents=True, exist_ok=True)
    (cache / "image").mkdir(parents=True, exist_ok=True)
    records = []
    for pick in select_pages():
        page_id = f"{pick['doc']}_p{pick['page']:03d}"
        text_pdf = cache / "text" / f"{page_id}.pdf"
        image_pdf = cache / "image" / f"{page_id}.pdf"
        with pymupdf.open(corpora.DEV / pick["file"]) as src, pymupdf.open() as one:
            one.insert_pdf(  # no annotations or form fields: they are not in the text layer
                src, from_page=pick["page"] - 1, to_page=pick["page"] - 1, links=False,
                annots=False, widgets=False,
            )  # fmt: skip
            one.save(text_pdf, garbage=3, deflate=True)
        with pymupdf.open(text_pdf) as one, pymupdf.open() as out:
            page = one[0]
            pix = page.get_pixmap(dpi=DPI, alpha=False)
            new = out.new_page(width=page.rect.width, height=page.rect.height)
            new.insert_image(new.rect, stream=pix.tobytes("png"))
            out.save(image_pdf, garbage=3, deflate=True)
        objs = extract_page_objects(text_pdf, engine="pymupdf", tables="none")[0]
        truth = [(ln.text, ln.x0, ln.y0, ln.x1, ln.y1) for ln in objs.lines]
        records.append({
            "id": page_id, "doc": pick["doc"], "page": pick["page"],
            "width": objs.width, "height": objs.height,
            "text_pdf": str(text_pdf.relative_to(cache)),
            "image_pdf": str(image_pdf.relative_to(cache)),
            "text_sha256": _sha(text_pdf), "image_sha256": _sha(image_pdf),
            "truth": truth,
        })  # fmt: skip
    manifest.write_text(json.dumps({"seed": SEED, "dpi": DPI, "pages": records}), encoding="utf-8")
    return records


def selection_digest(pages: Sequence[dict[str, Any]]) -> str:
    """A digest of the page ids and both PDFs of every page."""
    h = hashlib.sha256()
    for p in pages:
        h.update(f"{p['id']}:{p['text_sha256']}:{p['image_sha256']}\n".encode())
    return h.hexdigest()


# --- elements and routing ------------------------------------------------------------------


def element_regions(doc: Any) -> list[Region]:
    """The elements of a one-page document as ``(kind, box, characters)``, one box each (the
    union of the element's page-1 boxes)."""
    out: list[Region] = []
    for element in doc.elements:
        boxes = [b for b in element.boxes if b.page == 1]
        if not boxes:
            continue
        chars = sum(1 for ch in element.text if not ch.isspace())
        out.append((element.kind, min(b.x0 for b in boxes), min(b.y0 for b in boxes),
                    max(b.x1 for b in boxes), max(b.y1 for b in boxes), chars))  # fmt: skip
    return out


def image_routing(pages: Sequence[dict[str, Any]], cache: Path) -> dict[str, Any]:
    """``route_page`` on every image-only page: decisions and their reasons."""
    from docrooter.pdf.objects import extract_page_objects
    from docrooter.pdf.routing import route_page

    rows = []
    for p in pages:
        objs = extract_page_objects(cache / p["image_pdf"], engine="pymupdf", tables="none")[0]
        task, regions, decision = route_page(objs)
        rows.append({"id": p["id"], "decided": task,
                     "regions": len(regions), "reason": decision.reason})  # fmt: skip
    return {
        "confusion": routing_confusion(("ocr", r["decided"]) for r in rows),
        "pages": rows,
    }


def dev_routing(source: Path = ROUTING_SOURCE) -> dict[str, Any]:
    """The dev-corpus routing of ``routing-dev.json`` as a confusion (every page expected
    ``text``), with its commit and the pages that were not routed ``text``."""
    saved = json.loads(source.read_text(encoding="utf-8"))
    counts = saved["decisions"]
    pairs: list[tuple[str, str]] = []
    for decided, n in counts.items():
        pairs += [("text", "text" if decided == "region" else decided)] * n
    return {
        "source": str(source.relative_to(RESULTS_ROOT.parent.parent)),
        "commit": saved.get("commit"),
        "documents": saved["documents"],
        "pages": saved["pages"],
        "region_pages": counts.get("region", 0),
        "confusion": routing_confusion(pairs),
        "not_text": [
            {k: e[k] for k in ("id", "page", "chars", "reason")} | {"decided": decided}
            for decided, key in (("ocr", "replaced"), ("text+ocr", "merged"))
            for e in saved[key]
        ],
    }


# --- engine worker -------------------------------------------------------------------------


def _peak_rss_mb(who: int) -> float:
    """Peak resident set of this process (or its finished children) in MiB."""
    peak = resource.getrusage(who).ru_maxrss
    return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)


def engine_version(name: str) -> str | None:
    if name == "tesseract":
        binary = shutil.which("tesseract")
        if binary is None:
            return None
        out = subprocess.run([binary, "--version"], capture_output=True, text=True, check=False)
        return (out.stdout or out.stderr).splitlines()[0] if (out.stdout or out.stderr) else None
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def run_engine(name: str, pages: Sequence[dict[str, Any]], cache: Path) -> dict[str, Any]:
    """One engine over every page (in this process): text scores, timings, OCR elements."""
    from docrooter.elements.models import Box
    from docrooter.ocr import get_ocr_engine, render_page
    from docrooter.pdf.load import load_pdf

    first = cache / pages[0]["image_pdf"]
    render_page(first, 1, dpi=DPI)  # imports and the renderer, outside every timing
    load_pdf(cache / pages[0]["text_pdf"], elements=True)
    rss_before = _peak_rss_mb(resource.RUSAGE_SELF)
    started = time.perf_counter()
    engine = get_ocr_engine(name)
    png, w, h = render_page(first, 1, dpi=DPI)
    page_box = Box(page=1, x0=0.0, y0=0.0, x1=pages[0]["width"], y1=pages[0]["height"])
    engine.recognize(png, width_px=w, height_px=h, page=page_box)  # warm-up: lazy model loads
    load_s = time.perf_counter() - started
    rows = []
    try:
        for p in pages:
            path = cache / p["image_pdf"]
            clock = time.perf_counter()
            png, w, h = render_page(path, 1, dpi=DPI)
            render_s = time.perf_counter() - clock
            page_box = Box(page=1, x0=0.0, y0=0.0, x1=p["width"], y1=p["height"])
            clock = time.perf_counter()
            found = engine.recognize(png, width_px=w, height_px=h, page=page_box)
            ocr_s = time.perf_counter() - clock
            truth = page_text(tuple(t) for t in p["truth"])
            hyp = page_text((ln.text, ln.box.x0, ln.box.y0, ln.box.x1, ln.box.y1) for ln in found)
            ref_n, hyp_n = normalize_text(truth), normalize_text(hyp)
            char_edits = edit_distance(ref_n, hyp_n)
            word_edits = edit_distance(ref_n.split(), hyp_n.split())
            shared, _, hyp_words = word_overlap(ref_n, hyp_n)
            clock = time.perf_counter()
            doc = load_pdf(path, elements=True, ocr=engine, ocr_dpi=DPI)
            elements_s = time.perf_counter() - clock
            routes = [r["task"] for r in doc.meta.get("routes", []) if r["scope"] == "page"]
            rows.append({
                "id": p["id"], "ref_chars": len(ref_n), "char_edits": char_edits,
                "ref_words": len(ref_n.split()), "word_edits": word_edits,
                "shared_words": shared, "hyp_words": hyp_words,
                "cer": round(_rate(char_edits, len(ref_n)), 4),
                "wer": round(_rate(word_edits, len(ref_n.split())), 4),
                "lines": len(found), "render_s": round(render_s, 4), "ocr_s": round(ocr_s, 4),
                "elements_s": round(elements_s, 4), "route": routes[0] if routes else None,
                "warnings": doc.meta.get("warnings", []),
                "elements": element_regions(doc),
            })  # fmt: skip
            print(f"{name} {p['id']} CER {rows[-1]['cer']:.3f} WER {rows[-1]['wer']:.3f} "
                  f"{ocr_s:.2f}s", flush=True)  # fmt: skip
    finally:
        engine.close()
    return {
        "engine": name,
        "version": engine_version(name),
        "load_s": round(load_s, 2),
        "rss_before_engine_mb": rss_before,
        "peak_rss_mb": _peak_rss_mb(resource.RUSAGE_SELF),
        "children_peak_rss_mb": _peak_rss_mb(resource.RUSAGE_CHILDREN),
        "pages": rows,
    }


def summarise_engine(raw: dict[str, Any], text_regions: dict[str, list[Region]]) -> dict[str, Any]:
    """Pooled and per-page-mean CER and WER, seconds per page and element agreement."""
    rows = raw["pages"]
    n = len(rows)

    def mean(key: str) -> float:
        return sum(r[key] for r in rows) / n

    agreement = [
        kind_agreement(text_regions[r["id"]], [tuple(e) for e in r["elements"]]) for r in rows
    ]
    return {
        **{
            k: raw[k]
            for k in (
                "engine",
                "version",
                "load_s",
                "rss_before_engine_mb",
                "peak_rss_mb",
                "children_peak_rss_mb",
            )
        },  # fmt: skip
        "pages": n,
        "cer": sum(r["char_edits"] for r in rows) / max(sum(r["ref_chars"] for r in rows), 1),
        "wer": sum(r["word_edits"] for r in rows) / max(sum(r["ref_words"] for r in rows), 1),
        "word_recall": sum(r["shared_words"] for r in rows)
        / max(sum(r["ref_words"] for r in rows), 1),
        "word_precision": sum(r["shared_words"] for r in rows)
        / max(sum(r["hyp_words"] for r in rows), 1),
        "cer_page_mean": mean("cer"),
        "wer_page_mean": mean("wer"),
        "cer_page_median": float(np.median([r["cer"] for r in rows])),
        "ocr_s_per_page": mean("ocr_s"),
        "render_s_per_page": mean("render_s"),
        "elements_s_per_page": mean("elements_s"),
        "routed_ocr": sum(r["route"] == "ocr" for r in rows),
        "kind_agreement": _merge_agreement(agreement),
        "per_page": [{k: v for k, v in r.items() if k != "elements"} for r in rows],
    }


def _worker_main(name: str, out: Path) -> None:
    _guard.apply_env_caps()  # the parent holds the one-model-process check
    pages = json.loads((OCR_CACHE / "pages.json").read_text(encoding="utf-8"))["pages"]
    out.write_text(json.dumps(run_engine(name, pages, OCR_CACHE)), encoding="utf-8")


def installed(name: str) -> bool:
    import importlib.util

    if name == "tesseract":
        return shutil.which("tesseract") is not None
    if name == "rapidocr":
        return all(importlib.util.find_spec(m) is not None for m in ("rapidocr", "onnxruntime"))
    return sys.platform == "darwin" and importlib.util.find_spec(name) is not None


# --- run -----------------------------------------------------------------------------------


def evaluate(engines: Sequence[str], scratch: Path) -> dict[str, Any]:
    from docrooter.pdf.load import load_pdf

    pages = prepare()
    print(f"{len(pages)} pages from {len({p['doc'] for p in pages})} documents", flush=True)
    text_regions = {
        p["id"]: element_regions(load_pdf(OCR_CACHE / p["text_pdf"], elements=True)) for p in pages
    }
    image = image_routing(pages, OCR_CACHE)
    dev = dev_routing()
    combined = {e: dict.fromkeys(DECIDED, 0) for e in EXPECTED}
    for table in (dev["confusion"], image["confusion"]):
        for e, row in table.items():
            for d, n in row.items():
                combined[e][d] += n
    result: dict[str, Any] = {
        "benchmark": "ocr",
        "date": datetime.date.today().isoformat(),
        "commit": git_revision(),
        "env": collect_env(),
        "dpi": DPI,
        "selection": {
            "seed": SEED,
            "documents": sorted({p["doc"] for p in pages}),
            "pages": len(pages),
            "page_ids": [p["id"] for p in pages],
            "digest": selection_digest(pages),
            "rules": f"routed text with no OCR region, >= {MIN_CHARS} text-layer characters, "
            f"no invisible text, image coverage <= {MAX_IMAGE_COV}",
        },  # fmt: skip
        "ordering": "both sides: rows top to bottom (centre within half the smallest height), "
        "left to right within a row (page_text)",
        "routing": {"dev": dev, "image_pages": image, "combined": combined},
        "text_layer_elements": sum(len(v) for v in text_regions.values()),
        "engines": {},
    }
    for name in engines:
        if not installed(name):
            result["engines"][name] = {"engine": name, "installed": False}
            continue
        out = scratch / f"ocr-{name}.json"
        cmd = [sys.executable, "-m", "benchmarks.ocr_bench", "--worker", name, "--worker-out",
               str(out)]  # fmt: skip
        started = time.perf_counter()
        subprocess.run(cmd, check=True)
        raw = json.loads(out.read_text(encoding="utf-8"))
        result["engines"][name] = {
            "installed": True,
            "worker_s": round(time.perf_counter() - started, 1),
            **summarise_engine(raw, text_regions),
        }
        s = result["engines"][name]
        print(f"{name}: CER {s['cer']:.4f} WER {s['wer']:.4f} {s['ocr_s_per_page']:.2f} s/page "
              f"peak RSS {s['peak_rss_mb']} MiB", flush=True)  # fmt: skip
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="OCR engines and routing (spec §10.4)")
    parser.add_argument("--engines", default=",".join(ENGINES))
    parser.add_argument("--out", type=Path, default=None, help="results dir (default: dated)")
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", type=Path, default=None, help=argparse.SUPPRESS)
    _guard.add_guard_argument(parser)
    args = parser.parse_args(argv)
    if args.worker:
        _worker_main(args.worker, args.worker_out)
        return
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    unknown = [e for e in engines if e not in ENGINES]
    if unknown:
        raise SystemExit(f"unknown engine: {', '.join(unknown)}")
    _guard.single_model_process("ocr_bench", allow_concurrent=args.allow_concurrent)
    out_dir = args.out or RESULTS_ROOT / datetime.date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = OCR_CACHE / "runs"
    scratch.mkdir(parents=True, exist_ok=True)
    result = evaluate(engines, scratch)
    path = out_dir / RESULT_FILE
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
