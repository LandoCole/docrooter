"""Build the demo's samples and benchmark summary.

Each sample is classified live with Jev (``TYPESAFE_API_KEY``), read then tag (``doc_first``)
at all four rungs,
through the same code as the live upload (``demo/api/docrooter_demo.py``: :func:`describe` and
:func:`run`), so a sample shows exactly what the model saw and answered. Two kinds of sample:

- four synthetic financial documents from github.com/LandoCole/document-classifier (MIT;
  invented companies), copied into ``demo/samples/`` with gold tags in ``samples/gold.json``;
- four public documents from the benchmark whose licence allows redistribution (a GAO report,
  a CC BY 4.0 arXiv paper and CUAD contract, a U.S. federal court opinion), read from the
  benchmark's reading cache, with the benchmark's doc-type gold and section gold (v2).

The benchmark summary comes from the committed results (Jev, plus Claude Haiku 5.5 and
UltraBERT for comparison). About 70 Jev runs, a few cents.

Run from the repository root:  PYTHONPATH=src python demo/build_samples.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.classification.section_gold import read_jsonl  # noqa: E402
from benchmarks.classification.section_sample import title_sha1  # noqa: E402
from benchmarks.classification_bench import (  # noqa: E402
    PDF_DIR,
    READINGS,
    SECTION_GOLD_PATH,
    _loader,
    load_doc_type_gold,
)
from docrooter import load  # noqa: E402
from docrooter.taxonomies import builtin_taxonomy  # noqa: E402

_api = ROOT / "demo" / "api" / "docrooter_demo.py"
_spec = importlib.util.spec_from_file_location("docrooter_demo", _api)
assert _spec is not None and _spec.loader is not None
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)

RESULTS = ROOT / "benchmarks" / "results"
JEV = RESULTS / "2026-10-22" / "classification"
HAIKU = RESULTS / "2026-10-21" / "classification"
MANIFEST = ROOT / "benchmarks" / "training_corpus" / "manifest.jsonl"
SAMPLES = ROOT / "demo" / "samples"
OUT = ROOT / "demo" / "data"

SYNTHETIC = ["annual_report_10k-1", "earnings_release-1", "fund_prospectus-1", "proxy_statement-1"]
PUBLIC = [
    "gao-gao-07-707",
    "arxiv-2404_01334",
    "cuad-vapotherm_inc_-_manufacturing_and_supply_agreement",
    "uscourts-ca9-25-5220-0",
]
SYNTHETIC_NOTE = (
    "Synthetic: an invented company, written for the document classifier "
    "(github.com/LandoCole/document-classifier, MIT). Section gold labelled by Claude for the demo."
)
ATTRIBUTION = {
    "gao": "U.S. Government Accountability Office; GAO products are not protected by copyright.",
    "cuad": "From CUAD (Contract Understanding Atticus Dataset), The Atticus Project, CC BY 4.0.",
    "arxiv": "arXiv preprint, CC BY 4.0; the authors hold the copyright.",
    "uscourts": "U.S. federal court opinion, a work of the U.S. government.",
}


def manifest() -> dict[str, dict]:
    rows = (json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip())
    return {r["id"]: r for r in rows}


def section_gold(doc, doc_id: str, rows: dict[str, list[dict]]) -> dict[str, str]:
    """The benchmark's section gold for this reading of ``doc_id``: section id -> tag."""
    return {
        r["section_id"]: r["label"]
        for r in rows.get(doc_id, [])
        if r["section_id"] in doc.sections
        and title_sha1(doc.sections[r["section_id"]].title) == r["title_sha1"]
    }


def build(doc, meta: dict, gold: str, gold_tags: dict[str, str], backend) -> dict:
    sample = {**demo.describe(doc, gold_tags), **meta, "gold": gold, "runs": {}}
    for flow in ("doc_first",):  # read, then tag: the only approach the demo shows
        for level in demo.LEVELS:
            sample["runs"][f"{flow}/{level}"] = demo.run(doc, flow, level, backend, gold, gold_tags)
    return sample


def bench_summary() -> dict:
    """The benchmark rows the demo's tables and chart show."""
    rows = []
    for directory, wanted in ((JEV, {"jev"}), (HAIKU, {"haiku-5.5", "ultrabert"})):
        for path in sorted(directory.glob("*.json")):
            data = json.loads(path.read_text())
            c, m = data["config"], data["metrics"]
            if c["section_level"] != c["doc_level"] or data["backend"]["label"] not in wanted:
                continue
            if c["flow"] != "doc_first":  # read, then tag only
                continue
            if (data.get("settings") or {}).get("limit"):  # a smoke run
                continue
            sec = m.get("sections") or {}
            rows.append(
                {
                    "backend": data["backend"]["label"],
                    "docs": data["docs"],
                    "flow": c["flow"],
                    "level": c["doc_level"],
                    "calls": c.get("calls", "per_item"),
                    "doc_acc": m["doc_type"]["accuracy"],
                    "leaf": sec.get("leaf") if directory == JEV else None,
                    "prompt_tokens": m["prompt_tokens_mean"],
                    "exact": m["tokens_exact"],
                    "documents": m["documents"],
                    "decided_by_rules": m.get("decided_by_rules"),
                }
            )
    return {"rows": rows, "price": demo.PRICE, "model": demo.MODEL}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    backend = demo.jev_backend()
    meta, doc_gold = manifest(), load_doc_type_gold()
    gold_rows: dict[str, list[dict]] = {}
    for row in read_jsonl(SECTION_GOLD_PATH):
        gold_rows.setdefault(row["doc_id"], []).append(row)
    synthetic = json.loads((SAMPLES / "gold.json").read_text())["documents"]
    index, used = [], set()
    for name in SYNTHETIC:
        info = synthetic[name]
        doc = load(SAMPLES / f"{name}.md")
        by_title = info["tags"]
        gold_tags = {s.id: by_title[s.title] for s in doc.walk() if s.title in by_title}
        extra = {"title": info["title"], "publisher": "Synthetic", "url": None,
                 "attribution": SYNTHETIC_NOTE, "synthetic": True}  # fmt: skip
        sample = build(doc, extra, info["doc_type"], gold_tags, backend)
        index.append((name, sample))
    for doc_id in PUBLIC:
        doc = _loader(PDF_DIR / f"{doc_id}.pdf", readings=READINGS)()
        m = meta[doc_id]
        extra = {
            "title": m.get("title") or doc.title or doc_id,
            "publisher": m.get("publisher"),
            "url": m.get("url"),
            "attribution": ATTRIBUTION[doc_id.split("-", 1)[0]],
            "synthetic": False,
        }
        sample = build(doc, extra, doc_gold[doc_id], section_gold(doc, doc_id, gold_rows), backend)
        index.append((doc_id, sample))
    entries = []
    for sample_id, sample in index:
        (OUT / f"{sample_id}.json").write_text(json.dumps(sample, ensure_ascii=False))
        keep = ("title", "gold", "pages", "synthetic")
        entries.append({"id": sample_id} | {k: sample[k] for k in keep})
        used.add(sample["gold"])
        for r in sample["runs"].values():
            used.update([r["pred"], *(k for k, _ in r["probs"])])
            used.update(t[0] for t in r["tags"].values())
        used.update(s.get("gold") for s in sample["sections"])
        tokens = sum(r["tokens"]["prompt"] for r in sample["runs"].values())
        print(f"{sample_id}: {sample['sections_total']} sections, {tokens:,} prompt tokens")
    labels = demo.labels_for(sorted(i for i in used if i))
    for node in builtin_taxonomy("doc_types").leaves():  # every type: an upload's probabilities
        labels[node.id] = node.label
    (OUT / "index.json").write_text(json.dumps({"samples": entries, "labels": labels}, indent=1))
    (OUT / "bench.json").write_text(json.dumps(bench_summary(), indent=1))
    for stale in OUT.glob("*.json"):  # samples no longer listed
        if stale.stem not in {"index", "bench"} | {e["id"] for e in entries}:
            stale.unlink()
    print(f"wrote {len(entries)} samples, index.json and bench.json to {OUT}")


if __name__ == "__main__":
    main()
