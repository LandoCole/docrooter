"""Build the demo's sample runs and benchmark summary from the committed benchmark results.

No model is called: every sample run is the Claude Haiku 5.5 benchmark result for that document
(``benchmarks/results/2026-10-20/classification/haiku-5.5__gold__*.json``), joined to the
document's own section titles and the view text each rung of the ladder sends. Only documents
whose licence allows redistribution are used (GAO reports, CC-BY-4.0 arXiv papers and CUAD
contracts, US federal court opinions, RFCs). The PDFs must be in the benchmark cache.

Run from the repository root:  PYTHONPATH=src python demo/build_samples.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.classification.section_sample import read_document  # noqa: E402
from benchmarks.classification_bench import PDF_DIR  # noqa: E402
from docrooter.flow import select_sections  # noqa: E402
from docrooter.ladder import LEVELS, document_view  # noqa: E402
from docrooter.taxonomies import builtin_taxonomy  # noqa: E402

RESULTS = ROOT / "benchmarks" / "results" / "2026-10-20" / "classification"
MANIFEST = ROOT / "benchmarks" / "training_corpus" / "manifest.jsonl"
OUT = ROOT / "demo" / "data"
LABEL = "haiku-5.5"
FLOWS = ("doc_first", "sections_first")
PRICE = {"prompt": 0.10e-6, "completion": 0.50e-6}  # Claude Haiku 5.5, US$ per token (list)
FULL_PREVIEW_CHARS = 2400
SECTIONS_SHOWN = 24

SAMPLES = [
    "gao-gao-07-707",
    "cuad-arcgroupinc_20171211_8-k_ex-10_1_10976103_ex-10_1_sponsorship_agreement",
    "arxiv-2501_07309",
    "rfc-8962",
    "uscourts-ca9-25-5220-0",
    "cuad-innoviva_inc_08_07_2014-ex-10_1-collaboration_agreement",
    "gao-gao-06-946",
    "arxiv-2404_01334",
]
ATTRIBUTION = {
    "gao": "U.S. Government Accountability Office; GAO products are not protected by copyright.",
    "cuad": "From CUAD (Contract Understanding Atticus Dataset), The Atticus Project, CC BY 4.0.",
    "arxiv": "arXiv preprint, CC BY 4.0; the authors hold the copyright.",
    "uscourts": "U.S. federal court opinion, a work of the U.S. government.",
    "rfc": "IETF RFC, copyright IETF Trust and the authors; reproduced unmodified under the IETF "
    "Trust Legal Provisions.",
}


def manifest() -> dict[str, dict]:
    rows = (json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip())
    return {r["id"]: r for r in rows}


def results() -> dict[tuple[str, str], dict[str, dict]]:
    """(flow, level) -> document id -> the benchmark record."""
    out = {}
    for flow in FLOWS:
        for level in LEVELS:
            path = RESULTS / f"{LABEL}__gold__batched__{flow}__{level}__{level}.json"
            data = json.loads(path.read_text())
            out[(flow, level)] = {r["id"]: r for r in data["documents"]}
    return out


def cost(tokens: dict) -> float:
    return tokens["prompt"] * PRICE["prompt"] + tokens["completion"] * PRICE["completion"]


def view_text(doc, level: str) -> dict:
    rendered = document_view(level).render(doc)
    text = rendered.text
    shown = text if len(text) <= FULL_PREVIEW_CHARS else text[:FULL_PREVIEW_CHARS].rstrip() + " …"
    return {"text": shown, "chars": len(text), "truncated": rendered.truncated}


def build_sample(doc_id: str, meta: dict, runs: dict) -> dict:
    doc = read_document(PDF_DIR / f"{doc_id}.pdf")
    sections = select_sections(doc)
    titles = {s.id: s.title for s in sections}
    family = doc_id.split("-", 1)[0]
    sample = {
        "id": doc_id,
        "title": meta.get("title") or doc.title or doc_id,
        "publisher": meta.get("publisher"),
        "pages": meta.get("pages"),
        "url": meta.get("url"),
        "attribution": ATTRIBUTION[family],
        "gold": None,
        "views": {level: view_text(doc, level) for level in LEVELS},
        "sections": [
            {"id": s.id, "title": s.title, "level": s.level} for s in sections[:SECTIONS_SHOWN]
        ],
        "sections_total": len(sections),
        "runs": {},
    }
    for (flow, level), by_doc in runs.items():
        r = by_doc[doc_id]
        sample["gold"] = r["gold"]
        tags = {
            t["section_id"]: {"pred": t.get("pred"), "gold": t.get("gold")}
            for t in r.get("sections", [])
        }
        sample["runs"][f"{flow}/{level}"] = {
            "pred": r["pred"],
            "correct": r["correct"],
            "decided_by": r["decided_by"],
            "flags": r["flags"],
            "candidates": r.get("candidates") if isinstance(r.get("candidates"), list) else [],
            "tokens": r["tokens"],
            "cost_usd": round(cost(r["tokens"]), 6),
            "tags": {sid: tags[sid] for sid in titles if sid in tags},
        }
    return sample


def bench_summary() -> dict:
    """Main-grid and held-out rows for the demo's benchmark table."""
    rows = []
    for path in sorted(RESULTS.glob("*.json")):
        data = json.loads(path.read_text())
        c, m = data["config"], data["metrics"]
        if c["section_level"] != c["doc_level"] or c.get("calls", "per_item") != "batched":
            continue
        if data["backend"]["label"] not in (LABEL, "ultrabert"):
            continue
        rows.append(
            {
                "backend": data["backend"]["label"],
                "docs": data["docs"],
                "flow": c["flow"],
                "level": c["doc_level"],
                "doc_acc": m["doc_type"]["accuracy"],
                "leaf_f1": (m.get("sections") or {}).get("leaf", {}).get("macro_f1"),
                "prompt_tokens": m["prompt_tokens_mean"],
                "exact": m["tokens_exact"],
                "documents": m["documents"],
            }
        )
    return {"rows": rows, "price": PRICE, "model": "claude-haiku-5-5"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta, runs = manifest(), results()
    doc_types, catalogue = builtin_taxonomy("doc_types"), builtin_taxonomy("section_catalogue")
    index = []
    for doc_id in SAMPLES:
        sample = build_sample(doc_id, meta[doc_id], runs)
        (OUT / f"{doc_id}.json").write_text(json.dumps(sample, ensure_ascii=False, indent=1))
        keep = ("id", "title", "gold", "pages")
        index.append({k: sample[k] for k in keep})
        print(f"{doc_id}: {sample['sections_total']} sections")
    labels = {}
    for taxonomy in (doc_types, catalogue):
        stack = list(taxonomy.children())
        while stack:
            node = stack.pop()
            labels[node.id] = node.label
            stack += taxonomy.children(node.id)
    (OUT / "index.json").write_text(json.dumps({"samples": index, "labels": labels}, indent=1))
    (OUT / "bench.json").write_text(json.dumps(bench_summary(), indent=1))
    print(f"wrote {len(index)} samples, index.json and bench.json to {OUT}")


if __name__ == "__main__":
    main()
