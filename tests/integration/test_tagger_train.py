"""Integration smoke for the element tagger: stage A and stage B with the real UltraBERT body
on a ~500-example synthetic slice of the merged topic tree, then save, load and tag.

Records seconds per fine-tuning step and peak memory (RSS, process footprint, MPS driver) in
``benchmarks/.cache/tagger-smoke.json``. Run it alone, under the memory guard:

    pgrep -fl "benchmarks\\.|docrooter serve|corpus build|teach label"   # must print nothing
    PYTHONPATH=src .venv/bin/python -m pytest tests/integration/test_tagger_train.py -m models -s
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path

import pytest

from benchmarks import _guard
from benchmarks.memory_check import MpsSampler, footprint_bytes, peak_rss_bytes
from docrooter.corpus.store import CorpusElement
from docrooter.tagger import ElementTagger, FunctionRow, calibrate_tagger, train_tagger
from docrooter.taxonomies import builtin_taxonomy
from docrooter.teach.dataset import input_text
from docrooter.train import Dataset, Example
from docrooter.train import trainer as trainer_module

pytestmark = pytest.mark.models

MODEL_PROCESSES = r"benchmarks\.|docrooter serve|corpus build|teach label"

OUT = Path(__file__).resolve().parents[2] / "benchmarks" / ".cache" / "tagger-smoke.json"
FAMILIES = ("annual_report", "research_paper")
LEAVES_PER_FAMILY = 5
PER_LEAF = 50  # 2 x 5 x 50 = 500 examples
SPARSE = 8  # one sparse leaf per family: left untrained
VARIANTS = (
    "As discussed in this section, {d}",
    "{d} The figures below are for the fiscal year.",
    "In summary: {d}",
    "This part covers the following. {d}",
    "{d} See also the notes that follow.",
)


def _element(family: str, leaf, i: int) -> CorpusElement:
    description = leaf.description or leaf.label
    text = VARIANTS[i % len(VARIANTS)].format(d=description) + f" (item {i})"
    kind = "table" if i % 10 == 0 else "paragraph"
    table = None
    if kind == "table":
        table = f"| {leaf.label} | Value |\n|---|---|\n| row {i} | {i * 3} |"
    return CorpusElement(
        doc_id=f"{family}-{i % 7}",
        element_id=f"{leaf.id}-{i}",
        kind=kind,
        text=text,
        heading_path=[family.replace("_", " ").title(), leaf.label],
        section_title=leaf.label,
        page=1,
        table_markdown=table,
    )


def _slice() -> tuple[Dataset, list[FunctionRow], list[CorpusElement]]:
    topics = builtin_taxonomy("element_topics")
    examples, rows, probes = [], [], []
    for family in FAMILIES:
        leaves = topics.children(family)[: LEAVES_PER_FAMILY + 1]
        for j, leaf in enumerate(leaves):
            n = SPARSE if j == LEAVES_PER_FAMILY else PER_LEAF
            for i in range(n):
                element = _element(family, leaf, i)
                examples.append(Example(text=input_text(element), path=[family, leaf.id]))
                if i < 10:
                    risk = "risk" in leaf.id
                    rows.append(
                        FunctionRow(
                            text=input_text(element),
                            answers={"risk_disclosure": 0.9 if risk else 0.1},
                            prescreen={"citation": 0.05},
                        )
                    )
            probes.append(_element(family, leaf, 999))
    return Dataset(examples), rows, probes


def _timed_finetune(log: dict):
    original = trainer_module._finetune_with_fallback

    def wrapper(body, triplets, **kwargs):
        started = time.perf_counter()
        original(body, triplets, **kwargs)
        log["finetune_seconds"] = time.perf_counter() - started
        log["triplets"] = len(triplets)
        log["batch_size"] = body.batch_size
        log["device"] = body._device_in_use

    return wrapper


@pytest.fixture(scope="module")
def guarded():
    """Env caps before torch loads (4 threads, 8 GB budget, MPS watermarks); skip while any
    other model process runs."""
    _guard.apply_env_caps()
    found = subprocess.run(
        ["pgrep", "-fl", MODEL_PROCESSES], capture_output=True, text=True, check=False
    ).stdout.splitlines()
    others = [line for line in found if not line.startswith(f"{os.getpid()} ")]
    if others:
        pytest.skip(f"another model process is running: {others[0][:80]}")


def test_stage_a_and_b_smoke(guarded, tmp_path, monkeypatch):
    dataset, rows, probes = _slice()
    assert len(dataset) == len(FAMILIES) * (LEAVES_PER_FAMILY * PER_LEAF + SPARSE)
    results: dict = {"examples": len(dataset), "function_rows": len(rows)}

    with MpsSampler() as sampler:
        started = time.perf_counter()
        stage_a = train_tagger(dataset, rows, stage="A")
        results["stage_a_seconds"] = round(time.perf_counter() - started, 1)
        for family in FAMILIES:
            sparse = builtin_taxonomy("element_topics").children(family)[LEAVES_PER_FAMILY].id
            assert sparse in stage_a.untrained
        stage_a.close()

        log: dict = {}
        monkeypatch.setattr(trainer_module, "_finetune_with_fallback", _timed_finetune(log))
        started = time.perf_counter()
        stage_b = train_tagger(dataset, rows, stage="B", max_triplets=480, epochs=1)
        results["stage_b_seconds"] = round(time.perf_counter() - started, 1)
        steps = math.ceil(log["triplets"] / log["batch_size"])
        results["stage_b"] = {
            **{k: v for k, v in log.items() if k != "finetune_seconds"},
            "steps": steps,
            "finetune_seconds": round(log["finetune_seconds"], 1),
            "seconds_per_step": round(log["finetune_seconds"] / steps, 2),
        }
        calibration = calibrate_tagger(stage_b, dataset, rows)
        results["temperatures"] = dict(stage_b.temperatures)
        results["calibration_n"] = {k: v.get("n") for k, v in calibration.items()}

        bundle = tmp_path / "element-tagger-smoke"
        stage_b.save(bundle)
        expected = stage_b.tag(probes[:4], FAMILIES[0])
        stage_b.close()
        loaded = ElementTagger.load(bundle)
        tagged = loaded.tag(probes[:4], FAMILIES[0])
        free = loaded.tag(probes)
        loaded.close()

    for a, b in zip(tagged, expected, strict=True):
        assert a["topic"].labels == b["topic"].labels
        for label, p in b["topic"].probs.items():
            assert a["topic"].probs[label] == pytest.approx(p, abs=1e-3)
    correct = sum(
        tags["topic"].labels[0] == leaf
        for tags, leaf in zip(free, [p.element_id.rsplit("-", 1)[0] for p in probes], strict=True)
    )
    results["probe_top1"] = f"{correct}/{len(probes)}"
    results["peak_rss_gb"] = round(peak_rss_bytes() / 1024**3, 3)
    footprint = footprint_bytes()
    if footprint is not None:
        results["peak_footprint_gb"] = round(footprint[1] / 1024**3, 3)
    results["peak_mps_driver_gb"] = round(sampler.peak_driver / 1024**3, 3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    assert results["peak_footprint_gb" if footprint else "peak_rss_gb"] < 8.0
