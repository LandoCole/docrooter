"""Run benchmark suites and write their results as JSON, one row per system x dataset.

``uv run python -m benchmarks.run --suite zero-shot --device mps --n-test 1000``
"""

from __future__ import annotations

import argparse
import datetime
import gc
import importlib.metadata
import json
import logging
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from benchmarks.data import DATASET_KEYS, LOADERS, BenchData
from benchmarks.systems import (
    ZERO_SHOT_STRATEGY,
    System,
    available_systems,
    build_classifier,
    dataset_runs,
)
from docrooter import Dataset, evaluate
from docrooter.backends.recipes import get_recipe
from docrooter.backends.registry import parse_spec
from docrooter.evaluate.report import Report

logger = logging.getLogger("benchmarks")

Suite = Literal["zero-shot", "few-shot", "strategies"]
SUITES = ("zero-shot", "few-shot", "strategies")
RESULTS_ROOT = Path(__file__).parent / "results"
PACKAGES = ("torch", "transformers", "sentence-transformers", "docrooter")


# --- result rows ------------------------------------------------------------------------


def collect_env(device: str | None, specs: Sequence[str] = ()) -> dict[str, Any]:
    """Library versions (read from metadata, without importing torch) and pinned revisions."""
    env: dict[str, Any] = {"device": device or "auto"}
    for package in PACKAGES:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = None
        env[package.replace("-", "_")] = version
    revisions = {}
    for spec in specs:
        _, model, revision = parse_spec(spec)
        revisions[spec] = revision or get_recipe(model).revision
    env["revisions"] = revisions
    return env


def make_row(
    *,
    suite: str,
    dataset: str,
    system: str,
    strategy: str,
    shots: int,
    report: Report,
    env: dict[str, Any],
    wall_seconds: float,
) -> dict[str, Any]:
    """A result row. ``seconds_total`` is the timed classification pass (model loading and
    per-node diagnostics excluded); ``wall_seconds`` is the whole cell including both."""
    sps = report.sections_per_sec
    return {
        "suite": suite,
        "dataset": dataset,
        "system": system,
        "strategy": strategy,
        "shots": shots,
        "n": report.n,
        "macro_p": report.macro_p,
        "macro_r": report.macro_r,
        "macro_f1": report.macro_f1,
        "sections_per_sec": sps,
        "seconds_total": report.n / sps if sps > 0 else None,
        "seconds_per_example": 1 / sps if sps > 0 else None,
        "wall_seconds": wall_seconds,
        "report": report.model_dump(mode="json"),
        "env": env,
    }


def row_key(row: Mapping[str, Any]) -> tuple:
    return (row["suite"], row["dataset"], row["system"], row["strategy"], row["shots"])


class ResultStore:
    """Rows of one suite in ``<out_dir>/<suite>.json``, rewritten after every added row.

    Rows already on disk are kept, and a new row replaces the one with the same key, so a rerun
    of one system updates the file instead of discarding the others.
    """

    def __init__(self, out_dir: Path, suite: str):
        self.path = Path(out_dir) / f"{suite}.json"
        self.rows: list[dict[str, Any]] = []
        if self.path.exists():
            self.rows = json.loads(self.path.read_text(encoding="utf-8"))

    def add(self, row: dict[str, Any]) -> None:
        key = row_key(row)
        self.rows = [r for r in self.rows if row_key(r) != key] + [row]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(self.path)


# --- suites -----------------------------------------------------------------------------


def _free_memory() -> None:
    """Drop the last model so the next cell starts clean (torch is touched only if loaded)."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - best effort
        pass


def _load_data(
    keys: Sequence[str], n_test: int, loaders: Mapping[str, Callable[..., BenchData]]
) -> dict[str, BenchData]:
    data = {}
    for key in keys:
        start = time.perf_counter()
        data[key] = loaders[key](n_test=n_test, seed=0)
        logger.info(
            "loaded %s: %d test, %d pool examples, %d nodes (%.0fs)",
            key,
            len(data[key].test),
            len(data[key].train_pool),
            len(data[key].taxonomy),
            time.perf_counter() - start,
        )
    return data


def _test_set(data: BenchData, run_name: str) -> Dataset:
    if run_name.endswith("-subset"):
        subset = data.subset()
        assert subset is not None
        return subset
    return data.test


def run_zero_shot(
    *,
    device: str | None,
    n_test: int,
    store: ResultStore,
    only: Sequence[str] | None,
    datasets: Sequence[str],
    loaders: Mapping[str, Callable[..., BenchData]],
    node_decisions: bool,
    environ: Mapping[str, str] | None,
) -> list[dict[str, Any]]:
    """Every available system on every benchmark: Beam(3), threshold 0.0, text-only view."""
    systems = available_systems(only, environ)
    data = _load_data(datasets, n_test, loaders)
    rows: list[dict[str, Any]] = []
    failed: list[str] = []
    for system in systems:
        env = collect_env(device, [system.spec])
        for run_name in dataset_runs(system, datasets):
            try:
                row = _run_cell(
                    "zero-shot",
                    system,
                    run_name,
                    data[run_name.removesuffix("-subset")],
                    env,
                    node_decisions,
                )
            except Exception:  # noqa: BLE001 - keep going so one bad cell does not lose the run
                logger.exception("[zero-shot] %s on %s failed", system.name, run_name)
                failed.append(f"{system.name}/{run_name}")
                continue
            store.add(row)
            rows.append(row)
    if failed:
        raise RuntimeError(
            f"{len(failed)} cell(s) failed: {', '.join(failed)}; finished rows are in {store.path}"
            " (rerun a system with --only <system>, e.g. with --device cpu)"
        )
    return rows


def _run_cell(
    suite: str,
    system: System,
    run_name: str,
    data: BenchData,
    env: dict[str, Any],
    node_decisions: bool,
    *,
    strategy=ZERO_SHOT_STRATEGY,
    shots: int = 0,
) -> dict[str, Any]:
    test = _test_set(data, run_name)
    logger.info("[%s] %s on %s (%d examples) ...", suite, system.name, run_name, len(test))
    start = time.perf_counter()
    classifier = build_classifier(system, data, strategy=strategy)
    try:
        report = evaluate(classifier, test, node_decisions=node_decisions)
    finally:
        del classifier
        _free_memory()
    wall = time.perf_counter() - start
    logger.info(
        "[%s] %s on %s: hF1 %.3f macro-F1 %.3f, %.1f sections/s on %s (%.0fs)",
        suite,
        system.name,
        run_name,
        report.hf1,
        report.macro_f1,
        report.sections_per_sec,
        report.device,
        wall,
    )
    return make_row(
        suite=suite,
        dataset=run_name,
        system=system.name,
        strategy=strategy_label(strategy),
        shots=shots,
        report=report,
        env={**env, "device": report.device},
        wall_seconds=wall,
    )


def strategy_label(strategy: Any) -> str:
    """``"Beam(3)"``, ``"Greedy"``, ``"Flat"``."""
    width = getattr(strategy, "width", None)
    name = type(strategy).__name__
    return f"{name}({width})" if width is not None else name


def run_suite(
    suite: Suite,
    device: str | None,
    n_test: int,
    *,
    out_dir: str | Path | None = None,
    only: Sequence[str] | None = None,
    datasets: Sequence[str] = DATASET_KEYS,
    loaders: Mapping[str, Callable[..., BenchData]] | None = None,
    node_decisions: bool = True,
    environ: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Run one suite and return its rows, which are also written to ``<out_dir>/<suite>.json``.

    ``device`` (``None`` = automatic) is handed to the backends through ``$DOCROOTER_DEVICE``.
    """
    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}; choose from {SUITES}")
    unknown = [d for d in datasets if d not in DATASET_KEYS]
    if unknown:
        raise ValueError(f"unknown dataset(s) {unknown}; choose from {DATASET_KEYS}")
    if suite != "zero-shot":
        raise NotImplementedError(f"the {suite} suite is implemented in Task 15")
    if device:
        os.environ["DOCROOTER_DEVICE"] = device
    out = Path(out_dir) if out_dir else default_out_dir()
    return run_zero_shot(
        device=device,
        n_test=n_test,
        store=ResultStore(out, suite),
        only=only,
        datasets=datasets,
        loaders=loaders or LOADERS,
        node_decisions=node_decisions,
        environ=environ,
    )


def default_out_dir() -> Path:
    return RESULTS_ROOT / datetime.date.today().isoformat()


# --- CLI --------------------------------------------------------------------------------


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m benchmarks.run", description=__doc__)
    p.add_argument("--suite", choices=[*SUITES, "all"], default="zero-shot")
    p.add_argument("--device", default=None, help="torch device (default: cuda, mps, cpu)")
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--out", type=Path, default=None, help="default: benchmarks/results/<date>/")
    p.add_argument(
        "--only", action="append", default=None, metavar="SYSTEM", help="only this system (repeat)"
    )
    p.add_argument("--datasets", default=",".join(DATASET_KEYS), help="comma-separated: b1,b2,b3")
    p.add_argument(
        "--no-node-decisions",
        action="store_true",
        help="skip the per-node (teacher-forced) accuracy pass",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    out = args.out or default_out_dir()
    suites = SUITES if args.suite == "all" else (args.suite,)
    for suite in suites:
        try:
            rows = run_suite(
                suite,
                args.device,
                args.n_test,
                out_dir=out,
                only=args.only,
                datasets=[d.strip() for d in args.datasets.split(",") if d.strip()],
                node_decisions=not args.no_node_decisions,
            )
        except NotImplementedError as exc:
            if args.suite != "all":
                raise
            logger.warning("skipping suite %s: %s", suite, exc)
            continue
        logger.info("%s: %d rows written to %s", suite, len(rows), out / f"{suite}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
