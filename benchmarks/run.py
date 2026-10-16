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

from benchmarks import systems as bench_systems
from benchmarks.data import DATASET_KEYS, LOADERS, BenchData
from benchmarks.systems import (
    BASE_MODELS,
    FEW_SHOT_KS,
    VARIANTS,
    ZERO_SHOT_STRATEGY,
    System,
    available_systems,
    build_classifier,
    dataset_runs,
)
from docrooter import Beam, Classifier, Dataset, evaluate
from docrooter.backends.recipes import get_recipe
from docrooter.backends.registry import parse_spec
from docrooter.classify.strategies import Flat, Greedy
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
    train_seconds: float | None = None,
    n_train: int | None = None,
) -> dict[str, Any]:
    """A result row. ``seconds_total`` is the timed classification pass (model loading and
    per-node diagnostics excluded); ``wall_seconds`` is the whole cell including both.
    ``train_seconds`` and ``n_train`` (few-shot and trained rows) are the time spent training
    the model and its number of training examples."""
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
        "train_seconds": train_seconds,
        "n_train": n_train,
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
    plan = {system.name: dataset_runs(system, datasets) for system in systems}
    # load only the benchmarks the selected systems run on (EDGAR and DBPedia are big downloads)
    needed = [
        key
        for key in DATASET_KEYS
        if any(run.removesuffix("-subset") == key for runs in plan.values() for run in runs)
    ]
    data = _load_data(needed, n_test, loaders)
    rows: list[dict[str, Any]] = []
    failed: list[str] = []
    for system in systems:
        env = collect_env(device, [system.spec])
        for run_name in plan[system.name]:
            try:
                dataset = data[run_name.removesuffix("-subset")]
                row = _run_cell(
                    "zero-shot",
                    system.name,
                    run_name,
                    dataset,
                    env,
                    node_decisions,
                    lambda system=system, dataset=dataset: build_classifier(system, dataset),
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
    system_name: str,
    run_name: str,
    data: BenchData,
    env: dict[str, Any],
    node_decisions: bool,
    build: Callable[[], Classifier],
    *,
    strategy: Greedy | Beam | Flat = ZERO_SHOT_STRATEGY,
    shots: int = 0,
    train_seconds: float | None = None,
    n_train: int | None = None,
) -> dict[str, Any]:
    """Evaluate the classifier ``build()`` makes on ``run_name`` and return its result row."""
    test = _test_set(data, run_name)
    logger.info("[%s] %s on %s (%d examples) ...", suite, system_name, run_name, len(test))
    start = time.perf_counter()
    classifier = build()
    try:
        report = evaluate(classifier, test, node_decisions=node_decisions)
    finally:
        del classifier
        _free_memory()
    wall = time.perf_counter() - start
    logger.info(
        "[%s] %s %s on %s: hF1 %.3f macro-F1 %.3f, %.1f sections/s on %s (%.0fs)",
        suite,
        system_name,
        strategy_label(strategy),
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
        system=system_name,
        strategy=strategy_label(strategy),
        shots=shots,
        report=report,
        env={**env, "device": report.device},
        wall_seconds=wall,
        train_seconds=train_seconds,
        n_train=n_train,
    )


def strategy_label(strategy: Any) -> str:
    """``"Beam(3)"``, ``"Greedy"``, ``"Flat"``."""
    width = getattr(strategy, "width", None)
    name = type(strategy).__name__
    return f"{name}({width})" if width is not None else name


def _matches(name: str, only: Sequence[str] | None) -> bool:
    """Whether a system called ``name`` is selected by ``--only`` (``"bge-small/heads"``
    is selected by ``bge-small`` or by its full name)."""
    return not only or name in only or name.partition("/")[0] in only


def _finish(failed: list[str], store: ResultStore, hint: str) -> None:
    if failed:
        raise RuntimeError(
            f"{len(failed)} cell(s) failed: {', '.join(failed)}; finished rows are in {store.path}"
            f" ({hint})"
        )


def run_few_shot(
    *,
    device: str | None,
    n_test: int,
    store: ResultStore,
    only: Sequence[str] | None,
    bases: Sequence[str],
    shots: Sequence[int],
    datasets: Sequence[str],
    loaders: Mapping[str, Callable[..., BenchData]],
    node_decisions: bool,
    bundle_root: Path | None,
    train_batch_size: int | None = None,
    train_max_triplets: int | None = None,
) -> list[dict[str, Any]]:
    """Heads-only and fine-tuned models on k shots per leaf, for each base and benchmark.

    Models train on ``pool_shots(train_pool, k, seed=0)`` and are evaluated with Beam(3) and
    threshold 0.0. Bundles are kept under ``benchmarks/.cache/bundles/`` and reused.
    """
    only_bases = [n.partition("/")[0] for n in only or ()]
    unknown = [n for n in [*only_bases, *bases] if n not in BASE_MODELS]
    if unknown:
        raise KeyError(f"unknown base(s) {unknown}; known: {sorted(BASE_MODELS)}")
    selected = [b for b in bases if not only or b in only_bases]
    if not selected:
        raise ValueError(f"--only {list(only or ())} selects none of the bases {list(bases)}")
    data = _load_data([k for k in DATASET_KEYS if k in datasets], n_test, loaders)
    rows: list[dict[str, Any]] = []
    failed: list[str] = []
    for key, dataset in data.items():
        for base in selected:
            env = collect_env(device, [f"embedding:{BASE_MODELS[base]}"])
            env["train_batch_size"] = train_batch_size
            env["train_max_triplets"] = train_max_triplets
            base_system = System(base, f"embedding:{BASE_MODELS[base]}")
            runs = dataset_runs(base_system, [key])
            for k in shots:
                for variant in VARIANTS:
                    name = f"{base}/{variant}"
                    if not _matches(name, only):
                        continue
                    cell = f"{name}/{k}/{key}"
                    try:
                        model, train_seconds, n_train = bench_systems.trained_model(
                            dataset,
                            base,
                            k,
                            variant,
                            device=device,
                            root=bundle_root,
                            batch_size=train_batch_size,
                            max_triplets=train_max_triplets,
                        )
                    except Exception:  # noqa: BLE001 - keep going so one bad cell does not lose the run
                        logger.exception("[few-shot] training %s failed", cell)
                        failed.append(cell)
                        continue
                    logger.info(
                        "[few-shot] %s: %d examples, trained in %.0fs", cell, n_train, train_seconds
                    )
                    for run_name in runs:
                        try:
                            row = _run_cell(
                                "few-shot",
                                name,
                                run_name,
                                dataset,
                                env,
                                node_decisions,
                                lambda model=model, dataset=dataset: (
                                    bench_systems.build_trained_classifier(
                                        model, dataset, device=device
                                    )
                                ),
                                shots=k,
                                train_seconds=train_seconds,
                                n_train=n_train,
                            )
                        except Exception:  # noqa: BLE001
                            logger.exception("[few-shot] %s on %s failed", cell, run_name)
                            failed.append(f"{cell}/{run_name}")
                            continue
                        store.add(row)
                        rows.append(row)
                    del model
                    _free_memory()
    _finish(failed, store, "rerun with --only <base>, e.g. with --device cpu")
    return rows


def _best_row(path: Path, keep: Callable[[dict[str, Any]], bool]) -> dict[str, Any] | None:
    """The B2 row with the highest hF1 among those ``keep`` accepts in a result file, if any."""
    if not path.exists():
        return None
    rows = [
        r for r in json.loads(path.read_text(encoding="utf-8")) if r["dataset"] == "b2" and keep(r)
    ]
    return max(rows, key=lambda r: r["report"]["hf1"], default=None)


DEFAULT_STRATEGY_ZERO_SHOT = "ultrabert-embed"
DEFAULT_STRATEGY_TRAINED = "ultrabert-embed/finetuned"


def pick_strategy_systems(
    out_dir: Path, trained_shots: int, zero_shot: str | None, trained: str | None
) -> tuple[str, str]:
    """The zero-shot system and the trained ``base/variant`` the strategies suite compares.

    Given explicitly, or the best B2 hF1 in ``<out_dir>/zero-shot.json`` (Beam(3)) and in
    ``<out_dir>/few-shot.json`` (at ``trained_shots``); a default when the file is missing.
    """
    if zero_shot is None:
        best = _best_row(
            out_dir / "zero-shot.json", lambda r: r["strategy"] == "Beam(3)" and r["shots"] == 0
        )
        zero_shot = best["system"] if best else None
        if zero_shot is None:
            logger.warning(
                "no zero-shot B2 rows in %s; using %s", out_dir, DEFAULT_STRATEGY_ZERO_SHOT
            )
            zero_shot = DEFAULT_STRATEGY_ZERO_SHOT
    if trained is None:
        best = _best_row(
            out_dir / "few-shot.json", lambda r: r["shots"] == trained_shots and "/" in r["system"]
        )
        trained = best["system"] if best else None
        if trained is None:
            logger.warning(
                "no %d-shot B2 rows in %s; using %s",
                trained_shots,
                out_dir,
                DEFAULT_STRATEGY_TRAINED,
            )
            trained = DEFAULT_STRATEGY_TRAINED
    base, _, variant = trained.partition("/")
    if base not in BASE_MODELS or variant not in VARIANTS:
        raise ValueError(f"unknown trained system {trained!r}; use <base>/<variant>")
    logger.info(
        "strategies compare zero-shot %s and trained %s (%d shots)",
        zero_shot,
        trained,
        trained_shots,
    )
    return zero_shot, trained


def run_strategies(
    *,
    device: str | None,
    n_test: int,
    store: ResultStore,
    only: Sequence[str] | None,
    datasets: Sequence[str],
    loaders: Mapping[str, Callable[..., BenchData]],
    node_decisions: bool,
    environ: Mapping[str, str] | None,
    bundle_root: Path | None,
    trained_shots: int,
    zero_shot: str | None,
    trained: str | None,
    train_batch_size: int | None = None,
    train_max_triplets: int | None = None,
) -> list[dict[str, Any]]:
    """Greedy, Beam(3) and Flat with one zero-shot system, Greedy and Beam(3) with a trained one.

    B2 only. Flat runs only for embedding backends; the trained model is the one from
    ``trained_shots`` shots (see :func:`pick_strategy_systems`).
    """
    if "b2" not in datasets:
        return []
    zero_name, trained_name = pick_strategy_systems(
        store.path.parent, trained_shots, zero_shot, trained
    )
    data = _load_data(["b2"], n_test, loaders)["b2"]
    rows: list[dict[str, Any]] = []
    failed: list[str] = []

    def record(row_or_error: str | dict[str, Any]) -> None:
        if isinstance(row_or_error, str):
            failed.append(row_or_error)
        else:
            store.add(row_or_error)
            rows.append(row_or_error)

    if _matches(zero_name, only):
        (system,) = available_systems([zero_name], environ)
        env = collect_env(device, [system.spec])
        strategies: list[Greedy | Beam | Flat] = [Greedy(), Beam(3)]
        if system.spec.startswith("embedding:"):
            strategies.append(Flat())
        for strategy in strategies:
            try:
                record(
                    _run_cell(
                        "strategies",
                        system.name,
                        "b2",
                        data,
                        env,
                        node_decisions,
                        lambda strategy=strategy: build_classifier(system, data, strategy=strategy),
                        strategy=strategy,
                    )
                )
            except Exception:  # noqa: BLE001
                logger.exception("[strategies] %s %s failed", system.name, strategy_label(strategy))
                record(f"{system.name}/{strategy_label(strategy)}")
    if _matches(trained_name, only):
        base, _, variant = trained_name.partition("/")
        env = collect_env(device, [f"embedding:{BASE_MODELS[base]}"])
        env["train_batch_size"] = train_batch_size
        env["train_max_triplets"] = train_max_triplets
        try:
            model, train_seconds, n_train = bench_systems.trained_model(
                data,
                base,
                trained_shots,
                variant,
                device=device,
                root=bundle_root,
                batch_size=train_batch_size,
                max_triplets=train_max_triplets,
            )
        except Exception:  # noqa: BLE001
            logger.exception("[strategies] training %s failed", trained_name)
            record(f"{trained_name}/train")
        else:
            for strategy in (Greedy(), Beam(3)):
                try:
                    record(
                        _run_cell(
                            "strategies",
                            trained_name,
                            "b2",
                            data,
                            env,
                            node_decisions,
                            lambda strategy=strategy: bench_systems.build_trained_classifier(
                                model, data, strategy=strategy, device=device
                            ),
                            strategy=strategy,
                            shots=trained_shots,
                            train_seconds=train_seconds,
                            n_train=n_train,
                        )
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "[strategies] %s %s failed", trained_name, strategy_label(strategy)
                    )
                    record(f"{trained_name}/{strategy_label(strategy)}")
            del model
            _free_memory()
    if not rows and not failed:
        logger.warning(
            "--only %s selects neither %s nor %s", list(only or ()), zero_name, trained_name
        )
    _finish(failed, store, "rerun with --only <system>, e.g. with --device cpu")
    return rows


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
    shots: Sequence[int] = FEW_SHOT_KS,
    bases: Sequence[str] = tuple(BASE_MODELS),
    bundle_root: Path | None = None,
    trained_shots: int = 16,
    zero_shot_system: str | None = None,
    trained_system: str | None = None,
    train_batch_size: int | None = None,
    train_max_triplets: int | None = None,
) -> list[dict[str, Any]]:
    """Run one suite and return its rows, which are also written to ``<out_dir>/<suite>.json``.

    ``device`` (``None`` = automatic) is handed to the backends through ``$DOCROOTER_DEVICE``.
    ``shots`` and ``bases`` are for the few-shot suite; ``trained_shots``, ``zero_shot_system``
    and ``trained_system`` for the strategies suite (the latter two default to the best B2
    systems found in the output directory's earlier result files). ``train_batch_size`` replaces
    the base model's recipe batch size when training (both trained suites).
    """
    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}; choose from {SUITES}")
    unknown = [d for d in datasets if d not in DATASET_KEYS]
    if unknown:
        raise ValueError(f"unknown dataset(s) {unknown}; choose from {DATASET_KEYS}")
    if device:
        os.environ["DOCROOTER_DEVICE"] = device
    out = Path(out_dir) if out_dir else default_out_dir()
    store = ResultStore(out, suite)
    loaders = loaders or LOADERS
    if suite == "few-shot":
        return run_few_shot(
            device=device,
            n_test=n_test,
            store=store,
            only=only,
            bases=bases,
            shots=shots,
            datasets=datasets,
            loaders=loaders,
            node_decisions=node_decisions,
            bundle_root=bundle_root,
            train_batch_size=train_batch_size,
            train_max_triplets=train_max_triplets,
        )
    if suite == "strategies":
        return run_strategies(
            device=device,
            n_test=n_test,
            store=store,
            only=only,
            datasets=datasets,
            loaders=loaders,
            node_decisions=node_decisions,
            environ=environ,
            bundle_root=bundle_root,
            trained_shots=trained_shots,
            zero_shot=zero_shot_system,
            trained=trained_system,
            train_batch_size=train_batch_size,
            train_max_triplets=train_max_triplets,
        )
    return run_zero_shot(
        device=device,
        n_test=n_test,
        store=store,
        only=only,
        datasets=datasets,
        loaders=loaders,
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
    p.add_argument("--shots", default=",".join(map(str, FEW_SHOT_KS)), help="few-shot: k values")
    p.add_argument(
        "--bases", default=",".join(BASE_MODELS), help="few-shot: comma-separated base models"
    )
    p.add_argument("--trained-shots", type=int, default=16, help="strategies: the trained k")
    p.add_argument("--zero-shot-system", default=None, help="strategies: default = best on B2")
    p.add_argument(
        "--trained-system", default=None, help="strategies: <base>/<variant>, default = best on B2"
    )
    p.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="few-shot/strategies: training batch size (default: the base model's recipe, 32)",
    )
    p.add_argument(
        "--train-max-triplets",
        type=int,
        default=None,
        help="few-shot/strategies: cap on fine-tuning triplets (default: 20000)",
    )
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
        rows = run_suite(
            suite,
            args.device,
            args.n_test,
            out_dir=out,
            only=args.only,
            datasets=[d.strip() for d in args.datasets.split(",") if d.strip()],
            node_decisions=not args.no_node_decisions,
            shots=[int(k) for k in args.shots.split(",") if k.strip()],
            bases=[b.strip() for b in args.bases.split(",") if b.strip()],
            trained_shots=args.trained_shots,
            zero_shot_system=args.zero_shot_system,
            trained_system=args.trained_system,
            train_batch_size=args.train_batch_size,
            train_max_triplets=args.train_max_triplets,
        )
        logger.info("%s: %d rows written to %s", suite, len(rows), out / f"{suite}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
