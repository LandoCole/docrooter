"""Benchmark systems: a registry of named backend specs and the classifiers built from them."""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from benchmarks.data import BENCH_VIEW, INSTRUCTIONS, BenchData, pool_shots
from docrooter import Beam, Classifier, TrainedModel
from docrooter.classify.strategies import Flat, Greedy

logger = logging.getLogger("benchmarks")

ALL_DATASETS = ("b1", "b2", "b3")
ZERO_SHOT_STRATEGY = Beam(3)
ZERO_SHOT_THRESHOLD = 0.0  # force every section down to a leaf so accuracy is comparable


class SystemUnavailable(Exception):
    """A registered system cannot run in this environment (the message says why)."""


@dataclass(frozen=True)
class System:
    """One thing to benchmark: a backend spec and where it applies.

    ``datasets`` lists the benchmarks it runs on; ``b3_subset_only`` restricts it to the
    like-for-like B3 subset (for systems too slow to run on the full B3 test sample).
    """

    name: str
    spec: str
    datasets: tuple[str, ...] = ALL_DATASETS
    b3_subset_only: bool = False


def _fixed(
    spec: str, *, datasets: tuple[str, ...] = ALL_DATASETS, b3_subset_only: bool = False
) -> Callable[[str, Mapping[str, str]], System]:
    def build(name: str, env: Mapping[str, str]) -> System:
        return System(name, spec, datasets, b3_subset_only)

    return build


def _require_backend(module: str) -> None:
    if importlib.util.find_spec(f"docrooter.backends.{module}") is None:
        raise SystemUnavailable(f"docrooter.backends.{module} is not available yet")


def _openai(name: str, env: Mapping[str, str]) -> System:
    if not env.get("DOCROOTER_BENCH_LLM_URL"):
        raise SystemUnavailable("DOCROOTER_BENCH_LLM_URL is not set")
    model = env.get("DOCROOTER_BENCH_LLM_MODEL")
    if not model:
        raise SystemUnavailable("DOCROOTER_BENCH_LLM_MODEL is not set")
    _require_backend("openai")
    return System(name, f"openai:{model}")


def _jev(name: str, env: Mapping[str, str]) -> System:
    if not env.get("TYPESAFE_API_KEY"):
        raise SystemUnavailable("TYPESAFE_API_KEY is not set")
    _require_backend("jev")
    return System(name, "jev:jev-1.13.0")


# Registry keyed by short system name; each entry builds a System or raises SystemUnavailable.
REGISTRY: dict[str, Callable[[str, Mapping[str, str]], System]] = {
    "ultrabert-embed": _fixed("embedding:dmis-lab/UltraBERT-Embed-150M"),
    "bge-small": _fixed("embedding:BAAI/bge-small-en-v1.5"),
    "deberta-nli": _fixed("nli:MoritzLaurer/deberta-v3-base-zeroshot-v2.0", b3_subset_only=True),
    "ultrabert-mlm": _fixed("mlm-choice:dmis-lab/UltraBERT-380M", datasets=("b1", "b2")),
    "openai": _openai,
    "jev": _jev,
}


def available_systems(
    only: Sequence[str] | None = None, env: Mapping[str, str] | None = None
) -> list[System]:
    """The registered systems that can run here, in registry order.

    Unavailable ones are logged and left out, unless named in ``only``, which raises. Unknown
    names in ``only`` raise ``KeyError``.
    """
    env = os.environ if env is None else env
    unknown = [n for n in only or () if n not in REGISTRY]
    if unknown:
        raise KeyError(f"unknown system(s) {unknown}; known: {sorted(REGISTRY)}")
    systems = []
    for name, build in REGISTRY.items():
        if only and name not in only:
            continue
        try:
            systems.append(build(name, env))
        except SystemUnavailable as exc:
            if only:
                raise
            logger.warning("skipping system %s: %s", name, exc)
    return systems


def dataset_runs(system: System, data_keys: Sequence[str]) -> list[str]:
    """Result dataset names ``system`` runs on, given the requested benchmarks.

    ``b3`` is the full test sample, ``b3-subset`` the like-for-like subset every system reports.
    """
    runs = []
    for key in data_keys:
        if key not in system.datasets:
            continue
        if key == "b3":
            if not system.b3_subset_only:
                runs.append("b3")
            runs.append("b3-subset")
        else:
            runs.append(key)
    return runs


def build_classifier(
    system: System,
    data: BenchData,
    *,
    strategy: Greedy | Beam | Flat = ZERO_SHOT_STRATEGY,
    threshold: float = ZERO_SHOT_THRESHOLD,
) -> Classifier:
    """A classifier for ``system`` on a benchmark's taxonomy with the shared benchmark view.

    The backend is passed as a spec string, so each classifier loads its own fresh model and
    timings do not benefit from an earlier run's cache.
    """
    return Classifier(
        data.taxonomy,
        backend=system.spec,
        strategy=strategy,
        view=BENCH_VIEW,
        instruction=INSTRUCTIONS[data.name],
        threshold=threshold,
    )


# --- few-shot: trained models -----------------------------------------------------------

FEW_SHOT_KS = (4, 8, 16, 64)
BASE_MODELS = {
    "ultrabert-embed": "dmis-lab/UltraBERT-Embed-150M",
    "bge-small": "BAAI/bge-small-en-v1.5",
}
VARIANTS = {"heads": False, "finetuned": True}  # variant name -> finetune_body
BUNDLE_ROOT = Path(__file__).parent / ".cache" / "bundles"
TRAIN_SEED = 0


def bundle_dir(dataset: str, base: str, k: int, variant: str, root: Path | None = None) -> Path:
    """Where the bundle of ``base``/``variant`` trained on ``k`` shots of ``dataset`` lives."""
    return (root or BUNDLE_ROOT) / dataset / base / str(k) / variant


def trained_model(
    data: BenchData,
    base: str,
    k: int,
    variant: str,
    *,
    device: str | None = None,
    root: Path | None = None,
    batch_size: int | None = None,
    max_triplets: int | None = None,
) -> tuple[TrainedModel, float, int]:
    """The model trained on ``k`` shots per leaf of ``data.train_pool``: ``(model, seconds, n)``.

    A bundle already under :func:`bundle_dir` (for the same taxonomy) is loaded instead of
    retrained; ``seconds`` is then the training time recorded in it. ``n`` is the number of
    training examples (a leaf with fewer than ``k`` in the pool gives all it has).
    ``batch_size`` (default: the base model's recipe) is the fine-tuning and encoding batch size;
    a smaller one saves memory but changes the contrastive loss. ``max_triplets`` (default:
    ``train``'s 20000) caps the fine-tuning set. Either one gives the bundle its own directory.
    """
    from docrooter import train

    shots = pool_shots(data.train_pool, k, seed=TRAIN_SEED)
    suffix = ("" if batch_size is None else f"-bs{batch_size}") + (
        "" if max_triplets is None else f"-mt{max_triplets}"
    )
    path = bundle_dir(data.name, base, k, variant + suffix, root)
    if (path / "manifest.json").exists():
        model = TrainedModel.load(path, device=device)
        if model.taxonomy_hash == data.taxonomy.content_hash():
            logger.info("reusing trained bundle %s", path)
            return model, float(model.manifest["train_metrics"]["seconds"]), len(shots)
        logger.warning("bundle %s was trained on another taxonomy; retraining", path)
    shutil.rmtree(path, ignore_errors=True)
    model = train(
        data.taxonomy,
        shots,
        base_model=BASE_MODELS[base],
        finetune_body=VARIANTS[variant],
        device=device,
        seed=TRAIN_SEED,
        instruction=INSTRUCTIONS[data.name],
        batch_size=batch_size,
        **({} if max_triplets is None else {"max_triplets": max_triplets}),
    )
    model.save(path)
    seconds = float(model.manifest["train_metrics"]["seconds"])
    del model  # the weights stay on disk; classifiers load their own copy (see below)
    return TrainedModel.load(path, device=device), seconds, len(shots)


def build_trained_classifier(
    model: TrainedModel,
    data: BenchData,
    *,
    strategy: Greedy | Beam | Flat = ZERO_SHOT_STRATEGY,
    threshold: float = ZERO_SHOT_THRESHOLD,
    device: str | None = None,
) -> Classifier:
    """A classifier around a trained model, with the same view and settings as zero-shot.

    The classifier gets its own copy of the model, loaded from the model's bundle, so its
    embedding cache starts empty and timings do not benefit from an earlier cell's texts.
    """
    path = getattr(model, "path", None)
    backend = TrainedModel.load(path, device=device) if path is not None else model
    return Classifier(
        data.taxonomy,
        backend=backend,
        strategy=strategy,
        view=BENCH_VIEW,
        instruction=INSTRUCTIONS[data.name],
        threshold=threshold,
    )
