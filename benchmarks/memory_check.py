"""Peak memory of the two heaviest paths, against the targets in the v0.3 spec (section 2).

Cases, each in a fresh process so its peak is its own:

- ``classify``: ``load`` + ``classify_document`` on doc_042 (the 642-page RP2040 datasheet)
  with UltraBERT-Embed-150M against ``builtin:doc_types``.
- ``train``: ``train()`` on B1 with k=16 shots per leaf and ``finetune_body=True``.

The target for both is the memory budget (``DOCROOTER_MEMORY_BUDGET_GB``, 8 GiB by default)
for the process total. On macOS that is the peak physical footprint
(``task_info(TASK_VM_INFO).ledger_phys_footprint_peak``, which counts Metal buffers as well
as host memory on unified memory); elsewhere it is peak RSS plus peak Metal driver memory.
Also recorded: peak RSS (``ru_maxrss``, host pages only) and, from a sampler thread, the
peaks of ``torch.mps.current_allocated_memory()`` and ``driver_allocated_memory()``. The device is
the default order (CUDA, MPS, CPU) unless ``--device`` is given. Results are merged into
``benchmarks/results/<date>/memory.json`` by ``(case, device)``.

    PYTHONPATH=src python -m benchmarks.memory_check --case classify
    PYTHONPATH=src python -m benchmarks.memory_check --case train --device cpu
    PYTHONPATH=src python -m benchmarks.memory_check --case all      # both, one after another
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import datetime
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from benchmarks import _guard

RESULTS_ROOT = Path(__file__).parent / "results"
DOC_042 = Path(__file__).parent / "corpus" / "pdfs" / "doc_042.pdf"
CLASSIFY_BACKEND = "embedding:dmis-lab/UltraBERT-Embed-150M"
TRAIN_BASE = "ultrabert-embed"
CASES = ("classify", "train")
GB = 1024**3


def peak_rss_bytes() -> int:
    """This process's peak resident set size (``ru_maxrss`` is bytes on macOS, KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


class _TaskVmInfo(ctypes.Structure):
    """``task_vm_info_data_t`` (mach/task_info.h): named up to the footprint peak (rev 2)."""

    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("region_count", ctypes.c_int32),
        ("page_size", ctypes.c_int32),
        *(
            (name, ctypes.c_uint64)
            for name in (
                "resident_size", "resident_size_peak", "device", "device_peak", "internal",
                "internal_peak", "external", "external_peak", "reusable", "reusable_peak",
                "purgeable_volatile_pmap", "purgeable_volatile_resident",
                "purgeable_volatile_virtual", "compressed", "compressed_peak",
                "compressed_lifetime", "phys_footprint", "min_address", "max_address",
                "ledger_phys_footprint_peak",
            )
        ),
        # Later revisions' fields: the kernel fills a revision only if all of it fits.
        ("_rest", ctypes.c_uint64 * 64),
    ]  # fmt: skip


_TASK_VM_INFO = 22


def footprint_bytes() -> tuple[int, int] | None:
    """``(current, peak)`` physical footprint of this process on macOS, else ``None``.

    The footprint is what Activity Monitor shows as Memory: host pages, compressed pages and,
    on Apple silicon, Metal buffers.
    """
    if sys.platform != "darwin":
        return None
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        libc.mach_task_self.restype = ctypes.c_uint32
        info = _TaskVmInfo()
        count = ctypes.c_uint32(ctypes.sizeof(info) // 4)
        status = libc.task_info(
            libc.mach_task_self(), _TASK_VM_INFO, ctypes.byref(info), ctypes.byref(count)
        )
    except (OSError, AttributeError):
        return None
    if status != 0 or count.value * 4 < _TaskVmInfo._rest.offset:
        return None  # an old kernel without the rev-2 footprint peak
    return int(info.phys_footprint), int(info.ledger_phys_footprint_peak)


class MpsSampler:
    """Polls ``torch.mps.driver_allocated_memory()`` in a thread and keeps the peak.

    Torch is only consulted once something else has imported it; off MPS it records nothing.
    """

    def __init__(self, interval: float = 0.2) -> None:
        self.interval = interval
        self.peak_driver = 0
        self.peak_allocated = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        torch = sys.modules.get("torch")
        if torch is None or not torch.backends.mps.is_available():
            return
        self.peak_driver = max(self.peak_driver, int(torch.mps.driver_allocated_memory()))
        self.peak_allocated = max(self.peak_allocated, int(torch.mps.current_allocated_memory()))

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self._sample()
            except Exception:  # noqa: BLE001 - a failed sample must not kill the measurement
                pass

    def __enter__(self) -> MpsSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        self._thread.join()
        try:
            self._sample()
        except Exception:  # noqa: BLE001
            pass


def _gb(n: int) -> float:
    return round(n / GB, 3)


def case_classify(device: str, marks: dict[str, float]) -> dict[str, Any]:
    from docrooter import classify_document, load
    from docrooter.taxonomies import builtin_taxonomy

    doc = load(DOC_042)
    marks["peak_rss_after_parse_gb"] = _gb(peak_rss_bytes())
    result = classify_document(doc, builtin_taxonomy("doc_types"), backend=CLASSIFY_BACKEND)
    return {
        "document": DOC_042.name,
        "pages": doc.meta.get("n_pages"),
        "sections": len(doc.sections),
        "backend": CLASSIFY_BACKEND,
        "predicted": result.best.node_ids[-1] if result.best.node_ids else None,
        "confidence": round(result.confidence, 4),
    }


def case_train(device: str, marks: dict[str, float]) -> dict[str, Any]:
    from benchmarks.data import INSTRUCTIONS, load_b1, pool_shots
    from benchmarks.systems import BASE_MODELS, TRAIN_SEED
    from docrooter import train

    data = load_b1(n_test=1000, seed=0)  # the few-shot benchmark's pool
    shots = pool_shots(data.train_pool, 16, seed=TRAIN_SEED)
    marks["peak_rss_after_data_gb"] = _gb(peak_rss_bytes())
    model = train(
        data.taxonomy,
        shots,
        base_model=BASE_MODELS[TRAIN_BASE],
        finetune_body=True,
        device=device,
        seed=TRAIN_SEED,
        instruction=INSTRUCTIONS["b1"],
    )
    metrics = model.manifest["train_metrics"]
    recipe = model.manifest["recipe"]
    model.close()
    return {
        "dataset": "b1",
        "k": 16,
        "n_train": len(shots),
        "base_model": BASE_MODELS[TRAIN_BASE],
        "finetune_body": True,
        "triplets": metrics["triplets"],
        "max_seq_len": recipe["max_seq_len"],
        "batch_size": metrics["batch_size"],
        "trained_on": metrics["device"],  # differs from "device" after an MPS-to-CPU fallback
    }


RUNNERS: dict[str, Callable[[str, dict[str, float]], dict[str, Any]]] = {
    "classify": case_classify,
    "train": case_train,
}


def run_case(case: str, device: str | None) -> dict[str, Any]:
    """Run one case in this process on ``device`` (``None``: the default order)."""
    from docrooter import runtime
    from docrooter.backends.loading import select_device

    device = select_device(device)
    os.environ["DOCROOTER_DEVICE"] = device
    marks: dict[str, float] = {}
    started = time.perf_counter()
    with MpsSampler() as mps:
        details = RUNNERS[case](device, marks)
    peak = peak_rss_bytes()
    footprint = footprint_bytes()
    if footprint is not None:
        total, method = footprint[1], "peak phys_footprint (task_info)"
    else:
        total, method = peak + mps.peak_driver, "peak RSS + peak MPS driver"
    target = runtime.settings().memory_budget_gb
    row: dict[str, Any] = {
        "case": case,
        "device": device,
        "peak_total_gb": _gb(total),
        "total_method": method,
        "peak_rss_gb": _gb(peak),
        **marks,
        "target_gb": target,
        "within_target": total / GB <= target,
        "seconds": round(time.perf_counter() - started, 1),
        **details,
    }
    if device.startswith("mps"):
        row["peak_mps_driver_gb"] = _gb(mps.peak_driver)
        row["peak_mps_allocated_gb"] = _gb(mps.peak_allocated)
        row["mps_fraction"] = round(runtime.settings().mps_fraction(_recommended_bytes()), 4)
    names = [*_guard.ENV_CAPS, *_guard.MPS_CAP_VARS]
    row["env"] = {name: os.environ.get(name) for name in names}
    row["platform"] = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "system": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    return row


def _recommended_bytes() -> float:
    torch = sys.modules.get("torch")
    try:
        return float(torch.mps.recommended_max_memory())  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 - only for the record
        from docrooter import runtime

        return runtime.estimated_recommended_bytes()


def save(row: dict[str, Any], out_dir: Path) -> Path:
    """Merge ``row`` into ``memory.json`` (one row per case and device)."""
    path = out_dir / "memory.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    rows = [r for r in rows if (r["case"], r["device"]) != (row["case"], row["device"])]
    rows.append(row)
    rows.sort(key=lambda r: (CASES.index(r["case"]), r["device"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--case", choices=[*CASES, "all"], default="all")
    parser.add_argument("--device", default=None, help="torch device (default: cuda, mps, cpu)")
    parser.add_argument("--out", type=Path, default=None, help="results dir (default: dated)")
    _guard.add_guard_argument(parser)
    args = parser.parse_args(argv)
    _guard.single_model_process("benchmarks.memory_check", allow_concurrent=args.allow_concurrent)
    out_dir = args.out or RESULTS_ROOT / datetime.date.today().isoformat()

    if args.case == "all":  # one fresh process per case, one after the other
        for case in CASES:
            command = [sys.executable, "-m", "benchmarks.memory_check", "--case", case]
            command += ["--out", str(out_dir)]
            if args.device:
                command += ["--device", args.device]
            code = subprocess.run(command, check=False).returncode
            if code:
                return code
        return 0

    row = run_case(args.case, args.device)
    path = save(row, out_dir)
    verdict = "within" if row["within_target"] else "OVER"
    mps = f", MPS driver {row['peak_mps_driver_gb']} GB" if "peak_mps_driver_gb" in row else ""
    print(
        f"{args.case} on {row['device']}: peak total {row['peak_total_gb']} GB (RSS "
        f"{row['peak_rss_gb']} GB{mps}; {verdict} the {row['target_gb']} GB budget), "
        f"{row['seconds']}s -> {path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
