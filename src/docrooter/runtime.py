"""Process-level runtime settings: device, torch threads and the memory budget.

:func:`configure` only records the settings; nothing here imports torch. They are applied when
a backend loads a model: the MPS watermark variables before torch is imported
(:func:`apply_before_torch`), the thread count and the exact MPS fraction once it is
(:func:`apply_to_torch`). See :mod:`docrooter.backends.loading`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

DEVICE_ENV = "DOCROOTER_DEVICE"
THREADS_ENV = "DOCROOTER_THREADS"
BUDGET_ENV = "DOCROOTER_MEMORY_BUDGET_GB"
MPS_FRACTION_ENV = "DOCROOTER_MPS_MEMORY_FRACTION"
MPS_HIGH_WATERMARK_ENV = "PYTORCH_MPS_HIGH_WATERMARK_RATIO"
MPS_LOW_WATERMARK_ENV = "PYTORCH_MPS_LOW_WATERMARK_RATIO"
DEFAULT_BUDGET_GB = 8.0
GB = 1024**3  # budgets are in GiB, as macOS and torch report memory
# The low watermark (where the MPS allocator starts collecting) as a share of the high one.
_LOW_TO_HIGH = 0.8
# Before torch loads, Metal's recommended working set is estimated as this share of the RAM
# (11.84 of 16 GiB on an M1); torch's exact figure replaces the estimate once it is loaded.
_RECOMMENDED_SHARE = 0.74
# Unified memory: Metal buffers and the host side (Python, tokenizers, a CPU copy of the
# weights) share the budget, so Metal gets the budget minus this headroom (at least half).
MPS_HOST_HEADROOM_GB = 1.5


@dataclass(frozen=True)
class RuntimeSettings:
    """The effective settings: :func:`configure` values, else the environment, else defaults."""

    device: str | None = None
    threads: int | None = None
    memory_budget_gb: float = DEFAULT_BUDGET_GB
    mps_memory_fraction: float | None = None  # an explicit fraction; else from the budget

    @property
    def metal_budget_gb(self) -> float:
        """The share of the budget Metal may hold: the budget minus 1.5 GiB of host headroom,
        but at least half the budget."""
        budget = self.memory_budget_gb
        return max(budget - MPS_HOST_HEADROOM_GB, budget / 2)

    def mps_fraction(self, recommended_bytes: float) -> float:
        """The MPS per-process fraction: the explicit one, else the Metal share of the budget
        over Metal's recommended maximum (at most 1)."""
        if self.mps_memory_fraction is not None:
            return self.mps_memory_fraction
        return min(1.0, self.metal_budget_gb * GB / recommended_bytes)


_configured: dict[str, Any] = {}
_applied: tuple[int | None, float | None] = (None, None)  # threads, MPS fraction
_written: dict[str, str] = {}  # the watermark variables this process set


def configure(
    *,
    device: str | None = None,
    threads: int | None = None,
    memory_budget_gb: float | None = None,
    mps_memory_fraction: float | None = None,
) -> None:
    """Set process-wide defaults for model backends. ``None`` leaves a setting unchanged.

    - ``device``: the torch device backends use when they are not given one (beats
      ``$DOCROOTER_DEVICE``; a backend's own ``device=`` beats both). Unset, the order is
      CUDA, MPS, CPU.
    - ``threads``: ``torch.set_num_threads`` (else ``$DOCROOTER_THREADS``).
    - ``memory_budget_gb``: the memory a process should stay within, in GiB (else
      ``$DOCROOTER_MEMORY_BUDGET_GB``, default 8). It is meant for the whole process. On MPS,
      Metal and the host side share unified memory, so Metal gets a hard cap of the budget
      minus 1.5 GiB of host headroom (at least half the budget):
      ``torch.mps.set_per_process_memory_fraction(min(1, metal / recommended_max_memory()))``.
      The cap covers Metal memory only, not the process total. On CPU nothing caps the
      resident set; the budget only scales the default batch sizes (16 at 8 GiB or more,
      halved per halving below, at least 4).
    - ``mps_memory_fraction``: an explicit MPS fraction in ``(0, 1]`` instead of the one
      derived from the budget (else ``$DOCROOTER_MPS_MEMORY_FRACTION``).

    The settings take effect on the next model load. The MPS cap has an ordering caveat: if
    torch is not imported yet and neither ``PYTORCH_MPS_HIGH_WATERMARK_RATIO`` nor
    ``PYTORCH_MPS_LOW_WATERMARK_RATIO`` is set, both are first set from an estimate of the
    recommended maximum (high = the fraction, low = 0.8 x the fraction), so the cap holds
    from torch's first allocation; once torch is loaded the exact fraction is applied with
    ``set_per_process_memory_fraction``. Watermark variables you set yourself are never
    overridden, and then no fraction is applied. Call ``configure`` before anything imports
    torch to get the early cap.
    """
    if threads is not None:
        _check_threads(threads, "threads")
        _configured["threads"] = threads
    if memory_budget_gb is not None:
        _check_budget(memory_budget_gb, "memory_budget_gb")
        _configured["memory_budget_gb"] = float(memory_budget_gb)
    if mps_memory_fraction is not None:
        _check_fraction(mps_memory_fraction, "mps_memory_fraction")
        _configured["mps_memory_fraction"] = float(mps_memory_fraction)
    if device is not None:
        if not device.strip():
            raise ValueError("device must be a non-empty string such as 'cpu', 'mps' or 'cuda'")
        _configured["device"] = device


def settings() -> RuntimeSettings:
    """The effective settings: :func:`configure` values, else the environment variables."""
    threads = _configured.get("threads")
    if threads is None and (raw := _env(THREADS_ENV)):
        threads = _parse(raw, int, THREADS_ENV, "a positive integer")
        _check_threads(threads, f"${THREADS_ENV}")
    budget = _configured.get("memory_budget_gb")
    if budget is None and (raw := _env(BUDGET_ENV)):
        budget = _parse(raw, float, BUDGET_ENV, "a number of GiB")
        _check_budget(budget, f"${BUDGET_ENV}")
    fraction = _configured.get("mps_memory_fraction")
    if fraction is None and (raw := _env(MPS_FRACTION_ENV)):
        fraction = _parse(raw, float, MPS_FRACTION_ENV, "a number")
        _check_fraction(fraction, f"${MPS_FRACTION_ENV}")
    return RuntimeSettings(
        device=configured_device(),
        threads=threads,
        memory_budget_gb=DEFAULT_BUDGET_GB if budget is None else budget,
        mps_memory_fraction=fraction,
    )


def configured_device() -> str | None:
    """The :func:`configure` device, else ``$DOCROOTER_DEVICE``, else ``None``."""
    return _configured.get("device") or os.environ.get(DEVICE_ENV) or None


def default_batch_size(device: str, budget_gb: float | None = None) -> int:
    """The batch size when none is set: 32 on CUDA; else 16 at a budget of 8 GiB or more,
    halved for each halving of the budget below that, and never under 4."""
    if device.split(":")[0] == "cuda":
        return 32
    budget = settings().memory_budget_gb if budget_gb is None else budget_gb
    size = 16
    while budget < DEFAULT_BUDGET_GB and size > 4:
        budget *= 2
        size //= 2
    return size


def estimated_recommended_bytes() -> float:
    """An estimate of ``torch.mps.recommended_max_memory()`` from the RAM, without torch."""
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError, AttributeError):
        ram = 16 * GB
    return ram * _RECOMMENDED_SHARE


def watermark_env(current: RuntimeSettings) -> dict[str, str]:
    """The MPS watermark variables for ``current``, from the estimated recommended maximum."""
    fraction = current.mps_fraction(estimated_recommended_bytes())
    return {
        MPS_HIGH_WATERMARK_ENV: _ratio(fraction),
        MPS_LOW_WATERMARK_ENV: _ratio(fraction * _LOW_TO_HIGH),
    }


def _watermarks_are_ours(current: RuntimeSettings) -> bool:
    """True when no watermark variable is set, or both hold what :func:`watermark_env`
    gives (set by docrooter, maybe in a parent process), not values chosen by the user."""
    names = (MPS_HIGH_WATERMARK_ENV, MPS_LOW_WATERMARK_ENV)
    present = {k: os.environ[k] for k in names if k in os.environ}
    return not present or present in (_written, watermark_env(current))


def apply_before_torch(current: RuntimeSettings) -> bool:
    """Set the MPS watermark variables if torch is not loaded and neither is set by the user.

    Returns whether it set them.
    """
    if "torch" in sys.modules or not _watermarks_are_ours(current):
        return False
    _written.clear()
    _written.update(watermark_env(current))
    os.environ.update(_written)
    return True


def apply_to_torch(torch: Any, current: RuntimeSettings, device: str) -> None:
    """Apply the thread count and, on MPS, the exact memory fraction from the budget.

    Each value is applied once per change, so repeated model loads cost nothing. The fraction
    is skipped when the user set the watermark variables or torch lacks the call.
    """
    global _applied
    threads, applied_fraction = _applied
    if current.threads is not None and current.threads != threads:
        torch.set_num_threads(current.threads)
        threads = current.threads
    mps = getattr(torch, "mps", None)
    backend = getattr(getattr(torch, "backends", None), "mps", None)
    set_fraction = getattr(mps, "set_per_process_memory_fraction", None)
    recommended = getattr(mps, "recommended_max_memory", None)
    if (
        device.split(":")[0] == "mps"
        and callable(set_fraction)
        and backend is not None
        and backend.is_available()
        and _watermarks_are_ours(current)
    ):
        total = recommended() if callable(recommended) else estimated_recommended_bytes()
        fraction = current.mps_fraction(total)
        if fraction != applied_fraction:
            set_fraction(fraction)
            applied_fraction = fraction
    _applied = (threads, applied_fraction)


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _parse(raw: str, kind: type, name: str, what: str) -> Any:
    try:
        return kind(raw)
    except ValueError:
        raise ValueError(f"${name} must be {what}, got {raw!r}") from None


def _ratio(value: float) -> str:
    return f"{value:.4g}"


def _check_threads(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _check_budget(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        raise ValueError(f"{name} must be a positive number of GiB, got {value!r}")


def _check_fraction(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
        raise ValueError(f"{name} must be in (0, 1], got {value!r}")


def _reset() -> None:
    """Forget every :func:`configure` value and what was applied (for tests)."""
    global _applied
    _configured.clear()
    _written.clear()
    _applied = (None, None)
