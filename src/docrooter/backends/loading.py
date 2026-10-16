"""Device selection, runtime settings, memory release, attention implementation and the
remote-code trust gate."""

from __future__ import annotations

import gc
import importlib
import logging
import sys
from collections.abc import Callable
from typing import Any, TypeVar

from docrooter import runtime
from docrooter.backends.recipes import Recipe
from docrooter.errors import BackendError

DEVICE_ENV = runtime.DEVICE_ENV
logger = logging.getLogger(__name__)
T = TypeVar("T")


def select_device(preference: str | None = None) -> str:
    """Pick a torch device: ``preference``, else the :func:`docrooter.configure` device, else
    ``$DOCROOTER_DEVICE``, else cuda, mps, cpu.

    Returns ``"cpu"`` when torch is not installed.
    """
    chosen = preference or runtime.configured_device()
    if chosen:
        return chosen
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def prepare_model_load(preference: str | None = None) -> str:
    """Apply the :func:`docrooter.configure` settings that must precede torch, then pick the
    device for a model load.

    Backends call this before importing anything that imports torch, so the MPS watermark
    variables still take effect. It never imports torch itself (beyond device detection);
    :func:`after_model_load` applies the rest once the model is loaded.
    """
    runtime.apply_before_torch(runtime.settings())
    return select_device(preference)


def after_model_load(device: str) -> None:
    """Apply the thread count and the MPS memory fraction, if torch is loaded."""
    torch = sys.modules.get("torch")
    if torch is not None:
        runtime.apply_to_torch(torch, runtime.settings(), device)


def is_mps_oom(exc: BaseException, device: str) -> bool:
    """Whether ``exc`` is the MPS allocator running out of memory on an MPS ``device``."""
    return device.split(":")[0] == "mps" and "MPS backend out of memory" in str(exc)


def warn_cpu_fallback(name: str, what: str) -> None:
    """Log that ``name`` ran out of MPS memory while ``what`` and continues on the CPU."""
    logger.warning(
        "%s ran out of MPS memory while %s; falling back to the CPU (slower). Raise "
        "DOCROOTER_MEMORY_BUDGET_GB or pass device='cpu' to choose this up front.",
        name,
        what,
    )


def run_with_cpu_fallback(backend: Any, what: str, call: Callable[[], T]) -> T:
    """``call()``; if it runs out of MPS memory, drop the backend's model, move the backend to
    the CPU and call once more (``call`` must (re)load the model through the backend).

    The backend keeps its model in ``_model`` and its device in ``_device`` (requested) and
    ``_device_in_use`` (set before the model is moved there).
    """
    try:
        return call()
    except Exception as exc:
        if not is_mps_oom(exc, getattr(backend, "_device_in_use", "cpu")):
            raise
    backend._model = None  # outside the handler: the failed call's tensors are gone too
    release_memory()
    warn_cpu_fallback(backend.name, what)
    backend._device = backend._device_in_use = "cpu"
    return call()


def release_memory() -> None:
    """Collect garbage and return cached accelerator memory to the system.

    Torch is only consulted when something already imported it.
    """
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        torch.mps.empty_cache()
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available() and cuda.is_initialized():
        cuda.empty_cache()


def attn_implementation(device: str) -> str:
    """``flash_attention_2`` on CUDA when ``flash_attn`` is importable, else ``sdpa``."""
    if device.split(":")[0] == "cuda":
        try:
            importlib.import_module("flash_attn")
        except ImportError:
            return "sdpa"
        return "flash_attention_2"
    return "sdpa"


def trust_decision(model_id: str, recipe: Recipe, explicit: bool | None) -> bool:
    """Whether to load ``model_id`` with ``trust_remote_code=True``.

    Remote code runs only if the caller passed ``trust_remote_code=True`` or the recipe both
    needs it and pins a ``revision``. A recipe that needs remote code without a pin raises.
    """
    if explicit:
        return True
    if not recipe.trust_remote_code:
        return False
    if explicit is None and recipe.revision is not None:
        return True
    raise BackendError(
        f"{model_id} needs remote code but no pinned revision is trusted for it; "
        "pass trust_remote_code=True to run it, or pin a revision in its recipe"
    )
