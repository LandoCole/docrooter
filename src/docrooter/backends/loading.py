"""Device selection, runtime settings, memory release, attention implementation and the
remote-code trust gate."""

from __future__ import annotations

import gc
import importlib
import logging
import sys
from collections.abc import Callable
from typing import Any, TypeVar

from docrooter import events, runtime
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


def prepare_model_load(preference: str | None = None, name: str | None = None) -> str:
    """Apply the :func:`docrooter.configure` settings that must precede torch, then pick the
    device for a model load.

    This starts the ``model_load`` pipeline event (``name`` is the backend's); the matching end
    is reported by :func:`after_model_load`.

    Backends call this before importing anything that imports torch, so the MPS watermark
    variables still take effect. It never imports torch itself (beyond device detection);
    :func:`after_model_load` applies the rest once the model is loaded.
    """
    runtime.apply_before_torch(runtime.settings())
    device = select_device(preference)
    events.start("model_load", backend=name, device=device)
    return device


def after_model_load(device: str, name: str | None = None) -> None:
    """Apply the thread count and the MPS memory fraction, if torch is loaded, and end the
    ``model_load`` pipeline event started by :func:`prepare_model_load`."""
    events.finish("model_load", backend=name, device=device)
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


def place_on_device(backend: Any, model: Any, device: str) -> str:
    """Move a just-built ``model`` to ``device``; returns the device it ended up on.

    If the move runs out of MPS memory, the model goes to the CPU instead and the backend's
    ``_device`` and ``_device_in_use`` become ``"cpu"``. Handling it here, inside the load,
    keeps the ``model_load`` event to one start and one end and builds the model once.
    """
    backend._device_in_use = device
    try:
        model.to(device)
    except Exception as exc:
        if not is_mps_oom(exc, device):
            raise
    else:
        return device
    # Outside the handler, so the failed move's tensors can be freed.
    release_memory()
    warn_cpu_fallback(backend.name, "loading")
    model.to("cpu")
    backend._device = backend._device_in_use = "cpu"
    return "cpu"


def move_backend_to_cpu(backend: Any, what: str) -> None:
    """Continue ``backend`` on the CPU with the same weights.

    A loaded model is moved (``model.to("cpu")``), never reloaded: a reload would fetch the
    original weights and silently replace a body fine-tuned in memory. With no model loaded
    yet, the next load builds it on the CPU. The backend keeps its model in ``_model`` and its
    device in ``_device`` (requested) and ``_device_in_use``.
    """
    model = getattr(backend, "_model", None)
    if model is not None:
        model.to("cpu")
    backend._device = backend._device_in_use = "cpu"
    release_memory()
    warn_cpu_fallback(backend.name, what)


def run_with_cpu_fallback(backend: Any, what: str, call: Callable[[], T]) -> T:
    """``call()``; if it runs out of MPS memory, move the backend to the CPU
    (:func:`move_backend_to_cpu`) and call once more.

    ``call`` must reach the model through the backend, and inputs must follow
    ``_device_in_use`` (set before a model is moved to its device, so an OOM during that move
    is caught too).
    """
    try:
        return call()
    except Exception as exc:
        if not is_mps_oom(exc, getattr(backend, "_device_in_use", "cpu")):
            raise
    # Outside the handler: the failed call's frames, and the tensors they hold, are gone.
    move_backend_to_cpu(backend, what)
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
