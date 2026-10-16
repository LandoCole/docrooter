"""Device selection, attention implementation and the remote-code trust gate."""

from __future__ import annotations

import importlib
import os

from docrooter.backends.recipes import Recipe
from docrooter.errors import BackendError

DEVICE_ENV = "DOCROOTER_DEVICE"


def select_device(preference: str | None = None) -> str:
    """Pick a torch device: ``preference``, else ``$DOCROOTER_DEVICE``, else cuda, mps, cpu.

    Returns ``"cpu"`` when torch is not installed.
    """
    chosen = preference or os.environ.get(DEVICE_ENV)
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
