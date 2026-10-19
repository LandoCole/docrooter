"""The transformers teacher engine (spec §7.2): any platform, on MPS, CUDA or CPU.

Same prompt and scoring as the mlx engine, through the model's own chat template. Uses the
``models`` extra (torch and transformers), which already holds what this engine needs, so there
is no separate ``teach`` extra. Unquantised float16 weights take 2 bytes a parameter, so
within an 8 GiB budget this engine suits models up to about 2B parameters; larger teachers
belong on mlx.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from docrooter._extras import require
from docrooter.backends.loading import after_model_load, prepare_model_load
from docrooter.teach.base import LogprobTeacher, engine_name

__all__ = ["HFTeacher"]


class _TorchRunner:
    """Prefills a prefix into a transformers ``DynamicCache``; scores each suffix on top of it,
    computing the last position's logits only, then crops the cache back to the prefix."""

    def __init__(self, model: Any, torch: Any, device: str) -> None:
        self.model = model
        self.torch = torch
        self.device = device

    def _ids(self, ids: Sequence[int]) -> Any:
        return self.torch.tensor([list(ids)], dtype=self.torch.long, device=self.device)

    def prefill(self, ids: Sequence[int]) -> tuple[Any, int]:
        with self.torch.inference_mode():
            out = self.model(input_ids=self._ids(ids), use_cache=True, logits_to_keep=1)
        return out.past_key_values, len(ids)

    def score_next(
        self, state: tuple[Any, int], ids: Sequence[int], candidates: Sequence[int]
    ) -> list[float]:
        cache, length = state
        torch = self.torch
        try:
            with torch.inference_mode():
                out = self.model(
                    input_ids=self._ids(ids),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                logprobs = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
                picked = logprobs[torch.tensor(list(candidates), device=logprobs.device)]
                values = [float(v) for v in picked.cpu().tolist()]
        finally:
            _crop_to(cache, length)
        return values

    def close(self) -> None:
        self.model = None


def _crop_to(cache: Any, length: int) -> None:
    """Crop every layer back to ``length`` tokens (the prefix), whatever it holds now: a
    forward pass that failed part-way may have advanced some layers and not others. A negative
    ``crop`` removes that many tokens (transformers 5 rejects a positive one)."""
    for layer in cache.layers:
        excess = layer.get_seq_length() - length
        if excess > 0:
            layer.crop(-excess)


class HFTeacher(LogprobTeacher):
    """A teacher on transformers: ``HFTeacher(repo, revision)``, loaded on the first item.

    ``device`` defaults to the runtime's choice (:func:`docrooter.configure`, else cuda, mps,
    cpu). Weights load in ``dtype``: ``"float16"`` by default on every device, so the labels'
    dtype does not depend on the machine; ``None`` keeps the checkpoint's (``"auto"``). The
    dtype is part of :attr:`name` (``hf:<repo>@<rev>#float16``). Other keyword arguments go to
    :class:`~docrooter.teach.base.LogprobTeacher`.
    """

    def __init__(
        self,
        repo: str,
        revision: str,
        *,
        device: str | None = None,
        dtype: str | None = "float16",
        **options: Any,
    ) -> None:
        self.repo = repo
        self.revision = revision
        self.device = device
        self.dtype = dtype
        name = engine_name("hf", repo, revision, dtype)
        super().__init__(name, loader=self._load_model, **options)

    def _load_model(self) -> tuple[Any, _TorchRunner]:
        device = prepare_model_load(self.device, self.name)
        torch = require("torch", "models")
        transformers = require("transformers", "models")
        dtype = "auto" if self.dtype is None else getattr(torch, self.dtype)
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.repo, revision=self.revision)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.repo, revision=self.revision, dtype=dtype
        )
        model.to(device).eval()
        after_model_load(device, self.name)
        self.device = device
        return tokenizer, _TorchRunner(model, torch, device)
