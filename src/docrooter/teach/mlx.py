"""The mlx-lm teacher engine (spec §7.2): the default on Apple silicon (``teach-mlx`` extra).

The pinned teacher is Qwen3-4B-Instruct-2507 at 4 bits; the fallback, Qwen3-8B at 4 bits, is
a hybrid thinking model and is prompted with thinking disabled (``enable_thinking=False``, which
:class:`~docrooter.teach.base.LogprobTeacher` always passes to the chat template).

Metal memory is capped with ``mx.set_memory_limit`` at the runtime's Metal budget (the process
budget minus host headroom, :mod:`docrooter.runtime`) unless ``memory_limit_gb`` is given.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from docrooter import runtime
from docrooter._extras import require
from docrooter.errors import BackendError
from docrooter.teach.base import LogprobTeacher

__all__ = ["DEFAULT_TEACHER", "FALLBACK_TEACHER", "MlxTeacher"]

DEFAULT_TEACHER = (
    "mlx:mlx-community/Qwen3-4B-Instruct-2507-4bit@50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
)
FALLBACK_TEACHER = "mlx:mlx-community/Qwen3-8B-4bit@545dc4251c05440727734bcd94334791f6ab0192"

# Architectures whose logits are exactly ``head(model.model(x))``, so the answer position's
# logits can be computed alone instead of the full (positions x vocabulary) matrix.
_LAST_POSITION_HEAD = {"qwen3"}
_CLEAR_EVERY = 256  # items between returns of MLX's buffer cache to the system


@dataclass
class _MlxPrefix:
    cache: list[Any]
    length: int


class _MlxRunner:
    """Prefills a prefix into an mlx-lm prompt cache; scores each suffix on top of it, then
    trims the cache back to the prefix (research §2.4)."""

    def __init__(self, model: Any, mx: Any, cache_module: Any, prefill_step: int) -> None:
        self.model = model
        self.mx = mx
        self.cache_module = cache_module
        self.prefill_step = prefill_step
        self.head = _last_position_head(model)
        self._scored = 0

    def prefill(self, ids: Sequence[int]) -> _MlxPrefix:
        mx = self.mx
        cache = self.cache_module.make_prompt_cache(self.model)
        if not self.cache_module.can_trim_prompt_cache(cache):
            raise BackendError(
                "this model's prompt cache cannot be trimmed; use_prefix_cache=False"
            )
        ids = list(ids)
        for start in range(0, len(ids), self.prefill_step):
            chunk = mx.array(ids[start : start + self.prefill_step])[None]
            self.model(chunk, cache=cache)  # only the cache is evaluated, never the logits
            mx.eval([c.state for c in cache])
        return _MlxPrefix(cache, len(ids))

    def score_next(
        self, state: _MlxPrefix, ids: Sequence[int], candidates: Sequence[int]
    ) -> list[float]:
        mx = self.mx
        x = mx.array(list(ids))[None]
        try:
            if self.head is not None:
                hidden = self.model.model(x, cache=state.cache)
                logits = self.head(hidden[:, -1:, :])[0, -1]
            else:
                logits = self.model(x, cache=state.cache)[0, -1]
            logits = logits.astype(mx.float32)
            logprobs = (logits - mx.logsumexp(logits))[mx.array(list(candidates))]
            mx.eval(logprobs)
        finally:
            self.cache_module.trim_prompt_cache(state.cache, len(ids))
        self._scored += 1
        if self._scored % _CLEAR_EVERY == 0:
            mx.clear_cache()
        values = [float(v) for v in logprobs.tolist()]
        if not all(math.isfinite(v) for v in values):
            raise BackendError(
                "the teacher's answer log-probabilities are not finite (float16 overflow?); "
                "load it with dtype='bfloat16' or dtype=None"
            )
        return values

    def close(self) -> None:
        self.model = None
        self.mx.clear_cache()


def _last_position_head(model: Any) -> Any:
    """The output projection, for architectures in :data:`_LAST_POSITION_HEAD`; else None."""
    if getattr(model, "model_type", None) not in _LAST_POSITION_HEAD:
        return None
    args = getattr(model, "args", None)
    inner = getattr(model, "model", None)
    if args is None or inner is None:
        return None
    if getattr(args, "tie_word_embeddings", False):
        return inner.embed_tokens.as_linear
    return getattr(model, "lm_head", None)


class MlxTeacher(LogprobTeacher):
    """A teacher on mlx-lm: ``MlxTeacher(repo, revision)``, loaded on the first item.

    ``prefill_step`` bounds the tokens per prefill pass. ``dtype`` is the compute type of the
    unquantised parameters (scales, norms, embeddings): ``"float16"`` by default, which on an
    M1 scored the 20 integration items about 30% faster than the checkpoint's bfloat16 (M1 has
    no native bfloat16) with the same answers (probabilities within 0.02); ``None`` keeps the
    checkpoint's type. Non-finite scores raise, naming the fix. Other keyword arguments go to
    :class:`~docrooter.teach.base.LogprobTeacher` (``use_prefix_cache``,
    ``max_cached_prefixes``).
    """

    def __init__(
        self,
        repo: str,
        revision: str,
        *,
        memory_limit_gb: float | None = None,
        prefill_step: int = 512,
        dtype: str | None = "float16",
        **options: Any,
    ) -> None:
        self.repo = repo
        self.revision = revision
        self.memory_limit_gb = memory_limit_gb
        self.prefill_step = prefill_step
        self.dtype = dtype
        self._mx: Any = None
        super().__init__(f"mlx:{repo}@{revision}", loader=self._load_model, **options)

    def _load_model(self) -> tuple[Any, _MlxRunner]:
        mx = require("mlx.core", "teach-mlx")
        mlx_lm = require("mlx_lm", "teach-mlx")
        cache_module = require("mlx_lm.models.cache", "teach-mlx")
        limit = self.memory_limit_gb or runtime.settings().metal_budget_gb
        mx.set_memory_limit(int(limit * runtime.GB))
        model, tokenizer = mlx_lm.load(self.repo, revision=self.revision)
        if self.dtype is not None:
            model.set_dtype(getattr(mx, self.dtype))
        self._mx = mx
        return tokenizer, _MlxRunner(model, mx, cache_module, self.prefill_step)

    def peak_memory_gb(self) -> float:
        """MLX's peak memory in this process, in GiB (0 before the model loads)."""
        return float(self._mx.get_peak_memory()) / runtime.GB if self._mx is not None else 0.0
