"""Embedding backend: cosine similarity between instructed texts and label vectors."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from docrooter._extras import require
from docrooter.backends.base import Candidate, Scores
from docrooter.backends.loading import (
    after_model_load,
    attn_implementation,
    is_mps_oom,
    move_backend_to_cpu,
    prepare_model_load,
    release_memory,
    trust_decision,
    warn_cpu_fallback,
)
from docrooter.backends.recipes import Recipe, get_recipe
from docrooter.errors import BackendError

DEFAULT_EMBEDDING_MODEL = "dmis-lab/UltraBERT-Embed-150M"
DEFAULT_CACHE_SIZE = 50_000


def _normalise(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norm > 0, norm, 1.0)


class EmbeddingBackend:
    """Zero-shot classification with a sentence-embedding model (``similarity`` scores).

    Construction is cheap: the model loads on the first ``encode``, ``score`` or
    ``tokenizer()`` call. Text embeddings are kept in an LRU cache of ``cache_size`` entries
    (0 disables it). :meth:`close` releases the model and the caches; the next call reloads.
    """

    max_candidates: int | None = None

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        *,
        revision: str | None = None,
        device: str | None = None,
        trust_remote_code: bool | None = None,
        batch_size: int | None = None,
        cache_size: int = DEFAULT_CACHE_SIZE,
        **recipe_overrides: Any,
    ) -> None:
        if cache_size < 0:
            raise ValueError(f"cache_size must be >= 0, got {cache_size}")
        self.model_id = model
        self.name = f"embedding:{model}"
        self.recipe: Recipe = get_recipe(
            model, revision=revision, batch_size=batch_size, **recipe_overrides
        )
        self.temperature = self.recipe.temperature
        self._device = device
        # A caller-chosen revision is not the pinned one the recipe vouches for, so it
        # needs an explicit opt-in before any remote code runs.
        unpinned = revision is not None and revision != get_recipe(model).revision
        self.trust_remote_code = trust_decision(
            model,
            self.recipe.model_copy(update={"revision": None}) if unpinned else self.recipe,
            trust_remote_code,
        )
        self.cache_size = cache_size
        self._model: Any = None
        self._device_in_use = "cpu"
        self._cache: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()
        self._label_cache: dict[tuple, np.ndarray] = {}
        # Set when closing would lose weights that cannot be reloaded (an unsaved fine-tune).
        self._unrecoverable: str | None = None

    @property
    def batch_size(self) -> int:
        """The encoding batch: the recipe's, else the default for the device in use and the
        memory budget (32 on CUDA; elsewhere 16 at 8 GiB or more, less below that)."""
        return self.recipe.batch_size_for(self._device_in_use)

    def _load(self) -> Any:
        if self._model is None:
            if self._unrecoverable is not None:
                raise BackendError(self._unrecoverable)
            device = prepare_model_load(self._device, self.name)
            try:
                model = self._build(device)
            except Exception as exc:
                if not is_mps_oom(exc, device):
                    raise
                oom = True
            else:
                oom = False
            if oom:  # outside the handler, so the failed attempt's tensors can be freed
                release_memory()
                warn_cpu_fallback(self.name, "loading")
                device = self._device = "cpu"
                model = self._build(device)
            after_model_load(device, self.name)
            self._model, self._device_in_use = model, device
        return self._model

    def _build(self, device: str) -> Any:
        st = require("sentence_transformers", "models")
        model_kwargs: dict[str, Any] = {"attn_implementation": attn_implementation(device)}
        if device.split(":")[0] != "cuda":
            # transformers 5 defaults to dtype="auto" (the checkpoint's); pin float32 off CUDA
            model_kwargs["dtype"] = "float32"
        model = st.SentenceTransformer(
            self.model_id,
            device=device,
            revision=self.recipe.revision,
            trust_remote_code=self.trust_remote_code,
            model_kwargs=model_kwargs,
        )
        model.max_seq_length = self.recipe.max_seq_len
        return model

    def close(self) -> None:
        """Drop the model and the caches and release their memory; the next call reloads."""
        self._model = None
        self._cache.clear()
        self._label_cache.clear()
        release_memory()

    def encode(self, texts: list[str], *, prompt: str) -> np.ndarray:
        """L2-normalised embeddings, one row per text, cached by ``(prompt, text)`` (LRU)."""
        found: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for text in dict.fromkeys(texts):
            key = (prompt, text)
            vec = self._cache.get(key)
            if vec is None:
                missing.append(text)
            else:
                self._cache.move_to_end(key)
                found[text] = vec
        if missing:
            vectors = self._encode(missing, prompt)
            for text, vec in zip(missing, np.asarray(vectors, dtype=np.float32), strict=True):
                found[text] = vec
                self._remember((prompt, text), vec)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([found[t] for t in texts])

    def _encode(self, texts: list[str], prompt: str) -> np.ndarray:
        """Run the model; on an MPS out-of-memory error, once more on the CPU."""
        for attempt in range(2):
            model = self._load()
            try:
                return model.encode(
                    texts,
                    prompt=prompt,
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            except Exception as exc:
                if attempt or not is_mps_oom(exc, self._device_in_use):
                    raise
            del model
            move_backend_to_cpu(self, "encoding")  # the same weights, possibly fine-tuned
        raise AssertionError("unreachable")  # pragma: no cover

    def _remember(self, key: tuple[str, str], vec: np.ndarray) -> None:
        if self.cache_size == 0:
            return
        self._cache[key] = vec
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _label_vector(self, candidate: Candidate, query_prompt: str) -> np.ndarray:
        # Examples are encoded as queries, so the instruction is part of their cache key.
        key = (
            self.recipe.label_prompt,
            candidate.text,
            candidate.examples,
            query_prompt if candidate.examples else None,
        )
        vec = self._label_cache.get(key)
        if vec is None:
            rows = [self.encode([self.recipe.label_prompt + candidate.text], prompt="")]
            if candidate.examples:
                rows.append(self.encode(list(candidate.examples), prompt=query_prompt))
            vec = _normalise(np.concatenate(rows).mean(axis=0))
            self._label_cache[key] = vec
        return vec

    def score(
        self,
        texts: Sequence[str],
        candidates: Sequence[Candidate],
        *,
        instruction: str | None = None,
        multi_label: bool = False,
    ) -> Scores:
        query_prompt = self.recipe.query_prompt.format(
            instruction=instruction or self.recipe.default_instruction
        )
        if not texts or not candidates:
            return Scores(np.zeros((len(texts), len(candidates))), "similarity")
        queries = self.encode(list(texts), prompt=query_prompt)
        labels = np.stack([self._label_vector(c, query_prompt) for c in candidates])
        return Scores(queries @ labels.T, "similarity")

    def supports(self, node_id: str) -> bool:
        return True

    def tokenizer(self) -> Callable[[str], int] | None:
        tok = self._load().tokenizer
        return lambda text: len(tok.encode(text, add_special_tokens=False))
