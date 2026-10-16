"""Embedding backend: cosine similarity between instructed texts and label vectors."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from docrooter._extras import require
from docrooter.backends.base import Candidate, Scores
from docrooter.backends.loading import attn_implementation, select_device, trust_decision
from docrooter.backends.recipes import Recipe, get_recipe


def _normalise(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norm > 0, norm, 1.0)


class EmbeddingBackend:
    """Zero-shot classification with a sentence-embedding model (``similarity`` scores).

    Construction is cheap: the model loads on the first ``encode``, ``score`` or
    ``tokenizer()`` call.
    """

    max_candidates: int | None = None

    def __init__(
        self,
        model: str,
        *,
        revision: str | None = None,
        device: str | None = None,
        trust_remote_code: bool | None = None,
        batch_size: int | None = None,
        **recipe_overrides: Any,
    ) -> None:
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
        self._model: Any = None
        self._cache: dict[tuple[str, str], np.ndarray] = {}
        self._label_cache: dict[tuple, np.ndarray] = {}

    def _load(self) -> Any:
        if self._model is None:
            st = require("sentence_transformers", "models")
            device = select_device(self._device)
            model = st.SentenceTransformer(
                self.model_id,
                device=device,
                revision=self.recipe.revision,
                trust_remote_code=self.trust_remote_code,
                model_kwargs={"attn_implementation": attn_implementation(device)},
            )
            model.max_seq_length = self.recipe.max_seq_len
            self._model = model
        return self._model

    def encode(self, texts: list[str], *, prompt: str) -> np.ndarray:
        """L2-normalised embeddings, one row per text, cached by ``(prompt, text)``."""
        missing = list(dict.fromkeys(t for t in texts if (prompt, t) not in self._cache))
        if missing:
            vectors = self._load().encode(
                missing,
                prompt=prompt,
                batch_size=self.recipe.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            for text, vec in zip(missing, np.asarray(vectors, dtype=np.float32), strict=True):
                self._cache[(prompt, text)] = vec
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([self._cache[(prompt, t)] for t in texts])

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
