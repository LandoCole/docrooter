"""A dependency-free keyword-overlap backend for tests, demos and baselines."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import numpy as np

from docrooter.backends.base import Candidate, Scores

_WORD = re.compile(r"\w+")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


class KeywordBackend:
    """Scores by Jaccard overlap of lowercase word sets (``similarity`` kind)."""

    max_candidates: int | None = None

    def __init__(self, name: str = "keyword", temperature: float = 0.1):
        self.name = name
        self.temperature = temperature

    def score(
        self,
        texts: Sequence[str],
        candidates: Sequence[Candidate],
        *,
        instruction: str | None = None,
        multi_label: bool = False,
    ) -> Scores:
        cand_words = [_words(" ".join((c.text, *c.examples))) for c in candidates]
        values = np.zeros((len(texts), len(candidates)), dtype=float)
        for i, text in enumerate(texts):
            words = _words(text)
            for j, cw in enumerate(cand_words):
                union = len(words | cw)
                if union:
                    values[i, j] = len(words & cw) / union
        return Scores(values, "similarity")

    def close(self) -> None:
        """Nothing to release: this backend holds no model."""

    def supports(self, node_id: str) -> bool:
        return True

    def tokenizer(self) -> Callable[[str], int] | None:
        return None
