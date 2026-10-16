"""TypeSafe "Jev" (System One) backend: typed choice and yes/no probabilities per section."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from docrooter._extras import require
from docrooter.backends._async import run_sync
from docrooter.backends._secrets import redact
from docrooter.backends.base import Candidate, ScoreKind, Scores
from docrooter.errors import BackendError

API_KEY_ENV = "TYPESAFE_API_KEY"
DEFAULT_MODEL = "jev-1.13.0"
DEFAULT_INSTRUCTION = "Which option best describes this document section?"
_MAX_CHOICES = 255  # the API rejects a Choice with more options


class JevBackend:
    """Classify sections with TypeSafe's Jev model through the ``typesafe-sdk`` (``jev`` extra).

    Single-label scoring asks one ``Choice`` per text whose options are the candidates, keyed
    ``c0..cN`` with the candidate text (plus examples) as description, and returns its
    probabilities in candidate order (``softmax``). Multi-label scoring asks one yes/no ``Noul``
    per candidate in the same request and returns their probabilities (``independent``).
    The API key comes from ``api_key`` or the ``TYPESAFE_API_KEY`` environment variable. Pin
    ``model`` (default ``jev-1.13.0``); aliases such as ``jev-latest`` move without notice.

    ``score`` is synchronous. It runs its own event loop, or a worker thread when called from a
    thread that already runs one.
    """

    max_candidates: int | None = _MAX_CHOICES

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        concurrency: int = 16,
        timeout: float = 30,
        revision: str | None = None,
    ) -> None:
        if revision is not None:
            raise BackendError("the jev backend does not support model revisions")
        key = (api_key or os.environ.get(API_KEY_ENV, "")).strip()
        if not key:
            raise BackendError(
                f"no API key for the jev backend: set the {API_KEY_ENV} environment variable "
                "or pass api_key="
            )
        self._api_key = key
        self.model_id = model
        self.name = f"jev:{model}"
        self.temperature = 1.0
        self.concurrency = max(1, concurrency)
        self.timeout = timeout

    def _client(self, sdk: Any) -> Any:
        return sdk.AsyncTypeSafeClient(api_key=self._api_key, timeout=self.timeout)

    @staticmethod
    def _choice_criteria(candidates: Sequence[Candidate]) -> dict[str, str]:
        return {
            f"c{i}": c.text + (" Examples: " + "; ".join(c.examples) if c.examples else "")
            for i, c in enumerate(candidates)
        }

    async def _score_all(
        self,
        sdk: Any,
        texts: Sequence[str],
        candidates: Sequence[Candidate],
        instruction: str,
        multi_label: bool,
    ) -> np.ndarray:
        n = len(candidates)
        semaphore = asyncio.Semaphore(self.concurrency)

        def questions() -> dict[str, Any]:
            if multi_label:
                return {
                    f"n{i}": sdk.Noul(
                        instructions=f"Does this option apply to the section: {c.text}?"
                    )
                    for i, c in enumerate(candidates)
                }
            return {
                "choice": sdk.Choice(
                    instructions=instruction, criteria=self._choice_criteria(candidates)
                )
            }

        def read(response: Any) -> np.ndarray:
            if multi_label:
                return np.array(
                    [
                        float(response.nouls[f"n{i}"].noul) if f"n{i}" in response.nouls else 0.0
                        for i in range(n)
                    ]
                )
            probs = response.choices["choice"].probabilities
            row = np.array([float(probs.get(f"c{i}", 0.0)) for i in range(n)])
            total = row.sum()
            return row / total if total > 0 else np.full(n, 1.0 / n)

        async def one(client: Any, text: str) -> np.ndarray:
            async with semaphore:
                response = await client.system_one(
                    state={"section": text}, questions=questions(), model=self.model_id
                )
            try:
                return read(response)
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise BackendError(
                    f"{self.name}: unexpected response shape: {type(exc).__name__}: "
                    f"{redact(exc, self._api_key)}"
                ) from None

        # Client construction, entry and exit are inside the try too: SDK errors raised there
        # may quote the key, so they are redacted and wrapped like request errors.
        try:
            async with self._client(sdk) as client:
                rows = await asyncio.gather(*(one(client, t) for t in texts))
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(
                f"{self.name}: request failed: {type(exc).__name__}: {redact(exc, self._api_key)}"
            ) from None
        return np.vstack(rows)

    def score(
        self,
        texts: Sequence[str],
        candidates: Sequence[Candidate],
        *,
        instruction: str | None = None,
        multi_label: bool = False,
    ) -> Scores:
        n = len(candidates)
        kind: ScoreKind = "independent" if multi_label else "softmax"
        if not multi_label and n > _MAX_CHOICES:
            raise BackendError(f"jev takes at most {_MAX_CHOICES} options per choice, got {n}")
        if not len(texts) or not n:
            return Scores(np.zeros((len(texts), n)), kind)
        sdk = require("typesafe_sdk", "jev")
        values = run_sync(
            self._score_all(sdk, texts, candidates, instruction or DEFAULT_INSTRUCTION, multi_label)
        )
        return Scores(values, kind)

    def supports(self, node_id: str) -> bool:
        return True

    def tokenizer(self) -> Callable[[str], int] | None:
        return None
