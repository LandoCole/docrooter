"""OpenAI-compatible chat-completions backend: the model answers with an option number."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from docrooter._extras import require
from docrooter.backends._async import run_sync
from docrooter.backends.base import Candidate, Scores
from docrooter.backends.mlm_choice import build_choice_prompt
from docrooter.errors import BackendError

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "Answer with the option number only."
DEFAULT_INSTRUCTION = "Which option best describes this document section?"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
_TOP_LOGPROBS = 20
_INTEGER = re.compile(r"\d+")


def _first_env(*names: str) -> str | None:
    return next((os.environ[n] for n in names if os.environ.get(n)), None)


def _logprob_distribution(top_logprobs: Sequence[dict[str, Any]], n: int) -> np.ndarray | None:
    """Softmax over the option numbers among a first token's ``top_logprobs``.

    Options the model did not list get ``-inf``. Tokens that differ only in whitespace (``"2"``
    and ``" 2"``) pool their probability. Returns ``None`` when no option number is listed.
    """
    logits = np.full(n, -np.inf)
    for entry in top_logprobs:
        token = str(entry.get("token", "")).strip()
        logprob = entry.get("logprob")
        if not token.isdigit() or logprob is None:
            continue
        index = int(token) - 1
        if 0 <= index < n:
            logits[index] = np.logaddexp(logits[index], float(logprob))
    if not np.isfinite(logits).any():
        return None
    probs = np.exp(logits - logits.max())
    return probs / probs.sum()


def _parse_answer(data: dict[str, Any], n: int) -> np.ndarray | None:
    """Probabilities over ``n`` options from one chat-completion response, or ``None``."""
    choice = (data.get("choices") or [{}])[0]
    content = ((choice.get("logprobs") or {}).get("content")) or []
    if content:
        probs = _logprob_distribution(content[0].get("top_logprobs") or [], n)
        if probs is not None:
            return probs
    message = (choice.get("message") or {}).get("content") or ""
    match = _INTEGER.search(message)
    if match and 1 <= int(match.group()) <= n:
        probs = np.zeros(n)
        probs[int(match.group()) - 1] = 1.0
        return probs
    return None


class OpenAIBackend:
    """Zero-shot classification through any OpenAI-compatible chat-completions endpoint.

    Each text becomes a numbered multiple-choice prompt answered with one number. When the server
    returns logprobs the scores are a softmax over the first token's option numbers; otherwise
    the first integer in the reply gets probability 1. Replies that name no option score a
    uniform distribution, with one warning per ``score`` call. Option numbers above 9 depend on
    the server tokenising them as one token. The model cannot judge options independently, so
    ``multi_label`` is ignored and the result is a ``softmax``.

    ``base_url`` and ``api_key`` default to ``DOCROOTER_LLM_BASE_URL`` / ``DOCROOTER_LLM_API_KEY``,
    then ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``. Without a ``base_url`` the OpenAI API is used
    and an API key is required; for other servers the key is optional. Needs the ``llm`` extra.
    """

    max_candidates: int | None = 50

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60,
        concurrency: int = 8,
        revision: str | None = None,
    ) -> None:
        if revision is not None:
            raise BackendError("the openai backend does not support model revisions")
        self.model_id = model
        self.name = f"openai:{model}"
        self.temperature = 1.0
        self.timeout = timeout
        self.concurrency = max(1, concurrency)
        self.base_url = (
            base_url or _first_env("DOCROOTER_LLM_BASE_URL", "OPENAI_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self._api_key = api_key or _first_env("DOCROOTER_LLM_API_KEY", "OPENAI_API_KEY")
        if self._api_key is None and self.base_url == DEFAULT_BASE_URL:
            raise BackendError(
                "no API key for the openai backend: set DOCROOTER_LLM_API_KEY or OPENAI_API_KEY "
                "(or pass api_key=), or point base_url at a server that needs none"
            )
        self._transport: Any = None  # an httpx transport; tests replace it with a MockTransport

    def _request(self, text: str, options: list[str], instruction: str) -> dict[str, Any]:
        prompt = build_choice_prompt(
            text,
            options,
            instruction=instruction,
            answer="",
            markers=[str(i) for i in range(1, len(options) + 1)],
        )
        return {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "logprobs": True,
            "top_logprobs": _TOP_LOGPROBS,
            "max_tokens": 3,
            "temperature": 0,
        }

    async def _complete(self, client: Any, body: dict[str, Any]) -> dict[str, Any]:
        httpx = require("httpx", "llm")
        try:
            response = await client.post("/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise BackendError(f"{self.name}: request failed: {exc!r}") from exc
        if response.status_code >= 400:
            raise BackendError(
                f"{self.name}: HTTP {response.status_code}: {response.text[:300].strip()}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise BackendError(f"{self.name}: response was not JSON") from exc

    async def _score_all(self, bodies: list[dict[str, Any]], n: int) -> np.ndarray:
        httpx = require("httpx", "llm")
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        semaphore = asyncio.Semaphore(self.concurrency)

        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
            transport=self._transport,
        ) as client:

            async def one(body: dict[str, Any]) -> np.ndarray | None:
                async with semaphore:
                    return _parse_answer(await self._complete(client, body), n)

            answers = await asyncio.gather(*(one(b) for b in bodies))

        unparsed = sum(a is None for a in answers)
        if unparsed:
            logger.warning(
                "%s: %d of %d replies named no option; scoring them uniformly",
                self.name,
                unparsed,
                len(answers),
            )
        uniform = np.full(n, 1.0 / n)
        return np.vstack([uniform if a is None else a for a in answers])

    def score(
        self,
        texts: Sequence[str],
        candidates: Sequence[Candidate],
        *,
        instruction: str | None = None,
        multi_label: bool = False,
    ) -> Scores:
        n = len(candidates)
        if self.max_candidates is not None and n > self.max_candidates:
            raise BackendError(f"openai takes at most {self.max_candidates} candidates, got {n}")
        if not len(texts) or not n:
            return Scores(np.zeros((len(texts), n)), "softmax")
        require("httpx", "llm")
        options = [c.text for c in candidates]
        bodies = [self._request(t, options, instruction or DEFAULT_INSTRUCTION) for t in texts]
        return Scores(run_sync(self._score_all(bodies, n)), "softmax")

    def supports(self, node_id: str) -> bool:
        return True

    def tokenizer(self) -> Callable[[str], int] | None:
        return None
