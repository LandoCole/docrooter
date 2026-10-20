"""OpenAI-compatible chat-completions backend: the model answers with an option letter."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from docrooter._extras import require
from docrooter.backends._async import run_sync
from docrooter.backends._secrets import redact
from docrooter.backends.base import Candidate, Scores, TokenUsage
from docrooter.backends.mlm_choice import LETTERS, build_choice_prompt
from docrooter.errors import BackendError
from docrooter.views import estimate_tokens

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "Answer with the option letter only."
DEFAULT_INSTRUCTION = "Which option best describes this document section?"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
_TOP_LOGPROBS = 20
# A capital letter standing alone: "B", " B.", "(B)", but not the "O" of "Option".
_LETTER = re.compile(r"(?<![A-Za-z])[A-Z](?![A-Za-z])")


def _first_env(*names: str) -> str | None:
    return next((os.environ[n] for n in names if os.environ.get(n)), None)


def _logprob_distribution(top_logprobs: Sequence[Any], n: int) -> np.ndarray | None:
    """Softmax over the option letters among a first token's ``top_logprobs``.

    Options the model did not list get ``-inf``. Tokens that differ only in whitespace (``"B"``
    and ``" B"``) pool their probability. Returns ``None`` when no option letter is listed.
    """
    index_of = {letter: i for i, letter in enumerate(LETTERS[:n])}
    logits = np.full(n, -np.inf)
    for entry in top_logprobs:
        if not isinstance(entry, dict):
            continue
        index = index_of.get(str(entry.get("token", "")).strip())
        logprob = entry.get("logprob")
        if index is not None and logprob is not None:
            logits[index] = np.logaddexp(logits[index], float(logprob))
    if not np.isfinite(logits).any():
        return None
    probs = np.exp(logits - logits.max())
    return probs / probs.sum()


def _reply(data: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """The first choice of a chat-completion response and its message text."""
    choice = (data.get("choices") or [{}])[0]
    if not isinstance(choice, dict):
        return {}, ""
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    return choice, content if isinstance(content, str) else ""


def _parse_answer(data: dict[str, Any], n: int) -> np.ndarray | None:
    """Probabilities over ``n`` options from one chat-completion response, or ``None``.

    Logprobs over the option letters when the server returns them in the OpenAI format;
    otherwise the first standalone capital letter of the reply, if it names an option.
    """
    choice, message = _reply(data)
    logprobs = choice.get("logprobs")
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    if isinstance(content, list) and content and isinstance(content[0], dict):
        probs = _logprob_distribution(content[0].get("top_logprobs") or [], n)
        if probs is not None:
            return probs
    match = _LETTER.search(message)
    index = LETTERS.index(match.group()) if match else n
    if index < n:
        probs = np.zeros(n)
        probs[index] = 1.0
        return probs
    return None


def _count(value: object) -> int | None:
    """``value`` when it is a token count (a non-negative int), else ``None``."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _call_usage(body: dict[str, Any], data: dict[str, Any]) -> TokenUsage:
    """Tokens one call spent: the response's ``usage`` when complete, else an estimate.

    The estimate counts the message texts with :func:`~docrooter.views.estimate_tokens`
    (no chat-template overhead) and is marked ``exact=False``.
    """
    usage = data.get("usage")
    if isinstance(usage, dict):
        prompt, completion = (
            _count(usage.get("prompt_tokens")),
            _count(usage.get("completion_tokens")),
        )
        if prompt is not None and completion is not None:
            return TokenUsage(prompt, completion, 1, True)
    prompt_estimate = sum(estimate_tokens(m["content"]) for m in body["messages"])
    return TokenUsage(prompt_estimate, estimate_tokens(_reply(data)[1]), 1, False)


class OpenAIBackend:
    """Zero-shot classification through any OpenAI-compatible chat-completions endpoint.

    Each text becomes a multiple-choice prompt with options lettered A-Z, answered with one
    letter. When the server returns logprobs the scores are a softmax over the first token's
    option letters; otherwise the first standalone capital letter in the reply gets probability
    1. Replies that name no option score a uniform distribution, with one warning per ``score``
    call. At most 26 options fit one call; more raise ``BackendError`` (a hierarchical strategy
    keeps each call small). The model cannot judge options independently, so ``multi_label`` is
    ignored and the result is a ``softmax``.

    :meth:`usage` counts the tokens of every call, from the response's ``usage``. A response
    without one (some local servers) is estimated, and the running total is then marked
    ``exact=False``.

    ``base_url`` and ``api_key`` default to ``DOCROOTER_LLM_BASE_URL`` / ``DOCROOTER_LLM_API_KEY``,
    then ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``. Without a ``base_url`` the OpenAI API is used
    and an API key is required; for other servers the key is optional. Needs the ``llm`` extra.
    """

    max_candidates: int | None = len(LETTERS)

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
        key = (api_key or _first_env("DOCROOTER_LLM_API_KEY", "OPENAI_API_KEY") or "").strip()
        self._api_key: str | None = key or None
        if self._api_key is None and self.base_url == DEFAULT_BASE_URL:
            raise BackendError(
                "no API key for the openai backend: set DOCROOTER_LLM_API_KEY or OPENAI_API_KEY "
                "(or pass api_key=), or point base_url at a server that needs none"
            )
        self._transport: Any = None  # an httpx transport; tests replace it with a MockTransport
        self._usage = TokenUsage()
        self._usage_lock = threading.Lock()  # score may run on worker threads (see run_sync)

    def usage(self) -> TokenUsage:
        """Tokens spent by every call so far (cumulative); see :class:`TokenUsage`."""
        return self._usage

    def _request(self, text: str, options: list[str], instruction: str) -> dict[str, Any]:
        prompt = build_choice_prompt(
            text,
            options,
            instruction=instruction,
            answer="",
            markers=LETTERS,
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
            raise BackendError(
                f"{self.name}: request failed: {type(exc).__name__}: {redact(exc, self._api_key)}"
            ) from None
        if response.status_code >= 400:
            # Redact before cutting, so a key that straddles the cut leaks no prefix.
            excerpt = redact(response.text, self._api_key)[:300].strip()
            raise BackendError(f"{self.name}: HTTP {response.status_code}: {excerpt}")
        try:
            data = response.json()
        except ValueError:  # JSONDecodeError; its .doc is the whole body, so do not chain it
            raise BackendError(f"{self.name}: response was not JSON") from None
        if not isinstance(data, dict):
            raise BackendError(f"{self.name}: response was not a JSON object")
        return data

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
                    data = await self._complete(client, body)
                # Counted per reply, so calls that succeed before another fails still count.
                with self._usage_lock:
                    self._usage = self._usage + _call_usage(body, data)
                return _parse_answer(data, n)

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
            raise BackendError(
                f"openai takes at most {self.max_candidates} options per call (letters A-Z), "
                f"got {n}; use a hierarchical strategy (Greedy or Beam over a taxonomy whose "
                f"nodes have at most {self.max_candidates} children) so each call is smaller"
            )
        if not len(texts) or not n:
            return Scores(np.zeros((len(texts), n)), "softmax")
        require("httpx", "llm")
        options = [c.text for c in candidates]
        bodies = [self._request(t, options, instruction or DEFAULT_INSTRUCTION) for t in texts]
        return Scores(run_sync(self._score_all(bodies, n)), "softmax")

    def close(self) -> None:
        """Nothing to release: each ``score`` call opens and closes its own client."""

    def supports(self, node_id: str) -> bool:
        return True

    def tokenizer(self) -> Callable[[str], int] | None:
        return None
