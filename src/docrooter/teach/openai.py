"""The OpenAI-compatible teacher engine (spec §7.2): any chat-completions endpoint that returns
``top_logprobs``.

The same prompts as the local engines, sent as a system and a user message; the answer is the
first generated token's ``top_logprobs``. Tokens differing only in surrounding whitespace pool
their probability. An allowed answer missing from the top-k gets the least log-probability
seen there minus 1 nat (it ranked below everything listed), then the allowed answers are
softmaxed. Element text is cut to about 4 characters a token, as the server's tokenizer is not
at hand. There is no prefix cache here (``prefix_key`` is accepted and unused); servers may
cache prompts themselves.

``base_url`` and ``api_key`` default to ``DOCROOTER_LLM_BASE_URL`` / ``DOCROOTER_LLM_API_KEY``,
then ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``, as for the openai classification backend. The
user configures the endpoint and is responsible for the provider's terms. Needs the ``llm``
extra.
"""

from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

from docrooter._extras import require
from docrooter.backends._async import run_sync
from docrooter.backends._secrets import redact
from docrooter.errors import BackendError, TeacherNotAllowedError
from docrooter.teach import prompts
from docrooter.teach.base import JEV_REFUSAL, TeachItem, choice_plan, softmax

__all__ = ["OpenAITeacher"]

DEFAULT_BASE_URL = "https://api.openai.com/v1"


def _first_env(*names: str) -> str | None:
    return next((os.environ[n] for n in names if os.environ.get(n)), None)


def refuse_jev_endpoint(model: str, base_url: str) -> None:
    """Raise :class:`~docrooter.errors.TeacherNotAllowedError` for a model name containing
    ``jev`` (any case) or an endpoint whose host ends in ``typesafe.ai``: Jev output may never
    become teacher data (TypeSafe MCA §2.3(b))."""
    host = (urlsplit(base_url).hostname or "").lower().rstrip(".")
    if "jev" in model.lower() or host.endswith("typesafe.ai"):
        raise TeacherNotAllowedError(JEV_REFUSAL)


def answer_logprobs(top_logprobs: Sequence[dict[str, Any]], answers: Sequence[str]) -> list[float]:
    """Log-probabilities of ``answers`` from one position's ``top_logprobs``; answers missing
    there get the least log-probability seen minus 1 nat."""
    seen: list[float] = []
    found: dict[str, float] = {}
    for entry in top_logprobs:
        logprob = entry.get("logprob")
        if logprob is None:
            continue
        seen.append(float(logprob))
        token = str(entry.get("token", "")).strip()
        if token in answers:
            previous = found.get(token)
            found[token] = (
                float(logprob) if previous is None else float(_logaddexp(previous, logprob))
            )
    if not seen:
        raise BackendError("the endpoint returned no top_logprobs; it cannot be a teacher")
    floor = min(seen) - 1.0
    return [found.get(answer, floor) for answer in answers]


def _logaddexp(a: float, b: float) -> float:
    top = max(a, b)
    return top + math.log(math.exp(a - top) + math.exp(b - top))


class OpenAITeacher:
    """A teacher behind an OpenAI-compatible chat-completions endpoint (see the module).

    Refuses a model name containing ``jev`` or a ``typesafe.ai`` endpoint
    (:func:`refuse_jev_endpoint`)."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60,
        concurrency: int = 8,
        top_logprobs: int = 20,
    ) -> None:
        self.base_url = (
            base_url or _first_env("DOCROOTER_LLM_BASE_URL", "OPENAI_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        refuse_jev_endpoint(model, self.base_url)
        self.model = model
        self.name = f"openai:{model}"
        self.timeout = timeout
        self.concurrency = max(1, concurrency)
        self.top_logprobs = top_logprobs
        key = (api_key or _first_env("DOCROOTER_LLM_API_KEY", "OPENAI_API_KEY") or "").strip()
        self._api_key: str | None = key or None
        if self._api_key is None and self.base_url == DEFAULT_BASE_URL:
            raise BackendError(
                "no API key for the openai teacher: set DOCROOTER_LLM_API_KEY or OPENAI_API_KEY "
                "(or pass api_key=), or point base_url at a server that needs none"
            )
        self._transport: Any = None  # an httpx transport; tests replace it with a MockTransport

    def choose(
        self,
        items: Sequence[TeachItem],
        *,
        instruction: str,
        prefix_key: str,
        prefix_options: Sequence[tuple[str, str]] | None = None,
    ) -> list[dict[str, float]]:
        plans = [choice_plan(item, prefix_options) for item in items]
        if not plans:
            return []
        system = prompts.choose_system(instruction, prefix_options)
        bodies = []
        for item, (_, letters, listed) in zip(items, plans, strict=True):
            context, text = self._inputs(item)
            bodies.append(self._body(system, prompts.choose_suffix(context, text, letters, listed)))
        replies = run_sync(self._complete_all(bodies))
        return [
            dict(zip(tags, softmax(answer_logprobs(top, letters)), strict=True))
            for (tags, letters, _), top in zip(plans, replies, strict=True)
        ]

    def yes_no(self, items: Sequence[TeachItem], *, question: str) -> list[float]:
        if not items:
            return []
        system = prompts.yes_no_system(question)
        bodies = []
        for item in items:
            context, text = self._inputs(item)
            bodies.append(self._body(system, prompts.yes_no_suffix(context, text)))
        replies = run_sync(self._complete_all(bodies))
        return [softmax(answer_logprobs(top, (prompts.YES, prompts.NO)))[0] for top in replies]

    def close(self) -> None:
        pass

    def _inputs(self, item: TeachItem) -> tuple[str, str]:
        context = prompts.truncate_chars(
            prompts.neutralise(item.context), prompts.MAX_CONTEXT_TOKENS
        )
        text = prompts.truncate_chars(prompts.neutralise(item.text), prompts.MAX_TEXT_TOKENS)
        return context, text

    def _body(self, system: str, user: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            "max_tokens": 1,
            "temperature": 0,
        }

    async def _complete_all(self, bodies: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        httpx = require("httpx", "llm")
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        semaphore = asyncio.Semaphore(self.concurrency)
        async with httpx.AsyncClient(
            base_url=self.base_url, headers=headers, timeout=self.timeout, transport=self._transport
        ) as client:

            async def one(body: dict[str, Any]) -> list[dict[str, Any]]:
                async with semaphore:
                    return self._top_logprobs(await self._post(client, body))

            return list(await asyncio.gather(*(one(b) for b in bodies)))

    async def _post(self, client: Any, body: dict[str, Any]) -> dict[str, Any]:
        httpx = require("httpx", "llm")
        try:
            response = await client.post("/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise BackendError(
                f"{self.name}: request failed: {type(exc).__name__}: {redact(exc, self._api_key)}"
            ) from None
        if response.status_code >= 400:
            excerpt = redact(response.text, self._api_key)[:300].strip()
            raise BackendError(f"{self.name}: HTTP {response.status_code}: {excerpt}")
        try:
            return dict(response.json())
        except ValueError:
            raise BackendError(f"{self.name}: response was not JSON") from None

    def _top_logprobs(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        choice = (data.get("choices") or [{}])[0]
        content = (choice.get("logprobs") or {}).get("content") or []
        top = content[0].get("top_logprobs") if content else None
        if not top:
            raise BackendError(
                f"{self.name}: the reply has no top_logprobs; it cannot be a teacher"
            )
        return list(top)
