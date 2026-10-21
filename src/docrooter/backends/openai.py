"""OpenAI-compatible chat-completions backend: the model answers with an option letter."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
from collections.abc import Callable, Sequence
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from docrooter._extras import require
from docrooter.backends._async import run_sync
from docrooter.backends._secrets import redact
from docrooter.backends.base import (
    TOKEN_RATIO_MIN,
    Candidate,
    Scores,
    TokenUsage,
    clamp_token_ratio,
)
from docrooter.backends.mlm_choice import LETTERS, build_choice_prompt
from docrooter.errors import BackendError
from docrooter.views import estimate_tokens

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "Answer with the option letter only."
DEFAULT_INSTRUCTION = "Which option best describes this document section?"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
# How many alternatives to ask for at the first token. mlx_lm.server rejects more than 11 and
# OpenAI more than 20; options outside the top ones get no probability anyway.
_TOP_LOGPROBS = 10
# A reply that is an answer: a bare letter ("B", " (C) "), or one followed by "." or ")"
# ("B. Litigation", "(C) Leases"). Prose such as "I cannot", "I'm" or "A contract" is not.
_ANSWER = re.compile(r"\(?([A-Z])(?:[.)]|\s*$)")
# Word-start markers of byte-level BPE ("Ġ") and SentencePiece ("▁") vocabularies.
_SPACE_MARKERS = str.maketrans({"\u0120": " ", "\u2581": " "})

CallRecord = dict[str, Any]


def prompt_hash(messages: list[dict[str, str]]) -> str:
    """SHA-256 hex digest of the exact chat messages sent (canonical JSON, UTF-8)."""
    canonical = json.dumps(messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def choice_messages(
    text: str, options: Sequence[str], instruction: str | None = None
) -> list[dict[str, str]]:
    """The chat messages one per-item request sends: the system prompt, then the question, the
    text and the lettered options. ``OpenAIBackend`` sends exactly these, and token estimates
    for backends that report no usage are made from them. Past 26 options (which the backend
    refuses) the options are numbered, for estimates only."""
    markers: Sequence[str] = (
        LETTERS if len(options) <= len(LETTERS) else [str(i + 1) for i in range(len(options))]
    )
    prompt = build_choice_prompt(
        text,
        list(options),
        instruction=instruction or DEFAULT_INSTRUCTION,
        answer="",
        markers=markers,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _letter_of(token: object) -> str:
    """A logprob token as a bare letter candidate: space markers, spaces and "(", ".", ")" off."""
    return str(token).translate(_SPACE_MARKERS).strip().strip("(.)")


def _env_number(name: str, default: float) -> float:
    """A positive number from the environment variable ``name``, else ``default``."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise BackendError(f"{name} must be a positive number, got {raw!r}") from None
    if value <= 0:
        raise BackendError(f"{name} must be a positive number, got {raw!r}")
    return value


def _first_env(*names: str) -> str | None:
    return next((os.environ[n] for n in names if os.environ.get(n)), None)


def _env_object(name: str) -> dict[str, Any]:
    """A JSON object from the environment variable ``name``, else ``{}``."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except ValueError:
        raise BackendError(f"{name} must be a JSON object, got {raw[:80]!r}") from None
    if not isinstance(value, dict):
        raise BackendError(f"{name} must be a JSON object, got {raw[:80]!r}")
    return value


def _logprob_distribution(top_logprobs: Sequence[Any], n: int) -> np.ndarray | None:
    """Softmax over the option letters among a first token's ``top_logprobs``.

    Options the model did not list get ``-inf``. Tokens that differ only in whitespace or a
    tokenizer's space marker (``"B"``, ``" B"``, ``"ĠB"``, ``"▁B"``) pool their probability.
    Returns ``None`` when no option letter is listed.
    """
    index_of = {letter: i for i, letter in enumerate(LETTERS[:n])}
    logits = np.full(n, -np.inf)
    for entry in top_logprobs:
        if not isinstance(entry, dict):
            continue
        index = index_of.get(_letter_of(entry.get("token", "")))
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


def _parse_answer(data: dict[str, Any], n: int) -> tuple[np.ndarray | None, str]:
    """Probabilities over ``n`` options from one chat-completion response, and how they were
    read: ``"logprobs"``, ``"letter"`` or ``"none"`` (``None`` probabilities).

    Logprobs over the option letters when the server returns them in the OpenAI format;
    otherwise the reply's letter, when the reply is an answer (see ``_ANSWER``) naming an option
    (a one-hot, not a probability).
    """
    choice, message = _reply(data)
    logprobs = choice.get("logprobs")
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    if isinstance(content, list) and content and isinstance(content[0], dict):
        probs = _logprob_distribution(content[0].get("top_logprobs") or [], n)
        if probs is not None:
            return probs, "logprobs"
    match = _ANSWER.match(message.strip())
    index = LETTERS.index(match.group(1)) if match else n
    if index < n:
        probs = np.zeros(n)
        probs[index] = 1.0
        return probs, "letter"
    return None, "none"


def parse_json_reply(text: str) -> dict[str, Any] | None:
    """The JSON object in a reply: the whole text, else the span from its first ``{`` to its
    last ``}`` (a model that wraps the object in prose or a code fence). ``None`` when there is
    no object."""
    candidates = [text.strip()]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _count(value: object) -> int | None:
    """``value`` when it is a token count (a non-negative int), else ``None``."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _messages_estimate(body: dict[str, Any]) -> int:
    """The estimated prompt tokens of the messages in ``body`` (no chat-template overhead)."""
    return sum(estimate_tokens(m["content"]) for m in body["messages"])


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
    return TokenUsage(_messages_estimate(body), estimate_tokens(_reply(data)[1]), 1, False)


class OpenAIBackend:
    """Zero-shot classification through any OpenAI-compatible chat-completions endpoint.

    Each text becomes a multiple-choice prompt with options lettered A-Z, answered with one
    letter.
    - **With logprobs,** the scores are a softmax over the option letters among the first
      token's top 10 alternatives.
    - **Without them,** a reply that is a bare letter (``"B"``, ``"B."``, ``"(C)"``) gets
      probability 1 (a one-hot: the call record's ``answer`` is ``"letter"``, so a caller can
      tell it from a probability).
    - **A reply that names no option** scores a uniform distribution, with one warning per
      ``score`` call (``answer`` ``"none"``; :func:`docrooter.classify` flags it ``unparsed``
      rather than taking the first option).

    At most 26 options fit one call; more raise ``BackendError`` (a hierarchical strategy
    keeps each call small). The model cannot judge options independently, so ``multi_label`` is
    ignored and the result is a ``softmax``.

    **Token accounting.**
    - :meth:`usage` is cumulative over the backend's life. Each call's counts come from the
      response's ``usage``. A response without one is estimated and counted in
      ``estimated_calls``, so ``after - before`` gives a stage's exact or estimated usage.
    - ``on_call``, when set, is called once per request, after the reply arrives. It may run on a
      worker thread, one call at a time within a ``score`` call. Its record dict holds:
      - ``index``: the text's position in that ``score`` call;
      - ``prompt_hash``: :func:`prompt_hash` of the exact messages sent;
      - ``prompt_tokens``, ``completion_tokens`` and ``exact``;
      - ``options``, ``model`` and ``endpoint_host``;
      - ``answer`` (``score`` calls only): ``"logprobs"``, ``"letter"`` or ``"none"``.
      ``on_call`` is one attribute of the backend: a caller that needs tokens per section (the
      v1 ``classify`` flow) or one log row per decision sets it around its own ``score`` call
      and joins the records to its texts by ``index``. Two callers that share one backend
      concurrently would replace each other's sink, so run them one at a time (or give each
      its own backend); within one caller, records need no usage deltas.

    **Token calibration.** :meth:`token_ratio` is the exact ``prompt_tokens`` of every call that
    reported them over the estimate of the same messages, as a running, token-weighted mean
    (the sum of exact tokens over the sum of estimates), clamped to [1.0, 3.0]; 1.0 until the
    first exact call. :func:`docrooter.classify` divides its call budgets by it, since the
    word-based estimate undercounts dense text (tables, numbers) for a real tokenizer.

    ``timeout`` (seconds, default 60) and ``concurrency`` (requests in flight, default 8)
    default to ``DOCROOTER_LLM_TIMEOUT`` and ``DOCROOTER_LLM_CONCURRENCY``; a local server that
    answers one request at a time wants a long timeout and a concurrency of 1.

    ``base_url`` and ``api_key`` default to ``DOCROOTER_LLM_BASE_URL`` / ``DOCROOTER_LLM_API_KEY``,
    then ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``. Without a ``base_url`` the OpenAI API is used
    and an API key is required; for other servers the key is optional. Needs the ``llm`` extra.

    ``extra_body`` (default: the JSON object in ``DOCROOTER_LLM_EXTRA_BODY``) adds
    provider-specific fields to every request, such as ``{"thinking": {"type": "disabled"}}``
    for a Claude model that would otherwise spend completion tokens thinking; it never
    replaces a field the request sets itself. A server that refuses ``response_format`` or
    ``temperature`` is asked again without it.
    """

    max_candidates: int | None = len(LETTERS)

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        concurrency: int | None = None,
        revision: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        if revision is not None:
            raise BackendError("the openai backend does not support model revisions")
        self.model_id = model
        self.extra_body = dict(
            extra_body if extra_body is not None else _env_object("DOCROOTER_LLM_EXTRA_BODY")
        )
        self.name = f"openai:{model}"
        self.temperature = 1.0
        self.timeout = float(
            timeout if timeout is not None else _env_number("DOCROOTER_LLM_TIMEOUT", 60.0)
        )
        self.concurrency = max(
            1,
            int(
                concurrency
                if concurrency is not None
                else _env_number("DOCROOTER_LLM_CONCURRENCY", 8)
            ),
        )
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
        self.on_call: Callable[[CallRecord], None] | None = None
        self.endpoint_host = urlsplit(self.base_url).hostname or ""
        self._json_mode = True  # off once the server refuses response_format
        self._send_temperature = True  # off once the server refuses temperature
        self._exact_prompt = 0  # exact prompt tokens of the calls that reported usage
        self._estimated_prompt = 0  # the estimates of those same calls' messages

    def usage(self) -> TokenUsage:
        """Tokens spent by every call so far (cumulative); see :class:`TokenUsage`."""
        return self._usage

    def token_ratio(self) -> float:
        """Exact prompt tokens per estimated token over the calls that reported ``usage`` so
        far, clamped to [1.0, 3.0]; 1.0 before the first such call."""
        with self._usage_lock:
            exact, estimated = self._exact_prompt, self._estimated_prompt
        if not estimated:
            return TOKEN_RATIO_MIN
        return clamp_token_ratio(exact / estimated)

    def secret_values(self) -> tuple[str, ...]:
        """Values a log must scrub: this backend's API key, if it has one."""
        return (self._api_key,) if self._api_key else ()

    def request_messages(
        self, text: str, options: list[str], instruction: str | None = None
    ) -> list[dict[str, str]]:
        """The chat messages one ``score`` request sends for ``text`` (hash them with
        :func:`prompt_hash`): :func:`choice_messages`."""
        return choice_messages(text, options, instruction)

    def _request(self, text: str, options: list[str], instruction: str) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "messages": self.request_messages(text, options, instruction),
            "logprobs": True,
            "top_logprobs": _TOP_LOGPROBS,
            "max_tokens": 3,
            "temperature": 0,
        }

    async def _complete(self, client: Any, body: dict[str, Any]) -> dict[str, Any]:
        """Send ``body`` and return the reply. A server that refuses ``temperature`` (an HTTP 400
        naming it, as some hosted reasoning models do) is asked again without it, and, once that
        works, never again with it; ``body`` is changed in place to match what was sent."""
        for key, value in self.extra_body.items():
            body.setdefault(key, value)
        if not self._send_temperature:
            body.pop("temperature", None)
        try:
            return await self._post(client, body)
        except BackendError as exc:
            message = str(exc)
            if "temperature" not in body or "HTTP 400" not in message:
                raise
            if "temperature" not in message:
                raise
            del body["temperature"]
            data = await self._post(client, body)
            self._send_temperature = False  # the retry worked: never send it again
            return data

    async def _post(self, client: Any, body: dict[str, Any]) -> dict[str, Any]:
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

    def _record_call(
        self,
        index: int,
        body: dict[str, Any],
        data: dict[str, Any],
        options: list[str],
        answer: str | None = None,
    ) -> None:
        """Add one reply's tokens to :meth:`usage` and hand its record to ``on_call``."""
        call = _call_usage(body, data)
        estimate = _messages_estimate(body)
        with self._usage_lock:
            self._usage = self._usage + call
            if call.exact and estimate > 0:
                self._exact_prompt += call.prompt
                self._estimated_prompt += estimate
        if self.on_call is not None:
            record: CallRecord = {
                "index": index,
                "prompt_hash": prompt_hash(body["messages"]),
                "prompt_tokens": call.prompt,
                "completion_tokens": call.completion,
                "exact": call.exact,
                "options": list(options),
                "model": self.model_id,
                "endpoint_host": self.endpoint_host,
            }
            if answer is not None:
                record["answer"] = answer
            self.on_call(record)

    async def _score_all(self, bodies: list[dict[str, Any]], options: list[str]) -> np.ndarray:
        n = len(options)
        httpx = require("httpx", "llm")
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        semaphore = asyncio.Semaphore(self.concurrency)

        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
            transport=self._transport,
        ) as client:

            async def one(index: int, body: dict[str, Any]) -> np.ndarray | None:
                async with semaphore:
                    data = await self._complete(client, body)
                probs, answer = _parse_answer(data, n)
                # Counted per reply, so calls that succeed before another fails still count.
                self._record_call(index, body, data, options, answer)
                return probs

            answers = await asyncio.gather(*(one(i, b) for i, b in enumerate(bodies)))

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
        return Scores(run_sync(self._score_all(bodies, options)), "softmax")

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        options: Sequence[str] = (),
    ) -> dict[str, Any] | None:
        """Send ``messages`` as one chat call and return the JSON object of the reply (``None``
        when it holds none). For batched calls (:mod:`docrooter.batch`).

        Asks for ``response_format={"type": "json_object"}``; a server that refuses it (an
        HTTP 400 naming ``response_format`` or JSON) is asked again without it, and, once that
        works, never again with it. The reply is parsed
        tolerantly (:func:`parse_json_reply`). The call counts in :meth:`usage` and is reported
        to ``on_call`` (``index`` 0, ``options`` as given) like a ``score`` call.
        """
        require("httpx", "llm")
        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if self._json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            data = run_sync(self._chat(body))
        except BackendError as exc:
            # only a refusal of JSON mode itself, not any 400 (a context limit, say)
            message = str(exc)
            if "response_format" not in body or "HTTP 400" not in message:
                raise
            if "response_format" not in message and "json" not in message.lower():
                raise
            del body["response_format"]
            data = run_sync(self._chat(body))
            self._json_mode = False  # the retry worked: never ask for JSON mode again
        self._record_call(0, body, data, list(options))
        return parse_json_reply(_reply(data)[1])

    async def _chat(self, body: dict[str, Any]) -> dict[str, Any]:
        httpx = require("httpx", "llm")
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
            transport=self._transport,
        ) as client:
            return await self._complete(client, body)

    def close(self) -> None:
        """Nothing to release: each ``score`` call opens and closes its own client."""

    def supports(self, node_id: str) -> bool:
        return True

    def tokenizer(self) -> Callable[[str], int] | None:
        return None
