"""The teacher protocol, the shared log-probability scorer, label keys and engine specs.

A teacher labels an element by multiple-choice log-probability scoring (spec §7): one forward
pass per element, reading the next-token log-probabilities of the option letters at the answer
position. :class:`LogprobTeacher` does the prompt building, truncation, prefix caching and
softmax for every in-process engine; an engine only supplies a tokenizer (the Hugging Face
interface: ``encode``, ``decode``, ``apply_chat_template``) and a :class:`LogitRunner` that
runs token ids after a cached prefix.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from docrooter.errors import BackendError, TeacherNotAllowedError
from docrooter.teach import prompts

__all__ = [
    "COMPUTE_DTYPES",
    "JEV_REFUSAL",
    "NATIVE_DTYPE",
    "LogitRunner",
    "LogprobTeacher",
    "engine_name",
    "TeachItem",
    "TeacherEngine",
    "get_teacher",
    "label_key",
    "refuse_jev",
    "softmax",
]

NATIVE_DTYPE = "native"  # the dtype part of a name when the checkpoint's own dtype is kept
COMPUTE_DTYPES = ("float16", "bfloat16", "float32")

JEV_REFUSAL = (
    "Jev cannot be a teacher: TypeSafe MCA §2.3(b) forbids using Jev output to train, distil "
    "or calibrate another model, so no Jev output may enter teacher, training or calibration "
    "data. Use an mlx:, hf: or openai: teacher with a permissively licensed model."
)


class TeachItem(BaseModel):
    """One element to label (spec §7.1).

    - ``text``: the element text (a table: :func:`~docrooter.teach.prompts.table_text`); the
      engine cuts it to 384 tokens of the teacher's tokenizer;
    - ``context``: ``"Document type: …\\nSection: A > B\\nElement: paragraph"``
      (:func:`~docrooter.teach.prompts.context_text`);
    - ``options``: ``(tag id, "label: definition")`` pairs to choose among; unused by
      :meth:`TeacherEngine.yes_no`.
    """

    text: str
    context: str = ""
    options: list[tuple[str, str]] = Field(default_factory=list)


@runtime_checkable
class TeacherEngine(Protocol):
    """A teacher LLM.

    - :meth:`choose` returns, per item, a probability per option tag id: the softmax over the
      option letters' next-token log-probabilities. ``instruction`` and, when given,
      ``prefix_options`` (every option of the family, lettered A.. in order; each item's
      options must be among them and the answer is restricted to their letters) form the
      shared prefix, which local engines cache under ``prefix_key``. Without
      ``prefix_options`` each item's options (at most 8) are lettered A..H in its own turn.
    - :meth:`yes_no` returns P(yes) per item: the softmax over the "Yes" and "No" tokens.
    """

    name: str

    def choose(
        self,
        items: Sequence[TeachItem],
        *,
        instruction: str,
        prefix_key: str,
        prefix_options: Sequence[tuple[str, str]] | None = None,
    ) -> list[dict[str, float]]: ...

    def yes_no(self, items: Sequence[TeachItem], *, question: str) -> list[float]: ...

    def close(self) -> None: ...


class LogitRunner(Protocol):
    """Runs a causal LM for :class:`LogprobTeacher`."""

    def prefill(self, ids: Sequence[int]) -> Any:
        """Run ``ids`` and return the state (KV cache) just after them."""

    def score_next(self, state: Any, ids: Sequence[int], candidates: Sequence[int]) -> list[float]:
        """Run ``ids`` after ``state``'s prefix and return the full-vocabulary log-softmax of
        the next token after the last of them, at each of ``candidates``. Leaves ``state`` as
        it was (the cache trimmed back to the prefix)."""

    def close(self) -> None:
        """Release the model."""


Loader = Callable[[], tuple[Any, LogitRunner]]

_SENTINEL = "␞ELEMENT␞"  # where the element's turn goes in the chat template


@dataclass
class _Prefix:
    fingerprint: str
    text: str
    ids: list[int]
    tail: str  # the template after the user turn's content: end of turn, assistant opening
    state: Any = None
    boundary_checked: bool = False


def softmax(values: Sequence[float]) -> list[float]:
    """Probabilities from log-probabilities (or logits), max-shifted for stability."""
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


class LogprobTeacher:
    """A :class:`TeacherEngine` scoring with an in-process model (spec §7.2, research §2.4).

    ``loader`` returns ``(tokenizer, runner)``; it runs on the first item, not before. The
    chat template is applied with ``enable_thinking=False`` (Qwen3 hybrid models answer
    without a think block; other templates ignore it).

    Each distinct prefix (system turn) is prefilled once and kept, up to
    ``max_cached_prefixes`` of them (least recently used first out); every item then runs only
    its own turn on top and the cache is trimmed back. A prefix is keyed by ``prefix_key`` and
    its text, so a changed instruction under the same key is prefilled again, never scored on
    a stale cache. ``use_prefix_cache=False`` prefills the prefix for every item (to measure
    what the cache saves). If scoring an item raises, its prefix is dropped from the cache (a
    forward pass that failed part-way may have left the cache past the prefix), so the next
    item prefills it afresh.

    ``name`` is the teacher's identity in labels' keys (:func:`label_key`): engines put the
    engine, model, revision and compute dtype in it. A name that names Jev is refused.

    ``stats`` counts items, prefills, prefix and suffix tokens, and ``answer_mass_sum``: the
    summed full-vocabulary probability of the allowed answers (near 1 per item when the model
    answers with an option; low means it would have said something else).
    """

    def __init__(
        self,
        name: str,
        *,
        loader: Loader,
        use_prefix_cache: bool = True,
        max_cached_prefixes: int = 4,
    ) -> None:
        if max_cached_prefixes < 1:
            raise ValueError("max_cached_prefixes must be at least 1")
        refuse_jev(name)
        self.name = name
        self.use_prefix_cache = use_prefix_cache
        self.max_cached_prefixes = max_cached_prefixes
        self._loader = loader
        self._tokenizer: Any = None
        self._runner: LogitRunner | None = None
        self._prefixes: OrderedDict[str, _Prefix] = OrderedDict()
        self._single_tokens: dict[str, int] = {}
        self.stats: dict[str, float] = {
            "items": 0,
            "prefills": 0,
            "prefix_tokens": 0,
            "suffix_tokens": 0,
            "answer_mass_sum": 0.0,
        }

    # --- public ---------------------------------------------------------------------------

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
        runner = self._load()
        prefix = self._prefix(prefix_key, prompts.choose_system(instruction, prefix_options))
        candidates_by_letters: dict[tuple[str, ...], list[int]] = {}
        results = []
        for item, (tags, letters, listed) in zip(items, plans, strict=True):
            context, text = self._inputs(item)
            suffix = prompts.choose_suffix(context, text, letters, listed) + prefix.tail
            candidates = candidates_by_letters.get(letters)
            if candidates is None:
                candidates = [self._single_token(letter) for letter in letters]
                candidates_by_letters[letters] = candidates
            probs = self._score(runner, prefix, suffix, candidates)
            results.append(dict(zip(tags, probs, strict=True)))
        return results

    def yes_no(self, items: Sequence[TeachItem], *, question: str) -> list[float]:
        if not items:
            return []
        runner = self._load()
        system = prompts.yes_no_system(question)
        key = "yes_no:" + hashlib.sha1(question.encode("utf-8")).hexdigest()
        prefix = self._prefix(key, system)
        candidates = [self._first_token(prompts.YES), self._first_token(prompts.NO)]
        if candidates[0] == candidates[1]:
            raise BackendError(f"{self.name}: 'Yes' and 'No' start with the same token")
        out = []
        for item in items:
            context, text = self._inputs(item)
            suffix = prompts.yes_no_suffix(context, text) + prefix.tail
            out.append(self._score(runner, prefix, suffix, candidates)[0])
        return out

    def close(self) -> None:
        self._prefixes.clear()
        if self._runner is not None:
            self._runner.close()
        self._runner = None
        self._tokenizer = None

    # --- internals ------------------------------------------------------------------------

    def _load(self) -> LogitRunner:
        if self._runner is None:
            self._tokenizer, self._runner = self._loader()
        return self._runner

    def _encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def _decode(self, ids: Sequence[int]) -> str:
        return str(self._tokenizer.decode(list(ids)))

    def _single_token(self, text: str) -> int:
        if text not in self._single_tokens:
            ids = self._encode(text)
            if len(ids) != 1:
                raise BackendError(
                    f"{self.name}: answer {text!r} is {len(ids)} tokens for this tokenizer; "
                    "letter scoring needs one token per letter"
                )
            self._single_tokens[text] = ids[0]
        return self._single_tokens[text]

    def _first_token(self, text: str) -> int:
        ids = self._encode(text)
        if not ids:
            raise BackendError(f"{self.name}: {text!r} encodes to no tokens")
        return ids[0]

    def _inputs(self, item: TeachItem) -> tuple[str, str]:
        """Context and text, neutralised, then cut to their token limits (Review Focus 3)."""
        context = prompts.truncate_tokens(
            prompts.neutralise(item.context), prompts.MAX_CONTEXT_TOKENS, self._encode, self._decode
        )
        text = prompts.truncate_tokens(
            prompts.neutralise(item.text), prompts.MAX_TEXT_TOKENS, self._encode, self._decode
        )
        return context, text

    def _template(self, system: str) -> tuple[str, str]:
        full = self._tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": _SENTINEL}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        head, sep, tail = str(full).partition(_SENTINEL)
        if not sep or _SENTINEL in tail:
            raise BackendError(f"{self.name}: the chat template did not keep the user turn")
        return head, tail

    def _prefix(self, key: str, system: str) -> _Prefix:
        text, tail = self._template(system)
        fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cached = self._prefixes.get(key)
        if cached is not None and cached.fingerprint == fingerprint:
            self._prefixes.move_to_end(key)
            return cached
        prefix = _Prefix(fingerprint, text, self._encode(text), tail)
        if self.use_prefix_cache:
            prefix.state = self._prefill(prefix.ids)
            self._prefixes[key] = prefix
            self._prefixes.move_to_end(key)
            while len(self._prefixes) > self.max_cached_prefixes:
                self._prefixes.popitem(last=False)
        return prefix

    def _evict(self, prefix: _Prefix) -> None:
        """Drop ``prefix`` from the cache, so its key is prefilled again on next use."""
        for key in [k for k, cached in self._prefixes.items() if cached is prefix]:
            del self._prefixes[key]

    def _prefill(self, ids: list[int]) -> Any:
        assert self._runner is not None
        self.stats["prefills"] += 1
        self.stats["prefix_tokens"] += len(ids)
        return self._runner.prefill(ids)

    def _score(
        self, runner: LogitRunner, prefix: _Prefix, suffix: str, candidates: list[int]
    ) -> list[float]:
        ids = self._encode(suffix)
        if not prefix.boundary_checked:
            # the prefix and suffix are tokenised apart; that must equal tokenising them whole
            prefix.boundary_checked = True
            if self._encode(prefix.text + suffix) != prefix.ids + ids:
                warnings.warn(
                    f"{self.name}: the prompt does not split on a token boundary; scores may "
                    "differ slightly from scoring the whole prompt",
                    RuntimeWarning,
                    stacklevel=3,
                )
        state = prefix.state if self.use_prefix_cache else self._prefill(prefix.ids)
        try:
            logprobs = runner.score_next(state, ids, candidates)
        except BaseException:
            self._evict(prefix)
            raise
        self.stats["items"] += 1
        self.stats["suffix_tokens"] += len(ids)
        self.stats["answer_mass_sum"] += sum(math.exp(v) for v in logprobs)
        return softmax(logprobs)


def choice_plan(
    item: TeachItem, prefix_options: Sequence[tuple[str, str]] | None
) -> tuple[list[str], tuple[str, ...], list[tuple[str, str]] | None]:
    """``(tag ids, their letters, options to list in the element's turn or None)``, checked.

    Without ``prefix_options`` the item's options are lettered A.. (at most 8). With them,
    each option takes its family-wide letter and none is listed again.
    """
    tags = [tag for tag, _ in item.options]
    if not tags:
        raise ValueError("a TeachItem has no options to choose among")
    if len(set(tags)) != len(tags):
        raise ValueError(f"duplicate option tag ids: {tags}")
    if prefix_options is None:
        if len(tags) > prompts.MAX_ITEM_OPTIONS:
            raise ValueError(
                f"at most {prompts.MAX_ITEM_OPTIONS} options per item without prefix_options, "
                f"got {len(tags)}"
            )
        return tags, tuple(prompts.LETTERS[: len(tags)]), list(item.options)
    if len(prefix_options) > len(prompts.LETTERS):
        raise ValueError(
            f"at most {len(prompts.LETTERS)} prefix_options, got {len(prefix_options)}"
        )
    family_tags = [tag for tag, _ in prefix_options]
    if len(set(family_tags)) != len(family_tags):
        raise ValueError(f"duplicate tag ids in prefix_options: {family_tags}")
    position = {tag: i for i, tag in enumerate(family_tags)}
    missing = [tag for tag in tags if tag not in position]
    if missing:
        raise ValueError(f"options not in prefix_options: {missing}")
    return tags, tuple(prompts.LETTERS[position[tag]] for tag in tags), None


def label_key(
    *,
    doc_id: str,
    element_id: str,
    text_sha1: str,
    facet: str,
    teacher: str,
    prompt_hash: str,
    taxonomy_hash: str,
) -> str:
    """The cache key of one teacher label: sha256 over the element (document, id, text sha1),
    the facet, the teacher (engine, model, revision and compute dtype:
    :attr:`TeacherEngine.name`), the
    :func:`~docrooter.teach.prompts.prompt_hash` and the taxonomy's content hash. A cached
    label applies only when the key matches, so changing the prompt, the taxonomy or the
    teacher re-scores the element (Review Focus 2)."""
    payload = [doc_id, element_id, text_sha1, facet, teacher, prompt_hash, taxonomy_hash]
    canonical = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_JEV_PART = re.compile(r"^jev(\b|[-_.\d]|$)", re.IGNORECASE)


def _names_jev(target: str) -> bool:
    """Whether a model reference names Jev: any path part that is ``jev`` or starts ``jev-``,
    ``jev_``, ``jev.`` or ``jev<digit>``, or that starts with ``typesafe``."""
    parts = re.split(r"[/@:#\s]+", target.strip())
    return any(_JEV_PART.match(p) or p.lower().startswith("typesafe") for p in parts if p)


def engine_name(engine: str, repo: str, revision: str, dtype: str | None) -> str:
    """``<engine>:<repo>@<revision>#<dtype>`` (``#native`` when the checkpoint's dtype is
    kept), after checking ``dtype``. The dtype is part of the name, so labels scored in
    another dtype get other keys."""
    if dtype is not None and dtype not in COMPUTE_DTYPES:
        raise ValueError(f"dtype must be one of {COMPUTE_DTYPES} or None, got {dtype!r}")
    return f"{engine}:{repo}@{revision}#{dtype or NATIVE_DTYPE}"


def refuse_jev(name: str) -> None:
    """Raise :class:`~docrooter.errors.TeacherNotAllowedError` when an engine name
    (``<engine>:<model reference>``) is a Jev engine or names Jev (TypeSafe MCA §2.3(b))."""
    engine, _, target = name.strip().partition(":")
    if engine.strip().lower() == "jev" or _names_jev(target):
        raise TeacherNotAllowedError(JEV_REFUSAL)


def get_teacher(spec: str, **options: Any) -> TeacherEngine:
    """A teacher from a spec string; nothing loads until the first item.

    - ``mlx:<repo>@<revision>[#<dtype>]``: :class:`~docrooter.teach.mlx.MlxTeacher`
      (``teach-mlx``);
    - ``hf:<repo>@<revision>[#<dtype>]``: :class:`~docrooter.teach.hf.HFTeacher` (``models``);
    - ``openai:<model>``: :class:`~docrooter.teach.openai.OpenAITeacher` (``llm``).

    The revision is required: labels record it. The optional dtype (``float16``, ``bfloat16``,
    ``float32``, or ``native`` for the checkpoint's own) is the compute dtype; an engine's
    ``name`` is a spec that :func:`get_teacher` reads back to the same engine.

    ``jev:*``, or any spec naming Jev, raises :class:`~docrooter.errors.TeacherNotAllowedError`
    (TypeSafe MCA §2.3(b)); the engines refuse it at construction too. ``options`` go to the
    engine's constructor.
    """
    refuse_jev(spec)
    engine, sep, target = spec.strip().partition(":")
    engine = engine.strip().lower()
    target = target.strip()
    if not sep or not target:
        raise ValueError(f"bad teacher spec {spec!r}: expected mlx:, hf: or openai:<model>")
    if engine in ("mlx", "hf"):
        target, hashmark, dtype = target.partition("#")
        if hashmark:
            if not dtype or "dtype" in options:
                raise ValueError(f"bad teacher spec {spec!r}: give one dtype after '#'")
            options["dtype"] = None if dtype == NATIVE_DTYPE else dtype
        repo, at, revision = target.rpartition("@")
        if not at or not repo or not revision:
            raise ValueError(f"bad teacher spec {spec!r}: pin a revision, {engine}:<repo>@<rev>")
        if engine == "mlx":
            from docrooter.teach.mlx import MlxTeacher

            return MlxTeacher(repo, revision, **options)
        from docrooter.teach.hf import HFTeacher

        return HFTeacher(repo, revision, **options)
    if engine == "openai":
        from docrooter.teach.openai import OpenAITeacher

        return OpenAITeacher(target, **options)
    raise ValueError(f"unknown teacher engine {engine!r} in {spec!r}: expected mlx, hf or openai")
