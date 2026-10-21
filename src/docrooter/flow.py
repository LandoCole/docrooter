"""The v1 headline API: :func:`classify` a document's sections and type at a context level.

Two flows (spec G3):

- **sections_first** (the default) tags the document's level-1 and level-2 sections against the
  shared section catalogue (``builtin_taxonomy("section_catalogue")``), group first, then leaf
  (``Greedy``). It matches the doc-type section rules (:func:`~docrooter.rules.match_rules`)
  against the tags, then decides the type:

  - **by rules**, with no document-level call, when a single candidate's supporting tags are all
    at least ``decide_conf``;
  - **by the model** among the candidates only, plus the near misses when there is no candidate
    or a single undecided one (flagged ``near_miss``), plus, when one type is still alone, its
    doc-type siblings that no ``excludes`` hit argues against (flagged ``low_support``);
  - **by fallback** among every doc type when fewer than two types are left to offer; with none
    at all it is flagged ``rules_unmet``.

  When every candidate hit an ``excludes`` rule, the waived candidates are offered (flagged
  ``excludes_waived``). A document with no level-1 or level-2 section to tag runs the
  ``doc_first`` document step at the same document level, flagged ``no_sections``.
- **doc_first** classifies the type among every doc type from the document view, then labels
  the sections with that type's section taxonomy (as :func:`~docrooter.normalize_sections`
  does).

**Levels.** ``level`` (a rung of :mod:`docrooter.ladder`) sets both the section and the document
view; ``section_level`` and ``doc_level`` override one each. With ``cascade=True``, every section
tag or document decision whose confidence is below ``threshold`` is asked again one rung up,
until ``full``; only those items are asked again, every attempt's tokens count, and the last
attempt's answer stands, even when an earlier rung's was more confident. A section with no
title shows its lead even at ``headings`` and ``terms`` (within the rung's budget), so it is
never sent as an empty path.

**Tokens.** Every backend call is metered. A backend with a per-call record sink (``on_call``,
the OpenAI-compatible backend) gives each call's exact tokens from the API's ``usage``. One that
only reports a cumulative ``usage()`` has each call's change split over its texts. Any other
backend (keyword, embedding, NLI) is charged what an LLM would be sent, marked estimated: the
messages the OpenAI-compatible backend would build for the same call (system prompt, question,
text and options per item; :mod:`docrooter.batch` prompts when batched), each estimated with
:func:`~docrooter.views.estimate_tokens`, as the backend estimates a call without ``usage``.
``Classification.tokens`` is exact only when every call was.

**Calls.** ``calls="batched"`` (the default for a backend that can, the OpenAI-compatible one) asks
a few chat calls per stage, each option listed once (:mod:`docrooter.batch`). Sections are packed
into calls bounded by size as well as count: at most 40 views, and a prompt (the instructions, the
options and the views) of at most ``max_call_tokens`` (16,000 by default, packed to an estimated
85% of it); a view too large for any call is cut head-first, flagged ``truncated`` and sent alone.
Per-item calls are bounded the same way, and the document view is budgeted to fit beside the
document call's own instructions and options (flag ``truncated``, ``doc_truncated``). These bounds
are on estimated tokens: the measured prompt is expected, not guaranteed, to stay within the cap.
**Calibration.** Every budget (the packed calls, the per-item views and the document view) is
divided by the backend's ``token_ratio()``, the exact prompt tokens it has spent per estimated
token (:func:`~docrooter.backends.base.token_ratio_of`: 1.0 for a backend that is not an LLM and
until an LLM's first call reports ``usage``, at most 3.0), read again before each batched call. The
word-based estimate undercounts dense text (tables, numbers), so after the first exact call the
next calls are packed to fit the cap in exact tokens; the first call of a run may still overshoot.
Each :class:`CallRecord` keeps the ``token_ratio`` its call was budgeted with.
The sections stage takes two steps: one call sorts every section into a catalogue group (the groups
listed with a one-line gloss), then one call per group that received sections chooses among that
group's leaves only. ``calls="batched_flat"`` lists every leaf in one call instead (kept for the
benchmark comparison). The document stage is one call among the offered types. A batched answer has
confidence 1.0 (``confidence_source="batched"``); an id that is not an option is asked again once,
in one call for the failures only, then tagged ``general.other`` (or the group's ``other`` leaf)
with confidence 0 and flagged ``unparsed``. ``calls="per_item"`` asks one item at a time, with
probabilities; ``cascade`` needs it. A backend that cannot batch always asks per item, but its
token estimate follows the ``calls`` pattern (two-step batched unless told otherwise), since that
is what an LLM would be sent. ``Classification.tokens_split`` estimates how much of the prompt was
options, content and instructions.

**The rule gate without probabilities.** Batched tags all carry confidence 1.0, so the
``decide_conf`` gate cannot tell strong evidence from weak. Neither can per-item answers from an
endpoint that returns no logprobs: the reply's letter is read as the answer, and the tag's
confidence is ``None`` (``confidence_source="text"``); such a tag counts for the rules, is never
asked again by ``cascade``, and a document answer read this way has no confidence either. With
batched calls, or when any supporting tag is a text answer, a lone candidate is decided by rules
only when at least two tagged sections, in total, support its required groups. With fewer it
takes the ``low_support`` path: the candidate, its near misses and, when it is still alone, its
doc-type siblings are offered to the model. A per-item reply that names no option is never
read as the first option: the section gets the ``other`` leaf below the last answered step
(or ``general.other``) at confidence 0, the document no type, and the result is flagged
``unparsed``.

**Results log.** ``log=path`` appends one :class:`~docrooter.results_log.ResultsLog` row per
backend decision (each section step and each document call), with the level, the view hash, the
options, the probabilities, the tokens and, from the call record, the prompt hash. A batched
call's row has ``step`` ``batch_group``, ``batch_leaf``, ``batch`` (flat) or ``document``, its
``label`` maps each section id to its answer (the doc type id for the document), ``probs`` is
``{}`` since a batched answer has no probabilities, and ``confidence_source`` is ``"batched"``
(``"probability"`` on per-item rows). Each row's tokens are those of its own call: for a backend
that reports no usage, the per-item estimate of that call, while ``Classification.tokens``
follows ``tokens_pattern``, so such a log need not sum to the result.
"""

from __future__ import annotations

import functools
import os
import string
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from docrooter import events
from docrooter.backends.base import (
    Backend,
    Candidate,
    Scores,
    TokenUsage,
    candidate_text,
    close_backend,
    to_probs,
    token_ratio_of,
    usage_of,
)
from docrooter.batch import (
    DOC_TYPE_KEY,
    MAX_CALL_TOKENS,
    BatchPrompt,
    call_budget,
    cut_to_tokens,
    document_prompt,
    group_leaves,
    group_prompt,
    messages_tokens,
    pack,
    reply_tokens,
    section_prompt,
    section_reply_max_tokens,
    valid_id,
)
from docrooter.classify.classifier import DEFAULT_BACKEND, Classifier
from docrooter.classify.documents import DEFAULT_INSTRUCTION as DOC_INSTRUCTION
from docrooter.classify.documents import _has_content, classify_document
from docrooter.classify.engine import backend_temperature, score_batch
from docrooter.classify.normalize import (
    CUSTOM_TAXONOMY,
    DOC_TYPES,
    _section_taxonomy_name,
    section_classifier,
)
from docrooter.classify.results import DocTypeResult, PathResult
from docrooter.classify.strategies import Flat, Greedy
from docrooter.doc_view import DocumentView
from docrooter.document import Document, Section
from docrooter.ladder import (
    LEVELS,
    Level,
    check_level_and_view,
    document_view,
    next_level,
    section_view,
)
from docrooter.results_log import ResultsLog, secrets_of, text_hash
from docrooter.rules import (
    DocTypeRules,
    covers,
    doc_type_rules,
    match_rules,
    unknown_section_ids,
)
from docrooter.taxonomies import builtin_taxonomy
from docrooter.taxonomy import Taxonomy
from docrooter.views import SectionView, estimate_tokens

Flow = Literal["sections_first", "doc_first"]
FLOWS: tuple[str, ...] = ("sections_first", "doc_first")
DecidedBy = Literal["rules", "model", "fallback"]
Flag = Literal[
    "no_sections",
    "rules_unmet",
    "near_miss",
    "low_support",
    "excludes_waived",
    "truncated",
    "unparsed",
]
Step = Literal["group", "leaf", "batch", "batch_group", "batch_leaf", "document"]
Calls = Literal["batched", "batched_flat", "per_item"]
ConfidenceSource = Literal["probability", "batched", "text"]
CALLS: tuple[str, ...] = ("batched", "batched_flat", "per_item")
BATCHED_SUPPORT = 2
"""Tagged sections a lone candidate's required groups need, with batched calls, for the rules
to decide it alone."""
UNPARSED_LABEL = "general.other"
"""The tag of a section whose batched answer named no option twice."""
CATALOGUE = "section_catalogue"
"""The built-in section catalogue that ``sections_first`` tags against."""
SECTION_LEVELS = (1, 2)
"""The heading levels whose sections are tagged."""
SECTION_STAGE = "sections"
DOCUMENT_STAGE = "document"
_BODY_FIELDS = frozenset({"keyphrases", "lead", "text"})
_LETTERS = string.ascii_uppercase


class SectionTag(BaseModel):
    """One tagged section: its ``label`` (a node id of the section taxonomy, ``None`` when nothing
    was assigned) with the label's ``confidence``, the ``level_used`` for the final answer and
    the ``tokens`` of every attempt. ``confidence_source`` says what the confidence is: the path
    probability of a per-item answer (``"probability"``); 1.0 for a valid batched answer and 0
    for an unparsed one (``"batched"``); or, for a per-item answer read from the reply's letter
    because the endpoint returned no logprobs (``"text"``), ``None``, since a bare letter is no
    probability, and 0 for a reply that named no option."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    level: int
    label: str | None
    confidence: float | None = 0.0
    level_used: Level
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    confidence_source: ConfidenceSource = "probability"


class CallRecord(BaseModel):
    """One backend call (or, for a backend that cannot batch, one estimated call): the
    ``level`` of its views, the ``prompt_tokens`` and ``completion_tokens`` it spent (``exact``
    when from the API's usage), whether a view it carried was ``truncated``, how many
    options it offered (``options_count``: doc types, groups or leaves) and the
    ``token_ratio`` its budget was divided by (the backend's exact-to-estimated prompt token
    ratio when the call was packed or cut; 1.0 when uncalibrated)."""

    model_config = ConfigDict(extra="forbid")

    level: Level
    prompt_tokens: int
    completion_tokens: int
    exact: bool
    truncated: bool
    options_count: int
    token_ratio: float = 1.0


class Classification(BaseModel):
    """What :func:`classify` decided, how, and at what token cost.

    - ``doc_type`` (a doc-type leaf id, ``None`` when nothing could be classified) with its
      ``doc_type_confidence`` and ``doc_type_probs`` (the most probable types offered, most
      probable first; empty when the rules decided). A type decided by rules gets the lowest
      confidence among its supporting section tags. Without probabilities (a per-item answer
      read from the reply's letter, or rules decided on such tags) the confidence is ``None``
      and the probs are empty.
    - ``decided_by``: ``"rules"`` (no document-level call), ``"model"`` (the model chose among
      the narrowed ``candidates``, or among every type in ``doc_first``) or ``"fallback"`` (the
      model chose among every type: nothing was left to offer, a lone type had no siblings, or
      there were no sections; or, for a document with no content, nothing could be sent).
    - ``candidates``: the doc types offered, in order (most ``typical`` matches first): the
      rules' candidates, near misses when they were offered, and a lone type's siblings
      (``low_support``); empty in ``doc_first``.
    - ``flags``: ``no_sections``, ``rules_unmet``, ``near_miss``, ``low_support`` (a lone
      undecided candidate or near miss, offered with its doc-type siblings), ``excludes_waived``
      ``truncated`` (a section or document view was cut to its budget) and ``unparsed`` (an
      answer named no option: a batched one twice, or a per-item one; the item gets no type,
      or the ``other`` section leaf at confidence 0, never the first option).
    - ``document_calls``: one :class:`CallRecord` per document-stage attempt (cascade attempts
      included), in order; the last made the decision. ``doc_truncated``: whether the deciding
      document view was cut. ``section_calls``: one :class:`CallRecord` per sections-stage
      call, in order (re-asks and cascade attempts included); with a backend that cannot
      batch, the calls of ``tokens_pattern`` it is estimated as.
    - ``sections``: the tags, in the order tagged (level 1 first, then level 2, each in reading
      order), against ``section_taxonomy`` (``"section_catalogue"`` in ``sections_first``).
    - ``tokens``: everything spent, ``stage_tokens["sections"] + stage_tokens["document"]``,
      each the sum of its stage's calls. A section tag's ``tokens`` are its share of the calls
      that carried it: exact only for a call of its own (per item); a share of a call that
      carried several sections is an estimate. Exactness is a property of calls and of the
      totals, not of a share.
    - ``levels``: the ``section`` level asked for and the ``document`` level of the final
      document decision (``None`` when no document call was made).
    - ``backend``: the backend's name; ``calls``: how it was asked, ``"batched"`` (two-step
      sections), ``"batched_flat"`` or ``"per_item"``; ``tokens_pattern``: the call pattern the
      tokens describe. It equals ``calls`` when the backend reports usage; a backend that does
      not is asked per item but estimated as the pattern ``calls`` asked for (per item with
      ``cascade``).
    - ``tokens_split``: the estimated prompt tokens of the ``options``, the ``content`` (the
      views) and the ``instructions``. For an estimated run they sum to ``tokens.prompt``; next
      to exact totals they show the share the options take.
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    flow: Flow
    doc_type: str | None
    doc_type_confidence: float | None = 0.0
    doc_type_probs: dict[str, float] = Field(default_factory=dict)
    decided_by: DecidedBy
    candidates: list[str] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)
    document_calls: list[CallRecord] = Field(default_factory=list)
    section_calls: list[CallRecord] = Field(default_factory=list)
    doc_truncated: bool = False
    section_taxonomy: str | None = None
    sections: list[SectionTag] = Field(default_factory=list)
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    stage_tokens: dict[str, TokenUsage] = Field(default_factory=dict)
    levels: dict[str, Level | None] = Field(default_factory=dict)
    backend: str
    calls: Calls = "per_item"
    tokens_pattern: Calls = "per_item"
    tokens_split: dict[str, int] = Field(default_factory=dict)


def classify(
    source_or_doc: Document | str | os.PathLike[str],
    *,
    flow: Flow = "sections_first",
    level: Level = "terms",
    section_level: Level | None = None,
    doc_level: Level | None = None,
    backend: str | Backend = DEFAULT_BACKEND,
    log: str | os.PathLike[str] | None = None,
    max_sections: int = 40,
    rule_min_conf: float = 0.5,
    decide_conf: float = 0.7,
    cascade: bool = False,
    threshold: float = 0.5,
    doc_types: Taxonomy | None = None,
    catalogue: Taxonomy | None = None,
    calls: Calls | None = None,
    max_call_tokens: int = MAX_CALL_TOKENS,
) -> Classification:
    """Classify a document's sections and type with the minimum context (see the module docs).

    ``source_or_doc`` is a :class:`Document` or anything :func:`~docrooter.load` reads.
    ``section_level`` and ``doc_level`` default to ``level``. At most ``max_sections`` sections
    are tagged: level 1 first, then level 2, each in reading order. ``rule_min_conf`` is the
    least confidence a tag needs to count for a rule, ``decide_conf`` the least every supporting
    tag needs for the rules to decide alone. ``cascade`` and ``threshold`` control escalation
    (per-item calls only). ``calls`` is ``"batched"`` (two-step sections), ``"batched_flat"``
    or ``"per_item"``; ``None`` batches when the backend can (see the module docs).
    ``max_call_tokens`` budgets every call's prompt, on estimated tokens: batched sections are
    packed into calls of at most 40 views and :func:`~docrooter.batch.call_budget` of it (85%,
    the rest covering the estimate's error), a view too large for any call is cut (flagged
    ``truncated``) and sent alone, per-item views are cut to fit too, and the document view of
    every level is cut to fit beside the document call's instructions and options as sent
    (batched JSON or per-item letters; flag ``truncated`` and ``doc_truncated``). Every budget is
    divided by the backend's learned exact-to-estimated token ratio (``token_ratio()``, 1.0 to
    3.0; see the module docs), recorded per call as ``CallRecord.token_ratio``. It raises
    ``ValueError`` when the options leave no room for a view. With per-item calls, more doc
    types than the backend's options per call (26 for the OpenAI-compatible backend) raise
    ``ValueError`` up front.

    ``doc_types`` and ``catalogue`` replace the built-in doc-type taxonomy (with its ``sections``
    rules) and section catalogue; a rule naming an id the catalogue lacks raises ``ValueError``.
    A model loaded here from a spec string is released before returning; a backend instance is
    left open.
    """
    if flow not in FLOWS:
        raise ValueError(f"unknown flow {flow!r}; valid flows: {', '.join(FLOWS)}")
    levels = {"section": section_level or level, "document": doc_level or level}
    for name in (level, *levels.values()):
        check_level_and_view(name, None)
    if max_sections < 0:
        raise ValueError(f"max_sections must be at least 0, got {max_sections}")
    if calls is not None and calls not in CALLS:
        raise ValueError(f"unknown calls {calls!r}; use {', '.join(map(repr, CALLS))}")
    call_budget(max_call_tokens)  # ValueError below 1
    doc_type_taxonomy = doc_types if doc_types is not None else builtin_taxonomy(DOC_TYPES)
    catalogue_taxonomy = catalogue if catalogue is not None else builtin_taxonomy(CATALOGUE)
    rules = doc_type_rules(doc_type_taxonomy)
    unknown = unknown_section_ids(rules, catalogue_taxonomy)
    if unknown:
        shown = "; ".join(f"{doc_type}: {', '.join(ids)}" for doc_type, ids in unknown.items())
        raise ValueError(f"doc-type rules name section ids the catalogue lacks: {shown}")
    doc = source_or_doc if isinstance(source_or_doc, Document) else _load(source_or_doc)

    inner = _resolved(backend)
    try:
        can_batch = callable(getattr(inner, "chat_json", None))
        if cascade and (calls in ("batched", "batched_flat") or (calls is None and can_batch)):
            raise ValueError(
                "cascade re-asks single items, so it needs calls='per_item'; batched calls "
                "answer a whole stage at once"
            )
        per_item = not can_batch or calls == "per_item"
        limit = inner.max_candidates
        n_types = len(doc_type_taxonomy.leaves())
        if per_item and limit is not None and n_types > limit:
            raise ValueError(
                f"{inner.name} takes at most {limit} options per call, but the document call "
                f"can offer all {n_types} doc types; use calls='batched' or a doc-type "
                f"taxonomy of at most {limit} types"
            )
        estimating = usage_of(inner) is None and not hasattr(inner, "on_call")
        run = _Run(
            doc=doc,
            flow=flow,
            metered=_Metered(inner),
            doc_types=doc_type_taxonomy,
            cascade=cascade,
            threshold=threshold,
            batched=can_batch and calls != "per_item",
            flat=calls == "batched_flat",
            # cascade re-asks single items, which no batched call could: estimate per item
            batch_estimate=estimating and calls != "per_item" and not cascade,
            max_call_tokens=max_call_tokens,
        )
        if flow == "doc_first":
            result = run.doc_first(levels["section"], levels["document"], max_sections)
        else:
            result = run.sections_first(
                levels["section"],
                levels["document"],
                max_sections,
                rules,
                catalogue_taxonomy,
                CATALOGUE if catalogue is None else CUSTOM_TAXONOMY,
                rule_min_conf,
                decide_conf,
            )
        if log is not None:
            run.write_log(log, inner)
        return result
    finally:
        if isinstance(backend, str):
            close_backend(inner)


def select_sections(doc: Document, max_sections: int = 40) -> list[Section]:
    """The sections :func:`classify` tags: those at heading levels 1 and 2 with a title or text,
    level 1 first, then level 2, each in reading order, at most ``max_sections``."""
    usable = [s for s in doc.walk() if s.title.strip() or s.text.strip()]
    ordered = [s for level in SECTION_LEVELS for s in usable if s.level == level]
    return ordered[:max_sections]


# --- metering -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Call:
    """One text scored in one backend call: its options, probabilities and tokens."""

    text: str
    options: tuple[str, ...]
    probs: tuple[float, ...]
    usage: TokenUsage
    record: Mapping[str, Any] | None
    parts: Mapping[str, int] = field(default_factory=dict)  # estimated prompt parts


class _Metered:
    """A backend wrapper that records every text it scores with the tokens it cost.

    It keeps the wrapped backend's name and temperature, so the classifier treats it exactly as
    the backend itself; other attributes are passed through. ``spent`` is the usage of every
    call so far, counted per call (or per usage change), never from shares.
    """

    def __init__(self, backend: Backend) -> None:
        self.inner = backend
        self.name = backend.name
        self.max_candidates = backend.max_candidates
        self.temperature = backend_temperature(backend)
        self.calls: list[_Call] = []
        self.spent = TokenUsage()

    def __getattr__(self, name: str) -> Any:
        if name == "inner":  # not set yet (a copy in progress): never recurse
            raise AttributeError(name)
        return getattr(self.inner, name)

    def supports(self, node_id: str) -> bool:
        return self.inner.supports(node_id)

    def tokenizer(self) -> Callable[[str], int] | None:
        return self.inner.tokenizer()

    def score(
        self,
        texts: Sequence[str],
        candidates: Sequence[Candidate],
        *,
        instruction: str | None = None,
        multi_label: bool = False,
    ) -> Scores:
        texts = list(texts)
        records: list[Mapping[str, Any]] = []
        has_sink = hasattr(self.inner, "on_call")
        previous = getattr(self.inner, "on_call", None)
        before = usage_of(self.inner)
        if has_sink:

            def sink(record: Mapping[str, Any]) -> None:
                records.append(record)
                if previous is not None:
                    previous(record)

            self.inner.on_call = sink  # type: ignore[attr-defined]
        try:
            scores = score_batch(self.inner, texts, candidates, instruction, multi_label)
        finally:
            if has_sink:
                self.inner.on_call = previous  # type: ignore[attr-defined]
        if np.shape(scores.values) != (len(texts), len(candidates)):
            return scores  # the engine reports the bad shape
        after = usage_of(self.inner)
        by_index = {r.get("index"): r for r in records}
        parts = [_choice_parts(text, candidates, instruction) for text in texts]
        usages = self._usages(texts, parts, by_index, before, after)
        if not by_index and before is not None and after is not None:
            self.spent = self.spent + (after - before)  # the change, not its shares
        else:
            self.spent = self.spent + _sum(usages)
        probs = to_probs(scores, temperature=self.temperature, multi_label=multi_label)
        options = tuple(c.node_id for c in candidates)
        for i, text in enumerate(texts):
            row = tuple(float(p) for p in probs[i])
            self.calls.append(_Call(text, options, row, usages[i], by_index.get(i), parts[i]))
        return scores

    def chat(
        self, prompt: BatchPrompt, *, max_tokens: int, options: Sequence[str]
    ) -> tuple[dict[str, Any] | None, TokenUsage, Mapping[str, Any] | None]:
        """One batched call through the backend's ``chat_json``: the parsed reply, the tokens
        it spent and its call record (when the backend reports one)."""
        records: list[Mapping[str, Any]] = []
        previous = getattr(self.inner, "on_call", None)
        has_sink = hasattr(self.inner, "on_call")
        before = usage_of(self.inner)
        if has_sink:

            def sink(record: Mapping[str, Any]) -> None:
                records.append(record)
                if previous is not None:
                    previous(record)

            self.inner.on_call = sink  # type: ignore[attr-defined]
        try:
            reply = self.inner.chat_json(  # type: ignore[attr-defined]
                prompt.messages, max_tokens=max_tokens, options=list(options)
            )
        finally:
            if has_sink:
                self.inner.on_call = previous  # type: ignore[attr-defined]
        after = usage_of(self.inner)
        if records:
            usage = _record_usage(records[-1])
        elif before is not None and after is not None:
            usage = after - before
        else:
            usage = TokenUsage(prompt.estimate(), reply_tokens(reply) if reply else 1, 1, False)
        self.spent = self.spent + usage
        return reply, usage, records[-1] if records else None

    @staticmethod
    def _usages(
        texts: list[str],
        parts: list[dict[str, int]],
        records: Mapping[Any, Mapping[str, Any]],
        before: TokenUsage | None,
        after: TokenUsage | None,
    ) -> list[TokenUsage]:
        """Each text's tokens: its call record, else a share of the usage change, else an
        estimate of what an LLM would be sent."""
        if records:
            return [
                _record_usage(records[i]) if i in records else _estimate(parts[i])
                for i in range(len(texts))
            ]
        if before is not None and after is not None:
            return _split(after - before, len(texts))
        return [_estimate(p) for p in parts]


def _record_usage(record: Mapping[str, Any]) -> TokenUsage:
    return TokenUsage(
        int(record.get("prompt_tokens") or 0),
        int(record.get("completion_tokens") or 0),
        1,
        bool(record.get("exact")),
    )


def _estimate(parts: Mapping[str, int]) -> TokenUsage:
    """One per-item LLM call, estimated from its prompt parts; the reply is one letter."""
    return TokenUsage(sum(parts.values()), 1, 1, False)


def _choice_parts(
    text: str, candidates: Sequence[Candidate], instruction: str | None
) -> dict[str, int]:
    """The estimated prompt of the per-item request for ``text``, built as the OpenAI-compatible
    backend builds it (:func:`~docrooter.backends.openai.choice_messages`), in parts: the
    system prompt and the question, the options, and the rest (the text), summing to the
    estimate of the messages sent."""
    from docrooter.backends.openai import DEFAULT_INSTRUCTION, SYSTEM_PROMPT, choice_messages

    total = messages_tokens(choice_messages(text, [c.text for c in candidates], instruction))
    instructions = estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(
        instruction or DEFAULT_INSTRUCTION
    )
    options = estimate_tokens(options_block(candidates))
    return {
        "instructions": instructions,
        "options": options,
        "content": total - instructions - options,
    }


def options_block(candidates: Sequence[Candidate]) -> str:
    """The options as an LLM prompt lists them: ``A. text`` per line (numbers past 26)."""
    markers: Sequence[str] = (
        _LETTERS
        if len(candidates) <= len(_LETTERS)
        else [str(i + 1) for i in range(len(candidates))]
    )
    lines = [f"{m}. {c.text}" for m, c in zip(markers, candidates, strict=False)]
    return "\n".join(["Options:", *lines])


def _split(total: TokenUsage, n: int) -> list[TokenUsage]:
    """``total`` shared over ``n`` texts, remainders to the first ones. An even share is not
    what a text cost, so every share is an estimate (``exact=False``), however exact
    ``total`` is; a lone text keeps ``total`` as it is. Sum the calls, not the shares, for a
    stage's exactness."""
    if n == 1:
        return [total]

    def share(value: int, i: int) -> int:
        return value // n + (1 if i < value % n else 0)

    return [
        TokenUsage(share(total.prompt, i), share(total.completion, i), share(total.calls, i), False)
        for i in range(n)
    ]


def _sum(usages: Sequence[TokenUsage]) -> TokenUsage:
    return sum(usages, TokenUsage())


# --- the flows ----------------------------------------------------------------------------


@dataclass
class _Tag:
    section: Section
    label: str | None = None
    confidence: float = 0.0  # a text answer counts as 1.0 here, for the rules and the cascade
    level: Level = "terms"
    tokens: TokenUsage = field(default_factory=TokenUsage)
    source: ConfidenceSource = "probability"
    unparsed: bool = False


@dataclass
class _Decision:
    """A backend call as a log row's context: the stage, the section and the level."""

    stage: str
    step: Step
    section_id: str | None
    level: Level
    call: _Call
    label: Any = None  # a batched answer, or a document call's chosen type (None: unparsed)


@dataclass
class _DocDecision:
    result: DocTypeResult
    level: Level
    tokens: TokenUsage
    calls: list[CallRecord]
    text: bool = False  # the answer was a letter, without probabilities


@dataclass
class _Run:
    doc: Document
    flow: Flow
    metered: _Metered
    doc_types: Taxonomy
    cascade: bool
    threshold: float
    batched: bool = False  # few chat calls per stage
    flat: bool = False  # the batched sections stage lists every leaf in one call
    batch_estimate: bool = False  # per-item calls, estimated as the batched calls they stand for
    max_call_tokens: int = MAX_CALL_TOKENS
    truncated: bool = False
    unparsed: bool = False
    decisions: list[_Decision] = field(default_factory=list)
    section_usage: TokenUsage = field(default_factory=TokenUsage)  # the sections stage's calls
    section_calls: list[CallRecord] = field(default_factory=list)
    split: dict[str, int] = field(
        default_factory=lambda: {"options": 0, "content": 0, "instructions": 0}
    )

    # --- flows

    def sections_first(
        self,
        section_level: Level,
        doc_level: Level,
        max_sections: int,
        rules: Mapping[str, DocTypeRules],
        catalogue: Taxonomy,
        catalogue_name: str,
        rule_min_conf: float,
        decide_conf: float,
    ) -> Classification:
        selected = select_sections(self.doc, max_sections)
        if not selected:
            decision = self.decide_document(self.doc_types, doc_level)
            return self.result(
                decision,
                "fallback",
                [],
                ["no_sections"],
                None,
                [],
                section_level,
            )

        def tagger(view: SectionView) -> Classifier:
            return Classifier(catalogue, self.metered, strategy=Greedy(), view=view, threshold=0.0)

        tags = self.tag(selected, section_level, tagger, catalogue, catalogue_name)
        evidence: dict[str, float] = {}
        for tag in tags:
            if tag.label is not None:
                evidence[tag.label] = max(evidence.get(tag.label, 0.0), tag.confidence)
        match = match_rules(rules, evidence, min_conf=rule_min_conf, decide_conf=decide_conf)
        flags: list[Flag] = ["excludes_waived"] if match.excludes_waived else []
        support = _supporting(rules[match.decided], tags, rule_min_conf) if match.decided else []
        # batched or letter answers carry no probability: count supporting sections instead
        hard = self.batched or any(t.source == "text" for t in support)
        weak = match.decided is not None and hard and len(support) < BATCHED_SUPPORT
        if match.decided is not None and not weak:
            confidence: float | None = min(match.support[match.decided].values())
            if any(t.source == "text" for t in support):
                confidence = None
            return self.result(
                None,
                "rules",
                [match.decided],
                flags,
                catalogue_name,
                tags,
                section_level,
                doc_type=(match.decided, confidence),
            )
        offered = list(match.candidates)
        low_support = weak  # tags without probabilities: one section is too little
        if len(offered) <= 1 and match.near_misses:  # section 6.3: with its near misses
            offered += [t for t in match.near_misses if t not in offered]
            flags.append("near_miss")
        if len(offered) == 1:
            # a lone undecided candidate or near miss: offer it with its doc-type siblings,
            # but not those the rules' excludes argue against
            low_support = True
            siblings = _siblings(self.doc_types, offered[0])
            offered += [t for t in siblings if t not in match.barred]
        if low_support:
            flags.append("low_support")
        if len(offered) >= 2:
            decision = self.decide_document(_restricted(self.doc_types, offered), doc_level)
            decided_by: DecidedBy = "model"
        else:
            if not offered:
                flags.append("rules_unmet")
            decision = self.decide_document(self.doc_types, doc_level)
            decided_by = "fallback"
        return self.result(
            decision, decided_by, offered, flags, catalogue_name, tags, section_level
        )

    def doc_first(
        self, section_level: Level, doc_level: Level, max_sections: int
    ) -> Classification:
        decision = self.decide_document(self.doc_types, doc_level)
        best = decision.result.best
        doc_type = best.node_ids[-1] if best.node_ids else None
        name = _section_taxonomy_name(doc_type, self.doc_types)
        section_taxonomy = builtin_taxonomy(name)

        def tagger(view: SectionView) -> Classifier:
            return section_classifier(
                section_taxonomy, self.metered, doc_type, view, self.doc_types
            )

        selected = select_sections(self.doc, max_sections)
        tags = self.tag(selected, section_level, tagger, section_taxonomy, name)
        decided_by: DecidedBy = "model" if decision.calls else "fallback"  # nothing was sent
        return self.result(decision, decided_by, [], [], name, tags, section_level)

    # --- stages

    def tag(
        self,
        sections: list[Section],
        level: Level,
        tagger: Callable[[SectionView], Classifier],
        taxonomy: Taxonomy,
        taxonomy_name: str,
    ) -> list[_Tag]:
        """Tag ``sections`` at ``level``, asking the low-confidence ones again one rung up while
        ``cascade`` allows."""
        tags = {s.id: _Tag(s, level=level) for s in sections}
        if not sections:
            return []
        stage = events.start("normalize", taxonomy=taxonomy_name)
        pending: list[Section] = sections
        current: Level | None = level
        while pending and current is not None:
            if self.batched:
                self._tag_batched(pending, current, taxonomy, tags)
                break
            self._tag_once(pending, current, tagger, tags)
            if not self.cascade:
                break
            pending = [s for s in pending if tags[s.id].confidence < self.threshold]
            current = next_level(current)
        stage.end(sections=len(tags))
        return [tags[s.id] for s in sections]

    def ratio(self) -> float:
        """The backend's exact-to-estimated prompt token ratio now (:func:`token_ratio_of`)."""
        return token_ratio_of(self.metered.inner)

    def budget(self, ratio: float) -> int:
        """The estimated prompt tokens one call is packed or cut to: :func:`call_budget`
        divided by ``ratio``, so the exact prompt stays within it."""
        return max(1, int(call_budget(self.max_call_tokens) / ratio))

    def _tag_once(
        self,
        sections: list[Section],
        level: Level,
        tagger: Callable[[SectionView], Classifier],
        tags: dict[str, _Tag],
    ) -> None:
        view = section_view(level)
        classifier = tagger(view)
        texts, cut = self._render(sections, view, classifier._token_counter())
        ratio = self.ratio()
        texts, cut = self._fit_per_item(texts, cut, classifier, ratio)
        mark, before = len(self.metered.calls), self.metered.spent
        results = classifier.classify_texts(texts)
        owners = _owners(texts, self.metered.calls[mark:])
        leaves = {n.id for n in classifier.taxonomy.leaves()}
        spent: dict[int, TokenUsage] = {}
        answers: dict[int, list[str]] = {}  # how each section's replies were read, in order
        for call, index in owners:
            sid = sections[index].id if index is not None else None
            step: Step = "leaf" if set(call.options) <= leaves else "group"
            self.decisions.append(_Decision(SECTION_STAGE, step, sid, level, call))
            if not self.batch_estimate:
                self._add_split(call)
                truncated = cut[index] if index is not None else False
                record = _call_record(level, call.usage, truncated, len(call.options), ratio)
                self.section_calls.append(record)
            if index is not None:
                spent[index] = spent.get(index, TokenUsage()) + call.usage
                answers.setdefault(index, []).append(_answer(call))
        if not self.batch_estimate:  # every call, whether or not a section claimed it
            self.section_usage = self.section_usage + (self.metered.spent - before)
        for i, (section, result) in enumerate(zip(sections, results, strict=True)):
            tag = tags[section.id]
            best = result.best
            assigned = best is not None and bool(best.node_ids)
            tag.label = best.node_ids[-1] if assigned and best is not None else None
            tag.confidence = float(best.score) if assigned and best is not None else 0.0
            tag.level = level
            tag.source, tag.unparsed = "probability", False
            read = answers.get(i, [])
            if "none" in read:  # a reply named no option: never the option it was scored as
                failed = read.index("none")  # the first step whose reply named no option
                path = best.node_ids if best is not None else []
                parent = path[failed - 1] if 0 < failed <= len(path) else None
                tag.label = _unparsed_label(classifier.taxonomy, parent)
                tag.confidence, tag.source, tag.unparsed = 0.0, "text", True
            elif "letter" in read and assigned:
                tag.confidence, tag.source = 1.0, "text"  # a bare letter: no probability
        if self.batch_estimate:
            labels = [tags[s.id].label for s in sections]
            spent = self._batch_estimate(texts, labels, classifier.taxonomy, level, cut, ratio)
        for index, usage in spent.items():
            tag = tags[sections[index].id]
            tag.tokens = tag.tokens + usage

    def _render(
        self, sections: list[Section], view: SectionView, count: Callable[[str], int]
    ) -> tuple[list[str], list[bool]]:
        """The sections' views, and whether each was cut; a section with no title still shows
        its lead."""
        rendered = view.render_all(self.doc, [s.id for s in sections], count)
        blank = [s.id for s in sections if not s.title.strip()]
        if blank and not _BODY_FIELDS.intersection(view.fields):
            with_lead = view.model_copy(update={"fields": (*view.fields, "lead")})
            rendered |= with_lead.render_all(self.doc, blank, count)
        cut = [rendered[s.id].truncated for s in sections]
        self.truncated = self.truncated or any(cut)
        return [rendered[s.id].text for s in sections], cut

    def _fit_per_item(
        self, texts: list[str], cut: list[bool], classifier: Classifier, ratio: float
    ) -> tuple[list[str], list[bool]]:
        """``texts`` cut so that each per-item call, with the largest option list the
        classifier can send, stays within :meth:`budget` at ``ratio``."""
        room = self.budget(ratio) - _per_item_overhead(classifier)
        if room < 1:
            raise ValueError(
                f"max_call_tokens {self.max_call_tokens} leaves no room for a section view "
                "after the instructions and options of a per-item call"
            )
        fitted, flags = [], []
        for text, was_cut in zip(texts, cut, strict=True):
            if estimate_tokens(text) > room:
                text, was_cut = cut_to_tokens(text, room), True
                self.truncated = True
            fitted.append(text)
            flags.append(was_cut)
        return fitted, flags

    def _add_split(self, call: _Call) -> None:
        for part, tokens in call.parts.items():
            self.split[part] += tokens

    def _add_parts(self, prompt: BatchPrompt) -> None:
        for part, tokens in prompt.parts().items():
            self.split[part] += tokens

    @property
    def tokens_pattern(self) -> Calls:
        """The call pattern the tokens describe."""
        if self.batched or self.batch_estimate:
            return "batched_flat" if self.flat else "batched"
        return "per_item"

    def _two_step(self, taxonomy: Taxonomy) -> bool:
        """Whether the batched sections stage takes two steps: not flat, and the taxonomy has
        groups."""
        return not self.flat and any(taxonomy.children(n.id) for n in taxonomy.children())

    def _batch_estimate(
        self,
        texts: list[str],
        labels: list[str | None],
        taxonomy: Taxonomy,
        level: Level,
        cut: list[bool],
        ratio: float = 1.0,
    ) -> dict[int, TokenUsage]:
        """Each section's share of the batched calls an LLM would have been sent for
        ``texts`` (with the answers ``labels``), packed as :meth:`_ask` packs them: its own
        view, plus an even share of each call's options and instructions. Each call is
        recorded in the stage."""
        spent: dict[int, TokenUsage] = {i: TokenUsage() for i in range(len(texts))}
        everyone = list(range(len(texts)))
        Ask = tuple[list[int], Callable[[dict[str, str]], BatchPrompt], int, list[str | None]]
        asks: list[Ask] = []
        if not self._two_step(taxonomy):
            flat = functools.partial(section_prompt, taxonomy=taxonomy)
            asks.append((everyone, flat, len(taxonomy.leaves()), labels))
        else:
            groups = [_group_of(taxonomy, label) for label in labels]
            sort = functools.partial(group_prompt, taxonomy=taxonomy)
            asks.append((everyone, sort, len(taxonomy.children()), groups))
            for group in dict.fromkeys(g for g in groups if g):
                leaf_ids = group_leaves(taxonomy, group)
                if leaf_ids == [group]:
                    continue  # a top-level leaf: the group call answered it (as _tag_batched)
                members = [i for i in everyone if groups[i] == group]
                leaves = functools.partial(section_prompt, taxonomy=taxonomy, leaf_ids=leaf_ids)
                asks.append((members, leaves, len(leaf_ids), labels))
        for members, prompt_for, n_options, answers in asks:
            views = {str(i + 1): texts[i] for i in members}
            for packed in pack(views, prompt_for, self.budget(ratio)):
                chunk = [int(n) - 1 for n in packed]
                prompt = prompt_for(packed)
                self._add_parts(prompt)
                completion = reply_tokens({str(i + 1): answers[i] or "" for i in chunk})
                call = TokenUsage(prompt.estimate(), completion, 1, False)
                truncated = any(cut[i] or packed[str(i + 1)] != texts[i] for i in chunk)
                self.truncated = self.truncated or truncated
                self.section_usage = self.section_usage + call
                self.section_calls.append(_call_record(level, call, truncated, n_options, ratio))
                # each section gets its own item; the rest of the call is shared
                overhead = prompt.estimate() - sum(estimate_tokens(it) for it in prompt.items)
                shares = _split(TokenUsage(overhead, completion, 1, False), len(chunk))
                for item, i, share in zip(prompt.items, chunk, shares, strict=True):
                    prompt_tokens = share.prompt + estimate_tokens(item)
                    usage = TokenUsage(prompt_tokens, share.completion, share.calls, False)
                    spent[i] = spent[i] + usage
        return spent

    def _tag_batched(
        self, sections: list[Section], level: Level, taxonomy: Taxonomy, tags: dict[str, _Tag]
    ) -> None:
        """Tag ``sections`` with batched calls: the groups first, then each group's leaves
        (or every leaf at once, when flat)."""
        count = self.metered.tokenizer() or estimate_tokens
        texts, cut = self._render(sections, section_view(level), count)
        views = {str(i + 1): text for i, text in enumerate(texts)}
        ids = {str(i + 1): s.id for i, s in enumerate(sections)}
        was_cut = {str(i + 1): flag for i, flag in enumerate(cut)}
        spent = {n: TokenUsage() for n in views}
        fallback = UNPARSED_LABEL if UNPARSED_LABEL in taxonomy else None
        labels: dict[str, str | None]
        fallbacks: dict[str, str | None]
        ask = functools.partial(self._ask, ids=ids, spent=spent, level=level, cut=was_cut)
        if not self._two_step(taxonomy):
            leaf_ids = [leaf.id for leaf in taxonomy.leaves()]
            every_leaf = functools.partial(section_prompt, taxonomy=taxonomy)
            labels = ask(views, step="batch", options=leaf_ids, prompt_for=every_leaf)
            fallbacks = dict.fromkeys(views, fallback)
        else:  # the groups first, then each group's leaves
            group_ids = [node.id for node in taxonomy.children()]
            groups = ask(
                views,
                step="batch_group",
                options=group_ids,
                prompt_for=functools.partial(group_prompt, taxonomy=taxonomy),
                to_group=functools.partial(_group_of, taxonomy),
            )
            labels = dict.fromkeys(views)
            fallbacks = dict.fromkeys(views, fallback)
            for group in dict.fromkeys(g for g in groups.values() if g is not None):
                members = {n: views[n] for n, g in groups.items() if g == group}
                leaf_ids = group_leaves(taxonomy, group)
                if leaf_ids == [group]:  # a top-level leaf: the group is the answer
                    labels.update({n: group for n in members})
                    continue
                other = f"{group}.other"
                for n in members:
                    fallbacks[n] = other if other in taxonomy else fallback
                its_leaves = functools.partial(section_prompt, taxonomy=taxonomy, leaf_ids=leaf_ids)
                labels.update(
                    ask(members, step="batch_leaf", options=leaf_ids, prompt_for=its_leaves)
                )
        for n, section in zip(views, sections, strict=True):
            tag = tags[section.id]
            found = labels[n]
            tag.label = found if found is not None else fallbacks[n]
            tag.confidence = 1.0 if found is not None else 0.0
            tag.source = "batched"
            tag.level = level
            tag.tokens = tag.tokens + spent[n]
            self.unparsed = self.unparsed or found is None

    def _ask(
        self,
        views: Mapping[str, str],
        *,
        ids: Mapping[str, str],
        spent: dict[str, TokenUsage],
        level: Level,
        cut: Mapping[str, bool],
        step: Step,
        options: list[str],
        prompt_for: Callable[[dict[str, str]], BatchPrompt],
        to_group: Callable[[str], str | None] | None = None,
    ) -> dict[str, str | None]:
        """One batched step over ``views`` (by section number), packed into calls by
        :func:`~docrooter.batch.pack` (at most 40 views, and :meth:`budget` estimated tokens,
        each; a view too large for a call is cut and sent alone), packed again before each call
        so that a token ratio learned from the last one budgets the next: each section's answer
        among
        ``options``, ``None`` when it named none twice. ``to_group`` maps an answer that is not
        an option (a leaf id, in the groups step) to one. Each call's tokens are shared among
        its sections in ``spent``."""
        valid = {option.lower(): option for option in options}

        def answer_of(value: object) -> str | None:
            found = valid_id(value, valid)
            if found is None and to_group is not None and isinstance(value, str):
                group = to_group(value.strip().strip("\"'").strip())
                found = group if group in options else None
            return found

        answers: dict[str, str | None] = dict.fromkeys(views)
        remaining = dict(views)
        while remaining:
            ratio = self.ratio()
            packed = pack(remaining, prompt_for, self.budget(ratio))[0]
            for n in packed:
                del remaining[n]
            truncated = any(cut[n] or packed[n] != views[n] for n in packed)
            self.truncated = self.truncated or truncated
            pending = list(packed)
            for attempt in range(2):  # the first ask, then once more for the failures
                if not pending:
                    break
                prompt = prompt_for({n: packed[n] for n in pending})
                if attempt:
                    prompt = prompt.retry()
                reply, usage, record = self._chat(prompt, options, len(pending))
                self.section_usage = self.section_usage + usage
                self.section_calls.append(
                    _call_record(level, usage, truncated, len(options), ratio)
                )
                for n in pending:
                    answers[n] = answer_of((reply or {}).get(n))
                for n, share in zip(pending, _split(usage, len(pending)), strict=True):
                    spent[n] = spent[n] + share
                call = _Call(prompt.messages[1]["content"], tuple(options), (), usage, record)
                label = {ids[n]: answers[n] for n in pending}
                self.decisions.append(_Decision(SECTION_STAGE, step, None, level, call, label))
                pending = [n for n in pending if answers[n] is None]
        return answers

    def _chat(
        self, prompt: BatchPrompt, options: Sequence[str], n_answers: int
    ) -> tuple[dict[str, Any] | None, TokenUsage, Mapping[str, Any] | None]:
        self._add_parts(prompt)
        return self.metered.chat(
            prompt, max_tokens=section_reply_max_tokens(n_answers), options=options
        )

    def decide_document(self, taxonomy: Taxonomy, level: Level) -> _DocDecision:
        """The document's type among ``taxonomy``'s leaves, asked again one rung up while
        ``cascade`` allows and the confidence is below ``threshold``."""
        if self.batched:
            return self._decide_batched(taxonomy, level)
        current: Level | None = level
        tokens = TokenUsage()
        result: DocTypeResult | None = None
        used = level
        records: list[CallRecord] = []
        text = unparsed = False
        while current is not None:
            mark = len(self.metered.calls)
            ratio = self.ratio()
            result = classify_document(
                self.doc,
                taxonomy,
                backend=self.metered,
                view=self._document_view(current, taxonomy, ratio),
                strategy=Flat(),
            )
            used = current
            attempt = TokenUsage()
            read = [_answer(call) for call in self.metered.calls[mark:]]
            unparsed, text = "none" in read, "letter" in read
            if unparsed:  # the reply named no type: none is chosen, nor the first offered
                result = result.model_copy(
                    update={
                        "best": PathResult(node_ids=[], stopped_reason="unparsed"),
                        "confidence": 0.0,
                        "leaf_probs": {},
                    }
                )
            for call in self.metered.calls[mark:]:
                label = None if unparsed else _argmax(call)
                self.decisions.append(
                    _Decision(DOCUMENT_STAGE, "document", None, current, call, label=label)
                )
                attempt = attempt + call.usage
                if not self.batch_estimate:
                    self._add_split(call)
            if self.batch_estimate and attempt.calls:
                prompt = document_prompt(result.view_text, taxonomy)
                self._add_parts(prompt)
                best = result.best.node_ids[-1] if result.best.node_ids else ""
                completion = reply_tokens({DOC_TYPE_KEY: best})
                attempt = TokenUsage(prompt.estimate(), completion, 1, False)
            tokens = tokens + attempt
            if attempt.calls:  # an empty document is never sent
                records.append(
                    CallRecord(
                        level=current,
                        prompt_tokens=attempt.prompt,
                        completion_tokens=attempt.completion,
                        exact=attempt.exact,
                        truncated=result.truncated,
                        options_count=len(taxonomy.leaves()),
                        token_ratio=ratio,
                    )
                )
            self.truncated = self.truncated or result.truncated
            if (
                not self.cascade
                or result.confidence >= self.threshold
                or result.best.stopped_reason == "empty_text"
            ):
                break
            current = next_level(current)
        assert result is not None
        self.unparsed = self.unparsed or unparsed  # the deciding attempt's reply
        return _DocDecision(result, used, tokens, records, text=text and not unparsed)

    def _document_view(self, level: Level, taxonomy: Taxonomy, ratio: float) -> DocumentView:
        """The document view preset of ``level``, its budget cut so that the document call,
        with its instructions and options as they are sent (the batched JSON prompt, re-ask
        included, or the per-item lettered one), stays within :meth:`budget` at ``ratio``.
        ``ValueError`` when not even an empty view fits."""
        if self.batched or self.batch_estimate:
            overhead = document_prompt("", taxonomy).retry().estimate()
        else:
            from docrooter.backends.openai import choice_messages

            options = [
                candidate_text(n, taxonomy=taxonomy, full_path=True) for n in taxonomy.leaves()
            ]
            overhead = messages_tokens(choice_messages("", options, DOC_INSTRUCTION))
        room = self.budget(ratio) - overhead
        if room < 1:
            raise ValueError(
                f"max_call_tokens {self.max_call_tokens} leaves no room for the document view "
                f"after the {overhead}-token instructions and {len(taxonomy.leaves())} doc-type "
                "options of the document call; raise max_call_tokens"
            )
        try:
            preset = document_view(level, full_cap=self.max_call_tokens)
        except ValueError:  # a cap below the full view's fixed reserve: the room decides
            preset = document_view(level)
        return preset.model_copy(update={"max_tokens": min(preset.max_tokens, room)})

    def _decide_batched(self, taxonomy: Taxonomy, level: Level) -> _DocDecision:
        """The document's type among ``taxonomy``'s leaves in one chat call (asked once more
        when the reply names no option)."""
        if not _has_content(self.doc):
            empty = PathResult(node_ids=[], stopped_reason="empty_text")
            return _DocDecision(
                DocTypeResult(source=self.doc.source, best=empty), level, TokenUsage(), []
            )
        count = self.metered.tokenizer() or estimate_tokens
        ratio = self.ratio()
        view = self._document_view(level, taxonomy, ratio).render(self.doc, count)
        self.truncated = self.truncated or view.truncated
        leaf_ids = [leaf.id for leaf in taxonomy.leaves()]
        valid = {leaf_id.lower(): leaf_id for leaf_id in leaf_ids}
        prompt = document_prompt(view.text, taxonomy)
        tokens, records, chosen = TokenUsage(), [], None
        for attempt in range(2):
            reply, usage, record = self._chat(prompt.retry() if attempt else prompt, leaf_ids, 1)
            chosen = valid_id((reply or {}).get(DOC_TYPE_KEY), valid)
            call = _Call(view.text, tuple(leaf_ids), (), usage, record)
            self.decisions.append(
                _Decision(DOCUMENT_STAGE, "document", None, level, call, label=chosen)
            )
            tokens = tokens + usage
            records.append(
                CallRecord(
                    level=level,
                    prompt_tokens=usage.prompt,
                    completion_tokens=usage.completion,
                    exact=usage.exact,
                    truncated=view.truncated,
                    options_count=len(leaf_ids),
                    token_ratio=ratio,
                )
            )
            if chosen is not None:
                break
        if chosen is not None:
            nodes = taxonomy.path(chosen)
            best = PathResult(
                node_ids=[n.id for n in nodes], labels=[n.label for n in nodes], score=1.0
            )
            probs = {chosen: 1.0}
        else:
            self.unparsed = True
            best, probs = PathResult(node_ids=[], stopped_reason="unparsed"), {}
        result = DocTypeResult(
            source=self.doc.source,
            best=best,
            confidence=1.0 if chosen is not None else 0.0,
            leaf_probs=probs,
            view_text=view.text,
            truncated=view.truncated,
        )
        return _DocDecision(result, level, tokens, records)

    # --- output

    def result(
        self,
        decision: _DocDecision | None,
        decided_by: DecidedBy,
        candidates: list[str],
        flags: list[Flag],
        section_taxonomy: str | None,
        tags: list[_Tag],
        section_level: Level,
        *,
        doc_type: tuple[str, float | None] | None = None,
    ) -> Classification:
        probs: dict[str, float] = {}
        if decision is not None:
            best = decision.result.best
            confidence = None if decision.text else decision.result.confidence
            doc_type = (best.node_ids[-1], confidence) if best.node_ids else None
            probs = {} if decision.text else dict(decision.result.leaf_probs)
        section_tokens = self.section_usage
        document_tokens = decision.tokens if decision is not None else TokenUsage()
        if self.truncated:
            flags = [*flags, "truncated"]
        if self.unparsed or any(t.unparsed for t in tags):
            flags = [*flags, "unparsed"]
        return Classification(
            source=self.doc.source,
            flow=self.flow,
            doc_type=doc_type[0] if doc_type else None,
            doc_type_confidence=doc_type[1] if doc_type else 0.0,
            doc_type_probs=probs,
            decided_by=decided_by,
            candidates=candidates,
            flags=flags,
            document_calls=decision.calls if decision is not None else [],
            section_calls=list(self.section_calls),
            doc_truncated=decision.result.truncated if decision is not None else False,
            section_taxonomy=section_taxonomy,
            sections=[
                SectionTag(
                    id=t.section.id,
                    title=t.section.title,
                    level=t.section.level,
                    label=t.label,
                    confidence=None if t.source == "text" and not t.unparsed else t.confidence,
                    level_used=t.level,
                    tokens=t.tokens,
                    confidence_source=t.source,
                )
                for t in tags
            ],
            tokens=section_tokens + document_tokens,
            stage_tokens={SECTION_STAGE: section_tokens, DOCUMENT_STAGE: document_tokens},
            levels={
                "section": section_level,
                "document": decision.level if decision is not None and decision.calls else None,
            },
            backend=self.metered.name,
            calls=("batched_flat" if self.flat else "batched") if self.batched else "per_item",
            tokens_pattern=self.tokens_pattern,
            tokens_split=dict(self.split),
        )

    def write_log(self, path: str | os.PathLike[str], backend: Backend) -> None:
        """One results-log row per backend decision, in the order they were made."""
        log = ResultsLog(Path(path), secrets=secrets_of(backend))
        source = Path(self.doc.source).name
        for decision in self.decisions:
            call, record = decision.call, decision.call.record or {}
            batched = not call.probs  # a batched answer: a label, no probabilities
            text = _answer(call) in ("letter", "none")  # a letter, or no answer: no probability
            if batched:
                label, source = decision.label, "batched"
            elif decision.stage == DOCUMENT_STAGE:
                label, source = decision.label, "text" if text else "probability"
            else:
                label = None if _answer(call) == "none" else _argmax(call)
                source = "text" if text else "probability"
            log.write(
                {
                    "source": source,
                    "flow": self.flow,
                    "stage": decision.stage,
                    "step": decision.step,
                    "section_id": decision.section_id,
                    "level": decision.level,
                    "view_hash": text_hash(call.text),
                    "options": list(call.options),
                    "label": label,
                    "probs": (
                        {} if batched or text else dict(zip(call.options, call.probs, strict=True))
                    ),
                    "confidence_source": source,
                    "model": record.get("model", self.metered.name),
                    "backend": self.metered.name,
                    "endpoint_host": record.get("endpoint_host"),
                    "prompt_hash": record.get("prompt_hash"),
                    "tokens": {
                        "prompt": call.usage.prompt,
                        "completion": call.usage.completion,
                        "exact": call.usage.exact,
                    },
                }
            )


def _owners(texts: list[str], calls: list[_Call]) -> list[tuple[_Call, int | None]]:
    """Each call with the index of the text it scored. Sections that render the same text
    take that text's calls in turn, so each is charged for its own."""
    positions: dict[str, list[int]] = {}
    for i, text in enumerate(texts):
        positions.setdefault(text, []).append(i)
    seen: dict[str, int] = {}
    out: list[tuple[_Call, int | None]] = []
    for call in calls:
        indices = positions.get(call.text)
        if not indices:
            out.append((call, None))
            continue
        n = seen.get(call.text, 0)
        seen[call.text] = n + 1
        out.append((call, indices[n % len(indices)]))
    return out


def _argmax(call: _Call) -> str | None:
    """A per-item call's most probable option."""
    return call.options[int(np.argmax(call.probs))] if call.probs else None


def _answer(call: _Call) -> str:
    """How a per-item call's reply was read: ``"logprobs"``, ``"letter"`` or ``"none"`` from
    its record; ``"probability"`` for a backend that scores without replies."""
    record = call.record or {}
    answer = record.get("answer")
    return answer if isinstance(answer, str) else "probability"


def _call_record(
    level: Level, usage: TokenUsage, truncated: bool, n_options: int, ratio: float
) -> CallRecord:
    return CallRecord(
        level=level,
        prompt_tokens=usage.prompt,
        completion_tokens=usage.completion,
        exact=usage.exact,
        truncated=truncated,
        options_count=n_options,
        token_ratio=ratio,
    )


def _per_item_overhead(classifier: Classifier) -> int:
    """The estimated prompt tokens of the largest per-item request ``classifier`` can send,
    without its text: the system prompt, the question and the longest option list."""
    from docrooter.backends.openai import choice_messages

    taxonomy = classifier.taxonomy
    if isinstance(classifier.strategy, Flat):
        option_sets = [
            [candidate_text(n, taxonomy=taxonomy, full_path=True) for n in taxonomy.leaves()]
        ]
    else:
        parents = [n.id for n in taxonomy.nodes.values() if taxonomy.children(n.id)]
        option_sets = [[candidate_text(c) for c in taxonomy.children()]]
        option_sets += [[candidate_text(c) for c in taxonomy.children(p)] for p in parents]
    instruction = classifier.settings.instruction
    return max(
        messages_tokens(choice_messages("", options, instruction)) for options in option_sets
    )


def _supporting(rule: DocTypeRules, tags: Sequence[_Tag], min_conf: float) -> list[_Tag]:
    """The tagged sections that support any of ``rule``'s required groups."""
    ids = [rule_id for group in rule.requires for rule_id in group]
    return [
        tag
        for tag in tags
        if tag.label is not None
        and tag.confidence >= min_conf
        and any(covers(rule_id, tag.label) for rule_id in ids)
    ]


def _unparsed_label(taxonomy: Taxonomy, parent: str | None) -> str | None:
    """The tag of a section whose answer named no option below ``parent`` (``None`` at the
    top): the parent's ``other`` leaf, else ``general.other``, else nothing."""
    if parent is not None and f"{parent}.other" in taxonomy:
        return f"{parent}.other"
    return UNPARSED_LABEL if UNPARSED_LABEL in taxonomy else None


def _group_of(taxonomy: Taxonomy, label: str | None) -> str | None:
    """The top-level node above ``label`` (``None`` for no label or an unknown one)."""
    if label is None or label not in taxonomy:
        return None
    return taxonomy.path(label)[0].id


def _siblings(taxonomy: Taxonomy, leaf_id: str) -> list[str]:
    """The other leaves under ``leaf_id``'s parent group, in taxonomy order; none for a
    top-level leaf."""
    parent = taxonomy.node(leaf_id).parent_id
    if parent is None:
        return []
    return [n.id for n in taxonomy.children(parent) if n.id != leaf_id and not n.children]


def _restricted(taxonomy: Taxonomy, leaf_ids: Sequence[str]) -> Taxonomy:
    """``taxonomy`` cut down to ``leaf_ids`` and their ancestors; its leaves keep the order of
    ``leaf_ids``, the order the options are offered in."""
    nodes: dict[str, Any] = {}
    for leaf_id in leaf_ids:
        for node in taxonomy.path(leaf_id):
            if node.id not in nodes:
                nodes[node.id] = node.model_copy(update={"children": []})
    return Taxonomy(list(nodes.values()))


def _load(source: str | os.PathLike[str]) -> Document:
    from docrooter.document.load import load

    return load(source)


def _resolved(backend: str | Backend) -> Backend:
    if not isinstance(backend, str):
        return backend
    from docrooter.backends import registry

    return registry.resolve_backend(backend)


__all__ = [
    "CALLS",
    "CATALOGUE",
    "FLOWS",
    "CallRecord",
    "LEVELS",
    "Classification",
    "SectionTag",
    "classify",
    "options_block",
    "select_sections",
]
