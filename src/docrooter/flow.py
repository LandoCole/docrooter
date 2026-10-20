"""The v1 headline API: :func:`classify` a document's sections and type at a context level.

Two flows (spec G3):

- **sections_first** (the default) tags the document's level-1 and level-2 sections against the
  shared section catalogue (``builtin_taxonomy("section_catalogue")``), group first, then leaf
  (``Greedy``). It matches the doc-type section rules (:func:`~docrooter.rules.match_rules`)
  against the tags, then decides the type:

  - **by rules**, with no document-level call, when a single candidate's supporting tags are all
    at least ``decide_conf``;
  - **by the model** among the candidates only, plus the near misses when there is no candidate
    or a single undecided one (flagged ``near_miss``);
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
until ``full``; only those items are asked again, and every attempt's tokens count.

**Tokens.** Every backend call is metered. A backend with a per-call record sink (``on_call``,
the OpenAI-compatible backend) gives each call's exact tokens from the API's ``usage``. One that
only reports a cumulative ``usage()`` has each call's change split over its texts. Any other
backend (keyword, embedding, NLI) is charged what the same view and options would cost an LLM,
``estimate_tokens(view) + estimate_tokens(options block)`` per text, marked estimated.
``Classification.tokens`` is exact only when every call was.

**Calls.** ``calls="batched"`` (the default for a backend that can, the OpenAI-compatible one)
asks a few chat calls per stage, each option listed once (:mod:`docrooter.batch`), in batches of
up to 40 sections. The sections stage takes two steps: one call sorts every section into a
catalogue group (the groups listed with a one-line gloss), then one call per group that received
sections chooses among that group's leaves only. ``calls="batched_flat"`` lists every leaf in
one call instead (kept for the benchmark comparison). The document stage is one call among the
offered types. A batched answer has confidence 1.0 (``confidence_source="batched"``); an id that
is not an option is asked again once, in one call for the failures only, then tagged
``general.other`` (or the group's ``other`` leaf) with confidence 0 and flagged ``unparsed``.
``calls="per_item"`` asks one item at a time, with probabilities; ``cascade`` needs it. A
backend that cannot batch always asks per item, but its token estimate follows the ``calls``
pattern (two-step batched unless told otherwise), since that is what an LLM would be sent.
``Classification.tokens_split`` estimates how much of the prompt was options, content and
instructions.

**The rule gate with batched calls.** Batched tags all carry confidence 1.0, so the
``decide_conf`` gate cannot tell strong evidence from weak. With batched calls a lone candidate
is decided by rules only when at least two tagged sections, in total, support its required
groups. With fewer it takes the ``low_support`` path: the candidate and its doc-type siblings
are offered to the model.

**Results log.** ``log=path`` appends one :class:`~docrooter.results_log.ResultsLog` row per
backend decision (each section step and each document call), with the level, the view hash, the
options, the probabilities, the tokens and, from the call record, the prompt hash.
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
    close_backend,
    to_probs,
    usage_of,
)
from docrooter.batch import (
    DOC_TYPE_KEY,
    BatchPrompt,
    chunks,
    document_prompt,
    group_leaves,
    group_prompt,
    reply_tokens,
    section_prompt,
    section_reply_max_tokens,
    valid_id,
)
from docrooter.classify.classifier import DEFAULT_BACKEND, Classifier
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
    probability of a per-item answer (``"probability"``), or 1.0 for a valid batched answer and
    0 for an unparsed one (``"batched"``)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    level: int
    label: str | None
    confidence: float = 0.0
    level_used: Level
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    confidence_source: Literal["probability", "batched"] = "probability"


class CallRecord(BaseModel):
    """One document-stage attempt: the ``level`` of its view, the ``prompt_tokens`` and
    ``completion_tokens`` it spent (``exact`` when from the API's usage), whether its view was
    ``truncated``, and how many doc types it offered (``options_count``)."""

    model_config = ConfigDict(extra="forbid")

    level: Level
    prompt_tokens: int
    completion_tokens: int
    exact: bool
    truncated: bool
    options_count: int


class Classification(BaseModel):
    """What :func:`classify` decided, how, and at what token cost.

    - ``doc_type`` (a doc-type leaf id, ``None`` when nothing could be classified) with its
      ``doc_type_confidence`` and ``doc_type_probs`` (the most probable types offered, most
      probable first; empty when the rules decided). A type decided by rules gets the lowest
      confidence among its supporting section tags.
    - ``decided_by``: ``"rules"`` (no document-level call), ``"model"`` (the model chose among
      the narrowed ``candidates``, or among every type in ``doc_first``) or ``"fallback"`` (the
      model chose among every type: nothing was left to offer, a lone type had no siblings, or
      there were no sections).
    - ``candidates``: the doc types offered, in order (most ``typical`` matches first): the
      rules' candidates, near misses when they were offered, and a lone type's siblings
      (``low_support``); empty in ``doc_first``.
    - ``flags``: ``no_sections``, ``rules_unmet``, ``near_miss``, ``low_support`` (a lone
      undecided candidate or near miss, offered with its doc-type siblings), ``excludes_waived``
      ``truncated`` (a section or document view was cut to its budget) and ``unparsed`` (a
      batched answer named no option, twice).
    - ``document_calls``: one :class:`CallRecord` per document-stage attempt (cascade attempts
      included), in order; the last made the decision. ``doc_truncated``: whether the deciding
      document view was cut.
    - ``sections``: the tags, in the order tagged (level 1 first, then level 2, each in reading
      order), against ``section_taxonomy`` (``"section_catalogue"`` in ``sections_first``).
    - ``tokens``: everything spent, ``stage_tokens["sections"] + stage_tokens["document"]``.
    - ``levels``: the ``section`` level asked for and the ``document`` level of the final
      document decision (``None`` when no document call was made).
    - ``backend``: the backend's name; ``calls``: how it was asked, ``"batched"`` or
      ``"per_item"``.
    - ``tokens_split``: the estimated prompt tokens of the ``options``, the ``content`` (the
      views) and the ``instructions``. For an estimated run they sum to ``tokens.prompt``; next
      to exact totals they show the share the options take.
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    flow: Flow
    doc_type: str | None
    doc_type_confidence: float = 0.0
    doc_type_probs: dict[str, float] = Field(default_factory=dict)
    decided_by: DecidedBy
    candidates: list[str] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)
    document_calls: list[CallRecord] = Field(default_factory=list)
    doc_truncated: bool = False
    section_taxonomy: str | None = None
    sections: list[SectionTag] = Field(default_factory=list)
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    stage_tokens: dict[str, TokenUsage] = Field(default_factory=dict)
    levels: dict[str, Level | None] = Field(default_factory=dict)
    backend: str
    calls: Calls = "per_item"
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
) -> Classification:
    """Classify a document's sections and type with the minimum context (see the module docs).

    ``source_or_doc`` is a :class:`Document` or anything :func:`~docrooter.load` reads.
    ``section_level`` and ``doc_level`` default to ``level``. At most ``max_sections`` sections
    are tagged: level 1 first, then level 2, each in reading order. ``rule_min_conf`` is the
    least confidence a tag needs to count for a rule, ``decide_conf`` the least every supporting
    tag needs for the rules to decide alone. ``cascade`` and ``threshold`` control escalation
    (per-item calls only). ``calls`` is ``"batched"`` (two-step sections), ``"batched_flat"``
    or ``"per_item"``; ``None`` batches when the backend can (see the module docs).

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
            batch_estimate=estimating and calls != "per_item",
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
    options_tokens: int = 0


class _Metered:
    """A backend wrapper that records every text it scores with the tokens it cost.

    It keeps the wrapped backend's name and temperature, so the classifier treats it exactly as
    the backend itself; other attributes are passed through.
    """

    def __init__(self, backend: Backend) -> None:
        self.inner = backend
        self.name = backend.name
        self.max_candidates = backend.max_candidates
        self.temperature = backend_temperature(backend)
        self.calls: list[_Call] = []

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
        usages = self._usages(texts, candidates, by_index, before, after)
        probs = to_probs(scores, temperature=self.temperature, multi_label=multi_label)
        options = tuple(c.node_id for c in candidates)
        options_tokens = estimate_tokens(options_block(candidates))
        for i, text in enumerate(texts):
            row = tuple(float(p) for p in probs[i])
            call = _Call(text, options, row, usages[i], by_index.get(i), options_tokens)
            self.calls.append(call)
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
            usage = TokenUsage(prompt.estimate(), 0, 1, False)
        return reply, usage, records[-1] if records else None

    @staticmethod
    def _usages(
        texts: list[str],
        candidates: Sequence[Candidate],
        records: Mapping[Any, Mapping[str, Any]],
        before: TokenUsage | None,
        after: TokenUsage | None,
    ) -> list[TokenUsage]:
        """Each text's tokens: its call record, else a share of the usage change, else an
        estimate of what an LLM would be sent."""
        if records:
            return [
                _record_usage(records[i]) if i in records else _estimate(text, candidates)
                for i, text in enumerate(texts)
            ]
        if before is not None and after is not None:
            return _split(after - before, len(texts))
        return [_estimate(text, candidates) for text in texts]


def _record_usage(record: Mapping[str, Any]) -> TokenUsage:
    return TokenUsage(
        int(record.get("prompt_tokens") or 0),
        int(record.get("completion_tokens") or 0),
        1,
        bool(record.get("exact")),
    )


def _estimate(text: str, candidates: Sequence[Candidate]) -> TokenUsage:
    """What one LLM call on ``text`` with these options would be sent, estimated."""
    prompt = estimate_tokens(text) + estimate_tokens(options_block(candidates))
    return TokenUsage(prompt, 0, 1, False)


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
    """``total`` shared over ``n`` texts, remainders to the first ones."""

    def share(value: int, i: int) -> int:
        return value // n + (1 if i < value % n else 0)

    out = []
    for i in range(n):
        estimated = share(total.estimated_calls, i)
        out.append(
            TokenUsage(
                share(total.prompt, i),
                share(total.completion, i),
                share(total.calls, i),
                estimated == 0,
                estimated,
            )
        )
    return out


def _sum(usages: Sequence[TokenUsage]) -> TokenUsage:
    return sum(usages, TokenUsage())


# --- the flows ----------------------------------------------------------------------------


@dataclass
class _Tag:
    section: Section
    label: str | None = None
    confidence: float = 0.0
    level: Level = "terms"
    tokens: TokenUsage = field(default_factory=TokenUsage)
    source: Literal["probability", "batched"] = "probability"


@dataclass
class _Decision:
    """A backend call as a log row's context: the stage, the section and the level."""

    stage: str
    step: Step
    section_id: str | None
    level: Level
    call: _Call
    label: Any = None  # a batched call's answer; else the argmax option


@dataclass
class _DocDecision:
    result: DocTypeResult
    level: Level
    tokens: TokenUsage
    calls: list[CallRecord]


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
    truncated: bool = False
    unparsed: bool = False
    decisions: list[_Decision] = field(default_factory=list)
    unowned: TokenUsage = field(default_factory=TokenUsage)  # section calls no text claimed
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
        weak = (
            self.batched
            and match.decided is not None
            and _support_count(rules[match.decided], tags, rule_min_conf) < BATCHED_SUPPORT
        )
        if match.decided is not None and not weak:
            confidence = min(match.support[match.decided].values())
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
        if weak:  # batched tags: one section is too little to decide on
            flags.append("low_support")
            offered += _siblings(self.doc_types, offered[0])
        elif len(offered) <= 1 and match.near_misses:
            offered += [t for t in match.near_misses if t not in offered]
            flags.append("near_miss")
        if len(offered) == 1 and not weak:
            # a lone undecided candidate or near miss: offer it with its doc-type siblings
            flags.append("low_support")
            offered += _siblings(self.doc_types, offered[0])
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
        return self.result(decision, "model", [], [], name, tags, section_level)

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

    def _tag_once(
        self,
        sections: list[Section],
        level: Level,
        tagger: Callable[[SectionView], Classifier],
        tags: dict[str, _Tag],
    ) -> None:
        view = section_view(level)
        classifier = tagger(view)
        texts = self._render(sections, view, classifier._token_counter())
        mark = len(self.metered.calls)
        results = classifier.classify_texts(texts)
        owners = _owners(texts, self.metered.calls[mark:])
        leaves = {n.id for n in classifier.taxonomy.leaves()}
        spent: dict[int, TokenUsage] = {}
        for call, index in owners:
            sid = sections[index].id if index is not None else None
            step: Step = "leaf" if set(call.options) <= leaves else "group"
            self.decisions.append(_Decision(SECTION_STAGE, step, sid, level, call))
            if not self.batch_estimate:
                self._add_split(call)
            if index is not None:
                spent[index] = spent.get(index, TokenUsage()) + call.usage
            elif not self.batch_estimate:  # still spent: in the stage, if not in one section
                self.unowned = self.unowned + call.usage
        for section, result in zip(sections, results, strict=True):
            tag = tags[section.id]
            best = result.best
            assigned = best is not None and bool(best.node_ids)
            tag.label = best.node_ids[-1] if assigned and best is not None else None
            tag.confidence = float(best.score) if assigned and best is not None else 0.0
            tag.level = level
        if self.batch_estimate:
            labels = [tags[s.id].label for s in sections]
            spent = self._batch_estimate(texts, labels, classifier.taxonomy)
        for index, usage in spent.items():
            tag = tags[sections[index].id]
            tag.tokens = tag.tokens + usage

    def _render(
        self, sections: list[Section], view: SectionView, count: Callable[[str], int]
    ) -> list[str]:
        """The sections' views; a section with no title still shows its lead."""
        rendered = view.render_all(self.doc, [s.id for s in sections], count)
        blank = [s.id for s in sections if not s.title.strip()]
        if blank and not _BODY_FIELDS.intersection(view.fields):
            with_lead = view.model_copy(update={"fields": (*view.fields, "lead")})
            rendered |= with_lead.render_all(self.doc, blank, count)
        self.truncated = self.truncated or any(rendered[s.id].truncated for s in sections)
        return [rendered[s.id].text for s in sections]

    def _add_split(self, call: _Call) -> None:
        self.split["options"] += call.options_tokens
        self.split["content"] += estimate_tokens(call.text)

    def _add_parts(self, prompt: BatchPrompt) -> None:
        for part, tokens in prompt.parts().items():
            self.split[part] += tokens

    def _two_step(self, taxonomy: Taxonomy) -> bool:
        """Whether the batched sections stage takes two steps: not flat, and the taxonomy has
        groups."""
        return not self.flat and any(taxonomy.children(n.id) for n in taxonomy.children())

    def _batch_estimate(
        self, texts: list[str], labels: list[str | None], taxonomy: Taxonomy
    ) -> dict[int, TokenUsage]:
        """Each section's share of the batched calls an LLM would have been sent for
        ``texts`` (with the answers ``labels``): its own view, plus an even share of each call's
        options and instructions."""
        spent: dict[int, TokenUsage] = {i: TokenUsage() for i in range(len(texts))}
        everyone = list(range(len(texts)))
        asks: list[tuple[list[int], str | None, list[str | None]]]
        if not self._two_step(taxonomy):
            asks = [(everyone, None, labels)]
        else:
            groups = [_group_of(taxonomy, label) for label in labels]
            asks = [(everyone, "groups", groups)]
            for group in dict.fromkeys(g for g in groups if g):
                members = [i for i in everyone if groups[i] == group]
                asks.append((members, group, labels))
        for members, step, answers in asks:
            for chunk in chunks(members):
                views = {str(i + 1): texts[i] for i in chunk}
                if step == "groups":
                    prompt = group_prompt(views, taxonomy)
                elif step is None:
                    prompt = section_prompt(views, taxonomy)
                else:
                    prompt = section_prompt(views, taxonomy, group_leaves(taxonomy, step))
                self._add_parts(prompt)
                parts = prompt.parts()
                overhead = parts["instructions"] + parts["options"]
                completion = reply_tokens({str(i + 1): answers[i] or "" for i in chunk})
                shares = _split(TokenUsage(overhead, completion, 1, False), len(chunk))
                for item, i, share in zip(prompt.items, chunk, shares, strict=True):
                    # each section is one estimated part of the call (see TokenUsage)
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
        texts = self._render(sections, section_view(level), count)
        views = {str(i + 1): text for i, text in enumerate(texts)}
        ids = {str(i + 1): s.id for i, s in enumerate(sections)}
        spent = {n: TokenUsage() for n in views}
        fallback = UNPARSED_LABEL if UNPARSED_LABEL in taxonomy else None
        labels: dict[str, str | None]
        fallbacks: dict[str, str | None]
        if not self._two_step(taxonomy):
            leaf_ids = [leaf.id for leaf in taxonomy.leaves()]
            labels = self._ask(
                views, ids, spent, level, "batch", leaf_ids,
                lambda v: section_prompt(v, taxonomy),
            )  # fmt: skip
            fallbacks = dict.fromkeys(views, fallback)
        else:  # the groups first, then each group's leaves
            group_ids = [node.id for node in taxonomy.children()]
            groups = self._ask(
                views, ids, spent, level, "batch_group", group_ids,
                lambda v: group_prompt(v, taxonomy),
            )  # fmt: skip
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
                labels.update(
                    self._ask(
                        members, ids, spent, level, "batch_leaf", leaf_ids,
                        functools.partial(section_prompt, taxonomy=taxonomy, leaf_ids=leaf_ids),
                    )
                )  # fmt: skip
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
        ids: Mapping[str, str],
        spent: dict[str, TokenUsage],
        level: Level,
        step: Step,
        options: list[str],
        prompt_for: Callable[[dict[str, str]], BatchPrompt],
    ) -> dict[str, str | None]:
        """One batched step over ``views`` (by section number), in calls of at most 40: each
        section's answer among ``options``, ``None`` when it named none twice. Each call's
        tokens are shared among its sections in ``spent``."""
        valid = {option.lower(): option for option in options}
        answers: dict[str, str | None] = dict.fromkeys(views)
        for chunk in chunks(list(views)):
            pending = list(chunk)
            for _ in range(2):  # the first ask, then once more for the failures
                if not pending:
                    break
                prompt = prompt_for({n: views[n] for n in pending})
                reply, usage, record = self._chat(prompt, options, len(pending))
                for n in pending:
                    answers[n] = valid_id((reply or {}).get(n), valid)
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
        while current is not None:
            mark = len(self.metered.calls)
            result = classify_document(
                self.doc, taxonomy, backend=self.metered, level=current, strategy=Flat()
            )
            used = current
            attempt = TokenUsage()
            for call in self.metered.calls[mark:]:
                self.decisions.append(_Decision(DOCUMENT_STAGE, "document", None, current, call))
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
        return _DocDecision(result, used, tokens, records)

    def _decide_batched(self, taxonomy: Taxonomy, level: Level) -> _DocDecision:
        """The document's type among ``taxonomy``'s leaves in one chat call (asked once more
        when the reply names no option)."""
        if not _has_content(self.doc):
            empty = PathResult(node_ids=[], stopped_reason="empty_text")
            return _DocDecision(
                DocTypeResult(source=self.doc.source, best=empty), level, TokenUsage(), []
            )
        count = self.metered.tokenizer() or estimate_tokens
        view = document_view(level).render(self.doc, count)
        self.truncated = self.truncated or view.truncated
        leaf_ids = [leaf.id for leaf in taxonomy.leaves()]
        valid = {leaf_id.lower(): leaf_id for leaf_id in leaf_ids}
        prompt = document_prompt(view.text, taxonomy)
        tokens, records, chosen = TokenUsage(), [], None
        for _ in range(2):
            reply, usage, record = self._chat(prompt, leaf_ids, 1)
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
        doc_type: tuple[str, float] | None = None,
    ) -> Classification:
        probs: dict[str, float] = {}
        if decision is not None:
            best = decision.result.best
            doc_type = (best.node_ids[-1], decision.result.confidence) if best.node_ids else None
            probs = dict(decision.result.leaf_probs)
        section_tokens = _sum([t.tokens for t in tags]) + self.unowned
        document_tokens = decision.tokens if decision is not None else TokenUsage()
        if self.truncated:
            flags = [*flags, "truncated"]
        if self.unparsed:
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
            doc_truncated=decision.result.truncated if decision is not None else False,
            section_taxonomy=section_taxonomy,
            sections=[
                SectionTag(
                    id=t.section.id,
                    title=t.section.title,
                    level=t.section.level,
                    label=t.label,
                    confidence=t.confidence,
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
                "document": decision.level if decision is not None else None,
            },
            backend=self.metered.name,
            calls=("batched_flat" if self.flat else "batched") if self.batched else "per_item",
            tokens_split=dict(self.split),
        )

    def write_log(self, path: str | os.PathLike[str], backend: Backend) -> None:
        """One results-log row per backend decision, in the order they were made."""
        log = ResultsLog(Path(path), secrets=secrets_of(backend))
        source = Path(self.doc.source).name
        for decision in self.decisions:
            call, record = decision.call, decision.call.record or {}
            best = int(np.argmax(call.probs)) if call.probs else None
            label = decision.label if decision.label is not None or not call.probs else None
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
                    "label": label if best is None else call.options[best],
                    "probs": dict(zip(call.options, call.probs, strict=True)),
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


def _support_count(rule: DocTypeRules, tags: Sequence[_Tag], min_conf: float) -> int:
    """How many tagged sections support any of ``rule``'s required groups."""
    ids = [rule_id for group in rule.requires for rule_id in group]
    return sum(
        1
        for tag in tags
        if tag.label is not None
        and tag.confidence >= min_conf
        and any(covers(rule_id, tag.label) for rule_id in ids)
    )


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
