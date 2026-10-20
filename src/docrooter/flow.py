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

**Results log.** ``log=path`` appends one :class:`~docrooter.results_log.ResultsLog` row per
backend decision (each section step and each document call), with the level, the view hash, the
options, the probabilities, the tokens and, from the call record, the prompt hash.
"""

from __future__ import annotations

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
from docrooter.classify.classifier import DEFAULT_BACKEND, Classifier
from docrooter.classify.documents import classify_document
from docrooter.classify.engine import backend_temperature, score_batch
from docrooter.classify.normalize import (
    CUSTOM_TAXONOMY,
    DOC_TYPES,
    _section_taxonomy_name,
    section_classifier,
)
from docrooter.classify.results import DocTypeResult
from docrooter.classify.strategies import Flat, Greedy
from docrooter.document import Document, Section
from docrooter.ladder import LEVELS, Level, check_level_and_view, next_level, section_view
from docrooter.results_log import ResultsLog, secrets_of, text_hash
from docrooter.rules import DocTypeRules, doc_type_rules, match_rules, unknown_section_ids
from docrooter.taxonomies import builtin_taxonomy
from docrooter.taxonomy import Taxonomy
from docrooter.views import SectionView, estimate_tokens

Flow = Literal["sections_first", "doc_first"]
FLOWS: tuple[str, ...] = ("sections_first", "doc_first")
DecidedBy = Literal["rules", "model", "fallback"]
Flag = Literal["no_sections", "rules_unmet", "near_miss", "excludes_waived", "truncated"]
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
    was assigned) with the label's ``confidence`` (the path probability), the ``level_used`` for
    the final answer and the ``tokens`` of every attempt."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    level: int
    label: str | None
    confidence: float = 0.0
    level_used: Level
    tokens: TokenUsage = Field(default_factory=TokenUsage)


class Classification(BaseModel):
    """What :func:`classify` decided, how, and at what token cost.

    - ``doc_type`` (a doc-type leaf id, ``None`` when nothing could be classified) with its
      ``doc_type_confidence`` and ``doc_type_probs`` (the most probable types offered, most
      probable first; empty when the rules decided). A type decided by rules gets the lowest
      confidence among its supporting section tags.
    - ``decided_by``: ``"rules"`` (no document-level call), ``"model"`` (the model chose among
      the narrowed ``candidates``, or among every type in ``doc_first``) or ``"fallback"`` (the
      model chose among every type because the rules left fewer than two, or there were no
      sections).
    - ``candidates``: the doc types the rules left, in the order offered (most ``typical``
      matches first), near misses included when they were offered; empty in ``doc_first``.
    - ``flags``: ``no_sections``, ``rules_unmet``, ``near_miss``, ``excludes_waived`` and
      ``truncated`` (a view was cut to its budget).
    - ``sections``: the tags, in the order tagged (level 1 first, then level 2, each in reading
      order), against ``section_taxonomy`` (``"section_catalogue"`` in ``sections_first``).
    - ``tokens``: everything spent, ``stage_tokens["sections"] + stage_tokens["document"]``.
    - ``levels``: the ``section`` level asked for and the ``document`` level of the final
      document decision (``None`` when no document call was made).
    - ``backend``: the backend's name.
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
    section_taxonomy: str | None = None
    sections: list[SectionTag] = Field(default_factory=list)
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    stage_tokens: dict[str, TokenUsage] = Field(default_factory=dict)
    levels: dict[str, Level | None] = Field(default_factory=dict)
    backend: str


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
) -> Classification:
    """Classify a document's sections and type with the minimum context (see the module docs).

    ``source_or_doc`` is a :class:`Document` or anything :func:`~docrooter.load` reads.
    ``section_level`` and ``doc_level`` default to ``level``. At most ``max_sections`` sections
    are tagged: level 1 first, then level 2, each in reading order. ``rule_min_conf`` is the
    least confidence a tag needs to count for a rule, ``decide_conf`` the least every supporting
    tag needs for the rules to decide alone. ``cascade`` and ``threshold`` control escalation.

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
        run = _Run(
            doc=doc,
            flow=flow,
            metered=_Metered(inner),
            doc_types=doc_type_taxonomy,
            cascade=cascade,
            threshold=threshold,
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
        for i, text in enumerate(texts):
            row = tuple(float(p) for p in probs[i])
            self.calls.append(_Call(text, options, row, usages[i], by_index.get(i)))
        return scores

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

    return [
        TokenUsage(
            share(total.prompt, i), share(total.completion, i), share(total.calls, i), total.exact
        )
        for i in range(n)
    ]


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


@dataclass
class _Decision:
    """A backend call as a log row's context: the stage, the section and the level."""

    stage: str
    section_id: str | None
    level: Level
    call: _Call


@dataclass
class _DocDecision:
    result: DocTypeResult
    level: Level
    tokens: TokenUsage


@dataclass
class _Run:
    doc: Document
    flow: Flow
    metered: _Metered
    doc_types: Taxonomy
    cascade: bool
    threshold: float
    truncated: bool = False
    decisions: list[_Decision] = field(default_factory=list)

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

        tags = self.tag(selected, section_level, tagger, catalogue_name)
        evidence: dict[str, float] = {}
        for tag in tags:
            if tag.label is not None:
                evidence[tag.label] = max(evidence.get(tag.label, 0.0), tag.confidence)
        match = match_rules(rules, evidence, min_conf=rule_min_conf, decide_conf=decide_conf)
        flags: list[Flag] = ["excludes_waived"] if match.excludes_waived else []
        if match.decided is not None:
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
        if len(offered) <= 1 and match.near_misses:
            offered += [t for t in match.near_misses if t not in offered]
            flags.append("near_miss")
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

        tags = self.tag(select_sections(self.doc, max_sections), section_level, tagger, name)
        return self.result(decision, "model", [], [], name, tags, section_level)

    # --- stages

    def tag(
        self,
        sections: list[Section],
        level: Level,
        tagger: Callable[[SectionView], Classifier],
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
        count = classifier._token_counter()
        rendered = view.render_all(self.doc, [s.id for s in sections], count)
        blank = [s.id for s in sections if not s.title.strip()]
        if blank and not _BODY_FIELDS.intersection(view.fields):
            # a section with no title still shows something: its lead
            with_lead = view.model_copy(update={"fields": (*view.fields, "lead")})
            rendered |= with_lead.render_all(self.doc, blank, count)
        texts = [rendered[s.id].text for s in sections]
        self.truncated = self.truncated or any(rendered[s.id].truncated for s in sections)
        mark = len(self.metered.calls)
        results = classifier.classify_texts(texts)
        owners = _owners(texts, self.metered.calls[mark:])
        for call, index in owners:
            sid = sections[index].id if index is not None else None
            self.decisions.append(_Decision(SECTION_STAGE, sid, level, call))
            if index is not None:
                tag = tags[sections[index].id]
                tag.tokens = tag.tokens + call.usage
        for section, result in zip(sections, results, strict=True):
            tag = tags[section.id]
            best = result.best
            assigned = best is not None and bool(best.node_ids)
            tag.label = best.node_ids[-1] if assigned and best is not None else None
            tag.confidence = float(best.score) if assigned and best is not None else 0.0
            tag.level = level

    def decide_document(self, taxonomy: Taxonomy, level: Level) -> _DocDecision:
        """The document's type among ``taxonomy``'s leaves, asked again one rung up while
        ``cascade`` allows and the confidence is below ``threshold``."""
        current: Level | None = level
        tokens = TokenUsage()
        result: DocTypeResult | None = None
        used = level
        while current is not None:
            mark = len(self.metered.calls)
            result = classify_document(
                self.doc, taxonomy, backend=self.metered, level=current, strategy=Flat()
            )
            used = current
            for call in self.metered.calls[mark:]:
                self.decisions.append(_Decision(DOCUMENT_STAGE, None, current, call))
                tokens = tokens + call.usage
            self.truncated = self.truncated or result.truncated
            if (
                not self.cascade
                or result.confidence >= self.threshold
                or result.best.stopped_reason == "empty_text"
            ):
                break
            current = next_level(current)
        assert result is not None
        return _DocDecision(result, used, tokens)

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
        section_tokens = _sum([t.tokens for t in tags])
        document_tokens = decision.tokens if decision is not None else TokenUsage()
        if self.truncated:
            flags = [*flags, "truncated"]
        return Classification(
            source=self.doc.source,
            flow=self.flow,
            doc_type=doc_type[0] if doc_type else None,
            doc_type_confidence=doc_type[1] if doc_type else 0.0,
            doc_type_probs=probs,
            decided_by=decided_by,
            candidates=candidates,
            flags=flags,
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
        )

    def write_log(self, path: str | os.PathLike[str], backend: Backend) -> None:
        """One results-log row per backend decision, in the order they were made."""
        log = ResultsLog(Path(path), secrets=secrets_of(backend))
        source = Path(self.doc.source).name
        for decision in self.decisions:
            call, record = decision.call, decision.call.record or {}
            best = int(np.argmax(call.probs)) if call.probs else None
            log.write(
                {
                    "source": source,
                    "flow": self.flow,
                    "stage": decision.stage,
                    "section_id": decision.section_id,
                    "level": decision.level,
                    "view_hash": text_hash(call.text),
                    "options": list(call.options),
                    "label": call.options[best] if best is not None else None,
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
    "CATALOGUE",
    "FLOWS",
    "LEVELS",
    "Classification",
    "SectionTag",
    "classify",
    "options_block",
    "select_sections",
]
