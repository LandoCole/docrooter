"""Corpus labelling: TELEClass step 1 over the corpus store (spec §7.3–§7.4, §4.0 rulings 5–6).

:func:`label_corpus` labels a stratified sample of the ``train`` and ``calib`` elements:

- **topic:** the embedder shortlists the top 5 leaves of the document family's topic
  taxonomy, plus ``other`` (:mod:`docrooter.teach.shortlist`). The teacher then chooses among
  them with the family's cached prefix, which letters every leaf of the family; each element's
  turn names only its shortlisted letters (ruling 5);
- **function:** on a deterministic ``functions_share`` subsample, the teacher answers a yes/no
  question for each of the element's top 3 functions by prescreen. Items are grouped by
  question, so each question's prefix is prefilled once per batch;
- **weak labels** (:mod:`docrooter.teach.weak`) and the **prescreen scores** of all 12
  functions are recorded alongside, under their own labellers.

Labellers (``labels/<name>.jsonl`` in the store):

- ``teacher`` (:data:`TEACHER_LABELER`): facet ``topic`` (probabilities over the shortlist)
  and facet ``function`` (P(yes) for each asked tag; an element outside the subsample has no
  function row). Provenance: the label key, teacher name (engine, model, revision, dtype),
  embedder, prompt hash, taxonomy hash, timestamp, and the shortlist or the asked tags;
- ``prescreen`` (:data:`PRESCREEN_LABELER`): facet ``function``, every function tag's
  prescreen score: the percentile of the element's cosine to the tag among the whole
  sample's cosines to it, independent of the other tags (ruling 6's implicit negatives are
  tags scoring < 0.2). The raw cosines are in the provenance, and a resume reuses them;
- ``weak`` (:data:`WEAK_LABELER`): facets ``section_norm`` and ``edgar_item``, one leaf each,
  with the ``source`` that produced it.

**Never gold, except in full mode.** :class:`SampleSpec` refuses the ``gold`` split, and the
sampler drops any document whose manifest split is ``gold``. The one exception is
``SampleSpec(mode="full")`` (v0.6 spec §0 ruling 2): it labels the topic of EVERY content
element of the named splits' documents (calib and gold), with no function questions, through
the same topic prompt, shortlist and scoring as the normal mode, into the separate labeller
``teacher-full`` (:data:`TEACHER_FULL_LABELER`). Those labels measure recall and calibrate the
dial; v0.5 training reads only ``teacher`` and ``prescreen``, never ``teacher-full``. A full
run writes no weak or prescreen rows (the training chain reads those files).

**Resume and cache (Review Focus 2 and 5).** Each teacher row carries its
:func:`~docrooter.teach.label_key`: element, facet, teacher name, prompt hash (with the
shortlist settings and embedder folded in) and taxonomy hash. An element is scored again only
when its row is missing or its key differs, so an interrupted run resumes where it stopped and
a changed prompt, taxonomy, teacher or embedder re-scores. A batch's rows are appended in one
write after the batch, the teacher's last: the store's torn-line handling makes a killed run
lose at most the batch in flight. A resume does not append the weak and prescreen rows of an
element again while its prescreen row is current (same embedder, taxonomies and versions).

**Ceiling.** The wall clock is checked before each batch; at ``ceiling_hours`` the run stops
and reports it. It is the only limit on the run (§4.0 ruling 5 replaced the old 2 items/s
rule). The sample is ordered so that any prefix of it is stratified: families take
turns, documents take turns within a family, and each document's elements come in
kind-stratified, spread-out order (:func:`sample_elements`).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, get_args

import numpy as np
from pydantic import BaseModel, Field

from docrooter.backends.base import close_backend
from docrooter.backends.loading import release_memory
from docrooter.corpus import SKIPPED_KINDS, SPLITS, Corpus, CorpusElement, LabelRow
from docrooter.corpus.store import LabelKey, ManifestRecord
from docrooter.elements.models import ElementKind
from docrooter.taxonomies import ELEMENT_FAMILIES, builtin_taxonomy
from docrooter.taxonomy import Taxonomy
from docrooter.teach import prompts
from docrooter.teach.base import TeacherEngine, TeachItem, label_key
from docrooter.teach.shortlist import (
    FUNCTION_INSTRUCTION,
    MAX_QUERY_CHARS,
    PRESCREEN_K,
    SHORTLIST_K,
    TOPIC_INSTRUCTION,
    doc_family,
    element_text,
    family_options,
    function_cosines,
    function_tags,
    percentile_scores,
    shortlist_scores,
    top_functions,
)
from docrooter.teach.weak import weak_labels

__all__ = [
    "FUNCTION_FACET",
    "PRESCREEN_LABELER",
    "TEACHER_FULL_LABELER",
    "TEACHER_LABELER",
    "TOPIC_FACET",
    "WEAK_LABELER",
    "LabelReport",
    "Pick",
    "SampleSpec",
    "in_function_sample",
    "label_corpus",
    "sample_elements",
]

TEACHER_LABELER = "teacher"
TEACHER_FULL_LABELER = "teacher-full"
"""Full-mode topic labels of calib and gold documents: never read by v0.5 training."""
SAMPLE_MODE = "sample"
FULL_MODE = "full"
PRESCREEN_LABELER = "prescreen"
WEAK_LABELER = "weak"
TOPIC_FACET = "topic"
FUNCTION_FACET = "function"

TOPIC_TEACHER_INSTRUCTION = (
    "You label the elements of a {family}, one at a time. Choose the section topic that the "
    "element belongs to, judging by its content and its section headings."
)
SHORTLIST_VERSION = "shortlist-v2"  # bump when the embedder's input or scoring changes
PRESCREEN_SCORE = "percentile"  # per-tag percentile of the cosine among the sample
WEAK_VERSION = "weak-v1"  # bump when weak.py's matching changes
# Teacher text is pre-cut to this many characters before the engine cuts it to 384 tokens, so
# a huge element is never tokenised whole; 12 characters a token keeps the full 384 tokens of
# any ordinary text.
TEACHER_MAX_CHARS = 12 * prompts.MAX_TEXT_TOKENS

_KIND_ORDER = {kind: i for i, kind in enumerate(get_args(ElementKind))}


# --- the sample -------------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleSpec:
    """What :func:`label_corpus` labels (spec §7.4, §4.0 ruling 5).

    Normal mode (``"sample"``): at most ``per_doc`` elements per document, ``target`` in all if
    the ceiling allows, and at least ``floor`` (the report flags a run that stopped below it);
    ``splits`` never includes ``gold``.

    Full mode (``"full"``, v0.6 ruling 2): every content element of every document in
    ``splits``, in manifest then reading order; ``per_doc``, ``target`` and ``floor`` do not
    apply. ``gold`` is allowed in this mode only.
    """

    per_doc: int = 60
    target: int = 12000
    floor: int = 6000
    splits: tuple[str, ...] = ("train", "calib")
    mode: str = SAMPLE_MODE

    def __post_init__(self) -> None:
        if self.mode not in (SAMPLE_MODE, FULL_MODE):
            raise ValueError(f"unknown mode {self.mode!r}; use {SAMPLE_MODE!r} or {FULL_MODE!r}")
        if self.per_doc < 1 or self.target < 1:
            raise ValueError("per_doc and target must be at least 1")
        if not 0 <= self.floor <= self.target:
            raise ValueError(f"floor must be within [0, target], got {self.floor}")
        if not self.splits:
            raise ValueError("splits must name at least one split")
        if "gold" in self.splits and self.mode != FULL_MODE:
            raise ValueError(
                "the gold split is never labelled: gold documents are scored only after "
                "their annotations are committed"
            )
        unknown = sorted(set(self.splits) - set(SPLITS))
        if unknown:
            raise ValueError(f"unknown split(s) {unknown}; use train, calib or gold")

    @property
    def full(self) -> bool:
        return self.mode == FULL_MODE


@dataclass(frozen=True)
class Pick:
    """One sampled element: its document's record and family, a table's caption (its
    description, or ``None``), and whether its functions are asked."""

    element: CorpusElement
    record: ManifestRecord
    family: str
    caption: str | None = None
    functions: bool = False


def in_function_sample(doc_id: str, element_id: str, share: float) -> bool:
    """Whether an element's functions are asked: a fixed hash of its id below ``share``, so
    the subsample is the same on every run."""
    digest = hashlib.sha1(f"{doc_id}\0{element_id}".encode()).hexdigest()
    return int(digest[:8], 16) / 0x1_0000_0000 < share


def sample_elements(
    corpus: Corpus, spec: SampleSpec, *, functions_share: float = 0.5
) -> list[Pick]:
    """The elements to label, in labelling order, at most ``spec.target``.

    Within a document (furniture and contents excluded, elements without text skipped), the
    order is stratified by kind: kind ``k`` with ``n_k`` elements takes its ``j``-th turn at
    ``(j + 0.5) / sqrt(n_k)``, so any prefix holds kinds in proportion to the square root of
    their counts (paragraphs lead, rare kinds are not starved), and each kind's elements come
    spread over the document (van der Corput order), not from its start. The first
    ``spec.per_doc`` are the document's picks.

    Across documents, families take turns (in :data:`ELEMENT_FAMILIES` order), and within a
    family its documents take turns (in manifest order), one element each.

    In full mode every content element of every document of ``spec.splits`` is a pick (topic
    only: ``functions=False``), document by document in manifest order, each in reading order.
    """
    if spec.full:
        return _full_picks(corpus, spec)
    records = {r.id: r for r in corpus.manifest() if r.split in spec.splits}
    queues: dict[str, deque[deque[Pick]]] = {}
    for doc_id, elements in _by_document(corpus.iter_elements(split=spec.splits)):
        record = records.get(doc_id)
        if record is None or record.split == "gold":
            continue
        family = doc_family(record)
        picks = _document_picks(elements, record, family, spec.per_doc, functions_share)
        if picks:
            queues.setdefault(family, deque()).append(deque(picks))
    order = [f for f in ELEMENT_FAMILIES if f in queues] + sorted(
        f for f in queues if f not in ELEMENT_FAMILIES
    )
    out: list[Pick] = []
    while len(out) < spec.target and any(queues[f] for f in order):
        for family in order:
            docs = queues[family]
            if not docs:
                continue
            queue = docs.popleft()
            out.append(queue.popleft())
            if queue:
                docs.append(queue)
            if len(out) >= spec.target:
                break
    return out


def _full_picks(corpus: Corpus, spec: SampleSpec) -> list[Pick]:
    """Every content element of the documents of ``spec.splits`` (furniture and contents
    excluded, elements without text skipped), in manifest then reading order."""
    records = {r.id: r for r in corpus.manifest() if r.split in spec.splits}
    out: list[Pick] = []
    for doc_id, elements in _by_document(corpus.iter_elements(split=spec.splits)):
        record = records.get(doc_id)
        if record is None:
            continue
        family = doc_family(record)
        for i, element in enumerate(elements):
            if element.kind in SKIPPED_KINDS or not _has_text(element):
                continue
            caption = _caption(elements, i) if element.kind == "table" else None
            out.append(Pick(element=element, record=record, family=family, caption=caption))
    return out


def _by_document(elements: Iterable[CorpusElement]) -> Iterator[tuple[str, list[CorpusElement]]]:
    """Consecutive elements grouped by document (``iter_elements`` yields them so)."""
    current: str | None = None
    group: list[CorpusElement] = []
    for element in elements:
        if element.doc_id != current:
            if group and current is not None:
                yield current, group
            current, group = element.doc_id, []
        group.append(element)
    if group and current is not None:
        yield current, group


def _has_text(element: CorpusElement) -> bool:
    if element.kind == "table" and element.table_markdown and element.table_markdown.strip():
        return True
    return bool(element.text.strip())


def _caption(elements: Sequence[CorpusElement], index: int) -> str | None:
    """A table's description: the caption right before it, else the one right after."""
    for j in (index - 1, index + 1):
        if 0 <= j < len(elements) and elements[j].kind == "caption" and elements[j].text.strip():
            return elements[j].text.strip()
    return None


def _spread(n: int) -> list[int]:
    """``range(n)`` in van der Corput order: every prefix is spread over the range."""
    seen: set[int] = set()
    out: list[int] = []
    i = 0
    while len(out) < n:
        value, denominator, k = 0.0, 1.0, i
        while k:
            denominator *= 2
            value += (k & 1) / denominator
            k >>= 1
        j = int(value * n)
        if j not in seen:
            seen.add(j)
            out.append(j)
        i += 1
    return out


def _document_picks(
    elements: list[CorpusElement],
    record: ManifestRecord,
    family: str,
    per_doc: int,
    functions_share: float,
) -> list[Pick]:
    by_kind: dict[str, list[int]] = {}
    for i, element in enumerate(elements):
        if element.kind in SKIPPED_KINDS or not _has_text(element):
            continue
        by_kind.setdefault(element.kind, []).append(i)
    turns: list[tuple[float, int, int]] = []
    for kind, indices in by_kind.items():
        weight = math.sqrt(len(indices))
        for turn, position in enumerate(_spread(len(indices))):
            turns.append(((turn + 0.5) / weight, _KIND_ORDER.get(kind, 99), indices[position]))
    turns.sort()
    picks = []
    for _, _, i in turns[:per_doc]:
        element = elements[i]
        picks.append(
            Pick(
                element=element,
                record=record,
                family=family,
                caption=_caption(elements, i) if element.kind == "table" else None,
                functions=in_function_sample(record.id, element.element_id, functions_share),
            )
        )
    return picks


# --- keys and provenance -----------------------------------------------------------------------


def _sha256(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _embedder_id(embedder: Any) -> str:
    """The embedder's name and pinned revision (``unpinned`` when it has none)."""
    name = getattr(embedder, "name", None) or type(embedder).__name__
    revision = getattr(getattr(embedder, "recipe", None), "revision", None)
    return f"{name}@{revision or 'unpinned'}"


def _teacher_parts(name: str) -> dict[str, str | None]:
    """``engine:model@revision#dtype`` split up (an openai name has no revision or dtype)."""
    engine, _, target = name.partition(":")
    target, _, dtype = target.partition("#")
    model, at, revision = target.rpartition("@")
    if not at:
        model, revision = target, ""
    return {"engine": engine, "model": model, "revision": revision or None, "dtype": dtype or None}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class _Prompts:
    """The prompts, cache keys and provenance of one run."""

    def __init__(
        self,
        teacher: TeacherEngine,
        embedder: Any,
        topics: Taxonomy,
        functions: Taxonomy,
        shortlist_k: int,
        prescreen_k: int,
    ) -> None:
        self.teacher = teacher.name
        self.embedder = _embedder_id(embedder)
        self.topics, self.functions = topics, functions
        self.topic_hash = topics.content_hash()
        self.function_hash = functions.content_hash()
        self.shortlist_k, self.prescreen_k = shortlist_k, prescreen_k
        self._families: dict[str, tuple[str, list[tuple[str, str]], str, str]] = {}
        self.questions = [
            (n.id, prompts.function_question(n.label, n.description or ""))
            for n in functions.leaves()
        ]
        function_prompts = [prompts.prompt_hash(q, "") for _, q in self.questions]
        self.function_prompt_hash = _sha256(function_prompts)
        self.function_key_hash = _sha256(
            {
                "prompts": self.function_prompt_hash,
                "embedder": self.embedder,
                "k": prescreen_k,
                "instruction": FUNCTION_INSTRUCTION,
                "version": SHORTLIST_VERSION,
                "score": PRESCREEN_SCORE,
                "max_query_chars": MAX_QUERY_CHARS,
                "max_teacher_chars": TEACHER_MAX_CHARS,
            }
        )
        # what a prescreen row's cosines depend on, and what the element's weak rows do
        self.prescreen_key = _sha256(
            {
                "embedder": self.embedder,
                "taxonomy": self.function_hash,
                "instruction": FUNCTION_INSTRUCTION,
                "version": SHORTLIST_VERSION,
                "score": PRESCREEN_SCORE,
                "max_query_chars": MAX_QUERY_CHARS,
            }
        )
        self.weak_key = _sha256({"taxonomy": self.topic_hash, "version": WEAK_VERSION})
        self.teacher_parts = _teacher_parts(teacher.name)

    def family(self, family: str) -> tuple[str, list[tuple[str, str]], str, str]:
        """``(instruction, options, prompt hash, key hash)`` of a family's topic prompt."""
        if family not in self._families:
            label = self.topics.node(family).label.lower()
            instruction = TOPIC_TEACHER_INSTRUCTION.format(family=label)
            options = family_options(family, self.topics)
            prompt = prompts.prompt_hash(instruction, prompts.options_block(options))
            key_hash = _sha256(
                {
                    "prompt": prompt,
                    "embedder": self.embedder,
                    "k": self.shortlist_k,
                    "instruction": TOPIC_INSTRUCTION,
                    "version": SHORTLIST_VERSION,
                    "max_query_chars": MAX_QUERY_CHARS,
                    "max_teacher_chars": TEACHER_MAX_CHARS,
                }
            )
            self._families[family] = (instruction, options, prompt, key_hash)
        return self._families[family]

    def key(self, pick: Pick, facet: str) -> str:
        element = pick.element
        if facet == TOPIC_FACET:
            prompt_key, taxonomy_hash = self.family(pick.family)[3], self.topic_hash
        else:
            prompt_key, taxonomy_hash = self.function_key_hash, self.function_hash
        return label_key(
            doc_id=element.doc_id,
            element_id=element.element_id,
            text_sha1=element.text_sha1,
            facet=facet,
            teacher=self.teacher,
            prompt_hash=prompt_key,
            taxonomy_hash=taxonomy_hash,
        )

    def provenance(self, key: str, prompt_hash: str, taxonomy_hash: str) -> dict[str, Any]:
        return {
            "label_key": key,
            "teacher": self.teacher,
            **self.teacher_parts,
            "embedder": self.embedder,
            "prompt_version": prompts.PROMPT_VERSION,
            "prompt_hash": prompt_hash,
            "taxonomy_hash": taxonomy_hash,
            "timestamp": _now(),
        }


# --- the run ----------------------------------------------------------------------------------


class LabelReport(BaseModel):
    """What a labelling run did.

    - ``sampled``: elements in the sample; ``skipped``: of those, already labelled under the
      same keys; ``topic_labelled`` / ``function_labelled``: labelled by this run;
      ``labelled_total``: sample elements with a current topic label after the run;
    - ``teacher_items``: teacher forward passes (one per topic choice or yes/no question);
      ``items_per_second``: ``teacher_items / teacher_seconds``;
    - ``embedded``: sample elements whose function cosines were computed (the others' came
      from their prescreen rows); ``prescreen_rows`` / ``weak_labels``: rows written;
    - ``stopped_by_ceiling``: the wall clock reached ``ceiling_hours`` before the sample was
      done; ``below_floor``: fewer than ``floor`` elements are labelled;
    - full mode: ``mode`` is ``full`` and ``labeler`` ``teacher-full``; ``sampled`` is the
      element count, ``seconds`` the measured wall time of the pass, ``target`` and ``floor``
      equal ``sampled`` (a full run is below its floor when any element is unlabelled).
    """

    teacher: str
    embedder: str
    target: int
    floor: int
    ceiling_hours: float
    mode: str = SAMPLE_MODE
    labeler: str = TEACHER_LABELER
    sampled: int = 0
    skipped: int = 0
    topic_labelled: int = 0
    function_labelled: int = 0
    labelled_total: int = 0
    prescreen_rows: int = 0
    embedded: int = 0
    weak_labels: int = 0
    teacher_items: int = 0
    teacher_seconds: float = 0.0
    embed_seconds: float = 0.0
    seconds: float = 0.0
    items_per_second: float | None = None
    batches: int = 0
    stopped_by_ceiling: bool = False
    below_floor: bool = False
    by_family: dict[str, int] = Field(default_factory=dict)


@dataclass
class _Task:
    pick: Pick
    position: int  # in the sample
    topic: bool
    function: bool
    shortlist: list[tuple[str, float]] = field(default_factory=list)
    prescreen: dict[str, float] = field(default_factory=dict)
    asked: list[tuple[str, float]] = field(default_factory=list)
    cosine: dict[str, float] = field(default_factory=dict)
    write_side: bool = True  # write its weak and prescreen rows (not current already)


Progress = Callable[[int, int, LabelReport], None]


def label_corpus(
    corpus: Corpus,
    teacher: TeacherEngine,
    embedder: Any,
    *,
    sample: SampleSpec | None = None,
    ceiling_hours: float = 12.0,
    functions_share: float = 0.5,
    batch_size: int = 256,
    shortlist_k: int = SHORTLIST_K,
    prescreen_k: int = PRESCREEN_K,
    topics: Taxonomy | None = None,
    functions: Taxonomy | None = None,
    clock: Callable[[], float] = time.monotonic,
    on_progress: Progress | None = None,
    release_embedder: bool = True,
) -> LabelReport:
    """Label the ``sample`` of ``corpus`` with ``teacher`` (topic and function facets),
    shortlisting with ``embedder``; record weak labels and prescreen scores alongside.

    Two phases, so the embedder and the teacher never hold memory together: first the
    embedder shortlists and prescreens every element still to label, then (with
    ``release_embedder``) its ``close()`` frees the model and the teacher labels batch by
    batch. Resumable: elements whose teacher rows carry the current key are skipped (their
    shortlists are recomputed, not stored). The run stops before a batch, in either phase,
    once ``ceiling_hours`` of ``clock`` have passed. ``topics`` and ``functions`` default to
    the built-in ``element_topics`` and ``element_functions``. The teacher stays open, and a
    released embedder reloads on its next use; the caller closes both. ``on_progress(done,
    total, report)`` is called after each teacher batch.
    """
    sample = sample if sample is not None else SampleSpec()
    if not ceiling_hours > 0:
        raise ValueError(f"ceiling_hours must be positive, got {ceiling_hours}")
    if not 0.0 <= functions_share <= 1.0:
        raise ValueError(f"functions_share must be within [0, 1], got {functions_share}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    run = _Run(
        corpus,
        teacher,
        embedder,
        _Prompts(
            teacher,
            embedder,
            topics if topics is not None else builtin_taxonomy("element_topics"),
            functions if functions is not None else builtin_taxonomy("element_functions"),
            shortlist_k,
            prescreen_k,
        ),
        clock,
        release_embedder,
    )
    return run.label(sample, ceiling_hours, functions_share, batch_size, on_progress)


class _Run:
    def __init__(
        self,
        corpus: Corpus,
        teacher: TeacherEngine,
        embedder: Any,
        keys: _Prompts,
        clock: Callable[[], float],
        release_embedder: bool,
    ) -> None:
        self.corpus, self.teacher, self.embedder, self.keys = corpus, teacher, embedder, keys
        self.clock = clock
        self.release_embedder = release_embedder
        self.doc_types = builtin_taxonomy("doc_types")
        self.report: LabelReport
        self.full = False  # full mode: topic only, into TEACHER_FULL_LABELER
        self.labeler = TEACHER_LABELER

    def label(
        self,
        sample: SampleSpec,
        ceiling_hours: float,
        functions_share: float,
        batch_size: int,
        on_progress: Progress | None,
    ) -> LabelReport:
        start = self.clock()
        self.full = sample.full
        self.labeler = TEACHER_FULL_LABELER if self.full else TEACHER_LABELER
        report = self.report = LabelReport(
            teacher=self.keys.teacher,
            embedder=self.keys.embedder,
            target=sample.target,
            floor=sample.floor,
            ceiling_hours=ceiling_hours,
            mode=sample.mode,
            labeler=self.labeler,
        )
        picks = sample_elements(self.corpus, sample, functions_share=functions_share)
        if self.full:  # a full run is complete only when every element is labelled
            report.target = report.floor = len(picks)
        existing = self.corpus.labels(self.labeler)
        # a full run never reads or writes the prescreen file: the training chain owns it
        prescreened = {} if self.full else self.corpus.labels(PRESCREEN_LABELER)
        tasks, topic_done = [], 0
        for position, pick in enumerate(picks):
            topic = not self._current(existing, pick, TOPIC_FACET)
            function = (
                not self.full
                and pick.functions
                and not self._current(existing, pick, FUNCTION_FACET)
            )
            topic_done += not topic
            if topic or function:
                tasks.append(_Task(pick, position, topic, function))
        report.sampled = len(picks)
        report.skipped = len(picks) - len(tasks)
        ceiling = ceiling_hours * 3600

        def over() -> bool:
            if self.clock() - start >= ceiling:
                report.stopped_by_ceiling = True
            return report.stopped_by_ceiling

        # Phase 1: shortlist every task and embed the whole sample's function cosines (the
        # prescreen percentiles need them all), then release the embedder, so it never holds
        # memory while the teacher runs (together they peaked at 9.25 GB on the smoke).
        if tasks:
            self._phase_one(picks, tasks, prescreened, batch_size, over)
        if self.release_embedder:
            close_backend(self.embedder)
            self.embedder = None  # this run's reference; the model is gone after close()
            release_memory()  # gc, and the MPS cache when torch is loaded
        # Phase 2: the teacher, batch by batch; each batch's rows are written when it ends.
        done = report.skipped
        for begin in range(0, len(tasks), batch_size):
            if over():
                break
            batch = tasks[begin : begin + batch_size]
            self._batch(batch)
            report.batches += 1
            done += len(batch)
            if report.teacher_seconds > 0:
                report.items_per_second = round(report.teacher_items / report.teacher_seconds, 4)
            report.seconds = round(self.clock() - start, 3)
            if on_progress is not None:
                on_progress(done, len(picks), report)
        report.labelled_total = topic_done + report.topic_labelled
        report.below_floor = report.labelled_total < report.floor
        report.seconds = round(self.clock() - start, 3)
        return report

    def _current(self, existing: dict[LabelKey, LabelRow], pick: Pick, facet: str) -> bool:
        element = pick.element
        row = existing.get(LabelKey(element.doc_id, element.element_id, element.text_sha1, facet))
        return row is not None and row.provenance.get("label_key") == self.keys.key(pick, facet)

    # --- one batch ----------------------------------------------------------------------------

    def _batch(self, tasks: list[_Task]) -> None:
        if self.full:  # topic rows only, one write: the commit point of the batch
            self.corpus.write_labels(self.labeler, self._topics(tasks))
            return
        side = [t for t in tasks if t.write_side]
        weak_rows = self._weak_rows(side)
        prescreen_rows = [self._prescreen_row(t) for t in side]
        self.corpus.write_labels(WEAK_LABELER, weak_rows)
        self.corpus.write_labels(PRESCREEN_LABELER, prescreen_rows)
        rows = self._topics(tasks) + self._functions(tasks)
        self.corpus.write_labels(TEACHER_LABELER, rows)  # the commit point of the batch
        self.report.weak_labels += len(weak_rows)
        self.report.prescreen_rows += len(prescreen_rows)

    def _phase_one(
        self,
        picks: list[Pick],
        tasks: list[_Task],
        prescreened: dict[LabelKey, LabelRow],
        batch_size: int,
        over: Callable[[], bool],
    ) -> None:
        """Shortlists for the tasks that need a topic, function cosines for every sampled
        element (reused from current prescreen rows), then each task's percentile scores and
        asked tags. Stops early (leaving the tasks unscored) when ``over()``."""
        started = self.clock()
        tags = function_tags(self.keys.functions)
        cosines = np.zeros((len(picks), len(tags)))
        stored: dict[int, LabelRow] = {}
        missing: list[int] = []
        for i, pick in enumerate(picks):
            element = pick.element
            row = prescreened.get(
                LabelKey(element.doc_id, element.element_id, element.text_sha1, FUNCTION_FACET)
            )
            cosine = row.provenance.get("cosine") if row is not None else None
            if (
                row is not None
                and row.provenance.get("prescreen_key") == self.keys.prescreen_key
                and isinstance(cosine, dict)
                and all(tag in cosine for tag in tags)
            ):
                stored[i] = row
                cosines[i] = [float(cosine[tag]) for tag in tags]
            else:
                missing.append(i)
        topic_tasks = [t for t in tasks if t.topic]
        for begin in range(0, len(topic_tasks), batch_size):
            if over():
                return
            self._shortlist(topic_tasks[begin : begin + batch_size])
        if self.full:  # no function questions: no cosines, percentiles or weak rows
            self.report.embed_seconds = round(self.clock() - started, 3)
            return
        for begin in range(0, len(missing), batch_size):
            if over():
                return
            chunk = missing[begin : begin + batch_size]
            cosines[chunk] = function_cosines(
                [picks[i].element for i in chunk],
                self.embedder,
                taxonomy=self.keys.functions,
                captions=[picks[i].caption for i in chunk],
            )
        self.report.embedded = len(missing)
        ranks = percentile_scores(cosines)
        for task in tasks:
            i = task.position
            task.cosine = {tag: float(c) for tag, c in zip(tags, cosines[i], strict=True)}
            task.prescreen = {tag: float(r) for tag, r in zip(tags, ranks[i], strict=True)}
            task.asked = (
                top_functions(task.prescreen, self.keys.prescreen_k) if task.function else []
            )
            row = stored.get(i)
            task.write_side = row is None or row.provenance.get("weak_key") != self.keys.weak_key
        self.report.embed_seconds = round(self.report.embed_seconds + self.clock() - started, 3)

    def _shortlist(self, tasks: list[_Task]) -> None:
        by_family: dict[str, list[_Task]] = {}
        for task in tasks:
            by_family.setdefault(task.pick.family, []).append(task)
        for family, group in by_family.items():
            scored = shortlist_scores(
                [t.pick.element for t in group],
                family,
                self.embedder,
                self.keys.shortlist_k,
                taxonomy=self.keys.topics,
                captions=[t.pick.caption for t in group],
            )
            for task, ranked in zip(group, scored, strict=True):
                task.shortlist = ranked

    def _item(self, pick: Pick, options: list[tuple[str, str]]) -> TeachItem:
        element, record = pick.element, pick.record
        doc_type = record.doc_type_hint
        if doc_type is not None and doc_type in self.doc_types:
            label = self.doc_types.node(doc_type).label
        else:
            label = self.keys.topics.node(pick.family).label
        context = prompts.context_text(label, element.heading_path, element.kind)
        text = element_text(element, pick.caption)[:TEACHER_MAX_CHARS]
        return TeachItem(text=text, context=context, options=options)

    def _ask(self, call: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """``call(*args, **kwargs)``, its time added to the teacher's."""
        started = self.clock()
        out = call(*args, **kwargs)
        self.report.teacher_seconds = round(self.report.teacher_seconds + self.clock() - started, 3)
        return out

    def _topics(self, tasks: list[_Task]) -> list[LabelRow]:
        by_family: dict[str, list[_Task]] = {}
        for task in tasks:
            if task.topic:
                by_family.setdefault(task.pick.family, []).append(task)
        rows = []
        for family, group in by_family.items():
            instruction, options, prompt_hash, _ = self.keys.family(family)
            position = {tag: i for i, (tag, _) in enumerate(options)}
            text_of = dict(options)
            items = []
            for task in group:
                tags = sorted((tag for tag, _ in task.shortlist), key=position.__getitem__)
                items.append(self._item(task.pick, [(tag, text_of[tag]) for tag in tags]))
            results = self._ask(
                self.teacher.choose,
                items,
                instruction=instruction,
                prefix_key=f"topic:{family}",
                prefix_options=options,
            )
            self.report.teacher_items += len(items)
            for task, probs in zip(group, results, strict=True):
                key = self.keys.key(task.pick, TOPIC_FACET)
                provenance = self.keys.provenance(key, prompt_hash, self.keys.topic_hash)
                provenance["family"] = family
                provenance["shortlist"] = [[tag, round(s, 6)] for tag, s in task.shortlist]
                rows.append(self._row(task.pick, TOPIC_FACET, probs, provenance))
            self.report.topic_labelled += len(group)
            self.report.by_family[family] = self.report.by_family.get(family, 0) + len(group)
        return rows

    def _functions(self, tasks: list[_Task]) -> list[LabelRow]:
        asking = [t for t in tasks if t.function]
        answers: dict[int, dict[str, float]] = {id(t): {} for t in asking}
        for tag, question in self.keys.questions:
            group = [t for t in asking if any(a == tag for a, _ in t.asked)]
            if not group:
                continue
            items = [self._item(t.pick, []) for t in group]
            p_yes = self._ask(self.teacher.yes_no, items, question=question)
            self.report.teacher_items += len(items)
            for task, p in zip(group, p_yes, strict=True):
                answers[id(task)][tag] = p
        rows = []
        for task in asking:
            got = answers[id(task)]
            probs = {tag: got[tag] for tag, _ in task.asked}
            key = self.keys.key(task.pick, FUNCTION_FACET)
            provenance = self.keys.provenance(
                key, self.keys.function_prompt_hash, self.keys.function_hash
            )
            provenance["asked"] = [tag for tag, _ in task.asked]
            provenance["asked_prescreen"] = [round(s, 6) for _, s in task.asked]
            rows.append(self._row(task.pick, FUNCTION_FACET, probs, provenance))
        self.report.function_labelled += len(asking)
        return rows

    def _row(
        self, pick: Pick, facet: str, probs: dict[str, float], provenance: dict[str, Any]
    ) -> LabelRow:
        element = pick.element
        clean = {tag: min(1.0, max(0.0, float(p))) for tag, p in probs.items()}
        return LabelRow(
            doc_id=element.doc_id,
            element_id=element.element_id,
            text_sha1=element.text_sha1,
            facet=facet,
            probs=clean,
            provenance=provenance,
        )

    def _prescreen_row(self, task: _Task) -> LabelRow:
        provenance = {
            "prescreen_key": self.keys.prescreen_key,
            "weak_key": self.keys.weak_key,
            "embedder": self.keys.embedder,
            "instruction": FUNCTION_INSTRUCTION,
            "version": SHORTLIST_VERSION,
            "score": PRESCREEN_SCORE,
            "sample_size": self.report.sampled,
            "cosine": {tag: round(c, 6) for tag, c in task.cosine.items()},
            "taxonomy_hash": self.keys.function_hash,
            "timestamp": _now(),
        }
        return self._row(task.pick, FUNCTION_FACET, task.prescreen, provenance)

    def _weak_rows(self, tasks: list[_Task]) -> list[LabelRow]:
        rows = []
        for task in tasks:
            found = weak_labels(task.pick.element, task.pick.record, taxonomy=self.keys.topics)
            for kind, weak in found.items():
                provenance = {k: v for k, v in weak.items() if k not in ("label", "confidence")}
                provenance["taxonomy_hash"] = self.keys.topic_hash
                provenance["timestamp"] = _now()
                confidence = float(weak.get("confidence", 1.0))
                rows.append(self._row(task.pick, kind, {weak["label"]: confidence}, provenance))
        return rows
