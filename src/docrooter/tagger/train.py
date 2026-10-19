"""Training the element tagger (spec §9): stage A or B, function heads, calibration and one
round of active learning.

- **Stage A** fits the heads on the frozen body; **stage B** first fine-tunes the body
  contrastively (``train(finetune_body=True, max_seq_len=256, gradient_checkpointing=True)``).
  Topic heads are partial (``allow_partial``) with at least :data:`MIN_EXAMPLES` (20) examples
  per trained leaf: a sparse leaf is left out of its family's head and predicted only via its
  family (Review Focus 4), while the family head still decides among the other leaves.
- **Time budget:** stage B's steps are estimated from ``seconds_per_step`` before training;
  when they would exceed ``time_budget_hours``, the triplets are sub-sampled (stratified by
  node, as :func:`~docrooter.train.trainer.build_triplets` caps them) to fit, and the report
  says so.
- **Calibration** (:func:`calibrate_tagger`): the topic temperature through
  :func:`~docrooter.train.calibrate.calibrate` on the ``calib`` examples, the function
  temperature by the same grid search on the binary decisions.
- **Active learning** (:func:`active_learning_round`): the unlabelled train elements with the
  smallest top-2 margin, taken in turn from each predicted class, go to the teacher.
"""

from __future__ import annotations

import logging
import math
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np

from docrooter.backends.recipes import get_recipe
from docrooter.corpus import SKIPPED_KINDS
from docrooter.corpus.store import Corpus, CorpusElement, LabelKey, LabelRow, ManifestRecord
from docrooter.tagger.bundle import (
    CHARS_PER_TOKEN,
    ElementTagger,
    temper_binary,
)
from docrooter.tagger.functions import (
    FunctionRow,
    default_tags,
    function_targets,
    target_counts,
    train_function_heads,
)
from docrooter.taxonomies import ELEMENT_FAMILIES, builtin_taxonomy, family_for_doc_type
from docrooter.taxonomy import Taxonomy
from docrooter.teach import prompts
from docrooter.teach.base import TeacherEngine, TeachItem, label_key, refuse_jev
from docrooter.teach.confidence import MIN_PER_CLASS
from docrooter.teach.label import (
    TEACHER_LABELER,
    TEACHER_MAX_CHARS,
    TOPIC_FACET,
    TOPIC_TEACHER_INSTRUCTION,
)
from docrooter.teach.shortlist import SHORTLIST_K, element_text, family_options
from docrooter.train import Dataset
from docrooter.train.trainer import DEFAULT_BASE_MODEL, build_triplets, train

__all__ = [
    "MIN_EXAMPLES",
    "SECONDS_PER_STEP",
    "TOPIC_INSTRUCTION",
    "Uncertain",
    "active_learning_round",
    "calibrate_tagger",
    "select_uncertain",
    "train_tagger",
    "trained_tree",
]

logger = logging.getLogger("docrooter.tagger")

MIN_EXAMPLES = MIN_PER_CLASS  # 20 confident examples for a leaf to be trained (spec §8)
TOPIC_INSTRUCTION = "Classify the topic of this document element"
# Estimated seconds per fine-tuning step (a batch of 16 triplets, 256 tokens, gradient
# checkpointing) on a base M1 under the 8 GB budget, from research §4.1's throughput estimate
# (about 0.3k tokens/s). Replace it with the measured value: pass ``seconds_per_step``.
SECONDS_PER_STEP = 30.0
AL_N = 3000
AL_SHORTLIST_K = SHORTLIST_K  # the student's top leaves offered, as the first pass offers 5


# --- training ---------------------------------------------------------------------------------


def _capped(dataset: Dataset, max_chars: int) -> Dataset:
    return Dataset([e.model_copy(update={"text": e.text[:max_chars]}) for e in dataset])


def _budget_cap(
    n_triplets: int,
    *,
    batch_size: int,
    epochs: int,
    seconds_per_step: float,
    budget_seconds: float,
) -> int | None:
    """The triplet count that fits the budget, or ``None`` when all ``n_triplets`` fit."""
    steps = math.ceil(n_triplets / batch_size) * epochs
    if steps * seconds_per_step <= budget_seconds:
        return None
    steps_per_epoch = int(budget_seconds // (seconds_per_step * epochs))
    return max(1, steps_per_epoch) * batch_size


def train_tagger(
    dataset: Dataset,
    function_rows: Sequence[FunctionRow],
    *,
    stage: Literal["A", "B"],
    base_model: str = DEFAULT_BASE_MODEL,
    max_seq_len: int = 256,
    max_triplets: int = 40000,
    time_budget_hours: float = 6.0,
    seconds_per_step: float = SECONDS_PER_STEP,
    taxonomy: Taxonomy | None = None,
    min_examples: int = MIN_EXAMPLES,
    epochs: int | None = None,
    batch_size: int | None = None,
    device: str | None = None,
    seed: int = 0,
) -> ElementTagger:
    """Train the element tagger (module docstring).

    ``dataset`` holds the confident topic examples, texts from
    :func:`~docrooter.teach.dataset.to_dataset` (paths ``[family, leaf]``, or ``[family]`` for
    an untrained leaf); ``function_rows`` the function sample
    (:func:`~docrooter.tagger.functions.function_rows`, built with the same ``max_chars``:
    ``CHARS_PER_TOKEN * max_seq_len``). ``taxonomy`` defaults to the built-in
    ``element_topics``. Temperatures start at 1.0; see :func:`calibrate_tagger`.
    """
    if stage not in ("A", "B"):
        raise ValueError(f"stage must be 'A' or 'B', got {stage!r}")
    if not time_budget_hours > 0 or not seconds_per_step > 0:
        raise ValueError("time_budget_hours and seconds_per_step must be positive")
    started = time.perf_counter()
    taxonomy = taxonomy if taxonomy is not None else builtin_taxonomy("element_topics")
    max_chars = CHARS_PER_TOKEN * max_seq_len
    data = _capped(dataset, max_chars)

    budget: dict[str, Any] = {"hours": time_budget_hours, "seconds_per_step": seconds_per_step}
    cap = max_triplets
    if stage == "B":
        recipe = get_recipe(base_model, batch_size=batch_size)
        used_batch = recipe.batch_size_for(device or "cpu")
        used_epochs = epochs if epochs is not None else recipe.finetune_epochs
        n = len(build_triplets(taxonomy, data, max_triplets=max_triplets, seed=seed))
        fit = _budget_cap(
            n,
            batch_size=used_batch,
            epochs=used_epochs,
            seconds_per_step=seconds_per_step,
            budget_seconds=time_budget_hours * 3600,
        )
        budget.update(triplets_available=n, batch_size=used_batch, epochs=used_epochs)
        if fit is not None:
            cap = min(cap, fit)
            logger.warning(
                "stage B: %d triplets would take about %.1f h at %.1f s/step; sub-sampled to "
                "%d to fit the %.1f h budget",
                n,
                math.ceil(n / used_batch) * used_epochs * seconds_per_step / 3600,
                seconds_per_step,
                cap,
                time_budget_hours,
            )
        budget.update(subsampled=fit is not None, triplets_cap=cap)

    topic = train(
        taxonomy,
        data,
        base_model=base_model,
        finetune_body=stage == "B",
        epochs=epochs,
        batch_size=batch_size,
        min_examples_per_child=min_examples,
        max_triplets=cap,
        device=device,
        seed=seed,
        instruction=TOPIC_INSTRUCTION,
        max_seq_len=max_seq_len,
        gradient_checkpointing=True,
        allow_partial=True,
    )
    tagger = ElementTagger(topic, {}, taxonomy=taxonomy, max_chars=max_chars)
    functions: dict[str, Any] = {}
    counts: dict[str, dict[str, int]] = {}
    if function_rows:
        texts = [row.text[:max_chars] for row in function_rows]
        vectors = tagger.embed(texts)
        tags = default_tags()
        counts = target_counts(function_targets(function_rows, tags))
        functions = train_function_heads(vectors, function_rows, tags=tags, seed=seed)
    tagger.functions = functions
    tagger.info = {
        "stage": stage,
        "base_model": base_model,
        "max_seq_len": max_seq_len,
        "min_examples": min_examples,
        "examples": len(data),
        "untrained_leaves": tagger.untrained,
        "budget": budget,
        "topic_train_metrics": topic.manifest["train_metrics"],
        "function_counts": counts,
        "function_heads": list(functions),
        "seconds": round(time.perf_counter() - started, 3),
    }
    if tagger.untrained:
        logger.warning(
            "topic leaves without a head (predicted only via their family): %s",
            ", ".join(tagger.untrained),
        )
    return tagger


# --- calibration ------------------------------------------------------------------------------


def trained_tree(taxonomy: Taxonomy, heads: Mapping[str, Mapping[str, Any]]) -> Taxonomy:
    """``taxonomy`` without the leaves no head predicts: the tree a partial model can score with
    a :class:`~docrooter.classify.Classifier` (each head then sees exactly its classes). A
    family left with no leaves becomes a leaf itself."""
    covered = {c for info in heads.values() for c in info["classes"]}
    keep = [
        n for n in taxonomy.nodes.values() if n.children or n.id in covered or n.parent_id is None
    ]
    return Taxonomy([n.model_copy(update={"children": []}) for n in keep])


def _cut_paths(dataset: Dataset, tree: Taxonomy) -> Dataset:
    """Each example's path down to its deepest node still in ``tree``."""
    out = []
    for example in dataset:
        path = []
        for part in example.path:
            if part not in tree:
                break
            path.append(part)
        if path:
            out.append(example.model_copy(update={"path": path}))
    return Dataset(out)


def _binary_nll(probs: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    p = np.clip(temper_binary(probs, temperature), 1e-12, 1 - 1e-12)
    return float(-np.mean(targets * np.log(p) + (1 - targets) * np.log1p(-p)))


def calibrate_tagger(
    tagger: ElementTagger,
    calib: Dataset,
    function_rows: Sequence[FunctionRow] = (),
    *,
    grid: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Fit one temperature per facet on the ``calib`` split and set ``tagger.temperatures``
    (also recorded in ``tagger.info["calibration"]``). Returns the report per facet.

    - **topic:** :func:`~docrooter.train.calibrate.calibrate` on a classifier over the tagger's
      :func:`trained_tree`, the examples' paths cut to it; it fits the temperature of the
      topic model's decisions at the root and the family heads;
    - **function:** the grid temperature with the lowest mean binary NLL over the function
      targets of ``function_rows`` (the training rule, §4.0 ruling 6).

    A facet with fewer than ``MIN_DECISIONS`` decisions keeps its temperature.
    """
    from docrooter.classify.classifier import Classifier
    from docrooter.train.calibrate import DEFAULT_GRID, MIN_DECISIONS, calibrate

    grid = list(grid) if grid is not None else list(DEFAULT_GRID)
    report: dict[str, Any] = {}
    topic = tagger.topic
    tree = trained_tree(tagger.taxonomy, topic.manifest["heads"])
    data = _cut_paths(_capped(calib, tagger.max_chars), tree)
    with warnings.catch_warnings():
        # the trained tree's hash differs from the full tree's the model was trained on
        warnings.filterwarnings("ignore", message=".*trained on a different taxonomy.*")
        clf = Classifier(tree, backend=topic, threshold=0.0)
    calibrated = calibrate(clf, data, grid=grid)
    entry = (calibrated.calibration_report or {}).get(topic.name)
    if entry is not None and "temperature" in entry:
        tagger.temperatures["topic"] = float(entry["temperature"])
    report["topic"] = entry or {"n": 0, "skipped": "no decision of the topic model"}

    if tagger.functions and function_rows:
        tags = tagger.function_tags
        targets = function_targets(function_rows, tags)
        index = sorted({i for pairs in targets.values() for i, _ in pairs})
        raw: dict[int, dict[str, float]] = {}
        if index:
            vectors = tagger.embed([function_rows[i].text[: tagger.max_chars] for i in index])
            for tag in tags:
                head = tagger.functions[tag]
                column = np.asarray(head.predict_proba(vectors), dtype=float)[
                    :, list(head.classes_).index(1)
                ]
                for i, p in zip(index, column.tolist(), strict=True):
                    raw.setdefault(i, {})[tag] = p
        probs = np.array([raw[i][tag] for tag in tags for i, _ in targets[tag]])
        ys = np.array([y for tag in tags for _, y in targets[tag]], dtype=float)
        if len(ys) < MIN_DECISIONS:
            report["function"] = {"n": len(ys), "skipped": f"fewer than {MIN_DECISIONS}"}
        else:
            losses = [_binary_nll(probs, ys, t) for t in grid]
            best = grid[int(np.argmin(losses))]
            report["function"] = {
                "n": len(ys),
                "temperature": best,
                "nll_before": _binary_nll(probs, ys, tagger.temperatures["function"]),
                "nll_after": float(min(losses)),
            }
            tagger.temperatures["function"] = best
    else:
        report["function"] = {"n": 0, "skipped": "no function heads or rows"}
    tagger.info["calibration"] = report
    return report


# --- active learning --------------------------------------------------------------------------


class Uncertain:
    """One unlabelled element the student scored: its key, family, predicted class (a leaf, or
    the family when it has no head), top-2 margin and the leaves to offer the teacher."""

    __slots__ = ("doc_id", "element_id", "family", "margin", "options", "predicted", "text_sha1")

    def __init__(
        self,
        element: CorpusElement,
        family: str,
        predicted: str,
        margin: float,
        options: list[str],
    ) -> None:
        self.doc_id, self.element_id = element.doc_id, element.element_id
        self.text_sha1 = element.text_sha1
        self.family, self.predicted, self.margin, self.options = family, predicted, margin, options

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.doc_id, self.element_id, self.text_sha1)


def select_uncertain(scored: Iterable[Uncertain], n: int) -> list[Uncertain]:
    """The ``n`` elements to send to the teacher: each predicted class's elements by ascending
    margin (ties in input order), the classes taking turns (in order of first appearance),
    so every class gets an equal share while it has elements left."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    by_class: dict[str, list[Uncertain]] = {}
    for item in scored:
        by_class.setdefault(item.predicted, []).append(item)
    queues = [sorted(items, key=lambda u: u.margin) for items in by_class.values()]
    out: list[Uncertain] = []
    turn = 0
    while len(out) < n and any(turn < len(q) for q in queues):
        for queue in queues:
            if turn < len(queue) and len(out) < n:
                out.append(queue[turn])
        turn += 1
    return out


def doc_family(record: ManifestRecord, families: Sequence[str] = ELEMENT_FAMILIES) -> str:
    """A document's topic family: its ``family_hint`` when that names one of ``families``,
    else its doc type's (``generic`` without one): the teacher's rule
    (:func:`docrooter.teach.shortlist.doc_family`) over any topic tree's families."""
    if record.family_hint in families:
        return str(record.family_hint)
    return family_for_doc_type(record.doc_type_hint or "")


def _has_text(element: CorpusElement) -> bool:
    if element.kind == "table" and element.table_markdown and element.table_markdown.strip():
        return True
    return bool(element.text.strip())


def _score_batch(
    tagger: ElementTagger, batch: list[tuple[CorpusElement, str]], k: int
) -> list[Uncertain]:
    by_family: dict[str, list[CorpusElement]] = {}
    for element, family in batch:
        by_family.setdefault(family, []).append(element)
    out = []
    sparse = set(tagger.untrained)
    for family, elements in by_family.items():
        leaves = [c.id for c in tagger.taxonomy.children(family)]
        untrained = [leaf for leaf in leaves if leaf in sparse]
        other = f"{family}.other"
        for element, tags in zip(elements, tagger.tag(elements, family), strict=True):
            probs = tags["topic"].probs
            ranked = sorted(probs.values(), reverse=True)
            if probs.keys() == {family}:  # no family head: the student cannot tell the leaves
                out.append(Uncertain(element, family, family, 0.0, leaves))
                continue
            margin = ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)
            top = sorted(probs, key=lambda leaf: -probs[leaf])[:k]
            wanted = set(top) | set(untrained) | ({other} if other in leaves else set())
            options = [leaf for leaf in leaves if leaf in wanted]
            out.append(Uncertain(element, family, tags["topic"].labels[0], margin, options))
    return out


def active_learning_round(
    tagger: ElementTagger,
    corpus: Corpus,
    teacher: TeacherEngine,
    *,
    n: int = AL_N,
    split: str = "train",
    labeled: Mapping[LabelKey, LabelRow] | None = None,
    batch_size: int = 256,
    shortlist_k: int = AL_SHORTLIST_K,
    clock: Callable[[], float] = time.monotonic,
) -> list[LabelRow]:
    """One active-learning round (spec §9): the teacher's ``topic`` rows for the ``n``
    unlabelled ``split`` elements the student is least sure of. The caller adds them to the
    store (``corpus.write_labels("teacher", rows)``) and retrains stage B once.

    - **Unlabelled:** a text element (not furniture or contents) without a ``topic`` row in
      ``labeled`` (default: the store's ``teacher`` labels), whose document's family
      (:func:`doc_family`) is in the tagger's tree.
    - **Scored** by the student within its document's family; the margin is the gap between
      its two most probable leaves (0 in a family without a head).
    - **Selected** by :func:`select_uncertain`: smallest margins, stratified by predicted class.
    - **Asked** as the first pass asks (its instruction, text limit and family-wide lettered
      prefix, spec §4.0 ruling 5), the element's turn naming the student's top ``shortlist_k``
      leaves, the family's untrained leaves (the student cannot propose them) and ``other``
      (every leaf in a family without a head). Provenance records
      ``"source": "active_learning"`` and the student's shortlist.

    A Jev teacher is refused (TypeSafe MCA §2.3(b)).
    """
    refuse_jev(teacher.name)
    labeled = labeled if labeled is not None else corpus.labels(TEACHER_LABELER)
    done = {(k.doc_id, k.element_id, k.text_sha1) for k in labeled if k.facet == TOPIC_FACET}
    records = {r.id: r for r in corpus.manifest()}
    families = [n.id for n in tagger.taxonomy.children()]
    started = clock()

    scored: list[Uncertain] = []
    batch: list[tuple[CorpusElement, str]] = []
    for element in corpus.iter_elements(split=split):
        if element.kind in SKIPPED_KINDS or not _has_text(element):
            continue
        if (element.doc_id, element.element_id, element.text_sha1) in done:
            continue
        family = doc_family(records[element.doc_id], families)
        if family not in families:  # a tree without that family: nothing to ask
            continue
        batch.append((element, family))
        if len(batch) >= batch_size:
            scored.extend(_score_batch(tagger, batch, shortlist_k))
            batch = []
    if batch:
        scored.extend(_score_batch(tagger, batch, shortlist_k))
    chosen = {u.key: u for u in select_uncertain(scored, n)}
    logger.info(
        "active learning: %d unlabelled %s elements scored in %.1f s, %d selected",
        len(scored),
        split,
        clock() - started,
        len(chosen),
    )
    del scored
    if not chosen:
        return []

    picked: list[tuple[CorpusElement, Uncertain]] = []
    for element in corpus.iter_elements(split=split):
        item = chosen.get((element.doc_id, element.element_id, element.text_sha1))
        if item is not None:
            picked.append((element, item))
    return _ask_teacher(tagger, teacher, picked, records)


def _ask_teacher(
    tagger: ElementTagger,
    teacher: TeacherEngine,
    picked: list[tuple[CorpusElement, Uncertain]],
    records: Mapping[str, ManifestRecord],
) -> list[LabelRow]:
    taxonomy = tagger.taxonomy
    taxonomy_hash = taxonomy.content_hash()
    doc_types = builtin_taxonomy("doc_types")
    by_family: dict[str, list[tuple[CorpusElement, Uncertain]]] = {}
    for element, item in picked:
        by_family.setdefault(item.family, []).append((element, item))
    rows: list[LabelRow] = []
    for family, group in by_family.items():
        options = family_options(family, taxonomy)
        text_of = dict(options)
        instruction = TOPIC_TEACHER_INSTRUCTION.format(family=taxonomy.node(family).label.lower())
        prompt_hash = prompts.prompt_hash(instruction, prompts.options_block(options))
        items = []
        for element, item in group:
            doc_type = records[element.doc_id].doc_type_hint
            if doc_type is not None and doc_type in doc_types:
                label = doc_types.node(doc_type).label
            else:
                label = taxonomy.node(family).label
            context = prompts.context_text(label, element.heading_path, element.kind)
            items.append(
                TeachItem(
                    text=element_text(element)[:TEACHER_MAX_CHARS],
                    context=context,
                    options=[(tag, text_of[tag]) for tag in item.options],
                )
            )
        results = teacher.choose(
            items, instruction=instruction, prefix_key=f"topic:{family}", prefix_options=options
        )
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for (element, item), probs in zip(group, results, strict=True):
            key = label_key(
                doc_id=element.doc_id,
                element_id=element.element_id,
                text_sha1=element.text_sha1,
                facet=TOPIC_FACET,
                teacher=teacher.name,
                prompt_hash=prompt_hash,
                taxonomy_hash=taxonomy_hash,
            )
            rows.append(
                LabelRow(
                    doc_id=element.doc_id,
                    element_id=element.element_id,
                    text_sha1=element.text_sha1,
                    facet=TOPIC_FACET,
                    probs={tag: min(1.0, max(0.0, float(p))) for tag, p in probs.items()},
                    provenance={
                        "label_key": key,
                        "teacher": teacher.name,
                        "source": "active_learning",
                        "student": tagger.source,
                        "student_margin": round(item.margin, 6),
                        "student_label": item.predicted,
                        "prompt_version": prompts.PROMPT_VERSION,
                        "prompt_hash": prompt_hash,
                        "taxonomy_hash": taxonomy_hash,
                        "family": family,
                        "shortlist": list(item.options),
                        "timestamp": now,
                    },
                )
            )
    return rows
