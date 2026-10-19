"""Corpus-level runs of the element tagger (spec §10): :func:`train_elements`, the teacher-to-
bundle chain behind ``docrooter train-elements``, and :func:`tag_corpus`, behind
``docrooter tag-corpus``.

**train_elements** runs, on a corpus store, one stage after the other:

1. **label** (§7): :func:`~docrooter.teach.label.label_corpus` with the teacher and the
   shortlist embedder (labels ``teacher``, ``prescreen`` and ``weak``);
2. **dataset** (§8): :func:`~docrooter.teach.dataset.build_training_set` and
   :func:`~docrooter.teach.dataset.to_dataset` on the ``train`` split, the function rows of
   ``train`` and ``calib``, and the ``calib`` topic examples (each calib element's teacher
   choice, ``[family, leaf]``);
3. **train** (§9): :func:`~docrooter.tagger.train.train_tagger` at stage A or B, then
   :func:`~docrooter.tagger.train.calibrate_tagger` on ``calib``, saved as a bundle.

Each stage is resumable (Review Focus 5). Its output goes to ``<store>/train-elements/``
(the bundle to ``out``), each file written whole through a temporary file, and a stage's
marker file is written last:

- ``label.json``: the run's settings, the wall-clock hours labelling has spent in all runs so
  far (updated after every teacher batch) and its status. ``ceiling_hours`` is a ceiling on
  that total (§4.0 ruling 5: 12 h in all): each run gets what is left of it. Labelling that
  finished, or reached the ceiling, is not run again (the labels it has are used); an
  interrupted run resumes inside :func:`label_corpus`, which skips the elements already
  labelled. Raising ``ceiling_hours`` lets a later run label more. Other settings (teacher,
  embedder, sample) start a new count;
- ``dataset.json``: the dataset's fingerprint (the bytes of the ``teacher`` and ``prescreen``
  label files, the embedder and the student's input length) and counts. The dataset is
  rebuilt when the fingerprint changes;
- the bundle's ``tagger.json`` records the stage and the dataset fingerprint
  (``info["train_elements"]``): a bundle that matches is not trained again.

One model at a time (the memory budget): the teacher is closed before the embedder builds the
dataset, and the embedder before the student trains.

Active learning (§9's one round) is not part of this chain: run
:func:`~docrooter.tagger.train.active_learning_round` and train again.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Iterable, Sequence
from itertools import groupby
from pathlib import Path
from typing import Any, Literal

from docrooter.backends.embedding import DEFAULT_EMBEDDING_MODEL, EmbeddingBackend
from docrooter.corpus.store import Corpus, CorpusElement, LabelKey, LabelRow
from docrooter.tagger.bundle import CHARS_PER_TOKEN, TAGGER_MANIFEST
from docrooter.tagger.element_tags import TAGGED_KINDS, ElementTags, tagger_fingerprint
from docrooter.tagger.functions import FunctionRow, function_rows
from docrooter.tagger.train import SECONDS_PER_STEP, calibrate_tagger, train_tagger
from docrooter.taxonomies import builtin_taxonomy
from docrooter.taxonomy import Taxonomy
from docrooter.teach.base import get_teacher, refuse_jev
from docrooter.teach.dataset import build_training_set, input_text, to_dataset
from docrooter.teach.enrich import teacher_choices
from docrooter.teach.label import (
    PRESCREEN_LABELER,
    TEACHER_LABELER,
    TOPIC_FACET,
    WEAK_LABELER,
    SampleSpec,
    label_corpus,
)
from docrooter.teach.shortlist import doc_family
from docrooter.train import Dataset, Example

__all__ = ["DEFAULT_TAG_LABELER", "WORK_DIR", "tag_corpus", "train_elements"]

WORK_DIR = "train-elements"
LABEL_MARKER = "label.json"
DATASET_MARKER = "dataset.json"
TRAIN_FILE = "train.jsonl"
CALIB_FILE = "calib.jsonl"
FUNCTIONS_TRAIN_FILE = "functions-train.jsonl"
FUNCTIONS_CALIB_FILE = "functions-calib.jsonl"
ENRICHED_FILE = "element_topics.enriched.json"
TRAINING_SET_FILE = "training-set.json"
DEFAULT_TAG_LABELER = "tagger"
RESERVED_LABELERS = (TEACHER_LABELER, PRESCREEN_LABELER, WEAK_LABELER)
"""Label files the training chain reads: tag_corpus never writes into them."""
FUNCTION_FACET = "function"

Say = Callable[[str], None]
_clock = time.monotonic  # seconds; replaced in tests


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None  # missing or torn: the stage runs again


def _lines(items: Iterable[Any]) -> str:
    return "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items)


# --- train_elements ---------------------------------------------------------------------------


def train_elements(
    corpus: Corpus,
    teacher: str,
    out: str | os.PathLike[str],
    *,
    stage: Literal["A", "B"] = "B",
    embedder: str = DEFAULT_EMBEDDING_MODEL,
    sample: SampleSpec | None = None,
    ceiling_hours: float = 12.0,
    functions_share: float = 0.5,
    batch_size: int = 256,
    max_seq_len: int = 256,
    time_budget_hours: float = 6.0,
    seconds_per_step: float = SECONDS_PER_STEP,
    device: str | None = None,
    seed: int = 0,
    say: Say | None = None,
) -> dict[str, Any]:
    """Label, build the training set and train the element tagger on ``corpus`` (module
    docstring); the bundle goes to ``out``. Returns what each stage did: its report, or
    ``"skipped"`` when its output was already there. ``teacher`` is a teacher spec
    (:func:`~docrooter.teach.get_teacher`; Jev is refused before anything runs). ``say``
    gets one line per stage."""
    refuse_jev(teacher)
    if stage not in ("A", "B"):
        raise ValueError(f"stage must be 'A' or 'B', got {stage!r}")
    say = say or (lambda _: None)
    sample = sample if sample is not None else SampleSpec()
    work = corpus.dir / WORK_DIR
    work.mkdir(parents=True, exist_ok=True)
    bundle = Path(out)
    report: dict[str, Any] = {"work_dir": str(work), "out": str(bundle), "stage": stage}

    settings = {
        "teacher": teacher,
        "embedder": embedder,
        "per_doc": sample.per_doc,
        "target": sample.target,
        "floor": sample.floor,
        "splits": list(sample.splits),
        "functions_share": functions_share,
    }
    marker = _read_json(work / LABEL_MARKER)
    if marker is None or marker.get("settings") != settings:
        marker = {"settings": settings, "spent_hours": 0.0, "status": "new", "runs": []}
    spent = float(marker.get("spent_hours", 0.0))
    if marker["status"] == "complete":
        say("label: done before, skipped")
        report["label"] = "skipped"
    elif spent >= ceiling_hours:
        say(f"label: ceiling reached ({spent:.2f} h of {ceiling_hours:g} h), existing labels used")
        report["label"] = "skipped"
    else:
        say(f"label: teacher {teacher}, {ceiling_hours - spent:.2f} h of the ceiling left")
        report["label"] = _label(corpus, marker, sample, ceiling_hours, batch_size, work)

    fingerprint = _fingerprint(corpus, embedder, max_seq_len)
    marker = _read_json(work / DATASET_MARKER)
    if marker is not None and marker.get("fingerprint") == fingerprint:
        say("dataset: built before from the same labels, skipped")
        report["dataset"] = "skipped"
    else:
        say("dataset: term mining, centroids and the confidence filter")
        report["dataset"] = _dataset(corpus, embedder, max_seq_len, fingerprint, work)

    done = _read_json(bundle / TAGGER_MANIFEST)
    wanted = {"stage": stage, "dataset": fingerprint}
    if done is not None and (done.get("info") or {}).get("train_elements") == wanted:
        say(f"train: stage {stage} bundle at {bundle} is current, skipped")
        report["train"] = "skipped"
    else:
        say(f"train: stage {stage}")
        report["train"] = _train(
            work,
            bundle,
            wanted,
            stage=stage,
            max_seq_len=max_seq_len,
            time_budget_hours=time_budget_hours,
            seconds_per_step=seconds_per_step,
            device=device,
            seed=seed,
        )
    return report


def _label(
    corpus: Corpus,
    marker: dict[str, Any],
    sample: SampleSpec,
    ceiling_hours: float,
    batch_size: int,
    work: Path,
) -> dict[str, Any]:
    """Run :func:`label_corpus` with what is left of the ceiling (``ceiling_hours`` minus the
    hours earlier runs spent, kept in ``label.json``) and record the hours this run spends:
    after every teacher batch, on any exception, and at the end. A run that stops at the
    ceiling is as final as one that finishes (§4.0 ruling 5: "as many as fit")."""
    settings = marker["settings"]
    spent_before = float(marker.get("spent_hours", 0.0))
    started = _clock()

    def save(status: str, data: dict[str, Any] | None = None) -> None:
        marker["spent_hours"] = round(spent_before + (_clock() - started) / 3600, 6)
        marker["status"] = status
        marker["ceiling_hours"] = ceiling_hours
        if data is not None:
            marker["runs"] = [*marker.get("runs", []), data]
        _write_atomic(work / LABEL_MARKER, json.dumps(marker, indent=2))

    save("running")
    engine = get_teacher(settings["teacher"])
    backend = None
    status = "interrupted"
    try:
        backend = EmbeddingBackend(settings["embedder"])
        result = label_corpus(
            corpus,
            engine,
            backend,
            sample=sample,
            ceiling_hours=ceiling_hours - spent_before,
            functions_share=settings["functions_share"],
            batch_size=batch_size,
            on_progress=lambda done, total, report: save("running"),
        )
        status = "stopped_by_ceiling" if result.stopped_by_ceiling else "complete"
    finally:
        engine.close()
        if backend is not None:
            backend.close()
        if status == "interrupted":
            save(status)  # the hours spent still count against the ceiling
    data = result.model_dump(mode="json")
    save(status, data)
    return {**data, "spent_hours": marker["spent_hours"]}


def _fingerprint(corpus: Corpus, embedder: str, max_seq_len: int) -> str:
    """What the dataset depends on: the teacher and prescreen label files (their bytes), the
    embedder and the student's input length."""
    digest = hashlib.sha256()
    digest.update(json.dumps({"embedder": embedder, "max_seq_len": max_seq_len}).encode())
    for name in (TEACHER_LABELER, PRESCREEN_LABELER):
        path = corpus.dir / "labels" / f"{name}.jsonl"
        digest.update(f"\0{name}\0".encode())
        if path.is_file():
            with path.open("rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    digest.update(block)
    return digest.hexdigest()


def calib_examples(corpus: Corpus, labels: dict[LabelKey, LabelRow], taxonomy: Taxonomy) -> Dataset:
    """The ``calib`` split's topic examples: each element the teacher labelled, its text
    :func:`~docrooter.teach.dataset.input_text` and its path ``[family, teacher's leaf]``."""
    choices = teacher_choices(labels, taxonomy)
    examples = []
    for element in corpus.iter_elements(split="calib"):
        key = LabelKey(element.doc_id, element.element_id, element.text_sha1, TOPIC_FACET)
        choice = choices.get(key)
        if choice is None:
            continue
        family = taxonomy.nodes[choice.label].parent_id
        path = [family, choice.label] if family is not None else [choice.label]
        examples.append(
            Example(
                text=input_text(element),
                path=path,
                doc_id=element.doc_id,
                section_id=element.element_id,
            )
        )
    return Dataset(examples)


def _dataset(
    corpus: Corpus, embedder: str, max_seq_len: int, fingerprint: str, work: Path
) -> dict[str, Any]:
    taxonomy = builtin_taxonomy("element_topics")
    labels = corpus.labels(TEACHER_LABELER)
    prescreen = corpus.labels(PRESCREEN_LABELER)
    backend = EmbeddingBackend(embedder)
    try:
        enriched, training_set = build_training_set(corpus, labels, taxonomy, backend)
    finally:
        backend.close()
    train = to_dataset(training_set, corpus)
    calib = calib_examples(corpus, labels, taxonomy)
    max_chars = CHARS_PER_TOKEN * max_seq_len
    rows = {
        split: function_rows(corpus, labels, prescreen, split=split, max_chars=max_chars)
        for split in ("train", "calib")
    }
    for name, data in ((TRAIN_FILE, train), (CALIB_FILE, calib)):
        _write_atomic(
            work / name, _lines(e.model_dump(mode="json", exclude_none=True) for e in data)
        )
    for name, split in ((FUNCTIONS_TRAIN_FILE, "train"), (FUNCTIONS_CALIB_FILE, "calib")):
        _write_atomic(work / name, _lines(r.model_dump(mode="json") for r in rows[split]))
    _write_atomic(work / ENRICHED_FILE, json.dumps(enriched.to_dict(), indent=2))
    _write_atomic(work / TRAINING_SET_FILE, training_set.model_dump_json(indent=2))
    report = {
        "fingerprint": fingerprint,
        "train_examples": len(train),
        "calib_examples": len(calib),
        "function_rows": {split: len(r) for split, r in rows.items()},
    }
    _write_atomic(work / DATASET_MARKER, json.dumps(report, indent=2))  # the commit point
    return report


def _function_rows(path: Path) -> list[FunctionRow]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [FunctionRow.model_validate_json(line) for line in lines if line.strip()]


def _train(work: Path, bundle: Path, wanted: dict[str, str], **options: Any) -> dict[str, Any]:
    train = Dataset.load(work / TRAIN_FILE)
    calib = Dataset.load(work / CALIB_FILE)
    tagger = train_tagger(train, _function_rows(work / FUNCTIONS_TRAIN_FILE), **options)
    try:
        calibration = calibrate_tagger(tagger, calib, _function_rows(work / FUNCTIONS_CALIB_FILE))
        tagger.info["train_elements"] = wanted
        bundle.mkdir(parents=True, exist_ok=True)
        tagger.save(bundle)  # tagger.json last: the commit point
    finally:
        tagger.close()
    return {
        "examples": len(train),
        "calib_examples": len(calib),
        "temperatures": dict(tagger.temperatures),
        "calibration": calibration,
        "untrained_leaves": list(tagger.info.get("untrained_leaves", [])),
    }


# --- tag_corpus -------------------------------------------------------------------------------


def _source(tagger: ElementTags) -> str:
    return str(getattr(tagger, "source", None) or getattr(tagger, "name", type(tagger).__name__))


def _row(element: CorpusElement, facet: str, result: Any, extra: dict[str, Any]) -> LabelRow:
    provenance = {
        "source": result.source,
        "labels": list(result.labels),
        "confidence": result.confidence,
        **extra,
    }
    if result.family_probs is not None:
        provenance["family_probs"] = dict(result.family_probs)
    return LabelRow(
        doc_id=element.doc_id,
        element_id=element.element_id,
        text_sha1=element.text_sha1,
        facet=facet,
        probs=dict(result.probs),
        provenance=provenance,
    )


def tag_corpus(
    corpus: Corpus,
    tagger: ElementTags,
    *,
    labeler: str = DEFAULT_TAG_LABELER,
    splits: Sequence[str] | None = None,
    on_document: Callable[[int, str, int], None] | None = None,
) -> dict[str, Any]:
    """Tag the :data:`~docrooter.tagger.element_tags.TAGGED_KINDS` elements of ``corpus`` and
    write the results to ``labels/<labeler>.jsonl``: one ``topic`` and one ``function`` row
    per element, with the tagger's ``source``, labels, confidence and the document's family
    (its manifest hints, as the teacher's) in the provenance.

    One write per document, so a killed run loses at most the document in flight; elements
    that already have a ``topic`` row from the same tagger (its ``source`` and, for a bundle,
    the content hash :func:`~docrooter.tagger.element_tags.tagger_fingerprint`, so a bundle
    retrained in place tags again) are skipped. ``splits`` keeps those splits (all of them by
    default, ``gold`` included). ``on_document(done, doc_id, tagged)`` follows the run.
    ``labeler`` may not be one of :data:`RESERVED_LABELERS`: the last row of a key wins, so
    tags written there would replace the teacher's labels in the next training run."""
    if labeler in RESERVED_LABELERS:
        raise ValueError(
            f"--labeler {labeler!r} is a label file of the training chain "
            f"({', '.join(RESERVED_LABELERS)}): tags written there would replace the teacher's "
            "labels; choose another name, e.g. tagger"
        )
    source = _source(tagger)
    fingerprint = tagger_fingerprint(tagger)
    records = {r.id: r for r in corpus.manifest()}
    existing = corpus.labels(labeler)
    documents = tagged = skipped = 0
    elements = corpus.iter_elements(split=splits, kinds=TAGGED_KINDS)
    for doc_id, group in groupby(elements, key=lambda e: e.doc_id):
        found = list(group)
        todo = [e for e in found if not _current(existing, e, source, fingerprint)]
        skipped += len(found) - len(todo)
        if todo:
            family = doc_family(records[doc_id])
            results = tagger.tag(todo, family)
            rows = []
            for element, tags in zip(todo, results, strict=True):
                extra = {"family": family, "tagger_hash": fingerprint}
                for facet in ("topic", "function"):
                    if facet in tags:
                        rows.append(_row(element, facet, tags[facet], extra))
            corpus.write_labels(labeler, rows)  # one write per document
            tagged += len(todo)
        documents += 1
        if on_document is not None:
            on_document(documents, doc_id, len(todo))
    return {
        "labeler": labeler,
        "source": source,
        "tagger_hash": fingerprint,
        "documents": documents,
        "tagged": tagged,
        "skipped": skipped,
    }


def _current(
    existing: dict[LabelKey, LabelRow],
    element: CorpusElement,
    source: str,
    fingerprint: str | None,
) -> bool:
    row = existing.get(LabelKey(element.doc_id, element.element_id, element.text_sha1, "topic"))
    return (
        row is not None
        and row.provenance.get("source") == source
        and row.provenance.get("tagger_hash") == fingerprint
    )
