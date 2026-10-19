"""Full topic labelling of calib and gold documents (v0.6 spec §0 ruling 2).

Fakes only (the fixtures of test_label): no torch, mlx or network.
"""

from __future__ import annotations

import pytest
from test_label import PAPER_TEXTS, RecordingTeacher, WordEmbedder, _add

from docrooter.corpus import SKIPPED_KINDS, Corpus
from docrooter.tagger.pipeline import RESERVED_LABELERS, TRAINING_LABELERS, _fingerprint
from docrooter.teach.label import (
    PRESCREEN_LABELER,
    TEACHER_FULL_LABELER,
    TEACHER_LABELER,
    WEAK_LABELER,
    SampleSpec,
    label_corpus,
    sample_elements,
)

FULL = SampleSpec(mode="full", splits=("calib", "gold"))


@pytest.fixture
def corpus(tmp_path) -> Corpus:
    c = Corpus(tmp_path / "store")
    _add(c, "ar0")
    _add(c, "ar1", split="calib")
    _add(c, "paper0", split="calib", doc_type="research.research_paper", rows=PAPER_TEXTS)
    _add(c, "gold0", split="gold")
    _add(c, "gold1", split="gold", doc_type="research.research_paper", rows=PAPER_TEXTS)
    return c


def _content_ids(corpus: Corpus, *splits: str) -> list[tuple[str, str]]:
    return [
        (e.doc_id, e.element_id)
        for e in corpus.iter_elements(split=splits)
        if e.kind not in SKIPPED_KINDS and (e.text.strip() or e.table_markdown)
    ]


def test_full_mode_labels_every_content_element(corpus):
    expected = _content_ids(corpus, "calib", "gold")
    assert len(expected) > 20
    picks = sample_elements(corpus, FULL)
    # every content element, no per-doc cap, none of the train split, in manifest order
    assert [(p.element.doc_id, p.element.element_id) for p in picks] == expected
    assert not any(p.functions for p in picks)
    assert {p.record.split for p in picks} == {"calib", "gold"}

    teacher, embedder = RecordingTeacher(), WordEmbedder()
    report = label_corpus(corpus, teacher, embedder, sample=FULL, functions_share=1.0, batch_size=7)
    assert report.mode == "full" and report.labeler == TEACHER_FULL_LABELER
    assert report.sampled == len(expected) == report.topic_labelled == report.labelled_total
    assert report.function_labelled == 0 and not report.below_floor
    assert report.seconds >= 0 and not report.stopped_by_ceiling
    # topic questions only: no yes/no call, the same family prefixes as the normal mode
    assert teacher.yes_no_calls == []
    assert {c["prefix_key"] for c in teacher.choose_calls} == {
        "topic:annual_report",
        "topic:research_paper",
    }
    rows = corpus.labels(TEACHER_FULL_LABELER)
    assert {(k.doc_id, k.element_id) for k in rows} == set(expected)
    assert all(k.facet == "topic" for k in rows)
    # the normal sampler still skips gold; the full labels agree with the normal mode's keys
    normal = sample_elements(corpus, SampleSpec(target=1000, floor=0, splits=("calib",)))
    assert normal and not any(p.element.doc_id.startswith("gold") for p in normal)


def test_full_labels_in_separate_file_and_never_in_training(corpus):
    label_corpus(corpus, RecordingTeacher(), WordEmbedder(), sample=FULL)
    labels = corpus.dir / "labels"
    assert (labels / f"{TEACHER_FULL_LABELER}.jsonl").is_file()
    # nothing in the training chain's files: teacher, prescreen and weak stay absent
    for name in (TEACHER_LABELER, PRESCREEN_LABELER, WEAK_LABELER):
        assert not (labels / f"{name}.jsonl").exists()
    assert corpus.labelers() == [TEACHER_FULL_LABELER]

    # the training dataset's inputs are an allow-list without teacher-full, and its
    # fingerprint ignores the file
    assert TEACHER_FULL_LABELER not in TRAINING_LABELERS
    assert TEACHER_FULL_LABELER in RESERVED_LABELERS  # tag-corpus may not overwrite it either
    before = _fingerprint(corpus, "embedder", 128)
    label_corpus(
        corpus, RecordingTeacher(), WordEmbedder(), sample=SampleSpec(target=1000, floor=0)
    )
    with_teacher = _fingerprint(corpus, "embedder", 128)
    assert with_teacher != before  # the fingerprint does read teacher and prescreen
    full = labels / f"{TEACHER_FULL_LABELER}.jsonl"
    full.write_text(full.read_text() + full.read_text())
    assert _fingerprint(corpus, "embedder", 128) == with_teacher
    # and the normal run left the full labels alone: no gold document in the training files
    for name in (TEACHER_LABELER, PRESCREEN_LABELER, WEAK_LABELER):
        assert not any(k.doc_id.startswith("gold") for k in corpus.labels(name))


def test_gold_split_allowed_only_in_full_mode():
    with pytest.raises(ValueError, match="gold"):
        SampleSpec(splits=("train", "gold"))
    with pytest.raises(ValueError, match="gold"):
        SampleSpec(mode="sample", splits=("gold",))
    assert SampleSpec(mode="full", splits=("calib", "gold")).splits == ("calib", "gold")
    with pytest.raises(ValueError, match="unknown mode"):
        SampleSpec(mode="everything")
    with pytest.raises(ValueError, match="unknown split"):
        SampleSpec(mode="full", splits=("calib", "dev"))


def test_resume(corpus):
    total = len(_content_ids(corpus, "calib", "gold"))
    first = RecordingTeacher(fail_after=9)  # dies in the second batch of 6
    with pytest.raises(RuntimeError, match="teacher died"):
        label_corpus(corpus, first, WordEmbedder(), sample=FULL, batch_size=6)
    assert len(corpus.labels(TEACHER_FULL_LABELER)) == 6  # the finished batch only

    second = RecordingTeacher()
    report = label_corpus(corpus, second, WordEmbedder(), sample=FULL, batch_size=6)
    assert second.items == total - 6  # nothing finished is scored again
    assert report.skipped == 6 and report.topic_labelled == total - 6
    assert len(corpus.labels(TEACHER_FULL_LABELER)) == total
    lines = (corpus.dir / "labels" / f"{TEACHER_FULL_LABELER}.jsonl").read_text().splitlines()
    assert len(lines) == total  # no key written twice

    third = RecordingTeacher()
    again = label_corpus(corpus, third, WordEmbedder(), sample=FULL)
    assert third.items == 0 and again.skipped == total and again.topic_labelled == 0


def test_cli_full_labels_calib_and_gold(corpus, monkeypatch):
    import json

    from typer.testing import CliRunner

    from docrooter.backends import embedding as embedding_module
    from docrooter.cli import app

    class Embedder(WordEmbedder):
        def __init__(self, model: str) -> None:
            super().__init__()

        def close(self) -> None:
            pass

    monkeypatch.setattr("docrooter.teach.get_teacher", lambda spec: RecordingTeacher())
    monkeypatch.setattr(embedding_module, "EmbeddingBackend", Embedder)
    expected = len(_content_ids(corpus, "calib", "gold"))
    result = CliRunner().invoke(
        app, ["teach", "label", str(corpus.dir), "--full", "--splits", "calib", "--splits", "gold"]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["mode"] == "full" and report["sampled"] == expected
    assert len(corpus.labels(TEACHER_FULL_LABELER)) == expected
    assert corpus.labelers() == [TEACHER_FULL_LABELER]
    stray = CliRunner().invoke(app, ["teach", "label", str(corpus.dir), "--splits", "gold"])
    assert stray.exit_code == 1 and "--full" in stray.output
