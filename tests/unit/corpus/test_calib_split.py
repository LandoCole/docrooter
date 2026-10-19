"""The calib-fit / calib-crc partition of the ``calib`` split (v0.6 builder-check ruling):
deterministic halves by document, the calibration fitted on calib-fit only, and no calib-crc
document on any training, active-learning or term-mining path. Fake encoders, taggers and
teachers; no model is ever loaded."""

import hashlib
import json

import numpy as np
import pytest

from docrooter.corpus import Corpus, CorpusElement, LabelRow, ManifestRecord, calib_half
from docrooter.corpus.store import is_calib_crc
from docrooter.elements import TagResult
from docrooter.tagger import pipeline
from docrooter.tagger.functions import function_rows
from docrooter.tagger.train import active_learning_round
from docrooter.taxonomies import builtin_taxonomy
from docrooter.teach.confidence import TrainingExample, TrainingSet
from docrooter.teach.dataset import build_training_set, to_dataset
from docrooter.teach.enrich import mine_terms

LEAF = "annual_report.risk_factors"


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _ids(half, n, prefix):
    """The first ``n`` ids ``<prefix>-<i>`` in calib ``half``."""
    out, i = [], 0
    while len(out) < n:
        doc_id = f"{prefix}-{i}"
        if calib_half(doc_id) == half:
            out.append(doc_id)
        i += 1
    return out


FIT = _ids("fit", 2, "cal")
CRC = _ids("crc", 2, "cal")
TRAIN_CRC_HASH = _ids("crc", 1, "tr")[0]  # a train document whose id hashes to "crc"


def test_calib_half_is_deterministic_and_about_half_each():
    ids = [f"doc-{i:05d}" for i in range(4000)]
    halves = [calib_half(i) for i in ids]
    assert halves == [calib_half(i) for i in ids]  # deterministic
    assert set(halves) == {"fit", "crc"}
    assert 0.47 < halves.count("fit") / len(ids) < 0.53
    for doc_id in ids[:50]:  # the stated rule, computed independently
        digest = hashlib.sha256(f"calibsplit:{doc_id}".encode()).digest()
        expected = "fit" if int.from_bytes(digest[:8], "big") < 2**63 else "crc"
        assert calib_half(doc_id) == expected


def test_only_calib_documents_are_partitioned():
    record = ManifestRecord(id=TRAIN_CRC_HASH, sha256=_sha("x"), split="train")
    assert calib_half(TRAIN_CRC_HASH) == "crc" and not is_calib_crc(record)
    assert is_calib_crc(record.model_copy(update={"split": "calib"}))


# --- a store with both halves -----------------------------------------------------------------


def _elements(doc_id, marker, n=25):
    return [
        CorpusElement(
            doc_id=doc_id,
            element_id=f"e{i}",
            kind="paragraph",
            text=f"Rates may rise and {marker} suppliers fail ({doc_id} {i}).",
            heading_path=["Risk Factors"],
            section_title="Risk Factors",
            page=1,
        )
        for i in range(n)
    ]


@pytest.fixture
def store(tmp_path):
    """Train documents (one whose id hashes to crc: train is not partitioned), two calib-fit
    and two calib-crc documents; teacher topic and function rows and prescreen rows for every
    element, calib-crc included."""
    corpus = Corpus(tmp_path / "store")
    docs = [("train-a", "train", "fitmarker"), (TRAIN_CRC_HASH, "train", "fitmarker")]
    docs += [(d, "calib", "fitmarker") for d in FIT] + [(d, "calib", "crcmarker") for d in CRC]
    for doc_id, split, marker in docs:
        record = ManifestRecord(
            id=doc_id,
            sha256=_sha(doc_id),
            split=split,
            doc_type_hint="financial.annual_report",
            family_hint="annual_report",
        )
        corpus.add_document(record, _elements(doc_id, marker), [1])
    rows = []
    for e in corpus.iter_elements():
        for facet, probs in (("topic", {LEAF: 0.95}), ("function", {"risk_disclosure": 0.9})):
            rows.append(LabelRow(doc_id=e.doc_id, element_id=e.element_id,
                                 text_sha1=e.text_sha1, facet=facet, probs=probs,
                                 provenance={"teacher": "fake"}))  # fmt: skip
    corpus.write_labels("teacher", rows)
    corpus.write_labels(
        "prescreen",
        [r.model_copy(update={"probs": {"citation": 0.1}}) for r in rows if r.facet == "function"],
    )
    return corpus


class SpyEncoder:
    """Records every text it encodes; vectors by the text's last word."""

    def __init__(self, *args, **kwargs):
        self.name = "spy"
        self.texts = []

    def encode(self, texts, *, prompt=""):
        self.texts.extend(texts)
        out = np.zeros((len(texts), 8), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i, sum(map(ord, text)) % 8] = 1.0
        return out

    def close(self):
        pass


def test_iter_fit_elements_drops_calib_crc_only(store):
    fit_docs = {e.doc_id for e in store.iter_fit_elements()}
    assert fit_docs == {"train-a", TRAIN_CRC_HASH, *FIT}
    assert {e.doc_id for e in store.iter_fit_elements(split="calib")} == set(FIT)
    assert {e.doc_id for e in store.iter_elements(split="calib")} == {*FIT, *CRC}  # unchanged


def test_calibration_rows_come_only_from_calib_fit(store, tmp_path, monkeypatch):
    seen = {}

    class Student:
        def __init__(self):
            self.info, self.temperatures = {"untrained_leaves": []}, {"topic": 1.0}

        def save(self, path):
            (path / "tagger.json").write_text(json.dumps({"info": self.info}), encoding="utf-8")

        def close(self):
            pass

    def calibrate_tagger(tagger, calib, rows):
        seen["calib"], seen["rows"] = list(calib), list(rows)
        return {"topic": {"n": len(calib)}}

    monkeypatch.setattr(pipeline, "label_corpus", lambda *a, **k: pytest.fail("no labelling"))
    monkeypatch.setattr(pipeline, "EmbeddingBackend", SpyEncoder)
    monkeypatch.setattr(pipeline, "train_tagger", lambda *a, **k: Student())
    monkeypatch.setattr(pipeline, "calibrate_tagger", calibrate_tagger)
    work = store.dir / pipeline.WORK_DIR
    work.mkdir()  # labelling done before: the real dataset stage, then training
    sample = pipeline.SampleSpec(per_doc=60, target=10, floor=0)
    settings = {"teacher": "mlx:o/m@r", "embedder": "spy", "per_doc": 60, "target": 10,
                "floor": 0, "splits": ["train", "calib"], "functions_share": 0.5}  # fmt: skip
    (work / "label.json").write_text(json.dumps({"settings": settings, "status": "complete"}))
    out = tmp_path / "bundle"
    report = pipeline.train_elements(
        store, "mlx:o/m@r", out, embedder="spy", sample=sample, functions_share=0.5
    )

    assert {e.doc_id for e in seen["calib"]} == set(FIT)  # every fit document, no crc one
    assert len(seen["calib"]) == 50
    assert seen["rows"] and all("crcmarker" not in row.text for row in seen["rows"])
    assert len(seen["rows"]) == 50  # the calib-fit function rows
    for name in ("calib.jsonl", "functions-calib.jsonl", "train.jsonl", "functions-train.jsonl"):
        text = (work / name).read_text()
        assert "crcmarker" not in text and not any(d in text for d in CRC), name
    data = {"split": "calib-fit", "documents": 2, "elements": 50, "function_rows": 50}
    assert json.loads((out / "tagger.json").read_text())["info"]["calibration_data"] == data
    assert report["train"]["calibration_data"] == data


def test_calib_crc_never_reaches_training_term_mining_or_active_learning(store):
    taxonomy = builtin_taxonomy("element_topics")
    labels = store.labels("teacher")

    # term mining and the training set, even when asked for the calib split
    for split in ("train", "calib"):
        encoder = SpyEncoder()
        terms = mine_terms(store, labels, taxonomy, encoder, split=split, k=50)
        mined = {t for words in terms.values() for t in words}
        assert not any("crcmarker" in t for t in encoder.texts + sorted(mined)), split
        encoder = SpyEncoder()
        _, training_set = build_training_set(store, labels, taxonomy, encoder, split=split)
        assert not any("crcmarker" in t for t in encoder.texts), split
        assert any("fitmarker" in t for t in encoder.texts), split  # the fit half is read
        assert {e.doc_id for e in training_set.examples}.isdisjoint(CRC)

    # function rows: calib means calib-fit
    rows = function_rows(store, labels, store.labels("prescreen"), split="calib", max_chars=4096)
    assert len(rows) == 50 and all("crcmarker" not in r.text for r in rows)

    # a training set naming a calib-crc element cannot become a dataset
    crc = next(e for e in store.iter_elements(split="calib") if e.doc_id in CRC)
    rogue = TrainingSet(examples=[TrainingExample(
        doc_id=crc.doc_id, element_id=crc.element_id, text_sha1=crc.text_sha1, label=LEAF,
        path=["annual_report", LEAF], teacher_prob=0.9, gap=0.5, in_core=True, confidence=0.9,
    )])  # fmt: skip
    with pytest.raises(ValueError, match="not in the corpus"):
        to_dataset(rogue, store)

    # active learning over the calib split: calib-crc is never scored or asked about
    class Tagger:
        taxonomy = builtin_taxonomy("element_topics")
        untrained: list = []
        source = "fake"
        tagged: list = []

        def tag(self, elements, family):
            self.tagged.extend(elements)
            probs = {LEAF: 0.6, "annual_report.other": 0.4}
            return [{"topic": TagResult(labels=[LEAF], probs=probs, confidence=0.6,
                                        source="fake")} for _ in elements]  # fmt: skip

    class Teacher:
        name = "mlx:o/m@r#native"
        asked: list = []

        def choose(self, items, **kwargs):
            self.asked.extend(items)
            return [{tag: 1.0 / len(i.options) for tag, _ in i.options} for i in items]

    tagger, teacher = Tagger(), Teacher()
    picked = active_learning_round(tagger, store, teacher, n=1000, split="calib", labeled={})
    assert picked and {r.doc_id for r in picked} == set(FIT)
    assert {e.doc_id for e in tagger.tagged} == set(FIT)
    assert not any("crcmarker" in item.text for item in teacher.asked)
