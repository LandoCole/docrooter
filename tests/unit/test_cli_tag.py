"""``docrooter tag``, ``tag-corpus`` and ``train-elements`` (spec §10): the JSON and table
output, the corpus labels file and its resume, the resumable train chain and the Jev refusal.
Fake taggers and fake L7-L9 stages; no model is ever loaded."""

import hashlib
import json

import pytest
from typer.testing import CliRunner

from docrooter.cli import app
from docrooter.corpus import Corpus, CorpusElement, LabelRow, ManifestRecord
from docrooter.elements import TagResult
from docrooter.ocr import registry as ocr_registry
from docrooter.tagger import element_tags, pipeline
from docrooter.teach.confidence import TrainingSet
from docrooter.train import Dataset, Example

runner = CliRunner()

CONFIG = 'ocr = "none"\ndoc_type = "keyword"\nsections = "keyword"\n'
MARKDOWN = """# Annual Report

The company grew this year. Revenue rose by a third and costs fell.

## Risk Factors

- Interest rates may rise.
- Suppliers may fail to deliver.

| Year | Revenue |
| ---- | ------- |
| 2025 | 133 |
"""


@pytest.fixture(autouse=True)
def _no_real_ocr(monkeypatch):
    monkeypatch.setattr(ocr_registry, "resolve_auto_ocr", lambda: None)


class FakeTagger:
    source = "bundle:fake"

    def __init__(self):
        self.calls = []
        self.closed = 0

    def tag(self, elements, family):
        self.calls.append((len(elements), family))
        leaf = f"{family}.other"
        return [
            {
                "topic": TagResult(
                    labels=[leaf],
                    probs={leaf: 0.75},
                    confidence=0.75,
                    source=self.source,
                    family_probs={family: 0.6},
                ),
                "function": TagResult(
                    labels=["risk_disclosure"],
                    probs={"risk_disclosure": 0.9, "citation": 0.05},
                    confidence=0.9,
                    source=self.source,
                ),
            }
            for _ in elements
        ]

    def close(self):
        self.closed += 1


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    path = tmp_path / "bundle"
    path.mkdir()
    (path / "tagger.json").write_text("{}", encoding="utf-8")
    made = []

    def load(spec):
        made.append(FakeTagger())
        return made[-1]

    monkeypatch.setattr(element_tags, "load_bundle", load)
    return path, made


@pytest.fixture
def files(tmp_path):
    doc = tmp_path / "report.md"
    doc.write_text(MARKDOWN, encoding="utf-8")
    config = tmp_path / "rooter.toml"
    config.write_text(CONFIG, encoding="utf-8")
    return doc, config


def test_tag_cli_json(files, bundle):
    doc, config = files
    path, made = bundle
    result = runner.invoke(
        app, ["tag", str(doc), "--bundle", str(path), "--config", str(config), "--json"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert set(data) == {"file", "doc_type", "family", "element_tags", "elements"}
    assert data["element_tags"] == str(path) and data["file"] == str(doc)
    kinds = [e["kind"] for e in data["elements"]]
    assert kinds == ["paragraph", "list_item", "list_item", "table"]  # content kinds only
    first = data["elements"][0]
    assert first["tags"]["topic"]["labels"] == [f"{data['family']}.other"]
    assert first["tags"]["topic"]["family_probs"] == {data["family"]: 0.6}
    assert first["tags"]["function"]["labels"] == ["risk_disclosure"]
    assert {"id", "kind", "text", "section_id", "boxes"} <= set(first)
    assert len(made) == 1 and made[0].closed == 1  # loaded once, closed at the end
    assert made[0].calls == [(4, data["family"])]


def test_tag_cli_table(files, bundle):
    doc, config = files
    path, _ = bundle
    result = runner.invoke(app, ["tag", str(doc), "--bundle", str(path), "--config", str(config)])
    assert result.exit_code == 0, result.output
    for header in ("Kind", "Topic", "Functions", "Text"):
        assert header in result.stdout
    assert "other 75%" in result.stdout and "risk_disclosure 90%" in result.stdout
    assert "4 of 6 elements tagged" in result.stdout


def test_tag_cli_errors(files, bundle, tmp_path):
    doc, config = files
    missing = runner.invoke(
        app, ["tag", str(doc), "--bundle", str(tmp_path / "none"), "--config", str(config)]
    )
    assert missing.exit_code == 1 and "no element-tagger bundle" in missing.output
    jev = runner.invoke(
        app, ["tag", str(doc), "--bundle", "teacher:jev:jev-1.13.0", "--config", str(config)]
    )
    assert jev.exit_code == 1 and "2.3(b)" in jev.output
    fast = runner.invoke(app, ["tag", str(doc), "--bundle", str(bundle[0]), "--profile", "fast"])
    assert fast.exit_code == 1 and "builds none" in fast.output


# --- corpus stores ----------------------------------------------------------------------------


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _element(doc_id, i, kind, text):
    return CorpusElement(
        doc_id=doc_id,
        element_id=f"e{i}",
        kind=kind,
        text=text,
        heading_path=["Risk Factors"],
        section_title="Risk Factors",
        page=1,
    )


def _store(directory):
    corpus = Corpus(directory)
    docs = [
        ("ar1", "train", "financial.annual_report", "annual_report"),
        ("paper1", "calib", "research.research_paper", "research_paper"),
    ]
    for doc_id, split, doc_type, family in docs:
        record = ManifestRecord(
            id=doc_id, sha256=_sha(doc_id), split=split, doc_type_hint=doc_type, family_hint=family
        )
        elements = [
            _element(doc_id, 0, "heading", "Risk Factors"),
            _element(doc_id, 1, "paragraph", f"Markets may fall ({doc_id})."),
            _element(doc_id, 2, "list_item", "Rates may rise."),
            _element(doc_id, 3, "figure", "chart"),
        ]
        corpus.add_document(record, elements, [1])
    return corpus


def test_tag_corpus_writes_a_labels_file_and_resumes(tmp_path, bundle):
    path, made = bundle
    corpus = _store(tmp_path / "store")
    result = runner.invoke(app, ["tag-corpus", str(corpus.dir), "--bundle", str(path)])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["tagged"] == 4 and report["skipped"] == 0 and report["documents"] == 2
    rows = corpus.labels("tagger")
    assert {(k.doc_id, k.element_id, k.facet) for k in rows} == {
        (d, e, f) for d in ("ar1", "paper1") for e in ("e1", "e2") for f in ("topic", "function")
    }
    topic = next(r for k, r in rows.items() if k.doc_id == "ar1" and k.facet == "topic")
    assert topic.probs == {"annual_report.other": 0.75}
    assert topic.provenance["source"] == "bundle:fake"
    assert topic.provenance["family"] == "annual_report"
    assert topic.provenance["family_probs"] == {"annual_report": 0.6}
    assert made[0].calls == [(2, "annual_report"), (2, "research_paper")]
    assert made[0].closed == 1

    again = runner.invoke(app, ["tag-corpus", str(corpus.dir), "--bundle", str(path)])
    assert again.exit_code == 0, again.output
    assert json.loads(again.stdout)["skipped"] == 4 and made[1].calls == []

    only = runner.invoke(
        app,
        ["tag-corpus", str(corpus.dir), "--bundle", str(path), "--split", "calib",
         "--labeler", "student"],
    )  # fmt: skip
    assert only.exit_code == 0, only.output
    assert {k.doc_id for k in corpus.labels("student")} == {"paper1"}


def test_tag_corpus_needs_a_store(tmp_path, bundle):
    result = runner.invoke(app, ["tag-corpus", str(tmp_path), "--bundle", str(bundle[0])])
    assert result.exit_code == 1 and "no corpus" in result.output


# --- train-elements ---------------------------------------------------------------------------


class FakeStudent:
    def __init__(self, stage):
        self.info = {"stage": stage, "untrained_leaves": []}
        self.temperatures = {"topic": 1.0, "function": 1.0}
        self.closed = 0

    def save(self, path):
        (path / "tagger.json").write_text(json.dumps({"info": self.info}), encoding="utf-8")

    def close(self):
        self.closed += 1


class FakeClosable:
    def __init__(self, *args, **kwargs):
        self.name = "fake"

    def close(self):
        pass


@pytest.fixture
def stages(monkeypatch):
    """The L7-L9 stages replaced by fakes that count their calls; ``fail["train"]`` makes
    the next training raise, as a run killed while training."""
    calls = {"label": 0, "dataset": 0, "train": 0, "calibrate": 0}
    fail = {"train": False}

    def label_corpus(corpus, teacher, embedder, **kwargs):
        calls["label"] += 1
        from docrooter.teach.label import LabelReport

        rows = []
        for element in corpus.iter_elements(kinds=["paragraph"]):
            leaf = "annual_report.risk_factors"
            if element.doc_id == "paper1":
                leaf = "research_paper.other"
            rows.append(
                LabelRow(
                    doc_id=element.doc_id,
                    element_id=element.element_id,
                    text_sha1=element.text_sha1,
                    facet="topic",
                    probs={leaf: 0.9},
                    provenance={"teacher": teacher.name},
                )
            )
        corpus.write_labels("teacher", rows)
        return LabelReport(teacher=teacher.name, embedder="fake", target=1, floor=0,
                           ceiling_hours=1.0, topic_labelled=len(rows))  # fmt: skip

    def build_training_set(corpus, labels, taxonomy, embedder, **kwargs):
        calls["dataset"] += 1
        return taxonomy, TrainingSet()

    def to_dataset(training_set, corpus):
        return Dataset([Example(text="x", path=["annual_report", "annual_report.other"])])

    def train_tagger(dataset, rows, *, stage, **kwargs):
        calls["train"] += 1
        if fail["train"]:
            fail["train"] = False
            raise RuntimeError("killed mid-training")
        return FakeStudent(stage)

    def calibrate_tagger(tagger, calib, rows):
        calls["calibrate"] += 1
        calls["calib_paths"] = [e.path for e in calib]
        return {"topic": {"n": len(calib)}}

    monkeypatch.setattr(pipeline, "get_teacher", lambda spec: FakeClosable())
    monkeypatch.setattr(pipeline, "EmbeddingBackend", FakeClosable)
    monkeypatch.setattr(pipeline, "label_corpus", label_corpus)
    monkeypatch.setattr(pipeline, "build_training_set", build_training_set)
    monkeypatch.setattr(pipeline, "to_dataset", to_dataset)
    monkeypatch.setattr(pipeline, "train_tagger", train_tagger)
    monkeypatch.setattr(pipeline, "calibrate_tagger", calibrate_tagger)
    return calls, fail


def test_train_elements_cli_resumes_stages(tmp_path, stages):
    calls, fail = stages
    corpus = _store(tmp_path / "store")
    out = tmp_path / "bundle"
    args = ["train-elements", str(corpus.dir), "--teacher", "mlx:org/model@rev", "--out", str(out)]

    fail["train"] = True
    killed = runner.invoke(app, args)
    assert isinstance(killed.exception, RuntimeError)
    assert (calls["label"], calls["dataset"], calls["train"]) == (1, 1, 1)
    work = corpus.dir / "train-elements"
    assert (work / "label.json").is_file() and (work / "dataset.json").is_file()
    assert not (out / "tagger.json").exists()

    resumed = runner.invoke(app, args)
    assert resumed.exit_code == 0, resumed.output
    report = json.loads(resumed.stdout)
    assert report["label"] == "skipped" and report["dataset"] == "skipped"
    assert report["train"]["calibration"] == {"topic": {"n": 1}}
    assert (calls["label"], calls["dataset"], calls["train"], calls["calibrate"]) == (1, 1, 2, 1)
    assert calls["calib_paths"] == [["research_paper", "research_paper.other"]]
    info = json.loads((out / "tagger.json").read_text())["info"]
    assert info["train_elements"]["stage"] == "B"

    done = runner.invoke(app, args)
    assert done.exit_code == 0, done.output
    assert json.loads(done.stdout)["train"] == "skipped"
    assert calls["train"] == 2

    stage_a = runner.invoke(app, [*args, "--stage", "A"])  # another stage trains again
    assert stage_a.exit_code == 0, stage_a.output
    assert calls["train"] == 3 and calls["label"] == 1 and calls["dataset"] == 1

    # new teacher labels change the dataset's fingerprint: it is rebuilt, then retrained
    element = next(corpus.iter_elements(kinds=["list_item"]))
    corpus.write_labels(
        "teacher",
        [LabelRow(doc_id=element.doc_id, element_id=element.element_id,
                  text_sha1=element.text_sha1, facet="topic",
                  probs={"annual_report.other": 0.8}, provenance={"teacher": "fake"})],
    )  # fmt: skip
    rebuilt = runner.invoke(app, [*args, "--stage", "A"])
    assert rebuilt.exit_code == 0, rebuilt.output
    assert calls["dataset"] == 2 and calls["train"] == 4 and calls["label"] == 1


def test_train_elements_refuses_jev_before_anything(tmp_path, stages):
    calls, _ = stages
    corpus = _store(tmp_path / "store")
    for teacher in ("jev:jev-1.13.0", "openai:jev-1.13.0", "hf:typesafe/jev@abc"):
        result = runner.invoke(
            app,
            ["train-elements", str(corpus.dir), "--teacher", teacher, "--out",
             str(tmp_path / "b")],
        )  # fmt: skip
        assert result.exit_code == 1 and "2.3(b)" in result.output
    assert calls == {"label": 0, "dataset": 0, "train": 0, "calibrate": 0}
    assert not (corpus.dir / "train-elements").exists()
