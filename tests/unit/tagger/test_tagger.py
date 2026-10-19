"""The element tagger with a fake embedding body and a small topic tree: no torch."""

from __future__ import annotations

import numpy as np
import pytest

from docrooter.backends import embedding
from docrooter.backends.recipes import get_recipe
from docrooter.corpus.store import Corpus, CorpusElement, LabelKey, LabelRow, ManifestRecord
from docrooter.errors import TeacherNotAllowedError
from docrooter.taxonomy import Taxonomy
from docrooter.teach.dataset import input_text
from docrooter.train import Dataset, Example
from docrooter.train import trainer as trainer_module

pytest.importorskip("sklearn.linear_model")

from docrooter.tagger import (  # noqa: E402
    ElementTagger,
    FunctionRow,
    active_learning_round,
    calibrate_tagger,
    function_rows,
    function_targets,
    select_uncertain,
    train_function_heads,
    train_tagger,
)
from docrooter.tagger.bundle import CHARS_PER_TOKEN  # noqa: E402
from docrooter.tagger.train import Uncertain, trained_tree  # noqa: E402

BASE = "BAAI/bge-small-en-v1.5"
WORDS = ("risk", "money", "rare", "misc", "method", "cite", "aside", "alpha", "beta")
TAX = Taxonomy.from_dict(
    {
        "nodes": [
            {
                "id": "report",
                "label": "Report",
                "children": [
                    {"id": "report.risk", "label": "Risk"},
                    {"id": "report.money", "label": "Money"},
                    {"id": "report.rare", "label": "Rare"},
                    {"id": "report.other", "label": "Other"},
                ],
            },
            {
                "id": "paper",
                "label": "Paper",
                "children": [
                    {"id": "paper.method", "label": "Method"},
                    {"id": "paper.cite", "label": "Citation"},
                    {"id": "paper.other", "label": "Other"},
                ],
            },
            {
                "id": "tiny",
                "label": "Tiny",
                "children": [
                    {"id": "tiny.alpha", "label": "Alpha"},
                    {"id": "tiny.beta", "label": "Beta"},
                ],
            },
        ]
    }
)
# leaf -> (word, examples); report.rare and both tiny leaves are under the 20 needed
SPEC = {
    "report.risk": ("risk", 25),
    "report.money": ("money", 25),
    "report.rare": ("rare", 5),
    "report.other": ("misc", 25),
    "paper.method": ("method", 25),
    "paper.cite": ("cite", 25),
    "paper.other": ("aside", 25),
    "tiny.alpha": ("alpha", 12),
    "tiny.beta": ("beta", 12),
}


class FakeEmbedding:
    instances: list[FakeEmbedding] = []

    def __init__(self, model, *, device=None, batch_size=None, cache_size=None, **kw):
        kw.pop("trust_remote_code", None)
        self.name = f"embedding:{model}"
        self.model_id = model
        self.recipe = get_recipe(BASE if "/" not in model or "body" in model else model, **kw)
        if batch_size is not None:
            self.recipe = self.recipe.model_copy(update={"batch_size": batch_size})
        self.seen: list[str] = []
        self._device = device
        self._device_in_use = device or "cpu"
        FakeEmbedding.instances.append(self)

    @property
    def batch_size(self):
        return self.recipe.batch_size_for(self._device_in_use)

    def close(self):
        pass

    def tokenizer(self):
        return None

    def encode(self, texts, *, prompt):
        self.seen.extend(texts)
        rows = [[float(w in t.lower()) for w in WORDS] + [0.1] for t in texts]
        return np.array(rows, dtype=np.float32)

    def _load(self):
        return _Saver()


class _Saver:
    def save(self, path):
        from pathlib import Path

        Path(path).mkdir(parents=True, exist_ok=True)


@pytest.fixture(autouse=True)
def fake_body(monkeypatch):
    FakeEmbedding.instances.clear()
    monkeypatch.setattr(embedding, "EmbeddingBackend", FakeEmbedding)


def _element(text, *, kind="paragraph", path=("Section",), doc="d1", eid="e0", table=None):
    return CorpusElement(
        doc_id=doc,
        element_id=eid,
        kind=kind,
        text=text,
        heading_path=list(path),
        section_title=path[-1] if path else None,
        page=1,
        table_markdown=table,
    )


def _dataset(spec=SPEC) -> Dataset:
    """Examples as ``to_dataset`` gives them: ``[family, leaf]`` paths, texts from
    ``input_text``."""
    examples = []
    for leaf, (word, n) in spec.items():
        family = leaf.split(".")[0]
        for i in range(n):
            el = _element(f"{word} item {i}", eid=f"{leaf}-{i}")
            examples.append(Example(text=input_text(el), path=[family, leaf]))
    return Dataset(examples)


def _tagger(**kw) -> ElementTagger:
    return train_tagger(_dataset(), [], stage="A", base_model=BASE, taxonomy=TAX, **kw)


# --- Review Focus 4 ---------------------------------------------------------------------------


def test_sparse_class_falls_back_to_family():
    tagger = _tagger()
    heads = tagger.topic.manifest["heads"]
    # the sparse leaf is left out of its family's head; the family keeps a head for the rest
    assert heads["report"]["classes"] == ["report.money", "report.other", "report.risk"]
    assert heads["report"]["untrained"] == ["report.rare"]
    # a family whose leaves are all sparse has no head, but the root still knows it
    assert "tiny" not in heads
    assert heads[Taxonomy.ROOT_ID]["classes"] == ["paper", "report", "tiny"]
    assert tagger.untrained == ["report.rare", "tiny.alpha", "tiny.beta"]
    assert tagger.info["untrained_leaves"] == tagger.untrained

    rare, tiny = _element("rare item"), _element("alpha item", eid="e1")
    for tags in tagger.tag([rare, tiny], "report") + tagger.tag([rare, tiny]):
        assert "report.rare" not in tags["topic"].labels
        assert "report.rare" not in tags["topic"].probs
    (on_report,) = tagger.tag([rare], "report")
    assert on_report["topic"].labels[0] in heads["report"]["classes"]
    # the tiny family is predicted only at the family level, with its own probability
    (on_tiny,) = tagger.tag([tiny], "tiny")
    topic = on_tiny["topic"]
    assert topic.labels == ["tiny"]
    assert topic.probs == {"tiny": topic.family_probs["tiny"]} and topic.confidence > 0.5
    (free,) = tagger.tag([tiny])
    assert free["topic"].labels == ["tiny"]


def test_without_partial_heads_the_whole_family_would_be_zero_shot():
    """The default trainer rule (every child needs examples) is unchanged."""
    nodes = trainer_module.trainable_nodes(TAX, _dataset(), min_examples_per_child=20)
    assert "report" not in nodes
    partial = trainer_module.trainable_nodes(
        TAX, _dataset(), min_examples_per_child=20, allow_partial=True
    )
    assert partial["report"] == {"report.risk": 25, "report.money": 25, "report.other": 25}
    assert "tiny" not in partial


# --- topic restriction ------------------------------------------------------------------------


def test_tag_restricts_topic_to_family_and_keeps_family_probs():
    tagger = _tagger()
    risk = _element("risk item")
    (free,) = tagger.tag([risk])
    (restricted,) = tagger.tag([risk], "paper")
    assert free["topic"].labels == ["report.risk"]
    assert restricted["topic"].labels[0].startswith("paper.")
    assert set(restricted["topic"].probs) == {"paper.method", "paper.cite", "paper.other"}
    assert sum(restricted["topic"].probs.values()) == pytest.approx(1.0, abs=1e-5)
    # the family distribution is the unconstrained one either way, over every family
    assert restricted["topic"].family_probs == free["topic"].family_probs
    fam = free["topic"].family_probs
    assert set(fam) == {"report", "paper", "tiny"} and max(fam, key=fam.get) == "report"
    # Beam(3): at most three paths, joint probabilities, the best one reported
    assert len(free["topic"].probs) == 3
    assert free["topic"].confidence == max(free["topic"].probs.values())
    assert free["topic"].confidence <= fam["report"] + 1e-6
    with pytest.raises(ValueError, match="unknown element family"):
        tagger.tag([risk], "report.risk")


# --- Review Focus 3 ---------------------------------------------------------------------------


def test_truncation_and_table_inputs():
    table = _element(
        "a b c d", kind="table", path=("Fees",), table="| Fee | Rate |\n|---|---|\n| risk | 1% |"
    )
    huge = _element("money " * 50_000, eid="e1", path=())
    examples = [Example(text=input_text(huge), path=["report", "report.money"])]
    tagger = train_tagger(
        Dataset(list(_dataset()) + examples), [], stage="A", base_model=BASE, taxonomy=TAX
    )
    body = FakeEmbedding.instances[-1]
    max_chars = CHARS_PER_TOKEN * 256
    assert tagger.max_chars == max_chars
    assert tagger.topic.manifest["recipe"]["max_seq_len"] == 256  # the body's token cut
    trained_on = list(body.seen)
    body.seen.clear()

    out = tagger.tag([table, huge], "report")
    assert len(out) == 2 and all("topic" in tags for tags in out)
    table_in, huge_in = body.seen
    # the table is its Markdown, after the heading path and kind (teach.dataset.input_text)
    assert table_in == input_text(table) == "Fees\n\n[table] " + table.table_markdown
    # a huge element is cut, and exactly as it was cut for training (byte for byte)
    assert huge_in == input_text(huge)[:max_chars] and len(huge_in) == max_chars
    assert huge_in in trained_on


# --- function heads ---------------------------------------------------------------------------


def test_function_head_training_excludes_unasked():
    tags = ["risk_disclosure", "financial_data"]
    rows = [
        FunctionRow(text="0", answers={"risk_disclosure": 0.9}, prescreen={"financial_data": 0.5}),
        FunctionRow(text="1", answers={"risk_disclosure": 0.1}, prescreen={"financial_data": 0.1}),
        FunctionRow(text="2", answers={"risk_disclosure": 0.5, "financial_data": 0.8}),
        FunctionRow(text="3", answers={}, prescreen={"risk_disclosure": 0.3}),
        FunctionRow(text="4", answers={"financial_data": 0.2}, prescreen={"risk_disclosure": 0.0}),
    ]
    targets = function_targets(rows, tags)
    # risk: 0 yes, 1 no, 2 unsure (out), 3 unasked above the prescreen cut (out), 4 implicit no
    assert targets["risk_disclosure"] == [(0, 1), (1, 0), (4, 0)]
    # financial: 0 unasked at 0.5 (out), 1 implicit no, 2 yes, 3 no prescreen (out), 4 no
    assert targets["financial_data"] == [(1, 0), (2, 1), (4, 0)]
    with pytest.raises(ValueError, match="unknown tag"):
        function_targets([FunctionRow(text="x", answers={"nope": 1.0})], tags)

    rng = np.random.default_rng(0)
    many = [
        FunctionRow(text=str(i), answers={"risk_disclosure": float(i % 2)}) for i in range(20)
    ] + [FunctionRow(text="u", answers={}, prescreen={"risk_disclosure": 0.9})] * 5
    X = rng.normal(size=(len(many), 4))
    X[:20, 0] += np.array([3.0 if i % 2 else -3.0 for i in range(20)])
    heads = train_function_heads(X, many, tags=tags)
    # financial_data was never asked and never implicitly negative: no head
    assert set(heads) == {"risk_disclosure"}
    head = heads["risk_disclosure"]
    assert list(head.classes_) == [0, 1] and head.n_features_in_ == 4
    assert int(head.coef_.shape[0]) == 1
    # the five unasked rows (prescreen 0.9) were not used: refit without them is identical
    alone = train_function_heads(X[:20], many[:20], tags=tags)["risk_disclosure"]
    assert np.allclose(alone.coef_, head.coef_)


def test_function_rows_join_teacher_and_prescreen(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    elements = [_element("risk item", eid="e0"), _element("money item", eid="e1")]
    corpus.add_document(ManifestRecord(id="d1", sha256="0" * 64), elements, [1])
    e0 = elements[0]
    teacher = [
        LabelRow(
            doc_id="d1",
            element_id="e0",
            text_sha1=e0.text_sha1,
            facet="function",
            probs={"risk_disclosure": 0.9},
            provenance={"teacher": "mlx:some/model@abc"},
        )
    ]
    prescreen = [
        LabelRow(
            doc_id="d1",
            element_id="e0",
            text_sha1=e0.text_sha1,
            facet="function",
            probs={"financial_data": 0.05},
        )
    ]
    (row,) = function_rows(corpus, teacher, prescreen, max_chars=10)
    assert row.text == input_text(e0)[:10]
    assert row.answers == {"risk_disclosure": 0.9} and row.prescreen == {"financial_data": 0.05}
    jev = [teacher[0].model_copy(update={"provenance": {"teacher": "jev:jev-1"}})]
    with pytest.raises(TeacherNotAllowedError):
        function_rows(corpus, jev, prescreen, max_chars=10)


def _function_rows() -> list[FunctionRow]:
    rows = []
    for word, n in SPEC.values():
        for i in range(n):
            text = input_text(_element(f"{word} item {i}"))
            rows.append(
                FunctionRow(
                    text=text,
                    answers={"risk_disclosure": 0.95 if word == "risk" else 0.05},
                    prescreen={"financial_data": 0.5},
                )
            )
    return rows


# --- bundle -----------------------------------------------------------------------------------


def test_bundle_round_trip_with_temperatures(tmp_path):
    tagger = train_tagger(_dataset(), _function_rows(), stage="A", base_model=BASE, taxonomy=TAX)
    assert tagger.function_tags == ["risk_disclosure"]
    elements = [_element("risk item"), _element("cite item", eid="e1")]
    plain = tagger.tag(elements, "report")
    tagger.temperatures = {"topic": 0.5, "function": 2.0}
    tempered = tagger.tag(elements, "report")
    assert tempered[0]["topic"].probs != plain[0]["topic"].probs  # the temperature applies
    assert tempered[0]["function"].probs != plain[0]["function"].probs

    tagger.save(tmp_path / "element-tagger-v1")
    assert (tmp_path / "element-tagger-v1" / "tagger.json").is_file()
    loaded = ElementTagger.load(tmp_path / "element-tagger-v1")
    assert loaded.temperatures == {"topic": 0.5, "function": 2.0}
    assert loaded.function_tags == ["risk_disclosure"]
    assert loaded.max_chars == tagger.max_chars and loaded.info == tagger.info
    assert loaded.taxonomy.content_hash() == TAX.content_hash()
    again = loaded.tag(elements, "report")
    for a, b in zip(again, tempered, strict=True):
        assert a["topic"].probs == b["topic"].probs
        assert a["topic"].family_probs == b["topic"].family_probs
        assert a["function"].probs == b["function"].probs
        assert a["topic"].source == "bundle:element-tagger-v1"
    risk = again[0]["function"]
    assert risk.labels == ["risk_disclosure"] and risk.probs["risk_disclosure"] > 0.5
    assert again[1]["function"].labels == []


def test_calibrate_sets_one_temperature_per_facet():
    tagger = train_tagger(_dataset(), _function_rows(), stage="A", base_model=BASE, taxonomy=TAX)
    report = calibrate_tagger(tagger, _dataset(), _function_rows())
    assert report["topic"]["n"] > 0 and "temperature" in report["topic"]
    assert report["function"]["n"] > 0 and "temperature" in report["function"]
    assert tagger.temperatures == {
        "topic": report["topic"]["temperature"],
        "function": report["function"]["temperature"],
    }
    assert tagger.info["calibration"] == report
    # the tree a Classifier can score: untrained leaves gone, the headless family a leaf
    tree = trained_tree(TAX, tagger.topic.manifest["heads"])
    assert "report.rare" not in tree and "tiny.alpha" not in tree
    assert tree.children("tiny") == [] and len(tree.children("report")) == 3


# --- time budget ------------------------------------------------------------------------------


def test_time_budget_subsamples(monkeypatch):
    seen = []

    def fake_finetune(body, triplets, **kwargs):
        seen.append(len(triplets))

    monkeypatch.setattr(trainer_module, "_finetune_with_fallback", fake_finetune)
    roomy = train_tagger(
        _dataset(), [], stage="B", base_model=BASE, taxonomy=TAX, batch_size=4, epochs=1
    )
    available = roomy.info["budget"]["triplets_available"]
    assert seen == [available] and roomy.info["budget"]["subsampled"] is False
    # 10 s a step, a 20 s budget: two steps of four triplets
    tight = train_tagger(
        _dataset(),
        [],
        stage="B",
        base_model=BASE,
        taxonomy=TAX,
        batch_size=4,
        epochs=1,
        seconds_per_step=10.0,
        time_budget_hours=20 / 3600,
    )
    assert seen[-1] == 8 < available
    assert tight.info["budget"]["subsampled"] is True
    assert tight.info["budget"]["triplets_cap"] == 8
    assert tight.info["stage"] == "B"
    assert tight.topic.manifest["train_metrics"]["triplets"] == 8


def test_stage_a_does_not_finetune(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("stage A must not fine-tune")

    monkeypatch.setattr(trainer_module, "_finetune_with_fallback", fail)
    tagger = _tagger()
    assert tagger.info["stage"] == "A" and "subsampled" not in tagger.info["budget"]
    with pytest.raises(ValueError, match="stage"):
        train_tagger(_dataset(), [], stage="C", base_model=BASE, taxonomy=TAX)


# --- active learning --------------------------------------------------------------------------


def _uncertain(predicted, margin, i):
    return Uncertain(_element("x", eid=f"e{i}"), "report", predicted, margin, [])


def test_select_uncertain_smallest_margins_stratified():
    scored = [
        _uncertain("a", 0.9, 0),
        _uncertain("a", 0.1, 1),
        _uncertain("a", 0.2, 2),
        _uncertain("b", 0.5, 3),
        _uncertain("c", 0.3, 4),
        _uncertain("c", 0.05, 5),
    ]
    picked = select_uncertain(scored, 4)
    assert [u.element_id for u in picked] == ["e1", "e3", "e5", "e2"]
    assert len(select_uncertain(scored, 100)) == 6
    assert select_uncertain(scored, 0) == []


class FakeTeacher:
    name = "mlx:fake/teacher@abc"

    def __init__(self):
        self.calls = []

    def choose(self, items, *, instruction, prefix_key, prefix_options=None):
        self.calls.append((items, instruction, prefix_key, prefix_options))
        return [{tag: 1.0 / len(item.options) for tag, _ in item.options} for item in items]


def test_active_learning_round_labels_unlabelled_uncertain_elements(tmp_path):
    tagger = _tagger()
    corpus = Corpus(tmp_path / "corpus")
    items = [_element(f"{w} thing", eid=f"e{i}") for i, w in enumerate(["risk", "money", "misc"])]
    items.append(_element("   ", eid="e3"))  # no text: never scored
    items.append(_element("Page 3", kind="page_footer", eid="e4"))  # furniture
    corpus.add_document(ManifestRecord(id="d1", sha256="1" * 64, family_hint="report"), items, [1])
    tiny = [_element("alpha thing", doc="d2", eid="e0")]
    corpus.add_document(ManifestRecord(id="d2", sha256="2" * 64, family_hint="tiny"), tiny, [1])
    labeled = {
        LabelKey("d1", "e0", items[0].text_sha1, "topic"): LabelRow(
            doc_id="d1",
            element_id="e0",
            text_sha1=items[0].text_sha1,
            facet="topic",
            probs={"report.risk": 1.0},
        )
    }
    teacher = FakeTeacher()
    rows = active_learning_round(tagger, corpus, teacher, n=10, labeled=labeled)
    assert sorted((r.doc_id, r.element_id) for r in rows) == [
        ("d1", "e1"),
        ("d1", "e2"),
        ("d2", "e0"),
    ]
    assert all(r.facet == "topic" and r.provenance["source"] == "active_learning" for r in rows)
    by_id = {(r.doc_id, r.element_id): r for r in rows}
    # the student's shortlist, plus the untrained leaf it cannot propose, in family order
    assert by_id["d1", "e1"].provenance["shortlist"] == [
        "report.risk",
        "report.money",
        "report.rare",
        "report.other",
    ]
    # a family without a head offers every leaf
    assert by_id["d2", "e0"].provenance["shortlist"] == ["tiny.alpha", "tiny.beta"]
    assert by_id["d2", "e0"].provenance["student_margin"] == 0.0
    prefix_keys = sorted(call[2] for call in teacher.calls)
    assert prefix_keys == ["topic:report", "topic:tiny"]
    for _, _, key, options in teacher.calls:
        family = key.split(":")[1]
        assert [tag for tag, _ in options] == [c.id for c in TAX.children(family)]

    teacher.name = "jev:jev-1"
    with pytest.raises(TeacherNotAllowedError):
        active_learning_round(tagger, corpus, teacher, n=1, labeled=labeled)


def test_doc_family_is_the_teachers_rule():
    from docrooter.tagger.train import doc_family
    from docrooter.teach.shortlist import doc_family as teacher_doc_family

    for hint, doc_type in [("contract", None), (None, "financial.fact_sheet"), (None, None)]:
        record = ManifestRecord(id="x", sha256="3" * 64, family_hint=hint, doc_type_hint=doc_type)
        assert doc_family(record) == teacher_doc_family(record)
    record = ManifestRecord(id="x", sha256="3" * 64, family_hint="tiny")
    assert doc_family(record, ["report", "tiny"]) == "tiny"
