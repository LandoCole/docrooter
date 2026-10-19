"""The rooter's ``element_tags`` task (spec §10): which elements get tags, the family they are
tagged in, one tagger load per ``process_many``, ownership, spec checks and the Jev refusal.
Fake taggers, teachers and backends only: no model is ever loaded."""

import importlib
import json

import numpy as np
import pytest
from pydantic import ValidationError

from docrooter.backends import Scores
from docrooter.errors import DocrooterError
from docrooter.ocr import registry as ocr_registry
from docrooter.rooter import PROFILES, RouterConfig, process, process_many
from docrooter.tagger import element_tags
from docrooter.tagger.element_tags import (
    TAGGED_KINDS,
    EmbeddingTagger,
    TeacherTagger,
    content_elements,
)
from docrooter.taxonomies import family_for_doc_type
from docrooter.teach.label import TEACHER_MAX_CHARS
from docrooter.teach.shortlist import family_leaves, function_tags

MARKDOWN = """# Annual Report

The company grew this year. Revenue rose by a third and costs fell.

## Risk Factors

Markets may fall. Rates may rise. Suppliers may fail.

- Interest rates may rise.
- Suppliers may fail to deliver.

| Year | Revenue |
| ---- | ------- |
| 2024 | 100 |
| 2025 | 133 |
"""

# keyword backends and nothing else that could load a model
LIGHT = {"doc_type": "keyword", "sections": "none", "summaries": "none", "escalate": "none"}


@pytest.fixture(autouse=True)
def _no_real_ocr(monkeypatch):
    monkeypatch.setattr(ocr_registry, "resolve_auto_ocr", lambda: None)


class FakeTagger:
    """Tags every element with its family's ``other`` leaf; records calls and closes."""

    def __init__(self):
        self.calls = []
        self.closed = 0

    def tag(self, elements, family):
        from docrooter.elements import TagResult

        self.calls.append(([e.kind for e in elements], family, list(elements)))
        leaf = f"{family}.other"
        return [
            {
                "topic": TagResult(labels=[leaf], probs={leaf: 0.9}, confidence=0.9, source="fake"),
                "function": TagResult(
                    labels=["definition"],
                    probs={"definition": 0.8, "citation": 0.1},
                    confidence=0.8,
                    source="fake",
                ),
            }
            for _ in elements
        ]

    def close(self):
        self.closed += 1


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """A bundle directory (just its manifest) whose loader returns fresh fake taggers."""
    path = tmp_path / "bundle"
    path.mkdir()
    (path / "tagger.json").write_text("{}", encoding="utf-8")
    made = []

    def load(spec):
        assert spec == str(path)
        made.append(FakeTagger())
        return made[-1]

    monkeypatch.setattr(element_tags, "load_bundle", load)
    return path, made


def test_element_tags_task_tags_content_kinds_only():
    tagger = FakeTagger()
    analysis = process(MARKDOWN, element_tags=tagger, **LIGHT)
    doc = analysis.document
    kinds = {el.kind for el in doc.elements}
    assert {"heading", "paragraph", "list_item", "table"} <= kinds

    for el in doc.elements:
        if el.kind in TAGGED_KINDS:
            assert set(el.tags) == {"topic", "function"}
            assert el.tags["function"].labels == ["definition"]
        else:
            assert el.tags is None
    # one call for the document, with the content elements in reading order
    assert len(tagger.calls) == 1
    called_kinds, family, elements = tagger.calls[0]
    assert called_kinds == [el.kind for el in doc.elements if el.kind in TAGGED_KINDS]
    leaf = analysis.doc_type.best.node_ids[-1]
    assert family == family_for_doc_type(leaf)
    table = next(e for e in elements if e.kind == "table")
    assert table.table_markdown and "| 2025 | 133 |" in table.table_markdown
    assert elements[0].heading_path == ["Annual Report"]

    assert tagger.closed == 0  # the caller's tagger is never closed
    assert analysis.config.element_tags == "FakeTagger"
    tagger.source = "bundle:element-tagger-v1"  # a tagger names itself by its source
    assert process(MARKDOWN, element_tags=tagger, **LIGHT).config.element_tags == tagger.source
    assert "element_tags" in analysis.timings
    routes = [r for r in doc.meta["routes"] if r["task"] == "element_tags"]
    assert routes and routes[0]["reason"] == "passed by the caller"

    data = json.loads(analysis.to_json())
    tagged = [e for e in data["document"]["elements"] if "tags" in e]
    assert {e["kind"] for e in tagged} <= set(TAGGED_KINDS)
    assert tagged[0]["tags"]["topic"]["labels"] == [f"{family}.other"]
    assert "family_probs" not in tagged[0]["tags"]["topic"]


def test_element_tags_without_doc_type_uses_the_generic_family():
    tagger = FakeTagger()
    process(MARKDOWN, element_tags=tagger, **{**LIGHT, "doc_type": "none"})
    assert tagger.calls[0][1] == "generic"


def test_no_element_tags_by_default_and_the_json_is_unchanged():
    analysis = process(MARKDOWN, **LIGHT)
    assert analysis.config.element_tags == "none"
    assert all(el.tags is None for el in analysis.document.elements)
    data = json.loads(analysis.to_json())
    assert all("tags" not in e for e in data["document"]["elements"])
    assert "element_tags" not in analysis.timings
    assert not [r for r in analysis.document.meta["routes"] if r["task"] == "element_tags"]


def test_process_many_loads_tagger_once(bundle):
    path, made = bundle
    analyses = list(process_many([MARKDOWN, MARKDOWN, MARKDOWN], element_tags=str(path), **LIGHT))
    assert len(made) == 1  # one load for every document
    assert len(made[0].calls) == 3
    assert made[0].closed == 1  # created from the spec: closed when the iterator ends
    for analysis in analyses:
        assert analysis.config.element_tags == str(path)
        assert any(el.tags for el in analysis.document.elements)

    # process() closes what it loaded, before returning
    process(MARKDOWN, element_tags=str(path), **LIGHT)
    assert len(made) == 2 and made[1].closed == 1


def test_process_many_closes_the_tagger_when_stopped_early(bundle):
    path, made = bundle
    iterator = process_many([MARKDOWN, MARKDOWN], element_tags=str(path), **LIGHT)
    next(iterator)
    iterator.close()
    assert len(made) == 1 and made[0].closed == 1


def test_element_tags_default_is_none_in_every_profile():
    for name in PROFILES:
        assert PROFILES[name]["element_tags"] == "none"
        assert RouterConfig(profile=name).element_tags == "none"


@pytest.mark.parametrize(
    "spec",
    [
        "teacher:jev:jev-1.13.0",
        "teacher:openai:jev-1.13.0",
        "teacher:hf:typesafe/jev-small@abc123",
        "teacher:mlx:org/Jev-2@abc123",
        "jev:jev-1.13.0",
        "JEV:jev-1.13.0",
    ],
)
def test_teacher_jev_refused_in_config(spec):
    with pytest.raises(ValidationError, match=r"TypeSafe MCA §2\.3\(b\)"):
        RouterConfig(element_tags=spec)
    with pytest.raises(ValidationError, match=r"TypeSafe MCA §2\.3\(b\)"):
        RouterConfig.resolve(profile="accurate", element_tags=spec)
    with pytest.raises(ValidationError, match=r"TypeSafe MCA §2\.3\(b\)"):
        process(MARKDOWN, element_tags=spec, **LIGHT)


def test_teacher_tagger_refuses_a_jev_engine_even_when_built_directly():
    from docrooter.errors import TeacherNotAllowedError

    with pytest.raises(TeacherNotAllowedError):
        TeacherTagger("jev:jev-1.13.0")
    with pytest.raises(TeacherNotAllowedError):
        TeacherTagger("openai:gpt-x", teacher=FakeTeacher(name="openai:jev-1.13.0"))


def test_a_jev_endpoint_is_refused_when_the_teacher_tagger_is_built(monkeypatch):
    from docrooter.errors import TeacherNotAllowedError

    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.typesafe.ai/v1")
    monkeypatch.setattr(ocr_registry, "resolve_auto_ocr", lambda: None)
    with pytest.raises(TeacherNotAllowedError, match=r"2\.3\(b\)"):
        process(MARKDOWN, element_tags="teacher:openai:gpt-x", **LIGHT)


@pytest.mark.parametrize(
    ("spec", "error", "match"),
    [
        ("teacher:", ValidationError, "names no model"),
        ("embedding:", ValidationError, "names no model"),
        ("teacher:llama:some-model", DocrooterError, "bad element_tags teacher spec"),
        ("teacher:mlx", DocrooterError, "bad element_tags teacher spec"),
        ("embedding-ish:x", DocrooterError, "no element-tagger bundle"),
        ("/no/such/bundle", DocrooterError, "no element-tagger bundle"),
    ],
)
def test_bad_element_tags_specs_fail_before_any_document_is_read(spec, error, match, monkeypatch):
    def no_load(*args, **kwargs):
        raise AssertionError("a document was read")

    monkeypatch.setattr(importlib.import_module("docrooter.rooter.process"), "load", no_load)
    with pytest.raises(error, match=match):
        process(MARKDOWN, element_tags=spec, **LIGHT)
    with pytest.raises(error, match=match):
        process_many([MARKDOWN], element_tags=spec, **LIGHT)


def test_element_tags_need_element_mode(bundle):
    path, _ = bundle
    with pytest.raises(ValueError, match="needs element mode"):
        process(MARKDOWN, profile="fast", element_tags=str(path))


def test_a_bad_element_tags_instance_is_a_type_error():
    with pytest.raises(TypeError, match="element_tags must be"):
        process(MARKDOWN, element_tags=object(), **LIGHT)


# --- the alternatives ---------------------------------------------------------------------


class FakeEmbedding:
    """``score`` gives each candidate a cosine from a fixed per-id table (else 0)."""

    name = "embedding:fake"
    temperature = 0.05

    def __init__(self, favour):
        self.favour = favour
        self.closed = 0

    def score(self, texts, candidates, *, instruction=None, multi_label=False):
        row = [self.favour.get(c.node_id, 0.0) for c in candidates]
        return Scores(np.tile(np.array(row, dtype=float), (len(texts), 1)), "similarity")

    def close(self):
        self.closed += 1


def _content(markdown=MARKDOWN):
    from docrooter import load

    return content_elements(load(markdown, elements=True))


def test_embedding_tagger_scores_the_family_and_the_functions():
    elements = _content()
    backend = FakeEmbedding({"annual_report.risk_factors": 0.6, "risk_disclosure": 0.5})
    tagger = EmbeddingTagger("embedding:fake", backend=backend, batch_size=2)
    results = tagger.tag(elements, "annual_report")
    assert len(results) == len(elements)
    topic, function = results[0]["topic"], results[0]["function"]
    assert topic.labels == ["annual_report.risk_factors"]
    assert set(topic.probs) == {leaf.id for leaf in family_leaves("annual_report")}
    assert sum(topic.probs.values()) == pytest.approx(1.0, abs=1e-4)
    assert topic.source == "embedding:fake" and topic.family_probs is None
    assert set(function.probs) == set(function_tags())
    assert function.labels == ["risk_disclosure"]  # the only tag over 0.5 at this temperature
    tagger.close()
    assert backend.closed == 1


class FakeTeacher:
    def __init__(self, name="mlx:org/model@rev#native"):
        self.name = name
        self.choose_calls = []
        self.questions = []
        self.closed = 0

    def choose(self, items, *, instruction, prefix_key, prefix_options=None):
        self.choose_calls.append((items, instruction, prefix_key, prefix_options))
        return [{tag: (0.7 if i == 0 else 0.3 / (len(item.options) - 1))
                 for i, (tag, _) in enumerate(item.options)} for item in items]  # fmt: skip

    def yes_no(self, items, *, question):
        self.questions.append(question)
        return [0.9 if "risk disclosure" in question.lower() else 0.1 for _ in items]

    def close(self):
        self.closed += 1


def test_teacher_tagger_asks_every_leaf_and_every_function():
    long = "word " * 10000
    elements = _content(MARKDOWN + f"\n## Notes\n\n{long}\n")
    teacher = FakeTeacher()
    tagger = TeacherTagger("mlx:org/model@rev", teacher=teacher)
    results = tagger.tag(elements, "annual_report")
    items, instruction, prefix_key, options = teacher.choose_calls[0]
    leaves = [leaf.id for leaf in family_leaves("annual_report")]
    assert [tag for tag, _ in options] == leaves and prefix_key == "topic:annual_report"
    assert all([tag for tag, _ in item.options] == leaves for item in items)
    assert "annual report" in instruction
    assert max(len(item.text) for item in items) == TEACHER_MAX_CHARS  # cut, never crashing
    table = items[[e.kind for e in elements].index("table")]
    assert "| 2025 | 133 |" in table.text  # tables as Markdown
    assert len(teacher.questions) == len(function_tags())
    assert results[0]["topic"].labels == [leaves[0]]
    assert results[0]["topic"].source == "teacher:mlx:org/model@rev#native"
    assert results[0]["function"].labels == ["risk_disclosure"]
    assert results[0]["function"].confidence == pytest.approx(0.9)
    tagger.close()
    assert teacher.closed == 1
