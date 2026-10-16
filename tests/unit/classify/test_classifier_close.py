"""``Classifier.close`` and the context manager: models loaded from specs are released."""

import pytest
from conftest import fin_legal_taxonomy

from docrooter.backends import KeywordBackend, registry
from docrooter.classify.classifier import Classifier


class ClosableKeyword(KeywordBackend):
    closed: list[str] = []

    def __init__(self, model="", **kwargs):
        super().__init__(name=f"closable:{model}")

    def close(self) -> None:
        ClosableKeyword.closed.append(self.name)


class WithZeroShot(ClosableKeyword):
    """Stands in for a trained model: supports nothing itself, offers a zero-shot fallback."""

    def __init__(self, model="", **kwargs):
        super().__init__(model)
        self.name = f"trained-like:{model}"
        self.zero_shot = ClosableKeyword("zero-shot")

    def supports(self, node_id):
        return False

    def as_zero_shot(self):
        return self.zero_shot


@pytest.fixture(autouse=True)
def kinds(monkeypatch):
    monkeypatch.setattr(registry, "_KINDS", dict(registry._KINDS))
    registry.register_backend_kind("closable", ClosableKeyword)
    registry.register_backend_kind("trainedlike", WithZeroShot)
    ClosableKeyword.closed = []


def test_close_releases_spec_backends_and_keeps_instances_open():
    mine = ClosableKeyword("mine")
    clf = Classifier(
        fin_legal_taxonomy(),
        "closable:primary",
        fallback=mine,
        overrides={"depth>=1": "closable:deep"},
    )
    clf.classify_texts(["tax revenue"])
    clf.close()
    assert sorted(ClosableKeyword.closed) == ["closable:deep", "closable:primary"]
    assert clf._resolved == {}
    clf.close()  # nothing left to close
    assert len(ClosableKeyword.closed) == 2
    clf.classify_texts(["tax revenue"])  # usable again: the spec loads anew
    assert set(clf._resolved) == {"closable:primary", "closable:deep"}


def test_context_manager_closes_the_zero_shot_fallback_of_a_spec():
    with Classifier(fin_legal_taxonomy(), "trainedlike:m") as clf:
        clf.classify_texts(["tax revenue"])
        assert clf._fallback() is not None
    assert sorted(ClosableKeyword.closed) == ["closable:zero-shot", "trained-like:m"]


def test_backend_without_close_is_skipped():
    with Classifier(fin_legal_taxonomy(), "keyword") as clf:
        clf.classify_texts(["tax revenue"])
    assert clf._resolved == {}


DOC = (
    "# Income tax\n\nIncome tax and deferred tax on revenue earnings.\n\n"
    "# Audit\n\nAudit findings.\n"
)


def test_classify_document_closes_a_spec_backend_but_not_an_instance():
    from docrooter import classify_document, load

    doc = load(DOC)
    classify_document(doc, fin_legal_taxonomy(), backend="closable:doc")
    assert ClosableKeyword.closed == ["closable:doc"]
    mine = ClosableKeyword("mine")
    classify_document(doc, fin_legal_taxonomy(), backend=mine)
    assert ClosableKeyword.closed == ["closable:doc"]


@pytest.mark.parametrize("doc_type", [None, "financial.annual_report"])
def test_normalize_sections_closes_what_it_loaded(doc_type):
    from docrooter import load, normalize_sections

    doc = load(DOC)
    normalize_sections(doc, doc_type=doc_type, backend="closable:norm")
    assert ClosableKeyword.closed == ["closable:norm"]  # loaded once, closed once
    mine = ClosableKeyword("mine")
    normalize_sections(doc, doc_type=doc_type, backend=mine)
    assert ClosableKeyword.closed == ["closable:norm"]
