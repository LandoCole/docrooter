"""The relevance model: spec §5 features and the logistic combiner fitted on calib-fit."""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from docrooter.corpus.store import calib_half
from docrooter.elements.models import TagResult
from docrooter.select import Field
from docrooter.select.relevance import CONTENT_KINDS, FEATURE_NAMES, Combiner, features
from docrooter.tagger.element_tags import TAGGED_KINDS


def _ids(half: str, k: int) -> list[str]:
    out: list[str] = []
    i = 0
    while len(out) < k:
        doc_id = f"doc-{i}"
        i += 1
        if calib_half(doc_id) == half:
            out.append(doc_id)
    return out


def _tag(probs: dict[str, float]) -> TagResult:
    best = max(probs, key=lambda k: probs[k])
    return TagResult(labels=[best], probs=probs, confidence=probs[best], source="test")


RISK = Field(
    id="annual_report.risk_factors",
    label="Risk factors",
    description="The risks the company faces.",
    topics=["annual_report.risk_factors"],
    functions=["risk_disclosure"],
)
FEES = Field(
    id="annual_report.fees",
    label="Fees",
    description="Fees.",
    topics=["annual_report.executive_compensation", "annual_report.financial_statements"],
    functions=["financial_data"],
)


def test_content_kinds_are_the_tagged_kinds():
    assert CONTENT_KINDS == TAGGED_KINDS
    assert FEATURE_NAMES == (
        "p_topic",
        "p_func",
        "sim",
        "section_hit",
        *(f"kind_{k}" for k in TAGGED_KINDS),
        "log_tokens",
    )


def test_features_follow_spec_section_5():
    elements = [
        SimpleNamespace(kind="paragraph", text="Our business faces many risks today"),
        SimpleNamespace(kind="table", text="a b c d e f g h i j"),
        SimpleNamespace(kind="list_item", text=""),
    ]
    tags = [
        {
            "topic": _tag({"annual_report.risk_factors": 0.7, "annual_report.other": 0.3}),
            "function": _tag({"risk_disclosure": 0.9, "financial_data": 0.2}),
        },
        {
            "topic": _tag(
                {
                    "annual_report.executive_compensation": 0.5,
                    "annual_report.financial_statements": 0.4,
                    "annual_report.other": 0.1,
                }
            )
        },
        None,  # untagged: tag features are 0
    ]
    sims = [0.8, 0.1, -0.2]
    sections = ["risk_factors", "annual_report.financial_statements", None]

    X = features(elements, tags, sims, sections, RISK)
    assert X.shape == (3, len(FEATURE_NAMES)) and X.dtype == np.float64
    col = {name: X[:, i] for i, name in enumerate(FEATURE_NAMES)}
    assert col["p_topic"].tolist() == pytest.approx([0.7, 0.0, 0.0])
    assert col["p_func"].tolist() == pytest.approx([0.9, 0.0, 0.0])
    assert col["sim"].tolist() == pytest.approx(sims)
    assert col["section_hit"].tolist() == [1.0, 0.0, 0.0]  # label = the topic's leaf name
    assert col["kind_paragraph"].tolist() == [1.0, 0.0, 0.0]
    assert col["kind_table"].tolist() == [0.0, 1.0, 0.0]
    assert col["kind_list_item"].tolist() == [0.0, 0.0, 1.0]
    assert (
        X[:, [FEATURE_NAMES.index(f"kind_{k}") for k in CONTENT_KINDS]].sum(axis=1).tolist()
        == [1.0] * 3
    )
    # log(1 + estimate_tokens): 6 words -> 8 tokens, 10 words -> 13 tokens, empty -> 0.
    assert col["log_tokens"].tolist() == pytest.approx([math.log(9), math.log(14), 0.0])

    # A multi-topic field: the probability mass on all its topics; full topic ids match too.
    X2 = features(elements, tags, sims, sections, FEES)
    assert X2[1, 0] == pytest.approx(0.9)
    assert X2[1, 3] == 1.0
    # A list of fields is the union of their topics and functions.
    X3 = features(elements, tags, sims, sections, [RISK, FEES])
    assert X3[0, 0] == pytest.approx(0.7) and X3[1, 0] == pytest.approx(0.9)
    assert X3[0, 1] == pytest.approx(0.9)
    # A free-text query has no topics: only similarity, kind and length carry it.
    X4 = features(elements, tags, sims, sections, "What are the main risks?")
    assert X4[:, :2].tolist() == [[0.0, 0.0]] * 3 and X4[:, 3].tolist() == [0.0] * 3
    assert X4[:, 2].tolist() == pytest.approx(sims)


def test_features_reject_misaligned_inputs():
    elements = [SimpleNamespace(kind="paragraph", text="x")]
    with pytest.raises(ValueError):
        features(elements, [None, None], [0.1], [None], RISK)
    with pytest.raises(ValueError):
        features(elements, [None], [0.1, 0.2], [None], RISK)
    with pytest.raises(ValueError):
        features(elements, [None], [0.1], [], RISK)


def _synthetic(rng: np.random.Generator, m: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.zeros((m, len(FEATURE_NAMES)))
    X[:, 0] = rng.random(m)  # p_topic: the signal
    X[:, 1] = rng.random(m)
    X[:, 2] = rng.uniform(-1, 1, m)  # sim: a weaker signal
    X[:, 3] = rng.random(m) < 0.3
    X[np.arange(m), 4 + rng.integers(0, 6, m)] = 1.0
    X[:, 10] = np.log1p(rng.integers(1, 400, m))
    logit = 6.0 * (X[:, 0] - 0.5) + 1.5 * X[:, 2]
    y = (rng.random(m) < 1.0 / (1.0 + np.exp(-logit))).astype(float)
    return X, y


def test_combiner_fit_predict_and_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    X, y = _synthetic(rng, 3000)
    doc_ids = [_ids("fit", 30)[i % 30] for i in range(len(y))]
    model = Combiner(l2=1.0).fit(X, y, doc_ids=doc_ids)

    # The fitted weights solve the L2-regularised likelihood: its gradient is ~0.
    Z = (X - model.mean) / model.scale
    p = model.predict_proba(X)
    grad_w = Z.T @ (p - y) + model.l2 * model.coef
    assert np.abs(grad_w).max() < 1e-6
    assert abs(float(np.sum(p - y))) < 1e-6  # the unpenalised intercept
    # It learned the signal: p_topic dominates, sim helps, and it ranks well.
    w = dict(zip(FEATURE_NAMES, model.coef / model.scale, strict=True))
    assert w["p_topic"] > 4.0 and w["sim"] > 0.8
    order = np.argsort(-p)
    assert y[order[:300]].mean() > 0.85 and y[order[-300:]].mean() < 0.15
    assert ((p > 0.5) == (y > 0.5)).mean() > 0.75
    assert p.min() > 0.0 and p.max() < 1.0

    path = tmp_path / "selector.json"
    model.save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["feature_names"] == list(FEATURE_NAMES)
    assert data["n_train"] == 3000 and data["n_docs"] == 30
    again = Combiner.load(path)
    assert np.array_equal(again.predict_proba(X), model.predict_proba(X))


def test_combiner_handles_constant_columns_and_validates():
    rng = np.random.default_rng(1)
    X, y = _synthetic(rng, 400)
    X[:, 1] = 0.0  # e.g. no function tags at all: a constant column
    ids = [_ids("fit", 4)[i % 4] for i in range(len(y))]
    model = Combiner().fit(X, y, doc_ids=ids)
    assert np.isfinite(model.predict_proba(X)).all()
    assert model.coef[1] == 0.0
    with pytest.raises(ValueError):
        Combiner().fit(X, np.ones_like(y), doc_ids=ids)  # one class only
    with pytest.raises(ValueError):
        Combiner().fit(X[:, :5], y, doc_ids=ids)  # wrong width
    with pytest.raises(ValueError):
        Combiner().fit(X, y, doc_ids=ids[:-1])  # misaligned ids
    with pytest.raises(ValueError):
        Combiner().predict_proba(X)  # not fitted


def test_combiner_fits_only_calib_fit():
    rng = np.random.default_rng(2)
    X, y = _synthetic(rng, 200)
    fit_ids, crc_ids = _ids("fit", 5), _ids("crc", 1)
    ids = [fit_ids[i % 5] for i in range(len(y))]
    Combiner().fit(X, y, doc_ids=ids)  # calib-fit only: fine
    ids[17] = crc_ids[0]  # one calib-crc row is enough to refuse
    with pytest.raises(ValueError, match="calib-fit"):
        Combiner().fit(X, y, doc_ids=ids)


def test_features_reject_non_content_kind():
    elements = [
        SimpleNamespace(kind="paragraph", text="x"),
        SimpleNamespace(kind="heading", text="Risk factors"),
    ]
    with pytest.raises(ValueError, match="heading"):
        features(elements, [None, None], [0.1, 0.2], [None, None], RISK)


def test_newton_rejects_a_step_that_raises_the_loss(monkeypatch):
    rng = np.random.default_rng(4)
    X, y = _synthetic(rng, 300)
    ids = [_ids("fit", 3)[i % 3] for i in range(len(y))]
    # A solver that returns an ascent direction: every step size raises the loss.
    monkeypatch.setattr(np.linalg, "solve", lambda H, g: -1e3 * g)
    with pytest.warns(RuntimeWarning, match="converge"):
        model = Combiner().fit(X, y, doc_ids=ids)
    assert not model.converged
    assert np.all(model.coef == 0.0) and model.intercept == 0.0  # the start, never worsened


def test_combiner_flags_non_convergence():
    rng = np.random.default_rng(5)
    X, y = _synthetic(rng, 500)
    ids = [_ids("fit", 5)[i % 5] for i in range(len(y))]
    with pytest.warns(RuntimeWarning, match="converge"):
        short = Combiner(max_iter=1).fit(X, y, doc_ids=ids)
    assert short.converged is False
    assert Combiner.from_dict(short.to_dict()).converged is False
    full = Combiner().fit(X, y, doc_ids=ids)
    assert full.converged is True
    assert Combiner.from_dict(full.to_dict()).converged is True


def test_combiner_fingerprint_is_stable():
    rng = np.random.default_rng(6)
    X, y = _synthetic(rng, 400)
    ids = [_ids("fit", 4)[i % 4] for i in range(len(y))]
    a = Combiner().fit(X, y, doc_ids=ids)
    b = Combiner().fit(X, y, doc_ids=ids)
    fp = a.fingerprint()
    assert len(fp) == 64 and fp == b.fingerprint()
    assert Combiner.from_dict(json.loads(json.dumps(a.to_dict()))).fingerprint() == fp
    assert Combiner(l2=3.0).fit(X, y, doc_ids=ids).fingerprint() != fp
    fallback = Combiner(feature_names=("sim", "section_hit")).fit(X, y, doc_ids=ids)
    assert fallback.fingerprint() != fp
    with pytest.raises(ValueError):
        Combiner().fingerprint()  # not fitted


def test_fallback_combiner_on_feature_subset():
    # Spec §5: without a tagger, relevance falls back to sim and section_hit only. The
    # combiner takes the full features() matrix and picks its own columns.
    rng = np.random.default_rng(8)
    X, y = _synthetic(rng, 2000)
    ids = [_ids("fit", 10)[i % 10] for i in range(len(y))]
    fallback = Combiner(feature_names=("sim", "section_hit")).fit(X, y, doc_ids=ids)
    assert fallback.coef.shape == (2,)
    columns = [FEATURE_NAMES.index("sim"), FEATURE_NAMES.index("section_hit")]
    assert np.array_equal(fallback.project(X), X[:, columns])
    p = fallback.predict_proba(X)
    assert np.array_equal(p, fallback.predict_proba(X[:, columns]))
    assert fallback.coef[0] > 0.0  # sim carries signal in the synthetic data
    order = np.argsort(-p)
    assert y[order[:300]].mean() > y.mean()
    again = Combiner.from_dict(json.loads(json.dumps(fallback.to_dict())))
    assert np.array_equal(again.predict_proba(X), p)
    with pytest.raises(ValueError):
        fallback.predict_proba(X[:, :5])  # neither its own width nor features()'s


@pytest.mark.parametrize(
    "names",
    [("sim", "page_rank"), (), ("sim", "sim"), ("section_hit", "sim")],
    ids=["unknown", "empty", "repeated", "out-of-order"],
)
def test_feature_names_must_be_known(names):
    with pytest.raises(ValueError):
        Combiner(feature_names=names)
    rng = np.random.default_rng(10)
    X, y = _synthetic(rng, 200)
    ids = [_ids("fit", 2)[i % 2] for i in range(len(y))]
    data = Combiner(feature_names=("sim", "section_hit")).fit(X, y, doc_ids=ids).to_dict()
    data["feature_names"] = list(names)
    with pytest.raises(ValueError):
        Combiner.from_dict(data)
