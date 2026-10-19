"""``docrooter select`` and ``docrooter fit-selector`` (v0.6 S5): the JSON and readout output, the
need parsing, and the selector file. Fake taggers, embedders and PDF readers; no model is ever
loaded and no PDF is read."""

import hashlib
import json

import numpy as np
import pytest
from typer.testing import CliRunner

from docrooter import Document, RouterConfig
from docrooter.backends import registry
from docrooter.cli import _readout, app
from docrooter.corpus import Corpus, CorpusElement, LabelRow, ManifestRecord
from docrooter.corpus.store import calib_half, text_sha1
from docrooter.elements import Element, TagResult
from docrooter.ocr import registry as ocr_registry
from docrooter.rooter.analysis import Analysis
from docrooter.select import fit as fit_module
from docrooter.select.crc import DialTable
from docrooter.select.relevance import FEATURE_NAMES, Combiner
from docrooter.select.selection import Selection, Selector
from docrooter.tagger import element_tags

runner = CliRunner()

CONFIG = 'ocr = "none"\ndoc_type = "keyword"\nsections = "keyword"\n'
MARKDOWN = """# Annual Report

The company grew this year. Revenue rose by a third and costs fell.

## Risk Factors

Interest rate risk could harm our results.

- Supplier risk may delay deliveries.
- Our offices are in three cities.

| Year | Revenue |
| ---- | ------- |
| 2025 | 133 |
"""
RISK_LEAF = "annual_report.risk_factors"
OTHER_LEAF = "annual_report.other"


@pytest.fixture(autouse=True)
def _no_real_ocr(monkeypatch):
    monkeypatch.setattr(ocr_registry, "resolve_auto_ocr", lambda: None)


class FakeTagger:
    source = "bundle:fake"

    def __init__(self):
        self.closed = 0

    def tag(self, elements, family):
        out = []
        for element in elements:
            risky = "risk" in element.text.lower()
            leaf = f"{family}.risk_factors" if risky else f"{family}.other"
            out.append(
                {"topic": TagResult(labels=[leaf], probs={leaf: 0.9}, confidence=0.9, source="f")}
            )
        return out

    def close(self):
        self.closed += 1


class FakeEmbedder:
    name = "embedding:fake"

    def __init__(self):
        self.closed = 0

    def encode(self, texts, prompt=""):
        return np.array([[1.0, 0.0] if "risk" in t.lower() else [0.0, 1.0] for t in texts])

    def close(self):
        self.closed += 1


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    """A bundle directory whose tagger is fake, and a fake ``embedding:fake`` backend."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "tagger.json").write_text("{}", encoding="utf-8")
    made = {"taggers": [], "embedders": []}

    def load(spec):
        made["taggers"].append(FakeTagger())
        return made["taggers"][-1]

    real = registry.resolve_backend

    def resolve(spec, **kwargs):
        if spec == "embedding:fake":
            made["embedders"].append(FakeEmbedder())
            return made["embedders"][-1]
        return real(spec, **kwargs)

    monkeypatch.setattr(element_tags, "load_bundle", load)
    monkeypatch.setattr(registry, "resolve_backend", resolve)
    return bundle, made


# --- select ------------------------------------------------------------------------------------


def _combiner(feature_names, weights):
    names = list(feature_names)
    return Combiner.from_dict(
        {
            "format": "docrooter-relevance-combiner",
            "version": 1,
            "feature_names": names,
            "l2": 1.0,
            "max_iter": 100,
            "mean": [0.0] * len(names),
            "scale": [1.0] * len(names),
            "coef": [weights.get(name, 0.0) for name in names],
            "intercept": -5.0,
            "n_train": 10,
            "n_docs": 2,
            "converged": True,
        }
    )


def _selector_file(tmp_path):
    field = _combiner(FEATURE_NAMES, {"p_topic": 10.0})
    fallback = _combiner(("sim", "section_hit"), {"sim": 10.0})
    dial = DialTable(
        family="annual_report",
        n=100,
        n_docs=100,
        n_excluded=0,
        grid=(0.5, 0.9),
        curve_lambdas=[0.0, 0.5, 1.0],
        curve_risk=[0.0, 0.0, 1.0],
        combiner_fingerprint=field.fingerprint(),
    )
    selector = Selector(
        field=field,
        field_fallback=fallback,
        query=_combiner(FEATURE_NAMES, {"sim": 10.0}),
        query_fallback=fallback,
        dials={"annual_report": dial},
        embedder="embedding:fake",
    )
    return selector.save(tmp_path / "selector.json")


@pytest.fixture
def files(tmp_path):
    doc = tmp_path / "report.md"
    doc.write_text(MARKDOWN, encoding="utf-8")
    config = tmp_path / "rooter.toml"
    config.write_text(CONFIG, encoding="utf-8")
    return doc, config


def _select(files, tmp_path, bundle, *args):
    doc, config = files
    return runner.invoke(
        app,
        [
            "select",
            str(doc),
            "--config",
            str(config),
            "--selector",
            str(_selector_file(tmp_path)),
            "--tagger",
            str(bundle),
            *args,
        ],
    )


def test_cli_select_json(files, fakes, tmp_path):
    bundle, made = fakes
    result = _select(files, tmp_path, bundle, "--need", "annual_report.risk_factors", "--json")
    assert result.exit_code == 0, result.output
    selection = Selection.model_validate_json(result.stdout)
    assert selection.need == "annual_report.risk_factors"  # a field id is that field
    assert selection.completeness == 0.9
    assert selection.element_ids  # the risk paragraphs, with their expansion
    assert selection.relevant_expected is not None
    assert len(made["embedders"]) == 1 and made["embedders"][0].closed == 1
    assert all(t.closed == 1 for t in made["taggers"])

    # Two field ids are one need; anything else is a query, never guaranteed.
    both = _select(
        files,
        tmp_path,
        bundle,
        "--need",
        "annual_report.risk_factors",
        "--need",
        "annual_report.legal_proceedings",
        "--json",
    )
    assert both.exit_code == 0, both.output
    assert json.loads(both.stdout)["need"] == (
        "annual_report.risk_factors,annual_report.legal_proceedings"
    )
    query = _select(files, tmp_path, bundle, "--need", "interest rate risk", "--json")
    assert query.exit_code == 0, query.output
    data = json.loads(query.stdout)
    assert data["need"] == "interest rate risk" and data["guarantee"] is False

    # A budget, and the whole document at 1.0.
    budget = _select(
        files, tmp_path, bundle, "--need", "annual_report.risk_factors", "--budget", "40", "--json"
    )
    assert budget.exit_code == 0, budget.output
    assert json.loads(budget.stdout)["guarantee"] is False
    whole = _select(
        files,
        tmp_path,
        bundle,
        "--need",
        "annual_report.risk_factors",
        "--completeness",
        "1.0",
        "--json",
    )
    assert json.loads(whole.stdout)["guarantee"] is True


def test_cli_select_readout_and_elements(files, fakes, tmp_path):
    bundle, _ = fakes
    result = _select(files, tmp_path, bundle, "--need", "annual_report.risk_factors")
    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert "of relevant content kept" in lines[0] and "tokens (" in lines[0]
    assert any("Interest rate risk could harm our results." in line for line in lines[1:])
    query = _select(files, tmp_path, bundle, "--need", "interest rate risk")
    assert "(estimate)" in query.stdout.splitlines()[0]


def test_cli_select_errors(files, fakes, tmp_path):
    bundle, _ = fakes
    mixed = _select(
        files, tmp_path, bundle, "--need", "annual_report.risk_factors", "--need", "a query"
    )
    assert mixed.exit_code == 2 and "--need" in mixed.output
    families = _select(
        files,
        tmp_path,
        bundle,
        "--need",
        "annual_report.risk_factors",
        "--need",
        "fund_document.fees_and_expenses",
    )
    assert families.exit_code == 1 and "famil" in families.output
    doc, config = files
    missing = runner.invoke(
        app,
        ["select", str(doc), "--config", str(config), "--need", "annual_report.risk_factors"],
    )
    assert missing.exit_code == 1 and "fit-selector" in missing.output


def _selection(**overrides):
    values = {
        "need": "annual_report.risk_factors",
        "completeness": 0.9,
        "threshold": 0.5,
        "element_ids": ["e1"],
        "tokens": 3240,
        "token_share": 0.12,
        "expected_recall": 0.9,
        "guarantee": True,
        "chunks": [],
    }
    return Selection(**{**values, **overrides})


def test_readout_wording():
    assert _readout(_selection()) == (
        "≈ 90% of relevant content kept · 3,240 tokens (12%) · "
        "guaranteed on calibration-like documents"
    )
    assert _readout(_selection(guarantee=False, expected_recall=0.87)) == (
        "≈ 87% of relevant content kept (estimate) · 3,240 tokens (12%)"
    )
    capped = _readout(_selection(capped=True, expected_recall=1.0, threshold=None))
    assert capped.endswith("dial capped by calibration size")
    mismatch = _readout(_selection(guarantee=False, doc_type_mismatch=True))
    assert mismatch.endswith("doc type mismatch")
    nothing = _readout(_selection(guarantee=False, expected_recall=None, element_ids=[]))
    assert nothing.startswith("no relevant content expected")


# --- fit-selector ------------------------------------------------------------------------------


def _ids(half, n, prefix):
    out, i = [], 0
    while len(out) < n:
        if calib_half(f"{prefix}{i}") == half:
            out.append(f"{prefix}{i}")
        i += 1
    return out


ITEMS = [
    ("h0", "heading", "Risk Factors"),
    ("p0", "paragraph", "Interest rate risk could harm us."),
    ("p1", "paragraph", "Our offices are in three cities."),
    ("p2", "paragraph", "Supplier risk may delay deliveries."),
    ("p3", "paragraph", "We employ many people."),
]


def _store(directory):
    corpus = Corpus(directory)
    for doc_id in _ids("fit", 2, "fit-") + _ids("crc", 2, "crc-"):
        corpus.add_document(
            ManifestRecord(
                id=doc_id,
                sha256=hashlib.sha256(doc_id.encode()).hexdigest(),
                split="calib",
                family_hint="annual_report",
            ),
            [
                CorpusElement(
                    doc_id=doc_id,
                    element_id=eid,
                    kind=kind,
                    text=text,
                    heading_path=["Risk Factors"],
                    section_title="Risk Factors",
                    page=1,
                )
                for eid, kind, text in ITEMS
            ],
            [1],
        )
        corpus.write_labels(
            "teacher-full",
            [
                LabelRow(
                    doc_id=doc_id,
                    element_id=eid,
                    text_sha1=text_sha1(text),
                    facet="topic",
                    probs={RISK_LEAF if "risk" in text else OTHER_LEAF: 0.9},
                )
                for eid, kind, text in ITEMS
                if kind != "heading"
            ],
        )
    return corpus


class FakeReader:
    """Stands in for the PDF reader: rebuilds a stored document as an analysis."""

    made: list["FakeReader"] = []

    def __init__(self, corpus, pdf_dir, *, profile, max_pages, sections_backend=None):
        self.args = (pdf_dir, profile, max_pages)
        self.closed = 0
        FakeReader.made.append(self)

    def __call__(self, record):
        doc = Document.from_sections([{"title": "Risk Factors", "level": 1}])
        doc.elements = [
            Element(
                id=eid,
                kind=kind,
                text=text,
                boxes=[],
                order=i,
                section_id="s1",
                **({"level": 1} if kind == "heading" else {}),
            )
            for i, (eid, kind, text) in enumerate(ITEMS)
        ]
        return Analysis(document=doc, config=RouterConfig(profile="fast"))

    def close(self):
        self.closed += 1


def test_cli_fit_selector_writes_file(tmp_path, fakes, monkeypatch):
    bundle, made = fakes
    store = tmp_path / "store"
    _store(store)
    (store / "pdfs").mkdir()
    FakeReader.made = []
    monkeypatch.setattr(fit_module, "CorpusPdfReader", FakeReader)
    out = tmp_path / "selector.json"
    result = runner.invoke(
        app,
        [
            "fit-selector",
            str(store),
            "--bundle",
            str(bundle),
            "--out",
            str(out),
            "--embedder",
            "embedding:fake",
        ],
    )
    assert result.exit_code == 0, result.output
    selector = Selector.load(out)
    assert selector.embedder == "embedding:fake"
    assert selector.field.n_docs == 2 and selector.dials["annual_report"].n_docs == 2
    assert selector.meta["profile"] == "accurate"
    assert selector.meta["tagger"]["spec"] == str(bundle)
    summary = json.loads(result.stdout)
    assert summary["out"] == str(out)
    assert summary["documents"]["fit"] == {"annual_report": 2}
    # The default reader: the store's pdfs folder, the build's profile and page cap; closed.
    assert [r.args for r in FakeReader.made] == [(store / "pdfs", "accurate", 30)]
    assert FakeReader.made[0].closed == 1
    assert made["taggers"][0].closed == 1 and made["embedders"][0].closed == 1
    assert "crc-" in result.stderr or "fit-" in result.stderr  # progress on stderr

    no_store = runner.invoke(
        app, ["fit-selector", str(tmp_path / "nothing"), "--bundle", str(bundle), "--out", str(out)]
    )
    assert no_store.exit_code == 1 and "no corpus" in no_store.output
