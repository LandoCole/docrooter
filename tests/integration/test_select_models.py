"""``select()`` with a trained bundle on two corpus-style annual reports (spec §9, marker
``models``): the real tagger's tags, the embedder recipe's query prompt and a fitted
``selector.json`` working together.

It skips unless ``DOCROOTER_TEST_BUNDLE`` names a tagger bundle holding a ``selector.json``
(``docrooter fit-selector --out <bundle>/selector.json``). The two PDFs are invented reports in
``tests/fixtures/select/`` (``generate.py`` there). Run it alone, one model process at a time::

    pgrep -fl "benchmarks\\.|docrooter serve|corpus build|teach label"   # must print nothing
    DOCROOTER_TEST_BUNDLE=<bundle> PYTHONPATH=src .venv/bin/python -m pytest \\
        tests/integration/test_select_models.py -m models

Nothing heavy is imported at collection: a bare ``pytest`` run deselects and skips it cleanly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.models

BUNDLE_ENV = "DOCROOTER_TEST_BUNDLE"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "select"
PDFS = ("northwind_annual_report.pdf", "harbor_lane_annual_report.pdf")
RISK_ID = "annual_report.risk_factors"
OFF = {"summaries": "none", "escalate": "none", "embeddings": "none"}  # never Jev, even with a key


@pytest.fixture(scope="module")
def bundle() -> Path:
    value = os.environ.get(BUNDLE_ENV, "").strip()
    if not value:
        pytest.skip(f"set {BUNDLE_ENV} to a tagger bundle with a selector.json")
    path = Path(value)
    if not (path / "selector.json").is_file():
        pytest.skip(f"{BUNDLE_ENV}={value} holds no selector.json")
    return path


def _ids(analysis, kinds) -> set[str]:
    return {e.id for e in analysis.document.elements if e.kind in kinds}


@pytest.mark.parametrize("name", PDFS)
def test_select_with_the_trained_bundle(bundle: Path, name: str) -> None:
    from docrooter import builtin_fields, process, select
    from docrooter.select.relevance import CONTENT_KINDS
    from docrooter.select.selection import HEADING_KINDS, Selector

    selector = Selector.load(bundle)
    provenance = selector.provenance
    profile = provenance.profile if provenance is not None else "accurate"
    risk = {f.id: f for f in builtin_fields("annual_report")}[RISK_ID]
    # read as the selector was fitted: the fit's profile, tagged by the bundle
    analysis = process(FIXTURES / name, profile=profile, element_tags=str(bundle), **OFF)
    content = _ids(analysis, CONTENT_KINDS)
    selectable = content | _ids(analysis, HEADING_KINDS)
    assert content, "the reading found no content element"

    result = select(analysis, risk, completeness=0.9, selector=selector)
    # selected ids are content elements plus their expansion (headings), nothing else
    assert result.element_ids and set(result.element_ids) <= selectable
    # the guarantee flags behave as G1 says
    if provenance is not None and provenance.tagger is not None:
        assert result.fit_mismatch == [], result.fit_mismatch  # same reading, tagger, embedder
    expected = (
        not result.fit_mismatch
        and not result.doc_type_mismatch
        and not result.multi_field
        and selector.is_calibrated(risk)
        and risk.family in selector.dials
    )
    assert result.guarantee is expected
    if result.guarantee:
        assert result.expected_recall in (0.9, 1.0) and result.calibration_n
    # the same features select the same elements
    again = select(analysis, risk, completeness=0.9, selector=selector)
    assert again.element_ids == result.element_ids

    # c = 1.0 returns everything, guaranteed, with no model
    whole = select(analysis, risk, completeness=1.0)
    assert set(whole.element_ids) == selectable and whole.guarantee is True

    # a list of fields is an estimate, flagged
    legal = {f.id: f for f in builtin_fields("annual_report")}["annual_report.legal_proceedings"]
    pair = select(analysis, [risk, legal], completeness=0.9, selector=selector)
    assert pair.guarantee is False and pair.multi_field is True
    assert set(pair.element_ids) <= selectable


def test_another_reading_is_flagged(bundle: Path) -> None:
    """A profile with another layout engine than the fit's drops the guarantee, flagged."""
    from docrooter import builtin_fields, process, select
    from docrooter.rooter.profiles import profile_defaults
    from docrooter.select.selection import Selector

    selector = Selector.load(bundle)
    provenance = selector.provenance
    if provenance is None:
        pytest.skip("the bundle's selector records no provenance (an old selector.json)")
    other = next(
        (p for p in ("balanced", "accurate") if profile_defaults(p)["layout"] != provenance.layout),
        None,
    )
    if other is None:
        pytest.skip(f"no profile reads with another layout than {provenance.layout!r}")
    risk = {f.id: f for f in builtin_fields("annual_report")}[RISK_ID]
    analysis = process(FIXTURES / PDFS[0], profile=other, element_tags=str(bundle), **OFF)
    result = select(analysis, risk, completeness=0.9, selector=selector)
    assert "layout" in result.fit_mismatch and result.guarantee is False
