"""The completeness dial: conformal risk control over (document, field) units (spec §0.3, §6)."""

import json
import time

import numpy as np
import pytest

from docrooter.corpus.store import calib_half
from docrooter.select.crc import (
    DEFAULT_GRID,
    CalibUnit,
    DialTable,
    calibrate_dial,
    crc_threshold,
)


def _ids(half: str, k: int, prefix: str = "doc") -> list[str]:
    """``k`` document ids in the given calib half (the partition is a hash of the id)."""
    out: list[str] = []
    i = 0
    while len(out) < k:
        doc_id = f"{prefix}-{i}"
        i += 1
        if calib_half(doc_id) == half:
            out.append(doc_id)
    return out


def _unit(
    doc_id: str, probs, relevant, family: str = "annual_report", field: str = "f"
) -> CalibUnit:
    return CalibUnit(
        doc_id=doc_id,
        family=family,
        field_id=f"{family}.{field}",
        probs=np.asarray(probs, dtype=float),
        relevant=np.asarray(relevant, dtype=bool),
    )


def _simulate(rng: np.random.Generator, n_docs: int, k: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Exchangeable synthetic documents: ``k`` elements each, about a fifth relevant (at least
    one), scored by a noisy model whose sharpness varies by document."""
    relevant = rng.random((n_docs, k)) < 0.2
    relevant[~relevant.any(axis=1), 0] = True
    sharpness = rng.uniform(0.5, 2.5, size=(n_docs, 1))
    logits = sharpness * np.where(relevant, 1.0, -1.0) + rng.normal(0.0, 1.2, size=(n_docs, k))
    return 1.0 / (1.0 + np.exp(-logits)), relevant


def _full_grid_losses(units: list[CalibUnit]) -> dict[float, np.ndarray]:
    """Per-unit losses on the ruling's grid: every predicted probability on the units, plus 0
    and 1. The loss of a unit at λ is the share of its relevant elements with p < λ."""
    grid = sorted({0.0, 1.0} | {float(p) for u in units for p in u.probs})
    return {
        lam: np.array([float(np.mean(u.probs[u.relevant] < lam)) for u in units]) for lam in grid
    }


# --- crc_threshold: spec §6 exactly -----------------------------------------------------------


def test_crc_formula_on_synthetic_losses():
    # n = 4, so the bound is 0.8 * mean + 0.2.
    losses = {
        0.0: np.array([0.0, 0.0, 0.0, 0.0]),  # bound 0.2
        0.3: np.array([0.0, 0.0, 0.5, 0.0]),  # mean 0.125, bound 0.3
        0.6: np.array([0.5, 0.0, 1.0, 0.5]),  # mean 0.5,   bound 0.6
        0.9: np.array([1.0, 1.0, 1.0, 1.0]),  # mean 1.0,   bound 1.0
    }
    assert crc_threshold(losses, 0.4) == (0.3, 4)
    assert crc_threshold(losses, 0.6) == (0.6, 4)  # equality is allowed (≤)
    assert crc_threshold(losses, 0.2) == (0.0, 4)  # 1/(n+1) == α exactly: reachable
    assert crc_threshold(losses, 0.19) == (None, 4)  # below 1/(n+1): unreachable
    assert crc_threshold(losses, 1.0) == (0.9, 4)
    # A boundary that only holds in exact arithmetic still holds: n = 9, c = 0.9 (α = 1 - 0.9
    # is 0.09999999999999998 in floating point, and 1/(n+1) = 0.1).
    assert crc_threshold({0.0: np.zeros(9), 0.5: np.full(9, 0.5)}, 1 - 0.9) == (0.0, 9)
    # "max λ" is literal: it does not assume the losses are monotone in λ.
    assert crc_threshold({0.1: np.ones(4), 0.7: np.zeros(4)}, 0.5) == (0.7, 4)


@pytest.mark.parametrize(
    "losses, alpha",
    [
        ({}, 0.1),
        ({0.5: np.array([])}, 0.1),
        ({0.5: np.zeros(3), 0.6: np.zeros(4)}, 0.1),
        ({0.5: np.array([0.0, 1.5])}, 0.1),
        ({0.5: np.array([[0.0, 0.5]])}, 0.1),
        ({0.5: np.zeros(3)}, -0.1),
        ({0.5: np.zeros(3)}, 1.1),
    ],
)
def test_crc_threshold_rejects_bad_input(losses, alpha):
    with pytest.raises(ValueError):
        crc_threshold(losses, alpha)


# --- calibrate_dial and DialTable ---------------------------------------------------------------


def test_dial_matches_crc_threshold_on_the_full_probability_grid():
    rng = np.random.default_rng(7)
    probs, relevant = _simulate(rng, 60, k=12)
    units = [_unit(d, p, r) for d, p, r in zip(_ids("crc", 60), probs, relevant, strict=True)]
    dial = calibrate_dial(units)["annual_report"]
    full = _full_grid_losses(units)
    assert dial.n == 60
    for c in [*DEFAULT_GRID, 0.0, 0.3, 0.55, 0.875, 0.93, 0.97, 0.985]:
        expected, n = crc_threshold(full, 1.0 - c)
        assert n == 60
        assert dial.threshold(c) == (expected, expected is None), c
    for c, lam in zip(dial.grid, dial.grid_lambdas, strict=True):
        assert lam == crc_threshold(full, 1.0 - c)[0]


def test_crc_monotone_in_target():
    rng = np.random.default_rng(11)
    probs, relevant = _simulate(rng, 150)
    dial = calibrate_dial(
        [_unit(d, p, r) for d, p, r in zip(_ids("crc", 150), probs, relevant, strict=True)]
    )["annual_report"]
    cs = np.linspace(0.0, 1.0, 401)
    results = [dial.threshold(float(c)) for c in cs]
    lams = [0.0 if lam is None else lam for lam, _ in results]  # capped = the whole document
    assert all(a >= b for a, b in zip(lams, lams[1:], strict=False))
    capped = [cap for _, cap in results[:-1]]  # below 1.0, which bypasses CRC: never capped
    assert capped == sorted(capped)  # once capped, every higher target is capped
    assert results[-1] == (0.0, False)
    assert any(capped) and not all(capped)  # n = 150 reaches 0.99 but not 0.995
    stored = [lam for lam in dial.grid_lambdas if lam is not None]
    assert stored == sorted(stored, reverse=True)
    # Between two grid targets the threshold lies between theirs.
    for lo, hi in zip(dial.grid, dial.grid[1:], strict=False):
        mid = (lo + hi) / 2
        lam_lo, lam_hi, lam_mid = (
            dial.threshold(lo)[0],
            dial.threshold(hi)[0],
            dial.threshold(mid)[0],
        )
        assert lam_mid is not None and lam_lo is not None and lam_hi is not None
        assert lam_hi <= lam_mid <= lam_lo


def test_crc_guarantee_on_simulation():
    started = time.perf_counter()
    rng = np.random.default_rng(20261019)
    probs, relevant = _simulate(rng, 2000)
    units = [
        _unit(d, p, r) for d, p, r in zip(_ids("crc", 2000, "sim"), probs, relevant, strict=True)
    ]
    dial = calibrate_dial(units)["annual_report"]
    assert dial.n == 2000
    new_probs, new_relevant = _simulate(rng, 20000)  # held out, exchangeable with calibration
    realised = {}
    for c in (0.8, 0.9, 0.95, 0.98):
        lam, capped = dial.threshold(c)
        assert not capped and lam is not None and lam > 0.0  # a real cut, not the whole document
        kept = (new_probs >= lam) & new_relevant
        recall = kept.sum(axis=1) / new_relevant.sum(axis=1)
        realised[c] = float(recall.mean())
        assert realised[c] >= c - 0.01, (c, realised[c])
    assert time.perf_counter() - started < 5.0


def test_unreachable_target_caps_to_whole_document():
    # Review Focus 3. n = 10 units: 1/(n+1) ≈ 0.091, so any c > 1 - 1/11 ≈ 0.909 is unreachable
    # even with zero empirical loss.
    rng = np.random.default_rng(3)
    probs, relevant = _simulate(rng, 10, k=20)
    dial = calibrate_dial(
        [_unit(d, p, r) for d, p, r in zip(_ids("crc", 10), probs, relevant, strict=True)]
    )["annual_report"]
    assert dial.n == 10
    for c in (0.95, 0.98, 0.99):
        assert dial.threshold(c) == (None, True)
        mask, capped = dial.select(probs[0], c)
        assert capped and mask.all()  # the whole document
    assert dict(zip(dial.grid, dial.grid_lambdas, strict=True))[0.95] is None
    lam, capped = dial.threshold(0.5)
    assert not capped and lam is not None
    mask, capped = dial.select(probs[0], 0.5)
    assert not capped and np.array_equal(mask, probs[0] >= lam)
    # c = 1.0 bypasses CRC: everything, never capped (the guarantee is trivial).
    assert dial.threshold(1.0) == (0.0, False)
    mask, capped = dial.select(probs[0], 1.0)
    assert mask.all() and not capped


def test_dial_uses_only_calib_crc():
    crc, fit = _ids("crc", 3), _ids("fit", 1)
    units = [_unit(d, [0.9, 0.1], [True, False]) for d in crc]
    calibrate_dial(units)  # calib-crc only: fine
    with pytest.raises(ValueError, match="calib-crc"):
        calibrate_dial([*units, _unit(fit[0], [0.9, 0.1], [True, False])])


def test_dial_pools_fields_per_family_and_stores_n():
    crc = _ids("crc", 6)
    units = [
        # Two fields of one annual report document: both count, pooled in the family.
        _unit(crc[0], [0.9, 0.2, 0.4], [True, False, True], field="risk_factors"),
        _unit(crc[0], [0.8, 0.1, 0.3], [False, True, False], field="fees"),
        _unit(crc[1], [0.7, 0.6], [True, True]),
        # No relevant element: not a unit (recall is undefined there).
        _unit(crc[2], [0.7, 0.6], [False, False]),
        _unit(crc[3], [0.9], [True], family="contract"),
        _unit(crc[4], [0.5, 0.4], [False, True], family="contract"),
    ]
    dials = calibrate_dial(units)
    assert set(dials) == {"annual_report", "contract"}
    assert (dials["annual_report"].n, dials["annual_report"].n_docs) == (3, 2)
    assert dials["annual_report"].n_excluded == 1
    assert (dials["contract"].n, dials["contract"].n_docs) == (2, 2)


def test_dial_table_json_roundtrip(tmp_path):
    rng = np.random.default_rng(5)
    probs, relevant = _simulate(rng, 40)
    dial = calibrate_dial(
        [_unit(d, p, r) for d, p, r in zip(_ids("crc", 40), probs, relevant, strict=True)]
    )["annual_report"]
    path = tmp_path / "dial.json"
    path.write_text(json.dumps(dial.to_dict()), encoding="utf-8")
    again = DialTable.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert (again.family, again.n, again.n_docs, again.grid) == (
        dial.family,
        dial.n,
        dial.n_docs,
        dial.grid,
    )
    assert again.grid_lambdas == dial.grid_lambdas
    for c in np.linspace(0.0, 1.0, 51):
        assert again.threshold(float(c)) == dial.threshold(float(c))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"probs": np.array([0.5, 1.2]), "relevant": np.array([True, False])},
        {"probs": np.array([0.5, 0.2]), "relevant": np.array([True])},
        {"probs": np.array([[0.5]]), "relevant": np.array([[True]])},
    ],
)
def test_calib_unit_validates(kwargs):
    with pytest.raises(ValueError):
        CalibUnit(doc_id="d", family="annual_report", field_id="annual_report.f", **kwargs)


def test_threshold_rejects_targets_outside_unit_interval():
    dial = calibrate_dial([_unit(d, [0.9], [True]) for d in _ids("crc", 3)])["annual_report"]
    for c in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            dial.threshold(c)
