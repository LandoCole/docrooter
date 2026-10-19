"""The completeness dial: conformal risk control (CRC) from a target completeness c to a
relevance threshold λ (spec §0.3, §6; Angelopoulos et al., arXiv 2208.02814).

**Unit and loss.** A calibration unit is a (document, field) pair with at least one relevant
element (:class:`CalibUnit`). Units are pooled within each element family, over every field,
single-leaf and cross-cutting alike, and n is the number of units in the family. Pairs from one
document share that document, so they are correlated and not exchangeable in the strict sense;
``n_docs`` is stored beside n so this can be disclosed. At threshold λ an element is selected
when its relevance probability p ≥ λ, and a unit's loss is its missed share of relevant elements,

    L(λ) = 1 − |{relevant, p ≥ λ}| / |relevant|,

which can only rise with λ (a lower λ selects more), lies in [0, 1] (B = 1), and is
left-continuous in λ: it is constant on each interval (s_k, s_{k+1}] between consecutive
relevant probabilities.

**Rule.** For α = 1 − c,

    λ̂(c) = max { λ : (n/(n+1)) · mean L(λ) + 1/(n+1) ≤ α },

which gives E[recall] ≥ c on a new exchangeable unit. If no λ qualifies (1/(n+1) > α, or the
empirical risk is too high everywhere), the dial is **capped**: the whole document is returned,
for which recall is trivially 1. c = 1.0 bypasses CRC and returns everything.

**The λ grid.** The ruling's grid is every predicted probability on the calibration units plus
0 and 1. Because each L is constant on (s_k, s_{k+1}], the largest qualifying λ over that grid
(or over all of [0, 1]) is always a relevant element's probability, 0 or 1: the other grid
points never change the maximum. :func:`calibrate_dial` therefore evaluates the mean risk only at
the relevant probabilities plus 0 and 1, by a weighted cumulative sum, which keeps memory linear
in the number of relevant elements. ``test_dial_matches_crc_threshold_on_the_full_probability_grid``
checks the equivalence against :func:`crc_threshold` on the full grid.

**Grid of targets.** :func:`calibrate_dial` precomputes λ̂ on a grid of c (``grid_lambdas``,
for readouts and presets) and keeps the empirical risk curve, so :meth:`DialTable.threshold`
evaluates the same rule exactly at any c in between. That interpolation is monotone (λ̂ can only
fall as c rises) and keeps the guarantee, which a straight line between grid points would not.

**Data separation (spec §0.1).** Only calib-crc documents
(:func:`~docrooter.corpus.calib_half` ``== "crc"``) may calibrate the dial;
:func:`calibrate_dial` refuses any other unit.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from docrooter.corpus.store import calib_half

__all__ = ["DEFAULT_GRID", "CalibUnit", "DialTable", "calibrate_dial", "crc_threshold"]

DEFAULT_GRID: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99)
"""The completeness targets λ̂ is precomputed for."""

DIAL_FORMAT = "docrooter-dial-table"
DIAL_VERSION = 1
# Absorbs floating-point error at an exact boundary: 1 - 0.9 is 0.09999999999999998, while
# 1/(n+1) for n = 9 is 0.1. Far below any risk difference that one element can make.
_TOL = 1e-12


def _max_qualifying(lambdas: np.ndarray, risk: np.ndarray, n: int, alpha: float) -> float | None:
    """The spec §6 rule over candidate thresholds with mean losses ``risk``: the largest λ with
    (n/(n+1))·risk + 1/(n+1) ≤ α, or ``None`` when none qualifies."""
    bound = (n / (n + 1)) * risk + 1.0 / (n + 1)
    ok = bound <= alpha + _TOL
    if not ok.any():
        return None
    return float(lambdas[ok].max())


def _check_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return alpha


def crc_threshold(
    losses_by_lambda: Mapping[float, np.ndarray], alpha: float
) -> tuple[float | None, int]:
    """Spec §6 exactly: ``(λ̂, n)`` with λ̂ the largest λ whose per-unit losses satisfy
    (n/(n+1))·mean + 1/(n+1) ≤ α, or ``None`` when no λ does (the target is unreachable).

    ``losses_by_lambda[λ]`` holds the n units' losses at λ, each in [0, 1]. The maximum is taken
    literally over the given λ, without assuming the losses are monotone.
    """
    alpha = _check_alpha(alpha)
    if not losses_by_lambda:
        raise ValueError("no candidate thresholds")
    lambdas = np.array([float(lam) for lam in losses_by_lambda], dtype=np.float64)
    means = np.empty(len(lambdas))
    n = -1
    for i, values in enumerate(losses_by_lambda.values()):
        losses = np.asarray(values, dtype=np.float64)
        if losses.ndim != 1 or losses.size == 0:
            raise ValueError(f"losses must be non-empty 1-D arrays, got shape {losses.shape}")
        if n == -1:
            n = losses.size
        elif losses.size != n:
            raise ValueError(f"every λ needs the same {n} units, got {losses.size}")
        if not ((losses >= 0.0) & (losses <= 1.0)).all():
            raise ValueError("losses must lie in [0, 1]")
        means[i] = losses.mean()
    return _max_qualifying(lambdas, means, n, alpha), n


@dataclass(frozen=True, eq=False)
class CalibUnit:
    """One (document, field) pair of calib-crc: the combiner's relevance probability for each
    content element of the document, and which elements are relevant to the field (the teacher
    topic is one of its topics)."""

    doc_id: str
    family: str
    field_id: str
    probs: np.ndarray
    relevant: np.ndarray

    def __post_init__(self) -> None:
        probs = np.asarray(self.probs, dtype=np.float64)
        relevant = np.asarray(self.relevant, dtype=bool)
        if probs.ndim != 1 or relevant.shape != probs.shape:
            raise ValueError(
                f"{self.doc_id}/{self.field_id}: probs and relevant must be 1-D and aligned, "
                f"got {probs.shape} and {relevant.shape}"
            )
        if not ((probs >= 0.0) & (probs <= 1.0)).all():
            raise ValueError(f"{self.doc_id}/{self.field_id}: probabilities must lie in [0, 1]")
        object.__setattr__(self, "probs", probs)
        object.__setattr__(self, "relevant", relevant)


class DialTable:
    """One family's dial: λ̂ precomputed on a ``grid`` of targets, plus the empirical risk
    curve over n units that :meth:`threshold` evaluates at any target."""

    def __init__(
        self,
        *,
        family: str,
        n: int,
        n_docs: int,
        n_excluded: int,
        grid: Sequence[float],
        curve_lambdas: Sequence[float] | np.ndarray,
        curve_risk: Sequence[float] | np.ndarray,
    ) -> None:
        self.family = family
        self.n = int(n)
        self.n_docs = int(n_docs)
        self.n_excluded = int(n_excluded)  # (document, field) pairs with no relevant element
        self.curve_lambdas = np.asarray(curve_lambdas, dtype=np.float64)
        self.curve_risk = np.asarray(curve_risk, dtype=np.float64)
        if self.n < 1:
            raise ValueError(f"{family}: a dial needs at least one unit")
        if self.curve_lambdas.ndim != 1 or self.curve_lambdas.shape != self.curve_risk.shape:
            raise ValueError(f"{family}: the risk curve's thresholds and risks do not align")
        self.grid = tuple(_check_grid(grid))
        self.grid_lambdas: tuple[float | None, ...] = tuple(self.threshold(c)[0] for c in self.grid)

    def threshold(self, c: float) -> tuple[float | None, bool]:
        """``(λ̂, capped)`` for target completeness ``c`` in [0, 1]: select the elements with
        p ≥ λ̂. ``(None, True)`` when c is unreachable with this n (the dial is capped: return
        the whole document); ``(0.0, False)`` for c = 1.0, which bypasses CRC."""
        c = float(c)
        if not 0.0 <= c <= 1.0:  # also refuses NaN
            raise ValueError(f"completeness must be in [0, 1], got {c}")
        if c == 1.0:
            return 0.0, False
        lam = _max_qualifying(self.curve_lambdas, self.curve_risk, self.n, 1.0 - c)
        return lam, lam is None

    def select(self, probs: np.ndarray, c: float) -> tuple[np.ndarray, bool]:
        """``(mask, capped)``: which elements to keep at target ``c``, given their relevance
        probabilities. A capped dial, and c = 1.0, keep every element."""
        probs = np.asarray(probs, dtype=np.float64)
        lam, capped = self.threshold(c)
        if lam is None:
            return np.ones(probs.shape, dtype=bool), capped
        return probs >= lam, capped

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": DIAL_FORMAT,
            "version": DIAL_VERSION,
            "family": self.family,
            "n": self.n,
            "n_docs": self.n_docs,
            "n_excluded": self.n_excluded,
            "grid": list(self.grid),
            "grid_lambdas": list(self.grid_lambdas),  # for readers; recomputed on load
            "curve": {"lambdas": self.curve_lambdas.tolist(), "risk": self.curve_risk.tolist()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DialTable:
        if data.get("format") != DIAL_FORMAT or data.get("version") != DIAL_VERSION:
            raise ValueError(
                f"not a {DIAL_FORMAT} v{DIAL_VERSION}: "
                f"{data.get('format')!r} v{data.get('version')!r}"
            )
        return cls(
            family=data["family"],
            n=data["n"],
            n_docs=data["n_docs"],
            n_excluded=data["n_excluded"],
            grid=data["grid"],
            curve_lambdas=data["curve"]["lambdas"],
            curve_risk=data["curve"]["risk"],
        )


def _check_grid(grid: Iterable[float]) -> list[float]:
    values = sorted({float(c) for c in grid})
    if not values or not all(0.0 <= c < 1.0 for c in values):
        raise ValueError(f"grid targets must lie in [0, 1) (1.0 bypasses CRC), got {values}")
    return values


def _risk_curve(units: Sequence[CalibUnit]) -> tuple[np.ndarray, np.ndarray]:
    """The mean loss over ``units`` at each relevant probability and at 0 and 1, ascending.

    mean_i L_i(λ) = Σ over relevant elements j with p_j < λ of 1/(n·m_i(j)), where m_i is unit
    i's count of relevant elements: a weighted cumulative sum over the sorted probabilities.
    """
    n = len(units)
    scores = np.concatenate([u.probs[u.relevant] for u in units])
    weights = np.concatenate(
        [np.full(int(u.relevant.sum()), 1.0 / (n * int(u.relevant.sum()))) for u in units]
    )
    order = np.argsort(scores, kind="stable")
    scores, cumulative = scores[order], np.concatenate([[0.0], np.cumsum(weights[order])])
    lambdas = np.unique(np.concatenate([scores, [0.0, 1.0]]))
    below = np.searchsorted(scores, lambdas, side="left")  # relevant elements with p < λ
    risk = np.clip(cumulative[below], 0.0, 1.0)
    return lambdas, risk


def calibrate_dial(
    units: Iterable[CalibUnit], grid: Sequence[float] = DEFAULT_GRID
) -> dict[str, DialTable]:
    """One :class:`DialTable` per family, from calib-crc (document, field) units.

    Every unit must come from a calib-crc document (spec §0.1); a calib-fit one raises
    ``ValueError``. Units without a relevant element are not CRC units (recall is undefined
    there): they are left out and counted in ``n_excluded``. A family left with no unit gets no
    table.
    """
    by_family: dict[str, list[CalibUnit]] = {}
    excluded: dict[str, int] = {}
    outside: set[str] = set()
    for unit in units:
        if calib_half(unit.doc_id) != "crc":
            outside.add(unit.doc_id)
            continue
        if unit.relevant.any():
            by_family.setdefault(unit.family, []).append(unit)
        else:
            excluded[unit.family] = excluded.get(unit.family, 0) + 1
    if outside:
        raise ValueError(
            f"the dial calibrates on calib-crc documents only (spec §0.1); {len(outside)} are "
            f"calib-fit, e.g. {sorted(outside)[:3]}"
        )
    grid = _check_grid(grid)
    tables: dict[str, DialTable] = {}
    for family in sorted(by_family):
        family_units = by_family[family]
        lambdas, risk = _risk_curve(family_units)
        tables[family] = DialTable(
            family=family,
            n=len(family_units),
            n_docs=len({u.doc_id for u in family_units}),
            n_excluded=excluded.get(family, 0),
            grid=grid,
            curve_lambdas=lambdas,
            curve_risk=risk,
        )
    return tables
