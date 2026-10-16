"""Evaluation metrics over predicted and gold taxonomy paths (lists of node ids)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from docrooter.classify.results import SectionResult

Path = Sequence[str]


def _check_lengths(pred: Sequence[object], gold: Sequence[object]) -> None:
    if len(pred) != len(gold):
        raise ValueError(f"got {len(pred)} predictions for {len(gold)} gold paths")


def exact_path_accuracy(pred: Sequence[Path], gold: Sequence[Path]) -> float:
    """Share of examples whose predicted path equals the gold path; 0.0 for no examples."""
    _check_lengths(pred, gold)
    if not gold:
        return 0.0
    return sum(list(p) == list(g) for p, g in zip(pred, gold, strict=True)) / len(gold)


def depth_accuracy(pred: Sequence[Path], gold: Sequence[Path]) -> dict[int, float]:
    """Accuracy of the node at each 1-based depth ``d`` up to the deepest gold path.

    Depth ``d`` is scored over the examples whose gold path has at least ``d`` nodes; a
    prediction is correct there when it has at least ``d`` nodes and ``pred[d-1] == gold[d-1]``,
    so a prediction that stopped short is wrong at the depths it never reached.
    """
    _check_lengths(pred, gold)
    out: dict[int, float] = {}
    for d in range(1, max((len(g) for g in gold), default=0) + 1):
        pairs = [(p, g) for p, g in zip(pred, gold, strict=True) if len(g) >= d]
        out[d] = sum(len(p) >= d and p[d - 1] == g[d - 1] for p, g in pairs) / len(pairs)
    return out


def hierarchical_prf(pred: Sequence[Path], gold: Sequence[Path]) -> tuple[float, float, float]:
    """Micro-averaged hierarchical precision, recall and F1.

    Each path is the set of its node ids (the ancestors are on the path already), so the right
    branch with the wrong leaf earns partial credit. A zero denominator gives 0.0.
    """
    _check_lengths(pred, gold)
    hit = n_pred = n_gold = 0
    for p, g in zip(pred, gold, strict=True):
        ps, gs = set(p), set(g)
        hit += len(ps & gs)
        n_pred += len(ps)
        n_gold += len(gs)
    precision = hit / n_pred if n_pred else 0.0
    recall = hit / n_gold if n_gold else 0.0
    return precision, recall, _f1(precision, recall)


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def is_stopped(result: SectionResult) -> bool:
    """True when the section has no best path or its best path stopped before a leaf."""
    return result.best is None or result.best.stopped_reason is not None


def stop_rate(results: Sequence[SectionResult]) -> float:
    """Share of sections that stopped early (see :func:`is_stopped`); 0.0 for no sections."""
    if not results:
        return 0.0
    return sum(is_stopped(r) for r in results) / len(results)


def selective_accuracy(
    pred: Sequence[Path], gold: Sequence[Path], stopped: Sequence[bool]
) -> float:
    """Exact-path accuracy over the examples that did not stop; 0.0 when all of them stopped."""
    _check_lengths(pred, gold)
    _check_lengths(stopped, gold)
    kept = [(p, g) for p, g, s in zip(pred, gold, stopped, strict=True) if not s]
    if not kept:
        return 0.0
    return sum(list(p) == list(g) for p, g in kept) / len(kept)


def leaf_prf(pred: Sequence[Path], gold: Sequence[Path]) -> tuple[float, float, float]:
    """Macro precision, recall and F1 of the last node as a flat multi-class label.

    Averaged over the gold leaf classes. An empty predicted path counts as no label: a miss
    for its gold class and a hit for none. A class with no true or predicted hits scores 0.
    """
    _check_lengths(pred, gold)
    classes = sorted({g[-1] for g in gold if g})
    if not classes:
        return 0.0, 0.0, 0.0
    guessed = [p[-1] if p else None for p in pred]
    truth = [g[-1] if g else None for g in gold]
    ps, rs, fs = [], [], []
    for c in classes:
        tp = sum(a == c and t == c for a, t in zip(guessed, truth, strict=True))
        n_guessed = sum(a == c for a in guessed)
        n_true = sum(t == c for t in truth)
        precision = tp / n_guessed if n_guessed else 0.0
        recall = tp / n_true if n_true else 0.0
        ps.append(precision)
        rs.append(recall)
        fs.append(_f1(precision, recall))
    n = len(classes)
    return sum(ps) / n, sum(rs) / n, sum(fs) / n


def sibling_confusions(
    decisions: dict[str, list[tuple[str, str]]], top: int = 3
) -> dict[str, list[tuple[str, str, int]]]:
    """Per node, the ``top`` most frequent ``(gold_child, predicted_child, count)`` mistakes.

    ``decisions`` maps a node id to ``(gold_child, predicted_child)`` pairs. Correct decisions
    are ignored and nodes without a mistake are left out. Ties go to the pair sorted first.
    """
    out: dict[str, list[tuple[str, str, int]]] = {}
    for node_id, pairs in decisions.items():
        counts = Counter((g, p) for g, p in pairs if g != p)
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top]
        if ranked:
            out[node_id] = [(g, p, n) for (g, p), n in ranked]
    return out
