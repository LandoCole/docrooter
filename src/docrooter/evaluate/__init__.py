"""Evaluation: path metrics, ``evaluate``, ``compare`` and their reports."""

from docrooter.evaluate.metrics import (
    depth_accuracy,
    exact_path_accuracy,
    hierarchical_prf,
    leaf_prf,
    selective_accuracy,
    sibling_confusions,
    stop_rate,
)
from docrooter.evaluate.report import Comparison, Report, compare, evaluate

__all__ = [
    "Comparison",
    "Report",
    "compare",
    "depth_accuracy",
    "evaluate",
    "exact_path_accuracy",
    "hierarchical_prf",
    "leaf_prf",
    "selective_accuracy",
    "sibling_confusions",
    "stop_rate",
]
