"""Section classification: the classifier, traversal strategies, engine and result models."""

from docrooter.classify.classifier import Classifier
from docrooter.classify.documents import classify_document
from docrooter.classify.engine import EngineSettings, Router, decide, run_flat, run_tree
from docrooter.classify.results import (
    DocTypeResult,
    DocumentResult,
    PathResult,
    SectionResult,
    Step,
)
from docrooter.classify.strategies import Beam, Flat, Greedy

__all__ = [
    "Beam",
    "Classifier",
    "DocTypeResult",
    "DocumentResult",
    "EngineSettings",
    "Flat",
    "Greedy",
    "PathResult",
    "Router",
    "SectionResult",
    "Step",
    "classify_document",
    "decide",
    "run_flat",
    "run_tree",
]
