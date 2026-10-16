"""docrooter: document TOC extraction and taxonomy-based section classification."""

from docrooter.classify import Beam, Classifier, DocTypeResult, Flat, Greedy, classify_document
from docrooter.doc_view import DocumentView
from docrooter.document import Document, Section
from docrooter.document.load import load
from docrooter.errors import (
    BackendError,
    DatasetError,
    DocrooterError,
    MissingExtraError,
    TaxonomyError,
)
from docrooter.evaluate import compare, evaluate
from docrooter.taxonomies import builtin_taxonomy
from docrooter.taxonomy import Taxonomy, TaxonomyNode
from docrooter.train import Dataset
from docrooter.train.bundle import TrainedModel
from docrooter.train.calibrate import calibrate
from docrooter.train.trainer import train
from docrooter.views import SectionView

__version__ = "0.1.0.dev0"

__all__ = [
    "BackendError",
    "Beam",
    "Classifier",
    "Dataset",
    "DatasetError",
    "DocTypeResult",
    "DocrooterError",
    "Document",
    "DocumentView",
    "Flat",
    "Greedy",
    "MissingExtraError",
    "Section",
    "SectionView",
    "Taxonomy",
    "TaxonomyError",
    "TaxonomyNode",
    "TrainedModel",
    "__version__",
    "builtin_taxonomy",
    "calibrate",
    "classify_document",
    "compare",
    "evaluate",
    "load",
    "train",
]
