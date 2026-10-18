"""docrooter: document TOC extraction and taxonomy-based section classification."""

from docrooter.classify import (
    Beam,
    Classifier,
    DocTypeResult,
    Flat,
    Greedy,
    NormalizationResult,
    classify_document,
    normalize_sections,
)
from docrooter.doc_view import DocumentView
from docrooter.document import Document, Section
from docrooter.document.load import load
from docrooter.elements import Box, Element, ElementKind, FigureData, TableData
from docrooter.errors import (
    BackendError,
    DatasetError,
    DocrooterError,
    MissingExtraError,
    TaxonomyError,
    UnknownTaxonomyError,
)
from docrooter.evaluate import compare, evaluate
from docrooter.pdf.load import load_pdf
from docrooter.runtime import configure
from docrooter.taxonomies import builtin_taxonomy, list_builtin_taxonomies
from docrooter.taxonomy import Taxonomy, TaxonomyNode
from docrooter.train import Dataset
from docrooter.train.bundle import TrainedModel
from docrooter.train.calibrate import calibrate
from docrooter.train.trainer import train
from docrooter.views import SectionView

__version__ = "0.2.0"

__all__ = [
    "BackendError",
    "Beam",
    "Box",
    "Classifier",
    "Dataset",
    "DatasetError",
    "DocTypeResult",
    "DocrooterError",
    "Document",
    "DocumentView",
    "Element",
    "ElementKind",
    "FigureData",
    "Flat",
    "Greedy",
    "MissingExtraError",
    "NormalizationResult",
    "Section",
    "SectionView",
    "Taxonomy",
    "TaxonomyError",
    "TableData",
    "TaxonomyNode",
    "TrainedModel",
    "UnknownTaxonomyError",
    "__version__",
    "builtin_taxonomy",
    "calibrate",
    "classify_document",
    "compare",
    "configure",
    "evaluate",
    "list_builtin_taxonomies",
    "load",
    "load_pdf",
    "normalize_sections",
    "train",
]
