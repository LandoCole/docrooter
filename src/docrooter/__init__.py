"""docrooter: document TOC extraction and taxonomy-based section classification."""

from docrooter.chunking import Chunk, chunk
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
from docrooter.elements import Box, Element, ElementKind, FigureData, TableData, TagResult
from docrooter.embed import EmbeddingSet, embed
from docrooter.errors import (
    BackendError,
    DatasetError,
    DocrooterError,
    MissingExtraError,
    TaxonomyError,
    UnknownTaxonomyError,
)
from docrooter.evaluate import compare, evaluate
from docrooter.flow import Classification, SectionTag, classify
from docrooter.ladder import LEVELS
from docrooter.layout.registry import register_layout_engine
from docrooter.ocr.registry import register_ocr_engine
from docrooter.pdf.load import load_pdf
from docrooter.rooter import Analysis, RouterConfig, process, process_many
from docrooter.runtime import configure
from docrooter.select import Field, Selection, Selector, builtin_fields, load_fields, select
from docrooter.summarize import Summary, summarize
from docrooter.taxonomies import builtin_taxonomy, list_builtin_taxonomies
from docrooter.taxonomy import Taxonomy, TaxonomyNode
from docrooter.train import Dataset
from docrooter.train.bundle import TrainedModel
from docrooter.train.calibrate import calibrate
from docrooter.train.trainer import train
from docrooter.views import SectionView

__version__ = "0.2.0"

__all__ = [
    "Analysis",
    "BackendError",
    "Beam",
    "Box",
    "Chunk",
    "Classification",
    "Classifier",
    "Dataset",
    "DatasetError",
    "DocTypeResult",
    "DocrooterError",
    "Document",
    "DocumentView",
    "Element",
    "ElementKind",
    "EmbeddingSet",
    "Field",
    "FigureData",
    "Flat",
    "Greedy",
    "LEVELS",
    "MissingExtraError",
    "NormalizationResult",
    "RouterConfig",
    "Section",
    "SectionTag",
    "SectionView",
    "Selection",
    "Selector",
    "Summary",
    "Taxonomy",
    "TaxonomyError",
    "TableData",
    "TagResult",
    "TaxonomyNode",
    "TrainedModel",
    "UnknownTaxonomyError",
    "__version__",
    "builtin_fields",
    "builtin_taxonomy",
    "calibrate",
    "chunk",
    "classify",
    "classify_document",
    "compare",
    "configure",
    "embed",
    "evaluate",
    "list_builtin_taxonomies",
    "load",
    "load_fields",
    "load_pdf",
    "normalize_sections",
    "process",
    "process_many",
    "register_layout_engine",
    "register_ocr_engine",
    "select",
    "summarize",
    "train",
]
