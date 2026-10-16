"""docrooter: document TOC extraction and taxonomy-based section classification."""

from docrooter.classify import Beam, Classifier, Flat, Greedy
from docrooter.document import Document, Section
from docrooter.document.load import load
from docrooter.taxonomy import Taxonomy, TaxonomyNode
from docrooter.train import Dataset
from docrooter.views import SectionView

__version__ = "0.1.0.dev0"

__all__ = [
    "Beam",
    "Classifier",
    "Dataset",
    "Document",
    "Flat",
    "Greedy",
    "Section",
    "SectionView",
    "Taxonomy",
    "TaxonomyNode",
    "__version__",
    "load",
]
