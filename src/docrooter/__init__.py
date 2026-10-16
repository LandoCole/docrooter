"""docrooter: document TOC extraction and taxonomy-based section classification."""

from docrooter.document import Document, Section
from docrooter.document.load import load
from docrooter.taxonomy import Taxonomy, TaxonomyNode

__version__ = "0.1.0.dev0"

__all__ = ["Document", "Section", "Taxonomy", "TaxonomyNode", "__version__", "load"]
