"""docrooter: document TOC extraction and taxonomy-based section classification."""

from docrooter.document import Document, Section
from docrooter.document.load import load

__version__ = "0.1.0.dev0"

__all__ = ["Document", "Section", "__version__", "load"]
