"""Section normalisation: map each TOC entry to a canonical label of a section-type taxonomy."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from docrooter.backends.base import Backend
from docrooter.classify.classifier import DEFAULT_BACKEND, Classifier
from docrooter.classify.documents import classify_document
from docrooter.classify.strategies import Flat
from docrooter.document import Document
from docrooter.taxonomies import DOC_TYPE_TO_SECTION_TAXONOMY, builtin_taxonomy
from docrooter.taxonomy import Taxonomy
from docrooter.views import SectionView

DOC_TYPES = "doc_types"
GENERIC_TAXONOMY = "section_types/generic"
CUSTOM_TAXONOMY = "<custom>"
# Headings only: the path of parent titles, then the title. With the heading-synonym examples
# of the built-in section taxonomies this beat title + lead by 10-15 points of accuracy for
# both local embedding backends on the silver benchmark (docs/benchmarks-pdf.md).
NORMALIZE_VIEW = SectionView(fields=("path", "title"), max_tokens=96)


class SectionNorm(BaseModel):
    """The canonical label of one section; ``canonical`` and ``label`` are None when nothing was
    assigned (a blank title). ``score`` is the assigned label's probability."""

    model_config = ConfigDict(extra="forbid")

    canonical: str | None
    label: str | None
    score: float = 0.0
    title: str


class NormalizationResult(BaseModel):
    """Canonical labels keyed by section id.

    ``doc_type`` is the document-type node id used to choose the section taxonomy (None when a
    taxonomy was given without one); ``doc_type_confidence`` is set when it came from
    ``classify_document``. ``taxonomy`` is the built-in name or ``"<custom>"``.
    """

    model_config = ConfigDict(extra="forbid")

    doc_type: str | None
    doc_type_confidence: float | None = None
    taxonomy: str
    sections: dict[str, SectionNorm] = Field(default_factory=dict)


def normalize_sections(
    doc: Document,
    *,
    doc_type: str | None = None,
    taxonomy: Taxonomy | str | None = None,
    backend: str | Backend = DEFAULT_BACKEND,
    levels: Sequence[int] = (1, 2),
    classifier: Classifier | None = None,
    view: SectionView | None = None,
) -> NormalizationResult:
    """Map each section of ``doc`` at ``levels`` to a canonical section of a section taxonomy.

    The taxonomy is ``taxonomy`` (a :class:`Taxonomy` or built-in name, e.g.
    ``"research_paper"`` or ``"section_types/research_paper"``) if given, else the one mapped
    from ``doc_type`` (a leaf id, or a group id, which uses its first leaf; unknown types use
    ``section_types/generic``). With neither, :func:`classify_document` finds the doc type
    first. Every selected section is always assigned a label, in one batched classification
    (``Flat``) of each section rendered with ``view`` (default :data:`NORMALIZE_VIEW`). The
    preamble (level 0) is never normalised.

    A prebuilt ``classifier`` must be built on the chosen taxonomy (else ``ValueError``); it
    classifies as configured and also supplies the backend for the doc-type step.
    """
    view = view or NORMALIZE_VIEW
    confidence: float | None = None
    if isinstance(taxonomy, Taxonomy):
        section_taxonomy, name = taxonomy, CUSTOM_TAXONOMY
    elif taxonomy is not None:
        name = _builtin_name(taxonomy)
        section_taxonomy = builtin_taxonomy(name)
    else:
        if doc_type is None:
            result = classify_document(
                doc,
                builtin_taxonomy(DOC_TYPES),
                backend=classifier.backend if classifier is not None else backend,
            )
            doc_type = result.best.node_ids[-1] if result.best.node_ids else None
            confidence = result.confidence
        name = _section_taxonomy_name(doc_type)
        section_taxonomy = builtin_taxonomy(name)

    if classifier is not None:
        if classifier.taxonomy.content_hash() != section_taxonomy.content_hash():
            raise ValueError(
                f"classifier taxonomy does not match the section taxonomy {name!r}; "
                "build the classifier on that taxonomy or omit it"
            )
    else:
        classifier = Classifier(
            section_taxonomy,
            backend,
            strategy=Flat(),
            view=view,
            instruction=f"Which standard section of a {_doc_type_label(doc_type)} is this?",
            threshold=0.0,
        )

    wanted = set(levels) - {0}
    sections = [s for s in doc.walk() if s.level in wanted]
    # A section with neither title nor text has nothing to classify: it stays unlabelled.
    todo = [s for s in sections if s.title.strip() or s.text.strip()]
    norms: dict[str, SectionNorm] = {}
    if todo:
        views = view.render_all(doc, [s.id for s in todo], classifier._token_counter())
        results = classifier.classify_texts([views[s.id].text for s in todo])
        for section, result in zip(todo, results, strict=True):
            best = result.best
            assigned = best is not None and bool(best.node_ids)
            norms[section.id] = SectionNorm(
                canonical=best.node_ids[-1] if assigned else None,
                label=best.labels[-1] if assigned else None,
                score=best.score if assigned else 0.0,
                title=section.title,
            )
    out = {
        s.id: norms.get(s.id) or SectionNorm(canonical=None, label=None, title=s.title)
        for s in sections
    }
    return NormalizationResult(
        doc_type=doc_type, doc_type_confidence=confidence, taxonomy=name, sections=out
    )


def _builtin_name(name: str) -> str:
    """A built-in section taxonomy name, accepting ``"research_paper"`` for
    ``"section_types/research_paper"``. Unknown names raise ``KeyError`` listing the valid ones."""
    if "/" not in name:
        try:
            builtin_taxonomy(f"section_types/{name}")
            return f"section_types/{name}"
        except KeyError:
            pass
    builtin_taxonomy(name)  # raises KeyError for an unknown name
    return name


def _section_taxonomy_name(doc_type: str | None) -> str:
    if doc_type is None:
        return GENERIC_TAXONOMY
    if doc_type in DOC_TYPE_TO_SECTION_TAXONOMY:
        return DOC_TYPE_TO_SECTION_TAXONOMY[doc_type]
    doc_types = builtin_taxonomy(DOC_TYPES)
    if doc_type not in doc_types:
        return GENERIC_TAXONOMY
    stack = [doc_type]  # first leaf under a group, depth first in taxonomy order
    while stack:
        node = doc_types.node(stack.pop())
        if not node.children:
            return DOC_TYPE_TO_SECTION_TAXONOMY.get(node.id, GENERIC_TAXONOMY)
        stack.extend(reversed(node.children))
    return GENERIC_TAXONOMY


def _doc_type_label(doc_type: str | None) -> str:
    if doc_type is not None:
        doc_types = builtin_taxonomy(DOC_TYPES)
        if doc_type in doc_types:
            return doc_types.node(doc_type).label
    return "document"
