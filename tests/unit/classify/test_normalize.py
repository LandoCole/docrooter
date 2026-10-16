import pytest
from conftest import RecordingBackend

import docrooter
from docrooter import Classifier, Document, Flat, SectionView, Taxonomy, TaxonomyNode
from docrooter.backends import KeywordBackend
from docrooter.classify import normalize as normalize_module
from docrooter.classify.normalize import NormalizationResult, normalize_sections
from docrooter.taxonomies import DOC_TYPE_TO_SECTION_TAXONOMY, builtin_taxonomy


def paper() -> Document:
    return Document.from_sections(
        [
            {"title": "Abstract", "level": 1, "text": "We present results."},
            {"title": "Method", "level": 1, "text": "Our method is simple."},
            {"title": "Datasets", "level": 2, "text": "Experiments use datasets and baselines."},
            {"title": "References", "level": 1, "text": "Bibliography of cited works."},
            {"title": "Proofs", "level": 3, "text": "Extra proofs."},
        ],
        source="paper.md",
    )


def by_title(result: NormalizationResult) -> dict:
    return {n.title: n for n in result.sections.values()}


def test_references_maps_to_references_in_research_paper():
    result = normalize_sections(
        paper(), doc_type="research.research_paper", backend=KeywordBackend()
    )
    assert result.taxonomy == "section_types/research_paper"
    assert result.doc_type == "research.research_paper"
    assert result.doc_type_confidence is None
    norm = by_title(result)["References"]
    assert norm.canonical == "references"
    assert norm.label == "References"
    assert 0 < norm.score <= 1
    assert by_title(result)["Abstract"].canonical == "abstract"


@pytest.mark.parametrize(
    ("doc_type", "expected"),
    [
        ("legal.contract", "section_types/contract"),
        ("technical.datasheet", "section_types/technical_document"),
        ("financial.annual_report", "section_types/annual_report"),
        ("research", "section_types/research_paper"),  # group id -> its first leaf
        ("financial", "section_types/annual_report"),
        ("not.a.type", "section_types/generic"),
    ],
)
def test_taxonomy_chosen_from_doc_type(doc_type, expected):
    result = normalize_sections(paper(), doc_type=doc_type, backend=KeywordBackend())
    assert result.taxonomy == expected
    assert result.doc_type == doc_type


def test_every_leaf_doc_type_has_a_taxonomy():
    for leaf in builtin_taxonomy("doc_types").leaves():
        result = normalize_sections(paper(), doc_type=leaf.id, backend=KeywordBackend())
        assert result.taxonomy == DOC_TYPE_TO_SECTION_TAXONOMY[leaf.id]


def test_only_requested_levels_are_classified():
    default = normalize_sections(
        paper(), doc_type="research.research_paper", backend=KeywordBackend()
    )
    assert set(by_title(default)) == {"Abstract", "Method", "Datasets", "References"}
    only_two = normalize_sections(
        paper(), doc_type="research.research_paper", backend=KeywordBackend(), levels=(2,)
    )
    assert set(by_title(only_two)) == {"Datasets"}
    with_three = normalize_sections(
        paper(), doc_type="research.research_paper", backend=KeywordBackend(), levels=(0, 3)
    )
    assert set(by_title(with_three)) == {"Proofs"}  # the preamble is never normalised


def test_no_matching_levels_gives_empty_result():
    result = normalize_sections(
        paper(), doc_type="research.research_paper", backend=KeywordBackend(), levels=(7,)
    )
    assert result.sections == {}


def test_doc_type_none_classifies_the_document_first(monkeypatch):
    calls = []
    real = normalize_module.classify_document

    def spy(doc, taxonomy, **kwargs):
        calls.append((doc, taxonomy, kwargs))
        return real(doc, taxonomy, **kwargs)

    monkeypatch.setattr(normalize_module, "classify_document", spy)
    doc = paper()
    result = normalize_sections(doc, backend=KeywordBackend())
    assert len(calls) == 1
    assert calls[0][0] is doc
    assert calls[0][1].content_hash() == builtin_taxonomy("doc_types").content_hash()
    assert result.doc_type is not None
    assert result.doc_type_confidence is not None and result.doc_type_confidence > 0
    assert result.taxonomy == DOC_TYPE_TO_SECTION_TAXONOMY[result.doc_type]

    calls.clear()
    normalize_sections(doc, doc_type="legal.contract", backend=KeywordBackend())
    normalize_sections(doc, taxonomy="research_paper", backend=KeywordBackend())
    assert calls == []


def test_unknown_builtin_name_lists_valid_names():
    with pytest.raises(KeyError) as err:
        normalize_sections(paper(), taxonomy="section_types/nope", backend=KeywordBackend())
    assert "section_types/research_paper" in str(err.value)
    with pytest.raises(KeyError, match="available"):
        normalize_sections(paper(), taxonomy="nope", backend=KeywordBackend())


def test_builtin_name_with_or_without_prefix():
    for name in ("research_paper", "section_types/research_paper"):
        result = normalize_sections(paper(), taxonomy=name, backend=KeywordBackend())
        assert result.taxonomy == "section_types/research_paper"
        assert result.doc_type is None
        assert by_title(result)["References"].canonical == "references"


def test_custom_taxonomy():
    custom = Taxonomy(
        [
            TaxonomyNode(id="a", label="Alpha", description="abstract summary results"),
            TaxonomyNode(id="b", label="Bravo", description="bibliography cited works references"),
        ]
    )
    result = normalize_sections(paper(), taxonomy=custom, backend=KeywordBackend())
    assert result.taxonomy == "<custom>"
    assert by_title(result)["References"].canonical == "b"
    assert by_title(result)["References"].label == "Bravo"


def test_one_batched_call_with_instruction_and_view():
    backend = RecordingBackend(KeywordBackend())
    normalize_sections(paper(), doc_type="research.research_paper", backend=backend)
    assert backend.calls == 1
    assert backend.last_instruction == "Which standard section of a Research paper is this?"
    assert len(backend.last_texts) == 4
    # default view: the heading path and the title, no body text
    assert backend.last_texts[0] == "Path: Abstract\nTitle: Abstract"
    assert backend.last_texts[2] == "Path: Method > Datasets\nTitle: Datasets"

    backend = RecordingBackend(KeywordBackend())
    normalize_sections(paper(), taxonomy="generic", backend=backend)
    assert backend.last_instruction == "Which standard section of a document is this?"


def test_view_override_renders_sections_with_that_view():
    backend = RecordingBackend(KeywordBackend())
    view = SectionView(fields=("path", "title"), max_tokens=32)
    normalize_sections(paper(), doc_type="research.research_paper", backend=backend, view=view)
    assert backend.calls == 1
    assert backend.last_texts[2] == "Path: Method > Datasets\nTitle: Datasets"


def _custom(examples: bool) -> Taxonomy:
    return Taxonomy(
        [
            TaxonomyNode(
                id="a",
                label="Alpha",
                description="abstract summary",
                examples=["Abstract"] if examples else [],
            ),
            TaxonomyNode(id="b", label="Bravo", description="bibliography cited works"),
        ]
    )


def test_default_view_is_title_and_lead_for_a_taxonomy_without_examples():
    backend = RecordingBackend(KeywordBackend())
    normalize_sections(paper(), taxonomy=_custom(examples=False), backend=backend)
    assert backend.last_texts[0] == "Title: Abstract\nLead: We present results."
    assert backend.last_texts[2] == "Title: Datasets\nLead: Experiments use datasets and baselines."


def test_default_view_is_path_and_title_once_any_node_has_examples():
    backend = RecordingBackend(KeywordBackend())
    normalize_sections(paper(), taxonomy=_custom(examples=True), backend=backend)
    assert backend.last_texts[2] == "Path: Method > Datasets\nTitle: Datasets"


def test_explicit_view_wins_over_the_examples_rule():
    backend = RecordingBackend(KeywordBackend())
    view = SectionView(fields=("title",), max_tokens=32)
    normalize_sections(paper(), taxonomy=_custom(examples=False), backend=backend, view=view)
    assert backend.last_texts[0] == "Title: Abstract"


def test_blank_title_with_text_gets_its_lead_under_a_heading_only_view():
    doc = Document.from_sections(
        [
            {"title": "Method", "level": 1, "text": "Our method."},
            {"title": " ", "level": 1, "text": "We list the cited works. More."},
        ],
        source="x.md",
    )
    backend = RecordingBackend(KeywordBackend())
    normalize_sections(doc, taxonomy="research_paper", backend=backend)
    assert backend.last_texts[0] == "Path: Method\nTitle: Method"
    assert backend.last_texts[1].endswith("Lead: We list the cited works. More.")


def test_prebuilt_classifier():
    taxonomy = builtin_taxonomy("section_types/research_paper")
    backend = RecordingBackend(KeywordBackend())
    classifier = Classifier(
        taxonomy,
        backend,
        strategy=Flat(),
        view=SectionView(fields=("title", "lead")),
        threshold=0.0,
    )
    result = normalize_sections(paper(), doc_type="research.research_paper", classifier=classifier)
    assert backend.calls == 1
    assert by_title(result)["References"].canonical == "references"


def test_prebuilt_classifier_taxonomy_mismatch():
    classifier = Classifier(
        builtin_taxonomy("section_types/contract"), KeywordBackend(), strategy=Flat()
    )
    with pytest.raises(ValueError, match="does not match"):
        normalize_sections(paper(), doc_type="research.research_paper", classifier=classifier)


def test_blank_title_gets_no_label():
    doc = Document.from_sections([{"title": "", "level": 1, "text": ""}], source="x.md")
    result = normalize_sections(doc, taxonomy="generic", backend=KeywordBackend())
    (norm,) = result.sections.values()
    assert norm.canonical is None and norm.label is None and norm.score == 0.0


def test_exported_from_package():
    assert docrooter.normalize_sections is normalize_sections
    assert docrooter.NormalizationResult is NormalizationResult
    assert "normalize_sections" in docrooter.__all__
    assert "NormalizationResult" in docrooter.__all__


def test_doc_types_loaded_once_and_backend_spec_resolved_once(monkeypatch):
    from docrooter.backends import registry

    loads, resolved = [], []
    real_builtin, real_resolve = normalize_module.builtin_taxonomy, registry.resolve_backend

    def count_builtin(name):
        loads.append(name)
        return real_builtin(name)

    def count_resolve(spec):
        resolved.append(spec)
        return real_resolve(spec)

    monkeypatch.setattr(normalize_module, "builtin_taxonomy", count_builtin)
    monkeypatch.setattr(registry, "resolve_backend", count_resolve)
    result = normalize_sections(paper(), backend="keyword")
    assert loads.count("doc_types") == 1
    assert resolved == ["keyword"]
    assert result.doc_type is not None

    loads.clear()
    normalize_sections(paper(), doc_type="research", backend="keyword")  # a group id
    assert loads.count("doc_types") == 1
