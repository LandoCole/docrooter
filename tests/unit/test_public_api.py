import os
import subprocess
import sys
from pathlib import Path

import pytest

import docrooter

PUBLIC_NAMES = [
    "load",
    "Document",
    "Section",
    "Taxonomy",
    "TaxonomyNode",
    "SectionView",
    "Classifier",
    "Greedy",
    "Beam",
    "Flat",
    "Dataset",
    "train",
    "TrainedModel",
    "evaluate",
    "compare",
    "calibrate",
    "builtin_taxonomy",
    "DocrooterError",
    "TaxonomyError",
    "BackendError",
    "DatasetError",
    "MissingExtraError",
    "UnknownTaxonomyError",
    # v0.2
    "load_pdf",
    "classify_document",
    "normalize_sections",
    "list_builtin_taxonomies",
    "DocumentView",
    "DocTypeResult",
    "NormalizationResult",
    # v0.3
    "configure",
    # v0.4
    "Element",
    "ElementKind",
    "Box",
    "TableData",
    "FigureData",
    "process",
    "process_many",
    "RouterConfig",
    "Analysis",
    "register_ocr_engine",
    "register_layout_engine",
    "Chunk",
    "chunk",
    "EmbeddingSet",
    "embed",
    "summarize",
    "Summary",
    # v0.5
    "TagResult",
    # v0.6
    "Field",
    "builtin_fields",
    "load_fields",
    "select",
    "Selection",
    "Selector",
    # v1
    "classify",
    "Classification",
    "SectionTag",
    "LEVELS",
    # section context
    "TaggedDocument",
    "SectionContext",
    "SectionMatch",
]


def test_public_api_surface():
    for name in PUBLIC_NAMES:
        assert hasattr(docrooter, name), name
        assert name in docrooter.__all__, name


def test_all_lists_only_existing_names():
    for name in docrooter.__all__:
        assert hasattr(docrooter, name), name


@pytest.mark.parametrize(
    "module", ["docrooter", "docrooter.pdf", "docrooter.cli", "docrooter.doc_view"]
)
def test_core_import_does_not_import_torch(module):
    root = Path(__file__).resolve().parents[2]
    out = subprocess.run(
        [sys.executable, "-c", f"import {module}, sys; print('torch' in sys.modules)"],
        capture_output=True,
        text=True,
        cwd=root,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    assert out.stdout.strip() == "False", out.stderr


def test_load_pdf_is_exported():
    import docrooter.pdf
    from docrooter.pdf.load import load_pdf

    assert docrooter.load_pdf is load_pdf and docrooter.pdf.load_pdf is load_pdf
    assert "load_pdf" in docrooter.pdf.__all__


def test_classify_is_the_v1_function_and_the_subpackage_still_imports():
    from docrooter.classify.classifier import Classifier
    from docrooter.flow import classify
    from docrooter.ladder import LEVELS

    assert docrooter.classify is classify and docrooter.LEVELS is LEVELS
    assert docrooter.Classifier is Classifier


def test_errors_are_exported_from_their_module():
    from docrooter import errors

    for name in ("DocrooterError", "TaxonomyError", "BackendError", "DatasetError"):
        assert getattr(docrooter, name) is getattr(errors, name)
    assert issubclass(docrooter.MissingExtraError, docrooter.DocrooterError)
    assert issubclass(docrooter.MissingExtraError, ImportError)
