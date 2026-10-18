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


def test_errors_are_exported_from_their_module():
    from docrooter import errors

    for name in ("DocrooterError", "TaxonomyError", "BackendError", "DatasetError"):
        assert getattr(docrooter, name) is getattr(errors, name)
    assert issubclass(docrooter.MissingExtraError, docrooter.DocrooterError)
    assert issubclass(docrooter.MissingExtraError, ImportError)
